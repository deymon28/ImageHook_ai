"""Live recorder: DXGI capture of one monitor -> filters -> JPEG frames of the session.

The loop polls hotkeys at ~100 Hz and evaluates a frame every `tick` seconds (default 0.5 = the minimum interval).
Everything heavy is optional and loaded before the loop starts. The process runs at below-normal CPU priority.
"""
from __future__ import annotations

import ctypes
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from . import __version__, report
from .filters import CHANGE_IGNORE, ICON_ZONE, Box, CapturePolicy, Features, HudLayout, PolicyConfig, load_template
from .clip import ClipRecorder, find_ffmpeg
from .hotkeys import Keys, beep, parse_key
from .session import Session

GAME_PROCESSES: tuple[str, ...] = ()  # set game_processes in imagehook.toml; empty = any foreground window


@dataclass
class RecordConfig:
    out: Path
    output_idx: int | None = None
    height: int = 0  # 0 = native resolution
    ext: str = "jpg"
    quality: int = 95
    tick: float = 0.5
    toggle_key: str = "f8"
    quit_key: str = "f9"
    burst_key: str = "mouse5"
    clip_key: str = "f7"
    burst_back: float = 10.0  # a burst also keeps the unsaved frames of the last N seconds
    burst_forward: float = 5.0  # ...and every non-duplicate frame for the next N seconds
    any_window: bool = False  # record even when the game is not the foreground window
    game_processes: tuple[str, ...] = GAME_PROCESSES  # executable names of the game window; empty = any window
    icon_zone: Box = ICON_ZONE  # HUD layout, see filters.HudLayout
    change_ignore: tuple[Box, ...] = CHANGE_IGNORE
    ui_filter: bool = True
    template: Path | None = None  # HUD icon template; without one the interface check is off
    max_session_gb: float = 20.0
    min_free_gb: float = 20.0
    target_ram_gb: float = 2.0  # going over a target only prints a warning
    max_ram_gb: float = 4.0  # hard limit: detector off, then recording stops
    target_vram_gb: float = 2.0
    max_vram_gb: float = 4.0  # hard limit: detector off
    target_gpu_pct: float = 20.0  # own GPU (SM) load the detector is throttled to
    max_gpu_pct: float = 30.0  # sustained own load above this turns the detector off
    beep: bool = True
    autostart: bool = False
    duration: float | None = None
    detector: Path | None = None
    det_imgsz: int = 0  # 0 = native (the frame's long side)
    det_max_height: float = 0.15
    ffmpeg: str | None = None
    clip_fps: int = 30
    clip_seconds: float = 120.0
    clip_cq: int = 19
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    config: str | None = None  # name of the config file the settings came from (only reported)


class Foreground:
    """Name of the process that owns the foreground window (cached per pid)."""

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32
        self._names: dict[int, str] = {}

    def name(self) -> str:
        hwnd = self._user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        p = pid.value
        if p not in self._names:
            try:
                self._names[p] = psutil.Process(p).name().lower()
            except psutil.Error:
                return ""
        return self._names[p]


class GpuGovernor:
    """Keeps the detector's share of the GPU near the target by running it only on every `stride`-th evaluated
    frame. update() gets this tool's own GPU load every few seconds; it returns "off" when the load stayed above
    the hard limit `hard_checks` times in a row although the detector already runs as rarely as allowed."""

    def __init__(self, target: float = 20.0, hard: float = 30.0, max_stride: int = 8, hard_checks: int = 3):
        self.target, self.hard, self.max_stride, self.hard_checks = target, hard, max_stride, hard_checks
        self.stride = 1
        self._over = 0

    def update(self, load: float) -> str | None:
        if load > self.target and self.stride < self.max_stride:
            self.stride *= 2
        elif load < self.target / 2 and self.stride > 1:
            self.stride //= 2
        self._over = self._over + 1 if load > self.hard else 0
        if self._over >= self.hard_checks and self.stride >= self.max_stride:
            return "off"
        return None


class Resources:
    """RSS / CPU / VRAM / GPU load of this process and its clip encoder child, sampled every few seconds.
    GPU load comes from NVML per-process samples (the game shares the device, so device-wide load says little)."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.proc.cpu_percent()
        self.ncpu = psutil.cpu_count() or 1
        self.rss_peak = self.cpu_peak = self.vram_peak = self.gpu_peak = self.enc_peak = 0.0
        self.cpu_sum = self.gpu_sum = 0.0
        self.samples = 0
        self.rss = self.vram = self.cpu = self.gpu = self.enc = 0.0
        self.gpu_supported = False
        self._child: psutil.Process | None = None
        self._util_ts = 0
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_supported = True
        except Exception:  # no NVIDIA driver / nvidia-ml-py: VRAM is then taken from torch if a detector runs
            self._nvml = None

    def set_child(self, pid: int | None) -> None:
        try:
            self._child = psutil.Process(pid) if pid else None
            if self._child:
                self._child.cpu_percent()
        except psutil.Error:
            self._child = None

    def pids(self) -> set[int]:
        return {self.proc.pid} | ({self._child.pid} if self._child else set())

    def vram_gb(self, torch_mod=None) -> float:
        if self._nvml is not None:
            try:
                used = [p.usedGpuMemory for p in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._gpu)
                        if p.pid in self.pids() and p.usedGpuMemory]
                if used:
                    return sum(used) / 2**30
            except Exception:
                pass
        if torch_mod is not None:  # WDDM hides per-process usage: allocator reserve + ~0.5 GB CUDA context
            return torch_mod.cuda.memory_reserved() / 2**30 + 0.5
        return 0.0

    def gpu_load(self) -> tuple[float, float]:
        """(SM %, encoder %) of our processes since the previous call: mean of NVML samples, summed over pids."""
        if not self.gpu_supported:
            return 0.0, 0.0
        nv = self._nvml
        try:
            samples = nv.nvmlDeviceGetProcessUtilization(self._gpu, self._util_ts)
        except nv.NVMLError_NotFound:
            return 0.0, 0.0  # no samples since the last call: idle for us
        except nv.NVMLError:
            self.gpu_supported = False
            return 0.0, 0.0
        if samples:
            self._util_ts = max(s.timeStamp for s in samples)
        sm = enc = 0.0
        for pid in self.pids():
            mine = [s for s in samples if s.pid == pid]
            if mine:
                sm += sum(s.smUtil for s in mine) / len(mine)
                enc += sum(s.encUtil for s in mine) / len(mine)
        return sm, enc

    def sample(self, torch_mod=None) -> None:
        rss = self.proc.memory_info().rss
        cpu = self.proc.cpu_percent()
        if self._child:
            try:
                rss += self._child.memory_info().rss
                cpu += self._child.cpu_percent()
            except psutil.Error:
                self._child = None
        self.rss, self.cpu = rss / 2**30, cpu / self.ncpu  # CPU as a share of the whole CPU
        self.vram = self.vram_gb(torch_mod)
        self.gpu, self.enc = self.gpu_load()
        self.rss_peak, self.cpu_peak = max(self.rss_peak, self.rss), max(self.cpu_peak, self.cpu)
        self.vram_peak = max(self.vram_peak, self.vram)
        self.gpu_peak, self.enc_peak = max(self.gpu_peak, self.gpu), max(self.enc_peak, self.enc)
        self.cpu_sum += self.cpu
        self.gpu_sum += self.gpu
        self.samples += 1

    def report(self) -> dict:
        n = max(self.samples, 1)
        return {"rss_peak_gb": round(self.rss_peak, 3), "cpu_avg_pct": round(self.cpu_sum / n, 2),
                "cpu_peak_pct": round(self.cpu_peak, 2), "vram_peak_gb": round(self.vram_peak, 3),
                "gpu_sm_avg_pct": round(self.gpu_sum / n, 1), "gpu_sm_peak_pct": round(self.gpu_peak, 1),
                "gpu_encoder_peak_pct": round(self.enc_peak, 1), "gpu_per_process": self.gpu_supported}


def primary_output_index(bettercam) -> int:
    import re

    for line in bettercam.output_info().splitlines():
        m = re.search(r"Output\[(\d+)\].*Primary:True", line)
        if m:
            return int(m.group(1))
    return 0


@dataclass
class Pending:
    """An evaluated but unsaved frame kept for a possible burst."""
    t: float
    wall: datetime
    frame: np.ndarray
    feats: Features
    novelty: float
    ui: float


class Recorder:
    def __init__(self, cfg: RecordConfig):
        self.cfg = cfg
        self.keys = Keys()
        self.vk_toggle, self.vk_quit, self.vk_burst, self.vk_clip = (
            parse_key(k) for k in (cfg.toggle_key, cfg.quit_key, cfg.burst_key, cfg.clip_key))
        template = load_template(cfg.template) if cfg.template and cfg.ui_filter else None
        self.layout = HudLayout(tuple(cfg.icon_zone), tuple(tuple(b) for b in cfg.change_ignore))
        self.games = tuple(g.lower() for g in cfg.game_processes)
        self.policy = CapturePolicy(cfg.policy, template, self.layout)
        self.fg = Foreground()
        self.res = Resources()
        self.session: Session | None = None
        self.ring: deque[Pending] = deque()
        self.burst_until = -1e18
        self.collecting = False
        self.blocked = ""  # reason recording may not start (disk full)
        self.rec_seconds = 0.0
        self.evaluations = 0
        self.detector = None
        self.det_runs, self.det_time = 0, 0.0
        self.torch = None
        self.governor = GpuGovernor(cfg.target_gpu_pct, cfg.max_gpu_pct)
        self.clip: ClipRecorder | None = None
        self.over_target: set[str] = set()
        self._status_saved: tuple[float, int] | None = None  # (rec_seconds, saved) at the previous status line

    # ------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    def settings(self) -> dict:
        d = asdict(self.cfg)
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
        d.pop("out", None)
        if d.get("detector"):
            d["detector"] = Path(d["detector"]).name  # no local paths in the session summary
        if d.get("template"):
            d["template"] = Path(d["template"]).name
        if d.get("ffmpeg"):
            d["ffmpeg"] = Path(d["ffmpeg"]).name
        return d

    def extra(self) -> dict:
        e = {"tool": f"imagehook {__version__}", "ended": datetime.now().isoformat(timespec="seconds"),
             "recorded_seconds": round(self.rec_seconds, 1), "evaluations": self.evaluations,
             "settings": self.settings(), "resources": self.res.report(), "detector_runs": self.det_runs}
        if self.det_runs:
            e["detector_ms_avg"] = round(1000 * self.det_time / self.det_runs, 1)
            e["detector_stride_end"] = self.governor.stride
        if self.clip and self.clip.clips:
            e["clips"] = self.clip.clips
        return e

    def start(self) -> None:
        if self.blocked:
            self.log(f"cannot start: {self.blocked}")
            beep(400, 120, self.cfg.beep, times=3)
            return
        if self.session is None:
            self.cfg.out.mkdir(parents=True, exist_ok=True)
            self.session = Session(self.cfg.out, datetime.now(), self.cfg.ext, self.cfg.quality)
            self.policy.reset()
            self.log(f"new session {self.session.dir}")
        if self.session.free_gb() < self.cfg.min_free_gb:
            self.blocked = f"less than {self.cfg.min_free_gb} GB free on the output disk"
            self.close_session()
            self.start()
            return
        self.collecting = True
        beep(1200, 120, self.cfg.beep)
        self.log(f"STARTED  saved so far: {self.session.n_saved}")

    def stop(self, why: str = "") -> None:
        self.collecting = False
        self.ring.clear()
        self.burst_until = -1e18
        if why:
            beep(400, 150, self.cfg.beep, times=3)
        else:
            beep(600, 120, self.cfg.beep)
        self.log(f"STOPPED{' (' + why + ')' if why else ''}  {self.status_line()}")
        if self.session:
            self.session.write_summary(self.extra())

    def close_session(self) -> None:
        if self.session:
            self.session.close(self.extra())
            self.log(f"session closed: {self.session.dir}")
            try:
                for line in report.format_session(report.session_stats(self.session.dir)):
                    print(line, flush=True)
            except (OSError, ValueError, KeyError) as e:  # the summary is informative only
                self.log(f"could not summarize the session: {e}")
            self.session = None
            self._status_saved = None

    def status_line(self) -> str:
        s = self.session
        if s is None:
            return ""
        per_min = None
        if self._status_saved is not None and self.rec_seconds - self._status_saved[0] > 1:
            per_min = (s.n_saved - self._status_saved[1]) * 60 / (self.rec_seconds - self._status_saved[0])
        self._status_saved = (self.rec_seconds, s.n_saved)
        extra = ""
        if self.detector is not None:
            extra = f", detector every {self.governor.stride}. frame"
        return report.status_line(self.rec_seconds, s.saved, s.skipped, self.evaluations, s.bytes / 2**30, per_min,
                                  self.cfg.tick, self.res.rss, self.res.vram, self.res.gpu, extra)

    def disable_detector(self, why: str) -> None:
        if self.detector is None:
            return
        self.detector = None
        if self.torch is not None:
            import gc

            gc.collect()
            self.torch.cuda.empty_cache()
        self.log(f"detector turned OFF: {why}")
        beep(500, 200, self.cfg.beep, times=2)

    def settings_lines(self, idx: int, width: int, height: int, ffmpeg: str | None) -> list[str]:
        """The effective settings, printed once at startup."""
        cfg, p = self.cfg, self.cfg.policy
        cfg.out.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cfg.out).free / 2**30
        saved_as = f"{cfg.height}p" if cfg.height else "native resolution"
        quality = f" q{cfg.quality}" if cfg.ext == "jpg" else ""
        if cfg.any_window or not self.games:
            game = "any foreground window" + (" (--any-window)" if cfg.any_window else " (set game_processes to limit it)")
        else:
            game = f"only while {', '.join(self.games)} is the foreground window"
        if self.policy.template is not None:
            ui = f"interface screens skipped below HUD icon similarity {p.ui_threshold:g} ({cfg.template.name})"
        elif cfg.ui_filter:
            ui = "interface check off: no HUD template (set template; build one with: imagehook template)"
        else:
            ui = "interface check off (--no-ui-filter)"
        if self.detector is not None:
            det = (f"{cfg.detector.name}, imgsz {cfg.det_imgsz or 'native'}: a person at conf >= {p.det_conf:g} saves "
                   f"every {p.det_interval:g} s, nobody seen -> base interval {p.idle_interval:g} s")
        else:
            det = "off (--detector weights.pt saves more often while people are seen)"
        if self.clip is not None:
            clips = (f"{cfg.clip_key.upper()}: {cfg.clip_fps} FPS H.264, cq {cfg.clip_cq}, stops after "
                     f"{cfg.clip_seconds:g} s ({Path(ffmpeg).name})")
        else:
            clips = "unavailable: ffmpeg with ddagrab and h264_nvenc not found (PATH, IMAGEHOOK_FFMPEG or --ffmpeg)"
        return [
            f"ImageHook {__version__}, settings: {cfg.config or 'built-in defaults'} + command line "
            f"(see them all: imagehook config)",
            f"  output     {cfg.out}  ({free:.0f} GB free)",
            f"  monitor    output {idx}, {width}x{height}; frames saved at {saved_as}, {cfg.ext.upper()}{quality}",
            f"  game       {game}",
            f"  filters    {ui}; duplicates below novelty {p.dup_novelty:g}",
            f"  intervals  a check every {cfg.tick:g} s; saved after {p.min_interval:g} s for a big change, "
            f"{p.base_interval:g} s for an average one, {p.max_interval:g} s for a still view",
            f"  detector   {det}",
            f"  clips      {clips}",
            f"  limits     RAM {cfg.target_ram_gb:g}/{cfg.max_ram_gb:g} GB, VRAM {cfg.target_vram_gb:g}/"
            f"{cfg.max_vram_gb:g} GB, own GPU {cfg.target_gpu_pct:g}/{cfg.max_gpu_pct:g}% (target/hard); session "
            f"{cfg.max_session_gb:g} GB; stop below {cfg.min_free_gb:g} GB free",
            f"  keys       {cfg.toggle_key.upper()} start/stop | {cfg.burst_key.upper()} burst (last "
            f"{cfg.burst_back:g} s + next {cfg.burst_forward:g} s) | {cfg.clip_key.upper()} clip | "
            f"{cfg.quit_key.upper()} quit",
        ]

    # ------------------------------------------------------------ per-frame work
    def save(self, frame: np.ndarray, wall: datetime, reason: str, novelty: float, ui: float, det: float | None) -> bool:
        h = self.cfg.height
        if h and frame.shape[0] != h:
            # bilinear without antialiasing, like a GPU resize in a detector pipeline
            frame = cv2.resize(frame, (round(frame.shape[1] * h / frame.shape[0]), h), interpolation=cv2.INTER_LINEAR)
        if self.session.save(frame, wall, reason, novelty, ui, det) is None:
            self.log("disk writer is behind, frame dropped")
            return False
        return True

    def flush_ring(self) -> None:
        for p in self.ring:
            if self.save(p.frame, p.wall, "burst", p.novelty, p.ui, None):
                self.session.skipped["waiting"] -= 1
                if p.t > self.policy.last_t:
                    self.policy.saved(p.t, p.feats)
        self.ring.clear()

    def tick(self, cam) -> None:
        s = self.session
        now = time.monotonic()
        if not self.cfg.any_window and self.games and self.fg.name() not in self.games:
            s.skipped["not_game"] += 1
            return
        frame = cam.grab()
        wall = datetime.now()
        if frame is None:  # nothing on screen changed since the previous grab
            s.skipped["no_new_frame"] += 1
            return
        self.evaluations += 1
        f = self.policy.features(frame)
        ui = self.policy.ui(f)
        det = None
        if self.detector is not None and ui >= self.cfg.policy.ui_threshold:
            det = 0.0  # throttled frames count as "nothing new seen"; an earlier hit is still held
            if self.evaluations % self.governor.stride == 0:
                t0 = time.perf_counter()
                det = self.detector(frame)
                self.det_time += time.perf_counter() - t0
                self.det_runs += 1
        d = self.policy.evaluate(now, f, det, burst=now <= self.burst_until, ui=ui)
        if d.save:
            if self.save(frame, wall, d.reason, d.novelty, d.ui, det):
                self.policy.saved(now, f)
            return
        s.skipped[d.reason] += 1
        if d.reason == "waiting" and self.cfg.burst_back > 0:
            self.ring.append(Pending(now, wall, frame, f, d.novelty, d.ui))
        while self.ring and self.ring[0].t < now - self.cfg.burst_back:
            self.ring.popleft()

    def warn_target(self, name: str, value: float, target: float, unit: str) -> None:
        """Going over a target is allowed; say so once per crossing."""
        if value > target and name not in self.over_target:
            self.over_target.add(name)
            self.log(f"note: {name} {value:.2f}{unit} is above the {target:g}{unit} target")
        elif value <= target * 0.9:
            self.over_target.discard(name)

    def check_limits(self) -> None:
        s = self.session
        if self.clip is not None:
            if (info := self.clip.poll()) is not None:
                self.clip_done(info)
        self.res.sample(self.torch if self.detector is not None else None)
        self.warn_target("RAM", self.res.rss, self.cfg.target_ram_gb, " GB")
        self.warn_target("VRAM", self.res.vram, self.cfg.target_vram_gb, " GB")
        self.warn_target("own GPU load", self.res.gpu, self.cfg.target_gpu_pct, "%")
        if self.detector is not None:
            stride = self.governor.stride
            if self.governor.update(self.res.gpu) == "off":
                self.disable_detector(f"own GPU load {self.res.gpu:.0f}% > {self.cfg.max_gpu_pct:g}% even when throttled")
            elif self.governor.stride != stride:
                self.log(f"detector now runs on every {self.governor.stride}. frame (own GPU load {self.res.gpu:.0f}%)")
        if self.res.vram > self.cfg.max_vram_gb:
            self.disable_detector(f"VRAM {self.res.vram:.2f} GB > {self.cfg.max_vram_gb} GB")
        if self.res.rss > self.cfg.max_ram_gb:
            if self.detector is not None:
                self.disable_detector(f"RAM {self.res.rss:.2f} GB > {self.cfg.max_ram_gb} GB")
            elif self.collecting:
                self.stop(f"RAM {self.res.rss:.2f} GB > {self.cfg.max_ram_gb} GB")
        if s is None or not self.collecting:
            return
        if s.bytes >= self.cfg.max_session_gb * 2**30:
            self.stop(f"session reached {self.cfg.max_session_gb} GB; the next start opens a new session")
            self.close_session()
        elif s.free_gb() < self.cfg.min_free_gb:
            self.blocked = f"less than {self.cfg.min_free_gb} GB free on the output disk"
            self.stop(self.blocked)

    # ------------------------------------------------------------ clips
    def toggle_clip(self) -> None:
        if self.clip is None:
            self.log("clips need ffmpeg with ddagrab and h264_nvenc on PATH (or --ffmpeg)")
            beep(400, 120, self.cfg.beep, times=3)
            return
        if self.clip.active:
            self.clip_done(self.clip.stop())
            return
        self.cfg.out.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.cfg.out).free / 2**30 < self.cfg.min_free_gb:
            self.log(f"cannot start a clip: less than {self.cfg.min_free_gb} GB free")
            beep(400, 120, self.cfg.beep, times=3)
            return
        path = self.clip.start()
        self.res.set_child(self.clip.pid)
        beep(1000, 60, self.cfg.beep, times=2)
        self.log(f"CLIP started: {path.name} (stops by itself after {self.cfg.clip_seconds:g} s)")

    def clip_done(self, info: dict | None) -> None:
        if info is None:
            return
        self.res.set_child(None)
        beep(700, 60, self.cfg.beep, times=2)
        state = "saved" if info["ok"] else f"FAILED ({info.get('error', 'ffmpeg error')})"
        self.log(f"CLIP {state}: {info['file']} {info['seconds']} s, {info['bytes'] / 2**20:.0f} MB")

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        cfg = self.cfg
        try:
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
        except psutil.Error:
            pass
        cv2.setNumThreads(1)
        if cfg.detector:
            from .detector import Detector, DetectorConfig

            self.log(f"loading detector {cfg.detector.name} (imgsz {cfg.det_imgsz})...")
            self.detector = Detector(DetectorConfig(cfg.detector, cfg.det_imgsz, max_height=cfg.det_max_height,
                                                    max_vram_gb=min(1.5, cfg.target_vram_gb),
                                                    icon_zone=self.layout.icon_zone))
            self.torch = self.detector.torch
            self.res.sample(self.torch)
            self.log(f"detector ready: RAM {self.res.rss:.2f} GB, VRAM {self.res.vram:.2f} GB")
        import bettercam

        idx = primary_output_index(bettercam) if cfg.output_idx is None else cfg.output_idx
        cam = bettercam.create(output_idx=idx, output_color="BGR")
        if ffmpeg := find_ffmpeg(cfg.ffmpeg):
            self.clip = ClipRecorder(ffmpeg, cfg.out, idx, cfg.clip_fps, cfg.clip_seconds, cfg.clip_cq)
        for line in self.settings_lines(idx, cam.width, cam.height, ffmpeg):
            print(line, flush=True)
        self.log(f"ready: press {cfg.toggle_key.upper()} to start recording")
        if cfg.autostart:
            self.start()
        t_end = time.monotonic() + cfg.duration if cfg.duration else None
        next_tick = next_check = next_status = 0.0
        last = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                if self.collecting:
                    self.rec_seconds += now - last
                last = now
                if self.keys.pressed(self.vk_quit) or (t_end and now > t_end):
                    break
                if self.keys.pressed(self.vk_toggle):
                    self.stop() if self.collecting else self.start()
                if self.keys.pressed(self.vk_clip):
                    self.toggle_clip()
                if self.keys.pressed(self.vk_burst) and self.collecting:
                    self.burst_until = now + cfg.burst_forward
                    n = len(self.ring)
                    self.flush_ring()
                    beep(1600, 60, cfg.beep)
                    self.log(f"BURST: kept {n} recent frames, keeping every new frame for {cfg.burst_forward:g} s")
                if self.collecting and now >= next_tick:
                    next_tick = now + cfg.tick
                    self.tick(cam)
                if now >= next_check:
                    next_check = now + 2.0
                    self.check_limits()
                if self.collecting and now >= next_status:
                    if next_status:
                        self.log(self.status_line())
                        self.session.write_summary(self.extra())
                    next_status = now + 60.0
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.collecting = False
            if self.clip is not None and self.clip.active:
                self.clip_done(self.clip.stop())
            self.close_session()
            cam.release()
            self.log("done")
