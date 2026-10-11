"""Live recorder: DXGI capture of one monitor -> filters -> JPEG frames of the session.

The loop polls hotkeys at ~100 Hz and evaluates a frame every `tick` seconds (default 0.5 = the minimum interval).
Everything heavy is optional and loaded before the loop starts. The process runs at below-normal CPU priority.
"""
from __future__ import annotations

import ctypes
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from . import __version__
from .filters import CapturePolicy, Features, PolicyConfig, features, load_template
from .hotkeys import Keys, beep, parse_key
from .session import Session

GAME_PROCESSES = ("uagame.exe",)


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
    burst_back: float = 10.0  # a burst also keeps the unsaved frames of the last N seconds
    burst_forward: float = 5.0  # ...and every non-duplicate frame for the next N seconds
    any_window: bool = False  # record even when the game is not the foreground window
    ui_filter: bool = True
    template: Path | None = None
    max_session_gb: float = 20.0
    min_free_gb: float = 20.0
    max_ram_gb: float = 4.0
    max_vram_gb: float = 2.0
    beep: bool = True
    autostart: bool = False
    duration: float | None = None
    detector: Path | None = None
    det_imgsz: int = 1280
    det_max_height: float = 0.15
    policy: PolicyConfig = field(default_factory=PolicyConfig)


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


class Resources:
    """Own RSS / CPU / VRAM, sampled every few seconds."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.proc.cpu_percent()
        self.ncpu = psutil.cpu_count() or 1
        self.rss_peak = self.cpu_peak = self.vram_peak = 0.0
        self.cpu_sum, self.samples = 0.0, 0
        self.rss = self.vram = 0.0
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # no NVIDIA driver / nvidia-ml-py: VRAM is then taken from torch if a detector runs
            self._nvml = None

    def vram_gb(self, torch_mod=None) -> float:
        if self._nvml is not None:
            try:
                for p in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._gpu):
                    if p.pid == self.proc.pid and p.usedGpuMemory:
                        return p.usedGpuMemory / 2**30
            except Exception:
                pass
        if torch_mod is not None:  # WDDM hides per-process usage: allocator reserve + ~0.5 GB CUDA context
            return torch_mod.cuda.memory_reserved() / 2**30 + 0.5
        return 0.0

    def sample(self, torch_mod=None) -> None:
        self.rss = self.proc.memory_info().rss / 2**30
        cpu = self.proc.cpu_percent() / self.ncpu  # share of the whole CPU
        self.vram = self.vram_gb(torch_mod)
        self.rss_peak, self.cpu_peak, self.vram_peak = max(self.rss_peak, self.rss), max(self.cpu_peak, cpu), max(self.vram_peak, self.vram)
        self.cpu_sum += cpu
        self.samples += 1

    def report(self) -> dict:
        return {"rss_peak_gb": round(self.rss_peak, 3), "cpu_avg_pct": round(self.cpu_sum / max(self.samples, 1), 2),
                "cpu_peak_pct": round(self.cpu_peak, 2), "vram_peak_gb": round(self.vram_peak, 3)}


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
        self.vk_toggle, self.vk_quit, self.vk_burst = (parse_key(k) for k in (cfg.toggle_key, cfg.quit_key, cfg.burst_key))
        template = load_template(cfg.template) if cfg.template else load_template()
        self.policy = CapturePolicy(cfg.policy, template if cfg.ui_filter else None)
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
        return d

    def extra(self) -> dict:
        e = {"tool": f"imagehook {__version__}", "ended": datetime.now().isoformat(timespec="seconds"),
             "recorded_seconds": round(self.rec_seconds, 1), "evaluations": self.evaluations,
             "settings": self.settings(), "resources": self.res.report(), "detector_runs": self.det_runs}
        if self.det_runs:
            e["detector_ms_avg"] = round(1000 * self.det_time / self.det_runs, 1)
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
            self.log(f"session closed: {self.session.dir}  {self.status_line()}")
            self.session = None

    def status_line(self) -> str:
        s = self.session
        if s is None:
            return ""
        return (f"saved {s.n_saved} {dict(s.saved)} | skipped {dict(s.skipped)} | {s.bytes / 2**30:.2f} GB | "
                f"RAM {self.res.rss:.2f} GB, VRAM {self.res.vram:.2f} GB")

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

    # ------------------------------------------------------------ per-frame work
    def save(self, frame: np.ndarray, wall: datetime, reason: str, novelty: float, ui: float, det: float | None) -> bool:
        h = self.cfg.height
        if h and frame.shape[0] != h:
            # bilinear without antialiasing, like the detector's GPU resize in screendet
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
        if not self.cfg.any_window and self.fg.name() not in GAME_PROCESSES:
            s.skipped["not_game"] += 1
            return
        frame = cam.grab()
        wall = datetime.now()
        if frame is None:  # nothing on screen changed since the previous grab
            s.skipped["no_new_frame"] += 1
            return
        self.evaluations += 1
        f = features(frame)
        ui = self.policy.ui(f)
        det = None
        if self.detector is not None and ui >= self.cfg.policy.ui_threshold:
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

    def check_limits(self) -> None:
        s = self.session
        self.res.sample(self.torch if self.detector is not None else None)
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
                                                    max_vram_gb=min(1.0, cfg.max_vram_gb)))
            self.torch = self.detector.torch
            self.res.sample(self.torch)
            self.log(f"detector ready: RAM {self.res.rss:.2f} GB, VRAM {self.res.vram:.2f} GB")
        import bettercam

        idx = primary_output_index(bettercam) if cfg.output_idx is None else cfg.output_idx
        cam = bettercam.create(output_idx=idx, output_color="BGR")
        self.log(f"capturing output {idx} {cam.width}x{cam.height} -> {cfg.out}  (saved height: {cfg.height or 'native'})")
        self.log(f"{cfg.toggle_key.upper()} start/stop | {cfg.burst_key} burst (keep last {cfg.burst_back:g} s + next "
                 f"{cfg.burst_forward:g} s) | {cfg.quit_key.upper()} quit")
        if not cfg.any_window:
            self.log("frames are taken only while the game window is in the foreground")
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
            self.close_session()
            cam.release()
            self.log("done")
