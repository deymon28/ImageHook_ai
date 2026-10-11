"""Native-resolution gameplay clips for comparing models on the same real footage (`screendet run --source clip.mp4`).

ffmpeg does the whole job on the GPU: `ddagrab` (DXGI desktop duplication, frames stay in D3D11 memory) feeds
`h264_nvenc` (the dedicated video encoder, not the shader cores), so the CPU only runs a small ffmpeg process and
no frame passes through Python. Needs ffmpeg 6+ with ddagrab and NVENC on PATH (or --ffmpeg / IMAGEHOOK_FFMPEG).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BELOW_NORMAL = 0x00004000  # Windows process creation flag


def find_ffmpeg(explicit: str | None = None) -> str | None:
    cand = explicit or os.environ.get("IMAGEHOOK_FFMPEG") or "ffmpeg"
    return shutil.which(cand) or (cand if Path(cand).is_file() else None)


def ffmpeg_command(ffmpeg: str, out: Path, output_idx: int, fps: int, max_seconds: float, cq: int) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats", "-y",
        "-f", "lavfi", "-i", f"ddagrab=output_idx={output_idx}:framerate={fps}:draw_mouse=0",
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0", "-g", str(2 * fps),
        "-t", f"{max_seconds:g}", "-movflags", "+faststart", str(out),
    ]


def clip_path(root: Path, t: datetime) -> Path:
    d = root / "clips"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"clip_{t:%Y%m%d_%H%M%S}.mp4"
    n = 1
    while p.exists():
        p = d / f"clip_{t:%Y%m%d_%H%M%S}_{n}.mp4"
        n += 1
    return p


class ClipRecorder:
    """One ffmpeg child process per clip. stop() asks ffmpeg to finish the file cleanly ('q' on stdin)."""

    def __init__(self, ffmpeg: str, root: Path, output_idx: int, fps: int = 30, max_seconds: float = 120.0,
                 cq: int = 19):
        self.ffmpeg, self.root, self.output_idx = ffmpeg, root, output_idx
        self.fps, self.max_seconds, self.cq = fps, max_seconds, cq
        self.proc: subprocess.Popen | None = None
        self.path: Path | None = None
        self.started = 0.0
        self.clips: list[dict] = []

    @property
    def active(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.active else None

    def start(self) -> Path:
        self.path = clip_path(self.root, datetime.now())
        flags = BELOW_NORMAL if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            ffmpeg_command(self.ffmpeg, self.path, self.output_idx, self.fps, self.max_seconds, self.cq),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)
        self.started = time.monotonic()
        return self.path

    def poll(self) -> dict | None:
        """Call regularly: returns the clip info once ffmpeg has ended by itself (time limit or error)."""
        if self.proc is not None and self.proc.poll() is not None:
            return self._finish()
        return None

    def stop(self, timeout: float = 15.0) -> dict | None:
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"q")
                self.proc.stdin.flush()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=5)
        return self._finish()

    def _finish(self) -> dict:
        proc, self.proc = self.proc, None
        err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        size = self.path.stat().st_size if self.path.exists() else 0
        info = {"file": self.path.name, "seconds": round(time.monotonic() - self.started, 1), "bytes": size,
                "ok": proc.returncode == 0 and size > 0}
        if err:
            info["error"] = err.splitlines()[-1][:300]
        self.clips.append(info)
        return info
