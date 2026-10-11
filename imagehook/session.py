"""Session folder layout, frame names, the background JPEG writer and the per-session summary.

Layout (compatible with the labeling tools, which parse the frame names):
  <out>/session_YYYYmmdd_HHMMSS/frames/HHMMSS_mmm.jpg   time of day with milliseconds (sessions may cross midnight)
  <out>/session_YYYYmmdd_HHMMSS/capture_log.csv           one row per saved frame: why it was saved
  <out>/session_YYYYmmdd_HHMMSS/capture.json              settings, saved/skipped counts by reason, resource peaks
"""
from __future__ import annotations

import csv
import json
import queue
import shutil
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

LOG_FIELDS = ("name", "reason", "novelty", "ui", "det_conf")


def session_name(t: datetime) -> str:
    return t.strftime("session_%Y%m%d_%H%M%S")


def frame_stem(t: datetime) -> str:
    return f"{t:%H%M%S}_{t.microsecond // 1000:03d}"


class Session:
    """One recording session: unique frame names, an async writer thread, byte/disk budgets and the summary."""

    def __init__(self, root: Path, started: datetime, ext: str = "jpg", quality: int = 95, queue_size: int = 24):
        self.dir = root / session_name(started)
        suffix = 1
        while self.dir.exists():  # two sessions started within the same second
            self.dir = root / f"{session_name(started)}_{suffix}"
            suffix += 1
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True)
        self.started = started
        self.ext, self.quality = ext, quality
        self.saved: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self.bytes = 0
        self.dropped = 0
        self.write_errors = 0
        self._names: set[str] = set()
        self._jobs: queue.Queue = queue.Queue(maxsize=queue_size)
        self._log_file = open(self.dir / "capture_log.csv", "w", newline="", encoding="utf-8")
        self._log = csv.writer(self._log_file)
        self._log.writerow(LOG_FIELDS)
        self._lock = threading.Lock()
        self._writer = threading.Thread(target=self._write_loop, name="jpeg-writer", daemon=True)
        self._writer.start()

    def unique_stem(self, t: datetime) -> str:
        stem = frame_stem(t)
        ms = 0
        while stem in self._names:  # same millisecond: shift by 1 ms so names stay sortable times
            ms += 1
            stem = frame_stem(datetime.fromtimestamp(t.timestamp() + ms / 1000))
        self._names.add(stem)
        return stem

    def save(self, frame: np.ndarray, t: datetime, reason: str, novelty: float = 0.0, ui: float = 1.0,
             det_conf: float | None = None, timeout: float = 2.0) -> str | None:
        """Queue a frame for writing. Returns its name, or None if the writer stayed full for `timeout` s."""
        stem = self.unique_stem(t)
        try:
            self._jobs.put((self.frames_dir / f"{stem}.{self.ext}", frame), timeout=timeout)
        except queue.Full:
            self._names.discard(stem)
            self.dropped += 1
            return None
        self.saved[reason] += 1
        nov = "" if novelty == float("inf") else f"{novelty:.3f}"
        self._log.writerow((stem, reason, nov, f"{ui:.3f}", "" if det_conf is None else f"{det_conf:.3f}"))
        return stem

    def _write_loop(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality] if self.ext == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        while True:
            item = self._jobs.get()
            if item is None:
                return
            path, img = item
            ok, buf = cv2.imencode(f".{self.ext}", img, params)
            try:
                if not ok:
                    raise OSError("encode failed")
                path.write_bytes(buf.tobytes())
                with self._lock:
                    self.bytes += len(buf)
            except OSError as e:
                self.write_errors += 1
                print(f"[writer] failed to write {path.name}: {e}")

    @property
    def n_saved(self) -> int:
        return sum(self.saved.values())

    def free_gb(self) -> float:
        return shutil.disk_usage(self.dir).free / 2**30

    def summary(self, extra: dict) -> dict:
        return {
            "session": self.dir.name,
            "started": self.started.isoformat(timespec="seconds"),
            "saved_total": self.n_saved,
            "saved_by_reason": dict(self.saved),
            "skipped_by_reason": dict(self.skipped),
            "dropped_writer_full": self.dropped,
            "write_errors": self.write_errors,
            "bytes": self.bytes,
            **extra,
        }

    def write_summary(self, extra: dict) -> None:
        self._log_file.flush()
        tmp = self.dir / "capture.json.tmp"
        tmp.write_text(json.dumps(self.summary(extra), indent=2), encoding="utf-8")
        tmp.replace(self.dir / "capture.json")

    def close(self, extra: dict, timeout: float = 60.0) -> None:
        self._jobs.put(None)
        self._writer.join(timeout=timeout)
        self.write_summary(extra)
        self._log_file.close()
