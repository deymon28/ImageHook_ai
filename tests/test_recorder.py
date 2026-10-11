from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np
import pytest

from imagehook.capture import Recorder, RecordConfig
from tests.conftest import scene

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the recorder uses Win32 APIs")


class FakeCam:
    def __init__(self, frames):
        self.frames = list(frames)

    def grab(self):
        return self.frames.pop(0) if self.frames else None


def drifted(base: np.ndarray, k: int) -> np.ndarray:
    return np.clip(base.astype(np.int16) + 4 * k, 0, 255).astype(np.uint8)


def test_burst_keeps_recent_waiting_frames(tmp_path, monkeypatch):
    rec = Recorder(RecordConfig(out=tmp_path, any_window=True, beep=False, burst_back=10.0))
    rec.start()
    base = scene(1, 360, 640)
    # one new view, then slowly drifting frames that the adaptive interval does not save yet
    cam = FakeCam([base] + [drifted(base, 1)] * 3)
    clock = iter([100.0, 100.5, 101.0, 101.5])
    monkeypatch.setattr("imagehook.capture.time.monotonic", lambda: next(clock))
    for _ in range(4):
        rec.tick(cam)
    s = rec.session
    assert dict(s.saved) == {"first": 1}
    assert s.skipped["waiting"] == 3 and len(rec.ring) == 3
    rec.flush_ring()
    assert s.saved["burst"] == 3 and s.skipped["waiting"] == 0 and not rec.ring
    rec.close_session()
    session = next(tmp_path.glob("session_*"))
    assert len(list((session / "frames").glob("*.jpg"))) == 4
    summary = json.loads((session / "capture.json").read_text(encoding="utf-8"))
    assert summary["saved_by_reason"] == {"first": 1, "burst": 3}
    assert "out" not in summary["settings"]  # no local paths in the summary


def test_interface_frames_are_not_buffered_for_bursts(tmp_path, monkeypatch):
    rec = Recorder(RecordConfig(out=tmp_path, any_window=True, beep=False))
    rec.start()
    cam = FakeCam([scene(2, 360, 640, icon=False)] * 2)
    clock = iter([0.0, 0.5])
    monkeypatch.setattr("imagehook.capture.time.monotonic", lambda: next(clock))
    rec.tick(cam), rec.tick(cam)
    assert rec.session.skipped["ui"] == 2 and not rec.ring and rec.session.n_saved == 0
    rec.close_session()
