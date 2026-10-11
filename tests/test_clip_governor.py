from __future__ import annotations

import os
from datetime import datetime

import pytest

from imagehook.capture import GpuGovernor
from imagehook.clip import ClipRecorder, clip_path, ffmpeg_command, find_ffmpeg


def test_governor_throttles_above_target_and_recovers():
    g = GpuGovernor(target=20, hard=30, max_stride=8)
    assert g.update(25) is None and g.stride == 2
    assert g.update(22) is None and g.stride == 4
    assert g.update(15) is None and g.stride == 4  # between target/2 and target: keep
    assert g.update(5) is None and g.stride == 2
    assert g.update(0) is None and g.stride == 1
    assert g.update(0) is None and g.stride == 1


def test_governor_turns_detector_off_only_when_throttling_is_not_enough():
    g = GpuGovernor(target=20, hard=30, max_stride=8, hard_checks=3)
    # strides 2, 4, 8; "off" once the load stayed above 30% for 3 checks and the stride is already the largest
    assert [g.update(50) for _ in range(3)] == [None, None, "off"]
    assert g.stride == 8
    g2 = GpuGovernor(target=20, hard=30, max_stride=8, hard_checks=3)
    assert [g2.update(25) for _ in range(6)] == [None] * 6  # over the target but under the hard limit: never off


def test_ffmpeg_command_records_native_h264_on_the_gpu(tmp_path):
    cmd = ffmpeg_command("ffmpeg", tmp_path / "c.mp4", output_idx=1, fps=30, max_seconds=90, cq=19)
    joined = " ".join(cmd)
    assert "ddagrab=output_idx=1:framerate=30:draw_mouse=0" in joined
    assert "-c:v h264_nvenc" in joined and "-t 90" in joined
    assert "scale" not in joined  # native resolution, no resize
    assert cmd[-1].endswith("c.mp4")


def test_clip_paths_are_unique(tmp_path):
    t = datetime(2026, 10, 11, 7, 40, 0)
    a = clip_path(tmp_path, t)
    a.write_bytes(b"x")
    b = clip_path(tmp_path, t)
    assert a.name == "clip_20261011_074000.mp4" and b.name == "clip_20261011_074000_1.mp4"
    assert a.parent == tmp_path / "clips"


@pytest.mark.skipif(not os.environ.get("IMAGEHOOK_TEST_CLIP"), reason="records the screen; set IMAGEHOOK_TEST_CLIP=1")
def test_real_clip_is_readable_by_opencv(tmp_path):
    import cv2

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        pytest.skip("no ffmpeg")
    rec = ClipRecorder(ffmpeg, tmp_path, output_idx=0, fps=30, max_seconds=2)
    path = rec.start()
    rec.proc.wait(timeout=30)
    info = rec.poll()
    assert info["ok"], info
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    assert ok and frame.shape[0] >= 720
