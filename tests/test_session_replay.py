from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from imagehook.detector import relevant_conf
from imagehook.filters import CapturePolicy, PolicyConfig, load_template
from imagehook.hotkeys import parse_key
from imagehook.replay import iter_session, label_report, ordered_frames, replay
from imagehook.session import Session, frame_stem, session_name
from tests.conftest import scene

# the labeling tool's export key: "<session>_<HHMMSS>_<mmm>"
LABEL_KEY_RE = re.compile(r"(.+)_(\d{2})(\d{2})(\d{2})_(\d{3})$")


def test_names_are_compatible_with_the_labeling_tool():
    t = datetime(2026, 10, 8, 22, 36, 25, 72_500)
    assert session_name(t) == "session_20261008_223625"
    assert frame_stem(t) == "223625_072"
    assert LABEL_KEY_RE.match(f"{session_name(t)}_{frame_stem(t)}")


def test_session_writes_frames_log_and_summary(tmp_path):
    t0 = datetime(2026, 10, 8, 23, 59, 59, 999_000)
    s = Session(tmp_path, t0)
    img = scene(1, 144, 256)
    names = [s.save(img, t0, "change", 1.5, 0.9), s.save(img, t0, "burst", 0.4, 0.95, 0.7)]
    assert names[0] == "235959_999"
    assert names[1] != names[0] and names[1].startswith("00000")  # same ms -> next ms, here after midnight
    s.skipped["ui"] += 3
    s.close({"recorded_seconds": 1.0})
    files = sorted(p.name for p in (s.dir / "frames").iterdir())
    assert files == sorted(f"{n}.jpg" for n in names)
    assert all(cv2.imread(str(s.dir / "frames" / f)) is not None for f in files)
    rows = list(csv.DictReader(open(s.dir / "capture_log.csv", encoding="utf-8")))
    assert [r["reason"] for r in rows] == ["change", "burst"] and rows[1]["det_conf"] == "0.700"
    summary = json.loads((s.dir / "capture.json").read_text(encoding="utf-8"))
    assert summary["saved_by_reason"] == {"change": 1, "burst": 1}
    assert summary["skipped_by_reason"] == {"ui": 3}
    assert summary["bytes"] > 0 and summary["session"] == "session_20261008_235959"


def test_second_session_in_same_second_gets_its_own_folder(tmp_path):
    t = datetime(2026, 10, 8, 12, 0, 0)
    a, b = Session(tmp_path, t), Session(tmp_path, t)
    assert a.dir != b.dir
    a.close({}), b.close({})


def test_ordered_frames_cross_midnight(tmp_path):
    for n in ("235958_000", "235959_500", "000001_000", "notaframe"):
        (tmp_path / f"{n}.jpg").write_bytes(b"x")
    order = [p.stem for _, p in ordered_frames(tmp_path, session_start=23 * 3600 + 59 * 60)]
    assert order == ["235958_000", "235959_500", "000001_000"]


def make_session(root, frames: list[tuple[str, np.ndarray]], status: dict | None = None):
    s = root / "session_20261008_120000"
    (s / "frames").mkdir(parents=True)
    for name, img in frames:
        cv2.imwrite(str(s / "frames" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if status:
        (s / "review.json").write_text(json.dumps({"version": 1, "status": status}), encoding="utf-8")
    return s


def test_replay_skips_interface_and_duplicates_and_reports_labels(tmp_path):
    h, w = 540, 960
    a, b, c = scene(1, h, w), scene(2, h, w), scene(3, h, w)
    ui = scene(4, h, w, icon=False)
    seq = [("120001_000", a), ("120002_000", a), ("120003_000", ui), ("120004_000", b), ("120005_000", c),
           ("120006_000", c)]
    status = {"120001_000": "neg", "120002_000": "neg", "120003_000": "skip", "120004_000": "pos",
              "120005_000": "pos", "120006_000": "pos"}
    s = make_session(tmp_path, seq, status)
    policy = CapturePolicy(PolicyConfig(), load_template())
    out = tmp_path / "thin" / s.name / "frames"
    res = replay(iter_session(s, None, 0.15, full=False), policy, write_dir=out)
    assert [r for _, _, r in res.decisions] == ["first", "duplicate", "ui", "change", "change", "duplicate"]
    assert sorted(p.stem for p in out.iterdir()) == ["120001_000", "120004_000", "120005_000"]
    rep = label_report(s, res)
    assert rep["by_status"]["skip"] == {"kept": 0, "total": 1}
    assert rep["by_status"]["pos"] == {"kept": 2, "total": 3}
    assert rep["pos_covered"] == 1.0


def test_replay_emulates_detector_from_prelabel_output(tmp_path):
    h, w = 540, 960
    base = scene(5, h, w)
    drift = np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    s = make_session(tmp_path, [("120001_000", base), ("120002_000", drift)])
    (s / "labels_auto").mkdir()
    (s / "labels_auto" / "120002_000.txt").write_text("0 0.5 0.5 0.01 0.03 0.800\n", encoding="utf-8")
    policy = CapturePolicy(PolicyConfig(), load_template())
    res = replay(iter_session(s, s / "labels_auto", 0.15, full=False), policy)
    assert [r for _, _, r in res.decisions] == ["first", "detection"]


def test_relevant_conf_ignores_hud_icon_and_close_people():
    boxes = [(0.03, 0.83, 0.04, 0.15), (0.5, 0.5, 0.2, 0.6), (0.6, 0.45, 0.01, 0.03)]
    assert relevant_conf(boxes, [0.9, 0.95, 0.3], max_height=0.15) == 0.3
    assert relevant_conf(boxes[:2], [0.9, 0.95], max_height=0.15) == 0.0


@pytest.mark.parametrize("name,vk", [("f8", 0x77), ("F9", 0x78), ("mouse5", 0x06), ("mouse4", 0x05),
                                     ("x", 0x58), ("5", 0x35), ("numpad1", 0x61), ("0x2d", 0x2D)])
def test_parse_key(name, vk):
    assert parse_key(name) == vk


def test_parse_key_rejects_unknown():
    with pytest.raises(ValueError):
        parse_key("hyper")
