"""Offline replay: run recorded frames or a video through the same filters as the live recorder, without the game.

Sources:
  a session folder (or its frames/ folder) recorded at a fixed interval: frame times come from the names
  a video file: a frame is evaluated every `tick` seconds of video time
If the session was labeled (review.json with pos/neg/skip, labels/), the report also shows how many frames of each
status are kept and whether every frame with people still has a kept frame with people within +-2 s.
--write copies the kept frames into a new session folder with the same names, so old sessions can be thinned too.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .detector import relevant_conf
from .filters import CapturePolicy, Features, features

IMG_EXT = {".jpg", ".jpeg", ".png"}
NAME_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})_(\d{3})$")
SESSION_RE = re.compile(r"session_(\d{8})_(\d{6})")
COVER_S = 2.01  # a frame with people is "covered" if a kept frame with people is this close in time
SMALL_H = 40 / 1080  # "small" person: under 40 px at 1080p


def name_seconds(stem: str) -> float | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000


def ordered_frames(frames_dir: Path, session_start: float | None) -> list[tuple[float, Path]]:
    """Frames sorted by capture time. Names are only a time of day: a frame earlier than the session start was taken
    after midnight (same rule as the labeling tool)."""
    out = []
    for p in frames_dir.iterdir():
        if p.suffix.lower() not in IMG_EXT:
            continue
        t = name_seconds(p.stem)
        if t is None:
            continue
        if session_start is not None and t - session_start < -60:
            t += 86400
        out.append((t, p))
    return sorted(out)


def session_start_seconds(session: Path) -> float | None:
    m = SESSION_RE.search(session.name)
    if not m:
        return None
    hh, mm, ss = int(m[2][:2]), int(m[2][2:4]), int(m[2][4:])
    return hh * 3600 + mm * 60 + ss


@dataclass
class Item:
    t: float
    name: str
    feats: Features
    det: float | None = None
    path: Path | None = None
    frame: np.ndarray | None = None


@dataclass
class ReplayResult:
    decisions: list[tuple[str, bool, str]] = field(default_factory=list)  # (name, saved, reason)
    reasons: Counter = field(default_factory=Counter)

    @property
    def kept(self) -> list[str]:
        return [n for n, s, _ in self.decisions if s]


def read_det_labels(path: Path, max_height: float) -> float:
    """YOLO txt with a confidence column (prelabel output) -> relevant max confidence."""
    if not path.exists():
        return 0.0
    boxes, confs = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) >= 6:
            boxes.append(tuple(map(float, v[1:5])))
            confs.append(float(v[5]))
    return relevant_conf(boxes, confs, max_height)


def iter_session(session: Path, det_labels: Path | None, det_max_height: float, full: bool) -> Iterator[Item]:
    frames_dir = session / "frames" if (session / "frames").is_dir() else session
    sess_dir = frames_dir.parent if frames_dir.name == "frames" else frames_dir
    frames = ordered_frames(frames_dir, session_start_seconds(sess_dir))
    flag = cv2.IMREAD_COLOR if full else cv2.IMREAD_REDUCED_COLOR_2  # half size is enough for the features

    def load(tp: tuple[float, Path]) -> Item:
        t, p = tp
        img = cv2.imread(str(p), flag)
        det = read_det_labels(det_labels / f"{p.stem}.txt", det_max_height) if det_labels else None
        return Item(t, p.stem, features(img), det, p, img if full else None)

    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(frames), 64):  # bounded read-ahead
            yield from ex.map(load, frames[i:i + 64])


def iter_video(path: Path, tick: float, start: datetime) -> Iterator[Item]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx, next_t = 0, 0.0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                return
            t = idx / fps
            idx += 1
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                return
            next_t = t + tick
            wall = start + timedelta(seconds=t)
            yield Item(t, f"{wall:%H%M%S}_{wall.microsecond // 1000:03d}", features(frame), frame=frame)
    finally:
        cap.release()


def replay(items: Iterator[Item], policy: CapturePolicy, detector=None, write_dir: Path | None = None) -> ReplayResult:
    res = ReplayResult()
    if write_dir:
        write_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        det = it.det
        if detector is not None and it.frame is not None and policy.ui(it.feats) >= policy.cfg.ui_threshold:
            det = detector(it.frame)
        d = policy.evaluate(it.t, it.feats, det)
        res.decisions.append((it.name, d.save, d.reason))
        res.reasons[d.reason] += 1
        if d.save:
            policy.saved(it.t, it.feats)
            if write_dir:
                if it.path is not None:
                    shutil.copy2(it.path, write_dir / it.path.name)
                elif it.frame is not None:
                    cv2.imwrite(str(write_dir / f"{it.name}.jpg"), it.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return res


def label_report(session: Path, res: ReplayResult) -> dict | None:
    """Kept frames per review status and coverage of frames with people (needs review.json)."""
    review = session / "review.json"
    if not review.exists():
        return None
    status = json.loads(review.read_text(encoding="utf-8")).get("status", {})
    start = session_start_seconds(session)
    times = {}
    for name, _, _ in res.decisions:
        t = name_seconds(name)
        if start is not None and t - start < -60:
            t += 86400
        times[name] = t
    kept = set(res.kept)
    by_status: dict[str, list[int]] = {}
    for name, saved, _ in res.decisions:
        st = status.get(name, "unlabeled")
        k = by_status.setdefault(st, [0, 0])
        k[0] += saved
        k[1] += 1
    pos = [n for n, _, _ in res.decisions if status.get(n) == "pos"]
    kept_pos_t = np.array(sorted(times[n] for n in pos if n in kept))

    def covered(n: str) -> bool:
        return bool(len(kept_pos_t)) and bool(np.min(np.abs(kept_pos_t - times[n])) <= COVER_S)

    small = []
    for n in pos:
        lbl = session / "labels" / f"{n}.txt"
        hs = [float(v.split()[4]) for v in lbl.read_text(encoding="utf-8").splitlines() if v.strip()] if lbl.exists() else []
        if hs and min(hs) < SMALL_H:
            small.append(n)
    cov = [covered(n) for n in pos]
    cov_small = [covered(n) for n in small]
    return {"by_status": {k: {"kept": v[0], "total": v[1]} for k, v in sorted(by_status.items())},
            "pos_covered": round(float(np.mean(cov)), 4) if cov else None,
            "small_pos_covered": round(float(np.mean(cov_small)), 4) if cov_small else None,
            "small_pos_frames": len(small)}
