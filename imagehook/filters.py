"""Frame features and the save/skip policy. Pure numpy/OpenCV, no capture or file I/O, so it is unit-tested and
shared by the live recorder and the offline replay.

Per captured frame two small features are computed (a few ms at 1440p):
  thumb  320x180 grayscale, used to measure how much the scene changed since the last saved frame
  icon   the bottom-left body-status HUD icon area, normalized to 68x92 (the 540p size of the template); interface
         screens (inventory, map, loading, death/extraction screens) do not have this icon
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

THUMB_SIZE = (320, 180)  # (w, h)
ICON_SIZE = (68, 92)  # (w, h): the icon zone of a 960x540 frame
# normalized (x1, y1, x2, y2) area of the body-status silhouette icon, same zone as the labeling tool's HUD_ZONES
ICON_ZONE = (0.0, 0.75, 0.07, 0.92)
# HUD areas that change on their own (FPS/ping/timer/perf overlay at the top, chat and body icon bottom-left,
# weapon bottom-right, stamina bar bottom-centre): ignored when measuring scene change
CHANGE_IGNORE = ((0.0, 0.0, 1.0, 0.05), (0.0, 0.93, 1.0, 1.0), (0.0, 0.55, 0.22, 1.0), (0.85, 0.8, 1.0, 1.0))
LOCAL_CELL = 20  # thumb pixels per side of a local-change cell (16x9 grid)
LOCAL_DIFF = 20  # grayscale difference that counts a thumb pixel as changed
DEFAULT_TEMPLATE = Path(__file__).with_name("assets") / "hud_icon.png"
JITTER = 0.05  # s of timing slack in interval comparisons


def _change_mask() -> np.ndarray:
    w, h = THUMB_SIZE
    m = np.ones((h, w), bool)
    for x1, y1, x2, y2 in CHANGE_IGNORE:
        m[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = False
    return m


CHANGE_MASK = _change_mask()


@dataclass
class Features:
    thumb: np.ndarray  # uint8 (180, 320)
    icon: np.ndarray  # float32 (92, 68)


def features(frame: np.ndarray) -> Features:
    """BGR (or grayscale) frame of any 16:9 size -> Features."""
    gray = frame if frame.ndim == 2 else None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = ICON_ZONE
    icon = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    thumb = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    if gray is None:
        icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    icon = cv2.resize(icon, ICON_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return Features(thumb=thumb, icon=icon)


def load_template(path: Path = DEFAULT_TEMPLATE) -> np.ndarray:
    tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise FileNotFoundError(f"HUD icon template not found: {path}")
    if tmpl.shape != (ICON_SIZE[1], ICON_SIZE[0]):
        tmpl = cv2.resize(tmpl, ICON_SIZE, interpolation=cv2.INTER_AREA)
    return tmpl.astype(np.float32)


def build_template(icons: list[np.ndarray]) -> np.ndarray:
    """Median icon of many frames (most of them gameplay) -> template."""
    return np.median(np.stack(icons), axis=0).astype(np.float32)


def ui_score(icon: np.ndarray, template: np.ndarray) -> float:
    """Similarity of the HUD icon area to the template: ~1 in gameplay, ~0 or negative on interface screens."""
    if float(icon.std()) < 1.0:  # flat area (black loading screen): correlation is undefined
        return 0.0
    return float(cv2.matchTemplate(icon, template, cv2.TM_CCOEFF_NORMED)[0, 0])


def change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(global, local) change between two thumbs, HUD areas ignored.
    global: mean absolute grayscale difference (0..255); camera turns give 20-50, a still view 1-5.
    local: largest share of changed pixels in one 20x20 cell (0..1); catches something moving in a still view."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    g = float(d[CHANGE_MASK].mean())
    changed = (d > LOCAL_DIFF) & CHANGE_MASK
    h, w = changed.shape
    c = LOCAL_CELL
    local = float(changed[: h // c * c, : w // c * c].reshape(h // c, c, w // c, c).mean(axis=(1, 3)).max())
    return g, local


@dataclass
class PolicyConfig:
    min_interval: float = 0.5  # never save two frames closer than this (s)
    base_interval: float = 1.0  # a "normally" changing scene is saved about this often (s)
    max_interval: float = 10.0  # a nearly static scene is still saved this often (s), exact duplicates never
    idle_interval: float = 2.0  # base interval while the detector sees nobody (only with --detector)
    global_ref: float = 15.0  # global change that counts as one unit of novelty
    local_ref: float = 0.6  # local change that counts as one unit of novelty
    dup_novelty: float = 0.15  # below this a frame is a duplicate of the last saved one
    ui_threshold: float = 0.3  # ui_score below this = interface screen
    det_conf: float = 0.2  # detector confidence that counts as "a person is visible"
    det_interval: float = 1.0  # while the detector sees someone, save a non-duplicate frame this often (s)
    det_hold: float = 2.0  # a detector hit keeps the fast rate for this long (s)


@dataclass
class Decision:
    save: bool
    reason: str  # saved: first|change|periodic|detection|burst; skipped: ui|duplicate|waiting
    novelty: float = 0.0
    ui: float = 1.0


class CapturePolicy:
    """Decides for each evaluated frame whether to save it.

    novelty = max(global / global_ref, local / local_ref) against the LAST SAVED frame. A frame is saved when
    novelty >= base_interval / elapsed: a big change is saved after min_interval already, an average one after
    base_interval, a small one only after several seconds, and anything above dup_novelty after max_interval.
    With a detector, frames where nobody is seen use idle_interval as the base, and a hit (held for det_hold s)
    saves a non-duplicate frame every det_interval."""

    def __init__(self, cfg: PolicyConfig, template: np.ndarray | None):
        self.cfg = cfg
        self.template = template
        self.last: np.ndarray | None = None
        self.last_t = -1e18
        self.hit_until = -1e18

    def reset(self) -> None:
        self.last, self.last_t, self.hit_until = None, -1e18, -1e18

    def ui(self, f: Features) -> float:
        return 1.0 if self.template is None else ui_score(f.icon, self.template)

    def novelty(self, f: Features) -> float:
        if self.last is None:
            return float("inf")
        g, local = change(self.last, f.thumb)
        return max(g / self.cfg.global_ref, local / self.cfg.local_ref)

    def evaluate(self, now: float, f: Features, det_conf: float | None = None, burst: bool = False,
                 ui: float | None = None) -> Decision:
        """det_conf: the detector's max confidence for this frame, None when no detector runs.
        burst: the user asked to keep frames now (still skips interface screens and exact duplicates)."""
        cfg = self.cfg
        if det_conf is not None and det_conf >= cfg.det_conf:
            self.hit_until = now + cfg.det_hold
        ui = self.ui(f) if ui is None else ui
        if ui < cfg.ui_threshold:
            return Decision(False, "ui", 0.0, ui)
        nov = self.novelty(f)
        if self.last is None:
            return Decision(True, "first", nov, ui)
        if nov < cfg.dup_novelty:
            return Decision(False, "duplicate", nov, ui)
        if burst:
            return Decision(True, "burst", nov, ui)
        elapsed = now - self.last_t + JITTER  # ticks and recorded frame times drift by a few ms
        if elapsed < cfg.min_interval:
            return Decision(False, "waiting", nov, ui)
        if det_conf is not None and now <= self.hit_until and elapsed >= cfg.det_interval:
            return Decision(True, "detection", nov, ui)
        base = cfg.idle_interval if det_conf is not None else cfg.base_interval
        if nov * elapsed >= base:
            return Decision(True, "change", nov, ui)
        if elapsed >= cfg.max_interval:
            return Decision(True, "periodic", nov, ui)
        return Decision(False, "waiting", nov, ui)

    def saved(self, now: float, f: Features) -> None:
        self.last, self.last_t = f.thumb, now
