from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from imagehook.filters import ICON_SIZE, ICON_ZONE


def make_template() -> np.ndarray:
    """Synthetic HUD icon (float32, ICON_SIZE): a bright silhouette on a dark background."""
    w, h = ICON_SIZE
    img = np.full((h, w), 40, np.uint8)
    cv2.circle(img, (w // 2, h // 5), h // 9, 220, -1)  # head
    cv2.ellipse(img, (w // 2, h // 2 + 4), (w // 4, h // 4), 0, 0, 360, 200, -1)  # body
    cv2.rectangle(img, (w // 2 - 12, h * 3 // 4), (w // 2 - 4, h - 6), 180, -1)  # legs
    cv2.rectangle(img, (w // 2 + 4, h * 3 // 4), (w // 2 + 12, h - 6), 180, -1)
    return img.astype(np.float32)


def write_template(folder: Path) -> Path:
    path = folder / "hud_icon.png"
    cv2.imwrite(str(path), make_template().astype(np.uint8))
    return path


def scene(seed: int, h: int = 1440, w: int = 2560, icon: bool = True) -> np.ndarray:
    """Synthetic 16:9 BGR 'gameplay' frame: smooth random texture, optionally with the HUD icon pasted in."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (18, 32, 3), dtype=np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if icon:
        put_icon(img)
    return img


def put_icon(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = ICON_ZONE
    zx1, zy1, zx2, zy2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    tmpl = make_template().astype(np.uint8)
    icon = cv2.resize(tmpl, (zx2 - zx1, zy2 - zy1), interpolation=cv2.INTER_LINEAR)
    img[zy1:zy2, zx1:zx2] = icon[:, :, None]


@pytest.fixture
def template() -> np.ndarray:
    return make_template()
