from __future__ import annotations

import cv2
import numpy as np
import pytest

from imagehook.filters import ICON_ZONE, load_template


def scene(seed: int, h: int = 1440, w: int = 2560, icon: bool = True) -> np.ndarray:
    """Synthetic 16:9 BGR 'gameplay' frame: smooth random texture, optionally with the HUD body icon pasted in."""
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
    tmpl = load_template().astype(np.uint8)
    icon = cv2.resize(tmpl, (zx2 - zx1, zy2 - zy1), interpolation=cv2.INTER_LINEAR)
    img[zy1:zy2, zx1:zx2] = icon[:, :, None]


@pytest.fixture
def template() -> np.ndarray:
    return load_template()
