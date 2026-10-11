"""Optional person detector used only to decide WHEN to save (never to drop frames on its own).

PyTorch + ultralytics, loaded once at startup (never inside the capture loop). TensorRT engines are refused on
purpose: building or loading them is too heavy while the game runs. The PyTorch allocator is capped; the capture
loop also watches the process' total VRAM (CUDA context included) and turns the detector off above the budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .filters import ICON_ZONE


@dataclass
class DetectorConfig:
    weights: Path
    imgsz: int = 1280
    conf: float = 0.1  # proposals below this are not even reported; the policy threshold is PolicyConfig.det_conf
    max_height: float = 0.15  # ignore boxes taller than this share of the frame: close people matter little
    max_vram_gb: float = 1.0  # PyTorch allocator cap (the CUDA context comes on top)


def relevant_conf(boxes_xywhn: list[tuple[float, float, float, float]], confs: list[float], max_height: float) -> float:
    """Max confidence over boxes that are not the HUD body icon and not taller than max_height (0 if none)."""
    x1, y1, x2, y2 = ICON_ZONE
    best = 0.0
    for (cx, cy, _bw, bh), c in zip(boxes_xywhn, confs):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            continue
        if bh > max_height:
            continue
        best = max(best, c)
    return best


class Detector:
    def __init__(self, cfg: DetectorConfig):
        if cfg.weights.suffix.lower() != ".pt":
            raise ValueError("--detector needs PyTorch weights (.pt); TensorRT engines are not used during capture")
        if not cfg.weights.exists():
            raise FileNotFoundError(cfg.weights)
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available: the detector would run on the CPU and load it heavily")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(cfg.max_vram_gb * 2**30 / total, 1.0))
        self.cfg = cfg
        self.torch = torch
        self.model = YOLO(str(cfg.weights))
        self(np.zeros((1440, 2560, 3), np.uint8))  # warm-up so the first real frame is not slow

    def __call__(self, frame_bgr: np.ndarray) -> float:
        r = self.model.predict(frame_bgr, imgsz=self.cfg.imgsz, conf=self.cfg.conf, quantize=16, device=0,
                               max_det=50, verbose=False)[0]
        return relevant_conf(r.boxes.xywhn.tolist(), r.boxes.conf.tolist(), self.cfg.max_height)

    def vram_peak_gb(self) -> float:
        return self.torch.cuda.max_memory_reserved() / 2**30
