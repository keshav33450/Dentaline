"""Tooth crop used IDENTICALLY in training and inference (same padding, masking, size)."""
from __future__ import annotations

import cv2
import numpy as np

CROP_SIZE = 288          # saved crop size; classifiers resize/crop to their own input
PAD = 0.12               # context around the tooth box (fraction of box size)
OUTSIDE_DIM = 0.35       # keep 35% brightness outside the tooth mask (context without distraction)


def polygon_to_mask(poly_xy: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)], 255)
    return m


def tooth_crop(img: np.ndarray, mask: np.ndarray, size: int = CROP_SIZE) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    side = int(max(w, h) * (1 + 2 * PAD))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    H, W = img.shape[:2]
    canvas = np.zeros((side, side, 3), np.uint8)
    mcan = np.zeros((side, side), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, X0), max(0, Y0), min(W, X0 + side), min(H, Y0 + side)
    canvas[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = img[sy0:sy1, sx0:sx1]
    mcan[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = mask[sy0:sy1, sx0:sx1]
    soft = cv2.GaussianBlur((mcan > 0).astype(np.float32), (0, 0), max(1.0, side / 150))
    weight = OUTSIDE_DIM + (1 - OUTSIDE_DIM) * soft[..., None]
    out = (canvas.astype(np.float32) * weight).astype(np.uint8)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
