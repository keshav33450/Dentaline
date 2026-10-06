"""
SegmentAnyTooth wrapper (Nguyen et al., J Dent Sci 2025; code MIT, weights non-commercial research licence).
https://github.com/thangngoc89/SegmentAnyTooth

Their predict() reloads the models for every photo; this wrapper loads once and returns an FDI mask.
Expects the repo checked out (segmentanytooth.py, sam.py, utils.py) and the weight files:
    segmentanytooth_yolo11_upper.pt, segmentanytooth_yolo11_lower.pt, segmentanytooth_vit_tiny.pt  (+ front/right)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REQUIRED = ["segmentanytooth_vit_tiny.pt", "segmentanytooth_yolo11_upper.pt", "segmentanytooth_yolo11_lower.pt"]


def find_weights(*roots) -> Path | None:
    for r in roots:
        if not r:
            continue
        for p in Path(r).rglob("segmentanytooth_vit_tiny.pt"):
            if all((p.parent / n).exists() for n in REQUIRED):
                return p.parent
    return None


class SegmentAnyTooth:
    def __init__(self, repo_dir: str | Path, weight_dir: str | Path, device: str | None = None):
        repo_dir = str(Path(repo_dir).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        import torch
        from sam import sam_load, sam_predict   # from the SegmentAnyTooth repo
        from ultralytics import YOLO
        self._sam_predict = sam_predict
        self.wd = Path(weight_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_load(str(self.wd / "segmentanytooth_vit_tiny.pt")).to(self.device).eval()
        self.yolo = {}
        self._YOLO = YOLO

    def _det(self, view):
        if view not in self.yolo:
            self.yolo[view] = self._YOLO(str(self.wd / f"segmentanytooth_yolo11_{view}.pt"))
        return self.yolo[view]

    def predict(self, img_bgr: np.ndarray, view: str) -> np.ndarray:
        """-> uint8 mask, pixel value = FDI tooth number (0 = background). view: upper | lower."""
        r = self._det(view).predict(img_bgr, verbose=False, device=self.device)[0]
        h, w = img_bgr.shape[:2]
        out = np.zeros((h, w), np.uint8)
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        order = np.argsort(conf)              # paint low-confidence first so confident teeth win overlaps
        boxes, cls = boxes[order], cls[order]
        masks = self._sam_predict(sam=self.sam, boxes_xyxy=boxes, image=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                  batch_size=10)
        if masks.ndim == 2:
            masks = masks[None]
        for c, m in zip(cls, masks):
            try:
                fdi = int(str(r.names[int(c)])[-2:])
            except ValueError:
                continue
            out[m.astype(bool)] = fdi
        return out
