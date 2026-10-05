"""DentalX — shared detection metrics helper (mirrors the senior project's scripts/metrics.py).

Thin wrapper over ultralytics validation so every model reports metrics the same way:
per-class Precision / Recall / F1 / mAP@50 / mAP@50-95.
"""
from __future__ import annotations


def per_class_table(box, names) -> list[dict]:
    """Build a per-class P/R/F1/mAP table from an ultralytics `metrics.box` object."""
    rows = []
    for i, idx in enumerate(box.ap_class_index):
        p = float(box.p[i]); r = float(box.r[i])
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        rows.append({
            "class": names[int(idx)],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "map50": round(float(box.ap50[i]), 4), "map50_95": round(float(box.ap[i]), 4),
        })
    return rows


def evaluate(weights: str, data_yaml: str, split: str = "test", imgsz: int = 640, device: str = "0"):
    """Run validation and return {overall, per_class}. Nothing is estimated — all computed."""
    from ultralytics import YOLO
    m = YOLO(weights)
    mt = m.val(data=data_yaml, split=split, imgsz=imgsz, device=device, plots=True)
    return {
        "overall": {
            "map50": round(float(mt.box.map50), 4),
            "map50_95": round(float(mt.box.map), 4),
            "precision": round(float(mt.box.mp), 4),
            "recall": round(float(mt.box.mr), 4),
        },
        "per_class": per_class_table(mt.box, m.names),
    }
