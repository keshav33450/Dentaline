"""Shared constants + IOP-Compass helpers for the tooth-AI pipeline.

Verified against:
  - SegmentAnyTooth source (segmentanytooth.py, sam.py) - MIT
  - Teeth-YOLO README (IOP-Compass JSON format) - github.com/JohnTitor-elpsykongroo/Teeth-YOLO
  - IOP-Compass dataset page (ditto.ing.unimore.it/iop-compass) - CC BY-NC-SA 4.0
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ---- views -----------------------------------------------------------------
# IOP-Compass file tokens -> our canonical 5 views (matches SegmentAnyTooth view names).
FILE_VIEW = {"center": "front", "upper": "upper", "down": "lower", "left": "left", "right": "right"}
VIEW_CLASSES = ["front", "upper", "lower", "left", "right"]      # classifier output order
VIEW_INDEX = {v: i for i, v in enumerate(VIEW_CLASSES)}

# ---- FDI -------------------------------------------------------------------
# 32 permanent teeth. Detector classes are FDI numbers so numbering comes from the detector,
# exactly like SegmentAnyTooth (class name last-2-digits = FDI).
FDI_NUMBERS = [q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)]   # 11..18,21..28,31..38,41..48
FDI_INDEX = {f: i for i, f in enumerate(FDI_NUMBERS)}


def tooth_type(fdi: int) -> str:
    """FDI -> incisor / canine / premolar / molar (last digit)."""
    t = fdi % 10
    return {1: "incisor", 2: "incisor", 3: "canine", 4: "premolar",
            5: "premolar", 6: "molar", 7: "molar", 8: "molar"}[t]


TYPE_CLASSES = ["incisor", "canine", "premolar", "molar"]


def view_of_filename(name: str) -> str | None:
    """IOP_Center_1.png -> 'front', etc. Robust to case / separators."""
    s = name.lower()
    for tok, view in FILE_VIEW.items():
        if re.search(rf"(^|[_\-]){tok}([_\-]|\d|\.|$)", s):
            return view
    return None


def patient_of_path(p: Path) -> str:
    """Patient id from '.../Patient_37/IOP_Upper_37.png' -> 'Patient_37' (keeps a whole mouth in one split)."""
    for part in p.parts[::-1]:
        if re.match(r"(?i)patient[_\-]?\d+", part):
            return part
    m = re.search(r"(\d+)", p.stem)          # fallback: trailing number in the file name
    return f"pt_{m.group(1)}" if m else p.stem


def parse_iop_json(jpath: Path) -> list[dict]:
    """-> [{fdi:int, bbox:[x0,y0,x1,y1], polygon:[[x,y],...]}] in ORIGINAL pixel coords.

    Tolerant to the two field spellings seen in the wild (FDI_NUM / fdi, contours / contour)."""
    d = json.loads(Path(jpath).read_text(encoding="utf-8-sig"))
    teeth = d.get("teeth") or d.get("annotations") or []
    out = []
    for t in teeth:
        fdi = t.get("FDI_NUM", t.get("fdi", t.get("fdi_num")))
        if fdi is None:
            continue
        try:
            fdi = int(fdi)
        except (TypeError, ValueError):
            continue
        if fdi not in FDI_INDEX:
            continue
        poly = t.get("contours") or t.get("contour") or t.get("segmentation")
        if poly and isinstance(poly[0], (list, tuple)) and poly and isinstance(poly[0][0], (list, tuple)):
            poly = poly[0]                    # contours is [[[x,y],...]]; take the outer ring
        pts = [[float(x), float(y)] for x, y in poly] if poly else []
        if "x_min" in t:
            bbox = [float(t["x_min"]), float(t["y_min"]), float(t["x_max"]), float(t["y_max"])]
        elif pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        else:
            continue
        out.append({"fdi": fdi, "bbox": bbox, "polygon": pts})
    return out


def find_pairs(root: Path) -> list[tuple[Path, Path, str, str]]:
    """Walk an unpacked IOP-Compass root -> [(image, json, view, patient)]."""
    root = Path(root)
    pairs = []
    for img in root.rglob("*"):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        j = img.with_suffix(".json")
        if not j.exists():
            continue
        view = view_of_filename(img.name)
        if view is None:
            continue
        pairs.append((img, j, view, patient_of_path(img)))
    return pairs
