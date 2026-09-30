#!/usr/bin/env python3
"""
Pre-label occlusal photos so annotators only CORRECT polygons instead of drawing them.

Engines
  segmentanytooth  (default) open-source tooth segmentation + FDI numbering for occlusal photos
                   https://github.com/thangngoc89/SegmentAnyTooth  (weights: free research licence by e-mail)
                   needs the view per photo: filename containing 'upper'/'maxill' or 'lower'/'mandib', or --view
  yolo             our own trained occlusal segmenter (after the first batch of patients is labelled);
                   gives polygons without FDI - annotator fills the fdi attribute

Output: COCO json with polygons + attribute 'fdi' -> import into CVAT as "COCO 1.0".

  python scripts/occ_autolabel.py --images D:/study/photos --out prelabels.json --weights D:/SegmentAnyTooth/weights
  python scripts/occ_autolabel.py --images D:/study/photos --out prelabels.json --engine yolo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.labels import tooth_type_from_fdi  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def view_of(name: str, default: str | None) -> str | None:
    n = name.lower()
    if "upper" in n or "maxill" in n or "_u." in n or n.endswith("_u"):
        return "upper"
    if "lower" in n or "mandib" in n or "_l." in n or n.endswith("_l"):
        return "lower"
    return default


def mask_to_poly(m: np.ndarray, eps_frac: float = 0.004):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    c = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
    return c.reshape(-1).astype(float).tolist() if len(c) >= 3 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="prelabels_coco.json")
    ap.add_argument("--engine", choices=["segmentanytooth", "yolo"], default="segmentanytooth")
    ap.add_argument("--weights", help="SegmentAnyTooth weight_dir, or YOLO .pt (default <project>/weights/occlusal_seg.pt)")
    ap.add_argument("--view", choices=["upper", "lower"], help="force view for all photos")
    ap.add_argument("--conf", type=float, default=0.4)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG_EXT)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "tooth", "supercategory": ""}]}
    aid = 0
    if a.engine == "segmentanytooth":
        from segmentanytooth import predict
    else:
        from ultralytics import YOLO
        model = YOLO(a.weights or str(P.WEIGHTS / "occlusal_seg.pt"))

    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  skip unreadable {p.name}")
            continue
        H, W = img.shape[:2]
        coco["images"].append({"id": i, "file_name": p.name, "width": W, "height": H})
        polys = []
        if a.engine == "segmentanytooth":
            v = view_of(p.stem, a.view)
            if v is None:
                print(f"  skip {p.name}: cannot tell upper/lower (rename or pass --view)")
                continue
            fdi_mask = np.asarray(predict(image_path=str(p), view=v, weight_dir=a.weights))
            if fdi_mask.shape[:2] != (H, W):
                fdi_mask = cv2.resize(fdi_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            for f in np.unique(fdi_mask):
                if f == 0 or tooth_type_from_fdi(int(f)) is None:
                    continue           # keep permanent premolars / molars only
                poly = mask_to_poly(fdi_mask == f)
                if poly:
                    polys.append((str(int(f)), poly))
        else:
            r = model.predict(img, conf=a.conf, verbose=False, retina_masks=True)[0]
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy():
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    poly = mask_to_poly(m)
                    if poly:
                        polys.append(("", poly))
        for fdi, poly in polys:
            aid += 1
            xs, ys = poly[0::2], poly[1::2]
            coco["annotations"].append({
                "id": aid, "image_id": i, "category_id": 1, "segmentation": [poly], "iscrowd": 0,
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "area": float(cv2.contourArea(np.array(poly, np.float32).reshape(-1, 2))),
                "attributes": {"fdi": fdi, "icdas": "", "occluded": False}})
        print(f"  {p.name}: {len(polys)} posterior teeth")
    Path(a.out).write_text(json.dumps(coco))
    print(f"[done] {a.out}  ({aid} polygons) -> CVAT: Actions > Upload annotations > COCO 1.0")


if __name__ == "__main__":
    main()
