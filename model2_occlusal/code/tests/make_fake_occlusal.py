#!/usr/bin/env python3
"""Synthetic smartphone occlusal photos + CVAT-style COCO + clinical ICDAS sheet (for smoke tests only)."""
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 1400, 1050


def draw_tooth(img, cx, cy, rx, ry, icdas, rng):
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (215, 225, 230), -1, cv2.LINE_AA)
    for _ in range(3):  # fissures
        a = rng.uniform(0, np.pi)
        dx, dy = int(np.cos(a) * rx * .6), int(np.sin(a) * ry * .6)
        cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), (150, 165, 175), 2, cv2.LINE_AA)
    if icdas in (1, 2):
        cv2.circle(img, (cx, cy), int(rx * .18), (170, 205, 225) if icdas == 1 else (110, 150, 185), -1, cv2.LINE_AA)
    elif icdas in (3, 4):
        cv2.circle(img, (cx, cy), int(rx * .28), (70, 100, 130), -1, cv2.LINE_AA)
    elif icdas in (5, 6):
        cv2.circle(img, (cx, cy), int(rx * .42), (25, 35, 45), -1, cv2.LINE_AA)
    return cv2.ellipse2Poly((cx, cy), (rx, ry), 0, 0, 360, 12)


def main(out):
    out = Path(out); (out / "photos").mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "tooth"}]}
    clin = []
    aid = 0
    for pi in range(1, 21):
        for view, q in (("upper", 1), ("lower", 4)):
            iid = len(coco["images"]) + 1
            name = f"P{pi:03d}_{view}.jpg"
            npr = np.random.default_rng(iid)
            img = np.full((H, W, 3), (120, 110, 190), np.uint8)
            img = cv2.add(img, npr.integers(0, 40, (H, W, 3), dtype=np.uint8))
            coco["images"].append({"id": iid, "file_name": name, "width": W, "height": H})
            for k, n in enumerate([4, 5, 6, 7]):
                fdi = f"{q}{n}"
                icdas = rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6])
                cx, cy = 250 + k * 300, 520 + rng.randint(-40, 40)
                rx, ry = (95, 85) if n <= 5 else (130, 115)
                poly = draw_tooth(img, cx, cy, rx, ry, icdas, rng)
                aid += 1
                coco["annotations"].append({"id": aid, "image_id": iid, "category_id": 1, "iscrowd": 0,
                                            "segmentation": [poly.reshape(-1).astype(float).tolist()],
                                            "bbox": [cx - rx, cy - ry, 2 * rx, 2 * ry], "area": float(np.pi * rx * ry),
                                            "attributes": {"fdi": fdi, "occluded": False}})
                clin.append({"patient_id": f"P{pi:03d}", "image": name, "fdi": fdi, "icdas": icdas, "examiner": "E1"})
            img = cv2.add(img, npr.integers(0, 25, (H, W, 3), dtype=np.uint8))
            if pi == 19 and view == "upper":           # saliva/flash glare spots -> quality flag
                for _ in range(900):
                    cv2.circle(img, (rng.randint(0, W - 1), rng.randint(0, H - 1)), 5, (255, 255, 255), -1)
            if pi == 20 and view == "lower":
                img = cv2.GaussianBlur(img, (41, 41), 0)   # should be excluded by the quality gate
            cv2.imwrite(str(out / "photos" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "cvat_coco.json").write_text(json.dumps(coco))
    with open(out / "clinical_icdas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clin[0].keys())); w.writeheader(); w.writerows(clin)
    print("fake occlusal data ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
