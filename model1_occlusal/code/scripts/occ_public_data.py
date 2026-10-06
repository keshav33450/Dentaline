#!/usr/bin/env python3
"""
Real public data for Model 2 (before the study photos exist).

Downloads a Roboflow Universe caries dataset whose classes are ICDAS-based severity
(default: "Caries Classification ICDAS II", classes Healthy / Initial / Moderate / Extensive, CC BY 4.0)
and converts it to the Model 2 training format:

  data/occlusal/crops/ + teeth.csv   -> severity classifier (EfficientNet-B0 / MobileNetV3), grouped 5-fold CV
  data/occlusal/det/                 -> YOLOv8 detector that outputs severity boxes directly

Class mapping:  healthy/sound/ICDAS 0 -> No Caries | initial/ICDAS 1-2 -> Mild |
                moderate/ICDAS 3-4 -> Moderate    | extensive/advanced/ICDAS 5-6 -> Advanced

Leakage guard: Roboflow exports contain augmented copies of the same photo (name_jpg.rf.<hash>.jpg) spread
over train/valid/test. Copies are collapsed to ONE per original photo, and folds are grouped by original photo.

  python scripts/occ_public_data.py --api-key XXXXX
  python scripts/occ_public_data.py --local path/to/unzipped_yolov8_export     (already downloaded)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.crop import tooth_crop  # noqa: E402
from dentassist.occlusal.labels import SEVERITY, severity_from_icdas  # noqa: E402
from dentassist.occlusal.preprocess import preprocess  # noqa: E402
from occ_prepare import grouped_folds  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def map_class(name: str) -> str | None:
    n = name.strip().lower()
    # names that START with an ICDAS code, e.g. "-3-Localized-Enamel-Breakdown", "4 - dentin shadow", "ICDAS_5"
    lead = re.match(r"^\W*(?:icdas|code)?\W*([0-6])(?!\d)", n)
    if lead:
        return severity_from_icdas(lead.group(1))
    m = re.search(r"(?:icdas|code|^)\s*[_-]?\s*([0-6])\b", n)
    if m and ("icdas" in n or "code" in n or n.isdigit()):
        return severity_from_icdas(m.group(1))
    if any(k in n for k in ("healthy", "sound", "no caries", "no_caries", "normal")):
        return "no_caries"
    if any(k in n for k in ("initial", "early", "mild", "enamel", "white spot")):
        return "mild"
    if "moderate" in n:
        return "moderate"
    if any(k in n for k in ("extensive", "advanced", "severe", "cavit", "deep")):
        return "advanced"
    return None


def download(api_key, workspace, project, version, dest: Path) -> Path:
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    if version is None:
        vs = proj.versions()
        version = max(int(getattr(v, "version", str(v)).split("/")[-1]) for v in vs)
    print(f"[roboflow] {workspace}/{project} v{version}")
    ds = proj.version(version).download("yolov8", location=str(dest), overwrite=True)
    return Path(ds.location)


def original_stem(p: Path) -> str:
    return re.split(r"_(?:jpg|jpeg|png|bmp)\.rf\.", p.name, flags=re.I)[0].lower() if ".rf." in p.name else p.stem.lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key")
    ap.add_argument("--workspace", default="code-geass")
    ap.add_argument("--project", default="caries-classification-icdas-ii")
    ap.add_argument("--version", type=int)
    ap.add_argument("--local", help="already-downloaded YOLOv8 export folder (contains data.yaml)")
    ap.add_argument("--out", default=str(P.HOME / "data" / "occlusal"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    src = Path(a.local) if a.local else download(a.api_key, a.workspace, a.project, a.version, P.HOME / "data" / "public_raw")
    import yaml
    meta = yaml.safe_load((src / "data.yaml").read_text())
    names = meta["names"] if isinstance(meta["names"], list) else [meta["names"][k] for k in sorted(meta["names"])]
    cmap = {i: map_class(n) for i, n in enumerate(names)}
    icdas_code = {i: (m.group(1) if (m := re.match(r"^\W*(?:icdas|code)?\W*([0-6])(?!\d)", n.strip().lower())) else "")
                  for i, n in enumerate(names)}
    print("[classes]", {names[i]: cmap[i] for i in cmap})
    if not any(cmap.values()):
        sys.exit("Could not map any class name to a severity - check the dataset.")

    # collect images, keep ONE copy per original photo
    seen, items = set(), []
    for img in sorted(src.rglob("*")):
        if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
            continue
        stem = original_stem(img)
        if stem in seen:
            continue
        parts = list(img.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"   # last 'images' dir -> 'labels'
        lab = Path(*parts).with_suffix(".txt")
        if lab.exists():
            seen.add(stem)
            items.append((stem, img, lab))
    print(f"[images] {len(items)} unique photos (augmented duplicates removed)")

    for d in ("crops", "det", "samples"):
        shutil.rmtree(out / d, ignore_errors=True)
    for d in ("crops", "det/images/train", "det/images/val", "det/labels/train", "det/labels/val", "samples"):
        (out / d).mkdir(parents=True, exist_ok=True)

    stems = [s for s, _, _ in items]
    random.Random(a.seed).shuffle(stems)
    val = set(stems[:max(1, round(0.2 * len(stems)))])
    teeth, stats, skipped = [], Counter(), Counter()
    for k, (stem, img_p, lab_p) in enumerate(items):
        raw = cv2.imread(str(img_p))
        if raw is None:
            continue
        img = preprocess(raw)
        h, w = img.shape[:2]
        split = "val" if stem in val else "train"
        det_lines = []
        for j, line in enumerate(lab_p.read_text().split("\n")):
            v = line.split()
            if len(v) < 5:
                continue
            c = int(v[0]); sev = cmap.get(c)
            if sev is None:
                skipped[names[c]] += 1
                continue
            coords = np.array(list(map(float, v[1:])), np.float32)
            if len(coords) == 4:            # box: cx cy bw bh
                cx, cy, bw, bh = coords
                x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            else:                           # polygon
                xs, ys = coords[0::2] * w, coords[1::2] * h
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            mask = np.zeros((h, w), np.uint8)
            mask[int(y0):int(y1), int(x0):int(x1)] = 255
            crop = tooth_crop(img, mask)
            if crop is None:
                continue
            name = f"{stem[:60]}_{j}.jpg"
            cv2.imwrite(str(out / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            teeth.append({"crop": f"crops/{name}", "patient_id": stem, "image": img_p.name, "fdi": "",
                          "tooth_type": "", "icdas": icdas_code[c], "severity": sev, "fold": -1, "seg_split": split})
            det_lines.append(f"{SEVERITY.index(sev)} {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} {(x1 - x0) / w:.5f} {(y1 - y0) / h:.5f}")
            stats[sev] += 1
        cv2.imwrite(str(out / "det/images" / split / f"{stem[:80]}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        (out / "det/labels" / split / f"{stem[:80]}.txt").write_text("\n".join(det_lines))
        if k < 8:
            cv2.imwrite(str(out / "samples" / f"sample_{k}.jpg"), img)

    grouped_folds(teeth, a.folds, a.seed)
    (out / "det/data.yaml").write_text(f"path: {(out / 'det').resolve()}\ntrain: images/train\nval: images/val\nnc: 4\nnames:\n"
                                       + "".join(f"  {i}: {c}\n" for i, c in enumerate(SEVERITY)))
    with open(out / "teeth.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(teeth[0].keys())); wr.writeheader(); wr.writerows(teeth)
    summary = {"source": str(src), "classes": {names[i]: cmap[i] for i in cmap}, "photos": len(items),
               "regions": len(teeth), "severity": {s: stats[s] for s in SEVERITY}, "skipped_classes": dict(skipped),
               "folds": dict(sorted(Counter(t["fold"] for t in teeth).items())), "mode": "public"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
