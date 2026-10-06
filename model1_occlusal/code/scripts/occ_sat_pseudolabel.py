#!/usr/bin/env python3
"""
Build a premolar/molar tooth-segmentation dataset by running SegmentAnyTooth on the Zenodo
smartphone OCCLUSAL photos (output of occ_zenodo_data.py). The FDI number of each tooth gives its type:
    x4, x5 -> premolar      x6, x7, x8 -> molar      (primary teeth 51-85 and anterior teeth ignored)

Outputs (<project>/data/occlusal/zenodo/):
  toothseg/   YOLOv8-seg dataset, classes 0 premolar / 1 molar, same patient-grouped split as the caries data
  typeset/    tooth crops + teeth.csv  -> premolar/molar classifier (EfficientNet-B0 vs MobileNetV3)
  pseudolabel_qc/  a few overlays to eyeball the automatic labels

These are AUTOMATIC labels (pseudo-labels). Report them as such; the KGMU photos annotated by the team
remain the reference for the study's tooth-type accuracy.

  python scripts/occ_sat_pseudolabel.py --sat-repo /path/SegmentAnyTooth --sat-weights /path/weights
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zlib
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.crop import tooth_crop  # noqa: E402
from dentassist.occlusal.labels import tooth_type_from_fdi  # noqa: E402

TYPES = ["premolar", "molar"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zenodo", default=str(P.HOME / "data" / "occlusal" / "zenodo"))
    ap.add_argument("--sat-repo", required=True)
    ap.add_argument("--sat-weights", required=True)
    ap.add_argument("--min-teeth", type=int, default=2, help="skip photos where fewer posterior teeth were found")
    ap.add_argument("--min-area", type=float, default=0.002, help="min tooth area as fraction of the photo")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from dentassist.occlusal.sat import SegmentAnyTooth
    sat = SegmentAnyTooth(a.sat_repo, a.sat_weights)
    z = Path(a.zenodo)
    rows = [r for r in csv.DictReader(open(z / "photos.csv", encoding="utf-8")) if r["view"] in ("upper", "lower")]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[sat] {len(rows)} occlusal photos to label")

    seg, typ, qc = z / "toothseg", z / "typeset", z / "pseudolabel_qc"
    for d in (seg, typ, qc):
        shutil.rmtree(d, ignore_errors=True)
    for s in ("train", "val", "test"):
        (seg / "images" / s).mkdir(parents=True, exist_ok=True)
        (seg / "labels" / s).mkdir(parents=True, exist_ok=True)
    (typ / "crops").mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)

    teeth, stats = [], Counter()
    for k, r in enumerate(rows):
        img = cv2.imread(str(z / r["image"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        fdi_mask = sat.predict(img, r["view"])
        lines, found = [], []
        for f in np.unique(fdi_mask):
            t = tooth_type_from_fdi(int(f)) if f else None
            if t is None:
                continue
            m = (fdi_mask == f).astype(np.uint8)
            if m.sum() < a.min_area * h * w:
                continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(c) < 3:
                continue
            found.append((int(f), t, m, c))
        if len(found) < a.min_teeth:
            stats["photos_skipped_few_teeth"] += 1
            continue
        stem = Path(r["image"]).stem
        for f, t, m, c in found:
            lines.append(f"{TYPES.index(t)} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
            crop = tooth_crop(img, m * 255)
            if crop is not None:
                name = f"{stem}_{f}.jpg"
                cv2.imwrite(str(typ / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                fold = zlib.crc32(r["group"].encode()) % 5
                teeth.append({"crop": f"crops/{name}", "patient_id": r["group"], "image": Path(r["image"]).name,
                              "fdi": f, "tooth_type": t, "icdas": "", "severity": "", "fold": fold,
                              "seg_split": r["split"]})
            stats[t] += 1
        shutil.copy2(z / r["image"], seg / "images" / r["split"] / f"{stem}.jpg")
        (seg / "labels" / r["split"] / f"{stem}.txt").write_text("\n".join(lines))
        stats[f"photos_{r['split']}"] += 1
        if stats["qc"] < 12:
            vis = img.copy()
            for f, t, m, c in found:
                col = (0, 200, 255) if t == "premolar" else (255, 120, 0)
                cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, col, 3)
                x, y = c.min(0)
                cv2.putText(vis, f"{f} {t[0].upper()}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            cv2.imwrite(str(qc / f"{stem}.jpg"), vis)
            stats["qc"] += 1
        if k % 100 == 0:
            print(f"  {k}/{len(rows)}  {dict(stats)}", flush=True)

    (seg / "data.yaml").write_text(f"path: {seg.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                   "nc: 2\nnames:\n  0: premolar\n  1: molar\n")
    if teeth:
        with open(typ / "teeth.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(teeth[0].keys())); wr.writeheader(); wr.writerows(teeth)
    stats.pop("qc", None)
    summary = {**stats, "teeth": len(teeth), "note": "pseudo-labels from SegmentAnyTooth (FDI -> premolar/molar)"}
    (z / "pseudolabel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
