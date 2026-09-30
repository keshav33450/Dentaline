#!/usr/bin/env python3
"""
Build training data for the smartphone occlusal-caries study.

INPUTS (see docs/OCCLUSAL_ANNOTATION_GUIDE.md)
  --images    folder with the smartphone photos  (e.g. P001_upper.jpg, P001_lower.jpg)
  --coco      CVAT "COCO 1.0" export: one polygon per posterior tooth, attribute  fdi = 16/17/.../47
              (optional attribute icdas = 0-6 if the examiner coded it in CVAT)
  --clinical  clinical ICDAS sheet (the REFERENCE STANDARD), CSV or XLSX with columns
                  patient_id, image, fdi, icdas [, examiner]
              ICDAS from this sheet overrides any value typed in CVAT.

OUTPUTS  (<project>/data/occlusal/)
  processed/          pre-processed photos (Phase II)
  seg/                YOLOv8-seg dataset, 1 class 'tooth', patient-level train/val split
  crops/              one masked crop per tooth (input of the two classifiers)
  teeth.csv           one row per tooth: patient, fdi, tooth_type, icdas, severity, fold (patient-grouped 5-fold)
  image_quality.csv   quality metrics + reasons for every photo
  exclusions.csv      photos / teeth excluded and why

  python scripts/occ_prepare.py --images D:/study/photos --coco D:/study/cvat_export.json --clinical D:/study/icdas.xlsx
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.crop import polygon_to_mask, tooth_crop  # noqa: E402
from dentassist.occlusal.labels import SEVERITY, severity_from_icdas, tooth_type_from_fdi  # noqa: E402
from dentassist.occlusal.preprocess import QUALITY_DEFAULTS, preprocess, quality_check  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic"}


def norm_stem(name: str) -> str:
    return Path(str(name).replace("\\", "/")).stem.lower()


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() in (".xlsx", ".xls"):
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        head = [str(h).strip().lower() for h in rows[0]]
        return [{head[i]: ("" if v is None else str(v).strip()) for i, v in enumerate(r)} for r in rows[1:] if any(r)]
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k.strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def fdi_str(v) -> str | None:
    s = str(v).strip().split(".")[0]
    return s if len(s) == 2 and s.isdigit() else None


def patient_of(image_stem: str, clinical_pid: dict) -> str:
    return clinical_pid.get(image_stem) or image_stem.split("_")[0].split("-")[0]


def load_coco(path: Path) -> dict[str, list[dict]]:
    """image stem -> [{fdi, icdas, poly(Nx2)}]"""
    coco = json.loads(path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    imgs = {im["id"]: im for im in coco["images"]}
    out = defaultdict(list)
    for a in coco["annotations"]:
        im = imgs.get(a["image_id"])
        if im is None or not a.get("segmentation") or not isinstance(a["segmentation"], list):
            continue
        attrs = {str(k).lower(): v for k, v in (a.get("attributes") or {}).items()}
        fdi = fdi_str(attrs.get("fdi", "")) or fdi_str(cats.get(a["category_id"], ""))
        polys = [np.array(s, np.float32).reshape(-1, 2) for s in a["segmentation"] if len(s) >= 6]
        if not polys:
            continue
        poly = max(polys, key=lambda p: cv2.contourArea(p))
        out[norm_stem(im["file_name"])].append({"fdi": fdi, "icdas": attrs.get("icdas", ""), "poly": poly})
    return out


def grouped_folds(rows: list[dict], k: int, seed: int) -> None:
    from sklearn.model_selection import StratifiedGroupKFold
    labelled = [r for r in rows if r["severity"]]
    groups = [r["patient_id"] for r in labelled]
    k = max(2, min(k, len(set(groups))))
    y = [SEVERITY.index(r["severity"]) for r in labelled]
    try:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for f, (_, te) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            for i in te:
                labelled[i]["fold"] = f
    except ValueError:  # too few samples in a class -> plain grouped split
        pats = sorted(set(groups))
        random.Random(seed).shuffle(pats)
        pf = {p: i % k for i, p in enumerate(pats)}
        for r in labelled:
            r["fold"] = pf[r["patient_id"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--clinical", help="ICDAS reference sheet (csv/xlsx). Optional if icdas typed in CVAT.")
    ap.add_argument("--out", default=str(P.HOME / "data" / "occlusal"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2, help="patients held out for segmentation validation")
    ap.add_argument("--keep-bad-quality", action="store_true", help="do not exclude photos failing the quality gate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("processed", "crops", "seg/images/train", "seg/images/val", "seg/labels/train", "seg/labels/val"):
        (out / d).mkdir(parents=True, exist_ok=True)

    ann = load_coco(Path(args.coco))
    clinical, clinical_pid = {}, {}
    if args.clinical:
        for r in read_table(Path(args.clinical)):
            st, f = norm_stem(r.get("image", "")), fdi_str(r.get("fdi", ""))
            if st and f:
                clinical[(st, f)] = r.get("icdas", "")
            if st and r.get("patient_id"):
                clinical_pid[st] = r["patient_id"]
        print(f"[clinical] {len(clinical)} tooth records")

    files = {norm_stem(p.name): p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXT}
    print(f"[images] {len(files)} photos, {len(ann)} annotated")

    patients = sorted({patient_of(s, clinical_pid) for s in ann})
    rnd = random.Random(args.seed)
    rnd.shuffle(patients)
    n_val = max(1, round(len(patients) * args.val_frac)) if len(patients) > 1 else 0
    val_pat = set(patients[:n_val])

    teeth, quality, excl = [], [], []
    stats = Counter()
    for stem, objs in sorted(ann.items()):
        src = files.get(stem)
        if src is None:
            excl.append({"image": stem, "fdi": "", "reason": "photo file not found"})
            continue
        raw = cv2.imread(str(src))
        if raw is None:
            excl.append({"image": stem, "fdi": "", "reason": "cannot read photo (convert HEIC to JPG)"})
            continue
        q = quality_check(raw)
        quality.append({"image": src.name, "ok": q["ok"], "reasons": "; ".join(q["reasons"]), **q["metrics"]})
        if not q["ok"] and not args.keep_bad_quality:
            excl.append({"image": src.name, "fdi": "", "reason": "quality: " + "; ".join(q["reasons"])})
            stats["photos_excluded_quality"] += 1
            continue
        H0, W0 = raw.shape[:2]
        img = preprocess(raw)
        s = img.shape[1] / W0
        cv2.imwrite(str(out / "processed" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pid = patient_of(stem, clinical_pid)
        split = "val" if pid in val_pat else "train"
        h, w = img.shape[:2]
        seg_lines = []
        for o in objs:
            poly = o["poly"] * s
            seg_lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly))
            ttype = tooth_type_from_fdi(o["fdi"]) if o["fdi"] else None
            if ttype is None:
                excl.append({"image": src.name, "fdi": o["fdi"] or "?", "reason": "not a permanent premolar/molar FDI (or fdi missing)"})
                continue
            icdas = clinical.get((stem, o["fdi"]), o["icdas"]) if clinical else o["icdas"]
            sev = severity_from_icdas(icdas) if str(icdas) != "" else None
            crop = tooth_crop(img, polygon_to_mask(poly, img.shape))
            if crop is None:
                excl.append({"image": src.name, "fdi": o["fdi"], "reason": "polygon too small"})
                continue
            cpath = out / "crops" / f"{stem}_{o['fdi']}.jpg"
            cv2.imwrite(str(cpath), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            teeth.append({"crop": f"crops/{cpath.name}", "patient_id": pid, "image": src.name, "fdi": o["fdi"],
                          "tooth_type": ttype, "icdas": icdas, "severity": sev or "", "fold": -1, "seg_split": split})
            if sev is None:
                excl.append({"image": src.name, "fdi": o["fdi"], "reason": "no ICDAS code (used for segmentation/type only)"})
        (out / "seg/labels" / split / f"{stem}.txt").write_text("\n".join(seg_lines))
        cv2.imwrite(str(out / "seg/images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        stats[f"photos_{split}"] += 1

    grouped_folds(teeth, args.folds, args.seed)
    (out / "seg/data.yaml").write_text(f"path: {(out / 'seg').resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: tooth\n")

    def dump(name, rows):
        if rows:
            with open(out / name, "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)

    dump("teeth.csv", teeth)
    dump("image_quality.csv", quality)
    dump("exclusions.csv", excl)

    summary = {
        "patients": len({t['patient_id'] for t in teeth}), "teeth": len(teeth),
        "teeth_with_icdas": sum(bool(t["severity"]) for t in teeth),
        "tooth_type": dict(Counter(t["tooth_type"] for t in teeth)),
        "severity": {k: sum(t["severity"] == k for t in teeth) for k in SEVERITY},
        "folds": dict(sorted(Counter(t["fold"] for t in teeth if t["severity"]).items())),
        **stats, "excluded_rows": len(excl), "quality_thresholds": QUALITY_DEFAULTS,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    small = [k for k, v in summary["severity"].items() if v < 20]
    if small:
        print(f"\n  ! few examples for {small} - collect more of these (aim >= 40 teeth per class)")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
