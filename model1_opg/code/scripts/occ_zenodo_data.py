#!/usr/bin/env python3
"""
Zenodo smartphone caries dataset -> Model 2 training / validation data.

  "Annotated intraoral image dataset for dental caries detection" (Ahmed et al., Sci Data 2025)
  https://zenodo.org/records/14769743  - 6,313 smartphone photos (Samsung Galaxy A23), 5 views,
  decay boxes: 'D' = permanent tooth decay, 'd' = primary tooth decay.

Outputs (<project>/data/occlusal/zenodo/):
  det/            YOLO dataset, 1 class 'caries' (permanent 'D' only - study = permanent teeth), photo-grouped split
  photos.csv      every kept photo with its view (upper/lower/front/left/right), split, group id
  test_gt.json    held-out test photos + their 'D' boxes  (for external validation of the severity model)

Views: by default only occlusal views (upper/lower) are kept, matching the study protocol (--all-views to keep all).

  python scripts/occ_zenodo_data.py                     # download from Zenodo (~4.7 GB)
  python scripts/occ_zenodo_data.py --local D:/zenodo   # already downloaded/unzipped
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.preprocess import preprocess  # noqa: E402

RECORD = "14769743"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def download(dest: Path, record: str) -> Path:
    import requests
    dest.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60).json()
    for f in meta["files"]:
        key, url, size = f["key"], f["links"]["self"], f.get("size", 0)
        out = dest / key
        if out.exists() and out.stat().st_size == size:
            print(f"  have {key}")
            continue
        print(f"  downloading {key} ({size / 1e9:.2f} GB)", flush=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk)
    for z in dest.glob("*.zip"):
        marker = dest / f".{z.stem}.extracted"
        if not marker.exists():
            print(f"  extracting {z.name}", flush=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest / z.stem)
            marker.touch()
    return dest


def view_of(path: str) -> str:
    s = path.lower().replace("-", " ").replace("_", " ")
    if re.search(r"maxill|upper|\bmax\b|\bu occ", s):
        return "upper"
    if re.search(r"mandib|lower|\bmand\b|\bl occ", s):
        return "lower"
    if re.search(r"front|anterior|frontal", s):
        return "front"
    if re.search(r"\bleft\b|left lat|lateral left", s):
        return "left"
    if re.search(r"\bright\b|right lat|lateral right", s):
        return "right"
    return "unknown"


def cat_kind(name: str) -> str | None:
    n = name.strip()
    if n == "D" or re.search(r"perman", n, re.I):
        return "permanent"
    if n == "d" or re.search(r"primar|decid|milk", n, re.I):
        return "primary"
    return None


def group_of(stem: str) -> str:
    """patient/session id: the photos of one mouth must stay in the same split (5 views per person)."""
    m = re.search(r"(\d{2,})", stem)
    return m.group(1) if m else stem.lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", help="folder with the downloaded (and unzipped) dataset")
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--out", default=str(P.HOME / "data" / "occlusal" / "zenodo"))
    ap.add_argument("--all-views", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src = Path(a.local) if a.local else download(P.HOME / "data" / "zenodo_raw", a.record)
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    for s in ("train", "val", "test"):
        (out / "det/images" / s).mkdir(parents=True, exist_ok=True)
        (out / "det/labels" / s).mkdir(parents=True, exist_ok=True)

    # index images by basename (dataset ships several annotation formats for the same photos)
    img_index = defaultdict(list)
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXT and "__MACOSX" not in str(p):
            img_index[p.name.lower()].append(p)

    # COCO json files = the most self-describing format; one photo may appear in several -> dedupe
    photos: dict[str, dict] = {}
    n_json = 0
    for j in src.rglob("*.json"):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(d, dict) and "images" in d and "annotations" in d and "categories" in d):
            continue
        n_json += 1
        cats = {c["id"]: cat_kind(str(c["name"])) for c in d["categories"]}
        by_img = defaultdict(list)
        for an in d["annotations"]:
            by_img[an["image_id"]].append(an)
        for im in d["images"]:
            base = Path(im["file_name"]).name.lower()
            if base in photos:
                continue
            cands = img_index.get(base, [])
            if not cands:
                continue
            path = cands[0]
            boxes = [(cats.get(an["category_id"]), an["bbox"]) for an in by_img.get(im["id"], [])]
            photos[base] = {"path": path, "view": view_of(str(path.relative_to(src)) + " " + str(j.relative_to(src))),
                            "boxes": boxes}
    if not photos:
        sys.exit(f"No COCO annotations matched images under {src} (found {n_json} COCO files).")
    views = Counter(p["view"] for p in photos.values())
    print(f"[photos] {len(photos)} with annotations, views: {dict(views)}")
    keep_views = None if a.all_views or views.get("upper", 0) + views.get("lower", 0) == 0 else {"upper", "lower"}
    if keep_views is None and not a.all_views:
        print("  ! could not detect occlusal views from folder/file names -> keeping ALL views")

    groups = sorted({group_of(Path(k).stem) for k in photos})
    random.Random(a.seed).shuffle(groups)
    nt, nv = round(len(groups) * a.test_frac), round(len(groups) * a.val_frac)
    split_of = {g: ("test" if i < nt else "val" if i < nt + nv else "train") for i, g in enumerate(groups)}

    rows, test_gt, stats = [], [], Counter()
    for base, ph in sorted(photos.items()):
        if keep_views and ph["view"] not in keep_views:
            continue
        raw = cv2.imread(str(ph["path"]))
        if raw is None:
            continue
        H0, W0 = raw.shape[:2]
        img = preprocess(raw)
        s = img.shape[1] / W0
        h, w = img.shape[:2]
        grp = group_of(Path(base).stem)
        split = split_of[grp]
        stem = f"z_{Path(base).stem}"[:90]
        lines, gt = [], []
        for kind, (x, y, bw, bh) in ph["boxes"]:
            stats[kind or "other"] += 1
            if kind != "permanent":
                continue
            x0, y0, x1, y1 = x * s, y * s, (x + bw) * s, (y + bh) * s
            lines.append(f"0 {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} {(x1 - x0) / w:.5f} {(y1 - y0) / h:.5f}")
            gt.append([round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
        cv2.imwrite(str(out / "det/images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        (out / "det/labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        rows.append({"image": f"det/images/{split}/{stem}.jpg", "original": str(ph["path"]), "view": ph["view"],
                     "group": grp, "split": split, "permanent_caries_boxes": len(gt)})
        if split == "test":
            test_gt.append({"image": f"det/images/test/{stem}.jpg", "view": ph["view"], "caries_boxes": gt})
        stats[f"photos_{split}"] += 1

    (out / "det/data.yaml").write_text(f"path: {(out / 'det').resolve()}\ntrain: images/train\nval: images/val\n"
                                       "test: images/test\nnc: 1\nnames:\n  0: caries\n")
    with open(out / "photos.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    (out / "test_gt.json").write_text(json.dumps(test_gt))
    summary = {"photos_kept": len(rows), "views_kept": sorted(keep_views) if keep_views else "all",
               "boxes": {"permanent_caries(D)": stats["permanent"], "primary_excluded(d)": stats["primary"],
                         "unrecognised": stats["other"]},
               "split_photos": {k: stats[f"photos_{k}"] for k in ("train", "val", "test")},
               "patient_groups": len(groups), "licence": "see Zenodo record (CC BY 4.0 listed for v2)",
               "source": f"https://zenodo.org/records/{a.record}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
