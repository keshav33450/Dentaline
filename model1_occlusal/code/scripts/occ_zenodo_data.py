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


def _get_json(url):
    import requests
    r = requests.get(url, timeout=60, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def _file_list(record: str):
    """-> [(key, size, url)] for the record (tries the latest version first; handles old + new Zenodo APIs)."""
    tried = []
    ids = []
    try:
        latest = _get_json(f"https://zenodo.org/api/records/{record}/versions/latest")
        ids.append(str(latest.get("id") or latest.get("recid") or ""))
    except Exception as e:
        tried.append(f"versions/latest: {e}")
    ids += [record, "14827784"]
    for rid in [i for i in dict.fromkeys(ids) if i]:
        for url in (f"https://zenodo.org/api/records/{rid}/files", f"https://zenodo.org/api/records/{rid}"):
            try:
                d = _get_json(url)
            except Exception as e:
                tried.append(f"{url}: {e}"); continue
            ent = d.get("entries") if isinstance(d, dict) else None
            if ent is None and isinstance(d, dict):
                f = d.get("files")
                ent = f.get("entries") if isinstance(f, dict) else f
            if isinstance(ent, dict):
                ent = list(ent.values())
            out = []
            for f in ent or []:
                key = f.get("key") or f.get("filename")
                links = f.get("links") or {}
                link = links.get("content") or links.get("self") or links.get("download") or \
                    f"https://zenodo.org/records/{rid}/files/{key}?download=1"
                if key:
                    out.append((key, int(f.get("size") or f.get("filesize") or 0), link))
            if out:
                print(f"  Zenodo record {rid}: {len(out)} files", flush=True)
                return out
            tried.append(f"{url}: no files listed (keys: {list(d)[:12] if isinstance(d, dict) else type(d)})")
    raise SystemExit("Could not list the Zenodo files:\n  " + "\n  ".join(tried))


def _extract(arc: Path, dest: Path):
    import subprocess
    n = arc.name.lower()
    if n.endswith(".zip"):
        with zipfile.ZipFile(arc) as zf:
            zf.extractall(dest)
    elif n.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        import tarfile
        with tarfile.open(arc) as tf:
            tf.extractall(dest)
    else:                                   # .rar / .7z
        for tool in (["7z", "x", "-y", f"-o{dest}", str(arc)], ["unrar", "x", "-o+", str(arc), str(dest) + "/"],
                     ["bsdtar", "-xf", str(arc), "-C", str(dest)]):
            if shutil.which(tool[0]):
                dest.mkdir(parents=True, exist_ok=True)
                if subprocess.run(tool, stdout=subprocess.DEVNULL).returncode == 0:
                    return
        if n.endswith(".7z"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py7zr"], check=True)
            import py7zr
            with py7zr.SevenZipFile(arc) as z:
                z.extractall(dest)
            return
        raise SystemExit(f"cannot extract {arc.name} (no 7z/unrar/bsdtar)")


ARCHIVES = (".zip", ".rar", ".7z", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz")


def extract_all(root: Path):
    """extract every archive under root, including archives inside archives (up to 3 levels)."""
    for _ in range(3):
        todo = [a for a in root.rglob("*") if a.is_file() and a.name.lower().endswith(ARCHIVES)
                and not (a.parent / f".{a.name}.extracted").exists()]
        if not todo:
            return
        for a in todo:
            print(f"  extracting {a.relative_to(root)}", flush=True)
            _extract(a, a.parent / a.name.split(".")[0])
            (a.parent / f".{a.name}.extracted").touch()


def download(dest: Path, record: str) -> Path:
    import requests
    dest.mkdir(parents=True, exist_ok=True)
    for key, size, url in _file_list(record):
        out = dest / key
        if out.exists() and (not size or out.stat().st_size == size):
            print(f"  have {key}")
            continue
        print(f"  downloading {key} ({size / 1e9:.2f} GB)", flush=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk)
    extract_all(dest)
    return dest


def find_attached(root="/kaggle/input") -> Path | None:
    """a Kaggle dataset the user attached with the Zenodo files (zips or unzipped)."""
    r = Path(root)
    if not r.is_dir():
        return None
    for d in sorted(r.iterdir()):
        names = " ".join(p.name.lower() for p in list(d.rglob("*"))[:5000])
        if re.search(r"w_?/?o[-_ ]?r|cheek retractor|benchmark|labelme|maxillary", names):
            return d
    return None


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


def _norm_boxes_labelme(d):
    out = []
    for sh in d.get("shapes", []):
        pts = sh.get("points") or []
        if len(pts) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append((cat_kind(str(sh.get("label", ""))), [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]))
    return out


def collect(roots):
    """-> ({basename: {path, view, boxes[(kind, [x,y,w,h])]}}, Counter of annotation formats used).
    Priority per photo: LabelMe (original) > COCO > Pascal VOC > YOLO."""
    img_index = defaultdict(list)
    files = []
    for r in roots:
        for x in r.rglob("*"):
            if not x.is_file() or "__MACOSX" in str(x):
                continue
            if x.suffix.lower() in IMG_EXT:
                img_index[x.name.lower()].append(x)
            elif x.suffix.lower() in (".json", ".xml", ".txt"):
                files.append(x)
    stem_index = defaultdict(list)
    for k, v in img_index.items():
        stem_index[Path(k).stem].extend(v)
    photos, found = {}, Counter()

    def put(base, boxes, ann_path, fmt):
        base = base.lower()
        cands = img_index.get(base) or stem_index.get(Path(base).stem, [])
        if not cands or base in photos:
            return
        path = cands[0]
        photos[Path(path).name.lower()] = {"path": path, "view": view_of(f"{path} {ann_path}"), "boxes": boxes}
        found[fmt] += 1

    jsons = [f for f in files if f.suffix.lower() == ".json"]
    for fmt in ("labelme", "coco"):
        for j in jsons:
            try:
                d = json.loads(j.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            if fmt == "labelme" and "shapes" in d:
                put(Path(str(d.get("imagePath") or j.stem + ".jpg").replace("\\", "/")).name, _norm_boxes_labelme(d), j, fmt)
            elif fmt == "coco" and "images" in d and "annotations" in d:
                cats = {c["id"]: cat_kind(str(c["name"])) for c in d.get("categories", [])}
                by_img = defaultdict(list)
                for an in d["annotations"]:
                    by_img[an.get("image_id")].append(an)
                for im in d["images"]:
                    put(Path(str(im["file_name"]).replace("\\", "/")).name,
                        [(cats.get(an.get("category_id")), an["bbox"]) for an in by_img.get(im["id"], [])], j, fmt)
    import xml.etree.ElementTree as ET
    for x in (f for f in files if f.suffix.lower() == ".xml"):
        try:
            t = ET.parse(x).getroot()
        except Exception:
            continue
        if t.tag != "annotation":
            continue
        boxes = []
        for o in t.iter("object"):
            bb = o.find("bndbox")
            if bb is None:
                continue
            v = [float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append((cat_kind(o.findtext("name", "")), [v[0], v[1], v[2] - v[0], v[3] - v[1]]))
        put(t.findtext("filename") or x.stem + ".jpg", boxes, x, "pascal_voc")
    # YOLO: needs a class list (classes.txt / obj.names / data.yaml) next to or above the labels
    txts = [f for f in files if f.suffix.lower() == ".txt" and f.name.lower() not in ("classes.txt", "obj.names")]
    for t in txts:
        names = None
        for d in (t.parent, t.parent.parent, t.parent.parent.parent):
            for n in ("classes.txt", "obj.names"):
                if (d / n).exists():
                    names = [l.strip() for l in (d / n).read_text().splitlines() if l.strip()]
            for n in ("data.yaml", "dataset.yaml"):
                if names is None and (d / n).exists():
                    m = re.search(r"names:\s*\[([^\]]*)\]", (d / n).read_text())
                    if m:
                        names = [v.strip().strip("'\"") for v in m.group(1).split(",")]
            if names:
                break
        if not names:
            continue
        cands = stem_index.get(t.stem.lower(), [])
        if not cands:
            continue
        im = cv2.imread(str(cands[0]))
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = []
        for line in t.read_text().splitlines():
            v = line.split()
            if len(v) >= 5 and v[0].isdigit() and int(v[0]) < len(names):
                cx, cy, bw, bh = (float(z) for z in v[1:5])
                boxes.append((cat_kind(names[int(v[0])]), [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]))
        put(cands[0].name, boxes, t, "yolo")
    return photos, found


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

    raw = P.HOME / "data" / "zenodo_raw"
    local = Path(a.local) if a.local else find_attached()
    if local:
        print(f"[zenodo] using local/attached copy: {local}")
        roots = [local]
        arcs = [x for x in local.rglob("*") if x.is_file() and x.name.lower().endswith(ARCHIVES)]
        if arcs:                                # attached folders are read-only -> extract next to the project
            raw.mkdir(parents=True, exist_ok=True)
            for x in arcs:
                t = raw / x.name
                if not t.exists():
                    t.symlink_to(x)
            extract_all(raw)
            roots.append(raw)
    else:
        print("[zenodo] downloading from zenodo.org", flush=True)
        roots = [download(raw, a.record)]
    photos, found = collect(roots)
    if not photos:
        ext = Counter(x.suffix.lower() for r in roots for x in r.rglob("*") if x.is_file())
        sys.exit(f"No annotated photos found under {[str(r) for r in roots]}.\n"
                 f"  annotation files seen: {dict(found)}\n  file types: {dict(ext.most_common(15))}")
    print(f"[annotations] used: {dict(found)}")
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    for s_ in ("train", "val", "test"):
        (out / "det/images" / s_).mkdir(parents=True, exist_ok=True)
        (out / "det/labels" / s_).mkdir(parents=True, exist_ok=True)
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
