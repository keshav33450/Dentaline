#!/usr/bin/env python3
"""
Download DENTEX (MICCAI 2023) and convert it into two YOLO datasets:

  teeth/     32 FDI classes (11..48)            <- quadrant-enumeration set (all teeth labeled)
  findings/  caries | deep_caries | periapical_lesion | impacted
                                                <- quadrant-enumeration-diagnosis set

Also writes findings_gt_test.json (box + diagnosis + FDI per abnormal tooth) for the
end-to-end "right tooth + right diagnosis" evaluation.

Usage
  python scripts/prepare_data.py --raw /tmp/dentex_raw --out /tmp/dentex_yolo
  python scripts/prepare_data.py --local-zips a.zip b.zip --extra-json v.json --out ...   (offline)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.fdi import FDI_TEETH, FDI_TO_IDX, FINDINGS, fdi_from_parts, normalize_finding  # noqa: E402

HF_REPO = "ibrahimhamamci/DENTEX"
HF_FILES = ["DENTEX/training_data.zip", "DENTEX/validation_data.zip",
            "DENTEX/test_data.zip", "DENTEX/validation_triple.json"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- download
def download(raw: Path) -> tuple[list[Path], list[Path]]:
    from huggingface_hub import hf_hub_download
    raw.mkdir(parents=True, exist_ok=True)
    zips, jsons = [], []
    for f in HF_FILES:
        print(f"[download] {f}", flush=True)
        try:
            p = Path(hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir=str(raw)))
        except Exception as e:  # test labels may be missing in some mirrors — keep going
            print(f"  ! skipped ({e.__class__.__name__}: {e})")
            continue
        (zips if p.suffix == ".zip" else jsons).append(p)
    return zips, jsons


# --------------------------------------------------------------------------- discovery
class Source:
    """One COCO-style annotation file plus where to find its images."""

    def __init__(self, name: str, coco: dict, zip_path: Path | None, json_member: str | None):
        self.name, self.coco, self.zip_path, self.json_member = name, coco, zip_path, json_member
        anns = coco.get("annotations", [])
        keys = set().union(*(a.keys() for a in anns[:200])) if anns else set()
        if "category_id_3" in keys:
            self.kind = "diagnosis"
        elif "category_id_2" in keys:
            self.kind = "enumeration"
        elif "category_id_1" in keys:
            self.kind = "quadrant"
        else:
            self.kind = "unknown"
        lname = name.lower()
        self.split_hint = ("val" if "valid" in lname else "test" if "test" in lname else "train")

    def __repr__(self):
        return (f"<{self.kind:<11} {self.split_hint:<5} imgs={len(self.coco.get('images', [])):4d} "
                f"anns={len(self.coco.get('annotations', [])):5d}  {self.name}>")


def discover(zips: list[Path], jsons: list[Path]) -> tuple[list[Source], dict]:
    sources: list[Source] = []
    # image index: basename -> list of (zip_path, member)
    img_index: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                pm = PurePosixPath(m)
                if pm.name.startswith(".") or "__MACOSX" in m:
                    continue
                if pm.suffix.lower() in IMG_EXT:
                    img_index[pm.name].append((z, m))
                elif pm.suffix.lower() == ".json":
                    try:
                        coco = json.loads(zf.read(m))
                    except Exception:
                        continue
                    if isinstance(coco, dict) and "images" in coco and "annotations" in coco:
                        sources.append(Source(f"{z.name}:{m}", coco, z, m))
    for j in jsons:
        coco = json.loads(Path(j).read_text())
        if isinstance(coco, dict) and "images" in coco and "annotations" in coco:
            sources.append(Source(Path(j).name, coco, None, None))
    return sources, img_index


def resolve_image(file_name: str, src: Source, img_index: dict) -> tuple[Path, str] | None:
    base = PurePosixPath(file_name).name
    cands = img_index.get(base, [])
    if not cands:
        return None
    if src.zip_path is not None:
        jdir = str(PurePosixPath(src.json_member).parent)
        same_dir = [c for c in cands if c[0] == src.zip_path and c[1].startswith(jdir + "/")]
        if same_dir:
            return same_dir[0]
        same_zip = [c for c in cands if c[0] == src.zip_path]
        if same_zip:
            return same_zip[0]
    # external json (e.g. validation_triple.json) -> zip whose name shares its first token
    token = src.name.lower().split("_")[0].split(".")[0]
    tok = [c for c in cands if token and token in c[0].name.lower()]
    if tok:
        return tok[0]
    return cands[0] if len(cands) == 1 else None


# --------------------------------------------------------------------------- label parsing
# Official DENTEX ids (used when a file ships without category tables, e.g. validation_triple.json)
DENTEX_DEFAULT_CATS = {
    1: {i: str(i + 1) for i in range(4)},
    2: {i: str(i + 1) for i in range(8)},
    3: {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"},
}
GLOBAL_CATS: dict[int, dict[int, str]] = {}


def learn_categories(sources) -> None:
    """Take category tables from any file that has them; share with files that don't."""
    for lvl in (1, 2, 3):
        for s in sources:
            cats = s.coco.get(f"categories_{lvl}")
            if cats:
                GLOBAL_CATS[lvl] = {int(c["id"]): str(c["name"]) for c in cats}
                break
        else:
            GLOBAL_CATS[lvl] = DENTEX_DEFAULT_CATS[lvl]
            print(f"  (categories_{lvl} not found in any file -> using official DENTEX ids)")
    print(f"  diagnosis ids: {GLOBAL_CATS[3]}")


def cat_lookup(coco: dict, level: int) -> dict[int, str]:
    cats = coco.get(f"categories_{level}") or []
    return {int(c["id"]): str(c["name"]) for c in cats} or GLOBAL_CATS.get(level, DENTEX_DEFAULT_CATS[level])


def as_int(name: str, fallback: int) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else fallback


def ann_fdi(a: dict, c1: dict, c2: dict) -> str | None:
    try:
        qid, nid = int(a["category_id_1"]), int(a["category_id_2"])
    except (KeyError, TypeError, ValueError):
        return None
    q = as_int(c1.get(qid, ""), qid + 1)
    n = as_int(c2.get(nid, ""), nid + 1)
    # some exports store quadrant as "Q1"/"1" and tooth as "1".."8"; FDI already combined like "11"?
    if q > 4 and 11 <= q <= 48:
        q, n = divmod(q, 10)
    try:
        return fdi_from_parts(q, n)
    except ValueError:
        return None


def xywh_clip(b, w, h):
    x, y, bw, bh = map(float, b)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


# --------------------------------------------------------------------------- records
def build_records(src: Source, img_index: dict, zcache: dict) -> list[dict]:
    coco = src.coco
    c1, c2, c3 = cat_lookup(coco, 1), cat_lookup(coco, 2), cat_lookup(coco, 3)
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    recs, missing = [], 0
    for im in coco["images"]:
        loc = resolve_image(im["file_name"], src, img_index)
        if loc is None:
            missing += 1
            continue
        objs = []
        for a in by_img.get(im["id"], []):
            fdi = ann_fdi(a, c1, c2)
            diag = None
            if src.kind == "diagnosis":
                did = int(a["category_id_3"])
                diag = normalize_finding(c3.get(did, str(did)))
            seg = a.get("segmentation")
            poly = max(seg, key=len) if isinstance(seg, list) and seg and isinstance(seg[0], list) else None
            objs.append({"bbox_xywh": a["bbox"], "fdi": fdi, "diagnosis": diag, "poly": poly})
        recs.append({"src": src.name, "kind": src.kind, "file_name": im["file_name"],
                     "width": im.get("width"), "height": im.get("height"),
                     "zip": loc[0], "member": loc[1], "objects": objs})
    if missing:
        print(f"  ! {src.name}: {missing} images not found in zips (skipped)")
    return recs


def read_image(rec: dict, zcache: dict) -> np.ndarray:
    z = zcache.setdefault(rec["zip"], zipfile.ZipFile(rec["zip"]))
    buf = np.frombuffer(z.read(rec["member"]), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot decode {rec['member']}")
    return img


# --------------------------------------------------------------------------- writer
class YoloWriter:
    def __init__(self, root: Path, names: list[str], max_side: int):
        self.root, self.names, self.max_side = root, names, max_side
        self.counts = defaultdict(Counter)
        self.n_imgs = Counter()
        for s in ("train", "val", "test"):
            (root / "images" / s).mkdir(parents=True, exist_ok=True)
            (root / "labels" / s).mkdir(parents=True, exist_ok=True)

    def write(self, split: str, stem: str, img: np.ndarray, boxes: list[tuple[int, tuple]]):
        h, w = img.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.root / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for cls, (x0, y0, x1, y1) in boxes:
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            self.counts[split][self.names[cls]] += 1
        (self.root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        self.n_imgs[split] += 1
        return scale, img.shape[:2]

    def finish(self):
        yaml = [f"path: {self.root.resolve()}", "train: images/train", "val: images/val",
                "test: images/test", f"nc: {len(self.names)}", "names:"]
        yaml += [f"  {i}: '{n}'" for i, n in enumerate(self.names)]
        (self.root / "data.yaml").write_text("\n".join(yaml) + "\n")
        return {"images": dict(self.n_imgs), "instances": {k: dict(v) for k, v in self.counts.items()}}


def split_list(items, fracs, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    a = int(round(n * fracs[0]))
    b = a + int(round(n * fracs[1]))
    return items[:a], items[a:b], items[b:]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(P.RAW), help="download dir for DENTEX zips")
    ap.add_argument("--out", default=str(P.DATA), help="output dir for YOLO datasets")
    ap.add_argument("--local-zips", nargs="*", help="use these zips instead of downloading")
    ap.add_argument("--extra-json", nargs="*", default=[], help="external COCO jsons (with --local-zips)")
    ap.add_argument("--max-side", type=int, default=1600, help="downscale long side (saves disk/RAM)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delete-raw", action="store_true", help="delete downloaded zips when done")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    local = sorted(raw.glob("*.zip")) if raw.is_dir() else []
    if args.local_zips:
        zips, jsons = [Path(p) for p in args.local_zips], [Path(p) for p in args.extra_json]
    elif local:   # files downloaded manually into the raw folder -> use them, no download
        zips, jsons = local, sorted(raw.glob("*.json")) + [Path(p) for p in args.extra_json]
        args.local_zips = [str(z) for z in zips]
        print(f"[local] using files already in {raw}: {[p.name for p in zips + jsons]}")
        need = {"training_data.zip", "validation_data.zip", "test_data.zip", "validation_triple.json"}
        missing = need - {p.name for p in zips + jsons}
        if missing:
            print(f"  ! not found (optional, will split from train instead): {sorted(missing)}")
    else:
        zips, jsons = download(raw)

    print("[discover] scanning archives ...", flush=True)
    sources, img_index = discover(zips, jsons)
    for s in sources:
        print("  ", s)
    learn_categories(sources)
    zcache: dict = {}

    enum_src = [s for s in sources if s.kind == "enumeration"]
    diag_src = [s for s in sources if s.kind == "diagnosis"]
    if not enum_src or not diag_src:
        sys.exit("ERROR: could not find both the enumeration and the diagnosis annotation sets.")

    enum_recs = [r for s in enum_src for r in build_records(s, img_index, zcache)]
    diag_by_split = defaultdict(list)
    for s in diag_src:
        diag_by_split[s.split_hint] += build_records(s, img_index, zcache)

    # ---- findings splits: official val/test if labelled, else carve from train
    f_train = diag_by_split["train"]
    f_val, f_test = diag_by_split["val"], diag_by_split["test"]
    if not f_test:
        f_train, _, f_test = split_list(f_train, (0.85, 0.0), args.seed)
        print(f"  (no labelled test set found -> carved {len(f_test)} images from train)")
    if not f_val:
        f_train, _, f_val = split_list(f_train, (0.9, 0.0), args.seed + 1)
        print(f"  (no labelled val set found -> carved {len(f_val)} images from train)")

    # ---- hash everything to kill cross-set leakage (same X-ray in two sets)
    print("[hash] checking for duplicate images across sets ...", flush=True)

    def h(rec):
        if "_md5" not in rec:
            z = zcache.setdefault(rec["zip"], zipfile.ZipFile(rec["zip"]))
            rec["_md5"] = hashlib.md5(z.read(rec["member"])).hexdigest()
        return rec["_md5"]

    eval_hashes = {h(r) for r in f_val + f_test}
    test_hashes = {h(r) for r in f_test}
    before = (len(f_train), len(f_val), len(enum_recs))
    f_val = [r for r in f_val if h(r) not in test_hashes]
    f_train = [r for r in f_train if h(r) not in eval_hashes]
    enum_recs = [r for r in enum_recs if h(r) not in eval_hashes]
    def dedupe(recs):
        seen, keep = set(), []
        for r in recs:
            if h(r) not in seen:
                seen.add(h(r))
                keep.append(r)
        return keep

    f_train, enum_recs = dedupe(f_train), dedupe(enum_recs)
    print(f"  removed duplicates: findings-train {before[0]-len(f_train)}, "
          f"findings-val {before[1]-len(f_val)}, teeth {before[2]-len(enum_recs)}")

    t_train, t_val, t_test = split_list(enum_recs, (0.8, 0.1), args.seed)

    # ---- write teeth dataset
    print("[write] teeth (32 FDI classes) ...", flush=True)
    tw = YoloWriter(out / "teeth", FDI_TEETH, args.max_side)
    seg_root = out / "teeth_seg"                      # same images + tooth OUTLINES (tight boxes / masks)
    for s_ in ("train", "val", "test"):
        (seg_root / "images" / s_).mkdir(parents=True, exist_ok=True)
        (seg_root / "labels" / s_).mkdir(parents=True, exist_ok=True)
    n_poly = 0
    dropped = Counter()
    for split, recs in (("train", t_train), ("val", t_val), ("test", t_test)):
        for i, r in enumerate(recs):
            img = read_image(r, zcache)
            H, W = img.shape[:2]
            boxes = []
            for o in r["objects"]:
                bb = xywh_clip(o["bbox_xywh"], W, H)
                if bb is None or o["fdi"] is None:
                    dropped["bad_box_or_fdi"] += 1
                    continue
                boxes.append((FDI_TO_IDX[o["fdi"]], bb))
            stem = f"teeth_{split}_{i:04d}"
            tw.write(split, stem, img, boxes)
            seg_lines = []
            for o in r["objects"]:
                if o["fdi"] is None or not o.get("poly") or len(o["poly"]) < 6:
                    continue
                pts = np.array(o["poly"], np.float32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0] / W, 0, 1)
                pts[:, 1] = np.clip(pts[:, 1] / H, 0, 1)
                seg_lines.append(f"{FDI_TO_IDX[o['fdi']]} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in pts))
            if seg_lines:
                n_poly += len(seg_lines)
                shutil.copy2(out / "teeth" / "images" / split / f"{stem}.jpg", seg_root / "images" / split / f"{stem}.jpg")
                (seg_root / "labels" / split / f"{stem}.txt").write_text("\n".join(seg_lines))
    teeth_stats = tw.finish()
    if n_poly:
        yaml = [f"path: {seg_root.resolve()}", "train: images/train", "val: images/val", "test: images/test",
                f"nc: {len(FDI_TEETH)}", "names:"] + [f"  {i}: '{n}'" for i, n in enumerate(FDI_TEETH)]
        (seg_root / "data.yaml").write_text("\n".join(yaml) + "\n")
        print(f"[write] teeth_seg: {n_poly} tooth outlines")
    else:
        shutil.rmtree(seg_root, ignore_errors=True)
        print("[write] no tooth outlines in the annotations -> teeth_seg skipped")

    # ---- write findings dataset (+ end-to-end GT for test)
    print("[write] findings (caries / deep caries / periapical / impacted) ...", flush=True)
    fw = YoloWriter(out / "findings", FINDINGS, args.max_side)
    gt_all: dict = {}
    for split, recs in (("train", f_train), ("val", f_val), ("test", f_test)):
        for i, r in enumerate(recs):
            img = read_image(r, zcache)
            H, W = img.shape[:2]
            boxes, gt_objs = [], []
            for o in r["objects"]:
                bb = xywh_clip(o["bbox_xywh"], W, H)
                if bb is None or o["diagnosis"] is None:
                    dropped["bad_box_or_diag"] += 1
                    continue
                boxes.append((FINDINGS.index(o["diagnosis"]), bb))
                gt_objs.append({"diagnosis": o["diagnosis"], "fdi": o["fdi"], "box": bb})
            stem = f"findings_{split}_{i:04d}"
            scale, _ = fw.write(split, stem, img, boxes)
            if split in ("val", "test"):
                gt_all.setdefault(split, []).append({"image": f"images/{split}/{stem}.jpg", "source_file": r["file_name"],
                                "objects": [{**g, "box": [round(v * scale, 1) for v in g["box"]]}
                                            for g in gt_objs]})
    findings_stats = fw.finish()
    for sp, items in gt_all.items():       # val -> threshold calibration, test -> final report
        (out / "findings" / f"findings_gt_{sp}.json").write_text(json.dumps(items, indent=1))

    manifest = {"sources": [repr(s) for s in sources], "teeth": teeth_stats,
                "findings": findings_stats, "dropped": dict(dropped), "max_side": args.max_side,
                "license": "DENTEX data: CC BY-NC-SA 4.0 (non-commercial research)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"teeth": teeth_stats["images"], "findings": findings_stats["images"],
                      "findings_instances_train": findings_stats["instances"].get("train", {}),
                      "dropped": dict(dropped)}, indent=2))

    for z in zcache.values():
        z.close()
    if args.delete_raw and not args.local_zips:
        for p in zips:
            p.unlink(missing_ok=True)
        print("[cleanup] deleted raw zips")
    print(f"[done] datasets in {out}")


if __name__ == "__main__":
    main()
