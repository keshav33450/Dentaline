#!/usr/bin/env python3
"""
Fetch the Roboflow 'FDI_numbering' intraoral tooth-segmentation dataset (classes T11..T48) and
convert to our prepared/ layout. Downloads by API key straight into Kaggle - no login, no cookies.

  python scripts/roboflow_data.py --api-key KEY --out prepared
  (defaults target workspace/project/version below; override with --workspace/--project/--version)

Roboflow gives YOLOv8 instance-seg already (images + polygon .txt), so we mainly remap its class ids
(T11..T48, plus Bridge/Crown/Implant which we drop) to our FDI index 0..31 and split by our scheme.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tooth_ai_common import FDI_INDEX, FDI_NUMBERS  # noqa: E402
from scripts.prepare_dataset import VIEW_FOR_DETECTOR  # noqa: E402  (unused but keeps parity)

DEFAULT = dict(workspace="mikhakdental", project="fdi_numbering-v0e9z", version=2)


def load_rf_yaml(root: Path):
    """read data.yaml -> {class_index: name}"""
    y = next(root.rglob("data.yaml"))
    txt = y.read_text()
    m = re.search(r"names:\s*\[([^\]]*)\]", txt, re.S)
    if m:
        names = [n.strip().strip("'\"") for n in m.group(1).split(",")]
    else:
        names = [l.split(":", 1)[1].strip().strip("'\"")
                 for l in txt.splitlines() if re.match(r"\s*\d+\s*:", l)]
    return {i: n for i, n in enumerate(names)}, y.parent


def fdi_of(name: str):
    """T11 / 11 / tooth_11 -> 11 (int) if it is an FDI class, else None."""
    m = re.search(r"(\d{2})", name)
    if not m:
        return None
    f = int(m.group(1))
    return f if f in FDI_INDEX else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--workspace", default=DEFAULT["workspace"])
    ap.add_argument("--project", default=DEFAULT["project"])
    ap.add_argument("--version", type=int, default=DEFAULT["version"])
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--raw", default=None, help="already-downloaded Roboflow export dir (skip download)")
    a = ap.parse_args()
    out = Path(a.out)

    if a.raw:
        root = Path(a.raw)
    else:
        from roboflow import Roboflow
        rf = Roboflow(api_key=a.api_key)
        proj = rf.workspace(a.workspace).project(a.project)
        # list what actually exists, robustly (roboflow returns Version objects or url strings)
        vers = []
        try:
            for v in proj.versions():
                vid = getattr(v, "version", None) or getattr(v, "id", None) or str(v)
                m = re.search(r"(\d+)\s*$", str(vid).rstrip("/"))
                if m:
                    vers.append(int(m.group(1)))
        except Exception as e:
            print("could not list versions:", e, flush=True)
        vers = sorted(set(vers))
        print("available versions:", vers or "(none listed)", flush=True)
        candidates = ([a.version] if a.version in vers else []) + list(reversed(vers)) + [a.version, 1, 2, 3, 4, 5]
        ds = None; last = None
        for pick in dict.fromkeys(candidates):
            try:
                ds = proj.version(pick).download("yolov8", location=str(out.parent / "_rf_fdi"))
                print("downloaded version", pick, flush=True); break
            except Exception as e:
                last = e; continue
        if ds is None:
            raise SystemExit(f"no downloadable version found (tried {list(dict.fromkeys(candidates))}); last error: {last}")
        root = Path(ds.location)

    names, base = load_rf_yaml(root)
    # map roboflow class index -> our FDI index (drop non-FDI classes like Bridge/Crown/Implant)
    rf2fdi = {}
    for ci, nm in names.items():
        f = fdi_of(nm)
        if f is not None:
            rf2fdi[ci] = FDI_INDEX[f]

    fdi_names = {i: str(f) for i, f in enumerate(FDI_NUMBERS)}
    names_block = "\n".join(f"  {k}: {v}" for k, v in fdi_names.items())
    stats = Counter(); cls_count = Counter()
    # Roboflow splits: train/valid/test
    split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    seg = out / "seg"
    for s in ("train", "val", "test"):
        (seg / "images" / s).mkdir(parents=True, exist_ok=True)
        (seg / "labels" / s).mkdir(parents=True, exist_ok=True)

    for rfsplit, our in split_map.items():
        idir = base / rfsplit / "images"; ldir = base / rfsplit / "labels"
        if not idir.is_dir():
            continue
        for img in idir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lab = ldir / f"{img.stem}.txt"
            lines = []
            if lab.exists():
                for row in lab.read_text().splitlines():
                    v = row.split()
                    if len(v) < 7:
                        continue
                    ci = int(v[0])
                    if ci not in rf2fdi:
                        continue
                    lines.append(f"{rf2fdi[ci]} " + " ".join(v[1:]))
                    cls_count[FDI_NUMBERS[rf2fdi[ci]]] += 1
            if not lines:
                stats["no_fdi"] += 1
                continue
            shutil.copy2(img, seg / "images" / our / img.name)
            (seg / "labels" / our / f"{img.stem}.txt").write_text("\n".join(lines))
            stats[f"img_{our}"] += 1

    def write_yaml(fname):
        (seg / fname).write_text(f"path: {seg.resolve()}\ntrain: images/train\nval: images/val\n"
                                 f"test: images/test\nnc: {len(fdi_names)}\nnames:\n{names_block}\n")
    # single combined detector (this dataset is mixed-view, no per-view split)
    write_yaml("data_all.yaml")
    for v in ("front", "upper", "lower", "right"):
        write_yaml(f"data_{v}.yaml")   # same data; keeps notebook's per-view loop working

    summary = {"source": f"roboflow {a.workspace}/{a.project} v{a.version}",
               "images": {k.replace("img_", ""): stats[k] for k in stats if k.startswith("img_")},
               "dropped_no_fdi": stats["no_fdi"],
               "fdi_classes": len(rf2fdi), "instances": {str(k): cls_count[k] for k in FDI_NUMBERS},
               "licence": "CC BY 4.0 (Roboflow FDI_numbering)"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    (out / "splits.json").write_text(json.dumps({"note": "roboflow train/valid/test used as-is"}, indent=1))
    print(json.dumps(summary, indent=2, default=int))
    if sum(stats[k] for k in stats if k.startswith("img_")) == 0:
        sys.exit("no FDI-labelled images produced - check the Roboflow project/version.")


if __name__ == "__main__":
    main()
