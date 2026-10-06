#!/usr/bin/env python3
"""Analyse smartphone occlusal photo(s) -> <name>_occlusal.jpg, .json and PDF report next to each photo.

  python scripts/occ_analyze.py photo1.jpg [photo2.jpg ...] [--patient P001] [--out folder]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.pipeline import load_default  # noqa: E402
from dentassist.occlusal.report import make_pdf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--patient", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    an = load_default(P.WEIGHTS)
    if an is None:
        sys.exit(f"Need occlusal_seg.pt, occlusal_type.pt, occlusal_severity.pt in {P.WEIGHTS}")
    for f in a.images:
        p = Path(f)
        img = cv2.imread(str(p))
        if img is None:
            print(f"cannot read {p}"); continue
        res = an.analyze(img)
        vis = an.draw(res)
        out = Path(a.out) if a.out else p.parent
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{p.stem}_occlusal.jpg"), vis)
        make_pdf(res, vis, out / f"{p.stem}_report.pdf", patient_code=a.patient or p.stem.split("_")[0], photo_name=p.name)
        (out / f"{p.stem}_occlusal.json").write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=1))
        s = res["summary"]
        print(f"{p.name}: {s['teeth_analysed']} teeth | {s['severity_counts']} | quality ok={res['quality']['ok']} -> {out / (p.stem + '_report.pdf')}")


if __name__ == "__main__":
    main()
