#!/usr/bin/env python3
"""Analyze one or more X-rays with the trained models; writes <name>_result.jpg + .json beside each image.

  python scripts/analyze.py path/to/opg.jpg [more.jpg ...] [--out folder]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # never try to pip-install packages at run time
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.analyzer import OPGAnalyzer  # noqa: E402


def pick(*names):
    for n in names:
        if (P.WEIGHTS / n).exists():
            return str(P.WEIGHTS / n)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="output folder (default: next to each image)")
    ap.add_argument("--teeth", default=pick("teeth_seg_best.pt", "teeth_best.pt", "teeth.onnx"))
    ap.add_argument("--findings", default=pick("findings_best.pt", "findings.onnx"))
    a = ap.parse_args()
    if not a.teeth:
        sys.exit(f"No teeth model in {P.WEIGHTS} - copy teeth.onnx / teeth_best.pt there (or run windows\\set_location.bat)")
    an = OPGAnalyzer(a.teeth, a.findings)
    for img in a.images:
        p = Path(img)
        res = an.analyze(str(p))
        out = Path(a.out) if a.out else p.parent
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{p.stem}_result.jpg"), OPGAnalyzer.draw(str(p), res))
        (out / f"{p.stem}_result.json").write_text(json.dumps(res, indent=1))
        print(f"\n{p.name}: {res['summary']['teeth_detected']} teeth ({res['summary']['teeth_renumbered']} renumbered), "
              f"{len(res['findings'])} findings, {len(res['suppressed_findings'])} hidden (low confidence / duplicate) "
              f"({res['timing_ms']['total']} ms)")
        for f in res["findings"]:
            print(f"  tooth {f['fdi'] or '?'}: {f['display']}")
        print(f"  saved -> {out / (p.stem + '_result.jpg')}")


if __name__ == "__main__":
    main()
