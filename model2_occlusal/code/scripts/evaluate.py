#!/usr/bin/env python3
"""
Evaluate both models on the held-out TEST split and the combined pipeline.

1. Per-model detection metrics (mAP50, mAP50-95, per-class P/R) + confusion matrices.
2. End-to-end clinical metric on the findings test set:
     a finding counts as correct only if box IoU >= 0.5 AND diagnosis matches AND
     FDI tooth number matches  ->  "right tooth, right problem".
3. Error gallery: side-by-side GT (green) vs prediction images for the worst cases.

  python scripts/evaluate.py --data /tmp/dentex_yolo --teeth weights/teeth_best.pt \
         --findings weights/findings_best.pt --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.analyzer import OPGAnalyzer, iou  # noqa: E402
from dentassist.fdi import FINDINGS  # noqa: E402


def det_metrics(weights, data_yaml, imgsz, out_dir, name, device):
    from ultralytics import YOLO
    m = YOLO(weights)
    kw = dict(data=str(data_yaml), split="test", imgsz=imgsz, batch=4, conf=0.001, iou=0.6,
              project=str(out_dir), name=name, exist_ok=True, plots=True, verbose=False)
    if device:
        kw["device"] = device
    r = m.val(**kw)
    names = r.names
    per_class = {}
    for i, c in enumerate(r.box.ap_class_index):
        p, rcl, ap50, ap = r.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(rcl), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    return {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
            "per_class": per_class, "plots_dir": str(Path(out_dir) / name)}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def end_to_end(analyzer: OPGAnalyzer, root: Path, gt: list, out_dir: Path, gallery_n: int):
    det = defaultdict(lambda: [0, 0, 0])    # diag -> tp, fp, fn   (box + diagnosis)
    full = defaultdict(lambda: [0, 0, 0])   # diag -> tp, fp, fn   (box + diagnosis + FDI)
    fdi_ok = fdi_total = 0
    per_image, lat = [], []
    for item in gt:
        img_path = root / item["image"]
        res = analyzer.analyze(str(img_path))
        lat.append(res["timing_ms"]["total"])
        preds = sorted(res["findings"], key=lambda f: -f["confidence"])
        gts = item["objects"]
        used = set()
        errs = 0
        for p in preds:
            best, bi = 0.0, -1
            for j, g in enumerate(gts):
                if j in used or g["diagnosis"] != p["diagnosis"]:
                    continue
                v = iou(p["box"], g["box"])
                if v > best:
                    best, bi = v, j
            if best >= 0.5:
                used.add(bi)
                det[p["diagnosis"]][0] += 1
                fdi_total += 1
                if gts[bi]["fdi"] and p["fdi"] == gts[bi]["fdi"]:
                    fdi_ok += 1
                    full[p["diagnosis"]][0] += 1
                else:
                    full[p["diagnosis"]][1] += 1
                    full[p["diagnosis"]][2] += 1
                    errs += 1
            else:
                det[p["diagnosis"]][1] += 1
                full[p["diagnosis"]][1] += 1
                errs += 1
        for j, g in enumerate(gts):
            if j not in used:
                det[g["diagnosis"]][2] += 1
                full[g["diagnosis"]][2] += 1
                errs += 1
        per_image.append((errs, img_path, gts, res))

    def agg(d):
        out = {k: prf(*d[k]) for k in FINDINGS if sum(d[k])}
        tot = np.sum([d[k] for k in FINDINGS], axis=0) if d else [0, 0, 0]
        out["overall"] = prf(*map(int, tot))
        return out

    # gallery of worst images
    gdir = out_dir / "error_gallery"
    gdir.mkdir(parents=True, exist_ok=True)
    for errs, path, gts, res in sorted(per_image, key=lambda t: -t[0])[:gallery_n]:
        img = cv2.imread(str(path))
        left = img.copy()
        for g in gts:
            x0, y0, x1, y1 = map(int, g["box"])
            cv2.rectangle(left, (x0, y0), (x1, y1), (0, 220, 0), 3)
            cv2.putText(left, f"{g['fdi']} {g['diagnosis']}", (x0, max(20, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        right = OPGAnalyzer.draw(img, res, show_teeth=False)
        cv2.putText(left, "GROUND TRUTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(right, "PREDICTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
        cv2.imwrite(str(gdir / f"{errs:02d}err_{path.stem}.jpg"), np.vstack([left, right]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {"detection_box+diagnosis": agg(det),
            "clinical_box+diagnosis+tooth": agg(full),
            "fdi_accuracy_on_detected_findings": round(fdi_ok / fdi_total, 4) if fdi_total else None,
            "images": len(gt),
            "latency_ms_per_image": {"mean": round(float(np.mean(lat)), 1) if lat else None,
                                     "p95": round(float(np.percentile(lat, 95)), 1) if lat else None},
            "operating_point": {"teeth_conf": analyzer.teeth_conf, "findings_conf": analyzer.findings_conf},
            "gallery_dir": str(gdir)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(P.DATA))
    ap.add_argument("--teeth", default=str(P.WEIGHTS / "teeth_best.pt"))
    ap.add_argument("--findings", default=str(P.WEIGHTS / "findings_best.pt"))
    ap.add_argument("--teeth-imgsz", type=int, default=1024)
    ap.add_argument("--findings-imgsz", type=int, default=1280)
    ap.add_argument("--findings-conf", type=float, default=0.25)
    ap.add_argument("--teeth-conf", type=float, default=0.35)
    ap.add_argument("--out", default=str(P.RESULTS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--gallery", type=int, default=12)
    args = ap.parse_args()

    root, out = Path(args.data), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    print("[eval] Model 1 - teeth / FDI numbering (test split)")
    report["model1_teeth"] = det_metrics(args.teeth, root / "teeth/data.yaml", args.teeth_imgsz, out, "teeth_test", args.device)
    print("[eval] Model 5 - findings (test split)")
    report["model5_findings"] = det_metrics(args.findings, root / "findings/data.yaml", args.findings_imgsz, out, "findings_test", args.device)

    print("[eval] end-to-end: right tooth + right diagnosis")
    gt = json.loads((root / "findings/findings_gt_test.json").read_text())
    an = OPGAnalyzer(args.teeth, args.findings, teeth_imgsz=args.teeth_imgsz,
                     findings_imgsz=args.findings_imgsz, findings_conf=args.findings_conf,
                     teeth_conf=args.teeth_conf, device=args.device)
    report["end_to_end"] = end_to_end(an, root / "findings", gt, out, args.gallery)

    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    t, f, e = report["model1_teeth"], report["model5_findings"], report["end_to_end"]
    lines = [
        "# DentAssist OPG - test results", "",
        "| Model | mAP50 | mAP50-95 | Precision | Recall |", "|---|---|---|---|---|",
        f"| Model 1 - teeth (32 FDI) | {t['mAP50']} | {t['mAP50-95']} | {t['precision']} | {t['recall']} |",
        f"| Model 5 - findings | {f['mAP50']} | {f['mAP50-95']} | {f['precision']} | {f['recall']} |", "",
        "## Findings per class", "", "| Class | Precision | Recall | mAP50 |", "|---|---|---|---|",
        *[f"| {k} | {v['precision']} | {v['recall']} | {v['mAP50']} |" for k, v in f["per_class"].items()], "",
        f"## End-to-end (IoU>=0.5, conf>={args.findings_conf})", "",
        "| Class | P (box+dx) | R (box+dx) | F1 (box+dx) | F1 (+ correct tooth) |", "|---|---|---|---|---|",
    ]
    d, c = e["detection_box+diagnosis"], e["clinical_box+diagnosis+tooth"]
    for k in [*FINDINGS, "overall"]:
        if k in d:
            lines.append(f"| {k} | {d[k]['precision']} | {d[k]['recall']} | {d[k]['f1']} | {c.get(k, {}).get('f1', '-')} |")
    lines += ["", f"FDI accuracy on detected findings: **{e['fdi_accuracy_on_detected_findings']}**",
              f"Latency per image: mean {e['latency_ms_per_image']['mean']} ms, p95 {e['latency_ms_per_image']['p95']} ms",
              "", "Plots: confusion matrices, PR curves in results/teeth_test and results/findings_test.",
              "Worst cases: results/error_gallery/"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
