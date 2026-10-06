#!/usr/bin/env python3
"""
External validation on a DIFFERENT population + camera: Zenodo smartphone test photos (Pakistan, Samsung A23)
vs models trained on Roboflow ICDAS II.

  A. Severity detector (occlusal_det.pt, Roboflow):  any box with severity Mild/Moderate/Advanced = "caries".
  B. Full pipeline (tooth segmenter + severity classifier), if its weights exist: tooth graded >= Mild = "caries".
  C. Zenodo caries detector (occlusal_caries_det.pt) on its own held-out test split (mAP, for reference).

Reference = dentist-drawn permanent-caries ('D') boxes. The two datasets label differently (lesion box vs
tooth/ICDAS box), so a hit = the prediction covers >= 30 % of the reference box.

Reports lesion-level sensitivity, photo-level sensitivity/specificity, and share of predictions on a lesion.
-> results/occlusal/EXTERNAL_VALIDATION.md + external_validation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402

Z = P.HOME / "data" / "occlusal" / "zenodo"
RES = P.RESULTS / "occlusal"


def cover(ref, pred):
    ix0, iy0, ix1, iy1 = max(ref[0], pred[0]), max(ref[1], pred[1]), min(ref[2], pred[2]), min(ref[3], pred[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ra = max(1e-6, (ref[2] - ref[0]) * (ref[3] - ref[1]))
    pa = max(1e-6, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    return inter / ra, inter / pa


def score(gt_items, preds_by_img, thr=0.3):
    les_tp = les_n = pred_on = pred_n = 0
    tp = fp = fn = tn = 0
    for g in gt_items:
        refs = g["caries_boxes"]
        preds = preds_by_img.get(g["image"], [])
        les_n += len(refs)
        les_tp += sum(any(cover(r, p)[0] >= thr for p in preds) for r in refs)
        pred_n += len(preds)
        pred_on += sum(any(cover(r, p)[1] >= thr or cover(r, p)[0] >= thr for r in refs) for p in preds)
        pos, hit = bool(refs), bool(preds)
        tp += pos and hit; fn += pos and not hit; fp += (not pos) and hit; tn += (not pos) and not hit
    d = lambda a, b: round(a / b, 4) if b else None
    return {"photos": len(gt_items), "lesions": les_n,
            "lesion_sensitivity": d(les_tp, les_n),
            "photo_sensitivity": d(tp, tp + fn), "photo_specificity": d(tn, tn + fp),
            "predictions": pred_n, "share_of_predictions_on_a_lesion": d(pred_on, pred_n),
            "photo_confusion": {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    gt = json.loads((Z / "test_gt.json").read_text())
    RES.mkdir(parents=True, exist_ok=True)
    report, md = {"reference": "Zenodo 14769743 test photos, permanent-caries boxes"}, [
        "# External validation - Zenodo smartphone photos (different country, camera, labelling team)", "",
        f"Test photos: {len(gt)}, reference lesions: {sum(len(g['caries_boxes']) for g in gt)}. "
        "Hit = prediction covers >= 30% of the dentist's box.", ""]
    kw = {"device": a.device} if a.device else {}

    w = P.WEIGHTS / "occlusal_det.pt"
    if w.exists():
        m = YOLO(str(w))
        preds = {}
        for g in gt:
            r = m.predict(str(Z / g["image"]), conf=a.conf, verbose=False, **kw)[0]
            preds[g["image"]] = [b.tolist() for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())
                                 if r.names[int(c)] != "no_caries"] if r.boxes is not None else []
        report["A_severity_detector_roboflow"] = score(gt, preds)

    from dentassist.occlusal.pipeline import load_default
    an = load_default(P.WEIGHTS)
    if an is not None:
        preds = {}
        for g in gt:
            img = cv2.imread(str(Z / g["image"]))
            res = an.analyze(img)
            s = img.shape[1] / res["image"]["width"]      # pipeline works on the pre-processed size
            preds[g["image"]] = [[v * s for v in t["box"]] for t in res["teeth"] if t["severity"] != "no_caries"]
        report["B_full_pipeline"] = score(gt, preds)

    w = P.WEIGHTS / "occlusal_caries_det.pt"
    if w.exists():
        v = YOLO(str(w)).val(data=str(Z / "det/data.yaml"), split="test", verbose=False, plots=False, **kw)
        report["C_zenodo_caries_detector_internal_test"] = {"mAP50": round(float(v.box.map50), 4),
                                                            "precision": round(float(v.box.mp), 4),
                                                            "recall": round(float(v.box.mr), 4)}

    for k, v in report.items():
        if isinstance(v, dict) and "lesion_sensitivity" in v:
            md += [f"## {k}", "", "| Metric | Value |", "|---|---|",
                   *[f"| {kk} | {vv} |" for kk, vv in v.items()], ""]
        elif isinstance(v, dict):
            md += [f"## {k}", "", *[f"- {kk}: {vv}" for kk, vv in v.items()], ""]
    (RES / "external_validation.json").write_text(json.dumps(report, indent=2))
    (RES / "EXTERNAL_VALIDATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
