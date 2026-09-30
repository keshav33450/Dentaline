#!/usr/bin/env python3
"""
Model 1 calibration + ground-truth validation (no retraining).

On the VALIDATION split (never the test split):
  1. preprocessing A/B:  none  vs  clahe (denoise + contrast)       -> keep the better one
  2. per-class confidence thresholds that maximise F1                 -> 'likely' cut-off per class
On the TEST split (final, reported once):
  3. findings at the calibrated operating point: likely-only and likely+possible, per class, FP/FN counts
  4. FDI numbering accuracy with and without the anatomical repair (duplicates, wrong numbers)

Writes  <weights>/thresholds.json  (read automatically by the app)  and  results/CALIBRATION.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.analyzer import OPGAnalyzer  # noqa: E402
from dentassist.fdi import FDI_TEETH, FINDINGS  # noqa: E402
from dentassist.postprocess import DEFAULT_THRESHOLDS, iou  # noqa: E402

GRID = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]


def run_findings(an, root, gt):
    """-> list of (image_idx, diag, conf, box, fdi) predictions, all above a very low floor."""
    preds = []
    for k, item in enumerate(gt):
        r = an.analyze(str(root / item["image"]), include_teeth=False)
        for f in r["findings"]:
            preds.append((k, f["diagnosis"], f["confidence"], f["box"], f["fdi"]))
    return preds


def prf_at(preds, gt, diag, thr, need_tooth=False):
    tp = fp = 0
    used = defaultdict(set)
    for k, d, c, box, fdi in sorted(preds, key=lambda p: -p[2]):
        if d != diag or c < thr:
            continue
        best, bj = 0.0, -1
        for j, g in enumerate(gt[k]["objects"]):
            if g["diagnosis"] != diag or j in used[k]:
                continue
            v = iou(box, g["box"])
            if v > best:
                best, bj = v, j
        ok = best >= 0.5 and (not need_tooth or gt[k]["objects"][bj]["fdi"] == fdi)
        if ok:
            used[k].add(bj)
            tp += 1
        else:
            fp += 1
    n = sum(g["diagnosis"] == diag for it in gt for g in it["objects"])
    fn = n - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n if n else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "TP": tp, "FP": fp, "FN": fn}


def best_thresholds(preds, gt):
    out = {}
    for d in FINDINGS:
        rows = [(t, prf_at(preds, gt, d, t)) for t in GRID]
        t, m = max(rows, key=lambda x: (x[1]["f1"], -abs(x[0] - 0.5)))   # ties -> closest to 0.5
        if m["f1"] == 0:                     # no signal (no cases / no matches) -> keep the safe default
            t = DEFAULT_THRESHOLDS["findings"][d]
            m = dict(prf_at(preds, gt, d, t), note="not calibrated - no matches on validation")
        out[d] = (t, m)
    return out


def teeth_gt(root, split):
    items = []
    for lab in sorted((root / "teeth/labels" / split).glob("*.txt")):
        img = root / "teeth/images" / split / f"{lab.stem}.jpg"
        h, w = cv2.imread(str(img)).shape[:2]
        objs = []
        for line in lab.read_text().split("\n"):
            v = line.split()
            if len(v) == 5:
                c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:])
                objs.append((FDI_TEETH[c], [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
        items.append((img, objs))
    return items


def numbering(an, items):
    correct = total = found = dup = 0
    for img, objs in items:
        r = an.analyze(str(img))
        pred = r["teeth"]
        nums = [t["fdi"] for t in pred]
        dup += len(nums) - len(set(nums))
        for fdi, box in objs:
            total += 1
            m = max(pred, key=lambda t: iou(t["box"], box), default=None)
            if m is not None and iou(m["box"], box) >= 0.5:
                found += 1
                correct += m["fdi"] == fdi
    return {"teeth": total, "detected": round(found / total, 4) if total else None,
            "correct_number_of_detected": round(correct / found, 4) if found else None,
            "correct_number_overall": round(correct / total, 4) if total else None, "duplicate_numbers": dup}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(P.DATA))
    ap.add_argument("--teeth", default=None)
    ap.add_argument("--findings", default=str(P.WEIGHTS / "findings_best.pt"))
    ap.add_argument("--out", default=str(P.WEIGHTS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=0, help="limit (quick tests)")
    a = ap.parse_args()
    root = Path(a.data)
    teeth_w = a.teeth or next(str(P.WEIGHTS / n) for n in ("teeth_seg_best.pt", "teeth_best.pt") if (P.WEIGHTS / n).exists())
    fr = root / "findings"
    gt_val = json.loads((fr / "findings_gt_val.json").read_text())
    gt_test = json.loads((fr / "findings_gt_test.json").read_text())
    if a.max_images:
        gt_val, gt_test = gt_val[:a.max_images], gt_test[:a.max_images]

    def make(pre, repair=True, thr=None):
        t = json.loads(json.dumps(thr or DEFAULT_THRESHOLDS))
        return OPGAnalyzer(teeth_w, a.findings, thresholds=t, preprocess=pre, repair_numbers=repair, device=a.device)

    low = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    low["floor"] = 0.10
    low["findings"] = {k: 0.10 for k in FINDINGS}

    # ---- 1+2: validation
    report, choice = {"validation_images": len(gt_val), "test_images": len(gt_test)}, {}
    for pre in ("none", "clahe"):
        preds = run_findings(make(pre, thr=low), fr, gt_val)
        th = best_thresholds(preds, gt_val)
        macro = float(np.mean([m["f1"] for _, m in th.values()]))
        report[f"val_{pre}"] = {"macro_f1": round(macro, 4), **{d: {"threshold": t, **m} for d, (t, m) in th.items()}}
        choice[pre] = (macro, th)
        print(f"[val] preprocess={pre}: macro-F1 {macro:.3f}  " + ", ".join(f"{d} {t}" for d, (t, _) in th.items()), flush=True)
    pre = max(choice, key=lambda k: choice[k][0])
    th = choice[pre][1]
    thresholds = {"teeth": DEFAULT_THRESHOLDS["teeth"],
                  "floor": round(max(0.20, min(t for t, _ in th.values()) - 0.15), 2),
                  "findings": {d: t for d, (t, _) in th.items()}, "preprocess": pre,
                  "source": f"calibrated on DENTEX validation split ({len(gt_val)} X-rays), F1-optimal per class"}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    report["chosen"] = thresholds

    # ---- 3: test at the calibrated operating point
    preds = run_findings(make(pre, thr=dict(thresholds, floor=0.10)), fr, gt_test)
    fl = thresholds["floor"]
    report["test_likely"] = {d: prf_at(preds, gt_test, d, thresholds["findings"][d]) for d in FINDINGS}
    report["test_likely_plus_possible"] = {d: prf_at(preds, gt_test, d, fl) for d in FINDINGS}
    report["test_likely_right_tooth"] = {d: prf_at(preds, gt_test, d, thresholds["findings"][d], need_tooth=True) for d in FINDINGS}

    # ---- 4: numbering with / without repair
    items = teeth_gt(root, "test")[: a.max_images or None]
    report["numbering_without_repair"] = numbering(make(pre, repair=False, thr=thresholds), items)
    report["numbering_with_repair"] = numbering(make(pre, repair=True, thr=thresholds), items)

    res = P.RESULTS
    res.mkdir(parents=True, exist_ok=True)
    (res / "calibration.json").write_text(json.dumps(report, indent=2))
    md = ["# Model 1 - calibration & ground-truth validation", "",
          f"Validation X-rays: {len(gt_val)} (thresholds + preprocessing chosen here). Test X-rays: {len(gt_test)} (reported once).", "",
          "## Preprocessing A/B (validation, macro-F1)", "",
          f"- none: {report['val_none']['macro_f1']}", f"- clahe (denoise + contrast): {report['val_clahe']['macro_f1']}",
          f"- **chosen: {pre}**", "", "## Thresholds (F1-optimal on validation)", "",
          "| Finding | 'Likely' threshold |", "|---|---|",
          *[f"| {d} | {t} |" for d, t in thresholds["findings"].items()],
          f"| hidden below | {thresholds['floor']} (shown as 'Possible - needs review' between) |", "",
          "## Test set - findings", "",
          "| Finding | P (likely) | R (likely) | F1 (likely) | FP | FN | R (likely+possible) | F1 right tooth |",
          "|---|---|---|---|---|---|---|---|"]
    for d in FINDINGS:
        L, LP, T = report["test_likely"][d], report["test_likely_plus_possible"][d], report["test_likely_right_tooth"][d]
        md.append(f"| {d} | {L['precision']} | {L['recall']} | {L['f1']} | {L['FP']} | {L['FN']} | {LP['recall']} | {T['f1']} |")
    a_, b_ = report["numbering_without_repair"], report["numbering_with_repair"]
    md += ["", "## Test set - FDI numbering", "", "| | without repair | with repair |", "|---|---|---|",
           f"| teeth detected | {a_['detected']} | {b_['detected']} |",
           f"| correct number (of detected) | {a_['correct_number_of_detected']} | {b_['correct_number_of_detected']} |",
           f"| duplicate numbers | {a_['duplicate_numbers']} | {b_['duplicate_numbers']} |"]
    (res / "CALIBRATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
