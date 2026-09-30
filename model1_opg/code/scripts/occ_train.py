#!/usr/bin/env python3
"""
Train the occlusal-caries models (proforma Phases III-V).

  --stage seg        Phase III  YOLOv8-seg: detect + segment each posterior tooth
  --stage type       Phase IV   premolar vs molar classifier on the tooth crops
  --stage severity   Phase V    No Caries / Mild / Moderate / Advanced classifier

For type/severity, each --arch (efficientnet_b0, mobilenet_v3) is trained with PATIENT-GROUPED
k-fold cross-validation -> every tooth gets an out-of-fold prediction from a model that never saw
that patient (these predictions are what the statistics are computed on). A final model is then
trained on all data for the app.

  python scripts/occ_train.py --stage seg
  python scripts/occ_train.py --stage type     --arch efficientnet_b0 mobilenet_v3
  python scripts/occ_train.py --stage severity --arch efficientnet_b0 mobilenet_v3
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.labels import SEVERITY, TOOTH_TYPES  # noqa: E402

OCC = P.HOME / "data" / "occlusal"
RES = P.RESULTS / "occlusal"


# ----------------------------------------------------------------------------- segmentation
def _yolo_train(a, base, data_yaml, run_name, out_name, epochs_default=150):
    from ultralytics import YOLO
    import torch
    dev = a.device if a.device != "auto" else ("0" if torch.cuda.is_available() else "cpu")
    if not Path(data_yaml).exists():
        raise SystemExit(f"{data_yaml} not found - run the matching data script first")
    model = YOLO(base)
    model.train(data=str(data_yaml), imgsz=a.seg_imgsz, epochs=a.epochs or epochs_default,
                batch=a.batch or 8, patience=40, device=dev, workers=a.workers, project=str(P.RUNS),
                name=run_name, exist_ok=True, cos_lr=True, amp=dev != "cpu",
                fliplr=0.5, flipud=0.5, degrees=15, mosaic=0.5, close_mosaic=15,     # occlusal view: flips are safe
                hsv_h=0.005, hsv_s=0.3, hsv_v=0.3, scale=0.3, translate=0.1)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    P.WEIGHTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, P.WEIGHTS / out_name)
    print(f"[{run_name}] -> {P.WEIGHTS / out_name}")


ZEN = OCC / "zenodo"


def train_seg(a):
    """Study data: YOLOv8-seg, 1 class 'tooth' (type comes from the classifier)."""
    _yolo_train(a, a.seg_model, Path(a.data) / "seg" / "data.yaml", "occlusal_seg", "occlusal_seg.pt")


def train_toothseg(a):
    """Public data: YOLOv8-seg premolar/molar trained on SegmentAnyTooth pseudo-labels of Zenodo phone photos."""
    _yolo_train(a, a.seg_model, ZEN / "toothseg" / "data.yaml", "occlusal_toothseg", "occlusal_toothseg.pt")


def train_det(a):
    """Public data (Roboflow ICDAS II): YOLOv8 detector whose classes ARE the 4 severity levels."""
    _yolo_train(a, a.det_model, Path(a.data) / "det" / "data.yaml", "occlusal_det", "occlusal_det.pt")


def train_cariesdet(a):
    """Public data (Zenodo smartphone): YOLOv8 caries detector, permanent teeth."""
    _yolo_train(a, a.det_model, ZEN / "det" / "data.yaml", "occlusal_caries_det", "occlusal_caries_det.pt", 120)


# ----------------------------------------------------------------------------- classifiers
class CropDS:
    def __init__(self, root, rows, label_key, classes, tf):
        self.root, self.rows, self.key, self.classes, self.tf = Path(root), rows, label_key, classes, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / r["crop"])), cv2.COLOR_BGR2RGB)
        return self.tf(img), self.classes.index(r[self.key])


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


def fit(arch, train_rows, val_rows, a, key, classes, epochs, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from dentassist.occlusal.classifier import build_model, eval_transform, train_transform

    model = build_model(arch, len(classes), pretrained=not a.no_pretrained).to(device)
    counts = np.bincount([classes.index(r[key]) for r in train_rows], minlength=len(classes)).astype(float)
    w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * max(1, -(-len(train_rows) // a.batch_cls)), pct_start=0.15)
    dl = DataLoader(CropDS(a.data, train_rows, key, classes, train_transform(a.img_size)), batch_size=a.batch_cls,
                    shuffle=True, num_workers=a.workers, drop_last=len(train_rows) > a.batch_cls)
    vdl = (DataLoader(CropDS(a.data, val_rows, key, classes, eval_transform(a.img_size)), batch_size=64,
                      num_workers=a.workers) if val_rows else None)
    best, best_ep, best_state, bad = -1.0, epochs, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
        if vdl is not None:
            yp, yt = [], []
            model.eval()
            with torch.no_grad():
                for x, y in vdl:
                    yp.append(model(x.to(device)).argmax(1).cpu().numpy()); yt.append(y.numpy())
            f1 = macro_f1(np.concatenate(yt), np.concatenate(yp), len(classes))
            if f1 > best:
                best, best_ep, bad = f1, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best


def predict_rows(model, rows, a, key, classes, device):
    import torch
    from dentassist.occlusal.classifier import eval_transform
    from torch.utils.data import DataLoader
    dl = DataLoader(CropDS(a.data, rows, key, classes, eval_transform(a.img_size)), batch_size=64, num_workers=a.workers)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            p = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(classes)))


def train_cls(a):
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else (
        "cpu" if a.device == "cpu" else f"cuda:{a.device}")
    key, classes = ("tooth_type", TOOTH_TYPES) if a.stage == "type" else ("severity", SEVERITY)
    with open(Path(a.data) / "teeth.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r[key]]
    for r in rows:
        r["fold"] = int(r["fold"])
    folds = sorted({r["fold"] for r in rows if r["fold"] >= 0})
    if a.max_folds:
        folds = folds[:a.max_folds]      # large datasets: hold out fold 0 only (still grouped by photo/patient)
    print(f"[{a.stage}] {len(rows)} teeth, classes={classes}, folds={folds}, device={device}")
    out_dir = RES / a.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arch in a.arch:
        t0 = time.time()
        oof, best_eps = [], []
        for f in folds:
            test = [r for r in rows if r["fold"] == f]
            rest = [r for r in rows if r["fold"] != f]
            pats = sorted({r["patient_id"] for r in rest})
            random.Random(a.seed + f).shuffle(pats)
            inner = set(pats[:max(1, round(0.15 * len(pats)))])
            tr = [r for r in rest if r["patient_id"] not in inner]
            va = [r for r in rest if r["patient_id"] in inner]
            model, ep, vf1 = fit(arch, tr, va, a, key, classes, a.cls_epochs, device)
            last_model = model
            best_eps.append(ep)
            probs = predict_rows(model, test, a, key, classes, device)
            for r, pr in zip(test, probs):
                oof.append({"crop": r["crop"], "patient_id": r["patient_id"], "image": r["image"], "fdi": r["fdi"],
                            "icdas": r.get("icdas", ""), "fold": f, "true": r[key], "pred": classes[int(pr.argmax())],
                            "confidence": round(float(pr.max()), 4),
                            **{f"p_{c}": round(float(v), 5) for c, v in zip(classes, pr)}})
            acc = np.mean([o["true"] == o["pred"] for o in oof if o["fold"] == f])
            print(f"  {arch} fold {f}: test teeth={len(test)} acc={acc:.3f} (inner-val macroF1={vf1:.3f}, best epoch {ep})")
        y = np.array([classes.index(o["true"]) for o in oof]); p = np.array([classes.index(o["pred"]) for o in oof])
        summary[arch] = {"oof_accuracy": round(float((y == p).mean()), 4), "oof_macro_f1": round(macro_f1(y, p, len(classes)), 4),
                         "n": len(oof), "median_best_epoch": int(np.median(best_eps)), "minutes": round((time.time() - t0) / 60, 1)}
        with open(out_dir / f"oof_{arch}.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(oof[0].keys()))
            wr.writeheader(); wr.writerows(oof)
        print(f"  {arch}: OOF acc={summary[arch]['oof_accuracy']} macroF1={summary[arch]['oof_macro_f1']}")

        # final model on ALL data, epochs = median best epoch from CV (no test leakage)
        if a.no_final:
            final = last_model           # large datasets: ship the fold model (trained on 80%)
        else:
            final, _, _ = fit(arch, rows, [], a, key, classes, max(3, summary[arch]["median_best_epoch"]), device)
        P.WEIGHTS.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": arch, "classes": classes, "img_size": a.img_size, "state_dict": final.state_dict(),
                    "task": a.stage, "cv": summary[arch]}, P.WEIGHTS / f"occlusal_{a.stage}_{arch}.pt")

    best_arch = max(summary, key=lambda k: summary[k]["oof_macro_f1"])
    shutil.copy2(P.WEIGHTS / f"occlusal_{a.stage}_{best_arch}.pt", P.WEIGHTS / f"occlusal_{a.stage}.pt")
    summary["selected_for_app"] = best_arch
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[{a.stage}] app model = {best_arch} -> {P.WEIGHTS / f'occlusal_{a.stage}.pt'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["seg", "type", "severity", "det", "toothseg", "cariesdet"])
    ap.add_argument("--data", default=str(OCC))
    ap.add_argument("--arch", nargs="+", default=["efficientnet_b0", "mobilenet_v3"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    # seg
    ap.add_argument("--seg-model", default="yolov8s-seg.pt")
    ap.add_argument("--det-model", default="yolov8s.pt")
    ap.add_argument("--seg-imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    # classifiers
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--cls-epochs", type=int, default=40)
    ap.add_argument("--batch-cls", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--max-folds", type=int, default=0, help="evaluate only the first N folds (0 = all)")
    ap.add_argument("--no-final", action="store_true", help="skip the extra all-data retrain; ship the fold model")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    import torch
    torch.manual_seed(a.seed)
    {"seg": train_seg, "det": train_det, "toothseg": train_toothseg,
     "cariesdet": train_cariesdet}.get(a.stage, train_cls)(a)


if __name__ == "__main__":
    main()
