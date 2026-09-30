#!/usr/bin/env python3
"""
Train one of the two OPG models.

  --task teeth     Model 1: 32-class FDI tooth detector   (no flips, no mosaic: anatomy/side matter)
  --task findings  Model 5: caries / deep caries / periapical lesion / impacted

Examples
  python scripts/train.py --task teeth    --data /tmp/dentex_yolo --device 0
  python scripts/train.py --task findings --data /tmp/dentex_yolo --device 1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402

# X-ray-safe hyper-parameters. Grayscale -> no hue/saturation jitter.
PRESETS = {
    "teeth": dict(
        model="yolo11s.pt", imgsz=1024, epochs=150, batch=8, patience=30,
        fliplr=0.0,          # flipping swaps left/right quadrants -> wrong FDI numbers
        flipud=0.0,          # flipping swaps upper/lower jaw -> wrong FDI numbers
        mosaic=0.0,          # keep whole-mouth context; numbering depends on neighbours
        mixup=0.0, copy_paste=0.0,
        degrees=3.0, translate=0.05, scale=0.25, shear=0.0, perspective=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.35,
        lr0=0.01, cos_lr=True, close_mosaic=0,
    ),
    "findings": dict(
        model="yolo11m.pt", imgsz=1280, epochs=150, batch=8, patience=30,
        fliplr=0.5,          # diagnosis does not depend on side -> flip is fine here
        flipud=0.0,
        mosaic=0.5, close_mosaic=20,
        mixup=0.0, copy_paste=0.0,
        degrees=3.0, translate=0.08, scale=0.3, shear=0.0, perspective=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.35,
        lr0=0.01, cos_lr=True,
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=PRESETS)
    ap.add_argument("--data", default=str(P.DATA), help="root written by prepare_data.py")
    ap.add_argument("--model", help="override base weights / yaml (e.g. yolo11m.pt, yolo11s.yaml)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--device", default="auto", help="auto | 0 | 1 | 0,1 | cpu")
    ap.add_argument("--fast", action="store_true",
                    help="CPU / small-GPU preset: yolo11n, smaller images, fewer epochs (lower accuracy)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache", default="ram", help="ram | disk | none")
    ap.add_argument("--project", default=str(P.RUNS))
    ap.add_argument("--weights-out", default=str(P.WEIGHTS), help="where best.pt is copied")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run (runs/<task>/weights/last.pt)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        print("[train] WARNING: no NVIDIA GPU found - training on CPU is VERY slow (days). "
              "Using --fast preset automatically.")
        args.fast = True
    else:
        gpu = torch.cuda.get_device_properties(int(str(args.device).split(",")[0]))
        vram = gpu.total_memory / 1e9
        print(f"[train] GPU: {gpu.name} ({vram:.0f} GB)")
        if args.batch is None and vram < 10:
            args.batch = 4 if vram >= 6 else 2
            print(f"[train] small GPU -> batch {args.batch}")

    cfg = dict(PRESETS[args.task])
    if args.fast:
        cfg.update(model="yolo11n.pt", imgsz=640 if args.task == "teeth" else 800, epochs=60, patience=20)
    base = args.model or cfg.pop("model")
    cfg.pop("model", None)
    for k in ("epochs", "imgsz", "batch", "patience"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    data_yaml = Path(args.data) / args.task / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found - run prepare_data.py first")

    device = args.device
    if "," in device:
        device = [int(d) for d in device.split(",")]

    last = Path(args.project) / args.task / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"[train] resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        print(f"[train] task={args.task} base={base} " + json.dumps(cfg))
        model = YOLO(base)
        model.train(
            data=str(data_yaml), device=device, workers=args.workers,
            cache=False if args.cache == "none" else args.cache,
            amp=args.device != "cpu",
            project=str(Path(args.project).resolve()), name=args.task, exist_ok=True, seed=args.seed,
            plots=True, verbose=True, **cfg,
        )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    out = Path(args.weights_out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.task}_best.pt"
    shutil.copy2(best, dst)
    print(f"[train] done -> {dst}  (run dir: {run_dir})")


if __name__ == "__main__":
    main()
