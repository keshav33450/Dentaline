#!/usr/bin/env python3
"""
Export trained models for clinic deployment and benchmark CPU speed.

  ONNX (FP32)         -> runs anywhere with onnxruntime (default)
  OpenVINO INT8 (opt) -> ~2-3x faster on Intel CPUs typical of clinic PCs  (--int8)

  python scripts/export.py --teeth weights/teeth_best.pt --findings weights/findings_best.pt \
         --data /tmp/dentex_yolo [--int8]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402


def bench(weights: str, imgsz: int, sample: str | None, n: int = 10) -> dict:
    from ultralytics import YOLO
    m = YOLO(weights, task="detect")
    img = sample if sample else np.full((imgsz // 2, imgsz, 3), 127, np.uint8)
    m.predict(img, imgsz=imgsz, device="cpu", verbose=False)  # warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        m.predict(img, imgsz=imgsz, device="cpu", verbose=False)
        ts.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": round(float(np.mean(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teeth", default=str(P.WEIGHTS / "teeth_best.pt"))
    ap.add_argument("--findings", default=str(P.WEIGHTS / "findings_best.pt"))
    ap.add_argument("--teeth-imgsz", type=int, default=1024)
    ap.add_argument("--findings-imgsz", type=int, default=1280)
    ap.add_argument("--data", default=str(P.DATA), help="needed for INT8 calibration + sample image")
    ap.add_argument("--int8", action="store_true", help="also export OpenVINO INT8 (pip install openvino)")
    ap.add_argument("--out", default=str(P.WEIGHTS))
    ap.add_argument("--bench-runs", type=int, default=10)
    args = ap.parse_args()

    from ultralytics import YOLO
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = sorted((Path(args.data) / "findings/images/test").glob("*.jpg"))
    sample = str(samples[0]) if samples else None
    report = {}

    for task, w, sz in (("teeth", args.teeth, args.teeth_imgsz), ("findings", args.findings, args.findings_imgsz)):
        if not Path(w).exists():
            print(f"[export] skip {task}: {w} missing")
            continue
        print(f"[export] {task}: ONNX @ {sz}")
        onnx_path = YOLO(w).export(format="onnx", imgsz=sz, simplify=True, dynamic=False, opset=17)
        dst = out / f"{task}.onnx"
        if Path(onnx_path).resolve() != dst.resolve():
            shutil.move(onnx_path, dst)
        r = {"onnx": str(dst), "onnx_mb": round(dst.stat().st_size / 1e6, 1),
             "cpu_pt": bench(w, sz, sample, args.bench_runs),
             "cpu_onnx": bench(str(dst), sz, sample, args.bench_runs)}
        if args.int8:
            try:
                print(f"[export] {task}: OpenVINO INT8")
                ov = YOLO(w).export(format="openvino", imgsz=sz, int8=True,
                                    data=str(Path(args.data) / task / "data.yaml"), fraction=0.5)
                ov_dst = out / f"{task}_int8_openvino_model"
                if ov_dst.exists():
                    shutil.rmtree(ov_dst)
                shutil.move(ov, ov_dst)
                r["openvino_int8"] = str(ov_dst)
                r["cpu_openvino_int8"] = bench(str(ov_dst), sz, sample, args.bench_runs)
            except Exception as e:
                r["openvino_int8_error"] = f"{e.__class__.__name__}: {e}"
        report[task] = r
        print(json.dumps(r, indent=2))

    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"[export] done -> {out}/export_report.json")


if __name__ == "__main__":
    main()
