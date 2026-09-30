#!/usr/bin/env python3
"""Build notebooks/train_model1_v2_kaggle.ipynb - Model 1 improvements (tight outlines + calibration + validation)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import ROOT, bundle, code, md  # noqa: E402

CELLS = [
    md("""
# DentAssist - Model 1 v2 (panoramic X-ray): tighter outlines, calibrated thresholds, ground-truth validation

1. **Tooth segmentation** (YOLO11s-seg on DENTEX tooth outlines) -> tight tooth boxes/outlines
2. **Calibration on the validation split**: preprocessing A/B (none vs denoise+CLAHE) and per-class thresholds
3. **Ground-truth validation on the test split**: FP/FN per finding, 'likely' vs 'possible', numbering accuracy
   with/without the anatomical FDI repair
4. Writes `thresholds.json` - the app reads it automatically

**Before running:** *Add Data* -> **Notebook Output** of your first Model 1 run (so the detectors are reused),
GPU **T4 x2**, Internet **On** -> **Save Version -> Save & Run All** (~3-4 h).
"""),
    code("""
# 0) SETTINGS
PROJECT_DIR = ""            # auto: /kaggle/working
TRAIN_TEETH_SEG = True      # tight tooth outlines (~2-3 h)
"""),
    code("""
# 1) Unpack code + install
import base64, io, os, tarfile, subprocess, sys, glob, json, shutil, threading, time
PROJECT_DIR = PROJECT_DIR or ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())
WORK = PROJECT_DIR; os.makedirs(WORK, exist_ok=True)
os.environ["DENTASSIST_HOME"] = WORK; os.environ["YOLO_AUTOINSTALL"] = "false"
BUNDLE = "__BUNDLE__"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(WORK)
REPO = f"{WORK}/dentassist-opg"; open(f"{REPO}/project_location.txt", "w").write(WORK); os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.0", "huggingface_hub", "pydicom"], check=True)
import torch
n_gpu = torch.cuda.device_count(); print("GPUs:", n_gpu)
assert n_gpu > 0, "No GPU! Settings -> Accelerator -> GPU T4 x2"
os.makedirs(f"{WORK}/weights", exist_ok=True)
found = {}
for n in ("teeth_best.pt", "findings_best.pt"):
    c = sorted(glob.glob(f"/kaggle/input/**/{n}", recursive=True))
    if c:
        shutil.copy2(c[0], f"{WORK}/weights/{n}"); found[n] = c[0]
print("reused detectors:", found or "NONE - they will be retrained (+4 h)")
free = {p: shutil.disk_usage(p).free / 1e9 for p in ["/tmp", WORK] if os.path.isdir(p)}
DATA_DIR = max(free, key=free.get) + "/dentassist_data"; RAW, DATA = f"{DATA_DIR}/raw", f"{DATA_DIR}/yolo"
print("data ->", DATA)
"""),
    code("""
# 2) DENTEX download + conversion (now also tooth outlines + validation ground truth) ~20 min
!python scripts/prepare_data.py --raw {RAW} --out {DATA} --delete-raw
"""),
    code("""
# 3) Train: tooth segmentation (GPU0) + any missing detector (GPU1)
def job(cmd, log):
    return subprocess.Popen([sys.executable, *cmd], stdout=open(log, "w"), stderr=subprocess.STDOUT)
base = ["--data", DATA, "--project", f"{WORK}/runs", "--weights-out", f"{WORK}/weights", "--workers", "2"]
q0 = [["scripts/train.py", "--task", "teeth_seg", "--device", "0", *base]] if TRAIN_TEETH_SEG else []
q1 = [["scripts/train.py", "--task", t, "--device", "1" if n_gpu > 1 else "0", *base]
      for t, w in (("teeth", "teeth_best.pt"), ("findings", "findings_best.pt")) if w not in found]
if n_gpu == 1:
    q0, q1 = q0 + q1, []
status = {}
def worker(name, cmds):
    for c in cmds:
        status[name] = c[2]
        r = subprocess.run([sys.executable, *c], stdout=open(f"{WORK}/train_{name}.log", "a"), stderr=subprocess.STDOUT)
        if r.returncode:
            status[name] = f"FAILED {c[2]}"; return
    status[name] = "done"
th = [threading.Thread(target=worker, args=(n, q)) for n, q in (("gpu0", q0), ("gpu1", q1)) if q]
t0 = time.time(); [t.start() for t in th]
while any(t.is_alive() for t in th):
    time.sleep(180); print(f"[{(time.time()-t0)/60:4.0f} min] {status}", flush=True)
print("done:", status, sorted(os.listdir(f"{WORK}/weights")))
"""),
    code("""
# 4) Segmentation test metrics (box + mask)
from ultralytics import YOLO
if os.path.exists(f"{WORK}/weights/teeth_seg_best.pt") and os.path.exists(f"{DATA}/teeth_seg/data.yaml"):
    v = YOLO(f"{WORK}/weights/teeth_seg_best.pt").val(data=f"{DATA}/teeth_seg/data.yaml", split="test", imgsz=1024, plots=False)
    seg = {"box_mAP50": round(float(v.box.map50), 4), "mask_mAP50": round(float(v.seg.map50), 4),
           "mask_mAP50-95": round(float(v.seg.map), 4)}
    os.makedirs(f"{WORK}/results", exist_ok=True)
    json.dump(seg, open(f"{WORK}/results/teeth_seg_test.json", "w"), indent=2); print(seg)
"""),
    code("""
# 5) Calibration (validation) + ground-truth validation (test)  -> weights/thresholds.json, results/CALIBRATION.md
!python scripts/calibrate.py --data {DATA} --device 0
from IPython.display import Markdown, display
display(Markdown(open(f"{WORK}/results/CALIBRATION.md").read()))
"""),
    code("""
# 6) Standard test metrics + error gallery (detector mAP, end-to-end)
!python scripts/evaluate.py --data {DATA} --teeth {WORK}/weights/teeth_best.pt --findings {WORK}/weights/findings_best.pt --out {WORK}/results --device 0
"""),
    code("""
# 7) Look at the new result images (test X-rays never seen in training)
import cv2, matplotlib.pyplot as plt
sys.path.insert(0, REPO)
from dentassist.analyzer import OPGAnalyzer
tw = f"{WORK}/weights/teeth_seg_best.pt" if os.path.exists(f"{WORK}/weights/teeth_seg_best.pt") else f"{WORK}/weights/teeth_best.pt"
an = OPGAnalyzer(tw, f"{WORK}/weights/findings_best.pt", weights_dir=f"{WORK}/weights")
os.makedirs(f"{WORK}/results/examples", exist_ok=True)
for p in sorted(glob.glob(f"{DATA}/findings/images/test/*.jpg"))[:6]:
    r = an.analyze(p); vis = an.draw(p, r)
    cv2.imwrite(f"{WORK}/results/examples/{os.path.basename(p)}", vis)
    plt.figure(figsize=(20, 7)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()
"""),
    code("""
# 8) Package -> Output tab -> model1_v2_outputs.zip
pkg = f"{WORK}/pkg"; shutil.rmtree(pkg, ignore_errors=True); os.makedirs(f"{pkg}/weights")
for n in ("teeth_seg_best.pt", "thresholds.json", "teeth_best.pt", "findings_best.pt"):
    if os.path.exists(f"{WORK}/weights/{n}"):
        shutil.copy2(f"{WORK}/weights/{n}", f"{pkg}/weights/")
shutil.copytree(f"{WORK}/results", f"{pkg}/results", ignore=shutil.ignore_patterns("*.cache"))
if os.path.isdir(f"{WORK}/runs/teeth_seg"):
    shutil.copytree(f"{WORK}/runs/teeth_seg", f"{pkg}/training_curves_teeth_seg", ignore=shutil.ignore_patterns("weights"))
shutil.make_archive(f"{WORK}/model1_v2_outputs", "zip", pkg); shutil.rmtree(pkg)
print(open(f"{WORK}/results/CALIBRATION.md").read()); print("Download: Output tab -> model1_v2_outputs.zip")
"""),
]


def main():
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"},
                                       "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    b = bundle()
    for c in nb["cells"]:
        c["source"] = [s.replace("__BUNDLE__", b) for s in c["source"]]
    out = ROOT / "notebooks" / "train_model1_v2_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
