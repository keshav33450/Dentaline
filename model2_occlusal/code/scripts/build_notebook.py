#!/usr/bin/env python3
"""Build notebooks/train_kaggle.ipynb with the whole repo embedded (single-file upload to Kaggle)."""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["dentassist", "scripts", "api", "tests", "requirements.txt", "README.md"]


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in INCLUDE:
            p = ROOT / name
            if p.exists():
                tf.add(p, arcname=f"dentassist-opg/{name}",
                       filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md("""
# DentAssist - OPG models (Model 1 + Model 5)

Trains on **DENTEX** (MICCAI 2023, ~2.3k panoramic X-rays, FDI-labelled):

| Model | What it finds | Base |
|---|---|---|
| **Model 1 - teeth** | every tooth + FDI number (11-48) -> dental chart, missing teeth | YOLO11s @1024 |
| **Model 5 - findings** | caries, deep caries, periapical lesion, impacted tooth | YOLO11m @1280 |

**Before running:** right sidebar -> *Accelerator* = **GPU T4 x2**, *Internet* = **On** (needs phone-verified account).
Then **Save Version -> Save & Run All (Commit)** so it keeps running with the browser closed.

Total time ~4-6 h (both models train in parallel on the two GPUs). Outputs land in the *Output* tab.

Data licence: CC BY-NC-SA 4.0 - non-commercial research only. Cite DENTEX (Hamamci et al., 2023).
"""),
    code("""
# 0) YOUR PROJECT LOCATION  <- edit these two lines if you want (blank = automatic)
PROJECT_DIR = ""   # where code, weights, runs, results are saved.  auto: /kaggle/working (Kaggle) | /content/drive/MyDrive/dentassist (Colab) | current folder
DATA_DIR    = ""   # where the 12 GB dataset is downloaded/converted. auto: disk with most free space
"""),
    code("""
# 1) Unpack project code (embedded) + install deps
import base64, io, os, tarfile, subprocess, sys, shutil
if not PROJECT_DIR:
    if os.path.isdir("/kaggle/working"):
        PROJECT_DIR = "/kaggle/working"
    elif os.path.isdir("/content"):
        try:
            from google.colab import drive; drive.mount("/content/drive")
            PROJECT_DIR = "/content/drive/MyDrive/dentassist"
        except Exception:
            PROJECT_DIR = "/content"
    else:
        PROJECT_DIR = os.getcwd()
os.makedirs(PROJECT_DIR, exist_ok=True)
WORK = PROJECT_DIR
os.environ["DENTASSIST_HOME"] = WORK
BUNDLE = "__BUNDLE__"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(WORK)
REPO = f"{WORK}/dentassist-opg"
open(f"{REPO}/project_location.txt", "w").write(WORK)
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.0", "onnx", "onnxslim",
                "onnxruntime", "huggingface_hub", "pydicom"], check=True)
print("project location:", WORK)
print("code            :", REPO)
"""),
    code("""
# 2) Hardware + disk check
import torch, shutil
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
assert n_gpu > 0, "No GPU! Settings -> Accelerator -> GPU T4 x2"
if not DATA_DIR:
    cands = [p for p in ["/tmp", "/kaggle/temp", WORK] if os.path.isdir(p)]
    free = {p: shutil.disk_usage(p).free / 1e9 for p in cands}
    print({k: f"{v:.0f} GB free" for k, v in free.items()})
    DATA_DIR = max(free, key=free.get) + "/dentassist_data"
os.makedirs(DATA_DIR, exist_ok=True)
fr = shutil.disk_usage(DATA_DIR).free / 1e9
RAW, DATA = f"{DATA_DIR}/raw", f"{DATA_DIR}/yolo"
print(f"dataset location: {DATA_DIR}  ({fr:.0f} GB free)")
if fr < 16:
    print("WARNING: <16 GB free - download may fail. Set DATA_DIR to a bigger disk.")
print(f"outputs -> {WORK}/weights, {WORK}/runs, {WORK}/results")
"""),
    code("""
# 3) Download DENTEX (~12 GB) and convert to YOLO format (~15-25 min)
!python scripts/prepare_data.py --raw {RAW} --out {DATA} --delete-raw
"""),
    code("""
# 4) Train both models. 2 GPUs -> in parallel; 1 GPU -> one after another.
#    Progress is printed every 3 min. If the session dies, re-run with RESUME=True.
import subprocess, time, csv, pathlib
RESUME = False
def cmd(task, dev):
    c = [sys.executable, "scripts/train.py", "--task", task, "--data", DATA, "--device", str(dev),
         "--project", f"{WORK}/runs", "--weights-out", f"{WORK}/weights", "--workers", "2"]
    return c + (["--resume"] if RESUME else [])

def progress(task):
    f = pathlib.Path(f"{WORK}/runs/{task}/results.csv")
    if not f.exists(): return f"{task}: starting..."
    rows = list(csv.DictReader(open(f)))
    if not rows: return f"{task}: epoch 0"
    r = {k.strip(): v for k, v in rows[-1].items()}
    return f"{task}: epoch {r['epoch']}  mAP50={float(r['metrics/mAP50(B)']):.3f}  mAP50-95={float(r['metrics/mAP50-95(B)']):.3f}"

jobs = [("teeth", 0), ("findings", 1 if n_gpu > 1 else 0)]
t0 = time.time()
if n_gpu > 1:
    procs = [(t, subprocess.Popen(cmd(t, d), stdout=open(f"{WORK}/train_{t}.log", "w"), stderr=subprocess.STDOUT)) for t, d in jobs]
    while any(p.poll() is None for _, p in procs):
        time.sleep(180)
        print(f"[{(time.time()-t0)/60:5.0f} min] " + " | ".join(progress(t) for t, _ in procs), flush=True)
    for t, p in procs:
        print(t, "exit code", p.returncode)
        if p.returncode: print(open(f"{WORK}/train_{t}.log").read()[-3000:])
else:
    for t, d in jobs:
        subprocess.run(cmd(t, d), check=True)
print("training done in %.1f h" % ((time.time() - t0) / 3600))
"""),
    code("""
# 5) Evaluate on the held-out TEST split (per-model mAP + end-to-end 'right tooth, right diagnosis')
!python scripts/evaluate.py --data {DATA} --teeth {WORK}/weights/teeth_best.pt --findings {WORK}/weights/findings_best.pt --out {WORK}/results --device 0
"""),
    code("""
# 6) Export for clinic PCs: ONNX (+ OpenVINO INT8 for Intel CPUs) and CPU speed benchmark
!pip install -q openvino
!python scripts/export.py --teeth {WORK}/weights/teeth_best.pt --findings {WORK}/weights/findings_best.pt --data {DATA} --out {WORK}/weights --int8
"""),
    code("""
# 7) Demo on a test X-ray the models never saw
import glob, cv2, json, matplotlib.pyplot as plt
sys.path.insert(0, REPO)
from dentassist.analyzer import OPGAnalyzer
a = OPGAnalyzer(f"{WORK}/weights/teeth_best.pt", f"{WORK}/weights/findings_best.pt")
for p in sorted(glob.glob(f"{DATA}/findings/images/test/*.jpg"))[:3]:
    r = a.analyze(p)
    plt.figure(figsize=(18, 8)); plt.imshow(cv2.cvtColor(a.draw(p, r), cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()
    print(json.dumps(r["summary"], indent=1))
    for f in r["findings"]:
        print(f"  tooth {f['fdi']}: {f['label']} ({f['confidence']:.2f}){'  <- review' if f['needs_review'] else ''}")
"""),
    code("""
# 8) Package everything you need -> Output tab -> dentassist_opg_outputs.zip
import shutil
pkg = f"{WORK}/package"
shutil.rmtree(pkg, ignore_errors=True); os.makedirs(pkg)
shutil.copytree(f"{WORK}/weights", f"{pkg}/weights")
shutil.copytree(f"{WORK}/results", f"{pkg}/results", ignore=shutil.ignore_patterns("*.cache"))
for t in ("teeth", "findings"):
    run = f"{WORK}/runs/{t}"
    if os.path.isdir(run):
        shutil.copytree(run, f"{pkg}/training_curves_{t}", ignore=shutil.ignore_patterns("weights"))
shutil.make_archive(f"{WORK}/dentassist_opg_outputs", "zip", pkg)
shutil.rmtree(pkg)
print(open(f"{WORK}/results/REPORT.md").read())
print("\\nDownload: Output tab -> dentassist_opg_outputs.zip")
"""),
    md("""
## Next
1. Download **dentassist_opg_outputs.zip** (weights + `results/REPORT.md` + plots + error gallery).
2. Send the `REPORT.md` numbers back to Claude for review.
3. Run the API locally: unzip `weights/` into the repo, then `uvicorn api.main:app --port 8000`.
"""),
]


def main():
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"},
                                       "accelerator": "GPU", "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    b = bundle()
    for c in nb["cells"]:
        c["source"] = [s.replace("__BUNDLE__", b) for s in c["source"]]
    out = ROOT / "notebooks" / "train_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
