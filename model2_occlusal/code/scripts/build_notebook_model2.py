#!/usr/bin/env python3
"""Build notebooks/train_model2_kaggle.ipynb (Model 2: smartphone occlusal caries) with the code embedded."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import ROOT, bundle, code, md  # noqa: E402

CELLS = [
    md("""
# DentAssist - Model 2: smartphone occlusal caries (permanent premolars & molars)

| Part | Data | Model |
|---|---|---|
| Caries **severity** No/Mild/Moderate/Advanced | Roboflow *Caries Classification ICDAS II* (ICDAS 0-6) | EfficientNet-B0 vs MobileNetV3 + YOLOv8 severity detector |
| Caries **screening** on real phone photos | Zenodo 14769743 (6,313 smartphone photos, permanent-caries boxes) | YOLOv8 caries detector |
| Tooth outline + **premolar/molar** | SegmentAnyTooth labels the Zenodo occlusal photos automatically | YOLOv8-seg (premolar/molar) + type classifier |
| **External validation** | Zenodo held-out photos (other country / camera) | severity model tested there |

**Settings before running:** GPU **T4 x2**, Internet **On**. Add-ons -> Secrets: `ROBOFLOW_API_KEY`.
*Add Data* (optional):
- your **SegmentAnyTooth weights** (private dataset with `segmentanytooth_*.pt`) -> enables tooth outline + premolar/molar
- your earlier **model2_outputs** weights -> the severity part is not retrained
- the KGMU **study dataset** (photos + CVAT json + ICDAS sheet) -> STUDY mode

Then **Save Version -> Save & Run All** (~3-5 h).
"""),
    code("""
# 0) SETTINGS
PROJECT_DIR = ""                 # auto: /kaggle/working
ROBOFLOW_API_KEY = ""            # or Kaggle secret ROBOFLOW_API_KEY
ROBOFLOW_DATASET = "code-geass/caries-classification-icdas-ii"
TRAIN_SEVERITY = "auto"          # auto = only if no earlier occlusal_severity.pt is attached
USE_ZENODO = True                # smartphone caries detector + external validation
USE_SEGMENTANYTOOTH = "auto"     # auto = if its weights are attached
CLS_EPOCHS = 25
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.0", "openpyxl", "scikit-learn",
                "scipy", "roboflow", "segment-anything-hq"], check=True)
import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
assert n_gpu > 0, "No GPU! Settings -> Accelerator -> GPU T4 x2"
"""),
    code("""
# 2) What is attached?
sys.path.insert(0, REPO)
from dentassist.occlusal.sat import find_weights
IMG_EXT = (".jpg", ".jpeg", ".png")
if not ROBOFLOW_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        ROBOFLOW_API_KEY = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    except Exception:
        pass

def find_study(root="/kaggle/input"):
    coco = clinical = photos = None; best = 0
    for p in glob.glob(f"{root}/**/*.json", recursive=True):
        try:
            j = json.load(open(p))
            if isinstance(j, dict) and "annotations" in j and "images" in j and any(
                    "fdi" in (a.get("attributes") or {}) for a in j["annotations"][:50]):
                coco = p; break
        except Exception:
            pass
    for p in glob.glob(f"{root}/**/*", recursive=True):
        n = os.path.basename(p).lower()
        if n.endswith((".csv", ".xlsx")) and ("icdas" in n or "clinical" in n):
            clinical = p; break
    for d, _, fs in os.walk(root):
        k = sum(f.lower().endswith(IMG_EXT) for f in fs)
        if k > best: best, photos = k, d
    return coco, clinical, photos

coco, clinical, photos = find_study()
MODE = "STUDY" if coco else "PUBLIC"
os.makedirs(f"{WORK}/weights", exist_ok=True)
prev = [p for p in glob.glob("/kaggle/input/**/occlusal_*.pt", recursive=True)]
for p in prev:                                   # reuse earlier trained weights
    shutil.copy2(p, f"{WORK}/weights/")
have = {os.path.basename(p) for p in prev}
if TRAIN_SEVERITY == "auto":
    TRAIN_SEVERITY = MODE == "PUBLIC" and "occlusal_severity.pt" not in have
if TRAIN_SEVERITY and not ROBOFLOW_API_KEY:
    print("!! no ROBOFLOW_API_KEY -> severity part skipped"); TRAIN_SEVERITY = False
SAT_W = find_weights("/kaggle/input")
if USE_SEGMENTANYTOOTH == "auto":
    USE_SEGMENTANYTOOTH = SAT_W is not None
if USE_SEGMENTANYTOOTH:
    if not os.path.isdir(f"{WORK}/SegmentAnyTooth"):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/thangngoc89/SegmentAnyTooth",
                        f"{WORK}/SegmentAnyTooth"], check=True)
print(f"MODE={MODE} | severity={'train' if TRAIN_SEVERITY else ('reuse' if 'occlusal_severity.pt' in have else 'off')} | "
      f"zenodo={USE_ZENODO} | SegmentAnyTooth={'ON ' + str(SAT_W) if USE_SEGMENTANYTOOTH else 'off (attach weights to enable)'}")
print("reused weights:", sorted(have) or "none")
"""),
    code("""
# 3) Data
import matplotlib.pyplot as plt, cv2
def run(*cmd):
    print(">>", " ".join(cmd[:3]), flush=True); subprocess.run([sys.executable, *cmd], check=True)
if MODE == "STUDY":
    run("scripts/occ_prepare.py", "--images", photos, "--coco", coco, *(["--clinical", clinical] if clinical else []))
else:
    if TRAIN_SEVERITY:
        ws, pr = ROBOFLOW_DATASET.split("/")[:2]
        run("scripts/occ_public_data.py", "--api-key", ROBOFLOW_API_KEY, "--workspace", ws, "--project", pr)
    if USE_ZENODO:
        try:
            run("scripts/occ_zenodo_data.py")
        except subprocess.CalledProcessError:
            USE_ZENODO = False
            print("\n!!!! Zenodo data FAILED (see message above) -> continuing WITHOUT Zenodo.\n"
                  "!!!! Fix: download the dataset from zenodo.org/records/14769743, upload it as a private Kaggle\n"
                  "!!!! dataset, Add Data -> it is found automatically on the next run.\n", flush=True)
    samples = sorted(glob.glob(f"{WORK}/data/occlusal/samples/*.jpg"))[:3] + \\
              sorted(glob.glob(f"{WORK}/data/occlusal/zenodo/det/images/train/*.jpg"))[:3]
    if samples:
        fig, ax = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
        for a_, f in zip(ax if len(samples) > 1 else [ax], samples):
            a_.imshow(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)); a_.axis("off")
        plt.suptitle("left: Roboflow ICDAS  |  right: Zenodo smartphone"); plt.show()
for f in ("data/occlusal/summary.json", "data/occlusal/zenodo/summary.json"):
    if os.path.exists(f"{WORK}/{f}"):
        print(f, open(f"{WORK}/{f}").read())
"""),
    code("""
# 4) Train - two GPU queues in parallel
T = "scripts/occ_train.py"
big = ["--max-folds", "1", "--no-final", "--patience", "5", "--workers", "3", "--cls-epochs", str(CLS_EPOCHS)]
g0, g1 = [], []
if MODE == "STUDY":
    g0 += [[T, "--stage", "seg", "--device", "0"]]
    g1 += [[T, "--stage", "type", "--device", "1"], [T, "--stage", "severity", "--device", "1"]]
else:
    if TRAIN_SEVERITY:
        g0 += [[T, "--stage", "det", "--device", "0", "--epochs", "100"]]
        g1 += [[T, "--stage", "severity", "--device", "1", *big]]
    if USE_ZENODO:
        g0 += [[T, "--stage", "cariesdet", "--device", "0"]]
    if USE_SEGMENTANYTOOTH and USE_ZENODO:
        g1 += [["scripts/occ_sat_pseudolabel.py", "--sat-repo", f"{WORK}/SegmentAnyTooth", "--sat-weights", str(SAT_W)],
               [T, "--stage", "toothseg", "--device", "1", "--epochs", "100"],
               [T, "--stage", "type", "--device", "1", "--data", f"{WORK}/data/occlusal/zenodo/typeset", *big]]
if n_gpu == 1:
    to0 = lambda cmd: [("0" if i and cmd[i - 1] == "--device" else c) for i, c in enumerate(cmd)]
    g0, g1 = g0 + [to0(cmd) for cmd in g1], []
status = {}
def worker(name, cmds):
    log = open(f"{WORK}/train_{name}.log", "w")
    for c in cmds:
        status[name] = " ".join(c[:3]); log.write(f"\\n### {c}\\n"); log.flush()
        r = subprocess.run([sys.executable, *c], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            status[name] = f"FAILED: {' '.join(c[:3])} (see train_{name}.log)"; return
    status[name] = "done"
threads = [threading.Thread(target=worker, args=(n, c)) for n, c in (("gpu0", g0), ("gpu1", g1)) if c]
t0 = time.time(); [t.start() for t in threads]
while any(t.is_alive() for t in threads):
    time.sleep(120); print(f"[{(time.time()-t0)/60:4.0f} min] {status}", flush=True)
print("finished:", status)
for n in ("gpu0", "gpu1"):
    if "FAILED" in status.get(n, ""):
        print(open(f"{WORK}/train_{n}.log").read()[-4000:])
"""),
    code("""
# 5) Statistics + external validation
from IPython.display import Markdown, display
subprocess.run([sys.executable, "scripts/occ_evaluate.py"])
if USE_ZENODO and MODE == "PUBLIC":
    subprocess.run([sys.executable, "scripts/occ_external_validation.py", "--device", "0"])
for f in ("OCCLUSAL_REPORT.md", "EXTERNAL_VALIDATION.md"):
    p = f"{WORK}/results/occlusal/{f}"
    if os.path.exists(p):
        display(Markdown(open(p).read().split("![cm]")[0]))
"""),
    code("""
# 6) Look at predictions
from dentassist import paths as P
from dentassist.occlusal.pipeline import load_default
an = load_default(P.WEIGHTS)
test = sorted(glob.glob(f"{WORK}/data/occlusal/zenodo/det/images/test/*.jpg")) or \\
       sorted(glob.glob(f"{WORK}/data/occlusal/seg/images/val/*.jpg"))
if an:
    for p in test[:4]:
        r = an.analyze(cv2.imread(p))
        plt.figure(figsize=(11, 8)); plt.imshow(cv2.cvtColor(an.draw(r), cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()
        print(r["summary"])
else:
    from ultralytics import YOLO
    for w in ("occlusal_det.pt", "occlusal_caries_det.pt"):
        if os.path.exists(f"{P.WEIGHTS}/{w}"):
            m = YOLO(f"{P.WEIGHTS}/{w}")
            for p in test[:2]:
                plt.figure(figsize=(10, 7)); plt.imshow(cv2.cvtColor(m.predict(p, conf=0.3, verbose=False)[0].plot(), cv2.COLOR_BGR2RGB))
                plt.title(w); plt.axis("off"); plt.show()
qc = sorted(glob.glob(f"{WORK}/data/occlusal/zenodo/pseudolabel_qc/*.jpg"))[:3]
for p in qc:
    plt.figure(figsize=(9, 6)); plt.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)); plt.title("SegmentAnyTooth pseudo-labels"); plt.axis("off"); plt.show()
"""),
    code("""
# 7) Package -> Output tab -> model2_outputs.zip
pkg = f"{WORK}/model2_package"; shutil.rmtree(pkg, ignore_errors=True); os.makedirs(f"{pkg}/weights")
for f in glob.glob(f"{WORK}/weights/occlusal_*.pt"):
    shutil.copy2(f, f"{pkg}/weights/")
if os.path.isdir(f"{WORK}/results/occlusal"):
    shutil.copytree(f"{WORK}/results/occlusal", f"{pkg}/results")
os.makedirs(f"{pkg}/results", exist_ok=True)
for f in ("data/occlusal/summary.json", "data/occlusal/zenodo/summary.json", "data/occlusal/zenodo/pseudolabel_summary.json",
          "data/occlusal/exclusions.csv", "data/occlusal/image_quality.csv"):
    if os.path.exists(f"{WORK}/{f}"):
        shutil.copy2(f"{WORK}/{f}", f"{pkg}/results/{f.replace('/', '_').replace('data_occlusal_', '')}")
for d in ("occlusal_det", "occlusal_caries_det", "occlusal_toothseg", "occlusal_seg"):
    if os.path.isdir(f"{WORK}/runs/{d}"):
        shutil.copytree(f"{WORK}/runs/{d}", f"{pkg}/training_{d}", ignore=shutil.ignore_patterns("weights"))
qcd = f"{WORK}/data/occlusal/zenodo/pseudolabel_qc"
if os.path.isdir(qcd):
    shutil.copytree(qcd, f"{pkg}/results/segmentanytooth_qc")
shutil.make_archive(f"{WORK}/model2_outputs", "zip", pkg); shutil.rmtree(pkg)
print(sorted(os.listdir(f"{WORK}/weights")))
print("Download: Output tab -> model2_outputs.zip")
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
    out = ROOT / "notebooks" / "train_model2_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
