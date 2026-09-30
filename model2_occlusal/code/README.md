# DentAssist OPG: Model 1 + Model 5

AI decision support for **panoramic dental X-rays (OPG)**. Two YOLO11 detectors plus clinical post-processing:

| | Model 1: teeth | Model 5: findings |
|---|---|---|
| Output | every tooth + **FDI number 11–48** | **caries, deep caries, periapical lesion, impacted** |
| Base / input | YOLO11s @ 1024 | YOLO11m @ 1280 |
| Training data | DENTEX quadrant-enumeration set (634 X-rays) | DENTEX diagnosis set (705 train / 50 val / 250 test) |
| Augmentation | **no flips, no mosaic**, because flips swap left/right and upper/lower FDI numbers | flips + mosaic are fine, since the diagnosis doesn't depend on side |

The **analyzer** combines both models:
- It keeps one box per FDI number and resolves overlapping or duplicate numbers.
- It maps each finding to a tooth, e.g. *"36: deep caries"*.
- It lists teeth that were **not detected** (possible missing teeth, which are implant candidates for Model 3), with third molars listed separately.
- It marks findings for **review** when confidence is low or no tooth could be assigned, so the doctor stays in the loop.
- It returns a 32-tooth **chart**, an annotated overlay, timings and a disclaimer.

## Train on Kaggle (recommended)
1. Upload `notebooks/train_kaggle.ipynb`. The whole code is embedded, so it's a single file.
2. Set **Accelerator: GPU T4 x2** and **Internet: On**.
3. Choose **Save Version → Save & Run All**. Training takes about 4–6 h, with both models in parallel on the two GPUs.
4. From the **Output** tab, download `dentassist_opg_outputs.zip`. It contains the weights (`.pt`, `.onnx`, OpenVINO INT8), `results/REPORT.md`, plots and an error gallery.

## Run locally / step by step
```bash
pip install -r requirements.txt
python scripts/prepare_data.py --raw /tmp/dentex_raw --out /tmp/dentex_yolo --delete-raw
python scripts/train.py --task teeth    --data /tmp/dentex_yolo --device 0
python scripts/train.py --task findings --data /tmp/dentex_yolo --device 0      # or --device 1 in parallel
python scripts/evaluate.py --data /tmp/dentex_yolo --out results
python scripts/export.py --data /tmp/dentex_yolo --int8
```
If a run is interrupted, add `--resume` to the train command.

## Inference API (FastAPI)
```bash
# put teeth.onnx + findings.onnx (or *_best.pt) in ./weights
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -F "file=@opg.jpg" "http://localhost:8000/predict/opg?overlay=true"
```
- Accepts JPG, PNG and **DICOM (.dcm)**.
- Returns `teeth[]`, `findings[]` (diagnosis, FDI, box, confidence, needs_review), `summary`, `chart`, `overlay_jpeg_base64` and `timing_ms`.
- Docker: `docker build -f api/Dockerfile -t dentassist-opg . && docker run -p 8000:8000 -v $PWD/weights:/app/weights dentassist-opg`
- Env vars: `TEETH_MODEL`, `FINDINGS_MODEL`, `FINDINGS_CONF` (0.25), `REVIEW_BELOW` (0.5), `CORS_ORIGINS`.

## Metrics reported (`results/REPORT.md`)
- Per model on the held-out **test** split: mAP50, mAP50-95, precision, recall, per class, and confusion matrix.
- **End-to-end clinical score:** a finding counts only if the **box, diagnosis and tooth number are all correct**. This is what a dentist cares about.
- Latency per image and CPU speed of ONNX vs OpenVINO INT8.

Realistic DENTEX targets:
- Teeth mAP50: about 0.90+
- Findings mAP50: about 0.45–0.65, with caries the hardest and impacted the easiest

Ignore 99 % claims in single-clinic papers.

## Data leakage guards
- MD5 de-duplication across all sets: the same X-ray never appears in both train and test.
- The official DENTEX val/test sets are used when labelled; otherwise the split is carved from train with a fixed seed.

## Layout
```
dentassist/fdi.py        FDI classes, tooth names, finding names
dentassist/analyzer.py   OPGAnalyzer: both models -> clinical JSON + overlay
scripts/prepare_data.py  DENTEX download -> 2 YOLO datasets + end-to-end GT
scripts/train.py         X-ray-safe presets per task
scripts/evaluate.py      test metrics + clinical metric + error gallery
scripts/export.py        ONNX / OpenVINO INT8 + CPU benchmark
scripts/build_notebook.py  regenerate the Kaggle notebook after code changes
api/                     FastAPI service + Dockerfile
tests/                   fake-DENTEX generator + logic tests
```

## Limits
- **Research use only.** Clinical use in India needs CDSCO approval as software as a medical device (SaMD).
- DENTEX comes from 3 institutions outside India, so validate on local OPGs before trusting the numbers.
- DENTEX has no crown or restoration labels. Add them later with Roboflow labels if needed.
- The DENTEX licence is **CC BY-NC-SA 4.0**, which means non-commercial use only. A commercial product needs your own data.

Citation: Hamamci et al., *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays*, arXiv:2305.19112 (2023).
