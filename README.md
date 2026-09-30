# DentalX — Explainable Multi-Task Deep Learning for Dental Image Analysis

An explainable, multi-task deep-learning framework for automated dental image analysis and
clinical decision support. DentalX bundles three independent, production-ready models —
panoramic X-ray analysis, smartphone occlusal-caries screening, and intraoral tooth-type
detection — each trained on real dental datasets and runnable offline on a clinic PC.

> **Research prototype / AI screening aid — not a diagnostic device.**

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![Framework](https://img.shields.io/badge/framework-YOLO11%20%7C%20RF--DETR-green)
![License](https://img.shields.io/badge/code-MIT-lightgrey)
![Status](https://img.shields.io/badge/models-3%2F3%20trained-success)

---

## What is included

| Component | Description |
|-----------|-------------|
| **Model 1 — Panoramic X-ray (OPG)** | FDI tooth numbering + detection of caries, deep caries, periapical lesions, impacted teeth. |
| **Model 2 — Occlusal caries (smartphone)** | Caries detection on intraoral/occlusal photos + ICDAS-derived severity (No / Mild / Moderate / Advanced). |
| **Model 3 — Tooth type (intraoral)** | Detects each tooth one-by-one and labels it incisor / canine / premolar / molar. |
| **Unified runners** | Every model runs the same way: one folder-batch script, one drag-a-photo script, one local API. |
| **Training notebooks** | Kaggle-ready notebooks reproduce each model on a free T4 GPU. |
| **Reports** | Per-model metrics, confusion matrices, and training curves for the write-up / paper. |

---

## Repository structure

| Path | Purpose |
|------|---------|
| `model1_opg/` | Panoramic X-ray model (OPG). |
| `model2_occlusal/` | Smartphone occlusal-caries model + ICDAS severity. |
| `model3_tooth_type/` | Intraoral tooth-type model (incisor/canine/premolar/molar). |
| `<model>/code/` | Source: `scripts/` (analysis), `api/` (FastAPI), `windows/` (helper .bat), package modules. |
| `<model>/weights/` | Trained model weights (`.pt` / `.onnx`). |
| `<model>/notebooks/` | Kaggle training notebook(s). |
| `<model>/results/` | Metrics, confusion matrices, training curves. |
| `<model>/test_images/` | Drop your own photos here → results appear in `test_images/results/`. |
| `<model>/ANALYZE_TEST_IMAGES.bat` | One-click: analyse every photo in `test_images/`. |
| `<model>/START_HERE.txt` | What the model is + how to run it. |

Every model follows the **same layout and the same commands** — learn one, use all three.

---

## Models & results

All figures are on each model's held-out **test** split.

### Model 1 — Panoramic X-ray (OPG)
YOLO11 detection + FDI numbering, trained on DENTEX.

| Task | mAP@50 | Precision | Recall |
|------|--------|-----------|--------|
| Teeth + FDI numbering | 0.96 | 0.92 | 0.93 |
| Findings (overall) | 0.58 | — | — |
| &nbsp;&nbsp;• impacted | 0.92 | — | — |
| &nbsp;&nbsp;• caries | 0.53 | — | — |
| &nbsp;&nbsp;• deep caries | 0.47 | — | — |
| &nbsp;&nbsp;• periapical lesion | 0.38 | — | — |

### Model 2 — Occlusal caries (smartphone)
YOLO caries detector + EfficientNet severity grader, trained on Zenodo smartphone-caries data.

| Metric | Value |
|--------|-------|
| Caries detector mAP@50 | 0.82 |
| External-validation lesion sensitivity | 1.00 (125 lesions, 35 unseen photos) |
| Screening sensitivity / specificity | 0.80 / 0.80 |
| Severity classes | No / Mild / Moderate / Advanced (ICDAS-derived) |

### Model 3 — Tooth type (intraoral)
YOLO11s, trained on Dental_Dataset_Level2 (2,368 images).

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| incisor | 0.995 | 0.953 |
| canine | 0.995 | 0.884 |
| premolar | 0.995 | 0.856 |
| molar | 0.981 | 0.689 |
| **overall** | **0.991** | **0.845** |

---

## Datasets

| Dataset | Used by | Licence | Notes |
|---------|---------|---------|-------|
| DENTEX (panoramic X-rays) | Model 1 | CC BY-NC-SA 4.0 | FDI numbering + findings. |
| Zenodo smartphone caries (14769743) | Model 2 | CC BY | Caries boxes + ICDAS severity. |
| Dental_Dataset_Level2 | Model 3 | CC BY 4.0 | 2,149 / 112 / 107 train-val-test; 4 tooth classes. |

---

## Model experiments

| Model | Architecture | Training | Where |
|-------|--------------|----------|-------|
| Model 1 | YOLO11 detect + FDI post-processing | DENTEX, T4×2 | `model1_opg/notebooks/` |
| Model 2 | YOLO caries detector + EfficientNet-B0 severity | Zenodo, T4×2 | `model2_occlusal/notebooks/` |
| Model 3 | YOLO11s detection (4 classes) | Kaggle T4×2 | `model3_tooth_type/notebooks/` |

Each notebook is self-contained: pull the dataset → train → evaluate → export `best.pt` + `best.onnx`.

---

## Setup (Windows)

All three models share **one Python environment** (installed once).

```bat
:: 1) install Python 3.10-3.12 from python.org (tick "Add to PATH")
:: 2) create the shared environment (once):
model1_opg\code\windows\setup.bat
```

That's it — every model uses this environment.

---

## Usage

The same three commands work for **every** model.

```bat
:: A) Analyse a whole folder — put photos in <model>\test_images\, then:
<model>\ANALYZE_TEST_IMAGES.bat
::    -> results in <model>\test_images\results\

:: B) Analyse one photo — drag it onto:
<model>\code\windows\analyze_image.bat

:: C) Run the local API (for a web frontend):
<model>\code\windows\run_api.bat
::    Model 1 -> http://localhost:8000   Model 2 -> :8002   Model 3 -> :8003
```

Command line (identical shape across models):

```bash
python code/scripts/analyze.py IMAGE [IMAGE ...] --out OUTPUT_DIR
```

---

## Retraining

Open the model's notebook on Kaggle → **GPU T4 ×2, Internet On → Save & Run All** → download the
output zip → drop `best.pt` (+ `best.onnx`) into the model's `weights/`.

| Model | Notebook |
|-------|----------|
| Model 1 | `model1_opg/notebooks/train_model1_v2_kaggle.ipynb` |
| Model 2 | `model2_occlusal/notebooks/train_model2_v2b_kaggle.ipynb` |
| Model 3 | `model3_tooth_type/notebooks/train_model3_kaggle.ipynb` |

---

## Roadmap

- [x] Model 1 — panoramic X-ray (FDI + findings)
- [x] Model 2 — occlusal caries + ICDAS severity
- [x] Model 3 — tooth-type detection
- [ ] Explainability (Grad-CAM / attention overlays) across all models
- [ ] Unified web frontend
- [ ] Crack segmentation (Mask R-CNN) — future work
- [ ] CBCT structures (alveolar bone / mandibular canal) — future work

---

## Quick navigation

- [Model 1 — Panoramic X-ray](model1_opg/START_HERE.txt)
- [Model 2 — Occlusal caries](model2_occlusal/START_HERE.txt)
- [Model 3 — Tooth type](model3_tooth_type/START_HERE.txt)

---

## License & disclaimer

Code released under the MIT License. Datasets retain their own licences (see the Datasets table).
DentalX is a research prototype and AI screening aid; it does not provide a medical diagnosis and
must not be used as a substitute for a qualified dental professional.
