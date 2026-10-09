# DentalX — Multi-Model Dental Image Analysis

An offline, multi-model deep-learning system for automated dental image analysis. DentalX bundles three
independent, production-ready models — smartphone occlusal-caries screening, view-robust intraoral
tooth-type detection, and gingival-inflammation (gingivitis) screening — each trained on real dental
datasets and runnable on a clinic PC.

> **Research prototype / AI screening aid — not a diagnostic device.**

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![Framework](https://img.shields.io/badge/framework-YOLO11-green)
![License](https://img.shields.io/badge/code-MIT-lightgrey)
![Status](https://img.shields.io/badge/models-3%2F3%20trained-success)

---

## Overview

Dental images come in very different forms — intraoral photos, occlusal (biting-surface) shots, mobile
snaps — and each task needs its own model. DentalX provides three:

1. **Occlusal caries (smartphone):** detects caries on intraoral photos and grades ICDAS-derived severity.
2. **Tooth type (intraoral):** labels each tooth incisor / canine / premolar / molar — **view-robust** across
   frontal, upper-occlusal and lower-occlusal photos.
3. **Gingival inflammation (intraoral / mobile):** detects inflamed-gum regions (erythema, edema, visible
   bleeding-on-probing) — a single `gingivitis` screening signal, view-robust across frontal / upper / lower.

---

## Pipeline Architecture

```
            ┌────────────────────────── DentalX ──────────────────────────┐
            │                                                              │
 photo  ───▶│  Model 1 · Occlusal   → caries boxes + ICDAS severity        │
 photo  ───▶│  Model 2 · Tooth type → incisor / canine / premolar / molar  │
            │                          (frontal + upper + lower views)     │
 photo  ───▶│  Model 3 · Gingivitis → inflamed-gum regions (screening)     │
            └──────────────────────────────────────────────────────────────┘
                    each model: detect → annotate → JSON + overlay
```

---

## Application — DentalX Camp

`camp_web/` is a full-stack (React + FastAPI + MongoDB) screening app that runs all three models on one
intraoral photo, returns independent model status and findings, and lets a doctor review and document each
patient case. It can generate a PDF and optionally e-mail it through configured SMTP. Set up with
`camp_web/README.md`, then run `run_app.bat` → opens http://localhost:8080.

---

## Project Structure

```
DentalX/
├── README.md
├── requirements.txt
├── scripts/                     # shared: config.py (paths), metrics.py (P/R/F1/mAP)
├── testing/
│   └── evaluate.ipynb           # unified evaluation across all 3 models
├── tools/
│   └── evaluate.py              # per-model eval (per-class P/R/F1 + confusion + leakage)
├── model1_occlusal/
│   ├── code/                    # occ_*.py pipeline, api/, dentassist/occlusal package
│   ├── models/ · outputs/ · notebooks/ · test_images/
│   └── START_HERE.txt           # (holds the shared .venv under code/.venv)
├── model2_tooth_type/           # same layout
│   ├── train.ipynb · evaluate.ipynb
│   └── models/ · outputs/ · code/ · test_images/
├── model3_gingival/             # same layout (gingivitis)
│   ├── train.ipynb · evaluate.ipynb
│   └── models/ · outputs/ · code/ · test_images/
├── camp_web/                    # DentalX Camp — React + FastAPI app (all 3 models)
├── README.md · TRAINING_LOG.md · RESEARCH_EVALUATION.md · datasets.md
```

Every model follows the **same layout** — `models/`, `outputs/`, `code/`, `test_images/` (Models 2 & 3 also
ship cell-by-cell `train.ipynb` / `evaluate.ipynb`). Learn one, use all three.

---

## Dataset

| Model | Dataset | Licence | Notes |
|-------|---------|---------|-------|
| Model 1 | Zenodo 14769743 (smartphone caries) + Roboflow ICDAS II | CC BY 4.0 | caries boxes + ICDAS severity; 1,882 photos / 22,824 regions |
| Model 2 | DentalMate6v Front/Upper/Lower Intraoral | CC BY 4.0 | multi-view; FDI remapped to 4 tooth types |
| Model 3 | sampling/gingivitis + Pranta/gum-disease (Roboflow Universe) | CC BY 4.0 | merged to one `gingivitis` class; 571 imgs / 1,508 boxes |

---

## Models & Training Configuration

| Model | Architecture | imgsz | Epochs | Test mAP@50 |
|-------|--------------|-------|--------|-------------|
| Model 1 — caries detector | YOLO detect | 640 | 100 | **0.824** (internal); ext. val sens **1.00** |
| Model 2 — tooth type | YOLO11s (4 cls) | 640 | 100 (early 86) | **0.946** |
| Model 3 — gingivitis | YOLO11s (1 cls) | 640 | 120 | **0.804** (validation) |

**Model 2 per-view (view-robust):** frontal 0.949 · upper 0.947 · lower 0.945.
**Model 3** figure is the held-out **validation** mAP@50 (best epoch 99/120); loose region-level boxes in the
source data cap mAP@50-95 (0.369). Full numbers and epoch-by-epoch logs in `TRAINING_LOG.md` and each model's
`outputs/`.

---

## Evaluation Metrics

- **mAP@50 / mAP@50-95** — detection accuracy at IoU 0.50 and averaged 0.50–0.95.
- **Precision / Recall / F1** — per class, in each model's `outputs/metrics.json`.
- **Confusion matrix** — per model, in `outputs/`.
- Regenerate everything reproducibly with `tools/evaluate.py` or `testing/evaluate.ipynb`.

---

## Setup & Installation

All three models share **one Python environment** (lives under `model1_occlusal/code/.venv`).

```bat
:: 1) install Python 3.10–3.12 (tick "Add to PATH")
:: 2) create the shared environment (once):
cd D:\Works\DentalX\model1_occlusal\code
python -m venv .venv
.venv\Scripts\activate
pip install -r ..\..\requirements.txt
```

---

## Running the Pipeline

Per model, three ways to run (identical shape across all three):

```bat
:: A) analyse a whole folder — put photos in <model>\test_images\, then:
<model>\ANALYZE_TEST_IMAGES.bat        ::  → results in test_images\results\

:: B) analyse one photo — drag it onto:
<model>\code\windows\analyze_image.bat

:: C) local API:
<model>\code\windows\run_api.bat       ::  M1 :8001  M2 :8002  M3 :8003
```

Retrain Models 2 & 3 from their notebooks:
`<model>/train.ipynb` → `<model>/evaluate.ipynb` → unified `testing/evaluate.ipynb`.
Retrain Model 1 from `model1_occlusal/notebooks/`.

---

## Progress

- [x] Model 1 — occlusal caries + ICDAS severity + external validation
- [x] Model 2 — view-robust tooth-type detection (frontal + upper + lower)
- [x] Model 3 — gingival-inflammation (gingivitis) screening
- [x] Per-class P/R/F1, confusion matrices, training logs (`TRAINING_LOG.md`)
- [ ] Data-leakage check across all datasets (`tools/evaluate.py`)
- [ ] Result-stability (multi-seed / bootstrap CI)
- [ ] Ablation studies per model
- [x] DentalX Camp web app (React + FastAPI orchestration + MongoDB)

---

## Environment

Python 3.10–3.12 · Ultralytics ≥ 8.3.0 · PyTorch (CUDA for training; CPU fine for inference). NumPy < 2.0 for
Models 1–2; Model 3 was trained on Kaggle's NumPy 2.x + current Ultralytics (inference works on either).

---

## References

- Zenodo smartphone caries dataset (record 14769743) + Roboflow "Caries Classification ICDAS II" — Model 1.
- DentalMate6v Front / Upper / Lower Intraoral Tooth Numbering datasets (Roboflow Universe, CC BY 4.0) — credited for Model 2's multi-view training data.
- sampling/gingivitis-6uyts and Pranta/gum-disease-mmzcr (Roboflow Universe, CC BY 4.0) — credited for Model 3's gingivitis training data.
- Ultralytics YOLO11.

---

## License & disclaimer

Code under the MIT License. Datasets retain their own licences (see Dataset table). DentalX is a research
prototype and AI screening aid; it does not provide a medical diagnosis and must not substitute for a
qualified dental professional.
