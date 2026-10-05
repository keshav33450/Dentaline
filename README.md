# DentalX — Multi-Model Dental Image Analysis

An offline, multi-model deep-learning system for automated dental image analysis. DentalX bundles three
independent, production-ready models — panoramic X-ray analysis, smartphone occlusal-caries screening, and
view-robust intraoral tooth-type detection — each trained on real dental datasets and runnable on a clinic PC.

> **Research prototype / AI screening aid — not a diagnostic device.**

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![Framework](https://img.shields.io/badge/framework-YOLO11-green)
![License](https://img.shields.io/badge/code-MIT-lightgrey)
![Status](https://img.shields.io/badge/models-3%2F3%20trained-success)

---

## Overview

Dental images come in very different forms — panoramic X-rays, intraoral photos, occlusal (biting-surface)
shots — and each needs its own model. DentalX provides three:

1. **Panoramic X-ray (OPG):** numbers every tooth (FDI) and flags findings (caries, deep caries, periapical
   lesions, impacted teeth).
2. **Occlusal caries (smartphone):** detects caries on intraoral photos and grades ICDAS-derived severity.
3. **Tooth type (intraoral):** labels each tooth incisor / canine / premolar / molar — **view-robust** across
   frontal, upper-occlusal and lower-occlusal photos.

---

## Pipeline Architecture

```
            ┌────────────────────────── DentalX ──────────────────────────┐
            │                                                              │
 X-ray  ───▶│  Model 1 · OPG        → teeth + FDI numbering + findings     │
 photo  ───▶│  Model 2 · Occlusal   → caries boxes + ICDAS severity        │
 photo  ───▶│  Model 3 · Tooth type → incisor / canine / premolar / molar  │
            │                          (frontal + upper + lower views)     │
            └──────────────────────────────────────────────────────────────┘
                    each model: detect → annotate → JSON + overlay
```

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
├── model1_opg/
│   ├── train.ipynb              # cell-by-cell training
│   ├── evaluate.ipynb           # cell-by-cell evaluation
│   ├── models/                  # trained weights (.pt / .onnx)
│   ├── outputs/                 # metrics, curves, confusion matrices
│   ├── code/                    # analyze.py, api/, dentassist package
│   └── test_images/             # drop photos here → test_images/results/
├── model2_occlusal/             # same layout
├── model3_tooth_type/           # same layout
├── README.md · TRAINING_LOG.md · RESEARCH_EVALUATION.md
```

Every model follows the **same layout** — `train.ipynb`, `evaluate.ipynb`, `models/`, `outputs/`, `code/`.
Learn one, use all three.

---

## Dataset

| Model | Dataset | Licence | Notes |
|-------|---------|---------|-------|
| Model 1 | DENTEX (panoramic) | CC BY-NC-SA 4.0 | FDI numbering + findings |
| Model 2 | Zenodo 14769743 (smartphone caries) | CC BY | caries boxes + ICDAS severity; 1,882 photos / 22,824 regions |
| Model 3 | DentalMate6v Front/Upper/Lower Intraoral (CC BY 4.0) | CC BY 4.0 | multi-view; FDI remapped to 4 tooth types |

---

## Models & Training Configuration

| Model | Architecture | imgsz | Epochs | Test mAP@50 |
|-------|--------------|-------|--------|-------------|
| Model 1 — teeth + FDI | YOLO11s detect | 1024 | 150 (early 38) | **0.964** |
| Model 1 — findings | YOLO11s detect | 1024 | 150 (early 82) | ~0.54 |
| Model 2 — caries detector | YOLO detect | 640 | 100 | **0.824** (internal); ext. val sens **1.00** |
| Model 3 — tooth type | YOLO11s (4 cls) | 640 | 100 (early 86) | **0.946** |

**Model 3 per-view (view-robust):** frontal 0.949 · upper 0.947 · lower 0.945. Full numbers and
epoch-by-epoch logs in `TRAINING_LOG.md` and each model's `outputs/`.

---

## Evaluation Metrics

- **mAP@50 / mAP@50-95** — detection accuracy at IoU 0.50 and averaged 0.50–0.95.
- **Precision / Recall / F1** — per class, in each model's `outputs/metrics.json`.
- **Confusion matrix** — per model, in `outputs/`.
- Regenerate everything reproducibly with `tools/evaluate.py` or `testing/evaluate.ipynb`.

---

## Setup & Installation

All three models share **one Python environment**.

```bat
:: 1) install Python 3.10–3.12 (tick "Add to PATH")
:: 2) create the shared environment (once):
cd D:\Works\DentalX\model1_opg\code
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
<model>\code\windows\run_api.bat       ::  M1 :8000  M2 :8002  M3 :8003
```

Retrain any model from its notebook:
`<model>/train.ipynb` → `<model>/evaluate.ipynb` → unified `testing/evaluate.ipynb`.

---

## Progress

- [x] Model 1 — panoramic X-ray (FDI + findings)
- [x] Model 2 — occlusal caries + ICDAS severity + external validation
- [x] Model 3 — view-robust tooth-type detection (frontal + upper + lower)
- [x] Per-class P/R/F1, confusion matrices, training logs (`TRAINING_LOG.md`)
- [ ] Data-leakage check across all datasets (`tools/evaluate.py`)
- [ ] Result-stability (multi-seed / bootstrap CI)
- [ ] Ablation studies per model
- [ ] Unified web frontend

---

## Environment

Python 3.10–3.12 · Ultralytics ≥ 8.3.0 · NumPy < 2.0 · PyTorch (CUDA for training; CPU fine for inference).

---

## References

- DENTEX panoramic dental dataset.
- Zenodo smartphone caries dataset (record 14769743).
- DentalMate6v Front / Upper / Lower Intraoral Tooth Numbering datasets (Roboflow Universe, CC BY 4.0) — credited for Model 3's multi-view training data.
- Ultralytics YOLO11.

---

## License & disclaimer

Code under the MIT License. Datasets retain their own licences (see Dataset table). DentalX is a research
prototype and AI screening aid; it does not provide a medical diagnosis and must not substitute for a
qualified dental professional.
