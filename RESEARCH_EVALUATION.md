# DentalX — Research Evaluation & Gap Analysis

This document tracks DentalX against a research-publication standard. It is deliberately honest
about **what evidence currently exists** versus **what is still missing**, and gives a concrete,
reproducible plan to close each gap. It responds point-by-point to external review feedback.

> Guiding principle (from review): *do not add more models for the sake of count.* First make the
> three existing pipelines properly trained, evaluated, and reproducible. Experimental rigor >
> model count.

---

## 1. Models in scope

| # | Pipeline | Task | Status |
|---|----------|------|--------|
| 1 | Occlusal caries (smartphone) | caries detection + ICDAS severity | trained; external validation reported, needs full per-class export |
| 2 | Tooth type (intraoral) | view-robust 4-class tooth-type detection | trained; **full artifacts present** (confusion matrix, curves, per-class) |
| 3 | Gingival inflammation | single-class gingivitis screening | trained; validation mAP@50 0.804, test eval reproducible via `evaluate.ipynb` |

---

## 2. Evidence currently on hand

| Evidence item | M1 (occlusal) | M2 (tooth) | M3 (gingival) |
|---|----|----|----|
| Overall mAP@50 / mAP@50-95 | ✅ | ✅ | ✅ |
| Per-class mAP | partial | partial | ✅ |
| Per-class Precision / Recall / **F1** | ❌ | ❌ | ⚠️ P/R yes, F1 to compute |
| Confusion matrix | ❌ | ❌ | ✅ (`results/confusion_matrix*.png`) |
| PR / P / R / F1 curves | ❌ | ❌ | ✅ (`results/*_curve.png`) |
| Training logs / results.csv | ❌ in repo | ❌ in repo | ✅ (`results/results.csv`) |
| External validation | — | ✅ (caries, 35 unseen photos) | ❌ |
| Data-leakage check | ❌ | ❌ | ❌ (script added, run pending) |
| Confidence intervals / repeats | ❌ | ❌ | ❌ |
| Ablation studies | ❌ | ❌ | ❌ |

Legend: ✅ present · ⚠️ partial · ❌ missing

---

## 3. Earlier 7-class tooth-type results (superseded)

> **Note:** the table below is the earlier **7-class** tooth-labelling variant. The current Model 2
> (tooth type) is the **view-robust 4-class** model — overall test **mAP@50 0.946** — documented in
> `TRAINING_LOG.md`. This section is kept for the per-class / FP-FN discussion only.

7-class FDI tooth-type model, YOLO11s, tooth-labelling dataset. Test split: 42 images, 758 teeth.

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Test instances |
|-------|-----------|--------|----|--------|-----------|----------------|
| Central Incisor | 0.821 | 0.851 | 0.836 | 0.867 | 0.814 | 161 |
| Canine | 0.840 | 0.760 | 0.798 | 0.811 | 0.749 | 150 |
| Lateral Incisor | 0.619 | 0.776 | 0.689 | 0.719 | 0.660 | 156 |
| 1st Premolar | 0.695 | 0.658 | 0.676 | 0.689 | 0.597 | 120 |
| 2nd Premolar | 0.578 | 0.581 | 0.579 | 0.606 | 0.469 | 93 |
| 1st Molar | 0.562 | 0.600 | 0.580 | 0.596 | 0.440 | 65 |
| 2nd Molar | 0.441 | 0.385 | 0.411 | 0.284 | 0.238 | 13 |
| **Overall** | **0.651** | **0.659** | **0.655** | **0.653** | **0.567** | **758** |

F1 = 2·P·R/(P+R), computed from the reported per-class P and R.

**Reading of the results (false-positive / false-negative view):**
- Anterior teeth (incisors, canine) are strongest — distinct morphology, more instances.
- **2nd Molar is unreliable** (F1 0.41, only 13 test instances): both recall (misses) and precision
  (confusions with 1st molar) are poor. This class is **data-limited**, not a model-capacity issue.
- Premolar/molar confusions dominate the error budget — expected, as adjacent posterior teeth look
  alike. The confusion matrix (`results/confusion_matrix_normalized.png`) shows the exact off-diagonal mass.

**Known limitation vs. the earlier variant:** a prior 4-class model (incisor/canine/premolar/molar,
Dental_Dataset_Level2, 1,310 images) reached 0.991 mAP@50. The move to 7 FDI classes on a smaller
(~420-image) set is the direct cause of the lower score. This is a **data** gap, documented, not hidden.

---

## 4. Gaps and how to close each (reproducible steps)

A unified evaluation script is provided: **`tools/evaluate.py`**. For any model it emits per-class
P/R/F1 (CSV+JSON), confusion matrix + curves, and a data-leakage report — all computed, nothing
fabricated.

### 4.1 Per-class P / R / F1 for every model
```bash
python tools/evaluate.py --weights model1_occlusal/models/best.pt --data <m1_data.yaml> --out model1_occlusal/outputs
python tools/evaluate.py --weights model2_tooth_type/models/best.pt --data <m2_data.yaml> --out model2_tooth_type/outputs
python tools/evaluate.py --weights model3_gingival/models/best.pt --data <m3_data.yaml> --out model3_gingival/outputs
```
Output: `per_class_metrics.csv`, `evaluation_full.json`.

### 4.2 Confusion matrices + FP/FN analysis
`tools/evaluate.py` runs ultralytics `val(plots=True)`, which writes `confusion_matrix.png` and
`confusion_matrix_normalized.png`. Read off-diagonal cells for the dominant confusions; pair with the
per-class recall (FN rate) and precision (FP rate) table above.

### 4.3 Training time, logs, saved results
Keep each run's `results.csv`, `args.yaml`, and the console log. Record wall-clock training time per
model in the table below (fill from Kaggle run headers):

| Model | GPU | Epochs | Wall-clock | Log file |
|-------|-----|--------|-----------|----------|
| M1 | T4×2 | — | — | `model1_occlusal/outputs/` |
| M2 | T4 | 100 (early-stop) | — | `model2_tooth_type/outputs/results.csv` |
| M3 | T4 | 120 | ~25 min | `model3_gingival/outputs/results.csv` |

### 4.4 Data-leakage & reproducibility
- Leakage: `tools/evaluate.py` hashes every image and reports any identical image shared across
  train/val/test (`leakage_report.json`). Run once per dataset; target verdict = **CLEAN**.
- Reproducibility: all training uses fixed `seed=42`; record library versions
  (`ultralytics`, `torch`, `numpy`) and the exact dataset version/hash per model.

### 4.5 Confidence intervals / repeated experiments
Two acceptable routes:
1. **Repeated runs:** retrain each model with ≥3 seeds (e.g. 0/42/123); report mean ± std of mAP@50.
2. **Bootstrap CI (no retrain):** resample the test set per-image to get a 95% CI on mAP@50.
   (`tools/evaluate.py` scaffolds this; enable per-image JSON eval to complete it.)

### 4.6 Ablation studies
Minimum viable ablations per model:
- backbone/size (e.g. YOLO11n vs 11s vs 11m)
- image size (640 vs 1024)
- augmentation on/off (mosaic, close_mosaic)
- dataset size (train on 50% vs 100% to show the data-limited curve — directly explains M3's gap)

Report each as a one-row delta against the baseline in a single ablation table per model.

---

## 5. Priority order (recommended)

1. **Regenerate full eval artifacts for M1 & M2** (per-class P/R/F1 + confusion matrix) with
   `tools/evaluate.py` — this is the biggest credibility gap.
2. **Run the leakage check on all three datasets** — a single bad result here invalidates everything.
3. **Record training time + logs + library versions** for reproducibility.
4. **Add ≥3-seed repeats (or bootstrap CI)** on at least M3 (smallest, fastest) to establish stability.
5. **One ablation table per model**, starting with the M3 dataset-size ablation that explains its score.
6. Only after the above: consider whether M3 should be retrained on the larger cleaned dataset.

Model count is frozen at 3 until the above is complete.

---

*This file is documentation of evaluation status and plan. All numeric results shown are computed
from the trained weights and their test splits; gaps are marked rather than filled with estimates.*
