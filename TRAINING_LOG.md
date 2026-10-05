# DentalX — Training Logs & Dataset Splits

Consolidated record of train/validation/test splits and the final-epoch training log for every model.
All numbers are read from each model's own `results.csv`, `metrics.json`, and dataset config — nothing
is estimated. Per-model `results.csv` (full epoch-by-epoch log), `args.yaml` (hyperparameters), and
confusion matrices live under each model's `results/` folder.

---

## Model 1 — Panoramic X-ray (OPG)

Two YOLO11s detectors: **teeth + FDI numbering** and **findings** (caries / deep caries / periapical /
impacted). Dataset: DENTEX.

**Config:** `yolo11s.pt`, imgsz 1024, batch 8, epochs 150 (early-stopped).

| Sub-model | Epochs run | Final P | Final R | mAP@50 | mAP@50-95 | Log |
|-----------|-----------|---------|---------|--------|-----------|-----|
| Teeth + FDI | 38 (early stop) | 0.918 | 0.915 | 0.930 (val) / **0.964 (test)** | 0.549 | `results/training_curves_teeth/results.csv` |
| Findings | 82 (early stop) | 0.539 | 0.615 | 0.541 | 0.361 | `results/training_curves_findings/results.csv` |

Test (held-out) overall: **teeth mAP@50 0.964**, findings mAP@50 ≈ 0.54. Per-class P/R/mAP in
`results/metrics.json`; confusion matrices in `results/teeth_test/` and `results/findings_test/`.

---

## Model 2 — Occlusal caries (smartphone)

YOLO caries **detector** + EfficientNet-B0 / MobileNet-V3 **severity** grader (ICDAS-derived).
Dataset: Zenodo 14769743 smartphone caries.

**Dataset:** 1,882 photos, 22,824 annotated regions. Severity distribution: no-caries 14,562 · mild
3,211 · moderate 2,103 · advanced 2,948. Severity grader trained with **5-fold CV** (folds ≈
4,461–4,654 regions each).

**Detector config:** imgsz 640, batch 16, epochs 100.

| Component | Epochs | P | R | mAP@50 | mAP@50-95 | Log |
|-----------|--------|---|---|--------|-----------|-----|
| Caries detector | 100 | 0.576 | 0.609 | 0.597 (val) / **0.824 (internal test)** | 0.378 | `results/detector_training/results.csv` |
| Severity grader | 5-fold CV | — | — | — | — | `results/severity/cv_summary.json` |

**External validation** (Zenodo test photos, 35 images, 125 lesions):
lesion sensitivity **1.00**, photo sensitivity **1.00**, 0 FN / 0 FP at photo level
(`results_v2/results/external_validation.json`). Internal caries-detector test: mAP@50 0.824, P 0.821,
R 0.736.

---

## Model 3 — Tooth type (intraoral, view-robust)

YOLO11s, 4 tooth-type classes (incisor/canine/premolar/molar), trained on merged multi-view data:
DentalMate6v **Front + Upper + Lower** Intraoral Tooth Numbering (CC BY 4.0), FDI remapped to 4 types.

**Config:** `yolo11s.pt`, imgsz 640, batch 16, epochs 100, single GPU T4, seed 42.

**Splits:** merged train / valid / **test 134 images (2,243 teeth)** across all three views
(frontal 42 · upper 45 · lower 47 in test).

**Overall test: mAP@50 0.946, mAP@50-95 0.666** (P 0.953, R 0.913).

| Class | mAP@50 | mAP@50-95 | P | R |
|-------|--------|-----------|---|---|
| incisor | 0.973 | 0.683 | 0.986 | 0.940 |
| canine | 0.951 | 0.673 | 0.950 | 0.923 |
| premolar | 0.944 | 0.666 | 0.947 | 0.910 |
| molar | 0.915 | 0.644 | 0.927 | 0.880 |

**Per-view test mAP@50 (view-robustness):**

| View | Images | mAP@50 | mAP@50-95 |
|------|--------|--------|-----------|
| Frontal | 42 | 0.949 | 0.650 |
| Upper occlusal | 45 | 0.947 | 0.685 |
| Lower occlusal | 47 | 0.945 | 0.674 |

Consistent ~0.95 across all three views — resolves the earlier frontal-only model's failure on
occlusal images. Full log: `results/results.csv`; curves + confusion matrix in `results/`.

> Supersedes the earlier frontal-only variants (4-class 0.991 on frontal-only Dental_Dataset_Level2;
> 7-class 0.653 on tooth-labelling). This multi-view model is the current Model 3.


## Reproducibility notes
- Fixed `seed=42` across runs.
- Full epoch-by-epoch logs: each model's `results/.../results.csv`.
- Hyperparameters: each model's `results/.../args.yaml`.
- Evaluation can be regenerated with `tools/evaluate.py` (per-class P/R/F1 + confusion matrix + leakage check).
