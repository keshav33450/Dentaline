# DentalX Model 1 — Smartphone Occlusal Caries

AI screening for **occlusal (biting-surface) caries** on smartphone intraoral photos, plus ICDAS-derived
severity grading. Part of the DentalX pipeline.

| | Caries detector | Severity grader |
|---|---|---|
| Output | caries boxes on occlusal photos | No / Mild / Moderate / Advanced (ICDAS-derived) |
| Base / input | YOLO detector @ 640 | EfficientNet-B0 / MobileNet-V3 |
| Training data | Zenodo 14769743 (6,313 smartphone photos) | Roboflow "Caries Classification ICDAS II" (CC BY 4.0) |

The pipeline (`dentassist/occlusal/`) detects caries, grades severity, and returns an annotated overlay +
JSON. External validation on held-out Zenodo photos is included.

## Run locally
```bat
:: analyse a folder:
..\ANALYZE_TEST_IMAGES.bat
:: one photo: drag onto  windows\analyze_image.bat
:: local API (port 8001):  windows\run_api.bat
```

## Train on Kaggle
Upload `notebooks/train_model2_kaggle.ipynb` (GPU T4, Internet On). See `..\START_HERE.txt` for the full
4-part training flow (severity, phone-caries detector, tooth outline, external validation) and the
SegmentAnyTooth setup for the optional tooth-outline part.

Key scripts (all prefixed `occ_`): `occ_detect.py`, `occ_train.py`, `occ_evaluate.py`,
`occ_external_validation.py`, `occ_analyze.py`. Model export: `export.py`. Usability scoring: `sus_score.py`.

Licences: Roboflow ICDAS II CC BY 4.0 · Zenodo 14769743 CC BY 4.0 · SegmentAnyTooth code MIT / weights
non-commercial research only.

> Research prototype / AI screening aid — not a diagnostic device.
