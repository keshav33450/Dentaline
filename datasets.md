# DentalX — Datasets & Attribution

Every model is trained only on real, publicly licensed dental data. Each dataset is used under its own
licence; credits and source links are below. DentalX ships trained weights, not the raw datasets — the
links point to the original public sources, and all data is **CC BY 4.0**.

---

## Overview

| Model | Dataset | Source | Licence | Size | Used for |
|-------|---------|--------|---------|------|----------|
| 1 — Caries (occlusal) | Smartphone caries | Zenodo record 14769743 | CC BY 4.0 | 1,882 photos / 22,824 regions | caries detection boxes + ICDAS-derived severity |
| 1 — Caries (severity labels) | Caries Classification ICDAS II | Roboflow Universe — **code-geass** | CC BY 4.0 | ICDAS 0-6 classes | ICDAS severity reference for the classifier |
| 2 — Tooth type | DentalMate6v Front / Upper / Lower Intraoral Tooth Numbering | Roboflow Universe (DentalMate6v) | CC BY 4.0 | multi-view intraoral - 134 test imgs / 2,243 teeth | FDI numbers remapped to 4 tooth types (incisor / canine / premolar / molar) |
| 3 — Gingivitis | gingivitis-6uyts | Roboflow Universe — **sampling** | CC BY 4.0 | 468 imgs (detection) | gingivitis detection boxes |
| 3 — Gingivitis | gum-disease-mmzcr | Roboflow Universe — **Pranta** | CC BY 4.0 | segmentation | gingivitis boxes (Periodontitis class dropped; polygons -> boxes) |

---

## Model 1 — Caries (two-stage: detect + grade)

Model 1 has two components trained on the Zenodo smartphone caries dataset (Ahmed et al., *Sci Data* 2025),
with ICDAS severity labels.

**Detector (YOLOv8, class `caries`)** — "where is the caries"
- Occlusal views only (upper + lower), permanent-tooth decay (`D`) kept, primary (`d`) excluded.
- **1,956 photos** kept - **5,923 permanent-caries boxes** (503 primary excluded).
- **Patient-grouped split:** train **1,885** / val **36** / test **35** - **274 patient groups**.
- Internal test: **mAP@50 0.824**, precision 0.821, recall 0.736.

**Severity grader (EfficientNet-B0 / MobileNet-V3)** — "how severe (ICDAS)"
- **1,882 photos / 22,824 annotated regions.**
- ICDAS 0-6 mapped to 4 severity classes:

  | ICDAS code | Class |
  |---|---|
  | 0 Sound tooth structure | no_caries |
  | 1 Faint visual change - 2 Distinct visual change | mild |
  | 3 Localized enamel breakdown - 4 Dark dentinal shadow | moderate |
  | 5 Distinct cavity - 6 Extensive cavity | advanced |

- Severity distribution: **no_caries 14,562 - mild 3,211 - moderate 2,103 - advanced 2,948**
  (class imbalance handled with class-weighted loss).
- Trained with **patient-grouped 5-fold cross-validation** (folds ~ 4,461-4,654 regions each), so every
  tooth is scored out-of-fold by a model that never saw that patient.
- EfficientNet-B0 vs MobileNet-V3 compared with a **McNemar test**; the best by macro-F1 is selected for the app.

**External validation** (Zenodo test photos never seen in training): **35 photos, 125 lesions ->
lesion sensitivity 1.00, photo sensitivity 1.00, 0 FN / 0 FP at photo level.** (Small set — a larger
field trial is still needed.)

---

## Model 2 — Tooth type (view-robust)

- 4 classes: incisor / canine / premolar / molar; FDI numbers remapped to type.
- Merged multi-view data (frontal + upper + lower), **test 134 images / 2,243 teeth**.
- Overall test **mAP@50 0.946**, mAP@50-95 0.666; per-view ~ 0.95 (frontal 0.949 - upper 0.947 - lower 0.945).

---

## Model 3 — Gingivitis (merge notes)

The two Model 3 sources were merged into a **single `gingivitis` class**, all views (frontal / upper /
lower) combined:

- Only boxes whose class name contains "ging" were kept; every other class (e.g. Pranta's Periodontitis)
  was dropped.
- Segmentation polygons were reduced to their enclosing bounding box.
- Final merged set: **571 images / 1,508 boxes** — train 480 / valid 46 / test 45.
- Best (validation, epoch 99/120): **mAP@50 0.804**, mAP@50-95 0.369.
- Source annotations are region-level (inflamed gum + adjacent tooth), suitable for a screening signal
  rather than pixel-tight localisation.

---

## Source links

| Dataset | Link |
|---|---|
| Zenodo smartphone caries (record 14769743) | https://zenodo.org/records/14769743 |
| Caries Classification ICDAS II (code-geass) | https://universe.roboflow.com/code-geass/caries-classification-icdas-ii |
| DentalMate6v Intraoral Tooth Numbering | Roboflow Universe (DentalMate6v — Front / Upper / Lower) |
| gingivitis-6uyts (sampling) | https://universe.roboflow.com/keshav-raja-s-workspace/gingivitis-6uyts-qbisk |
| gum-disease-mmzcr (Pranta) | https://universe.roboflow.com/keshav-raja-s-workspace/gum-disease-mmzcr |

---

## Attribution

- **Zenodo 14769743** — smartphone dental caries dataset (Ahmed et al., *Scientific Data*, 2025).
- **code-geass** — "Caries Classification ICDAS II" (Roboflow Universe, CC BY 4.0).
- **DentalMate6v** — Front / Upper / Lower Intraoral Tooth Numbering (Roboflow Universe, CC BY 4.0).
- **sampling** — "gingivitis" dataset (Roboflow Universe, CC BY 4.0).
- **Pranta** — "gum-disease" dataset (Roboflow Universe, CC BY 4.0).

All datasets are CC BY 4.0 and credited as required by their licence. DentalX is a research prototype and
AI screening aid, not a commercial product.
