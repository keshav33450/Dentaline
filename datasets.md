# DentalX — Datasets & Attribution

Every model is trained only on real, publicly licensed dental data. Each dataset is used under its own
licence; credits below. DentalX ships trained weights, not the raw datasets — download links are the
original sources.

| Model | Dataset | Source | Licence | Size | Used for |
|-------|---------|--------|---------|------|----------|
| 1 — OPG | DENTEX (panoramic X-rays) | DENTEX challenge / Hugging Face | CC BY-NC-SA 4.0 | — | FDI tooth numbering + findings (caries, deep caries, periapical, impacted) |
| 2 — Occlusal | Smartphone caries | Zenodo record 14769743 | CC BY 4.0 | 1,882 photos / 22,824 regions | caries boxes + ICDAS-derived severity |
| 3 — Tooth type | DentalMate6v Front / Upper / Lower Intraoral Tooth Numbering | Roboflow Universe (DentalMate6v) | CC BY 4.0 | multi-view intraoral | FDI numbers remapped to 4 tooth types (incisor/canine/premolar/molar) |
| 4 — Gingivitis | gingivitis-6uyts | Roboflow Universe — **sampling** | CC BY 4.0 | 468 imgs | gingivitis detection boxes |
| 4 — Gingivitis | gum-disease-mmzcr | Roboflow Universe — **Pranta** | CC BY 4.0 | segmentation | gingivitis boxes (Periodontitis class dropped; polygons → boxes) |

## Model 4 merge notes

The two Model 4 sources were merged into a **single `gingivitis` class**, all views (frontal / upper /
lower) combined:

- Only boxes whose class name contains "ging" were kept; every other class (e.g. Pranta's Periodontitis)
  was dropped.
- Segmentation polygons were reduced to their enclosing bounding box.
- Final merged set: **571 images / 1,508 boxes** — train 480 / valid 46 / test 45.
- The source annotations are region-level (inflamed gum + adjacent tooth), suitable for a screening
  signal rather than pixel-tight localisation.

## Attribution

- **DENTEX** — DENTEX dental panoramic X-ray dataset.
- **Zenodo 14769743** — smartphone dental caries dataset.
- **DentalMate6v** — Front / Upper / Lower Intraoral Tooth Numbering (Roboflow Universe, CC BY 4.0).
- **sampling** — "gingivitis" dataset (Roboflow Universe, CC BY 4.0).
- **Pranta** — "gum-disease" dataset (Roboflow Universe, CC BY 4.0).

All CC BY / CC BY-SA datasets are credited as required by their licence. The DENTEX dataset is
**non-commercial** (CC BY-NC-SA 4.0): DentalX is a research prototype and is not distributed commercially.
