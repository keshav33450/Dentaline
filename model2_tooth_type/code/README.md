# Model 2 — tooth-type detector

Detects **each tooth, one by one**, on an intraoral photo and labels it
**incisor / canine / premolar / molar**.

```
intraoral photo -> RF-DETR detector -> per-tooth boxes + type + confidence -> overlay + JSON
```

Trained in **Roboflow** (workspace `keshav-raja`, project `dental_dataset_level2-fazpu`, v1) on
**2,368 labelled intraoral images** (2,149 train / 112 val / 107 test). The deployed model is the
RF-DETR NAS pick `da7025`, ~1.5 ms/image on a T4 GPU.

**Test-split accuracy** (held-out 107 images):

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| incisor  | 100.0% | 87.5% |
| canine   | 99.1%  | 77.4% |
| premolar | 99.3%  | 74.1% |
| molar    | 94.7%  | 53.6% |
| **overall** | **98.3%** | **73.1%** |

Class instances in the data: incisor 4,286 · canine 8,524 · premolar 8,321 · molar 6,879.
Full numbers in `reports/model4_eval.json`.

## Run it

```bash
pip install inference inference-sdk opencv-python
export ROBOFLOW_API_KEY=rf_iwIiyh43vKbF03xfqxNwN0x3CO42     # publishable key (browser-safe)

# local backend: model downloaded + cached once, then runs offline on your machine
python scripts/analyze.py --image YOUR_PHOTO.jpg --output outputs/result

# hosted backend: no model on disk, one network call per image
python scripts/analyze.py --image YOUR_PHOTO.jpg --backend hosted
```

Outputs `original.jpg`, `tooth_types.jpg` (annotated), and `result.json`:

```json
{ "backend": "local", "n_teeth": 3,
  "counts": {"incisor": 1, "canine": 1, "premolar": 0, "molar": 1},
  "teeth": [{"type": "molar", "confidence": 0.91, "bbox": [10.0, 60.0, 70.0, 130.0]}, ...] }
```

Or the API:

```python
from tooth_ai import ToothTyper
model = ToothTyper()                     # ROBOFLOW_API_KEY from env; backend="hosted" also works
res = model.predict("photo.jpg")
print(res["counts"])
```

## Two backends

| Backend | Package | Model on disk | Network per call | Use for |
|---------|---------|---------------|------------------|---------|
| `local` (default) | `inference` | cached after first run | first run only | clinic app / edge / offline |
| `hosted` | `inference-sdk` | none | every call | quick checks, no local GPU |

The **hosted** endpoint is the live Roboflow workflow `dental_dataset_level2-fazpu` (its default model is
set to `d991c1`). The **local** backend loads that same model by id via `inference.get_model`.

## Retrain / regenerate

All training was done in Roboflow via MCP — no local GPU, no dataset download. To retrain or pick a
different speed/accuracy point from the 72-model NAS frontier, use the project's Train page:
https://app.roboflow.com/keshav-raja/dental_dataset_level2-fazpu/train

`scripts/` and `notebooks/` retain the earlier YOLO-seg + FDI pipeline for reference; the shipped
Model 4 is the Roboflow detector above.

## Layout
```
scripts/analyze.py        the Model 4 inference pipeline (local + hosted)
tooth_ai.py                ToothTyper convenience API
tooth_ai_common.py         shared constants (tooth-type helpers)
docs/                      research + dataset + licence notes
scripts/ training/ models/ notebooks/   earlier training pipeline (reference)
```

Research prototype — not a diagnostic system.
