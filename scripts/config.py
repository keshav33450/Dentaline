"""DentalX — centralized paths & shared configuration (mirrors the senior project's scripts/config.py).

Every model folder is self-contained; this module gives shared constants so notebooks and scripts
don't hardcode paths. Import with:  from scripts.config import ROOT, MODELS, CLASSES
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root (DentalX/)

MODELS = {
    "model1_opg":        ROOT / "model1_opg",        # panoramic X-ray (OPG)
    "model2_occlusal":   ROOT / "model2_occlusal",   # occlusal caries + severity
    "model3_tooth_type": ROOT / "model3_tooth_type", # intraoral tooth type (view-robust)
}

# per-model sub-folders follow one convention (mirrors senior repo: models/ + outputs/)
def model_paths(name: str) -> dict:
    base = MODELS[name]
    return {
        "root":    base,
        "models":  base / "models",       # trained weights (.pt / .onnx)
        "outputs": base / "outputs",      # metrics, curves, confusion matrices
        "code":    base / "code",
        "test_images": base / "test_images",
    }

# class definitions per model
CLASSES = {
    "model3_tooth_type": ["incisor", "canine", "premolar", "molar"],
}

# local API ports
PORTS = {"model1_opg": 8000, "model2_occlusal": 8002, "model3_tooth_type": 8003}
