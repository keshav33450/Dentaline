"""DentalX — centralized paths & shared configuration.

Every model folder is self-contained; this module gives shared constants so notebooks and scripts
don't hardcode paths. Import with:  from scripts.config import ROOT, MODELS, CLASSES
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root (DentalX/)

MODELS = {
    "model1_occlusal":   ROOT / "model1_occlusal",   # occlusal caries + severity
    "model2_tooth_type": ROOT / "model2_tooth_type", # intraoral tooth type (view-robust)
    "model3_gingival":   ROOT / "model3_gingival",   # gingival inflammation (gingivitis)
}

# per-model sub-folders follow one convention (models/ + outputs/)
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
    "model2_tooth_type": ["incisor", "canine", "premolar", "molar"],
    "model3_gingival":   ["gingivitis"],
}

# local API ports
PORTS = {"model1_occlusal": 8001, "model2_tooth_type": 8002, "model3_gingival": 8003}
