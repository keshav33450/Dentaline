"""Model 2 - occlusal caries API (matches Model 1/3 shape).

  pip install fastapi uvicorn python-multipart
  uvicorn api.main:app --host 127.0.0.1 --port 8001
  POST /analyze  (multipart: file=<occlusal photo>)  -> caries JSON
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
from fastapi import FastAPI, File, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402

# reuse occ_detect's analyze() + model loading
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "occ_detect", str(Path(__file__).resolve().parents[1] / "scripts" / "occ_detect.py"))
_occ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_occ)

from ultralytics import YOLO  # noqa: E402

_det_path = P.WEIGHTS / "occlusal_caries_det.pt"
if not _det_path.exists():
    _det_path = P.WEIGHTS / "occlusal_det.pt"
_MODEL = YOLO(str(_det_path))
_SEV = None
for _c in [P.WEIGHTS / "occlusal_severity.pt", P.WEIGHTS / "occlusal_severity_efficientnet_b0.pt"]:
    if _c.exists():
        _SEV = YOLO(str(_c)); break

app = FastAPI(title="DentalX Model 2 - Occlusal Caries")


@app.get("/health")
def health():
    return {"status": "ok", "model": _det_path.name}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), floor: float = 0.30, likely: float = 0.50):
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "x.jpg").suffix, delete=False) as tf:
        tf.write(await file.read()); tmp = tf.name
    raw = cv2.imread(tmp)
    if raw is None:
        return {"error": "cannot read image"}
    _, res = _occ.analyze(_MODEL, raw, floor, likely, 1024, sev_model=_SEV)
    return res
