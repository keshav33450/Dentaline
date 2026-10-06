"""Minimal FastAPI wrapper for Model 3 (gingival inflammation detection).

  pip install fastapi uvicorn python-multipart
  uvicorn api.main:app --host 127.0.0.1 --port 8003
  POST /analyze  (multipart: file=<intraoral photo>)  -> JSON {gingivitis_detected, regions[]}
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

_spec = importlib.util.spec_from_file_location(
    "analyze", str(Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"))
_an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_an)

app = FastAPI(title="DentalX Model 3 - Gingival Inflammation")


@app.get("/health")
def health():
    return {"status": "ok", "model": "yolo11s-gingivitis", "classes": ["gingivitis"]}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), conf: float = 0.4):
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "x.jpg").suffix, delete=False) as tf:
        tf.write(await file.read())
        tmp = tf.name
    _, res = _an.run(tmp, backend="weights", conf=conf)
    return res
