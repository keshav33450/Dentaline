"""Minimal FastAPI wrapper for Model 2 (tooth-type detection).

  pip install fastapi uvicorn python-multipart
  uvicorn api.main:app --host 127.0.0.1 --port 8002
  POST /analyze  (multipart: file=<intraoral photo>)  -> JSON {n_teeth, counts, teeth[]}
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

app = FastAPI(title="DentalX Model 2 - Tooth Type")


@app.get("/health")
def health():
    return {"status": "ok", "model": _an.LOCAL_MODEL_ID}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), backend: str = "local", conf: float = 0.4):
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "x.jpg").suffix, delete=False) as tf:
        tf.write(await file.read())
        tmp = tf.name
    _, res = _an.run(tmp, backend=backend, conf=conf)
    return res
