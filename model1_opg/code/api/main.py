"""
DentAssist OPG inference API.

  uvicorn api.main:app --host 0.0.0.0 --port 8000

  POST /predict/opg   multipart 'file' (jpg/png/dcm)  ?overlay=true&teeth=true
  GET  /health
  GET  /classes

Env: DENTASSIST_HOME (project folder; weights read from <home>/weights)
     TEETH_MODEL (default weights/teeth.onnx, falls back to teeth_best.pt)
     FINDINGS_MODEL (default weights/findings.onnx, falls back to findings_best.pt)
     FINDINGS_CONF (0.25) REVIEW_BELOW (0.5)  CORS_ORIGINS ("*")  MAX_UPLOAD_MB (40)
"""
from __future__ import annotations

import base64
import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import sys
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dentassist import paths as P  # noqa: E402
from dentassist.analyzer import DISCLAIMER, OPGAnalyzer, load_image  # noqa: E402
from dentassist.fdi import FDI_TEETH, FINDING_LABEL, FINDINGS  # noqa: E402
from dentassist.occlusal.pipeline import load_default as load_occlusal  # noqa: E402
from dentassist.occlusal.report import make_pdf  # noqa: E402


def _pick(env: str, *cands: str) -> str | None:
    if os.getenv(env):
        return os.environ[env]
    for c in cands:
        p = P.WEIGHTS / c
        if p.exists():
            return str(p)
    return None


TEETH = _pick("TEETH_MODEL", "teeth_seg_best.pt", "teeth_best.pt", "teeth.onnx")
FINDS = _pick("FINDINGS_MODEL", "findings_best.pt", "findings.onnx")
MAX_MB = float(os.getenv("MAX_UPLOAD_MB", "40"))

app = FastAPI(title="DentAssist OPG API", version="1.0.0",
              description="Panoramic X-ray: FDI tooth numbering + caries / deep caries / periapical lesion / impacted.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

analyzer: OPGAnalyzer | None = None
occlusal = None


@app.on_event("startup")
def _load():
    global analyzer, occlusal
    occlusal = load_occlusal(P.WEIGHTS)
    print("occlusal models:", "loaded" if occlusal else "not found (occlusal_seg/type/severity.pt)")
    if not TEETH:
        print("WARNING: no teeth model found - put weights in ./weights or set TEETH_MODEL")
        return
    analyzer = OPGAnalyzer(TEETH, FINDS,
                           findings_conf=float(os.getenv("FINDINGS_CONF", "0.25")),
                           review_below=float(os.getenv("REVIEW_BELOW", "0.5")))
    # warm-up so the first doctor request is fast
    import numpy as np
    analyzer.analyze(np.full((512, 1024, 3), 120, np.uint8))


@app.get("/health")
def health():
    return {"status": "ok" if analyzer else "no_model", "teeth_model": TEETH, "findings_model": FINDS,
            "occlusal_models": bool(occlusal)}


@app.get("/classes")
def classes():
    return {"teeth": FDI_TEETH, "findings": [{"key": k, "label": FINDING_LABEL[k]} for k in FINDINGS]}


@app.post("/predict/opg")
async def predict_opg(file: UploadFile = File(...),
                      overlay: bool = Query(True, description="return annotated PNG (base64)"),
                      teeth: bool = Query(True, description="include per-tooth boxes")):
    if analyzer is None:
        raise HTTPException(503, "model not loaded")
    data = await file.read()
    if len(data) > MAX_MB * 1e6:
        raise HTTPException(413, f"file larger than {MAX_MB} MB")
    try:
        img = load_image(data)
    except ValueError as e:
        raise HTTPException(415, str(e))
    result = analyzer.analyze(img, include_teeth=True)
    if overlay:
        vis = OPGAnalyzer.draw(img, result, show_teeth=teeth)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        result["overlay_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
    if not teeth:
        result["teeth"] = []
    result["filename"] = file.filename
    result["disclaimer"] = DISCLAIMER
    return result


@app.post("/predict/occlusal")
async def predict_occlusal(file: UploadFile = File(...),
                           pdf: bool = Query(False, description="return the PDF report instead of JSON"),
                           patient: str = Query("", description="patient code printed on the report"),
                           overlay: bool = Query(True)):
    """Smartphone occlusal photo -> premolar/molar + caries severity per posterior tooth."""
    from fastapi.responses import Response
    if occlusal is None:
        raise HTTPException(503, "occlusal models not loaded")
    data = await file.read()
    if len(data) > MAX_MB * 1e6:
        raise HTTPException(413, f"file larger than {MAX_MB} MB")
    try:
        img = load_image(data)
    except ValueError as e:
        raise HTTPException(415, str(e))
    res = occlusal.analyze(img)
    vis = occlusal.draw(res)
    if pdf:
        return Response(make_pdf(res, vis, patient_code=patient, photo_name=file.filename or ""),
                        media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{Path(file.filename or "report").stem}_report.pdf"'})
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    if overlay:
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out["overlay_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
    out["filename"] = file.filename
    return out
