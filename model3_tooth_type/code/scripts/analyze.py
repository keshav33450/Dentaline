#!/usr/bin/env python3
"""
Model 4 - tooth-type detector.

Intraoral photo -> detect each tooth one by one -> label incisor / canine / premolar / molar
-> annotated overlay + JSON.

Backed by the RF-DETR model trained in Roboflow
(workspace keshav-raja, project dental_dataset_level2-fazpu, version 1;
NAS model da7025: test mAP@50 98.3%, per-class mAP@50 incisor 100 / canine 99.1 / premolar 99.3 /
molar 94.7). Two backends:

  weights (default) - local YOLO11 best.pt via ultralytics. Fully offline, no API key.
                      Get the .pt from notebooks/train_model3_kaggle.ipynb -> weights/best.pt.
  local             - cached Roboflow RF-DETR model via the `inference` package (downloads once).
  hosted            - Roboflow serverless workflow via `inference-sdk` (one network call per image).

  python scripts/analyze.py --image photo.jpg                  # weights\\best.pt, offline
  python scripts/analyze.py --image photo.jpg --backend hosted

API key: pass --api-key, or set ROBOFLOW_API_KEY. The publishable key (rf_...) works.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toothtype_common import TYPE_CLASSES  # noqa: E402

# Roboflow identifiers (confirmed live 2026-09-28).
WORKSPACE = "keshav-raja"
LOCAL_MODEL_ID = "keshav-raja/dental_dataset_level2-fazpu-1-rfdetr-nas-t1--da7025"  # NAS model, get_model()
HOSTED_WORKFLOW_ID = "dental_dataset_level2-fazpu"        # live workflow endpoint (default model = above)
HOSTED_URL = "https://serverless.roboflow.com"

# local YOLO11 .pt (produced by notebooks/train_model3_kaggle.ipynb) - offline, no API key.
DEFAULT_WEIGHTS = str(Path(__file__).resolve().parents[2] / "weights" / "best.pt")

# one stable colour per tooth type (BGR)
TYPE_COLORS = {"incisor": (66, 133, 244), "canine": (219, 68, 55),
               "premolar": (244, 180, 0), "molar": (15, 157, 88)}
DEFAULT_COLOR = (171, 71, 188)


def _norm_type(label: str) -> str:
    """Normalise a class label. The model's own class names are kept as-is (so FDI-type
    models like 'Central Incisor', '1st Molar' pass through unchanged); the legacy 4-type
    labels are lower-cased/de-pluralised to incisor/canine/premolar/molar."""
    raw = str(label).strip()
    s = raw.lower().rstrip("s")
    if s in TYPE_CLASSES:
        return s
    return raw  # keep the model's native class name (e.g. "Central Incisor")


def _color_for(label: str):
    """Stable BGR colour for any class: fixed palette for the 4 legacy types,
    deterministic hash-based colour for everything else."""
    if label in TYPE_COLORS:
        return TYPE_COLORS[label]
    h = hashlib.md5(label.encode()).digest()
    return (60 + h[0] % 196, 60 + h[1] % 196, 60 + h[2] % 196)


def _to_teeth(preds: list[dict]) -> list[dict]:
    """Roboflow prediction dicts -> our tooth records (center x/y/w/h -> xyxy bbox), left-to-right."""
    teeth = []
    for p in preds:
        x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
        teeth.append({
            "type": _norm_type(p.get("class", p.get("class_name", ""))),
            "confidence": round(float(p.get("confidence", 0.0)), 3),
            "bbox": [round(x - w / 2, 1), round(y - h / 2, 1), round(x + w / 2, 1), round(y + h / 2, 1)],
        })
    return sorted(teeth, key=lambda t: (t["bbox"][0], t["bbox"][1]))


def _infer_local(image_path, model_id, api_key, conf):
    from inference import get_model
    model = get_model(model_id=model_id, api_key=api_key)
    res = model.infer(str(image_path), confidence=conf)[0]
    return [p.dict() if hasattr(p, "dict") else dict(p) for p in res.predictions]


def _infer_hosted(image_path, workflow_id, api_key, conf):
    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=HOSTED_URL, api_key=api_key)
    out = client.run_workflow(workspace_name=WORKSPACE, workflow_id=workflow_id,
                              images={"image": str(image_path)}, use_cache=True)
    block = out[0] if isinstance(out, list) else out
    preds = block.get("predictions", {})
    preds = preds.get("predictions", preds) if isinstance(preds, dict) else preds
    return [p for p in preds if p.get("confidence", 0) >= conf]


def _infer_weights(image_path, weights_path, conf):
    """Local YOLO11 .pt via ultralytics - fully offline, no API key. Returns Roboflow-shaped dicts."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    r = model.predict(str(image_path), conf=conf, verbose=False)[0]
    out = []
    if r.boxes is not None:
        names = model.names
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        xywh = r.boxes.xywh.cpu().numpy()  # center x, center y, w, h
        for i in range(len(cls)):
            x, y, w, h = xywh[i]
            out.append({"x": float(x), "y": float(y), "width": float(w), "height": float(h),
                        "confidence": float(cf[i]), "class": names[int(cls[i])]})
    return out


def run(image_path, backend="weights", api_key=None, conf=0.4, model_id=None, weights=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"cannot read {image_path}")

    if backend == "weights":
        wp = weights or DEFAULT_WEIGHTS
        if not Path(wp).exists():
            raise SystemExit(f"no local weights at {wp} - train with notebooks/train_model3_kaggle.ipynb, "
                             f"or use --backend hosted / --backend local")
        preds = _infer_weights(image_path, wp, conf)
    else:
        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            raise SystemExit("no API key - pass --api-key or set ROBOFLOW_API_KEY")
        if backend == "local":
            preds = _infer_local(image_path, model_id or LOCAL_MODEL_ID, api_key, conf)
        else:
            preds = _infer_hosted(image_path, model_id or HOSTED_WORKFLOW_ID, api_key, conf)
    teeth = _to_teeth(preds)

    counts = {}
    for t in teeth:
        counts[t["type"]] = counts.get(t["type"], 0) + 1
    res = {"backend": backend, "n_teeth": len(teeth), "counts": counts, "teeth": teeth,
           "disclaimer": "Research prototype - not a diagnostic system."}
    return img, res


def draw(img, res):
    out = img.copy()
    for t in res["teeth"]:
        c = _color_for(t["type"])
        x0, y0, x1, y1 = map(int, t["bbox"])
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2)
        lbl = f"{t['type']} {t['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, max(y0 - th - 6, 0)), (x0 + tw + 4, y0), c, -1)
        cv2.putText(out, lbl, (x0 + 2, max(y0 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    # Same CLI shape as Models 1 & 2:  python scripts/analyze.py IMG [IMG ...] --out DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="output folder (default: next to each image)")
    ap.add_argument("--backend", choices=["weights", "local", "hosted"], default="weights",
                    help="weights = local best.pt (offline, default) | local = cached Roboflow model | hosted = API")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--model", default=None, help="override model id (local) or workflow id (hosted)")
    ap.add_argument("--weights", default=None, help="path to a YOLO11 .pt (default weights/best.pt)")
    a = ap.parse_args()

    for img_path in a.images:
        p = Path(img_path)
        img, res = run(str(p), a.backend, a.api_key, a.conf, a.model, a.weights)
        out = Path(a.out) if a.out else p.parent
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{p.stem}_result.jpg"), draw(img, res))
        (out / f"{p.stem}_result.json").write_text(json.dumps(res, indent=1))
        print(f"\n{p.name}: {res['n_teeth']} teeth  {res['counts']}")
        for t in res["teeth"]:
            print(f"  {t['type']:9s} {t['confidence']}")
        print(f"  saved -> {out / (p.stem + '_result.jpg')}")


if __name__ == "__main__":
    main()
