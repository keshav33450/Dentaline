#!/usr/bin/env python3
"""
Model 3 - gingival inflammation (gingivitis) detector.

Intraoral / mobile photo -> detect inflamed-gum regions (erythema, edema, visible
bleeding-on-probing) -> annotated overlay + JSON. Single class: 'gingivitis'.
View-robust: trained on frontal + upper + lower intraoral photos merged together.

Backend:
  weights (default) - local YOLO11s best.pt via ultralytics. Fully offline, no API key.
                      Get the .pt from notebooks/train_model4_kaggle.ipynb -> models/best.pt.

  python scripts/analyze.py --image photo.jpg              # models/best.pt, offline
  python scripts/analyze.py photo1.jpg photo2.jpg --out results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

# local YOLO11 .pt (produced by notebooks/train_model4_kaggle.ipynb) - offline, no API key.
DEFAULT_WEIGHTS = str(Path(__file__).resolve().parents[2] / "models" / "best.pt")

GINGIVITIS_COLOR = (0, 0, 255)   # BGR red - inflamed-gum box


def _infer_weights(image_path, weights_path, conf):
    """Local YOLO11 .pt via ultralytics - fully offline. Returns Roboflow-shaped dicts."""
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


def _to_regions(preds: list[dict]) -> list[dict]:
    """Prediction dicts -> gingivitis region records (center x/y/w/h -> xyxy bbox)."""
    regions = []
    for p in preds:
        x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
        regions.append({
            "label": "gingivitis",
            "confidence": round(float(p.get("confidence", 0.0)), 3),
            "bbox": [round(x - w / 2, 1), round(y - h / 2, 1), round(x + w / 2, 1), round(y + h / 2, 1)],
        })
    return sorted(regions, key=lambda r: -r["confidence"])


def run(image_path, backend="weights", conf=0.4, weights=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"cannot read {image_path}")

    wp = weights or DEFAULT_WEIGHTS
    if not Path(wp).exists():
        raise SystemExit(f"no local weights at {wp} - train with notebooks/train_model4_kaggle.ipynb")
    preds = _infer_weights(image_path, wp, conf)
    regions = _to_regions(preds)

    top = max((r["confidence"] for r in regions), default=0.0)
    res = {
        "backend": backend,
        "n_regions": len(regions),
        "gingivitis_detected": len(regions) > 0,
        "max_confidence": round(top, 3),
        "regions": regions,
        "disclaimer": "Research prototype / screening aid - not a diagnosis.",
    }
    return img, res


def draw(img, res):
    out = img.copy()
    for r in res["regions"]:
        x0, y0, x1, y1 = map(int, r["bbox"])
        cv2.rectangle(out, (x0, y0), (x1, y1), GINGIVITIS_COLOR, 2)
        lbl = f"gingivitis {r['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, max(y0 - th - 6, 0)), (x0 + tw + 4, y0), GINGIVITIS_COLOR, -1)
        cv2.putText(out, lbl, (x0 + 2, max(y0 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    # Same CLI shape as Models 1/2/3:  python scripts/analyze.py IMG [IMG ...] --out DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="output folder (default: next to each image)")
    ap.add_argument("--backend", choices=["weights"], default="weights",
                    help="weights = local best.pt (offline, default)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--weights", default=None, help="path to a YOLO11 .pt (default models/best.pt)")
    a = ap.parse_args()

    for img_path in a.images:
        p = Path(img_path)
        img, res = run(str(p), a.backend, a.conf, a.weights)
        out = Path(a.out) if a.out else p.parent
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{p.stem}_result.jpg"), draw(img, res))
        (out / f"{p.stem}_result.json").write_text(json.dumps(res, indent=1))
        flag = "GINGIVITIS" if res["gingivitis_detected"] else "none"
        print(f"\n{p.name}: {res['n_regions']} region(s)  [{flag}]  max conf {res['max_confidence']}")
        for r in res["regions"]:
            print(f"  gingivitis {r['confidence']}")
        print(f"  saved -> {out / (p.stem + '_result.jpg')}")


if __name__ == "__main__":
    main()
