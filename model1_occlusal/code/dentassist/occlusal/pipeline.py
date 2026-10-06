"""
OcclusalAnalyzer - full inference for one smartphone occlusal photo (proforma Phases II-V):

  quality gate -> pre-processing -> YOLOv8-seg teeth -> masked crops
  -> premolar/molar classifier -> caries severity classifier -> structured result (+ overlay, + PDF)
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .classifier import Classifier
from .crop import tooth_crop
from .labels import SEVERITY_COLOR_BGR, SEVERITY_LABEL, SEVERITY_NOTE
from .preprocess import preprocess, quality_check

DISCLAIMER = ("AI screening aid for a qualified dental professional - not a diagnosis. "
              "Findings must be confirmed clinically (ICDAS examination). Research use only.")


class OcclusalAnalyzer:
    def __init__(self, seg_weights: str, type_weights: str | None, severity_weights: str, *,
                 caries_det_weights: str | None = None, seg_conf: float = 0.4, seg_imgsz: int | None = None,
                 review_below: float = 0.6, device: str | None = None):
        """seg: 1-class 'tooth' (study) or 2-class premolar/molar (public toothseg model).
        type_weights optional when the segmenter already predicts premolar/molar.
        caries_det_weights optional: independent screening detector used as a cross-check."""
        from ultralytics import YOLO
        self.seg = YOLO(seg_weights)
        names = [str(n).lower() for n in self.seg.names.values()]
        self.seg_types = names if set(names) == {"premolar", "molar"} else None
        self.type_clf = Classifier(type_weights, device) if type_weights else None
        if self.type_clf is None and self.seg_types is None:
            raise ValueError("need a type classifier or a premolar/molar segmenter")
        self.sev_clf = Classifier(severity_weights, device)
        self.cdet = YOLO(caries_det_weights) if caries_det_weights else None
        # always infer at the size the segmenter was trained at
        trained = (getattr(self.seg, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 1024)
        self.seg_conf, self.seg_imgsz, self.review_below = seg_conf, seg_imgsz or trained, review_below

    def analyze(self, img_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        q = quality_check(img_bgr)
        img = preprocess(img_bgr)
        H, W = img.shape[:2]
        r = self.seg.predict(img, conf=self.seg_conf, imgsz=self.seg_imgsz, retina_masks=True, verbose=False)[0]
        teeth, crops = [], []
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            seg_cls = r.boxes.cls.cpu().numpy().astype(int)
            order = np.argsort([np.where(m > 0)[1].mean() if (m > 0).any() else 0 for m in masks])  # left -> right
            for k, i in enumerate(order, 1):
                m = cv2.resize(masks[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                c = tooth_crop(img, m * 255)
                if c is None:
                    continue
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(cs, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                teeth.append({"tooth": k, "box": [int(x), int(y), int(x + w), int(y + h)],
                              "polygon": cv2.approxPolyDP(cnt, 2, True).reshape(-1, 2).tolist(),
                              "seg_confidence": round(float(confs[i]), 3), "_mask": m,
                              "_seg_type": self.seg_types[seg_cls[i]] if self.seg_types else None})
                crops.append(c)
        sp = self.sev_clf.predict(crops)
        if self.type_clf is not None:
            tp = self.type_clf.predict(crops)
            tclasses = self.type_clf.classes
        else:   # type straight from the premolar/molar segmenter
            tclasses = ["premolar", "molar"]
            tp = np.array([[t["seg_confidence"], 1 - t["seg_confidence"]] if t["_seg_type"] == "premolar"
                           else [1 - t["seg_confidence"], t["seg_confidence"]] for t in teeth]).reshape(-1, 2)
        cboxes = []
        if self.cdet is not None:
            cr = self.cdet.predict(img, conf=0.25, verbose=False)[0]
            if cr.boxes is not None and len(cr.boxes):
                cboxes = list(zip(cr.boxes.xyxy.cpu().numpy(), cr.boxes.conf.cpu().numpy()))
        for t, a, b in zip(teeth, tp, sp):
            ti, si = int(a.argmax()), int(b.argmax())
            m = t.pop("_mask"); t.pop("_seg_type", None)
            det_conf = 0.0
            for (x0, y0, x1, y1), cf in cboxes:
                x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
                area = max(1, (x1 - x0) * (y1 - y0))
                if x1 > x0 and y1 > y0 and m[y0:y1, x0:x1].sum() / area >= 0.3:   # box lies on this tooth
                    det_conf = max(det_conf, float(cf))
            sev = self.sev_clf.classes[si]
            reasons = []
            if b[si] < self.review_below:
                reasons.append("low severity confidence")
            if a[ti] < self.review_below:
                reasons.append("uncertain tooth type")
            if self.cdet is not None:
                sev_now = self.sev_clf.classes[si]
                if det_conf >= 0.5 and sev_now == "no_caries":
                    reasons.append("screening detector sees caries")
                elif det_conf < 0.1 and sev_now in ("moderate", "advanced"):
                    reasons.append("screening detector sees no caries")
            if not q["ok"]:
                reasons.append("photo quality: " + ", ".join(q["reasons"]))
            t.update({
                "tooth_type": tclasses[ti], "type_confidence": round(float(a[ti]), 3),
                "caries_detector_confidence": round(det_conf, 3) if self.cdet is not None else None,
                "severity": sev, "severity_label": SEVERITY_LABEL[sev], "severity_confidence": round(float(b[si]), 3),
                "severity_probs": {c: round(float(v), 3) for c, v in zip(self.sev_clf.classes, b)},
                "note": SEVERITY_NOTE[sev], "needs_review": bool(reasons), "review_reasons": reasons})
        counts = {c: sum(t["severity"] == c for t in teeth) for c in self.sev_clf.classes}
        worst = next((c for c in reversed(self.sev_clf.classes) if counts.get(c)), None)
        return {
            "quality": q, "image": {"width": W, "height": H}, "teeth": teeth,
            "summary": {"teeth_analysed": len(teeth),
                        "premolars": sum(t["tooth_type"] == "premolar" for t in teeth),
                        "molars": sum(t["tooth_type"] == "molar" for t in teeth),
                        "severity_counts": counts,
                        "highest_severity": SEVERITY_LABEL[worst] if worst else None,
                        "needs_review": sum(t["needs_review"] for t in teeth)},
            "timing_ms": round((time.perf_counter() - t0) * 1000), "disclaimer": DISCLAIMER,
            "_processed_image": img,
        }

    @staticmethod
    def draw(result: dict) -> np.ndarray:
        img = result["_processed_image"].copy()
        th = max(2, round(max(img.shape[:2]) / 500))
        fs = max(0.5, max(img.shape[:2]) / 1600)
        for t in result["teeth"]:
            col = SEVERITY_COLOR_BGR[t["severity"]]
            pts = np.array(t["polygon"], np.int32).reshape(-1, 1, 2)
            ov = img.copy()
            cv2.fillPoly(ov, [pts], col)
            img = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
            cv2.polylines(img, [pts], True, col, th, cv2.LINE_AA)
            x0, y0 = t["box"][:2]
            lab = f"#{t['tooth']} {t['tooth_type'][0].upper()} {t['severity_label']} {t['severity_confidence']:.2f}" + ("*" if t["needs_review"] else "")
            (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            yb = max(tht + 8, y0 - 6)
            cv2.rectangle(img, (x0, yb - tht - 6), (x0 + tw + 8, yb + 4), col, -1)
            cv2.putText(img, lab, (x0 + 4, yb), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        return img


def load_default(weights_dir: str | Path, **kw) -> OcclusalAnalyzer | None:
    """study models first (occlusal_seg.pt), else public ones (occlusal_toothseg.pt)."""
    w = Path(weights_dir)
    seg = next((w / n for n in ("occlusal_seg.pt", "occlusal_toothseg.pt") if (w / n).exists()), None)
    sev = w / "occlusal_severity.pt"
    typ = w / "occlusal_type.pt"
    cdet = w / "occlusal_caries_det.pt"
    if seg is None or not sev.exists():
        return None
    if not typ.exists() and seg.name != "occlusal_toothseg.pt":
        return None
    return OcclusalAnalyzer(str(seg), str(typ) if typ.exists() else None, str(sev),
                            caries_det_weights=str(cdet) if cdet.exists() else None, **kw)
