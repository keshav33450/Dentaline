"""
OPGAnalyzer — runs Model 1 (teeth/FDI) + Model 5 (findings) on a panoramic X-ray and
returns a structured, doctor-reviewable result.

    from dentassist.analyzer import OPGAnalyzer
    a = OPGAnalyzer("weights/teeth_best.pt", "weights/findings_best.pt")
    result = a.analyze("opg.jpg")            # dict (JSON-serialisable)
    overlay = a.draw(image, result)          # BGR numpy image

Works with .pt, .onnx or OpenVINO weights (anything ultralytics.YOLO can load).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .fdi import (FDI_TEETH, FINDING_COLOR_BGR, FINDING_LABEL, FINDINGS,
                  describe_tooth, wisdom_teeth)

DISCLAIMER = ("AI-generated decision support for a qualified dentist. Not a diagnosis. "
              "Every finding must be confirmed by a clinician. Research use only.")


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def overlap_of(a, b) -> float:
    """fraction of box a covered by box b"""
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area > 0 else 0.0


def load_image(src) -> np.ndarray:
    """path | bytes | numpy -> BGR uint8. Handles DICOM (.dcm) when pydicom is installed."""
    if isinstance(src, np.ndarray):
        img = src
    else:
        data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = _read_dicom(data)
    if img is None:
        raise ValueError("unsupported or corrupt image")
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _read_dicom(data: bytes):
    try:
        import io
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        return arr
    except Exception:
        return None


def model_imgsz(weights: str | None, default: int) -> int:
    """Static ONNX / OpenVINO exports have a fixed input size - read it so inference matches export."""
    if not weights:
        return default
    try:
        if str(weights).endswith(".onnx"):
            import onnxruntime as ort
            shp = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"]).get_inputs()[0].shape
            if isinstance(shp[-1], int):
                return int(shp[-1])
        meta = Path(weights) / "metadata.yaml"
        if meta.exists():
            import yaml
            sz = yaml.safe_load(meta.read_text()).get("imgsz")
            if sz:
                return int(sz[-1] if isinstance(sz, (list, tuple)) else sz)
    except Exception:
        pass
    return default


class OPGAnalyzer:
    def __init__(self, teeth_weights: str, findings_weights: str | None = None, *,
                 teeth_imgsz: int = 1024, findings_imgsz: int = 1280,
                 teeth_conf: float = 0.35, findings_conf: float = 0.25,
                 review_below: float = 0.50, device: str | None = None):
        from ultralytics import YOLO
        self.teeth = YOLO(teeth_weights, task="detect")
        self.findings = YOLO(findings_weights, task="detect") if findings_weights else None
        self.teeth_imgsz = model_imgsz(teeth_weights, teeth_imgsz)
        self.findings_imgsz = model_imgsz(findings_weights, findings_imgsz)
        self.teeth_conf, self.findings_conf = teeth_conf, findings_conf
        self.review_below = review_below
        self.device = device

    # ------------------------------------------------------------------ core
    def _predict(self, model, img, imgsz, conf):
        kw = dict(imgsz=imgsz, conf=conf, iou=0.5, verbose=False)
        if self.device:
            kw["device"] = self.device
        r = model.predict(img, **kw)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy, cls, cf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        names = r.names
        return [(names[c], xyxy[i].tolist(), float(cf[i])) for i, c in enumerate(cls)]

    @staticmethod
    def _resolve_teeth(raw):
        """one box per FDI number; drop weaker box when two numbers claim the same tooth."""
        raw = sorted(raw, key=lambda t: -t[2])
        kept, used = [], set()
        for fdi, box, conf in raw:
            if fdi in used:
                continue
            if any(iou(box, k[1]) > 0.6 for k in kept):
                continue
            kept.append((fdi, box, conf))
            used.add(fdi)
        return kept

    def analyze(self, src, include_teeth: bool = True) -> dict:
        t0 = time.perf_counter()
        img = load_image(src)
        H, W = img.shape[:2]

        teeth = self._resolve_teeth(self._predict(self.teeth, img, self.teeth_imgsz, self.teeth_conf))
        t1 = time.perf_counter()
        raw_findings = (self._predict(self.findings, img, self.findings_imgsz, self.findings_conf)
                        if self.findings else [])
        t2 = time.perf_counter()

        findings = []
        for diag, box, conf in sorted(raw_findings, key=lambda t: -t[2]):
            best, best_score = None, 0.0
            for fdi, tbox, _ in teeth:
                s = max(iou(box, tbox), overlap_of(box, tbox) * 0.8)
                if s > best_score:
                    best, best_score = fdi, s
            fdi = best if best_score >= 0.3 else None
            reasons = []
            if conf < self.review_below:
                reasons.append("low confidence")
            if fdi is None:
                reasons.append("could not assign tooth number")
            findings.append({
                "diagnosis": diag, "label": FINDING_LABEL.get(diag, diag),
                "fdi": fdi, "tooth": describe_tooth(fdi) if fdi else None,
                "box": [round(v, 1) for v in box], "confidence": round(conf, 3),
                "tooth_match_score": round(best_score, 3),
                "needs_review": bool(reasons), "review_reasons": reasons,
            })

        present = {t[0] for t in teeth}
        not_seen = [f for f in FDI_TEETH if f not in present]
        chart = {f: {"present": f in present, "findings": []} for f in FDI_TEETH}
        for f in findings:
            if f["fdi"]:
                chart[f["fdi"]]["findings"].append(f["diagnosis"])

        return {
            "image": {"width": W, "height": H},
            "teeth": [{"fdi": f, "box": [round(v, 1) for v in b], "confidence": round(c, 3),
                       **describe_tooth(f)} for f, b, c in sorted(teeth, key=lambda t: t[0])]
            if include_teeth else [],
            "findings": findings,
            "summary": {
                "teeth_detected": len(teeth),
                "not_detected": [f for f in not_seen if f not in wisdom_teeth()],
                "third_molars_not_detected": [f for f in not_seen if f in wisdom_teeth()],
                "findings_count": {k: sum(f["diagnosis"] == k for f in findings) for k in FINDINGS},
                "needs_review": sum(f["needs_review"] for f in findings),
            },
            "chart": chart,
            "timing_ms": {"teeth": round((t1 - t0) * 1000), "findings": round((t2 - t1) * 1000),
                          "total": round((time.perf_counter() - t0) * 1000)},
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------------------ overlay
    @staticmethod
    def draw(img, result: dict, show_teeth: bool = True) -> np.ndarray:
        img = load_image(img).copy()
        H, W = img.shape[:2]
        th = max(1, round(max(H, W) / 1000))
        fs = max(0.4, max(H, W) / 2600)
        if show_teeth:
            for t in result["teeth"]:
                x0, y0, x1, y1 = map(int, t["box"])
                cv2.rectangle(img, (x0, y0), (x1, y1), (170, 200, 170), th)
                cv2.putText(img, t["fdi"], (x0 + 2, y0 + int(22 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX,
                            fs, (140, 255, 140), th, cv2.LINE_AA)
        for f in result["findings"]:
            x0, y0, x1, y1 = map(int, f["box"])
            color = FINDING_COLOR_BGR.get(f["diagnosis"], (0, 0, 255))
            cv2.rectangle(img, (x0, y0), (x1, y1), color, th * 3)
            label = f"{f['fdi'] or '?'} {f['label']} {f['confidence']:.2f}" + (" [review]" if f["needs_review"] else "")
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            yb = max(tht + 6, y0 - 4)
            cv2.rectangle(img, (x0, yb - tht - 6), (x0 + tw + 6, yb + 2), color, -1)
            cv2.putText(img, label, (x0 + 3, yb - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        return img
