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


def enhance(img: np.ndarray, mode: str) -> np.ndarray:
    """optional X-ray preprocessing (must match what the thresholds were calibrated with)."""
    if mode == "clahe":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.fastNlMeansDenoising(g, None, 5, 7, 21)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return img


class OPGAnalyzer:
    """Model 1 (teeth/FDI) + Model 5 (findings) + clinical post-processing (see postprocess.py)."""

    def __init__(self, teeth_weights: str, findings_weights: str | None = None, *,
                 teeth_imgsz: int = 1024, findings_imgsz: int = 1280,
                 teeth_conf: float | None = None, findings_conf: float | None = None,
                 review_below: float | None = None, thresholds: dict | None = None,
                 weights_dir: str | None = None, preprocess: str | None = None,
                 repair_numbers: bool = True, device: str | None = None):
        from ultralytics import YOLO
        from .postprocess import load_thresholds
        self.thr = thresholds or load_thresholds(weights_dir or Path(teeth_weights).parent)
        if teeth_conf is not None:
            self.thr["teeth"] = teeth_conf
        if findings_conf is not None:            # evaluation: lower the floor to score the full PR range
            self.thr["floor"] = findings_conf
        if review_below is not None:
            self.thr["findings"] = {k: review_below for k in self.thr["findings"]}
        self.teeth = YOLO(teeth_weights)
        self.findings = YOLO(findings_weights, task="detect") if findings_weights else None
        self.teeth_imgsz = model_imgsz(teeth_weights, teeth_imgsz)
        self.findings_imgsz = model_imgsz(findings_weights, findings_imgsz)
        self.teeth_conf, self.findings_conf = self.thr["teeth"], self.thr["floor"]
        self.review_below = min(self.thr["findings"].values())
        self.preprocess = preprocess or self.thr.get("preprocess", "none")
        self.repair_numbers = repair_numbers
        self.device = device

    def _predict(self, model, img, imgsz, conf, masks=False):
        kw = dict(imgsz=imgsz, conf=conf, iou=0.5, verbose=False)
        if self.device:
            kw["device"] = self.device
        r = model.predict(img, retina_masks=masks, **kw)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy, cls, cf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        out = []
        for i, c in enumerate(cls):
            d = {"name": r.names[c], "box": xyxy[i].tolist(), "confidence": float(cf[i])}
            if masks and r.masks is not None:     # segmentation model -> tight box + outline from the mask
                m = r.masks.data[i].cpu().numpy().astype(np.uint8)
                if m.shape != img.shape[:2]:
                    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    cnt = max(cs, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                    d["box"] = [float(x), float(y), float(x + w), float(y + h)]
                    d["outline"] = cv2.approxPolyDP(cnt, 2, True).reshape(-1, 2).tolist()
            out.append(d)
        return out

    def analyze(self, src, include_teeth: bool = True) -> dict:
        from .postprocess import attach_and_filter, dedupe_teeth, repair_fdi
        t0 = time.perf_counter()
        img = enhance(load_image(src), self.preprocess)
        H, W = img.shape[:2]
        is_seg = getattr(self.teeth, "task", "detect") == "segment"
        raw_t = self._predict(self.teeth, img, self.teeth_imgsz, self.teeth_conf, masks=is_seg)
        teeth = [{"fdi": d["name"], "box": d["box"], "confidence": d["confidence"],
                  **({"outline": d["outline"]} if "outline" in d else {})}
                 for d in raw_t if len(str(d["name"])) == 2 and str(d["name"]).isdigit()]
        teeth = dedupe_teeth(teeth)
        if self.repair_numbers:
            teeth = repair_fdi(teeth)
        t1 = time.perf_counter()
        raw_f = (self._predict(self.findings, img, self.findings_imgsz, self.findings_conf)
                 if self.findings else [])
        findings = [{"diagnosis": d["name"], "label": FINDING_LABEL.get(d["name"], d["name"]),
                     "box": d["box"], "confidence": d["confidence"]} for d in raw_f]
        findings, suppressed = attach_and_filter(findings, teeth, self.thr)
        t2 = time.perf_counter()

        def clean(f):
            f = dict(f)
            f["confidence"] = round(f["confidence"], 3)
            f["box"] = [round(v, 1) for v in f["box"]]
            f["display_box"] = [round(v, 1) for v in f.get("display_box", f["box"])]
            f["tooth"] = describe_tooth(f["fdi"]) if f.get("fdi") else None
            f["display"] = (f"{f['label']} - {round(f['confidence'] * 100)}%" if f.get("status") == "likely"
                            else f"Possible {f['label'].lower()} - {round(f['confidence'] * 100)}% (needs review)")
            return f

        findings = [clean(f) for f in sorted(findings, key=lambda f: -f["confidence"])]
        present = {t["fdi"] for t in teeth}
        not_seen = [f for f in FDI_TEETH if f not in present]
        chart = {f: {"present": f in present, "findings": []} for f in FDI_TEETH}
        for f in findings:
            if f["fdi"]:
                chart[f["fdi"]]["findings"].append({"diagnosis": f["diagnosis"], "status": f["status"]})
        return {
            "image": {"width": W, "height": H},
            "teeth": [{"fdi": t["fdi"], "box": [round(v, 1) for v in t["box"]], "confidence": round(t["confidence"], 3),
                       "renumbered": t.get("renumbered", False), "number_uncertain": t.get("number_uncertain", False),
                       "model_number": t.get("fdi_original", t["fdi"]),
                       **({"outline": t["outline"]} if "outline" in t else {}), **describe_tooth(t["fdi"])}
                      for t in teeth] if include_teeth else [],
            "findings": findings,
            "suppressed_findings": [{"diagnosis": f["diagnosis"], "confidence": round(f["confidence"], 3),
                                     "fdi": f.get("fdi"), "reason": f.get("suppressed_reason")} for f in suppressed],
            "summary": {
                "teeth_detected": len(teeth),
                "teeth_renumbered": sum(t.get("renumbered", False) for t in teeth),
                "not_detected": [f for f in not_seen if f not in wisdom_teeth()],
                "third_molars_not_detected": [f for f in not_seen if f in wisdom_teeth()],
                "likely": {k: sum(f["diagnosis"] == k and f["status"] == "likely" for f in findings) for k in FINDINGS},
                "possible": {k: sum(f["diagnosis"] == k and f["status"] == "possible" for f in findings) for k in FINDINGS},
                "needs_review": sum(f["needs_review"] for f in findings),
            },
            "chart": chart,
            "settings": {"thresholds": self.thr["findings"], "floor": self.thr["floor"], "teeth": self.thr["teeth"],
                         "preprocess": self.preprocess, "threshold_source": self.thr.get("source")},
            "timing_ms": {"teeth": round((t1 - t0) * 1000), "findings": round((t2 - t1) * 1000),
                          "total": round((time.perf_counter() - t0) * 1000)},
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------------------ overlay
    @staticmethod
    def draw(img, result: dict, show_teeth: bool = True, panel: bool = True) -> np.ndarray:
        from .render import render
        out = render(load_image(img), result, show_teeth=show_teeth)
        return out if panel else out[:, :result["image"]["width"]]
