#!/usr/bin/env python3
"""
Model 2 quick test: caries severity on intraoral / occlusal PHOTOS with the v1 severity detector.

  python scripts/occ_detect.py photo1.jpg [photo2.jpg ...] [--out folder] [--show-sound]

For each photo writes  <name>_caries.jpg  (photo + side panel)  and  <name>_caries.json
  >= 0.50  -> "Advanced caries - 81%"                     (likely)
  0.30-0.50 -> "Possible moderate caries - 42% (needs review)"
  < 0.30   -> hidden
Research use only - AI screening aid, not a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402
from dentassist.occlusal.preprocess import preprocess, quality_check  # noqa: E402

LABEL = {"no_caries": "No caries", "mild": "Mild caries", "moderate": "Moderate caries", "advanced": "Advanced caries"}
COLOR = {"no_caries": (90, 180, 60), "mild": (0, 215, 255), "moderate": (0, 130, 255), "advanced": (40, 40, 220), "caries": (0, 165, 255)}  # BGR
RANK = {"advanced": 3, "moderate": 2, "mild": 1, "caries": 1, "no_caries": 0}
F = cv2.FONT_HERSHEY_SIMPLEX


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0


SEV_NAMES = {0: "no_caries", 1: "mild", 2: "moderate", 3: "advanced"}


def _grade(sev_model, img_rgb_crop, imgsz=224):
    """severity model on a tooth/lesion crop -> (severity, confidence). Handles cls or det checkpoints."""
    if sev_model is None or img_rgb_crop.size == 0:
        return None, None
    r = sev_model.predict(img_rgb_crop, imgsz=imgsz, verbose=False)[0]
    if getattr(r, "probs", None) is not None:                     # classifier
        i = int(r.probs.top1); return r.names[i], round(float(r.probs.top1conf), 3)
    if getattr(r, "boxes", None) is not None and len(r.boxes):    # detector -> most confident box
        j = int(r.boxes.conf.argmax()); c = int(r.boxes.cls[j])
        return r.names[c], round(float(r.boxes.conf[j]), 3)
    return None, None


def analyze(model, img_bgr, floor=0.30, likely=0.50, imgsz=1024, sev_model=None):
    """model = caries detector (class 'caries') OR the 4-class severity detector.
    If it is the caries detector and sev_model is given, each lesion box is graded for severity."""
    q = quality_check(img_bgr)
    img = preprocess(img_bgr)
    r = model.predict(img, conf=floor, iou=0.5, imgsz=imgsz, verbose=False)[0]
    caries_mode = set(model.names.values()) == {"caries"}
    dets = []
    if r.boxes is not None:
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            box = [round(float(v), 1) for v in b]
            if caries_mode:
                sev, sconf = "caries", None
                if sev_model is not None:
                    x0, y0, x1, y1 = [int(max(0, v)) for v in box]
                    sv, sc = _grade(sev_model, cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
                    if sv and sv != "no_caries":
                        sev, sconf = sv, sc
                dets.append({"severity": sev, "box": box, "confidence": round(float(cf), 3), "severity_conf": sconf})
            else:
                dets.append({"severity": r.names[int(c)], "box": box, "confidence": round(float(cf), 3), "severity_conf": None})
    # overlapping boxes -> keep the more confident (class-agnostic NMS)
    kept = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        if all(iou(d["box"], k["box"]) < 0.5 for k in kept):
            kept.append(d)
    for d in kept:
        d["status"] = "likely" if d["confidence"] >= likely else "possible"
        name = LABEL.get(d["severity"], "Caries" if d["severity"] == "caries" else d["severity"])
        pct = round(d["confidence"] * 100)
        sev_bit = f" ({d['severity']} {round(d['severity_conf']*100)}%)" if d.get("severity_conf") else ""
        base = f"{name}{sev_bit} - {pct}%"
        d["display"] = base if d["status"] == "likely" else f"Possible {name.lower()}{sev_bit} - {pct}% (needs review)"
    caries = [d for d in kept if d["severity"] != "no_caries"]
    worst = max(caries, key=lambda d: (RANK.get(d["severity"], 1), d["confidence"]), default=None)
    return img, {
        "quality": q, "teeth_found": len(kept), "mode": "caries+severity" if caries_mode else "severity",
        "caries": sorted(caries, key=lambda d: (-RANK.get(d["severity"], 1), -d["confidence"])),
        "sound_teeth": [d for d in kept if d["severity"] == "no_caries"],
        "summary": {k: sum(d["severity"] == k for d in kept) for k in list(LABEL) + ["caries"]},
        "highest": worst["display"] if worst else "No caries detected",
        "settings": {"hidden_below": floor, "likely_from": likely},
        "disclaimer": "AI screening aid - not a diagnosis. Confirm clinically (ICDAS).",
    }


def draw(img, res, show_sound=False):
    img = img.copy()
    H, W = img.shape[:2]
    if H < 1000:
        k = 1000 / H
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        H, W = img.shape[:2]
    else:
        k = 1.0
    s = H / 1000
    th = max(2, round(2 * s))
    items = res["caries"] + (res["sound_teeth"] if show_sound else [])
    placed = []
    for i, d in enumerate(items, 1):
        x0, y0, x1, y1 = [int(v * k) for v in d["box"]]
        col = COLOR.get(d["severity"], (0, 0, 255))
        if d["status"] == "likely":
            cv2.rectangle(img, (x0, y0), (x1, y1), col, th, cv2.LINE_AA)
        else:
            for xa in range(x0, x1, 14):
                cv2.line(img, (xa, y0), (min(xa + 7, x1), y0), col, th); cv2.line(img, (xa, y1), (min(xa + 7, x1), y1), col, th)
            for ya in range(y0, y1, 14):
                cv2.line(img, (x0, ya), (x0, min(ya + 7, y1)), col, th); cv2.line(img, (x1, ya), (x1, min(ya + 7, y1)), col, th)
        r = int(14 * s) + 6
        cx, cy = (x0 + x1) // 2, y0 - r - 4
        for n in range(10):
            dx = ((n + 1) // 2) * (2 * r + 3) * (1 if n % 2 else -1)
            if all(abs(cx + dx - px) > 2 * r or abs(cy - py) > 2 * r for px, py in placed):
                break
        c = (int(np.clip(cx + dx, r, W - r)), int(np.clip(cy, r, H - r)))
        placed.append(c)
        cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
        cv2.circle(img, c, r, (255, 255, 255), 2, cv2.LINE_AA)
        (tw, tht), _ = cv2.getTextSize(str(i), F, 0.6 * s + 0.2, 2)
        cv2.putText(img, str(i), (c[0] - tw // 2, c[1] + tht // 2), F, 0.6 * s + 0.2, (255, 255, 255), 2, cv2.LINE_AA)
    PW = max(620, int(W * 0.42))
    panel = np.full((H, PW, 3), 250, np.uint8)
    y = int(45 * s)
    fs = 0.75 * s

    def line(t, col=(30, 30, 30), sc=1.0, w=1, ind=0):
        nonlocal y
        cv2.putText(panel, t, (int(22 * s) + ind, y), F, fs * sc, col, w, cv2.LINE_AA)
        y += int(38 * s * sc)

    line("DentAssist - caries screening (photo)", w=2, sc=1.05)
    q = res["quality"]
    line("Photo quality: OK" if q["ok"] else "Photo quality: " + ", ".join(q["reasons"]),
         (40, 130, 40) if q["ok"] else (0, 0, 200), sc=0.85)
    line(f"Teeth analysed: {res['teeth_found']}", (70, 70, 70), sc=0.85)
    line(f"Highest: {res['highest']}", w=2, sc=0.9)
    y += int(10 * s)
    line("Findings", w=2)
    if not res["caries"]:
        line("No caries above threshold", (90, 90, 90))
    for i, d in enumerate(items, 1):
        if y > H - 260 * s:
            line(f"... {len(items) - i + 1} more in the JSON", (90, 90, 90), sc=0.8)
            break
        col = COLOR.get(d["severity"], (0, 0, 255))
        cv2.circle(panel, (int(34 * s), y - int(9 * s)), int(13 * s), col, -1, cv2.LINE_AA)
        cv2.putText(panel, str(i), (int(28 * s), y - int(3 * s)), F, 0.5 * s, (255, 255, 255), 1, cv2.LINE_AA)
        line(d["display"], (30, 30, 30) if d["status"] == "likely" else (0, 110, 200),
             w=2 if d["status"] == "likely" else 1, sc=0.85, ind=int(34 * s))
    y = max(y + int(10 * s), H - int(230 * s))
    line("Legend", w=2, sc=0.9)
    for kk in ("mild", "moderate", "advanced"):
        cv2.rectangle(panel, (int(22 * s), y - int(16 * s)), (int(46 * s), y + int(4 * s)), COLOR[kk], -1)
        line(LABEL[kk] + {"mild": " (ICDAS 1-2)", "moderate": " (ICDAS 3-4)", "advanced": " (ICDAS 5-6)"}[kk], sc=0.8, ind=int(34 * s))
    line("solid = likely   dashed = possible (review)", (80, 80, 80), sc=0.75)
    line("AI screening aid - not a diagnosis.", (0, 0, 170), sc=0.75)
    return np.hstack([img, panel])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--weights", default=str(P.WEIGHTS / "occlusal_caries_det.pt"),
                    help="caries detector (v2); falls back to occlusal_det.pt if missing")
    ap.add_argument("--severity", default=str(P.WEIGHTS / "occlusal_severity.pt"),
                    help="severity model to grade each lesion; optional")
    ap.add_argument("--show-sound", action="store_true", help="also draw teeth graded 'no caries'")
    ap.add_argument("--floor", type=float, default=0.30)
    ap.add_argument("--likely", type=float, default=0.50)
    a = ap.parse_args()
    wpath = Path(a.weights)
    if not wpath.exists():
        alt = P.WEIGHTS / "occlusal_det.pt"
        if alt.exists():
            print(f"note: {wpath.name} not found, using {alt.name}"); wpath = alt
        else:
            sys.exit(f"Model not found: {a.weights}")
    from ultralytics import YOLO
    model = YOLO(str(wpath))
    sev_model = None
    for cand in [a.severity, str(P.WEIGHTS / "occlusal_severity_efficientnet_b0.pt"), str(P.WEIGHTS / "occlusal_det.pt")]:
        if cand and Path(cand).exists() and Path(cand).resolve() != wpath.resolve():
            sev_model = YOLO(cand); break
    imgsz = (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 1024)
    for f in a.images:
        p = Path(f)
        raw = cv2.imread(str(p))
        if raw is None:
            print(f"cannot read {p}"); continue
        img, res = analyze(model, raw, a.floor, a.likely, imgsz, sev_model=sev_model)
        out = Path(a.out) if a.out else p.parent
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{p.stem}_caries.jpg"), draw(img, res, a.show_sound), [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / f"{p.stem}_caries.json").write_text(json.dumps(res, indent=1))
        print(f"\n{p.name}: {res['teeth_found']} teeth | {res['summary']} | {res['highest']}")
        for d in res["caries"]:
            print("   ", d["display"])


if __name__ == "__main__":
    main()
