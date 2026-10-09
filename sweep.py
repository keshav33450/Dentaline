"""Sweep caries detector settings on ONE photo to find what actually detects the caries.
Run from model1_occlusal/code:
   .venv\\Scripts\\python.exe ..\\..\\sweep.py "C:\\path\\to\\the_photo.jpg"
Prints detections for each (imgsz, conf, TTA) combo so we pick the real best.
"""
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
for cand in (here / "model1_occlusal" / "code", here):
    if (cand / "dentassist").exists():
        sys.path.insert(0, str(cand)); break

import cv2
from dentassist import paths as P
from ultralytics import YOLO

det = next((P.WEIGHTS / n for n in ("occlusal_caries_det.pt","occlusal_det.pt","best.pt") if (P.WEIGHTS / n).exists()))
m = YOLO(str(det))
print("detector:", det.name, "| classes:", m.names)

if len(sys.argv) < 2:
    sys.exit("pass the photo path:  python sweep.py \"C:\\path\\to\\photo.jpg\"")
img = cv2.imread(sys.argv[1])
if img is None: sys.exit("cannot read " + sys.argv[1])
print("image:", sys.argv[1], f"{img.shape[1]}x{img.shape[0]}\n")

print(f"{'imgsz':>6} {'conf':>5} {'TTA':>4} | boxes | confidences")
print("-"*60)
best = None
for imgsz in (640, 800, 960, 1280):
    for conf in (0.30, 0.20, 0.10):
        for tta in (False, True):
            r = m.predict(img, conf=conf, iou=0.5, imgsz=imgsz, augment=tta, verbose=False)[0]
            n = 0 if r.boxes is None else len(r.boxes)
            cs = [] if r.boxes is None else sorted((round(float(c),2) for c in r.boxes.conf.cpu().numpy()), reverse=True)
            print(f"{imgsz:>6} {conf:>5} {str(tta):>4} | {n:>5} | {cs}")
            if best is None or n > best[0]:
                best = (n, imgsz, conf, tta)
print("\nMOST detections:", best[0], "at imgsz", best[1], "conf", best[2], "TTA", best[3])
