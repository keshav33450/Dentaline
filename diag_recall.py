"""Compare caries detection recall: old (floor 0.30, no TTA) vs new (floor 0.15, TTA).
Run from model1_occlusal/code:
   .venv\\Scripts\\python.exe ..\\..\\diag_recall.py [optional_image_path]
"""
import sys, importlib.util
from pathlib import Path
here = Path(__file__).resolve().parent
for cand in (here / "model1_occlusal" / "code", here):
    if (cand / "dentassist").exists():
        sys.path.insert(0, str(cand)); code_dir = cand; break
else:
    code_dir = Path.cwd()

import cv2
from dentassist import paths as P
from ultralytics import YOLO

def find_det():
    for p in (P.WEIGHTS / "occlusal_caries_det.pt", P.WEIGHTS / "occlusal_det.pt", P.WEIGHTS / "best.pt"):
        if p.exists(): return p

det = find_det()
m = YOLO(str(det))

# pick image: arg, else a test image
img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if img_path is None:
    for cand in [code_dir.parent / "test_images"]:
        imgs = sorted(cand.glob("*.jpg")) + sorted(cand.glob("*.png"))
        if imgs: img_path = imgs[0]; break
img = cv2.imread(str(img_path))
print("image:", img_path, img.shape[1], "x", img.shape[0])

def count(conf, imgsz, aug):
    r = m.predict(img, conf=conf, iou=0.5, imgsz=imgsz, augment=aug, verbose=False)[0]
    n = 0 if r.boxes is None else len(r.boxes)
    confs = [] if r.boxes is None else [round(float(c),2) for c in r.boxes.conf.cpu().numpy()]
    return n, sorted(confs, reverse=True)

n0, c0 = count(0.30, 640, False)
n1, c1 = count(0.15, 960, True)
print(f"\nOLD (floor .30, 640, no TTA): {n0} caries boxes  confs={c0}")
print(f"NEW (floor .15, 960, +TTA ): {n1} caries boxes  confs={c1}")
print(f"\n{'IMPROVED +'+str(n1-n0)+' detections' if n1>n0 else 'same count' if n1==n0 else 'fewer'}")
