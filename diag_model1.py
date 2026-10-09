"""Diagnose Model 1 (caries) loading + inference. Run from model1_occlusal/code:
   .venv\\Scripts\\python.exe ..\\..\\diag_model1.py
Prints exactly where it fails, if anywhere."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "model1_occlusal" / "code"))
# allow running from code/ too
here = Path(__file__).resolve().parent
for cand in (here / "model1_occlusal" / "code", here):
    if (cand / "dentassist").exists():
        sys.path.insert(0, str(cand)); code_dir = cand; break
else:
    code_dir = Path.cwd()

import importlib.util
from dentassist import paths as P

def find_det():
    for p in (P.WEIGHTS / "occlusal_caries_det.pt", P.WEIGHTS / "occlusal_det.pt", P.WEIGHTS / "best.pt"):
        if p.exists():
            return p
    return None

def find_sev(det):
    for p in (P.WEIGHTS / "occlusal_severity.pt", P.WEIGHTS / "occlusal_severity_efficientnet_b0.pt"):
        if p.exists() and (det is None or p.resolve() != det.resolve()):
            return p
    return None

print("WEIGHTS dir:", P.WEIGHTS)
det = find_det(); print("detector:", det)
sev = find_sev(det); print("severity ckpt:", sev)

print("\n[1] loading YOLO detector...")
try:
    from ultralytics import YOLO
    m = YOLO(str(det)); print("    detector loaded OK, classes:", m.names)
except Exception:
    traceback.print_exc(); sys.exit("DETECTOR LOAD FAILED")

print("\n[2] loading severity classifier...")
try:
    from dentassist.occlusal.classifier import Classifier
    c = Classifier(str(sev)); print("    severity loaded OK, classes:", c.classes)
except Exception:
    print("    SEVERITY LOAD FAILED (detection will still work):")
    traceback.print_exc()
    c = None

print("\n[3] running analyze on a test image...")
import cv2
spec = importlib.util.spec_from_file_location("occ_detect", str(code_dir / "scripts" / "occ_detect.py"))
occ = importlib.util.module_from_spec(spec); spec.loader.exec_module(occ)
test = None
for cand in [code_dir.parent / "test_images", code_dir / "test_images",
             code_dir.parent.parent / "model3_gingival" / "test_images"]:
    if cand.exists():
        imgs = list(cand.glob("*.jpg")) + list(cand.glob("*.png"))
        if imgs: test = imgs[0]; break
if test is None:
    sys.exit("no test image found - pass one manually")
print("    test image:", test)
img = cv2.imread(str(test))
try:
    _, res = occ.analyze(m, img, 0.20, 0.50, None, sev_model=c)
    print("    ANALYZE OK")
    print("    caries regions:", len(res["caries"]), "| highest:", res["highest"])
except Exception:
    print("    ANALYZE FAILED:")
    traceback.print_exc()
print("\nDONE")
