"""Logic tests: perfect predictions -> F1 = 1; wrong tooth -> clinical F1 drops; tooth mapping works."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dentassist.analyzer import OPGAnalyzer
import evaluate as ev

root = Path(sys.argv[1])
gt = json.loads((root / "findings/findings_gt_test.json").read_text())
lookup = {str(root / "findings" / g["image"]): g for g in gt}

class Fake(OPGAnalyzer):
    def __init__(self, wrong_tooth=False):
        self.teeth_conf, self.findings_conf, self.review_below = .35, .25, .5
        self.wrong = wrong_tooth
    def analyze(self, path, include_teeth=True):
        g = lookup[path]
        fs = [{"diagnosis": o["diagnosis"], "label": o["diagnosis"], "box": o["box"], "confidence": .9,
               "fdi": ("11" if self.wrong and o["fdi"] != "11" else o["fdi"]), "needs_review": False} for o in g["objects"]]
        return {"findings": fs, "teeth": [], "timing_ms": {"total": 1}}

out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
r = ev.end_to_end(Fake(), root / "findings", gt, out, 0)
assert r["detection_box+diagnosis"]["overall"]["f1"] == 1.0, r
assert r["clinical_box+diagnosis+tooth"]["overall"]["f1"] == 1.0, r
assert r["fdi_accuracy_on_detected_findings"] == 1.0
r2 = ev.end_to_end(Fake(True), root / "findings", gt, out, 0)
assert r2["detection_box+diagnosis"]["overall"]["f1"] == 1.0
assert r2["clinical_box+diagnosis+tooth"]["overall"]["f1"] < 0.2, r2
print("metric logic OK  | wrong-tooth clinical F1 =", r2["clinical_box+diagnosis+tooth"]["overall"]["f1"])

# analyzer end-to-end with mocked models (new post-processing path)
import numpy as np
from dentassist.postprocess import DEFAULT_THRESHOLDS
a = OPGAnalyzer.__new__(OPGAnalyzer)
a.thr = json.loads(json.dumps(DEFAULT_THRESHOLDS)); a.teeth_conf = 0.4; a.findings_conf = 0.3
a.teeth_imgsz = a.findings_imgsz = 640; a.device = None; a.preprocess = "none"; a.repair_numbers = True
class M: task = "detect"
a.teeth, a.findings = M(), object()
teeth = [{"name": "36", "box": [100,100,160,300], "confidence": .9}, {"name": "36", "box": [300,100,360,300], "confidence": .6},
         {"name": "37", "box": [170,100,230,300], "confidence": .8}, {"name": "35", "box": [102,102,158,298], "confidence": .5}]
finds = [{"name": "caries", "box": [110,120,150,180], "confidence": .8}, {"name": "impacted", "box": [600,50,650,100], "confidence": .4},
         {"name": "caries", "box": [300,100,360,300], "confidence": .25}]
a._predict = lambda m, img, s, c, masks=False: [dict(d) for d in (teeth if m is a.teeth else finds)]
res = a.analyze(np.zeros((400,800,3),np.uint8))
nums = [t["fdi"] for t in res["teeth"]]
assert len(nums) == len(set(nums)) == 3, nums                  # duplicate 36 renumbered, overlapping 35 removed
assert any(f["status"] == "likely" and f["diagnosis"] == "caries" for f in res["findings"])
assert len(res["suppressed_findings"]) == 1                      # 0.25 < floor 0.30 hidden
print("analyzer logic OK |", nums, [(f["fdi"], f["display"]) for f in res["findings"]])
