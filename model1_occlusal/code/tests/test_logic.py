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

# tooth assignment + duplicate-FDI resolution inside the real analyze() path
a = OPGAnalyzer.__new__(OPGAnalyzer)
a.teeth_conf = a.findings_conf = 0.1; a.review_below = .5; a.teeth_imgsz = a.findings_imgsz = 640; a.device=None; a.findings = object()
teeth = [("36", [100,100,160,300], .9), ("36", [300,100,360,300], .6), ("37", [170,100,230,300], .8), ("35", [102,102,158,298], .5)]
finds = [("caries", [110,120,150,180], .8), ("impacted", [600,50,650,100], .4)]
a._predict = lambda m, img, s, c: teeth if m is not a.findings else finds
a.teeth = "T"
import numpy as np
res = a.analyze(np.zeros((400,800,3),np.uint8))
fd = {t["fdi"] for t in res["teeth"]}
assert fd == {"36","37"}, fd                       # dup 36 + overlapping 35 removed
assert res["findings"][0]["fdi"] == "36" and not res["findings"][0]["needs_review"]
assert res["findings"][1]["fdi"] is None and res["findings"][1]["needs_review"]
assert "38" in res["summary"]["third_molars_not_detected"] and "11" in res["summary"]["not_detected"]
print("analyzer logic OK |", json.dumps(res["summary"]))
