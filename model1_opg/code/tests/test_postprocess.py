"""FDI repair + findings filtering tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist.postprocess import repair_fdi, attach_and_filter, dedupe_teeth, DEFAULT_THRESHOLDS, slot_of, fdi_of

UP = ["18","17","16","15","14","13","12","11","21","22","23","24","25","26","27","28"]
LO = ["48","47","46","45","44","43","42","41","31","32","33","34","35","36","37","38"]

def mouth(skip=()):
    t = []
    for i, f in enumerate(UP):
        if f not in skip: t.append({"fdi": f, "box": [100+i*110, 300, 190+i*110, 600], "confidence": 0.9})
    for i, f in enumerate(LO):
        if f not in skip: t.append({"fdi": f, "box": [100+i*110, 640, 190+i*110, 940], "confidence": 0.9})
    return t

def fdis(ts): return {t["fdi"] for t in ts}

for f in UP + LO: assert fdi_of(*slot_of(f)) == f

# 1 perfect mouth untouched
r = repair_fdi(mouth()); assert fdis(r) == set(UP + LO) and not any(t["renumbered"] for t in r)
# 2 two neighbours swapped (14 <-> 15) with lower confidence -> fixed
m = mouth(); m[3]["fdi"], m[4]["fdi"] = "14", "15"; m[3]["confidence"] = m[4]["confidence"] = 0.5
r = repair_fdi(m); byx = sorted([t for t in r if t["fdi"][0] in "12"], key=lambda t: t["box"][0])
assert [t["fdi"] for t in byx] == UP, [t["fdi"] for t in byx]
# 3 duplicate number (two teeth both called 36) -> one renumbered to 37
m = mouth(); m[16+13]["fdi"] = "36"; m[16+13]["confidence"] = 0.4
r = repair_fdi(m); assert fdis(r) == set(UP + LO), sorted(fdis(r) ^ set(UP + LO))
# 4 lower tooth labelled as upper number (quadrant error: 46 called 16) -> fixed by position
m = mouth(); m[16+2]["fdi"] = "16"; m[16+2]["confidence"] = 0.45
r = repair_fdi(m); assert fdis(r) == set(UP + LO)
# 5 left/right mix-up on one tooth (26 called 16 on the left side) -> fixed
m = mouth(); m[13]["fdi"] = "16"; m[13]["confidence"] = 0.4
r = repair_fdi(m); assert fdis(r) == set(UP + LO)
# 6 missing teeth stay missing, others untouched
r = repair_fdi(mouth(skip=("18","36","37","48"))); assert fdis(r) == set(UP + LO) - {"18","36","37","48"} and not any(t["renumbered"] for t in r)
# 7 NMS: two numbers on the same tooth -> one box
d = dedupe_teeth([{"fdi":"11","box":[0,0,100,300],"confidence":0.9},{"fdi":"12","box":[5,5,100,300],"confidence":0.6}]); assert len(d) == 1
# 8 findings: duplicate caries+deep caries on one tooth -> one; thresholds -> status; floor hides
teeth = repair_fdi(mouth())
t36 = next(t for t in teeth if t["fdi"] == "36")["box"]; t11 = next(t for t in teeth if t["fdi"] == "11")["box"]
fs = [{"diagnosis":"deep_caries","box":t36,"confidence":0.87}, {"diagnosis":"caries","box":t36,"confidence":0.6},
      {"diagnosis":"caries","box":t11,"confidence":0.38}, {"diagnosis":"caries","box":t11,"confidence":0.2},
      {"diagnosis":"impacted","box":t11,"confidence":0.9},
      {"diagnosis":"periapical_lesion","box":[t36[0]-200, t36[1]-50, t36[2]+200, t36[3]+300],"confidence":0.7}]
shown, supp = attach_and_filter(fs, teeth, DEFAULT_THRESHOLDS)
st = {(f["fdi"], f["diagnosis"]): f for f in shown}
assert ("36","deep_caries") in st and ("36","caries") not in st and st[("36","deep_caries")]["status"] == "likely"
assert st[("11","caries")]["status"] == "possible" and st[("11","caries")]["needs_review"]
assert st[("11","impacted")]["status"] == "possible" and any("unusual" in r for r in st[("11","impacted")]["review_reasons"])
pa = st[("36","periapical_lesion")]; assert pa["display_box"] == t36   # oversized box clipped to the tooth
assert len(supp) == 2
print("postprocess tests OK")
