"""
Clinical post-processing for OPG detections (no retraining needed).

1. FDI repair   - one number per tooth, correct arch, correct left-right order
                  (image left = patient right:  upper 18..11 | 21..28,  lower 48..41 | 31..38)
2. Findings     - attach each finding to one tooth, drop duplicates, caries vs deep caries = one call,
                  class-specific thresholds -> status  likely / possible / hidden
3. Plausibility - impacted findings on unusual teeth are flagged for review
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- thresholds
DEFAULT_THRESHOLDS = {
    "teeth": 0.40,               # tooth boxes below this are ignored
    "floor": 0.30,               # findings below this are hidden (kept in JSON 'suppressed')
    "findings": {                # >= threshold -> 'likely'; floor..threshold -> 'possible' (needs review)
        "caries": 0.50, "deep_caries": 0.50, "periapical_lesion": 0.45, "impacted": 0.55},
    "preprocess": "none",        # none | clahe   (set by calibration on the validation set)
    "source": "defaults (not yet calibrated)",
}


def load_thresholds(weights_dir) -> dict:
    t = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    f = Path(weights_dir) / "thresholds.json" if weights_dir else None
    if f and f.exists():
        user = json.loads(f.read_text())
        t.update({k: v for k, v in user.items() if k != "findings"})
        t["findings"].update(user.get("findings", {}))
    return t


# ---------------------------------------------------------------- geometry
def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cover(a, b) -> float:
    """fraction of box a inside box b"""
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area > 0 else 0.0


def _cx(b):
    return (b[0] + b[2]) / 2


def _cy(b):
    return (b[1] + b[3]) / 2


# ---------------------------------------------------------------- FDI repair
def slot_of(fdi: str) -> tuple[str, int]:
    """arch + left-to-right slot 0..15 on the image."""
    q, n = int(fdi[0]), int(fdi[1])
    arch = "upper" if q in (1, 2) else "lower"
    slot = (8 - n) if q in (1, 4) else (7 + n)
    return arch, slot


def fdi_of(arch: str, slot: int) -> str:
    if arch == "upper":
        return f"1{8 - slot}" if slot <= 7 else f"2{slot - 7}"
    return f"4{8 - slot}" if slot <= 7 else f"3{slot - 7}"


def _occlusal_curve(teeth):
    """y(x) of the occlusal plane from upper-tooth bottoms and lower-tooth tops."""
    xs, ys = [], []
    for t in teeth:
        arch, _ = slot_of(t["fdi"])
        xs.append(_cx(t["box"]))
        ys.append(t["box"][3] if arch == "upper" else t["box"][1])
    if len(xs) < 4:
        return None
    deg = 2 if len(xs) >= 8 else 1
    try:
        return np.poly1d(np.polyfit(xs, ys, deg))
    except Exception:
        return None


def _assign_slots(items):
    """Optimal strictly-increasing slot assignment (items sorted left->right).
    cost = confidence of every tooth whose number is changed (+ tiny distance term) -> minimum change."""
    n, S = len(items), 16
    INF = 1e18
    dp = [[INF] * S for _ in range(n)]
    back = [[-1] * S for _ in range(n)]
    for i, it in enumerate(items):
        p, w = it["_slot"], it["confidence"]
        for s in range(S):
            c = (w if s != p else 0.0) + 0.01 * abs(s - p)
            if i == 0:
                dp[i][s] = c
                continue
            best, arg = INF, -1
            for q in range(s):
                if dp[i - 1][q] < best:
                    best, arg = dp[i - 1][q], q
            if arg >= 0:
                dp[i][s], back[i][s] = best + c, arg
    s = min(range(S), key=lambda k: dp[n - 1][k])
    slots = [0] * n
    for i in range(n - 1, -1, -1):
        slots[i] = s
        s = back[i][s]
    return slots


def repair_fdi(teeth: list[dict]) -> list[dict]:
    """teeth: [{'fdi','box','confidence'}] -> corrected 'fdi' (+ 'fdi_original', 'renumbered', 'number_uncertain').
    Every jaw must read left->right in anatomical order with no duplicates; missing teeth simply leave gaps."""
    if not teeth:
        return teeth
    for t in teeth:
        t["fdi_original"] = t["fdi"]
        t["renumbered"] = False
        t.setdefault("number_uncertain", False)
    curve = _occlusal_curve(teeth)
    by_arch = {"upper": [], "lower": []}
    for t in teeth:
        arch, slot = slot_of(t["fdi"])
        if curve is not None:
            h = t["box"][3] - t["box"][1]
            geo = "upper" if _cy(t["box"]) < curve(_cx(t["box"])) else "lower"
            if geo != arch and abs(_cy(t["box"]) - curve(_cx(t["box"]))) > 0.25 * h:
                arch = geo                     # clearly on the other jaw -> trust the position
        t["_slot"] = slot
        by_arch[arch].append(t)
    out = []
    for arch, items in by_arch.items():
        items.sort(key=lambda t: _cx(t["box"]))
        extra = []
        if len(items) > 16:                    # impossible anatomy: drop the weakest extras
            items.sort(key=lambda t: -t["confidence"])
            items, extra = sorted(items[:16], key=lambda t: _cx(t["box"])), items[16:]
        for t, s in zip(items, _assign_slots(items) if items else []):
            new = fdi_of(arch, s)
            if new != t["fdi"]:
                t["fdi"], t["renumbered"] = new, True
                t["number_uncertain"] = t["confidence"] >= 0.6   # a confident box was overruled -> show
            out.append(t)
    for t in out:
        t.pop("_slot", None)
    return sorted(out, key=lambda t: t["fdi"])


def dedupe_teeth(raw: list[dict], iou_thr: float = 0.55) -> list[dict]:
    """class-agnostic NMS: two numbers on the same tooth -> keep the more confident box."""
    kept = []
    for t in sorted(raw, key=lambda t: -t["confidence"]):
        if all(iou(t["box"], k["box"]) < iou_thr for k in kept):
            kept.append(t)
    return kept


# ---------------------------------------------------------------- findings
EXCLUSIVE = [{"caries", "deep_caries"}]            # one call per tooth for these
IMPACTION_TEETH = {3, 5, 8}                        # canines, 2nd premolars, third molars



def _infer_fdi(f: dict, teeth: list[dict]) -> str | None:
    """Finding that overlaps no detected tooth: impaction beyond the last molar -> the undetected
    third molar of that quadrant; anything else -> the nearest tooth if it is within ~1.5 tooth widths."""
    if not teeth:
        return None
    fx, fy = _cx(f["box"]), _cy(f["box"])
    centrals = [_cx(t["box"]) for t in teeth if t["fdi"][1] == "1"]
    xs = sorted(centrals or [_cx(t["box"]) for t in teeth])
    midx = xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2
    near = min(teeth, key=lambda t: (_cx(t["box"]) - fx) ** 2 + (_cy(t["box"]) - fy) ** 2)
    upper = near["fdi"][0] in "12"
    q = ("1" if fx < midx else "2") if upper else ("4" if fx < midx else "3")
    have = {t["fdi"] for t in teeth}
    if f["diagnosis"] == "impacted":
        quad = [t for t in teeth if t["fdi"][0] == q]
        distal = max(quad, key=lambda t: int(t["fdi"][1]), default=None)
        beyond = distal is None or abs(fx - midx) >= abs(_cx(distal["box"]) - midx) - 0.5 * (distal["box"][2] - distal["box"][0])
        if f"{q}8" not in have and beyond:
            return f"{q}8"
    widths = sorted(t["box"][2] - t["box"][0] for t in teeth)
    w = widths[len(widths) // 2]
    d = ((_cx(near["box"]) - fx) ** 2 + (_cy(near["box"]) - fy) ** 2) ** 0.5
    return near["fdi"] if d <= 1.5 * w else None


def attach_and_filter(findings: list[dict], teeth: list[dict], thr: dict) -> tuple[list[dict], list[dict]]:
    """-> (shown, suppressed). Each finding gets fdi, status ('likely'|'possible'), display box clipped to its tooth."""
    for f in findings:
        best, score = None, 0.0
        for t in teeth:
            s = max(iou(f["box"], t["box"]), cover(f["box"], t["box"]) * 0.8)
            if s > score:
                best, score = t, s
        if score < 0.3:   # oversized box around several teeth -> the tooth nearest its centre
            inside = [t for t in teeth if cover(t["box"], f["box"]) >= 0.8]
            if inside:
                fx, fy = _cx(f["box"]), _cy(f["box"])
                best = min(inside, key=lambda t: (_cx(t["box"]) - fx) ** 2 + (_cy(t["box"]) - fy) ** 2)
                score = 0.3
        f["fdi"] = best["fdi"] if best is not None and score >= 0.3 else None
        f["tooth_match_score"] = round(score, 3)
        f["number_inferred"] = False
        if f["fdi"] is None:
            inf = _infer_fdi(f, teeth)
            if inf:
                f["fdi"], f["number_inferred"] = inf, True
                best = next((t for t in teeth if t["fdi"] == inf), None)
                score = 0.3 if best is not None else 0.0
        if best is not None and score >= 0.3:
            tb = best["box"]
            f["display_box"] = [max(f["box"][0], tb[0]), max(f["box"][1], tb[1]),
                                min(f["box"][2], tb[2]), min(f["box"][3], tb[3])]
        else:
            f["display_box"] = f["box"]

    # duplicates: same tooth + same (or mutually exclusive) diagnosis -> keep the most confident
    kept, suppressed = [], []
    for f in sorted(findings, key=lambda f: -f["confidence"]):
        group = next((g for g in EXCLUSIVE if f["diagnosis"] in g), {f["diagnosis"]})
        dup = any(k["diagnosis"] in group and ((f["fdi"] and k["fdi"] == f["fdi"]) or iou(f["box"], k["box"]) > 0.5)
                  for k in kept)
        if dup:
            f["suppressed_reason"] = "duplicate of a stronger finding on the same tooth"
            suppressed.append(f)
        else:
            kept.append(f)

    shown = []
    for f in kept:
        cthr = thr["findings"].get(f["diagnosis"], 0.5)
        reasons = []
        if f["confidence"] < thr["floor"]:
            f["suppressed_reason"] = f"confidence below {thr['floor']:.2f}"
            suppressed.append(f)
            continue
        f["status"] = "likely" if f["confidence"] >= cthr else "possible"
        if f["status"] == "possible":
            reasons.append(f"confidence below {cthr:.2f}")
        if f["fdi"] is None:
            reasons.append("not linked to a tooth")
        elif f.get("number_inferred"):
            reasons.append("tooth number inferred from position")
        if f["diagnosis"] == "impacted" and f["fdi"] and int(f["fdi"][1]) not in IMPACTION_TEETH:
            reasons.append("impaction unusual for this tooth - verify")
            f["status"] = "possible"
        f["needs_review"] = bool(reasons)
        f["review_reasons"] = reasons
        shown.append(f)
    return shown, suppressed
