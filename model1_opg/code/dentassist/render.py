"""
Clean OPG result image:  X-ray (left)  +  result panel (right).

On the X-ray
  teeth     : thin grey outline + small FDI badge (badges nudged apart so anterior teeth stay readable)
  findings  : coloured outline clipped to the tooth, solid = likely, dashed = possible/needs review,
              only a numbered marker on the image  (1) (2) ...  - details live in the panel
Panel
  numbered findings  "1  36  Deep caries  87%  Likely"  /  "Possible caries 38% - review"
  legend, tooth count, teeth not detected, disclaimer
"""
from __future__ import annotations

import cv2
import numpy as np

STYLE = {  # BGR
    "tooth": (185, 185, 185),
    "caries": (0, 165, 255),          # orange
    "deep_caries": (40, 40, 230),     # red
    "periapical_lesion": (200, 60, 200),  # purple
    "impacted": (230, 150, 30),       # blue
}
NAME = {"caries": "Caries", "deep_caries": "Deep caries", "periapical_lesion": "Periapical lesion",
        "impacted": "Impacted tooth"}
F = cv2.FONT_HERSHEY_SIMPLEX


def _dashed_rect(img, p0, p1, color, th, dash):
    x0, y0 = p0
    x1, y1 = p1
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) / dash))
        for i in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
            e = (int(a[0] + (b[0] - a[0]) * min(i + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(i + 1, n) / n))
            cv2.line(img, s, e, color, th, cv2.LINE_AA)


def _badge(img, text, center, fs, th, fg, bg):
    (tw, tht), bl = cv2.getTextSize(text, F, fs, th)
    x, y = int(center[0] - tw / 2), int(center[1] + tht / 2)
    cv2.rectangle(img, (x - 3, y - tht - 3), (x + tw + 3, y + bl), bg, -1)
    cv2.putText(img, text, (x, y), F, fs, fg, th, cv2.LINE_AA)
    return (x - 3, y - tht - 3, x + tw + 3, y + bl)


def _overlaps(r, placed):
    return any(not (r[2] < p[0] or r[0] > p[2] or r[3] < p[1] or r[1] > p[3]) for p in placed)


def _scaled(result, k):
    import copy
    r = copy.deepcopy(result)
    for t in r.get("teeth", []):
        t["box"] = [v * k for v in t["box"]]
    for f in r.get("findings", []):
        f["box"] = [v * k for v in f["box"]]
        if "display_box" in f:
            f["display_box"] = [v * k for v in f["display_box"]]
    return r


def render(img_bgr: np.ndarray, result: dict, show_teeth: bool = True, show_possible: bool = True,
           min_height: int = 1100) -> np.ndarray:
    img = img_bgr.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[0] < min_height:                       # small uploads: draw on an upscaled copy (readable text)
        k = min_height / img.shape[0]
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        result = _scaled(result, k)
    H, W = img.shape[:2]
    s = max(H, W) / 2000
    th = max(1, round(1.5 * s))
    fs = 0.55 * s * 1.4

    # ---- teeth
    placed = []
    if show_teeth:
        for t in result.get("teeth", []):
            x0, y0, x1, y1 = map(int, t["box"])
            cv2.rectangle(img, (x0, y0), (x1, y1), STYLE["tooth"], th, cv2.LINE_AA)
            upper = t["fdi"][0] in "12"
            cx = (x0 + x1) / 2
            cy = y0 + 0.14 * (y1 - y0) if upper else y1 - 0.14 * (y1 - y0)
            fsz = min(fs, max(0.35 * s, (x1 - x0) / 60))
            (tw, tht), _ = cv2.getTextSize(t["fdi"], F, fsz, th)
            for k in range(6):                                  # nudge until it does not collide
                dy = (k + 1) // 2 * (tht + 8) * (1 if k % 2 else -1) * (1 if upper else -1)
                r = (int(cx - tw / 2 - 3), int(cy + dy - tht / 2 - 3), int(cx + tw / 2 + 3), int(cy + dy + tht / 2 + 3))
                if not _overlaps(r, placed):
                    break
            fg = (0, 200, 255) if t.get("renumbered") or t.get("number_uncertain") else (255, 255, 255)
            placed.append(_badge(img, t["fdi"], (cx, cy + dy), fsz, max(1, th - 1), fg, (60, 60, 60)))

    # ---- findings (outline on the tooth + numbered marker)
    items = [f for f in result.get("findings", []) if show_possible or f.get("status") == "likely"]
    items.sort(key=lambda f: (f.get("status") != "likely", -f["confidence"]))
    tint = img.copy()
    for f in items:
        if f.get("status") == "likely":
            x0, y0, x1, y1 = map(int, f.get("display_box", f["box"]))
            cv2.rectangle(tint, (x0, y0), (x1, y1), STYLE.get(f["diagnosis"], (0, 0, 255)), -1)
    img = cv2.addWeighted(tint, 0.22, img, 0.78, 0)
    r = int(15 * s) + 6
    markers = list(placed)
    upper_fdi = {t["fdi"] for t in result.get("teeth", []) if t["fdi"][0] in "12"}
    for i, f in enumerate(items, 1):
        col = STYLE.get(f["diagnosis"], (0, 0, 255))
        x0, y0, x1, y1 = map(int, f.get("display_box", f["box"]))
        if f.get("status") == "likely":
            cv2.rectangle(img, (x0, y0), (x1, y1), col, th * 2, cv2.LINE_AA)
        else:
            _dashed_rect(img, (x0, y0), (x1, y1), col, th * 2, max(8, int(12 * s)))
        up = f.get("fdi") in upper_fdi if f.get("fdi") else (y0 + y1) / 2 < H / 2
        cx = (x0 + x1) // 2
        cy = y0 - r - 6 if up else y1 + r + 6                     # outside the dentition
        for k in range(12):                                        # avoid other markers / badges
            dx = ((k + 1) // 2) * (2 * r + 4) * (1 if k % 2 else -1)
            box = (cx + dx - r, cy - r, cx + dx + r, cy + r)
            if not _overlaps(box, markers):
                break
        c = (int(np.clip(cx + dx, r, W - r)), int(np.clip(cy, r, H - r)))
        markers.append((c[0] - r, c[1] - r, c[0] + r, c[1] + r))
        cv2.line(img, c, (int(np.clip(c[0], x0, x1)), y0 if up else y1), col, max(1, th), cv2.LINE_AA)
        cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
        cv2.circle(img, c, r, (255, 255, 255), max(1, th), cv2.LINE_AA)
        _badge(img, str(i), c, fs * 0.8, th, (255, 255, 255), col)

    # ---- panel
    PW = int(max(700, W * 0.36))
    panel = np.full((H, PW, 3), 250, np.uint8)
    pf = max(0.9, H / 1000)
    lh = int(34 * pf)
    y = int(40 * pf)

    def line(text, color=(30, 30, 30), bold=False, indent=0, scale=1.0):
        nonlocal y
        cv2.putText(panel, text, (int(20 * pf) + indent, y), F, pf * 0.62 * scale, color, 2 if bold and pf >= 0.9 else 1, cv2.LINE_AA)
        y += int(lh * scale)

    line("DentAssist - OPG analysis", bold=True, scale=1.15)
    sm = result.get("summary", {})
    line(f"Teeth detected: {sm.get('teeth_detected', 0)}", (70, 70, 70))
    y += lh // 3
    line("Findings", bold=True)
    if not items:
        line("No findings above threshold", (90, 90, 90))
    for i, f in enumerate(items, 1):
        col = STYLE.get(f["diagnosis"], (0, 0, 255))
        cv2.circle(panel, (int(30 * pf), y - int(8 * pf)), int(12 * pf), col, -1, cv2.LINE_AA)
        cv2.putText(panel, str(i), (int(24 * pf), y - int(2 * pf)), F, pf * 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        pct = f"{round(f['confidence'] * 100)}%"
        tooth = f["fdi"] or "?"
        if f.get("status") == "likely":
            txt, c2 = f"{tooth}  {NAME.get(f['diagnosis'], f['diagnosis'])} - {pct}", (30, 30, 30)
        else:
            txt, c2 = f"{tooth}  Possible {NAME.get(f['diagnosis'], f['diagnosis']).lower()} - {pct}", (60, 60, 60)
        line(txt, c2, bold=f.get("status") == "likely", indent=int(28 * pf))
        if f.get("needs_review"):
            line("needs review: " + "; ".join(f.get("review_reasons", [])), (0, 110, 200), indent=int(28 * pf), scale=0.8)
        if y > H - 8 * lh:
            line(f"... {len(items) - i} more in the JSON report", (90, 90, 90))
            break
    y += lh // 2
    line("Legend", bold=True)
    for k in ("caries", "deep_caries", "periapical_lesion", "impacted"):
        cv2.rectangle(panel, (int(20 * pf), y - int(14 * pf)), (int(44 * pf), y + int(4 * pf)), STYLE[k], -1)
        line(NAME[k], indent=int(34 * pf), scale=0.9)
    line("solid = likely    dashed = possible (review)", (80, 80, 80), scale=0.85)
    line("orange number = renumbered / uncertain", (0, 140, 230), scale=0.85)
    nd = sm.get("not_detected", [])
    if nd:
        y += lh // 3
        line("Teeth not detected (possible missing):", bold=True, scale=0.9)
        for k in range(0, len(nd), 8):
            line(" ".join(nd[k:k + 8]), (70, 70, 70), scale=0.9)
    y = max(y, H - int(2.2 * lh))
    line("AI decision support - not a diagnosis.", (0, 0, 160), scale=0.8)
    line("Confirm every finding clinically.", (0, 0, 160), scale=0.8)
    return np.hstack([img, panel])
