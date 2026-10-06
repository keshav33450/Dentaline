"""
Phase II - image quality gate + pre-processing for smartphone occlusal photos (OpenCV).

quality_check(img)  -> {"ok": bool, "reasons": [...], "metrics": {...}}
    blur (variance of Laplacian), under/over-exposure, specular reflections
    -> mirrors the proforma's image exclusion criteria so bad photos are flagged, not silently scored.

preprocess(img)     -> cleaned BGR image
    reflection reduction (inpaint specular highlights) -> shadow / uneven-light correction
    -> contrast enhancement (CLAHE on L channel) -> white-patch colour balance -> size normalisation

Thresholds are defaults for 12-50 MP phone photos resized to 1600 px; calibrate on your first
~50 study photos with scripts/occ_quality_report.py and set them in QUALITY_DEFAULTS.
"""
from __future__ import annotations

import cv2
import numpy as np

QUALITY_DEFAULTS = {
    "blur_min": 60.0,          # Laplacian variance below this -> blurred
    "dark_mean_max": 55.0,     # mean luminance below -> under-exposed
    "bright_mean_min": 215.0,  # mean luminance above -> over-exposed
    "clip_frac_max": 0.08,     # fraction of pixels at 0 or 255
    "specular_frac_max": 0.02, # fraction of glare pixels (saliva shine / flash hotspots)
    "work_side": 1600,         # long side used for analysis / processing
}


def _resize_long(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def specular_mask(img: np.ndarray) -> np.ndarray:
    """Glare = small, near-saturated, colourless spots much brighter than their surroundings.
    (Plain bright enamel is large and smooth, so the white top-hat keeps it out.)"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)   # blur: ignore sensor noise
    v, s = hsv[..., 2], hsv[..., 1]
    k = max(15, (min(img.shape[:2]) // 40) | 1)
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = ((v >= 240) & (s <= 40) & (tophat >= 20)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))           # blobs, not single pixels
    return cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)


def quality_check(img: np.ndarray, cfg: dict | None = None) -> dict:
    c = {**QUALITY_DEFAULTS, **(cfg or {})}
    im = _resize_long(img, c["work_side"])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    clip = float(((gray <= 2) | (gray >= 253)).mean())
    spec = float((specular_mask(im) > 0).mean())
    reasons = []
    if blur < c["blur_min"]:
        reasons.append("blurred / out of focus")
    if mean < c["dark_mean_max"]:
        reasons.append("under-exposed")
    if mean > c["bright_mean_min"]:
        reasons.append("over-exposed")
    if clip > c["clip_frac_max"]:
        reasons.append("clipped highlights/shadows")
    if spec > c["specular_frac_max"]:
        reasons.append("strong reflections (saliva / flash glare)")
    return {"ok": not reasons, "reasons": reasons,
            "metrics": {"blur_laplacian_var": round(blur, 1), "mean_luminance": round(mean, 1),
                        "clipped_fraction": round(clip, 4), "specular_fraction": round(spec, 4)}}


def reduce_reflections(img: np.ndarray) -> np.ndarray:
    m = specular_mask(img)
    if m.mean() < 0.05:            # (0-255 scale) nothing meaningful to fix
        return img
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)


def correct_shadows(img: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """Flatten slow illumination fall-off (vignetting, cheek shadow) WITHOUT flattening the teeth:
    background light is estimated at a scale much larger than a tooth (sigma = 1/4 of the short side)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    h, w = L.shape
    small = cv2.resize(L, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), max(2.0, min(small.shape) / 4))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR) + 1.0
    gain = (float(bg.mean()) / bg) ** strength
    lab[..., 0] = np.clip(L * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_contrast(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def white_balance(img: np.ndarray) -> np.ndarray:
    """White-patch balance on the brightest (enamel) pixels. Gray-world is NOT used: intraoral photos are
    dominated by red soft tissue, so gray-world would tint enamel cyan and distort lesion colour."""
    f = img.astype(np.float32)
    lum = f.mean(2)
    sel = lum >= np.percentile(lum, 97)
    ref = f[sel].mean(0) + 1e-6                       # average colour of the brightest 3 %
    gain = np.clip(ref.mean() / ref, 0.85, 1.18)      # gentle: only neutralise the enamel cast
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray, work_side: int = QUALITY_DEFAULTS["work_side"],
               shadows: bool = True, balance: bool = True) -> np.ndarray:
    im = _resize_long(img, work_side)
    im = reduce_reflections(im)
    if shadows:
        im = correct_shadows(im)
    im = enhance_contrast(im)
    if balance:
        im = white_balance(im)
    return im
