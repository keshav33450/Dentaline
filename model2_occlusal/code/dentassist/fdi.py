"""FDI tooth numbering + class definitions shared by data prep, training, inference."""
from __future__ import annotations

# ---- Model 1: teeth (32 FDI classes) -------------------------------------
# class index = (quadrant-1)*8 + (tooth-1)  ->  0:"11" ... 31:"48"
FDI_TEETH: list[str] = [f"{q}{n}" for q in range(1, 5) for n in range(1, 9)]
FDI_TO_IDX = {fdi: i for i, fdi in enumerate(FDI_TEETH)}

TOOTH_TYPE = {1: "central incisor", 2: "lateral incisor", 3: "canine",
              4: "first premolar", 5: "second premolar",
              6: "first molar", 7: "second molar", 8: "third molar"}
QUADRANT_NAME = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right"}


def fdi_from_parts(quadrant: int, tooth: int) -> str:
    if not (1 <= quadrant <= 4 and 1 <= tooth <= 8):
        raise ValueError(f"invalid FDI parts q={quadrant} n={tooth}")
    return f"{quadrant}{tooth}"


def describe_tooth(fdi: str) -> dict:
    q, n = int(fdi[0]), int(fdi[1])
    return {"fdi": fdi, "quadrant": q, "position": n,
            "type": TOOTH_TYPE[n], "side": QUADRANT_NAME[q],
            "group": "incisor" if n <= 2 else "canine" if n == 3 else "premolar" if n <= 5 else "molar"}


# ---- Model 5 (+ impacted): findings -----------------------------------------
FINDINGS: list[str] = ["caries", "deep_caries", "periapical_lesion", "impacted"]
FINDING_LABEL = {"caries": "Caries", "deep_caries": "Deep caries",
                 "periapical_lesion": "Periapical lesion", "impacted": "Impacted tooth"}
FINDING_COLOR_BGR = {"caries": (0, 200, 255), "deep_caries": (0, 90, 255),
                     "periapical_lesion": (60, 60, 230), "impacted": (230, 120, 40)}


def normalize_finding(name: str) -> str | None:
    """Map raw dataset category names (e.g. 'Deep Caries', 'Periapical Lesion') to canonical keys."""
    k = name.strip().lower().replace("-", " ").replace("_", " ")
    if "impact" in k:
        return "impacted"
    if "periapical" in k or "lesion" in k:
        return "periapical_lesion"
    if "deep" in k and "caries" in k:
        return "deep_caries"
    if "caries" in k:
        return "caries"
    return None


def wisdom_teeth() -> set[str]:
    return {"18", "28", "38", "48"}
