"""Label definitions for the occlusal caries study (single source of truth)."""
from __future__ import annotations

# ---- tooth type (FDI position 4-5 = premolar, 6-8 = molar; permanent dentition only)
TOOTH_TYPES = ["premolar", "molar"]
POSTERIOR_POSITIONS = {4, 5, 6, 7, 8}


def tooth_type_from_fdi(fdi: str | int) -> str | None:
    s = str(fdi).strip()
    if len(s) != 2 or not s.isdigit():
        return None
    q, n = int(s[0]), int(s[1])
    if q not in (1, 2, 3, 4) or n not in POSTERIOR_POSITIONS:   # 5-8 = primary teeth -> excluded
        return None
    return "premolar" if n in (4, 5) else "molar"


# ---- ICDAS (0-6) -> simplified severity (proforma: No Caries / Mild / Moderate / Advanced)
# Default grouping follows ICCMS (Pitts et al. 2014): sound 0 | initial 1-2 | moderate 3-4 | extensive 5-6.
#   Mild     = enamel lesion                     (ICDAS 1-2)
#   Moderate = early dentin involvement          (ICDAS 3-4)
#   Advanced = deep dentin involvement/cavitation (ICDAS 5-6)
# Change here if the study team decides otherwise (e.g. move code 3 to Mild).
SEVERITY = ["no_caries", "mild", "moderate", "advanced"]
SEVERITY_LABEL = {"no_caries": "No Caries", "mild": "Mild", "moderate": "Moderate", "advanced": "Advanced"}
ICDAS_TO_SEVERITY = {0: "no_caries", 1: "mild", 2: "mild", 3: "moderate", 4: "moderate", 5: "advanced", 6: "advanced"}

# screening summary text shown in the PDF report (decision support, clinician confirms)
SEVERITY_NOTE = {
    "no_caries": "Sound occlusal surface. Routine recall and preventive advice.",
    "mild": "Enamel-level lesion. Consider non-invasive / preventive management and monitoring.",
    "moderate": "Early dentin involvement suspected. Clinical and radiographic confirmation advised.",
    "advanced": "Deep dentin involvement / cavitation suspected. Prioritise clinical evaluation for treatment.",
}
SEVERITY_COLOR_BGR = {"no_caries": (80, 180, 60), "mild": (0, 210, 240), "moderate": (0, 140, 255), "advanced": (40, 40, 220)}


def severity_from_icdas(code) -> str | None:
    try:
        c = int(float(str(code).strip()))
    except ValueError:
        return None
    return ICDAS_TO_SEVERITY.get(c)
