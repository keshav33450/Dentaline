#!/usr/bin/env python3
"""Model 2 - occlusal caries. Unified entry (same CLI as Models 1 & 3):

    python scripts/analyze.py IMG [IMG ...] --out DIR

Delegates to occ_detect.py (the caries detector + ICDAS severity grader),
which writes <name>_caries.jpg + <name>_caries.json for each image.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# hand off argv unchanged to the real occlusal analyzer
runpy.run_path(str(Path(__file__).resolve().parent / "occ_detect.py"), run_name="__main__")
