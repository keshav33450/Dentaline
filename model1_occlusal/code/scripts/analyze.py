#!/usr/bin/env python3
"""CLI alias matching Models 2 & 3:  python scripts/analyze.py IMG [IMG ...] --out DIR"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occ_detect import main  # noqa: E402

if __name__ == "__main__":
    main()
