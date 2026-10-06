"""
One place for every folder the project uses.

Set the project location ONCE, any of these ways (first match wins):
  1. environment variable  DENTASSIST_HOME   e.g.  set DENTASSIST_HOME=D:\\Projects\\dentassist
  2. a file  project_location.txt  next to this repo containing the folder path
  3. default: the repo folder itself

Everything else is derived from it:
  <home>/data/raw       downloaded DENTEX zips
  <home>/data/yolo      converted YOLO datasets
  <home>/runs           training runs (curves, last.pt)
  <home>/weights        best models (.pt / .onnx / openvino)
  <home>/results        test metrics, REPORT.md, error gallery
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def project_home() -> Path:
    env = os.getenv("DENTASSIST_HOME", "").strip().strip('"')
    if env:
        return Path(env).expanduser().resolve()
    f = REPO / "project_location.txt"
    if f.exists():
        txt = f.read_text(encoding="utf-8").strip().strip('"')
        if txt:
            return Path(txt).expanduser().resolve()
    return REPO


HOME = project_home()
RAW = HOME / "data" / "raw"
DATA = HOME / "data" / "yolo"
RUNS = HOME / "runs"
WEIGHTS = HOME / "models"
RESULTS = HOME / "outputs"


def ensure() -> Path:
    for p in (RAW, DATA, RUNS, WEIGHTS, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
    return HOME


if __name__ == "__main__":
    ensure()
    for k in ("HOME", "RAW", "DATA", "RUNS", "WEIGHTS", "RESULTS"):
        print(f"{k:8s} {globals()[k]}")
