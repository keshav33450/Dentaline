MODEL 3 WEIGHTS
===============
best.pt   - YOLO11s tooth-type detector (trained on Kaggle T4). Used by the default backend.
best.onnx - same model in ONNX (for cross-platform / faster CPU inference if you want it).

Test accuracy (107 held-out images):
  overall  mAP50 99.1   mAP50-95 84.5
  incisor 99.5 | canine 99.5 | premolar 99.5 | molar 98.1   (full numbers in results\metrics.json)

Three inference backends (code\scripts\analyze.py):
  weights (default) : local best.pt via ultralytics - fully offline, no API key, no internet.
  local             : cached Roboflow RF-DETR model (downloads once).
  hosted            : Roboflow serverless API (one call per image).

Classes (index order): 0 canine, 1 incisor, 2 molar, 3 premolar.
