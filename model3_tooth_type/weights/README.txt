MODEL 3 WEIGHTS
===============
best.pt   - YOLO11s tooth-type detector (7 FDI classes, trained on Kaggle T4). Default backend.
best.onnx - same model in ONNX (cross-platform / faster CPU inference if you want it).

Test accuracy (42 held-out images, 758 teeth):
  overall  mAP50 65.3   mAP50-95 56.7
  Central Incisor 86.7 | Canine 81.1 | Lateral Incisor 71.9 | 1st Premolar 68.9
  2nd Premolar 60.6 | 1st Molar 59.6 | 2nd Molar 28.4      (full numbers in results\metrics.json)

Three inference backends (code\scripts\analyze.py):
  weights (default) : local best.pt via ultralytics - fully offline, no API key, no internet.
  local             : cached Roboflow model (downloads once).
  hosted            : Roboflow serverless API (one call per image).

Classes (index order): 0 1st Molar, 1 1st Premolar, 2 2nd Molar, 3 2nd Premolar,
                       4 Canine, 5 Central Incisor, 6 Lateral Incisor.

Note: an earlier 4-class model (incisor/canine/premolar/molar, 1,310 imgs) scored 0.991 mAP50;
this 7-class model trades accuracy for finer FDI-type detail and is data-limited (~420 imgs).
