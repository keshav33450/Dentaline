MODEL 3 WEIGHTS  (view-robust, 4 tooth-type classes)
====================================================
best.pt   - YOLO11s tooth-type detector, trained on merged multi-view data
            (frontal + upper + lower intraoral). Default offline backend.
best.onnx - same model in ONNX (cross-platform / faster CPU inference).

Test accuracy (134 images, 2,243 teeth):
  overall  mAP50 94.6   mAP50-95 66.6
  incisor 97.3 | canine 95.1 | premolar 94.4 | molar 91.5

Per-view mAP50 (view-robust):
  frontal 94.9 | upper 94.7 | lower 94.5

Classes (index order): 0 incisor, 1 canine, 2 premolar, 3 molar.
Data: DentalMate6v Front/Upper/Lower Intraoral Tooth Numbering (CC BY 4.0),
      FDI numbers remapped to 4 tooth types. Credit: DentalMate6v.
