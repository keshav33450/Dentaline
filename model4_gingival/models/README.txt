MODEL 4 WEIGHTS  (gingival inflammation, single class)
======================================================
best.pt   - YOLO11s gingivitis detector, trained on merged intraoral/mobile
            gum-disease photos (frontal + upper + lower). Default offline backend.
best.onnx - same model in ONNX (cross-platform / faster CPU inference).

Validation accuracy (46 images, best epoch 99 of 120):
  mAP50 80.4   mAP50-95 36.9   Precision 78.4   Recall 72.9

Class (index order): 0 gingivitis.

Data: sampling/gingivitis-6uyts + Pranta/gum-disease-mmzcr
      (Roboflow Universe, CC BY 4.0). Pranta's Periodontitis class was dropped;
      only gingivitis boxes were kept and merged into one class.
      571 images / 1,508 boxes total (train 480 / valid 46 / test 45).

Note: the source datasets use loose, region-level boxes (gum + adjacent tooth),
which caps mAP50-95; the model's intended output is screening-level detection
(is gingivitis present, and roughly where), not pixel-tight segmentation.
