# DentAssist OPG - test results

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Model 1 - teeth (32 FDI) | 0.9644 | 0.5494 | 0.9223 | 0.9312 |
| Model 5 - findings | 0.5768 | 0.3866 | 0.6624 | 0.5268 |

## Findings per class

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| caries | 0.6186 | 0.3684 | 0.5302 |
| deep_caries | 0.5789 | 0.4576 | 0.474 |
| periapical_lesion | 0.6898 | 0.3333 | 0.3811 |
| impacted | 0.7624 | 0.9478 | 0.9221 |

## End-to-end (IoU>=0.5, conf>=0.25)

| Class | P (box+dx) | R (box+dx) | F1 (box+dx) | F1 (+ correct tooth) |
|---|---|---|---|---|
| caries | 0.572 | 0.4294 | 0.4905 | 0.4494 |
| deep_caries | 0.5833 | 0.4746 | 0.5234 | 0.4299 |
| periapical_lesion | 0.6667 | 0.381 | 0.4848 | 0.4848 |
| impacted | 0.7685 | 0.9432 | 0.8469 | 0.7755 |
| overall | 0.6241 | 0.518 | 0.5661 | 0.5145 |

FDI accuracy on detected findings: **0.9088**
Latency per image: mean 70.2 ms, p95 78.0 ms

Plots: confusion matrices, PR curves in results/teeth_test and results/findings_test.
Worst cases: results/error_gallery/
