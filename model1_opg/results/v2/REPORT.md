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
| caries | 0.5805 | 0.4294 | 0.4936 | 0.4554 |
| deep_caries | 0.5854 | 0.4068 | 0.48 | 0.4 |
| periapical_lesion | 0.6667 | 0.381 | 0.4848 | 0.4848 |
| impacted | 0.7905 | 0.9432 | 0.8601 | 0.7979 |
| overall | 0.6353 | 0.5104 | 0.566 | 0.5199 |

FDI accuracy on detected findings: **0.9185**
Latency per image: mean 76.1 ms, p95 87.5 ms

Plots: confusion matrices, PR curves in results/teeth_test and results/findings_test.
Worst cases: results/error_gallery/
