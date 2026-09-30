# Model 1 - calibration & ground-truth validation

Validation X-rays: 50 (thresholds + preprocessing chosen here). Test X-rays: 106 (reported once).

## Preprocessing A/B (validation, macro-F1)

- none: 0.6045
- clahe (denoise + contrast): 0.5487
- **chosen: none**

## Thresholds (F1-optimal on validation)

| Finding | 'Likely' threshold |
|---|---|
| caries | 0.2 |
| deep_caries | 0.35 |
| periapical_lesion | 0.5 |
| impacted | 0.5 |
| hidden below | 0.2 (shown as 'Possible - needs review' between) |

## Test set - findings

| Finding | P (likely) | R (likely) | F1 (likely) | FP | FN | R (likely+possible) | F1 right tooth |
|---|---|---|---|---|---|---|---|
| caries | 0.533 | 0.49 | 0.511 | 155 | 184 | 0.49 | 0.442 |
| deep_caries | 0.575 | 0.39 | 0.465 | 17 | 36 | 0.39 | 0.323 |
| periapical_lesion | 0.714 | 0.238 | 0.357 | 2 | 16 | 0.381 | 0.286 |
| impacted | 0.82 | 0.932 | 0.872 | 18 | 6 | 0.955 | 0.521 |

## Test set - FDI numbering

| | without repair | with repair |
|---|---|---|
| teeth detected | 0.9335 | 0.9335 |
| correct number (of detected) | 0.9539 | 0.9585 |
| duplicate numbers | 9 | 0 |
