# Stage61 DAVIS Anchor Export Preflight

## Summary

- DAVIS root: `/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS`.
- Ready sequences: `90` / `90`.
- Frames: `6208`.
- Gaps: `[1, 2, 4, 8, 16]`.
- Free space at heavy root mount: `1059217.49` MiB.
- Required reserve: `2048.00` MiB.
- Estimated pair-pt output for all gaps: `21950.30` MiB.
- Full all-gap export safe: `true`.

## Totals

| scope | gap | sequences | frames | pairs | pair-pt MiB | dedup anchor MiB |
|---|---:|---:|---:|---:|---:|---:|
| all | 1 | 90 | 6208 | 6118 | 11184.47 | 5674.50 |
| all | 2 | 90 | 6208 | 3089 | 5647.08 | 2905.80 |
| all | 4 | 90 | 6208 | 1568 | 2866.50 | 1515.52 |
| all | 8 | 90 | 6208 | 807 | 1475.30 | 819.91 |
| all | 16 | 90 | 6208 | 425 | 776.95 | 470.74 |
| train | 1 | 60 | 4209 | 4149 | 7584.89 | 3847.29 |
| train | 2 | 60 | 4209 | 2094 | 3828.09 | 1968.89 |
| train | 4 | 60 | 4209 | 1062 | 1941.47 | 1025.58 |
| train | 8 | 60 | 4209 | 545 | 996.33 | 553.01 |
| train | 16 | 60 | 4209 | 286 | 522.84 | 316.27 |
| val | 1 | 30 | 1999 | 1969 | 3599.58 | 1827.21 |
| val | 2 | 30 | 1999 | 995 | 1818.98 | 936.91 |
| val | 4 | 30 | 1999 | 506 | 925.03 | 489.94 |
| val | 8 | 30 | 1999 | 262 | 478.97 | 266.91 |
| val | 16 | 30 | 1999 | 139 | 254.11 | 154.48 |
| all_gaps_total | all | 90 | 6208 | 12007 | 21950.30 | 11386.48 |

## Decision

Full DAVIS all-gap anchor export is safe to launch with the requested reserve.
