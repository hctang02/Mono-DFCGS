# Stage104 Render-Energy Selector Mismatch Diagnostic

## Configuration

- input rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_rows.csv`
- compared candidates: `shared_energy_regression, shared_topk_bce`
- reference candidate: `endpoint_diff_baseline`
- no rendering or heavy tensor output

## Summary

| candidate | base | gap | tasks | energy delta | psnr delta | energy up | psnr up | both up | energy up psnr down | corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_energy_regression | linear | 4 | 23 | 0.03452943816133167 | 0.026827877460275387 | 22 | 15 | 15 | 7 | 0.24394404220006974 |
| shared_energy_regression | linear | 8 | 19 | 0.045587826716272456 | 0.03086322039306108 | 19 | 10 | 10 | 9 | 0.3631481188513308 |
| shared_energy_regression | linear | 16 | 18 | 0.043746052516831294 | 0.13353966957143293 | 17 | 14 | 14 | 3 | 0.6456802836474845 |
| shared_energy_regression | stage65_adapter | 4 | 23 | 0.02671546327031177 | -0.15380783202852794 | 22 | 7 | 7 | 15 | 0.6192052211416481 |
| shared_energy_regression | stage65_adapter | 8 | 19 | 0.03388543348563345 | -0.19199111803343827 | 18 | 7 | 7 | 11 | 0.3184123356771083 |
| shared_energy_regression | stage65_adapter | 16 | 18 | 0.03499703254136774 | -0.016197329840340835 | 18 | 8 | 8 | 10 | 0.483226032535633 |
| shared_topk_bce | linear | 4 | 23 | 0.0336741439026335 | -0.08991556072924042 | 20 | 10 | 10 | 10 | 0.38927610301670823 |
| shared_topk_bce | linear | 8 | 19 | 0.04330471942299291 | -0.10740330389783524 | 17 | 7 | 7 | 10 | 0.2988772116489903 |
| shared_topk_bce | linear | 16 | 18 | 0.04484673134154744 | 0.03729370398841593 | 17 | 11 | 10 | 7 | 0.24443686390755395 |
| shared_topk_bce | stage65_adapter | 4 | 23 | 0.02759676394255265 | -0.18816386153627007 | 22 | 6 | 6 | 16 | 0.5194473422733707 |
| shared_topk_bce | stage65_adapter | 8 | 19 | 0.03471723159677104 | -0.19973996309924674 | 18 | 5 | 5 | 13 | 0.21836590422597538 |
| shared_topk_bce | stage65_adapter | 16 | 18 | 0.033893570510877505 | -0.032410577247806396 | 18 | 8 | 8 | 10 | 0.34862299170332045 |

## Notes

- Positive `energy delta` means the learned selector captured more residual energy than endpoint selection.
- Positive `psnr delta` means rendered PSNR improved over endpoint selection.
- `energy up psnr down` is the key mismatch count for render-aware selector diagnosis.
