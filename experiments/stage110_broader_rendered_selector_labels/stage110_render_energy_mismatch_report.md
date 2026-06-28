# Stage110 Broader Render-Energy Selector Mismatch Diagnostic

## Configuration

- input rows: `experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_rows.csv`
- compared candidates: `shared_energy_regression, shared_topk_bce`
- reference candidate: `endpoint_diff_baseline`
- no rendering or heavy tensor output

## Summary

| candidate | base | gap | tasks | energy delta | psnr delta | energy up | psnr up | both up | energy up psnr down | corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_energy_regression | linear | 4 | 83 | 0.031175289392830378 | -0.023422587923175493 | 78 | 37 | 37 | 41 | 0.4052231008087529 |
| shared_energy_regression | linear | 8 | 79 | 0.03677648032390619 | 0.009307271828446134 | 75 | 43 | 42 | 33 | 0.44188566326293155 |
| shared_energy_regression | linear | 16 | 78 | 0.03649166226387024 | 0.026461930821385912 | 71 | 46 | 46 | 25 | 0.6072215610660064 |
| shared_energy_regression | stage65_adapter | 4 | 83 | 0.024832078311816757 | -0.22540383262011054 | 81 | 15 | 15 | 66 | 0.39426318864842186 |
| shared_energy_regression | stage65_adapter | 8 | 79 | 0.03036606472127045 | -0.19372914974113 | 77 | 23 | 23 | 54 | 0.45895494987272756 |
| shared_energy_regression | stage65_adapter | 16 | 78 | 0.032561146725828834 | -0.1363193230658923 | 76 | 27 | 27 | 49 | 0.523518086018581 |
| shared_topk_bce | linear | 4 | 83 | 0.028763428285538434 | -0.15837973075354014 | 68 | 26 | 25 | 43 | 0.5263232473872927 |
| shared_topk_bce | linear | 8 | 79 | 0.03395696503074863 | -0.11703232358972818 | 64 | 34 | 32 | 32 | 0.5640993291852238 |
| shared_topk_bce | linear | 16 | 78 | 0.03752737349042526 | -0.06810500216136936 | 69 | 39 | 38 | 31 | 0.5289893075745488 |
| shared_topk_bce | stage65_adapter | 4 | 83 | 0.025669775723692882 | -0.24715257971309498 | 80 | 14 | 13 | 67 | 0.34624120787505264 |
| shared_topk_bce | stage65_adapter | 8 | 79 | 0.03088846972471551 | -0.21419415416029447 | 74 | 18 | 18 | 56 | 0.4071804727253955 |
| shared_topk_bce | stage65_adapter | 16 | 78 | 0.03234666733978651 | -0.15850752115702738 | 76 | 29 | 29 | 47 | 0.47885855550593825 |

## Notes

- Positive `energy delta` means the learned selector captured more residual energy than endpoint selection.
- Positive `psnr delta` means rendered PSNR improved over endpoint selection.
- `energy up psnr down` is the key mismatch count for render-aware selector diagnosis.
