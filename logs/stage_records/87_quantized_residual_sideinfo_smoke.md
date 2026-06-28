# Stage87 Quantized Residual Side-Info Smoke

Date: 2026-06-28

## Goal

Test whether q6/q8 quantized residual side-info preserves the rendered PSNR gains observed in Stage86.

## Scope

- Tasks: same 12-task q12 eval slice as Stage86.
- Gaps: `4`, `8`, `16`.
- Base methods: linear and Stage65 adapter.
- Keep fractions: `0.1`, `0.25`.
- Side bits: `6`, `8`.
- Rate includes Gaussian indices, quantized residual attributes, and per-attribute min/max metadata.

## Implementation

Added:

```text
scripts/run_stage87_quantized_residual_sideinfo_smoke.py
```

Residual values are quantized per frame and per attribute over the kept Gaussian subset.

## Run

GPU check was performed before execution. GPU0 was busy and GPU1 was idle, so the smoke used:

```text
CUDA_VISIBLE_DEVICES=1
```

## Outputs

```text
experiments/stage87_quantized_residual_sideinfo_smoke/stage87_quantized_residual_sideinfo_smoke_summary.json
experiments/stage87_quantized_residual_sideinfo_smoke/stage87_quantized_residual_sideinfo_smoke_report.md
experiments/stage87_quantized_residual_sideinfo_smoke/stage87_quantized_residual_sideinfo_rows.csv
experiments/stage87_quantized_residual_sideinfo_smoke/stage87_quantized_residual_sideinfo_summary.csv
```

## Result

q6 top10% residual side-info:

| base | gap | side MiB/intermediate-frame | delta PSNR |
|---|---:|---:|---:|
| linear | 4 | 0.041353702545166016 | 3.3371957812025705 |
| linear | 8 | 0.041353702545166016 | 2.8243534745057453 |
| linear | 16 | 0.041353702545166016 | 4.476855554636258 |
| stage65_adapter | 4 | 0.041353702545166016 | 2.869845655272099 |
| stage65_adapter | 8 | 0.041353702545166016 | 2.411187534241218 |
| stage65_adapter | 16 | 0.041353702545166016 | 3.406900716022574 |

q8 top10% residual side-info:

| base | gap | side MiB/intermediate-frame | delta PSNR |
|---|---:|---:|---:|
| linear | 4 | 0.05277824401855469 | 3.3417250739461744 |
| linear | 8 | 0.05277824401855469 | 2.830301690996175 |
| linear | 16 | 0.05277824401855469 | 4.482808916885427 |
| stage65_adapter | 4 | 0.05277824401855469 | 2.8765203485467516 |
| stage65_adapter | 8 | 0.05277824401855469 | 2.4118763463117925 |
| stage65_adapter | 16 | 0.05277824401855469 | 3.412887380817623 |

## Conclusion

- q6 residual side-info retains nearly all of the q8 rendered PSNR gain in this smoke.
- Top10% residual side-info is a promising operating point around `0.0414-0.0528 MiB/intermediate-frame`.
- This is still a smoke test; full RD must combine main q12 anchor rate and side-info rate.
