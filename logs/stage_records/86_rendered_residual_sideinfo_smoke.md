# Stage86 Rendered Residual Side-Info Smoke

Date: 2026-06-28

## Goal

Validate whether top-k residual side-info can improve rendered RGB PSNR on a small task slice.

## Scope

- Tasks: `12` Stage79 eval q12 tasks.
- Gaps: `4`, `8`, `16`.
- Base methods: linear and Stage65 adapter.
- Keep fractions: `0`, `0.1`, `0.25`.
- Side bits: `8` for rate estimate.
- Residual values are unquantized teacher residuals, so this is an optimistic smoke.

## Implementation

Added:

```text
scripts/run_stage86_rendered_residual_sideinfo_smoke.py
```

The script constructs a base anchor, applies top-k residuals computed from dense target anchors, renders the resulting anchor, and reports PSNR delta plus estimated side-info rate.

## Run

GPU check was performed before execution. GPU1 was idle, so the smoke used:

```text
CUDA_VISIBLE_DEVICES=1
```

The first run emitted a render/target shape broadcast warning. The script was fixed to align target dimensions to the rendered tensor and rerun. Final metrics are from the warning-free run.

## Outputs

```text
experiments/stage86_rendered_residual_sideinfo_smoke/stage86_rendered_residual_sideinfo_smoke_summary.json
experiments/stage86_rendered_residual_sideinfo_smoke/stage86_rendered_residual_sideinfo_smoke_report.md
experiments/stage86_rendered_residual_sideinfo_smoke/stage86_rendered_residual_sideinfo_rows.csv
experiments/stage86_rendered_residual_sideinfo_smoke/stage86_rendered_residual_sideinfo_summary.csv
```

## Result

Top10% q8 residual side-info:

| base | gap | side MiB/intermediate-frame | delta PSNR |
|---|---:|---:|---:|
| linear | 4 | 0.05272865295410156 | 3.34178357354566 |
| linear | 8 | 0.05272865295410156 | 2.829455800539616 |
| linear | 16 | 0.05272865295410156 | 4.482252047941727 |
| stage65_adapter | 4 | 0.05272865295410156 | 2.8758664220815766 |
| stage65_adapter | 8 | 0.05272865295410156 | 2.4121275447105757 |
| stage65_adapter | 16 | 0.05272865295410156 | 3.414016361458769 |

Top25% q8 residual side-info:

| base | gap | side MiB/intermediate-frame | delta PSNR |
|---|---:|---:|---:|
| linear | 4 | 0.1318359375 | 6.079208939121096 |
| linear | 8 | 0.1318359375 | 5.362593643634623 |
| linear | 16 | 0.1318359375 | 6.849065423269733 |
| stage65_adapter | 4 | 0.1318359375 | 4.826898315812073 |
| stage65_adapter | 8 | 0.1318359375 | 4.331580420489802 |
| stage65_adapter | 16 | 0.1318359375 | 5.420300667021456 |

## Conclusion

- Residual side-info has clear rendered PSNR potential on this small smoke slice.
- The current result is optimistic because residual values are not quantized and entropy packing is not implemented.
- Any future residual side-info must be included in side-info rate and total rate.
- Next step: quantized residual side-info codec and small RD table.
