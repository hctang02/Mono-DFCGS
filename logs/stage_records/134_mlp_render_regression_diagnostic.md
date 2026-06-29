# Stage134 MLP Render Regression Diagnostic

Date: 2026-06-29

## Goal

Diagnose why the dedicated MLP residual predictor reduces residual MSE but regresses rendered PSNR.

## Plan

- Add a CPU diagnostic script comparing Stage125 adapter-delta rows and Stage129 dedicated MLP rows task-by-task.
- Summarize regressions by setting and reference gap.
- Identify worst MLP regressions relative to linear base and adapter-delta predictor.
- Do not use teacher residual side-info as an optimization target.
- Check `nvidia-smi` before running Python, even though this diagnostic is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage134_mlp_render_regression_diagnostic.py
```

The script aligns Stage125 adapter-delta rows and Stage129 dedicated MLP rows by task and setting.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this diagnostic is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage134_mlp_render_regression_diagnostic.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage134_mlp_render_regression_diagnostic.py
```

## Outputs

```text
experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_rows.csv
experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_summary.csv
experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_worst_rows.csv
experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_package.json
experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_report.md
```

## Results

| group | setting | gap | tasks | adapter delta | MLP delta base | MLP delta adapter | MLP base regressions | MLP adapter regressions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| setting | q4_top10 | all | 60 | 0.04401048394920189 | -0.08502524287394495 | -0.12903572682314685 | 28 | 47 |
| setting | q4_top20 | all | 60 | 0.059456354043700026 | -0.1855999457880447 | -0.24505629983174473 | 34 | 47 |

Worst MLP regression vs adapter-delta:

- `stage79_00027031`, sequence `india`, gap `8`, q4/top20: MLP minus adapter-delta `-1.2118810842726546 dB`.

## Conclusion

- MLP residual MSE reduction does not align with rendered PSNR.
- MLP underperforms adapter-delta on `47/60` tasks for both q4/top10 and q4/top20.
- Stage135 should use render-aware gating/training criteria rather than attribute MSE alone.
