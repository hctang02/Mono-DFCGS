# Stage125 Broader Feed-Forward Residual Value Predictor Validation

Date: 2026-06-29

## Goal

Broaden the Stage124 no-teacher feed-forward residual value predictor smoke from 12 eval tasks to a 60-task validation slice.

## Plan

- Add a Stage125 validation script using the Stage124 predictor logic.
- Evaluate `adapter_delta_selected_v1` on Stage123 primary `q4_top20` and low-rate `q4_top10` settings.
- Use deterministic endpoint-diff selected indices reproduced from left/right anchors.
- Transmit no residual payload and no selected-index payload.
- Do not load target dense anchors; use target RGB only for offline rendered metrics.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage125_broader_feedforward_residual_value_predictor_validation.py
```

The script reuses the Stage124 no-teacher predictor utility and runs the same `adapter_delta_selected_v1` predictor on a 60-task eval slice.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage125_broader_feedforward_residual_value_predictor_validation.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage125_broader_feedforward_residual_value_predictor_validation.py
```

## Outputs

```text
experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_rows.csv
experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_summary.csv
experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_summary.json
experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_report.md
```

## Results

| setting | role | keep | tasks | rate | base PSNR | full adapter PSNR | selected PSNR | delta base | delta full | positives | near full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_top10 | low_rate | 0.1 | 60 | 0.11729838135687401 | 18.950802996431143 | 19.1876432931255 | 18.994813480380337 | 0.04401048394920189 | -0.19282981274515912 | 51/60 | 13/60 |
| q4_top20 | primary | 0.2 | 60 | 0.11729838135687401 | 18.950802996431143 | 19.1876432931255 | 19.010259350474836 | 0.059456354043700026 | -0.17738394265066096 | 48/60 | 13/60 |

## Conclusion

- The no-teacher feed-forward predictor gain over linear base survives the broader 60-task validation.
- q4/top20 is stronger than q4/top10 on the broader slice.
- The selected predictor remains below full Stage65 adapter quality, so Stage126/127 should prepare and train a dedicated selected residual value predictor.
