# Stage128 Predictor Codec Integration Package

Date: 2026-06-29

## Goal

Package the Stage127 selected residual predictors into an integrated predictor-only codec manifest.

## Plan

- Add a Stage128 package script.
- Consume Stage123 codec policy, Stage126 stats, and Stage127 training metrics/checkpoint paths.
- Verify external checkpoint files exist and can be loaded into `SelectedResidualValueMLP`.
- Emit integrated policy JSON, settings CSV, package JSON, and report Markdown.
- Keep checkpoints outside git and record paths only.
- Mark residual payload bytes and selected-index bytes as zero for predictor-only deployment.
- Check `nvidia-smi` before running Python.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage128_predictor_codec_integration_package.py
```

The script consumes Stage123, Stage126, and Stage127 artifacts, verifies checkpoint existence and strict loading, and emits a predictor-only codec integration policy.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage128_predictor_codec_integration_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage128_predictor_codec_integration_package.py
```

## Outputs

```text
experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_policy.json
experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_settings.csv
experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_package.json
experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_report.md
```

## Results

| setting | role | keep | hidden | eval reduction | checkpoint exists | load ok | residual bytes | index bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| q4_top20 | primary | 0.2 | 128 | 0.08808295024199653 | 1 | 1 | 0 | 0 |
| q4_top10 | low_rate | 0.1 | 128 | 0.10293283119315721 | 1 | 1 | 0 | 0 |

Policy:

- `predictor_only_selected_residual_codec_v1`
- Status: `predictor_integrated_pending_rendered_validation`
- Checkpoints remain outside git under `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke`.

## Conclusion

- Stage128 packages the predictor-only codec integration.
- Both checkpoints exist and load successfully.
- Stage129 should run rendered validation using the integrated predictor-only codec.
