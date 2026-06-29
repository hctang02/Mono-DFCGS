# Stage132 Deployable Predictor Policy Package

Date: 2026-06-29

## Goal

Package the current best deployable no-teacher predictor policy.

## Plan

- Add a Stage132 deployable policy package script.
- Select `adapter_delta_selected_predictor/q4_top20` based on Stage131.
- Include q4/top10 as optional low-rate candidate.
- Mark teacher residual side-info as reference-only and MLP predictor as rejected for final deployment.
- Define decoder inputs, forbidden inputs, selected-index rule, residual value prediction rule, and rate accounting.
- Check `nvidia-smi` before running Python, even though this package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage132_deployable_predictor_policy_package.py
```

The script packages the current best no-teacher policy based on Stage131 ablations.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage132_deployable_predictor_policy_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage132_deployable_predictor_policy_package.py
```

## Outputs

```text
experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy.json
experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy_settings.csv
experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy_package.json
experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy_report.md
```

## Results

- Policy: `deployable_adapter_delta_selected_residual_codec_v1`.
- Status: `current_best_no_teacher_deployable`.
- Primary setting: `q4_top20`.
- Optional low-rate setting: `q4_top10`.
- Stage65 adapter checkpoint exists: `1`.
- Residual payload bytes per frame: `0`.
- Selected-index payload bytes per frame: `0`.

| role | setting | keep | rate | PSNR | delta base | delta full | residual bytes | index bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 0.11729838135687401 | 19.010259350474836 | 0.059456354043700026 | -0.17738394265066096 | 0 | 0 |
| low-rate | q4_top10 | 0.1 | 0.11729838135687401 | 18.994813480380337 | 0.04401048394920189 | -0.19282981274515912 | 0 | 0 |

## Conclusion

- Stage132 packages the current best no-teacher deployable policy.
- Teacher residual side-info remains reference-only.
- Dedicated MLP predictor is rejected for final deployment until render-aware training fixes Stage129 regression.
