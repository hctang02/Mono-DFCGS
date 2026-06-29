# Stage138 Render-Aware Deployable Policy

Date: 2026-06-29

## Goal

Package the Stage137 render-aware calibrated adapter-delta predictor as the updated no-teacher deployable policy.

## Plan

- Use Stage137 broader validation as the policy source.
- Select q4/top20 scale `0.75` as the primary deployable setting.
- Select q4/top10 scale `0.75` as the optional low-rate setting.
- Preserve zero transmitted residual payload bytes and zero transmitted selected-index bytes.
- Update decoder contract to multiply the adapter-delta residual by the policy scale before applying selected residuals.
- Keep target dense anchors, teacher residual side-info, target RGB, oracle labels, selected indices, and residual values forbidden at decoder side.
- Check `nvidia-smi` before running Python, even though the package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage138_render_aware_deployable_policy_package.py
```

The script packages `deployable_render_aware_scaled_adapter_delta_selected_residual_codec_v1` from the Stage137 broader validation summary.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage138_render_aware_deployable_policy_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage138_render_aware_deployable_policy_package.py
```

## Outputs

```text
experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy.json
experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy_settings.csv
experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy_package.json
experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy_report.md
```

Output size: `24K`.

## Results

- Policy: `deployable_render_aware_scaled_adapter_delta_selected_residual_codec_v1`.
- Status: `current_best_no_teacher_deployable_render_aware`.
- Replaces: `deployable_adapter_delta_selected_residual_codec_v1`.
- Primary: q4/top20 scale `0.75`, PSNR `19.022109503207204`, delta vs base `+0.07130650677606927`, delta vs Stage132 `+0.011850152732368002`.
- Optional low-rate: q4/top10 scale `0.75`, PSNR `18.997890662360874`, delta vs base `+0.047087665929735116`, delta vs Stage132 `+0.0030771819805366363`.
- Residual payload bytes: `0`.
- Selected-index payload bytes: `0`.
- Adapter checkpoint exists: `1`.

## Conclusion

Stage138 is the current best no-teacher deployable predictor policy. Stage139 should package the full pipeline rate accounting around this policy.
