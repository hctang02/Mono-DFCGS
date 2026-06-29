# Stage135 Render-Aware Predictor Protocol

Date: 2026-06-29

## Goal

Define the next render-aware no-teacher residual predictor protocol after Stage134 showed MSE-only MLP is not render-safe.

## Plan

- Package a render-aware calibration protocol for the adapter-delta predictor.
- Use only no-teacher decoder-side residuals: `adapter_attrs - linear_attrs` at deterministic endpoint-diff indices.
- Tune a global residual scale from rendered validation, not per-frame target inputs.
- Keep target RGB only for offline protocol selection and validation.
- Do not use teacher residual side-info or target dense anchors.
- Check `nvidia-smi` before running Python, even though this package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage135_render_aware_predictor_protocol.py
```

The script packages `render_aware_adapter_delta_scale_calibration_v1`.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage135_render_aware_predictor_protocol.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage135_render_aware_predictor_protocol.py
```

## Outputs

```text
experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol.csv
experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol.json
experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol_package.json
experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol_report.md
```

## Results

- Protocol: `render_aware_adapter_delta_scale_calibration_v1`.
- Predictor family: adapter-delta selected residual predictor.
- Residual rule: `scaled_residual = adapter_delta_scale * (adapter_attrs - linear_attrs)` at deterministic endpoint-diff indices.
- Scale candidates: `0.0, 0.25, 0.5, 0.75, 1.0, 1.25`.
- Settings: q4/top20 primary and q4/top10 low-rate.
- Target dense anchors and teacher residuals are not used.

## Conclusion

Stage136 should run the render-aware scale sweep smoke and Stage137 should validate the selected scale on a broader slice.
