# Stage169 Combined Adaptive Rendered Validation Protocol

Date: 2026-06-30

## Goal

Design a broader but still controlled rendered validation protocol that combines Stage167 false-negative stress targets and Stage168 positive promoted targets.

## Plan

- Use Stage166 sampled-row consequences as the source of target labels and schedule status.
- Reuse Stage167 and Stage168 target lists to avoid repeating already-rendered smoke evidence.
- Select target categories:
  - adaptive hard false negatives that remain middle-recovery targets;
  - adaptive-promoted hard/high-payload targets from positive sequences;
  - high-payload residual controls that remain middle-recovery targets.
- Mark per-target schedule status for uniform gap8, Stage165 adaptive, and uniform gap4.
- Emit a Stage170 run plan that distinguishes existing reusable rows from rows requiring new rendering.
- Keep this stage lightweight: no rendering, no heavy media.

## Success Criteria

- A protocol package lists exact targets, categories, priorities, and reuse/render action.
- The report states target counts by category and how many target/schedule rows need new rendering.
- The decoder/rate contract remains explicit: adaptive keyframes are schedule metadata/keyframes, not decoder-side RGB/motion features.

## Command

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage169_combined_adaptive_rendered_validation_protocol.py && CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage169_combined_adaptive_rendered_validation_protocol.py
```

## Outputs

- `experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_targets.csv`
- `experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_schedule_rows.csv`
- `experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_summary.csv`
- `experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_adaptive_rendered_validation_protocol_package.json`
- `experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_adaptive_rendered_validation_protocol_report.md`

## Result

- Target count: `26`.
- Schedule row count: `78`.
- Existing reusable schedule rows from Stage167/168: `42`.
- New Stage170 render rows: `26`.
- Keyframe marker rows not requiring render: `10`.

Category summary:

- `false_negative_residual`: `8` targets, all `24` schedule rows reusable from Stage167.
- `positive_promoted`: `14` targets, `18` reusable schedule rows, `14` new render rows, `10` keyframe marker rows.
- `high_payload_residual_control`: `4` targets, `12` new render rows.

## Decision

- Stage170 should execute this protocol, reusing Stage167/168 rows and rendering only rows with `requires_stage170_render=1`.
- Adaptive promoted rows remain `target_keyframe_no_middle_render`; do not claim middle-render metrics for them.
