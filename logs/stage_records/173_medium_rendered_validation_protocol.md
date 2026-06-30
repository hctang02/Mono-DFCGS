# Stage173 Medium Rendered Validation Protocol

Date: 2026-06-30

## Goal

Build a medium rendered validation protocol for the Stage165 adaptive keyframe schedule, reusing Stage167/168/170 rows and rendering only missing schedule rows in Stage174.

## Plan

- Keep all Stage170 combined-validation targets as the protocol core.
- Add new Stage166 sampled targets across positive promoted extensions, high-payload residual controls, selector false-positive/keyframe controls, and normal/easy residual controls.
- Build `uniform_gap8`, `stage165_adaptive`, and `uniform_gap4` schedule rows for each target.
- Mark rows already present in Stage167/168/170 as reusable.
- Mark target-keyframe rows as `target_keyframe_no_middle_render` and never claim middle-render metrics for them.
- Export lightweight target/schedule/summary CSVs and package JSON/report only.

## Success Criteria

- Protocol includes exact targets and schedule rows for Stage174.
- Expected new Stage174 render count is bounded and explicit.
- Protocol includes both positive and negative/false-positive controls.

## Execution

- Checked `nvidia-smi` before running; GPU 2/3/5/6/7 were idle, though this stage is CPU-only.
- Compiled and ran `scripts/run_stage173_medium_rendered_validation_protocol.py` with `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.

## Result

- Package: `experiments/stage173_medium_rendered_validation_protocol/stage173_medium_rendered_validation_protocol_package.json`.
- Report: `experiments/stage173_medium_rendered_validation_protocol/stage173_medium_rendered_validation_protocol_report.md`.
- Targets CSV: `experiments/stage173_medium_rendered_validation_protocol/stage173_medium_validation_targets.csv`.
- Schedule rows CSV: `experiments/stage173_medium_rendered_validation_protocol/stage173_medium_validation_schedule_rows.csv`.
- Summary CSV: `experiments/stage173_medium_rendered_validation_protocol/stage173_medium_validation_summary.csv`.
- Target count: `50`.
- Schedule row count: `150`.
- Reusable existing schedule rows: `84`.
- Stage174 new render rows: `54`.
- Keyframe marker rows not already covered by existing rows: `12`.
- Stage170 core targets retained: `26`.

## Category Summary

| category | targets | schedule rows | existing | new renders | keyframe markers | hard | high payload |
|---|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | 8 | 24 | 24 | 0 | 0 | 8 | 0 |
| high_payload_residual_control | 4 | 12 | 12 | 0 | 0 | 0 | 4 |
| high_payload_residual_control_extension | 8 | 24 | 0 | 23 | 1 | 0 | 8 |
| normal_residual_control | 4 | 12 | 0 | 12 | 0 | 0 | 0 |
| positive_promoted | 14 | 42 | 42 | 0 | 0 | 12 | 13 |
| positive_promoted_extension | 8 | 24 | 6 | 11 | 7 | 8 | 3 |
| selector_false_positive_keyframe_control | 4 | 12 | 0 | 8 | 4 | 0 | 0 |

## Next Step

- Proceed to Stage174 execution with the fixed Stage158 policy, reusing Stage167/168/170 rows and rendering only the `54` missing rows.
