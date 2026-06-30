# Stage179 Broader Sampled Adaptive Validation Protocol

Date: 2026-06-30

## Goal

Create a broader sampled adaptive validation protocol for the Stage176/177 candidate, larger than the Stage174 50-target medium set, without rendering in this stage.

## Plan

- Use Stage166 sampled row consequences as the candidate pool.
- Include all Stage173/174 medium targets as reusable core coverage.
- Add broader targets from false negatives, positive adaptive promotions, selector false positives, high-payload residual controls, weak subjective sequences, and normal residual controls.
- Compare `uniform_gap8`, `stage165_adaptive`, and `uniform_gap4` for every target.
- Reuse Stage174 rendered rows and Stage177 final-quality rows where possible.
- Mark missing middle-recovery rows as `requires_stage180_render=1`.
- Mark missing target-keyframe final-quality rows as `requires_stage180_keyframe_metric=1`.

## Success Criteria

- Protocol CSVs list targets and schedule rows with deterministic categories and reuse/new-work flags.
- Summary reports target count, schedule-row count, reusable rows, new render rows, and new keyframe metric rows.
- No rendering or heavy media are produced in Stage179.

## Execution

- Checked `nvidia-smi` before running Stage179.
- Compiled `scripts/run_stage179_broader_sampled_adaptive_validation_protocol.py` with `py_compile`.
- Ran the protocol generator with `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.
- No rendering was performed and no heavy media were produced.

## Result

- Stage179 status: `broader_sampled_adaptive_validation_protocol_packaged`.
- Targets: `90`.
- Schedule rows: `270`.
- Stage174 core targets retained: `50`.
- New targets beyond Stage174: `40`.
- Existing middle metric rows: `118`.
- Existing keyframe metric rows: `32`.
- Stage180 middle renders required: `88`.
- Stage180 q12 keyframe metrics required: `32`.
- Keyframe schedule rows: `64`.

## Target Categories

| category | targets | core | new renders | new keyframe metrics |
|---|---:|---:|---:|---:|
| broader_false_negative_residual | 1 | 0 | 3 | 0 |
| broader_normal_residual_control | 5 | 0 | 15 | 0 |
| broader_positive_promoted | 18 | 0 | 34 | 20 |
| broader_sequence_coverage_probe | 9 | 0 | 20 | 7 |
| broader_weak_sequence_probe | 7 | 0 | 16 | 5 |
| false_negative_residual | 8 | 8 | 0 | 0 |
| high_payload_residual_control | 4 | 4 | 0 | 0 |
| high_payload_residual_control_extension | 8 | 8 | 0 | 0 |
| normal_residual_control | 4 | 4 | 0 | 0 |
| positive_promoted | 14 | 14 | 0 | 0 |
| positive_promoted_extension | 8 | 8 | 0 | 0 |
| selector_false_positive_keyframe_control | 4 | 4 | 0 | 0 |

## Reuse And New Work

| source | rows |
|---|---:|
| Stage174 middle metrics from Stage168 | 4 |
| Stage174 middle metrics from Stage170 | 60 |
| Stage174 middle metrics from Stage174 rendered rows | 54 |
| Stage177 keyframe metrics from Stage168 | 2 |
| Stage177 keyframe metrics from Stage170 | 18 |
| Stage177 keyframe metrics from Stage174 keyframe markers | 12 |
| Stage180 middle render required | 88 |
| Stage180 q12 keyframe metric required | 32 |

## Decision

- Use Stage179 as the Stage180 execution protocol.
- Stage180 should render only rows with `requires_stage180_render=1` and q12 keyframe metrics only for rows with `requires_stage180_keyframe_metric=1`.
- Keep Stage158 recovery fixed and keep heavy contact sheets outside git.

## Outputs

- Package: `experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_sampled_adaptive_validation_protocol_package.json`
- Report: `experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_sampled_adaptive_validation_protocol_report.md`
- Targets CSV: `experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_targets.csv`
- Schedule rows CSV: `experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_schedule_rows.csv`
- Summary CSV: `experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_summary.csv`
- Source summary CSV: `experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_source_summary.csv`
