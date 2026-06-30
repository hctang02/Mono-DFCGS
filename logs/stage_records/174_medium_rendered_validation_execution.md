# Stage174 Medium Rendered Validation Execution

Date: 2026-06-30

## Goal

Execute the Stage173 medium validation protocol by reusing Stage167/168/170 rows, rendering only missing schedule rows, and recording keyframe markers without middle metrics.

## Plan

- Load Stage173 target and schedule-row protocol.
- Load existing Stage167, Stage168, and Stage170 rendered/keyframe rows.
- Reuse existing rows whenever present.
- Render only rows with `requires_stage174_render=1` using fixed Stage158 `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Mark target keyframes as `target_keyframe_no_middle_render`.
- Summarize by category, schedule, and row source.
- Export lightweight CSV/JSON/report in repo and a heavy contact sheet outside repo only.

## Success Criteria

- Stage174 covers all `150` Stage173 protocol rows.
- Stage174 performs exactly `54` new renders unless the protocol changes.
- No heavy contact sheet/checkpoint/anchor files are committed.

## Execution

- Checked `nvidia-smi` before running; GPU 3/5/6/7 were idle and GPU 2 was newly occupied, so Stage174 ran on GPU 3.
- Compiled and ran `scripts/run_stage174_medium_rendered_validation_execution.py` with `CUDA_VISIBLE_DEVICES=3` and `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.
- LPIPS and MS-SSIM were available; warnings were limited to expected xFormers/torchvision/LPIPS notices.

## Result

- Package: `experiments/stage174_medium_rendered_validation_execution/stage174_medium_rendered_validation_execution_package.json`.
- Report: `experiments/stage174_medium_rendered_validation_execution/stage174_medium_rendered_validation_execution_report.md`.
- Rows CSV: `experiments/stage174_medium_rendered_validation_execution/stage174_medium_validation_rows.csv`.
- Summary CSV: `experiments/stage174_medium_rendered_validation_execution/stage174_medium_validation_summary.csv`.
- Source summary CSV: `experiments/stage174_medium_rendered_validation_execution/stage174_medium_validation_source_summary.csv`.
- Decision: `medium_validation_ready_for_decision`.
- Protocol rows covered: `150 / 150`.
- New renders completed: `54 / 54`.
- Reused rows: `84`.
- Keyframe marker rows total: `32`.
- Heavy contact sheet outside repo: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage174_medium_rendered_validation_execution/stage174_medium_rendered_validation_contact_sheet.jpg`.

## Source Coverage

| source | rows | rendered | keyframe markers |
|---|---:|---:|---:|
| stage168 | 6 | 4 | 2 |
| stage170 | 78 | 60 | 18 |
| stage174_keyframe_marker | 12 | 0 | 12 |
| stage174_rendered | 54 | 54 | 0 |

## Key Metrics

| category | schedule | rows | rendered | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | stage165_adaptive | 8 | 8 | 0 | 26.192677 | 0.833774 | 0.981409 | 0.208831 | 155993.625000 |
| false_negative_residual | uniform_gap8 | 8 | 8 | 0 | 26.203641 | 0.834367 | 0.981569 | 0.207921 | 156389.625000 |
| high_payload_residual_control_extension | stage165_adaptive | 8 | 8 | 0 | 29.539710 | 0.862902 | 0.983418 | 0.173480 | 225694.750000 |
| high_payload_residual_control_extension | uniform_gap8 | 8 | 8 | 0 | 29.539710 | 0.862902 | 0.983418 | 0.173480 | 225694.750000 |
| normal_residual_control | stage165_adaptive | 4 | 4 | 0 | 33.406462 | 0.905324 | 0.981935 | 0.075904 | 193691.000000 |
| normal_residual_control | uniform_gap8 | 4 | 4 | 0 | 33.406462 | 0.905324 | 0.981935 | 0.075904 | 193691.000000 |
| positive_promoted | stage165_adaptive | 14 | 0 | 14 | NA | NA | NA | NA | NA |
| positive_promoted_extension | stage165_adaptive | 8 | 0 | 8 | NA | NA | NA | NA | NA |
| selector_false_positive_keyframe_control | stage165_adaptive | 4 | 0 | 4 | NA | NA | NA | NA | NA |
| selector_false_positive_keyframe_control | uniform_gap8 | 4 | 4 | 0 | 30.071755 | 0.876903 | 0.984161 | 0.183046 | 157481.250000 |

## Interpretation

- Stage174 successfully expands rendered evidence while keeping Stage158 fixed.
- Residual rows that adaptive does not promote often match uniform gap8 exactly because their adaptive segment remains the same.
- Promoted adaptive rows are keyframes/no-middle-render; they should be evaluated through keyframe rate and schedule decisions, not middle metrics.
- Selector false-positive keyframe controls show the cost/risk side of adaptive promotion and should be considered in Stage175.

## Next Step

- Proceed to Stage175 decision branch using Stage172 rate accounting plus Stage174 medium rendered evidence.
