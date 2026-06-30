# Stage170 Combined Adaptive Rendered Validation Execution

Date: 2026-06-30

## Goal

Execute the Stage169 combined validation protocol by reusing Stage167/168 smoke rows and rendering only missing target/schedule rows.

## Plan

- Load Stage169 target and schedule-row protocol.
- Load existing Stage167 and Stage168 rendered rows.
- For each protocol target/schedule row:
  - reuse existing rows when available;
  - mark adaptive/keyframe rows as `target_keyframe_no_middle_render` without rendering;
  - render missing middle-recovery rows with the fixed Stage158 policy.
- Keep Stage158 policy fixed: `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Summarize by category and schedule, separating reused rows, newly rendered rows, and keyframe marker rows.
- Export lightweight CSV/JSON/report in repo and heavy contact sheet outside repo only.

## Success Criteria

- Stage170 package exists with complete protocol coverage.
- Report clearly separates reused, rendered, and keyframe-marker rows.
- No heavy media/checkpoint/anchor files are committed.

## Execution

- Checked `nvidia-smi` before running; GPU 2 was idle.
- Compiled and ran `scripts/run_stage170_combined_adaptive_rendered_validation_execution.py` with `CUDA_VISIBLE_DEVICES=2` and `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.
- LPIPS and MS-SSIM were available; warnings were limited to expected xFormers/torchvision/LPIPS notices.

## Result

- Package: `experiments/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_adaptive_rendered_validation_execution_package.json`.
- Report: `experiments/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_adaptive_rendered_validation_execution_report.md`.
- Rows CSV: `experiments/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_validation_rows.csv`.
- Summary CSV: `experiments/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_validation_summary.csv`.
- Source summary CSV: `experiments/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_validation_source_summary.csv`.
- Protocol rows covered: `78 / 78`.
- New renders completed: `26`.
- Reused rows: `42`.
- Keyframe marker rows: `10`.
- Decision: `combined_validation_ready_for_review`.
- Heavy contact sheet outside repo: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_adaptive_rendered_validation_contact_sheet.jpg`.

## Source Coverage

| source | rows | rendered | keyframe markers |
|---|---:|---:|---:|
| stage167 | 24 | 23 | 1 |
| stage168 | 18 | 11 | 7 |
| stage170_keyframe_marker | 10 | 0 | 10 |
| stage170_rendered | 26 | 26 | 0 |

## Key Metrics

| category | schedule | rows | rendered | keyframes | new renders | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | stage165_adaptive | 8 | 8 | 0 | 0 | 26.192677 | 0.833774 | 0.981409 | 0.208831 | 155993.625000 |
| false_negative_residual | uniform_gap8 | 8 | 8 | 0 | 0 | 26.203641 | 0.834367 | 0.981569 | 0.207921 | 156389.625000 |
| high_payload_residual_control | stage165_adaptive | 4 | 4 | 0 | 4 | 31.354837 | 0.890757 | 0.987188 | 0.150451 | 239237.250000 |
| high_payload_residual_control | uniform_gap8 | 4 | 4 | 0 | 4 | 31.039273 | 0.880826 | 0.985841 | 0.167492 | 239014.000000 |
| positive_promoted | stage165_adaptive | 14 | 0 | 14 | 0 | NA | NA | NA | NA | NA |
| positive_promoted | uniform_gap8 | 14 | 14 | 0 | 8 | 28.467816 | 0.846066 | 0.980992 | 0.215784 | 238069.214286 |

## Interpretation

- Stage170 executed the Stage169 contract without changing the Stage158 policy.
- Positive-promoted adaptive rows are recorded as target keyframes/no-middle-render, so no rendered middle-frame metrics are claimed for those rows.
- False-negative residual rows remain essentially unchanged versus uniform gap8, matching Stage167.
- High-payload residual controls show Stage165 adaptive residual rows are not worse than uniform gap8 on this small control set.
- Next step is Stage171 review of the combined evidence before deciding whether to scale to broader/full-sequence adaptive RD validation.
