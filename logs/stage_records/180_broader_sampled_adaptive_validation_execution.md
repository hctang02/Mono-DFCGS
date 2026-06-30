# Stage180 Broader Sampled Adaptive Validation Execution

Date: 2026-06-30

## Goal

Execute the Stage179 broader sampled adaptive validation protocol and package a 90-target final-quality comparison of Stage165 adaptive against uniform gap8 and uniform gap4.

## Plan

- Load Stage179 targets and schedule rows.
- Reuse Stage174 middle-recovery metrics where available.
- Reuse Stage177 q12 keyframe final-quality metrics where available.
- Render Stage158 middle recovery only for rows with `requires_stage180_render=1`.
- Render q12 target keyframe metrics only for rows with `requires_stage180_keyframe_metric=1`.
- Produce schedule rows, source summary, final-quality rows, target deltas, category summaries, package JSON, and report.
- Keep heavy media outside git; Stage180 should commit only lightweight CSV/JSON/Markdown/script/log files.

## Success Criteria

- All `270` Stage179 schedule rows are covered.
- Stage180 completes the expected `88` new middle renders and `32` new q12 keyframe metrics.
- Final-quality comparison reports adaptive vs uniform gap8/gap4 PSNR/SSIM/MS-SSIM/LPIPS on `90` targets.
- No target dense anchors, target RGB, unencoded residuals, or oracle labels are used as decoder-side inputs.

## Execution

- Checked `nvidia-smi` before running Stage180; GPU 3 was selected as idle.
- Compiled `scripts/run_stage180_broader_sampled_adaptive_validation_execution.py` with `py_compile`.
- First run completed the rendering work but failed during final-quality aggregation because reused Stage177 CSV keyframe metrics were strings.
- Patched `existing_final_quality_row()` to convert reused metrics to floats.
- Recompiled and reran Stage180 successfully on GPU 3.
- No heavy media were generated; only lightweight CSV/JSON/Markdown outputs were written in-repo.

## Result

- Stage180 status: `broader_sampled_adaptive_validation_execution_packaged`.
- Decision: `broader_validation_ready_for_review`.
- Protocol rows covered: `270 / 270`.
- Final-quality rows: `270`.
- Targets: `90`.
- New middle renders: `88 / 88`.
- New q12 keyframe metrics: `32 / 32`.
- Reused Stage174 rows: `150`.
- Keyframe marker rows: `64`.

## Overall Final Quality

| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage165_adaptive | 90 | 56 | 34 | 29.770753 | 25.424549 | 0.142780 | 74673.433 |
| uniform_gap4 | 90 | 8 | 82 | 29.464217 | 25.431154 | 0.162457 | 196008.433 |
| uniform_gap8 | 90 | 0 | 90 | 29.206326 | 25.418355 | 0.176652 | 219878.578 |

Paired adaptive deltas:

- Adaptive minus uniform gap8 PSNR: `+0.5644261202320328` dB.
- Adaptive minus uniform gap4 PSNR: `+0.306535729521994` dB.
- Adaptive minus uniform gap8 LPIPS: `-0.0338725696835253`.
- Adaptive minus uniform gap4 LPIPS: `-0.019677375422583687`.

## Category Delta Highlights

| category | targets | adaptive keyframes | delta vs gap8 PSNR | delta vs gap4 PSNR |
|---|---:|---:|---:|---:|
| broader_positive_promoted | 18 | 18 | +0.829807 | +0.443999 |
| broader_sequence_coverage_probe | 9 | 7 | +0.627988 | +0.271456 |
| broader_weak_sequence_probe | 7 | 5 | +1.028786 | +0.563861 |
| false_negative_residual | 8 | 0 | -0.010964 | -0.326294 |
| high_payload_residual_control | 4 | 0 | +0.315565 | +0.276767 |
| positive_promoted | 14 | 14 | +1.248305 | +0.879899 |

## Interpretation

- Stage180 strengthens the Stage177 conclusion on a broader sampled set: adaptive remains better than both uniform gap8 and uniform gap4 in final target PSNR and LPIPS.
- The gains mainly come from adaptive-promoted keyframe targets and broader weak/sequence coverage probes.
- False-negative residual rows remain the main unresolved category and still behave essentially like uniform gap8.
- This is still sampled broader validation, not final full-sequence RD.
- Adaptive keyframe rows use q12 keyframe reconstruction quality, not Stage158 middle-recovery quality.

## Outputs

- Package: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_broader_sampled_adaptive_validation_execution_package.json`
- Report: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_broader_sampled_adaptive_validation_execution_report.md`
- Render rows CSV: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_broader_validation_rows.csv`
- Render summary CSV: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_broader_validation_render_summary.csv`
- Source summary CSV: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_broader_validation_source_summary.csv`
- Final quality CSV: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_final_quality_by_schedule.csv`
- Per-target delta CSV: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_adaptive_vs_fixed_gap_target_deltas.csv`
- Final summary CSV: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_final_quality_summary.csv`
- Category delta CSV: `experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_category_delta_summary.csv`
