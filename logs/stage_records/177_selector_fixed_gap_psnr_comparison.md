# Stage177 Selector Fixed-Gap PSNR Comparison

Date: 2026-06-30

## Goal

Compare the final target PSNR of the adaptive selector schedule against fixed uniform gap8 and uniform gap4 on the Stage174 medium validation targets.

## Plan

- Load Stage174 medium validation rows.
- For rendered middle-recovery rows, reuse Stage174 PSNR/SSIM/MS-SSIM/LPIPS.
- For schedule rows where the target is transmitted as a keyframe, render the target q12 keyframe anchor and compute PSNR/SSIM/MS-SSIM/LPIPS.
- Produce per-schedule final quality rows, per-target deltas, and per-category summaries.
- Keep the distinction explicit: adaptive keyframe PSNR is q12 keyframe reconstruction PSNR, not middle-recovery PSNR.

## Success Criteria

- Report shows adaptive vs fixed gap8/gap4 mean final PSNR and paired deltas.
- Row-level CSV makes selected/keyframe targets visible.
- No heavy media/checkpoint/anchor files are committed.

## Execution

- Checked `nvidia-smi` before running; GPU 1 was idle.
- Compiled and ran with `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.
- Stage177 rendered q12 target keyframe anchors only for target-keyframe rows and reused Stage174 metrics for rendered middle-recovery rows.
- LPIPS and MS-SSIM modules loaded successfully.

## Results

Overall final target quality on the Stage174 medium target set:

| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 50 | 0 | 50 | 28.756927 | 25.130052 | 0.189479 | 211348.720 |
| stage165_adaptive | 50 | 26 | 24 | 29.217096 | 25.223247 | 0.157234 | 95704.400 |
| uniform_gap4 | 50 | 6 | 44 | 28.945814 | 25.108181 | 0.177610 | 181242.240 |

Paired adaptive deltas:

- Adaptive minus uniform gap8 PSNR: `+0.4601686250283185` dB.
- Adaptive minus uniform gap4 PSNR: `+0.27128186202823` dB.
- Adaptive minus uniform gap8 LPIPS: `-0.03224517673254013`.
- Adaptive minus uniform gap4 LPIPS: `-0.02037559390068054`.
- Adaptive keyframe targets: `26 / 50`.

Category PSNR deltas:

| category | targets | adaptive keyframes | delta vs gap8 | delta vs gap4 |
|---|---:|---:|---:|---:|
| false_negative_residual | 8 | 0 | -0.010964 | -0.326294 |
| high_payload_residual_control | 4 | 0 | +0.315565 | +0.276767 |
| high_payload_residual_control_extension | 8 | 0 | 0.000000 | -0.291864 |
| normal_residual_control | 4 | 0 | 0.000000 | -0.011207 |
| positive_promoted | 14 | 14 | +1.248305 | +0.879899 |
| positive_promoted_extension | 8 | 8 | +0.346645 | +0.501244 |
| selector_false_positive_keyframe_control | 4 | 4 | +0.396114 | +0.279645 |

## Decision

- Stage177 status: `selector_fixed_gap_psnr_comparison_packaged`.
- The Stage165 adaptive schedule has higher sampled-medium final target PSNR than both uniform gap8 and uniform gap4 under this comparison definition.
- Adaptive keyframe improvements are q12 keyframe reconstruction quality, not Stage158 middle-recovery quality.
- This remains a Stage174 medium-set comparison, not final full-sequence RD.

## Outputs

- Package: `experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_selector_fixed_gap_psnr_comparison_package.json`
- Report: `experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_selector_fixed_gap_psnr_comparison_report.md`
- Per-schedule quality CSV: `experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_final_quality_by_schedule.csv`
- Per-target delta CSV: `experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_adaptive_vs_fixed_gap_target_deltas.csv`
- Schedule summary CSV: `experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_final_quality_summary.csv`
- Category delta CSV: `experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_category_delta_summary.csv`
