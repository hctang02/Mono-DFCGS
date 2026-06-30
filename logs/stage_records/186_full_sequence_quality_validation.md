# Stage186 Full-Sequence Quality Validation

Date: 2026-07-01

## Goal

Measure full-sequence reconstruction quality for the three schedules using Stage184 measured payload decisions and Stage185 measured rates.

## Plan

- Render and score all unique q12 keyframe reconstructions with PSNR, SSIM, MS-SSIM, and LPIPS.
- Re-render all unique Stage158 residual reconstructions using the Stage184 selected half and compute PSNR, SSIM, MS-SSIM, and LPIPS.
- Join the unique quality rows back to all Stage183 frame/schedule rows.
- Aggregate full-sequence quality by schedule for uniform gap8, Stage165 adaptive, and uniform gap4.
- Merge Stage186 quality with Stage185 measured rates into measured RD-quality points.
- Support smoke limits and resume because this stage is GPU-heavy.

## Success Criteria

- Unique keyframe quality rows: `596 / 596`.
- Unique residual quality rows: `3472 / 3472`.
- Final frame/schedule quality rows: `5997 / 5997`.
- Full-sequence multi-metric quality summary and measured RD-quality points exist for downstream ablations and paper-facing reporting.

## Execution

- Pre-run `nvidia-smi`: GPU0/GPU4 busy; GPU1 idle.
- Smoke command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage186_full_sequence_quality_validation.py --device cuda --max_keyframes 2 --max_residuals 2 --batch_size 2 --flush_every 1`.
- Full command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage186_full_sequence_quality_validation.py --device cuda --batch_size 8 --flush_every 16`.
- Report regeneration after correcting interpretation: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage186_full_sequence_quality_validation.py --device cuda --skip_keyframes --skip_residuals`.

## Results

- Status: `full_sequence_quality_validation_complete`.
- Decision: `adaptive_quality_rate_between_gap8_and_gap4`.
- Unique keyframe quality rows: `596 / 596`.
- Unique residual quality rows: `3472 / 3472`.
- Final frame/schedule quality rows: `5997 / 5997`.

Full-sequence quality and measured rate:

| schedule | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS | delta PSNR vs gap8 | delta LPIPS vs gap8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 0.275866175962 | 29.373965 | 0.867626 | 0.984343 | 0.168692 | 0.000000 | 0.000000 |
| stage165_adaptive | 0.290742932826 | 29.425583 | 0.869294 | 0.984647 | 0.165937 | 0.051618 | -0.002754 |
| uniform_gap4 | 0.330768944443 | 29.535716 | 0.873944 | 0.985529 | 0.159472 | 0.161751 | -0.009220 |

Interpretation:

- Adaptive improves all measured full-sequence quality metrics over uniform gap8.
- Adaptive remains below uniform gap4 quality, but also uses lower measured rate than gap4.
- The frozen Stage165 candidate is therefore a middle RD point between gap8 and gap4, not a strict lower-rate-than-gap8 result.
- Stage188 should search for a lower-budget adaptive variant that keeps most of the gap8 quality gain while reducing the gap8 rate overhead.

## Outputs

- Package: `experiments/stage186_full_sequence_quality_validation/stage186_full_sequence_quality_validation_package.json`
- Report: `experiments/stage186_full_sequence_quality_validation/stage186_full_sequence_quality_validation_report.md`
- Unique keyframe quality CSV: `experiments/stage186_full_sequence_quality_validation/stage186_unique_keyframe_quality_metrics.csv`
- Unique residual quality CSV: `experiments/stage186_full_sequence_quality_validation/stage186_unique_stage158_residual_quality_metrics.csv`
- Full frame/schedule quality CSV: `experiments/stage186_full_sequence_quality_validation/stage186_full_sequence_quality_by_schedule.csv`
- Full quality summary CSV: `experiments/stage186_full_sequence_quality_validation/stage186_full_sequence_quality_summary.csv`
- Measured RD-quality CSV: `experiments/stage186_full_sequence_quality_validation/stage186_measured_rd_quality_points.csv`

## Next Step

- Stage187 should run selector ablations using the full-sequence measured metric tables where possible.
- Stage188 should prioritize lower-budget selector variants because measured adaptive rate is `+0.014876756863691831` MiB/frame above gap8.
