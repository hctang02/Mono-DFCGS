# Stage192 Expanded Fixed-Gap Measurement

Date: 2026-07-01

## Goal

Measure and aggregate full-sequence RD-quality for the expanded fixed-gap baseline set: `gap2`, `gap4`, `gap6`, `gap8`, `gap16`, and current `stage165_adaptive`.

## Plan

- Seed Stage192 outputs from existing Stage184/186 payload and quality measurements for reusable gap4/gap8/adaptive rows.
- Measure missing q12 single-keyframe payloads, schedule-packed keyframe payload groups, Stage158 residual payloads, q12 keyframe quality, and Stage158 residual quality for gap2/gap6/gap16 rows.
- Aggregate schedule-packed keyframe bytes, residual bytes, exact metadata, PSNR, SSIM, MS-SSIM, and LPIPS for every expanded schedule.
- Identify the best fixed-gap baseline and compare current adaptive against it.

## Success Criteria

- All Stage191 expected payload and quality rows are covered.
- Expanded RD-quality table includes `uniform_gap2`, `uniform_gap4`, `uniform_gap6`, `uniform_gap8`, `uniform_gap16`, and `stage165_adaptive`.
- The report states whether current adaptive beats the best fixed-gap baseline, and by how much.
- Heavy intermediate artifacts remain outside git under `/data/hctang/tmp/opencode/mono_dfcgs_runs/`.

## Execution

- Pre-run GPU check: `nvidia-smi` recorded on 2026-07-01 14:06:31 and again before full measurement at 14:09:26.
- GPU execution: `CUDA_VISIBLE_DEVICES=2 ... --device cuda` to avoid the prior physical `cuda:1` issue.
- Smoke command: `CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage192_expanded_fixed_gap_measurement.py --device cuda --max_payload_keyframes 10 --max_schedule_keyframe_groups 2 --max_payload_residuals 4 --max_quality_keyframes 10 --max_quality_residuals 4 --flush_every 2`
- Full command: `CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage192_expanded_fixed_gap_measurement.py --device cuda --payload_batch_size 2 --quality_batch_size 8 --flush_every 500`
- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage192_expanded_fixed_gap_measurement.py`

## Outputs

- Package: `experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_measurement_package.json`
- Report: `experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_measurement_report.md`
- RD-quality table: `experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_rd_quality_points.csv`
- Total RD table: `experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_total_rd.csv`
- Quality summary: `experiments/stage192_expanded_fixed_gap_measurement/stage192_full_sequence_quality_summary.csv`
- Final frame/schedule quality rows: `experiments/stage192_expanded_fixed_gap_measurement/stage192_full_sequence_quality_by_schedule.csv`
- Payload measurements: `experiments/stage192_expanded_fixed_gap_measurement/stage192_unique_keyframe_payload_measurements.csv`, `experiments/stage192_expanded_fixed_gap_measurement/stage192_unique_stage158_residual_payload_measurements.csv`, `experiments/stage192_expanded_fixed_gap_measurement/stage192_schedule_packed_keyframe_payload_measurements.csv`
- Quality measurements: `experiments/stage192_expanded_fixed_gap_measurement/stage192_unique_keyframe_quality_metrics.csv`, `experiments/stage192_expanded_fixed_gap_measurement/stage192_unique_stage158_residual_quality_metrics.csv`
- Validation: `experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_validation.csv`

## Results

Validation:

- All schedules have `1999/1999` final quality rows.
- Unique keyframe payload/quality rows: `1065/1065`.
- Unique residual payload/quality rows: `7791/7791`.
- Schedule-packed keyframe payload groups are complete for all expanded schedules.

Expanded full-sequence RD-quality:

| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs best fixed | dLPIPS vs best fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform_gap2` | `0.4495468821866683` | `1025` | `29.654815328772308` | `0.878375951948018` | `0.9866168332910943` | `0.15168131759139583` | `0.0` | `0.0` |
| `uniform_gap4` | `0.33076894444307725` | `536` | `29.535715839048734` | `0.8739438994697716` | `0.9855294218654929` | `0.15947172297849663` | `-0.11909948972357398` | `0.007790405387100796` |
| `uniform_gap6` | `0.29344506237493745` | `372` | `29.448737801531657` | `0.8706193330169857` | `0.9848943316024086` | `0.16457491443157493` | `-0.20607752724065165` | `0.0128935968401791` |
| `uniform_gap8` | `0.2758661759621266` | `292` | `29.373964871839835` | `0.867625699572828` | `0.9843430183660156` | `0.16869177970254404` | `-0.28085045693247324` | `0.017010462111148206` |
| `uniform_gap16` | `0.2514788301781811` | `169` | `29.199328663742687` | `0.8603573565843285` | `0.9830296424521751` | `0.1773263292425331` | `-0.4554866650296212` | `0.02564501165113728` |
| `stage165_adaptive` | `0.2907429328258184` | `358` | `29.4255826920606` | `0.8692941793565335` | `0.9846469353830415` | `0.16593745923142186` | `-0.22923263671170702` | `0.014256141640026032` |

## Decision

- Decision: `current_adaptive_not_strong_against_expanded_fixed_gaps`.
- Best fixed gap by PSNR is `uniform_gap2`.
- Current Stage165 adaptive is below `uniform_gap2` by `-0.22923263671170702` dB PSNR and has worse LPIPS by `+0.014256141640026032`.
- Stage193 should compute oracle headroom before designing a new selector; the current adaptive selector is not enough for the user's requested strong full-sequence claim.
