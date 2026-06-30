# Stage185 Measured Full-Sequence RD Aggregation

Date: 2026-07-01

## Goal

Aggregate the Stage184 measured payloads into full-sequence rate totals for uniform gap8, Stage165 adaptive, and uniform gap4.

## Plan

- Join Stage183 frame/schedule rows with Stage184 residual payload measurements by `measurement_key`.
- Sum residual payload bytes per schedule over all non-keyframe recovered frames.
- Sum Stage184 schedule/sequence-packed q12 keyframe bitstream bytes per schedule.
- Add exact schedule metadata bytes from Stage181.
- Report measured keyframe, residual, metadata, and total MiB/frame for each schedule.
- Compare measured totals against the Stage181 proxy and compute deltas versus uniform gap8 and uniform gap4.

## Success Criteria

- All `5997` Stage183 frame/schedule rows are covered.
- Schedule counts match Stage183/181: gap8 `292/1707`, adaptive `358/1641`, gap4 `536/1463` keyframe/residual rows.
- A measured full-sequence RD table exists for Stage186+ and paper-facing reporting.

## Execution

- Pre-run `nvidia-smi`: GPU0/GPU4 busy, GPU1+ idle. Stage185 is CPU-only aggregation.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage185_measured_full_sequence_rd_aggregation.py`

## Results

- Status: `measured_full_sequence_rd_aggregation_packaged`.
- Decision: `adaptive_measured_rate_not_lower_than_gap8`.
- Coverage validation: all `5997 / 5997` frame/schedule rows covered; no missing residual or keyframe measurements.

Measured full-sequence rate:

| schedule | keyframes | residuals | keyframe MiB/frame | residual MiB/frame | metadata MiB/frame | total MiB/frame | delta vs gap8 | delta vs gap4 | Stage180 PSNR | Stage180 LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 292 | 1707 | 0.100233103288 | 0.175633072197 | 0.000000000477 | 0.275866175962 | 0.000000000000 | -0.054902768481 | 29.206326 | 0.176652 |
| stage165_adaptive | 358 | 1641 | 0.122888083694 | 0.167854693128 | 0.000000156004 | 0.290742932826 | 0.014876756864 | -0.040026011617 | 29.770753 | 0.142780 |
| uniform_gap4 | 536 | 1463 | 0.183987849351 | 0.146781094615 | 0.000000000477 | 0.330768944443 | 0.054902768481 | 0.000000000000 | 29.464217 | 0.162457 |

Interpretation:

- Adaptive remains lower-rate than uniform gap4 while giving higher Stage180 sampled quality than both fixed schedules.
- Adaptive is not lower-rate than uniform gap8 under measured payloads; it costs `+0.014876756864` MiB/frame for `+0.5644261202320328` dB sampled PSNR and lower LPIPS vs gap8.
- Stage188 should run threshold/budget sensitivity to recover a lower-rate adaptive variant if the final claim requires beating gap8 in measured rate.

## Outputs

- Package: `experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_full_sequence_rd_aggregation_package.json`
- Report: `experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_full_sequence_rd_aggregation_report.md`
- Total RD CSV: `experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_full_sequence_total_rd.csv`
- Sequence breakdown CSV: `experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_sequence_rd_breakdown.csv`
- Component breakdown CSV: `experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_component_breakdown.csv`
- Validation CSV: `experiments/stage185_measured_full_sequence_rd_aggregation/stage185_aggregation_validation.csv`

## Next Step

- Stage186 should produce an expanded measured-RD quality report using Stage180 quality plus Stage185 measured rates, and explicitly state that full all-frame quality remains future work unless rendered.
- Stage188 should explore lower-budget adaptive variants because Stage185 shows the frozen Stage165 candidate does not beat gap8 measured rate.
