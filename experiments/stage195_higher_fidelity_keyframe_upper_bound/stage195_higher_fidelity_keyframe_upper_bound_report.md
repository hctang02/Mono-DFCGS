# Stage195 Higher-Fidelity Keyframe Upper-Bound

## Decision

- Decision: `higher_fidelity_keyframes_improve_gap2_but_below_target_margin`.
- Complete: `True`.

## RD-Quality / Quality Upper Bounds

| representation | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs gap2 | dLPIPS vs gap2 | +1dB pass | rate scope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| q16_keyframe | 0.914693216016 | 29.884665 | 0.886870 | 0.988630 | 0.136015 | 0.229850 | -0.015666 | 0 | measured_schedule_packed_q16_keyframes_plus_fixed_gap_metadata |
| float_dense_anchor | NA | 29.884931 | 0.886882 | 0.988631 | 0.135990 | 0.230116 | -0.015692 | 0 | quality_upper_bound_no_payload_rate |

## Validation

| item | expected | actual | status |
|---|---:|---:|---|
| protocol_frame_count | 1999 | 1999 | ok |
| q16_schedule_keyframe_groups | 30 | 30 | ok |
| q16_keyframe_quality_rows | 1999 | 1999 | ok |
| q16_keyframe_missing_quality_rows | 0 | 0 | ok |
| float_dense_anchor_quality_rows | 1999 | 1999 | ok |
| float_dense_anchor_missing_quality_rows | 0 | 0 | ok |

## Interpretation

- q16 includes measured schedule-packed keyframe rate; float dense-anchor is quality-only and has no deployable payload claim.
- If float dense-anchor quality is still below the +1 dB target over Stage192 `uniform_gap2`, a stronger selector or higher quantization alone cannot satisfy the requested full-sequence gain.

## Outputs

- Summary CSV: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_higher_fidelity_keyframe_summary.csv`
- Quality CSV: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_higher_fidelity_keyframe_quality_metrics.csv`
- q16 schedule payload CSV: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_q16_schedule_packed_keyframe_payload_measurements.csv`
