# Stage172 Keyframe Rate Accounting Audit

## Decision

- Decision: `adaptive_rate_promising_for_medium_protocol`.
- Adaptive remains rate-promising under the Stage166 sampled proxy after charging extra keyframes and metadata.
- This still does not constitute final full-sequence RD; it is sufficient to design Stage173 medium rendered validation.

## Components

| schedule | keyframes | extra vs gap8 | main anchor MiB/frame | residual MiB/sample | metadata MiB/frame | total proxy MiB/frame | delta vs gap8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 292 | 0 | 0.097625386754 | 0.202827795347 | 0.000000000477 | 0.300453182577 | 0.000000000000 |
| stage165_adaptive | 358 | 66 | 0.120431317266 | 0.073750042915 | 0.000000155646 | 0.194181515827 | -0.106271666750 |
| uniform_gap4 | 536 | 244 | 0.181938220768 | 0.188585289319 | 0.000000000477 | 0.370523510564 | 0.070070327987 |

## Notes

- `per_extra_keyframe_mib_per_frame`: `0.00034554440169835566`. Derived from uniform gap8/gap4 main-anchor proxy rate difference divided by keyframe-count difference.
- `per_extra_keyframe_payload_mib`: `0.690743258995013`. Dataset-level proxy payload for one added keyframe, equal to per-frame cost times total frames.
- `accounting_scope`: `sampled_proxy`. Residual side-info is measured on the 120 Stage163/166 sampled rows; full-sequence residual decisions remain future work.
- `decoder_contract`: `schedule_metadata_transmitted`. Decoder receives counted schedule/keyframe metadata and does not recompute selector RGB/motion features.

## Interpretation

- Stage165 adaptive adds `66` keyframes over uniform gap8 but avoids much more sampled residual side-info on selected rows.
- The extra schedule metadata is tiny at `327` bytes total for the Stage165 schedule.
- Because false negatives remain, Stage173 must include false-negative controls and should not only sample selected/promoted rows.
