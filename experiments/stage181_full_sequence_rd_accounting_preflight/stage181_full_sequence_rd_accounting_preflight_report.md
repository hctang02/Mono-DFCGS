# Stage181 Full-Sequence RD Accounting Preflight

## Scope

This stage separates exact full-sequence keyframe/metadata counting from sampled-estimated residual payload accounting.
It is a preflight, not final full-sequence RD.

## Decision

- Decision: `adaptive_rate_promising_under_broader_sampled_proxy`.
- Adaptive total proxy delta vs gap8: `-0.11567233082795791` MiB/frame.
- Adaptive total proxy delta vs gap4: `-0.17722082115678273` MiB/frame.

## Full-Sequence Keyframe And Metadata Counts

| schedule | frames | keyframes | keyframe ratio | main anchor MiB/frame proxy | metadata bytes | metadata MiB/frame |
|---|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 1999 | 292 | 0.146073 | 0.097625386754 | 1 | 0.000000000477 |
| stage165_adaptive | 1999 | 358 | 0.179090 | 0.120431317266 | 327 | 0.000000156004 |
| uniform_gap4 | 1999 | 536 | 0.268134 | 0.181938220768 | 1 | 0.000000000477 |

## Stage180 Broader Residual Proxy

| schedule | targets | keyframes | middle recovery | mean payload bytes/target | residual MiB/target | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 90 | 0 | 90 | 219878.578 | 0.209692552355 | 29.206326 | 0.176652 |
| stage165_adaptive | 90 | 56 | 34 | 74673.433 | 0.071214135488 | 29.770753 | 0.142780 |
| uniform_gap4 | 90 | 8 | 82 | 196008.433 | 0.186928208669 | 29.464217 | 0.162457 |

## Combined Proxy

| schedule | main anchor | metadata | residual proxy | total proxy | delta vs gap8 | delta vs gap4 |
|---|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 0.097625386754 | 0.000000000477 | 0.209692552355 | 0.307317939585 | 0.000000000000 | -0.061548490329 |
| stage165_adaptive | 0.120431317266 | 0.000000156004 | 0.071214135488 | 0.191645608757 | -0.115672330828 | -0.177220821157 |
| uniform_gap4 | 0.181938220768 | 0.000000000477 | 0.186928208669 | 0.368866429914 | 0.061548490329 | 0.000000000000 |

## Requirements Before Final RD

| item | status | next required work |
|---|---|---|
| full_sequence_keyframe_indices | counted_exact_from_stage165_schedule_rows | none_for_schedule_counting |
| schedule_metadata | counted_exact_as_327_bytes_for_adaptive_and_1_byte_mode_for_uniforms | replace with actual bitstream syntax if final codec changes |
| main_anchor_payload | proxy_from_stage172_interpolated_anchor_accounting | measure actual q12 keyframe bitstreams for every transmitted keyframe before paper-level RD |
| stage158_residual_payload | stage180_broader_sampled_estimate | run all-frame/full-sequence residual payload encode for every non-keyframe recovered frame |
| all_frame_quality | not_final_full_sequence_quality | evaluate all frames or declared sampled protocol with sequence-level reporting |
| decoder_contract | schedule_and_payload_transmitted_no_rgb_motion_features_at_decoder | keep in final package and count any additional side-info |

## Outputs

- Keyframe/metadata CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage181_full_sequence_rd_accounting_preflight/stage181_full_sequence_keyframe_metadata_accounting.csv`
- Residual proxy CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage181_full_sequence_rd_accounting_preflight/stage181_stage180_residual_payload_proxy.csv`
- Total proxy CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage181_full_sequence_rd_accounting_preflight/stage181_total_rate_proxy_comparison.csv`
- Requirements CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage181_full_sequence_rd_accounting_preflight/stage181_final_rd_requirements.csv`
