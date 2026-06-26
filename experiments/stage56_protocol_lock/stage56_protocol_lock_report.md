# Stage56 Protocol Lock

## Locked Decisions

- Default user-facing quality metric: all-frame PSNR.
- Main keyframe-Gaussian-only rate: transmitted Gaussian anchor bitstream MiB/frame plus required anchor metadata.
- Optional side-information variants must report anchor rate, side-info rate, and total rate.
- `rendered_prior_0p1` is an oracle/calibrated upper bound, not the final deployable selector.
- Final adaptive selector must be frozen, feed-forward, deterministic, and must not use rendered oracle or PSNR lookahead at test time.
- Medium/long training runs are required for strong adapter/selector claims; short runs are smoke/infrastructure only.

## Rate Accounting

| Rate name | Main rate | Total rate | Unit | Description |
|---|---|---|---|---|
| anchor_bitstream_rate | yes | yes | MiB/frame | Compressed transmitted keyframe Gaussian anchor payload plus required anchor container metadata. |
| keyframe_indices_timestamps_metadata | yes | yes | MiB/frame or bytes/sequence | Keyframe indices, timestamps, quantization bits, tensor shapes, field schema, and codec header metadata. |
| side_information_rate | no_for_keyframe_gaussian_only; yes_for_side_info_variant_total | yes_if_transmitted | MiB/frame | Any transmitted non-keyframe depth, motion hint, residual, importance map, or correction payload. |
| decoder_model_weights | no | reported_separately | MiB/model | Shared decoder/adapter weights are not counted as per-video payload, but must be reported separately where relevant. |
| non_keyframe_rgb_depth_or_gaussians | not_allowed_in_current_main_method | yes_if_future_variant_transmits_it | MiB/frame | Current Mono-DFCGS-KG does not transmit these. Future side-info variants must count them if used. |

## Method Deployability

| Method family | Final claim | Allowed test-time inputs | Forbidden test-time inputs | Notes |
|---|---|---|---|---|
| uniform_keyframes | yes | video length, fixed gap, transmitted keyframe Gaussian anchors | rendered error, PSNR, reconstructed output lookahead | Baseline for all adaptive comparisons. |
| rendered_prior_0p1_oracle_calibrated | no | offline analysis labels only | must not be used as final deployed selector because it relies on rendered-error-related information | Upper bound / training target only. |
| feed_forward_selector | yes_if_frozen_and_deterministic | original input video, encoder-side features, endpoint keyframe candidate stats, deterministic DP output | rendered oracle, PSNR labels, test-time optimization over reconstructed frames | Final adaptive selector target. |
| teacher_distilled_adapter | yes_if_teacher_not_used_at_test_time | transmitted keyframe anchors, timestamps, optional counted side information | StreamSplat RGB/depth-conditioned teacher output at test time | Training may use teacher supervision; deployment may not depend on teacher inputs. |
| side_information_variant | yes_if_all_side_info_is_transmitted_and_counted | anchor bitstream plus explicitly counted side-information bitstream | free non-keyframe RGB/depth/motion/residual inputs | Report anchor rate, side-info rate, total rate, and all-frame PSNR separately. |

## Standard Tables

| Table | Required fields | Notes |
|---|---|---|
| main_rd_table | method, compression, adapter, selector, side_info, anchor_rate_mib_per_frame, side_info_rate_mib_per_frame, total_rate_mib_per_frame, all_psnr | Main paper/user-facing table. Do not include middle-only metrics unless requested. |
| compression_ablation_table | codec, bits_or_profile, raw_rate_mib_per_frame, compressed_rate_mib_per_frame, rate_saving_percent, all_psnr | Shows compression contribution. |
| adapter_ablation_table | adapter_variant, training_steps, train_dataset, val_dataset, best_step, all_psnr, delta_vs_linear | Shows adapter contribution and training scale. |
| selector_ablation_table | selector_variant, deployable, oracle_used_at_test_time, rate_mib_per_frame, all_psnr, delta_vs_uniform, negative_point_count | Separates oracle upper bounds from final feed-forward selectors. |
| training_run_summary | run_id, stage, model_variant, dataset, steps, best_step, checkpoint_path_external, validation_all_psnr, notes | Used for medium/long training runs. Checkpoints stay outside git. |

## Output Files

- Summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage56_protocol_lock/stage56_protocol_lock_summary.json`
- Rate accounting CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage56_protocol_lock/stage56_rate_accounting_rules.csv`
- Method deployability CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage56_protocol_lock/stage56_method_deployability_rules.csv`
- Standard table schema CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage56_protocol_lock/stage56_standard_table_schemas.csv`
