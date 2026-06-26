# Stage52 FCGS/D-FCGS Baseline Preflight

## Outputs
- Log records CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_dfcgs_log_records.csv`
- Summary records CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_baseline_summary_records.csv`
- Summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_fcgs_dfcgs_baseline_preflight_summary.json`

## Availability
| Source group | Root exists | File count | Pattern |
|---|---:|---:|---|
| legacy_driving_dfcgs_logs | True | 40 | `D-FCGS_20260620_*.log` |
| multisequence_dfcgs_logs | True | 159 | `**/D-FCGS_20260626_*.log` |
| multisequence_dfcgs_fcgs_i_summaries | True | 15 | `**/dfcgs_gop2_target_topology_summary.json` |
| compactworld_dfcgs_fcgs_i_lambda_sweep | True | 12 | `**/dfcgs_gop2_target_topology_summary.json` |
| compactworld_dfcgs_raw_i_rate_sweep | True | 12 | `**/dfcgs_gop2_target_topology_summary.json` |
| compactworld_dfcgs_fcgs_i_gop2 | True | 4 | `**/dfcgs_gop2_target_topology_summary.json` |
| lowrate_dfcgs_fcgs_i_summaries | True | 16 | `**/dfcgs_gop2_target_topology_summary.json` |
| compactworld_standalone_fcgs_summaries | True | 6 | `fcgs_*_out/fcgs_sequence_summary.json` |
| multisequence_standalone_fcgs_summaries | True | 33 | `**/fcgs_sequence_summary.json` |
| lowrate_standalone_fcgs_summaries | True | 17 | `**/fcgs_sequence_summary.json` |

## Parse Summary
- D-FCGS log records: 199
- D-FCGS log records with all key fields: 199
- Summary records: 199
- Summary records with rate: 199
- Summary records with input-video quality: 173
- Stage53 candidate summary records: 173

## Codec Modes
| Codec mode | Records |
|---|---:|
| fcgs_i_plus_dfcgs_p_gop2 | 47 |
| fcgs_per_frame | 140 |
| raw_i_plus_dfcgs_p_gop2 | 12 |

## Sample Coverage
| Source group | Sample | Records |
|---|---|---:|
| compactworld_dfcgs_fcgs_i_gop2 | driving | 1 |
| compactworld_dfcgs_fcgs_i_gop2 | meetroom | 1 |
| compactworld_dfcgs_fcgs_i_gop2 | n3dv | 1 |
| compactworld_dfcgs_fcgs_i_gop2 | robot | 1 |
| compactworld_dfcgs_fcgs_i_lambda_sweep | driving | 3 |
| compactworld_dfcgs_fcgs_i_lambda_sweep | meetroom | 3 |
| compactworld_dfcgs_fcgs_i_lambda_sweep | n3dv | 3 |
| compactworld_dfcgs_fcgs_i_lambda_sweep | robot | 3 |
| compactworld_dfcgs_raw_i_rate_sweep | driving | 3 |
| compactworld_dfcgs_raw_i_rate_sweep | meetroom | 3 |
| compactworld_dfcgs_raw_i_rate_sweep | n3dv | 3 |
| compactworld_dfcgs_raw_i_rate_sweep | robot | 3 |
| compactworld_standalone_fcgs_summaries | driving | 6 |
| compactworld_standalone_fcgs_summaries | meetroom | 5 |
| compactworld_standalone_fcgs_summaries | n3dv | 6 |
| compactworld_standalone_fcgs_summaries | robot | 5 |
| lowrate_dfcgs_fcgs_i_summaries | meetroom | 2 |
| lowrate_dfcgs_fcgs_i_summaries | n3dv | 5 |
| lowrate_dfcgs_fcgs_i_summaries | shaky_video | 2 |
| lowrate_dfcgs_fcgs_i_summaries | tree_and_building | 3 |
| lowrate_dfcgs_fcgs_i_summaries | video_editing_case1 | 4 |
| lowrate_standalone_fcgs_summaries | driving | 2 |
| lowrate_standalone_fcgs_summaries | meetroom | 2 |
| lowrate_standalone_fcgs_summaries | n3dv | 5 |
| lowrate_standalone_fcgs_summaries | shaky_video | 2 |
| lowrate_standalone_fcgs_summaries | tree_and_building | 4 |
| lowrate_standalone_fcgs_summaries | video_editing_case1 | 4 |
| multisequence_dfcgs_fcgs_i_summaries | n3dv | 15 |
| multisequence_standalone_fcgs_summaries | meetroom | 45 |
| multisequence_standalone_fcgs_summaries | n3dv | 54 |

## Stage53 Use Notes
- Use GOP summary rows, not raw single-P-frame logs, for first baseline tables.
- Treat rates as full FCGS/D-FCGS codec MiB/frame, not as Mono-DFCGS transmitted Gaussian-anchor MiB/frame.
- Exclude `dummy_reference_images=true` rows from input-video PSNR/SSIM comparisons; keep their `codec_psnr` only as compression-fidelity diagnostics.
- LPIPS is often per P-frame only or null for I-frames, so Stage53 should not assume a complete all-frame LPIPS curve.
