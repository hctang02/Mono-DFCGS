# Stage53 Baseline Comparison Scaffold

## Outputs
- Unified scaffold CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage53_baseline_comparison_scaffold/stage53_baseline_comparison_scaffold.csv`
- Mono-DFCGS rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage53_baseline_comparison_scaffold/stage53_mono_dfcgs_rows.csv`
- External baseline rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage53_baseline_comparison_scaffold/stage53_external_baseline_rows.csv`
- Method/sample aggregate CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage53_baseline_comparison_scaffold/stage53_method_sample_aggregate.csv`
- Summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage53_baseline_comparison_scaffold/stage53_baseline_comparison_scaffold_summary.json`

## Counts
- Unified rows: 391
- Mono-DFCGS rows: 192
- External baseline rows: 199
- External rows with input-video quality and rate: 173
- Fair external apples-to-apples rows: 0

## Methods
| Method | Rows |
|---|---:|
| FCGS | 140 |
| FCGS-I + D-FCGS-P | 47 |
| Mono-DFCGS adaptive | 96 |
| Mono-DFCGS uniform | 96 |
| Raw-I + D-FCGS-P | 12 |

## Comparison Status
| Status | Rows |
|---|---:|
| local external diagnostic only; dummy references do not provide input-video quality | 25 |
| local external incomplete candidate | 1 |
| local external protocol reference; rate/quality available but not apples-to-apples with Mono-DFCGS | 173 |
| our local deployable uniform-keyframe baseline | 96 |
| our local oracle-calibrated selector; not final deployable selector | 96 |

## Usage Notes
- Use Mono-DFCGS rows for our own RD curves; rate is transmitted Gaussian-anchor bitstream MiB/frame and excludes model weights.
- Use external rows as local protocol-reference baselines only unless a future Stage adds matched inputs, matched frame sets, and matched rate accounting.
- Do not mix `full FCGS/D-FCGS codec MiB/frame` with Mono-DFCGS anchor-only MiB/frame on an unlabeled primary plot.
- Rows with `quality_reliable_for_input_video=false` are diagnostic only; their codec PSNR can be reported separately from input-video PSNR.
