# Stage120 Rendered Compressed Deterministic Shortlist

## Configuration

- task count: `12`
- shortlist: `[{'label': 'q6_top10', 'keep_fraction': 0.1, 'side_bits': 6}, {'label': 'q5_top10', 'keep_fraction': 0.1, 'side_bits': 5}, {'label': 'q4_top10', 'keep_fraction': 0.1, 'side_bits': 4}, {'label': 'q6_top5', 'keep_fraction': 0.05, 'side_bits': 6}, {'label': 'q4_top20', 'keep_fraction': 0.2, 'side_bits': 4}]`
- zlib level: `9`
- all side-info bytes are counted in direct/amortized total rates
- residual values are teacher-derived; no residual value predictor is trained

## Setting Summary

| setting | keep | bits | payload bytes | side MiB | direct rate | amortized rate | PSNR | delta base | delta q6 | near q6 | positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q6_top10 | 0.1 | 6 | 29368.583333 | 0.028008 | 0.132660 | 0.128902 | 20.509201 | 1.449323 | 0.000000 | 24/24 | 24/24 |
| q5_top10 | 0.1 | 5 | 24682.291667 | 0.023539 | 0.128190 | 0.125040 | 20.503631 | 1.443753 | -0.005570 | 24/24 | 24/24 |
| q4_top10 | 0.1 | 4 | 15117.083333 | 0.014417 | 0.119068 | 0.117143 | 20.474518 | 1.414639 | -0.034684 | 23/24 | 24/24 |
| q6_top5 | 0.05 | 6 | 15099.250000 | 0.014400 | 0.119051 | 0.117119 | 19.891789 | 0.831911 | -0.617412 | 0/24 | 24/24 |
| q4_top20 | 0.2 | 4 | 28241.333333 | 0.026933 | 0.131584 | 0.127964 | 21.530767 | 2.470888 | 1.021566 | 24/24 | 24/24 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_rows.csv`
- group summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_group_summary.csv`
- setting summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_setting_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_summary.json`

## Notes

- q6_top10 is the quality anchor for this deterministic endpoint-diff selector line.
- `near q6` counts rows with PSNR no worse than 0.10 dB below q6_top10 for the same task/base.
- If a lower-rate setting is close to q6_top10, it should be broadened in Stage121.
