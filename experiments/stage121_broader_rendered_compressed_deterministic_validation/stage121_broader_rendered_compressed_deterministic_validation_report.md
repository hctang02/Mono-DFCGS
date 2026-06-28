# Stage121 Broader Rendered Compressed Deterministic Validation

## Configuration

- task count: `60`
- shortlist: `[{'label': 'q6_top10', 'keep_fraction': 0.1, 'side_bits': 6}, {'label': 'q5_top10', 'keep_fraction': 0.1, 'side_bits': 5}, {'label': 'q4_top10', 'keep_fraction': 0.1, 'side_bits': 4}, {'label': 'q4_top20', 'keep_fraction': 0.2, 'side_bits': 4}]`
- zlib level: `9`
- all side-info bytes are counted in direct/amortized total rates
- residual values are teacher-derived; no residual value predictor is trained

## Setting Summary

| setting | keep | bits | payload bytes | side MiB | direct rate | amortized rate | PSNR | delta base | delta q6 | near q6 | positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q6_top10 | 0.1 | 6 | 29442.208333 | 0.028078 | 0.134838 | 0.130956 | 19.766969 | 1.282949 | 0.000000 | 120/120 | 120/120 |
| q5_top10 | 0.1 | 5 | 24809.950000 | 0.023661 | 0.130420 | 0.127150 | 19.761048 | 1.277028 | -0.005921 | 120/120 | 120/120 |
| q4_top10 | 0.1 | 4 | 15190.475000 | 0.014487 | 0.121246 | 0.119247 | 19.738488 | 1.254468 | -0.028480 | 114/120 | 120/120 |
| q4_top20 | 0.2 | 4 | 28320.791667 | 0.027009 | 0.133768 | 0.130054 | 20.689271 | 2.205251 | 0.922302 | 120/120 | 120/120 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_rows.csv`
- group summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_group_summary.csv`
- setting summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_setting_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_summary.json`

## Notes

- q6_top10 remains the same-task quality anchor.
- q4_top20 and q4_top10 are the primary candidates for RD packaging if they remain stable.
- The next stage should package RD only after reviewing group-level stability.
