# Stage121 Broader Rendered Compressed Deterministic Validation

Date: 2026-06-29

## Goal

Validate the Stage120 compressed deterministic shortlist on a broader 60-task eval slice.

## Implementation

Added:

```text
scripts/run_stage121_broader_rendered_compressed_deterministic_validation.py
```

The script evaluates q6/top10, q5/top10, q4/top10, and q4/top20. q6/top5 was dropped after Stage120.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage121_broader_rendered_compressed_deterministic_validation.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage121_broader_rendered_compressed_deterministic_validation.py
```

## Outputs

```text
experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_rows.csv
experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_group_summary.csv
experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_setting_summary.csv
experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_summary.json
experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_report.md
```

## Configuration

| item | value |
|---|---:|
| selector policy | `strict_safe_endpoint_selector_v1` |
| selected candidate | `endpoint_diff_baseline` |
| task count | 60 |
| row count | 480 |
| group count | 24 |
| zlib level | 9 |
| max decode diff | 0.0 |

## Results

| setting | keep | bits | payload bytes | side MiB | direct rate | amortized rate | PSNR | delta base | delta q6 | near q6 | positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q6_top10 | 0.1 | 6 | 29442.208333333332 | 0.028078277905782063 | 0.1348375550108561 | 0.13095624756787308 | 19.766968650683353 | 1.2829489099582205 | 0.0 | 120/120 | 120/120 |
| q5_top10 | 0.1 | 5 | 24809.95 | 0.023660612106323243 | 0.13041988921139727 | 0.127149533351005 | 19.761047533309117 | 1.2770277925839815 | -0.005921117374238853 | 120/120 | 120/120 |
| q4_top10 | 0.1 | 4 | 15190.475 | 0.01448676586151123 | 0.12124604296658527 | 0.11924717736218463 | 19.73848817438193 | 1.2544684336567946 | -0.028480476301425663 | 114/120 | 120/120 |
| q4_top20 | 0.2 | 4 | 28320.791666666668 | 0.027008811632792156 | 0.1337680887378662 | 0.13005386354500142 | 20.689270746602087 | 2.2052510058769683 | 0.9223020959187475 | 120/120 | 120/120 |

## Conclusion

- q4/top20 is the primary candidate: higher PSNR than q6/top10 and slightly lower rate.
- q4/top10 is the low-rate candidate: much lower side-info with small quality loss vs q6/top10.
- q5/top10 is a near-anchor candidate with lower rate.
- Stage122 should package RD with all side-info bytes counted.
