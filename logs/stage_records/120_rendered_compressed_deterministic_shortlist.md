# Stage120 Rendered Compressed Deterministic Shortlist

Date: 2026-06-29

## Goal

Render the Stage119 compressed deterministic shortlist to identify low-rate settings that preserve or improve quality.

## Implementation

Added:

```text
scripts/run_stage120_rendered_compressed_deterministic_shortlist.py
```

The script renders compressed deterministic decoded anchors for q6/top10, q5/top10, q4/top10, q6/top5, and q4/top20. It records payload bytes and direct/amortized rates with all side-info bytes counted.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage120_rendered_compressed_deterministic_shortlist.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage120_rendered_compressed_deterministic_shortlist.py
```

No broadcast warning was emitted.

## Outputs

```text
experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_rows.csv
experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_group_summary.csv
experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_setting_summary.csv
experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_summary.json
experiments/stage120_rendered_compressed_deterministic_shortlist/stage120_rendered_compressed_deterministic_shortlist_report.md
```

## Configuration

| item | value |
|---|---:|
| selector policy | `strict_safe_endpoint_selector_v1` |
| selected candidate | `endpoint_diff_baseline` |
| task count | 12 |
| row count | 120 |
| group count | 30 |
| zlib level | 9 |
| max decode diff | 0.0 |

## Results

| setting | keep | bits | payload bytes | side MiB | direct rate | amortized rate | PSNR | delta base | delta q6 | near q6 | positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q6_top10 | 0.1 | 6 | 29368.583333333332 | 0.028008063634236652 | 0.13265951988895094 | 0.12890187420248345 | 20.509201246149463 | 1.4493226762458462 | 0.0 | 24/24 | 24/24 |
| q5_top10 | 0.1 | 5 | 24682.291666666668 | 0.023538867632548015 | 0.1281903238872623 | 0.12503963726559633 | 20.503631356341803 | 1.4437527864381863 | -0.00556988980765986 | 24/24 | 24/24 |
| q4_top10 | 0.1 | 4 | 15117.083333333334 | 0.014416774113972982 | 0.11906823036868726 | 0.11714270480274513 | 20.474517727609136 | 1.41463915770552 | -0.03468351854032611 | 23/24 | 24/24 |
| q6_top5 | 0.05 | 6 | 15099.25 | 0.01439976692199707 | 0.11905122317671135 | 0.11711924789149915 | 19.89178909516828 | 0.8319105252646616 | -0.6174121509811845 | 0/24 | 24/24 |
| q4_top20 | 0.2 | 4 | 28241.333333333332 | 0.026933034261067707 | 0.131584490515782 | 0.12796449678936953 | 21.530766808788716 | 2.4708882388850957 | 1.02156556263925 | 24/24 | 24/24 |

## Conclusion

- q4/top20 is the best smoke candidate: higher PSNR than q6/top10 and slightly lower rate.
- q4/top10 is the best low-rate candidate: roughly half q6/top10 side-info with only a small mean PSNR drop.
- q5/top10 is near-identical to q6/top10 with lower rate.
- q6/top5 should be dropped due to large PSNR loss.
- Stage121 should broaden q6/top10, q5/top10, q4/top10, and q4/top20.
