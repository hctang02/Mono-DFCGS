# Stage122 Compressed Deterministic RD Package

Date: 2026-06-29

## Goal

Package the Stage121 broader rendered compressed deterministic validation into RD rows, RD points, a setting summary, JSON package, and Markdown report.

## Implementation

Added:

```text
scripts/run_stage122_compressed_deterministic_rd_package.py
```

The script consumes Stage121 group/setting summaries and Stage96 entropy-coded q6/top10 reference rows. It emits compressed deterministic value-only RD package rows for q4/top20, q4/top10, q5/top10, and q6/top10.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage122_compressed_deterministic_rd_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage122_compressed_deterministic_rd_package.py
```

## Outputs

```text
experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_rows.csv
experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_points.csv
experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_setting_summary.csv
experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_package.json
experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_report.md
```

## Configuration

| item | value |
|---|---:|
| row count | 24 |
| point count | 60 |
| entropy reference | Stage96 q6/top10 index+value side-info |
| primary | q4_top20 |
| low-rate | q4_top10 |
| near-anchor | q5_top10 |
| anchor | q6_top10 |

## Results

| role | setting | keep | bits | payload bytes | direct rate | amortized rate | PSNR | delta q6 | direct delta vs Stage96 entropy | PSNR delta vs Stage96 entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 4 | 28320.791666666668 | 0.1337680887378662 | 0.13005386354500142 | 20.689270746602087 | 0.9223020959187475 | -0.004194490114847849 | -0.8305321438068527 |
| low-rate | q4_top10 | 0.1 | 4 | 15190.475 | 0.12124604296658527 | 0.11924717736218463 | 19.73848817438193 | -0.028480476301425663 | -0.016716535886128772 | -1.781314716027025 |
| near-anchor | q5_top10 | 0.1 | 5 | 24809.95 | 0.13041988921139727 | 0.127149533351005 | 19.761047533309117 | -0.005921117374238853 | -0.007542689641316756 | -1.7587553570998395 |
| anchor | q6_top10 | 0.1 | 6 | 29442.208333333332 | 0.1348375550108561 | 0.13095624756787308 | 19.766968650683353 | 0.0 | -0.0031250238418579373 | -1.7528342397255992 |

## Conclusion

- q4/top20 is the primary RD package point: higher PSNR than q6/top10 and slightly lower rate.
- q4/top10 is the low-rate RD package point: much lower rate with small quality loss vs q6/top10.
- Stage96 entropy reference remains higher quality, so Stage122 is a lower-rate/lower-quality deterministic value-only side-info package.
- Residual values remain teacher-derived from dense target anchors; this is not residual value prediction.
- Stage123 should package the codec policy manifest.
