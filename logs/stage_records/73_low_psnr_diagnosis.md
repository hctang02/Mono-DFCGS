# Stage73 Low-PSNR Diagnosis

Date: 2026-06-27

## Goal

Diagnose why Stage70 Gaussian-anchor-only DAVIS RD numbers are lower than the original StreamSplat full dynamic baseline.

This stage is the Phase B diagnostic follow-up to Stage72.

## Inputs

```text
experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv
experiments/stage70_scoped_davis_rd_package/stage70_all_psnr_table.csv
experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_rows.csv
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors
```

## Script

```text
scripts/run_stage73_low_psnr_diagnosis.py
```

## GPU Check

Before running code, `nvidia-smi` was checked. GPU1 was idle and used with:

```text
CUDA_VISIBLE_DEVICES=1
```

## Commands

Smoke:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage73_low_psnr_diagnosis.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage73_low_psnr_diagnosis.py --device cuda --sequences bmx-trees --gaps 16 --summary_root experiments/stage73_low_psnr_diagnosis_smoke
```

Full scoped diagnosis:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage73_low_psnr_diagnosis.py --device cuda
```

## Outputs

```text
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis_summary.json
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis.csv
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_gap_summary.csv
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis_report.md
```

Smoke outputs:

```text
experiments/stage73_low_psnr_diagnosis_smoke/
```

## Results

| gap | original all | float static adapter all | q8 static adapter all | q8 loss | original - float static | original given - float given | original given - q8 given |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23.244598036349643 | 21.347280410784887 | 20.608531857721726 | 0.7387485530631608 | 1.8973176255647592 | 0.020452617603702095 | 2.762040748859361 |
| 8 | 20.682715912446714 | 18.93647863322668 | 18.54248864942665 | 0.39398998380002936 | 1.7462372792200318 | 0.011830362977251596 | 2.74592976460412 |
| 16 | 18.353465837484507 | 17.244538002072254 | 17.01303254555753 | 0.23150545651472498 | 1.1089278354122554 | -0.026037018245909316 | 2.745552680459971 |

## Diagnosis

- Stage73 q8 adapter uniform exactly reproduces Stage70 adapter uniform all-frame PSNR.
- Float static keyframe anchors match original given-keyframe PSNR within about `0.03 dB` mean absolute scale across gaps.
- The low Stage70 all-frame PSNR is therefore not explained by RGB target resize, range, color conversion, frame index, static-anchor renderer bridge, or checkpoint loading bugs.
- The main quality gap comes from the reduced representation: Stage70 uses static anchors and a zero-dynamic wrapper, while original StreamSplat renders full dynamic `pred_gs`.
- q8 quantization is also aggressive for DAVIS keyframe anchors, reducing given-keyframe PSNR by roughly `2.75 dB` on average.

## Decision

Do not overwrite Stage70 as a bug-fixed result. Instead, mark Stage70 as a q8 static-anchor-only scoped lower-quality point and use Stage73 to motivate the next experiments:

- q-bit and per-field quantization sweep.
- Optional dynamic-field side information or better anchor representation.
- Continue training from original StreamSplat checkpoint or train a stronger rendered-label predictor.
- FCGS/D-FCGS fair DAVIS baselines before final comparison claims.

Phase A/B are complete. Stop for user review before starting more large-scale runs.
