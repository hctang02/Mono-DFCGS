# Stage159 Stage158 Subjective Examples

Date: 2026-06-30

## User Request

- Record the Stage158 optimization explanation and selected gap4 sequence examples in the logs.
- Report the size/rate associated with those effects.
- Provide subjective visual video paths for the selected examples.

## Plan

- Treat Stage158 as the packaged policy `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Record that the middle-frame improvement comes from original StreamSplat target-time half-anchor plus counted Gaussian-domain entropy residual side-info.
- Use the previously inspected aligned gap4 exact-middle examples: `car-shadow`, `goat`, and `soapbox`.
- Report per-example residual payload bytes and direct rate reference from Stage157 rows.
- Check whether subjective videos already exist; if not, generate lightweight comparison videos/contact sheets outside git.
- Keep heavy visual outputs outside the repository.

## Initial Example Metrics

| sequence | frames L-M-R | key avg PSNR/SSIM | Stage158 middle PSNR/SSIM/MS-SSIM/LPIPS | original middle PSNR/LPIPS | delta PSNR/LPIPS | payload bytes |
|---|---|---:|---:|---:|---:|---:|
| car-shadow | 8-10-12 | 29.801/0.9260 | 29.405/0.9021/0.9866/0.1530 | 22.210/0.2650 | +7.196/-0.1120 | 220697 |
| goat | 44-46-48 | 27.736/0.8860 | 27.287/0.8589/0.9870/0.1750 | 18.116/0.4181 | +9.171/-0.2431 | 251693 |
| soapbox | 76-78-80 | 31.885/0.9420 | 30.962/0.9083/0.9884/0.1436 | 21.174/0.3203 | +9.787/-0.1767 | 246995 |

## Method Note

The Stage158 improvement is not from retraining. It uses original StreamSplat as the middle-frame motion/geometry base, selects one target-time half-anchor, transmits a counted q6/keep1.0 entropy-coded Gaussian residual plus a counted one-byte half selector, and renders the corrected half-anchor. The target dense anchor is only used encoder-side to construct the residual payload and is forbidden as decoder input.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage159_stage158_subjective_examples.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage159_stage158_subjective_examples.py --device cuda --fps 1
```

## Outputs

- Lightweight rows: `experiments/stage159_stage158_subjective_examples/stage159_stage158_subjective_examples_rows.csv`
- Summary JSON: `experiments/stage159_stage158_subjective_examples/stage159_stage158_subjective_examples_summary.json`
- Package JSON: `experiments/stage159_stage158_subjective_examples/stage159_stage158_subjective_examples_package.json`
- Report: `experiments/stage159_stage158_subjective_examples/stage159_stage158_subjective_examples_report.md`
- Heavy video: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage159_stage158_subjective_examples/stage159_gap4_stage158_subjective_examples.mp4`
- Heavy contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage159_stage158_subjective_examples/stage159_gap4_stage158_subjective_examples_contact_sheet.jpg`

## Result

- Video layout: `left keyframe | target middle | original StreamSplat middle | Stage158 recovered middle | right keyframe`.
- Video file size: `518852` bytes.
- Contact sheet file size: `1091636` bytes.
- LPIPS and MS-SSIM were available during export.
- Recomputed Stage159 metrics match the Stage157 rows for the selected examples.

| sequence | frames L-M-R | key avg PSNR/SSIM | Stage158 middle PSNR/SSIM/MS-SSIM/LPIPS | original middle PSNR/LPIPS | delta PSNR/LPIPS | payload bytes | side MiB/intermediate | direct rate ref MiB/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| car-shadow | 8-10-12 | 29.801/0.9260 | 29.405/0.9021/0.9866/0.1530 | 22.210/0.2650 | +7.196/-0.1120 | 220697 | 0.210473 | 0.392411 |
| goat | 44-46-48 | 27.736/0.8860 | 27.287/0.8589/0.9870/0.1750 | 18.116/0.4181 | +9.171/-0.2431 | 251693 | 0.240033 | 0.421971 |
| soapbox | 76-78-80 | 31.885/0.9420 | 30.962/0.9083/0.9884/0.1436 | 21.174/0.3203 | +9.787/-0.1767 | 246995 | 0.235553 | 0.417491 |

## Size Interpretation

- `payload bytes` is the transmitted Gaussian residual payload plus the counted one-byte half selector for that middle frame.
- `side MiB/intermediate` is `payload bytes / 1024^2`.
- `direct rate ref MiB/frame` adds the q12 main-anchor per-frame reference rate for gap4 to the side-info MiB/intermediate.
