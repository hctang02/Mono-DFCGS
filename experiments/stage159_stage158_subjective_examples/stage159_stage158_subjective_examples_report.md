# Stage159 Stage158 Subjective Examples

## Videos

- Video: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage159_stage158_subjective_examples/stage159_gap4_stage158_subjective_examples.mp4`
- Contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage159_stage158_subjective_examples/stage159_gap4_stage158_subjective_examples_contact_sheet.jpg`
- Video file size: `518852` bytes
- Contact sheet file size: `1091636` bytes

## Rows

| sequence | frames | key avg PSNR/SSIM | middle PSNR/SSIM/MS-SSIM/LPIPS | original PSNR/LPIPS | delta PSNR/LPIPS | payload bytes | direct rate ref |
|---|---|---:|---:|---:|---:|---:|---:|
| car-shadow | 8-10-12 | 29.801/0.9260 | 29.405/0.9021/0.9866/0.1530 | 22.210/0.2650 | +7.196/-0.1120 | 220697 | 0.392411 |
| goat | 44-46-48 | 27.736/0.8860 | 27.287/0.8589/0.9870/0.1750 | 18.116/0.4181 | +9.171/-0.2431 | 251693 | 0.421971 |
| soapbox | 76-78-80 | 31.885/0.9420 | 30.962/0.9083/0.9884/0.1436 | 21.174/0.3203 | +9.787/-0.1767 | 246995 | 0.417491 |

## Layout

Each video frame is: left keyframe | target middle | original StreamSplat middle | Stage158 recovered middle | right keyframe.

## Contract

The Stage158 recovered middle panel uses original StreamSplat target-time half-anchor plus counted q6/keep1.0 entropy residual side-info and one counted half-selector byte.
