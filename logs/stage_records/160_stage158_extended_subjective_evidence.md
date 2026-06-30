# Stage160 Stage158 Extended Subjective Evidence

Date: 2026-06-30

## Goal

Expand subjective evidence for the Stage158 recovered middle-frame policy without changing the method or optimizing rate.

## User Direction

- Treat the current Stage158 middle recovery as an innovation/quality-oriented component.
- Do not over-optimize bitrate now; a somewhat larger rate is acceptable if quality is good.
- Continue in stage order: first expand subjective evidence, then package the method, then start keyframe selector work.
- For future selector work, RGB/motion information may be allowed on the encoder side, but its source and feed-forward validity must be evaluated explicitly.

## Plan

- Reuse Stage158/Stage159 policy `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Select representative gap4 Stage157 rows from weak, hard, and strong sequences:
  - weak/low-PSNR: `cows`, `breakdance`, `camel`, `bike-packing`, `scooter-black`
  - perceptual/hard motion: `dance-twirl`, `motocross-jump`
  - reference/good visual examples: `soapbox`, `car-shadow`, `goat`, `gold-fish`, `kite-surf`
- Export a video and contact sheet with layout:
  - `left keyframe | target middle | original StreamSplat middle | Stage158 recovered middle | right keyframe`
- Store heavy video/contact sheet outside git under `/data/hctang/tmp/opencode/mono_dfcgs_runs/`.
- Store only lightweight rows/summary/report/package in git.
- Keep all payload bytes and selector bytes counted; do not use target dense anchor as decoder input.

## Success Criteria

- Video/contact sheet paths are produced for representative Stage158 examples.
- Lightweight report records per-example PSNR/SSIM/MS-SSIM/LPIPS, payload bytes, side MiB/intermediate, and direct total MiB/frame.
- Recomputed metrics match Stage157 evidence rows for selected examples.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage160_stage158_extended_subjective_evidence.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage160_stage158_extended_subjective_evidence.py --device cuda --fps 1
```

## Outputs

- Rows CSV: `experiments/stage160_stage158_extended_subjective_evidence/stage160_stage158_extended_subjective_evidence_rows.csv`
- Summary CSV: `experiments/stage160_stage158_extended_subjective_evidence/stage160_stage158_extended_subjective_evidence_summary.csv`
- Summary JSON: `experiments/stage160_stage158_extended_subjective_evidence/stage160_stage158_extended_subjective_evidence_summary.json`
- Package JSON: `experiments/stage160_stage158_extended_subjective_evidence/stage160_stage158_extended_subjective_evidence_package.json`
- Report: `experiments/stage160_stage158_extended_subjective_evidence/stage160_stage158_extended_subjective_evidence_report.md`
- Heavy video: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence.mp4`
- Heavy contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg`

## Result

- Exported `24` gap4 examples from `12` representative DAVIS sequences.
- Video layout: `left q12 keyframe render | target middle RGB | original StreamSplat middle | Stage158 recovered middle | right q12 keyframe render`.
- Video file size: `4180215` bytes.
- Contact sheet file size: `8739496` bytes.
- LPIPS and MS-SSIM were available during export.
- Recomputed Stage160 middle metrics match the Stage157 evidence rows for selected examples.

| sequence | tasks | key avg PSNR/LPIPS | Stage158 middle PSNR/LPIPS | original middle PSNR/LPIPS | delta PSNR/LPIPS | payload bytes | direct rate ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| bike-packing | 2 | 26.329/0.172648 | 25.916/0.200759 | 20.774/0.330460 | +5.142/-0.129701 | 173028 | 0.346951 |
| breakdance | 2 | 25.791/0.149225 | 25.629/0.170077 | 20.273/0.230913 | +5.355/-0.060835 | 123729 | 0.299935 |
| camel | 2 | 25.609/0.182846 | 25.581/0.187973 | 21.879/0.245850 | +3.702/-0.057876 | 228002 | 0.399378 |
| car-shadow | 2 | 29.794/0.127689 | 29.523/0.151694 | 23.489/0.248052 | +6.033/-0.096358 | 214037 | 0.386060 |
| cows | 2 | 24.715/0.205230 | 24.706/0.206553 | 22.211/0.252227 | +2.494/-0.045674 | 216943 | 0.388831 |
| dance-twirl | 2 | 27.554/0.204801 | 26.756/0.249167 | 18.425/0.389624 | +8.331/-0.140457 | 206272 | 0.378654 |
| goat | 2 | 27.469/0.169521 | 27.288/0.175273 | 17.199/0.406487 | +10.088/-0.231214 | 251509 | 0.421796 |
| gold-fish | 2 | 33.873/0.067136 | 33.412/0.079089 | 26.859/0.145390 | +6.554/-0.066301 | 191200 | 0.364281 |
| kite-surf | 2 | 32.446/0.094652 | 32.252/0.108727 | 25.022/0.203144 | +7.230/-0.094417 | 217840 | 0.389687 |
| motocross-jump | 2 | 34.905/0.075903 | 31.391/0.300439 | 13.738/0.635566 | +17.653/-0.335127 | 247438 | 0.417914 |
| scooter-black | 2 | 27.822/0.095169 | 27.337/0.138989 | 16.437/0.339682 | +10.900/-0.200693 | 246170 | 0.416704 |
| soapbox | 2 | 31.490/0.091349 | 30.907/0.125260 | 21.464/0.280982 | +9.443/-0.155722 | 241610 | 0.412356 |

## Decision

- Stage158 remains quality-oriented and visually inspectable; rate is explicitly reported but not optimized in Stage160.
- The next planned step is Stage161: package the Stage158 method narrative/evidence while preserving decoder contract and side-info accounting.
