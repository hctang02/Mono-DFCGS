# Stage71 Baseline Availability Preflight

## Bottom Line

No local FCGS/D-FCGS/CWGS artifact is ready to enter the Stage70 DAVIS scoped RD table as an apples-to-apples baseline.
FCGS and D-FCGS code are present, but their input protocols do not directly match the current Mono-DFCGS DAVIS anchor/eval setup.

## Code Inventory

| Method | Code status | DAVIS mentions | Entrypoints | Missing tools |
|---|---|---:|---|---|
| FCGS | present | 0 | `encode_single_scene.py decode_single_scene.py decode_single_scene_validate.py` | `tmc3` |
| D-FCGS | present | 0 | `scripts/run_dfcgs_infer.py scripts/run_dfcgs_train.py train_fcgsd.py validate.py summerize.py` | `tmc3` |
| CWGS | missing_optional | 0 | `` | `code_checkout` |

## Artifact Inventory

| Group | Method | Records | DAVIS-related | Rate rows | Quality rows | Fair rows | Status |
|---|---|---:|---:|---:|---:|---:|---|
| stage52_fcgs_dfcgs_summary_records | FCGS/D-FCGS | 199 | 0 | 199 | 173 | 0 | old local diagnostic/reference only |
| stage53_external_baseline_rows | FCGS/D-FCGS | 199 | 0 | 199 | 173 | 0 | explicitly not apples-to-apples |
| stage70_baseline_status | FCGS/D-FCGS/CWGS | 3 | 3 | 0 | 0 | 0 | baseline status only |
| legacy_cwgs_rd_summaries | CWGS | 264 | 0 | 264 | 264 | 0 | old non-DAVIS reference only |

## Fairness Status

| Method | Ready | Input protocol | Rate | All-frame PSNR | Next action |
|---|---|---|---|---|---|
| FCGS | false | expects 3DGS .ply and Gaussian-Splatting Scene cameras; Stage61 DAVIS anchors are .pt tensors | FCGS bitstream bytes can be counted after a DAVIS wrapper exists; no DAVIS scoped bitstream MiB/frame yet | no DAVIS eval-subset all-frame PSNR table from local FCGS run | implement a DAVIS FCGS wrapper or anchor-to-ply conversion, run lmd sweep, decode/render same frames, and count actual bitstreams plus metadata |
| D-FCGS | false | expects multiview per-frame GS sequences; direct monocular DAVIS use would require a protocol-safe adapter | old GOP summaries include full I-frame plus P-frame codec rate; DAVIS scoped rate missing | no DAVIS eval-subset all-frame PSNR table from local D-FCGS run | decide whether to adapt D-FCGS to StreamSplat/DAVIS Gaussian sequences without multiview leakage or keep it as external-reference-only |
| CWGS | false | local code/protocol not inventoried; old artifacts are multisequence_rd, not DAVIS | old bitstream_mib exists but not DAVIS scoped or protocol-aligned | old psnr_avg exists for old references; no DAVIS scoped all-frame PSNR | only pursue after primary FCGS/D-FCGS baselines or if a local CWGS code checkout/protocol is provided |

## High-Priority Missing Items

| Method | Category | Requirement | Status |
|---|---|---|---|
| FCGS | frame_protocol | same DAVIS val eval sequences/frames/gaps as Stage70 | not yet run for any external baseline |
| FCGS | quality | all-frame PSNR against the same resized DAVIS RGB targets | missing |
| FCGS | rate | actual transmitted baseline bitstreams MiB/frame plus necessary metadata | missing for DAVIS scoped runs |
| D-FCGS | frame_protocol | same DAVIS val eval sequences/frames/gaps as Stage70 | not yet run for any external baseline |
| D-FCGS | quality | all-frame PSNR against the same resized DAVIS RGB targets | missing |
| D-FCGS | rate | actual transmitted baseline bitstreams MiB/frame plus necessary metadata | missing for DAVIS scoped runs |
| CWGS | frame_protocol | same DAVIS val eval sequences/frames/gaps as Stage70 | not yet run for any external baseline |
| CWGS | quality | all-frame PSNR against the same resized DAVIS RGB targets | missing |
| CWGS | rate | actual transmitted baseline bitstreams MiB/frame plus necessary metadata | missing for DAVIS scoped runs |
| FCGS | input_adapter | convert each DAVIS/StreamSplat keyframe Gaussian anchor or full frame GS to FCGS-compatible .ply without adding non-monocular side input | not implemented |
| FCGS | runner | batch encode/decode/validate lmd sweep over Stage70 scoped DAVIS sequences | not implemented |
| D-FCGS | input_adapter | construct a protocol-safe monocular DAVIS Gaussian sequence accepted by D-FCGS, or document why this baseline is incompatible | not implemented; upstream expects multiview/3DGStream-style sequences |
| D-FCGS | gop_policy | align I/P frame or GoF policy with Stage70 gaps 4/8/16 and count I-frame side rate | not implemented |

## Summary JSON

- `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_summary.json`
