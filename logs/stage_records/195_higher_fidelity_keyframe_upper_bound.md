# Stage195 Higher-Fidelity Keyframe Upper-Bound

Date: 2026-07-01

## Goal

After Stage194 showed all-frame q12 keyframes still miss the requested `+1 dB` full-sequence target, test whether higher-fidelity keyframe representations provide the missing headroom.

## Plan

- Reuse the Stage194 `uniform_gap1` all-frame protocol over `1999` DAVIS validation frames.
- Render every frame as a q16 dense-anchor keyframe and as a float dense-anchor keyframe.
- Measure q16 schedule-packed keyframe bitstreams for a rate reference.
- Treat float dense-anchor quality as a quality upper bound only, with no deployable rate claim.
- Compare q16 and float quality against Stage192 best fixed `uniform_gap2`.

## Success Criteria

- Complete `1999/1999` q16 quality rows and `1999/1999` float quality rows.
- Complete `30/30` q16 schedule-packed keyframe payload groups.
- Decide whether q16 or float dense-anchor keyframes can beat `uniform_gap2` by about `+1 dB` PSNR without SSIM/MS-SSIM/LPIPS regression.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage195_higher_fidelity_keyframe_upper_bound.py`
- Pre-smoke GPU check: `nvidia-smi` at 2026-07-01 17:51:49; GPU2 idle.
- Smoke command: `CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage195_higher_fidelity_keyframe_upper_bound.py --device cuda --max_schedule_keyframe_groups 1 --max_quality_rows 5 --flush_every 1`
- Pre-full GPU check: `nvidia-smi` at 2026-07-01 17:52:22; GPU2 idle.
- Full command: `CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage195_higher_fidelity_keyframe_upper_bound.py --device cuda --flush_every 200`

## Outputs

- Output root: `experiments/stage195_higher_fidelity_keyframe_upper_bound/`
- Package: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_higher_fidelity_keyframe_upper_bound_package.json`
- Report: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_higher_fidelity_keyframe_upper_bound_report.md`
- Summary: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_higher_fidelity_keyframe_summary.csv`
- Quality rows: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_higher_fidelity_keyframe_quality_metrics.csv`
- q16 schedule payload rows: `experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_q16_schedule_packed_keyframe_payload_measurements.csv`

## Results

Validation:

- Protocol frame count: `1999/1999`.
- q16 schedule-packed keyframe groups: `30/30`.
- q16 quality rows: `1999/1999`.
- float dense-anchor quality rows: `1999/1999`.

Higher-fidelity upper bounds:

| representation | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs gap2 | dLPIPS vs gap2 | +1dB/no-regression pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q16_keyframe` | `0.9146932160156617` | `29.884665362865746` | `0.8868697442192623` | `0.9886300584386145` | `0.13601533181894535` | `0.22985003409343818` | `-0.015665985772450486` | `0` |
| `float_dense_anchor` | `NA` | `29.88493146578025` | `0.8868824350291219` | `0.9886313845897806` | `0.13598978392716465` | `0.23011613700794342` | `-0.015691533664231178` | `0` |

Reference:

- Best Stage192 fixed gap: `uniform_gap2`, PSNR/SSIM/MS-SSIM/LPIPS `29.654815328772308/0.878375951948018/0.9866168332910943/0.15168131759139583`.
- q16 rate delta vs gap2: `+0.4651463338289934` MiB/frame.

## Decision

- Decision: `higher_fidelity_keyframes_improve_gap2_but_below_target_margin`.
- q16 improves gap2 by only `+0.22985003409343818` dB PSNR, far below `+1 dB` and at much higher rate.
- Float dense-anchor quality improves gap2 by only `+0.23011613700794342` dB PSNR, so the current dense-anchor/rendering representation itself lacks the requested headroom.
- A stronger selector, q16 quantization, or float keyframes alone cannot satisfy the user's desired full-sequence gain; next work must change the reconstruction objective/model or introduce a different counted correction payload with enough quality headroom.
