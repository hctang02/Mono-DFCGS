# Stage194 All-Keyframe Q12 Upper-Bound

Date: 2026-07-01

## Goal

Test whether a stronger all-keyframe q12 representation (`uniform_gap1`) has enough full-sequence headroom to beat the best Stage192 fixed-gap baseline by about `+1 dB` PSNR without SSIM/MS-SSIM/LPIPS regression.

## Plan

- Build an all-frame q12 keyframe protocol over the same 30 DAVIS validation sequences / `1999` frames used in Stage192.
- Seed already measured q12 keyframe payload/quality rows from Stage192.
- Measure missing q12 keyframe payload/quality rows only; no Stage158 residual rows are needed.
- Measure schedule-packed all-frame q12 keyframe bitstreams per sequence for rate consistency with Stage192.
- Compare `uniform_gap1` against Stage192 best fixed `uniform_gap2`.

## Success Criteria

- Complete all `1999` q12 keyframe quality rows and all `30` schedule-packed keyframe groups.
- Report full-sequence PSNR, SSIM, MS-SSIM, LPIPS, and measured schedule-packed MiB/frame.
- Decide whether q12 all-keyframes pass the requested `+1 dB` / no-regression target over `uniform_gap2`.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage194_all_keyframe_q12_upper_bound.py`
- Pre-smoke GPU check: `nvidia-smi` at 2026-07-01 17:37:04; GPU2 idle.
- Smoke command: `CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage194_all_keyframe_q12_upper_bound.py --device cuda --max_payload_keyframes 5 --max_schedule_keyframe_groups 1 --max_quality_keyframes 5 --flush_every 1`
- Pre-full GPU check: `nvidia-smi` at 2026-07-01 17:38:10; GPU2 idle.
- Full command: `CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage194_all_keyframe_q12_upper_bound.py --device cuda --flush_every 200`

## Outputs

- Output root: `experiments/stage194_all_keyframe_q12_upper_bound/`
- Package: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_all_keyframe_q12_upper_bound_package.json`
- Report: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_all_keyframe_q12_upper_bound_report.md`
- Summary: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_all_keyframe_q12_summary.csv`
- Final quality rows: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_all_keyframe_quality_rows.csv`
- Unique keyframe payload rows: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_unique_keyframe_payload_measurements.csv`
- Unique keyframe quality rows: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_unique_keyframe_quality_metrics.csv`
- Schedule-packed keyframe rows: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_schedule_packed_keyframe_payload_measurements.csv`

## Results

Validation:

- Protocol frame count: `1999/1999`.
- Unique keyframe payload rows: `1999/1999`.
- Unique keyframe quality rows: `1999/1999`.
- Schedule-packed keyframe groups: `30/30`.

All-keyframe q12 RD-quality:

| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs gap2 | dLPIPS vs gap2 | +1dB/no-regression pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform_gap1` | `0.686173294471943` | `1999` | `29.85646819580043` | `0.8857439302277005` | `0.9884901848240099` | `0.13756442577459324` | `0.20165286702812324` | `-0.01411689181680259` | `0` |

Reference:

- Best Stage192 fixed gap: `uniform_gap2`, PSNR/SSIM/MS-SSIM/LPIPS `29.654815328772308/0.878375951948018/0.9866168332910943/0.15168131759139583`.
- Stage194 rate delta vs gap2: `+0.2366264122852747` MiB/frame.

## Decision

- Decision: `all_keyframe_q12_improves_gap2_but_below_target_margin`.
- All-frame q12 keyframes improve gap2 by only `+0.20165286702812324` dB PSNR, far below the requested `+1 dB` margin.
- Since Stage193 showed current selector candidates lack headroom and Stage194 shows even all q12 keyframes lack target headroom, further selector-threshold tuning is not sufficient.
- Next diagnostic should test whether higher-fidelity keyframe representation, e.g. q16 or float dense anchors, can provide the missing `+1 dB` headroom before any new adaptive schedule design.
