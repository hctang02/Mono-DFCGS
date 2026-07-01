# Stage193 Oracle Upper-Bound Analysis

Date: 2026-07-01

## Goal

Determine whether the current Stage158 recovery representation and measured fixed-gap/adaptive candidate space has enough headroom to beat the best expanded fixed-gap baseline by about `+1 dB` PSNR on full-sequence metrics.

## Plan

- Use Stage192 measured full-sequence rows for `gap2/gap4/gap6/gap8/gap16/stage165_adaptive`.
- Build a framewise PSNR oracle that picks the highest-PSNR measured schedule output per frame. This is an optimistic non-schedule-consistent upper bound.
- Build a schedule-consistent path oracle over measured keyframe nodes and measured Stage158 residual edges. This estimates what a perfect selector could do using the measured transition set.
- Compare oracle PSNR/SSIM/MS-SSIM/LPIPS against the best fixed gap from Stage192.

## Success Criteria

- Report whether the `+1 dB` target over the best fixed gap is achievable even under optimistic oracle assumptions.
- If not achievable, stop treating selector tuning alone as sufficient and identify representation/payload-policy bottleneck.
- Outputs are lightweight CSV/JSON/Markdown only.

## Execution

- Pre-run GPU check: `nvidia-smi` recorded on 2026-07-01 17:19:34; Stage193 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage193_oracle_upper_bound_analysis.py`
- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage193_oracle_upper_bound_analysis.py`

## Outputs

- Package: `experiments/stage193_oracle_upper_bound_analysis/stage193_oracle_upper_bound_analysis_package.json`
- Report: `experiments/stage193_oracle_upper_bound_analysis/stage193_oracle_upper_bound_analysis_report.md`
- Summary: `experiments/stage193_oracle_upper_bound_analysis/stage193_oracle_summary.csv`
- Framewise oracle rows: `experiments/stage193_oracle_upper_bound_analysis/stage193_framewise_psnr_oracle_rows.csv`
- Schedule path oracle rows: `experiments/stage193_oracle_upper_bound_analysis/stage193_schedule_path_psnr_oracle_rows.csv`
- Sequence summary: `experiments/stage193_oracle_upper_bound_analysis/stage193_oracle_sequence_summary.csv`

## Results

Best fixed gap from Stage192:

- `uniform_gap2`, PSNR `29.654815328772308`.

Oracle upper bounds:

| oracle | schedule-consistent | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs best fixed | dLPIPS vs best fixed | +1dB/no-regression pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `framewise_psnr_oracle` | `0` | `29.749038180432017` | `0.8816605037662493` | `0.9870865034603846` | `0.14641779118318626` | `0.09422285165970834` | `-0.005263526408209568` | `0` |
| `schedule_path_psnr_oracle` | `1` | `29.670134277041633` | `0.8788577944949724` | `0.9867078401912386` | `0.1508482470754357` | `0.015318948269325006` | `-0.000833070515960127` | `0` |

## Decision

- Decision: `framewise_oracle_upper_bound_below_target_margin`.
- Even the optimistic non-schedule-consistent oracle only improves best fixed gap by `+0.09422285165970834` dB PSNR, far below the requested `+1 dB`.
- The schedule-consistent oracle improves best fixed by only `+0.015318948269325006` dB PSNR.
- Tuning the current selector over the current measured Stage158/fixed-gap candidate space cannot satisfy the requested strong full-sequence claim.
- Next diagnostic should test a stronger representation upper bound, e.g. all-frame q12 keyframes (`gap1`), to verify whether the keyframe/recovery representation itself can reach the target.
