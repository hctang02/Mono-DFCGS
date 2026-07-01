# Stage196 Target Feasibility Branch

Date: 2026-07-01

## Goal

Summarize Stage193-195 upper-bound evidence into a branch decision: whether the requested full-sequence target can be reached by selector/keyframe representation work, or whether the method must change payload/model/claim scope.

## Plan

- Read Stage192 best fixed-gap baseline, Stage193 oracle rows, Stage194 q12 all-keyframe upper bound, and Stage195 q16/float keyframe upper bounds.
- Compute the requested target PSNR as `best_fixed_gap2_psnr + 1.0`.
- Report each ceiling's remaining gap to the target.
- Package branch options for the next research direction.

## Success Criteria

- Produce a lightweight branch report with no new heavy artifacts.
- State whether selector/keyframe-quantization work should continue.
- List next viable directions and their claim risks.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage196_target_feasibility_branch.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-01 18:07:14; Stage196 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage196_target_feasibility_branch.py`

## Outputs

- Output root: `experiments/stage196_target_feasibility_branch/`
- Package: `experiments/stage196_target_feasibility_branch/stage196_target_feasibility_branch_package.json`
- Report: `experiments/stage196_target_feasibility_branch/stage196_target_feasibility_branch_report.md`
- Ceiling summary: `experiments/stage196_target_feasibility_branch/stage196_target_feasibility_summary.csv`
- Branch options: `experiments/stage196_target_feasibility_branch/stage196_branch_options.csv`

## Results

Requested target:

- Best fixed reference: `uniform_gap2`.
- Target PSNR: `30.654815328772308` (`uniform_gap2 + 1 dB`).

Ceilings:

| source | ceiling | PSNR | dPSNR vs gap2 | dPSNR vs target | pass |
|---:|---|---:|---:|---:|---:|
| 193 | `framewise_psnr_oracle` | `29.749038180432017` | `0.09422285165970834` | `-0.9057771483402917` | `0` |
| 193 | `schedule_path_psnr_oracle` | `29.670134277041633` | `0.015318948269325006` | `-0.984681051730675` | `0` |
| 194 | `all_q12_keyframes` | `29.85646819580043` | `0.20165286702812324` | `-0.7983471329718768` | `0` |
| 195 | `q16_keyframe` | `29.884665362865746` | `0.22985003409343818` | `-0.7701499659065618` | `0` |
| 195 | `float_dense_anchor` | `29.88493146578025` | `0.23011613700794342` | `-0.7698838629920566` | `0` |

Branch options:

- Stop selector/keyframe quantization tuning for the requested strong claim.
- Viable next diagnostic: counted RGB/image residual correction full-sequence upper bound, with the caveat that it is less Gaussian-native.
- Viable but heavy: new dense-anchor reconstruction objective/model.
- Writing fallback: adjust claim scope to sampled-target selector gains and measured middle RD point.

## Decision

- Decision: `selector_keyframe_representation_cannot_meet_target`.
- The current selector/keyframe representation branch is exhausted for the requested `+1 dB` full-sequence target.
- Continuing selector threshold tuning would not be scientifically honest because even float dense-anchor all-keyframes are `0.7698838629920566` dB below the target.
- The next step requires a research-direction choice: counted RGB/image residual correction, new reconstruction model/objective, or claim-scope adjustment.
