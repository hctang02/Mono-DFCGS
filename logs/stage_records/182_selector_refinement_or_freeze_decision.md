# Stage182 Selector Refinement Or Freeze Decision

Date: 2026-06-30

## Goal

Decide whether the Stage165 adaptive keyframe selector should be frozen as the current candidate or refined before the next validation step.

## Plan

- Review Stage180 broader sampled quality deltas by category.
- Review Stage181 rate accounting preflight.
- Weigh false-negative and false-positive risks against observed quality/rate evidence.
- Produce a branch decision with clear next work.

## Success Criteria

- Decision is explicit: freeze current candidate, tune selector first, or run full-sequence payload measurement next.
- Evidence table includes quality, rate, false-negative, false-positive, decoder contract, and non-claim status.

## Execution

- Checked `nvidia-smi` before Stage182.
- Compiled and ran `scripts/run_stage182_selector_refinement_or_freeze_decision.py`.
- Inputs: Stage176 candidate package, Stage180 broader validation package/category deltas, and Stage181 RD accounting preflight package.

## Result

- Stage182 status: `selector_refinement_or_freeze_decision_packaged`.
- Decision: `freeze_current_candidate_and_run_full_sequence_payload_measurement_next`.
- Frozen policy: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate`.
- Next required stage: `full_sequence_payload_measurement`.

## Key Evidence

| area | value | source |
|---|---|---|
| quality delta vs gap8 / gap4 | `+0.5644261202320328 / +0.306535729521994` dB | Stage180 |
| LPIPS delta vs gap8 / gap4 | `-0.0338725696835253 / -0.019677375422583687` | Stage180 |
| adaptive/gap8/gap4 total proxy | `0.1916456087572328 / 0.3073179395851907 / 0.3688664299140155` MiB/frame | Stage181 |
| broader positive promoted delta vs gap8 / gap4 | `+0.829806900266485 / +0.4439986578568402` dB | Stage180 |
| broader weak sequence delta vs gap8 / gap4 | `+1.028785745355083 / +0.5638605351918257` dB | Stage180 |
| false-negative residual delta vs gap8 / gap4 | `-0.010963752281610173 / -0.32629360438858646` dB | Stage180 |
| false-positive keyframe control delta vs gap8 / gap4 | `+0.39611419615648114 / +0.2796445234109868` dB | Stage180 |

## Interpretation

- Current evidence supports freezing Stage165 adaptive for the next measurement rather than tuning threshold/min-votes immediately.
- Remaining risks are mainly final-RD measurement risks: actual q12 keyframe bitstreams and all-frame Stage158 residual payloads still need measurement.
- False negatives remain a known risk and should stay in stress categories.
- False-positive keyframe cost should be revisited only if exact full payload measurement shows rate regression.

## Outputs

- Package: `experiments/stage182_selector_refinement_or_freeze_decision/stage182_selector_refinement_or_freeze_decision_package.json`
- Report: `experiments/stage182_selector_refinement_or_freeze_decision/stage182_selector_refinement_or_freeze_decision_report.md`
- Evidence CSV: `experiments/stage182_selector_refinement_or_freeze_decision/stage182_freeze_decision_evidence.csv`
- Risks CSV: `experiments/stage182_selector_refinement_or_freeze_decision/stage182_freeze_decision_risks.csv`
- Next steps CSV: `experiments/stage182_selector_refinement_or_freeze_decision/stage182_freeze_decision_next_steps.csv`
