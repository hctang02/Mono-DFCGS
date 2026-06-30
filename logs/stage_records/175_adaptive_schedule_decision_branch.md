# Stage175 Adaptive Schedule Decision Branch

Date: 2026-06-30

## Goal

Combine Stage172 rate accounting and Stage174 medium rendered validation to decide whether the Stage165 adaptive keyframe schedule should be packaged as a candidate, scaled to broader validation, or sent back for selector refinement first.

## Plan

- Load Stage172 rate-accounting package and Stage174 medium-validation package.
- Evaluate decision factors: rate proxy, protocol completeness, false negatives, positive promotions, residual controls, and selector false-positive keyframes.
- Keep adaptive keyframe rows as schedule/rate events, not middle-render metric rows.
- Output a clear decision, risks, and next action.

## Success Criteria

- Stage175 package states a concrete branch decision.
- Decision separates sampled evidence from final full-sequence RD claims.
- Risks are explicit enough for Stage176 packaging.

## Execution

- Checked `nvidia-smi` before running; Stage175 is CPU-only.
- Compiled and ran `scripts/run_stage175_adaptive_schedule_decision_branch.py` with `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.

## Result

- Package: `experiments/stage175_adaptive_schedule_decision_branch/stage175_adaptive_schedule_decision_branch_package.json`.
- Report: `experiments/stage175_adaptive_schedule_decision_branch/stage175_adaptive_schedule_decision_branch_report.md`.
- Factors CSV: `experiments/stage175_adaptive_schedule_decision_branch/stage175_decision_factors.csv`.
- Decision: `package_sampled_validated_candidate_and_scale_broader_validation`.

## Decision Factors

- Rate proxy passes: adaptive/gap8/gap4 total proxy MiB/frame `0.194181515827 / 0.300453182577 / 0.370523510564`.
- Stage174 completeness passes: `150/150` rows and `54/54` new renders.
- False negatives remain unresolved but neutral: adaptive delta vs gap8 PSNR `-0.0109637522816`, LPIPS `+0.000909611582756`.
- Positive promotions support adaptive schedule: positive extension uniform gap8 PSNR/LPIPS/payload `26.9100212409 / 0.211998779327 / 227130.875` and adaptive marks them keyframes/no-middle-render.
- Residual controls and normal controls match uniform gap8 when adaptive does not promote.
- Selector false-positive keyframes are a precision risk: uniform gap8 for false-positive controls is already PSNR/LPIPS/payload `30.0717550621 / 0.183046225458 / 157481.25`.

## Boundary

- This is not a final full-sequence RD claim.
- Adaptive keyframe rows remain schedule/rate events, not rendered middle metrics.

## Next Step

- Proceed to Stage176 candidate method package with limitations and broader-validation path.
