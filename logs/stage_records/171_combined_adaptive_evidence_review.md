# Stage171 Combined Adaptive Evidence Review

Date: 2026-06-30

## Goal

Review Stage166-170 evidence for the Stage165 RGB/motion adaptive keyframe schedule and decide whether to proceed to a stricter rate audit and medium rendered validation.

## Plan

- Load Stage165 selector preflight package, Stage166 label/RD proxy package, and Stage170 combined rendered validation package.
- Summarize selector coverage, sampled proxy rate, rendered false-negative behavior, positive promotions, and high-payload residual controls.
- Keep Stage158 recovery fixed; do not introduce a new codec or selector.
- Explicitly separate confirmed evidence from remaining risks.
- Output lightweight CSV/JSON/report in `experiments/stage171_combined_adaptive_evidence_review/`.

## Success Criteria

- Stage171 package states a clear continue/stop decision.
- Decision explains why the next step is rate accounting before larger rendered validation.
- No heavy media/checkpoint/anchor files are generated or committed.

## Execution

- Checked `nvidia-smi` before running; GPU 2/3/5/6/7 were idle, though this stage is CPU-only.
- Compiled and ran `scripts/run_stage171_combined_adaptive_evidence_review.py` with `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.

## Result

- Package: `experiments/stage171_combined_adaptive_evidence_review/stage171_combined_adaptive_evidence_review_package.json`.
- Report: `experiments/stage171_combined_adaptive_evidence_review/stage171_combined_adaptive_evidence_review_report.md`.
- Evidence CSV: `experiments/stage171_combined_adaptive_evidence_review/stage171_combined_adaptive_evidence_rows.csv`.
- Decision: `proceed_to_rate_audit_and_medium_protocol`.

## Key Evidence

- Selector coverage remains promising but imperfect: hard recall `0.733333333333`, payload recall `0.819444444444`.
- Adaptive schedule size is bounded between uniform gap8 and gap4: `358 / 292 / 536` keyframes.
- Stage166 proxy rate favors adaptive over uniform gap8: `0.194181515827` vs `0.300453182577` MiB/frame, but this is not final accounting.
- Stage170 false-negative residual rows are essentially unchanged vs uniform gap8: PSNR delta `-0.0109637522816`, LPIPS delta `+0.000909611582756`.
- Positive-promoted adaptive rows are keyframes/no-middle-render; no middle-render metrics are claimed for them.
- High-payload residual controls are small but positive: adaptive vs uniform gap8 PSNR delta `+0.315564515849`, LPIPS delta `-0.0170409046113`.

## Next Step

- Proceed to Stage172 keyframe/residual/metadata rate accounting before designing the Stage173 medium rendered validation protocol.
