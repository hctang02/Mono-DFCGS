# Stage172 Keyframe Rate Accounting Audit

Date: 2026-06-30

## Goal

Decompose the Stage165 adaptive schedule rate proxy into keyframe, extra-keyframe, residual side-info, schedule metadata, and total sampled proxy components.

## Plan

- Load Stage166 schedule comparison and sampled row consequences.
- Derive per-extra-keyframe main-anchor proxy cost from uniform gap8/gap4 keyframe counts and main-anchor proxy rates.
- Recompute total proxy rate as main-anchor proxy plus sampled residual side-info plus schedule metadata.
- Compare uniform gap8, Stage165 adaptive, and uniform gap4 with explicit component deltas.
- Mark limitations: this is sampled/proxy accounting, not final full-sequence RD.

## Success Criteria

- Stage172 package shows adaptive rate after charging extra keyframes and metadata.
- Component CSV separates keyframes, residual payload, metadata, and total proxy rate.
- Decision says whether adaptive remains promising enough for Stage173 medium protocol.

## Execution

- Checked `nvidia-smi` before running; GPU 2/3/5/6/7 were idle, though this stage is CPU-only.
- Compiled and ran `scripts/run_stage172_keyframe_rate_accounting_audit.py` with `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.

## Result

- Package: `experiments/stage172_keyframe_rate_accounting_audit/stage172_keyframe_rate_accounting_audit_package.json`.
- Report: `experiments/stage172_keyframe_rate_accounting_audit/stage172_keyframe_rate_accounting_audit_report.md`.
- Component CSV: `experiments/stage172_keyframe_rate_accounting_audit/stage172_rate_component_audit.csv`.
- Notes CSV: `experiments/stage172_keyframe_rate_accounting_audit/stage172_rate_accounting_notes.csv`.
- Decision: `adaptive_rate_promising_for_medium_protocol`.

## Key Accounting

| schedule | keyframes | extra vs gap8 | main anchor MiB/frame | residual MiB/sample | metadata MiB/frame | total proxy MiB/frame | delta vs gap8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 292 | 0 | 0.097625386754 | 0.202827795347 | 0.000000000477 | 0.300453182577 | 0.000000000000 |
| stage165_adaptive | 358 | 66 | 0.120431317266 | 0.073750042915 | 0.000000155646 | 0.194181515827 | -0.106271666750 |
| uniform_gap4 | 536 | 244 | 0.181938220768 | 0.188585289319 | 0.000000000477 | 0.370523510564 | 0.070070327987 |

## Notes

- Per-extra-keyframe proxy cost: `0.00034554440169835566` MiB/frame.
- Per-extra-keyframe dataset-level proxy payload: `0.690743258995013` MiB.
- Accounting scope remains `stage166_sampled_proxy`; full-sequence residual decisions are not yet evaluated.
- Decoder receives counted schedule/keyframe metadata and does not recompute RGB/motion selector features.

## Next Step

- Proceed to Stage173 medium rendered validation protocol using false-negative controls, positive promotions, high-payload controls, and easy/normal controls.
