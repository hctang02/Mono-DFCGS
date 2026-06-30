# Stage158 Package Recovered Middle Policy

Date: 2026-06-30

## Goal

Freeze the Stage157 selected half-anchor Gaussian residual method as the current recovered middle-frame policy candidate.

## Plan

- Package policy `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Use Stage157 broader validation as the primary evidence.
- Include Stage153/154/155/156 context to explain why the method is StreamSplat-guided and GS-domain rather than linear/image-residual final.
- State decoder-allowed and decoder-forbidden inputs explicitly.
- Count residual payload bytes and half-selector metadata.
- Store only lightweight JSON/CSV/Markdown artifacts in git.

## Success Criteria

- Policy manifest records method contract, rate accounting, and validation evidence.
- Report states that the 120-task sampled protocol exceeds the requested `26-27 dB` target on gap4/gap8 and improves SSIM/MS-SSIM/LPIPS over original StreamSplat.

## Command

```bash
CUDA_VISIBLE_DEVICES=5 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage158_package_recovered_middle_policy.py && CUDA_VISIBLE_DEVICES=5 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage158_package_recovered_middle_policy.py
```

## Outputs

- `experiments/stage158_recovered_middle_policy_package/stage158_recovered_middle_policy.json`
- `experiments/stage158_recovered_middle_policy_package/stage158_recovered_middle_policy_package.json`
- `experiments/stage158_recovered_middle_policy_package/stage158_recovered_middle_policy_summary.json`
- `experiments/stage158_recovered_middle_policy_package/stage158_recovered_middle_policy_evidence.csv`
- `experiments/stage158_recovered_middle_policy_package/stage158_recovered_middle_policy_decisions.csv`
- `experiments/stage158_recovered_middle_policy_package/stage158_recovered_middle_policy_report.md`

## Result

- Packaged policy: `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Base: original StreamSplat target-time half-anchor.
- Correction: entropy-coded residual to encoder-side target dense anchor.
- Half policy: best half selector, counted as `1` byte/intermediate.
- Keep fraction: `1.0`.
- Side bits: `6`.
- Validation scope: 120 sampled q12 gap4/gap8 eval tasks from the Stage153/154 protocol.

| gap | tasks | PSNR | p10 PSNR | SSIM | MS-SSIM | LPIPS | payload bytes | direct rate ref | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 60 | 29.780485398070507 | 25.726690616240614 | 0.8779375642538071 | 0.9850881884495417 | 0.16601951060195763 | 209392.83333333334 | 0.3816307879574477 | true |
| 8 | 60 | 29.578682359235195 | 25.326197674163037 | 0.8696596751610438 | 0.9838472485542298 | 0.17853523269295693 | 215967.88333333333 | 0.3035884102571357 | true |

## Decision

- Stage158 passes the sampled quality gate: both gaps exceed the requested `26-27 dB` range.
- SSIM, MS-SSIM, and LPIPS all improve over original StreamSplat in the Stage157 broader validation rows.
- Stage155 image residual remains upper-bound only; Stage158 is the current Gaussian-domain recovered middle-frame policy package.
- Target dense anchor is allowed only encoder-side for residual construction and offline diagnostics, never as decoder input.
