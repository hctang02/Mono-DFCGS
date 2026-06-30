# Stage178 Results And Innovation Summary

Date: 2026-06-30

## Goal

Create a dedicated handoff log that consolidates current Mono-DFCGS results, module-level innovation points, decoder contracts, non-claims, and the next validation route.

## Plan

- Summarize Stage158/161 middle-frame recovery evidence.
- Summarize Stage162/165/172/176/177 adaptive schedule evidence.
- Record module-level innovations in a paper-facing form.
- Keep decoder-side allowed/forbidden inputs explicit.
- Mark sampled/proxy limitations and non-claims clearly.

## Execution

- Checked `nvidia-smi` before the stage.
- Read Stage161, Stage162, Stage172, Stage176, Stage177 reports and the canonical current-status log.
- Created `logs/MONO_DFCGS_RESULTS_AND_INNOVATION_SUMMARY_2026-06-30.md`.

## Result

- Stage178 status: `results_and_innovation_summary_created`.
- Summary log path: `logs/MONO_DFCGS_RESULTS_AND_INNOVATION_SUMMARY_2026-06-30.md`.
- The log includes current best components, middle-frame evidence, adaptive schedule evidence, Stage177 fixed-gap comparison, innovation points, non-claims, evidence chain, next validation plan, and canonical paths.

## Key Recorded Results

| item | value |
|---|---|
| Stage158 gap4 PSNR / LPIPS | 29.780485 / 0.166020 |
| Stage158 gap8 PSNR / LPIPS | 29.578682 / 0.178535 |
| Stage165 adaptive keyframes | 358 / 1999 |
| Stage172 adaptive proxy rate | 0.194181515827 MiB/frame |
| Stage172 gap8 proxy rate | 0.300453182577 MiB/frame |
| Stage177 adaptive PSNR | 29.21709585981302 |
| Stage177 gap8 PSNR | 28.7569272347847 |
| Stage177 gap4 PSNR | 28.94581399778479 |
| Stage177 adaptive delta vs gap8 | +0.4601686250283185 dB |
| Stage177 adaptive delta vs gap4 | +0.27128186202823 dB |

## Decision

- Use the Stage178 summary as the canonical compact handoff for current results and innovation claims.
- Proceed to Stage179 broader sampled validation protocol before any final full-sequence RD claim.
