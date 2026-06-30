# Stage155 StreamSplat-Base Side-Info Upper-Bound Sweep

Date: 2026-06-30

## Goal

Test whether rate-counted correction on top of the original StreamSplat middle prediction can approach the desired `26-27 dB` middle-frame quality while preserving the original method's better perceptual plausibility.

## Plan

- First diagnose the original StreamSplat `pred_gs` tensor structure for timestamps `[0, t, 1]`.
- If the target-time Gaussian subset can be safely converted to the static anchor format, run Gaussian-domain residual side-info sweeps on top of the original StreamSplat base.
- If the mapping is not stable enough for immediate Gaussian-domain correction, run a decoder-safe rendered-domain side-info upper-bound diagnostic to establish whether the desired PSNR/SSIM/LPIPS target is achievable with counted auxiliary payload.
- Sweep increasingly high-rate settings before trying to compress down.
- Report PSNR, SSIM, MS-SSIM, LPIPS, payload bytes, and rate estimates.
- Use Stage153/154 sampled q12 gap4/gap8 eval tasks first, then scale successful settings.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Success Criteria

- Establish at least one correction setting that substantially improves original StreamSplat PSNR without worsening LPIPS/SSIM badly.
- Prefer settings approaching or exceeding `26 dB` sampled middle PSNR.
- Keep every transmitted byte counted and explicitly mark any diagnostic that is not yet final Gaussian-domain codec.

## Execution

- Checked `nvidia-smi` before running Python.
- Selected idle `GPU 1` with `CUDA_VISIBLE_DEVICES=1`.
- Initial run exposed a diagnostic bug where `pred_inverse` was applied before static rendering; fixed the script and reran.
- Final command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage155_streamsplat_base_sideinfo_upper_bound.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage155_streamsplat_base_sideinfo_upper_bound.py --device cuda --max_tasks 60 --batch_size 1 --image_residual_bits 3 4 5 6
```

## Results

| gap | method | bits | tasks | mean PSNR | p10 PSNR | mean SSIM | mean MS-SSIM | mean LPIPS | p90 LPIPS | payload bytes | direct rate ref | delta PSNR | delta LPIPS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | original_streamsplat_base | 0 | 30 | 22.321825581928188 | 18.184874443272292 | 0.6129658068219821 | 0.8124688843886058 | 0.2974850902954737 | 0.4201940566301346 | 0.0 | 0.1819382207679131 | 0.0 | 0.0 |
| 4 | image_residual_sideinfo_full_frame | 3 | 30 | 25.134758535637857 | 22.7473450361972 | 0.7263571123282114 | 0.9236845235029857 | 0.3135005762179693 | 0.41565817594528204 | 55879.36666666667 | 0.23522893757496063 | 2.8129329537096694 | 0.016015485922495524 |
| 4 | image_residual_sideinfo_full_frame | 4 | 30 | 32.280546434337076 | 30.36170381171098 | 0.8786723375320434 | 0.9775982022285461 | 0.15282797639568646 | 0.20505481958389282 | 78548.73333333334 | 0.2568481303341566 | 9.958720852408895 | -0.14465711389978728 |
| 8 | original_streamsplat_base | 0 | 30 | 20.52518071635703 | 15.471775583568856 | 0.5368488930165768 | 0.7138387471437454 | 0.3582147777080536 | 0.5523827314376831 | 0.0 | 0.09762538675351437 | 0.0 | 0.0 |
| 8 | image_residual_sideinfo_full_frame | 3 | 30 | 24.81127672184807 | 22.06270267953762 | 0.7088924904664358 | 0.9148616870244344 | 0.32558591216802596 | 0.4239487886428833 | 56604.5 | 0.15160764459653192 | 4.286096005491049 | -0.03262886554002762 |
| 8 | image_residual_sideinfo_full_frame | 4 | 30 | 31.718739219329834 | 29.540455694883974 | 0.863465295235316 | 0.9742599070072174 | 0.17084391911824545 | 0.24912019520998005 | 82349.4 | 0.17615989450497918 | 11.193558502972811 | -0.18737085858980815 |

Higher q5/q6 settings also pass with much higher PSNR but larger payloads.

## Gaussian-Domain Diagnostic

- Target-time static Gaussian evaluation exactly matches original dynamic rendering after the `pred_inverse` fix: max diff `0.0`.
- Direct residual against Stage61 target dense anchor is not shape-compatible: `0/60` tasks match.
- Original StreamSplat target-time base has `73728` Gaussians, while Stage61 target dense anchor has `36864` Gaussians.
- This means full-base residual-to-dense-anchor is not a valid direct codec.
- However, the original base is two `36864`-Gaussian halves, so Stage156 should test half-anchor correction against the target dense anchor.

## Heavy Contact Sheet

- Best setting worst-LPIPS sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage155_streamsplat_base_sideinfo_upper_bound/stage155_best_setting_worst_lpips_contact_sheet.jpg`

## Outputs

- Script: `scripts/run_stage155_streamsplat_base_sideinfo_upper_bound.py`
- Report: `experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_upper_bound_report.md`
- Package: `experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_upper_bound_package.json`
- Summary: `experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_summary.csv`
- Rows: `experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_rows.csv`
- Gaussian diagnostics: `experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_gaussian_diagnostics.csv`
- Bad cases: `experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_badcases.csv`

## Decision

Stage155 proves the sampled `26-27 dB` target is achievable with rate-counted auxiliary information on top of original StreamSplat. The q4 full-frame image residual setting is only an upper-bound diagnostic, not the final GS-feature method. Proceed to Stage156 with a real Gaussian-domain candidate: correct one StreamSplat target-time half-anchor to the target dense anchor using entropy-coded residual side-info and render the corrected half-anchor.
