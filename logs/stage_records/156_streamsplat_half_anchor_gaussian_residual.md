# Stage156 StreamSplat Half-Anchor Gaussian Residual Side-Info

Date: 2026-06-30

## Goal

Convert the Stage155 image-residual achievability result into a Gaussian-feature method by correcting one original StreamSplat target-time half-anchor with rate-counted entropy residual side-info.

## Plan

- Evaluate original StreamSplat target-time static Gaussian base and split it into left/right halves.
- Each half has `36864` Gaussians, matching the Stage61 target dense anchor shape.
- Use the target dense anchor only encoder-side to build residual payloads.
- Decode the residual payload into the selected half-anchor and render that corrected half-anchor.
- Sweep keep fraction and residual qbits.
- Evaluate left half, right half, and a `best_half_selector` policy that sends one extra byte of metadata per intermediate frame.
- Report PSNR, SSIM, MS-SSIM, LPIPS, payload bytes, and reference direct rate.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Success Criteria

- Find a Gaussian-domain side-info setting that approaches or exceeds `26 dB` sampled middle PSNR.
- The selected setting must improve SSIM/MS-SSIM/LPIPS over original StreamSplat base.
- All residual and selector bytes must be counted.

## Execution

- Checked `nvidia-smi` before running Python.
- Selected idle `GPU 1` with `CUDA_VISIBLE_DEVICES=1`.
- First run showed `keep=1.0/q4` reached PSNR but regressed LPIPS; updated best-setting criterion to require PSNR >= `26`, SSIM non-regression, and LPIPS non-regression.
- Final command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage156_streamsplat_half_anchor_gaussian_residual.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage156_streamsplat_half_anchor_gaussian_residual.py --device cuda --max_tasks 60 --batch_size 1 --keep_fractions 0.2 0.4 1.0 --side_bits 4 6
```

## Key Results

Original StreamSplat sampled baseline:

| gap | PSNR | SSIM | MS-SSIM | LPIPS |
|---:|---:|---:|---:|---:|
| 4 | 22.321825581928188 | 0.6129658068219821 | 0.8124688843886058 | 0.2974850902954737 |
| 8 | 20.52518071635703 | 0.5368488930165768 | 0.7138387471437454 | 0.3582147777080536 |

Best passing Gaussian-domain setting: `best_half_selector`, `keep_fraction=1.0`, `side_bits=6`, with one counted selector byte.

| gap | tasks | PSNR | p10 PSNR | SSIM | MS-SSIM | LPIPS | p90 LPIPS | payload bytes | direct rate ref | delta PSNR | delta LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 30 | 29.88060850586717 | 26.06568680730128 | 0.8795753101507823 | 0.9853506326675415 | 0.16458002875248592 | 0.2101434350013733 | 207591.66666666666 | 0.3799130615678806 | 7.55878292393898 | -0.1329050615429878 |
| 8 | 30 | 29.54743990416001 | 25.535196358023246 | 0.8700549105803171 | 0.9840767443180084 | 0.17757029533386232 | 0.23322499841451647 | 214067.13333333333 | 0.30177571380022644 | 9.022259187802984 | -0.18064448237419128 |

Lower-rate observation:

- `keep=1.0/q4` reaches gap4 PSNR `26.349792144483867` and gap8 PSNR `25.882061006724747`, but gap4 LPIPS regresses by `+0.022418021162350973`, so it is not the selected quality-safe setting.
- Partial keep settings are not enough for the desired `26-27 dB` target.

## Heavy Contact Sheet

- Best setting worst-LPIPS sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage156_streamsplat_half_anchor_gaussian_residual/stage156_best_half_selector_worst_lpips_contact_sheet.jpg`

## Outputs

- Script: `scripts/run_stage156_streamsplat_half_anchor_gaussian_residual.py`
- Report: `experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_gaussian_residual_report.md`
- Package: `experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_gaussian_residual_package.json`
- Summary: `experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_summary.csv`
- Rows: `experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_rows.csv`
- Bad cases: `experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_badcases.csv`

## Decision

Stage156 finds a sampled Gaussian-domain candidate that exceeds the requested `26-27 dB` target while improving SSIM/MS-SSIM/LPIPS over original StreamSplat. Proceed to Stage157 broader validation of the single selected setting `best_half_selector/keep1.0/q6` on the full Stage153/154 120-task sample before packaging it as the current recovered middle-frame policy.
