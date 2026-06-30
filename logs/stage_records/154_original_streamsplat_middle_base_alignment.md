# Stage154 Original StreamSplat Middle Base Alignment

Date: 2026-06-30

## Goal

Evaluate original StreamSplat-guided middle-frame prediction as the primary base for future enhancement, because Stage153 showed that linear-base residual correction improves PSNR but is not visually reliable enough.

## Plan

- Reuse the original StreamSplat DAVIS inference utilities from Stage72.
- Evaluate original StreamSplat predictions on q12 gap4/gap8 eval middle tasks sampled consistently with Stage153 where possible.
- Compute PSNR, SSIM, MS-SSIM, and LPIPS.
- Compare original StreamSplat against Stage153 linear base and Stage151 recovered side-info rows when task IDs overlap.
- Export bad-case rankings and contact sheets outside git.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Success Criteria

- Original StreamSplat middle-base metrics are produced for gap4/gap8 tasks.
- Bad cases identify whether original StreamSplat is visually more plausible than linear-base recovery.
- The next Stage155 side-info upper-bound sweep can use this StreamSplat base rather than linear interpolation.

## Execution

- Checked `nvidia-smi` before running Python.
- Selected idle `GPU 1` with `CUDA_VISIBLE_DEVICES=1`.
- Command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage154_original_streamsplat_middle_base_alignment.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage154_original_streamsplat_middle_base_alignment.py --device cuda --max_tasks 120 --batch_size 1
```

## Results

| gap | method | tasks | mean PSNR | p10 PSNR | mean SSIM | mean MS-SSIM | mean LPIPS | p90 LPIPS | mean delta PSNR vs Stage151 | mean delta LPIPS vs Stage151 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | original_streamsplat_middle_base | 60 | 22.06421822428011 | 17.192961834877085 | 0.6008085365096728 | 0.8013669446110725 | 0.3014947975675265 | 0.4241745501756668 | -0.8312378934877117 | -0.04603325972954432 |
| 8 | original_streamsplat_middle_base | 60 | 20.33727549162514 | 14.745158801376014 | 0.5203309365858634 | 0.7005373592178027 | 0.3593370050191879 | 0.551969712972641 | -1.4725764599412898 | -0.02489723116159439 |

## Interpretation

- Original StreamSplat middle base has lower PSNR than Stage151 recovered side-info on the same sampled tasks.
- Original StreamSplat has better LPIPS than Stage151 on both gaps, especially gap4 (`-0.04603325972954432` lower LPIPS), which matches the user's visual observation that the original method is more semantically plausible.
- SSIM is roughly tied for gap4 and lower for gap8, so the next method should preserve StreamSplat's perceptual/motion prior while adding rate-counted correction for PSNR/structure.
- Stage155 should use original StreamSplat as the base for the upper-bound sweep instead of linear interpolation.

## Heavy Contact Sheets

- Highest original LPIPS: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_highest_original_lpips.jpg`
- Lowest original SSIM: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_lowest_original_ssim.jpg`
- Lowest original PSNR: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_lowest_original_psnr.jpg`
- Largest PSNR drop vs Stage151: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_largest_psnr_drop_vs_stage151.jpg`

## Outputs

- Script: `scripts/run_stage154_original_streamsplat_middle_base_alignment.py`
- Report: `experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_base_alignment_report.md`
- Package: `experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_base_alignment_package.json`
- Summary: `experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_summary.csv`
- Rows: `experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_rows.csv`
- Bad cases: `experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_badcases.csv`

## Decision

Proceed to Stage155 with original StreamSplat as the base. The goal is now to add rate-counted correction that closes the PSNR gap while keeping LPIPS/visual plausibility close to the original StreamSplat base.
