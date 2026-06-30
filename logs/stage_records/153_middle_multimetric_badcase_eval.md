# Stage153 Middle Multi-Metric Bad-Case Evaluation

Date: 2026-06-30

## Goal

Build a multi-metric evaluation and subjective bad-case package for middle-frame prediction quality, because PSNR alone hid visually broken outputs in Stage152.

## Plan

- Evaluate middle-frame predictions with PSNR, SSIM, MS-SSIM, and LPIPS where available.
- Compare at least the current Stage151 recovered policy against its linear base on sampled q12 gap4/gap8 eval tasks.
- Rank worst cases by LPIPS, SSIM, and PSNR to expose visually broken sequences.
- Export lightweight CSV/JSON/Markdown reports in `experiments/stage153_middle_multimetric_badcase_eval/`.
- Store any contact sheets or videos outside git if they become large.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Success Criteria

- Multi-metric rows are produced for gap4/gap8 middle frames.
- Bad-case tables identify sequences/tasks that need visual sanity improvements.
- The report explicitly states whether the current recovered policy is visually safe enough to continue, or whether the next stage must use original StreamSplat-guided base instead of linear base.

## Execution

- Checked `nvidia-smi` before running Python.
- Selected idle `GPU 2` with `CUDA_VISIBLE_DEVICES=2`.
- Command:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage153_middle_multimetric_badcase_eval.py && CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage153_middle_multimetric_badcase_eval.py --device cuda --max_tasks 120 --disable_cache
```

## Results

| gap | method | tasks | mean PSNR | p10 PSNR | mean SSIM | mean MS-SSIM | mean LPIPS | p90 LPIPS | payload bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | linear_base | 60 | 19.77900530163776 | 14.861659071195339 | 0.4726536904772123 | 0.6174249157309533 | 0.36224671031037964 | 0.511952155828476 | 0.0 |
| 4 | stage151_recovered_linear_base_sideinfo | 60 | 22.895456117767825 | 17.6536168157772 | 0.6020039414366086 | 0.7726786529024442 | 0.3475280572970708 | 0.5083782434463501 | 29908.516666666666 |
| 8 | linear_base | 60 | 18.526649919631087 | 14.04183779315019 | 0.43580265579124294 | 0.5489159627507131 | 0.4003311740855376 | 0.5667300224304199 | 0.0 |
| 8 | stage151_recovered_linear_base_sideinfo | 60 | 21.809851951566433 | 17.0729545199017 | 0.5636065999666849 | 0.7226341560482978 | 0.38423423618078234 | 0.5388027191162109 | 30091.616666666665 |

## Bad Cases

- Highest recovered LPIPS includes `motocross-jump`, `bmx-trees`, `shooting`, `libby`, `loading`, and `soapbox`.
- Lowest recovered SSIM includes `goat`, `bmx-trees`, `drift-straight`, `cows`, `parkour`, and `libby`.
- Lowest recovered PSNR includes `motocross-jump`, `scooter-black`, `drift-straight`, `bmx-trees`, `shooting`, and `loading`.
- LPIPS regressions exist even when PSNR increases, for example `soapbox`, `india`, `drift-straight`, and `camel`.

## Heavy Contact Sheets

- Highest recovered LPIPS: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_highest_recovered_lpips.jpg`
- Lowest recovered SSIM: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_lowest_recovered_ssim.jpg`
- Lowest recovered PSNR: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_lowest_recovered_psnr.jpg`
- Largest LPIPS regression: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_largest_lpips_regression.jpg`

## Outputs

- Script: `scripts/run_stage153_middle_multimetric_badcase_eval.py`
- Report: `experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_badcase_eval_report.md`
- Package: `experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_badcase_eval_package.json`
- Summary: `experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_summary.csv`
- Per-method rows: `experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_rows.csv`
- Pair rows: `experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_pair_rows.csv`
- Bad cases: `experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_badcases.csv`

## Decision

Stage151 is useful as a rate-counted PSNR recovery reference, but it is not sufficient as the final subjective-quality method. The next stages must use original StreamSplat-guided middle prediction as the base and optimize residual correction around that base, with PSNR/SSIM/MS-SSIM/LPIPS and bad-case visual checks as gating criteria.
