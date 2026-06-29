# Stage140 Multi-Setting Ablation Package

Date: 2026-06-29

## Goal

Package the final predictor ablations around Stage138/139 without rerendering.

## Plan

- Summarize Stage137 adapter-delta scale sweep across q4/top20 and q4/top10.
- Mark Stage138 q4/top20 scale `0.75` and q4/top10 scale `0.75` as selected deployable settings.
- Include linear base and full Stage65 adapter references from the same Stage137 metrics.
- Include Stage129/134 dedicated MLP as rejected due to rendered PSNR regression.
- Preserve no-teacher/deployable flags and zero side-info accounting for selected settings.
- Do not use teacher residual side-info as an optimization target.
- Check `nvidia-smi` before running Python, even though the package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage140_multi_setting_ablation_package.py
```

The script packages ablation rows from Stage137 scale sweep, Stage138 policy, Stage139 full-pipeline package, and Stage129/134 MLP regression artifacts.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage140_multi_setting_ablation_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage140_multi_setting_ablation_package.py
```

## Outputs

```text
experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_rows.csv
experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_summary.json
experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_package.json
experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_report.md
```

Output size: `20K`.

## Results

- Row count: `18`.
- Final primary: q4/top20 scale `0.75`, PSNR `19.022109503207204`, rate `0.11729838135687401`, delta vs Stage132 scale1 `+0.011850152732368002`.
- Final low-rate: q4/top10 scale `0.75`, PSNR `18.997890662360874`, rate `0.11729838135687401`, delta vs Stage132 scale1 `+0.0030771819805366363`.
- Linear reference: PSNR `18.950802996431143`.
- Full Stage65 adapter reference: PSNR `19.1876432931255`.
- Dedicated MLP rejected rows: q4/top10 PSNR `18.865777753557193`, q4/top20 PSNR `18.76520305064309`.

## Conclusion

Stage140 supports Stage138 as the final selected sparse residual codec policy and keeps the dedicated MLP rejected due to rendered PSNR regression. Stage141 should package the final deployable manifest.
