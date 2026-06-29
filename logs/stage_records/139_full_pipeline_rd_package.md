# Stage139 Full-Pipeline RD Package

Date: 2026-06-29

## Goal

Package the full-pipeline rate accounting for the Stage138 render-aware deployable predictor policy.

## Plan

- Use Stage138 policy as the current deployable no-teacher predictor.
- Use Stage137 row-level rendered metrics for all/gap PSNR breakdown.
- Use Stage78 q12 anchor-only rate table for the main anchor stream rate.
- Report residual payload, selected-index payload, and policy-scale payload as zero per frame.
- Keep teacher residual side-info reference-only and outside the deployable pipeline.
- Do not rerender or write heavy tensors.
- Check `nvidia-smi` before running Python, even though the package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage139_full_pipeline_rd_package.py
```

The script packages all/gap full-pipeline RD accounting from Stage138 policy, Stage137 row-level metrics, and Stage78 q12 anchor rates.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage139_full_pipeline_rd_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage139_full_pipeline_rd_package.py
```

## Outputs

```text
experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_rows.csv
experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_summary.json
experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_package.json
experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_report.md
```

Output size: `20K`.

## Results

- Row count: `8`.
- Primary aggregate: q4/top20 scale `0.75`, rate `0.11729838135687401`, PSNR `19.022109503207204`, delta vs base `+0.07130650677606927`, delta vs Stage132 scale1 `+0.011850152732368002`.
- Low-rate aggregate: q4/top10 scale `0.75`, rate `0.11729838135687401`, PSNR `18.997890662360874`, delta vs base `+0.047087665929735116`, delta vs Stage132 scale1 `+0.0030771819805366363`.
- q12 gap rates validated against Stage78: gap4 `0.18193822076791313`, gap8 `0.09762538675351436`, gap16 `0.055468969746314975`.
- Residual payload MiB/frame: `0`.
- Selected-index payload MiB/frame: `0`.
- Policy-scale payload MiB/frame: `0`.
- Teacher side-info included: `0`.

## Gap Notes

- Primary q4/top20 scale `0.75` is positive vs Stage132 scale1 for gap4, gap8, and gap16.
- Low-rate q4/top10 scale `0.75` is aggregate-positive but gap16 is slightly negative vs Stage132 scale1 (`-0.0028379883519171756 dB`).

## Conclusion

Stage139 confirms the Stage138 policy has deployable full-pipeline accounting with q12 anchor rate only and no residual/index/scale side-info bytes. Stage140 should package multi-setting ablations and highlight the low-rate gap16 tradeoff.
