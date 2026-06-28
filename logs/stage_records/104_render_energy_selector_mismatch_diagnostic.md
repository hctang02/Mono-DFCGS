# Stage104 Render-Energy Selector Mismatch Diagnostic

Date: 2026-06-28

## Goal

Diagnose why learned selectors can improve residual-energy recall but fail to consistently improve rendered PSNR.

## Implementation

Added:

```text
scripts/run_stage104_render_energy_selector_mismatch_diagnostic.py
```

The script reads Stage103 per-task rendered rows and compares learned selectors against `endpoint_diff_baseline` without rerendering.

## Run

GPU check was performed before execution. Stage104 is a CPU summary script. Syntax check:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage104_render_energy_selector_mismatch_diagnostic.py
```

Diagnostic run:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage104_render_energy_selector_mismatch_diagnostic.py
```

## Outputs

```text
experiments/stage104_render_energy_selector_mismatch_diagnostic/stage104_render_energy_mismatch_rows.csv
experiments/stage104_render_energy_selector_mismatch_diagnostic/stage104_render_energy_mismatch_summary.csv
experiments/stage104_render_energy_selector_mismatch_diagnostic/stage104_render_energy_mismatch_summary.json
experiments/stage104_render_energy_selector_mismatch_diagnostic/stage104_render_energy_mismatch_report.md
```

## Results

| candidate | base | gap | energy delta | PSNR delta | energy up | PSNR up | energy-up PSNR-down |
|---|---|---:|---:|---:|---:|---:|---:|
| shared_energy_regression | linear | 4 | 0.03452943816133167 | 0.026827877460275387 | 22 | 15 | 7 |
| shared_energy_regression | linear | 8 | 0.045587826716272456 | 0.03086322039306108 | 19 | 10 | 9 |
| shared_energy_regression | linear | 16 | 0.043746052516831294 | 0.13353966957143293 | 17 | 14 | 3 |
| shared_energy_regression | stage65_adapter | 4 | 0.02671546327031177 | -0.15380783202852794 | 22 | 7 | 15 |
| shared_energy_regression | stage65_adapter | 8 | 0.03388543348563345 | -0.19199111803343827 | 18 | 7 | 11 |
| shared_energy_regression | stage65_adapter | 16 | 0.03499703254136774 | -0.016197329840340835 | 18 | 8 | 10 |
| shared_topk_bce | linear | 4 | 0.0336741439026335 | -0.08991556072924042 | 20 | 10 | 10 |
| shared_topk_bce | linear | 8 | 0.04330471942299291 | -0.10740330389783524 | 17 | 7 | 10 |
| shared_topk_bce | linear | 16 | 0.04484673134154744 | 0.03729370398841593 | 17 | 11 | 7 |
| shared_topk_bce | stage65_adapter | 4 | 0.02759676394255265 | -0.18816386153627007 | 22 | 6 | 16 |
| shared_topk_bce | stage65_adapter | 8 | 0.03471723159677104 | -0.19973996309924674 | 18 | 5 | 13 |
| shared_topk_bce | stage65_adapter | 16 | 0.033893570510877505 | -0.032410577247806396 | 18 | 8 | 10 |

## Conclusion

- Learned selectors consistently improve residual-energy recall over endpoint selection.
- Rendered PSNR does not consistently improve, especially for Stage65 adapter base.
- Energy-up PSNR-down mismatch is common, so residual-energy topk is not render-aligned enough.
- Next steps should introduce render-aware selector labels or task-level rendered ranking before residual value prediction.
