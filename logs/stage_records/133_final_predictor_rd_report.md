# Stage133 Final Predictor RD Report

Date: 2026-06-29

## Goal

Generate final RD tables, plots, and report for the Stage122-132 predictor/side-info line.

## Plan

- Add a Stage133 final report script.
- Consume Stage130 comparison rows and Stage132 deployable policy.
- Emit final RD rows, package JSON, Markdown report, and RD plot.
- Clearly distinguish teacher side-info reference from deployable no-teacher predictor points.
- Check `nvidia-smi` before running Python, even though this package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage133_final_predictor_rd_report.py
```

The script consumes Stage130 comparison rows and the Stage132 deployable policy, then emits final RD rows, JSON package, Markdown report, and PNG/PDF RD plots.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage133_final_predictor_rd_report.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage133_final_predictor_rd_report.py
```

## Outputs

```text
experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_rows.csv
experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_package.json
experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_report.md
experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_plot.png
experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_plot.pdf
```

## Results

Final recommendation:

- Deployable policy: `deployable_adapter_delta_selected_residual_codec_v1`.
- Primary deployable setting: `q4_top20`.
- Optional low-rate setting: `q4_top10`.
- Teacher residual side-info is retained only as a quality reference.
- Dedicated MLP predictor is rejected until render-aware training fixes PSNR regression.

| role | method | setting | deployable | teacher | rate | PSNR | delta base | residual bytes | index bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| final_deployable_low_rate | adapter_delta_selected_predictor | q4_top10 | 1 | 0 | 0.11729838135687401 | 18.994813480380337 | 0.04401048394920189 | 0 | 0 |
| final_deployable_primary | adapter_delta_selected_predictor | q4_top20 | 1 | 0 | 0.11729838135687401 | 19.010259350474836 | 0.059456354043700026 | 0 | 0 |
| rejected_render_regression | dedicated_mlp_selected_predictor | q4_top10 | 1 | 0 | 0.11729838135687401 | 18.865777753557193 | -0.08502524287394495 | 0 | 0 |
| rejected_render_regression | dedicated_mlp_selected_predictor | q4_top20 | 1 | 0 | 0.11729838135687401 | 18.76520305064309 | -0.1855999457880447 | 0 | 0 |
| teacher_reference_only | teacher_compressed_sideinfo | q4_top10 | 0 | 1 | 0.12124604296658527 | 19.73848817438193 | 1.2544684336567946 | 15190.475 | 0 |
| teacher_reference_only | teacher_compressed_sideinfo | q4_top20 | 0 | 1 | 0.1337680887378662 | 20.689270746602087 | 2.2052510058769683 | 28320.791666666668 | 0 |

## Conclusion

- Stage133 completes the Stage125-133 predictor/side-info package line.
- The current final deployable predictor is adapter-delta q4/top20 with zero residual/index payload bytes.
- Teacher side-info remains the upper-quality but non-deployable reference.
- Dedicated MLP requires render-aware training before reconsideration.
