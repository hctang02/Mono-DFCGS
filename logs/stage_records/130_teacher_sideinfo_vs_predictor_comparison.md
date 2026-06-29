# Stage130 Teacher Side-Info Vs Predictor Comparison

Date: 2026-06-29

## Goal

Compare teacher residual side-info, adapter-delta no-teacher predictor, and dedicated MLP predictor points in one package.

## Plan

- Add a Stage130 comparison script.
- Consume Stage122 setting summary, Stage125 summary, and Stage129 summary.
- Emit comparison rows, package JSON, and report Markdown.
- Mark deployability and decoder input status explicitly.
- Check `nvidia-smi` before running Python, even though this package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage130_teacher_sideinfo_vs_predictor_comparison.py
```

The script compares Stage122 teacher compressed deterministic side-info, Stage125 adapter-delta selected predictor, and Stage129 dedicated MLP selected predictor.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage130_teacher_sideinfo_vs_predictor_comparison.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage130_teacher_sideinfo_vs_predictor_comparison.py
```

## Outputs

```text
experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_rows.csv
experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_package.json
experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_report.md
```

## Results

| stage | method | setting | deployable | teacher | rate | PSNR | delta base | residual bytes |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 122 | teacher_compressed_sideinfo | q4_top10 | 0 | 1 | 0.12124604296658527 | 19.73848817438193 | 1.2544684336567946 | 15190.475 |
| 125 | adapter_delta_selected_predictor | q4_top10 | 1 | 0 | 0.11729838135687401 | 18.994813480380337 | 0.04401048394920189 | 0 |
| 129 | dedicated_mlp_selected_predictor | q4_top10 | 1 | 0 | 0.11729838135687401 | 18.865777753557193 | -0.08502524287394495 | 0 |
| 122 | teacher_compressed_sideinfo | q4_top20 | 0 | 1 | 0.1337680887378662 | 20.689270746602087 | 2.2052510058769683 | 28320.791666666668 |
| 125 | adapter_delta_selected_predictor | q4_top20 | 1 | 0 | 0.11729838135687401 | 19.010259350474836 | 0.059456354043700026 | 0 |
| 129 | dedicated_mlp_selected_predictor | q4_top20 | 1 | 0 | 0.11729838135687401 | 18.76520305064309 | -0.1855999457880447 | 0 |

Best no-teacher deployable point in this comparison: Stage125 `adapter_delta_selected_predictor` q4/top20.

## Conclusion

- Teacher residual side-info remains far higher quality but requires target residual values.
- Adapter-delta selected predictor is the current best no-teacher point.
- Dedicated MLP is not render-safe yet and should not be selected for final deployment.
