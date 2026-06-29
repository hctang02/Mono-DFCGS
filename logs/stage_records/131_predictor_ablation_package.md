# Stage131 Predictor Ablation Package

Date: 2026-06-29

## Goal

Summarize predictor ablations before choosing the Stage132 deployable policy.

## Plan

- Add a Stage131 ablation script.
- Compare adapter-delta predictor on 12-task and 60-task validations.
- Compare dedicated MLP residual-MSE improvement against rendered PSNR regression.
- Compare q4/top10 and q4/top20 keep fractions across teacher, adapter-delta, and MLP modes.
- Emit CSV/JSON/Markdown package.
- Check `nvidia-smi` before running Python, even though this package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage131_predictor_ablation_package.py
```

The script consumes Stage124, Stage125, Stage127, Stage129, and Stage130 outputs and emits an ablation package.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage131_predictor_ablation_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage131_predictor_ablation_package.py
```

## Outputs

```text
experiments/stage131_predictor_ablation_package/stage131_predictor_ablation_rows.csv
experiments/stage131_predictor_ablation_package/stage131_predictor_ablation_package.json
experiments/stage131_predictor_ablation_package/stage131_predictor_ablation_report.md
```

## Results

- Row count: `14`.
- Recommended deployable predictor: `adapter_delta_selected_predictor/q4_top20`.
- Rejected final predictor: `dedicated_mlp_selected_predictor_v1`.
- Rejection reason: MSE-trained MLP regresses rendered PSNR in Stage129.

Key takeaways:

- Adapter-delta selected predictor has small but stable positive rendered gain over linear base.
- Dedicated MLP reduces residual MSE but regresses rendered PSNR, so MSE labels are not enough.
- q4/top20 is the best no-teacher adapter-delta point.
- Teacher residual side-info remains a non-deployable upper-quality reference.

## Conclusion

Stage132 should package `adapter_delta_selected_predictor/q4_top20` as the current deployable predictor policy and keep teacher side-info as a reference, not as decoder-side deployment.
