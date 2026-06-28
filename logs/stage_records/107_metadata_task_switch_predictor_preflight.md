# Stage107 Metadata Task-Level Switch Predictor Preflight

Date: 2026-06-28

## Goal

Test whether a deployable metadata-only task-level switch predictor can beat the Stage106 fixed group policy.

## Implementation

Added:

```text
scripts/run_stage107_metadata_task_switch_predictor_preflight.py
```

The model uses only decoder-side metadata features: base method, reference gap, normalized time, and frame indices. Rendered PSNR labels from Stage103 are used only for training/evaluation.

## Run

GPU check was performed before execution. Stage107 uses CPU and a small torch MLP; it does not load anchors or render. Syntax check:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage107_metadata_task_switch_predictor_preflight.py
```

Preflight run:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage107_metadata_task_switch_predictor_preflight.py
```

## Outputs

```text
experiments/stage107_metadata_task_switch_predictor_preflight/stage107_metadata_switch_rows.csv
experiments/stage107_metadata_task_switch_predictor_preflight/stage107_metadata_switch_summary.csv
experiments/stage107_metadata_task_switch_predictor_preflight/stage107_metadata_switch_group_summary.csv
experiments/stage107_metadata_task_switch_predictor_preflight/stage107_metadata_switch_train_log.csv
experiments/stage107_metadata_task_switch_predictor_preflight/stage107_metadata_switch_summary.json
experiments/stage107_metadata_task_switch_predictor_preflight/stage107_metadata_switch_report.md
```

## Results

| policy | selected PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |
|---|---:|---:|---:|---:|---:|---|
| endpoint_only | 20.316812710325653 | 0.0 | 20.403744235652297 | -0.08693152532665556 | 0.4583333333333333 | endpoint_diff_baseline:120 |
| global_train_best_policy | 20.316812710325653 | 0.0 | 20.403744235652297 | -0.08693152532665556 | 0.4583333333333333 | endpoint_diff_baseline:120 |
| metadata_mlp_cv | 20.314275971964204 | -0.002536738361441741 | 20.403744235652297 | -0.0894682636880973 | 0.39166666666666666 | endpoint_diff_baseline:48;shared_energy_regression:45;shared_topk_bce:27 |
| stage106_fixed_group_policy | 20.34687234717015 | 0.030059636844502392 | 20.403744235652297 | -0.05687188848215317 | 0.5416666666666666 | endpoint_diff_baseline:60;shared_energy_regression:60 |
| train_fold_group_policy | 20.33946471381412 | 0.02265200348847392 | 20.403744235652297 | -0.06427952183818163 | 0.5333333333333333 | endpoint_diff_baseline:57;shared_energy_regression:63 |
| oracle_task_best | 20.403744235652297 | 0.08693152532665556 | 20.403744235652297 | 0.0 | 1.0 | endpoint_diff_baseline:55;shared_energy_regression:44;shared_topk_bce:21 |

## Conclusion

- Metadata-only MLP does not beat endpoint-only or Stage106 fixed group policy.
- Stage106 fixed group policy remains the best deployable switch baseline from this line.
- The task-level oracle still has headroom, but reaching it likely needs decoder-side anchor statistics or selector-score statistics.
- Residual value prediction should still wait until selector switching is render-aligned.
