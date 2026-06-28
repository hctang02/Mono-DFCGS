# Stage108 Anchor-Stat Task-Level Switch Predictor Preflight

Date: 2026-06-28

## Goal

Test whether decoder-side anchor aggregate statistics can improve task-level switch prediction beyond metadata-only features.

## Implementation

Added:

```text
scripts/run_stage108_anchor_stat_task_switch_predictor_preflight.py
```

The script loads only left/right q12 anchors and predicted base anchors. It does not load target dense anchors or target residuals for features.

## Run

GPU check was performed before execution. GPU0 was busy, GPU6 had a process, and GPU2 was idle. Syntax check:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage108_anchor_stat_task_switch_predictor_preflight.py
```

Preflight run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage108_anchor_stat_task_switch_predictor_preflight.py
```

## Outputs

```text
experiments/stage108_anchor_stat_task_switch_predictor_preflight/stage108_anchor_stat_switch_rows.csv
experiments/stage108_anchor_stat_task_switch_predictor_preflight/stage108_anchor_stat_switch_summary.csv
experiments/stage108_anchor_stat_task_switch_predictor_preflight/stage108_anchor_stat_switch_group_summary.csv
experiments/stage108_anchor_stat_task_switch_predictor_preflight/stage108_anchor_stat_switch_train_log.csv
experiments/stage108_anchor_stat_task_switch_predictor_preflight/stage108_anchor_stat_switch_summary.json
experiments/stage108_anchor_stat_task_switch_predictor_preflight/stage108_anchor_stat_switch_report.md
```

## Results

| policy | selected PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |
|---|---:|---:|---:|---:|---:|---|
| endpoint_only | 20.316812710325653 | 0.0 | 20.403744235652297 | -0.08693152532665556 | 0.4583333333333333 | endpoint_diff_baseline:120 |
| metadata_mlp_cv | 20.31646985845114 | -0.00034285187450664443 | 20.403744235652297 | -0.0872743772011622 | 0.38333333333333336 | endpoint_diff_baseline:46;shared_energy_regression:44;shared_topk_bce:30 |
| anchor_stat_mlp_cv | 20.33017523703834 | 0.013362526712690951 | 20.403744235652297 | -0.07356899861396461 | 0.475 | endpoint_diff_baseline:51;shared_energy_regression:47;shared_topk_bce:22 |
| stage106_fixed_group_policy | 20.34687234717015 | 0.030059636844502392 | 20.403744235652297 | -0.05687188848215317 | 0.5416666666666666 | endpoint_diff_baseline:60;shared_energy_regression:60 |
| train_fold_group_policy | 20.33946471381412 | 0.02265200348847392 | 20.403744235652297 | -0.06427952183818163 | 0.5333333333333333 | endpoint_diff_baseline:57;shared_energy_regression:63 |
| oracle_task_best | 20.403744235652297 | 0.08693152532665556 | 20.403744235652297 | 0.0 | 1.0 | endpoint_diff_baseline:55;shared_energy_regression:44;shared_topk_bce:21 |

## Conclusion

- Anchor-stat features improve over metadata-only features and endpoint-only.
- Anchor-stat MLP still underperforms the Stage106 fixed group policy.
- Train accuracy reaches `1.0`, suggesting overfitting on the small 120-task rendered-label set.
- Stage106 remains the safest deployable switch baseline; richer selector-score features or more rendered labels are needed before replacing it.
