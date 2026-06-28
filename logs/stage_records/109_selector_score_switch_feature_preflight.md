# Stage109 Selector-Score Switch Feature Preflight

Date: 2026-06-28

## Goal

Test whether decoder-side selector score/logit statistics can improve task-level selector switching beyond the Stage106 fixed group policy.

## Implementation

Added:

```text
scripts/run_stage109_selector_score_switch_feature_preflight.py
```

The script trains the same shared selector objectives used by Stage103, builds task-level selector-score features, and evaluates switch predictors with deterministic 5-fold CV. Switch features use decoder-side metadata, left/right/base anchor statistics, and selector logits. Rendered PSNR labels from Stage103 are used only for train/eval labels, not as predictor inputs.

## Run

GPU check was performed before execution. GPU0 was busy, GPU1 had existing processes, and GPU2 was idle. Syntax check:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage109_selector_score_switch_feature_preflight.py
```

Full run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage109_selector_score_switch_feature_preflight.py
```

## Outputs

```text
experiments/stage109_selector_score_switch_feature_preflight/stage109_selector_score_switch_rows.csv
experiments/stage109_selector_score_switch_feature_preflight/stage109_selector_score_switch_summary.csv
experiments/stage109_selector_score_switch_feature_preflight/stage109_selector_score_switch_group_summary.csv
experiments/stage109_selector_score_switch_feature_preflight/stage109_selector_score_selector_train_log.csv
experiments/stage109_selector_score_switch_feature_preflight/stage109_selector_score_switch_train_log.csv
experiments/stage109_selector_score_switch_feature_preflight/stage109_selector_score_switch_summary.json
experiments/stage109_selector_score_switch_feature_preflight/stage109_selector_score_switch_report.md
```

Output size is about `316K`; no checkpoint, anchor tensor, payload, or heavy tensor is saved.

## Configuration

| item | value |
|---|---:|
| switch task rows | 120 |
| folds | 5 |
| selector train tasks | 96 |
| selector train examples | 589824 |
| anchor-stat feature dim | 64 |
| score-stat feature dim | 82 |
| anchor+score feature dim | 133 |

## Results

| policy | selected PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |
|---|---:|---:|---:|---:|---:|---|
| score_stat_mlp_cv | 20.32781855154445 | 0.01100584121880117 | 20.403744235652297 | -0.07592568410785439 | 0.5 | endpoint_diff_baseline:56;shared_energy_regression:46;shared_topk_bce:18 |
| anchor_score_mlp_cv | 20.328372726184103 | 0.01156001585845603 | 20.403744235652297 | -0.07537150946819952 | 0.4666666666666667 | endpoint_diff_baseline:52;shared_energy_regression:50;shared_topk_bce:18 |
| anchor_stat_mlp_cv | 20.33017523703834 | 0.013362526712690951 | 20.403744235652297 | -0.07356899861396461 | 0.475 | endpoint_diff_baseline:51;shared_energy_regression:47;shared_topk_bce:22 |
| stage106_fixed_group_policy | 20.34687234717015 | 0.030059636844502392 | 20.403744235652297 | -0.05687188848215317 | 0.5416666666666666 | endpoint_diff_baseline:60;shared_energy_regression:60 |
| oracle_task_best | 20.403744235652297 | 0.08693152532665556 | 20.403744235652297 | 0.0 | 1.0 | endpoint_diff_baseline:55;shared_energy_regression:44;shared_topk_bce:21 |

## Conclusion

- Selector-score statistics have positive signal over endpoint-only and metadata-only switching.
- Score-stat and anchor+score MLPs still underperform the Stage106 fixed group policy.
- Adding score statistics does not fix Stage65 adapter-group regressions.
- Stage106 remains the safest deployable switch baseline.
- Next step should be Stage110 broader rendered selector labels, not residual value prediction.
