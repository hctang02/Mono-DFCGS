# Stage111 Broader Switch Predictor

Date: 2026-06-28

## Goal

Train and evaluate broader task-level switch predictors on Stage110's 480 rendered switch rows, comparing learned predictors against Stage106 fixed group policy and the Stage110 broader group-best candidate.

## Implementation

Updated:

```text
scripts/run_stage109_selector_score_switch_feature_preflight.py
```

The runner now supports stage/output-prefix/report-title parameters and an optional fixed group choices CSV. Default Stage109 behavior remains unchanged.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage109_selector_score_switch_feature_preflight.py
```

Full run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage109_selector_score_switch_feature_preflight.py --stage 111 --mode "broader task-level switch predictor" --stage103_rows experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_rows.csv --fixed_group_choices experiments/stage110_broader_rendered_selector_labels/stage110_broader_render_aware_policy_group_choices.csv --fixed_group_policy_name stage110_group_best_policy --summary_root experiments/stage111_broader_switch_predictor --output_prefix stage111_broader_switch_predictor --report_title "Stage111 Broader Switch Predictor"
```

## Outputs

```text
experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_rows.csv
experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_summary.csv
experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_group_summary.csv
experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_selector_train_log.csv
experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_switch_train_log.csv
experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_summary.json
experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_report.md
```

Output size is about `1.2M`; no checkpoint, anchor tensor, payload, or heavy tensor is saved.

## Configuration

| item | value |
|---|---:|
| switch task rows | 480 |
| folds | 5 |
| selector train tasks | 96 |
| selector train examples | 589824 |
| anchor-stat feature dim | 64 |
| score-stat feature dim | 82 |
| anchor+score feature dim | 133 |

## Results

| policy | selected PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |
|---|---:|---:|---:|---:|---:|---|
| stage106_fixed_group_policy | 20.322996715243978 | 0.0017817297518578745 | 20.382843220952545 | -0.05984650570856027 | 0.51875 | endpoint_diff_baseline:240;shared_energy_regression:240 |
| stage110_group_best_policy | 20.32704687107235 | 0.005831885580240304 | 20.382843220952545 | -0.055796349880177856 | 0.55625 | endpoint_diff_baseline:323;shared_energy_regression:157 |
| anchor_stat_mlp_cv | 20.331745776419964 | 0.010530790927825506 | 20.382843220952545 | -0.051097444532592656 | 0.6041666666666666 | endpoint_diff_baseline:261;shared_energy_regression:134;shared_topk_bce:85 |
| score_stat_mlp_cv | 20.33325220653739 | 0.012037221045259486 | 20.382843220952545 | -0.04959101441515865 | 0.6041666666666666 | endpoint_diff_baseline:274;shared_energy_regression:130;shared_topk_bce:76 |
| anchor_score_mlp_cv | 20.33140381862592 | 0.010188833133790712 | 20.382843220952545 | -0.05143940232662742 | 0.59375 | endpoint_diff_baseline:270;shared_energy_regression:122;shared_topk_bce:88 |
| oracle_task_best | 20.382843220952545 | 0.06162823546041816 | 20.382843220952545 | 0.0 | 1.0 | endpoint_diff_baseline:278;shared_energy_regression:121;shared_topk_bce:81 |

## Best Learned Group Behavior

`score_stat_mlp_cv` is best overall, but it still has an adapter regression:

| base | gap | gain vs endpoint |
|---|---:|---:|
| linear | 4 | 0.005069813517757055 |
| linear | 8 | 0.025965933345715158 |
| linear | 16 | 0.02894712770300507 |
| stage65_adapter | 4 | -0.00797889356792674 |
| stage65_adapter | 8 | 0.003851082004932165 |
| stage65_adapter | 16 | 0.018024353491706453 |

## Conclusion

- Broader labels help: learned switch predictors now beat Stage106 and Stage110 fixed group policies overall.
- `score_stat_mlp_cv` is the best overall policy, but it still regresses Stage65 adapter gap4.
- Because the current safety condition requires no adapter-group regression, learned switch should not be packaged yet.
- Stage112 should package the conservative Stage110 broader group policy candidate instead of the learned MLP, then Stage113 should validate it on held-out rows/sequences.
