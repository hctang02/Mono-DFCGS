# Stage107 Metadata Task-Level Switch Predictor Preflight

## Configuration

- task rows: `120`
- folds: `5`
- features: `method_id, is_linear, gap_norm, is_gap4, is_gap8, is_gap16, normalized_time, distance_to_mid_time, left_index_norm, right_index_norm, target_index_norm, target_relative_position, right_relative_position`
- labels are per-task best rendered deployable candidate from Stage103 rows
- no anchors, rendering, checkpoints, or heavy tensor output

## Overall Summary

| policy | tasks | selected PSNR | endpoint PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| endpoint_only | 120 | 20.316812710325653 | 20.316812710325653 | 0.0 | 20.403744235652297 | -0.08693152532665556 | 0.4583333333333333 | endpoint_diff_baseline:120 |
| global_train_best_policy | 120 | 20.316812710325653 | 20.316812710325653 | 0.0 | 20.403744235652297 | -0.08693152532665556 | 0.4583333333333333 | endpoint_diff_baseline:120 |
| metadata_mlp_cv | 120 | 20.314275971964204 | 20.316812710325653 | -0.002536738361441741 | 20.403744235652297 | -0.0894682636880973 | 0.39166666666666666 | endpoint_diff_baseline:48;shared_energy_regression:45;shared_topk_bce:27 |
| oracle_task_best | 120 | 20.403744235652297 | 20.316812710325653 | 0.08693152532665556 | 20.403744235652297 | 0.0 | 1.0 | endpoint_diff_baseline:55;shared_energy_regression:44;shared_topk_bce:21 |
| stage106_fixed_group_policy | 120 | 20.34687234717015 | 20.316812710325653 | 0.030059636844502392 | 20.403744235652297 | -0.05687188848215317 | 0.5416666666666666 | endpoint_diff_baseline:60;shared_energy_regression:60 |
| train_fold_group_policy | 120 | 20.33946471381412 | 20.316812710325653 | 0.02265200348847392 | 20.403744235652297 | -0.06427952183818163 | 0.5333333333333333 | endpoint_diff_baseline:57;shared_energy_regression:63 |

## Group Summary

| policy | base | gap | tasks | selected PSNR | gain vs endpoint | accuracy | selections |
|---|---|---:|---:|---:|---:|---:|---|
| endpoint_only | linear | 4 | 23 | 20.95034838676545 | 0.0 | 0.34782608695652173 | endpoint_diff_baseline:23 |
| endpoint_only | linear | 8 | 19 | 20.891092240107927 | 0.0 | 0.47368421052631576 | endpoint_diff_baseline:19 |
| endpoint_only | linear | 16 | 18 | 18.688359710095114 | 0.0 | 0.16666666666666666 | endpoint_diff_baseline:18 |
| endpoint_only | stage65_adapter | 4 | 23 | 21.160561318782808 | 0.0 | 0.6956521739130435 | endpoint_diff_baseline:23 |
| endpoint_only | stage65_adapter | 8 | 19 | 20.97013749190139 | 0.0 | 0.631578947368421 | endpoint_diff_baseline:19 |
| endpoint_only | stage65_adapter | 16 | 18 | 18.76182012897664 | 0.0 | 0.3888888888888889 | endpoint_diff_baseline:18 |
| global_train_best_policy | linear | 4 | 23 | 20.95034838676545 | 0.0 | 0.34782608695652173 | endpoint_diff_baseline:23 |
| global_train_best_policy | linear | 8 | 19 | 20.891092240107927 | 0.0 | 0.47368421052631576 | endpoint_diff_baseline:19 |
| global_train_best_policy | linear | 16 | 18 | 18.688359710095114 | 0.0 | 0.16666666666666666 | endpoint_diff_baseline:18 |
| global_train_best_policy | stage65_adapter | 4 | 23 | 21.160561318782808 | 0.0 | 0.6956521739130435 | endpoint_diff_baseline:23 |
| global_train_best_policy | stage65_adapter | 8 | 19 | 20.97013749190139 | 0.0 | 0.631578947368421 | endpoint_diff_baseline:19 |
| global_train_best_policy | stage65_adapter | 16 | 18 | 18.76182012897664 | 0.0 | 0.3888888888888889 | endpoint_diff_baseline:18 |
| metadata_mlp_cv | linear | 4 | 23 | 20.96681417154974 | 0.016465784784290904 | 0.5652173913043478 | endpoint_diff_baseline:6;shared_energy_regression:14;shared_topk_bce:3 |
| metadata_mlp_cv | linear | 8 | 19 | 20.897864182118994 | 0.006771942011065141 | 0.3157894736842105 | endpoint_diff_baseline:4;shared_energy_regression:10;shared_topk_bce:5 |
| metadata_mlp_cv | linear | 16 | 18 | 18.786970419077075 | 0.09861070898196367 | 0.3333333333333333 | endpoint_diff_baseline:5;shared_energy_regression:7;shared_topk_bce:6 |
| metadata_mlp_cv | stage65_adapter | 4 | 23 | 21.09773175801372 | -0.06282956076908697 | 0.391304347826087 | endpoint_diff_baseline:14;shared_energy_regression:6;shared_topk_bce:3 |
| metadata_mlp_cv | stage65_adapter | 8 | 19 | 20.893291350840755 | -0.07684614106063267 | 0.3684210526315789 | endpoint_diff_baseline:7;shared_energy_regression:4;shared_topk_bce:8 |
| metadata_mlp_cv | stage65_adapter | 16 | 18 | 18.779507643673515 | 0.017687514696874313 | 0.3333333333333333 | endpoint_diff_baseline:12;shared_energy_regression:4;shared_topk_bce:2 |
| oracle_task_best | linear | 4 | 23 | 21.019524528230413 | 0.0691761414649643 | 1.0 | endpoint_diff_baseline:8;shared_energy_regression:13;shared_topk_bce:2 |
| oracle_task_best | linear | 8 | 19 | 20.992289288396925 | 0.10119704828900276 | 1.0 | endpoint_diff_baseline:9;shared_energy_regression:7;shared_topk_bce:3 |
| oracle_task_best | linear | 16 | 18 | 18.84960515929944 | 0.16124544920432715 | 1.0 | endpoint_diff_baseline:3;shared_energy_regression:10;shared_topk_bce:5 |
| oracle_task_best | stage65_adapter | 4 | 23 | 21.199724393145935 | 0.039163074363127645 | 1.0 | endpoint_diff_baseline:16;shared_energy_regression:5;shared_topk_bce:2 |
| oracle_task_best | stage65_adapter | 8 | 19 | 21.047765403031338 | 0.0776279111299487 | 1.0 | endpoint_diff_baseline:12;shared_energy_regression:3;shared_topk_bce:4 |
| oracle_task_best | stage65_adapter | 16 | 18 | 18.852925060116338 | 0.0911049311396992 | 1.0 | endpoint_diff_baseline:7;shared_energy_regression:6;shared_topk_bce:5 |
| stage106_fixed_group_policy | linear | 4 | 23 | 20.977176264225722 | 0.026827877460275387 | 0.5652173913043478 | shared_energy_regression:23 |
| stage106_fixed_group_policy | linear | 8 | 19 | 20.921955460500985 | 0.03086322039306108 | 0.3684210526315789 | shared_energy_regression:19 |
| stage106_fixed_group_policy | linear | 16 | 18 | 18.821899379666544 | 0.13353966957143293 | 0.5555555555555556 | shared_energy_regression:18 |
| stage106_fixed_group_policy | stage65_adapter | 4 | 23 | 21.160561318782808 | 0.0 | 0.6956521739130435 | endpoint_diff_baseline:23 |
| stage106_fixed_group_policy | stage65_adapter | 8 | 19 | 20.97013749190139 | 0.0 | 0.631578947368421 | endpoint_diff_baseline:19 |
| stage106_fixed_group_policy | stage65_adapter | 16 | 18 | 18.76182012897664 | 0.0 | 0.3888888888888889 | endpoint_diff_baseline:18 |
| train_fold_group_policy | linear | 4 | 23 | 20.977176264225722 | 0.026827877460275387 | 0.5652173913043478 | shared_energy_regression:23 |
| train_fold_group_policy | linear | 8 | 19 | 20.921955460500985 | 0.03086322039306108 | 0.3684210526315789 | shared_energy_regression:19 |
| train_fold_group_policy | linear | 16 | 18 | 18.821899379666544 | 0.13353966957143293 | 0.5555555555555556 | shared_energy_regression:18 |
| train_fold_group_policy | stage65_adapter | 4 | 23 | 21.160561318782808 | 0.0 | 0.6956521739130435 | endpoint_diff_baseline:23 |
| train_fold_group_policy | stage65_adapter | 8 | 19 | 20.97013749190139 | 0.0 | 0.631578947368421 | endpoint_diff_baseline:19 |
| train_fold_group_policy | stage65_adapter | 16 | 18 | 18.71243590660312 | -0.04938422237352312 | 0.3333333333333333 | endpoint_diff_baseline:15;shared_energy_regression:3 |

## Notes

- `metadata_mlp_cv` uses only coarse task metadata and is evaluated out-of-fold.
- `stage106_fixed_group_policy` is the packaged metadata group switch from Stage106.
- If metadata MLP does not exceed Stage106, richer decoder-side anchor statistics are needed before task-level switching.
