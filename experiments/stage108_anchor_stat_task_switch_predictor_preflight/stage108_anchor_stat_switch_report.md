# Stage108 Anchor-Stat Task-Level Switch Predictor Preflight

## Configuration

- task rows: `120`
- folds: `5`
- feature dim: `64`
- features are decoder-side metadata plus left/right/base anchor aggregate statistics
- no target dense anchor, target residual, rendering, checkpoint, or heavy tensor output

## Overall Summary

| policy | tasks | selected PSNR | endpoint PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| anchor_stat_mlp_cv | 120 | 20.33017523703834 | 20.316812710325653 | 0.013362526712690951 | 20.403744235652297 | -0.07356899861396461 | 0.475 | endpoint_diff_baseline:51;shared_energy_regression:47;shared_topk_bce:22 |
| endpoint_only | 120 | 20.316812710325653 | 20.316812710325653 | 0.0 | 20.403744235652297 | -0.08693152532665556 | 0.4583333333333333 | endpoint_diff_baseline:120 |
| metadata_mlp_cv | 120 | 20.31646985845114 | 20.316812710325653 | -0.00034285187450664443 | 20.403744235652297 | -0.0872743772011622 | 0.38333333333333336 | endpoint_diff_baseline:46;shared_energy_regression:44;shared_topk_bce:30 |
| oracle_task_best | 120 | 20.403744235652297 | 20.316812710325653 | 0.08693152532665556 | 20.403744235652297 | 0.0 | 1.0 | endpoint_diff_baseline:55;shared_energy_regression:44;shared_topk_bce:21 |
| stage106_fixed_group_policy | 120 | 20.34687234717015 | 20.316812710325653 | 0.030059636844502392 | 20.403744235652297 | -0.05687188848215317 | 0.5416666666666666 | endpoint_diff_baseline:60;shared_energy_regression:60 |
| train_fold_group_policy | 120 | 20.33946471381412 | 20.316812710325653 | 0.02265200348847392 | 20.403744235652297 | -0.06427952183818163 | 0.5333333333333333 | endpoint_diff_baseline:57;shared_energy_regression:63 |

## Group Summary

| policy | base | gap | tasks | selected PSNR | gain vs endpoint | accuracy | selections |
|---|---|---:|---:|---:|---:|---:|---|
| anchor_stat_mlp_cv | linear | 4 | 23 | 20.981301285727856 | 0.030952898962409957 | 0.5652173913043478 | endpoint_diff_baseline:6;shared_energy_regression:13;shared_topk_bce:4 |
| anchor_stat_mlp_cv | linear | 8 | 19 | 20.921661488348366 | 0.030569248240443252 | 0.3684210526315789 | endpoint_diff_baseline:9;shared_energy_regression:7;shared_topk_bce:3 |
| anchor_stat_mlp_cv | linear | 16 | 18 | 18.75488200152066 | 0.06652229142554766 | 0.3888888888888889 | endpoint_diff_baseline:4;shared_energy_regression:9;shared_topk_bce:5 |
| anchor_stat_mlp_cv | stage65_adapter | 4 | 23 | 21.16125324371534 | 0.000691924932532272 | 0.6956521739130435 | endpoint_diff_baseline:14;shared_energy_regression:7;shared_topk_bce:2 |
| anchor_stat_mlp_cv | stage65_adapter | 8 | 19 | 20.919985270084425 | -0.05015222181696469 | 0.42105263157894735 | endpoint_diff_baseline:11;shared_energy_regression:4;shared_topk_bce:4 |
| anchor_stat_mlp_cv | stage65_adapter | 16 | 18 | 18.764617212767384 | 0.0027970837907384588 | 0.3333333333333333 | endpoint_diff_baseline:7;shared_energy_regression:7;shared_topk_bce:4 |
| endpoint_only | linear | 4 | 23 | 20.95034838676545 | 0.0 | 0.34782608695652173 | endpoint_diff_baseline:23 |
| endpoint_only | linear | 8 | 19 | 20.891092240107927 | 0.0 | 0.47368421052631576 | endpoint_diff_baseline:19 |
| endpoint_only | linear | 16 | 18 | 18.688359710095114 | 0.0 | 0.16666666666666666 | endpoint_diff_baseline:18 |
| endpoint_only | stage65_adapter | 4 | 23 | 21.160561318782808 | 0.0 | 0.6956521739130435 | endpoint_diff_baseline:23 |
| endpoint_only | stage65_adapter | 8 | 19 | 20.97013749190139 | 0.0 | 0.631578947368421 | endpoint_diff_baseline:19 |
| endpoint_only | stage65_adapter | 16 | 18 | 18.76182012897664 | 0.0 | 0.3888888888888889 | endpoint_diff_baseline:18 |
| metadata_mlp_cv | linear | 4 | 23 | 20.964427433268913 | 0.014079046503468516 | 0.5217391304347826 | endpoint_diff_baseline:3;shared_energy_regression:17;shared_topk_bce:3 |
| metadata_mlp_cv | linear | 8 | 19 | 20.89490917453362 | 0.0038169344256912745 | 0.2631578947368421 | endpoint_diff_baseline:4;shared_energy_regression:7;shared_topk_bce:8 |
| metadata_mlp_cv | linear | 16 | 18 | 18.788227312917435 | 0.09986760282232471 | 0.3888888888888889 | endpoint_diff_baseline:5;shared_energy_regression:7;shared_topk_bce:6 |
| metadata_mlp_cv | stage65_adapter | 4 | 23 | 21.115867750565528 | -0.04469356821728223 | 0.43478260869565216 | endpoint_diff_baseline:16;shared_energy_regression:4;shared_topk_bce:3 |
| metadata_mlp_cv | stage65_adapter | 8 | 19 | 20.900721036377863 | -0.06941645552352477 | 0.3684210526315789 | endpoint_diff_baseline:7;shared_energy_regression:5;shared_topk_bce:7 |
| metadata_mlp_cv | stage65_adapter | 16 | 18 | 18.76802934145075 | 0.006209212474106095 | 0.2777777777777778 | endpoint_diff_baseline:11;shared_energy_regression:4;shared_topk_bce:3 |
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

- Anchor-stat features are decoder-available because they use only transmitted left/right anchors and predicted base anchors.
- Rendered PSNR labels are used only for offline training/evaluation.
- If anchor-stat MLP does not beat Stage106, switch learning should wait for more data or selector-score features.
