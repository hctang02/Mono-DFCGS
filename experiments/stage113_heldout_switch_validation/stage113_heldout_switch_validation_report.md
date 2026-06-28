# Stage113 Held-Out Switch Validation

## Configuration

- input rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_rows.csv`
- held-out unit: `stage97_task_id modulo fold from Stage111 rows`
- fold count: `5`
- Stage112 policy: `render_aware_group_switch_v2`
- Stage112 alias mismatch count: `0`
- no rerendering, no training, no checkpoint, no heavy tensor output

## Overall Summary

| policy | tasks | selected PSNR | gain vs endpoint | oracle task gap | accuracy | selections |
|---|---:|---:|---:|---:|---:|---|
| endpoint_only | 480 | 20.3212149854921 | 0.0 | -0.06162823546041816 | 0.5791666666666667 | endpoint_diff_baseline:480 |
| oracle_task_best | 480 | 20.382843220952545 | 0.06162823546041816 | 0.0 | 1.0 | endpoint_diff_baseline:278;shared_energy_regression:121;shared_topk_bce:81 |
| score_stat_mlp_cv | 480 | 20.33325220653739 | 0.012037221045259486 | -0.04959101441515865 | 0.6041666666666666 | endpoint_diff_baseline:274;shared_energy_regression:130;shared_topk_bce:76 |
| stage106_fixed_group_policy | 480 | 20.322996715243978 | 0.0017817297518578745 | -0.05984650570856027 | 0.51875 | endpoint_diff_baseline:240;shared_energy_regression:240 |
| stage112_group_switch_v2 | 480 | 20.32704687107235 | 0.005831885580240304 | -0.055796349880177856 | 0.55625 | endpoint_diff_baseline:323;shared_energy_regression:157 |
| train_fold_group_policy | 480 | 20.325313259771452 | 0.00409827427933979 | -0.05752996118107836 | 0.5541666666666667 | endpoint_diff_baseline:338;shared_energy_regression:142 |

## Safety Summary

| policy | mean gain | min fold gain | neg folds | min group gain | neg groups | min fold-group gain | neg fold-groups | Stage65 gap4 gain | aggregate safe | fold-group safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| endpoint_only | 0.0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1 |
| oracle_task_best | 0.06162823546041816 | 0.054184702328432775 | 0 | 0.024125012306258554 | 0 | 0.012998149861998876 | 0 | 0.024125012306258554 | 1 | 1 |
| score_stat_mlp_cv | 0.012037221045259486 | 0.0002971711561230069 | 0 | -0.00797889356792674 | 1 | -0.039626239935121585 | 10 | -0.00797889356792674 | 0 | 0 |
| stage106_fixed_group_policy | 0.0017817297518578745 | -0.007288316466769011 | 3 | -0.023422587923175493 | 1 | -0.06427740265352 | 8 | 0.0 | 0 | 0 |
| stage112_group_switch_v2 | 0.005831885580240304 | -0.0059710516126523045 | 1 | 0.0 | 0 | -0.03366017781158855 | 4 | 0.0 | 1 | 0 |
| train_fold_group_policy | 0.00409827427933979 | -0.0059710516126523045 | 1 | -0.0012260626580379986 | 1 | -0.03366017781158855 | 4 | 0.0 | 0 | 0 |

## Group Summary

| policy | base | gap | tasks | gain vs endpoint | selections |
|---|---|---:|---:|---:|---|
| score_stat_mlp_cv | linear | 4 | 83 | 0.005069813517757055 | endpoint_diff_baseline:41;shared_energy_regression:28;shared_topk_bce:14 |
| score_stat_mlp_cv | linear | 8 | 79 | 0.025965933345715158 | endpoint_diff_baseline:32;shared_energy_regression:27;shared_topk_bce:20 |
| score_stat_mlp_cv | linear | 16 | 78 | 0.02894712770300507 | endpoint_diff_baseline:31;shared_energy_regression:25;shared_topk_bce:22 |
| score_stat_mlp_cv | stage65_adapter | 4 | 83 | -0.00797889356792674 | endpoint_diff_baseline:66;shared_energy_regression:9;shared_topk_bce:8 |
| score_stat_mlp_cv | stage65_adapter | 8 | 79 | 0.003851082004932165 | endpoint_diff_baseline:53;shared_energy_regression:22;shared_topk_bce:4 |
| score_stat_mlp_cv | stage65_adapter | 16 | 78 | 0.018024353491706453 | endpoint_diff_baseline:51;shared_energy_regression:19;shared_topk_bce:8 |
| stage106_fixed_group_policy | linear | 4 | 83 | -0.023422587923175493 | shared_energy_regression:83 |
| stage106_fixed_group_policy | linear | 8 | 79 | 0.009307271828446134 | shared_energy_regression:79 |
| stage106_fixed_group_policy | linear | 16 | 78 | 0.026461930821385912 | shared_energy_regression:78 |
| stage106_fixed_group_policy | stage65_adapter | 4 | 83 | 0.0 | endpoint_diff_baseline:83 |
| stage106_fixed_group_policy | stage65_adapter | 8 | 79 | 0.0 | endpoint_diff_baseline:79 |
| stage106_fixed_group_policy | stage65_adapter | 16 | 78 | 0.0 | endpoint_diff_baseline:78 |
| stage112_group_switch_v2 | linear | 4 | 83 | 0.0 | endpoint_diff_baseline:83 |
| stage112_group_switch_v2 | linear | 8 | 79 | 0.009307271828446134 | shared_energy_regression:79 |
| stage112_group_switch_v2 | linear | 16 | 78 | 0.026461930821385912 | shared_energy_regression:78 |
| stage112_group_switch_v2 | stage65_adapter | 4 | 83 | 0.0 | endpoint_diff_baseline:83 |
| stage112_group_switch_v2 | stage65_adapter | 8 | 79 | 0.0 | endpoint_diff_baseline:79 |
| stage112_group_switch_v2 | stage65_adapter | 16 | 78 | 0.0 | endpoint_diff_baseline:78 |
| train_fold_group_policy | linear | 4 | 83 | 0.0 | endpoint_diff_baseline:83 |
| train_fold_group_policy | linear | 8 | 79 | -0.0012260626580379986 | endpoint_diff_baseline:15;shared_energy_regression:64 |
| train_fold_group_policy | linear | 16 | 78 | 0.026461930821385912 | shared_energy_regression:78 |
| train_fold_group_policy | stage65_adapter | 4 | 83 | 0.0 | endpoint_diff_baseline:83 |
| train_fold_group_policy | stage65_adapter | 8 | 79 | 0.0 | endpoint_diff_baseline:79 |
| train_fold_group_policy | stage65_adapter | 16 | 78 | 0.0 | endpoint_diff_baseline:78 |

## Fold Summary

| fold | policy | tasks | gain vs endpoint | selections |
|---:|---|---:|---:|---|
| 0 | score_stat_mlp_cv | 96 | 0.0002971711561230069 | endpoint_diff_baseline:51;shared_energy_regression:35;shared_topk_bce:10 |
| 0 | stage112_group_switch_v2 | 96 | 0.004283020330720015 | endpoint_diff_baseline:64;shared_energy_regression:32 |
| 0 | train_fold_group_policy | 96 | 0.004283020330720015 | endpoint_diff_baseline:64;shared_energy_regression:32 |
| 1 | score_stat_mlp_cv | 96 | 0.010633342290857273 | endpoint_diff_baseline:55;shared_energy_regression:26;shared_topk_bce:15 |
| 1 | stage112_group_switch_v2 | 96 | -0.0059710516126523045 | endpoint_diff_baseline:67;shared_energy_regression:29 |
| 1 | train_fold_group_policy | 96 | -0.0059710516126523045 | endpoint_diff_baseline:67;shared_energy_regression:29 |
| 2 | score_stat_mlp_cv | 96 | 0.031199149081095723 | endpoint_diff_baseline:52;shared_energy_regression:22;shared_topk_bce:22 |
| 2 | stage112_group_switch_v2 | 96 | 0.021492052426497654 | endpoint_diff_baseline:65;shared_energy_regression:31 |
| 2 | train_fold_group_policy | 96 | 0.012823995921995087 | endpoint_diff_baseline:80;shared_energy_regression:16 |
| 3 | score_stat_mlp_cv | 96 | 0.007771476145027989 | endpoint_diff_baseline:53;shared_energy_regression:26;shared_topk_bce:17 |
| 3 | stage112_group_switch_v2 | 96 | 0.009220813070045408 | endpoint_diff_baseline:65;shared_energy_regression:31 |
| 3 | train_fold_group_policy | 96 | 0.009220813070045408 | endpoint_diff_baseline:65;shared_energy_regression:31 |
| 4 | score_stat_mlp_cv | 96 | 0.010284966553193442 | endpoint_diff_baseline:63;shared_energy_regression:21;shared_topk_bce:12 |
| 4 | stage112_group_switch_v2 | 96 | 0.0001345936865907449 | endpoint_diff_baseline:62;shared_energy_regression:34 |
| 4 | train_fold_group_policy | 96 | 0.0001345936865907449 | endpoint_diff_baseline:62;shared_energy_regression:34 |

## Notes

- The Stage111 fold split is by sorted `stage97_task_id` modulo fold count; this is task-id held-out CV, not a fresh sequence-heldout render run.
- `stage112_group_switch_v2` is evaluated by aliasing Stage111 `stage110_group_best_policy` rows and verifying the Stage112 package selection table.
- Stage110/111 rows still use teacher residual values at selected indices, so this validates selector switching, not a complete residual-value codec.
- If fold-group stability is required as a hard safety condition, any negative fold-group cell should be treated as a blocker before residual value prediction.
