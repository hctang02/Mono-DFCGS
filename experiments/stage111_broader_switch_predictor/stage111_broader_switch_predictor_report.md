# Stage111 Broader Switch Predictor

## Configuration

- task rows: `480`
- folds: `5`
- selector train tasks: `96`
- selector train examples: `589824`
- feature dims: `{'anchor_stat_mlp_cv': 64, 'score_stat_mlp_cv': 82, 'anchor_score_mlp_cv': 133}`
- features are decoder-side metadata, left/right/base anchor aggregate statistics, and selector score/logit statistics
- switch labels come from Stage103 rendered rows and are used only for offline train/eval
- no target dense anchor, target residual, rendered PSNR, checkpoint, or heavy tensor is used as switch predictor input/output

## Overall Summary

| policy | tasks | selected PSNR | endpoint PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| anchor_score_mlp_cv | 480 | 20.33140381862592 | 20.3212149854921 | 0.010188833133790712 | 20.382843220952545 | -0.05143940232662742 | 0.59375 | endpoint_diff_baseline:270;shared_energy_regression:122;shared_topk_bce:88 |
| anchor_stat_mlp_cv | 480 | 20.331745776419964 | 20.3212149854921 | 0.010530790927825506 | 20.382843220952545 | -0.051097444532592656 | 0.6041666666666666 | endpoint_diff_baseline:261;shared_energy_regression:134;shared_topk_bce:85 |
| endpoint_only | 480 | 20.3212149854921 | 20.3212149854921 | 0.0 | 20.382843220952545 | -0.06162823546041816 | 0.5791666666666667 | endpoint_diff_baseline:480 |
| global_train_best_policy | 480 | 20.3212149854921 | 20.3212149854921 | 0.0 | 20.382843220952545 | -0.06162823546041816 | 0.5791666666666667 | endpoint_diff_baseline:480 |
| metadata_mlp_cv | 480 | 20.27337705662733 | 20.3212149854921 | -0.047837928864800545 | 20.382843220952545 | -0.10946616432521869 | 0.4395833333333333 | endpoint_diff_baseline:211;shared_energy_regression:153;shared_topk_bce:116 |
| oracle_task_best | 480 | 20.382843220952545 | 20.3212149854921 | 0.06162823546041816 | 20.382843220952545 | 0.0 | 1.0 | endpoint_diff_baseline:278;shared_energy_regression:121;shared_topk_bce:81 |
| score_stat_mlp_cv | 480 | 20.33325220653739 | 20.3212149854921 | 0.012037221045259486 | 20.382843220952545 | -0.04959101441515865 | 0.6041666666666666 | endpoint_diff_baseline:274;shared_energy_regression:130;shared_topk_bce:76 |
| stage106_fixed_group_policy | 480 | 20.322996715243978 | 20.3212149854921 | 0.0017817297518578745 | 20.382843220952545 | -0.05984650570856027 | 0.51875 | endpoint_diff_baseline:240;shared_energy_regression:240 |
| stage110_group_best_policy | 480 | 20.32704687107235 | 20.3212149854921 | 0.005831885580240304 | 20.382843220952545 | -0.055796349880177856 | 0.55625 | endpoint_diff_baseline:323;shared_energy_regression:157 |
| train_fold_group_policy | 480 | 20.325313259771452 | 20.3212149854921 | 0.00409827427933979 | 20.382843220952545 | -0.05752996118107836 | 0.5541666666666667 | endpoint_diff_baseline:338;shared_energy_regression:142 |

## Group Summary

| policy | base | gap | tasks | selected PSNR | gain vs endpoint | accuracy | selections |
|---|---|---:|---:|---:|---:|---:|---|
| anchor_score_mlp_cv | linear | 4 | 83 | 21.303722881714897 | 0.00417571062015979 | 0.5301204819277109 | endpoint_diff_baseline:38;shared_energy_regression:31;shared_topk_bce:14 |
| anchor_score_mlp_cv | linear | 8 | 79 | 20.3956991646984 | 0.020150733418887467 | 0.5063291139240507 | endpoint_diff_baseline:33;shared_energy_regression:26;shared_topk_bce:20 |
| anchor_score_mlp_cv | linear | 16 | 78 | 19.086122510486728 | 0.03921282331924403 | 0.48717948717948717 | endpoint_diff_baseline:28;shared_energy_regression:25;shared_topk_bce:25 |
| anchor_score_mlp_cv | stage65_adapter | 4 | 83 | 21.35623328474749 | -0.0006339883919905541 | 0.7831325301204819 | endpoint_diff_baseline:66;shared_energy_regression:10;shared_topk_bce:7 |
| anchor_score_mlp_cv | stage65_adapter | 8 | 79 | 20.528638942191044 | 0.003371547583492312 | 0.6835443037974683 | endpoint_diff_baseline:58;shared_energy_regression:14;shared_topk_bce:7 |
| anchor_score_mlp_cv | stage65_adapter | 16 | 78 | 19.1866308315619 | -0.004104916138558223 | 0.5641025641025641 | endpoint_diff_baseline:47;shared_energy_regression:16;shared_topk_bce:15 |
| anchor_stat_mlp_cv | linear | 4 | 83 | 21.307688871844395 | 0.00814170074965776 | 0.5542168674698795 | endpoint_diff_baseline:40;shared_energy_regression:32;shared_topk_bce:11 |
| anchor_stat_mlp_cv | linear | 8 | 79 | 20.400655891217234 | 0.025107459937721562 | 0.5443037974683544 | endpoint_diff_baseline:29;shared_energy_regression:29;shared_topk_bce:21 |
| anchor_stat_mlp_cv | linear | 16 | 78 | 19.091699529840874 | 0.04478984267338267 | 0.5128205128205128 | endpoint_diff_baseline:29;shared_energy_regression:28;shared_topk_bce:21 |
| anchor_stat_mlp_cv | stage65_adapter | 4 | 83 | 21.348929577164483 | -0.007937695974995159 | 0.7831325301204819 | endpoint_diff_baseline:61;shared_energy_regression:14;shared_topk_bce:8 |
| anchor_stat_mlp_cv | stage65_adapter | 8 | 79 | 20.5162897055817 | -0.008977689025852805 | 0.6962025316455697 | endpoint_diff_baseline:57;shared_energy_regression:14;shared_topk_bce:8 |
| anchor_stat_mlp_cv | stage65_adapter | 16 | 78 | 19.194197127424793 | 0.003461379724330349 | 0.5256410256410257 | endpoint_diff_baseline:45;shared_energy_regression:17;shared_topk_bce:16 |
| endpoint_only | linear | 4 | 83 | 21.299547171094733 | 0.0 | 0.5421686746987951 | endpoint_diff_baseline:83 |
| endpoint_only | linear | 8 | 79 | 20.375548431279512 | 0.0 | 0.43037974683544306 | endpoint_diff_baseline:79 |
| endpoint_only | linear | 16 | 78 | 19.046909687167492 | 0.0 | 0.38461538461538464 | endpoint_diff_baseline:78 |
| endpoint_only | stage65_adapter | 4 | 83 | 21.356867273139475 | 0.0 | 0.8072289156626506 | endpoint_diff_baseline:83 |
| endpoint_only | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | 0.7088607594936709 | endpoint_diff_baseline:79 |
| endpoint_only | stage65_adapter | 16 | 78 | 19.190735747700458 | 0.0 | 0.5897435897435898 | endpoint_diff_baseline:78 |
| global_train_best_policy | linear | 4 | 83 | 21.299547171094733 | 0.0 | 0.5421686746987951 | endpoint_diff_baseline:83 |
| global_train_best_policy | linear | 8 | 79 | 20.375548431279512 | 0.0 | 0.43037974683544306 | endpoint_diff_baseline:79 |
| global_train_best_policy | linear | 16 | 78 | 19.046909687167492 | 0.0 | 0.38461538461538464 | endpoint_diff_baseline:78 |
| global_train_best_policy | stage65_adapter | 4 | 83 | 21.356867273139475 | 0.0 | 0.8072289156626506 | endpoint_diff_baseline:83 |
| global_train_best_policy | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | 0.7088607594936709 | endpoint_diff_baseline:79 |
| global_train_best_policy | stage65_adapter | 16 | 78 | 19.190735747700458 | 0.0 | 0.5897435897435898 | endpoint_diff_baseline:78 |
| metadata_mlp_cv | linear | 4 | 83 | 21.24452898222081 | -0.05501818887392087 | 0.3253012048192771 | endpoint_diff_baseline:24;shared_energy_regression:43;shared_topk_bce:16 |
| metadata_mlp_cv | linear | 8 | 79 | 20.333874125633066 | -0.041674305646451006 | 0.35443037974683544 | endpoint_diff_baseline:16;shared_energy_regression:38;shared_topk_bce:25 |
| metadata_mlp_cv | linear | 16 | 78 | 19.062108033281625 | 0.01519834611413123 | 0.358974358974359 | endpoint_diff_baseline:23;shared_energy_regression:29;shared_topk_bce:26 |
| metadata_mlp_cv | stage65_adapter | 4 | 83 | 21.28964018050177 | -0.06722709263770256 | 0.6144578313253012 | endpoint_diff_baseline:63;shared_energy_regression:9;shared_topk_bce:11 |
| metadata_mlp_cv | stage65_adapter | 8 | 79 | 20.43972777970121 | -0.08553961490633653 | 0.4810126582278481 | endpoint_diff_baseline:47;shared_energy_regression:17;shared_topk_bce:15 |
| metadata_mlp_cv | stage65_adapter | 16 | 78 | 19.140076532791852 | -0.050659214908609425 | 0.5 | endpoint_diff_baseline:38;shared_energy_regression:17;shared_topk_bce:23 |
| oracle_task_best | linear | 4 | 83 | 21.348889566566452 | 0.04934239547172279 | 1.0 | endpoint_diff_baseline:45;shared_energy_regression:27;shared_topk_bce:11 |
| oracle_task_best | linear | 8 | 79 | 20.452041094253882 | 0.07649266297437256 | 1.0 | endpoint_diff_baseline:34;shared_energy_regression:27;shared_topk_bce:18 |
| oracle_task_best | linear | 16 | 78 | 19.14875439768973 | 0.10184471052224166 | 1.0 | endpoint_diff_baseline:30;shared_energy_regression:26;shared_topk_bce:22 |
| oracle_task_best | stage65_adapter | 4 | 83 | 21.380992285445735 | 0.024125012306258554 | 1.0 | endpoint_diff_baseline:67;shared_energy_regression:9;shared_topk_bce:7 |
| oracle_task_best | stage65_adapter | 8 | 79 | 20.572239134838952 | 0.046971740231397476 | 1.0 | endpoint_diff_baseline:56;shared_energy_regression:15;shared_topk_bce:8 |
| oracle_task_best | stage65_adapter | 16 | 78 | 19.26491757977017 | 0.07418183206971249 | 1.0 | endpoint_diff_baseline:46;shared_energy_regression:17;shared_topk_bce:15 |
| score_stat_mlp_cv | linear | 4 | 83 | 21.304616984612498 | 0.005069813517757055 | 0.5783132530120482 | endpoint_diff_baseline:41;shared_energy_regression:28;shared_topk_bce:14 |
| score_stat_mlp_cv | linear | 8 | 79 | 20.40151436462523 | 0.025965933345715158 | 0.46835443037974683 | endpoint_diff_baseline:32;shared_energy_regression:27;shared_topk_bce:20 |
| score_stat_mlp_cv | linear | 16 | 78 | 19.07585681487049 | 0.02894712770300507 | 0.5 | endpoint_diff_baseline:31;shared_energy_regression:25;shared_topk_bce:22 |
| score_stat_mlp_cv | stage65_adapter | 4 | 83 | 21.34888837957155 | -0.00797889356792674 | 0.7710843373493976 | endpoint_diff_baseline:66;shared_energy_regression:9;shared_topk_bce:8 |
| score_stat_mlp_cv | stage65_adapter | 8 | 79 | 20.52911847661248 | 0.003851082004932165 | 0.7215189873417721 | endpoint_diff_baseline:53;shared_energy_regression:22;shared_topk_bce:4 |
| score_stat_mlp_cv | stage65_adapter | 16 | 78 | 19.208760101192166 | 0.018024353491706453 | 0.5769230769230769 | endpoint_diff_baseline:51;shared_energy_regression:19;shared_topk_bce:8 |
| stage106_fixed_group_policy | linear | 4 | 83 | 21.27612458317156 | -0.023422587923175493 | 0.3253012048192771 | shared_energy_regression:83 |
| stage106_fixed_group_policy | linear | 8 | 79 | 20.384855703107963 | 0.009307271828446134 | 0.34177215189873417 | shared_energy_regression:79 |
| stage106_fixed_group_policy | linear | 16 | 78 | 19.07337161798887 | 0.026461930821385912 | 0.3333333333333333 | shared_energy_regression:78 |
| stage106_fixed_group_policy | stage65_adapter | 4 | 83 | 21.356867273139475 | 0.0 | 0.8072289156626506 | endpoint_diff_baseline:83 |
| stage106_fixed_group_policy | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | 0.7088607594936709 | endpoint_diff_baseline:79 |
| stage106_fixed_group_policy | stage65_adapter | 16 | 78 | 19.190735747700458 | 0.0 | 0.5897435897435898 | endpoint_diff_baseline:78 |
| stage110_group_best_policy | linear | 4 | 83 | 21.299547171094733 | 0.0 | 0.5421686746987951 | endpoint_diff_baseline:83 |
| stage110_group_best_policy | linear | 8 | 79 | 20.384855703107963 | 0.009307271828446134 | 0.34177215189873417 | shared_energy_regression:79 |
| stage110_group_best_policy | linear | 16 | 78 | 19.07337161798887 | 0.026461930821385912 | 0.3333333333333333 | shared_energy_regression:78 |
| stage110_group_best_policy | stage65_adapter | 4 | 83 | 21.356867273139475 | 0.0 | 0.8072289156626506 | endpoint_diff_baseline:83 |
| stage110_group_best_policy | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | 0.7088607594936709 | endpoint_diff_baseline:79 |
| stage110_group_best_policy | stage65_adapter | 16 | 78 | 19.190735747700458 | 0.0 | 0.5897435897435898 | endpoint_diff_baseline:78 |
| train_fold_group_policy | linear | 4 | 83 | 21.299547171094733 | 0.0 | 0.5421686746987951 | endpoint_diff_baseline:83 |
| train_fold_group_policy | linear | 8 | 79 | 20.37432236862148 | -0.0012260626580379986 | 0.3291139240506329 | endpoint_diff_baseline:15;shared_energy_regression:64 |
| train_fold_group_policy | linear | 16 | 78 | 19.07337161798887 | 0.026461930821385912 | 0.3333333333333333 | shared_energy_regression:78 |
| train_fold_group_policy | stage65_adapter | 4 | 83 | 21.356867273139475 | 0.0 | 0.8072289156626506 | endpoint_diff_baseline:83 |
| train_fold_group_policy | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | 0.7088607594936709 | endpoint_diff_baseline:79 |
| train_fold_group_policy | stage65_adapter | 16 | 78 | 19.190735747700458 | 0.0 | 0.5897435897435898 | endpoint_diff_baseline:78 |

## Notes

- `score_stat_mlp_cv` uses metadata plus selector score/logit statistics only.
- `anchor_score_mlp_cv` combines metadata, anchor aggregate statistics, and selector score/logit statistics.
- If score features do not beat Stage106, Stage106 remains the safe switch baseline until broader rendered labels are available.
