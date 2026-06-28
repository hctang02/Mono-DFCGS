# Stage110 Broader Render-Aware Selector Policy Preflight

## Configuration

- input rows: `experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_rows.csv`
- task count: `480`
- policies: `endpoint_only, always_shared_energy_regression, always_shared_topk_bce, stage106_fixed_group_policy, group_best_mean_psnr, oracle_task_best`
- no rendering or heavy tensor output

## Overall Summary

| policy | tasks | selected PSNR | endpoint PSNR | gain vs endpoint | teacher PSNR | gap to teacher | selections |
|---|---:|---:|---:|---:|---:|---:|---|
| always_shared_energy_regression | 480 | 20.229984156626987 | 20.3212149854921 | -0.09123082886513802 | 22.077800340877268 | -1.8478161842502654 | shared_energy_regression:480 |
| always_shared_topk_bce | 480 | 20.159752797305014 | 20.3212149854921 | -0.16146218818711972 | 22.077800340877268 | -1.9180475435722473 | shared_topk_bce:480 |
| endpoint_only | 480 | 20.3212149854921 | 20.3212149854921 | 0.0 | 22.077800340877268 | -1.7565853553851296 | endpoint_diff_baseline:480 |
| group_best_mean_psnr | 480 | 20.327046871072337 | 20.3212149854921 | 0.005831885580240304 | 22.077800340877268 | -1.750753469804888 | endpoint_diff_baseline:323;shared_energy_regression:157 |
| oracle_task_best | 480 | 20.382843220952523 | 20.3212149854921 | 0.06162823546041816 | 22.077800340877268 | -1.6949571199247109 | endpoint_diff_baseline:278;shared_energy_regression:121;shared_topk_bce:81 |
| stage106_fixed_group_policy | 480 | 20.322996715243953 | 20.3212149854921 | 0.0017817297518578745 | 22.077800340877268 | -1.7548036256332702 | endpoint_diff_baseline:240;shared_energy_regression:240 |

## Group Policy Choices

| base | gap | selected candidate | mean PSNR | endpoint PSNR | gain |
|---|---:|---|---:|---:|---:|
| linear | 4 | endpoint_diff_baseline | 21.299547171094726 | 21.299547171094726 | 0.0 |
| linear | 8 | shared_energy_regression | 20.38485570310796 | 20.37554843127951 | 0.009307271828451036 |
| linear | 16 | shared_energy_regression | 19.073371617988876 | 19.046909687167496 | 0.026461930821380264 |
| stage65_adapter | 4 | endpoint_diff_baseline | 21.356867273139482 | 21.356867273139482 | 0.0 |
| stage65_adapter | 8 | endpoint_diff_baseline | 20.525267394607557 | 20.525267394607557 | 0.0 |
| stage65_adapter | 16 | endpoint_diff_baseline | 19.19073574770046 | 19.19073574770046 | 0.0 |

## Group Summary

| policy | base | gap | tasks | selected PSNR | gain vs endpoint | gap to teacher | selections |
|---|---|---:|---:|---:|---:|---:|---|
| always_shared_energy_regression | linear | 4 | 83 | 21.276124583171555 | -0.023422587923175493 | -1.7397247945374348 | shared_energy_regression:83 |
| always_shared_energy_regression | linear | 8 | 79 | 20.384855703107966 | 0.009307271828446134 | -1.9862619189663924 | shared_energy_regression:79 |
| always_shared_energy_regression | linear | 16 | 78 | 19.073371617988872 | 0.026461930821385912 | -2.080985030251295 | shared_energy_regression:78 |
| always_shared_energy_regression | stage65_adapter | 4 | 83 | 21.131463440519376 | -0.22540383262011054 | -1.5801793374114348 | shared_energy_regression:83 |
| always_shared_energy_regression | stage65_adapter | 8 | 79 | 20.331538244866433 | -0.19372914974113 | -1.8065089452844398 | shared_energy_regression:79 |
| always_shared_energy_regression | stage65_adapter | 16 | 78 | 19.054416424634564 | -0.1363193230658923 | -1.916076857089164 | shared_energy_regression:78 |
| always_shared_topk_bce | linear | 4 | 83 | 21.141167440341185 | -0.15837973075354014 | -1.8746819373677999 | shared_topk_bce:83 |
| always_shared_topk_bce | linear | 8 | 79 | 20.258516107689786 | -0.11703232358972818 | -2.1126015143845662 | shared_topk_bce:79 |
| always_shared_topk_bce | linear | 16 | 78 | 18.978804685006114 | -0.06810500216136936 | -2.175551963234049 | shared_topk_bce:78 |
| always_shared_topk_bce | stage65_adapter | 4 | 83 | 21.109714693426387 | -0.24715257971309498 | -1.6019280845044193 | shared_topk_bce:83 |
| always_shared_topk_bce | stage65_adapter | 8 | 79 | 20.31107324044725 | -0.21419415416029447 | -1.8269739497036048 | shared_topk_bce:79 |
| always_shared_topk_bce | stage65_adapter | 16 | 78 | 19.032228226543427 | -0.15850752115702738 | -1.9382650551802998 | shared_topk_bce:78 |
| endpoint_only | linear | 4 | 83 | 21.299547171094726 | 0.0 | -1.7163022066142593 | endpoint_diff_baseline:83 |
| endpoint_only | linear | 8 | 79 | 20.375548431279505 | 0.0 | -1.9955691907948385 | endpoint_diff_baseline:79 |
| endpoint_only | linear | 16 | 78 | 19.046909687167492 | 0.0 | -2.10744696107268 | endpoint_diff_baseline:78 |
| endpoint_only | stage65_adapter | 4 | 83 | 21.356867273139482 | 0.0 | -1.354775504791325 | endpoint_diff_baseline:83 |
| endpoint_only | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | -1.6127797955433103 | endpoint_diff_baseline:79 |
| endpoint_only | stage65_adapter | 16 | 78 | 19.19073574770046 | 0.0 | -1.779757534023271 | endpoint_diff_baseline:78 |
| group_best_mean_psnr | linear | 4 | 83 | 21.299547171094726 | 0.0 | -1.7163022066142593 | endpoint_diff_baseline:83 |
| group_best_mean_psnr | linear | 8 | 79 | 20.384855703107966 | 0.009307271828446134 | -1.9862619189663924 | shared_energy_regression:79 |
| group_best_mean_psnr | linear | 16 | 78 | 19.073371617988872 | 0.026461930821385912 | -2.080985030251295 | shared_energy_regression:78 |
| group_best_mean_psnr | stage65_adapter | 4 | 83 | 21.356867273139482 | 0.0 | -1.354775504791325 | endpoint_diff_baseline:83 |
| group_best_mean_psnr | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | -1.6127797955433103 | endpoint_diff_baseline:79 |
| group_best_mean_psnr | stage65_adapter | 16 | 78 | 19.19073574770046 | 0.0 | -1.779757534023271 | endpoint_diff_baseline:78 |
| oracle_task_best | linear | 4 | 83 | 21.348889566566456 | 0.04934239547172279 | -1.6669598111425363 | endpoint_diff_baseline:45;shared_energy_regression:27;shared_topk_bce:11 |
| oracle_task_best | linear | 8 | 79 | 20.45204109425389 | 0.07649266297437256 | -1.9190765278204662 | endpoint_diff_baseline:34;shared_energy_regression:27;shared_topk_bce:18 |
| oracle_task_best | linear | 16 | 78 | 19.148754397689725 | 0.10184471052224166 | -2.0056022505504387 | endpoint_diff_baseline:30;shared_energy_regression:26;shared_topk_bce:22 |
| oracle_task_best | stage65_adapter | 4 | 83 | 21.38099228544574 | 0.024125012306258554 | -1.3306504924850666 | endpoint_diff_baseline:67;shared_energy_regression:9;shared_topk_bce:7 |
| oracle_task_best | stage65_adapter | 8 | 79 | 20.572239134838956 | 0.046971740231397476 | -1.5658080553119134 | endpoint_diff_baseline:56;shared_energy_regression:15;shared_topk_bce:8 |
| oracle_task_best | stage65_adapter | 16 | 78 | 19.264917579770174 | 0.07418183206971249 | -1.705575701953559 | endpoint_diff_baseline:46;shared_energy_regression:17;shared_topk_bce:15 |
| stage106_fixed_group_policy | linear | 4 | 83 | 21.276124583171555 | -0.023422587923175493 | -1.7397247945374348 | shared_energy_regression:83 |
| stage106_fixed_group_policy | linear | 8 | 79 | 20.384855703107966 | 0.009307271828446134 | -1.9862619189663924 | shared_energy_regression:79 |
| stage106_fixed_group_policy | linear | 16 | 78 | 19.073371617988872 | 0.026461930821385912 | -2.080985030251295 | shared_energy_regression:78 |
| stage106_fixed_group_policy | stage65_adapter | 4 | 83 | 21.356867273139482 | 0.0 | -1.354775504791325 | endpoint_diff_baseline:83 |
| stage106_fixed_group_policy | stage65_adapter | 8 | 79 | 20.525267394607553 | 0.0 | -1.6127797955433103 | endpoint_diff_baseline:79 |
| stage106_fixed_group_policy | stage65_adapter | 16 | 78 | 19.19073574770046 | 0.0 | -1.779757534023271 | endpoint_diff_baseline:78 |

## Notes

- `group_best_mean_psnr` is a fixed group policy selected from rendered validation rows.
- `oracle_task_best` uses per-task rendered PSNR and is only an upper bound, not deployable.
- All candidates still use teacher residual values at selected indices; residual value prediction remains unresolved.
