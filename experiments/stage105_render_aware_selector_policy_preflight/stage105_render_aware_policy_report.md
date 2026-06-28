# Stage105 Render-Aware Selector Policy Preflight

## Configuration

- input rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_rows.csv`
- task count: `120`
- policies: `endpoint_only, always_shared_energy_regression, always_shared_topk_bce, group_best_mean_psnr, oracle_task_best`
- no rendering or heavy tensor output

## Overall Summary

| policy | tasks | selected PSNR | endpoint PSNR | gain vs endpoint | teacher PSNR | gap to teacher | selections |
|---|---:|---:|---:|---:|---:|---:|---|
| always_shared_energy_regression | 120 | 20.284564319533338 | 20.316812710325646 | -0.03224839079231098 | 22.010204667924707 | -1.7256403483913645 | shared_energy_regression:120 |
| always_shared_topk_bce | 120 | 20.21561560612798 | 20.316812710325646 | -0.10119710419766939 | 22.010204667924707 | -1.7945890617967224 | shared_topk_bce:120 |
| endpoint_only | 120 | 20.316812710325646 | 20.316812710325646 | 0.0 | 22.010204667924707 | -1.6933919575990535 | endpoint_diff_baseline:120 |
| group_best_mean_psnr | 120 | 20.346872347170144 | 20.316812710325646 | 0.030059636844502392 | 22.010204667924707 | -1.6633323207545507 | endpoint_diff_baseline:60;shared_energy_regression:60 |
| oracle_task_best | 120 | 20.403744235652294 | 20.316812710325646 | 0.08693152532665556 | 22.010204667924707 | -1.606460432272398 | endpoint_diff_baseline:55;shared_energy_regression:44;shared_topk_bce:21 |

## Group Policy Choices

| base | gap | selected candidate | mean PSNR | endpoint PSNR | gain |
|---|---:|---|---:|---:|---:|
| linear | 4 | shared_energy_regression | 20.977176264225726 | 20.950348386765448 | 0.026827877460277705 |
| linear | 8 | shared_energy_regression | 20.921955460500982 | 20.891092240107927 | 0.030863220393055002 |
| linear | 16 | shared_energy_regression | 18.821899379666544 | 18.68835971009511 | 0.1335396695714337 |
| stage65_adapter | 4 | endpoint_diff_baseline | 21.16056131878281 | 21.16056131878281 | 0.0 |
| stage65_adapter | 8 | endpoint_diff_baseline | 20.97013749190139 | 20.97013749190139 | 0.0 |
| stage65_adapter | 16 | endpoint_diff_baseline | 18.76182012897664 | 18.76182012897664 | 0.0 |

## Group Summary

| policy | base | gap | tasks | selected PSNR | gain vs endpoint | gap to teacher | selections |
|---|---|---:|---:|---:|---:|---:|---|
| always_shared_energy_regression | linear | 4 | 23 | 20.977176264225726 | 0.026827877460275387 | -1.586844781180443 | shared_energy_regression:23 |
| always_shared_energy_regression | linear | 8 | 19 | 20.921955460500982 | 0.03086322039306108 | -1.906250041111643 | shared_energy_regression:19 |
| always_shared_energy_regression | linear | 16 | 18 | 18.82189937966654 | 0.13353966957143293 | -1.9501607917837827 | shared_energy_regression:18 |
| always_shared_energy_regression | stage65_adapter | 4 | 23 | 21.00675348675428 | -0.15380783202852794 | -1.4712990167476079 | shared_energy_regression:23 |
| always_shared_energy_regression | stage65_adapter | 8 | 19 | 20.778146373867948 | -0.19199111803343827 | -1.7131768589366094 | shared_energy_regression:19 |
| always_shared_energy_regression | stage65_adapter | 16 | 18 | 18.7456227991363 | -0.016197329840340835 | -1.8259738389774267 | shared_energy_regression:18 |
| always_shared_topk_bce | linear | 4 | 23 | 20.860432826036206 | -0.08991556072924042 | -1.7035882193699587 | shared_topk_bce:23 |
| always_shared_topk_bce | linear | 8 | 19 | 20.783688936210087 | -0.10740330389783524 | -2.04451656540254 | shared_topk_bce:19 |
| always_shared_topk_bce | linear | 16 | 18 | 18.725653414083524 | 0.03729370398841593 | -2.0464067573667992 | shared_topk_bce:18 |
| always_shared_topk_bce | stage65_adapter | 4 | 23 | 20.97239745724654 | -0.18816386153627007 | -1.5056550462553502 | shared_topk_bce:23 |
| always_shared_topk_bce | stage65_adapter | 8 | 19 | 20.770397528802146 | -0.19973996309924674 | -1.720925704002418 | shared_topk_bce:19 |
| always_shared_topk_bce | stage65_adapter | 16 | 18 | 18.729409551728835 | -0.032410577247806396 | -1.8421870863848921 | shared_topk_bce:18 |
| endpoint_only | linear | 4 | 23 | 20.95034838676544 | 0.0 | -1.6136726586407186 | endpoint_diff_baseline:23 |
| endpoint_only | linear | 8 | 19 | 20.89109224010793 | 0.0 | -1.9371132615047044 | endpoint_diff_baseline:19 |
| endpoint_only | linear | 16 | 18 | 18.688359710095106 | 0.0 | -2.0837004613552157 | endpoint_diff_baseline:18 |
| endpoint_only | stage65_adapter | 4 | 23 | 21.16056131878281 | 0.0 | -1.31749118471908 | endpoint_diff_baseline:23 |
| endpoint_only | stage65_adapter | 8 | 19 | 20.970137491901387 | 0.0 | -1.5211857409031715 | endpoint_diff_baseline:19 |
| endpoint_only | stage65_adapter | 16 | 18 | 18.761820128976645 | 0.0 | -1.8097765091370859 | endpoint_diff_baseline:18 |
| group_best_mean_psnr | linear | 4 | 23 | 20.977176264225726 | 0.026827877460275387 | -1.586844781180443 | shared_energy_regression:23 |
| group_best_mean_psnr | linear | 8 | 19 | 20.921955460500982 | 0.03086322039306108 | -1.906250041111643 | shared_energy_regression:19 |
| group_best_mean_psnr | linear | 16 | 18 | 18.82189937966654 | 0.13353966957143293 | -1.9501607917837827 | shared_energy_regression:18 |
| group_best_mean_psnr | stage65_adapter | 4 | 23 | 21.16056131878281 | 0.0 | -1.31749118471908 | endpoint_diff_baseline:23 |
| group_best_mean_psnr | stage65_adapter | 8 | 19 | 20.970137491901387 | 0.0 | -1.5211857409031715 | endpoint_diff_baseline:19 |
| group_best_mean_psnr | stage65_adapter | 16 | 18 | 18.761820128976645 | 0.0 | -1.8097765091370859 | endpoint_diff_baseline:18 |
| oracle_task_best | linear | 4 | 23 | 21.019524528230413 | 0.0691761414649643 | -1.544496517175754 | endpoint_diff_baseline:8;shared_energy_regression:13;shared_topk_bce:2 |
| oracle_task_best | linear | 8 | 19 | 20.992289288396925 | 0.10119704828900276 | -1.835916213215702 | endpoint_diff_baseline:9;shared_energy_regression:7;shared_topk_bce:3 |
| oracle_task_best | linear | 16 | 18 | 18.849605159299436 | 0.16124544920432715 | -1.922455012150888 | endpoint_diff_baseline:3;shared_energy_regression:10;shared_topk_bce:5 |
| oracle_task_best | stage65_adapter | 4 | 23 | 21.19972439314594 | 0.039163074363127645 | -1.2783281103559523 | endpoint_diff_baseline:16;shared_energy_regression:5;shared_topk_bce:2 |
| oracle_task_best | stage65_adapter | 8 | 19 | 21.047765403031338 | 0.0776279111299487 | -1.4435578297732223 | endpoint_diff_baseline:12;shared_energy_regression:3;shared_topk_bce:4 |
| oracle_task_best | stage65_adapter | 16 | 18 | 18.85292506011634 | 0.0911049311396992 | -1.7186715779973865 | endpoint_diff_baseline:7;shared_energy_regression:6;shared_topk_bce:5 |

## Notes

- `group_best_mean_psnr` is a fixed group policy selected from rendered validation rows.
- `oracle_task_best` uses per-task rendered PSNR and is only an upper bound, not deployable.
- All candidates still use teacher residual values at selected indices; residual value prediction remains unresolved.
