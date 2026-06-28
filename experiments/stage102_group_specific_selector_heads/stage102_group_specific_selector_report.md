# Stage102 Group-Specific Selector Heads

## Configuration

- train tasks: `96`
- eval tasks: `60`
- train examples: `589824`
- base feature dim: `67`
- keep fraction: `0.1`
- objectives: `topk_bce, energy_regression`
- no rendering, checkpoint, or heavy tensor output

## Group Comparison

| base | gap | endpoint recall | best shared | shared recall | best group | group recall | group-shared | group relative | group precision |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| linear | 4 | 0.2578457370400429 | energy_regression | 0.29237517520137457 | energy_regression | 0.28957911483619525 | -0.0027960603651793203 | 0.4642657028592151 | 0.342235016434089 |
| linear | 8 | 0.27749040722846985 | energy_regression | 0.3230782339447423 | energy_regression | 0.3191431258854113 | -0.0039351080593309495 | 0.47870409018114995 | 0.36073907190247584 |
| linear | 16 | 0.2058291733264923 | topk_bce | 0.25067590466803974 | energy_regression | 0.24646301691730818 | -0.004212887750731559 | 0.4052088227536943 | 0.25529028594286907 |
| stage65_adapter | 4 | 0.13412074485550757 | topk_bce | 0.1617175087980602 | energy_regression | 0.1613874824150749 | -0.00033002638298532117 | 0.686178427675496 | 0.40539998010448786 |
| stage65_adapter | 8 | 0.13563174087750285 | topk_bce | 0.17034897247427389 | energy_regression | 0.17020083375667272 | -0.00014813871760116504 | 0.6869090183785087 | 0.41855384644709137 |
| stage65_adapter | 16 | 0.11971673224535254 | energy_regression | 0.15471376478672028 | energy_regression | 0.15460273540682262 | -0.0001110293798976536 | 0.6332634886105856 | 0.3378850882872939 |

## Summary

| scope | group | objective | candidate | base | gap | tasks | precision@keep | energy recall | oracle recall | relative recall |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| endpoint_only | linear__gap4 | endpoint_diff_baseline | endpoint_diff_baseline | linear | 4 | 23 | 0.3100568529056466 | 0.2578457370400429 | 0.6033855663693469 | 0.4116023273571678 |
| endpoint_only | linear__gap8 | endpoint_diff_baseline | endpoint_diff_baseline | linear | 8 | 19 | 0.3119484817511157 | 0.27749040722846985 | 0.6419753777353387 | 0.4117828874211562 |
| endpoint_only | linear__gap16 | endpoint_diff_baseline | endpoint_diff_baseline | linear | 16 | 18 | 0.20097063968165052 | 0.2058291733264923 | 0.5684697247213788 | 0.3390035058061282 |
| endpoint_only | stage65_adapter__gap4 | endpoint_diff_baseline | endpoint_diff_baseline | stage65_adapter | 4 | 23 | 0.29103069072184357 | 0.13412074485550757 | 0.23996665620285532 | 0.5757957541424296 |
| endpoint_only | stage65_adapter__gap8 | endpoint_diff_baseline | endpoint_diff_baseline | stage65_adapter | 8 | 19 | 0.27700831074463694 | 0.13563174087750285 | 0.2529745478379099 | 0.5572832179696936 |
| endpoint_only | stage65_adapter__gap16 | endpoint_diff_baseline | endpoint_diff_baseline | stage65_adapter | 16 | 18 | 0.1901941259081165 | 0.11971673224535254 | 0.2512796148657799 | 0.49458976917796665 |
| group_specific | linear__gap4 | energy_regression | mlp_selector | linear | 4 | 23 | 0.342235016434089 | 0.28957911483619525 | 0.6033855663693469 | 0.4642657028592151 |
| group_specific | linear__gap4 | topk_bce | mlp_selector | linear | 4 | 23 | 0.32478945022043976 | 0.28498703241348267 | 0.6033855663693469 | 0.45689631804176 |
| group_specific | linear__gap8 | energy_regression | mlp_selector | linear | 8 | 19 | 0.36073907190247584 | 0.3191431258854113 | 0.6419753777353387 | 0.47870409018114995 |
| group_specific | linear__gap8 | topk_bce | mlp_selector | linear | 8 | 19 | 0.34557500560032695 | 0.31702931303727 | 0.6419753777353387 | 0.478698531263753 |
| group_specific | linear__gap16 | energy_regression | mlp_selector | linear | 16 | 18 | 0.25529028594286907 | 0.24646301691730818 | 0.5684697247213788 | 0.4052088227536943 |
| group_specific | linear__gap16 | topk_bce | mlp_selector | linear | 16 | 18 | 0.24061011709272861 | 0.23468620764712492 | 0.5684697247213788 | 0.39048904760016334 |
| group_specific | stage65_adapter__gap4 | energy_regression | mlp_selector | stage65_adapter | 4 | 23 | 0.40539998010448786 | 0.1613874824150749 | 0.23996665620285532 | 0.686178427675496 |
| group_specific | stage65_adapter__gap4 | topk_bce | mlp_selector | stage65_adapter | 4 | 23 | 0.4064026034396628 | 0.16085052684597348 | 0.23996665620285532 | 0.6839258593061696 |
| group_specific | stage65_adapter__gap8 | energy_regression | mlp_selector | stage65_adapter | 8 | 19 | 0.41855384644709137 | 0.17020083375667272 | 0.2529745478379099 | 0.6869090183785087 |
| group_specific | stage65_adapter__gap8 | topk_bce | mlp_selector | stage65_adapter | 8 | 19 | 0.4173258707711571 | 0.1695655258862596 | 0.2529745478379099 | 0.6846243143081665 |
| group_specific | stage65_adapter__gap16 | energy_regression | mlp_selector | stage65_adapter | 16 | 18 | 0.3378850882872939 | 0.15460273540682262 | 0.2512796148657799 | 0.6332634886105856 |
| group_specific | stage65_adapter__gap16 | topk_bce | mlp_selector | stage65_adapter | 16 | 18 | 0.3329263842768139 | 0.1538888874153296 | 0.2512796148657799 | 0.6287363817294439 |
| shared | all_groups | energy_regression | mlp_selector | linear | 4 | 23 | 0.3413385528585185 | 0.29237517520137457 | 0.6033855663693469 | 0.46866350588591205 |
| shared | all_groups | topk_bce | mlp_selector | linear | 4 | 23 | 0.32778550878815027 | 0.2915198809426764 | 0.6033855663693469 | 0.46782151901203656 |
| shared | all_groups | energy_regression | mlp_selector | linear | 8 | 19 | 0.3649513094048751 | 0.3230782339447423 | 0.6419753777353387 | 0.4839692037356527 |
| shared | all_groups | topk_bce | mlp_selector | linear | 8 | 19 | 0.35094383045246724 | 0.32079512665146276 | 0.6419753777353387 | 0.4809588648770985 |
| shared | all_groups | energy_regression | mlp_selector | linear | 16 | 18 | 0.255636946660363 | 0.2495752258433236 | 0.5684697247213788 | 0.41003381212552387 |
| shared | all_groups | topk_bce | mlp_selector | linear | 16 | 18 | 0.2554410095844004 | 0.25067590466803974 | 0.5684697247213788 | 0.4148368760943413 |
| shared | all_groups | energy_regression | mlp_selector | stage65_adapter | 4 | 23 | 0.4008115277342174 | 0.16083620812581934 | 0.23996665620285532 | 0.683561042599056 |
| shared | all_groups | topk_bce | mlp_selector | stage65_adapter | 4 | 23 | 0.4125008809825648 | 0.1617175087980602 | 0.23996665620285532 | 0.6878124009007993 |
| shared | all_groups | energy_regression | mlp_selector | stage65_adapter | 8 | 19 | 0.41325641619531733 | 0.1695171743631363 | 0.2529745478379099 | 0.6841765124546854 |
| shared | all_groups | topk_bce | mlp_selector | stage65_adapter | 8 | 19 | 0.4220949819213466 | 0.17034897247427389 | 0.2529745478379099 | 0.6874244464071173 |
| shared | all_groups | energy_regression | mlp_selector | stage65_adapter | 16 | 18 | 0.33554892314391005 | 0.15471376478672028 | 0.2512796148657799 | 0.6332139886087842 |
| shared | all_groups | topk_bce | mlp_selector | stage65_adapter | 16 | 18 | 0.33260987926688457 | 0.15361030275623003 | 0.2512796148657799 | 0.6280805104308658 |

## Notes

- Group-specific heads are trained per base method and reference gap.
- Features are the Stage100 base decoder-available features only.
- Target dense anchors are used only for offline labels/metrics.
