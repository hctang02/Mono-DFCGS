# Stage112 Broader Group Switch Policy Package

## Policy

- name: `render_aware_group_switch_v2`
- type: `metadata_group_switch`
- decoder inputs: `base_method`, `reference_gap`
- fallback: `endpoint_diff_baseline`
- no target dense anchor, target residual, rendered PSNR, target RGB, or oracle labels are decoder inputs

## Selection Table

| base | gap | selected candidate | validation gain vs endpoint |
|---|---:|---|---:|
| linear | 4 | endpoint_diff_baseline | 0.0 |
| linear | 8 | shared_energy_regression | 0.009307271828451036 |
| linear | 16 | shared_energy_regression | 0.026461930821380264 |
| stage65_adapter | 4 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 8 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 16 | endpoint_diff_baseline | 0.0 |

## Validation Summary

| metric | value |
|---|---:|
| task count | 480 |
| endpoint PSNR | 20.3212149854921 |
| policy PSNR | 20.327046871072337 |
| gain vs endpoint | 0.005831885580240304 |
| teacher oracle PSNR | 22.077800340877268 |
| gap to teacher | -1.750753469804888 |

## Policy Comparison

| policy | selected PSNR | gain vs endpoint | note |
|---|---:|---:|---|
| endpoint_only | 20.3212149854921 | 0.0 | baseline |
| stage106_fixed_group_policy | 20.322996715243953 | 0.0017817297518578745 | previous packaged policy |
| render_aware_group_switch_v2 | 20.327046871072337 | 0.005831885580240304 | packaged conservative policy |
| score_stat_mlp_cv | 20.33325220653739 | 0.012037221045259486 | not packaged; adapter gap4 regression |
| oracle_task_best | 20.382843220952523 | 0.06162823546041816 | upper bound, not deployable |

## Notes

- This package only selects the index-selection candidate; it does not predict residual values.
- Stage110/111 validation candidates still used teacher residual values at selected indices.
- Side-info rate is unchanged across candidates because all use the same q6 top10 residual payload shape.
- The learned Stage111 switch is not packaged because it regresses Stage65 adapter gap4.
- This policy should be validated in Stage113 before being treated as final.
