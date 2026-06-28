# Stage106 Render-Aware Group Policy Package

## Policy

- name: `render_aware_group_switch_v1`
- type: `metadata_group_switch`
- decoder inputs: `base_method`, `reference_gap`
- fallback: `endpoint_diff_baseline`
- no target residual, rendered PSNR, or oracle task labels are decoder inputs

## Selection Table

| base | gap | selected candidate | validation gain vs endpoint |
|---|---:|---|---:|
| linear | 4 | shared_energy_regression | 0.026827877460277705 |
| linear | 8 | shared_energy_regression | 0.030863220393055002 |
| linear | 16 | shared_energy_regression | 0.1335396695714337 |
| stage65_adapter | 4 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 8 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 16 | endpoint_diff_baseline | 0.0 |

## Validation Summary

| metric | value |
|---|---:|
| task count | 120 |
| endpoint PSNR | 20.316812710325646 |
| policy PSNR | 20.346872347170144 |
| gain vs endpoint | 0.030059636844502392 |
| teacher oracle PSNR | 22.010204667924707 |
| gap to teacher | -1.6633323207545507 |

## Notes

- This package only selects the index-selection candidate; it does not predict residual values.
- All Stage105 validation candidates still used teacher residual values at selected indices.
- Side-info rate is unchanged across candidates because all use the same q6 top10 residual payload shape.
