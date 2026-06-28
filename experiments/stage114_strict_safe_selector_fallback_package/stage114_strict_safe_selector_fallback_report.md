# Stage114 Strict-Safe Selector Fallback Package

## Policy

- name: `strict_safe_endpoint_selector_v1`
- type: `fixed_candidate_selector`
- fixed selected candidate: `endpoint_diff_baseline`
- decoder inputs: none beyond normal endpoint anchors already available to the decoder
- no target dense anchor, target residual, rendered PSNR, target RGB, or oracle labels are decoder inputs

## Validation Summary

| metric | value |
|---|---:|
| task count | 480 |
| selected PSNR | 20.3212149854921 |
| endpoint PSNR | 20.3212149854921 |
| gain vs endpoint | 0.0 |
| min fold gain | 0.0 |
| min group gain | 0.0 |
| min fold-group gain | 0.0 |
| aggregate safe | 1 |
| fold-group safe | 1 |

## Policy Comparison

| policy | selected PSNR | gain | min group gain | min fold-group gain | aggregate safe | fold-group safe | decision | reason |
|---|---:|---:|---:|---:|---:|---:|---|---|
| endpoint_only | 20.3212149854921 | 0.0 | 0.0 | 0.0 | 1 | 1 | package | strict-safe fallback selected by user |
| stage106_fixed_group_policy | 20.322996715243978 | 0.0017817297518578745 | -0.023422587923175493 | -0.06427740265352 | 0 | 0 | reject | linear gap4 aggregate regression |
| stage112_group_switch_v2 | 20.32704687107235 | 0.005831885580240304 | 0.0 | -0.03366017781158855 | 1 | 0 | reject_as_final | fold-group regression under Stage113 strict criterion |
| score_stat_mlp_cv | 20.33325220653739 | 0.012037221045259486 | -0.00797889356792674 | -0.039626239935121585 | 0 | 0 | reject | Stage65 adapter gap4 aggregate regression |

## Notes

- This package intentionally gives up Stage112's average gain to guarantee strict no-regression under the Stage113 diagnostic.
- It is only an index-selection policy; residual values in Stage110/111/113 diagnostics are still teacher values.
- It should be used as the default safe selector for deterministic-index side-info codec work unless broader rendered validation re-qualifies Stage112 v2.
