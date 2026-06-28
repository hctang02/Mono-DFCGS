# Stage114 Strict-Safe Selector Fallback Package

Date: 2026-06-29

## Goal

Package the user-selected strict-safe endpoint-only selector fallback after Stage113 showed that Stage112 v2 is aggregate-safe but not fold-group safe.

## Implementation

Added:

```text
scripts/run_stage114_package_strict_safe_selector_fallback.py
```

The script reads Stage113 overall/safety summaries and the Stage112 policy package, then writes a strict-safe fixed-candidate selector package.

## Run

GPU check was performed before execution. Stage114 is CPU-only and did not use CUDA.

Syntax check and run:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage114_package_strict_safe_selector_fallback.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage114_package_strict_safe_selector_fallback.py
```

## Outputs

```text
experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_policy.json
experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_summary.csv
experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_comparison.csv
experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_report.md
```

## Packaged Policy

| item | value |
|---|---|
| policy name | `strict_safe_endpoint_selector_v1` |
| policy type | `fixed_candidate_selector` |
| selected candidate | `endpoint_diff_baseline` |
| decoder inputs | none beyond decoder-available left/right anchors |
| forbidden inputs | `target_dense_anchor`, `target_residual`, `rendered_psnr`, `oracle_task_label`, `target_rgb` |

## Validation Summary

| metric | value |
|---|---:|
| task count | 480 |
| selected PSNR | 20.3212149854921 |
| gain vs endpoint | 0.0 |
| min fold gain | 0.0 |
| min group gain | 0.0 |
| min fold-group gain | 0.0 |
| aggregate safe | 1 |
| fold-group safe | 1 |

## Comparison

| policy | selected PSNR | gain | min group gain | min fold-group gain | aggregate safe | fold-group safe | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| endpoint_only | 20.3212149854921 | 0.0 | 0.0 | 0.0 | 1 | 1 | package |
| stage106_fixed_group_policy | 20.322996715243978 | 0.0017817297518578745 | -0.023422587923175493 | -0.06427740265352 | 0 | 0 | reject |
| stage112_group_switch_v2 | 20.32704687107235 | 0.005831885580240304 | 0.0 | -0.03366017781158855 | 1 | 0 | reject_as_final |
| score_stat_mlp_cv | 20.33325220653739 | 0.012037221045259486 | -0.00797889356792674 | -0.039626239935121585 | 0 | 0 | reject |

## Conclusion

- Stage114 freezes `strict_safe_endpoint_selector_v1` as the default safe selector.
- The policy sacrifices Stage112's average gain but passes the strict Stage113 fold/group/fold-group no-regression criterion.
- Stage112 remains an aggregate-safe candidate only, not the final strict-safe selector.
- The package still only selects residual indices; residual value prediction remains future work.
- Next step is deterministic-index residual side-info codec using this strict-safe selector.
