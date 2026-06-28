# Stage106 Render-Aware Group Policy Package

Date: 2026-06-28

## Goal

Package Stage105's render-aware group switch as a decoder-side metadata policy baseline.

## Implementation

Added:

```text
scripts/run_stage106_render_aware_group_policy_package.py
```

The policy uses only `base_method` and `reference_gap` as decoder inputs. It does not use target residuals, target dense anchors, rendered PSNR, or oracle task labels at decode time.

## Run

GPU check was performed before execution. Stage106 is a CPU packaging script. Syntax check:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage106_render_aware_group_policy_package.py
```

Package run:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage106_render_aware_group_policy_package.py
```

## Outputs

```text
experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy.json
experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy_summary.csv
experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy_table.csv
experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy_report.md
```

## Policy

| base | gap | selected candidate |
|---|---:|---|
| linear | 4 | shared_energy_regression |
| linear | 8 | shared_energy_regression |
| linear | 16 | shared_energy_regression |
| stage65_adapter | 4 | endpoint_diff_baseline |
| stage65_adapter | 8 | endpoint_diff_baseline |
| stage65_adapter | 16 | endpoint_diff_baseline |

## Validation Summary

| metric | value |
|---|---:|
| task count | 120 |
| endpoint PSNR | 20.316812710325646 |
| policy PSNR | 20.346872347170144 |
| gain vs endpoint | 0.030059636844502392 |
| teacher oracle PSNR | 22.010204667924707 |
| gap to teacher | -1.6633323207545507 |

## Conclusion

- The packaged group policy is decoder-side metadata-only for index candidate switching.
- It is safer than always using learned selectors because it keeps endpoint selection for Stage65 adapter groups.
- It is not a complete deployable residual codec because residual values remain teacher-derived in validation.
- Next step should compare a learned task-level switch predictor against this group-policy baseline.
