# Stage112 Broader Group Switch Policy Package

Date: 2026-06-29

## Goal

Package a conservative decoder-side selector-switch policy candidate from the Stage110 broader group-best pattern, rather than packaging the Stage111 learned switch that still regresses Stage65 adapter gap4.

## Implementation

Added:

```text
scripts/run_stage112_package_broader_group_switch_policy.py
```

The script reads Stage110 policy summaries/group choices, Stage111 broader switch predictor summary, and the Stage106 policy JSON. It writes a small JSON/CSV/Markdown package and does not save checkpoints, anchor tensors, payload tensors, or other heavy artifacts.

## Run

GPU check was performed before execution. GPU2 was idle; Stage112 is CPU packaging, so no CUDA computation was needed.

Syntax check:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage112_package_broader_group_switch_policy.py
```

Package run:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage112_package_broader_group_switch_policy.py
```

## Outputs

```text
experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy.json
experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy_summary.csv
experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy_table.csv
experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy_comparison.csv
experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy_report.md
```

## Packaged Policy

| item | value |
|---|---|
| policy name | `render_aware_group_switch_v2` |
| policy type | `metadata_group_switch` |
| decoder inputs | `base_method`, `reference_gap` |
| fallback | `endpoint_diff_baseline` |
| forbidden inputs | `target_dense_anchor`, `target_residual`, `rendered_psnr`, `oracle_task_label`, `target_rgb` |

## Selection Table

| base | gap | selected candidate | gain vs endpoint |
|---|---:|---|---:|
| linear | 4 | endpoint_diff_baseline | 0.0 |
| linear | 8 | shared_energy_regression | 0.009307271828451036 |
| linear | 16 | shared_energy_regression | 0.026461930821380264 |
| stage65_adapter | 4 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 8 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 16 | endpoint_diff_baseline | 0.0 |

## Results

| policy | selected PSNR | gain vs endpoint | note |
|---|---:|---:|---|
| endpoint_only | 20.3212149854921 | 0.0 | baseline |
| stage106_fixed_group_policy | 20.322996715243953 | 0.0017817297518578745 | previous packaged policy |
| render_aware_group_switch_v2 | 20.327046871072337 | 0.005831885580240304 | packaged conservative policy |
| score_stat_mlp_cv | 20.33325220653739 | 0.012037221045259486 | not packaged; adapter gap4 regression |
| oracle_task_best | 20.382843220952523 | 0.06162823546041816 | upper bound, not deployable |

Validation summary:

| metric | value |
|---|---:|
| task count | 480 |
| endpoint PSNR | 20.3212149854921 |
| policy PSNR | 20.327046871072337 |
| gain vs endpoint | 0.005831885580240304 |
| teacher oracle PSNR | 22.077800340877268 |
| gap to teacher oracle | -1.750753469804888 |

## Conclusion

- Stage112 packages the conservative Stage110 broader group-best pattern as `render_aware_group_switch_v2`.
- The policy is decoder-side safe because it uses only `base_method` and `reference_gap`.
- Stage111 learned switch is not packaged because Stage65 adapter gap4 still regresses.
- The package only selects index-selection candidates; Stage110/111 validation still used teacher residual values at selected indices.
- Stage113 should perform held-out validation before this policy is treated as final.
