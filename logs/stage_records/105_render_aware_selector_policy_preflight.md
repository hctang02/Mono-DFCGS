# Stage105 Render-Aware Selector Policy Preflight

Date: 2026-06-28

## Goal

Check whether render-aware selector switching has useful headroom before training a deployable task-level policy.

## Implementation

Added:

```text
scripts/run_stage105_render_aware_selector_policy_preflight.py
```

The script reads Stage103 rendered per-task rows and compares endpoint-only, always-learned, group-level, and oracle per-task policies without rerendering.

## Run

GPU check was performed before execution. Stage105 is a CPU summary script. Syntax check:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage105_render_aware_selector_policy_preflight.py
```

Preflight run:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage105_render_aware_selector_policy_preflight.py
```

## Outputs

```text
experiments/stage105_render_aware_selector_policy_preflight/stage105_render_aware_policy_rows.csv
experiments/stage105_render_aware_selector_policy_preflight/stage105_render_aware_policy_group_summary.csv
experiments/stage105_render_aware_selector_policy_preflight/stage105_render_aware_policy_overall_summary.csv
experiments/stage105_render_aware_selector_policy_preflight/stage105_render_aware_policy_group_choices.csv
experiments/stage105_render_aware_selector_policy_preflight/stage105_render_aware_policy_summary.json
experiments/stage105_render_aware_selector_policy_preflight/stage105_render_aware_policy_report.md
```

## Results

| policy | selected PSNR | endpoint PSNR | gain vs endpoint | teacher PSNR | selections |
|---|---:|---:|---:|---:|---|
| endpoint_only | 20.316812710325646 | 20.316812710325646 | 0.0 | 22.010204667924707 | endpoint_diff_baseline:120 |
| always_shared_energy_regression | 20.284564319533338 | 20.316812710325646 | -0.03224839079231098 | 22.010204667924707 | shared_energy_regression:120 |
| always_shared_topk_bce | 20.21561560612798 | 20.316812710325646 | -0.10119710419766939 | 22.010204667924707 | shared_topk_bce:120 |
| group_best_mean_psnr | 20.346872347170144 | 20.316812710325646 | 0.030059636844502392 | 22.010204667924707 | endpoint_diff_baseline:60;shared_energy_regression:60 |
| oracle_task_best | 20.403744235652294 | 20.316812710325646 | 0.08693152532665556 | 22.010204667924707 | endpoint_diff_baseline:55;shared_energy_regression:44;shared_topk_bce:21 |

## Group Choices

| base | gap | selected candidate | gain vs endpoint |
|---|---:|---|---:|
| linear | 4 | shared_energy_regression | 0.026827877460277705 |
| linear | 8 | shared_energy_regression | 0.030863220393055002 |
| linear | 16 | shared_energy_regression | 0.1335396695714337 |
| stage65_adapter | 4 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 8 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 16 | endpoint_diff_baseline | 0.0 |

## Conclusion

- Always using the learned selector is worse than endpoint-only overall.
- A simple group policy gives a small positive gain by using learned selection only for the linear base.
- Per-task oracle switching gives a larger but still modest upper bound and is not deployable.
- Next step should train or preflight a deployable task-level switch predictor, not residual value prediction yet.
