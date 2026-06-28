# Stage110 Broader Rendered Selector Labels

Date: 2026-06-28

## Goal

Broaden rendered selector labels from the Stage103 60-task eval set to 240 eval tasks, then re-run render-energy mismatch and render-aware policy diagnostics on the broader rows.

## Implementation

Updated these scripts with stage/output-prefix/report-title parameters while preserving their default Stage103/104/105 behavior:

```text
scripts/run_stage103_broader_rendered_selector_validation.py
scripts/run_stage104_render_energy_selector_mismatch_diagnostic.py
scripts/run_stage105_render_aware_selector_policy_preflight.py
```

Stage105 policy preflight also now supports an optional `stage106_fixed_group_policy` candidate loaded from the Stage106 policy JSON.

## Run

GPU checks were performed before execution. GPU2 was idle and used for rendered validation.

Syntax checks:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage103_broader_rendered_selector_validation.py scripts/run_stage105_render_aware_selector_policy_preflight.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage104_render_energy_selector_mismatch_diagnostic.py
```

Rendered validation:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage103_broader_rendered_selector_validation.py --stage 110 --mode "broader rendered selector labels" --summary_root experiments/stage110_broader_rendered_selector_labels --output_prefix stage110_broader_rendered_selector_labels --report_title "Stage110 Broader Rendered Selector Labels" --max_eval_tasks 240
```

Mismatch diagnostic:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage104_render_energy_selector_mismatch_diagnostic.py --stage 110 --mode "broader render-energy selector mismatch diagnostic" --stage103_rows experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_rows.csv --summary_root experiments/stage110_broader_rendered_selector_labels --output_prefix stage110_render_energy_mismatch --report_title "Stage110 Broader Render-Energy Selector Mismatch Diagnostic"
```

Policy preflight:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage105_render_aware_selector_policy_preflight.py --stage 110 --mode "broader render-aware selector policy preflight" --stage103_rows experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_rows.csv --summary_root experiments/stage110_broader_rendered_selector_labels --output_prefix stage110_broader_render_aware_policy --report_title "Stage110 Broader Render-Aware Selector Policy Preflight" --policies endpoint_only always_shared_energy_regression always_shared_topk_bce stage106_fixed_group_policy group_best_mean_psnr oracle_task_best
```

## Outputs

```text
experiments/stage110_broader_rendered_selector_labels/
```

Important files:

```text
experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_rows.csv
experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_summary.csv
experiments/stage110_broader_rendered_selector_labels/stage110_broader_rendered_selector_labels_report.md
experiments/stage110_broader_rendered_selector_labels/stage110_render_energy_mismatch_report.md
experiments/stage110_broader_rendered_selector_labels/stage110_broader_render_aware_policy_report.md
```

Output size is about `1.8M`; no checkpoint, anchor tensor, payload, or heavy tensor is saved.

## Configuration

| item | value |
|---|---:|
| train tasks | 96 |
| eval tasks | 240 |
| train examples | 589824 |
| policy task count | 480 |
| keep fraction | 0.1 |
| side bits | 6 |

## Policy Results

| policy | selected PSNR | gain vs endpoint | selections |
|---|---:|---:|---|
| endpoint_only | 20.3212149854921 | 0.0 | endpoint_diff_baseline:480 |
| stage106_fixed_group_policy | 20.322996715243953 | 0.0017817297518578745 | endpoint_diff_baseline:240;shared_energy_regression:240 |
| group_best_mean_psnr | 20.327046871072337 | 0.005831885580240304 | endpoint_diff_baseline:323;shared_energy_regression:157 |
| oracle_task_best | 20.382843220952523 | 0.06162823546041816 | endpoint_diff_baseline:278;shared_energy_regression:121;shared_topk_bce:81 |

## Group Choices

| base | gap | selected candidate | gain vs endpoint |
|---|---:|---|---:|
| linear | 4 | endpoint_diff_baseline | 0.0 |
| linear | 8 | shared_energy_regression | 0.009307271828451036 |
| linear | 16 | shared_energy_regression | 0.026461930821380264 |
| stage65_adapter | 4 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 8 | endpoint_diff_baseline | 0.0 |
| stage65_adapter | 16 | endpoint_diff_baseline | 0.0 |

## Mismatch Diagnostic

For `shared_energy_regression`, residual-energy recall improves in every group but rendered PSNR is not consistently better:

| base | gap | energy delta | PSNR delta | energy-up PSNR-down |
|---|---:|---:|---:|---:|
| linear | 4 | 0.031175289392830378 | -0.023422587923175493 | 41 |
| linear | 8 | 0.03677648032390619 | 0.009307271828446134 | 33 |
| linear | 16 | 0.03649166226387024 | 0.026461930821385912 | 25 |
| stage65_adapter | 4 | 0.024832078311816757 | -0.22540383262011054 | 66 |
| stage65_adapter | 8 | 0.03036606472127045 | -0.19372914974113 | 54 |
| stage65_adapter | 16 | 0.032561146725828834 | -0.1363193230658923 | 49 |

## Conclusion

- Stage106 fixed group policy remains slightly positive on broader rows, but its gain shrinks to `+0.0017817297518578745 dB`.
- The broader group-best pattern differs from Stage106 by using endpoint selection for linear gap4.
- Stage110 group-best improves endpoint by `+0.005831885580240304 dB`, but it is selected on the same rows and should not be frozen without Stage111/held-out validation.
- Learned selectors still degrade Stage65 adapter groups, and linear gap4 is also not robust.
- Next step should be Stage111 broader switch predictor using Stage110 labels and comparing against endpoint, Stage106 fixed, Stage110 group-best, and oracle task best.
