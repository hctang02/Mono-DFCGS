# Stage113 Held-Out Switch Validation

Date: 2026-06-29

## Goal

Validate the Stage112 `render_aware_group_switch_v2` policy on held-out Stage111 fold rows and compare it against endpoint-only, Stage106 fixed policy, train-fold group policy, Stage111 learned switch, and oracle task best.

## Implementation

Added:

```text
scripts/run_stage113_heldout_switch_validation.py
```

The script reuses Stage111 out-of-fold rows, aliases `stage110_group_best_policy` to `stage112_group_switch_v2`, verifies that the aliased rows match the Stage112 package selection table, and writes overall/group/fold/fold-group/sequence/safety summaries.

## Run

GPU check was performed before execution. Stage113 is CPU-only and did not use CUDA.

Syntax check and run:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage113_heldout_switch_validation.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage113_heldout_switch_validation.py
```

## Outputs

```text
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_rows.csv
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_overall_summary.csv
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_group_summary.csv
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_fold_summary.csv
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_fold_group_summary.csv
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_sequence_summary.csv
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_safety_summary.csv
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_summary.json
experiments/stage113_heldout_switch_validation/stage113_heldout_switch_validation_report.md
```

## Configuration

| item | value |
|---|---:|
| held-out unit | Stage111 `stage97_task_id` modulo fold |
| fold count | 5 |
| task-policy rows | 2880 |
| unique tasks per policy | 480 |
| Stage112 alias mismatch count | 0 |

This is a CPU diagnostic over existing rendered rows, not a new sequence-heldout render run.

## Overall Results

| policy | selected PSNR | gain vs endpoint | accuracy | selections |
|---|---:|---:|---:|---|
| endpoint_only | 20.3212149854921 | 0.0 | 0.5791666666666667 | endpoint_diff_baseline:480 |
| stage106_fixed_group_policy | 20.322996715243978 | 0.0017817297518578745 | 0.51875 | endpoint_diff_baseline:240;shared_energy_regression:240 |
| stage112_group_switch_v2 | 20.32704687107235 | 0.005831885580240304 | 0.55625 | endpoint_diff_baseline:323;shared_energy_regression:157 |
| train_fold_group_policy | 20.325313259771452 | 0.00409827427933979 | 0.5541666666666667 | endpoint_diff_baseline:338;shared_energy_regression:142 |
| score_stat_mlp_cv | 20.33325220653739 | 0.012037221045259486 | 0.6041666666666666 | endpoint_diff_baseline:274;shared_energy_regression:130;shared_topk_bce:76 |
| oracle_task_best | 20.382843220952545 | 0.06162823546041816 | 1.0 | endpoint_diff_baseline:278;shared_energy_regression:121;shared_topk_bce:81 |

## Safety Results

| policy | mean gain | min fold gain | neg folds | min group gain | neg groups | min fold-group gain | neg fold-groups | Stage65 gap4 gain | aggregate safe | fold-group safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| endpoint_only | 0.0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 1 | 1 |
| stage106_fixed_group_policy | 0.0017817297518578745 | -0.007288316466769011 | 3 | -0.023422587923175493 | 1 | -0.06427740265352 | 8 | 0.0 | 0 | 0 |
| stage112_group_switch_v2 | 0.005831885580240304 | -0.0059710516126523045 | 1 | 0.0 | 0 | -0.03366017781158855 | 4 | 0.0 | 1 | 0 |
| train_fold_group_policy | 0.00409827427933979 | -0.0059710516126523045 | 1 | -0.0012260626580379986 | 1 | -0.03366017781158855 | 4 | 0.0 | 0 | 0 |
| score_stat_mlp_cv | 0.012037221045259486 | 0.0002971711561230069 | 0 | -0.00797889356792674 | 1 | -0.039626239935121585 | 10 | -0.00797889356792674 | 0 | 0 |

## Conclusion

- Stage112 v2 improves overall PSNR and is aggregate group-safe.
- Stage112 v2 is not fold-group safe if any negative held-out fold-group cell is a blocker.
- Stage111 learned `score_stat_mlp_cv` remains unsafe because Stage65 adapter gap4 regresses.
- Endpoint-only is the only policy that is trivially safe across aggregate, fold, group, and fold-group summaries.
- Do not treat Stage112 v2 as final before either broader held-out rendered validation or a stricter fallback policy decision.
