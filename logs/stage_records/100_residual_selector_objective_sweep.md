# Stage100 Residual Selector Objective Sweep

Date: 2026-06-28

## Goal

Improve residual side-info index selection after Stage99 showed a persistent gap between MLP-selected top10 and teacher-oracle top10 rendering.

## Implementation

Added:

```text
scripts/run_stage100_residual_selector_objective_sweep.py
```

The script reuses Stage97 tasks and Stage98 decoder-available features. It compares four training objectives against endpoint-difference selection:

- `topk_bce`
- `energy_weighted_bce`
- `hybrid_bce_energy`
- `energy_regression`

No rendering, checkpoint, or heavy tensor is saved.

## Run

GPU check was performed before execution. GPU0 was busy and GPU1 was idle. Syntax check:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage100_residual_selector_objective_sweep.py
```

Full run:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage100_residual_selector_objective_sweep.py
```

## Outputs

```text
experiments/stage100_residual_selector_objective_sweep/stage100_residual_selector_objective_rows.csv
experiments/stage100_residual_selector_objective_sweep/stage100_residual_selector_objective_summary.csv
experiments/stage100_residual_selector_objective_sweep/stage100_residual_selector_objective_train_log.csv
experiments/stage100_residual_selector_objective_sweep/stage100_residual_selector_objective_summary.json
experiments/stage100_residual_selector_objective_sweep/stage100_residual_selector_objective_report.md
```

## Configuration

| item | value |
|---|---:|
| train tasks | 96 |
| eval tasks | 60 |
| train examples | 589824 |
| keep fraction | 0.1 |
| train steps/objective | 300 |

## Results

| base | gap | endpoint energy recall | best objective | best energy recall | gain vs endpoint | best relative recall | precision@keep |
|---|---:|---:|---|---:|---:|---:|---:|
| linear | 4 | 0.2578457370400429 | energy_regression | 0.29329851334509643 | +0.03545277630505353 | 0.4703915430151898 | 0.34324942593989166 |
| linear | 8 | 0.27749040722846985 | energy_regression | 0.32311024320752996 | +0.04561983597906011 | 0.4835715215457113 | 0.36375188905941813 |
| linear | 16 | 0.2058291733264923 | topk_bce | 0.2517792292767101 | +0.045950055950217806 | 0.41649167074097526 | 0.2579128210329347 |
| stage65_adapter | 4 | 0.13412074485550757 | topk_bce | 0.16204810336880063 | +0.02792735851329306 | 0.6889743960422018 | 0.4132322079461554 |
| stage65_adapter | 8 | 0.13563174087750285 | energy_weighted_bce | 0.17072841211369164 | +0.03509667123618879 | 0.6882316815225702 | 0.4208384501306634 |
| stage65_adapter | 16 | 0.11971673224535254 | energy_regression | 0.15512513783242968 | +0.03540840558707714 | 0.6353713091876771 | 0.3373726398373644 |

## Conclusion

- Learned selectors outperform endpoint-difference selection on every base/gap group in the broader 60-task eval.
- The tested objectives are close; no single objective clearly solves the Stage99 selection gap.
- Best learned relative oracle recall remains about `0.42-0.69`, so index selection is still a bottleneck.
- The next step should improve model/features or train scale before residual value prediction.
