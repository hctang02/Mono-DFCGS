# Stage102 Group-Specific Selector Heads

Date: 2026-06-28

## Goal

Test whether per-`base_method x reference_gap` selector heads improve residual top10 index selection over the shared selector.

## Implementation

Added:

```text
scripts/run_stage102_group_specific_selector_heads.py
```

The script uses Stage100 base decoder-available features only. It trains shared selectors and group-specific selectors for two objectives:

- `topk_bce`
- `energy_regression`

No rendering, checkpoint, or heavy tensor is saved.

## Run

GPU check was performed before execution. GPU0 was busy, GPU1 had a small existing Python process, and GPU2 was idle. Syntax check:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage102_group_specific_selector_heads.py
```

Full run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage102_group_specific_selector_heads.py
```

## Outputs

```text
experiments/stage102_group_specific_selector_heads/stage102_group_specific_selector_rows.csv
experiments/stage102_group_specific_selector_heads/stage102_group_specific_selector_summary.csv
experiments/stage102_group_specific_selector_heads/stage102_group_specific_selector_comparison.csv
experiments/stage102_group_specific_selector_heads/stage102_group_specific_selector_train_log.csv
experiments/stage102_group_specific_selector_heads/stage102_group_specific_selector_summary.json
experiments/stage102_group_specific_selector_heads/stage102_group_specific_selector_report.md
```

## Configuration

| item | value |
|---|---:|
| train tasks | 96 |
| eval tasks | 60 |
| train examples | 589824 |
| base feature dim | 67 |
| keep fraction | 0.1 |
| train steps/model | 300 |

## Results

| base | gap | endpoint recall | best shared | shared recall | best group | group recall | group-shared | group relative |
|---|---:|---:|---|---:|---|---:|---:|---:|
| linear | 4 | 0.2578457370400429 | energy_regression | 0.29237517520137457 | energy_regression | 0.28957911483619525 | -0.0027960603651793203 | 0.4642657028592151 |
| linear | 8 | 0.27749040722846985 | energy_regression | 0.3230782339447423 | energy_regression | 0.3191431258854113 | -0.0039351080593309495 | 0.47870409018114995 |
| linear | 16 | 0.2058291733264923 | topk_bce | 0.25067590466803974 | energy_regression | 0.24646301691730818 | -0.004212887750731559 | 0.4052088227536943 |
| stage65_adapter | 4 | 0.13412074485550757 | topk_bce | 0.1617175087980602 | energy_regression | 0.1613874824150749 | -0.00033002638298532117 | 0.686178427675496 |
| stage65_adapter | 8 | 0.13563174087750285 | topk_bce | 0.17034897247427389 | energy_regression | 0.17020083375667272 | -0.00014813871760116504 | 0.6869090183785087 |
| stage65_adapter | 16 | 0.11971673224535254 | energy_regression | 0.15471376478672028 | energy_regression | 0.15460273540682262 | -0.0001110293798976536 | 0.6332634886105856 |

## Conclusion

- Group-specific selector heads do not improve over the shared selector.
- Every group-specific best energy recall is slightly lower than the shared best energy recall.
- The selection bottleneck is not explained by mixed-group shared-head capacity.
- Next steps should prioritize rendered validation of the current best shared selector or structured Gaussian-neighborhood/coordinate context.
