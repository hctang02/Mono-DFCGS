# Stage101 Enhanced Selector Feature Sweep

Date: 2026-06-28

## Goal

Test whether decoder-available feature additions can improve residual side-info index selection before residual value prediction.

## Implementation

Added:

```text
scripts/run_stage101_enhanced_selector_feature_sweep.py
```

The script builds a full feature tensor once and slices it into feature modes:

- `stage100_base`: Stage98/100 base features.
- `gap_endpoint_norms`: base features plus reference gap and normalized endpoint motion norms.
- `gap_endpoint_rank`: norms features plus endpoint-score rank percentile.

Each feature mode trains `topk_bce` and `energy_regression`. No rendering, checkpoint, or heavy tensor is saved.

## Run

GPU check was performed before execution. GPU0 was busy and GPU1 was idle. Syntax check:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage101_enhanced_selector_feature_sweep.py
```

Full run:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage101_enhanced_selector_feature_sweep.py
```

## Outputs

```text
experiments/stage101_enhanced_selector_feature_sweep/stage101_enhanced_selector_feature_rows.csv
experiments/stage101_enhanced_selector_feature_sweep/stage101_enhanced_selector_feature_summary.csv
experiments/stage101_enhanced_selector_feature_sweep/stage101_enhanced_selector_feature_train_log.csv
experiments/stage101_enhanced_selector_feature_sweep/stage101_enhanced_selector_feature_summary.json
experiments/stage101_enhanced_selector_feature_sweep/stage101_enhanced_selector_feature_report.md
```

## Configuration

| item | value |
|---|---:|
| train tasks | 96 |
| eval tasks | 60 |
| train examples | 589824 |
| full feature dim | 75 |
| keep fraction | 0.1 |
| train steps/model | 300 |

## Results

| base | gap | endpoint energy recall | best feature | best objective | best energy recall | best relative recall | precision@keep |
|---|---:|---:|---|---|---:|---:|---:|
| linear | 4 | 0.2578457370400429 | stage100_base | energy_regression | 0.2935174204733061 | 0.47064504675243213 | 0.3431432687717935 |
| linear | 8 | 0.27749040722846985 | gap_endpoint_norms | energy_regression | 0.32395490298145696 | 0.4845810360030124 | 0.36720735462088333 |
| linear | 16 | 0.2058291733264923 | stage100_base | topk_bce | 0.2517792292767101 | 0.41649167074097526 | 0.2579128210329347 |
| stage65_adapter | 4 | 0.13412074485550757 | stage100_base | topk_bce | 0.16204810336880063 | 0.6889743960422018 | 0.4132322079461554 |
| stage65_adapter | 8 | 0.13563174087750285 | stage100_base | topk_bce | 0.17060783230944684 | 0.6881267146060341 | 0.42220921579160187 |
| stage65_adapter | 16 | 0.11971673224535254 | gap_endpoint_norms | energy_regression | 0.15503261362512907 | 0.634711515572336 | 0.3392114285379648 |

## Conclusion

- Learned selectors still outperform endpoint-difference selection across all groups.
- Hand-crafted endpoint/gap/rank extras do not produce a material improvement over Stage100 base features.
- The next step should be group-specific heads, larger train scale, or rendered validation of the current best selector rather than more endpoint feature additions.
