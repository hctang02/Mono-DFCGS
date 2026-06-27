# Stage66 DAVIS Feed-Forward Selector Dataset

Date: 2026-06-27

## Goal

Build a DAVIS segment-cost dataset for the deployable feed-forward selector path.

This stage does not make a selector quality claim. It prepares training/evaluation rows where:

- Features are encoder-side only: segment length/position, endpoint Gaussian-anchor statistics, and RGB motion statistics from the original input frames.
- Labels are offline supervision: Stage65 best `rgb_h256` adapter prediction error against dense gap1 anchors.
- Dense gap1 anchors and label errors are not test-time inputs.

## Inputs

| item | path |
|---|---|
| DAVIS all-gap manifest | `experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv` |
| Stage65 adapter checkpoint | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors` |
| Output summary root | `experiments/stage66_davis_feedforward_selector_dataset` |

## Planned Dataset

Default scoped run:

| item | value |
|---|---:|
| train sequences | up to `8` |
| eval sequences | up to `4` |
| max segment length | `16` |
| max segments per sequence | `384` |
| image feature size | `64` |
| quant bits | `8` |
| adapter hidden dim | `256` |

## Expected Outputs

```text
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_sequence_summary.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_feature_correlations.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset_summary.json
```

## Notes

- This stage uses dense anchors only to produce offline labels for Stage67 selector training.
- Final selector inference must use a frozen feed-forward predictor plus deterministic DP/selection and must not use rendered oracle, PSNR labels, dense-anchor labels, or test-time reconstruction lookahead.

## Execution

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage66_davis_feedforward_selector_dataset.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage66_davis_feedforward_selector_dataset.py --device cuda
```

## Results

Selected sequences:

| selector split | count | sequences |
|---|---:|---|
| train | `8` | `bus`, `cat-girl`, `elephant`, `flamingo`, `lady-running`, `scooter-gray`, `swing`, `tuk-tuk` |
| eval | `4` | `bmx-trees`, `car-shadow`, `goat`, `soapbox` |

Dataset summary:

| item | value |
|---|---:|
| row count | `4608` |
| train rows | `3072` |
| eval rows | `1536` |
| sequence count | `12` |
| max segment length | `16` |
| max segments per sequence | `384` |
| adapter better than linear in anchor space | `0 / 4608` |
| mean adapter anchor MSE label | `0.031690189455081855` |
| mean linear anchor MSE label | `0.011579127648414848` |

Strongest all-scope feature correlations with `log10(adapter_anchor_mse_mean)`:

| feature | Pearson |
|---|---:|
| `endpoint_anchor_l1` | `0.7819601460791495` |
| `endpoint_rgb_mse` | `0.7630952706790444` |
| `endpoint_anchor_mse` | `0.7528988639175856` |
| `rgb_motion_max` | `0.7304595140192872` |
| `rgb_motion_mean` | `0.7021857588764513` |

Tracked outputs:

```text
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_sequence_summary.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_feature_correlations.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset_summary.json
```

Output size:

```text
2.4M experiments/stage66_davis_feedforward_selector_dataset
```

## Conclusion

- Stage66 successfully creates a DAVIS feed-forward selector segment dataset with only encoder-side features and offline labels.
- Endpoint-anchor and RGB-motion features correlate strongly with the offline anchor-space adapter error label.
- The Stage65 RGB-trained adapter is worse than linear in dense-anchor MSE for every Stage66 segment, even though it improved rendered eval PSNR in Stage65. This confirms that anchor-space MSE is a proxy/difficulty label, not a final rendered-quality label.
- Stage67 should either train a selector predictor on this proxy with clear caveats or add a rendered-distortion label subset before making deployable selector quality claims.
