# Stage68 DAVIS Feed-Forward Selector Rendered Validation

Date: 2026-06-27

## Goal

Use the Stage67 feed-forward selector cost predictor to produce deterministic DP keyframe selections, then validate them with rendered DAVIS anchor-only reconstruction.

This stage is the first DAVIS rendered validation of the feed-forward selector path after Stage66/67 proxy training.

## Scope

| item | value |
|---|---:|
| eval sequences | Stage66 eval sequences |
| gaps | `4, 8, 16` |
| selector model | `full_feature_ridge` from Stage67 |
| max DP segment length | `16` |
| adapter checkpoint | Stage65 best `rgb_h256` |
| metric focus | all-frame PSNR |

## Inputs

| item | path |
|---|---|
| DAVIS all-gap manifest | `experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv` |
| Stage66 selector dataset summary | `experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset_summary.json` |
| Stage67 model params | `experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_model_params.json` |
| Stage65 adapter checkpoint | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors` |

## Expected Outputs

```text
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_selections.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation_summary.json
```

## Notes

- Selection uses only feed-forward features and deterministic DP.
- No rendered oracle, PSNR labels, dense-anchor labels, or reconstruction lookahead are used at selection time.
- This is still a scoped eval-subset validation, not the final all-frame RD package.

## Execution

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage68_davis_feedforward_selector_rendered_validation.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage68_davis_feedforward_selector_rendered_validation.py --device cuda
```

## Outputs

Tracked outputs:

```text
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_selections.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation_summary.json
```

Output size:

```text
28K experiments/stage68_davis_feedforward_selector_rendered_validation
```

## Results

Aggregate all-frame PSNR selector result:

| item | value |
|---|---:|
| comparison points | `12` |
| positive adapter all-frame PSNR points | `7` |
| mean selector delta adapter all-frame PSNR | `+0.030738190041048163 dB` |
| positive linear all-frame PSNR points | `7` |
| mean selector delta linear all-frame PSNR | `+0.02925622579667782 dB` |

Per-point all-frame PSNR comparison:

| sample | gap | rate MiB/frame | uniform adapter all PSNR | predicted adapter all PSNR | predicted - uniform |
|---|---:|---:|---:|---:|---:|
| `DAVIS/val/bmx-trees` | `4` | `0.119970703125` | `20.070097715317623` | `20.133674327209114` | `+0.06357661189149155` |
| `DAVIS/val/bmx-trees` | `8` | `0.062841796875` | `17.822039666501944` | `17.7858798253786` | `-0.03615984112334303` |
| `DAVIS/val/bmx-trees` | `16` | `0.03427734375` | `16.10083832134726` | `16.10083832134726` | `0.0` |
| `DAVIS/val/car-shadow` | `4` | `0.12568359375` | `21.10255897802341` | `21.102003395694947` | `-0.000555582328463089` |
| `DAVIS/val/car-shadow` | `8` | `0.0685546875` | `19.18506378945881` | `19.17832982344394` | `-0.006733966014870418` |
| `DAVIS/val/car-shadow` | `16` | `0.045703125` | `18.12073768533537` | `18.166228999092716` | `+0.04549131375734561` |
| `DAVIS/val/goat` | `4` | `0.121875` | `19.73679707382765` | `19.742322296239898` | `+0.0055252224122490645` |
| `DAVIS/val/goat` | `8` | `0.066015625` | `17.986255403638253` | `17.876470475541243` | `-0.10978492809701024` |
| `DAVIS/val/goat` | `16` | `0.035546875` | `16.74081067318615` | `16.76797396499338` | `+0.02716329180723065` |
| `DAVIS/val/soapbox` | `4` | `0.12002840909090909` | `21.524673663718215` | `21.558829687362664` | `+0.03415602364444936` |
| `DAVIS/val/soapbox` | `8` | `0.06463068181818182` | `19.176595738107597` | `19.259998745495473` | `+0.08340300738787576` |
| `DAVIS/val/soapbox` | `16` | `0.036931818181818184` | `17.089743502361333` | `17.352520629516956` | `+0.26277712715562274` |

## Conclusion

- Stage68 is the first fully feed-forward deterministic-DP selector rendered validation on DAVIS eval sequences.
- The result is mixed but positive on average: `+0.030738190041048163 dB` mean all-frame PSNR over uniform at matched keyframe budget/rate.
- The selector is not robust yet: `5/12` points are non-positive, including a clear negative point on `DAVIS/val/goat gap8`.
- `DAVIS/val/bmx-trees gap16` collapses to the uniform layout, giving exactly zero delta.
- Stage69/next selector work should add rendered-distortion labels or fallback calibration, then rerun DP validation before final RD packaging.
