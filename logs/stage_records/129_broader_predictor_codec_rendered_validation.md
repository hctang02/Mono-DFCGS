# Stage129 Broader Predictor Codec Rendered Validation

Date: 2026-06-29

## Goal

Render-validate the Stage128 predictor-only selected residual codec on a broader 60-task eval slice.

## Plan

- Add a Stage129 rendered validation script.
- Load Stage128 integrated policy, Stage126 normalization stats, and Stage127 external checkpoints.
- Recompute deterministic endpoint-diff selected indices at decoder side.
- Predict residual values with the selected residual MLP and apply them to the linear base.
- Render q4/top20 and q4/top10 predictor-only outputs.
- Transmit no residual payload and no selected-index payload.
- Do not load target dense anchors; target RGB is used only for offline rendered metrics.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage129_broader_predictor_codec_rendered_validation.py
```

The script loads Stage128 settings, Stage126 stats, and Stage127 checkpoints, applies predicted selected residual values to linear base anchors, and renders the resulting anchors. It does not load dense target anchors.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage129_broader_predictor_codec_rendered_validation.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage129_broader_predictor_codec_rendered_validation.py
```

## Outputs

```text
experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_rows.csv
experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_summary.csv
experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_summary.json
experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_report.md
```

## Results

| setting | role | keep | tasks | rate | base PSNR | full adapter PSNR | predictor PSNR | delta base | delta full | positives | near full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_top10 | low_rate | 0.1 | 60 | 0.11729838135687401 | 18.950802996431143 | 19.1876432931255 | 18.865777753557193 | -0.08502524287394495 | -0.321865539568306 | 32/60 | 13/60 |
| q4_top20 | primary | 0.2 | 60 | 0.11729838135687401 | 18.950802996431143 | 19.1876432931255 | 18.76520305064309 | -0.1855999457880447 | -0.4224402424824056 | 26/60 | 12/60 |

## Conclusion

- The MSE-trained dedicated selected residual MLP is not render-safe in this validation.
- Although Stage127 reduced residual MSE, Stage129 rendered PSNR regresses below the linear base.
- Stage130 should explicitly compare teacher side-info, adapter-delta predictor, dedicated MLP predictor, and linear base.
- Stage132 deployable package should not select this MLP predictor as final unless later render-aware training fixes the regression.
