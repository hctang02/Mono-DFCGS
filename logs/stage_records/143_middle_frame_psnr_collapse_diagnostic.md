# Stage143 Middle-Frame PSNR Collapse Diagnostic

Date: 2026-06-29

## Goal

Run a real rendered diagnostic to determine whether the current middle-frame PSNR gap is caused mainly by data/renderer, anchor quantization, or the dynamic prediction model.

## Plan

- Use the same scoped DAVIS val sequences as Stage77: `bmx-trees`, `car-shadow`, `goat`, `soapbox`.
- Evaluate dense-direct anchor rendering as a renderer/data ceiling.
- Evaluate linear and Stage65 adapter interpolation at uniform gaps `4`, `8`, and `16`.
- Compare codecs `float32`, `q8`, `q12`, and `q16` to isolate quantization effects.
- Report all/middle/given PSNR and middle-frame gap to Stage75 targets.
- Keep outputs to CSV/JSON/Markdown only; do not save rendered images or tensors.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage143_middle_frame_psnr_collapse_diagnostic.py
```

The script renders dense-direct anchors and uniform-gap linear/Stage65-adapter predictions on the Stage77 scoped DAVIS val sequences, across `float32`, `q8`, `q12`, and `q16` anchor codecs.

## Run

GPU check was performed before execution. GPU3 was idle and used with `CUDA_VISIBLE_DEVICES=3`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage143_middle_frame_psnr_collapse_diagnostic.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage143_middle_frame_psnr_collapse_diagnostic.py
```

## Outputs

```text
experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_rows.csv
experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_summary.csv
experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_findings.csv
experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_summary.json
experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_package.json
experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_report.md
```

Output size: `60K`.

## Results

- Row count: `144`.
- Float32 dense-direct middle PSNR: gap4 `29.749654363336436`, gap8 `29.74550454012203`.
- Float32 adapter middle PSNR: gap4 `18.255332640417755`, gap8 `17.067872741131573`.
- Dense-direct minus adapter: gap4 `11.49432172291868 dB`, gap8 `12.677631798990458 dB`.
- q12 adapter middle PSNR: gap4 `18.25528373980314`, gap8 `17.06788939572344`.
- q16 adapter middle PSNR: gap4 `18.255332804105823`, gap8 `17.067879472412265`.
- q16 minus q12 adapter middle: gap4 `+0.000049064302682921834 dB`, gap8 `-0.00000992331117544154 dB`.
- q8 dense-direct middle PSNR drops to gap4 `27.056614034484408`, gap8 `27.05319433167501`, so aggressive q8 quantization hurts keyframe/dense reconstruction, but q12/q16/float32 are already near the dense-direct ceiling.

## Conclusion

The current `17-18 dB` middle-frame collapse is not caused by the renderer/data path and is not meaningfully fixed by raising q12 to q16 or float32. The main bottleneck is the dynamic middle-frame prediction model. Stage144 should still package the high-rate upper-bound cleanly, then the next meaningful path is large-scale render-loss adapter training and/or rate-counted motion/residual side-info.
