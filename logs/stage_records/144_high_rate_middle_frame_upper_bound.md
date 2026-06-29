# Stage144 High-Rate Middle-Frame Upper Bound

Date: 2026-06-30

## Goal

Formalize the Stage143 high-rate/uncompressed diagnosis into a decision package: determine whether increasing anchor quantization rate can recover paper-level middle-frame PSNR, or whether model training/side-info is required.

## Plan

- Use Stage143 rendered rows; do not rerender.
- Compare q12, q16, and float32 adapter middle PSNR against Stage75 targets.
- Compare dense-direct q12/q16/float32 against Stage75 targets to verify data/renderer ceiling.
- Report quantization sensitivity and dynamic-model gap.
- Output a go/no-go decision for further q-bit tuning vs large-scale adapter training / rate-counted side-info.
- Check `nvidia-smi` before running Python, even though the package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage144_high_rate_middle_frame_upper_bound.py
```

The script uses Stage143 rendered rows to package a high-rate/uncompressed upper-bound decision without rerendering.

## Run

GPU check was performed before execution. GPU3 was idle and used with `CUDA_VISIBLE_DEVICES=3`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage144_high_rate_middle_frame_upper_bound.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage144_high_rate_middle_frame_upper_bound.py
```

## Outputs

```text
experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_rows.csv
experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_decisions.csv
experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_summary.json
experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_package.json
experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_report.md
```

Output size: `24K`.

## Results

- Gap4 target middle PSNR: `23.004337221027775`.
- Gap4 q12 adapter middle PSNR: `18.25528373980314`.
- Gap4 float32 adapter middle PSNR: `18.255332640417755`.
- Gap4 float32 dense-direct middle PSNR: `29.749654363336436`.
- Gap4 float32-q12 adapter gain: `+0.00004890061461537698 dB`.
- Gap4 float32 adapter gap to target: `-4.74900458061002 dB`.
- Gap8 target middle PSNR: `21.56004909948801`.
- Gap8 q12 adapter middle PSNR: `17.06788939572344`.
- Gap8 float32 adapter middle PSNR: `17.067872741131573`.
- Gap8 float32 dense-direct middle PSNR: `29.74550454012203`.
- Gap8 float32-q12 adapter gain: `-0.000016654591867393265 dB`.
- Gap8 float32 adapter gap to target: `-4.492176358356435 dB`.

## Decisions

- Raising anchor q-bit is rejected as the primary fix.
- Renderer/data ceiling is not the bottleneck.
- Dynamic model is the primary bottleneck.
- Next stage should start large-scale adapter training and keep rate-counted side-info as fallback.

## Conclusion

The first quality-rescue phase identifies the failure mode: the middle-frame collapse is dynamic-model-side. Q12 is already enough for this adapter, and high-rate/float32 anchors do not recover the missing `4.5-4.75 dB`. Stage145 should start meaningful larger-scale adapter training rather than more q-bit tuning.
