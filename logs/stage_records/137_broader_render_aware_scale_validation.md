# Stage137 Broader Render-Aware Scale Validation

Date: 2026-06-29

## Goal

Validate the Stage136 smoke-selected render-aware adapter-delta scale on the broader 60-task eval slice.

## Plan

- Reuse the Stage136 sweep mechanics on the Stage125-style 60-task eval slice.
- Sweep all Stage135 scale candidates for q4/top20 and q4/top10.
- Track the Stage136 smoke-selected candidate separately from the broader best candidate.
- Keep rate accounting unchanged: no residual payload and no selected-index payload.
- Do not load target dense anchors or teacher residual side-info.
- Use target RGB only for offline rendered metrics.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage137_broader_render_aware_scale_validation.py
```

The script reuses Stage136 fields and summary logic, broadening the sweep to a 60-task validation slice.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage137_broader_render_aware_scale_validation.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage137_broader_render_aware_scale_validation.py
```

## Outputs

```text
experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_rows.csv
experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_summary.csv
experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_summary.json
experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_report.md
```

Output size: `260K`.

## Results

- Row count: `720`.
- Stage136 smoke-selected candidate broader result: q4/top20 scale `0.5`, PSNR `19.01421663314399`, delta vs base `+0.06341363671285936`, positives `49/60`.
- Broader best candidate: q4/top20 scale `0.75`, PSNR `19.022109503207204`, delta vs base `+0.07130650677606927`, positives `49/60`.
- Original Stage132/Stage125 q4/top20 scale `1.0`: PSNR `19.010259350474836`, delta vs base `+0.059456354043700026`, positives `48/60`.
- q4/top10 best scale is `0.75`: PSNR `18.997890662360874`, delta vs base `+0.047087665929735116`.

## Conclusion

The render-aware scale sweep improves over the current Stage132 unscaled adapter-delta policy on the 60-task slice. Stage138 should package an updated deployable policy using q4/top20 scale `0.75` as the primary candidate, with q4/top10 scale `0.75` as the low-rate candidate. The validation remains no-teacher and decoder-side: no target dense anchor, no teacher residual side-info, no transmitted residual values, and no transmitted selected indices.
