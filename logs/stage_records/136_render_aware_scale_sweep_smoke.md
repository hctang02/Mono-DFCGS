# Stage136 Render-Aware Scale Sweep Smoke

Date: 2026-06-29

## Goal

Run a small rendered validation smoke for the Stage135 adapter-delta scale calibration protocol.

## Plan

- Reuse the Stage124/125 rendered validation path.
- Sweep Stage135 scale candidates for q4/top20 and q4/top10.
- Use deterministic endpoint-diff selected indices reproduced from the two endpoint anchors.
- Apply `adapter_delta_scale * (adapter_attrs - linear_attrs)` at selected indices.
- Keep residual and selected-index transmitted bytes at zero.
- Do not load target dense anchors or teacher residual side-info.
- Use target RGB only for offline rendered metrics.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage136_render_aware_scale_sweep_smoke.py
```

The script reuses the Stage124 rendered validation path and sweeps the Stage135 adapter-delta scale candidates.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage136_render_aware_scale_sweep_smoke.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage136_render_aware_scale_sweep_smoke.py
```

## Outputs

```text
experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_rows.csv
experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_summary.csv
experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_summary.json
experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_report.md
```

Output size: `72K`.

## Results

- Row count: `144`.
- Best smoke candidate: `q4_top20`, scale `0.5`.
- Best mean selected PSNR: `20.135994499746502`.
- Best mean delta vs base: `+0.056354919137378445`.
- Best positive delta count: `8/12`.
- Original scale `1.0` q4/top20 mean selected PSNR: `20.099010301231182`.
- Original scale `1.0` q4/top20 mean delta vs base: `+0.019370720622054066`.

## Conclusion

Scale `0.5` improves the smoke result over scale `1.0` and should be validated on the broader Stage137 slice. The protocol remains no-teacher and decoder-side: no target dense anchor, no teacher residual side-info, no transmitted residual values, and no transmitted selected indices.
