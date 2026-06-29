# Stage141 Deployable Full-Pipeline Manifest

Date: 2026-06-29

## Goal

Package the final no-teacher deployable full-pipeline manifest after Stage135-140.

## Plan

- Use Stage138 as the final policy source.
- Use Stage139 as the full-pipeline RD/rate accounting source.
- Use Stage140 as the ablation/rejection evidence source.
- Explicitly list decoder-side inputs, steps, policy constants, forbidden inputs, and rate accounting.
- State that teacher residual side-info and target dense anchors are not decoder inputs.
- State that residual values and selected indices are not transmitted.
- State that the Stage65 adapter checkpoint is pre-shared; if transmitted in-session, it must be accounted separately.
- Check `nvidia-smi` before running Python, even though the package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage141_deployable_full_pipeline_manifest.py
```

The script packages the final deployable manifest from Stage135 protocol, Stage137 validation, Stage138 policy, Stage139 full-pipeline RD, and Stage140 ablations.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage141_deployable_full_pipeline_manifest.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage141_deployable_full_pipeline_manifest.py
```

## Outputs

```text
experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_manifest.json
experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_checklist.csv
experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_manifest_package.json
experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_manifest_report.md
```

Output size: `28K`.

## Results

- Manifest: `deployable_render_aware_scaled_adapter_delta_full_pipeline_v1`.
- Policy: `deployable_render_aware_scaled_adapter_delta_selected_residual_codec_v1`.
- Primary: q4/top20 scale `0.75`, rate `0.11729838135687401`, PSNR `19.022109503207204`, delta vs base `+0.07130650677606927`.
- Low-rate: q4/top10 scale `0.75`, rate `0.11729838135687401`, PSNR `18.997890662360874`, delta vs base `+0.047087665929735116`.
- Residual payload bytes: `0`.
- Selected-index payload bytes: `0`.
- Policy scale payload bytes: `0`.
- Teacher side-info deployable: `0`.
- Checklist: all pass.

## Conclusion

Stage141 completes the Stage135-141 render-aware no-teacher deployable predictor line. The final manifest is decoder-side and does not require teacher residual side-info, target dense anchors, target residuals, target RGB, transmitted selected indices, or transmitted residual values.
