# Stage148 Rate-Counted Side-Info Rendered Revalidation

Date: 2026-06-30

## Goal

Revalidate the Stage147 q6/top10 entropy index+value residual side-info fallback by actually encoding, decoding, and rendering the side-info corrected anchors on a broader gap4/gap8 DAVIS eval sample.

## Plan

- Reuse `scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py` because it already performs actual codec encode/decode/render validation:
  - encode fixed top-k residual side-info,
  - encode entropy top-k residual side-info,
  - decode entropy payload,
  - verify entropy decode equals fixed decode,
  - render decoded anchor and compute PSNR.
- Run with Stage148 labels/output paths.
- Use q12, gaps `4 8`, `keep_fraction=0.1`, `side_bits=6`, zlib level `9`.
- Use `max_tasks=120` for broader validation than Stage95 gap4/gap8 subset while keeping runtime feasible.
- Then package target/rate alignment against Stage142 corrected targets and Stage96 q12 main-anchor rates.
- Explicitly record that all side-info payload bytes are counted and that decoder-side dense target anchors are forbidden.
- Check `nvidia-smi` before each Python execution.

## Candidate Render Command

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py --stage 148 --mode "rate-counted side-info rendered revalidation" --output_prefix stage148_rate_counted_sideinfo_rendered_revalidation --report_title "Stage148 Rate-Counted Side-Info Rendered Revalidation" --summary_root experiments/stage148_rate_counted_sideinfo_rendered_revalidation --gaps 4 8 --max_tasks 120 --keep_fraction 0.1 --side_bits 6 --zlib_level 9 --device cuda
```

## Success Criteria

- Actual entropy decode max diff vs fixed decode remains `0`.
- Side-info PSNR remains near Stage142 corrected target range on the broader sample.
- Positive delta count remains high for Stage65 adapter gap4/gap8.
- Package records direct/amortized total rates with side-info counted.

## Risk

This is still a sampled rendered revalidation, not full all-row DAVIS eval. If it passes, the next step is either full all-row revalidation or full-video RD packaging using the same counted payload contract.

## Status

Completed as a broader sampled rendered revalidation.

## Implementation

Reused:

```text
scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py
```

Updated its generated note text to describe the residual values as encoder-side transmitted payload values, not decoder-only predictor outputs.

Added:

```text
scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
```

The package script aligns actual Stage148 rendered validation rows to Stage142 corrected targets and computes direct/amortized total rates with side-info bytes counted.

## Runs

GPU checks were performed before Python executions.

Compile and rendered revalidation used GPU3:

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py --stage 148 --mode "rate-counted side-info rendered revalidation" --output_prefix stage148_rate_counted_sideinfo_rendered_revalidation --report_title "Stage148 Rate-Counted Side-Info Rendered Revalidation" --summary_root experiments/stage148_rate_counted_sideinfo_rendered_revalidation --gaps 4 8 --max_tasks 120 --keep_fraction 0.1 --side_bits 6 --zlib_level 9 --device cuda
```

Package run used GPU2 binding for a CPU-only script:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
```

After updating Stage93 note text, py_compile was rerun successfully:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
```

## Outputs

```text
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_rows.csv
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_summary.csv
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_summary.json
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_report.md
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_rows.csv
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_decisions.csv
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_summary.json
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package.json
experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_report.md
```

Output size: `96K`.

## Results

- Total sampled tasks: `120`.
- Gap4 sampled tasks: `60`.
- Gap8 sampled tasks: `60`.
- Codec: q12 endpoint anchors plus q6/top10 entropy index+value residual side-info.
- Keep fraction: `0.1`.
- Side bits: `6`.
- Entropy decode max abs diff vs fixed decode: `0.0` for both gap4/gap8.
- Gap4 target: `23.004337221027775`.
- Gap4 base PSNR: `20.264133489387127`.
- Gap4 entropy side-info PSNR: `22.850143675432175`.
- Gap4 entropy gap to target: `-0.15419354559560006 dB`.
- Gap4 delta vs base: `+2.5860101860450535 dB`.
- Gap4 positive deltas: `60/60`.
- Gap4 mean payload bytes/intermediate frame: `34785.51666666667`.
- Gap4 direct total rate: `0.21511227459583468 MiB/frame`.
- Gap4 amortized total rate: `0.2068187611388543 MiB/frame`.
- Gap8 target: `21.56004909948801`.
- Gap8 base PSNR: `19.062352986913425`.
- Gap8 entropy side-info PSNR: `21.965723744155056`.
- Gap8 entropy gap to target: `+0.40567464466704806 dB`.
- Gap8 delta vs base: `+2.9033707572416314 dB`.
- Gap8 positive deltas: `60/60`.
- Gap8 mean payload bytes/intermediate frame: `35153.03333333333`.
- Gap8 direct total rate: `0.13114993178728715 MiB/frame`.
- Gap8 amortized total rate: `0.12695936365806554 MiB/frame`.

## Decisions

- Stage148 sampled rendered revalidation passes.
- Actual entropy decode matches fixed decode exactly on the sampled validation.
- All side-info payload bytes are counted.
- Decoder-side target dense anchor/RGB/unencoded residual inputs remain forbidden.
- Next step should be full all-row or full-video RD validation using the same counted payload contract.

## Conclusion

Stage148 confirms that the Stage147 fallback is not just a package-level inference: actual entropy payload encode/decode/render passes on a 120-task q12 gap4/gap8 DAVIS eval sample. Gap4 is within `0.155 dB` of the corrected target and gap8 exceeds the corrected target, with positive gains on all sampled tasks. This is still sampled validation; the next required step is full all-row/full-video RD packaging before making a final paper-level claim.
