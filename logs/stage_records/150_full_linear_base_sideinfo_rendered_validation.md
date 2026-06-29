# Stage150 Full Linear-Base Side-Info Rendered Validation

Date: 2026-06-30

## Goal

Close the remaining Stage149 gap4 shortfall by validating a decoder-safe linear-base q6/top10 entropy index+value residual side-info fallback on all q12 gap4/gap8 eval rows.

## Rationale

Stage149 adapter-base side-info passed full eval under the `0.25 dB` tolerance, but gap4 remained `-0.235742 dB` below the corrected target. Earlier Stage95/148 evidence suggests linear-base residual side-info gives higher PSNR and lower payload than adapter-base residual side-info. Linear base is decoder-safe because it is computed from endpoint anchors and normalized time.

## Plan

- Reuse the Stage149 full-eval validator with `--base_methods linear`.
- Use q12 gaps `4 8`, `max_tasks=0`, `keep_fraction=0.1`, `side_bits=6`, zlib level `9`.
- Keep `--disable_cache` and progress logging for full eval stability.
- Extend the package script to accept `--base_method linear`.
- Package actual rendered PSNR against Stage142 targets and count all payload bytes in direct/amortized rates.
- Check `nvidia-smi` before every Python execution.

## Candidate Render Command

```text
CUDA_VISIBLE_DEVICES=<idle> /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py --stage 150 --mode "full q12 gap4/gap8 linear-base rate-counted side-info rendered validation" --output_prefix stage150_full_linear_base_sideinfo_rendered_validation --report_title "Stage150 Full Linear-Base Side-Info Rendered Validation" --summary_root experiments/stage150_full_linear_base_sideinfo_rendered_validation --gaps 4 8 --max_tasks 0 --keep_fraction 0.1 --side_bits 6 --zlib_level 9 --base_methods linear --disable_cache --progress_interval 100 --device cuda
```

## Success Criteria

- Gap4 and gap8 entropy side-info PSNR meet or exceed Stage142 corrected targets on full eval rows.
- Actual entropy decode max diff vs fixed decode remains `0.0`.
- Positive delta fraction remains `1.0` or near `1.0`.
- Direct/amortized total rates are lower than or comparable to Stage149.

## Risk

Linear-base side-info may outperform adapter-base side-info but changes the selected base reconstruction path for the fallback. The package must make this explicit and keep the decoder contract unchanged.

## Status

Completed.

## Implementation

Updated:

```text
scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
```

The package script now accepts `--base_method`, allowing Stage150 to package `linear` base results.

## Runs

GPU checks were performed before each Python execution. GPU1 was idle and selected for compile, full rendered validation, and package runs.

Compile:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
```

Full rendered validation:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py --stage 150 --mode "full q12 gap4/gap8 linear-base rate-counted side-info rendered validation" --output_prefix stage150_full_linear_base_sideinfo_rendered_validation --report_title "Stage150 Full Linear-Base Side-Info Rendered Validation" --summary_root experiments/stage150_full_linear_base_sideinfo_rendered_validation --gaps 4 8 --max_tasks 0 --keep_fraction 0.1 --side_bits 6 --zlib_level 9 --base_methods linear --disable_cache --progress_interval 100 --device cuda
```

Package:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py --stage 150 --mode "full q12 gap4/gap8 linear-base rate-counted side-info rendered validation package" --output_prefix stage150_full_linear_base_sideinfo_rendered_validation_package --report_title "Stage150 Full Linear-Base Side-Info Rendered Validation Package" --validation_scope full_eval --base_method linear --stage148_summary experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_summary.json --summary_root experiments/stage150_full_linear_base_sideinfo_rendered_validation
```

## Outputs

```text
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_rows.csv
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_summary.csv
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_summary.json
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_report.md
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_rows.csv
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_decisions.csv
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_summary.json
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package.json
experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_report.md
```

Output size: `684K`.

## Results

- Total q12 gap4/gap8 eval rows processed: `3170/3170`.
- Base method: `linear`.
- Keep fraction: `0.1`.
- Side bits: `6`.
- Entropy decode max abs diff vs fixed decode: `0.0`.
- Gap4 task count: `1463`.
- Gap4 corrected target: `23.004337221027775`.
- Gap4 base PSNR: `19.983964182290645`.
- Gap4 entropy side-info PSNR: `23.104893423851635`.
- Gap4 entropy gap to target: `+0.10055620282386002 dB`.
- Gap4 delta vs base: `+3.120929241560934 dB`.
- Gap4 positive deltas: `1463/1463`.
- Gap4 mean payload bytes/intermediate frame: `30062.867395762132`.
- Gap4 direct total rate: `0.21060840528077832 MiB/frame`.
- Gap4 amortized total rate: `0.20344085915256202 MiB/frame`.
- Gap8 task count: `1707`.
- Gap8 corrected target: `21.56004909948801`.
- Gap8 base PSNR: `18.76642882721105`.
- Gap8 entropy side-info PSNR: `22.020188948523128`.
- Gap8 entropy gap to target: `+0.4601398490351194 dB`.
- Gap8 delta vs base: `+3.253760121312108 dB`.
- Gap8 positive deltas: `1707/1707`.
- Gap8 mean payload bytes/intermediate frame: `30203.919156414762`.
- Gap8 direct total rate: `0.12643008870779784 MiB/frame`.
- Gap8 amortized total rate: `0.12282950096351242 MiB/frame`.

## Decisions

- Full q12 gap4/gap8 eval-row validation passes with both gaps above corrected targets.
- Linear base is selected over adapter base for the rate-counted side-info fallback because it has higher PSNR and lower payload.
- All side-info payload bytes are counted.
- Decoder-side target dense anchor/RGB/unencoded residual inputs remain forbidden.
- Next stage should package the final full-video RD policy around the Stage150 linear-base side-info fallback.

## Conclusion

Stage150 achieves the requested middle-frame quality recovery on full q12 gap4/gap8 eval rows: both gap4 and gap8 are above the corrected StreamSplat middle-frame targets, with exact entropy decode and positive gains on every row. This is the first full eval-row result that fully clears the target range while keeping payload rate counted.
