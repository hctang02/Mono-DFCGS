# Stage149 Full Gap4/Gap8 Side-Info Rendered Validation

Date: 2026-06-30

## Goal

Upgrade Stage148 from sampled rendered revalidation to full q12 gap4/gap8 DAVIS eval task validation for the rate-counted q6/top10 entropy index+value residual side-info fallback.

## Plan

- Reuse the Stage93 entropy side-info encode/decode/render validator.
- Add a `--base_methods` filter so Stage149 can evaluate only `stage65_adapter`, avoiding unnecessary linear-base rendering during the full run.
- Run q12 gaps `4 8` with `max_tasks=0` to select all available eval rows from Stage79.
- Use `keep_fraction=0.1`, `side_bits=6`, zlib level `9`.
- Package actual rendered PSNR against Stage142 corrected targets and count all side-info payload bytes in direct/amortized total rates.
- Keep decoder contract explicit: decoder gets endpoint anchors, normalized time, and encoded payload only; target dense anchors/RGB/unencoded residuals remain forbidden decoder inputs.
- Check `nvidia-smi` before every Python execution.

## Expected Scale

- q12 gap4 eval rows: `1463`.
- q12 gap8 eval rows: `1707`.
- Total expected eval rows: `3170`.
- Base methods: `stage65_adapter` only.

## Candidate Render Command

```text
CUDA_VISIBLE_DEVICES=<idle> /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py --stage 149 --mode "full q12 gap4/gap8 rate-counted side-info rendered validation" --output_prefix stage149_full_gap48_sideinfo_rendered_validation --report_title "Stage149 Full Gap4/Gap8 Side-Info Rendered Validation" --summary_root experiments/stage149_full_gap48_sideinfo_rendered_validation --gaps 4 8 --max_tasks 0 --keep_fraction 0.1 --side_bits 6 --zlib_level 9 --base_methods stage65_adapter --device cuda
```

## Success Criteria

- Actual entropy decode max diff vs fixed decode remains `0.0`.
- Gap4/gap8 side-info PSNR is in corrected StreamSplat target range or clearly closer than feed-forward adapter-only.
- Positive delta fraction remains high on all rows.
- Package reports all side-info payload bytes and total rates.

## Risk

Full all-row rendering can be long. If runtime is excessive, continue with chunked/resumable validation rather than reducing protocol rigor.

## Status

Completed.

## Implementation

Updated:

```text
scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py
scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
```

Stage93 now supports:

- `--base_methods`, used by Stage149 to evaluate only `stage65_adapter`.
- `--disable_cache`, used by Stage149 to avoid unbounded GPU anchor cache growth on full eval.
- `--progress_interval`, used by Stage149 to log long-run progress.

Stage148 package script now supports stage/output parameterization and `--validation_scope full_eval`.

## Runs

GPU checks were performed before every Python execution. GPU1 was selected for compile, full render validation, and package runs.

Compile:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py
```

Full rendered validation:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py --stage 149 --mode "full q12 gap4/gap8 rate-counted side-info rendered validation" --output_prefix stage149_full_gap48_sideinfo_rendered_validation --report_title "Stage149 Full Gap4/Gap8 Side-Info Rendered Validation" --summary_root experiments/stage149_full_gap48_sideinfo_rendered_validation --gaps 4 8 --max_tasks 0 --keep_fraction 0.1 --side_bits 6 --zlib_level 9 --base_methods stage65_adapter --disable_cache --progress_interval 100 --device cuda
```

Package:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage148_rate_counted_sideinfo_revalidation_package.py --stage 149 --mode "full q12 gap4/gap8 rate-counted side-info rendered validation package" --output_prefix stage149_full_gap48_sideinfo_rendered_validation_package --report_title "Stage149 Full Gap4/Gap8 Side-Info Rendered Validation Package" --validation_scope full_eval --stage148_summary experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_summary.json --summary_root experiments/stage149_full_gap48_sideinfo_rendered_validation
```

## Outputs

```text
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_rows.csv
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_summary.csv
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_summary.json
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_report.md
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_rows.csv
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_decisions.csv
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_summary.json
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package.json
experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_report.md
```

Output size: `708K`.

## Results

- Total q12 gap4/gap8 eval rows processed: `3170/3170`.
- Base method: `stage65_adapter`.
- Keep fraction: `0.1`.
- Side bits: `6`.
- Entropy decode max abs diff vs fixed decode: `0.0`.
- Gap4 task count: `1463`.
- Gap4 corrected target: `23.004337221027775`.
- Gap4 base PSNR: `20.210770635920667`.
- Gap4 entropy side-info PSNR: `22.768595216050993`.
- Gap4 entropy gap to target: `-0.23574200497678177 dB`.
- Gap4 delta vs base: `+2.55782458013036 dB`.
- Gap4 positive deltas: `1463/1463`.
- Gap4 mean payload bytes/intermediate frame: `34943.74094326726`.
- Gap4 direct total rate: `0.21526316902466064 MiB/frame`.
- Gap4 amortized total rate: `0.20693193196047374 MiB/frame`.
- Gap8 task count: `1707`.
- Gap8 corrected target: `21.56004909948801`.
- Gap8 base PSNR: `19.067097958467713`.
- Gap8 entropy side-info PSNR: `21.857517703953395`.
- Gap8 entropy gap to target: `+0.2974686044653865 dB`.
- Gap8 delta vs base: `+2.790419745485695 dB`.
- Gap8 positive deltas: `1707/1707`.
- Gap8 mean payload bytes/intermediate frame: `35172.31634446397`.
- Gap8 direct total rate: `0.13116832149974542 MiB/frame`.
- Gap8 amortized total rate: `0.12697545465646654 MiB/frame`.

## Decisions

- Full q12 gap4/gap8 eval-row validation passes under the Stage148/149 `0.25 dB` target tolerance.
- Gap8 exceeds the corrected target.
- Gap4 remains `0.235742 dB` below the corrected target, but inside the tolerance and vastly improved over adapter-only.
- All side-info payload bytes are counted.
- Next stage should package full-video RD and/or test a modest q/keep setting bump for closing the remaining gap4 `0.236 dB` without excessive rate increase.

## Conclusion

Stage149 pulls full q12 gap4/gap8 eval-row middle-frame validation back to the corrected StreamSplat reference range using a rate-counted transmitted side-info payload. It is now appropriate to move from sampled validation to full-video RD packaging and, if needed, a small rate/quality refinement focused only on the remaining gap4 shortfall.
