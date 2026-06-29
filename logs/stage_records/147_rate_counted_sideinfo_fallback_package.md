# Stage147 Rate-Counted Side-Info Fallback Package

Date: 2026-06-30

## Goal

After Stage146 showed that simply continuing the current feed-forward adapter objective regresses on a broader eval sample, package a rate-counted side-info fallback candidate that can recover middle-frame PSNR toward the corrected StreamSplat target range.

## Plan

- Use existing Stage96 q6/top10 entropy index+value residual side-info results as the high-quality fallback evidence.
- Compare q12 `stage65_adapter` side-info PSNR for gap4/gap8 against Stage142 corrected middle-frame targets.
- Count all side-info bytes in direct and amortized total rates.
- Make the encoder/decoder contract explicit:
  - Encoder may use target/intermediate-frame information to compute a transmitted residual side-info payload.
  - Decoder receives only endpoint anchors, normalized time, and the encoded payload.
  - Target dense anchors, target RGB, and unencoded residuals remain forbidden decoder inputs.
- Separate this from the previously rejected uncounted/decoder-unsafe teacher side-info framing.
- Output a decision package for Stage148 full rendered revalidation.
- Check `nvidia-smi` before running Python even though the package is CPU-only.

## Expected Evidence

Stage96 q12 `stage65_adapter` q6/top10 entropy side-info already reported:

- Gap4 PSNR near `22.84` with all side-info counted.
- Gap8 PSNR near `21.40` with all side-info counted.
- Positive delta on all sampled tasks.

These values are near the Stage142 corrected middle-frame targets (`23.004337221027775` and `21.56004909948801`) and are much closer than the no-side-info adapter path.

## Success Criteria

- Script compiles and runs.
- Package reports target gaps, side-info payload/rates, and decision for gap4/gap8.
- Package explicitly records that this is a rate-counted transmitted payload candidate, not a decoder-side dense-anchor input.
- Stage147 remains a package/decision stage, not a final full-video claim.

## Risk

Stage96 is a broader 60-task side-info eval slice, not the full Stage75 paper-protocol video evaluation. Stage148 must revalidate the selected fallback in a full rendered setting before making a final quality claim.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage147_rate_counted_sideinfo_fallback_package.py
```

The script packages Stage96 q6/top10 entropy index+value residual side-info as a rate-counted fallback candidate and aligns it to the Stage142 corrected middle-frame targets.

## Run

GPU check was performed before execution. GPU1 had `0%` util and was used via `CUDA_VISIBLE_DEVICES=1`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage147_rate_counted_sideinfo_fallback_package.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage147_rate_counted_sideinfo_fallback_package.py
```

## Outputs

```text
experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv
experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_decisions.csv
experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_policy.json
experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_summary.json
experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_package.json
experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_report.md
```

Output size: `36K`.

## Results

- Policy: `rate_counted_entropy_index_value_residual_sideinfo_fallback_v1`.
- Setting: `q6_top10_entropy_index_value`.
- Keep fraction: `0.1`.
- Side bits: `6`.
- Gap4 corrected target: `23.004337221027775`.
- Gap4 base PSNR: `20.041732308843873`.
- Gap4 side-info PSNR: `22.841151135422116`.
- Gap4 side-info gap to target: `-0.16318608560565906 dB`.
- Gap4 side-info MiB/intermediate frame: `0.033147705925835505`.
- Gap4 direct total rate: `0.21508592669374865 MiB/frame`.
- Gap4 amortized total rate: `0.20679900021228975 MiB/frame`.
- Gap4 positive deltas: `18/18`.
- Gap8 corrected target: `21.56004909948801`.
- Gap8 base PSNR: `18.706547218120296`.
- Gap8 side-info PSNR: `21.39901144086742`.
- Gap8 side-info gap to target: `-0.1610376586205895 dB`.
- Gap8 side-info MiB/intermediate frame: `0.033867986578690376`.
- Gap8 direct total rate: `0.13149337333220473 MiB/frame`.
- Gap8 amortized total rate: `0.12725987500986843 MiB/frame`.
- Gap8 positive deltas: `19/19`.

## Decisions

- Feed-forward-only current path is not enough as the primary path.
- Higher q-bit path remains rejected as primary fix.
- Rate-counted side-info fallback is promoted for Stage148 revalidation.
- Index payload is transmitted and counted.
- This is not uncounted teacher side-info and does not permit decoder-side dense target anchors.

## Conclusion

Stage147 identifies the first path that reaches the corrected StreamSplat target range: q6/top10 entropy index+value residual side-info is within about `0.16 dB` of gap4/gap8 corrected targets on the Stage96 broader slice, with all side-info bytes counted. Because Stage96 is not the full paper-protocol video eval, Stage148 must revalidate this fallback before any final quality claim.
