# Stage168 Positive-Coverage Rendered Smoke

Date: 2026-06-30

## Goal

Run a complementary rendered/visual smoke on Stage165 adaptive targets that are promoted to keyframes, focusing on hard-quality or high-payload rows that Stage167 did not cover.

## Plan

- Use Stage166 sampled-row consequences and smoke candidates.
- Select a small set of promoted targets where Stage165 adaptive makes the target a keyframe.
- Prioritize hard-quality and high-payload rows in sequences such as `motocross-jump`, `cows`, `camel`, `scooter-black`, `india`, `shooting`, and `car-roundabout`.
- Render uniform gap8 Stage158 recovery for those targets to measure what adaptive promotion avoids.
- Mark Stage165 adaptive as `target_keyframe_no_middle_render` for promoted targets; do not pretend this is rendered keyframe RD.
- Render uniform gap4 where the target is not already a uniform gap4 keyframe.
- Export lightweight CSV/JSON/report in repo and heavy contact sheet under `/data/hctang/tmp/opencode/mono_dfcgs_runs/` only.

## Success Criteria

- A positive-coverage smoke package exists and complements Stage167's hard-false-negative stress smoke.
- The report states whether promoted targets are genuinely hard/expensive under uniform gap8.
- No heavy media/checkpoint/anchor files are committed.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage168_positive_coverage_rendered_smoke.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage168_positive_coverage_rendered_smoke.py
```

## Outputs

- `experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_targets.csv`
- `experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rows.csv`
- `experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_summary.csv`
- `experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rendered_smoke_package.json`
- `experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rendered_smoke_report.md`
- Heavy contact sheet, not committed: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rendered_smoke_contact_sheet.jpg`

## Result

- Decision: `positive_promotions_confirmed_for_broader_validation`.
- Target count: `8` promoted targets.
- All `8 / 8` selected targets have both hard-quality and high-payload labels.
- Stage165 adaptive: `8 / 8` are target keyframes, so no middle render is run or claimed.
- uniform gap8 rendered recovery on the same targets: PSNR `27.889178289574676`, SSIM `0.8163857758045197`, MS-SSIM `0.9782927110791206`, LPIPS `0.21933318674564362`, mean residual payload `242546.25` bytes.
- uniform gap4 rendered recovery: `7 / 8` rendered and one target keyframe; PSNR `28.175228642729053`, LPIPS `0.23104432225227356`, mean residual payload `232906.14285714287` bytes.

## Interpretation

- Stage168 complements Stage167: promoted rows are genuinely expensive/hard under uniform gap8, so adaptive promotion is meaningful on the positive side.
- This still does not measure rendered q12 keyframe reconstruction quality; it only validates that adaptive avoids costly middle recovery on selected hard/high-payload targets.
- Next step should combine Stage167 false-negative stress cases and Stage168 positive promotions into a broader rendered validation design.
