# Stage167 Adaptive Schedule Rendered Smoke

Date: 2026-06-30

## Goal

Run a small rendered smoke for the Stage165/166 adaptive keyframe schedule to check whether inserted keyframes remain visually/RD plausible beyond the Stage166 label proxy.

## Plan

- Use Stage166 smoke candidate sequences, starting with a small subset rather than all frames.
- Keep Stage158 middle recovery fixed: `streamsplat_guided_half_anchor_entropy_residual_v1`.
- For each sampled target, compare:
  - uniform gap8 adjacent-keyframe segment;
  - Stage165 adaptive adjacent-keyframe segment after inserted keyframes;
  - available uniform gap4/Stage158 row when directly comparable.
- Render and measure PSNR/SSIM/MS-SSIM/LPIPS for a small set of targets.
- Count side-info payload and adaptive schedule metadata where applicable.
- Store lightweight CSV/JSON/report in repo; store any heavy images/contact sheets under `/data/hctang/tmp/opencode/mono_dfcgs_runs/` and do not commit them.

## Success Criteria

- A small rendered smoke package exists with per-target metrics and qualitative risk notes.
- The report states whether Stage165/166 should be scaled to a broader rendered validation.
- No large media/checkpoint/anchor files are committed.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage167_adaptive_schedule_rendered_smoke.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage167_adaptive_schedule_rendered_smoke.py
```

## Outputs

- `experiments/stage167_adaptive_schedule_rendered_smoke/stage167_rendered_smoke_targets.csv`
- `experiments/stage167_adaptive_schedule_rendered_smoke/stage167_rendered_smoke_rows.csv`
- `experiments/stage167_adaptive_schedule_rendered_smoke/stage167_rendered_smoke_summary.csv`
- `experiments/stage167_adaptive_schedule_rendered_smoke/stage167_adaptive_schedule_rendered_smoke_package.json`
- `experiments/stage167_adaptive_schedule_rendered_smoke/stage167_adaptive_schedule_rendered_smoke_report.md`
- Heavy contact sheet, not committed: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage167_adaptive_schedule_rendered_smoke/stage167_adaptive_schedule_rendered_smoke_contact_sheet.jpg`

## Result

- Decision: `inspect_smoke_before_scaling`.
- Target count: `8`, intentionally biased toward Stage166 hard false negatives.
- uniform gap8 rendered: `8 / 8`, PSNR `26.203640896378754`, SSIM `0.8343665823340416`, MS-SSIM `0.9815688729286194`, LPIPS `0.20792134664952755`, mean payload `156389.625` bytes.
- Stage165 adaptive rendered: `8 / 8`, PSNR `26.192677144097146`, SSIM `0.8337735459208488`, MS-SSIM `0.9814087599515915`, LPIPS `0.2088309582322836`, mean payload `155993.625` bytes.
- Adaptive delta vs uniform gap8 on this stress set: PSNR `-0.010963752281610173`, LPIPS `+0.0009096115827560425`.
- uniform gap4 rendered: `7 / 8`, one target is a keyframe; rendered PSNR `25.68324570510097`, SSIM `0.8433339936392648`, MS-SSIM `0.983516446181706`, LPIPS `0.19355605755533492`.

## Interpretation

- For the hard false-negative stress set, Stage165 adaptive is effectively unchanged from uniform gap8 because most missed hard rows were not promoted and remain in the same adjacent segment.
- The result does not invalidate Stage165/166; it shows that the selector's false negatives need subjective inspection before scaling.
- Next step should inspect the contact sheet and then run a complementary positive-coverage smoke on promoted/high-payload sequences such as `motocross-jump`, `cows`, `camel`, and `scooter-black`.
