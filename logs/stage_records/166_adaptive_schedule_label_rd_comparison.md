# Stage166 Adaptive Schedule Label/RD Comparison

Date: 2026-06-30

## Goal

Compare the Stage165 adaptive keyframe schedule against uniform gap4 and uniform gap8 using lightweight label/RD accounting before any heavy rendered validation.

## Plan

- Use Stage165 selected rows and adaptive schedule rows.
- Use Stage163 Stage158 metrics/payload labels as offline diagnostics only.
- Compare schedule-level keyframe counts and metadata for:
  - uniform gap8;
  - Stage165 adaptive gap8 plus extra target keyframes;
  - uniform gap4.
- Estimate row-level consequences on the 120 sampled middle-frame tasks:
  - which rows would remain middle recovery tasks;
  - which selected target frames would be promoted to keyframes;
  - hard-quality and high-payload label coverage;
  - approximate residual payload avoided by promotion.
- Count schedule metadata explicitly for all compared schedules.
- Pick a small rendered-smoke candidate set based on hard-label coverage, false negatives, and high-payload selected rows.
- Do not claim final rendered RD; this is a pre-render decision package.

## Success Criteria

- A Stage166 package/report exists with schedule-level and row-level comparisons.
- The report clearly states whether Stage165 adaptive scheduling is promising enough for rendered validation.
- The smoke set is small and targeted.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage166_adaptive_schedule_label_rd_comparison.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage166_adaptive_schedule_label_rd_comparison.py
```

## Outputs

- `experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_schedule_comparison.csv`
- `experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_sampled_row_consequences.csv`
- `experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_sequence_label_coverage.csv`
- `experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_smoke_candidates.csv`
- `experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_adaptive_schedule_label_rd_comparison_package.json`
- `experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_adaptive_schedule_label_rd_comparison_report.md`

## Result

- Decision: `promising_for_small_rendered_smoke`.
- Adaptive schedule keyframes: `358`, between uniform gap8 `292` and uniform gap4 `536`.
- Adaptive metadata: `2610` bits / `327` bytes.
- Main-anchor MiB/frame proxy: `0.12043131726560583`.
- Total proxy MiB/frame: `0.19418151582689588`.
- Sampled target promotions: `70 / 120`.
- Hard-quality sampled coverage: `22 / 30`.
- High-payload sampled coverage: `59 / 72`.
- Sampled residual payload avoided: `16241740` bytes.
- Remaining hard false negatives: `8` sampled rows.

## Smoke Candidates

- `motocross-jump`
- `cows`
- `camel`
- `breakdance`
- `dance-twirl`
- `scooter-black`
- `india`
- `shooting`
- `car-roundabout`
- `bike-packing`

## Limitations

- This is a pre-render proxy only. Promoted rows mean the sampled target index becomes a keyframe; this does not replace rendered uniform-gap quality evaluation.
- Inserted keyframes change subsequent interpolation intervals, so Stage167 should run a small rendered smoke before claiming final RD.
