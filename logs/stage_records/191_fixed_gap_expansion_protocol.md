# Stage191 Fixed-Gap Expansion Protocol

Date: 2026-07-01

## Goal

Build the full-sequence protocol needed to compare the current adaptive selector against a broader fixed-gap set, not just gap4/gap8.

## Plan

- Generate schedules for `uniform_gap2`, `uniform_gap4`, `uniform_gap6`, `uniform_gap8`, `uniform_gap16`, and the current `stage165_adaptive` schedule over the same 30 DAVIS validation sequences / 1999 frames.
- Reuse the Stage183 measurement-key convention so existing Stage184/186 payload and quality measurements can be reused exactly where possible.
- Report unique q12 keyframe, Stage158 residual, and schedule-packed keyframe-group measurement counts.
- Report existing Stage184/186 reuse coverage and missing measurement counts for the new gaps.
- Do not run rendering, payload measurement, or heavy artifact generation in this stage.

## Success Criteria

- A complete frame/schedule protocol exists for the expanded fixed-gap set.
- Missing measurement counts are explicit for Stage192.
- Existing gap4/gap8/adaptive rows are recognized as reusable.
- Outputs are lightweight CSV/JSON/Markdown only.

## Execution

- Pre-run GPU check: `nvidia-smi` recorded on 2026-07-01 13:52:27; Stage191 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage191_fixed_gap_expansion_protocol.py`
- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage191_fixed_gap_expansion_protocol.py`

## Outputs

- Package: `experiments/stage191_fixed_gap_expansion_protocol/stage191_fixed_gap_expansion_protocol_package.json`
- Report: `experiments/stage191_fixed_gap_expansion_protocol/stage191_fixed_gap_expansion_protocol_report.md`
- Frame/schedule rows: `experiments/stage191_fixed_gap_expansion_protocol/stage191_expanded_fixed_gap_frame_schedule_rows.csv`
- Unique keyframe rows: `experiments/stage191_fixed_gap_expansion_protocol/stage191_unique_keyframe_measurement_rows.csv`
- Unique residual rows: `experiments/stage191_fixed_gap_expansion_protocol/stage191_unique_stage158_residual_measurement_rows.csv`
- Schedule summary: `experiments/stage191_fixed_gap_expansion_protocol/stage191_expanded_fixed_gap_schedule_summary.csv`
- Reuse coverage: `experiments/stage191_fixed_gap_expansion_protocol/stage191_existing_measurement_reuse_coverage.csv`
- Missing measurement manifest: `experiments/stage191_fixed_gap_expansion_protocol/stage191_missing_measurements_for_stage192.csv`

## Results

Schedule counts:

| schedule | frames | keyframes | residual rows | keyframe ratio | metadata bytes |
|---|---:|---:|---:|---:|---:|
| `uniform_gap2` | `1999` | `1025` | `974` | `0.5127563781890946` | `1` |
| `uniform_gap4` | `1999` | `536` | `1463` | `0.2681340670335168` | `1` |
| `uniform_gap6` | `1999` | `372` | `1627` | `0.18609304652326164` | `1` |
| `uniform_gap8` | `1999` | `292` | `1707` | `0.14607303651825912` | `1` |
| `uniform_gap16` | `1999` | `169` | `1830` | `0.08454227113556778` | `1` |
| `stage165_adaptive` | `1999` | `358` | `1641` | `0.1790895447723862` | `327` |

Existing Stage184/186 reuse coverage:

| scope | expected | existing ok | missing | reuse fraction |
|---|---:|---:|---:|---:|
| payload single keyframe | `1065` | `596` | `469` | `0.5596244131455399` |
| payload residual | `7791` | `3472` | `4319` | `0.44564240790655885` |
| payload schedule-packed keyframe group | `180` | `90` | `90` | `0.5` |
| quality single keyframe | `1065` | `596` | `469` | `0.5596244131455399` |
| quality residual | `7791` | `3472` | `4319` | `0.44564240790655885` |

## Decision

- Decision: `measure_expanded_fixed_gap_baselines_next`.
- Stage192 should measure missing `gap2/gap6/gap16` payload and quality rows while reusing existing `gap4/gap8/stage165_adaptive` rows.
- Expanded fixed-gap baselines are required before claiming the adaptive selector beats fixed-gap schedules.
