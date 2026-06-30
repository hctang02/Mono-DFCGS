# Stage165 Multi-Feature Keyframe Schedule Preflight

Date: 2026-06-30

## Goal

Convert Stage163/164 row-level hard-window signal into the first adaptive keyframe schedule candidate with counted schedule metadata.

## Plan

- Use Stage163 RGB/motion feature rows and Stage157/158 offline labels.
- Keep inference logic limited to deployable RGB/motion features:
  - RGB motion proxy score;
  - RGB left-right difference;
  - RGB linear-interpolation-to-target proxy;
  - edge left-right difference;
  - histogram left-right distance.
- Sweep multi-feature rank gates:
  - threshold percentiles;
  - minimum number of feature votes.
- Select a candidate by hard-quality F1 and payload recall.
- Convert selected hard windows into an adaptive schedule:
  - start from uniform gap8 keyframes;
  - insert selected target frame indices as extra keyframes;
  - keep endpoints/last frame included;
  - count adaptive keyframe-index metadata with `ceil(log2(total_frames))` bits/index plus a mode-id byte.
- Compare metadata/keyframe count against uniform gap4 and uniform gap8.
- Do not claim final RD quality yet; this is a pre-render schedule preflight.

## Success Criteria

- A schedule package exists with per-sequence adaptive indices and metadata accounting.
- The selected heuristic uses only Stage162-allowed inference features.
- The report clearly states label coverage, false-negative risk, and that rendered RD is still pending.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage165_multifeature_keyframe_schedule_preflight.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage165_multifeature_keyframe_schedule_preflight.py
```

## Outputs

- `experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_gate_sweep.csv`
- `experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_selected_rows.csv`
- `experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv`
- `experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_summary.csv`
- `experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_keyframe_schedule_preflight_package.json`
- `experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_keyframe_schedule_preflight_report.md`

## Result

- Selected gate: rank threshold `0.65`, minimum feature votes `1`.
- Selected rows: `70 / 120`.
- Hard-quality precision/recall/F1: `0.314286 / 0.733333 / 0.44`.
- Payload precision/recall: `0.842857 / 0.819444`.
- Selected mean payload: `232024.85714285713` bytes.
- Unselected mean payload: `185598.06` bytes.

## Schedule Summary

- Schedule: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1`.
- Logic: start from uniform gap8 and add selected target indices as extra keyframes.
- Sequence count: `30`.
- Total frames: `1999`.
- Total adaptive keyframes: `358`.
- Mean adaptive keyframe ratio: `0.1790895447723862`.
- Metadata: `2610` bits / `327` bytes / `0.00031113624572753906` MiB.
- Selected hard-quality rows: `22 / 30`.
- Selected high-payload rows: `59 / 72`.

## Decision

- Multi-feature gating is better than Stage164 single-feature edge gating and recovers `motocross-jump` hard cases.
- This is still a pre-render schedule candidate. Stage166 should evaluate label/RD implications versus uniform gap4/gap8 and decide whether to render a small smoke.
- Decoder only receives transmitted schedule/keyframes; it does not reproduce RGB/motion feature extraction.
