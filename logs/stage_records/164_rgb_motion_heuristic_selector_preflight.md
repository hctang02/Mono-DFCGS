# Stage164 RGB/Motion Heuristic Selector Preflight

Date: 2026-06-30

## Goal

Evaluate whether cheap RGB/motion features from Stage163 contain enough signal to identify hard middle-frame segments for adaptive keyframe selection.

## Plan

- Use Stage163 row-level features and offline Stage158 labels.
- Keep inference features limited to RGB/motion proxy columns derived from input frames.
- Define offline hard labels from Stage158 outcomes:
  - low PSNR: `Stage158 PSNR < 26`;
  - high LPIPS: `Stage158 LPIPS > 0.22`;
  - high payload: `payload_bytes > 220000`.
- Sweep simple heuristic scores and top-fraction thresholds:
  - RGB motion proxy score;
  - linear-interpolation target RGB difference;
  - left-right RGB difference;
  - edge difference;
  - combined percentile rank score.
- Select a first `rgb_motion_heuristic_v1` hard-segment selector by hard-quality F1, with payload recall as a secondary diagnostic.
- This stage does not yet instantiate full keyframe schedules or render full RD; it is a selector-signal preflight.

## Success Criteria

- A lightweight report identifies whether RGB/motion features can detect hard segments better than random/top-rate guesses.
- Selected heuristic uses only deployable encoder-side features.
- Next stage can convert selected hard windows into actual adaptive keyframe schedules with counted metadata.

## Command

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage164_rgb_motion_heuristic_selector_preflight.py && CUDA_VISIBLE_DEVICES=6 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage164_rgb_motion_heuristic_selector_preflight.py
```

## Outputs

- `experiments/stage164_rgb_motion_heuristic_selector_preflight/stage164_rgb_motion_heuristic_sweep.csv`
- `experiments/stage164_rgb_motion_heuristic_selector_preflight/stage164_rgb_motion_heuristic_selected_rows.csv`
- `experiments/stage164_rgb_motion_heuristic_selector_preflight/stage164_rgb_motion_heuristic_sequence_summary.csv`
- `experiments/stage164_rgb_motion_heuristic_selector_preflight/stage164_rgb_motion_heuristic_selector_preflight_package.json`
- `experiments/stage164_rgb_motion_heuristic_selector_preflight/stage164_rgb_motion_heuristic_selector_preflight_report.md`

## Result

- Best simple heuristic: `edge_left_right`, score column `edge_mad_left_right`, top fraction `0.4`.
- Selected rows: `48 / 120`.
- Hard-quality labels: `30` rows.
- High-payload labels: `72` rows.
- Hard-quality precision/recall/F1: `0.333333 / 0.533333 / 0.410256`.
- High-payload recall: `0.555556`.
- Mean selected PSNR: `28.144267909805347`.
- Mean unselected PSNR: `30.703127857884528`.
- Mean selected payload: `233635.29166666666` bytes.
- Mean unselected payload: `198710.40277777778` bytes.

## Decision

- RGB/motion features have useful signal for payload/difficulty, but a single simple heuristic is not reliable enough as a final selector.
- The heuristic misses important hard-quality cases such as `motocross-jump`, so Stage165 should combine multiple features and possibly include a learned/gated selector or conservative fallback.
- No target dense anchors, rendered metrics, or labels are used as inference features; labels are used only for this preflight evaluation.
