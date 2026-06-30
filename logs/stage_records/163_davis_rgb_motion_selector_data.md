# Stage163 DAVIS RGB/Motion Selector Data

Date: 2026-06-30

## Goal

Build the first lightweight DAVIS data package for adaptive keyframe selector development under the Stage162 protocol.

## Plan

- Use the Stage157/158 120-task sampled q12 gap4/gap8 rows as the first selector-data slice.
- Compute cheap encoder-side RGB/motion proxy features from input RGB only:
  - left-target/right-target/left-right RGB mean absolute difference;
  - RGB MSE;
  - edge/gradient difference;
  - histogram distance;
  - temporal asymmetry and normalized time.
- Attach Stage158 outcome labels for offline training/evaluation:
  - PSNR, SSIM, MS-SSIM, LPIPS;
  - payload bytes and direct rate reference;
  - deltas vs original StreamSplat.
- Keep rendered metrics and Stage158 quality as labels only, not selector inference inputs.
- Produce sequence/gap summaries and a feature-source compliance report.

## Success Criteria

- Lightweight CSV/JSON/Markdown package exists.
- Each row clearly separates deployable encoder-side features from offline labels.
- Package identifies candidate difficulty signals for later heuristic/learned keyframe selector stages.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage163_davis_rgb_motion_selector_data.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage163_davis_rgb_motion_selector_data.py
```

## Outputs

- `experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_feature_rows.csv`
- `experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_summary.csv`
- `experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_compliance.csv`
- `experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_data_package.json`
- `experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_data_report.md`

## Result

- Packaged `120` Stage157/158 sampled q12 gap4/gap8 task rows.
- Feature resolution: `448x256`.
- Sequence/gap summaries: `60` rows.
- Inference features are derived from DAVIS/input RGB only.
- Stage158 metrics and payloads are attached as offline labels, not selector inference inputs.

## Feature/Label Separation

- Deployable inference feature groups: RGB mean absolute difference, RGB MSE, edge/gradient difference, histogram distance, temporal asymmetry, normalized time, segment length.
- Offline label groups: Stage158 PSNR/SSIM/MS-SSIM/LPIPS, original StreamSplat metrics, deltas, payload bytes, direct total rate, hard-case binary labels.

## Notable Summary Signals

- `motocross-jump` and `scooter-black` have high RGB/motion proxy scores and high payload/high LPIPS flags, matching the intuition that motion-heavy sequences are hard.
- `cows`, `breakdance`, `camel`, and `bike-packing` contain low-PSNR flags despite lower proxy scores in some cases, so future selector should not rely on one RGB difference scalar only.
- Stage164 should convert these row-level features into candidate adaptive schedules and evaluate a first RGB/motion heuristic against uniform gap4/gap8 under Stage158 accounting.
