# Stage152 Subjective Visual Export

Date: 2026-06-30

## Goal

Generate human-viewable visual outputs for the recovered Stage151 policy so subjective quality can be inspected directly.

## Plan

- Use the Stage150/151 recovered policy: q12 endpoint anchors, decoder-safe linear base, q6/top10 entropy index+value residual side-info.
- Export side-by-side mp4 videos for q12 gap4 and gap8 sampled eval tasks.
- Panels: target RGB, linear base render, recovered side-info render.
- Store videos outside git under `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export`.
- Store only lightweight manifest/report files under `experiments/stage152_subjective_visual_export`.
- Check `nvidia-smi` before running Python.

## Success Criteria

- MP4 videos are generated for gap4 and gap8.
- Manifest records video paths and frame/task counts.
- No large video files are committed to git.

## Execution

- Checked `nvidia-smi` before running Python.
- Selected idle `GPU 1` via `CUDA_VISIBLE_DEVICES=1`.
- Command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage152_subjective_visual_export.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage152_subjective_visual_export.py --device cuda --frames_per_gap 24 --fps 3 --disable_cache
```

## Results

| gap | frames | mean base PSNR | mean recovered PSNR | mean delta PSNR | mean payload bytes |
|---:|---:|---:|---:|---:|---:|
| 4 | 24 | 20.934637105918473 | 24.157467503772878 | 3.2228303978544113 | 29700.708333333332 |
| 8 | 24 | 19.312465970173346 | 22.586569251235034 | 3.2741032810616915 | 29989.375 |

## Video Outputs

- Gap4 mp4: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap4_target_base_recovered.mp4`
- Gap4 contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap4_contact_sheet.jpg`
- Gap8 mp4: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap8_target_base_recovered.mp4`
- Gap8 contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap8_contact_sheet.jpg`

## Lightweight Repo Outputs

- Script: `scripts/run_stage152_subjective_visual_export.py`
- Summary: `experiments/stage152_subjective_visual_export/stage152_subjective_visual_export_summary.json`
- Package: `experiments/stage152_subjective_visual_export/stage152_subjective_visual_export_package.json`
- Report: `experiments/stage152_subjective_visual_export/stage152_subjective_visual_export_report.md`
- Frame rows: `experiments/stage152_subjective_visual_export/stage152_subjective_visual_frames.csv`

## Status

Completed. Stage152 generated subjective comparison videos for the Stage151 recovered policy. The committed artifacts should remain lightweight; videos and contact sheets stay outside git under the heavy output path.
