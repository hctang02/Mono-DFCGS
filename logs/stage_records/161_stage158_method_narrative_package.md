# Stage161 Stage158 Method Narrative Package

Date: 2026-06-30

## Goal

Package the Stage158 recovered middle-frame method as a quality-first, StreamSplat-guided Gaussian-domain innovation before starting keyframe selector work.

## Plan

- Freeze the method name `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Summarize the evidence chain:
  - Stage151 linear-base side-info recovers PSNR but is not the final visual base.
  - Stage153/154 show why original StreamSplat is a better perceptual base.
  - Stage155 image residual is an upper-bound diagnostic only.
  - Stage156/157 validate half-anchor Gaussian residual correction.
  - Stage158 packages the decoder contract and quality gate.
  - Stage159/160 provide subjective visual evidence and per-example size/rate.
- Keep the package lightweight: JSON/CSV/Markdown only.
- Keep heavy videos under `/data/hctang/tmp/opencode/mono_dfcgs_runs/` and reference paths only.

## Success Criteria

- The package states the innovation claim, decoder contract, side-info accounting, quality metrics, sequence-level evidence, and subjective video paths.
- It marks rate as explicitly counted but not currently over-optimized, per user direction.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage161_stage158_method_narrative_package.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage161_stage158_method_narrative_package.py
```

## Outputs

- `experiments/stage161_stage158_method_narrative_package/stage161_stage158_method_narrative_package.json`
- `experiments/stage161_stage158_method_narrative_package/stage161_stage158_method_narrative_report.md`
- `experiments/stage161_stage158_method_narrative_package/stage161_stage158_method_comparison.csv`
- `experiments/stage161_stage158_method_narrative_package/stage161_stage158_evidence_chain.csv`
- `experiments/stage161_stage158_method_narrative_package/stage161_stage158_subjective_sequence_summary.csv`

## Result

- Packaged policy: `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Status: `quality_first_middle_frame_recovery_method_packaged`.
- Rate position: explicitly counted but not over-optimized.
- Innovation claim: original StreamSplat target-time half-anchor is corrected in Gaussian space by counted entropy-coded residual side-info.

## Key Evidence

| gap | method | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes | direct rate ref |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | original StreamSplat | 22.064218 | 0.600809 | 0.801367 | 0.301495 | 0 |  |
| 4 | Stage158 policy | 29.780485 | 0.877938 | 0.985088 | 0.166020 | 209392.833333 | 0.381631 |
| 8 | original StreamSplat | 20.337275 | 0.520331 | 0.700537 | 0.359337 | 0 |  |
| 8 | Stage158 policy | 29.578682 | 0.869660 | 0.983847 | 0.178535 | 215967.883333 | 0.303588 |

## Subjective Evidence

- Stage160 video: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence.mp4`
- Stage160 contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg`

## Next

- Proceed to Stage162 keyframe selector protocol and RGB/motion feature-source/feed-forward audit.
