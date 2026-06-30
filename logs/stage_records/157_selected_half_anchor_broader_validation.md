# Stage157 Selected Half-Anchor Broader Validation

Date: 2026-06-30

## Goal

Broaden-validate the Stage156 selected Gaussian-domain middle-frame recovery setting on the 120-task sampled protocol used by Stage153/154.

## Plan

- Validate only `best_half_selector`, `keep_fraction=1.0`, `side_bits=6`.
- Use original StreamSplat target-time left/right half-anchors and the target dense anchor only encoder-side to produce entropy residual payloads.
- Decode and render left/right corrected half-anchors.
- Select the better half encoder-side and count one selector byte per intermediate frame.
- Compute PSNR, SSIM, MS-SSIM, LPIPS, payload bytes, and reference direct rate.
- Export bad-case contact sheets outside git.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Success Criteria

- Both gap4 and gap8 remain above `26 dB` sampled middle PSNR.
- SSIM/MS-SSIM improve over original StreamSplat.
- LPIPS does not regress and ideally improves substantially.
- All residual and selector bytes are counted.

## Execution

- Checked `nvidia-smi` before running Python.
- Selected idle `GPU 1` with `CUDA_VISIBLE_DEVICES=1`.
- Command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage157_selected_half_anchor_broader_validation.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage157_selected_half_anchor_broader_validation.py --device cuda --max_tasks 120 --batch_size 1 --keep_fraction 1.0 --side_bits 6
```

## Results

Selected policy: `streamsplat_half_anchor_entropy_residual_best_half`, `keep_fraction=1.0`, `side_bits=6`, one counted half-selector byte.

| gap | tasks | PSNR | p10 PSNR | SSIM | MS-SSIM | LPIPS | p90 LPIPS | original PSNR | original LPIPS | payload bytes | direct rate ref | delta PSNR | delta LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 60 | 29.780485398070507 | 25.726690616240614 | 0.8779375642538071 | 0.9850881884495417 | 0.16601951060195763 | 0.20979373008012772 | 22.06421822428011 | 0.3014947975675265 | 209392.83333333334 | 0.3816307879574477 | 7.7162671737904 | -0.13547528696556885 |
| 8 | 60 | 29.578682359235195 | 25.326197674163037 | 0.8696596751610438 | 0.9838472485542298 | 0.17853523269295693 | 0.2392323151230812 | 20.33727549162514 | 0.3593370050191879 | 215967.88333333333 | 0.3035884102571357 | 9.24140686761006 | -0.18080177232623101 |

## Decision

Stage157 passes the requested middle-frame target on the 120-task Stage153/154 sampled protocol. Both gap4 and gap8 are well above `26-27 dB` and improve SSIM/MS-SSIM/LPIPS over original StreamSplat. This is the current quality-safe Gaussian-domain recovered middle-frame candidate.

## Heavy Contact Sheet

- Worst-LPIPS sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_worst_lpips_contact_sheet.jpg`

## Outputs

- Script: `scripts/run_stage157_selected_half_anchor_broader_validation.py`
- Report: `experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_broader_validation_report.md`
- Package: `experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_broader_validation_package.json`
- Summary: `experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_summary.csv`
- Rows: `experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_rows.csv`
- Bad cases: `experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_badcases.csv`

## Next Step

Package this as the current recovered middle-frame policy, with explicit decoder contract and rate-counting caveats.
