# Stage151 Middle-Frame Recovery Policy Package

Date: 2026-06-30

## Goal

Package the Stage150 result as the current middle-frame recovery policy: q12 endpoint anchors plus linear-base q6/top10 entropy index+value residual side-info, with all payload bytes counted.

## Plan

- Read Stage150 package outputs.
- Emit a compact policy manifest and report.
- Explicitly record achieved corrected target gaps for q12 gap4/gap8 full eval rows.
- Preserve decoder contract: endpoint anchors, normalized time, and encoded payload only.
- Mark target dense anchors/RGB/unencoded residual tensors as forbidden decoder inputs.
- Check `nvidia-smi` before Python execution even though this package is CPU-only.

## Success Criteria

- Policy shows both gap4 and gap8 exceed corrected StreamSplat middle-frame targets.
- Policy records direct/amortized total rates and payload bytes.
- Package is lightweight and commit-safe.

## Note

This packages the achieved middle-frame recovery for full q12 gap4/gap8 eval rows. Further optional work can add full-video RD plots, all-gap coverage, and lower-rate refinements, but the immediate middle-frame PSNR recovery target is met by Stage150.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage151_middle_frame_recovery_policy_package.py
```

## Run

GPU check was performed before execution. GPU1 was idle and used via `CUDA_VISIBLE_DEVICES=1`, although this package is CPU-only.

Syntax check and package run:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage151_middle_frame_recovery_policy_package.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage151_middle_frame_recovery_policy_package.py
```

## Outputs

```text
experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy.json
experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_evidence.csv
experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy_summary.json
experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy_package.json
experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy_report.md
```

Output size: `28K`.

## Results

- Policy: `middle_frame_recovery_linear_base_entropy_sideinfo_v1`.
- Status: `target_recovered_on_full_q12_gap4_gap8_eval_rows`.
- Target recovery: `true`.
- Minimum achieved margin over corrected target: `+0.10055620282386002 dB`.
- Maximum entropy decode diff vs fixed decode: `0.0`.
- Minimum positive delta fraction: `1.0`.
- Gap4: target `23.004337221027775`, achieved `23.104893423851635`, margin `+0.10055620282386002`, direct rate `0.21060840528077832 MiB/frame`.
- Gap8: target `21.56004909948801`, achieved `22.020188948523128`, margin `+0.4601398490351194`, direct rate `0.12643008870779784 MiB/frame`.

## Conclusion

Stage151 freezes the recovered middle-frame policy. The immediate target has been met on full q12 gap4/gap8 eval rows with rate-counted side-info and a decoder-safe contract. Optional next work is RD presentation/refinement, not emergency quality rescue.
