# Stage142 Middle-Frame Protocol Alignment Audit

Date: 2026-06-29

## Goal

Audit the current middle-frame evaluation protocol before doing any large training, because the present Stage78/141 numbers are far below the StreamSplat/paper-level reference.

## Plan

- Compare Stage75 corrected StreamSplat paper-protocol reference against Stage77/78 anchor-only evaluation scope.
- Explicitly record split, sequence scope, window mode, metric space, gaps, middle/given/all metric definitions, and rate accounting.
- Identify whether current `19/20 dB` values are final comparable paper-protocol numbers or diagnostic scoped numbers.
- Produce a target table for the next diagnostic stages.
- Do not rerender in Stage142; use existing artifacts only.
- Check `nvidia-smi` before running Python, even though this package is CPU-only.

## Status

Completed.

## Implementation

Added:

```text
scripts/run_stage142_middle_frame_protocol_alignment_audit.py
```

The script audits Stage75 corrected StreamSplat paper-protocol targets against Stage77/78 anchor-only diagnostics and Stage141 deployable manifest metrics.

## Run

GPU check was performed before execution. GPU3 was idle and used with `CUDA_VISIBLE_DEVICES=3`, although this package is CPU-only.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage142_middle_frame_protocol_alignment_audit.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage142_middle_frame_protocol_alignment_audit.py
```

## Outputs

```text
experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_targets.csv
experiments/stage142_middle_frame_protocol_alignment_audit/stage142_protocol_comparison_rows.csv
experiments/stage142_middle_frame_protocol_alignment_audit/stage142_protocol_alignment_findings.csv
experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_protocol_alignment_summary.json
experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_protocol_alignment_package.json
experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_protocol_alignment_report.md
```

Output size: `36K`.

## Results

- Target Middle-4 corrected local middle PSNR: `23.004337221027775`; paper PSNR: `23.66`.
- Target 8-frame corrected local middle PSNR: `21.56004909948801`; paper PSNR: `22.10`.
- Current q12 adapter gap4 diagnostic middle PSNR: `18.256196169477683`; gap to corrected target: `-4.748141051550093 dB`.
- Current q12 adapter gap8 diagnostic middle PSNR: `17.06969395261803`; gap to corrected target: `-4.490355146869977 dB`.
- Stage78 reference comparison is not a final apples-to-apples claim because Stage75 uses full DAVIS val 30 sequences while Stage77/78 uses 4 scoped DAVIS val sequences.
- Stage141 is decoder-safe but not a paper-level quality solution.

## Conclusion

The middle-frame quality gap is real and too large to treat as a reporting issue. Stage143 must decompose renderer/data/quantization/model contributions, and Stage144 must test high-rate/uncompressed anchors before any more final packaging.
