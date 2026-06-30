# Stage181 Full-Sequence RD Accounting Preflight

Date: 2026-06-30

## Goal

Move from Stage172 sampled/proxy rate accounting toward full-sequence/frame accounting while clearly separating exact keyframe/metadata counts from sampled-estimated residual payload costs.

## Plan

- Use Stage165 full-sequence schedules over `30` DAVIS sequences / `1999` frames to count keyframes and metadata.
- Reuse Stage172 main-anchor proxy rates and per-extra-keyframe proxy cost.
- Use Stage180 broader validation payload means as a broader sampled residual-side proxy.
- Report keyframe-only rate, residual proxy rate, metadata rate, and combined Stage172-style total proxy.
- State what remains missing before final full-sequence RD.

## Success Criteria

- Full-sequence keyframe and metadata counts are exact for uniform gap8, Stage165 adaptive, and uniform gap4.
- Residual payload costs are explicitly marked as Stage180 broader sampled estimates, not full-sequence measurements.
- Output includes a total proxy comparison and a requirements checklist for final full-sequence RD.

## Execution

- Checked `nvidia-smi` before running Stage181.
- Compiled `scripts/run_stage181_full_sequence_rd_accounting_preflight.py` with `py_compile`.
- First run failed because Stage165 uses column `adaptive_keyframe_count`, not `stage165_adaptive_keyframe_count`.
- Patched the schedule-column mapping and reran successfully.
- Stage181 is CPU/lightweight and produced no heavy media.

## Result

- Stage181 status: `full_sequence_rd_accounting_preflight_packaged`.
- Decision: `adaptive_rate_promising_under_broader_sampled_proxy`.
- Accounting scope: `full_sequence_keyframe_metadata_plus_stage180_broader_sampled_residual_proxy`.
- Not final full-sequence RD: `true`.
- Total frames: `1999`.
- Sequences: `30`.

## Full-Sequence Keyframe And Metadata Counts

| schedule | keyframes | keyframe ratio | main anchor MiB/frame proxy | metadata bytes | metadata MiB/frame |
|---|---:|---:|---:|---:|---:|
| uniform_gap8 | 292 | 0.146073 | 0.097625386754 | 1 | 0.000000000477 |
| stage165_adaptive | 358 | 0.179090 | 0.120431317266 | 327 | 0.000000156004 |
| uniform_gap4 | 536 | 0.268134 | 0.181938220768 | 1 | 0.000000000477 |

## Stage180 Broader Residual Proxy

| schedule | targets | keyframes | middle recovery | mean payload bytes/target | residual MiB/target | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 90 | 0 | 90 | 219878.578 | 0.209692552355 | 29.206326 | 0.176652 |
| stage165_adaptive | 90 | 56 | 34 | 74673.433 | 0.071214135488 | 29.770753 | 0.142780 |
| uniform_gap4 | 90 | 8 | 82 | 196008.433 | 0.186928208669 | 29.464217 | 0.162457 |

## Combined Proxy

| schedule | main anchor | metadata | residual proxy | total proxy | delta vs gap8 | delta vs gap4 |
|---|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 0.097625386754 | 0.000000000477 | 0.209692552355 | 0.307317939585 | 0.000000000000 | -0.061548490329 |
| stage165_adaptive | 0.120431317266 | 0.000000156004 | 0.071214135488 | 0.191645608757 | -0.115672330828 | -0.177220821157 |
| uniform_gap4 | 0.181938220768 | 0.000000000477 | 0.186928208669 | 0.368866429914 | +0.061548490329 | 0.000000000000 |

## Final-RD Requirements

- Keyframe indices and schedule metadata are now counted exactly for the Stage165 full schedule.
- Main-anchor payload remains a Stage172 interpolated/proxy accounting term.
- Stage158 residual payload remains a Stage180 broader sampled estimate, not all-frame/full-sequence payload measurement.
- Final RD still requires actual q12 keyframe bitstreams for every transmitted keyframe and all-frame residual payload encode for every non-keyframe recovered frame.

## Outputs

- Package: `experiments/stage181_full_sequence_rd_accounting_preflight/stage181_full_sequence_rd_accounting_preflight_package.json`
- Report: `experiments/stage181_full_sequence_rd_accounting_preflight/stage181_full_sequence_rd_accounting_preflight_report.md`
- Keyframe/metadata CSV: `experiments/stage181_full_sequence_rd_accounting_preflight/stage181_full_sequence_keyframe_metadata_accounting.csv`
- Residual proxy CSV: `experiments/stage181_full_sequence_rd_accounting_preflight/stage181_stage180_residual_payload_proxy.csv`
- Total proxy CSV: `experiments/stage181_full_sequence_rd_accounting_preflight/stage181_total_rate_proxy_comparison.csv`
- Requirements CSV: `experiments/stage181_full_sequence_rd_accounting_preflight/stage181_final_rd_requirements.csv`
