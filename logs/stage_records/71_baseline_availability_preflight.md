# Stage71 Baseline Availability Preflight

Date: 2026-06-27

## Goal

Inventory local FCGS/D-FCGS/CWGS baseline code and old artifacts, then determine whether any result is ready for DAVIS scoped apples-to-apples comparison.

This stage is a lightweight preflight. It does not run baseline training, rendering, or compression.

## Inputs

```text
/mnt/hdd2tC/hctang/third_party/FCGS
/mnt/hdd2tC/hctang/third_party/D-FCGS
experiments/stage52_fcgs_dfcgs_baseline_preflight/
experiments/stage53_baseline_comparison_scaffold/
experiments/stage70_scoped_davis_rd_package/
```

## Expected Outputs

```text
experiments/stage71_baseline_availability_preflight/stage71_baseline_code_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_artifact_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_fairness_status.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_missing_fields.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_summary.json
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_report.md
```

## Notes

- Default metric remains all-frame PSNR.
- Baseline rate must be explicitly scoped. Full FCGS/D-FCGS codec MiB/frame is not the same as Mono-DFCGS keyframe-anchor MiB/frame.
- Old non-DAVIS artifacts can inform implementation readiness but should not be promoted to final DAVIS apples-to-apples results.
- Heavy baseline outputs, bitstreams, checkpoints, and datasets must stay outside git.

## Execution

运行前按要求使用 `nvidia-smi` 检查 GPU。Stage71 是 CPU-only file/inventory scan。

Command:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage71_baseline_availability_preflight.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage71_baseline_availability_preflight.py
```

## Outputs

Tracked outputs:

```text
experiments/stage71_baseline_availability_preflight/stage71_baseline_code_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_artifact_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_fairness_status.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_missing_fields.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_summary.json
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_report.md
```

Output size:

```text
32K experiments/stage71_baseline_availability_preflight
```

## Results

| item | value |
|---|---:|
| fairness-ready methods | `0` |
| not-ready methods | `3` |
| FCGS high-priority missing items | `5` |
| D-FCGS high-priority missing items | `5` |
| CWGS high-priority missing items | `3` |

Code inventory:

| method | local code status | DAVIS mentions | checkpoint count | missing tools |
|---|---|---:|---:|---|
| `FCGS` | `present` | `0` | `5` | `tmc3` |
| `D-FCGS` | `present` | `0` | `1` | `tmc3` |
| `CWGS` | `missing_optional` | `0` | `0` | `code_checkout` |

Artifact inventory:

| group | records | DAVIS-related | rate rows | quality rows | fair rows |
|---|---:|---:|---:|---:|---:|
| `stage52_fcgs_dfcgs_summary_records` | `199` | `0` | `199` | `173` | `0` |
| `stage53_external_baseline_rows` | `199` | `0` | `199` | `173` | `0` |
| `stage70_baseline_status` | `3` | `3` | `0` | `0` | `0` |
| `legacy_cwgs_rd_summaries` | `264` | `0` | `264` | `264` | `0` |

## Conclusion

- No local FCGS/D-FCGS/CWGS artifact is ready for Stage70 DAVIS scoped apples-to-apples RD comparison.
- FCGS and D-FCGS code are present, but no DAVIS/Mono-DFCGS adapter is detected.
- Old Stage52/53 and CWGS artifacts remain diagnostic/reference-only because data, protocol, and rate scopes differ.
- Next fair-baseline work should start with a DAVIS-specific FCGS wrapper or a documented D-FCGS compatibility decision.
