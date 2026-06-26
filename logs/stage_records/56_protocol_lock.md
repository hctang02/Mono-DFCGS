# Stage56 Protocol Lock

Date: 2026-06-26

## Objective

Freeze the experiment/reporting protocol before adding compression, stronger training, feed-forward selector, and side-information variants.

## Commands

```text
nvidia-smi
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage56_protocol_lock.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage56_protocol_lock.py
```

Stage56 is CPU-only and does not use CUDA.

## Outputs

```text
experiments/stage56_protocol_lock/stage56_protocol_lock_summary.json
experiments/stage56_protocol_lock/stage56_protocol_lock_report.md
experiments/stage56_protocol_lock/stage56_rate_accounting_rules.csv
experiments/stage56_protocol_lock/stage56_method_deployability_rules.csv
experiments/stage56_protocol_lock/stage56_standard_table_schemas.csv
scripts/run_stage56_protocol_lock.py
```

## Locked Decisions

| Item | Decision |
|---|---|
| Default quality metric | all-frame PSNR |
| Main rate metric | transmitted Gaussian anchor bitstream MiB/frame |
| Side-info rule | if transmitted, count in side-info rate and total rate |
| Final selector | fully feed-forward frozen predictor + deterministic selection/DP |
| Forbidden final selector inputs | rendered oracle, PSNR lookahead, test-time reconstruction optimization |
| Oracle selector status | upper bound / training label only |
| Training scale | short runs are smoke only; medium/long runs required for strong claims |

## Conclusions

- Stage56 locks the protocol needed for the next contribution stages.
- Compression, adapter, selector, and side-info variants now have explicit reporting rules.
- Future result tables should include size/rate and all-frame PSNR.
- Stage57 should implement compact Gaussian anchor bit-packing and rate validation.

## Caveats

- Stage56 is a protocol/reporting stage, not a new codec or RD experiment.
- It does not change existing Stage51 PSNR/rate numbers.

## Commit

Pending at record creation.
