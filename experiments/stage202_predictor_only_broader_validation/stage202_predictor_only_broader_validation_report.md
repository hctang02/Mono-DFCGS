# Stage202 Predictor-Only Broader Validation

## Decision

- Decision: `predictor_only_broader_training_headroom_not_observed`.
- Best config: `anchor_only_lr1e3`.
- Best eval delta vs linear: `0.0006680372369380905` dB.
- Per-frame payload bytes: `0`.

## Config Results

| config | steps | lr | render weight | linear PSNR | best PSNR | final PSNR | best dPSNR | final dPSNR | best step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_render_lr2e4 | 32 | 0.0002 | 0.02 | 19.239937 | 19.239937 | 19.238465 | 0.000000 | -0.001471 | 0 |
| anchor_only_lr1e3 | 32 | 0.001 | 0.0 | 19.239937 | 19.240605 | 19.240605 | 0.000668 | 0.000668 | 32 |
| render_heavy_lr1e4 | 24 | 0.0001 | 0.1 | 19.239937 | 19.239937 | 19.239130 | 0.000000 | -0.000807 | 0 |

## Gates

| gate | status | value | threshold | detail |
|---|---|---:|---|---|
| metric_rows_ok | pass | 0 | 0 | all rendered metrics must use explicit target-shape alignment |
| endpoint_identity_all_configs | pass | 0.0 | <=1e-6 | t=0/t=1 output equals decoded endpoints |
| predictor_only_payload | pass | 0 | 0 per-frame payload bytes | Stage202 still sends no residual or latent payload |
| any_config_no_regression | pass | 0.0006680372369380905 | >= -0.05 dB | best checkpoint can fall back to zero-init linear |
| predictor_headroom_positive | fail | 0.0006680372369380905 | > 0.05 dB vs linear | best_config=anchor_only_lr1e3; best_step=32 |

## Interpretation

- Stage202 remains predictor-only and transmits no residual/latent payload.
- If `predictor_headroom_positive` fails, Stage203 should prioritize GS-native residual/latent side-info rather than spending more effort on selector training.
- Target dense anchors and RGB remain training/evaluation labels only.

## Outputs

- selected tasks: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage202_predictor_only_broader_validation/stage202_selected_tasks.csv`
- config results: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_config_results.csv`
- summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_summary.csv`
- per-task metrics: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_metrics.csv`
- train log: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_train_log.csv`
- gates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_gates.csv`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_broader_validation_package.json`
