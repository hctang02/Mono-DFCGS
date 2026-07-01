# Stage201 Predictor-Only Smoke

## Decision

- Decision: `predictor_only_smoke_passed_no_regression_gate`.
- Train tasks: `8`; eval tasks: `8`.
- Best step: `0`; best eval PSNR: `20.52573876541323`.
- Per-frame payload bytes: `0`.

## Summary

| phase | split | tasks | PSNR | dPSNR vs linear | dPSNR vs old adapter ref | anchor MSE | render MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| linear | eval | 8 | 20.525739 | 0.000000 | 2.862794 | 0.00770555 | 0.01514361 |
| linear | train | 8 | 19.382861 | 0.000000 | 1.719916 | 0.00827749 | 0.01424690 |
| predictor_best | eval | 8 | 20.525739 | 0.000000 | 2.862794 | 0.00770555 | 0.01514361 |
| predictor_best | train | 8 | 19.382861 | 0.000000 | 1.719916 | 0.00827749 | 0.01424690 |
| predictor_final | eval | 8 | 20.509261 | -0.016477 | 2.846316 | 0.00770551 | 0.01512540 |
| predictor_final | train | 8 | 19.382482 | -0.000379 | 1.719537 | 0.00827740 | 0.01422850 |
| predictor_initial | eval | 8 | 20.525739 | 0.000000 | 2.862794 | 0.00770555 | 0.01514361 |

## Gates

| gate | status | value | threshold | detail |
|---|---|---:|---|---|
| metric_rows_ok | pass | 0 | 0 | render/target shapes are explicitly aligned; errors indicate discarded metrics |
| endpoint_identity | pass | 0.0 | <=1e-6 | t=0/t=1 predictor output must match transmitted endpoints |
| predictor_only_payload | pass | 0 | 0 per-frame payload bytes | no residual or latent payload is transmitted in Stage201 |
| best_eval_no_regression_vs_linear | pass | 0.0 | >= -0.05 dB | best=20.52573876541323; linear=20.52573876541323; best_checkpoint=/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke/temporal_basis_gs_refiner_best.safetensors |
| best_eval_vs_stage78_old_adapter_reference | pass | 2.862793704365373 | >0 dB vs Stage78 q12 old adapter mean for smoke gaps | reference is historical and protocol-different; used only as sanity floor |
| final_checkpoint_written | pass | /data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke/temporal_basis_gs_refiner_final.safetensors | exists outside git | checkpoint is intentionally saved under heavy_root, not committed |

## Decoder Contract

- Decoder inputs are q12 left/right GS keyframes, normalized time, shared `TemporalBasisGSRefiner` weights, and transmitted schedule metadata.
- Stage201 transmits no residual or latent payload and uses zero per-frame side-info bytes.
- Target dense anchors and target RGB are used only for training/evaluation labels, never as decoder inputs.

## Outputs

- selected tasks: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage201_predictor_only_smoke/stage201_selected_tasks.csv`
- per-task metrics: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage201_predictor_only_smoke/stage201_predictor_only_smoke_metrics.csv`
- summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage201_predictor_only_smoke/stage201_predictor_only_smoke_summary.csv`
- train log: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage201_predictor_only_smoke/stage201_predictor_only_train_log.csv`
- gates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage201_predictor_only_smoke/stage201_predictor_only_gates.csv`
- best checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke/temporal_basis_gs_refiner_best.safetensors`
- final checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke/temporal_basis_gs_refiner_final.safetensors`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage201_predictor_only_smoke/stage201_predictor_only_smoke_package.json`
