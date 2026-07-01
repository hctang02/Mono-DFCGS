# Stage199 Learned GS Training Manifest

## Decision

- Decision: `manifest_ready_for_stage200_architecture_package`.
- Task rows: `29204`.
- Missing reference rows: `0`.
- No anchors, checkpoints, residual tensors, or heavy payloads are copied.

## Summary

| split | codec | gap | sequences | segments | tasks | labels |
|---|---|---:|---:|---:|---:|---:|
| eval | q12 | 2 | 30 | 974 | 974 | 974 |
| eval | q12 | 4 | 30 | 495 | 1463 | 1463 |
| eval | q12 | 6 | 30 | 334 | 1627 | 1627 |
| eval | q12 | 8 | 30 | 252 | 1707 | 1707 |
| eval | q12 | 12 | 30 | 176 | 1788 | 1788 |
| eval | q12 | 16 | 30 | 132 | 1830 | 1830 |
| train | q12 | 2 | 60 | 2055 | 2055 | 2055 |
| train | q12 | 4 | 60 | 1048 | 3087 | 3087 |
| train | q12 | 6 | 60 | 703 | 3430 | 3430 |
| train | q12 | 8 | 60 | 540 | 3604 | 3604 |
| train | q12 | 12 | 60 | 369 | 3776 | 3776 |
| train | q12 | 16 | 60 | 283 | 3863 | 3863 |

## Split Coverage

| split | sequences | frames |
|---|---:|---:|
| eval | 30 | 1999 |
| train | 60 | 4209 |

## Contract Audit

| audit | status | value | detail |
|---|---|---:|---|
| dense_anchor_coverage | pass | 0 | missing_sources=0; missing_pair_items=0; non_contiguous_sequences=0 |
| rgb_label_coverage | pass | 0 | target RGB is training/encoder-side only, never decoder-side image residual |
| split_separation | pass | 0 |  |
| gap_coverage | pass | 0 |  |
| stage197_decoder_contract | pass | 0 | manifest fields mark those sources as training/encoder-side labels only; payloads must be GS-native and counted |
| lightweight_reference_only | pass | 0 | CSV rows contain existing file paths and metadata only |

## Decoder Contract Notes

- Runtime decoder inputs: transmitted GS keyframes, transmitted schedule metadata, normalized time, shared weights, and counted GS-native latent/residual payloads.
- Training/encoder-only labels: target dense anchors and target RGB render losses.
- Forbidden decoder inputs: target dense anchors, target RGB/image residuals, and oracle schedule/quality labels.

## Outputs

- tasks: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv`
- summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_summary.csv`
- sequence coverage: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage199_learned_gs_training_manifest/stage199_sequence_coverage.csv`
- contract audit: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage199_learned_gs_training_manifest/stage199_contract_audit.csv`
- missing references: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage199_learned_gs_training_manifest/stage199_missing_references.csv`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_manifest_package.json`
