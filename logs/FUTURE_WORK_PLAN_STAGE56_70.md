# Future Work Plan: Stage56-70

Date: 2026-06-26

This is the latest plan after the user explicitly requested adding StreamSplat-scale datasets, especially DAVIS and YouTube-VOS, into the large-scale training and evaluation path.

## Core Requirements

- Default reporting metric: all-frame PSNR only unless the user explicitly asks for other metrics.
- Every important result should include size/rate tables and all-frame PSNR tables, plus RD curves when applicable.
- Compression must become a concrete contribution, not only q8/q16 plus generic zlib.
- Gaussian adapter training must become a major contribution with medium/long training and stronger baselines, not just a short smoke run.
- Keyframe selector must become fully feed-forward for final claims. It must not use rendered error, PSNR lookahead, or test-time reconstruction optimization.
- Baseline-inspired adapter routes should be explored, including StreamSplat teacher distillation and baseline-style architecture, while preserving final codec test-time inputs.
- Optional side information may be explored if needed, but every transmitted bit must be counted in rate.
- Data scale must expand beyond current local videos to include DAVIS and YouTube-VOS when available.
- Large files, datasets, anchors, checkpoints, rendered caches, and bitstreams stay outside git under `/mnt/hdd2tC/tmp/opencode/...`.
- Each stage gets an individual record under `logs/stage_records/`.

## Dataset Plan

| Dataset/source | Role | Current state | Next action |
|---|---|---|---|
| n3dv | local development/debug | anchors/RD available | keep for fast regression |
| meetroom | local development/debug | anchors/RD available | keep for fast regression |
| NeoVerse driving/robot | local development/debug | anchors/RD available | keep for fast regression |
| DAVIS | StreamSplat-style single-view video training/eval | not detected locally | download/mount, preprocess depth, export anchors |
| YouTube-VOS | StreamSplat-style video training/eval | not detected locally | download/mount, preprocess depth, export anchors |
| long-video set | temporal robustness | not prepared | add after DAVIS/VOS path is stable |

### DAVIS Preparation

Reference: `https://davischallenge.org/`

Expected local root:

```text
/mnt/hdd2tC/tmp/opencode/datasets/DAVIS/
  ImageSets/2017/train.txt
  ImageSets/2017/val.txt
  JPEGImages/Full-Resolution/<sequence>/*.jpg
  Annotations_unsupervised/Full-Resolution/<sequence>/*.png
  depthImages/Full-Resolution/<sequence>/*_pred.png
```

Required steps:

1. Download or mount DAVIS 2017 train/val frames and annotations.
2. Verify StreamSplat `provider_davis.py` layout.
3. Run depth preprocessing to create `depthImages/*_pred.png`.
4. Export dense or multi-gap keyframe Gaussian anchors.
5. Build train/val/test manifests.
6. Use for medium/long adapter training and feed-forward selector training.

### YouTube-VOS Preparation

Reference: `https://youtube-vos.org/`

Expected local root:

```text
/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS/
  train/JPEGImages/<sequence>/*.jpg
  train/Annotations/<sequence>/*.png
  train/depthImages/<sequence>/*_pred.png
  valid/JPEGImages/<sequence>/*.jpg
  valid/depthImages/<sequence>/*_pred.png
```

Required steps:

1. Download or mount YouTube-VOS train/valid frames and training annotations.
2. Verify StreamSplat `provider_vos.py` layout.
3. Generate depth images.
4. Export Gaussian anchors.
5. Build train/val/test manifests.
6. Use for long adapter and selector training.

## Updated Stage Plan

| Stage | Name | Primary goal |
|---:|---|---|
| 56 | Protocol Lock | Completed. Lock all-frame PSNR, rate accounting, selector deployability, and side-info rules. |
| 57 | Compact Anchor Codec | Implement true bit-packing and compact Gaussian anchor containers. |
| 58 | Compression RD Ablation | Compare raw/zlib/bitpacked/delta/per-field quantization using all-frame PSNR and rate. |
| 59 | Dataset Download/Prepare Preflight | Produce explicit DAVIS/YouTube-VOS download, directory, and missing-item checklist. |
| 60 | DAVIS/YouTube-VOS Depth Preprocess | Generate `depthImages/*_pred.png` for mounted datasets. |
| 61 | Large-scale Anchor Export | Export dense/multi-gap Gaussian anchors for local + DAVIS/VOS data. |
| 62 | Adapter Training Infra v2 | Add resume, train/val/test split, best checkpoint, long-run logging. |
| 63 | Medium Adapter Training | Run 5k-20k step training on local + small DAVIS/VOS if available. |
| 64 | Teacher/Baseline Adapter Study | Test StreamSplat teacher distillation and baseline-style adapter designs. |
| 65 | Long Adapter Training | Run 50k-200k step training on larger DAVIS/VOS/long-video data. |
| 66 | Feed-forward Selector Dataset | Build offline labels and features for selector training without test-time oracle. |
| 67 | Medium Feed-forward Selector Training | Train decision-aware / DP-aware deployable selector. |
| 68 | Long Feed-forward Selector Training | Large-data selector training with calibrated fallback to uniform. |
| 69 | Optional Side-info Codec | Explore low-rate depth/motion/residual hints, all counted in total rate. |
| 70 | Final RD Package | Produce final all-frame PSNR tables, size tables, RD curves, and ablations. |

## Contribution Line 1: Compression

Planned components:

| Component | Purpose |
|---|---|
| true bit-packing | q6/q10/q12 avoid uint8/uint16 storage waste |
| per-field quantization | use different bits for rgb/opacity/scale/xyz/rot |
| temporal delta coding | code keyframe anchor changes instead of only absolute anchors |
| entropy coding baseline | provide codec-specific compression beyond generic zlib where feasible |
| importance pruning | reduce or lower precision for low-importance Gaussians |

Required outputs:

- size table: raw, zlib, bitpacked, delta, entropy.
- all-frame PSNR table.
- RD curve: all-frame PSNR vs MiB/frame.
- roundtrip correctness table.

## Contribution Line 2: Gaussian Adapter

Adapter variants:

| Variant | Description |
|---|---|
| Adapter-A | current linear-residual adapter with longer training |
| Adapter-B | larger hidden dimension / deeper network / temporal embeddings |
| Adapter-C | RGB + Gaussian attribute + temporal consistency losses |
| Adapter-D | StreamSplat teacher distillation |
| Adapter-E | baseline-style architecture inspired by StreamSplat temporal/dynamic decoder |

Training scale:

| Level | Steps | Data |
|---|---:|---|
| short | hundreds | debug only |
| medium | 5k-20k | local + small DAVIS/VOS |
| long | 50k-200k | DAVIS/VOS + long videos |

Goal: make all-frame PSNR gains over linear interpolation materially larger than the current small `+0.0798 dB`.

## Contribution Line 3: Feed-forward Keyframe Selector

Final inference form:

```text
input video / encoder-side features
-> frozen predictor
-> deterministic DP / deterministic selector
-> keyframe indices
```

Forbidden at test time:

- rendered oracle error
- PSNR labels
- reconstructed-output lookahead
- optimization over candidate reconstructions

Selector variants:

| Variant | Goal |
|---|---|
| Selector-A | segment cost predictor baseline |
| Selector-B | adaptive-or-uniform classifier |
| Selector-C | pairwise layout ranker |
| Selector-D | DP-aware selector |
| Selector-E | fallback-calibrated selector |

Required outputs:

- uniform vs feed-forward adaptive all-frame PSNR table.
- actual bitstream rate table.
- failure table.
- RD curve vs uniform and oracle upper bound.

## Optional Side-information Line

If keyframe-Gaussian-only quality remains insufficient, explore rate-counted side information.

| Side info | Purpose | Rate rule |
|---|---|---|
| low-resolution depth | geometry/occlusion hint | count side-info rate |
| sparse depth/keypoints | compact geometry hint | count side-info rate |
| motion hint | dynamic interpolation help | count side-info rate |
| Gaussian residual correction | correct hard non-keyframes | count side-info rate |
| importance map | guide selective refinement | count side-info rate |

Recommended naming:

- `Mono-DFCGS-KG`: keyframe-Gaussian-only main method.
- `Mono-DFCGS-SI`: keyframe Gaussian plus side information enhancement.

## Final Reporting Format

Main final table schema:

| Method | Compression | Adapter | Selector | Side info | Rate MiB/frame | All PSNR |
|---|---|---|---|---|---:|---:|
| uniform baseline | zlib/compact anchor | linear | uniform | none | x | x |
| trained adapter | compact anchor | trained | uniform | none | x | x |
| teacher adapter | compact anchor | teacher-distilled | uniform | none | x | x |
| feed-forward adaptive | compact anchor | best adapter | feed-forward | none | x | x |
| side-info variant | compact anchor | best adapter | feed-forward | depth/motion/residual | x | x |
| oracle upper bound | compact anchor | best adapter | oracle | none | x | x |

Required figures:

- main all-frame PSNR RD curve.
- compression ablation RD.
- adapter ablation RD.
- selector ablation RD.
- optional side-info RD.
- long-training curves.

## Immediate Execution Order

1. Record this updated Stage56-70 plan.
2. Continue Stage57 compact anchor codec.
3. Run Stage58 compression RD ablation.
4. Run Stage59 DAVIS/YouTube-VOS download/preparation preflight.
5. Continue through dataset preprocessing, anchor export, medium/long adapter training, feed-forward selector training, side-info exploration, and final RD package.
