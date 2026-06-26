# Updated Plan Record: Stage56-70

Date: 2026-06-26

## User Request

The user requested updating the future plan to explicitly include large-scale data preparation using the datasets used by StreamSplat, especially DAVIS and YouTube-VOS:

- DAVIS: `https://davischallenge.org/`
- YouTube-VOS: `https://youtube-vos.org/`

The user then asked to begin execution according to this latest plan, with per-stage records continuing under `logs/stage_records/`.

## Update Summary

- Added `logs/FUTURE_WORK_PLAN_STAGE56_70.md` as the latest plan.
- Kept `logs/FUTURE_WORK_PLAN_STAGE56_68.md` as a superseded historical plan.
- Inserted new dataset-focused stages:
  - Stage59: Dataset Download/Prepare Preflight.
  - Stage60: DAVIS/YouTube-VOS Depth Preprocess.
  - Stage61: Large-scale Anchor Export.
- Shifted adapter, selector, side-info, and final package stages to Stage62-70.

## Key Dataset Requirements

DAVIS expected root:

```text
/mnt/hdd2tC/tmp/opencode/datasets/DAVIS/
  ImageSets/2017/train.txt
  ImageSets/2017/val.txt
  JPEGImages/Full-Resolution/<sequence>/*.jpg
  Annotations_unsupervised/Full-Resolution/<sequence>/*.png
  depthImages/Full-Resolution/<sequence>/*_pred.png
```

YouTube-VOS expected root:

```text
/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS/
  train/JPEGImages/<sequence>/*.jpg
  train/Annotations/<sequence>/*.png
  train/depthImages/<sequence>/*_pred.png
  valid/JPEGImages/<sequence>/*.jpg
  valid/depthImages/<sequence>/*_pred.png
```

## Immediate Next Action

Continue with Stage57 compact anchor codec, then Stage58 compression RD ablation, then Stage59 dataset download/preparation preflight for DAVIS and YouTube-VOS.
