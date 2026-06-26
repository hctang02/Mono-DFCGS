# Stage59 DAVIS/YouTube-VOS Prepare Preflight

Date: 2026-06-26

## Goal

Run a focused download/preparation preflight for the StreamSplat-scale datasets requested by the user: DAVIS and YouTube-VOS.

## Code

```text
scripts/run_stage59_davis_vos_prepare_preflight.py
```

The script checks candidate dataset roots, expected StreamSplat provider layout, provider file availability, and produces a manual download/prepare checklist.

## Verification

GPU status was checked before running code with `nvidia-smi`. Stage59 only inspects filesystem paths and does not use CUDA.

Commands:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m compileall scripts/run_stage59_davis_vos_prepare_preflight.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage59_davis_vos_prepare_preflight.py
```

## Outputs

```text
experiments/stage59_davis_vos_prepare_preflight/stage59_dataset_root_status.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_expected_provider_layout.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_streamsplat_provider_check.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_download_prepare_checklist.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_davis_vos_prepare_report.md
experiments/stage59_davis_vos_prepare_preflight/stage59_davis_vos_prepare_preflight_summary.json
```

## Results

- DAVIS provider-ready roots: `0`.
- DAVIS anchor-export-ready roots: `0`.
- YouTube-VOS provider-ready roots: `0`.
- YouTube-VOS anchor-export-ready roots: `0`.
- StreamSplat DAVIS provider exists: `/mnt/hdd2tC/tmp/opencode/StreamSplat/datasets/provider_davis.py`.
- StreamSplat YouTube-VOS provider exists: `/mnt/hdd2tC/tmp/opencode/StreamSplat/datasets/provider_vos.py`.

## Expected Primary Roots

DAVIS:

```text
/mnt/hdd2tC/tmp/opencode/datasets/DAVIS/
  ImageSets/2017/train.txt
  ImageSets/2017/val.txt
  JPEGImages/Full-Resolution/<sequence>/*.jpg
  Annotations_unsupervised/Full-Resolution/<sequence>/*.png
  depthImages/Full-Resolution/<sequence>/*_pred.png
```

YouTube-VOS:

```text
/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS/
  train/JPEGImages/<sequence>/*.jpg
  train/Annotations/<sequence>/*.png
  train/depthImages/<sequence>/*_pred.png
  valid/JPEGImages/<sequence>/*.jpg
  valid/depthImages/<sequence>/*_pred.png
```

## Download/Prepare Checklist

| step | dataset | action | url | target |
|---:|---|---|---|---|
| 1 | DAVIS | Download or mount DAVIS 2017 train/val images and unsupervised annotations | https://davischallenge.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS` |
| 2 | DAVIS | Verify split txt, frames, and annotations | https://davischallenge.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS` |
| 3 | DAVIS | Generate `depthImages/Full-Resolution/<sequence>/*_pred.png` | local Stage60/StreamSplat preprocess | `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS/depthImages/Full-Resolution` |
| 4 | YouTube-VOS | Download or mount train/valid JPEGImages and train Annotations | https://youtube-vos.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS` |
| 5 | YouTube-VOS | Verify train and valid sequence folders | https://youtube-vos.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS` |
| 6 | YouTube-VOS | Generate `train/valid/depthImages/<sequence>/*_pred.png` | Stage60 script needed | `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS` |

## Blocking Status

Stage60 depth preprocessing and Stage61 large-scale anchor export are blocked until DAVIS and/or YouTube-VOS data is downloaded or mounted under the expected roots, or equivalent roots are supplied via script arguments.

## Post-Preflight Data Acquisition Update

Date: 2026-06-27

After the initial Stage59 preflight, DAVIS and part of YouTube-VOS were downloaded to the expected external dataset root.

### DAVIS

- Downloaded DAVIS 2017 Full-Resolution unsupervised trainval zip.
- Extracted to `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS`.
- Removed the zip archive after extraction.
- Ran Stage60 DepthAnything V2 preprocessing for DAVIS.
- Reran Stage59 preflight after depth generation.
- Latest DAVIS provider-ready roots: `1`.
- Latest DAVIS anchor-export-ready roots: `1`.

### YouTube-VOS

- Installed `gdown` in `/mnt/hdd2tC/tmp/opencode/streamsplat_venv`.
- Downloaded YouTube-VOS 2019 `valid.tar` from Google Drive id `1bw8KcpzfrT08HYbuROZmY0bp4TkYl4_g`.
- Extracted valid split to `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS/valid`.
- Removed `valid.tar` after extraction.
- Probed YouTube-VOS 2019 `train.tar` from Google Drive id `1lU9jCX-H0ntwh87tt2cA0xEPeWOJzD6S`; reported size is about `9.26G`.
- Did not complete train download because `/mnt/hdd2tC` does not have enough free space for the tar plus extraction.
- Removed the partial `.part` file.
- Latest YouTube-VOS provider-ready roots: `0`.
- Latest YouTube-VOS anchor-export-ready roots: `0`.
- Missing items: `train/JPEGImages`, `train/Annotations`, and depth images.

### Latest Stage59 Summary

```text
experiments/stage59_davis_vos_prepare_preflight/stage59_dataset_root_status.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_davis_vos_prepare_preflight_summary.json
experiments/stage59_davis_vos_prepare_preflight/stage59_davis_vos_prepare_report.md
```

Latest result:

| Dataset | Provider-ready roots | Anchor-export-ready roots | Status |
|---|---:|---:|---|
| DAVIS | 1 | 1 | ready for Stage61 anchor export |
| YouTube-VOS | 0 | 0 | blocked by missing train split and depth |
