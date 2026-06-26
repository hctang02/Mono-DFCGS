# Stage59 DAVIS/YouTube-VOS Prepare Preflight

## Summary

- DAVIS provider-ready roots: `0`.
- DAVIS anchor-export-ready roots: `0`.
- YouTube-VOS provider-ready roots: `0`.
- YouTube-VOS anchor-export-ready roots: `0`.

## Root Status

| Dataset | Root | Exists | Provider-ready | Depth-preprocess-ready | Anchor-export-ready | Missing | Next action |
|---|---|---|---|---|---|---|---|
| DAVIS | `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS` | false | false | false | false | root | download or mount missing DAVIS files |
| DAVIS | `/mnt/hdd2tC/tmp/opencode/datasets/davis` | false | false | false | false | root | download or mount missing DAVIS files |
| DAVIS | `/mnt/hdd2tC/tmp/opencode/DAVIS` | false | false | false | false | root | download or mount missing DAVIS files |
| DAVIS | `/mnt/hdd2tC/datasets/DAVIS` | false | false | false | false | root | download or mount missing DAVIS files |
| DAVIS | `/mnt/hdd2tC/haocheng/datasets/DAVIS` | false | false | false | false | root | download or mount missing DAVIS files |
| YouTube-VOS | `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS` | false | false | false | false | root | download or mount missing YouTube-VOS files |
| YouTube-VOS | `/mnt/hdd2tC/tmp/opencode/datasets/youtube_vos` | false | false | false | false | root | download or mount missing YouTube-VOS files |
| YouTube-VOS | `/mnt/hdd2tC/tmp/opencode/YouTube-VOS` | false | false | false | false | root | download or mount missing YouTube-VOS files |
| YouTube-VOS | `/mnt/hdd2tC/datasets/YouTube-VOS` | false | false | false | false | root | download or mount missing YouTube-VOS files |
| YouTube-VOS | `/mnt/hdd2tC/haocheng/datasets/YouTube-VOS` | false | false | false | false | root | download or mount missing YouTube-VOS files |

## Download/Prepare Checklist

| Step | Dataset | Action | URL | Target | Blocking |
|---:|---|---|---|---|---|
| 1 | DAVIS | Download or mount DAVIS 2017 train/val images and unsupervised annotations | https://davischallenge.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS` | yes_if_root_missing |
| 2 | DAVIS | Verify ImageSets/2017 train.txt and val.txt plus JPEGImages/Full-Resolution and Annotations_unsupervised/Full-Resolution | https://davischallenge.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS` | yes_if_missing |
| 3 | DAVIS | Generate depthImages/Full-Resolution/<sequence>/*_pred.png | local StreamSplat preprocess_depth_davis.py or Mono-DFCGS Stage60 script | `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS/depthImages/Full-Resolution` | yes_for_anchor_export |
| 4 | YouTube-VOS | Download or mount YouTube-VOS train and valid JPEGImages plus train Annotations | https://youtube-vos.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS` | yes_if_root_missing |
| 5 | YouTube-VOS | Verify train/JPEGImages, train/Annotations, and valid/JPEGImages sequence folders | https://youtube-vos.org/ | `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS` | yes_if_missing |
| 6 | YouTube-VOS | Generate train/valid/depthImages/<sequence>/*_pred.png | Mono-DFCGS Stage60 script to be added | `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS` | yes_for_anchor_export |
