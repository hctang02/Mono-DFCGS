# Stage 8 Dataset Setup Requirements

Stage 8 prepares sequence manifests for DAVIS and YouTube-VOS before Mono-DFCGS anchor export.

## Current Status

- Dataset roots with sequences detected: 0
- Dataset roots checked: 12

## Required DAVIS Layout

```text
DAVIS/
  ImageSets/2017/train.txt
  ImageSets/2017/val.txt
  JPEGImages/Full-Resolution/<sequence>/*.jpg
  Annotations_unsupervised/Full-Resolution/<sequence>/*.png
  depthImages/Full-Resolution/<sequence>/*.png
```

`JPEGImages/480p`, `Annotations/480p`, and `depthImages/480p` are also accepted by this manifest script, but StreamSplat provider compatibility may require path adaptation.

## Required YouTube-VOS Layout

```text
YouTube-VOS/
  train/JPEGImages/<sequence>/*.jpg
  valid/JPEGImages/<sequence>/*.jpg
  train/Annotations/<sequence>/*.png
  train/depthImages/<sequence>/*.png
```

## Next Action

Mount or download a DAVIS 2017 root into one of the checked paths, or pass it explicitly with `--davis_roots /path/to/DAVIS`. Then run depth preprocessing before anchor export.
