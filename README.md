# Mono-DFCGS

Mono-DFCGS is a research codebase for a keyframe-Gaussian-only dynamic 3D Gaussian video codec built around a StreamSplat-style decoder.

The intended codec setting is:

```text
full monocular video
-> selected keyframes
-> transmitted compressed keyframe Gaussian anchors
-> decoder-side dynamic Gaussian prediction
-> reconstructed full video / dynamic 3DGS
```

This repository should contain only our own codec code, experiment scripts, documentation, and lightweight metadata. Large external dependencies are intentionally kept outside git.

## Local External Paths

Current external resources are stored outside this repository:

```text
Official StreamSplat repo: /mnt/hdd2tC/tmp/opencode/StreamSplat
StreamSplat repro scripts: /mnt/hdd2tC/tmp/opencode/streamsplat_repro
StreamSplat venv: /mnt/hdd2tC/tmp/opencode/streamsplat_venv
Depth/model checkpoints: /mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints
```

These paths are local machine state, not git-tracked project content.

## Git Policy

- Track source code, scripts, configs, and Markdown logs.
- Do not track checkpoints, venvs, compiled rasterizers, cache directories, datasets, renders, or output videos.
- Keep this project separate from the NeoVerse/CompactWorld repository because the research target and rate accounting are different.
