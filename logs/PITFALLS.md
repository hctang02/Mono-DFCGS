# Pitfalls And Notes

## Storage

- `/mnt/ssd2tB` is almost full and should not receive copied StreamSplat checkpoints or output videos.
- `/mnt/hdd2tC/tmp/opencode` is acceptable for temporary experiments but not ideal for a long-term source repository.
- Long-term project source should live under `/mnt/hdd2tC/haocheng/KeyStreamSplat`.

## Git Hygiene

- Do not commit official StreamSplat checkpoints.
- Do not commit Python virtual environments.
- Do not commit compiled rasterizer build directories.
- Do not commit rendered videos or large image outputs.
- Keep NeoVerse/CompactWorld work separate from KeyStreamSplat work.

## StreamSplat Dependency State

The official StreamSplat checkout currently has untracked local runtime artifacts such as checkpoints, `__pycache__`, and compiled rasterizer build products. It should be treated as an external dependency rather than the main project repository.
