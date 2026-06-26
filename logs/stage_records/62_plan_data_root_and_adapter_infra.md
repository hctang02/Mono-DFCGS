# Stage62 Plan: /data DAVIS Root And Adapter Infra

Date: 2026-06-27

## Goal

Continue from Stage61 using the newly downloaded official DAVIS Full-Resolution root on `/data`, then proceed to Stage62 adapter training infrastructure v2.

## New Data Root

```text
/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS
```

Zip archive root:

```text
/data/hctang/tmp/opencode/datasets/DAVIS_official_zips
```

## Immediate Plan

1. Inspect `/data` DAVIS root and confirm split/frame/depth readiness.
2. Reuse or generate train/val depth under the `/data` DAVIS root.
3. Re-run Stage61 preflight with `/data` heavy root.
4. Export DAVIS train/val multi-gap anchors to `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full`.
5. Implement Stage62 adapter training infra v2: manifest merging, train/val split, resume, best checkpoint, log CSV, storage-safe external checkpoint root.
6. Run a small Stage62 smoke on DAVIS anchors.

## Constraints

- YouTube-VOS remains paused per user request.
- Large outputs stay outside git under `/data/hctang/tmp/opencode/...`.
- Repository should only track scripts, manifests/summaries, and logs.
- Continue default reporting with all-frame PSNR for evaluation stages; Stage62 infra may only produce training/validation losses until rendering evaluation is added.
