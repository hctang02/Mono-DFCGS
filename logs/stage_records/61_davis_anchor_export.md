# Stage61 DAVIS Anchor Export Preflight And Smoke

Date: 2026-06-27

## Goal

Prepare DAVIS large-scale Gaussian anchor export after Stage60 depth preprocessing, while avoiding accidental full output on a nearly full `/mnt/hdd2tC` mount.

## Code

```text
scripts/run_stage61_davis_anchor_export_preflight.py
scripts/run_stage61_davis_anchor_export.py
```

The preflight script estimates output size for DAVIS gaps without writing `.pt` anchors. The export script is full-capable but defaults to a safe smoke export: one sequence, one pair.

## Verification

GPU status was checked with `nvidia-smi` before running Python scripts.

Preflight commands:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage61_davis_anchor_export_preflight.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export_preflight.py
```

Smoke export commands:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage61_davis_anchor_export.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export.py --splits train --gaps 16 --max_sequences 1 --max_pairs_per_sequence 1 --batch_size 1 --device cuda:0
```

## Outputs

Preflight outputs:

```text
experiments/stage61_davis_anchor_export_preflight/stage61_davis_sequences.csv
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_plan.csv
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_totals.csv
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_preflight_summary.json
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_preflight_report.md
```

Smoke outputs:

```text
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.json
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.json
```

External heavy output, not tracked by git:

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export/DAVIS/train/bear/gap16/pair_000000_000016.pt
```

## Preflight Results

| Metric | Value |
|---|---:|
| DAVIS ready sequences | 90 / 90 |
| DAVIS frames | 6208 |
| free MiB at heavy root mount | 4034.95703125 |
| reserve MiB | 2048.0 |
| estimated all-gap pair-pt MiB | 21950.296875 |
| estimated all-gap dedup static-anchor MiB | 11386.4765625 |
| needed MiB including reserve | 23998.296875 |
| safe to full export | false |

Per-gap all-DAVIS estimates:

| gap | pairs | pair-pt MiB | dedup static-anchor MiB |
|---:|---:|---:|---:|
| 1 | 6118 | 11184.46875 | 5674.5 |
| 2 | 3089 | 5647.078125 | 2905.8046875 |
| 4 | 1568 | 2866.5 | 1515.515625 |
| 8 | 807 | 1475.296875 | 819.9140625 |
| 16 | 425 | 776.953125 | 470.7421875 |

## Smoke Export Results

| Metric | Value |
|---|---:|
| dataset | DAVIS |
| split/sequence | train/bear |
| gap | 16 |
| exported pairs | 1 |
| exported pair | 0 -> 16 |
| middle frames | 15 |
| anchor MiB | 1.828125 |
| gaussians per anchor | 36864 |

## Gap16 Partial Large-Scale Export

After the smoke run, the Stage61 preflight table showed that DAVIS `gap16` alone was feasible while all-gap export was not. The following command exported gap16 anchors for all DAVIS train/val sequences:

```text
CUDA_VISIBLE_DEVICES=4 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export.py --splits train val --gaps 16 --max_sequences 0 --max_pairs_per_sequence 0 --batch_size 2 --device cuda:0
```

Results:

| Metric | Value |
|---|---:|
| splits | train, val |
| sequences | 90 |
| gap | 16 |
| exported pairs | 425 |
| total anchor MiB | 776.953125 |
| external heavy root size | about 781M |
| tracked summary size | about 360K |
| free space after export | about 3.3G |

Updated tracked files:

```text
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.json
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.json
```

External `.pt` files are under:

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export/DAVIS/{train,val}/<sequence>/gap16/*.pt
```

## Conclusions

- DAVIS depth preprocessing is sufficient for StreamSplat Gaussian anchor export.
- Full DAVIS all-gap export is blocked by current free space, not by data layout or code wiring.
- The DAVIS export code path is verified by smoke and by the full DAVIS train+val gap16 partial export.
- Further gap exports should wait until storage is freed or a larger external mount is available.

## Next Step

Use the DAVIS gap16 anchor manifest for limited Stage62 adapter-infra testing. Free additional storage before gap1/2/4/8 exports or YouTube-VOS preparation.
