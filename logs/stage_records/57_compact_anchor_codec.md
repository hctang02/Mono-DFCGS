# Stage57 Compact Anchor Codec

Date: 2026-06-26

## Goal

Implement a true compact Gaussian anchor codec for q1-q16 instead of the Stage50 dtype-storage prototype where q6 used uint8 and q10/q12 used uint16.

## Code Changes

- `mono_dfcgs/anchor_bitstream.py`
  - Added bitpacked payload helpers for q1-q16.
  - `encode_anchor_bitstream(...)` now defaults to `payload_encoding="bitpack"`.
  - `payload_encoding="dtype"` preserves the legacy Stage50 storage behavior for comparisons.
  - `decode_anchor_bitstream(...)` supports both new bitpacked records and old dtype records.
  - q8 and q16 have fast paths because their bitpacked representation is byte-aligned.
- `scripts/run_stage50_multibit_anchor_bitstream_prototype.py`
  - Explicitly passes `payload_encoding="dtype"` so historical Stage50 semantics remain unchanged when rerun.
- `scripts/run_stage57_compact_anchor_codec.py`
  - New Stage57 verification and size table script.
  - Compares legacy dtype storage vs compact bitpacked storage.
  - Reports raw and zlib container size, payload size, savings, and roundtrip error.

## Verification Commands

GPU status was checked before running code via `nvidia-smi`. Stage57 itself used CPU anchor encode/decode only.

Syntax check:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m compileall mono_dfcgs/anchor_bitstream.py scripts/run_stage50_multibit_anchor_bitstream_prototype.py scripts/run_stage57_compact_anchor_codec.py
```

Smoke verification with all encodings decoded:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage57_compact_anchor_codec.py --samples n3dv --methods uniform --gaps 16 --bits 1 6 8 10 12 16 --limit_rows 1 --verify_encodings all
```

Formal Stage57 size table:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage57_compact_anchor_codec.py --methods uniform --gaps 16
```

## Outputs

```text
experiments/stage57_compact_anchor_codec/stage57_compact_anchor_codec.csv
experiments/stage57_compact_anchor_codec/stage57_compact_anchor_codec_summary.json
```

## Formal Run Scope

- Samples: `n3dv`, `meetroom`, `driving`, `robot`.
- Method: `uniform`.
- Gap: `16`.
- Bits: `1`, `2`, `4`, `6`, `8`, `10`, `12`, `16`.
- Rows: `32`.
- Roundtrip target: direct quantize-dequantize anchor attributes.
- Max roundtrip absolute difference: `0.0`.

Full default coverage and a 128-row representative run were both CPU-time limited in this environment. Stage57 therefore records a compact formal verification table and leaves broader RD integration to Stage58.

## Aggregate Results

| bits | legacy raw MiB/frame | compact raw MiB/frame | legacy zlib MiB/frame | compact zlib MiB/frame | payload saving vs legacy | compact zlib saving vs legacy zlib | max abs diff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.03433817985330386 | 0.004331399600475631 | 0.0010658563167712082 | 0.0009784991301193592 | 87.5 | 8.612936337775304 | 0.0 |
| 2 | 0.034339528483650295 | 0.008619560613566453 | 0.0024068143785582804 | 0.002325710203664051 | 75.0 | 3.4976338942978846 | 0.0 |
| 4 | 0.03434024117821319 | 0.017193754953662994 | 0.007907483943555721 | 0.007540968417433261 | 50.0 | 4.605790262408465 | 0.0 |
| 6 | 0.03434082092106562 | 0.025767816342049082 | 0.015097553958987625 | 0.016897197142040306 | 25.0 | -11.983035440949726 | 0.0 |
| 8 | 0.034340745041495396 | 0.03434119825468665 | 0.022427201366623337 | 0.02242765457981459 | 0.0 | -0.0020224984468543783 | 0.0 |
| 10 | 0.06863539991459824 | 0.04291540819118855 | 0.034567841089018385 | 0.029996156310240045 | 37.5 | 13.216750008373802 | 0.0 |
| 12 | 0.06863569265301074 | 0.05148915872180884 | 0.039116560122667454 | 0.035456150641172995 | 25.0 | 9.35167743481183 | 0.0 |
| 16 | 0.06863602862557636 | 0.06863641027879003 | 0.04598391474640788 | 0.045984296399621566 | 0.0 | -0.0008302180533818683 | 0.0 |

## Conclusions

- True bit-packing is implemented and lossless relative to direct quantize-dequantize anchors.
- q1/q2/q4/q10/q12 compact+zlib improves rate versus legacy dtype+zlib in this formal table.
- q6 compact raw saves the expected 25% payload, but compact+zlib is worse than legacy+zlib here because byte-aligned dtype payloads expose more zeros/redundancy to zlib.
- q8 and q16 have identical payload size to dtype storage; tiny negative savings come from the new v2 metadata/header.

## Next Step

Stage58 should integrate compact bitpacked rates into all-frame PSNR RD ablations and compare raw, zlib, bitpacked, bitpacked+zlib, and future per-field/delta options.
