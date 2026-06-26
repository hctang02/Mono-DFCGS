# Stage58 Compression RD Ablation

Date: 2026-06-26

## Goal

Integrate the compact anchor codec into RD reporting without rerendering, using existing Stage51 all-frame PSNR rows and Stage57 actual compact rate measurements.

## Inputs

```text
experiments/stage51_high_rate_multibit_rd/stage51_high_rate_multibit_rd.csv
experiments/stage57_compact_anchor_codec/stage57_compact_anchor_codec.csv
```

Stage51 provides all-frame PSNR for q8/q10/q12/q16 over local samples, methods, and gaps. Stage57 provides actual compact bitpacked container rates for the formal subset: 4 samples, uniform, gap16, q1/q2/q4/q6/q8/q10/q12/q16.

## Code

```text
scripts/run_stage58_compression_rd_ablation.py
```

The script builds these codec variants:

| Variant | Scope |
|---|---|
| `legacy_dtype_raw` | actual Stage50 dtype raw container including metadata |
| `legacy_dtype_zlib` | actual Stage50 dtype zlib container including metadata |
| `compact_bitpack_raw_payload_estimate` | theoretical bitpacked payload only, no metadata/header |
| `stage57_compact_raw_actual` | actual Stage57 bitpacked raw container including metadata, Stage57 subset only |
| `stage57_compact_zlib_actual` | actual Stage57 bitpacked zlib container including metadata, Stage57 subset only |

## Verification

GPU status was checked before running code with `nvidia-smi`. Stage58 only reads CSV files and plots with matplotlib; it does not use CUDA or rerender frames.

Commands:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m compileall scripts/run_stage58_compression_rd_ablation.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage58_compression_rd_ablation.py
```

## Outputs

```text
experiments/stage58_compression_rd_ablation/stage58_compression_rd_ablation.csv
experiments/stage58_compression_rd_ablation/stage58_mean_compression_rd.csv
experiments/stage58_compression_rd_ablation/stage58_codec_summary.csv
experiments/stage58_compression_rd_ablation/stage58_actual_compact_savings.csv
experiments/stage58_compression_rd_ablation/stage58_actual_compact_savings_by_bits.csv
experiments/stage58_compression_rd_ablation/stage58_compression_rd_ablation_summary.json
experiments/stage58_compression_rd_ablation/stage58_full_mean_compression_rd.png
experiments/stage58_compression_rd_ablation/stage58_actual_compact_subset_rd.png
```

## Run Summary

- Stage51 input rows: `192`.
- Stage57 compact input rows: `32`.
- RD rows: `608`.
- Mean RD rows: `152`.
- All quality values are `adapter_all_psnr` from Stage51, reported as all-frame PSNR.

## Codec Summary

| codec variant | rows | mean rate MiB/frame | mean all PSNR | scope |
|---|---:|---:|---:|---|
| `compact_bitpack_raw_payload_estimate` | 192 | 0.2545978056231963 | 30.073573039629064 | payload-only estimate |
| `legacy_dtype_raw` | 192 | 0.3101746928587697 | 30.073573039629064 | actual Stage50 dtype raw container |
| `legacy_dtype_zlib` | 192 | 0.18328456795742495 | 30.073573039629064 | actual Stage50 dtype zlib container |
| `stage57_compact_raw_actual` | 16 | 0.04934554386161852 | 25.744420008373275 | actual compact subset |
| `stage57_compact_zlib_actual` | 16 | 0.033466064482712304 | 25.744420008373275 | actual compact subset |

The compact actual subset has lower PSNR because it only covers uniform gap16 points, not because the codec changes reconstruction quality.

## Actual Compact Savings By Bits

| baseline | compact | bits | mean rate saving % | baseline MiB/frame | compact MiB/frame | all PSNR |
|---|---|---:|---:|---:|---:|---:|
| legacy raw | compact raw | 8 | -0.0078843090532823 | 0.03433849090220207 | 0.034341198254686636 | 25.08719681572384 |
| legacy raw | compact raw | 10 | 37.471310438888885 | 0.06863314577530491 | 0.04291540819118855 | 25.88434351291017 |
| legacy raw | compact raw | 12 | 24.979487140749605 | 0.06863343851371742 | 0.05148915872180884 | 25.99829100228245 |
| legacy raw | compact raw | 16 | -0.0038403725683813852 | 0.06863377448628302 | 0.06863641027879004 | 26.007848702576645 |
| legacy zlib | compact zlib | 8 | -0.012082983041822714 | 0.022424947227330017 | 0.02242765457981459 | 25.08719681572384 |
| legacy zlib | compact zlib | 10 | 13.21108787307025 | 0.03456558694972506 | 0.029996156310240045 | 25.88434351291017 |
| legacy zlib | compact zlib | 12 | 9.3464515290433 | 0.03911430598337413 | 0.035456150641172995 | 25.99829100228245 |
| legacy zlib | compact zlib | 16 | -0.005733974631345334 | 0.045981660607114554 | 0.045984296399621566 | 26.007848702576645 |

## Conclusions

- Stage58 provides the first all-frame PSNR RD package that separates compression variants.
- Actual compact q10/q12 bitpacked+zlib reduces rate versus legacy dtype+zlib on the Stage57 subset, with unchanged all-frame PSNR.
- q8/q16 are byte-aligned and show only tiny metadata overhead differences.
- Full compact+zlib coverage is still limited to the Stage57 formal subset; broader actual compact rate generation should be added if Stage58 becomes the final compression ablation table.

## Next Step

Stage59 should run the DAVIS/YouTube-VOS download and preparation preflight from the updated Stage56-70 plan.
