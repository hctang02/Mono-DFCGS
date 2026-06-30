# Stage165 Multi-Feature Keyframe Schedule Preflight

## Scope

This is a metadata/label preflight. It converts row-level RGB/motion hard-window signals into adaptive keyframe schedules but does not run heavy rendered RD yet.

## Selected Gate

- Rank threshold: `0.65`
- Minimum feature votes: `1`
- Selected rows: `70` / `120`
- Hard-quality precision/recall/F1: `0.314286` / `0.733333` / `0.440000`
- Payload precision/recall: `0.842857` / `0.819444`

## Schedule Summary

- Schedule: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1`
- Sequences: `30`
- Total frames: `1999`
- Total keyframes: `358`
- Mean keyframe ratio: `0.179090`
- Metadata: `2610` bits / `327` bytes / `0.000311136` MiB
- Selected hard-quality rows: `22` / `30`
- Selected high-payload rows: `59` / `72`

## Top Gate Sweep Rows

| candidate | threshold | votes | selected | precision hard | recall hard | F1 hard | recall payload | selected payload | unselected payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_gate_t0.65_votes1 | 0.65 | 1 | 70 | 0.314286 | 0.733333 | 0.440000 | 0.819444 | 232024.857 | 185598.060 |
| rank_gate_t0.6_votes1 | 0.60 | 1 | 75 | 0.306667 | 0.766667 | 0.438095 | 0.875000 | 231288.707 | 181666.444 |
| rank_gate_t0.55_votes1 | 0.55 | 1 | 80 | 0.287500 | 0.766667 | 0.418182 | 0.888889 | 228326.438 | 181388.200 |
| rank_gate_t0.7_votes1 | 0.70 | 1 | 62 | 0.306452 | 0.633333 | 0.413043 | 0.750000 | 233931.081 | 189964.069 |
| rank_gate_t0.5_votes1 | 0.50 | 1 | 85 | 0.270588 | 0.766667 | 0.400000 | 0.944444 | 228001.706 | 175471.371 |
| rank_gate_t0.5_votes2 | 0.50 | 2 | 69 | 0.275362 | 0.633333 | 0.383838 | 0.847222 | 237090.246 | 179655.216 |
| rank_gate_t0.5_votes3 | 0.50 | 3 | 59 | 0.288136 | 0.566667 | 0.382022 | 0.722222 | 237577.898 | 188599.131 |
| rank_gate_t0.5_votes4 | 0.50 | 4 | 54 | 0.296296 | 0.533333 | 0.380952 | 0.666667 | 238266.167 | 191746.515 |
| rank_gate_t0.55_votes4 | 0.55 | 4 | 49 | 0.306122 | 0.500000 | 0.379747 | 0.625000 | 238830.735 | 194632.915 |
| rank_gate_t0.55_votes3 | 0.55 | 3 | 56 | 0.285714 | 0.533333 | 0.372093 | 0.694444 | 238475.143 | 190109.922 |

## Schedule Rows

| sequence | frames | gap8 keys | gap4 keys | adaptive keys | extra vs gap8 | metadata bytes | selected/hard | selected/high payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bike-packing | 69 | 10 | 18 | 11 | 1 | 11 | 1/3 | 0/0 |
| blackswan | 50 | 8 | 14 | 8 | 0 | 7 | 0/0 | 0/4 |
| bmx-trees | 80 | 11 | 21 | 13 | 2 | 13 | 0/0 | 4/4 |
| breakdance | 84 | 12 | 22 | 12 | 0 | 12 | 0/4 | 0/0 |
| camel | 90 | 13 | 24 | 17 | 4 | 16 | 3/3 | 4/4 |
| car-roundabout | 75 | 11 | 20 | 15 | 4 | 15 | 1/1 | 4/4 |
| car-shadow | 40 | 6 | 11 | 9 | 3 | 8 | 0/0 | 3/3 |
| cows | 104 | 14 | 27 | 18 | 4 | 17 | 4/4 | 2/2 |
| dance-twirl | 90 | 13 | 24 | 16 | 3 | 15 | 2/3 | 1/1 |
| dog | 60 | 9 | 16 | 11 | 2 | 10 | 0/0 | 2/4 |
| dogs-jump | 66 | 10 | 18 | 11 | 1 | 11 | 0/1 | 0/0 |
| drift-chicane | 52 | 8 | 14 | 10 | 2 | 9 | 0/0 | 0/0 |
| drift-straight | 50 | 8 | 14 | 12 | 4 | 10 | 0/0 | 4/4 |
| goat | 90 | 13 | 24 | 17 | 4 | 16 | 0/0 | 4/4 |
| gold-fish | 78 | 11 | 21 | 11 | 0 | 11 | 0/0 | 0/0 |
| horsejump-high | 50 | 8 | 14 | 10 | 2 | 9 | 0/0 | 1/2 |
| india | 81 | 11 | 21 | 14 | 3 | 14 | 2/2 | 3/3 |
| judo | 34 | 6 | 10 | 6 | 0 | 6 | 0/0 | 0/0 |
| kite-surf | 50 | 8 | 14 | 10 | 2 | 9 | 0/0 | 2/3 |
| lab-coat | 47 | 7 | 13 | 9 | 2 | 8 | 0/0 | 2/3 |
| libby | 49 | 7 | 13 | 10 | 3 | 9 | 0/0 | 3/4 |
| loading | 50 | 8 | 14 | 11 | 3 | 10 | 0/0 | 4/4 |
| mbike-trick | 79 | 11 | 21 | 11 | 0 | 11 | 0/0 | 0/0 |
| motocross-jump | 40 | 6 | 11 | 9 | 3 | 8 | 4/4 | 4/4 |
| paragliding-launch | 80 | 11 | 21 | 11 | 0 | 11 | 0/0 | 0/1 |
| parkour | 100 | 14 | 26 | 17 | 3 | 16 | 2/2 | 2/3 |
| pigs | 79 | 11 | 21 | 11 | 0 | 11 | 0/0 | 0/1 |
| scooter-black | 43 | 7 | 12 | 11 | 4 | 10 | 2/2 | 3/3 |
| shooting | 40 | 6 | 11 | 10 | 4 | 9 | 1/1 | 4/4 |
| soapbox | 99 | 14 | 26 | 17 | 3 | 16 | 0/0 | 3/3 |

## Interpretation

- The candidate starts from uniform gap8 and adds selected target frames as extra keyframes.
- Keyframe indices are transmitted and counted as metadata; decoder does not reproduce RGB/motion feature extraction.
- This remains a pre-render schedule candidate. Stage166 should evaluate rendered/label RD for the proposed schedules and compare against uniform gap4/gap8.
