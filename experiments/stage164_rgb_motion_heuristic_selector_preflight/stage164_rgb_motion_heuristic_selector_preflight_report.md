# Stage164 RGB/Motion Heuristic Selector Preflight

## Scope

This stage evaluates row-level hard-segment selection using only Stage163 deployable RGB/motion features.
It does not instantiate full adaptive GOP schedules yet; Stage165 should convert selected hard windows into schedule metadata.

## Selected Heuristic

- Candidate: `edge_left_right`
- Score column: `edge_mad_left_right`
- Top fraction: `0.4`
- Threshold: `0.25868186354637146`
- Precision/recall/F1 for hard quality: `0.333333` / `0.533333` / `0.410256`
- Payload recall: `0.555556`

## Top Sweep Rows

| candidate | score | top frac | selected | precision hard | recall hard | F1 hard | recall payload | selected PSNR | unselected PSNR | selected payload | unselected payload |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_left_right | edge_mad_left_right | 0.40 | 48 | 0.333333 | 0.533333 | 0.410256 | 0.555556 | 28.144268 | 30.703128 | 233635.292 | 198710.403 |
| interp_target_mad | rgb_mad_linear_interp_target | 0.50 | 60 | 0.300000 | 0.600000 | 0.400000 | 0.750000 | 29.684011 | 29.675157 | 238973.050 | 186387.667 |
| proxy_score | rgb_motion_proxy_score | 0.50 | 60 | 0.300000 | 0.600000 | 0.400000 | 0.736111 | 29.411846 | 29.947322 | 237368.767 | 187991.950 |
| edge_left_right | edge_mad_left_right | 0.50 | 60 | 0.300000 | 0.600000 | 0.400000 | 0.680556 | 28.394502 | 30.964666 | 233064.250 | 192296.467 |
| proxy_score | rgb_motion_proxy_score | 0.40 | 48 | 0.312500 | 0.500000 | 0.384615 | 0.597222 | 29.379318 | 29.879761 | 238643.208 | 195371.792 |
| edge_left_right | edge_mad_left_right | 0.30 | 36 | 0.333333 | 0.400000 | 0.363636 | 0.416667 | 27.649871 | 30.549461 | 235522.194 | 202891.000 |
| left_right_mad | rgb_mad_left_right | 0.50 | 60 | 0.266667 | 0.533333 | 0.355556 | 0.750000 | 29.655357 | 29.703811 | 238654.300 | 186706.417 |
| combined_percentile | combined_percentile_score | 0.50 | 60 | 0.266667 | 0.533333 | 0.355556 | 0.750000 | 29.641634 | 29.717533 | 238240.417 | 187120.300 |
| interp_target_mad | rgb_mad_linear_interp_target | 0.40 | 48 | 0.270833 | 0.433333 | 0.333333 | 0.638889 | 29.744499 | 29.636307 | 241425.729 | 193516.778 |
| left_right_mad | rgb_mad_left_right | 0.40 | 48 | 0.270833 | 0.433333 | 0.333333 | 0.611111 | 29.762496 | 29.624309 | 239563.458 | 194758.292 |

## Sequence/Gap Summary For Selected Heuristic

| sequence | gap | tasks | selected | hard | selected hard | high payload | selected high payload | mean score | PSNR | LPIPS | payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bike-packing | 4 | 2 | 1 | 1 | 1 | 0 | 0 | 0.214344 | 25.916473 | 0.200759 | 173028.500 |
| bike-packing | 8 | 2 | 0 | 2 | 0 | 0 | 0 | 0.147255 | 25.335488 | 0.212680 | 169975.000 |
| blackswan | 4 | 2 | 0 | 0 | 0 | 2 | 0 | 0.220562 | 29.964644 | 0.160808 | 225588.500 |
| blackswan | 8 | 2 | 0 | 0 | 0 | 2 | 0 | 0.236848 | 30.052148 | 0.158135 | 231567.500 |
| bmx-trees | 4 | 2 | 0 | 0 | 0 | 2 | 0 | 0.232147 | 31.033645 | 0.109259 | 250143.000 |
| bmx-trees | 8 | 2 | 0 | 0 | 0 | 2 | 0 | 0.230362 | 30.849808 | 0.114973 | 252166.500 |
| breakdance | 4 | 2 | 0 | 2 | 0 | 0 | 0 | 0.153292 | 25.628598 | 0.170077 | 123729.000 |
| breakdance | 8 | 2 | 0 | 2 | 0 | 0 | 0 | 0.172151 | 25.335114 | 0.188348 | 149152.000 |
| camel | 4 | 2 | 2 | 2 | 2 | 2 | 2 | 0.285769 | 25.580679 | 0.187973 | 228002.500 |
| camel | 8 | 2 | 2 | 1 | 1 | 2 | 2 | 0.312674 | 25.769723 | 0.189888 | 237815.500 |
| car-roundabout | 4 | 2 | 2 | 1 | 1 | 2 | 2 | 0.345462 | 26.823942 | 0.150965 | 223383.000 |
| car-roundabout | 8 | 2 | 2 | 0 | 0 | 2 | 2 | 0.349264 | 27.513423 | 0.142667 | 221243.000 |
| car-shadow | 4 | 2 | 1 | 0 | 0 | 1 | 1 | 0.263012 | 29.522755 | 0.151694 | 214037.000 |
| car-shadow | 8 | 2 | 2 | 0 | 0 | 2 | 2 | 0.296248 | 29.683608 | 0.151380 | 232117.500 |
| cows | 4 | 2 | 2 | 2 | 2 | 1 | 1 | 0.288260 | 24.705746 | 0.206553 | 216943.000 |
| cows | 8 | 2 | 2 | 2 | 2 | 1 | 1 | 0.314699 | 24.680568 | 0.212888 | 223713.500 |
| dance-twirl | 4 | 2 | 1 | 2 | 1 | 0 | 0 | 0.256290 | 26.756008 | 0.249167 | 206271.500 |
| dance-twirl | 8 | 2 | 2 | 1 | 1 | 1 | 1 | 0.272149 | 28.731378 | 0.217736 | 218506.500 |
| dog | 4 | 2 | 0 | 0 | 0 | 2 | 0 | 0.196374 | 30.875154 | 0.162797 | 233641.000 |
| dog | 8 | 2 | 0 | 0 | 0 | 2 | 0 | 0.185346 | 33.475071 | 0.145589 | 241565.500 |
| dogs-jump | 8 | 2 | 0 | 1 | 0 | 0 | 0 | 0.112383 | 31.080910 | 0.229504 | 161858.000 |
| drift-straight | 4 | 2 | 2 | 0 | 0 | 2 | 2 | 0.376334 | 31.621132 | 0.128612 | 252306.000 |
| drift-straight | 8 | 2 | 2 | 0 | 0 | 2 | 2 | 0.455198 | 30.252893 | 0.113037 | 248552.500 |
| goat | 4 | 2 | 2 | 0 | 0 | 2 | 2 | 0.341433 | 27.287905 | 0.175273 | 251509.000 |
| goat | 8 | 2 | 2 | 0 | 0 | 2 | 2 | 0.336344 | 26.583443 | 0.179627 | 247414.500 |
| horsejump-high | 4 | 2 | 1 | 0 | 0 | 1 | 1 | 0.246401 | 27.807040 | 0.157850 | 210518.000 |
| horsejump-high | 8 | 2 | 2 | 0 | 0 | 1 | 1 | 0.288227 | 27.816431 | 0.183649 | 220016.500 |
| india | 4 | 2 | 0 | 0 | 0 | 1 | 0 | 0.212046 | 33.462700 | 0.119263 | 217111.500 |
| india | 8 | 2 | 0 | 2 | 0 | 2 | 0 | 0.222825 | 31.885708 | 0.241756 | 231995.000 |
| kite-surf | 4 | 2 | 0 | 0 | 0 | 1 | 0 | 0.167758 | 32.252243 | 0.108727 | 217840.500 |
| kite-surf | 8 | 2 | 0 | 0 | 0 | 2 | 0 | 0.159827 | 32.353322 | 0.097114 | 224779.000 |
| lab-coat | 4 | 2 | 0 | 0 | 0 | 1 | 0 | 0.232115 | 27.944376 | 0.190608 | 222231.000 |
| lab-coat | 8 | 2 | 2 | 0 | 0 | 2 | 2 | 0.292430 | 30.912601 | 0.149737 | 251296.000 |
| libby | 4 | 2 | 0 | 0 | 0 | 2 | 0 | 0.183565 | 32.898613 | 0.169676 | 257462.500 |
| libby | 8 | 2 | 0 | 0 | 0 | 2 | 0 | 0.193007 | 32.892570 | 0.170078 | 253948.500 |
| loading | 4 | 2 | 2 | 0 | 0 | 2 | 2 | 0.360028 | 28.846953 | 0.141979 | 237104.000 |
| loading | 8 | 2 | 2 | 0 | 0 | 2 | 2 | 0.372765 | 28.663316 | 0.148916 | 248738.500 |
| motocross-jump | 4 | 2 | 0 | 2 | 0 | 2 | 0 | 0.252610 | 31.391121 | 0.300439 | 247438.500 |
| motocross-jump | 8 | 2 | 0 | 2 | 0 | 2 | 0 | 0.185149 | 31.789746 | 0.282665 | 255678.500 |
| paragliding-launch | 8 | 2 | 0 | 0 | 0 | 1 | 0 | 0.138085 | 30.591214 | 0.190953 | 215446.000 |
| parkour | 4 | 2 | 0 | 0 | 0 | 2 | 0 | 0.234771 | 29.184944 | 0.188123 | 224234.000 |
| parkour | 8 | 2 | 2 | 2 | 2 | 1 | 1 | 0.300195 | 25.675776 | 0.213913 | 221249.000 |
| pigs | 8 | 2 | 0 | 0 | 0 | 1 | 0 | 0.176859 | 31.765246 | 0.183093 | 216138.000 |
| scooter-black | 4 | 2 | 2 | 0 | 0 | 2 | 2 | 0.486362 | 27.336615 | 0.138989 | 246170.000 |
| scooter-black | 8 | 2 | 2 | 2 | 2 | 1 | 1 | 0.384853 | 24.813841 | 0.204781 | 221069.500 |
| shooting | 4 | 2 | 1 | 0 | 0 | 2 | 1 | 0.234936 | 35.270957 | 0.138333 | 235373.000 |
| shooting | 8 | 2 | 2 | 1 | 1 | 2 | 2 | 0.275767 | 33.697522 | 0.205121 | 241836.500 |
| soapbox | 4 | 2 | 2 | 0 | 0 | 2 | 2 | 0.290134 | 30.906671 | 0.125260 | 241610.500 |
| soapbox | 8 | 2 | 1 | 0 | 0 | 1 | 1 | 0.255356 | 29.583052 | 0.195172 | 233226.500 |

## Interpretation

- A selected hard-segment row means a future adaptive scheduler should consider shortening the GOP or placing an extra keyframe near that window.
- False negatives remain important because Stage163 showed some low-PSNR cases are not captured by simple RGB motion alone.
- This heuristic uses no target dense anchors, rendered metrics, or labels at inference; labels are used only for this preflight evaluation.
