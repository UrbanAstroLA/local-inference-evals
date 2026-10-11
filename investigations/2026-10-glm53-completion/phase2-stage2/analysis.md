## Rows captured

Capture files analysed: 440 (v091 replay records 440, v070 440).

| capture route | rows | 4k | 8k | 16k | 32k | 64k | 120k | DSA layers |
|---|---|---|---|---|---|---|---|---|
| fused | 1400 | 120 | 296 | 428 | 68 | 408 | 80 | [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43] |
| tiled | 9944 | 1320 | 1320 | 2508 | 1144 | 2640 | 1012 | [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43] |

## Main table: path x capture route (all lengths, all layers)

Rows x repeats. bad480 = column-511 pool has fp64 reference rank < 480 and is not an exact boundary tie; (tol) = not within the fp32 evaluation bound of the 512th score either.

| path | capture route | n | rank<480 non-tie | rank<480 (tol) | rank<511 non-tie | rank<256 | min | p10 | median | set mismatch exact / tol |
|---|---|---|---|---|---|---|---|---|---|---|
| in-situ 0.9.1 (live) | tiled | 9944 | 88.1% | 88.1% | 97.4% | 45.3% | 0 | 58 | 280 | 606 / 606 |
| in-situ 0.9.1 (live) | fused | 1400 | 0.5% | 0.5% | 3.4% | 0.0% | 380 | 511 | 511 | 16 / 16 |
| v091_auto | tiled | 29832 | 88.1% | 88.1% | 97.5% | 46.3% | 0 | 61 | 275 | 1808 / 1808 |
| v091_auto | fused | 4200 | 0.4% | 0.4% | 3.6% | 0.0% | 388 | 511 | 511 | 48 / 48 |
| v091_tiled_forced | fused | 4200 | 92.2% | 92.2% | 97.9% | 47.9% | 0 | 58 | 264 | 48 / 48 |
| v091_fused_split16 | tiled | 29832 | 0.4% | 0.4% | 8.9% | 0.0% | 258 | 511 | 511 | 1815 / 1815 |
| v070_dcp2_merge | tiled | 29832 | 6.9% | 6.9% | 62.6% | 3.0% | 0 | 496 | 510 | 1824 / 1824 |
| v070_dcp2_merge | fused | 4200 | 5.6% | 5.6% | 60.8% | 2.5% | 0 | 499 | 510 | 48 / 48 |

## Sensitivity: rows without NaN-byte pools only

| path | capture route | n | rank<480 non-tie | median rank | set mismatch exact / tol |
|---|---|---|---|---|---|
| in-situ 0.9.1 (live) | tiled | 8904 | 91.5% | 268 | 0 / 0 |
| in-situ 0.9.1 (live) | fused | 1360 | 0.5% | 511 | 0 / 0 |
| v091_auto | tiled | 26712 | 91.4% | 265 | 0 / 0 |
| v091_auto | fused | 4080 | 0.4% | 511 | 0 / 0 |
| v091_tiled_forced | fused | 4080 | 92.1% | 264 | 0 / 0 |
| v091_fused_split16 | tiled | 26712 | 0.5% | 511 | 0 / 0 |
| v070_dcp2_merge | tiled | 26712 | 6.8% | 510 | 0 / 0 |
| v070_dcp2_merge | fused | 4080 | 5.6% | 510 | 0 / 0 |

## Path x capture route x length bin (pool count)

Cell = rank<480 non-tie share (n rows x repeats); min / p10 / median rank below.

| path | route | 4k | 8k | 16k | 32k | 64k | 120k |
|---|---|---|---|---|---|---|---|
| in-situ 0.9.1 (live) | tiled | 96.7% (n=1320; 1/46/244) | 96.9% (n=1320; 0/59/268) | 96.1% (n=2508; 0/56/262) | 96.2% (n=1144; 0/55/257) | 97.0% (n=2640; 0/53/251) | 13.4% (n=1012; 0/365/507) |
| in-situ 0.9.1 (live) | fused | 0.8% (n=120; 456/511/511) | 1.0% (n=296; 420/511/511) | 0.0% (n=428; 493/511/511) | 0.0% (n=68; 482/511/511) | 0.5% (n=408; 380/511/511) | 1.2% (n=80; 465/505/511) |
| v091_auto | tiled | 96.2% (n=3960; 1/56/245) | 96.4% (n=3960; 0/62/268) | 96.2% (n=7524; 0/55/250) | 96.1% (n=3432; 0/58/251) | 97.3% (n=7920; 0/52/248) | 13.1% (n=3036; 0/373/507) |
| v091_auto | fused | 0.6% (n=360; 454/511/511) | 0.8% (n=888; 422/511/511) | 0.1% (n=1284; 474/511/511) | 1.0% (n=204; 454/511/511) | 0.3% (n=1224; 388/511/511) | 0.4% (n=240; 407/509/511) |
| v091_tiled_forced | fused | 97.2% (n=360; 10/47/250) | 96.6% (n=888; 0/47/259) | 96.4% (n=1284; 0/60/249) | 98.5% (n=204; 0/43/273) | 97.1% (n=1224; 0/60/254) | 15.0% (n=240; 1/387/510) |
| v091_fused_split16 | tiled | 0.2% (n=3960; 417/511/511) | 0.6% (n=3960; 421/511/511) | 0.6% (n=7524; 391/511/511) | 0.6% (n=3432; 360/511/511) | 0.4% (n=7920; 258/508/511) | 0.2% (n=3036; 456/506/511) |
| v070_dcp2_merge | tiled | 7.5% (n=3960; 0/498/510) | 5.6% (n=3960; 0/499/510) | 7.9% (n=7524; 0/496/510) | 7.1% (n=3432; 2/496/510) | 5.5% (n=7920; 0/498/509) | 8.9% (n=3036; 3/480/508) |
| v070_dcp2_merge | fused | 3.3% (n=360; 29/504/510) | 6.4% (n=888; 0/497/510) | 4.9% (n=1284; 8/501/510) | 5.9% (n=204; 32/497/509) | 4.9% (n=1224; 7/501/510) | 13.3% (n=240; 15/414/509) |

## Path x capture route x DSA layer

| path | route | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 | L35 | L39 | L43 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in-situ 0.9.1 (live) | tiled | 93.3% (904) | 88.6% (904) | 88.1% (904) | 87.4% (904) | 86.2% (904) | 87.5% (904) | 87.9% (904) | 87.5% (904) | 88.1% (904) | 87.6% (904) | 87.5% (904) |
| in-situ 0.9.1 (live) | fused | 0.8% (128) | 1.5% (136) | 0.0% (116) | 0.0% (132) | 0.0% (136) | 0.0% (128) | 0.0% (124) | 0.0% (116) | 1.6% (124) | 0.8% (132) | 0.8% (128) |
| v091_auto | tiled | 92.9% (2712) | 88.7% (2712) | 87.6% (2712) | 87.5% (2712) | 85.7% (2712) | 86.6% (2712) | 88.8% (2712) | 87.7% (2712) | 87.8% (2712) | 89.0% (2712) | 86.4% (2712) |
| v091_auto | fused | 0.8% (384) | 1.5% (408) | 0.0% (348) | 0.0% (396) | 0.5% (408) | 0.0% (384) | 0.3% (372) | 0.0% (348) | 0.8% (372) | 0.3% (396) | 0.3% (384) |
| v091_tiled_forced | fused | 96.9% (384) | 94.4% (408) | 85.9% (348) | 88.6% (396) | 92.9% (408) | 89.8% (384) | 91.9% (372) | 91.7% (348) | 90.6% (372) | 96.2% (396) | 94.3% (384) |
| v091_fused_split16 | tiled | 0.5% (2712) | 0.5% (2712) | 0.5% (2712) | 0.3% (2712) | 0.7% (2712) | 0.4% (2712) | 0.3% (2712) | 0.4% (2712) | 0.3% (2712) | 0.6% (2712) | 0.4% (2712) |
| v070_dcp2_merge | tiled | 5.9% (2712) | 3.2% (2712) | 7.4% (2712) | 6.0% (2712) | 10.7% (2712) | 11.2% (2712) | 11.9% (2712) | 5.3% (2712) | 2.6% (2712) | 3.2% (2712) | 8.6% (2712) |
| v070_dcp2_merge | fused | 5.5% (384) | 2.9% (408) | 2.9% (348) | 4.8% (396) | 8.6% (408) | 8.6% (384) | 9.9% (372) | 4.0% (348) | 2.7% (372) | 3.0% (396) | 8.6% (384) |

## Run-to-run variation (replays, 3 repeats each)

| path | capture route | rows whose col-511 pool differed across repeats | rank<480 non-tie share per repeat |
|---|---|---|---|
| v091_auto | tiled | 9576/9944 (96.3%) | 88.2%, 87.9%, 88.1% |
| v091_auto | fused | 39/1400 (2.8%) | 0.4%, 0.4%, 0.4% |
| v091_tiled_forced | fused | 1377/1400 (98.4%) | 91.8%, 92.6%, 92.1% |
| v091_fused_split16 | tiled | 390/9944 (3.9%) | 0.5%, 0.4%, 0.5% |
| v070_dcp2_merge | tiled | 6708/9944 (67.5%) | 6.9%, 6.9%, 6.9% |
| v070_dcp2_merge | fused | 946/1400 (67.6%) | 5.9%, 5.3%, 5.7% |

## In-situ (live) vs 0.9.1 replay of the same bytes

| replay path | capture route | col-511 pool same as live | full 512 order same | selected set same | n |
|---|---|---|---|---|---|
| v091_auto | fused | 97.6% | 0.0% | 99.6% | 4200 |
| v091_auto | tiled | 6.9% | 0.0% | 99.7% | 29832 |
| v091_fused_split16 | tiled | 3.6% | 0.0% | 99.4% | 29832 |
| v091_tiled_forced | fused | 1.9% | 0.0% | 99.6% | 4200 |

## Score distribution near the boundary (fp64 reference, per captured row)

Rows: 11344. Rows containing pools whose stored K bytes decode to NaN: 1080. Rows with >1 pool exactly equal to the 512th score: 63; with >1 pool within the fp32 bound of it: 6694; any exactly-equal scores among the top 520: 6230; any exact-zero score among all pools: 11304; zeros inside the top 512: 6881. Relative gap |s512 - s513| / |s512|: median 0.000884, p10 0.000106.

## Probe index (every armed decode-branch call)

| phase tag | mixed prefill+decode step | route | captured | calls |
|---|---|---|---|---|
| fused-F1 | False | paged_fused | False | 121 |
| fused-F1 | False | paged_fused | True | 66 |
| fused-F1 | True | paged_fused | False | 99 |
| fused-F2 | False | paged_fused | False | 127 |
| fused-F2 | False | paged_fused | True | 55 |
| fused-F2 | True | paged_fused | False | 121 |
| tiled-T1 | False | paged_fused | False | 88 |
| tiled-T1 | False | paged_fused | True | 44 |
| tiled-T1 | False | paged_tiled | False | 22 |
| tiled-T1 | False | paged_tiled | True | 11 |
| tiled-T1 | True | paged_fused | False | 165 |
| tiled-T1 | True | paged_tiled | False | 99 |
| tiled-T3 | False | paged_fused | False | 110 |
| tiled-T3 | False | paged_fused | True | 44 |
| tiled-T3 | False | paged_tiled | False | 495 |
| tiled-T3 | False | paged_tiled | True | 165 |
| tiled-T3 | True | paged_fused | False | 154 |
| tiled-T3 | True | paged_tiled | False | 110 |
| tiled-T4 | False | paged_fused | False | 99 |
| tiled-T4 | False | paged_fused | True | 44 |
| tiled-T4 | False | paged_tiled | False | 484 |
| tiled-T4 | False | paged_tiled | True | 154 |
| tiled-T4 | True | paged_fused | False | 154 |
| tiled-T4 | True | paged_tiled | False | 110 |

rows / requests / next_n / route: 4/1/1/paged_fused: 303, 8/2/1/paged_fused: 407, 12/3/1/paged_fused: 440, 16/4/1/paged_fused: 341, 20/5/1/paged_tiled: 209, 24/6/1/paged_tiled: 275, 28/7/1/paged_tiled: 220, 32/8/1/paged_tiled: 946

