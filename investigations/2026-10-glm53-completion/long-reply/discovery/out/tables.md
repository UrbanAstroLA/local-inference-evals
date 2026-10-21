### T1. Per-window metrics by set (median over replies of each reply's median over full 4k windows)

| set | replies | cpt | rep20 | rep50 | rep100 | zr | dlr | line_seen | cjk | digit | punct | space | tok_le1char | chk_verify_per1k | chk_verify_rep | chk_wait_per1k | chk_wait_rep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comparison 50-100k (finished) | 24 | 3.092 | 0.016 | 0.000 | 0.000 | 0.368 | 1.000 | 0.000 | 0.000 | 0.034 | 0.086 | 0.165 | 0.288 | 1.099 | 0.000 | 3.113 | 0.367 |
| finished >=100k: all windows | 19 | 3.152 | 0.012 | 0.000 | 0.000 | 0.356 | 1.000 | 0.000 | 0.000 | 0.033 | 0.086 | 0.173 | 0.276 | 0.977 | 0.000 | 5.371 | 0.500 |
| finished >=100k: windows before 100k | 19 | 3.128 | 0.012 | 0.000 | 0.000 | 0.359 | 1.000 | 0.000 | 0.000 | 0.034 | 0.083 | 0.168 | 0.281 | 0.977 | 0.000 | 5.615 | 0.481 |
| capped: windows before 100k | 10 | 2.291 | 0.015 | 0.000 | 0.000 | 0.386 | 1.000 | 0.000 | 0.000 | 0.073 | 0.137 | 0.202 | 0.407 | 0.732 | 0.000 | 8.545 | 0.360 |
| capped: pre-onset windows | 10 | 2.312 | 0.047 | 0.001 | 0.000 | 0.357 | 1.000 | 0.000 | 0.000 | 0.069 | 0.138 | 0.179 | 0.399 | 0.732 | 0.000 | 9.765 | 0.446 |
| capped: post-onset windows | 8 | 2.500 | 1.000 | 1.000 | 1.000 | 0.008 | 1.000 | 0.091 | 0.000 | 0.000 | 0.175 | 0.200 | 0.443 | 0.000 | 0.250 | 277.466 | 1.000 |

### T2. Characters per token: whole reply, before onset and after onset (token-weighted)

| id | T | outcome | cpt whole | cpt before onset (or whole) | cpt after onset | cpt first 100k | digit share first 100k |
|---|---:|---|---:|---:|---:|---:|---:|
| q79p1 | 327680 | capped | 2.11 | 2.02 | 2.76 | 2.15 | 0.174 |
| q88p1 | 327680 | capped | 2.63 | 2.79 | 2.50 | 2.76 | 0.058 |
| q8p2 | 327680 | capped | 2.74 | 2.83 | 2.50 | 2.83 | 0.076 |
| q69p2 | 327680 | capped | 2.58 | 2.36 | 2.84 | 2.31 | 0.066 |
| q88p2 | 327680 | capped | 2.60 | 2.84 | 2.55 | 2.71 | 0.049 |
| q79p2 | 327680 | capped | 2.05 | 2.05 | – | 2.26 | 0.165 |
| q79p3 | 327680 | capped | 2.39 | 2.39 | – | 2.31 | 0.150 |
| q127p3 | 327680 | capped | 2.29 | 2.28 | 2.29 | 2.28 | 0.000 |
| q120p3 | 327680 | capped | 2.57 | 2.63 | 2.50 | 2.64 | 0.087 |
| q158p3 | 327680 | capped | 2.01 | 1.91 | 2.50 | 1.70 | 0.430 |
| q127p2 | 206688 | finished | 2.17 | 2.17 | – | 2.13 | 0.079 |
| q115p3 | 197631 | finished | 3.06 | 3.06 | – | 3.05 | 0.041 |
| q88p3 | 173347 | finished | 3.24 | 3.24 | – | 3.16 | 0.044 |
| q94p2 | 163343 | finished | 3.10 | 3.11 | 2.93 | 3.04 | 0.033 |
| q48p3 | 157835 | finished | 3.86 | 3.86 | – | 3.95 | 0.022 |
| q147p1 | 150616 | finished | 2.65 | 2.65 | – | 2.60 | 0.100 |
| q165p2 | 150052 | finished | 2.95 | 2.96 | 2.86 | 2.98 | 0.029 |
| q13p3 | 149478 | finished | 4.23 | 4.23 | – | 4.24 | 0.003 |
| q71p3 | 148563 | finished | 3.33 | 3.33 | – | 3.31 | 0.031 |
| q69p1 | 144012 | finished | 3.09 | 3.09 | – | 3.20 | 0.032 |
| q145p2 | 142105 | finished | 3.35 | 3.35 | – | 3.27 | 0.029 |
| q71p2 | 134402 | finished | 3.29 | 3.29 | – | 3.24 | 0.030 |
| q81p2 | 126276 | finished | 3.31 | 3.31 | – | 3.22 | 0.037 |
| q81p3 | 124863 | finished | 2.65 | 2.65 | – | 2.60 | 0.054 |
| q71p1 | 124709 | finished | 3.19 | 3.19 | – | 3.15 | 0.036 |
| q186p1 | 124517 | finished | 2.87 | 2.87 | – | 2.84 | 0.090 |
| q69p3 | 108084 | finished | 2.74 | 2.74 | – | 2.74 | 0.051 |
| q147p3 | 103977 | finished | 2.62 | 2.62 | – | 2.61 | 0.082 |
| q7p3 | 100217 | finished | 3.66 | 3.66 | – | 3.66 | 0.033 |

### T3. Capped replies (and finished replies with a terminal onset): onset and loop type

| id | T | outcome | onset g (R50>=0.5) | onset g @0.3 / @0.7 | onset abs | R50 at 100k | prespec class | E | LRS after onset (tokens) | LRS start g | LRS offset = period | exact lag match m(period) | post hoc class | episodes (recovered) | R50 final / max |
|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|
| q165p2 | 150052 | stop | 138657 | 137270 / 140093 | 139544 | 0.018 | short-period exact loop | 0.71 | 7140 | 142085 | 14 | – | exact loop, period 14 (short) | 1 (0) | 0.99 / 1.00 |
| q94p2 | 163343 | stop | 149160 | 147309 / 151529 | 149399 | 0.000 | fragmentary recycling | 0.00 | 1439 | 153594 | 1814 | 0.173 | partial / fragmentary (LRS 1439 of 13625) | 1 (0) | 0.83 / 0.85 |
| q79p1 | 327680 | length | 288394 | 287569 / 289314 | 288485 | 0.011 | fragmentary recycling | 0.00 | 35700 | 291730 | 246 | 0.930 | exact loop, period 246 (short) | 1 (0) | 1.00 / 1.00 |
| q88p1 | 327680 | length | 143158 | 142275 / 144001 | 143387 | 0.000 | short-period exact loop | 0.99 | 182166 | 145512 | 2 | 0.991 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| q8p2 | 327680 | length | 236765 | 235946 / 237656 | 237003 | 0.000 | short-period exact loop | 0.98 | 88653 | 239024 | 2 | – | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| q69p2 | 327680 | length | 181125 | 177951 / 186773 | 181273 | 0.002 | fragmentary recycling | 0.00 | 116469 | 210209 | 1001 | 0.804 | exact loop, period 1001 (short) | 1 (0) | 1.00 / 1.00 |
| q88p2 | 327680 | length | 54812 | 53667 / 55756 | 55041 | 1.000 | fragmentary recycling | 0.00 | 240863 | 86303 | 514 | 0.987 | exact loop, period 514 (short) | 1 (0) | 1.00 / 1.00 |
| q79p2 | 327680 | length | – | – / – | – | 0.000 | no terminal loop | – | – | – | – | – | no terminal loop | 0 (0) | 0.02 / 0.07 |
| q79p3 | 327680 | length | – | 263727 / – | – | 0.000 | no terminal loop | – | – | – | – | – | no terminal loop | 1 (1) | 0.45 / 0.60 |
| q127p3 | 327680 | length | 5376 | 4557 / 6196 | 8171 | 1.000 | short-period exact loop | 0.99 | 220513 | 107132 | 35 | 0.991 | exact loop, period 35 (short) | 1 (0) | 1.00 / 1.00 |
| q120p3 | 327680 | length | 185274 | 184433 / 186094 | 185550 | 0.019 | short-period exact loop | 0.99 | 140328 | 187349 | 2 | – | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| q158p3 | 327680 | length | 280750 | 279931 / 281570 | 280945 | 0.035 | short-period exact loop | 0.96 | 44837 | 282840 | 2 | – | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |

### T4. High-repetition episodes in every analysed reply (R50 >= 0.5 for >= 2,048 positions)

| id | T | outcome | episodes | recovered | terminal | episode spans (g start-end) |
|---|---:|---|---:|---:|---:|---|
| q127p1 | 54226 | stop | 2 | 2 | 0 | 17170-26309, 42817-46568 |
| q165p2 | 150052 | stop | 1 | 0 | 1 | 138657-145120 |
| q94p2 | 163343 | stop | 1 | 0 | 1 | 149160-158641 |
| q79p1 | 327680 | length | 1 | 0 | 1 | 288394-323532 |
| q88p1 | 327680 | length | 1 | 0 | 1 | 143158-323536 |
| q8p2 | 327680 | length | 1 | 0 | 1 | 236765-323535 |
| q69p2 | 327680 | length | 1 | 0 | 1 | 181125-323535 |
| q88p2 | 327680 | length | 1 | 0 | 1 | 54812-323536 |
| q79p3 | 327680 | length | 1 | 1 | 0 | 285238-289671 |
| q127p3 | 327680 | length | 1 | 0 | 1 | 5376-323536 |
| q120p3 | 327680 | length | 1 | 0 | 1 | 185274-323535 |
| q158p3 | 327680 | length | 1 | 0 | 1 | 280750-323535 |

Replies with any episode: capped 9/10, finished >=100k 2/19, comparison 1/24.

### T5. Finished replies >= 100k: what precedes termination

| id | T | reasoning tokens | content tokens / chars | valid letter | stated correct | last rep50>=0.5 window: distance to reasoning end | same at 0.25 | episodes (recovered) | R50 max | R50 final | last-window rep50 / chk_verify per 1k vs reply median |
|---|---:|---:|---|---|---|---:|---:|---|---:|---:|---|
| q7p3 | 100217 | 99548 | 667 / 2483 | yes | yes | – | – | 0 (0) | 0.025 | 0.014 | 0.001 / 1.61 vs 0.003 / 1.47 |
| q147p3 | 103977 | 103590 | 384 / 943 | yes | no | – | – | 0 (0) | 0.012 | 0.000 | 0.000 / 0.00 vs 0.000 / 0.49 |
| q69p3 | 108084 | 107432 | 647 / 1950 | yes | no | – | – | 0 (0) | 0.007 | 0.000 | 0.000 / 1.47 vs 0.000 / 1.34 |
| q186p1 | 124517 | 124020 | 492 / 1204 | yes | no | – | – | 0 (0) | 0.000 | 0.000 | 0.000 / 2.63 vs 0.000 / 0.98 |
| q71p1 | 124709 | 124510 | 195 / 601 | yes | no | – | – | 0 (0) | 0.000 | 0.000 | 0.000 / 0.61 vs 0.000 / 0.49 |
| q81p3 | 124863 | 124110 | 751 / 2197 | yes | yes | – | – | 0 (0) | 0.016 | 0.007 | 0.025 / 2.44 vs 0.000 / 0.98 |
| q81p2 | 126276 | 126018 | 258 / 762 | yes | yes | – | – | 0 (0) | 0.008 | 0.000 | 0.000 / 0.32 vs 0.000 / 0.00 |
| q71p2 | 134402 | 134208 | 194 / 612 | yes | yes | – | – | 0 (0) | 0.019 | 0.000 | 0.000 / 0.96 vs 0.000 / 0.49 |
| q145p2 | 142105 | 141894 | 208 / 658 | yes | no | – | – | 0 (0) | 0.009 | 0.000 | 0.000 / 1.14 vs 0.000 / 0.49 |
| q69p1 | 144012 | 143400 | 611 / 1916 | yes | yes | – | – | 0 (0) | 0.011 | 0.000 | 0.000 / 4.15 vs 0.000 / 1.22 |
| q71p3 | 148563 | 148334 | 227 / 738 | yes | yes | – | – | 0 (0) | 0.020 | 0.000 | 0.000 / 1.22 vs 0.000 / 0.49 |
| q13p3 | 149478 | 148970 | 506 / 2202 | yes | yes | – | – | 0 (0) | 0.011 | 0.006 | 0.000 / 2.64 vs 0.000 / 1.47 |
| q165p2 | 150052 | 149264 | 780 / 2323 | yes | yes | 0 | 0 | 1 (0) | 1.000 | 0.994 | 0.986 / 0.00 vs 0.014 / 2.69 |
| q147p1 | 150616 | 150210 | 403 / 1167 | yes | no | – | 0 | 0 (0) | 0.423 | 0.406 | 0.341 / 6.17 vs 0.016 / 3.91 |
| q48p3 | 157835 | 157326 | 507 / 1560 | yes | no | – | – | 0 (0) | 0.000 | 0.000 | 0.000 / 2.38 vs 0.000 / 3.91 |
| q94p2 | 163343 | 162785 | 556 / 1539 | yes | yes | 0 | 0 | 1 (0) | 0.850 | 0.831 | 0.920 / 10.52 vs 0.000 / 4.39 |
| q88p3 | 173347 | 173024 | 323 / 1053 | yes | no | – | – | 0 (0) | 0.000 | 0.000 | 0.000 / 0.98 vs 0.000 / 0.49 |
| q115p3 | 197631 | 196909 | 720 / 2105 | yes | yes | – | – | 0 (0) | 0.038 | 0.001 | 0.003 / 2.20 vs 0.000 / 1.95 |
| q127p2 | 206688 | 206107 | 578 / 1863 | yes | yes | 128283 | 5403 | 0 (0) | 0.686 | 0.032 | 0.071 / 1.53 vs 0.184 / 0.98 |

### T6. Onset positions against engine geometry (absolute positions = prompt + generated)

| event | id | abs pos | mod 3,584 | dist. to nearest 3,584k | dist. to nearest 2,043k | mod 4 | nearest of 65,536/131,072/262,144 (dist.) | onset spread @0.3-0.7 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| terminal onset (R50) | q165p2 | 139544 | 3352 | 232 | 620 | 0 | 131072 (8472) | 2823 |
| exact-loop start (LRS, post hoc) | q165p2 | 142972 | 3196 | 388 | 38 | 0 | 131072 (11900) | – |
| terminal onset (R50) | q94p2 | 149399 | 2455 | 1129 | 260 | 3 | 131072 (18327) | 4220 |
| exact-loop start (LRS, post hoc) | q94p2 | 153833 | 3305 | 279 | 608 | 1 | 131072 (22761) | – |
| terminal onset (R50) | q79p1 | 288485 | 1765 | 1765 | 422 | 1 | 262144 (26341) | 1745 |
| exact-loop start (LRS, post hoc) | q79p1 | 291821 | 1517 | 1517 | 328 | 1 | 262144 (29677) | – |
| terminal onset (R50) | q88p1 | 143387 | 27 | 27 | 377 | 3 | 131072 (12315) | 1726 |
| exact-loop start (LRS, post hoc) | q88p1 | 145741 | 2381 | 1203 | 688 | 1 | 131072 (14669) | – |
| terminal onset (R50) | q8p2 | 237003 | 459 | 459 | 15 | 3 | 262144 (25141) | 1710 |
| exact-loop start (LRS, post hoc) | q8p2 | 239262 | 2718 | 866 | 231 | 2 | 262144 (22882) | – |
| terminal onset (R50) | q69p2 | 181273 | 2073 | 1511 | 554 | 1 | 131072 (50201) | 8822 |
| exact-loop start (LRS, post hoc) | q69p2 | 210357 | 2485 | 1099 | 72 | 1 | 262144 (51787) | – |
| terminal onset (R50) | q88p2 | 55041 | 1281 | 1281 | 120 | 1 | 65536 (10495) | 2089 |
| exact-loop start (LRS, post hoc) | q88p2 | 86532 | 516 | 516 | 726 | 0 | 65536 (20996) | – |
| terminal onset (R50) | q127p3 | 8171 | 1003 | 1003 | 1 | 3 | 65536 (57365) | 1639 |
| exact-loop start (LRS, post hoc) | q127p3 | 109927 | 2407 | 1177 | 395 | 3 | 131072 (21145) | – |
| terminal onset (R50) | q120p3 | 185550 | 2766 | 818 | 363 | 2 | 131072 (54478) | 1661 |
| exact-loop start (LRS, post hoc) | q120p3 | 187625 | 1257 | 1257 | 331 | 1 | 131072 (56553) | – |
| terminal onset (R50) | q158p3 | 280945 | 1393 | 1393 | 989 | 1 | 262144 (18801) | 1639 |
| exact-loop start (LRS, post hoc) | q158p3 | 283035 | 3483 | 101 | 942 | 3 | 262144 (20891) | – |
| recovered-episode start | q127p1 | 19965 | 2045 | 1539 | 465 | 1 | 65536 (45571) | – |
| recovered-episode start | q127p1 | 45612 | 2604 | 980 | 666 | 0 | 65536 (19924) | – |
| recovered-episode start | q79p3 | 285329 | 2193 | 1391 | 691 | 1 | 262144 (23185) | – |

Uniformity (descriptive): mean resultant length R of the phase, Monte Carlo p under uniform phase.

| events | n | R (mod 3,584) | p | R (mod 4) | p | median distance to nearest power of two |
|---|---:|---:|---:|---:|---:|---:|
| terminal onset (R50) | 10 | 0.12 | 0.87 | 0.00 | 1.00 | 21971.000 |
| exact-loop start (LRS, post hoc) | 10 | 0.29 | 0.43 | 0.32 | 0.38 | 21953.000 |
| all events | 23 | 0.22 | 0.32 | 0.23 | 0.29 | 22761 |

### T7. Long-reply panel: passes with T >= 100,000 per question (capped in brackets), by configuration

| doc | 0.7.0 orig (2) | 0.7.0 rep-F (3) | 0.9.1 (3) | col511 (3) | B0 (3) | B1 (1) | 0.7.0 family (5) | 0.9.1 family (10) | all (15) | empty in unlogged 0.7.0 passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 79 | 1 [1] | 3 [3] | 3 [2] | 3 [2] | 3 [3] | 1 [1] | 4 | 10 | 14 |  |
| 127 | 2 [0] | 2 [1] | 3 [2] | 3 [3] | 2 [1] | 1 [1] | 4 | 9 | 13 | 0.7.0 kpoolfix pass1 |
| 147 | 1 [0] | 2 [0] | 3 [2] | 2 [1] | 1 [1] | 1 [1] | 3 | 7 | 10 |  |
| 88 | 0 [0] | 3 [2] | 2 [2] | 1 [1] | 2 [1] | 1 [1] | 3 | 6 | 9 | 0.7.0 kpoolfix pass1 |
| 81 | 1 [1] | 2 [0] | 2 [2] | 1 [1] | 1 [1] | 1 [1] | 3 | 5 | 8 |  |
| 69 | 0 [0] | 3 [1] | 1 [0] | 1 [1] | 1 [0] | 1 [1] | 3 | 4 | 7 |  |
| 71 | 0 [0] | 3 [0] | 2 [0] | 2 [0] | 0 [0] | 0 [0] | 3 | 4 | 7 |  |
| 120 | 1 [0] | 1 [1] | 2 [1] | 1 [1] | 0 [0] | 0 [0] | 2 | 3 | 5 |  |
| 145 | 2 [1] | 1 [0] | 0 [0] | 0 [0] | 1 [1] | 1 [1] | 3 | 2 | 5 |  |
| 186 | 2 [0] | 1 [0] | 0 [0] | 2 [1] | 0 [0] | 0 [0] | 3 | 2 | 5 |  |
| 8 | 1 [0] | 1 [1] | 1 [0] | 0 [0] | 0 [0] | 1 [1] | 2 | 2 | 4 |  |
| 90 | 0 [0] | 0 [0] | 1 [0] | 1 [0] | 1 [0] | 0 [0] | 0 | 3 | 3 |  |
| 121 | 1 [1] | 0 [0] | 1 [1] | 0 [0] | 1 [1] | 0 [0] | 1 | 2 | 3 |  |
| 7 | 1 [0] | 1 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 2 | 0 | 2 |  |
| 48 | 0 [0] | 1 [0] | 0 [0] | 1 [0] | 0 [0] | 0 [0] | 1 | 1 | 2 |  |
| 94 | 0 [0] | 1 [0] | 0 [0] | 1 [0] | 0 [0] | 0 [0] | 1 | 1 | 2 |  |
| 130 | 0 [0] | 0 [0] | 0 [0] | 1 [0] | 0 [0] | 1 [1] | 0 | 2 | 2 |  |
| 170 | 0 [0] | 0 [0] | 1 [1] | 1 [1] | 0 [0] | 0 [0] | 0 | 2 | 2 |  |
| 13 | 0 [0] | 1 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 | 0 | 1 |  |
| 55 | 0 [0] | 0 [0] | 1 [1] | 0 [0] | 0 [0] | 0 [0] | 0 | 1 | 1 |  |
| 74 | 0 [0] | 0 [0] | 0 [0] | 1 [1] | 0 [0] | 0 [0] | 0 | 1 | 1 |  |
| 76 | 1 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 | 0 | 1 |  |
| 108 | 0 [0] | 0 [0] | 0 [0] | 1 [1] | 0 [0] | 0 [0] | 0 | 1 | 1 |  |
| 109 | 0 [0] | 0 [0] | 1 [1] | 0 [0] | 0 [0] | 0 [0] | 0 | 1 | 1 |  |
| 110 | 0 [0] | 0 [0] | 0 [0] | 1 [0] | 0 [0] | 0 [0] | 0 | 1 | 1 |  |
| 115 | 0 [0] | 1 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 | 0 | 1 |  |
| 118 | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 [0] | 0 [0] | 0 | 1 | 1 |  |
| 157 | 0 [0] | 0 [0] | 0 [0] | 1 [1] | 0 [0] | 0 [0] | 0 | 1 | 1 |  |
| 158 | 0 [0] | 1 [1] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 | 0 | 1 |  |
| 160 | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 [1] | 0 [0] | 0 | 1 | 1 |  |
| 164 | 0 [0] | 0 [0] | 1 [0] | 0 [0] | 0 [0] | 0 [0] | 0 | 1 | 1 |  |
| 165 | 0 [0] | 1 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 | 0 | 1 |  |
| 182 | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 1 [1] | 0 [0] | 0 | 1 | 1 |  |

Concentration and concordance: see summary.json `concentration`, `concordance`.

### T8. Characters per completion token, every rep070-F request (proxy: (reasoning + content chars) / T)

| T band | requests | median | IQR |
|---|---:|---:|---|
| 0-10,000 | 404 | 2.92 | 2.54-3.42 |
| 10,000-25,000 | 96 | 3.19 | 3.02-3.50 |
| 25,000-50,000 | 41 | 3.25 | 2.94-3.48 |
| 50,000-100,000 | 24 | 3.09 | 2.86-3.39 |
| 100,000-cap (finished) | 19 | 3.10 | 2.86-3.33 |
| capped (327,680) | 10 | 2.48 | 2.11-2.60 |

### T9. Within question: rep070-F passes of every question with a capped pass (cpt proxy over the whole reply; first-100k cpt where tokenized)

| doc | pass | T | outcome | cpt proxy | cpt first 100k (re-tokenized) | digit share first 100k |
|---|---:|---:|---|---:|---:|---:|
| 8 | 1 | 18099 | stop | 3.10 | – | – |
| 8 | 2 | 327680 | length | 2.74 | 2.83 | 0.076 |
| 8 | 3 | 80636 | stop | 3.58 | 3.58 | 0.027 |
| 69 | 1 | 144012 | stop | 3.09 | 3.20 | 0.032 |
| 69 | 2 | 327680 | length | 2.58 | 2.31 | 0.066 |
| 69 | 3 | 108084 | stop | 2.74 | 2.74 | 0.051 |
| 79 | 1 | 327680 | length | 2.11 | 2.15 | 0.174 |
| 79 | 2 | 327680 | length | 2.05 | 2.26 | 0.165 |
| 79 | 3 | 327680 | length | 2.39 | 2.31 | 0.150 |
| 88 | 1 | 327680 | length | 2.63 | 2.76 | 0.058 |
| 88 | 2 | 327680 | length | 2.60 | 2.71 | 0.049 |
| 88 | 3 | 173347 | stop | 3.24 | 3.16 | 0.044 |
| 120 | 1 | 14771 | stop | 3.19 | – | – |
| 120 | 2 | 39680 | stop | 4.09 | – | – |
| 120 | 3 | 327680 | length | 2.57 | 2.64 | 0.087 |
| 127 | 1 | 54226 | stop | 2.16 | 2.15 | 0.071 |
| 127 | 2 | 206688 | stop | 2.18 | 2.13 | 0.079 |
| 127 | 3 | 327680 | length | 2.28 | 2.28 | 0.000 |
| 158 | 1 | 7528 | stop | 2.02 | – | – |
| 158 | 2 | 7899 | stop | 2.16 | – | – |
| 158 | 3 | 327680 | length | 2.01 | 1.70 | 0.430 |

### T10. Before the onset: last 32k tokens before a terminal onset against earlier windows (token-weighted)

| id | onset g | rep20 earlier / last 32k | rep50 earlier / last 32k | chk_wait per 1k earlier / last 32k | chk_verify per 1k earlier / last 32k | cpt earlier / last 32k |
|---|---:|---|---|---|---|---|
| q165p2 | 138657 | 0.043 / 0.244 | 0.012 / 0.058 | 29.824 / 83.287 | 2.324 / 4.813 | 2.983 / 2.860 |
| q94p2 | 149160 | 0.038 / 0.099 | 0.001 / 0.005 | 22.879 / 88.484 | 3.793 / 4.848 | 3.081 / 3.255 |
| q79p1 | 288394 | 0.045 / 0.138 | 0.010 / 0.030 | 6.159 / 6.592 | 0.472 / 0.279 | 2.033 / 1.970 |
| q88p1 | 143158 | 0.012 / 0.021 | 0.001 / 0.000 | 26.039 / 64.453 | 1.239 / 1.674 | 2.761 / 2.876 |
| q8p2 | 236765 | 0.019 / 0.043 | 0.000 / 0.003 | 14.105 / 45.201 | 1.141 / 1.744 | 2.847 / 2.672 |
| q69p2 | 181125 | 0.028 / 0.346 | 0.001 / 0.034 | 63.694 / 63.930 | 0.685 / 0.139 | 2.281 / 2.762 |
| q88p2 | 54812 | 0.015 / 0.128 | 0.000 / 0.015 | 14.356 / 15.137 | 0.684 / 0.209 | 2.845 / 2.837 |
| q120p3 | 185274 | 0.061 / 0.115 | 0.010 / 0.028 | 7.595 / 20.856 | 2.527 / 2.930 | 2.630 / 2.568 |
| q158p3 | 280750 | 0.085 / 0.069 | 0.020 / 0.013 | 5.457 / 20.508 | 1.567 / 3.104 | 1.836 / 2.609 |

### T11. Finishing among replies >= 100k, on questions long in both families (question as stratum)

| doc | 0.7.0 family: finished / long | 0.9.1 family: finished / long |
|---|---|---|
| 8 | 1 / 2 | 1 / 2 |
| 48 | 1 / 1 | 1 / 1 |
| 69 | 2 / 3 | 2 / 4 |
| 71 | 3 / 3 | 4 / 4 |
| 79 | 0 / 4 | 2 / 10 |
| 81 | 2 / 3 | 0 / 5 |
| 88 | 1 / 3 | 1 / 6 |
| 94 | 1 / 1 | 1 / 1 |
| 120 | 1 / 2 | 1 / 3 |
| 121 | 0 / 1 | 0 / 2 |
| 127 | 3 / 4 | 2 / 9 |
| 145 | 2 / 3 | 0 / 2 |
| 147 | 3 / 3 | 2 / 7 |
| 186 | 3 / 3 | 1 / 2 |
| **pooled (14 questions)** | **23 / 36** | **18 / 58** |

Questions where the 0.7.0 family finishes a larger share: 8; smaller: 1; equal: 5.

### T12. First 100k tokens of the 29 at-risk replies: rank separation of capped (10) from finished (19)

AUC = P(capped value > finished value); 0.5 = no separation. Replies are not independent (q79 contributes 3 capped).

| feature (first 100k) | capped median | finished median | AUC |
|---|---:|---:|---:|
| cpt | 2.309 | 3.151 | 0.12 (n 10/19) |
| digit | 0.082 | 0.036 | 0.79 (n 10/19) |
| chk_wait | 8.982 | 5.951 | 0.64 (n 10/19) |
| chk_verify | 0.732 | 1.078 | 0.36 (n 10/19) |
| rep20 | 0.023 | 0.012 | 0.71 (n 10/19) |
| R50_100k | 0.006 | 0.000 | 0.74 (n 10/17) |

### T13. Characters per completion token by engine (request-log character counts / completion_tokens)

| configuration | capped: n, median [min-max] | capped with cpt >= 3.0 | finished >= 100k: n, median | 50-100k: median |
|---|---|---:|---|---:|
| 0.7.0 orig | 4, 2.66 [2.00-3.24] | 1 | 10, 2.71 | 3.20 |
| 0.7.0 rep-F | 10, 2.48 [2.01-2.74] | 0 | 19, 3.10 | 3.09 |
| 0.9.1 | 15, 2.58 [1.97-3.64] | 1 | 10, 2.84 | 3.27 |
| col511 | 15, 2.60 [1.57-4.97] | 2 | 10, 3.31 | 3.16 |
| B0 | 11, 2.57 [2.05-3.45] | 2 | 5, 2.98 | 3.19 |
| B1 | 9, 2.73 [1.48-3.83] | 4 | 0, – | 3.08 |

### T14. Rough cost of continuation designs (GPU-hours, 2 GPUs; upper bound - every continuation runs its full horizon)

Inputs: per-request decode rate on replies >= 100k (median / slowest observed, T14 note), KV-limited concurrency, 13.5 min start, 2,410 tok/s prefill.

| design | 0.7.0 (conc., waves, GPU-h central / pess.) | 0.9.1 | B1 | total central / pessimistic |
|---|---|---|---|---|
| A: 8 prefixes x 4 seeds, continue 100k -> cap (227,680 tokens), 0.7.0 + 0.9.1 + B1 | 8, 4, 13.0 / 18.7 | 14, 3, 6.3 / 8.7 | 14, 3, 5.7 / 8.1 | 25.0 / 35.5 |
| B: 8 prefixes x 4 seeds, continue +64,000 tokens, 0.7.0 + 0.9.1 + B1 | 16, 2, 2.4 / 3.2 | 16, 2, 1.7 / 2.1 | 16, 2, 1.6 / 2.0 | 5.7 / 7.3 |
| B+: 12 prefixes x 4 seeds, continue +64,000 tokens, 0.7.0 + 0.9.1 + B1 | 16, 3, 3.3 / 4.5 | 16, 3, 2.3 / 3.0 | 16, 3, 2.2 / 2.8 | 7.8 / 10.3 |
| C: B1 only, 8 prefixes forced through decode (hook, oracle schedule) vs prefill-built, 4 seeds each, +64,000 | – | – | 16, 4, 3.4 / 4.3 | 3.4 / 4.3 |
