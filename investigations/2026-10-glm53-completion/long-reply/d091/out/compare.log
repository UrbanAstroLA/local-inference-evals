# Long-reply side-by-side: stock 0.9.1 d091 (1 pass, seed 70704) vs stock 0.7.0 rep070-F (3 passes, seeds 70701-70703)

**DESCRIPTIVE: one pass, about 8 replies past 100k expected, different seeds from rep070-F; no test, no threshold, no reading.** Definitions: discovery PRESPEC.md (sha256 97040a5c...) through trajectory.py / periods.py run unchanged; table rules as discovery's analyse.py. Labels: [P] prespecified rule, [PH] post hoc (discovery's LRS supplement).

Sets: rep070-F capped 10, finished >= 100k 19, 50-100k 24; d091 capped 6, finished >= 100k 4, 50-100k 7. Re-tokenization drift d091 {'min': -1, 'max': 7, 'hash_verify_failures': 0}.

## 1. First 100k tokens: chars/token, digit share, interjection and verification rates

Token-weighted over full windows ending at or before 100k (discovery T12 features); medians over replies.

| feature | rep070-F capped | rep070-F finished >= 100k | AUC | d091 capped | d091 finished >= 100k | AUC |
|---|---:|---:|---:|---:|---:|---:|
| cpt | 2.309 (n 10) | 3.151 (n 19) | 0.12 | 2.965 (n 6) | 3.270 (n 4) | 0.42 |
| digit | 0.082 (n 10) | 0.036 (n 19) | 0.79 | 0.049 (n 6) | 0.028 (n 4) | 0.67 |
| chk_wait | 8.982 (n 10) | 5.951 (n 19) | 0.64 | 9.384 (n 6) | 2.721 (n 4) | 0.83 |
| chk_verify | 0.732 (n 10) | 1.078 (n 19) | 0.36 | 1.063 (n 6) | 2.146 (n 4) | 0.29 |
| rep20 | 0.023 (n 10) | 0.012 (n 19) | 0.71 | 0.040 (n 6) | 0.014 (n 4) | 0.67 |
| R50_100k | 0.006 (n 10) | 0.000 (n 17) | 0.74 | 0.001 (n 6) | 0.020 (n 4) | 0.50 |

Per-set medians of each reply's window medians (discovery T1; chk_* per 1,000 tokens):

| set | engine | n | cpt | digit | punct | tok_le1char | chk_verify_per1k | chk_wait_per1k | rep50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comparison 50-100k (finished) | rep070-F | 24 | 3.092 | 0.034 | 0.086 | 0.288 | 1.099 | 3.113 | 0.000 |
| comparison 50-100k (finished) | d091 | 7 | 3.412 | 0.024 | 0.079 | 0.251 | 0.977 | 2.930 | 0.000 |
| finished >=100k: all windows | rep070-F | 19 | 3.152 | 0.033 | 0.086 | 0.276 | 0.977 | 5.371 | 0.000 |
| finished >=100k: all windows | d091 | 4 | 3.334 | 0.024 | 0.074 | 0.207 | 2.075 | 2.869 | 0.000 |
| finished >=100k: windows before 100k | rep070-F | 19 | 3.128 | 0.034 | 0.083 | 0.281 | 0.977 | 5.615 | 0.000 |
| finished >=100k: windows before 100k | d091 | 4 | 3.295 | 0.026 | 0.073 | 0.229 | 2.014 | 2.747 | 0.000 |
| capped: windows before 100k | rep070-F | 10 | 2.291 | 0.073 | 0.137 | 0.407 | 0.732 | 8.545 | 0.000 |
| capped: windows before 100k | d091 | 6 | 2.957 | 0.049 | 0.102 | 0.313 | 0.977 | 8.789 | 0.000 |
| capped: pre-onset windows | rep070-F | 10 | 2.312 | 0.069 | 0.138 | 0.399 | 0.732 | 9.765 | 0.001 |
| capped: pre-onset windows | d091 | 6 | 2.987 | 0.046 | 0.101 | 0.307 | 0.855 | 10.803 | 0.000 |
| capped: post-onset windows | rep070-F | 8 | 2.500 | 0.000 | 0.175 | 0.443 | 0.000 | 277.466 | 1.000 |
| capped: post-onset windows | d091 | 6 | 2.500 | 0.000 | 0.200 | 0.500 | 0.000 | 500.000 | 1.000 |

Request-level chars/token proxy (discovery T13): rep070-F capped median 2.48 (0 of 10 >= 3.0); d091 capped median 2.67 (1 of 6 >= 3.0); finished >= 100k 3.10 vs 3.29.

## 2. Capped replies (and finished replies with a terminal onset): onset and loop type

| engine | id | T | outcome | onset g [P] | onset @0.3 / @0.7 | R50 at 100k | loop class [P] | E | LRS after onset | period | loop class [PH] | episodes (recovered) | R50 final / max |
|---|---|---:|---|---:|---|---:|---|---:|---:|---:|---|---|---|
| rep070-F | q165p2 | 150052 | stop | 138657 | 137270 / 140093 | 0.018 | short-period exact loop | 0.71 | 7140 | 14 | exact loop, period 14 (short) | 1 (0) | 0.99 / 1.00 |
| rep070-F | q94p2 | 163343 | stop | 149160 | 147309 / 151529 | 0.000 | fragmentary recycling | 0.00 | 1439 | 1814 | partial / fragmentary (LRS 1439 of 13625) | 1 (0) | 0.83 / 0.85 |
| rep070-F | q79p1 | 327680 | length | 288394 | 287569 / 289314 | 0.011 | fragmentary recycling | 0.00 | 35700 | 246 | exact loop, period 246 (short) | 1 (0) | 1.00 / 1.00 |
| rep070-F | q88p1 | 327680 | length | 143158 | 142275 / 144001 | 0.000 | short-period exact loop | 0.99 | 182166 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| rep070-F | q8p2 | 327680 | length | 236765 | 235946 / 237656 | 0.000 | short-period exact loop | 0.98 | 88653 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| rep070-F | q69p2 | 327680 | length | 181125 | 177951 / 186773 | 0.002 | fragmentary recycling | 0.00 | 116469 | 1001 | exact loop, period 1001 (short) | 1 (0) | 1.00 / 1.00 |
| rep070-F | q88p2 | 327680 | length | 54812 | 53667 / 55756 | 1.000 | fragmentary recycling | 0.00 | 240863 | 514 | exact loop, period 514 (short) | 1 (0) | 1.00 / 1.00 |
| rep070-F | q79p2 | 327680 | length | – | – / – | 0.000 | no terminal loop | – | – | – | no terminal loop | 0 (0) | 0.02 / 0.07 |
| rep070-F | q79p3 | 327680 | length | – | 263727 / – | 0.000 | no terminal loop | – | – | – | no terminal loop | 1 (1) | 0.45 / 0.60 |
| rep070-F | q127p3 | 327680 | length | 5376 | 4557 / 6196 | 1.000 | short-period exact loop | 0.99 | 220513 | 35 | exact loop, period 35 (short) | 1 (0) | 1.00 / 1.00 |
| rep070-F | q120p3 | 327680 | length | 185274 | 184433 / 186094 | 0.019 | short-period exact loop | 0.99 | 140328 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| rep070-F | q158p3 | 327680 | length | 280750 | 279931 / 281570 | 0.035 | short-period exact loop | 0.96 | 44837 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q127p4 | 145775 | stop | 136746 | 71027 / – | 0.708 | fragmentary recycling | 0.23 | 436 | 785 | partial / fragmentary (LRS 436 of 8622) | 7 (6) | 0.66 / 0.94 |
| d091 | q13p4 | 327680 | length | 148004 | 147164 / 148909 | 0.003 | short-period exact loop | 0.97 | 174564 | 46 | exact loop, period 46 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q79p4 | 327680 | length | 238331 | 237512 / 240179 | 0.017 | short-period exact loop | 0.93 | 81438 | 300 | exact loop, period 300 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q81p4 | 327680 | length | 178177 | 177358 / 178997 | 0.000 | short-period exact loop | 0.99 | 147453 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q109p4 | 327680 | length | 132921 | 132102 / 133922 | 0.000 | short-period exact loop | 0.99 | 192515 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q149p4 | 327680 | length | 75929 | 75110 / 76799 | 1.000 | short-period exact loop | 0.99 | 249503 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q170p4 | 327680 | length | 124796 | 123977 / 304509 | 0.000 | short-period exact loop | 0.98 | 176927 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |

- rep070-F: 10 capped; [P] {'fragmentary recycling': 3, 'short-period exact loop': 5, 'no terminal loop': 2}; [PH] {'exact loop': 8, 'no terminal loop': 2}; already repeating at 100k (R50 >= 0.5): 2

- d091: 6 capped; [P] {'short-period exact loop': 6}; [PH] {'exact loop': 6}; already repeating at 100k (R50 >= 0.5): 1

## 3. High-repetition episodes and recovery [P]

An episode is recovered when R50 falls below 0.5 again before the end of the reasoning; a terminal onset holds to the end by definition.

| engine | set | replies | with any episode | with a recovered episode | recovered episodes |
|---|---|---:|---:|---|---:|
| rep070-F | capped | 10 | 9 | q79p3 | 1 |
| rep070-F | finished>=100k | 19 | 2 | - | 0 |
| rep070-F | comparison 50-100k | 24 | 1 | q127p1 | 2 |
| rep070-F | finished from inside repetition (terminal onset, then finished) | | | q165p2, q94p2 | |
| d091 | capped | 6 | 6 | - | 0 |
| d091 | finished>=100k | 4 | 1 | q127p4 | 6 |
| d091 | comparison 50-100k | 7 | 0 | - | 0 |
| d091 | finished from inside repetition (terminal onset, then finished) | | | q127p4 | |

## 4. P(finish | still generating at 100,000 tokens)

| group | finished / at risk | questions | P | 95% question bootstrap |
|---|---:|---:|---:|---|
| d091, pass 1 (seed 70704) | 4 / 10 | 10 | 0.40 | [0.1, 0.7273] |
| stock 0.7.0 rep070-F, passes 1-3 | 19 / 29 | 18 | 0.66 | [0.4444, 0.8571] |
| 0.9.1 record, passes 1-3 | 10 / 25 | 15 | 0.40 | [0.2143, 0.6111] |
| 0.9.1 record, pass 1 only | 3 / 7 | 7 | 0.43 | - |

- d091_minus_rep070F: -0.2552 (95% joint question bootstrap [-0.6291, 0.1053]; 0 undefined replicates)

- d091_minus_v091_record: 0.0 (95% joint question bootstrap [-0.3565, 0.3523]; 0 undefined replicates)
- d091 has one reply per question, so its question bootstrap is a reply bootstrap; the records' intervals cluster passes by question

## 5. Question overlap with the discovery's long-reply panel

- d091 questions past 100k: 10 (13, 27, 79, 81, 109, 121, 127, 145, 149, 170); capped: 13, 79, 81, 109, 149, 170
- of them ever long in the panel's 15 logged passes: 8; never long before: [27, 149]
- also past 100k in rep070-F: 5; long in the stock 0.9.1 record: 6; long in both families in discovery: 5
- of discovery's 14 both-family questions, past 100k again in d091: [79, 81, 121, 127, 145]

| doc | d091: outcome past 100k | panel all (15) | 0.7.0 family (5) | 0.9.1 family (10) | stock 0.9.1 record (3) | rep070-F (3) |
|---|---|---:|---:|---:|---:|---:|
| 13 | capped | 1 | 1 | 0 | 0 | 1 |
| 27 | finished | 0 | 0 | 0 | 0 | 0 |
| 79 | capped | 14 | 4 | 10 | 3 | 3 |
| 81 | capped | 8 | 3 | 5 | 2 | 2 |
| 109 | capped | 1 | 0 | 1 | 1 | 0 |
| 121 | finished | 3 | 1 | 2 | 1 | 0 |
| 127 | finished | 13 | 4 | 9 | 3 | 2 |
| 145 | finished | 5 | 3 | 2 | 0 | 1 |
| 149 | capped | 0 | 0 | 0 | 0 | 0 |
| 170 | capped | 2 | 0 | 2 | 1 | 0 |

All of the above is descriptive: one pass, a handful of long replies, seeds different from every comparator; it cannot adjudicate a termination mechanism, and reaching 100k selects different trajectories on each engine.
