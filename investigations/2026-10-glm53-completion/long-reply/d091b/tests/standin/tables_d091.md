# Long-reply side-by-side: STAND-IN: rep070-F pass 1 (stock 0.7.0, seed 70701) run through the d091 pipeline vs stock 0.7.0 rep070-F (3 passes, seeds 70701-70703)

**DESCRIPTIVE: one pass, about 8 replies past 100k expected, different seeds from rep070-F; no test, no threshold, no reading.** Definitions: discovery PRESPEC.md (sha256 97040a5c...) through trajectory.py / periods.py run unchanged; table rules as discovery's analyse.py. Labels: [P] prespecified rule, [PH] post hoc (discovery's LRS supplement).

Sets: rep070-F capped 10, finished >= 100k 19, 50-100k 24; d091 capped 2, finished >= 100k 4, 50-100k 9. Re-tokenization drift d091 {'min': 0, 'max': 5, 'hash_verify_failures': 0}.

## 1. First 100k tokens: chars/token, digit share, interjection and verification rates

Token-weighted over full windows ending at or before 100k (discovery T12 features); medians over replies.

| feature | rep070-F capped | rep070-F finished >= 100k | AUC | d091 capped | d091 finished >= 100k | AUC |
|---|---:|---:|---:|---:|---:|---:|
| cpt | 2.309 (n 10) | 3.151 (n 19) | 0.12 | 2.452 (n 2) | 2.997 (n 4) | 0.12 |
| digit | 0.082 (n 10) | 0.036 (n 19) | 0.79 | 0.116 (n 2) | 0.063 (n 4) | 0.75 |
| chk_wait | 8.982 (n 10) | 5.951 (n 19) | 0.64 | 15.518 (n 2) | 10.625 (n 4) | 0.62 |
| chk_verify | 0.732 (n 10) | 1.078 (n 19) | 0.36 | 0.839 (n 2) | 1.027 (n 4) | 0.25 |
| rep20 | 0.023 (n 10) | 0.012 (n 19) | 0.71 | 0.021 (n 2) | 0.008 (n 4) | 0.75 |
| R50_100k | 0.006 (n 10) | 0.000 (n 17) | 0.74 | 0.005 (n 2) | 0.000 (n 4) | 0.69 |

Per-set medians of each reply's window medians (discovery T1; chk_* per 1,000 tokens):

| set | engine | n | cpt | digit | punct | tok_le1char | chk_verify_per1k | chk_wait_per1k | rep50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comparison 50-100k (finished) | rep070-F | 24 | 3.092 | 0.034 | 0.086 | 0.288 | 1.099 | 3.113 | 0.000 |
| comparison 50-100k (finished) | d091 | 9 | 2.985 | 0.035 | 0.087 | 0.292 | 0.732 | 3.052 | 0.000 |
| finished >=100k: all windows | rep070-F | 19 | 3.152 | 0.033 | 0.086 | 0.276 | 0.977 | 5.371 | 0.000 |
| finished >=100k: all windows | d091 | 4 | 3.009 | 0.061 | 0.095 | 0.315 | 1.099 | 11.719 | 0.000 |
| finished >=100k: windows before 100k | rep070-F | 19 | 3.128 | 0.034 | 0.083 | 0.281 | 0.977 | 5.615 | 0.000 |
| finished >=100k: windows before 100k | d091 | 4 | 2.987 | 0.062 | 0.095 | 0.304 | 0.977 | 10.681 | 0.000 |
| capped: windows before 100k | rep070-F | 10 | 2.291 | 0.073 | 0.137 | 0.407 | 0.732 | 8.545 | 0.000 |
| capped: windows before 100k | d091 | 2 | 2.430 | 0.119 | 0.134 | 0.408 | 0.732 | 15.503 | 0.000 |
| capped: pre-onset windows | rep070-F | 10 | 2.312 | 0.069 | 0.138 | 0.399 | 0.732 | 9.765 | 0.001 |
| capped: pre-onset windows | d091 | 2 | 2.392 | 0.125 | 0.145 | 0.414 | 0.611 | 16.724 | 0.002 |
| capped: post-onset windows | rep070-F | 8 | 2.500 | 0.000 | 0.175 | 0.443 | 0.000 | 277.466 | 1.000 |
| capped: post-onset windows | d091 | 2 | 2.631 | 0.021 | 0.175 | 0.443 | 0.000 | 256.103 | 1.000 |

Request-level chars/token proxy (discovery T13): rep070-F capped median 2.48 (0 of 10 >= 3.0); d091 capped median 2.37 (0 of 2 >= 3.0); finished >= 100k 3.10 vs 2.98.

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
| d091 | q79p1 | 327680 | length | 288394 | 287569 / 289314 | 0.011 | fragmentary recycling | 0.00 | 35700 | 246 | exact loop, period 246 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q88p1 | 327680 | length | 143158 | 142275 / 144001 | 0.000 | short-period exact loop | 0.99 | 182166 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |

- rep070-F: 10 capped; [P] {'fragmentary recycling': 3, 'short-period exact loop': 5, 'no terminal loop': 2}; [PH] {'exact loop': 8, 'no terminal loop': 2}; already repeating at 100k (R50 >= 0.5): 2

- d091: 2 capped; [P] {'fragmentary recycling': 1, 'short-period exact loop': 1}; [PH] {'exact loop': 2}; already repeating at 100k (R50 >= 0.5): 0

## 3. High-repetition episodes and recovery [P]

An episode is recovered when R50 falls below 0.5 again before the end of the reasoning; a terminal onset holds to the end by definition.

| engine | set | replies | with any episode | with a recovered episode | recovered episodes |
|---|---|---:|---:|---|---:|
| rep070-F | capped | 10 | 9 | q79p3 | 1 |
| rep070-F | finished>=100k | 19 | 2 | - | 0 |
| rep070-F | comparison 50-100k | 24 | 1 | q127p1 | 2 |
| rep070-F | finished from inside repetition (terminal onset, then finished) | | | q165p2, q94p2 | |
| d091 | capped | 2 | 2 | - | 0 |
| d091 | finished>=100k | 4 | 0 | - | 0 |
| d091 | comparison 50-100k | 9 | 1 | q127p1 | 2 |
| d091 | finished from inside repetition (terminal onset, then finished) | | | - | |

## 4. P(finish | still generating at 100,000 tokens)

| group | finished / at risk | questions | P | 95% question bootstrap |
|---|---:|---:|---:|---|
| d091, pass 1 (seed 70704) | 3 / 7 | 7 | 0.43 | [0.0, 0.8333] |
| stock 0.7.0 rep070-F, passes 1-3 | 19 / 29 | 18 | 0.66 | [0.4444, 0.8571] |
| 0.9.1 record, passes 1-3 | 10 / 25 | 15 | 0.40 | [0.2143, 0.6111] |
| 0.9.1 record, pass 1 only | 3 / 7 | 7 | 0.43 | - |

- d091_minus_rep070F: -0.2266 (95% joint question bootstrap [-0.7391, 0.2353]; 4 undefined replicates)

- d091_minus_v091_record: 0.0286 (95% joint question bootstrap [-0.4, 0.4167]; 0 undefined replicates)
- d091 has one reply per question, so its question bootstrap is a reply bootstrap; the records' intervals cluster passes by question

## 5. Question overlap with the discovery's long-reply panel

- d091 questions past 100k: 6 (69, 71, 79, 88, 147, 186); capped: 79, 88
- of them ever long in the panel's 15 logged passes: 6; never long before: -
- also past 100k in rep070-F: 6; long in the stock 0.9.1 record: 5; long in both families in discovery: 6
- of discovery's 14 both-family questions, past 100k again in d091: [69, 71, 79, 88, 147, 186]

| doc | d091: outcome past 100k | panel all (15) | 0.7.0 family (5) | 0.9.1 family (10) | stock 0.9.1 record (3) | rep070-F (3) |
|---|---|---:|---:|---:|---:|---:|
| 69 | finished | 7 | 3 | 4 | 1 | 3 |
| 71 | finished | 7 | 3 | 4 | 2 | 3 |
| 79 | capped | 14 | 4 | 10 | 3 | 3 |
| 88 | capped | 9 | 3 | 6 | 2 | 3 |
| 147 | finished | 10 | 3 | 7 | 3 | 2 |
| 186 | finished | 5 | 3 | 2 | 0 | 1 |

All of the above is descriptive: one pass, a handful of long replies, seeds different from every comparator; it cannot adjudicate a termination mechanism, and reaching 100k selects different trajectories on each engine.
