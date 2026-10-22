# Long-reply side-by-side: stock 0.9.1 d091 (1 pass, seed 70705) vs stock 0.7.0 rep070-F (3 passes, seeds 70701-70703)

**DESCRIPTIVE: one pass, about 8 replies past 100k expected, different seeds from rep070-F; no test, no threshold, no reading.** Definitions: discovery PRESPEC.md (sha256 97040a5c...) through trajectory.py / periods.py run unchanged; table rules as discovery's analyse.py. Labels: [P] prespecified rule, [PH] post hoc (discovery's LRS supplement).

Sets: rep070-F capped 10, finished >= 100k 19, 50-100k 24; d091 capped 8, finished >= 100k 4, 50-100k 9. Re-tokenization drift d091 {'min': -2, 'max': 6, 'hash_verify_failures': 0}.

## 1. First 100k tokens: chars/token, digit share, interjection and verification rates

Token-weighted over full windows ending at or before 100k (discovery T12 features); medians over replies.

| feature | rep070-F capped | rep070-F finished >= 100k | AUC | d091 capped | d091 finished >= 100k | AUC |
|---|---:|---:|---:|---:|---:|---:|
| cpt | 2.309 (n 10) | 3.151 (n 19) | 0.12 | 2.684 (n 8) | 3.374 (n 4) | 0.25 |
| digit | 0.082 (n 10) | 0.036 (n 19) | 0.79 | 0.072 (n 8) | 0.030 (n 4) | 0.88 |
| chk_wait | 8.982 (n 10) | 5.951 (n 19) | 0.64 | 11.597 (n 8) | 2.309 (n 4) | 1.00 |
| chk_verify | 0.732 (n 10) | 1.078 (n 19) | 0.36 | 1.190 (n 8) | 0.483 (n 4) | 0.91 |
| rep20 | 0.023 (n 10) | 0.012 (n 19) | 0.71 | 0.022 (n 8) | 0.015 (n 4) | 0.56 |
| R50_100k | 0.006 (n 10) | 0.000 (n 17) | 0.74 | 0.000 (n 8) | 0.000 (n 3) | 0.42 |

Per-set medians of each reply's window medians (discovery T1; chk_* per 1,000 tokens):

| set | engine | n | cpt | digit | punct | tok_le1char | chk_verify_per1k | chk_wait_per1k | rep50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| comparison 50-100k (finished) | rep070-F | 24 | 3.092 | 0.034 | 0.086 | 0.288 | 1.099 | 3.113 | 0.000 |
| comparison 50-100k (finished) | d091 | 9 | 3.091 | 0.041 | 0.083 | 0.284 | 0.488 | 2.686 | 0.000 |
| finished >=100k: all windows | rep070-F | 19 | 3.152 | 0.033 | 0.086 | 0.276 | 0.977 | 5.371 | 0.000 |
| finished >=100k: all windows | d091 | 4 | 3.385 | 0.028 | 0.067 | 0.229 | 0.488 | 2.197 | 0.000 |
| finished >=100k: windows before 100k | rep070-F | 19 | 3.128 | 0.034 | 0.083 | 0.281 | 0.977 | 5.615 | 0.000 |
| finished >=100k: windows before 100k | d091 | 4 | 3.357 | 0.022 | 0.062 | 0.227 | 0.488 | 2.197 | 0.000 |
| capped: windows before 100k | rep070-F | 10 | 2.291 | 0.073 | 0.137 | 0.407 | 0.732 | 8.545 | 0.000 |
| capped: windows before 100k | d091 | 8 | 2.692 | 0.073 | 0.122 | 0.376 | 1.099 | 11.414 | 0.000 |
| capped: pre-onset windows | rep070-F | 10 | 2.312 | 0.069 | 0.138 | 0.399 | 0.732 | 9.765 | 0.001 |
| capped: pre-onset windows | d091 | 8 | 2.737 | 0.063 | 0.123 | 0.371 | 1.038 | 14.038 | 0.000 |
| capped: post-onset windows | rep070-F | 8 | 2.500 | 0.000 | 0.175 | 0.443 | 0.000 | 277.466 | 1.000 |
| capped: post-onset windows | d091 | 7 | 2.500 | 0.000 | 0.200 | 0.500 | 0.000 | 500.000 | 1.000 |

Request-level chars/token proxy (discovery T13): rep070-F capped median 2.48 (0 of 10 >= 3.0); d091 capped median 2.65 (1 of 8 >= 3.0); finished >= 100k 3.10 vs 3.39.

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
| d091 | q69p5 | 327680 | length | 166285 | 165423 / 167105 | 0.000 | short-period exact loop | 0.99 | 159157 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q79p5 | 327680 | length | – | – / – | 0.000 | no terminal loop | – | – | – | no terminal loop | 0 (0) | 0.06 / 0.08 |
| d091 | q81p5 | 327680 | length | 210046 | 209227 / 210866 | 0.000 | short-period exact loop | 0.98 | 115585 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q109p5 | 327680 | length | 151581 | 150762 / 152401 | 0.000 | short-period exact loop | 0.99 | 174051 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q120p5 | 327680 | length | 241801 | 240982 / 242630 | 0.025 | short-period exact loop | 0.97 | 83619 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q161p5 | 327680 | length | 120980 | 120161 / 121800 | 0.001 | short-period exact loop | 0.99 | 204601 | 73 | exact loop, period 73 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q170p5 | 327680 | length | 123485 | 122666 / 124305 | 0.000 | short-period exact loop | 0.99 | 202148 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |
| d091 | q186p5 | 327680 | length | 158143 | 157324 / 158963 | 0.000 | short-period exact loop | 0.99 | 167421 | 2 | exact loop, period 2 (short) | 1 (0) | 1.00 / 1.00 |

- rep070-F: 10 capped; [P] {'fragmentary recycling': 3, 'short-period exact loop': 5, 'no terminal loop': 2}; [PH] {'exact loop': 8, 'no terminal loop': 2}; already repeating at 100k (R50 >= 0.5): 2

- d091: 8 capped; [P] {'short-period exact loop': 7, 'no terminal loop': 1}; [PH] {'exact loop': 7, 'no terminal loop': 1}; already repeating at 100k (R50 >= 0.5): 0

## 3. High-repetition episodes and recovery [P]

An episode is recovered when R50 falls below 0.5 again before the end of the reasoning; a terminal onset holds to the end by definition.

| engine | set | replies | with any episode | with a recovered episode | recovered episodes |
|---|---|---:|---:|---|---:|
| rep070-F | capped | 10 | 9 | q79p3 | 1 |
| rep070-F | finished>=100k | 19 | 2 | - | 0 |
| rep070-F | comparison 50-100k | 24 | 1 | q127p1 | 2 |
| rep070-F | finished from inside repetition (terminal onset, then finished) | | | q165p2, q94p2 | |
| d091 | capped | 8 | 7 | - | 0 |
| d091 | finished>=100k | 4 | 1 | q127p5 | 6 |
| d091 | comparison 50-100k | 9 | 0 | - | 0 |
| d091 | finished from inside repetition (terminal onset, then finished) | | | - | |

## 4. P(finish | still generating at 100,000 tokens)

| group | finished / at risk | questions | P | 95% question bootstrap |
|---|---:|---:|---:|---|
| d091, pass 1 (seed 70705) | 4 / 12 | 12 | 0.33 | [0.0833, 0.625] |
| stock 0.7.0 rep070-F, passes 1-3 | 19 / 29 | 18 | 0.66 | [0.4444, 0.8571] |
| 0.9.1 record, passes 1-3 | 10 / 25 | 15 | 0.40 | [0.2143, 0.6111] |
| 0.9.1 record, pass 1 only | 3 / 7 | 7 | 0.43 | - |

- d091_minus_rep070F: -0.3218 (95% joint question bootstrap [-0.6842, 0.0327]; 0 undefined replicates)

- d091_minus_v091_record: -0.0667 (95% joint question bootstrap [-0.3492, 0.2399]; 0 undefined replicates)
- d091 has one reply per question, so its question bootstrap is a reply bootstrap; the records' intervals cluster passes by question

## 5. Question overlap with the discovery's long-reply panel

- d091 questions past 100k: 12 (8, 69, 71, 79, 81, 88, 109, 120, 127, 161, 170, 186); capped: 69, 79, 81, 109, 120, 161, 170, 186
- of them ever long in the panel's 15 logged passes: 11; never long before: [161]
- also past 100k in rep070-F: 9; long in the stock 0.9.1 record: 10; long in both families in discovery: 9
- of discovery's 14 both-family questions, past 100k again in d091: [8, 69, 71, 79, 81, 88, 120, 127, 186]

| doc | d091: outcome past 100k | panel all (15) | 0.7.0 family (5) | 0.9.1 family (10) | stock 0.9.1 record (3) | rep070-F (3) |
|---|---|---:|---:|---:|---:|---:|
| 8 | finished | 4 | 2 | 2 | 1 | 1 |
| 69 | capped | 7 | 3 | 4 | 1 | 3 |
| 71 | finished | 7 | 3 | 4 | 2 | 3 |
| 79 | capped | 14 | 4 | 10 | 3 | 3 |
| 81 | capped | 8 | 3 | 5 | 2 | 2 |
| 88 | finished | 9 | 3 | 6 | 2 | 3 |
| 109 | capped | 1 | 0 | 1 | 1 | 0 |
| 120 | capped | 5 | 2 | 3 | 2 | 1 |
| 127 | finished | 13 | 4 | 9 | 3 | 2 |
| 161 | capped | 0 | 0 | 0 | 0 | 0 |
| 170 | capped | 2 | 0 | 2 | 1 | 0 |
| 186 | capped | 5 | 3 | 2 | 0 | 1 |

All of the above is descriptive: one pass, a handful of long replies, seeds different from every comparator; it cannot adjudicate a termination mechanism, and reaching 100k selects different trajectories on each engine.
