# Component screen on tpurtell v0.9.1, K3.25 - preregistration (written 2026-10-09 before any data)

Purpose: a SCREEN, sized for information per GPU hour, to pick which component of v0.7.0's layout and image moves the two
remaining non-completion questions. A positive result nominates a candidate for a separate confirmatory study; it is not
itself a confirmation.

## Part 1 - doc 88 (tracks the layout: DCP1 62/64 vs DCP2 bundle 29/40 failed to finish, layout bisection 2026-10-07/08)
Half-fraction factorial (2^(3-1), resolution III) over three switches the bisection's DCP2 control changed together.
All arms: recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1 (official v0.9.1 image), DCP1, DFlash2 x3, defaults otherwise.
| Arm | ENABLE_EXPERT_PARALLEL | NOPE_RECORD | GLM53_MLA_OWNERS (+ DRAFT_SLOT_SHARING) |
|---|---|---|---|
| B  | 0 (TP2 experts) | default 1 (528 B record) | default split:25 (sharing default 1) |
| EN | 1 (EP2)         | 0 (656 B record)         | default |
| EO | 1               | default                  | tp (sharing 0) |
| NO | 0               | 0                        | tp (sharing 0) |
Each arm: doc 88 x 12, seeds 5001-5012 (seed = 5000 + rep, same set in every arm), concurrency 12 (one wave; sized so
12 requests fit the KV pool without preemption), temperature 1.0, top_p 0.95, max_tokens 327,680, thinking on, fresh
server per arm, client empties_probe6.py (= the bisection's empties_probe5.py with per-rep seeds and settable
concurrency; same early loop stop: three consecutive last-30k zlib checks < 0.10; same classifier).
Order: B, EN, EO, NO.
Assumptions (stated, not tested here): interactions between the three switches are small (each main effect is aliased
with the other two switches' interaction); slot sharing does not affect doc 88 (GPQA: v0.8.0 K5 + sharing off failed it
3/3), so it rides with GLM53_MLA_OWNERS=tp, which the launcher pairs with sharing off.
Primary outcome: failed to finish (loop or exhaust) per request.
Primary analysis: for each switch, requests with it on (24, two arms) vs off (24): failures, difference with a 95%
Newcombe interval, Fisher exact p (two-sided). Also a logistic fit with the three main effects (reported, no rule).
Screening rule per switch: on-arms have >= 6 fewer failures than off-arms AND Fisher p < 0.10 -> CANDIDATE (propose a
confirmatory single-switch study); otherwise -> "no candidate at this size" (power ~50% for a drop from ~95% to ~72%,
~83% for a drop to 60%; simulated). Not "no effect".
Secondary (descriptive, no rule): loop-onset position (reasoning characters at the early stop) per switch; finished
lengths; arm B vs the historical DCP1 rate 62/64 (drift check).
Validity gates per arm (else the arm is re-run once, then reported as missing): container env/args show the arm's
switches; >= 11 of 12 records without error; KV usage and any preemption in the server log recorded; if preemption
occurs it is reported per arm.

## Part 2 - doc 79 (tracks the v0.7.0 image or 5 draft tokens; GPQA: v0.8.0 with 5 drafts and sharing off still failed it 3/3)
| Arm | Image / stack | Layout | Drafts |
|---|---|---|---|
| B79 | v0.9.1 (glm53-flash-exl3-k3.25@tpurtell+v0.9.1) | DCP1 defaults | 3 |
| V79 | v0.7.0 (glm53-flash-exl3-k3.25@tpurtell, same weight files) | v0.7.0 defaults (EP2 + DCP2) | 3 (DFLASH_TOKENS=3) |
Each: doc 79 x 12, seeds 5001-5012, concurrency 12, otherwise as Part 1.
Rule: V79 has >= 5 fewer failures than B79 AND Fisher p < 0.10 -> the v0.7.0 image (kernel pin + its layout, at equal
draft depth) is the CANDIDATE; >= 5 MORE failures -> draft depth is implicated instead; otherwise unresolved.
Caveat: V79 also differs in layout; Part 1 and the bisection (DCP2 control failed doc 79 like DCP1) inform that.

## Not allowed after seeing data
Changing doc, seeds, concurrency, budget, classifier, thresholds, or dropping requests. Additional arms only as a dated
amendment that says what had been seen.
