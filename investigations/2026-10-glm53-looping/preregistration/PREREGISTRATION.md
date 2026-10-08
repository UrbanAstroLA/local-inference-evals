# Layout bisection: why does tpurtell v0.8.0+ loop more than v0.7.0? (preregistered 2026-10-07, before any data)

## Question
On the same K3.25 checkpoint, v0.7.0 fails to finish (loop or exhaustion) about half as often as v0.8.0 on the
hard-question screen (12-13 vs 20-24 of 40) and leaves a third as many GPQA answers empty (6 vs 20 of 594).
Already ruled out: draft depth (0.8.0 at 5 drafts, sharing off: 17/594), draft-slot sharing, the kpool fixes
(screen 20 vs 22), and the chat template (all 198 GPQA prompts render identically under both templates).
Remaining differences: (1) parallel layout - routed experts TP2 (0.8.0) vs EP2 (0.7.0); attention DCP1 + MLA layer
ownership + 528-byte NOPE records (0.8.0) vs DCP2 (0.7.0); (2) the B12x kernel fork (sparkinfer-glmrt 7fcc094e in 0.8.0).

## Design
Image: tpurtell v0.9.0 (sha256:f36dfb87...; = 0.8.0 + kpool fixes). Weights: K3.25 @ 0490d2f7. DFlash2, 3 drafts, both arms.
- Arm A `glm53-flash-exl3-k3.25@tpurtell+v0.9.0`: v0.9.0 defaults (TP2 experts, DCP1 + ownership split:25, 528 B records).
- Arm D `glm53-flash-exl3-k3.25@tpurtell+v0.9.0+ep2dcp2`: ENABLE_EXPERT_PARALLEL=1, DECODE_CONTEXT_PARALLEL_SIZE=2
  (ownership falls back to tp, records off) = v0.7.0's layout on v0.9.0's kernels.
Instrument: hard-prompt-screen/v1 unchanged (empties_probe5.py: docs 79, 13, 127, 88, 121 x 8 reps, 8 concurrent,
seed 1234 per request, temp 1.0, top_p 0.95, 327,680 max tokens, zlib early stop). Two screens per arm, interleaved
A1, D1, A2, D2, each on a freshly started server. n = 80 requests per arm.

## Outcome and decision rule
Failures = loop + exhaust per arm (of valid requests). A screen is INVALID if it has > 4 errors or < 40 records;
an INVALID screen is rerun once, then the arm is INVALID.
- LAYOUT: D <= A - 10 AND Fisher exact two-sided p < 0.05. Then single-switch arms B (EP2 only) and C (DCP2 only)
  follow, same design, to attribute it.
- KERNELS/OTHER: |A - D| < 10 AND p >= 0.05 AND D >= 36/80 (not 0.7.0-like). The fork is the remaining candidate;
  a reproducible report to tpurtell is the next step (with operator approval).
- INCONCLUSIVE: anything else; report and propose more repeats rather than reinterpreting.
Reported alongside (not part of the rule): per-question failures, finished-request completion tokens, decode speed.
Context only: historical 0.7.0 screens 12-13/40 (S30 v0, S7, S7p), 0.8.0 screens 20-24/40.
