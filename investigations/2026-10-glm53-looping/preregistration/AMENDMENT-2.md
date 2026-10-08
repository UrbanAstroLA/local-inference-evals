# Amendment 2 (2026-10-07, written before the tail-fix GPU validation and before any arm-E data)

Context: a code-verified decode bug in v0.8.0/v0.9.0 DCP1 (the newest incomplete kpool pool - the current token and up
to two before it - is masked out of all MLA layers whenever context < ~2,044 tokens and seq_len % 4 != 0). DCP2 (arm D)
is unaffected, so arm D differs from arm A in both layout and this bug. A minimal fix exists as an overlay image
(local overlay image, recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.0+tailfix; CPU-verified).

## Gate for running arm E (decided by the GPU validation, (local directory))
Arm E runs only if BOTH: (a) the index check on GPU shows stock dropping the tail at L = 1001-1003 and the patched image
attending it; (b) the decode-vs-prefill check shows, on stock, larger disagreement at positions i < 2044 with i % 4 != 0
than at i % 4 == 0 (KL or |delta logprob| higher), and the patched arm shows no such i % 4 pattern.
If either fails, arm E does not run and the bisection continues as A2, D2.

## Arm E and order
Arm E = arm A's configuration (v0.9.0 defaults, DCP1 + ownership, TP2 experts, 3 drafts, K3.25) on the tail-fix image.
Instrument and counts as before (hard-prompt-screen/v1, 2 screens x 40, fresh server each). Remaining order: A2, E1, D2, E2.

## Decision rule for E (in addition to the unchanged A-vs-D rule)
Failures = loop + exhaust of valid requests; same INVALID rule.
- TAILFIX HELPS LOOPING: E <= A - 10 and Fisher exact two-sided p < 0.05.
- NO DETECTABLE LOOP EFFECT: |A - E| < 10 and p >= 0.05.
- INCONCLUSIVE otherwise.
Joint reading (reported, not reinterpreted): if A-vs-D says LAYOUT and E is not distinguishable from D (|E - D| < 10,
p >= 0.05) while E <= A - 10, the masking bug accounts for the layout effect; if A-vs-D says LAYOUT and E is not
distinguishable from A, the cause is elsewhere in the EP2/DCP2 layout.
