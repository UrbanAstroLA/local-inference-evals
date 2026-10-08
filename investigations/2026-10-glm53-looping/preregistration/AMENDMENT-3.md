# Amendment 3 (2026-10-07, after the tail-fix validation, before any arm-E data) - decided AFTER seeing the validation

The Amendment 2 gate for arm E FAILED as written (GATE-E.txt): the predicted i % 4 signature did not separate stock from
the tail-fix build. The validation instead showed a larger, different effect: stock v0.9.0 decode disagrees with prefill
re-scoring of the same tokens at mean KL 0.0656 for positions < 2044 vs 0.0103 with the tail fix (6.4x), and 0.0307 vs
0.0187 for positions >= 2048 (cache contamination persists). On that basis the operator chose to run arm E anyway.
This is a new experimental arm motivated by the validation result, not a reinterpretation of the gate; it is reported
as decided post-validation.

Arm E, order, instrument and the E decision rule are exactly as written in Amendment 2 (arm E = arm A on the tail-fix
image; A2, E1, D2, E2; hard-prompt-screen/v1, 2 x 40; TAILFIX HELPS LOOPING if E <= A - 10 and Fisher p < 0.05;
NO DETECTABLE LOOP EFFECT if |A - E| < 10 and p >= 0.05; INCONCLUSIVE otherwise; joint reading as in Amendment 2).
The A-vs-D rule and verdict are unchanged and computed without arm E.
