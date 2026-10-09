# Amendment 1 (2026-10-09 ~09:45, after arms B, EN, EO, NO and B79 were seen; V79 still running)
Seen: arm B (v0.9.1 defaults, doc 88, seeds 5001-5012, concurrency 12, doc 88 alone) failed 3/12, against 62/64 for
the same DCP1 layout in the 2026-10-07/08 bisection (seed 1234 on every request, 5 docs mixed, concurrency 8).
Added arm S1234 = arm B with seed 1234 on every request (SEED_BASE unset), all else identical (doc 88 x 12, concurrency
12, doc 88 alone, fresh server, same client and classifier). Question: does the fixed seed explain the difference?
Reading: >= 9/12 failed -> the fixed seed explains most of it; <= 5/12 -> it does not (batch mix / concurrency /
image remain); 6-8 -> unresolved. The doc-88 factorial verdict ("no candidate") is unchanged by this arm.
