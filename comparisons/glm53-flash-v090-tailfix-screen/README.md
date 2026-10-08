# tpurtell 0.9.0 with and without the DCP1 tail fix (hard-question screen)

`3.25bpw · tpurtell 0.9.0 · DFlash2 ×3` vs `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix · DFlash2 ×3`
(tpurtell/glm-5.3-flash-ext3-2x-rtx#6). Same layout, weights and settings; only the fix differs.

Failures (loop or exhaustion): 41/80 without the fix vs 33/80 (95% Wilson 31.1-52.2%) with it, lower in both screens
(18 and 15 vs 20 and 21). Fisher exact p = 0.27: not yet significant at 80 requests per configuration (about 390 would be
needed); the preregistered rule's label for a difference below its threshold is "no detectable loop effect". This
configuration was added to the screen after the tail-fix checks had been seen, and is recorded as such in the
preregistration folder. Question 13 failed 6 of 16 times without the fix and 0 of 16
with it; that observation is post hoc, not preregistered.
