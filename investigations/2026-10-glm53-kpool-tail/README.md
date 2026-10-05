# GLM-5.3-Flash: kpool tail bugs in the tpurtell engine images (2026-10)

**Question.** Does the GLM-5.3-Flash no-answer problem on long reasoning (loops and exhaustion to the 327,680-token
cap) come from runtime bugs fixed upstream after the engine's base was cut?

**Finding 1: two upstream bugs are present in v0.7.0 and v0.8.0.** vllm-project/vllm#57477 (prefill tail-seed kernel
ignores the padded tail stride) and #58454 (one-pool tail ring overwritten by drafts behind a rejected
pool-completing draft). Upstream's own regression tests fail on both images (4 failed, 29 passed) and pass with both
fixes ported (33 passed); see `protocols/kpool-kernel-tests/v1.md`. Ported as tpurtell/glm-5.3-flash-ext3-2x-rtx#5.

**Finding 2: the fixes are not the main cause of non-completion.** On the preregistered hard-prompt screen
(`comparisons/glm53-flash-kpool-screen`): unpatched 22/40 non-ok, one draft token 21/40, both fixes 20/40. Neither
preregistered rule was met. Doc 13 is the one item that clearly improves.

**Finding 3: the fixes cost nothing measurable.** Same KV capacity, decode speed, acceptance and teacher-forced NLL
(`comparisons/glm53-flash-serving-probe`).

**Finding 4: the engine is not bitwise reproducible at temperature 0.** The same configuration run twice diverges
after a median ~318 characters, so greedy parity cannot certify speculative exactness on this stack.

**Open.** The engine gap: on the same weights, v0.7.0 failed 12/40 vs v0.8.0's 24/40 on the screen
(`comparisons/glm53-flash-engine-screen-v0`) and gave 6 vs 20 empty answers in 594 GPQA samples, at the same accuracy.
Next: v0.7.0 with the fixes; an amplified test of the seed bug (small KV pool, long-lived cached prompt, many
requests), which the screens here do not exercise.

Files: `preregistration-kpool-screen.md` (decision rules for arms A/B/C), `preregistration-eval-matrix.md`
(matrix plan and amendments). Both were written before their data.
