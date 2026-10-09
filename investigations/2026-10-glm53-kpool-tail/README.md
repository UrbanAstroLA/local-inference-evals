# GLM-5.3-Flash: kpool tail bugs in the tpurtell engine images (2026-10)

**Question.** Does the GLM-5.3-Flash no-answer problem on long reasoning (loops and exhaustion to the 327,680-token
cap) come from runtime bugs fixed upstream after the engine's base was cut?

**Finding 1: two upstream bugs are present in tpurtell 0.7.0 and 0.8.0.** vllm-project/vllm#57477 (prefill tail-seed
kernel ignores the padded tail stride) and #58454 (one-pool tail ring overwritten by drafts behind a rejected
pool-completing draft). Upstream's own regression tests fail on both images (4 failed, 29 passed) and pass with both
fixes ported (33 passed); see `protocols/kpool-kernel-tests/v1.md`. Ported as tpurtell/glm-5.3-flash-ext3-2x-rtx#5 and
shipped in tpurtell 0.9.0.

**Finding 2 (withdrawn 2026-10-09).** This finding rested on hard-question screen counts (arms A, B and C of the
preregistered kpool screen, and the tpurtell 0.7.0 pair of 2026-10-05). Those screens sent request seed 1234 on every
repeat; batching made the repeats vary, but not in a statistically meaningful way, and their rates and rule verdicts were withdrawn (notice:
[`../2026-10-glm53-looping`](../2026-10-glm53-looping), section 4). Repeat 1 of each question from arms B and C and
from the patched 0.7.0 screen is kept as a single draw. Whether the fixes change how often hard questions fail to
finish is not established either way.

**Finding 3: the fixes cost nothing measurable in speed.** Decode speed and acceptance show no consistent change
(`comparisons/glm53-flash-serving-probe`, single runs). KV capacity and NLL are not among the published receipts.

**Finding 4: the engine is not bitwise reproducible at temperature 0.** The same configuration run twice diverges
after a median of about 318 characters, so greedy parity cannot certify speculative exactness on this stack.

**Expected benefit of the fixes.** They correct real cache corruption. Their effect is expected mainly in long-lived
servers with prefix caching and reused system prompts, where cached blocks are reused across requests; the fresh-server
runs here rarely exercise that, and this repository has no measurement of it.

Files: `preregistration-kpool-screen.md` (decision rules for arms A/B/C), `preregistration-eval-matrix.md`
(matrix plan and amendments). Both were written before their data and are kept as written; the screen results they
anticipate are withdrawn as above.

**Labels.** v0.7.0 and v0.8.0 above are `tpurtell 0.7.0` and `tpurtell 0.8.0`; the patched v0.8.0 image is
`tpurtell 0.8.0 + kpool fixes ≈ 0.9.0`, because tpurtell merged #5 and shipped it in v0.9.0 with byte-identical kpool
kernel files; the patched v0.7.0 image is `tpurtell 0.7.0 + kpool fixes`, a separate backport (see `SCHEMA.md`).
The eval-matrix preregistration's 86.4 for tpurtell 0.8.0 (3.25bpw) is a local rescoring that is not published.
