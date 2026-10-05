> Published as written before the data, except that local file paths are replaced with repo paths or '(local file)'.

# kpool k1 screen - preregistration (written 2026-10-04 before any data)

**Question.** Does the GLM-5.3-Flash kpool rejected-draft corruption (vLLM PR #58454) drive the GPQA
loops on tpurtell v0.8.0?

**Why this test.** The kernel test (the kpool-kernel-tests runs) shows both v0.7.0 and
v0.8.0 corrupt a key pool when a pool-completing draft is rejected with 2+ draft tokens, and do NOT with 1.
So `DFLASH_TOKENS=1` removes the bug and changes nothing else in the stack except draft depth (and speed).

**Arms** (same session, same machine state, run in this order):
- B: `glm53-flash-exl3-k3.25@tpurtell+v0.8.0+k1` - DFLASH_TOKENS=1 (bug cannot fire). First, so the new arm
  completes even if the run is interrupted.
- A: `glm53-flash-exl3-k3.25@tpurtell+v0.8.0` - defaults, DFLASH_TOKENS=3 (bug can fire). Same-session control.

**Requests** - identical to the 2026-09-30 seed replay: docs 79, 13, 127, 88, 121 x 8 repeats = 40 per arm,
8 concurrent, fixed shuffled order Random(7), seed 1234 on every request, temperature 1.0, top_p 0.95,
max_tokens 327,680, thinking on. Client `tools/clients/hard_prompt_screen.py` = empties_probe2.py + early stop.

**Early stop** (decided before data): from 30,000 reasoning characters, every 10,000 new characters check the
zlib ratio of the last 30,000; three consecutive checks < 0.10 stop the request and class it `loop`.
Validated offline on the 80 saved 2026-09-30 traces: stops 27/28 loops, 0/44 ok, 0/8 exhaust.

**Classes** (unchanged): ok = finish stop; loop = stopped early, or finish length with tail zlib < 0.15;
exhaust = finish length with tail zlib >= 0.15; error = transport/server error.

**Primary metric:** non-ok (loop + exhaust) out of 40, per arm.

**Decision rule** (same form as 2026-09-30):
- SUPPORTED - kpool corruption drives the loops on this screen - if A's non-ok >= 2 x B's non-ok
  AND A - B >= 4.
- NOT SUPPORTED otherwise.
- INVALID if either arm has > 4 errors or fails to load; report, do not interpret.

**Secondary, descriptive only (no rule):** loop count alone; per-doc outcomes; doc 88 (looped on v0.7.0 too);
comparison with 2026-09-30 (v0.8.0 default 24/40 non-ok, v0.7.0 12/40) - different session, not pooled.

**Not allowed after seeing data:** changing the budget, classifier, thresholds, stop rule, docs, seed or
order; dropping or re-running requests.

**Known limits.** Five hard docs chosen because they loop, one seed, reps not independent beyond the
divergence batching introduces: this ranks configurations, it does not estimate population rates. B decodes
slower, which also changes batch timing. A SUPPORTED result shows the bug matters on these prompts; the full
GPQA arm is what would confirm it on the benchmark.

**Next step by outcome.**
- SUPPORTED: note to tpurtell (kernel test + this screen + PR #58454); then a full 3-pass GPQA arm with +k1,
  or a patched kernel, against the existing 20/594.
- NOT SUPPORTED: the bug is real but not the main driver on these prompts; next suspect is DCP2 (v0.7.0) vs
  DCP1 (v0.8.0), and the other open upstream issues (#56868, #56605).

---

## Amendment: arm C (written 2026-10-04 11:57 PDT, before any arm C data; arms A/B unchanged)

**Why.** Arm B removes the bug's trigger but also changes draft depth and decode speed. Arm C removes the bug
itself and changes nothing else: the same v0.8.0 image digest plus a 119 kB layer applying a port of vLLM
PR #58454 (`port-kpool-spec-ring-glm53.py`), default 3 draft tokens. A vs C isolates the fix.

**Gate before arm C loads** (runs on GPU after A/B finish): the kernel test on the patched image must show
every scenario correct (2, 3, 5, 7 drafts with a rejected pool-completing draft, ring 8/8/16/16); the unpatched
image must still show corruption. If the patched test fails, arm C does not run.

**Arm C:** `glm53-flash-exl3-k3.25@tpurtell+v0.8.0+kpoolring`, run directly after arm A, same requests,
client, early-stop rule and classes as A and B.

**Decision rule (C):** the FIX HELPS on this screen if A's non-ok >= 2 x C's non-ok AND A - C >= 4.
Otherwise NOT SHOWN on this screen. INVALID if > 4 errors, missing requests, or a failed load.
Secondary, descriptive only: loops alone, per-doc outcomes, C vs B, decode speed (C should match A).

**Not allowed after seeing data:** as above, plus changing the patch.

---

## Amendment 2: arm C redefined (written 2026-10-04 18:36 PDT, before any arm C data)

**What happened.** Arm C attempt 1 never served a request. It failed at startup on a guard added to the port:
the tail cache is not contiguous (`stride = (71808, 1024, 128, 1)`), and tpurtell's prefill seed kernel addresses
it densely. Upstream fixed exactly this on 2026-09-20 (vLLM PR #57477: the dense seed writes 2 KB of raw keys into
another block's indexer region on every prefill and leaves the request's own tail unseeded). Both v0.7.0 and v0.8.0
carry the dense seed kernel. Log: `armC-attempt1-server.log`.

**Arm C is now:** tpurtell v0.8.0 + ports of BOTH upstream fixes, applied in upstream order:
`port-kpool-seed-stride-glm53.py` (#57477) then `port-kpool-spec-ring-glm53.py` (#58454). Default 3 draft tokens.
Everything else (requests, client, early stop, classes, decision rule "FIX HELPS iff A >= 2 x C and A - C >= 4")
is unchanged. A vs C now tests the two fixes together; it does not attribute an effect to either one.

**Gate (extended):** official vLLM `tests/kernels/test_kpool_decode_update_batched.py` (main) must pass on the
patched image with 0 failures, and our rejected-draft test must be all-correct. Pre-run result
(`official-tests.txt`): unpatched 4 failed / 29 passed (the four tests written for these bugs); patched 33 passed.
