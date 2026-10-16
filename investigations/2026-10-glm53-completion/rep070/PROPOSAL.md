# Proposal: repeatability run of stock tpurtell 0.7.0's K3.25 GPQA record (amendment 12)

Status: **DRAFT, not authorized, nothing run.** Written 2026-10-11 (CPU only). For the operator and the independent
assessor. Thresholds and the budget below are proposals to fix before any request is sent.

## 1. Question and framing

Does stock 0.7.0's representative completion record (4 empty replies of 594; the 0.9.1 record left 16) reproduce when
the same engine is run again under the same recorded per-pass seeds and the same session lifecycle?

This is **repeatability under the recorded seeds**, not a fresh-seed estimate of 0.7.0's empty-reply rate. The run
re-sends the same requests (same prompts, same seeds 1234/1235/1236) to the same image; vLLM draws its sampling noise
from the seed, so the new run shares sampler noise with the old record until batching and floating-point
nondeterminism make the trajectories diverge. Agreement with the old record is therefore expected to be higher than for
independent draws, and a reproduced advantage would say "this record is repeatable", not "0.7.0's population rate is
4/594".

Outcomes (amendment 12) and what follows:

| Outcome | Operational rule (section 6, PROPOSED) | Consequence (amendment 12) |
|---|---|---|
| O1 substantially fewer empties again | rep empties <= 8 and C1 one-sided p < 0.05 | replication support; the tail-mapping/DCP interaction on 0.7.0 becomes the concrete candidate (smallest valid correction, its separate ring and seed-stride defects accounted for; corruption is never restored as a remedy) |
| O2 movement toward the newer engine's count | otherwise, rep empties >= 12 or C2 one-sided p < 0.05 | the original gap is reported as not reproduced, without claiming equivalence; no expensive bisection without a reproducible target |
| O3 ambiguous | neither | report the uncertainty and stop this allocation |

Kept separate (not answered by this run): correcting real defects; repeatability of the engine difference against a
contemporaneous 0.9.1; its mechanism; improving representative completion while preserving correctness.

## 2. Configuration (to reproduce the record)

Everything below is taken from the 0.7.0 record's server log (passes 2-3, 2026-10-10:
`<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+ps.server.log`), its chain logs, the recipe and the stack.

| Item | Value | Source |
|---|---|---|
| Image | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629`, launched **by digest** (still present locally; image Id = that digest) | recipe.json `image`; both chain logs' `image.txt` |
| Recipe / stack | `glm53-flash-exl3-k3.25@tpurtell` (env.sh: `MODEL_PROFILE=k325` only, unchanged since 2026-09-05); stack `<stacks>/glm53-flash-v0.7.0` (start.sh unchanged since 2026-09-05) | recipe, stack |
| Weights | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@701cd7456c13d87bf0147ad946f828a999afb59c`; every snapshot file resolves to the same blob as in the 2026-10-10 config-diff listing | server log args; snapshot listing |
| Chat template | the checkpoint's own (revision 701cd745, `chat_template.jinja` sha256 `34d5ee66...`); no `--chat-template` | server log; config-diff item 11 |
| Draft | DFlash2 `incoai/GLM-5.3-Flash-DFlash2@bf582e4e`, **5 speculative tokens**, bf16 draft KV, separate draft pages | server log `speculative_config` |
| Parallel layout | TP2, **EP2** (`enable_expert_parallel`), **DCP2** (`decode_context_parallel_size 2`, ag_rs), DCP A2A, owner top-k exchange on (`VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=1`, `VLLM_DCP_GLOBAL_TOPK=1`, `VLLM_DCP_TOPK_OWNER_MERGE=1`) | server log args; recipe env |
| Engine settings | attention B12X_MLA_SPARSE; KV `fp8_ds_mla` (656-byte record); max_model_len 1,048,576; max_num_seqs 16; max_num_batched_tokens 2,048; gpu_memory_utilization 0.95; prefix caching on; chunked prefill on; mamba_cache_mode align; baseline full-state rollback; vision on (image limit 16); CUDA graphs FULL_AND_PIECEWISE, capture max 96; reasoning parser glm45, tool parser glm47 | server log line 7 (non-default args sha256 `74f26837...`); engine config line |
| Engine env | recipe.json `env` (as recorded; `B12X_EXL3_BF16_EPILOGUE=0` is the effective value in both sessions; the 2026-10-08 recipe edit corrected a record error, not the setting) | recipe.json; changes.json 2026-10-08 |
| vLLM | `0.1.dev20051+g487ecf187` | server log |
| Startup warm-up | `GLM53_STARTUP_WARMUP=1`; 9 release warm-up passes logged (passes 2-3 session); the eval chain waits for the ready file written after warm-up, then the API | server log; chain |
| Client | lm-eval 0.4.14.dev0 @ b954108c (installed 2026-09-08, unchanged), task `gpqa_diamond_thinking` (yaml sha256 `149658b4...`, unchanged since 2026-09-09; task hash `87e807b3...` in both 0.7.0 sessions); `local-chat-completions`, `--apply_chat_template`, num_concurrent **8**, timeout 10,800, max_retries 3, `max_gen_toks` **327,680**, temperature 1.0, top_p 0.95, lm-eval `--seed 1234+p` | chain log; results json |
| Request seeds | pass 1: **1234**; pass 2: **1235**; pass 3: **1236** | see below |

**What the 0.7.0 pass-1 requests carried.** Pass 1 (2026-09-29) has no request log. Its lm-eval model_args had no
`seed` (results json: `seed_arg` null; the chain of that date passed none), and lm-eval's `local-chat-completions` puts
its model seed into every request payload with a default of 1234 (`api_models.py`, `openai_completions.py`). So every
pass-1 request carried seed 1234. Empirical confirmation with the same lm-eval install and the same model_args: the
0.9.1 record's pass 1 (2026-10-08, also `seed_arg` null) has a proxy log showing seed 1234 on all 198 requests. The
published 0.7.0 rows record seed 1234 for pass 1. Passes 2-3 carried 1235 and 1236 (proxy log). The lm-eval process
seed was 1235/1236/1237 (`--seed 1234+p`), as in every record.

For the new run, pass 1 is sent with `GPQA_SEED_PER_PASS=1` (model arg `seed=1234`): the request payload is identical
(seed 1234); only the lm-eval command line differs from 2026-09-29. Alternatively pass 1 can run without the seed
argument (exactly as on 2026-09-29); the proxy log verifies seed 1234 either way. Operator's choice; recommended: the
explicit argument, for a self-describing command.

### Session lifecycle (reconstructed from the record)

- **Session 1, pass 1 (2026-09-29):** eval chain started 06:51:39 (stops any running GLM container, then starts the
  0.7.0 server fresh). That chain left no further log line; a second chain attached to the same running server at
  06:59:53 with `NO_RESTART=1` and saw it ready after 61 s (07:00:54). Requests 07:01:03 to 10:34:37 (lm-eval log),
  direct to port 8001 (no proxy). The server was stopped at 10:34:58 by the next chain (4bpw 0.8.0). One server start;
  no restart between warm-up and pass 1.
- **Session 2, passes 2-3 (2026-10-10):** fresh server, chain start 00:59:36 via the clean re-run driver, ready after
  498 s, requests 01:08:04 to 08:45:28 through the recording proxy, server log saved and container stopped 08:45:31.
  Pass 1's directory was copied in (`+ps`) and skipped.
- **Replication:** the same split. Session 1: fresh server, warm-up gate, pass 1 only, stop. Session 2: fresh server,
  warm-up gate, passes 2 and 3 back to back, stop. Both sessions through the passive recording proxy (pass 1 originally
  had none; the proxy only forwards and logs). New run tag (for example `+rep070`) so no record directory is touched.
  A runner modelled on `<eval-tools>/diag/gpqa-col511-arm.sh` (abort if a GLM container runs, image-digest check,
  GPU log every 5 minutes, ceiling watchdog, server log and `docker inspect` saved per session, `costs.jsonl` per
  session). The interrupted first chain of 2026-09-29 is not reproduced (it did not restart the server).

**Verified warm-up.** Gate: the chain's ready-file wait (written after the engine's start-up warm-up) and the API
answering; then, from the saved server log, check that the first GPQA request arrived after the last start-up warm-up
line and that the number of release warm-up passes equals the record's (9). For the record's pass 1 only the ready
gate is evidenced (no server log).

## 3. Comparator audit (against the 0.7.0 record)

| Item | Record | Replication | Match |
|---|---|---|---|
| Image | digest 48e254d9 (both sessions; `image.txt`) | same digest, launched by digest | yes |
| Recipe env / stack start.sh | as recorded | unchanged files (env.sh, start.sh dated 2026-09-05; local-env.sh only made IMAGE overridable, same default, 2026-10-04) | yes |
| Published port | both sessions published 8001 on all interfaces (PORT default before 2026-10-10 15:57) | stack now publishes `127.0.0.1:8001` (network rule) | differs: publish address only; the server binds 0.0.0.0:8001 inside the container either way, client on loopback either way. Not material to computation; recorded |
| Weights, draft, template | 701cd745, bf582e4e, template sha 34d5ee66 | same blobs on disk | yes |
| Effective server settings | passes 2-3: logged (args sha 74f26837...); pass 1: **no server log**; chain banner (profile, TP2/EP2, DFlash2 K5, rollback, vision) and startup markers only | logged in both sessions | pass 1 inferred (same launch path and digest); cannot be verified |
| KV pool | 2,768,240 tokens (2026-09-29, chain markers); 2,758,919 (2026-10-10) | recorded at start-up | small run-to-run memory difference already present in the record; peak KV usage <= 9.8% at 8 concurrent, no admission pressure |
| Client | lm-eval b954108c, same task yaml/hash, same model_args except `seed=` on pass 1 (payload identical) | same | yes (pass-1 command line differs; request identical) |
| Proxy | pass 1 none; passes 2-3 recording proxy | both sessions | differs for pass 1 (one extra local hop; no payload change) |
| Request seeds | 1234 (inferred, see section 2) / 1235 / 1236 | 1234 / 1235 / 1236, verified from the proxy log | yes (pass 1 record side inferred) |
| Host kernel / driver / container runtime | 7.0.0-30-generic, driver 595.84, docker 29.7.2, containerd 2.3.4, nvidia-container-toolkit 1.20.0 | unchanged: no package change to any of these since 2026-09-29 (dpkg log; only linux-libc-dev and linux-tools-common headers on 2026-10-09, not the running kernel) | yes; one reboot on 2026-10-03 (between the record's pass 1 and passes 2-3) |
| Desktop graphics on GPU 1 | present on 2026-09-29 (GPU 1 at 2,069 MiB, 100% utilization when the next chain started) | present; display path changed on 2026-10-03 (DP-to-HDMI adapter) | uncontrolled; logged every 5 minutes, as in the arm |
| Compile cache (`<cache>/vllm`, rw mount) | state on each date unknown | current state | uncontrolled; affects start-up time, expected not to change computed kernels (not verified) |
| Timing, batch composition, thermal state | historical | new | uncontrolled (inherent to a repeat; vLLM batching is not bitwise reproducible) |
| Finish reasons and token counts | pass 1 not logged; passes 2-3 logged | all logged | pass 1 comparison of how empties ended is not possible (the record's pass 1 had 0 empties) |

Not determinable: the record's pass-1 effective settings beyond the chain banner; its finish reasons and lengths; its
exact warm-up behaviour. No material unexplained difference is known; the pass-1 server log gap and the shared-seed
dependence are stated limits.

## 4. Cost: the original run, reconstructed, and the earlier estimate

Occupied = 2 GPUs x (server start to server stop), per session (`reconstruct_timing.py`, `timing-reconstruction.json`):

| Session (passes) | Start | Ready | First request | Last request ends | Stop | Wall h | GPU-h |
|---|---|---|---|---|---|---:|---:|
| 2026-09-29 (1) | 06:51:39 | 07:00:54 | 07:01:03 | 10:34:37 | 10:34:58 | 3.722 | 7.44 |
| 2026-10-10 (2-3) | 00:59:36 | 01:07:54 | 01:08:04 | 08:45:28 | 08:45:31 | 7.765 | 15.53 |
| **0.7.0 three passes** | | | | | | 11.487 | **22.97** |

Sources: chain logs (2026-09-29 and 2026-10-10+ps), lm-eval log (pass 1: generation 07:01:03, 198/198 after 3:33:34,
results saved 10:34:37), the next chain's start (pass 1 stop), the proxy log and the re-run driver log (passes 2-3).
For reference, the same reconstruction gives 16.63 for the 0.9.1 record (5.15 + 11.48) and 17.11 for the column-511
arm (5.74 + 11.37).

**The earlier figure of about 15.5 GPU-hours is not the cost of the 0.7.0 three-pass run.** It equals the passes 2-3
session alone (15.53, from that session's server log and chain log); pass 1 ran on 2026-09-29 in its own server
session, for which no server log exists, and occupied a further 7.44. The same omission underlay amendment 11's arm
estimate (13 GPU-hours from about 11.5, the 0.9.1 passes 2-3 session, 11.48), corrected in amendment 11b. 0.7.0's
passes are also slower than 0.9.1's (request windows 3.56 h for pass 1 and 7.62 h for passes 2-3, against 2.37 h and
5.52 h), so a 0.9.1-based estimate understates it further.

## 5. Budget request (PROPOSED)

| Component | GPU-h |
|---|---:|
| Expected (the record's sessions as reconstructed) | 22.97 |
| Session 1 ceiling: 10 min start-up + pass-1 window x 1.15 + 1 min shutdown = 4.28 h | 8.55 |
| Session 2 ceiling: 10 min start-up + passes 2-3 window x 1.15 + 1 min shutdown = 8.95 h | 17.90 |
| **Ceiling for a valid run** (startup, warm-up, shutdown included) | **26.45** |
| One invalid-run allowance: one session-1-sized run (a failed start, or an invalid pass 1 re-run on a fresh server) | 8.55 |
| **Maximum budget** | **35.0** |

The 15% margin covers run-to-run duration variation: the 0.7.0 record's passes took 213, 238 and 218 minutes; the
column-511 arm's pass 1 window was 13% longer than its comparator's. An invalid passes 2-3 session cannot be re-run
inside this budget (it would need about 17.9 more): stop and amend. Watchdog at the maximum (17.5 h wall on 2 GPUs);
the only early stop is the ceiling; no stopping or extension on interim scores. If the operator prefers a smaller
request, the ceiling for a valid run (26.45) without the invalid-run allowance is the minimum that does not risk a
budget stop on a normal run.

## 6. Analysis (draft, to be frozen before running)

Specification: this section. Implementation: `analysis_rep070_draft.py` (reuses the frozen `analysis_col511.py`,
sha256 b5b13ce2..., by import and checks its hash: row loading and validity checks, pairing checks, the exact
sign-flip test, the question bootstrap, the stated-answer adjudicator, log parsing and timing). Both are to be frozen
(sha256 in a dated amendment) before the first request. Roles: **rep** = the new run; **v091** = the
0.9.1 three-pass record; **old070** = the 0.7.0 three-pass record.

- **C1, primary: does the advantage reproduce against 0.9.1?** Exact one-sided sign-flip on per-question differences
  `d_q = e_q(v091) - e_q(rep)` (198 questions, three passes each), alternative fewer empties in rep; effect = empty-rate
  difference with a 20,000-replicate question bootstrap (seed 20261011). Same test and seed as the arm.
- **C2, co-primary: repeatability against the old record.** Exact one-sided sign-flip on `d_q = e_q(rep) - e_q(old070)`,
  alternative more empties in rep (movement toward the newer engine's count); question-bootstrap interval (seed 20261021).
- **Reading (PROPOSED thresholds, evaluated in order):** O1 if rep empties <= 8 and p1 < 0.05; else O2 if rep empties
  >= 12 or p2 < 0.05; else O3. The count thresholds sit either side of the midpoint (10) between the two records (4 and
  16); they are operational thresholds, not causal boundaries.
- **Secondary (descriptive):** raw and stated accuracy over all 594 vs each record (stated final only after the hand
  read of every row where stated and raw differ); rescued and newly empty question-passes vs each record; how empties
  ended (cap exhaustion vs premature stop); completion lengths; answered replies whose hash equals the old record's
  (a direct repeatability measure under shared seeds); empties shared with the old record; GPU-hours and throughput;
  server-log status inside the request windows.
- **Validity gate:** the frozen module's checks (198 questions per pass, both filters, HTTP 200, the pass's seed,
  327,680 cap, temperature 1.0, top_p 0.95, no re-sent request, request-log content hashes equal reply hashes), every
  (question, pass) pair matching both records on seed, prompt, doc, target and message hashes and on server-rendered
  prompt token counts where both are logged, and a proxy log for every rep pass. A failed check marks the run invalid.
- **Definitions** unchanged from the frozen column-511 analysis (`analysis-col511.md` section 2): empty reply, raw and
  stated correct (rep's stated figure final only after every rep row where stated and raw differ is read by hand, then
  the same invocation with `--rep-hedged P:DOC ...` for any reply committing to no single option), cap exhaustion,
  premature stop; all 594 question-passes, nothing excluded. alpha 0.05 for each test; no multiplicity correction (the
  rules use them in a fixed order). An incomplete run (fewer than 594) is not read by these rules.
- **Alternatives the operator may prefer** (not implemented): O1 at <= 6 (stricter); O2 by count alone (>= 12), dropping
  the p2 clause.
- **Invocation after the run:** `analysis_rep070_draft.py --rep-pass <3 pass dirs> --rep-reqlog <s1> <s2>
  --rep-serverlog <s1> <s2> --rep-chainlog <chain.log> --rep-gpu-hours <ledger> --out result-rep070.json >
  result-rep070.md`.
- **Dry runs (done, existing records only):** old 0.7.0 record as rep gives O1 (C1 p = 0.0044, C2 p = 1.0, all
  answered reply hashes identical by construction); 0.9.1 record as rep gives O2 (C2 p = 0.0044). Both valid.

### Power (CPU, `power_rep070.py`, 20,000 simulations per scenario, seed 20261012)

Model: the beta-binomial of `power_col511.py` (empirical-Bayes per-question rates; question-level clustering; passes
independent given the rate), conditional on both historical records as observed; only rep is simulated.

| Scenario for rep's per-question rates | Mean rep empties | P(O1) | P(O2) | P(O3) | P(p1 < 0.05) | P(p2 < 0.05) |
|---|---:|---:|---:|---:|---:|---:|
| S1a old 0.7.0 rates (own fit, pooled) | 4.0 | 0.818 | 0.001 | 0.181 | 0.818 | 0.001 |
| S1b old 0.7.0 rates (0.9.1 concentration) | 4.0 | 0.869 | 0.006 | 0.125 | 0.869 | 0.005 |
| S2 0.9.1 record's rates | 15.8 | 0.018 | 0.877 | 0.105 | 0.022 | 0.840 |
| S3 halfway | 9.9 | 0.255 | 0.380 | 0.365 | 0.266 | 0.369 |
| S4 0.25 x 0.9.1 rates | 4.0 | 0.943 | 0.002 | 0.055 | 0.944 | 0.002 |

Monte Carlo standard errors <= 0.004. If 0.7.0's record reflects its true behaviour, O1 is read 82-94% of the time and
O2 at most 0.6%; if 0.7.0 actually behaves like the 0.9.1 record, O2 is read 88% of the time and O1 1.8%; a halfway
engine splits across all three outcomes, so this design cannot locate an intermediate rate. The model treats rep as an
independent draw; shared seeds can make it resemble the old record more than modelled, which would raise P(O1) under
S1 and lower P(O2) under S2, to the extent that trajectories stay correlated (the identical-hash measure above shows
how far they do). Figures rest on 4 and 16 events
and are approximate.

## 7. Limits

- **Historical comparators for both comparisons.** C1 compares a new run with the 0.9.1 record of 2026-10-08/09; C2
  with the 0.7.0 record of 2026-09-29 and 2026-10-10. Exchangeability under the null is assumed, not designed; matching
  image, recipe, weights, prompts and seeds does not make either comparison a randomized experiment. Anything that
  changed since (desktop load, compile cache, thermal state, time of day) is confounded with the comparison.
- **Repeatability, not a population estimate.** Same seeds on the same engine: the run tests whether this record
  repeats, not whether 0.7.0's empty-reply rate at fresh seeds is 4/594. A reproduced advantage does not identify its
  mechanism; a non-reproduced one does not establish equivalence of the engines.
- **No contemporaneous 0.9.1 control.** If O1, the gap's reproduction is still against a historical 0.9.1 record.
- **Record gaps.** The record's pass 1 has no server log and no request log; its settings and seed are inferred.
- **Defects present.** Stock 0.7.0 carries the tail-ring omission (more affected than 0.9.1), its ring-size and
  seed-stride defects and the column-511 behaviour of its own top-k path; they are part of what is being repeated, not
  controlled.

## 8. Files (this directory)

`PROPOSAL.md` (this; section 6 is the draft analysis specification), `analysis_rep070_draft.py`, `power_rep070.py`, `power-rep070.json`,
`power-rep070.txt`, `run_dryruns_rep070.sh`, `dryrun-rep-old070.{md,json}`, `dryrun-rep-v091.{md,json}`,
`reconstruct_timing.py`, `timing-reconstruction.{json,md}`, `SHA256SUMS-draft`.
