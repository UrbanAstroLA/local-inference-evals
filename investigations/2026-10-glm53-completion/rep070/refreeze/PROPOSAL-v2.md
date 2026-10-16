# Proposal v2: replication of stock tpurtell 0.7.0's K3.25 GPQA record (amendments 12, 22l)

Status: **RE-FREEZE CANDIDATE, not authorized, nothing run.** Written 2026-10-15, CPU only, from the 2026-10-11 draft
(`../PROPOSAL.md`). Change set 22n (same day) applies the independent review's disposition: **variant F** (70701/70702/
70703) and **budget B-lean** (36.5 occupied GPU-hours, no pre-authorized invalid-run retry) are the reviewed choices, the
frozen three-pass analysis and outcome rules are kept, their meaning is clarified, and incomplete runs get their own
report. **Operator authorization is pending.** Variant R stays fully specified and tested as the alternative. Thresholds
stay PROPOSED until a dated amendment freezes this package by its `SHA256SUMS`. File-level changes and before/after
sha256: `CHANGES-22n.json` (each changed file's earlier version kept as `<file>.pre22n`).

Labels: [M] measured on this machine on 2026-10-15; [R] reconstructed from logs; [S] read from source; [I] inferred.
No GPQA question, answer or reply text appears in this package: ids, hashes, counts and token lengths only.

## 0a. Change set 22n (review disposition, 2026-10-15)

1. **Budget B-lean** (section 5): `PLAN_MAX_GPU_H=36.5` in the runner; no invalid-run allowance (an invalid or incomplete
   session means stop and amend). Refusals tested: 36.5 accepted, 36.51 refused, unset refused. Probability that the
   total reaches 36.5 (the watchdog stops the run): 0 under old 0.7.0 rates with either slope and under 0.9.1-like
   rates with the empirical slope; **0.044** under 0.9.1-like rates with the token-based slope. Probability that session
   2 is not started under the runner's rule: 0 in every scenario [I, model]. No change to seeds or the cap, no request
   dropped.
2. **Watchdog interruptions are recorded** (sections 6e, 7): a proxy copy (`runner/record_proxy_rep070.py`, frozen diff)
   writes start and interrupted events to a sidecar and never crashes on a broken client pipe; the main request log stays
   byte-compatible (byte-identical under a fixed clock [M]); the server is stopped before the proxy. `incomplete_rep070.py`
   reports an incomplete run as INCOMPLETE with its cause, keeps completed passes' results, counts completed /
   cap-reached / interrupted / unstarted requests, never counts interrupted or unstarted requests as empty replies and
   never applies O1/O2/O3 or the 594-based tests. A complete run still goes to the frozen analysis, with the same result
   as dry run D3 [M].
3. **Seeds and task identity explicit** (section 2b): the only change from the records is the request sampling seed.
   New gating checks: the lm-eval task hash 87e807b3b27c8661... in every rep and record pass; negative dry runs for a
   wrong task hash (N7) and a target mismatch (N8), both invalid. The per-question prompt/doc/target/message hash checks
   against both records were already gating. Planned and observed seeds are recorded per pass.
4. **Session split and pass order documented**, with every known difference from the record (section 2c).
5. **Wording:** the coupling-0.25/0.5 power figures are labelled sensitivity assumptions; the measured length
   correlations do not quantify coupling of empty outcomes (sections 2a, 6d).
6. **Category meanings** (review section 4) in this proposal and in the analysis report (section 6a): rules unchanged;
   both comparisons always shown with counts, effects and uncertainty; the sign-flip tests' assumptions stated.
7. **Tail secondary stays descriptive** (review section 5; section 6c): its limits are stated in the report.
8. **Durable local data** (review section 6; section 6f): full request and response bodies in a local raw file per
   session (0600, never published); token ids are not retained (the server returns them only when the request asks for
   them, and the request is not changed [S]); timings, server logs, hashes; a compile-cache inventory before session 1
   and after each session.
9. **Prespecified next steps per outcome** (review section 7; section 6a).
10. **Runner hardening found by the tests:** the GPU logger's and watchdog's `sleep` children were orphaned when the
    loops were killed (harmless, never verified before); the loops now kill their own sleep on TERM [M].

## 0. Changes from the 2026-10-11 draft

1. **Seed policy: two fully specified variants** (section 2a). *R* keeps the recorded seeds 1234/1235/1236 (the draft).
   *F* uses three fresh seeds, 70701/70702/70703, none used by any GPQA pass or request on this machine (inventory of
   every request log, results json and published row, plus a text scan of 10,529 files [M]). Both run through the same
   runner, analysis and dry runs; only the seed list, the run tag (`+rep070-R` / `+rep070-F`) and the meaning of C1/C2
   differ. Recommendation: F, with reasons; presented as a choice.
2. **New evidence on same-seed dependence [M]** (`checks/same-seed-identity.json`, 24 pairs of existing passes). No pair
   of passes ever shares a byte-identical answered reply, including the same image at the same seed in a fresh session
   (0.7.0 at seed 1234 on 2026-09-29 against the retired fixed-seed passes of 2026-09-30: 0 of 195). But same-seed
   pairs stay coupled: the correlation of log reply length over questions is 0.66-0.91 (median 0.86, 13 pairs, across
   engines too), against 0.37-0.48 for different-seed pairs (median 0.45, 11 pairs, same image and same session
   included). "Zero identical replies" therefore does not mean independent draws. The draft's caveat ("to the extent
   trajectories stay correlated") is now a measured property: they stay correlated in outcome-related measures.
3. **Long-reply finishing secondary** (section 6c): P(finish | still generating at 100,000 completion tokens) per
   engine, definitions and code path of `tail-audit/audit.py` (amendment 20c), joint question-clustered bootstrap.
   Descriptive, prespecified, never the reading. Its power is in section 6d.
4. **Verified warm-up now implemented.** The draft specified it (section 2) but `analysis_rep070_draft.py` only
   reported the warm-up count. It is now a gating check per session (section 6b).
5. **Configuration identity added:** the new server logs' non-default args, vLLM version, model snapshots, logged
   environment names and graph mode must equal the old record's passes 2-3 server log (gating).
6. **Seed validity checks are explicit:** request-log seed set, lm-eval `model_args` seed per pass, process seed
   1234+pass, and for F the inventory check. Two negative dry runs (wrong seed, missing question) and four more
   (variant mismatch, missing ready line, configuration mismatch, missing server log) are invalid as required.
7. **Frozen module path.** The draft imported `analysis_col511.py` from a temporary scratch directory. The byte-identical
   copy now sits in `frozen/` and is hash-checked exactly as before (sha256 b5b13ce2...). `power_col511.py` likewise
   (92466804...).
8. **Runner written** (the draft had none): `runner/run_rep070.sh` and a copy of the eval chain with two changes (section
   7). The original chain hardcodes request seed 1233+pass and appends lm-eval's output to
   `<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell.log`, an existing file of the old record [S]; the copy takes a
   per-pass seed list and writes `<recipe>+rep070-<V>.log` instead.
9. **Budget:** the reconstruction reproduces 22.97 / 26.45 / 35.0 exactly [R]. New: the cost depends on the outcome. If
   the new run leaves 0.9.1-like numbers of capped replies, the expected cost is 26.4-31.2 GPU-hours and the 35.0
   watchdog would stop up to 11% of such runs before the end [I, model], i.e. a budget stop most likely in the scenario
   that reads O2. Option B (valid-run ceiling 36.5, maximum 50.0) is offered (section 5). The runner was set to the
   plan maximum then, 35.0 (superseded by change set 22n: B-lean, 36.5).
10. **Comparator audit re-verified** (section 3). Corrections to the draft: the stack's `start.sh` is not the upstream
    file but carries a one-line TileLang cache-directory addition (made 2026-09-05, so present in both record sessions)
    [M]; the compile cache was "current state" in the draft and is now measured (61 files written after the record's end,
    none after the draft) [M]. Nothing else changed since the draft.
11. **Power:** the draft's table reproduces exactly (same seeds). Added: a coupling sensitivity for variant R (section 6d).
12. **Pass-1 seed argument:** the runner sends `seed=1234` explicitly for R's pass 1 (the draft's recommended option);
    the request payload equals the record's.

## 1. Question and framing

Does stock 0.7.0's representative completion record (4 empty replies of 594; the 0.9.1 record left 16) reproduce when the
same engine is run again with the same image, weights, prompts, protocol and session lifecycle?

- **Variant R, repeatability under the recorded seeds.** The run re-sends the record's requests (same prompts, seeds
  1234/1235/1236). vLLM seeds a per-request generator from the request seed [S `gpu_model_runner.py` 1404-1405], so the new
  run shares sampler noise with the old record. Measured consequence (item 0.2): replies are never byte-identical, but
  outcomes stay coupled. A reproduced advantage says "this record repeats at these seeds", not "0.7.0's rate is 4/594".
- **Variant F, a population-rate replication.** Three seeds never used on this machine. The new run is an independent
  draw of 0.7.0 on the same 198 questions. A reproduced advantage says the 0.7.0 engine leaves fewer empties than the
  0.9.1 record at seeds other than the ones that produced its record.

Outcomes (amendment 12), operational rules unchanged from the draft:

| Outcome | Rule (section 6a, PROPOSED) | Consequence (amendment 12) |
|---|---|---|
| O1 substantially fewer empties again | rep empties <= 8 and C1 one-sided p < 0.05 | replication support; the tail-mapping/DCP interaction on 0.7.0 becomes the concrete candidate (smallest valid correction, its separate ring and seed-stride defects accounted for; corruption is never restored as a remedy) |
| O2 movement toward the newer engine's count | otherwise, rep empties >= 12 or C2 one-sided p < 0.05 | the original gap is reported as not reproduced, without claiming equivalence; no expensive bisection without a reproducible target |
| O3 ambiguous | neither | report the uncertainty and stop this allocation |

Kept separate (not answered by this run): correcting real defects; the engine difference against a contemporaneous
0.9.1; its mechanism; improving representative completion while preserving correctness.

## 2. Configuration (re-verified 2026-10-15)

| Item | Value | Status, evidence |
|---|---|---|
| Image | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d9...`, launched by digest through `local-env.sh`'s default `IMAGE` | **re-verified** [M]: present locally, `docker image inspect` Id = sha256:48e254d9... (containerd image store: Id = digest), created 2026-09-06, 8.70 GB; both record sessions' `image.txt` name the same reference |
| Recipe | `glm53-flash-exl3-k3.25@tpurtell`: `env.sh` exports `MODEL_PROFILE=k325` only | **re-verified** [M]: env.sh sha256 ed50bc7d, mtime 2026-09-05; recipe.json mtime 2026-10-10 08:55 (a note edit, before the draft); launch fields (stack dir, container, default ready file) as recorded; /data git status clean for the recipe |
| Stack | `<stacks>/glm53-flash-v0.7.0`: start.sh c852511c, local-env.sh f1f3ced5, start.sh 869b0c0a, model-profiles.sh 13c07d5b | **re-verified, one draft statement corrected** [M]: upstream checkout at 6041c588 (2026-09-06); `start.sh` differs from upstream by one added line (`--env TILELANG_CACHE_DIR=...`), file mtime 2026-09-05 = in place for both record sessions (changes.json 2026-09-05: "TileLang cache line re-applied"); local-env.sh changed 2026-10-04 (IMAGE as default, same digest) and 2026-10-10 15:57 (PORT=127.0.0.1:8001), both before the draft; nothing changed since |
| Published port | stack publishes `127.0.0.1:8001` | **re-verified** [M/S]: local-env.sh `PORT="${PORT:-127.0.0.1:8001}"`, start.sh `--publish "${PORT}:8001"`; the runner refuses a caller-set PORT and resolves the launcher to `127.0.0.1:8001` before starting; after each session it reports `HostConfig.PortBindings` from `docker inspect`. The recording proxy binds 127.0.0.1:8002 [S] |
| Weights | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@701cd745` | **re-verified** [M]: the 35-file snapshot listing equals the 2026-10-10 config-diff listing; every blob's content matches its name (sha256 for LFS, git blob id otherwise): 35/35 |
| Chat template | the checkpoint's own, sha256 34d5ee66...; no `--chat-template` | **re-verified** [M] |
| Draft | DFlash2 `incoai/GLM-5.3-Flash-DFlash2@bf582e4e`, 5 speculative tokens | **re-verified** [M]: 5/5 blobs match; K5 from the stack profile (record banner "DFlash2 K5") |
| Effective server settings | record passes 2-3: non-default args sha256 74f26837..., vLLM 0.1.dev20051+g487ecf187, KV pool 2,758,919, FULL_AND_PIECEWISE, 9 release warm-up lines | **re-verified on the record log** [M]; the new run's logs must match these (gating, section 6b) |
| Engine env | recipe.json `env` (as recorded) | unchanged [M] (file not modified since 2026-10-10) |
| Client | lm-eval 0.4.14.dev0 @ b954108c; task `gpqa_diamond_thinking` yaml 149658b4, utils.py b6038287; `local-chat-completions`, `--apply_chat_template`, concurrency 8, timeout 10,800, max_retries 3, max_gen_toks 327,680, temperature 1.0, top_p 0.95, `--seed 1234+pass` | **re-verified** [M]: HEAD b954108c; the only working-tree entry is the untracked task yaml; no file under the install newer than 2026-10-11 (outside `.git`); seed default 1234 placed in every payload [S `openai_completions.py` 66/84, `api_models.py` 126/174]; prompts identical across passes (198 shared message hashes between seeds 1235 and 1236 [M]) |
| Eval chain | original `<stacks>/glm53-flash-v0.8.0/eval-chain.sh` c4dbb500 (mtime 2026-10-08 20:15; used for the record's passes 2-3) | **re-verified** [M]; the run uses a copy with two changes, checked by an exact diff (section 7) |
| Recording proxy | `<eval-tools>/diag/record_proxy.py` 946d880e (mtime 2026-10-06) | **re-verified** [M]; passive: forwards and appends one line per request [S] |
| Request seeds | R: 1234 / 1235 / 1236; F: 70701 / 70702 / 70703 | section 2a |
| Host | kernel 7.0.0-30-generic (`iommu=pt`), driver 595.84 (open), docker 29.7.2, containerd 2.3.4, nvidia-container-toolkit 1.20.0 | **re-verified** [M]: dpkg log since 2026-10-11 lists libx11 and linux-libc-dev / linux-tools-common 6.8.0-147 headers (2026-10-14), systemd/udev 255.4-1ubuntu8.18, curl, openssh, bluez, bind9, python3-urllib3, python3-jwt (2026-10-15); none of kernel, driver, docker, containerd or toolkit. No reboot since 2026-10-03 21:48; docker and containerd services active since that boot |
| Compile cache | `<cache>/vllm` (rw mount) | **changed vs the record, unchanged since the draft** [M]: 2,884 files (896 root-owned and not readable to the operator account, so no content hash); 2,121 predate 2026-09-29, 702 were written between then and the record's end, 61 on 2026-10-10 after it, 0 after the draft; metadata manifest sha256 0c64c2f1... |
| Disk | logs, request logs and pass directories of a run: about 25 MB [I from the record] | **re-verified** [M]: /data 1.8 TB free; the runner refuses below 5 GiB |

**Pass 1 of the old record** (2026-09-29) has no request log and no server log. Its results json has no seed argument
and lm-eval's default model seed 1234 goes into every payload [S]; the 0.9.1 record's pass 1, run the same way, has a
proxy log showing seed 1234 on all 198 requests [M]. So its requests carried 1234 [I].

### 2b. Seeds and task identity (change set 22n)

The **only** change from the records is the request sampling seed (variant F: 70701/70702/70703; variant R: the
records' own). Everything else is held identical, and checked:

| Item | Value | How it is enforced |
|---|---|---|
| Request seed per pass | F 70701 / 70702 / 70703 (R 1234 / 1235 / 1236) | gating: request-log seed set, `model_args` seed per pass; planned seeds in `run-<V>/seeds-planned.json`, observed in the report's `seeds_observed` |
| lm-eval process / task seed (`--seed`) | 1235 / 1236 / 1237 (1234 + pass), as both records | gating: results json `random_seed` per pass |
| Task | `gpqa_diamond_thinking`, task hash 87e807b3b27c86618285efdab99466609ad12f022ae453146595fc8306b053b0 (every pass of both records [M]) | gating, new in 22n (N7 negative dry run) |
| Rendered prompts, answer ordering, question / target / doc / message hashes | identical to both records per (question, pass) | gating (frozen pairing checks: prompt_hash, doc_hash, target_hash, messages_sha256; server-rendered prompt token counts); N8 negative dry run |
| Sampling, stopping, parser, concurrency, cap | temperature 1.0, top_p 0.95, normal stopping, reasoning parser glm45, concurrency 8, 327,680 tokens | gating (request-log checks, configuration identity of the server's non-default args); chain command line unchanged except the seed (tested) |

### 2c. Session split and pass order; differences from the historical arrangement

Session 1 = pass 1; session 2 = pass 2 then pass 3, back to back on one server, as in the record. Each session starts a
fresh server, waits for the warm-up gate, and stops the server at the end. Known differences from the record's
arrangement:
- **Proxy on pass 1:** the record's pass 1 ran without the recording proxy (direct to 8001); the replication's pass 1
  goes through the proxy (one local hop, passive). The proxy is now the rep070 copy (sidecar, local raw file).
- **Start-up of the record's session 1:** an eval chain started the server at 06:51:39 and left no further log line; a
  second chain attached at 06:59:53 with `NO_RESTART=1` (ready after 61 s) and ran pass 1. The replication's session 1
  is one chain that starts the server and runs pass 1. Not reproduced deliberately (it did not restart the server).
- **Chain version on pass 1:** the record's pass 1 used an older chain (fixed base URL, no seed argument; payload seed
  1234 by lm-eval's default); the replication uses the copy of the current chain with an explicit per-pass seed and a
  run-tag log name.
- **Publish address:** 127.0.0.1:8001 now; all interfaces in both record sessions.
- **Gap between sessions:** 11 days in the record (2026-09-29, 2026-10-10), with a reboot (2026-10-03) and other models'
  sessions in between (the record's session 2 followed a K4 0.9.1 session in the same driver); minutes in the replication.
- **Seeds:** variant F changes the request seeds by design (none for R, except the explicit pass-1 argument).
- **Host state:** compile cache, desktop GPU clients and thermal state differ (section 3); package versions do not.

### Session lifecycle (reconstructed; unchanged from the draft)

- Record session 1 (2026-09-29): one fresh server, pass 1, 07:01:03-10:34:37 lm-eval window, direct to port 8001. Stopped
  at 10:34:58 by the next chain. Record session 2 (2026-10-10): fresh server, ready after 498 s, passes 2-3 through the
  recording proxy 01:08:04-08:45:28, stopped 08:45:31.
- Replication (both variants): session 1 = fresh server, warm-up gate, pass 1, stop; session 2 = fresh server, warm-up
  gate, passes 2 and 3 back to back, stop. Both sessions through the recording proxy (pass 1 originally had none; the
  proxy only forwards and logs). The interrupted first chain of 2026-09-29 is not reproduced (it did not restart the
  server).

## 2a. Seed policy: variant R or variant F (decision for the operator and the assessor)

**Seeds already used on this machine** [M] (`checks/seed-inventory.json`, `checks/seed_scan.py`):
- full GPQA passes, request seeds: **1234, 1235, 1236** only (14 request logs holding full passes, archived ones
  included; 47 results jsons, 28 without a seed argument = default 1234 [S], 19 with 1234/1235/1236; the published rows
  of every public run);
- GPQA prompts outside full passes: **4242** (two gate-run client logs, 8 requests each) and **5001-5012** (published
  single-question screens: doc 79 and doc 88 screens, 12 seeds each);
- no request log anywhere carries any other seed; the broad text scan (every integer after a seed token in 10,529 text
  files under <data-runs>, the public repo and <eval-tools>) found 1,189 distinct values, none of them 70701-70703.

**Chosen for F: 70701, 70702, 70703** (pass 1, 2, 3; "0.7.0, pass n"). Distinct seeds key distinct Philox streams in
torch's CUDA generator (one `torch.Generator.manual_seed(seed)` per request [S]), so consecutive values do not share
noise. The lm-eval process seed stays 1234+pass in both variants, so prompt construction is identical to the records.

| | Variant R (recorded seeds) | Variant F (fresh seeds) |
|---|---|---|
| Request seeds, pass 1 / 2 / 3 | 1234 / 1235 / 1236 | 70701 / 70702 / 70703 |
| Run tag | `+rep070-R` | `+rep070-F` |
| What it measures | repeatability of the record at its seeds (same image, same requests) | 0.7.0's empty-reply behaviour at new seeds: a population-rate replication |
| C1 (rep vs 0.9.1 record) | matched seeds on both sides (common random numbers, as in the original 4-vs-16 comparison); same-seed outcomes are coupled across engines (length correlation 0.66-0.80 between 0.7.0 and 0.9.1 at one seed [M]) | rep is independent of the 0.9.1 record; compares an independent 0.7.0 sample with the fixed 0.9.1 sample |
| C2 (rep vs old 0.7.0 record) | same engine, same seeds: tests whether the record repeats; coupling pulls rep toward the old record, so C2 is weak at revealing that the record was a favourable draw | independent draws of the same engine: tests whether 0.7.0 at new seeds leaves more empties than its own record, i.e. whether the record's low count was favourable draws (or drift) |
| Sign-flip tests | valid under the null for both comparisons: the paired draws are exchangeable under "no difference" with or without coupling | valid; draws independent |
| Power model (beta-binomial, rep independent of the records) | approximate; ignores coupling with both records; sensitivity in 6d | describes this design |
| Thresholds | unchanged | unchanged (no change needed: the rules compare counts and paired tests; independence makes the existing power figures apply directly) |
| Validity | every request at its variant seed; R's seeds equal the records' | same, plus the inventory check |
| Cost | same (section 5) | same |

**Recommendation: F.** Reasons, in order:
1. The decision this run feeds (amendment 12: whether 0.7.0's tail-mapping/DCP interaction becomes the concrete
   candidate) depends on whether the engine leaves fewer empties, not on whether three particular seeds do.
2. Same-seed draws are measurably coupled in reply length (item 0.2), so R may partly re-draw the record's luck. How
   strongly empty outcomes are coupled is not measured: the length correlations do not quantify it. Under assumed
   coupling levels (sensitivity assumptions, section 6d), if 0.7.0 truly behaves like the 0.9.1 record, R reads O2 with
   probability 0.88 at no coupling, 0.70 if a quarter of outcomes were inherited and 0.44 if half were. F does not have
   this exposure.
3. Existing evidence already bears on R's narrower question: the same image at seed 1234 in fresh sessions gave 0, 3 and
   3 empties (2026-09-29 and the two retired fixed-seed passes of 2026-09-30, archived; context only, not pooled, not
   used in any reading) [M]. Pass-level counts do not repeat at a fixed seed.
4. F costs the same, and its independence also suits the secondary.

**For R:** amendment 12's assessment asked for the original per-pass seeds and the "repeatability under the recorded
seeds" framing; R repeats the record's requests exactly and keeps the matched-seed structure of the original comparison,
which helps C1 if coupling is strong. If the assessor's question is strictly "is this record repeatable", R is the
design. A run of both is not proposed (double cost).

## 3. Comparator audit (against the 0.7.0 record, re-verified 2026-10-15)

| Item | Record | Replication | Match | Status |
|---|---|---|---|---|
| Image | digest 48e254d9 (both sessions; `image.txt`) | same, by digest; Id checked before start, reported from `docker inspect` after | yes | re-verified [M] |
| Recipe env / stack launcher | as recorded | env.sh 2026-09-05; start.sh 2026-09-05 incl. the TileLang line; local-env.sh: IMAGE default (same digest) 2026-10-04, PORT 2026-10-10 15:57 | yes, except the publish address | re-verified [M]; start.sh statement corrected |
| Published port | both sessions on all interfaces | `127.0.0.1:8001` | differs: publish address only; the server binds 0.0.0.0:8001 inside the container either way; client on loopback either way; not material to computation | re-verified [M/S] |
| Weights, draft, template | 701cd745, bf582e4e, 34d5ee66 | same blobs, content-verified 40/40 | yes | re-verified [M] |
| Effective server settings | passes 2-3 logged (args 74f26837...); pass 1 not logged (chain banner and start-up markers only) | must equal the passes 2-3 log (gating) | pass 1 inferred | re-verified on the record [M]; pass 1 not determinable |
| KV pool | 2,768,240 (2026-09-29), 2,758,919 (2026-10-10) | recorded | run-to-run difference already in the record; peak KV <= 9.8% | not gating |
| Client | lm-eval b954108c, task yaml 149658b4, same model_args except `seed=` on pass 1 | same; R pass 1 with explicit `seed=1234` (payload identical); F with fresh seeds | yes (R); seeds differ by design (F) | re-verified [M] |
| Eval chain | 2026-09-29: an older chain version (fixed base URL, no seed argument); 2026-10-10: c4dbb500 | copy of c4dbb500 with two changes (seed list; lm-eval log name) | request payload identical for R; log location differs | re-verified [M]; diff frozen |
| Proxy | pass 1 none; passes 2-3 recording proxy 946d880e | both sessions, the rep070 copy (main log byte-compatible; adds a sidecar and a local raw file; frozen diff) | differs for pass 1 (one local hop); passive in both | re-verified [M]; copy tested (22n) |
| Request seeds | 1234 (inferred) / 1235 / 1236 | R: same; F: 70701-70703 | R yes; F by design no | inventory [M] |
| Host kernel / driver / runtime | 7.0.0-30, 595.84, docker 29.7.2, containerd 2.3.4, toolkit 1.20.0 | unchanged (dpkg log since 2026-09-29 and since 2026-10-11) | yes | re-verified [M]; reboot 2026-10-03 between record sessions; none since |
| Desktop graphics on GPU 1 | present (2026-09-29: GPU 1 at 2,069 MiB) | present (2026-10-15: 2,269 MiB, gnome-shell, Xwayland, firefox, snap-store, Claude Desktop) | uncontrolled | logged every 5 minutes [M] |
| Compile cache | state per date unknown | 61 files newer than the record's end, none newer than the draft | uncontrolled; affects start-up time; not expected to change computed kernels (not verified) | measured [M] |
| Timing, batch composition, thermal state | historical | new | uncontrolled (inherent to a repeat; batching is not bitwise reproducible) | — |
| Finish reasons and lengths | pass 1 not logged; passes 2-3 logged | all logged | pass 1 not comparable (it had 0 empties) | — |

Not determinable: the record's pass-1 effective settings beyond the chain banner, its finish reasons and lengths, and its
warm-up behaviour. No material unexplained difference is known.

## 4. Cost of the original run (reconstruction, re-run unchanged)

`reconstruct_timing.py` (copied unchanged) reproduces the draft's `timing-reconstruction.{md,json}` byte for byte [R]:

| Session (passes) | Start | Ready | First request | Last request ends | Stop | Wall h | GPU-h |
|---|---|---|---|---|---|---:|---:|
| 2026-09-29 (1) | 06:51:39 | 07:00:54 | 07:01:03 | 10:34:37 | 10:34:58 | 3.722 | 7.44 |
| 2026-10-10 (2-3) | 00:59:36 | 01:07:54 | 01:08:04 | 08:45:28 | 08:45:31 | 7.765 | 15.53 |
| **0.7.0 three passes** | | | | | | 11.487 | **22.97** |

The earlier figure of about 15.5 GPU-hours was the passes 2-3 session alone (draft section 4, unchanged).

## 5. Budget (PROPOSED; one option to be chosen)

**Option A, the draft's rule [R]** (`budget_rep070.py`): expected 22.97; session ceilings 10 min start-up + window x 1.15
+ 1 min shutdown = 8.55 + 17.90 = **26.45** for a valid run; plus one session-1-sized invalid-run allowance (8.55) =
**35.0** maximum, which is also the watchdog (17.5 h wall on 2 GPUs).

**The cost depends on the outcome [I, model].** A capped reply runs to 327,680 tokens (100-149 minutes each on 0.7.0
[M]). Per pass, the request window grows by 8.7 minutes per cap (least squares over the record's three passes, 0/3/1
caps) to 20.9 minutes per cap (all cap tokens at the passes' 257-265 tok/s aggregate rate). With cap counts drawn from
the power model:

| Scenario for the new run | Slope | Mean GPU-h | p95 | P(> 26.45) | P(> 35.0) |
|---|---|---:|---:|---:|---:|
| old 0.7.0 rates (S1b) | empirical | 22.96 | 24.50 | 0.000 | 0.000 |
| old 0.7.0 rates (S1b) | token-based | 22.96 | 25.84 | 0.028 | 0.000 |
| 0.9.1 record's rates (S2) | empirical | 26.35 | 28.75 | 0.459 | 0.000 |
| 0.9.1 record's rates (S2) | token-based | 31.20 | 36.32 | 0.954 | 0.107 |

So under Option A a run that is heading for O2 can be stopped by the budget before it completes (0-11% of such runs), and
an incomplete run is not read. **Option B (outcome-robust):** valid-run ceiling = the S2 token-based p95 rounded up =
**36.5**; invalid-run allowance = one session 1 at its S2 p95 = 13.5; maximum **50.0**. A leaner variant of B sets the
watchdog at 36.5 with no pre-authorized invalid run (any invalid session: stop and amend).

**Adopted for review (change set 22n): B-lean.** The runner enforces `PLAN_MAX_GPU_H=36.5`, the watchdog stops
everything at APPROVED_GPU_H / 2 GPUs of wall time, and there is no invalid-run allowance: an invalid or incomplete run
is reported (section 6e) and the work stops for an amendment; nothing restarts automatically. Probabilities at 36.5
(`budget_rep070.py --thresholds 35.0 36.5 --approved 36.5`, 20,000 simulations per row) [I, model]:

| Scenario | Slope | Mean GPU-h | p99 | P(total >= 35.0) | P(total >= 36.5) = P(watchdog stop) | P(session 2 not started) |
|---|---|---:|---:|---:|---:|---:|
| S1b old 0.7.0 rates | empirical | 22.96 | 25.19 | 0 | 0 | 0 |
| S1b old 0.7.0 rates | token-based | 22.96 | 27.35 | 0 | 0 | 0 |
| S2 0.9.1 record's rates | empirical | 26.35 | 29.81 | 0 | 0 | 0 |
| S2 0.9.1 record's rates | token-based | 31.20 | 38.69 | 0.107 | **0.044** | 0 |

Session 2 starts only if at least the record's session-2 wall (7.765 h) of approved wall remains after session 1; at
36.5 GPU-hours (18.25 h wall) that needs a session 1 longer than 10.5 h, which no scenario produces (p95 of session 1:
6.6 h). The residual risk is a watchdog stop late in session 2 in the O2-like scenario with the slow slope (4.4%);
such a run is reported INCOMPLETE, never read by O1/O2/O3. Seeds and the cap are not changed and no request is dropped
to fit the budget.

**Variant F costs the same as R.** Evidence: every request in both variants carries a seed, so both use vLLM's
per-request-generator sampling path (seeded requests fall back to the native top-k/top-p sampler [S
`topk_topp_sampler.py` 164-175]); a seed only keys a Philox stream [S]; across the recorded seeds 1234/1235/1236 the
0.7.0 pass windows (213.6, 238.8, 218.5 min) differ by what their cap counts explain (residuals <= 2.3 min after 8.7
min/cap) [M/R]. The model has no seed term.

Disk: about 25 MB per run for logs and pass directories, plus the local raw bodies: the record's replies carry 10-13
million characters per pass (2.9-3.0 characters per completion token; largest single reply 1.2 million) [M], so the raw
files hold at most about 50 MB uncompressed for three passes (gzip members; smaller on disk) [I]; compile-cache
inventories about 0.6 MB each [M]. 1.8 TB free [M]; the runner refuses below 5 GiB.

## 6. Analysis (to be frozen before running)

Implementation: `analysis_rep070.py --variant R|F` (imports the frozen `frozen/analysis_col511.py`, sha256 b5b13ce2...,
and checks its hash at start, as the draft). Roles: **rep** = the new run; **v091** = the 0.9.1 three-pass record;
**old070** = the 0.7.0 three-pass record.

### 6a. Primary (unchanged from the draft)

- **C1:** exact one-sided sign-flip on `d_q = e_q(v091) - e_q(rep)` (198 questions, three passes each), alternative fewer
  empties in rep; effect with a 20,000-replicate question bootstrap (seed 20261011).
- **C2:** exact one-sided sign-flip on `d_q = e_q(rep) - e_q(old070)`, alternative more empties in rep; interval seed
  20261021.
- **Reading (PROPOSED, in order):** O1 if rep empties <= 8 and p1 < 0.05; else O2 if rep empties >= 12 or p2 < 0.05; else
  O3. alpha 0.05 each, no multiplicity correction (fixed order). An incomplete run is not read (section 6e).
  Alternatives the operator may prefer (not implemented): O1 at <= 6; O2 by count alone.
- **What the categories mean** (review section 4; rules unchanged; the analysis prints the meaning beside the category):
  - **O1:** the fresh-seed replication sustains a low empty count and separates from the historical 0.9.1 comparator
    under the specified test. It does not establish equivalence to the original 0.7.0 rate, nor a contemporaneous causal
    engine advantage.
  - **O2:** the replication does not sustain the original low-empty result under the specified rules. This does not
    necessarily mean that no advantage over 0.9.1 remains: p2 can indicate deterioration relative to the old 0.7.0
    record while some advantage over 0.9.1 persists.
  - **O3:** the prespecified evidence is inconclusive.
  - The report always shows both comparisons, C1 and C2, with counts, effect estimates and 95% question-bootstrap
    intervals beside the category.
- **Test assumptions:** both are exact sign-flip tests. "Exact" refers to the calculation (full enumeration of the sign
  patterns), given the assumptions: under the null, each question's pair of per-question empty counts (new run,
  historical record) is exchangeable, i.e. the historical comparator behaves on that question as a draw from the same
  process as the new run, and questions are the independent units. Neither assumption is designed: the comparators are
  historical records, so changes since then are confounded with the comparison.
- **Prespecified next steps after a complete run** (review section 7):
  - **O1:** the historical advantage merits further investigation; consider a contemporaneous comparison before another
    broad mechanism search.
  - **O2:** reassess whether the original 4-versus-16 contrast remains an appropriate optimization target; verified
    correctness fixes keep their value.
  - **O3:** report the uncertainty and make a new budget decision; do not add passes automatically.
- **Secondary (descriptive, as the draft):** raw and stated accuracy (stated final only after the hand read of every
  rep row where stated and raw differ, then `--rep-hedged`); rescued and newly empty question-passes; how empties ended;
  completion lengths; identical answered-reply hashes vs old070; **new:** log reply-length correlation vs each record at
  matched and mismatched passes (the coupling measure of item 0.2); GPU-hours and throughput; server status in the
  request windows.

### 6b. Validity gate (a failed check marks the run invalid)

- The frozen module's checks: 198 questions per pass, both filters, HTTP 200, the pass's seed (rows are matched to a
  pass by request seed: variant R in one call, exactly as the draft; variant F one call per pass with the frozen
  module's seed base set for that pass and restored on every exit path; a self-test shows the per-pass loader returns
  the single-call loader's 594 rows and 15 checks on seeds 1234-1236 [M]), cap 327,680, temperature 1.0, top_p 0.95, no
  re-sent request, request-log content hashes equal reply hashes.
- Pairing with both records on prompt, doc, target and message hashes and server-rendered prompt token counts; seed
  agreement for R, seed difference on all 594 pairs for F.
- **Seeds:** the rep request logs hold exactly the variant's three seeds, 198 requests each, nothing else; each pass's
  lm-eval `model_args` seed equals its variant seed and the process seed is 1234+pass; F's seeds are outside the frozen
  inventory {1234, 1235, 1236, 4242, 5001-5012}.
- **Verified warm-up**, per session (two server logs required): exactly one engine start; 9 release warm-up lines (the
  record's count; also both 0.9.1 logs) ending with the "container is ready" line; no GPQA chat-completion request logged
  before that line; the session's first proxy request later than the last timestamped server line before it. (The
  chain's own gate, ready file plus API, stays in place.) All pass on the record's logs [M].
- **Configuration identity:** both rep server logs' non-default args, vLLM version, model snapshots, logged env names and
  graph mode equal the old record's passes 2-3 log.
- A proxy log for every rep pass.

### 6c. Long-reply finishing secondary (prespecified, descriptive, never the reading)

- **Statistic:** P(finish | still generating at 100,000) = finished / at risk; at risk = proxy `completion_tokens` >=
  100,000; finished = finish reason not `length` (generation ended before the 327,680 cap); censored = `length` at the
  cap. Definitions of `tail-audit/audit.py` (amendment 20c); 200,000 reported descriptively.
- **Groups:** rep passes 1-3; the 0.9.1 record passes 1-3 (10 of 25 finished, 15 questions); the old 0.7.0 record
  passes 2-3 only (10 of 14, 11 questions; pass 1 has no request log, so its lengths are unknown and it is excluded, as in
  amendment 20c). For R also rep passes 2-3 against old070 passes 2-3 (seed-matched).
- **Contrasts:** rep - v091 and rep - old070, estimate and 95% interval from a joint question-clustered bootstrap (the
  same resampled questions on both sides; 2,000 replicates; seeds 20261015, +1, +2). No test, no threshold.
- **Reproduction check [M]:** with the old record standing in as rep the secondary returns 10/14 against 10/25, a
  difference of +0.31 [0.03, 0.56] (amendment 20c: +0.31 [0.03, 0.55], different bootstrap seed).
- Small counts: about 10 finishing events per record. Seeds are shared with both records in R (coupled), not in F.
- **Limits, stated in the report:** the numerator, denominator and question-clustered interval are always shown; the
  small event count and wide intervals prevent this measure from adjudicating a termination mechanism; reaching 100,000
  tokens selects different trajectories on different engines, so the conditional comparison is not a causal engine
  effect.

### 6d. Power (CPU)

**Primary** (`power_rep070.py`; 20,000 simulations per scenario; seed 20261012; the draft's table reproduces exactly
[M]). Columns: P(O1) / P(O2) / P(O3).

| Scenario for rep's per-question rates | Mean empties | no coupling (F; draft) | R, coupling 0.25 | R, coupling 0.5 |
|---|---:|---|---|---|
| S1a old 0.7.0 rates (own fit) | 4.0 | 0.818 / 0.001 / 0.181 | 0.884 / 0.004 / 0.112 | 0.937 / 0.005 / 0.059 |
| S1b old 0.7.0 rates (0.9.1 concentration) | 4.0 | 0.869 / 0.006 / 0.125 | 0.906 / 0.005 / 0.089 | 0.948 / 0.003 / 0.050 |
| S2 0.9.1 record's rates | 15.8 | 0.018 / 0.877 / 0.105 | 0.075 / 0.697 / 0.228 | 0.265 / 0.439 / 0.296 |
| S3 halfway | 9.9 | 0.255 / 0.380 / 0.365 | 0.424 / 0.247 / 0.329 | 0.654 / 0.135 / 0.211 |
| S4 0.25 x 0.9.1 rates | 4.0 | 0.943 / 0.002 / 0.055 | 0.960 / 0.003 / 0.037 | 0.978 / 0.003 / 0.019 |

Coupling C = the share of rep's (question, pass) outcomes assumed to copy the old record's outcome at the same seed.
**These are sensitivity assumptions, not measured empty-outcome dependence.** The measured length correlations (item
0.2) show that same-seed draws are coupled in reply length; they do not quantify coupling of empty outcomes, which
existing records cannot estimate (too few events; the 0/3/3 empties of the same-seed 0.7.0 passes only show it is not
complete). Monte Carlo SE <= 0.004. The mean empty count
under S2 falls with coupling (12.8 at 0.25, 9.9 at 0.5) because rep inherits the old record's 4.

**Long-reply secondary** (`power_long_finish_rep070.py`; 4,000 simulations, 1,000 bootstrap replicates each, seed
20261016; question clustering through per-question reach rates, a beta-binomial fit with strong clustering, and the
clustered bootstrap; rep independent of the records, i.e. variant F):

| Scenario | Mean at risk / finished | P(interval rep - 0.9.1 above 0) | P(interval rep - old070 below 0) | Median interval width (vs 0.9.1 / vs old070) | Unclustered Fisher vs 0.9.1 |
|---|---|---:|---:|---|---:|
| S1 rep = old 0.7.0 record | 21.0 / 15.0 | **0.59** | 0.001 | 0.54 / 0.63 | 0.60 |
| S2 rep = 0.9.1 record | 25.0 / 9.9 | 0.004 | **0.41** | 0.53 / 0.63 | 0.003 |
| S3 0.9.1 reach, 0.7.0 finishing | 25.0 / 18.0 | 0.66 | 0.002 | 0.51 / 0.61 | 0.69 |
| S4 0.7.0 reach, 0.9.1 finishing | 21.1 / 8.5 | 0.005 | 0.37 | 0.56 / 0.65 | 0.004 |

Monte Carlo SE <= 0.008. The fitted finishing prior has no detectable overdispersion (about 10 events per record); with
an imposed finishing heterogeneity (concentration 2, intraclass 0.33) the figures barely move (S1 0.57, S2 0.42). So if
0.7.0's long replies finish as in its record, the interval separates from the 0.9.1 record about 6 times in 10; if they
finish as 0.9.1's, the interval separates from the old record about 4 times in 10; intervals are about 0.5-0.65 wide.
The existing unclustered estimate (`mechanism-ranking/power_long_finish.py`: 0.59 at p = 0.71) is confirmed; clustering
costs little here because finishing shows no detectable question-level heterogeneity. Under R the rep - old070 contrast
is pulled toward 0 to the extent outcomes are coupled.

### 6e. Incomplete runs (change set 22n)

`incomplete_rep070.py probe` decides at the end of every run (also after an interruption): the run is COMPLETE only if
both sessions are in the cost ledger, the watchdog did not fire, and every pass has exactly one samples file with 198
questions and both filters. A complete run goes to the frozen analysis unchanged. Otherwise `incomplete_rep070.py report`:
- labels the replication **INCOMPLETE** with the cause(s) (watchdog fired, session 2 not started, pass p incomplete);
- reports each fully completed pass through the frozen loader (empties, raw correct, provisional stated, finish reasons,
  that pass's own validity checks);
- for any other pass, counts from the proxy log and its sidecar: completed (finish `stop`), cap-reached (finish `length`
  at 327,680; empty by the frozen definition only when the reply is empty, judged from the proxy's `content_chars` because
  lm-eval writes no samples for an unfinished pass), other finishes, interrupted (started, no response) and unstarted
  (198 minus started); without a sidecar these three are "not determinable";
- never inserts interrupted or unstarted requests as empty replies, never applies O1/O2/O3 or the 594-based tests;
  empties among completed requests are descriptive only;
- lists every preserved artifact (pass directories, request logs, sidecars, server logs, inspect files) with size and
  sha256. Nothing is deleted on any exit path; no automatic restart or replacement run.
Tests [M] (`tests/test_incomplete.py`, 14/14): (a) watchdog during pass 3 with passes 1-2 complete; (b) session 2 not
started; (c) pass 2 interrupted with a cap-reached request and interrupted requests, pass 3 unstarted; (d) a complete
run: probe COMPLETE, and the frozen analysis gives the same validity, reading, C1, C2, secondary and descriptives as dry
run D3. Counts in (a) and (c) sum to 198 per pass.

### 6f. Data retained locally (review section 6; never published)

Client and proxy capabilities only: no engine hook, no change to serving behaviour or request payloads.
- **Response bodies:** the proxy copy appends the full request body and the full upstream response body (reasoning and
  final text, finish reason, usage) of every chat-completion exchange, and the request body of every interrupted one, to
  `run-<V>/raw/s<n>.raw.jsonl.gz` (one gzip member per exchange). The file is created under umask 077 with mode 0600 in a
  0700 directory, outside `<gpqa-runs>`, outside this package and its `SHA256SUMS`, and is in no public copy list.
  The main request log stays byte-compatible (tested byte-identical with a fixed clock) [M].
- **Token ids: not retained.** The 0.7.0 image returns `token_ids` / `prompt_token_ids` only when the request sets
  `return_token_ids` (default off; `chat_completion/protocol.py` 415, `serving.py` 1032 and 1101) [S]; lm-eval does not
  set it and the request is not changed.
- **Timings and admission:** per-request start and duration (proxy main log), start/interrupted events (sidecar); the
  server logs are saved whole, including the 10-second status lines (running, waiting, KV usage), speculative metrics and
  any preemption lines.
- **Hashes:** configuration (pinned files, image Id, non-default args hash) and response hashes (main log).
- **Compile cache:** `runner/cache_inventory.py` writes an inventory of `<cache>/vllm` (path, size, mtime,
  sha256 where readable; read only) before session 1 and after each session, and a diff after each (new compilation
  activity). The cache is never cleared or retuned. Current state: 2,884 files, 896 unreadable [M].
- **Scoring:** the raw (flexible-extract) and audited stated-answer procedures are unchanged.

### Invocation after the run

The runner calls, per variant: `analysis_rep070.py --variant V --rep-pass <3 pass dirs> --rep-reqlog <s1> <s2>
--rep-serverlog <s1> <s2> --rep-chainlog <chain logs> --rep-gpu-hours <ledger sum> --out result-rep070-V.json >
result-rep070-V.md`; after the hand read, the same with `--rep-hedged P:DOC ...`.

### Dry runs (done; existing records only; `dryruns/`)

| Run | Expected | Result |
|---|---|---|
| D1 old 0.7.0 record as rep, R | valid, O1 | valid (66/66 gating checks), O1, C1 p = 0.0044, C2 p = 1.0, as the draft |
| D2 0.9.1 record as rep, R | valid, O2 | valid (70/70), O2, C2 p = 0.0044, as the draft |
| D3 0.9.1 record relabelled to 70701/70702/70703, F | valid, O2 | valid (72/72), O2; F's seed checks and the 594 seed differences pass |
| N1 F, pass 2 carried 1235 | invalid | invalid: pass-2 request log missing at 70702, seed set, model_args seed |
| N2 F, question 17 missing from pass 3 | invalid | invalid: 198-question check, 594-pair checks; no reading |
| N3 fresh-seed run analysed as R | invalid | invalid: seeds |
| N4 F, session-2 log without the ready line | invalid | invalid: warm-up check |
| N5 F, 0.9.1 logs with configuration gating | invalid | invalid: configuration identity |
| N6 F, one server log | invalid | invalid: two server logs required |
| N7 F, pass 2 with a different lm-eval task hash (22n) | invalid | invalid: task hash |
| N8 F, one question's target hash altered in pass 1 (22n) | invalid | invalid: target_hash pairing with both records |

Gating counts after change set 22n (task-hash checks added): D1 71/71, D2 75/75, D3 77/77.

Stand-in flags (documented in `run_dryruns_rep070.sh`): the records' pass 1 ran without a seed argument, the old record
has one server log, and the 0.9.1 logs are a different engine; those checks are reported as informative in the stand-ins
only.

## 7. Runner (`runner/run_rep070.sh`)

- `VARIANT=R|F`, `APPROVED_GPU_H` (required; refused if unset, not a number, <= 0 or above `PLAN_MAX_GPU_H`), `DRY_RUN=1`.
- Refusals before anything is created: package `SHA256SUMS`; pinned hashes of recipe env.sh, stack start.sh /
  local-env.sh / start.sh / model-profiles.sh, the original chain, proxy, scorer, task yaml and utils, chat
  template; the chain copy differs from the original by exactly the frozen diff; recipe launch fields; lm-eval HEAD and
  working tree; no launcher or chain override variable set (IMAGE and PORT included); the launcher resolves to the digest
  reference and 127.0.0.1:8001; image Id = digest; weights and draft snapshot listings; no file of the run tag and no
  output directory exists; no GLM container and no container of the 0.7.0 image; no lm_eval process; ports 8001/8002
  free; >= 5 GiB free on /data; GPUs: no compute process except the Claude Desktop GPU process, no GPU client other than
  the desktop graphics clients (the 16d/17c filter, verbatim), GPU0 <= 1 GiB, GPU1 <= 4 GiB, utilization <= 10%, 2 GPUs.
- Run: GPU log every 300 s; watchdog at APPROVED_GPU_H / 2 GPUs of wall from script start (escaped pattern; the only
  early stop); session 1 (proxy, chain with `GPQA_PASSES=1`, server log, `docker inspect` with digest and publish-address
  report, stop, cost line); session 2 only if pass 1 exists and the remaining wall is at least the record's session-2
  wall (7.765 h), else a stated stop; then the analysis. No retry. Cleanup on EXIT/INT/TERM: chain, lm_eval, proxy,
  logger, watchdog, server log and inspect, server stop, one cost line per session. Not `set -e`.
- **Chain copy** (`runner/local-eval-chain-rep070.sh`, frozen diff `local-eval-chain-rep070.diff`): (1)
  `GPQA_SEED_LIST="s1 s2 s3"` sends seed s_pass; (2) lm-eval output to `<gpqa-runs>/<recipe>+rep070-<V>.log`. The
  `lm_eval` command line is otherwise unchanged; for R it equals the original chain's with `GPQA_SEED_PER_PASS=1` (tested).
- Outputs: `../run-<V>/` (server logs, inspect, GPU log, watchdog log, `costs.jsonl`, results); pass directories and
  request logs under `<gpqa-runs>/` with the new tag only; the chain's log under `<data-runs>/benchmarks/<date>-<recipe>+rep070-<V>/`.
- **Change set 22n:** `PLAN_MAX_GPU_H=36.5`; the recording proxy is the frozen copy `runner/record_proxy_rep070.py`
  (checked by an exact diff, like the chain copy); after each chain the server is stopped first and the proxy second
  (SIGTERM; in-flight requests are written as interrupted); `run-<V>/raw/` (0700) holds the local raw bodies;
  `seeds-planned.json` records request and lm-eval seeds per pass; compile-cache inventories and diffs; at the end
  (and on every exit after a session started) the probe sends a complete run to the frozen analysis and anything else to
  the INCOMPLETE report. The GPU logger and the watchdog kill their own sleep on TERM.
- **Tests after 22n [M]:** refusals 18/18 (36.5 accepted, 36.51 refused, unset refused, and the earlier cases); session
  simulation 61/61, including (f) the watchdog firing while a request is in flight through the proxy copy (sidecar start
  + interrupted, raw file 0600 with the interrupted record, no main-log row, INCOMPLETE report counting it as interrupted
  and 197 unstarted); proxy tests 17/17 (`tests/test_proxy_sidecar.py`: main log byte-identical to the original proxy
  under a fixed clock, raw bodies present, mode 0600, both logs consistent after an upstream kill, a broken client pipe
  and SIGTERM); incomplete-run tests 14/14; chain seed 6/6.
- **Tests before 22n [M]:** refusals 17/17 in DRY_RUN (budget unset / above maximum / not a number / zero; variant unset / invalid; a dummy lm_eval process;
  IMAGE, PORT and seed-list overrides; a GLM container simulated by a docker stub; a wrong digest by a stub; a compute
  process by an nvidia-smi stub; the real machine passes for both variants; nothing created); Docker touched read-only
  only (stubs refuse every other subcommand; `docker ps` unchanged before and after). Chain seed lines 6/6. Session
  simulation 42/42 with shims and a fake chain (normal two-session run; SIGTERM during session 2; watchdog; budget too
  small for session 2; session 1 failure). The simulation found and fixed one runner defect: two log lines printed `$?`
  after a `$(date)` substitution, so they always showed 0. The lm_eval match (refusal, watchdog, cleanup) is narrowed to
  the executable's command line, `bin/lm_eval --model`; the earlier runners' `pkill -f lm_eval` would also hit any process naming it.

## 8. Limits

- **Historical comparators for both comparisons.** Exchangeability under the null is assumed, not designed. Anything
  that changed since (desktop load, compile cache, thermal state, time of day) is confounded with the comparison.
- **R: repeatability, coupled to both records; F: one new sample of 594.** Neither identifies a mechanism; a
  non-reproduced gap does not establish equivalence of the engines.
- **No contemporaneous 0.9.1 control.**
- **Record gaps:** the record's pass 1 has no server log and no request log; its settings and seed are inferred, and it
  is excluded from the long-reply secondary.
- **Defects present:** stock 0.7.0 carries the tail-ring omission, its ring-size and seed-stride (indexer) defects and
  the column-511 behaviour of its own top-k path; they are part of what is repeated, not controlled.
- **Models:** the power and budget models rest on 4 and 16 events (primary) and about 10 finishing events per record
  (secondary); figures are approximate. The coupling sensitivity is a stated range, not an estimate.

## 9. Files (this directory)

`PROPOSAL-v2.md` (this), `CHECKS.md`, `CHANGES-22n.json` and the `*.pre22n` copies, `SHA256SUMS`, `make_sums.sh`;
`analysis_rep070.py`, `incomplete_rep070.py`; `run_dryruns_rep070.sh` and
`dryruns/`; `power_rep070.py`, `power_long_finish_rep070.py`, `budget_rep070.py` and `power/`; `reconstruct_timing.py`,
`timing-reconstruction.{md,json}`; `frozen/` (analysis_col511.py, analysis-col511.md, power_col511.py, byte-identical);
`checks/` (seed scan and inventory, same-seed identity, snapshot listings, blob check); `runner/` (runner, chain copy and
diff, proxy copy and diff, compile-cache inventory); `tests/` (refusal, chain-seed, proxy, incomplete-run and
session-simulation tests with their outputs).

## 10. Open items, not verified

- The record's pass-1 configuration and warm-up (no server log): not determinable.
- Whether the compile cache's 61 newer files touch 0.7.0 kernels: not determinable without reading root-owned files
  (expected only to change start-up time).
- Coupling strength for empty replies specifically: not measurable from existing records (too few events); bracketed
  by the sensitivity.
- The budget models' cap-to-time slopes rest on three 0.7.0 passes.
- Decisions pending: the operator's authorization of the reviewed choices (variant F, budget B-lean at 36.5) or another
  option; then a dated amendment freezing `SHA256SUMS`, and a GPU authorization.
