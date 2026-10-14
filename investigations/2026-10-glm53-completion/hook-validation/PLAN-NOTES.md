# Hook validation and D2-window preparation (PLAN.md amendment 21, sections 1-2 and 4-8)

2026-10-13. **CPU only. Nothing here has run on a GPU.** A GPQA pass held the GPUs throughout. CPU containers
only (`--network none`, no `--gpus`, `CUDA_VISIBLE_DEVICES=`, Triton interpreter). No GPU or glm53 container was
started or stopped, and nothing under `<stacks>` was touched (the prepared runner only reads its start and
stop scripts, as the earlier runners do).

One image was built, under a new tag only: `local/glm53-v0.9.1-b1-forcehook-mg:local`
(`sha256:1b9f8b15…`). It is the B1 mapguard image plus the hook, and differs from its base in exactly six `.py`
files (`build/manifest-diff.txt`). No reasoning or GPQA text is read or written here: ids, positions, metrics and
hashes only.

This file is the design and the answers to A and C-F. **VALIDATION.md** is B: the validation plan, references,
tolerances, frozen criteria, the GPU runner and its cost.

| Item | Where | Status |
|---|---|---|
| A. Force-history hook | `hook/forcewin_hooks.py` plus marked lines from `hook/make_patch.py` | Implemented; CPU-tested in the B1 image |
| B. Validation | `VALIDATION.md`, `runner/`, `analysis/analyze_hookval.py` | CPU parts pass; GPU runner prepared, not run |
| C. Decode-window-only D2 injection | Same hook (`local_fw_d2`), `tests/test_d2_inject.py` | **Feasible and cheap.** Implemented and CPU-validated; GPU check is phase V4 of the validation session |
| D. Window-exposure instrumentation | `on_selection` (in-graph), shadow ledger, window dumps | Implemented; CPU-tested |
| E. Fixture accounting and frozen list | `fixtures/account_fixtures.py`, `accounting.json`, `first_contrast.json` | Done |
| F. Metrics and aggregation | Section 6 below | Specified before data |

## 1. Summary

- **Hook (A).**
  - One module plus 16 call sites in five engine files (35 marked lines, 80 added lines including the import
    blocks). Inert unless `FORCEWIN=on` **and** the
    request carries `forcewin_hooks`.
  - It records pre-mask log-probabilities of 154842/154820/154827/154829 and of the forced token, the mass
    strictly below and at-or-below each (top_p 0.95 eligibility bounds), the served post-mask probabilities,
    entropy, argmax and top-1.
  - It records once per position, on the intended prefix, before forcing changes the decision. Section 2 gives the
    proof; tests F2, F3, O1-O3 and negative controls N1-N2 check it.
- **Validation (B).**
  - CPU: the hook passes against the installed B1 sampler code, the real `RequestState`, the real `post_update`
    kernel and a deterministic toy model. Every draft schedule is covered: zero accepted, all accepted (oracle),
    rejection at draft positions 1, 2 and 3, and mixed. Measured values equal the single-token reference
    bit for bit. Committed integer state is exact after every step.
  - GPU (to run later): one session, about 1.5 GPU-hours estimated, **proposed ceiling 2.4 GPU-hours**, at most two
    attempts.
- **D2 injection (C): feasible and cheap, done on CPU.** In the tail builder, the tokens of keyed requests at
  window positions get the old mapping. That is the generic per-group slot output, which collapses onto null tail
  block 0 at slot `pos % 8`. Every other token keeps B1's mapping: prefill rows (so the D7 writer stays correct),
  other requests and padding.
  - The real builder, slot kernel, decode writer and the frozen guard show:
    - the injected mapping;
    - pool corruption at 3 in flight and none for 1 in flight (D2's signature);
    - the guard flagging exactly the injected tokens at `tail_w_stash` (mism plus null class), with `idx_w_dec`
      slots still correct.
  - The GPU verification adds about 0.13 GPU-hours inside the validation session (phase V4).
- **Exposure (D).** At designated positions, for every indexer layer, the hook counts how many of the 511 final
  attended pools were decode-written in the window, and how many of those are corrupt or racing according to the
  shadow ledger. Full selections are kept at a few positions. End-of-window page dumps against a prefill rebuild
  give the measured corruption per pool, which joins exactly with those selections.
- **58 histories (E).** All 58 are K4-weight histories, so none can serve a K3.25 probe. The 40 "error" rows are
  the S7K4p screen (engine crash, declared invalid): 36 have no text and 4 are cut by the server failure. The 18
  "exhaustion cut at 120k" rows are client-censored, non-repeating replies from v0.8.0 with K4. Separately, 79
  K4 rows sit *inside* the 527/224/92 classes and are excluded for the same reason. The first contrast uses 8
  K3.25 histories over 6 questions, with a fixed batch.
- **Metrics (F).** The primary measure is the mean of ln P_pre(`</think>`) over prespecified active-reasoning
  positions, including those near natural closes. It is aggregated per window, per run and per arm. The primary
  contrast is three paired runs (B1 and D2-window share a cached prefix in each pair). The run is the unit, because
  D2 couples the windows of a batch. The 0.5-nat and 6-of-8 rules are operational screens.

## 2. A: the force-history hook

### 2.1 Files and call sites

Module: `vllm/forcewin_hooks.py` (`hook/forcewin_hooks.py`, sha256 `c079560e…`, 1,088 lines).

`hook/make_patch.py` inserts the marked lines.
- Every anchor must occur exactly once.
- Pre-hashes are asserted for two bases: B1 (`146a7c6b…` index-K file) and the B1-mapguard image (`30ff8bde…`).
  Both apply.
- Every call is behind `if _FW_ON:`, which is the module's `ON` flag, read once at import.

Line numbers below are in the hooked B1 copies (`hook/hooked-b1/`).

| File | Line | Call | Runs | Purpose |
|---|---|---|---|---|
| `v1/worker/gpu/model_runner.py` | 988 | `register` | `capture_model`, before graph capture | Allocate every persistent table, so the in-graph hook is captured with fixed addresses |
| | 1143 | `add_request` | Request admission | Forced ids, prompt length, draft mode, D2 window, designated positions |
| | 1053 | `remove_request` | Before the request's state is freed | Write `req_<tag>_r<rank>.pt`; window page dump (`local_fw_dumpwin`) |
| | 1465 | `on_batch` | `prepare_attn`, real batches only | Arm the step; per-batch-row tables for the builder and the in-graph counters |
| | 1496 | `on_logits` | `sample`, **before** the (rejection) sampler | Raw metrics of every row; disarm the in-graph counters |
| | 1903 | `prefill_ref` | After `sample`, keyed reference requests only | Metrics at every computed prompt row (prefill reference) |
| | 1950 | `after_commit` | After `postprocess_sampled` | Integer invariants; shadow ledger step; snapshots (validation only) |
| | 1980 | `after_propose` | After the speculator writes `draft_tokens` | Draft-schedule modes |
| `spec_decode/rejection_sampler.py` | 217 | `set_chunk` | Per verification chunk | Logits-row offset |
| | 165 | `on_processed` | In `_verify`, after top_p, **before** `rejection_sample` | Served post-mask probabilities |
| | 290 | `override` | Before `get_num_sampled_and_rejected` | Commit history tokens for forced requests |
| `sample/sampler.py` | 289 | `on_processed(single)` | In `Sampler.sample`, after top_p, **before** `gumbel_sample` | Post-mask probabilities (steps without drafts) |
| | 119 | `override_single` | After `Sampler.sample` | Commit the history token (first token after prefill) |
| `attention/backends/mla/indexer.py` | 643, 651 | `d2_tail_mapping`, `note_tail_mapping` | `KpoolTailMetadataBuilder.build`, before the copy into the persistent buffer | D2 injection; shadow-ledger input |
| `layers/sparse_attn_indexer_kpool.py` | 1112 | `on_selection` | Decode branch, after the final selection (cols :511, col-511 policy), **inside FULL CUDA graphs** | Window exposure counters; full selections at chosen positions |

**Request keys** (`vllm_xargs`):

| Key | Purpose |
|---|---|
| `forcewin_hooks` | The forced ids |
| `local_fw_tag` | Label |
| `local_fw_draft` | Draft mode: `natural`, `oracle`, `poison`, `reject:j` or `sched:seed` |
| `local_fw_d2` | D2 window `[lo, hi)` in output indices |
| `local_fw_desig` | Designated positions for exposure |
| `local_fw_selrec` | Positions whose full selections are recorded |
| `local_fw_ref` | Prefill reference |
| `local_fw_snap` | Snapshot frontiers |
| `local_fw_dumpwin`, `local_fw_dumprange` | Window page dumps |

**Env:**

| Variable | Effect |
|---|---|
| `FORCEWIN=on` | Master switch |
| `_D2=1` | Allows injection for keyed requests |
| `_SHADOW=1` | Ledger, with one small D2H per step while a forced request is in the batch |
| `_SNAP=1` | Validation only; one D2H per step while a snapshot request is in the batch |
| `_WMAX` | Maximum forced tokens per request |
| `_LAYERS`, `_NDESIG`, `_NSEL` | Sizes of the exposure tables |

**Recorded per window position** (`COLS`, 42 float32 columns):
- For each of the four termination ids and the forced token: `lp_pre`, `mass_lt`, `mass_le`, `p_post`, `elig_lo`
  (certainly kept) and `elig_hi` (possibly kept, ties).
- Entropy, argmax, top-1 p, argmax == forced, argmax is a termination id.
- Step, row, rows of the request, accepted count, kind (verify, single or prefill-ref), `post_ok`, and an
  out-of-nucleus flag (the history token is outside top_p).

**Graph safety and cost.**
- Sampling, metadata building and drafting run eagerly, and the hooks there use device ops only.
- `on_selection` runs inside captured graphs. It uses fixed-shape device ops on tables allocated in `register`
  (before capture). Per-step values come from in-place-updated tensors (`bidx`, `xstep`, `xarmed`).
- Lint X3 finds no `.item`, `.tolist`, `.cpu`, `nonzero` or `bool()` in any per-step device function.
- Persistent GPU memory is about 75 MB at the defaults (`max_num_seqs` 16, W 8,448). It is about 45 MB at the
  validation's `WMAX=5120`.
- The transient cost of the metric rows is about 0.2 GB at 32 rows × 154,880 vocab.

**Inertness (test I1).** The same deterministic workload gives byte-identical outputs in three cases: the original
B1 files, the hooked files with the env unset, and the hooked files with the env on and no keyed request. The
workload is the real rejection sampler, the real sampler and the real tail builder.

### 2.2 How committed state advances (unchanged from the proposal, now in code)

Per forced request and step, row 0 is fed the last committed token and rows 1..n-1 the drafts. Row j predicts
output index o_j.

- Draft j+1 is accepted iff it equals the history token F[o_j]. Acceptance is the longest matching prefix, a
  drafts.
- The committed tokens are F[o_0..o_0+a], clamped at the window end.
- The engine's own code then consumes `(sampled, num_sampled)`:
  - `get_num_sampled_and_rejected` → `num_rejected`;
  - `post_update` (num_computed, last token, all_token_ids, total_len);
  - KDA rollback (`postprocess_state(num_sampled)`: the next step reads state slot `num_accepted − 1`);
  - DFlash2 proposal;
  - scheduler accounting.

**Draft modes** override `req_states.draft_tokens` after the proposal and before `set_draft_tokens`, so the
scheduler and the next step see them.

| Mode | Drafts | Accepted per step |
|---|---|---|
| `oracle` | Next history tokens | a = K |
| `poison` | `(F + 1) mod V`, never equal | a = 0: the single-token history advance through the served verify path |
| `reject:j` | The first j−1 equal | a = j−1 |
| `sched:seed` | Pseudo-random | a ∈ 0..K per step |

### 2.3 Measurement ordering: proof

Claim: every value written for output index o is computed from the target distribution conditioned on exactly
`prompt + F[:o]`. It is written once, and it is fixed before any forcing or sampling decision of that step.

1. **The inputs are fixed before the step.** A step's logits rows are produced by the forward from that step's
   inputs: the last committed token and the drafts. The hook never writes logits or inputs. It only reads them.
2. **Raw metrics are taken before the decision.** `on_logits` runs in `GPUModelRunner.sample` after
   `compute_logits` and before `self.sampler(...)` or `self.rejection_sampler(...)`.
3. **Post-mask metrics are taken before the draw.**
   - `on_processed` runs inside `RejectionSampler._verify` between `apply_sampling_params` (top_p) and
     `rejection_sample`, and inside `Sampler.sample` between `apply_top_k_top_p` and `gumbel_sample`.
   - `rejection_sample` does not modify the processed logits it reads (test O1).
4. **The decision is replaced only afterwards.** `override` runs after the real sampler and before
   `get_num_sampled_and_rejected`. It changes only `(sampled, num_sampled)`, which feed the **next** step's inputs.
5. **Only rows on the intended prefix are recorded.** Row j is conditioned on committed history plus drafts 1..j.
   The hook records row j iff j ≤ a. For those rows drafts 1..j equal F[o_0..o_0+j−1], so the conditioning prefix
   is exactly `prompt + F[:o_j]`. Rows after the first mismatch are conditioned on a rejected draft and are
   discarded (sink row).
6. **Each position is recorded once.** A step records a+1 rows (o_0..o_0+a) and commits a+1 tokens. The next step's
   row 0 predicts o_0+a+1. Recorded positions therefore partition the window. The hook counts writes per position
   (`nmeas`), so a second write is detectable.
7. **The code order is checked.** Test O2 asserts, from the installed hooked sources:
   on_logits < sampler; on_processed < rejection_sample / gumbel; override < get_num_sampled_and_rejected;
   sample < postprocess_sampled < after_commit < propose < after_propose < set_draft_tokens.

Tests (`tests/test_forcing.py`, installed B1 sampler code, real `RequestState` and `post_update`, toy model whose
logits depend on the token prefix only):

| Test | What it shows |
|---|---|
| F3 | For every schedule, recorded metrics equal the single-token reference `row_metrics(toy(prompt + F[:o]))` at every o, max \|Δ\| 0 |
| F2 | `nmeas == 1` everywhere |
| O3 | Two request seeds (different sampler draws) give bit-identical buffers |
| N1 (power) | An off-by-one index mutant is detected |
| N2 (power) | An every-row-valid mutant is detected (positions measured more than once) |
| R1 | The prefill reference (chunked, every prompt row) equals the same single-token reference |

### 2.4 What forcing changes relative to serving (unchanged; measured per window)

- **Acceptance.** Acceptance is "draft equals history", not p(draft). Re-tokenized or foreign histories are
  accepted less often, so the reject-and-redo path is exercised more. Per-window accepted-count histograms are
  recorded (`acc_hist`), and `oracle` brackets the low-rejection end.
- **Out-of-nucleus history tokens.** A history token outside top_p is a state serving never reaches. These are
  flagged per position (`out_of_nucleus`).
- **Prefix built by prefill.** The prefix is prefill-built: the window covers about W/4 decode-written pools of
  16k-30k (section 5).

## 3. B: validation (summary; full plan in VALIDATION.md)

**Fixtures.**
- The 24 exact-id traces under 5k tokens (two cross the 4,352 block boundary).
- 8 histories B1 samples itself in-session (exact ids, plus B1's own decode log-probabilities).
- Schedules: natural, oracle, poison, reject:1/2/3 and sched. Windows start unaligned (P = 124-365) and at
  P ≡ 0 mod 8 across the block boundary.

**References.**
- *Single-token history advance:* `poison`, through the served verify path; no draft is ever accepted.
- *Prefill reference at every committed boundary:* `local_fw_ref`, one causal prefill of prompt + history
  recording every row.
- *Committed-state references:* prefill-built requests ending exactly at each snapshot frontier.
- **Why not speculation off.** Speculation-off is **not valid** as a reference on this stack. Without DFlash2,
  `VllmConfig.use_v2_model_runner` falls back to the V1 model runner for the hybrid target
  (`config/vllm.py:630-681`). The hook does not exist there, and the decode path differs (tail ring 4 at K = 0,
  other graph sizes). Forcing V2 with `VLLM_USE_V2_MODEL_RUNNER=1` is a never-served configuration that would need
  its own validation.

**Tolerances.**
- They are relative to an in-session null: B1's own unhooked decode-vs-prefill gap on its own sampled histories.
  Floors come from the historical spec-on x3 measurement on 0.9.1 (|Δlp| q50 0.0011, q90 0.162, q99 0.745; top-1
  0.972; `step2-spec-depth/k3-r2.json`).
- An alignment-power control requires a one-position shift to be grossly worse. Otherwise the check is
  INCONCLUSIVE, not passed.

**Committed state.**
- Exact integers: tokens, positions, total_len, num_computed and last token, after every step and at snapshots.
- KDA recurrent state: the committed slot (`num_accepted − 1`) identified among the K+1 speculative slots
  against a prefill-built state.
- KDA convolution state: the window row.
- MLA records and index-K pools before the frontier.
- Tail ring for the incomplete pool.
- Records past the frontier are permitted scratch.
- The guard (on in this session) checks positions against scheduler truth independently.

**CPU results.** See VALIDATION.md section 8; all suites pass.

**GPU.** `runner/run_hookval.sh` (budget-gated, image pin, frozen hashes, pre-flight, watchdog, cleanup trap, no
`set -e`) with `runner/client_hookval.py`. **Estimate 1.47 occupied GPU-hours; proposed ceiling 2.4** (×1.3 + one
invalid-start allowance); per-attempt wall cap 3,240 s.

## 4. C: decode-window-only D2 fault injection

### 4.1 What is injected

D2 is the tail ring written without positions or ownership. Without `positions` in the metadata, the tail builder
keeps the generic per-group slot output, `bt[req][pos // 8] * 8 + pos % 8`. Only column 0 of the tail table is
ever written, so every token at pos ≥ 8 lands in **null tail block 0** at slot `pos % 8`. All requests then share
one 8-slot ring: stash, seed and pool completion all hit block 0.

The fix (variant U: positions, `own_block != 0`, graph-safe copy) is what B1 runs.

Injection, in `KpoolTailMetadataBuilder.build` before the graph-safe copy into the persistent buffer:

```
out = where(inject_token, generic_slot_mapping, B1_circular_mapping)
inject_token = request carries local_fw_d2 = [lo, hi)  AND  P + lo <= position < P + hi
```

| Kept as B1 | Why |
|---|---|
| Prefill rows, including the request's own prefix remainder chunk | positions < P, so the D7 straddle writer and the tail seed stay correct |
| Every other request | Not keyed |
| Padding rows (PAD) | H2 is kept |
| The persistent-buffer copy | So FULL-graph replays read the injected mapping |
| Positions for the B12X builder and every other correction (D1, D3, D4, D7) | Untouched |

A step that is not armed (dummy runs, capture) gets B1's mapping byte for byte (D1e). The per-request count of
injected tokens is recorded (`d2_injected_tokens`).

**Per request versus a global time switch.** Every window request of the contrast batch carries the key, so the
injection acts like a global switch over the window's decode rows. The difference from a global switch is that
prefill rows and non-window requests are never injected. That is what "D7-correct prefill, every other correction
intact" requires.

### 4.2 Differences from stock D2 (labelled, not hidden)

1. **Window start.** In stock 0.9.1 the prefill seed also collapses onto block 0. With the frozen rule
   P ≡ 0 (mod 8), the first window pool has no prefill-seeded member, so the injection never mixes a corrected seed
   with the old decode mapping.
2. **Padding rows.** Stock padding rows also write block 0. The injection keeps them as PAD (H2). At 8 in flight ×
   4 rows the batch is a capture size, so usually no padding occurs. Padded tokens are logged per step.
3. **Co-scheduled prefill.** A co-scheduled prefill of another request does not touch block 0. In stock it would,
   for pos ≥ 8. In the contrast all prefill happens before the windows; remainder chunks overlap only the first
   steps, inside the 512-token lead-in.
4. **Positions are still passed** to the other builders (B1 behaviour). Without positions, stock B12X uses
   `seq_lens_cpu_upper_bound` for per-token causal lengths, which differs only below 2,176 tokens of context.
   That is irrelevant at 64k+.
5. **Stale block-0 content.** Block 0 keeps content from earlier injected runs, as in stock serving. B1-arm requests
   never read block 0.

### 4.3 Verification and expected corruption pattern

**CPU** (`tests/test_d2_inject.py`, 16/16; real hooked builder, real `_compute_slot_mappings_kernel`, real
`kpool_decode_update_and_maybe_write_cache_batched` and `kpool_compress_and_write_cache`, frozen
`d7/step2/guard/mapguard_hooks.py`):

| Test | What it shows |
|---|---|
| D1a-g | Mapping behaviour as in 4.1 |
| D2 | B1 arm: every committed decode pool equals the compress of the request's own keys, including rejected-draft redo |
| D3 | Injected, 3 in flight: 17 of 21 committed pools wrong. Shadow ledger: every *clean* pool byte-correct, every *corrupt* pool wrong |
| D4 | One injected request alone: 0 corrupted |
| D5 | Frozen guard, injected arm: `tail_w_stash` mism = null-class = every injected token; `idx_w_dec` slots correct. B1 arm: zero |

Expected GPU pattern (frozen in VALIDATION.md H5):

- **Guard.**
  - `tail_w_stash` mismatches, all in the null-block class, equal injected tokens × indexer layers, and only in
    injected phases.
  - Zero `tail_w_seed`, `idx_w_dec`, `idx_w_pf` (including D7 straddle `s_mism` and `d7_fallback`) and position
    violations anywhere.
  - D2 corrupts index-K **content**, not pool slots, so the slot-level guard sees the ring, not the pools.
- **Content.** End-of-window dumps of the decode-written index-K pools are compared with a prefill rebuild of
  prompt + window, per pool. "Gross" means relative L2 > 0.25.

  | Arm | Gross pools |
  |---|---|
  | B1 arm | ≤ 0.1 %. This is the positive control for page addressing |
  | Injected, 8 in flight | > 30 % (CPU emulation of stock at 8 in flight: 78-90 %) |
  | Injected request alone | ≤ 0.1 % |

  Shadow *clean* pools are non-gross (≥ 99 %) and *corrupt* pools gross (≥ 95 %). *Race* pools (a same-step
  collision whose GPU order is unknown) are reported.

### 4.4 Verdict

**It can be implemented and validated cheaply. It is implemented, and validated on CPU.**

- Code: 2 marked call sites in the builder plus about 40 lines in the module. No new image beyond the
  validation image.
- GPU validation: phase V4 of the hook-validation session, about 230 s of work, about 0.13 GPU-hours marginal. No
  extra server start, because injection is per request in the same image.
- D2 is **not** removed globally, and D7 is retained.

**Batch design for coupling.** D2 couples requests, because all injected windows share block 0's ring. So:
- The D2 arm injects **all 8** windows of the fixed batch at once. The B1 arm runs the identical batch uninjected.
- Arms run sequentially, never overlapping, with a drain between runs.
- Pairs share the cached prefix (same `cache_salt`), so the prefix state is identical between arms.
- The run is the unit for the D2 effect. Per-window results stay nested in runs.
- The shadow classes (clean, corrupt, race) and the step log (requests, tokens, drafts, prefill, padding) record
  what coupled each window.

## 5. D: window limitation (exposure at designated positions)

**In-graph counters.** In the decode branch, after the final 511-pool selection, for every indexer layer and every
designated position o of a forced request, the hook writes `[step, n_sel, n_sel_win, n_sel_win_corrupt,
n_sel_win_race, n_win, n_win_corrupt, n_win_race]`:
- `n_sel`: valid selected pools.
- `n_sel_win`: selected pools whose completing position 4i+3 ≥ P, that is, written by the decode writer in this
  window.
- `n_sel_win_corrupt` / `_race`: of those, the shadow class is corrupt or race.
- `n_win`: window pools that exist at that position.
- `n_win_corrupt` / `_race`: of those, corrupt or race.

The record keeps the step id. The last write per position is the valid row: a row whose input token is a rejected
draft is always re-processed later by a valid row at the same position. The analysis joins step ids with the
metric rows.

**Full selections** (`local_fw_selrec`, up to 64 positions per request) store the 511 pool ids. Joined with the
measured per-pool corruption from the end-of-window dump and rebuild, they give the **exact** count of selected
pools that are corrupted. Pool records are never rewritten after commit, so the bytes at the window end are the
bytes at selection time.

**Shadow ledger.** A host-side replay of tail-ring slot writes, with request, position and validity, classifies
each committed pool:
- *clean*: own valid keys;
- *corrupt*: another request's key or a stale or rejected key;
- *race*: same-step collision, outcome undefined on GPU;
- *seed*: a member precedes the window.

It runs in **both** arms (B1 should come out all clean), at one small D2H per step.

**Why this matters.** The fraction of all pools that are decode-written is small (about 2k of 16k-30k). What
matters is whether the selection uses them. Recent pools may be selected disproportionately, so the counts above
are reported per position and layer, not as a fraction of the pool set.

A null window result cannot exclude corruption accumulated over a whole generated history (amendment 21, section 7).

## 6. F: metrics and aggregation (specified before data)

**Positions (frozen at the Stage-0 freeze from `first_contrast.json`).** For each window, the designated set
D = {e} ∪ last 512 ∪ 64-grid after the 512-token lead-in ∪ non-repeating commitment events inside the window
(positions only).

**Primary measure.** `M_close(w, run)` is the mean over D \ {e} of ln P_pre(`</think>` = 154842). These are the
active-reasoning positions, and they include the last 512 before a natural close for near-close windows. The mean
of logs is a geometric mean, appropriate across orders of magnitude.

Reported alongside, never pooled with the primary:
- ln P_pre(`</think>`) at e: the positive control for near-close windows, where `</think>` is the next token.
- The `</think>` eligibility share over D \ {e}: certainly kept (`elig_lo`) and possibly kept (`elig_hi`).
- **Each generation-stop token separately** (154820, 154827, 154829): mean and maximum ln P_pre over D. This is
  premature-stop propensity. **Reasoning close and premature stop are never combined.**
- Forced-token log-likelihood (mean ln P of the history token), entropy, argmax == forced, and argmax is a
  termination id.
- Post-mask probabilities.
- Out-of-nucleus count, the accepted-count histogram, and padded or co-scheduled-prefill steps.

**Aggregation.**
- Window → arm-run: the equal-weight mean over the 8 windows.
- Within-arm variation: the range over the 3 runs, per window and for the arm mean.

**Primary contrast** (amendment 21, section 5): B1 ×3 against B1 + D2-window ×3 on the fixed batch, in **paired
runs**.
- Pairs 1-2 share cache salt A, with the order alternated (B1→D2, D2→B1). Pair 3 uses fresh salt B (B1→D2).
- Every run uses new, distinct, recorded request seeds.
- Pair difference: d_k = mean_w M_close(D2_k) − mean_w M_close(B1_k), for k = 1..3.
- Reported: d_1..d_3, their mean, and per-window differences nested in pairs.
- The run is the unit. Windows within a run are dependent under D2.

**Operational screens (not statistical proof, not equivalence):**

| Screen | Rule |
|---|---|
| R0 repeatability | An arm whose run-to-run range of the per-window M_tail exceeds 0.5 nat on ≥ 4 of 8 windows makes cross-arm readings "not interpretable at this n" |
| Run-level consistency | All three d_k have the same sign, and \|mean d\| exceeds the larger of the two arms' run-to-run ranges of the arm mean |
| Window-level consistency (6-of-8) | A window "differs" if all D2 runs lie outside the range of all B1 runs; the direction is consistent if ≥ 6 of 8 windows differ the same way |

**Reading** (amendment 20 section 5; amendment 21 sections 6-7):
- A difference reproduced by the window injection **nominates** D2's decode-time corruption as able to move
  termination probabilities on these histories. It does not prove D2 explains anything about completion rates.
- No difference narrows the hypothesis for 8k windows at 64k-120k with the measured exposure. It cannot exclude
  history-long accumulation.
- Exposure counts are reported next to every per-window difference.
- No corrupted path is a remedy.
- Fixtures are not comparable across model families. All here are K3.25.

**Validity per run.** These are the H1 checks of VALIDATION.md applied to contrast data:
- ids equal;
- once per position;
- `post_ok`;
- rank equality;
- no hook errors;
- injected-token counts > 0 only in the D2 arm;
- ≥ 90 % of designated positions recorded in steps with 8 decoding requests and no co-scheduled prefill.

**First price** (`runner/cost_hookval.py`):
- One session, two warm salts, 6 windows, end-of-window dumps and 2 rebuilds.
- 2.24 GPU-hours estimate, ceiling 3.4.
- It is re-priced from the validation's measured step time before it is proposed.

## 7. E: fixture accounting (section 8) and the frozen first-contrast list

Receipts: `fixtures/account_fixtures.py` → `accounting.json` (inventory sha256 checked) and `first_contrast.json`.

**Totals.** 901 text histories = 527 natural close + 224 loop + 92 exhaustion + **18 exhaustion_cut120k + 40
error**. This confirms the inventory.

**The 58.** All are **K4 weights** (`K4@a5fee929`); none is K3.25.

| Rows | Source | Detail |
|---|---|---|
| 40 "error" | `eval-matrix-2026-10-04/screens/S7K4p` (v0.7.0 local kpoolring, K4, DFlash2 ×5, 8 in flight, seed 1234, docs 13/79/88/121/127 ×8) | Declared **INVALID: engine crash** |
| of which 36 | | "no stream data"; zero reasoning tokens |
| of which 4 | | "stream ended without a finish reason"; partial text of 925, 33,269, 194,602 and 198,565 tokens, none repeating at the end |
| 18 "exhaustion_cut120k" | `k4-k5-share-screen-2026-10-01/replay-C-k4-default` (v0.8.0 stock, K4, DFlash2 ×3, 8 in flight; docs 79×8, 127×6, 13×2, 88×2; seeds 1234-1241) | The **client** cut every reply at 120,000 tokens (119,987-120,003) |

The 40 error rows are terminated by the server failure. They are neither close, loop nor exhaustion.

The 18 cut rows:
- None was repeating at the cut. 17 never repeated at any checkpoint; 1 was repeating at 88k-96k but not at the
  end.
- End zlib30k is 0.23-0.38, and end rep200 is 0-0.22.
- Under the amendment-20b rule they are **client-censored exhaustions**. They are kept apart from the 92 because
  the censoring mechanism (a client cut) differs from the server's `max_tokens`.

**Inside the 843.** 79 further K4 rows: natural 62, loop 12, exhaustion 5, from S8K4p, empties-k4, replay-C
naturals and replay-K5. They are excluded from K3.25 fixtures for the same reason. K3.25-eligible:
465 natural / 212 loop / 87 exhaustion.

**Frozen first-contrast batch.** 8 windows of W = 8,192 + d forced tokens (d = 0..7, so the prefix P ≡ 0 mod 8),
ending below 131,072, in this order, one batch of 8, identical in every run of both arms:

| # | Fixture | Doc | Source | Class at window | P | End (absolute) |
|---|---|---|---|---|---|---|
| 1 | L1 | 13 | stock 0.7.0 | Near close (ends at the close) | 94,512 | 102,710 |
| 2 | L4 | 88 | stock 0.9.1 s5005 | Near close | 72,360 | 80,553 |
| 3 | M3 | 121 | stock 0.7.0 seed-replay | Near close | 63,488 | 71,682 |
| 4 | M4 | 127 | stock 0.7.0 S7 (2,795-token prompt) | Near close | 83,416 | 91,614 |
| 5 | L3 | 79 | stock 0.9.1 s5004 | Near close | 89,304 | 97,499 |
| 6 | L5 | 170 | pilot r0 | Active, exhaustion | 112,024 | 120,217 |
| 7 | A8 | 79 | stock 0.7.0 | Active, loop pre-onset (onset 184k) | 111,896 | 120,091 |
| 8 | L8 | 79 | stock 0.9.1 s5010 | Active, exhaustion | 111,896 | 120,091 |

- Composition: 6 questions; 4 histories from 0.7.0 and 4 from 0.9.1; 5 near-close and 3 active (non-repeating at
  their windows, per amendment 20b).
- All windows are re-tokenized text: shared counterfactual fixtures, not replays.
- Exact window ids and hashes are fixed at the Stage-0 freeze, by re-tokenizing in a CPU container.

## 8. Blockers and open risks

1. **No GPU** until the GPQA pass ends. The validation session needs its own authorization (proposed ceiling 2.4
   GPU-hours). The contrast is priced again after validation.
2. **Unverified on GPU; gated by the validation, which fails safe.**
   - In-graph capture of `on_selection` (H5h).
   - Snapshot page addressing and the KDA spec-row order. Positive controls H4 and H5b report INVALID rather than
     a false pass.
   - Memory headroom at `gpu_memory_utilization` 0.95 for about 45-75 MB of hook tables plus about 0.2 GB of
     transient rows.
   - The Triton top-p path at ≥ 8 rows. CPU tests used the engine's PyTorch path, which the GPU uses below 8 rows;
     H1f checks the served mask against the bounds on GPU.
3. **CPU-test shims.**
   - UVA buffers become plain CPU tensors.
   - Pinned memory is off.
   - `libdevice.log1p` becomes log(1 + x) in the Triton interpreter. This changes random draws only, never
     measured logits or forced commits.
   - Top-p uses the engine's PyTorch implementation.
   - The model forward is a deterministic toy.
4. **Per-step host syncs.** The shadow ledger (both arms) and snapshots (validation only) add one small D2H per
   step. This changes timing, not computation. Measured step time is reported.
5. **Assumptions the runner checks in server logs.** Adaptive verification and adaptive MTP are off
   (`VLLM_ADAPTIVE_MTP=0` in the recipe), so every step has K drafts. Baseline state rollback is in use.
6. **Race pools.** At 8 in flight most D2 pools may fall in the race class. The shadow ledger then bounds exposure
   (corrupt ≤ measured ≤ corrupt + race). The measured per-pool bytes, joined with the recorded selections, give
   the exact count.

## 9. Files

| Path | What it is |
|---|---|
| `hook/forcewin_hooks.py` | The module |
| `hook/make_patch.py` | Marked-line generator (pre-hash gated) |
| `hook/hooked-b1/`, `hook/hooked-b1-mapguard/` | Hooked copies |
| `hook/patch-report-*.json` | Patch hashes and marker counts |
| `src/b1/`, `src/b1-mapguard/` | Installed files extracted from the images (read-only inputs) |
| `build/` | Validation image: `make_build.sh`, `Dockerfile`, `build.log`, `image-id.txt`, `manifest-*.txt` |
| `tests/` | CPU suites, runner (`run_cpu_tests.sh`, `inner.sh`), `make_synthetic_run.py`; `tests/out/` holds the logs |
| `runner/` | `run_hookval.sh`, `client_hookval.py`, `images.txt`, `cost_hookval.{py,out,json}`, `plan-dryrun/` |
| `analysis/analyze_hookval.py` | Frozen validation analysis (dry-run on synthetic good/bad runs) |
| `fixtures/` | `account_fixtures.py`, `accounting.json`, `first_contrast.json` |
| `VALIDATION.md` | Item B |
| `SHA256SUMS` | Hashes of the above |
