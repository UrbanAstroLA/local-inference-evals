# P4 proposal amendments (local; written before any GPU step; PROPOSAL.md and AUTHORIZATION.txt unchanged)

The approved design is PROPOSAL.md. Building it on the machine's actual records and engine forced the changes
below. Each says what the proposal said, what was found, what the package does and what it costs. The GPU runner
refuses to start unless the operator acknowledges this file: `P4_AMENDMENTS_ACK` must equal the first 12 hex
digits of its sha256 (`sha256sum PROPOSAL-AMENDMENTS.md | cut -c1-12`). Changing this file changes the
acknowledgement.

## A1. Histories: no stock 0.9.1 record keeps generated token ids beyond 5k tokens
**Proposal:** 8 finished-after-100k and 8 capped replies "from local stock 0.9.1 GPQA records with full token ids".

**Found:**
- `fixture-inventory/INVENTORY.md` ("Gaps"): the only exact generated ids on the machine are the 24 step-5/6 traces,
  all under 5k tokens, from modified engines. Nothing since has stored ids: the raw bodies of the two later stock
  0.9.1 passes have `token_ids: null`.
- Those two passes keep the full reply text: `long-reply/d091` (seed 70704) and `long-reply/d091b` (seed 70705). Both
  ran on the stock image a9e985c0 with max_tokens 327,680.
  - Raw bodies: `d091/PRIVATE/raw/s1.raw.jsonl.gz` sha256 `4d3f6313…ca40f`; `d091b/PRIVATE/raw/s1.raw.jsonl.gz`
    sha256 `a07fc7bd…be07a`.
  - Finished with logged completion ≥ 100,000: 4 + 4 = **8, all used**.
  - Capped (finish "length"): 6 + 8 = 14, of which **8 are chosen** by the fixed seed 20261023.

**Closest faithful alternative (implemented):** the reasoning text of these 16 stock 0.9.1 replies is re-tokenized
with the served tokenizer and template, exactly as long-reply B+/D/E built their prefixes.
- The prompt ids equal the server's logged `prompt_tokens` for all 16.
- No stop id and no `</think>` appears inside any reasoning.
- The rule and its checks are in `windows/select_windows.py` and `windows/out/window-manifest.json` (all checks pass).

**Effect:** the forced histories are shared counterfactual histories, not replays of the generated ids.
- The A-vs-B comparison is unaffected: both arms force identical tokens.
- Positions whose history token falls outside the top_p 0.95 nucleus are flagged per position (forcewin
  `out_of_nucleus`) and reported.

## A2. The capped replies' cap is 327,680, not about 131k
**Proposal:** near-close window for a capped reply = "the 512 tokens before the cap". Its cost line assumes about 2.1M
prefill tokens per session, which implies a cap near 131k.

**Found:** the stock 0.9.1 passes with text capped at 327,680. Taken literally, each capped window needs a 327k-token
prefix:
- 1.58M more prefill tokens per session;
- 1.64 GPU-h per session at the planning rates;
- three sessions plus the checks come to about 5.0 GPU-h, over the 4.0 ceiling.
- A 327k prefix also nearly fills the measured prefix-cache horizon (about 0.66M tokens) by itself, so the in-session
  repeat would re-prefill it.

**Implemented:** a capped reply's near-close window is the 512 tokens **before output index 131,072**. This is the
historical stock record cap, which these replies passed while still unfinished. It is deep in the unfinished regime
and at the depth where the finished replies close (106k-146k logged).
- Planning cost: 1.27 GPU-h per session.

**Alternatives for the operator (not built):**
- (a) The literal window with sessions A and B only: about 3.3 GPU-h. It loses the A' session control.
- (b) The literal window for 4 of the 8 capped replies and 131,072 for the rest: about 1.45 GPU-h per session, 4.4 in
  total. Over the ceiling.

## A3. Window positions aligned to the prefix-cache block
**Proposal:** early window "at P ≈ 16k", deep "at P ≈ 96k".

**Implemented:** P (prompt + history tokens before the window) is the value with P ≡ 1 (mod 4,352) nearest to
prompt + 16,000 and prompt + 96,000, which is within ±2,176 tokens.

**Why:** on this engine a request whose prompt is cached gets back only ⌊(P−1)/4352⌋ − 1 full blocks. The
EAGLE-type drop of one block was measured in `first-contrast/diagnosis-fc1`. The aligned P makes the uncached
remainder exactly 4,353 tokens, its minimum. Near-close windows keep their fixed positions, with remainders of
4,353-8,704 tokens.

## A4. How the prefix is built, and the start of each batch
- **Prefix building.** Each batch of 8 windows first sends one unkeyed warm request per reply (max_tokens 1, its
  longest prefix in the batch). The forced requests then hit the prefix cache, which keeps the KDA state at every
  block boundary (align mode).
- **Batch composition.** Batches are composed reply by reply so that a batch needs at most 3 warm prefixes (about
  0.4M tokens). That is under the measured prefix-cache horizon of about 0.66M tokens. Batches 1-5 hold
  3 + 3 + 2 windows of 3 replies; batch 6 holds the last reply's 3 windows plus the 5 remaining early windows.
- **Mixed steps at the start of each batch.** Each forced request still prefills its remainder (4,353-8,704
  tokens) in-batch, in 2,048-token chunks. Requests that finish this prefill first decode while the others prefill,
  so the first steps of a batch are mixed prefill + verify steps. Their MoE m is above 32, so both arms use the
  prefill plan in those steps.
  - Expected share: a few percent of positions (about 20-60 per window at the start).
  - Every position carries its step key: host step facts plus the composition of forced requests. A position enters
    a comparison only if its keys match. Positions in mixed steps are kept, matched, and reported with their step
    shape.
  - The 8 requests of a pass are released together from a barrier, so arrival order is the same in every session.
- **Repeats.** Repeat 2 of a batch reuses the cache built by repeat 1 under the same rule.

## A5. The acceptance schedule `sched`
**Proposal:** "the accepted counts of a recorded natural run".

**Implemented:** the per-step accepted counts of the forcewin natural-mode runs of hook-validation attempt 2. These
are the only recorded natural forced runs with per-step counts: B1 image, DFlash2 ×3, exact-id histories. The 8
records are concatenated in tag order and the first 1,024 steps are kept.
- Mean 3.06 tokens per step; histogram a = 0/1/2/3: 193/133/118/580 (`windows/out/sched-summary.json`).
- The list itself is in `PRIVATE-fixtures/sched.json`.

**Mechanism:** a new forcewin draft mode, `list`. Draft k of verify step s is the history token for k ≤ list[s] and
a poison token otherwise. All 8 requests of a batch share the list, so they advance in lock step, and both arms see
the same accepted counts step for step (E5: acceptance fixed by design; checked).

## A6. Endpoint definitions the proposal left open
- **E1, "MoE output after the TP all-reduce":** the output of the model's MoE block (`Glm5NextMoE`, routed plus
  shared expert, after vLLM's final all-reduce) at layers 3, 23 and 44, taken on rank 0. The shared expert is
  identical in both arms.
- **E2, hidden state at layers 12 and 34:** the model keeps 4 mHC residual streams. The recorded state is the
  contracted post-layer hidden state, `hc_contract(hc_post(...))`, the same read-only materialisation the DFlash tap
  uses. The final-norm state is the input of `compute_logits`.
- **E4, "full fp32 logits":** the logits tensor the sampler receives, cast to fp32 (exact), staged before the sampler
  runs.
- **Sampler of the forced requests:** T 1.0, top_p 0.95 (the GPQA sampler; it defines the recorded post-mask
  probabilities and nucleus eligibility). Seeds are fixed per window and repeat, identical in every session.
- **"A = A" and "differs reproducibly":** defined bitwise in `analysis/analyze_p4.py`, with a distributional fallback
  if A is not bit-stable, fixed before data:
  - A-stable: both repeats of A and both repeats of A' are equal.
  - A difference counts when it appears in both repeats of B.

## A7. Session caps and coverage
- Each session's wall cap is its share of the ceiling: 2,350 s, or 1.306 GPU-h of the 4.0, after 0.067 GPU-h for the
  single-GPU checks.
- Planning estimate 2,289 s; typical about 2,020 s (`runner/cost_p4.json`).
- The client skips a batch it cannot finish before the deadline. Batches run in the same order in every session, so a
  short session loses the same trailing batches in every arm. The analysis reads the intersection and reports it.

## A8. Additions not in the proposal (cheap; reported, not gating)
- **G4b:** the G4 replay is run a second time with `GLM53_P4_TILE` set. It shows that the switch reaches the bound
  decode launch and gives the prefill-plan arithmetic on captured rows. This is the env-driven equivalent of
  captured-p3b's forced-tile confirmation.
- **Single-GPU pre-check** (proposal section 7): graph capture with every P4 hook, a negative control, and bit-exact
  stores.
