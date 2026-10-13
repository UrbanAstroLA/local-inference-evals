# D7: Step 0-1 results and a Step 2 proposal (verify B1 in serving)

Under amendment 18. Steps 0 and 1 are complete, CPU only, with no GPU use. Receipts are local and unpublished:
`d7/mechanism/REPORT.md` and `d7/exposure/REPORT.md`, each with a SHA256SUMS. They can be pushed for independent review on
request.

Evidence labels: [M] measured by running code; [R] reconstructed, deterministic from measured inputs through arithmetic
verified on a measured run; [S] read from source; [I] inferred.

## Step 0a: release comparison

| Item | stock 0.7.0 | stock 0.8.0 | stock 0.9.1 / B0 |
|---|---|---|---|
| Faulty assumption in the actual prefill writer | **yes**: `_kpool_compress_insert` byte-identical [M] | **yes**, identical [M] | **yes**, identical; B0 adds only the col-511 helpers [M] |
| Execution can reach an unaligned start | **yes**: K=5, so a solo chunk is 2,043 (residue 3); 2,043 − 11·D with D decode rows; shared steps give any residue [M: the image's own split function plus budget arithmetic] | **yes**: 2,045 − 7·D [M] | **yes**: 2,045 − 7·D; matches attempt 3's measured starts [M] |
| Affected pools by order, chunking and rank | first prefill row: pool unwritten; later row: filled from the previous request's tokens [M]; about 1 per 3,584-token block; DCP2: only the pool's owner rank holds it, so each event hits one rank, all 11 MLA layers [M] | same outcomes [M]; about 2 per 4,352-token block; DCP1 with MLA owners: every MLA layer, each on its owner GPU [S] | same as 0.8.0; attempt 3: 217 events per layer set [M] |

**Reading.** This is a shared source defect. The same writer is in every release and every served configuration can reach
it. Exposure still differs by release, through the draft slots (K), the attention block size and DCP. Whether those
pattern differences matter for selection is unmeasured.

## Step 0b: upstream comparison

- **vLLM main** (`c5422f4c46`): the writer is unchanged, including its "Assumes pool-aligned chunk starts" docstring. The
  scheduler has no pool alignment. The open kpool rework PR #57161 keeps the assumption. No issue or PR about this defect
  was found [S].
- **SGLang main:** repairs it writer-side. A pool straddling a chunk start reads its first `start % kpool` members from the
  request's tail ring (`kpool_plan.py:152-210`, `kpool_fp8_index.py:1217-1264`) [S].
- **Compatibility with B0** [S + M]:
  - B0's tail ring holds each request's last 4 prefill tokens at `pos % ring` in one per-request block that is never
    shared and never prefix-cached.
  - Ring slots carry positions since the D2 fix, and attempt 3 verified ring seeds and stashes in serving (2.78 M seed
    writes, 0 violations).
  - Prefix hits are block-aligned; preemption resumes at an aligned hit or at 0; drafts do not exist during prefill.
  - So the 1-3 preceding keys and gates are available with verified ownership and positions, which is amendment 18's
    condition for a writer-side repair.

## Step 0c: GPQA exposure

**Simulator gap.** The pre-run simulator omitted the scheduler's per-request draft-slot reservation. That reservation is
what produces the 2,045-token chunks. With it, the rebuilt simulator reproduces attempt 3 as follows:

| Comparison | Agreement |
|---|---|
| Scheduled rows | 14,712 / 14,712 exact [R] |
| Unaligned-start events (step, start, length) | 217 / 217 exact [R] |
| Prefill chunks | 481 / 481 exact [R] |
| Workload-only prediction | 224-227 events [E] |

Those exact matches are conditional on attempt 3's recorded step timing, arrival order and decode counts.

**Exact prompt tokens.** All 1,980 logged GPQA requests were re-tokenized with the model's tokenizer and template, and
every count equals the server's [M]. This replaces my earlier character-length comparison.

**Exposure by record:**

| Record | Exposure |
|---|---|
| 0.9.1 family (stock, column-511 arm, bundle; 8 logged passes) | **0 events [R/E]**. 197 prompts per pass cannot be split under any plausible schedule. Doc 127 (2,795 tokens) is always split, but at 1,996 = 2,045 − 7·7: at its admission exactly 7 requests were in flight in all 10 logged passes, so the split is pool-aligned. |
| stock 0.7.0, passes 2-3 | doc 127 **known exposed** (split at 1,966 = 2,043 − 11·7, residue 2) [R, conditional on 0.7.0 chunking from source]. It finished in both passes. |
| stock 0.7.0, pass 1 | no request log, so 116 requests are UNKNOWN. Of those, 115 cannot be split under the measured ceiling of 2 co-sends. Doc 127 is exposed unless D ∈ {1, 5}. |
| all records | **no POSSIBLE exposure**: shared-step exposure is nil. |

**Correction to my earlier proposal.** I wrote that doc 127 is "always exposed" because it is longer than a chunk. It is
always split, but on 0.9.1 the split falls on a pool boundary.

**Limits.**
- The 0.9.1-family result is a reconstruction, not a measurement.
- Its weakest link is the 7 decode rows at doc 127's admission, inferred from proxy timestamps (no request starts or ends
  within ±1 s). With 6, the split would be at 2,003, which is exposed.
- No server log was found for the stock 0.9.1 record. The column-511 arm and bundle server logs confirm a 2,048-token
  budget and 3 draft tokens.

**Reading under amendment 18's outcome rules.** GPQA exposure is narrow, and it runs opposite to the gap: the exposed
question was exposed on 0.7.0, where it finished. D7 is a correctness defect, not supported as a contributor to the 4 vs
16 gap. No GPQA pass is proposed.

## Step 1: chosen correction (writer-side)

**`d7-fix.diff`**: one file (`sparse_attn_indexer_kpool.py`), 114 lines.
- `_kpool_compress_insert` takes positions and the tail-ring cache and slot mapping.
- A pool member is taken from the batch only when it belongs to the same request (same tail block, position `pos − j`).
  Otherwise it is read from that request's ring at `pos % ring`.
- The ring is read before the existing seed overwrites it.
- Rows or calls without a ring keep B0's rule unchanged.
- There is no host sync.
- **Interpretation:** the writer is made correct for arbitrary unaligned inputs. Exposure is not prevented.

**Tests** (CPU, B0's real call site and kernels under the Triton interpreter; every pool compared with an independent
plain-torch reference of the request's own tokens by logical position):

| Group | B1 (patched) | B0 (unpatched) |
|---|---|---|
| Residues 0-3: first request; later request in a packed batch; both unaligned in either order | ok | r1-r3 unwritten or wrong |
| Chunks of 1-4 tokens; successive boundaries; served 9,000-token chunking; starts 4,349-4,355 across the block boundary | ok | unwritten |
| Prefix hit at 4,352 on a reused stale ring block, then an unaligned continuation; preemption and resume | ok | unwritten |
| Decode rows ahead of the prefill slice; random shared-step schedules (53 unaligned starts, 20,799 pools) | 0 bad | 53 bad |
| Aligned inputs against B0's verbatim function; op-level cache and ring hashes over 4 steps | byte-identical | — |
| **Total** | **28/28** | **defect reproduced as expected** |

**Regressions against the patched code:**
- D1 col-511 checks: pass
- D2 tail fix: 17/17
- D2 scenarios: 6/6
- D3: pass
- D4: pass
- D5/D6: still reproduced, as expected
- earlier observers and check tests: 0 failures
- guard CPU suite: 28/28
- a2 regression: ok
- image smoke tests: pass

**CPU limit [M].** The images' Triton interpreter truncates fp32→bf16 and mis-rounds one E4M3 carry, so CPU bytes are
not GPU-faithful. The reference applies the interpreter's conversions (0 of 6,000 pools differ). GPU bytes are verified
only in Step 2.

**Not covered:**
- decode-classified short prefill chunks of ≤ 4 tokens (the existing ring decode kernel, covered by the D2 tests);
- KV-connector external-token starts (not configured).

## Step 2 proposal: verify B1 in serving

**2a. Guard amendment (CPU, before any GPU use).** As frozen, the guard's index-K prefill predicate encodes B0's rule
(`pred = loc >= kpool−1`). On B1 it would flag every repaired pool.
- **Proposed change:** model B1's rule. An unaligned start's straddling pool is expected written, from the request's own
  ring members plus own batch rows.
- **Byte-level check:** add one for straddling pools. Recompute each from the request's own keys by logical position
  (gathered at the point of consumption), compare it with the stored record, and record mismatch counts.
- **Unchanged:** every other site, invariant, coverage minimum and verdict rule.
- **Before freezing:**
  - CPU fault-injection on B1: a repaired pool removed, a wrong-request ring member, and a stale ring position must each
    be flagged;
  - the full 28-test suite and the a2 regression;
  - new hashes recorded before any GPU data.
- **Estimate:** about 1-2 hours of CPU work.

**2b. Images.** B1 is B0 plus `d7-fix.diff` only, with its own image id and a recipe fork marked `draft-untested`.
The guard image is B1 plus the amended guard.

**2c. Session.**
- **Workload:** amendment 17's frozen workload, unchanged, on both ranks with shipped graphs, DFlash2 ×3 and the hardened
  runner.
- **Pre-registered pass criteria:**
  - every applicable coverage minimum met;
  - unaligned starts exercised, with at least 100 straddling-pool events (attempt 3 had 217);
  - zero missing, foreign or cross-request pools, and zero byte mismatches on straddling pools;
  - correct ownership and position at every site;
  - D1-D4 gates unchanged;
  - the four short single-chunk greedy probes bitwise identical to B0's attempt-3 record.
- **Recorded:** throughput per phase against attempt 3, and instrumentation overhead.
- **Not expected to match:** long-prompt outputs, by design.

**2d. Optional appended phase:** the doc-127 chunk-boundary measurement. It runs after the frozen phases, is labelled
separately, and is outside every coverage count and verdict.
- **What:** admit the 2,795-token doc-127 prompt behind D = 0-7 decoding requests, made from the existing fixture
  prefixes, with max_tokens bounded. Record the measured chunk start each time.
- **Prediction (registered):** 2,045 − 7·D, so aligned only at D = 3 and 7.
- **Why:** it turns the 0.9.1-family "zero exposure" reconstruction into a measurement of the scheduling arithmetic under
  matching conditions. It cannot measure the historical passes themselves.
- **Cost:** about 3-5 minutes inside the same session; no extra startup.
- **Text:** none of the prompt's text leaves the machine. Token counts and offsets only.

**Cost.** About 0.6-0.7 occupied GPU-hours with 2d, from attempt 3's 0.57. Ceiling 1.0, including startup, warm-up,
shutdown and one invalid-run allowance. There is no automatic retry; an invalid attempt is diagnosed and returned to the
operator.

## Outcomes

| Result | Meaning | Next |
|---|---|---|
| CLEAN with exposure exercised | the D7 repair verified in serving; B1 becomes the correctness checkpoint | completion impact stays untested; no GPQA pass |
| Violation at the straddling pool | the repair is incomplete on GPU | minimise; no B1 |
| Violation elsewhere | a new finding | handled as amendment 17's outcomes |
| Exposure not exercised | INCOMPLETE; does not verify the repair | price only the missing exposure |
| 2d confirms 2,045 − 7·D | the GPQA reconstruction's arithmetic is measured | — |
| 2d contradicts it | the exposure classes are revised | GPQA relevance reassessed |

**Step 3** (localized selection effect) is not requested. Given narrow GPQA exposure it is unlikely to bear on the gap;
it would come back only with a separate costed case.

**Upstream.** The defect is present in upstream vLLM main, and SGLang shows a writer-side repair. A minimal reproduction
and patch are prepared in the D1-D6 form. Offering them upstream is the operator's separate decision.

## Asks

1. Approve 2a and 2b (CPU only).
2. Approve 2c, with or without 2d: ceiling 1.0 occupied GPU-hours, run only after 2a's tests pass and its hashes are
   frozen.

Cumulative so far: 48.90 occupied GPU-hours, and nothing spent since amendment 17.

---
**Correction (2026-10-13, after review):** "No server log was found for the stock 0.9.1 record" is wrong. Its server
logs exist (`gpqa/…k3.25@tpurtell+v0.9.1.server.log` for pass 1, `…v0.9.1+ps.server.log` for passes 2-3) and show a
2,048-token budget and 3 draft tokens. Recorded in amendment 18a.
