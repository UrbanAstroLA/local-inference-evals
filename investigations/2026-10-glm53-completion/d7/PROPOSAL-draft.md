# Proposal: D7, from confirmed defect to verified B1

**Context.** Amendment 17's live slot-mapping guard on checkpoint B0 found one defect, D7. Every other mapping the MLA and
index-K kernels consumed was clean over the frozen coverage.

- **What D7 is:** the index-K prefill writer (`_kpool_compress_insert`) pools tokens t−3..t of the batch and assumes every
  prefill chunk starts on a 4-token pool boundary. The scheduler does not guarantee that.
- **What happens at an unaligned start:** the pool straddling the start is either never written (first request in the
  step) or filled from the previous request's tokens (a later request in the same step).
- **Attempt 3 counts:** 217 events in 50 requests, identical on both GPUs. 186 pools were unwritten and 31 cross-request.
- **Effect:** on selection, outputs or completion, unmeasured.

Amendment 17 prescribes the next steps for a violation: minimise and reproduce, verify a correction, build B1 = B0 + that
correction only, then propose a behavioural comparison. This proposal covers everything up to the behavioural comparison,
and adds the exposure facts that comparison would need to be worth its cost.

Receipts (public, commit `49ff440`):
- `investigations/2026-10-glm53-completion/b0-mapguard/README.md`: method, coverage, verdict and attribution.
- `b0-mapguard/run-a3/verdict.json` and `run-a3/d7-attribution.txt`: per-rank counts; every record at an unaligned
  chunk start.
- `b0-mapguard/tests/test_kpool_unaligned.py` and `tests/out-postA2/kpool-unaligned.json`: CPU reproduction of both
  cases with B0's own kernel.
- `PLAN.md` amendments 17-17e; `FINDINGS.md` item 4j.

## Facts from zero-GPU checks today (not yet published)

1. **Chunking.** Most unaligned starts come from the 2,045-token prefill chunk (2,048 budget − 3 draft slots). Others
   arise whenever several requests prefill in one step and the step's remaining budget splits a prompt.
   - In attempt 3 such starts fell at offsets as low as 21 tokens.
   - 29 of the 217 events had a start below 2,003 tokens into a block.
2. **GPQA exposure by length.** One GPQA prompt (doc 127, 2,795 tokens) is longer than a chunk; the next longest is
   about 2.5× shorter in characters. Every other prompt fits in one chunk on its own, so exposure for those depends on
   several prefills sharing a step (for example, at the start of a pass). The exact count per pass is not yet known.
3. **Doc 127 outcomes in published records**, descriptive only (one question; seeds reused across engines):

   | Record | Doc 127 empty |
   |---|---|
   | stock 0.7.0 | 0 of 3 |
   | stock 0.9.1 | 2 of 3 |
   | 0.9.1 col-511 arm | 3 of 3 |
   | bundle passes | 1 of 2 |
   | 0.7.0 + kpool fix | 1 of 1 |
   | 0.8.0 + kpool fix | 1 of 1 |
   | k4 on 0.8.0 | 1 of 1 |

   - It is a long, hard question that runs to the cap on several engines, including 0.7.0 with the kpool fix.
   - It contributes 2 of 0.9.1's 16 empties and 0 of 0.7.0's 4.
   - Whether 0.7.0 shares D7 is unknown.
   - **Reading:** even if D7 fully explained doc 127, it could account for only a small part of the 4 vs 16 gap unless
     shared-step exposure turns out to be common.

## Step 0: discovery (zero GPU, about half a day of CPU work)

- **Upstream vLLM:** find how upstream's index-K (kpool) prefill compression handles chunk starts that are not on a pool
  boundary. Look for an existing fix, issue or PR, and for any scheduler-side alignment of prefill chunks. Prefer
  upstream's behaviour as the correction if one exists.
- **0.7.0 and 0.8.0:** read their prefill index-K writer and scheduler budget (0.7.0 runs DFlash2 ×5, DCP2). Run the
  existing unaligned-start CPU test against 0.7.0's kernel under the Triton interpreter. This settles whether D7 differs
  between the engines in the direction of the gap, which matters for the original question.
- **GPQA exposure accounting:**
  - Replay the scheduler's chunking against the recorded request logs (admission times, prompt lengths) of the existing
    GPQA passes, using the allocator simulator already used for amendment 17.
  - Output: per pass, which requests met an unaligned start.
  - Then tabulate exposed questions against empty replies per record, descriptively. Doc 127 is the known case; the
    simulation covers shared-step splits.
  - Simulation accuracy is checked against attempt 3: same workload, measured events.

## Step 1: smallest correction (CPU)

There are two candidates; choose using step 0, with upstream behaviour preferred.

- **(a) Writer-side.** The writer pools only the request's own tokens, and completes the pool straddling an unaligned
  start from that request's earlier raw keys (the 1-3 preceding tokens, which the tail ring appears to hold; to be confirmed in step 0). It fixes both cases
  where they occur and leaves chunking unchanged.
- **(b) Scheduler-side.** Non-final prefill chunk lengths are rounded down to a multiple of the pool size (4). This
  prevents the condition, but changes chunk boundaries for every long prompt and must cover shared-step budget splits.

**Tests** (CPU, B0's real kernels, Triton interpreter):
- the unaligned-start test passes for both cases;
- the aligned control is byte-identical to B0;
- the guard's 28 CPU tests pass;
- D1-D4's existing tests are unchanged.

The minimal reproduction and the patch are written in the same form as D1-D6's, ready to offer upstream if the operator
decides to.

## Step 2: verify B1 in serving (GPU)

- **B1** = B0 + the step-1 correction only: a new image and a recipe fork.
- **Rerun** of amendment 17's guard session on B1, with the identical frozen workload, coverage minimums and verdict
  rules, and the hardened runner.
- **Pass criteria:**
  - CLEAN at every site, including the index-K prefill write, with all minimums met;
  - short greedy probes bitwise identical to B0, because single-chunk prompts must be untouched. Long-prompt outputs are
    expected to differ.
- **Cost:** expected about 0.6 occupied GPU-hours (attempt 3 took 0.57). Ceiling 1.0, including one invalid-run allowance.

## Step 3: mechanism-level effect (GPU, optional, priced after step 0)

Before any behavioural pass, measure whether D7 changes what attention reads.

- **What:** for long-prefix requests, including doc 127's prompt, compare index-K top-k selection on decode rows between
  B0 and B1.
- **Metrics:** how often the affected pool is selected (on B0: a never-written or foreign pool), and how often the
  selected set differs.
- **Cost:** about 0.3-0.5 GPU-hours, reusing the column-511 stage-2 selection capture.
- **Purpose:** decide whether a behavioural comparison could plausibly show anything.

## Step 4: behavioural comparison (not proposed now)

Priced after steps 0-3, under amendment 15's preference for contemporaneous controls on new prespecified seeds. Given
fact 2, a full GPQA pass is likely uninformative about D7 unless shared-step exposure is common.

If one is warranted, the candidate is a targeted repeat screen of the exposed questions (doc 127 first), B0 vs B1, with
distinct seeds and the question as the unit. Its cost is dominated by cap-length replies of up to 327,680 tokens, so it
needs its own costing.

## Outcomes

| Result | Meaning | Next |
|---|---|---|
| Step 0: 0.7.0 shares D7 | not an engine difference; correctness only | steps 1-2; step 3 optional |
| Step 0: 0.7.0 lacks D7 | an engine difference in the gap's direction, with narrow GPQA exposure | steps 1-3, then price step 4 |
| Step 2 CLEAN | B1 becomes the corrected checkpoint; D7 retired for this coverage | step 3 or close |
| Step 2 not clean | minimise again; no B1 | report |

## Asks

1. Approve step 0 and step 1 (zero GPU).
2. Approve step 2: about 0.6 GPU-hours, ceiling 1.0, run only after step 1's tests pass.
3. Step 3 returns with a price after step 0; step 4 is not requested.

Cumulative before this proposal: 48.90 occupied GPU-hours.
