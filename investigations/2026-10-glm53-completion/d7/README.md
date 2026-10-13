# D7: mechanism, exposure, correction and serving verification (amendments 18-18c)

D7 is the index-K prefill writer's assumption that every prefill chunk starts on a 4-token pool boundary
([found on checkpoint B0](../b0-mapguard/README.md)). This folder holds four things:

- the comparison across releases and with upstream;
- the GPQA exposure accounting;
- a writer-side correction;
- its verification in one guarded serving session of **B1 = B0 + the D7 correction only**.

**Result:**
- **B1 verified, CLEAN.** At 219 unaligned chunk starts every straddling pool was written and byte-identical to an
  independent recompute; every other mapping was clean; every coverage minimum was met. It took 0.6244 occupied GPU-hours
  in one attempt.
- **Same writer everywhere.** The defect's writer is byte-identical in tpurtell 0.7.0, 0.8.0 and 0.9.1, and upstream vLLM
  main keeps the same assumption.
- **GPQA exposure is narrow and runs opposite to the 0.7.0-vs-0.9.1 completion gap.** This is a reconstruction, not a
  measurement.
- **Not covered:** completion impact is untested, and no GPQA pass was run.

Evidence labels used in the reports: [M] measured by running code; [R] reconstructed, deterministic from measured inputs
through arithmetic verified on a measured run; [E] estimated across plausible schedules; [S] read from source; [I]
inferred.

## The defect and how it differs by release ([`mechanism/REPORT.md`](mechanism/REPORT.md))

- **What the writer does.** `_kpool_compress_insert` pools batch tokens t−3..t and writes a pool only when token t
  completes it.
- **What goes wrong.** The scheduler does not align prefill chunks to 4 tokens. At an unaligned start the pool that
  straddles it is either never written (first prefill row of the step) or compressed from the previous request's tokens
  (a later row).
- **It reaches every release.** The writer is byte-identical in 0.7.0, 0.8.0 and 0.9.1 [M]. Each release's served
  configuration reaches unaligned starts [M: each image's own split function plus the budget arithmetic].
- **The pattern still differs by release:**

| | 0.7.0 | 0.8.0 / 0.9.1 / B0 |
|---|---|---|
| Draft slots K; solo prefill chunk | 5; 2,043 tokens | 3; 2,045 tokens |
| Chunk with D decode rows in the step | 2,043 − 11·D | 2,045 − 7·D |
| Attention block | 3,584 | 4,352 |
| Affected pools per long prompt | about 1 per block | about 2 per block |
| Where an event lands | DCP2: the straddling pool's owner rank, all 11 MLA layers | DCP1 with MLA owners: every MLA layer, on its owner GPU |

- **Upstream.** vLLM main (`c5422f4c46`) keeps the same writer and its "Assumes pool-aligned chunk starts" docstring. Its
  scheduler has no pool alignment, and the open kpool rework PR #57161 keeps the assumption. No issue or PR about it was
  found. SGLang main repairs it writer-side, reading the straddling pool's earlier members from the request's tail ring
  [S].

## GPQA exposure ([`exposure/REPORT.md`](exposure/REPORT.md))

- **Simulator gap.** The allocator simulation used before the B0 session predicted 73 unaligned starts against 217
  measured, because it omitted the scheduler's per-request draft-slot reservation. With the reservation, a rebuilt
  simulator reproduces all 14,712 scheduled rows, 481 prefill chunks and 217 events of that session exactly [R, given the
  recorded step timing, arrival order and decode counts]. From the workload alone it predicts 224-227 [E].
- **Exact prompt tokens.** Every logged GPQA request (1,980) was re-rendered with the model's tokenizer and template, and
  every count equals the server's [M]. Only doc 127 (2,795 tokens) is longer than a chunk; the next longest is 887.
- **0.9.1 family** (stock, column-511 arm, bundle; 8 logged passes): **no exposure** [R/E].
  - Doc 127 is always split, but at 2,045 − 7·7 = 1,996, a multiple of 4. In all logged passes exactly 7 requests were in
    flight at its admission, with none starting or ending within ±1 s; this decode count is reconstructed from proxy
    timestamps.
  - No other prompt can be split under any plausible schedule.
- **Stock 0.7.0:**
  - Passes 2-3: doc 127 is exposed (split at 2,043 − 11·7 = 1,966) [R, conditional on 0.7.0's chunking read from source].
    It finished in both passes.
  - Pass 1 has no request log.
- **Server logs** of the stock records confirm the 2,048-token budget and the draft counts (3 for 0.9.1, 5 for 0.7.0).
- **Descriptive only.** The exposed question was exposed on the engine where it finished. No test was run and no causal
  reading is drawn. The historical decode count stays a reconstruction; phase 2d below measures the arithmetic, not the
  historical passes.

## The correction ([`mechanism/d7-fix.diff`](mechanism/d7-fix.diff))

**One file, `sparse_attn_indexer_kpool.py`.** A pool member is taken from the batch only when it belongs to the same
request at the expected logical position. Otherwise it is read from that request's tail ring:
- the ring is written with positions since the D2 correction;
- it has one per-request block that is never shared or prefix-cached;
- its seeds and stashes were verified in the B0 session.

The ring is read before the existing seed overwrites it. Rows without a ring keep the old rule, and that fallback is
unreachable in B1's configuration [S/M]. There is no host synchronisation.

**CPU tests** (`mechanism/tests/`, under the Triton interpreter with B0's real call site; every pool compared with an
independent plain-torch reference): 28/28.
- They cover every residue modulo 4, first and later rows of a step, chunks of 1-4 tokens, block boundaries, a prefix
  hit followed by an unaligned start, preemption, and random shared-step schedules.
- Unpatched B0 fails exactly the unaligned cases, and aligned inputs are byte-identical to B0.
- D1-D4 regressions and the guard's CPU suite pass on the patched code.
- The interpreter rounds differently from the GPU (bf16 truncation, one E4M3 carry), so CPU bytes are not the GPU oracle.

## Serving verification ([`step2/`](step2), rules in [`step2/FROZEN.md`](step2/FROZEN.md))

- **Images.** B1 is `sha256:799d3cf8…`; a full filesystem diff against B0 shows only the corrected file and its bytecode
  changed. The guard image is `sha256:4d7bd73e…`.
- **Guard amendment.** The index-K prefill check models B1's rule from the writer's actual ring status.
- **Independent oracle.** The guard keeps its own shadow of each request's last ≤3 keys and gates by layer, request and
  ledger position, never the tail ring that B1 reads. It recomputes each straddling pool with the engine's own compress
  kernel into guard scratch, so GPU arithmetic is used by construction.
- **Positive control.** Every aligned pool is recomputed the same way and must match exactly. A failure would make the
  byte checks INVALID, not violations.
- **CPU fault injection detected all four classes:** missing repaired pools, foreign keys, stale keys from the same
  request, and wrong logical positions. An unfaithful oracle turned byte checks INVALID, as designed.
- **Suites before freezing:** new guard tests 30/30, the existing 28-test suite 28/28, the earlier regression, smoke tests
  and a pipeline dry run.

**Session `run-b1-a1`:** the amendment-17 workload unchanged, both GPUs, shipped CUDA graphs, DFlash2 ×3. Readiness took
732 s; 1,124 s from start to GPU release; 0.6244 occupied GPU-hours of the 2.0 authorized (amendment 18b).

| Criterion | Result |
|---|---|
| All 28 coverage minimums | met (e.g. 1,993 graph-replay steps, 129 mixed steps, 32 prefix hits, 861 decode steps at 5-8 in flight, accesses above 2³¹ bytes on every site) |
| Straddling pools exercised (≥ 100) | 219 events, on every owned layer: 1,314 checks on rank 0 and 1,095 on rank 1 |
| Missing / byte-mismatched / fallback | 0 / 0 / 0 |
| Positive control (≥ 10,000 per rank) | 833,736 and 694,780 aligned pools, 0 mismatches |
| All other mapping sites | 0 violations |
| Short single-chunk greedy probes | identical to B0's |
| **Verdict** | **CLEAN** |

The guarded prefix phases (warm, T, F, D, S) ran 3-7% slower than in the B0 session; the GPQA phase ran faster. Phase times are in the client logs.

**Phase 2d** (after the frozen workload, outside the verdict): doc 127 was admitted behind D = 0-7 decoding requests.
- In every case the actual scheduled D equalled the requested D, and the first chunk was exactly 2,045 − 7·D.
- The split is pool-aligned only at D = 3 and 7 ([`step2/run-b1-a1/phase2d.json`](step2/run-b1-a1/phase2d.json)).
- This measures the scheduling arithmetic under matching conditions. It does not measure the historical passes.

## Not covered

- completion or accuracy impact (no GPQA pass);
- the localized selection effect (step 3, not requested);
- external-token and KV-connector paths (not configured);
- contention-dependent races that one run cannot exclude.

The guard session's workload reuses 8 seeds between phases T1 and D, as the B0 session did (amendment 18c); that does not
affect the mapping checks.

## Files

| Path | Contents |
|---|---|
| `PROPOSAL-draft.md`, `PROPOSAL-step2.md` | the two proposals as reviewed (the first with a character-length comparison later replaced; the second with a correction appended) |
| `mechanism/` | release and upstream comparison, the correction, CPU fix tests, regressions, a copy of the guard suite run against the patch |
| `exposure/` | simulator reconstruction, tokenizer re-implementation, exposure classes, tables |
| `step2/` | frozen rules, guard, analysis, runner, client, CPU tests, build files, the session's verdict, phase 2d, gzip guard records |
| `RAW-SHA256SUMS-d7.txt` | hashes of local files not published (fixture token ids, probe outputs, container logs, pre-flight listings, full engine files) |
