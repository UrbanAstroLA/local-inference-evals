# Draft-slot CPU reproduction (recon, 2026-10-22)

**Question:** can the DFlash2 drafter's slot-mapping kernel write draft K/V into a block the requesting request does
not own? With draft-slot sharing on the 0.9.1 family and B1, a block id names bytes inside an MLA layer's page, so such
a write would corrupt target KV.

**Why only the 0.9.1 family:** rejection sampling keeps the output distribution exact whatever the drafts are. A
draft-slot error can change outputs only by writing into memory the target model reads, which happens only under
sharing. On 0.7.0 (DCP2, sharing off) a draft-slot error could cost acceptance and speed, never correctness.

## What ran
- **Image:** the stock tpurtell 0.9.1 image `a9e985c0`, as a CPU container: `--network none`, no GPU flag, default
  runtime runc, `TRITON_INTERPRET=1`, read-only mounts (`run.sh`).
- **Real code:**
  - the image's `KVCacheManager` / `BlockPool` / sliding-window manager with the served 10-group geometry (draft group:
    block 1,088, window 2,048, page padded to the MLA page);
  - the image's `_prepare_dflash_inputs_kernel`, byte-identical in every release (`482b4511…`).
- **Emulated, stated:**
  - the scheduler loop: FCFS, chunked prefill 2,048, K = 3, lookahead K + 1 as `config/vllm.py:616-620`; async
    optimistic `num_computed` with rollback after the next schedule, and a synchronous variant of every scenario;
  - acceptance draws;
  - the runner's persistent block-table rows: append-only, rows reused across requests, scheduler window frees not
    propagated, as in the runner.
- **Reachable-state source analysis:**
  - the scheduler reserves K + 1 lookahead slots for DFlash, so query rows reaching `last_valid + 1 + K` are allocated
    by design;
  - rejected context rows lie at `last_valid + 1 … last_valid + num_rejected`, inside the query rows' range
    `last_valid + 1 … last_valid + 1 + K`, so the query write of the same step rewrites them, provided the eager
    context write precedes the in-graph query write on the stream.

## Invariants (on every kernel output slot)

| Id | Statement |
|---|---|
| I1 | The slot lies in a block the same request holds in the draft group now, and not block 0 |
| I2 | The block is held by no other request (any group) and by no non-draft group of the same request |
| I3 | Every rejected context row's slot is rewritten by a query row of the same step |
| I4 | Every query position lies inside the request's allocated draft-group horizon |

## Results (`out/result.json`, full length; `out/result-quick.json`, short)

| Scenario (async and sync scheduler) | Steps | Rows | Rejected rows | Rows with window-freed entries | Query rows crossing a block | Prefix-hit requests | Violations | Minimal-patch control |
|---|---:|---:|---:|---:|---:|---:|---|---|
| GPQA-like, 8 in flight, mixed acceptance | 2,119 | 13,567 | 18,473 | 4,292 | 39 | 0 | **0** | valid rows identical 2,119/2,119; 18,473 rejected rows → PAD |
| All drafts accepted (query to lv+1+K) | 1,251 | 5,004 | 0 | 2,212 | 13 | 0 | **0** | identical 1,251/1,251 |
| All drafts rejected | 2,501 | 10,004 | 30,000 | 0 | 24 | 0 | **0** | identical; 30,000 → PAD |
| Multi-chunk prefill (7k) then decode | 947 | 1,877 | 2,475 | 1,873 | 6 | 0 | **0** | identical |
| Shared 9k prefix × 4, staggered | 977 | 3,796 | 5,123 | 3,794 | 7 | 3 | **0** | identical |
| Concurrency 1, block crossings | 3,759 | 3,759 | 5,028 | 1,608 | 8 | 0 | **0** | identical |

**Positive controls (the checker must detect them):**

| Plant | Full run | Quick runs |
|---|---|---|
| Frontier block-table entry replaced by another live request's block | I1 / I2: 18,422 | 3,145-3,183 |
| Missed block-table append (each new block's first append skipped once) | block 0: 15 | 16-47 (probabilistic version: 0 once) |
| Bonus query slot dropped when drafts were rejected (checker-only tamper) | I3: 4,979 | 899 |
| Lookahead K instead of K+1 | **0 (async and sync)** | async runs: 0, 0, 2; sync runs: 1, 1 |

**The lookahead plant is a weak control:**
- It bites only when a query row's last position falls exactly on the first position of an unallocated 1,088-token
  block.
- That depends on prompt-length alignment and, under async scheduling, on the optimistic over-allocation.
- It was detected in 3 of 7 runs.
- So I4 is shown to be able to fire, but its sensitivity to this fault class is low in these scenarios.

## Reading
- **Clean, within coverage.** In the exercised reachable states (async and sync scheduling, window frees, block
  crossings, high block ids, prefix hits, row reuse, 0 to K acceptance), the stock kernel's draft writes stayed in the
  request's own draft blocks. No write touched another request's or group's block, and every rejected row was
  rewritten in the same step.
- **The upstream-style guard changes nothing that matters.** It turns rejected context rows into PAD and leaves every
  valid slot identical. In these states it is a no-op for correctness.
- **Not covered here:**
  - the GPU stream order of the eager context write against the FULL-graph query write;
  - drafter FULL-graph replay with padded rows;
  - the physical page aliasing on each rank;
  - preemption and resume;
  - KV-connector paths.

## Files
`harness.py`, `run.sh`, `out/image-id.txt`, `out/result.json`, `out/result-quick.json`, `out/full.log`,
`out/quick.log`.
