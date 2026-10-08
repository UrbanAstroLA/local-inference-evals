# GLM-5.3-Flash: looping, exhaustion and empty answers across tpurtell's runtimes (2026-09 to 2026-10)

**Question.** Why does GLM-5.3-Flash fail to finish long reasoning more often on tpurtell 0.8.0 and 0.9.0 than on
0.7.0? Which runtime defects affect inference quality on the way?

Local GPQA Diamond scores (84.7-86.2% over five full runs) stay below the published 92.1 (NVIDIA) and 90.6 (Red Hat).
These runs cannot reach parity, because the weights and harness differ (`FINDINGS.md`, inference 6). Runtime defects
still matter in their own right: they change what the model computes.

## Glossary
- **Loop / exhaustion:** a request that does not finish within the 327,680-token budget. A loop repeats itself
  (compressed tail below 15% of its raw size, or stopped early by the screen's loop detector). An exhaustion keeps
  producing varied text until the budget runs out. On GPQA, either one gives an **empty answer**, which is scored wrong.
- **TP2 / EP2 experts:** routed experts are either split across both GPUs (tensor parallel, TP2) or placed whole on
  one GPU each (expert parallel, EP2).
- **DCP1 / DCP2:** decode context parallelism.
  - DCP2 splits each MLA layer's KV cache across both GPUs by token.
  - DCP1 does not split it. tpurtell 0.8.0 and 0.9.0 combine DCP1 with **MLA layer ownership**: each MLA layer runs
    on one GPU (layers below 25 on the first GPU, the rest on the second) and keeps its KV cache there.
- **kpool tail:** for each decode step the sparse-attention indexer selects up to 2,048 earlier tokens in pools of 4.
  The newest pool is still incomplete; it holds the current token and up to two before it. That pool is the tail.

## Configurations
All use 3.25bpw weights at revision 0490d2f7, the tpurtell v0.9.0 image (digest in the configs; the tail fix as an overlay on
it), DFlash2 with 3 draft
tokens, 8 concurrent requests and `gpu_memory_utilization` 0.95.

| Label | What it is |
|---|---|
| `3.25bpw · tpurtell 0.9.0 · DFlash2 ×3` | tpurtell 0.9.0 as released: DCP1 + MLA layer ownership, TP2 experts |
| `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix · DFlash2 ×3` | The same, with the fix proposed in [tpurtell/glm-5.3-flash-ext3-2x-rtx#6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6) |
| `3.25bpw · tpurtell 0.9.0 · 0.7.0 layout (DCP2, EP2) · DFlash2 ×3` | The 0.9.0 image run with v0.7.0's parallel layout. **A diagnostic control** to separate the tail bug from the rest of the 0.7.0 → 0.8.0 layout change, not a recommended configuration |

### Why 0.9.0 uses DCP1 + MLA layer ownership
The layout is tpurtell's design choice. It frees KV memory: with the same weights and memory setting, the server
reported a KV cache of 4,707,515 tokens on every start with the default layout: two as released, two with the fix.
With v0.7.0's layout the two starts reported 3,165,056 and 3,140,895 tokens. The default layout holds about 1.5 times as many KV tokens.

The masking path described below already existed in the vendored attention code. Only the DCP1 layout exercises it.
With the fix, the design works as intended.

## How it was measured
- **GPQA Diamond** (`protocols/gpqa-diamond/v1.md`), 3 passes of 198 questions. Every request carried seed 1234;
  passes differ through batching nondeterminism.
- **Hard-question screen** (`protocols/hard-prompt-screen/v1.md`): the 5 GPQA questions that most often came back
  empty, × 8 repeats = 40 requests per screen, 8 concurrent, with an early loop stop. The questions were chosen from
  tpurtell 0.8.0's empty answers.
- **Kpool kernel tests** (`protocols/kpool-kernel-tests/v1.md`) and the **kpool tail index check**
  (`protocols/kpool-tail-index/v1.md`): which tokens a decode step attends, using the image's own kernels.
- **Decode vs prefill** (`protocols/decode-prefill-consistency/v1.md`): does decoding a token give the same
  distribution as re-reading it in prefill?
- **Serving probe** (`protocols/serving-probe/v1.md`) and **tool-eval-bench** (`protocols/tool-eval-bench/v1.md`):
  speed, draft acceptance and tool calling, with and without the fix.

Decision rules came before the data:
- **The kpool investigation:** [`../2026-10-glm53-kpool-tail`](../2026-10-glm53-kpool-tail).
- **The layout bisection:** [`preregistration/`](preregistration), which holds the preregistration, three amendments
  and the gate record.
  - Amendment 1 had to change the control configuration before it could start.
  - Amendment 2 defined a gate for testing the tail fix on the screen. The gate failed.
  - **Amendment 3 was written after the validation had been seen.** It records the decision to run the tail-fix
    screens anyway, under the rule already written in Amendment 2.

## What was noticed
On the same 3.25bpw weights, tpurtell 0.7.0 left 6 of 594 GPQA answers empty and 0.8.0 left 20 (Fisher p = 0.009).
On the screen, 0.7.0 failed 12-13 of 40 and 0.8.0 failed 20-24. Accuracy did not differ (`FINDINGS.md`, inference 2).

Ruled out or reduced before the bisection:
- **Draft depth:** 0.8.0 with 0.7.0's five drafts and slot sharing off still left 17 of 594 empty.
- **The two kpool bugs:**
  - They were real and are fixed in 0.9.0 ([tpurtell/glm-5.3-flash-ext3-2x-rtx#5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5)).
  - They are not the main cause. The screen went 22 → 20 on 0.8.0 and 13 → 13 on 0.7.0. On 4bpw TR3 (Brandon),
    0.9.0 left 23 of 594 answers empty vs 15 on 0.8.0 (p = 0.25).
- **The chat template:** the preregistration records that all 198 prompts render identically under both
  templates.

What remained: the parallel layout, and the B12x kernel fork pinned since 0.8.0.

## The DCP1 tail bug
Read in code and confirmed with the image's own kernels:
- The kpool indexer writes the selected pools to columns 0-2043 and the tail to the fixed columns 2044-2046.
- In the DCP1 branch the selection length becomes min(causal length, 2,048). Every column at or beyond that length is
  then masked out.
- So at causal lengths up to 2,043 that are not a multiple of 4, every MLA layer misses the current token and up to two
  tokens before it.
- A GPQA prompt is a few hundred tokens; the six used below are 124-365. So three of every four decode steps in the
  first ~1,700-1,900 generated tokens of an answer are affected.

Not affected:
- DCP2, which compacts its selection.
- The dense short-prefill path.

Upstream vLLM avoids the case with an exact causal fill for short decodes (vllm-project/vllm#53906).

The fix in #6 compacts the valid entries before the existing mask, in the DCP1 branch only. Rows the stock code
already handled are left untouched.

**Index check** (runs `2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_kpool-tail-index` and the `-tailfix-` twin):

| Selection layout | Without the fix | With the fix |
|---|---|---|
| packed (as the indexer emits) | tail dropped in 9 of 23 cases (lengths 5-7, 1001-1003, 2041-2043), 18 tokens | 0 dropped |
| scattered (synthetic worst case) | tail dropped in 9 cases; 2,062 tokens dropped in all | 0 dropped |

- Index rows the stock code already handled are byte-identical between the two images.
- With packed selections and lengths up to 2,047, the fixed rows attend exactly tokens 0..L-1, the set the dense
  short-prefill path attends.

## Results

### Hard-question screen, layout bisection (2 screens of 40 per configuration, interleaved, fresh server each)
`tools/analyze.py screen-pool k3.25-v0.9.0-dflash3 k3.25-v0.9.0-tailfix-dflash3 k3.25-v0.9.0-ep2dcp2-dflash3`

| Configuration | Failures / 80 (95% Wilson) | Loops | Exhaustions | Q13 | Q79 | Q88 | Q121 | Q127 |
|---|---|---|---|---|---|---|---|---|
| 0.9.0 as released | 41 (40.5-61.9%) | 28 | 13 | 6 | 14 | 16 | 0 | 5 |
| 0.9.0 + DCP1 tail fix | 33 (31.1-52.2%) | 25 | 8 | 0 | 13 | 15 | 0 | 5 |
| 0.9.0, 0.7.0 layout (control) | 29 (26.6-47.2%) | 21 | 8 | 0 | 13 | 12 | 0 | 4 |

Per question, each count is out of 16.

- **As released vs the control:** 41 vs 29, Fisher exact p = 0.079. Preregistered verdict: **INCONCLUSIVE**.
- **With vs without the fix:** 33 vs 41, p = 0.27. Verdict: **NO DETECTABLE LOOP EFFECT**.
- **With the fix vs the control:** 33 vs 29, p = 0.63.
- The joint reading in Amendment 2 was not triggered, because the first comparison is not a LAYOUT verdict.
- Failure rates fall in the order 51%, 41%, 36%, but none of the differences is significant.
- **Post hoc, not preregistered:** question 13 failed 6 of 16 times as released, and 0 of 16 with the fix or with the
  control layout. In the earlier tpurtell 0.8.0 screens with DFlash2 ×3 it failed 5 and 3 of 8; it never failed on 0.7.0.

**Speed and capacity** (finished requests from the rows; the rest from server logs, summarised in each run's notes):

| Configuration | Median completion tok/s, finished requests | Server throughput, ≥ 8 running | Mean acceptance length | KV cache, tokens |
|---|---|---|---|---|
| 0.9.0 as released | 48.9, 50.2 | 486, 497 tok/s | 2.99, 3.05 | 4,707,515 |
| 0.9.0 + DCP1 tail fix | 46.7, 46.5 | 484, 490 tok/s | 2.98, 2.97 | 4,707,515 |
| 0.9.0, 0.7.0 layout (control) | 44.6, 44.1 | 459, 461 tok/s | 2.92, 2.96 | 3,165,056; 3,140,895 |

### Decode vs prefill, with and without the fix (speculation off, prefix caching off, 6 prompts × 2,600 tokens)
| Positions | Mean KL as released | With the fix | Top-1 agreement as released | With the fix |
|---|---|---|---|---|
| i < 2,044 | 0.0656 | 0.0103 | 93.8% | 97.6% |
| i ≥ 2,048 | 0.0307 | 0.0187 | 96.1% | 97.1% |

- **Per prompt:** the fix is lower in 6 of 6 prompts below 2,044 and in 5 of 6 from 2,048.
- **Past 2,048:** the difference persists even though the fix changes no selection there, because keys and values
  written under the bug stay in the cache.
- **i mod 4 split:** the predicted split (larger at i mod 4 ≠ 0) appeared in both builds, so it was not the signature
  (gate record).
- **Limits:** this is one generation per build per prompt.

### Downstream checks, with and without the fix (single server each)
- **tool-eval-bench (88 scenarios, two repeats):**
  - 157 and 157 of 176 points without the fix, 159 and 163 with it.
  - TC-80 and TC-88 fail in both repeats without the fix and pass in both with it.
  - Ten other scenarios vary between repeats of the same build.
- **Serving probe (16 prompts, single runs):**
  - Per-request decode 147.3 vs 146.9 tok/s at concurrency 1 and 62.5 vs 64.9 at 8.
  - Acceptance rate 0.534 vs 0.526 at 1 and 0.529 vs 0.550 at 8.
  - KV capacity is unchanged.

## Tangible gains
1. **Kpool fixes** (#5, merged and shipped in tpurtell 0.9.0). They correct cache corruption that upstream's own
   regression tests detect (33/33 pass, from 29/33).
2. **DCP1 tail fix** (#6):
   - Decode attends every token the indexer selected.
   - Decode-vs-prefill KL falls from 0.066 to 0.010 below 2,044 tokens and from 0.031 to 0.019 after.
   - Two tool-calling scenarios go from failing to passing in both repeats.
   - Server throughput, acceptance and KV capacity unchanged. The median rate of finished screen requests was lower
     (46.5-46.7 vs 48.9-50.2 tok/s), but it also depends on which requests finish.
3. **A smaller open gap.** The failure rate trends 51% (as released) > 41% (with the fix) > 36% (control layout). It
   is not significant, but it is consistent with the fix closing part of the difference the layout control shows.

## What is not claimed
- That the tail fix reduces looping or empty answers. The preregistered verdict is "no detectable loop effect", and
  there is no GPQA run with the fix yet.
- That the parallel layout causes the 0.7.0 vs 0.8.0 completion gap. That verdict is inconclusive.
- That v0.7.0's layout is preferable. It is a diagnostic control, and it holds about a third less KV cache.
- That any of this closes the gap to published GPQA scores.
- That the question-13 pattern is more than a post-hoc observation.

## Open questions, and the experiment that would answer each
1. **Does the layout itself matter (as released vs control, 41 vs 29 of 80)?** At the observed rates, 80% power at
   p < 0.05 needs about 170 requests per configuration, about three more screens each (`recompute.py`, power section). That is
   optimistic, because failures cluster by question. A better instrument is a screen version with more questions and
   a different seed per repeat (`protocols/hard-prompt-screen/v1.md`, known limitations).
2. **Is there a layout effect left after the fix (33 vs 29 of 80)?** A 5-point difference would need ~1,500 requests
   per configuration on this screen. A 3-pass GPQA run per configuration (594 answers, empty count) is the more
   practical test.
3. **Does the fix change GPQA empty answers or accuracy?** A 3-pass GPQA run with the fix, against 20 of 594 (0.8.0)
   on the same weights.
4. **Batch composition.**
   - Scoring answer letters on vLLM, local measurements put repeats at concurrency 1 within KL ~3e-4 of each other and
     at concurrency 8 ~0.02. Those receipts are not published.
   - Does batch composition change completion? Experiment: the hardest screen questions at concurrency 1 vs 8, plus a
     concurrency-1 repeat with prefix caching off to attribute the residual.
5. **The 511-pool slice (read in code; effect not measured).**
   - From 2,048 tokens of context, the indexer keeps 511 of the 512 pools it selected.
   - Which one it drops depends on the order the top-k emits them, not on score.
   - Experiment: teacher-forced replay of saved screen traces, logging the dropped pool's score rank on looping vs
     finishing traces.
6. **`swiglu_limit` on routed experts (read in code, all versions).**
   - The checkpoint sets `swiglu_limit` 10.0. The shared expert clamps, but the routed-expert path does not pass the
     limit.
   - Experiment: count routed activations above 10 on long traces, then compare against the BF16 teacher with and
     without the clamp.
7. **Top-k ties (read in code).**
   - The DCP2 merge path uses a stable top-k with lowest-index ties.
   - The DCP1 fused path places ties by atomic arrival order, so exact ties can resolve differently between layouts
     and between batch compositions.
   - Experiment: count exact score ties at the selection boundary along long traces, and compare selections at
     concurrency 1 vs 8.

## Receipts and how to recompute them
Every number above comes from files in this repository, except the batch-composition figures in open question 4
(marked there as unpublished). Server logs are not published; their aggregates (KV cache
size, server throughput, acceptance length) are quoted in each screen run's `run.json` notes. From the repository root,
standard library only:

```bash
python3 tools/verify.py                                     # recompute every summary.json; check every comparison and label
python3 investigations/2026-10-glm53-looping/recompute.py   # every number in this README, from the rows
python3 tools/analyze.py screen-pool k3.25-v0.9.0-dflash3 k3.25-v0.9.0-tailfix-dflash3 k3.25-v0.9.0-ep2dcp2-dflash3
python3 tools/analyze.py screen-speed                       # median completion tok/s of finished screen requests
python3 tools/analyze.py gpqa-pairs                         # the GPQA empty-answer comparisons quoted from FINDINGS.md
```

| Runs (`runs/<id>/`: `run.json`, `results.jsonl`, `summary.json`) | Contents | Protocol |
|---|---|---|
| [`2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_screen-bisect1`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_screen-bisect1)<br>[`2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_screen-bisect2`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_screen-bisect2) | Screens as released, 40 rows each | [`hard-prompt-screen/v1`](../../protocols/hard-prompt-screen/v1.md) |
| [`2026-10-07_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_screen-bisect1`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_screen-bisect1)<br>[`2026-10-08_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_screen-bisect2`](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_screen-bisect2) | Screens with the fix | [`hard-prompt-screen/v1`](../../protocols/hard-prompt-screen/v1.md) |
| [`2026-10-07_glm53-flash_k3.25-v0.9.0-ep2dcp2-dflash3_screen-bisect1`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-ep2dcp2-dflash3_screen-bisect1)<br>[`2026-10-08_glm53-flash_k3.25-v0.9.0-ep2dcp2-dflash3_screen-bisect2`](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.0-ep2dcp2-dflash3_screen-bisect2) | Screens, control layout | [`hard-prompt-screen/v1`](../../protocols/hard-prompt-screen/v1.md) |
| [`2026-10-07_glm53-flash_k3.25-v0.9.0-nospec-nocache_decode-prefill`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-nospec-nocache_decode-prefill)<br>[`2026-10-07_glm53-flash_k3.25-v0.9.0-tailfix-nospec-nocache_decode-prefill`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-tailfix-nospec-nocache_decode-prefill) | 15,600 positions each: region, i mod 4, abs Δlogprob, top-1 agreement, KL | [`decode-prefill-consistency/v1`](../../protocols/decode-prefill-consistency/v1.md) |
| [`2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_kpool-tail-index`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_kpool-tail-index)<br>[`2026-10-07_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_kpool-tail-index`](../../runs/2026-10-07_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_kpool-tail-index) | 46 cases each: tail columns before and after the mask, drops, row hashes | [`kpool-tail-index/v1`](../../protocols/kpool-tail-index/v1.md) |
| [`2026-10-08_glm53-flash_k3.25-v0.9.0-dflash3_tool-eval`](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.0-dflash3_tool-eval)<br>[`2026-10-08_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_tool-eval`](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_tool-eval) | 2 × 88 scenarios: id, status, points | [`tool-eval-bench/v1`](../../protocols/tool-eval-bench/v1.md) |
| [`2026-10-08_glm53-flash_k3.25-v0.9.0-dflash3_serving-probe`](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.0-dflash3_serving-probe)<br>[`2026-10-08_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_serving-probe`](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.0-tailfix-dflash3_serving-probe) | 16-prompt speed and acceptance probe | [`serving-probe/v1`](../../protocols/serving-probe/v1.md) |
| [`2026-10-05_glm53-flash_k3.25-v0.7.0-dflash5_kpool-kernel-tests`](../../runs/2026-10-05_glm53-flash_k3.25-v0.7.0-dflash5_kpool-kernel-tests)<br>[`2026-10-05_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5_kpool-kernel-tests`](../../runs/2026-10-05_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5_kpool-kernel-tests)<br>[`2026-10-05_glm53-flash_k3.25-v0.8.0-dflash3_kpool-kernel-tests`](../../runs/2026-10-05_glm53-flash_k3.25-v0.8.0-dflash3_kpool-kernel-tests)<br>[`2026-10-05_glm53-flash_k3.25-v0.8.0-kpoolfix-dflash3_kpool-kernel-tests`](../../runs/2026-10-05_glm53-flash_k3.25-v0.8.0-kpoolfix-dflash3_kpool-kernel-tests) | Upstream kpool regression tests (fixes of #5) | [`kpool-kernel-tests/v1`](../../protocols/kpool-kernel-tests/v1.md) |

- Configurations: `configs/glm53-flash/k3.25-v0.9.0-dflash3.json`, `k3.25-v0.9.0-tailfix-dflash3.json`,
  `k3.25-v0.9.0-ep2dcp2-dflash3.json`, `k3.25-v0.9.0-nospec-nocache.json`, `k3.25-v0.9.0-tailfix-nospec-nocache.json`.
- Comparisons, checked by `tools/verify.py`:
  [`glm53-flash-v090-layout-screen`](../../comparisons/glm53-flash-v090-layout-screen),
  [`glm53-flash-v090-tailfix-screen`](../../comparisons/glm53-flash-v090-tailfix-screen),
  [`glm53-flash-v090-tailfix-decode-prefill`](../../comparisons/glm53-flash-v090-tailfix-decode-prefill),
  [`glm53-flash-v090-tailfix-tool-eval`](../../comparisons/glm53-flash-v090-tailfix-tool-eval).
- Preregistration, amendments, original hashes and the gate record: [`preregistration/`](preregistration).
- Clients:
  - [`tools/clients/hard_prompt_screen.py`](../../tools/clients/hard_prompt_screen.py).
  - [`tools/clients/decode_prefill_consistency.py`](../../tools/clients/decode_prefill_consistency.py) and
    [`tools/kernel/kpool_tail_index_check.py`](../../tools/kernel/kpool_tail_index_check.py): published copies of the
    scripts that produced these runs, differing only in local names and in how the fix kernel is found.
- Earlier work: [`../2026-10-glm53-kpool-tail`](../2026-10-glm53-kpool-tail),
  [`comparisons/glm53-flash-gpqa-configs`](../../comparisons/glm53-flash-gpqa-configs),
  [`comparisons/glm53-flash-k4-v080-v090-gpqa`](../../comparisons/glm53-flash-k4-v080-v090-gpqa).
