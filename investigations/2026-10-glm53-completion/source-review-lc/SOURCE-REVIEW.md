# Focused source review: what depends on context length (amendment 20, section 2)

2026-10-13. CPU only: no GPU, no containers, no sudo. Nothing outside this folder was modified.

**Question.** Between stock 0.7.0 and 0.9.1/B1, what differs in a way that depends on context length? Could such a
difference make generation past about 64-100k tokens stop terminating on 0.9.1 (amendment 19f)? Or could it make 0.7.0
terminate for a wrong reason? This is not a release audit.

**Sources.**
- **Engine trees** under `kdcp/` in the session scratchpad, as extracted earlier:
  - `v070` is 0.7.0 with B12x `fe054789`;
  - `v091` is 0.9.1 with B12x `7fcc094e`;
  - `b0-cache/src` is B0. B1 is B0 plus `d7/mechanism/d7-fix.diff`, which changes only `_kpool_compress_insert`. B0's B12x
    files are byte-identical to 0.9.1's.
- **Configuration:** model `config.json` (snapshot `0490d2f7`); the recipes `glm53-flash-exl3-k3.25@tpurtell` and
  `…+v0.9.1+bundle-d1234+d7`; the start scripts `<stacks>/glm53-flash-v0.{7,9}.0/start.sh` (read only); the GPQA
  server logs (`<gpqa-runs>/*server.log`; `b1-screen/run-s1236/server-s1.log`).

**Receipts in this folder.**
- `thresholds.py` / `thresholds.out`: every engagement length below, computed from constants cited in the script.
- `sources.sha256`: hashes of the cited files.

**Line references.** `v091:` and `v070:` mean paths in those trees. Where 0.7.0 and 0.9.1 are byte-identical, one line
is cited.

**Defect labels.** inv-D1…D7 are this investigation's defect matrix. "DCP-audit Dn" refers to `audits/phase1-dcp-attention.md`.

**Classes.**
- **KB**: known bug.
- **EV**: expected numerical variation.
- **UI**: untested invariant.

## Model geometry that decides the answer

- **No RoPE anywhere in the target.** `qk_rope_head_dim` is 0 (config) and `mla_use_nope` is true. The indexer takes its
  rope width from the same field (`models/glm5next/nvidia/attention.py:244`), and its RoPE branch is skipped when that
  width is 0 (`:354`). Position reaches the target only three ways:
  - the 34 KDA layers;
  - the kpool `compress_ape`, which depends on `pos % 4` (`attention.py:249-251`);
  - causality.

  RoPE theta, YaRN and `max_position_embeddings` (1,048,576) therefore cannot act as a long-context mechanism in the
  target. The DFlash2 draft has RoPE (theta 10⁴, sliding window 2,048), which affects acceptance only.
- **Selection size.** `index_topk` 2048 with `index_kpool` 4 gives 512 selected pools, sliced to 511 plus 3 tail tokens
  (`sparse_attn_indexer_kpool.py:574, 644, 855, 1009`). The pool count is P = L/4. Under 0.7.0's DCP2 each rank holds
  L/8 pools.
- **Selection route.** Rows per step = requests × (K+1), padded to the capture size:
  - 0.9.1/B1, K=3: 1-4 requests use the fused route; 5 or more use tiled.
  - 0.7.0, K=5: 1-2 requests use fused; 3 or more use tiled.

  This follows `fused_indexer.py v091:140, 901-930`. GPQA ran mostly at 8 in flight, which is the tiled route: 87% of
  B1's status lines, 88% of 0.7.0's and 81% of stock 0.9.1's.
- **Cache geometry.**

  | | 0.9.1 / B1 | 0.7.0 |
  |---|---|---|
  | Attention block | 4,352 tokens | 3,584 local tokens (7,168 global) |
  | Block-table width W0 | 241 | 147 |
  | Index-K pages per request (64 pool rows each) | 4,097 | 2,058 per rank |
  | KDA (mamba) block | 4,352 | 3,584 (DCP scaling undone, `single_type_kv_cache_manager.py v070:1388-1391`) |

## Ranked candidates

| # | Candidate | Engages at (tokens) | B1 vs 0.7.0 | Class |
|---|---|---|---|---|
| 1 | Decode-written index-K is wrong in 0.7.0 and stock 0.9.1 but correct in B0/B1 (inv-D2, with DCP-audit D6/D7 on 0.7.0) | From the first pool completed in decode. Almost all of the context is affected after a few thousand tokens, and the number of corrupted candidates grows with L/4 | 0.7.0: 100% of decode-completed pools wrong at 1, 2 and 8 requests. Stock 0.9.1: 0%, 47-57% and 78-90%. B1: 0% | **KB** (fixed in B0/B1). Prefill cannot see it |
| 2 | Sparse-selection kernel regime transitions, set-exact by design | B1 tiled: 2nd slice from 65,536; 2nd supertile chunk and 3rd slice from **131,072**; then 196,608 and 262,144. B1 fused: local trim from 22.5k / 47k / 96k (4 / 2 / 1 requests); repeated trims from 42k / 88k / 180k. 0.7.0: tiled 131,072 and 262,144; fused 45k / 94k and 84k / 177k | Every threshold in 0.7.0 sits at about twice the length (DCP2 halves per-rank pools), and 0.7.0's fused merge is always serial | **UI** beyond about 120k. Exact selection was verified on real inputs only to about 120k (stage 2). No test or real data puts live pools in supertile chunk ≥ 1 |
| 3 | KDA running-state migration across blocks under speculative decoding ("align" mode) | Every 4,352 tokens (B1; 75 per capped reply) or 3,584 (0.7.0; 91). Count grows with L | Code and dtypes identical. Only the block size and K differ | **UI**. Phase 1 decode-vs-prefill covered 3 crossings, to 16k. No guard site covers it (amendment 17) |
| 4 | Prefill-path specifics, which matter for the fixed-history probe rather than for generation | inv-D7 at every unaligned chunk start: 2,043-token chunks on 0.7.0, about 1 pool per chunk, about 49 pools in a 100k prefix. Query sub-chunking above 262.5k / 262.8k. Prefill top-k is a different kernel from decode | D7 is fixed in B1, unfixed in 0.7.0 | **KB** (D7) and **EV** |
| 5 | DCP partition, merge and cache geometry | Byte offsets above 2³¹ from block 935 (B1). DFlash sliding-window block recycling every 1,088 tokens in shared MLA pages (B1 only). Column 2047 (−1) sits inside `nsa_len` from 2,048 (DCP1) | DCP2 combine and merge have no length dependence beyond halving the indexer thresholds | **EV**. The above-2³¹ addressing was tested clean (amendments 16-17). Draft writes into shared pages under real scheduling are **UI**, with a low prior |
| 6 | Speculative verification, sampling and stopping | No length threshold | Byte-identical code. K changes the sample path at a fixed seed, not the distribution | **EV**. No mechanism |

**Switches between 16k and 300k tokens.**
- **B1, path switches:**
  - fused local trim at 22.5k, 47k and 96k;
  - fused repeated trims at 42k, 88k and 180k;
  - tiled slices at 65,536, 131,072, 196,608 and 262,144;
  - tiled supertile chunks at 131,072 and 262,144;
  - prefill query sub-chunking at 262,528;
  - a KDA block migration every 4,352 tokens.
- **0.7.0, path switches:**
  - fused local trim at 45k and 94k;
  - fused repeated trims at 84.5k and 177k;
  - tiled slice and chunk changes at 131,072 and 262,144;
  - prefill sub-chunking at 262,784;
  - a KDA migration every 3,584 tokens.
- **Stock 0.9.1 only:** the tiled column-511 quick-exit regime (inv-D1) ends at 65,536.
- **Precision:** no switch in this range on any engine. Index-K is stored as e4m3 with fp32 scales and scores are fp32.
  The KDA state is fp32, the conv state bf16, and the MLA record e4m3 with fp32 scales. None depends on length.

The B1 thresholds at 65,536 and 131,072 lie near where amendment 19f sees finishing fall (64-100k; almost none past
about 150k). That closeness is not evidence. The 19f thresholds were chosen after the results, and amendment 20 forbids
inferring a change point from them.

## 1. Pool-count and fold transitions in the sparse indexer

**What differs.**
- **Tiled route** (`b12x/.../dsa_indexer/paged.py`, same structure in both B12x versions):
  - The supertile is 32,768 pool rows (`v091:51`; `scratch.py v091:32, 780-813`; capacity-capped). The chunk count is
    fixed by the allocated page-table width: 9 on B1 and 5 on 0.7.0 (`paged.py v091:907`; width from
    `indexer.py v091:797-807`).
  - The two-level fold is on whenever the width is at least 32,768 rows (`paged.py v091:150-230`, threshold at `:185`;
    the candidate slab is about 4.5 MB against a 256 MiB budget).
  - Each chunk is split into 16,384-row slices, selected in parallel (`:1172-1196`). One level-2 `run_row_topk` then runs
    over 17 × 512 candidates on B1 (9 × 512 on 0.7.0, per rank) (`:1197-1206`).
  - Slice extents are clipped to each row's live length (`tiled_topk.py v091:672-716`).
  - `active_width` is not supplied for kpool decode (`sparse_attn_indexer_kpool.py:919`), so the scorer covers the full
    capacity and masks by length.
- **Fused route** (`fused_indexer.py`):
  - The number of CTAs per row is fixed per captured row count: 47 / 23 / 11 on B1 and 23 / 11 on 0.7.0 (`v091:952-962`).
  - Each CTA takes a live-length share of the pages (`v091:1843-1850`) and keeps a carry of 512 + 448 entries
    (`v091:1137, 1605`).
  - A local radix trim runs when a CTA's share passes 512 rows. Repeated trims start near 960 rows
    (`v091:2328-2370`; v070 `:1396`, `:1899`).
  - **Cross-CTA merge:**
    - B1 always merges cooperatively (`v091:964-983`). Its docstring says it was benchmarked at 4k-64k live tokens.
    - 0.7.0's crossover is 22,000 + 117 × CTAs, but the cap is CTAs × 512, which is below the crossover at 0.7.0's
      served row counts. So **0.7.0's fused merge is always serial** (`v070:593-621`, `:612`, `:621`).
    - Correction: the earlier notes' "serial→cooperative at about 176k" (DCP-audit D4; `notes-b12x.md` §5) is not
      reachable as served.
- **Per-rank top-k merge.** Under DCP2, 0.7.0 merges the two ranks' top-512 lists over 1,024 candidates
  (`sparse_attn_indexer.py:1139-1218`). B1 is DCP1 and has no merge.
- **Column 511.**
  - Stock 0.9.1 tiled hits the quick exit only while one slice is live, so up to 65,536 tokens. The stage-2 replay saw
    96-97% high-rank drops below 64k and 13-15% in the band above 96k.
  - B1 moves the lowest-scoring pool to the last column on every route (`b0 sparse_attn_indexer_kpool.py:63-76, 966`)
    and in prefill (`:662-667`).

**When it engages.** See the table and `thresholds.out`. Coverage so far:
- Stage-2 real inputs reached about 120k tokens (about 30k pools), all within supertile chunk 0, with sets exact against
  fp64.
- The B12x tests stop at 32,768 rows for the two-level fold, with one chunk (`tests/attention/test_paged_prefill_topk_long_context.py:259-266`).
- The fused tests use top-k 2048 at 65,536 rows (`test_fused_indexer.py:490-518`). That is not the served top-k 512 at
  11-47 CTAs.

**Class.** UI beyond about 120k (B1's chunk-≥1 and multi-slice regimes; the fused regime with repeated trims at 1-2
requests). inv-D1 is fixed in B1.

**Tests.**
- **Model-free (decisive, cheap):** replay synthetic and recorded indexer inputs at L = 128k-330k through B1's tiled route
  (rows 24 and 32) and fused route (rows 4, 8 and 16), against an fp64 top-512. Require set equality and the minimum in
  column 511.
- **Prefill probe:** cannot see this. Prefill uses `fp8_fp4_mqa_logits` plus `top_k_per_row_prefill`
  (`sparse_attn_indexer_kpool.py:561, 600`).
- **Target decode:** only decode exercises it. A one-request decode control takes the fused route with 47 CTAs, not
  GPQA's tiled route. Run it with 5 or more requests in flight, or with `B12X_FUSED_INDEXER=0`.

## 2. Sequence-length and index representation limits

**What differs.** Only the widths and geometry listed above. Every limit sits far from the 64k-330k range:
- Positions, `seq_lens`, slots and global pool ids are int32 (below 2³¹). No int16 or uint16 appears on any served
  path; the only hits are in `mxfp4.py`, which is not used.
- Addresses: B12x KV and index-K byte offsets are computed in Int64 (`notes-b12x.md` §4). The Triton mamba copy widens
  block ids to int64 (`v1/worker/mamba_utils.py:176-179`).
- B1's MLA and index-K offsets pass 2³¹ bytes from block 935. That is a property of block id, not length, and was
  exercised clean at blocks ≥ 935 (amendments 16-17).
- Prefill logits element offsets are bounded by the 512 MB budget (134M < 2³¹) through query sub-chunking
  (`indexer.py:81-127, 1131-1138`).
- No KV preemption or recompute in any log: maximum KV usage was 85.8% on 0.7.0 and 41.5-41.6% on 0.9.1 and B1, with 0
  preemptions.

**Class.** EV. No overflow or limit candidate.

**Test.** None needed. If extending past 262k, note that prefill sub-chunks the query dimension there; selection per row
is unchanged.

## 3. Sparse selection and attention paths at long lengths

**What differs.**
- **MLA decode versus extend.** The choice depends on query length and rows only (`b12x_mla_sparse.py v091:3170-3175`),
  never on context. Width is fixed at 2,048 with 32 splits.
- **Masking.** `nsa_len` is min(L, 2,048). On DCP1, column 2047 (−1) lies within the length from L = 2,048. It is
  masked and reads slot 0 with weight exactly 0 (`v091:3033-3046`; `notes-b12x.md` §1b). 0.7.0's DCP2 compaction drops
  −1 instead.
- **Selection path.** This is the long-context dependence (section 1). A further dependence comes from where index-K
  bytes come from: decode-built pools on every engine, prefill-built pools in a probe.
- **inv-D2, the ranked first candidate:**
  - 0.7.0 builds every decode-completed pool wrongly (tail ring without positions, plus the 4-slot ring with 5 drafts,
    plus the seed stride).
  - Stock 0.9.1 does so depending on concurrency.
  - B0/B1 are correct.
  - Prompts are short (median 204 tokens), so beyond a few thousand tokens almost every candidate pool on 0.7.0 is
    corrupted. As L grows, more corrupted candidates compete for the 511 slots.
  - Phase 1 step 4 (0.7.0 decode drifting from its own prefill at 8-16k) agrees with this; it is not proof.

**Class.**
- inv-D2: KB.
- The decode-versus-extend split: EV.
- The slot-0 read: EV, from 2,048. It is harmless while slot 0 is finite and would be catastrophic, not subtle,
  otherwise.

**Tests.**
- **Prefill probe:** it does not reproduce inv-D2. Prefill rebuilds index-K through `_kpool_compress_insert`, so 0.7.0
  scored by prefill is a computation 0.7.0 never ran while generating past the prompt.
- **Target decode:**
  - On 0.7.0, compare P(reasoning close / EOS) from prefill with the same quantity after force-decoding the last W
    tokens. Only those W/4 pools are rebuilt by decode, so only a full forced decode reproduces the whole state.
  - On B1 the same pair serves as the control, expected to agree up to KDA rounding.
  - A cheaper discriminator: compare index-K bytes for the same history written by decode and by prefill on each engine.

## 4. DCP partition, merge and cache geometry

**What differs.**
- **Partition.** 0.7.0 uses DCP2 (token t on rank t%2, pool p on rank p%2), combined across ranks by LSE. B1 uses DCP1
  with MLA owners (`split:25`).
- **Geometry.** As in the geometry table. The 528-byte and 656-byte records are equivalent up to rounding (amendment
  16). The combine has no length term.
- **Coupling to length.** The only one is that 0.7.0's indexer thresholds sit at twice the token count (section 1).
- **Draft slot sharing (B1 and stock 0.9.1, not 0.7.0).**
  - The DFlash2 draft has a 2,048-token sliding window in 1,088-token blocks, inside shared MLA pages
    (log: `page=2228224 of 2297856`).
  - Its block-table entries are recycled every 1,088 tokens. A long reply cycles about 300 draft blocks through the
    shared pool.
  - The draft slot mapping (`spec_decode/dflash/speculator.py:544-574`) was not among amendment 17's guard sites. The
    upstream null-block and rejected-suffix guard (97388c44f) is absent, per the earlier audit.
  - Amendment 16's fixture covered the draft kernels with fixture mappings.

**Class.** EV (partition). UI with a low prior (draft writes into shared pages under real scheduling; rate grows with
length; 0.9.1 family only).

**Tests.**
- **Prefill probe:** cannot see draft writes, because the probe has no drafting.
- **Target decode:** extend the amendment-17 guard with one site, the draft KV write slot against the shared BlockPool
  owner, over a 100k+ decode. Alternatively, run the decode control with `VLLM_GLM53_DRAFT_SLOT_SHARING=0`. That changes
  pool size and block geometry, so it is diagnostic only.

## 5. Recurrent state (KDA) precision and update paths

**What differs.** Nothing in code:
- `kda.py`, the FLA kernels, `_flashkda_C`, `v1/worker/mamba_utils.py` and `mamba_hybrid.py` are byte-identical in
  0.7.0 and 0.9.1. B0 only passes `positions` (`mamba_hybrid.py b0:367`).
- The recurrent state is fp32 and the conv state bf16 (`model_executor/layers/mamba/mamba_utils.py:177-183`;
  `kda.py:138-150`).
- "Baseline full-state rollback" (`start.sh v0.9.0:581`, `REPLAYSSM_ACTIVE=0`): `fused_recurrent_kda` writes one fp32
  state per spec token into K+1 slots and reads slot `num_accepted − 1` next step (`kda.py:613-628`). Rollback is exact
  and adds no rounding per step. K changes the number of slots, not the arithmetic.

**Length dependence.**
- **Align mode** moves the running state to a new block column once per mamba block. It applies a `num_accepted − 1`
  bias and copies the conv window (`mamba_hybrid.py:211-258, 372-417`; `mamba_utils.py:117-252, 393-524`). That is 75
  times per capped reply on B1 and 91 on 0.7.0.
- **Prefill builds the state** with the chunked kernel (`kda.py:671`); decode builds it with the sequential fp32
  recurrence. They differ numerically, and accumulated error is expected to grow slowly with L (fp32 sums).

**Class.**
- Prefill-versus-decode state: EV. This is the amendment-20 caveat that a cold prefill does not reconstruct decode
  state.
- Migration under spec rejection near a boundary: UI. The code is identical, so it is not a version mechanism unless
  the effect depends on K or block size.

**Test.**
- **Target decode:** look for a discontinuity in decode-versus-prefill agreement on the same prefix just after versus
  just before a multiple of 4,352 tokens, at 64-128k.
- **Prefill probe:** shows no migration.

## 6. Long-context speculative verification, stopping and logits processing

**What differs.** Nothing except K.
- The sampler, `gumbel.py`, `rejection_sampler*.py`, the DFlash and DFlash2 code and `thinking_budget.py` are
  byte-identical across releases.
- **Acceptance is distribution-exact.**
  - Penalties, temperature and top-p are applied first (`sample/sampler.py:153-218`).
  - A one-hot greedy draft x is accepted when log p(x) > log u (`rejection_sampler_utils.py:640-666`).
  - On rejection the token is resampled from p with x removed (`:812-823`); the bonus token comes from p.
  - Noise is keyed by (seed, position) (`gumbel.py:111`; `rejection_sampler_utils.py:569`). Philox has no wrap at
    these positions.
- **K changes which random draw a position uses** (acceptance u versus resample Gumbel), so trajectories differ at a
  fixed seed with identical logits. The distribution does not.
- **EOS ids** come from the identical `generation_config.json`: 154820, 154827 and 154829.
- **Requests.** They carry seed, `max_tokens` 327,680, temperature 1.0 and top_p 0.95 (proxy fields).
  - The structured-output edits in 0.9.1 apply only with a grammar.
  - The thinking budget applies only with `thinking_token_budget`.
  - Neither appears in the recorded fields, but the proxy logs a subset, so this is not fully excluded.
- **Draft.** Positions are clamped at `max_model_len − 1` (`dflash/speculator.py:573`), with a sliding window of 2,048.
  Draft quality affects speed only.

**Class.** EV. No length-dependent stop, EOS or logits path was found.

**Test.** None specific. The probe's termination measures (sequence probability of reasoning close, EOS / end of turn,
eligibility after top-p) do not depend on K. K matters only through route selection (section 1).

## Settled items, not repeated

- Selection-set exactness up to about 120k.
- The column-511 mechanism and its fix (inv-D1).
- The tail ring (inv-D2) and its fix.
- Writer and query projection, and the 528-byte versus 656-byte records (amendment 16).
- TP2 all-reduce (amendment 16).
- Addressing above 2³¹ for the driven kernels (amendment 16).
- Live MLA and index-K mappings (amendment 17).
- inv-D7 and its fix (amendment 18).
- E1 routing weights.
- DCP2 LSE combine semantics (DCP-audit D1).
- B12x decode and merge rewrite as rounding only (DCP-audit D3).
- 0.7.0 ring and seed bugs (DCP-audit D6/D7).

## Implications for the fixed-history probe (amendment 20, sections 3-4)

1. **Stock 0.7.0 by prefill is not "old shipped computation" at generation time.**
   - It omits inv-D2's decode-built index-K state, which covers about the whole context past the prompt.
   - It adds inv-D7 at unaligned 2,043-token chunk starts, about 1 pool per chunk.

   Label it "0.7.0 prefill computation", or reproduce decode state with a forced-decode window.
2. **Match the generation route in any decode control.**
   - GPQA long replies ran mostly on the tiled route, at 8 in flight.
   - A one-request decode on B1 uses fused with 47 CTAs; on 0.7.0, fused with 23 CTAs and a serial merge.
3. **Run the model-free kernel replay before extending past 131,072 tokens on B1** (or 262,144 on 0.7.0). It is the only
   new boundary in that range, and no test or real data covers it yet.

## Limits

- **CuTe DSL lowering not compiled or run.** Kernel claims rest on source reading. Exactness beyond 120k is by design,
  not demonstrated.
- **SM count** (188) comes from `HANDOFF.md`. Captured row counts are inferred from the logged capture sizes.
- **Concurrency during a reply.** The status-line shares are pass-level. Per-reply concurrency at long lengths was not
  reconstructed.
- **Compiled `top_k_per_row_prefill`.** Its ordering is not inspectable. It is the same `.so` in all trees.
