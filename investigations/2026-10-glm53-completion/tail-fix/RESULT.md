> Update 2026-10-11: the two GPU runs prepared below were executed (one shipped-graph-mode run each for variants T and U);
> results in `summary-T-graph.txt` and `summary-U-graph.txt` and in the investigation README, section Kpool tail ring.

# Tail fix completed, variants T and U (PLAN.md amendment 11, item 2) and 0.7.0 two-rank check (item 3)

Date 2026-10-11. CPU only: no GPU used, no server started, no `--gpus`. Values are measured unless marked
(inferred).

## Variants (diffs against the 0.9.1 release package)

| hunk | file | change | T | U |
|---|---|---|---|---|
| H1 | `v1/worker/gpu/model_states/mamba_hybrid.py` (call at :353-370) | `positions=input_batch.positions` to `build_attn_metadata` (upstream 98ed0856f3; upstream main mamba_hybrid.py:339) | yes | yes |
| H2 | `v1/attention/backends/mla/indexer.py` `compute_kpool_tail_slot_mapping` (:566-568) | `torch.where(own_block != 0, slots, -1)`; port of upstream 240785b82 / #59528, indexer.py:766-767 at that commit (main :840-841); docstring updated | yes | yes |
| H3 | `indexer.py` `KpoolTailMetadataBuilder.build` (:617-625) | copy the circular mapping into `common_attn_metadata.slot_mapping` (persistent `BlockTables.slot_mappings` view) and return it; same as the tested posfix hunk, comment reworded | yes | yes |
| H4 | `v1/attention/backends/mla/b12x_mla_sparse.py` (:1439) | `causal_len_from_positions = False` gate: the non-pure-decode per-token causal length stays on the scheduler CPU bound (:1447-1484) although positions are present | yes | no |

What each changes at run time:
- Both: tail slots become `own_block * 8 + pos % 8` for every real request (no null-block or stale-column
  writes); rows whose tail block is the null block (padding rows of FULL-graph batches, capture and dummy
  runs) get PAD and write nothing; FULL graph replays read the current mapping.
- H2 vs the tested posfix image: only rows on the null block differ (padding rows were writing tail block 0's
  ring). Measured with the real decode kernel (test K1): posfix writes block 0 at slots [4,5,6,7,4,5,6,7],
  T/U write nothing. No output effect expected, since block 0's ring is not read by real requests once
  positions are passed (inferred).
- U only: for batches that are not pure single-token decode (every speculative-verify step, prefill, mixed),
  the B12X builder takes per-token causal lengths from `positions + 1` instead of `seq_lens_cpu_upper_bound`.
  Identical when the bound equals the true length (test M1); differs only when it deviates (M2: bound 60 vs
  true 54 gives 57-60 under T and 51-54 under U). Consumed as `nsa_cache_seqlens = min(len, 2176)`
  (b12x_mla_sparse.py:3036-3037), so only contexts shorter than the index width can differ (inferred). How
  often the bound deviates in serving was not measured (the bound is `num_computed_tokens_np` + scheduled,
  model_runner.py:1370-1378; optimistic under async scheduling, inferred).

Files: `tailfix-T.diff`, `tailfix-U.diff`, `make_variants.py` (exact-match hunks, each asserted to match once),
`variants/{T,U}/`, `orig-release/` (extracted from the release image; tailguard image identical for these three
files).

## Images (docker requires lowercase repository names, so the tags are lowercased)

| tag | image id | FROM | contents |
|---|---|---|---|
| `local/glm53-v0.9.1-tailfix-t-guard:local` | `sha256:0db4d1d06077c2ea9e94ebbefa5c392c8c191abc054be689cdb70cca21eaec0f` | tailguard `sha256:3cacbb48ba93…` | H1-H4 + observe-only guard (inert unless `GLM53_TAILGUARD`) + inert col511 probe |
| `local/glm53-v0.9.1-tailfix-u-guard:local` | `sha256:301baa73c6e8464b81a2a0e60bc433afc121e78fc97b508fa1e37dafcf2ba99b` | tailguard `sha256:3cacbb48ba93…` | H1-H3 + guard + probe; b12x_mla_sparse.py sha256-checked equal to release |
| `local/glm53-v0.9.1-tailfix-t:local` | `sha256:757bfb7ea4904d9d17631ce80975571bacb9d5569fa2711e0efe3eb5df209110` | release `sha256:a9e985c0…` | H1-H4 only; build gate: indexer op sha256 = release, no guard module, no probe marker |

Build gates (all passed): py_compile, `cmp` of installed vs patched files, marker greps, guard inert by default
(guard images), `test_tailfix.py` with the variant's expectation. Build logs: `build-*/build.log`.

## CPU tests (CPU-only containers, `--network none`, `CUDA_VISIBLE_DEVICES=`, `TRITON_INTERPRET=1`)

| image | test_tailfix.py (17) | T2 scenarios A-D, CPU emulation (6) | tailguard tests (22) |
|---|---|---|---|
| tailfix-t-guard | 17/17 (expect T) | 6/6 | 22/22 |
| tailfix-u-guard | 17/17 (expect U) | 6/6 | 22/22 |
| tailfix-t | 17/17 (expect T) | 6/6 | n/a (no guard) |
| release a9e985c0 | 17/17 (expect release: 5 fix checks do not hold) | (6/6 earlier) | n/a |
| previous posfix 0583aa9e | 14/17 against U: G4, G6, K1 fail = what the guard adds | | |

Logs: `tests/*.log`. test_tailfix.py covers: P1 plumbing (AST of prepare_attn); G1-G6 the 240785b82 test cases
ported to ring 8 (collapse, isolation, short requests, padding request on null block, empty batch, dummy batch);
B1-B3 builder output and persistent-buffer write; K1-K3 real decode kernel on a padded FULL-graph batch and pool
completion across two steps; M1-M2 B12X causal-length source; I1-I2 the real `build_attn_metadata` with a tail
group and a B12X group.

## Prepared GPU runs (NOT RUN)

`run_tailfix.sh T|U [ATTEMPT]` (copy of run_tailguard_fix.sh): shipped graph mode (no EXTRA_SERVE_ARGS), device
guard, DFlash2 x3, stage-2 run-2 workload, REPEATS=8, pre-flight GPU and container checks, cleanup trap, wall
cap 2,400 s, cost line to `tailfix/costs.jsonl`, outputs to
`<runs>/tailfix/{T,U}-graph-a1/`.

```bash
<runs>/tailfix/run_tailfix.sh T 1
<runs>/tailfix/run_tailfix.sh U 1
```
(the script reads `client_tg.py` / `summarize_tg.py` from its own directory; identical copy in the session
scratchpad `kdcp/tailfix/`).

Expected about 0.6 GPU-h each (T1 0.568, T3 0.584 on the same workload), 1.2 GPU-h for both; cap 1.33 per run.
Measured (`../costs.jsonl`, server start to stop): T 0.58, U 0.57, 1.15 GPU-h for both.
Pass criteria as amendment 10e (zero writes outside the own block, zero null-block aliasing, no new guard
violations, all requests complete), plus throughput via `../tailguard/throughput_tg.py`. The guard does not
observe the MLA causal-length source; a T vs U difference would have to be measured separately.

## 0.7.0 two-rank tail ring (CPU, real kernels)

`ring070/two_rank_tailring.py`, run in each release image under TRITON_INTERPRET=1 (no GPU). Real: generic
slot-mapping kernel with per-rank CP masking (`block_table.py:262-328`, CP_SIZE 2, interleave 1), compressed
index-K slot mapping (`compressor_utils.py`), tail builder, seed and decode kernels, reference compression.
Synthetic: scheduler (arrivals every 2 steps, whole-prompt prefills, uniform verify windows), draft acceptance
(i.i.d.), keys (random). 3 seeds per cell; prompts 64-400 tokens; 160 generated tokens per request.

Share of decode-completed pools whose index-K differs from the compression of the request's committed keys:

| config | tail mapping | acceptance | 1 req | 2 req | 8 req |
|---|---|---|---|---|---|
| 0.9.1 (DCP1, ring 8, K=3) | shipped | 0.6 | 0% | 57.3% | 89.8% |
| 0.9.1 | shipped | 0.85 | 0% | 47.1% | 77.5% |
| 0.9.1 | positions passed | 0.6 / 0.85 | 0% | 0% | 0% |
| 0.7.0 (DCP2, ring 4, K=5) | shipped | 1.0 / 0.85 / 0.6 | 100% | 100% | 100% |
| 0.7.0 | positions passed | 1.0 | 0.8% | 0.8% | 1.9% (boundary pools only) |
| 0.7.0 | positions passed | 0.85 | 32.5% | 27.0% | 25.9% |
| 0.7.0 | positions passed | 0.6 | 69.2% | 66.2% | 67.9% |

- The 0.9.1 rows reproduce sub-audit C's transcription (0% / 56% / 88%).
- 0.7.0 shipped: 100% on both ranks at every concurrency, including with every draft accepted, so the
  positions omission under per-rank CP masking is sufficient. At 1 request the slots read hold another
  position of the same request or were never written on that rank; from 2 requests, other requests' rows.
- 0.7.0 with positions: the remaining errors are the prompt-boundary pool (all 18/18; dense seed addressing)
  and rejected drafts overwriting committed ring slots (ring 4 < verify window 6); both fixed in 0.9.1.
- A Python shadow of the ring addressing classified every pool and agreed with the real-kernel outcome in
  every cell.
- Measured: the kernel outcomes on this harness. Inferred: that serving behaves the same (real scheduling,
  acceptance, GPU program races, stale warm-up columns and real block ids are not modelled).
- Reading (inferred): this defect affects every 0.7.0 decode-built pool and 0.9.1's only with 2+ requests, so
  it cannot explain 0.7.0's better completion by 0.9.1 being more affected. Greater corruption in 0.7.0 does not
  exclude an accidental completion benefit from incorrect computation either; completion effects are untested
  (sentence added 2026-10-11, amendment 12).

Files: `ring070/{two_rank_tailring.py,run_all.sh,v0*-a*.json,v0*-a*.log,summary.md}`.

## Repro folders

`../repro/col511/` and `../repro/tailring/` (README, patch, CPU scripts with outputs, GPU commands, evidence;
no local paths or names). Both `run_cpu.sh` scripts were run end to end on the release image.
