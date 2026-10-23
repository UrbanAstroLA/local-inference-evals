# Coverage map: what each check exercised, by version and dimension (recon, 2026-10-22)

This replaces a standalone recalibration of the first-contrast gates. Missing captures stay missing. Repeating
synthetic trials on records that were never captured does not stand in for independent real measurements, so
nothing here is recomputed.

**Live** means a serving engine on GPUs. **Model-free** means kernels driven with fixture inputs. **CPU** means
the Triton interpreter or a CPU harness. **Real** means captured from a serving session. **Synth** means generated.

## A. By instrument

| Instrument (amendment) | Image | Live / model-free / CPU | Contexts | Concurrency | Ranks | Graph modes | Branches / boundaries covered | Not covered |
|---|---|---|---|---|---|---|---|---|
| Phase 1 decode-vs-prefill, steps 1-6 (1-7) | stock 0.7.0, stock 0.9.1, 0.7.0 + kpool, 0.9.1 ×5, 0.9.1 + 0.7.0 layout | live | ≤ 16k; the 8-16k bin is forced continuation after the reply had ended | 1 | both (served) | FULL decode | fused top-k route only (≤ 2-4 requests); Trellis MoE only (≤ 32 rows) | concurrency 8 shapes: tiled route, 0.7.0's prefill-plan MoE; contexts > 16k |
| Column-511 stages 1-2, gate (10a-11b) | 0.9.1 (+ checker) | live, eager then shipped graphs | GPQA-like | served | both | FULL (gate) | tiled / fused top-k selection, 748k decode + 15.6M prefill rows (gate) | stock 0.7.0 served path |
| Tail-ring guard, fixes T/U (10d-11) | 0.9.1 derivatives | live + GPU unit test | short | served | both | FULL | tail-ring writes (D2) | 0.7.0 served (CPU harness only) |
| D2 two-rank harness (10e; 22l p2-exposure) | stock kernels of 0.7.0, 0.9.1, B1 | CPU, real kernels | 160 generated tokens per request | 1, 2, 8 | 2 (emulated) | n/a | wrong-pool shares by acceptance 0.6 / 0.85 / 1.0 | real scheduler, keys and drafts (synthetic) |
| Stage A verify13 (13-14) | 0.9.1 + capture | live | GPQA prefill / mixed | served | both | **eager only (mode NONE), M 150-2,045** | router fp32 effect (D3); D4 plan carries no limit | decode-graph MoE rows; gate/up pre-activations never captured |
| Bundle gate (14-14d) | B0 check images | live | short | served | both | FULL + eager | D1-D4 present in compiled keys; limit-1 reachability (6/6 probes differ) | effect of limit 10 |
| All-reduce fixture (16b) | B0 | model-free, live GPUs | n/a | n/a | 2 | captured graph | B12x TP2 push, 1-48 rows, 2,040,800 calls bit-exact | 0.7.0's old pull protocol; PyNCCL above 48 rows (cross-check only); contention with model kernels |
| KV-writer comparison (16b, 16f) | B0 vs 0.7.0 | model-free | n/a | n/a | — | eager | 528 against 656 B record writer and query projection, real-weight proxies | latents captured from a server |
| Cache fixture (16c-16f) | B0 | model-free, live GPUs | blocks 934-1,274 | n/a | rank 0 tensors | eager + graph replay | draft writer `reshape_and_cache_flash` at **fixture-chosen** slots, the FA2 draft reader, 2³¹ offsets, shared-slot sentinels | scheduler-built draft slot mappings; `_prepare_dflash_inputs_kernel`; rank 1 tensors |
| Mapguard B0 a3 / B1 (17-18c) | B0 / B1 + guard | **live** | about 1.8-4.5 min workload: 8 stage-2 prefixes (long-prefix phases), 16 GPQA prompts with max_tokens 1,024 | 8 | both | FULL decode + eager / piecewise prefill | MLA, index-K, tail writers and readers, positions, ownership, 10-group block exclusivity; D7 found | **draft group writer and slot mapping; KDA states**; contexts past the workload's; stock images |
| K selection replay (21b-21d) | stock 0.7.0, B1 kernels | model-free | 2k-331k | n/a | per rank | n/a | B1 34 / 34 exact; 0.7.0 tiled atomic-slot merge launch-dependent | **0.7.0 fused cells INVALID-ROUTE** (no CTA count exposed); stock 0.9.1 (differs from B1 by D1) |
| Hook validation a1 / a2 (21e-21j) | B1 + hook | live | P about 3.8k, windows 256-2,000 | fixture-specific | both | FULL decode | forcing, KDA decode snapshots (SSM as a 256-dim **sketch**), V4 index-K window dumps with a real D2 injection (a2) | long contexts; cached KDA block at the boundary |
| First contrast fc1 / fcr1 (22-22j) | B1 + hook | live | **P about 83-113k, windows about 8.2k** | 8 | both | FULL | C1-3, D1-3, LW1-2 window dumps (157,696 units per run); PS-a / b / c | PS: 2 pre-injection + 1 post-injection only; SSM sketch only; **cached KDA block at c never read**; no same-build clean repeat; fc1 stopped at GATE-1 (C1, LW1, LW2 only) |
| Port gates, 5 PRs (2026-10-22) | stock 0.9.1 (CPU) | CPU | n/a | n/a | n/a | n/a | each defect reproduced unpatched (`--expect-unpatched`) and absent patched | GPU, DCP2, SM121 |
| Draft-slot reproduction (this recon) | stock 0.9.1 (CPU) | CPU, real allocator + real kernel | up to 7k prompts, 5k generated | 1, 2, 4, 8 | n/a | n/a | see `../draft-slot/README.md` | GPU stream ordering of context vs query writes; FULL-graph replay of the drafter; rank-local view |
| GPQA records and fresh passes | stock | live | to 327,680 | 8 | both | as served | outcomes, token counts, finish reasons; acceptance aggregates per 10 s | per-step dispatch; record reply text (only rep070, D, D2 kept text) |
| B+ / E continuations (22s-22ak) | stock 0.7.0, stock 0.9.1, B1 | live | 100k prefixes + 64k | 8 | both | as served | outcome by prefix; acceptance per session | decode-built state up to 100k (prefill reconstructed) |

## B. By version: what has instrumented live coverage

| Version | Live instrumented (guards / captures on the served path) | Only CPU / model-free / source |
|---|---|---|
| **Stock 0.7.0** | **none** (Phase 1 measured outputs only, at concurrency 1) | D2/P2 harness; K replay (fused cells invalid); source audits |
| **Stock 0.9.1** | verify13 (eager prefill captures); `gate/` (checker image, D1 counters) | D2 harness; port gates; draft-slot reproduction; source audits |
| **B0 / B1** | all-reduce, cache fixture, mapguard, hook validation, first contrast | K replay (B1) |

The engine pair the question is about, stock 0.7.0 against stock 0.9.1, has the least live instrumentation. Nearly
all invariant evidence is on B0 / B1 derivatives.

## C. By dimension: gaps that stay gaps

| Dimension | Covered | Not captured |
|---|---|---|
| Ranks | both, in mapguard and hook runs | rank 1 tensors in the cache fixture |
| Graph modes | FULL target decode (mapguard, gate); eager prefill (verify13) | **drafter FULL-graph replays** (never observed); stock-image per-step graph mode |
| Batch-shape branches | fused / tiled top-k (gate; harness); Trellis MoE at ≤ 32 rows | **MoE prefill plan at 33-48 decode rows (0.7.0 at 6-8 in flight)**: never instrumented or compared numerically |
| Boundaries | unaligned prefill chunk starts (D7); 2³¹ offsets; block 4,352 crossings in the KDA metadata (H4-meta, explained 21k) | **KDA convolution state at the block boundary** (21k); cached KDA block at c (VALIDATION-DESIGN §4) |
| Context length, live instrumented | up to about 121k (fcr1 windows) | 121k-327k on any instrumented engine; model-free replay only (K, to 331k) |
| Concurrency | 8 (GPQA, mapguard); 1 (Phase 1) | never varied as a treatment in GPQA |
| State construction | prefill-reconstructed (B+, E, hook) | **decode-built state past 16k on stock engines**: not captured anywhere |
