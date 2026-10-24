# replay-topk: self-critique (written with the package, before any GPU data)

For each reading: what could make it spurious, and which gate or test covers it. "Covered" names the mechanism.
"Residual" is what stays open.

## R1: top-k repeatability

| Risk | Effect if unchecked | Covered by | Residual |
|---|---|---|---|
| Synthetic logits lack the real value and tie structure | Variation rate or set stability unlike serving | Regime K uses the served producer (DeepGEMM `fp8_fp4_mqa_logits`) on FP8 inputs with ue8m0 scales: real granularity. Regime T forces exact ties. Ties at the threshold are counted per row | The real distribution (indexer weights, prompt) is not reproduced. A REPEATABLE reading on synthetic logits would not exclude variation on real ones |
| Wrong window rule (`cu_seqlen_ks/ke`) | Wrong rows select; the 2,051 boundary moves | The image's own `build_prefill_chunk_metadata`, in the run and in CPU test S3; gate "window = rule"; geometry gate (2,039 selecting rows, first at 2,051) | none known |
| Wrong logits strides (served tensor padded?) | A different memory walk than served | K records the served tensor's real shape and strides; C and T are copied into those strides. If K fails, the strides are flagged ASSUMED (caveat) | Strides cannot change atomic ordering by themselves; only fidelity is at stake |
| Wrong argument order or semantics | Garbage rows that "vary" | Schema read from the `_C` binary (S2); fp64 reference-set agreement on untied rows (gate); shortcut-row negative control (gate) | none known |
| Warm-up not done (first-call effects, lazy module load) | First-call artefacts read as variation | 3 warm-up calls before the reference call (gate) | |
| Input reused or mutated across calls | Apparent variation from changed input | Bitwise input check after every call (gate) | |
| Output buffer not re-initialised | Stale entries look like reordering | Re-filled with -1 before each same-allocation call, as production's fresh `torch.full(..., -1)`; relocated calls use fresh buffers | |
| Measuring after a sort | Order variation hidden | Order is compared on raw outputs. The set is compared on sorted copies, separately. Pattern statistics are computed on raw outputs | |
| Relocation not achieved (caching allocator hands back the same block) | B-ALLOC unreadable | Dummy allocations shift the allocator; addresses recorded; gate "≥ half the relocated calls at new addresses" | |
| Comparator blind | False REPEATABLE | Planted swap and replacement must be flagged on exactly those rows (gate) | |
| Different stream than serving | Different scheduling | Default stream, synchronised per call. Atomic slot order depends on warp scheduling inside one kernel, not on the stream | An idle GPU may vary less than a busy one: REPEATABLE means "in N calls on an idle GPU" |
| N too small to see rare variation | False REPEATABLE | 24 + 24 calls (v0.9.1). capture-p0 saw the difference in all 6 pairs, so the serving rate is high | A low-rate effect could be missed; the wording says "in N calls" |
| The producer itself varies | Set differences blamed on top-k | K is computed twice on identical inputs; the in-window bitwise equality is recorded (informational) | Real logits are not tested |
| v0.7.0 geometry | Over-reading the v0.7.0 R1 | Labelled a binary confirmation at the v0.9.1 geometry. Its served per-rank top-k never selects in this chunk (≤ 511 local pools) | v0.7.0's served selection (DCP merge) is not replayed (PROTOCOL section 9) |

## R2: sparse-MLA prefill order sensitivity

| Risk | Effect if unchecked | Covered by | Residual |
|---|---|---|---|
| Not the served kernel or config | Wrong conclusion about the served path | Caps keyword set equals the served `_make_plan` call (AST, S4); values follow the served config (S4, S6); v0.9.1 prepares through vLLM's own `_prepare_b12x_sparse_mla_plan`; MG is the only prefill path (v0.9.1 records `prefill_mg_enabled`); cited files are hash-checked (S1) | First GPU contact with the extend plan is the run itself. The decode plan path on this image ran in captured-p3b |
| Wrong cache record format | The kernel reads garbage | Records written by the served writer (GLM_NEXT 528 B / `concat_and_cache_mla` 656 B); decode of records vs the source latents (caveat if > 10%) | |
| Index set or length changes with the variant | Set change read as order sensitivity | Per variant, `same_set` and `same_nsa` (gate) | |
| Variant did not change order | False ORDER-INSENSITIVE | `rows_index_order_changed > 0` per variant (gate) | |
| Kernel nondeterministic | Noise read as order sensitivity | 8 same-order repeats and a repeat of each variant; A-EXEC takes precedence over every order reading | |
| Hidden reads (scratch, unwritten cache records, layout) | Allocation effects read as order effects | Scratch poisoned (0xFF, 0x00); unwritten records poisoned differently; relocation to the same and to other pages; any difference gives B-ALLOC | |
| Scratch zeroed where production does not zero it | Masked initialisation defects | Scratch is `torch.empty`, as the served workspace; poisoning is a separate measurement | |
| Rows with identical inputs differ | Instrument fault | Shortcut rows have identical inputs in every variant; a difference gives INCOMPLETE | |
| Comparator or index path dead | False ORDER-INSENSITIVE | Positive control: a gross set shift must change ≥ 90% of selecting rows | |
| Positive control too weak under peaked attention | Spurious INCOMPLETE | The shift changes almost every attended token, not one low-weight token | Not GPU-proven yet |
| Synthetic Q / KV magnitudes | Over- or under-stated effect size | Two Q regimes; magnitudes reported next to the fp32 reference error of the same kernel | Serving magnitudes are not reproduced |
| `base` / `atomic2` threshold-block size (16) assumed | Pattern not the kernel's | asc, desc and random permutations cover the order space; R3 uses the kernel's own orderings | |
| v0.7.0: one rank only, v0.9.1 geometry, post-filter permutations | Over-reading v0.7.0 | Stated in the outputs and the protocol. The filter's own repeatability and order preservation are measured separately | The cross-rank LSE combine and the DCP merge are not replayed |

## R3: composition

| Risk | Covered by | Residual |
|---|---|---|
| Logits unrelated to Q / KV | Stated: mechanism only, synthetic | Real-input composition needs captured inputs (below) |
| Substituting another regime when K fails | Pre-registered: R3 needs regime K; otherwise not run (INCOMPLETE) | |
| Output differs for a reason other than the top-k | Gate: ANOMALY if any output row differs without a top-k difference | |

## Runner and budget

| Risk | Covered by |
|---|---|
| Running without approval, or over it | `APPROVED_GPU_H` required (no default); full caps reserved; tested refusals: unset, NaN, inf, 0, text, too small |
| Edited package after the tests | `sha256sum -c SHA256SUMS` before any container (tested on a tampered copy) |
| Wrong image | Digest check against `runner/images.txt`; `--pull=never` |
| GPU not idle, including graphics clients missed by the compute-apps query | `preflight_gpu.py` reads the full process table; tested with a compute process, a large graphics client, busy and full GPU |
| Cost not recorded on failure or interrupt | One ledger line per container on every exit path; interrupt tested (rc 130 line, container removed, handled within seconds) |
| Re-running a valid reading | Refused; an INCOMPLETE attempt needs `REPLAY_PRIOR_ACK` |
| Compile cache taken from serving | Copied, never written back; a hit is the same compiled artefact for the same key; regenerable by deleting `cache/` |

## Open items

1. **The extend-plan path and the DeepGEMM call have not touched a GPU.** CPU tests cover:
   - imports, Caps, index construction and the window, through the served Triton helpers under the interpreter;
   - the harness logic, with stand-ins.

   A failure at run time gives INCOMPLETE and costs at most the cap.
2. **Cold compile time is an estimate.** If both images compile cold near the top of the range, the soft deadlines
   stop the R2 regimes early (INCOMPLETE, never negative).
3. **Real-input follow-up (not in this package).** capture-p0 retained, under its PRIVATE storage:
   - layer-3 MLA KV bytes for blocks 0-2;
   - the real top-k rows of the first differing positions.

   A PRIVATE replay could feed real KV and real orderings through R2's v0.9.1 path, with synthetic Q because the
   query projection is not retained. It needs its own protocol and a PRIVATE output path.
4. **v0.7.0's served selection** (DCP merge) needs a two-rank replay or a post-gather kernel call; each needs a
   separate estimate.
5. **Ordering inside atomics vs a quiet GPU.** If R1 reads REPEATABLE on the idle GPU while serving varied, the next
   question is contention: the same calls with a concurrent load. That is not pre-registered here.
