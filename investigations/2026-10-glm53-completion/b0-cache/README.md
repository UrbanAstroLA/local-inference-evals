# B0 cache isolation and high-offset addressing fixture (amendments 16, 16c-16e)

**Result: every driven kernel passed, after fixture faults found in the data were fixed.** With checkpoint B0's real
cache geometry (a 2.93 GB shared slot tensor whose pages cross 2³¹ bytes from block 935) and B0's own kernels, every
writer and reader driven stayed inside its own page, wrote exactly the expected bytes, and read back what was written,
at the lowest and highest blocks and on the record that straddles 2³¹. The first run with data (run 2) passed 15 of 18 parts byte-exact
against a full-size shadow image with sentinels intact; its four failures were three faults in the fixture and one
obsolete upstream test, none specific to high addresses. The fixture was corrected after inspecting those data (no
criterion or tolerance loosened) and the affected parts rerun twice; the second rerun passed all 80 checks. Cost
**0.0392 of 0.5 occupied GPU-hours**, plus one launch the runner's own pre-flight refused at zero cost. No B1.

Plan and rules: [`../PLAN.md`](../PLAN.md#dated-amendments), amendments 16 (second priority), 16c (authorization: one GPU,
ceiling 0.5 GPU-hours, no automatic retry), 16d (pre-flight fix before any data) and 16e (results). Design, geometry,
kernel audit and reachability: [`PROPOSAL.md`](PROPOSAL.md). Investigation page:
[checkpoint B0 and targeted invariants](../README.md#checkpoint-b0-and-targeted-invariants-amendment-16).

## What it tests

Whether B0's cache writers and readers keep to their own bytes, and address correctly above 2³¹ bytes, in the layout B0
actually serves: one BlockPool of 1,275 blocks shared by 10 KV groups. Per owned MLA layer, one **slot tensor** of
2,929,766,400 bytes is shared by the MLA records (528-byte records, page 2,297,856 bytes), the recurrent state of one KDA
layer in each of the 7 mamba groups (padded to the MLA page) and one DFlash2 draft layer (padded likewise); one **idx
tensor** (183 MB) holds the index keys and the kpool tail ring. Pages start above 2³¹ for blocks 935-1,274 (340 blocks,
26.7% of the pool); the record at slot 4,067,203 (block 934) straddles 2³¹. The hazard is real only for byte-unit
addressing (the MLA record writer and readers, the byte-copy kernels); element-unit views stay below 2³¹ elements.

**Reachability in serving** (CPU, inside the B0 image: the real allocator and B0's 10-group scheduler configuration,
driven by the per-request prompt and completion token counts of B0's two GPQA passes; [`reachability/`](reachability)):
every block id from 1 to 1,274 is handed out in both passes; ids at or above 935 are first reached about 9% into a pass;
cross-group reuse of the same block id is routine (about 10,700-11,400 reallocations per pass); no id was held by two
groups at once. The scheduler loop is approximated (2.6 accepted tokens per step; 2.0 and 3.2 tested).

**Method.** One real-size slot tensor and idx tensor on one GPU, every view built by B0's own code, both tensors filled
with a sentinel derived from (offset, salt) and mirrored by a same-size shadow image of the bytes that must be present.
After every operation the full 2.93 GB must equal the shadow; a differing byte is reported with its block, region and
decoded owner tag. Payloads carry group, block, token and epoch tags. Criteria were fixed in the fixture code before data
([`fixture.sha256`](fixture.sha256)). CPU validation before any GPU use: geometry against the server log, 107 view-offset
checks with B0's code on full-size meta tensors, sentinel detection of misplaced, overrunning and missing writes
including at exactly 2³¹ ([`cpu-tests/`](cpu-tests)).

## Results

| Part | Real B0 kernel driven | Run 2 | Rerun 1 | Rerun 2 |
|---|---|---|---|---|
| A | draft K/V writer `reshape_and_cache_flash` on the padded view, with padding rows (blocks 1, 934, 935, 1274) | pass | | |
| B | draft FlashAttention-2 paged reader, high block sets bit-equal to low; vs fp32 | pass | | |
| C1-C4 | KDA scatter/gather, conv update (decode, speculative) and prefill, fused recurrent (speculative, 4 columns), align precopy | pass (5 parts) | | |
| C5 | partial-prefix-hit page copy-on-write | pass | | |
| D1 | MLA record writer with vLLM's primed plan: low block, around the straddling slot, block 935, the last 64 records, a kernel-page crossing | pass | pass | pass |
| D2 | B12x sparse-MLA decode and extend readers on the same sets | **fixture fault**: extend at width 64, which the GLM_NEXT extend kernel rejects | **fixture fault**: extend returns (O, LSE) | pass: decode and extend at the serving width 2,048, decode at 64 |
| D3 | writer and decode reader captured once, replayed with new slots and selections (high, straddle, reuse), plus a control | **fixture fault**: compared the decode binding's own scratch output with itself | **fixture fault**: the fixture's writer replays overwrote its own reader sets | pass: each replay equals an eager read of the same selection; the control follows the new selection |
| E1, E2 | kpool tail seed; index-K compress and write | pass | | |
| E3 | kpool decode update (speculative, pool completion) | **fixture fault**: compared never-written tail-ring slots | pass | pass |
| F1 | new-block zeroer | pass | | |
| F2 | cross-group reuse of blocks 935 and 1274 (draft → MLA → KDA → KDA → draft → KDA → MLA), each owner reading only its own data | pass | | |
| G | draft writer captured in a graph, replayed with high, padding and reused slots | pass | | |
| P | upstream B12x tests above 2³¹ | GLM_NSA high-page tests: 2 passed. GLM_NEXT high-page test as shipped: **obsolete** (calls the writer without the now-required `plan=`); the repaired copy passed | not run | not run |
| | **checks** | 129, 2 failed (plus D2's error) | 65, 2 failed (plus D2's error) | **80, 0 failed** |

Part names are the fixture's own (A-G, P), unrelated to the defect ids D1-D6. Results: [`run-20261012T221551Z/`](run-20261012T221551Z) (run 2), [`run-20261012T222239Z/`](run-20261012T222239Z)
(rerun 1), [`run-20261012T222337Z/`](run-20261012T222337Z) (rerun 2); `results.json` per run, runner consoles
`run-2.out` to `run-4.out`.

### What went wrong in the fixture, and what was changed

Stated plainly: the fixture had five faults, all found from stored outputs and source **after** the data were inspected,
and all fixed in fixture code only. No criterion or tolerance was loosened, and no high-address-specific failure was
found. Run 2's results are not relabelled; the reruns are recorded separately
([`post-run2-changes.txt`](post-run2-changes.txt); hashes after the fixes in
[`fixture-post-run2.sha256`](fixture-post-run2.sha256)).

1. **Part D2, run 2.** The fixture asked the GLM_NEXT extend kernel for width 64; it supports widths 512-2,112, and B0 serves
   at 2,048. Fix: keep decode at 64, add 2,048-record sets (written and twin-checked) read by decode and extend at 2,048.
2. **Part D3, run 2.** Eager reference outputs aliased the decode binding's scratch output buffer, so the replay checks
   compared a buffer with itself and the control check could not pass. Fix: clone eager references; compare each replay
   with an eager read of the same selection.
3. **Part E3, run 2.** The twin check compared the whole tail block, including ring slots 0-3 that positions 12-15 never
   write and that hold offset-dependent sentinel. Fix: compare the written slots 4-7 and the exact stash bytes.
4. **Part D2, rerun 1.** The extend call returns (O, LSE); the fixture treated it as a tensor. Fix: take O.
5. **Part D3, rerun 1.** The fixture's own writer replays overwrote 16 records each of two reader sets, so the added twin check
   could not hold. Fix: restore the D1 latents on all reader sets (with writer, twin and shadow checks) before the reader
   graph section.

The fourth failure in run 2 was not the fixture's: the upstream B12x GLM_NEXT high-page test calls the writer with an
obsolete signature, as the proposal had predicted from source; a repaired copy
([`fixture/test_repaired_high_page.py`](fixture/test_repaired_high_page.py)) passed. The runner also gained options for
partial reruns and a per-run ceiling so that spent plus ceiling stays within the approved 0.5. No further reruns were
made.

### Pre-flight refusal and fix (amendment 16d)

The first launch was refused by the runner's own pre-flight: no container started, 0 GPU-hours
([`run-1.out`](run-1.out)). Its GPU-idle check counted the desktop's graphics client (type G, 8 MiB, 0% utilisation)
on GPU 0. Before any data, the check was changed to ignore type-G desktop graphics clients (gnome-shell, Xwayland,
xdg-desktop-portal) only; the memory (1 GiB) and utilisation (5%) limits and the refusal for any compute process or other
client are unchanged, matching the other fixture runners. Runner sha256 `0cb18df9…` → `d6fac94f…`. No criterion, kernel
call or fixture code changed.

## Cost

| Launch | Parts | Wall | Occupied GPU-hours |
|---|---|---:|---:|
| run 1 | refused by pre-flight, no container | 0 s | 0 |
| run 2 | all, plus upstream tests | 100 s | 0.0278 |
| rerun 1 | D1, D2, D3, E3 | 20 s | 0.0056 |
| rerun 2 | D1, D2, D3, E3 | 21 s | 0.0058 |
| **total** | | | **0.0392** (ceiling 0.5) |

One GPU, container start to removal ([`costs.jsonl`](costs.jsonl); the refused launch wrote no line because no GPU was
claimed). The proposal had estimated 0.17-0.33.

## What it rules down, and what it does not

Ruled down for the driven kernels, on B0's shipped geometry: 32-bit truncation of byte or element offsets at blocks
934-1,274 and on the record straddling 2³¹; writes outside a group's page or into padding (MLA, draft, KDA, kpool, zeroer,
copy kernels); padded-stride errors in the draft, KDA and tail views; stale data surviving reuse of a block id across
groups; replayed graphs writing or reading capture-time addresses (draft writer; MLA writer and decode reader). The DCP
audit's items D8 (draft slot sharing) and D9 (tensors above 2³¹ bytes) ([audit](../audits/phase1-dcp-attention.md)) are
retired for these kernels.

Not ruled down: block-table and slot-mapping construction under real scheduling, which the fixture bypasses (the class
the tail-ring defect D2 belonged to); the index-K readers (B12x paged top-k; the compiled prefill readers); extend under
graph replay; the mamba postprocess kernel (it shares its copy body with the tested precopy); numerical effects; rank 1
(same geometry per tensor, fewer tensors).

## Files

- [`PROPOSAL.md`](PROPOSAL.md): geometry, writer and reader audit with citations, reachability, fixture design, criteria,
  price, outcomes, secondary findings.
- [`fixture/`](fixture): geometry, sentinel/shadow and view modules, `run_fixture.py`, `inner.sh`,
  `run_b0_cache_fixture.sh`, CPU tests and the repaired upstream test, **as they stand after the fixes**. The versions that
  ran in run 2 and rerun 1, and the runner before the pre-flight fix, are kept locally and hashed in the raw list; their
  hashes also appear in [`fixture.sha256`](fixture.sha256), the per-run `fixture.sha256` files and
  [`post-run2-changes.txt`](post-run2-changes.txt).
- [`reachability/`](reachability): `sim_pool.py` and its outputs (token counts only, no request content).
- [`cpu-tests/`](cpu-tests): CPU validation and the import smoke test.
- Per-run folders: `results.json` (every check), `fixture.log`, `exit.txt`, `fixture.sha256`, `gpu.txt` (GPU id shortened),
  upstream test output (run 2).
- [`RAW-SHA256SUMS-cache.txt`](RAW-SHA256SUMS-cache.txt): sha256 of local files not published (fixture backups, bytecode,
  empty container logs).
- Public copies replace local paths with `<runs>` and `<gpqa-runs>` and shorten GPU ids, so `PROPOSAL.md`,
  `run_b0_cache_fixture.sh`, three runner consoles and the `gpu.txt` files are not byte-identical to the local files;
  [`../PUBLIC-COPIES-16.txt`](../PUBLIC-COPIES-16.txt) lists both hashes. The amendment 16d runner hash and the
  post-run hashes refer to the local files.
