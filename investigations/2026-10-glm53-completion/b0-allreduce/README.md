# B0 two-GPU all-reduce fixture (amendments 16-16b)

**Result: CLEAN.** The all-reduce transport that checkpoint B0 dispatches under its shipped CUDA graphs (B12x PCIe
one-shot, TP2 peer-push) returned the exact bf16 sum on both GPUs in every call: **2,040,800 peer-push all-reduces over
26,650 graph forwards, 41 cases, 0 mismatched elements on either rank**, against an exact reference fixed before data.
The transport's epoch counter advanced by exactly the number of scheduled peer-push calls in every phase, so every
intended call, and no other, ran on that transport. 900 forwards through the in-graph PyNCCL reference also matched
exactly. No hangs. Cost **0.07 of 0.5 occupied GPU-hours** (smoke 0.0139, full 0.0561). No B1 follows.

Plan and rules: [`../PLAN.md`](../PLAN.md#dated-amendments), amendments 16 (first priority), 16a (authorization: smoke
then full, ceiling 0.5 GPU-hours including initialization, compile and cleanup; verdicts fixed in `analyze.py`) and 16b
(result). Design, dispatch audit and pre-data rules: [`PROPOSAL.md`](PROPOSAL.md). Investigation page:
[checkpoint B0 and targeted invariants](../README.md#checkpoint-b0-and-targeted-invariants-amendment-16).

## What it tests

Checkpoint B0 is the correctness bundle (D1-D4) as gated and screened: image
`sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548`. None of the five bundle files is on the
communication path. The proposal read B0's server logs, environment and source to establish what B0 actually dispatches
for tensor-parallel all-reduces (hidden size 4096, bf16):

| Rows (tokens) | Execution | Transport |
|---|---|---|
| 1-48 (up to 384 KB) | FULL and breakable PIECEWISE graphs | B12x TP2 peer-push (six launch geometries) |
| 49-64 | FULL and PIECEWISE graphs | PyNCCL, captured in the graph |
| above 64 (prefill, up to 2,048) | eager | PyNCCL |

The fixture drives that path end to end through vLLM's own communicator and adapter in the unchanged B0 image (plan,
preparation, capture and kernels are the shipped code), on both GPUs, with no model. B12x's own GPU tests for this
transport call an interface that no longer exists and use weak checks (constant fills, `assert_close` at 1e-2); the
fixture replaces them rather than repairing them in the image.

- **Inputs** change every replay and identify their source: per-rank tables of exact integers (sums exact, so a peer
  value can be read back) and of normal values needing rounding, indexed by a hash of (step, layer) computed in the
  graph; one rank contributes zero on every third layer, as the owner and embedding all-reduces do.
- **Execution** mirrors the engine: one capture context, an eager warm-up per shape that prepares the plan, capture in
  vLLM's order, 91 all-reduces back to back per graph (as in a target forward), input-address reuse, GEMMs between
  calls, replays back to back on one stream with a host synchronisation only every 256 replays.
- **Failure classes** the analysis can name (CPU-validated by injection, T4 and T7): arithmetic, wrong-rank,
  missing-peer, stale same or other slot, cross-step, addressing, local-stale, torn pack, unexplained, hang.

## Coverage (full profile)

| Phase | Content | Forwards | B12x peer-push calls | Epoch delta |
|---|---|---:|---:|---:|
| main | FULL 4-48 rows ×2,000 each; FULL 56/64 ×200 (PyNCCL); PIECEWISE 1-48 ×600 each; PIECEWISE 56/64 ×100 | 20,000 | 1,765,400 | 1,765,400 |
| skew | FULL 1, 4, 16, 24, 48 with a rank-alternating delay (about 25 µs) before every other all-reduce | 1,500 | 136,500 | 136,500 |
| boundary | FULL rows 3, 5, 6, 7, 9, 14, 15, 17, 22, 23, 25, 47 (every launch-geometry edge) ×300; 49 rows ×50 (switch to PyNCCL) | 3,650 | 111,600 | 111,600 |
| mixed | a B12x FULL replay, then an eager 2,048-row PyNCCL prefill all-reduce (checked), ×300 | 600 | 27,300 | 27,300 |
| nccl-ref | FULL 1, 8, 48 with in-graph PyNCCL | 900 | 0 | 0 |
| **total** | 41 graphs; 2.05·10¹¹ checked elements | **26,650** | **2,040,800** | **2,040,800** |

Rank 0 shown; rank 1 identical. Dispatch per graph, launch geometry per row count, backend set and transport policy
matched the proposal's audit (any difference would have made the run INVALID). The smoke profile (9 cases, 175 forwards
including warm-ups) ran the same frozen code first and was also CLEAN; no fixture fix was needed.

## Reference and tolerance (frozen before data)

- Each element must equal `(a.float() + b.float()).to(bf16)`: one round-to-nearest-even of the exact sum of the two
  bf16 inputs, which the CPU checks confirmed against exact rational rounding on 40,000 generated and adversarial pairs
  (T2). **Exact bitwise equality, no tolerance**, on both ranks; the two ranks' results must also be identical.
- Alternative validated transport: PyNCCL (NCCL 2.30.7) on the same graphs, patterns and checks. A mismatch on any
  PyNCCL arm would have made the run INVALID (a reference or fixture problem), not a B12x verdict.
- Verdicts fixed in `analyze.py` before the smoke run: INCOMPLETE (error, hang or wall cap), INVALID (dispatch,
  geometry, epoch or reference disagreement, or a PyNCCL mismatch), FAIL (any B12x-dispatched mismatch on either rank),
  CLEAN otherwise. No re-run to confirm a CLEAN.
- CPU checks before any GPU use: 29 of 29 passed (T1-T7; [`runs/cpucheck-20261012T214610Z/`](runs/cpucheck-20261012T214610Z)).

## Cost

| Run | Wall | GPUs | Occupied GPU-hours |
|---|---:|---:|---:|
| smoke | 25 s | 2 | 0.0139 |
| full | 101 s | 2 | 0.0561 |
| **total** | | | **0.07** (ceiling 0.5) |

Container start to removal, 2 GPUs ([`costs.jsonl`](costs.jsonl)). The proposal had estimated 0.17-0.33.

## What it rules down, and what it does not

Retired, for B0's shipped graph all-reduce transport on this host (two RTX PRO 6000 Max-Q on PCIe, NCCL 2.30.7): stale,
torn, cross-step, wrong-rank, addressing and slot-parity errors and hangs, at every engine graph size and every
launch-geometry edge, in FULL and breakable-PIECEWISE capture, with back-to-back replays of 91 calls, input-address
reuse, rank skew and interleaved eager PyNCCL prefill. The arithmetic equals the exact reference and PyNCCL bit for bit,
so substituting the transport could not change numerics on these shapes.

Not excluded: contention with real model kernels and PCIe traffic beyond the GEMM and copy interleave; the drafter's
exact call sequence (inferred to use the same shapes); epoch rollover (about 4.3·10⁹ calls away, unreachable in a
serving session); the other communication paths (PyNCCL above 48 rows is only cross-checked here; the owner exchange);
numerical effects elsewhere in B0.

## Files

- [`PROPOSAL.md`](PROPOSAL.md): dispatch audit with source citations, reference, obsolete upstream tests, fixture design,
  failure classes, verdict rules, cost estimate.
- Scripts, published unchanged and frozen in [`SHA256SUMS`](SHA256SUMS): [`fixture.py`](fixture.py) (GPU fixture),
  [`patterns.py`](patterns.py) (inputs, reference, classification), [`analyze.py`](analyze.py) (verdict),
  [`cpu_tests.py`](cpu_tests.py) (T1-T7), [`run.sh`](run.sh) (pre-flight, budget gate, wall cap, cost line, cleanup).
- [`runs/smoke-20261012T215014Z/`](runs/smoke-20261012T215014Z) and [`runs/full-20261012T215101Z/`](runs/full-20261012T215101Z):
  `RESULT.md` and `RESULT.json` (verdict, per-case mismatches, phases, plans, dispatch), `plan.json`, per-rank
  `result.json`, `progress.json` and `status.json`, GPU utilisation log, budget and post-flight lines, the scripts'
  hashes as run. [`runs/cpucheck-20261012T214610Z/`](runs/cpucheck-20261012T214610Z): CPU checks.
  [`runs/preflight-20261012T214143Z/`](runs/preflight-20261012T214143Z): the runner's earlier no-GPU self-test (budget
  gate only; it recorded an earlier draft of the Python scripts, before they were frozen).
- `smoke-1.out`, `full-1.out`: runner console output.
- [`RAW-SHA256SUMS-allreduce.txt`](RAW-SHA256SUMS-allreduce.txt): sha256 of local files not published (pre-flight
  process listings, container logs, the per-rank step schedules, the compile cache).
- Public copies replace local paths with `<runs>`, so `PROPOSAL.md`, `costs.jsonl` and the two console files are not
  byte-identical to the local files; [`../PUBLIC-COPIES-16.txt`](../PUBLIC-COPIES-16.txt) lists both hashes. The scripts
  are byte-identical to the frozen files.
