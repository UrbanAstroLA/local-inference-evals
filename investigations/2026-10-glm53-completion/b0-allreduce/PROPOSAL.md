# B0 TP2 all-reduce correctness fixture — proposal (PLAN.md amendment 16, first priority)

Status: **PREPARED, NOT RUN.** CPU-only preparation: no GPU, no server, no image change. This
document asks for one smoke run and one full run of a model-free two-GPU fixture, within an
initial ceiling of **0.5 occupied GPU-hours**. Plan text: `PLAN.md` amendment 16 (unchanged).

Checkpoint B0 = image `sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548`
(FROM release a9e985c0; B12x 7fcc094e, `/opt/b12x` package version 1.3.0). The bundle diff
(949b2301…) touches five vLLM files (gate_linear, exl3, sparse_attn_indexer_kpool, mla/indexer,
mamba_hybrid). None of them is on the communication path. The B12x tree and
`vllm/distributed` copied out of the B0 image are byte-identical to the 0.9.1 extraction used
by the earlier audit (`diff -rq`, pycache excluded).

Citations: `b12x:` = `/opt/b12x/b12x/comm/pcie/` in the B0 image; `vllm:` = the image's
`/usr/local/lib/python3.12/dist-packages/vllm/`. Copies are under
`$S/b0-allreduce/src/` (scratch). **INFERRED** marks statements not read directly from code or logs.

---

## 1. What B0 dispatches (confirmed)

**Configuration, from the B0 server logs** (gate `bundle/gate-runs/s1-d1234-a1/server.log`,
`screen/run-d1234/server-s1.log`, `screen2/run-s1235/server-s2.log`; all three are identical
in these lines):
- L57: `Enabled B12x PCIe one-shot all-reduce (world_size=2, max_size=393216 bytes, graph_channel=single, eager=False)`.
- L58: `Using ['B12X_PCIE_ONESHOT', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'`.
  The `ep:0` group uses `['PYNCCL']` only.
- L43: `Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1`.
- L47: `CompilationMode.NONE`, `cudagraph_mode FULL_AND_PIECEWISE`,
  `cudagraph_capture_sizes [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]`, `max_num_batched_tokens 2048`.
- Graphs captured: PIECEWISE (11 graphs), FULL (9 graphs), and dflash2 FULL (9 graphs).

**Environment** (recipe and container inspect):
- `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_ALLREDUCE_BACKEND=b12x`,
  `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=384KB`, `VLLM_B12X_PCIE_EAGER=0`.
- No `B12X_PCIE_*` variable is set. The plain TP2 peer-push therefore takes its default, which is **on**
  (b12x `pcie_oneshot.py:137-148`).
- CPU check T5, run against the real function: `_transport_policy_contract()` = (push F,
  fused-TP2 F, **plain-TP2 T**, TP4 F, TP8 T).

**Selection rule** (CPU-checked against the real functions, T5):

1. vLLM `cuda_communicator.py:299-358` tries backends in order: symm-mem NCCL, quick-reduce,
   FlashInfer, AITER, `ca_comm` (= `B12xPcieAllReduce`, :118-137), then PyNCCL. All slots before
   `ca_comm` are absent or disabled on this machine (log L55-58).
2. `B12xPcieAllReduce.should_custom_ar` (vllm `b12x_pcie_all_reduce.py:115-134`) returns False
   unless one of these holds: `VLLM_B12X_PCIE_EAGER`, the adapter's `capture()` context is
   active (:181-199), or the stream is capturing. During capture it also requires a plan
   prepared before capture (:132-133, else PyNCCL). It also requires
   `should_allreduce` (b12x `pcie_oneshot.py:2888-2900`): dtype in {fp16, bf16, fp32}, bytes
   ≤ max_size (393216), bytes % 16 == 0, and weak-contiguous.
   - The adapter does **not** use `should_route_plain_allreduce` (:2902-2933, the 64-row
     extension).
3. Hidden size is 4096 and dtype bf16 for every TP all-reduce. These are the attention output (owner MLA via
   `placement.py:109-117`, KDA o_proj), MoE/MLP output (`moe_runner.py:451-487`) and the split
   embedding (`placement.py:189-205`). **384 KB = 48 rows.**
   - Rows ≤ 48 → B12x.
   - Rows 49-64 → PyNCCL, captured in-graph.
   - Prefill and mixed batches above 64 tokens run eager → PyNCCL (eager B12x is off).
4. Inside B12x (`_plain_launch_config` :1581-1600 → `_tp2_plain_remote_push_geometry`
   :1535-1579), a call takes peer-push when all of these hold:
   - world size 2
   - plain policy on
   - graph channel: capturing, or the device slot was already frozen
   - fp16/bf16, last dim 4096, rows ≤ 512
   - bytes ≤ eager_buffer_bytes (393216)

   Otherwise it takes the "pull" transport. Eager calls before the first capture use pull with host-alternated slots
   (:1797-1812). From the first capture on, `device_slot_selection` is frozen with
   `slot_bias = eager_slot & 1` (:1382-1391; prepared path `_prepared_launcher` :3008-3016). Every later call, eager or replayed,
   then uses the device-epoch slot.

**Coverage that results:**

| execution | tokens (rows) | transport |
|---|---|---|
| FULL decode (uniform 4/request, DFlash2 ×3) | 4, 8, 16, 24, 32, 40, 48 | B12x TP2 peer-push |
| FULL decode | 56, 64 | PyNCCL in-graph |
| PIECEWISE (breakable; mixed ≤ 64 tokens) | 1, 2, 4, 8, 16, 24, 32, 40, 48 | B12x TP2 peer-push |
| PIECEWISE | 56, 64 | PyNCCL in-graph |
| dflash2 drafter FULL (9 sizes) | INFERRED: same token counts as FULL decode | as above by rows |
| eager prefill / mixed > 64 tokens (chunks ≤ 2048) | up to 2048 | PyNCCL eager |
| startup warm-ups inside the capture context | each prepared size | B12x eager: pull before the first capture, peer-push after |

**Launch geometry per row count** (`:1535-1579`; T5 checks rows 1-48 against the real code):

| rows | variant, threads × CTAs |
|---|---|
| 1 | push 64×36 |
| 2 | stream 128×36 |
| 3-4 | stream 256×36 |
| 5-6 | push 64×16 |
| 7-8 | push 256×32 |
| 9-14 | push 64×16 |
| 15-16 | stream 64×36 |
| 17-22 | push 128×16 |
| 23-24 | stream 128×32 |
| 25-48 | push 128×16 |

The engine sizes therefore use six of the seven variants. All TP all-reduces in a process share
**one channel** (`single_channel=True, max_concurrent_channels=1`, vllm `b12x_pcie_all_reduce.py:86-93`): target and drafter graphs, all
sizes, the eager warm-ups, and one device epoch counter. INFERRED (structure count): about 91 TP
all-reduces per target forward (45 layers × attention + MLP/MoE, plus the embedding) and about
11 per drafter forward. Both counts are odd, so slot parity alternates across replays.

**Protocol actually executed** (b12x `_oneshot_cute.py:287-485`, `_cute_intrinsics.py`):
- Each 16-byte input pack is sent as two 16-byte `st.relaxed.sys.v4.u32` lines
  {w0, tag, w1, tag}, {w2, tag, w3, tag} into the peer's incoming slot (:385-429).
- tag = epoch + 2. The epoch is per channel, read at kernel start and advanced by the last CTA
  (`plain_graph_epoch_arrive`, intrinsics :544-588).
- The slot is (epoch + bias) mod 2.
- The receiver spins on `ld.relaxed.sys.v4` until all four tags match (:430-448). It then adds
  `add.rn.bf16x2(local, peer)` (:449-458; intrinsics :233).
- The reuse of a slot two calls later is protected only by stream order: a rank cannot finish
  call n+1 until its peer has started n+1.
- Payload and tag share each 8-byte half of a line. The design therefore relies on {word, tag}
  pairs never being observed torn. That is the NCCL LL assumption, which the source comment cites (:391-396).
  PTX guarantees single-copy atomicity only per 32-bit element of a vector access (INFERRED from the PTX
  memory model, not measured).
- At epoch 0xFFFFFFFD the kernel drains both ranks, clears both incoming slots and wraps
  (:469-485). That point is about 4.3·10⁹ calls away, which a serving session never reaches.
  INFERRED: about 5·10³ calls/s gives 10 days.

**Current API** (what the fixture drives, unmodified):
- `CudaCommunicator(cpu_group=gloo, device, device_group=nccl, unique_name="tp:0")` builds
  `B12xPcieAllReduce`.
- That calls `OneshotAllReducePool.from_exchange_group(..., eager_buffer_bytes=max_size=393216, single_channel=True)`.
  This allocates IPC slabs per channel: signal 150 528 B plus 2 eager slots × (1 base + 2 incoming) shards
  × 384 KB (`_eager_storage_shards` :179-199), exchanged over the NCCL group. Pointer tables are
  kept in `rank_data`.
- Per shape, `plan(query_from_runtime(channel, surface="OneshotAllReducePool.all_reduce", call={"inp": x}), runtime=channel)`
  is followed by `prepare_b12x_plan` (vllm `b12x_preparation.py:58-83` → `b12x.preparation.prepare_default`).
  This compiles three launcher variants (eager pull, device-slot bias 0 and 1;
  `_oneshot_preparation.py:269-305`) and installs `_PlainGraphLaunchPlan` (`state.bind_input`).
- Run path: `runtime.all_reduce(x, plan=plan)` → `_run_prepared_plain` (:3018-3032). The output
  is `torch.empty_like(x)`.
- Capture: `ca.capture()` → `pool.capture()` → `channel.capture()`. Capturing a shape that was
  never prepared fails in B12x (legacy path :1866-1883; the prepared path needs a bound plan); the vLLM adapter falls back to PyNCCL first (vllm :132-133).

## 2. Reference and tolerance (frozen before data)

**Semantics.** Each element is out = RN_bf16(a + b): one round-to-nearest-even of the exact sum
of the two bf16 inputs. Operand order is local + peer, and addition is commutative, so both
ranks must hold **bitwise-identical** results. Signed zeros follow RN (+0 + −0 = +0, −0 + −0 = −0).
NCCL with two ranks rounds the same two-operand sum once (INFERRED). The PyNCCL arm checks this.

**Reference.** `(a.float() + b.float()).to(bf16)` (`patterns.reference_sum`). Rounding twice
through fp32 gives the same result here, because 24 ≥ 2·8 + 2 (Figueroa). CPU checks:
- T2 agrees with exact rational RN on 20 000 generated pairs and 20 000 adversarial pairs
  (ties, exponent gaps 0-30, cancellation): 0 differences.
- Torch's native bf16 add agrees with the reference.

**Criterion: exact bitwise equality, no tolerance.** Inputs contain no NaN, Inf or subnormal
values (T1). Any differing bit is a mismatch.

**Alternative validated transport.** PyNCCL (NCCL 2.30.7, log L53), the transport stock vLLM
uses for the same shapes.
- The `nccl-ref` phase runs the same graphs, patterns and checks with in-graph PyNCCL.
- The 56/64-row engine graphs and the eager 2048-row prefill run PyNCCL through the real dispatch.
- Any mismatch on these arms makes the run **INVALID** (a reference or fixture problem), not a B12x verdict.
- The old pull protocol is in the same image, both as the eager path and as the graph path under
  `B12X_PCIE_TP2_PLAIN_REMOTE_PUSH=0`. It is the substitution candidate, not part of this run (§7).

## 3. Obsolete B12x tests

B12x's own GPU tests for this transport cannot run against the shipped API:
- `tests/comm/test_pcie_oneshot_torture.py:153, 356, 411, 453, 644` and
  `benchmarks/benchmark_pcie_tp2_plain_allreduce.py:156` call
  `channel.prepare_graph_all_reduce(...)`, which no longer exists.
  - Only an error string still names it (`pcie_oneshot.py:1881`).
- The same files call `all_reduce(x, out=…)` / `pool.all_reduce(x, channel_id=…)` without the
  now-required `plan=` argument (`:3092-3104`, `:3905-3914`).

Their checks were also weak:
- Constant fills with `assert_close(atol=rtol=1e-2)` (`:42-44`).
- 64-256 iterations.
- A host synchronisation after every replay.
- The benchmark JSON checks only `output == 3.0`.

The image and the repository are not modified. The fixture replaces these tests by driving the
current path end to end through vLLM's own adapter. Nothing is transcribed: dispatch, plan,
preparation, capture and kernels are all the shipped code.

## 4. Fixture design (`fixture.py`, `patterns.py`)

One container with both GPUs and `--ipc host`. Two processes (`mp.spawn`), rank r on GPU r,
NCCL world group plus gloo CPU group, and the engine's environment. No model is loaded.

**Inputs that change every replay and identify their source.**
- Per rank there are two fixed random tables of 4 194 304 entries:
  - an integer family in [−127, 127], whose sums are exact, so a peer value can be read back;
  - an arithmetic family of normal values with magnitudes 2⁻¹⁰…2¹¹ and −0.0 lanes, whose sums need rounding.
- Element i of the call at (step s, layer l) is `table[r][fam(s,l)·P + (off(s,l)+i) mod P]`.
  - `off` is a 32-bit hash of (s, l), computed in-graph from a device step counter.
  - `fam` alternates with s + l.
- Owner/embedding emulation: at layers l ≡ 2 (mod 3), one rank contributes +0.0.
- Any candidate source can be regenerated exactly on CPU: another step, another layer,
  the other rank, a shifted index, or zero.
- T3: a wrong source reproduces a correct 16-byte pack with probability < 1e-3 (worst
  observed 4.9e-4, from absorption). T1: no two nearby calls share table positions.

**Execution pattern, mirroring the engine.**
- One communicator capture context.
- Per shape: one eager warm-up, which prepares the plan exactly as vLLM's warm-up does, then capture.
- Order is vLLM's (`cudagraph_utils.py:336-389`): PIECEWISE largest first, then FULL.
  - PIECEWISE: vLLM's `BreakableCUDAGraphCapture`, with an eager GEMM break after every second all-reduce.
  - FULL: `torch.cuda.graph` on one shared pool.
- Each graph runs **91 all-reduces back to back** (31 for the boundary set):
  - even layers read a static input slice; odd layers reuse one scratch input address;
  - a GEMM runs before every fourth all-reduce;
  - each output is copied out immediately;
  - one batched bitwise check against the in-graph reference closes the graph.
- Replays are back-to-back on one stream. The host synchronises only every 256 replays, as a watchdog.
- The schedule is a shuffled multiset of graph ids, identical on both ranks.

**Phases (full profile).** 41 graphs, 26 691 forwards, **2 040 800 B12x peer-push calls**,
2.05·10¹¹ checked elements:

| phase | content | replays |
|---|---|---|
| main | FULL 4…48 ×2000 each; FULL 56/64 ×200 (PyNCCL); PIECEWISE 1…48 ×600 each; PIECEWISE 56/64 ×100 | 20 000 |
| skew | FULL 1, 4, 16, 24, 48 with rank-alternating `_sleep` (≈25 µs) before every other all-reduce, so each rank alternately waits for the peer | 1 500 |
| boundary | FULL rows 3, 5, 6, 7, 9, 14, 15, 17, 22, 23, 25, 47 (every geometry edge), 31 all-reduces, ×300; rows 49 ×50 (384 KB switch → PyNCCL) | 3 650 |
| mixed | B12x FULL replay, then an eager 2048-row PyNCCL prefill all-reduce (checked) | 300 |
| nccl-ref | FULL 1, 8, 48 with in-graph PyNCCL | 900 |

The smoke profile runs the same code path on 9 graphs and 175 forwards.

**Recorded per rank:**
- backend set, transport policy, prepared plans (transport/threads/CTAs), observed dispatch per
  graph at capture;
- the **B12x plain epoch counter before and after each phase**, read from the signal buffer.
  The delta must equal the number of peer-push calls the schedule issued, which is direct
  evidence that every intended call, and no other, ran peer-push;
- mismatch counts per graph and layer and per step;
- the first and last failing output of every layer;
- digests of the final outputs and references, compared across ranks;
- the full step schedule.

**Hang handling.** An all-reduce that never completes is a failure of the synchronisation class.
- The watchdog (90 s without progress) writes `hang.json` and exits.
- The runner's wall cap kills the container.

## 5. Failure classes (`patterns.classify`, `analyze.py`)

Each failing 32-bit output word (two bf16, the wire payload unit) is attributed to the first
matching source below. All matching sources are listed in `word_matches`. A 16-byte pack takes
the first class that all four of its words match.

| class | what it looks like | points to |
|---|---|---|
| arithmetic | right operands, output differs by RZ or 1 ulp | rounding semantics (add.rn) |
| wrong-rank | out = local + local | peer pointer/table error |
| missing-peer | out = local (+0 peer) | unpublished or cleared record read as ready |
| stale-same-slot | peer word = that of the push call 2k earlier | {word, tag} tear or premature slot reuse |
| stale-other-slot | peer word = call 2k+1 earlier | slot parity/bias disagreement |
| cross-step | peer word = same layer, earlier replay | stale graph-pool or input buffer reuse |
| addressing | peer word from ±1 pack, ±2 packs or ±1 row of the right call | index/geometry error |
| local-stale | own operand from an earlier call, right peer | local input read before written |
| torn pack | words of one pack from different sources | partial visibility within a pack |
| unexplained | none of the above | corruption |
| hang | no progress | tag/epoch disagreement (parity or count) |

T4 injects each class into a correct output and recovers it at pack level, including torn and
unexplained cases. A stale all-zero owner input matches both missing-peer and stale-same-slot,
and both are listed. T7 runs the whole analysis on a synthetic clean run (CLEAN) and on an
injected stale-same-slot run (FAIL, 3 packs, stale-same-slot).

## 6. Verdict rules (frozen; `analyze.py`)

- **INCOMPLETE**: a rank errored, hung, or hit the wall cap. A hang inside a B12x replay is
  reported as a synchronisation failure.
- **INVALID**, no B12x verdict, if any of these holds:
  - backend set, transport policy, dispatch per graph, or launch geometry per row count
    differs from §1;
  - an epoch delta differs from the scheduled peer-push count;
  - the two ranks' schedules differ;
  - any PyNCCL arm (nccl-ref, 56/64 graphs, eager prefill, PyNCCL warm-ups) has a mismatch;
  - the references differ across ranks.
- **FAIL**: any B12x-dispatched element differs from the reference on either rank (replays or
  warm-ups), or the ranks' final outputs differ. Classification follows.
- **CLEAN**: none of the above.

No re-run is made to "confirm" a CLEAN or overturn a FAIL without a new proposal.

## 7. Runtime and cost

| item | estimate | basis |
|---|---|---|
| container start, imports, NCCL + B12x IPC init | 30-45 s | imports 7 s on CPU; INFERRED rest |
| JIT on first use (B12x CuTe, per rank, in parallel) | 1-3 min cold; seconds warm | 13 launchers per rank: 6 peer-push geometries × 2 slot variants + eager pull. Compile cache in `cache/` is shared by smoke and full. INFERRED |
| warm-up and capture, 41 graphs | 10-30 s | INFERRED |
| full replays (26 691 forwards) | 1.5-3 min | about 2.5-5 ms per 91-all-reduce replay (in-graph generation about 150 B/element, about 25 µs per layer). INFERRED, the smoke run measures it |
| smoke run total | 2-4 min → **0.07-0.13 GPU-h** | wall cap 6 min = 0.20 GPU-h |
| full run total (warm cache) | 3-6 min → **0.10-0.20 GPU-h** | wall cap 10 min = 0.33 GPU-h |
| **both runs** | **0.17-0.33 GPU-h expected** | **ceiling 0.5 GPU-h** |

Occupied GPU-hours = wall time × 2 GPUs, from container start to removal.

Budget gate (`run.sh`): a run starts only if the hours already logged in `costs.jsonl` plus
this run's wall cap fit within 0.5.
- If the smoke run exceeds about 0.17 GPU-h, the full run is blocked.
- In that case either lower its cap with `CAP_S`, if the smoke timing shows it fits, or
  reduce replays by a new amendment.
- The full profile counts are frozen here. They are not tuned after seeing data.

The analysis runs CPU-only after the GPUs are released.

The smoke run is also the fixture's first GPU execution. If a fixture-code error (not a
transport verdict) stops it before any verdict, the fix may be made on CPU and the smoke run
repeated within the same ceiling. The profile counts and verdict rules stay unchanged, and the
new hashes are recorded.

## 8. What a result rules down

**CLEAN** retires, for B0's shipped graph all-reduce transport on this host (two RTX PRO 6000
Max-Q, PCIe, `iommu=pt`, NCCL 2.30.7):
- stale, torn, cross-step, wrong-rank, addressing and parity errors and hangs;
- under every engine graph size and every launch-geometry edge;
- in FULL and breakable-PIECEWISE capture;
- back-to-back replays with 91 calls per graph;
- input-address reuse;
- rank skew;
- interleaving with eager PyNCCL prefill;
- over about 2·10⁶ calls.

It also shows the arithmetic equals both the exact reference and PyNCCL bit for bit, so swapping
the transport cannot change numerics.

It does **not** exclude:
- epoch rollover (unreachable in serving);
- contention from real model kernels and PCIe traffic beyond the GEMM/copy interleave;
- the drafter's exact call sequence (INFERRED to be the same shapes);
- the other communication paths (PyNCCL above 48 rows, which is only cross-checked here;
  DCP A2A, which is off under DCP1; the owner exchange);
- any numerical effect elsewhere in B0.

Per amendment 16: retire the tested failure modes without claiming all communication paths
are excluded. No serving A/B follows automatically.

**FAIL**:
1. Minimize: rerun only the failing size, phase and layer count. The class and step histogram
   say whether it needs skew, slot reuse or a size switch.
2. Reproduce.
3. Verify the smallest correction or a validated substitution in the same fixture:
   - `B12X_PCIE_TP2_PLAIN_REMOTE_PUSH=0`, which keeps B12x but uses the graph pull transport, or
   - `VLLM_ENABLE_PCIE_ALLREDUCE=0`, which is PyNCCL only, at a latency cost.

   `analyze.py` currently expects the B0 policy. A substitution run needs its expected
   policy and geometry made a parameter (≈1 h CPU work, not built).
4. Create B1 = B0 + that one change (a recipe env change; no image change), then propose an
   end-to-end test.

## 9. Files and how to run

`<runs>/b0-allreduce/`. sha256 (also in `SHA256SUMS`;
each run records its own copy):
```
d43f6b002fe3cd71e631e6a533b464d954f55136c19584a3c68deb449d6bde1f  fixture.py
73cf0ad14ee149ca063fd573eed24baa0ea518a9eb314651102cd2fc9f06f663  patterns.py
c941f1b15bbe44127b1336ad5564256f2df0a79f77166038a1ef111dbf688175  analyze.py
37fe1b487cbcc24bc5fb4ecaa9eebe7df07037330de83b449fbee4476b4ed358  cpu_tests.py
ee2316cfd87def76f93808a879d826c03c0cc606eacd3253ce7f94b8ec3862b7  run.sh
```
- Scripts are mounted read-only into the unchanged B0 image. No new image is needed;
  `--entrypoint python3` bypasses the serving entrypoint.
- CPU checks: `runs/cpucheck-20261012T214610Z/`, **29 of 29 passed** (T1-T7).
- Runner self-test: `runs/preflight-20261012T214143Z/`. Pre-flight and budget gate passed;
  no GPU was used.

```bash
cd <runs>/b0-allreduce
./run.sh smoke "B0 all-reduce fixture smoke (PLAN.md amendment 16)"
./run.sh full  "B0 all-reduce fixture full (PLAN.md amendment 16)"
```

Each GPU run does the following:
- **Pre-flight** aborts if any of these holds:
  - a `glm53*` or fixture container is running;
  - GPU0 has a compute process, more than 1 GiB in use, or more than 5 % utilisation;
  - GPU1 has more than 4 GiB in use or more than 15 % utilisation. GPU1 carries the desktop;
    graphics clients are not listed by `--query-compute-apps`, so memory is checked.
- **Budget gate.**
- `docker run --gpus all --ipc host --network none`, run as the invoking user, with NCCL/gloo on `lo`.
- **Wall cap**, enforced with `docker kill`.
- On every exit path: a `costs.jsonl` line (start, end, seconds, gpus 2, gpu_hours, rc, run_dir,
  purpose), container logs and removal, and a postflight `nvidia-smi`.
- Then `analyze.py` writes `RESULT.json` and `RESULT.md` (CPU only).
