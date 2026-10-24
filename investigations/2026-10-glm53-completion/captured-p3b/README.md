# captured-p3b: captured-input MoE-plan test and attention confirmation

**State.** Built and CPU-tested on 2026-10-23. **No GPU has been used.**

- Design, quotas, outcome rules and budget: `PROPOSAL.md`.
- Authorization: `AUTHORIZATION.txt` (ceiling 0.65 occupied GPU-hours in total, retries included).
- Test results: `tests/RESULTS.txt`.

## Layout

| Path | What |
|---|---|
| `build/files/vllm/p3b_capture_hooks.py` | The capture module: graph-safe device rings, host arming, seqlock poller |
| `build/files/vllm/…/exl3.py`, `…/b12x_mla_sparse.py` | Stock 0.9.1 plus 13 and 20 marked additive hook lines |
| `build/make_hooked.py` | Generates the hooked files from `build/stock/` copies; additive gate (`--check`) |
| `build/make_build.sh`, `Dockerfile.p3b` (generated) | `docker build --network none --pull=false` from the local `a9e985c0`; in-build and post-build gates |
| `build/image-id.txt`, `post-build-gates.txt`, `*-vllm-py.sha256` | Built image `local/glm53-v091-p3b-capture:local` = `sha256:5be4b853e99fc4695593dc2b97307ccf754973921e3357ec77f84a71a9d24374`; only 3 of its 2,435 vLLM files differ from stock |
| `runner/run_p3b.sh` | Step A, capture run (2 GPUs). Also `client_p3b.py` (workload), `coverage_p3b.py` (capture verdict, logs only) and `images.txt` (image pin) |
| `replay/run_replay.sh` | Step B wrapper (GPU0). Also `in_container.sh` (part order inside an image), `replay_attn.py`, `replay_moe.py` (reuses P3's `moe_plan_eq.py`, hash-pinned; includes the forced-tile confirmation), `replay_common.py`, `summarize_p3b.py` (CPU) |
| `tests/` | CPU tests. Run all with `tests/run_cpu_tests.sh`; results in `RESULTS.txt` |
| `SHA256SUMS` | Hashes of everything above except `tests/out/` |

Created at run time:

| Path | What |
|---|---|
| `costs.jsonl` | The shared ledger |
| `runs/capture-attempt-N/` | Capture logs, coverage, client records, server log |
| `runs/replay-attempt-N/` | Replay JSON and summary |
| `PRIVATE-capture/` (0700) | Captured tensors, and the replay's per-row positions and lengths |

## Running it (operator)

No `sudo` is needed. Each command refuses without `ATTEMPT_REASON`, and the budget gates read `costs.jsonl`.

```bash
cd <runs>/captured-p3b
# 1. gates only (starts nothing, writes nothing):
ATTEMPT_REASON="dry check" DRY_RUN=1 runner/run_p3b.sh
# 2. capture (about 0.47-0.59 GPU-h; stops the client as soon as the quotas are met):
ATTEMPT_REASON="capture per PROPOSAL.md" runner/run_p3b.sh
#    -> runs/capture-attempt-1/coverage.md says CAPTURE COMPLETE or INCOMPLETE (with the deficit)
# 3. replay on GPU0 (whatever budget is left, at most 600 s):
ATTEMPT_REASON="replay per PROPOSAL.md" CAPTURE_ATTEMPT=1 replay/run_replay.sh
#    -> runs/replay-attempt-1/summary.md
```

**What the capture does:**
- serves `<stacks>/glm53-flash-v0.9.0` with recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1`;
- uses `SPECULATIVE_METHOD=dflash2 DFLASH_TOKENS=3`, as 22ao arm A did;
- uses `IMAGE=local/glm53-v091-p3b-capture:local`;
- sets `DEBUG_ENV="GLM53_P3B=capture GLM53_P3B_DIR=/tmp/p3b GLM53_P3B_POLL_SECS=1"`.

## CPU tests (`tests/RESULTS.txt`)

| Test | Covers |
|---|---|
| T1-T11 (`test_capture_cpu.py`) | Run twice: on the source file and on the copy installed in the built image. Ring and sink slots, narrowed copies, seqlock header, device counter; MoE end to end (quotas, stride, byte-exact `.npz`, more than 3 ring wraps); backpressure; abandoned entries and quota return; attention H1/H2/H3 (pre table before compaction, KV records by the pre table, slot 0, out-of-range clamp counted, no commit without H1 → H2 → H3); **graph-static op sequence** (identical aten ops for every meta value, no host-sync op, Python host conversions trapped); **poller race safety** (22ap allocation race reproduced and fixed, torn copy detected and re-read, engine thread against the real poller thread); row choice; FLUSH; privacy modes and log contents; Caps serialization |
| G1-G4 (`test_register_cpu.py`) | `register()` against the image's real `B12xMLASparseImpl` / Caps classes: served layer names found, a draft subtree ignored, a duplicate candidate becomes a gate issue, ring memory 121.5 MiB |
| A1-A8, M1-M9, E1 (`test_replay_dry.sh`, `check_dry.py`) | Replays `--dry` on captures fabricated **through the real hooks and poller**, with the image's real compaction kernels under `TRITON_INTERPRET=1`. Checks: injected non-reproduction detected (MoE and attention); regimes from exact p and L; DIFFERENCES and EQUAL paths; P1 control; forced-tile PREDICTION HOLDS and FAILS paths; forced-tile CPU probe through the image code (`plan_weights` None → (64, 256, 64, 256), weight plan tile forced, the image's tile rule, P3 fixture inputs bit-identical); wrong-image refusal; the in-container driver with production mounts |
| R1-R8, Q1-Q8 (`test_wrappers.sh`) | Refusals; budget gate and attempt numbering; DRY_RUN writes nothing; MOCK sessions (early stop on quotas, ramp to exactly 8 in flight, seeds, copy-out at 0700, coverage, watchdog, no process left); the replay wrapper with fake docker (argv: `--network none`, `--gpus device=0`, `:ro` mounts, served env, digests), v070 skipped with a cost line when the budget is short |

## Notes

- **Build history.** Two intermediate builds of the same new tag preceded the final one (the module changed between
  them). The image store dropped them when the tag moved; no pre-existing image was retagged or removed.
- **Panel fixture.** The package references its path only. It is hashed and read only by a real capture attempt
  (runner and client), never by the build, the tests or DRY_RUN.
