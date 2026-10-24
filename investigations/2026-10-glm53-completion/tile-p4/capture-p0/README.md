# capture-p0: point-0 exact-byte capture and warm-prefill localisation (package)

This package resolves the exact bytes of the valid cached state that differed between independently warmed
replicates in Stage 0, and the layer-3/4 boundary at which they first part.

- **Protocol:** `PROTOCOL.md`
- **Self-critique:** `SELF-CRITIQUE.md`
- **Status:** attempt 1 ran on 2026-10-24 and read INCOMPLETE (two package defects; `PROTOCOL.md` amendments A1-A4).
  Attempt 2 (image a2) ran the same day: all 6 replicates passed every gate, and the pre-registered reading is
  **operation located** (`RESULT.md`; analysis output `runs/session-attempt-2/cap0-result.md`).
  Attempt 2 is prepared and CPU-tested, not run. It needs a new pre-check of the new image (commands in `PROTOCOL.md`
  section 9).
- **Private outputs** go to `../PRIVATE-capture-p0/` (mode 0700). Nothing in this directory holds model data.

## Files

| Path | What |
|---|---|
| `PROTOCOL.md` | Purpose, exact slices, construction, gates, the pre-registered decision table mapped onto the assessor's 22ax table, cost, limits, commands (public-safe) |
| `SELF-CRITIQUE.md` | For each reading: spurious causes, which gate or test catches them, what was fixed, what stays open |
| `build/files/vllm/cap0_hooks_layout.py` | Byte addressing of every slice (pure Python). One source for the hook and the analysis |
| `build/files/vllm/cap0_hooks.py` | Hook module: the point-0 capture (both ranks) and the localisation hooks. Inert unless `GLM53_CAP0=on` |
| `build/files/vllm/p4_hooks.py` | The Stage-0 module plus 21 lines marked `# local-cap0` (A2 adds the rank-0 step/sched flush); without them it is byte-identical to the Stage-0 module |
| `build/make_build_cap0.sh` | CPU build of `local/glm53-v091-p4-cap0:local` FROM the Stage-0 image, with gates. `GATES_ONLY=1` re-runs the post-build gates |
| `build/Dockerfile.cap0`, `build-cap0.log`, `image-id.txt`, `post-build-gates.txt`, `*-vllm-py.sha256` | Build record of `local/glm53-v091-p4-cap0-a2:local` = `sha256:8cb49fcabcfe85e8624d03822d0ca7a02b10469f6a3ea6d8713aed2459abd3bd` (attempt 2) |
| `build/attempt1/` | Build record of the attempt-1 image `local/glm53-v091-p4-cap0:local` = `sha256:5831fc27…` |
| `precheck/precheck_cap0.py` | Single-GPU pre-check: capture exactness, graph capture with hooks, a hashing step at served sizes, negative control. `P4_PRECHECK_DEV=cpu` runs a CPU dry run |
| `runner/run_precheck_cap0.sh` | Pre-check wrapper: 1 GPU, cap 180 s, budget gate, cost line. `DRY_RUN=1`; `FAKE=1` for tests |
| `runner/run_cap0.sh` | Session wrapper: 2 GPUs, cap 1,100 s, protocol acknowledgement, frozen-hash and image gates, start-up gates before traffic, client, flush, stop, cost line, private copy-out, outcome, analysis. `DRY_RUN=1`; `MOCK=1`; `CAP0_LOC=off` |
| `runner/budget_cap0.py` | Budget, order and attempt gate. Ledger `costs-cap0.jsonl` only (created at the first GPU step) |
| `runner/client_cap0.py` | The replicates: warm, capture request, sentinel; one request at a time. Writes `client.jsonl` (sha256 only) |
| `runner/outcome_cap0.py` | COMPLETE / PARTIAL / INCOMPLETE of an attempt |
| `runner/cost_cap0.py`, `cost_cap0.json` | GPU-time estimate from the measured Stage-0 figures and the pre-check's hashing rate |
| `runner/images.txt` | Image pin |
| `analysis/analyze_cap0.py` | Pre-registered analysis: gates, exact comparisons, readings. Writes `cap0-result.{json,md}` (public) and `cap0-detail.json` (private) |
| `analysis/run_cap0_analysis.sh` | Runs the analysis in a CPU container |
| `tests/run_cpu_tests.sh` | CPU test entry point. Suites: `layout hooked image enginepath precheck analysis runner gates attempt1 s0regress` |
| `tests/test_engine_path_cpu.py` | The real forcewin → p4_hooks → cap0_hooks path with keyed, non-forced requests (toy engine). Passes on the fixed module; the suite also runs it against the attempt-1 image, where it must show the defect |
| `tests/check_attempt1.py`, `tests/diag_attempt1_loc.py` | Read-only re-analysis of the real attempt-1 records: the causes, and the localisation instrument's gate on real data |
| `tests/test_layout.py` | Addressing, including the audit's task-4 byte ranges and the Stage-0 recorded warm geometry |
| `tests/test_hook_cpu.py`, `tests/test_switch_off.py`, `tests/inner_cap0.sh` | Hook capture and localisation on served-layout SYNTHETIC tensors, negative controls, call sites, switch off |
| `tests/make_fake_cap0.py`, `tests/check_analysis_cap0.py` | SYNTHETIC sessions: every outcome class and planted defect |
| `tests/mock_server_cap0.py`, `tests/make_mock_root_cap0.py`, `tests/test_runner_cap0.sh` | Runner tests against a mock server, with a SYNTHETIC fixture (never under a PRIVATE path) |
| `tests/test_gates_cap0.py` | Budget-gate tests |
| `tests/RESULTS.txt` | CPU test results of the frozen package |
| `make_sums.sh`, `SHA256SUMS` | Frozen-hash list. The GPU wrappers refuse when it does not verify |

## Order of use

1. CPU tests: `tests/run_cpu_tests.sh layout image enginepath precheck analysis runner gates attempt1 s0regress`
2. Pre-check (1 GPU).
3. Session (2 GPUs).
4. Read `runs/session-attempt-N/cap0-result.md`.

Commands: `PROTOCOL.md` section 9.
