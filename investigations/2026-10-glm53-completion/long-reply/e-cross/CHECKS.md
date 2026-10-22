# CHECKS: option E package (e-cross), CPU only; re-frozen under amendment 22ah, 2026-10-22

**Zero GPU.** No model server was started, no container ran with `--gpus`, no request went to a model server, and no
`sudo` was used. CPU containers ran with `--network none`, `CUDA_VISIBLE_DEVICES=` and read-only inputs. GPU state was
only read (the runner's pre-flight `nvidia-smi` queries). Test scratch lived under `PRIVATE/` and was deleted. Every file
22ah changed keeps a `.pre22ah` copy, and `CHANGES-22ah.json` lists before / after sha256.

Labels: [M] measured here; [R] reproduced against an earlier package's frozen output; [S] source-read.

| # | check | evidence | result |
|---|---|---|---|
| 1 | D's prefixes rebuilt with the unchanged build code | `tests/prefix-build-repro.txt` | 0.7.0 image: build 58 / 58; manifest **byte-identical** to d091's; private files = d091's = `PRIVATE/src-d091/`; **10 / 10** [R] |
| 2 | D2's prefixes rebuilt with the unchanged build code | `tests/prefix-build-repro-d2.txt` | build 58 / 58; manifest **byte-identical** to d091b's (2ebb5223…); private files = d091b's; **8 / 8** [R] |
| 3 | Rendering on both images | `checks/prefix-verify-{v070,v091}.json`, `checks/prefix-verify-d2-{v070,v091}.json` | D: 114 / 114 on each image; D2: 114 / 114 on each image; **the 5 new prompts 70 / 70 on each**; both 0.9.1 files byte-identical to their sources' [M] |
| 4 | 13-prefix composition | `checks/compose-prefixes.out` | **35 / 35**: source private files = source manifests; every id list hashes to its source entry; D's 8 keep P01-P08; D2 adds exactly q69p5, q120p5, q186p5, q88p5, q71p5 as P09-P13; 13 distinct questions, 8 capped / 5 finished [M] |
| 5 | Byte-identity of everything reused | `checks/identity.txt`, `checks/*.diff` | 24 files identical by sha256 to B+ / D / D2; plan model, params (c53443e3…) and order seed identical to B+'s; B+'s source reading rerun on both images, every block identical; **26 / 26** [M] |
| 6 | Seeds | `checks/seed-inventory.json`, `seed-scan.out`, `raw-seed-scan.json`, `seed-formula-scan.txt` | 8,270,101-8,281,304 (104; the 64 of P01-P08 kept, verified identical to the first freeze): **104 / 104 eligible** over 11,622 files including D2's request logs; none in any raw-body file (D2's included); no intersection with B+, D, D2 or rep070-F planned seeds, any seed-line integer within 20,000, or any formula family [M] |
| 7 | Gate calibration | `checks/gate-calibration.txt`, `.json` | Unchanged `gates()` with E's configuration on **9 reference sessions** (0.7.0: rep070-F × 2, +ps, B+; 0.9.1: record × 2, B+, **D, D2**): **every configuration gate passes on every session**. KV floor 2,666,496 / 2,646,016 met by every reference start (9 server logs, 11 startup markers; margins ≥ +83,103); attention blocks and the longest record prompt (2,795) measured from the references [M] |
| 8 | Analysis dry runs and negatives | `tests/analysis-dryruns.txt`, `tests/out/` | **21 / 21** as expected (detail below) [M] |
| 9 | Power, 13 prefixes | `out/power.json`, `checks/power.out` | P(R-E) 0.75-0.88 at the full gap, 0.21-0.27 at half, ≤ 0.013 per direction with no gap [M on the model] |
| 10 | Costing, 52 continuations per engine | `out/cost.json`, `checks/cost.out` | central 10.2, simulated median 10.65, pessimistic p95 15.4, **ceiling 19.9** GPU-h; P(ceiling) 0 of 20,000 under both models [M, model I] |
| 11 | D7 exposure on the 13-prefix plan | `out/d7_prefill_exposure.json` | 0.7.0 52 / 52, mean 0.14%; 0.9.1 32 / 52, mean 0.07% [E] |
| 12 | Runner refusals (DRY_RUN=1, read-only stubs) | `tests/runner-refusals.txt` | **24 / 24**, at the new ceiling (approval 0 / 20.0 / 31.6 / text refused; the real configuration passes at 19.9) [M] |
| 13 | Session simulation with shims and a watchdog stop | `tests/session-sim.txt` | **62 / 62** over six scenarios: normal end (52 requests per session, 13 / 13 render checks, E seeds only); SIGTERM; watchdog with requests in flight; v091 budget stop; launcher failure; HTTP 500 [M] |
| 14 | Privacy | `checks/privacy-scan.txt` | 262 files outside `PRIVATE/`: 0 prompt-text, 0 reply-text (D, D2 and B+ raw records), 0 token-id-run (all 13 prefixes), 0 answer-field hits; `PRIVATE/` 0700, every file 0600, subfolders 0700 [M] |
| 15 | Real-machine DRY_RUN=1 | `tests/dry-run-real.txt` | Every refusal passed at APPROVED_GPU_H=19.9 (GPUs idle apart from the desktop); nothing created [M] |

## Detail for check 8 (analysis dry runs)

| case | expected | result |
|---|---|---|
| all-finish, none-finish | COMPLETE, R-I, E1 SMALL, PREFIX_HIGH False | as expected |
| engine-effect / reverse | R-E (EFFECT) / R-E− (REVERSE) | as expected |
| state-dominated | R-I with SMALL and PREFIX_HIGH True | as expected |
| null | R-I | as expected |
| kv-2749599, kv-bplus-log | COMPLETE (sizes the 0.7.0 image produced) | as expected |
| **kv-unseen-above-floor** (v070 2,740,000, never produced) | **COMPLETE** (the pool is not identity-gated) | as expected |
| **kv-unseen-v091** (4,698,112) | **COMPLETE** | as expected |
| **neg-kv-below-floor** (v070 2,600,000) | **INVALID**: KV floor | as expected |
| **neg-kv-v091-below-floor** (2,600,000) | **INVALID**: KV floor | as expected |
| neg-wrong-seed, neg-seed-reused | INVALID: seed as planned, seeds distinct | as expected |
| neg-prompt-hash, neg-raw-mismatch, neg-image, neg-no-ready | INVALID on the respective gate | as expected |
| neg-missing-request, neg-watchdog, neg-v091-budget-stop | INCOMPLETE (rc 4), no reading | as expected |

The two first-freeze KV identity negatives (`neg-kv-outside`, `neg-kv-v091`) no longer apply under 22ah. They were
replaced by the floor cases above; their old outputs survive as `.pre22ah` copies.

## How to rerun (all CPU only)

```bash
cd <runs>/long-reply/e-cross
bash tests/prefix_build_repro.sh; bash tests/prefix_build_repro_d2.sh
python3 -B scripts/compose_prefixes_e.py && python3 -B scripts/plan.py
bash checks/identity_check.sh; python3 -B checks/seed_cross_e.py checks/seed-formula-scan.txt
bash checks/gate_calibration_e.sh
bash tests/run_analysis_dryruns.sh                     # about 15 min
bash scripts/run_cpu.sh v070 scripts/power_e.py /work/out/power.json 2000
python3 -B scripts/cost_e.py out/cost.json
bash tests/test_runner_refusals.sh; bash tests/session_sim.sh   # about 5 min
DRY_RUN=1 APPROVED_GPU_H=19.9 bash runner/run_e.sh
```

Rewriting a file that SHA256SUMS lists makes the runner's package check fail until `make_sums.sh` is rerun. The frozen
hash is the one in `FROZEN-VERIFY.txt`.

## Not verified on CPU

Prefix-cache hits on 0.7.0; decode rates on 0.9.1-built prefixes; the live render endpoint; 0.7.0's scheduler chunking
in the D7 estimate. See PROPOSAL.md section 10.
