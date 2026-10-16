# CPU checks run for the re-freeze (2026-10-15)

Zero GPU: no model server or GPU container was started and no request was sent to a model server. Docker was used
read-only (`docker ps`, `docker image inspect`) plus CPU-only containers with `--network none -e CUDA_VISIBLE_DEVICES=`
to read source in the 0.7.0 image. No sudo. Nothing outside this directory was written (the draft's files verify against
`../SHA256SUMS-draft`). Results: [M] measured, [R] reconstructed, [S] source read.

| # | Check | Result |
|---:|---|---|
| 1 | Draft files unchanged: `sha256sum -c ../SHA256SUMS-draft` | OK, 4/4 [M] |
| 2 | Image present; `docker image inspect` Id of the digest reference | sha256:48e254d9... = the digest; created 2026-09-06; 8,698,063,358 bytes [M] |
| 3 | Running containers | open-webui, portainer; no GLM container [M] |
| 4 | Weights snapshot 701cd745: listing vs the 2026-10-10 config-diff listing | identical, 35 files (`checks/snapshot-701cd745.txt`) [M] |
| 5 | Weights and draft blob contents vs blob names (`checks/blobcheck.sh`) | 40/40 OK (35 weights files, 5 draft files) (`checks/blobcheck.out`) [M] |
| 6 | Chat template sha256 | 34d5ee66... [M] |
| 7 | Recipe files | env.sh ed50bc7d (2026-09-05), recipe.json 0f9ee0cc (2026-10-10 08:55), README 4d28884d; /data git status clean for the recipe and stack paths [M] |
| 8 | Stack files and git state | start.sh c852511c, local-env.sh f1f3ced5, start.sh 869b0c0a, model-profiles.sh 13c07d5b, stop.sh 431739f5; upstream HEAD 6041c588; `git diff`: start.sh +1 line (TILELANG_CACHE_DIR), file mtime 2026-09-05; local-env.sh mtime 2026-10-10 15:57 (PORT loopback), changes.json entries 2026-09-05, 2026-10-04, 2026-10-10 [M/S] |
| 9 | Publish address | local-env.sh PORT default 127.0.0.1:8001; start.sh `--publish "${PORT}:8001"`, `--host 0.0.0.0` inside; the runner resolves the launcher to 127.0.0.1:8001 (DRY_RUN) [M/S] |
| 10 | Eval chain, proxy, scorer | chain c4dbb500 (2026-10-08 20:15), proxy 946d880e (2026-10-06, binds 127.0.0.1), scorer e13116ca (2026-10-08) [M] |
| 11 | lm-eval install | HEAD b954108c; `git status --porcelain` = only the untracked task yaml; yaml 149658b4, utils.py b6038287, openai_completions.py 0afb2771, api_models.py 8da25fde; no file newer than 2026-10-11 outside `.git` (the `.git` directory mtime is this session's `git status`) [M]; model seed default 1234 placed in every payload [S] |
| 12 | Prompt identity across passes | 198 of 198 message hashes shared by seeds 1235 and 1236 in the 0.9.1 record's log [M] |
| 13 | Record process and model seeds | results jsons: pass 1 model seed absent, process seeds 1235/1236/1237 for passes 1-3 [M] |
| 14 | Host | kernel 7.0.0-30-generic, cmdline `iommu=pt`; driver 595.84 (open); docker 29.7.2; containerd 2.3.4; nvidia-container-toolkit 1.20.0; boot 2026-10-03 21:48, docker/containerd active since; dpkg log 2026-10-11..15: libx11, linux-libc-dev / linux-tools-common 6.8.0-147, systemd/udev 255.4-1ubuntu8.18, curl, openssh, bluez, bind9, python3-urllib3, python3-jwt; none of kernel/driver/runtime [M] |
| 15 | Compile cache `<cache>/vllm` | 2,884 files, 896 not readable to the operator account; 2,121 before 2026-09-29 06:51, 702 between then and 2026-10-10 08:46, 61 on 2026-10-10 after the record, 0 after 2026-10-11 22:10; metadata manifest sha256 0c64c2f1... [M] |
| 16 | GPU state at check time | GPU0 20 MiB, 0%; GPU1 2,269 MiB, 1% (gnome-shell, Xwayland, xdg-desktop-portal, firefox, snap-store, one C+G `--disable-logging` client); persistence off [M] |
| 17 | Disk | /data 1.8 TB free (50% used); / 5% used [M] |
| 18 | Timing reconstruction re-run (`reconstruct_timing.py`, copied unchanged) | md and json identical to the draft's; 22.97 GPU-h [R] |
| 19 | Seed inventory (`checks/seed_scan.py` -> `seed-inventory.json`, `seed-scan.out`) | GPQA request seeds: 1234, 1235, 1236 (full passes), 4242 (gate runs), 5001-5012 (screens); 10,529 text files scanned, 1,189 distinct seed-like values; 70701, 70702, 70703 eligible (in none of the sources) [M] |
| 20 | Same-seed identity (`checks/same_seed_identity.py` -> `same-seed-identity.json`) | 24 pairs: identical answered replies 0 in every pair; log reply-length correlation same seed 0.655-0.909 (13 pairs), different seed 0.373-0.477 (11 pairs) [M] |
| 21 | vLLM seeding in the 0.7.0 image (CPU-only container, `--network none`) | one `torch.Generator(device).manual_seed(seed)` per seeded request (`gpu_model_runner.py` 1404-1405); seeded requests use the native top-k/top-p sampler (`topk_topp_sampler.py` 164-175) [S] |
| 22 | Record server logs: warm-up lines, starts, first chat request | 0.7.0 +ps: 9 warm-up lines, 1 start, ready line 430, first chat access line 442; 0.9.1 logs: 9, 1, 488 / 498 and 488 / 496 [M] |
| 23 | Record server-log facts (frozen parser) | 0.7.0 +ps: args 74f26837..., vLLM 0.1.dev20051+g487ecf187, snapshots 701cd745 + bf582e4e, KV 2,758,919, FULL_AND_PIECEWISE, 9 warm-ups [M] |
| 24 | Stated adjudicator source (export_evals.py lines 233-259) | unchanged, 731f27d4 [M] |
| 25 | Frozen module copies in `frozen/` | analysis_col511.py b5b13ce2..., power_col511.py 92466804..., analysis-col511.md 523ec228... (byte-identical to the draft's import source) [M] |
| 26 | Analysis dry runs (`run_dryruns_rep070.sh` -> `dryruns/`) | D1 valid O1 (66/66 gating); D2 valid O2 (70/70); D3 F relabel valid O2 (72/72); N1 wrong seed, N2 missing question, N3 variant mismatch, N4 missing ready line, N5 configuration mismatch, N6 one server log: all invalid, exit 3; self-test per-pass loader = single-call loader (594 rows, 15 checks); C1/C2 tests and effects of D1 equal the draft's dry run [M] |
| 27 | Secondary reproduction | old record as rep: 10/14 vs 10/25, +0.31 [0.03, 0.56] (amendment 20c: +0.31 [0.03, 0.55]) [M] |
| 28 | Primary power, no coupling (`power/power-rep070-c0.*`) | identical to the draft's `power-rep070.txt` and json results [M] |
| 29 | Primary power, coupling 0.25 and 0.5 (`power/power-rep070-c025.*`, `-c05.*`) | section 6d table [M] |
| 30 | Secondary power (`power/power-long-finish*.{txt,json}`) | S1 0.59, S2 0.41 (heterogeneity sensitivity 0.57, 0.42); MC SE <= 0.008 [M] |
| 31 | Budget (`power/budget-rep070.*`) | reconstruction 22.97 / 26.45 / 35.0; outcome model as section 5; option B 36.5 / 50.0 [R/I] |
| 32 | Runner refusals, DRY_RUN (`tests/test_runner_refusals.sh` -> `tests/runner-refusals.txt`) | 17/17 (incl. a dummy lm_eval process refused, and a process merely naming lm_eval not refused); `docker ps` identical before and after [M] |
| 33 | Chain copy seed lines and diff (`tests/test_chain_seed.sh` -> `tests/chain-seed.txt`) | 6/6 [M] |
| 34 | Runner session simulation (`tests/session_sim.sh` -> `tests/session-sim.txt`) | 42/42; found and fixed a `$?`-after-`$(date)` logging defect; lm_eval matching narrowed to the executable's command line (`bin/lm_eval --model`; the earlier runners' `pkill -f lm_eval` matches any process naming it) [M] |
| 35 | Package hashes | `SHA256SUMS` written last by `make_sums.sh`; `sha256sum -c SHA256SUMS` and a real-machine DRY_RUN of both variants against it are the final step [M] |

## Change set 22n (2026-10-15, same rules: zero GPU, read-only Docker, writes only in this directory)

Rows 26, 31, 32 and 34 above are superseded by rows 41, 37, 44 and 45 below (their earlier outputs are kept as
`*.pre22n`).

| # | Check | Result |
|---:|---|---|
| 36 | Baseline before 22n | package verified against its SHA256SUMS 29a2fb11... and snapshotted; every changed file kept as `<file>.pre22n`; before/after sha256 in `CHANGES-22n.json` [M] |
| 37 | Budget B-lean (`budget_rep070.py --thresholds 35.0 36.5 --approved 36.5`) | P(total >= 36.5): S1b 0 / 0, S2 0 (empirical) / 0.044 (token-based); P(session 2 not started) 0 in all four rows; reconstruction unchanged 22.97 / 26.45 / 35.0 [I/R] |
| 38 | Task hashes of both records | 87e807b3b27c8661... in all six passes; process seeds 1235/1236/1237 [M] |
| 39 | Token ids in the 0.7.0 image's chat responses (CPU-only container, `--network none`) | `token_ids` / `prompt_token_ids` only when the request sets `return_token_ids` (default None): protocol.py 415, serving.py 1032, 1101 [S]; not retained |
| 40 | Raw-file size estimate | record replies 10.2-12.6 million characters per pass, 2.87-3.02 characters per token, largest reply 1.19 million [M]; at most about 50 MB uncompressed for a run [I] |
| 41 | Analysis dry runs after 22n (`dryruns/`) | D1 valid O1 71/71; D2 valid O2 75/75; D3 valid O2 77/77; N1-N8 all invalid (N7 wrong task hash, N8 target mismatch: new); self-test per-pass loader 594 rows / 15 checks [M] |
| 42 | Proxy copy (`tests/test_proxy_sidecar.py` -> `tests/proxy-sidecar.txt`) | 17/17: completed row equal to the original's; byte-identical main log under a fixed clock; sidecar start / interrupted (upstream SIGKILL: RemoteDisconnected) / client_gone / proxy_terminated on SIGTERM; raw bodies present, mode 0600; logs consistent in every case [M] |
| 43 | Incomplete-run handling (`tests/test_incomplete.py` -> `tests/incomplete.txt`) | 14/14: (a) watchdog during pass 3; (b) session 2 not started; (c) cap-reached and interrupted in one pass; (d) complete run = frozen analysis, same as D3 [M] |
| 44 | Runner refusals (`tests/runner-refusals.txt`) | 18/18 incl. 36.5 accepted, 36.51 refused, unset refused; `docker ps` identical before and after [M] |
| 45 | Session simulation (`tests/session-sim.txt`) | 61/61 incl. (f) watchdog with a request in flight (sidecar start + interrupted, raw interrupted record 0600, no main-log row, INCOMPLETE report: 1 interrupted, 197 unstarted); orphaned-sleep check (the logger/watchdog leak found and fixed) [M] |
| 46 | Chain seed lines (`tests/chain-seed.txt`) | 6/6 [M] |
| 47 | Compile-cache inventory (`runner/cache_inventory.py`) | 2,884 files, 896 unreadable; two consecutive snapshots identical (diff 0/0/0); 0.4 s [M] |
| 48 | Final | `sha256sum -c SHA256SUMS` and a real-machine DRY_RUN of variant F at APPROVED_GPU_H=36.5 against the final package [M] |
