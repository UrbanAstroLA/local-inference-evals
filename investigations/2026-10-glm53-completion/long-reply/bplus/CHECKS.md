# B+ CPU checks before freezing (2026-10-16)

All checks ran with **zero GPU**. Containers ran CPU-only (`--network none`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`) on the
existing local images. No model server was started and no model request was sent. Labels: [M] measured, [S] source-read.

| # | check | how | result |
|---|---|---|---|
| 1 | prefix selection is rule-driven and reproducible | `scripts/select_prefixes.py out/selection.json` (input hashes checked) | 12 prefixes, 6 capped / 6 finished, 12 distinct questions [M] |
| 2 | prefix construction | `run_cpu.sh v070 scripts/build_prefixes.py build` | 86 / 86 checks [M]. Prompt length = server `prompt_tokens` (raw response and request log); prompt ids = d7's independent tokenizer; transformers = tokenizers; reasoning length = discovery N; no stop id or `</think>` in the reasoning part |
| 3 | construction reproducible, id hashes identical on re-run | two builds; verify mode in each image, run twice | `PRIVATE/prefixes.json` sha256 `4abffaf1…` both builds; every rebuilt prefix equal id for id [M] |
| 4 | template / tokenizer equality across images | `run_cpu.sh {v070,v091,b1} scripts/build_prefixes.py verify` with each engine's served template and snapshot | 170 / 170 per image. Templates differ as files (34d5ee66 vs 0c4099f3) but render byte-identical text and ids for all 12 prompts [M] (`checks/prefix-verify-*.json`) |
| 5 | completions endpoint, seed, sampling, stop tokens, return_token_ids, reasoning parser | `scripts/source_check.sh` → `checks/source-check.txt` | serving and sampling sources byte-identical in all three images; token-id prompts accepted; seed / temperature / top_p / max_tokens passed; stop ids 154820 / 154827 / 154829 from tokenizer + generation_config in the shared input processor; 154842 not special, not a stop; return_token_ids output-only; 0 reasoning references in the completion endpoint [S] |
| 6 | seeds unused | `rep070/refreeze/checks/seed_scan.py` unchanged with the 144 candidates; formula-base grep | 144 / 144 eligible; no formula range intersects [M] (`checks/seed-inventory.json`, `checks/seed-formula-scan.txt`) |
| 7 | plan | `scripts/plan.py` | 48 requests, 144 distinct seeds, first 8 in flight = 8 prefixes (4 per stratum) [M] |
| 8 | D7 prefill exposure | `scripts/d7_prefill_exposure.py` (d7 `sim_d7.py`, hash-checked) | 0.7.0: every continuation, mean 0.14% of its prefix's index-K pools; 0.9.1: 32 / 48, mean 0.08%, max 0.20%; B1: repaired [E] |
| 9 | power of the frozen screens | `run_cpu.sh v070 scripts/power_bplus.py out/power.json 2000` | P(R-E) 0.70-0.81 at the full gap, about 0.2 at half, ≤ 0.016 under no gap; P(R-S) 0.40 under strong state, ≤ 0.002 under the full gap [M on the model] |
| 10 | cost | `scripts/rates_from_logs.py`; `scripts/cost_bplus.py` | central 15.8, pessimistic 28.4, ceiling 31.6 GPU-h; P(exceed) < 0.1% [M on the model] |
| 11 | analysis dry runs on synthetic outcomes | `tests/run_analysis_dryruns.sh` → `tests/analysis-dryruns.txt`, `tests/out/` | 13 / 13: all-finish R-I, none-finish R-I, engine effect R-E, state-dominated R-S, null R-I, C1-ONLY R-E [M] |
| 12 | negative cases marked invalid | same suite | wrong seed, server-echoed prompt hash mismatch, raw/log mismatch, wrong image, missing ready line → INVALID (rc 3); missing request, watchdog → INCOMPLETE (rc 4) [M] |
| 13 | runner refusals | `tests/test_runner_refusals.sh` → `tests/runner-refusals.txt` (DRY_RUN=1, read-only docker stubs) | 23 / 23, including the real configuration passing every check [M] |
| 14 | session simulation with shims | `tests/session_sim.sh` → `tests/session-sim.txt` (docker / nvidia-smi shims, the real stack launchers, fake server on 127.0.0.1:8001) | 61 / 61. Scenarios: normal completion (COMPLETE); SIGTERM in session 2; **watchdog interruption with requests in flight**; budget stop before B1 (C1-ONLY); launcher failure; one HTTP 500 → INCOMPLETE [M] |
| 15 | privacy | `checks/privacy-scan.txt` | 0 prompt-text windows and 0 reasoning-id runs outside `PRIVATE/` and `run/`; `PRIVATE/` 0700, files 0600 [M] |
| 16 | real dry run | `DRY_RUN=1 APPROVED_GPU_H=31.6 runner/run_bplus.sh` | all refusal checks pass on this machine; nothing created [M] |

Notes:
- **Real launchers in the session simulation:** the simulation ran the real, pinned stack launchers. Their
  `docker run --detach --gpus …` was answered by the shim with a fake CPU server. The shim refuses any other docker call
  and passes only `--rm --network none` CPU containers (the analysis) to the real docker. No refused call occurred.
- **Synthetic run folders** (random ids, placeholder text) were written to the session scratchpad and deleted afterwards.
- **Not verifiable on CPU:**
  - the actual prefix-cache hits;
  - the scheduler's chunking on 0.7.0 (source-read, as in d7);
  - decode rates at 8 × 100-165k contexts on 0.7.0 and stock 0.9.1;
  - the render endpoint's actual output on each server. [S]: `/v1/chat/completions/render` takes a
    `ChatCompletionRequest`, so `chat_template_kwargs` is accepted, and the source is identical in all three images. The
    client records any error and continues; the check is not a gate.
