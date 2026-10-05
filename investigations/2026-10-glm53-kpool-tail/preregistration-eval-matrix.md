> Published as written before the data, except that local file paths are replaced with repo paths or '(local file)'.

# GLM-5.3-Flash evaluation matrix - plan (written 2026-10-04, before any matrix data)

**Goal.** A reliable, reproducible GLM recipe whose GPQA Diamond score is comparable with published numbers.
Precision first; throughput is reported but is not the objective.

## Reference numbers - and why exact comparability is not achievable
| Source | Weights | GPQA Diamond | Published protocol |
|---|---|---|---|
| NVIDIA model card | BF16 | 92.17 | temp 1.0, top_p 0.95, max_new_tokens 327,680. Harness, template, extraction, repeats: not stated |
| NVIDIA model card | NVFP4 | 92.11 | same |
| Red Hat model card | NVFP4 | 90.57 pass@1 | lm-eval / lighteval (Neural Magic forks), vLLM, MTP 5 tokens, 3 seeds averaged. Temp, template, extraction: not stated |
| This machine (2026-10-02) | EXL3 K3.25, v0.8.0 | 86.4 | below |

The same NVFP4 weights score 92.1 (NVIDIA) and 90.6 (Red Hat): harness/protocol alone moves the number ~1.5 points.
Neither checkpoint fits this machine (204 / 198 GB vs ~180 GB usable VRAM), so no local anchor run is possible.
Therefore: our protocol is fully specified and frozen, matches every setting that WAS published, and every score is
reported with a confidence interval. "Comparable" means "same published knobs, stated harness", never "identical".

## Claims under test (what the primary sources actually say, checked 2026-10-04)
Sources: wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 card, brandonmusic/GLM-5.3-Flash-tr3-4bpw card,
tpurtell/glm-5.3-flash-ext3-2x-rtx README.md and benchmarks/RESULTS.md.

| Claim | What the sources publish | How this matrix tests it |
|---|---|---|
| Fits two 96 GB cards | K3.25: 146 GB, 2.76M-token KV on v0.7.0; K4: 176 GB, 1.37M KV | Already verified here (K3.25 4.7M KV on v0.8.0; K4 ran) |
| Quality comparable to NVIDIA NVFP4 | **No GPQA number and no comparison against NVIDIA's NVFP4 *weights* in any of these sources.** "NVFP4" appears there only as a KV-cache format. Published quality evidence: frozen-text NLL, K4 KLD vs a BF16 teacher (0.025 FP8 KV), 1M needles 6/6, tool-call suite 86/100 | GPQA under the frozen protocol vs 92.1 / 90.6 with CI. KLD ranks EXL3 variants against BF16, but no published NVFP4-vs-BF16 KLD exists to compare with |
| Mixed quant (K3.25) beats uniform K3 at similar size | K3.25 card: tool-call 86/100 vs 88/100 for the matched uniform-K3 control (no gain); local NLL 0.7147 vs 0.7197 (small gain, beyond the +-0.002 floor) | Optional row: patched uniform K3, same protocol |
| Performance benefit from offloading | Not found in these sources | Needs the claim's source before it can be tested |
| Precision headroom from more bits | K4 KLD 0.027 vs K3.25 0.049 (local) | Optional row R1-K4: patched, K4 weights. KV holds ~4 concurrent 327K-token runs; vLLM preempts beyond that (slower, not wrong) |

Also noted: tpurtell's own frozen-text NLL probe scores v0.8.0 at 1.1786 vs v0.7.1 at 1.1743 (README). That probe is
prefill-only; it does not exercise either decode bug fixed here.

## Frozen GPQA protocol (all configurations)
- lm-eval 0.4.14.dev0 at lm-evaluation-harness, task `gpqa_diamond_thinking` (198 questions, CoT zero-shot, choices
  shuffled per lm-eval's gpqa preprocessing), thinking on, chat template applied server-side.
- temperature 1.0, top_p 0.95, max_gen_toks 327,680 (= NVIDIA's cap), `until: ["</s>"]`, 8 concurrent.
- 3 passes, seeds 1235/1236/1237 (eval-chain convention 1234+pass). Report the mean of the 3 passes (as Red Hat does).
- Score: flexible-extract (headline) and strict-match, plus a local rescoring script (not used for published scores) rescoring.
- Uncertainty: 95% CI by bootstrap over questions (each question's 3 samples resampled together).
  With 198 questions the CI half-width is ~2.5-3 points: differences smaller than that are not resolvable.
- Always reported alongside: no-answer count (empty content), loop/exhaust split, and accuracy over answered samples
  (separates "knows the answer" from "failed to finish").

## Configurations (rows)
| ID | Image | Speculation | Purpose |
|---|---|---|---|
| R0 | tpurtell v0.8.0 (unpatched) | DFlash2, 3 | baseline; existing GPQA receipts (86.4, 20/594 empty) |
| R0n | tpurtell v0.8.0 (unpatched) | none | does the seed bug change outputs without speculation |
| R1 | v0.8.0 + #57477 + #58454 | DFlash2, 3 | candidate recipe (`+kpoolring`) |
| R1n | v0.8.0 + #57477 + #58454 | none | correctness reference for R1/R2 |
| R2 | v0.8.0 + #57477 + #58454 | DFlash2, 5 | v0.7.0's draft depth, now safe with the ring fix |

## Measurements, cheapest first
**Phase 1 - fidelity and speed (~2.5 h, every row):**
1. Kernel tests: official vLLM test_kpool_decode_update_batched.py + our rejected-draft test (done: R0 4 fail, R1 33 pass).
2. Greedy parity: 16 GPQA prompts, temperature 0, 2,048 tokens, one request at a time. Speculative decoding
   is designed to be exact, so R1 vs R1n should agree token-for-token up to numerical noise; the noise floor is R1n
   run twice. Report first-divergence position per prompt. R0 vs R0n and R0n vs R1n are diagnostic.
3. Acceptance: the same 16 prompts, temperature 1.0 / top_p 0.95, 4,096 tokens, at 1 and 8 concurrent. From the
   server's counters: accepted / drafted, mean acceptance length, per-position acceptance.
4. Throughput from the same runs: decode tok/s per request and aggregate, time to first token.
5. NLL on the frozen corpus, twice (local): prefill numerics guard - R1 must equal R0 within +-0.002.

**Phase 2 - GPQA (~10 h per row, one row per night):** R1 first. Then R2 if R1's acceptance/parity are sound.
R1n (no speculation) is not run on GPQA: ~3x slower decode (~30 h); greedy parity establishes exactness instead.

**Phase 3 - regressions:** tool-eval and IFBench on the chosen recipe.

## Decision rules (fixed now)
- A configuration is ELIGIBLE as the recipe of record only if: kernel tests pass, greedy parity vs its no-spec
  reference is within the R1n-vs-R1n noise floor, NLL within +-0.002 of R0, and zero server errors across phases.
- Among eligible configurations, prefer higher GPQA mean; a difference smaller than the CI half-width is a tie,
  and ties go to the configuration with fewer no-answers, then to higher throughput.
- R1 "closes the gap" by the amount its GPQA mean exceeds R0's 86.4, reported with the CI. No claim of matching
  NVIDIA is made unless the reference value lies inside R1's 95% CI.

## Not allowed after seeing data
Changing the protocol, seeds, cap, extraction, prompts, or decision rules; dropping passes or requests.

## Contribution and attribution (recorded as we go; public work only after the matrix is complete)
Credit, by what each party contributed:
- **vLLM upstream fixes being ported:** #57477 (seed stride) by @JaredforReal; #58454 (spec ring) by @mmastrac, which
  credits #55219 by @ivanium. Ports name the upstream PR in each script docstring and in the PR description.
- **Runtime and K3.25 weights:** tpurtell, who is also wrldsuksgo2mars on Hugging Face / Twitter (glm-5.3-flash-ext3-2x-rtx,
  Apache-2.0; no CONTRIBUTING/DCO file; quant wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1). **K4 weights:** Brandon M. Music / brandonmusic, @BrandonMusicKy on Twitter (K4, ShapleyMCG License 1.0 - derivatives must carry its identifier).
  **Drafter:** incoai/GLM-5.3-Flash-DFlash2 (CC-BY-NC-ND-4.0). **Model:** zai-org GLM-5.3-Flash.
Practices to follow: issue before PR (offer the fix, let the maintainer choose); one PR per upstream fix, in upstream
order; link every claim to a receipt anyone can rerun; state what was NOT tested (DCP2, AMD path); state AI assistance
in the PR description, as upstream #58846 did; commits as UrbanAstroLA with the GitHub noreply address.

---

## Amendment 1 (written 2026-10-04 22:18 PDT, before any data from these screens): Phase 2 replaced

**Why.** The kpool screen's arm C (v0.8.0 + both fixes) failed 20/40 vs 22/40 unpatched: the fixes are sound but are not
the main cause of non-completion on the hard prompts, so a 3-pass GPQA on R1 would almost certainly be inconclusive
(empties ~15-20 of 594 vs 20; accuracy CI +-2.5-3). The largest open difference is between engines: v0.7.0 failed 12/40
on the 2026-09-30 screen vs 22-24 for v0.8.0, at the same GPQA accuracy. v0.7.0 runs 5 draft tokens, where the ring
bug wraps furthest. Both ports apply to v0.7.0 with exact matches.

**Phase 2 is now two screens** (same requests, client, early stop, classes as the kpool screen), after Phase 1:
- S7p: `glm53-flash-exl3-k3.25@tpurtell+kpoolring` - v0.7.0 + #57477 + #58454, 5 drafts. First, plus NLL x2 before it.
- S7:  `glm53-flash-exl3-k3.25@tpurtell` - v0.7.0 unpatched, 5 drafts. Same-session control.
Gate before S7p: our rejected-draft test all-correct and official vLLM kpool tests 0 failures on the patched v0.7.0
image; unpatched v0.7.0 must still fail.

**Rules (fixed now).** Fixes help on v0.7.0 iff S7 >= 2 x S7p and S7 - S7p >= 4.
GPQA (3 passes, frozen protocol) then runs on one configuration: the fewest non-ok of 40 among S7p, S7 and R1
(kpool arm C, 20/40). Differences under 4 are ties; ties go to a patched configuration (the fixes are correct
regardless), then to v0.7.0 (lower historical empty rate).
Deliverables shaped for downstream reuse (recipes are widely reposted and ported, e.g. to DGX Spark, mostly without deep
benchmarking): (1) a two-minute self-check - extract `glm5next/nvidia/ops/kpool_compress.py` from any build without
running it, run upstream's test_kpool_decode_update_batched.py against it; (2) a calibrated "who is affected" note:
rarely short/fresh-server use, most plausibly long-lived servers with prefix caching, reused system prompts, 2+ draft
tokens and long reasoning (to be backed by the amplified test). Route: fix lands in tpurtell's repo first so ports that
rebuild inherit it. Lead to check before drafting: tpurtell's GB10/Spark build (local copy stacks/glm53-flash-exl3-sparks).
