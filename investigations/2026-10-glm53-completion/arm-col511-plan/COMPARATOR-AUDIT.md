# Comparator matching audit (amendments 11-11b; CPU, read-only, before arm data)

Comparator: the clean 0.9.1 K3.25 three-pass GPQA record. Pass 1 ran in its own server session (2026-10-08); passes 2-3
in a second session (2026-10-09). Planned arm: the same release image plus the column-511 correction (one changed file).

| Item | Comparator | Arm | Status |
|---|---|---|---|
| Image | release 0.9.1 (digest a9e985c0) | release + one copied file + bytecode layer; the release's 97 layers are an identical prefix | differs: the treatment |
| vLLM / B12x / DFlash2 / EXL3 | 0.1.dev20051+g487ecf187; B12x 7fcc094e; dflash2 b389ac29; exl3-vllm 30038602 | inherited layers | match by construction |
| CUDA in image; NCCL | 13.0.1; 2.30.7+cuda13.3 | same layers | match by construction |
| Driver, vbios, kernel, iommu=pt | 595.84; 98.02.6A.00.04; 7.0.0-30; on | same; same boot; no driver, kernel or docker package changes since 10-07 | match |
| Weights | snapshot 0490d2f7 | same; 21 files rehashed equal to their content addresses | match |
| Tokenizer, chat template | template sha 0c4099f3 | same | match; prompt token counts identical across all comparator passes |
| Parsers | glm45 / glm47 | same | match |
| Effective server args | fingerprint identical in both sessions (FULL_AND_PIECEWISE, captures 1-64, KV pool 4,707,515, TP2/EP1/DCP1) | same recipe and launcher, IMAGE only | match expected; checked against the arm's logs in the analysis |
| VLLM_* / B12X environment | same name set in both sessions | same files; no debug or probe variables | match expected |
| Sampling as sent | temperature 1.0, top_p 0.95, max_tokens 327,680; all 594 requests HTTP 200; no re-sent request | same chain | match |
| Seeds per pass | 1234 / 1235 / 1236 in the request bodies (pass 1 via the client default) | 1234 / 1235 / 1236 (pass 1 sent explicitly) | request bodies identical; immaterial |
| Concurrency and admission | lm-eval, 8 concurrent, timeout 10,800 s, max 3 retries; admission order identical across passes and engines; never waiting, no preemption, peak KV 39.6-41.6% | same | match; any re-sent request fails the analysis validity gate |
| Answer order | task hash 87e807b3 every pass | same cached dataset | match expected; checked by the analysis |
| Warm-up and readiness gate | release start-up warm-up; ready file plus API check; ready after 725 / 709 s | same | match |
| Server sessions | pass 1 fresh; passes 2-3 on a second fresh server | mirrored (amendment 11b) | match |
| Other GPU load | desktop graphics on GPU 1, about 2.2 GB, activity not recorded | about 2.3 GB; logged every 5 minutes | uncontrolled (inherent to a historical control) |
| Port binding | all interfaces | loopback | immaterial |
| Kpool tail-ring defect; NaN-byte index rows | present | present | same; stated limitation |

Verdict: no material unexplained configuration difference, given the image built from the release digest with the
recorded file hash and the mirrored session split. Unexplained or uncontrolled by design: other load on GPU 1, and
run-to-run timing and batch composition. The comparison remains a historical control, not a randomized experiment.
