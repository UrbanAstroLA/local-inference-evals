# Datasheet

Following the questions of "Datasheets for Datasets" (Gebru et al.).

**Motivation.** To measure locally served LLMs end to end (accuracy, completion, speed, kernel correctness) under
fixed, versioned protocols, so that configurations can be compared like for like and claims can be checked.

**Composition.** One directory per run (`runs/<id>/`): a manifest, one JSON line per item, and a summary recomputable
from those lines. Items are benchmark questions (GPQA Diamond: pass 1 of every configuration, and three passes with distinct request seeds
for three configurations), screen requests, probe requests, kernel tests,
decode positions compared with prefill, attention-index check cases, or tool-calling scenarios. Screens under
`hard-prompt-screen/v2` hold 12 independent draws of one question (distinct request seeds); the earlier `v0`/`v1`
screens hold one draw per question (repeat 1), because every repeat sent the same seed.
Rows contain ids, hashes, outcomes, token counts, timings and compression ratios. They contain **no benchmark text
and no model output text.** Some runs add `server_log.jsonl`, numeric fields parsed from the engine's server log (KV
pool, throughput, acceptance, waiting requests); the log text is not published. Field definitions, and the rule that names configurations (weights · engine version ·
speculation): `SCHEMA.md`.

**Benchmark terms.** GPQA (Idavidrein/gpqa, CC BY 4.0) asks that examples not be revealed in plain text or images
online. Accordingly this repository holds only lm-evaluation-harness `doc_id`, `doc_hash`, `prompt_hash` and
`target_hash` values and SHA-256 hashes of responses. To verify a run, regenerate prompts from the dataset with the
protocol's harness settings and compare hashes.

**Collection process.** Models: GLM-5.3-Flash EXL3 weights tpurtell's K3.25 checkpoint wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 (`3.25bpw`) and Brandon M. Music's TR3
checkpoint brandonmusic/GLM-5.3-Flash-tr3-4bpw (`4bpw TR3 (Brandon)`). Two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs (96 GB each, PCIe), driver
595.84, one machine, served through Docker images pinned by digest. Clients and settings are in `tools/clients/` and
`protocols/`. Decision rules for each investigation were written before its data (`investigations/*/preregistration*`).
Known collection issues are recorded in each run's `notes` (for example a run kept as INVALID after an engine crash).

**Preprocessing.** Exported by a local script that copies numeric fields, computes hashes and drops all text and
local paths. It keeps GPQA passes only when each has its own request seed (pass *p* sends 1233 + *p*) and repeat 1 of
each question in the fixed-seed screens; `tools/verify.py` fails if a GPQA pass carries another seed or another repeat
appears. Server-log figures of GPQA runs cover only the measured requests' time window (the engine's start-up warm-up
requests are excluded). The headline score is lm-eval's raw `flexible-extract` filter;
`strict-match` is published too but records whether a reply used the phrase "The answer is", not accuracy. `flexible-extract` scores some correct answers as wrong on particular questions (0.5 to 3.5 points per pass in
the published passes); see the known limitation in `protocols/gpqa-diamond/v1.md`. A secondary, audited score
(`correct_stated`: whether the reply's stated final answer is right) was determined from the reply text, which is not
published; only the per-row judgement is. Passes added on 2026-10-10 were judged by the same script and hand check (see
the protocol). Likewise the greedy shared-prefix lengths of the serving probe are derived from
unpublished output text; output hashes are published.

**Uses.** Comparing configurations, including different engines, under the same protocol and hardware (enforced by `tools/verify.py`), reproducing
the findings, or as a baseline for other hardware. **Not suitable** for: ranking models in general, comparing with
scores from other harnesses as if equal, estimating benchmark-wide failure rates from the hard-question screen (it samples
the hardest items, one question per run), reading the single draws of the v0/v1 screens as rates, measuring the
accuracy cost of quantization (no higher-precision reference was run on this hardware), or comparing individual
responses between runs (the engine is not bitwise reproducible even one request at a time, and concurrent batching adds
variation), or treating the passes of one GPQA question as independent draws in a test (they share the question; compare
with questions as clusters).

**Distribution.** Public repository. Results and docs CC BY 4.0, code Apache-2.0. Model weights, engines and
datasets are not redistributed.

**Maintenance.** Corrections and withdrawals are new commits recorded in `CHANGELOG.md` and summarised in `LEDGER.md`; withdrawn rows stay in the
git history. Protocol changes get a
new version number; old runs are never rescored under a new version.

**See also.** [`README.md`](README.md) (start here), [`FINDINGS.md`](FINDINGS.md) (statements, graded),
[`CONFOUNDS.md`](CONFOUNDS.md) (confounds and controls), [`LEDGER.md`](LEDGER.md) (dated account), [`SCHEMA.md`](SCHEMA.md) (fields).
