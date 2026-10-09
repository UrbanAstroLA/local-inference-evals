# Datasheet

Following the questions of "Datasheets for Datasets" (Gebru et al.).

**Motivation.** To measure locally served LLMs end to end (accuracy, completion, speed, kernel correctness) under
fixed, versioned protocols, so that configurations can be compared like for like and claims can be checked.

**Composition.** One directory per run (`runs/<id>/`): a manifest, one JSON line per item, and a summary recomputable
from those lines. Items are benchmark questions (GPQA Diamond), screen requests, probe requests, kernel tests, decode positions
compared with prefill, attention-index check cases, or tool-calling scenarios.
Rows contain ids, hashes, outcomes, token counts, timings and compression ratios. They contain **no benchmark text
and no model output text.** Field definitions, and the rule that names configurations (weights · engine version ·
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
local paths. Scores are lm-eval's raw `flexible-extract` and `strict-match` filters, not rescored. `flexible-extract` scores some
correct answers as wrong on particular questions (about 0.7 points per pass on average); see the known limitation in
`protocols/gpqa-diamond/v1.md`.

**Uses.** Comparing configurations, including different engines, under the same protocol and hardware (enforced by `tools/verify.py`), reproducing
the findings, or as a baseline for other hardware. **Not suitable** for: ranking models in general, comparing with
scores from other harnesses as if equal, or estimating population failure rates from the hard-question screen (it
deliberately samples the hardest items), quoting the screen's cross-engine gap as an effect size (its items
were selected from one engine's failures), or comparing individual responses between runs (concurrent batching
makes outputs nondeterministic even with a fixed seed).

**Distribution.** Public repository. Results and docs CC BY 4.0, code Apache-2.0. Model weights, engines and
datasets are not redistributed.

**Maintenance.** Runs are append-only; corrections are new commits recorded in `CHANGELOG.md`. Protocol changes get a
new version number; old runs are never rescored under a new version.
