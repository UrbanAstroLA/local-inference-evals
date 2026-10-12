# KV record writer and query projection, and E1 (amendments 16-16b)

Two items from amendment 16's parallel work: a model-free GPU comparison of the KV record writer and the query projection
between checkpoint B0 and stock 0.7.0, and a CPU-only assessment of E1 (batch-size-dependent routing weights).

**Results.**
- **KV writer and query projection: within expected rounding** under the criteria with the errata recorded before any GPU
  data (the operator decided that this verdict governs). B0's 528-byte record writer and 0.7.0's 656-byte writer store
  the same bytes except at 0.06-0.11% of elements, each exactly one E4M3 code apart and every one explained by the
  frozen rounding model (mostly exact rounding ties, which the two writers break differently); group scales within 1 ulp;
  0.7.0's RoPE tail all zero. The two query projections are bitwise equal on the 512 shared lanes. Under the criteria as
  first written, the verdict is "beyond expected rounding", with failures that reproduce the counts a CPU dry run had
  produced on defect-free outputs. 0.0069 occupied GPU-hours (ceiling 0.25). No B1.
- **E1: expected numerical variation** of the upstream router kernel used above 1,024 batch rows, not a defect. Expert
  selection cannot be affected; the same kernel binary is in every release; no workaround. CPU only.

Plan and rules: [`../PLAN.md`](../PLAN.md#dated-amendments), amendments 16, 16a (authorization: one GPU per image, ceiling
0.25 GPU-hours, after the all-reduce fixture) and 16b (results and the operator decision). Investigation page:
[checkpoint B0 and targeted invariants](../README.md#checkpoint-b0-and-targeted-invariants-amendment-16).

## KV writer and query projection

### What it tests

Whether the record and query path that changed between the engines (item D2 of the
[DCP audit](../audits/phase1-dcp-attention.md): "FP only, unverified") differs by more than expected rounding:

| Component | B0 (0.9.1 bundle, 528-byte record) | 0.7.0 (656-byte record) |
|---|---|---|
| KV writer | B12x `concat_and_cache_glm_next_mla`: 512 E4M3 + 4 fp32 group scales; `scale = amax·fl(1/448)`, `byte = e4m3(v·rcp.approx(scale))` | vLLM `concat_and_cache_mla(fp8_ds_mla)`: the same 528 bytes plus a 128-byte bf16 RoPE tail held at zero; IEEE divisions |
| Query absorption | `torch.bmm`, 512 wide | B12x H64 projection for 2-16 rows (576 wide, 64 zero lanes); `torch.bmm` otherwise |

The vLLM writer library is byte-identical in both images, so both writers ran in the B0 image and the vLLM writer ran
again in the 0.7.0 image. One container per image, one GPU, no server and no checkpoint load.

**Inputs** (sha256 `ed4498e6…`, built on CPU by [`kvw_prep_inputs.py`](kvw_prep_inputs.py); withheld, hash in the raw
list): real-weight proxies at MLA layers 3, 23 and 43 (latents from captured bf16 hidden states through the checkpoint's
own projections and norms, 4,869 / 2,368 / 2,501 tokens; queries 64 heads × 64 rows × 256), plus a 4,096-token synthetic
set with all-zero groups, single outliers, tiny groups and values placed on E4M3 midpoints. Realistic in distribution,
not bitwise to serving: no MLA latents or queries were ever captured from a server.

**References.** Writers: fp64 values of the bf16 latents (not either writer's dequantised records, which would favour
one side). Projection: fp64 `bmm` of the identical bf16 inputs.

### Criteria, and the errata found before GPU data

The metrics and decision rule were frozen in [`KVWRITER-PREP.md`](KVWRITER-PREP.md) (local sha256 `545a0fe2…`, recorded
in [`freeze/`](freeze) before the remaining engineering): W1 vLLM GPU bytes equal a CPU emulation; W2 B12x bytes inside
the emulated `rcp` ±1 ulp bracket; W3 cross-writer differences (rate, size, scales, RoPE tail); W4 dequantisation error;
R repeatability and slot independence; P1 H64 equals `bmm`; P2 each projection within a bound of fp64.

Finishing the CPU engineering included a dry run of the frozen analysis on outputs synthesised from the frozen rounding
model itself, so defect-free by construction. It failed, which can only mean errors in the criteria. Three were found
and recorded in [`KVWRITER-ERRATA.md`](KVWRITER-ERRATA.md), hashed in [`freeze/pre-gpu.sha256`](freeze/pre-gpu.sha256)
at 21:54 UTC, before the GPU runs at 22:10:

| Erratum | As written | Error | Corrected |
|---|---|---|---|
| 1, P2 bound | `2⁻⁹·|ref| + 256·2⁻²⁴·Σ|q||w|` | half a bf16 ulp is up to 2⁻⁸·\|ref\|, so a correctly rounded output fails in the lower half of every binade | exact half ulp (P2c) |
| 2, W4 tolerance | ½ step + 1e-6 | the frozen error model itself allows ½ + 6.7e-6 steps | 0.5000067 steps |
| 3, W3 rate | 2e-6 to 1.3e-5 differing bytes per element; above 1.3e-4 counts as beyond rounding | omitted exact E4M3 ties, which bf16 inputs hit on 0.3-0.8% of elements; vLLM breaks them to even, B12x by its `rcp` perturbation | every difference must be explainable by the frozen `rcp` bracket, and the non-tie rate must stay below 1.3e-4 (W3e) |

The analysis reports both verdicts; which governs was left to the operator, who chose "with errata" after the results
(amendment 16b). Everything else stayed as frozen, including "every difference exactly one code", scales within 1 ulp
and the zero RoPE tail.

**Loader wrapper.** The runner's analysis step failed to load the result files: torch's weights-only loading rejects a
version object (`torch.torch_version.TorchVersion`) that this investigation's own GPU script wrote into them. The frozen
`kvw_analyze.py` was re-run unchanged through [`kvw-runs/analyze_wrapper.py`](kvw-runs/analyze_wrapper.py), which allows
that one type, before any analysis output existed; the frozen file hashes verified ([`runner.log`](kvw-runs/runner.log)).
The re-run overwrote the failed step's `analysis.log`; the runner's [`SHA256SUMS`](kvw-runs/SHA256SUMS) still records the
failed log's hash. The published `kvw_analyze.py` is byte-identical to the frozen file.

### Results ([`kvw-runs/kvw_analysis.json`](kvw-runs/kvw_analysis.json))

| Check | Synthetic | Layer 3 | Layer 23 | Layer 43 |
|---|---:|---:|---:|---:|
| Elements | 2,097,152 | 2,492,928 | 1,212,416 | 1,280,512 |
| W1 vLLM writer vs emulation, both images | 0 mismatches | 0 | 0 | 0 |
| W2 B12x bytes outside the bracket; scales equal emulation | 0; yes | 0; yes | 0; yes | 0; yes |
| W3 cross-writer differing bytes (rate) | 1,409 (6.7e-4) | 2,254 (9.0e-4) | 1,362 (1.12e-3) | 830 (6.5e-4) |
| of which at exact ties | 94% | 98% | 99% | 90% |
| non-tie rate (W3e limit 1.3e-4) | 4.2e-5 | 1.6e-5 | 9.1e-6 | 6.5e-5 |
| differences larger than one code; outside the emulated bracket | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| W4 max dequant error, vLLM / B12x (steps) | 0.5000008 / 0.5 | 0.5000008 / 0.5000001 | 0.5000008 / 0.5 | 0.5000008 / 0.5 |
| R repeatable, slot independent | yes | yes | yes | yes |

Group scales within 1 fp32 ulp on non-zero groups; RoPE tail all zero; the two writers' RMS dequantisation errors equal
to 8 digits.

Projection, layers 3, 23 and 43 at rows 1, 2, 4, 8 and 16, in each image: H64 `[..., :512]` equals `torch.bmm` bitwise
(P1, 0 differing elements), tail lanes exactly 0, both repeatable; under the corrected bound (P2c) 0 elements outside for
either path (largest error 0.979-0.990 of the bound). Under the bound as first written (P2), 5,691-104,184 elements per
case lie outside, identical across images and between `bmm` and H64: the dry run's defect-free counts.

| Verdict | Failures |
|---|---|
| **With errata (governs)**: not beyond expected rounding | none |
| As first written: beyond expected rounding | 64: the four W3 rates (6.48e-4 to 1.12e-3) and 60 P2 counts, all reproducing the dry run on defect-free outputs |

Cost: B0 12 s, 0.7.0 13 s, one GPU each: **0.0069 of 0.25 occupied GPU-hours** ([`costs.jsonl`](costs.jsonl)); no
repeat was needed.

**Rules down:** the KV record writer and the query projection as a 0.7.0-vs-B0 mechanism beyond rounding, for these
inputs and shapes; DCP-audit item D2 moves from "FP only, unverified" to "measured equal within rounding". **Does not rule
out:** a completion effect of rounding-level differences (the writers break exact ties differently); latents and queries
captured from a live server (these are proxies); the sparse-MLA readers of the records; the record size's other
consequences (page size, pool size and where 2³¹ bytes falls; see [`../b0-cache/`](../b0-cache)).

## E1: batch-size-dependent routing weights

[`E1-ASSESSMENT.md`](E1-ASSESSMENT.md); CPU only, on the stage A captures of stock 0.9.1 (1,593 records, 103,608 rows) and
the bundle gate's B0 captures (500 records, 34,672 rows), in CPU-only containers of the B0 image.

- **Mechanism.** vLLM's grouped top-k router picks one of two kernels by row count. At ≤ 1,024 rows the block kernel
  weights experts from the stored sigmoid `s`; above, the warp kernel rebuilds `s` as `(s + b) − b` in fp32. This model's
  correction biases `b` are 5.0-14.5, so the rebuilt score sits on an absolute grid of 2⁻²¹ to 2⁻²⁰. The image's own
  SASS matches the source ([`sass-excerpts.txt`](sass-excerpts.txt)).
- **Reproduced.** A CPU emulation of the warp arithmetic reproduces the served weights bitwise on 98.0% of rows above
  1,024 (98.2% on B0); the block arithmetic reproduces 0 of those rows and vice versa. The residual comes from the
  hardware's approximate exp/reciprocal inside the sigmoid. Alternative arithmetics (other summation orders, fp64
  renormalisation, bf16 or fp16 intermediates) reproduce none.
- **Within its own error model.** Against an error bound stated before checking, 0 of 729,600 weights exceed the warp
  model (largest ratio 0.83; B0 the same); 35% exceed the block model's bound. Against fp64: median absolute error
  1.5e-7 above 1,024 rows vs 1.4e-8 below; maximum 1.3e-4 absolute, 1.1e-3 relative, on tokens whose eight selected
  scores sum to only 0.009-0.18.
- **Selection unaffected by construction** (both kernels select from the same biased scores; 0 differences on 103,608
  and 34,672 rows). The weight effect (p99 5.6e-7) is about three orders of magnitude below D3's (about 1.6e-3).
- **Same in every release.** The router library has the same sha256 prefix in the stock 0.7.0 image, 0.8.0, a 0.9.1-based image and B0; B0's
  diff does not touch it. It cannot by itself explain a version difference. 0.7.0's router invocation was not re-traced.
- Classification: expected numerical variation of a design choice in the upstream warp kernel (a precision limit for
  models with large correction biases, and a batch-invariance property), not a malfunction. No workaround; invoking the
  router at the serving batch size stays the only accommodation in offline checks. Completion effect untested.

## Files

- KV writer: [`KVWRITER-PREP.md`](KVWRITER-PREP.md) (frozen design and criteria), [`KVWRITER-ERRATA.md`](KVWRITER-ERRATA.md),
  [`freeze/`](freeze) (hashes recorded before engineering and before GPU data), [`kvw_prep_inputs.py`](kvw_prep_inputs.py),
  [`kvw_gpu_compare.py`](kvw_gpu_compare.py), [`kvw_analyze.py`](kvw_analyze.py) (emulators and both verdicts),
  [`run_kvw.sh`](run_kvw.sh), [`kvw-inputs/`](kvw-inputs) (input hash, emulator self-test), [`kvw-runs/`](kvw-runs)
  (analysis JSON and log, loader wrapper, per-image GPU logs and budget lines, runner log, hashes),
  [`kvw-runs.stdout`](kvw-runs.stdout), [`costs.jsonl`](costs.jsonl).
- E1: [`E1-ASSESSMENT.md`](E1-ASSESSMENT.md), `e1_*.py` and their JSON outputs on the stage A captures,
  [`b0-gate-s1/`](b0-gate-s1) (the same on B0's captures), [`sass-excerpts.txt`](sass-excerpts.txt) (written for this
  repository from the full listings).
- [`RAW-SHA256SUMS-kvwriter.txt`](RAW-SHA256SUMS-kvwriter.txt): sha256 of local files not published (the prepared inputs,
  the two GPU result tensors, pre-flight listings, the full SASS listings, compile caches).
- Public copies replace local paths and image-name prefixes (`<runs>`, `<models>`, `local/…`), so `KVWRITER-PREP.md`,
  `E1-ASSESSMENT.md`, `run_kvw.sh`, `runner.log` and `kvw-runs.stdout` are not byte-identical to the local files;
  [`../PUBLIC-COPIES-16.txt`](../PUBLIC-COPIES-16.txt) lists both hashes. The frozen hashes in `freeze/` and the amendments
  refer to the local files.
