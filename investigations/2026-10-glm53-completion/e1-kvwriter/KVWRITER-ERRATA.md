# KVWRITER-PREP errata, recorded before any GPU data (2026-10-12)

**The frozen document is unchanged.** `KVWRITER-PREP.md` sha256 is
`545a0fe272a55319daabc17a354d3a38e1600b9923e4cded0d3a42b4bf99d8b5`, recorded in `freeze/` before the remaining CPU
engineering.

**How these errors were found.** Finishing that engineering included a CPU dry run of `kvw_analyze.py`. The dry run
used outputs synthesised from the frozen rounding model itself: the vLLM emulation, the B12x emulation with
`rcp = RN(1/scale)`, and a CPU bf16 `bmm`. By construction it contains no defect, so any failure in it is an error in
the criteria. It found three.

**Both verdicts will be reported.** `kvw_analyze.py` reports the frozen rules as written alongside the errata-corrected
rules, so the coordinator can decide which governs. No GPU output had been produced when this file was written.

## ERRATUM-1: P2 bound, bf16 term

**Frozen as written:** `|out − ref| ≤ 2⁻⁹·|ref| + 256·2⁻²⁴·Σ|q||w|`.

**The error:** half a bf16 ulp is 2⁻⁸·2^⌊log2|ref|⌋, which is up to 2⁻⁸·|ref|. The frozen term is only 2⁻⁹·|ref|, so a
correctly rounded bf16 output fails it in the lower half of every binade.

**Dry-run effect** (a correct CPU bf16 `bmm`):
- 5,691-104,184 elements outside the frozen bound per (layer, rows) case;
- max ratio to the frozen bound 1.95;
- under the corrected bound `½ulp_bf16(|ref|) + 256·2⁻²⁴·Σ|q||w|`: 0 outside, max ratio 0.99.

**Corrected P2 (`P2c`)** uses the exact half ulp.

## ERRATUM-2: W4 tolerance

**Frozen as written:** max dequant error ≤ ½ step + 1e-6.

**The error:** the frozen prediction itself allows a relative perturbation before rounding of δ ≤ 7·2⁻²⁴, from
`fl(1/448)`, the product, the 1-ulp `rcp` and the IEEE roundings. A value within δ of a midpoint may round the other
way. Its error is then ½ step + δ·|t|/step, and |t|/step < 16, so the bound is ½ + 6.7e-6 steps.

**Corrected tolerance:** ½ + 16·7·2⁻²⁴ ≈ 0.5000067 steps.

**Dry-run maxima:** 0.5000008 (vLLM) and 0.5000001 (B12x).

## ERRATUM-3: W3 rate, exact ties

**Frozen prediction:** 2e-6 to 1.3e-5 differing bytes per element. Rule: a rate above 1.3e-4 counts as beyond
rounding.

**The error:** the prediction treated exact ties as rare (only when amax = 448·2ᵏ). They are frequent. Latents and amax
are bf16, with 8-bit significands, so `fl(v/scale)` lands exactly on an E4M3 midpoint for 0.3-0.8% of elements.

**How each writer breaks a tie:**
- **vLLM:** exact IEEE arithmetic, so round-half-to-even;
- **B12x:** the `rcp.approx` perturbation pushes the value to one side, so about half of the ties go the other way.

**Dry-run, emulated rounding only, `rcp = RN`:**

| set | emulated differing rate | ties among elements | differences at ties | non-tie rate | elements where some rcp ±1 ulp outcome differs |
|---|---|---|---|---|---|
| synth | 6.7e-4 | 0.79% | 94% | 4.2e-5 | 0.58% |
| proxy L3 | 9.0e-4 | 0.57% | 98% | 1.6e-5 | 0.34% |
| proxy L23 | 1.12e-3 | 0.52% | 99% | 9.1e-6 | 0.32% |
| proxy L43 | 6.5e-4 | 0.53% | 90% | 6.5e-5 | 0.33% |

The rates above are all over the frozen 1.3e-4 threshold. The non-tie rates are below it, though L43 is about 5× the
frozen 1.3e-5 point estimate.

**Corrected W3 (`W3e`), using only quantities fixed before data.** Every GPU cross-writer byte difference must lie in
the set of elements where the emulated vLLM byte differs from at least one emulated B12x bracket outcome. Equivalently,
a difference must be explainable by the rcp ±1 ulp bracket. The non-tie difference rate must not exceed 1.3e-4.

The parts of W3 that stay unchanged and binding:
- every difference is exactly one code;
- scales within 1 ulp (non-zero groups);
- the RoPE tail is all zero.

## Unchanged

- **W1, W2, R and P1:** as frozen.
- **"Beyond expected rounding":** as frozen, with P2 → P2c, the W4 tolerance → 0.5000067, and the W3 rate rule → W3e.
