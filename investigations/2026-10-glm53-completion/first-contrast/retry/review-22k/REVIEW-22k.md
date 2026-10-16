# Bounded CPU review of the retry's failed gates (amendment 22k s.1)

Post-hoc. Written after the data, from the session's own records, CPU only. The frozen verdict stays **NO READING**,
nothing here is a retrospective PASS, and no pool, row or gate is removed. Scripts and outputs are in this folder.

## H5f-fc, like with like (`h5f_units.py`, `h5f_units.out`, `h5f_units.json`)

**The judged subset.** H5f judges each injected run's clean-labelled window pools (pool class 1 in that run's own
ledger) against the reference LW1. Across D1-D3 that is **176 distinct units** (fixture × rank × layer × window pool),
286 unit-runs:
- 110 units are clean-labelled in one D run, 22 in two and 44 in all three;
- by fixture: M4 88, L4 33, M3 33, L3 22; none on L1, L5, A8 or L8;
- by window pool: pools 0-8 at the start of the window, and six scattered later pools (1707-1791), 11 layers each.

Its composition follows the fault's exposure pattern. It is not a random sample of window pools.

**The same units in every comparison.** Each unit is measured in all of the following; a pool is gross above a
relative L2 of 0.25.

| Comparison on the same 176 units | Measured | Gross |
|---|---:|---:|
| D1-D3 vs LW1 (tested, 286 unit-runs) | 286 | **10** |
| C1 vs LW1 (clean arm) | 176 | 1 |
| C2 vs LW1 (clean arm) | 176 | 5 |
| C3 vs LW1 (clean arm) | 176 | 6 |
| N_dd: C1 vs C2 (decode null) | 176 | 6 |
| N_pp: LW2 vs LW1 (prefill reference null) | 176 | 4 |

**The 10 gross unit-runs are 4 units, all M4, all on window pools 3 or 5:**

| Unit | D runs | C1 / C2 / C3 | N_dd | N_pp |
|---|---|---|---|---|
| M4 rank 0 layer 23 pool 3 | 0.328 / 0.307 / 0.306 | 0.226 / **0.305** / **0.280** | 0.211 | **0.289** |
| M4 rank 1 layer 27 pool 3 | 0.532 / 0.449 / 0.373 | 0.236 / **0.464** / **0.354** | **0.431** | **0.444** |
| M4 rank 1 layer 27 pool 5 | D2 only: 0.304 | **0.309** / **0.357** / **0.334** | 0.195 | **0.271** |
| M4 rank 1 layer 43 pool 3 | 0.500 / 0.320 / 0.404 | 0.209 / **0.469** / **0.355** | **0.435** | **0.406** |

Each is gross in at least two of the three clean-arm runs and in the prefill reference null, and three of the four in
the decode null. Five further M4 units in the subset are gross in a clean-arm or null comparison but in no D run.

**M4's fragility flags, kept.**
- M4 is the fixture whose F6 pages are reported as not independently verified (12 of 2,486 judged pages in GATE-1R C1
  and C2, and in C3's residue).
- Those flags sit on M4's window pages 334, 336 and 337 (window pools 522-777), at the same deep layers (23-43).
- The gross H5f units sit on page 325 (window pools 0-9). It holds fewer than the 16 pools the page rule needs, so it
  was never judged and carries no flag. Its unit-level nulls above show the same fragility.

**Reading of the review.**
- On the units H5f actually judges, the injected runs' gross rate (10 of 286) lies within the spread of the clean
  arm and the nulls on the same units (1-6 of 176 per comparison).
- The failure is consistent with a pooled reference (1.51% over all window pools) applied to a subset that includes
  numerically non-discriminating units.
- It does not show that the ledger and the bytes agree. With 4 of 176 units fragile, the clean-labelled subset has
  little power to detect a labelling error either.
- H5g (placement) does not replace this content check.

## Sentinel boundary state (`sentinel_calib.py`, `sentinel_calib.out`; distances in `sentinel_rows.json`)

The distances come from `sentinel.state_rows` run unchanged on the session's snapshots
(`../diagnosis-fcr1/diag_sentinel.py`). The original component-wise failures stand: **SSM 3 of 544, convolution 13 of
544.**

**Exact equality.**
- In 48 rows per component, PS-a and PS-b are bitwise equal. Those are KDA layers 0-2, every fixture and both ranks.
- In all 48, PS-c is bitwise equal too.

**Per-component distances (496 non-identical rows).**

| | rel(PS-a, PS-b) | rel(PS-a, PS-c) | rel(PS-b, PS-c) |
|---|---|---|---|
| SSM median (p10-p90) | 0.0304 (0.0065-0.0743) | 0.0294 (0.0068-0.0745) | 0.0302 (0.0066-0.0694) |
| Conv median (p10-p90) | 0.0531 (0.0107-0.2012) | 0.0488 (0.0086-0.1598) | 0.0537 (0.0075-0.1386) |

**Both baseline orientations.**
- With reference PS-a (the gate), SSM 3 and conv 13 rows fail.
- With reference PS-b, SSM 3 and conv 12; 3 and 12 rows fail in both.
- In the failing rows, PS-c sits about equally far from both baselines, while the two baselines sit close together.

**Odd one out, judged symmetrically.** A phase is the odd one out when it is more than 2× farther from each of the
other two than they are from each other.

| Odd one out | SSM | Conv | Same fixture and layer on both ranks (conv) | Concentration (conv) |
|---|---:|---:|---:|---|
| PS-a (before the injection) | 3 | **38** | 18 | L3 32 |
| PS-b (before the injection) | 2 | 22 | 11 | L5 10 |
| PS-c (after the injection) | 3 | 12 | 6 | L4 4, others 2 each |

**Concentration.**
- SSM failures: L1 layer 5; M3 layer 6 on both ranks.
- Conv failures: layers 4, 6, 12, 13 and 16, across six fixtures; 12 of the 13 rows form 6 rank pairs (same fixture and layer on both ranks).

**Small denominators, partly supported.**
- Every failing row's rel(PS-a, PS-b) is in the lowest 26% of its own fixture's rows (1.6-25.8th percentile).
- Within its layer it is spread from the 6th to the 44th percentile.
- Of the 31 rows in the lowest within-layer decile of rel(PS-a, PS-b), 1 fails for SSM and 3 for conv. The remaining
  failures lie outside that decile.
- In most failing rows, rel(PS-a, PS-c) is in the upper part of its layer's distribution (31st-100th percentile).

**Reading of the review.**
- The single-pair 2× rule is unstable. Pre-injection phases are flagged as the odd one out as often as the
  post-injection phase (SSM) or more often (conv), and both-rank agreement occurs for every phase.
- So rank agreement marks a fixture- and layer-level state difference between restorations. It is not specific to
  the injection.
- This is evidence about the gate's calibration, not proof that contamination was absent.
- Only rebuilt state at P − 1 was captured: the cached KDA state at c itself was not read before and after treatment.
  An effect small enough to hide in the restorations' own spread is not excluded.

## Not done

- No revised gate is applied to this session, and no retrospective PASS is computed.
- No new GPU work.
