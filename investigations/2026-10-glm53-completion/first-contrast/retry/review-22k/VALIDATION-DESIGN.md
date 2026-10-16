# Prospective validation design for the force-history instrument (amendment 22k s.3)

For future use only. Not implemented and not GPU-validated. It is adopted, with any changes, only in a dated amendment
for a specified experiment, before that experiment's data. It records what the retry taught about gate construction.

## 1. Exact structural checks (unchanged in kind)

These passed in the retry and stay exact, with zero tolerance:
- forcing and position;
- request and block attribution;
- injection placement (H5g);
- cache safety S1-S4 and treatment balance;
- shared-block identity;
- MLA and index-K page bytes on the shared block.

A missing measurement counts as incomplete coverage, never as a pass.

## 2. Content controls matched to the judged subset

The retry's H5f judged a small, non-random subset (176 units) against a rate pooled over all window pools. Instead:

- **Define the subset first, then its null.** For each injected run, the unit list (fixture × rank × layer × pool)
  that the content check will judge is fixed by the ledger. Its null is computed on **the same units** in the clean-arm
  runs and in the prefill and decode null pairs. No pooled rate is used in its place.
- **Discriminating units.** A rule fixed before data excludes a unit whose own null cannot separate a fault, for
  example a unit gross in the prefill null or in two clean-arm runs.
  - Every exclusion is listed by identity, and coverage is reported (judged / excluded / total).
  - The rule is checked against planted faults so that it cannot exclude a faulty unit by construction.
- **Minimum judged size.** If fewer than a fixed number of discriminating units remain (set from planted-fault power,
  not from the data), the check is INCOMPLETE rather than PASS or FAIL.
- **Counts and identities** of gross units are reported beside every rate.

## 3. Multiple clean sentinel baselines

The retry's sentinel judged each row against one pre-injection pair, so the limit carried the noise of a single
difference. Under a symmetric odd-one-out count, the pre-injection phases were flagged as often as or more often than
the post-injection phase. Instead:

- **Several baselines.** At least three pre-injection restorations (PS-a1..a3) and at least two post-injection ones
  (PS-c1, PS-c2), interleaved where the schedule allows, so restoration order is not confounded with treatment.
- **Per-component limits from the pooled spread.** SSM and convolution are kept separate (as in 22f). For each
  component and row, the limit comes from the spread of all pre-injection pairs: for example the maximum pairwise
  distance among the baselines times a factor fixed by planted-fault calibration, with a floor per layer from the
  pooled distribution.
- **Symmetric statistic.** The pre-registered statistic compares post-injection restorations with the baselines as a
  group, and the same statistic computed with each baseline held out serves as its own null.
- **Exact equality kept.** Where all baselines are bitwise equal, post-injection states must be bitwise equal.

## 4. Direct inspection of cached state where feasible

Rebuilt state at P − 1 adds the numerical variation of re-prefilling c..P − 1. Where the hook can do it without
writing:
- read the **cached KDA state block at c itself** (SSM state and convolution window) before the injected run and after
  it, with its physical block identity recorded, and compare bytes exactly;
- read the shared MLA / index-K pages the same way (as the retry already did).

The rebuilt-state comparison then becomes a secondary check, not the gate.

## 5. Sensitivity to planted faults

Before freezing, each content and sentinel check must detect, on real session records:
- a single-unit fault inside the judged subset;
- a fault confined to non-discriminating units, to show that it is reported as excluded rather than silently passed;
- a small post-injection offset in one KDA component, at the size of the baselines' median spread;
- a wrong or stale cached state block at c.

Clean controls must stay clean, including a plant where the post-injection restoration is merely a different but valid
schedule. Detection rates and false-alarm rates are reported per check.

## 6. Cost note

A session with five sentinel restorations instead of three adds two restorations of about 28 s each on one fixture
panel: under 0.05 GPU-hours [estimate from the retry's PS-phase times].

The content and planted-fault work is CPU only. Its main cost is calibration time on records that already exist,
provided the tensor records it needs are kept.
