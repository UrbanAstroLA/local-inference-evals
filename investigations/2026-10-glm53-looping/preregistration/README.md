> **Withdrawn (2026-10-09).** The layout bisection planned here used one request seed on every repeat; its results
> were withdrawn. The documents are kept unchanged so their hashes verify. See [`WITHDRAWN.md`](WITHDRAWN.md).

# Layout bisection: preregistration and amendments

**Names used in these files:** arm A = `3.25bpw · tpurtell 0.9.0 · DFlash2 ×3` (v0.9.0 as released: DCP1 + MLA layer
ownership, TP2 experts); arm D = `3.25bpw · tpurtell 0.9.0 · 0.7.0 layout (DCP2, EP2) · DFlash2 ×3` (a diagnostic
control); arm E = `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix · DFlash2 ×3`. Screens A1/A2, D1/D2, E1/E2 are the runs
ending `_screen-bisect1` / `_screen-bisect2` of those configurations. "S30 v0, S7, S7p" are the tpurtell 0.7.0 screens
of 2026-09-30 and 2026-10-05.

The files are published as written, with these exceptions. `AMENDMENT-2.md` had a local image name and a local
directory replaced by "(local overlay image)" and "(local directory)". The other three are byte-identical to the
originals. SHA-256 of the originals, recorded when each was written:

| File | SHA-256 | Written |
|---|---|---|
| `PREREGISTRATION.md` | `fb32e89e9b5fe0c29caa0851e8372e8db51e9720f3a199ade32bc730d5af800b` | 2026-10-07, before any data |
| `AMENDMENT-1.md` | `dfa160a5c266e64e941a5db9fc92de5344521df95c167b07dc7267cb38ab3cab` | 2026-10-07 13:35 PDT, before any data from the v0.7.0-layout configuration. Its timestamp was corrected from 13:50 to 13:35 and the file re-hashed before that configuration served a request |
| `AMENDMENT-2.md` | `8fc02c0384986f0368bc6193e7596c4dd4c10848461cff159726628d5bd56739` (original; this copy `bcc4be20…`) | 2026-10-07, before the tail-fix GPU validation and before any data with the fix |
| `AMENDMENT-3.md` | `1ac621cc65f4329b1cfed79c9b12937863e67be4820245ddb6cad6fb207f8036` | 2026-10-07, **after** the tail-fix validation, before any screen with the fix |

## Gate evaluation for arm E (record written 2026-10-07, after the validation, before any arm-E data)

Amendment 2's gate (b) needed stock decode to disagree more with prefill at positions i < 2044 with i % 4 ≠ 0 than at
i % 4 = 0, and the fix to remove that pattern. Values from the two `decode-prefill-consistency/v1` runs:

| Positions i < 2044 | Stock: mean KL / mean abs Δlogprob | With the fix |
|---|---|---|
| i % 4 = 0 | 0.0577 / 0.1426 | 0.0090 / 0.0443 |
| i % 4 = 1 | 0.0636 / 0.1687 | 0.0115 / 0.0510 |
| i % 4 = 2 | 0.0728 / 0.1671 | 0.0086 / 0.0463 |
| i % 4 = 3 | 0.0683 / 0.1640 | 0.0121 / 0.0559 |

- Gate (a), the index check: **passed**. Stock drops the tail at lengths 1001-1003, and the fix attends it.
- Gate (b): **failed**. Stock does sit higher at i % 4 = 1-3 than at 0, but the fix shows the same pattern, so the
  predicted signature does not separate them.
- Reported beyond the gate: mean KL below 2,044 was 0.0656 on stock vs 0.0103 with the fix; from 2,048, 0.0307 vs 0.0187.
- The gate as written meant: no arm E, continue with A2 and D2.
- Amendment 3 records the decision, taken after seeing these values, to run arm E anyway.

Results and context: [`../README.md`](../README.md).
