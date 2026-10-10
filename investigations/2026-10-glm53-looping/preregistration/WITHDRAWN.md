# WITHDRAWN (2026-10-09)

The documents in this folder (`PREREGISTRATION.md`, `AMENDMENT-1.md`, `AMENDMENT-2.md`, `AMENDMENT-3.md` and the gate
record in `README.md`) planned the layout bisection of 2026-10-07/08. They are kept unchanged, so their recorded
SHA-256 hashes still verify, and they are all marked withdrawn by this note.

The bisection's instrument (`hard-prompt-screen/v1`) sent request seed 1234 on every repeat. The engine draws each
request's sampling noise from its seed. Concurrent batching made the repeats vary, but every repeat drew on the same
sampler noise, so that variation was not statistically meaningful. The bisection's failure rates, its
Fisher tests, its verdicts ("inconclusive", "no detectable loop effect") and the question-level observations drawn
from it are withdrawn. Only repeat 1 of each question from each configuration's first screen is kept, as a single
draw. The gate record's decode-vs-prefill values come from a different protocol and are not affected.

Notice, evidence and what replaced it: [`LEDGER.md`](../../../LEDGER.md#withdrawn-and-what-replaced-it).
