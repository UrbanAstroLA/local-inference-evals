# Long-reply completion: close-out (amendment 22al)

**Status: closed as unresolved, with positive findings.**

This allocation closes on the independent review's recommendation, relayed by the operator. E is kept as suggestive
evidence that needs independent confirmation. The frozen readings are unchanged: B+ is R-I (after its disclosed,
outcome-blind gate amendment 22v) and E is R-I without SMALL. No threshold was relaxed and no one-sided test was
substituted.

## Summary (the review's wording)

> In B+, the stock engines had identical aggregate stopping rates on selected 0.7.0-generated prefixes. In E, 0.7.0
> stopped more often on selected 0.9.1-generated prefixes: +13.5 percentage points, with five prefix-level differences
> favoring 0.7.0 and none favoring 0.9.1. Neither experiment met its frozen criterion for an engine effect or state
> dominance. The cross-experiment pattern suggests possible dependence on prefix source, but does not establish an
> interaction or locate its cause.

Separately, as a hypothesis:

> One hypothesis is that the engine bundles respond differently to features of 0.9.1-generated histories after prefill
> reconstruction. Differences in selected histories, reconstruction, and sessions remain alternative explanations.

**Reporting E1.** E1's positive bootstrap interval (+0.038 to +0.250) and its exact sign-flip p (0.0625) come from
different inferential constructions; both are reported. With five discordant prefixes, all in one direction, 0.0625 is
the smallest two-sided p the exact test can return. That explains the result's discreteness. It is not grounds to
override the rule.

**A correction to earlier framing.** Earlier exploratory summaries pooled B0 and B1 to P(finish | ≥ 100k) = 0.12 on 17
replies (amendment 19f). These results replace any suggestion of a near-universal completion cliff on the corrected
engines. In B+, B1 stopped on 25 / 48 continuations after reconstructing 100k-token histories. The remaining problem is
conditional on history, and possibly on how the history was constructed.

## What was established (descriptive unless stated)
1. **Same terminal-loop forms on both stock engines.** Capped long replies end in exact short-period loops. On 0.7.0
   it was 8 of 10 in the replication; on 0.9.1 it was 6 of 6 in D and 7 of 8 in D2. Most loops begin well after 100k.
2. **Some observable differences precede terminal repetition.**
   - Interjection-class phrases rise in capped replies on both engines.
   - Low characters per token with a high digit share appears on 0.7.0 and in D2, but only weakly in D.
3. **Prefix identity strongly predicts continuation outcomes in these selected samples.** η² was 0.67 (B+) and 0.72
   (E), as between-prefix variation in the observed samples, not a causal share. Prefixes from originally finished
   replies mostly stop again; prefixes from originally capped replies mostly do not.
4. **Prefill-reconstructed long histories can finish on every tested engine** (stock 0.7.0, stock 0.9.1, B1).
5. **E shows a directional stock-engine contrast** (+0.135) on 0.9.1-generated prefixes. Neither the frozen
   engine-effect rule nor SMALL is satisfied.
6. **Additional stops are not equivalent to rescued correct answers.** See the table below.

## E answer audit over all attempted continuations

| Outcome in E | 0.7.0 | 0.9.1 |
|---|---:|---:|
| Stopped with extracted final answer | 22 / 52 — 42.3% | 15 / 52 — 28.8% |
| Correct final answer, all attempts | **12 / 52 — 23.1%** | **10 / 52 — 19.2%** |
| Incorrect final answer, all attempts | 10 / 52 — 19.2% | 5 / 52 — 9.6% |
| Did not meet stopping endpoint | 30 / 52 — 57.7% | 37 / 52 — 71.2% |
| Correct among stopped | 12 / 22 — 54.5% | 10 / 15 — 66.7% |

- **Size of the differences:** the descriptive correct-completion difference is +3.8 percentage points, much smaller
  than the stopping difference. This establishes neither equivalence nor superiority on correctness.
- **Aggregate, not paired:** 0.7.0 had seven more stops and two more correct answers in aggregate. These are not seven
  individually paired rescues.
- **Capped-origin prefixes:** 4 stops (0.7.0) against 1 (0.9.1), all incorrect.

## What remains open
- **Does the stock-engine contrast depend on prefix source?** That is, does an interaction exist, and where does it
  come from (history features, prefill reconstruction, or session)?
- **Does a history's construction path matter?** The original decode-built state against prefill reconstruction was
  never tested. Option C would test only B1's construction path; it does not reproduce stock 0.7.0's or 0.9.1's
  defective decode-built state, and it is unfunded.
- **Does any stopping difference translate into more correct answers per GPU-hour?** That is the outcome a future
  decision would turn on.
- **The original question is still open.** The historical 4-versus-16 empty-reply contrast replicated only partly
  (10 of 594 at fresh seeds; O3, amendment 22p), and no mechanism is established.

## Requirements before any interaction study (from the review)
A new proposal is needed; this is not an extension that keeps sampling until E becomes significant.

| Element | Minimum requirement |
|---|---|
| Primary target | Define the interaction explicitly; specify whether the decision concerns stopping or correct completion |
| New evidence | A fresh confirmatory cohort; B+ and E inform design but are not silently pooled into confirmation |
| Prefix acquisition | Multiple source-generation sessions per engine; selection rules fixed before outcomes are inspected |
| Question matching | A question-matched core with independently generated histories from both sources; report the selection induced by requiring both to reach 100k |
| Original outcome strata | Fix sampling and weighting prospectively, so different capped / finished mixtures cannot masquerade as a source effect |
| Continuation sessions | Multiple sessions per engine, with randomized or balanced AB / BA ordering and prefixes distributed across blocks |
| Replication allocation | Prioritize additional questions and histories over many more seeds on the same few prefixes |
| Analysis | Account for question, history and session clustering; disclose the assumptions needed to interpret source differences |
| Sample size | Simulate power for a prespecified, practically meaningful interaction, including session variability; at least 80% power for the effect that would change the decision |

- **Funding test:** a larger study is justified only if a concrete decision depends on it, for example whether
  switching continuation engines improves correct answers per GPU-hour enough to justify the complexity. More
  precision about stopping alone does not meet that test.
- **What a matched design can and cannot show:** it still estimates behaviour among selected long histories, not the
  total causal effect of choosing a source engine.

## Spend
Long-reply line (zero-GPU discovery aside), 36.36 GPU-hours:
- B+ 13.06;
- D 5.88;
- D2 7.16;
- E 10.24.

Investigation cumulative: 128.31 GPU-hours.
