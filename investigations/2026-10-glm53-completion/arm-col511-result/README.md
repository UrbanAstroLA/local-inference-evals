# Completion arm: result receipts

`result-col511.json` has local paths replaced by placeholders; the frozen script's output is otherwise unchanged.

## Final stated figure (amendment 12)

The frozen script counts 12 arm rows where the stated answer differs from raw `flexible-extract`; under the protocol
(`protocols/gpqa-diamond/v1.md`, `correct_stated`) the stated figure is final only after each of them is read by hand.
They were read on 2026-10-11 under the existing rules only: a reply that commits to no single option counts as not
correct; otherwise the adjudicator's reading stands. Reply text is not published.

| Question-pass | Hand verdict | Why raw differs |
|---|---|---|
| q98p1 | commits to one option, the adjudicator's | raw took a parenthesised non-option token (formula group) |
| q102p1 | commits to one option, the adjudicator's | raw took an option mentioned after the commitment |
| q159p1 | commits to one option, the adjudicator's | raw took an option mentioned after the commitment |
| q4p2 | commits to one option, the adjudicator's | raw took an option mentioned after the commitment |
| q64p2 | commits to one option, the adjudicator's | raw took a parenthesised non-option token (stereodescriptor) |
| q72p2 | commits to one option, the adjudicator's | raw took an option mentioned after the commitment |
| q110p2 | commits to one option, the adjudicator's | raw took a parenthesised non-option token (stereodescriptor) |
| q152p2 | commits to one option, the adjudicator's | raw took a parenthesised non-option token (stereodescriptor) |
| q170p2 | commits to one option, the adjudicator's | raw found no parenthesised label |
| q70p3 | commits to one option, the adjudicator's | raw found no parenthesised label |
| q98p3 | commits to one option, the adjudicator's | raw took a parenthesised non-option token (formula group) |
| q134p3 | commits to one option, the adjudicator's | raw found no parenthesised label |

None is hedged, so the frozen script was re-run unchanged with no hedged rows (`--arm-hedged` empty) and the same
inputs as the first run. `result-col511-final.md` and `result-col511-final.json` are that run's output (placeholders as
above); their content equals `result-col511.*`. Final: stated 515 of 594 (86.7%) with the correction vs 522 (87.9%) on
the comparator, -1.18 points (95% question-bootstrap interval -4.04 to +1.52). 11 of the 12 rows raise the stated
score above raw, 1 lowers it.

## Cost

`costs.jsonl`: two server sessions, 2 GPUs x server start to stop: 5.739 + 11.368 = 17.107 GPU-hours. Per-session
start, ready, request and stop times are in the investigation README
([occupied GPU-hours](../README.md#occupied-gpu-hours-of-the-three-k325-records)).
