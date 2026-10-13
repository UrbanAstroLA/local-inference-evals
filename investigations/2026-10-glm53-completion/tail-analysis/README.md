# Long-context tail analysis (amendment 19f; exploratory)

[`tail_analysis.py`](tail_analysis.py) reads the recording-proxy request logs of the 11 logged K3.25 GPQA passes (counts
only: completion tokens, finish reason, reply length). It reports, per pass and pooled by engine, how many replies reach
32k, 64k, 100k and 200k tokens and how many of those finish before the 327,680-token cap, with intervals bootstrapped
over questions. It also gives the per-band finishing hazard and a question-level comparison. The output is in
[`tail_analysis.out`](tail_analysis.out).

It was chosen after the B1 seed-1236 result and is not pre-registered. The request logs are not published, because they
hold question and reply text; their hashes are in the screen folders' RAW-SHA256SUMS files.
