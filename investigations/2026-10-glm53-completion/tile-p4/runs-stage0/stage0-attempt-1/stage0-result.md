# Stage-0 result (analysis/analyze_stage0.py; aggregates and first-difference descriptors only)

**Interpretation:** Neither reproducible: investigate the first differing input, state or operation (V1 G-entry, V2 G-entry); do not label stock kernels nondeterministic

Outputs matching for 64 positions are an operational gate; they do not certify every persistent-state byte. Entry digests cover the KV-cache tensors listed as covered in entry_init; logical block 0 (shared null block) is reported, not required.

| variant | verdict | first failed gate | G-enq | G-entry | G-out |
|---|---|---|---|---|---|
| V1 | NOT REPRODUCIBLE | G-entry | PASS | FAIL | - |
| V2 | NOT REPRODUCIBLE | G-entry | PASS | FAIL | - |

## V1: first failed gate G-entry

```
{
 "why": null,
 "first_difference": {
  "window": "R01-early",
  "rank": 0,
  "point": 0,
  "point_kind": "prefill",
  "tensor": "language_model.model.layers.7.self_attn.indexer.k_cache",
  "group": 0,
  "logical_block": 0,
  "point0_cached_state_equal": false
 }
}
```

## V2: first failed gate G-entry

```
{
 "why": null,
 "first_difference": {
  "window": "R01-early",
  "rank": 0,
  "point": 0,
  "point_kind": "prefill",
  "tensor": "language_model.model.layers.7.self_attn.indexer.k_cache",
  "group": 0,
  "logical_block": 0,
  "point0_cached_state_equal": false
 }
}
```

## Instrument health

```
{
 "hook_errors": 0,
 "entry_coverage": {
  "0": {
   "covered": 91,
   "uncovered": 0,
   "n_groups": 10,
   "groups_covered": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
   ],
   "uncovered_layers": []
  },
  "1": {
   "covered": 88,
   "uncovered": 0,
   "n_groups": 10,
   "groups_covered": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
   ],
   "uncovered_layers": []
  }
 },
 "sched_steps": 3556,
 "forced_adds": 32
}
```
