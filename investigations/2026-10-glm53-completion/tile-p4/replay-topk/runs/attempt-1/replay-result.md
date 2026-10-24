# replay-topk result

SYNTHETIC inputs throughout (no model data). Readings per PROTOCOL.md section 7.

- v091 R1 [K: A-ORDER, C: A-ORDER, T: A-SET]; R2 [flat: ORDER-SENSITIVE, peaked: ORDER-SENSITIVE]; R3 COMPOSED
- v070 R1 [C: A-ORDER, T: A-SET]; R2 [flat: ORDER-SENSITIVE, peaked: ORDER-SENSITIVE]

## v091

| Unit | Reading | Key facts | Failed gates / caveats |
|---|---|---|---|
| R1 K | A-ORDER | order varied: 1.0 same-alloc / 1.0 relocated of selecting rows; set varied rows 0/0; last col varied 0 |  |
| R1 C | A-ORDER | order varied: 1.0 same-alloc / 1.0 relocated of selecting rows; set varied rows 0/0; last col varied 0 |  |
| R1 T | A-SET | order varied: 1.0 same-alloc / 1.0 relocated of selecting rows; set varied rows 171/199; last col varied 347 |  |
| R2 flat | ORDER-SENSITIVE | asc: 2039/2039 rows, max_abs 0.00390625; desc: 2039/2039 rows, max_abs 0.00390625; perm1: 2039/2039 rows, max_abs 0.00390625; perm2: 2039/2039 rows, max_abs 0.00390625; perm3: 2039/2039 rows, max_abs 0.00390625; atomic2: 2039/2039 rows, max_abs 0.00390625 |  |
| R2 peaked | ORDER-SENSITIVE | asc: 2039/2039 rows, max_abs 0.015625; desc: 2039/2039 rows, max_abs 0.015625; perm1: 2039/2039 rows, max_abs 0.015625; perm2: 2039/2039 rows, max_abs 0.015625; perm3: 2039/2039 rows, max_abs 0.015625; atomic2: 2039/2039 rows, max_abs 0.015625 |  |
| R3 | COMPOSED | |  |

## v070

| Unit | Reading | Key facts | Failed gates / caveats |
|---|---|---|---|
| R1 C | A-ORDER | order varied: 1.0 same-alloc / 1.0 relocated of selecting rows; set varied rows 0/0; last col varied 0 | logits stride assumed (served producer unavailable) |
| R1 T | A-SET | order varied: 1.0 same-alloc / 1.0 relocated of selecting rows; set varied rows 162/188; last col varied 328 | logits stride assumed (served producer unavailable) |
| R2 flat | ORDER-SENSITIVE | asc: 2039/2039 rows, max_abs 0.0078125; desc: 2039/2039 rows, max_abs 0.0078125; perm1: 2039/2039 rows, max_abs 0.0078125; perm2: 2039/2039 rows, max_abs 0.0078125; perm3: 2039/2039 rows, max_abs 0.0078125; atomic2: 2039/2039 rows, max_abs 0.0078125 |  |
| R2 peaked | ORDER-SENSITIVE | asc: 2039/2039 rows, max_abs 0.03125; desc: 1973/2039 rows, max_abs 0.015625; perm1: 2039/2039 rows, max_abs 0.03125; perm2: 2039/2039 rows, max_abs 0.015625; perm3: 2039/2039 rows, max_abs 0.015625; atomic2: 2039/2039 rows, max_abs 0.015625 |  |
