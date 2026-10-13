| Record | Pass (seed) | Timing | KNOWN E/A | POSSIBLE E/A | NOT-aligned-split E/A | NOT-unsplittable E/A | UNKNOWN E/A | Empty total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| stock 0.7.0 (DFlash2 x5, DCP2) | 1 (1234) | not logged | - | - | - | 0/82 | 0/116 | 0 |
| stock 0.7.0 (DFlash2 x5, DCP2) | 2 (1235) | logged (proxy) | 0/1 | - | - | 3/194 | - | 3 |
| stock 0.7.0 (DFlash2 x5, DCP2) | 3 (1236) | logged (proxy) | 0/1 | - | - | 1/196 | - | 1 |
| stock 0.9.1 (DFlash2 x3) | 1 (1234) | logged (proxy) | - | - | 1/0 | 4/193 | - | 5 |
| stock 0.9.1 (DFlash2 x3) | 2 (1235) | logged (proxy) | - | - | 0/1 | 5/192 | - | 5 |
| stock 0.9.1 (DFlash2 x3) | 3 (1236) | logged (proxy) | - | - | 1/0 | 5/192 | - | 6 |
| col511 arm (0.9.1 + column-511 correction) | 1 (1234) | logged (proxy) | - | - | 1/0 | 3/194 | - | 4 |
| col511 arm (0.9.1 + column-511 correction) | 2 (1235) | logged (proxy) | - | - | 1/0 | 4/193 | - | 5 |
| col511 arm (0.9.1 + column-511 correction) | 3 (1236) | logged (proxy) | - | - | 1/0 | 5/192 | - | 6 |
| bundle D1-D4 = B0 (0.9.1 + D1-D4) | 1 (1234) | logged (proxy) | - | - | 1/0 | 2/195 | - | 3 |
| bundle D1-D4 = B0 (0.9.1 + D1-D4) | 2 (1235) | logged (proxy) | - | - | 0/1 | 3/194 | - | 3 |

Per record (all passes pooled; question-runs):

| Record | KNOWN E/A | POSSIBLE E/A | NOT-aligned-split E/A | NOT-unsplittable E/A | UNKNOWN E/A | Empty / runs |
|---|---:|---:|---:|---:|---:|---:|
| stock 0.7.0 (DFlash2 x5, DCP2) | 0/2 | - | - | 4/472 | 0/116 | 4/594 |
| stock 0.9.1 (DFlash2 x3) | - | - | 2/1 | 14/577 | - | 16/594 |
| col511 arm (0.9.1 + column-511 correction) | - | - | 3/0 | 12/579 | - | 15/594 |
| bundle D1-D4 = B0 (0.9.1 + D1-D4) | - | - | 1/1 | 5/389 | - | 6/396 |

Doc 127 (2,795 prompt tokens) per pass:

| Record | Pass | Class | Ensemble exposed | Decode rows at admission (analytic range) | First chunk | Empty | Finish | Completion tokens |
|---|---|---|---:|---|---:|---|---|---:|
| stock 0.7.0 (DFlash2 x5, DCP2) | 1 | UNKNOWN | - | unknown | unknown (exposed unless d in {1,5}) | False | None | None |
| stock 0.7.0 (DFlash2 x5, DCP2) | 2 | KNOWN | 1.00 | [7, 7] | 1966 | False | stop | 230448 |
| stock 0.7.0 (DFlash2 x5, DCP2) | 3 | KNOWN | 1.00 | [7, 7] | 1966 | False | stop | 220231 |
| stock 0.9.1 (DFlash2 x3) | 1 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | True | length | 327680 |
| stock 0.9.1 (DFlash2 x3) | 2 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | False | stop | 106687 |
| stock 0.9.1 (DFlash2 x3) | 3 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | True | length | 327680 |
| col511 arm (0.9.1 + column-511 correction) | 1 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | True | length | 327680 |
| col511 arm (0.9.1 + column-511 correction) | 2 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | True | length | 327680 |
| col511 arm (0.9.1 + column-511 correction) | 3 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | True | length | 327680 |
| bundle D1-D4 = B0 (0.9.1 + D1-D4) | 1 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | True | length | 327680 |
| bundle D1-D4 = B0 (0.9.1 + D1-D4) | 2 | NOT-aligned-split | 0.00 | [7, 7] | 1996 | False | stop | 57358 |
