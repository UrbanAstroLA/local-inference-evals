# B0 TP2 all-reduce fixture: CLEAN

## Phases (rank 0)
| phase | forwards | seconds | B12x push calls | epoch delta |
|---|---|---|---|---|
| main | 110 | 0.4 | 9464 | 9464 |
| skew | 12 | 0.02 | 1092 | 1092 |
| boundary | 16 | 0.01 | 372 | 372 |
| mixed | 16 | 0.04 | 728 | 728 |
| nccl-ref | 12 | 0.06 | 0 | 0 |

## Mismatched elements by case (rank0, rank1)
- boundary/FULL/rows3/b12x: 0, 0
- boundary/FULL/rows49/pynccl: 0, 0
- main/FULL/rows4/b12x: 0, 0
- main/FULL/rows48/b12x: 0, 0
- main/FULL/rows64/pynccl: 0, 0
- main/PIECEWISE/rows1/b12x: 0, 0
- main/PIECEWISE/rows48/b12x: 0, 0
- nccl-ref/FULL/rows48/pynccl-direct: 0, 0
- skew/FULL/rows4/b12x: 0, 0

