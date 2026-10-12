# B0 TP2 all-reduce fixture: CLEAN

## Phases (rank 0)
| phase | forwards | seconds | B12x push calls | epoch delta |
|---|---|---|---|---|
| main | 20000 | 64.46 | 1765400 | 1765400 |
| skew | 1500 | 5.05 | 136500 | 136500 |
| boundary | 3650 | 2.74 | 111600 | 111600 |
| mixed | 600 | 1.36 | 27300 | 27300 |
| nccl-ref | 900 | 2.35 | 0 | 0 |

## Mismatched elements by case (rank0, rank1)
- boundary/FULL/rows14/b12x: 0, 0
- boundary/FULL/rows15/b12x: 0, 0
- boundary/FULL/rows17/b12x: 0, 0
- boundary/FULL/rows22/b12x: 0, 0
- boundary/FULL/rows23/b12x: 0, 0
- boundary/FULL/rows25/b12x: 0, 0
- boundary/FULL/rows3/b12x: 0, 0
- boundary/FULL/rows47/b12x: 0, 0
- boundary/FULL/rows49/pynccl: 0, 0
- boundary/FULL/rows5/b12x: 0, 0
- boundary/FULL/rows6/b12x: 0, 0
- boundary/FULL/rows7/b12x: 0, 0
- boundary/FULL/rows9/b12x: 0, 0
- main/FULL/rows16/b12x: 0, 0
- main/FULL/rows24/b12x: 0, 0
- main/FULL/rows32/b12x: 0, 0
- main/FULL/rows4/b12x: 0, 0
- main/FULL/rows40/b12x: 0, 0
- main/FULL/rows48/b12x: 0, 0
- main/FULL/rows56/pynccl: 0, 0
- main/FULL/rows64/pynccl: 0, 0
- main/FULL/rows8/b12x: 0, 0
- main/PIECEWISE/rows1/b12x: 0, 0
- main/PIECEWISE/rows16/b12x: 0, 0
- main/PIECEWISE/rows2/b12x: 0, 0
- main/PIECEWISE/rows24/b12x: 0, 0
- main/PIECEWISE/rows32/b12x: 0, 0
- main/PIECEWISE/rows4/b12x: 0, 0
- main/PIECEWISE/rows40/b12x: 0, 0
- main/PIECEWISE/rows48/b12x: 0, 0
- main/PIECEWISE/rows56/pynccl: 0, 0
- main/PIECEWISE/rows64/pynccl: 0, 0
- main/PIECEWISE/rows8/b12x: 0, 0
- nccl-ref/FULL/rows1/pynccl-direct: 0, 0
- nccl-ref/FULL/rows48/pynccl-direct: 0, 0
- nccl-ref/FULL/rows8/pynccl-direct: 0, 0
- skew/FULL/rows1/b12x: 0, 0
- skew/FULL/rows16/b12x: 0, 0
- skew/FULL/rows24/b12x: 0, 0
- skew/FULL/rows4/b12x: 0, 0
- skew/FULL/rows48/b12x: 0, 0

