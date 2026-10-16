#!/bin/bash
# [R-11] Acceptance stress scenarios (PLAN.md amendment 22f s.3) for the proposed retry session: CPU containers of the
# serving image (run_cpu.sh: --network none, no GPU), 4 at a time. Outputs out/sim_stress_<name>.json; summary by
# stress_report.py -> out/F3_stress.json, out/F3_stress.out. Stress scenarios, NOT predictions of D2.
#   single     single-token advance throughout (poison-like: no draft ever accepted)
#   max        maximum accepted advance throughout (oracle-like: every scheduled draft accepted)
#   unequal-a  strongly unequal between fixtures: the 4 longest (L1 A8 L8 L5, build half a) at max, the 4 shortest at
#              single;  unequal-b the reverse
# each at the served 1,275 blocks (exact run-fc1 start), at the 1,150-block refusal floor (model start), and the two
# unequal assignments also with reverse (adversarial) admission order. measured_recheck = the base scenario re-run
# with the R-11 code (must equal out/sim_base.json).
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1
LONG='["L1","A8","L8","L5"]'
SHORT='["M3","L4","M4","L3"]'
SC=(
 '{"name":"stress_measured_recheck","variant":"B","start":"trace","seed":1,"occ_every":8}'
 '{"name":"stress_single","variant":"B","start":"trace","seed":1,"acceptance":"single"}'
 '{"name":"stress_max","variant":"B","start":"trace","seed":1,"acceptance":"max"}'
 '{"name":"stress_unequal_a","variant":"B","start":"trace","seed":1,"acceptance":"unequal","fast":'"$LONG"'}'
 '{"name":"stress_unequal_b","variant":"B","start":"trace","seed":1,"acceptance":"unequal","fast":'"$SHORT"'}'
 '{"name":"stress_unequal_a_reverse","variant":"B","start":"trace","seed":1,"acceptance":"unequal","fast":'"$LONG"',"order_policy":"reverse"}'
 '{"name":"stress_unequal_b_reverse","variant":"B","start":"trace","seed":1,"acceptance":"unequal","fast":'"$SHORT"',"order_policy":"reverse"}'
 '{"name":"stress_single_blocks1150","variant":"B","start":"model","seed":1,"acceptance":"single","num_blocks":1150}'
 '{"name":"stress_max_blocks1150","variant":"B","start":"model","seed":1,"acceptance":"max","num_blocks":1150}'
 '{"name":"stress_unequal_a_blocks1150","variant":"B","start":"model","seed":1,"acceptance":"unequal","fast":'"$LONG"',"num_blocks":1150}'
 '{"name":"stress_unequal_b_blocks1150","variant":"B","start":"model","seed":1,"acceptance":"unequal","fast":'"$SHORT"',"num_blocks":1150}'
)
N=${PAR:-4}
i=0
for s in "${SC[@]}"; do
  ./run_cpu.sh session_sim.py "$s" 2>&1 | grep -v "cpp_extension\|vllm._C" | tail -1 > "out/stress_console_$i.txt" &
  i=$((i+1))
  if (( i % N == 0 )); then wait; fi
done
wait
cat out/stress_console_*.txt > out/stress_console.txt
rm -f out/stress_console_[0-9]*.txt
