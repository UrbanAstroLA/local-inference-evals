#!/bin/bash
# F7 scenario sweep for the proposed retry session (CPU containers, 4 at a time). Outputs out/sim_<name>.json.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
SC=(
 '{"name":"base","variant":"B","start":"trace","seed":1,"occ_every":8}'
 '{"name":"model_start","variant":"B","start":"model","seed":1}'
 '{"name":"seed2","variant":"B","start":"trace","seed":2}'
 '{"name":"seed3","variant":"B","start":"trace","seed":3}'
 '{"name":"seed4","variant":"B","start":"trace","seed":4}'
 '{"name":"seed5","variant":"B","start":"trace","seed":5}'
 '{"name":"stagger0.1","variant":"B","start":"trace","seed":1,"stagger_scale":0.1}'
 '{"name":"stagger2","variant":"B","start":"trace","seed":1,"stagger_scale":2}'
 '{"name":"stagger4","variant":"B","start":"trace","seed":1,"stagger_scale":4}'
 '{"name":"jitter1","variant":"B","start":"trace","seed":11,"arrival_jitter":1.0}'
 '{"name":"jitter3","variant":"B","start":"trace","seed":12,"arrival_jitter":3.0}'
 '{"name":"order_plan","variant":"B","start":"trace","seed":1,"order_policy":"plan"}'
 '{"name":"order_reverse","variant":"B","start":"trace","seed":1,"order_policy":"reverse"}'
 '{"name":"order_random1","variant":"B","start":"trace","seed":21,"order_policy":"random"}'
 '{"name":"order_random2","variant":"B","start":"trace","seed":22,"order_policy":"random"}'
 '{"name":"blocks1250","variant":"B","start":"model","seed":1,"num_blocks":1250}'
 '{"name":"blocks1225","variant":"B","start":"model","seed":1,"num_blocks":1225}'
 '{"name":"blocks1200","variant":"B","start":"model","seed":1,"num_blocks":1200}'
 '{"name":"blocks1150","variant":"B","start":"model","seed":1,"num_blocks":1150}'
 '{"name":"blocks1100","variant":"B","start":"model","seed":1,"num_blocks":1100}'
 '{"name":"build_sequential","variant":"B","start":"trace","seed":1,"build_sequential":true}'
 '{"name":"build_longest","variant":"B","start":"trace","seed":1,"build_order":"longest"}'
 '{"name":"gate600","variant":"B","start":"trace","seed":1,"gate_s":600}'
 '{"name":"no_touch","variant":"B","start":"trace","seed":1,"no_touch":true}'
 '{"name":"no_touch_jitter1","variant":"B","start":"trace","seed":11,"no_touch":true,"arrival_jitter":1.0}'
 '{"name":"no_touch_blocks1225","variant":"B","start":"model","seed":1,"no_touch":true,"num_blocks":1225}'
 '{"name":"A0","variant":"A0","start":"trace","seed":1}'
 '{"name":"A0_blocks1200","variant":"A0","start":"model","seed":1,"num_blocks":1200}'
)
N=${PAR:-4}
i=0
for s in "${SC[@]}"; do
  ./run_cpu.sh session_sim.py "$s" 2>&1 | grep -v "cpp_extension\|vllm._C" | tail -1 > "out/sim_console_$i.txt" &
  i=$((i+1))
  if (( i % N == 0 )); then wait; fi
done
wait
cat out/sim_console_*.txt > out/sim_console.txt
rm -f out/sim_console_[0-9]*.txt
