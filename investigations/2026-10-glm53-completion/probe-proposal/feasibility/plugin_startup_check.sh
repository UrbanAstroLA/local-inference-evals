#!/bin/sh
# E4b: a general plugin installed as a .dist-info on PYTHONPATH at process start (as a derived image would ship it).
D=/tmp/plg; mkdir -p $D/local_probe_selftest-0.0.dist-info
printf 'HIT = []\ndef register():\n    HIT.append(1)\n' > $D/local_probe_selftest.py
printf 'Metadata-Version: 2.1\nName: local-probe-selftest\nVersion: 0.0\n' > $D/local_probe_selftest-0.0.dist-info/METADATA
printf '[vllm.general_plugins]\nlocal_probe_selftest = local_probe_selftest:register\n' > $D/local_probe_selftest-0.0.dist-info/entry_points.txt
PYTHONPATH=$D python3 -c "
import vllm.plugins as P
P.load_general_plugins()
import local_probe_selftest as m
print('PASS E4b general plugin discovered at process start' if m.HIT==[1] else 'FAIL E4b')"
