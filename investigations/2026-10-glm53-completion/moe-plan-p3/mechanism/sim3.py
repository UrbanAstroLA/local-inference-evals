# Each plan's own error vs an un-staged (fp32 intermediates) reference.
import sys
sys.argv = ['x', '0']
exec(open('/w/sim.py').read().split('res = {}')[0])
ref = run(2, 'rne', (False, False, False))
for mode in ('rne', 'rz'):
    A = run(2, mode, (True, True, True)); B = run(4, mode, (True, True, True))
    print(mode, 'K64/N256 vs ref', json.dumps(cmp(A, ref)), '| K128/N128 vs ref', json.dumps(cmp(B, ref)))
