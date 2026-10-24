#!/usr/bin/env python3
"""Mock root for the capture-p0 runner tests: a SYNTHETIC fixture (fabricated token ids, not benchmark data) under
ROOT/fixtures-synthetic/ (never a PRIVATE path).  Host python3.   usage: make_mock_root_cap0.py ROOT"""
import json
import os
import sys

root = sys.argv[1]
os.makedirs(os.path.join(root, "fixtures-synthetic"), exist_ok=True)
ids = [1000 + (i * 7) % 5000 for i in range(20000)]
fx = dict(note="SYNTHETIC fixture for runner tests (fabricated ids)",
          replies={"R01": dict(ids=ids, windows={"early": dict(P=17409), "deep": dict(P=18000), "close": dict(P=19000)})})
json.dump(fx, open(os.path.join(root, "fixtures-synthetic", "windows.json"), "w"))
print("mock root", root)
