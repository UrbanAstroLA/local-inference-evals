#!/usr/bin/env bash
# Runs INSIDE the B0 container (GPU 0 only). Caps come from the environment.
# FIXTURE_PARTS (optional, e.g. "D1,D2,D3,E3"): run only those fixture parts and skip the
# pytest steps (post-run-2 partial reruns, PLAN amendment 16e).
set -u
nvidia-smi -L > /o/gpu.txt 2>&1
timeout "$FIXTURE_CAP_S" python3 /f/run_fixture.py /o/results.json "${FIXTURE_PARTS:-}" > /o/fixture.log 2>&1
echo "fixture_exit=$?" > /o/exit.txt
if [ -n "${FIXTURE_PARTS:-}" ]; then
  echo "pytest=skipped (partial rerun: $FIXTURE_PARTS)" >> /o/exit.txt
  exit 0
fi
cd /opt/b12x || exit 0
timeout "$PYTEST_CAP_S" python3 -m pytest -q -rs -p no:cacheprovider \
  tests/attention/test_attention_mla_sm120.py -k high > /o/pytest-sm120-high.txt 2>&1
echo "pytest_sm120_exit=$?" >> /o/exit.txt
timeout "$PYTEST_CAP_S" python3 -m pytest -q -rs -p no:cacheprovider \
  tests/attention/test_glm_next_mla.py::test_glm_next_writer_and_reader_use_int64_for_live_high_page \
  /f/test_repaired_high_page.py > /o/pytest-glm-next-high.txt 2>&1
echo "pytest_glm_next_exit=$?" >> /o/exit.txt
