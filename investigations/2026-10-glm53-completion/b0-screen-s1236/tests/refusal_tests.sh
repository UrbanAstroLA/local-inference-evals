#!/usr/bin/env bash
# Refusal-path tests of gpqa-b0-screen-s1236.sh without GPUs, containers or writes outside WORK (CPU only).
#   refusal_tests.sh WORK_DIR
# Copy of ../../b1-screen/tests/refusal_tests.sh adapted to B0. docker, nvidia-smi and ss are replaced by shims on PATH.
# The shims answer only read-only queries (image inspect, ps, GPU/process listings, socket listing) and refuse and log ANY
# other call, so no container can be started or stopped.
# Part 1 runs the FROZEN runner (DRY_RUN=1) under environment scenarios (pass state = the real <gpqa-runs>, read only).
# Part 2 runs a copy whose B0S and G paths point into WORK, for file-state refusals. WORK's gpqa directory holds SYMLINKS
# to B0's real __pass1/__pass2 (read only; never written) so the frozen samples hashes verify.
# Every scenario must end with the expected message and the expected exit code. Not set -e.
WORK=${1:?usage: refusal_tests.sh WORK_DIR}
HERE=$(cd "$(dirname "$0")" && pwd); B0S=$(dirname "$HERE"); RUNNER=$B0S/gpqa-b0-screen-s1236.sh
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234"; GR=<gpqa-runs>
IMG=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548
mkdir -p "$WORK/shim" || exit 2
SHIMLOG=$WORK/shim-calls.log; : > "$SHIMLOG"
cat > "$WORK/shim/docker" <<EOF
#!/usr/bin/env bash
echo "docker \$*" >> "$SHIMLOG"
if [ "\$1 \$2" = "image inspect" ]; then
  case "\$3" in
    $IMG) [ "\$SCEN" = image_missing ] && exit 1; echo $IMG ;;
    local/glm53-v0.9.1-bundle-d1234:local) [ "\$SCEN" = tag_moved ] && { echo sha256:0000000000000000000000000000000000000000000000000000000000000000; exit 0; }; echo $IMG ;;
    *) exit 1 ;;
  esac; exit 0
fi
if [ "\$1" = ps ]; then echo open-webui; echo portainer; [ "\$SCEN" = glm_running ] && echo glm53-flash-exl3-b12x-vllm; exit 0; fi
echo "SHIM REFUSED: docker \$*" >&2; echo "REFUSED docker \$*" >> "$SHIMLOG"; exit 99
EOF
cat > "$WORK/shim/nvidia-smi" <<EOF
#!/usr/bin/env bash
echo "nvidia-smi \$*" >> "$SHIMLOG"
[ "\$SCEN" = smi_fail ] && exit 9
case "\$*" in
  -L) echo "GPU 0: shim"; echo "GPU 1: shim" ;;
  *query-compute-apps*) [ "\$SCEN" = compute ] && echo "\$TEST_PID" ;;
  *query-gpu*) echo "0, 20, 0"; [ "\$SCEN" = onegpu ] || { [ "\$SCEN" = busy ] && echo "1, 9000, 50" || echo "1, 2246, 2"; } ;;
  "") cat <<'T'
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|=========================================================================================|
|    0   N/A  N/A           89543      G   /usr/bin/gnome-shell                      8MiB |
|    1   N/A  N/A           91069      G   .../8969/usr/lib/firefox/firefox        578MiB |
|    1   N/A  N/A           97151      G   ...nap-store/1419/bin/snap-store        368MiB |
|    1   N/A  N/A           99539    C+G   ...88185955192 --disable-logging        346MiB |
T
      [ "\$SCEN" = graphics ] && echo "|    1   N/A  N/A           12345      G   /usr/bin/some-other-app                 900MiB |" ;;
  *) echo "SHIM REFUSED: nvidia-smi \$*" >&2; echo "REFUSED nvidia-smi \$*" >> "$SHIMLOG"; exit 99 ;;
esac
exit 0
EOF
cat > "$WORK/shim/ss" <<EOF
#!/usr/bin/env bash
echo "ss \$*" >> "$SHIMLOG"
[ "\$SCEN" = port ] && echo "LISTEN 0 5 127.0.0.1:8002 0.0.0.0:*"
exit 0
EOF
chmod +x "$WORK/shim/"*
sleep 300 & TEST_PID=$!            # a process without --type=gpu-process, posed as a compute client
trap 'kill $TEST_PID 2>/dev/null' EXIT
PASS=0; FAIL=0
t() {  # t NAME EXPECTED_RC EXPECTED_TEXT [VAR=VALUE ...] -- RUNNER_PATH
  local name=$1 erc=$2 etext=$3; shift 3; local envs=(); while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  local out rc
  out=$(env "${envs[@]}" DRY_RUN=1 TEST_PID=$TEST_PID PATH="$WORK/shim:$PATH" bash "$1" 2>&1); rc=$?
  if { [ "$erc" = nonzero ] && [ $rc -ne 0 ] || [ "$erc" = "$rc" ]; } && echo "$out" | grep -qF -- "$etext"; then
    PASS=$((PASS+1)); echo "PASS  $name (rc $rc): $(echo "$out" | grep -F -- "$etext" | head -1 | cut -c1-140)"
  else
    FAIL=$((FAIL+1)); echo "FAIL  $name (rc $rc, expected $erc / '$etext'):"; echo "$out" | tail -5
  fi
}
echo "== part 1: the frozen runner ($(sha256sum "$RUNNER" | cut -c1-16)...) under environment scenarios"
t baseline            0       "DRY_RUN: all refusal checks passed"  SCEN=none -- "$RUNNER"
t ceiling_above_7.5   nonzero "must be in (0, 7.5]"                 SCEN=none CEIL_GH=8 -- "$RUNNER"
t image_missing       nonzero "ABORT: image"                        SCEN=image_missing -- "$RUNNER"
t tag_moved           nonzero "tag moved"                           SCEN=tag_moved -- "$RUNNER"
t override_extra_args nonzero "override variables set"              SCEN=none EXTRA_SERVE_ARGS=--enforce-eager -- "$RUNNER"
t override_gpqa_tag   nonzero "override variables set"              SCEN=none GPQA_TAG=+x -- "$RUNNER"
t override_dflash     nonzero "override variables set"              SCEN=none DFLASH_TOKENS=5 -- "$RUNNER"
t override_port       nonzero "override variables set"              SCEN=none PORT=0.0.0.0:8001 -- "$RUNNER"
t override_no_restart nonzero "override variables set"              SCEN=none NO_RESTART=1 -- "$RUNNER"
t glm_container       nonzero "a GLM container is running"          SCEN=glm_running -- "$RUNNER"
t port_in_use         nonzero "port 8001 or 8002 is in use"         SCEN=port -- "$RUNNER"
t nvidia_smi_fails    nonzero "nvidia-smi failed"                   SCEN=smi_fail -- "$RUNNER"
t compute_process     nonzero "compute process"                     SCEN=compute -- "$RUNNER"
t graphics_client     nonzero "GPU clients other than the desktop"  SCEN=graphics -- "$RUNNER"
t gpu_busy            nonzero "GPU1 busy"                           SCEN=busy -- "$RUNNER"
t one_gpu             nonzero "expected 2 GPUs"                     SCEN=onegpu -- "$RUNNER"
echo "== part 2: path-substituted copy (B0S, G -> $WORK) for file-state refusals"
CP=$WORK/b0s; GW=$WORK/gpqa; rm -rf "$CP" "$GW"; mkdir -p "$CP" "$GW"
cp "$B0S/b0s1236_analysis.py" "$B0S/analysis_col511.py" "$B0S/FROZEN-INTERPRETATION.md" "$CP/"
sed -e "s#^B0S=.*#B0S=$CP#" -e "s#R=\"$R\"; G=<gpqa-runs>;#R=\"$R\"; G=$GW;#" "$RUNNER" > "$CP/gpqa-b0-screen-s1236.sh"
grep -q "^B0S=$CP" "$CP/gpqa-b0-screen-s1236.sh" && grep -q "G=$GW;" "$CP/gpqa-b0-screen-s1236.sh" || { echo "FAIL  could not build the copy"; exit 1; }
( cd "$B0S" && grep -v ' gpqa-b0-screen-s1236.sh$' SHA256SUMS ) > "$CP/SHA256SUMS"
( cd "$CP" && sha256sum gpqa-b0-screen-s1236.sh >> SHA256SUMS )
ln -s "$GR/${R}__pass1" "$GW/${R}__pass1"; ln -s "$GR/${R}__pass2" "$GW/${R}__pass2"
C2=$CP/gpqa-b0-screen-s1236.sh
t copy_baseline       0       "DRY_RUN: all refusal checks passed"  SCEN=none -- "$C2"
mkdir "$CP/run-s1236";    t earlier_attempt nonzero "exists (an earlier attempt"         SCEN=none -- "$C2"; rmdir "$CP/run-s1236"
echo '{"session":3}' > "$CP/costs.jsonl"; t cost_line_present nonzero "already records a B0 seed-1236 session" SCEN=none -- "$C2"; rm "$CP/costs.jsonl"
mkdir "$GW/${R}__pass3";  t pass3_dir_exists nonzero "a B0 pass-3 (or later) directory exists" SCEN=none -- "$C2"; rmdir "$GW/${R}__pass3"
mkdir "$GW/${R}__pass4";  t pass4_dir_exists nonzero "a B0 pass-3 (or later) directory exists" SCEN=none -- "$C2"; rmdir "$GW/${R}__pass4"
mkdir "$GW/${R}__pass12"; t pass12_dir_exists nonzero "a B0 pass-3 (or later) directory exists" SCEN=none -- "$C2"; rmdir "$GW/${R}__pass12"
: > "$GW/${R}.s3.requests.jsonl"; t s3_reqlog_exists nonzero "s3.requests.jsonl exists" SCEN=none -- "$C2"; rm "$GW/${R}.s3.requests.jsonl"
echo '{"seed": 1236}' > "$GW/${R}.s9.requests.jsonl"; t seed1236_in_b0_log nonzero "already holds seed-1236 requests" SCEN=none -- "$C2"; rm "$GW/${R}.s9.requests.jsonl"
echo '{"seed": 1234}' > "$GW/${R}.s1.requests.jsonl"; echo '{"seed": 1235}' > "$GW/${R}.s2.requests.jsonl"
t b0_logs_other_seeds_ok 0 "DRY_RUN: all refusal checks passed"     SCEN=none -- "$C2"
rm "$GW/${R}.s1.requests.jsonl" "$GW/${R}.s2.requests.jsonl"
mkdir "$GW/${R}+d7__pass3"; echo '{"seed": 1236}' > "$GW/${R}+d7.s1.requests.jsonl"
t b1_files_not_b0     0       "DRY_RUN: all refusal checks passed"  SCEN=none -- "$C2"
rmdir "$GW/${R}+d7__pass3"; rm "$GW/${R}+d7.s1.requests.jsonl"
rm "$GW/${R}__pass1";     t pass1_missing nonzero "__pass1 missing or not the frozen B0 record" SCEN=none -- "$C2"; ln -s "$GR/${R}__pass1" "$GW/${R}__pass1"
rm "$GW/${R}__pass2"; ln -s "$GR/${R}__pass1" "$GW/${R}__pass2"
t pass2_not_frozen    nonzero "__pass2 missing or not the frozen B0 record" SCEN=none -- "$C2"; rm "$GW/${R}__pass2"; ln -s "$GR/${R}__pass2" "$GW/${R}__pass2"
cp "$CP/FROZEN-INTERPRETATION.md" "$WORK/fi.bak"; echo " " >> "$CP/FROZEN-INTERPRETATION.md"
t frozen_file_changed nonzero "SHA256SUMS does not verify"          SCEN=none -- "$C2"; cp "$WORK/fi.bak" "$CP/FROZEN-INTERPRETATION.md"
t copy_baseline_again 0       "DRY_RUN: all refusal checks passed"  SCEN=none -- "$C2"
echo "== shim calls other than read-only queries:"
if grep -q '^REFUSED' "$SHIMLOG"; then grep '^REFUSED' "$SHIMLOG"; FAIL=$((FAIL+1)); else echo "none"; fi
echo "== distinct shim calls:"; sed -E 's/[0-9a-f]{64}/<id>/' "$SHIMLOG" | sort | uniq -c
LEFT=$(ls "$GW" | grep -v -e "^${R}__pass1\$" -e "^${R}__pass2\$")
[ -z "$LEFT" ] && echo "WORK gpqa dir holds only the two read-only symlinks after tests" || { echo "FAIL  leftovers in $GW: $LEFT"; FAIL=$((FAIL+1)); }
rm "$GW/${R}__pass1" "$GW/${R}__pass2"
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
