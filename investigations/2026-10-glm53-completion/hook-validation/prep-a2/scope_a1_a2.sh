#!/bin/bash
# PA2-17 (amendment 21g item 3): image scope for the H2/H3 carry-over. CPU only: two CPU containers per image,
# --network none, no --gpus, nothing written inside them. No `set -e` (each step's rc checked).
# Hashes EVERY file under the installed vllm package (.py, .so, .pyc, data) in the attempt-1 image and the attempt-2
# image, diffs the two listings, and for every differing .pyc compares the marshalled body (bytes 16..) so that a
# header-only difference (source mtime) is told apart from a code difference.
# Writes prep-a2/out-21g/image-scope.json (+ image-scope-diff.txt). Read by prep-a2/make_carryover.py.
HV=$(cd "$(dirname "$0")/.." && pwd)
A1=sha256:1b9f8b15d63197ad8b2ef2410321baedf9dcd55a112603bc77b887e8dac293d5   # local/glm53-v0.9.1-b1-forcehook-mg:local
A2=$(awk '{print $2}' "$HV/build-a2/image-id.txt")                         # local/glm53-v0.9.1-b1-forcehook-mg2:local
OUT=$HV/prep-a2/out-21g      # prep-a2/out is root-owned (an earlier container wrote it); 21g outputs go here
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT
VP=/usr/local/lib/python3.12/dist-packages
for X in a1 a2; do
  I=$A1; [ $X = a2 ] && I=$A2
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint sh "$I" -c \
    "cd $VP && find vllm -type f -print0 | sort -z | xargs -0 sha256sum" > "$TMP/$X.txt"
  RC=$?
  [ $RC -eq 0 ] || { echo "listing $X failed rc=$RC"; exit 1; }
done
diff "$TMP/a1.txt" "$TMP/a2.txt" | grep '^[<>]' | awk '{print $1, $3}' | sort -k2 > "$OUT/image-scope-diff.txt"
FILES=$(awk '{print $2}' "$OUT/image-scope-diff.txt" | sort -u)
PYC=$(echo "$FILES" | grep '\.pyc$')
for X in a1 a2; do
  I=$A1; [ $X = a2 ] && I=$A2
  : > "$TMP/body-$X.txt"
  for f in $PYC; do
    docker run --rm --network none --entrypoint sh "$I" -c \
      "cd $VP && printf '%s %s\n' \$(tail -c +17 $f | sha256sum | cut -d' ' -f1) $f" >> "$TMP/body-$X.txt"
  done
done
python3 - "$TMP" "$OUT" "$A1" "$A2" <<'EOF'
import hashlib, json, sys
tmp, out, a1, a2 = sys.argv[1:5]
L = {x: dict(reversed(l.split(None, 1)) for l in open(f"{tmp}/{x}.txt").read().splitlines()) for x in ("a1", "a2")}
L = {x: {k.strip(): v for k, v in d.items()} for x, d in L.items()}
B = {x: dict(reversed(l.split(None, 1)) for l in open(f"{tmp}/body-{x}.txt").read().splitlines() if l.strip())
     for x in ("a1", "a2")}
diff = sorted(f for f in set(L["a1"]) | set(L["a2"]) if L["a1"].get(f) != L["a2"].get(f))
src_diff = [f for f in diff if not f.endswith(".pyc")]
pyc = {f: dict(body_equal=B["a1"].get(f) == B["a2"].get(f) and B["a1"].get(f) is not None,
               source=f.replace("__pycache__/", "").rsplit(".cpython", 1)[0] + ".py") for f in diff if f.endswith(".pyc")}
pyc_code_diff = [f for f, v in pyc.items() if not v["body_equal"] and v["source"] not in src_diff]
ok = src_diff == ["vllm/forcewin_hooks.py"] and not pyc_code_diff
res = dict(
    what="every file under the installed vllm package, attempt-1 image vs attempt-2 image (CPU containers, "
         "--network none); differing .pyc compared without their 16-byte header (source mtime)",
    image_a1=a1, image_a2=a2, files_a1=len(L["a1"]), files_a2=len(L["a2"]),
    listing_sha256=dict(a1=hashlib.sha256(open(f"{tmp}/a1.txt", "rb").read()).hexdigest(),
                        a2=hashlib.sha256(open(f"{tmp}/a2.txt", "rb").read()).hexdigest()),
    differing_files=diff, differing_sources=src_diff, pyc=pyc,
    pyc_with_code_difference_for_unchanged_source=pyc_code_diff,
    only_hook_module_changed=ok)
json.dump(res, open(f"{out}/image-scope.json", "w"), indent=1)
print(json.dumps(dict(only_hook_module_changed=ok, differing_sources=src_diff, files=len(L["a1"]),
                      pyc_header_only=[f for f, v in pyc.items() if v["body_equal"]]), indent=1))
sys.exit(0 if ok else 3)
EOF
