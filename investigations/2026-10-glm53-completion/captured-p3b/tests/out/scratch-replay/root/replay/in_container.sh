#!/bin/sh
# Runs INSIDE one stock image (launched by replay/run_replay.sh): the replay parts in priority order, each with a soft
# deadline taken from what is left of SOFT_S.  Mounts: /r replay (ro), /p3h P3 harness (ro), /p3out P3 attempt-2
# results (ro), /model-repo (ro), /cap captured tensors (ro), /caplog capture logs (ro), /o public outputs,
# /po private outputs (0700), /cc compile cache.
#   in_container.sh ENGINE SOFT_S PARTS [--dry]     PARTS: comma list of attn, moe, forced
ENGINE=$1; SOFT=$2; PARTS=$3; DRY=${4:-}
T0=$(date +%s)
left() { echo $(( SOFT - ($(date +%s) - T0) )); }
umask 077
case ",$PARTS," in *,attn,*)
  A=$(left); [ "$A" -gt 90 ] && A=90
  echo "== attn soft ${A}s"
  python3 /r/replay_attn.py --cap /cap --caplog /caplog --out /o/attn-$ENGINE.json --priv /po --deadline-s "$A" $DRY
  echo "attn rc=$?";;
esac
case ",$PARTS," in *,moe,*)
  M=$(left)
  FT=""; case ",$PARTS," in *,forced,*) FT="--forced-tile --p3-json /p3out/v091.json";; esac
  echo "== moe $ENGINE soft ${M}s $FT"
  [ "$M" -gt 20 ] && python3 /r/replay_moe.py --engine "$ENGINE" --cap /cap --caplog /caplog --out /o/moe-$ENGINE.json \
      --deadline-s "$M" $FT $DRY
  echo "moe rc=$?";;
esac
echo "in-container elapsed $(( $(date +%s) - T0 ))s"
