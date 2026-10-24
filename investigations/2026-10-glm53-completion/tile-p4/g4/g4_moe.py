#!/usr/bin/env python3
"""G4 (PROPOSAL.md section 3): captured-p3b's MoE replay (replay/replay_moe.py, unchanged, mounted read only at /r)
run INSIDE the patched P4 image on one GPU.  The only adaptation: replay_moe refuses an image whose exl3.py is not
the stock file (its engine-identity pin); this wrapper re-pins that one hash to the reviewed P4 exl3.py and asserts
that everything else it checks is unchanged.  Run twice by in_container_g4.sh:
  G4a  GLM53_P4 and GLM53_P4_TILE unset: every captured entry of layers 3, 23, 44 must reproduce the served stock
       output bit for bit through the bound plan AND through Exl3MoEMethod.apply (the gate);
  G4b  GLM53_P4_TILE=64,256,64,256 (reported, not a gate): the env switch reaches the decode launch (K64/N256 at
       m 16-32) and the decode plan then equals the prefill plan bit for bit on the captured m 16-32 rows.
usage: g4_moe.py <replay_moe.py arguments>
"""
import hashlib
import os
import sys

STOCK_EXL3 = "3b4ca47cb1168bffebcfe3cf1db48daf86691c140cbf40f9c1f3aa6ad6a444d9"
P4_EXL3 = "bdb4f14c63ccaddb084e5b7c23df591318528084b76c6535b39daf32174a388c"
REPLAY_MOE = "619cbf72b711b479be55b27b34fe81f1f7189c4b5bc2d89a4740972358de4b51"
REPLAY_COMMON = "a72cb130fa711a36f0933ea4bc4af5e6b2089a721744b7a28ed374643bbbfaa3"
sys.path.insert(0, "/r")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if sha("/r/replay_moe.py") != REPLAY_MOE or sha("/r/replay_common.py") != REPLAY_COMMON:
        raise SystemExit("captured-p3b replay scripts are not the frozen ones")
    if os.environ.get("GLM53_P4"):
        raise SystemExit("GLM53_P4 must be unset for G4 (recorder off)")
    import vllm.model_executor.layers.quantization.exl3 as e
    if sha(e.__file__) != P4_EXL3:
        raise SystemExit("this image's exl3.py is not the reviewed P4 file")
    print("G4 arm:", "B-tile" if os.environ.get("GLM53_P4_TILE") else "stock (env unset)", "_p4_forced_tile() =",
          e._p4_forced_tile(), flush=True)
    import replay_moe as RM
    orig = RM.load_p3

    def load_p3(path):
        P = orig(path)
        assert P.ENGINES["v091"]["exl3_sha256"] == STOCK_EXL3
        P.ENGINES["v091"]["exl3_sha256"] = P4_EXL3        # the single re-pin (documented above)
        return P
    RM.load_p3 = load_p3
    sys.argv = ["replay_moe.py"] + sys.argv[1:]
    return RM.main()


if __name__ == "__main__":
    sys.exit(main())
