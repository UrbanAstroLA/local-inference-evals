"""First-contrast session geometry (PROPOSAL.md s.3, amendment-22 design). Pure functions over
fixtures/position_plan.json and fixtures/ids_manifest.json; imported by runner/client_fc.py, runner/plan_check.py,
analysis/gate_fc.py and analysis/analyze_fc.py so that the client, the plan check and the analyses use ONE definition
of every length, position set, salt and expected cache value. No token ids are read here.

Labels: [A22-n] marks a quantity introduced or changed by PLAN.md amendment 22 item n (PROPOSAL.md keeps the pre-22
text in PROPOSAL.md.pre22).
"""
import hashlib
import json
import os

BLOCK, KPOOL, RING = 4352, 4, 8
LEAD, TAIL = 512, 512
LP_SHIFT = 7168            # LP construction: P_L = P + 7168
REF2_LEN = 8192            # [A22-3] second prefill reference: build length P + 8192 (LP warm of round 2, adapted)
WMAX = 8448                # FORCEWIN_WMAX for this session (module default; attempts 1-2 ran 5120)
REGIME = 65536             # B1 tiled-route selection regime boundary crossed by M3 (amendment 20d item 2)
NDESIG, NSEL = 1024, 64
CLOSE = 154842
STOPS = (154820, 154827, 154829)

# [A22-5] conv-state boundary control (CV): fixtures, schedules, frontier offsets relative to the first block crossing
CV_FIXTURES = ("L5", "L1")
CV_MODES = ("poison", "oracle", "reject:2", "natural")
CV_BACK = 63               # CV prompt P' = B1 - 63 (P' = 1 mod 4: oracle and reject:2 commit counts hit B1 - 8 .. B1 + 4)
CV_LIST = 72               # forced list length: frontier reaches B1 + 7
CV_FRONTIER_OFFSETS = {
    "poison": (-8, -6, -5, -4, -3, -2, -1, 0, 4),     # consecutive run straddles the align-mode column change
    "oracle": (-8, -4, 0, 4),                         # oracle commits 4 per step: F = 1 + 4k exactly
    "reject:2": (-8, -4, -2, 0, 4),                   # reject:2 commits 2 per step: F odd exactly
    "natural": (-8, -6, -4, -2, 0, 4),
}
PS_DUMP = 1024             # [A22-4] prefix sentinel: dump [c0 - 1024, c0) of the last cached (shared) block

SALT_NONCE = "20261014-fc1"
SALT_ROLES = {"primary": ("s1", "s2", "s3"), "lp": ("l1", "l2", "l3")}


def salt(name):
    """Independent cache salts: one per round and role (primary s1-s3, LP l1-l3)."""
    return "fc1-%s-%s" % (name, SALT_NONCE)


def _sha(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def load(fc_root):
    pp = json.load(open(os.path.join(fc_root, "fixtures", "position_plan.json")))
    man = json.load(open(os.path.join(fc_root, "fixtures", "ids_manifest.json")))
    return geometry(pp, man)


def geometry(pp, man):
    G = {"order": list(pp["batch_order"]), "fx": {}}
    for f in pp["fixtures"]:
        n = f["fixture"]
        m = man["fixtures"][n]
        P, W = int(f["P"]), int(f["W"])
        assert P == m["P"] and W == m["W"], n
        L = W + 1                                            # forced list length (window + history's next token)
        P_L = P + LP_SHIFT
        Q = list(f["positions"]["Q_primary"]["list"])
        T = list(range(W - TAIL + 1, W + 1))
        desig = sorted(set(Q) | set(T))
        assert _sha(Q) == f["positions"]["Q_primary"]["sha256"] and _sha(desig) == f["positions"]["designated"]["sha256"]
        sel = list(f["positions"]["selrec"]["list"])
        cross = [int(b) for b in f["primary"]["blocks_crossed_in_window"]]
        g = dict(
            fixture=n, doc=int(f["doc"]), cls=f["cls"], near_close=f["cls"] == "near_close",
            P=P, W=W, L=L, c0=(P // BLOCK) * BLOCK, rem=P % BLOCK,
            P_L=P_L, W_s=W - LP_SHIFT, L_s=W - LP_SHIFT + 1, c0_L=(P_L // BLOCK) * BLOCK,
            ref2_len=P + REF2_LEN,
            crossings_abs=cross, crossings_o=[b - P for b in cross],
            regime_o=[REGIME - P] if P <= REGIME < P + W + 1 else [],
            Q=Q, Q_noend=[o for o in Q if o != W], T=T, end=W, desig=desig, selrec=sel,
            lp_common=[o for o in desig if o >= LP_SHIFT + LEAD],
            Q_common=[o for o in Q if o >= LP_SHIFT + LEAD],
            desig_lp=sorted(o - LP_SHIFT for o in desig if o >= LP_SHIFT),
            selrec_lp=sorted(o - LP_SHIFT for o in sel if o >= LP_SHIFT),
            # pools compared against the prefill references: wholly inside [P, P_L) (P = 0 mod 8)
            ref_pools=(P // KPOOL, P_L // KPOOL),
        )
        if n in CV_FIXTURES:
            B = cross[0]
            Pc = B - CV_BACK
            assert (Pc // BLOCK) * BLOCK == g["c0"] and Pc > P, n
            # snapshot targets in committed-output counts F: frontier = P' + F - 2 = B + off  ->  F = B + off - P' + 2
            g["cv"] = dict(B=B, P_cv=Pc, x=Pc - P, list_len=CV_LIST,
                           targets={m_: sorted(B + off - Pc + 2 for off in offs)
                                    for m_, offs in CV_FRONTIER_OFFSETS.items()},
                           note="frontier offsets from B1: %s" % json.dumps(CV_FRONTIER_OFFSETS))
        G["fx"][n] = g
    return G


def doc_units(G):
    """Question units for the question-weighted sensitivity: fixtures of the same doc averaged first."""
    u = {}
    for n in G["order"]:
        u.setdefault(G["fx"][n]["doc"], []).append(n)
    return u
