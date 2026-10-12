"""Rebuild jc-c2's pre-fusion main list (v4) under alternative PubChem gates, from a run's saved lists.

Cell 17 leaves molecules with lib_max >= 0.9 'untouched' (no PubChem candidates). lib_max is not saved, so a
molecule counts as gated when its PubChem list is non-empty but none of those keys reached v4. For those, the
base list is v4 itself; the variants re-apply cell 17's merge/promote rules as if the gate were open.
"""
import argparse
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
SLOTS_AGG, SLOTS_GENTLE, REL_TH, S_TAU, POP_TAU = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20], 600.0, 6.0, 5.0
pc = json.load(open(os.path.join(a.run, "pc_lists.json")))
v4 = pd.read_csv(os.path.join(a.run, "v4_submission.csv"))
eng = json.load(open(os.path.join(a.run, "eng_lists.json")))


def merge(base, base_keys, p, pk, slots, n=25):
    bk = set(base_keys)
    pcs = [s for s, k in zip(p, pk) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out


def promote(base, base_keys, p, pk, slots, n=25):
    usual = merge(base, base_keys, p, pk, slots, n=n)
    return ([p[0]] + [x for x in usual if x != p[0]])[:n] if p else usual


def rel_of(d):
    import math
    b = d.get("best_pool_fz")
    return (d.get("fz_top") if d.get("fz_top") is not None else d["pc_fz"][0]) - b if (b is not None and math.isfinite(b)) else 1e9


gated = []
for mid, s in zip(v4.molecule_id, v4.smiles):
    d = pc.get(mid) or {}
    lst = [x for x in s.split(";") if x]
    if d.get("pc") and not (set(d["pc"]) & set(lst)):
        gated.append(mid)
print(f"gated (lib_max >= 0.9 inferred): {len(gated)} / {len(v4)}")
G = set(gated)

variants = {
    # open the gate but only interleave at the gentle slots: ranks 1-3 can never change
    "gentle": lambda base, d: merge(base, [], d["pc"], d["pc_keys"], SLOTS_GENTLE),
    # open the gate with cell 17's full rule (aggressive slots + promotion) - the risky upper bound
    "full": lambda base, d: (promote if ((d.get("S") or 0) > S_TAU and float(d.get("top_pop") or 0) >= POP_TAU) or rel_of(d) > 250
                             else merge)(base, [], d["pc"], d["pc_keys"], SLOTS_AGG if rel_of(d) > REL_TH else SLOTS_GENTLE),
    # gentle, but only PubChem's top 2 (fewer slots spent)
    "gentle2": lambda base, d: merge(base, [], d["pc"][:2], d["pc_keys"][:2], SLOTS_GENTLE),
    # slots 2 and 4: one step braver than gentle, rank 1 still protected
    "slot2": lambda base, d: merge(base, [], d["pc"][:2], d["pc_keys"][:2], [2, 4]),
}
os.makedirs(a.out, exist_ok=True)
for name, fn in variants.items():
    rows = []
    for mid, s in zip(v4.molecule_id, v4.smiles):
        lst = [x for x in s.split(";") if x]
        if mid in G:
            # base keys: v4 holds pool/generated SMILES; dedupe against them by SMILES only (keys absent offline)
            new = fn([x for x in lst if x != "CCO"], pc[mid])
            seen, ded = set(), []
            for x in new:
                if x not in seen:
                    seen.add(x); ded.append(x)
            lst = ded[:25] or ["CCO"]
        rows.append((mid, ";".join(lst)))
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(os.path.join(a.out, f"v4_{name}.csv"), index=False)
    print("wrote", name)
