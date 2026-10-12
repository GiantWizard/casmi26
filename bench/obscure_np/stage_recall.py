"""Where does the truth get lost? Rank of the truth (scorer key) in each saved list of an eval run.

    python stage_recall.py --run OUT_DIR --solution solution.csv --final sub_c2.csv --out ranks.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import keys_parallel, scorer_key

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--solution", required=True)
ap.add_argument("--final", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sol = pd.read_csv(a.solution).set_index("molecule_id")
eng = json.load(open(os.path.join(a.run, "eng_lists.json")))
pc = json.load(open(os.path.join(a.run, "pc_lists.json")))
split = lambda s: [x for x in str(s).split(";") if x and x != "CCO"]
v4 = {m: split(s) for m, s in pd.read_csv(os.path.join(a.run, "v4_submission.csv")).values}
fin = {m: split(s) for m, s in pd.read_csv(a.final).values}
lists = {"v4": v4, "eng": {m: d.get("smiles", []) for m, d in eng.items()},
         "pc": {m: d.get("pc", []) for m, d in pc.items()}, "final": fin}
allsmi = {s for L in lists.values() for v in L.values() for s in v}
K = keys_parallel(allsmi, scorer_key)


def rank(lst, key):
    seen = []
    for s in lst:
        k = K.get(s)
        if k and k not in seen:
            seen.append(k)
            if k == key:
                return len(seen)
    return np.nan


rows = []
for mid, r in sol.iterrows():
    row = {"molecule_id": mid, "cls": r.get("cls"), "adduct_group": r.get("adduct_group")}
    for name, L in lists.items():
        row[f"r_{name}"] = rank(L.get(mid, []), r["scorer_key"])
    pcd = pc.get(mid, {})
    row["pc_promoted_S"] = pcd.get("S")
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
for cls, g in [("all", df)] + list(df.groupby("cls")):
    print(f"\n== cls {cls}  n={len(g)}")
    for name in lists:
        r = g[f"r_{name}"]
        print(f"  {name:<6} present {r.notna().mean():6.1%}  top1 {(r == 1).mean():6.1%}  top5 {(r <= 5).mean():6.1%}  "
              f"top25 {(r <= 25).mean():6.1%}")
any_ = g[["r_v4", "r_eng", "r_pc"]].notna().any(axis=1)
print("\nlost between candidate lists and final top25:",
      int((df[["r_v4", "r_eng", "r_pc"]].notna().any(axis=1) & df.r_final.isna()).sum()))
print("in pc list but not final:", int((df.r_pc.notna() & df.r_final.isna()).sum()),
      "| pc rank dist of those:", df.loc[df.r_pc.notna() & df.r_final.isna(), "r_pc"].describe().round(1).to_dict())
print("final rank>1 but some list has it at rank 1:",
      int(((df.r_final > 1) & (df[["r_v4", "r_eng", "r_pc"]] == 1).any(axis=1)).sum()))
