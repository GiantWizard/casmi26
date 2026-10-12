"""Variant: after fusion + rerank, put the PubChem channel's top candidate at rank 1 when its confidence
(S = z-scored fingerprint fit + 0.25*pop, or gap = fz[0]-fz[1]) clears a threshold.

    python promote_after_fusion.py --run OUT --final rp_v4gentle.csv --col S --thr 3.9 --out promo.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--final", required=True)
ap.add_argument("--col", choices=["S", "gap"], default="S")
ap.add_argument("--thr", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
pc = json.load(open(os.path.join(a.run, "pc_lists.json")))


def conf(d):
    if a.col == "S":
        return d.get("S")
    fz = d.get("pc_fz") or []
    return fz[0] - fz[1] if len(fz) > 1 else np.nan


rows, n = [], 0
for mid, s in pd.read_csv(a.final).values:
    lst = [x for x in str(s).split(";") if x]
    d = pc.get(mid) or {}
    c = conf(d) if d.get("pc") else None
    if c is not None and np.isfinite(c) and c > a.thr:
        top = d["pc"][0]
        lst = [top] + [x for x in lst if x != top]
        n += 1
    rows.append((mid, ";".join(lst[:25])))
pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(a.out, index=False)
print(f"promoted {n} molecules")
