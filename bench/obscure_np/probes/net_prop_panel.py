"""Analog propagation on the Enveda NP panel (250 timsTOF natural products; dmitriigluzdov NP-panel benchmark).

Base ranking: honest FPNet A+B score (both nets held the panel out) over each molecule's 10-ppm PubChem window, optionally
+ w_pop * popularity. Edges: modified cosine between the panel molecules' own spectra (train rows, ingest_lib
enveda-np-examples). Rerank each molecule's top-N by base + lam * z(prop), prop = similarity of a candidate to its spectral
neighbours' current top-K candidates (weighted by edge score and 1/rank). Reports MRR@25 (metric-style dedup by key).
"""
import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from net_prop import merged_spectrum, mod_cosine  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ap = argparse.ArgumentParser()
ap.add_argument("--panel-dir", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--train-structs", required=True)
ap.add_argument("--pc-smiles", required=True)
ap.add_argument("--top-n", type=int, default=50)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--thr", type=float, default=0.6)
ap.add_argument("--min-peaks", type=int, default=5)
ap.add_argument("--w-pop", type=float, nargs="+", default=[0.0, 0.25])
ap.add_argument("--lam", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 4])
ap.add_argument("--iters", type=int, default=1)
a = ap.parse_args()

c = pd.read_parquet(f"{a.panel_dir}/pubchem_candidates.parquet")
c = c[np.isfinite(c.fpnet_score)].copy()
c["molecule"] = c.molecule.astype(str)
c["ident"] = np.where(c.scorer_key != "", c.scorer_key, c.connectivity)
c["z"] = c.groupby("molecule").fpnet_score.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
c["pop"] = c.log_substances.fillna(0) + c.log_pubmed.fillna(0)
pcs = np.load(a.pc_smiles, mmap_mode="r")                       # concatenated SMILES bytes
pco = np.load(a.pc_smiles.replace("pc_smiles", "pc_off"), mmap_mode="r")


def smiles_of(row):
    return bytes(pcs[pco[row]:pco[row + 1]]).decode()

# panel spectra from train
ts = pd.read_parquet(a.train_structs, columns=["inchikey14", "key"])
k_of = dict(zip(ts.inchikey14, ts.key))
tr = pq.read_table(a.train, filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas()
tr["molecule"] = tr.inchikey14.map(k_of).fillna(tr.inchikey14)
spec = {m: merged_spectrum(g) for m, g in tr.groupby("molecule")}
mols = sorted(set(c.molecule) & set(spec))
print("panel molecules with candidates and spectra:", len(mols))
nb = {m: [] for m in mols}
for i, x in enumerate(mols):
    for y in mols[i + 1:]:
        s, n = mod_cosine(spec[x], spec[y])
        if s >= a.thr and n >= a.min_peaks:
            nb[x].append((y, s)); nb[y].append((x, s))
deg = np.array([len(nb[m]) for m in mols])
print(f"spectral edges: share with >=1 neighbour {np.mean(deg > 0):.2f}, mean degree {deg.mean():.2f}")

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fp = {}


def fp(row):
    if row not in _fp:
        m = Chem.MolFromSmiles(smiles_of(row))
        _fp[row] = GEN.GetFingerprint(m) if m is not None else None
    return _fp[row]


def lists_for(w_pop):
    out = {}
    for m, g in c[c.molecule.isin(mols)].groupby("molecule"):
        g = g.assign(base=g.z + w_pop * g["pop"]).sort_values("base", ascending=False)
        g = g.drop_duplicates("ident").head(a.top_n)
        out[m] = g[["tier_row", "ident", "is_truth", "base"]].reset_index(drop=True)
    return out


def mrr(order_of):
    rr = []
    for m in mols:
        g = order_of[m]
        hit = np.flatnonzero(g.is_truth.values[:25])
        rr.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return np.array(rr)


for w_pop in a.w_pop:
    base = lists_for(w_pop)
    r0 = mrr(base)
    print(f"\nbase = z(FPNet A+B) + {w_pop} * pop: MRR@25 {r0.mean():.4f}  top1 {(r0 == 1).mean():.2f}")
    for lam in a.lam[1:]:
        cur = base
        for _ in range(a.iters):
            new = {}
            for m in mols:
                g = base[m]
                if not nb[m]:
                    new[m] = g; continue
                prop = np.zeros(len(g)); ws = 0.0
                cf = [fp(r) for r in g.tier_row]
                for n, w in nb[m]:
                    for rk, r2 in enumerate(cur[n].tier_row.values[:a.k], 1):
                        f2 = fp(r2)
                        if f2 is None:
                            continue
                        prop += w / rk * np.array([DataStructs.TanimotoSimilarity(f, f2) if f is not None else 0.0 for f in cf])
                    ws += w
                prop /= ws
                zb = (g.base - g.base.mean()) / (g.base.std() + 1e-9)
                zp = (prop - prop.mean()) / (prop.std() + 1e-9)
                new[m] = g.iloc[np.argsort(-(zb.values + lam * zp), kind="stable")].reset_index(drop=True)
            cur = new
        r = mrr(cur); d = r - r0; hasnb = np.array([bool(nb[m]) for m in mols])
        lo, hi = np.percentile([np.random.default_rng(i).choice(d, len(d)).mean() for i in range(1000)], [5, 95])
        print(f"  lam {lam:<4} MRR {r.mean():.4f}  delta {d.mean():+.4f} [90% CI {lo:+.4f},{hi:+.4f}]  "
              f"with-neighbour delta {d[hasnb].mean():+.4f}  improved {(d > 0).sum()} / worse {(d < 0).sum()}")
