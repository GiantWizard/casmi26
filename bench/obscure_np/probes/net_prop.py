"""Test-set analog propagation (molecular networking) as a final rerank, evaluated on saved eval-run lists.

Edges: modified cosine between molecules' spectra (same polarity; peaks matched directly or shifted by the precursor
difference). For molecule A with neighbours N (edge weight w), each of A's candidates c gets
    prop(c) = sum_n w_n * sum_{r<=K} (1/r) * Tanimoto(c, cand_n[r]) / sum_n w_n
and the list is re-sorted by  base(c) + lam * z(prop)  where base = -log(rank).

    python net_prop.py --set NAME LIST.csv TEST.parquet SOLUTION.csv [--set ...] [--lam 0 0.5 1 2]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from common import keys_parallel, scorer_key  # noqa: E402

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def merged_spectrum(g, topk=60):
    """Union of a molecule's spectra in one polarity (prefer positive), binned at 0.01, normalised, sqrt intensities."""
    pos = g[g.ionization_mode == "positive"]
    g = pos if len(pos) else g
    pk = {}
    for mz, it in zip(g.ms2_mzs, g.ms2_normalized_intensities):
        for m, i in zip(mz, it):
            b = round(float(m), 2)
            pk[b] = max(pk.get(b, 0.0), float(i))
    if not pk:
        return None
    mz = np.array(sorted(pk)); it = np.sqrt(np.array([pk[m] for m in mz]))
    keep = np.argsort(-it)[:topk]; mz, it = mz[keep], it[keep]
    o = np.argsort(mz)
    return dict(mz=mz[o], it=it[o] / np.linalg.norm(it), prec=float(g.precursor_mz.median()),
                pol=g.ionization_mode.iloc[0])


def mod_cosine(a, b, tol=0.01):
    if a["pol"] != b["pol"]:
        return 0.0, 0
    shift = a["prec"] - b["prec"]
    pairs = []
    for i, m in enumerate(a["mz"]):
        for d in (0.0, shift) if abs(shift) > tol else (0.0,):
            j = np.searchsorted(b["mz"], m - d - tol)
            while j < len(b["mz"]) and b["mz"][j] <= m - d + tol:
                pairs.append((a["it"][i] * b["it"][j], i, j)); j += 1
    pairs.sort(reverse=True); ua, ub, s, n = set(), set(), 0.0, 0
    for p, i, j in pairs:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); s += p; n += 1
    return s, n


def fp(smi, cache={}):
    if smi not in cache:
        m = Chem.MolFromSmiles(smi)
        cache[smi] = GEN.GetFingerprint(m) if m is not None else None
    return cache[smi]


def run(name, lists_csv, test_pq, sol_csv, lams, K, thr, min_peaks):
    sub = pd.read_csv(lists_csv); sol = pd.read_csv(sol_csv).set_index("molecule_id").scorer_key
    L = {m: [x for x in str(s).split(";") if x] for m, s in zip(sub.molecule_id, sub.smiles)}
    te = pd.read_parquet(test_pq)
    spec = {m: merged_spectrum(g) for m, g in te.groupby("molecule_id")}
    mids = [m for m in L if spec.get(m) is not None]
    nb = {m: [] for m in mids}
    for i, a in enumerate(mids):
        for b in mids[i + 1:]:
            s, n = mod_cosine(spec[a], spec[b])
            if s >= thr and n >= min_peaks:
                nb[a].append((b, s)); nb[b].append((a, s))
    keys = keys_parallel([x for v in L.values() for x in v], scorer_key)
    deg = np.array([len(nb[m]) for m in mids])
    out = {lam: [] for lam in lams}
    for m in L:
        lst = L[m]; t = sol.get(m)
        for lam in lams:
            order = lst
            if lam > 0 and nb.get(m):
                prop = np.zeros(len(lst)); wsum = 0.0
                for n, w in nb[m]:
                    nf = [fp(x) for x in L[n][:K]]
                    for r, f2 in enumerate(nf, 1):
                        if f2 is None:
                            continue
                        sims = np.array([DataStructs.TanimotoSimilarity(fp(c), f2) if fp(c) is not None else 0 for c in lst])
                        prop += w * sims / r
                    wsum += w
                prop /= max(wsum, 1e-9)
                z = (prop - prop.mean()) / (prop.std() + 1e-9)
                score = -np.log(np.arange(1, len(lst) + 1)) + lam * z
                order = [lst[i] for i in np.argsort(-score, kind="stable")]
            ks = [keys.get(x) for x in order]
            out[lam].append(dict(mid=m, has_nb=bool(nb.get(m)), rr=1.0 / (ks.index(t) + 1) if t in ks else 0.0))
    print(f"{name}: {len(mids)} molecules, with >=1 neighbour {np.mean(deg > 0):.2f}, mean degree {deg.mean():.1f}")
    base = pd.DataFrame(out[lams[0]]).set_index("mid").rr
    for lam in lams:
        d = pd.DataFrame(out[lam]).set_index("mid")
        diff = d.rr - base
        print(f"  lam {lam:<4} MRR {d.rr.mean():.4f}  delta {diff.mean():+.4f}  (nb-only {diff[d.has_nb].mean():+.4f}, "
              f"improved {(diff > 0).sum()} / worse {(diff < 0).sum()})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=4, action="append", required=True, metavar=("NAME", "LIST", "TEST", "SOL"))
    ap.add_argument("--lam", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--min-peaks", type=int, default=5)
    a = ap.parse_args()
    for s in a.set:
        run(*s, a.lam, a.k, a.thr, a.min_peaks)
