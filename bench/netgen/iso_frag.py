"""Does rule-based fragment explanation (MAGMa, as in frag_rank.py) separate the answer from same-formula isomers on
the LB-calibrated slice, where every learned signal (FPNet, GLACIER, ICEBERG) agrees on the wrong #1?

For each non-conjugate class-2 molecule: the final list's #1 and every same-formula candidate in its top --top are
scored by the intensity share of measured peaks explained by a MAGMa fragment (sqrt intensities, --ppm). Reports the
pairwise test (near-misses: answer above #1; hits: #1 above the best other isomer) and writes the scores for replay.
usage: iso_frag.py OUT.json TEST.parquet SOLUTION.csv FINAL.csv [TEST SOLUTION FINAL ...] [--depth 2] [--ppm 15]
"""
import argparse, json, os, sys, types
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
sys.path.insert(0, '/Users/b/casmi26-repo/bench/obscure_np')
os.environ.setdefault('ONP_KEY_CACHE', '/Users/b/casmi26/mist/keycache')
from common import keys_parallel, scorer_key

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('sets', nargs='+')
ap.add_argument('--depth', type=int, default=2)
ap.add_argument('--ppm', type=float, default=15.0)
ap.add_argument('--top', type=int, default=10)
ap.add_argument('--pkg', default='/Users/b/casmi26/inputs/glacier')
a = ap.parse_args()
sys.path.insert(0, a.pkg)
import gl_runner as G  # noqa: E402
G.setup_env(types.SimpleNamespace(pkg=a.pkg, site='none', wheels=None, any_rdkit=True), lambda *x: None)
from ms_pred.magma import fragmentation as FR  # noqa: E402

PROTON = 1.007276
METAL = {'[M+Na]+': 22.989221, '[M+K]+': 38.963158}
aa = Chem.MolFromSmarts('[NX3][CX4]C(=O)[OX2H1,OX1-]')
_cache = {}


def frag_masses(smi):
    if smi not in _cache:
        try:
            fe = FR.FragmentEngine(smi, max_tree_depth=a.depth, max_broken_bonds=6)
            fe.generate_fragments()
            _cache[smi] = np.unique(np.round(fe.get_frag_masses()[3], 5))
        except Exception:
            _cache[smi] = None
    return _cache[smi]


def explained(masses, spectra):
    vals = []
    for mz, it, ad, mode in spectra:
        if masses is None or len(mz) == 0:
            continue
        ions = np.concatenate([masses + (PROTON if mode == 'positive' else -PROTON)]
                              + ([masses + METAL[ad]] if ad in METAL else []))
        ions.sort()
        idx = np.clip(np.searchsorted(ions, mz), 1, len(ions) - 1)
        hit = np.minimum(np.abs(ions[idx] - mz), np.abs(ions[idx - 1] - mz)) <= mz * a.ppm * 1e-6
        it = np.sqrt(np.clip(it, 0, None)); vals.append(float(it[hit].sum() / max(it.sum(), 1e-12)))
    return float(np.mean(vals)) if vals else np.nan


def formula(s):
    m = Chem.MolFromSmiles(s)
    return CalcMolFormula(m) if m is not None else None


out, rows = {}, []
for tp, sp, fp in zip(a.sets[0::3], a.sets[1::3], a.sets[2::3]):
    te, sol = pd.read_parquet(tp), pd.read_csv(sp)
    fin = {str(m): [x for x in str(v).split(';') if x and x != 'CCO'] for m, v in pd.read_csv(fp).values}
    spec = {}
    for r in te.itertuples():
        keep = r.ms2_mzs < r.precursor_mz - 1.0
        spec.setdefault(str(r.molecule_id), []).append((np.asarray(r.ms2_mzs)[keep],
                                                        np.asarray(r.ms2_normalized_intensities)[keep], r.adduct, r.ionization_mode))
    K = keys_parallel(list({s for l in fin.values() for s in l[:a.top]} | set(sol.smiles)), scorer_key)
    for r in sol.itertuples():
        m = Chem.MolFromSmiles(r.smiles); mid = str(r.molecule_id); L = fin.get(mid, [])
        if r.cls != 2 or m is None or m.HasSubstructMatch(aa) or not L:
            continue
        t = K.get(r.smiles); f1 = formula(L[0])
        uniq, seen = [], set()
        for s in L[:a.top]:
            k = K.get(s)
            if k and k not in seen:
                seen.add(k); uniq.append((s, k))
        rank = next((i + 1 for i, (_, k) in enumerate(uniq) if k == t), 999)
        iso = [(s, k) for s, k in uniq if formula(s) == f1]
        sc = {s: explained(frag_masses(s), spec.get(mid, [])) for s, _ in iso}
        out[f'{os.path.basename(fp)}|{mid}'] = sc
        top_s = uniq[0][0]
        d = dict(set=tp, mid=mid, rank=rank, n_iso=len(iso), s_top=sc.get(top_s))
        if rank > 1 and formula(r.smiles) == f1 and rank <= a.top:
            d['s_ans'] = sc.get(next(s for s, k in uniq if k == t))
        elif rank == 1:
            others = [v for s, v in sc.items() if s != top_s and not np.isnan(v)]
            d['s_best_other'] = max(others) if others else np.nan
        rows.append(d)
    print(tp, 'done', len(rows), flush=True)
json.dump(out, open(a.out, 'w'))
D = pd.DataFrame(rows)
nm = D[D.s_ans.notna()] if 's_ans' in D else D.iloc[:0]
print(f'\nmolecules {len(D)} | rank 1: {(D["rank"] == 1).sum()}')
print(f'near-misses with the answer a same-formula isomer in the top {a.top}: {len(nm)}; MAGMa prefers answer over #1: '
      f'{int((nm.s_ans > nm.s_top).sum())}, ties {int((nm.s_ans == nm.s_top).sum())}')
h = D[(D['rank'] == 1) & D.s_best_other.notna()] if 's_best_other' in D else D.iloc[:0]
print(f'hits with another isomer in the top {a.top}: {len(h)}; MAGMa keeps #1 above every other isomer: '
      f'{int((h.s_top > h.s_best_other).sum())}, ties {int((h.s_top == h.s_best_other).sum())}')
D.to_csv(a.out.replace('.json', '.csv'), index=False)
