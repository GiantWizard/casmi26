"""Which offline subset reproduces the leaderboard? Per-variant MRR change vs base2 on subsets of an eval set, next to
the public-LB change of the same variant.

LB (public, ~124 molecules): base2 0.438, c 0.414 (-0.024), gate 0.438 (0), lr2 0.410 (-0.028 vs gate),
netgen pending, jc-gen 0.431 (not replayable here).
usage: calibrate.py SOLUTION.csv base2=... c=... gate=... lr2=... netgen=...
"""
import sys, os, numpy as np, pandas as pd
from rdkit import Chem
sys.path.insert(0, '/Users/b/casmi26-repo/bench/obscure_np')
os.environ.setdefault('ONP_KEY_CACHE', '/Users/b/casmi26/mist/keycache')
from common import keys_parallel, scorer_key

LB = {'c': -0.024, 'gate': 0.0, 'lr2': -0.028, 'netgen': np.nan}
sol = pd.read_csv(sys.argv[1]); subs = dict(x.split('=', 1) for x in sys.argv[2:])
aa = Chem.MolFromSmarts('[NX3][CX4]C(=O)[OX2H1,OX1-]')
sol['conj'] = [bool(Chem.MolFromSmiles(s)) and Chem.MolFromSmiles(s).HasSubstructMatch(aa) for s in sol.smiles]
L = {n: {str(m): [x for x in str(v).split(';') if x and x != 'CCO'] for m, v in pd.read_csv(p).values} for n, p in subs.items()}
K = keys_parallel(list({s for d in L.values() for l in d.values() for s in l[:25]} | set(sol.smiles)), scorer_key)


def rr(lst, t):
    seen, r = set(), 0
    for s in lst:
        k = K.get(s)
        if not k or k in seen:
            continue
        seen.add(k); r += 1
        if r > 25:
            return 0.0
        if k == t:
            return 1.0 / r
    return 0.0


R = pd.DataFrame({n: [rr(d.get(str(m), []), K.get(s)) for m, s in zip(sol.molecule_id, sol.smiles)] for n, d in L.items()})
R['cls'] = sol.cls.values; R['conj'] = sol.conj.values; R['np'] = sol.is_np.values if 'is_np' in sol else True
subsets = {'all': R.cls > 0, 'non-conj': ~R.conj, 'non-conj cls2': (~R.conj) & (R.cls == 2),
           'non-conj cls3': (~R.conj) & (R.cls == 3), 'conj': R.conj}
ref = 'base2'
print(f'{"variant":8s} {"LB":>7s} ' + ' '.join(f'{k:>15s}' for k in subsets))
print(f'{"n":8s} {"":>7s} ' + ' '.join(f'{int(m.sum()):>15d}' for m in subsets.values()))
for n in L:
    if n == ref:
        continue
    lb = LB.get(n, np.nan)
    cells = []
    for k, m in subsets.items():
        d = (R[n] - R[ref])[m]
        cells.append(f'{d.mean():+.4f} {int((d > 1e-9).sum()):2d}/{int((d < -1e-9).sum()):2d}')
    print(f'{n:8s} {lb:+7.3f} ' + ' '.join(f'{c:>15s}' for c in cells))
