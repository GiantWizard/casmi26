"""Idea 2: transductive molecular-networking rerank over a whole test set.

For molecule X, neighbours Y are other test molecules whose spectra share fragments (modified cosine >= --th, from
net_analysis.py's pairs.parquet). Each candidate c of X gets
    net(c) = sum_Y link(X, Y) * sum_{d in top-k(Y)} conf_Y(d) * Tanimoto(c, d)
and candidates are reordered only inside their same-formula group (slots kept), by z(position) + w * z(net), i.e. the
network breaks isomer ties. Modes: 'oracle' uses each neighbour's true structure (ceiling); 'pred' uses the
neighbour's own current top-k list with 1/rank confidence (deployable).

usage: net_rerank.py SUB.csv SOLUTION.csv PAIRS.parquet OUT_PREFIX [--th 0.7] [--k 3] [--w 1,2]
"""
import argparse
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')

ap = argparse.ArgumentParser()
ap.add_argument('sub'); ap.add_argument('solution'); ap.add_argument('pairs'); ap.add_argument('out')
ap.add_argument('--th', type=float, default=0.7)
ap.add_argument('--k', type=int, default=3)
ap.add_argument('--w', default='0.5,1,2')
ap.add_argument('--conf', default='rank', help="neighbour confidence: rank (1/r) | gap (z-gap of the neighbour's own list) | top1")
ap.add_argument('--typed', type=int, default=0, help='1 = keep only links whose precursor mass gap is a known biotransformation')
ap.add_argument('--iters', type=int, default=1, help='>1: re-run with the reranked lists as neighbour evidence')
ap.add_argument('--modes', default='oracle,pred')
ap.add_argument('--fpkind', default='morgan2', help='morgan2 | atompair | tt')
ap.add_argument('--relgate', type=float, default=0.0, help='use a neighbour candidate only if it resembles some own candidate at >= this')
a = ap.parse_args()
gen = {'morgan2': rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
       'atompair': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
       'tt': rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)}[a.fpkind]
_fp, _form = {}, {}


def fp(s):
    if s not in _fp:
        m = Chem.MolFromSmiles(s)
        _fp[s] = gen.GetFingerprint(m) if m else None
        _form[s] = CalcMolFormula(m) if m else s
    return _fp[s]


sub = {m: [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(a.sub).values}
truth = dict(pd.read_csv(a.solution)[['molecule_id', 'smiles']].values)
P = pd.read_parquet(a.pairs)
P = P[P.modcos >= a.th]
# common natural-product biotransformation mass gaps (Da), both signs; 0 = same mass (isomers / adduct twins)
BIO = np.array([0.0, 2.01565, 14.01565, 15.99491, 18.01056, 28.0313, 30.01056, 31.98983, 42.01057, 44.02621, 56.0626,
                68.0626, 79.96633, 86.00039, 132.04226, 146.05791, 162.05282, 176.03209, 308.11073, 324.10565])
if a.typed:
    dm = np.abs(P.dm.values)
    P = P[np.min(np.abs(dm[:, None] - BIO[None, :]), axis=1) < 0.01]
nbr = {}
for x, y, w in zip(P.a, P.b, P.modcos):
    nbr.setdefault(x, []).append((y, w)); nbr.setdefault(y, []).append((x, w))


CONF = {}


def evidence(y, mode, cur):
    if mode == 'oracle':
        return [(truth[y], 1.0)] if y in truth else []
    lst = cur.get(y, [])[:a.k]
    if a.conf == 'top1':
        return [(lst[0], CONF.get(y, 1.0))] if lst else []
    if a.conf == 'gap':
        return [(s, CONF.get(y, 1.0) / (r + 1)) for r, s in enumerate(lst)]
    return [(s, 1.0 / (r + 1)) for r, s in enumerate(lst)]


def conf_of(lst):
    if len(lst) < 2 or fp(lst[0]) is None or fp(lst[1]) is None:
        return 1.0 if lst else 0.0
    return 1.0 - DataStructs.TanimotoSimilarity(fp(lst[0]), fp(lst[1]))


def z(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


for mode in a.modes.split(','):
    for w in [float(x) for x in a.w.split(',')]:
      cur = dict(sub)
      for it in range(a.iters):
        CONF.clear(); CONF.update({m: conf_of(l) for m, l in cur.items()})
        out = []
        for m, lst in sub.items():
            if not lst or m not in nbr:
                out.append((m, ';'.join(lst[:25]) if lst else 'CCO')); continue
            ev = [(fp(s), c * lw) for y, lw in nbr[m] for s, c in evidence(y, mode, cur) if fp(s) is not None]
            if a.relgate > 0 and ev:
                own = [fp(s) for s in lst if fp(s) is not None]
                ev = [(f, c) for f, c in ev if own and max(DataStructs.BulkTanimotoSimilarity(f, own)) >= a.relgate]
            net = [sum(c * DataStructs.TanimotoSimilarity(fp(s), f) for f, c in ev) if fp(s) is not None and ev else 0.0
                   for s in lst]
            order = list(lst)
            groups = {}
            for i, s in enumerate(lst):
                fp(s); groups.setdefault(_form[s], []).append(i)
            for idx in groups.values():
                if len(idx) < 2:
                    continue
                sc = z([-i for i in idx]) + w * z([net[i] for i in idx])
                for slot, j in zip(idx, [idx[t] for t in np.argsort(-sc, kind='stable')]):
                    order[slot] = lst[j]
            out.append((m, ';'.join(order[:25])))
        cur = {m: s.split(';') for m, s in out}
      if True:
        pd.DataFrame(out, columns=['molecule_id', 'smiles']).to_csv(f'{a.out}_{mode}_w{w:g}.csv', index=False)
        print('wrote', f'{a.out}_{mode}_w{w:g}.csv', flush=True)
