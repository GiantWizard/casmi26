"""Idea 2b: network-guided analog generation. For each test molecule X, single edits (edits.py) of the predicted top-k
structures of its spectrally linked neighbours Y are kept when their mass matches X's neutral mass (10 ppm),
scored, and inserted into X's list.

Generated-candidate score: link strength (best modified cosine to the seeding neighbour) and the FPNet score of the
candidate against X's spectrum (z within X's generated set + own top 10). Policies (--policy):
  slot2      : insert the best generated candidates from rank 2 on (own top-1 kept)
  slot1      : insert from rank 1
  gated      : rank 1 when the FPNet prefers the generated candidate over the own top-1, else rank 2
usage: gen_insert.py TEST.parquet SUB.csv PAIRS.parquet OUT_PREFIX --ckpt fpnet.pt [--th 0.6] [--k 1] [--n_ins 2]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/b/casmi26/fp')
from edits import edit_products  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('test'); ap.add_argument('sub'); ap.add_argument('pairs'); ap.add_argument('out')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--th', type=float, default=0.6)
ap.add_argument('--k', type=int, default=1)
ap.add_argument('--n_ins', type=int, default=2)
ap.add_argument('--code', default='/Users/b/casmi26/inputs/v4b/code')
ap.add_argument('--work', default='/Users/b/casmi26/fp/run/work')
ap.add_argument('--device', default='mps')
ap.add_argument('--policies', default='slot2,slot1,gated')
a = ap.parse_args()
sys.path.insert(0, a.code)
import torch  # noqa: E402
from casmi import fpnet, chem  # noqa: E402
from casmi.spectra import merge_spectra  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem.Descriptors import ExactMolWt  # noqa: E402

ADD = {'[M+H]+': (0, 1.007276), '[M-H]-': (6, -1.007276), '[M+Na]+': (4, 22.989221), '[M+CH2O2-H]-': (8, 44.998203),
       '[M+NH4]+': (1, 18.033826), '[M+Cl]-': (9, 34.969401), '[M-H2O+H]+': (2, -17.003288), '[M+H-H2O]+': (2, -17.003288),
       '[M+K]+': (5, 38.963158), '[M]+': (12, -0.000549), '[M-H2O-H]-': (7, -19.01784)}
bits = np.load(os.path.join(a.work, 'fp_bits.npy'))
model = fpnet.load_model(a.ckpt, a.device)
te = pd.read_parquet(a.test)
sub = {str(m): [x for x in str(v).split(';') if x and x != 'CCO'] for m, v in pd.read_csv(a.sub).values}
P = pd.read_parquet(a.pairs); P = P[P.modcos >= a.th]
nb = {}
for x, y, w in zip(P.a, P.b, P.modcos):
    nb.setdefault(x, []).append((y, w)); nb.setdefault(y, []).append((x, w))
_key, _fp = {}, {}


def key(s):
    if s not in _key:
        _key[s] = chem.score_key(s)
    return _key[s]


def fpv(s):
    if s not in _fp:
        f = chem.raw_fingerprint(s); _fp[s] = None if f is None else f[bits].astype(np.float32)
    return _fp[s]


def fp_scores(g, smis):
    """FPNet score of smis against molecule g's spectra (per polarity merged, z-summed)."""
    F = [fpv(s) for s in smis]; ok = [i for i, f in enumerate(F) if f is not None]
    out = np.full(len(smis), -9.0)
    if not ok:
        return out
    Fm = torch.from_numpy(np.stack([F[i] for i in ok])).to(a.device); tot = np.zeros(len(ok))
    for mode, gm in g.groupby('ionization_mode'):
        specs = [(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float)) for r in gm.itertuples()]
        mz, it = merge_spectra(specs) if len(specs) > 1 else specs[0]
        r0 = gm.iloc[0]
        pm, pi = fpnet.prep_peaks(np.asarray(mz, float), np.asarray(it, float), float(r0.precursor_mz))
        item = dict(mz=pm, it=pi, prec=float(r0.precursor_mz), adduct_ix=ADD[r0.adduct][0], ce=0.0, ce_known=0.0,
                    n_merged=min(len(specs), 8), mode=1.0 if mode == 'positive' else -1.0)
        with torch.no_grad():
            z = model(*fpnet.collate([item], a.device))[0].float()[0]
        sc = (Fm @ z).cpu().numpy(); tot += (sc - sc.mean()) / (sc.std() + 1e-6) if len(sc) > 1 else sc
    out[ok] = tot
    return out


res = {p: [] for p in a.policies.split(',')}
n_gen = 0
diag = []
gensets = {}
for mid, g in te.groupby('molecule_id'):
    mid = str(mid); lst = sub.get(mid, [])
    g = g[g.adduct.isin(ADD)]
    if not len(g) or mid not in nb:
        for p in res: res[p].append((mid, ';'.join(lst[:25]) if lst else 'CCO'))
        continue
    neutral = float(np.median([p - ADD[ad][1] for p, ad in zip(g.precursor_mz, g.adduct)]))
    own = {key(s) for s in lst[:25]}
    gen = {}
    for y, w in nb[mid]:
        for seed in sub.get(str(y), [])[:a.k]:
            for smi in edit_products(seed):
                m = Chem.MolFromSmiles(smi)
                if m is None or abs(ExactMolWt(m) - neutral) > neutral * 1e-5:
                    continue
                k = key(smi)
                if k and k not in own and (k not in gen or gen[k][1] < w):
                    gen[k] = (smi, w)
    if not gen:
        for p in res: res[p].append((mid, ';'.join(lst[:25]) if lst else 'CCO'))
        continue
    n_gen += 1
    gl = [v[0] for v in gen.values()]; gw = np.array([v[1] for v in gen.values()])
    pool = gl + lst[:10]
    sc = fp_scores(g, pool)
    gsc, osc = sc[:len(gl)], sc[len(gl):]
    order = np.argsort(-(gsc + 2.0 * gw))
    best = [gl[i] for i in order[:a.n_ins]]
    gensets[mid] = dict(smiles=gl, link=gw.tolist(), fp=gsc.tolist(), own=lst[:10], own_fp=osc.tolist())
    diag.append(dict(molecule_id=mid, n_gen=len(gl), g_fp=float(gsc[order[0]]), g_link=float(gw[order[0]]),
                     own1_fp=float(osc[0]) if len(osc) else np.nan, own_max_fp=float(osc.max()) if len(osc) else np.nan,
                     best=best[0]))
    for p in res:
        if p == 'slot1':
            new = best + lst
        elif p == 'slot2':
            new = lst[:1] + best + lst[1:]
        else:  # gated
            top_own = osc[0] if len(osc) else -9.0
            new = (best + lst) if gsc[order[0]] > top_own else (lst[:1] + best + lst[1:])
        res[p].append((mid, ';'.join(new[:25])))
for p, rows in res.items():
    pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(f'{a.out}_{p}.csv', index=False)
pd.DataFrame(diag).to_csv(f'{a.out}_diag.csv', index=False)
json.dump(gensets, open(f'{a.out}_gensets.json', 'w'))
print('molecules with generated candidates', n_gen, '| wrote', list(res))
