"""How good a 4096-bit ECFP4 (Morgan r2) can we predict for out-of-pool molecules, as FRIGID conditioning?

Sources: MIST (FRIGID MassSpecGym encoder, true formula), consensus of the pipeline's fused candidate list (rank-weighted
mean of candidate fingerprints, weight 1/(rank+k)), and blends. Metric: Tanimoto of the thresholded prediction vs the
true fingerprint, best threshold per source, on non-conjugate class-3 molecules of ONP sets A (d2) and B (d0).
usage: fp_consensus.py OUT.parquet
"""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, '/Users/b/casmi26/frigid/src'); sys.path.insert(0, '/Users/b/casmi26/mist/pkg')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
sys.argv, out = sys.argv[:1] + ['x', 'x', 'x', 'x', '--fp', 'truth'], sys.argv[1]
from importlib.machinery import SourceFileLoader  # reuse mist_fp from frigid_eval without running its main loop
src = open('/Users/b/casmi26/frigid/frigid_eval.py').read().split('# ---- FRIGID')[0]
ns = {}; exec(compile(src, 'frigid_eval_head', 'exec'), ns)
mist_fp = ns['mist_fp']
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
fp = lambda m: gen.GetFingerprintAsNumPy(m).astype(np.float32)
aa = Chem.MolFromSmarts('[NX3][CX4]C(=O)[OX2H1,OX1-]')
R = '/Users/b/casmi26/mist/runs'
rows = []
for s, run, tp in [('onpa', 'jc-onp-eval-d2', 'onpa/test_draw2.parquet'), ('onpb', 'jc-onp-eval-b-d0', 'onpb/test_draw0.parquet')]:
    sol = pd.read_csv(f'{R}/{s}/solution.csv'); te = pd.read_parquet(f'{R}/{tp}')
    full = {str(m): [x for x in str(v).split(';') if x and x != 'CCO'] for m, v in pd.read_csv(f'{R}/{run}.rp_full/rp_c2fix.csv').values}
    for r in sol.itertuples():
        m = Chem.MolFromSmiles(r.smiles)
        if r.cls != 3 or m is None or m.HasSubstructMatch(aa):
            continue
        t = fp(m); form = CalcMolFormula(m)
        g = te[te.molecule_id == r.molecule_id]
        spectra = [(np.asarray(x.ms2_mzs, float), np.asarray(x.ms2_normalized_intensities, float), float(x.precursor_mz), x.adduct) for x in g.itertuples()]
        try:
            p = mist_fp(form, spectra)
        except Exception:
            p = None
        cands = [c for c in (Chem.MolFromSmiles(x) for x in full.get(str(r.molecule_id), [])[:50]) if c is not None]
        if p is None or not cands:
            continue
        F = np.stack([fp(c) for c in cands])
        same = np.array([CalcMolFormula(c) == form for c in cands])
        d = dict(set=s, mid=r.molecule_id, nbits=t.sum())
        for k in (1, 5, 20):
            w = 1 / (np.arange(len(F)) + k); d[f'cons{k}'] = (w[:, None] * F).sum(0) / w.sum()
        d['top1'] = F[0]
        d['mist'] = p
        rows.append((d, t))
    print(s, len(rows), flush=True)


def tani(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


res = []
for name in ['mist', 'top1', 'cons1', 'cons5', 'cons20', 'mist+cons5', 'mist+cons20', 'max(mist,cons5)']:
    for th in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        v = []
        for d, t in rows:
            if name.startswith('mist+'):
                q = 0.5 * d['mist'] + 0.5 * d[name[5:]]
            elif name.startswith('max('):
                q = np.maximum(d['mist'], d['cons5'])
            else:
                q = d[name]
            v.append(tani(q > th, t > 0))
        res.append(dict(src=name, th=th, mean=np.mean(v), median=np.median(v), ge05=np.mean(np.array(v) >= 0.5)))
Rz = pd.DataFrame(res)
print(Rz.loc[Rz.groupby('src')['mean'].idxmax()].round(3).to_string(index=False))
Rz.to_parquet(out)
