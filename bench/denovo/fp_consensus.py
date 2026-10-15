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
        d['top5'] = F[:5]
        d['mist'] = p
        rows.append((d, t))
    print(s, len(rows), flush=True)


def tani(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def build(d, name):
    m, c5, t1 = d['mist'], d['cons5'], d['top1'] > 0
    agree = d['top5'].mean(0)  # fraction of the top 5 candidates that set each bit
    return {
        'mist': m > 0.2, 'top1': t1, 'cons5': c5 > 0.3,
        'top1&mist.05': t1 & (m > 0.05), 'top1&mist.1': t1 & (m > 0.1),
        '(top1&mist.05)|mist.5': (t1 & (m > 0.05)) | (m > 0.5),
        '(top1&mist.1)|mist.4': (t1 & (m > 0.1)) | (m > 0.4),
        'agree5>=0.6': agree >= 0.6, 'agree5>=0.8': agree >= 0.8,
        '(agree5>=0.6)|mist.4': (agree >= 0.6) | (m > 0.4),
        '(agree5>=0.8)|mist.3': (agree >= 0.8) | (m > 0.3),
        'mist+cons5>.2': 0.5 * m + 0.5 * c5 > 0.2,
        'oracle_k.7a.4': d['deg74'], 'oracle_k.5a.6': d['deg56'],
    }[name]


NAMES = ['oracle_k.7a.4', 'oracle_k.5a.6', 'mist', 'top1', 'cons5', 'top1&mist.05', 'top1&mist.1', '(top1&mist.05)|mist.5', '(top1&mist.1)|mist.4',
         'agree5>=0.6', 'agree5>=0.8', '(agree5>=0.6)|mist.4', '(agree5>=0.8)|mist.3', 'mist+cons5>.2']
rng = np.random.default_rng(0)
for d, t in rows:
    for key, keep, add in (('deg74', .7, .4), ('deg56', .5, .6)):
        q = t > 0; on = np.where(q)[0]; off = np.where(~q)[0]; q = q.copy()
        q[rng.choice(on, int(round(len(on) * (1 - keep))), replace=False)] = False
        q[rng.choice(off, int(round(len(on) * add)), replace=False)] = True
        d[key] = q
res = []
for name in NAMES:
    tt, t1, mg = [], [], []
    for d, t in rows:
        q = build(d, name); a1 = tani(q, t > 0); b1 = max(tani(q, f > 0) for f in d['top5'])
        tt.append(a1); t1.append(b1); mg.append(a1 - b1)
    mg = np.array(mg)
    res.append(dict(src=name, T_truth=np.mean(tt), T_nearest_top5=np.mean(t1), margin=mg.mean(), frac_closer_truth=(mg > 0).mean()))
Rz = pd.DataFrame(res)
print(Rz.round(3).to_string(index=False))
Rz.to_parquet(out)
