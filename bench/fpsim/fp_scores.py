"""FPNet score of every fused candidate, as a GLACIER-style score dict {mid: {smiles: score}} for the same-formula
rerank (replay_rerank.py ONP_EXTRA). The molecule's spectra are merged per polarity; with both polarities the
per-polarity scores are summed after z-scoring over the molecule's candidates. Candidate fingerprints come from
casmi.chem.raw_fingerprint(smiles)[fp_bits], so PubChem-only candidates outside the pool are scored too.

usage: fp_scores.py TEST.parquet CANDS.csv OUT.json --ckpt a.pt [--device mps]
CANDS.csv: molecule_id,smiles with ';'-joined candidates (e.g. a replay written with ONP_FULL=1)
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('test'); ap.add_argument('cands'); ap.add_argument('out')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--code', default='/Users/b/casmi26/inputs/v4b/code')
ap.add_argument('--work', default='/Users/b/casmi26/fp/run/work')
ap.add_argument('--device', default='mps')
a = ap.parse_args()
sys.path.insert(0, a.code)
import torch  # noqa: E402
from casmi import fpnet, chem  # noqa: E402
from casmi.spectra import merge_spectra  # noqa: E402

ADD = {'[M+H]+': 0, '[M-H]-': 6, '[M+Na]+': 4, '[M+CH2O2-H]-': 8, '[M+NH4]+': 1, '[M+Cl]-': 9, '[M-H2O+H]+': 2,
       '[M+H-H2O]+': 2, '[M+K]+': 5, '[M]+': 12, '[M-H2O-H]-': 7}
bits = np.load(os.path.join(a.work, 'fp_bits.npy'))
models = [fpnet.load_model(p, a.device) for p in a.ckpt.split(',')]
te = pd.read_parquet(a.test)
cands = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(a.cands).values}
_fp = {}


def fpv(s):
    if s not in _fp:
        f = chem.raw_fingerprint(s)
        _fp[s] = None if f is None else f[bits].astype(np.float32)
    return _fp[s]


out = {}
for mid, g in te.groupby('molecule_id'):
    lst = cands.get(str(mid), [])
    g = g[g.adduct.isin(ADD)]
    if not lst or not len(g):
        continue
    F = [fpv(s) for s in lst]
    ok = [i for i, f in enumerate(F) if f is not None]
    if not ok:
        continue
    Fm = torch.from_numpy(np.stack([F[i] for i in ok])).to(a.device)
    total = np.zeros(len(ok))
    for mode, gm in g.groupby('ionization_mode'):
        specs = [(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float)) for r in gm.itertuples()]
        mz, it = merge_spectra(specs) if len(specs) > 1 else specs[0]
        r0 = gm.iloc[0]
        ces = [float(np.mean(c)) for c in gm.collision_energy_ev if c is not None and len(c)]
        ce = float(np.mean(ces)) if ces else np.nan
        pm, pi = fpnet.prep_peaks(np.asarray(mz, float), np.asarray(it, float), float(r0.precursor_mz))
        item = dict(mz=pm, it=pi, prec=float(r0.precursor_mz), adduct_ix=ADD[r0.adduct], ce=0.0 if np.isnan(ce) else ce,
                    ce_known=0.0 if np.isnan(ce) else 1.0, n_merged=min(len(specs), 8), mode=1.0 if mode == 'positive' else -1.0)
        with torch.no_grad():
            z = torch.stack([m(*fpnet.collate([item], a.device))[0].float()[0] for m in models]).mean(0)
        sc = (Fm @ z).cpu().numpy()
        total += (sc - sc.mean()) / (sc.std() + 1e-6) if len(sc) > 1 else sc
    out[str(mid)] = {lst[i]: float(v) for i, v in zip(ok, total)}
json.dump(out, open(a.out, 'w'))
print('molecules scored', len(out))
