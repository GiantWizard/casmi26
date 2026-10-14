"""Fingerprint-retrieval channel: for each test molecule, rank the pool structures in its +-10 ppm neutral-mass
window by an FPNet checkpoint's score and keep the top K, as a third list for the final fusion.

Spectra of one molecule are merged per polarity (as in Gluzdov's evaluation); with both polarities the per-window
z-scores are summed. Output: {molecule_id: {'smiles': [...], 'keys': [...], 'score': [...]}} like eng_lists.json.

usage: fp_channel.py TEST.parquet OUT.json --ckpt a.pt[,b.pt] [--k 50] [--device mps]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('test'); ap.add_argument('out')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--code', default='/Users/b/casmi26/inputs/v4b/code')
ap.add_argument('--work', default='/Users/b/casmi26/fp/run/work')
ap.add_argument('--k', type=int, default=50)
ap.add_argument('--device', default='mps')
a = ap.parse_args()
sys.path.insert(0, a.code)
import torch  # noqa: E402
from casmi import fpnet  # noqa: E402
from casmi.spectra import merge_spectra  # noqa: E402

ADD = {'[M+H]+': (0, 1.007276), '[M-H]-': (6, -1.007276), '[M+Na]+': (4, 22.989221), '[M+CH2O2-H]-': (8, 44.998203),
       '[M+NH4]+': (1, 18.033826), '[M+Cl]-': (9, 34.969401), '[M-H2O+H]+': (2, -17.003288), '[M+H-H2O]+': (2, -17.003288),
       '[M+K]+': (5, 38.963158), '[M]+': (12, -0.000549), '[M-H2O-H]-': (7, -19.01784)}
pool = pd.read_parquet(os.path.join(a.work, 'pool_meta.parquet'), columns=['key', 'smiles', 'mass'])
pmass = pool.mass.values.astype(np.float64)
nbits = int(np.load(os.path.join(a.work, 'fp_bits.npy')).shape[0])
PFP = torch.from_numpy(np.load(os.path.join(a.work, 'pool_fp.npy'))).to(a.device)
models = [fpnet.load_model(p, a.device) for p in a.ckpt.split(',')]
te = pd.read_parquet(a.test)
out = {}
for mid, g in te.groupby('molecule_id'):
    g = g[g.adduct.isin(ADD)]
    if not len(g):
        continue
    neutral = np.median([p - ADD[ad][1] for p, ad in zip(g.precursor_mz, g.adduct)])
    lo = np.searchsorted(pmass, neutral * (1 - 1e-5)); hi = np.searchsorted(pmass, neutral * (1 + 1e-5), 'right')
    if hi <= lo:
        continue
    fpw = fpnet.unpack_gpu(PFP[lo:hi], nbits).float()
    total = torch.zeros(hi - lo, device=a.device)
    for mode, gm in g.groupby('ionization_mode'):
        specs = [(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float)) for r in gm.itertuples()]
        mz, it = merge_spectra(specs) if len(specs) > 1 else specs[0]
        r0 = gm.iloc[0]
        ces = [float(np.mean(c)) for c in gm.collision_energy_ev if c is not None and len(c)]
        ce = float(np.mean(ces)) if ces else np.nan
        pm, pi = fpnet.prep_peaks(np.asarray(mz, float), np.asarray(it, float), float(r0.precursor_mz))
        item = dict(mz=pm, it=pi, prec=float(r0.precursor_mz), adduct_ix=ADD[r0.adduct][0], ce=0.0 if np.isnan(ce) else ce,
                    ce_known=0.0 if np.isnan(ce) else 1.0, n_merged=min(len(specs), 8), mode=1.0 if mode == 'positive' else -1.0)
        with torch.no_grad():
            z = torch.stack([m(*fpnet.collate([item], a.device))[0].float()[0] for m in models]).mean(0)
        sc = fpw @ z
        total += (sc - sc.mean()) / (sc.std() + 1e-6) if len(sc) > 1 else sc
    o = torch.argsort(total, descending=True)[:a.k].cpu().numpy()
    out[str(mid)] = dict(smiles=[pool.smiles.values[lo + i] for i in o], keys=[str(pool.key.values[lo + i]) for i in o],
                         score=[float(total[i]) for i in o])
json.dump(out, open(a.out, 'w'))
print('molecules', len(out), 'of', te.molecule_id.nunique())
