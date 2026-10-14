"""Can the existing fold-safe FPNet retrieve a structure from GLACIER's simulated spectrum of it? Window MRR@25 for
the pilot answers' simulated spectra, next to the same answers' real spectra (from res_pilot_fold0 ranks)."""
import glob, os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, '/Users/b/casmi26/inputs/v4b/code'); os.chdir('/Users/b/casmi26/fp/run')
from casmi import fpnet
pool = pd.read_parquet('work/pool_meta.parquet', columns=['mass']); pmass = pool.mass.values
PFP = torch.from_numpy(np.load('work/pool_fp.npy')).to('mps'); nbits = int(np.load('work/fp_bits.npy').shape[0])
m = fpnet.load_model('/Users/b/casmi26/inputs/foldsafe/fpnet_fold0.pt', 'mps')
ans = set(np.load('/Users/b/casmi26/fp/pilot_answer_pids.npy').tolist())
rr = []; rnd = []
rng = np.random.default_rng(0)
for mp in sorted(glob.glob('/Users/b/casmi26/fp/sim/pilot_part*_meta.parquet')):
    pre = mp[:-13]; meta = pd.read_parquet(mp); off = np.load(pre + '_off.npy'); mz = np.load(pre + '_mz.npy'); it = np.load(pre + '_it.npy')
    for j, (p, ce) in enumerate(zip(meta.pid.values, meta.ce.values)):
        tgt = rr if p in ans else (rnd if rng.random() < 0.02 else None)
        if tgt is None: continue
        pm, pi = fpnet.prep_peaks(mz[off[j]:off[j+1]].astype(float), it[off[j]:off[j+1]].astype(float), pmass[p] + 1.007276)
        inp = fpnet.collate([dict(mz=pm, it=pi, prec=pmass[p] + 1.007276, adduct_ix=0, ce=float(ce), ce_known=1.0, n_merged=1, mode=1.0)], 'mps')
        with torch.no_grad(): z = m(*inp)[0].float()[0]
        lo = np.searchsorted(pmass, pmass[p] * (1 - 1e-5)); hi = np.searchsorted(pmass, pmass[p] * (1 + 1e-5), 'right')
        sc = fpnet.unpack_gpu(PFP[lo:hi], nbits).float() @ z
        r = int((sc > sc[p - lo]).sum().item()) + 1; tgt.append(r)
for name, r in (('pilot answers (sim spectrum)', rr), ('random pool structures (sim spectrum)', rnd)):
    r = np.array(r); print(f'{name}: n {len(r)} MRR@25 {np.where(r <= 25, 1 / r, 0).mean():.3f} top1 {(r == 1).mean():.3f}')
