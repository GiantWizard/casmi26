"""Simulate MS/MS spectra for pool structures with the MassSpecGym GLACIER (casmi26-glacier package, gl_runner's
tested featurise/predict/post path) and write them as FPNet training spectra.

usage: sim_dump.py PIDS.npy OUT_PREFIX [--ces 20,40] [--adduct [M+H]+] [--device cpu|mps|cuda] [--bs 32] [--limit N]
writes OUT_PREFIX_{off,mz,it}.npy and OUT_PREFIX_meta.parquet (pid, ce, adduct), appending-safe per chunk.
"""
import argparse, os, sys, time, types
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('pids'); ap.add_argument('out')
ap.add_argument('--pkg', default='/Users/b/casmi26/inputs/glacier')
ap.add_argument('--pool', default='/Users/b/casmi26/inputs/v2-pool/pool_meta.parquet')
ap.add_argument('--ces', default='20,40')
ap.add_argument('--adduct', default='[M+H]+')
ap.add_argument('--instr', default='Orbitrap')
ap.add_argument('--device', default='cpu')
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--chunk', type=int, default=2000)
ap.add_argument('--on-chunk', default='', help='shell command run after each written chunk (e.g. an upload)')
a = ap.parse_args()
sys.path.insert(0, a.pkg)
import gl_runner as G  # noqa: E402

log = G.Log() if hasattr(G, 'Log') else (lambda *x: print(*x, flush=True))
G.setup_env(types.SimpleNamespace(pkg=a.pkg, site='none', wheels=None, any_rdkit=True), log)
import torch  # noqa: E402

dev = a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
model = G.load_model(a.pkg, dev, log)
pred = G.Predictor(model, dev, a.bs, log)
pool = pd.read_parquet(a.pool, columns=['smiles'])
pids = np.load(a.pids)
if a.limit:
    pids = pids[:a.limit]
ces = [float(x) for x in a.ces.split(',')]
import glob
done_parts = sorted(glob.glob(a.out + '_part*_meta.parquet'))
done_pids = set()
for mp in done_parts:
    done_pids.update(pd.read_parquet(mp).pid.unique().tolist())
part0 = len(done_parts)
pids = np.array([p for p in pids if int(p) not in done_pids])
print(f'device {dev} | to simulate {len(pids)} structures x {len(ces)} CE', flush=True)
t0 = time.time(); n_ok = 0
for c0 in range(0, len(pids), a.chunk):
    chunk = pids[c0:c0 + a.chunk]
    smis = {}
    for p in chunk:
        s = pool.smiles.values[p]
        ok = G.smiles_ok(s) if hasattr(G, 'smiles_ok') else (True,)
        ok = ok[0] if isinstance(ok, tuple) else ok
        if ok:
            smis[int(p)] = s
    jobs = {(s, a.adduct, a.instr, ce) for s in smis.values() for ce in ces}
    pred.predict(jobs, deadline=time.time() + 1e9)
    rows, mzs, its = [], [], []
    for p, s in smis.items():
        for ce in ces:
            v = pred.cache.get((s, a.adduct, a.instr, ce))
            if v is None or len(v[0]) < 3:
                continue
            rows.append((p, ce, a.adduct)); mzs.append(v[0].astype(np.float32)); its.append(v[1].astype(np.float32))
    pred.cache.clear(); pred.pcache.clear()
    if rows:
        part = f'{a.out}_part{part0 + c0 // a.chunk:05d}'
        lens = np.array([len(m) for m in mzs]); off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        np.save(part + '_off.npy', off); np.save(part + '_mz.npy', np.concatenate(mzs)); np.save(part + '_it.npy', np.concatenate(its))
        pd.DataFrame(rows, columns=['pid', 'ce', 'adduct']).to_parquet(part + '_meta.parquet')
        n_ok += len(rows)
        if a.on_chunk:
            os.system(a.on_chunk)
    el = time.time() - t0
    print(f'{min(c0 + a.chunk, len(pids))}/{len(pids)} structures, {n_ok} spectra, {el:.0f}s, '
          f'{n_ok / max(el, 1e-9):.1f} spectra/s', flush=True)
print('done', n_ok, flush=True)
