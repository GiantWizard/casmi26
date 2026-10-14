"""Main-channel development harness on the enveda-np-examples panel (conditional proxy, NOT clean CV: the frozen
ranker and feature resources may have seen these molecules).

Scenarios (the engine's own simulation switches):
  lib     hide only the panel's NP-example spectra; spectra of the same molecule from other libraries stay   (class 1)
  nospec  hide every reference spectrum of the molecule; its structure stays in the pool                    (class 2, in pool)
  dbdrop  nospec + remove the structure from the pool: only generation or the PubChem channel can find it   (class 2 PubChem-only / class 3)

usage: python main_tables.py <v4b_dir> <pool_dir> <train.parquet> <pop_dir> <work_dir> <out_dir> <scenario> [limit]
Writes <out_dir>/<scenario>.pkl: per molecule the full ranked candidate table (keys, pids, scores, formulas, flags).
"""
import json, os, pickle, sys, time
import numpy as np
import pandas as pd

V4, POOL, TRAIN, POP, WORK, OUT, SCEN = sys.argv[1:8]
LIMIT = int(sys.argv[8]) if len(sys.argv) > 8 else 0
V4CODE = os.path.join(V4, 'code')
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(WORK.rstrip('/'))))
wd = os.path.basename(WORK.rstrip('/'))
for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy']:
    dst = os.path.join(wd, f)
    if not os.path.exists(dst):
        os.symlink(os.path.join(POOL, f), dst)
sys.path.insert(0, V4CODE)
T0 = time.time()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


from casmi.build import build_spec_cache
if not os.path.exists(os.path.join(wd, 'spec_meta.parquet')):
    build_spec_cache(TRAIN, wd)
sys.path.insert(1, os.path.join(V4CODE, 'fe_v4'))
import torch, lightgbm as lgb
import pyarrow.parquet as pq
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet, chem
import v1engine

MAN = json.load(open(os.path.join(V4, 'MANIFEST.json')))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = Library(wd); P = Pool(wd)
tfp = np.load(os.path.join(wd, 'train_fp_sel.npy'))
bank = fpnet.ModelBank([os.path.join(V4, 'fpnet_0.pt')], device=dev)
_cfg = EngineCfg(generate=True)
for _k, _v in json.loads(os.environ.get('GEN_CFG', '{}')).items():      # ADDED: generation-config overrides
    setattr(_cfg, _k, type(getattr(_cfg, _k))(_v))
print('EngineCfg', _cfg, flush=True)
E = Engine(L, P, tfp, _cfg, bank)
if os.environ.get('GEN_TF'):                       # ADDED: parents restricted to transform-compatible analogs
    from casmi import derive as _derive
    _orig_generate = E.generate
    def _generate_tf(analogs, target, exclude_keys):
        tol = max(E.cfg.gen_tol_da, target * E.cfg.ppm_win * 1e-6)
        ok = [a for a in analogs if a[1] >= E.cfg.gen_min_sim
              and _derive.matching_transforms(target - float(L.struct_mass[a[0]]), tol, 2)]
        return _orig_generate(ok, target, exclude_keys)
    E.generate = _generate_tf
    print('GEN_TF: parents = transform-compatible analogs', flush=True)
V = v1engine.V1FE(E, [os.path.join(V4, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)
RK = pickle.load(open(os.path.join(V4, MAN['ranker']['file']), 'rb'))
RK_FEATS = list(RK['features']); RK_B = [lgb.Booster(model_str=s) for s in RK['boosters']]


def rank_score(X, names):
    pos = {n: i for i, n in enumerate(names)}
    Xs = np.ascontiguousarray(X[:, [pos[f] for f in RK_FEATS]], dtype=np.float32)
    return np.mean([b.predict(Xs, num_threads=4) for b in RK_B], axis=0)


POOL_POP = np.load(os.path.join(POP, 'pool_lsid.npy')).astype(np.float32) + np.load(os.path.join(POP, 'pool_lpmid.npy')).astype(np.float32)
ts = pd.read_parquet(os.path.join(wd, 'train_structs.parquet'))
NPBIT = 9
np_keys = sorted(set(ts.loc[(ts.libmask.astype('int64') & (1 << NPBIT)) != 0, 'key']))
alias = ts[ts.key.isin(np_keys)]
raw2key = dict(zip(alias.inchikey14, alias.key))
sids_of = alias.groupby('key').sid.apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
pool_pid = {k: i for i, k in enumerate(P.key)}
cols = ['ingest_lib', 'inchikey14', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev']
pf = pq.ParquetFile(TRAIN); parts = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=cols).to_pandas()
    parts.append(t[t.ingest_lib == 'enveda-np-examples'])
q = pd.concat(parts, ignore_index=True)
q['key'] = q.inchikey14.map(raw2key)
q['nm'] = chem.neutral_mass(q.precursor_mz.values.astype(np.float64), q.adduct.values)
log('engine ready', f'{time.time()-T0:.0f}s', 'queries', len(q), 'molecules', q.key.nunique(), 'device', dev, 'scenario', SCEN)


import pickle as _pk
from casmi import derive as _derive
W = _pk.load(open(os.path.join(OUT, 'dbdrop_wide.pkl'), 'rb'))
pmeta = pd.read_parquet(os.path.join(wd, 'pool_meta.parquet'), columns=['key', 'train_sid', 'mass'])
fpa = np.load(os.path.join(wd, 'pool_fp.npy'))
POPC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)
cnt = POPC[fpa].sum(1).astype(np.float32)
miss = [k for k, r in W.items() if 'ckeys' in r and k not in set(r['ckeys'])]
G = dict(list(q.groupby('key', sort=False)))
out = []
for key in miss:
    if key not in pool_pid: continue
    i = pool_pid[key]; inter = POPC[np.bitwise_and(fpa, fpa[i])].sum(1).astype(np.float32)
    tan = inter / (cnt + cnt[i] - inter + 1e-9); tan[i] = 0
    rel = [j for j in np.argsort(-tan)[:30] if tan[j] >= 0.6 and pmeta.train_sid.iloc[j] >= 0
           and _derive.matching_transforms(float(pmeta.mass.iloc[i] - pmeta.mass.iloc[j]), 0.004, 2)]
    if not rel: continue
    sub = G[key]; nms = sub.nm.values[np.isfinite(sub.nm.values)]; target = float(np.median(nms)); spectra = []
    for r in sub.itertuples():
        ce = r.collision_energy_ev; ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
        spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                            mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                            ce=float(ce_arr.mean()) if len(ce_arr) else np.nan, ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
    sids = sids_of[key]
    an = E.analogs(spectra[:E.cfg.max_query_spectra], target, np.isin(L.sid, np.array(sids)), sids[0], -1)
    pos = {s_: (r_ + 1, v_) for r_, (s_, v_, _, _) in enumerate(an)}
    best = min([(pos[int(pmeta.train_sid.iloc[j])][0], pos[int(pmeta.train_sid.iloc[j])][1], float(tan[j])) for j in rel if int(pmeta.train_sid.iloc[j]) in pos], default=None)
    out.append(dict(key=key, n_rel=len(rel), best_tan=float(tan[rel[0]]), analog_rank=best[0] if best else None,
                    analog_sim=best[1] if best else None, n_analogs=len(an)))
d = pd.DataFrame(out); d.to_csv(os.path.join(OUT, 'analogcheck.csv'), index=False)
print('missed with spectra-bearing reachable relative:', len(d), '| relative among the 100 analogs:', int(d.analog_rank.notna().sum()))
x = d[d.analog_rank.notna()]
print('analog rank of best relative: quantiles', x.analog_rank.quantile([.25, .5, .75]).tolist(), '| <=20:', int((x.analog_rank <= 20).sum()),
      '| sim quantiles', x.analog_sim.quantile([.25, .5, .75]).round(3).tolist(), '| sim<0.2:', int((x.analog_sim < 0.2).sum()))
