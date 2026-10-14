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
if os.environ.get('GEN_SG'):                       # ADDED: structure-guided parents (predicted fingerprint)
    from casmi import derive as _derive
    from casmi.engine import merge_spectra as _merge
    _SG_K = int(os.environ.get('GEN_SG_K', '30'))
    _DELTAS = sorted(set(round(d, 4) for _, d in _derive._combos(2)))
    _orig_run, _orig_gen = E.run, E.generate
    _SG = {}
    def _query_z(spectra):
        items = []
        for md in (1, -1):
            grp = [x for x in spectra if x['mode'] == md]
            if not grp: continue
            mm, ii = _merge([(x['mz'], x['it']) for x in grp]); prec = float(np.median([x['prec'] for x in grp]))
            pm_, pi_ = fpnet.prep_peaks(mm, ii, prec)
            ces = [x['ce'] for x in grp if x['ce'] is not None and not np.isnan(x['ce'])]
            ads = [x['adduct'] for x in grp]; ad = max(set(ads), key=ads.count)
            items.append(dict(mz=pm_, it=pi_, prec=prec, adduct_ix=chem.adduct_index(ad), ce=float(np.mean(ces)) if ces else 0.0,
                              ce_known=1.0 if ces else 0.0, n_merged=min(len(grp), 8), mode=float(md)))
        z, _ = E.bank.logits_el(items)
        return z.mean(0)
    def _run_sg(spectra, target, *a, **k):
        _SG['z'] = _query_z(spectra[:E.cfg.max_query_spectra]); _SG['drop'] = k.get('drop_pid', -1)
        return _orig_run(spectra, target, *a, **k)
    def _gen_sg(analogs, target, exclude_keys):
        out = _orig_gen(analogs, target, exclude_keys)
        z = _SG.get('z')
        if z is None: return out
        tol = max(E.cfg.gen_tol_da, target * E.cfg.ppm_win * 1e-6)
        cand = []
        for d in _DELTAS:
            m = target - d
            if m < 60: continue
            lo = np.searchsorted(P.mass, m - tol, 'left'); hi = np.searchsorted(P.mass, m + tol, 'right')
            for pid in range(lo, hi):
                if P.train_sid[pid] >= 0 and pid != _SG['drop']: cand.append(pid)
        if not cand: return out
        cand = np.unique(np.array(cand))
        sc = np.empty(len(cand), np.float32)
        for b0 in range(0, len(cand), 2048):               # batched: unpacked fingerprints are 10,226 bits per row
            fb = P.fps(cand[b0:b0 + 2048]).astype(np.float32)
            sc[b0:b0 + 2048] = (fb @ z) / np.sqrt(np.maximum(fb.sum(1), 1))
        top = cand[np.argsort(-sc)[:_SG_K]]
        asim = {int(a_[0]): float(a_[1]) for a_ in analogs}
        par = [(int(P.train_sid[p_]), asim.get(int(P.train_sid[p_]), E.cfg.gen_min_sim), 0.0, False) for p_ in top]
        have = set(exclude_keys) | {g['key'] for g in out}
        extra = _orig_gen(par, target, have)
        return out + extra
    E.run, E.generate = _run_sg, _gen_sg
    print('GEN_SG: structure-guided parents, K =', _SG_K, '| deltas', len(_DELTAS), flush=True)
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

res = {}
groups = list(q.groupby('key', sort=False))
if LIMIT:
    groups = groups[:LIMIT]
if os.environ.get('START'):
    groups = groups[int(os.environ['START']):]
for gi, (key, sub) in enumerate(groups):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    if not len(nms):
        continue
    target = float(np.median(nms))
    spectra = []
    for r in sub.itertuples():
        ce = r.collision_energy_ev
        ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
        spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                            mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                            ce=float(ce_arr.mean()) if len(ce_arr) else np.nan, ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
    sids = sids_of[key]
    in_sid = np.isin(L.sid, np.array(sids))
    if SCEN == 'lib':
        mask = in_sid & (L.lib == NPBIT); sim = dict(exclude=mask, exclude_sid=sids[0], exclude_lib=NPBIT)
    else:
        mask = in_sid; sim = dict(exclude=mask, exclude_sid=sids[0])
        if SCEN == 'dbdrop':
            sim['drop_pid'] = pool_pid.get(key, -1)
    try:
        C, X, F, names, info = V.run(spectra, target, **sim)
        rec = dict(target=target, n_spectra=len(spectra), n_sid=len(sids), lib_max=float(info.get('lib_max', 0.0)), n_cand=len(C.pid),
                   n_other_lib_spectra=int((in_sid & (L.lib != NPBIT)).sum()))
        if len(C.pid):
            score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
            order = np.argsort(-score, kind='stable')
            pid = np.asarray(C.pid)[order]
            rec.update(keys=[C.key[i] for i in order], pid=pid.astype(np.int64), score=score[order].astype(np.float32),
                       formula=[C.formula[i] for i in order], smiles=[C.smiles[i] for i in order[:80]],
                       pop=np.where(pid >= 0, POOL_POP[np.maximum(pid, 0)], 0).astype(np.float32),
                       src=np.where(pid >= 0, P.src[np.maximum(pid, 0)], 0).astype(np.int8),
                       truth_rank=next((j + 1 for j, i in enumerate(order) if C.key[i] == key), None))
            # ADDED (feature dump): full candidate feature matrix in candidate order, for per-model / ranker replays
            rec.update(Xall=np.concatenate([X, F], 1).astype(np.float32), names=list(FEATURES) + list(names),
                       ckeys=list(C.key), cpid=np.asarray(C.pid).astype(np.int64), cform=list(C.formula),
                       raw_score=score.astype(np.float32))
        res[key] = rec
    except Exception as e:
        log('ERROR', key, repr(e)); res[key] = dict(error=repr(e))
    if (gi + 1) % 10 == 0 or gi + 1 == len(groups):
        tr = [r.get('truth_rank') for r in res.values() if 'error' not in r]
        mrr = np.mean([1 / t if t and t <= 25 else 0 for t in tr])
        log(f'{gi+1}/{len(groups)} raw-order MRR@25 so far {mrr:.4f} top1 {np.mean([t == 1 for t in tr]):.3f} {time.time()-T0:.0f}s')
        pickle.dump(res, open(os.path.join(OUT, SCEN + os.environ.get('TAG', '') + '.pkl'), 'wb'))
pickle.dump(res, open(os.path.join(OUT, SCEN + os.environ.get('TAG', '') + '.pkl'), 'wb'))
log('ALL_DONE')
