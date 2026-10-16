"""How much is the spectrum->fingerprint channel worth to engine 1 once the ranker is retrained?

Structure-grouped K-fold CV of a LightGBM lambdarank ranker on leak-free engine rows (rk/gen_rows.py: held-out training
structures simulated as class 1/2/3, fold-safe FPNet bank). Feature sets: safe (no train-memorizing families), safe
without any fingerprint-derived feature, and fingerprint + priors only. MRR@25 per simulated class (class 3 rows have
no truth, so they score 0 and are reported only for completeness).
usage: fp_value.py ROWS_DIR [--folds 4] [--rounds 300]
"""
import argparse, glob, os
import numpy as np, lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument('rows'); ap.add_argument('--folds', type=int, default=4); ap.add_argument('--rounds', type=int, default=300)
ap.add_argument('--sets', default='safe,safe_nofp,fp_only')
ap.add_argument('--nnsim', default='', help='parquet sid,nn_sim: also report class 2 by nearest-neighbour similarity')
a = ap.parse_args()
Xs, ys, gs, cl, names = [], [], [], {}, None
for f in sorted(glob.glob(os.path.join(a.rows, 'rows_*.npz'))):
    z = np.load(f); names = names or [str(n) for n in z['names']]
    Xs.append(z['X'].astype(np.float32)); ys.append(z['y'].astype(np.int32)); gs.append(z['g'])
    cl.update(dict(zip(z['sids'].tolist(), z['cls'].tolist())))
X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
o = np.argsort(g, kind='stable'); X, y, g = X[o], y[o], g[o]
ug, gstart = np.unique(g, return_index=True)
gcls = np.array([cl.get(int(m), -1) for m in ug]) if all(int(m) in cl for m in ug[:5]) else None
if gcls is None:  # g is a molecule index; map via per-file sids order
    gcls = np.zeros(len(ug), int)
    gg = []
    for f in sorted(glob.glob(os.path.join(a.rows, 'rows_*.npz'))):
        z = np.load(f)
        for m, c in zip(np.unique(z['g']), z['cls']):
            gg.append((m, c))
    d = dict(gg); gcls = np.array([d.get(m, -1) for m in ug])
LEAKY = ('model_views__', 'fragnet__')
isfp = lambda n: ('fz' in n) or n.startswith(('fp_', 'el_', 'xs_')) or n in ('agree_lib', 'agree_ap')
prior = {'mass_ppm', 'mass_ppm_rank', 'n_heavy', 'fp_bits', 'n_cand', 'n_query', 'n_pos', 'n_neg', 'target_mass', 'q_npeaks',
         'formula_share', 'n_formula'}
SETS = {'safe': [n for n in names if not n.startswith(LEAKY)],
        'safe_nofp': [n for n in names if not n.startswith(LEAKY) and not isfp(n)],
        'fp_only': [n for n in names if not n.startswith(LEAKY) and (isfp(n) or n in prior)]}
rng = np.random.default_rng(0); fold = rng.integers(0, a.folds, len(ug))
row_fold = fold[np.searchsorted(ug, g)]
P = dict(objective='lambdarank', metric='None', learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1, seed=0)


def gsize(gv):
    _, c = np.unique(gv, return_counts=True); return c


def rr_per_group(s):
    out = np.zeros(len(ug))
    for i, (st, en) in enumerate(zip(gstart, np.r_[gstart[1:], len(g)])):
        yy = y[st:en]
        if yy.any():
            r = int((s[st:en] > s[st:en][yy > 0].max()).sum()) + 1
            out[i] = 1 / r if r <= 25 else 0
    return out


res = {}
for name in a.sets.split(','):
    cols = [names.index(n) for n in SETS[name]]
    s = np.zeros(len(y))
    for k in range(a.folds):
        tr = row_fold != k
        b = lgb.train(P, lgb.Dataset(X[tr][:, cols], y[tr], group=gsize(g[tr])), a.rounds)
        s[~tr] = b.predict(X[~tr][:, cols])
    rr = rr_per_group(s); res[name] = rr
    msg = ' '.join(f'cls{c} {rr[gcls == c].mean():.4f} (n {int((gcls == c).sum())})' for c in (1, 2, 3))
    print(f'{name:10s} features {len(cols):3d} | all {rr.mean():.4f} | {msg}', flush=True)
if a.nnsim:
    import pandas as pd
    nn = pd.read_parquet(a.nnsim).set_index('sid').nn_sim.reindex(ug).values
    nd = pd.read_parquet(a.nnsim).set_index('sid').reindex(ug)
    for lo, hi in ((-9, -1), (-1, 0), (0, 1), (1, 9)):
        m = (gcls == 2) & (nd.np.values > lo) & (nd.np.values <= hi)
        print(f'  class 2, NP score ({lo},{hi}] n {int(m.sum()):4d} | ' + ' '.join(f'{k} {v[m].mean():.4f}' for k, v in res.items()))
    for flag in ('e180_only',):
        for val in (True, False):
            m = (gcls == 2) & (nd[flag].values == val)
            print(f'  class 2, {flag}={val} n {int(m.sum()):4d} | ' + ' '.join(f'{k} {v[m].mean():.4f}' for k, v in res.items()))
    for lo, hi in ((0, .5), (.5, .6), (.6, .7), (.7, .85), (.85, 1.01)):
        m = (gcls == 2) & (nn > lo) & (nn <= hi)
        print(f'  class 2, nn_sim ({lo:.2f},{hi:.2f}] n {int(m.sum()):4d} | ' + ' '.join(f'{k} {v[m].mean():.4f}' for k, v in res.items()))
if 'safe' in res and 'safe_nofp' in res:
    d = res['safe'] - res['safe_nofp']; m2 = gcls == 2
    bs = [d[m2][rng.integers(0, m2.sum(), m2.sum())].mean() for _ in range(1000)]
    print(f'value of fingerprint channel on class 2: {d[m2].mean():+.4f} (95% CI {np.percentile(bs, 2.5):+.4f} .. {np.percentile(bs, 97.5):+.4f})')
