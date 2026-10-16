"""Pipeline-level (engine 1) comparison of FP banks on the NP-like held-out harness.

Each ROWS_DIR holds rk/gen_rows.py rows of the SAME queries (same sids list + seed -> same simulated classes) built with a
different FP bank. For each bank a LightGBM lambdarank ranker is trained by structure-grouped K-fold CV (identical folds
across banks, safe features: no train-memorizing families), and molecules are scored MRR@25. Reports per class, the FP
channel's own top-1 (fz_rank), and a paired bootstrap of each bank minus the first on class-2 molecules.
usage: compare_banks.py NAME=ROWS_DIR [NAME=ROWS_DIR ...] [--folds 4] [--rounds 300] [--seeds 2]
"""
import argparse, glob, os
import numpy as np, lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument('sets', nargs='+'); ap.add_argument('--folds', type=int, default=4)
ap.add_argument('--rounds', type=int, default=300); ap.add_argument('--seeds', type=int, default=2)
a = ap.parse_args()
LEAKY = ('model_views__', 'fragnet__')
P = dict(objective='lambdarank', metric='None', learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1)


def load(d):
    Xs, ys, gs, cl, names = [], [], [], {}, None
    for f in sorted(glob.glob(os.path.join(d, 'rows_*.npz'))):
        z = np.load(f); names = names or [str(n) for n in z['names']]
        Xs.append(z['X'].astype(np.float32)); ys.append(z['y'].astype(np.int32)); gs.append(z['g'])
        cl.update(dict(zip(z['sids'].tolist(), z['cls'].tolist())))
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    o = np.argsort(g, kind='stable')
    return X[o], y[o], g[o], cl, names


def rr_groups(s, y, g):
    ug, st = np.unique(g, return_index=True); en = np.r_[st[1:], len(g)]
    out = np.zeros(len(ug))
    for i, (a0, b0) in enumerate(zip(st, en)):
        yy = y[a0:b0]
        if yy.any():
            r = int((s[a0:b0] > s[a0:b0][yy > 0].max()).sum()) + 1
            out[i] = 1 / r if r <= 25 else 0
    return ug, out


R = {}
for item in a.sets:
    name, d = item.split('=', 1)
    X, y, g, cl, names = load(d)
    cols = [i for i, n in enumerate(names) if not n.startswith(LEAKY)]
    ug = np.unique(g)
    fold = (ug * 2654435761 % 2**32) % a.folds          # deterministic per sid: identical folds across banks
    rf = fold[np.searchsorted(ug, g)]
    s = np.zeros(len(y))
    for k in range(a.folds):
        tr = rf != k
        _, gsz = np.unique(g[tr], return_counts=True)
        bs = [lgb.train(dict(P, seed=sd), lgb.Dataset(X[tr][:, cols], y[tr], group=gsz), a.rounds) for sd in range(a.seeds)]
        s[~tr] = np.mean([b.predict(X[~tr][:, cols]) for b in bs], 0)
    ug, rr = rr_groups(s, y, g)
    _, fz = rr_groups(X[:, names.index('fz')], y, g)
    c = np.array([cl.get(int(m), -1) for m in ug])
    R[name] = dict(ug=ug, rr=rr, fz=fz, c=c)
    print(f'{name:8s} n {len(ug)} | all {rr.mean():.4f} | ' + ' '.join(f'cls{k} {rr[c == k].mean():.4f}' for k in (1, 2, 3)) +
          f' | FP-alone class-2 MRR {fz[c == 2].mean():.4f} top-1 {(fz[c == 2] == 1).mean():.3f}', flush=True)
names_ = list(R); base = R[names_[0]]
rng = np.random.default_rng(0)
for n in names_[1:]:
    o = R[n]; common = np.intersect1d(base['ug'], o['ug'])
    bi = np.searchsorted(base['ug'], common); oi = np.searchsorted(o['ug'], common)
    m = base['c'][bi] == 2
    d = (o['rr'][oi] - base['rr'][bi])[m]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f'{n} - {names_[0]} on class 2 (n {len(d)}): {d.mean():+.4f}  95% CI {np.percentile(bs, 2.5):+.4f} .. {np.percentile(bs, 97.5):+.4f} '
          f'| fixed {int((d > 0).sum())} broke {int((d < 0).sum())}')
