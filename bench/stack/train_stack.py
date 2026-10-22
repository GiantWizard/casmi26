"""Stacked reranker vs v465's hand-tuned fusion on the same simulated held-out molecules.

LightGBM lambdarank over build_rows.py rows, 5-fold cross-validation grouped by molecule (each molecule is predicted by a
model that never saw it), repeated over seeds. Baseline: v465's fused order (fus_rank, then its overflow), i.e. what the
pipeline would have submitted. Reports MRR@25 overall / by class / Enveda panel, with a paired bootstrap of the gain.
usage: train_stack.py ROWS.parquet [--seeds 3] [--save model.pkl]
"""
import argparse, pickle
import numpy as np, pandas as pd, lightgbm as lgb

ap = argparse.ArgumentParser(); ap.add_argument('rows'); ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--save', default=''); ap.add_argument('--rounds', type=int, default=400)
a = ap.parse_args()
D = pd.read_parquet(a.rows).sort_values('mid', kind='stable').reset_index(drop=True)
FEATS = ['e1_rank', 'e1pp_rank', 'pre_rank', 'e2_rank', 'pc_rank', 'fus_rank', 'e1_score', 'e1_z', 'e1_gap', 'e1_gen', 'pop', 'pop_zm',
         'e2_blend', 'e2_z', 'e2_gap', 'e2_pv', 'e2_ours', 'pc_S', 'pc_z', 'pc_gap', 'ice', 'gl', 'ice_zm', 'ice_zf', 'ice_rf',
         'gl_zm', 'gl_zf', 'gl_rf', 'n_src', 'form_n', 'form_best_fus', 'heavy', 'lib_max', 'pc_delta', 'pc_top_pop', 'n_cand']
P = dict(objective='lambdarank', metric='None', learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=25, verbose=-1, num_threads=6)


def rr(score, d):
    out = {}
    for mid, g in d.assign(s=score).groupby('mid', sort=False):
        o = g.sort_values('s', ascending=False, kind='stable').y.values[:25]
        h = np.flatnonzero(o)
        out[mid] = 1.0 / (h[0] + 1) if len(h) else 0.0
    return pd.Series(out)


mids = D.mid.unique()
base = rr(-D.fus_rank.fillna(999).values, D)
preds = []
for seed in range(a.seeds):
    rng = np.random.default_rng(seed); fold = dict(zip(mids, rng.integers(0, 5, len(mids))))
    f = D.mid.map(fold).values; pred = np.zeros(len(D))
    for k in range(5):
        tr = f != k
        dt = D[tr]; sizes = dt.groupby('mid', sort=False).size().values
        m = lgb.train(dict(P, seed=seed), lgb.Dataset(dt[FEATS], dt.y, group=sizes), a.rounds)
        pred[~tr] = m.predict(D.loc[~tr, FEATS])
    preds.append(rr(pred, D))
st = pd.concat(preds, axis=1).mean(1)
meta = D.groupby('mid').agg(cls=('cls', 'first'), panel=('panel', 'first'), hit=('y', 'max'))
gain = (st - base).reindex(meta.index)
bs = np.random.default_rng(0)
boot = [gain.values[bs.integers(0, len(gain), len(gain))].mean() for _ in range(2000)]
print(f'molecules {len(meta)} | truth among candidates {meta.hit.mean():.3f}')
print(f'v465 fusion MRR@25 {base.mean():.4f} | stacked {st.mean():.4f} | gain {gain.mean():+.4f} '
      f'(95% CI {np.percentile(boot, 2.5):+.4f} .. {np.percentile(boot, 97.5):+.4f})')
for name, sel in [('class 1', meta.cls == 1), ('class 2', meta.cls == 2), ('Enveda panel', meta.panel), ('class 2 non-panel', (meta.cls == 2) & ~meta.panel)]:
    i = meta.index[sel]
    print(f'  {name:18s} n={len(i):4d}  v465 {base[i].mean():.4f}  stacked {st[i].mean():.4f}  gain {gain[i].mean():+.4f}')
print(f'top-1: v465 {(base == 1).mean():.3f}  stacked {(st == 1).mean():.3f}')
if a.save:
    sizes = D.groupby('mid', sort=False).size().values
    ms = [lgb.train(dict(P, seed=s), lgb.Dataset(D[FEATS], D.y, group=sizes), a.rounds) for s in range(a.seeds)]
    pickle.dump(dict(features=FEATS, boosters=[m.model_to_string() for m in ms]), open(a.save, 'wb'))
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in ms], 0), FEATS).sort_values(ascending=False)
    print('top features:', ', '.join(f'{k} {v / imp.sum():.2f}' for k, v in imp.head(10).items()))
