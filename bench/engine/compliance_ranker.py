"""Cost of a compliant engine-1 ranker. Train OUR LightGBM lambdarank on our own leak-free rows (rk/gen_rows.py, fold0
bank: drug-like hold rows + the NP train half), safe features only (no model_views / fragnet, whose weights have no
published training code), and compare it with the public ranker_v4b on the NP evaluation half (same rows, same bank).
usage: compliance_ranker.py [--out ranker_ours.pkl]
"""
import argparse, glob, json, os, pickle, sys
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/Users/b/casmi26/rk/ranker_ours.pkl')
ap.add_argument('--rounds', type=int, default=600); ap.add_argument('--seeds', type=int, default=4)
ap.add_argument('--np-weight', type=float, default=3.0)
a = ap.parse_args()
R = '/Users/b/casmi26/rk'
LEAKY = ('model_views__', 'fragnet__')


def load(d, keep=None, drop=None):
    Xs, ys, gs, cl, names = [], [], [], {}, None
    for f in sorted(glob.glob(os.path.join(d, 'rows_*.npz'))):
        z = np.load(f); names = names or [str(n) for n in z['names']]
        m = np.ones(len(z['g']), bool)
        if keep is not None: m &= np.isin(z['g'], keep)
        if drop is not None: m &= ~np.isin(z['g'], drop)
        Xs.append(z['X'][m].astype(np.float32)); ys.append(z['y'][m].astype(np.int32)); gs.append(z['g'][m])
        cl.update(dict(zip(z['sids'].tolist(), z['cls'].tolist())))
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    o = np.argsort(g, kind='stable'); return X[o], y[o], g[o], cl, names


ev = pd.read_parquet(f'{R}/np1_evalhalf.parquet').sid.values
tr_np = pd.read_parquet(f'{R}/np1_trainhalf.parquet').sid.values
Xn, yn, gn, cln, names = load(f'{R}/np1_fold0', keep=tr_np)
Xd, yd, gd, cld, nd = load(f'{R}/colab_rows', drop=np.concatenate([ev, tr_np]))
assert nd == names
Xe, ye, ge, cle, _ = load(f'{R}/np1_fold0', keep=ev)
feats = [n for n in names if not n.startswith(LEAKY)]
ci = [names.index(f) for f in feats]
X = np.concatenate([Xn, Xd])[:, ci]; y = np.concatenate([yn, yd]); g = np.concatenate([gn, gd + 10**9])
w_grp = {}
o = np.argsort(g, kind='stable'); X, y, g = X[o], y[o], g[o]
ug, cnt = np.unique(g, return_counts=True)
gw = np.where(ug < 10**9, a.np_weight, 1.0)                    # NP queries weighted up
P = dict(objective='lambdarank', metric='None', learning_rate=0.03, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1)
ds = lgb.Dataset(X, y, group=cnt, weight=np.repeat(gw, cnt))
boosters = [lgb.train(dict(P, seed=s, bagging_seed=s, feature_fraction_seed=s), ds, a.rounds) for s in range(a.seeds)]
pickle.dump(dict(features=feats, params=P, rounds=a.rounds, boosters=[b.model_to_string() for b in boosters],
                 trained_on=dict(np_train_half=int(len(np.unique(gn))), druglike_hold=int(len(np.unique(gd))))), open(a.out, 'wb'))
print(f'trained on {len(np.unique(gn))} NP + {len(np.unique(gd))} drug-like hold molecules, {len(feats)} features -> {a.out}')
pub = pickle.load(open('/Users/b/casmi26/inputs/v4b/' + json.load(open('/Users/b/casmi26/inputs/v4b/MANIFEST.json'))['ranker']['file'], 'rb'))
PB = [lgb.Booster(model_str=s) for s in pub['boosters']]
s_pub = np.mean([b.predict(Xe[:, [names.index(f) for f in pub['features']]]) for b in PB], 0)
s_our = np.mean([b.predict(Xe[:, ci]) for b in boosters], 0)


def rr(s):
    out = {}
    for sid in np.unique(ge):
        k = ge == sid; yy = ye[k]
        r = int((s[k] > s[k][yy > 0].max()).sum()) + 1 if yy.any() else 999
        out[int(sid)] = 1 / r if r <= 25 else 0.0
    return out


A, B = rr(s_pub), rr(s_our)
cls = np.array([cle.get(s, -1) for s in A]); a_ = np.array(list(A.values())); b_ = np.array([B[s] for s in A])
for c in (1, 2, 3):
    m = cls == c
    print(f'class {c} (n {m.sum()}): public ranker_v4b {a_[m].mean():.4f} | ours {b_[m].mean():.4f} | diff {b_[m].mean() - a_[m].mean():+.4f}')
m = cls == 2; d = (b_ - a_)[m]; rng = np.random.default_rng(0)
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
print(f'class 2 ours - public: {d.mean():+.4f}  95% CI {np.percentile(bs, 2.5):+.4f} .. {np.percentile(bs, 97.5):+.4f}')
