"""Cross-set test of the learned same-formula rerank with different feature sets.

Train on one set's rerank_features rows, apply (slots mode, like jc-lr2) to the other set, write submissions; score
them with mrr_eval.py against the replayed baseline. Feature sets:
  lr2   : jc-lr2's features (popularity and PubChem-channel ranks included)
  spec  : spectrum evidence and list position only (no popularity, no PubChem-channel features)
  spec+X: spec plus extra simulator columns merged from score dicts (e.g. mist)

usage: lr_xset.py --train A.feats.parquet [...] --test B.feats.parquet --out DIR [--extra mist=A.json,B.json]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, '/Users/b/casmi26-repo/bench/obscure_np')
import lr_core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--train', nargs='+', required=True)
ap.add_argument('--test', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--extra', nargs='*', default=[], help='name=train1.json[,train2.json...]:test.json score dicts')
ap.add_argument('--rounds', type=int, default=150)
ap.add_argument('--seeds', type=int, default=5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

SPEC = ['fused', 'fused_rank', 'r_main', 'r_eng', 'agree', 'n_cand',
        'ice', 'gl', 'glft', 'ice_cov', 'gl_cov', 'glft_cov', 'ice_minus_max', 'gl_minus_max', 'glft_minus_max',
        'ice_z', 'gl_z', 'glft_z', 'ice_grank', 'gl_grank', 'glft_grank', 'grp_size', 'grp_rank',
        'fused_minus_max', 'is_top_group']
PARAMS = dict(objective='lambdarank', metric='map', eval_at=[1], learning_rate=0.05, num_leaves=7,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, lambdarank_truncation_level=10)


def add_extra(df, name, d):
    df[name] = [np.nan if (v := (d.get(str(m)) or {}).get(s)) is None else float(v)
                for m, s in zip(df.molecule_id, df.smiles)]
    g = df.groupby(['molecule_id', 'form'])
    mx, mu, sd = g[name].transform('max'), g[name].transform('mean'), g[name].transform('std')
    df[f'{name}_cov'] = df[name].notna().astype(int)
    df[f'{name}_minus_max'] = df[name] - mx
    df[f'{name}_z'] = (df[name] - mu) / sd.replace(0, np.nan)
    df[f'{name}_grank'] = g[name].rank(ascending=False, method='min')
    return [name, f'{name}_cov', f'{name}_minus_max', f'{name}_z', f'{name}_grank']


trs = [pd.read_parquet(p) for p in a.train]
te = pd.read_parquet(a.test)
extra_cols = []
for spec in a.extra:
    name, _, paths = spec.partition('=')
    tr_paths, _, te_path = paths.rpartition(':')
    for df, p in zip(trs, tr_paths.split(',')):
        add_extra(df, name, json.load(open(p)))
    extra_cols += add_extra(te, name, json.load(open(te_path)))
tr = pd.concat(trs, ignore_index=True)
tr['molecule_id'] = [f'{i}_{m}' for i, df in enumerate(trs) for m in df.molecule_id]


def fit(feats, seed):
    d = tr[tr.groupby('molecule_id').label.transform('max') == 1].sort_values(['molecule_id', 'fused_rank'])
    ds = lgb.Dataset(d[feats], d.label, group=d.groupby('molecule_id', sort=False).size().values)
    return lgb.train(dict(PARAMS, seed=seed), ds, a.rounds)


sets = {'lr2': list(lr_core.FEATS), 'spec': SPEC}
if extra_cols:
    sets['spec+' + '+'.join(s.split('=')[0] for s in a.extra)] = SPEC + extra_cols
te = te.sort_values(['molecule_id', 'fused_rank'])
for name, feats in sets.items():
    ms = [fit(feats, s) for s in range(a.seeds)]
    sc = np.mean([m.predict(te[feats]) for m in ms], axis=0)
    te['_m'] = sc
    out = [(mid, ';'.join(lr_core.slot_order(g, g['_m'].values)[:25])) for mid, g in te.groupby('molecule_id', sort=False)]
    pd.DataFrame(out, columns=['molecule_id', 'smiles']).to_csv(os.path.join(a.out, f'lr_{name}.csv'), index=False)
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in ms], axis=0), index=feats).sort_values()[::-1]
    print(name, 'top features:', ', '.join(f'{k} {v / imp.sum():.2f}' for k, v in imp.head(6).items()), flush=True)
# fused order without any rerank, for reference
out = [(mid, ';'.join(g.smiles.tolist()[:25])) for mid, g in te.groupby('molecule_id', sort=False)]
pd.DataFrame(out, columns=['molecule_id', 'smiles']).to_csv(os.path.join(a.out, 'fused_only.csv'), index=False)
print('wrote', a.out)
