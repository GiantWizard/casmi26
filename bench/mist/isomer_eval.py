"""Isomer discrimination per signal: for molecules whose answer is among the fused candidates, rank the answer inside
its same-formula group by each signal (fused order, ICEBERG, GLACIER, MIST, ...) and by z-sum combinations.
Reports group Hit@1 and group MRR over groups with >= 2 members, split by cls (2 = answer in pool, 3 = not).

usage: isomer_eval.py --feats RUN.feats.parquet --solution solution.csv --sig mist=RUN.mist_msg.json [...]
"""
import argparse, itertools, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--feats', required=True)
ap.add_argument('--solution', required=True)
ap.add_argument('--sig', nargs='*', default=[])
ap.add_argument('--combo', nargs='*', default=[], help='e.g. fusedsig+gl+0.5*ice')
a = ap.parse_args()
df = pd.read_parquet(a.feats)
sol = pd.read_csv(a.solution).set_index('molecule_id')
for spec in a.sig:
    n, _, p = spec.partition('=')
    d = json.load(open(p))
    df[n] = [np.nan if (v := (d.get(str(m)) or {}).get(s)) is None else float(v) for m, s in zip(df.molecule_id, df.smiles)]
df['fusedsig'] = -df.fused_rank.astype(float)
truth_mols = df[df.label == 1].molecule_id.unique()
tf = df[df.label == 1].groupby('molecule_id').form.first()
g = df[df.molecule_id.isin(truth_mols)]
g = g[g.form.values == tf.reindex(g.molecule_id).values]
g = g[g.groupby('molecule_id').smiles.transform('size') >= 2]
sigs = ['fusedsig', 'ice', 'gl'] + [s.partition('=')[0] for s in a.sig]


def z(v):
    v = v.astype(float)
    if v.notna().sum() < 2:
        return pd.Series(0.0, index=v.index)
    sd = v.std()
    return ((v - v.mean()) / (sd if sd > 0 else 1)).fillna(0.0)


rows = []
for mid, gg in g.groupby('molecule_id'):
    r = dict(mid=mid, n=len(gg), cls=int(sol.loc[mid, 'cls']) if mid in sol.index else -1)
    zs = {s: z(gg[s]) for s in sigs}
    combos = {s: zs[s] for s in sigs}
    combos['fused+gl'] = zs['fusedsig'] + zs['gl']
    for s in sigs[3:]:
        combos[f'fused+gl+{s}'] = zs['fusedsig'] + zs['gl'] + zs[s]
        combos[f'fused+gl+.5{s}'] = zs['fusedsig'] + zs['gl'] + 0.5 * zs[s]
        combos[f'fused+{s}'] = zs['fusedsig'] + zs[s]
    for c in a.combo:
        tot = 0
        for term in c.split('+'):
            w, _, nm = term.rpartition('*')
            tot = tot + (float(w) if w else 1.0) * zs[nm]
        combos[c] = tot
    lab = gg.label.values == 1
    for name, v in combos.items():
        cov = gg[name].notna().sum() if name in gg else len(gg)
        vv = v.values
        rank = 1 + int(np.sum(vv > vv[lab].max()))
        r[name] = 1.0 / rank
        r[name + '_cov'] = cov
    rows.append(r)
R = pd.DataFrame(rows)
print(f'groups with answer and >= 2 members: {len(R)} (cls2 {int((R.cls == 2).sum())}, cls3 {int((R.cls == 3).sum())}); '
      f'median group size {R.n.median():.0f}')
names = [c for c in R.columns if c not in ('mid', 'n', 'cls') and not c.endswith('_cov')]
base = R['fused+gl']
print(f'{"signal":24s} {"all":>7s} {"cls2":>7s} {"cls3":>7s}  vs fused+gl (all): up/down')
for c in names:
    d = R[c] - base
    print(f'{c:24s} {R[c].mean():7.3f} {R[R.cls == 2][c].mean():7.3f} {R[R.cls == 3][c].mean():7.3f}  '
          f'{d.mean():+.3f} {(d > 1e-9).sum():3d}/{(d < -1e-9).sum():3d}')
