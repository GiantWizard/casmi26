"""Class-3 detector check: can the pipeline's own run outputs flag molecules whose final list misses the answer?

Label miss = answer not in the c2fix top 25. Features per molecule from the eval run (engine list, GLACIER scores,
candidate audit). AUC per feature on each set; logistic model fit on A, tested on B (and the reverse).
usage: detect.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/b/casmi26-repo/bench/obscure_np')
os.environ.setdefault('ONP_KEY_CACHE', '/Users/b/casmi26/mist/keycache')
from common import keys_parallel, scorer_key
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = '/Users/b/casmi26/mist/runs'
SETS = {'A': ('jc-onp-eval-d2', '/Users/b/casmi26/fp/net_calib/sol_a.csv'),
        'B': ('jc-onp-eval-b-d0', '/Users/b/casmi26/fp/net_calib/sol_b.csv')}
aa = Chem.MolFromSmarts('[NX3][CX4]C(=O)[OX2H1,OX1-]')


def zgap(v):
    v = np.sort(np.asarray(v, float))[::-1]
    if len(v) < 2:
        return 0.0, 0.0
    sd = v.std() or 1.0
    return float(v[0]), float((v[0] - v[1]) / sd)


def build(name):
    run, solp = SETS[name]
    d = f'{R}/{run}'
    sol = pd.read_csv(solp).drop_duplicates('molecule_id')
    eng = json.load(open(f'{d}/eng_lists.json'))
    gl = json.load(open(f'{d}/gl_work/gl_out.json'))
    aud = pd.read_csv(f'{d}/candidate_audit.csv').set_index('molecule_id')
    fin = {m: [x for x in str(v).split(';') if x and x != 'CCO'] for m, v in pd.read_csv(f'{R}/{run}.rp/rp_c2fix.csv').values}
    K = keys_parallel(list({s for l in fin.values() for s in l[:25]}), scorer_key)
    rows = []
    for m, smi, key in zip(sol.molecule_id, sol.smiles, sol.scorer_key):
        e = eng.get(m, {}); g = gl.get(m, {}); f = fin.get(m, [])
        fk = [K.get(s) for s in f[:25]]
        es = e.get('score') or e.get('scores') or []
        e1, egap = zgap(es)
        g1, ggap = zgap(list(g.values()))
        ek = e.get('keys', [])
        mol = Chem.MolFromSmiles(smi)
        rows.append(dict(
            mid=m, miss=int(key not in fk),
            conj=bool(mol) and mol.HasSubstructMatch(aa),
            eng_n=len(ek), eng_top=e1, eng_gap=egap,
            gl_n=len(g), gl_top=g1, gl_gap=ggap,
            orig_n=float(aud.original_count.get(m, 0)), added=float(aud.ranked_added.get(m, 0)),
            agree=int(bool(ek) and bool(fk) and ek[0] == fk[0]),
            fin_n=len(f)))
    return pd.DataFrame(rows)


FEATS = ['eng_n', 'eng_top', 'eng_gap', 'gl_n', 'gl_top', 'gl_gap', 'orig_n', 'added', 'agree', 'fin_n']
D = {n: build(n) for n in SETS}
for n, df in D.items():
    print(f'\nset {n}: {len(df)} molecules, miss rate {df.miss.mean():.2f} (non-conj {df[~df.conj].miss.mean():.2f}, n {int((~df.conj).sum())})')
    for sub, m in (('all', df.index == df.index), ('non-conj', ~df.conj)):
        x = df[m]
        auc = {f: roc_auc_score(x.miss, x[f]) for f in FEATS if x[f].nunique() > 1}
        print(f'  {sub:9s} AUC (0.5 = useless, <0.5 = inverse signal): ' + ' '.join(f'{f} {v:.2f}' for f, v in auc.items()))


def transfer(tr, te):
    a, b = D[tr], D[te]
    clf = LogisticRegression(max_iter=2000).fit((a[FEATS] - a[FEATS].mean()) / (a[FEATS].std() + 1e-9), a.miss)
    p = clf.predict_proba((b[FEATS] - a[FEATS].mean()) / (a[FEATS].std() + 1e-9))[:, 1]
    print(f'\nfit {tr} -> test {te}: AUC all {roc_auc_score(b.miss, p):.3f}, non-conj {roc_auc_score(b.miss[~b.conj], p[~b.conj]):.3f}')
    for q in (0.1, 0.2, 0.3):
        top = p >= np.quantile(p, 1 - q)
        print(f'  flag top {int(q * 100)}%: {int(top.sum())} rows, precision {b.miss[top].mean():.2f} (base {b.miss.mean():.2f}), '
              f'hits wrongly flagged {int((top & (b.miss == 0)).sum())}')


transfer('A', 'B'); transfer('B', 'A')


print('\nnon-conj only: fit on the other set\'s non-conj rows')
for tr, te in (('A', 'B'), ('B', 'A')):
    a, b = D[tr][~D[tr].conj], D[te][~D[te].conj]
    mu, sd = a[FEATS].mean(), a[FEATS].std() + 1e-9
    p = LogisticRegression(max_iter=2000).fit((a[FEATS] - mu) / sd, a.miss).predict_proba((b[FEATS] - mu) / sd)[:, 1]
    print(f'  {tr}->{te}: n {len(b)}, AUC {roc_auc_score(b.miss, p):.3f}, base miss {b.miss.mean():.2f}')
    for q in (0.1, 0.2, 0.3):
        top = p >= np.quantile(p, 1 - q)
        print(f'    flag top {int(q * 100)}%: {int(top.sum())} rows, precision {b.miss[top].mean():.2f}, hits wrongly flagged {int((top & (b.miss == 0)).sum())}')
