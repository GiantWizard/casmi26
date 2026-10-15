"""Why do we miss on the LB-calibrated slice (non-conjugate molecules whose answer is in the pool)?

For each molecule: answer rank in the final list, in engine 1's main list (v4), in engine 2's list, in the 80-long
fused list; for misses with the answer present: is #1 a same-formula isomer, is #1 more popular than the answer, did
GLACIER (and ICEBERG) score the answer above #1, which engine had the answer higher.
usage: miss_diag.py RUN SOLUTION FINAL.csv FUSED_FULL.csv [more RUN SOLUTION FINAL FULL ...]
"""
import json, os, sys
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
sys.path.insert(0, '/Users/b/casmi26-repo/bench/obscure_np')
os.environ.setdefault('ONP_KEY_CACHE', '/Users/b/casmi26/mist/keycache')
from common import keys_parallel, scorer_key

aa = Chem.MolFromSmarts('[NX3][CX4]C(=O)[OX2H1,OX1-]')
pop = pd.read_parquet('/Users/b/casmi26/mist/runs/pool_pop.parquet').groupby('scorer_key').popularity.max().to_dict()
rows = []
args = sys.argv[1:]
for run, solp, finp, fullp in zip(args[0::4], args[1::4], args[2::4], args[3::4]):
    sol = pd.read_csv(solp)
    lists = lambda p: {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(p).values}
    fin, full, v4 = lists(finp), lists(fullp), lists(os.path.join(run, 'v4_submission.csv'))
    eng = {m: (e.get('smiles') or []) for m, e in json.load(open(os.path.join(run, 'eng_lists.json'))).items()}
    gl = json.load(open(os.path.join(run, 'gl_work/gl_out.json')))
    ice = json.load(open(os.path.join(run, 'ice_work/ice_out.json')))
    allsmi = {s for d in (fin, full, v4, eng) for l in d.values() for s in l[:80]} | set(sol.smiles)
    K = keys_parallel(list(allsmi), scorer_key)
    for r in sol.itertuples():
        m = Chem.MolFromSmiles(r.smiles)
        if r.cls != 2 or m is None or m.HasSubstructMatch(aa):
            continue
        t = K.get(r.smiles); mid = str(r.molecule_id)

        def rank(lst):
            seen = []
            for s in lst:
                k = K.get(s)
                if k and k not in seen:
                    seen.append(k)
                    if k == t:
                        return len(seen)
            return 999
        L = fin.get(mid, [])
        d = dict(set=os.path.basename(run), mid=mid, r_final=rank(L), r_v4=rank(v4.get(mid, [])),
                 r_eng=rank(eng.get(mid, [])), r_full=rank(full.get(mid, [])))
        if L and d['r_final'] > 1:
            top = L[0]
            tm = Chem.MolFromSmiles(top)
            d['iso'] = int(tm is not None and CalcMolFormula(tm) == CalcMolFormula(m))
            d['pop_top'] = pop.get(K.get(top), 0.0); d['pop_ans'] = pop.get(t, 0.0)
            g = gl.get(mid) or {}; ic = ice.get(mid) or {}
            ga, gt = g.get(r.smiles), g.get(top)
            ans_s = next((s for s in full.get(mid, []) if K.get(s) == t), r.smiles)
            ga = g.get(ans_s, ga); ia, it = ic.get(ans_s), ic.get(top)
            d['gl_prefers_ans'] = np.nan if ga is None or gt is None else int(ga > gt)
            d['ice_prefers_ans'] = np.nan if ia is None or it is None else int(ia > it)
        rows.append(d)
D = pd.DataFrame(rows)
n = len(D)
print(f'molecules {n} | MRR {np.where(D.r_final <= 25, 1 / D.r_final, 0).mean():.3f} | rank 1: {(D.r_final == 1).sum()} | '
      f'rank 2-25: {((D.r_final > 1) & (D.r_final <= 25)).sum()} | not in top 25: {(D.r_final > 25).sum()}')
miss = D[D.r_final > 1]
print(f'\nmisses {len(miss)}')
print('  answer not in top 25:', int((miss.r_final > 25).sum()),
      '| of those in the 80-long fused list:', int(((miss.r_final > 25) & (miss.r_full <= 80)).sum()),
      '| in engine 1 list:', int(((miss.r_final > 25) & (miss.r_v4 < 999)).sum()),
      '| in engine 2 list:', int(((miss.r_final > 25) & (miss.r_eng < 999)).sum()))
m2 = miss[miss.r_final <= 25]
print(f'  answer at rank 2-25: {len(m2)}')
print('    #1 is a same-formula isomer:', int(m2.iso.sum()))
print('    #1 more popular than the answer:', int((m2.pop_top > m2.pop_ans).sum()),
      '| answer more popular:', int((m2.pop_ans > m2.pop_top).sum()))
print('    GLACIER prefers the answer over #1:', int(m2.gl_prefers_ans.sum()), 'of', int(m2.gl_prefers_ans.notna().sum()),
      '| ICEBERG:', int(m2.ice_prefers_ans.sum()), 'of', int(m2.ice_prefers_ans.notna().sum()))
print('    engine 1 had the answer at #1:', int((m2.r_v4 == 1).sum()), '| engine 2 at #1:', int((m2.r_eng == 1).sum()),
      '| neither:', int(((m2.r_v4 != 1) & (m2.r_eng != 1)).sum()))
print('    answer rank distribution:', m2.r_final.value_counts().sort_index().to_dict())
D.to_csv('/Users/b/casmi26/fp/miss_diag.csv', index=False)
