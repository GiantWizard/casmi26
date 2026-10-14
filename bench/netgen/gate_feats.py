"""Per-molecule confidence features of the pipeline's own answer, for gating generated-candidate insertion."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/b/casmi26/inputs/v4b/code')
from casmi import chem
run = sys.argv[1]
v4 = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(f'{run}/v4_submission.csv').values}
eng = json.load(open(f'{run}/eng_lists.json')); pc = json.load(open(f'{run}/pc_lists.json'))
gl = json.load(open(f'{run}/gl_work/gl_out.json')) if __import__('os').path.exists(f'{run}/gl_work/gl_out.json') else {}
rows = []
for m, lst in v4.items():
    e = eng.get(m) or {}
    k1 = chem.score_key(lst[0]) if lst else None
    ek = (e.get('keys') or [None])
    agree = int(bool(k1) and k1 == ek[0])
    agree3 = int(bool(k1) and k1 in ek[:3])
    p = pc.get(m) or {}
    pk = p.get('pc_keys') or []
    rows.append(dict(molecule_id=m, agree=agree, agree3=agree3, pc_top_is_main=int(bool(k1) and pk[:1] == [k1]),
                     pc_S=float(p.get('S') or np.nan), n_main=len(lst)))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False); print('wrote', sys.argv[2], len(rows))
