"""For out-of-pool truths: where does the truth appear (engine-1 v4 list, final list), and was it a generated candidate?"""
import sys, json, pandas as pd, numpy as np
sys.path.insert(0,'/home/user/casmi26/bench/obscure_np')
from common import keys_parallel, scorer_key
S='/tmp/claude-0/-home-user-casmi26/d38e5d0b-b943-5975-bfa4-a7b4cfe15d4f/scratchpad'
pool=set(pd.read_parquet(f'{S}/pool/pool_meta.parquet',columns=['key']).key)
rows=[]
for name, run, final, sol in [a.split(',') for a in sys.argv[1:]]:
    so=pd.read_csv(sol).set_index('molecule_id')
    v4=pd.read_csv(f'{run}/v4_submission.csv'); fin=pd.read_csv(final)
    pc=json.load(open(f'{run}/pc_lists.json')); eng=json.load(open(f'{run}/eng_lists.json'))
    L4={m:str(x).split(';') for m,x in zip(v4.molecule_id,v4.smiles)}; LF={m:str(x).split(';') for m,x in zip(fin.molecule_id,fin.smiles)}
    k=keys_parallel([y for d in (L4,LF) for v in d.values() for y in v], scorer_key)
    for m,r in so.iterrows():
        t=r.scorer_key
        r4=[k.get(y) for y in L4.get(m,[])]; rf=[k.get(y) for y in LF.get(m,[])]
        pck=set((pc.get(m) or {}).get('pc_keys',[])); ek=set((eng.get(m) or {}).get('keys',[]))
        ngen4=sum(1 for x in r4 if x and x not in pool and x not in pck and x not in ek)
        rows.append(dict(set=name, mid=m, cls=r.cls, in_pool=t in pool, in_pc=t in pck,
                         rk4=r4.index(t)+1 if t in r4 else 0, rkf=rf.index(t)+1 if t in rf else 0, ngen4=ngen4, n4=len(r4)))
d=pd.DataFrame(rows); d.to_csv(f'{S}/gen_probe.csv',index=False)
for nm,g in d.groupby('set'):
    novel=g[~g.in_pool & ~g.in_pc]
    print(nm, 'mols',len(g),'novel-like',len(novel),
          '| truth in v4 top25:',(novel.rk4>0).sum(),'ranks',sorted(novel.rk4[novel.rk4>0].tolist()),
          '| in final:',(novel.rkf>0).sum(), sorted(novel.rkf[novel.rkf>0].tolist()),
          '| mean generated cands in v4 list (novel mols):',round(novel.ngen4.mean(),1),'all:',round(g.ngen4.mean(),1))
