import sys, pandas as pd, numpy as np
sys.path.insert(0,'/home/user/casmi26/bench/obscure_np')
from common import keys_parallel, scorer_key
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
E, P = 0.00054857990946, 1.007276466812
M = dict(H=1.00782503223, C=12.0, N=14.00307400443, O=15.99491461957, Na=22.9897692820, K=38.9637064864, Cl=34.968852682)
H2O = 2*M['H']+M['O']
AD = {"[M+H]+": P, "[M+NH4]+": M['N']+4*M['H']-E, "[M+Na]+": M['Na']-E, "[M+K]+": M['K']-E, "[M-H2O+H]+": P-H2O,
      "[M-2H2O+H]+": P-2*H2O, "[M-H]-": -P, "[M-H2O-H]-": -P-H2O, "[M+CH2O2-H]-": M['C']+2*M['H']+2*M['O']-P, "[M+Cl]-": M['Cl']+E}
def mw(s):
    try: m=Chem.MolFromSmiles(s); return ExactMolWt(m), CalcMolFormula(m)
    except Exception: return np.nan, None
out=[]
for name, sub, sol, test in [(a.split(',')) for a in sys.argv[1:]]:
    t=pd.read_parquet(test, columns=['molecule_id','adduct','precursor_mz'])
    t=t[t.adduct.isin(AD)]; t['nm']=t.precursor_mz-t.adduct.map(AD)
    nm=t.groupby('molecule_id').nm.median()
    s=pd.read_csv(sub); so=pd.read_csv(sol).set_index('molecule_id')
    lists={m:str(x).split(';') for m,x in zip(s.molecule_id,s.smiles)}
    k=keys_parallel([y for v in lists.values() for y in v], scorer_key)
    for m,r in so.iterrows():
        if m not in lists or m not in nm: continue
        L=lists[m]; ks=[k.get(y) for y in L]
        rk=ks.index(r.scorer_key)+1 if r.scorer_key in ks else 0
        w1,f1=mw(L[0]); wt,ft=mw(r.smiles)
        ppm=lambda w: (w-nm[m])/nm[m]*1e6
        out.append(dict(set=name, cls=r.cls, rk=rk, f1=f1, ft=ft, ppm1=ppm(w1), ppmt=ppm(wt)))
d=pd.DataFrame(out)
d.to_csv(sys.argv[0].replace('.py','.csv'), index=False)
c=d[(d.cls==2)]
print('all cls2 truth |ppm| quantiles', c.ppmt.abs().quantile([.5,.9,.95,.99]).round(2).to_dict())
w=c[(c.rk!=1)&(c.f1!=c.ft)]
print('wrong-formula #1 cases', len(w)); print(w[['set','rk','f1','ft','ppm1','ppmt']].round(2).to_string())
