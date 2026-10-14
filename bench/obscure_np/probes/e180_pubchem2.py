import sys, numpy as np, pandas as pd, pyarrow.parquet as pq, multiprocessing as mp
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Descriptors import ExactMolWt
S=sys.argv[1]
def canon(s):
    m=Chem.MolFromSmiles(s)
    if m is None: return None
    Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
def in_pc(smi):
    pm=np.load(f'{S}/pc/pc_mass.npy', mmap_mode='r'); ps=np.load(f'{S}/pc/pc_smiles.npy', mmap_mode='r'); po=np.load(f'{S}/pc/pc_off.npy', mmap_mode='r')
    m=Chem.MolFromSmiles(smi); c=canon(smi); w=ExactMolWt(m)
    lo,hi=np.searchsorted(pm,[w-2e-5,w+2e-5])
    blob=bytes(ps[po[lo]:po[hi]]).decode(); offs=po[lo:hi+1]-po[lo]
    for i in range(hi-lo):
        if canon(blob[offs[i]:offs[i+1]])==c: return 1, hi-lo
    return 0, hi-lo
if __name__=='__main__':
    for lib in sys.argv[2:]:
        t=pq.read_table(f'{S}/train/train.parquet',filters=[('ingest_lib','=',lib)],columns=['inchikey14','normalized_smiles']).to_pandas().drop_duplicates('inchikey14').sample(80,random_state=1)
        with mp.Pool(4) as p: res=p.map(in_pc, list(t.normalized_smiles))
        print(f'{lib}: in PubChem tier {np.mean([r[0] for r in res]):.2f} (n={len(res)}, median same-mass window {np.median([r[1] for r in res]):.0f})', flush=True)
