"""Flag per molecule: is the pipeline's own top-1 itself a single edit of a linked neighbour's top-k prediction
(or identical to one)? -> csv molecule_id, own_fam (0/1), own_fam_sim (max Tanimoto own top-1 vs neighbours' top-1)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/b/casmi26/fp'); sys.path.insert(0, '/Users/b/casmi26/inputs/v4b/code')
from edits import edit_products
from casmi import chem
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
sub_p, pairs_p, out, th, k = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
sub = {str(m): [x for x in str(v).split(';') if x and x != 'CCO'] for m, v in pd.read_csv(sub_p).values}
P = pd.read_parquet(pairs_p); P = P[P.modcos >= th]
nb = {}
for a, b in zip(P.a, P.b):
    nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fpc, kc = {}, {}
def fp(s):
    if s not in fpc:
        m = Chem.MolFromSmiles(s); fpc[s] = gen.GetFingerprint(m) if m else None
    return fpc[s]
def key(s):
    if s not in kc: kc[s] = chem.score_key(s)
    return kc[s]
rows = []
for m, lst in sub.items():
    if not lst or m not in nb:
        rows.append(dict(molecule_id=m, own_fam=0, own_fam_sim=0.0)); continue
    k1 = key(lst[0]); fam = 0; sims = []
    for y in nb[m]:
        for seed in sub.get(str(y), [])[:k]:
            if fp(seed) is not None and fp(lst[0]) is not None:
                sims.append(DataStructs.TanimotoSimilarity(fp(seed), fp(lst[0])))
            if key(seed) == k1 or k1 in {key(p) for p in edit_products(seed)}:
                fam = 1
    rows.append(dict(molecule_id=m, own_fam=fam, own_fam_sim=max(sims) if sims else 0.0))
pd.DataFrame(rows).to_csv(out, index=False); print('wrote', out)
