"""Nearest-neighbour Tanimoto (ECFP4 2048) of each simulated query structure to every OTHER training structure: how
much close-relative evidence the engine's analog channel can lean on. usage: nn_sim.py ROWS_DIR TRAIN_STRUCTS OUT.parquet"""
import glob, os, sys
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
rows, tsp, out = sys.argv[1:4]
sids = set()
for f in glob.glob(os.path.join(rows, 'rows_*.npz')):
    sids.update(np.load(f)['sids'].tolist())
T = pd.read_parquet(tsp, columns=['sid', 'smiles', 'key'])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None for s in T.smiles]
ok = [i for i, f in enumerate(fps) if f is not None]
bank = [fps[i] for i in ok]; bank_key = T.key.values[ok]
res = []
for n, sid in enumerate(sorted(sids)):
    i = int(np.flatnonzero(T.sid.values == sid)[0]); f = fps[i]
    if f is None:
        continue
    s = np.array(DataStructs.BulkTanimotoSimilarity(f, bank))
    s[bank_key == T.key.values[i]] = -1          # itself and metric-key twins
    res.append((sid, float(s.max())))
    if n % 1000 == 0:
        print(n, flush=True)
pd.DataFrame(res, columns=['sid', 'nn_sim']).to_parquet(out)
