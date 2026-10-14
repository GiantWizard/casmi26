"""Are DreaMS embeddings better than modified cosine at finding structural relatives within a test set?

Embeds every spectrum (minimal DreaMS model, hengck23 dataset, MIT), averages the L2-normalised embeddings per
molecule and polarity, and scores each molecule pair by the best same-polarity cosine. Compares link precision
(share of linked pairs with Tanimoto >= 0.6) against modified cosine at equal numbers of links, and writes
DREAMS pairs (a, b, dcos, tani) next to the solution for net_rerank.py.

usage: dreams_links.py SOLUTION.csv TEST_DRAW.parquet [--device mps]
"""
import argparse, itertools, sys
import numpy as np, pandas as pd, torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '/Users/b/casmi26/ext/dreams')
import dreams_minimal_model as DM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('solution'); ap.add_argument('test')
ap.add_argument('--device', default='cpu')
a = ap.parse_args()
sol = pd.read_csv(a.solution); te = pd.read_parquet(a.test)
model = DM.load_model('/Users/b/casmi26/ext/dreams/embedding_model_state_dict.pt', a.device)
X = np.stack([DM.preprocess_spectrum(np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32),
                                     float(r.precursor_mz)) for r in te.itertuples()])
E = np.concatenate([DM.embed_preprocessed(model, torch.from_numpy(X[i:i + 256]).to(a.device)) for i in range(0, len(X), 256)])
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
emb = {}
for (m, mode), idx in te.groupby(['molecule_id', 'ionization_mode']).indices.items():
    v = E[idx].mean(0); emb[(m, mode)] = v / (np.linalg.norm(v) + 1e-12)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = {r.molecule_id: gen.GetFingerprint(Chem.MolFromSmiles(r.smiles)) for r in sol.itertuples() if Chem.MolFromSmiles(r.smiles)}
mids = sorted({m for m, _ in emb} & set(fp))
modes = {}
for m, md in emb:
    modes.setdefault(m, []).append(md)
rows = []
for x, y in itertools.combinations(mids, 2):
    best = max((float(emb[(x, md)] @ emb[(y, md)]) for md in modes[x] if (y, md) in emb), default=None)
    if best is not None:
        rows.append((x, y, best, DataStructs.TanimotoSimilarity(fp[x], fp[y])))
D = pd.DataFrame(rows, columns=['a', 'b', 'dcos', 'tani'])
D.to_parquet(a.solution.replace('solution.csv', 'pairs_dreams.parquet').replace('_solution.csv', '_pairs_dreams.parquet'))
base = (D.tani >= 0.6).mean()
M = pd.read_parquet(a.solution.replace('solution.csv', 'pairs.parquet').replace('_solution.csv', '_pairs.parquet')) \
    if a.solution.endswith('/solution.csv') else None
print(f'pairs {len(D)} | base rate of relatives among all pairs {base:.4f}')
for n in (100, 300, 1000, 2000):
    dp = (D.nlargest(n, 'dcos').tani >= 0.6).mean()
    mp = (M.nlargest(n, 'modcos').tani >= 0.6).mean() if M is not None else float('nan')
    print(f'top {n:5d} links: precision DreaMS {dp:.2f} | modified cosine {mp:.2f}')
