"""Run netgen.run offline on an eval run's final lists (validation of the notebook module)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '/Users/b/casmi26/fp'); sys.path.insert(0, '/Users/b/casmi26/inputs/v4b/code')
import netgen
from casmi import fpnet, chem
from casmi.spectra import merge_spectra
test, sub, out, ckpt = sys.argv[1:5]
te = pd.read_parquet(test)
lists = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(sub).values}
pop = pd.read_parquet('/Users/b/casmi26/mist/runs/pool_pop.parquet').groupby('scorer_key').popularity.max().to_dict()
bits = np.load('/Users/b/casmi26/fp/run/work/fp_bits.npy')
model = fpnet.load_model(ckpt, 'mps')
new, st = netgen.run(te, lists, model, bits, chem, pop, 'mps', merge_spectra, fpnet)
pd.DataFrame([(m, ';'.join(l[:25]) if l else 'CCO') for m, l in new.items()], columns=['molecule_id', 'smiles']).to_csv(out, index=False)
print(st)
