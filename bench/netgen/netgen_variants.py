"""netgen variants on one run: seeds top-k and multi-round propagation."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/b/casmi26/fp'); sys.path.insert(0, '/Users/b/casmi26/inputs/v4b/code')
import netgen
from casmi import fpnet, chem
from casmi.spectra import merge_spectra
test, sub, outdir, ckpt = sys.argv[1:5]
te = pd.read_parquet(test)
base = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(sub).values}
pop = pd.read_parquet('/Users/b/casmi26/mist/runs/pool_pop.parquet').groupby('scorer_key').popularity.max().to_dict()
bits = np.load('/Users/b/casmi26/fp/run/work/fp_bits.npy'); model = fpnet.load_model(ckpt, 'mps')
for k, rounds in ((1, 2), (3, 1), (3, 2)):
    netgen.K_SEED = k; cur = base
    for r in range(rounds):
        cur, st = netgen.run(te, cur, model, bits, chem, pop, 'mps', merge_spectra, fpnet)
    pd.DataFrame([(m, ';'.join(l[:25]) if l else 'CCO') for m, l in cur.items()], columns=['molecule_id', 'smiles']).to_csv(f'{outdir}/k{k}_r{rounds}.csv', index=False)
    print(k, rounds, st, flush=True)
