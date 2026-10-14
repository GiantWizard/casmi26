"""Per-molecule reciprocal rank (scorer key, dedup, top 25) for several submissions -> parquet."""
import sys, os, pandas as pd
sys.path.insert(0, '/Users/b/casmi26-repo/bench/obscure_np')
os.environ.setdefault('ONP_KEY_CACHE', '/Users/b/casmi26/mist/keycache')
from common import keys_parallel, scorer_key
sol = pd.read_csv(sys.argv[1]); out = sys.argv[2]; subs = dict(x.split('=', 1) for x in sys.argv[3:])
L = {n: {str(m): [s for s in str(v).split(';') if s and s != 'CCO'] for m, v in pd.read_csv(p).values} for n, p in subs.items()}
allsmi = {s for d in L.values() for l in d.values() for s in l} | set(sol.smiles)
K = keys_parallel(list(allsmi), scorer_key)
rows = []
for r in sol.itertuples():
    t = K.get(r.smiles); row = dict(molecule_id=r.molecule_id, cls=r.cls)
    for n, d in L.items():
        seen, rank, rr = set(), 0, 0.0
        for s in d.get(str(r.molecule_id), []):
            k = K.get(s)
            if not k or k in seen: continue
            seen.add(k); rank += 1
            if rank > 25: break
            if k == t: rr = 1 / rank; break
        row[n] = rr
    rows.append(row)
pd.DataFrame(rows).to_parquet(out); print('wrote', out)
