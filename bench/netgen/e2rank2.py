"""Offline replica of the public e2rank2 overlay: when engine 2's top-1 differs (metric key) from the final #1,
insert it at rank 2."""
import sys, json, pandas as pd
sys.path.insert(0, '/Users/b/casmi26/inputs/v4b/code'); from casmi import chem
sub, eng_p, out = sys.argv[1:4]
eng = json.load(open(eng_p)); rows = []
for m, s in pd.read_csv(sub).values:
    toks = [t for t in str(s).split(';') if t]
    e = eng.get(str(m)) or {}; l = [x for x in (e.get('smiles') or []) if x and x != 'CCO']
    if toks and toks[0] != 'CCO' and l and chem.score_key(l[0]) and chem.score_key(l[0]) != chem.score_key(toks[0]):
        kx = chem.score_key(l[0]); rest = [t for t in toks if chem.score_key(t) != kx]
        toks = (rest[:1] + [l[0]] + rest[1:])[:25]
    rows.append((m, ';'.join(toks)))
pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(out, index=False); print('wrote', out)
