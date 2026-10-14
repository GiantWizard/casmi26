"""Combine insertion policies per molecule by a gate column: gate true -> SAFE file's list, else BOLD file's list."""
import sys, pandas as pd
safe, bold, gate, col, out = sys.argv[1:6]
S = dict(pd.read_csv(safe).values); Bd = dict(pd.read_csv(bold).values)
g = pd.read_csv(gate).set_index('molecule_id')[col].astype(bool).to_dict()
rows = [(m, S[m] if g.get(str(m), False) else Bd[m]) for m in Bd]
pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(out, index=False); print('gated', sum(g.get(str(m), False) for m in Bd), 'of', len(rows))
