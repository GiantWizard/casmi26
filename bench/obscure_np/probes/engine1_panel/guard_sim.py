"""Cell-17 merge on jc-gen lists (default + wide-pass generated inserted) with a promotion guard:
skip promotion when engine 1's top entry is generated and its ranker score >= tau."""
import pickle, sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from insert import combine, dedup
a = pickle.load(open(f'{sys.argv[2]}/dbdrop.pkl', 'rb')); b = pickle.load(open(f'{sys.argv[2]}/dbdrop_wide.pkl', 'rb'))
c = pd.read_parquet(sys.argv[3]); c = c[np.isfinite(c.fpnet_score)]; c['molecule'] = c.molecule.astype(str)
c['pop'] = c.log_substances.fillna(0) + c.log_pubmed.fillna(0); c['ident'] = np.where(c.scorer_key != '', c.scorer_key, c.connectivity)
G = dict(tuple(c.groupby('molecule', observed=True)))
LIB_TAU, REL_TH, S_TAU, POP_TAU = 0.9, 600.0, 6.0, 5.0; AGG, GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]
def merge(base, pc, slots, n=25):
    bk = set(base); pcs = [k for k in pc if k not in bk]; out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs): out.append(pcs[pj]); pj += 1
        elif bi < len(base): out.append(base[bi]); bi += 1
        elif pj < len(pcs): out.append(pcs[pj]); pj += 1
    return out
# score / generated flag per key in the combined list
def info(key):
    d, w = a[key], b[key]; L = combine(d, w, n=25)
    sc = {}; gen = {}
    for src in (d, w):
        for k, p, s in zip(src['keys'], src['pid'], src['score']):
            sc.setdefault(k, float(s)); gen.setdefault(k, p < 0)
    return L, sc, gen
rows = []
for key in a:
    if 'keys' not in a[key] or 'keys' not in b.get(key, {}) or key not in G: continue
    L, sc, gen = info(key); g = G[key]; lib_max = a[key]['lib_max']
    for regime in ('class3', 'class2pc'):
        gg = g[~g.in_pool | g.is_truth] if regime == 'class2pc' else g[~g.in_pool & ~g.is_truth]
        gg = gg.assign(z=(gg.fpnet_score - g.fpnet_score.mean()) / (g.fpnet_score.std() + 1e-9))
        gg = gg.assign(S=gg.z + 0.25 * gg['pop']).sort_values('S', ascending=False).drop_duplicates('ident')
        pc = list(np.where(gg.is_truth, key, gg.ident))[:25]
        S_top, pop_top = float(gg.S.iloc[0]), float(gg['pop'].iloc[0])
        bp = g[g.in_pool & ~g.is_truth].fpnet_score; rel = float(gg.fpnet_score.iloc[0] - bp.max()) if len(bp) else 1e9
        for tau in (None, 3.0, 2.0, 1.0, 0.0):
            if lib_max >= LIB_TAU: final = merge(L, pc, GENTLE)
            else:
                slots = AGG if rel > REL_TH else GENTLE; usual = merge(L, pc, slots)
                fire = (S_top > S_TAU and pop_top >= POP_TAU) or rel > 250
                if tau is not None and L and gen.get(L[0]) and sc.get(L[0], -9) >= tau: fire = False
                final = ([pc[0]] + [x for x in usual if x != pc[0]])[:25] if fire else usual
            rows.append(dict(key=key, regime=regime, tau=str(tau), rr=1 / (final.index(key) + 1) if key in final else 0.0))
d = pd.DataFrame(rows)
print(d.pivot_table(index='tau', columns='regime', values='rr', aggfunc='mean').round(4).to_string())
