"""Cell-17 gated PubChem merge replayed on the NP panel's dropped-from-database engine-1 lists.
class3: the answer is also removed from PubChem (truly novel); class2pc: the answer stays in PubChem."""
import pickle, sys, numpy as np, pandas as pd
res = pickle.load(open(sys.argv[1], 'rb'))
c = pd.read_parquet(sys.argv[2]); c = c[np.isfinite(c.fpnet_score)]; c['molecule'] = c.molecule.astype(str)
c['pop'] = c.log_substances.fillna(0) + c.log_pubmed.fillna(0)
c['ident'] = np.where(c.scorer_key != '', c.scorer_key, c.connectivity)
POOL_MU, LIB_TAU, REL_TH, S_TAU, POP_TAU = 0.15, 0.9, 600.0, 6.0, 5.0
AGG, GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]
def merge(base, pc, slots, n=25):
    bk = set(base); pcs = [k for k in pc if k not in bk]; out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs): out.append(pcs[pj]); pj += 1
        elif bi < len(base): out.append(base[bi]); bi += 1
        elif pj < len(pcs): out.append(pcs[pj]); pj += 1
    return out
def base_list(r):
    keys, pid, sc, pop = r['keys'][:60], np.asarray(r['pid'][:60]), np.asarray(r['score'][:60], float), np.asarray(r['pop'][:60], float)
    if len(keys) > 1:                                   # v19 pool popularity rerank; generated keep their slots
        z = (sc - sc.mean()) / (sc.std() + 1e-9); isg = pid < 0
        f = z + POOL_MU * np.where(isg, 0, pop); pi = np.where(~isg)[0]; o = np.empty(len(pid), int)
        o[isg] = np.where(isg)[0]; o[~isg] = pi[np.argsort(-f[pi], kind='stable')]; keys = [keys[i] for i in o]
    seen, out = set(), []
    for k in keys:
        if k not in seen: seen.add(k); out.append(k)
    return out[:25]
G = dict(tuple(c.groupby('molecule', observed=True)))
rows = []
for key, r in res.items():
    if 'keys' not in r or key not in G: continue
    g = G[key]
    base = base_list(r)
    for regime in ('class3', 'class2pc'):
        gg = g[~g.in_pool | g.is_truth] if regime == 'class2pc' else g[~g.in_pool & ~g.is_truth]
        gg = gg.assign(z=(gg.fpnet_score - g.fpnet_score.mean()) / (g.fpnet_score.std() + 1e-9))
        gg = gg.assign(S=gg.z + 0.25 * gg['pop']).sort_values('S', ascending=False).drop_duplicates('ident')
        pc = list(np.where(gg.is_truth, key, gg.ident))[:25]
        S_top, pop_top = float(gg.S.iloc[0]), float(gg['pop'].iloc[0])
        bp = g[g.in_pool & ~g.is_truth].fpnet_score
        rel = float(gg.fpnet_score.iloc[0] - bp.max()) if len(bp) else 1e9
        lib_max = r['lib_max']
        if lib_max >= LIB_TAU:
            final, path = merge(base, pc, GENTLE), 'gated'
        else:
            slots = AGG if rel > REL_TH else GENTLE
            usual = merge(base, pc, slots)
            if (S_top > S_TAU and pop_top >= POP_TAU) or rel > 250:
                final, path = ([pc[0]] + [x for x in usual if x != pc[0]])[:25], 'promoted'
            else:
                final, path = usual, 'merged'
        rr = lambda L: 1.0 / (L.index(key) + 1) if key in L else 0.0
        rows.append(dict(key=key, regime=regime, path=path, base_rr=rr(base), final_rr=rr(final), gen_top=bool(np.asarray(r['pid'])[0] < 0)))
d = pd.DataFrame(rows)
for regime, x in d.groupby('regime'):
    print(f'{regime}: n {len(x)}  engine-1 list MRR {x.base_rr.mean():.3f}  after cell-17 merge {x.final_rr.mean():.3f}  '
          f'paths {x.path.value_counts().to_dict()}')
    hit = x[x.base_rr == 1]
    print(f'   molecules with the answer at engine-1 rank 1: {len(hit)}; after merge still rank 1: {(hit.final_rr == 1).sum()}  '
          f'(lost by path: {hit[hit.final_rr < 1].path.value_counts().to_dict()})')
