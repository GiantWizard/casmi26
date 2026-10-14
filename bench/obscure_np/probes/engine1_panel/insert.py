"""Default engine-1 list + extra generated candidates from the wide-generation run inserted at their wide-list positions."""
import pickle, sys, numpy as np
def dedup(keys):
    seen, out = set(), []
    for k in keys:
        if k not in seen: seen.add(k); out.append(k)
    return out
def combine(d, w, n=25, max_ins=25):
    D = dedup(d['keys']); Wk = w['keys']; Wp = np.asarray(w['pid'])
    W = []; seen = set()
    for k, p in zip(Wk, Wp):
        if k in seen: continue
        seen.add(k); W.append((k, p < 0 and k not in set(D)))
    out, di, ins = [], 0, 0
    for k, new in W[:n]:
        if len(out) >= n: break
        if new and ins < max_ins:
            out.append(k); ins += 1
        else:
            while di < len(D) and D[di] in out: di += 1
            if di < len(D): out.append(D[di]); di += 1
    while len(out) < n and di < len(D):
        if D[di] not in out: out.append(D[di])
        di += 1
    return out
if __name__ == '__main__':
    for scen in sys.argv[2:]:
        a = pickle.load(open(f'{sys.argv[1]}/{scen}.pkl', 'rb')); b = pickle.load(open(f'{sys.argv[1]}/{scen}_wide.pkl', 'rb'))
        rr = lambda L, k: 1 / (L.index(k) + 1) if k in L[:25] else 0.0
        for max_ins in (1, 2, 3, 25):
            x, y = [], []
            for k in a:
                if 'keys' not in a[k] or 'keys' not in b.get(k, {}): continue
                x.append(rr(dedup(a[k]['keys']), k)); y.append(rr(combine(a[k], b[k], max_ins=max_ins), k))
            x, y = np.array(x), np.array(y); d = y - x
            ci = np.percentile([np.random.default_rng(i).choice(d, len(d)).mean() for i in range(2000)], [5, 95])
            print(f'{scen:<7} insert<= {max_ins:<2}: default {x.mean():.3f} -> combined {y.mean():.3f}  delta {d.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  better {(d > 0).sum()} worse {(d < 0).sum()}')
