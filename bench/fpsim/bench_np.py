"""Honest class-2 benchmark for spectrum -> fingerprint models.

Queries: structures of the fold-safe 'hold' fold (never seen by fpnet_fold0) that are natural products (pool NP flag)
and are NOT in MassSpecGym's training fold (so the MassSpecGym GLACIER/ICEBERG used to simulate spectra never saw
them either). Each query merges up to 8 spectra of one (structure, polarity) group, as in Gluzdov's eval_ckpt.py.
Score: rank of the answer inside its +-10 ppm pool window by the fingerprint score alone (no popularity), which is
the class-2 situation: answer in the pool, no reference spectrum, popularity uninformative.

usage: bench_np.py --ckpt a.pt[,b.pt] [--n 1500] [--device mps] [--out res.json] [--write-queries q.parquet]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--code', default='/Users/b/casmi26/inputs/v4b/code')
ap.add_argument('--run', default='/Users/b/casmi26/fp/run')
ap.add_argument('--msg-keys', default='/Users/b/casmi26/mist/runs/msg_train_keys.json')
ap.add_argument('--np-flag', default='/Users/b/casmi26/inputs/pop/pool_np.npy')
ap.add_argument('--n', type=int, default=1500)
ap.add_argument('--device', default='mps')
ap.add_argument('--out', default='')
ap.add_argument('--sids', default='', help='npy of hold sids to use (overrides --n sampling)')
ap.add_argument('--only', default='', help='comma list of sets to run')
a = ap.parse_args()
sys.path.insert(0, a.code)
os.chdir(a.run)
import torch  # noqa: E402
from casmi import fpnet, traindata  # noqa: E402
from casmi.spectra import merge_spectra  # noqa: E402

D = traindata.Data('cpu', seed=0, holdout_folds=('hold', 'np'))
st = pd.read_parquet('work/train_structs.parquet', columns=['sid', 'key'])
key_of = st.key.values
msg = set(json.load(open(a.msg_keys)))
npf = np.load(a.np_flag)
meta = pd.read_parquet('work/spec_meta.parquet', columns=['instr'])
instr = meta.instr.values


def groups_for(fold):
    ok = (D.fold[D.sid] == fold) & (D.sid2pid[D.sid] >= 0)
    g = {}
    for i in np.where(ok)[0]:
        g.setdefault((int(D.sid[i]), int(D.mode[i])), []).append(int(i))
    return g


rng = np.random.default_rng(12345)
sets = {}
G = groups_for('hold')
keys = [k for k in sorted(G) if npf[D.sid2pid[k[0]]] > 0 and str(key_of[k[0]])[:14] not in msg]
print('hold groups', len(G), '| NP and not in MassSpecGym train:', len(keys), flush=True)
if a.sids:
    want = set(np.load(a.sids).tolist()); keys = [k for k in keys if k[0] in want]
    print('restricted to --sids:', len(keys), 'groups', flush=True)
elif len(keys) > a.n:
    keys = [keys[i] for i in sorted(rng.choice(len(keys), a.n, replace=False))]
sets['hold_np_clean'] = [G[k][:8] for k in keys]
Gn = groups_for('np')
sets['np_panel'] = [v[:8] for k, v in sorted(Gn.items())]
Gh = [k for k in sorted(G) if npf[D.sid2pid[k[0]]] <= 0]
Gh = [Gh[i] for i in sorted(rng.choice(len(Gh), min(a.n, len(Gh)), replace=False))]
sets['hold_nonnp'] = [G[k][:8] for k in Gh]
dev = a.device
PFP = D.PFP.to(dev)
models = [fpnet.load_model(p, dev) for p in a.ckpt.split(',')]


@torch.no_grad()
def window_eval(entries, bs=64):
    ranks, sizes, tims = [], [], []
    for s in range(0, len(entries), bs):
        chunk = entries[s:s + bs]; items = []
        for idxs in chunk:
            i = idxs[0]
            if len(idxs) == 1:
                mz = D.mz[D.off[i]:D.off[i + 1]]; it = D.it[D.off[i]:D.off[i + 1]]; ce = D.ce[i]; nm = max(1, D.ce_n[i])
            else:
                mz, it = merge_spectra([(D.mz[D.off[j]:D.off[j + 1]], D.it[D.off[j]:D.off[j + 1]]) for j in idxs])
                ces = D.ce[idxs]; ces = ces[~np.isnan(ces)]; ce = ces.mean() if len(ces) else np.nan
                nm = int(sum(max(1, D.ce_n[j]) for j in idxs))
            pm, pi = fpnet.prep_peaks(mz, it, D.prec[i])
            items.append(dict(mz=pm, it=pi, prec=float(D.prec[i]), adduct_ix=int(D.adduct[i]), ce=0.0 if np.isnan(ce) else float(ce),
                              ce_known=0.0 if np.isnan(ce) else 1.0, n_merged=min(nm, 8), mode=float(D.mode[i])))
        inp = fpnet.collate(items, dev)
        if dev == 'cuda':
            with torch.autocast('cuda', dtype=fpnet.autocast_dtype('cuda')):
                z = torch.stack([m(*inp)[0].float() for m in models]).mean(0)
        else:
            z = torch.stack([m(*inp)[0].float() for m in models]).mean(0)
        for r, idxs in enumerate(chunk):
            pid = D.sid2pid[D.sid[idxs[0]]]; m = D.pmass[pid]; tol = m * 10e-6
            lo = np.searchsorted(D.pmass, m - tol, 'left'); hi = np.searchsorted(D.pmass, m + tol, 'right')
            sc = fpnet.unpack_gpu(PFP[lo:hi], D.nbits).float() @ z[r]
            ranks.append(int((sc > sc[pid - lo]).sum().item()) + 1); sizes.append(int(hi - lo))
            tims.append(int(np.bincount(instr[idxs]).argmax()))
    return np.array(ranks), np.array(sizes), np.array(tims)


def summ(r):
    return dict(n=int(len(r)), mrr25=round(float(np.where(r <= 25, 1.0 / r, 0).mean()), 4), top1=round(float((r == 1).mean()), 4),
                top5=round(float((r <= 5).mean()), 4), top25=round(float((r <= 25).mean()), 4))


res = {}
if a.only:
    sets = {k: v for k, v in sets.items() if k in a.only.split(',')}
for name, ent in sets.items():
    r, sz, ins = window_eval(ent)
    res[name] = summ(r); res[name]['median_window'] = float(np.median(sz))
    for code in np.unique(ins):
        if (ins == code).sum() >= 30:
            res[f'{name}_instr{code}'] = summ(r[ins == code])
    np.save(os.path.join(os.path.dirname(a.out) or '.', f'ranks_{name}.npy'), r) if a.out else None
    print(name, res[name], flush=True)
res['ckpt'] = a.ckpt
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, 'w'), indent=1)
