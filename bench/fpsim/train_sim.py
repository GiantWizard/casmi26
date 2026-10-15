"""Fine-tune an FPNet checkpoint on real training spectra plus GLACIER-simulated spectra of pool structures
(MIST/FRIGID-style forward-model augmentation), with Gluzdov's mass-window ranking loss. sim_p = 0 is the control.

Real spectra: casmi traindata.Data with the fold-safe holdout (hold, np) excluded. Simulated spectra map straight to
their pool entry (pid), so structures with no measured spectrum (COCONUT) become training targets.

usage: train_sim.py --init fpnet_fold0.pt --sim sim/pilot --sim_p 0.5 --steps 3000 --tag NAME [--device mps]
"""
import argparse, glob, json, math, os, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='/Users/b/casmi26/inputs/v4b/code')
ap.add_argument('--run', default='/Users/b/casmi26/fp/run')
ap.add_argument('--init', required=True)
ap.add_argument('--sim', default='')
ap.add_argument('--sim_p', type=float, default=0.5)
ap.add_argument('--exclude_sim_pids', default='', help='npy of pids whose simulated spectra are dropped (ablation)')
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--warmup', type=int, default=200)
ap.add_argument('--merge_p', type=float, default=0.3)
ap.add_argument('--w_bce', type=float, default=1.0)
ap.add_argument('--w_el', type=float, default=0.1)
ap.add_argument('--device', default='mps')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', required=True)
ap.add_argument('--outdir', default='/Users/b/casmi26/fp')
ap.add_argument('--hours', type=float, default=0.0, help='stop early after this many hours (0 = steps only)')
ap.add_argument('--holdout', default='hold,np', help="folds kept out of training ('' = train on everything, deployment)")
a = ap.parse_args()
sys.path.insert(0, a.code)
os.chdir(a.run)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from casmi import fpnet, traindata  # noqa: E402

D = traindata.Data('cpu', seed=a.seed, holdout_folds=tuple(f for f in a.holdout.split(',') if f))
rng = np.random.default_rng(a.seed + 7)
D.rng = rng

# ---- simulated spectra ------------------------------------------------------------------------------------------
S = None
if a.sim and a.sim_p > 0:
    parts = sorted(f for pre in a.sim.split(',') for f in glob.glob(pre + '_part*_meta.parquet'))
    metas, mzs, its, offs = [], [], [], []
    base = 0
    for mp in parts:
        pre = mp[:-len('_meta.parquet')]
        m = pd.read_parquet(mp); o = np.load(pre + '_off.npy')
        metas.append(m); mzs.append(np.load(pre + '_mz.npy')); its.append(np.load(pre + '_it.npy'))
        offs.append(o[:-1] + base); base += int(o[-1])
    meta = pd.concat(metas, ignore_index=True)
    ADD = {'[M+H]+': (0, 1.007276), '[M+Na]+': (4, 22.989218)}       # adduct_ix as in spec_meta, ion mass shift
    S = dict(add_ix=np.array([ADD[x][0] for x in meta.adduct.values], np.int64),
             add_mass=np.array([ADD[x][1] for x in meta.adduct.values], np.float64),
             pid=meta.pid.values.astype(np.int64), ce=meta.ce.values.astype(np.float64),
             off=np.concatenate(offs + [np.array([base])]), mz=np.concatenate(mzs), it=np.concatenate(its))
    keep = np.ones(len(S['pid']), bool)
    if a.exclude_sim_pids:
        keep &= ~np.isin(S['pid'], np.load(a.exclude_sim_pids))
    S['idx'] = np.where(keep)[0]
    print(f'simulated spectra {len(S["pid"]):,} (usable {len(S["idx"]):,}) over {len(np.unique(S["pid"])):,} structures', flush=True)
# precursor m/z of [M+H]+ from the pool's neutral mass
PROTON = 1.007276


def sim_item(j, aug=True):
    s, e = S['off'][j], S['off'][j + 1]
    mz = S['mz'][s:e].astype(np.float64); it = S['it'][s:e].astype(np.float64)
    prec = float(D.pmass[S['pid'][j]] + S['add_mass'][j])
    if aug:
        if len(mz) > 4:
            keep = rng.random(len(mz)) > rng.uniform(0.0, 0.3)
            if keep.sum() >= 3:
                mz, it = mz[keep], it[keep]
        it = it * np.exp(rng.normal(0.0, 0.3, size=len(it)))
        mz = mz * (1.0 + rng.normal(0.0, 4e-6, size=len(mz)))
    pm, pi = fpnet.prep_peaks(mz, it, prec)
    return dict(mz=pm, it=pi, prec=prec, adduct_ix=int(S['add_ix'][j]), ce=float(S['ce'][j]), ce_known=1.0, n_merged=1, mode=1.0)


def batch():
    n_sim = int(rng.binomial(a.bs, a.sim_p)) if S is not None else 0
    ids = D.sample_ids(a.bs - n_sim)
    items = [D.item(int(i), a.merge_p, True) for i in ids]
    pos = list(D.sid2pid[D.sid[ids]])
    if n_sim:
        js = rng.choice(S['idx'], n_sim, replace=False)
        items += [sim_item(int(j)) for j in js]
        pos += list(S['pid'][js])
    pos = np.array(pos, np.int64)
    return fpnet.collate(items, dev), pos, D.negatives(pos)


dev = a.device
torch.manual_seed(a.seed)
PFP = D.PFP.to(dev); ELEM = D.elem.to(dev)
model = fpnet.load_model(a.init, dev).train()
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.01), dict(params=no_decay, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.98))
t0 = time.time(); ema = {}
target = torch.zeros(a.bs, dtype=torch.long, device=dev)
for step in range(1, a.steps + 1):
    if a.hours and time.time() - t0 > a.hours * 3600:
        print('time budget reached at step', step, flush=True); break
    prog = step / a.steps
    lr = a.lr * min(1.0, step / a.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
    for g in opt.param_groups:
        g['lr'] = lr
    inp, pos, negs = batch()
    cand = torch.as_tensor(np.concatenate([pos[:, None], negs], 1), device=dev)
    if dev == 'cuda':
        with torch.autocast('cuda', dtype=fpnet.autocast_dtype('cuda')):
            z, el = model(*inp)
        z = z.float(); el = el.float()
    else:
        z, el = model(*inp)
    fp = fpnet.unpack_gpu(PFP[cand], D.nbits)
    sc = torch.einsum('bkn,bn->bk', fp, z)
    l_rank = F.cross_entropy(sc, target)
    l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])
    l_el = F.smooth_l1_loss(el, ELEM[torch.as_tensor(pos, device=dev)])
    loss = l_rank + a.w_bce * l_bce + a.w_el * l_el
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    vals = dict(loss=loss.item(), rank=l_rank.item(), acc=(sc.argmax(1) == 0).float().mean().item())
    for k, v in vals.items():
        ema[k] = v if step == 1 else 0.98 * ema[k] + 0.02 * v
    if step % 100 == 0:
        print(f'step {step} lr {lr:.2e} ' + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) +
              f' {step / (time.time() - t0):.2f} it/s', flush=True)
model.eval()
out = os.path.join(a.outdir, f'{a.tag}.pt')
fpnet.save_ckpt(out, model, step, extra=dict(args=vars(a)))
print('saved', out, f'{(time.time() - t0) / 60:.1f} min', flush=True)
