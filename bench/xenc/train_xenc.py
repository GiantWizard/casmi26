"""Train the engine's cross-encoder (casmi/xenc.py XEnc) as a same-formula isomer reranker.

The spectrum side is a frozen FPNet (token states + its f.z score); the candidate side is the molecular graph. Each
training group = one real training spectrum (fold-safe holdout excluded) + its true pool structure + NEG negatives from
the same mass window: the HARD hardest by the frozen FPNet's f.z, preferring the true structure's own formula, plus random
window members. Loss: softmax cross-entropy over the group. XEnc's score is a residual on standardised f.z, so at step 0
it ranks exactly like the FPNet and only has to learn corrections.
Logged per 100 steps: XEnc group accuracy vs the f.z-only accuracy on the same groups (the gap = what it adds).

usage: train_xenc.py --fp fpnet_fold0.pt --tag NAME [--hours 2] [--device cuda] [--outdir DIR] [--run RUN_DIR]
"""
import argparse, math, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='/Users/b/casmi26/inputs/v4b/code')
ap.add_argument('--run', default='/Users/b/casmi26/fp/run')
ap.add_argument('--fp', required=True)
ap.add_argument('--init', default='')
ap.add_argument('--arch', default='256,4,8', help='d,layers,heads')
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--neg', type=int, default=15)
ap.add_argument('--hard', type=int, default=12)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--warmup', type=int, default=500)
ap.add_argument('--steps', type=int, default=60000)
ap.add_argument('--hours', type=float, default=0.0)
ap.add_argument('--merge_p', type=float, default=0.3)
ap.add_argument('--device', default='cuda')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', required=True)
ap.add_argument('--outdir', default='/Users/b/casmi26/xenc')
ap.add_argument('--save_every', type=float, default=1.0)
ap.add_argument('--no_fz', action='store_true', help='independent cross-encoder: f.z is NOT an input (zeros, residual weight frozen at 0); the ranker combines it with f.z')
ap.add_argument('--profile', action='store_true', help='print per-phase timings every 10 steps')
ap.add_argument('--holdout', default='hold,np')
ap.add_argument('--train_fold', default='hold', help="train on this split fold, which the frozen FPNet never saw ('' = the FPNet's own training spectra)")
ap.add_argument('--exclude_sids', default='', help='parquet/npy of sids kept out of training (the harness evaluation half)')
a = ap.parse_args()
sys.path.insert(0, a.code)
os.chdir(a.run)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from casmi import fpnet, traindata, graphs, xenc  # noqa: E402
from casmi.pool import Pool  # noqa: E402

os.makedirs(a.outdir, exist_ok=True)
dev = a.device
torch.manual_seed(a.seed)
D = traindata.Data('cpu', seed=a.seed, holdout_folds=tuple(f for f in a.holdout.split(',') if f))
rng = np.random.default_rng(a.seed + 11); D.rng = rng
if a.train_fold:                                   # honest f.z: spectra of structures the frozen FPNet never trained on
    import pandas as pd
    excl = set()
    if a.exclude_sids:
        excl = set((pd.read_parquet(a.exclude_sids).sid if a.exclude_sids.endswith('.parquet') else np.load(a.exclude_sids)).tolist())
    ok = (D.fold[D.sid] == a.train_fold) & (D.sid2pid[D.sid] >= 0) & ~np.isin(D.sid, list(excl))
    D.tr = np.where(ok)[0]
    key = D.sid * 2 + (D.mode > 0)
    D.tr_groups = np.array([D.key2g[k] for k in np.unique(key[D.tr])])
    print(f'training on fold {a.train_fold!r}: {len(D.tr):,} spectra, {len(np.unique(D.sid[D.tr])):,} structures '
          f'({len(excl):,} sids excluded)', flush=True)
P = Pool('work')
FORM = np.asarray(P.formula)
PFP = D.PFP.to(dev)
fp = fpnet.load_model(a.fp, dev).eval()
for p_ in fp.parameters():
    p_.requires_grad_(False)
if a.init:
    model = xenc.load_model(a.init, dev).train()
else:
    d, L, H = (int(x) for x in a.arch.split(','))
    model = xenc.XEnc(d_spec=fp.d, d=d, layers=L, heads=H).to(dev).train()
if a.no_fz:
    with torch.no_grad():
        model.fz_w.zero_()
    model.fz_w.requires_grad_(False)
print(f'XEnc {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | spectra {len(D.tr):,}', flush=True)
_gcache = {}


def graph(pid):
    g = _gcache.get(pid)
    if g is None:
        g = graphs.mol_tokens(P.smiles[pid])
        if g is None:
            g = (np.zeros((1, graphs.N_ATOM_FEAT), np.uint8), np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint8))
        if len(_gcache) < 400_000:
            _gcache[pid] = g
    return g


decay = [p for n, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.01), dict(params=no_decay, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.98))
amp = dict(device_type='cuda', dtype=fpnet.autocast_dtype('cuda'), enabled=dev == 'cuda')
G = 1 + a.neg
t0 = time.time(); last_save = t0; ema = {}
for step in range(1, a.steps + 1):
    if a.hours and time.time() - t0 > a.hours * 3600:
        print('time budget reached at step', step, flush=True); break
    lr = a.lr * min(1.0, step / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
    for g_ in opt.param_groups:
        g_['lr'] = lr
    _T = [time.time()]
    ids = D.sample_ids(a.bs)
    items = [D.item(int(i), a.merge_p, True) for i in ids]
    pos = np.asarray(D.sid2pid[D.sid[ids]], np.int64)
    win = np.concatenate([D.negatives(pos) for _ in range(2)], 1)          # 126 same-window draws per query
    _T.append(time.time())
    args = fpnet.collate(items, dev)
    with torch.no_grad(), torch.autocast(**amp):
        sx, spad = fp.encode(*args)
        z = fp(*args)[0].float()
    sx = sx.float()
    _T.append(time.time())
    groups = np.empty((a.bs, G), np.int64)
    with torch.no_grad():
        for b in range(a.bs):
            c = np.unique(win[b]); c = c[c != pos[b]]
            fzc = (fpnet.unpack_gpu(PFP[torch.as_tensor(c, device=dev)], D.nbits) @ z[b]).cpu().numpy()
            same = FORM[c] == FORM[pos[b]]
            order = np.lexsort((-fzc, ~same))                              # same formula first, then by f.z
            hard = c[order[:a.hard]]
            rest = np.setdiff1d(c, hard)
            rnd = rng.choice(rest, a.neg - len(hard), replace=len(rest) < a.neg - len(hard)) if len(rest) else rng.choice(c, a.neg - len(hard))
            g_ = np.concatenate([[pos[b]], hard, rnd])
            if len(g_) < G:
                g_ = np.concatenate([g_, rng.integers(0, len(P.smiles), G - len(g_))])
            groups[b] = g_[:G]
    _T.append(time.time())
    gt = torch.as_tensor(groups, device=dev)
    cfp = fpnet.unpack_gpu(PFP[gt.reshape(-1)], D.nbits).reshape(a.bs, G, -1)
    fz = torch.einsum('bgn,bn->bg', cfp, z)
    fzs = (fz - fz.mean(1, keepdim=True)) / (fz.std(1, keepdim=True) + 1e-6)
    atoms, spd, bond, apad = xenc.collate_graphs([graph(int(p_)) for p_ in groups.reshape(-1)], dev)
    _T.append(time.time())
    si = torch.arange(a.bs, device=dev).repeat_interleave(G)
    with torch.autocast(**amp):
        sc = model(sx[si], spad[si], atoms, spd, bond, apad, (torch.zeros_like(fzs) if a.no_fz else fzs).reshape(-1)).float().reshape(a.bs, G)
    tgt = torch.zeros(a.bs, dtype=torch.long, device=dev)
    loss = F.cross_entropy(sc, tgt)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if a.profile:
        loss.item(); _T.append(time.time())
        _d = np.diff(_T)
        _acc = globals().setdefault('_PROF', np.zeros(len(_d))); _acc += _d
        if step % 10 == 0:
            print('profile s/step: items %.2f | fp encode %.2f | negatives %.2f | graphs+fz %.2f | xenc fwd/bwd %.2f' % tuple(_acc / 10), flush=True); _acc[:] = 0
    vals = dict(loss=loss.item(), acc=(sc.argmax(1) == 0).float().mean().item(), acc_fz=(fzs.argmax(1) == 0).float().mean().item(),
                same=float(np.mean(FORM[groups[:, 1:]] == FORM[groups[:, :1]])))
    for k, v in vals.items():
        ema[k] = v if step == 1 else 0.99 * ema[k] + 0.01 * v
    if step % 100 == 0:
        print(f'step {step} lr {lr:.2e} ' + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) +
              f' gain {ema["acc"] - ema["acc_fz"]:+.4f} | {step / (time.time() - t0):.2f} it/s', flush=True)
    if a.save_every and time.time() - last_save > a.save_every * 3600:
        model.eval(); xenc.save_ckpt(os.path.join(a.outdir, f'{a.tag}_partial.pt'), model, step, extra=dict(args=vars(a))); model.train()
        last_save = time.time(); print('saved partial at step', step, flush=True)
model.eval()
xenc.save_ckpt(os.path.join(a.outdir, f'{a.tag}.pt'), model, step, extra=dict(args=vars(a)))
print('saved', os.path.join(a.outdir, f'{a.tag}.pt'), f'{(time.time() - t0) / 60:.1f} min', flush=True)
