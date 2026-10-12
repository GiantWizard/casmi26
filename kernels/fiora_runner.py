"""FIORA isomer re-scorer runner (third forward model next to ICEBERG / GLACIER). Standalone script, run in its own
process so a crash or CUDA OOM never reaches the notebook.

    python fiora_runner.py <fiora_root> <in.json> <out.json> [--device cuda] [--budget 3600] [--site DIR]
                           [--ce-missing 40] [--batch 64] [--max-conds 4]

fiora_root: the dataset ahmedberatozer/casmi26-fiora-os-v1 (FIORA OS v1.0.0, MIT code by Nowatzky / BAM, weights
trained on MSnLib v7 only, CC BY 4.0). Its source/ is used as-is; only treelib + torch_geometric (pure-python wheels)
are installed into --site. The cp312-only wheels (torch_scatter, fastobo) are skipped on other interpreters: FIORA's
inference path does not need them (PyG falls back to native torch scatter), and the plotting module that pulls in
spectrum_utils / pyteomics / fastobo is replaced by a stub.

in.json: the ICEBERG / GLACIER runner input (fuse.build_ice_input(..., covered_only=False)):
    [{mid, spectra: [{mz, it, prec, adduct, mode, instrument, ce_ev, ...}], cands: [smiles, ...]}, ...]
Spectra with an adduct FIORA has no token for are skipped (OS v1.0.0: [M+H]+, [M-H]-, [M]+, [M]-).
Collision energy: mean of ce_ev (MSnLib loader convention, stepped 20/40/60 -> 40), clamped to (1, 100]; missing ->
--ce-missing. Instrument token is always HCD (the only one the model was trained on).

out.json: {mid: {smiles: score | None}} = mean over the molecule's supported spectra of the weighted spectral-entropy
similarity (Li et al. 2021, 0.01 Da greedy matching; same as the dataset's scorer/fiora_score.py) between the measured
and the FIORA-predicted spectrum. out_cos.json (same layout): sqrt-intensity matched-peak cosine, for offline choice.
out.json.meta.json: status / counts / timing (ice_runner conventions). Checkpointed while running; stops at --budget.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
import traceback
import types

T0 = time.time()
LOG = []


def log(*a):
    s = f'[fiora {time.time() - T0:7.1f}s] ' + ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('fin')
ap.add_argument('fout')
ap.add_argument('--device', default='cuda')
ap.add_argument('--budget', type=float, default=3600.0)
ap.add_argument('--site', default='/tmp/fiora_site')
ap.add_argument('--ce-missing', type=float, default=40.0)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--max-conds', type=int, default=4, help='distinct (adduct, CE) conditions per molecule')
ap.add_argument('--min-prob', type=float, default=0.001)
ap.add_argument('--checkpoint-s', type=float, default=60.0)
ap.add_argument('--workers', type=int, default=0, help='parallel worker processes (0 = min(4, cpus)); each holds its '
                'own copy of the model, the CPU-side graph / fragment-tree build is the bottleneck')
ap.add_argument('--shard', type=int, default=-1, help=argparse.SUPPRESS)
ap.add_argument('--nshards', type=int, default=1, help=argparse.SUPPRESS)
args = ap.parse_args()
if args.workers <= 0:
    args.workers = max(1, min(4, os.cpu_count() or 1))

OUT, OUT_COS = {}, {}
META = dict(model='fiora', status='starting', n_mols=0, n_mols_covered=0, n_mols_scored=0, n_cands=0, n_scored=0,
            n_pred=0, n_pred_fail=0, n_graph_fail=0, n_spectra=0, n_spectra_unsupported=0, n_ce_missing=0,
            n_ce_clamped=0, errors=[], seconds=0.0)
COS_PATH = args.fout[:-5] + '_cos.json' if args.fout.endswith('.json') else args.fout + '_cos.json'


def dump():
    META['seconds'] = round(time.time() - T0, 2)
    META['pred_per_s'] = round(META['n_pred'] / max(1e-9, META['seconds']), 2)   # wall clock, all workers
    META['log_tail'] = LOG[-25:]
    for path, value in ((args.fout, OUT), (COS_PATH, OUT_COS), (args.fout + '.meta.json', META)):
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(value, f)
        os.replace(tmp, path)


def err(**k):
    if len(META['errors']) < 20:
        META['errors'].append(k)


# ------------------------------------------------------------------------------------------------ install -------
def install(root, site):
    """treelib + torch_geometric from the dataset's wheels into site (idempotent via a marker)."""
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    whls = sorted(glob.glob(os.path.join(root, '**', '*.whl'), recursive=True))
    want = [w for w in whls if os.path.basename(w).split('-')[0].lower() in ('treelib', 'torch_geometric')]
    want = [w for w in want if '-none-any' in w or f'-{tag}-' in w]
    marker = os.path.join(site, '.fiora_site_ok')
    names = '\n'.join(sorted(os.path.basename(w) for w in want))
    if os.path.exists(marker) and open(marker).read() == names:
        log('site ready (marker matches)', site)
    else:
        os.makedirs(site, exist_ok=True)
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--no-compile',
                            '--disable-pip-version-check', '--upgrade', '--target', site] + want,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log('pip rc', r.returncode, [os.path.basename(w) for w in want], (r.stdout or '')[-300:])
        if r.returncode:
            raise RuntimeError('FIORA wheel install failed: ' + (r.stdout or '')[-500:])
        open(marker, 'w').write(names)
    skipped = [os.path.basename(w) for w in whls if w not in want]
    log('wheels not installed (not needed for inference / wrong cp tag):', skipped)
    sys.path.insert(0, site)
    src = os.path.dirname(os.path.dirname(os.path.dirname(
        glob.glob(os.path.join(root, '**', 'fiora', 'GNN', 'FioraModel.py'), recursive=True)[0])))
    sys.path.insert(0, src)
    return src


def _stub_attr(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return lambda *a, **k: None


def stub_modules():
    """Modules FIORA imports only for plotting / saving. Stubbed only when absent (or always, for the visualizer)."""
    stubbed = []
    sv = types.ModuleType('fiora.visualization.spectrum_visualizer')   # pulls spectrum_utils/pyteomics/fastobo
    sv.__getattr__ = _stub_attr
    sys.modules['fiora.visualization.spectrum_visualizer'] = sv
    import fiora.visualization as _fv                                  # empty package __init__
    _fv.spectrum_visualizer = sv
    stubbed.append('fiora.visualization.spectrum_visualizer')
    for name in ('dill', 'psutil', 'IPython', 'IPython.display'):   # psutil: only PyG's loader affinity mixin
        try:
            __import__(name)
        except Exception:
            m = types.ModuleType(name)
            m.__getattr__ = _stub_attr
            sys.modules[name] = m
            stubbed.append(name)
    if 'IPython.display' in stubbed and 'IPython' in sys.modules:
        sys.modules['IPython'].display = sys.modules['IPython.display']
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        pass
    return stubbed


# --------------------------------------------------------------------------------------------- similarity -------
import numpy as np  # noqa: E402


def clean(mz, it, prec, floor=0.01, topk=150):
    """= scorer/fiora_score.py::_clean: finite, positive, <= precursor + 1, >= 1% of base peak, top 150, sorted."""
    mz = np.asarray(mz, float); it = np.asarray(it, float)
    if mz.ndim != 1 or it.ndim != 1 or len(mz) != len(it) or not np.isfinite(prec):
        return np.array([]), np.array([])
    k = np.isfinite(mz) & (mz > 0) & (mz <= prec + 1.0) & np.isfinite(it) & (it > 0)
    mz, it = mz[k], it[k]
    if len(it) == 0:
        return mz, it
    k = it >= floor * it.max()
    mz, it = mz[k], it[k]
    if len(it) > topk:
        o = np.argsort(-it)[:topk]; mz, it = mz[o], it[o]
    o = np.argsort(mz)
    return mz[o], it[o]


def _weight(p):
    p = p / p.sum()
    s = -np.sum(p * np.log(p))
    if s < 3.0:
        p = p ** (0.25 + 0.25 * s); p = p / p.sum()
    return p


def _match(mz1, a, mz2, b, tol):
    """greedy (strongest peak of spectrum 1 first) one-to-one matching within tol Da -> (i, j) pairs."""
    used = np.zeros(len(mz2), bool); pairs = []
    for i in np.argsort(-a):
        j0 = np.searchsorted(mz2, mz1[i] - tol); j1 = np.searchsorted(mz2, mz1[i] + tol, 'right')
        best, bj = -1.0, -1
        for j in range(j0, j1):
            if not used[j] and b[j] > best:
                best, bj = b[j], j
        if bj >= 0:
            used[bj] = True; pairs.append((i, bj))
    return pairs


def entropy_sim(mz1, i1, mz2, i2, tol=0.01):
    """= scorer/fiora_score.py::entropy_sim (weighted spectral entropy similarity, Li et al. 2021)."""
    if len(mz1) == 0 or len(mz2) == 0:
        return 0.0
    a = _weight(np.asarray(i1, float)); b = _weight(np.asarray(i2, float))
    pairs = _match(mz1, a, mz2, b, tol)
    ua = np.zeros(len(mz1), bool); ub = np.zeros(len(mz2), bool)
    merged = []
    for i, j in pairs:
        ua[i] = ub[j] = True; merged.append(a[i] + b[j])
    ab = np.concatenate([a[~ua], b[~ub], np.asarray(merged)]) / 2.0
    ab = ab[ab > 0]
    h = lambda p: -np.sum(p * np.log(p))
    return float(max(0.0, 1.0 - (2 * h(ab) - h(a[a > 0]) - h(b[b > 0])) / np.log(4)))


def sqrt_cosine(mz1, i1, mz2, i2, tol=0.01):
    if len(mz1) == 0 or len(mz2) == 0:
        return 0.0
    a = np.sqrt(np.asarray(i1, float)); b = np.sqrt(np.asarray(i2, float))
    pairs = _match(mz1, a, mz2, b, tol)
    num = sum(a[i] * b[j] for i, j in pairs)
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float(num / den) if den > 0 else 0.0


def energy(ce):
    try:
        v = np.atleast_1d(np.asarray(ce if ce is not None else [], float))
        v = v[np.isfinite(v)]
    except (TypeError, ValueError):
        v = np.array([])
    if not len(v):
        META['n_ce_missing'] += 1
        return float(args.ce_missing)
    e = float(np.abs(v).mean())
    if not 1.0 < e <= 100.0:
        META['n_ce_clamped'] += 1
        e = min(100.0, max(1.01, e))
    return round(e, 1)


def patch_batched_subgraph_pooling(torch):
    """FIORA v1.0.0 EdgePropertyPredictor.concat_node_pairs adds the per-graph node offset (batch.ptr) to the -1
    padding of subgraph_idx_left/right too, so in a batch of >1 graphs every graph after the first pools the previous
    graph's last node into its subgraph features. With one graph per batch (fiora-predict) ptr is 0 and the padding
    stays -1. Same code with the padding kept at -1, so a batched forward == one-at-a-time inference."""
    from fiora.GNN.EdgePropertyPredictor import EdgePropertyPredictor

    def concat_node_pairs(self, X, batch):
        src, dst = batch["edge_index"]
        node_pairs = torch.cat([X[src], X[dst]], dim=1)
        if self.subgraph_features:
            num_edges = batch.edge_index.size(1)
            off = batch.ptr[batch.batch[batch.edge_index[0]]].unsqueeze(1)
            li, ri = batch.subgraph_idx_left, batch.subgraph_idx_right
            left_indices = torch.where(li == -1, li, li + off)
            right_indices = torch.where(ri == -1, ri, ri + off)
            left_batch_vec = torch.arange(num_edges, device=X.device).repeat_interleave(left_indices.size(1))
            right_batch_vec = torch.arange(num_edges, device=X.device).repeat_interleave(right_indices.size(1))
            left_flat, right_flat = left_indices.flatten(), right_indices.flatten()
            left_mask, right_mask = left_flat != -1, right_flat != -1
            pooled_left = torch.zeros(num_edges, X.size(1), device=X.device)
            pooled_right = torch.zeros(num_edges, X.size(1), device=X.device)
            if left_mask.any():
                pooled_left = self.pooling_func(X[left_flat[left_mask]], left_batch_vec[left_mask], size=num_edges)
            if right_mask.any():
                pooled_right = self.pooling_func(X[right_flat[right_mask]], right_batch_vec[right_mask], size=num_edges)
            node_pairs = torch.cat([node_pairs, pooled_left, pooled_right, batch["edge_elem_comp"]], dim=1)
        return node_pairs

    EdgePropertyPredictor.concat_node_pairs = concat_node_pairs


# ------------------------------------------------------------------------------------------------------ main -----
def parent():
    """Split the items round-robin over args.workers copies of this script, wait, merge their outputs."""
    items = json.load(open(args.fin))
    for it in items:
        OUT[str(it['mid'])] = {str(c): None for c in it.get('cands') or []}
        OUT_COS[str(it['mid'])] = {str(c): None for c in it.get('cands') or []}
    META.update(n_mols=len(items), n_cands=sum(len(it.get('cands') or []) for it in items), workers=args.workers,
                status='running')
    dump()
    install(args.root, args.site)                      # once, before the workers start (they find the marker)
    K = args.workers
    procs, parts = [], []
    for i in range(K):
        fin_i, fout_i = f'{args.fout}.in{i}.json', f'{args.fout}.part{i}.json'
        json.dump(items[i::K], open(fin_i, 'w'))
        cmd = [sys.executable, os.path.abspath(__file__), args.root, fin_i, fout_i, '--device', args.device,
               '--budget', str(max(1.0, args.budget - (time.time() - T0))), '--site', args.site,
               '--ce-missing', str(args.ce_missing), '--batch', str(args.batch), '--max-conds', str(args.max_conds),
               '--min-prob', str(args.min_prob), '--checkpoint-s', str(args.checkpoint_s),
               '--workers', '1', '--shard', str(i), '--nshards', str(K)]
        logf = open(f'{args.fout}.worker{i}.log', 'w')
        procs.append((subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf))
        parts.append(fout_i)
    log('started', K, 'workers on', len(items), 'molecules')
    deadline = T0 + args.budget + 120.0
    while any(p.poll() is None for p, _ in procs):
        if time.time() > deadline:
            for p, _ in procs:
                if p.poll() is None:
                    p.kill()
            log('workers killed at the hard deadline')
            break
        time.sleep(5)
    rcs = [p.poll() for p, _ in procs]
    for _, f in procs:
        f.close()
    sts = []
    for i, fp in enumerate(parts):
        try:
            for path, dst in ((fp, OUT), (fp[:-5] + '_cos.json', OUT_COS)):
                for mid, d in json.load(open(path)).items():
                    dst.setdefault(mid, {}).update(d)
            m = json.load(open(fp + '.meta.json'))
            sts.append(m.get('status'))
            for k in ('n_mols_covered', 'n_mols_scored', 'n_scored', 'n_pred', 'n_pred_fail', 'n_graph_fail',
                      'n_spectra', 'n_spectra_unsupported', 'n_ce_missing', 'n_ce_clamped', 't_pred', 't_build'):
                META[k] = META.get(k, 0) + (m.get(k) or 0)
            META['errors'] = (META['errors'] + m.get('errors', []))[:20]
            for k in ('device', 'torch', 'torch_geometric', 'rdkit', 'fiora_version', 'training_library', 'modes',
                      'instrument', 'stubbed', 't_load'):
                META.setdefault(k, m.get(k))
            LOG.extend(f'[w{i}] ' + x for x in (m.get('log_tail') or [])[-4:])
        except Exception as e:
            sts.append('missing')
            err(stage='merge', part=i, error=repr(e)[:200])
    META['worker_status'], META['worker_rc'] = sts, rcs
    META['status'] = 'ok' if all(x == 'ok' for x in sts) else ('timeout' if 'timeout' in sts else
                                                                 ('partial' if 'ok' in sts else 'failed'))
    META['pred_per_s_worker'] = round(META['n_pred'] / max(1e-9, META.get('t_pred') or 0), 2)
    log('merged', {k: META.get(k) for k in ('status', 'worker_status', 'n_mols_covered', 'n_mols_scored', 'n_scored',
                                            'n_pred', 'n_pred_fail', 'n_graph_fail')})
    dump()
    for i in range(K):
        for f in (f'{args.fout}.in{i}.json',):
            try:
                os.remove(f)
            except OSError:
                pass


def main():
    items = json.load(open(args.fin))
    META['n_mols'] = len(items)
    META['n_cands'] = sum(len(it.get('cands') or []) for it in items)
    for it in items:                                      # a shell entry per input molecule, like ice_runner
        OUT[str(it['mid'])] = {str(c): None for c in it.get('cands') or []}
        OUT_COS[str(it['mid'])] = {str(c): None for c in it.get('cands') or []}
    dump()
    src = install(args.root, args.site)
    stubbed = stub_modules()
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    import torch_geometric
    if args.shard >= 0 and args.nshards > 1:
        torch.set_num_threads(1)
    from rdkit import RDLogger
    import rdkit
    RDLogger.DisableLog('rdApp.*')
    from fiora.GNN.FioraModel import FioraModel
    from fiora.MS.SimulationFramework import SimulationFramework
    from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
    from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
    from fiora.GNN.CovariateFeatureEncoder import CovariateFeatureEncoder
    from fiora.MOL.Metabolite import Metabolite
    patch_batched_subgraph_pooling(torch)
    dev = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    state = glob.glob(os.path.join(args.root, '**', 'fiora_OS_v1.0.0_state.pt'), recursive=True)[0]
    model = FioraModel.load_from_state_dict(state.replace('_state.pt', '.pt')).eval().to(dev)
    params = model.model_params
    sets = params.get('setup_features_categorical_set') or {}
    MODES = tuple(sets.get('precursor_mode') or ('[M+H]+', '[M-H]-'))
    INSTR = (sets.get('instrument') or ['HCD'])[0]
    node = AtomFeatureEncoder(feature_list=['symbol', 'num_hydrogen', 'ring_type'])
    bond = BondFeatureEncoder(feature_list=['bond_type', 'ring_type'])
    cov = ['collision_energy', 'molecular_weight', 'precursor_mode', 'instrument'] + \
        (['element_composition'] if params.get('version_number') != '0.1.0' else [])
    setup_enc = CovariateFeatureEncoder(feature_list=cov, sets_overwrite=sets or None)
    rt_enc = CovariateFeatureEncoder(feature_list=['molecular_weight', 'precursor_mode', 'instrument'],
                                     sets_overwrite=sets or None)
    setup_enc.normalize_features['collision_energy']['max'] = 100.0      # = fiora-predict
    setup_enc.normalize_features['molecular_weight']['max'] = 1000.0
    rt_enc.normalize_features['molecular_weight']['max'] = 1000.0
    sim = SimulationFramework(None, dev=dev)
    META.update(device=dev, torch=torch.__version__, torch_geometric=torch_geometric.__version__,
                rdkit=rdkit.__version__, fiora_version=params.get('version'),
                training_library=params.get('training_library'), modes=list(MODES), instrument=INSTR,
                stubbed=stubbed, src=src, t_load=round(time.time() - T0, 2), t_pred=0.0, t_build=0.0, status='running')
    log('model loaded', params.get('version'), params.get('training_library'), 'on', dev, '| torch', torch.__version__,
        '| pyg', torch_geometric.__version__, '| rdkit', rdkit.__version__, '| modes', MODES, '| stubbed', stubbed)
    dump()

    def build(smi):
        t = time.time()
        try:
            m = Metabolite(smi)
            m.create_molecular_structure_graph()
            m.compute_graph_attributes(node, bond)
            m.add_metadata({'name': 'q', 'collision_energy': 40.0, 'instrument': INSTR, 'precursor_mode': MODES[0]},
                           setup_enc, rt_enc)
            m.fragment_MOL(depth=1)
            META['t_build'] += time.time() - t
            return m
        except Exception as e:
            META['n_graph_fail'] += 1
            err(stage='graph', smiles=smi, error=repr(e)[:200])
            return None

    def predict_batch(mets, cond):
        """[(metabolite)] at one (adduct, eV) condition -> [(mz, int) | None]; one batched forward pass."""
        adduct, ev = cond
        datas = []
        for m in mets:
            m.add_metadata({'name': 'q', 'collision_energy': float(ev), 'instrument': INSTR, 'precursor_mode': adduct},
                           setup_enc, rt_enc)
            datas.append(m.as_geometric_data(with_labels=False))
        t = time.time()
        res = []
        with torch.no_grad():
            batch = torch_geometric.data.Batch.from_data_list(datas).to(dev)
            out = model(batch, with_RT=False, with_CCS=False)
            probs, ptr = out['fragment_probs'].detach().cpu(), out['segment_ptr'].tolist()
        for k, m in enumerate(mets):
            try:
                setattr(m, 'compiled_probsALL_pred', probs[ptr[k]:ptr[k + 1]])
                p = sim.simulate_spectrum(m, 'compiled_probsALL_pred', precursor_mode=adduct,
                                          min_intensity=args.min_prob)
                pm, pi = np.asarray(p['mz'], float), np.asarray(p['intensity'], float)
                if not len(pi) or not np.isfinite(pm).all() or not np.isfinite(pi).all() or pi.sum() <= 0:
                    raise ValueError('invalid predicted spectrum')
                o = np.argsort(pm)
                res.append((pm[o], pi[o]))
                META['n_pred'] += 1
            except Exception as e:
                META['n_pred_fail'] += 1
                err(stage='predict', smiles=m.SMILES, cond=list(cond), error=repr(e)[:200])
                res.append(None)
        META['t_pred'] += time.time() - t
        return res

    last = time.time()
    for it in items:
        if time.time() - T0 > args.budget:
            META['status'] = 'timeout'
            log('budget reached; stopping')
            break
        mid = str(it['mid'])
        specs = []
        for sp in it.get('spectra') or []:
            META['n_spectra'] += 1
            if sp.get('adduct') not in MODES:
                META['n_spectra_unsupported'] += 1
                continue
            mz, ii = clean(sp.get('mz') or [], sp.get('it') or [], float(sp.get('prec') or float('nan')))
            if not len(mz):
                continue
            specs.append(((sp['adduct'], energy(sp.get('ce_ev'))), mz, ii))
        if not specs or not it.get('cands'):
            continue
        META['n_mols_covered'] += 1
        conds = sorted({c for c, _, _ in specs}, key=lambda c: -sum(1 for x, _, _ in specs if x == c))[:args.max_conds]
        specs = [s for s in specs if s[0] in conds]
        cands = list(dict.fromkeys(str(c) for c in it['cands']))
        n_ok = 0
        for b0 in range(0, len(cands), args.batch):
            if time.time() - T0 > args.budget:
                break
            chunk = cands[b0:b0 + args.batch]
            mets = [(s, build(s)) for s in chunk]
            mets = [(s, m) for s, m in mets if m is not None]
            if not mets:
                continue
            pred = {}
            for c in conds:
                try:
                    for (s, _), p in zip(mets, predict_batch([m for _, m in mets], c)):
                        pred[(s, c)] = p
                except Exception as e:                    # batch failure (e.g. OOM): fall back to one at a time
                    err(stage='batch', mid=mid, cond=list(c), error=repr(e)[:200])
                    if dev != 'cpu':
                        torch.cuda.empty_cache()
                    for s, m in mets:
                        try:
                            pred[(s, c)] = predict_batch([m], c)[0]
                        except Exception as e2:
                            META['n_pred_fail'] += 1
                            err(stage='single', smiles=s, error=repr(e2)[:200])
                            pred[(s, c)] = None
            for s, _ in mets:
                ent, cos = [], []
                for c, mz, ii in specs:
                    p = pred.get((s, c))
                    if p is None:
                        ent = None
                        break
                    ent.append(entropy_sim(mz, ii, p[0], p[1]))
                    cos.append(sqrt_cosine(mz, ii, p[0], p[1]))
                if ent:
                    OUT[mid][s] = round(float(np.mean(ent)), 8); OUT_COS[mid][s] = round(float(np.mean(cos)), 8)
                    META['n_scored'] += 1; n_ok += 1
        META['n_mols_scored'] += int(n_ok > 0)
        if time.time() - last > args.checkpoint_s:
            last = time.time()
            dump()
            log(f"{META['n_mols_scored']}/{META['n_mols_covered']} mols scored, {META['n_scored']} cands, "
                f"{META['n_pred']} preds ({META['n_pred'] / max(1e-9, META['t_pred']):.1f}/s in model+simulate)")
    if META['status'] == 'running':
        META['status'] = 'ok'
    log('done', {k: META[k] for k in ('status', 'n_mols', 'n_mols_covered', 'n_mols_scored', 'n_cands', 'n_scored',
                                      'n_pred', 'n_pred_fail', 'n_graph_fail', 'n_spectra_unsupported',
                                      'n_ce_missing', 'n_ce_clamped')})
    dump()


if __name__ == '__main__':
    try:
        parent() if (args.workers > 1 and args.shard < 0) else main()
    except Exception as e:
        META['status'] = 'failed'
        err(stage='fatal', error=repr(e)[:500], tb=traceback.format_exc()[-1500:])
        log('FAILED', repr(e))
        try:
            dump()
        except Exception:
            pass
        sys.exit(1)
