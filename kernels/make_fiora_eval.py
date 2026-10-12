"""Add FIORA scoring (offline replay only) to an already generated eval notebook: kernels/<src> -> kernels/<src>-fiora.

FIORA OS v1.0.0 (MIT code, weights trained on MSnLib v7 only, CC BY 4.0; Kaggle dataset
ahmedberatozer/casmi26-fiora-os-v1) is a third forward model next to ICEBERG / GLACIER. The new cell runs right
before the final two-engine fusion cell (after the GLACIER scoring and the gated PubChem merge) and scores every
candidate the final same-formula rerank can move, for all molecules with an [M+H]+ / [M-H]- spectrum:
members of same-formula groups (>= 2) in  main list (submission.csv) + engine top 40 + PubChem top 25 + the
ICEBERG / GLACIER input candidates. Scores go to fiora_work/fiora_out.json ({molecule_id: {smiles: score | None}},
spectral-entropy similarity; fiora_out_cos.json = sqrt cosine) + .meta.json. Nothing downstream reads them, so every
sub_* / final_* output is unchanged; bench/obscure_np/replay_fiora.py tunes the FIORA weight offline.

    python kernels/make_onp_eval.py --tag b --draw 0 --dataset giantwizard/casmi26-jc-onp-b   # if not generated yet
    python kernels/make_fiora_eval.py jc-onp-eval-b-d0 jc-onp-eval-d2
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIORA_DS = "ahmedberatozer/casmi26-fiora-os-v1"
ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="+", help="eval notebook dirs under kernels/, e.g. jc-onp-eval-b-d0")
ap.add_argument("--budget", type=float, default=3600.0, help="FIORA wall-clock budget in seconds")
ap.add_argument("--workers", type=int, default=0, help="FIORA worker processes (0 = min(4, cpus))")
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

RUNNER = (ROOT / "kernels" / "fiora_runner.py").read_text()

CELL = '''# ---- FIORA (offline replay only; added by kernels/make_fiora_eval.py) ----------------------------------------------
# Third forward model: FIORA OS v1.0.0 (MIT code, Nowatzky / BAM; weights trained on MSnLib v7 only, CC BY 4.0;
# dataset ahmedberatozer/casmi26-fiora-os-v1). Scores every candidate the final same-formula rerank can move
# (same-formula groups in main list + engine top 40 + PubChem top 25 + the ICEBERG/GLACIER input) on the molecule's
# [M+H]+ / [M-H]- spectra. Output: /kaggle/working/fiora_work/fiora_out.json. NOTHING below reads FIORA_SCORES, so the
# submission / sub_* / final_* files are exactly what they are without this cell. Any failure -> {} and we go on.
FIORA_SCORES, FIORA_BUDGET, FIORA_WORKERS, FIORA_BATCH = {}, __BUDGET__, __WORKERS__, __BATCH__
FIORA_RUNNER_SRC = __RUNNER__
try:
    import gc, subprocess
    from rdkit import Chem as _FCh
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _FCMF
    _t0 = time.time()
    _FROOT = os.path.dirname(os.path.dirname(find('casmi26-fiora-os-v1/**/fiora_OS_v1.0.0_state.pt')))
    _FW = '/kaggle/working/fiora_work'
    os.makedirs(_FW, exist_ok=True)
    with open(os.path.join(_FW, 'fiora_runner.py'), 'w') as _f:
        _f.write(FIORA_RUNNER_SRC)
    _fform = {}
    def _ff(s):
        if s not in _fform:
            try:
                _fform[s] = _FCMF(_FCh.MolFromSmiles(s))
            except Exception:
                _fform[s] = None
        return _fform[s]
    _v4 = pd.read_csv('submission.csv')                       # main list after the gated PubChem merge (cell 17)
    _icec = {str(_it['mid']): list(_it['cands']) for _it in (items if isinstance(globals().get('items'), list) else [])}
    _FMODES = ('[M+H]+', '[M-H]-', '[M]+', '[M]-')
    _fmids = set(te.loc[te.adduct.isin(_FMODES), 'molecule_id'].astype(str))
    _fc = {}
    for _m, _s in zip(_v4.molecule_id.astype(str), _v4.smiles.astype(str)):
        if _m not in _fmids:
            continue
        _e = (ENG.get(_m) or {}) if isinstance(ENG, dict) else {}
        _p = (PC.get(_m) or {}) if isinstance(PC, dict) else {}
        _pool = list(dict.fromkeys([x for x in _s.split(';') if x and x != 'CCO'] + list(_e.get('smiles', [])[:40]) +
                                   list(_p.get('pc', [])[:25]) + _icec.get(_m, [])))
        _fs = [_ff(x) for x in _pool]
        _cnt = {}
        for _x in _fs:
            _cnt[_x] = _cnt.get(_x, 0) + 1
        _keep = [x for x, f in zip(_pool, _fs) if f is not None and _cnt[f] >= 2]
        if _keep:
            _fc[_m] = _keep
    _fitems = [dict(_it, spectra=[s for s in _it['spectra'] if s.get('adduct') in _FMODES])
               for _it in ice_fuse.build_ice_input(te, _fc, covered_only=False)]
    _fitems = [_it for _it in _fitems if _it['spectra'] and _it['cands']]
    _fin, _fout = os.path.join(_FW, 'fiora_in.json'), os.path.join(_FW, 'fiora_out.json')
    with open(_fin, 'w') as _f:
        json.dump(_fitems, _f)
    print('FIORA input molecules', len(_fitems), 'candidates', sum(len(_it['cands']) for _it in _fitems),
          f'(built in {time.time() - _t0:.0f}s)', flush=True)
    gc.collect(); torch.cuda.empty_cache()
    _cmd = [sys.executable, os.path.join(_FW, 'fiora_runner.py'), _FROOT, _fin, _fout, '--device', 'cuda',
            '--budget', str(float(FIORA_BUDGET)), '--site', '/kaggle/working/fiora_site',
            '--workers', str(FIORA_WORKERS), '--batch', str(FIORA_BATCH)]
    try:
        _r = subprocess.run(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            timeout=FIORA_BUDGET + 600)
        print(_r.stdout[-3000:], flush=True)
    except subprocess.TimeoutExpired as _ex:
        print('FIORA TIMEOUT', str(_ex.stdout or '')[-2000:], flush=True)
    FIORA_SCORES = json.load(open(_fout)) if os.path.exists(_fout) else {}
    if os.path.exists(_fout + '.meta.json'):
        _fm = json.load(open(_fout + '.meta.json'))
        print('FIORA meta', {k: _fm.get(k) for k in ('status', 'worker_status', 'n_mols', 'n_mols_covered',
              'n_mols_scored', 'n_cands', 'n_scored', 'n_pred', 'n_pred_fail', 'n_graph_fail', 'n_spectra_unsupported',
              'n_ce_missing', 'n_ce_clamped', 'seconds', 'pred_per_s', 'device', 'torch', 'torch_geometric', 'rdkit')},
              flush=True)
    import glob as _glob
    for _x in _glob.glob(os.path.join(_FW, 'fiora_out.json.part*')):   # per-worker copies, already merged
        os.remove(_x)
    print('FIORA scored molecules', sum(1 for d in FIORA_SCORES.values() if any(v is not None for v in d.values())),
          f'| {time.time() - _t0:.0f}s', flush=True)
except Exception as _ex:
    FIORA_SCORES = {}
    print('FIORA FAILED (no effect on any output):', repr(_ex), flush=True)
'''


def build(src_name):
    src = ROOT / "kernels" / src_name
    ipynb = next(src.glob("*.ipynb"))
    nb = json.loads(ipynb.read_text())
    meta = json.loads((src / "kernel-metadata.json").read_text())
    cells = nb["cells"]
    S = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    fus = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and S(c).lstrip().startswith("ALPHA, KRR = 0.6, 3.0")]
    assert len(fus) == 1, fus
    k = fus[0]
    gl = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "gl_fuse.run_gl(GL_PKG, gl_items" in S(c)]
    assert gl and gl[0] < k, (gl, k)          # (the fusion cell also calls it, for the mix-GLACIER term)
    assert any("sub.to_csv('submission.csv'" in S(c) for c in cells[gl[0] + 1:k]), "gated merge not between GL and fusion"
    if cells[k - 1]["cell_type"] == "markdown":           # keep the '## 9. Two-engine fusion' header on its cell
        k -= 1
    code = (CELL.replace("__BUDGET__", repr(float(args.budget))).replace("__WORKERS__", repr(int(args.workers)))
            .replace("__BATCH__", repr(int(args.batch))).replace("__RUNNER__", repr(RUNNER)))
    compile(code, "fiora_cell", "exec")
    cells.insert(k, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code})
    cells.insert(k, {"cell_type": "markdown", "metadata": {}, "source":
                     "## 8b. FIORA scores (offline replay only)\n\nThird forward model (FIORA OS v1.0.0, MSnLib-only "
                     "weights) scores the same-formula candidate groups; the scores are saved to "
                     "`fiora_work/fiora_out.json` and not used by any output of this notebook."})
    slug_old = meta["id"].split("/")[1]
    slug = slug_old + "-fiora"
    cells[0]["source"] = (f"# {slug}: {slug_old} + FIORA scores saved for offline replay "
                          "(generated by kernels/make_fiora_eval.py; outputs identical to the source notebook)\n\n") + S(cells[0])
    for c in cells:
        if c["cell_type"] == "code":
            compile(S(c), f"{slug}-cell", "exec") if not S(c).lstrip().startswith(("!", "%")) else None
    out = ROOT / "kernels" / (src_name + "-fiora")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta.update(id=f"giantwizard/{slug}", title=slug, code_file=f"{slug}.ipynb")
    meta.pop("id_no", None)
    if FIORA_DS not in meta["dataset_sources"]:
        meta["dataset_sources"] = list(meta["dataset_sources"]) + [FIORA_DS]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out, "| FIORA cell at index", k + 1, "of", len(cells))


for s in args.src:
    build(s)
