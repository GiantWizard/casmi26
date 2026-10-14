"""Build kernels/jc-netgen: jc-c2fix + network-guided analog generation (netgen.py) between the final fusion (cell 19)
and the metric audit, so the audit still validates every inserted SMILES. CPU only.

The generation code ships as the private dataset giantwizard/casmi26-jc-netgen-code (edits.py, netgen.py).
"""
import json, pathlib, sys
PROBE = len(sys.argv) > 1 and sys.argv[1] == "probe"

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = ROOT / "kernels/jc-c2fix/casmi26-jc-c2fix.ipynb"
nb = json.loads(src.read_text())
cell = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": r"""
# ---- jc-netgen: network-guided analog generation (CPU). Test molecules whose spectra share fragments are linked;
# single biotransformation edits of each neighbour's top-1 that match a molecule's mass become new candidates,
# scored by the engine FPNet + link strength and inserted at rank 1 (rank 2 when the own top-1 is a documented pool
# structure). Any failure keeps the fused lists unchanged.
import time as _t, json
PROBE_MODE = __PROBE__
_t0 = _t.time()
try:
    sys.path.insert(0, os.path.dirname(find('casmi26-jc-netgen-code/**/netgen.py')))
    import netgen as NG
    from casmi.spectra import merge_spectra as _ms
    _sub = pd.read_csv('submission.csv')
    _lists = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in _sub.values}
    _te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _te = _te[_te.molecule_id.astype(str).isin(_lists)]
    _ck = _FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')
    _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    _m = fpnet.load_model(_ck, _dev)
    _pop = {}
    for _k, _v in zip(map(str, P.key), POOL_POP):
        _pop[_k] = max(_pop.get(_k, 0.0), float(_v))
    _new, _st = NG.run(_te, _lists, _m, np.asarray(P.bits), chem, _pop, _dev, _ms, fpnet)
    if PROBE_MODE:   # LB probe: only the generated candidates of molecules netgen changed, 'CCO' elsewhere
        _rows = []
        for m, l in _new.items():
            _g = [x for x in l if x not in set(_lists.get(m, []))]
            _rows.append((m, ';'.join(_g[:2]) if _g else 'CCO'))
        _st['probe_rows'] = sum(1 for _, v in _rows if v != 'CCO')
    else:
        _rows = [(m, ';'.join(l[:25]) if l else 'CCO') for m, l in _new.items()]
    pd.DataFrame(_rows, columns=['molecule_id', 'smiles']).to_csv('submission.csv', index=False)
    _st['seconds'] = round(_t.time() - _t0)
    json.dump(_st, open('netgen_stats.json', 'w'))
    print('jc-netgen:', _st, flush=True)
    del _m
except Exception as _ex:
    import traceback; traceback.print_exc()
    json.dump({'failed': repr(_ex)[:500]}, open('netgen_stats.json', 'w'))
    print('jc-netgen FAILED -> fused lists kept:', repr(_ex), flush=True)
""".replace("__PROBE__", str(PROBE))}
cells = nb["cells"]
assert "Final metric audit" in "".join(cells[20]["source"]), "unexpected cell layout"
cells.insert(20, cell)
cells[0]["source"] = ("# jc-netgen = jc-c2fix + network-guided analog generation (CPU) before the metric audit\n\n"
                      + "".join(cells[0]["source"]))
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
slug = "casmi26-jc-netgen-probe" if PROBE else "casmi26-jc-netgen"
out = ROOT / "kernels" / slug.replace("casmi26-", ""); out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / "kernels/jc-c2fix/kernel-metadata.json").read_text())
meta.update(id=f"giantwizard/{slug}", title=slug, code_file=f"{slug}.ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + ["giantwizard/casmi26-jc-netgen-code"]
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", out)
