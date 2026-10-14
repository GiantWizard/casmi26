"""Generate kernels/jc-gen: jc-gate + a second, wider class-3 generation pass in engine 1 (cell 13).

Engine 1 builds class-3 candidates by applying biosynthetic transforms to library analogs of the query
(casmi.engine.Engine.generate). The shipped EngineCfg uses at most 6 parent analogs and 60 / 150 products per
parent / molecule (median 7 generated candidates per molecule). jc-gen keeps the shipped engine-1 list exactly and
runs engine 1 a second time with wider generation (GEN_WIDE); generated structures that only the wide pass proposes
are inserted at the positions the wide pass ranks them, every other entry keeps its order. (Ranking the wide pass
directly also perturbs pool candidates through the ranker's list-context features: in-pool -0.006.)

Measured on the Enveda natural-product panel (250 timsTOF NPs = the hidden test's chemistry; FPNet A, which held
the panel out; CPU replay with bench/obscure_np/probes/engine1_panel/*):
  answer removed from the pool, no reference spectra (class-3-like): engine-1 MRR 0.361 -> 0.430
      (+0.070, 90% CI +0.047..+0.096; 31 molecules better, 0 worse)
  answer in the pool, no reference spectra (class 2): 0.971 -> 0.970 (-0.0004; 2 molecules one rank lower)

    python kernels/make_gen_notebook.py [--src jc-gate] [--out jc-gen]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDE = dict(gen_n_analog=20, gen_min_sim=0.2, gen_max_per_parent=200, gen_max_total=600)
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="jc-gate")
ap.add_argument("--out", default="jc-gen")
args = ap.parse_args()
src = ROOT / "kernels" / args.src
meta = json.loads((src / "kernel-metadata.json").read_text())
nb = json.loads((src / meta["code_file"]).read_text())
cells = nb["cells"]


def must_replace(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


i13 = next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
           and "C, X, F, names, info = V.run(spectra, target)" in "".join(c["source"]))
s = "".join(cells[i13]["source"])
s = must_replace(s, "BASE, n_err = {}, 0\n", f"""BASE, n_err = {{}}, 0
# ---- jc-gen: second engine-1 pass with wider class-3 generation; only its NEW generated structures are inserted ----
GEN_WIDE = {WIDE!r}
GEN_DEFAULT = {{k: getattr(E.cfg, k) for k in GEN_WIDE}}
n_gen2, n_gen2_ins, n_gen2_err = 0, 0, 0


def gen2_combine(base, wide, n):
    \"\"\"base / wide: lists of (smiles, key, score, formula, pid) in rank order (deduplicated by key).
    Walk the wide list; a generated key absent from the base list takes its wide position, otherwise the next
    base entry fills the slot. Base order is kept exactly.\"\"\"
    bk = set(x[1] for x in base); out, used, bi, ins = [], set(), 0, 0
    for w in wide[:n]:
        if len(out) >= n:
            break
        if w[4] < 0 and w[1] not in bk and w[1] not in used:
            out.append(w); used.add(w[1]); ins += 1
            continue
        while bi < len(base) and base[bi][1] in used:
            bi += 1
        if bi < len(base):
            out.append(base[bi]); used.add(base[bi][1]); bi += 1
    while len(out) < n and bi < len(base):
        if base[bi][1] not in used:
            out.append(base[bi]); used.add(base[bi][1])
        bi += 1
    return out, ins


""")
s = must_replace(s, """                    if len(smis) >= TOPN: break
        except Exception as e:""", """                    if len(smis) >= TOPN: break
                try:                                     # jc-gen second pass (falls back to the shipped list)
                    for _k, _v in GEN_WIDE.items(): setattr(E.cfg, _k, _v)
                    C2, X2, F2, n2, _ = V.run(spectra, target)
                    if len(C2.pid):
                        s2 = rank_score(np.concatenate([X2, F2], 1), list(FEATURES) + list(n2))
                        wide, seen2 = [], set()
                        for i in np.argsort(-s2, kind='stable'):
                            k = C2.key[i]
                            if k in seen2: continue
                            seen2.add(k); wide.append((C2.smiles[i], k, float(s2[i]), C2.formula[i], int(C2.pid[i])))
                            if len(wide) >= TOPN: break
                        comb, ins = gen2_combine(list(zip(smis, keys, scs, forms, pids_)), wide, TOPN)
                        smis, keys, scs, forms, pids_ = [list(t) for t in zip(*comb)]
                        n_gen2 += 1; n_gen2_ins += ins
                except Exception as e2:
                    n_gen2_err += 1; print('jc-gen second pass failed', mid, repr(e2))
                finally:
                    for _k, _v in GEN_DEFAULT.items(): setattr(E.cfg, _k, _v)
        except Exception as e:""")
s = must_replace(s, "print('errors', n_err, '| popularity re-ranked lists', n_pp)",
                 "print('errors', n_err, '| popularity re-ranked lists', n_pp)\n"
                 "print('jc-gen: wide pass on', n_gen2, 'molecules,', n_gen2_ins, 'generated structures inserted,', n_gen2_err, 'failures')")
cells[i13]["source"] = s

# self-test of the combine rule in the cheap pass, right after cell 13's definitions are available
cells[0]["source"] = ("# jc-gen = jc-gate + second engine-1 pass with wider class-3 generation "
                      f"({', '.join(f'{k}={v}' for k, v in WIDE.items())}); only its new generated structures are "
                      "inserted at their ranked positions\n\n") + "".join(cells[0]["source"])
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        compile("".join(c["source"]), "cell", "exec")
out = ROOT / "kernels" / args.out
out.mkdir(exist_ok=True)
slug = f"casmi26-{args.out}"
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta.update(id=f"giantwizard/{slug}", title=slug, code_file=f"{slug}.ipynb")
meta.pop("id_no", None)
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", out)
