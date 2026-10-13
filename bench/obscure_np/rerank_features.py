"""Per-candidate features for a learned final rerank, built from a finished eval run's saved outputs.

Rebuilds cell 19's fused list (main list + engine 2, weighted RRF, top FUSE_N) for each molecule and describes
every candidate with what the pipeline already knows about it: fused score and rank, rank in each engine,
PubChem-channel membership, spectrum-simulator scores (ICEBERG, GLACIER, fine-tuned GLACIER) raw and relative to
the candidate's same-formula group, group size, pool popularity, and molecule-level context.

    python rerank_features.py --run OUT_DIR --main v4_gentle.csv --lib REPLAY_LIB --pool-pop pool_pop.parquet \
        [--solution solution.csv] --out feats.parquet
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--main", required=True, help="pre-fusion main list (v4), e.g. the gentle-gate list")
ap.add_argument("--lib", required=True, help="dir with fuse.py, gl_fuse.py, casmi/chem.py")
ap.add_argument("--pool-pop", required=True, help="parquet: scorer_key, popularity")
ap.add_argument("--solution", help="adds label column (candidate is the truth)")
ap.add_argument("--fiora", help="fiora_out.json from kernels/fiora_runner.py (adds FIORA features)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.lib)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casmi import chem  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

from common import keys_parallel, scorer_key  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ALPHA, KRR, FUSE_N = 0.6, 3.0, 80
R = a.run
load = lambda p: json.load(open(os.path.join(R, p))) if os.path.exists(os.path.join(R, p)) else {}
ENG = load("eng_lists.json")
PC = load("pc_lists.json")
SC = {"ice": load("ice_work/ice_out.json"), "gl": load("gl_work/gl_out.json"), "glft": load("glft_work/gl_out.json")}
if a.fiora:
    SC["fiora"] = json.load(open(a.fiora))
POP = pd.read_parquet(a.pool_pop).groupby("scorer_key").popularity.max().to_dict()
main = pd.read_csv(a.main)

_form = {}


def formula(s):
    if s not in _form:
        try:
            _form[s] = CalcMolFormula(Chem.MolFromSmiles(s))
        except Exception:
            _form[s] = s
    return _form[s]


_sk = {}


def skey(s):
    if s not in _sk:
        _sk[s] = chem.score_key(s) or s
    return _sk[s]


def fuse2(v, e_smis, e_keys):
    sc, smi_of = {}, {}
    for r, s in enumerate(v, 1):
        k = skey(s); sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if k:
            sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:FUSE_N]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]


def num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


rows = []
for mid, s in zip(main.molecule_id, main.smiles):
    v = [x for x in str(s).split(";") if x and x != "CCO"]
    e = ENG.get(str(mid)) or {}
    es, ek = e.get("smiles", []), e.get("keys", [])
    fsm, fk, fsc = fuse2(v, es, ek)
    if not fsm:
        continue
    vrank = {skey(x): i + 1 for i, x in enumerate(v)}
    erank = {k: i + 1 for i, k in enumerate(ek) if k}
    pcl = (PC.get(mid) or {})
    prank = {k: i + 1 for i, k in enumerate(pcl.get("pc_keys", [])) if k}
    agree = bool(v and ek and skey(v[0]) == ek[0])
    for i, (smi, k, f) in enumerate(zip(fsm, fk, fsc)):
        r = dict(molecule_id=mid, smiles=smi, key=k, fused=f, fused_rank=i + 1, form=formula(smi),
                 r_main=vrank.get(k, 99), r_eng=erank.get(k, 99), r_pc=prank.get(k, 99),
                 pop=POP.get(k, np.nan), agree=int(agree), n_cand=len(fsm), pc_S=num(pcl.get("S")))
        for name, d in SC.items():
            dm = d.get(str(mid), {}) if isinstance(d, dict) else {}
            val = dm.get(smi)
            if val is None:
                val = dm.get(k)
            r[name] = num(val)
        rows.append(r)
df = pd.DataFrame(rows)

# group-relative features (same formula within a molecule's fused list = the isomer race)
g = df.groupby(["molecule_id", "form"])
df["grp_size"] = g.smiles.transform("size")
df["grp_rank"] = g.fused_rank.rank(method="first")
df["fused_minus_max"] = df.fused - g.fused.transform("max")
for name in SC:
    df[f"{name}_cov"] = df[name].notna().astype(int)
    mx, mu, sd = g[name].transform("max"), g[name].transform("mean"), g[name].transform("std")
    df[f"{name}_minus_max"] = df[name] - mx
    df[f"{name}_z"] = ((df[name] - mu) / sd.replace(0, np.nan))
    df[f"{name}_grank"] = g[name].rank(ascending=False, method="min")
df["pop_minus_max"] = df["pop"] - g["pop"].transform("max")
df["is_top_group"] = (df.form == df.groupby("molecule_id").form.transform("first")).astype(int)

if a.solution:
    sol = pd.read_csv(a.solution).set_index("molecule_id").scorer_key
    keys = keys_parallel(df.smiles.unique(), scorer_key)
    df["label"] = (df.smiles.map(keys) == df.molecule_id.map(sol)).astype(int)
df.to_parquet(a.out, index=False)
print(f"{len(df):,} candidate rows, {df.molecule_id.nunique()} molecules"
      + (f", truth present for {int(df.groupby('molecule_id').label.max().sum())}" if a.solution else ""))
