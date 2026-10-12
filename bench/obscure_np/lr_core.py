"""Learned final rerank: feature rows for one molecule's fused list, shared by offline training and the notebook.

Everything here must match what the notebook has at cell 19: the main list (v4, after the PubChem gate), engine 2's
list, the PubChem channel's list, the simulator score dicts and pool popularity keyed like the fused keys.
"""
import math

import numpy as np
import pandas as pd

FEATS = ["fused", "fused_rank", "r_main", "r_eng", "r_pc", "pop", "agree", "n_cand", "pc_S",
         "ice", "gl", "glft", "ice_cov", "gl_cov", "glft_cov", "ice_minus_max", "gl_minus_max", "glft_minus_max",
         "ice_z", "gl_z", "glft_z", "ice_grank", "gl_grank", "glft_grank", "grp_size", "grp_rank",
         "fused_minus_max", "pop_minus_max", "is_top_group"]
SIMS = ("ice", "gl", "glft")


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def molecule_rows(mid, main_smis, main_keys, eng_keys, pcl, fsm, fk, fsc, forms, sims, pop):
    """Rows for one molecule. main_keys: skey of each main-list SMILES; pcl: PubChem-channel dict for the molecule;
    fsm/fk/fsc: fused SMILES/keys/scores in fused order; forms: formula per fused SMILES;
    sims: {'ice'|'gl'|'glft': {smiles_or_key: score}} for this molecule; pop: key -> popularity."""
    vrank = {k: i + 1 for i, k in enumerate(main_keys)}
    erank = {k: i + 1 for i, k in enumerate(eng_keys) if k}
    prank = {k: i + 1 for i, k in enumerate((pcl or {}).get("pc_keys", [])) if k}
    agree = int(bool(main_keys and eng_keys and main_keys[0] == eng_keys[0]))
    rows = []
    for i, (smi, k, f, form) in enumerate(zip(fsm, fk, fsc, forms)):
        r = dict(molecule_id=mid, smiles=smi, key=k, fused=f, fused_rank=i + 1, form=form,
                 r_main=vrank.get(k, 99), r_eng=erank.get(k, 99), r_pc=prank.get(k, 99),
                 pop=pop.get(k, np.nan), agree=agree, n_cand=len(fsm), pc_S=_num((pcl or {}).get("S")))
        for name in SIMS:
            d = sims.get(name) or {}
            v = d.get(smi)
            if v is None:
                v = d.get(k)
            r[name] = _num(v)
        rows.append(r)
    return rows


def add_group_feats(df):
    g = df.groupby(["molecule_id", "form"])
    df["grp_size"] = g.smiles.transform("size")
    df["grp_rank"] = g.fused_rank.rank(method="first")
    df["fused_minus_max"] = df.fused - g.fused.transform("max")
    for name in SIMS:
        df[f"{name}_cov"] = df[name].notna().astype(int)
        mx, mu, sd = g[name].transform("max"), g[name].transform("mean"), g[name].transform("std")
        df[f"{name}_minus_max"] = df[name] - mx
        df[f"{name}_z"] = (df[name] - mu) / sd.replace(0, np.nan)
        df[f"{name}_grank"] = g[name].rank(ascending=False, method="min")
    df["pop_minus_max"] = df["pop"] - g["pop"].transform("max")
    df["is_top_group"] = (df.form == df.groupby("molecule_id").form.transform("first")).astype(int)
    return df


def slot_order(g, scores):
    """Reorder one molecule's fused list (g in fused order) inside each formula group by model score,
    putting group members back into that group's positions. Returns the list of SMILES."""
    g = g.assign(m=scores)
    order = list(g.smiles)
    for _, gg in g.groupby("form"):
        slots = sorted(gg.fused_rank - 1)
        for slot, smi in zip(slots, gg.sort_values("m", ascending=False, kind="stable").smiles):
            order[slot] = smi
    return order
