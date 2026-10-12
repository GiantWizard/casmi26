"""MRR@25 on the obscure-NP set, sliced, with paired bootstrap against a baseline.

Usage:
    python bench/obscure_np/mrr_eval.py --solution OUT/solution.csv \
        --sub jc-c2=preds/jc-c2_draw0.csv --sub cand=preds/cand_draw0.csv \
        --baseline jc-c2

Predictions may be wide (molecule_id + smiles_1..smiles_25, or any columns ending
in a rank number) or long (molecule_id, rank, smiles). Candidates are deduplicated
on the scorer key after ranking, as the metric does, so duplicate tails cost a slot
only in the way the grader would count them.

Rule from the handoff: nothing ships unless it beats jc-c2 here. A "win" below
means the bootstrap 90% interval of the paired difference is above zero.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from common import adduct_group, scorer_key

K = 25


def load_preds(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path) if not path.endswith(".parquet") else pd.read_parquet(path)
    if "rank" in df.columns:
        smi = next(c for c in df.columns if c.lower() in ("smiles", "pred", "candidate"))
        df = df.sort_values(["molecule_id", "rank"])
        return df.groupby("molecule_id")[smi].apply(list).to_dict()
    rank_cols = sorted((c for c in df.columns if re.search(r"\d+$", c) and c != "molecule_id"),
                       key=lambda c: int(re.search(r"(\d+)$", c).group(1)))
    return {r["molecule_id"]: [r[c] for c in rank_cols if isinstance(r[c], str)]
            for r in df.to_dict("records")}


def reciprocal_ranks(preds: dict[str, list[str]], truth: pd.Series, cache: dict) -> pd.Series:
    out = {}
    for mid, tkey in truth.items():
        seen, rank = set(), 0
        rr = 0.0
        for s in preds.get(mid, []):
            k = cache.get(s)
            if k is None and s not in cache:
                k = cache[s] = scorer_key(s)
            if k is None or k in seen:
                continue
            seen.add(k)
            rank += 1
            if rank > K:
                break
            if k == tkey:
                rr = 1.0 / rank
                break
        out[mid] = rr
    return pd.Series(out)


def bootstrap_diff(a: pd.Series, b: pd.Series, n=2000, seed=0):
    d = (a - b).to_numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(means, 5), np.percentile(means, 95)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--sub", nargs="+", required=True, help="name=path")
    ap.add_argument("--baseline", help="name of the --sub to compare others against")
    ap.add_argument("--test", help="The test_drawN.parquet the predictions were made on. Recomputes "
                                   "adduct_group from that draw, which is what routing actually saw.")
    args = ap.parse_args()

    sol = pd.read_csv(args.solution).set_index("molecule_id")
    if args.test:
        draw = pd.read_parquet(args.test, columns=["molecule_id", "adduct"])
        sol = sol.loc[sol.index.isin(draw["molecule_id"])].copy()
        has_base = draw.groupby("molecule_id")["adduct"].agg(
            lambda x: any(adduct_group(a) == "base_glacier" for a in x))
        sol["adduct_group"] = has_base.reindex(sol.index).map({True: "base_glacier", False: "ft_only"})
    truth = sol["scorer_key"]
    cache: dict = {}
    rr = {}
    for spec in args.sub:
        name, _, path = spec.partition("=")
        preds = load_preds(path)
        rr[name] = reciprocal_ranks(preds, truth, cache)
        missing = len(truth) - len(set(preds) & set(truth.index))
        if missing:
            print(f"warning: {name} has no predictions for {missing} molecules (scored 0)")
    rr = pd.DataFrame(rr)

    slices = {"all": pd.Series(True, index=sol.index)}
    for col in ("adduct_group", "cls", "obscure"):
        if col in sol.columns and sol[col].notna().any():
            for v in sorted(sol[col].dropna().unique(), key=str):
                slices[f"{col}={v}"] = sol[col] == v

    rows = []
    for sname, mask in slices.items():
        sub = rr[mask.reindex(rr.index).fillna(False)]
        row = {"slice": sname, "n": len(sub)}
        for name in rr.columns:
            row[name] = sub[name].mean()
            row[f"{name}@1"] = (sub[name] == 1).mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.baseline:
        print(f"\nPaired difference vs {args.baseline} (mean [90% bootstrap CI])")
        for sname, mask in slices.items():
            sub = rr[mask.reindex(rr.index).fillna(False)]
            if len(sub) < 10:
                continue
            for name in rr.columns:
                if name == args.baseline:
                    continue
                m, lo, hi = bootstrap_diff(sub[name], sub[args.baseline])
                fixed = int(((sub[name] == 1) & (sub[args.baseline] < 1)).sum())
                broke = int(((sub[name] < 1) & (sub[args.baseline] == 1)).sum())
                verdict = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "noise")
                print(f"  {sname:<26} {name:<14} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                      f"top1 fixed {fixed} / broke {broke}  {verdict}")


if __name__ == "__main__":
    main()
