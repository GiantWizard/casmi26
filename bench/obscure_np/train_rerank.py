"""Train a small LightGBM ranker on rerank_features.py output and write reranked submissions for a test run.

Modes: 'slots' reorders only inside each same-formula group and puts members back into that group's positions
(like the pipeline's simulator rerank); 'free' sorts the whole fused list by model score.

    python train_rerank.py --train feat_vis.parquet --test feat_d0.parquet --out DIR [--folds 2]
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATS = ["fused", "fused_rank", "r_main", "r_eng", "r_pc", "pop", "agree", "n_cand", "pc_S",
         "ice", "gl", "glft", "ice_cov", "gl_cov", "glft_cov", "ice_minus_max", "gl_minus_max", "glft_minus_max",
         "ice_z", "gl_z", "glft_z", "ice_grank", "gl_grank", "glft_grank", "grp_size", "grp_rank",
         "fused_minus_max", "pop_minus_max", "is_top_group"]
PARAMS = dict(objective="lambdarank", metric="map", eval_at=[1], learning_rate=0.05, num_leaves=7,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=26, lambdarank_truncation_level=10)


def fit(df, rounds):
    df = df[df.groupby("molecule_id").label.transform("max") == 1].sort_values(["molecule_id", "fused_rank"])
    ds = lgb.Dataset(df[FEATS], df.label, group=df.groupby("molecule_id", sort=False).size().values)
    return lgb.train(PARAMS, ds, rounds)


def rerank(df, model, mode):
    df = df.copy()
    df["m"] = model.predict(df[FEATS])
    out = []
    for mid, g in df.sort_values("fused_rank").groupby("molecule_id", sort=False):
        if mode == "free":
            order = g.sort_values("m", ascending=False).smiles.tolist()
        else:
            order = list(g.smiles)
            for _, gg in g.groupby("form"):
                slots = sorted(gg.fused_rank - 1)
                for slot, smi in zip(slots, gg.sort_values("m", ascending=False).smiles):
                    order[slot] = smi
        out.append((mid, ";".join(order[:25])))
    return pd.DataFrame(out, columns=["molecule_id", "smiles"])


def mrr(sub, df):
    """Mean over every molecule in df (a molecule missing from sub scores 0)."""
    truth = df[df.label == 1].groupby("molecule_id").smiles.first()
    lists = dict(sub.values)
    rr = []
    for mid in df.molecule_id.unique():
        lst = lists.get(mid, "").split(";")
        t = truth.get(mid)
        rr.append(1.0 / (lst.index(t) + 1) if t in lst else 0.0)
    return float(np.mean(rr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--folds", type=int, default=0, help="k-fold within --test instead of training on --train")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    te = pd.read_parquet(a.test)
    base = te.sort_values("fused_rank").groupby("molecule_id").smiles.apply(lambda s: ";".join(s[:25])).reset_index()
    print(f"fused-order proxy MRR (before any simulator rerank): {mrr(base, te):.4f}")
    if a.folds:
        mids = np.array(sorted(map(str, te.molecule_id.unique())), dtype=object)
        rng = np.random.default_rng(26); rng.shuffle(mids)
        parts = {m: i % a.folds for i, m in enumerate(mids)}
        te["fold"] = te.molecule_id.map(parts)
        tr_extra = pd.concat([pd.read_parquet(p) for p in a.train]) if a.train != ["none"] else None
        for mode in ("slots", "free"):
            subs = []
            for f in range(a.folds):
                tr = te[te.fold != f]
                if tr_extra is not None:
                    tr = pd.concat([tr, tr_extra])
                subs.append(rerank(te[te.fold == f], fit(tr, a.rounds), mode))
            sub = pd.concat(subs)
            sub.to_csv(os.path.join(a.out, f"lr_{mode}.csv"), index=False)
            print(f"{mode:<5} {a.folds}-fold MRR {mrr(sub, te):.4f}")
    else:
        tr = pd.concat([pd.read_parquet(p) for p in a.train])
        model = fit(tr, a.rounds)
        for mode in ("slots", "free"):
            sub = rerank(te, model, mode)
            sub.to_csv(os.path.join(a.out, f"lr_{mode}.csv"), index=False)
            print(f"{mode:<5} MRR {mrr(sub, te):.4f}")
        imp = pd.Series(model.feature_importance("gain"), FEATS).sort_values(ascending=False)
        print("top features:", imp.head(8).round(0).to_dict())
