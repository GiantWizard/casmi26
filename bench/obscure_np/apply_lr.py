"""Apply a saved learned-rerank model (lr_model.txt) to rerank_features.py output, as the jc-lr notebooks do
(slots mode: reorder only inside same-formula groups). Writes a competition-format submission.

    python apply_lr.py --feats feats.parquet --model lr_model.txt --out lr.csv
"""
import argparse
import os
import sys

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--feats", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = lgb.Booster(model_file=a.model)
names = list(m.feature_name())
df = pd.read_parquet(a.feats).sort_values(["molecule_id", "fused_rank"])
out = [(mid, ";".join(lr_core.slot_order(g, m.predict(g[names]))[:25])) for mid, g in df.groupby("molecule_id", sort=False)]
pd.DataFrame(out, columns=["molecule_id", "smiles"]).to_csv(a.out, index=False)
print("reranked", len(out), "molecules with", len(names), "features ->", a.out)
