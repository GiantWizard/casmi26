"""Build an in-pool timsTOF held-out set from train.parquet (the hidden test's dominant regime: timsTOF, answer in the
candidate pool, no library spectrum of the answer).

Picks train molecules with timsTOF spectra in the competition's adducts whose structure stays in both engines'
candidate pools once their train rows are gone. Default: pool_meta src == 3 (train AND COCONUT; engine 1's pool_meta
is fixed, engine 2's is COCONUT U ChEBI U train). --any-pool also takes train-only structures (src 1); the eval
notebook then appends holdout_structs.parquet to engine 2's pool. Their spectra become the test; every train row of
every structure with the same scorer key goes into holdout_ik14.txt, which the eval notebook
(kernels/make_inpool_eval.py) strips from train.parquet before either engine builds its library.

Caveat (why this is a HARM detector, not a gain estimate): the pipeline's learned parts (engine-2 fingerprint models,
FPNet, engine rankers, fine-tuned GLACIER) were trained on train.parquet, these spectra included, so the base list is
optimistic here. A lever that loses on this set breaks answers the base finds; one that wins may still be inflated.
Exclude MassSpecGym (ICEBERG/GLACIER training). Excluding the GLACIER fine-tune labels as well leaves ~40 molecules
(they cover nearly every timsTOF train structure), so the fine-tuned GLACIER is leaky here: compare with it off.

    ONP_KEY_CACHE=$S/dl/keycache python build_inpool_set.py --train $S/train/train.parquet \
        --train-structs $S/pool/train_structs.parquet --pool-meta $S/pool/pool_meta.parquet \
        --exclude $S/dl/MassSpecGym.tsv --any-pool --out $S/inpool
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import adduct_group, load_key_set  # noqa: E402

TEST_ADDUCTS = ["[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-"]  # visible test's
TEST_COLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct",
             "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_ev", "collision_energy_orig",
             "collision_energy_orig_units"]


def draw(df, per_mol, single_adduct, seed):
    """Poisson(per_mol) spectra per molecule clipped to [1, available]; with prob single_adduct all from one adduct
    (the visible test: 73% of molecules have one adduct, mean 3.03 spectra)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("molecule_id", sort=False):
        if rng.random() < single_adduct:
            vc = g.adduct.value_counts()
            g = g[g.adduct == rng.choice(vc.index, p=(vc / vc.sum()).values)]
        k = int(np.clip(rng.poisson(per_mol), 1, len(g)))
        parts.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--train-structs", required=True)
    ap.add_argument("--pool-meta", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="structure tables whose molecules may not be picked")
    ap.add_argument("--any-pool", action="store_true",
                    help="also train-only structures (pool_meta src 1); the eval notebook then adds the held-out "
                         "structures back to engine 2's pool (engine 1's pool_meta keeps them anyway)")
    ap.add_argument("--max-molecules", type=int, default=400)
    ap.add_argument("--min-peaks", type=int, default=3)
    ap.add_argument("--spectra-per-mol", type=float, default=3.03)
    ap.add_argument("--single-adduct", type=float, default=0.73)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = {}

    ts = pd.read_parquet(a.train_structs, columns=["inchikey14", "key", "smiles", "n_tims"])
    pm = pd.read_parquet(a.pool_meta, columns=["key", "src"])
    coco = set(pm.key[pm.src == 3])
    cand = ts[ts.key.isin(set(pm.key) if a.any_pool else coco) & (ts.n_tims > 0)]
    log["timstof_structs_in_pool"] = int(cand.key.nunique())
    ex_s, ex_p = set(), set()
    for p in a.exclude:
        s_, p_ = load_key_set(p); ex_s |= s_; ex_p |= p_
    cand = cand[~cand.key.isin(ex_s | ex_p)]
    log["after_exclusions"] = int(cand.key.nunique())

    # timsTOF rows of the candidate structures (pushdown on inchikey14)
    ik = sorted(set(cand.inchikey14))
    t = pq.read_table(a.train, filters=[("inchikey14", "in", ik), ("instrument_type", "=", "timsTOF")])
    sp = t.to_pandas()
    sp = sp[sp.adduct.astype(str).isin(TEST_ADDUCTS) & (sp.num_peaks >= a.min_peaks)]
    sp["scorer_key"] = sp.inchikey14.map(dict(zip(ts.inchikey14, ts.key)))
    keys = sorted(sp.scorer_key.dropna().unique())
    rng = np.random.default_rng(a.seed)
    if len(keys) > a.max_molecules:
        keys = sorted(rng.choice(keys, a.max_molecules, replace=False))
    sp = sp[sp.scorer_key.isin(keys)].copy()
    log["molecules"], log["spectra_pool"] = len(keys), len(sp)

    mid = {k: "m_" + hashlib.md5(f"inpool:{k}".encode()).hexdigest()[:8] for k in keys}
    sp["molecule_id"] = sp.scorer_key.map(mid)
    sp["spectrum_id"] = ["s_" + hashlib.md5(f"inpool:{i}".encode()).hexdigest()[:8] for i in sp.index]
    assert sp.spectrum_id.is_unique and len(set(mid.values())) == len(mid)
    for d in range(a.draws):
        dr = draw(sp, a.spectra_per_mol, a.single_adduct, a.seed + d)
        dr[TEST_COLS].to_parquet(out / f"test_draw{d}.parquet", index=False)
        log[f"draw{d}"] = dict(spectra=len(dr), adducts=dr.adduct.value_counts().to_dict())

    # every train structure that shares a held-out scorer key leaves the library
    hold = sorted(set(ts.inchikey14[ts.key.isin(keys)]))
    (out / "holdout_ik14.txt").write_text("\n".join(hold) + "\n")
    # structures engine 2 must keep proposing (its pool is COCONUT U ChEBI U train rows, which no longer have them)
    ts[ts.key.isin(keys)][["inchikey14", "smiles"]].to_parquet(out / "holdout_structs.parquet", index=False)
    log["holdout_ik14"] = len(hold)

    smi = sp.groupby("scorer_key").normalized_smiles.first()
    has_base = sp.groupby("scorer_key").adduct.agg(lambda x: any(adduct_group(v) == "base_glacier" for v in x))
    sol = pd.DataFrame({"scorer_key": keys, "smiles": [smi[k] for k in keys], "molecule_id": [mid[k] for k in keys],
                        "cls": 2, "obscure": False, "timstof": True,
                        "in_coconut": [k in coco for k in keys],
                        "adduct_group": [("base_glacier" if has_base[k] else "ft_only") for k in keys],
                        "n_spectra": sp.groupby("scorer_key").size().reindex(keys).values})
    sol.to_csv(out / "solution.csv", index=False)
    pd.DataFrame({"molecule_id": sol.molecule_id, "smiles": ";".join(["CCO"] * 25)}).to_csv(
        out / "sample_submission.csv", index=False)
    (out / "build_log.json").write_text(json.dumps(dict(args=vars(a), **log), indent=2, default=str))
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
