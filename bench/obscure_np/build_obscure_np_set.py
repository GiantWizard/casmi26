"""Build an honest "obscure natural product" held-out test set.

Every molecule kept is absent from train.parquet under both the scorer identity
and a stricter parent identity (salt/charge/tautomer/stereo-stripped), so no
model trained on train.parquet (FPNets, GLACIER fine-tunes, model_views, fragnet,
rankers) has seen it. Output mimics the competition test layout so the existing
kernels can run on it unchanged.

Example (Mac):
    python bench/obscure_np/build_obscure_np_set.py \
        --train ~/casmi26/data/train.parquet \
        --train-structs ~/casmi26/data/train_structs.parquet \
        --spectra enveda180=~/casmi26/ext/enveda180.mgf \
                  massbank=~/casmi26/ext/MassBank_NISTformat.msp \
                  mona=~/casmi26/ext/MoNA-export-LC-MS-MS_Spectra.msp \
                  gnps=~/casmi26/ext/ALL_GNPS_NO_PROPOGATED.mgf \
        --coconut ~/casmi26/data/coconut.parquet \
        --candidate-db ~/casmi26/data/pubchem_coconut_keys.parquet \
        --popularity ~/casmi26/data/pubchem_popularity.parquet \
        --exclude-panel ~/casmi26/data/np_panel_keys.txt \
        --out ~/casmi26/bench/obscure_np_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    STANDARD_ADDUCTS, adduct_group, is_clean_answer, keys_parallel, load_key_set,
    normalize_adduct, parent_key, read_spectra, scorer_key,
)

# Library names that are NIST or NIST-derived; never let them into an eval set
# that could later feed training (D744470, D745832).
NIST_MARKERS = ("nist", "gnps-nist14", "nist14", "nist20", "nist23")


def np_likeness(smiles_list):
    """RDKit NP-likeness (Ertl 2008); higher = more NP-like. None if contrib missing."""
    try:
        from rdkit import Chem
        from rdkit.Contrib.NP_Score import npscorer
    except ImportError:
        return None
    model = npscorer.readNPModel()
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        out.append(npscorer.scoreMol(m, model) if m is not None else np.nan)
    return np.asarray(out)


def normalize_mode(raw, adduct):
    if isinstance(raw, str):
        r = raw.strip().lower()
        if r.startswith("pos") or r == "p" or r == "+":
            return "positive"
        if r.startswith("neg") or r == "n" or r == "-":
            return "negative"
    if adduct:
        return "positive" if adduct.endswith("+") else "negative"
    return None


def ce_array(raw: str, nce: bool, precursor_mz: float) -> np.ndarray:
    """All energies in a CE string as an eV array, like test.parquet ('20,40,60' -> [20, 40, 60]).

    NCE uses the usual Thermo approximation eV = NCE * m/z / 500. Unparseable -> empty array,
    which the kernels already treat as "no CE".
    """
    if not isinstance(raw, str):  # pandas 3 keeps NaN through astype(str)
        return np.zeros(0)
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", raw)]
    if nce and np.isfinite(precursor_mz):
        vals = [v * precursor_mz / 500.0 for v in vals]
    return np.asarray(vals, dtype=float)


def resample(df: pd.DataFrame, per_mol: float, seed: int) -> pd.DataFrame:
    """Draw spectra per molecule so the mean count matches the test (~3.03, D742055).

    Counts are Poisson(per_mol) clipped to [1, available], which keeps singletons
    present the way the hidden test has them, instead of a flat top-k cut.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("molecule_id", sort=False):
        k = int(np.clip(rng.poisson(per_mol), 1, len(g)))
        parts.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="train.parquet (needs a SMILES or inchikey column)")
    ap.add_argument("--train-structs", help="train_structs.parquet, merged into the 'seen' set")
    ap.add_argument("--extra-seen", nargs="*", default=[],
                    help="More structure tables any pipeline model trained on (e.g. MassSpecGym, "
                         "fine-tune sets). Their molecules are excluded too.")
    ap.add_argument("--spectra", nargs="+", required=True, help="name=path (.msp/.mgf/.parquet/.csv)")
    ap.add_argument("--coconut", help="COCONUT structures; membership counts as NP")
    ap.add_argument("--np-score-min", type=float, default=0.3,
                    help="NP-likeness cutoff for molecules not in --coconut (Ertl score)")
    ap.add_argument("--candidate-db", help="Structure keys of the retrieval candidate pool "
                                           "(PubChem ∪ COCONUT). Used to label class 2 vs 3.")
    ap.add_argument("--popularity", help="Table with smiles/inchikey + a 'popularity' column")
    ap.add_argument("--obscure-max-pop", type=float, default=None,
                    help="Popularity at or below this is 'obscure' (default: bottom half)")
    ap.add_argument("--exclude-panel", help="Keys to drop (e.g. the Enveda NP panel) so sets stay disjoint")
    ap.add_argument("--adducts", nargs="*", default=list(STANDARD_ADDUCTS))
    ap.add_argument("--instruments", nargs="*", default=None,
                    help="Keep only these instrument_type values (substring, case-insensitive), e.g. timsTOF")
    ap.add_argument("--min-peaks", type=int, default=3)
    ap.add_argument("--max-precursor", type=float, default=1500.0)
    ap.add_argument("--spectra-per-mol", type=float, default=3.03)
    ap.add_argument("--draws", type=int, default=5, help="Independent resampling draws")
    ap.add_argument("--max-molecules", type=int, default=None, help="Cap after filtering (random)")
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "funnel": {}}

    def step(name, n):
        log["funnel"][name] = int(n)
        print(f"{name:<38} {n:>9,}")

    # ---- seen structures --------------------------------------------------
    seen_s, seen_p = load_key_set(args.train)
    for extra in [args.train_structs, *args.extra_seen]:
        s, p = load_key_set(extra)
        seen_s |= s
        seen_p |= p
    step("seen scorer keys", len(seen_s))
    step("seen parent keys", len(seen_p))
    if not seen_p:
        print("WARNING: no SMILES in seen tables; parent-key (tautomer/salt) leak check is OFF")

    # ---- spectra ------------------------------------------------------------
    # Rows whose stored InChIKey14 is already seen are skipped while reading (memory); the
    # SMILES-based scorer/parent checks below still catch the ~1.6% whose stored key differs.
    spec = pd.concat([read_spectra(p, skip_ik14=seen_s) for p in args.spectra], ignore_index=True)
    step("spectra read", len(spec))

    lib = spec["library"].fillna("").str.lower() + " " + spec["source"].str.lower()
    spec = spec[~lib.str.contains("|".join(NIST_MARKERS))]
    step("after dropping NIST-derived", len(spec))

    spec["adduct"] = spec["adduct_raw"].map(normalize_adduct)
    spec = spec[spec["adduct"].isin(args.adducts)]
    step("after standard-adduct filter", len(spec))

    spec = spec[spec["instrument_type"] != "other"]
    step("after high-res instrument filter", len(spec))

    if args.instruments:
        inst = spec["instrument_type"].fillna("").str.lower()
        spec = spec[inst.apply(lambda x: any(i.lower() in x for i in args.instruments))]
        step("after instrument filter", len(spec))

    npk = spec["ms2_mzs"].map(len)
    spec = spec[(npk >= args.min_peaks) & (spec["precursor_mz"] <= args.max_precursor)]
    step("after peak/precursor filter", len(spec))

    # Structure identity, computed once per unique SMILES.
    uniq = spec["smiles"].dropna().unique()
    skey = keys_parallel(uniq, scorer_key)
    pkey = keys_parallel(uniq, parent_key)
    clean = {s: is_clean_answer(s) for s in uniq}
    spec["scorer_key"] = spec["smiles"].map(skey)
    spec["parent_key"] = spec["smiles"].map(pkey)
    spec = spec[spec["scorer_key"].notna() & spec["smiles"].map(clean).fillna(False)]
    step("after valid neutral single-fragment", len(spec))

    unseen = ~spec["scorer_key"].isin(seen_s) & ~spec["parent_key"].isin(seen_p)
    spec = spec[unseen]
    step("after unseen-in-train filter", len(spec))

    if args.exclude_panel:
        ex_s, ex_p = load_key_set(args.exclude_panel)
        spec = spec[~spec["scorer_key"].isin(ex_s) & ~spec["parent_key"].isin(ex_p)]
        step("after excluding panel", len(spec))

    # ---- one canonical structure per scorer key ----------------------------
    mols = (spec.groupby("scorer_key")
                .agg(smiles=("smiles", "first"), n_spectra=("record_id", "size"),
                     sources=("source", lambda x: ",".join(sorted(set(x)))),
                     adducts=("adduct", lambda x: ",".join(sorted(set(x)))))
                .reset_index())
    step("unique unseen molecules", len(mols))

    # ---- NP filter ----------------------------------------------------------
    coco_s, coco_p = load_key_set(args.coconut)
    mols["in_coconut"] = mols["scorer_key"].isin(coco_s) | mols["smiles"].map(parent_key).isin(coco_p)
    nps = np_likeness(mols["smiles"])
    mols["np_score"] = nps if nps is not None else np.nan
    if nps is None and not coco_s:
        raise SystemExit("Need --coconut or RDKit Contrib NP_Score to decide what counts as an NP")
    mols["is_np"] = mols["in_coconut"] | (mols["np_score"] >= args.np_score_min)
    mols = mols[mols["is_np"]].copy()
    step("unseen natural products", len(mols))

    # ---- class + obscurity labels -----------------------------------------
    if args.candidate_db:
        cand_s, _ = load_key_set(args.candidate_db)
        mols["cls"] = np.where(mols["scorer_key"].isin(cand_s), 2, 3)
    else:
        mols["cls"] = np.nan
    if args.popularity:
        pop = pd.read_parquet(args.popularity) if args.popularity.endswith(".parquet") else pd.read_csv(args.popularity)
        if "scorer_key" not in pop.columns:
            if "smiles" in pop.columns:
                pop["scorer_key"] = pop["smiles"].map(scorer_key)
            else:
                kcol = next(c for c in pop.columns if "inchikey" in c.lower())
                pop["scorer_key"] = pop[kcol].astype(str).str[:14]
        pop = pop.groupby("scorer_key")["popularity"].max()
        mols["popularity"] = mols["scorer_key"].map(pop).fillna(0.0)
        thr = args.obscure_max_pop if args.obscure_max_pop is not None else mols["popularity"].median()
        mols["obscure"] = mols["popularity"] <= thr
    else:
        mols["popularity"] = np.nan
        mols["obscure"] = np.nan

    if args.max_molecules and len(mols) > args.max_molecules:
        mols = mols.sample(args.max_molecules, random_state=args.seed)
        step("after --max-molecules cap", len(mols))

    mols = mols.sort_values("scorer_key").reset_index(drop=True)
    # Same id shapes as test.parquet (m_xxxxxx / s_xxxxxxxx); stable across rebuilds.
    mols["molecule_id"] = ["m_" + hashlib.md5(k.encode()).hexdigest()[:8] for k in mols["scorer_key"]]
    assert mols["molecule_id"].is_unique
    spec = spec.merge(mols[["scorer_key", "molecule_id"]], on="scorer_key")

    # ---- competition-shaped output -----------------------------------------
    spec["ionization_mode"] = [normalize_mode(m, a) for m, a in zip(spec["ionization_mode"], spec["adduct"])]
    spec["ms2_normalized_intensities"] = spec["ms2_intensities"].map(
        lambda a: a / a.max() if len(a) and a.max() > 0 else a)
    spec["base_peak_intensity"] = spec["ms2_intensities"].map(lambda a: float(a.max()) if len(a) else np.nan)
    spec["collision_energy_orig"] = spec["collision_energy_raw"].map(lambda v: v if isinstance(v, str) else "")
    is_nce = spec["collision_energy_orig"].str.contains("%|nce|hcd", case=False)
    spec["collision_energy_orig_units"] = "eV"  # test.parquet is all eV; NCE is converted below
    spec["collision_energy_ev"] = [
        ce_array(raw, nce, pmz) for raw, nce, pmz in
        zip(spec["collision_energy_orig"], is_nce, spec["precursor_mz"])]
    test_cols = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
                 "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
                 "precursor_mz", "collision_energy_ev", "collision_energy_orig",
                 "collision_energy_orig_units"]

    full = spec.copy()
    full["spectrum_id"] = ["s_" + hashlib.md5(f"{a}:{b}:{i}".encode()).hexdigest()[:8]
                           for i, (a, b) in enumerate(zip(full["source"], full["record_id"]))]
    full["record_ref"] = full["source"] + ":" + full["record_id"].astype(str)
    assert full["spectrum_id"].is_unique
    full[test_cols + ["source", "record_ref", "scorer_key"]].to_parquet(out / "spectra_all.parquet", index=False)

    for d in range(args.draws):
        draw = resample(full, args.spectra_per_mol, args.seed + d)
        draw[test_cols].to_parquet(out / f"test_draw{d}.parquet", index=False)
    step("spectra in pool", len(full))

    # Per-molecule truth + slice labels. adduct_group uses the full pool so a molecule
    # is 'ft_only' only if no base-GLACIER adduct exists anywhere for it.
    has_base = full.groupby("molecule_id")["adduct"].agg(lambda x: any(adduct_group(a) == "base_glacier" for a in x))
    mols["adduct_group"] = mols["molecule_id"].map(has_base).map({True: "base_glacier", False: "ft_only"})
    tims = full.groupby("molecule_id")["instrument_type"].agg(
        lambda x: any("tims" in str(v).lower() for v in x))
    mols["timstof"] = mols["molecule_id"].map(tims)
    mols.to_csv(out / "solution.csv", index=False)
    pd.DataFrame({"molecule_id": mols["molecule_id"], "smiles": ";".join(["CCO"] * 25)}).to_csv(
        out / "sample_submission.csv", index=False)

    # Summary
    summ = {
        "molecules": len(mols),
        "spectra_pool": len(full),
        "by_source": full.groupby("source")["molecule_id"].nunique().to_dict(),
        "by_adduct_group": mols["adduct_group"].value_counts().to_dict(),
        "by_class": mols["cls"].value_counts(dropna=False).to_dict() if args.candidate_db else None,
        "obscure_share": float(mols["obscure"].mean()) if args.popularity else None,
        "by_adduct_spectra": full["adduct"].value_counts().to_dict(),
    }
    log["summary"] = summ
    (out / "build_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
