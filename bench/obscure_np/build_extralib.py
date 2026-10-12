"""Build extra reference spectra in train.parquet format ("extra library" lever).

Takes the host-approved public libraries (Enveda-180 Zenodo CC BY 4.0, GNPS, MassBank), keeps
high-resolution spectra with a train-vocabulary adduct whose structure is absent from
train.parquet (scorer key AND parent key, as in build_obscure_np_set.py), and writes them with
exactly train.parquet's schema so the pipeline can concatenate them onto train.parquet.

NIST-derived records (e.g. GNPS-NIST14-MATCHES) are always dropped.

Two outputs:
  extra_full.parquet  every kept row (for a final submission)
  extra_eval.parquet  same minus every molecule of the offline eval sets (--exclude solution.csv
                      files, scorer + parent key), so evaluation answers never enter the library.

    ONP_KEY_CACHE=$S/dl/keycache python bench/obscure_np/build_extralib.py \
        --train $S/train/train.parquet --train-structs $S/pool/train_structs.parquet \
        --spectra enveda180=$S/e180/f.parquet gnps=$S/dl/gnps.mgf massbank=$S/dl/massbank.msp \
        --exclude $S/dl/onp/solution.csv $S/dl/onp_b/solution.csv --out $S/xlib
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (is_clean_answer, keys_parallel, load_key_set, normalize_adduct,  # noqa: E402
                    parent_key, read_spectra, scorer_key)

NIST_MARKERS = ("nist", "gnps-nist14", "nist14", "nist20", "nist23")
LIB_NAME = {"enveda180": "enveda-180", "gnps": "gnps", "massbank": "massbank"}   # must be in casmi.build.LIBS

# extra spellings on top of common.normalize_adduct (train.parquet carries these dimers too)
_EXTRA_ALIASES = {
    "[2M+H]+": ["2M+H", "[2M+H]", "[2M+H]+", "[2M+H]1+"],
    "[2M+Na]+": ["2M+Na", "[2M+Na]", "[2M+Na]+", "[2M+Na]1+"],
    "[2M-H]-": ["2M-H", "[2M-H]", "[2M-H]-", "[2M-H]1-"],
    "[M+CH2O2-H]-": ["[M+FA-H]-", "M+FA-H"],
    "[M+C2H4O2-H]-": ["[M+Hac-H]-", "[M+CH3COO]-", "M+CH3COO", "[M+C2H3O2]-"],
}
_EXTRA = {a.upper().replace(" ", ""): k for k, v in _EXTRA_ALIASES.items() for a in v}

E, PROTON = 0.00054857990946, 1.007276466812
MASS = dict(H=1.00782503223, C=12.0, N=14.00307400443, O=15.99491461957, Na=22.9897692820, K=38.9637064864,
            Cl=34.968852682)
H2O = 2 * MASS["H"] + MASS["O"]
FORMATE = MASS["C"] + 2 * MASS["H"] + 2 * MASS["O"]
ACETATE = 2 * MASS["C"] + 4 * MASS["H"] + 2 * MASS["O"]
NH4 = MASS["N"] + 4 * MASS["H"]
ADDUCTS = {  # (n_M, z, delta) as in the engine-2 pv.ADDUCTS table
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS["Na"] - E),
    "[M+K]+": (1, 1, MASS["K"] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS["Cl"] + E),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS["Na"] - E), "[2M-H]-": (2, 1, -PROTON),
}


def norm_adduct(raw):
    a = normalize_adduct(raw)
    if a is None and isinstance(raw, str):
        a = _EXTRA.get(raw.strip().upper().replace(" ", ""))
    return a


def mol_info(smiles):
    """(normalized_smiles, inchikey, formula, exact mass): stereo removed, like train.parquet."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        Chem.RemoveStereochemistry(m)
        ik = Chem.MolToInchiKey(m)
        if not ik:
            return None
        return Chem.MolToSmiles(m), ik, rdMolDescriptors.CalcMolFormula(m), float(Descriptors.ExactMolWt(m))
    except Exception:
        return None


def ce_parse(raw, prec):
    if not isinstance(raw, str) or not raw.strip():
        return np.zeros(0), "", "unknown"
    nce = bool(re.search(r"%|nce|hcd", raw, re.I))
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", raw)]
    if not vals:
        return np.zeros(0), raw, "unknown"
    if nce and np.isfinite(prec):
        return np.asarray([v * prec / 500.0 for v in vals]), raw, "NCE"
    return np.asarray(vals, float), raw, "eV"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--train-structs", required=True)
    ap.add_argument("--spectra", nargs="+", required=True, help="name=path; name in enveda180/gnps/massbank")
    ap.add_argument("--exclude", nargs="*", default=[], help="eval solution.csv files (scorer_key, smiles)")
    ap.add_argument("--min-peaks", type=int, default=3)
    ap.add_argument("--max-precursor", type=float, default=1500.0)
    ap.add_argument("--max-ppm", type=float, default=50.0, help="drop rows whose precursor misfits the structure")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    log = {}

    def step(name, n):
        log[name] = int(n); print(f"{name:<40} {n:>9,}", flush=True)

    seen_s, seen_p = load_key_set(args.train)
    s2, p2 = load_key_set(args.train_structs)
    seen_s |= s2; seen_p |= p2
    step("seen scorer keys", len(seen_s)); step("seen parent keys", len(seen_p))

    parts = []
    for p in args.spectra:
        d = read_spectra(p, skip_ik14=seen_s)
        if "inchikey" in d:  # mgf/msp are not pre-filtered while reading; shrink early
            d = d[~d["inchikey"].astype(str).str[:14].isin(seen_s)]
        parts.append(d); print("read", p, len(d), flush=True)
    spec = pd.concat(parts, ignore_index=True); del parts
    step("spectra read (stored key not in train)", len(spec))
    lib = spec["library"].fillna("").astype(str).str.lower() + " " + spec["source"].str.lower()
    spec = spec[~lib.str.contains("|".join(NIST_MARKERS))]
    step("after dropping NIST-derived", len(spec))
    spec["adduct"] = spec["adduct_raw"].map(norm_adduct)
    spec = spec[spec["adduct"].isin(ADDUCTS)]
    step("after adduct filter", len(spec))
    spec = spec[spec["instrument_type"] != "other"]
    step("after high-res instrument filter", len(spec))
    spec = spec[(spec["ms2_mzs"].map(len) >= args.min_peaks) & (spec["precursor_mz"] <= args.max_precursor)]
    step("after peak/precursor filter", len(spec))

    uniq = spec["smiles"].dropna().unique()
    skey = keys_parallel(uniq, scorer_key); pkey = keys_parallel(uniq, parent_key)
    spec["scorer_key"] = spec["smiles"].map(skey); spec["parent_key"] = spec["smiles"].map(pkey)
    clean = {s: is_clean_answer(s) for s in uniq}
    spec = spec[spec["scorer_key"].notna() & spec["smiles"].map(clean).fillna(False).astype(bool)]
    step("after valid neutral single-fragment", len(spec))
    spec = spec[~spec["scorer_key"].isin(seen_s) & ~spec["parent_key"].isin(seen_p)]
    step("after unseen-in-train filter", len(spec))

    info = {s: mol_info(s) for s in spec["smiles"].unique()}
    spec = spec[spec["smiles"].map(lambda s: info.get(s) is not None)]
    spec["normalized_smiles"] = spec["smiles"].map(lambda s: info[s][0])
    spec["inchikey"] = spec["smiles"].map(lambda s: info[s][1])
    spec["molecular_formula"] = spec["smiles"].map(lambda s: info[s][2])
    mass = spec["smiles"].map(lambda s: info[s][3]).to_numpy(float)
    nzd = np.array([ADDUCTS[a] for a in spec["adduct"]])
    theo = (nzd[:, 0] * mass + nzd[:, 2]) / nzd[:, 1]
    spec["precursor_error_ppm"] = np.abs(spec["precursor_mz"].to_numpy(float) - theo) / theo * 1e6
    spec = spec[spec["precursor_error_ppm"] <= args.max_ppm]
    step(f"after precursor misfit <= {args.max_ppm:g} ppm", len(spec))
    # The scorer key is the identity everywhere downstream; one structure per scorer key (first SMILES seen)
    first = spec.groupby("scorer_key")["normalized_smiles"].first()
    fik = spec.groupby("scorer_key")["inchikey"].first(); ffo = spec.groupby("scorer_key")["molecular_formula"].first()
    spec["normalized_smiles"] = spec["scorer_key"].map(first)
    spec["inchikey"] = spec["scorer_key"].map(fik); spec["molecular_formula"] = spec["scorer_key"].map(ffo)
    spec["inchikey14"] = spec["inchikey"].str[:14]

    # peaks: finite, positive, sorted by m/z, intensities to base peak = 1
    mzs, its, bpi = [], [], []
    for mz, it in zip(spec["ms2_mzs"], spec["ms2_intensities"]):
        mz = np.asarray(mz, float); it = np.asarray(it, float)
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz, it = mz[ok], it[ok]; o = np.argsort(mz, kind="mergesort"); mz, it = mz[o], it[o]
        b = float(it.max()) if len(it) else np.nan
        mzs.append(mz); its.append(it / b if len(it) else it); bpi.append(b)
    spec["ms2_mzs"] = mzs; spec["ms2_normalized_intensities"] = its; spec["base_peak_intensity"] = bpi
    spec["num_peaks"] = spec["ms2_mzs"].map(len).astype("int64")
    spec = spec[spec["num_peaks"] >= args.min_peaks]
    ce = [ce_parse(r, p) for r, p in zip(spec["collision_energy_raw"], spec["precursor_mz"])]
    spec["collision_energy_ev"] = [c[0] for c in ce]
    spec["collision_energy_orig"] = [c[1] for c in ce]
    spec["collision_energy_orig_units"] = [c[2] for c in ce]
    spec["ingest_lib"] = spec["source"].map(LIB_NAME)
    assert spec["ingest_lib"].notna().all(), spec["source"].unique()
    spec["adduct_orig"] = spec["adduct_raw"].astype(str)
    spec["ionization_mode"] = np.where(spec["adduct"].str.endswith("+"), "positive", "negative")

    schema = pq.ParquetFile(args.train).schema_arrow
    cols = schema.names
    spec = spec.sort_values(["inchikey14", "ingest_lib", "adduct"], kind="mergesort").reset_index(drop=True)
    tab = pa.Table.from_pandas(spec[cols], schema=schema, preserve_index=False)
    pq.write_table(tab, out / "extra_full.parquet", compression="zstd")
    step("extra_full rows", len(spec)); step("extra_full structures", spec["inchikey14"].nunique())

    ex_s, ex_p = set(), set()
    for f in args.exclude:
        s, p = load_key_set(f); ex_s |= s; ex_p |= p
    step("eval keys excluded (scorer)", len(ex_s))
    drop = spec["scorer_key"].isin(ex_s) | spec["parent_key"].isin(ex_p) | spec["inchikey14"].isin(ex_s)
    ev = spec[~drop]
    pq.write_table(pa.Table.from_pandas(ev[cols], schema=schema, preserve_index=False), out / "extra_eval.parquet",
                   compression="zstd")
    step("extra_eval rows", len(ev)); step("extra_eval structures", ev["inchikey14"].nunique())
    step("eval molecules present in extra_full", spec.loc[drop, "scorer_key"].nunique())

    def mix(d):
        return dict(rows=len(d), structures=int(d["inchikey14"].nunique()),
                    by_lib_rows=d["ingest_lib"].value_counts().to_dict(),
                    by_lib_structures=d.groupby("ingest_lib")["inchikey14"].nunique().to_dict(),
                    adduct=d["adduct"].value_counts().to_dict(), instrument=d["instrument_type"].value_counts().to_dict(),
                    mode=d["ionization_mode"].value_counts().to_dict(),
                    structures_with_timstof=int(d.loc[d["instrument_type"] == "timsTOF", "inchikey14"].nunique()))
    summ = dict(funnel=log, full=mix(spec), eval=mix(ev))
    (out / "build_log.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
