"""Shared helpers: scorer-identity keys, adduct normalisation, spectrum readers."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

_uncharger = rdMolStandardize.Uncharger()
_frag_chooser = rdMolStandardize.LargestFragmentChooser()


_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()


def scorer_key(smiles: str | None) -> str | None:
    """The competition metric's identity (Kaggle metric 'casmi-mean-reciprocal-rank' v14, as copied
    into the jc-* notebooks): RemoveStereochemistry, then TautomerEnumerator().Canonicalize,
    then InChIKey14. Returns None when the metric would treat the SMILES as a non-match.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        key = Chem.MolToInchiKey(_tautomer_canonicalizer.Canonicalize(mol))
        return key[:14] if key else None
    except (ValueError, RuntimeError, TypeError):
        return None


def keys_parallel(smiles, fn, processes=None) -> dict:
    """{smiles: fn(smiles)} over unique SMILES using all cores (tautomer canonicalisation is slow)."""
    import multiprocessing as mp
    uniq = list(dict.fromkeys(s for s in smiles if isinstance(s, str)))
    if len(uniq) < 2000:
        return {s: fn(s) for s in uniq}
    with mp.Pool(processes) as pool:
        return dict(zip(uniq, pool.map(fn, uniq, chunksize=500)))


def parent_key(smiles: str | None) -> str | None:
    """Stricter identity for leak checks: largest fragment, neutralised, canonical
    tautomer, stereo removed. Two structures sharing this key count as "seen"
    even if their scorer keys differ (salts, charge states, ~8% tautomer copies).
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _frag_chooser.choose(mol)
        mol = _uncharger.uncharge(mol)
        mol = rdMolStandardize.CanonicalTautomer(mol)
    except Exception:  # standardiser can choke on odd valences; fall back to raw
        pass
    Chem.RemoveStereochemistry(mol)
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def is_clean_answer(smiles: str) -> bool:
    """RDKit-valid, single fragment, net-neutral: the shape a truth answer should have."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return False
    return Chem.GetFormalCharge(mol) == 0


# --------------------------------------------------------------------------
# Adducts
# --------------------------------------------------------------------------

_ADDUCT_ALIASES = {  # keys are the competition's test.parquet spellings
    "[M+H]+": ["M+H", "[M+H]", "[M+H]+", "M+H+", "[M+H]1+"],
    "[M+Na]+": ["M+Na", "[M+Na]", "[M+Na]+", "M+Na+", "[M+Na]1+"],
    "[M+NH4]+": ["M+NH4", "[M+NH4]", "[M+NH4]+", "M+NH4+", "[M+NH4]1+"],
    "[M+K]+": ["M+K", "[M+K]", "[M+K]+", "M+K+", "[M+K]1+"],
    "[M-H2O+H]+": ["M+H-H2O", "[M+H-H2O]", "[M+H-H2O]+", "[M-H2O+H]+", "M-H2O+H", "[M+H-H2O]1+"],
    "[M-2H2O+H]+": ["M+H-2H2O", "[M+H-2H2O]+", "[M-2H2O+H]+", "M-2H2O+H"],
    "[M-H]-": ["M-H", "[M-H]", "[M-H]-", "M-H-", "[M-H]1-"],
    "[M+Cl]-": ["M+Cl", "[M+Cl]", "[M+Cl]-", "M+Cl-", "[M+Cl]1-"],
    "[M+CH2O2-H]-": ["M+FA-H", "[M+FA-H]-", "[M+HCOO]-", "M+HCOO", "[M+CH2O2-H]-", "[M+HCOOH-H]-",
                     "M+CH2O2-H", "[M+CHO2]-", "M+CHO2"],
    "[M-H2O-H]-": ["M-H2O-H", "[M-H2O-H]-", "[M-H-H2O]-", "M-H-H2O"],
}
_ALIAS_TO_ADDUCT = {a.upper().replace(" ", ""): k for k, v in _ADDUCT_ALIASES.items() for a in v}

STANDARD_ADDUCTS = tuple(_ADDUCT_ALIASES)
# What base (MassSpecGym) GLACIER can score; everything else is the fine-tune's subset.
BASE_GLACIER_ADDUCTS = ("[M+H]+", "[M+Na]+")


def normalize_adduct(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    return _ALIAS_TO_ADDUCT.get(raw.strip().upper().replace(" ", ""))


def adduct_group(adduct: str | None) -> str:
    if adduct in BASE_GLACIER_ADDUCTS:
        return "base_glacier"
    if adduct is None:
        return "unknown"
    return "ft_only"


# --------------------------------------------------------------------------
# Spectrum readers -> one row per spectrum with a common schema
# --------------------------------------------------------------------------

SPEC_COLUMNS = [
    "source", "record_id", "smiles", "inchikey", "adduct_raw", "ionization_mode",
    "instrument_type", "precursor_mz", "collision_energy_raw", "library", "date",
    "ms2_mzs", "ms2_intensities",
]

_KEY_MAP = {
    "smiles": "smiles", "computed smiles": "smiles", "ch$smiles": "smiles",
    "inchikey": "inchikey", "ch$iupac_inchikey": "inchikey", "ch$link: inchikey": "inchikey",
    "precursortype": "adduct_raw", "precursor_type": "adduct_raw", "adduct": "adduct_raw",
    "ms$focused_ion: precursor_type": "adduct_raw", "ionmode": "ionization_mode",
    "ion_mode": "ionization_mode", "ac$mass_spectrometry: ion_mode": "ionization_mode",
    "instrument_type": "instrument_type", "instrumenttype": "instrument_type",
    "ac$instrument_type": "instrument_type", "source_instrument": "instrument_type",
    "precursormz": "precursor_mz", "precursor_mz": "precursor_mz", "pepmass": "precursor_mz",
    "ms$focused_ion: precursor_m/z": "precursor_mz",
    "collision_energy": "collision_energy_raw", "collisionenergy": "collision_energy_raw",
    "collision_energies": "collision_energy_raw", "instrument": "instrument_type",
    "ac$mass_spectrometry: collision_energy": "collision_energy_raw",
    "name": "_name", "db#": "record_id", "accession": "record_id", "spectrumid": "record_id",
    "spectrum_id": "record_id", "scans": "_scans", "library": "library",
    "organism": "library", "comment": "_comment", "date": "date",
    "library_name": "library", "librarymembership": "library",
    "mslevel": "_mslevel", "ms_level": "_mslevel", "spectrum_type": "_mslevel", "libraryquality": "_quality",
}


def _finish(rec: dict, peaks: list, source: str, idx: int) -> dict | None:
    if not peaks:
        return None
    lvl = re.sub(r"\D", "", str(rec.get("_mslevel", "")))
    if lvl and lvl != "2":
        return None
    if str(rec.get("_quality", "1")).strip() in ("4", "10"):  # GNPS 'challenge'/unverified tiers
        return None
    if not rec.get("adduct_raw") and rec.get("_name"):  # GNPS: 'Compound name M+H'
        rec["adduct_raw"] = str(rec["_name"]).split()[-1]
    out = {c: rec.get(c) for c in SPEC_COLUMNS}
    out["source"] = source
    if not out["record_id"]:
        out["record_id"] = str(rec.get("_scans") or idx)
    arr = np.asarray(peaks, dtype=float)
    out["ms2_mzs"] = arr[:, 0]
    out["ms2_intensities"] = arr[:, 1]
    try:
        out["precursor_mz"] = float(str(out["precursor_mz"]).split()[0])
    except (TypeError, ValueError):
        out["precursor_mz"] = np.nan
    return out


def _parse_kv(line: str, sep: str):
    k, _, v = line.partition(sep)
    return k.strip().lower(), v.strip()


def read_msp(path: Path, source: str) -> list[dict]:
    rows, rec, peaks, in_peaks, idx = [], {}, [], False, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                r = _finish(rec, peaks, source, idx)
                if r:
                    rows.append(r)
                rec, peaks, in_peaks, idx = {}, [], False, idx + 1
                continue
            if in_peaks or (line[0].isdigit() and ":" not in line):
                parts = re.split(r"[\s;,]+", line)
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                    continue
                except (ValueError, IndexError):
                    pass
            k, v = _parse_kv(line, ":")
            if k == "num peaks":
                in_peaks = True
            elif k == "instrument":  # model name beats MassBank's 'LC-ESI-ITFT' Instrument_type
                rec["instrument_type"] = v
            elif k in _KEY_MAP:
                rec.setdefault(_KEY_MAP[k], v)
            elif k == "comments":  # MoNA packs SMILES / instrument in Comments
                for m in re.finditer(r'"([^"=]+)=([^"]*)"', v):
                    kk = m.group(1).strip().lower()
                    if kk in _KEY_MAP:
                        rec.setdefault(_KEY_MAP[kk], m.group(2).strip())
    r = _finish(rec, peaks, source, idx)
    if r:
        rows.append(r)
    return rows


def read_mgf(path: Path, source: str) -> list[dict]:
    rows, rec, peaks, idx = [], {}, [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line == "BEGIN IONS":
                rec, peaks = {}, []
            elif line == "END IONS":
                r = _finish(rec, peaks, source, idx)
                if r:
                    rows.append(r)
                idx += 1
            elif line and (line[0].isdigit()) and "=" not in line:
                parts = line.split()
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except (ValueError, IndexError):
                    pass
            elif "=" in line:
                k, v = _parse_kv(line, "=")
                if k in _KEY_MAP:
                    rec.setdefault(_KEY_MAP[k], v)
    return rows


def normalize_instrument(raw) -> str | None:
    """Collapse vendor strings onto 'timsTOF' / 'QTOF' / 'Orbitrap' (test.parquet is all timsTOF).

    Anything else (triple quads, ion traps, unknown) maps to 'other' so the builder can drop
    low-resolution spectra that look nothing like the test.
    """
    if not isinstance(raw, str):
        return "other"
    r = raw.lower().replace("-", " ")
    if "tims" in r:
        return "timsTOF"
    if any(t in r for t in ("orbitrap", "exactive", "exploris", "ltq ft", "itft", "fusion", "lumos", "hcd")):
        return "Orbitrap"
    if any(t in r for t in ("tof", "maxis", "impact", "synapt", "xevo g2", "triple tof", "6530", "6545", "6550")):
        return "QTOF"
    return "other"


def read_table(path: Path, source: str, skip_ik14: set | None = None) -> list[dict]:
    """Parquet/CSV/JSON already in a tabular layout (e.g. Enveda-180's parquet)."""
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        names = {n.lower(): n for n in tbl.column_names}
        if skip_ik14 and "inchikey" in names:  # drop seen rows before the expensive to_pandas
            ik14 = pc.utf8_slice_codeunits(tbl[names["inchikey"]], 0, 14)
            tbl = tbl.filter(pc.invert(pc.is_in(ik14, value_set=pa.array(sorted(skip_ik14)))))
        df = tbl.to_pandas()
    elif path.suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        df = pd.DataFrame(json.loads(path.read_text()))
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    if "instrument" in lower:  # specific model name beats a generic type like 'Q-TOF'
        ren[lower["instrument"]] = "instrument_type"
    for k, v in _KEY_MAP.items():
        if k in lower and not v.startswith("_") and v not in ren.values():
            ren[lower[k]] = v
    for alt, tgt in (("ms2_mzs", "ms2_mzs"), ("mzs", "ms2_mzs"),
                     ("ms2_normalized_intensities", "ms2_intensities"),
                     ("intensities", "ms2_intensities"), ("ms2_intensities", "ms2_intensities")):
        if alt in lower and tgt not in ren.values():
            ren[lower[alt]] = tgt
    df = df[list(ren)].rename(columns=ren)
    if skip_ik14 and "inchikey" in df.columns:
        df = df[~df["inchikey"].astype(str).str[:14].isin(skip_ik14)]
    rows = []
    for i, r in enumerate(df.to_dict("records")):
        out = {c: r.get(c) for c in SPEC_COLUMNS}
        out["source"] = source
        out["record_id"] = out["record_id"] or str(i)
        if out["ms2_mzs"] is None:
            continue
        out["ms2_mzs"] = np.asarray(out["ms2_mzs"], dtype=float)
        out["ms2_intensities"] = np.asarray(out["ms2_intensities"], dtype=float)
        try:
            out["precursor_mz"] = float(str(out["precursor_mz"]).split()[0])
        except (TypeError, ValueError):
            out["precursor_mz"] = np.nan
        rows.append(out)
    return rows


def read_spectra(path: str, skip_ik14: set | None = None) -> pd.DataFrame:
    """`path` may be `name=file` to label the source; otherwise the file stem is used."""
    source, _, file = path.rpartition("=")
    p = Path(file)
    source = source or p.stem
    suf = p.suffix.lower()
    if suf == ".msp":
        rows = read_msp(p, source)
    elif suf == ".mgf":
        rows = read_mgf(p, source)
    else:
        rows = read_table(p, source, skip_ik14)
    df = pd.DataFrame(rows, columns=SPEC_COLUMNS)
    df["instrument_type"] = df["instrument_type"].map(normalize_instrument)
    return df


def load_key_set(path: str | None, smiles_cols=("smiles", "SMILES", "canonical_smiles"),
                 key_cols=("scorer_key", "inchikey14", "inchikey", "InChIKey")) -> tuple[set, set]:
    """Load a structure table and return (scorer keys, parent keys).

    Uses SMILES when present (recomputes both keys, so the shipped-key mismatch in
    D742042 can't leak through); otherwise falls back to the first 14 chars of a key
    column, which only fills the scorer-key set.
    """
    if not path:
        return set(), set()
    p = Path(path).expanduser()
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        names = pq.ParquetFile(p).schema_arrow.names
        want = [c for c in (*smiles_cols, *key_cols) if c in names]
        df = pq.read_table(p, columns=want).to_pandas()
    elif p.suffix == ".txt":
        df = pd.DataFrame({"inchikey": p.read_text().split()})
    else:
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    scol = next((c for c in smiles_cols if c in df.columns), None)
    kcol = next((c for c in key_cols if c in df.columns), None)
    skeys, pkeys = set(), set()
    if kcol:
        skeys |= {str(k)[:14] for k in df[kcol].dropna()}
    if scol:
        uniq = df[scol].dropna().unique()
        skeys |= {k for k in keys_parallel(uniq, scorer_key).values() if k}
        pkeys |= {k for k in keys_parallel(uniq, parent_key).values() if k}
    return skeys, pkeys
