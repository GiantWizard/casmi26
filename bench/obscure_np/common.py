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


def scorer_key(smiles: str | None) -> str | None:
    """InChIKey14 after RemoveStereochemistry, the identity the metric uses (D746878).

    Returns None for unparseable SMILES so callers can drop them explicitly.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


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

_ADDUCT_ALIASES = {
    "[M+H]+": ["M+H", "[M+H]", "[M+H]+", "M+H+", "[M+H]1+"],
    "[M+Na]+": ["M+Na", "[M+Na]", "[M+Na]+", "M+Na+", "[M+Na]1+"],
    "[M+NH4]+": ["M+NH4", "[M+NH4]", "[M+NH4]+", "M+NH4+", "[M+NH4]1+"],
    "[M+K]+": ["M+K", "[M+K]", "[M+K]+", "M+K+", "[M+K]1+"],
    "[M+H-H2O]+": ["M+H-H2O", "[M+H-H2O]", "[M+H-H2O]+", "[M-H2O+H]+", "M-H2O+H", "[M+H-H2O]1+"],
    "[M-H]-": ["M-H", "[M-H]", "[M-H]-", "M-H-", "[M-H]1-"],
    "[M+Cl]-": ["M+Cl", "[M+Cl]", "[M+Cl]-", "M+Cl-", "[M+Cl]1-"],
    "[M+CH2O2-H]-": ["M+FA-H", "[M+FA-H]-", "[M+HCOO]-", "M+HCOO", "[M+CH2O2-H]-", "[M+HCOOH-H]-", "M+CH2O2-H"],
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
    "ac$mass_spectrometry: collision_energy": "collision_energy_raw",
    "name": "_name", "db#": "record_id", "accession": "record_id", "spectrumid": "record_id",
    "spectrum_id": "record_id", "scans": "_scans", "library": "library",
    "organism": "library", "comment": "_comment", "date": "date",
    "library_name": "library", "librarymembership": "library",
}


def _finish(rec: dict, peaks: list, source: str, idx: int) -> dict | None:
    if not peaks:
        return None
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


def read_table(path: Path, source: str) -> list[dict]:
    """Parquet/CSV/JSON already in a tabular layout (e.g. a converted Enveda-180 dump)."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        df = pd.DataFrame(json.loads(path.read_text()))
    lower = {c.lower(): c for c in df.columns}
    ren = {lower[k]: v for k, v in _KEY_MAP.items() if k in lower and not v.startswith("_")}
    for alt, tgt in (("ms2_mzs", "ms2_mzs"), ("mzs", "ms2_mzs"),
                     ("ms2_normalized_intensities", "ms2_intensities"),
                     ("intensities", "ms2_intensities"), ("ms2_intensities", "ms2_intensities")):
        if alt in lower:
            ren.setdefault(lower[alt], tgt)
    df = df.rename(columns=ren)
    rows = []
    for i, r in enumerate(df.to_dict("records")):
        out = {c: r.get(c) for c in SPEC_COLUMNS}
        out["source"] = source
        out["record_id"] = out["record_id"] or str(i)
        if out["ms2_mzs"] is None:
            continue
        out["ms2_mzs"] = np.asarray(out["ms2_mzs"], dtype=float)
        out["ms2_intensities"] = np.asarray(out["ms2_intensities"], dtype=float)
        rows.append(out)
    return rows


def read_spectra(path: str) -> pd.DataFrame:
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
        rows = read_table(p, source)
    return pd.DataFrame(rows, columns=SPEC_COLUMNS)


def load_key_set(path: str | None, smiles_cols=("smiles", "SMILES", "canonical_smiles"),
                 key_cols=("scorer_key", "inchikey14", "inchikey", "InChIKey")) -> tuple[set, set]:
    """Load a structure table and return (scorer keys, parent keys).

    Uses SMILES when present (recomputes both keys, so the shipped-key mismatch in
    D742042 can't leak through); otherwise falls back to the first 14 chars of a key
    column, which only fills the scorer-key set.
    """
    if not path:
        return set(), set()
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
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
        for s in uniq:
            k = scorer_key(s)
            if k:
                skeys.add(k)
            pk = parent_key(s)
            if pk:
                pkeys.add(pk)
    return skeys, pkeys
