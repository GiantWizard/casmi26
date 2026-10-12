# Obscure-NP held-out benchmark

Implements the handoff's "Next step": a test set where **every molecule is unseen by
every model in the pipeline**, so `model_views` / `fragnet` / FPNet memorisation can't
inflate scores, and harder than the Enveda NP panel (popularity is near-useless here).

Requirements: `pip install rdkit pandas pyarrow` (tested with RDKit 2026.09, Python 3.13).

## 1. Download sources (host-approved, Mac side)
- Enveda-180 Zenodo record 21346580 (CC BY 4.0). Has the 816 standard-adduct InChIKey14s missing from train (D745148).
- MassBank latest release (`MassBank_NISTformat.msp` from MassBank-data GitHub releases).
- MoNA LC-MS/MS export (`.msp`).
- GNPS `ALL_GNPS_NO_PROPOGATED.mgf`. Records whose library names NIST (e.g. GNPS-NIST14-MATCHES) are **dropped automatically**.

The readers handle `.msp`, `.mgf`, and `.parquet/.csv` with common column names. If
Enveda-180 ships in another layout, convert it to a parquet with `smiles, adduct,
precursor_mz, ms2_mzs, ms2_intensities, ionization_mode, collision_energy` first.

## 2. Build
```
python bench/obscure_np/build_obscure_np_set.py \
  --train ~/casmi26/data/train.parquet --train-structs ~/casmi26/data/train_structs.parquet \
  --extra-seen <any fine-tune set used for glacier_mix.pt / glacier_tims.pt> \
  --spectra enveda180=... massbank=... mona=... gnps=... \
  --coconut <COCONUT structures> --candidate-db <PubChem ∪ COCONUT pool keys> \
  --popularity <popularity table with 'popularity' column> \
  --exclude-panel <np panel keys> --out ~/casmi26/bench/obscure_np_v1
```
The filter funnel prints as it runs and is saved to `build_log.json`. Outputs:
- `test_draw{0..4}.parquet` uses the competition `test.parquet` columns. Each molecule is resampled to Poisson(3.03) spectra, matching the hidden test (D742055).
- `solution.csv` holds the truth scorer key plus slice labels: `cls` (2 = in candidate pool, 3 = not), `obscure`, and `adduct_group`.
- `spectra_all.parquet` is the full unsampled pool.

"Unseen" uses two identities: the scorer key (InChIKey14 after `RemoveStereochemistry`) and a
parent key (largest fragment, uncharged, canonical tautomer). A molecule that matches train
on either is dropped, which covers the ~8% tautomer copies and the shipped-key mismatch
(D742042, D743254). The parent-key pass over all of train takes a while (tens of minutes,
untimed), so cache the output directory.

## 3. Run kernels and score
Upload the out dir as a private Kaggle dataset (e.g. `casmi26-jc-onp`). In each kernel
(`jc-c2` first), point the test path at `test_drawN.parquet` instead of the competition
file, and write the 25-candidate submission. Then:
```
python bench/obscure_np/mrr_eval.py --solution OUT/solution.csv --test OUT/test_draw0.parquet \
  --sub jc-c2=jc-c2_d0.csv cand=cand_d0.csv --baseline jc-c2
```
Wide (`smiles_1..25`) and long (`molecule_id, rank, smiles`) submissions both work. The
`--test` flag recomputes `adduct_group` from the draw, so `ft_only` means "no [M+H]+/[M+Na]+
spectrum in what the kernel saw", which is jc-c2's routing condition.

**Gate:** promote only if the paired-bootstrap 90% CI vs jc-c2 is above zero on `all` in
at least 4 of 5 draws, and the `obscure=True` slice does not lose.

## First experiments to run on it (from the report's action plan)
1. Sanity check: jc-base2 vs jc-c2 vs jc-c. Expect jc-c below both, as on the LB. If the harness can't reproduce that order, fix it before trusting anything else.
2. Conditional popularity weight: compare MRR on the `obscure` and `cls` slices against the NP panel.
3. Contrastive GLACIER fine-tune on the `ft_only` slice.
4. Fusion weights / margin-preserving fusion vs RRF.
