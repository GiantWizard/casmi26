---
status: active
area: portfolio
tags: [casmi26, kaggle, handoff]
updated: 2026-10-11
---

# CASMI 2026 handoff

Kaggle: Enveda CASMI 2026 (MRR@25, ~400 timsTOF test molecules, 9h T4, no internet, NIST banned). Account `giantwizard`.
Full research report: [[CASMI 2026 approach deep dive]] (in `reports/`), notes in `research_notes/`.

## Scores (public LB)
- **Best 0.438**: `casmi26-jc-base2` (published 0.438 notebook, unchanged) and `casmi26-jc-c2` (same + fine-tuned GLACIER only for molecules with no [M+H]+/[M+Na]+ spectrum). Use jc-c2 as one final pick.
- jc-c 0.414 (fine-tuned GLACIER added at full weight on [M+H]+ hurt), jc-base 0.423, jc-b 0.416, jc-a 0.408.
- Medal lines (3,072 teams, Oct 11): bronze ~0.438, silver ~0.440-0.441, gold ~0.448+. Leader 0.484.

## Ruled out (measured, don't redo)
- Leak-free LightGBM ranker: loses to public ranker on the clean Enveda NP panel (0.679 vs 0.722; all-features version 0.604, leakage trap).
- More NP training rows: learning curve flat after ~300 NP molecules.
- Better fingerprint model: FP top-1 0.23 -> 0.37 changed pipeline MRR 0.722 -> 0.719 on the panel.
- Mix GLACIER as extra full-weight term (-0.024 LB). De novo generation: oracle sim +0.001-0.005 realistic.

## Key lessons
- `model_views` (fe_A/B/D) and `fragnet` features memorize training molecules (fe_A top-1 0.70 on held-out train NPs vs 0.37 on the unseen panel). Any validation on train-derived molecules is inflated.
- Only clean test set so far: Enveda NP panel (split fold 'np', 265 molecules), but it is too easy (class-2 MRR 0.88 vs ~0.6 implied on test).
- Rule: nothing is submitted unless it beats jc-c2 on a clean held-out test first.

## Next step
Build an honest "obscure natural product" test set from public library releases newer than train.parquet (host-approved GNPS/MassBank/MoNA, Enveda-180 Zenodo extra 816 structures), filtered to NP structures absent from train.parquet, so every model in the pipeline is unseen. Then re-test levers there (contrastive GLACIER fine-tune, conditional popularity weight, fusion weights).

## Where things are
- Local (Mac only): `~/casmi26/` (data, scripts in `rk/`, `bench/`, `tune/`, `ft/`, notebooks in `kernels/`, Colab notebooks in `colab/`).
- Kaggle private datasets: `giantwizard/casmi26-jc-{bench,tims,mix,ftcode,glacier-ft,rk,rkrows,diag}`; fine-tuned weights `glacier_mix.pt`, `glacier_tims.pt` in `casmi26-jc-glacier-ft`.
- Kaggle private notebooks: `giantwizard/casmi26-jc-{base,a,b,c,c2,base2,val,bench-run,bench2,bench3,diag-run,ft}`.
- Colab (alt Google account, ~390 compute units): Drive `MyDrive/casmi26/`.
