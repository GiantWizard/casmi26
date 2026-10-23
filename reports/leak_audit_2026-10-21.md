# Leak audit of the v465 components, 2026-10-21

Purpose: one learned reranker over all of v465's channels, trained on rows whose features come from spectra that no
upstream model was fitted on. For each component: what it was trained on, and what replaces it when building rows.
Holdout used throughout: identity fold 0 of dmitriigluzdov/casmi26-identity-disjoint-validation-folds (55,102 structures)
plus the 265 natural-product panel structures, i.e. the split in our foldsafe dataset (`split.parquet`).

| Component | Trained on | Clean for fold-0 rows? | Replacement for row building |
|---|---|---|---|
| Engine 1 FPNet A (`fpnet_0`), `fpnet_full1` bank | all train ("ALL train folds incl. np-examples", dataset card) | no | `fpnet_fold0.pt` (ours, fold 0 withheld) |
| Engine 1 `fe_A/B/D`, `ranker_v4b` | all train (no training code) | no | drop fe features; our ranker is superseded by the new reranker |
| Engine 2 FP models (`fp_single_s2`, `fp_merged_m1`, 6,930 bits) | training source recovered at github.com/cubres/enveda-casmi-2026 (legacy-v2); split not yet checked | unknown | check its split; else retrain with its own trainers on non-fold-0 (Colab) |
| Engine 2 rankers (`rank_train.npz` 142,762 rows; `sim_rank_rows_nofp.npz`) | simulated queries on train; card says G is not proof of structure-disjointness | no | replaced by the new reranker; engine 2 contributes scores/ranks as features only |
| ICEBERG, GLACIER | MassSpecGym (external) | partly: fold-0 structures that are also in MassSpecGym train are leaky | flag MassSpecGym overlap per structure; train on non-overlap rows or add the flag only for analysis |
| PubChem channel | FPNet z + popularity | inherits FPNet | rerun with `fpnet_fold0.pt` |
| Popularity prior, MetFrag-lite, library match | untrained | yes | as is |

Distribution caveat: fold-0 structures come from spectral libraries and are more popular than the hidden Enveda natural
products, so the reranker could over-trust popularity. Rows are therefore built in three simulated regimes per query
(class 1: library kept; class 2: the structure's spectra removed from the library; class 3: also removed from the pool),
weighted to the estimated test mix, and evaluated on the NP panel separately.

Next: row generator (Colab) that runs every channel with the fold-safe replacements and saves per-candidate features.

## Result, 2026-10-22: the stacked reranker fails on the leaderboard

| Submission | Public LB | Offline (simulated held-out molecules) |
|---|---|---|
| v465 copy | 0.456 | fusion 0.687 |
| stack-full (v465 models + reranker) | **0.356** | reranker 0.803 (+0.116, CI +0.103..+0.129) |
| stack-safe (fold-safe models + reranker) | **0.340** | same |

The reranker learned "trust engine 1" (engine-1 score gap = 58% of importance). In the simulation engine 1 alone beats
v465's fusion (0.769 vs 0.694); on the hidden test the opposite holds, so the reranker scores about like an engine-1-only
pipeline (the v4n-era 0.38-0.40). Simulated test molecules built from held-out training structures do not reproduce the
hidden test's balance between channels, even with every model fold-safe: a +0.116 offline gain became -0.100 on the LB.
Consequence: no offline harness built from training structures can be trusted to rank pipeline-level changes; only the
LB can.
