---
status: active
area: portfolio
tags: [casmi26, kaggle, session-log]
updated: 2026-10-15
---

# Local session 2026-10-13/14 (Mac): what was built, measured and learned

Ran in parallel with the cloud session (`session-2026-10-12-cloud.md`), which built jc-gen. Scripts from this
session are under `bench/netgen/`, `bench/fpsim/`, `bench/mist/`, Colab notebooks under `colab/`. They hard-code the
Mac paths `~/casmi26/...` (data, `fp/run/work` = the casmi spectrum cache from `rk/work` + fold-safe `split.parquet`).

## Leaderboard

| Submission | Public LB | Note |
|---|---|---|
| jc-base2, jc-c2fix, jc-gate | 0.438 | baseline family |
| jc-c | 0.414 | mix GLACIER at full weight |
| jc-lr2 | 0.410 | learned rerank |
| jc-gen (cloud session) | 0.431 | generation from pool relatives |
| jc-netgen (this session) | **0.436** | network-guided analog generation (-0.002; proxy predicted -0.005..-0.007 before the score) |

Public notebooks top out at 0.443 (`antriksharya/casmi26-v7c-e2rank2`: engine 2's top-1 at rank 2 when it disagrees
with the final #1). The 0.445-0.450 crowd are private edits by teams with 100+ entries. Kaggle weekly GPU quota is
exhausted (2026-10-14 14:45 CT); `kernels/jc-v7c-e2r2` (copy of the 0.443 notebook) and `kernels/jc-netgen-probe`
are ready to push after the reset.

## Main lesson: which offline proxy predicts the LB

`bench/netgen/calibrate.py` on ONP set B (447 molecules; base2 = replayed rp_base2):

| Variant | LB change | all | non-conjugate | **non-conj in-pool (n 61)** | non-conj out-of-pool | conjugates |
|---|---|---|---|---|---|---|
| c | -0.024 | -0.004 | -0.010 | **-0.022** | +0.002 | -0.002 |
| gate | 0.000 | +0.010 | +0.004 | **-0.005** | +0.013 | +0.012 |
| lr2 | -0.028 | +0.050 | +0.026 | **-0.011** | +0.063 | +0.059 |
| netgen | pending | +0.094 | +0.013 | **-0.005** | +0.032 | +0.124 |
| e2rank2 replica | +0.002..+0.003 | -0.000 | -0.004 | **-0.007** | 0 | +0.001 |

- **73-74% of ONP sets A and B (81-83% of their out-of-pool slice) are synthetic amino-acid conjugates** from GNPS
  combinatorial series (SMARTS `[NX3][CX4]C(=O)[OX2H1,OX1-]`). Their homolog families inflate every out-of-pool gain.
- The non-conjugate in-pool slice gets the sign of all three known LB results right (c, gate, lr2); every other slice
  gets at least one badly wrong. It is too small (n 61) to resolve +/-0.003 effects (e2rank2 replica disagrees in
  sign at 1 vs 4 molecules). Use it as a harm filter for large changes, not to rank small tweaks.
- The out-of-pool slices are consistently optimistic: gains there do not reach the LB.
- Set A draw 2 (non-conj in-pool n 40; lr2 excluded because it trained on set A): c **-0.056** (2 / 6), gate +0.023
  (4 / 1), e2rank2 +0.013 (1 / 1), netgen **-0.007** (1 / 5). Across A + B (~100 molecules) the proxy flags c's harm
  clearly, puts gate and e2rank2 near zero, and predicts netgen slightly negative on both sets.
- LB experiments prepared for the quota reset (`kernels/make_v7c_variants.py`): `jc-v7c-popx0p5` and `jc-v7c-popx2`
  (both popularity weights x0.5 / x2 on the 0.443 notebook). Popularity is the one knob no library-molecule proxy can
  judge, and the hidden test's class-2 answers have no public spectra.

## Miss diagnosis on the LB-calibrated slice (`bench/netgen/miss_diag.py`, 2026-10-14 evening)

101 non-conjugate in-pool molecules (ONP A d2 + B d0), final lists = replayed c2fix: MRR 0.707; rank 1: 59, rank 2-25:
34 (17 at rank 2), not in the top 25: 8 (in neither engine's list; several are odd records: an Mg complex, explicit-H
SMILES, single spectra).
- 30 of 34 near-misses lose to a same-formula isomer. Popularity is not the cause (winner more popular 14, answer 16).
  GLACIER prefers the answer in only 6/22, ICEBERG 11/21; both engines rank the wrong isomer first in 32/34.
- Half are close family isomers (Tanimoto answer vs #1 >= 0.5, mostly cyclic peptides/lipopeptides of one family),
  half are wrong scaffolds (Tanimoto 0.10-0.45).
- The honest fold-safe FPNet also prefers the wrong #1: family 3/17, wrong scaffold 2/17 (answer's median FP rank 4 and
  7 among the top 10). Not a fusion/weighting bug: every signal agrees on the wrong structure, so only a better
  spectrum-to-structure model can move this slice. Targeted check for any new FP model (e.g. the Colab simulation-
  trained one): prefers the answer over our #1 in more than 5 of these 34 near-misses (rows in `miss_diag_AB.csv`).

## Network-guided analog generation (`bench/netgen/`)

Idea (prior art: NAP, PLOS Comp Biol 2018; NetID, Nat Methods 2021; ModiFinder, JASMS 2024): link test molecules by
modified cosine (>= 0.6), apply single biotransformation edits (`edits.py`: +/-CH3, +/-OH, CH2 homologs, methoxy,
+/-H2, -CO2, +/-hexose/deoxyhexose/pentose, glucuronide, +/-acetyl, +COOH, hydroxymethyl, aliphatic methoxy, O/N-ethyl,
same structure) to each neighbour's top-1, keep exact-mass matches (10 ppm), score by FPNet + 2*link, insert the best
2 at rank 1 (rank 2 if the own top-1's pool popularity > 1). CPU, ~60 s for 400 molecules (`netgen.py`).

- ONP: A +0.099 (34 fixed / 0 broke), B one-shot +0.097 (39 / 1); module +0.093 on B. Visible test (class 1, answers
  recovered from train by spectrum match) -0.003; family-free in-pool set -0.01.
- **All of it comes from the amino-acid conjugates** (+43.5 / +43.0 RR-sum); everything else -4.5 / -1.1. Calibrated
  prediction for jc-netgen: about 0.433-0.437.
- Enveda's own timsTOF NP examples: 33% of molecules have a relative in the set (vs ~80% on ONP); oracle reach of
  one-edit generation 10.8% (basic edits) -> 16.8% (extended, glycosides matter).
- Rejected variants: 2 propagation rounds (+0.044 vs +0.099: re-insertion pushes good answers down), top-3 seeds (noise),
  relevance gate, DreaMS links (top-100 precision 0.36 vs modified cosine 0.53; requiring both 0.39 but fewer fixes),
  topological-torsion similarity (A +0.022, B +0.0035).
- Network re-ranking only (no generation): A +0.019, B +0.005; oracle ceiling on B +0.008.
- Positional-isomer enumeration of the own top-1/top-3 (move OH/CH3/OCH3): reaches 13 / 33 of 302 class-3 answers not
  in the top 25, median 17 / 50 candidates; ranking unexplored.

## Simulated-spectra FPNet (`bench/fpsim/`, Colab)

Literature basis (MassSpecGym audit, arXiv 2606.19624, raw text checked): MIST + 92% ICEBERG-simulated PubChem spectra
9.57% -> 53.76% Hit@1; test structures in the simulated pool 63.85% (inflated for benchmarks, realistic when the whole
candidate pool is simulated).
- Honest benchmark `bench_np.py` (reproduces Gluzdov's NP-panel 0.506): held-out NPs not in MassSpecGym train,
  fingerprint-only pool-window ranking; fold0 1,500 queries MRR 0.550 (top-1 39.7%).
- GLACIER simulations are realistic enough: fold0 retrieves answers from their simulated spectra at MRR 0.438 vs 0.541
  from real spectra.
- First pilot lost to a Colab disconnect. The full-pool simulation (`colab/casmi26_fpsim_simfull.ipynb`, QTOF token,
  [M+H]+ 20/40 eV + [M+Na]+ 30 eV, Drive-backed) was started by Jeffery 2026-10-14; then run
  `colab/casmi26_fpsim_train.ipynb` (control / simA / simB arms, kill rule simA or simB >= control + 0.03).
- Integration warning: the fold-safe FPNet as a third RRF list (-0.003..-0.015) or same-formula tiebreak (-0.004..-0.027,
  in-pool -0.026..-0.17) hurts; a new FP model must clearly beat the pipeline's own ranking before integration, and
  must pass the non-conj in-pool proxy.

## Other measurements

- Forward models are inflated by MassSpecGym membership: NP-panel isomer MRR GLACIER 0.737 (answer in MSG train) vs
  0.411 (not), ICEBERG 0.724 vs 0.454, MIST 0.834 vs 0.517; 88% of NP-panel answers are in MSG train.
- MIST (FRIGID MassSpecGym encoder, CC BY-NC weights, `bench/mist/mist_score.py`): weak alone, end-to-end noise, in-pool
  set LOSS -0.022; ICEBERG back in the final rerank: in-pool set LOSS -0.011.
- Host (D748363): CC-BY-NC-SA / "other"-licence data "should be fine"; FRIGID checkpoints still unanswered (D746430).
- Aryan Awasthi's probes: cosine library search 0.150 on public LB (~16% class 1); popularity is badly overrated by any
  validation built from library molecules.

## Simulated-spectra FPNet results (Colab, 2026-10-15)

Held-out NP benchmark (`bench_np.py`, 1,500 clean NP queries; NP panel 499): fold0 0.550 / 0.508, control (more
training, no simulation) 0.549 / 0.511, **simA** (50% GLACIER-simulated pool spectra) **0.575** / 0.519, simB (same,
answers' simulations excluded) 0.564 / 0.522; non-NP unchanged (0.80-0.81). Real gain, but below the pre-set kill rule
(+0.03 over control). Near-miss test on the calibrated slice (answer preferred over our #1): fold0 7/34, simA 7/34,
simB 9/34. As a same-formula re-scorer in the replay (ONP_EXTRA weight 0.5 / 1) every variant loses on the non-conj
in-pool slice (A: fold0 -0.041/-0.076, simA -0.030/-0.149, simB -0.032/-0.094; B: -0.031/-0.028, -0.019/-0.029,
-0.008/-0.013). Not integrated. Only untested route: swap in as engine 1's FP bank (needs a Kaggle GPU run + LB).

## De novo generation with FRIGID (2026-10-15, `bench/denovo/`)

FRIGID-base (MIST MassSpecGym encoder + DLM, official checkpoints, allowed per D746430) run locally on CPU with our own
token-length model (`frigid_eval.py`). Paper (arXiv 2604.16648): MassSpecGym top-1 16.1% (base), 18.3% (with ICEBERG
refinement). 12 non-conjugate class-3 molecules of ONP set B, true formula unless noted:

| Fingerprint given to FRIGID | top-1 | top-25 |
|---|---|---|
| true ECFP4 (oracle) | 7/12 | 9/12 |
| true, degraded to Tanimoto ~0.50 | 3/12 | 7/12 |
| true, degraded to Tanimoto ~0.31 | 1/12 | 2/12 |
| MIST prediction (Tanimoto ~0.31) | 0/6 | 0/6 |
| our #1 candidate's fingerprint + its formula (deployment) | 0/12 | 0/12 |

- **Memorization caveat:** none of the ONP answers are in FRIGID's exclusion list (`data/exclude_inchikeys.csv`, the
  MassSpecGym/NPLIB1 test structures masked from the ~230M-structure pretraining corpus), so the DLM may have seen them.
  Oracle (true fingerprint + formula) on molecules FRIGID never trained on (excluded MassSpecGym test, NP score > 1):
  12 out-of-pool, median 36 heavy atoms: top-1 1/12, top-25 2/12 (`frigid/msgx/`); **14 in the hidden test's mass range
  (279-374 Da; hidden test neutral mass median 328): top-1 0/14, top-25 4/14, all at ranks 7-8** (`frigid/msgx3/`).
  Same size range on ONP (24-41 heavy atoms): top-25 5/6. Most of the ONP oracle accuracy is memorization; hidden
  class-3 molecules are not in PubChem, so the honest ceiling with a perfect fingerprint is ~0 at rank 1.
- **Why deployment fingerprints fail (`fp_consensus.py`, 168 non-conj class-3 ONP molecules):** fingerprint of our
  #1 candidate has Tanimoto 0.48 to the truth (MIST 0.31), but FRIGID simply regenerates #1 (best generated structure
  is as close to the answer as #1 itself). The right screen is the margin T(fp, truth) - max T(fp, top-5 candidates):
  a randomly degraded true fingerprint at Tanimoto 0.31 is closer to the truth than to any top-5 candidate for 95% of
  molecules (margin +0.11); MIST at the same Tanimoto only 26% (margin -0.04); every blend of #1 / top-5 consensus /
  MIST tried is 0-15% (margins -0.07..-0.52). MIST's errors are structured toward known molecules, which are the ones
  already in our pool, so generation lands on candidates we already have.
- Conclusion: FRIGID de novo is not viable for class 3 here: deployment fingerprints have structured errors (land on
  pool molecules), and even a perfect fingerprint gives 0/14 at rank 1 on never-trained molecules. Direction stopped
  2026-10-15. Any new
  fingerprint model can be screened with the margin test in seconds, before any FRIGID run (target: closer to the
  truth than to the top-5 for a clear majority).

## Rule-based fragment explanation on the calibrated slice (2026-10-15 afternoon, `bench/netgen/iso_frag.py`)

Question: does a non-learned signal break the pattern that every learned model (FPNet, GLACIER, ICEBERG) agrees on the
wrong isomer? MAGMa combinatorial fragmentation (ms-pred's engine, as in `frag_rank.py`; prior art MetFrag, MAGMa) scores
each candidate by the share of measured sqrt-intensity explained by a fragment ion. Scored for the final list's #1 and
every same-formula candidate in the top 10 (rp_c2fix, ONP A d2 + B d0, non-conjugate class 2: 97 molecules, 59 at
rank 1; 27 near-misses where the answer is a same-formula isomer in the top 10, 31 hits with another isomer in the top 10).

| MAGMa setting | near-misses: answer above #1 (ties) | hits: #1 above every other isomer (ties) |
|---|---|---|
| depth 2, 15 ppm | 8 / 27 (11) | 16 / 31 (4) |
| depth 2, 5 ppm | 7 / 27 (12) | 16 / 31 (7) |
| depth 1, 10 ppm | 5 / 27 (21) | 13 / 31 (8) |
| depth 3, 10 ppm | 7 / 27 (17) | 11 / 31 (11) |

At best 8 possible fixes against 11 hits where another isomer would overtake #1, and many ties (same-formula isomers
explain the same peaks). Fragment explanation does not separate the isomers we lose on; not replayed, not integrated.
## Engine-bank swap prepared (not pushed)

- `simA.pt` / `simB.pt` have the engine's checkpoint format (same keys, nbits 10226, d 512, 8 layers) and load in
  `casmi.fpnet.ModelBank` locally, so the swap is mechanical.
- But they were fine-tuned from the fold-safe fold0, which never saw part of the training data, while the engine
  bank `fpnet_full1` was trained on everything; swapping fold0-derived weights in would confound "simulation helps"
  with "less training data". So prepared the deployment route instead:
  - `colab/casmi26_fpsim_bank.ipynb` (generator `bench/fpsim/make_colab_bank.py`): fine-tune `fpnet_full1` on real +
    the full-pool simulations (simA recipe, sim_p 0.5), no holdout (`train_sim.py --holdout ""`, new option), lr 3e-5,
    2 h on an L4; saves `full1_sim.pt` to Drive; last cell creates the private Kaggle dataset
    `giantwizard/casmi26-jc-fpsim-bank`. Not run.
  - `kernels/jc-v7c-simbank` (`make_v7c_variants.py simbank`): the 0.443 notebook with engine 1's bank =
    `full1_sim.pt`, asserting the dataset is attached (no silent fallback). Not pushed.
- Expectation is low: on the calibrated near-misses simA prefers the answer over #1 in 7/34, the same as fold0, and
  simB 9/34. It is the only untested route for the simulation work and needs one GPU run plus one LB submission.

## Next steps

1. jc-netgen scored 0.436 (2026-10-14): stop the generation direction. First LB result predicted in advance by the
   non-conjugate in-pool proxy.
2. After the GPU quota reset: push `jc-v7c-e2r2` (0.443, final-pick candidate); optionally `jc-netgen-probe` to count
   generated hits on the hidden public split.
3. When the Colab simulation finishes: run the training notebook; gate any new FP model with the non-conj in-pool proxy.
   For deployment, fine-tune from the engine's own bank (`ahmedberatozer/casmi26-fpnet-full1`) rather than fold0, at a
   small learning rate, so the v4b ranker's FP features stay close to what it was trained on; swap it in as the
   engine bank and judge it on the LB (the proxy cannot replay engine 1).
4. Any new idea: measure on the non-conjugate in-pool slice of sets A and B first; only large effects are resolvable.
5. Ruled out 2026-10-15 afternoon: MAGMa fragment explanation as an isomer discriminator (8 fixes vs 11 breaks at best).
   Ready if Jeffery wants the last simulation test: run `colab/casmi26_fpsim_bank.ipynb`, then push
   `kernels/jc-v7c-simbank` and submit (both need his OK).
