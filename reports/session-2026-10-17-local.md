# Session 2026-10-17/18 (local, Jeffery mostly away)

Competition deadline (Kaggle API, 2026-10-17): final submissions 2026-12-14 23:59 UTC; team-merger and new-entrant
deadline 2026-12-07 23:59 UTC. About 8 weeks left. 3,503 teams.

## 1. The long Colab FPNet run (`colab/casmi26_fplong.ipynb`)

What it was: fold-safe FPNet (`fpnet_fold0.pt`) fine-tuned on real training spectra + the full GLACIER simulation
(2,114,679 simulated spectra over 704,893 pool structures, [M+H]+ and [M+Na]+), 50% simulated per batch, hardest
same-window negatives (`--hard_k 16` of 63), bs 256, lr 1e-4 cosine, `--steps 65000 --hours 11`, L4 GPU on the alt
Google account (`bobofrogoo@gmail.com`), checkpoints to Drive `MyDrive/casmi26/fplong/`.

How it went:
- Started ~07:26 CT, ran without a disconnect, stopped at the 65,000-step cap after 639.7 min (10.7 h, before the 11 h
  budget). 1.69 it/s throughout, hourly partial checkpoints ~6,100 steps apart.
- Training metrics (EMA): rank loss 4.31 -> 0.81, 64-way accuracy 0.62 -> 0.84, still creeping up at the end
  (0.829 -> 0.839 over the last 15k steps at lr ~1e-5 -> 2e-6).
- **Bug: the hourly Kaggle publish only ever produced version 1** of `giantwizard/casmi26-jc-fplong` (08:26 CT,
  step-6096 partial + 4 KB log). Every later hour `kaggle datasets create` failed with "title already in use" (expected),
  and the fallback `kaggle datasets version` must also have failed, but both run with `check=False`, so nothing stopped
  or flagged it. Cause not identified: the cell-4 output in Colab holds the version command's stderr; read it before
  reusing `bench/fpsim/make_colab_long.py`. A local hourly pull meanwhile saved 10 identical copies of that partial to
  `~/casmi26/fplong/2026*` (~2 GB, safe to delete).
- The final checkpoint was recovered from the Drive-for-desktop sync on the Mac (no Kaggle needed):
  `~/casmi26/fp/ckpt/fplong.pt` (198,784,858 bytes, sha256 005367cf...0901e) + `train_fplong.log`.

Results (all on the Mac, MPS):

| `bench_np.py` MRR@25 | fold0 | simA (1 h pilot) | fplong |
|---|---|---|---|
| hold_np_clean (1,500) | 0.550 | 0.575 | **0.575** |
| top-1 | 39.7% | 43.0% | 42.4% |
| np_panel (499) | 0.506 | 0.519 | 0.516 |
| hold_nonnp (1,500) | 0.805 | n/a | 0.828 |

fold0 re-run locally reproduces the Colab number (0.5503 vs 0.5499). fplong = simA: 10x the simulated structures and
11x the training time added nothing on the fingerprint benchmark, still below the +0.03 kill rule.

As a same-formula re-scorer in the replay (`ONP_EXTRA`, outputs `mist/runs/fpsimtest/*fplong*`), conjugate-free slice
vs base2: A -0.011 (w 0.5) / -0.041 (w 1), B -0.006 / -0.005. Slightly less harmful than fold0 (-0.012 / -0.014 at
w 0.5), still negative.

**Engine-level test, run 2026-10-18 (refp cache harness, ranker retrained per bank, 2,086 class-2 NP queries):**
fold0 FP-alone class-2 MRR 0.343 / engine 0.761; simA 0.356 / 0.766 (+0.005, CI -0.001..+0.011); **fplong 0.365 / 0.763
(+0.001, CI -0.005..+0.007), fails the pre-registered bar.** A real +0.022 FP-alone gain bought less than the oracle value
curve predicts for that size (~+0.005): the model's improvements fall on molecules the other channels already solve.
Longer training of this recipe is closed. (Superseded text below kept for the record.)

~~Not yet run: the pre-registered engine-level test~~ (bank swap + retrained ranker on the NP harness,
`bench/engine/compare_banks.py`, pass bar class-2 +0.010 with CI > 0). That is the test this run was built for; the
two tests above do not replace it. Next step for this checkpoint: `rk/run_np1.sh fplong ~/casmi26/fp/ckpt/fplong.pt`,
then `compare_banks.py` against fold0.

## 2. Network rerank re-checked on the conjugate-free slices

`fp/net_rerank.py` on rp_c2fix, frozen config (th 0.7, conf top1, iters 2, pred), scored with
`bench/netgen/calibrate.py` vs base2:

| | A non-conj (tuned) | B non-conj (one-shot) |
|---|---|---|
| c2fix alone | +0.005 | +0.001 |
| net w 0.5 | +0.020 (6 fixed / 3 broke) | +0.006 (6 / 1) |
| net w 1 | +0.011 | +0.010 (8 / 2) |

First lever positive on the clean slice of both sets. n ~110 per set and Enveda NPs have fewer relatives (33% vs ~80%),
so the LB gain is probably small (UNVERIFIED guess +0.002 to +0.005).

**Built, not pushed to Kaggle or submitted: `kernels/jc-v7c-e2r2-net`** (generator `kernels/make_e2r2_net.py`) =
jc-v7c-e2r2 (public 0.443) + the rerank as a self-contained last cell (numpy + rdkit, no new dataset, reorders only).
Checks: embedded code reproduces `net_rerank.py` on both ONP sets (400/400, 447/447 rows identical); the full cell in a
mock Kaggle folder matches byte for byte (~9 s; A: 337/400 linked, top-1 changed for 59); a forced failure leaves
submission.csv unchanged. On Kaggle it reranks e2r2's final lists, not c2fix's, so the effect can differ.

## 3. Class-3 detector check (`bench/class3/detect.py`)

Label = answer not in the c2fix top 25. Features from the saved run outputs (engine list length, GLACIER score
top/gap/count, candidate-audit counts, engine/final top-1 agreement). Engine match scores were not saved in these runs.
Non-conjugate rows, logistic fit on one set, tested on the other: AUC 0.650 (A->B) / 0.727 (B->A); the top 10% flagged
are true misses 100% (13 rows) / 82% (11 rows), 70-77% at 30%. A detector can hand ~10% of rows to a generator almost
for free; +0.03 would then need ~0.3 MRR on those rows from the generator. FRIGID gave 0/14 top-1 on never-trained
molecules (2026-10-15), so generation is the bottleneck, not detection.

## 4. Research (de novo, 2026)

- MS-GPT (arXiv 2607.23607, code Apache-2.0 at github.com/VIKI623/MS-GPT, checkpoints on SJTU Cloud): SAFE-GPT
  molecule LM pretrained on ~100M structures, conditioned on a band of thresholded MIST fingerprint posteriors +
  formula. Reported top-1/top-10: MassSpecGym 23.9/28.7, NPLIB1 29.8/41.1 (FRIGID 18.3/22.0, 25.0/33.4 in their table).
  Uses the same MIST encoder whose structured errors sank FRIGID here, but its band querying targets exactly that
  failure. Untested; needs a host ruling and a GPU.
- MassSpecGym audit (arXiv 2606.19624): evaluation issues in 17 of 26 papers; MIST+MolForge corrected 31.8% -> 10.7%
  top-1; fingerprint-to-molecule decoders' gains are likely memorization of close pretraining analogs (they exclude only
  exact InChIKeys). Same pattern as our FRIGID result.
- End-to-end generators stay low: test-time-tuned transformer (arXiv 2510.23746) 3.2% MassSpecGym / 12.9% NPLIB1;
  DiffMS 2.3% / 8.3%; MADGEN 1.3% / 2.1%.

Decision tree for the remaining 8 weeks: `reports/decision-tree.md`.

## LB probes, 2026-10-18 (kernels/make_probes.py, submitted by Jeffery)

Five submissions on the public split (~124 molecules), all built on our copy of the public 0.443 notebook
(`kernels/jc-v7c-e2r2`, reproduced **0.443** exactly):
- top-1 only: **0.401** = share of molecules answered at rank 1.
- top 25 shuffled and padded to 25 with decoy alkanes (expected score = recall@25 x 0.15264 for any rank distribution;
  validated offline on ONP set B: estimate 0.256 vs true 0.255): seeds 1/2/3 = 0.091 / 0.087 / 0.075, mean 0.0843
  -> **recall@25 = 0.553 +- ~0.03**.

Breakdown of public molecules: 40% answered at rank 1; ~15% answered at ranks 2-25 (worth only 0.042 together, typical
rank 3-4); ~45% not in our top 25. With ~90% offline in-list recall for class 1/2, class 3 ~ 39% (range ~33-45%),
matching the 16/45/39 participant estimate (D741597) rather than ~52% or ~25%.
Implications: moving all rank 2-25 answers to rank 1 is worth up to ~+0.11 (a third of them ~+0.04): the reachable pool
for ranking work. Class 3 remains the largest block.

Same day, harness (refp cache): 3-model FP ensemble (fold0 + simA + fplong) +0.004 (CI -0.001..+0.010), fails;
ranker trained with 7,457 extra drug-like query rows -0.003 (CI -0.009..+0.002), fails.
Next: cross-encoder isomer reranker (casmi/xenc.py architecture exists, no weights or training code published).

## 2026-10-19: network rerank on the LB, cross-encoder pilot

- **jc-v7c-e2r2-net: public LB 0.422** vs 0.443 for the same notebook without the rerank (-0.021, ~2-3 public
  molecules). Offline it was positive on the conjugate-free slice of both ONP sets (+0.006..+0.020, n ~110 each): the
  first time that slice called a sign wrong. Likely: small n, the kernel reranked e2r2's final lists rather than c2fix's,
  and the hidden test has far fewer within-set relatives. Network rerank closed; the clean slice is a harm filter only.
- Cross-encoder (casmi/xenc.py, trainer bench/xenc/train_xenc.py, Colab T4 bs 16, ~30k steps on hold-fold spectra the
  frozen fold0 FPNet never saw): group accuracy gain over f.z flat at -0.005..+0.003 through the run; on 200 of its own
  training molecules (truth vs 15 hardest same-formula isomers) MRR 0.473 vs f.z 0.474; on the held-back NP half it
  hurts (isomer MRR 0.286 vs 0.378, worse on 406 / better on 38) and the harness gives -0.000 (CI -0.006..+0.006).
  Likely cause: the frozen FPNet encoder has already reduced the spectrum to fingerprint information. Not pursued
  further unless an unfrozen, f.z-free pilot is run (option prepared: --no_fz).
- **Generator-only LB probe** (kernels/probes/genonly: jc-gen's engine-1 generated structures alone, wide + default
  pass, in engine-1 rank order; everything else 'CCO'): **public 0.011**, i.e. about one hidden answer. Upper bound for
  any merge of this generator: +0.011 with zero displacement; jc-gen's merge lost -0.007. Analog generation (one/two
  biotransformation edits of library/pool relatives) is closed as a route to class 3; FRIGID de novo was ~0 on
  never-trained molecules. Remaining effort: compliance retraining (E1) and final picks.
