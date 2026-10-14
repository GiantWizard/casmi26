---
status: active
area: portfolio
tags: [casmi26, kaggle, session-log]
updated: 2026-10-14
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
| jc-netgen (this session) | pending at push time | network-guided analog generation |

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

## Next steps

1. Read jc-netgen's LB score; if ~0.438 or below, stop the generation direction (proxy predicted ~-0.005).
2. After the GPU quota reset: push `jc-v7c-e2r2` (0.443, final-pick candidate); optionally `jc-netgen-probe` to count
   generated hits on the hidden public split.
3. When the Colab simulation finishes: run the training notebook; gate any new FP model with the non-conj in-pool proxy.
4. Any new idea: measure on the non-conjugate in-pool slice of sets A and B first; only large effects are resolvable.
