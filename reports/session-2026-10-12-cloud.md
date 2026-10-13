---
status: active
area: portfolio
tags: [casmi26, kaggle, handoff, session-log]
updated: 2026-10-13
---

# Cloud session 2026-10-12/13: what was built, measured and left open

Branch `claude/eager-einstein-jn21vy` of `giantwizard/casmi26` has everything below. This session ran in an
ephemeral cloud container: scratch data (downloaded libraries, run outputs, key caches) is gone; anything needed
again is either in this repo, on Kaggle (datasets / kernel outputs), or rebuildable with the scripts listed.

## Leaderboard outcome (added 2026-10-13)

- `jc-gate` 0.438 (no change). **`jc-lr2` 0.410 (-0.028): the learned rerank hurts on the hidden test.**
- Set B (447 unseen NPs, never trained on) explains it: rerank all +0.050, but **in-pool -0.013 (3 fixed / 5 broke)**;
  set A in-pool was 5 / 5. The whole offline gain came from out-of-pool molecules whose answer only PubChem had
  (84-89% of the ONP sets, mostly Orbitrap). The hidden test is mostly the in-pool regime on timsTOF, which no
  offline set covered (the visible test is class 1). Gate on set B: +0.010 overall but -0.004 in-pool.
- Lesson: no candidate ships without an offline set that matches the hidden test's dominant regime
  (timsTOF, answer in pool, no library hit). Do not use jc-lr2 as a final pick; picks for now: jc-base2 / jc-gate.

## Bottom line (as written before the LB result)

- Best public score still 0.438. `jc-c2fix` was submitted and scored 0.438 (expected: its fix touches ~7% of molecules).
- **Ready to submit: `giantwizard/casmi26-jc-lr2` version 1** (commit run done, log confirms the learned rerank ran on
  40/40 smoke molecules, 0 fallbacks). It contains: gentle PubChem gate + fine-tuned-GLACIER fix + learned final rerank.
- Kaggle GPU quota: ~1 h of 30 h left this week (most used by this session's eval runs). Resets weekly (believed
  Saturday 00:00 UTC; check the quota panel). New notebook versions need a commit run (~35 min for these notebooks).
- Not yet done: independent check of the learned rerank on **ONP set B** (447 unseen NPs). Its pipeline outputs
  come from the Colab runner (job `jc-onp-eval-b-d0`), which was still running at handover.

## Offline evaluation sets ("ONP" = obscure natural products)

Built by `bench/obscure_np/build_obscure_np_set.py` from GNPS + MassBank + Enveda-180 (no NIST), keeping only
natural products unseen by train (scorer key and tautomer/salt parent key), MassSpecGym and the GLACIER fine-tune
labels. Scorer key = the metric's (RemoveStereochemistry -> TautomerEnumerator -> InChIKey14).

| Set | Kaggle dataset | Molecules | Notes |
|---|---|---|---|
| A | `giantwizard/casmi26-jc-onp` | 400 (5 draws) | 63 in train+COCONUT pool, 337 outside; ~98% Orbitrap |
| B | `giantwizard/casmi26-jc-onp-b` | 447 (2 draws) | disjoint from A; 83 in pool |
| Visible test | competition `test.parquet` | 400 | train-derived = class-1-like; truth recovered by matching spectra to train rows (adduct, precursor m/z, base-peak intensity, peak count) |

Eval notebooks (`kernels/make_onp_eval.py`) run jc-c's pipeline once and emit base2/c/c2/c2fix variants;
`bench/obscure_np/replay_rerank.py` replays cell 19 offline from a run's saved scores (byte-identical to Kaggle).
Score with `bench/obscure_np/mrr_eval.py` (paired bootstrap vs a baseline, slices by class/adduct group).

## Findings (all measured; MRR@25)

1. **Bug: jc-c2's fine-tuned GLACIER never ran.** ICEBERG returns an all-None dict for molecules it can't score, so
   cell 19's `if ice:` was true and the `elif glft_of(mid)` branch was skipped. jc-c2 == jc-base2 byte for byte;
   that is why both scored 0.438. Fixed in `jc-c2fix` (drop all-None shell dicts). Effect ~0 (noise).
2. **PubChem library gate (cell 17) was throwing away the right answers.** `lib_max >= 0.9` skips the PubChem merge;
   on unseen molecules it fired 330/400 (analog/isomer library hits). Of 44 out-of-pool truths present in the
   PubChem tier, the channel found 36 and ranked 21 first, but only 4 reached the final top 25.
   *Gentle gate* (interleave PubChem at slots 4,8,12,16,20 when lib_max >= 0.9; no promotion):
   set A draw 0 +0.011 (CI +0.007..+0.015), draw 1 +0.013 (CI +0.008..+0.017), visible test +0.002 (noise).
   Built into `jc-gate` (= jc-c2fix + gentle gate).
3. **Rejected levers:** confidence promotion of the PubChem top-1 (+0.031 on A but **-0.195** on the visible test);
   fixed fusion weights (36-setting sweep of alpha/krr/GLACIER weight: nothing wins on both sets); the old
   `rel > 250` promotion trigger is net negative on unseen molecules; FIORA as a fixed-weight rerank term (+0.012..+0.018
   on A at w>=2 but -0.008..-0.040 on the visible test).
4. Small win: GLACIER weight 0.5 when main list and engine 2 agree on top-1 (+0.005 visible, ~0 on A; ~+0.001 LB).
5. **Learned final rerank (biggest lever).** LightGBM lambdarank over each fused list (features: fused score/rank,
   rank in each engine, PubChem rank, pool popularity, ICEBERG/GLACIER/fine-tuned GLACIER scores raw and within
   same-formula group, group size), reordering only inside formula groups ("slots"). Trained jointly on set A +
   visible test (training on A alone destroys class-1 molecules: visible 1.00 -> 0.24).
   2-fold CV x5 seeds vs jc-gate: **unseen +0.042 (CI +0.027..+0.057; 25 fixed / 6 broke)**, library-match
   +0.005 (noise; 6 / 2). Gain comes from out-of-pool molecules whose answer the PubChem channel found.
   (Draw 1 shows +0.082 but the model trained on the same molecules, so that is not independent.)
6. **FIORA as a rerank feature:** unseen 0.198 -> 0.205 (+0.007, all 5 seeds up), library-match unchanged. Not yet
   in a submission notebook (would be `jc-lr3`; needs FIORA scoring in-notebook, ~30 min CPU).
7. Extra reference spectra lever built (Enveda-180 timsTOF + newer GNPS/MassBank, 2,721 structures absent from
   train; `giantwizard/casmi26-jc-extralib`), only engine 2 uses them. Unmeasured; its Colab jobs are queued. On ONP
   sets it can only show harm/analog help, not the class-1 hit gain it targets. 865 Enveda-180 structures have only
   water-loss spectra (possibly why hosts filtered them).

Leaderboard context (public, Oct 12, 3,160 teams): 529 teams >= 0.438, 412 >= 0.440 (public-notebook wall),
40 >= 0.445, 17 >= 0.450 (~gold), 6 >= 0.460, top 0.491.

## Kaggle state

- Notebooks (private): `casmi26-jc-c2fix` (submitted, 0.438), `casmi26-jc-gate`, `casmi26-jc-lr2` (ready),
  eval runs `casmi26-jc-onp-eval-d0/d1/vis` (outputs hold all saved scores).
- Datasets: `casmi26-jc-onp`, `casmi26-jc-onp-b`, `casmi26-jc-lr` (model `lr_model.txt` + `lr_core.py`),
  `casmi26-jc-extralib`, `casmi26-jc-colab-jobs` (Colab job list), `casmi26-jc-colab-out` (Colab results).
- Note: a kernel slug equal to a dataset slug (`casmi26-jc-lr`) is refused with HTTP 409; hence `jc-lr2`.

## Colab runner (offloads GPU work from Kaggle quota)

`colab/casmi26_colab_runner.ipynb` (generated by `kernels/make_colab_runner.py`): reads jobs from
`casmi26-jc-colab-jobs`, downloads each job's Kaggle inputs to `/content/kaggle/input` (Colab's own `/kaggle/input`
is read-only), runs the notebook's code cells as a plain script on a uv Python 3.13 venv (pinned RDKit wheels are
cp313; Colab's IPython config breaks a separate kernel, so no papermill), uploads `<job>__<file>` results to
`casmi26-jc-colab-out`, marks DONE only on success. Needs Colab secret `KAGGLE_API_TOKEN`; L4 recommended.
Queue: `jc-onp-eval-b-d0`, `-b-d0-xlib`, `-d2`, `-b-d1`, `-d2-xlib`. Add jobs with
`python kernels/make_colab_runner.py --jobs ...` then `kaggle datasets version -p colab/jobs --dir-mode zip`.
First real Colab run was in progress at handover (earlier attempts failed on setup and were fixed).

## How to finish the open checks

1. Set B check of the learned rerank: when `jc-onp-eval-b-d0__DONE` appears in `casmi26-jc-colab-out`, download it,
   then: `gate_variants.py` -> `replay_rerank.py --v4 gentle=...` -> `rerank_features.py` (main list = the gentle
   list, `--solution` = set B `solution.csv`) -> apply `lr_model.txt` with `lr_core.slot_order` -> `mrr_eval.py`
   vs `rp_v4gentle.csv`. Promote only if unseen gains and nothing breaks. Repeat with FIORA features
   (`kernels/fiora_runner.py` runs on CPU, ~25 min per set) to decide on `jc-lr3`.
2. After the quota reset: build `jc-lr3` (jc-lr2 + FIORA scoring cell + model retrained with `lr_core.FEATS_F`,
   ideally on A + B + visible), consider cutting the smoke test from 40 to ~5 molecules to save commit time.
3. Final picks: one evidence-backed candidate (jc-lr2/jc-lr3 if set B confirms) + one conservative (jc-gate or jc-base2).

## Key files

- `bench/obscure_np/`: `build_obscure_np_set.py`, `common.py`, `mrr_eval.py`, `replay_rerank.py`, `gate_variants.py`,
  `promote_after_fusion.py`, `pubchem_coverage.py`, `stage_recall.py`, `rerank_features.py`, `train_rerank.py`,
  `lr_core.py`, `build_extralib.py`, `replay_fiora.py`
- `kernels/`: `make_onp_eval.py`, `make_lr_notebook.py`, `make_extralib_eval.py`, `make_fiora_eval.py`,
  `fiora_runner.py`, `make_colab_runner.py`, notebook dirs (`jc-gate`, `jc-lr2`, `jc-c2fix`, eval variants)
- `colab/casmi26_colab_runner.ipynb`
