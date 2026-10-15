---
status: active
area: portfolio
tags: [casmi26, kaggle, research, forum]
updated: 2026-10-14
---

# CASMI 2026 forum scrape, 2026-10-14

Full read of all 98 discussion topics (170 messages) through Kaggle's discussion API on 2026-10-14. Base URL for
every topic: `https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<id>`.
Quotes are from host (Enveda: David Healey, Marie Killian) or Kaggle staff (inversion) unless marked.

## Compliance: the ruling that matters most

- **Third-party weights need reproducible training code.** "Third party weights that cannot be reproduced with the
  original training code and permitted training data are not allowed. All solutions must be reproducible to ensure
  fairness. We reserve the right to check all submissions and potentially remove from the competition those that do
  not abide by this rule." (Marie Killian, 2026-10-14, D748882). Also: "a team cannot use the fine-tuned weights unless
  they can provide the original training code and access to the permitted data needed to reproduce them" (same post).
- Retraining a shared architecture yourself on permitted data, with your own training code, is sufficient (D748882,
  answer A-2, 2026-10-13).
- Non-compliance can mean removal from the leaderboard, not only losing prizes (D745841, 2026-10-05; D745615,
  2026-10-13). Hosts "reserve the right to check any and all submissions" (D748882).
- **Our exposure (checked 2026-10-14):** the pipeline loads weights from `ahmedberatozer/casmi26-v4b-models`
  (FPNet A, model_views FPNets, LightGBM ranker), `ahmedberatozer/casmi26-fpnet-full1` and
  `prvsiyan/casmi26-fp-models-v2`. Those datasets ship model and data-access code; no FPNet training script was found
  in them (only `ranker.py` contains training code). Gluzdov's fold-safe dataset provides a training loop for the same
  FPNet architecture (`dmitriigluzdov/casmi26-fold-safe-fpnet`, `train_fpnet.py`). UNVERIFIED whether the original
  authors published training code elsewhere. A user at rank 34 raised the same point (D745072, 2026-10-02: "The
  training scripts for the weights are not published").

## Other host rulings since 2026-10-11

- FRIGID official checkpoints (CC BY-NC) are **allowed** with a reference to the authors' training code and data
  (D746430 and D748882 B-2, 2026-10-13).
- Approved released checkpoints (MassSpecGym GLACIER/ICEBERG, DreaMS incl. the self-supervised backbone, FRIGID): ship
  the checkpoint, cite the authors' code and data, and include your own fine-tuning and inference code (D748882 A-1).
- ICEBERG PubChem atlas (predicted by a NIST'23-trained model) can be used (D745134, 2026-10-14).
- PubChem structures, per-compound CID-PMID / CID-Patent / CID-SID counts, LOTUS and NPAtlas are allowed (D749209,
  D748882, 2026-10-14). The approved list is not exhaustive (D748552, 2026-10-13).
- CC-BY-NC-SA and "other"-licence data "should be fine"; the concern is restrictive licences like NIST (D748363).
- TabPFN v2 and 2.5+ allowed with a reference to training code and data (D748882 B-1).
- Scorer = published first-match MRR (casmi-mean-reciprocal-rank v13); appending valid distinct guesses after an
  unchanged rank 1 never lowers a row (D748882 C-1).
- Electron-mass correction applied only to the hidden test; subtract 0.000548 Da from positive-mode m/z in train /
  Enveda-180 to match (D748882 C-2).
- All three novelty classes are present in both public and private splits (D748882 C-3).
- "By definition, class 3 structures are not in PubChem at all and must be predicted de novo." (D748552, 2026-10-13)
- All notebooks submitted before the metric/test update were rescored (D749430, 2026-10-14); our scores were unchanged.

## What competitors report (not host-verified)

- Public test ~132 molecules, "almost half of it are novel architectures" (Antriksh Arya, rank 57, D748029,
  2026-10-14). Bull's CV-fitted class shares put class 3 at ~52% (cited by Aryan Awasthi, D748029; source thread
  D748524, not independently checked).
- MarvinTMB (rank 3): analytical maximum ~0.76, realistic ~0.55 (D746302, 2026-10-09); "Why do you assume all the top
  is using forbidden code/data?" (D745072).
- Antriksh's masked probes on E006 (D749412): where engine 2's top-1 differs from the final #1, E006 is right ~13 times,
  engine 2 ~3; inserting engine 2's pick at rank 2 gives +0.002 (0.440 -> 0.442; v7c 0.441 -> 0.443).
- Aryan Awasthi (D749251, D748029, D748524): reorder gains transfer weakly (structure-disjoint CV +0.14 vs LB +0.03-0.04);
  the only change that moved the 0.437 lineage was adding an independent engine's candidates (0.437 -> 0.438);
  validates only on held-out Enveda-180 structures without public spectra (popularity #1 for 67% of library molecules
  vs ~0% of those).
- Others: "The public notebooks dont deal with class-3 molecules ... nobody have yet made any progress in it"
  (D749251, rank 682).

## Implications for us

1. If ~half the public test is class 3 and every public pipeline scores ~0 there, our 0.438 comes almost entirely from
   classes 1-2, and the gap to 0.49 most plausibly comes from de novo generation (FRIGID now allowed) or from a
   stronger class-2 model. Consistent with our calibration: in-pool tweaks barely move the LB.
2. Compliance risk for the final picks: the FPNet/ranker weights from other competitors' datasets may be removable.
   Mitigation: retrain the needed models ourselves with our own code on train.parquet (FPNet via Gluzdov's
   `train_fpnet.py`, ranker via `casmi/ranker.py`), and check whether the original authors published training code.
