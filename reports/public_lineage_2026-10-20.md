---
status: active
area: portfolio
tags: [casmi26, kaggle, research, lineage]
updated: 2026-10-20
---

# Public notebook lineage, 2026-10-20

Source: `kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra --sort-by scoreDescending` (200
notebooks, pulled to `~/casmi26/lineage/all/`, parsed by `casmi26-repo/bench/lineage/extract.py` into
`~/casmi26/lineage/lineage.csv`). Kaggle's score sort is the reliable ranking; scores below are the authors' own
statements. Notebooks that state 0.455-0.470 in their text but sort below 0.452 notebooks (the rajrajak99 / nursrijan
"Apex Sovereign" series, mayankraaz "0.458+ SOTA") are treated as false or stale claims.

## The chain that moved the hidden score

| step | change | stated public LB |
|---|---|---|
| canqiang E006 (window 40, ICE fallback) | fusion window, ICEBERG where GLACIER is blind | 0.440 |
| antriksharya v7c + e2rank2 | engine 2's #1 inserted at rank 2 on disagreement | 0.442-0.443 |
| seyitkaangunes: PubChem promotion at rel > 150 (was > 250; > 100 gave 0.444) | trust PubChem's #1 more often | 0.446 |
| easonyanyan / saddamaditya: metric-identity E2R2 + CC BY ChEBI/LMSD 98k (sergiomalv) | more engine-2 candidates | 0.451 |
| seyitkaangunes N1A: GLACIER via [M+NH4]+ / [M-H2O+H]+ proxies | forward-model coverage | 0.452 |
| ahmedberatozer v14: + PubChemLite CC0 in engine 2 (franciscoangulo) | more engine-2 candidates | 0.455 |
| v15 = v14 + N1A; guhongbin v465: popularity weight 0.25 -> 0.15 for the PubChem-only list | prior tuning | >= 0.455 (sorts above v14) |
| saddamaditya dual gate: PubChem #1 promoted if rel > 150, or rel > 130 and pop >= 2.5 | trust PubChem when documented | sorts 2nd |
| dmitriigluzdov "Data Behind Spectra to Structures": own documentation prior, recent PubChem deposits, LOTUS/NPAtlas | prior + recall | 0.456, sorts 1st |

## Lesson

Every step after 0.443 is candidate recall (curated structure sets, PubChem) or the decision of when to trust the
PubChem channel's #1 (promotion thresholds, popularity gate). None is a better spectrum model or generation. Matches our
measurements: 45% of public molecules have no answer in our top 25; the PubChem channel alone scores 0.040 on the hidden
test; offline it ranks answers #1 that the pipeline drops. Related: [[forum_scrape_2026-10-14]], [[handoff]].

## LB results, 2026-10-20

- v465 copy (giantwizard/casmi26-jc-v465): **0.456**, matching the public number. This is a copy, not our improvement.
- Disagreement probe B (final #1 only, on molecules where the PubChem channel's #1 differs): **0.377**.
- Disagreement probe A (the channel's #1 on those same molecules): **0.000**.
- Reading, per the rule fixed before submitting: A < B. On the public split the channel's #1 is never right when v465 overrules it, so a better PubChem-trust gate has nothing to recover there. B ≈ 0.377 against a top-1 rate of about 0.40–0.41 means the channel disagrees with the final #1 on nearly every molecule v465 gets right.
- Next: candidate recall (about 45% of answers are absent from the top 25), using candidate-only source probes.

## Candidate-bank check, 2026-10-21

Source: dmitriigluzdov "The Data Behind Spectra to Structures" (pulled copy in ~/casmi26/lineage/all/), its Sections 4 and 6.
- Natural Product Knowledge Table: 152,717 LOTUS + NPAtlas natural products; 98.2% are already in the public candidate pool.
- Its version table, one change per version: natural-product bonus 0.451 -> 0.451; PubChemLite in engine 2 0.451 -> 0.448
  (0.455 only after the engine-2 trees went back to depth 6); LOTUS literature prior 0.448 -> 0.448; PubChem supplements
  (recent deposits) + external reference spectra: no public score given.
- The author's own conclusion: added catalogues also add competing structures that can displace correct answers.
Decision: no new candidate bank. LOTUS/NPAtlas would add about 2.7k structures, and the one-change-per-version scores show
no gain from the natural-product additions. The missing hidden answers are more likely in the pool and ranked too low, which
is the stacked reranker's job.

## Public frontier and forum, 2026-10-22

Public best scores (notebook pages, "Best Score"): 0.464 for dmitriigluzdov "The Data Behind Spectra to Structures" (V20),
guhongbin casmi26-v492-dg-v19, lavinwins casmi-26-lb-top-1, amrkhalid124 casmi26-upstream-data-v16, rajrajak99 apex-v800;
0.463 for ahmedberatozer v22-n1-challenger and v25-unq. Private copy pushed: giantwizard/casmi26-jc-v492 (guhongbin v492, unchanged).

Forum (discussion/750330, 751373, 751300):
- Host: open-source data with non-commercial terms (e.g. CC BY-NC) is now allowed; NIST remains banned, including derived weights.
- Host, 2026-10-22: weights trained on public libraries with public code are allowed; SIRIUS/CSI:FingerID are not;
  structure-only pretrained models are allowed; "We need the code that generates the intermediates as well"; PubChemLite is fine.
- Open question to the host: coleygroup MIST checkpoints released 2026-10-20 on Zenodo (record 23303289, MIT, trained on
  MassSpecGym, training code public). Not yet confirmed.
- Two participants report the same failure as ours: validation carved from training data improves offline but not on the LB.
