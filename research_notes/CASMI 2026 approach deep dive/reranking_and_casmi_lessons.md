---
status: active
area: portfolio
tags: [casmi26, research, reranking, learning-to-rank, ensembling]
updated: 2026-10-11
---

# Reranking, Score Fusion, Ensembling, and Lessons from CASMI 2012-2022 (for Enveda CASMI 2026 on Kaggle)

State as of 2026-10-11. Every claim carries an inline source link. Items I could not verify are in the Gaps sections. "Inference" bullets are my own reasoning from the cited findings and are not sourced facts.

Context for the reader: the current pipeline fuses a LightGBM lambdarank ranker (about 160 features) with a second engine via reciprocal-rank fusion (RRF), then re-ranks same-formula groups using z-scored forward-model similarities plus a popularity prior; the metric is MRR@25. 96% of misses are same-formula isomers. Adding a second, correlated forward-model score at full weight dropped public LB from 0.438 to 0.414.

## Q0. What is CASMI 2026 itself (facts that shape strategy)?

### Takeaway
CASMI 2026 is a 90-day Kaggle competition run by Enveda with about 400 test molecules represented by roughly 2,500 unpublished spectra, and about 2.5 million training spectra from 275,000 molecules (open libraries plus Enveda-180). This means several spectra per test molecule, and a small effective number of independent test units.

### Cited Findings
- Enveda announced CASMI 2026 as a 90-day ML/AI competition hosted on Kaggle, the first CASMI since 2022 and the first aimed at the broad ML community (Source: [Business Wire, 2026-09-15](https://secure.businesswire.com/news/home/20260915334355/en/Enveda-Is-Launching-the-Next-AlphaFold-Contest.-This-Time-to-Solve-the-Structures-of-90-of-Biological-Molecules-That-No-One-Can-Identify), seen via search snippet; the page itself returned HTTP 403 to direct fetch).
- Test set: about 400 molecules represented by roughly 2,500 spectra, none previously published, generated on Enveda's own platforms; training: about 2.5M spectra from 275,000 molecules including the newly released open Enveda-180 dataset; winning methods to be open-sourced and the test kept as a persistent benchmark (same Business Wire source, search snippet).

### Inferences
- About 6 spectra per molecule on average. If test spectra of the same molecule can be grouped legitimately (same precursor m/z, same RT if given, or explicit IDs), aggregating evidence across them (summing log-scores, or averaging forward-model similarities over collision energies) is a cheap test-time gain. This must be checked against the competition rules and data format.
- Effective sample size for LB noise is closer to the number of molecules in the public split than to the number of spectra, because spectra of one molecule are strongly correlated. With MRR per-query standard deviation typically around 0.4 and perhaps 100-150 molecules in a public split, the standard error of public MRR would be roughly 0.03-0.04. Under that assumption the 0.438 to 0.414 drop (0.024) is within about one standard error and is weak evidence by itself. The public split size is not something I verified; plug in the real number.

### Gaps
- Exact public/private split proportions, whether spectra are grouped by molecule in the test file, instrument types (Orbitrap vs timsTOF vs QTOF), ionization modes, collision energies, and candidate-set definition were not found in sources I could read. Check the Kaggle data page.

## Q1. CASMI 2012-2022: results, winning approaches, organizer conclusions

### Takeaway
Across every CASMI round, the biggest single lever was metadata/database priors (reference counts, database membership, lowest database ID), which took Top-1 from roughly 30% (in-silico only) to roughly 70%+; second, equal-weight consensus of complementary in-silico methods beat every single method; and failures concentrated on positional isomers, formula errors, and obscure compounds where priors mislead.

### Cited Findings
**CASMI 2012**
- Winners: Category 1 (formula) Dunn et al., combining automated formula generation and database search (KEGG, ChemSpider, MetFrag) with manual expert review; Category 2 (structure) Oberacher using automated searches across several spectral libraries (MassBank, METLIN, NIST, MSforID) (Source: [CASMI: And the Winner is..., PMC3901266](https://pmc.ncbi.nlm.nih.gov/articles/PMC3901266/)).
- Lessons: spectral library search was unbeatable when the compound was in a library; small specialized DBs (KEGG) ranked well when they contained the answer but missed it more often; an overly strict 5 ppm window and calibration problems caused misses; several methods returned the positional isomer 2-aminoanthraquinone instead of 1-aminoanthraquinone (Tanimoto 0.842) (Source: [PMC3901266](https://pmc.ncbi.nlm.nih.gov/articles/PMC3901266/)).

**CASMI 2013**
- Nikolic and Newsome won by manual interpretation plus database searching, identifying 15 of 16 challenge molecules (Source: search summary of [CASMI Birmingham/Nikolic materials](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3901277/); not independently re-read).
- MAGMa was selected as the best automated tool of CASMI 2013 (Source: search snippet referencing [PMC4337369](https://pmc.ncbi.nlm.nih.gov/articles/PMC4337369)).

**CASMI 2014**
- CFM-ID won the automated category, correctly identifying 24 of 42 challenge compounds and placing the correct candidate in the top 10 for 33 (Source: search snippet of [CFM-ID CASMI 2014 paper, PMC4337369](https://pmc.ncbi.nlm.nih.gov/articles/PMC4337369); unverified by full-text read).
- MAGMa (Ridder) retrieved candidates from HMDB and PubChem; correct formula ranked top in 40 of 48, correct structure top in 22 of 42 (same snippet source; unverified).

**CASMI 2016 (Schymanski et al., J. Cheminform. 2017)**
- 208 challenges (127 positive, 81 negative); ChemSpider candidates retrieved via MetFrag2.3 at ±5 ppm, averaging about 1,114 candidates each (Source: [Schymanski et al. 2017, PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- Category 2 (in-silico only): CSI:IOKR (Brouard) won with 86 gold (41%); CSI:FingerID (Dührkop) 82 gold, 70 Top-1 (34%), 100 Top-10; MS-FINDER 70 gold; CFM_orig had the most Top-10 hits (123, 59%) (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- Category 3 (with metadata): MS-FINDER+MD (Kind) won with 159 gold and 146 Top-1; CFM-ID (Allen) 156 gold; top Top-1 about 70%, Top-10 above 80% (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- Metadata alone: ChemSpider reference count gave 143 Top-1 (median rank 1); lowest ChemSpider ID alone gave 113 Top-1, which beat the combined fragmenters; RT alone gave 1 Top-1; MoNA library alone 70 (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- Post-contest consensus: equal-weight combination of six fragmenters (CFM_orig, CSI:FID, CSI:IOKR_A, MAGMa+, MetFrag, MS-FINDER) reached 82 Top-1, beating the best single entry (70); adding RT and references reached 164 Top-1 and 194 Top-10; adding MoNA gave no further Top-1 gain, which the authors attribute to an over-optimistic MoNA weight (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- In negative mode, combining MetFrag (9 Top-1) and CFM (12 Top-1) gave 20 Top-1 (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- Training data mattered: retraining CSI:IOKR on more data raised Top-1 to 78; a leave-out CSI:FID with all challenge compounds removed from training dropped positive-mode Top-1 from 70 to 36 (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- Organizer conclusions: ML and chemistry-based fragmenters are complementary and consensus is recommended; metadata is essential for known substances but can mislead method evaluation; remaining hard cases are true unknowns, positional isomers, and stereoisomers (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)). The paper does not state the normalization used for the equal-weight combination in the main text (details are in Additional file 1, not read).
- Blaženović et al. 2017 on CASMI 2016 data: single-tool in-silico Top-1 on validation (208 spectra) MetFragCL about 25%, CFM-ID 14%, MAGMa+ about 13%, MS-FINDER about 11%; MetFragCL-based voting models did not beat MetFragCL alone on validation; adding database-membership and MS/MS-library boosting gave 87% Top-1 on validation (CFM-ID + ID-sorted + DB + MS/MS), and boosting alone was nearly as good as the full model, so the choice of in-silico tool mattered little once boosting was applied (Source: [Blaženović et al. 2017, PMC5445034](https://pmc.ncbi.nlm.nih.gov/articles/PMC5445034/)).
- Blaženović boosting formula: final = in-silico consensus rank + DB presence (13 metabolomics DBs, 220,213 entries) + 2 × STOFF-IDENT presence + 4 × MS/MS library hit (NIST/MassBank dot product > 400, binary); the 93% headline is on the tuning set, 87% on validation (Source: [PMC5445034](https://pmc.ncbi.nlm.nih.gov/articles/PMC5445034/)).

**CASMI 2017**
- SIRIUS/CSI:FingerID won Categories 1, 2 and 4; in Category 4 it identified 66 of 198 (33.3%), more than 6x the best non-CSI:FingerID entrant; candidates were restricted to SIRIUS's top formula, and missing isotope patterns in 2017 data caused formula errors that excluded correct structures (Source: [Jena group post](https://bio.informatik.uni-jena.de/2017/11/casmi-2017-results-are-out)).
- EPA retrospective: in 2017, combining metadata (data source counts) with CFM-ID lowered top-5 performance (130 vs 144 for CFM-ID alone), because the true compounds were obscure; registering missing structures raised 2017 top-5 nearly threefold (Source: [Revisiting Five Years of CASMI with EPA tools, PMC7345619](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345619/)).

**CASMI 2022 (Fiehn lab, raw LC-MS/MS, 500 compounds, ±ESI)**
- Scored separately on adducts, formulas, ClassyFire class, and 2D structure. 2D structure was hardest: Dührkop 26% (priority set) / 29% (bonus), Nothias and Rutz 25% / 28%, Nikolic_KUCA 38% on only 55 attempts; best formula accuracy about 94-95% (Dührkop, Nothias, Rutz); adducts up to 93-96% (Source: [Fiehn lab CASMI 2022 results](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)).
- Organizers framed 2022 as a trial "aimed mostly to gauge which methods are used by the community" (Source: [Fiehn lab CASMI 2022 results](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)). Per-team methods were in an external Google Drive, not read.

### Inferences
- Pattern across rounds: the gap between formula accuracy (about 94%) and structure accuracy (about 26-29%) in 2022 mirrors the 96% same-formula-miss profile of the current pipeline. Structure discrimination within a formula is the field-wide bottleneck, not something peculiar to this pipeline.
- Popularity priors are double-edged: very strong for "known knowns" (2016) and harmful for obscure compounds (2017). Enveda's test set is drawn from its own natural-product-heavy platforms and described as unpublished spectra; if the molecules skew obscure, a popularity prior at fixed weight may be miscalibrated relative to the public training libraries. Its weight should be tuned on a validation split that mimics the test's obscurity, not on library-wide CV.
- Train/test structure leakage inflated CASMI 2016 results (70 vs 36 Top-1 for CSI:FID with and without overlap). Any CV for this competition should be grouped by structure (InChIKey first block), else the ranker learns to trust leaky features.

### Gaps
- Official CASMI 2013 and 2014 overview papers were not read in full; numbers from search snippets are marked.
- No published CASMI 2022 overview paper with methods was found.
- The exact normalization behind the CASMI 2016 "Combined MS/MS" entry is in Additional file 1, not read.

## Q2. Score fusion and learning-to-rank for metabolite ID; calibrated vs raw scores

### Takeaway
Published systems almost always normalize each score within the candidate list (divide by max or rank-transform) before a weighted sum, and tune the weights jointly on held-out data with a sum-to-one constraint; the best confidence work (COSMIC) shows that per-query context (score gap to runner-up, candidate count) matters more than raw score level. Rank-based fusion such as RRF is robust but discards score margins, which is exactly the information that separates isomers.

### Cited Findings
- MetFrag 2.2 final score is a weighted sum of terms (fragmentation, RT, references, suspect lists, user scores), each normalized to the maximum value within the candidate list; weights were chosen by 1,000 random draws constrained to sum to 1; suggested defaults about ωFrag 0.5, ωRT 0.2, ωRefs 0.3 (Source: [MetFrag relaunched, PMC4732001](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732001/)).
- On 473 Orbitrap spectra vs PubChem: fragmentation only 30 Top-1; references only 311; all three 325 (69%); optimized weights 336 (71%); ChemSpider all three 411 (87%), optimized 420 (89%) (Source: [PMC4732001](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732001/)).
- Combining MetFrag with CFM-ID at weights 0.67/0.33 roughly doubled Top-1 (30 to 61/62) and improved median rank from 12 to 8; with RT and references, 343 Top-1 (73%) (Source: [PMC4732001](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732001/)).
- MetFrag 2.4.5 learned peak and neutral-loss annotation probabilities (Bayesian mean posterior with Dirichlet pseudo-counts), max-normalized each term, and grid-searched weights summing to 1; Top-1 on 87 CASMI 2016 spectra rose from 5 to 21 (Source: [Ruttkies et al. 2019, PMC6612146](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612146/)).
- EPA approach: equal-weight sum of the normalized data-source count and normalized CFM-ID match score within each candidate set; the authors note low raw match scores can still be correct because the true compound ranks near the top of its set, and recommend reporting raw and normalized scores together (Source: [PMC7345619](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345619/)).
- COSMIC: CSI:FingerID raw scores are converted to E-value-like quantities via kernel density estimates over PubChem candidates (a decoy stand-in); a linear SVM with enforced feature directionality ("a high CSI:FingerID score of a hit should increase, but must never decrease, our confidence") combines features; score gap to the runner-up and number of candidates were among the most important; separate SVMs were trained for single-candidate queries; Platt scaling maps outputs to posterior-like values (Source: [Hoffmann et al. 2021, PMC8926923](https://pmc.ncbi.nlm.nih.gov/articles/PMC8926923/)).
- COSMIC on CASMI 2016: AUC 0.82 vs 0.40-0.55 for MetFrag, MAGMa+, CFM-ID and raw CSI:FingerID; on a Sciex dataset raw CSI:FingerID scores were worse than random for confidence (Source: [PMC8926923](https://pmc.ncbi.nlm.nih.gov/articles/PMC8926923/)).
- LC-MS2Struct (Bach et al. 2022) is a structured SVM that jointly combines any MS2 scorer (CFM-ID, MetFrag, SIRIUS) with retention-order information across all features in a sample, improving MS2 scorer accuracy by up to 106% (published) and outperforming RT filtering and logP/RO-prediction baselines; stereochemistry-aware fingerprints helped (Source: [Bach et al. 2022 PDF](https://orbilu.uni.lu/bitstream/10993/54236/1/Bach_etal_2022_LCMS2Struct_s42256-022-00577-2.pdf); preprint figure was up to 66.1%, [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.02.11.480137v2)).
- RRF (Cormack, Clarke, Büttcher 2009) with k=60 fixed in a pilot "consistently equaled or bettered" Condorcet Fuse and CombMNZ; MAP gains were consistent, P@10 results were mixed (Source: [Cormack et al. 2009 PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf); [Cormack fusion chapter](https://cormack.uwaterloo.ca/fusion.pdf)).

### Inferences
- The 0.438 to 0.414 drop is consistent with what MetFrag/CASMI 2016 found: an extra score helps only at a jointly tuned weight, and over-weighting a correlated or noisy source hurts (the MoNA case in CASMI 2016). Concretely: (a) put both forward-model similarities on the same per-group scale; (b) fit weights jointly on grouped-by-structure OOF data with a sum-to-one constraint (random search or grid, as MetFrag did), or feed both as features into the lambdarank model and let it learn them; (c) consider using the residual of the second score after regressing on the first (orthogonalization), so the reranker sees only new information.
- Per-group z-scoring is a reasonable analogue of max-normalization, but with small isomer groups (2-5 candidates) z-scores are unstable and can amplify noise. Alternatives worth testing: rank-within-group, max-normalization, softmax with a learned temperature, or z-score shrunk toward zero when group size is small. Also add group size and top-two gap as features (COSMIC found these among the most informative).
- RRF between the LightGBM ranker and the second engine throws away score margins. For MRR the key decision is which candidate goes first within a formula group; a calibrated score fusion (e.g., logistic regression stacked on OOF scores from both engines plus group size and gap features) usually preserves more of that margin than rank fusion. RRF's benefit is robustness when scores are uncalibrated. Testing a weighted RRF (per-engine weights, tuned k) is a minimal step.
- COSMIC's directionality constraint is a useful guard: monotone constraints in LightGBM on similarity features (higher similarity never lowers rank) reduce overfitting with a small validation signal.

### Gaps
- No paper found that directly compares per-group z-scoring vs max-normalization vs rank transforms for MS/MS candidate fusion.
- Passatutto (decoy-based FDR for spectral library search) was not researched in this pass.

## Q3. Ensembling fingerprint predictors and forward models; diversity vs correlation; stacking

### Takeaway
Gains from ensembling in this domain came from combining methodologically different evidence (ML fingerprint prediction vs combinatorial fragmentation vs metadata), while combining similar methods gave little. Stacking-style weight fitting on held-out data was the norm; voting over correlated tools did not beat the best single tool on validation.

### Cited Findings
- CASMI 2016 organizers: combined fragmenters outperformed every individual entry, "showing the complementarity of the different approaches"; they note MetFrag and MAGMa+ behave similarly "as both are combinatorial fragmentation approaches" (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- Blaženović: MetFragCL-based voting models did not beat MetFragCL alone on validation, while adding MS-FINDER to a consensus gave no Top-1 gain but 9% more correct structures in the Top-10 (Source: [PMC5445034](https://pmc.ncbi.nlm.nih.gov/articles/PMC5445034/); Top-10 detail from search snippet of the same paper at [Springer](https://link.springer.com/article/10.1186/s13321-017-0219-x)).
- CSI:FingerID itself is an internal ensemble: multiple kernels (fragmentation tree kernels, element kernel, recalibrated MS/MS kernel, alignment pseudo-kernels) combined with MKL (ALIGNF) weights; the earlier MKL-on-fragmentation-trees work roughly doubled top-ranked metabolites (Source: [Shen et al. 2014, PMC4058957](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4058957/); [CSI:FingerID PNAS 2015, PMC4611636](https://pmc.ncbi.nlm.nih.gov/articles/PMC4611636); [INRAE note on ALIGNF](https://hal.inrae.fr/hal-02786616v1/document)).
- In a recent competition review, almost all well-performing submissions used ensembles (Source: [CEUR-WS paper369](https://ceur-ws.org/Vol-4283/paper369.pdf), via search snippet).

### Inferences
- For the isomer problem, the most valuable additional engine is one whose errors are least correlated with the current forward model (e.g., a fingerprint/retrieval model or a combinatorial fragmenter if the current forward model is neural, or a different architecture trained on different data). Measuring OOF error correlation (or the fraction of misses one engine fixes that the other does not) before adding an engine is a cheap way to predict whether fusion will help.
- Stacking with OOF predictions grouped by structure is the analogue of how MetFrag and Blaženović fit weights on a training split and checked on a separate validation split; the 93% vs 87% train/validation gap in Blaženović shows how much in-sample weight tuning overstates gains.

### Gaps
- No published head-to-head of ensembles of neural fingerprint predictors (e.g., MIST variants) vs a single model on CASMI-like data was found.
- ICEBERG isomer-specific retrieval numbers were not found ([ICEBERG arXiv 2304.13136](https://arxiv.org/abs/2304.13136v2) was located but its retrieval tables were not read).

## Q4. Test-time / domain adaptation across instruments

### Takeaway
I found little direct published evidence quantifying cross-instrument (Orbitrap vs QTOF vs timsTOF) degradation for structure retrieval, or gains from instrument-matched fine-tuning; the evidence that exists is indirect (training data overlap effects, raw scores becoming uninformative on a new dataset, collision-energy conversion in tools).

### Cited Findings
- Instrument-method mismatch between library and user spectra is identified as a core challenge for ML annotation (Source: [PMC12143282](https://pmc.ncbi.nlm.nih.gov/articles/PMC12143282/), search snippet only).
- On an independent Sciex dataset, raw CSI:FingerID scores were worse than random for confidence while calibrated COSMIC still worked, which suggests score calibration shifts across datasets/instruments (Source: [PMC8926923](https://pmc.ncbi.nlm.nih.gov/articles/PMC8926923/)).
- The MIT ICEBERG retrieval tool accepts multiple collision energies and converts normalized collision energy (NCE) to absolute eV using precursor m/z (Source: [ICEBERG retrieval tool](https://iceberg-ms.mit.edu/)).
- EPA workflow summed CFM-ID cosine matches over 10, 20 and 40 eV predictions (max 3.0) rather than choosing one energy (Source: [PMC7345619](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345619/)).
- Fine-tuning a pretrained spectrum predictor on a small specialized library significantly improved prediction for that domain (peptides, not small molecules) (Source: [Waterloo thesis 2024](https://uwspace.uwaterloo.ca/handle/10012/20364)).
- MassSpecGym "in the wild" audit (June 2026) found evaluation issues (data leakage, shortcut learning, implementation bugs/metric divergence) in at least 17 of 26 papers reporting MassSpecGym results and released MassSpecGym v1.5 (Source: [arXiv 2606.19624](https://arxiv.org/abs/2606.19624)); MIST-CF formula top-1 rose from 50.5% to 66.5% when trained on test samples, and a public ICEBERG checkpoint trained on overlapping data was flagged as a leakage risk (Source: search snippet of [arXiv 2606.19624 PDF](https://arxiv.org/pdf/2606.19624); not verified in full text).

### Inferences
- Practical, low-risk test-time steps: (1) NCE to eV conversion and matching forward-model predictions to the test energy, or averaging similarities over several predicted energies as EPA did; (2) re-fitting fusion weights and z-score statistics on Enveda-instrument training spectra (Enveda-180) rather than on pooled public libraries, since calibration shifts by dataset; (3) if Enveda-180 shares instruments with the test, fine-tune or weight those spectra up and validate on a held-out Enveda-180 split grouped by structure.
- Pretrained public checkpoints may have trained on structures that appear in public validation splits; this inflates CV and can mislead weight tuning (per the MassSpecGym audit).

### Gaps
- No benchmark found quantifying Orbitrap vs timsTOF vs QTOF retrieval degradation, or gains from instrument-matched fine-tuning of fingerprint predictors for small molecules.

## Q5. Kaggle reranking lessons transferable to MRR@25

### Takeaway
Top Kaggle retrieval solutions use multi-stage retrieve-then-rerank with a stronger reranker on a shrinking candidate list; final-submission choice should rest on robust grouped CV and simulated public/private splits, not small public LB moves.

### Cited Findings
- Eedi Mining Misconceptions (metric MAP@25, which equals MRR@25 when there is one relevant item): 1st place used a multi-stage retrieve-and-rerank pipeline with fine-tuned Qwen retrievers and a reranker; 2nd place used listwise reranking of the top 25; a 13th-place pipeline reranked top 100 with a 32B model then top 25 with a 72B model (Source: [Eedi blog](https://www.eedi.com/news/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions); [Kaggle discussion](https://kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551651), via search summary).
- Grandmaster practice: build robust local validation first and iterate against it; simulate public/private splits with multiple CV folds to estimate shake-up; treat the public LB as a holdout to avoid adaptive overfitting (Source: [NVIDIA Kaggle Grandmasters blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/); [Becoming a Kaggle Master slides](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf)).
- Overfitting to public LB is a recognized failure mode (Source: [H2O grandmaster panel](https://h2o.ai/resources/video/h2o-gen-ai-world-sf-kaggle-grandmaster-panel/)).

### Inferences
- Cascade design for this pipeline: stage 1 recall (formula and mass candidates), stage 2 LightGBM lambdarank, stage 3 a specialized isomer reranker trained only on same-formula groups (pairwise or listwise loss) with features that are differences within the group (similarity minus group max, rank within group, gap to second). Training stage 3 only on groups where the correct answer is present focuses capacity on the 96% miss category.
- Since MRR@25 rewards rank 1 heavily, the objective for the isomer reranker should be top-1 accuracy within group (softmax/listwise cross-entropy), not NDCG over all ranks.
- Pseudo-labeling: high-confidence test predictions (COSMIC-style: large gap, high calibrated score) could be added to fine-tune instrument-specific components. Risk: confirmation bias on isomers, which are the low-confidence cases anyway. Not sourced for this domain.
- Select the two final submissions as (a) best grouped-CV model and (b) a more conservative, diverse ensemble, rather than the best public LB variant.

### Gaps
- BELKA and Open Polymer Prediction 2025 winning write-ups were not researched in this pass.
- No published MS/MS-specific pseudo-labeling result found.

## Q6. Candidate recall: formula/candidate generation, ppm windows, adducts, negative mode

### Takeaway
Recall failures upstream (wrong formula, wrong adduct, too tight a ppm window, structure missing from DB) cap performance, and several CASMI wins/losses hinged on them; restricting to a single top formula is risky, and negative mode historically had less training data and weaker tools.

### Cited Findings
- CASMI 2016 candidates: ChemSpider at ±5 ppm via MetFrag (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)); EPA used ±10 ppm against MS-Ready DSSTox structures (Source: [PMC7345619](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345619/)).
- CASMI 2012: an overly strict 5 ppm window and calibration errors caused misses (Source: [PMC3901266](https://pmc.ncbi.nlm.nih.gov/articles/PMC3901266/)).
- CASMI 2017: CSI:FingerID restricted candidates to SIRIUS's top formula; missing isotope patterns produced formula errors that excluded correct structures (Source: [Jena post](https://bio.informatik.uni-jena.de/2017/11/casmi-2017-results-are-out)).
- MetFrag and Blaženović collapse stereoisomers by InChIKey first block; tautomers can have different first blocks and escape deduplication (Source: [PMC4732001](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732001/); [PMC5445034](https://pmc.ncbi.nlm.nih.gov/articles/PMC5445034/); [PMC7345619](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345619/)).
- Negative mode: Team Dührkop submitted no negative-mode challenges in 2016 for lack of training data; combining MetFrag and CFM doubled negative-mode Top-1 (Source: [PMC5368104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/)).
- CASMI 2022: adduct accuracy ranged 13% to 96% across teams, formula up to 95% (Source: [Fiehn lab CASMI 2022](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)).
- In MassSpecGym, retrieval candidate sets are either mass-based or formula-based, which changes difficulty (Source: [MassSpecGym arXiv 2410.23326](https://arxiv.org/html/2410.23326v3), via search snippet).

### Inferences
- Since 96% of misses are already within the correct formula, recall is not the main issue now; but keep a small fraction of probability mass on second-ranked formulas/adducts rather than hard filtering, and check that the minority of non-isomer misses are not adduct errors.
- Deduplicate by InChIKey first block before ranking so stereoisomers do not split credit or push the correct connectivity down; check how the competition scores stereochemistry before doing so.
- Negative-mode spectra deserve separate fusion weights and possibly a separate isomer reranker, given the historically different behavior of tools in negative mode.

### Gaps
- Competition-specific candidate-set definition (whether Kaggle provides candidates, and whether stereo matters in scoring) was not found.
