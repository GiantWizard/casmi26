---
status: active
area: portfolio
tags: [casmi26, denovo, research]
updated: 2026-10-11
---

# De novo structure generation from MS/MS for novel natural products (Enveda CASMI 2026 Kaggle), state as of 2026-10-11

Scope: generative and analog-based structure generation, realistic exact-match accuracy, evaluation caveats, code/weights/licenses, cost, formula inference, and safe merging into an MRR@25 retrieval ranking. Every number below names its evaluation protocol. "Known formula" means the ground-truth molecular formula was given to the model, which the Kaggle test set does NOT give.

## 1. Realistic exact-match accuracy of de novo methods (leaderboard after the "MassSpecGym in the Wild" corrections)

### Takeaway
On MassSpecGym (MCES-split, positive mode, known formula), the only audited numbers put honest top-1 exact match at about 2% (DiffMS) to 18% (FRIGID) and top-10 at 4% to 22%. Every claim above about 20% top-1 on MassSpecGym (MolForge 31%, MSFlow 32%, MS-GPT 23.9%, DualLGD 23.89%) is either confirmed inflated (MolForge) or unaudited and post-dates or ignores the audit, so treat them as unverified upper bounds.

### Cited Findings
**Audited / corrected de novo leaderboard, MassSpecGym de novo task "with bonus" (bonus = known formula), Table 9 of MassSpecGym in the Wild (arXiv 2606.19624), exact match on 14-char 2D InChIKey** — [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)
- MIST + MSNovelist: top-1 0.00%, top-10 0.00% (Tanimoto 0.06/0.15)
- Spec2Mol: 0.00% / 0.00%
- MIST + Neuraldecipher: 0.00% / 0.00%
- MADGEN: 1.31% / 1.54%
- MS-BART: 1.07% / 1.11%
- DiffMS: 2.30% / 4.25% (MCES 13.96/11.68)
- FOAM (DiffMS + PubChem): 1.50% / 10.28%
- MIST + MolForge (corrected, single-spectrum inference): 10.73% / 14.48%
- MIST + MolForge (inflated, batch size 24, marked "not valid"): 31.75% / 40.55%
- FRIGID: 18.29% / 22.00% (Tanimoto 0.43/0.47)

**The "MolForge bug"** — [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)
- MIST attention code applied the raw boolean padding mask as `attn += attn_mask`, giving padding tokens a finite +1 attention offset instead of excluding them; this affects batched training and inference.
- The "vDiffMS" MIST checkpoint (used in DiffMS and reused in MIST+MolForge) produced batched-mode fingerprint predictions "consistent with patterns of data leakage or contamination"; authors could not reproduce it from the documented recipe. Under batch size 24, MIST fingerprint Tanimoto was 0.812 vs 0.457 corrected.
- Original MolForge claim (Neo et al., NeurIPS 2025, arXiv 2508.04180): top-1 31% / top-10 40% on MassSpecGym, "tenfold improvement" — [NeurIPS 2025 page](https://neurips.cc/virtual/2025/128931); [arXiv 2508.04180](https://arxiv.org/html/2508.04180v3)
- The audit found evaluation issues in at least 17 of 26 papers reporting MassSpecGym results in its first year; three failure classes: data leakage, shortcut learning, implementation bugs / metric divergence — [arXiv abs 2606.19624](https://arxiv.org/abs/2606.19624)

**Decoder pretraining analog leakage (MolForge decoder, MassSpecGym, Acc@1/Acc@10, Table 6)** — [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)
- 2.5M pretraining set: include_all 11.6/14.8% -> exclude_exact 11.0/14.1% -> exclude Tanimoto>=0.7 8.0/11.1% -> exclude >=0.6 7.8/11.5%.
- 10M: include_all 14.1/16.7% -> exclude_tani070 10.6/12.1%.
- 50M: include_all 12.7/14.4% -> exclude_tani070 11.8/12.9% (larger pretraining is less sensitive to analog exclusion).
- Example given: ground truth reproduced "perfectly" from a fingerprint with Tanimoto 0.61 by slightly modifying a pretraining molecule at Tanimoto 0.98 to the target.
- Recommendations: exclude exact 2D InChIKey matches and, for generalization claims, Tanimoto>=0.7 / MCES<10 analogs from pretraining; stratify results by similarity to pretraining set; pin metrics (InChIKey-14, 2048-bit ECFP4, MCES threshold 15); 99.9% bootstrap CIs. MassSpecGym v1.5 ships corrected MIST and 2.5M/50M decoder pretraining sets excluding Tanimoto>=0.7 analogs.
- InChIKey-14 vs full InChIKey for DiffMS de novo gave identical values: top-1 2.46%, top-10 4.43% (99.9% CI 3.94-4.95%) (Table 10).

**Per-method claims (not audited unless stated)**
- FRIGID (Coley group, ICML 2026, PMLR v306): known formula; MassSpecGym top-1 18.29% / top-10 22.00% (FRIGID-base 16.09/18.19); NPLIB1 25.03 / 33.37 (base 19.80/24.41); strongest NPLIB1 baseline listed DiffMS 8.34/15.44 — [GitHub coleygroup/FRIGID](https://github.com/coleygroup/FRIGID); [PMLR](https://proceedings.mlr.press/v306/bohde26a.html). ICML poster abstract states lower figures ("surpassing 15%" top-1, "more than doubling" on NPLIB1), PMLR says 18% and "triples" — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/60806). FRIGID's 18.29/22.00 is the number reproduced in the audited Table 9.
- MS-GPT (arXiv 2607.23607, 26 Jul 2026): known formula, MIST encoder; MassSpecGym 23.9 / 28.7, NPLIB1 29.8 / 41.1; 100 samples per spectrum (20 queries x 5 samples), candidates ranked by generation frequency — [arXiv](https://arxiv.org/abs/2607.23607); [GitHub VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT). Not in the audit (posted after it).
- DualLGD (arXiv 2605.07048, May 2026): MassSpecGym top-1 23.89%, NPLIB1 34.37%; claims ~3x prior SOTA; formula status and code not stated on abstract page — [arXiv](https://arxiv.org/abs/2605.07048)
- MSFlow (arXiv 2602.19912): MassSpecGym 32.00 / 42.53, NPLIB1 44.70 / 58.53; MIST-architecture encoder using "a predicted molecular formula assignment produced by SIRIUS"; decodes CDDD embeddings with discrete flow matching; code "openly available for non-commercial usage" at github.com/ghaith-mq/MSFlow — [arXiv HTML](https://arxiv.org/html/2602.19912v1)
- FlowMS (ICLR 2026, distinct from MSFlow): NPLIB1 top-1 9.15% (9.7% relative over DiffMS) — [ICLR 2026](https://iclr.cc/virtual/2026/10014532)
- MBGen ("Many-Body Enhanced Diffusion", AAAI 2026, arXiv 2602.01643): known formula (node set fixed to SIRIUS-obtainable formula), MIST encoder following DiffMS, 100 candidates; MassSpecGym 7.58 / 12.54, NPLIB1 12.20 / 22.29; code github biomed-AI/MBGen — [arXiv HTML](https://arxiv.org/html/2602.01643v1); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37074)
- MSAnchor (AAAI 2026): anchor-extended scaffold representation + conditional information bottleneck; benchmark numbers not retrieved — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37064)
- Test-time tuned LM (arXiv 2510.23746, Mismetti, Alberts, Krause, Graziani): end-to-end transformer, known formula, pretrained on simulated spectra, test-time adaptation; top-1 3.16% MassSpecGym, 12.88% NPLIB1 — [arXiv](https://arxiv.org/abs/2510.23746)
- MARLIN (arXiv 2607.04774, Jul 2026, same authors as DualLGD): explicitly NO ground-truth formula; block-diffusion LM conditioned on predicted fingerprint + precursor mass with a mass-shell ppm constraint; claims strongest formula-free method on NPLIB1 and recovers the formula about as often as a dedicated formula predictor; no numbers or code on abstract page — [arXiv](https://arxiv.org/abs/2607.04774)
- MSNovelist (Nature Methods 2022): on 3,863 GNPS spectra, 25% correct at rank 1, 45% retrieved overall (CSI:FingerID fingerprints, different protocol) — [PMC9262714](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9262714/); on MassSpecGym with MIST it scores 0.00% (Table 9 above).
- MS2Mol (Enveda, ChemRxiv 2023): EnvedaDark benchmark of 226 natural products absent from major databases; 21% within "close-match" threshold, 62% within "meaningful similarity" (similarity thresholds, NOT exact match); a confidence scorer raises this to 63% close-match / 98% meaningful similarity on the top 10% most confident predictions — [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6492507524989702c2b082fc)

### Inferences
- For the Kaggle setting (unknown formula, timsTOF, natural products, novel structures), a realistic expectation for the best available released model (FRIGID) is single-digit to low-teens % top-10 exact match on truly novel NPs, before instrument-shift losses. The claimed >20% MassSpecGym numbers share the MIST encoder lineage; whether they used the corrected MIST or the vDiffMS checkpoint is unverified, so they should not be used for planning.
- NPLIB1 numbers run systematically higher than MassSpecGym (NPLIB1 is not MCES-split, per general knowledge; unverified in this session), so NPLIB1 figures overstate novel-structure performance.
- The analog-leakage table shows exact-match accuracy drops ~30% relative (11.6 -> 8.0 top-1, 2.5M decoder) once Tanimoto>=0.7 analogs are removed. Novel NPs that do have close PubChem analogs are therefore the most recoverable subset; truly scaffold-novel molecules are likely near 0.
- Against the gating threshold (true structure in top ~5 for >=10% of novel molecules): FRIGID formula-unknown top-10 on MassSpecGym is 17.43% (see section 3), so top-5 on in-distribution spectra plausibly clears 10%, but domain shift to timsTOF and to novel NPs likely pushes it below. This is borderline, not a clear win.

### Gaps
- DualLGD formula setting, code, and whether it used the corrected MIST encoder: not found.
- MSAnchor numbers: not retrieved.
- MS-BART and MADGEN repo licenses: not checked; only audited accuracies (~1%) recorded, which make them not worth integrating.
- Spec2Mol details beyond 0.00% on MassSpecGym: not researched.
- MS2Mol code/weights: no public release found in this session (Enveda is the competition host; unverified whether anything was released).
- No paper reports de novo accuracy on Bruker timsTOF spectra or on negative-mode spectra; MassSpecGym adduct coverage not verified in this session.

## 2. FRIGID specifics: formula requirement, speed, licenses, retraining, commercial path

### Takeaway
FRIGID code is Apache 2.0 but the only pretrained checkpoints (Zenodo 19685145, 2.8 GB) are CC BY-NC 4.0, so a commercial or prize-eligible path would require retraining FRIGID-base, MIST, and ICEBERG from scratch; it needs a formula but degrades only moderately with MIST-CF top-5 formulas (MassSpecGym top-1 18.29 -> 14.28%).

### Cited Findings
- Architecture: diffusion LM generating SAFE strings conditioned on MIST-predicted Morgan fingerprints and the chemical formula; optional ICEBERG-guided refinement that simulates candidate spectra, flags inconsistent substructures, remasks and re-denoises — [GitHub](https://github.com/coleygroup/FRIGID)
- Paper: "In line with peer methods, we focus on the setting where the ground-truth chemical formula is provided" — [arXiv HTML 2604.16648](https://arxiv.org/html/2604.16648)
- Robustness to formula uncertainty (Appendix A.4.1, Table 8): MIST-CF top-5 formula hypotheses, fixed generation budget split across 5 formulas. MassSpecGym FRIGID known 18.29/22.00 -> unknown 14.28/17.43; FRIGID-base 16.09/18.19 -> 11.60/13.13. NPLIB1 FRIGID 25.03/33.37 -> 21.79/30.14; base 19.80/24.41 -> 18.68/22.91 — [arXiv HTML 2604.16648](https://arxiv.org/html/2604.16648)
- Speed: FRIGID backbone 6.58 s per spectrum, 247.0M parameters; with inference-time refinement 86.49 s per sample; DiffMS 131.2 s; DiffMS + ICEBERG refinement 1423.30 s/sample; all on a single NVIDIA RTX 6000 Ada — [arXiv HTML 2604.16648](https://arxiv.org/html/2604.16648); [GitHub](https://github.com/coleygroup/FRIGID)
- Refinement defaults: per-round sample budget B=128, M=8 renoised variants per candidate, K=12 candidates refined per round, sweep up to R=25 rounds (repo default 10 rounds) — [arXiv HTML 2604.16648](https://arxiv.org/html/2604.16648); [GitHub](https://github.com/coleygroup/FRIGID)
- Inference-time scaling is log-linear in compute — [PMLR](https://proceedings.mlr.press/v306/bohde26a.html)
- Licenses: README states Apache 2.0 for code (LICENSE/license_code.txt plus third-party.txt and 3rd_party/) — [GitHub](https://github.com/coleygroup/FRIGID); [LICENSE folder](https://github.com/coleygroup/FRIGID/tree/main/LICENSE). Zenodo record 19685145: single file frigid_pretrained_checkpoints.tar.gz, 2.8 GB, CC BY-NC 4.0 — [Zenodo](https://zenodo.org/records/19685145)
- Training data: FRIGID-base pretrained on SAFE dataset (paper: up to 1B structures from Noutahi et al. 2023; abstract says "hundreds of millions"; README says up to 230M), with NPLIB1 and MassSpecGym test/val InChIKeys removed; ICEBERG trained only on each benchmark's training split; NIST20 not used because it is not publicly available — [arXiv HTML 2604.16648](https://arxiv.org/html/2604.16648); [GitHub](https://github.com/coleygroup/FRIGID)
- Retraining: FRIGID-base via scripts/train.py with configs/fp2mol_pretraining.yaml, defaults 500,000 steps, global batch 2048, EMA 0.9999; NGBoost length models in token_models/; no MIST or ICEBERG retraining commands in README — [GitHub](https://github.com/coleygroup/FRIGID)
- Fingerprint threshold 0.187 calibrated on validation data — [arXiv HTML 2604.16648](https://arxiv.org/html/2604.16648)
- FRIGID as generative retrieval on MassSpecGym: 45.27% Hit@1 (Table 3 of audit, retrieval task) — [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)

### Inferences
- FRIGID pretrained exclusion only removed benchmark test InChIKeys, so the Kaggle test molecules (unknown to the authors) were not excluded; if a "novel" Kaggle molecule happens to be in the SAFE pretraining corpus, FRIGID may recover it via memorization (this helps the competition, but means held-out benchmark numbers understate it for in-corpus molecules and overstate it for truly out-of-corpus ones).
- T4 cost (inference, not measured anywhere): T4 has roughly 1/5 to 1/8 the throughput of an RTX 6000 Ada, so backbone-only sampling would be on the order of 30-50 s per spectrum and full ICEBERG refinement on the order of 7-12 min per spectrum. Backbone-only on the gated low-confidence subset is the only plausible budget; full refinement is not.
- Commercially licensable path: retrain FRIGID-base on a permissively licensed corpus (e.g. the SAFE dataset, license unverified), retrain MIST on MassSpecGym (license unverified here) with the corrected attention mask, skip ICEBERG. Cost of 500k steps at batch 2048 on a 247M model is multi-GPU-days; not cheap. Whether CC BY-NC weights are permitted under the Kaggle competition rules is a rules question for the applicant to check, not answerable by web search.

### Gaps
- Exact FRIGID per-spectrum time on T4 or with fp16: not reported.
- MIST encoder training data for FRIGID's released checkpoint and whether it uses the corrected attention mask: not stated in visible text.
- License of the SAFE pretraining dataset and of MassSpecGym spectra for commercial use: not checked.
- Kaggle "Enveda CASMI 2026" rules on non-commercial pretrained weights: not checked (requires reading the competition rules page while logged in).

## 3. Formula inference (SIRIUS/ZODIAC, MIST-CF, BUDDY/msbuddy) and what predicted formulas do to generators

### Takeaway
Formula top-1 accuracy from MS/MS alone is poor (0.19-0.40 top-1 for SIRIUS / pretrained MIST-CF on NIST and MoNA positive mode) while top-5 is 0.44-0.76, so generators must sample across the top-5 formulas; this costs FRIGID about 4 points top-1 on MassSpecGym. msbuddy is Apache 2.0 and handles both polarities, but no head-to-head with MIST-CF or negative-mode benchmarks was found.

### Cited Findings
- Che, Du, Xu (arXiv 2601.00941, Jan 2026), positive mode only: NIST SIRIUS top-1/5/10 0.284/0.758/0.825; MIST-CF pretrained 0.296/0.685/0.727; MIST-CF retrained 0.378/0.503/0.622. MoNA SIRIUS 0.193/0.435/0.500; MIST-CF pretrained 0.402/0.655/0.744; retrained 0.744/0.892/0.915 — [arXiv HTML 2601.00941](https://arxiv.org/html/2601.00941v1)
- Same paper, top-5 by adduct: [M+H]+ 0.73-0.99; [M+Na]+ 0.52-0.95; [M+K]+ 0.16-0.99 (very unstable); [M-H+2Na]+ 0.00-0.18; negative mode not evaluated — [arXiv HTML 2601.00941](https://arxiv.org/html/2601.00941v1)
- Same paper, DiffMS top-5 structure accuracy: pretrained oracle 0.077 (NIST) vs MIST-CF formulas 0.052; retrained 0.664 vs 0.662 — [arXiv HTML 2601.00941](https://arxiv.org/html/2601.00941v1). The retrained 0.66 figure is implausibly high relative to the audited MassSpecGym leaderboard and likely reflects a non-structure-disjoint split (inference, split not verified).
- MassSpecGym in the Wild, retrieval Hit@1 under formula settings: random 0.43%, random with ground-truth formula 12.98%, MIST with MIST-CF top-1 formula 26.64%, MIST with ground-truth formula 51.45%; some published MIST-CF results inflated by test-spectra leakage; ground-truth-formula results belong in the formula-based track — [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)
- FRIGID with MIST-CF top-5: see section 2 (MassSpecGym 18.29 -> 14.28 top-1).
- msbuddy (BUDDY, Nature Methods 20:881-890, 2023): Apache 2.0; bottom-up MS/MS interrogation shrinks candidate space, ML annotation with FDR estimation; accepts either polarity, minimum input one m/z plus polarity, optional MS1 isotope pattern — [msbuddy docs](https://msbuddy.readthedocs.io/en/stable/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.08.03.502704.full.pdf)
- SIRIUS: Jena group page states SIRIUS has been GPL since v3.4, other licenses on request — [bio.informatik.uni-jena.de](https://bio.informatik.uni-jena.de/?p=6547). (This is an old page; current SIRIUS 6 terms and whether CSI:FingerID/COSMIC require online login were not confirmed.)
- MARLIN claims formula recovery "about as often as a dedicated formula predictor" as a byproduct of mass-shell-constrained generation — [arXiv 2607.04774](https://arxiv.org/abs/2607.04774)

### Inferences
- In the Kaggle setting, the candidate-formula set from retrieval (formulas of PubChem candidates within ppm of precursor for each adduct hypothesis) is a strong prior. Generating only under formulas already supported by retrieval candidates or by msbuddy top-k is the cheapest guard; using top-5 formulas mirrors FRIGID's A.4.1 setup.
- Negative-mode and exotic adducts (e.g. [M+K]+, [M-H+2Na]+) are where formula inference collapses, so generated candidates for those spectra should be gated off or down-weighted.

### Gaps
- No natural-product-specific formula accuracy for SIRIUS/ZODIAC/msbuddy found in this session; ZODIAC accuracy not retrieved.
- MIST-CF license and checkpoint provenance: not checked.
- No negative-mode formula benchmark found.

## 4. Non-neural and analog-based structure generation

### Takeaway
Analog-based methods (ModiFinder site localization, BioTransformer rules, molecular networking, COSMIC confidence) are well established for annotation and localization, but no source found reports a measured exact-structure recovery rate for novel natural products from analog modification; treat any such pipeline as unbenchmarked.

### Cited Findings
- ModiFinder (JASMS 2024): aligns MS/MS of a known molecule and an unknown analog, uses shifted fragment peaks to score each atom of the known structure as the modification site; uses in silico fragmentation, formula refinement, helper spectra; reports outperforming baselines at site localization, not full-structure exact match — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.17.580849.full.pdf); [code](https://github.com/ahmad00m/ModiFinder_base)
- BioTransformer: 2019 paper reports precision/recall up to 7x better than Meteor Nexus / ADMET Predictor on drug, pesticide, phytochemical, endobiotic sets — [PMC6689873](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6689873/); BioTransformer 3.0 claims 40-50% more accurate and fewer combinatorial explosions — [PMC9252798](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9252798/). License: README says LGPL v3, with the EAWAG environmental module under CC BY-NC-SA 4.0 and commercial use requiring author permission; a Galaxy wrapper says GPL v3 (conflict) — [GitLink mirror](https://www.gitlink.org.cn/NSCCN/biotransformer); [Galaxy toolshed](https://toolshed.g2.bx.psu.edu/repos/recetox/biotransformer/annotate/200c5c982dba/macros.xml)
- COSMIC (Hoffmann et al., Nature Biotechnology 40:411-421, 2022): confidence score combining kernel-density P-values and an SVM with enforced feature directionality, output 0-1 probability that a CSI:FingerID annotation is correct; authors found raw in silico hit scores cannot separate correct from incorrect annotations; 1,715 high-confidence annotations absent from spectral libraries across 17,400 experiments; 315 manually validated structures absent from HMDB; in SIRIUS since 4.8 — [PMC8926923](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8926923/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.03.18.435634.full.pdf)
- FOAM (DiffMS + PubChem) on MassSpecGym: top-1 1.50%, top-10 10.28%, best top-10 MCES (6.06) and Tanimoto (0.53) in the audited table, showing that combining generation with database structures improves closeness more than exact match — [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)
- A 2026 evaluation notes de novo performance "drastically drops" when oracle scaffolds are replaced by predicted ones (cited via search snippet; primary paper not opened) — [MSU Istina record](https://istina.msu.ru/publications/article/815677344/) (unverified)

### Inferences
- Scaffold-plus-modification enumeration constrained by the inferred formula (take retrieval top hits, apply BioTransformer / common NP mass-difference rules such as hydroxylation, methylation, glycosylation, enumerate formula-matching products, rescore with a forward simulator or the retrieval model) is cheap and CPU-friendly but has no published exact-recovery rate; it needs an internal validation on held-out library spectra before use.
- Site ambiguity is the main failure: ModiFinder-style localization yields a distribution over atoms, so enumerated isomers split probability mass, which hurts top-1 but can still put the true isomer in top-5.

### Gaps
- No measured exact-match recovery for MolNetEnhancer, ZODIAC-based networks, or mass-difference networks for novel NPs found.
- ModiFinder dissertation's full-structure generation results: not accessible.

## 5. Merging generated candidates into a retrieval ranking under MRR@25 without hurting known molecules

### Takeaway
No surveyed system publishes a calibrated "generated vs retrieved" interleaving scheme; the closest evidence is COSMIC-style confidence on retrieval hits, MS2Mol's confidence scorer (63% close-match on the top 10% most confident), and generation-frequency ranking in MS-GPT. Gating on low retrieval confidence plus per-candidate confidence (sample frequency, formula agreement, forward-simulated spectral match) is the defensible design.

### Cited Findings
- COSMIC demonstrates that a dedicated confidence model, not the raw hit score, is needed to separate correct from incorrect database annotations — [PMC8926923](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8926923/)
- MS2Mol confidence scorer enriches close-match accuracy from 21% overall to 63% for the top 10% most confident predictions — [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6492507524989702c2b082fc)
- MS-GPT ranks pooled candidates by generation frequency across a band of fingerprint queries; reports candidate-pool scaling improves recall with little extra cost — [arXiv 2607.23607](https://arxiv.org/abs/2607.23607)
- FRIGID uses ICEBERG forward-simulated spectra to flag inconsistent substructures; FRIGID-as-retrieval reaches 45.27% Hit@1 on MassSpecGym retrieval, i.e. generated candidates can be scored in the same space as library candidates — [GitHub](https://github.com/coleygroup/FRIGID); [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)
- Retrieval with ground-truth formula vs MIST-CF top-1 formula: 51.45% vs 26.64% Hit@1, so formula confidence is itself a strong gating signal — [arXiv 2606.19624 HTML](https://arxiv.org/html/2606.19624)

### Inferences
- Concrete design consistent with the evidence: (a) gate on retrieval confidence (top-1 score margin, COSMIC-like calibrated probability trained on held-out library spectra); (b) for gated spectra, insert only generated candidates that are not already in the retrieval list (InChIKey-14 dedup), whose formula matches a top-k formula, and that appear repeatedly across samples (frequency >= some threshold); (c) rescore inserted candidates with the same retrieval model (embed generated SMILES as candidates) so they compete on the same scale, and only interleave at fixed slots when their rescored score exceeds a validated threshold.
- Because the simulation says +0.01 MRR needs true-in-top-5 for >=10% of novel molecules, and the only audited formula-unknown number (FRIGID MassSpecGym top-10 17.43%) is on in-distribution positive-mode data, the expected gain on Kaggle is likely below the break-even unless validated on a held-out set of timsTOF NP spectra whose structures are removed from the candidate pool (simulated novelty).
- The ~45% FRIGID generative-retrieval Hit@1 suggests an alternative use: generation as a reranking signal for retrieval candidates (similarity of each PubChem candidate to the generated set), which risks no slot loss on known molecules. This is an inference, not a measured Kaggle result.

### Gaps
- No published calibration study of mixing de novo candidates into retrieval lists under MRR was found.
- No Kaggle-specific discussion threads or winning solutions were researched (out of scope / not yet public as of this date, unverified).
