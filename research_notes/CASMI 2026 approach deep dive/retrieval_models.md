---
status: active
area: portfolio
tags: [casmi26, retrieval, ms-ms, isomers, research]
updated: 2026-10-11
---

# Spectrum-to-structure retrieval models for isomer discrimination (CASMI 2026 / Enveda Kaggle)

Research date: 2026-10-11. Scope: models that score (spectrum, candidate structure) pairs and could be added as same-formula re-ranking features on top of the existing FPNet + LightGBM pipeline (pipeline baseline: 88% top-25 recall on held-out natural products, MRR 0.525, 96% of misses are same-formula isomers ranked above the truth).

Conventions: "UNVERIFIED" means I did not read a primary source confirming it. The host-approved model list (MassSpecGym-trained permissive models, DreaMS, CFM-ID 4, ICEBERG/GLACIER MassSpecGym checkpoints) comes from the task brief. I did not re-check it against the Kaggle rules page.

## Q1. Benchmark baseline and the corrected 2026 picture (MassSpecGym, audit, what numbers to trust)

### Takeaway
On MassSpecGym's formula-based ("bonus") retrieval track, which is the closest public analogue to "rank same-formula isomers", the original baselines are weak (MIST 9.57% Hit@1). The best credible 2026 numbers come from forward-simulation scorers (GLACIER 49.9%, ICEBERG 2.0 44.4%) and MIST trained with leakage-free ICEBERG augmentation (53.76%). A June 2026 audit shows that several headline contrastive results (80%+ Hit@1) are artifacts of SMILES canonicalization and should be ignored.

### Cited Findings
- MassSpecGym: about 231k spectra, 29k molecules, from MoNA, MassBank, GNPS plus in-house MSnLib (about 33% of molecules). NIST is excluded because of its licensing. Adducts are only [M+H]+ and [M+Na]+. Instruments are Orbitrap and QTOF (98%), with no timsTOF called out. The MCES split clusters molecules so that no cross-fold pair has edge-edit distance below 10. Source: [MassSpecGym paper](https://arxiv.org/html/2410.23326)
- MassSpecGym code is MIT licensed. The dataset is on HuggingFace (roman-bushuiev/MassSpecGym). The README adds MassSpecGym-compatible ICEBERG weights (Dropbox, codebase coleygroup/ms-pred), noted as a 10/27/2025 update. Source: [MassSpecGym GitHub](https://github.com/pluskal-lab/MassSpecGym)
- Original retrieval table, given as HR@1 / HR@5 / HR@20 (%). Source: [MassSpecGym paper, Table 5](https://arxiv.org/html/2410.23326)
  - Mass candidates: Random 0.37/2.01/8.22; DeepSets 1.47/6.21/19.23; Fingerprint FFN 2.54/7.59/20.00; DeepSets+Fourier 5.24/12.58/28.21; MIST 14.64/34.87/59.15.
  - Formula (bonus) candidates: Random 3.06/11.35/27.74; DeepSets 4.42/14.46/30.76; Fingerprint FFN 5.09/14.69/31.97; DeepSets+Fourier 6.56/16.46/33.46; MIST 9.57/22.11/41.12.
- "MassSpecGym in the Wild" (arXiv 2606.19624, 17 June 2026; authors include Bushuiev, Coley, Healey, Pluskal). Source: [arXiv 2606.19624](https://arxiv.org/html/2606.19624v1)
  - Ground-truth SMILES kept repository formatting while decoys were RDKit-canonicalized, so models could exploit the formatting.
  - Corrected Hit@1: DreaMS+ChemBERTa 82.41% to 6.70%; ChemFormer two-stage 63.89% to 7.60%; a ChemBERTa SMILES-only classifier 99.66% to 8.2%.
  - An ICEBERG model trained on test data inflated MIST R@1 to 67.75%, against 53.76% leakage-free and 9.57% with no augmentation.
  - A test-contaminated MIST-CF formula annotator raised MIST Hit@1 from 26.64% to 35.15%.
  - A spectrum-blind PubChem default ranking reached 49.98% Hit@1.
  - A padding-mask bug in batched MIST also affected results.
  - The authors released MassSpecGym v1.5 with fixed implementations and recommend comparing on the 14-character InChIKey first block.
- Literature Hit@1 on the formula-known (bonus) setting, as tabulated by the audit: Random 3.06; Fingerprint FFN 5.09; MIST (MassSpecGym version) 9.57; JESTR 11.82; MVP 13.96; FLARE 22.66; FRIGID (generative retrieval) 45.27; MIST (DiffMS version) 46.42; MIST (FRIGID version) 53.76. The audit says the DiffMS MIST checkpoint showed patterns consistent with contamination, but it could not reproduce the inflation. Source: [arXiv 2606.19624](https://arxiv.org/html/2606.19624v1)

### Inferences
- The formula-bonus track is the right yardstick for this pipeline's failure mode, since 96% of misses are same-formula isomers. On that track, model-based ranking moved from about 10% to about 50% Hit@1 between 2024 and 2026. Most of the gain came from (a) forward-simulation scoring and (b) MIST trained on synthetic ICEBERG spectra.
- Any candidate model's claimed gains need to be checked on our own held-out natural products with InChIKey first-block evaluation. Candidate SMILES must be canonicalized identically for truth and decoys, so our features can't learn the same shortcut.
- A popularity or PubChem-prior feature is strong on its own (49.98% Hit@1 spectrum-blind). Make sure the LightGBM already has something like it. Note that it may transfer poorly to obscure COCONUT natural products.

### Gaps
- I did not get v1.5-corrected Hit@k for MIST, JESTR, MVP or FLARE. The audit table reproduces literature values for these. The rest of the paper (beyond character 100k) was not read.
- I did not check the live leaderboard (massspecgym.onrender.com).
- I found no MassSpecGym results restricted to natural products, and no MCES-split breakdown by compound class.

## Q2. Forward-simulation scorers: ICEBERG 2.0, GLACIER, CFM-ID 4 (host-approved per brief)

### Takeaway
GLACIER (Coley lab, v2 posted 5 Oct 2026) is the best-documented same-formula discriminator: 49.88% Hit@1 on the MassSpecGym formula challenge, about 1.3 ms per simulated spectrum. Simulating spectra for every same-formula candidate and adding a spectrum-similarity feature is the most promising single addition for isomer discrimination. The main risk is adduct coverage: the MassSpecGym checkpoints only cover [M+H]+ and [M+Na]+.

### Cited Findings
- GLACIER (Wang, Wang, Coley; arXiv 2606.29161v2, 5 Oct 2026) frames MS/MS prediction as object detection over fragments, using a Graphormer backbone and DETR-style queries, with optional contrastive fine-tuning ("CF"). Source: [arXiv 2606.29161](https://arxiv.org/html/2606.29161)
- MassSpecGym retrieval, given as R@1 / R@5 / R@20 (%). Source: [arXiv 2606.29161](https://arxiv.org/html/2606.29161)
  - Formula challenge: GLACIER w/ CF 49.88/78.13/90.75; GLACIER w/o CF 48.49/76.56/90.60; ICEBERG 2.0 44.35/73.57/89.45; MARASON 34.03/64.04/85.39; FraGNNet 31.93/63.20/82.70; NEIMS (FFN) 7.62/22.70/44.12.
  - Mass challenge: GLACIER w/ CF 69.95/86.47/93.52; ICEBERG 2.0 63.95/83.83/92.56; MolSpecFlow 55.32; FraGNNet 46.64.
  - The paper samples 256 candidates per query.
- GLACIER is also trained on NIST'20 (530,640 spectra). Those checkpoints would be NIST-derived and so are banned under the competition rules. Adduct type is a one-hot input. Code is at github.com/coleygroup/ms-pred. The paper does not state a weights release or code license, and the paper itself is CC BY-NC-ND 4.0. Source: [arXiv 2606.29161](https://arxiv.org/html/2606.29161)
- Inference: GLACIER takes about 1.3 ms per spectrum (about 369 spectra/s per A5000) against about 10.8 ms for ICEBERG 2.0, roughly 8x faster. Source: [arXiv 2606.29161](https://arxiv.org/html/2606.29161)
- ICEBERG MassSpecGym-compatible weights are hosted via the MassSpecGym README (Dropbox link, 10/27/2025 update). Source: [MassSpecGym GitHub](https://github.com/pluskal-lab/MassSpecGym)

### Inferences
- **Integration idea (GLACIER / ICEBERG):**
  - For each query, take the top-K same-formula candidates from the current ranker (K of 50 to 200). Simulate each candidate's spectrum for the query adduct where it is supported.
  - Compute entropy similarity and cosine between the simulated and observed spectra, after converting the observed spectrum to the model's expected format.
  - Add three LightGBM features: the raw similarity, the similarity minus the group max, and the rank within the same-formula group.
  - Cost: 400 queries x 200 candidates = 80k simulations, which is about 2 minutes on one GPU for GLACIER, so it fits in a Kaggle notebook if the weights can be loaded offline.
- **Adduct mismatch:** MassSpecGym only has [M+H]+ and [M+Na]+, so the MassSpecGym checkpoints probably can't simulate [M+NH4]+, [M-H2O+H]+, [M+K]+ or any negative-mode adduct.
  - Option 1: simulate as [M+H]+ and compare fragment-level (neutral-loss-shifted) spectra.
  - Option 2: set the feature to missing for unsupported adducts and let LightGBM handle the NaN.
  - Either way, validate per adduct.
- **timsTOF:** CASMI test spectra are Bruker timsTOF, while MassSpecGym is about 98% Orbitrap/QTOF. Expect some domain shift in intensities. Similarity computed on matched peaks may still rank isomers usefully.

### Gaps
- I did not verify whether GLACIER MassSpecGym weights are actually released, or what the ms-pred repo license is. The repo was not fetched.
- CFM-ID 4 was not researched (license, weights, negative-mode support), because of the tool-call budget. It is reported to support [M-H]- in earlier versions (UNVERIFIED).
- Whether ICEBERG 2.0 or GLACIER perform better than ICEBERG 1 on natural products specifically was not found.

## Q3. MIST and MIST-CF (Goldman et al.)

### Takeaway
MIST is MIT-licensed. Released weights exist for two datasets: csi2022, which includes NIST and so is banned, and NPLIB1 (canopus_train), which is GNPS-derived and permitted. The released models handle only [M+H]+. MIST trained on MassSpecGym with leakage-free ICEBERG augmentation reaches about 54% formula-bonus Hit@1, but that is an evaluation configuration. I did not confirm a released checkpoint for it.

### Cited Findings
- MIST repo: MIT license. Training sets are "csi2022" (H+ spectra from GNPS, NIST, MoNA and others, not redistributable) and "canopus_train", renamed NPLIB1 (public GNPS data from the 2021 CANOPUS paper). Training uses ICEBERG-simulated augmentation. Source: [MIST GitHub](https://github.com/samgoldman97/mist)
- Pretrained weights come from quickstart/00_download_models.sh. Models are [M+H]+ only. The changelog says other adducts are encodable and training them would be "straightforward". The repo includes its own subformula labeler to reduce reliance on SIRIUS. Source: [MIST GitHub](https://github.com/samgoldman97/mist)
- MIST-CF (precursor formula prediction) is at github.com/samgoldman97/mist-cf. Source: [MIST GitHub](https://github.com/samgoldman97/mist)
- MIST scores 9.57% formula-bonus Hit@1 in the original benchmark and 53.76% with leakage-free ICEBERG augmentation in the FRIGID version. A test-contaminated MIST-CF can inflate Hit@1 by up to about 9 points. Source: [arXiv 2606.19624](https://arxiv.org/html/2606.19624v1)

### Inferences
- MIST is the same family as FPNet: both predict fingerprints. Its value as an extra feature is that its subformula-annotated peak encoder captures different peak-formula information, so a disagreement-based ensemble may help. The FRIGID-version MIST (53.76%) would be the one worth obtaining, but its training data and weights availability need checking.
- **Integration idea:** compute the Tanimoto (or the MIST-style cosine on predicted fingerprint probabilities) between MIST's predicted fingerprint and each candidate's fingerprint. Add the raw value plus the within-formula-group delta and rank as features.
  - Use only NPLIB1-trained or MassSpecGym-trained weights.
  - Retraining on MassSpecGym plus competition training data with an adduct embedding would cover the 10 adducts.
- Use MIST-CF only for formula proposal where the formula is unknown, and never with checkpoints exposed to test data.

### Gaps
- I did not verify the MIST-CF license file or its training data (it may include NIST-containing data).
- I did not confirm the exact download URLs or licenses for the NPLIB1 weights. The weights are fetched by a script, so check the license on that host.
- I found no NPLIB1 Hit@1 numbers for MIST in this session.

## Q4. DreaMS and DreaMS-based contrastive models (SpecBridge, GLMR)

### Takeaway
DreaMS (Nature Biotechnology, 23 May 2025) has MIT-licensed code and CC BY 4.0 weights on Zenodo, and it is self-supervised on GNPS (GeMS), so no NIST is involved. It is host-approved per the brief. It is a spectrum embedding, not a structure scorer, so it needs a spectrum-to-molecule alignment head trained on our own data. SpecBridge (DreaMS + ChemBERTa) and GLMR claim 68 to 85% Hit@1, but those claims are likely inflated by the canonicalization shortcut and should not be trusted.

### Cited Findings
- DreaMS: transformer pretrained self-supervised on GeMS (mined from MassIVE GNPS). Code is MIT. Weights are on Zenodo record 10997887. Fine-tuning targets mentioned are spectral similarity, property prediction and fluorine detection. The README does not mention adduct or negative-mode support. Source: [DreaMS GitHub](https://github.com/pluskal-lab/DreaMS)
- Paper DOI is 10.1038/s41587-025-02663-3 (Nature Biotechnology, 23 May 2025). The Zenodo weights record ("Weights of pre-trained DreaMS models", v1, 19 April 2024) is CC BY 4.0, about 2.6 GB (embedding_model.ckpt 1.2 GB, ssl_model.ckpt 1.4 GB). The record cites the ChemRxiv preprint DOI, so it may be the preprint-era release. Sources: [Zenodo 10997887](https://zenodo.org/records/10997887), [DreaMS docs](https://dreams-docs.readthedocs.io/en/latest/)
- SpecBridge (Wang, Chen, Chen, Liu, Hassoun; arXiv 2601.17204v2, 27 Jan 2026). Source: [arXiv 2601.17204](https://arxiv.org/html/2601.17204v2)
  - Architecture: DreaMS encoder with the last 2 blocks fine-tuned, frozen ChemBERTa-2, and a residual mapper trained with MSE alignment.
  - MassSpecGym R@1 / R@5 / R@20 / MRR: SpecBridge 84.70/92.31/96.85/88.15; GLMR 68.48/78.09/84.22/72.47; MVP 13.96/36.88/68.12/26.26; JESTR 11.77/33.26/61.01/22.83; MIST 10.94/23.82/44.63/18.26.
  - Mean candidate pool size is 162.5.
  - Cost: about 100 ms to encode one spectrum, and about 250 queries/s batched. Code is at github.com/HassounLab/SpecBridge, and the paper does not state a license.
- GLMR ("Breaking the Modality Barrier", Zhang et al., arXiv 2511.06259, 9 Nov 2025, AAAI 2026) is two-stage: contrastive pre-retrieval, then a generative model produces refined structures that re-rank candidates by similarity. It claims an over 40% top-1 improvement on MassSpecGym and the authors' MassRET-20k dataset. No code link appears on the abstract page. Source: [arXiv 2511.06259](https://arxiv.org/abs/2511.06259)
- The audit found that a "DreaMS + ChemBERTa" method drops from 82.41% to 6.70% Hit@1 once canonicalization is fixed, and that a SMILES-only ChemBERTa classifier with no spectrum scores 99.66% on the buggy version. Source: [arXiv 2606.19624](https://arxiv.org/html/2606.19624v1)

### Inferences
- SpecBridge's architecture is DreaMS + ChemBERTa, and its reported 84.7% sits close to the audit's 82.41% inflated "DreaMS + ChemBERTa" number. It is very likely, though not confirmed, that SpecBridge (and probably GLMR, which reports 68.5% in SpecBridge's table) benefited from the SMILES canonicalization shortcut. Treat their real isomer-discrimination ability as roughly 7% Hit@1 until shown otherwise.
- Any model that reads a SMILES string can learn formatting cues. This includes ChemBERTa, ChemFormer, MolT5 and GLMR. Feed it canonical SMILES produced identically for every candidate.
- **Integration idea (DreaMS):**
  - Embed each query spectrum with frozen DreaMS.
  - Train a small head on the competition training set plus MassSpecGym that maps the DreaMS embedding to our 10,226-bit fingerprint, or contrastively to a graph or fingerprint embedding of the molecule.
  - Use the head's candidate score, plus its within-formula-group rank, as a feature.
  - Separately, use DreaMS-embedding kNN against training spectra as an analog feature: the maximum Tanimoto between a candidate and the structures of the k nearest training spectra.
  - DreaMS is pretrained on diverse GNPS data, which probably includes both ion modes and multiple instruments (UNVERIFIED), so it may handle timsTOF and negative mode better than MassSpecGym-only encoders.

### Gaps
- I did not find whether the DreaMS paper reports MassSpecGym retrieval numbers, or whether there is a released retrieval or fingerprint fine-tuned checkpoint.
- DreaMS negative-mode support and m/z or precursor-range limits are not confirmed.
- No GLMR code or license was found.

## Q5. Contrastive joint-embedding rankers: JESTR, MVP, FLARE (Hassoun lab and others)

### Takeaway
These are designed exactly to rank candidates by spectrum-molecule embedding similarity. JESTR is MIT-licensed with released NPLIB1 and MassSpecGym weights. Formula-bonus Hit@1 is modest (JESTR 11.8%, MVP 14.0%, FLARE 22.7%), well below forward-simulation methods. They are cheap and could still add complementary signal as a feature.

### Cited Findings
- JESTR (Kalia, Chen, Krishnan, Hassoun; arXiv 2411.14464, Bioinformatics 2025) ranks candidates by cosine similarity between spectrum and molecule embeddings. Source: [JESTR PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12233093/)
  - On MassSpecGym formula candidates it reports outperforming pretrained SIRIUS by 31% and CFM-ID by 238%, with the caveat that those tools may have trained on test molecules.
  - It reports 15.62% rank@1 with mass candidates.
  - Regularization with candidate molecules during training gave +5.72% rank@1.
- JESTR repo: MIT license. Pretrained GPU-trained weights are provided for NPLIB1 (data/NPLIB1) and MassSpecGym (data/MassSpecGym). NIST is not mentioned, but the other datasets are "under licensing agreements that prohibit their public release". The repo does not mention adduct support. Source: [JESTR GitHub](https://github.com/HassounLab/JESTR)
- MVP (bioRxiv 10.1101/2025.11.12.688047, Nov 2025) is a multi-view contrastive model (molecular graph plus fingerprint; individual plus consensus spectra). MassSpecGym rank@1 is 26.37% by mass and 11.10% by formula for individual spectra, and 35.99% and 13.96% with the consensus spectrum. The repo, reportedly github.com/HassounLab/MVP, is UNVERIFIED. Source: [MVP bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.12.688047.full.pdf)
- FLARE (bioRxiv 10.64898/2026.01.27.702086, Jan 2026) does fine-grained peak-to-atom alignment with max-sim interactions. MassSpecGym rank@1 is 43.15% by mass and 22.66% by formula. Sources: [FLARE bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702086.full.pdf), [FLARE PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12873900/)

### Inferences
- FLARE's peak-to-substructure alignment is the most isomer-relevant of the contrastive models, because it explicitly ties fragment peaks to atoms, and that is what differs between regioisomers. Of this group, it is the best candidate to try first, provided its code and weights are available.
- **Integration idea:** use the JESTR MassSpecGym weights (MIT, no NIST) as a quick experiment. Score cosine(spec_emb, mol_emb) for all same-formula candidates, and add the raw score plus the group-relative z-score.
- Fine-tuning on competition training data with adduct tokens is probably needed for non-[M+H]+ adducts.

### Gaps
- MVP and FLARE code, weights and license were not verified.
- None of these papers' adduct or negative-mode support was confirmed.
- MVP's consensus-spectrum gains require multiple spectra per compound, which may not exist per test molecule.

## Q6. Generative retrieval (FRIGID, DiffMS-style) as an isomer re-ranker

### Takeaway
FRIGID (ICML 2026) generates structures from MIST fingerprints plus formula with a diffusion language model, refined by forward fragmentation models. In the audit's table it reaches 45.27% formula-bonus Hit@1 as a generative-retrieval method. It is heavy and probably not worth integrating directly in a Kaggle notebook. The "spectrum-inconsistent fragment" check it uses is essentially the GLACIER/ICEBERG feature in Q2.

### Cited Findings
- FRIGID: diffusion LM conditioned on spectra via intermediate fingerprints and formulae, pretrained on hundreds of millions of unlabeled structures. Forward fragmentation models flag spectrum-inconsistent fragments for remasking. Inference-time scaling gives over 15% Top-1 on MassSpecGym de novo and more than doubles the leading methods' Top-1 on NPLIB1. Source: [ICML 2026 poster](https://icml.cc/virtual/2026/poster/60806)
- FRIGID generative retrieval: 45.27% formula-bonus Hit@1. MIST in the FRIGID version: 53.76%. Source: [arXiv 2606.19624](https://arxiv.org/html/2606.19624v1)

### Inferences
- The NPLIB1 gains are notable for natural products. However, the cheap, transferable part for a candidate-ranking pipeline is the improved MIST encoder (trained with ICEBERG augmentation) and the forward-model consistency check, not the diffusion generator.

### Gaps
- FRIGID code, weights, license and training data (whether NIST is involved) were not verified.

## Q7. Spectral-similarity and analog-search tools: MS2DeepScore, Spec2Vec, MS2Query, spectral entropy

### Takeaway
These compare spectra with spectra, not spectra with structures. They are useful in this pipeline only as (a) the similarity function between observed and forward-simulated spectra (entropy similarity is the recommended default) and (b) analog features from nearest training-library spectra. MS2DeepScore 2.0 handles cross-ion-mode similarity, which is relevant for the negative-mode adducts.

### Cited Findings
- Spectral entropy (Li et al., Nat Methods 18:1524, 2021) outperformed 42 alternative similarity algorithms, including dot product, on 434,287 spectra against NIST20. FDR was under 10% at entropy similarity 0.75 on 37,299 natural-product spectra. Flash entropy (Nat Methods 20:1475, 2023) is mathematically equivalent and more than 10,000x faster. The open-source ms-entropy package's default removes peaks below 1% of base peak. Sources: [Nat Methods 2021](https://link.springer.com/article/10.1038/s41592-021-01331-z), [MSEntropy GitHub](https://github.com/YuanyueLi/MSEntropy)
- MS2Query combines Spec2Vec, MS2DeepScore and precursor m/z. With exact matches removed, it finds analogues for 35% of spectra at an average Tanimoto of 0.63 (published version; 0.67 in the preprint), against 0.45 for modified cosine at the same recall. A separate negative-mode library is provided, and models are on Zenodo. Sources: [MS2Query Nat Commun (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10060387/), [MS2Query bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.22.501125.full.pdf)
- MS2DeepScore 2.0 is described as one unified model that predicts chemical similarity within and across ion modes, enabling negative-mode queries against positive-mode libraries. Only secondary summaries were found. Source: [WUR activity page](https://www.wur.nl/en/activity/automating-metabolomics-analysis-machine-learning-models-and-software-development-mass)

### Inferences
- **Integration idea (entropy):** use flash entropy as the comparison metric for the Q2 forward-simulation feature. Also use it for a "library hit" feature: when a training-set spectrum of the same adduct matches above 0.75, add a candidate feature for exact InChIKey first-block agreement.
- **Integration idea (MS2DeepScore / DreaMS kNN):** for each candidate, compute the maximum structural Tanimoto to the structures of the k most spectrally similar training spectra. This gives an analog prior that may separate isomer scaffolds. It is weaker for regioisomers that share a scaffold.
- MS2DeepScore and Spec2Vec training data and weights licensing were not checked. The public models are trained on GNPS/MoNA-type data (UNVERIFIED), so check them for NIST before use.

### Gaps
- I found no quantitative negative-mode benchmark for MS2DeepScore 2.0.
- MS2DeepScore and MS2Query weight licenses and training-data composition were not verified.

## Q8. SIRIUS / CSI:FingerID / CANOPUS licensing and offline use

### Takeaway
The SIRIUS client is AGPL-3.0, but CSI:FingerID, CANOPUS and MSNovelist run as web services. The academic service is free for research and education only, and commercial use goes through Bright Giant. The fingerprint and structure step therefore cannot be run fully offline in a Kaggle notebook, and its terms conflict with the competition's commercial-licensability requirement. Its models are also trained on data that includes NIST. Treat it as not usable.

### Cited Findings
- SIRIUS web services (CSI:FingerID, CANOPUS, MSNovelist) are for academic research and education only. Bright Giant GmbH is the sole license provider for non-academic users. Re-distribution of the methods for commercial purposes requires the authors' permission (one Jena page) or is prohibited (another Jena page), so the wording is inconsistent. Sources: [Jena p=8451](https://bio.informatik.uni-jena.de/?p=8451), [Jena p=5312](https://bio.informatik.uni-jena.de/?p=5312), [Bright Giant](https://bright-giant.com/?p=2128)
- The client software is listed as AGPL-3.0. The current release is 6.3.7 (2026-05-23). Source: [Wikipedia: SIRIUS](https://en.wikipedia.org/wiki/SIRIUS_(software)) (secondary source)
- The CSI:FingerID training data ("csi2022") includes NIST. Source: [MIST GitHub](https://github.com/samgoldman97/mist)

### Inferences
- SIRIUS's local fragmentation-tree and formula scoring (non-web part) might be usable offline under AGPL for formula ranking. It does not help with same-formula isomers, which is the actual bottleneck. Skip it.

### Gaps
- I did not read SIRIUS 6 terms of service directly.
- Whether Kaggle's offline-notebook rule allows any web service at all was not checked.

## Q9. Summary: shortlist for same-formula isomer re-ranking (permissive, non-NIST)

### Takeaway
Priority order based on the evidence above:
1. GLACIER or ICEBERG MassSpecGym checkpoint forward simulation, compared with flash entropy, as a feature (49.9% / 44.4% formula-bonus Hit@1, fast, host-approved per brief).
2. MIST retrained on MassSpecGym plus competition data with ICEBERG augmentation (53.76% formula-bonus Hit@1 in the leakage-free setting), as a second fingerprint opinion.
3. FLARE or JESTR contrastive score (22.7% / 11.8%, cheap; JESTR is MIT with MassSpecGym weights).
4. A DreaMS embedding plus a trained head, and DreaMS/MS2DeepScore kNN analog features.

Avoid the SpecBridge/GLMR-style SMILES-LM retrieval numbers (likely shortcut-inflated), SIRIUS/CSI:FingerID (license and NIST), and all NIST'20 checkpoints.

### Cited Findings
- Formula-bonus Hit@1 figures and the leakage and shortcut corrections: [arXiv 2606.19624](https://arxiv.org/html/2606.19624v1), [arXiv 2606.29161](https://arxiv.org/html/2606.29161), [FLARE bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702086.full.pdf)
- Licenses: MIST MIT ([GitHub](https://github.com/samgoldman97/mist)); JESTR MIT ([GitHub](https://github.com/HassounLab/JESTR)); DreaMS code MIT, weights CC BY 4.0 ([GitHub](https://github.com/pluskal-lab/DreaMS), [Zenodo](https://zenodo.org/records/10997887)); MassSpecGym MIT ([GitHub](https://github.com/pluskal-lab/MassSpecGym))

### Inferences
- **General integration pattern for every feature:**
  - Compute the model score s for each candidate.
  - Then compute s - max(s within same formula group), the rank within the formula group, and the z-score within the group.
  - Let LightGBM (lambdarank) learn from the result.
  - Train these features on held-out folds split by the MCES or scaffold of the competition training molecules, so they are not overfit to memorized structures.
- **Expected impact:** the pipeline already has about 88% top-25 recall, so MRR gains come almost entirely from moving the truth from ranks 2 to 5 up to rank 1 within a formula group. That is exactly what formula-bonus Hit@1 measures. Going from FPNet-like performance (Fingerprint FFN 5% / MIST 10% formula-bonus Hit@1) to forward-simulation performance (about 45 to 50%) is the largest documented jump available. Benchmark gains won't transfer one-for-one to timsTOF natural products.

### Gaps
- Adduct coverage is the largest unresolved risk. None of the verified permissive checkpoints are confirmed to support [M+NH4]+, water-loss adducts, [M+K]+ or any negative-mode adduct.
- Not researched because of the call budget:
  - CFM-ID 4 details.
  - The arXiv papers 2602.16507 ("what are we optimizing for?") and 2603.10950 (selective prediction for retrieval). They surfaced in search but were not read.
  - MolBERT-style models, "CLERMS" and "ESA". No sources were found for the last two names.
