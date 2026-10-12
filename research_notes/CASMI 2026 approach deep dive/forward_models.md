# Forward MS/MS spectrum predictors for isomer re-ranking (CASMI 2026 / Enveda Kaggle)

State as of 2026-10-11. Research budget was about 20 tool calls, so several items below are marked unverified or listed as gaps. Model names (ICEBERG 2.1, GLACIER, MARASON, SCARF) follow the coleygroup/ms-pred README.

## Q1. Which forward models lead the isomer-retrieval benchmarks, and how do they compare on accuracy and speed?

### Takeaway
GLACIER (Coley lab, arXiv 2606.29161, June 2026) is the current published leader on both MassSpecGym and NIST'20 retrieval, and it is about 8x faster than ICEBERG 2.0. The ranking is GLACIER > ICEBERG 2.0/2.1 > MARASON > FraGNNet > NEIMS/MassFormer-class models. CFM-ID and 3DMolMS are not head-to-head in the GLACIER tables I could read.

### Cited Findings
**ms-pred repo and versions**
- ms-pred lists these models: GLACIER, "state-of-the-art single-stage MS/MS simulator"; ICEBERG 2.1 (recommended; fragment generator plus intensity predictor; about 3x speedup and a "substantial GPU speedup"); ICEBERG 1.0 (legacy, on the `iceberg_1.0` branch, the only branch that still supports NPLIB1); MARASON (retrieval-augmented intensity head, arXiv 2502.17874, ICML 2025); SCARF (formula-level prefix-tree decoding, arXiv 2303.06470, NeurIPS 2023). Baselines are NEIMS (FFN/GNN), MassFormer, 3DMolMS, GrAFF-MS and CFM-ID (CFM-ID is not retrained). — [ms-pred README](https://github.com/coleygroup/ms-pred)
- Changelog: on 7/7/2026, ICEBERG 2.1 was released with a pretrained NIST'23 model and GLACIER was introduced. On 4/4/2026 the ICEBERG atlas went live (iceberg-ms.mit.edu, with predicted spectra for PubChem). — [ms-pred README](https://github.com/coleygroup/ms-pred)

**MassSpecGym results**
- GLACIER top-1 retrieval on MassSpecGym:
  - Mass challenge: 69.95% with contrastive finetuning (CF), 69.7% without, vs. ICEBERG 2.0 at 63.95%.
  - Formula (same-formula isomer) challenge: GLACIER 49.88% (with CF) vs. ICEBERG 2.0 44.35%, MARASON 34.03%, FraGNNet 31.93%.
  - Cosine similarity: GLACIER 0.58 vs. ICEBERG 0.47.
  - Source: [GLACIER arXiv HTML](https://arxiv.org/html/2606.29161) and [abstract](https://arxiv.org/abs/2606.29161).
- Original MassSpecGym simulation baselines:

  | Model | Cosine | JS | HR@1 | HR@5 | HR@20 |
  |---|---|---|---|---|---|
  | Precursor m/z | 0.15 | 0.15 | 0.38 | – | – |
  | FFN fingerprint | 0.25 | 0.24 | 8.44 | – | – |
  | GNN | 0.19 | 0.20 | 3.95 | – | – |
  | FraGNNet | 0.52 | 0.47 | 46.64 | 72.56 | 83.58 |

  FraGNNet on the formula bonus challenge: HR@1/@5/@20 of 31.93/63.20/82.70. — [MassSpecGym paper](https://arxiv.org/html/2410.23326)
- A later paper (GIF) quotes FraGNNet's JS similarity as 0.91, which conflicts with the 0.47 in the MassSpecGym paper; trust the original. — [GIF arXiv 2511.09571](https://arxiv.org/pdf/2511.09571)

**NIST'20 results**
- GLACIER top-1 with CF: 52.5% on the random split and 46.0% on the scaffold split, vs. ICEBERG 2.0 at 33.5% and 31.6%.
- GLACIER cosine: 0.821 without CF and 0.802 with CF, vs. ICEBERG 0.773.
- [M+H]+ subset (Table 21): top-1 53.3% (random) and 45.2% (scaffold), vs. ICEBERG 2.0 at 40.0% and 33.5%. Top-10 is about 93% (random) and 91% (scaffold).
- Source: [GLACIER arXiv HTML](https://arxiv.org/html/2606.29161)
- ICEBERG 2.0 paper: [M+H]+ top-1 among PubChem isomers is 40.0% (random) and 33.5% (scaffold), up from 25.1% and 22.9% for ICEBERG 1.0. About 92% of correct structures land in the top 10. — [bioRxiv 2025.05.28.656653](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full)
- MARASON reports 28% top-1 vs. 19% for the prior non-RAG state of the art on NIST'20. — [arXiv 2502.17874](https://arxiv.org/html/2502.17874v1)

**Contrastive finetuning trade-off**
- CF improves retrieval but slightly lowers spectral cosine. Its negatives come from PubChem and from random mutation. The authors note a possible shortcut-learning risk depending on decoy sampling, though it did not appear empirically. — [GLACIER](https://arxiv.org/html/2606.29161)
- ms-pred's contrastive finetuning uses `pubchem_formulae_inchikey.hdf5`. — [Zenodo 15529765](https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5), via [ms-pred README](https://github.com/coleygroup/ms-pred)

**Inference speed**
- GLACIER: 1.3 ms/spectrum (368.7 spectra/s/GPU). ICEBERG 2.0: 10.8 ms/spectrum (46.4 spectra/s/GPU). That is a 7.95x speedup, measured on 2x NVIDIA A5000 over the 2.7M-spectrum NIST'20 scaffold run. — [GLACIER](https://arxiv.org/html/2606.29161)
- CFM-ID 4.0 web server: under 1 min per molecule below 1000 Da, with a 3 min timeout. — [CFM-ID 4.0 web server paper, PMC9252813](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252813/)

**Other 2025-2026 models**
- "Neural Spectral Prediction for Structure Elucidation with Tandem Mass Spectrometry" (bioRxiv 2025.05.28) is the ICEBERG 2.0 paper. It describes collision-energy- and polarity-aware fragment generation and scoring. — [PMC12154671](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12154671/)
- FRIGID (arXiv 2604.16648, April 2026) is not a forward model. It is a spectrum-to-structure diffusion generator that uses ICEBERG for inference-time refinement. — [emergentmind summary](https://www.emergentmind.com/papers/2604.16648)

### Inferences
- **Ranking for re-ranking same-formula candidates:** GLACIER is the strongest single forward signal published, ahead of ICEBERG 2.x, MARASON, FraGNNet and the rest. On the MassSpecGym formula challenge, GLACIER beats ICEBERG 2.0 by about 5.5 points top-1. The two are probably correlated (same lab, same fragment vocabulary), so an ensemble gain is plausible but unproven (see Q3).
- **GLACIER is MSG-trained here:** the pipeline's GLACIER checkpoint was trained on MassSpecGym, so its expected same-formula top-1 is about 50% (CF variant) on MSG-like data. Natural-product timsTOF spectra are out of domain.
- **T4 throughput estimate (unverified):** a T4 has roughly 1/2 to 1/3 of A5000 FP32 throughput. That gives about 120-180 spectra/s for GLACIER and about 15-25 spectra/s for ICEBERG 2.0. ICEBERG 2.1's claimed 3x speedup would bring it to about 45-70 spectra/s.
- **Budget for the Kaggle workload:** 400 molecules × 100 candidates × about 3 spectra each (adduct/CE combinations) is about 120k predictions.
  - GLACIER: about 15 min on a T4.
  - ICEBERG 2.1: about 30-45 min.
  - ICEBERG 2.0: about 1.5-2 h.
  - CFM-ID: 40k candidates × about 1 CPU-min each is about 28 CPU-days, which is infeasible on Kaggle CPUs without heavy pruning (for example, the top 10 candidates only).
  - Doing 10 CE values per candidate scales these numbers linearly.

### Gaps
- No public MassSpecGym simulation leaderboard snapshot was retrieved for October 2026. Whether anything newer than GLACIER tops it is unverified.
- No direct measured T4 timings were found for any model. The figures above are extrapolations.
- Benchmark numbers for SCARF, GrAFF-MS, MassFormer, 3DMolMS, NEIMS and CFM-ID on the same-formula challenge were not retrieved from primary sources. The GLACIER visible text only includes NEIMS, MolSpecFlow, GrAFF-MS and MassFormer in its tables.
- FraGNNet's code and weights availability could not be confirmed. A search found the paper (TMLR 2025) but no repo: [arXiv 2404.02360](https://arxiv.org/html/2404.02360v2), [TMLR](https://mlanthology.org/tmlr/2025/young2025tmlr-fragnnet).

## Q2. Adducts, negative mode, CE/instrument conditioning, non-NIST weights and licenses

### Takeaway
No available non-NIST forward-model weights cover the full 10-adduct test set.
- MassSpecGym-trained ms-pred checkpoints only see [M+H]+ and [M+Na]+.
- ICEBERG 2.0/2.1 and GLACIER support negative mode and water-loss adducts only in their NIST-trained form, which is banned here. Even that NIST adduct list lacks formate [M+CH2O2-H]-.
- CFM-ID 4 covers only [M+H]+ and [M-H]- properly.
- 3DMolMS covers [M+H]+, [M-H]-, [M+H-H2O]+ and [M+Na]+, but is CC BY-NC-SA.

So for [M+NH4]+, [M+K]+, [M-2H2O+H]+, [M-H2O-H]-, formate and [M+Cl]-, the team's own fine-tuning on competition data is the only route.

### Cited Findings
**ICEBERG 2.0 (NIST'20 HCD)**
- Positive adducts: [M+H]+, [M+Na]+, [M+K]+, [M−H2O+H]+, [M−2H2O+H]+, [M+NH3+H]+.
- Negative adducts: [M−H]−, [M+Cl]−, [M−H2O−H]−, [M−CO2−H]−.
- Collision energy goes in as a 64-d positional embedding.
- Trained on Orbitrap HCD only. QTOF is noted as possible via retraining or fine-tuning but was not tested.
- Weights are available only to NIST'20 license holders.
- Source: [bioRxiv 2025.05.28.656653](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full)

**GLACIER**
- Takes adduct (one-hot) and collision energy (positional encoding) as inputs to node embeddings and attention biases.
- NIST'20 evaluation covers "all adduct types." Positive+negative cosine is 0.821, positive-only 0.829, [M+H]+ 0.838 (random split, no CF). There is no separate negative-mode table.
- Source: [GLACIER](https://arxiv.org/html/2606.29161)

**MassSpecGym**
- Spectra are [M+H]+ or [M+Na]+ only. The simulation task is restricted to [M+H]+.
- 98% of spectra are Orbitrap or QTOF, and about 53% have collision energy.
- Inputs are graph, instrument type, adduct and CE. Spectra are binned at 0.01 Da up to 1005 Da.
- Retrieval uses up to 256 same-mass or same-formula candidates, ranked by cosine.
- Source: [MassSpecGym](https://arxiv.org/html/2410.23326)
- ms-pred offers two MSG settings: `msg_simulation` (119,029 spectra) and `msg_all`, where missing CEs are imputed with ICEBERG. Baselines do not use CE or instrument as covariates. NIST is the only database with CE for every spectrum. — [ms-pred README](https://github.com/coleygroup/ms-pred)

**ms-pred checkpoints and licenses**
- Public (Dropbox) checkpoints: ICEBERG 2.1 on MSG `msg_all` and on `msg_simulation`, and GLACIER on MSG. ICEBERG on NIST'23 is "official." NIST'20 weights are available on request with a license. No MARASON or SCARF checkpoints are listed. — [ms-pred README](https://github.com/coleygroup/ms-pred)
- ms-pred code is under the MIT license (© 2023 Samuel Goldman). The license text does not mention model weights, and the README states no separate weight license. — [LICENSE](https://raw.githubusercontent.com/coleygroup/ms-pred/main/LICENSE), [README](https://github.com/coleygroup/ms-pred)
- MARASON retrieves references from the NIST training library, and its pretrained weights are only for NIST licensees. — [arXiv 2502.17874](https://arxiv.org/html/2502.17874v1), [search summary](https://arxiv.org/pdf/2502.17874)

**CFM-ID 4.0**
- Fully supports [M+H]+ and [M-H]-. Other adducts are only "partially supported": the output is the [M+H]+ or [M-H]- spectrum plus a precursor-adduct peak.
- Spectra are ESI-QTOF at 10, 20 and 40 eV, and the model was "trained exclusively on QTOF CID data."
- The paper does not name its training libraries.
- Source: [PMC9252813](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252813/)
- Earlier CFM-ID work trained on METLIN Agilent 6510 Q-TOF positive spectra at 10/20/40 V. — [arXiv 1312.0264](https://arxiv.org/pdf/1312.0264)
- Docker image `wishartlab/cfmid`:
  - Bundles CFM-ID 4.0 trained models for [M+H]+ and [M-H]- under `/trained_models_cfmid4.0/`.
  - Run with `docker run -v $(pwd):/cfmid/public/ wishartlab/cfmid:latest sh -c "cfm-predict ..."`.
  - Tag `4.2.6.0`/`latest` contains MSML 4.0.8 and MSRB 1.1.4.
  - The image was last updated about 4 years ago, and no license is stated on the page.
  - Source: [Docker Hub](https://hub.docker.com/r/wishartlab/cfmid)
  - Newer tags such as `msml_4.4.10` exist. — [Docker Hub tags](https://hub.docker.com/r/wishartlab/cfmid/tags)

**3DMolMS**
- Precursor types: [M+H]+, [M-H]-, [M+H-H2O]+, [M+Na]+, [M+2H]2+.
- Instrument flag `qtof` or `orbitrap`, with any CE value.
- Checkpoints download automatically. Transfer learning is supported via `transfer=True` and `freeze_encoder`.
- License is CC BY-NC-SA 4.0. Training data is not named in the README.
- Source: [3DMolMS GitHub](https://github.com/JosieHong/3DMolMS)
- Reported cross-lab fine-tuning cosine is 0.691 (positive) and 0.478 (negative). — [search summary of PNNL/3DMolMS](https://www.pnnl.gov/publications/advancing-prediction-msms-spectra-using-machine-learning) (unverified against the primary paper)

**MassFormer**
- Restricted to positive-mode ESI Orbitrap with specific adducts. — [arXiv 2111.04824](https://arxiv.org/pdf/2111.04824)
- The public Zenodo data excludes NIST. The public demo checkpoint is trained on MoNA, and other pretrained models are by request. — [Zenodo 8371051](https://zenodo.org/records/8371051/latest)

### Inferences
- **Formate is not covered by any public model.** The [M+CH2O2-H]- adduct is not in ICEBERG 2.0's NIST adduct list, and none of the public models cover it. The model would have to learn it from competition timsTOF data.
- **Water-loss adducts are an approximation.** ICEBERG models [M−H2O+H]+ as a distinct adduct. When using MSG-only weights, a crude fallback is to predict [M+H]+ and score only fragments below the water-loss precursor. This is untested.
- **Keep CFM-ID if it is used.** CFM-ID's QTOF CID training is closer to timsTOF CID than the Orbitrap HCD behind NIST and much of MSG. That is the main argument for keeping it as a weak [M+H]+/[M-H]- signal, but its runtime makes it top-k-only.
- **Check competition rules for licenses.** The CC BY-NC-SA terms on 3DMolMS (and any weights whose license is unstated) need checking against the competition rules on external models and winner licensing. Not checked here.

### Gaps
- **CFM-ID 4.0 training libraries:** whether METLIN or NIST was used for the 4.0 positive and negative parameter sets is not stated in the sources found. The 4.0 Anal. Chem. paper (doi:10.1021/acs.analchem.1c01465) or its SI must be read. Flag a possible NIST/METLIN-derived restriction as unresolved.
- **CFM-ID license:** not found. The versions page timed out, and neither Docker Hub nor the paper names one.
- **CFM-ID on Kaggle:** whether it can run in a Kaggle notebook (no Docker daemon) would require extracting the binary from the image. Untested.
- **Training data of public checkpoints:** the datasets behind the public 3DMolMS checkpoints are not stated in the README. They may include NIST, so verify before use.
- **SCARF:** adducts and training data were not retrieved. The abstract page lacks them, and no checkpoints are listed in ms-pred.

## Q3. Ensembling, instrument fine-tuning and the similarity function

### Takeaway
I found no published study on ensembling multiple forward predictors for retrieval.

Domain adaptation evidence is thin:
- 3DMolMS reports small-set fine-tuning across instruments.
- The ICEBERG authors suggest QTOF fine-tuning but did not test it.
- Contrastive finetuning is the documented lever for retrieval specifically.

On similarity, entropy similarity is the best-documented measure for experimental library matching. The forward-model papers themselves rank by binned cosine.

### Cited Findings
- **Ranking metric in the forward-model papers:** GLACIER and MassSpecGym rank by cosine (with Jensen-Shannon / "spectral entropy" similarity also reported) on binned spectra: 0.1 Da bins for NIST'20, 0.01 Da for MSG. — [GLACIER](https://arxiv.org/html/2606.29161), [MassSpecGym](https://arxiv.org/html/2410.23326)
- **Entropy similarity (Li et al., Nat. Methods 18:1524, 2021):**
  - Outperformed 42 alternatives, including dot product, across 434,287 query spectra searched against NIST20.
  - Robust to added noise ions.
  - FDR below 10% at score 0.75 on 37,299 natural-product spectra.
  - The authors recommend removing noise ions before scoring.
  - Source: [Nature Methods](https://link.springer.com/article/10.1038/s41592-021-01331-z), [MSEntropy GitHub](https://github.com/YuanyueLi/MSEntropy)
  - Flash entropy (Nat. Methods 2023) computes the same score much faster. — [MSEntropy docs](https://msentropy.readthedocs.io/en/latest/classical_entropy_similarity.html)
- **Contrastive finetuning:** improves retrieval at a small cosine cost, with τ = 0.5. Decoy filtering had little effect (Table 15). — [GLACIER](https://arxiv.org/html/2606.29161)
- **Instrument adaptation:**
  - ICEBERG 2.0 notes QTOF could be supported via fine-tuning, untested. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full)
  - 3DMolMS supports transfer learning with a frozen or unfrozen encoder. — [3DMolMS](https://github.com/JosieHong/3DMolMS)
- **Benchmarking caution:** a PNNL review warns that generic ML benchmarking can give misleading accuracy. It recommends curation and restricting predictions to sufficiently high CE. — [PNNL 2024](https://www.pnnl.gov/publications/advancing-prediction-msms-spectra-using-machine-learning)

### Inferences
- **Context from the team's own run:** the GLACIER fine-tune on about 120k timsTOF/NP spectra (MRR 0.375 → 0.526 on non-[M+H]+, 0.718 → 0.698 on [M+H]+ NP) matches the published pattern that adaptation helps out-of-domain adducts and can cost in-domain accuracy.
- **Cheap mitigation (unverified):** route by adduct. Use the base MSG GLACIER (or ICEBERG 2.1, or an average) for [M+H]+ and [M+Na]+, and the fine-tuned GLACIER for everything else.
- **Rank-level ensembling:** combine z-scored or rank-normalized similarities across models (GLACIER base, GLACIER-FT, ICEBERG 2.1). Fit the weights on a held-out NP panel per adduct group rather than assuming equal weights.
- **Contrastive finetuning on the competition data:** applying ms-pred's CF stage (PubChem same-formula decoys) to the timsTOF-fine-tuned GLACIER is the most direct published lever for isomer ranking, because CF targets retrieval rather than cosine.
- **Similarity to use:**
  - Compute both entropy similarity (with noise removal and an m/z tolerance of about 0.01-0.02 Da, suitable for timsTOF) and binned cosine with sqrt intensity weighting.
  - Pick or blend them on the held-out panel.
  - Entropy's evidence is for experimental-vs-library matching, not predicted-vs-measured, so treat its advantage as a hypothesis to test.
  - The tolerance values are a suggestion, not sourced.

### Gaps
- No paper found that quantifies ensembling of ICEBERG/GLACIER/CFM-ID for retrieval.
- No timsTOF-specific small-molecule forward predictor was found. A search for timsTOF/ion-mobility forward prediction returned nothing relevant.
- No study found that compares similarity functions specifically for predicted-vs-experimental spectra in the isomer-ranking setting.
