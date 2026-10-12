---
status: active
area: portfolio
tags: [casmi26, research, natural-products, ion-mobility, priors]
updated: 2026-10-11
---

# Natural-product priors, structure databases, and orthogonal signals (CCS, RT) for CASMI 2026

Scope: what can help rank same-formula isomer candidates for Enveda's natural-product test set, given the test file carries no CCS and no RT. State as of 2026-10-11. Items not confirmed against a source are marked UNVERIFIED.

## Q1. Does any competition data include CCS or retention time?

### Takeaway
The test schema used by public notebooks has no CCS, 1/K0 or RT column, so ion mobility and RT cannot be used directly as signals. CCS appears only indirectly: Enveda measures it on its timsTOF platform, but it is not released. I could not open the Kaggle data page to confirm the train-file schema, so "no CCS anywhere in the release" stays unconfirmed for the training files.

### Cited Findings
- Test columns, as hard-coded in a local copy of a public notebook that rebuilds a validation `test.parquet` in the competition format: `molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_ev, collision_energy_orig, collision_energy_orig_units`. No CCS or RT field. Source: local notebook `/Users/b/casmi26/ref/v4n/` (code cell defining `test_cols`). The same notebook notes that an "enveda-np-examples" library in the provided data "really is timsTOF".
- Competition files referenced by public notebooks: `test.parquet`, `train.parquet`, `train_structs.parquet`, `sample_submission.csv`. Grep of all 10 local reference notebooks found no column name containing ccs, mobility, 1/K0 or retention. Source: local notebooks under `/Users/b/casmi26/ref/`.
- The Kaggle data page (https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) returned only its title to WebFetch (JS-rendered), so the official column descriptions were not read.
- Test set: about 400 molecules, about 2,500 spectra, generated on Enveda's isolation and profiling platforms, none published; molecules are "real or potential natural molecules and their analogs". Training: about 2.5M spectra from 275,000 molecules, including Enveda-180 (about 1.16M spectra, 184,330 synthetic small molecules). [Business Wire, 2026-09-15](https://secure.businesswire.com/news/home/20260915334355/en/Enveda-Is-Launching-the-Next-AlphaFold-Contest.-This-Time-to-Solve-the-Structures-of-90-of-Biological-Molecules-That-No-One-Can-Identify.)
- Enveda uses timsTOF Pro 2 and says the key to NP identification is combining clean MS/MS with ion mobility; a senior scientist calls CCS "a single-point encoding of a molecule's shape", used to link molecules across acquisitions. [Bruker customer page](https://www.bruker.com/en/landingpages/bdal/customer-insight-enveda.html)

### Inferences
- Enveda holds CCS for the test molecules but chose not to release it, so any CCS-based method has nothing to compare predictions against at test time. CCS predictors are only useful (weakly) as a learned 3D-shape feature, not as a filter.
- The precursor m/z plus adduct are the only physical constraints. Same-formula isomers have identical precursor m/z, so isomer ranking rests on MS2 and priors alone.

### Gaps
- Train/`train_structs` column list not verified from the official data page; check the Data tab or `pd.read_parquet(...).columns` in a Kaggle session.
- Whether Enveda-180 was acquired on timsTOF: not stated in any source found. No dataset paper for Enveda-180 was found.

## Q2. Natural-product structure databases: size, license, overlap

### Takeaway
COCONUT 2.0 already aggregates 63 NP collections (including NPAtlas, NPASS, SuperNatural 3, KNApSAcK, StreptomeDB, PubChem NPs), which explains the forum report that adding NPAtlas/ChEBI/LIPID MAPS gave no new coverage. The only plausible independent addition is LOTUS (Wikidata, CC0 by inheritance, UNVERIFIED), mainly for its organism links rather than new structures. Most other NP databases are non-commercial or unlicensed and are risky for prize eligibility.

### Cited Findings
- COCONUT 2.0 (NAR 2025, 53(D1):D634-D643): 695,133 unique NP structures (539,350 with defined stereo, 82,220 with no stereocenters, 73,563 with undefined stereo); about 1.5M entries imported from 63 collections (up from 53); 1,022,536 loaded before dedup by canonical SMILES; synthetic-looking compounds removed, but the authors say the NP-likeness distribution suggests synthetics remain. [PMC11701633](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701633/)
- COCONUT sources named in the paper include NPAtlas, NPASS, KNApSAcK, StreptomeDB, SuperNatural 3, PubChem NPs, FooDB, CMNPD, TIPdb, NANPDB, Phyto4Health, LANaPDB and others; LOTUS and CMAUP are not named in the article text (full list is in its Supplementary 1). [PMC11701633](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701633/)
- COCONUT license: the About page says code is MIT and "Complete curated dataset is available for download under CC0 License." [COCONUT about](https://coconut.naturalproducts.net/about). This conflicts with the brief's "COCONUT CC BY 4.0"; the CC BY 4.0 may refer to the NAR article itself, which is CC BY 4.0 per [PMC11701633](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701633/). Either way it is commercially usable.
- COCONUT Zenodo full dump: [zenodo 13897048](https://zenodo.org/records/13897048).
- LOTUS: 750,000+ referenced structure-organism pairs, hosted on Wikidata and mirrored at lotus.naturalproducts.net; less than 10% of species hold over 80% of its structural diversity. [eLife/PMC9135406](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9135406/). License of the data: not stated in the sources read; Wikidata content is CC0 in general (UNVERIFIED for this note).
- NPAtlas: data under CC BY-NC 4.0 (non-commercial) per its download page; NPAtlas 2.0 had 32,552 microbial compounds; latest release v2024_09. [npatlas.org/download](https://npatlas.org/download); [NPAtlas 2.0](https://kerwa.ucr.ac.cr/items/2529a38a-573e-4da8-b186-6fccecd20833/full)
- NPASS: no explicit license; another registry inferred CC BY-NC. [Bioregistry](https://bioregistry.io/npass)
- SuperNatural 3.0: 449,058 compounds, free without login, no license stated. [NGDC Database Commons](https://ngdc.cncb.ac.cn/databasecommons/database/id/1597); [SuperNatural FAQ](https://bioinf-applied.charite.de/supernatural_3/subpages/faq.php)
- HMDB: data under CC BY-NC 4.0; commercial use requires explicit author permission. [HMDB about](https://hmdb.ca/about); [HMDB downloads](https://hmdbfix.wishartlab.com/downloads)
- KNApSAcK: "freely available", no license found. [J-STAGE](https://www.jstage.jst.go.jp/article/jcac/7/0/7_0_94/_article)
- CMAUP: asks for citation, no license found. [bidd.group/CMAUP](https://www.bidd.group/CMAUP)
- StreptomeDB: freely accessible, no license found; 4.0 reports over 8,500 NPs. [StreptomeDB 2.0](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4702922/); [StreptomeDB 4.0 PubMed](https://pubmed.ncbi.nlm.nih.gov/39498494/)
- MIBiG 3.0/4.0: papers CC BY 4.0; data license not confirmed on the site. [MIBiG 4.0](https://wrap.warwick.ac.uk/id/eprint/189432)

| DB | Size | License (commercial?) | Adds beyond PubChem+COCONUT? |
|---|---|---|---|
| COCONUT 2.0 | 695k unique | CC0 data (yes) | baseline |
| LOTUS/Wikidata | 750k+ structure-organism pairs | CC0 likely (UNVERIFIED) | taxonomy links; structure overlap with COCONUT likely high (UNVERIFIED) |
| NPAtlas | ~33k+ microbial | CC BY-NC (no) | none per forum; already inside COCONUT |
| NPASS | not read | none / inferred NC (risky) | already inside COCONUT |
| SuperNatural 3.0 | 449k | none stated (risky) | already inside COCONUT |
| KNApSAcK, StreptomeDB, CMAUP | not read | none stated (risky) | KNApSAcK, StreptomeDB inside COCONUT |
| HMDB | not read | CC BY-NC, permission needed (no) | mostly human metabolites, low NP gain |
| MIBiG | not read | papers CC BY; data UNVERIFIED | small, BGC products only |
| Dictionary of Natural Products | not read | commercial (excluded) | n/a |

### Inferences
- New structure coverage from more NP databases is near zero because COCONUT is already the union. The marginal value lies in metadata (organism of origin, NP class, citation count), not in candidate pools.
- Since the test includes "analogs" and Enveda-sourced unpublished molecules, a sizable share of answers may be in no database at all; database expansion cannot fix those.

### Gaps
- Official LOTUS data license and current structure count not read.
- Exact COCONUT-to-LOTUS structural overlap not found.
- Dictionary of Natural Products size/terms not researched (assumed excluded as commercial).

## Q3. Priors for ranking isomers among NP candidates

### Takeaway
The strongest published NP prior is taxonomic: Rutz et al. 2019 raised top-1 annotation from 10.2% to about 72% for an in-silico DB, and from 46% to about 72% for SIRIUS. But it needs the sample's source organism, which the competition does not give. Organism-free priors (NP-likeness, class consistency) have smaller and older evidence.

### Cited Findings
- Taxonomically informed scoring (Rutz et al., Front. Plant Sci. 2019, 10:1329) combines spectral similarity, taxonomic distance between query organism and candidate source, cluster structural consistency and physico-chemical consistency. On a 2,107-spectrum benchmark: ISDB-DNP top-1 from 214 (10.2%) to 1,510 (about 71.7%); SIRIUS from 975 (46.3%) to 1,508 (about 71.6%); MS-Finder from 180 (8.5%) to 546 (about 25.9%). [PMC6824209](https://pmc.ncbi.nlm.nih.gov/articles/PMC6824209/). Maintained successor: R package `tima`. [r-universe](https://taxonomicallyinformedannotation.r-universe.dev/tima/citation.txt)
- NP-likeness for candidate ranking (BMC Bioinformatics 2014, NMR-based CASE, small set): top-1 among correctly predicted cases rose from 9/34 (NMR only) to 17/35 (NMR + NP-likeness). [PMC4089557](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4089557/)
- CASMI 2013: MetFrag plus a metabolite-likeness score helped on the NP challenges. [Metabolites 2013](https://www.mdpi.com/2218-1989/3/3/623)
- CANOPUS predicts 2,497 compound classes from MS/MS without reference DB; it is a classifier, not a candidate re-ranker. Its authors warn ClassyFire classes can be counterintuitive (e.g. phytosteroids labelled bile acids). [Jena Bioinformatics blog](https://bio.informatik.uni-jena.de/tag/canopus); paper doi:10.1038/s41587-020-0740-8. The Böcker web services are restricted to academic use per a third-party note, not verified against official terms.

### Inferences
- Without organism metadata, a full taxonomic prior is not applicable. A weak version could still work: Enveda samples are plants ([Bruker page](https://www.bruker.com/en/landingpages/bdal/customer-insight-enveda.html)), so a "reported in any plant (LOTUS/COCONUT organism field)" flag, or "reported in a plant family frequent in COCONUT", could act as a kingdom-level prior that demotes microbial/marine/synthetic isomers. Benefit untested.
- Consensus class consistency is closer to what Rutz call "cluster structural consistency": spectra from the same `molecule_id` (multiple CEs/adducts) should agree on class, and molecules can be clustered by spectral similarity in test so that candidates sharing an NPClassifier class with the cluster consensus get a boost. Untested here.
- NP-likeness (RDKit Contrib `NP_Score`) is free and orthogonal to PubChem popularity; it may soften the popularity prior's bias toward famous NPs by rewarding NP-shaped but unpopular isomers. Gains in the literature are modest and from small sets.

### Gaps
- No published MS-based benchmark found that measures NPClassifier/CANOPUS class filtering of structure candidates.
- NP-Scout and biosynthetic-plausibility scores: not researched within the call budget.

## Q4. CCS predictors and their isomer-separating power

### Takeaway
Best CCS predictors reach about 1 to 1.2% median relative error, which is about the same size as typical CCS differences between constitutional isomers, so predicted CCS separates isomers only coarsely. Since the test has no measured CCS, there is nothing to match predictions against; CCS predictors are not useful here beyond being a shape descriptor.

### Cited Findings
- SigmaCCS (GNN): R² 0.9945, median relative error 1.175% on test; predicted 282M CCS values for 94,161,201 PubChem compounds (three adducts). In a lipid case, recall@1 rose from 15.2% to 24.6% then 28.9% as RT then CCS filters were added; recall@30 from 47.6% to 91.2%. Isomer discrimination is not quantified. [PMC10319785](https://pmc.ncbi.nlm.nih.gov/articles/PMC10319785/)
- DeepCCS: R² 0.976, median relative error 2.67%. [Commun. Chem. 2023](https://link.springer.com/10.1038/s42004-023-00939-w)
- A CCS compendium filter narrowed the chemical search space by 60% for human serum metabolites (single application). [Chem. Sci. 2019](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04396e)
- LASSO lipid CCS model: MRE <1.1%, allowed differentiating cis/trans and sn-positional lipid isomers (qualitative). [Analyst 2022](https://pubs.rsc.org/en/content/articlelanding/2022/an/d1an02161c)
- SIRIUS candidates filtered by predicted vs measured CCS in a 2025 spatial metabolomics workflow with experimental CCS accuracy below 0.4%. [PMC12142677](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12142677/)
- Waters/Novartis note: predicted CCS can show when isomers need cyclic IMS separation, implying predictions often cannot resolve them alone. [Waters app note](https://www.waters.com/nextgen/en/library/application-notes/2022/use-of-predicted-versus-measured-ccs-values-from-different-instrument-platforms-and-isomer-separation-on-the-select-series-cyclic-ims.html)
- Bruker CCS-Predict Pro can rank candidates by predicted CCS (vendor claim, no numbers). [Bruker](https://bruker.com/en/landingpages/bdal/ccs-predict-pro.html)

### Inferences
- With no measured CCS, CCS-based filtering is not applicable. Do not spend effort on AllCCS/CCSbase/Mol2CCS integration.
- No Enveda publication using CCS for structure ranking was found; their use is cross-run alignment.

### Gaps
- AllCCS2, CCSbase, Mol2CCS, CCSP 2.0 accuracy figures not retrieved.
- No head-to-head isomer discrimination rate for any predictor found.

## Q5. Retention-time predictors

### Takeaway
Irrelevant for the test file because no RT is given. Literature numbers are listed only for completeness.

### Cited Findings
- GNN-RT: mean relative error 4.9%, median 3.2% on METLIN SMRT (seminar abstract). [DICP](https://i.dicp.ac.cn/info/1005/13731.htm)
- METLIN SMRT paper: correct identity in the top 3 by predicted RT in 70% of cases. [PMC6925099](https://pmc.ncbi.nlm.nih.gov/articles/PMC6925099)

### Inferences
- RT predictors could only enter as an extra learned descriptor (lipophilicity proxy), with no target to compare against. Not worth pursuing.

### Gaps
- None material to the competition.

## Q6. Enveda publications: hints about test chemistry

### Takeaway
Enveda's public work points to plant-derived NPs on timsTOF, and its EnvedaDark benchmark (226 NPs absent from major DBs) suggests many test answers may be outside all databases. No source names the dominant plant families or classes in the test set.

### Cited Findings
- MS2Mol (ChemRxiv 2023): seq2seq transformer for de novo structure; EnvedaDark benchmark of 226 NPs not in major databases; 21% close-match, 62% meaningful similarity; top-10% confidence reaches 63% / 98%. [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6492507524989702c2b082fc)
- PRISM: BERT-style masked-peak pretraining on 1.2B unannotated spectra; company-reported +7 to 16% on 12 property predictions and +23% structure prediction for unknowns. [Enveda PRISM blog](https://structure-function.notion.site/PRISM-A-foundation-model-for-life-s-chemistry-Enveda-Biosciences-8a3b58b127df4e728978f3e5350e7df5); [SynBioBeta](https://www.synbiobeta.com/read/enveda-biosciences-collaborates-with-microsoft-to-unveil-prism-revolutionizing-drug-discovery-with-ai)
- Bruker page: samples are plant-derived phytochemicals; no specific classes named. [Bruker](https://www.bruker.com/en/landingpages/bdal/customer-insight-enveda.html)
- "Chemotaxonomic map of labdane diterpenoids" (J. Nat. Prod. 2026): not found by search; UNVERIFIED that it exists or is by Enveda.
- MS2Prop, Enveda-180 dataset paper: not found within the call budget.

### Inferences
- If test molecules resemble EnvedaDark, database-retrieval ceilings will be low and the popularity prior will mislead on the "unpublished analog" subset. A prior that prefers NP-likeness and plant origin without rewarding fame is better matched to that subset than PubChem counts.

### Gaps
- No source on plant families or chemical-class mix of the test set.
- Labdane paper, MS2Prop and an Enveda-180 data card not located.

## Integration ideas (from the findings above, all untested)
1. Keep COCONUT as the NP pool; stop adding NP DBs for coverage. Optionally pull LOTUS organism links (after confirming its license) to build a "reported in Plantae" flag.
2. Replace or temper the PubChem popularity prior with a mix: popularity, RDKit NP-likeness, a plant-origin flag, and NPClassifier class agreement across spectra of the same `molecule_id`. Tune weights on a held-out set of NP-like training molecules, separately for "famous" and "obscure" bins.
3. Skip CCS and RT entirely as filters; at most, use predicted CCS as a shape feature in a learned reranker.
