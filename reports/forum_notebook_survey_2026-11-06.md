# CASMI26: forum and public-notebook survey, 2026-11-06

Sources read: all 149 forum threads (full text with replies, saved at `~/casmi26/forum/forum_all.md`), the scored list of all
1,100 public notebooks (885 with a public score, via the logged-in Code tab), 285 notebook sources (every notebook with votes,
a score in its title, or by a lineage author; `~/casmi26/nbs/src`), and comment threads of the six top notebooks.
Forum URLs: `https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<id>`.

## 1. Host rulings that matter

| Ruling | Thread |
|---|---|
| Newer releases of train.parquet's sources (GNPS, MassBank, MoNA) are allowed; CC BY-SA fine | 745228, 750684 |
| Non-commercial open data (CC BY-NC) allowed from 2026-10-17; NIST and anything trained on it banned | 750330, 744470 |
| Enveda-180 Zenodo release (816 structures absent from train) may be used; those structures "likely were filtered out in the train.parquet preparation" | 745148 |
| No molecule_id overlap between visible and hidden test; "low-observation test set ... don't interpret noise with causation" | 745715 |
| Metric: stereo removed, tautomer-canonical InChIKey14, first match counts (appending can never lower a row) | 746878, 748029 |
| Hidden test peaks were corrected by one electron mass on 10-07; all submissions rescored 10-14 | 746954, 749430 |

Consequence for us: the host's answer in 745148 says structures missing from train were filtered, not held out. That
undercuts the premise of our library-match overlay (232 MassSpecGym structures absent from train), which scored 0.474 vs
0.478 base.

## 2. Leaderboard noise (the most important number)

- Public split is about 125-132 molecules (31%); private about 275 (69%). One molecule moving rank 2 to 1 is 0.004; miss to rank 1 is about 0.008 (753525, 753511, 748524).
- GPU stages are not bit-reproducible: identical code has scored 0.437 and 0.440 (748029). Seen again this week: alexchilton's mass-prior notebook and popwenman's byte-identical copy score 0.480 and 0.481.
- Simulation (753511): with 10% of molecules changing, a zero-gain change shows +0.005 on public 34% of the time; best of 7 zero-gain probes shows +0.019 on public and 0 on private.
- Seed changes alone move a GBDT ranker by ±0.007 and class-2 MRR by about 0.02 (743254, 748524).

## 3. How much tuning works: measured on the board

guhongbin's blend-weight sweep on one base (scored list):
w_mid 0.50 → 0.473, 0.625 → 0.476, 0.70 → 0.477, 0.75 → 0.477, 0.775 → 0.476, 0.875 → 0.473, 1.00 → 0.473.

Neural ensemble balance (guhongbin v562/v563, bgunshui): 50/50 → 0.480; 75/25 → 0.473; 25/75 → 0.470; one scorer alone → 0.460-0.479.

Leader-rule knobs on a 0.477 base (busyaprime/parallax notebooks): library gate 0.90, two-spectra rule, margin .028→.015 all
+0.000; vote step and new margin -0.004.

guhongbin v573-v578 (MSI gate, NP promotion, symmetric outsiders, LOTUS np_map) on a 0.479 base: all 0.479.
Dozens of 0.480 forks with one changed setting: 0.468-0.480. Titles claiming 0.482-0.484 (rajrajak99 v2100-v2140): 0.470.

Conclusion: every knob in the current lineage is worth about ±1 molecule, which is inside the noise.

## 4. Add-ons that were measured, with board results

| Change | Board result | Source |
|---|---|---|
| PubChem channel (removing PubChem-only candidates) | 0.464 → 0.409 without it | flexonafft / gluzdov notebooks |
| Neural spectrum scorer on ranks 2-25, rank 1 protected | 0.465 → 0.477 in four steps | parallax / busyaprime notebooks |
| Second self-trained scorer, 50/50 z-blend | 0.479 → 0.480/0.481 | bgunshui, zz3t3rn41 V22 |
| Mass prior (ppm penalty) on V22 | 0.480-0.481 (no measurable change) | alexchilton, popwenman |
| Long-trained neural model instead of public one (offline MRR 0.48 → 0.72) | 0.472 → 0.453 | parallax notebook |
| NX isomer reranker (ahmedberatozer), weight 2 | 0.472 → 0.414; on V2 order 0.477 → 0.469 | parallax notebook |
| hosen42 isomer reranker a20 | 0.475 | scored list |
| Own retrained FPNet, 7 swaps (+26% offline Tanimoto) | every swap lost (0.418-0.443 vs 0.456) | spectral-parallax notebook |
| More analogue parents 6 → 20 | 0.464 → 0.456 | gluzdov notebook |
| Quad/tri-neural ensembles (easonyanyan v31-v43, rajrajak99) | 0.467-0.480 | scored list |
| extlib quad-neural (ozertuu) | 0.476 | scored list |
| heliosli: 0.468 notebook + private isomer reranker (rr3) + newer-library rank 1 | 0.482 (team private LB 0.486) | heliosli/casmi26-0-482 |
| Ours: V22 copy without second scorer | 0.478 | our submissions |
| Ours: + own scorer (wider) at 0.20/0.33 | 0.469/0.470 | our submissions |
| Ours: + MIST at 0.15/0.25 | 0.465/0.465 (base 0.469) | our submissions |
| Ours: + library match (MSG not in train) | 0.474 | our submissions |

Pattern: offline gains (isomer rerankers, better-trained models, MIST) do not transfer; the only add-ons that moved the board
were new candidate sources (PubChem) and the neural scorer with rank 1 protected. Forum reports agree: class-3 and isomer
work improves local validation but not the board (751300, 743254 item 6, 748029 thread).

## 5. Test composition estimates (unconfirmed)

- Bull's LB-fitted class shares: class 1 ~28%, class 2 ~20%, class 3 ~52% (748524). Speculation that private is richer in class 3 (755039), no data.
- Class 1 is cross-instrument: no public library has timsTOF natural-product spectra; true matches have median entropy similarity ~0.90 vs ~0.99 between public libraries (756504). Our calibration agrees (true-match cosine median 0.67).
- Ranked by PubChem popularity alone, the answer is #1 for 67% of class-1-like molecules but ~0% of class-2-like ones (748524 reply).

## 6. Working 0.481 recipe today

`popwenman/casmi26-0482-gold-apex-v22` (0.481, run 2026-11-06) = V22 + mass prior, second scorer from the public kernel
`popwenman/casmi26-scorer-train-v80` (replaces the deleted bgunshui kernel). alexchilton's byte-identical notebook scores
0.480. Our 0.478 copy was missing the second scorer entirely (bgunshui's kernel deleted; the code silently fell back to the
public scorer twice).
