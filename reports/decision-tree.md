# CASMI 2026 decision tree (2026-10-18)

Score = first-match MRR@25. Every lost point is a molecule whose answer is ranked 2-25 (partial credit) or not in the
top 25 at all (zero). Each leaf below is one way to win some of those back, with its status.

Status: **DONE** measured, keep / **OUT** measured, don't redo / **READY** built, needs a run or submission /
**OPEN** not tried / **BLOCKED** needs something we don't have.

Class shares are competitor estimates, not host-verified (D748029, D748524): class 1 ~16% (Aryan's cosine-only probe
0.150), class 3 ~52%, class 2 the rest. If they hold, classes 1-2 already average ~0.9 MRR inside our 0.438, and a
+0.03 gain can only come from class 3.

```
0.438 public LB (public best 0.443, leader 0.484)
|
+-- A. Measurement: we can't tell good ideas from bad offline
|   +-- A1 ONP sets A/B ................ OUT as a ranker (74% amino-acid conjugates, 98% Orbitrap)
|   +-- A2 LB-calibrated slice (non-conj in-pool ONP, ~100 mol) ... DONE, harm filter only, too small for +-0.003
|   +-- A3 NP engine harness + pre-registered pass bar (3,568 NP>1 structures) ... DONE (Phase 0), use for engine work
|   +-- A4 LB probes (subset's candidates, 'CCO' elsewhere) ......... OPEN, the only direct read of the hidden test
|   |       use: measure class shares, measure any generator's hits on hidden rows
|   +-- A5 bigger clean slice (rebuild with conjugate + in-pool filter) ... OPEN, ~1 evening, data-limited (few timsTOF NPs)
|
+-- B. Class 1: library match (~16%) ............ near-solved, no lever worth time
|
+-- C. Class 2: answer in pool, no reference spectrum
|   +-- C1 answer ranked 2-25 (isomer / same-formula confusion)
|   |   +-- better FP model as re-scorer ......... OUT (fold0, simA, fplong all negative on calibrated slice)
|   |   +-- better FP model as engine bank + retrained ranker ... READY (fplong.pt; rk/run_np1.sh + compare_banks.py)
|   |   +-- forward models (GLACIER, ICEBERG, MIST) as extra terms ... OUT
|   |   +-- MAGMa fragment explanation ........... OUT
|   |   +-- learned reranker (lr2) ............... OUT (-0.028 LB)
|   |   +-- engine-2 top-1 at rank 2 ............. DONE (public e2r2, +0.002)
|   |   +-- molecular-networking rerank .......... READY (kernels/jc-v7c-e2r2-net, offline +0.006..0.020)
|   |   +-- positional-isomer ranking ............ OPEN (reach 13-33/302 of ONP class-3 misses, ranking unexplored)
|   +-- C2 answer beyond rank 25
|   |   +-- popularity weight x0.5 / x2 .......... READY (kernels/jc-v7c-popx0p5, -popx2)
|   |   +-- add an independent engine's candidates ... OPEN (only change that moved the 0.437 lineage, D749251)
|   +-- C3 wrong formula / adduct upstream ....... OPEN (not audited: how often is the answer's formula outside our windows?)
|
+-- D. Class 3: answer not in PubChem (~52%, all public pipelines score ~0)
|   +-- D1 detect class-3 rows ................... DONE: top 10% flagged = 82-100% true misses (bench/class3/detect.py)
|   |   +-- add engine match scores as features .. BLOCKED (need a Kaggle GPU eval run that saves them)
|   +-- D2 generate the exact structure
|   |   +-- FRIGID (MIST fp + DLM) ............... OUT (0/14 top-1 never-trained, deployment fp regenerates the pool)
|   |   +-- MS-GPT (MIST posterior band + SAFE-GPT) ... OPEN, best untested shot; needs host ruling + GPU
|   |   +-- end-to-end generators (DiffMS, test-time tuning, flow) ... OUT on paper (2-3% top-1 MassSpecGym)
|   |   +-- one-edit analogs of linked test neighbours (netgen) ... OUT (gain was all conjugates; LB 0.436)
|   |   +-- one-edit analogs of POOL molecules at mass M - delta ... OPEN, cheap reach test first:
|   |   |       is answer = +-O / +-hexose / +-CH3 / ... of any pool structure? (edits.py, pool_meta.parquet)
|   |   +-- biotransformation-expanded pool (offline, BioTransformer-style) ... OPEN, only if the reach test is high
|   +-- D3 rank generated vs pool candidates ..... depends on D2 producing hits
|
+-- E. Protect what we have
    +-- E1 compliance: weights from other competitors' datasets may be removable (D745841) ... OPEN, retrain own FPNet/ranker
    +-- E2 final picks: public 0.443 + best READY variant ... after the GPU-quota reset
```

## Order of work

1. Cheap and decisive first: D2 pool-edit reach test (local CPU, hours), fplong engine-level test (C1, local),
   A4 probes to pin down class shares (needs submissions).
2. If the reach test is high (say >= 20% of class-3 answers one edit from a pool molecule): build the expansion and rank
   it with the class-2 machinery; detector D1 decides which rows get it.
3. If MS-GPT is allowed: run it on the never-trained FRIGID set (`frigid/msgx3/`, 14 molecules in the test mass range).
   Kill rule as for FRIGID: >= 3/14 top-1 or stop.
4. In parallel, low effort: submit e2r2, e2r2-net, popularity variants; start E1 before the last two weeks.

Kill rule everywhere: a branch is closed only by a measurement on the right test (A2 for pipeline changes, A3 for
engine changes, A4 for anything the offline sets can't represent), never by a guess.

Sources: MS-GPT https://arxiv.org/abs/2607.23607 (code https://github.com/VIKI623/MS-GPT); MassSpecGym audit
https://arxiv.org/abs/2606.19624; test-time tuning https://arxiv.org/abs/2510.23746; forum threads per
`research_notes/.../forum_scrape_2026-10-14.md` in the vault.
