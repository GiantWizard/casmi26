# Engine-1 replays on the Enveda natural-product panel (CPU)

250 timsTOF natural products (`enveda-np-examples` in train.parquet) run through the public engine 1 (v4b models,
FPNet A, which held the panel out) with the engine's own simulation switches, on CPU (~3 s per molecule, 4 cores).
`panel_dump.py` is dmitriigluzdov's `main_tables.py` (casmi26-natural-product-panel-benchmark) plus a per-candidate
feature dump and `GEN_CFG` / `TAG` overrides for the generation settings.

    python panel_dump.py V4B_DIR POOL_DIR train.parquet POP_DIR WORK OUT {lib|nospec|dbdrop}
    GEN_CFG='{"gen_n_analog": 20, "gen_min_sim": 0.2, "gen_max_per_parent": 200, "gen_max_total": 600}' TAG=_wide python panel_dump.py ...
    python insert.py OUT nospec dbdrop          # default list + wide-pass generated structures at their wide positions
    python merge_sim.py OUT/dbdrop_wide.pkl pubchem_candidates.parquet   # cell-17 merge, answer absent from PubChem

Scenarios: lib = answer's NP-example spectra hidden (class 1); nospec = all reference spectra hidden, structure in
pool (class 2); dbdrop = nospec + structure removed from the pool (only generation can find it; class-3-like, with
the caveat that famous NPs have many library relatives to derive from).
Diagnostics: permodel.py (per-FPNet top-1), allfeat.py / ablate.py (what the ranker relies on), prov*.py (provenance).
