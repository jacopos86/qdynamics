# Paper-I L=3 fixed-width noise rerun with Phase 0

- Problem: half-filled open (L=3) Hubbard chain, (U/t=1.25).
- Noise levels: `low`, `high`, and `extreme`, using the existing fixed seeds
  and noise tuples.
- Horizon: 20 accepted controller rounds.
- Phase 0: active absolute-gradient screen, cap 24.  Because the eligible
  pool contains 24 generators, the cap is nonrestrictive but the screen and
  its estimator acquisitions are executed.
- Phases I--III: fixed widths `24/12/1`; adaptive shortlisting is disabled.
- Cost: exact Qiskit compiled marginals in all scored phases, FakeMarrakesh,
  optimization level 1, transpiler seed 7, weights `0.30/0.30/0.25` for
  (N_{2q}/D_{2q}/D_c).
- Accepted refits: Powell with `xtol=1e-5`, `ftol=1e-12`, matched to the
  Paper-I L=3 comparator convention.
- Caches: candidate-record and Hubbard-pool caches disabled so noise draws
  are not skipped.
- Output: `output/local_runs/ra_adapt_l3_noise_fixed_phase0_20260824/`.

Each result records the application source, full protocol and scientific
receipts, the source commit, and a SHA-256 of the dirty source diff used for
the run.  Promotion into Paper I is a separate step after all three cells and
their receipts are audited.
