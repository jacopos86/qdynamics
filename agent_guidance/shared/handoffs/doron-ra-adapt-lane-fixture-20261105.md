# Doron Wu's model through the Paper IV RA-ADAPT lane (2026-11-05)

Branch `claude/paper-iv-fixture-from-builder-20261102` (the Paper IV campaign lineage, where the
lane launcher with its segment flags lives), on top of the fixture producer of 2026-11-03 and the
cherry-picked interop (`2b78d65a`) and vendoring (`f60a183a`) commits of the builder branch.  The
previous handoff, `doron-bridge-qdynamics-entry-20261104.md` (builder branch, main lineage), ended
with "RA-ADAPT is not in the entry point"; this one closes that gap.  Jake, 2026-11-05: "Yes, build it".

## What exists

- `src/quantum/chemistry/qdynamics_vendored.py::vibronic_model_from_qdynamics_outputs`: his
  driver's three HDF5 files -> our `VibronicQubitModel` with his numbers (full MO space,
  `C_I = sqrt(2 omega_I) g_I` from `eph_mat`, `omega` from the e-ph file, `norm_mode` of the
  vibration file mass-weighted with the deck's isotope masses, ascending frequency).  It checks the
  interleaved spin-orbital layout, the chemists' `(pq|rs)` Coulomb layout, the closed-shell
  determinant energy of the tensors against `e_scf` (1e-8), the e-ph/vibration frequency match
  (rtol 1e-6) and the mode normalization (1e-6), and refuses files written without
  `isotope_masses` (his default masses make the files inconsistent, see the vendored README).
- `src/quantum/chemistry/paper_iv_h2o_fixture_from_vibronic_model.py`: any `VibronicQubitModel`
  with tensor couplings -> the lane's `molecular_vibronic_h2o_linear_fd_fixture_v1` file.  Modes
  labelled bend / symmetric_stretch / antisymmetric_stretch by internal-coordinate character and
  oriented by the lane policy (the oriented bend is the builder's negated; the coupling tensors
  carry the sign); derivative records `dH/dQ_mu` = the coupling tensors, `derivative_source`
  analytic, no displaced tensors, `active_equilibrium_force` = <HF|dH/dQ_mu|HF>.  CLI:
  `--source builder` (our builder in his conventions, `--active-space full|frozen_core`,
  `--counter-term`) or `--source qdynamics-h5 --ele-h5 --vib-h5 --eph-h5`;
  `--compare-qdynamics-json` compares the fixture's Hamiltonian with a file in his schema term by
  term (modes matched by frequency, register signs chosen).  Writes `<tag>_fixture.json`,
  `<tag>_qdynamics.json` (the work model in his schema) and `<tag>_summary.json`.
- Generator: `build_h2o_linear_fd_fixture_from_components` split out of
  `build_h2o_linear_fd_fixture_from_record` (encode, reference state, exact sector reference,
  cutoff diagnostics, pool rows, manifest, evidence gate); the record path calls it and its
  output on the synthetic record is byte-identical before and after (checked with the
  `created_utc` field removed).
- Validators (`vibronic_h2o_linear_fd.py`): `H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE`
  admitted next to the finite-difference source; every derivative record must carry the
  manifest's source; an analytic record may not reference displaced tensors, must declare
  `derivative_method: analytic` and carries no finite-difference drift; finite-difference
  records still need their drift diagnostic.  `plus/minus_aligned_tensor_id` may be null.
- Pool locks (`ra_adapt/pools.py`): `h2o_lane_active_space(problem)` reads the resolved fixture;
  `require_h2o_symmetry_complete_problem` requires the active space the fixture declares (default
  CAS(8e,6o), 12 fermion qubits); the parent inventory is locked against
  `report_summary.derivative_resolved_pool_v2_parent_count` when the fixture records it (the
  producer counts the derivative-resolved pool at production time; the fixture bytes are what
  every checkpoint attests), else against the constant 448 for CAS(8e,6o), and a variant fixture
  without a recorded count fails closed.  `_parent_pool_spec` takes `num_sites` from the request.
- `test_support/qdynamics_h2o/`: his deck, the three HDF5 files and his qubitizer's JSON at
  cutoffs 1 and 3 (his code at 38f40805 run here on 2026-11-04; origin "simulated"), sha256-pinned.
- Tests: `test/test_paper_iv_h2o_fixture_from_vibronic_model.py` (7): mode labelling/orientation
  of scrambled real H2O modes; a synthetic two-orbital model through producer, gate, registry,
  application lock and parent-inventory lock (tampered count -> "drifted", missing count ->
  fail closed); counter-term model has no reference force; validator refusals; the frozen-core
  model in his conventions (15 qubits, frequencies 2170.79/4138.69/4389.24, pool count below 448
  and locked); his HDF5 files -> fixture equal to his JSON; the pins.

GitNexus before editing: `validate_production_vibronic_h2o_fixture` CRITICAL (24 upstream, 5
processes), `validate_paper_iv_h2o_linear_fd_evidence_fixture` CRITICAL (34, 5),
`require_h2o_symmetry_complete_problem` CRITICAL (287, 87), the others LOW.  The changes are
additive and guarded: the finite-difference path and the CAS(8e,6o) 448 lock keep their exact
behaviour.  Evidence: see the test section below.  `detect-changes --scope all` after the edits:
5 files, 26 symbols, 0 affected processes, risk low.

## Fixtures produced (`work/doron_interop_20261104/lane_fixtures/`)

H2O/STO-3G at the Paper IV geometry (the RHF/STO-3G optimum), 10 electrons in 7 orbitals, binary
registers; Hartree-Fock (x) vacuum energy -74.9415276 Ha for every operator (his table's "direct
classical ground state").  "his HDF5 files" = his driver run here with `isotope_masses` set.

| fixture | source | cutoffs / qubits | sector dim | exact ground state (Ha) | at the reference cutoffs (Ha) | cutoff drift (Ha) | boundary weight | derivative-resolved parents | fixture Hamiltonian vs his JSON |
|---|---|---|---|---|---|---|---|---|---|
| `his_h5_c111` | his HDF5 files (`test_support/qdynamics_h2o`) | 1,1,1 / 17 | 3528 | -75.1190471 | -75.2011488 (2,2,2) | 0.0821 | 0.69 | 584 | 1207/1207 words shared, none missing, max 2.1e-14 |
| `builder_his_conventions_c111` | our builder in his conventions | 1,1,1 / 17 | 3528 | -75.1190469 | -75.2011485 (2,2,2) | 0.0821 | 0.69 | 584 | 1207/1207 words shared, none missing, max 4.4e-06 |
| `builder_counter_term_c111` | our builder in his conventions, counter-term on | 1,1,1 / 17 | 3528 | -75.0052072 | -75.0090347 (2,2,2) | 0.0038 | 0.08 | 584 | not compared |
| `his_h5_c333` | his HDF5 files | 3,3,3 / 20 | 28224 | -75.2644600 | -75.3163222 (4,4,4) | 0.0519 | 0.34 | 584 | 1564/1564 words shared, none missing, max 2.8e-14 |

The fixture built from his files carries his operator exactly (2e-14 on 1207 words); our
builder in his conventions differs from it by 4e-6 on the coupling words (his Hessian keeps
translations and rotations) and 2e-7 Ha on the ground state.  Neither operator is converged
in the cutoff (drift 0.082 Ha, 69 % of the ground state on the highest level): the uncompensated
mean-field force displaces the oscillators.  <HF|dH/dQ|HF> per mode in lane orientation:
his file {"antisymmetric_stretch": 5.32575820802552e-16, "bend": 0.009645750378036779, "symmetric_stretch": -0.012730299044139323}; counter-term model
{"antisymmetric_stretch": 0.0, "bend": -0.0, "symmetric_stretch": 0.0}.

## RA-ADAPT runs (`work/doron_interop_20261104/ra_adapt_runs/`)

Lane launcher unchanged: `python -m pipelines.static_adapt.ra_adapt.h2o_application --fixture F
--output-dir D --maximum-controller-rounds 6 --candidate-representation macro_generator_v1
--insertion append_only --cost-mode proxy --checkpoint-history-tail 1 --max-wall-seconds 10800`,
`STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_NQ=17`, main `.venv`, this Mac (14 CPUs, 24 GiB), both runs at
once.  Route profile
`paper_iv_h2o_ra_adapt__macro_generator_v1__symmetry_complete_macro_v1__append_only__measured_residual_response_v1__all_phase_resource_weighting_v1__incremental_active_baseline__exact_guarded_full_response`,
exact-ED stop at 1.6 mHa from the fixture's exact energy.

| run (macro arm, append-only, proxy cost, 6 rounds max, 3 h wall) | controller rounds | operators | energy after the last round (Ha) | exact same-cutoff (Ha) | above exact | stop | wall clock |
|---|---|---|---|---|---|---|---|
| his file (fixture from his HDF5 files), 17 qubits | 6 | 6 | -75.0889771 | -75.1190471 | 30.1 mHa | `maximum_controller_rounds` | 1860 s |
| counter-term model, 17 qubits | 6 | 6 | -74.9750170 | -75.0052072 | 30.2 mHa | `maximum_controller_rounds` | 1949 s |

`his_h5_c111_macro`:

| round | selected generator (Phase-I trust-region score) | energy after refit (Ha) | change (mHa) | `max_grad` over the pool | optimizer evaluations |
|---|---|---|---|---|---|
| 1 | `coupled::bend::dH_dQ_one_body_factor[5]_times_p` | -75.0053413 | -63.81 | 1.861 | 20 |
| 2 | `boson::symmetric_stretch::p` | -75.0621390 | -56.80 | 2.204 | 21 |
| 3 | `el::uccsd_dbl(ab:2,9->6,13)` | -75.0759071 | -13.77 | 2.204 | 62 |
| 4 | `el::uccsd_dbl(ab:2,10->6,12)` | -75.0805638 | -4.66 | 1.435 | 31 |
| 5 | `el::uccsd_dbl(ab:3,10->5,12)` | -75.0844887 | -3.92 | 1.431 | 42 |
| 6 | `el::uccsd_dbl(ab:3,9->5,13)` | -75.0889771 | -4.49 | 1.336 | 51 |

`counter_term_c111_macro`:

| round | selected generator (Phase-I trust-region score) | energy after refit (Ha) | change (mHa) | `max_grad` over the pool | optimizer evaluations |
|---|---|---|---|---|---|
| 1 | `el::uccsd_dbl(ab:2,9->6,13)` | -74.9555935 | -14.07 | 1.861 | 31 |
| 2 | `el::uccsd_dbl(ab:3,10->5,12)` | -74.9599283 | -4.33 | 1.861 | 21 |
| 3 | `el::uccsd_dbl(ab:3,9->5,13)` | -74.9645158 | -4.59 | 1.856 | 31 |
| 4 | `el::uccsd_dbl(ab:2,10->6,12)` | -74.9692108 | -4.70 | 1.856 | 41 |
| 5 | `el::uccsd_dbl(ab:3,10->6,13)` | -74.9722487 | -3.04 | 1.850 | 63 |
| 6 | `el::uccsd_dbl(ab:2,9->5,12)` | -74.9750170 | -2.77 | 1.851 | 74 |

Reading: on his operator the lane's Phase-I selection takes the displacement generators first
(`dH/dQ_bend` one-body factor times P_bend, then P_symmetric_stretch: 121 of the 178 mHa between
HF (x) vacuum and the exact ground state in two rounds), then UCCSD doubles; the counter-term
model has no reference force, so it takes UCCSD doubles from round 1.  Both stop at the round
limit about 30 mHa above exact; the exact-ED stop at 1.6 mHa never fired (the CAS(8e,6o) bridge
run of 2026-10-20 needed 17 rounds for that).  On the same operator, six RA-ADAPT operators
(-75.0890 Ha) already lie below the entry point's QSE Ritz values from HF (x) vacuum (-75.0846 Ha
with 207 basis elements, -75.0859 Ha with 564; handoff of 2026-11-04): the P_mu and dH/dQ x P
generators of this pool are what the bridge's lifted-UCCSD bases lacked.  Every checkpoint's
`problem_attestation` carries the fixture sha256 and `fixture_derivative_source` =
`analytic_linear_coupling_tensors_mass_weighted_normal_modes_v1`.  Per round: 87-120 s of
gradient screening over 584 parents on 8 threads, 20-74 Powell evaluations; peak resident set
about 0.2 GB per run (two runs at once).

## Not done / untested

- RA-ADAPT at 20 qubits (cutoff 3) and the singleton (sector-complete Pauli-block) arm: not run.
- The fixtures above live in `work/` (untracked); only his HDF5 files and JSON are checked in.
- The coupling-model decision (his mean-field vertex vs ours; counter-term or not) is Jake's and
  Doron's; nothing here changes the Paper IV production model or campaign.
- Merged 2026-11-06: `claude/molecular-vibronic-builder-20261102` (the entry point, main's
  bridge, the Paper V work) was merged into `claude/paper-iv-fixture-from-builder-20261102`, so
  this branch holds the entry point, this producer and the Paper IV campaign lineage together and
  contains main; nothing pushed. The merge commit message carries the conflict resolution and
  the test evidence.
