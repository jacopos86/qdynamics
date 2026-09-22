# Merging the psi4/pyscf chemistry backend with RA — plan

Written 2026-09-22 after reading both trees. Scope: bring Doron's molecular
electronic-structure and electron--phonon solvers in as the front end that
produces RA-admissible problems, choose the cost path for large systems, and
define a production route without the comparison bookkeeping.

## What each side actually is

**Doron's work**: `jacopos86/qdynamics`, branch `doron-chemistry` (tip
2026-08-26 `tested-eph`, author `doronwu`; `doron-pyscf-archive` is the earlier
line). It is a standalone package: `src/chemistry/psi4/` and
`src/chemistry/pyscf/`, each with `electron_solver.py`, `vibration_solver.py`
and `electron_phonon_fd_solver.py`, plus a Psi4 C++ derivative plugin
(`eph_deriv_plugin/qdynamics_eph_deriv.cc`). Input is a plain `.inp` file
(method, basis, xc, charge, spin, unit, geometry block or `.xyz`, output
prefix). Output is HDF5 through `read_write_*_results.py`:

| Writer | Contents |
|---|---|
| `write_h5` | `mo_coeff`, `mo_energy`, `mo_occ` |
| `write_matrix_elements_h5` | `h1`, `h2`, `nuclear_repulsion` |
| `write_vibration_h5` | `freq_au`, `freq_wavenumber`, `norm_mode (nmodes, natm, 3)`, `reduced_mass`, `force_const_dyne` |
| `write_eph_h5` | `eph_mat`, `omega` |

FCIDUMP is an option. Translations and rotations are projected out, so the
mode set is already 3N−6 / 3N−5.

**Our side**: `src/quantum/chemistry/generate_h2o_linear_fd_fixture.py` reads a
*backend record* (`h2o_linear_fd_backend_record_v1`, validated at line 1480)
and emits the fixture that `pipelines.static_adapt.ra_adapt.h2o_application
--fixture` consumes. The fixture carries geometry, normal modes, displacements,
the active space with its one- and two-body integrals, aligned tensors and
alignment diagnostics, first derivatives, layout, the physical sector
(`n_alpha`, `n_beta`, `n_ph_max_by_mode`, mode labels), `encoded_operators`
(`h_electronic`, `h_vibronic`, `dH_dQ_by_mode`, `q/p/n_by_mode`), the reference
state, the exact reference and cutoff diagnostics.

**This is not a git merge.** The `RA` branch in `qdynamics` is a whole copy of
`Holstein_test`; `doron-chemistry` is the original qdynamics tree. Against their
merge base `c2b942f0` the two differ by 21651 files. The work is an interface,
not a merge commit.

## The seam

One adapter: **Doron's HDF5 → `h2o_linear_fd_backend_record_v1`**. Everything
downstream (orbital alignment across displacements, the central differences,
the boson encoding, mode cutoffs, the exact same-cutoff reference) is then
reused unchanged, which is what keeps the fixture's provenance meaningful.

Do **not** map his `eph_mat` straight into `encoded_operators`. That path skips
the alignment diagnostics and the cutoff policy the current fixture's
attestation rests on.

Two decisions inside the adapter:

1. **Who differentiates.** Our generator does its own linear finite differences
   from per-displacement integrals; his FD solver differentiates too. Feed our
   generator the displaced-geometry integrals and keep the derivative
   convention and alignment on our side. His `eph_mat` then becomes an
   independent cross-check rather than the input — which is exactly the
   comparison already reported in the advisor overview ("I also reproduce
   Doron's values at his geometry").
2. **Molecule identity.** The fixture hard-codes H2O family keys and mode
   labels. Generalising means parameterising those, and giving each new
   molecule its own named route plus fixture-sha attestation, rather than
   loosening the admission check.

## Where RA is pinned to Hubbard--Holstein

`pipelines/static_adapt/ra_adapt/pools.py:280,364` and `append.py:198` admit
only the Hubbard--Holstein `L=2` problem, and `AGENTS.md` says the same about
the typed `run_ra_adapt` facade. **Paper IV does not go through that facade** —
`h2o_application` is its own entry point, which is why H2O runs today. So the
pinning blocks the typed facade, not molecular work; a new molecule follows
`h2o_application`'s pattern.

## Cost: proxy versus qiskit

- Proxy: `pipelines/resource_estimation/pauli_graph_span.py`
  (`marrakesh_graph_span_v1`) — BFS distances over the coupling map and support
  spans per term; cost grows with terms times support.
- Qiskit: `pipelines/static_adapt/hh_backend_compile_oracle.py` —
  `transpile_single_v1` against FakeMarrakesh with a fixed `seed_transpiler`.
  The route profiles already restrict it to later phases
  (`phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1`,
  `phase0_proxy_or_off_...`), which is itself a statement about its price.

**There is no timing measurement in the repository.** What exists in
`MATH/Math.md` is compile *quality* data: two-qubit counts and depth across
`optimization_level` and `seed_transpiler`. So the comparison the plan needs
has to be produced, and the decisive number is not only seconds per candidate
but whether the proxy *ranks* candidates the way the transpiler does: if the
ranking agrees, large systems can select on the proxy and spot-check with a
transpile.

## Production route without the comparison bookkeeping

Already exposed: `--no-estimator-ledger` and `--checkpoint-history-tail`.
Measured on the running campaign at round 14: `estimator_call_ledger.json`
41.5 MB, its checkpoint sidecar 25.7 MB, `current.json` 57.9 MB. The dense
refit Gram that made `current.json` 2 GB is already gone (`620da53d`); what
remains grows linearly in the runtime term count.

The rest of the bookkeeping is route-profile shaped (beam archival knobs,
per-phase receipts, `qiskit_compile_cost_policy`). Express the lean setting as
a **named route profile**, not a pile of flags, so a production run still says
in one string what it was.

## Increments

1. **Cost benchmark.** Time both cost paths over one candidate set at H2O's 19
   qubits and at a larger support, and report seconds per candidate plus rank
   agreement between proxy and transpile. No engine change.
2. **Adapter** from Doron's HDF5 to `h2o_linear_fd_backend_record_v1`, with a
   parity test: regenerate the current H2O fixture from his outputs at the same
   geometry and require the fixture sha to match, or account for every
   difference.
3. **Generalise** the H2O-specific family keys, molecule name and mode labels;
   one named route and attestation per molecule.
4. **Lean route profile**, then one H2O comparison run against the present
   route to bound what the lean setting gives up.

## Unresolved

- The exact field list of `h2o_linear_fd_backend_record_v1` has to be read off
  the parser at `generate_h2o_linear_fd_fixture.py:1480` when increment 2 is
  written; this plan did not enumerate it.
- Whether Doron's active-space selection matches ours (ours is 8 electrons in 6
  orbitals for H2O) is unverified; the parity test in increment 2 is what
  settles it.
- Nothing here has been run, and no molecule beyond H2O has an admitted route.

Files to edit: none yet — increment 1 adds a benchmark under `pipelines/`,
increment 2 adds the adapter under `src/quantum/chemistry/`.
