# matrix-reference-speedup (2026-09-15)

Overnight loop, round 2.  Plan step 8's remainder: round 1 left 47 percent of
the profiled 31-coordinate joint-Gram cone route inside
`paper_5/src/paper5/stability/matrix_reference.py`
(`agent_guidance/paper-v/overnight/cone-correction-speedup-20260914.md`,
"Negative results first", item 3).  Branch `claude/paper-v-routes-20260914`,
worktree `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`.
Not committed; the parent session commits.

Files owned and changed: `paper_5/src/paper5/stability/matrix_reference.py`,
`paper_5/tests/test_matrix_scalar_parity.py`, and the new
`paper_5/tests/test_matrix_reference_moment_assembly.py`.  `cone_correction.py`
was not touched.  No option, flag, profile or default was added or changed:
every change is a bitwise-neutral rewrite, and the acceptance evidence is
`np.array_equal` against the pre-change build, not a tolerance.

## Result

The joint-Gram energy-neutral barrier over `t=0` to `0.05` at strong coupling
falls from `0.92371 s` to `0.59877 s` of integration wall time, median of seven
repetitions in one process, **35.2 percent lower**, with every stored array of
the pinned run byte-identical.  The profiled cost of
`electron_phonon_moment_derivative` over that same interval falls from
`0.714 s` of a `1.559 s` route (45.8 percent) to `0.145 s` of a `0.975 s` route
(14.9 percent).

## What changed

Line numbers are post-change.  `ComplexArray`, `MatrixDimerState` and the
Pauli basis `_ELECTRON_FLUCTUATION_BASIS` are the record's own names; the
tables below carry no new quantity.

| change | file:line | what it removes |
|---|---|---|
| the three Pauli operators and their nine pairwise products are tabulated once at import as the diagonal index and unit coefficient of each monomial column (`_monomial_diagonal_terms`, `_PAULI_TRACE_ROWS`, `_PAULI_TRACE_COEFFICIENTS`, `_PAULI_PRODUCT_TRACE_ROWS`, `_PAULI_PRODUCT_TRACE_COEFFICIENTS`), and `trace(M @ sigma_a)` and `trace(M @ sigma_a @ sigma_b)` are read off them (`_pauli_traces`, `_pauli_product_traces`) | `matrix_reference.py:1076-1146`, `1177-1186` | the 15 matrix products and 15 `np.trace` calls per `_electron_fluctuation_moment_derivative` call and the Python double loop over the basis |
| `_electron_fluctuation_moment_matrix` and `_electron_fluctuation_moment_derivative` assemble the 3-by-3 block as one array expression | `matrix_reference.py:1188-1217` | 9 element assignments into `np.empty((3, 3))` per call |
| `_correlation_pauli_block` contracts both correlation matrices against the basis in one expression | `matrix_reference.py:1219-1228` | 6 matrix products and 6 `np.trace` calls per call |
| the Pauli means are cached on the exact bytes of their operand (`_pauli_trace_means`, cache bound 8) | `matrix_reference.py:1148-1174` | 31 of the 32 recomputations of the state's means per structured correction, and the repeat evaluations for the 28 velocity directions whose electronic block is zero |
| `boson_moment_matrix`, `_boson_moment_derivative_from_blocks`, `electron_phonon_moment_matrix` and `electron_phonon_moment_derivative` assemble by slice assignment into a preallocated array instead of `np.block`, and the boson block of the derivative is written straight into the 7-by-7 array (`_fill_boson_moment_derivative`) | `matrix_reference.py:1043-1066`, `1230-1256`, `1258-1290`, `1292-1318` | `np.block`'s recursive depth and shape check, the `np.vstack` of the cross block, and one 4-by-4 temporary per derivative call |
| the field-name/shape pairing is zipped once at import, and a field that already carries its declared shape skips `np.asarray` | `matrix_reference.py:79-81`, `96-106` | one `zip` and five `np.asarray` calls per validation |
| `np.eye(2)` in `boson_moment_matrix` becomes the module constant `_TWO_BY_TWO_IDENTITY` | `matrix_reference.py:82`, `1065` | one 2-by-2 identity allocation per call |

### Why the trace rewrite is bitwise-neutral and not merely close

Each Pauli operator, and each product of two of them, is monomial: exactly one
nonzero entry per column, of modulus one and drawn from `{1, -1, i, -i}`.  For
a monomial `M`, `(A @ M)[i, i]` is a sum of two terms of which one is an exact
zero, so it equals the single product `A[i, k] * M[k, i]` with no rounding, and
multiplication by a unit entry of that set is exact in IEEE complex
arithmetic.  `numpy.trace` then adds the two diagonal entries in ascending
index order, which is the order `_pauli_traces` and `_pauli_product_traces`
reproduce.  The `np.block` replacements are pure copies.  The mean cache is
keyed on `tobytes`, and equal bytes are equal floating-point values.

Before touching the file this was checked numerically: 200,000 random draws
each for `_electron_fluctuation_moment_matrix`,
`_electron_fluctuation_moment_derivative` and `_correlation_pauli_block`, and
20,000 draws for the associativity of `trace(D @ sigma_a @ sigma_b)`, gave
0 departures under `np.array_equal`.  The same comparisons are now the new
test file.

## Impact analysis (read-only, from the main checkout's index)

`node /Users/lordskyler/Holstein_test/.gitnexus/run.cjs impact "<name>" --direction upstream --repo /Users/lordskyler/Holstein_test`.
The index is 8 commits behind this worktree and says so; no `UNKNOWN` verdict
was returned, so nothing needed a grep confirmation, but the private helpers
were confirmed by `grep -rn` to have no user outside `matrix_reference.py`.

| symbol | risk | impacted | direct callers |
|---|---|---:|---|
| `_validate_matrix_state` | **HIGH** | 155 | 14, all inside `matrix_reference.py` |
| `boson_moment_matrix` | **HIGH** | 70 | 11, including `cone_correction.structured_boson_barrier_correction`, `structured_joint_barrier_correction`, `initial_conditions.closed_boson_moment_eigenvalues` |
| `electron_phonon_moment_matrix` | **HIGH** | 62 | 16, including `cone_correction.structured_electron_phonon_barrier_correction`, `apcm_carried_witness_analysis.main`, `adaptive_positive_moment_propagation.integrate_apcm_ssprk3`, `closure_identifiability.closure_identifiability_witness` |
| `_correlation_pauli_block` | **HIGH** | 48 | `electron_phonon_moment_matrix`, `electron_phonon_moment_derivative` |
| `_electron_fluctuation_moment_matrix` | **HIGH** | 44 | `electron_phonon_moment_matrix` |
| `_boson_moment_derivative_from_blocks` | LOW | 13 | `boson_boundary_flux_decomposition`, `boson_moment_derivative_terms`, `correlation_fluxes`, `electron_phonon_moment_derivative` |
| `electron_phonon_moment_derivative` | LOW | 13 | `cone_correction.structured_electron_phonon_moment_velocity_lift` and two others; one execution flow, `archive_gram_admission_signals` |
| `_electron_fluctuation_moment_derivative` | LOW | 7 | `electron_phonon_moment_derivative` |

**HIGH-risk warning, as required before editing:** five of the eight symbols
returned HIGH.  They were edited anyway because the entire change is an
identity on the floating-point path, and that is what the four gates below
test: the two structured-barrier controllers that consume `boson_moment_matrix`
and `electron_phonon_moment_matrix` reproduce their pinned runs byte for byte,
and the carried-witness driver that consumes `electron_phonon_moment_matrix`
reproduces its recorded trajectory arrays byte for byte.

## Gates

Artifacts under `output/local_runs/paper_v_matrixref_20260915_v1/`
(`matrix_reference_before.py` and `matrix_reference_after.py` are the two
builds compared; `gate1_before`, `gate1_after`, `gate2_before`, `gate2_after`,
`gate3_after`, `profile_before`, `profile_after`, `time_before.json`,
`time_after.json`).

### Gate 1: ten-coordinate energy-neutral matrix barrier, t = 0 to 2

`structured_boson_barrier_correction` with `energy_neutral=True` through
`closed_cone_projected_rhs`, `lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude
1, exact ground-state contraction at phonon cutoff 16, `activation_margin =
1e-5`, `target_flux = 0`, `barrier_rate = 5`, DOP853 rtol 1e-9, atol 1e-11,
maximum step 0.05, sampled every 0.05.  Harness
`output/local_runs/paper_v_cone_ref_t2_20260914_v1/run_pinned_energy_neutral_barrier.py`.

| quantity | before | after |
|---|---:|---:|
| sampled times, states, bosonic minima, electronic minima, energies, initial state | \- | all six arrays `np.array_equal` |
| right-hand-side evaluations | 749 | 749 |
| half-space projections (and the whole `SOLVER_COUNTERS` census) | 484 | 484, census identical |
| minimum bosonic moment eigenvalue over t = 0 to 2 | 2.893142825017192e-05 | 2.893142825017192e-05 |
| integration wall seconds | 0.22629 | 0.22609 |

The minimum bosonic moment eigenvalue is the working notes' pinned value
2.893142825017192e-05
(`MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md:1859`).

### Gate 2: 31-coordinate joint-Gram energy-neutral barrier, t = 0 to 0.05

`structured_electron_phonon_barrier_correction` through
`closed_electron_phonon_cone_projected_rhs`, same physical settings plus
`preserve_correlation_trace=True`, `cone_tolerance = 1e-8`,
`projection_tolerance = 1e-12`, euclidean correction metric,
`subtract_initial_residual=False`, sampled every 0.0025.

| quantity | before | after |
|---|---:|---:|
| sampled times, states, bosonic minima, electronic minima, energies, initial state | \- | all six arrays `np.array_equal` |
| right-hand-side evaluations | 224 | 224 |
| half-space projections (and the whole census) | 1,843 | 1,843, census identical |
| minimum bosonic moment eigenvalue over t = 0 to 0.05 | 2.4870900303620437e-05 | 2.4870900303620437e-05 |
| integration wall seconds, median of 7 repetitions in one process | 0.92371 | 0.59877 |
| integration wall seconds, minimum of 7 | 0.90714 | 0.59643 |
| sum of absolute sampled state coordinates, as a cross-build checksum | 148.50903290514307 | 148.50903290514307 |

Harness `output/local_runs/paper_v_matrixref_20260915_v1/time_gram_t005.py`,
which rebuilds the right-hand side for every repetition.  Three other heavy
Paper V runs were on the machine throughout, so the wall times are comparable
to each other and not to a quiet machine.

### Gate 2, cProfile split

`output/local_runs/paper_v_matrixref_20260915_v1/profile_gram_t005.py`, same
right-hand side, integrator and settings, `t = 0` to `0.05`.

| quantity | before | after |
|---|---:|---:|
| total profiled seconds | 1.559 | 0.975 |
| function calls | 4,437,442 | 2,083,870 |
| `electron_phonon_moment_derivative`, cumulative seconds over 7,168 calls | 0.714 | 0.145 |
| the same as a fraction of the profiled route | 45.8 percent | 14.9 percent |
| own time in `matrix_reference.py` | 0.315 | 0.160 |
| the same as a fraction of the profiled route | 20.2 percent | 16.5 percent |
| `ndarray.trace` calls | 155,008 | 0 |
| `numpy.block` invocations (`_block` calls) | 14,784 | 0 |
| `numpy.vstack` calls | 9,235 | 1,843 |
| `_electron_fluctuation_moment_derivative`, cumulative seconds over 7,168 calls | 0.305 | 0.060 |
| `_correlation_pauli_block`, cumulative seconds over 7,392 calls | 0.098 | 0.024 |
| the boson block of the moment derivative, cumulative seconds over 7,168 calls | 0.123, through `_boson_moment_derivative_from_blocks` | 0.008, through `_fill_boson_moment_derivative` writing into the 7-by-7 array; `_boson_moment_derivative_from_blocks` is no longer called from this route and keeps its eleven other call sites in `matrix_reference.py` |
| `_validate_matrix_state`, cumulative seconds over 15,232 calls | 0.027 | 0.018 |

The 1,843 remaining `np.vstack` calls are in `cone_correction.py`, not in
`matrix_reference.py`.  What is left of the route is
`structured_electron_phonon_barrier_correction` itself (0.119 s own time),
`np.linalg.eigh` (6,201 calls, 0.067 s) and SLSQP (1,843 solves, 0.243 s
cumulative) — all outside the files this task owns.

### Gate 3: no-guard carried-witness anchor, t = 0 to 0.05, h = 0.0025

The driver does consume `matrix_reference`:
`apcm_carried_witness_analysis.py:24-26` imports `electron_phonon_moment_matrix`
(used at `:404` for the retained joint-Gram eigenvalue series) and
`apcm_carried_witness.py:57` imports `matrix_dimer_rhs`, so the gate was run
rather than skipped.  `python -m paper5.stability.apcm_carried_witness_analysis
--final-time 0.05 --time-step 0.0025 --no-gram-guard`, strict numerical
profile, canonical cutoff-16 correlated preparation, compared against
`output/local_runs/paper_v_cwrmf_final_noguard_t005_h0025_20260914_v1`.

Every array of `autonomous_trajectory.npz` and of `trajectory.npz` is
`np.array_equal` to the recorded run, with two exceptions: `velocity_margins`
carries `NaN` entries and is equal under `np.array_equal(..., equal_nan=True)`,
and `maximum_atom_seconds` is a per-atom wall-clock series and differs as a
timing measurement must.  The `accuracy` and `feasibility` blocks of
`summary.json` compare equal as whole dictionaries; in particular the
all-coordinate scalar RMS error against the exact cutoff-16 reference is
5.836875347948307e-05 and its dynamic form 3.235916676961365e-06 in both runs,
and the minimum carried unshifted 62-row Gram eigenvalue is
-8.290500630490342e-05 in both.  Autonomous solver wall time 0.6314 s recorded
against 0.6828 s here, which is not a usable comparison: the recorded run was
made with fewer competing processes, and the unguarded velocity tier is
round 1's measurement, not this change's.

### Gate 4: tests

`cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider`:
**376 passed** in 154.13 s (326 at the end of round 1; the other 50 are this
round's additions across three agents editing the same worktree, 11 of them
mine).  The named gate files
`tests/test_matrix_scalar_parity.py tests/test_cone_correction.py
tests/test_exact_driven_reference.py`: **50 passed**; with the new file added,
**61 passed**.

Added: `paper_5/tests/test_matrix_reference_moment_assembly.py`, 10 tests,
which transcribes the pre-change assembly verbatim as `recorded_*` and
requires `np.array_equal` on 2,000 random draws per identity (500 for the
four block assemblies), checks that `_monomial_diagonal_terms` rejects a
non-monomial operator, that the mean cache returns the uncached values, is
read-only and stays inside its bound, and that `_validate_matrix_state` still
raises its declared message and still accepts a field given as nested lists.
`paper_5/tests/test_matrix_scalar_parity.py` gains one test (10 to 11) pinning
the same identity at the operating point the barrier runs actually use: the
strong-coupling exact cutoff-16 ground state, its closed-state velocity, and
the 31 closed-state unit directions.  That test transcribes the recorded
`np.block`/`np.trace` assembly inside itself rather than importing it from the
sibling test module, so it passes both from `paper_5` and from the repository
root; an earlier cross-module import failed with `ModuleNotFoundError: No
module named 'tests'` when pytest was invoked from the repository root.

## Kept

Every item in the "What changed" table.  All of it is on the default path,
because none of it is a behaviour: the four gates are byte-identity against
the pre-change build, and there is no new option, flag or profile to leave off.

## Dropped

- **Memoizing `local_holstein_couplings` on the `DimerParameters` object.**
  The function returns a fresh writable `(2, 2, 2)` array to eight call sites,
  three of them outside `matrix_reference.py`
  (`apcm_carried_witness_stitch.py:150`, `cone_long_horizon.py:181`), so a
  shared cached array would alias.  Measured share of the gate-2 route: 224
  calls, below the profile's 60-row cut, against 7,168 calls of the moment
  derivative.  Not worth the aliasing surface.
- **Folding the two outer products of the electronic means into one.**
  The recorded expression is `(trace - md_a * m_b) - m_a * md_b`; summing the
  two corrections first and subtracting once is a different accumulation order
  and is not bitwise-neutral.
- **Replacing the remaining `np.trace` calls in `matrix_total_energy`
  (`:775-777`), `matrix_state_to_closed_scalar_coordinates` (`:869`) and the
  three embedding-residual diagnostics (`:964`, `:995`, `:1029`, `:1037`).**
  Each is called once per right-hand-side evaluation or once per sampled time,
  not 32 times per correction; none appears in the profile's top rows.
- **Building the Pauli means from Python scalars instead of a fancy-index
  gather.** Measured at 10.06 microseconds per `_electron_fluctuation_moment_derivative`
  call before the mean cache and 7.62 after; the scalar form was not pursued
  once the cache had taken the repeat evaluations, and it trades a provable
  argument about `numpy.trace`'s summation order for one about Python complex
  arithmetic.

## Blockers

None.
