# Architecture receipt: the electronic one-body density matrix as a Dykstra set (Claude, 2026-10-10)

**What this is.** The written, corrected architecture receipt that the Codex
coordinator's dispatch addendum (`classical-dispatch-20261010-007`,
`/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/classical-dispatch-addendum-20261010.md`,
SHA-256 `dfb55b9c8061e767b9dff40517ebf13c40ae59a355de25966331738f85ad5c77`,
section "Electronic cone lane") requires before the electronic cone lane's
increment 3 may run. It settles the addendum's five points from the code and
gives the corrected increment 3. **It supersedes increment 3 of the handoff by
reference**; the handoff file is not edited.

| item | value |
|---|---|
| handoff amended | `agent_guidance/paper-v/handoffs/codex-paper-v-electronic-cone-projection-20261010.md` (identical copy in `prompt-exports/`), committed at `6b27596e` |
| SHA-256 of that handoff snapshot | `73128739ec2101d684d6d6c8741b037fd1ba1fdbf3ad090ccb4c9cc2084a8fb5` (the tracked file, the `prompt-exports/` copy and the blob at `6b27596e` all hash to this) |
| module read | `paper_5/src/paper5/stability/cheap_projection_route.py`, SHA-256 `25c4b0e3c9f2a070e4009c7d15a8c094a064c4dc6eab987fe03a94fb3385304d`; last changed at `3f51146a`, byte-identical at `b9f99271` and at this branch's head |
| tests read | `paper_5/tests/test_cheap_projection_route.py`, SHA-256 `b6038250ef0b3b226d66ee5c75da84b98cd60a46eacf89d8b14ee791a6412a49` |
| helper read | `paper_5/src/paper5/stability/adaptive_positive_moment.py` (`raw_moment_coordinates_to_matrix_state`, `uncentered_joint_moment_matrix`) |
| status | not run, not committed; exploratory architecture, not evidence |

Line numbers below are of those files at `b9f99271`.

---

## 0. The code this receipt rests on

**The retained joint Gram and its electronic rows.**
`uncentered_joint_moment_matrix` (`adaptive_positive_moment.py:458-493`)
builds the 8-by-8 Gram in the row order
`(1, b_0, b_1, b_0^dagger, b_1^dagger, sigma_x, sigma_y, sigma_z)`: the means
row is `[coherent, conj(coherent), bloch]` with `bloch = values[:3]`
(`:463`, `:483`), the electronic block in rows 5 to 7 is
`Tr(rho sigma_a sigma_b)` (`:480`), and the result is symmetrized (`:493`).
The density matrix is `rho = (I + sum_a bloch_a sigma_a) / 2`
(`raw_moment_coordinates_to_matrix_state`, `:257`, `:267`), Pauli order
`x, y, z` (`_PAULI_LABELS`, `:71`). So the principal 4-by-4 block on rows
`S = (0, 5, 6, 7)` is

```text
G_e(r) = [[1, r^T], [r, I_3 + i [epsilon_abc r_c]]],    r = bloch
```

the Gram of `(1, sigma_x, sigma_y, sigma_z)` in `rho`.

**The scaling.** The cheap repair works on the scaled retained matrix
`D^-1 G D^-1` with `D = diag(1, s, s, s, s, 1, 1, 1)`,
`s = sqrt(phonon_envelope + 1)` (`base_scales`,
`cheap_projection_route.py:773-786`, repeated in `_verify_affinity`,
`:894-907`; `phonon_envelope = 16.0` by default,
`apcm_positive_extension.py:56`). The rows of `S` carry scale exactly `1.0`.

**The features.** `_hermitian_features` (`:563-575`) is the
Frobenius-isometric realification: the `n` diagonal entries, then
`sqrt(2) Re` and `sqrt(2) Im` of the upper triangle in `np.triu_indices(n, 1)`
order. `_target_features` (`:1014-1031`) concatenates the extension block
(`extension_rows**2` reals; `1296` at 27 entrance keys) and then the retained
block (`64` reals) for `target="both"`.

**The cones.** `_cone_projection` (`:1033-1050`) clips each block to its
floor with `_clip_to_floor` (`:593-602`), which returns the symmetrized input
`0.5 (M + M^H)` even when no eigenvalue is below the floor (`:598-599`), and
re-realifies it; so the existing cone step is not a bitwise pass-through even
when inactive. Floors: `block_floors` (`:396-404`), `applied_retained_floor`
(`:387-394`); all `0` in the task-11 configuration.

**The affine set and the map back.** `factorization` (`:965-1012`): the
design matrix of the target features in the movable coordinates,
`pseudo_inverse = pinv(design * scales)` (`:1006`) and an orthonormal
`range_basis` of the design's column space at relative rank cut `1e-10`
(`:1008-1009`). The affine Gram set is `features + range(range_basis)`
through the stage input's own features.

**The alternation** (`_dykstra`, `:1315-1413`). With `x_0` the stage input's
features, `p_0 = q_0 = 0`, for `k = 0, ..., K-1` (`K = dykstra_iterations`):

| line | quantity | name used below |
|---|---|---|
| `:1386` | `shifted = point + cone_correction` | `y_k = x_k + p_k` |
| `:1387` | `projected = _cone_projection(shifted)` | `c_k = P_C(y_k)` (product of the cones) |
| `:1388` | `cone_correction = shifted - projected` | `p_{k+1} = y_k - c_k` |
| `:1389` | `shifted = projected + affine_correction` | `z_k = c_k + q_k` |
| `:1390-1403` | affine projection, with the Eq. (22) hyperplane met in closed form (`:1392-1402`) | `a_k = P_{A cap H}(z_k)` |
| `:1404` | `affine_correction = shifted - affine_point` | `q_{k+1} = z_k - a_k` |
| `:1405` | `point = affine_point` | `x_{k+1} = a_k` |
| `:1406-1412` | if every block minimum of `a_k` is at or above its floor, `return (a_k, k + 1)` | early exit |
| `:1413` | `return (a_{K-1}, K)` | return at the cap, floors met or not |

**The return policy.** The alternation always returns an affine iterate
`a_k`, never a cone output. `repair` (`:1166-1313`) skips the repair when the
stage input already meets every floor and the energy tolerance
(`:1233-1242`), otherwise maps the returned point back,
`residual = corrected - features` (`:1272`),
`increment[movable] = scales * (pseudo_inverse @ residual)` (`:1274`), and
reports `extension_minimum_after`, `retained_minimum_after` (`:1280-1289`) and
`target_residual = ||achieved - corrected||` (`:1303`), where `achieved` is
the target features of the mapped-back coordinates (`:1277`). There is no
convergence flag, and the returned iteration count is ambiguous: `K` is
returned both when the floors are first met at the last iteration and when
they are never met. The integrator records exactly these fields per stage
(`:2030-2049`).

---

## 1. Point 1: the identity is algebraic

For any real Bloch vector `r` (physical or not), `G_e(r)` is twice the matrix
of the map `Y -> Y rho` on 2-by-2 matrices in the Hilbert--Schmidt basis
`{I, sigma}/sqrt(2)`, because `Tr(rho A^dagger B) = <A, B rho>_HS`. Its
eigenvalues are therefore `2 lambda_i(rho)`, each twice:
`1 - |r|` and `1 + |r|`. Hence, exactly,

```text
lambda_min(G_e) = 2 lambda_min(rho) = 1 - |r|.
```

This holds for every coordinate vector the code can produce, so it is an
identity of the conversion, not a property of the dynamics. **An identity
error above `1e-10` is a stop**, and its diagnosis is one of: the conversion
(31 closed scalar coordinates to raw coordinates through the shared
correlation trace, as `_retained_gram_minimum` in `run_unrepaired.py` does),
the row ordering (`S` must be `(0, 5, 6, 7)`), the scaling (`D` must be `1`
on `S`), the realification index map, or a source change. It is **never**
evidence that the electronic cone is not implied by the retained cone, and
never a reason for the projection. The handoff's increment 1 stop clause
(snapshot lines 358 to 360, "an identity failure larger than `1e-10` is not a
stop ... strengthens the case for the option") is withdrawn by this receipt.

Expected size of the identity error: roundoff. A spot check by the receipt's
author on 600 random raw coordinate vectors (normal entries at scales `0.3`,
`1.0` and `2.0`, so Bloch norms well above 1 included) gave a largest
`|lambda_min(G_e) - 2 lambda_min(rho)|` of `3.6e-15`; this is a check of the
derivation, not a measurement on the stored chains, which increment 1 makes.

---

## 2. Point 2: positive stage inputs do not make the added set inert

**Where an added set `E` can act.** Let
`E = { features : the S-block of the scaled retained matrix >= 2 epsilon I }`
(point 3 for the factor 2). Every cone output `c_k` lies in the retained cone,
so with `2 epsilon <= f_r` (the retained floor; `0 = 0` in the task-11
configuration) every `c_k` is in `E`: a projection placed right after the cone
step (`:1387`) is inert by construction. The only iterates at which `E` can
act are the **affine iterates `a_k`** (`:1403`), because the affine projection
moves the `S`-block away from the cone output. A stage input `x_0` inside `E`
says nothing about `a_0, a_1, ...`.

**The placement, and every input the set touches.** `E` is inserted after
the affine step and its early-exit check, before the next cone step:

```text
y_k = x_k + p_k ;  c_k = P_C(y_k) ;  p_{k+1} = y_k - c_k
z_k = c_k + q_k ;  a_k = P_{A cap H}(z_k) ;  q_{k+1} = z_k - a_k
exit check on a_k (all block floors, and the S-block against 2 epsilon)  -> return (a_k, k + 1)
if k < K - 1:  w_k = a_k + e_k ;  d_k = P_E(w_k) ;  e_{k+1} = w_k - d_k ;  x_{k+1} = d_k
at the cap: return (a_{K-1}, K)
```

with `e_0 = 0`. The added set reads `a_k` and its own correction memory
`e_k`; it writes `e_{k+1}` and `x_{k+1}`, which is the input of the next cone
step, `y_{k+1} = x_{k+1} + p_{k+1}`, and therefore changes `c_{k+1}`,
`p_{k+2}`, `z_{k+1}`, `a_{k+1}`, `q_{k+2}` and every later iterate, the
early-exit iteration and the returned point. In `repair` it also enters the
pre-check that decides whether a repair runs at all (`:1233-1242`).

**When a bitwise prediction is justified.** If at every `k` of every repair
`a_k` is in `E` and `P_E` returns its input **bitwise** when the `S`-block
minimum is at or above `2 epsilon`, then by induction `e_k = 0` exactly,
`d_k = a_k` exactly, and every iterate, the exit, the returned point and the
map back are bitwise those of the default run. So a bitwise prediction for a
window and an `epsilon` is justified **only after** an audit of the default
run's affine iterates `a_k` (not its stage inputs) shows that none has an
`S`-block minimum below `2 epsilon`, at every repair in that window, with the
pre-check unchanged. The handoff's increment 2 reading rule (snapshot lines
401 to 404: no stage input below `-1e-8` implies bitwise equality) and its
increment 3 test "never active means nothing changes" (line 445) are withdrawn
on this ground.

**The default path is preserved independently.** Whatever the audit shows,
`electronic_density_floor=None` must leave the code path textually and
numerically the default one: no extra eigendecomposition, no extra array
operation, no change to `_clip_to_floor`, no new keys in the stage records.
This is tested against outputs recorded from the unmodified module before
the edit (increment 3, step 3a), not only by comparing the new module with
itself.

---

## 3. Point 3: unit scaling, the factor 2, and the reconstruction

**The floor.** The `S` rows have scale `1.0` (section 0), so the `S`-block of
the scaled retained matrix equals the `S`-block of the unscaled Gram, and by
point 1 a density-matrix eigenvalue floor `epsilon` is the principal-Gram
bound `G_e >= 2 epsilon I`, **not** `epsilon I`. The option is declared in
density-matrix units (`electronic_density_floor = epsilon`, admissible
`0 <= epsilon <= 1/2`, since a trace-one 2-by-2 density matrix has
`lambda_min <= 1/2`) and applied to the block as `2 epsilon`.

**Relation to the retained floor.** With retained floor `f_r` (scaled
metric), `E` contains the retained cone exactly when `2 epsilon <= f_r`. In
the task-11 configuration (`f_r = 0`) the option at `epsilon = 0` adds a set
that contains the retained cone, so it cannot change the intersection the
alternation converges toward; it can change only the finite-iteration path
and the returned iterate. Only `epsilon > f_r / 2` tightens the target set.

**The projection.** `E` constrains only the 16 real features of the
`S`-block (4 diagonal entries; the real and imaginary parts of the 6 pairs
`(0,5), (0,6), (0,7), (5,6), (5,7), (6,7)`), located inside the retained
block at offset `extension_rows**2` for `target="both"` (0 for
`"retained"`), at diagonal positions `0, 5, 6, 7` and at `8 + k` and
`8 + 28 + k` for the pair's index `k` in `np.triu_indices(8, 1)`. Because the
realification is Frobenius-isometric and `E` is a cylinder over the other
features, the Frobenius projection onto `E` is: rebuild the 4-by-4 block from
those 16 features, clip its eigenvalues to `2 epsilon`, write back **only**
those 16 features, leave the other 48 retained features and every extension
feature bitwise untouched, and return the input array unchanged (the same
values, no symmetrization or round trip) when the block's minimum eigenvalue
is already at or above `2 epsilon`. Do not reuse `_clip_to_floor` for the
inactive case (it symmetrizes, `:596-599`), and do not rebuild the whole
8-by-8 from features (the `/sqrt(2)` then `*sqrt(2)` round trip is not
bitwise).

**Reconstruction and residual validation**, each a test or a per-repair
check with the stated tolerance:

1. *Index map.* The 4-by-4 block rebuilt from the 16 features equals
   `_hermitian_from_features(retained_features, 8)[np.ix_(S, S)]` bitwise,
   and a write-back of a modified block changes exactly those 16 positions.
2. *Scaling.* `base_scales[[0, 5, 6, 7]]` are exactly `1.0`;
   `scaled_retained_matrix(state)[np.ix_(S, S)]` equals
   `uncentered_joint_moment_matrix(raw)[np.ix_(S, S)]` to `1e-12`
   (the affine evaluation is a tensor contraction, not bitwise).
3. *Identity.* The block's minimum eigenvalue equals `2 lambda_min(rho)` with
   `rho` from `raw_moment_coordinates_to_matrix_state(raw)` to `1e-13`, on
   random raw coordinates including `|r| > 1`; above `1e-10` it is a stop
   (point 1).
4. *Map back.* Per repair with the option on: `target_residual` (`:1303`) at
   most `1e-10`; the `S`-block minimum of the returned affine point
   (`corrected`) and of the mapped-back features (`achieved`) differ by at
   most `target_residual + 1e-14` (Weyl's bound, since the realification is
   Frobenius-isometric); and the mapped-back `lambda_min(rho)` equals half the
   mapped-back block minimum to `1e-12`. A failure of any of these is an
   implementation defect (point 4).

---

## 4. Point 4: the fixed-iteration repair can return with constraints unmet

The repair returns the last affine iterate after `K` iterations whether or
not the floors are met (`:1413`), and the map back is a least-squares step,
so adding `E` guarantees nothing by construction. The record measures this
already: every constrained chain violates the retained cone at all 4,800
samples past `t = 8`
(`MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`
section 3.2), and the minimum unshifted retained joint Gram of the divergent
chain reaches `-5.561e-4` over `t <= 20`
(`agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
section 4.1).

**Three outcomes, reported separately, per repair and counted per run:**

| outcome | definition (all in the scaled metric; electronic values also in density-matrix units) |
|---|---|
| floors met at return | the alternation exited early at iteration `k + 1 <= K` (`:1412`), and after the map back every block minimum is at or above its floor minus `1e-12` and `lambda_min(rho)` is at or above `epsilon - 5e-13` |
| finite-iteration residual violation | the alternation returned at the cap without meeting the floors, or the mapped-back minima are below floor minus `1e-12`, with every check of point 3 item 4 passing; the depth of the violation of each block (extension, retained, electronic) is reported |
| implementation defect | any nonfinite value; `target_residual` above `1e-10`; a failed Weyl or identity check (point 3 item 4); the inactive branch of `P_E` not returning its input bitwise; a nonzero `e_k` with no activation of `E`; an early exit reported while the mapped-back minima fall below floor minus `target_residual + 1e-12` |

Fields that carry this (populated only when the option is set; their
defaults leave the default path's dataclasses and records unchanged):
`CheapProjectionRepair.exit_iteration: int | None` (the `k + 1` of an early
exit, `None` at the cap), `floors_met_at_return: bool | None`,
`electronic_minimum_before` and `electronic_minimum_after` (density-matrix
units), `electronic_activations: int | None` (iterates at which `P_E` changed
its input), `electronic_first_activation: int | None`. For default-path runs
the same classification is computed post hoc from the existing fields
(`extension_minimum_after`, `retained_minimum_after`, `target_residual`),
which suffices because the default path has no electronic set; the returned
iteration count is not used for it (section 0, the ambiguity at `K`).

**The default return policy is preserved**: at the cap the repair returns
the last affine iterate and maps it back, exactly as now; no extra
iterations, no best-iterate selection, no exception, no fallback to another
policy, no retry. Any change to that is a separate named option with its own
review, outside this lane.

A finite-iteration residual violation is a **result** and is reported as
such; only an implementation defect is a stop. The handoff's increment 5
stop clause (snapshot lines 576 to 578: an electronic minimum below `-1e-8`
with the option on is "a defect of the option") is replaced by this
classification.

---

## 5. Point 5: what the lane may and may not claim

Electronic positivity is weaker than retained joint Gram positivity: the
`S`-block is a principal submatrix of the retained Gram, so a positive
semidefinite retained Gram implies `rho >= 0` (and, for a trace-one 2-by-2
`rho`, also `rho <= I`), not conversely. Neither condition, nor the two
together with the scaled positive extension, certifies that the moment vector
is the moment sequence of a quantum state: they are necessary conditions on a
truncated moment sequence, not a representability certificate.

**The lane may claim**, each with its definition, configuration and horizon:
the measured identity error; the counts of affine iterates at which `E` would
act in the default run, per window and per `epsilon`; with the option on, the
counts of the three outcomes of point 4, the activation counts, the measured
electronic, retained and extension minima after the map back, the
all-coordinate scalar RMS and Eq. (22) energy RMS against the exact
reference, and the wall ratio to the default run under the same load; and
"floors met at return" for exactly the repairs at which it was measured.

**The lane may not claim**: that the option makes the closure representable,
holds the retained joint Gram cone, or prevents or delays the divergence;
that electronic positivity implies retained or joint Gram positivity; that
the repaired states are electronically positive "by construction" (only
measured, repair by repair); that "nothing changes" without the iterate audit
and a bitwise comparison; that an `epsilon = 0` result changes the set the
alternation converges toward (it cannot, section 3); or any ordering of
accuracies inside the 7.1 percent band at `t = 4` (the one measured band,
`cheap_repair_route_results_20260918.md` section 3.1). All results are
`exploratory_local_not_promoted`.

---

## 6. The corrected increment 3 (supersedes the handoff's increment 3)

### Increment 3 — the iterate audit, then `CheapProjectionSettings.electronic_density_floor`, default off, with a bitwise default-path test

**Goal.** (3a) The default path's outputs are recorded from the unmodified
module; (3b) the default run's affine iterates are audited for where an
electronic set would act; (3c) the named option
`electronic_density_floor: float | None = None` exists, implemented as
sections 2 to 4 specify, and the `None` path is bitwise the recorded one.

**Preconditions.** Increments 0 to 2 accepted, including the addendum's
executable parity gate; the increment-1 identity error at or below `1e-10`
(point 1); `cheap_projection_route.py` hashes to
`25c4b0e3c9f2a070e4009c7d15a8c094a064c4dc6eab987fe03a94fb3385304d` in your
worktree before the edit.

**Bound.** Every command below finishes in under 20 minutes end to end,
including preparation and scoring; split any audit window that approaches
that into two invocations with distinct outputs.

**Step 3a — record the default path before editing (under 5 minutes).**
Write `paper_5/tests/data/record_cheap_projection_default_path.py` that, with
the unmodified module, computes and saves to
`paper_5/tests/data/cheap_projection_default_path_b9f99271.npz` (under
1 MB): from the test module's `prepared` fixture construction
(`prepare_cheap_projection_route(DimerParameters(), phonon_cutoff=16, settings=CheapProjectionSettings(policy="dykstra", target="both", dykstra_iterations=30))`),
(i) `repair` outputs (state, frontier, iterations, all four minima,
`target_residual`) on three seeded perturbations of size `1e-3` with
`energy_reference_mode="drive_power"` and a declared reference, and with the
hyperplane off; (ii) a rollout to `t = 0.0075` at `h = 0.0025` with
`drive_power` (states, frontier, both minima series, internal energy and
reference). Run it twice in two processes and compare.

```bash
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python paper_5/tests/data/record_cheap_projection_default_path.py --output "$OUT/golden_run1.npz"
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python paper_5/tests/data/record_cheap_projection_default_path.py --output "$OUT/golden_run2.npz"
cd "$WT" && /Users/lordskyler/Holstein_test/.venv/bin/python -c "import numpy as np; a=np.load('$OUT/golden_run1.npz'); b=np.load('$OUT/golden_run2.npz'); print(all(np.array_equal(a[k], b[k]) for k in a.files))"
cp "$OUT/golden_run1.npz" paper_5/tests/data/cheap_projection_default_path_b9f99271.npz
```

If the two runs differ, the linear algebra is not reproducible across
processes on this machine: do not commit the recording; instead vendor the
unmodified module as `paper_5/tests/data/cheap_projection_route_b9f99271.py`
and compare in-process in step 3c, and report the difference.

**Step 3b — the affine-iterate audit, read-only (each invocation under
20 minutes).** Add to `paper_5/src/paper5/stability/electronic_cone_audit.py`
a replica of `_dykstra` that performs the same operations in the same order,
calling the module's own `_cone_projection` and `_block_minima`, and records
per iterate `k`: the `S`-block minimum of `y_k`, `c_k` and `a_k`, the norms of
`p_k` and `q_k`, and whether the exit fired. For every audited repair it
asserts that its returned point and iteration count equal the module's
`_dykstra` output bitwise. Audit (i) the task-11 stage inputs of windows
`(0, 2]`, `(2, 4]` and `(6, 8]`, replayed as increment 2 does, and (ii) every
stage repair of a fresh default rollout from the Hartree--Fock
zero-correlation preparation with the `exact_moments` lift over `t = 0` to
`0.5`, captured by wrapping `route.coordinates.repair` in the audit script
only (no module edit). Report per window, at `epsilon = 0` and
`epsilon = 1e-6`: the number of repairs, the number with at least one affine
iterate whose `S`-block minimum is below `2 epsilon`, the deepest such
minimum, and the distribution of the first such iterate.

**Step 3c — the option (under 20 minutes including the suite).** Run the
impact analysis first (handoff C12) on `CheapProjectionSettings`,
`AffineGramCoordinates.repair`, `AffineGramCoordinates._dykstra`,
`AffineGramCoordinates._target_features` and
`integrate_cheap_projection_ssprk3`. Implement:

* `CheapProjectionSettings.electronic_density_floor: float | None = None`,
  density-matrix units; `__post_init__` refuses a value outside `[0, 1/2]`,
  a policy other than `"dykstra"`, a target other than `"retained"` or
  `"both"`, and `repair_coordinates="frontier_only"` (the retained block does
  not depend on the frontier; compare `:286-293`);
  `CarriedWitnessAffineGram.factorization` refuses it, as it refuses the
  band (`:2531`);
* `_dykstra(..., electronic_floor=None)`: when `None`, the existing loop
  runs unchanged; otherwise the placement of section 2, the projection of
  section 3 (block floor `2 epsilon`, 16-feature write-back, bitwise inactive
  branch), the exit check extended to the `S`-block, and the return of the
  last affine iterate at the cap;
* `repair`: the pre-check extended to the `S`-block only when the option is
  set; the fields of section 4 populated only when the option is set; the
  validation of section 3 item 4 computed only when the option is set;
* `integrate_cheap_projection_ssprk3`: the stage records and trajectory
  totals gain the section-4 fields only when the option is set.

Tests in a new file `paper_5/tests/test_electronic_density_floor.py`:

1. *Default path, against the recording*: with the option absent and with
   `electronic_density_floor=None`, the step-3a repairs and rollout reproduce
   `cheap_projection_default_path_b9f99271.npz` bitwise (`np.array_equal`
   on every array), and the stage records have exactly the pre-edit keys.
2. *Default path, in-process*: the established pattern of
   `test_energy_reference_band_is_off_and_changes_no_number`
   (`test_cheap_projection_route.py:1506-1562`).
3. *Reconstruction*: section 3 items 1 to 3 (index map bitwise, unit scaling
   to `1e-12`, identity to `1e-13` including `|r| > 1`).
4. *Inertness is per iterate*: on an input for which the step-3b replica
   shows every affine iterate inside `E` at `epsilon = 0`, the option's
   outputs equal the default's bitwise and `electronic_activations == 0`; on
   an input whose stage features are inside `E` but some affine iterate is
   not (taken from the step-3b audit, or constructed), `electronic_activations
   > 0`, which demonstrates point 2.
5. *Floor conversion*: at `epsilon = 0.05` on a constructed state with
   `lambda_min(rho) = -1e-3`, when `floors_met_at_return` is true the
   mapped-back `lambda_min(rho) >= 0.05 - 5e-13` and the `S`-block minimum
   `>= 0.1 - 1e-12` (an `epsilon I` implementation would stop at `0.025`).
6. *Return policy and reporting*: with `dykstra_iterations=1` on an input that
   does not meet the floors, the repair returns (no exception) the
   mapped-back last affine iterate with `exit_iteration is None` and
   `floors_met_at_return is False`; with enough iterations on an easy input,
   `exit_iteration` is an integer and `floors_met_at_return is True`.
7. *Refusals*: every refusal listed above raises `ValueError`.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q tests/test_electronic_density_floor.py tests/test_cheap_projection_route.py tests/test_electronic_cone_audit.py
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q
```

Then add `--electronic-density-floor` (absent by default, density-matrix
units) to the fixed driver copies, recorded in `summary.json` with the
outcome counts of section 4, and copy them to `$DATA/drivers/`.

**Acceptance.** Step 3a's two recordings are bitwise equal (or the vendored
fallback is in place and reported). Step 3b's replica equals `_dykstra`
bitwise at every audited repair, and the per-window counts are reported for
both `epsilon`. Step 3c: the seven tests pass; `test_cheap_projection_route.py`
passes unchanged; the full suite equals the increment-1 baseline plus the new
tests with no new failure. A bitwise prediction for increment 5 or 6 is
recorded **only** for windows where step 3b found zero affine iterates
outside `E` at the chosen `epsilon`; everywhere else increments 5 and 6 report
differences and the section-4 outcome counts.

**Stop.** The `None` path differs from the recording in any array (the
default path changed: fix before anything else; if the fix needs an existing
test changed, stop and report). The step-3b replica differs from `_dykstra`
in any audited repair (the audit is invalid). An identity or reconstruction
check above `1e-10` (point 1: diagnose conversion, ordering, scaling or
source). Any implementation defect of section 4 in the tests. Any existing
test changing outcome.

**Commit** the recording (or the vendored fallback), the audit module and
its JSON summaries under `$DATA`, the module change, the new tests and the
driver copies, with explicit paths.

---

## 7. Consequences for the other increments of the handoff (for the coordinator)

Read by reference with the snapshot `73128739...`; the handoff is not edited.

* Increment 1, stop clause (lines 358 to 360): replaced by point 1; an
  identity error above `1e-10` is a stop.
* Increment 2, reading rule (lines 401 to 404): stage inputs do not decide
  inertness; step 3b's affine-iterate audit does (point 2).
* Increment 3: replaced by section 6.
* Increment 5, answer and stop (lines 570 to 578): "nothing changes" requires
  zero affine-iterate activations and a bitwise comparison; an electronic
  minimum below the floor with the option on is classified by section 4, and
  only an implementation defect stops the lane.
* Increments 5 and 6 report the section-4 outcome counts beside every row,
  for the option and for the default reruns (post hoc for the default).
