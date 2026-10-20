# Codex handoff: a regularized product-state initialization (Paper V classical repair, 2026-10-10)

**What this is.** A standalone, exploratory research-and-implementation
handoff for one Codex session. It is one of four independent threads on the
Paper V classical repair (index:
`prompt-exports/codex-paper-v-INDEX-20261010.md`). It follows
`agent_guidance/shared/agent-handoff-contract.md`: anchors, scope,
increments that each end committable, evidence standards, traps and the
report-back block. Section C (the common rules, at the end) is binding; read
it before increment 0. Nothing here is evidence or manuscript prose.

**The question in one sentence.** The cheap repair's one conic solve, the
declared proximal log-det completion at `t = 0`, refuses at the
Hartree--Fock zero-correlation preparation because that pure product state's
retained joint Gram is exactly singular; does a declared, small,
state-realizable admixture make the declared completion certify at `t = 0`
without any frontier moment taken from a state vector, and what does that
admixture do to the repair of the divergent case over `t = 0` to `20`.

**What counts as the answer.** (i) For each of four declared admixtures, the
smallest weight at which the declared completion certifies at `t = 0`, at 15
and at 27 entrance keys, at the committed rung and at any rung of the
declared retry ladder; (ii) the unrepaired archive closure's `1e4` event and
electronic positivity loss from each admixed preparation, beside
`54.4612958` and `27.9633944`; (iii) for the smallest certifying weight of
each certifying admixture, the cheap repair over `t = 0` to `20` with the
declared completion as the `t = 0` lift, in the record's row format, scored
against the exact reference from the same admixed start and against the
exact reference from the Hartree--Fock zero-correlation start.

---

## 0. Identity of this thread

Run this block first in every shell (zsh or bash):

```bash
export TOPIC=regularized-product-state
export BRANCH=codex/paper-v-regularized-product-state-20261010
export WT=/Users/lordskyler/Holstein_test.worktrees/codex-paper-v-regularized-product-state-20261010
export SRC=/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914          # read-only
export PY=/Users/lordskyler/Holstein_test/.venv/bin/python
export OUT=$WT/output/local_runs/paper_v_regularized_product_state_codex_20261010
export REPORTFILE=$WT/agent_guidance/paper-v/handoffs/REPORT-regularized-product-state-codex-20261010.md
export DATA=$WT/agent_guidance/paper-v/handoffs/regularized-product-state-codex-20261010-data
export REC=$SRC/output/local_runs/paper_v_divergence_20260920                               # read-only
export CHAIN=$SRC/output/local_runs/paper_v_divergence_hf_dyk120_20260920                   # read-only; segments ${CHAIN}_seg1 .. _seg5 cover t = 0 to 20
export LOCK=$SRC/agent_guidance/paper-v/handoffs/codex-paper-v-regularized-product-state-20261010-inputs.sha256   # = appendix L
```

**Drivers this thread runs.** The preparation is a new **committed** module,
`paper_5/src/paper5/stability/regularized_preparation.py` (increment 1). The
probe, the unrepaired closure and the chained repaired runs use copies of
**uncommitted** scripts under
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/`
(`probe_preparation.py`, `run_closed_route.py`, `run_divergence_segment.py`,
`score_chain.py`; SHA-256 in appendix L). The segment driver is fixed as C20
decides; increment 0 runs C20's parity gate at the record's setting and
increment 4 runs it again at the admixed setting before any chain starts.

Inputs, all **read-only**, by absolute path (C5 lists what is inside each):

* `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/`:
  `hartree_fock_preparation.py` (`hartree_fock_state_vector`,
  `hartree_fock_joint_moment_initial_state`), `exact_lift.py`
  (`exact_moment_lift`), `probe_preparation.py` (rung-by-rung probe of the
  declared completion), `run_closed_route.py` (the unrepaired archive closure
  from a named preparation), `run_unrepaired.py` (`AMPLITUDE_THRESHOLD`,
  `_closed_diagnostics`), `run_divergence_segment.py` (one cheap-repair
  segment, `prepare_route`, `_reference_coordinates`), `score_chain.py`; the
  score files `preparation_probe_hf.json` (the recorded refusals) and
  `hf_chain_score_t20.json` (the record's rows);
* the record's repaired chain over `t = 0` to `20`:
  `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/`
  through `..._seg5/` (`trajectory.npz`, `summary.json`, `resume.npz`,
  `reselections.json`); this thread reads them only for comparison.

The record this handoff cites (tracked): `agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
(**DI**, by section and line); `MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`
(**CR**); `paper_5/notes/electron_phonon_closure_worklog_20260803.md`, the
2026-09-20 block "the cheap repair on the initialization that actually
diverges", lines 8200 to 8285 (**WL**);
`MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md`
(**WN**).

---

## 1. Why this thread exists

The record's own divergence, the one the repair exists to fix, starts from
the Hartree--Fock zero-correlation preparation (DI section 0). The cheap
repair could not be prepared there: its declared completion refused at every
rung, and the record substituted the exact frontier moments of the
preparation's own state vector as the `t = 0` lift (DI section 3). That
substitution is a diagnostic: it uses a state vector, which a route applied
without an exact reference does not have. This thread tries the honest
alternative: change the **preparation** by a declared, small amount so that
the route's own declared completion certifies, and measure what that change
costs against the record's rows.

---

## 2. Anchors from the record

**The refusal** (DI section 3, lines 106 to 158; WL lines 8231 to 8250).

| fact | number | source |
|---|---|---|
| outcome of the declared completion at the Hartree--Fock zero-correlation preparation | refuses at 15 and at 27 entrance keys at every rung tried: `logdet_shift` `1e-10` (committed), `1e-8`, `1e-7`, `1e-6`, `1e-4`, `1e-2`, `1e-1`, and `logdet_weight` `1e-1` | DI section 3 table, lines 116 to 119 |
| refusal message, identical every time | `moment_extension_degeneracy: facial reduction removed every frontier direction` | DI line 114 |
| seconds per attempt | `0.4` to `0.5 s` at 15 keys, `2.2 s` at 27 keys | DI lines 118 to 119 |
| the stated cause | a pure product state; its retained joint Gram's two smallest eigenvalues are `-4.44e-16` and `-4.21e-16` against a largest of `8.405`, stated as "rank 6 of 8"; facial reduction acts on the lower moment matrix before the log-det barrier is seen, so no shift restores a direction | DI lines 121 to 129 |
| the exact correlated preparation at the same settings | retained joint Gram minimum `+7.228147e-6`; electronic one-body density matrix minimum `0.1088344`; boson-moment minimum `2.893143e-5`; at 27 keys the declared completion refuses at the committed rung (40 Newton iterations, relative face support 35/36) and the first ladder rung, `logdet_shift 1e-8`, certifies in 12 iterations | DI section 2.2 table, lines 89 to 94; CR section 2(a), lines 105 to 110 |
| what was substituted | the 70 frontier moments of the preparation's own state vector, contracted through `moment_hierarchy(5)` (`exact_lift.py`), recorded as `lift = exact_moments`; a diagnostic substitution for one solve at `t = 0` | DI lines 134 to 147 |
| after `t = 0` | re-selections certify; the first, at step 16, moved the retained joint Gram from `-4.10e-6` to `-5.79e-8` | DI lines 149 to 152 |
| Eq. (22) internal energy at `t = 0` | Hartree--Fock zero-correlation `-3.5`; exact correlated `-3.775326553` | DI lines 44 to 48 |
| the preparation as a state vector | the Hartree--Fock determinant times one coherent state per local oscillator at `-g / omega_ph = -1.2247449`; reproduces `hartree_fock_closed_scalar_coordinates` to `1.02e-10`, truncation weight `6.74e-13` at cutoff 16 | DI section 1, lines 32 to 42 |

**The rows to compare against** (the cheap repair in the task-11
configuration, glossary C19, `lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude 1, phonon cutoff 16; RMS is the all-coordinate scalar RMS of the
31 closed scalar coordinates against the exact driven reference from **the
same** preparation, cumulative from `t = 0`).

| quantity, `t = 0` to `20` | Hartree--Fock zero-correlation, `exact_moments` lift (DI sections 4.1, 4.2) | exact correlated, declared completion at `logdet_shift 1e-8` (CR sections 3.1, 3.2; DI section 4.2) |
|---|---:|---:|
| all-coordinate scalar RMS at `t = 4` / `8` / `20` | `1.0803e-2` / `3.7209e-2` / `1.5368e-1` | `9.290e-3` / `3.304e-2` / `5.604e-2` (band at `t = 4`: `9.29e-3` to `9.95e-3`, 7.1 percent, four members) |
| Eq. (22) energy RMS against the reference at `t = 20` | `1.1888e-3` | `5.336e-3` |
| electronic one-body density matrix minimum | `-2.220e-16` at `t = 0`, `+1.447e-6` over `0 < t <= 20`, at `t = 0.0025` | not recorded |
| minimum unshifted retained joint Gram | `-5.561e-4` | `-4.460e-4` |
| minimum scaled positive extension | `-6.352e-5` | `-3.936e-5` |
| minimum boson-moment eigenvalue | `-2.163e-3` | not in the record's rows |
| maximum `\|x_i\|` | `1.9431` | not in the record's rows |
| lift re-selections / solver refusals | `500` / `44` | `500` / `117` |
| autonomous wall | `4,296.9 s` (five segments) | `5,183.6 s` |

**The unrepaired archive closure from the two preparations** (31 closed
scalar coordinates, DOP853 `rtol 1e-10`, `atol 1e-12`; DI section 2.2): from
the Hartree--Fock zero-correlation preparation the electronic one-body
density matrix first falls below `-1e-8` at `27.9633944` and the `1e4`
magnitude event is at `54.4612958`; from the exact correlated preparation
the electronic crossing is at `1.0362132` and there is no event through
`t = 140` (maximum `|x_i|` `24.2421`), the working notes' 31-coordinate
protocol placing it at `141.4056` (WN line 416).

---

## 3. Code facts you will rely on (verified from source at `b9f99271`, not from the record's prose)

1. **Where the solve sits.** `prepare_route` in `run_divergence_segment.py`
   contracts the preparation into raw moment coordinates and third-order
   hierarchy coordinates, packs the 56-coordinate state and the 41 lower
   moments (`state_lower_moments`), and calls
   `SymmetryReducedPositiveExtension.complete(lower)`
   (`paper_5/src/paper5/stability/apcm_positive_extension.py:748`;
   `APCMExtensionSettings` at line 53, committed `logdet_shift = 1e-10`,
   `logdet_weight = 1.0`), falling back through `build_completion_ladder`
   (`projected_apcm.py:65`) when `--initialization-ladder` is given (default
   `--ladder-shifts 1e-8 1e-7`, `--ladder-weights 1e-1`). The completion
   needs only the lower moments; the frontier is its output.
2. **Moments of a mixture.** The hierarchy coordinates are expectations of
   fixed, state-independent observables (`_moment_hierarchy_observables`,
   `exact_reference.py:1026`; `_contract_moment_hierarchy_coordinates`,
   line 1065), so they are linear in the density matrix. The matrix state
   returned by `_contract_matrix_state` (`exact_reference.py:474`) is
   **not**: its phonon covariance, anomalous covariance and electron--phonon
   correlation are connected quantities (an expectation minus a product of
   expectations). The raw moment coordinates
   (`matrix_state_to_raw_moment_coordinates`) are uncentered and linear. A
   mixture's preparation is therefore formed by mixing raw moment
   coordinates and hierarchy coordinates, then converting raw to a matrix
   state and to the 31 closed scalar coordinates. Mixing matrix states or
   closed scalar coordinates directly is wrong.
3. **Linear quantities.** The retained joint Gram is affine in the raw
   coordinates (`uncentered_joint_moment_matrix`,
   `adaptive_positive_moment.py:458`), so a mixture's Gram is the same convex
   combination of the components' Grams. The Eq. (22) internal energy is
   linear in the 29 raw moment coordinates (CR section 2(d)).
4. **The record's "rank 6 of 8" is an inference from two eigenvalues.** The
   probe stored only the minimum, second and maximum eigenvalue
   (`preparation_grams` in `preparation_probe_hf.json`). Reading the
   matrix's definition suggests more null directions: for a coherent state
   `(b_q - alpha_q)` annihilates the state, one null direction per
   oscillator (two); and for the idempotent electronic one-body density
   matrix the `(1, sigma)` block, which equals twice `rho` tensored with the
   2-by-2 identity up to a change of basis, has two zero eigenvalues, two
   more; that would be rank 4 of 8. This is the handoff author's derivation,
   not a measurement; increment 0 prints all eight eigenvalues and settles
   it. It matters because an admixture has to lift every null direction the
   completion's facial reduction sees.
5. **What the drivers need changed.** `probe_preparation.py`,
   `run_closed_route.py` and `run_divergence_segment.py` choose the
   preparation from `{"exact_correlated", "hartree_fock"}`; each needs a
   preparation-arm and weight flag that calls the new module.
   `_reference_coordinates` must return the mixture's exact reference.
   `score_chain.py` scores against the `exact_coordinates` each segment
   stored (the reference from the segment's own preparation); a second
   score against the Hartree--Fock zero-correlation reference needs a flag
   that recomputes that reference on the stored grid.

---

## 4. The declared admixtures

All are **genuine density matrices** built from a finite, declared set of
pure states. That is the requirement, for two reasons: a genuine state's
moments are a genuine moment sequence, so a positive extension exists; and a
genuine state has an exact driven reference from the same start (the same
weighted mixture of the components' exact trajectories, combined at the
level of raw moments). A Gram inflation that no state realizes has neither,
and is out of scope.

* **`correlated_admixture` (arm A).**
  `rho_A(eps) = (1 - eps) |HF><HF| + eps |GS><GS|`, with `|HF>` the
  Hartree--Fock zero-correlation product state vector at cutoff 16 and
  `|GS>` the exact zero-field ground state at cutoff 16 (the exact
  correlated preparation). Its retained joint Gram is
  `(1 - eps) G_HF + eps G_GS`, with minimum eigenvalue at least about
  `eps * 7.228147e-6`. Weights `eps` in `{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`;
  `eps = 0` and `eps = 1` are reproduction checks.
* **`uncorrelated_inflation` (arm B).** A mixed product state that keeps
  every electron--phonon correlation at zero: each spin's orbital state is
  `(1 - delta) |bonding><bonding| + delta |antibonding><antibonding|` (four
  electronic product components), and each local oscillator is a displaced
  thermal state about the same coherent displacement `-g / omega_ph` with
  mean occupation `delta`, Fock weights `p_n = delta^n / (1 + delta)^(n + 1)`
  truncated at the smallest `n_max` whose discarded weight is below `1e-12`
  (record `n_max` and the discarded weight, and renormalize). Weights
  `delta` in `{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`; `delta = 0` is the
  reproduction check.
* **`electronic_inflation` (arm B-e)** and **`phonon_inflation` (arm B-ph).**
  Arm B with only the electronic mixing, or only the phonon thermal part,
  so the probe shows which null directions block the completion.

The option values are code identifiers, not quantity names; in prose call
them by their definitions.

---

## 5. Scope specific to this thread

* In scope: a new preparation module with the arms as named options; the
  spectrum of the retained joint Gram at `t = 0`; the rung-by-rung probe; the
  unrepaired archive closure from each admixed preparation; cheap-repair
  rollouts over `t = 0` to `20` from admixed preparations with the
  **declared** completion as the `t = 0` lift.
* Out of scope: any change to `SymmetryReducedPositiveExtension.complete`,
  its facial reduction, its tolerances, or the ladder's defaults; horizons
  beyond `t = 20`; the `exact_moments` lift for admixed preparations except
  as a reported distance at `t = 0`.
* Machine headroom: the probe costs seconds per attempt; each rollout is
  about `650 MiB` and about 4,300 to 5,200 s to `t = 20`; run up to four
  concurrently after `memory_pressure -Q`.

---

## 6. Increments

Proceed through all of them autonomously (C15).

### Increment 0 — worktree, input lock, readable inputs, the full spectrum, driver fix and the parity gate (under 20 minutes)

**Goal.** Your worktree exists at `b9f99271`; every input matches its
SHA-256; the drivers are copied byte for byte; every input opens; the
Hartree--Fock zero-correlation preparation's retained joint Gram spectrum is
printed in full; the stored chain's Eq. (22) energy reference is read per
segment; the copied segment driver is fixed as C20 decides; and C20's parity
gate passes at the task-11 configuration from the Hartree--Fock
zero-correlation preparation with the `exact_moments` lift (the record's
setting; increment 4 repeats the gate at the admixed setting).

**Bound.** Each command below finishes in under 20 minutes; the whole
increment takes about 10 minutes.

```bash
git -C /Users/lordskyler/Holstein_test worktree add -b "$BRANCH" "$WT" b9f99271fc3ddf3db604d3acb64617a291bec7e1
git -C "$WT" rev-parse HEAD
mkdir -p "$OUT" "$DATA/drivers"
shasum -a 256 -c "$LOCK"                                    # every line must print OK
cp "$REC"/*.py "$REC"/*.sh "$OUT"/
(cd "$REC" && shasum -a 256 *.py *.sh) > "$OUT/copies_source.sha256"
(cd "$OUT" && shasum -a 256 *.py *.sh) > "$OUT/copies_copied.sha256"
diff "$OUT/copies_source.sha256" "$OUT/copies_copied.sha256" && echo "copies byte-identical"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python - <<'PY'
import json, os, numpy as np
import paper5.stability.apcm_positive_extension as m; print('import', m.__file__)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.moment_hierarchy import THIRD_ORDER_HIERARCHY
from paper5.stability.adaptive_positive_moment import matrix_state_to_raw_moment_coordinates, uncentered_joint_moment_matrix
from hartree_fock_preparation import hartree_fock_joint_moment_initial_state
rec = os.environ['REC']; chain = os.environ['CHAIN']
p = DimerParameters(gamma=0.5, lambda_ep=1.5, drive_amplitude=1.0)
hf = hartree_fock_joint_moment_initial_state(p, hierarchy=THIRD_ORDER_HIERARCHY, phonon_cutoff=16)
g = uncentered_joint_moment_matrix(matrix_state_to_raw_moment_coordinates(hf.matrix_state))
w = np.linalg.eigvalsh(0.5 * (g + g.conj().T))
print('HF energy', hf.energy, 'truncation weight', hf.truncation_weight)
print('HF retained joint Gram eigenvalues', w, 'rank at 1e-12 x largest', int(np.sum(w > 1e-12 * w[-1])))
print('probe rows', len(json.load(open(rec + '/preparation_probe_hf.json'))['rows']))
print('record score keys', list(json.load(open(rec + '/hf_chain_score_t20.json')).keys()))
previous = None
for k in range(1, 6):
    base = f'{chain}_seg{k}'
    d = np.load(base + '/trajectory.npz'); s = json.load(open(base + '/summary.json')); r = np.load(base + '/resume.npz')
    ref = np.asarray(d['internal_energy_reference'])
    print(f'seg{k}', s['dimensions']['lift'], 'wall', s['rollout']['autonomous_seconds'], 'reference at start', float(ref[0]),
          'previous reference at resume', previous, 'difference', None if previous is None else abs(float(ref[0]) - previous))
    previous = float(ref[int(np.argmin(np.abs(d['times'] - float(r['time']))))])
PY
```

Then fix `$OUT/run_divergence_segment.py` exactly as C20 items 1 to 4 say,
write the parity checker and run the gate:

```bash
cat > "$OUT/parity_check.py" <<'PY'
"""C20 parity gate: unbroken run against two segments spliced at an accepted resume time."""
import sys, numpy as np
unbroken, first, second = sys.argv[1:4]
u = np.load(unbroken + '/trajectory.npz'); a = np.load(first + '/trajectory.npz'); b = np.load(second + '/trajectory.npz')
resume_time = float(np.load(first + '/resume.npz')['time'])
cut = int(np.searchsorted(a['times'], resume_time + 1e-9))
times = np.concatenate([a['times'][:cut], b['times'][1:]])
coords = np.concatenate([a['archive_coordinates'][:cut], b['archive_coordinates'][1:]])
ref_first = float(a['internal_energy_reference'][cut - 1]); ref_second = float(b['internal_energy_reference'][0])
print('resume time', resume_time)
print('strictly increasing', bool(np.all(np.diff(times) > 0)), 'same grid as unbroken', times.shape == u['times'].shape and bool(np.max(np.abs(times - u['times'])) < 1e-12))
print('seam state bit-equal', bool(np.array_equal(a['archive_coordinates'][cut - 1], b['archive_coordinates'][0])))
print('reference: first segment at resume', ref_first, 'second segment at start', ref_second, 'difference', abs(ref_first - ref_second))
print('max abs difference, spliced minus unbroken, 31 closed scalar coordinates', float(np.max(np.abs(coords - u['archive_coordinates']))))
PY
SEG="$OUT/run_divergence_segment.py"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" --preparation hartree_fock --lift exact_moments \
  --output-directory "$OUT/parity_task11_unbroken" --final-time 0.2 --wall-budget-seconds 900 --record-reselections
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" --preparation hartree_fock --lift exact_moments \
  --output-directory "$OUT/parity_task11_segA" --final-time 0.1 --wall-budget-seconds 900 --record-reselections
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" --preparation hartree_fock --lift exact_moments \
  --output-directory "$OUT/parity_task11_segB" --resume-from "$OUT/parity_task11_segA" --final-time 0.2 --wall-budget-seconds 900 --record-reselections
cd "$WT" && /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/parity_check.py" \
  "$OUT/parity_task11_unbroken" "$OUT/parity_task11_segA" "$OUT/parity_task11_segB"
cp "$OUT/run_divergence_segment.py" "$OUT/parity_check.py" "$DATA/drivers/"
```

Write `$DATA/input_manifest.json` (expected and actual SHA-256 of every
locked input, every copy, and the edited drivers) and put the spectrum and
the per-segment reference table into the report-back.

**Acceptance.** HEAD is `b9f99271fc3ddf3db604d3acb64617a291bec7e1`; every
lock line prints `OK`; the copies are byte-identical; imports resolve under
`$WT`; Eq. (22) energy `-3.5` to `1e-9`; truncation weight `6.74e-13` to two
digits; the two smallest retained joint Gram eigenvalues within `1e-15` of
`-4.44e-16` and `-4.21e-16` and the largest within `5e-4` of `8.405`, with
all eight eigenvalues and the rank recorded (this settles section 3 fact 4);
segments 1 to 5 report `lift = exact_moments` and walls summing to
`4,296.9 s` to `0.1 s`; the stored chain's reference at each segment start
differs from the previous segment's reference at its resume time by at most
`1e-12`; the parity gate passes as C20 states.
**Stop.** Any lock line not `OK`, any missing input, an import outside
`$WT`, the eigenvalues not matching the record, a stored reference
discontinuity above `1e-12`, or a parity failure (C20; a re-selection
decision flip at the feasibility edge is reported with its two decision
records).
Commit `$DATA/input_manifest.json`, `$DATA/drivers/run_divergence_segment.py`
and `$DATA/drivers/parity_check.py`.

### Increment 1 — the preparation module with the admixtures as named options

**Goal.** The test baseline is measured before any source change; then a
committed module contracts the Hartree--Fock zero-correlation product state
and the four admixtures into every representation the route needs, and
returns the exact driven reference of the same mixture.

**Bound.** The suite takes about `250` to `265 s`; the new tests run in under
2 minutes.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q   # the baseline (C16), before any edit
```

Write `paper_5/src/paper5/stability/regularized_preparation.py` by porting
`hartree_fock_preparation.py` and adding:

* `regularized_preparation(parameters, *, phonon_cutoff=16, arm="none", weight=0.0, hierarchy=THIRD_ORDER_HIERARCHY)`
  with `arm` in `{"none", "correlated_admixture", "uncorrelated_inflation", "electronic_inflation", "phonon_inflation"}`;
  it returns the `(weight, state vector)` components, the raw moment
  coordinates, the matrix state, the 31 closed scalar coordinates, the
  hierarchy coordinates, the Eq. (22) energy, the truncation weights and
  `n_max`, mixing only linear quantities (section 3 fact 2). `arm="none"`
  (and any arm at `weight=0.0`) is the Hartree--Fock zero-correlation
  preparation;
* `mixture_exact_reference(parameters, preparation, times, *, phonon_cutoff=16, time_step=0.0025)`:
  propagate each pure component with the settings of `_reference_coordinates`
  (DOP853, `rtol 1e-9`, `atol 1e-11`, `max_step = min(0.01, 4 h)`), contract
  raw moment coordinates per component, mix, convert to the 31 closed scalar
  coordinates. (A density-matrix propagation is acceptable instead; then
  check it against the component sum at one weight to `1e-10`.)

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q tests/test_regularized_preparation.py
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q
```

Tests in `paper_5/tests/test_regularized_preparation.py`:

* default path: the closed scalar coordinates equal
  `hartree_fock_closed_scalar_coordinates` to `<= 1.1e-10` (the record's
  `1.02e-10`), the Eq. (22) energy equals `-3.5` to `1e-10`, and every
  default-path output is bitwise equal at `weight=0.0` for every arm (C7);
* `correlated_admixture` at `weight=1.0`: raw moment coordinates equal those
  of `exact_holstein_joint_moment_initial_state(..., canonical_embedding=True)`
  at cutoff 16 to `1e-12`; energy `-3.775326553` to `1e-9`;
* linearity: at `weight=0.5` the retained joint Gram equals the average of
  the two components' Grams to `1e-13`, **and** the closed scalar
  coordinates differ from the average of the components' closed scalar
  coordinates by more than `1e-6` somewhere (proving the connected blocks
  were formed from mixed raw moments);
* `uncorrelated_inflation` at `weight=1e-2`: electron--phonon correlation
  zero to `1e-12`, retained joint Gram minimum eigenvalue above `1e-6`;
* `mixture_exact_reference` at `weight=0.0` equals `_reference_coordinates`
  for the Hartree--Fock preparation at `t = 0.01, 0.02, 0.05` to `1e-10`.

**Acceptance.** The new test file passes; the full suite equals the
baseline count plus the new tests with no new failure.
**Stop.** The `weight=1.0` check fails beyond `1e-10`: the contraction path
differs from the exact correlated preparation's; report the coordinate and
stop.
Commit the module and its tests.

### Increment 2 — the spectrum and the rung-by-rung certification probe

**Goal.** For every admixture and weight, the full retained joint Gram
spectrum and the outcome of the declared completion at every rung, at 15 and
27 entrance keys; and, for certified attempts, the distance of the declared
frontier from the mixture's exact frontier (a reported number only, C8).

Extend your copy `$OUT/probe_preparation.py`: a flag
`--admixtures ARM:WEIGHT [ARM:WEIGHT ...]` that adds preparations from
`regularized_preparation`; per preparation, all eight retained joint Gram
eigenvalues; per attempt, `success`, `certified` if the result exposes it,
`iterations`, `scaled_minimum_eigenvalue`, `relative_face_support_dimension`
and `relative_face_direction_dimension` when present on the result,
`message`, `seconds`; a flag `--exact-frontier-distance` that, for every
successful attempt, reports the norm and maximum absolute difference between
the 70 (or 62) standardized frontier coordinates the completion returned and
the mixture's exact ones (contract `moment_hierarchy(5)` per component and
mix, as `exact_lift.py` does for one vector). Copy the edited driver to
`$DATA/drivers/`.

```bash
probe () {   # usage: probe NAME ADMIXTURES...   (one invocation per arm keeps each run under 20 minutes)
  local name=$1; shift
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/probe_preparation.py" \
    --output "$OUT/probe_${name}.json" --preparations hartree_fock --admixtures "$@" \
    --dictionaries entrance15 entrance15_plus_two_spin_boson \
    --shifts 1e-8 1e-7 1e-6 1e-4 1e-2 1e-1 --weights 1e-1 --exact-frontier-distance \
    > "$OUT/probe_${name}.log" 2>&1)
}
probe correlated correlated_admixture:1e-6 correlated_admixture:1e-5 correlated_admixture:1e-4 correlated_admixture:1e-3 correlated_admixture:1e-2 correlated_admixture:1e-1 correlated_admixture:1
probe uncorrelated uncorrelated_inflation:1e-6 uncorrelated_inflation:1e-5 uncorrelated_inflation:1e-4 uncorrelated_inflation:1e-3 uncorrelated_inflation:1e-2 uncorrelated_inflation:1e-1
probe subarms electronic_inflation:1e-4 electronic_inflation:1e-2 electronic_inflation:1e-1 phonon_inflation:1e-4 phonon_inflation:1e-2 phonon_inflation:1e-1
cp "$OUT"/probe_*.json "$DATA/"
```

**Bound.** Each invocation is at most 7 preparations by 2 dictionaries by 8
rungs at `0.4` to `2.2 s` per attempt (DI section 3), under 5 minutes plus
the exact frontier contractions; if an invocation approaches 20 minutes,
split its admixture list across two invocations with distinct output names.

**Acceptance.** Reproduction: `hartree_fock` refuses at all 16 attempts (8
rungs, 2 dictionaries) with the recorded message; `correlated_admixture:1`
at 27 keys refuses at the committed rung and certifies at `logdet_shift 1e-8`
(the record: 40 and 12 Newton iterations). The answer to part (i): for each
admixture, the smallest weight that certifies at 27 keys at the committed
rung, and the smallest that certifies at any rung, with the rung; the same
at 15 keys; and which of `electronic_inflation` and `phonon_inflation`
certifies, which identifies the null directions that block the completion.
**Stop.** If no admixture certifies at 27 keys at any weight up to `1e-1`
at any rung, run increment 3, then stop: report the face dimensions at the
largest weight per admixture. That is a negative result about the
completion's facial reduction, not about the admixtures.
Commit `$DATA/probe_*.json` and the driver copy.

### Increment 3 — the unrepaired archive closure from each admixed preparation

**Goal.** Separate what the admixture does to the divergence on its own from
what the repair does.

Extend your copy `$OUT/run_closed_route.py` with `--preparation-arm` and
`--preparation-weight` (calling `regularized_preparation(...).closed_coordinates`),
and add the electronic crossing times and the retained joint Gram crossing
to its JSON (reuse `_closed_diagnostics` from `run_unrepaired.py` and
`_first_below` from `run_divergence_segment.py`). Copy it to
`$DATA/drivers/`.

**Bound.** Ten propagations of `400` units at about `6.5 s` each, under
2 minutes.

```bash
cd "$WT" && for spec in none:0 correlated_admixture:1e-6 correlated_admixture:1e-4 correlated_admixture:1e-2 correlated_admixture:1e-1 correlated_admixture:1 \
    uncorrelated_inflation:1e-6 uncorrelated_inflation:1e-4 uncorrelated_inflation:1e-2 uncorrelated_inflation:1e-1; do
  arm=${spec%%:*}; w=${spec##*:}
  PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/run_closed_route.py" \
    --route archive --preparation hartree_fock --preparation-arm "$arm" --preparation-weight "$w" \
    --final-time 400 --sample-step 0.0025 --output "$OUT/unrepaired_${arm}_${w}.json"
done
```

**Acceptance.** `none:0` gives the `1e4` event within `1e-3` of `54.4612958`
and the electronic crossing within `1e-3` of `27.9633944` (the module's
contraction differs from the analytic preparation by up to `1.02e-10`;
report the difference); `correlated_admixture:1` gives the electronic
crossing within `1e-6` of `1.0362132` and an event time, to be reported
beside the working notes' `141.4056`. The answer to part (ii): the event and
electronic crossing times per admixture and weight. Reading rule recorded in
`$REPORTFILE`: a weight that moves the unrepaired event time by more than
`1` time unit from `54.4612958` defines a materially different initial value
problem, and increment 4 must say so beside its rows.
**Stop.** `correlated_admixture:1` with an event time outside
`[140, 143]`: the exact correlated contraction differs from the working
notes' protocol; report it and stop.
Commit `$DATA/unrepaired_from_admixtures.json` (one file collecting the ten)
and the driver copy.

### Increment 4 — the cheap repair from admixed preparations, `t = 0` to `20`, declared completion as the lift

**Goal.** The record's rows re-measured from admixed preparations whose
`t = 0` lift is the route's own declared completion.

Choose from increment 2: for each admixture that certifies at 27 keys, its
smallest certifying weight, and ten times that weight (at most four chains;
if only one admixture certifies, also run it at one hundred times its
smallest weight). Extend your copy `$OUT/run_divergence_segment.py`:
`--preparation-arm`, `--preparation-weight` (calling `regularized_preparation`
inside `_contract_preparation` and `mixture_exact_reference` inside
`_reference_coordinates`), and record arm, weight, rung and
`lift = declared_completion` in `summary.json`. Extend your copy
`$OUT/score_chain.py` with `--reference-preparation hartree_fock`, which
recomputes the Hartree--Fock zero-correlation exact reference on the stored
grid instead of using the stored `exact_coordinates`. Copy both to
`$DATA/drivers/`. `RUNG_SHIFTS` below is the list of shifts up to and
including the certifying rung found in increment 2.

**Step 1, the parity gate at each chosen admixed setting (C20).** Bound:
three short rollouts per setting, each under 5 minutes. On a resume the fixed
driver takes the lift from `resume.npz` and does not re-solve the `t = 0`
completion (C20 item 2), so segment B must not refuse where segment A
certified.

```bash
parity_admixed () {   # usage: parity_admixed ARM WEIGHT RUNG_SHIFTS...
  local arm=$1 w=$2; shift 2
  local SEG="$OUT/run_divergence_segment.py" P="$OUT/parity_${arm}_${w}"
  local common=(--preparation hartree_fock --preparation-arm "$arm" --preparation-weight "$w" \
    --lift declared_completion --initialization-ladder --ladder-shifts "$@" --ladder-weights 1e-1 --wall-budget-seconds 900 --record-reselections)
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" "${common[@]}" \
    --output-directory "${P}_unbroken" --final-time 0.2) || return 1
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" "${common[@]}" \
    --output-directory "${P}_segA" --final-time 0.1) || return 1
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" "${common[@]}" \
    --output-directory "${P}_segB" --resume-from "${P}_segA" --final-time 0.2) || return 1
  /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/parity_check.py" "${P}_unbroken" "${P}_segA" "${P}_segB"
}
parity_admixed correlated_admixture <EPS_MIN> <RUNG_SHIFTS>     # one line per chosen setting
```

**Step 2, the chains.** One segment of at most `900 s` of propagation per
process (under 20 minutes with its exact reference), each resumed from the
previous segment's checkpoint with its carried reference printed (C20).

```bash
run_admixed () {   # usage: run_admixed ARM WEIGHT RUNG_SHIFTS...
  local arm=$1 w=$2; shift 2
  local name="${arm}_${w}" prev="" k out
  for k in {1..10}; do
    out="$OUT/chain_${name}_seg$k"
    if [[ -f "$out/summary.json" ]]; then prev=$out; continue; fi
    if [[ -e "$out" ]]; then echo "refusing to reuse $out (C15); rename and retry"; return 1; fi
    local resume=()
    [[ -n "$prev" ]] && resume=(--resume-from "$prev")
    (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/run_divergence_segment.py" \
      --preparation hartree_fock --preparation-arm "$arm" --preparation-weight "$w" \
      --lift declared_completion --initialization-ladder --ladder-shifts "$@" --ladder-weights 1e-1 \
      --output-directory "$out" --final-time 20.0 --wall-budget-seconds 900 --record-reselections "${resume[@]}") || return 1
    prev=$out
    /Users/lordskyler/Holstein_test/.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$out/summary.json'))['rollout']['final_time_reached'] >= 20.0 - 1e-9 else 1)" && break
  done
}
memory_pressure -Q
# one line per chosen (arm, weight); example with placeholders filled from increment 2:
run_admixed correlated_admixture <EPS_MIN> <RUNG_SHIFTS> > "$OUT/chain_A_min.log" 2>&1 &
run_admixed correlated_admixture <10xEPS_MIN> <RUNG_SHIFTS> > "$OUT/chain_A_10x.log" 2>&1 &
wait
grep -h "carried Eq. (22) energy reference" "$OUT"/chain_*.log      # one line per resumed segment, into the report
for name in $(cd "$OUT" && ls -d chain_*_seg1 | sed 's/_seg1$//'); do
  ordered=$(cd "$OUT" && ls -d ${name}_seg* | awk -F seg '{print $NF, $0}' | sort -n | cut -d' ' -f2 | sed "s|^|$OUT/|")
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/score_chain.py" \
    $(echo $ordered) --horizons 1 2 4 8 12 16 20 --output "$OUT/${name}_score_same_start.json")
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/score_chain.py" \
    $(echo $ordered) --horizons 1 2 4 8 12 16 20 --reference-preparation hartree_fock --output "$OUT/${name}_score_hf_start.json")
done
```

(The `<...>` placeholders are the only values this handoff cannot fix in
advance; they are increment 2's result and must be written into
`$REPORTFILE` before the chains start.)

**Acceptance.** Each chosen setting passes its parity gate (C20 criteria);
each chosen chain reaches `t = 20`, its preparation certified at the rung
increment 2 found, every resumed segment printing a carried reference equal
to its predecessor's at the resume time to `1e-14`. The answer to part (iii), per chain beside the
two record rows of section 2: all-coordinate scalar RMS at `t = 4`, `8`, `20`
against both references; Eq. (22) energy RMS against the reference at
`t = 20`; electronic one-body density matrix minimum over `0 < t <= 20`, its
time and its value at `t = 0` (an admixed start begins inside the cone; arm
A's electronic minimum at `t = 0` is about `eps * 0.1088344`); minimum
unshifted retained joint Gram and scaled positive extension and their
times; minimum boson-moment eigenvalue; maximum `|x_i|`; lift re-selections
and solver refusals; autonomous wall; peak resident memory. Reading rule in
numbers: at the smallest certifying weight, the admixture is reported as
costing nothing measurable on a row if that row differs from the record's
Hartree--Fock `exact_moments` row by less than 7.1 percent (the one measured
band on record, CR section 3.1); otherwise the difference is reported as its
cost.
**Stop.** A parity gate that fails: do not start that chain. A preparation
that certified in increment 2 but refuses inside the driver: stop that chain and report the difference (the driver's lower
moments differ from the probe's). A chain whose electronic one-body density
matrix falls below `-1e-8` or whose maximum `|x_i|` exceeds `10` before
`t = 20` is finished (or fails) and is reported; that is a result, not a
stop.
Commit `$DATA/rollouts_t20.json` (both scores and the physicality rows) and
the driver copies.

### Increment 5 — the report-back file and the final commit

**Bound.** The suite takes about `250` to `265 s`.

Write `$REPORTFILE` per C17 (report-back blocks; findings with the probe
table, the unrepaired-closure table and the rollout table beside the record's
rows; trajectory paths under `$OUT`; SHAs; the proposed worklog block), then:

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q
git -C "$WT" add agent_guidance/paper-v/handoffs/REPORT-regularized-product-state-codex-20261010.md agent_guidance/paper-v/handoffs/regularized-product-state-codex-20261010-data
git -C "$WT" commit -m "paper-v (Codex): regularized product-state initialization — <the answer with its numbers>"
```

**Acceptance.** The suite stands at the increment-1 baseline plus the new tests
with no new failure; nothing tracked is left modified.
**Stop.** A new test failure: fix it if it is yours, otherwise report it.

---

## 7. Evidence standards for this thread

* Reproduce the two endpoints (the Hartree--Fock zero-correlation
  preparation refuses as recorded; `correlated_admixture` at weight 1 matches
  the exact correlated preparation) before reporting any intermediate
  weight.
* A certification is the completion's own `success` at a named rung, read
  from the result object; never infer it from a later rollout running.
* Score every admixed rollout against both references; a score against only
  the admixed reference hides the bias, a score against only the
  Hartree--Fock reference hides the closure's error.
* The record's rows are single runs except the exact correlated `t = 4`
  band (`9.29e-3` to `9.95e-3`, 7.1 percent); differences smaller than that
  are unresolved (CR section 7).
* The exact frontier of a mixture is a reporting number only (C8).

## 8. Traps

* Mixing connected blocks or closed scalar coordinates linearly is wrong
  (section 3, fact 2); the linearity test of increment 1 guards it.
* Your worktree has none of the drivers or chains (C5); `run_chain.sh` in
  the source worktree hardwires `ROOT` there and the retired `ram_guard`, so
  increment 4 uses its own loop.
* The record's segment driver does not carry the Eq. (22) energy reference
  across a resume and re-solves the `t = 0` completion on every resume (C20).
  For an admixed preparation with the declared completion that re-solve
  would repeat a conic solve at every segment; the fixed copy takes the lift
  from `resume.npz` instead.
* `source_connected_stationary_state` (`initial_conditions.py:200`) is a
  scalar stationary solve seeded from exact contractions; it is not an
  admixture and must not silently replace the requested preparation
  (readiness audit, Increment 2 table).
* The record's "rank 6 of 8" may be wrong (section 3, fact 4); increment 0
  settles it before any admixture is designed around it.
* The exact correlated preparation itself needs the `logdet_shift 1e-8` rung
  at 27 keys (CR section 2(a)); an admixture that also needs a rung is not
  worse than the recorded route.
* `hartree_fock_closed_scalar_coordinates` (analytic) and the state-vector
  contraction differ by up to `1.02e-10`; the rollout uses the contraction.
* Chaining is not bitwise (CR section 7): a chained run agrees with a single
  rollout to `2.5e-14` through three re-selections and can flip a fourth.
* `run_divergence_segment.py` creates its output directory with
  `exist_ok=False`; the loop skips finished segments instead of rerunning
  them.
* zsh does not word-split an unquoted scalar; the commands use
  `$(echo ...)` where a list is passed.

## 9. Report back

`$REPORTFILE` per C17, committed on your branch. Its first lines state, one
sentence each: which admixtures certify, at what smallest weight and rung;
how far the admixture moves the unrepaired `1e4` event from `54.4612958`;
and, for the best certifying chain, the all-coordinate scalar RMS at
`t = 20` against both references beside the record's `1.5368e-1`.


---

## C. Common rules for the four Paper V Codex threads of 2026-10-10

These rules are identical in all four handoffs and in
`prompt-exports/codex-paper-v-INDEX-20261010.md`. The shell variables
(`$WT`, `$BRANCH`, `$OUT`, `$REPORTFILE`, `$DATA`, `$SRC`, `$PY`, `$REC`,
`$CHAIN`, `$LOCK`) are set by the `export` block in section 0 of the handoff
that contains this block; every command assumes that block was run in the
same shell. They also incorporate the Codex readiness audit of 2026-10-10,
`/Users/lordskyler/Holstein_test.worktrees/codex-paper-v-classical-readiness-20261010/agent_guidance/paper-v/handoffs/classical-campaign-readiness-20261010.md`
(its Increment 2 traps 1 to 6 and its dispatch gates 1 to 5), which you may
read as context.

**C1. Base, worktree, branch.** Work only in your own worktree, created from
the commit below; never in the source worktree and never in the main
checkout.

```bash
git -C /Users/lordskyler/Holstein_test worktree add -b "$BRANCH" "$WT" \
  b9f99271fc3ddf3db604d3acb64617a291bec7e1
git -C "$WT" rev-parse HEAD        # must print b9f99271fc3ddf3db604d3acb64617a291bec7e1
git -C "$WT" status --short        # must print nothing
```

`b9f99271` is branch `claude/paper-v-routes-20260914` with `main` at
`d1436ab1` merged in. That branch has since gained commits `a264ca23` to
`f4b52e06`: handoff, campaign and coordination documents, two reports, worklog
entries, and one new module with its tests
(`paper_5/src/paper5/stability/quantum_target_form.py`,
`paper_5/tests/test_quantum_target_form.py`, commit `d110f2dd`). None of
them changes a file these threads run or edit; you base at `b9f99271` as
stated, and read the two new reports from the source worktree (C4). The Paper V modules
are **not on `main`**: `paper_5/src/paper5/stability/cheap_projection_route.py`
does not exist there, and the branch is 67 commits ahead of `main`.

**C2. Staging and commits.** Stage with explicit paths only
(`git -C "$WT" add <path> <path> ...`). Never `git add -A`, `git add .`,
`git add -u` or `git commit -a`: another agent's broad `git add` once swept
uncommitted work into the wrong commit (incident `ba7f2ac9`, 2026-08-18).
Commit at every increment boundary with a message that states the result
with its numbers. Never push, merge, rebase or touch another branch.

**C3. Writable paths, inside `$WT` only.** `paper_5/**` **except**
`paper_5/notes/electron_phonon_closure_worklog_20260803.md`;
`agent_guidance/paper-v/**`; `output/local_runs/paper_v_*`. Everything else
is read-only. In particular do not edit: the worklog named above (C17 says
who does); `MATH/**` (the manuscript and the working notes; manuscript prose
is the user's); `agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`
(the coordinator updates it from your report); any existing report under
`agent_guidance/paper-v/overnight/`; any existing test assertion. **Shared
core modules are read-only for every thread** —
`paper_5/src/paper5/stability/matrix_reference.py` (it holds
`closed_scalar_rhs`, the HIGH-risk shared right-hand side),
`initial_conditions.py`, `cone_correction.py` — **and
`cheap_projection_route.py` is edited by exactly one thread, the electronic
cone projection thread**; the others import it unchanged. If the work
appears to need a path outside this list, stop and report: a conflicting
edit costs more than the delay.

**C4. Paths owned by other agents; do not create or edit them.**
`paper_5/src/paper5/stability/quantum_target_form.py` and its test
(committed on the source branch at `d110f2dd`, not in your base; do not
create files of those names);
`MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md`;
`agent_guidance/paper-v/overnight/closure-instability-20261010.md` (landed at
`40771435`): the Jacobian spectrum, finite-time Lyapunov exponents and
step-size convergence of the unrepaired archive closure are that report's.
Do not duplicate that analysis: event-time spreads under declared
perturbations are in scope, Jacobians and exponents are not. And the other
three Codex threads' paths:

| thread | branch | own output directory (under its own worktree) | report-back file (under its own worktree) |
|---|---|---|---|
| repair early, then bare | `codex/paper-v-repair-early-then-bare-20261010` | `output/local_runs/paper_v_repair_early_then_bare_codex_20261010/` | `agent_guidance/paper-v/handoffs/REPORT-repair-early-then-bare-codex-20261010.md` |
| regularized product-state initialization | `codex/paper-v-regularized-product-state-20261010` | `output/local_runs/paper_v_regularized_product_state_codex_20261010/` | `agent_guidance/paper-v/handoffs/REPORT-regularized-product-state-codex-20261010.md` |
| the electronic density cone as a projected set | `codex/paper-v-electronic-cone-projection-20261010` | `output/local_runs/paper_v_electronic_cone_projection_codex_20261010/` | `agent_guidance/paper-v/handoffs/REPORT-electronic-cone-projection-codex-20261010.md` |
| why the closure diverges from exact data | `codex/paper-v-divergence-parameter-map-20261010` | `output/local_runs/paper_v_divergence_parameter_map_codex_20261010/` | `agent_guidance/paper-v/handoffs/REPORT-divergence-parameter-map-codex-20261010.md` |

**C4a. Measured since `b9f99271`, bearing on every thread** (read-only at
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/agent_guidance/paper-v/overnight/closure-instability-20261010.md`;
the unrepaired archive closure at `lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude 1, phonon cutoff 16):

* from the exact correlated preparation at `t = 0` the `1e4` magnitude event
  is at `141.4057436` with the record's DOP853 `rtol 1e-10`, `atol 1e-12`
  (report line 103) and `141.405885` converged across RK4, DOP853 and Radau
  (line 222): the record's "never, through `t = 140`" (DI section 2.2) was a
  window `1.4` units too short;
* from the exact driven trajectory's state at `t = 60` (Hartree--Fock
  preparation) the event is `433.8964` at `rtol 1e-10`, `434.1811` at
  `1e-12` and `434.1878` at `1e-13`, converged to about `0.07`; an admissible
  perturbation of the starting state of size `1e-12` moves it by `7.1e-3`,
  one of size `1e-10` by `1.31` (report section 0, item 2);
* across 64 admissible perturbations of size `1e-4`, the elapsed time from
  `t = 60` to the event spans `293.1` to `478.1` units (median `351.7`) from
  the exact state and `296.1` to `779.1` (median `395.5`) from the repaired
  chain's state (report lines 299 and 300);
* the flow is volume-preserving (`Tr J` at most `6.1e-13` at every state),
  conserves the Eq. (22) energy to six decimals through the event, and has
  largest finite-time Lyapunov exponent `0.047` over its bounded phase
  (report section 0, item 3; lines 467 and 468).

So a single event time is meaningful to about one time unit against a
`1e-10` change of the starting state and to a few tenths against the
tolerance; read every event-time comparison at that resolution, and measure
it at your own starts where an increment says so.

**C5. Your inputs are absent from your worktree; read them from the source
worktree by absolute path, read-only, and lock them by SHA-256.** A fresh
worktree contains only tracked files, and `output/` is gitignored
(`.gitignore:50`), so every run artifact, stored trajectory, checkpoint,
exact-reference cache and driver script under `output/local_runs/` exists
only in the source worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`.
Never write there. Write every output of your own under `$OUT`
(`$WT/output/local_runs/paper_v_<topic>_codex_20261010/`). Each handoff's
appendix L lists every input it reads with its SHA-256, in `shasum` format;
the same list is the tracked file `$LOCK` in the source worktree. Increment 0
runs `shasum -a 256 -c "$LOCK"`, copies the drivers into `$OUT`, checks the
copies byte for byte against the lock, and writes `$OUT/input_manifest.json`
with the expected and the actual hash of every input and copy (readiness
gate 2). A path or a label is not a lock: a changed or missing input stops
the thread for re-anchoring.

| input, **read-only** | absolute path | what is inside |
|---|---|---|
| the record's divergence drivers and score files | `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/` | drivers `run_unrepaired.py` (bundles seven propagations; `AMPLITUDE_THRESHOLD`, `_closed_diagnostics`, `_retained_gram_minimum`), `run_closed_route.py` (one preparation, `--route archive` or `equation_112`; `--initial-from` = the **last** sample of a `trajectory.npz`), `boundary_basin_sweep.py` (the closure from each segment's last sample and from the exact state at the same time), `run_divergence_segment.py` (one cheap-repair segment; **uncommitted**, with the defects of C20), `run_chain.sh` (hardwires `ROOT` to the source worktree and wraps every segment in the retired `ram_guard`; do not run it), `hartree_fock_preparation.py` and `exact_lift.py` (import-only helpers that the segment driver imports as siblings), `probe_preparation.py`, `score_chain.py` (trims each segment at its resume time and drops repeated seam samples), `closed_horizon_table.py`, `exact_state_npz.py`; score files `boundary_basin_sweep.json` (140 units, both starts), `boundary_basin_span400_repaired.json`, `boundary_basin_span400_exact.json`, `unrepaired_t140.json`, `preparation_probe_hf.json`, `hf_chain_score_t20.json`, `hf_chain_score_t60.json`, `unrepaired_from_exact_t60.json` |
| the repaired chain from the Hartree--Fock zero-correlation preparation, `t = 0` to `60` | `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/` through `..._seg14/` | per segment: `trajectory.npz` with `times`, `archive_coordinates` (the 31 closed scalar coordinates of every accepted step, `h = 0.0025`), `exact_coordinates` (the exact driven reference from the same Hartree--Fock product state, same grid), `electron_minimum_eigenvalues`, `boson_moment_minimum_eigenvalues`, `retained_minimum_eigenvalues`, `scaled_extension_minimum_eigenvalues`, `internal_energy`, `internal_energy_reference`; `summary.json` (`rollout.autonomous_seconds`, `rollout.completed_steps`, `rollout.lift_reselections`, `rollout.lift_reselection_failures`, `physicality`, `accuracy`); `resume.npz` (`time`, `state` = the 56 propagated coordinates, `frontier_standardized` = the 70-coordinate lift, `internal_energy` = the state's Eq. (22) energy, **not** the carried reference) at the segment's last re-selection boundary, from which the next segment resumed; `reselections.json` (one record per re-selection: `seconds`, `decision`, `certified`, ...) |
| **the stored boundaries** | the **last sample** of each segment's `trajectory.npz`: `times[-1]` and `archive_coordinates[-1]` | boundary times `4.76`, `9.60`, `13.82`, `18.60`, `20.00`, `24.44`, `29.12`, `34.43`, `38.6325`, `43.12`, `48.48`, `52.71`, `56.92`, `60.00` (segments 1 to 14). Four segments end after their resume point (segment 3 ends at `13.82`, resumes at `13.80`; segment 8 at `34.43`, `34.40`; segment 9 at `38.6325`, `38.60`; segment 12 at `52.71`, `52.68`), so at those four boundaries the stored state lies on a short tail that the next segment re-propagates; the other ten end exactly at their resume time. `boundary_basin_sweep.py` uses the last samples; `score_chain.py` uses the resume-trimmed splice; these are two different conventions |
| the recorded exact-correlated task-11 chain, `t = 0` to `20` | `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1/` through `..._seg6/` | `trajectory.npz` with `times`, `states` (56), `frontier_standardized` (70), `retained_minimum_eigenvalues`, `scaled_extension_minimum_eigenvalues`, `internal_energy`, `internal_energy_reference`, `archive_coordinates`, `exact_archive_coordinates`; `summary.json` with `internal_energy.declared_reference_value`; segment 1 covers `[0, 3.955]` and resumes at `3.92`; segment 2 covers `[3.92, 8.3225]` |
| the task-11 drivers and the exact correlated reference cache | `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/` | `run_energy_segment.py` (the task-11 driver; **uncommitted**, same reference-carry defect as C20), `run_chain.py`, `score_energy_chain.py`, `probe_cone_correction_share_20260918.py`, `exact_t20.npz` (8,001 samples), `exact_t8.npz`, `task11_base_score_t20.json`, `task13_base_score_t20.json`, `cone_correction_share_base_w68_20260918.json` |

The committed drivers, for contrast: `paper_5/src/paper5/stability/cheap_projection_route_analysis.py`
carries the declared reference across resumes (`_resume_state`, line 426)
but runs the 15-key, 44-coordinate route from the exact correlated
preparation only; `apcm_projective_event_guard_analysis.py` is the
projective route. Neither runs the 27-key chain from the Hartree--Fock
zero-correlation preparation, which is why the threads that chain repaired
segments use a fixed copy of the uncommitted driver (C20).

**C6. Python, imports and the exact form of every command.** Interpreter
`/Users/lordskyler/Holstein_test/.venv/bin/python` (Python 3.12.14). Every
command runs from `$WT` with `PYTHONPATH=paper_5/src:$OUT`, so the Paper V
modules resolve into your worktree and the copied drivers (with their
sibling helpers) into `$OUT`. Absolute script paths alone do not pin the
imported `paper5`. Verify once per shell:

```bash
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -c \
  "import paper5.stability.cheap_projection_route as m; print(m.__file__)"
# must print $WT/paper_5/src/paper5/stability/cheap_projection_route.py
```

Create every output's parent directory first (`mkdir -p`); several of the
record's scorers do not. Never pass `--skip-reference` to a segment driver:
the scorers need populated exact histories.

**C7. Defaults and acceptance rules never change.** Every new behaviour is a
named, keyword-only option whose default reproduces the existing behaviour,
with a test that asserts the default path is **bitwise** identical
(`np.array_equal`, not `allclose`) to the behaviour at `b9f99271` on a
concrete input. Existing test files pass unchanged; new tests go in new test
files or new test functions.

**C8. Exact references are diagnostics only.** The exact driven reference
may score a trajectory and may supply the initial condition of a separate
diagnostic propagation, as the record's boundary sweep does
(`agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
section 13). It never enters a rollout's velocity, repair, lift
re-selection, acceptance rule or any switching decision. The record's single
exception, the `exact_moments` lift at `t = 0` from the Hartree--Fock
zero-correlation preparation (same file, section 3), is a declared diagnostic
substitution for one solve at `t = 0`; every rollout that uses it records
`lift = exact_moments` in its `summary.json`, and yours must too.

**C9. Memory: no guard, no cap.** The mandatory RAM guard is retired
(`agent_guidance/shared/memory-budget.md`, user revision 2026-09-19). Run
commands directly. Do not wrap anything in `pipelines/shell/ram_guard.py`,
do not invent a per-process memory kill threshold, and strip the
`pipelines/shell/ram_guard.py --limit-mb N --` prefix from every command you
copy from the record. Before each substantial launch run
`memory_pressure -Q` and look at running jobs
(`ps -o pid,rss,command -U "$USER" | sort -k2 -n | tail -15`), and size
concurrency by the actual headroom. The machine has 24 GB and 14 cores and
is shared with at least two Claude agents and the three other Codex threads;
four disjoint filesystem scopes do not authorize four simultaneous heavy
runs. Recorded sizes: one 27-key cheap-repair segment peaks at `593` to
`656 MiB` (`divergent-initialization-20260920.md` sections 4.1 and 9); the
unrepaired-closure boundary sweep peaked at `259 MB` (worklog block
"2026-09-20: the basin test at all fourteen stored boundaries"). Record the
peak resident memory of every run you report.

**C10. The 20-minute bound.** No single local run, and no first increment,
exceeds 20 minutes of wall time (the coordinator's limit, readiness gate 1;
a scheduling bound, not a memory cap). Longer scientific targets are later
increments resumed from a checkpoint, never a longer run:

* a repaired rollout runs as one segment per process with
  `--wall-budget-seconds 900`, resumed from the previous segment's
  checkpoint (C20), so preparation, propagation and its exact reference fit
  in 20 minutes;
* a sweep writes one JSON line per finished start to its output as it goes,
  takes `--max-wall-seconds 1080` and `--resume`, and is re-invoked with the
  same command until it reports itself complete;
* a start or a segment stopped by the wall budget is recorded as
  **incomplete** and is never read as "no event" or as reaching its horizon:
  a budget must not silently redefine a scientific stopping threshold.

**C11. Vocabulary.** Use the record's names (glossary C19). Never introduce a
name for a quantity. Every number you report carries, in the same sentence
or table caption, its definition, its configuration (preparation,
`lambda_ep`, `gamma`, drive amplitude, phonon cutoff, dictionary, integrator
and step, repair settings, tolerances) and its horizon or window. If a new
quantity is unavoidable, define it in a full sentence before any short
label.

**C12. GitNexus.** `AGENTS.md` requires impact analysis before editing a
function and `detect-changes` before a commit, and readiness gate 4 asks for
a current index of the execution worktree. Index your own worktree only:
`cd "$WT" && npx gitnexus@latest analyze --index-only` (or `bunx`), then
`node .gitnexus/run.cjs impact "<symbol>" --direction upstream --repo .` and
`node .gitnexus/run.cjs detect-changes --scope all --repo .` from `$WT`. If
indexing fails, takes more than 20 minutes, or would rewrite the main
checkout's index (`/Users/lordskyler/Holstein_test/.gitnexus`, which indexes
`main` at `d1436ab1` and does not contain most Paper V modules), record
impact as `UNKNOWN` (not low), confirm callers with
`rg -n "<symbol>" "$WT/paper_5"`, and list them in the commit message.

**C13. No UI.** No browser, Finder, `open`, `open -R`, image viewer or any
graphical action (root `AGENTS.md`, hard invariants). Plots, if you make
any, are files you do not open; report their paths.

**C14. Status of everything you produce.** Exploratory, not evidence. Write
`"classification": "exploratory_local_not_promoted"` into every summary JSON.
Adoption, promotion and demotion are the user's decisions; no manuscript or
working-notes edit.

**C15. Autonomy, acceptance and stop criteria.** Proceed through all
increments without pausing. Every increment states its exact command, its
time bound, an **acceptance** criterion and a **stop** criterion in numbers.
Acceptance met: commit and continue. Stop met: write the report-back with
the blocker and stop the thread; do not improvise a workaround. A surprise
is reported even when its increment is accepted; a surprise usually means an
assumption in this plan was wrong. Never reuse an output directory: every
run writes to a fresh path, a failed run's directory is left in place and
the retry gets a new suffix (`_retry1`, `_retry2`).

**C16. Test baseline.** Increment 1 measures it before any source change:
`cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q`
(about `250` to `265 s` on the record). The last full-suite count in the
record is `499 passed` in `257.04 s`
(`paper_5/notes/electron_phonon_closure_worklog_20260803.md:5835`, a
2026-09-16 block); `paper_5/tests/test_cheap_projection_route.py` was
`53 passed` in `22.9 s` at `3f51146a`
(`agent_guidance/paper-v/overnight/cheap-projection-20260915b.md:4839`), and
`3f51146a` is still the last commit touching `cheap_projection_route.py`.
Tests added after 2026-09-16 were not counted in the record; the Codex
readiness audit measured `508 passed` at its own base (coordination record,
commit `81e2f2bb`). Your measured count is the baseline. Record any pre-existing failure and do not fix it. At
every commit the suite stands at that count plus your new tests, with no new
failure.

**C17. Report-back ownership and durable outputs.** You own one report file,
`$REPORTFILE` = `agent_guidance/paper-v/handoffs/REPORT-<topic>-codex-20261010.md`
in **your** worktree, committed on **your** branch. It holds, in order: the
contract's report-back block for every increment
(`agent_guidance/shared/agent-handoff-contract.md` section 6); the findings,
negative result first, one table per increment, every number with
definition, configuration and horizon; for every chained run, the Eq. (22)
energy reference value at the start and end of each segment (C20); the
absolute paths of every trajectory under `$OUT`; every commit SHA; and a
**proposed worklog block** in the shape below. You do **not** edit
`paper_5/notes/electron_phonon_closure_worklog_20260803.md`: the Claude side
reviews your report and merges the worklog entry, which avoids a conflict
between four branches on that file. Commit code under
`paper_5/src/paper5/stability/` with tests under `paper_5/tests/`; small
result JSON files (each under 1 MB) and `input_manifest.json` under `$DATA`;
and a copy of every driver you edited in `$OUT` under `$DATA/drivers/`,
because `$OUT` is gitignored and an uncommitted driver cannot be reviewed or
rerun. Trajectories stay under `$OUT`.

```markdown
## Increment N — <goal>
Status: done | blocked | skipped
Commands run: <exact>
Result: <numbers, decisions, diffs>
Verification: <test counts, run outputs, before/after values>
Deviations: <anything done differently, and why>
```

Proposed worklog block (the worklog's headings are
`## YYYY-MM-DD: <finding stated as a sentence>`):

```markdown
## 2026-10-DD: <the finding, stated as a sentence with its number> (Codex, branch <BRANCH>)

Branch `<BRANCH>`, base `b9f99271`, commits `<sha> ...`.  Exploratory local
work, not promoted evidence; no default or acceptance rule changed.  Report
`<REPORTFILE>`; data `<DATA>`; trajectories under `<OUT>`.

<The negative result first, one sentence with its number.  Then the measured
table or sentences, every number with definition, configuration and horizon.
Then one paragraph headed "Reading."  Then "Next:" naming the one cheapest
test the result points to.>
```

**C18. Campaign context only.** The campaign plan is
`prompt-exports/campaign-plan-codex-claude-gpt-20261010.md` in the source
worktree (tracked copy
`agent_guidance/paper-v/handoffs/campaign-plan-codex-claude-gpt-20261010.md`,
commit `a264ca23`). It is context, not an instruction source, and this
handoff is executable without it.

**C19. Glossary: the record's names, which are the only names to use.**

| name | definition | where defined |
|---|---|---|
| the 31 closed scalar coordinates | the archive closure's state: electronic one-body density matrix (3), coherent phonon amplitudes (4), phonon covariance (4), anomalous covariance (6), electron--phonon correlation (14) | `CLOSED_SCALAR_STATE_NAMES` (`paper_5/src/paper5/stability/matrix_reference.py:32`); `agent_guidance/paper-v/overnight/scaling-dimension-growth-20260920.md` section (a) |
| the unrepaired archive closure | `closed_scalar_rhs` (`matrix_reference.py:919`) propagated in the 31 closed scalar coordinates with no repair, no lift and no re-selection; the user's brief calls it "the bare closure" | `divergent-initialization-20260920.md` sections 2.2 and 13 |
| the `1e4` magnitude event | the terminal event at which the largest absolute value among the 31 closed scalar coordinates reaches `1e4` (`AMPLITUDE_THRESHOLD` in `run_unrepaired.py`) | same, section 2 |
| electronic one-body density matrix minimum eigenvalue; electronic positivity loss | `closed_electron_eigenvalues(x)[0]` (`initial_conditions.py:143`); positivity loss is its first crossing below `-1e-8`, reported beside the first crossings below `0` and below `-1e-3`, linearly interpolated between samples (`_first_below` in `run_divergence_segment.py`) | same, section 2 |
| minimum unshifted retained joint Gram eigenvalue | minimum eigenvalue of `uncentered_joint_moment_matrix`, the 8-by-8 Gram of `(1, b, b^dagger, sigma)` (`adaptive_positive_moment.py:458`) | `MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md` section 3.2 |
| minimum scaled positive extension eigenvalue | minimum eigenvalue of the cheap repair's scaled positive extension (36 rows at 27 entrance keys); undefined for a trajectory propagated in the 31 closed scalar coordinates, and never invented for one | same; `divergent-initialization-20260920.md` section 8 |
| minimum boson-moment eigenvalue | `closed_boson_moment_eigenvalues(x)[0]` (`initial_conditions.py:165`) | `divergent-initialization-20260920.md` section 2.2 |
| maximum `\|x_i\|` | the largest absolute value among the 31 closed scalar coordinates over the stated horizon | same |
| all-coordinate scalar RMS | `sqrt(mean(d**2))` over samples and the 31 closed scalar coordinates against the exact driven reference from the same preparation, cumulative from `t = 0` on the `h = 0.0025` grid (`_accuracy_metrics`, `apcm_carried_witness_analysis.py:259` at `b9f99271`) | `cheap_repair_route_results_20260918.md`, preamble |
| Eq. (22) internal energy; the Eq. (22) energy reference; Eq. (22) energy RMS against the reference | `matrix_total_energy` (`matrix_reference.py:757`); the declared value the repair's hyperplane holds the internal energy on (`energy_reference_mode`, `drive_power` in the task-11 configuration); the sample RMS of the internal energy's difference from the exact reference's over the horizon | same, sections 2(d) and 3.2 |
| lift re-selections, solver refusals, autonomous wall | from each segment's `summary.json`: `rollout.lift_reselections`, `rollout.lift_reselection_failures`, `rollout.autonomous_seconds` | `divergent-initialization-20260920.md` section 4.1 |
| the Hartree--Fock zero-correlation preparation, the divergent initialization | electronic one-body density matrix `[[1/2, 1/2], [1/2, 1/2]]`, every other coordinate zero except the coherent displacement `-g / omega_ph = -1.2247449` on both local oscillators; a pure product state | same, section 1 |
| the exact correlated preparation | the exact zero-field ground state at phonon cutoff 16 contracted into the closure's coordinates (`exact_holstein_joint_moment_initial_state`, `exact_ground_closed_scalar_coordinates`) | `cheap_repair_route_results_20260918.md`, preamble |
| the exact driven reference from the same preparation | the truncated Hamiltonian propagated with DOP853 at `rtol 1e-9`, `atol 1e-11`, maximum step `0.01` from the same state (`_reference_coordinates` in `run_divergence_segment.py`) | `divergent-initialization-20260920.md` section 4 |
| the cheap repair in the task-11 configuration | 27 entrance keys (`entrance15_plus_two_spin_boson`): 56 propagated state coordinates, 36 extension rows, 70 frontier keys, 41 lower keys; `CheapProjectionRoute.unguarded_velocity` with SSPRK(3,3) at `h = 0.0025`; the declared proximal log-det completion re-selected every 16 accepted steps at `conic_tolerance 1e-9`; the Dykstra repair (`policy="dykstra"`, `target="both"`, `cadence="stage"`, `dykstra_iterations=120`, all floors `0`) onto the affine Gram sets and the two cones; the Eq. (22) reference in `drive_power` mode; the anchor-free measured-candidate acceptance rule at margin `1e-3`, no trust radius; `lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon cutoff 16 | `cheap_repair_route_results_20260918.md` section 2; `divergent-initialization-20260920.md` section 4 |

**C20. Chained repaired segments: the driver fix and the parity gate.** The
record's 27-key chains ran through uncommitted drivers with two measured
defects (readiness audit, Increment 2 traps 1 and 2):
`run_divergence_segment.py` creates its output directory with
`exist_ok=False` **before** preparation (line 371), so a failed preparation
leaves a path that cannot be reused, and rebuilds the `t = 0` preparation
and its completion even on a resume (lines 402 to 426); and its integration
call (lines 455 to 467) does **not** pass the current integrator's
`energy_reference_value`, while its `resume.npz` (lines 617 to 626) stores
the state's `internal_energy` rather than the carried reference. The
task-11 driver `run_energy_segment.py` likewise does not pass
`energy_reference_value` on resume. The committed integrator does accept a
resume reference (`integrate_cheap_projection_ssprk3(..., energy_reference_value=...)`,
`cheap_projection_route.py:1856` and `2070` to `2076`), pinned by
`paper_5/tests/test_cheap_projection_route.py:782-838`. The audit measured
no numerical mismatch, because at a repaired boundary the stored internal
energy equals the reference to roundoff (the hyperplane holds it to
`1.78e-15`, CR section 2(d)), but the gap is real.

**Decision: fix your copy of the driver as a named change in your own
worktree, using the committed integrator's resume path.** In
`$OUT/run_divergence_segment.py` (and, in the thread that uses it,
`$OUT/run_energy_segment.py`):

1. create the output directory only after preparation succeeds, and refuse
   an existing one; never reuse a directory (C15);
2. on `--resume-from DIR`, do not rebuild the `t = 0` completion: take the
   state and lift from `DIR/resume.npz`, and take the carried Eq. (22) energy
   reference as the value of `internal_energy_reference` in
   `DIR/trajectory.npz` at the sample whose time equals `resume.npz`'s
   `time` (to `1e-9`), or from `DIR/resume.npz`'s `energy_reference_value`
   when the previous segment was written by the fixed driver;
3. pass it as `energy_reference_value=` to
   `integrate_cheap_projection_ssprk3`, and print
   `carried Eq. (22) energy reference <value> from <DIR> at t=<time>`;
   with `--energy-reference-mode none` print `none`;
4. write `energy_reference_value` into the new `resume.npz`, and
   `energy_reference_at_segment_start`, `energy_reference_at_resume_point`
   and `energy_reference_at_segment_end` into `summary.json`.

Copy the fixed driver to `$DATA/drivers/` and record its SHA-256 in
`input_manifest.json`.

**The parity gate.** Before any chained run at a given setting, run the same
short rollout twice at that setting, each command under 20 minutes: once
unbroken, and once as two segments spliced at an accepted resume time
(a re-selection boundary). Splice with the `score_chain.py` convention
(truncate the first segment at its resume time, drop the second segment's
first sample). Accept when: the spliced times are strictly increasing and
equal the unbroken run's times; the second segment's first state equals the
first segment's state at the resume time bit for bit; the carried reference
printed by the second segment equals the first segment's
`internal_energy_reference` at the resume time to `1e-14`; and the spliced
31 closed scalar coordinates match the unbroken run's to `1e-12` at every
sample. Stop when the coordinates differ by more than `1e-10` anywhere or a
reference differs by more than `1e-12`, and report the first differing
sample.

---

## L. Input lock (SHA-256 of every ignored input this thread reads)

Generated on 2026-10-10 from the source worktree at `b9f99271`. Identical to
the tracked file
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/agent_guidance/paper-v/handoffs/codex-paper-v-regularized-product-state-20261010-inputs.sha256`
(`$LOCK`), which increment 0 checks with `shasum -a 256 -c "$LOCK"`. If that
file is missing, save the block below to `$OUT/input_lock.sha256` and check
that instead. Any line that does not print `OK` stops the thread for
re-anchoring (C5). The record's JSON files carry no hashes of their own
inputs; this lock is the receipt.

~~~ sha256
0bee5a694cac0b77fcefe52b596dcae1dfb17bfbb7ef28129450d53ac88a6c51  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/boundary_basin_sweep.py
fecda1c8fd0eeb119469fe9290cd13f7ed505b9c4827f8c43c7f8162c05f2cf7  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/closed_horizon_table.py
14f5fe4468e7027dd97aded7460a156f1b3c9c06f2c8b23961f6fb3513973cdf  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/electron_trace.py
d18bcf1461e19d991f95739d07c599688c6e0419e4a8c1b13fca09ccda17ce5f  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/exact_lift.py
a3c4bcdaf86e1ad0561e7b5dee28e57a92b3e26d6e82d739e8e74dd82e4c7503  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/exact_state_npz.py
83f345acbaede63f6f648e549de88f78996b1fa186c15048b311069e072fecac  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/hartree_fock_preparation.py
1f4146c76f960090b8ccdfdc7d5ef2b064f2531a722c3eca51bf895f2f9c924b  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/magnitude_argmax.py
2f7cfc2695bd21feb1c40fa03e1ea40ef1c4de8f7f262fe350a8ec4c1c2b1bdd  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/make_resume.py
adba4d9f15ca75ca5e83d74ff0d3318f487079601f2c5cbf1a1a5feae17b162b  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/probe_diagnostic_cost.py
32b607706da203cedc071340731c27f2bcebf30b46b8f381995625daf7f91d1e  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/probe_preparation.py
4d862b78df6e77a0b803081b08e0a9b56efbc2a0627763e04e86f99852783cab  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_closed_route.py
2b59070ca5271d7f5b2406b47227d7f97898e47a9f93eb59a4d0715add8d36ed  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_cone_route.py
0e2636f739f60b414ed46ac88c4e883e8881a9d9ce2937ba05408b0a5ba483e9  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_divergence_segment.py
16402222cc335a651579d09188d337a4f4705691690bc1b3cc24ec89cf6e9217  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_unrepaired.py
ff2145c564db786988598f7737f523eb47d1dbff71dcdf47015ab6f124e52086  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/score_archive_routes.py
a6fdfdbeccf8ee34e1e9c4a556012b065199408eee72c26af853c90efd97d253  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/score_chain.py
0d7b5dfe5d74ef992adc531921dcd714ee459d8f402f77d9c4b3c13ce42883c2  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/score_trajectory.py
3de62394da4e03f73b91a09ad77531b89f3284897576866b1d6f0583663eca37  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/tabulate_boundary_basin.py
2d28f884c6fe15f3c23520fc7a611f8b9ca7b77ee3da163fb36031083c3d1de8  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/tabulate_chain.py
af3d6a830a84fa75d854e70f59177e82592f77496b4167d89bafb1fedcf69cf3  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_chain.sh
eda0aaae3a96318b5212a362c759909054346fd632f32a760b4e6da2fad65821  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_cone_chain.sh
873465a161d6d3cda7038dff539b43d8d9f627a9e777fb1862a0207adbf83832  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/boundary_basin_sweep.json
057975e5cc84714dd5a5c96bc649d3491e0e9f7f23da2a6637c8d0ee8481141a  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/boundary_basin_span400_repaired.json
70f0eeb9391a2b799effa94e8c67daa5d1846b4de4c3dcc7be4eadc2e7d12366  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/boundary_basin_span400_exact.json
bef62e4d9d83ece06e8c4af0e33756c4f7f917c872ff0219997e81d0df1fdd37  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/unrepaired_t140.json
7cc2025ffb20f46b11b3fb196c2f862f871afa728546495a6d6ba374333e2b86  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/preparation_probe_hf.json
e81b84d3b0b997ee0cba6698054350f3da16c26cd2091ed226378687417db987  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/hf_chain_score_t20.json
aa306f31d032c4c67acde2fdde99f55b24ae1bbcddf947d29f74ca967a1214be  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/hf_chain_score_t60.json
0e242ec2efb341e01088a5c33386b36acbde20dc97693a5c2c63a00655a16520  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/unrepaired_from_exact_t60.json
9f7eefcf015b8d21929b516ac75a58a759f1dd3ef8fd0157054d96f4a82b8deb  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/trajectory.npz
5b83103d03566a1094077921ae6db02bc15f7b7d2a2b02ba3435d3c324611062  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/trajectory.npz
a64012362e1be6024dd5044ce29af5aaf1707a069e1bab9f016813765bb15780  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/trajectory.npz
6af709eccb04d2dfbd27f9f646c50566ae345bc978649cff2953e496e400c62d  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/trajectory.npz
ec1d0c3feb73f24e44b9bc7b0efd27a690e4a7e6db0fb2c785048785489149f9  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/trajectory.npz
a359ff7f2bee7e08f7d1b6b78cf68d39842c401c669bfbb62a32dd7ca9495def  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/summary.json
71cffcb00b5b282b0377f4b4b76168dc00af9738aa51afff42fa218104c125dd  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/summary.json
84f7792cb6496f751500e7621dd7b16cb318f8ebc3b371c8bf1a81eef5114b8e  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/summary.json
8097edb45718a75748943b3f736e3825fb64cd75adec1064f880202a86deb0c0  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/summary.json
ce9a4239181a4b59f4d09740c32a188327060fc1b3a6e99be035511fd647646e  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/summary.json
53894713c5da93e553dc8d3577e6005d5f3d44713721fca5869d18e634b9e0c9  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/resume.npz
182a1f10b74b6c3a2151a6fc10ddf4ba3a2009ec4c355c1a11d0c861e21c63f1  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/resume.npz
b60f6d0b5f2c205dc5dcbe63bdd70d70e59f40973ed618017f71edebc7a01e58  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/resume.npz
6de343565a4e3b30770c6a3002e4369155925ad1ba7fef17dff2efad52a2268c  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/resume.npz
52b7531afb58d9fce3ed195d938e1dd5c6065a5e35426aee55f2140879f99965  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/resume.npz
6871f6cf30c2ac9defd3e8ab54478cd60cb96269fbda67acc9be5aa0d5be0540  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/reselections.json
08d183dee9247bd0667a2f73cb16ecc56a5f6270ae1085767aae59efc1a06112  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/reselections.json
39d43c853f5321527283e10f4d779bdb4b1a8b3847bdfbc1691eaa65ac725414  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/reselections.json
7544dfb150593bb8a0d43f13460918a6e12d71faa11b6c79e6f212e0d08858db  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/reselections.json
bbac7abe19050736fa98cd64e0c06554b93600f14c663876d149ed246afbb48d  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/reselections.json
~~~
