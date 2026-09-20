# The cheap repair on the initialization that actually diverges

**Round of 2026-09-20, branch `claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `5ac501d0`.**
Exploratory local work, not promoted evidence; no default, acceptance rule or
module was changed, and nothing here is manuscript prose.

## 0. Why this round exists

Every run of the 2026-09-14 to 2026-09-19 rounds prepared its state from the
**exact correlated** ground-state contractions
(`exact_holstein_joint_moment_initial_state`, phonon cutoff 16). The record's
own divergence, the one the repair of the archive moment closure exists to
fix, is the **Hartree--Fock electronic density matrix with every other
correlation variable set to zero** — the initialization
`stability_quantum_workflow_working_notes.md` reports as losing electronic
positive semidefiniteness at `t = 15.5177` and reaching state magnitude `1e4`
at `t = 130.47`, and which the same notes record as diverging where the exact
correlated initialization does not. The cheap repair had never been run from
it.

## 1. The divergent initialization and protocol, stated

| item | value | where the record states it |
|---|---|---|
| electronic initialization | Hartree--Fock ground state of the hopping-only electronic Hamiltonian; one-body density matrix `[[1/2, 1/2], [1/2, 1/2]]`, eigenvalues `0` and `1` | working notes, "Evidence for sensitive initial conditions" (Figure 34 of the Hubbard-dimer working document); `hartree_fock_zero_correlation_state` (13 scalar coordinates), `hartree_fock_closed_scalar_coordinates` (31 closed scalar coordinates) |
| every other coordinate | zero, except the decoupled centre mode at its equilibrium coherent displacement `-g / omega_ph = -1.2247449` on both local oscillators | `initial_conditions.py:116` |
| coupling | `lambda = 1.5`, `gamma = 0.5`, so `omega_ph = 0.5`, `g = 0.6123724` | working notes, "Matched scalar initialization results at lambda = 1.5" |
| drive | the declared pulse at amplitude 1, `2 A sin(pi t / 4) exp(-t^2 / 2)`, i.e. `v / t_hop = 1` | `DimerParameters.drive_difference` |
| integrator and tolerances used here | DOP853, `rtol 1e-10`, `atol 1e-12`, samples every `0.005`, terminal event at `max_i |x_i| = 1e4` | this round |

**The same state in the Fock space.** The divergent initialization is a pure
product state: the Hartree--Fock determinant times one coherent state per
local oscillator at `-g / omega_ph`. Built as a state vector in the same
truncated Fock space the exact preparation uses and contracted by the same
private helpers of `exact_reference.py`, it reproduces
`hartree_fock_closed_scalar_coordinates` to `1.02e-10` in maximum absolute
coordinate difference, the residual being the coherent-state truncation at
cutoff 16 (discarded weight `6.74e-13`). That is what lets the adaptive
positive-moment preparation of section 3 be given the recorded divergent
initialization and not an invention
(`output/local_runs/paper_v_divergence_20260920/hartree_fock_preparation.py`).

Its Eq. (22) internal energy is `-3.5` against the exact correlated
preparation's `-3.775326553`; its archive-closure initial residual
`||F(x_0, 0)||` is `0.5303300859` in the 31 closed scalar coordinates and
`0.3423265984` in the 13 (the record's figure for the latter is
`0.3423265984`).

## 2. Task (a): the unrepaired equations from that initialization

Driver
`output/local_runs/paper_v_divergence_20260920/run_unrepaired.py`, artifact
`output/local_runs/paper_v_divergence_20260920/unrepaired_t140.json`,
wall `25 s`.

Two quantities are reported at three floors because the divergent preparation
is a **pure** product state and therefore sits exactly **on** both cone
boundaries at `t = 0`: its electronic one-body density matrix is idempotent
(minimum eigenvalue exactly `0`), its phonon fluctuation matrix is exactly
zero, and its retained joint Gram is exactly singular (`-1.07e-15`). A bare
"first negative" time is not informative there, so the first time below `-1e-8`
and below `-1e-3` are given beside it.

### 2.1 The record's own case, reproduced

| quantity, 13-coordinate Fan--Migdal system from the Hartree--Fock/zero-correlation state | record | this round |
|---|---:|---:|
| first electronic positive-semidefiniteness loss | `15.5177` | `15.5177447` |
| state magnitude reaches `1e4` | `130.4674620` | `130.4674620` |
| initial residual norm | `0.3423265984` | `0.3423265984` |

The harness was also checked against the record's Eq. (112) row, which the
record quotes for **exact ground-state contractions** and not for the
divergent initialization: bounded through `t = 400` with maximum `|x_i|`
`1.8592243` (record `1.8592`), minimum electronic eigenvalue `0.0251665`
(record `0.0252`) and minimum retained phonon eigenvalue `2.9068e-5` (record
`2.9e-5`).

### 2.2 The 31-coordinate archive closure, which is what the cheap repair propagates

`closed_scalar_rhs`, same protocol. The retained joint Gram here is the
8-by-8 Gram of `(1, b, b^dagger, sigma)`,
`uncentered_joint_moment_matrix`; the boson-moment cone is
`closed_boson_moment_eigenvalues`.

| quantity | from the divergent Hartree--Fock state | from the exact correlated state |
|---|---:|---:|
| electronic minimum eigenvalue at `t = 0` | `0` (idempotent) | `0.1088344` |
| first electronic eigenvalue below `-1e-8` | `27.9633944` | `1.0362132` |
| boson-moment minimum eigenvalue at `t = 0` | `0` | `2.893143e-5` |
| first boson-moment eigenvalue below `-1e-8` | `1.3226952` | `3.6820771` |
| retained joint Gram at `t = 0` | `-1.073e-15` | `7.228147e-6` |
| first retained joint Gram below `-1e-8` | `0.0124346` | `0.1607047` (record `0.1607116782`) |
| state magnitude reaches `1e4` | **`54.4612958`** | never, through `t = 140` (maximum `24.2421`) |
| maximum `|Tr C[q]|`, a moment that vanishes on every state | `5151.44` | `10.57` |

**The initialization dependence the record reports survives into the 31
closed scalar coordinates**: the divergent initialization reaches the `1e4`
magnitude threshold at `t = 54.46`; the exact correlated one does not reach it
by `t = 140`. In both, representability is lost long before amplitude: from
the Hartree--Fock state the retained joint Gram is below `-1e-8` at
`t = 0.0124`, the boson-moment cone at `t = 1.3227`, and the electronic cone
at `t = 27.96`, against magnitude `1e4` at `t = 54.46`.

## 3. The cheap repair cannot be prepared at that initialization

**Stated first because it is the round's decisive negative result.** The
route's one conic solve is the declared proximal log-det completion at
`t = 0` (`cheap_repair_route_results_20260918.md` §2(b)). At the divergent
preparation it refuses, at **both** dictionaries and at **every** rung of the
declared completion retry ladder, always with the same message:

> `moment_extension_degeneracy: facial reduction removed every frontier direction`

| dictionary | rungs tried | outcome | seconds per attempt |
|---|---|---|---:|
| 15 entrance keys | `logdet_shift` at `1e-10` (committed), `1e-8`, `1e-7`, `1e-6`, `1e-4`, `1e-2`, `1e-1`; `logdet_weight` `1e-1` | every one refuses | `0.4` to `0.5` |
| 27 entrance keys | the same eight rungs | every one refuses | `2.2` |

The reason is structural, not numerical. The divergent initialization is a
**pure** product state, so its retained joint Gram is exactly singular: its
two smallest eigenvalues are `-4.44e-16` and `-4.21e-16` against a largest of
`8.405`, i.e. rank 6 of 8. Facial reduction onto that face leaves the
completion no free frontier direction, and the log-determinant shift — which
inflates the completion cone — does not restore one, because the reduction
happens on the lower moment matrix before the barrier is seen. The same
preparation path succeeds at the exact correlated state, whose retained joint
Gram minimum is `+7.23e-6`.

Artifacts: `output/local_runs/paper_v_divergence_20260920/probe_preparation.py`,
`preparation_probe_hf.json`, `preparation_probe_hf.log`.

**What was substituted, and why it is the honest substitute.** Because the
preparation is a pure state of the truncated Fock space, every frontier
moment the completion would have chosen is an *expectation of a known
operator in that state*; the extension's frontier keys reach total degree 5,
so `moment_hierarchy(5)` contracted through the same private helpers of
`exact_reference.py` supplies all 70 of them exactly. The rollouts of
section 4 therefore use **the exact moments of the divergent preparation's
own state vector as the initial lift**, and nothing else about the route is
changed: the propagated coordinates, the velocity, the Dykstra repair, the
Eq. (22) hyperplane, the re-selection cadence and the acceptance rule are the
committed ones. Every rollout records `lift = exact_moments` in its
`summary.json`. This is a diagnostic substitution for one solve at `t = 0`,
not a proposed method change, and the exact lift is a reporting input at
`t = 0` only — no exact quantity enters the propagation afterwards.

The re-selections *after* `t = 0` do not have this problem: the first one, at
step 16 of the `t = 0` to `0.05` smoke rollout, certified and moved the
retained joint Gram from `-4.10e-6` to `-5.79e-8`. Only the preparation
refuses.

The exact lift is a genuine positive extension of the divergent preparation,
which is what the refusing completion could not supply: at `t = 0` the
unshifted retained joint Gram is `-4.44e-16` and the scaled positive
extension is `-1.58e-15`, both at machine zero, exactly on their cones as a
pure state must be.

## 4. Task (b): the cheap repair from the divergent initialization

Configuration: the task-11 configuration of
`cheap_repair_route_results_20260918.md` §2 --- 27 entrance keys
(`entrance15_plus_two_spin_boson`), 56 propagated state coordinates, 36
extension rows, 70 frontier keys, 41 lower keys; `unguarded_velocity` with
SSPRK(3,3) at `h = 0.0025`; the declared proximal log-det completion
re-selected every 16 accepted steps at `conic_tolerance 1e-9`; the Dykstra
repair onto the affine Gram sets and the two cones, `policy = "dykstra"`,
`target = "both"`, `cadence = "stage"`, `dykstra_iterations = 120`, all three
floors `0`; the Eq. (22) energy reference in `drive_power` mode; the
anchor-free measured-candidate acceptance rule at margin `1e-3`, no trust
radius. `lambda = 1.5`, `gamma = 0.5`, drive amplitude 1, phonon cutoff 16.
The only difference from the recorded chain is the preparation (`hartree_fock`)
and, forced by section 3, the initial lift (`exact_moments`).

**The exact reference the divergent initialization admits.** It admits one:
the same truncated Hamiltonian, the same DOP853 settings the recorded
reference uses (`rtol 1e-9`, `atol 1e-11`, maximum step `0.01`), propagated
from the same Hartree--Fock product state vector. Scores below are the
all-coordinate scalar RMS of the 31 closed scalar coordinates against it,
cumulative from `t = 0`, on the `h = 0.0025` grid. The rollout's `t = 0`
coordinate offset against that reference is exactly `0`.

### 4.1 It reaches `t = 20` and it holds the electronic cone

Five chained segments, `output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg{1..5}`
(`t = 4.76`, `9.6`, `13.82`, `18.6`, `20.0`; walls `1002.2`, `1003.2`,
`1000.0`, `1001.3`, `290.2 s`), **8,001 samples on one `h = 0.0025` grid,
`4,296.9 s` autonomous wall, 500 lift re-selections of which 44 were solver
refusals, peak resident `654 MiB`**. Score file
`output/local_runs/paper_v_divergence_20260920/hf_chain_score_t20.json`.

| horizon | all-coordinate scalar RMS | Eq. (22) energy RMS against the reference | minimum unshifted retained joint Gram | minimum scaled positive extension | minimum boson-moment eigenvalue | maximum `\|x_i\|` |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `1.9079e-3` | `4.502e-5` | `-6.308e-6` | `-8.852e-7` | `-2.522e-5` | `1.2309` |
| `2` | `3.6436e-3` | `9.240e-4` | `-6.308e-6` | `-8.852e-7` | `-2.522e-5` | `1.3166` |
| `4` | `1.0803e-2` | `1.0543e-3` | `-7.450e-6` | `-8.852e-7` | `-2.522e-5` | `1.5894` |
| `8` | `3.7209e-2` | `1.1402e-3` | `-4.370e-5` | `-2.324e-6` | `-6.711e-5` | `1.8976` |
| `12` | `6.0626e-2` | `1.1675e-3` | `-5.561e-4` | `-6.352e-5` | `-2.163e-3` | `1.8976` |
| `16` | `1.3055e-1` | `1.1808e-3` | `-5.561e-4` | `-6.352e-5` | `-2.163e-3` | `1.8976` |
| `20` | `1.5368e-1` | `1.1888e-3` | `-5.561e-4` | `-6.352e-5` | `-2.163e-3` | `1.9431` |

**The electronic one-body density matrix never loses positive
semidefiniteness.** Its minimum eigenvalue is `-2.220e-16` at `t = 0` — the
preparation is a determinant, exactly on the cone — and its minimum over
`0 < t <= 20` is `+1.447e-6`, attained at the first step `t = 0.0025`; the
largest it reaches is `0.2150`. The record's `t = 15.5177` and this round's
31-coordinate `t = 27.9634` are both inside the horizon and neither occurs.
**The state magnitude stays bounded**: maximum `|x_i| = 1.9431` through
`t = 20`, against `1e4` at `t = 54.4613` unrepaired.

**Neither Gram quantity stays at its floor.** The retained joint Gram leaves
`-1e-8` at `t = 0.0025` and reaches `-5.561e-4`; the scaled positive extension
reaches `-6.352e-5`. This is the same finding the record reports for every
constrained chain — no chain with the Eq. (22) hyperplane on holds a cone past
`t = 8` — and the divergent initialization does not change it.

### 4.2 Beside the recorded chain from the exact correlated preparation

The recorded task-11 chain is the same configuration at the same horizon, the
only differences being the preparation and the `t = 0` lift
(`output/local_runs/paper_v_projective_dictionary_energy_20260915e/task11_base_score_t20.json`).
Its score is against the exact reference started from *its* preparation, this
round's against the exact reference started from *its* preparation, so the two
columns are each the distance from the trajectory the closure is approximating.

| quantity, `t = 0` to `20` | divergent Hartree--Fock, exact lift | recorded exact correlated, declared completion at `logdet_shift 1e-8` |
|---|---:|---:|
| all-coordinate scalar RMS at `t = 4` | `1.0803e-2` | `9.290e-3` |
| the same at `t = 8` | `3.7209e-2` | `3.304e-2` |
| the same at `t = 20` | `1.5368e-1` | `5.604e-2` |
| site-0 occupation RMS at `t = 20`, `rho[0,0]` | `0.0736` | `0.0681` |
| Eq. (22) energy RMS against the reference at `t = 20` | `1.1888e-3` | `5.336e-3` |
| minimum unshifted retained joint Gram | `-5.561e-4` | `-4.460e-4` |
| minimum scaled positive extension | `-6.352e-5` | `-3.936e-5` |
| lift re-selections / solver refusals | `500` / `44` | `500` / `117` |
| autonomous wall | `4,296.9 s` | `5,183.6 s` |

The divergent initialization costs `16` percent of accuracy at `t = 4` and
`2.74` times at `t = 20`, holds its two Grams within `25` and `61` percent of
the recorded chain's depths, and carries **one quarter** of its Eq. (22)
energy error and `38` percent of its solver refusals.

## 5. Task (c): the alternatives, same initialization, same horizon

All rows `t = 0` to `20`, `lambda = 1.5`, `gamma = 0.5`, drive amplitude 1;
the unrepaired and Eq. (112) rows DOP853 at `rtol 1e-10`, `atol 1e-12`,
samples every `0.005`. The last two rows are the same two vector fields from
the **exact correlated** preparation, as controls.

| vector field, preparation | maximum `\|x_i\|` | minimum electronic eigenvalue | minimum boson-moment eigenvalue | minimum retained joint Gram | `\|x_i\| = 1e4` at |
|---|---:|---:|---:|---:|---:|
| unrepaired archive closure, Hartree--Fock | `2.3385` | `0` | `-0.59742` | `-0.55727` | `54.4613` |
| Eq. (112) residual subtraction, Hartree--Fock | `9560.98` | `-293.37` | `-2057.24` | `-38636.5` | **`15.3724`** |
| the cheap repair, Hartree--Fock, exact lift | `1.9431` | `-2.220e-16` at `t = 0`, `+1.447e-6` after | `-2.163e-3` | `-5.561e-4` | never, through `20` |
| energy-neutral joint electronic--bosonic matrix barrier, Hartree--Fock | `1.7999` | `0` | `-5.103e-16` | `-0.36128` | never, through `20` |
| unrepaired archive closure, exact correlated | `1.6846` | `-0.054822` | `-0.70828` | `-1.02769` | never, through `140` |
| Eq. (112) residual subtraction, exact correlated | `1.9787` | `+0.055081` | `-0.30514` | `-0.45102` | never, through `140` |

Accuracy and cost of the same four rows from the divergent initialization,
against the exact driven trajectory started from the same Hartree--Fock
product state, all-coordinate scalar RMS of the 31 closed scalar coordinates:

| vector field, Hartree--Fock preparation | `t = 4` | `t = 8` | `t = 20` | Eq. (22) energy RMS at `t = 20` | wall to `t = 20` |
|---|---:|---:|---:|---:|---:|
| the cheap repair, 27 keys, exact lift | **`1.0803e-2`** | **`3.7209e-2`** | **`1.5368e-1`** | `1.1888e-3` | `4,296.9 s` |
| energy-neutral joint matrix barrier | `2.0232e-1` | `3.7137e-1` | `3.4704e-1` | `1.0638e-1` | `793.2 s` |
| unrepaired archive closure | `1.2957e-1` | `2.9372e-1` | `3.3200e-1` | `4.5174e-2` | under `1 s` |
| Eq. (112) residual subtraction | `2.2891e-1` | `4.7133e-1` | `5.2318e9` | `1.1495e-1` | under `1 s` |

The cheap repair is `12.0` times the unrepaired closure's accuracy at `t = 4`
and `2.16` times at `t = 20`, and `18.7` and `2.26` times the joint matrix
barrier's. The joint matrix barrier is the only row that holds a cone: its
two constrained quantities, the electronic and boson-moment eigenvalues, stay
at `0` and `-5.10e-16`, at `0.185` of the cheap repair's wall — but the
retained joint Gram, which is not in its constraint set, reaches `-0.36128`,
`650` times the cheap repair's depth. Eq. (112) inside that barrier is
bounded; Eq. (112) alone is not.

**Eq. (112) does not remove the amplitude divergence on the initialization it
exists for.** The record's Eq. (112) rows are quoted for exact
contractions, and those reproduce here exactly — the exact-correlated
Eq. (112) row above matches the working notes' cone-correction table
(`-3.05033e-1`, `5.51945e-2`, `1.97874`) to every printed digit, and the
13-coordinate exact-contraction row is bounded through `t = 400` at
`1.8592243`. From the Hartree--Fock/zero-correlation state the same
subtraction moves the `1e4` crossing **earlier**, from `t = 54.4613` to
`t = 15.3724` in the 31 closed scalar coordinates and from `t = 130.4675` to
`t = 14.5178` in the 13. Its electronic positivity loss also moves earlier,
from `t = 27.9634` to `t = 1.11532`. The working notes' sentence that
Eq. (112) "removes the amplitude divergence and preserves the electron
density" holds for the exact-contraction protocol and is refuted for the
divergent initialization.


## 6. The answer, plainly

**Does the cheap repair prevent the representability loss that precedes
divergence on the initialization that actually diverges?** Partly, and it
cannot start there on its own.

1. **It cannot be prepared at that initialization.** The declared proximal
   log-det completion — the route's one conic solve, at `t = 0` — refuses at
   the divergent preparation at both dictionaries and at every rung of the
   declared retry ladder, with
   `moment_extension_degeneracy: facial reduction removed every frontier
   direction`. The cause is that the divergent initialization is a pure
   product state whose retained joint Gram is exactly singular (rank 6 of 8,
   smallest eigenvalues `-4.44e-16` and `-4.21e-16`). Supplying the initial
   lift some other way is not optional; this round supplied the exact
   frontier moments of the preparation's own state vector, and everything
   after `t = 0` is the committed route.
2. **Given that lift, it prevents the electronic loss completely.** The
   electronic one-body density matrix is positive semidefinite for the whole
   rollout: `-2.220e-16` at `t = 0` (the preparation is a determinant) and a
   minimum of `+1.447e-6` over `0 < t <= 20`. The record's `t = 15.5177` and
   this round's 31-coordinate `t = 27.9634` are both inside the horizon and
   neither happens. The state magnitude stays at `1.9431` against `1e4` at
   `t = 54.4613` unrepaired, and the accuracy is `12.0` times the unrepaired
   closure's at `t = 4`.
3. **It does not hold either Gram cone.** The retained joint Gram leaves
   `-1e-8` at the first step and reaches `-5.561e-4` by `t = 12`; the scaled
   positive extension reaches `-6.352e-5`. Both are within `25` and `61`
   percent of the depths the same configuration reaches from the exact
   correlated preparation, so this is the known behaviour of the Eq. (22)
   hyperplane and not something the divergent initialization causes. The part
   that fails is the Dykstra repair's hold on the retained joint Gram, and it
   fails from `t = 0.0025` onward, deepening in two steps — to `-4.370e-5` by
   `t = 8` and to `-5.561e-4` by `t = 12`.
4. **The alternative that does hold its cones is much less accurate.** The
   energy-neutral joint electronic--bosonic matrix barrier holds the
   electronic and boson-moment cones at `0` and `-5.10e-16` through `t = 20`
   at `793.2 s`, but scores `3.4704e-1` against the cheap repair's
   `1.5368e-1`, and its retained joint Gram reaches `-0.36128`.
5. **Eq. (112) is refuted on this initialization.** It moves the `1e4`
   crossing earlier, from `t = 54.4613` to `t = 15.3724` in the 31 closed
   scalar coordinates and from `t = 130.4675` to `t = 14.5178` in the 13,
   while reproducing the record's exact-contraction rows exactly. It is a
   strong-coupling regularization of the *correlated* preparation only.

## 7. Reproduction

Run from the worktree with
`PY=/Users/lordskyler/Holstein_test/.venv/bin/python`,
`DIR=output/local_runs/paper_v_divergence_20260920`,
`PYTHONPATH=paper_5/src:$DIR`.

```bash
# (a) and the Eq. (112) rows of (c), plus the two harness controls
$PY pipelines/shell/ram_guard.py --limit-mb 1500 -- \
  $PY $DIR/run_unrepaired.py --output $DIR/unrepaired_t140.json \
  --scalar-final-time 140 --closed-final-time 140 --control-final-time 400 \
  --sample-step 0.005

# section 3: where the declared completion stands at the divergent preparation
$PY pipelines/shell/ram_guard.py --limit-mb 4500 -- \
  $PY $DIR/probe_preparation.py --output $DIR/preparation_probe_hf.json \
  --preparations hartree_fock \
  --shifts 1e-10 1e-8 1e-7 1e-6 1e-4 1e-2 1e-1 --weights 1e-1

# (b) the cheap repair from the divergent initialization, chained to t = 20
/bin/zsh $DIR/run_chain.sh \
  output/local_runs/paper_v_divergence_hf_dyk120_20260920 \
  hartree_fock exact_moments 1 9 20.0 1000

# cumulative score and physicality of that chain (five segments)
$PY $DIR/score_chain.py \
  output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg{1,2,3,4,5} \
  --output $DIR/hf_chain_score_t20.json --horizons 1 2 4 8 12 16 20
$PY $DIR/tabulate_chain.py $DIR/hf_chain_score_t20.json <the segments>
$PY $DIR/electron_trace.py <the segments>

# (c) the recorded energy-neutral joint electronic--bosonic matrix barrier
$PY pipelines/shell/ram_guard.py --limit-mb 4000 -- \
  $PY $DIR/run_cone_route.py --output $DIR/cone_route_hf_t20.json \
  --preparation hartree_fock --final-time 20 --wall-budget-seconds 1100
$PY $DIR/score_trajectory.py $DIR/cone_route_hf_t20.npz \
  --output $DIR/cone_route_hf_score.json --preparation hartree_fock

# (c) the accuracy column of the unrepaired and Eq. (112) rows
$PY pipelines/shell/ram_guard.py --limit-mb 3000 -- \
  $PY $DIR/score_archive_routes.py \
  --output $DIR/archive_route_scores_t20.json
```

Files written by this round, all under
`output/local_runs/paper_v_divergence_20260920/` unless stated:
`hartree_fock_preparation.py` (the divergent preparation as a Fock-space
product state), `exact_lift.py` (the exact frontier moments of a pure
preparation), `run_unrepaired.py`, `probe_preparation.py`,
`run_divergence_segment.py`, `run_chain.sh`, `score_chain.py`,
`tabulate_chain.py`, `electron_trace.py`, `run_cone_route.py`,
`score_trajectory.py`, `score_archive_routes.py`, and the chain segments
`output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg{1..5}`.
Score files: `hf_chain_score_t20.json`, `cone_route_hf_score.json`,
`archive_route_scores_t20.json`, `unrepaired_t140.json`,
`preparation_probe_hf.json`, `cone_route_hf_t20.json` with its `.npz`.
