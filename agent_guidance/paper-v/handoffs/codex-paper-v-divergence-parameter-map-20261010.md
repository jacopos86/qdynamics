# Codex handoff: why the closure diverges from exact data — a parameter map (Paper V classical repair, 2026-10-10)

**What this is.** A standalone, exploratory research-and-implementation
handoff for one Codex session. It is one of four independent threads on the
Paper V classical repair (index:
`prompt-exports/codex-paper-v-INDEX-20261010.md`). It follows
`agent_guidance/shared/agent-handoff-contract.md`: anchors, scope,
increments that each end committable, evidence standards, traps and the
report-back block. Section C (the common rules, at the end) is binding; read
it before increment 0. Nothing here is evidence or manuscript prose.

**The question in one sentence.** The unrepaired archive closure diverges
even from exact data, so the divergence belongs to the closure's vector
field and not only to a bad initialization; from the exact correlated
preparation, where does the `1e4` magnitude event and the electronic
positivity loss happen as a function of the phonon cutoff of the exact
initial contraction, the drive amplitude and the coupling `lambda_ep`, and is
the divergence a strong-coupling phenomenon or generic.

**What counts as the answer.** A table over `lambda_ep` and drive amplitude,
for the exact correlated preparation and, beside it, the Hartree--Fock
zero-correlation preparation, of the `1e4` magnitude event time and the
electronic positivity loss time of the unrepaired archive closure in windows
of `500` and `2000` time units, read at the resolution increment 2 measures;
the event time against the phonon cutoff `16` to `32`, compared with that
resolution; the same closure started from exact driven states of the
correlated preparation at later times; and a statement, by the reading rule
of increment 4, of whether the divergent region is confined to strong
coupling.

---

## 0. Identity of this thread

Run this block first in every shell (zsh or bash):

```bash
export TOPIC=divergence-parameter-map
export BRANCH=codex/paper-v-divergence-parameter-map-20261010
export WT=/Users/lordskyler/Holstein_test.worktrees/codex-paper-v-divergence-parameter-map-20261010
export SRC=/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914          # read-only
export PY=/Users/lordskyler/Holstein_test/.venv/bin/python
export OUT=$WT/output/local_runs/paper_v_divergence_parameter_map_codex_20261010
export REPORTFILE=$WT/agent_guidance/paper-v/handoffs/REPORT-divergence-parameter-map-codex-20261010.md
export DATA=$WT/agent_guidance/paper-v/handoffs/divergence-parameter-map-codex-20261010-data
export REC=$SRC/output/local_runs/paper_v_divergence_20260920                               # read-only
export LOCK=$SRC/agent_guidance/paper-v/handoffs/codex-paper-v-divergence-parameter-map-20261010-inputs.sha256   # = appendix L
```

**Drivers this thread runs.** One new **committed** module,
`paper_5/src/paper5/stability/divergence_parameter_map.py` (increment 1),
which ports the diagnostics it needs and imports nothing from `$OUT`. The
record's **uncommitted** scripts under
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/`
are read for reference and for the reproduction targets only (SHA-256 in
appendix L): `run_unrepaired.py` bundles seven propagations with a fixed
`400`-unit control horizon and is not a single-arm grid worker, and
`score_archive_routes.py` hard-codes `lambda = 1.5`, `gamma = 0.5`,
drive `1.0` (readiness audit, Increment 2). This thread runs **no chained
repaired segments**, so the Eq. (22) energy-reference carry and the parity
gate of C20 do not apply to it; the unrepaired archive closure has no energy
reference.

Inputs, all **read-only**, by absolute path (C5 lists what is inside each):

* `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/`:
  `run_unrepaired.py` (both unrepaired closures and their diagnostics:
  `AMPLITUDE_THRESHOLD`, `_closed_diagnostics`, `_retained_gram_minimum`,
  `_first_threshold`), `run_closed_route.py` (one preparation, `--route
  archive` or `equation_112`, `--lambda-ep`, `--gamma`, `--drive`,
  `--phonon-cutoff`, `--final-time`, `--initial-from`), `boundary_basin_sweep.py`
  (the closure from exact driven states at given times),
  `run_divergence_segment.py` (for `_reference_coordinates`, the exact driven
  reference from either preparation), `exact_state_npz.py` (one exact state
  at one time), `hartree_fock_preparation.py`; score files
  `unrepaired_t140.json` (seven recorded runs with their crossing times and
  initial residual norms), `boundary_basin_span400_exact.json` (the closure
  from the exact driven trajectory of the Hartree--Fock product state at
  fourteen times, 400 units), `unrepaired_from_exact_t60.json`.

The record this handoff cites (tracked): `agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
(**DI**); `MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md`
(**WN**); `paper_5/notes/electron_phonon_closure_worklog_20260803.md`, the
2026-09-20 blocks, lines 8200 to 8447 (**WL**);
`MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`
(**CR**).

---

## 1. Why this thread exists, and the premise stated exactly

The record's last round found that the unrepaired archive closure fires the
`1e4` magnitude event from **exact** states: started from the exact driven
trajectory's state at `t = 60` it fires at `t = 433.8964`, and at thirteen of
fourteen stored times within `400` time units (DI section 13.2, lines 791 to
812; WL lines 8406 to 8414). The quantity that test measures is "the time
from a state to the `1e4` magnitude event under the unrepaired closure" (DI
section 14, lines 822 to 825), and the divergence is therefore the vector
field's.

**Two different "exact" states appear in the premise; keep them apart.**

* The record's `433.8964` and its thirteen-of-fourteen count come from the
  exact driven trajectory **of the Hartree--Fock zero-correlation product
  state** (`boundary_basin_sweep.py` calls `_reference_coordinates(...,
  preparation="hartree_fock")`).
* "Exact correlated initial conditions" means the exact zero-field ground
  state at cutoff 16 contracted into the closure's coordinates (glossary
  C19). The working notes record, as reported by the source document,
  "exact correlated initial conditions are reported to remove the
  divergence" (WN line 212), and their own tables already show that it does
  so only over a finite window:

| protocol, `lambda = 1.5`, `gamma = 0.5`, `v / t_hop = 1` | `1e4` magnitude event | first electronic positivity loss | source |
|---|---:|---:|---|
| 31 closed scalar coordinates, Hartree--Fock zero-correlation | `54.4613` | `27.96` | WN line 415 |
| 31 closed scalar coordinates, exact ground-state contractions | `141.4056` | `1.04` | WN line 416 |
| 31 closed scalar coordinates, exact contractions with Eq. (112) residual subtraction | none through `400`, maximum `\|x_i\|` about `2.2613` | none through `400` | WN lines 417 and 419 |
| 15-coordinate form, exact ground-state contractions | `132.70` | `1.019` | WN line 359 |
| 13-coordinate Fan--Migdal form, exact ground-state contractions | `154.25` | `0.9909` | WN line 1308 |
| 13-coordinate form, Hartree--Fock zero-correlation, `lambda = 0.5` | bounded through `t = 140` (a window statement) | --- | WN line 225 |

  The record's own harness, in the 31 closed scalar coordinates from the
  exact correlated preparation, saw no event through `t = 140` with maximum
  `|x_i|` `24.2421` and electronic minimum `-7.7127` over that window, the
  electronic crossing below `-1e-8` at `1.0362132`, the boson-moment crossing
  at `3.6820771` and the retained joint Gram crossing at `0.1607047` (DI
  section 2.2, lines 89 to 95; `unrepaired_t140.json`, run
  `closed_archive_exact_correlated`). The `140`-unit window ended `1.4` time
  units before the working notes' event time, and the closure-instability
  round of 2026-10-10 measured the event at `141.4057436` with the record's
  protocol and `141.405885` converged (C4a).

So the premise "exact correlated initial conditions remove the divergence
only over short horizons" is already in the record at one parameter point.
This thread maps it over parameters, measures how many digits of an event
time are meaningful, and adds the exact driven states of the correlated
preparation as later starts.

---

## 2. Further anchors

| fact | number | source |
|---|---|---|
| initial residual norm `\|\|F(x_0, 0)\|\|` of the 31-coordinate archive closure | `0.5303300858899106` (Hartree--Fock zero-correlation), `0.4096860567524964` (exact correlated) | `unrepaired_t140.json`, `preparation` block; DI section 1 |
| the same closure from the Hartree--Fock zero-correlation preparation, 31 coordinates | retained joint Gram below `-1e-8` at `0.0124346`, boson-moment at `1.3226952`, electronic at `27.9633944`, `1e4` event at `54.4612958`, maximum `\|Tr C[q]\|` `5151.44` | DI section 2.2, lines 87 to 96 |
| integrator independence of an event time | "RK4 refinement and converged DOP853, Radau, and BDF integrations agree on the failure time" (13-coordinate Hartree--Fock case) | WN line 223 |
| cost of the closure | `0.0157 s` per time unit without per-sample diagnostics; with per-sample diagnostics every `0.0025`, `182.3 s` for 28 propagations of 400 units | DI sections 11 and 15 |
| the exact cutoff-16 driven reference | `3.06 s` to `t = 20` at two sites | `agent_guidance/paper-v/overnight/scaling-exact-cost-law-20260920.md` (via the campaign plan's table) |
| the drive | `2 A sin(pi t / 4) exp(-(t / pulse_width)^2 / 2)`, `pulse_width = 1.0` (`DimerParameters.drive_difference`, `paper_5/src/paper5/stability/hubbard_dimer.py:83-88`); the envelope is about `1.3e-14` at `t = 8`, so after `t = 8` the closure is autonomous to that precision | source code |
| the parameters | `omega_ph = gamma * t_hop`; `g = sqrt(lambda_ep * t_hop * omega_ph / 2)` (`hubbard_dimer.py:72-81`); at `lambda_ep = 1.5`, `gamma = 0.5`: `omega_ph = 0.5`, `g = 0.6123724`, coherent displacement `-g / omega_ph = -1.2247449` | source code; DI section 1 |
| the exact correlated preparation does not depend on the drive amplitude | `exact_ground_closed_scalar_coordinates` replaces the drive amplitude by `0` before the ground-state solve (`initial_conditions.py:101-113`); the Hartree--Fock zero-correlation preparation does not depend on it either (`initial_conditions.py:116-140`) | source code |

---

## 3. Scope specific to this thread

* In scope: the unrepaired archive closure (and, as a control, the Eq. (112)
  residual subtraction route already in `run_closed_route.py`) from the two
  preparations and from exact driven states, over a declared grid of
  `lambda_ep`, drive amplitude and phonon cutoff; event-time spreads under
  declared perturbations and tolerances.
* Out of scope: Jacobians, finite-time Lyapunov exponents and step-size
  studies along these trajectories (C4: another agent's running task;
  coordinate by reporting your event times, which that agent can use); any
  repair; the 56-coordinate route; any change to `closed_scalar_rhs`.
* Machine headroom: every run is single-core and under about `300 MB`; run
  up to 8 workers after `memory_pressure -Q`.

---

## 4. Increments

Proceed through all of them autonomously (C15).

### Increment 0 — worktree, input lock, readable inputs (under 20 minutes)

**Goal.** Your worktree exists at `b9f99271`; every input matches its
SHA-256; the drivers are copied byte for byte; every input opens.

**Bound.** Under 2 minutes.

```bash
git -C /Users/lordskyler/Holstein_test worktree add -b "$BRANCH" "$WT" b9f99271fc3ddf3db604d3acb64617a291bec7e1
git -C "$WT" rev-parse HEAD
mkdir -p "$OUT" "$DATA/drivers"
shasum -a 256 -c "$LOCK"                                    # every line must print OK
cp "$REC"/*.py "$REC"/*.sh "$OUT"/
(cd "$REC" && shasum -a 256 *.py *.sh) > "$OUT/copies_source.sha256"
(cd "$OUT" && shasum -a 256 *.py *.sh) > "$OUT/copies_copied.sha256"
diff "$OUT/copies_source.sha256" "$OUT/copies_copied.sha256" && echo "copies byte-identical"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -c "
import json
import paper5.stability.matrix_reference as m; print('import', m.__file__)
d = json.load(open('$REC/unrepaired_t140.json'))
print('residuals', d['preparation']['hartree_fock_initial_residual_norm'], d['preparation']['exact_initial_residual_norm'])
for r in d['runs']:
    print(r['label'], r['final_time_reached'], r['maximum_magnitude'], r.get('electron_eigenvalue_first_below_1e-08'))
e = json.load(open('$REC/boundary_basin_span400_exact.json'))
print([(b['boundary_time'], b['from_exact_trajectory_state']['amplitude_threshold_event_time']) for b in e['boundaries']])
from run_closed_route import main
print('run_closed_route importable')
"
memory_pressure -Q
```

Write `$DATA/input_manifest.json` with the expected and actual SHA-256 of
every locked input and every copy.

**Acceptance.** HEAD is `b9f99271fc3ddf3db604d3acb64617a291bec7e1`; every
lock line prints `OK`; the copies are byte-identical; the import is under
`$WT`; the two residual norms print as in section 2; the seven runs
print with `closed_archive_exact_correlated` at final time `140.0`, maximum
magnitude `24.24214267917956`, electronic crossing `1.0362131873909188`; the
fourteen exact-state event times match DI section 13.2 (thirteen times, one
`None` at `24.44`).
**Stop.** Any lock line not `OK`, any input missing or differing, or an
import outside `$WT`.
Commit `$DATA/input_manifest.json`.

### Increment 1 — the map module, and the record's points reproduced

**Goal.** The test baseline is measured before any source change; then a
committed module runs the unrepaired archive closure (or the Eq. (112)
control) from a declared preparation at declared parameters and reproduces
the record.

**Bound.** The suite takes about `250` to `265 s`; each reproduction command
takes under 2 minutes.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q   # the baseline (C16), before any edit
```

Write `paper_5/src/paper5/stability/divergence_parameter_map.py` with this
command-line contract (flags may be added, not renamed). It writes one JSON
line per finished cell as it goes; `--resume` skips cells already in the
output; `--max-wall-seconds` stops cleanly, marks unfinished cells
`incomplete` and says the command must be repeated (C10):

```text
python -m paper5.stability.divergence_parameter_map
  --preparation {exact_correlated,hartree_fock}
  --start-time T [T ...]        0 starts at the preparation; T > 0 starts from the exact driven
                                trajectory of that preparation at time T (same Hamiltonian and cutoff)
  --lambda-ep L [L ...]  --drive A [A ...]  --gamma G  --phonon-cutoff N [N ...]
  --route {archive,equation_112}
  --final-time TF               absolute end time (window = TF - T)
  --only-without-event-in JSON  restrict to cells with no event in that earlier run
  --perturbation-size S --perturbation-draws N --perturbation-seed SEED
                                N seeded relative perturbations of size S of the 29 coordinates other than indices 17, 18
  --relative-tolerance 1e-10 --absolute-tolerance 1e-12 --sample-step 0.0025
  --workers N --max-wall-seconds S --resume --output JSON
```

Per cell it records: the configuration; the initial 31 closed scalar
coordinates' largest absolute difference from the cutoff-16 preparation at
the same parameters (for the cutoff study); the ground-state energy and, if
available, the discarded weight; the initial residual norm `||F(x_0, 0)||`
of `closed_scalar_rhs`; the `1e4` magnitude event time or `null`; the first
crossings of the electronic one-body density matrix minimum eigenvalue below
`0`, `-1e-8`, `-1e-3`; the first crossings below `-1e-8` of the boson-moment
eigenvalue and of the unshifted retained joint Gram; the maximum `|x_i|` and
its time; wall seconds; `"classification": "exploratory_local_not_promoted"`.
Reuse `closed_scalar_rhs`, `exact_ground_closed_scalar_coordinates`,
`hartree_fock_closed_scalar_coordinates`, `closed_residual_subtracted_rhs`,
the diagnostics of `run_unrepaired.py` (port them into the module; the module
must not import from `$OUT`) and, for `--start-time T > 0`, an exact
propagation with the settings of `_reference_coordinates` (DOP853,
`rtol 1e-9`, `atol 1e-11`, `max_step 0.01`) contracted with
`_contract_matrix_state` and `matrix_state_to_closed_scalar_coordinates`.

Test `paper_5/tests/test_divergence_parameter_map.py`: the Hartree--Fock
zero-correlation preparation at `lambda_ep = 1.5`, `gamma = 0.5`, drive `1`,
final time `60`: event `54.4612958` and electronic crossing below `-1e-8`
`27.9633944`, each to `1e-6`; initial residual norm `0.5303300858899106` to
`1e-12`.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q tests/test_divergence_parameter_map.py
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 0 --lambda-ep 1.5 --drive 1.0 --gamma 0.5 --phonon-cutoff 16 \
  --route archive --final-time 500 --max-wall-seconds 1080 --resume --output "$OUT/reproduce_exact_correlated_t500.json"
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation hartree_fock --start-time 0 60 --lambda-ep 1.5 --drive 1.0 --gamma 0.5 --phonon-cutoff 16 \
  --route archive --final-time 500 --max-wall-seconds 1080 --resume --output "$OUT/reproduce_hartree_fock_t500.json"
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 0 --lambda-ep 1.5 --drive 1.0 --gamma 0.5 --phonon-cutoff 16 \
  --route equation_112 --final-time 400 --max-wall-seconds 1080 --resume --output "$OUT/reproduce_exact_correlated_eq112_t400.json"
cp "$OUT"/reproduce_*.json "$DATA"/
```

**Acceptance.** The test passes. Exact correlated, archive route: electronic
crossing below `-1e-8` within `1e-6` of `1.0362132`, boson-moment crossing
within `1e-6` of `3.6820771`, retained joint Gram crossing within `1e-6` of
`0.1607047`, maximum `|x_i|` over `t <= 140` within `1e-4` of `24.2421`,
initial residual norm within `1e-12` of `0.4096860567524964`, and an event
time; within `1e-4` of `141.4057436` (the closure-instability round's
value at this protocol, C4a) and within `0.01` of the working notes'
`141.4056` it is recorded as reproduced. Hartree--Fock, start `0`: `54.4612958` and `27.9633944` to
`1e-6`; start `60` (the exact driven trajectory of the Hartree--Fock product
state at `t = 60`): event within `1e-3` of `433.8964` (DI section 13.2).
Eq. (112) control: no event through `400`, maximum `|x_i|` within `1e-3` of
`2.2613` (WN line 419).
**Stop.** The exact correlated event time more than `1.0` time unit from
`141.4056`, or the start-`60` event more than `1e-3` from `433.8964`: the
harness differs from the record's; report and stop. A difference between
`0.01` and `1.0` from `141.4056` is a surprise to report, not a stop (the
working notes' tolerances and sampling are not stated beside that number).
Commit the module, its test and the JSON copies.

### Increment 2 — resolution: how many digits of an event time mean anything

**Goal.** The spread of the event time at the baseline point under declared
small perturbations and a tighter tolerance, for both preparations. C4a gives
the expected scale from the exact `t = 60` state (`7.1e-3` at `1e-12`, `1.31`
at `1e-10`); this increment measures it at the two `t = 0` starts the map
uses.

**Bound.** Eight invocations of at most ten `2000`-unit starts (about `33 s`
each with per-sample diagnostics), each under 2 minutes on 8 workers.

```bash
cd "$WT" && for prep in exact_correlated hartree_fock; do
  for s in 1e-12 1e-10 1e-8; do
    PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
      --preparation $prep --start-time 0 --lambda-ep 1.5 --drive 1.0 --gamma 0.5 --phonon-cutoff 16 --route archive \
      --final-time 2000 --perturbation-size $s --perturbation-draws 10 --perturbation-seed 20261010 --workers 8 \
      --max-wall-seconds 1080 --resume --output "$OUT/resolution_${prep}_${s}.json"
  done
  PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
    --preparation $prep --start-time 0 --lambda-ep 1.5 --drive 1.0 --gamma 0.5 --phonon-cutoff 16 --route archive \
    --final-time 2000 --relative-tolerance 1e-12 --absolute-tolerance 1e-14 --max-wall-seconds 1080 --resume --output "$OUT/resolution_${prep}_tol.json"
done
```

Do not compute Jacobians or exponents (C4).

**Acceptance.** All 62 starts complete. Reported per preparation and size:
how many of the ten fire within `2000` units and the spread (largest minus
smallest) of their event times and of their electronic crossing times; and
the tolerance run's difference from the unperturbed time. Decision recorded
in `$REPORTFILE`, in numbers: if the `1e-10` spread is below `1` time unit
for both preparations, increment 4 reports one start per cell; otherwise
every cell of increment 4 is run as the unperturbed start plus four
perturbations of size `1e-10`, and reported as the count firing and the
median event time. If the tolerance change alone moves an event time by more
than the `1e-10` spread, increments 3 to 6 use `rtol 1e-12`, `atol 1e-14`.
**Stop.** More than three of the 62 starts end in a solver error other than
the terminal event.
Commit `$DATA/resolution_summary.json`.

### Increment 3 — the phonon cutoff of the exact initial contraction

**Goal.** Whether the cutoff used to contract the exact correlated ground
state moves the event time by more than a perturbation of the same size
does.

**Bound.** Eight `2000`-unit cells plus ground-state solves at cutoffs up to
32 (state dimension `4 * 33^2 = 4,356`), under 5 minutes.

```bash
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 0 --lambda-ep 1.5 --drive 1.0 --gamma 0.5 \
  --phonon-cutoff 16 20 24 28 32 --route archive --final-time 2000 --workers 5 \
  --max-wall-seconds 1080 --resume --output "$OUT/cutoff_lambda1.5.json"
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 0 --lambda-ep 3.0 --drive 1.0 --gamma 0.5 \
  --phonon-cutoff 16 24 32 --route archive --final-time 2000 --workers 3 \
  --max-wall-seconds 1080 --resume --output "$OUT/cutoff_lambda3.0.json"
cp "$OUT"/cutoff_lambda*.json "$DATA"/
```

**Acceptance.** All eight cells complete. Reported per cutoff: the largest
absolute difference of the initial 31 closed scalar coordinates from the
cutoff-16 preparation, the ground-state energy, the event time and the
electronic crossing times. The answer, in numbers: the cutoff is reported as
not mattering if, at every cutoff, the event-time change from cutoff 16 is no
larger than the increment-2 spread at the perturbation size nearest to that
cutoff's initial-coordinate change; otherwise the change is reported as the
cutoff's effect. The largest cutoff whose initial coordinates differ from
the next smaller cutoff's by less than `1e-12` is stated as converged.
**Stop.** A ground-state solve failing or exceeding `memory_pressure`
headroom at a cutoff: skip that cutoff and report it.
Commit the two JSON files.

### Increment 4 — the grid over `lambda_ep` and drive amplitude

**Goal.** The map itself.

**Bound.** 54 cells of `500` units per preparation, about 1 minute on
8 workers; the `2000`-unit continuations at most 54 cells at about `33 s`,
under 4 minutes per preparation. Each invocation stops itself at `1080 s`.

```bash
LAMBDAS="0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0"
DRIVES="0 0.25 0.5 1.0 1.5 2.0"
cd "$WT" && for prep in exact_correlated hartree_fock; do
  PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
    --preparation $prep --start-time 0 --lambda-ep $(echo $LAMBDAS) --drive $(echo $DRIVES) --gamma 0.5 \
    --phonon-cutoff 16 --route archive --final-time 500 --workers 8 \
    --max-wall-seconds 1080 --resume --output "$OUT/grid_${prep}_w500.json"
  PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
    --preparation $prep --only-without-event-in "$OUT/grid_${prep}_w500.json" --final-time 2000 --workers 8 \
    --max-wall-seconds 1080 --resume --output "$OUT/grid_${prep}_w2000.json"
done
```

(If increment 2 required perturbed draws per cell, add
`--perturbation-size 1e-10 --perturbation-draws 4 --perturbation-seed 20261010`
to both commands.)

Cost: 54 cells of `500` units per preparation at about `8` to `10 s` each with per-sample
diagnostics, about 2 minutes on 8 workers per preparation, more with draws
and for the `2000`-unit continuations.

**Acceptance.** All 108 cells complete. The cells at `lambda_ep = 1.5`, drive
`1.0` equal increment 1's numbers bit for bit. Reported, per preparation and
per `lambda_ep`: how many of the six drive amplitudes fire within `500` and
within `2000` units, the median event time among those that fire, the
median electronic crossing time below `-1e-8`, the initial residual norm,
and the drive-`0` cell separately (with no drive the exact correlated
preparation is stationary under the exact dynamics, so any event there is the
closure's alone). The answer, by this reading rule, stated with its window:
the divergence is reported as **confined to strong coupling** if, from the
exact correlated preparation, no cell with `lambda_ep <= 0.75` fires within
`2000` units while at least five of six cells fire at every
`lambda_ep >= 1.5`; as **generic** if at least half the cells with
`lambda_ep <= 0.75` fire within `2000` units; and as **mixed** otherwise,
with the table.
**Stop.** More than 5 percent of cells end in a solver error other than the
terminal event.
Commit `$DATA/grid_exact_correlated.json` and `$DATA/grid_hartree_fock.json`
(per-cell rows; under 1 MB each).

### Increment 5 — the closure from exact driven states of the correlated preparation

**Bound.** Fourteen starts of about `2000` units plus one exact propagation
to `t = 60` (about `9 s`), under 3 minutes.

**Goal.** Whether the record's finding from the Hartree--Fock product
state's exact trajectory (thirteen of fourteen fire within `400` units) holds
for the exact trajectory of the correlated preparation, where the moments
are those of a genuinely correlated driven state.

```bash
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 5 10 20 40 60 --lambda-ep 1.5 --drive 1.0 --gamma 0.5 \
  --phonon-cutoff 16 --route archive --final-time 2060 --workers 5 --max-wall-seconds 1080 --resume --output "$OUT/exact_states_lambda1.5.json"
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 20 --lambda-ep $(echo $LAMBDAS) --drive 1.0 --gamma 0.5 \
  --phonon-cutoff 16 --route archive --final-time 2020 --workers 8 --max-wall-seconds 1080 --resume --output "$OUT/exact_states_t20_lambda_grid.json"
cp "$OUT"/exact_states_*.json "$DATA"/
```

**Acceptance.** All fourteen starts complete. Reported: per start, the event
time (absolute and elapsed) or "no event within the window" with the window,
and the electronic crossing time; beside the record's fourteen
Hartree--Fock-trajectory rows (DI section 13.2). The answer, in numbers: the
fraction of the five correlated-trajectory starts at `lambda_ep = 1.5` that
fire within `400` elapsed units, beside the record's thirteen of fourteen.
**Stop.** None beyond C15.
Commit the two JSON files.

### Increment 6 — controls: Eq. (112) and `gamma`

**Goal.** Two cheap controls that bound the reading.

**Bound.** Eleven `2000`-unit cells, under 3 minutes.

```bash
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 0 --lambda-ep $(echo $LAMBDAS) --drive 1.0 --gamma 0.5 \
  --phonon-cutoff 16 --route equation_112 --final-time 2000 --workers 8 --max-wall-seconds 1080 --resume --output "$OUT/grid_eq112_exact_correlated.json"
cd "$WT" && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.divergence_parameter_map \
  --preparation exact_correlated --start-time 0 --lambda-ep 1.5 --drive 1.0 --gamma 0.25 1.0 \
  --phonon-cutoff 16 --route archive --final-time 2000 --workers 2 --max-wall-seconds 1080 --resume --output "$OUT/gamma_control.json"
cp "$OUT"/grid_eq112_exact_correlated.json "$OUT"/gamma_control.json "$DATA"/
```

(`--gamma` takes a list in your module; if it does not, run it twice.)

**Acceptance.** All cells complete. Reported: per `lambda_ep`, whether the
Eq. (112) residual subtraction from the exact correlated preparation fires
within `2000` units (the record: none through `400` at `lambda_ep = 1.5`, WN
lines 417 and 419; the record also refutes Eq. (112) from the Hartree--Fock
zero-correlation preparation, moving the event earlier to `15.3724`, DI
section 5); and the two `gamma` cells' event times beside the
`gamma = 0.5` cell.
**Stop.** None beyond C15.
Commit the two JSON files.

### Increment 7 — the report-back file and the final commit

**Bound.** The suite takes about `250` to `265 s`.

Write `$REPORTFILE` per C17, then:

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q
git -C "$WT" add agent_guidance/paper-v/handoffs/REPORT-divergence-parameter-map-codex-20261010.md agent_guidance/paper-v/handoffs/divergence-parameter-map-codex-20261010-data
git -C "$WT" commit -m "paper-v (Codex): the unrepaired closure's divergence over lambda_ep, drive and cutoff — <the answer with its numbers>"
```

**Acceptance.** The suite stands at the increment-1 baseline plus the new tests
with no new failure; nothing tracked is left modified.
**Stop.** A new test failure: fix it if it is yours, otherwise report it.

---

## 5. Evidence standards for this thread

* Reproduce the record's points (increment 1) before any new cell is read.
* Resolution before the map (increment 2): an event time is reported to no
  more digits than its measured spread supports.
* Every "no event" carries its window and is never written as bounded; the
  record withdrew exactly that reading once (DI section 13).
* Keep the two "exact" starts apart in every table: the exact correlated
  preparation and its exact driven trajectory, against the Hartree--Fock
  product state's exact driven trajectory.
* The working notes' numbers are a separate harness whose tolerances are not
  stated beside them; a small disagreement is reported, not forced.

## 6. Traps

* Your worktree has none of the drivers or score files (C5).
* `run_closed_route.py --initial-from` starts from the **last** sample of a
  file; your module computes exact states at the requested time instead.
* The exact correlated preparation and the Hartree--Fock zero-correlation
  preparation do not depend on the drive amplitude (section 2); the drive
  enters only the propagation, and is negligible after `t = 8`.
* At large `lambda_ep` the coherent displacement `g / omega_ph` grows as
  `sqrt(lambda_ep)`; increment 3's `lambda_ep = 3.0` cutoff check guards the
  contraction there. At `lambda_ep = 0` the coupling vanishes and
  `DimerParameters` accepts it, but the grid starts at `0.25`.
* Indices 17 and 18 of the 31 closed scalar coordinates are the shared
  correlation trace, which vanishes on every state; do not perturb them.
* The 13-, 15- and 31-coordinate forms give different times (section 1
  table); this thread uses the 31.
* Per-sample diagnostics at `0.0025` over `2000` units are 800,000 samples
  per run; compute window minima on the fly rather than storing trajectories,
  and store trajectories only for the cells you plot.

## 7. Report back

`$REPORTFILE` per C17, committed on your branch. Its first lines state, one
sentence each: the reading of increment 4 (confined to strong coupling,
generic, or mixed) with its window; the cutoff's effect against the measured
resolution; and the fraction of correlated-trajectory starts that fire within
`400` elapsed units beside the record's thirteen of fourteen.


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
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/agent_guidance/paper-v/handoffs/codex-paper-v-divergence-parameter-map-20261010-inputs.sha256`
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
~~~
