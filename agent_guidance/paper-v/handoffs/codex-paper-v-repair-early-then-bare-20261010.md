# Codex handoff: repair early, then bare (Paper V classical repair, 2026-10-10)

**What this is.** A standalone, exploratory research-and-implementation
handoff for one Codex session. It is one of four independent threads on the
Paper V classical repair (index:
`prompt-exports/codex-paper-v-INDEX-20261010.md`). It follows
`agent_guidance/shared/agent-handoff-contract.md`: anchors, scope,
increments that each end committable, evidence standards, traps and the
report-back block. Section C (the common rules, at the end) is binding; read
it before increment 0. Nothing here is evidence or manuscript prose.

**The question in one sentence.** If the cheap repair is applied only over
`[0, T_r]` and the unrepaired archive closure takes over from the repaired
state at `T_r`, when do the `1e4` magnitude event and the electronic
positivity loss happen, as a function of `T_r`, and what does each second of
repair wall buy.

**What counts as the answer.** A table, from the stored chain, of the `1e4`
magnitude event time and the electronic positivity loss time of the
unrepaired archive closure started at `T_r` from the repaired state and,
beside it, from the exact state at the same `T_r`, for `T_r` every `0.05`
on `[0, 20]` and every `0.25` on `(20, 60]`, in windows of `400` and `2000`
time units, read at the resolution increment 2 measures; the smallest `T_r`
after which the repaired state fires no more often than the exact state,
with its window, its statistic and the repair wall spent to reach it; and
the same reading for three cheaper repair configurations run over `[0, 10]`.

---

## 0. Identity of this thread

Run this block first in every shell (zsh or bash):

```bash
export TOPIC=repair-early-then-bare
export BRANCH=codex/paper-v-repair-early-then-bare-20261010
export WT=/Users/lordskyler/Holstein_test.worktrees/codex-paper-v-repair-early-then-bare-20261010
export SRC=/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914          # read-only
export PY=/Users/lordskyler/Holstein_test/.venv/bin/python
export OUT=$WT/output/local_runs/paper_v_repair_early_then_bare_codex_20261010
export REPORTFILE=$WT/agent_guidance/paper-v/handoffs/REPORT-repair-early-then-bare-codex-20261010.md
export DATA=$WT/agent_guidance/paper-v/handoffs/repair-early-then-bare-codex-20261010-data
export REC=$SRC/output/local_runs/paper_v_divergence_20260920                               # read-only
export CHAIN=$SRC/output/local_runs/paper_v_divergence_hf_dyk120_20260920                   # read-only; segments ${CHAIN}_seg1 .. ${CHAIN}_seg14
export LOCK=$SRC/agent_guidance/paper-v/handoffs/codex-paper-v-repair-early-then-bare-20261010-inputs.sha256   # = appendix L
```

**Drivers this thread runs.** The bare-closure sweeps run through a new
**committed** module, `paper_5/src/paper5/stability/repair_handover_sweep.py`
(increment 1). The only chained repaired runs (increment 6) use a fixed copy
of the **uncommitted** script
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_divergence_segment.py`
(SHA-256 in appendix L), fixed as C20 decides and gated by C20's parity
check at each setting before its chain starts; increment 0 runs that gate at
the task-11 configuration.

Inputs, all **read-only**, by absolute path (C5 lists what is inside each;
appendix L gives every SHA-256):

* the record's drivers and score files:
  `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/`,
  of which this thread uses `run_unrepaired.py`, `run_closed_route.py`,
  `boundary_basin_sweep.py`, `run_divergence_segment.py`,
  `hartree_fock_preparation.py`, `exact_lift.py`, `score_chain.py`, and the
  score files `boundary_basin_sweep.json` (140-unit sweep, both starts),
  `boundary_basin_span400_repaired.json`,
  `boundary_basin_span400_exact.json`, `hf_chain_score_t20.json`,
  `hf_chain_score_t60.json`;
* the repaired chain:
  `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/`
  through `..._seg14/`, each with `trajectory.npz`, `summary.json`,
  `resume.npz`, `reselections.json`;
* the stored boundaries: the **last sample** (`times[-1]`,
  `archive_coordinates[-1]`) of each segment's `trajectory.npz`.

The record this handoff cites (tracked; readable in your own worktree too):
`agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
(**DI**, by section and line);
`MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`
(**CR**); `paper_5/notes/electron_phonon_closure_worklog_20260803.md`, the
three 2026-09-20 blocks on the divergent initialization, lines 8200 to 8447
(**WL**); `MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md`
(**WN**).

---

## 1. Why this thread exists

The record's last round ran the unrepaired archive closure from the repaired
chain's state at each of its fourteen stored segment boundaries and found
two things that point here (DI section 14, lines 853 to 864; WL lines 8423
to 8429):

* the repair's whole effect on that measurement is delivered **early**: the
  earliest stored boundary from which the unrepaired closure shows no `1e4`
  magnitude event is `t = 9.60` in a `140`-unit window and `t = 20.00` in a
  `400`-unit window, and the `12,990.1 s` spent carrying the chain from
  `t = 20` to `t = 60` changed the measurement no further;
* the repair is **four orders of magnitude dearer** than the closure it
  repairs: the unrepaired closure propagates `140` time units in `2.2 s`,
  `0.0157 s` per time unit, against `217.3 s` per time unit for the repair
  over `t = 20` to `60`, a factor `13,800` (DI section 11, lines 594 to 598).

So the cheapest candidate route the record suggests is a hybrid: repair for a
short declared time `T_r`, then hand the 31 closed scalar coordinates to the
unrepaired archive closure. This thread maps what that hybrid buys.

**The free data.** The repaired chain is causal, so the state of "the repair
applied over `[0, T_r]`" is exactly the stored chain's state at `T_r`. Every
accepted step of the record's chain is stored (section 3), so for every
`T_r` on the `h = 0.0025` grid in `[0, 60]` the handover state already
exists. Increments 0 to 5 run **no repair at all**; only increment 6 runs
new repair rollouts.

---

## 2. Anchors from the record

**Configuration of the stored chain** (DI section 4, lines 162 to 175, and
section 9): the cheap repair in the task-11 configuration (glossary C19)
from the Hartree--Fock zero-correlation preparation, `lambda_ep = 1.5`,
`gamma = 0.5` (so `omega_ph = 0.5`, `g = 0.6123724`), drive
`2 A sin(pi t / 4) exp(-t^2 / 2)` at `A = 1`, phonon cutoff 16. The one
difference from the recorded exact-correlated chain is the preparation and,
forced because the declared completion refuses at a pure product state, the
`exact_moments` lift at `t = 0` (DI section 3).

| fact | number, with definition and horizon | source |
|---|---|---|
| unrepaired archive closure from the Hartree--Fock zero-correlation preparation, 31 closed scalar coordinates, DOP853 `rtol 1e-10`, `atol 1e-12` | retained joint Gram first below `-1e-8` at `t = 0.0124346`; boson-moment eigenvalue first below `-1e-8` at `1.3226952`; electronic one-body density matrix first below `-1e-8` at `27.9633944`; `1e4` magnitude event at `54.4612958` | DI section 2.2, lines 87 to 96 |
| the same closure from the exact correlated preparation | electronic first below `-1e-8` at `1.0362132`; no `1e4` event through `t = 140`, maximum `\|x_i\|` `24.2421` over `t <= 140`; the working notes' 31-coordinate protocol records the event at `141.4056` | DI section 2.2; WN line 416 |
| the repaired chain, `t = 0` to `60` | 14 chained segments, 24,001 samples on one `h = 0.0025` grid, `12,990.1 s` autonomous wall, 1,500 lift re-selections of which 156 were solver refusals, peak resident `656 MiB` | DI section 9, lines 440 to 446 |
| its electronic one-body density matrix | `-2.220e-16` at `t = 0`; minimum over `0 < t <= 60` is `+1.4466e-6`, attained at the first step `t = 0.0025` | DI section 9, lines 458 to 467 |
| its maximum `\|x_i\|` over `t = 0` to `60` | `2.1827` at `t = 51.0675`; `1.3080` for the state at `t = 60` | DI section 9, lines 469 to 474 |
| its Gram cones, minima over `t = 0` to `60` | unshifted retained joint Gram `-8.511e-4`, scaled positive extension `-9.532e-5`, boson-moment `-3.391e-3` | DI section 9 table, line 456 |
| its accuracy | all-coordinate scalar RMS against the exact driven reference from the same Hartree--Fock product state: `1.0803e-2` at `t = 4`, `1.5368e-1` at `t = 20`, `2.5235e-1` at `t = 60` | DI sections 4.1 and 9 |
| segment end times | `4.76`, `9.60`, `13.82`, `18.60`, `20.00`, `24.44`, `29.12`, `34.43`, `38.6325`, `43.12`, `48.48`, `52.71`, `56.92`, `60.00` | DI section 13, lines 707 to 708 |
| segment walls | `1002.2`, `1003.2`, `1000.0`, `1001.3`, `290.2 s` (to `t = 20`, total `4,296.9 s`), then `1005.9`, `1006.6`, `1000.1`, `1000.1`, `1002.9`, `1003.1`, `1000.1`, `1007.2`, `667.3 s` (to `t = 60`, total `12,990.1 s`) | DI section 4.1, lines 186 to 190; section 9, lines 440 to 444 |
| repair cost per time unit | `217.3 s` over `t = 20` to `60`; `4,296.9 s` over `t = 0` to `20` | DI sections 11 and 4.1 |
| unrepaired closure cost | `0.0157 s` per time unit (140 units in `2.2 s`) without per-sample diagnostics; with per-sample diagnostics at `0.0025` the record's sweeps took `75.9 s` for 28 propagations of 140 units and `182.3 s` for 28 of 400 units | DI sections 11 and 15 |

**The boundary sweep, the free first data** (DI section 13; protocol: the
unrepaired archive closure in the 31 closed scalar coordinates, DOP853
`rtol 1e-10`, `atol 1e-12`, samples every `0.0025`, `1e4` terminal event,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude `1.0`, phonon cutoff 16).
"From the exact state" means from the exact driven trajectory **of the same
Hartree--Fock product state** at the same time, not from the exact correlated
ground state. Event times are absolute.

| boundary `T_r` | `140`-unit window, from the repaired state | `140`-unit window, from the exact state | `400`-unit window, from the repaired state | `400`-unit window, from the exact state |
|---:|---|---|---|---|
| `4.7600` | none (maximum `\|x_i\|` `2807.5440` at the window's last sample, still growing) | none | `150.9977` | `190.9565` |
| `9.6000` | none | none | `268.8277` | `193.6878` |
| `13.8200` | none | none | `206.9941` | `256.9274` |
| `18.6000` | none | none | `405.9956` | `313.7340` |
| `20.0000` | none | `147.3861` | none in `400` units | `147.3861` |
| `24.4400` | none | none | `380.5341` | none in `400` units |
| `29.1200` | none | `74.8594` | none | `74.8594` |
| `34.4300` | none | none | none | `304.3658` |
| `38.6325` | none | none | none | `283.3837` |
| `43.1200` | none | `152.9483` | none | `152.9483` |
| `48.4800` | none | none | none | `277.2111` |
| `52.7100` | none | none | `322.3897` | `300.3730` |
| `56.9200` | none | none | none | `375.2728` |
| `60.0000` | none | none | `410.7827` | `433.8964` |

Sources: DI section 13.1 (lines 719 to 751) and 13.2 (lines 774 to 806). At
`400` units the event fires at seven of fourteen boundaries from the repaired
state and thirteen of fourteen from the exact state; the verdict alternates
between adjacent boundaries (`52.71` and `60.00` fire, `56.92` between them
does not), so it carries no onset time (DI section 14, lines 827 to 836).
The window minima of the electronic eigenvalue, the maximum `|x_i|` and the
retained joint Gram for every row are in the same tables and in the JSON
files named in section 0.

**Withdrawn reading you must not repeat.** DI section 11 read one `140`-unit
test at `t = 60` as "the repaired chain has left the divergent basin"; DI
section 13 withdrew it because the window was shorter than the time either
state needs to reach the event. The record's standing rule: any future use
of this test must state its window, and a window that does not exceed the
times reported here decides nothing (WL lines 8445 to 8447). The longest
elapsed time to an event in the table above is `387.3956` (repaired state,
boundary `18.60`).

---

## 3. Code and artifact facts you will rely on (verified from source and artifacts at `b9f99271`, not from the record's prose)

1. **Every accepted step is stored.** Each segment's `trajectory.npz` holds
   the 31 closed scalar coordinates of every accepted step
   (`archive_coordinates`) and the exact driven reference from the same
   Hartree--Fock product state on the same grid (`exact_coordinates`), with
   the cone minima, energy and reference (C5). Segment sample counts are
   1,905, 1,937, 1,689, 1,921, 561, 1,777, 1,873, 2,125, 1,694, 1,809,
   2,145, 1,693, 1,697, 1,233.
2. **Segments overlap.** Each segment resumes from the previous segment's
   `resume.npz`, at its last re-selection boundary. Ten segments end exactly
   there; segments 3, 8, 9 and 12 end a few steps later (`13.82` against
   resume `13.80`, `34.43` against `34.40`, `38.6325` against `38.60`,
   `52.71` against `52.68`), so the next segment re-propagates that tail.
   Concatenate the way the record's `score_chain.py` does: truncate each
   segment at its own `resume.npz` time and drop each later segment's first
   sample, so every time on `[0, 60]` comes from the chain's continuation;
   record the maximum absolute difference between the two copies at every
   overlapping time (the record's chaining agreement is `2.5e-14` through
   three re-selections, CR section 7). The record's boundary states at those
   four times are the earlier segment's tail samples, which this
   concatenation drops.
3. **Cost per segment** is `summary.json` → `rollout.autonomous_seconds` and
   `rollout.completed_steps`; per re-selection seconds are in
   `reselections.json` (`seconds`, one record per re-selection, 119 in
   segment 1).
4. **The handover discards the lift.** The repaired chain propagates 56
   coordinates (the 29 raw moment coordinates and the 27 dictionary moments)
   plus a 70-coordinate lift; `archive_coordinates` is the 31-coordinate
   readout. Handing it to `closed_scalar_rhs` drops the dictionary moments
   and the lift. That is the definition of "then bare" here; state it in the
   report.
5. **The drive is off after about `t = 8`.** `DimerParameters.drive_difference`
   (`paper_5/src/paper5/stability/hubbard_dimer.py:83-88`) is
   `2 A sin(pi t / 4) exp(-(t / pulse_width)^2 / 2)` with `pulse_width = 1.0`;
   at `t = 8` the envelope is `exp(-32)`, about `1.3e-14`. For `T_r >= 8` the
   unrepaired closure started at `T_r` is autonomous to that precision and
   the elapsed time to the event depends on the starting state alone; for
   `T_r < 8` the drive still acts after the handover.
6. **The record's start helpers take only the last sample.**
   `run_closed_route.py --initial-from` starts from the last sample of the
   file it is given; `boundary_basin_sweep.py` takes each segment's last
   sample. You need a driver that starts from an arbitrary stored sample.
7. **The `t = 0` sample is not bit-equal to the analytic preparation.** The
   chain was prepared by contracting the Hartree--Fock product state vector,
   which reproduces `hartree_fock_closed_scalar_coordinates` to `1.02e-10` in
   maximum absolute coordinate difference (DI section 1, lines 32 to 42). The
   record's `54.4612958` was propagated from the analytic coordinates. The
   difference between the two `T_r = 0` event times is a free first
   measurement of the event time's sensitivity.
8. **Two coordinates vanish on every state.** Indices 17 and 18 of the 31
   closed scalar coordinates are the real and imaginary parts of the
   correlation shared trace `Tr C[q]`, which vanishes on every state
   (`run_unrepaired.py`, `_retained_gram_minimum` docstring); under the
   unrepaired closure from the divergent preparation it grows to `5151.44`
   (DI section 2.2, line 96). Do not perturb those two in increment 2.

---

## 4. Scope specific to this thread

* In scope: the handover sweep from stored samples; its resolution; long
  windows; the cost reading; three cheaper repair configurations run over
  `[0, 10]` from the same preparation with the `exact_moments` lift, each
  followed by the same sweep.
* Out of scope: Jacobians, Lyapunov exponents and step-size studies of the
  closure (C4); any change to `closed_scalar_rhs`, the cheap repair's
  defaults or its acceptance rule; the exact correlated preparation (thread
  4 maps it); the 13-coordinate Fan--Migdal system (the repair does not
  propagate it).
* Machine headroom: increments 1 to 5 are many single-core propagations of
  about `260 MB` each; run up to 8 workers after `memory_pressure -Q`.
  Increment 6 runs three repair chains concurrently at about `650 MiB` each.

---

## 5. Increments

Proceed through all of them autonomously (C15).

### Increment 0 — worktree, input lock, readable inputs, driver fix and the parity gate (under 20 minutes)

**Goal.** Your worktree exists at `b9f99271`; every input matches its
SHA-256; the drivers are copied byte for byte; every input opens; the stored
chain's Eq. (22) energy reference is read per segment; the copied segment
driver is fixed as C20 decides; and C20's parity gate passes at the task-11
configuration from the Hartree--Fock zero-correlation preparation with the
`exact_moments` lift.

**Bound.** Each command below finishes in under 20 minutes; the whole
increment takes about 10 minutes (the three short rollouts about 4).

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
import json, os, numpy as np, paper5.stability.matrix_reference as m
print('import', m.__file__)
chain = os.environ['CHAIN']; rec = os.environ['REC']
total, previous = 0, None
for k in range(1, 15):
    base = f'{chain}_seg{k}'
    d = np.load(base + '/trajectory.npz'); s = json.load(open(base + '/summary.json')); r = np.load(base + '/resume.npz')
    total += len(d['times'])
    ref = np.asarray(d['internal_energy_reference'])
    start_ref = float(ref[0])
    carried = None if previous is None else previous
    print(f'seg{k}', d['archive_coordinates'].shape, 'start', float(d['times'][0]), 'end', float(d['times'][-1]),
          'resume', float(r['time']), 'wall', s['rollout']['autonomous_seconds'],
          'reference at start', start_ref, 'previous reference at resume', carried,
          'difference', None if carried is None else abs(start_ref - carried))
    i = int(np.argmin(np.abs(d['times'] - float(r['time']))))
    previous = float(ref[i])
print('samples with overlaps', total)
for f in ('boundary_basin_sweep.json', 'boundary_basin_span400_repaired.json', 'boundary_basin_span400_exact.json', 'hf_chain_score_t60.json'):
    print(f, 'opens', len(json.load(open(f'{rec}/{f}'))))
PY
```

Then fix `$OUT/run_divergence_segment.py` exactly as C20 items 1 to 4 say,
write the parity checker, and run the gate:

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

Segment A completes 40 steps, so its resume point is the re-selection
boundary at step 32, `t = 0.08`; segment B prints the carried reference and
continues to `t = 0.2`. Write `$DATA/input_manifest.json` (expected and
actual SHA-256 of every locked input, of every copy, and of the two edited
drivers) and the per-segment reference table of the stored chain into the
report-back.

**Acceptance.** HEAD is `b9f99271fc3ddf3db604d3acb64617a291bec7e1`; every
lock line prints `OK`; the copies are byte-identical; the import is under
`$WT`; all fourteen stored segments open with `archive_coordinates` of shape
`(n, 31)`, `n` as in section 3 fact 1 (24,059 with overlaps), last times
equal to the fourteen boundary times, autonomous walls summing to
`4,296.9 s` over segments 1 to 5 and `12,990.1 s` over 1 to 14 (each to
`0.1 s`); the stored chain's reference at each segment start differs from
the previous segment's reference at its resume time by at most `1e-12`
(reported per segment: the stored driver re-declared the reference at the
resumed state's energy, which the hyperplane holds to roundoff); and the
parity gate passes as C20 states (strictly increasing spliced times on the
unbroken grid, bit-equal seam state, references equal to `1e-14`, spliced
coordinates within `1e-12` of the unbroken run).
**Stop.** Any lock line not `OK`, any missing input or differing count, an
import outside `$WT`, a stored reference discontinuity above `1e-12`, or a
parity failure (coordinates differing by more than `1e-10`, or a reference by
more than `1e-12`; if the cause is a re-selection decision flipping at the
feasibility test's edge, the record's known non-bitwise mode, CR section 7,
report the two decision records and stop).
Commit `$DATA/input_manifest.json`, `$DATA/drivers/run_divergence_segment.py`
and `$DATA/drivers/parity_check.py`.

### Increment 1 — the handover module, and the fourteen boundaries reproduced

**Goal.** The test baseline is measured before any source change; then a
committed module that starts the unrepaired archive closure from any stored
sample of a repaired chain, and from the exact driven trajectory's state at
the same time, reproduces every event time of section 2's table.

**Bound.** The suite takes about `250` to `265 s`; each sweep command below
takes under 5 minutes on 8 workers and carries `--max-wall-seconds 1080`.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q   # the baseline (C16), before any edit
```

Write `paper_5/src/paper5/stability/repair_handover_sweep.py` with this
command-line contract (flags may be added, not renamed). It writes one JSON
line per finished start as it goes; `--resume` skips starts already in the
output; `--max-wall-seconds` stops cleanly, marks unfinished starts
`incomplete` and exits with a message saying the command must be repeated;
the final summary JSON is written when every start is done (C10):

```text
python -m paper5.stability.repair_handover_sweep
  --segments DIR [DIR ...]          chain segment directories, read-only
  --boundaries                      start at each segment's last sample (the record's boundary states)
  --start-times T [T ...]           start at these stored times (must lie on the stored grid)
  --start-grid START STOP STEP      start at every stored time START, START+STEP, ..., <= STOP
  --from {repaired,exact,both}
  --window W                        propagate from T_r to T_r + W
  --only-without-event-in JSON      restrict starts to those with no event in that earlier sweep
  --neighbours K                    also start from the K stored samples on each side of every T_r
  --perturbation-size S --perturbation-draws N --perturbation-seed SEED
                                    N seeded relative perturbations of size S of the 29 coordinates other than indices 17, 18
  --relative-tolerance 1e-10 --absolute-tolerance 1e-12 --sample-step 0.0025
  --workers N
  --max-wall-seconds S --resume     the 20-minute bound (C10)
  --compare-to JSON [JSON ...]      print the largest event-time difference against record sweep files
  --output JSON
```

Protocol, unchanged from `boundary_basin_sweep.py`: `closed_scalar_rhs` with
`DimerParameters(gamma=0.5, lambda_ep=1.5, drive_amplitude=1.0)`, DOP853,
the stated tolerances, `t_eval` every `0.0025` clipped to the interval, the
`1e4` terminal event; propagation starts at the clock time `T_r`. Exact
states come from `_reference_coordinates(parameters, grid, preparation="hartree_fock", phonon_cutoff=16, time_step=0.0025)`
(one propagation for all requested times; import it from your copy
`$OUT/run_divergence_segment.py`), and the module reports their largest
absolute difference to the stored `exact_coordinates` at the same times.
Per start it records: event time (absolute and elapsed) or `null`; first
crossings of the electronic one-body density matrix minimum eigenvalue below
`0`, `-1e-8`, `-1e-3`; first crossings below `-1e-8` of the boson-moment
eigenvalue and of the unshifted retained joint Gram; window minima and the
maximum `|x_i|` with its time; wall seconds; plus the configuration block of
`boundary_basin_sweep.py` and
`"classification": "exploratory_local_not_promoted"`. Concatenation follows
section 3 fact 2. Test file `paper_5/tests/test_repair_handover_sweep.py`:
from the analytic `hartree_fock_closed_scalar_coordinates` at `T_r = 0`,
`W = 60`, the event time is `54.4612958` and the electronic crossing below
`-1e-8` is `27.9633944`, each to `1e-6` (about `1 s`; tests must not read the
source worktree).

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q tests/test_repair_handover_sweep.py
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --segments "$CHAIN"_seg{1..14} --boundaries --from both --window 400 --workers 8 --max-wall-seconds 1080 --resume \
  --compare-to "$REC/boundary_basin_span400_repaired.json" "$REC/boundary_basin_span400_exact.json" \
  --output "$OUT/boundaries_w400.json"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --segments "$CHAIN"_seg{1..14} --boundaries --from both --window 140 --workers 8 --max-wall-seconds 1080 --resume \
  --compare-to "$REC/boundary_basin_sweep.json" --output "$OUT/boundaries_w140.json"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --segments "$CHAIN"_seg{1..14} --start-times 0 --from both --window 400 --max-wall-seconds 1080 --resume --output "$OUT/start_t0_w400.json"
cp "$OUT"/boundaries_w400.json "$OUT"/boundaries_w140.json "$OUT"/start_t0_w400.json "$DATA"/
```

**Acceptance.** The test passes. All 28 starts at `W = 400` and all 28 at
`W = 140` agree with the record: same `null` pattern, every event time within
`1e-4` (for example `410.7827` from the repaired state at `60.00`, `433.8964`
from the exact state at `60.00`, `null` from the repaired state at `20.00`,
`null` from the exact state at `24.44`), every window minimum to the
record's printed digits. The `T_r = 0` row from the stored sample is reported
beside the analytic `54.4612958` and `27.9633944` with its difference.
**Stop.** Any event time differing from the record by more than `1e-4`, or
any `null` mismatch: the harness differs; report the row and stop.
Commit the module, its test and the three JSON copies under `$DATA`.

### Increment 2 — resolution: how many digits of an event time mean anything

**Goal.** A measured spread of the event time under declared small changes
of the starting state and of the integrator tolerance, at twelve `T_r`. C4a
gives the scale at one start, the exact state at `t = 60`: `7.1e-3` at
`1e-12`, `1.31` at `1e-10`, `0.3` from `rtol 1e-10` to `1e-12`, and `293` to
`478` (exact) or `296` to `779` (repaired) elapsed units across `1e-4`
perturbations; this increment measures it at the handover starts.

**Bound.** About 600 starts of `400` units, about 8 minutes on 8 workers in
four invocations; each stops itself at `1080 s` and is repeated if
incomplete.

```bash
T12="0 2 4.76 6 9.6 12 20 24.44 30 40 52.71 60"
cd "$WT" && for spec in "neighbours 2 0 0" "perturb 0 1e-10 8" "perturb 0 1e-8 8"; do
  set -- $spec   # zsh: use ${=spec} instead of $spec
  PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
    --segments "$CHAIN"_seg{1..14} --start-times ${=T12} --from both --window 400 --workers 8 --max-wall-seconds 1080 --resume \
    --neighbours $2 --perturbation-size $3 --perturbation-draws $4 --perturbation-seed 20261010 \
    --output "$OUT/resolution_$1_$3.json"
done
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --segments "$CHAIN"_seg{1..14} --start-times ${=T12} --from both --window 400 --workers 8 --max-wall-seconds 1080 --resume \
  --relative-tolerance 1e-12 --absolute-tolerance 1e-14 --output "$OUT/resolution_tolerance.json"
```

(In bash write `$T12` and `$spec` unquoted; in zsh write `${=T12}` and
`${=spec}`, because zsh does not word-split an unquoted scalar.)

Report per `T_r`, per state and per kind (neighbours, `1e-10`, `1e-8`,
tolerance): how many starts fire within `400` units, and the spread (largest
minus smallest) of the event times among those that fire. Do not compute
Jacobians or exponents (C4).

**Acceptance.** Every start of the four runs (about 600: twelve `T_r`, two
states, four neighbours, sixteen perturbations, one tolerance change, plus
the unperturbed starts) completes with either solver success or the terminal
event. Decision,
in numbers, recorded in `$REPORTFILE`: if the median over the twelve `T_r`
of the `1e-10` spread is below `1` time unit, increment 3 reports curves;
otherwise it reports, per 1-unit `T_r` bin, the fraction of starts firing
within the window and the quartiles of the event times. If the tolerance
change alone moves the event time by more than the `1e-10` spread at seven or
more of the twelve `T_r`, increments 3 to 6 use `rtol 1e-12`, `atol 1e-14`.
**Stop.** More than 5 percent of the starts end in a solver error other than
the terminal event.
Commit `$DATA/resolution_summary.json` (spreads and counts, not the
per-start rows if those exceed 1 MB).

### Increment 3 — the dense map from stored samples

**Goal.** Event time and electronic positivity loss time of the unrepaired
archive closure from the repaired state and from the exact state, for every
`T_r` in the dense grid, `W = 400`.

```bash
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --segments "$CHAIN"_seg{1..14} --start-grid 0 20 0.05 --from both --window 400 --workers 8 --max-wall-seconds 1080 --resume \
  --output "$OUT/map_0_20_w400.json"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --segments "$CHAIN"_seg{1..14} --start-grid 20.25 60 0.25 --from both --window 400 --workers 8 --max-wall-seconds 1080 --resume \
  --output "$OUT/map_20_60_w400.json"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --segments "$CHAIN"_seg{1..14} --start-times 4.76 13.82 24.44 29.12 34.43 38.6325 43.12 48.48 52.71 56.92 \
  --from both --window 400 --workers 8 --max-wall-seconds 1080 --resume --output "$OUT/map_offgrid_boundaries_w400.json"
```

Starts are matched to stored times by the nearest index of the `0.0025`
grid, to `1e-9`. The module sorts the segment directories by their first
stored time, so the order on the command line does not matter.

**Bound.** About `6.5 s` per start with per-sample diagnostics (the record's
`182.3 s` for 28 starts of 400 units): 802 starts are about 11 minutes and
320 about 5 minutes on 8 workers. Each invocation stops itself at
`1080 s`; if one reports itself incomplete, repeat the identical command,
which resumes. You may add a cheaper diagnostic path
(window diagnostics on a `0.05` grid, electronic crossings by non-terminal
event functions at `-1e-8` and `-1e-3`) as a named flag only after showing
on the 28 boundary starts that it reproduces event times and crossing times
to `1e-4`.

**Acceptance.** 802 starts on `[0, 20]` and 320 on `(20, 60]` complete. At
the boundaries that lie on the dense grids (`9.60`, `18.60`, `20.00`,
`60.00`) and at the six off-grid boundaries that are exact segment ends
(`4.76`, `24.44`, `29.12`, `43.12`, `48.48`, `56.92`) the event times are
bit-equal to increment 1; at the four tail boundaries (`13.82`, `34.43`,
`38.6325`, `52.71`), where the map uses the continuation segment's sample
rather than the record's tail sample, the coordinate difference and the
event-time difference from increment 1 are reported (an event-time
difference larger than the increment-2 `1e-10` spread there is a surprise to
report). The thread's answer to the first
half of the question is the smallest `T_r` after which, in every later
1-unit bin, the fraction of repaired-state starts firing within `400` units
is no larger than the exact-state fraction in the same bin; report it with
the window, the statistic, and the resolution decision of increment 2. If no
such `T_r` exists below `60`, say so with the two fractions per bin.
**Stop.** More than 5 percent (57 or more) of starts end in a solver error
other than the terminal event.
Commit `$DATA/map_w400_bins.json` (per-bin fractions and quartiles; the
per-start rows stay under `$OUT` unless under 1 MB).

### Increment 4 — long windows

**Goal.** Know which "no event in 400 units" starts are long-lived and which
are merely late.

```bash
cd "$WT" && for part in 0_20 20_60; do
  PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
    --segments "$CHAIN"_seg{1..14} --only-without-event-in "$OUT/map_${part}_w400.json" \
    --from both --window 2000 --workers 8 --max-wall-seconds 1080 --resume --output "$OUT/map_${part}_w2000.json"
done
```

(Window diagnostics on a `0.05` grid are acceptable here if the flag of
increment 3 was validated; say which was used.)

**Bound.** A `2000`-unit start costs about `30 s` with per-sample
diagnostics; each invocation stops itself at `1080 s` and is repeated until
complete. A start the bound interrupts is `incomplete`, never "no event".

**Acceptance.** Every start without an event in `400` units is carried to
`2000` units or to its event. Reported per state and per 10-unit `T_r` bin:
the counts with no event in `400` and in `2000` units, and the quartiles of
the event times between `400` and `2000`. A start with no event in `2000`
units is reported as exactly that, never as bounded.
**Stop.** More than 5 percent of the carried starts end in a solver error.
Commit `$DATA/map_w2000_bins.json`.

### Increment 5 — the cost reading: what `T_r` buys per second of repair

**Goal.** For each `T_r`, the repair wall spent to reach it beside what the
handover measures.

Add a `--cost-reading` mode to the module (or a second module,
`paper5.stability.repair_handover_cost`) that reads the segments' walls and
the two maps (**Bound.** reading only, under a minute):

```bash
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
  --cost-reading --segments "$CHAIN"_seg{1..14} \
  --maps "$OUT"/map_0_20_w400.json "$OUT"/map_20_60_w400.json "$OUT"/map_0_20_w2000.json "$OUT"/map_20_60_w2000.json \
  --output "$OUT/cost_reading.json"
cp "$OUT/cost_reading.json" "$DATA/"
```

It reports: the cumulative autonomous wall from `t = 0` to `T_r`, exact at
segment ends (sum of `rollout.autonomous_seconds`) and linear in completed
steps inside a segment, labelled as interpolation (a better interpolation
from `reselections.json` is welcome; re-selection carries `0.803` of the wall
on the recorded exact-correlated chain, CR section 2(b)); and, defined here
in a sentence before any label: for each `T_r`, the `1e4` magnitude event
time of the unrepaired archive closure started from the repaired state at
`T_r`, minus `54.4612958` (the same closure's event time from the
Hartree--Fock zero-correlation preparation at `t = 0`), divided by the
cumulative autonomous wall to `T_r`; where the start has no event within the
window the numerator is the lower bound `T_r + W - 54.4612958`, marked as a
bound. The same with the electronic positivity loss time against
`27.9633944`. The exact-state numerators at the same `T_r` stand beside them.
A table at `T_r` in `{0, 1, 2, 4, 4.76, 6, 8, 9.60, 12, 16, 20, 30, 40, 60}`
gives, over the increment-2 neighbourhood of each `T_r`, the fraction firing
within `400` and within `2000` units, the median event time, the cumulative
wall and the ratio.

**Acceptance.** The cumulative walls at the fourteen segment ends reproduce
the sums of `summary.json` (`4,296.9 s` at `t = 20` and `12,990.1 s` at
`t = 60`, to `0.1 s`). The thread's answer to the second half of the
question is the cumulative wall at the `T_r` found in increment 3, beside
`12,990.1 s` for repairing throughout `t = 0` to `60`.
**Stop.** Walls failing to reproduce the sums by more than `0.1 s`.
Commit `$DATA/cost_reading.json`.

### Increment 6 — cheaper repairs over `[0, 10]`, each followed by the same sweep

**Goal.** Whether a cheaper repair buys the same handover distribution as
the task-11 configuration for less wall.

Three named variants, each a chain from `t = 0` to `t = 10` from the
Hartree--Fock zero-correlation preparation with `--lift exact_moments`
(recorded in `summary.json`), run through your **fixed** copy of the
uncommitted `run_divergence_segment.py` (C20), one segment of at most
`900 s` per process, everything else the task-11 configuration (the
defaults of that driver): **(a)** the 15 entrance keys,
`--dictionary entrance15`; **(b)** 27 keys with
`--lift-reselection-interval 64`; **(c)** 27 keys with
`--energy-reference-mode none`, the task-13 setting, the one 27-key chain on
record that held both cones from the exact correlated preparation
(unshifted retained joint Gram `-1.549e-12`, scaled positive extension
`-1.350e-9` over `t <= 20`, at `6.36` times the task-11 score at `t = 20`;
CR section 4, item 6).

**Step 1, the parity gate at each variant's own setting (C20).** Bound:
three rollouts per variant, each under 5 minutes. Variant (b) needs a resume
point at a multiple of 64 steps, so its short runs are longer.

```bash
parity_variant () {   # usage: parity_variant NAME T_FIRST T_TOTAL EXTRA_FLAGS...
  local name=$1 t1=$2 t2=$3; shift 3
  local SEG="$OUT/run_divergence_segment.py" P="$OUT/parity_${name}"
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" --preparation hartree_fock --lift exact_moments \
    --output-directory "${P}_unbroken" --final-time "$t2" --wall-budget-seconds 900 --record-reselections "$@") || return 1
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" --preparation hartree_fock --lift exact_moments \
    --output-directory "${P}_segA" --final-time "$t1" --wall-budget-seconds 900 --record-reselections "$@") || return 1
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$SEG" --preparation hartree_fock --lift exact_moments \
    --output-directory "${P}_segB" --resume-from "${P}_segA" --final-time "$t2" --wall-budget-seconds 900 --record-reselections "$@") || return 1
  /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/parity_check.py" "${P}_unbroken" "${P}_segA" "${P}_segB"
}
parity_variant entrance15 0.1 0.2 --dictionary entrance15
parity_variant interval64 0.2 0.4 --lift-reselection-interval 64
parity_variant energynone 0.1 0.2 --energy-reference-mode none
```

**Step 2, the chains.** Each `run_divergence_segment.py` invocation is one
segment of at most `900 s` of propagation (under 20 minutes with its
preparation and exact reference), resumed from the previous segment's
checkpoint with the carried reference printed (C20). The loop below only
sequences those invocations.

```bash
run_variant () {   # usage: run_variant NAME EXTRA_FLAGS...
  local name=$1; shift
  local prev="" k out
  for k in {1..15}; do
    out="$OUT/variant_${name}_seg$k"
    if [[ -f "$out/summary.json" ]]; then prev=$out; continue; fi
    if [[ -e "$out" ]]; then echo "refusing to reuse $out (C15); rename and retry"; return 1; fi
    local resume=()
    [[ -n "$prev" ]] && resume=(--resume-from "$prev")
    (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/run_divergence_segment.py" \
      --preparation hartree_fock --lift exact_moments --output-directory "$out" \
      --final-time 10.0 --wall-budget-seconds 900 --record-reselections "${resume[@]}" "$@") || return 1
    prev=$out
    /Users/lordskyler/Holstein_test/.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$out/summary.json'))['rollout']['final_time_reached'] >= 10.0 - 1e-9 else 1)" && break
  done
}
memory_pressure -Q
run_variant entrance15 --dictionary entrance15 > "$OUT/variant_entrance15.log" 2>&1 &
run_variant interval64 --lift-reselection-interval 64 > "$OUT/variant_interval64.log" 2>&1 &
run_variant energynone --energy-reference-mode none > "$OUT/variant_energynone.log" 2>&1 &
wait
grep -h "carried Eq. (22) energy reference" "$OUT"/variant_*.log      # one line per resumed segment
for name in entrance15 interval64 energynone; do
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.repair_handover_sweep \
    --segments "$OUT"/variant_${name}_seg* --start-grid 0 10 0.25 --from both --window 400 --workers 4 --max-wall-seconds 1080 --resume \
    --output "$OUT/variant_${name}_map_w400.json")
  ordered=$(cd "$OUT" && ls -d variant_${name}_seg* | awk -F seg '{print $NF, $0}' | sort -n | cut -d' ' -f2 | sed "s|^|$OUT/|")
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/score_chain.py" \
    $(echo $ordered) --horizons 1 2 4 8 10 --output "$OUT/variant_${name}_score_t10.json")
done
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/score_chain.py" \
  "$CHAIN"_seg{1..5} --horizons 1 2 4 8 10 --output "$OUT/stored_chain_score_t10.json"
```

(`score_chain.py` concatenates segments in the order given, and a shell glob
sorts `seg10` before `seg2`; the `ordered` line puts them in numeric order.
The loop skips segments whose `summary.json` exists and refuses to reuse a
directory without one (C15); the fixed driver creates its directory only
after preparation succeeds (C20). Record the carried reference printed by
every resumed segment in the report.)

Report per variant, beside the stored chain at the same horizons:
autonomous wall to `t = 10`; lift re-selections and solver refusals;
electronic one-body density matrix minimum over `0 < t <= 10`; minimum
unshifted retained joint Gram and scaled positive extension over `t <= 10`;
maximum `|x_i|`; all-coordinate scalar RMS against the exact driven reference
from the same Hartree--Fock product state at `t = 4`, `8`, `10`; and the
increment-5 table on `[0, 10]`.

**Acceptance.** Each variant passes its parity gate (C20 criteria) and then
reaches `t = 10` or is skipped with its message; every resumed segment
printed a carried reference equal to its predecessor's at the resume time to
`1e-14` (variant (c) prints `none`). The answer: a variant buys the same handover for less if, at the
`T_r` found in increment 3 (or `T_r = 10` if that `T_r` exceeds 10), its
fraction of starts firing within `400` units over the increment-2
neighbourhood is no larger than the stored chain's while its cumulative wall
is lower; report the walls and fractions either way.
**Stop, per variant.** Its parity gate fails (C20): do not start its chain.
Its preparation fails (for example `exact_moment_lift`
raising `KeyError` because a frontier key exceeds degree 5, or any
`RuntimeError` at preparation): skip it and report the message. Its first
segment's autonomous wall per time unit exceeds `215 s` (the task-11
configuration's `4,296.9 s / 20`): stop it after that segment, because a
variant that is not cheaper answers nothing here. A variant whose electronic
one-body density matrix falls below `-1e-8` before `t = 10` is finished and
reported; that is a result.
Commit `$DATA/variants_t10.json` (walls, physicality, scores, handover
tables).

### Increment 7 — the report-back file and the final commit

Write `$REPORTFILE` per C17: the report-back block of every increment, the
findings (negative result first; one table per increment; every number with
definition, configuration and horizon), the absolute paths of every
trajectory and sweep under `$OUT`, the commit SHAs, and the proposed worklog
block. Then:

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q
git -C "$WT" add agent_guidance/paper-v/handoffs/REPORT-repair-early-then-bare-codex-20261010.md agent_guidance/paper-v/handoffs/repair-early-then-bare-codex-20261010-data
git -C "$WT" commit -m "paper-v (Codex): repair early then bare — <the answer with its numbers>"
```

**Bound.** The suite takes about `250` to `265 s`.

**Acceptance.** The suite stands at the increment-1 baseline plus the new
tests with no new failure; `git -C "$WT" status --short` shows nothing staged
or modified under tracked paths.
**Stop.** A new test failure: fix it if it is yours, otherwise report it.

---

## 6. Evidence standards for this thread

* Reproduce before extending: increment 1 must match the record before any
  new number is reported.
* Resolution before the map: no event-time curve is drawn at a resolution
  finer than increment 2 measured.
* Every "no event" carries its window, and "no event in `W` units" is never
  written as bounded or stable.
* The exact-state start at the same `T_r` is always reported beside the
  repaired-state start; a statement about the repaired state alone, without
  that control, is not a finding.
* Walls come from `summary.json`; any interpolated wall is labelled.

## 7. Traps

* Your worktree has none of the stored chain or the drivers (C5); read them
  at the absolute paths in section 0, and never write there.
* The record's `run_chain.sh` hardwires `ROOT` to the source worktree (it
  would import the source worktree's modules) and wraps each segment in the
  retired `ram_guard`; increment 6 uses its own loop instead.
* The record's segment driver does not carry the Eq. (22) energy reference
  across a resume and rebuilds the `t = 0` completion on every resume
  (C20); your fixed copy does both correctly, and the parity gate proves it
  at each setting before a chain starts.
* `run_closed_route.py --initial-from` uses the **last** sample of the file
  it is given; it cannot start at an arbitrary `T_r`.
* Segments overlap; four of them by more than one sample (section 3, fact 2).
* The exact states in the record's sweep come from the exact trajectory of
  the Hartree--Fock product state, not of the exact correlated ground state.
* The 13-coordinate Fan--Migdal form gives different times (electronic loss
  `15.5177447`, `1e4` at `130.4674620`; DI section 2.1) from the 31 closed
  scalar coordinates (`27.9633944`, `54.4612958`). This thread uses the 31.
* Chaining is not bitwise: a chained run agrees with a single rollout to
  `2.5e-14` through three re-selections and can flip a fourth at the
  feasibility test's edge (CR section 7). Each increment-6 variant is one
  realization; say so.
* The scaled positive extension does not exist after the handover; never
  report one for the unrepaired continuation (DI section 8, lines 432 to
  436).
* zsh does not word-split an unquoted scalar; the commands above note where
  that matters.

## 8. Report back

`$REPORTFILE` per C17, committed on your branch. Its first lines state, one
sentence each: the smallest `T_r` (with window, statistic and resolution)
after which the repaired state fires no more often than the exact state; the
repair wall to reach it; and whether any increment-6 variant buys the same
for less.


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
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/agent_guidance/paper-v/handoffs/codex-paper-v-repair-early-then-bare-20261010-inputs.sha256`
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
d6752cd7c5db4dfba5472a820190e687bd544ef4b6bac2b8d4c516b5bd61bc40  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg10/trajectory.npz
ecf20a6e3618278a2bf953e6065049fe8c48b641f966679f0926127fdf21ca9e  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg11/trajectory.npz
2c534d515b00dd4d941e9d8dcedba3999dd3384a3017c07efcb9637cbb9d6fce  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg12/trajectory.npz
3da423deb5291333345ef573b0920aa54b3468898b26cde2a4b4be8b0639fc6c  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg13/trajectory.npz
b9f7f2cc847c4ee5c2315eb6e6167bc97e2e7ceb0b051cdbf4e7ed06c5cbf112  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg14/trajectory.npz
5b83103d03566a1094077921ae6db02bc15f7b7d2a2b02ba3435d3c324611062  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/trajectory.npz
a64012362e1be6024dd5044ce29af5aaf1707a069e1bab9f016813765bb15780  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/trajectory.npz
6af709eccb04d2dfbd27f9f646c50566ae345bc978649cff2953e496e400c62d  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/trajectory.npz
ec1d0c3feb73f24e44b9bc7b0efd27a690e4a7e6db0fb2c785048785489149f9  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/trajectory.npz
4a1949cb21fed530294f1985fed0e24ea19b1f3ed6624e243ad88131c7157f4d  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg6/trajectory.npz
ae358252e88d22e8d2f6838746452b419a9e5769088e6e6153a8e18bcc937f47  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg7/trajectory.npz
dc0dfa74b605a86d9470f19982d8122b26eadbce7aeabc253b60ea55860171b2  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg8/trajectory.npz
1f6cbd76437763ccd60c0e4a628a6665dd67f5b30fd47d3528403f438adf90ac  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg9/trajectory.npz
a359ff7f2bee7e08f7d1b6b78cf68d39842c401c669bfbb62a32dd7ca9495def  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/summary.json
705eaa8760c54e0400c1ccbaf4bb43023812fc387c498daf8951ab605a24b0f2  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg10/summary.json
62cad8482a9fb3b93ad5e88b108b860334c02a507d2a349bd2ec7eb2cd88fcff  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg11/summary.json
4e7bc05e6996757b056369d6720f2b3c525cb97e53aabdda8cd30942b725fda4  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg12/summary.json
bbbee6bd48fc103e1238ae65e28beac7af7ebcdceabdc843d2895f4b84688207  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg13/summary.json
716fd2e088851e72a9c241aee4d29dc7a126244351a55990e6fa0d3b509f4e86  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg14/summary.json
71cffcb00b5b282b0377f4b4b76168dc00af9738aa51afff42fa218104c125dd  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/summary.json
84f7792cb6496f751500e7621dd7b16cb318f8ebc3b371c8bf1a81eef5114b8e  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/summary.json
8097edb45718a75748943b3f736e3825fb64cd75adec1064f880202a86deb0c0  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/summary.json
ce9a4239181a4b59f4d09740c32a188327060fc1b3a6e99be035511fd647646e  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/summary.json
d5d7b2ac2d224824531bc75013c7f745c8bb2f873bdfaf4cdfef693deb1579f6  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg6/summary.json
0a540c4ab6ec3fb8c37f875c8c43bcb821ac3b8db6828e13d343ad182b146db0  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg7/summary.json
5ce9922e8e531aa63d9c582c7fe8640f5ba026098e3fc714bec2d2e6dd392067  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg8/summary.json
9b3fdf8d4f66c5606bf86b88f335d6e9a2806f1a0e0d0f64c340c59e4a9a71ff  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg9/summary.json
53894713c5da93e553dc8d3577e6005d5f3d44713721fca5869d18e634b9e0c9  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/resume.npz
e415574027a70cc2e0aa0ea732833a4c270ac3fe5f6b57fb422b37511ca8ff62  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg10/resume.npz
1e24129ad743e7e1a4660314bbfc25eadfd7d0e2888253618edf673c7b67b6f3  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg11/resume.npz
922d3b3db29f2f98a79f88ed523fe31937bde27e19cb9b44f77861ed9cc942b8  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg12/resume.npz
a6639e22601472fbf6b8dc7bfb2d6e46b942ca2d36e62ae349d6a3ae739ad0a0  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg13/resume.npz
83c9c3cd372b377f81e4b69ec81c3b15e8aae073a6c61c1fb81161e4f5895945  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg14/resume.npz
182a1f10b74b6c3a2151a6fc10ddf4ba3a2009ec4c355c1a11d0c861e21c63f1  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/resume.npz
b60f6d0b5f2c205dc5dcbe63bdd70d70e59f40973ed618017f71edebc7a01e58  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/resume.npz
6de343565a4e3b30770c6a3002e4369155925ad1ba7fef17dff2efad52a2268c  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/resume.npz
52b7531afb58d9fce3ed195d938e1dd5c6065a5e35426aee55f2140879f99965  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/resume.npz
84a0236ecdeff3d3659ffb472043c131081d5c99e5d7594542e3a6b843b68275  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg6/resume.npz
a3c19ed655573fc0fe7b72389cebbd5ad0def54bd39966990c291fc1e26ecb83  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg7/resume.npz
0eaca78f4571920bace475dae6138fe1ad3f1310c360fc85858775c1e8af98dc  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg8/resume.npz
f4ef98d0c00ec611c0c53266fc0cf232bb8956e9ab3932617fcc4d6ab9f608c2  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg9/resume.npz
6871f6cf30c2ac9defd3e8ab54478cd60cb96269fbda67acc9be5aa0d5be0540  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/reselections.json
90d5f6f60a32682759ba17fabb77ce2a8721e4deb8e49cbff29805fe04ab8820  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg10/reselections.json
9a0f1adde9513c5d349f89ead7938fe416a38ec93ae7f07d517987c4d5eead67  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg11/reselections.json
f1bcb473929977f6da5b3100a2d1a5cb383abf7cc2b444c19cdc8b7722e8fd44  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg12/reselections.json
2c65626094c1853bc455cc8a13be44693a0830267759847b23c046c419ba5e4f  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg13/reselections.json
3404ec152c6ddfdbfb035445205bc877a85ada1394910733fdc7874ea1f95d58  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg14/reselections.json
08d183dee9247bd0667a2f73cb16ecc56a5f6270ae1085767aae59efc1a06112  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg2/reselections.json
39d43c853f5321527283e10f4d779bdb4b1a8b3847bdfbc1691eaa65ac725414  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg3/reselections.json
7544dfb150593bb8a0d43f13460918a6e12d71faa11b6c79e6f212e0d08858db  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg4/reselections.json
bbac7abe19050736fa98cd64e0c06554b93600f14c663876d149ed246afbb48d  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg5/reselections.json
cddb253cf0c60ba3a2a7f774770e8b4afa80e681243ea21d813a1d8a9ab6e84f  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg6/reselections.json
ac4c2432ac6e2715c5434ed979298df10e32a64119e4db08191d9b9ad6eb9faa  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg7/reselections.json
a5b8467a84a731b1e3d8f4fec1c405b298e612dfb1dcb72842eeee7cad7c20c2  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg8/reselections.json
f634b363bccfe1171fe0d30df1ee6f135daafd1a22246ad21b75e27bb76f7fc2  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg9/reselections.json
~~~
