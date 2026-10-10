# Codex handoff: the electronic density cone as a projected set (Paper V classical repair, 2026-10-10)

**What this is.** A standalone, exploratory research-and-implementation
handoff for one Codex session. It is one of four independent threads on the
Paper V classical repair (index:
`prompt-exports/codex-paper-v-INDEX-20261010.md`). It follows
`agent_guidance/shared/agent-handoff-contract.md`: anchors, scope,
increments that each end committable, evidence standards, traps and the
report-back block. Section C (the common rules, at the end) is binding; read
it before increment 0. Nothing here is evidence or manuscript prose.

**The question in one sentence.** Electronic positivity under the cheap
repair is a measured outcome, not a construction, because the electronic
one-body density matrix is not one of the sets the Dykstra alternation
projects onto; add it as a named projection target, run the divergent
initialization over `t = 0` to `20` and the exact correlated preparation over
`t = 0` to `4` with and without it, and report whether anything changes and
at what cost.

**What counts as the answer.** (i) Whether the electronic cone is already
implied by one of the projected sets (a measured identity on every stored
sample); (ii) how often, at the stage inputs the repair actually sees, the
electronic one-body density matrix is outside its cone; (iii) a named,
default-off option that puts the electronic cone into the alternation, with
the number of stage repairs at which it acts; (iv) for the two
preparations, the record's rows with the option on, beside the rows with it
off, with the wall cost of the option.

---

## 0. Identity of this thread

Run this block first in every shell (zsh or bash):

```bash
export TOPIC=electronic-cone-projection
export BRANCH=codex/paper-v-electronic-cone-projection-20261010
export WT=/Users/lordskyler/Holstein_test.worktrees/codex-paper-v-electronic-cone-projection-20261010
export SRC=/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914          # read-only
export PY=/Users/lordskyler/Holstein_test/.venv/bin/python
export OUT=$WT/output/local_runs/paper_v_electronic_cone_projection_codex_20261010
export REPORTFILE=$WT/agent_guidance/paper-v/handoffs/REPORT-electronic-cone-projection-codex-20261010.md
export DATA=$WT/agent_guidance/paper-v/handoffs/electronic-cone-projection-codex-20261010-data
export REC=$SRC/output/local_runs/paper_v_divergence_20260920                               # read-only
export CHAIN=$SRC/output/local_runs/paper_v_divergence_hf_dyk120_20260920                   # read-only; segments _seg1 .. _seg5 cover t = 0 to 20
export T11DIR=$SRC/output/local_runs/paper_v_projective_dictionary_energy_20260915e         # read-only; task-11 drivers and exact cache
export T11=$SRC/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m   # read-only; segments _seg1 .. _seg6
export LOCK=$SRC/agent_guidance/paper-v/handoffs/codex-paper-v-electronic-cone-projection-20261010-inputs.sha256   # = appendix L
```

**Drivers this thread runs.** The audit is a new **committed** module,
`paper_5/src/paper5/stability/electronic_cone_audit.py` (increment 1), and
the option is a change to the committed `cheap_projection_route.py`
(increment 3; this thread is its only editor, C3). The chained repaired runs
use fixed copies of two **uncommitted** scripts:
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/run_divergence_segment.py`
(the divergent initialization) and
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/run_energy_segment.py`
(the task-11 chain), both with the reference-carry defect of C20 (SHA-256 in
appendix L). Increment 0 fixes both and runs C20's parity gate for each at
the record's settings; increments 5 and 6 rerun the gate with the option on
before their chains. The record's `run_cone_route.py` calls the joint
electronic--bosonic barrier of `cone_correction.py`, a different route; it
is not used here.

Inputs, all **read-only**, by absolute path (C5 lists what is inside each):

* `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_20260920/`:
  `run_divergence_segment.py` (the cheap repair from the Hartree--Fock
  zero-correlation preparation; `--lift exact_moments`), its siblings
  `hartree_fock_preparation.py` and `exact_lift.py`, `score_chain.py`, and
  `hf_chain_score_t20.json` (the record's rows);
* the record's repaired chain from the Hartree--Fock zero-correlation
  preparation:
  `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg1/`
  through `..._seg5/` (`trajectory.npz` holds the 31 closed scalar
  coordinates of every accepted step but not the 56-coordinate states;
  `resume.npz` holds the 56-coordinate state and the lift at each segment's
  last re-selection boundary);
* the recorded exact-correlated task-11 chain:
  `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1/`
  through `..._seg6/` (`trajectory.npz` holds `states` (56) and
  `frontier_standardized` (70) at every accepted step, which is what makes a
  stage-input replay possible; segment 1 covers `[0, 3.955]` and resumes at
  `3.92`, segment 2 covers `[3.92, 8.3225]`);
* its drivers and the exact correlated reference cache:
  `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/`:
  `run_energy_segment.py` (the task-11 driver), `score_energy_chain.py`,
  `exact_t20.npz`, `exact_t8.npz`, `task11_base_score_t20.json`, and
  `probe_cone_correction_share_20260918.py` (replays the three SSPRK(3,3)
  stage repairs from stored samples; its `--chain` is a directory-name
  prefix resolved against `_LOCAL_RUNS = Path(__file__).resolve().parents[1]`,
  line 73, which in your copy would point at your own `output/local_runs/`;
  set it to `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs`).

The record this handoff cites (tracked): `agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
(**DI**); `MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`
(**CR**); `paper_5/notes/electron_phonon_closure_worklog_20260803.md`, the
2026-09-20 blocks, lines 8200 to 8447 (**WL**).

---

## 1. Why this thread exists

On the divergent initialization the cheap repair holds the electronic one-body
density matrix positive semidefinite over `t = 0` to `60` while holding
neither Gram cone, and the record says plainly that this is not by
construction: "the electronic one-body density matrix is *not* one of the
sets the Dykstra repair projects onto ... so electronic positivity over
`0 < t <= 60` is a measured outcome of this configuration and not a
construction of it" (DI section 11, lines 605 to 615; WL lines 8363 to
8366). The obvious experiment is to make it a construction and see whether
anything moves.

---

## 2. Anchors from the record

**The sets the alternation projects onto** (CR section 2(c) and (d); code
`AffineGramCoordinates.repair` and `_dykstra`,
`paper_5/src/paper5/stability/cheap_projection_route.py:1166` and `:1315`):
the affine Gram set (the image of the propagated coordinates), the retained
joint Gram cone (8 rows), the scaled positive extension cone (36 rows at 27
entrance keys), and, with an energy reference, the Eq. (22) hyperplane, met
together with the affine set in closed form. `target = "both"` carries both
cones; floors `eigenvalue_floor`, `retained_eigenvalue_floor` and
`repair_tolerance` are all `0` in the task-11 configuration.

| fact | number, with definition and horizon | source |
|---|---|---|
| electronic one-body density matrix, cheap repair from the Hartree--Fock zero-correlation preparation with the `exact_moments` lift, task-11 configuration | `-2.220e-16` at `t = 0` (a determinant, exactly on the cone); minimum `+1.447e-6` over `0 < t <= 20`, attained at `t = 0.0025`; largest `0.2150`; the same minimum `+1.4466e-6` over `0 < t <= 60` | DI section 4.1, lines 203 to 209; section 9, lines 458 to 467 |
| the same chain's other rows, `t = 0` to `20` | all-coordinate scalar RMS `1.0803e-2` / `3.7209e-2` / `1.5368e-1` at `t = 4` / `8` / `20`; Eq. (22) energy RMS against the reference `1.1888e-3` at `t = 20`; minimum unshifted retained joint Gram `-5.561e-4`; minimum scaled positive extension `-6.352e-5`; minimum boson-moment eigenvalue `-2.163e-3`; maximum `\|x_i\|` `1.9431`; 500 lift re-selections, 44 solver refusals; autonomous wall `4,296.9 s` in five segments; peak `654 MiB` | DI sections 4.1 and 4.2 |
| the retained joint Gram under the same chain | leaves `-1e-8` at `t = 0.0025` and deepens to `-4.370e-5` by `t = 8` and `-5.561e-4` by `t = 12` | DI section 6, item 3 |
| unrepaired archive closure from the same preparation | electronic one-body density matrix first below `-1e-8` at `t = 27.9633944`; `1e4` magnitude event at `54.4612958` | DI section 2.2 |
| the exact correlated task-11 chain, `t = 0` to `4` | all-coordinate scalar RMS `9.290e-3` at `t = 4`, band of four members `9.29e-3` to `9.95e-3` (7.1 percent); Eq. (22) internal-energy RMS `4.033e-3` at `t = 4`; minimum unshifted retained joint Gram `-4.460e-4` (at `t = 11.04`) and scaled positive extension `-3.936e-5` (at `9.4975`) over `t <= 20`; wall `5,183.6 s` to `t = 20` | CR sections 3.1 and 3.2 |
| that chain's electronic one-body density matrix | **not recorded anywhere** (its `summary.json` has no electronic field); increment 1 measures it from the stored states | --- |
| the energy-neutral joint electronic--bosonic matrix barrier, which does hold the electronic cone by construction | electronic eigenvalue `0` at `t = 0`, `+7.525e-14` afterwards through `t = 39`; all-coordinate scalar RMS `3.4704e-1` at `t = 20` against the cheap repair's `1.5368e-1`; its retained joint Gram, not in its constraint set, reaches `-0.36128` | DI section 5 and section 10 |

---

## 3. A structural fact the record does not state (derived from the source; increment 1 verifies it)

`uncentered_joint_moment_matrix` (`adaptive_positive_moment.py:458`) orders
its rows `(1, b_0, b_1, b_0^dagger, b_1^dagger, sigma_x, sigma_y, sigma_z)`;
the means row carries the Bloch coordinates `values[:3]` and the electronic
block is `Tr(rho sigma_a sigma_b)`. So the principal 4-by-4 submatrix on rows
`0, 5, 6, 7` is the Gram matrix of `(1, sigma_x, sigma_y, sigma_z)` in the
state `rho`. Because `{1, sigma}` spans all 2-by-2 matrices, that block is
positive semidefinite exactly when `rho` is, and its eigenvalues are twice
those of `rho`, each twice. Two consequences, both to be measured, not
assumed:

1. **The electronic cone is implied by the retained joint Gram cone.** A
   state whose retained joint Gram is positive semidefinite has a positive
   semidefinite electronic one-body density matrix. The record's repaired
   chain sits at retained joint Gram `-5.561e-4` with electronic minimum
   `+1.447e-6`: the retained violation lies in other directions.
2. **Placement decides whether the option can act at all.** Inside one
   Dykstra iteration the cone step clips the whole retained block, so its
   output already has a positive semidefinite principal block; an electronic
   projection placed immediately after that step is inert by construction.
   It can act only as its own set with its own Dykstra correction memory,
   met at a point the retained cone step has not just produced (between the
   affine step, with the hyperplane, and the next cone step), or as a final
   step on the returned point. Implement the first, report the second only
   if the first never acts.

If the scaled retained matrix the repair uses
(`AffineGramCoordinates.scaled_retained_matrix`, line 840) is the unshifted
Gram under a diagonal congruence, the same statements hold in the scaled
metric; increment 1 checks that too.

---

## 4. Scope specific to this thread

* In scope: the identity and scan audit; the stage-input replay on the
  stored task-11 states; the named option in `cheap_projection_route.py`
  with its tests; rollouts from the Hartree--Fock zero-correlation
  preparation (`exact_moments` lift) to `t = 20` and from the exact
  correlated preparation (task-11 configuration) to `t = 4`, with and
  without the option.
* Out of scope: changing any default, floor, Dykstra count, the energy
  reference, the re-selection rule or the lift; the carried-witness lane's
  repair (`CarriedWitnessAffineGram`, line 2458), which should refuse the
  new option the way it refuses the declared metric and the band (see the
  existing tests named in increment 3); horizons past `t = 20`.
* Machine headroom: increments 1 to 2 are light (under `300 MB`); each
  rollout is about `650 MiB`; run up to three concurrently after
  `memory_pressure -Q`.

---

## 5. Increments

Proceed through all of them autonomously (C15).

### Increment 0 — worktree, input lock, readable inputs, driver fixes and both parity gates (under 20 minutes)

**Goal.** Your worktree exists at `b9f99271`; every input matches its
SHA-256; the drivers are copied byte for byte; every input opens; the Eq.
(22) energy reference of both stored chains is read per segment; both copied
segment drivers are fixed as C20 decides; and C20's parity gate passes for
both: the Hartree--Fock zero-correlation preparation with the
`exact_moments` lift through `run_divergence_segment.py`, and the exact
correlated preparation in the task-11 configuration through
`run_energy_segment.py`.

**Bound.** Each command below finishes in under 20 minutes; the whole
increment takes about 12 minutes (six short rollouts, about 8).

```bash
git -C /Users/lordskyler/Holstein_test worktree add -b "$BRANCH" "$WT" b9f99271fc3ddf3db604d3acb64617a291bec7e1
git -C "$WT" rev-parse HEAD
mkdir -p "$OUT" "$DATA/drivers"
shasum -a 256 -c "$LOCK"                                    # every line must print OK
cp "$REC"/*.py "$REC"/*.sh "$OUT"/
cp "$T11DIR"/run_energy_segment.py "$T11DIR"/score_energy_chain.py "$T11DIR"/run_chain.py "$T11DIR"/probe_cone_correction_share_20260918.py "$OUT"/
(cd "$REC" && shasum -a 256 *.py *.sh; cd "$T11DIR" && shasum -a 256 run_energy_segment.py score_energy_chain.py run_chain.py probe_cone_correction_share_20260918.py) > "$OUT/copies_source.sha256"
(cd "$OUT" && shasum -a 256 $(cd "$REC" && echo *.py *.sh) run_energy_segment.py score_energy_chain.py run_chain.py probe_cone_correction_share_20260918.py) > "$OUT/copies_copied.sha256"
diff "$OUT/copies_source.sha256" "$OUT/copies_copied.sha256" && echo "copies byte-identical"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python - <<'PY'
import json, os, numpy as np
import paper5.stability.cheap_projection_route as m; print('import', m.__file__)
chain, t11, t11dir = os.environ['CHAIN'], os.environ['T11'], os.environ['T11DIR']
for label, prefix, count, key in (('hf', chain, 5, 'archive_coordinates'), ('t11', t11, 6, 'states')):
    previous = None
    for k in range(1, count + 1):
        base = f'{prefix}_seg{k}'
        d = np.load(base + '/trajectory.npz'); r = np.load(base + '/resume.npz')
        ref = np.asarray(d['internal_energy_reference'])
        print(label, k, d[key].shape, 'start', float(d['times'][0]), 'end', float(d['times'][-1]), 'resume', float(r['time']),
              'reference at start', float(ref[0]), 'previous reference at resume', previous,
              'difference', None if previous is None else abs(float(ref[0]) - previous))
        previous = float(ref[int(np.argmin(np.abs(d['times'] - float(r['time']))))])
e = np.load(t11dir + '/exact_t20.npz'); print('exact_t20', {k: e[k].shape for k in e.files})
print('task11 score keys', list(json.load(open(t11dir + '/task11_base_score_t20.json')).keys()))
PY
```

Then fix `$OUT/run_divergence_segment.py` and `$OUT/run_energy_segment.py`
exactly as C20 items 1 to 4 say, write the parity checker, and run both
gates:

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
HF=(--preparation hartree_fock --lift exact_moments --wall-budget-seconds 900 --record-reselections)
T11ARGS=(--dictionary entrance15_plus_two_spin_boson --lift-reselection-interval 16 --energy-reference-mode drive_power \
  --initialization-ladder --reselection-acceptance-margin 1e-3 --reselection-candidates measured \
  --reselection-anchor-weight 0 --dykstra-iterations 120 --wall-budget-seconds 900 --record-reselections)
gate () {   # usage: gate NAME DRIVER ARGS...
  local name=$1 drv=$2; shift 2; local P="$OUT/parity_${name}"
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$drv" "$@" --output-directory "${P}_unbroken" --final-time 0.2) || return 1
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$drv" "$@" --output-directory "${P}_segA" --final-time 0.1) || return 1
  (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$drv" "$@" --output-directory "${P}_segB" --resume-from "${P}_segA" --final-time 0.2) || return 1
  /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/parity_check.py" "${P}_unbroken" "${P}_segA" "${P}_segB"
}
gate hf_default "$OUT/run_divergence_segment.py" "${HF[@]}"
gate t11_default "$OUT/run_energy_segment.py" "${T11ARGS[@]}"
cp "$OUT/run_divergence_segment.py" "$OUT/run_energy_segment.py" "$OUT/parity_check.py" "$DATA/drivers/"
```

Write `$DATA/input_manifest.json` (expected and actual SHA-256 of every
locked input, every copy, and the edited drivers) and put both stored chains'
per-segment reference tables into the report-back.

**Acceptance.** HEAD is `b9f99271fc3ddf3db604d3acb64617a291bec7e1`; every
lock line prints `OK`; the copies are byte-identical; the import is under
`$WT`; the five Hartree--Fock segments end at `4.76`, `9.60`, `13.82`,
`18.60`, `20.00`; the six task-11 segments have `states` of width 56,
segment 1 spanning `[0, 3.955]` and resuming at `3.92`; `exact_t20.npz` opens
with 8,001 samples; in both stored chains the reference at each segment start
differs from the previous segment's reference at its resume time by at most
`1e-12`; both parity gates pass as C20 states.
**Stop.** Any lock line not `OK`, any missing input or different shape, an
import outside `$WT`, a stored reference discontinuity above `1e-12`, or a
parity failure (C20; a re-selection decision flip at the feasibility edge is
reported with its two decision records).
Commit `$DATA/input_manifest.json` and the three driver copies under
`$DATA/drivers/`.

### Increment 1 — the identity, the scan, and the task-11 electronic minimum

**Bound.** The suite takes about `250` to `265 s`; the identity over about
16,000 stored samples and the scan take a few minutes each.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q   # the baseline (C16), before any edit
```

**Goal.** Measure section 3's identity on every stored sample of both chains;
measure the task-11 chain's electronic one-body density matrix minimum over
`t = 0` to `20`; list every stored rollout in the source worktree whose
electronic one-body density matrix ever went below `-1e-8`.

Write `paper_5/src/paper5/stability/electronic_cone_audit.py` with two
subcommands:

* `identity --segments DIR [DIR ...] --output JSON`: for every stored sample,
  the raw moment coordinates (from `states[:, :29]` when present, otherwise
  from `archive_coordinates` through `closed_scalar_to_matrix_state` and
  `matrix_state_to_raw_moment_coordinates`, subtracting the shared
  correlation trace exactly as `_retained_gram_minimum` in
  `run_unrepaired.py` does); the minimum eigenvalue of the principal block
  on rows `0, 5, 6, 7` of `uncentered_joint_moment_matrix`; twice
  `closed_electron_eigenvalues(...)[0]`; their largest absolute difference;
  the electronic minimum over the horizon and its time;
* `scan --root DIR --output JSON`: every `trajectory.npz` under the root
  that has `archive_coordinates`; its electronic minimum and time, and
  whether it crosses `-1e-8`.

Test `paper_5/tests/test_electronic_cone_audit.py`: on three constructed
2-by-2 density matrices (pure, mixed, and one with eigenvalue `-0.1`)
embedded in otherwise arbitrary raw coordinates, the block's minimum
eigenvalue equals twice `rho`'s to `1e-13`.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q tests/test_electronic_cone_audit.py
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.electronic_cone_audit identity \
  --segments "$CHAIN"_seg{1..5} --output "$OUT/identity_hf_t20.json"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.electronic_cone_audit identity \
  --segments "$T11"_seg{1..6} --output "$OUT/identity_task11_t20.json"
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.electronic_cone_audit scan \
  --root /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs --output "$OUT/electronic_scan.json"
cp "$OUT"/identity_hf_t20.json "$OUT"/identity_task11_t20.json "$OUT"/electronic_scan.json "$DATA"/
```

Also check, on ten stored task-11 states, whether `scaled_retained_matrix`
equals `D G D` for a diagonal `D` (build the route the way
`run_energy_segment.py` does, or read `AffineGramCoordinates` to find the
scaling), and record the answer.

**Acceptance.** The test passes. On the Hartree--Fock chain the electronic
minimum over `0 < t <= 20` reproduces `+1.447e-6` at `t = 0.0025` (to four
digits). The identity's largest difference over all samples of both chains
is reported; at or below `1e-12` it is recorded as holding. The task-11
electronic minimum over `t = 0` to `20` and its time are reported (a number
the record does not have). The scan lists every stored rollout with an
electronic crossing below `-1e-8`, separating repaired chains from
unrepaired-closure runs (the latter are expected to cross).
**Stop.** The Hartree--Fock chain's electronic minimum not reproducing
`+1.447e-6`: the stored data or the conversion differs; report and stop. An
identity failure larger than `1e-10` is not a stop: report the samples, and
note that the electronic cone is then not implied by the retained cone, which
strengthens the case for the option.
Commit the module, its test and the three JSON files.

### Increment 2 — the stage inputs the repair actually sees

**Goal.** How often, and how deeply, the electronic one-body density matrix
is outside its cone at the SSPRK(3,3) stage inputs, before the repair, and
after the declared repair, on the stored task-11 states.

**Bound.** The record's share mode took at most `243 s` for 800 samples;
the `(0, 4]` window is 1,600 samples, under 10 minutes. If a run approaches
20 minutes, split the window in two invocations with distinct outputs.

Copy-extend `$OUT/probe_cone_correction_share_20260918.py` (set `_LOCAL_RUNS`
to the source worktree's `output/local_runs`, as in section 0) with
`--mode electronic`: for each stored sample in the window, replay the three
stage repairs exactly as `--mode share` does and record, at each stage input,
the electronic one-body density matrix minimum eigenvalue before the declared
repair and after it, and the retained joint Gram minimum before and after.
Copy it to `$DATA/drivers/`.

```bash
# first, the unmodified share mode on its own reference window, as a check of the copy
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/probe_cone_correction_share_20260918.py" \
  --chain paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg --mode share \
  --window 6 8 --stride 1 --output "$OUT/share_check_task11_w68.json"
for w in "0 4" "6 8"; do
  cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/probe_cone_correction_share_20260918.py" \
    --chain paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg --mode electronic \
    --window $(echo $w) --stride 1 --output "$OUT/stage_inputs_task11_w$(echo $w | tr ' ' '_').json"
done
cp "$OUT"/stage_inputs_task11_w*.json "$DATA"/
```

The record's own output of the share check is
`$T11DIR/cone_correction_share_base_w68_20260918.json`; compare the stage
count and the aggregate shares against it.

**Acceptance.** Both windows complete (`(0, 4]` is 1,600 stored samples,
`(6, 8]` 800; three stage inputs each). Reported per window: the count of
stage inputs whose electronic minimum is below `0` and below `-1e-8` before
the repair, the same after it, and the deepest value. Reading rule recorded
in `$REPORTFILE`: if no stage input in either window is below `-1e-8` before
the repair, the option cannot act on the task-11 chain in those windows, and
increment 5's comparison is expected to be bitwise equal to the default.
**Stop.** The share check on `(6, 8]` does not reproduce the record's
2,400 stage repairs (CR section 4, item 8) or differs from
`cone_correction_share_base_w68_20260918.json` in its aggregate shares by
more than `1e-6`: the copy or the source differs from the record's; stop and
report.
Commit the JSON files and the driver copy.

### Increment 3 — the named option and its tests

**Goal.** `CheapProjectionSettings.electronic_density_floor: float | None = None`.
`None` (the default) is the behaviour at `b9f99271`, bit for bit. A number
puts the electronic one-body density matrix into the Dykstra alternation of
`AffineGramCoordinates._dykstra` as its own set, with its own correction
memory, met between the affine step (with the Eq. (22) hyperplane when
active) and the next cone step (section 3, consequence 2): the principal
block on rows `0, 5, 6, 7` of the **scaled** retained matrix is clipped to
the floor scaled the same way, and its projection returns its input
unchanged (no eigendecomposition round trip) whenever the block's minimum
eigenvalue is already at or above the floor, so that an option that never
acts leaves every number bitwise unchanged. `repair` includes the block in
its `cones_met` test and reports, per repair, the electronic minimum before
and after and whether the electronic set acted. The option is valid only
with `target` in `{"retained", "both"}`; the carried-witness lane refuses
it.

Run the impact analysis first (C12) for `CheapProjectionSettings`,
`AffineGramCoordinates.repair`, `AffineGramCoordinates._dykstra`,
`AffineGramCoordinates._cone_projection` and
`integrate_cheap_projection_ssprk3`, and list the `rg` callers in the commit
message.

Tests, in a new test file `paper_5/tests/test_electronic_density_floor.py`
following the pattern of
`test_raw_coordinate_motion_scales_are_off_and_change_no_number` and
`test_energy_reference_band_is_off_and_changes_no_number`
(`paper_5/tests/test_cheap_projection_route.py:1342` and `:1506`):

* off by default: `CheapProjectionSettings().electronic_density_floor is None`,
  and a repair and a 20-step rollout from the test module's prepared fixture
  are bitwise equal with the option absent and explicitly `None`;
* never active means nothing changes: on a state whose electronic block is
  interior, `electronic_density_floor=0.0` gives arrays bitwise equal to the
  default;
* it acts: on a constructed state whose electronic one-body density matrix
  has eigenvalue `-1e-3` and whose retained block is otherwise interior, the
  repair with `electronic_density_floor=0.0` returns an electronic minimum at
  or above `-1e-10` when the alternation reports that all sets were met;
* rejections: `target="extension"` with a floor raises `ValueError`; the
  carried-witness lane refuses it, as
  `test_the_carried_witness_lane_refuses_a_declared_band` does for the band.

Add the driver flag `--electronic-density-floor` (default absent) to your
copies `$OUT/run_divergence_segment.py` and `$OUT/run_energy_segment.py`,
recording it in `summary.json` and the per-run activation count; copy both
to `$DATA/drivers/`.

**Bound.** The two test files take under 2 minutes; the suite about
`250` to `265 s`.

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q tests/test_electronic_density_floor.py tests/test_cheap_projection_route.py
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q
```

**Acceptance.** The new tests pass; `test_cheap_projection_route.py` passes
unchanged; the full suite equals the increment-1 baseline plus the new tests
with no new failure.
**Stop.** Any existing test changes outcome: the default path is not
bitwise; fix before continuing, and if the fix needs a change to an existing
test, stop and report.
Commit the module change, the tests and the driver copies.

### Increment 4 — reproduction and a short smoke, both preparations

**Goal.** Your drivers, with the option absent, reproduce the stored chains
over `t = 0` to `0.05`; with the option at `0.0` the same window is reported.

**Bound.** Four short rollouts of `0.05` time units, each under 2 minutes.

```bash
cd "$WT" && for flag in none 0.0; do
  extra=(); [[ "$flag" != none ]] && extra=(--electronic-density-floor "$flag")
  PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/run_divergence_segment.py" \
    --preparation hartree_fock --lift exact_moments --output-directory "$OUT/smoke_hf_floor_${flag}" \
    --final-time 0.05 --wall-budget-seconds 900 --record-reselections "${extra[@]}"
  PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/run_energy_segment.py" \
    --dictionary entrance15_plus_two_spin_boson --output-directory "$OUT/smoke_t11_floor_${flag}" \
    --final-time 0.05 --lift-reselection-interval 16 --energy-reference-mode drive_power --initialization-ladder \
    --wall-budget-seconds 900 --record-reselections \
    --reselection-acceptance-margin 1e-3 --reselection-candidates measured --reselection-anchor-weight 0 \
    --dykstra-iterations 120 "${extra[@]}"
done
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python -c "
import numpy as np
for name, stored in (('hf', '$CHAIN' + '_seg1'), ('t11', '$T11' + '_seg1')):
    s = np.load(stored + '/trajectory.npz')['archive_coordinates'][:21]
    for flag in ('none', '0.0'):
        f = np.load('$OUT' + f'/smoke_{name}_floor_{flag}/trajectory.npz')['archive_coordinates'][:21]
        print(name, flag, 'max abs difference to the stored chain over t = 0 to 0.05:', float(np.max(np.abs(f - s))))
"
```

**Acceptance.** With the option absent, both smokes match their stored
chains over the 21 samples to `1e-12` or better (bitwise is expected for a
fresh segment from `t = 0`). With the option at `0.0`, the difference and the
activation count are reported.
**Stop.** An option-absent difference above `1e-10`: the copied driver or
the source differs from the record's; report it and stop.
Commit `$DATA/smoke.json` (the four differences and activation counts).

### Increment 5 — the divergent initialization over `t = 0` to `20`, option on

**Goal.** The record's Hartree--Fock `exact_moments` rows, with the
electronic one-body density matrix as a projected set.

**Step 1, the parity gate with the option on (C20).** Bound: three short
rollouts, each under 5 minutes; `gate` and `HF` are defined in increment 0
(redefine them in a new shell).

```bash
gate hf_floor0 "$OUT/run_divergence_segment.py" "${HF[@]}" --electronic-density-floor 0.0
```

**Step 2, the chain.** One segment of at most `900 s` of propagation per
process, each resumed from the previous segment's checkpoint with its
carried reference printed (C20).

```bash
run_floor_chain () {   # usage: run_floor_chain NAME FINAL DRIVER_ARGS...
  local name=$1 final=$2; shift 2
  local prev="" k out
  for k in {1..10}; do
    out="$OUT/${name}_seg$k"
    if [[ -f "$out/summary.json" ]]; then prev=$out; continue; fi
    if [[ -e "$out" ]]; then echo "refusing to reuse $out (C15); rename and retry"; return 1; fi
    local resume=()
    [[ -n "$prev" ]] && resume=(--resume-from "$prev")
    (cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$@" \
      --output-directory "$out" --final-time "$final" --wall-budget-seconds 900 --record-reselections "${resume[@]}") || return 1
    prev=$out
    /Users/lordskyler/Holstein_test/.venv/bin/python -c "import json,sys; s=json.load(open('$out/summary.json')); r=s['rollout']; t=r.get('final_time_reached', r.get('horizon_reached')); sys.exit(0 if t >= $final - 1e-9 else 1)" && break
  done
}
memory_pressure -Q
run_floor_chain hf_floor0 20.0 "$OUT/run_divergence_segment.py" --preparation hartree_fock --lift exact_moments \
  --electronic-density-floor 0.0 > "$OUT/hf_floor0.log" 2>&1 &
wait
grep -h "carried Eq. (22) energy reference" "$OUT/hf_floor0.log"      # one line per resumed segment, into the report
ordered=$(cd "$OUT" && ls -d hf_floor0_seg* | awk -F seg '{print $NF, $0}' | sort -n | cut -d' ' -f2 | sed "s|^|$OUT/|")
cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/score_chain.py" \
  $(echo $ordered) --horizons 1 2 4 8 12 16 20 --output "$OUT/hf_floor0_score_t20.json"
```

(Increments 5 and 6 can run concurrently; start increment 6's two chains
before the `wait` if `memory_pressure -Q` allows.)

**Acceptance.** The parity gate passes (C20 criteria); the chain reaches
`t = 20`, every resumed segment printing a carried reference equal to its
predecessor's at the resume time to `1e-14`. Reported beside the record's row
(section 2): all-coordinate scalar RMS at `t = 4`, `8`, `20`; Eq. (22) energy
RMS against the reference at `t = 20`; electronic minimum over
`0 < t <= 20` and its time; minimum unshifted retained joint Gram and scaled
positive extension; minimum boson-moment eigenvalue; maximum `|x_i|`; lift
re-selections and solver refusals; the number of stage repairs at which the
electronic set acted; autonomous wall and its ratio to `4,296.9 s`; peak
memory. The answer for this preparation, in numbers: "nothing changes" if
the electronic set acted at zero stage repairs after `t = 0` (then the
trajectory must be bitwise equal to a default run over any window you
rerun); otherwise the row differences, read against the 7.1 percent band at
`t = 4` (the only measured band on record, CR section 3.1) and the chaining
perturbation (CR section 7).
**Stop.** The parity gate fails: do not start the chain. The chain's
electronic minimum falling below `-1e-8` with the option on: a defect of the
option; stop and report the stage record.
Commit `$DATA/hf_floor0_t20.json`.

### Increment 6 — the exact correlated preparation over `t = 0` to `4`, with and without

**Goal.** The task-11 `t = 4` row with the option on, beside a fresh default
rerun (the stored chain is one realization and chaining is not bitwise, so
the default is rerun under the same conditions).

**Step 1, the parity gate with the option on (C20).** Bound: three short
rollouts, under 5 minutes each (the default setting's gate passed in
increment 0).

```bash
gate t11_floor0 "$OUT/run_energy_segment.py" "${T11ARGS[@]}" --electronic-density-floor 0.0
```

**Step 2, the two chains**, one segment of at most `900 s` per process:

```bash
T11CHAIN=(--dictionary entrance15_plus_two_spin_boson --lift-reselection-interval 16 --energy-reference-mode drive_power \
  --initialization-ladder --reselection-acceptance-margin 1e-3 --reselection-candidates measured \
  --reselection-anchor-weight 0 --dykstra-iterations 120)
run_floor_chain t11_default 4.0 "$OUT/run_energy_segment.py" "${T11CHAIN[@]}" > "$OUT/t11_default.log" 2>&1 &
run_floor_chain t11_floor0 4.0 "$OUT/run_energy_segment.py" "${T11CHAIN[@]}" --electronic-density-floor 0.0 > "$OUT/t11_floor0.log" 2>&1 &
wait
grep -h "carried Eq. (22) energy reference" "$OUT"/t11_*.log      # one line per resumed segment, into the report
for name in t11_default t11_floor0; do
  ordered=$(cd "$OUT" && ls -d ${name}_seg* | awk -F seg '{print $NF, $0}' | sort -n | cut -d' ' -f2 | sed "s|^|$OUT/|")
  cd "$WT" && PYTHONPATH=paper_5/src:$OUT /Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/score_energy_chain.py" \
    $(echo $ordered) --horizons 1 2 4 --exact-cache "$T11DIR/exact_t8.npz" --output "$OUT/${name}_score_t4.json"
done
```

**Acceptance.** The gate passes; both chains reach `t = 4` (the stored
segment 1 reached `3.955` in `1,000 s`, so expect two or three `900 s`
segments each), every resumed segment printing its carried reference. The default rerun scores
inside the band `9.29e-3` to `9.95e-3` at `t = 4`. Reported for both: the
`t = 4` all-coordinate scalar RMS, Eq. (22) internal-energy RMS (record
`4.033e-3`), electronic minimum over `0 < t <= 4`, retained joint Gram and
scaled positive extension minima, activation count, wall. The answer for
this preparation: the option's `t = 4` score against the default rerun's,
read against the band.
**Stop.** The default rerun outside `[9.0e-3, 1.02e-2]` at `t = 4`: the
harness differs from the record's task-11 configuration; stop and report
before reading the option's row.
Commit `$DATA/t11_t4.json`.

### Increment 7 — conditional: a stored chain where the electronic cone is lost

Only if increment 1's scan found a **repaired** chain (any cheap-repair or
projective rollout, not an unrepaired closure) whose electronic one-body
density matrix crosses `-1e-8`: rerun that chain's configuration from the
segment that contains the first crossing, with and without the option, over
a window from that segment's start to one time unit past the crossing, using
the driver and flags recorded in that chain's `summary.json`. Report both
rows. If the scan found none, record "no repaired chain on record loses the
electronic cone" with the count of repaired chains scanned, and skip.

**Bound.** Each rerun is one segment of at most `900 s`, preceded by its
parity gate at that chain's setting (C20).

**Acceptance.** Either the skip with its count, or both rows with the
activation count. **Stop.** A chain whose driver cannot be identified from
its `summary.json`: skip it and report.

### Increment 8 — the report-back file and the final commit

**Bound.** The suite takes about `250` to `265 s`.

Write `$REPORTFILE` per C17, then:

```bash
cd "$WT/paper_5" && PYTHONPATH=src /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q
git -C "$WT" add agent_guidance/paper-v/handoffs/REPORT-electronic-cone-projection-codex-20261010.md agent_guidance/paper-v/handoffs/electronic-cone-projection-codex-20261010-data
git -C "$WT" commit -m "paper-v (Codex): the electronic one-body density matrix as a projected set — <the answer with its numbers>"
```

**Acceptance.** The suite stands at the increment-1 baseline plus the new tests
with no new failure; nothing tracked is left modified.
**Stop.** A new test failure: fix it if it is yours, otherwise report it.

---

## 6. Evidence standards for this thread

* The identity of section 3 is a derivation until increment 1 measures it;
  never cite it as the record's.
* "Nothing changes" is a bitwise statement backed by a zero activation count,
  or it is a difference with its size; never a judgement.
* The task-11 comparison is against a fresh default rerun as well as the
  stored row, because chaining is not bitwise (CR section 7).
* The option's cost is the wall ratio from `summary.json` against the default
  under the same machine load; note concurrent jobs.

## 7. Traps

* Your worktree has none of the drivers or chains (C5). The task-11 driver
  `run_energy_segment.py` was written against a source snapshot
  (`head_src_e3374f6f`); with `PYTHONPATH=paper_5/src:$OUT` it imports your
  worktree's modules, which is intended. Its defaults are not the task-11
  configuration (`--dykstra-iterations 30`, `--energy-reference-mode none`);
  pass every flag listed in increment 4.
* `probe_cone_correction_share_20260918.py` resolves `--chain` against its
  own grandparent directory; in a copy that is your worktree, which has no
  chains (section 0).
* The Hartree--Fock chain's `trajectory.npz` has no 56-coordinate states, so
  the stage-input replay of increment 2 runs only on the task-11 chain.
* An electronic projection placed right after the retained cone step is
  inert by construction (section 3); a zero activation count from that
  placement says nothing.
* Duplicating the electronic entries as a new block of the feature vector
  would change the metric of the affine projection and of the least-squares
  map back, and therefore every number, even where the electronic set never
  acts; the clip on the principal block of the existing retained block
  avoids that.
* The two floors the task-11 configuration sets are `0`; the HF preparation
  starts exactly on the electronic cone (`-2.220e-16` at `t = 0`), so a floor
  of `0` acts at `t = 0` at roundoff level: report activations at `t = 0`
  separately from later ones.

## 8. Report back

`$REPORTFILE` per C17, committed on your branch. Its first lines state, one
sentence each: whether the identity holds (with its largest difference);
how many stage inputs on the task-11 chain fall outside the electronic cone;
how many stage repairs the option acted at on each preparation; and the
resulting change in the `t = 20` (divergent) and `t = 4` (exact correlated)
all-coordinate scalar RMS, with the wall ratio.


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
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/agent_guidance/paper-v/handoffs/codex-paper-v-electronic-cone-projection-20261010-inputs.sha256`
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
40637207645cf4079b0510e08c7e316ee48a6d8c7e1b96c734e54cb1f76e9ed0  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1/trajectory.npz
8598f745ca802f77c6c9918d12a86ce2c075d19d25a2d91c2c9d11d6d3e1ef32  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg2/trajectory.npz
dc4732197462f65b129c010fa1dfbd4c5f166e718a7a62b2278857f5a0fc0910  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg3/trajectory.npz
050a76c3cf4dcac6c2dc0a406167cbb48f758da1fad47865c494d8b78c79a110  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg4/trajectory.npz
b9a1d30c6e49e46a24115f4d469f3b92bd3c9e14dee1508ff21bcadad7c1f2c2  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg5/trajectory.npz
18be81b3e0f51ac4332d70e22272a1872984d4fb72c1becfdf32072836ef8252  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg6/trajectory.npz
5cbe144d1d6a4d5cb831fa7ed87be92611066ae6f311d25c6f74386ee26b2da0  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1/summary.json
b8dfbcce39564d3fe9478a6db6b7c08720e5ca08533f7d7c643c9017152df3ee  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg2/summary.json
6f5685a99e9e5892d41b9eaddb02e84cb52d589b2c4e9cd6ad36accce1f35a27  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg3/summary.json
fbcad4453756ea808c644b503da0c629190af06dc5046ff1a22716a027939d50  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg4/summary.json
2a21102d5f16551dc1df7c4827f25175f37e4997e308da4dcbd41792222061dc  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg5/summary.json
e560c2a5b0172b8ea062de11375a6322f49a072db5b1d25284bd08c944cf0a32  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg6/summary.json
10bf140ae3ab6911629c8d426552dc09927fdcf625313b2a76425f05a4892587  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1/resume.npz
09244046a754103f4344e0ee868d0d2c873cc407e0e9dc270bf0c3e39d0175ca  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg2/resume.npz
6354821af633f69200992dc4e90dd050b62e53cbcbf0a022357bc182d370a044  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg3/resume.npz
24fbcd9f4ed4287c9f468b07ca170bce80d36ae48a8de10ff485761531eb0287  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg4/resume.npz
5d35f583a8c6aa15a7ba97b2acd35cec9136a8a0432c40be1830597abe1a5df0  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg5/resume.npz
b5f48232e1630b418db07f12c03a61cf52833ad6c117ee6221dc395678ffc57b  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg6/resume.npz
ffbfde2008e2753dd7424c2361dbac0ee32de438a46ed73971ebf9f8bb960dd9  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg1/reselections.json
52ccb3acf55f0d2d699f33d2d0678c473ab58bab81e3bd59b9a795e277746bfb  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg2/reselections.json
b41523afb0cd635140556fc4f7991b0926ba22517f9ff5d60b7242816cfc17dd  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg3/reselections.json
3a653d37bfd27619eb2b401aeca6248837b1835da96169fda43d7a5b1b569be7  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg4/reselections.json
c85c2f56c860257729e49a462999ca91b7533e152ffd17a986439256c159dc49  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg5/reselections.json
2be2e266fa51701e79438faf40244578c35ba4e138b84185725762ff569064f8  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg6/reselections.json
bed18372b2c2201007625288a30bd45d2da766a5bd3efb1a669000e11fa30b30  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/run_energy_segment.py
5f2bbb5bbf29236d575a3749076eab1d87c18fef2e9e49a2f076d0626e7db2fb  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/score_energy_chain.py
938dc339b261622f76a6acb5a0816a3357124beca21f47445071cb12b3ed3af5  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/run_chain.py
890dbe83d8f5d18b3f3eec49f5289989aed055f41838df9d0fb3acbbe471d91a  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/probe_cone_correction_share_20260918.py
a66c26dd46918da03d3e1e0fe75edd0965bd7e08d473683e7f34218d66295ad4  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/exact_t20.npz
70521177d52f2a2a602dd3f2fe3bfcc9fa61cd7f36852ec762955aaf529cf422  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/exact_t8.npz
67b5f009c70aa3deff67527b6490e1054b8d0b4c88b331d8074fa18209bf46c7  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/task11_base_score_t20.json
cb16b74add3d3135fa84504ddd7316119e3da95957e8fa48dfda8a0272ac8f30  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/task13_base_score_t20.json
f2aa537066ac5e25d0353138c9f389461ee17f88e26498821d6480839993ec04  /Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/paper_v_projective_dictionary_energy_20260915e/cone_correction_share_base_w68_20260918.json
~~~
