# Index: four Codex threads on the Paper V classical repair (2026-10-10)

**What this is.** The index of four standalone, exploratory Codex handoffs on
the Paper V classical repair of the archive moment closure, written to run
in parallel ("throw things at the wall"), each in its own worktree on its own
branch from `claude/paper-v-routes-20260914` at `b9f99271`. Each handoff is
executable cold, follows `agent_guidance/shared/agent-handoff-contract.md`,
and carries the common rules below verbatim as its section C, plus its own
input lock as its appendix L. This file is in `prompt-exports/` (gitignored)
with an identical tracked copy at
`agent_guidance/paper-v/handoffs/codex-paper-v-INDEX-20261010.md`. Campaign
context only: `prompt-exports/campaign-plan-codex-claude-gpt-20261010.md`.
Readiness audit these handoffs answer:
`/Users/lordskyler/Holstein_test.worktrees/codex-paper-v-classical-readiness-20261010/agent_guidance/paper-v/handoffs/classical-campaign-readiness-20261010.md`.

## The four threads

| thread | one line | handoff (`prompt-exports/` and tracked copy in `agent_guidance/paper-v/handoffs/`) | input lock (tracked) |
|---|---|---|---|
| 1. repair early, then bare | Apply the cheap repair over `[0, T_r]` only, hand the 31 closed scalar coordinates to the unrepaired archive closure, and map the `1e4` magnitude event and the electronic positivity loss against `T_r` (every stored sample of the record's `t = 0` to `60` chain is a free handover state), with what each second of repair wall buys and three cheaper repairs over `[0, 10]`. | `codex-paper-v-repair-early-then-bare-20261010.md` | `codex-paper-v-repair-early-then-bare-20261010-inputs.sha256` (85 files) |
| 2. regularized product-state initialization | Admix a declared small correlated component, or a declared uncorrelated mixed-state inflation, into the Hartree--Fock zero-correlation preparation so the declared completion certifies at `t = 0` without frontier moments from a state vector; measure the admixture's effect on the divergent-case repair over `t = 0` to `20` against the record's rows. | `codex-paper-v-regularized-product-state-20261010.md` | `codex-paper-v-regularized-product-state-20261010-inputs.sha256` (49 files) |
| 3. the electronic density cone as a projected set | Add the electronic one-body density matrix to the Dykstra alternation as a named, default-off set; run the divergent initialization over `t = 0` to `20` and the exact correlated preparation over `t = 0` to `4` with and without it; report whether anything changes and at what cost (and verify that this cone is implied by the retained joint Gram cone). The only thread that edits `cheap_projection_route.py`. | `codex-paper-v-electronic-cone-projection-20261010.md` | `codex-paper-v-electronic-cone-projection-20261010-inputs.sha256` (82 files) |
| 4. why the closure diverges from exact data | From the exact correlated preparation (and, beside it, the Hartree--Fock one), map the unrepaired archive closure's `1e4` event and electronic positivity loss over `lambda_ep`, drive amplitude and phonon cutoff `16` to `32`, at measured resolution, plus starts from the exact driven states of the correlated preparation; say whether the divergence is a strong-coupling phenomenon or generic. No chained repaired runs. | `codex-paper-v-divergence-parameter-map-20261010.md` | `codex-paper-v-divergence-parameter-map-20261010-inputs.sha256` (29 files) |

## First increment of each thread (under 20 minutes)

Each first increment proves every input readable and locked before anything
else runs; the three threads that chain repaired segments also fix their
copy of the segment driver and pass the continuous-versus-segmented parity
gate (C20). Run the handoff's section 0 `export` block first; the commands
are spelled out in full in each handoff's increment 0.

| thread | increment 0 opens with | and ends with |
|---|---|---|
| 1 | `git -C /Users/lordskyler/Holstein_test worktree add -b codex/paper-v-repair-early-then-bare-20261010 /Users/lordskyler/Holstein_test.worktrees/codex-paper-v-repair-early-then-bare-20261010 b9f99271fc3ddf3db604d3acb64617a291bec7e1 && shasum -a 256 -c "$LOCK"` | `/Users/lordskyler/Holstein_test/.venv/bin/python "$OUT/parity_check.py" "$OUT/parity_task11_unbroken" "$OUT/parity_task11_segA" "$OUT/parity_task11_segB"` (task-11 configuration, Hartree--Fock preparation, `exact_moments` lift) |
| 2 | `git -C /Users/lordskyler/Holstein_test worktree add -b codex/paper-v-regularized-product-state-20261010 /Users/lordskyler/Holstein_test.worktrees/codex-paper-v-regularized-product-state-20261010 b9f99271fc3ddf3db604d3acb64617a291bec7e1 && shasum -a 256 -c "$LOCK"` | the full retained joint Gram spectrum of the Hartree--Fock preparation, then the same parity gate as thread 1 |
| 3 | `git -C /Users/lordskyler/Holstein_test worktree add -b codex/paper-v-electronic-cone-projection-20261010 /Users/lordskyler/Holstein_test.worktrees/codex-paper-v-electronic-cone-projection-20261010 b9f99271fc3ddf3db604d3acb64617a291bec7e1 && shasum -a 256 -c "$LOCK"` | `gate hf_default ...` and `gate t11_default ...`: the parity gate for both fixed drivers |
| 4 | `git -C /Users/lordskyler/Holstein_test worktree add -b codex/paper-v-divergence-parameter-map-20261010 /Users/lordskyler/Holstein_test.worktrees/codex-paper-v-divergence-parameter-map-20261010 b9f99271fc3ddf3db604d3acb64617a291bec7e1 && shasum -a 256 -c "$LOCK"` | the readable-inputs check of `unrepaired_t140.json` and `boundary_basin_span400_exact.json` |

## Running them together

The four threads have disjoint write scopes and no ordering dependency.
Their heavy steps are the repaired chains (thread 1 increment 6, up to three
chains; thread 2 increment 4, up to four; thread 3 increments 5 and 6, up to
three), each process about `650 MiB` and single segments of at most `900 s`;
the sweeps of threads 1 and 4 are many small single-core propagations. Four
disjoint scopes do not authorize four simultaneous heavy runs: each thread
sizes its concurrency by `memory_pressure -Q` at launch (C9). Each thread
writes its report-back to its own `REPORT-<topic>-codex-20261010.md` on its
own branch; the Claude side reviews and merges the worklog entries (C17).


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
