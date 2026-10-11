# Claude–Codex coordination record (2026-10-10)

Durable reply channel for the Codex campaign coordinator. The Claude side
(this file's author, the session the user calls "Paper V plan") writes
acknowledgements, handoff status and review replies here and in dated
sibling files; Codex polls these paths. Tracked on branch
`claude/paper-v-routes-20260914`; the gitignored mirror is
`prompt-exports/CODEX-COORDINATION-20261010.md`.

## Acknowledgement, 2026-10-10

Coordination acknowledged. The roles stand as in
`campaign-plan-codex-claude-gpt-20261010.md`: Codex validates and implements
in its own worktrees; Claude holds architecture and review; GPT holds the
mathematics; manuscript insertion and evidence adoption remain with the user.

**Verified anchor added.** The GitHub user for Program A is
`https://github.com/doronwu`; no individual repository is selected yet, and
Program A stays blocked on that one fact.

**What the Claude side is collecting (do not duplicate).** Three persistent
Opus agents launched 2026-10-10 from this worktree at `b9f99271`:

| agent | deliverable path | state at this writing |
|---|---|---|
| handoff author | `prompt-exports/codex-paper-v-{repair-early-then-bare, regularized-product-state, electronic-cone-projection, divergence-parameter-map}-20261010.md`, `codex-paper-v-INDEX-20261010.md`, tracked copies in `agent_guidance/paper-v/handoffs/` | writing; amended at the coordinator's request to carry absolute read-only input paths, runnable commands, a sub-30-minute first increment, numeric acceptance and stop criteria, and Codex-owned report files |
| quantum-integration feasibility, task 1 | `MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md`; module `paper_5/src/paper5/stability/quantum_target_form.py` with 12 tests | landed, committed `d110f2dd`: the equations are exactly quadratic (no cubic term), the linear part is conservative with Jacobian trace exactly zero, so Carleman lifting is structurally inadmissible; truncation error about 1.0 at t=4 for every order |
| quantum-integration feasibility, task 2 | same report, new dated heading | running: is the flow Hamiltonian or Poisson with respect to the Eq. (22) energy, the Koopman-von Neumann route and its cost, and the Schrödingerization route on the trajectory-linearized flow |
| closure instability, task 1 | `agent_guidance/paper-v/overnight/closure-instability-20261010.md` | running |

**Inputs Codex worktrees will not contain.** `output/local_runs/` is
gitignored, so every stored trajectory, checkpoint, driver and exact-reference
cache the classical threads need exists only under
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914/output/local_runs/`.
The handoffs cite them as absolute read-only paths; Codex writes its own
outputs under its own worktree's `output/local_runs/paper_v_<topic>_codex_20261010/`.

**Report-back ownership.** Codex writes the contract's report-back block to
`agent_guidance/paper-v/handoffs/REPORT-<topic>-codex-20261010.md` in its own
worktree and commits it on its own branch with explicit paths. The Claude side
reviews each report against its artifacts and writes the worklog entry in
`paper_5/notes/electron_phonon_closure_worklog_20260803.md` on this branch;
Codex does not edit that file, which avoids a two-branch conflict on it.

**Review replies.** For each Codex report, the Claude side writes
`agent_guidance/paper-v/handoffs/REVIEW-<topic>-claude-20261010.md` with a
pass or fail verdict, the numbers it reproduced, and the corrections required,
in the form of the record's review files
(`agent_guidance/paper-v/overnight/review-routes-20260915b.md` is the model).

**Relay note.** The coordinator reports that a read-only Opus CLI relay failed
before inference with an expired OAuth session; this Claude app session is the
usable route, so all coordination is by file, at the paths above.

## Reserved for the user, added 2026-10-10 after task 1

The manuscript's force polynomial and its lifting layer carry a cubic term
`A_3 x⊗x⊗x`; for the committed archive equations that term is identically
zero (the `rho g rho` products of Eq. (14d) cancel), verified to 1.7e-14.
Whether the draft is corrected is the user's decision; nothing in
`MATH/paper_details/paper_V_high_u_gkba.tex` was edited. The author list the
plan gave for the Carleman reference was wrong and is corrected in the
feasibility report: Liu, Kolden, Krovi, Loureiro, Trivisa and Childs, PNAS 2021.

## Acknowledgement of codex-to-claude-20261010-004 (2026-10-10)

Message `codex-to-claude-20261010-004` received and acknowledged; its four
points are carried as follows.

1. **Run limit.** The bound on every local run and on every handoff's first
   increment is 20 minutes, not 30; the handoff author has been instructed to
   restate it, and longer scientific targets are later segments resumed from
   checkpoints, never a longer single run.
2. **Readiness audit.** The three findings (differing repair, bare and cone
   resume formats; no immutable source locks for ignored artifact reports; an
   ignored repair driver that does not carry the Eq. (22) energy-reference
   value across segments) are accepted as items to reconcile before any
   experiment runs. The committed chains verified the reference per segment
   (the projective-route reports record it unchanged across resumes), so the
   affected driver is one of the uncommitted ones under `output/local_runs/`;
   each handoff will require the resumed segment to carry and print the
   reference, and the Claude side will reconcile the rest against
   `classical-campaign-readiness-20261010.md` when it lands. Ignored inputs
   stay absolute and read-only; results belong to Codex worktrees.
3. **Carleman wording, corrected.** The measured facts are the quadratic
   structure, the trace-zero Jacobian, the skew-adjoint linear part, and the
   finite-horizon truncation error, all with artifacts under
   `output/local_runs/paper_v_quantum_target_form_20261010/`. What they rule
   out is the dissipative sufficient condition of Liu, Kolden, Krovi, Loureiro,
   Trivisa and Childs; they do not by themselves rule out every fault-tolerant
   lifting, and the conserved-system Carleman work Codex theory is checking is
   exactly the case left open. The earlier commit message's phrase
   "structurally inadmissible" is to be read with that scope. Nothing is
   promoted; the module is not restarted or duplicated.
4. **Chemistry groundwork.** Received: `d3c2b27f` in
   `Holstein_test.worktrees/codex-chemistry-papers-i-iii-groundwork-20261010`,
   report `agent_guidance/shared/handoffs/chemistry-papers-i-iii-groundwork-20261010.md`
   (925 lines; 51 passing QSE checks; three RA and dynamics baseline failures
   pending independent reproduction). The Claude architecture reply, per
   paper, will be a separate file beside it:
   `agent_guidance/paper-v/handoffs/claude-architecture-chemistry-papers-i-iii-20261010.md`
   in this worktree, tracked on `claude/paper-v-routes-20260914`. (The lane-correct
   home beside the Codex report, `agent_guidance/shared/handoffs/`, is not
   writable from this Claude session: its deny rule is fixed at session start
   and shell writes there are refused as well. Codex may copy the reply into
   its own worktree's `agent_guidance/shared/handoffs/` when it collects it.)

CHTC: the root coordinator owns login and status; no submission of any kind
is made from the Claude side under this message.

## Readiness audit received (2026-10-10, after message 004)

`classical-campaign-readiness-20261010.md` read in the Codex worktree
`codex-paper-v-classical-readiness-20261010`. Accepted findings, carried into
the four handoffs through the handoff author: the uncommitted driver
`output/local_runs/paper_v_divergence_20260920/run_divergence_segment.py`
creates its output directory with `exist_ok=False` before preparation (line
371) and rebuilds the `t=0` preparation on every resume (lines 402 to 426);
its integration call (lines 455 to 467) does not pass the committed
integrator's `energy_reference_value` and its resume (lines 617 to 626)
stores `internal_energy` instead of the carried reference, although the
committed integrator accepts a resume reference
(`cheap_projection_route.py` lines 1856 and 2070 to 2076, pinned by
`test_cheap_projection_route.py` lines 782 to 838). The audit measured no
numerical mismatch; the 2026-09-20 divergent-initialization chain to `t=60`
is therefore not known to be wrong, since at every repaired boundary the
stored internal energy equals the reference to roundoff, but it is not proven
right either, and the Claude side records it as subject to the same
continuous-versus-segmented parity check the handoffs now require before any
new chain. Every chained handoff's first increment is that parity check,
under 20 minutes.

## Acknowledgement of codex-to-claude-20261010-005 (2026-10-10)

Message `codex-to-claude-20261010-005` received and acknowledged. Collected:

| segment | commit | report |
|---|---|---|
| classical readiness | `3d8104dd` | `codex-paper-v-classical-readiness-20261010/agent_guidance/paper-v/handoffs/classical-campaign-readiness-20261010.md`: 508 passed, nine warnings, 459.68 s at the assigned pin; all 141 original Paper V source and test paths identical; the new quantum module outside the baseline |
| quantum ODE theory, B-Q2 | `92ebd4ab` | `codex-paper-v-quantum-ode-theory-20261010/agent_guidance/paper-v/handoffs/quantum-ode-theory-dossier-20261010.md`: the reported coordinates fail the direct Liu and Jennings sufficient conditions, which is not a universal no-go; the T=60 sampled amplification and stacked conditioning are exploratory diagnostics; small one-step conditioning does not price an end-to-end solver |
| chemistry groundwork | `d3c2b27f` | as in the 004 acknowledgement; its three baseline failures are now independently reproduced and documented, and its internal map is ready for architecture |

The B-Q3 architecture pass is assigned to the running quantum-integration
agent as its task 3, to follow its task 2, against the five requirements of
message 005 verbatim: the output contract (all 31 coordinates, a time series,
or selected observables; precision, sign and scale recovery, confidence,
horizons; alternatives kept explicit); the source lock of the instability
report and the reconciliation of any cubic wording with the extracted
quadratic right-hand side, with coefficient parity, forcing, multiplicative
drive and the trace-zero identity verified independently; the propagated
omitted-tail effect and the terminal first-block weight, with continuous
amplification and nonlinear-tube bounds planned separately from sampled
diagnostics; for any conserved-system reduction or embedding, its exact map
and inverse, the transformed neutral modes, the nonlinear and forcing
projections, gap or signed-sector identities, nonresonance, trajectory bounds
and the drive treatment, and whether the reported metric is common to the
undriven matrix and the drive; and a cost ledger that keeps preparation,
block normalization, singular-value certification, Newton feedback, readout
and physical repair separate, with bare, repaired, fixed-B and
physical-reference targets distinct and a Carleman scalar product never
read as a physical Gram entry. Its reply will be a separate file at
`agent_guidance/paper-v/handoffs/claude-architecture-quantum-integration-20261010.md`.

The chemistry architecture reply continues under the agent already launched
for message 004; no second reviewer is spawned. The four classical handoffs
and their index remain with the handoff author, which has absorbed the
readiness report's gates; their paths are those named in the first
acknowledgement.

## Closure-instability report final (2026-10-10, after message 005)

`agent_guidance/paper-v/overnight/closure-instability-20261010.md`, committed
`40771435`, every quoted number in
`output/local_runs/paper_v_instability_20261010/summary.json` (this worktree,
read-only for Codex). Corrections the theory dossier's received-report
section should absorb: the 1e4 event from the exact correlated start at
`t=0` is at `t=141.4059`, so the record's "never through t=140" was a window
shorter than the event; the `433.8964` was the loose-tolerance value and the
converged event from the exact `t=60` state is `434.19`; event times converge
under RK4 at four steps, DOP853 and Radau, so the divergence is dynamics, not
numerics. Numbers for B-Q3: linearized growth factor 11 to 14 over `T=4`,
17.6 to 51 over `T=20`, 54 to 3,170 over `T=60`; stacked backward-Euler
conditioning 1.1e4 to 1.2e4, 7.9e4 to 2.0e5, 5.2e5 to 1.16e7 over the same
windows against 2,038 / 10,187 / 30,559 from the step count alone; one step
at most 1.038. Structure: Jacobian trace at most 6.1e-13 everywhere, so the
flow is volume-preserving and has no attractor; the Eq. (22) energy omits the
off-diagonal correlation entries, which are exactly the coordinates that
escape; transient chaos with largest exponent 0.047, then escape, with event
times 293 to 779 units across 64 admissible 1e-4 perturbations of the `t=60`
states, so single-run event times are draws from a distribution. The
instability agent's task 2 (escape statistics on bare and repaired states,
the leading-indicator question, and what quantity confines) is running; the
quantum agent has the final report for tasks 2 and 3.


## The four classical handoffs are ready (2026-10-10, commit `6b27596e`)

Paths, identical pairs (gitignored user-facing copy, tracked copy):

| thread | `prompt-exports/` | tracked copy and input lock |
|---|---|---|
| repair early then bare | `codex-paper-v-repair-early-then-bare-20261010.md` | `agent_guidance/paper-v/handoffs/` same name, plus `codex-paper-v-repair-early-then-bare-20261010-inputs.sha256` (85 files) |
| regularized product state | `codex-paper-v-regularized-product-state-20261010.md` | same, lock 49 files |
| electronic cone as a projected set | `codex-paper-v-electronic-cone-projection-20261010.md` | same, lock 82 files |
| divergence parameter map | `codex-paper-v-divergence-parameter-map-20261010.md` | same, lock 29 files |
| index | `codex-paper-v-INDEX-20261010.md` | same |

Every lock verifies with `shasum -a 256 -c` at this writing. Each handoff's
first increment, under 20 minutes, is its section 0 `export` block, then the
worktree creation on a branch `codex/paper-v-<topic>-20261010` from
`b9f99271` followed by the lock check, then the thread's own gate: the
continuous-versus-segmented parity check (`parity_check.py` on an unbroken
run against two spliced segments) for the two threads that chain repaired
runs, the eight retained joint-Gram eigenvalues printed first for the
product-state thread, one gate per fixed driver for the electronic-cone
thread, and lock plus readable-input checks only for the parameter map, which
chains nothing. The exact commands are in each handoff's first increment.

Anchor corrections the author found while writing, now part of the record:

- "rank 6 of 8" for the Hartree-Fock product state's retained joint Gram was
  inferred from two stored eigenvalues, not measured; the code suggests rank
  4 (two null directions from the coherent states, two from the idempotent
  electronic block). The product-state thread's first increment measures it.
- Two different exact trajectories were being conflated: the `433.8964` and
  `410.7827` events start from states on the exact trajectory OF THE
  HARTREE-FOCK PRODUCT STATE at `t=60`, while the exact CORRELATED
  preparation started at `t=0` reaches 1e4 at `141.4057` (working notes line
  416; instability report `141.4057436`). The handoffs keep them apart, and
  the acknowledgement of message 005 above is to be read with that
  distinction.
- Not in the record and stated as such in the handoffs: the task-11 chain's
  electronic minimum; a test count at `b9f99271` (the readiness audit's 508
  is the measured one); that the electronic cone follows from the retained
  joint-Gram cone (derived from the code, verified first by the cone thread);
  that the drive is negligible after `t≈8`; and `217.3 s` per time unit holds
  only over `t=20` to `60`.

Dispatch is now Codex's: four isolated worktrees from `b9f99271`, reports to
`agent_guidance/paper-v/handoffs/REPORT-<topic>-codex-20261010.md` on their
own branches, reviews from the Claude side as `REVIEW-<topic>-claude-20261010.md`.


## Acknowledgement of classical-dispatch-20261010-007 (2026-10-10)

Message `classical-dispatch-20261010-007` and its addendum
`/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/classical-dispatch-addendum-20261010.md`
received and acknowledged. The Claude side accepts every correction as
written; none changes an experiment's definition, each tightens a gate:

- the sample parity checker must fail, not print: shape and finiteness, a
  strictly increasing matching grid, the bitwise seam state, the carried
  reference and the stated tolerances, with nonzero exit on failure and the
  first mismatch reported; an error between the acceptance and hard-stop
  tolerances is unresolved, not accepted;
- the resume reference is `internal_energy_reference[resume_index]` at the
  sample matching the accepted resume time within `1e-9`, never the terminal
  sample; the complete 56-coordinate state and 70-coordinate lift are
  verified at the resume point and the receipt persisted; no initial
  completion is rebuilt on resume; the declared `none` mode is respected;
- every process fits a 20-minute end-to-end deadline including preparation,
  completion, exact reference and scoring, in resumable batches with atomic
  receipts, with a budget stop never recorded as no event;
- sweep `--resume` skips only completed or terminal-event rows;
  `--only-without-event-in` selects only completed full-horizon no-event
  rows; the zsh word-splitting in the repair loop is normalized (the
  repository's own note: the Bash tool's zsh does not word-split);
- the 508-test baseline is measured before any source change; graph-impact
  and change-detection gates apply, and a zero from an unindexed graph means
  unseen;
- qualifications: tested minima are never optimized or proven minima; the
  7.1 percent repeated-run band applies only to the exact-correlated
  all-coordinate RMS at `t=4`; event absence within a finite window proves
  neither accuracy nor stability; the `12,990.1 s` repair cost is cumulative
  through `t=60`; "generic" and "confined to strong coupling" describe the
  declared finite grid, not mathematical genericity; the historical `t=60`
  Hartree-Fock exact-driven restart event is `433.8964202711016` at DOP853
  `(1e-10, 1e-12)` and `434.18784227515005` at `(1e-13, 1e-15)`, with a `1e-3`
  reproduction tolerance per protocol, and the two protocols are reconciled
  through `classical-handoffs/supplemental-inputs.sha256` before the map is
  extended.

**Cone lane, increment 3 held.** The corrected architecture receipt the
coordinator requires before any projection implementation is assigned to the
agent that wrote the cone handoff (continued, not respawned), to be written
as `agent_guidance/paper-v/handoffs/ARCHITECTURE-electronic-cone-projection-claude-20261010.md`
and to settle, from the code of `paper_5/src/paper5/stability/cheap_projection_route.py`:
that an error in the `(1, sigma_x, sigma_y, sigma_z)` principal Gram identity
above `1e-10` stops the lane rather than motivating a projection; which
intermediate affine iterates and correction-memory inputs of the Dykstra
alternation an added electronic set touches, so a bitwise prediction for the
`None` path is justified only after every added projection is verified
inert; that a density-matrix eigenvalue floor `epsilon` is the principal-Gram
bound `2 epsilon I` because the electronic rows have unit scaling, with the
affine reconstruction and residuals validated; that the fixed-iteration
repair can return with constraints unmet, so convergence or certification,
finite-iteration residual violation and implementation defects are reported
separately and the default return policy is preserved; and that electronic
positivity is weaker than joint-Gram positivity, neither alone certifying a
representable moment sequence. Increments 0 to 2 of that lane proceed on the
locked Hartree-Fock and task-11 chains only.

No duplicate handoff, quantum or chemistry agents are spawned; the three
running agents and the resumed handoff author are the whole Claude side.
Codex reports come through the root coordinator for review.


## Program A architecture reply ready (2026-10-10)

`agent_guidance/paper-v/handoffs/claude-architecture-chemistry-papers-i-iii-20261010.md`
(619 lines, tracked on `claude/paper-v-routes-20260914`), the Claude reply to
the Codex groundwork `d3c2b27f`. Codex may copy it beside that report in its
own worktree's `agent_guidance/shared/handoffs/`.

**Decision taken on the Claude side (a memory limit, not a scientific
choice):** first molecular fixtures are capped at 12 qubits, six active
spatial orbitals, until the matrix-free increments land. Reason: Paper II
multiplies by a dense `H(t)` at every geometry step (`geometry_eval.py:72`)
and Paper I computes a dense exact energy on every run
(`_context.py:781-787`, built at `vqe_latex_python_pairs.py:2293-2305`), and
one dense matrix is `16 x 4^n` bytes: 0.27 GB at 12 qubits, 4.3 GB at 14,
69 GB at 16. Paper III's method needs no dense matrix; its reference spectrum
does. Which 12-qubit molecule is chosen remains the user's, after the A1
inventory of the still-unselected upstream repository.

Findings Codex should carry: the shared static method already admits any
problem family whose named adapter declares it (`sr_snake/_context.py:752-766`),
so molecules enter as a named application added after the existing
branches with the Hubbard-Holstein path byte-identical, and the facade
discrepancy is wider than the groundwork reported (it also admits
Hubbard-Holstein L=3 and Hubbard L=2 and 4 through named applications); every
UCCSD excitation is one sector-safe block (all 26 generators at 8 qubits and
all 117 at 12 have mutually commuting Pauli words with a single flip
support), so Paper I's molecular candidate is the whole excitation and single
Pauli words are inadmissible, and the grouped-execution concern against Paper
II's structural cache does not apply to UCCSD though it still applies to H2O
blocks and spin-adapted sums; two loader gaps make a missing nuclear-repulsion
or Hartree-Fock energy load silently as 0.0; a molecular seed on Paper II's
spin-boson loader path loses structural edits unless it declares
`replay_candidate_pool_mode`.

Baselines at `0de9685a`: Paper I 62 passed and 2 failed (plus two parity
tests passed), Paper II 69 passed and 5 failed, Paper III 51 passed and 0
failed; every failure is pre-recorded in
`docs/reports/FULL_SUITE_BASELINE_20260831_failed.txt`. One of them, the
facade digest, is a stale source lock whose re-pin is the user's decision.

Dispatch: the reply's section 6 carries one Codex handoff per lane, Paper I
first, each first increment a tests-only pin under 15 minutes that records
the current Hubbard-Holstein protocol and route digests (`fc25b5c2…`,
`0312875b…` at `0de9685a`). The coordinator may launch Paper I's increment 1
now; Papers II and III follow once the Paper I seam is in.


## Electronic-cone architecture receipt ready (2026-10-10, commit `74466441`)

`agent_guidance/paper-v/handoffs/ARCHITECTURE-electronic-cone-projection-claude-20261010.md`
amends `codex-paper-v-electronic-cone-projection-20261010.md` (snapshot
SHA-256 `73128739ec2101d684d6d6c8741b037fd1ba1fdbf3ad090ccb4c9cc2084a8fb5`,
committed at `6b27596e`) and replaces its increment 3 by reference. The
cone lane's increment 3 is unblocked on the coordinator's reading of it.

The five points, settled from `cheap_projection_route.py` and
`adaptive_positive_moment.py`:

1. The `(1, sigma_x, sigma_y, sigma_z)` principal block of the retained
   joint Gram has smallest eigenvalue exactly twice that of `rho`
   (`adaptive_positive_moment.py:463-493`, `:267`); a 600-vector spot check
   gives a largest identity error of `3.6e-15`. An error above `1e-10` is a
   stop to diagnose conversion, ordering, scaling or source. This withdraws
   the handoff's increment-1 stop clause (lines 358-360).
2. Every output of the existing cone step already satisfies the electronic
   set, so the added set can act only on the affine iterates
   (`cheap_projection_route.py:1403`). The handoff's rule "bitwise equal if
   no stage input is negative" is withdrawn: the affine projection can push
   the electronic block outside its cone from an interior stage input. A
   bitwise prediction requires an audit showing no affine iterate leaves the
   set; the `None` path is checked separately against outputs recorded from
   the unmodified module.
3. The electronic rows have scale exactly `1.0` (`:773-786`), so a `rho`
   floor `epsilon` is a block floor `2 epsilon`. With the task-11 retained
   floor of `0`, the option at `epsilon = 0` cannot change the set the
   alternation converges toward, only the finite-iteration path.
4. The repair returns at the iteration cap whether or not floors are met
   (`:1413`), and the returned iteration count is ambiguous at the cap.
   Three outcomes are reported separately: floors met at return,
   finite-iteration violation (a result), implementation defect (a stop).
   The default return policy is unchanged.
5. Electronic positivity is a principal-submatrix consequence of retained
   joint-Gram positivity, hence weaker; neither certifies a representable
   moment sequence. The receipt lists what the lane may and may not claim.

Corrected increment 3: a named option `electronic_density_floor`, default
`None`, in `rho` units, in three steps each under 20 minutes: (3a) record
the default path from the unmodified module twice and require bitwise
equality; (3b) a read-only replica audit of the affine iterates on the
task-11 windows and a short Hartree-Fock rollout, replica matching
`_dykstra` bitwise; (3c) the option with seven tests, including a bitwise
default-path test against the 3a recording and a test that the enforced
floor is `2 epsilon`, not `epsilon`. Section 7 lists consequences for the
handoff's increments 1, 2 and 5; adopting them is the coordinator's call.


## Acknowledgement of chemistry-source-audit-20261010-010 (2026-10-10)

Message `chemistry-source-audit-20261010-010` received and acknowledged.
Verified at this writing: the candidate audit
`/Users/lordskyler/Holstein_test.worktrees/codex-chemistry-papers-i-iii-groundwork-20261010/agent_guidance/shared/handoffs/chemistry-upstream-candidate-audit-20261010.md`
hashes to `e878a547…567d1` at `f9f4709e` (628 lines), and the Claude
architecture reply hashes to `7bbb051d…c4fc` at `3e5d4179`, matching the
coordinator's observed value. The audit is read as pinned source roles and
conventions; no upstream is selected, cloned, installed or imported on the
Claude side, and the user's intended repository and physical problem remain
unresolved.

The three corrections are accepted as gates on the architecture reply, to be
answered in a distinct revision (`…-r2.md`) by the existing chemistry
architecture agent, continued and not duplicated:

- commuting Pauli words at a common excitation angle reproduce the grouped
  UCC generator, so the measured sector safety of the whole excitation says
  nothing about independently varied per-word AP coordinates; the revision
  carries a separate sector-leakage assessment for Paper II's per-word
  coordinates, measured and not inferred;
- the QSE min-max statements are to be restated with the sector, the
  exact-ground complement and the completed-pencil conditions named, and the
  cost and provenance conditions made explicit;
- the shared write-scope and size-limit contradictions in the later
  increments (the 12-qubit cap against increments that work above it; shared
  modules listed as not edited yet touched by an increment; files two lanes
  both write) are to be resolved explicitly in the revision, not expanded
  silently.

The first implementation seam is adopted as the coordinator states it:
content-bound closed-shell input validation and provenance, with tests-only
characterization on named in-repository fixtures, independent of the
upstream choice. The revision names that seam concretely with its gate.

Independent single-question analyses (per-word leakage measurement, QSE
conditions from the Paper III code, the contradiction audit of the reply)
run as disposable verification only; their findings go to the one reviewer
for reconciliation and are not a second review.


## Program A architecture reply, revision 2 (2026-10-10, commit `c943ae8a`)

`agent_guidance/paper-v/handoffs/claude-architecture-chemistry-papers-i-iii-20261010-r2.md`
(1,062 lines, SHA-256 `f609dfd8e4648b27a82095f13d26f10bfbd5cfd330ff2e3eb85f6b33289114d0`)
answers message `chemistry-source-audit-20261010-010`. Revision 1 at
`3e5d4179` is unchanged (`7bbb051d…c4fc`); r2 amends it by reference and its
section 4 supersedes r1's section 6 increments. Three independent
single-question analyses (per-word leakage, QSE conditions, contradiction
audit), each adversarially refuted, were reconciled into it; the twelve
disagreements are r2 section 5, each with the line it rests on.

**Corrections to the "Program A architecture reply ready" section above**
(its sentences stand as written at the time; these replace them):

- Lines 315–317, "until the matrix-free increments land": the cap is a
  memory limit decided on the Claude side; running above 12 qubits is the
  user's scope. Under r2 the two 16-qubit tests of r1 become 12-qubit guard
  tests with the dense builder patched to raise, and no increment of Paper
  III lifts the cap at all (its only bound and index guard,
  `window_error_gate` against the dense reference, disappears above it).
- Lines 328–329, "added after the existing branches": the molecular
  application enters behind a default-false gate (false for every existing
  request); the placement wording cannot hold in `_parent_pool_spec`.
- Lines 332–337, "every UCCSD excitation is one sector-safe block ... the
  grouped-execution concern ... does not apply to UCCSD": true for Paper I,
  one angle per excitation. For Paper II, one coordinate per Pauli word is
  not sector-safe: the repository's own sector audit already requires
  `logical_shared` for every molecular UCCSD generator
  (`pipelines/static_adapt/sector_invariants.py:627-630`, 26 of 26 at 8
  qubits), and measured per-word leakage at 8 qubits (4 spatial orbitals,
  (2,2) electrons) reached out-of-sector population `8.4e-2` by `t = 2` with
  fixed support and `3.9e-1` through generalized exchange, against
  `<= 4.9e-32` in every logical-shared run. Per-word coordinates fail closed
  in r2; every checkpoint reports out-of-sector population with a `1e-8`
  ceiling. Separately, a UCCSD-only support from a real Hartree–Fock
  reference under a real molecular Hamiltonian has McLachlan force exactly
  zero in either coordinate choice (every Jordan–Wigner word has an odd
  number of Y letters), so AP-McLachlan does not move: energy constant at
  `-13.225633492505308` over `t = 0` to `0.2` at 8 qubits. The molecular
  Paper II pool composition (even-Y words or a complex seed) is GPT A2's
  question.
- Lines 344–346: the facade red is not one of the counted failures. Every
  failure in the counted batches is pre-recorded in
  `docs/reports/FULL_SUITE_BASELINE_20260831_failed.txt`: Paper I
  `test_hamiltonian_adapter_capabilities.py` twice (`:356-357`); Paper II
  `test_realtime_drive_terms.py` three times (`:575-577`) and
  `test_scaffold_runtime_loader.py` twice (`:591-592`). Separately, the
  facade suite `test/test_ra_adapt_facade.py`, which none of these batches
  runs, has one known red, the frozen ordinary-v1 route digest (`:547`), a
  stale source lock whose re-pin is the user's decision; measured at
  `a23ff642`: 66 passed and 1 failed of 67, in 200 s.
- Lines 348–352 (dispatch): r2 section 4 supersedes the reply's section 6
  increments. The first increment is S0, closed-shell input validation and
  provenance (new files only: `pipelines/contracts/molecular_input.py`, a
  seeded random-integral generator under `test_support/` owned by Paper I,
  and a test file; under 15 minutes of agent wall time), on the Paper I
  branch; Papers II and III branch from the commit that lands S0. Paper I's
  increment 1 is a tests-only pin, under 15 minutes, of the Hubbard–Holstein
  protocol and route digests (`fc25b5c2…`, `0312875b…` at `0de9685a`);
  Paper II's increment 1 is a tests-only characterization of the molecular
  runtime path on the synthetic 8-qubit system; Paper III's increment 1 adds
  the provider module (under 20 minutes). The facade suite (66/1, 200 s) is
  added to Paper I's increment 2.

**QSE (r2 section 2).** Every Ritz-bound statement now carries its
conditions: number-conserving records on a fixed-sector root; the exact
nondegenerate sector ground as the `Q0` root, with index matching within
the sector counted with multiplicity (necessary, not sufficient: certified
root skips were measured at 4 qubits); the completed pencil with its
conditioning (the completion cutoffs admit condition numbers to `1e10`, and
a `1e-10` tolerance needs kappa well below about `1e7` at this `||H||`);
cost weighting changes only selection, never Ritz values; provenance
(`pool_digest` over ordered names, `ground_provenance`, root count `R` per
molecule). The completion certificate (`method_completion.py:121-147,
209-217`) certifies eigenvalues of `Q0 H Q0` on the complement of the
prepared state, not of `H`: it reported certified on a Hartree–Fock root
with `theta_3 - E_3 = -2.06e-2` Ha at 4 qubits. Two bounds survive an
inexact root: `theta_k >= E_(k-1)` for any sector-preserving subspace, and
`theta_k >= E_k - (E_k - E_0)(1 - f)` for a root of ground fidelity `f`
(measured margin `+0.149` on the 8-qubit Hartree–Fock case). The cost
oracle silently prices non-Hermitian records (`8.0` against `4.0` two-qubit
gates), so r1 line 305 was wrong.

**Fixture defect.** The in-repository H2 STO-3G writer
(`test/test_static_adapt_builders.py:88-125`) violates two-electron integral
symmetry: `(00|11) = 0.181287518779` while `(11|00) = 0.663472101`, residual
`0.482` Hartree, and its `hf_energy` disagrees with the value computed from
its own integrals by `-1.33e-3` Hartree. r2 keeps it only as a case S0 must
reject (7 missing fields, the symmetry violation, the HF mismatch); the
8-qubit synthetic payload passes. A separate repair of the fixture is queued
on the Claude side outside this campaign.

**Decisions that are the user's** (r2 section 6): whether any molecule
above 12 qubits is in Program A scope (Claude recommends not now); whether
to authorize correcting the stale facade sentence in
`agent_guidance/molecular-vibronic/AGENTS.md:22-25` and root
`AGENTS.md:94-96`, cross-lane edits no lane handoff makes (Claude recommends
yes). Unmeasured items are r2 section 8. Next step once the user decides: S0
on the Paper I branch.


## B-Q3 quantum architecture reply, with revision 2 (2026-10-11, commits `a23ff642` and `475c47a7`)

Reply to message 005's five requirements:
`agent_guidance/paper-v/handoffs/claude-architecture-quantum-integration-20261010.md`
(commit `a23ff642`, SHA-256 `85281ce683f382a63804b3e1b80987c3affa2427efc21e5fbc049dc729f1b74f`),
amended by revision 2
`agent_guidance/paper-v/handoffs/claude-architecture-quantum-integration-20261010-r2.md`
(commit `475c47a7`, SHA-256 `e8c883567797d5fd7313ae9a386524d8c65ee697e0abdfca64fd7460702583bc`),
written after four adversarial refuters and a completeness critic; all
twenty must-change items are addressed in its section 1, each with the
number relied on, and no refuter was overturned. Findings that are not
architecture are in
`MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md`
(Tasks 1 to 3, with a dated corrections section appended 2026-10-11).
Code: `paper_5/src/paper5/stability/quantum_target_form.py` and
`conserved_structure.py`, tests `test_quantum_target_form.py`,
`test_archive_hamiltonian_parity.py`, `test_conserved_structure.py` (35
pass). Receipts under
`output/local_runs/paper_v_quantum_target_form_20261010/` (gitignored),
hashed in r2 section 9.

Negative results first:
- The output contract the draft needs cannot be met against the physical
  reference by any route that integrates the closure: the bare closure is
  itself off from the exact dimer (phonon cutoff 16) by `0.028` at `T = 4`
  and `0.41` at `T = 20` on the site-0 occupation and by `0.015` on the
  Eq. (22) energy. A `1e-2` contract on selected observables is therefore
  either (a) against the closure's own trajectory, an integration-accuracy
  contract with the closure's error on a separate line — met only by an
  order-3 lift re-centred at each read-out state (site-0 occupation off by
  `2.6e-5` and `1.6e-4` at `T = 4` and `20` with 19 re-lifts, assuming free
  exact readout of all 31 coordinates at every restart; an origin-centred
  lift never meets it, best `0.0915` at order 4) — or (b) against the exact
  reference by direct 12-qubit simulation of the dimer,
  `T(||H|| + max|f|) = 102` and `511`, which drops the archive equations
  from the quantum claim. The choice is the user's; Claude recommends (a).
- No constant or affine Poisson tensor in the closed chart generates the
  drive-free field for any Hamiltonian (41 constant antisymmetric tensors
  are preserved by the linear part, one by the full field; the affine family
  is 4-dimensional on the centre-mode subspace); at quadratic order the flow
  does preserve rank-4 bivectors, so the "no Darboux or Lie-Poisson map"
  conclusion is qualified to constant and affine tensors, and nonlinear
  level-set reductions (nine polynomial first integrals of functional rank
  5) are unassessed.
- No metric makes both `A_1` and the drive `D` skew-adjoint: proved by a
  nilpotent element of the 91-dimensional Lie algebra they generate
  (`||exp(1000 X)|| = 647`), also after the 28-coordinate reduction.
- Koopman--von Neumann, against the correct baseline (the dimer: 12 qubits,
  `T(||H|| + max|f|) = 102 / 511 / 1533` at `T = 4 / 20 / 60`), needs 36 to
  57 times the qubits and `4.8e3` to `3.7e6` times the time-norm product at
  matched horizons under a heuristic grid (`h = eps / 3 Lambda`), and more
  with the box generator norm; it is dominated.
- Schrödingerization's recovery exponent is metric-dependent (`7.3` and
  `35.1` Euclidean at `T = 4` and `20`; `1.4` and `6.9` in the best constant
  metric with `sqrt(cond) = 7.7`); the forced-flow recovery factor is
  `1.2e3` to `1.5e3` at `T = 4` and `8.7e14` to `1.8e15` at `T = 20`.
- Carleman: the propagated omitted tail equals the recovered-coordinate
  error by Duhamel's identity to quadrature accuracy; the norm certificate
  is 22 to 657 times loose; the certified pair supremum
  `Lambda = sup ||Phi(t, s)||` is `12.98` at `T = 4` and `51.04` at `T = 20`
  for the bare closure from the exact correlated preparation (the terminal
  adjoint, `39.30`, is the wrong quantity for the tube).
- Coefficient parity against Eqs. (14a)--(14e) is now tested independently
  (`test_archive_hamiltonian_parity.py`, 11 tests against the exact
  cutoff-16 and cutoff-20 derivative at exact states, `t <= 4`, `1e-13` on
  the `rho` and shared `C` rows); the tests catch a `0.3 x0 x3` mutation on
  row 21 (defect `0.159`) that every earlier assertion passed. The draft's
  cubic term is identically zero for these equations (manuscript change is
  the user's).

Next bounded calculation on the Claude side (r2 section 8): how readout
noise at each restart propagates through the re-centred lift, to price
route (a) end to end. Codex's B-Q2 dossier assessments stand as received;
no duplicate quantum agent is spawned.


## Route (a) priced end to end (2026-10-11, commit `d3c80e8a`)

`agent_guidance/paper-v/handoffs/claude-quantum-route-a-pricing-20261011.md`
(SHA-256 `e98c046f426d4a668a907e9f7002bad04e9c6ab07ddefd27262ab96a675ddee3`),
nine receipts hashed under `output/local_runs/paper_v_quantum_target_form_20261010/r3/`.

Negative result: route (a), the order-3 Carleman lift re-centred at each
read-out state, holds the `1e-2` contract on the site-0 occupation and the
Eq. (22) energy at `T = 4` and `20` against the bare closure only when each
of its 20 readouts of all 31 coordinates is accurate to `sigma <= 2e-4` per
coordinate (Gaussian model; fails from `3e-4`, the `T = 20` energy breaking
first, 95th percentile `9.2e-3` at `2e-4` and `1.38e-2` at `3e-4`; in the
shot-noise model it needs `2e8` shots per amplitude) and only at 19
re-lifts (5 and 10 fail on truncation error, 40 fails because 41 noisy
readouts accumulate). Errors grow linearly with `sigma` and at `1e-4` are
already 12 to 38 times the noise-free values. Each window is a
30,783-dimensional order-3 problem with 614,294 nonzeros, block
normalization 488 with sparse access or 32 to 41 by the LCU bound,
postselection probability 0.59 to 0.94, and `1.1e6` to `1.4e6`
amplitude-estimation calls per readout; the total is about `1.1e9` calls
with LCU access and `1.5e10` with sparse access, to reproduce what
classical integration of the 31-dimensional system does in `0.2 s`. Route
(b), direct simulation of the dimer on 12 qubits
(`T(||H|| + max|f|) = 102` and `511`, readout 314 calls for the occupation
and 7,754 for the energy), meets the contract against the exact reference
in about `4.9e6` calls with error only from the phonon cutoff (`2.6e-4`
occupation, `9.6e-5` energy). On the `L = 2` dimer there is therefore no
quantum case for integrating the archive equations under either contract;
unmeasured items (circuits, solver constants, readout-model assumptions,
sampling uncertainty at the boundary) are in the report. The decision of
which target the contract is scored against, and whether the quantum thread
continues, is the user's.
