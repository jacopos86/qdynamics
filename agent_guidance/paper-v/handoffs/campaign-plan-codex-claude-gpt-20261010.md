# Campaign plan: three-model token churn on two programs (2026-10-10)

**What this document is.** The anchor for a new planning chat whose job is to
keep Claude, Codex and GPT agents busy in parallel on two programs, with the
roles and mechanics below. It is written to be executed cold: every fact
carries its file or commit, and nothing refers to a prior conversation.
Written from the worktree `Holstein_test.worktrees/claude-paper-v-routes-20260914`
at `b9f99271` (which has `main` at `d1436ab1` merged in). A tracked copy sits
at `agent_guidance/paper-v/handoffs/campaign-plan-codex-claude-gpt-20261010.md`;
this `prompt-exports/` copy is gitignored.

**The user's intent, in their words rearranged into rules.** Throw things at
the wall; use the Codex capacity as well as the Claude capacity; Claude and
Codex agents run concurrently; each program is a job for agents. Roles:

| model | role | tier |
|---|---|---|
| GPT (GPT-Pro handoffs through `prompt-exports/`) | how to integrate it mathematically: the specifics and the high-level theory | GPT-Pro |
| Claude | code architecture mastermind: reads the repo, names candidate implementation sites, specifies seams, reviews | Fable 5.1 for the main loop, every subagent on Opus (`model: "opus"`, user rule of 2026-09-24) with ultracode on |
| Codex | deep research agent that reads the external material, then the implementer that writes the code | the highest reasoning tier the Codex client offers (the user's words: "6.1 Sol Ultra") |

The flow for any increment: GPT states the mathematics; Claude turns it into
an implementation plan with candidate sites and a handoff; Codex implements in
its own worktree and reports back; Claude reviews against the plan.

---

## 1. Mechanics every agent obeys (anchors, not advice)

- Entry contract: `AGENTS.md` at the repository root (Codex reads it directly;
  Claude through `CLAUDE.md`). Paper identity: `agent_guidance/paper-registry.yaml`.
- One worktree per agent, on its own branch, under
  `/Users/lordskyler/Holstein_test.worktrees/`; commits staged with explicit
  paths only, never `git add -A` (incident `ba7f2ac9`, 2026-08-18).
- The handoff format is `agent_guidance/shared/agent-handoff-contract.md`:
  anchors, scope, increments that each end committable, evidence standards,
  traps, and the report-back block. Its test: could an agent with no memory of
  the authoring conversation execute it and produce acceptable work.
- Handoffs live in `prompt-exports/` (gitignored; standalone; substantially
  longer than compact, per `prompt-exports/AGENTS.md`) with a tracked copy
  beside the work when they must survive.
- The mandatory RAM guard is retired (`agent_guidance/shared/memory-budget.md`,
  user revision 2026-09-19): run commands directly, size them by
  `memory_pressure -Q`, do not wrap work in `pipelines/shell/ram_guard.py`, and
  no handoff may reinstate a kill cap (`agent-handoff-contract.md:40`).
- Scientific invariants (`AGENTS.md`, hard invariants): defaults and
  acceptance rules never change silently; every new behaviour is a named
  option with a default-path test; exact and classical references are
  reporting inputs and diagnostics, never online controller inputs; evidence
  adoption, promotion and demotion are the user's decisions; manuscripts are
  never rewritten silently.
- Vocabulary (user rule 2026-09-05): use names that already appear in the
  manuscript or the record; never introduce a name for a quantity; every
  number carries its definition and its horizon in the same breath.
- Responses to the user: lead with the decision-relevant fact, negative result
  first, numbers over narration, end on the next action; status questions get
  two sentences (`memory/answer-in-two-sentences.md`).
- GitNexus impact analysis before editing a function (`CLAUDE.md`); the
  worktrees are unindexed, so a zero from `detect_changes` means unseen.

---

## 2. Program A: molecular quantum chemistry into the Papers I–III methods

**Goal.** Bring an external quantum-chemistry code base's Hamiltonians and
integrals into the three method lanes, the way Paper IV brought the
molecular-vibronic H2O problem to the static-construction method: Paper I
(Resource-Aware ADAPT, `agent_guidance/static-adapt/`), Paper II
(AP-McLachlan dynamics, `agent_guidance/time-dynamics/`), Paper III
(cost-weighted operator-selected QSE, `agent_guidance/qse/`).

**The one anchor still missing.** The individual repository. The GitHub user
is verified as `https://github.com/doronwu` (Codex coordinator, 2026-10-10);
no individual repository under it is selected yet, and no file in this
checkout cites one. **First action of the new
chat: get the URL from the user and record it here.** Until then nothing in
this program can be executed; a Codex research agent given the URL can do
the inventory in Increment A1 unaided.

**The Paper IV precedent, which is the template and the warning.**
`agent_guidance/molecular-vibronic/AGENTS.md`: a static-construction request
resolves the complete physical problem under its own lane, passes it to the
shared method, and takes the result back for validation and evidence
ownership; it must not inherit Paper-I Hubbard–Holstein regimes, source locks
or reporting defaults. The typed `run_ra_adapt` facade rejects non-Hubbard–
Holstein problems, and the contract says that restriction is replaced only by
explicit problem-family admissibility with focused regression tests, never
bypassed through a compatibility route. Program A is the same shape three
times over, so the admissibility seam is the architecture question. Paper IV
state as of 2026-09: `memory/paper-iv-h2o-campaign-state.md` (19-qubit
(3,7,3) fixture, campaign branch, arm flags, quota facts).

**Increments.**

- **A1 (Codex, research).** Inventory the external repository: what
  Hamiltonians it produces (second-quantized electronic structure, which
  basis sets, which active-space tools), what it exports (integrals, Pauli
  strings, which fermion-to-qubit mappings), its license, its dependencies,
  and the smallest molecule it ships that fits a 20-qubit simulation. Report
  as the contract's block; no code written into this checkout.
- **A2 (GPT, theory).** For each lane, the mathematics of the embedding: what
  the lane's method assumes about the Hamiltonian (Paper I: the operator pool
  and the cost model; Paper II: the generalized exchange and the tangent
  metric; Paper III: the operator-selected Krylov space and the cost-weighted
  selection) and what a molecular Hamiltonian changes, with the quantities
  that must be redefined stated in each manuscript's own notation. Handoff
  written by Claude in the GPT-Pro dossier convention
  (`agent_guidance/skills/paper-i-gpt-pro-review/`, read for format only).
- **A3 (Claude, architecture).** Candidate implementation sites for the
  problem-family admissibility seam in each lane, with impact analysis, the
  tests that would pin the existing Hubbard–Holstein behaviour bitwise, and a
  Codex handoff per lane.
- **A4 (Codex, implementation).** One lane at a time, Paper I first because
  Paper IV already exercised that seam; each increment ends with the existing
  lane's regression suite green and a new admissibility test.

**Reserved for the user.** Whether any molecular result becomes paper
evidence; which molecules; whether a fourth manuscript is implied.

---

## 3. Program B: the Paper V archive equations of motion

Two threads: the classical fix (make the closure stay physical where it
diverges) and the quantum integration (can the equations be integrated on a
quantum computer). Everything below is on branch
`claude/paper-v-routes-20260914`, 68 commits ahead of `main` at `b9f99271`,
not merged; the user said on 2026-09-20 that merging does not matter now.

**State, with the numbers that matter** (all at strong coupling lambda 1.5,
gamma 0.5, drive amplitude 1, phonon cutoff 16; scores are the all-coordinate
scalar RMS of the 31 closed scalar coordinates against the exact cutoff-16
driven reference; full ledger in
`MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md`).

| fact | number | where |
|---|---|---|
| exact reference cost at two sites | 3.06 s to t=20; 0.24 s per 0.5 units | `agent_guidance/paper-v/overnight/scaling-exact-cost-law-20260920.md` |
| exact reference at three sites, built and propagated | state dimension 44,217; 19.02 s to t=4 | same |
| cheap repair, exact-correlated start | 0.00929 at t=4 (band to 0.00995), 0.0560 at t=20; 5,183.6 s to t=20 | results summary, section 3 |
| best recorded routes before it | cell A 0.01258 at t=4; both cones 0.1286 at t=20 | same |
| the repair cannot scale | extension rows 36 at two sites, counted 281 at three; conic dense block 74.8(n(2n+1))^2 bytes caps 18 GB at 87 rows; locality does not fix it (one Holstein commutator step demands 84/173/267 off-bond moments at L=3/4/5) | `scaling-dimension-growth-20260920.md`, `scaling-locality-20260920.md` |
| the divergence, reproduced | Hartree–Fock zero-correlation start: electronic positivity lost at t=15.5177447, magnitude 1e4 at t=130.4674620 (13-coordinate form); in 31 coordinates, electronic cone lost at 27.9634, 1e4 at 54.4613 | `divergent-initialization-20260920.md` |
| the repair on the divergent start | electronic one-body density matrix minimum +1.4466e-6 over t=0 to 60, magnitude bounded at 2.1827; neither Gram cone held (−8.5e-4, −9.5e-5) | same, sections 8–12 |
| the divergence is the vector field's | bare closure from the exact trajectory's own t=60 state fires 1e4 at t=433.8964; from the repaired t=60 state at 410.7827; the repair's effect is delivered in the first 6 to 20 time units | same, sections 13–15 |
| Eq. (112), the record's own regularization | refuted on the divergent start: moves the 1e4 event earlier, 54.4613 to 15.3724 | same |
| what limits the cheap repair's accuracy | one coordinate, the relative-mode phonon population over (6, 8], displaced by the positive-extension projection in the Dykstra alternation; eleven repair-side fixes refused with numbers | results summary, sections 4–5 |
| the closure conserves energy | its drive-free velocity moves the Eq. (22) internal energy by 4.4e-17 per step | `cheap-projection-20260915b.md`, closure energy drift section |

**Already written for Codex (2026-10-10), classical thread.** Four standalone
handoffs following the contract, each on its own worktree:

- `prompt-exports/codex-paper-v-repair-early-then-bare-20261010.md`
- `prompt-exports/codex-paper-v-regularized-product-state-20261010.md`
- `prompt-exports/codex-paper-v-electronic-cone-projection-20261010.md`
- `prompt-exports/codex-paper-v-divergence-parameter-map-20261010.md`
- index: `prompt-exports/codex-paper-v-INDEX-20261010.md`; tracked copies under
  `agent_guidance/paper-v/handoffs/`.

(If a file above is missing, the writing agent had not finished when this
plan was written; the specifications are in this plan's own git history at
the commit that added it, and in `agent_guidance/paper-v/handoffs/`.)

**Running now (Claude, Opus, persistent; do not duplicate).**

- Quantum integration feasibility, task 1: the explicit polynomial form of
  the 31-coordinate right-hand side (F_0, A_1, A_2, A_3), the spectrum and
  dissipativity of A_1, the Carleman ratio R for both preparations, and the
  truncation residual at orders 2 to 4 along the exact trajectory. Report:
  `MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md`.
- Closure instability, task 1: the Jacobian spectrum and leading unstable
  mode along the exact-initialized bare closure to its 1e4 event near t=434,
  finite-time Lyapunov exponents, step-size dependence, the same along the
  repaired chain over t=0 to 20, and the condition number of (I − Δt J) with
  the finite-horizon amplification over T = 4, 20, 60. Report:
  `agent_guidance/paper-v/overnight/closure-instability-20261010.md`.

**Quantum thread, increments after task 1 lands.**

- **B-Q2 (GPT, theory).** Given the measured polynomial degree, the spectrum
  of A_1, the Carleman ratio and the amplification factors, which quantum
  ODE algorithms are admissible for these equations and under what
  modification: Carleman lifting (Liu et al. 2021), the fault-tolerant
  linearization with a quantum linear-system solver on implicit steps, the
  Schrödingerization of non-unitary flows (Jin, Liu, Yu), Koopman–von Neumann
  embeddings, and the NISQ residual layer the draft names. The draft's own
  framing is `MATH/paper_details/paper_V_high_u_gkba.tex` sections
  "McLachlan pathway", "Fault-tolerant ideas", "Numerical plan", and block C
  of `MATH/paper_facing/paper_V_high_u_gkba/numerical_plan_draft_20260920.tex`.
  Specifically: the equations conserve the Eq. (22) energy and are not
  dissipative, and their bare flow has a genuine instability; what does each
  algorithm's cost bound become under those two facts.
- **B-Q3 (Claude, architecture).** Where in `paper_5/src/paper5/stability/`
  the quantum-target form lives (the new module the feasibility agent is
  writing), what a block-encoding or a lifted-system builder would need from
  it, and the tests that pin the classical right-hand side as its ground
  truth; a Codex handoff for the lifted-system builder.
- **B-Q4 (Codex, implementation).** The Carleman-lifted linear system at
  declared orders as a classical object with its dimension, sparsity and
  condition number measured along the stored trajectories, so the cost bound
  in B-Q2 is evaluated on real numbers rather than assumed.

**Classical thread, increments after the four Codex handoffs return.**

- **B-C5 (GPT, theory).** Why the closure's vector field is unstable even from
  exact data: the record's candidate causes (`stability_quantum_workflow_working_notes.md`,
  "Leading Eq. (14d) nonlinearity hypothesis", "Separating instability from
  chaos") read against the Jacobian measurements from the instability agent;
  what regularization is justified by the mechanism rather than by the cone.
- **B-C6 (Claude, architecture, then Codex).** Whatever B-C5 names, as a
  named option with its default-path test, measured on the divergent start
  against the rows in `divergent-initialization-20260920.md`.

**Also pending, needs the user.** CHTC segment 2 of the accelerated
carried-witness lane (cluster 10988383, `t=0.5` to `1.0`, submitted
2026-09-20T17:07:09Z) finished long ago and is uncollected; collection and
the next segment need `ssh -MNf ap2001` (Duo) and then the receipt-driven
generator `chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py`.

**Reserved for the user.** Whether the draft blocks in
`numerical_plan_draft_20260920.tex` go into the manuscript (three verified
insertion points, compiles at 7 pages); whether any Paper V number becomes
adopted evidence; whether a separate quantum-integration manuscript is
started (the draft's own plan already carries the encoding slot, so the
cheaper path is the existing manuscript until the quantum thread has a
measured cost bound).

---

## 4. First actions for the new chat, in order

1. Ask the user for the Program A repository URL and write it into section 2.
2. Load `SendMessage` (ToolSearch) and `ListAgents`; the two persistent Claude
   agents above are continued by SendMessage, not respawned.
3. Hand the four Codex classical handoffs to Codex sessions, each in its own
   worktree; collect their report-back blocks into
   `paper_5/notes/electron_phonon_closure_worklog_20260803.md`.
4. When the feasibility agent's task 1 lands, write the B-Q2 GPT dossier and
   the A2 dossier in the same sitting; while GPT works, run A3 and B-Q3 as
   Claude architecture tasks.
5. Keep one reviewer agent (Opus, persistent) that verifies every returned
   number against its artifact before it enters the worklog; the record's
   conventions audit (`agent_guidance/paper-v/overnight/conventions-audit-20260915.md`)
   lists the traps.
6. Do not start the L-site closure or any repair-side lever already refused
   (results summary section 4 lists eleven); the counted dimensions say the
   corrected closure is a two-site demonstration.
