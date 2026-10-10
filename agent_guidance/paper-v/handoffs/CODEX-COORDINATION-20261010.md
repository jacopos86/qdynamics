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
