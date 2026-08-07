# Molecular-Vibronic Agent Contract

Scope: Paper IV, molecular-vibronic water, finite active-space H2O Hamiltonian
construction, derivative/alignment/cutoff diagnostics, and static benchmark
framing.

## Static-construction route

Paper IV and Paper I are peer static-construction producer lanes. For a
Paper-IV static ADAPT/SNAKE request:

1. resolve the complete molecular-vibronic physical problem under this lane;
2. pass that resolved problem to the shared static-construction method;
3. return the result to Paper IV for validation, reporting, provenance, and
   evidence ownership.

Do not inherit Paper-I Hubbard--Holstein regimes, `L=2` provenance, source
locks, append comparators, or Paper-I reporting defaults. Method policies may
be shared only where the molecular-vibronic problem family explicitly supports
them.

The current typed `run_ra_adapt` facade rejects non-Hubbard--Holstein problems.
Paper-IV execution through that seam remains deferred until the restriction is
replaced by explicit problem-family admissibility and focused regression tests.
Do not bypass it through a compatibility route.

Follow this authority chain:

1. Root `AGENTS.md`.
2. `MATH/AGENTS.md` for paper-program policy.
3. `MATH/paper_facing/paper_IV_molecular_vibronic_h2o/` for Paper-IV support
   notes.
4. `MATH/paper_details/molecular_vibronic_h2o_paper_IV.tex` only when the task
   explicitly requires Paper-IV manuscript text or table/figure context.
5. `agent_guidance/skills/paper-i-run/SKILL.md` only when Paper-IV work invokes
   an actual static SNAKE/ADAPT run or run-evidence workflow. No Paper-I results
   skill exists; explicit evidence transfer follows the root/MATH and target
   support contracts.
6. Existing Paper-II run/results skills only when Paper-IV dynamics evidence is
   explicitly in scope. No Paper-III run/results skills exist in this checkout;
   Paper-III paper-facing execution or evidence transfer fails closed when the
   lane contracts are insufficient.

Paper IV does not yet have a dedicated run or results skill. Do not create one,
launch application runs, or promote evidence unless the current user explicitly
asks for that work and the underlying method gate has been followed.
