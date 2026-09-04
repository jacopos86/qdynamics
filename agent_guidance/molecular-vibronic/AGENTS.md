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

The typed `run_ra_adapt` facade admits the molecular-vibronic H2O problem only
through its named application lane,
`pipelines/static_adapt/ra_adapt/h2o_application.py`. Canonical accepted-state
resume admits it (2026-09-03) through a third named branch beside canonical
Hubbard--Holstein and the named pure-Hubbard controller-noise application:
`paper_iv_h2o_linear_fd_application_admitted` in
`pipelines/static_adapt/sr_snake/_resume.py`, consumed by
`_validate_problem_binding` there. The branch is entered only when the resolved
problem is the `molecular_vibronic_h2o_linear_fd` family requested under its own
problem key *and* the authenticated route contract identifies an H2O RA
application lane; it then requires the linear-FD fixture extension and no other
problem extension.

Problem identity is the fixture bytes, never a path. Every H2O checkpoint
carries `problem_attestation`
(`paper_iv_h2o_linear_fd_fixture_attestation_v1`: fixture SHA-256 and byte
count, fixture manifest metadata, active-space size, particle number, mode
labels and cutoffs, and the fermion/boson/total qubit counts), and resume
refuses any disagreement with the freshly resolved problem. The fixture path
string is deliberately excluded, so the same bytes resume from the CHTC sandbox
path or a local checkout path. Route settings are pinned from
`canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()` rather than
copied, and the two candidate arms and the two `--cost-mode` variants are kept
apart by the route-contract digest, so neither can resume into the other.

The Hubbard--Holstein and pure-Hubbard admission conditions, their pinned route
settings, and their route-mismatch message are unchanged. Their path gained
exactly two fail-closed refusals, for envelopes those families never write: a
checkpoint whose declared `settings.problem` is not the resolved problem key,
and a non-H2O checkpoint carrying a `problem_attestation`. Do not bypass any
branch through a compatibility route.

Regressions: `test/test_paper_iv_h2o_resume_admissibility.py` (the predicate,
attestation, and cross-family refusals run in the default suite; the tests that
execute production rounds are opt-in through `PAPER_IV_H2O_CLI_RUNS=1`) and the
CLI segment chain in `test/test_ra_adapt_h2o_application.py`.

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
