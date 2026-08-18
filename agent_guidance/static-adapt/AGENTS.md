# Paper-I Static-ADAPT Agent Router

This file is the Paper-I lane router. Root `AGENTS.md` and `MATH/AGENTS.md`
remain higher authority.

Paper IV may eventually invoke the same static-construction method after its
own lane resolves a molecular-vibronic physical problem. That does not make
Paper IV a Paper-I route: the Hubbard--Holstein defaults and Paper-I evidence
contracts below remain Paper-I-owned.

## Ordinary RA-ADAPT work

For a canonical Hubbard--Holstein RA-ADAPT request:

1. Read `agent_guidance/static-adapt/run-guide.md`.
2. Use the public seam
   `pipelines.static_adapt.ra_adapt.run_ra_adapt(problem, request=None)`.
3. Read a conditional policy file only when the user names that policy.
4. Read `agent_guidance/static-adapt/reporting/run-summary.md` only for run
   completion, accepted-prefix resources, or a later summary query.

Do not inspect route registries, historical handoffs, artifact trees, CLI
flags, or manuscript sources to resolve an ordinary request.

## Conditional routing

| User intent | Read next |
|---|---|
| greedy or combinatorial batching | `agent_guidance/static-adapt/policies/batching.md` |
| metric or trust-region pruning | `agent_guidance/static-adapt/policies/pruning.md` |
| beam or multiple accepted lineages | `agent_guidance/static-adapt/policies/beam.md` |
| accepted-state checkpoint resume | `agent_guidance/static-adapt/policies/resume.md` |
| insertion or append-only ablation | `agent_guidance/static-adapt/policies/insertion.md` |
| round limit or exact-ED stop | `agent_guidance/static-adapt/policies/stopping.md` |
| summary, plateau, common accuracy, Qiskit resources, or `S_alg` | `agent_guidance/static-adapt/reporting/run-summary.md` |
| Page-12 global-singleton gradient-Phase-0 candidate | `agent_guidance/static-adapt/page12-global-singleton-gradient-phase0-route-20260810.md` |

If the request plans, launches, monitors, repairs, aggregates, or reports an
actual benchmark/evidence run, also follow the Paper-I run-skill gate from root
`AGENTS.md` and `MATH/AGENTS.md`. This lane router does not bypass run,
evidence, manuscript, or promotion gates.

## Compatibility quarantine

`agent_guidance/static-adapt/history/` holds every quarantined compatibility
and provenance document for this lane (relocated 2026-08-17 with user
authorization; contents unchanged — see `history/README.md`). The following
are not ordinary navigation surfaces:

- `agent_guidance/static-adapt/history/route-identities.md`;
- `agent_guidance/static-adapt/history/route-a-language.md`;
- `agent_guidance/static-adapt/history/sr-snake-refactor-plan.md`;
- `agent_guidance/static-adapt/history/handoffs/`;
- numbered `sr-snake-issue-*-handoff.md` files;
- `agent_guidance/static-adapt/history/paper-i-sr-snake-current-run-map.md`;
- `agent_guidance/static-adapt/history/post-refactor-paper-i-evidence-queue.md`.

Read one of those files only when an explicit compatibility, replay,
implementation-history, or provenance task names the corresponding identity or
artifact. Use the exact supplied identity. Never enumerate, infer, retry, or
fall back across compatibility routes from a canonical request.

Compatibility and provenance files remain physically preserved. Do not move,
delete, normalize, relabel, or rewrite their identities as part of ordinary
canonical work.

The compact inert-archive provenance pointer is
`archive/paper_i_static_adapt_legacy_20260727/MANIFEST.json`. Read it only for
an explicit retirement, compatibility, or provenance audit; the archive is
not an execution surface.

## Lane invariants

- The ordinary interface is exactly `run_ra_adapt(problem, request=None)`.
- Canonical resolution is Hubbard--Holstein, `L=2`, and RA-ADAPT.
- Qiskit, exact-reference diagnostics, and reporting cannot affect controller
  decisions.
- Canonical executed-work accounting is the closed occurrence receipt
  `S_alg = N_H_outer + N_H_refit + N_grad + N_metric`.
- Append-ADAPT and Geo-ADAPT are separate explicitly named comparators.
  JR-SNAKE, FM-SNAKE, Route-A/B/C, H2O routes, and versioned historical
  profiles are compatibility or lane-owned research paths.
- A fixed-regime RA-ADAPT/Geo-ADAPT/Append-ADAPT comparison uses the same inner
  SPSA budget and the same Pauli-child policy across methods. A mixed budget or
  mixed child policy is an explicitly named diagnostic/ablation only.
- Manuscript changes are never implied by algorithm or agent-control edits.
