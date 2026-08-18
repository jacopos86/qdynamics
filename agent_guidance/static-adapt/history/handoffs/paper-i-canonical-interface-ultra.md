# Paper-I Canonical Interface Refactor Handoff

Intended execution model: GPT-5.6-sol ultra.

## Objective

Implement the complete Paper-I canonical interface refactor. An ordinary agent
must be able to request a canonical Hubbard--Holstein SR-SNAKE run through one
small typed interface without seeing or resolving historical routes, dormant
policy settings, artifact paths, or reporting mechanics. Preserve explicit
compatibility/replay access without allowing it into canonical resolution.

This is an implementation task, not another audit or planning exercise.
Complete ordinary in-scope repairs, tests, and documentation needed to reach the
outcome.

## Decisions already made

- Use the explicit decisions in
  `agent_guidance/paper-lane-refactor-plan.md`, then the locked provenance at
  `MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/paper_i_hh_macro_common_accuracy_20260723_provenance.json`.
  Existing Markdown, unqualified aliases, and registry defaults are not
  authority when they conflict.
- The provenance baseline is accepted wholesale. Do not re-interview the user
  about its solver, optimizer, seed, trust, refit, shortlist, cost, or numerical
  fields.
- Canonical scope is Hubbard--Holstein, `L=2`, SR-SNAKE, unfiltered
  `full_meta` with HVA included, physical macro lanes followed by
  exact-cardinality-one symmetry-retained Pauli children, and a mandatory hard
  sector/padding guard.
- Preserve full active-plus-singleton Phase-III response, source-metric
  no-endpoint-overlap trust, and whitened complete accepted refits.
- Promote plateau-triggered commutation-reduced insertion to the silent
  canonical policy. Append-only insertion remains an explicit ablation/replay
  identity; do not relabel old evidence.
- Batching, pruning, and beam are off and absent from the ordinary request.
  Enabling batching chooses greedy or combinatorial. Enabling pruning chooses
  metric or trust-region pruning as peers; both use measured
  delete-and-complete-refit acceptance. Enabling beam uses the settled
  fork-local energy plus `S_alg` comparison, with suggested `3` live branches,
  `2` children per parent, and uncalibrated `0.01` work weight.
- Phase-live/retirement hysteresis and unchanged-parent beam survival are not
  canonical options.
- Maximum controller rounds default to `50`; an optional predefined
  same-problem, same-cutoff exact-ED target is checked only after accepted full
  refits.
- Unwhitened accepted refitting is a one-off experiment, not a permanent
  interface, route, adapter, or architecture seam.
- Append-ADAPT is hidden from the ordinary interface. Existing append artifacts
  are frozen comparator/replay sources; explicitly named comparator studies may
  still invoke append. Geo-ADAPT is a hidden explicitly named Paper-I
  benchmark. JR-SNAKE, FM-SNAKE, Route-A/B/C, versioned SR profiles, H2O
  routes, and other historical implementations are explicit compatibility or
  lane-owned research paths only.
- A completed or deliberately stopped accepted run automatically emits the
  accepted energy-error trace, effective-plateau Qiskit resources,
  append-matched common-accuracy resources, canonical `S_alg`, provenance, and
  requested-round resources. Qiskit remains observational and outside the
  numerical controller.

## Authority to implement and repair

Read root `AGENTS.md`, `MATH/AGENTS.md`,
`agent_guidance/paper-lane-refactor-plan.md`,
`agent_guidance/static-adapt/CONTEXT.md`, the governing portions of
`agent_guidance/static-adapt/history/sr-snake-refactor-plan.md`, the Issue-7 provenance
anchor, and the exact route-specific code/tests. Use the `codebase-design`,
`domain-modeling`, `tdd`, and `code-review` skills. Do not use `grilling`; the
scientific and interface decisions are settled.

You may refactor production code, repair missing plumbing, replace stale
internal callers, add typed contracts/adapters, update tests, and create the
thin Paper-I agent-control documents. Preserve unrelated dirty work. Do not
commit, push, mutate issues, launch scientific runs, edit manuscripts, promote
evidence, or delete compatibility code.

## Constraints

- Keep `run_sr_snake(problem, request=None)` as the ordinary two-argument seam.
- Prefer one deep canonical resolver/controller over wrappers that merely
  forward the legacy flag union.
- Disabled policies must not serialize or expose dormant subtype settings.
- Canonical resolution must not enumerate, auto-detect, infer, retry, or fall
  back to compatibility routes.
- Historical manifests and replay paths retain their exact identities.
- Preserve the Issue-7 characterized append-only trajectory under its original
  explicit route; add new canonical characterization for the intentional
  insertion change.
- Preserve estimator-ledger closure and canonical executed-work `S_alg`.
- Design for successful execution: add hard failures only for concrete
  scientific, provenance, or logical invariants.

## Definition of done

1. The typed facade resolves exactly one canonical provenance-derived preset
   plus only explicit typed overrides.
2. Plateau-triggered commutation-reduced insertion is the actual default and is
   protected by public behavior tests.
3. Singleton, greedy/combinatorial batching, peer pruning, optional beam,
   stopping, resume, observation, and accounting compose without importing
   legacy control flow into the canonical resolver.
4. Compatibility methods remain explicitly runnable where already supported
   but are unreachable from an ordinary Paper-I request.
5. One deep run-summary module owns automatic and requested-prefix reporting;
   campaign/PDF builders consume it rather than duplicating selectors.
6. `agent_guidance/static-adapt/AGENTS.md`, `run-guide.md`, conditional
   `policies/*.md`, and `reporting/run-summary.md` reflect implemented behavior
   and use shallow if-then routing.
7. Route-faithful focused tests and the relevant broader regression suite pass.
   Run a path-limited reachability audit and independent Standards and Spec
   reviews, repairing findings before handoff.

Return a concise report listing behavior delivered, compatibility boundaries,
files changed, validation commands/results, and any genuinely deferred work.

## True stop conditions

Stop only if verified executable/provenance evidence creates a new scientific
contradiction not resolved above, required inputs are irrecoverably absent, or
completion would require a destructive/external action outside this authority.
Name the exact evidence and ask one focused question. Ordinary implementation,
serialization, test, documentation, or adapter problems are repair-and-continue
work.
