# paper-i-hh-sr-macro-beam3x2-fsprune-symcost-six-r50-20260719-v4

> **SUPERSEDED — DO NOT SUBMIT.** Cluster `8894497` exposed a
> non-scientific exact-prune estimator-consumer identity collision across beam
> parents. Use the immutable `v4` sibling, which preserves the route digest and
> changes only parent-branch scoping of prune-trial consumer IDs.

Six fresh round-0 to round-50 Paper-I Hubbard--Holstein SR-SNAKE jobs.

- Macro-only intact logical parent candidates with physical lanes.
- Historical beam: 3 live parents x 2 admission children, at most 6 continuations per round.
- Live-only undamped full-logical Fubini--Study trust pruning; measured delete/refit acceptance.
- Cost policy: `family_robust_symmetric_arctan_v1`.
- Ordinary Phase-II/III novelty multipliers off; all-infeasible fallback retained with telemetry.
- Weak-Holstein cutoff `n_ph=3`; strong-Holstein cutoff `n_ph=7`; same cutoff references.
- Exact horizon: 50 controller rounds for every regime.
- Route digest: `a05ecc8b709db8beac9115d9d0ca39f4faf09e1cbaa10e57bdd674abef9215f0`.
- Source archive SHA-256: `b968d3781d6c37001f78239e844d5eb9ac2a67f91bfef91c69dcb04c0b3a1720`.
- Supersedes non-scientific post-run fidelity-validation predecessor cluster
  `8893083` without changing scientific settings.

## v4 non-scientific repair

Derived only from the immutable v3 source archive. Exact prune-trial estimator consumer IDs now include the parent beam branch ID, preventing cross-parent ledger aliasing. Route settings and digest are unchanged.
