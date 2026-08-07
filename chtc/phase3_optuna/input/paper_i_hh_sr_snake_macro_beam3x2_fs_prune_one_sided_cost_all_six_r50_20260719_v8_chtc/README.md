# paper-i-hh-sr-macro-beam3x2-fsprune-onesided-six-r50-20260719-v8

Six fresh round-0 to round-50 Paper-I Hubbard--Holstein SR-SNAKE jobs.

- Macro-only intact logical parent candidates with physical lanes.
- Historical beam: 3 live parents x 2 admission children, at most 6 continuations per round.
- Live-only undamped full-logical Fubini--Study trust pruning; measured delete/refit acceptance.
- Cost policy: `family_robust_penalty_only_v1`.
- Ordinary Phase-II/III novelty multipliers off; all-infeasible fallback retained with telemetry.
- Weak-Holstein cutoff `n_ph=3`; strong-Holstein cutoff `n_ph=7`; same-cutoff references.
- Exact horizon: 50 controller rounds for every regime.
- Route digest: `e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096`.
- Source archive SHA-256: `4c40399410b67b34a89f3cadeae59a0fd901c39132ff5cc746101c78e5acccd7`.

## v6 non-scientific repair

Derived directly from the immutable v3 source archive. Exact prune-trial consumer IDs include the parent beam branch; beam history serializes and validates the existing physical-lane receipt; and validation separates the 50-round controller/frontier from the selected terminal winner while cross-checking the checkpoint and estimator-receipt graph. Route settings, selection, terminal archive policy, and digest are unchanged.

## v8 validator/reporting repair

The scientific source archive, route digest, settings, and command semantics are byte-for-byte/scientifically unchanged. The post-run validator now reads all route-resolved source-only settings from the immutable normalized command and route contract because those fields are not flattened into result.settings. Every-round full-response validation remains mandatory. revalidate_v6_archive.py can validate and report completed v6 payloads without modifying their raw archives.
