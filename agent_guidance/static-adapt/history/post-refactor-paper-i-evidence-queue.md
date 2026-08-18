# Post-refactor Paper-I evidence queue

## Status

```text
state = ready_for_explicit_run_planning
execution_authorized = false
cleanup_gate = satisfied_2026-07-26
current_priority = campaign_launcher_plan_locked_pending_implementation_authority
planning_contract = agent_guidance/shared/icm-gitnexus-pilot-plan.md
```

This is an agent-facing queue, not a run manifest or hidden authorization to
launch work. The cleanup gate is satisfied, but no study is authorized. Do not
assign, stage, submit, aggregate, plot, or report a study until the user
explicitly authorizes that run phase.

After cleanup, every study follows root `AGENTS.md`, `MATH/AGENTS.md`,
`agent_guidance/skills/paper-i-run/SKILL.md`, and
`agent_guidance/shared/run-guide.md`. Resolve the current visible or explicit
reader-facing target and its source settings before generating commands. Keep
Hamiltonian, cutoff, seed, optimizer, optimizer budget, Pauli-child policy,
backend, and accounting matched except for the named ablation variable.

## Requested studies

### E1. Physical macro pool versus symmetry-retained Pauli-child pool

Directly compare the intact physical macro-generator pool with the
symmetry-retained Pauli-child pool across all six Hubbard--Holstein regimes:

```text
weak-weak
intermediate-weak
strong-weak
weak-strong
intermediate-strong
strong-strong
```

Primary result:

- accepted energy-error trajectory by controller round;
- first target crossing or terminal plateau;
- final same-cutoff energy error;
- accepted generator/child identities and operator-family ancestry;
- matched optimizer evaluations and algorithmic cost.

The comparison must visibly distinguish regimes where intact macros plateau
from regimes where symmetry-retained children reach the requested accuracy
target. Pool exposure is the one intended variable. Do not mix unrelated
route, optimizer, budget, cutoff, seed, pruning, batching, beam, or insertion
changes into this comparison.

### E2. Regime-by-operator-family admission heatmap

Derive one compact heatmap from E1 trajectory/accounting artifacts rather than
launching another large study when the required family ancestry is already
recorded.

```text
rows = six Hubbard--Holstein regimes
columns = stable physical parent-operator families
cell = admitted count or normalized admitted share
```

The heatmap must retain the mapping from an admitted Pauli child to its physical
parent family. Record whether each cell counts admitted records, distinct
accepted coordinates, or normalized share; do not mix these quantities.

### E3. Additional no-Phase-III ablations

Repeat the existing weak--weak no-Phase-III comparison in two additional
representative regimes. The preferred pair is:

```text
strong-weak  = U/t 8.0, lambda 0.25
weak-strong  = U/t 0.25, lambda 1.25
```

This pair isolates a strong Hubbard sector and a strong Holstein sector rather
than adding two redundant intermediate points. Compare Phase III disabled
minus the matched full method. Preserve all non-Phase-III settings.

### E4. Whitened versus unwhitened accepted refits

Run a source-locked one-variable accepted-refit comparison:

```text
variant A = whitened accepted refit
variant B = unwhitened accepted refit
```

Primary outcomes are optimizer evaluations and final same-cutoff energy error;
also retain accepted trajectory, refit success/failure receipts, and numerical
conditioning diagnostics. This is an empirical optimizer/refit ablation, not
permission to change selection, trust, pool, or stopping simultaneously.

### E5. Append-only ADAPT baseline for the existing L=3 study

Add append-only ADAPT to the exact existing `L=3` scientific point and
scalability contract. Resolve the existing `L=3` source artifact before
constructing the comparator. Match the physical pool exposure, Pauli-child
policy, optimizer, optimizer budget, cutoff, seed/reference state, stopping
contract, and accounting semantics. Do not compare the existing scalability
row against a cheaper or differently exposed append baseline.

### E6. Batching and pruning evidence

The typed batching and peer metric/trust pruning policies are implemented and
their composition is validated. When this study is explicitly authorized, run
matched ablations:

```text
baseline                 = batching off, pruning off
batching-only            = batching on, pruning off
pruning-only             = batching off, one named pruning policy
batching-plus-pruning     = one named batching policy, one named pruning policy
```

Keep greedy and combinatorial batching distinct when both are evaluated.
Keep metric and trust-region pruning distinct until evidence selects a default.
Report batch construction/admission counts and realized energy/cost outcomes.
For pruning, report permission-open exposure, eligible deletion trials,
rejected trials, accepted deletions, and true rollbacks separately. Measured
delete-and-refit energy remains deletion authority.

## Cleanup gate closure

Satisfied on 2026-07-26:

- Issues 9--20 reached their definitions of done.
- The default no-prune facade characterization is green.
- The plateau-triggered commutation-insertion route is stable and separately
  identified.
- Ordinary selection, accepted transition, controller, optional-policy,
  resume, and accounting ownership use the direct typed controller rather than
  the legacy mega-controller.
- The contraction audit retains explicit compatibility reachability without
  allowing canonical fallback or deleting historical code.
- The repaired aggregate passed `500` tests with one opt-in provenance audit
  skipped, and fresh independent Standards and Spec reviews returned zero
  findings.

This closure permits planning only. `execution_authorized` remains `false`.

## Post-cleanup planning order

1. Resolve source artifacts, visible/explicit targets, and matched settings.
2. Check whether completed evidence already fills any requested cell.
3. Define run classes and manifests; these studies are not automatically
   paper-facing.
4. Run E1 first and derive E2 from it when possible.
5. Run E3 and E4 as source-locked ablations.
6. Add E5 to the existing `L=3` contract.
7. Run E6 with separately named typed batching and pruning combinations.
8. Aggregate and report without changing manuscript tables or making a
   promotion decision unless the user separately authorizes that work.
