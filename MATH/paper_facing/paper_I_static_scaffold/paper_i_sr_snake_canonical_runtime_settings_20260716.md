# Conventional Paper-I SR-SNAKE v3.1 Runtime Identity

Date: 2026-07-16
Scope: agent-facing method identity, exact rerun routing, and provenance. This
document does not edit, reinterpret, or promote a manuscript result.

## Stable identity

The unqualified conventional method name `SR-SNAKE` resolves to:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3_1
request = --sr-route-profile sr_snake
versioned_request = --sr-route-profile sr_snake_v3_1
phase3_response_coordinate_scope = full_active_plus_singleton_v1
phase_live_hysteresis_enabled = false
```

Both requests materialize the same executable contract from
`pipelines/static_adapt/sr_snake_route_profile.py`. Its SHA-256 is:

```text
9b96179935ed80967a3335dfbbf8eece86a04c2d412e6b92aa8a466fa6913542
```

The contract and digest must be preserved in command, manifest, checkpoint,
resume, and result records. Agents must not reconstruct this profile from
current defaults.

## Phase-III response-coordinate contract

On every controller round, each Phase-III singleton candidate is evaluated in
the complete pre-admission active-logical-ansatz-plus-candidate response space.
The hashed v3.1 execution contract requires
`phase_live_hysteresis_enabled=false`; Phase III may not be retired into a
Phase-II-only admission path. Current full-response candidate profiles must
also record the disabled setting explicitly and fail closed if it is absent or
true.
Before supported-rank reduction:

```text
phase3_response_coordinate_count
  = active_logical_coordinate_count + 1
```

The recorded response indices therefore contain every active logical index in
its ordered pre-admission position plus the candidate insertion index.
Supported-rank projection may subsequently remove genuine Fubini--Study
Gram-null modes; it may not exclude coordinates merely because they are outside
an optimizer window.

The Phase-III response-coordinate scope is independent of:

- `adapt_reopt_policy`;
- `adapt_window_size`;
- `adapt_window_topk`;
- `adapt_full_refit_every`;
- periodic-full-refit triggers;
- terminal refit triggers.

Every round records the response indices, pre-support coordinate count,
supported rank, and accepted-refit coordinate count. A canonical v3.1 invocation
with a missing, contradictory, or legacy-coupled response scope fails before
scientific execution.

## Preserved v3 controller fields

V3.1 changes only the phase-live hysteresis setting relative to the frozen v3
route. V3 itself changed only the Phase-III response-coordinate scope relative
to historical conventional v2. V3.1 therefore preserves:

- Hubbard--Holstein `full_meta` pool with HVA operators included;
- Phase 0 off and Phase-II/III batching off;
- exact-cardinality-one Pauli-child forwarding;
- fixed-sector and binary-padding hard enforcement before scoring;
- Phase-II behavior unchanged;
- Phase-III `supported_metric_whitened_eigh_v1` supported solve;
- adaptive trust policy `displacement_calibrated_unbounded_v2`;
- Powell base chart `expanded_runtime_projected_logical_v1`;
- full accepted-ansatz refit in
  `supported_fs_whitened_fixed_v1` coordinates;
- singleton admission and repeat-generator allowance;
- novelty, damping, beam, pruning, cost, shortlist, optimizer-budget, and
  stopping settings from the v2 execution contract.

Regime physics remains outside the method profile. Each run must source-lock
its Hamiltonian parameters, working cutoff, same-cutoff reference, and horizon.

## Historical replay profiles

Frozen conventional v3 remains explicitly selectable only as:

```text
request = --sr-route-profile sr_snake_v3
route_profile = supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3
phase3_response_coordinate_scope = full_active_plus_singleton_v1
contract_sha256 = 435910592e88f0136a0d45f611f79fe96b21d75fd25bad58276c871f39dc080e
```

That digest predates serialization of the phase-live hysteresis field. It is
retained only to replay existing artifacts together with their recorded
effective runtime settings; it is no longer the unqualified conventional
route. Existing v3 results, FM shadow contracts, and their digest remain
unchanged.

Historical conventional v2 remains explicitly selectable only as:

```text
request = --sr-route-profile sr_snake_v2
route_profile = supported_whitened_adaptive_trust_full_accepted_refit_v2
phase3_response_coordinate_scope = legacy_reopt_coupled_v1
contract_sha256 = 32d2bdf2b05818be6f4add74137447a313605d7ed35ffb880651863b793a0f64
```

Its Phase-III response coordinates are coupled to the reoptimization-window
and periodic-refit state. That is a historical replay contract, not the
unqualified conventional route. Existing v2 results and their digest remain
unchanged.

Historical v1 remains:

```text
request = --sr-route-profile sr_snake_v1
route_profile = supported_whitened_adaptive_trust_v1
contract_sha256 = fab7b5a6c4bd2ab019139367aa2a507356a5c969b6b88cd72d32365ae766e13e
```

The explicit local-window ablation scope is
`fixed_local_window_v1`. Neither that policy nor
`legacy_reopt_coupled_v1` may execute under the canonical v3.1 route identity.

## No-drift rule

Future unqualified `SR-SNAKE` invocations must use `sr_snake` or
`sr_snake_v3_1`, preserve the resolved response scope, disabled hysteresis,
and route digest across all
serialization boundaries, and enforce the full response-count invariant before
Gram support reduction. No Powell/refit scheduling field may silently change
the Phase-III response indices.

This identity registration authorizes no scientific run, CHTC submission,
Paper-I manuscript edit, result rewrite, result promotion, or PDF regeneration.
