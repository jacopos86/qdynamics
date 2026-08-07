# APM Package Contract

This package owns the fresh Paper-II APM implementation. APM is the short name
for append-prune McLachlan, previously written AP-McLachlan. Use `APM` in new
agent-facing docs, progress reports, and chat updates; keep `ap_mclachlan` in
package paths until a deliberate code rename is planned.

## Terms

- Use `time point`, `time step`, or `time iteration` for the discrete dynamics
  index `k`.
- Use `support patch` for the controller action
  `nu_k = (B_k^-, R_k^+)`.
- Use `no edit`, `append`, `delete`, and `exchange` as the active support-patch
  cases. `append` means zero-angle coordinates are added at the tail of the
  ANZATS/runtime coordinate list. Treat old `insert` payloads as legacy
  compatibility records only; reserve future `insert` language for a deliberate
  internal placement operation inside the ANZATS.

## Implementation Direction

New Paper-II APM controller work should move toward one support-patch scorer and
one support-patch action-selection surface:

```text
state.py             neutral ScaffoldRuntimeInput -> APM runtime manifold adapter
hamiltonian.py       neutral H(t)=H_static+c(t)D provider via drive_terms adapter
inverse.py            shared supported inverse, ridge, retained eigenspace, Gamma
geometry.py           validated K, f, ||b||^2 geometry payloads
geometry_eval.py      statevector geometry evaluator for active runtime support
fixed_step.py         Eq. (8) fixed-support McLachlan solve
integrators.py        Euler/RK4 parameter integration methods
trajectory.py         fixed-support APM trajectory propagation
support_patch.py      pure array-level APM support-patch scoring
support_frontier.py   non-authoritative parent/frontier filtering before child support scoring
../runners/ap_fixed_from_adapt_artifact.py
                     JSON-producing fixed-support artifact smoke/runner
```

Build order for the fresh route is:

```text
state/hamiltonian -> inverse -> geometry -> geometry_eval -> fixed_step -> integrators -> trajectory -> support_patch -> controller
```

`state.py` is the only APM-facing seed adapter. It consumes the shared scaffold
runtime contract produced by ADAPT, SNAKE, Geo-ADAPT, or other static scaffold
routes. Do not parse artifact JSON directly inside controller logic.

`hamiltonian.py` is the only APM-facing time-dependent Hamiltonian provider.
Static Hamiltonians come from the resolved scaffold problem; driven terms come
from `pipelines.time_dynamics.adapters.drive_terms.resolve_realtime_drive_model`.
Do not add Hubbard-Holstein-specific Hamiltonian construction inside APM core.

Keep integration separate from the Eq. (8) solve. The solve produces
``theta_dot = K^+ f`` on a support; `integrators.py` decides how theta advances
from that derivative.

Default APM smoke runs use Euler integration with ridge `1.0e-7`. RK4 is a
diagnostic/accuracy option, not the default integrator. Runs that intentionally
test the unregularized solve must pass ridge `0.0` explicitly.

Do not add a new repair lane. Exchange is the support-patch case with both
deleted and appended blocks nonempty.

The optional parent/frontier scout is not a new support-patch lane. In
`per_pauli_term` mode it may construct a temporary parent tangent to rank
parents, but retained parents must be expanded back into Pauli/poly-child atoms
before append, delete, or exchange finalists are scored. Cheap modes
`parent_tangent_schur_gain` and `parent_linear_residual_v1` are candidate
diagnostic search-budget routes; `full_child_block_diagnostic` is explicitly
non-cheap diagnostic telemetry. Macro parents become committed support atoms
only in an explicit `logical_shared` route.

When exchange candidates are requested, `append_macro_scout_exchange_fail_open`
is the canonical default. A parent scout scored against the current support can
miss an append child that becomes novel after a deletion, so canonical exchange
runs preserve the full child append frontier and record
`exchange_fail_open_frontier_preserved`. Disabling this guard is
diagnostic-only and must be labeled as uncertified/noncanonical telemetry; do
not add deletion-conditioned exchange-aware parent filtering unless that
diagnostic extension is explicitly requested.

Reference trajectories are post-run reporting inputs only. Support-patch
scoring, action selection, integration, and online tuning must not import or
query reference trajectory objects.

## Reporting Contract

The weak-weak progress artifact should be called the Results pdf in agent-facing
work:

```text
output/pdf/ap_mclachlan_weak_weak_snake_progress_diagnostic.pdf
```

For that report, Qiskit cost means the compiled cost of the final active APM
ansatz at the trajectory terminal time. It is a structural final-support report
field, not a Qiskit comparator trajectory, not full-horizon repeated cost, and
not controller input. Any new completed Results-pdf row should have a
machine-readable cost sidecar or manifest entry with `N2q`, `D2q`, `Dc`,
backend/transpiler settings, and a final-support digest before the PDF update is
considered complete.

## Migration Rule

New fields, telemetry, docs, and tests should use APM and support-patch language
unless compatibility with existing AP-McLachlan artifact fields is required.
