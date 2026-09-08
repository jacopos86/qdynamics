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
commutation.py        generator-block certificates and complete insertion-position classes
schur_identity.py     exact conditional capture and deletion-loss specialization
deletion_geometry.py  effective-angle and finite ray-distance bounds for deletion blocks
deletion_permission.py cached permission decisions, thresholds, and summary reporting
support_patch.py      retained payload types and array-level scoring/diagnostic APIs
exchange_structural.py generalized-exchange enumeration and historical acquisition utilities
compiled_patch_cost.py complete-patch compiled resource counts and marginals
population_scoring.py population normalization and ranking arithmetic
exchange_integration.py route-specific population ranker and AP realization adapter
exchange_selector.py  ranked finalist certification and atomic support commit
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

Generalized exchange uses the historical scalar utility to order singleton
acquisition and seal insertion frontiers, including on compiled-cost routes.
The route-specific population ranker in `exchange_integration.py` then ranks
complete patches using the acquired candidate population. The retired
parent/macro scout is no longer an executable surface.

The mathematical interfaces are directly importable:

```python
from pipelines.time_dynamics.ap_mclachlan.commutation import commutation_reduce
from pipelines.time_dynamics.ap_mclachlan.deletion_geometry import (
    effective_deletion_angle, deletion_ray_bound,
)
from pipelines.time_dynamics.ap_mclachlan.schur_identity import conditional_capture

position_classes = commutation_reduce(a, c)  # all placements in each class
angle = effective_deletion_angle(theta, coefficients, D)
ray_upper = deletion_ray_bound(angle)
capture_added = conditional_capture(K, f, retained=R, block=A)
capture_deleted = conditional_capture(K, f, retained=R, block=D)
exact_exchange_capture = capture_added - capture_deleted
```

Here `K` and `f` must describe all named blocks on the same frozen reference
ray; the insertion cross terms must already be available. Conditional capture
uses complete blocks, including empty and singleton limits. It exposes the raw
Schur quadratic, while `exact_deletion_loss` preserves the existing permission
policy's nonnegative clipping. The live ranker continues to use the complete
patched safeguarded solve. Neither this exact reference nor the angle bound
certifies a post-refit candidate.

`commutation_reduce` describes one inserted generator, possibly a multi-Pauli
block, against the current ordered ansatz. It returns every placement and does
not pick the compiled-cost minimum. Simultaneous insertions retain the distinct
whole-word quotient in `insertion_words.py`; their geometry and compiled cost
cannot be assembled by summing independently optimized singleton values.
These interfaces preserve the existing frontier, batch escalation, placement,
and ranking policies. Their docstrings carry LaTeX equations independently of
any paper-to-code viewer.

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
