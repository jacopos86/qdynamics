# HH Static Sweep Current-Route Regression Audit

- Date: 2026-04-11
- Scope: current checkout, `L=2`, HH, static ground-state sweep
- Physics point(s): `t=1`, `U=0`, `omega0=0.5`, `dv=1`, `n_ph_max=1`
- Sweep values completed so far:
  - `lambda=0.0`, `g=0.0`
  - `lambda=0.2`, `g=0.2236068`
  - `lambda=0.5`, `g=0.3535534`
- Objective: identify and fix the current-route code regression blocking the Stage 1 legacy-vs-current sweep

## Issue statement

The current static ADAPT route is failing before ADAPT execution begins on the new Stage 1 sweep. This is a code regression, not a scientific outcome.

The failing error is:

`name '_compile_polynomial_action_shared' is not defined`

This occurs on the current route for at least:

- `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/cases/current_lam0p0/logs/stdout.log`
- `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/cases/current_lam0p2/logs/stdout.log`
- `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/cases/current_lam0p5/logs/stdout.log`

## Important scoping note

Do **not** conflate this with the legacy-route behavior on the same sweep.

Two separate facts are true:

1. **Current route has a real code bug**
   - it crashes immediately with `_compile_polynomial_action_shared` undefined
   - this blocks scientific comparison

2. **Legacy route is not crashing, but is performing badly on this new physics point**
   - that is a separate scientific/route-validity question
   - it should not be treated as evidence that the current crash is acceptable or expected

This audit is only about the **current-route code regression**.

## Reproduction

The current-route sweep command shape is in:

- `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/cases/current_lam0p0/logs/command.sh`

The crash occurs after:

- Hamiltonian build
- pool build
- phase-3 registry ready

and before:

- compiled cache ready
- any meaningful ADAPT iteration

The failure log fragment is:

- `AI_LOG {"event": "hardcoded_adapt_phase3_registry_ready", ...}`
- `AI_LOG {"L": 2, "error": "name '_compile_polynomial_action_shared' is not defined", "event": "hardcoded_adapt_vqe_failed", ...}`

## Root cause seam

In:

- `pipelines/static_adapt/adapt_pipeline.py`

the local helper

```python
def _compile_polynomial_action(
    poly: Any,
    tol: float = 1e-15,
    *,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction:
    terms = poly.return_polynomial()
    if not terms:
        return CompiledPolynomialAction(nq=0, terms=tuple())
    return _compile_polynomial_action_shared(
        poly,
        tol=float(tol),
        pauli_action_cache=pauli_action_cache,
    )
```

calls `_compile_polynomial_action_shared(...)` directly.

But the imports in the same file only bring in:

```python
from pipelines.static_adapt.statevector_runtime import (
    _apply_compiled_pauli,
    _apply_compiled_polynomial,
    _apply_exp_term,
    _compile_pauli_action,
    _compile_polynomial_action,
    ...
)
```

So `_compile_polynomial_action_shared` is not defined in this module namespace.

Relevant supporting file:

- `pipelines/static_adapt/statevector_runtime.py`

This file *does* import:

```python
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    compile_polynomial_action as _compile_polynomial_action_shared,
)
```

So the shared symbol exists, but `adapt_pipeline.py` is not importing it.

## Likely regression interpretation

This looks like a refactor seam where:

- `adapt_pipeline.py` kept an older local wrapper body that still expects `_compile_polynomial_action_shared`
- while the module import surface was changed to import the wrapper helper from `statevector_runtime` instead of the shared symbol directly

In short:

- the helper body and the import surface are out of sync

## Expected fix shape

The refactor agent should make the current route internally consistent again.

The most likely correct fix is one of these:

1. **Preferred:** update `adapt_pipeline.py` so its local helper delegates to the imported wrapper helper consistently, without referencing the missing shared symbol directly
2. or import `_compile_polynomial_action_shared` explicitly into `adapt_pipeline.py` if that is the intended architecture
3. or remove the duplicate local helper entirely if `statevector_runtime._compile_polynomial_action(...)` is now the canonical wrapper

The agent should choose the option that best matches the current static-adapt architecture, but the resulting code must be namespace-consistent and must not leave duplicate compile paths drifting apart again.

## Acceptance criteria

The fix is accepted only if all of the following hold:

1. The current route no longer crashes on:
   - `lambda=0.0`
   - `lambda=0.2`
   - `lambda=0.5`

2. These cases produce:
   - `result.json`
   - `compile_scout_fake_marrakesh.json`

3. The run gets past:
   - pool build
   - phase-3 registry ready
   - compiled cache setup
   - real ADAPT iteration

4. No legacy-route code is changed as part of this bug fix

5. The bug fix is treated as a code-regression repair only
   - no scientific interpretation changes should be mixed into the patch

## Follow-up after fix

After the code regression is fixed:

1. rerun the failed current cases:
   - `current_lam0p0`
   - `current_lam0p2`
   - `current_lam0p5`

2. continue the remaining sweep:
   - `legacy_lam0p7`
   - `current_lam0p7`
   - `legacy_lam1p0`
   - `current_lam1p0`

3. only after the current route is unblocked should we compare:
   - legacy vs current energy convergence
   - cost-vs-energy
   - whether the legacy poor rows reflect a route mismatch or genuine physics-point difficulty

## Primary artifact paths

- Sweep bundle:
  - `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/`
- Live summary:
  - `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/json/summary.json`
- Current failing case logs:
  - `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/cases/current_lam0p0/logs/stdout.log`
  - `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/cases/current_lam0p2/logs/stdout.log`
  - `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_lambda_sweep_legacy_vs_current_v1/cases/current_lam0p5/logs/stdout.log`
- Code seam:
  - `pipelines/static_adapt/adapt_pipeline.py`
  - `pipelines/static_adapt/statevector_runtime.py`
