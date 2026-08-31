# Investigation: HH L2 U0 w0.5 bad ΔE regression

## Summary
The primary issue is a **static `dv` sign/convention mismatch** in the HH adapter layer: the ADAPT Hamiltonian is built with `v_t=None, v0=dv`, while the exact HH anchor/reference-state helpers use ED with `delta_v=dv`. At half-filling for `L=2`, that creates a sector-constant `+4` shift between the reported exact energy and the actually optimized Hamiltonian, which is why both legacy and current can report `|ΔE| ≈ 4` even when the state is actually good.

A second issue is **path interpretation**: my fresh rerun used only the legacy hybrid compatibility lane, not both legacy and current. After correcting for the `dv` seam, the remaining residual errors appear to be **route-quality differences**, not the same reporting bug: legacy-at-`λ=0.5` has corrected `|ΔE| ≈ 0.0276`, while current-at-`λ=0.5` has corrected `|ΔE| ≈ 0.00352` on the same physics point.

## Symptoms
- Legacy-only sweep at `L=2`, `U=0`, `omega0=0.5`, `dv=1`, `n_ph_max=1`, `lambda ∈ {0, 0.2, 0.5, 0.7, 1}` completed with uniformly bad reported `|ΔE| ≈ 3.94–4.00`.
- Prior note claimed the current path crashed with `name '_compile_polynomial_action_shared' is not defined`.
- User expectation: both legacy and current normally converge well for `L=2`.
- My rerun did not use both paths; it used legacy only.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** The poor `ΔE` likely reflects a code/convention error rather than genuine convergence difficulty.
**Findings:** Initial symptoms recorded; broad discovery then focused on HH Hamiltonian assembly, exact-ground-state anchoring, and path coverage.
**Evidence:**
- `artifacts/agent_runs/20260411T185821Z_hh_l2_u0_w0p5_lambda_sweep_legacy_only_v1/json/summary.json`
- `notes/2026-04-11-hh-static-sweep-current-regression-audit.md`
**Conclusion:** A broad context pass was required before trusting any single hypothesis.

### Phase 2 - Broad context gathering
**Hypothesis:** The relevant seam could live in HH Hamiltonian assembly, exact-energy anchoring, legacy/current wrappers, or the run harness.
**Findings:** `context_builder` selected the static-adapt builders, HH Pauli/ED builders, current/legacy entrypoints, scoring bridge, tests, and the live sweep harness. Its initial assessment identified a likely `dv` convention mismatch and a path-selection ambiguity.
**Evidence:**
- `context_builder` chat `hh-sweep-audit-31D322`
- Selected files included `pipelines/static_adapt/builders/problem_setup.py`, `src/quantum/hubbard_latex_python_pairs.py`, `src/quantum/ed_hubbard_holstein.py`, `pipelines/static_adapt/adapt_pipeline.py`, `pipelines/static_adapt/adapt_pipeline_legacy_20260322.py`, and the live run harness
**Conclusion:** Proceed to exact line-number verification of the `dv` path and the actual executed entrypoints.

### Phase 3 - Verify HH `dv` seam
**Hypothesis:** The ADAPT HH Hamiltonian and the exact HH anchor use opposite static-`dv` conventions.
**Findings:** Confirmed.

`build_problem_hamiltonian(...)` builds HH with:
- `pipelines/static_adapt/builders/problem_setup.py:91-103`

```python
return build_hubbard_holstein_hamiltonian(
    ...
    v_t=None,
    v0=float(dv),
    ...
)
```

But the HH drive builder defines the effective static term via `delta = v_t - v0`:
- `src/quantum/hubbard_latex_python_pairs.py:1037-1045`

```python
r"""
Time-dependent drive for Holstein:
    H_{\rm drive} = \sum_{i,\sigma} \bigl(v_i(t) - v_{0,i}\bigr) \hat{n}_{i\sigma}
"""
```

and then converts that `delta` into the existing Hubbard potential convention. With `v_t=None -> 0` and `v0=dv`, the Pauli-built HH Hamiltonian seen by ADAPT is effectively shifted by `-dv * N`.

The exact ground-state helper does something different:
- `pipelines/static_adapt/builders/problem_setup.py:324-346`

```python
h_sector = build_hh_sector_hamiltonian_ed(
    ...
    delta_v=float(dv),
    ...
)
```

The ED builder documents and implements that as `+delta_v * n`:
- `src/quantum/ed_hubbard_holstein.py:317-343`
- `src/quantum/ed_hubbard_holstein.py:388-392`

```python
H_{drv} = \sum_{i\sigma} \delta v_i \, n_{i\sigma}
...
diag += float(delta_v_i[site]) * float(n_up + n_dn)
```

The existing cross-check test explicitly documents the intended matching convention for static `dv` on the Pauli side as `v_t=delta_v, v0=None`:
- `test/test_ed_crosscheck.py:56-63`

```python
# delta_v in the ED module means: H_drive = sum_i,sigma delta_v_i * n_i,sigma
# In the Pauli builder: pass as v_t (static values), v0=None  →  delta = v_t - 0
# ...
# Net: +delta_v * n  — matches.
```

**Evidence:**
- `pipelines/static_adapt/builders/problem_setup.py:91-103`
- `src/quantum/hubbard_latex_python_pairs.py:1037-1045`
- `pipelines/static_adapt/builders/problem_setup.py:324-346`
- `src/quantum/ed_hubbard_holstein.py:317-343`
- `src/quantum/ed_hubbard_holstein.py:388-392`
- `test/test_ed_crosscheck.py:56-63`
**Conclusion:** Confirmed primary root cause for the reported `≈4` errors.

### Phase 3 - Numeric verification of the `+4` shift
**Hypothesis:** If the `dv` seam is the root cause, the exact-energy helper should differ from the Pauli-built Hamiltonian’s exact sector energy by exactly `4` on this half-filled `L=2`, `dv=1` sweep.
**Findings:** Confirmed numerically for all tested sweep points.
**Evidence:** One-off in-process verification comparing:
- `pauli_exact = exact_ground_energy_sector_hh(build_problem_hamiltonian(...))`
- `helper_exact = _exact_gs_energy_for_problem(...)`

Observed gaps:
- `g_ep=0.0`: gap `≈ 4.0`
- `g_ep=0.2236068`: gap `≈ 4.0`
- `g_ep=0.3535534`: gap `≈ 4.0`
- `g_ep=0.41833`: gap `≈ 4.0`
- `g_ep=0.5`: gap `≈ 4.0`

Corresponding corrected legacy-only errors against the actually built Pauli Hamiltonian were:
- `λ=0.0`: `~3.6e-15`
- `λ=0.2`: `~0.01042`
- `λ=0.5`: `~0.02759`
- `λ=0.7`: `~0.04000`
- `λ=1.0`: `~0.059996`

Artifact evidence for the originally reported values:
- `artifacts/agent_runs/20260411T185821Z_hh_l2_u0_w0p5_lambda_sweep_legacy_only_v1/cases/*/json/result.json`
**Conclusion:** The huge reported `|ΔE|` values are mostly a **reporting/anchoring bug**, not a genuine 4-Hartree convergence collapse.

### Phase 3 - Verify which path was actually run
**Hypothesis:** My rerun may not have exercised both paths.
**Findings:** Confirmed. The rerun was legacy-only.
**Evidence:**
- `artifacts/agent_runs/20260411T185821Z_hh_l2_u0_w0p5_lambda_sweep_legacy_only_v1/run_legacy_matrix.sh:191-228`

```bash
local -a cmd=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/pipelines/static_adapt/adapt_pipeline_legacy_20260322.py"
  ...
)
```

No current entrypoint appears anywhere in that harness.
**Conclusion:** The criticism was valid: my rerun did **not** use both paths.

### Phase 3 - Check whether current still crashes in this checkout
**Hypothesis:** The old `_compile_polynomial_action_shared` crash may be stale.
**Findings:** Confirmed stale for this checkout.

The current file imports `_compile_polynomial_action_shared` directly:
- `pipelines/static_adapt/adapt_pipeline.py:65-72`

```python
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    compile_polynomial_action as _compile_polynomial_action_shared,
    energy_via_one_apply,
)
```

I then ran two direct current-path reproductions with `pipelines/static_adapt/adapt_pipeline.py`:

1. `g_ep=0.0`
   - artifact: `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_current_lam0p0_investigation_v1/json/result.json`
   - result: `energy=-3.5000000000000027`, `exact_gs_energy=0.49999999999999883`, reported `|ΔE|=4.000000000000002`
   - corrected same-Hamiltonian error: `~3.1e-15`

2. `g_ep=0.3535534` (`λ=0.5`)
   - artifact: `artifacts/agent_runs/20260411_hh_l2_u0_w0p5_current_lam0p5_investigation_v1/json/result.json`
   - result: `energy=-3.5490381082368545`, `exact_gs_energy=0.4474396281326062`, reported `|ΔE|=3.9964777363694606`
   - corrected same-Hamiltonian error: `~0.003522`

**Conclusion:** In this checkout, current does **not** crash on the tested points; it reproduces the same reporting bug and, at `λ=0.5`, outperforms the exercised legacy lane after correction.

### Phase 4 - Check adjacent static-`dv` seam
**Hypothesis:** The same static-`dv` misuse may exist elsewhere.
**Findings:** Confirmed same pattern in the HH layerwise primitive pool builder.
**Evidence:**
- `pipelines/static_adapt/builders/primitive_pools.py:216-226`

```python
layerwise = HubbardHolsteinLayerwiseAnsatz(
    ...
    v=None,
    v_t=None,
    v0=float(dv),
    ...
)
```

**Conclusion:** This is not the primary cause of the reported `≈4` exact-energy bug, but it is the same convention seam and should be audited/fixed consistently.

## Root Cause
**Primary root cause:** a static-`dv` adapter bug in `pipelines/static_adapt/builders/problem_setup.py`.

For HH, `build_problem_hamiltonian(...)` currently uses `v_t=None, v0=dv`, while `_exact_gs_energy_for_problem(...)` and `_exact_reference_state_for_hh(...)` use ED with `delta_v=dv`. The repo’s own cross-check test says the matching Pauli convention for static `dv` is `v_t=dv, v0=None`. Because the sweep is half-filled with fixed total particle number `N=2`, the mismatch appears as a sector-constant `+4` shift in the reported exact energy, which is why both legacy and current can report `|ΔE| ≈ 4` at this physics point.

**Secondary finding:** once the reporting bug is removed, there is still evidence of a route-quality gap in the exercised legacy hybrid lane. At `λ=0.5`, the corrected legacy error is `~0.0276`, while the corrected current error on the same heavy/full-meta surface is `~0.00352`.

## Eliminated Hypotheses
- **Eliminated:** “The `≈4` errors mean the optimizer genuinely failed catastrophically.”
  The `≈4` is explained by the static-`dv` anchor mismatch.
- **Eliminated:** “Low-level Pauli HH construction and ED disagree on static `dv` sign.”
  They agree when the Pauli side uses the convention documented in `test_ed_crosscheck.py`.
- **Eliminated for this checkout:** “Current still crashes here with `_compile_polynomial_action_shared` undefined.”
  The present file imports that symbol, and current ran successfully on the tested points.
- **Confirmed instead of eliminated:** my fresh rerun did **not** exercise both paths; it was legacy-only.

## Recommendations
1. **Fix the primary HH adapter seam** in `pipelines/static_adapt/builders/problem_setup.py:91-103`.
   - Change the HH builder call from:
     - `v_t=None, v0=float(dv)`
   - to:
     - `v_t=float(dv), v0=None`
   - This is the minimal fix that aligns the built Pauli Hamiltonian with the ED exact anchor.

2. **Add a regression test** that locks the adapter behavior.
   - New test target: for HH with nonzero static `dv`, assert that:
     - `build_problem_hamiltonian(... problem_key='hh' ...)`
     - and `_exact_gs_energy_for_problem(...)`
     refer to the **same** Hamiltonian convention.
   - A simple version can compare `exact_ground_energy_sector_hh(build_problem_hamiltonian(...))` against `_exact_gs_energy_for_problem(...)` at `L=2`, `dv=1`.

3. **Audit and likely fix the same static-`dv` pattern** in `pipelines/static_adapt/builders/primitive_pools.py:216-226`.
   - The HH layerwise ansatz constructor currently also uses `v_t=None, v0=float(dv)`.
   - That may affect HVA-based HH pool materialization and should be made convention-consistent.

4. **Re-run a true dual-path comparison after the fix** using an explicit two-entrypoint harness.
   - Either use `pipelines/static_adapt/compare_adapt_current_vs_legacy_20260322.py`
   - or a new harness that logs both entrypoints unambiguously.
   - This will separate the fixed reporting bug from any real legacy/current route-quality gap.

## Preventive Measures
- Add a dedicated HH static-`dv` regression test at the adapter level, not only low-level Pauli-vs-ED builder tests.
- Add a test that checks `_exact_reference_state_for_hh(...)` is convention-consistent with `build_problem_hamiltonian(...)` for nonzero static `dv`.
- When creating investigation or sweep harnesses, write the chosen entrypoint(s) into the manifest and summary explicitly so path coverage is never ambiguous.
- Be cautious interpreting repo notes in this checkout: `git status --short` shows `pipelines/static_adapt/*.py`, `pipelines/static_adapt/builders/problem_setup.py`, and `run_guide.md` as untracked working-tree files, so note-level claims may not match the live code state.
