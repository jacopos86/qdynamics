"""Continuation parity: resuming mid-trajectory must match running straight through."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from pipelines.time_dynamics.resume import (
    RESUME_SCHEMA_V1,
    runtime_input_from_resume_state,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    AppendControllerConfig,
    run_append_ap_mclachlan_from_runtime_input,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*components: tuple[str, float], nq: int = 2) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return poly


X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 0.8)))
CA = AnsatzTerm(label="cand_a", polynomial=_poly(("xe", 0.7)))
HAM_POLY = _poly(("ez", 2.0), ("xx", 0.6))


def _runtime_input(theta=(0.35, -0.22), tmp_path=None):
    selected = (X0, Z0)
    layout = build_parameter_layout(selected)
    executor = CompiledAnsatzExecutor(
        selected, parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=HAM_POLY),
        psi_ref=psi_ref,
        psi_initial=np.asarray(
            executor.prepare_state(np.asarray(theta, dtype=float), psi_ref), dtype=complex
        ),
        base_layout=layout,
        theta_runtime=np.asarray(theta, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(CA,),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete",
        ),
        provenance={"artifact_json": "resume_toy.json"},
    )


def _run(runtime_input, times):
    return run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=times,
        integrator_method="rk4",
        controller_config=AppendControllerConfig(max_append_candidates=0),
        support_patch_config=SupportPatchControllerConfig(
            residual_ratio_threshold=0.02,
            max_structural_pool_size=1,
            min_runtime_parameter_count=1,
            max_append_batch_size=1,
        ),
        solve_repair_config=SolveRepairConfig.minimal_profile(),
    )


GRID = tuple(0.05 * k for k in range(7))


def test_resume_state_is_recorded_and_complete() -> None:
    payload = _run(_runtime_input(), GRID[:4])
    resume = payload["resume_state"]
    assert resume["schema"] == RESUME_SCHEMA_V1
    assert resume["time"] == pytest.approx(GRID[3])
    # One entry per runtime coordinate, each a single Pauli rotation.
    n = payload["summary"]["runtime_parameter_count_final"]
    assert len(resume["coordinates"]) == n
    assert len(resume["theta_runtime"]) == n
    for c in resume["coordinates"]:
        assert c["pauli_exyz"] and c["nq"] >= 1


def test_resumed_state_reproduces_the_interrupted_state() -> None:
    """The rebuilt input must sit exactly where the first leg finished."""
    first = _run(_runtime_input(), GRID[:4])
    resumed_input = runtime_input_from_resume_state(
        first["resume_state"], base_runtime_input=_runtime_input()
    )
    # Energy of the rebuilt state equals the energy at the interruption point.
    tail_energy = first["plot_rows"][-1]["energy_expectation"]
    payload = _run(resumed_input, (GRID[3], GRID[3]))
    assert payload["plot_rows"][0]["energy_expectation"] == pytest.approx(
        tail_energy, abs=1e-9
    )


def test_continuation_matches_an_uninterrupted_trajectory() -> None:
    """Split propagation and continuous propagation must agree."""
    straight = _run(_runtime_input(), GRID)
    first = _run(_runtime_input(), GRID[:4])
    resumed_input = runtime_input_from_resume_state(
        first["resume_state"], base_runtime_input=_runtime_input()
    )
    second = _run(resumed_input, GRID[3:])

    assert second["plot_rows"][-1]["energy_expectation"] == pytest.approx(
        straight["plot_rows"][-1]["energy_expectation"], abs=1e-7
    )
    assert (
        second["summary"]["runtime_parameter_count_final"]
        == straight["summary"]["runtime_parameter_count_final"]
    )
