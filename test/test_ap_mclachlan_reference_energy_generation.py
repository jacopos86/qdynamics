from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.problem import RegisterBlockSpec, RegisterLayoutSpec
from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.reference_energy_generation import (
    REFERENCE_KIND_EXACT_INITIAL_STATE_V1,
    REFERENCE_KIND_SEED_PREPARED_STATE_V1,
    REFERENCE_METHOD_STATIC_SPECTRAL,
    ReferenceEnergyGenerationConfig,
    generate_reference_energy_trajectory,
    generate_seed_reference_energy_trajectory,
)
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact import (
    run_fixed_ap_mclachlan_from_runtime_input,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(label: str, coeff: float = 1.0) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _poly_nq(label: str, coeff: float = 1.0, *, nq: int) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(int(nq), ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _runtime_input() -> ScaffoldRuntimeInput:
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("z")),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=build_parameter_layout([]),
        theta_runtime=np.zeros(0, dtype=float),
        theta_logical=np.zeros(0, dtype=float),
        structure_locked=False,
        exact_energy=-1.0,
        selected_terms=(),
        candidate_pool_terms=(),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy.json"},
    )


def _hh_runtime_input() -> ScaffoldRuntimeInput:
    psi = np.zeros(16, dtype=complex)
    psi[5] = 1.0  # blocked ordering: site0 up/down occupied.
    layout = RegisterLayoutSpec(
        total_qubits=4,
        fermion_qubits=4,
        boson_qubits=0,
        ordering="blocked",
        boson_encoding=None,
        blocks=(RegisterBlockSpec("fermion", "fermion", 0, 4),),
    )
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="hh",
            request=SimpleNamespace(num_sites=2, ordering="blocked"),
            layout=layout,
            capabilities=SimpleNamespace(observable_kind="hh_spinful_boson"),
            hamiltonian=_poly_nq("eeee", 1.0, nq=4),
        ),
        psi_ref=psi,
        psi_initial=psi,
        base_layout=build_parameter_layout([]),
        theta_runtime=np.zeros(0, dtype=float),
        theta_logical=np.zeros(0, dtype=float),
        structure_locked=False,
        exact_energy=1.0,
        selected_terms=(),
        candidate_pool_terms=(),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="hh_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "hh_toy.json"},
    )


def test_default_reference_energy_uses_exact_initial_state_seed_error() -> None:
    runtime_input = _runtime_input()
    state = state_from_scaffold_runtime_input(runtime_input)
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(runtime_input)
    reference = generate_reference_energy_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=(0.0, 0.1, 0.2),
        config=ReferenceEnergyGenerationConfig(method=REFERENCE_METHOD_STATIC_SPECTRAL),
    )

    assert reference.metadata["reference_scope"] == "post_run_reporting"
    assert reference.metadata["reference_kind"] == REFERENCE_KIND_EXACT_INITIAL_STATE_V1
    assert reference.metadata["propagator_method"] == REFERENCE_METHOD_STATIC_SPECTRAL
    assert reference.metadata["initial_reference_state_source"] == "dense_static_eigenstate_nearest_imported_exact_energy"
    assert reference.metadata["exact_energy_target"] == pytest.approx(-1.0)
    assert reference.metadata["exact_energy_target_delta"] == pytest.approx(0.0)
    assert reference.points[0].energy == pytest.approx(-1.0)
    assert len(reference.points) == 3
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1, 0.2),
        reference_energy_trajectory=reference,
    )
    assert payload["summary"]["reference_energy_matched_count"] == 3
    assert payload["plot_rows"][0]["energy_expectation"] == pytest.approx(1.0)
    assert payload["plot_rows"][0]["abs_energy_error"] == pytest.approx(2.0)


def test_seed_prepared_reference_remains_explicit_debug_option() -> None:
    runtime_input = _runtime_input()
    state = state_from_scaffold_runtime_input(runtime_input)
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(runtime_input)
    reference = generate_seed_reference_energy_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=(0.0, 0.1, 0.2),
        config=ReferenceEnergyGenerationConfig(
            reference_kind=REFERENCE_KIND_SEED_PREPARED_STATE_V1,
            method=REFERENCE_METHOD_STATIC_SPECTRAL,
        ),
    )

    assert reference.metadata["reference_kind"] == REFERENCE_KIND_SEED_PREPARED_STATE_V1
    assert reference.metadata["initial_reference_state_source"] == "imported_seed_prepared_state"
    assert reference.points[0].energy == pytest.approx(1.0)
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1, 0.2),
        reference_energy_trajectory=reference,
    )
    assert payload["summary"]["reference_energy_matched_count"] == 3
    assert payload["summary"]["max_abs_energy_error"] == pytest.approx(0.0)
    assert payload["plot_rows"][0]["abs_energy_error"] == pytest.approx(0.0)


def test_hh_reference_and_ap_rows_emit_site_and_doublon_bundle() -> None:
    runtime_input = _hh_runtime_input()
    state = state_from_scaffold_runtime_input(runtime_input)
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(runtime_input)
    reference = generate_seed_reference_energy_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=(0.0, 0.1),
        config=ReferenceEnergyGenerationConfig(
            reference_kind=REFERENCE_KIND_SEED_PREPARED_STATE_V1,
            method=REFERENCE_METHOD_STATIC_SPECTRAL,
        ),
    )
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        reference_energy_trajectory=reference,
    )

    row = payload["plot_rows"][0]
    assert row["n_up_site"] == pytest.approx([1.0, 0.0])
    assert row["n_dn_site"] == pytest.approx([1.0, 0.0])
    assert row["site_occupations"] == pytest.approx([2.0, 0.0])
    assert row["doublon"] == pytest.approx(1.0)
    assert row["site_occupations_exact"] == pytest.approx([2.0, 0.0])
    assert row["doublon_exact"] == pytest.approx(1.0)
    assert row["abs_doublon_error"] == pytest.approx(0.0)
    assert row["site_occupations_abs_error_max"] == pytest.approx(0.0)
    assert payload["summary"]["reference_observable_diagnostics_enabled"] is True
    assert payload["summary"]["final_abs_doublon_error"] == pytest.approx(0.0)
    assert reference.metadata["reference_observable_support"] == "hh_site_doublon"


def test_reference_generation_rejects_duplicate_times() -> None:
    runtime_input = _runtime_input()
    with pytest.raises(ValueError, match="must not contain duplicates"):
        generate_seed_reference_energy_trajectory(
            state=state_from_scaffold_runtime_input(runtime_input),
            hamiltonian=time_dependent_hamiltonian_from_runtime_input(runtime_input),
            times=(0.0, 0.0),
        )
