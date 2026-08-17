from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.scaffold.runtime_contract import (
    CandidatePoolSource,
    ScaffoldRuntimeInput,
)
from pipelines.time_dynamics.benchmarks.avqds_tetris import (
    build_tetris_pool_contract,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    reject_removed_online_redundancy_flags,
    run_append_ap_mclachlan_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    AppendControllerConfig,
)
from pipelines.time_dynamics.normalized_pauli_pool import (
    NORMALIZED_POOL_FULL_META_CHILDREN,
    NORMALIZED_POOL_HAMILTONIAN_DRIVE,
    build_normalized_pauli_pool,
    runtime_input_with_normalized_candidate_pool,
)
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    iter_runtime_rotation_terms,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*terms: tuple[str, float]) -> PauliPolynomial:
    return PauliPolynomial(
        "JW",
        [
            PauliTerm(len(pauli), ps=str(pauli), pc=float(coefficient))
            for pauli, coefficient in terms
        ],
    )


def _term(
    label: str,
    *terms: tuple[str, float],
    execution_mode: str = "termwise_product",
) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=_poly(*terms),
        execution_mode=str(execution_mode),
    )


def test_hamiltonian_drive_profile_deduplicates_to_unit_pauli_set() -> None:
    contract = build_normalized_pauli_pool(
        profile=NORMALIZED_POOL_HAMILTONIAN_DRIVE,
        static_poly=_poly(("x", 0.7), ("z", -0.2)),
        drive_poly=_poly(("y", 3.0), ("z", 5.0)),
    )

    assert contract.ordered_paulis == ("x", "y", "z")
    assert contract.source_occurrence_count == 4
    assert contract.source_parent_count == 2
    assert contract.untruncated_atom_count == 3
    assert contract.truncated is False
    assert len(contract.ordered_unique_pauli_sha256) == 64
    assert contract.atoms[2].source_labels == (
        "static_hamiltonian",
        "drive_hamiltonian",
    )


def test_full_meta_profile_deduplicates_children_and_skips_grouped_exact() -> None:
    candidates = (
        _term("first", ("x", 0.5), ("z", -0.5)),
        _term("second", ("y", 2.0), ("z", 4.0)),
        _term(
            "not_a_pauli_child_candidate",
            ("x", 1.0),
            execution_mode="grouped_exact",
        ),
    )

    contract = build_normalized_pauli_pool(
        profile=NORMALIZED_POOL_FULL_META_CHILDREN,
        static_poly=_poly(("x", 1.0)),
        candidate_pool_terms=candidates,
    )

    assert contract.ordered_paulis == ("x", "y", "z")
    assert contract.source_occurrence_count == 4
    assert contract.source_parent_count == 2
    assert contract.atoms[2].source_labels == ("first", "second")


def test_runtime_pool_replacement_preserves_seed_and_uses_unit_children() -> None:
    selected = (_term("seed", ("z", 0.25)),)
    layout = build_parameter_layout(selected)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(hamiltonian=_poly(("x", 1.0))),
        psi_ref=np.asarray([1.0, 0.0], dtype=complex),
        psi_initial=np.asarray([1.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.asarray([0.3], dtype=float),
        theta_logical=np.asarray([0.3], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(
            _term("first", ("x", 0.5), ("z", -0.5)),
            _term("second", ("y", 2.0), ("z", 4.0)),
        ),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="full_meta",
            completeness="complete",
        ),
    )
    contract = build_normalized_pauli_pool(
        profile=NORMALIZED_POOL_FULL_META_CHILDREN,
        static_poly=runtime_input.h_poly,
        candidate_pool_terms=runtime_input.candidate_pool_terms,
    )

    replaced = runtime_input_with_normalized_candidate_pool(
        runtime_input,
        contract,
    )

    assert replaced.selected_terms == runtime_input.selected_terms
    assert np.array_equal(replaced.theta_runtime, runtime_input.theta_runtime)
    assert replaced.candidate_pool_source.candidate_pool_complete is True
    assert replaced.candidate_pool_source.pool_key == (
        "normalized::full_meta_pauli_children"
    )
    assert len(replaced.candidate_pool_terms) == 3
    assert [
        next(
            iter(
                iter_runtime_rotation_terms(
                    term.polynomial,
                    ignore_identity=True,
                    sort_terms=True,
                )
            )
        ).coeff_real
        for term in replaced.candidate_pool_terms
    ] == [1.0, 1.0, 1.0]


def test_avqds_uses_the_same_shared_pool_contracts() -> None:
    static_poly = _poly(("x", 0.7), ("z", -0.2))
    drive_poly = _poly(("y", 3.0), ("z", 5.0))
    candidates = (
        _term("first", ("x", 0.5), ("z", -0.5)),
        _term("second", ("y", 2.0), ("z", 4.0)),
    )
    flow = SimpleNamespace(
        hamiltonian=SimpleNamespace(
            static_poly=static_poly,
            drive_poly=drive_poly,
        )
    )
    runtime_input = SimpleNamespace(candidate_pool_terms=candidates)

    hamiltonian_contract = build_tetris_pool_contract(
        flow=flow,
        runtime_input=runtime_input,
        pool_source="hamiltonian_pauli",
    )
    full_meta_contract = build_tetris_pool_contract(
        flow=flow,
        runtime_input=runtime_input,
        pool_source="runtime_candidate_pool",
    )

    assert hamiltonian_contract.profile == NORMALIZED_POOL_HAMILTONIAN_DRIVE
    assert full_meta_contract.profile == NORMALIZED_POOL_FULL_META_CHILDREN
    assert hamiltonian_contract.ordered_paulis == ("x", "y", "z")
    assert full_meta_contract.ordered_paulis == ("x", "y", "z")
    assert (
        hamiltonian_contract.ordered_unique_pauli_sha256
        == full_meta_contract.ordered_unique_pauli_sha256
    )


def test_apm_runner_serializes_the_selected_normalized_pool_contract() -> None:
    selected: tuple[AnsatzTerm, ...] = ()
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy",
            hamiltonian=_poly(("x", 0.7), ("z", -0.2)),
        ),
        psi_ref=np.asarray([1.0, 0.0], dtype=complex),
        psi_initial=np.asarray([1.0, 0.0], dtype=complex),
        base_layout=build_parameter_layout(selected),
        theta_runtime=np.zeros(0, dtype=float),
        theta_logical=np.zeros(0, dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(_term("unused_full_meta", ("y", 1.0)),),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="full_meta",
            completeness="complete",
        ),
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.01),
        normalized_candidate_pool_profile=NORMALIZED_POOL_HAMILTONIAN_DRIVE,
    )

    contract = payload["normalized_candidate_pool"]
    assert contract["profile"] == NORMALIZED_POOL_HAMILTONIAN_DRIVE
    assert contract["atom_count"] == 2
    assert payload["summary"]["candidate_pool_term_count"] == 2
    assert (
        payload["initial_state"]["candidate_pool_source"]["filter_payload"][
            "normalized_pauli_pool"
        ]["ordered_atom_contract_sha256"]
        == contract["ordered_atom_contract_sha256"]
    )


def test_apm_runner_has_no_online_redundancy_injection() -> None:
    """The runner must not accept or perform online zero-angle redundancy injection."""

    selected: tuple[AnsatzTerm, ...] = ()
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy",
            hamiltonian=_poly(("x", 0.7), ("z", -0.2)),
        ),
        psi_ref=np.asarray([1.0, 0.0], dtype=complex),
        psi_initial=np.asarray([1.0, 0.0], dtype=complex),
        base_layout=build_parameter_layout(selected),
        theta_runtime=np.zeros(0, dtype=float),
        theta_logical=np.zeros(0, dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(_term("unused_full_meta", ("y", 1.0)),),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="full_meta",
            completeness="complete",
        ),
    )
    controller = AppendControllerConfig(max_append_candidates=0)

    signature = inspect.signature(run_append_ap_mclachlan_from_runtime_input)
    assert "redundancy_stress_config" not in signature.parameters

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.01),
        normalized_candidate_pool_profile=NORMALIZED_POOL_HAMILTONIAN_DRIVE,
        controller_config=controller,
    )

    assert "diagnostic_redundancy_stress" not in payload
    assert payload["summary"]["runtime_parameter_count_initial"] == 0
    stress = payload["fixed_vqe_conditioning_stress"]
    assert stress["online_injection_used"] is False
    assert stress["present"] is False
    assert stress["source"] == "serialized_seed_artifact"


def test_apm_runner_rejects_removed_online_redundancy_command_line() -> None:
    """An old command line must fail loudly and point at the offline builder."""

    with pytest.raises(SystemExit) as excinfo:
        reject_removed_online_redundancy_flags(
            ["--artifact-json", "seed.json", "--diagnostic-redundancy-layer-count", "2"]
        )
    message = str(excinfo.value)
    assert "has been removed" in message
    assert "build_fixed_vqe_conditioning_seed" in message
