from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.exact_bench import qiskit_community_dynamics_adapter as adapter
from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase
from pipelines.time_dynamics.tables.table_lock_contract import validate_shared_benchmark_surface_rows
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(label: str, coeff: float = 1.0) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps=label, pc=coeff))
    poly._reduce()
    return poly


def _layout() -> object:
    term = SimpleNamespace(label="x_rotation", polynomial=_poly("x", 1.0))
    return build_parameter_layout((term,))


def _hamiltonian_term(label: str = "x", coeff: float = 0.2) -> SimpleNamespace:
    return SimpleNamespace(pauli_exyz=label, coeff_real=float(coeff))


def _require_qiskit_algorithms() -> None:
    try:
        adapter.import_qiskit_community_components()
    except adapter.QiskitCommunityDynamicsUnavailable as exc:
        pytest.skip(f"optional qiskit-algorithms unavailable: {exc}")


@pytest.mark.parametrize("algorithm_id", adapter.QISKIT_COMMUNITY_ALGORITHMS)
def test_qiskit_community_adapter_runs_without_exact_inputs(algorithm_id: str) -> None:
    _require_qiskit_algorithms()
    result = adapter.run_qiskit_community_dynamics(
        config=adapter.QiskitCommunityDynamicsConfig(qubit_cap=2, pvqd_optimizer_maxiter=3),
        case=SimpleNamespace(family="unit", case_id="unit_qiskit_community"),
        algorithm_id=algorithm_id,
        terms_for_interval=lambda _left, _right: (_hamiltonian_term(),),
        times=np.asarray([0.0, 0.05], dtype=float),
        layout=_layout(),
        theta_runtime=np.asarray([0.0], dtype=float),
        psi_ref=np.asarray([1.0, 0.0], dtype=complex),
    )

    assert result.public_payload["status"] == "completed"
    assert result.public_payload["exact_reference_controller_inputs"] is False
    assert result.public_payload["exact_data_policy"].endswith("not_qiskit_algorithm_input")
    assert len(result.states_by_time) == 2
    assert all(np.isclose(np.linalg.norm(state), 1.0) for state in result.states_by_time)


def test_qiskit_primary_rows_do_not_require_parity_sidecars() -> None:
    case = DynamicsBenchmarkCase(
        case_id="unit",
        family="hubbard",
        table_class="fermionic",
        artifact_json="seed.json",
        metadata={"qiskit_dynamics": {"mode": "parity_required"}},
    )
    row = common._row_from_payload(
        case=case,
        algorithm_id="dyn_qiskit_trotter_qrte",
        payload={
            "trajectory": [
                {"energy_total": 0.0, "energy_total_exact": 0.0, "fidelity_exact": 1.0}
            ],
            "row_contract": {"qpu_faithful": True, "exact_assisted": False, "diagnostic": True},
            "provenance": {
                "qiskit_primary_mode": True,
                "qiskit_parity_sidecar": False,
                "repo_native_comparator": False,
            },
        },
        artifact_json=Path("raw_payload.json"),
        command=("unit",),
    )

    assert row.status == "completed"
    assert row.method_label == "Qiskit-community TrotterQRTE dynamics"
    assert row.table_fields.table_status_label == "Qiskit TrotterQRTE"


def test_shared_surface_validator_accepts_qiskit_community_group() -> None:
    surface = {
        "same_seed_comparator_group_id": "group",
        "static_seed_artifact_sha256": "seed",
        "drive_signature": "drive",
        "time_grid_signature": "grid",
        "observable_set_signature": "obs",
        "diagnostic_reference_signature": "ref",
        "compile_target_signature": "compile",
    }
    rows = [
        {"algorithm_id": algorithm_id, "case_id": "case", "provenance": {"benchmark_surface": surface}}
        for algorithm_id in adapter.QISKIT_COMMUNITY_ALGORITHMS
    ]

    report = validate_shared_benchmark_surface_rows(
        rows,
        required_algorithm_ids=adapter.QISKIT_COMMUNITY_ALGORITHMS,
    )

    assert report["passed"] is True
    assert report["group_count"] == 1


def test_varqrte_preflight_skips_large_qgt_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden_import():
        raise AssertionError("qiskit imports should not be reached after VarQRTE preflight skip")

    monkeypatch.setattr(adapter, "import_qiskit_community_components", _forbidden_import)

    with pytest.raises(adapter.QiskitCommunityDynamicsUnsupported, match="QGT preflight"):
        adapter.run_qiskit_community_dynamics(
            config=adapter.QiskitCommunityDynamicsConfig(
                qubit_cap=2,
                varqrte_max_runtime_parameters=4,
                varqrte_max_qgt_entries=16,
            ),
            case=SimpleNamespace(family="unit", case_id="unit_varqrte_preflight"),
            algorithm_id="dyn_qiskit_varqrte",
            terms_for_interval=lambda _left, _right: (_hamiltonian_term(),),
            times=np.asarray([0.0, 0.05], dtype=float),
            layout=_layout(),
            theta_runtime=np.zeros(5, dtype=float),
            psi_ref=np.asarray([1.0, 0.0], dtype=complex),
        )


def test_qiskit_community_metadata_parses_varqrte_preflight_caps() -> None:
    config = adapter.qiskit_community_config_from_metadata(
        {
            "qiskit_community_dynamics": {
                "varqrte_max_runtime_parameters": 8,
                "varqrte_max_qgt_entries": 64,
            }
        }
    )

    assert config.varqrte_max_runtime_parameters == 8
    assert config.varqrte_max_qgt_entries == 64
