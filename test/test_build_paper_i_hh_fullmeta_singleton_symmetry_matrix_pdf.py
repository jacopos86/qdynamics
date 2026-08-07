from __future__ import annotations

import pytest

from pipelines.reporting.build_paper_i_hh_fullmeta_singleton_symmetry_matrix_pdf import (
    _compiled_value_if_qiskit,
    _validated_generic_qiskit_cost,
)


def _valid_generic_cost() -> dict[str, object]:
    return {
        "compiled_resource_qiskit_validated": True,
        "compiled_circuit_stats_status": "ok",
        "compiled_resource_source_kind": "qiskit_compiled_final_ansatz_circuit",
        "compiled_count_2q_total": 12,
        "compiled_depth_2q_total": 7,
        "compiled_depth_total": 21,
    }


def _valid_sidecar_cost() -> dict[str, object]:
    return {
        "status": "ok",
        "compiled_resource_qiskit_validated": True,
        "compiled_resource_source_kind": "qiskit_compiled_terminal_ansatz_circuit",
        "compiled_count_2q_total": 12,
        "compiled_depth_2q_total": 7,
        "compiled_depth_total": 21,
    }


@pytest.mark.parametrize(
    ("updates", "expected_reason"),
    [
        ({"compiled_resource_qiskit_validated": False}, "blocked:qiskit_validation_flag_false"),
        ({"compiled_circuit_stats_status": ""}, "blocked:qiskit_status_missing"),
        (
            {"compiled_resource_source_kind": "deterministic_pauli_rotation_proxy"},
            "blocked:qiskit_source_kind_missing",
        ),
    ],
)
def test_validated_generic_qiskit_cost_rejects_source_only_or_unvalidated_payloads(
    updates: dict[str, object],
    expected_reason: str,
) -> None:
    payload = _valid_generic_cost()
    payload.update(updates)

    assert _validated_generic_qiskit_cost(payload) == (False, expected_reason)


def test_validated_generic_qiskit_cost_accepts_fully_valid_payload() -> None:
    assert _validated_generic_qiskit_cost(_valid_generic_cost()) == (True, "ok")


@pytest.mark.parametrize(
    "missing_field",
    [
        "compiled_count_2q_total",
        "compiled_depth_2q_total",
        "compiled_depth_total",
    ],
)
def test_validated_generic_qiskit_cost_rejects_missing_cost_fields(missing_field: str) -> None:
    payload = _valid_generic_cost()
    payload.pop(missing_field)

    assert _validated_generic_qiskit_cost(payload) == (
        False,
        "blocked:qiskit_cost_columns_missing",
    )


@pytest.mark.parametrize(
    ("updates", "expected_reason"),
    [
        ({"compiled_count_2q_total": -1}, "blocked:qiskit_cost_negative"),
        (
            {"compiled_depth_2q_total": 8, "compiled_depth_total": 7},
            "blocked:qiskit_depth_order_invalid",
        ),
    ],
)
def test_validated_generic_qiskit_cost_rejects_inconsistent_cost_fields(
    updates: dict[str, object],
    expected_reason: str,
) -> None:
    payload = _valid_generic_cost()
    payload.update(updates)

    assert _validated_generic_qiskit_cost(payload) == (False, expected_reason)


def test_compiled_value_if_qiskit_accepts_validated_qiskit_values_and_fallback_keys() -> None:
    payload = _valid_sidecar_cost()
    assert _compiled_value_if_qiskit(payload, "compiled_count_2q_total", "N2q") == 12

    payload.pop("compiled_count_2q_total")
    payload["N2q"] = 13
    assert _compiled_value_if_qiskit(payload, "compiled_count_2q_total", "N2q") == 13


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "blocked:compile_failed"},
        {"compiled_resource_qiskit_validated": False},
        {"compiled_resource_source_kind": "deterministic_pauli_rotation_proxy"},
        {"compiled_count_2q_total": None},
    ],
)
def test_compiled_value_if_qiskit_rejects_nonvalidated_or_missing_values(
    updates: dict[str, object],
) -> None:
    payload = _valid_sidecar_cost()
    payload.update(updates)

    assert _compiled_value_if_qiskit(payload, "compiled_count_2q_total") is None
