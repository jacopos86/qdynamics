"""Method-boundary locks for Paper-III selected-QSE completion."""

from __future__ import annotations

import pytest
import numpy as np

from pipelines.qse_spectra.method_completion import (
    ROUTE_SELECTED_QSE_FIXED_POOL,
    STATUS_POOL_EXHAUSTED,
    STATUS_TARGET_WINDOW_CERTIFIED,
    STATUS_TARGET_WINDOW_NOT_CERTIFIED,
    SUPPORT_CERTIFICATE_SCHEMA_VERSION,
    certify_fixed_pool_support,
    classify_selected_qse_completion,
)
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    pauli_string_basis_element,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _selector_receipt(stop_reason: str = "residual_converged") -> dict:
    return {
        "stop_reason": stop_reason,
        "final_max_target_residual_norm": 1.0e-5,
        "residual_converged_pending": True,
        "residual_stop_threshold": 1.0e-3,
        "round_trace": [],
    }


def _support_certificate(
    *,
    certified: bool,
    support_count: int,
    residual: float | None = None,
) -> dict:
    max_residual = (1.0e-5 if certified else 2.0e-2) if residual is None else residual
    return {
        "schema_version": SUPPORT_CERTIFICATE_SCHEMA_VERSION,
        "support_indices": list(range(support_count)),
        "support_size": support_count,
        "target_root_count": 6,
        "retained_frame_size": 6,
        "max_target_residual_norm": max_residual,
        "residual_stop_threshold": 1.0e-3,
        "window_pressure": not certified,
        "window_closed": certified,
        "target_window_certified": certified,
        "uses_exact_reference": False,
        "online_only": True,
    }


def test_fixed_pool_completion_accepts_actual_support_certificate() -> None:
    completion = classify_selected_qse_completion(
        _selector_receipt(),
        trace_terminal_record_count=12,
        selector_terminal_record_count=12,
        target_root_count=6,
        trace_reached_selector_terminal=True,
        terminal_support_certificate=_support_certificate(
            certified=True, support_count=12
        ),
    )

    assert completion["completion_route"] == ROUTE_SELECTED_QSE_FIXED_POOL
    assert completion["terminal_status"] == STATUS_TARGET_WINDOW_CERTIFIED
    assert completion["target_window_certified"] is True
    assert completion["davidson_invoked"] is False


def test_pool_exhaustion_is_terminal_and_never_implies_fallback() -> None:
    completion = classify_selected_qse_completion(
        _selector_receipt("pool_exhausted"),
        trace_terminal_record_count=17,
        selector_terminal_record_count=17,
        target_root_count=6,
        trace_reached_selector_terminal=True,
        terminal_support_certificate=_support_certificate(
            certified=False, support_count=17
        ),
    )

    assert completion["terminal_status"] == STATUS_POOL_EXHAUSTED
    assert completion["target_window_certified"] is False
    assert completion["davidson_invoked"] is False


def test_pre_exchange_selector_certificate_cannot_certify_changed_support() -> None:
    completion = classify_selected_qse_completion(
        _selector_receipt("residual_converged"),
        trace_terminal_record_count=12,
        selector_terminal_record_count=12,
        target_root_count=6,
        trace_reached_selector_terminal=True,
        terminal_support_certificate=_support_certificate(
            certified=False, support_count=12
        ),
    )

    assert completion["terminal_status"] == STATUS_TARGET_WINDOW_NOT_CERTIFIED
    assert completion["termination_reason"] == (
        "terminal_support_did_not_certify_target_window"
    )


def test_bare_selector_residual_stop_cannot_claim_target_window_coverage() -> None:
    completion = classify_selected_qse_completion(
        {"stop_reason": "residual_converged"},
        trace_terminal_record_count=6,
        selector_terminal_record_count=6,
        target_root_count=6,
        trace_reached_selector_terminal=True,
        terminal_support_certificate=None,
    )

    assert completion["terminal_status"] == STATUS_TARGET_WINDOW_NOT_CERTIFIED
    assert completion["target_window_certified"] is False


def test_contradictory_support_certificate_fails_closed() -> None:
    certificate = _support_certificate(certified=True, support_count=12)
    certificate["max_target_residual_norm"] = 2.0e-3

    with pytest.raises(ValueError, match="contradictory verdict"):
        classify_selected_qse_completion(
            _selector_receipt(),
            trace_terminal_record_count=12,
            selector_terminal_record_count=12,
            target_root_count=6,
            trace_reached_selector_terminal=True,
            terminal_support_certificate=certificate,
        )


def test_dimension_cap_uses_actual_cap_support_telemetry() -> None:
    certificate = _support_certificate(
        certified=False, support_count=8, residual=3.0e-2
    )
    completion = classify_selected_qse_completion(
        _selector_receipt(),
        trace_terminal_record_count=8,
        selector_terminal_record_count=12,
        target_root_count=6,
        trace_reached_selector_terminal=False,
        terminal_support_certificate=certificate,
    )

    assert completion["terminal_status"] == STATUS_TARGET_WINDOW_NOT_CERTIFIED
    assert completion["termination_reason"] == "dimension_cap_before_selector_terminal"
    assert completion["trace_terminal_record_count"] == 8
    assert completion["selector_terminal_record_count"] == 12
    assert completion["terminal_online_metrics_available"] is True
    assert completion["final_max_target_residual_norm"] == 3.0e-2
    assert completion["final_window_pressure"] is True
    assert completion["final_window_closed"] is False


def test_support_certificate_uses_only_actual_support_and_fixed_pool() -> None:
    hamiltonian = PauliPolynomial("JW")
    hamiltonian.add_term(PauliTerm(1, ps="z", pc=1.0))
    pool = (pauli_string_basis_element("x", nq=1, name="flip"),)
    certificate = certify_fixed_pool_support(
        pool,
        [0],
        hamiltonian=hamiltonian,
        prepared_state=np.asarray([1.0, 0.0], dtype=complex),
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        target_root_count=1,
        residual_stop_threshold=1.0e-3,
    )

    assert certificate["support_indices"] == [0]
    assert certificate["target_window_certified"] is True
    assert certificate["max_target_residual_norm"] == pytest.approx(0.0)
    assert certificate["uses_exact_reference"] is False
    assert certificate["online_only"] is True
