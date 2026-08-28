"""Behavior locks for the canonical fixed-pool selected-QSE trace."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pipelines.exact_bench import paper_iii_growth_trace_campaign as campaign
from pipelines.qse_spectra.method_completion import (
    STATUS_POOL_EXHAUSTED,
    STATUS_TARGET_WINDOW_CERTIFIED,
    STATUS_TARGET_WINDOW_NOT_CERTIFIED,
    SUPPORT_CERTIFICATE_SCHEMA_VERSION,
)


_ROW = {
    "k": 1,
    "support_size": 1,
    "support_indices": [3],
    "exchanged": True,
    "retained_rank": 1,
    "max_root_abs_error": None,
    "n2q": 5.0,
    "d2q": 7.0,
    "dc": 11.0,
}


def _receipt(stop_reason: str, *, certified: bool) -> dict[str, Any]:
    return {
        "stop_reason": stop_reason,
        "final_max_target_residual_norm": 1.0e-5 if certified else 2.0e-2,
        "residual_converged_pending": certified,
        "residual_stop_threshold": 1.0e-3,
        "round_trace": [
            {
                "selected_count_before": 1,
                "retained_frame_size": 6,
                "max_target_residual_norm": 1.0e-5 if certified else 2.0e-2,
                "window_closed": certified,
            }
        ],
    }


def _support_certificate(*, certified: bool) -> dict[str, Any]:
    return {
        "schema_version": SUPPORT_CERTIFICATE_SCHEMA_VERSION,
        "support_indices": [3],
        "support_size": 1,
        "target_root_count": 6,
        "retained_frame_size": 6,
        "max_target_residual_norm": 1.0e-5 if certified else 2.0e-2,
        "residual_stop_threshold": 1.0e-3,
        "window_pressure": not certified,
        "window_closed": certified,
        "target_window_certified": certified,
        "uses_exact_reference": False,
        "online_only": True,
    }


def _problem() -> SimpleNamespace:
    return SimpleNamespace(basis=(), hamiltonian=None, ground=None)


def test_pool_exhaustion_returns_records_only_and_does_not_run_davidson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "trace_arm", lambda *_a, **_k: [dict(_ROW)])
    monkeypatch.setattr(
        campaign,
        "certify_fixed_pool_support",
        lambda *_a, **_k: _support_certificate(certified=False),
    )

    def fail_if_davidson_runs(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("canonical selected-QSE must not invoke Davidson")

    monkeypatch.setattr(campaign, "run_adaptive_qse_benchmark", fail_if_davidson_runs)

    rows, completion = campaign.trace_selected_qse_fixed_pool(
        _problem(),
        [3],
        k_max=4,
        stride=1,
        record_stop=_receipt("pool_exhausted", certified=False),
        target_root_count=6,
    )

    assert [row["phase"] for row in rows] == ["records"]
    assert completion["terminal_status"] == STATUS_POOL_EXHAUSTED
    assert completion["davidson_invoked"] is False


def test_selector_certificate_finishes_the_same_fixed_pool_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "trace_arm", lambda *_a, **_k: [dict(_ROW)])
    monkeypatch.setattr(
        campaign,
        "certify_fixed_pool_support",
        lambda *_a, **_k: _support_certificate(certified=True),
    )

    rows, completion = campaign.trace_selected_qse_fixed_pool(
        _problem(),
        [3],
        k_max=4,
        stride=1,
        record_stop=_receipt("residual_converged", certified=True),
        target_root_count=6,
    )

    assert rows[-1]["support_indices"] == [3]
    assert completion["terminal_status"] == STATUS_TARGET_WINDOW_CERTIFIED
    assert completion["davidson_invoked"] is False


def test_selector_receipt_does_not_transfer_across_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "trace_arm", lambda *_a, **_k: [dict(_ROW)])
    monkeypatch.setattr(
        campaign,
        "certify_fixed_pool_support",
        lambda *_a, **_k: _support_certificate(certified=False),
    )

    _rows, completion = campaign.trace_selected_qse_fixed_pool(
        _problem(),
        [3],
        k_max=4,
        stride=1,
        record_stop=_receipt("residual_converged", certified=True),
        target_root_count=6,
    )

    assert completion["terminal_status"] == STATUS_TARGET_WINDOW_NOT_CERTIFIED
