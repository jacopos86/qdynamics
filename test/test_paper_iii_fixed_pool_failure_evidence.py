"""Schema locks for native fixed-pool failure-frequency evidence."""

from __future__ import annotations

import pytest

from pipelines.exact_bench.paper_iii_fixed_pool_failure_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    METHOD_ID,
    RELEVANT_SOURCE_PATHS,
    SELECTION_POLICY,
    assert_native_fixed_pool_trace,
    summarize_terminal_records,
    validate_failure_evidence,
)
from pipelines.qse_spectra.method_completion import (
    ROUTE_SELECTED_QSE_FIXED_POOL,
    STATUS_POOL_EXHAUSTED,
    STATUS_TARGET_WINDOW_CERTIFIED,
    STATUS_TARGET_WINDOW_NOT_CERTIFIED,
    SUPPORT_CERTIFICATE_SCHEMA_VERSION,
    classify_selected_qse_completion,
)
from pipelines.qse_spectra.static_campaign import canonical_digest


def test_terminal_summary_has_an_explicit_denominator_and_separate_counts() -> None:
    records = [
        {"regime": "b", "algorithm_completion": {"terminal_status": STATUS_POOL_EXHAUSTED}},
        {
            "regime": "a",
            "algorithm_completion": {"terminal_status": STATUS_TARGET_WINDOW_CERTIFIED},
        },
        {
            "regime": "c",
            "algorithm_completion": {
                "terminal_status": STATUS_TARGET_WINDOW_NOT_CERTIFIED
            },
        },
    ]

    summary = summarize_terminal_records(records)

    assert summary["cell_count"] == 3
    assert summary["uncertified_cell_count"] == 2
    assert summary["uncertified_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["terminal_status_counts"] == {
        STATUS_POOL_EXHAUSTED: 1,
        STATUS_TARGET_WINDOW_CERTIFIED: 1,
        STATUS_TARGET_WINDOW_NOT_CERTIFIED: 1,
    }


def test_native_fixed_pool_evidence_rejects_any_synthesis_phase() -> None:
    with pytest.raises(ValueError, match="synthesis"):
        assert_native_fixed_pool_trace(
            [{"phase": "records"}, {"phase": "synthesis"}]
        )

    assert_native_fixed_pool_trace([{"phase": "records"}])


def _valid_payload() -> dict:
    support = [0, 1, 2]
    certificate = {
        "schema_version": SUPPORT_CERTIFICATE_SCHEMA_VERSION,
        "support_indices": support,
        "support_size": 3,
        "target_root_count": 6,
        "retained_frame_size": 3,
        "max_target_residual_norm": 2.0e-2,
        "residual_stop_threshold": 1.0e-3,
        "window_pressure": True,
        "window_closed": False,
        "target_window_certified": False,
        "uses_exact_reference": False,
        "online_only": True,
    }
    selector_receipt = {"stop_reason": "pool_exhausted"}
    completion = {
        "method_id": METHOD_ID,
        **classify_selected_qse_completion(
            selector_receipt,
            trace_terminal_record_count=3,
            selector_terminal_record_count=3,
            target_root_count=6,
            trace_reached_selector_terminal=True,
            terminal_support_certificate=certificate,
        ),
        "record_stop": selector_receipt,
    }
    record = {
        "regime": "weak_weak",
        "dimension_cap": 64,
        "stride": 2,
        "selected_record_count": 3,
        "traced_terminal_k": 3,
        "terminal_support_size": 3,
        "terminal_support_indices": support,
        "algorithm_completion": completion,
        "selector_terminal_receipt": selector_receipt,
        "trace": [
            {
                "phase": "records",
                "k": 3,
                "support_size": 3,
                "support_indices": support,
            }
        ],
    }
    records = [record]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "completion_route": ROUTE_SELECTED_QSE_FIXED_POOL,
        "davidson_invoked": False,
        "dimension_cap": 64,
        "stride": 2,
        "target_root_count": 6,
        "requested_regimes": ["weak_weak"],
        "acquired_regimes": ["weak_weak"],
        "expected_cell_count": 1,
        "selection_policy": dict(SELECTION_POLICY),
        "selection_policy_digest": canonical_digest(SELECTION_POLICY),
        "summary": summarize_terminal_records(records),
        "records": records,
        "source_provenance": {
            "relevant_path_sha256": {path: "ab" * 32 for path in RELEVANT_SOURCE_PATHS},
            "relevant_source_tree_clean": True,
        },
    }


def test_full_evidence_validator_rejects_route_mixing_and_summary_drift() -> None:
    payload = _valid_payload()
    validate_failure_evidence(payload, require_clean_sources=True)

    payload["records"][0]["trace"].append({"phase": "synthesis", "k": 4})
    with pytest.raises(ValueError, match="synthesis"):
        validate_failure_evidence(payload)


def test_full_evidence_validator_requires_clean_source_when_locking() -> None:
    payload = _valid_payload()
    payload["source_provenance"]["relevant_source_tree_clean"] = False
    with pytest.raises(ValueError, match="dirty"):
        validate_failure_evidence(payload, require_clean_sources=True)


def test_full_evidence_validator_recomputes_terminal_status() -> None:
    payload = _valid_payload()
    payload["records"][0]["algorithm_completion"]["terminal_status"] = (
        STATUS_TARGET_WINDOW_CERTIFIED
    )
    payload["summary"] = summarize_terminal_records(payload["records"])

    with pytest.raises(ValueError, match="does not recompute"):
        validate_failure_evidence(payload)


def test_full_evidence_validator_binds_certificate_to_trace_support() -> None:
    payload = _valid_payload()
    payload["records"][0]["terminal_support_indices"] = [3, 4, 5]
    payload["records"][0]["algorithm_completion"][
        "terminal_support_certificate"
    ]["support_indices"] = [3, 4, 5]

    with pytest.raises(ValueError, match="terminal_support_indices"):
        validate_failure_evidence(payload)


def test_full_evidence_validator_rejects_policy_and_root_count_drift() -> None:
    payload = _valid_payload()
    payload["selection_policy"] = {
        **payload["selection_policy"],
        "geometry_residual_stop": 1.0e-2,
    }
    payload["selection_policy_digest"] = canonical_digest(
        payload["selection_policy"]
    )
    with pytest.raises(ValueError, match="locked selection policy"):
        validate_failure_evidence(payload)

    payload = _valid_payload()
    payload["target_root_count"] = 5
    with pytest.raises(ValueError, match="locked target-root count"):
        validate_failure_evidence(payload)
