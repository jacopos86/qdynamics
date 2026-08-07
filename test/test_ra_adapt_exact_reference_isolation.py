from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt import (
    AppendAdaptRequest,
    RAAdaptRequest,
    SinglePauliWordCandidateAdapter,
    run_append_adapt,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.append import (
    APPEND_ADAPT_ALGORITHM_ID,
    build_resolved_append_protocol,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RESOURCE_WEIGHTING_LATE,
    _attach_validated_bundle_protocol_authority,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_ALGORITHM_ID,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
    STUDY1_EXACT_REFERENCE_EVENT_PHASE,
    build_study1_trusted_execution_receipt,
    validate_study1_exact_reference_isolation_receipt,
    validate_study1_trusted_execution_receipt,
)
from pipelines.static_adapt.sr_snake import (
    SRExecutionPolicy,
    SRStopPolicy,
)


def _problem(*, reporting_exact_energy: float) -> Any:
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )
    return replace(
        problem,
        exact_target=replace(
            problem.exact_target,
            resolve_energy=lambda **_kwargs: float(reporting_exact_energy),
        ),
    )


def _source_lock_refs() -> dict[str, str]:
    return {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "fixture_lock",
        "cell_source_lock_sha256": "3" * 64,
        "visible_provenance_sha256": "4" * 64,
        "provenance_tracker_sha256": "5" * 64,
        "ed_cutoff_reference_sha256": "6" * 64,
        "resolver_script_sha256": "7" * 64,
    }


def _study1_protocol(problem: Any, *, method: str) -> Any:
    execution = SRExecutionPolicy(
        stop=SRStopPolicy(maximum_controller_rounds=1)
    )
    if method == "ra_adapt":
        request: Any = RAAdaptRequest(
            adapter=SinglePauliWordCandidateAdapter(),
            execution=execution,
        )
        algorithm_id = RA_ADAPT_ALGORITHM_ID
        selector_family = "ra_adapt"
        build = build_resolved_ra_protocol
    else:
        request = AppendAdaptRequest(
            adapter=SinglePauliWordCandidateAdapter(),
            execution=execution,
        )
        algorithm_id = APPEND_ADAPT_ALGORITHM_ID
        selector_family = "append_adapt"
        build = build_resolved_append_protocol
    cell = bundle_module.BundleCellSpec(
        cell_id=f"g8_{method}_fixture",
        stage="validation",
        regime_id="g8_fixture",
        nph=1,
        route_id=(
            bundle_module.ROUTE_RA_SINGLETON_APPEND_ONLY
            if method == "ra_adapt"
            else bundle_module.ROUTE_APPEND_SINGLETON
        ),
        algorithm_id=algorithm_id,
        selector_family=selector_family,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        horizon=1,
        source_lock_id="fixture_lock",
    )
    authority_kwargs = {
        "cell": cell,
        "bundle_id": bundle_module.MEASURED_BUNDLE_ID,
        "bundle_manifest_sha256": "8" * 64,
        "source_locks_sha256": "1" * 64,
        "source_lock_refs": _source_lock_refs(),
        "active_gradient_policy": ACTIVE_GRADIENT_MEASURED,
        "resource_weighting_scope": RESOURCE_WEIGHTING_LATE,
    }
    protocol = build(
        problem,
        request,
        materialization_authority=(
            bundle_module._bundle_protocol_materialization_authority(
                **authority_kwargs
            )
        ),
    )
    bound = bundle_module._bundle_protocol_materialization_authority(
        **authority_kwargs,
        protocol_sha256=protocol.sha256,
    )
    return _attach_validated_bundle_protocol_authority(protocol, bound)


def _controller_signature(result: Any, *, method: str) -> dict[str, Any]:
    if method == "ra_adapt":
        return {
            "accepted_trajectory": [
                row.to_dict() for row in result.accepted_trajectory
            ],
            "scientific_replay": [
                row.to_dict() for row in result.run.scientific_replay
            ],
            "controller_replay_evidence": result.scientific_receipts[
                "controller_replay_evidence"
            ],
        }
    return {
        "accepted_labels": result.result_payload[
            "accepted_operator_labels"
        ],
        "history": result.result_payload["history"],
        "controller_replay_evidence": result.scientific_receipts[
            "controller_replay_evidence"
        ],
    }


def test_study1_trusted_execution_receipt_reverifies_source_and_fails_closed(
) -> None:
    receipt = build_study1_trusted_execution_receipt().to_dict()
    assert validate_study1_trusted_execution_receipt(receipt) == receipt
    assert receipt["source_dataflow_regression_passed"] is True
    assert receipt["source_dataflow_regression_test_id"]
    assert receipt["source_dataflow_regression_receipt_sha256"]

    drifted = dict(receipt)
    drifted["controller_exact_reference_inputs"] = ["exact_energy"]
    with pytest.raises(ValueError):
        validate_study1_trusted_execution_receipt(
            drifted, reverify_source=False
        )


@pytest.mark.parametrize("method", ("ra_adapt", "append_adapt"))
def test_study1_reporting_reference_differential_preserves_controller_trajectory_and_replay_v1(
    method: str,
) -> None:
    first_problem = _problem(reporting_exact_energy=-1.25)
    second_problem = _problem(reporting_exact_energy=-7.5)
    protocol = _study1_protocol(first_problem, method=method)
    execute = run_ra_adapt if method == "ra_adapt" else run_append_adapt

    first = execute(first_problem, protocol)
    second = execute(second_problem, protocol)

    assert _controller_signature(first, method=method) == (
        _controller_signature(second, method=method)
    )
    first_g8 = first.scientific_receipts[
        "study1_g8_exact_reference_isolation"
    ]
    second_g8 = second.scientific_receipts[
        "study1_g8_exact_reference_isolation"
    ]
    assert validate_study1_exact_reference_isolation_receipt(
        first_g8, protocol=protocol
    ) == first_g8
    assert first_g8["controller_consumed_exact_reference"] is False
    assert first_g8["exact_reference_events"][0]["phase"] == (
        STUDY1_EXACT_REFERENCE_EVENT_PHASE
    )
    assert (
        first_g8["exact_reference_events"][0][
            "exact_reference_value_sha256"
        ]
        != second_g8["exact_reference_events"][0][
            "exact_reference_value_sha256"
        ]
    )
