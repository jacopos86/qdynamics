from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import AppendControllerConfig
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    run_append_ap_mclachlan_from_runtime_input,
)
from pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact import (
    run_fixed_ap_mclachlan_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.reference_diagnostics import (
    attach_reference_energy_diagnostics_with_prefix,
    reference_energy_summary,
    reference_energy_trajectory_from_payload,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(label: str, coeff: float = 1.0) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _runtime_input(*candidates: AnsatzTerm) -> ScaffoldRuntimeInput:
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x")),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=build_parameter_layout([]),
        theta_runtime=np.zeros(0, dtype=float),
        theta_logical=np.zeros(0, dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=(),
        candidate_pool_terms=tuple(candidates),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy.json"},
    )


def test_fixed_runner_attaches_cached_reference_energy_errors() -> None:
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        _runtime_input(),
        times=(0.0, 0.1),
        reference_energy_trajectory={
            "points": [
                {"time": 0.0, "energy": 0.0},
                {"time": 0.1, "energy": 0.5},
            ],
            "source": "cached_test_reference",
        },
    )

    assert payload["plot_rows"][0]["reference_energy"] == 0.0
    assert payload["plot_rows"][0]["energy_error"] == payload["plot_rows"][0]["energy_expectation"]
    assert payload["plot_rows"][1]["reference_energy"] == 0.5
    assert payload["plot_rows"][1]["abs_energy_error"] == abs(
        payload["plot_rows"][1]["energy_expectation"] - 0.5
    )
    assert payload["summary"]["reference_energy_diagnostics_enabled"] is True
    assert payload["summary"]["reference_energy_matched_count"] == 2
    assert payload["decision_data_flow"]["uses_exact_reference_for_decision"] is False
    assert payload["decision_data_flow"]["reference_energy_error_scope"] == "post_run_reporting"


def test_append_reference_energy_does_not_change_decision_or_theta() -> None:
    runtime_input = _runtime_input(
        AnsatzTerm(label="candidate_x", polynomial=_poly("x")),
        AnsatzTerm(label="candidate_y", polynomial=_poly("y")),
    )
    no_reference = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        controller_config=AppendControllerConfig(
            max_append_candidates=2,
            append_gain_threshold=0.0,
        ),
    )
    absurd_reference = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        controller_config=AppendControllerConfig(
            max_append_candidates=2,
            append_gain_threshold=0.0,
        ),
        reference_energy_trajectory={
            "points": [
                {"time": 0.0, "energy": 1.0e6},
                {"time": 0.1, "energy": -1.0e6},
            ],
            "source": "absurd_cached_test_reference",
        },
    )

    assert absurd_reference["summary"]["accepted_insert_count"] == no_reference["summary"]["accepted_insert_count"]
    assert absurd_reference["plot_rows"][0]["patch_selected_label"] == no_reference["plot_rows"][0]["patch_selected_label"]
    for key in ("patch_candidate_count", "patch_scored_count", "patch_batch_score_count"):
        assert absurd_reference["plot_rows"][0][key] == no_reference["plot_rows"][0][key]
    no_reference_scores = no_reference["trajectory"]["points"][0]["patch_decision"]["batch_evaluation"]["candidate_scores"]
    absurd_reference_scores = absurd_reference["trajectory"]["points"][0]["patch_decision"]["batch_evaluation"]["candidate_scores"]
    assert absurd_reference_scores == no_reference_scores
    theta_no_ref = [
        point["theta_runtime"]
        for point in no_reference["trajectory"]["points"]
    ]
    theta_absurd_ref = [
        point["theta_runtime"]
        for point in absurd_reference["trajectory"]["points"]
    ]
    assert theta_absurd_ref == theta_no_ref
    assert absurd_reference["decision_data_flow"]["uses_exact_reference_for_decision"] is False
    assert absurd_reference["summary"]["reference_energy_diagnostics_enabled"] is True


def test_reference_energy_missing_time_match_is_report_only() -> None:
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        _runtime_input(),
        times=(0.0, 0.1),
        reference_energy_trajectory={
            "points": [
                {"time": 0.5, "energy": 0.0},
            ],
            "source": "off_grid_reference",
        },
    )

    assert payload["plot_rows"][0]["reference_energy"] is None
    assert payload["plot_rows"][0]["abs_energy_error"] is None
    assert payload["plot_rows"][0]["reference_energy_missing_reason"] == "no_time_match"
    assert payload["summary"]["reference_energy_diagnostics_enabled"] is False
    assert payload["summary"]["reference_energy_reference_provided"] is True
    assert payload["summary"]["reference_energy_unmatched_count"] == 2
    assert payload["decision_data_flow"]["uses_exact_reference_for_decision"] is False


def test_partial_reference_match_does_not_report_final_error() -> None:
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        _runtime_input(),
        times=(0.0, 0.1),
        reference_energy_trajectory={
            "points": [
                {"time": 0.0, "energy": 0.0},
            ],
            "source": "partial_reference",
        },
    )

    assert payload["summary"]["reference_energy_reference_provided"] is True
    assert payload["summary"]["reference_energy_matched_count"] == 1
    assert payload["summary"]["reference_energy_unmatched_count"] == 1
    assert payload["summary"]["final_abs_energy_error"] is None
    assert payload["plot_rows"][-1]["reference_energy_missing_reason"] == "no_time_match"


def test_seed_reference_energy_uses_prefixed_report_fields() -> None:
    rows = [
        {
            "time": 0.0,
            "energy_expectation": 1.25,
            "doublon": 0.2,
            "site_occupations": [1.0, 0.0],
        },
        {
            "time": 0.1,
            "energy_expectation": 1.5,
            "doublon": 0.4,
            "site_occupations": [0.8, 0.2],
        },
    ]
    reference = reference_energy_trajectory_from_payload(
        {
            "points": [
                {
                    "time": 0.0,
                    "energy": 1.0,
                    "observables": {"doublon": 0.25, "site_occupations": [0.9, 0.1]},
                },
                {
                    "time": 0.1,
                    "energy": 1.4,
                    "observables": {"doublon": 0.35, "site_occupations": [0.75, 0.25]},
                },
            ],
            "source": "seed_prepared_state_v1",
        }
    )

    attached = attach_reference_energy_diagnostics_with_prefix(
        plot_rows=rows,
        reference=reference,
        field_prefix="seed_",
    )
    summary = reference_energy_summary(
        attached,
        field_prefix="seed_",
        summary_prefix="seed_",
    )

    assert attached[0]["seed_reference_energy"] == 1.0
    assert attached[0]["seed_abs_energy_error"] == pytest.approx(0.25)
    assert attached[0]["seed_doublon_exact"] == pytest.approx(0.25)
    assert attached[0]["seed_abs_doublon_error"] == pytest.approx(0.05)
    assert attached[1]["seed_site_occupations_exact"] == pytest.approx([0.75, 0.25])
    assert attached[1]["seed_site_occupations_abs_error_max"] == pytest.approx(0.05)
    assert summary["seed_reference_energy_diagnostics_enabled"] is True
    assert summary["seed_reference_energy_matched_count"] == 2
    assert summary["seed_final_abs_energy_error"] == pytest.approx(0.1)
    assert summary["seed_final_abs_doublon_error"] == pytest.approx(0.05)


def test_absent_seed_reference_fields_are_not_reported_as_provided() -> None:
    summary = reference_energy_summary(
        [{"time": 0.0, "energy_expectation": 1.0}],
        field_prefix="seed_",
        summary_prefix="seed_",
    )

    assert summary["seed_reference_energy_reference_provided"] is False
    assert summary["seed_reference_energy_diagnostics_enabled"] is False
    assert summary["seed_reference_energy_matched_count"] == 0


def test_reference_payload_rejects_unsupported_match_kind_and_missing_time() -> None:
    with pytest.raises(ValueError, match="Unsupported reference energy schema"):
        reference_energy_trajectory_from_payload(
            {
                "schema": "wrong_schema",
                "points": [{"time": 0.0, "energy": 0.0}],
            }
        )
    with pytest.raises(ValueError, match="Unsupported reference energy match_kind"):
        reference_energy_trajectory_from_payload(
            {
                "match_kind": "linear",
                "points": [{"time": 0.0, "energy": 0.0}],
            }
        )
    with pytest.raises(ValueError, match="missing `time`"):
        reference_energy_trajectory_from_payload(
            {
                "points": [{"energy": 0.0}],
            }
        )
