from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.scaffold import hh_continuation_scoring as scoring
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
)


def _acquired_geometry() -> dict[str, object]:
    return {
        "schema": "phase2_joint_geometry_reuse_v1",
        "append_position": 1,
        "G_AA": [[1.0]],
        "G_AB": [0.25],
        "G_BB": 1.0,
        "H_AA": [[1.0]],
        "H_AB": [0.5],
        "H_BB": 1.0,
        "descent_gradient": 0.5,
    }


def _scaffold() -> SimpleNamespace:
    return SimpleNamespace(
        old_old_geometry_measured=True,
        old_old_metric_measured=True,
        old_old_hessian_measured=True,
        old_old_hessian_status="measured",
        old_old_hessian_fingerprint="hessian",
        old_old_hessian_provenance={
            "source": "test",
            "measured": True,
        },
        refit_window_indices=(0,),
        state_reconstruction_delta_norm=0.0,
        dpsi_window=(np.asarray([1.0 + 0.0j]),),
        hpsi_state=np.asarray([2.0 + 0.0j]),
        state_fingerprint="state",
        ordered_scaffold_fingerprint="scaffold",
        theta_fingerprint="theta",
    )


def test_stationary_and_measured_gradient_policies_have_distinct_live_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = EstimatorCallLedger()
    gradient_key = EstimatorCallKey(
        projective_state_fingerprint="test:state",
        hamiltonian_fingerprint="test:hamiltonian",
        backend_fingerprint="test:exact_backend",
        precision_contract="test:float64_exact",
        primitive_kind="energy_gradient",
        observable_or_formula_identity="active_residual_contraction_v1",
    )
    original_vdot = np.vdot

    def _measured_active_residual(
        left: np.ndarray,
        right: np.ndarray,
    ) -> complex:
        ledger.record_call(
            gradient_key,
            component="N_grad",
            consumer_scope="phase3_active_gradient",
        )
        return complex(original_vdot(left, right))

    monkeypatch.setattr(
        scoring,
        "_compiled_polynomial_fingerprint",
        lambda _compiled: "hamiltonian",
    )
    monkeypatch.setattr(
        scoring,
        "_candidate_coordinate_fingerprint",
        lambda _term, *, position_id: f"candidate:{position_id}",
    )
    monkeypatch.setattr(scoring.np, "vdot", _measured_active_residual)

    stationary = scoring._promote_fresh_phase3_joint_geometry_receipt(
        acquired_payload=_acquired_geometry(),
        scaffold_context=_scaffold(),
        candidate_term=object(),
        position_id=1,
        h_compiled=object(),
        state_consistency_tolerance=1.0e-12,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
    )
    stationary_ledger = ledger.to_payload()
    assert stationary["active_gradient_indices_acquired"] == []
    assert stationary["g_A"] == [0.0]
    assert stationary["active_gradient_source"] == (
        "not_acquired_stationary_source_protocol"
    )
    assert stationary["G_AB"] == [0.25]
    assert stationary["H_AB"] == [0.5]
    assert stationary_ledger["occurrences"] == []
    assert stationary_ledger["occurrence_summary"][
        "component_occurrence_counts"
    ]["N_grad"] == 0

    measured = scoring._promote_fresh_phase3_joint_geometry_receipt(
        acquired_payload=_acquired_geometry(),
        scaffold_context=_scaffold(),
        candidate_term=object(),
        position_id=1,
        h_compiled=object(),
        state_consistency_tolerance=1.0e-12,
        active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
    )
    measured_ledger = ledger.to_payload()
    assert measured["active_gradient_indices_acquired"] == [0]
    assert measured["g_A"] == pytest.approx([-4.0])
    assert measured["active_gradient_source"] == (
        "measured_active_residual_response_v1"
    )
    assert measured["G_AB"] == stationary["G_AB"]
    assert measured["H_AB"] == stationary["H_AB"]
    assert len(measured_ledger["occurrences"]) == 1
    assert measured_ledger["occurrence_summary"][
        "component_occurrence_counts"
    ]["N_grad"] == 1
