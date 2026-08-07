from __future__ import annotations

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase1CompileCostOracle,
    Phase2CurvatureOracle,
    OrderedInsertionGeometryOracle,
    SimpleScoreConfig,
    build_candidate_features,
    build_full_candidate_features,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    EXACT_ORDERED_INSERTION_CHART,
)
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def _term(label: str):
    return type(
        "_LockAnsatzTerm",
        (),
        {
            "label": label,
            "polynomial": PauliPolynomial(
                "JW", [PauliTerm(len(label), ps=label, pc=1.0)]
            ),
        },
    )()


def _feature(*, position: int):
    compile_cost = Phase1CompileCostOracle().estimate(
        candidate_term_count=1,
        position_id=position,
        append_position=1,
        refit_active_count=1,
    )
    return build_candidate_features(
        stage_name="core",
        candidate_label="y",
        candidate_family="lock",
        candidate_pool_index=0,
        position_id=position,
        append_position=1,
        positions_considered=[position, 1],
        gradient_signed=0.3,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=compile_cost,
        measurement_stats=MeasurementCacheAudit().estimate(["y"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=True,
        trough_detected=True,
        cfg=SimpleScoreConfig(),
    )


def _scaffold():
    state = np.asarray([1.0, 0.0], dtype=complex)
    h_compiled = compile_polynomial_action(
        _term("z").polynomial, pauli_action_cache={}
    )
    h_state = apply_compiled_polynomial(state, h_compiled)
    context = OrderedInsertionGeometryOracle().prepare_scaffold_context(
        selected_ops=[_term("x")],
        theta=np.asarray([0.0], dtype=float),
        psi_ref=state,
        psi_state=state,
        h_compiled=h_compiled,
        hpsi_state=h_state,
        refit_window_indices=[0],
        pauli_action_cache={},
    )
    return context, h_compiled


def _build(*, fresh: bool):
    context, h_compiled = _scaffold()
    return build_full_candidate_features(
        base_feature=_feature(position=0),
        candidate_term=_term("y"),
        cfg=FullScoreConfig(shortlist_size=2),
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=context,
        phase2_scaffold_context=context,
        phase3_scaffold_context=context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
        emit_fresh_phase3_joint_geometry_receipt=fresh,
    )


def test_lock_records_historical_append_chart_and_exact_repair_lever() -> None:
    historical = dict(_build(fresh=False).phase2_joint_geometry_reuse or {})
    repaired = dict(_build(fresh=True).phase2_joint_geometry_reuse or {})

    assert historical["schema"] == "phase2_joint_geometry_reuse_v1"
    assert (
        historical["coordinate_chart"]
        == "append_candidate_after_current_ansatz_v1"
    )
    assert historical["candidate_position_id"] == 0
    assert historical["append_position"] == 1

    assert repaired["schema"] == "phase2_joint_geometry_reuse_v2"
    assert repaired["coordinate_chart"] == EXACT_ORDERED_INSERTION_CHART
    assert repaired["candidate_position_id"] == 0
    assert repaired["append_position"] == 1
    assert repaired["status"] == "populated"
    assert (
        repaired["acquisition_mode"]
        == "fresh_projected_phase3_population_v1"
    )
    assert (
        repaired["acquisition_authority"]
        == "fresh_projected_phase3_child_measurement_v1"
    )
    assert repaired["cross_outer_iteration_reuse_permitted"] is False
