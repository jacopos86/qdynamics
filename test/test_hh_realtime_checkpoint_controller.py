from __future__ import annotations

from dataclasses import replace as dataclass_replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.noise_oracle_runtime import OracleConfig
from pipelines.hardcoded.hh_realtime_checkpoint_controller import (
    ControllerDriveConfig,
    MotionSchedulerTelemetry,
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    CandidateProbeSummary,
    RealtimeCheckpointConfig,
    make_checkpoint_context,
)
from pipelines.hardcoded.hh_realtime_measurement import DerivedGeometryMemo, ExactCheckpointValueCache
from pipelines.hardcoded.hh_realtime_measurement import OracleCheckpointValueCache
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _basis(idx: int) -> np.ndarray:
    out = np.zeros(2, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _toy_context(theta_x: float = 0.2) -> tuple[ReplayScaffoldContext, np.ndarray, np.ndarray, np.ndarray]:
    x_term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    y_term = AnsatzTerm(
        label="op_y",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="y", pc=1.0)]),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    hmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    psi_ref = _basis(0)
    base_layout = build_parameter_layout([x_term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    executor = CompiledAnsatzExecutor(
        [x_term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    best_theta = np.array([float(theta_x)], dtype=float)
    psi_initial = executor.prepare_state(best_theta, psi_ref)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1, L=1, ordering="blocked"),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}},
        family_info={"resolved": "toy_pool"},
        family_pool=(x_term, y_term),
        pool_meta={"candidate_pool_complete": True},
        replay_terms=(x_term,),
        base_layout=base_layout,
        adapt_theta_runtime=np.array([float(theta_x)], dtype=float),
        adapt_theta_logical=np.array([float(theta_x)], dtype=float),
        adapt_depth=1,
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=2,
    )
    return replay_context, h_poly, hmat, psi_initial


def _duplicate_label_context() -> tuple[ReplayScaffoldContext, np.ndarray, np.ndarray, np.ndarray]:
    x_term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    dup_y = AnsatzTerm(
        label="dup",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="y", pc=1.0)]),
    )
    dup_z = AnsatzTerm(
        label="dup",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    hmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    psi_ref = _basis(0)
    base_layout = build_parameter_layout([x_term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    executor = CompiledAnsatzExecutor(
        [x_term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    best_theta = np.array([0.2], dtype=float)
    psi_initial = executor.prepare_state(best_theta, psi_ref)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1, L=1, ordering="blocked"),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}},
        family_info={"resolved": "toy_pool_dup"},
        family_pool=(x_term, dup_y, dup_z),
        pool_meta={"candidate_pool_complete": True},
        replay_terms=(x_term,),
        base_layout=base_layout,
        adapt_theta_runtime=np.array([0.2], dtype=float),
        adapt_theta_logical=np.array([0.2], dtype=float),
        adapt_depth=1,
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=3,
    )
    return replay_context, h_poly, hmat, psi_initial


def test_realtime_controller_appends_candidate_and_hits_same_checkpoint_cache() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
    )
    result = controller.run()

    assert int(result.summary["append_count"]) >= 1
    assert str(result.ledger[0]["action_kind"]) == "append_candidate"
    assert int(result.ledger[0]["exact_cache_misses"]) >= 1
    assert int(result.ledger[0]["geometry_memo_hits"]) >= 1
    assert int(result.summary["final_runtime_parameter_count"]) >= 2


def test_realtime_controller_stays_when_miss_threshold_is_high() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
    )
    result = controller.run()

    assert int(result.summary["append_count"]) == 0
    assert all(str(row["action_kind"]) == "stay" for row in result.ledger)


def test_realtime_controller_writes_progress_file(tmp_path: Path) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    progress_path = tmp_path / "controller_progress.json"
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        progress_path=progress_path,
        progress_every_s=0.0,
    )

    result = controller.run()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    assert progress["stage"] == "run_complete"
    assert progress["status"] == "completed"
    assert progress["summary"]["append_count"] == result.summary["append_count"]
    assert progress["trajectory_points"] == len(result.trajectory)


def test_realtime_controller_writes_partial_payload_file(tmp_path: Path) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    partial_payload_path = tmp_path / "controller_partial.json"
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        partial_payload_path=partial_payload_path,
    )

    result = controller.run()
    partial = json.loads(partial_payload_path.read_text(encoding="utf-8"))

    assert partial["stage"] == "run_complete"
    assert partial["status"] == "completed"
    assert len(partial["trajectory"]) == len(result.trajectory)
    assert len(partial["ledger"]) == len(result.ledger)
    assert partial["summary"]["append_count"] == result.summary["append_count"]


def test_oracle_commit_override_rejects_bootstrap_negative_noisy_improvement() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1e-6,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="shots", shots=32, oracle_repeats=1, oracle_aggregate="mean"),
    )

    reason = controller._oracle_commit_override_reason(
        motion=MotionSchedulerTelemetry(
            regime="bootstrap",
            direction_cosine=None,
            rate_change_l2=None,
            rate_change_ratio=None,
            acceleration_l2=None,
            curvature_cosine=None,
            direction_reversal=False,
            curvature_sign_flip=False,
            kink_score=0.0,
        ),
        selected={"gain_ratio": 0.049},
        action_kind="append_candidate",
        oracle_commit_payload={"selected_noisy_improvement_abs": -0.375},
        predicted_displacement=0.09,
        runtime_parameter_count_before=2,
    )

    assert reason == "bootstrap_negative_noisy_commit"


def test_oracle_commit_override_rejects_kink_negative_noisy_improvement_even_with_strong_exact_gain() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1e-6,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="shots", shots=32, oracle_repeats=1, oracle_aggregate="mean"),
    )

    reason = controller._oracle_commit_override_reason(
        motion=MotionSchedulerTelemetry(
            regime="kink",
            direction_cosine=-0.9,
            rate_change_l2=0.4,
            rate_change_ratio=2.5,
            acceleration_l2=0.5,
            curvature_cosine=-0.6,
            direction_reversal=True,
            curvature_sign_flip=True,
            kink_score=1.4,
        ),
        selected={"gain_ratio": 0.34118721281858394},
        action_kind="append_candidate",
        oracle_commit_payload={"selected_noisy_improvement_abs": -2.3690836180540353},
        predicted_displacement=0.04,
        runtime_parameter_count_before=2,
    )

    assert reason == "kink_negative_noisy_commit"


def test_oracle_commit_override_rejects_late_kink_reappend_with_large_displacement() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1e-6,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="shots", shots=32, oracle_repeats=1, oracle_aggregate="mean"),
    )

    reason = controller._oracle_commit_override_reason(
        motion=MotionSchedulerTelemetry(
            regime="kink",
            direction_cosine=-0.8,
            rate_change_l2=1.1,
            rate_change_ratio=1.7,
            acceleration_l2=1.1,
            curvature_cosine=-0.7,
            direction_reversal=True,
            curvature_sign_flip=True,
            kink_score=1.7,
        ),
        selected={"gain_ratio": 1.9426707547767408},
        action_kind="append_candidate",
        oracle_commit_payload={
            "selected_noisy_improvement_abs": 0.5000000000001719,
            "selected_noisy_improvement_ratio": 0.2500000000001077,
        },
        predicted_displacement=2.9203355642971713,
        runtime_parameter_count_before=3,
    )

    assert reason == "kink_large_displacement_commit"


def test_oracle_commit_override_rejects_first_kink_append_with_weak_noisy_margin() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1e-6,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="shots", shots=32, oracle_repeats=1, oracle_aggregate="mean"),
    )

    reason = controller._oracle_commit_override_reason(
        motion=MotionSchedulerTelemetry(
            regime="kink",
            direction_cosine=-0.7,
            rate_change_l2=0.45,
            rate_change_ratio=1.2,
            acceleration_l2=0.45,
            curvature_cosine=-0.4,
            direction_reversal=True,
            curvature_sign_flip=True,
            kink_score=0.72,
        ),
        selected={"gain_ratio": 2.9901167415300542},
        action_kind="append_candidate",
        oracle_commit_payload={
            "selected_noisy_improvement_abs": 0.37499999999999956,
            "selected_noisy_improvement_ratio": 0.150000128648925,
        },
        predicted_displacement=0.10118187545377694,
        runtime_parameter_count_before=2,
    )

    assert reason == "kink_weak_margin_first_append"


def test_exact_forecast_override_reason_rejects_dual_metric_regression() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            exact_forecast_guardrail_mode="dual_metric_v1",
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="shots", shots=32, oracle_repeats=1, oracle_aggregate="mean"),
    )

    reason = controller._exact_forecast_override_reason(
        stay_forecast={
            "fidelity_exact_next": 0.80,
            "abs_energy_total_error_next": 0.10,
        },
        selected_forecast={
            "fidelity_exact_next": 0.75,
            "abs_energy_total_error_next": 0.15,
        },
    )

    assert reason == "exact_forecast_dual_metric_regression"


def test_exact_forecast_override_reason_allows_single_metric_trade() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            exact_forecast_guardrail_mode="dual_metric_v1",
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="shots", shots=32, oracle_repeats=1, oracle_aggregate="mean"),
    )

    reason = controller._exact_forecast_override_reason(
        stay_forecast={
            "fidelity_exact_next": 0.80,
            "abs_energy_total_error_next": 0.10,
        },
        selected_forecast={
            "fidelity_exact_next": 0.79,
            "abs_energy_total_error_next": 0.08,
        },
    )

    assert reason is None


def test_oracle_commit_payload_reuses_measured_baseline_for_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1e-6,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="shots", shots=32, oracle_repeats=1, oracle_aggregate="mean"),
    )

    monkeypatch.setattr(
        controller,
        "_oracle_energy_estimate",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not estimate stay energy twice")),
    )

    payload, degraded_reason = controller._oracle_commit_payload(
        checkpoint_ctx=SimpleNamespace(checkpoint_index=0, checkpoint_id="cp0"),
        oracle_cache=SimpleNamespace(),
        raw_group_pool=None,
        baseline={
            "summary": SimpleNamespace(energy=1.2345),
            "backend_info": {"noise_mode": "backend_scheduled"},
            "observable_estimates": {"baseline": {"mean": 1.2345}},
            "theta_dot_step": np.zeros_like(controller.current_theta),
        },
        selected=None,
        action_kind="stay",
        dt=0.1,
        oracle_observable=None,
        budget_scale=1.0,
    )

    assert degraded_reason is None
    assert payload["stay_noisy_energy_mean"] == pytest.approx(1.2345)
    assert payload["selected_noisy_energy_mean"] == pytest.approx(1.2345)
    assert payload["selected_noisy_improvement_abs"] == pytest.approx(0.0)
    assert payload["selected_noisy_improvement_ratio"] == pytest.approx(0.0)


def test_incremental_candidate_gain_matches_full_augmented_recompute() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    cfg = RealtimeCheckpointConfig(
        mode="exact_v1",
        miss_threshold=0.0,
        gain_ratio_threshold=1e-12,
        append_margin_abs=1e-12,
        regularization_lambda=1e-8,
        candidate_regularization_lambda=1e-8,
    )
    controller = RealtimeCheckpointController(
        cfg=cfg,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=controller.current_theta,
        psi=controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_pool",
        grouping_mode=str(cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline = controller._baseline_geometry(checkpoint_ctx, cache, geometry_memo)
    shortlist = controller._scout_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline,
        predicted_displacement=0.0,
    )
    confirmed = controller._confirm_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline,
        shortlist=shortlist,
    )
    best = max(confirmed, key=lambda rec: float(rec["gain_exact"]))

    T = np.asarray(baseline["T"], dtype=complex)
    U_cols = [
        np.asarray(best["candidate_data"]["raw_tangents"][idx], dtype=complex)
        - complex(np.vdot(baseline["psi"], best["candidate_data"]["raw_tangents"][idx])) * np.asarray(baseline["psi"], dtype=complex)
        for idx in best["candidate_data"]["runtime_block_indices"]
    ]
    U = np.column_stack(U_cols)
    G = np.asarray(np.real(T.conj().T @ T), dtype=float)
    B = np.asarray(np.real(T.conj().T @ U), dtype=float)
    C = np.asarray(np.real(U.conj().T @ U), dtype=float)
    f = np.asarray(baseline["f"], dtype=float)
    q = np.asarray(np.real(U.conj().T @ baseline["b_bar"]), dtype=float).reshape(-1)
    K = np.asarray(G + float(cfg.regularization_lambda) * np.eye(int(G.shape[0])), dtype=float)
    full_K = np.block(
        [
            [K, B],
            [B.T, C + float(cfg.candidate_regularization_lambda) * np.eye(int(C.shape[0]))],
        ]
    )
    full_f = np.concatenate([f, q])
    baseline_value = float(f @ baseline["theta_dot_step"])
    theta_dot_full = np.linalg.pinv(full_K, rcond=float(cfg.pinv_rcond)) @ full_f
    full_value = float(full_f @ theta_dot_full)
    incremental_gain_full = float(full_value - baseline_value)

    assert abs(incremental_gain_full - float(best["gain_exact"])) < 1e-8


def test_scout_candidates_use_unique_candidate_identity_when_labels_repeat() -> None:
    replay_context, h_poly, hmat, psi_initial = _duplicate_label_context()
    cfg = RealtimeCheckpointConfig(
        mode="exact_v1",
        miss_threshold=0.0,
        shortlist_size=8,
        shortlist_fraction=1.0,
    )
    controller = RealtimeCheckpointController(
        cfg=cfg,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=controller.current_theta,
        psi=controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_pool_dup",
        grouping_mode=str(cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline = controller._baseline_geometry(checkpoint_ctx, cache, geometry_memo)
    shortlist = controller._scout_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline,
        predicted_displacement=0.0,
    )

    dup_identities = {
        str(item["candidate_identity"])
        for item in shortlist
        if str(item["candidate_label"]) == "dup"
    }
    assert dup_identities == {"dup__pool1", "dup__pool2"}


def test_motion_telemetry_detects_calm_reversal_and_curvature_flip() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )

    bootstrap = controller._motion_telemetry(theta_dot=np.array([1.0, 0.0]), predicted_displacement=0.01)
    assert str(bootstrap.regime) == "bootstrap"

    controller._record_theta_dot_history(np.array([1.0, 0.0]))
    calm = controller._motion_telemetry(theta_dot=np.array([1.01, 0.01]), predicted_displacement=0.01)
    assert str(calm.regime) == "calm"
    assert calm.direction_reversal is False
    assert calm.curvature_sign_flip is False

    reversal = controller._motion_telemetry(theta_dot=np.array([-1.0, 0.0]), predicted_displacement=0.20)
    assert str(reversal.regime) == "kink"
    assert reversal.direction_reversal is True

    controller._theta_dot_history = []
    controller._record_theta_dot_history(np.array([0.0, 0.0]))
    steady_departure = controller._motion_telemetry(theta_dot=np.array([0.01, 0.0]), predicted_displacement=0.01)
    assert str(steady_departure.regime) == "steady"
    assert steady_departure.direction_reversal is False

    controller._theta_dot_history = []
    controller._record_theta_dot_history(np.array([0.0, 0.0]))
    controller._record_theta_dot_history(np.array([1.0, 0.0]))
    curvature_flip = controller._motion_telemetry(theta_dot=np.array([0.0, 0.0]), predicted_displacement=0.02)
    assert str(curvature_flip.regime) == "kink"
    assert curvature_flip.curvature_sign_flip is True


def test_motion_scheduler_policy_scales_shortlist_confirm_and_budget() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", shortlist_size=4, shortlist_fraction=0.4),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )

    calm = MotionSchedulerTelemetry(
        regime="calm",
        direction_cosine=0.999,
        rate_change_l2=0.01,
        rate_change_ratio=0.01,
        acceleration_l2=0.0,
        curvature_cosine=1.0,
        direction_reversal=False,
        curvature_sign_flip=False,
        kink_score=0.01,
    )
    kink = MotionSchedulerTelemetry(
        regime="kink",
        direction_cosine=-1.0,
        rate_change_l2=1.5,
        rate_change_ratio=1.5,
        acceleration_l2=1.0,
        curvature_cosine=-1.0,
        direction_reversal=True,
        curvature_sign_flip=True,
        kink_score=1.5,
    )

    calm_cfg = controller._shortlist_cfg_for_motion(calm)
    kink_cfg = controller._shortlist_cfg_for_motion(kink)

    assert int(calm_cfg.shortlist_size) < int(controller._shortlist_cfg.shortlist_size)
    assert float(calm_cfg.shortlist_fraction) < float(controller._shortlist_cfg.shortlist_fraction)
    assert int(kink_cfg.shortlist_size) > int(controller._shortlist_cfg.shortlist_size)
    assert float(kink_cfg.shortlist_fraction) >= float(controller._shortlist_cfg.shortlist_fraction)
    assert int(controller._oracle_confirm_limit_for_motion(confirmed_count=3, refresh_pressure="low", motion=calm)) == 1
    assert int(controller._oracle_confirm_limit_for_motion(confirmed_count=3, refresh_pressure="high", motion=kink)) == 3
    assert float(controller._oracle_budget_scale_for_motion(refresh_pressure="low", motion=calm)) == pytest.approx(
        float(controller.cfg.motion_calm_oracle_budget_scale)
    )
    assert float(controller._oracle_budget_scale_for_motion(refresh_pressure="high", motion=kink)) == pytest.approx(
        float(controller.cfg.motion_kink_oracle_budget_scale)
    )


def test_realtime_controller_drive_step_hamiltonian_uses_time_dependent_total_h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)

    class _FakeDrive:
        @staticmethod
        def coeff_map_exyz(time_value: float) -> dict[str, float]:
            return {"z": float(time_value)}

    monkeypatch.setattr(
        "pipelines.hardcoded.hh_realtime_checkpoint_controller.build_gaussian_sinusoid_density_drive",
        lambda **kwargs: _FakeDrive(),
    )

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        drive_config=ControllerDriveConfig(
            enabled=True,
            n_sites=1,
            ordering="blocked",
            drive_A=0.6,
            drive_omega=1.0,
            drive_tbar=1.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_weights=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=1,
        ),
    )

    step0 = controller._step_hamiltonian_artifacts(0.0)
    step1 = controller._step_hamiltonian_artifacts(0.1)

    assert int(step0.drive_term_count) == 0
    assert int(step1.drive_term_count) == 1
    assert np.allclose(np.asarray(step1.hmat, dtype=complex), np.asarray([[1.1, 0.0], [0.0, -1.1]], dtype=complex))

    result = controller.run()

    assert str(result.reference["kind"]) == "driven_piecewise_constant_reference_from_replay_seed"
    assert result.reference["drive_profile"]["A"] == pytest.approx(0.6)
    assert any(int(row.get("drive_term_count", 0)) >= 1 for row in result.ledger[1:])
    assert all("physical_time" in row for row in result.ledger)
    assert all("staggered" in row and "staggered_exact" in row for row in result.trajectory)
    assert all("doublon" in row and "doublon_exact" in row for row in result.trajectory)
    assert all("site_occupations" in row and "site_occupations_exact" in row for row in result.trajectory)
    assert "max_abs_staggered_error" in result.summary
    assert "max_abs_doublon_error" in result.summary
    assert "max_abs_site_occupations_error" in result.summary


def test_realtime_controller_off_mode_produces_stay_only_driven_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)

    class _FakeDrive:
        @staticmethod
        def coeff_map_exyz(time_value: float) -> dict[str, float]:
            return {"z": float(time_value)}

    monkeypatch.setattr(
        "pipelines.hardcoded.hh_realtime_checkpoint_controller.build_gaussian_sinusoid_density_drive",
        lambda **kwargs: _FakeDrive(),
    )

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="off"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        drive_config=ControllerDriveConfig(
            enabled=True,
            n_sites=1,
            ordering="blocked",
            drive_A=1.0,
            drive_omega=1.0,
            drive_tbar=1.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_weights=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=1,
        ),
    )

    result = controller.run()

    assert str(result.summary["mode"]) == "off"
    assert str(result.summary["decision_backend"]) == "off"
    assert list(result.summary["executed_decision_backends"]) == ["off"]
    assert int(result.summary["append_count"]) == 0
    assert int(result.summary["stay_count"]) == 3
    assert all(str(row["decision_backend"]) == "off" for row in result.trajectory)
    assert all(str(row["action_kind"]) == "stay" for row in result.trajectory)
    assert all(row.get("shortlist") == [] for row in result.trajectory)


def test_realtime_controller_off_mode_uses_measured_baseline_when_oracle_surface_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="off"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            shots=32,
            oracle_repeats=1,
        ),
    )

    measured_calls = {"count": 0}

    def _fake_measured_baseline(**kwargs):
        measured_calls["count"] += 1
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(float(kwargs["checkpoint_ctx"].time_start)),
        )
        return {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.25,
                solve_mode="grouped_raw_measured",
            ),
            "backend_info": {"noise_mode": "backend_scheduled"},
            "observable_estimates": {"baseline": {"mean": float(baseline["summary"].energy) + 0.25}},
            "raw_group_pool_summary": {"calls": 1},
        }

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _fake_measured_baseline)

    result = controller.run()

    assert measured_calls["count"] == 1
    assert str(result.summary["mode"]) == "off"
    assert str(result.summary["decision_backend"]) == "off"
    assert str(result.summary["decision_noise_mode"]) == "backend_scheduled"
    assert str(result.summary["oracle_estimate_kind"]) == "oracle_backend_scheduled"
    assert int(result.summary["oracle_attempted_checkpoints"]) == 1
    assert all(str(row["decision_backend"]) == "off" for row in result.trajectory)
    assert all(str(row["action_kind"]) == "stay" for row in result.trajectory)
    assert str(result.trajectory[0]["decision_noise_mode"]) == "backend_scheduled"


def test_realtime_controller_off_mode_uses_measured_baseline_when_shots_oracle_surface_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="off"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="shots",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
    )

    measured_calls = {"count": 0}

    def _fake_measured_baseline(**kwargs):
        measured_calls["count"] += 1
        assert kwargs["raw_group_pool"] is None
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(float(kwargs["checkpoint_ctx"].time_start)),
        )
        return {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.25,
                solve_mode="grouped_oracle_measured",
            ),
            "backend_info": {"noise_mode": "shots"},
            "observable_estimates": {"baseline": {"mean": float(baseline["summary"].energy) + 0.25}},
            "raw_group_pool_summary": {},
        }

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _fake_measured_baseline)

    result = controller.run()

    assert measured_calls["count"] == 1
    assert str(result.summary["mode"]) == "off"
    assert str(result.summary["decision_backend"]) == "off"
    assert str(result.summary["decision_noise_mode"]) == "shots"
    assert str(result.summary["oracle_estimate_kind"]) == "oracle_shots"
    assert int(result.summary["oracle_attempted_checkpoints"]) == 1
    assert int(result.summary["degraded_checkpoints"]) == 0
    assert all(str(row["decision_backend"]) == "off" for row in result.trajectory)
    assert all(str(row["action_kind"]) == "stay" for row in result.trajectory)
    assert str(result.trajectory[0]["decision_noise_mode"]) == "shots"


def test_oracle_for_tier_allows_off_mode_when_oracle_surface_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="off"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            shots=32,
            oracle_repeats=1,
        ),
    )

    class _DummyOracle:
        def __init__(self, cfg):
            self.cfg = cfg

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "pipelines.exact_bench.noise_oracle_runtime.ExpectationOracle",
        _DummyOracle,
    )

    oracle = controller._oracle_for_tier("confirm")

    assert isinstance(oracle, _DummyOracle)
    assert str(oracle.cfg.noise_mode) == "backend_scheduled"


def test_realtime_controller_oracle_v1_shots_uses_direct_measured_geometry_without_raw_group_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="oracle_v1", miss_threshold=0.0, gain_ratio_threshold=2.0),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="shots",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    measured_calls = {"count": 0}

    def _fake_scout_candidates(**kwargs):
        return [
            {
                "candidate_label": "dummy",
                "candidate_pool_index": 0,
                "position_id": 0,
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
                "residual_overlap_l2": 0.0,
                "compile_proxy_total": 1.0,
                "groups_new": 0.0,
                "novelty": None,
                "position_jump_penalty": 0.0,
                "temporal_prior_bonus": 0.0,
                "simple_score": 1.0,
            }
        ]

    def _fake_confirm_candidates(**kwargs):
        return [
                {
                    "candidate_label": "dummy",
                    "candidate_identity": "dummy",
                    "candidate_pool_index": 0,
                    "position_id": 0,
                    "adjusted_gain": 1.0,
                "gain_exact": 1.0,
                "gain_ratio": 1.0,
                "groups_new": 0.0,
                "candidate_summary": CandidateProbeSummary(
                    candidate_label="dummy",
                    candidate_pool_index=0,
                    position_id=0,
                    runtime_insert_position=0,
                    runtime_block_indices=(),
                    residual_overlap_l2=0.0,
                    directional_change_l2=None,
                    gain_exact=1.0,
                    gain_ratio=1.0,
                    compile_proxy_total=1.0,
                    groups_new=0.0,
                    novelty=None,
                    position_jump_penalty=0.0,
                    admissible=True,
                    rejection_reason=None,
                    tier_reached="confirm",
                    decision_metric="measured_incremental_gain_ratio",
                    oracle_estimate_kind="oracle_shots",
                ),
                "candidate_data": {
                    "theta_aug": np.asarray(controller.current_theta, dtype=float).copy(),
                    "aug_layout": controller.current_layout,
                },
                "theta_dot_aug": np.asarray(controller.current_theta, dtype=float) * 0.0,
            }
        ]

    def _fake_confirm_oracle_geometry(**kwargs):
        measured_calls["count"] += 1
        assert kwargs["raw_group_pool"] is None
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(float(kwargs["checkpoint_ctx"].time_start)),
        )
        measured_baseline = {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.1,
                rho_miss=0.5,
                solve_mode="grouped_oracle_measured",
            ),
            "backend_info": {"noise_mode": "shots"},
            "observable_estimates": {"baseline": {"mean": float(baseline["summary"].energy) + 0.1}},
            "raw_group_pool_summary": {},
        }
        return measured_baseline, list(_fake_confirm_candidates()), None

    monkeypatch.setattr(controller, "_scout_candidates", _fake_scout_candidates)
    monkeypatch.setattr(controller, "_confirm_candidates", _fake_confirm_candidates)
    monkeypatch.setattr(controller, "_confirm_candidates_oracle_geometry", _fake_confirm_oracle_geometry)

    result = controller.run()

    assert measured_calls["count"] == 1
    assert str(result.summary["mode"]) == "oracle_v1"
    assert str(result.summary["decision_noise_mode"]) == "shots"
    assert str(result.summary["oracle_estimate_kind"]) == "oracle_shots"
    assert int(result.summary["oracle_decision_checkpoints"]) >= 1
    assert int(result.summary["degraded_checkpoints"]) == 0
    assert any(str(row["decision_backend"]) == "oracle" for row in result.trajectory)


def test_realtime_controller_oracle_v1_policy_reranks_measured_candidates_by_noisy_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            oracle_selection_policy="measured_topk_oracle_energy",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="shots",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    rerank_sizes: list[int] = []

    def _candidate_record(label: str, adjusted_gain: float, gain_exact: float, gain_ratio: float):
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": 0 if label == "candidate_a" else 1,
            "position_id": 0 if label == "candidate_a" else 1,
            "runtime_insert_position": 0,
            "runtime_block_indices": [],
            "adjusted_gain": float(adjusted_gain),
            "gain_exact": float(gain_exact),
            "gain_ratio": float(gain_ratio),
            "groups_new": 0.0,
            "candidate_term": replay_context.family_pool[0 if label == "candidate_a" else 1],
            "candidate_summary": CandidateProbeSummary(
                candidate_label=label,
                candidate_pool_index=0 if label == "candidate_a" else 1,
                position_id=0 if label == "candidate_a" else 1,
                runtime_insert_position=0,
                runtime_block_indices=[],
                residual_overlap_l2=0.0,
                directional_change_l2=None,
                gain_exact=float(gain_exact),
                gain_ratio=float(gain_ratio),
                compile_proxy_total=1.0,
                groups_new=0.0,
                novelty=None,
                position_jump_penalty=0.0,
                admissible=True,
                rejection_reason=None,
                tier_reached="confirm",
                decision_metric="measured_incremental_gain_ratio",
                oracle_estimate_kind="oracle_shots",
            ),
            "candidate_data": {
                "theta_aug": np.asarray(controller.current_theta, dtype=float).copy(),
                "aug_layout": controller.current_layout,
                "aug_executor": controller.current_executor,
                "aug_terms": list(controller.current_terms),
            },
            "theta_dot_aug": np.asarray(controller.current_theta, dtype=float) * 0.0,
        }

    def _fake_scout_candidates(**kwargs):
        return [
            {
                "candidate_label": "candidate_a",
                "candidate_pool_index": 0,
                "position_id": 0,
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
                "residual_overlap_l2": 0.0,
                "compile_proxy_total": 1.0,
                "groups_new": 0.0,
                "novelty": None,
                "position_jump_penalty": 0.0,
                "temporal_prior_bonus": 0.0,
                "simple_score": 2.0,
            },
            {
                "candidate_label": "candidate_b",
                "candidate_pool_index": 1,
                "position_id": 1,
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
                "residual_overlap_l2": 0.0,
                "compile_proxy_total": 1.0,
                "groups_new": 0.0,
                "novelty": None,
                "position_jump_penalty": 0.0,
                "temporal_prior_bonus": 0.0,
                "simple_score": 1.5,
            },
        ]

    def _fake_confirm_candidates(**kwargs):
        return [
            _candidate_record("candidate_a", adjusted_gain=3.0, gain_exact=3.0, gain_ratio=3.0),
            _candidate_record("candidate_b", adjusted_gain=2.0, gain_exact=2.0, gain_ratio=2.0),
        ]

    def _fake_confirm_oracle_geometry(**kwargs):
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(float(kwargs["checkpoint_ctx"].time_start)),
        )
        measured_baseline = {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.1,
                rho_miss=0.5,
                solve_mode="grouped_oracle_measured",
            ),
            "backend_info": {"noise_mode": "shots"},
            "observable_estimates": {"baseline": {"mean": float(baseline["summary"].energy) + 0.1}},
            "raw_group_pool_summary": {},
        }
        return measured_baseline, list(_fake_confirm_candidates()), None

    def _fake_confirm_oracle(**kwargs):
        rerank_sizes.append(len(kwargs["confirmed"]))
        out = []
        for rec in kwargs["confirmed"]:
            row = dict(rec)
            if str(row["candidate_label"]) == "candidate_b":
                row["predicted_noisy_improvement_abs"] = 0.25
                row["predicted_noisy_improvement_ratio"] = 0.25
                row["adjusted_noisy_improvement"] = 0.25
            else:
                row["predicted_noisy_improvement_abs"] = -0.25
                row["predicted_noisy_improvement_ratio"] = -0.25
                row["adjusted_noisy_improvement"] = -0.25
            row["predicted_noisy_energy_mean"] = 0.5
            row["predicted_noisy_energy_stderr"] = 0.0
            row["confirm_backend_info"] = {"noise_mode": "shots"}
            row["confirm_error"] = None
            out.append(row)
        return out, {"mean": 1.0, "stderr": 0.0}, None

    def _fake_commit_payload(**kwargs):
        selected = kwargs["selected"]
        assert selected is not None
        assert str(selected["candidate_label"]) == "candidate_b"
        return (
            {
                "stay_noisy_energy_mean": 1.0,
                "stay_noisy_energy_stderr": 0.0,
                "selected_noisy_energy_mean": 0.75,
                "selected_noisy_energy_stderr": 0.0,
                "selected_noisy_improvement_abs": 0.25,
                "selected_noisy_improvement_ratio": 0.25,
            },
            None,
        )

    monkeypatch.setattr(controller, "_scout_candidates", _fake_scout_candidates)
    monkeypatch.setattr(controller, "_confirm_candidates", _fake_confirm_candidates)
    monkeypatch.setattr(controller, "_confirm_candidates_oracle_geometry", _fake_confirm_oracle_geometry)
    monkeypatch.setattr(controller, "_confirm_candidates_oracle", _fake_confirm_oracle)
    monkeypatch.setattr(controller, "_oracle_confirm_limit_for_motion", lambda **kwargs: 1)
    monkeypatch.setattr(controller, "_oracle_commit_payload", _fake_commit_payload)

    result = controller.run()

    assert rerank_sizes == [2]
    assert int(result.summary["append_count"]) == 1
    assert str(result.summary["oracle_selection_policy"]) == "measured_topk_oracle_energy"
    assert str(result.trajectory[0]["candidate_label"]) == "candidate_b"
    assert str(result.trajectory[0]["selection_metric"]) == "oracle_energy_improvement"
    assert str(result.ledger[0]["candidate_label"]) == "candidate_b"
    assert str(result.ledger[0]["selection_metric"]) == "oracle_energy_improvement"


def test_realtime_controller_oracle_v1_policy_falls_back_to_measured_selection_on_rerank_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            oracle_selection_policy="measured_topk_oracle_energy",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="shots",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    def _candidate_record(label: str, adjusted_gain: float):
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": 0 if label == "candidate_a" else 1,
            "position_id": 0 if label == "candidate_a" else 1,
            "runtime_insert_position": 0,
            "runtime_block_indices": [],
            "adjusted_gain": float(adjusted_gain),
            "gain_exact": float(adjusted_gain),
            "gain_ratio": float(adjusted_gain),
            "groups_new": 0.0,
            "candidate_term": replay_context.family_pool[0 if label == "candidate_a" else 1],
            "candidate_summary": CandidateProbeSummary(
                candidate_label=label,
                candidate_pool_index=0 if label == "candidate_a" else 1,
                position_id=0 if label == "candidate_a" else 1,
                runtime_insert_position=0,
                runtime_block_indices=[],
                residual_overlap_l2=0.0,
                directional_change_l2=None,
                gain_exact=float(adjusted_gain),
                gain_ratio=float(adjusted_gain),
                compile_proxy_total=1.0,
                groups_new=0.0,
                novelty=None,
                position_jump_penalty=0.0,
                admissible=True,
                rejection_reason=None,
                tier_reached="confirm",
                decision_metric="measured_incremental_gain_ratio",
                oracle_estimate_kind="oracle_shots",
            ),
            "candidate_data": {
                "theta_aug": np.asarray(controller.current_theta, dtype=float).copy(),
                "aug_layout": controller.current_layout,
                "aug_executor": controller.current_executor,
                "aug_terms": list(controller.current_terms),
            },
            "theta_dot_aug": np.asarray(controller.current_theta, dtype=float) * 0.0,
        }

    def _fake_confirm_oracle_geometry(**kwargs):
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(float(kwargs["checkpoint_ctx"].time_start)),
        )
        measured_baseline = {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.1,
                rho_miss=0.5,
                solve_mode="grouped_oracle_measured",
            ),
            "backend_info": {"noise_mode": "shots"},
            "observable_estimates": {"baseline": {"mean": float(baseline["summary"].energy) + 0.1}},
            "raw_group_pool_summary": {},
        }
        return measured_baseline, [
            _candidate_record("candidate_a", adjusted_gain=3.0),
            _candidate_record("candidate_b", adjusted_gain=2.0),
        ], None

    def _fake_commit_payload(**kwargs):
        selected = kwargs["selected"]
        assert selected is not None
        assert str(selected["candidate_label"]) == "candidate_a"
        return (
            {
                "stay_noisy_energy_mean": 1.0,
                "stay_noisy_energy_stderr": 0.0,
                "selected_noisy_energy_mean": 0.8,
                "selected_noisy_energy_stderr": 0.0,
                "selected_noisy_improvement_abs": 0.2,
                "selected_noisy_improvement_ratio": 0.2,
            },
            None,
        )

    monkeypatch.setattr(
        controller,
        "_scout_candidates",
        lambda **kwargs: [
            {
                "candidate_label": "candidate_a",
                "candidate_pool_index": 0,
                "position_id": 0,
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
                "residual_overlap_l2": 0.0,
                "compile_proxy_total": 1.0,
                "groups_new": 0.0,
                "novelty": None,
                "position_jump_penalty": 0.0,
                "temporal_prior_bonus": 0.0,
                "simple_score": 2.0,
            },
            {
                "candidate_label": "candidate_b",
                "candidate_pool_index": 1,
                "position_id": 1,
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
                "residual_overlap_l2": 0.0,
                "compile_proxy_total": 1.0,
                "groups_new": 0.0,
                "novelty": None,
                "position_jump_penalty": 0.0,
                "temporal_prior_bonus": 0.0,
                "simple_score": 1.5,
            },
        ],
    )
    monkeypatch.setattr(
        controller,
        "_confirm_candidates",
        lambda **kwargs: [
            _candidate_record("candidate_a", adjusted_gain=3.0),
            _candidate_record("candidate_b", adjusted_gain=2.0),
        ],
    )
    monkeypatch.setattr(controller, "_confirm_candidates_oracle_geometry", _fake_confirm_oracle_geometry)
    monkeypatch.setattr(controller, "_confirm_candidates_oracle", lambda **kwargs: ([], None, "boom"))
    monkeypatch.setattr(controller, "_oracle_confirm_limit_for_motion", lambda **kwargs: 1)
    monkeypatch.setattr(controller, "_oracle_commit_payload", _fake_commit_payload)

    result = controller.run()

    assert int(result.summary["append_count"]) == 1
    assert str(result.trajectory[0]["candidate_label"]) == "candidate_a"
    assert str(result.trajectory[0]["selection_metric"]) == "measured_incremental_gain_ratio"
    assert str(result.trajectory[0]["degraded_reason"]).startswith("oracle_rerank_error:")


def test_realtime_controller_oracle_v1_exact_forecast_guardrail_vetoes_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            exact_forecast_guardrail_mode="dual_metric_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="shots",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    def _candidate_record(label: str, adjusted_gain: float):
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": 0,
            "position_id": 0,
            "runtime_insert_position": 0,
            "runtime_block_indices": [],
            "adjusted_gain": float(adjusted_gain),
            "gain_exact": float(adjusted_gain),
            "gain_ratio": float(adjusted_gain),
            "groups_new": 0.0,
            "candidate_term": replay_context.family_pool[0],
            "candidate_summary": CandidateProbeSummary(
                candidate_label=label,
                candidate_pool_index=0,
                position_id=0,
                runtime_insert_position=0,
                runtime_block_indices=[],
                residual_overlap_l2=0.0,
                directional_change_l2=None,
                gain_exact=float(adjusted_gain),
                gain_ratio=float(adjusted_gain),
                compile_proxy_total=1.0,
                groups_new=0.0,
                novelty=None,
                position_jump_penalty=0.0,
                admissible=True,
                rejection_reason=None,
                tier_reached="confirm",
                decision_metric="measured_incremental_gain_ratio",
                oracle_estimate_kind="oracle_shots",
            ),
            "candidate_data": {
                "theta_aug": np.asarray(controller.current_theta, dtype=float).copy(),
                "aug_layout": controller.current_layout,
                "aug_executor": controller.current_executor,
                "aug_terms": list(controller.current_terms),
            },
            "theta_dot_aug": np.asarray(controller.current_theta, dtype=float) * 0.0,
        }

    def _fake_confirm_oracle_geometry(**kwargs):
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(float(kwargs["checkpoint_ctx"].time_start)),
        )
        measured_baseline = {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.1,
                rho_miss=0.5,
                solve_mode="grouped_oracle_measured",
            ),
            "backend_info": {"noise_mode": "shots"},
            "observable_estimates": {"baseline": {"mean": float(baseline["summary"].energy) + 0.1}},
            "raw_group_pool_summary": {},
        }
        return measured_baseline, [_candidate_record("candidate_a", adjusted_gain=3.0)], None

    def _fake_commit_payload(**kwargs):
        selected = kwargs["selected"]
        assert selected is not None
        return (
            {
                "stay_noisy_energy_mean": 1.0,
                "stay_noisy_energy_stderr": 0.0,
                "selected_noisy_energy_mean": 0.8,
                "selected_noisy_energy_stderr": 0.0,
                "selected_noisy_improvement_abs": 0.2,
                "selected_noisy_improvement_ratio": 0.2,
            },
            None,
        )

    forecast_calls: list[str] = []

    def _fake_exact_step_forecast(**kwargs):
        forecast_calls.append("call")
        if len(forecast_calls) == 1:
            return {
                "fidelity_exact_next": 0.80,
                "abs_energy_total_error_next": 0.10,
                "abs_staggered_error_next": 0.20,
                "abs_doublon_error_next": 0.30,
            }
        return {
            "fidelity_exact_next": 0.70,
            "abs_energy_total_error_next": 0.20,
            "abs_staggered_error_next": 0.25,
            "abs_doublon_error_next": 0.35,
        }

    monkeypatch.setattr(
        controller,
        "_scout_candidates",
        lambda **kwargs: [
            {
                "candidate_label": "candidate_a",
                "candidate_pool_index": 0,
                "position_id": 0,
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
                "residual_overlap_l2": 0.0,
                "compile_proxy_total": 1.0,
                "groups_new": 0.0,
                "novelty": None,
                "position_jump_penalty": 0.0,
                "temporal_prior_bonus": 0.0,
                "simple_score": 2.0,
            }
        ],
    )
    monkeypatch.setattr(
        controller,
        "_confirm_candidates",
        lambda **kwargs: [_candidate_record("candidate_a", adjusted_gain=3.0)],
    )
    monkeypatch.setattr(controller, "_confirm_candidates_oracle_geometry", _fake_confirm_oracle_geometry)
    monkeypatch.setattr(controller, "_oracle_commit_payload", _fake_commit_payload)
    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    result = controller.run()

    assert int(result.summary["append_count"]) == 0
    assert int(result.summary["stay_count"]) == 2
    assert int(result.summary["decision_override_count"]) == 1
    assert int(result.summary["exact_forecast_veto_count"]) == 1
    assert str(result.summary["exact_forecast_guardrail_mode"]) == "dual_metric_v1"
    assert str(result.trajectory[0]["action_kind"]) == "stay"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "append_candidate"
    assert str(result.trajectory[0]["proposed_candidate_label"]) == "candidate_a"
    assert str(result.trajectory[0]["decision_override_reason"]) == "exact_forecast_dual_metric_regression"
    assert float(result.trajectory[0]["forecast_stay_fidelity_exact_next"]) == pytest.approx(0.80)
    assert float(result.trajectory[0]["forecast_selected_abs_energy_total_error_next"]) == pytest.approx(0.20)
    assert str(result.ledger[0]["decision_override_reason"]) == "exact_forecast_dual_metric_regression"


def test_realtime_controller_oracle_v1_exact_forecast_guardrail_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            exact_forecast_guardrail_mode="dual_metric_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="shots",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    def _candidate_record(label: str, adjusted_gain: float):
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": 0,
            "position_id": 0,
            "runtime_insert_position": 0,
            "runtime_block_indices": [],
            "adjusted_gain": float(adjusted_gain),
            "gain_exact": float(adjusted_gain),
            "gain_ratio": float(adjusted_gain),
            "groups_new": 0.0,
            "candidate_term": replay_context.family_pool[0],
            "candidate_summary": CandidateProbeSummary(
                candidate_label=label,
                candidate_pool_index=0,
                position_id=0,
                runtime_insert_position=0,
                runtime_block_indices=[],
                residual_overlap_l2=0.0,
                directional_change_l2=None,
                gain_exact=float(adjusted_gain),
                gain_ratio=float(adjusted_gain),
                compile_proxy_total=1.0,
                groups_new=0.0,
                novelty=None,
                position_jump_penalty=0.0,
                admissible=True,
                rejection_reason=None,
                tier_reached="confirm",
                decision_metric="measured_incremental_gain_ratio",
                oracle_estimate_kind="oracle_shots",
            ),
            "candidate_data": {
                "theta_aug": np.asarray(controller.current_theta, dtype=float).copy(),
                "aug_layout": controller.current_layout,
                "aug_executor": controller.current_executor,
                "aug_terms": list(controller.current_terms),
            },
            "theta_dot_aug": np.asarray(controller.current_theta, dtype=float) * 0.0,
        }

    def _fake_confirm_oracle_geometry(**kwargs):
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(float(kwargs["checkpoint_ctx"].time_start)),
        )
        measured_baseline = {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.1,
                rho_miss=0.5,
                solve_mode="grouped_oracle_measured",
            ),
            "backend_info": {"noise_mode": "shots"},
            "observable_estimates": {"baseline": {"mean": float(baseline["summary"].energy) + 0.1}},
            "raw_group_pool_summary": {},
        }
        return measured_baseline, [_candidate_record("candidate_a", adjusted_gain=3.0)], None

    def _fake_commit_payload(**kwargs):
        selected = kwargs["selected"]
        assert selected is not None
        return (
            {
                "stay_noisy_energy_mean": 1.0,
                "stay_noisy_energy_stderr": 0.0,
                "selected_noisy_energy_mean": 0.8,
                "selected_noisy_energy_stderr": 0.0,
                "selected_noisy_improvement_abs": 0.2,
                "selected_noisy_improvement_ratio": 0.2,
            },
            None,
        )

    monkeypatch.setattr(
        controller,
        "_scout_candidates",
        lambda **kwargs: [
            {
                "candidate_label": "candidate_a",
                "candidate_pool_index": 0,
                "position_id": 0,
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
                "residual_overlap_l2": 0.0,
                "compile_proxy_total": 1.0,
                "groups_new": 0.0,
                "novelty": None,
                "position_jump_penalty": 0.0,
                "temporal_prior_bonus": 0.0,
                "simple_score": 2.0,
            }
        ],
    )
    monkeypatch.setattr(
        controller,
        "_confirm_candidates",
        lambda **kwargs: [_candidate_record("candidate_a", adjusted_gain=3.0)],
    )
    monkeypatch.setattr(controller, "_confirm_candidates_oracle_geometry", _fake_confirm_oracle_geometry)
    monkeypatch.setattr(controller, "_oracle_commit_payload", _fake_commit_payload)
    monkeypatch.setattr(controller, "_exact_step_forecast", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    result = controller.run()

    assert int(result.summary["append_count"]) == 1
    assert int(result.summary["decision_override_count"]) == 0
    assert int(result.summary["exact_forecast_veto_count"]) == 0
    assert str(result.trajectory[0]["action_kind"]) == "append_candidate"
    assert result.trajectory[0]["decision_override_reason"] is None
    assert "RuntimeError: boom" in str(result.trajectory[0]["exact_forecast_error"])
    assert "exact_forecast_error: RuntimeError: boom" in str(result.trajectory[0]["degraded_reason"])


def test_confirm_candidates_oracle_prefers_damped_candidate_step_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            candidate_step_scales=(0.25, 1.0),
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="shots",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.2,
        scaffold_labels=[str(block.candidate_label) for block in controller.current_layout.blocks],
        theta=np.asarray(controller.current_theta, dtype=float),
        psi=np.asarray(
            controller.current_executor.prepare_state(
                controller.current_theta,
                replay_context.psi_ref,
            ),
            dtype=complex,
        ),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family=str(replay_context.family_info.get("resolved", "toy_pool")),
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=True,
    )
    exact_cache = ExactCheckpointValueCache(
        checkpoint_id=checkpoint_ctx.checkpoint_id,
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=checkpoint_ctx.checkpoint_id)
    candidate_data = controller._candidate_executor_data(
        checkpoint_ctx=checkpoint_ctx,
        cache=exact_cache,
        geometry_memo=geometry_memo,
        candidate_term=replay_context.family_pool[1],
        candidate_pool_index=1,
        position_id=1,
    )
    confirmed = [
        {
            "candidate_label": str(replay_context.family_pool[1].label),
            "candidate_identity": f"{replay_context.family_pool[1].label}__pool1",
            "candidate_pool_index": 1,
            "position_id": 1,
            "runtime_insert_position": int(candidate_data["runtime_insert_position"]),
            "runtime_block_indices": list(candidate_data["runtime_block_indices"]),
            "groups_new": 0.0,
            "candidate_data": candidate_data,
            "theta_dot_aug": np.array([0.2, 1.0], dtype=float),
            "theta_dot_aug_existing": np.array([0.2], dtype=float),
            "eta_dot": np.array([1.0], dtype=float),
            "candidate_summary": CandidateProbeSummary(
                candidate_label=str(replay_context.family_pool[1].label),
                candidate_pool_index=1,
                position_id=1,
                runtime_insert_position=int(candidate_data["runtime_insert_position"]),
                runtime_block_indices=list(candidate_data["runtime_block_indices"]),
                residual_overlap_l2=1.0,
                gain_exact=1.0,
                gain_ratio=1.0,
                compile_proxy_total=1.0,
                groups_new=0.0,
                novelty=None,
                position_jump_penalty=0.0,
                directional_change_l2=0.0,
                tier_reached="confirm",
                admissible=True,
                rejection_reason=None,
                decision_metric="measured_incremental_gain_ratio",
                oracle_estimate_kind="oracle_shots",
            ),
        }
    ]
    baseline = {
        "theta_dot_step": np.array([0.2], dtype=float),
    }

    def _fake_oracle_energy_estimate(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        if kwargs["candidate_label"] is None:
            mean = 1.0
        else:
            append_amp = float(theta_runtime[-1])
            mean = 0.6 if append_amp <= 0.3 else 1.4
        return {"mean": float(mean), "stderr": 0.0, "backend_info": {"noise_mode": "shots"}}, False

    monkeypatch.setattr(controller, "_oracle_energy_estimate", _fake_oracle_energy_estimate)

    confirmed_oracle, stay_estimate, degraded_reason = controller._confirm_candidates_oracle(
        checkpoint_ctx=checkpoint_ctx,
        baseline=baseline,
        confirmed=confirmed,
        dt=1.0,
        oracle_cache=OracleCheckpointValueCache(checkpoint_id=checkpoint_ctx.checkpoint_id),
        raw_group_pool=None,
        oracle_observable=None,
        budget_scale=1.0,
    )

    assert degraded_reason is None
    assert stay_estimate is not None
    assert stay_estimate["mean"] == pytest.approx(1.0)
    assert len(confirmed_oracle) == 1
    rec = confirmed_oracle[0]
    assert float(rec["candidate_step_scale"]) == pytest.approx(0.25)
    assert float(rec["predicted_noisy_energy_mean"]) == pytest.approx(0.6)
    assert float(rec["predicted_noisy_improvement_abs"]) == pytest.approx(0.4)
    assert np.asarray(rec["theta_dot_aug"], dtype=float) == pytest.approx(np.array([0.2, 0.25]))
    assert float(rec["candidate_summary"].selected_step_scale) == pytest.approx(0.25)


def test_oracle_sampling_targets_scale_total_shots_even_with_single_repeat() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="oracle_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            shots=128,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    tier_cfg = controller._oracle_tier_configs["confirm"]
    base_total_shots = int(tier_cfg.shots) * max(1, int(tier_cfg.oracle_repeats))
    base_samples = max(1, int(tier_cfg.oracle_repeats))

    calm_total_shots, calm_samples = controller._oracle_sampling_targets(
        tier_name="confirm",
        budget_scale=0.5,
    )
    kink_total_shots, kink_samples = controller._oracle_sampling_targets(
        tier_name="confirm",
        budget_scale=2.0,
    )

    assert int(calm_total_shots) == int(np.ceil(float(base_total_shots) * 0.5))
    assert int(calm_samples) == max(1, int(np.ceil(float(base_samples) * 0.5)))
    assert int(kink_total_shots) == int(np.ceil(float(base_total_shots) * 2.0))
    assert int(kink_samples) == max(1, int(np.ceil(float(base_samples) * 2.0)))


def test_oracle_sampling_targets_floor_measured_baseline_to_base_surface() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="oracle_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            shots=32,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    controller._oracle_tier_configs["confirm"] = dataclass_replace(
        controller._oracle_tier_configs["confirm"],
        shots=8,
        oracle_repeats=1,
    )

    tier_total_shots, tier_samples = controller._oracle_sampling_targets(
        tier_name="confirm",
        budget_scale=1.0,
    )
    baseline_total_shots, baseline_samples = controller._oracle_sampling_targets(
        tier_name="confirm",
        budget_scale=1.0,
        floor_to_base_config=True,
    )

    assert int(tier_total_shots) == 8
    assert int(tier_samples) == 1
    assert int(baseline_total_shots) == 32
    assert int(baseline_samples) == 1


def test_oracle_scheduler_deescalates_on_calm_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor

    replay_context, h_poly, hmat, psi_initial = _duplicate_label_context()
    call_counter = {"count": 0}

    monkeypatch.setattr(
        nor,
        "build_runtime_layout_circuit",
        lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()},
    )
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", lambda poly, tol=1e-12: object())

    class _OracleStub:
        def __init__(self, config):
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "stub",
                    "backend_name": config.backend_name,
                    "using_fake_backend": bool(config.use_fake_backend),
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            call_counter["count"] += 1
            theta = np.asarray(circuit["theta"], dtype=float)
            mean = float(-theta.size)
            return type(
                "EstimateStub",
                (),
                {"mean": mean, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=8,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(noise_mode="ideal", oracle_aggregate="mean"),
        wallclock_cap_s=60,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=controller.current_theta,
        psi=controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_pool_dup",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline = controller._baseline_geometry(checkpoint_ctx, cache, geometry_memo)
    shortlist = controller._scout_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline,
        predicted_displacement=0.0,
    )
    confirmed = controller._confirm_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline,
        shortlist=shortlist,
    )
    assert len(confirmed) >= 2

    seen_counts: list[int] = []
    seen_budget_scales: list[float] = []
    original_confirm_oracle = controller._confirm_candidates_oracle

    def _motion_stub(*, theta_dot, predicted_displacement):
        return MotionSchedulerTelemetry(
            regime="calm",
            direction_cosine=0.999,
            rate_change_l2=0.01,
            rate_change_ratio=0.01,
            acceleration_l2=0.0,
            curvature_cosine=1.0,
            direction_reversal=False,
            curvature_sign_flip=False,
            kink_score=0.01,
        )

    def _confirm_wrapper(*args, **kwargs):
        seen_counts.append(len(kwargs["confirmed"]))
        seen_budget_scales.append(float(kwargs["budget_scale"]))
        return original_confirm_oracle(*args, **kwargs)

    monkeypatch.setattr(controller, "_motion_telemetry", _motion_stub)
    monkeypatch.setattr(controller, "_effective_refresh_pressure", lambda **kwargs: "low")
    monkeypatch.setattr(controller, "_confirm_candidates_oracle", _confirm_wrapper)
    result = controller.run()

    assert int(call_counter["count"]) >= 1
    assert seen_counts and max(seen_counts) == 1
    assert seen_budget_scales and max(seen_budget_scales) == pytest.approx(
        float(controller.cfg.motion_calm_oracle_budget_scale)
    )
    assert any(str(row["motion_regime"]) == "calm" for row in result.ledger)
    assert any(int(row["oracle_confirm_limit"]) == 1 for row in result.ledger if bool(row["oracle_attempted"]))


def test_oracle_scheduler_escalates_on_kink_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor

    replay_context, h_poly, hmat, psi_initial = _duplicate_label_context()
    call_counter = {"count": 0}

    monkeypatch.setattr(
        nor,
        "build_runtime_layout_circuit",
        lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()},
    )
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", lambda poly, tol=1e-12: object())

    class _OracleStub:
        def __init__(self, config):
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "stub",
                    "backend_name": config.backend_name,
                    "using_fake_backend": bool(config.use_fake_backend),
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            call_counter["count"] += 1
            theta = np.asarray(circuit["theta"], dtype=float)
            mean = float(-theta.size)
            return type(
                "EstimateStub",
                (),
                {"mean": mean, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=8,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(noise_mode="ideal", oracle_aggregate="mean"),
        wallclock_cap_s=60,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=controller.current_theta,
        psi=controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_pool_dup",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline = controller._baseline_geometry(checkpoint_ctx, cache, geometry_memo)
    shortlist = controller._scout_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline,
        predicted_displacement=0.0,
    )
    confirmed = controller._confirm_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline,
        shortlist=shortlist,
    )
    assert len(confirmed) >= 2

    seen_counts: list[int] = []
    seen_budget_scales: list[float] = []
    original_confirm_oracle = controller._confirm_candidates_oracle

    def _motion_stub(*, theta_dot, predicted_displacement):
        return MotionSchedulerTelemetry(
            regime="kink",
            direction_cosine=-1.0,
            rate_change_l2=1.5,
            rate_change_ratio=1.5,
            acceleration_l2=1.0,
            curvature_cosine=-1.0,
            direction_reversal=True,
            curvature_sign_flip=True,
            kink_score=1.5,
        )

    def _confirm_wrapper(*args, **kwargs):
        seen_counts.append(len(kwargs["confirmed"]))
        seen_budget_scales.append(float(kwargs["budget_scale"]))
        return original_confirm_oracle(*args, **kwargs)

    monkeypatch.setattr(controller, "_motion_telemetry", _motion_stub)
    monkeypatch.setattr(controller, "_effective_refresh_pressure", lambda **kwargs: "high")
    monkeypatch.setattr(controller, "_confirm_candidates_oracle", _confirm_wrapper)
    result = controller.run()

    assert int(call_counter["count"]) >= 1
    assert seen_counts and max(seen_counts) >= 2
    assert seen_budget_scales and max(seen_budget_scales) == pytest.approx(
        float(controller.cfg.motion_kink_oracle_budget_scale)
    )
    assert any(str(row["motion_regime"]) == "kink" for row in result.ledger)
    assert any(int(row["oracle_confirm_limit"]) >= 2 for row in result.ledger if bool(row["oracle_attempted"]))


def test_measurement_state_key_distinguishes_duplicate_label_candidates_by_pool_index() -> None:
    replay_context, h_poly, hmat, psi_initial = _duplicate_label_context()
    cfg = RealtimeCheckpointConfig(mode="exact_v1")
    controller = RealtimeCheckpointController(
        cfg=cfg,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=controller.current_theta,
        psi=controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_pool_dup",
        grouping_mode=str(cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    dup_y = controller._candidate_executor_data(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        candidate_term=replay_context.family_pool[1],
        candidate_pool_index=1,
        position_id=1,
    )
    dup_z = controller._candidate_executor_data(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        candidate_term=replay_context.family_pool[2],
        candidate_pool_index=2,
        position_id=1,
    )

    key_y = controller._measurement_state_key(
        layout=dup_y["aug_layout"],
        theta_runtime=np.asarray(dup_y["theta_aug"], dtype=float).reshape(-1),
    )
    key_z = controller._measurement_state_key(
        layout=dup_z["aug_layout"],
        theta_runtime=np.asarray(dup_z["theta_aug"], dtype=float).reshape(-1),
    )

    assert key_y != key_z


def test_realtime_controller_oracle_v1_appends_when_candidate_noisy_energy_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor

    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    call_counter = {"count": 0}

    monkeypatch.setattr(
        nor,
        "build_runtime_layout_circuit",
        lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()},
    )
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", lambda poly, tol=1e-12: object())

    class _OracleStub:
        def __init__(self, config):
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "stub",
                    "backend_name": config.backend_name,
                    "using_fake_backend": bool(config.use_fake_backend),
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            call_counter["count"] += 1
            theta = np.asarray(circuit["theta"], dtype=float)
            mean = float(-theta.size)
            return type(
                "EstimateStub",
                (),
                {"mean": mean, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="ideal", oracle_aggregate="mean"),
        wallclock_cap_s=60,
    )
    result = controller.run()

    assert int(result.summary["append_count"]) >= 1
    assert str(result.summary["mode"]) == "oracle_v1"
    assert str(result.summary["decision_backend"]) == "mixed"
    assert int(result.summary["oracle_decision_checkpoints"]) >= 1
    assert str(result.summary["oracle_estimate_kind"]) == "oracle_ideal"
    assert int(call_counter["count"]) >= 2
    assert any(str(row["decision_backend"]) == "oracle" for row in result.ledger)


def test_realtime_controller_oracle_v1_calm_exit_makes_no_oracle_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor

    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    call_counter = {"count": 0}

    monkeypatch.setattr(
        nor,
        "build_runtime_layout_circuit",
        lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()},
    )
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", lambda poly, tol=1e-12: object())

    class _OracleStub:
        def __init__(self, config):
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "stub",
                    "backend_name": config.backend_name,
                    "using_fake_backend": bool(config.use_fake_backend),
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            call_counter["count"] += 1
            return type(
                "EstimateStub",
                (),
                {"mean": -1.0, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=2.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(noise_mode="ideal", oracle_aggregate="mean"),
        wallclock_cap_s=60,
    )
    result = controller.run()

    assert int(result.summary["append_count"]) == 0
    assert str(result.summary["decision_backend"]) == "exact"
    assert int(result.summary["oracle_attempted_checkpoints"]) == 0
    assert int(call_counter["count"]) == 0
    assert all(str(row["decision_backend"]) == "exact" for row in result.ledger)


def test_realtime_controller_oracle_v1_fake_marrakesh_backend_scheduled_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeMarrakesh",
            use_fake_backend=True,
            shots=32,
            oracle_repeats=1,
            seed=7,
        ),
        wallclock_cap_s=60,
    )
    original_measured_baseline = controller._oracle_measured_baseline_geometry

    def _baseline_with_miss(*args, **kwargs):
        out = dict(original_measured_baseline(*args, **kwargs))
        out["summary"] = dataclass_replace(out["summary"], rho_miss=0.5)
        return out

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _baseline_with_miss)
    result = controller.run()

    assert str(result.summary["mode"]) == "oracle_v1"
    assert str(result.summary["decision_noise_mode"]) == "backend_scheduled"
    assert str(result.summary["oracle_estimate_kind"]) == "oracle_backend_scheduled"
    assert int(result.summary["oracle_decision_checkpoints"]) >= 1
    oracle_rows = [row for row in result.ledger if str(row["decision_backend"]) == "oracle"]
    assert oracle_rows
    assert any(int(row["raw_group_cache_misses"]) >= 1 for row in oracle_rows)


def test_realtime_controller_oracle_v1_backend_scheduled_uses_raw_group_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor
    import pipelines.hardcoded.hh_fixed_manifold_observables as obs
    from qiskit.quantum_info import SparsePauliOp

    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    collect_counter = {"count": 0}
    evaluate_counter = {"count": 0}

    runtime_circuit_stub = lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()}
    sparse_stub = lambda poly, tol=1e-12: SparsePauliOp.from_list([("Z", 1.0)])
    monkeypatch.setattr(nor, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", sparse_stub)
    monkeypatch.setattr(obs, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(obs, "pauli_poly_to_sparse_pauli_op", sparse_stub)

    class _OracleStub:
        def __init__(self, config):
            self.config = config
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": config.backend_name,
                    "using_fake_backend": bool(config.use_fake_backend),
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            evaluate_counter["count"] += 1
            return type(
                "EstimateStub",
                (),
                {"mean": -1.0, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            collect_counter["count"] += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": int(self.config.shots),
                "counts": {"1": int(self.config.shots)},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {
                    "active_logical_qubits": [0],
                    "active_physical_qubits": [0],
                    "pauli_weight": 1,
                    "label": str(pauli_label_ixyz),
                },
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            shots=128,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )
    original_measured_baseline = controller._oracle_measured_baseline_geometry

    def _baseline_with_miss(*args, **kwargs):
        out = dict(original_measured_baseline(*args, **kwargs))
        out["summary"] = dataclass_replace(out["summary"], rho_miss=0.5)
        return out

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _baseline_with_miss)
    result = controller.run()

    assert int(result.summary["oracle_decision_checkpoints"]) >= 1
    assert int(collect_counter["count"]) >= 2
    assert int(evaluate_counter["count"]) == 0
    oracle_rows = [row for row in result.ledger if str(row["decision_backend"]) == "oracle"]
    assert oracle_rows
    assert any(int(row["raw_group_cache_hits"]) >= 1 for row in oracle_rows)


def test_realtime_controller_oracle_v1_candidate_confirm_uses_incremental_reducer_not_full_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor
    import pipelines.hardcoded.hh_fixed_manifold_observables as obs
    import pipelines.hardcoded.hh_realtime_checkpoint_controller as ctrl_mod
    from qiskit.quantum_info import SparsePauliOp

    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    runtime_circuit_stub = lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()}
    sparse_stub = lambda poly, tol=1e-12: SparsePauliOp.from_list([("Z", 1.0)])
    monkeypatch.setattr(nor, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", sparse_stub)
    monkeypatch.setattr(obs, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(obs, "pauli_poly_to_sparse_pauli_op", sparse_stub)

    full_geometry_calls = {"count": 0}
    incremental_calls = {"count": 0}
    original_full_geometry = ctrl_mod.estimate_grouped_raw_mclachlan_geometry
    original_incremental = ctrl_mod.estimate_grouped_raw_mclachlan_incremental_block

    def _full_geometry_spy(*args, **kwargs):
        full_geometry_calls["count"] += 1
        return original_full_geometry(*args, **kwargs)

    def _incremental_spy(*args, **kwargs):
        incremental_calls["count"] += 1
        return original_incremental(*args, **kwargs)

    monkeypatch.setattr(ctrl_mod, "estimate_grouped_raw_mclachlan_geometry", _full_geometry_spy)
    monkeypatch.setattr(
        ctrl_mod,
        "estimate_grouped_raw_mclachlan_incremental_block",
        _incremental_spy,
    )

    class _OracleStub:
        def __init__(self, config):
            self.config = config
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": config.backend_name,
                    "using_fake_backend": bool(config.use_fake_backend),
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            return type(
                "EstimateStub",
                (),
                {"mean": -1.0, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            return {
                "repeat_index": int(repeat_idx),
                "shots": int(self.config.shots),
                "counts": {"1": int(self.config.shots)},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {
                    "active_logical_qubits": [0],
                    "active_physical_qubits": [0],
                    "pauli_weight": 1,
                    "label": str(pauli_label_ixyz),
                },
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            shots=128,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )
    original_measured_baseline = controller._oracle_measured_baseline_geometry

    def _baseline_with_miss(*args, **kwargs):
        out = dict(original_measured_baseline(*args, **kwargs))
        out["summary"] = dataclass_replace(out["summary"], rho_miss=0.5)
        return out

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _baseline_with_miss)
    result = controller.run()

    assert int(result.summary["oracle_decision_checkpoints"]) >= 1
    assert int(full_geometry_calls["count"]) == 1
    assert int(incremental_calls["count"]) >= 1


def test_realtime_controller_oracle_v1_geometry_failure_falls_back_to_scalar_confirm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor
    import pipelines.hardcoded.hh_fixed_manifold_observables as obs
    from qiskit.quantum_info import SparsePauliOp

    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    collect_counter = {"count": 0}

    runtime_circuit_stub = lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()}
    sparse_stub = lambda poly, tol=1e-12: SparsePauliOp.from_list([("Z", 1.0)])
    monkeypatch.setattr(nor, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", sparse_stub)
    monkeypatch.setattr(obs, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(obs, "pauli_poly_to_sparse_pauli_op", sparse_stub)

    class _OracleStub:
        def __init__(self, config):
            self.config = config
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": config.backend_name,
                    "using_fake_backend": bool(config.use_fake_backend),
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            return type(
                "EstimateStub",
                (),
                {"mean": -1.0, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            collect_counter["count"] += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": int(self.config.shots),
                "counts": {"1": int(self.config.shots)},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {
                    "active_logical_qubits": [0],
                    "active_physical_qubits": [0],
                    "pauli_weight": 1,
                    "label": str(pauli_label_ixyz),
                },
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(
            noise_mode="backend_scheduled",
            oracle_aggregate="mean",
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            shots=128,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    original_measured_baseline = controller._oracle_measured_baseline_geometry

    def _baseline_with_miss(*args, **kwargs):
        out = dict(original_measured_baseline(*args, **kwargs))
        out["summary"] = dataclass_replace(out["summary"], rho_miss=0.5)
        return out

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _baseline_with_miss)
    monkeypatch.setattr(controller, "_oracle_measured_candidate_incremental_block", _boom)
    result = controller.run()

    assert int(result.summary["oracle_decision_checkpoints"]) >= 1
    assert int(result.summary["degraded_checkpoints"]) >= 1
    assert int(collect_counter["count"]) >= 2
    assert any(
        "measured_candidate_geometry_error" in str(row.get("degraded_reason"))
        for row in result.ledger
    )
    assert any(str(row["decision_backend"]) == "oracle" for row in result.ledger)


def test_realtime_controller_oracle_v1_runtime_uses_group_sampling_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines.exact_bench.noise_oracle_runtime as nor
    import pipelines.hardcoded.hh_fixed_manifold_observables as obs
    from qiskit.quantum_info import SparsePauliOp

    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    collect_counter = {"count": 0}
    evaluate_counter = {"count": 0}

    runtime_circuit_stub = lambda layout, theta_runtime, num_qubits, reference_state=None: {"theta": np.asarray(theta_runtime, dtype=float).tolist()}
    sparse_stub = lambda poly, tol=1e-12: SparsePauliOp.from_list([("Z", 1.0)])
    monkeypatch.setattr(nor, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(nor, "pauli_poly_to_sparse_pauli_op", sparse_stub)
    monkeypatch.setattr(obs, "build_runtime_layout_circuit", runtime_circuit_stub)
    monkeypatch.setattr(obs, "pauli_poly_to_sparse_pauli_op", sparse_stub)

    class _OracleStub:
        def __init__(self, config):
            self.config = config
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": str(config.noise_mode),
                    "estimator_kind": "qiskit_ibm_runtime.SamplerV2",
                    "backend_name": config.backend_name,
                    "using_fake_backend": False,
                    "details": {},
                },
            )()

        def evaluate(self, circuit, observable):
            evaluate_counter["count"] += 1
            return type(
                "EstimateStub",
                (),
                {"mean": -1.0, "stderr": 0.01, "std": 0.0, "stdev": 0.0, "n_samples": 1, "aggregate": "mean"},
            )()

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            collect_counter["count"] += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": int(self.config.shots),
                "counts": {"1": int(self.config.shots)},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {
                    "active_logical_qubits": [0],
                    "active_physical_qubits": None,
                    "pauli_weight": 1,
                    "label": str(pauli_label_ixyz),
                },
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

        def close(self):
            return None

    monkeypatch.setattr(nor, "ExpectationOracle", _OracleStub)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=OracleConfig(
            noise_mode="runtime",
            oracle_aggregate="mean",
            backend_name="ibm_fake_runtime",
            shots=128,
            oracle_repeats=1,
            mitigation="none",
        ),
        wallclock_cap_s=60,
    )
    result = controller.run()

    assert int(result.summary["oracle_decision_checkpoints"]) >= 1
    assert int(collect_counter["count"]) >= 2
    assert int(evaluate_counter["count"]) == 0
    oracle_rows = [row for row in result.ledger if str(row["decision_backend"]) == "oracle"]
    assert oracle_rows
    assert any(int(row["raw_group_cache_hits"]) >= 1 for row in oracle_rows)
