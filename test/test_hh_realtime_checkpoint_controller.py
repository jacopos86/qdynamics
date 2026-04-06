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
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm, hamiltonian_matrix


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


def _two_qubit_drive_context(
    theta_x: float = 0.2,
) -> tuple[ReplayScaffoldContext, PauliPolynomial, np.ndarray, np.ndarray]:
    x_term = AnsatzTerm(
        label="op_x0",
        polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="ex", pc=1.0)]),
    )
    y_term = AnsatzTerm(
        label="op_y0",
        polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="ey", pc=1.0)]),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(2, ps="ez", pc=1.0)])
    hmat = np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
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
        family_info={"resolved": "toy_pool_drive"},
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


def _two_block_context(
    theta_x: float = 0.2,
    theta_y: float = 0.01,
) -> tuple[ReplayScaffoldContext, np.ndarray, np.ndarray, np.ndarray]:
    x_term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    y_term = AnsatzTerm(
        label="op_y",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="y", pc=1.0)]),
    )
    z_term = AnsatzTerm(
        label="op_z",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    hmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    psi_ref = _basis(0)
    base_layout = build_parameter_layout([x_term, y_term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    executor = CompiledAnsatzExecutor(
        [x_term, y_term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    best_theta = np.array([float(theta_x), float(theta_y)], dtype=float)
    psi_initial = executor.prepare_state(best_theta, psi_ref)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1, L=1, ordering="blocked"),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}},
        family_info={"resolved": "toy_two_block"},
        family_pool=(x_term, y_term, z_term),
        pool_meta={"candidate_pool_complete": True},
        replay_terms=(x_term, y_term),
        base_layout=base_layout,
        adapt_theta_runtime=np.asarray(best_theta, dtype=float),
        adapt_theta_logical=np.asarray(best_theta, dtype=float),
        adapt_depth=2,
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=3,
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


def test_realtime_controller_appends_candidate_and_hits_same_checkpoint_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(controller, "_exact_v1_forecast_override_reason", lambda **kwargs: None)
    result = controller.run()

    assert int(result.summary["append_count"]) >= 1
    assert str(result.ledger[0]["action_kind"]) == "append_candidate"
    assert str(result.ledger[0]["controller_lane"]) == "append"
    assert str(result.ledger[0]["controller_lane_reason"]) == "exact_rho_miss_above_threshold"
    assert str(result.trajectory[0]["controller_lane"]) == "append"
    assert str(result.trajectory[0]["controller_lane_reason"]) == "exact_rho_miss_above_threshold"
    assert float(result.trajectory[0]["confirmed"][0]["confirm_score"]) == pytest.approx(
        float(result.trajectory[0]["confirmed"][0]["adjusted_gain"])
    )
    assert str(result.trajectory[0]["confirmed"][0]["confirm_score_kind"]) == "compressed_whitened_lower_gain_ratio_minus_penalties"
    assert int(result.trajectory[0]["confirmed"][0]["confirm_compress_modes_used"]) >= 1
    assert str(result.trajectory[-1]["controller_lane"]) == "stay"
    assert str(result.trajectory[-1]["controller_lane_reason"]) == "terminal_checkpoint"
    assert int(result.ledger[0]["exact_cache_misses"]) >= 1
    assert int(result.ledger[0]["geometry_memo_hits"]) >= 1
    assert int(result.summary["final_runtime_parameter_count"]) >= 2


def test_realtime_controller_select_action_scans_past_surrogate_top_candidate_that_fails_exact_thresholds() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.5,
            append_margin_abs=0.1,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )

    def _record(label: str, *, confirm_score: float, gain_ratio: float, gain_exact: float, pool_index: int) -> dict[str, object]:
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": int(pool_index),
            "position_id": int(pool_index),
            "adjusted_gain": float(confirm_score),
            "confirm_score": float(confirm_score),
            "gain_exact": float(gain_exact),
            "gain_ratio": float(gain_ratio),
            "groups_new": 0.0,
            "candidate_summary": CandidateProbeSummary(
                candidate_label=label,
                candidate_pool_index=int(pool_index),
                position_id=int(pool_index),
                runtime_insert_position=0,
                runtime_block_indices=[],
                residual_overlap_l2=0.0,
                gain_exact=float(gain_exact),
                gain_ratio=float(gain_ratio),
                compile_proxy_total=1.0,
                groups_new=0.0,
                novelty=None,
                position_jump_penalty=0.0,
                directional_change_l2=0.0,
                tier_reached="confirm",
                admissible=True,
                rejection_reason=None,
                decision_metric="compressed_whitened_confirm_gain_ratio",
            ),
        }

    action_kind, selected = controller._select_action(
        baseline={"summary": SimpleNamespace(rho_miss=1.0)},
        confirmed=[
            _record("candidate_a", confirm_score=3.0, gain_ratio=0.1, gain_exact=1.0, pool_index=0),
            _record("candidate_b", confirm_score=2.0, gain_ratio=1.0, gain_exact=1.0, pool_index=1),
        ],
    )

    assert str(action_kind) == "append_candidate"
    assert selected is not None
    assert str(selected["candidate_label"]) == "candidate_b"


def test_realtime_controller_prune_lane_can_commit_coordinate_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=1.0e-3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0,
            prune_theta_block_tol=1.0,
            prune_state_jump_l2_tol=1.0,
            prune_safe_miss_increase_tol=1.0,
            prune_max_candidates=1,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 1.0e-3],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    monkeypatch.setattr(
        controller,
        "_motion_telemetry",
        lambda **kwargs: MotionSchedulerTelemetry(
            regime="calm",
            direction_cosine=1.0,
            rate_change_l2=0.0,
            rate_change_ratio=0.0,
            acceleration_l2=0.0,
            curvature_cosine=1.0,
            direction_reversal=False,
            curvature_sign_flip=False,
            kink_score=0.0,
        ),
    )

    result = controller.run()

    assert int(result.summary["prune_count"]) == 1
    assert str(result.trajectory[0]["controller_lane"]) == "prune"
    assert str(result.trajectory[0]["action_kind"]) == "prune_coordinate"
    assert int(result.trajectory[0]["logical_block_count"]) == 2
    assert int(result.summary["final_logical_block_count"]) == 1
    assert float(result.trajectory[0]["selected_prune_cached_loss"]) >= 0.0
    assert result.trajectory[0]["post_prune_energy_total"] is not None
    assert result.trajectory[0]["post_prune_fidelity_exact"] is not None


def test_realtime_controller_stays_when_prune_theta_block_is_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=0.3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0,
            prune_theta_block_tol=1.0e-4,
            prune_state_jump_l2_tol=1.0,
            prune_safe_miss_increase_tol=1.0,
            prune_max_candidates=1,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 0.3],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    monkeypatch.setattr(
        controller,
        "_motion_telemetry",
        lambda **kwargs: MotionSchedulerTelemetry(
            regime="calm",
            direction_cosine=1.0,
            rate_change_l2=0.0,
            rate_change_ratio=0.0,
            acceleration_l2=0.0,
            curvature_cosine=1.0,
            direction_reversal=False,
            curvature_sign_flip=False,
            kink_score=0.0,
        ),
    )

    result = controller.run()

    assert int(result.summary["prune_count"]) == 0
    assert str(result.trajectory[0]["controller_lane"]) == "prune"
    assert str(result.trajectory[0]["action_kind"]) == "stay"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "prune_coordinate"
    assert str(result.trajectory[0]["decision_override_reason"]) == "prune_rejected_theta_block_above_tol"


def test_realtime_controller_stays_when_prune_loss_proxy_is_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=1.0e-3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0e-3,
            prune_theta_block_tol=1.0,
            prune_state_jump_l2_tol=1.0,
            prune_safe_miss_increase_tol=1.0,
            prune_max_candidates=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 1.0e-3],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    monkeypatch.setattr(
        controller,
        "_motion_telemetry",
        lambda **kwargs: MotionSchedulerTelemetry(
            regime="calm",
            direction_cosine=1.0,
            rate_change_l2=0.0,
            rate_change_ratio=0.0,
            acceleration_l2=0.0,
            curvature_cosine=1.0,
            direction_reversal=False,
            curvature_sign_flip=False,
            kink_score=0.0,
        ),
    )
    monkeypatch.setattr(controller, "_cached_prune_loss", lambda **kwargs: 0.5)

    result = controller.run()

    assert int(result.summary["prune_count"]) == 0
    assert str(result.trajectory[0]["controller_lane"]) == "prune"
    assert str(result.trajectory[0]["action_kind"]) == "stay"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "prune_coordinate"
    assert str(result.trajectory[0]["decision_override_reason"]) == "prune_rejected_cached_prune_loss_above_tol"


def test_realtime_controller_stays_when_miss_threshold_is_high(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        controller,
        "_scout_candidates",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("scout should stay closed below miss threshold")),
    )
    result = controller.run()

    assert int(result.summary["append_count"]) == 0
    assert all(str(row["action_kind"]) == "stay" for row in result.ledger)
    assert all(str(row["controller_lane"]) == "stay" for row in result.ledger)
    assert all(str(row["controller_lane_reason"]) == "exact_rho_miss_below_threshold" for row in result.trajectory[:-1])
    assert all(row.get("shortlist") == [] for row in result.trajectory[:-1])
    assert all(row.get("confirmed") == [] for row in result.trajectory[:-1])
    assert str(result.trajectory[-1]["controller_lane"]) == "stay"
    assert str(result.trajectory[-1]["controller_lane_reason"]) == "terminal_checkpoint"


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


def test_scout_candidates_follow_manuscript_lower_bound_score() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    cfg = RealtimeCheckpointConfig(
        mode="exact_v1",
        miss_threshold=0.0,
        shortlist_size=8,
        shortlist_fraction=1.0,
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

    assert shortlist
    record = dict(shortlist[0])
    U_cols = [
        np.asarray(record["candidate_data"]["raw_tangents"][idx], dtype=complex)
        - complex(np.vdot(baseline["psi"], record["candidate_data"]["raw_tangents"][idx])) * np.asarray(baseline["psi"], dtype=complex)
        for idx in record["candidate_data"]["runtime_block_indices"]
    ]
    U = np.column_stack(U_cols) if U_cols else np.zeros((baseline["psi"].size, 0), dtype=complex)
    residual_overlap_vec = np.asarray(
        np.real(U.conj().T @ np.asarray(baseline["residual_step"], dtype=complex)),
        dtype=float,
    ).reshape(-1)
    C = np.asarray(np.real(U.conj().T @ U), dtype=float)
    C_reg = np.asarray(
        C + float(cfg.candidate_regularization_lambda) * np.eye(int(C.shape[0])),
        dtype=float,
    )
    C_reg_pinv = np.linalg.pinv(C_reg, rcond=float(cfg.pinv_rcond)) if C_reg.size else np.zeros((0, 0), dtype=float)
    lower_gain = (
        float(max(0.0, float(residual_overlap_vec @ C_reg_pinv @ residual_overlap_vec)))
        if residual_overlap_vec.size
        else 0.0
    )
    scout_gain_ratio = float(lower_gain / max(float(baseline["norm_b_sq"]), 1e-14))
    scout_score = float(
        scout_gain_ratio
        + float(record.get("temporal_prior_bonus", 0.0))
        - float(cfg.compile_penalty_weight) * float(record["compile_proxy_total"])
        - float(cfg.measurement_penalty_weight) * float(record["groups_new"])
        - float(cfg.directional_penalty_weight) * float(record["position_jump_penalty"])
    )
    legacy_simple_score = float(
        float(np.linalg.norm(residual_overlap_vec))
        + float(record.get("temporal_prior_bonus", 0.0))
        - float(cfg.compile_penalty_weight) * float(record["compile_proxy_total"])
        - float(cfg.measurement_penalty_weight) * float(record["groups_new"])
        - float(cfg.directional_penalty_weight) * float(record["position_jump_penalty"])
    )

    assert float(record["residual_overlap_l2"]) == pytest.approx(float(np.linalg.norm(residual_overlap_vec)))
    assert float(record["scout_lower_gain"]) == pytest.approx(lower_gain)
    assert float(record["scout_gain_ratio"]) == pytest.approx(scout_gain_ratio)
    assert float(record["scout_score"]) == pytest.approx(scout_score)
    assert float(record["simple_score"]) == pytest.approx(legacy_simple_score)
    assert str(record["scout_score_kind"]) == "shared_baseline_lower_gain_ratio_minus_penalties"


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
    assert result.reference["projection_time_sampling"] == "midpoint"
    assert result.reference["geometry_sample_time_policy"] == "interval_midpoint_plus_t0_with_final_endpoint_fallback"
    assert bool(result.summary["drive_aligned_density_active"]) is False
    assert result.summary["drive_aligned_density_label"] is None
    assert result.trajectory[0]["physical_time"] == pytest.approx(0.05)
    assert result.trajectory[1]["physical_time"] == pytest.approx(0.15)
    assert result.trajectory[2]["physical_time"] == pytest.approx(0.2)
    assert all("baseline_step_scale" in row for row in result.trajectory)
    assert all("baseline_gain_scale" in row for row in result.trajectory)
    assert any(int(row.get("drive_term_count", 0)) >= 1 for row in result.ledger[1:])
    assert all("physical_time" in row for row in result.ledger)
    assert all("staggered" in row and "staggered_exact" in row for row in result.trajectory)
    assert all("doublon" in row and "doublon_exact" in row for row in result.trajectory)
    assert all("site_occupations" in row and "site_occupations_exact" in row for row in result.trajectory)
    assert "max_abs_staggered_error" in result.summary
    assert "max_abs_doublon_error" in result.summary
    assert "max_abs_site_occupations_error" in result.summary


def test_realtime_controller_drive_exact_v1_augments_with_drive_aligned_density(
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    assert bool(controller._drive_aligned_density_active) is True
    assert str(controller._drive_aligned_density_label) == "drive_aligned_density(pattern=staggered)"
    assert int(controller.current_layout.logical_parameter_count) == (
        int(replay_context.base_layout.logical_parameter_count) + 1
    )
    assert int(controller.current_theta.size) == int(controller.current_layout.runtime_parameter_count)
    assert int(controller.current_layout.runtime_parameter_count) > int(replay_context.base_layout.runtime_parameter_count)
    psi_reconstructed = controller.current_executor.prepare_state(
        np.asarray(controller.current_theta, dtype=float),
        replay_context.psi_ref,
    )
    assert np.linalg.norm(
        np.asarray(psi_reconstructed, dtype=complex).reshape(-1)
        - np.asarray(psi_initial, dtype=complex).reshape(-1)
    ) <= 1.0e-10


def test_realtime_controller_drive_aligned_runtime_indices_match_suffixed_runtime_labels() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    indices = controller._drive_aligned_runtime_indices()
    assert tuple(indices) == tuple(range(int(controller.current_layout.blocks[-1].runtime_start), int(controller.current_layout.blocks[-1].runtime_stop)))
    assert str(controller.current_layout.blocks[-1].candidate_label).startswith(
        "drive_aligned_density(pattern=staggered)__r"
    )


@pytest.mark.parametrize(
    ("sampling", "expected_physical_times"),
    [
        ("midpoint", [0.25, 0.35, 0.4]),
        ("left", [0.2, 0.3, 0.4]),
        ("right", [0.3, 0.4, 0.4]),
    ],
)
def test_realtime_controller_projection_sample_time_variants(
    monkeypatch: pytest.MonkeyPatch,
    sampling: str,
    expected_physical_times: list[float],
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
            drive_time_sampling=str(sampling),
            drive_t0=0.2,
            exact_steps_multiplier=1,
        ),
    )

    observed_physical_times = []
    for idx, time_start in enumerate(controller.times):
        time_stop = None if idx + 1 >= len(controller.times) else float(controller.times[idx + 1])
        sample_time = controller._projection_sample_time(float(time_start), time_stop)
        observed_physical_times.append(
            float(controller._step_hamiltonian_artifacts(float(sample_time)).physical_time)
        )

    assert observed_physical_times == pytest.approx(expected_physical_times)


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
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
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
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
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


def test_realtime_controller_oracle_geometry_clears_deferred_confirm_payload() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=1.0e-9,
            append_margin_abs=1.0e-12,
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
            noise_mode="ideal",
            oracle_aggregate="mean",
            shots=64,
            oracle_repeats=1,
        ),
        wallclock_cap_s=60,
    )

    def _record(label: str, pool_index: int) -> dict[str, object]:
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": int(pool_index),
            "position_id": int(pool_index),
            "adjusted_gain": 2.0,
            "confirm_score": 2.0,
            "confirm_score_kind": "compressed_whitened_lower_gain_ratio_minus_penalties",
            "confirm_compress_modes_used": 2,
            "confirm_support_rank": 2,
            "confirm_compressed_gain_ratio": 2.0,
            "confirm_compressed_gain_exact": 2.0,
            "gain_exact": 2.0,
            "gain_ratio": 2.0,
            "groups_new": 0.0,
            "candidate_summary": CandidateProbeSummary(
                candidate_label=label,
                candidate_pool_index=int(pool_index),
                position_id=int(pool_index),
                runtime_insert_position=0,
                runtime_block_indices=[],
                residual_overlap_l2=0.0,
                gain_exact=2.0,
                gain_ratio=2.0,
                compile_proxy_total=1.0,
                groups_new=0.0,
                novelty=None,
                position_jump_penalty=0.0,
                directional_change_l2=0.0,
                tier_reached="confirm",
                admissible=True,
                rejection_reason=None,
                decision_metric="measured_incremental_gain_ratio",
                oracle_estimate_kind="oracle_ideal",
            ),
        }

    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.2,
        scaffold_labels=[str(block.candidate_label) for block in controller.current_layout.blocks],
        theta=np.asarray(controller.current_theta, dtype=float),
        psi=np.asarray(
            controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
            dtype=complex,
        ),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family=str(replay_context.family_info.get("resolved", "toy_pool")),
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=True,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=checkpoint_ctx.checkpoint_id,
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=checkpoint_ctx.checkpoint_id)

    controller._oracle_measured_baseline_geometry = lambda **kwargs: {  # type: ignore[method-assign]
        "summary": SimpleNamespace(rho_miss=0.5)
    }

    baseline_measured, measured_records, degraded_reason = controller._confirm_candidates_oracle_geometry(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        confirmed=[_record("candidate_a", 0), _record("candidate_b", 1)],
        raw_group_pool=None,
        h_poly_step=h_poly,
        confirm_limit=0,
        budget_scale=1.0,
    )

    assert degraded_reason is None
    assert baseline_measured is not None
    assert len(measured_records) == 2
    for rec in measured_records:
        assert rec["gain_exact"] is None
        assert rec["gain_ratio"] is None
        assert rec["adjusted_gain"] == float("-inf")
        assert rec["confirm_score"] is None
        assert str(rec["confirm_score_kind"]) == "not_confirmed"
        assert int(rec["confirm_compress_modes_used"]) == 0
        assert int(rec["confirm_support_rank"]) == 0
        assert rec["confirm_compressed_gain_ratio"] is None
        assert rec["confirm_compressed_gain_exact"] is None
        assert rec["confirm_backend_info"] is None
        assert str(rec["confirm_error"]) == "deferred_by_refresh_pressure"
        assert rec["candidate_summary"].gain_exact is None
        assert rec["candidate_summary"].gain_ratio is None
        assert str(rec["candidate_summary"].rejection_reason) == "deferred_by_refresh_pressure"
        assert str(rec["candidate_summary"].decision_metric) == "not_confirmed"


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
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
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
    assert str(result.trajectory[0]["controller_lane"]) == "append"
    assert str(result.trajectory[0]["controller_lane_reason"]) == "exact_rho_miss_above_threshold"
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
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
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


def test_realtime_controller_exact_v1_vetoes_append_when_stay_forecast_is_already_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        num_times=2,
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
                decision_metric="compressed_whitened_confirm_gain_ratio",
                oracle_estimate_kind=None,
            ),
            "candidate_data": {
                "theta_aug": np.asarray(controller.current_theta, dtype=float).copy(),
                "aug_layout": controller.current_layout,
                "aug_executor": controller.current_executor,
                "aug_terms": list(controller.current_terms),
                "runtime_insert_position": 0,
                "runtime_block_indices": [],
            },
            "theta_dot_aug": np.asarray(controller.current_theta, dtype=float) * 0.0,
        }

    forecast_calls: list[str] = []

    def _fake_exact_step_forecast(**kwargs):
        forecast_calls.append("call")
        if len(forecast_calls) == 1:
            return {
                "fidelity_exact_next": 0.9995,
                "abs_energy_total_error_next": 1.0e-4,
                "abs_staggered_error_next": 5.0e-3,
                "abs_doublon_error_next": 5.0e-4,
                "site_occupations_abs_error_max_next": 5.0e-3,
            }
        return {
            "fidelity_exact_next": 0.9997,
            "abs_energy_total_error_next": 1.2e-4,
            "abs_staggered_error_next": 4.0e-3,
            "abs_doublon_error_next": 4.0e-4,
            "site_occupations_abs_error_max_next": 4.0e-3,
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
    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    result = controller.run()

    assert int(result.summary["append_count"]) == 0
    assert int(result.summary["stay_count"]) == 2
    assert int(result.summary["decision_override_count"]) == 1
    assert int(result.summary["exact_forecast_veto_count"]) == 1
    assert str(result.trajectory[0]["action_kind"]) == "stay"
    assert str(result.trajectory[0]["controller_lane"]) == "append"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "append_candidate"
    assert str(result.trajectory[0]["decision_override_reason"]) == "exact_forecast_stay_within_bounded_defect"
    assert float(result.trajectory[0]["forecast_stay_site_occupations_abs_error_max_next"]) == pytest.approx(
        4.0e-3
    )
    assert str(result.ledger[0]["decision_override_reason"]) == "exact_forecast_stay_within_bounded_defect"


def test_exact_v1_forecast_override_reason_rejects_nonimproving_tracking_score() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )

    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast={
            "fidelity_exact_next": 0.97,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.03,
            "abs_doublon_error_next": 0.01,
            "site_occupations_abs_error_max_next": 0.03,
        },
        selected_forecast={
            "fidelity_exact_next": 0.975,
            "abs_energy_total_error_next": 0.02,
            "abs_staggered_error_next": 0.04,
            "abs_doublon_error_next": 0.015,
            "site_occupations_abs_error_max_next": 0.04,
        },
        action_kind="append_candidate",
        selected={"candidate_label": "candidate_a"},
    )

    assert reason == "exact_forecast_nonimproving_tracking_score"


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


def test_exact_v1_selects_damped_candidate_step_scale_from_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
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
    selected = {
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
            decision_metric="compressed_whitened_confirm_gain_ratio",
            oracle_estimate_kind=None,
        ),
    }

    def _fake_exact_step_forecast(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        append_amp = float(theta_runtime[-1])
        if append_amp <= 0.3:
            return {
                "fidelity_exact_next": 0.995,
                "abs_energy_total_error_next": 0.01,
                "abs_staggered_error_next": 0.01,
                "abs_doublon_error_next": 0.005,
                "site_occupations_abs_error_max_next": 0.01,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.4,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_selected, scaled_forecast = controller._select_exact_v1_candidate_step_scale(
        baseline_theta_dot=np.array([0.2], dtype=float),
        selected=selected,
        dt=1.0,
        time_stop=0.2,
    )

    assert float(scaled_selected["candidate_step_scale"]) == pytest.approx(0.25)
    assert np.asarray(scaled_selected["theta_dot_aug"], dtype=float) == pytest.approx(np.array([0.2, 0.25]))
    assert float(scaled_selected["candidate_summary"].selected_step_scale) == pytest.approx(0.25)
    assert float(scaled_forecast["fidelity_exact_next"]) == pytest.approx(0.995)


def test_exact_v1_forecast_tracking_score_uses_horizon_weights() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=3,
            exact_forecast_tracking_horizon_weights=(3.0, 2.0, 1.0),
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
        wallclock_cap_s=60,
    )
    forecasts = [
        {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
        },
        {
            "fidelity_exact_next": 0.95,
            "abs_energy_total_error_next": 0.10,
            "abs_staggered_error_next": 0.20,
            "abs_doublon_error_next": 0.30,
            "site_occupations_abs_error_max_next": 0.40,
        },
        {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.50,
            "abs_staggered_error_next": 0.60,
            "abs_doublon_error_next": 0.70,
            "site_occupations_abs_error_max_next": 0.80,
        },
    ]

    score = controller._forecast_tracking_score(forecast=forecasts)

    step_1 = 0.01 + 0.02 + 0.03 + 0.04 + 0.01
    step_2 = 0.05 + 0.20 + 0.30 + 0.40 + 0.10
    step_3 = 0.10 + 0.60 + 0.70 + 0.80 + 0.50
    expected = (3.0 * step_1 + 2.0 * step_2 + 1.0 * step_3) / 6.0
    assert float(score) == pytest.approx(expected)


def test_exact_v1_forecast_tracking_score_adds_energy_slope_term() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_slope_weight=100.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        wallclock_cap_s=60,
    )
    forecasts = [
        {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
            "energy_total_controller_next": 0.10,
            "energy_total_exact_next": 0.10,
        },
        {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
            "energy_total_controller_next": 0.10,
            "energy_total_exact_next": 0.11,
        },
    ]

    score = controller._forecast_tracking_score(forecast=forecasts)

    base = 0.01 + 0.02 + 0.03 + 0.04 + 0.01
    slope_err = 0.01
    expected = base + 100.0 * slope_err
    assert float(score) == pytest.approx(expected)


def test_exact_v1_forecast_tracking_score_adds_energy_slope_term_with_anchor() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_slope_weight=100.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        wallclock_cap_s=60,
    )
    forecasts = [
        {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
            "energy_total_controller_next": 0.10,
            "energy_total_exact_next": 0.10,
        },
        {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
            "energy_total_controller_next": 0.10,
            "energy_total_exact_next": 0.11,
        },
    ]
    slope_anchor = {
        "energy_total_controller_next": 0.20,
        "energy_total_exact_next": 0.18,
    }

    score = controller._forecast_tracking_score(
        forecast=forecasts,
        curvature_anchor=slope_anchor,
    )

    base = 0.01 + 0.02 + 0.03 + 0.04 + 0.01
    slope_err = 0.015
    expected = base + 100.0 * slope_err
    assert float(score) == pytest.approx(expected)


def test_exact_v1_forecast_tracking_score_adds_energy_curvature_term_with_anchor() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_curvature_weight=50.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        wallclock_cap_s=60,
    )
    forecasts = [
        {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
            "energy_total_controller_next": 0.10,
            "energy_total_exact_next": 0.10,
        },
        {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.01,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
            "energy_total_controller_next": 0.10,
            "energy_total_exact_next": 0.11,
        },
    ]
    curvature_anchor = {
        "energy_total_controller_next": 0.08,
        "energy_total_exact_next": 0.08,
    }

    score = controller._forecast_tracking_score(
        forecast=forecasts,
        curvature_anchor=curvature_anchor,
    )

    base = 0.01 + 0.02 + 0.03 + 0.04 + 0.01
    curvature_err = 0.01
    expected = base + 50.0 * curvature_err
    assert float(score) == pytest.approx(expected)


def test_forecast_tracking_score_uses_stored_horizon_score_for_rollout_record() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
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
        wallclock_cap_s=60,
    )

    score = controller._forecast_tracking_score(
        forecast={
            "tracking_score_horizon": 1.2345,
            "fidelity_exact_next": 0.0,
            "abs_energy_total_error_next": 999.0,
            "abs_staggered_error_next": 999.0,
            "abs_doublon_error_next": 999.0,
            "site_occupations_abs_error_max_next": 999.0,
        }
    )

    assert float(score) == pytest.approx(1.2345)


def test_exact_v1_horizon_prefers_gentler_candidate_step_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.25, 1.0),
            exact_forecast_tracking_horizon_steps=3,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=3.0,
        num_times=4,
        wallclock_cap_s=60,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=1.0,
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
    selected = {
        "candidate_label": str(replay_context.family_pool[1].label),
        "candidate_identity": f"{replay_context.family_pool[1].label}__pool1",
        "candidate_pool_index": 1,
        "position_id": 1,
        "runtime_insert_position": int(candidate_data["runtime_insert_position"]),
        "runtime_block_indices": list(candidate_data["runtime_block_indices"]),
        "groups_new": 0.0,
        "candidate_data": candidate_data,
        "theta_dot_aug": np.array([0.0, 1.0], dtype=float),
        "theta_dot_aug_existing": np.array([0.0], dtype=float),
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
            decision_metric="compressed_whitened_confirm_gain_ratio",
            oracle_estimate_kind=None,
        ),
    }

    def _fake_exact_step_forecast(**kwargs):
        append_amp = float(np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)[-1])
        if append_amp < 0.9:
            return {
                "fidelity_exact_next": 0.97,
                "abs_energy_total_error_next": 0.02,
                "abs_staggered_error_next": 0.02,
                "abs_doublon_error_next": 0.01,
                "site_occupations_abs_error_max_next": 0.02,
            }
        if append_amp < 1.1:
            return {
                "fidelity_exact_next": 0.995,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
            }
        return {
            "fidelity_exact_next": 0.80,
            "abs_energy_total_error_next": 0.40,
            "abs_staggered_error_next": 0.20,
            "abs_doublon_error_next": 0.10,
            "site_occupations_abs_error_max_next": 0.30,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_selected, scaled_forecast = controller._select_exact_v1_candidate_step_scale(
        baseline_theta_dot=np.array([0.0], dtype=float),
        selected=selected,
        dt=1.0,
        time_stop=1.0,
    )

    assert float(scaled_selected["candidate_step_scale"]) == pytest.approx(0.25)
    assert np.asarray(scaled_selected["theta_dot_aug"], dtype=float) == pytest.approx(np.array([0.0, 0.25]))
    assert int(scaled_forecast["tracking_horizon_steps_scored"]) == 3
    assert list(scaled_forecast["tracking_horizon_weights_used"]) == pytest.approx([1.0, 1.0, 1.0])


def test_exact_v1_drive_bootstrap_adds_drive_aligned_density_and_preserves_state() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    assert bool(controller._drive_aligned_density_active) is True
    assert str(controller._drive_aligned_density_label) == "drive_aligned_density(pattern=staggered)"
    assert int(controller.current_layout.logical_parameter_count) == 2
    assert int(controller.current_layout.runtime_parameter_count) > 1
    assert float(controller.current_theta[0]) == pytest.approx(0.2)
    assert np.allclose(np.asarray(controller.current_theta[1:], dtype=float), 0.0)
    psi_current = np.asarray(
        controller.current_executor.prepare_state(
            controller.current_theta,
            replay_context.psi_ref,
        ),
        dtype=complex,
    )
    assert np.linalg.norm(psi_current - np.asarray(psi_initial, dtype=complex)) <= 1.0e-10


def test_exact_v1_selects_damped_drive_aligned_baseline_step_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", candidate_step_scales=(1.0,)),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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
    assert bool(controller._drive_aligned_density_active) is True
    baseline_theta_dot = np.zeros(int(controller.current_theta.size), dtype=float)
    baseline_theta_dot[0] = 0.2
    baseline_theta_dot[-1] = 0.4

    def _fake_exact_step_forecast(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        amp = float(theta_runtime[-1])
        if 0.035 <= amp <= 0.045:
            return {
                "fidelity_exact_next": 0.999,
                "abs_energy_total_error_next": 0.01,
                "abs_staggered_error_next": 0.01,
                "abs_doublon_error_next": 0.005,
                "site_occupations_abs_error_max_next": 0.01,
            }
        if abs(amp) <= 1.0e-12:
            return {
                "fidelity_exact_next": 0.995,
                "abs_energy_total_error_next": 0.03,
                "abs_staggered_error_next": 0.02,
                "abs_doublon_error_next": 0.01,
                "site_occupations_abs_error_max_next": 0.02,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.3,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=0.2,
    )

    assert float(step_scale) == pytest.approx(0.1)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(np.array([0.02, 0.0, 0.04]))
    assert float(forecast["fidelity_exact_next"]) == pytest.approx(0.999)


def test_exact_v1_selects_damped_baseline_step_scale_from_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    def _fake_exact_step_forecast(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        drive_amp = float(theta_runtime[-1])
        if abs(drive_amp - 0.1) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.999,
                "abs_energy_total_error_next": 0.002,
                "abs_staggered_error_next": 0.002,
                "abs_doublon_error_next": 0.001,
                "site_occupations_abs_error_max_next": 0.002,
            }
        if drive_amp < 0.1:
            return {
                "fidelity_exact_next": 0.995,
                "abs_energy_total_error_next": 0.01,
                "abs_staggered_error_next": 0.01,
                "abs_doublon_error_next": 0.005,
                "site_occupations_abs_error_max_next": 0.01,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.4,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([0.2, 0.0, 1.0], dtype=float),
        dt=1.0,
        time_stop=0.2,
    )

    assert float(step_scale) == pytest.approx(0.1)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(
        np.array([0.02, 0.0, 0.1], dtype=float)
    )
    assert float(forecast["fidelity_exact_next"]) == pytest.approx(0.999)


def test_exact_v1_refines_baseline_step_scale_between_coarse_grid_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.2,),
            exact_forecast_baseline_step_refine_rounds=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    def _fake_exact_step_forecast(**kwargs):
        drive_amp = float(np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)[-1])
        if abs(drive_amp - 0.15) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.9995,
                "abs_energy_total_error_next": 0.001,
                "abs_staggered_error_next": 0.001,
                "abs_doublon_error_next": 0.0005,
                "site_occupations_abs_error_max_next": 0.001,
            }
        if abs(drive_amp - 0.1) <= 1.0e-9 or abs(drive_amp - 0.2) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.01,
                "abs_staggered_error_next": 0.008,
                "abs_doublon_error_next": 0.004,
                "site_occupations_abs_error_max_next": 0.008,
            }
        if drive_amp < 0.25:
            return {
                "fidelity_exact_next": 0.995,
                "abs_energy_total_error_next": 0.02,
                "abs_staggered_error_next": 0.015,
                "abs_doublon_error_next": 0.007,
                "site_occupations_abs_error_max_next": 0.015,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.4,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([0.2, 0.0, 1.0], dtype=float),
        dt=1.0,
        time_stop=0.2,
    )

    assert float(step_scale) == pytest.approx(0.15)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(
        np.array([0.03, 0.0, 0.15], dtype=float)
    )
    assert float(forecast["fidelity_exact_next"]) == pytest.approx(0.9995)


def test_exact_v1_selects_blended_baseline_direction_when_forecast_prefers_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_blend_weights=(0.0, 1.0),
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    monkeypatch.setattr(
        controller,
        "_drive_only_theta_dot_from_baseline",
        lambda **_kwargs: np.array([0.0, 0.0, 1.0], dtype=float),
    )

    def _fake_exact_step_forecast(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        drive_amp = float(theta_runtime[-1])
        lead_amp = float(theta_runtime[0])
        if abs(drive_amp - (1.0 / np.sqrt(2.0))) <= 1.0e-9 and abs(lead_amp - (0.2 + (1.0 / np.sqrt(2.0)))) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.999,
                "abs_energy_total_error_next": 0.002,
                "abs_staggered_error_next": 0.002,
                "abs_doublon_error_next": 0.001,
                "site_occupations_abs_error_max_next": 0.002,
            }
        if abs(lead_amp - 1.2) <= 1.0e-9 and abs(drive_amp) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.95,
                "abs_energy_total_error_next": 0.03,
                "abs_staggered_error_next": 0.02,
                "abs_doublon_error_next": 0.01,
                "site_occupations_abs_error_max_next": 0.02,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.4,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([1.0, 0.0, 0.0], dtype=float),
        baseline=None,
        dt=1.0,
        time_stop=0.2,
    )

    assert float(step_scale) == pytest.approx(1.0)
    assert float(blend_weight) == pytest.approx(1.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(
        np.array([1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)], dtype=float)
    )
    assert float(forecast["fidelity_exact_next"]) == pytest.approx(0.999)


def test_exact_v1_selects_negative_residual_blend_when_forecast_prefers_early_anti_drive_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_blend_weights=(-0.5, 0.0, 1.0),
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    monkeypatch.setattr(
        controller,
        "_drive_only_theta_dot_from_baseline",
        lambda **_kwargs: np.array([0.0, 0.0, 1.0], dtype=float),
    )

    def _fake_exact_step_forecast(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        lead_amp = float(theta_runtime[0])
        drive_amp = float(theta_runtime[-1])
        if abs(lead_amp - (0.2 + 1.0 / np.sqrt(1.25))) <= 1.0e-9 and abs(drive_amp + (0.5 / np.sqrt(1.25))) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.9992,
                "abs_energy_total_error_next": 0.0015,
                "abs_staggered_error_next": 0.0015,
                "abs_doublon_error_next": 0.0008,
                "site_occupations_abs_error_max_next": 0.0015,
            }
        if abs(lead_amp - 1.2) <= 1.0e-9 and abs(drive_amp) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.004,
                "abs_staggered_error_next": 0.004,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.004,
            }
        if abs(lead_amp - (0.2 + 1.0 / np.sqrt(2.0))) <= 1.0e-9 and abs(drive_amp - (1.0 / np.sqrt(2.0))) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.993,
                "abs_energy_total_error_next": 0.01,
                "abs_staggered_error_next": 0.01,
                "abs_doublon_error_next": 0.004,
                "site_occupations_abs_error_max_next": 0.01,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.4,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([1.0, 0.0, 0.0], dtype=float),
        baseline=None,
        dt=1.0,
        time_stop=0.2,
    )

    assert float(step_scale) == pytest.approx(1.0)
    assert float(blend_weight) == pytest.approx(-0.5)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(
        np.array([1.0 / np.sqrt(1.25), 0.0, -0.5 / np.sqrt(1.25)], dtype=float)
    )
    assert float(forecast["fidelity_exact_next"]) == pytest.approx(0.9992)


def test_exact_tangent_secant_proposal_projects_one_step_exact_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_include_tangent_secant_proposal=True,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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
    baseline = {
        "psi": np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        "T": np.array([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=complex),
        "G": np.array([[1.0]], dtype=float),
        "K_pinv": np.array([[1.0]], dtype=float),
    }
    monkeypatch.setattr(
        controller,
        "_exact_state_at",
        lambda _time_value: np.array([np.sqrt(0.99) + 0.0j, 0.1 + 0.0j], dtype=complex),
    )

    proposal = controller._exact_tangent_secant_proposal(
        baseline=baseline,
        dt=0.5,
        time_stop=0.5,
    )

    assert proposal is not None
    assert str(proposal["proposal_kind"]) == "tangent_secant_exact_v1"
    assert np.asarray(proposal["theta_dot_direction"], dtype=float) == pytest.approx(
        np.array([0.2], dtype=float)
    )
    assert float(proposal["tangent_secant_displacement_norm"]) == pytest.approx(0.1)
    assert float(proposal["tangent_secant_projection_quality"]) == pytest.approx(1.0)
    assert float(proposal["tangent_secant_raw_metric_norm"]) == pytest.approx(0.2)
    assert float(proposal["tangent_secant_metric_norm"]) == pytest.approx(0.2)


def test_exact_tangent_secant_proposal_can_be_suppressed_by_signed_energy_lead_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_include_tangent_secant_proposal=True,
            exact_forecast_tangent_secant_signed_energy_lead_limit=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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
    baseline = {
        "psi": np.array([np.sqrt(0.5) + 0.0j, np.sqrt(0.5) + 0.0j], dtype=complex),
        "T": np.array([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=complex),
        "G": np.array([[1.0]], dtype=float),
        "K_pinv": np.array([[1.0]], dtype=float),
    }

    def _fake_exact_state_at(time_value: float) -> np.ndarray:
        if time_value <= 0.0 + 1.0e-12:
            return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        return np.array([np.sqrt(0.99) + 0.0j, 0.1 + 0.0j], dtype=complex)

    monkeypatch.setattr(controller, "_exact_state_at", _fake_exact_state_at)
    monkeypatch.setattr(
        controller,
        "_step_hamiltonian_artifacts",
        lambda _time_value: SimpleNamespace(
            hmat=np.diag([0.0, 1.0]).astype(complex),
            physical_time=float(_time_value),
        ),
    )

    proposal = controller._exact_tangent_secant_proposal(
        baseline=baseline,
        dt=0.5,
        time_stop=0.5,
    )

    assert proposal is None


def test_exact_v1_selection_can_pick_tangent_secant_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_include_tangent_secant_proposal=True,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    monkeypatch.setattr(
        controller,
        "_baseline_theta_dot_candidates",
        lambda **_kwargs: [(0.0, np.array([1.0, 0.0, 0.0], dtype=float))],
    )
    monkeypatch.setattr(
        controller,
        "_exact_tangent_secant_proposal",
        lambda **_kwargs: {
            "proposal_kind": "tangent_secant_exact_v1",
            "blend_weight": 0.0,
            "theta_dot_direction": np.array([0.0, 0.0, 1.0], dtype=float),
            "current_baseline_norm": None,
            "current_drive_norm": None,
            "lookahead_drive_norm": None,
            "tangent_secant_displacement_norm": 0.2,
            "tangent_secant_projection_quality": 0.95,
            "tangent_secant_raw_metric_norm": 1.0,
            "tangent_secant_metric_norm": 1.0,
        },
    )

    def _fake_exact_forecast_rollout(**kwargs):
        theta_dot_step = np.asarray(kwargs["theta_dot_step"], dtype=float).reshape(-1)
        if abs(float(theta_dot_step[2])) > abs(float(theta_dot_step[0])):
            forecast = {
                "fidelity_exact_next": 0.9995,
                "abs_energy_total_error_next": 0.001,
                "abs_staggered_error_next": 0.001,
                "abs_doublon_error_next": 0.001,
                "site_occupations_abs_error_max_next": 0.001,
                "energy_total_controller_next": 0.09,
                "energy_total_exact_next": 0.09,
            }
            return dict(forecast), [dict(forecast)], 0.01
        forecast = {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.02,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.02,
            "site_occupations_abs_error_max_next": 0.02,
            "energy_total_controller_next": 0.11,
            "energy_total_exact_next": 0.09,
        }
        return dict(forecast), [dict(forecast)], 1.0

    monkeypatch.setattr(controller, "_exact_forecast_rollout", _fake_exact_forecast_rollout)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([1.0, 0.0, 0.0], dtype=float),
        baseline={"G": np.eye(3, dtype=float)},
        dt=0.1,
        time_stop=0.1,
    )

    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(
        np.array([0.0, 0.0, 0.05], dtype=float)
    )
    assert float(step_scale) == pytest.approx(0.05)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert str(forecast["baseline_proposal_kind"]) == "tangent_secant_exact_v1"
    assert bool(forecast["baseline_include_tangent_secant_proposal"]) is True
    assert float(forecast["baseline_tangent_secant_projection_quality"]) == pytest.approx(0.95)
    assert float(forecast["baseline_tangent_secant_displacement_norm"]) == pytest.approx(0.2)


def test_anticipatory_drive_basis_proposals_include_lookahead_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_baseline_proposal_mode="anticipatory_drive_basis_v1",
            exact_forecast_baseline_blend_weights=(-0.5, 0.0, 1.0),
        ),
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
            drive_omega=2.0,
            drive_tbar=2.5,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_weights=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=2,
        ),
        wallclock_cap_s=60,
    )
    baseline = {
        "theta_dot_step": np.array([1.0e-6, 0.0], dtype=float),
        "G": np.eye(2, dtype=float),
    }
    lookahead_baseline = {
        "theta_dot_step": np.array([1.0e-6, 0.0], dtype=float),
        "G": np.eye(2, dtype=float),
    }

    def _fake_drive_only(*, baseline):
        if baseline is lookahead_baseline:
            return np.array([0.0, 2.0e-2], dtype=float)
        return np.array([2.0e-6, 0.0], dtype=float)

    monkeypatch.setattr(controller, "_drive_only_theta_dot_from_baseline", _fake_drive_only)
    monkeypatch.setattr(
        controller,
        "_lookahead_drive_baseline",
        lambda **kwargs: lookahead_baseline,
    )

    proposals = controller._baseline_theta_dot_proposals(
        checkpoint_index=0,
        baseline_theta_dot=np.array([1.0e-6, 0.0], dtype=float),
        baseline=baseline,
    )

    proposal_kinds = {str(item["proposal_kind"]) for item in proposals}
    assert "drive_only_lookahead" in proposal_kinds
    lookahead = next(item for item in proposals if str(item["proposal_kind"]) == "drive_only_lookahead")
    assert np.asarray(lookahead["theta_dot_direction"], dtype=float) == pytest.approx(np.array([0.0, 1.0]))
    assert float(lookahead["current_baseline_norm"]) == pytest.approx(1.0e-6)
    assert float(lookahead["lookahead_drive_norm"]) == pytest.approx(2.0e-2)


def test_anticipatory_drive_basis_can_select_lookahead_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_baseline_proposal_mode="anticipatory_drive_basis_v1",
            exact_forecast_baseline_gain_scales=(1.0,),
        ),
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
            drive_omega=2.0,
            drive_tbar=2.5,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_weights=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=2,
        ),
        wallclock_cap_s=60,
    )
    monkeypatch.setattr(
        controller,
        "_baseline_theta_dot_proposals",
        lambda **kwargs: [
            {
                "proposal_kind": "baseline_current",
                "blend_weight": 0.0,
                "theta_dot_direction": np.array([1.0, 0.0, 0.0], dtype=float),
                "current_baseline_norm": 1.0e-6,
                "current_drive_norm": 2.0e-6,
                "lookahead_drive_norm": 2.0e-2,
            },
            {
                "proposal_kind": "drive_only_lookahead",
                "blend_weight": 0.0,
                "theta_dot_direction": np.array([0.0, 0.0, 1.0], dtype=float),
                "current_baseline_norm": 1.0e-6,
                "current_drive_norm": 2.0e-6,
                "lookahead_drive_norm": 2.0e-2,
            },
        ],
    )

    def _fake_exact_forecast_rollout(**kwargs):
        theta_dot_step = np.asarray(kwargs["theta_dot_step"], dtype=float).reshape(-1)
        if abs(float(theta_dot_step[2])) > abs(float(theta_dot_step[0])):
            forecast = {
                "fidelity_exact_next": 0.9995,
                "abs_energy_total_error_next": 0.001,
                "abs_staggered_error_next": 0.001,
                "abs_doublon_error_next": 0.001,
                "site_occupations_abs_error_max_next": 0.001,
                "energy_total_controller_next": 0.09,
                "energy_total_exact_next": 0.09,
            }
            return dict(forecast), [dict(forecast)], 0.01
        forecast = {
            "fidelity_exact_next": 0.99,
            "abs_energy_total_error_next": 0.02,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.02,
            "site_occupations_abs_error_max_next": 0.02,
            "energy_total_controller_next": 0.11,
            "energy_total_exact_next": 0.09,
        }
        return dict(forecast), [dict(forecast)], 1.0

    monkeypatch.setattr(controller, "_exact_forecast_rollout", _fake_exact_forecast_rollout)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        checkpoint_index=0,
        baseline_theta_dot=np.array([1.0e-6, 0.0, 0.0], dtype=float),
        baseline={"theta_dot_step": np.array([1.0e-6, 0.0, 0.0], dtype=float), "G": np.eye(3, dtype=float)},
        dt=0.1,
        time_stop=0.1,
    )

    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(np.array([0.0, 0.0, 0.05]))
    assert float(step_scale) == pytest.approx(0.05)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert str(forecast["baseline_proposal_kind"]) == "drive_only_lookahead"
    assert str(forecast["baseline_proposal_mode"]) == "anticipatory_drive_basis_v1"
    assert float(forecast["baseline_lookahead_drive_only_norm"]) == pytest.approx(2.0e-2)


def test_exact_v1_selects_baseline_gain_scale_above_one_when_forecast_prefers_stronger_same_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_blend_weights=(0.0, 1.0),
            exact_forecast_baseline_gain_scales=(1.0, 1.1, 1.2),
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
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

    monkeypatch.setattr(
        controller,
        "_drive_only_theta_dot_from_baseline",
        lambda **_kwargs: np.array([0.0, 0.0, 1.0], dtype=float),
    )

    def _fake_exact_step_forecast(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        lead_amp = float(theta_runtime[0])
        drive_amp = float(theta_runtime[-1])
        if abs(lead_amp - (0.2 + 1.2 / np.sqrt(2.0))) <= 1.0e-9 and abs(drive_amp - (1.2 / np.sqrt(2.0))) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.9995,
                "abs_energy_total_error_next": 0.001,
                "abs_staggered_error_next": 0.001,
                "abs_doublon_error_next": 0.0005,
                "site_occupations_abs_error_max_next": 0.001,
            }
        if abs(lead_amp - (0.2 + 1.0 / np.sqrt(2.0))) <= 1.0e-9 and abs(drive_amp - (1.0 / np.sqrt(2.0))) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.998,
                "abs_energy_total_error_next": 0.004,
                "abs_staggered_error_next": 0.004,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.004,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.4,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([1.0, 0.0, 0.0], dtype=float),
        baseline=None,
        dt=1.0,
        time_stop=0.2,
    )

    assert float(step_scale) == pytest.approx(1.0)
    assert float(blend_weight) == pytest.approx(1.0)
    assert float(gain_scale) == pytest.approx(1.2)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(
        np.array([1.2 / np.sqrt(2.0), 0.0, 1.2 / np.sqrt(2.0)], dtype=float)
    )
    assert float(forecast["fidelity_exact_next"]) == pytest.approx(0.9995)


def test_exact_v1_energy_excursion_under_term_prefers_higher_post_step_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    baseline_theta_dot = np.array([0.2, 0.0, 1.0], dtype=float)

    def _fake_exact_step_forecast(**kwargs):
        drive_amp = float(np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)[-1])
        time_stop = float(kwargs["time_stop"])
        if abs(drive_amp) <= 1.0e-9:
            if abs(time_stop) <= 1.0e-9:
                return {
                    "fidelity_exact_next": 0.999,
                    "abs_energy_total_error_next": 0.0,
                    "abs_staggered_error_next": 0.0,
                    "abs_doublon_error_next": 0.0,
                    "site_occupations_abs_error_max_next": 0.0,
                    "energy_total_controller_next": 0.100,
                    "energy_total_exact_next": 0.100,
                }
            return {
                "fidelity_exact_next": 0.996,
                "abs_energy_total_error_next": 0.006,
                "abs_staggered_error_next": 0.006,
                "abs_doublon_error_next": 0.003,
                "site_occupations_abs_error_max_next": 0.006,
                "energy_total_controller_next": 0.100,
                "energy_total_exact_next": (0.104 if abs(time_stop - 1.0) <= 1.0e-9 else 0.108),
            }
        if abs(drive_amp - 1.0) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.101,
                "energy_total_exact_next": 0.104,
            }
        if abs(drive_amp - 2.0) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.103,
                "energy_total_exact_next": 0.108,
            }
        if abs(drive_amp - 1.2) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.1025,
                "energy_total_exact_next": 0.104,
            }
        if abs(drive_amp - 2.4) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.1065,
                "energy_total_exact_next": 0.108,
            }
        if abs(drive_amp - 1.4) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.9973,
                "abs_energy_total_error_next": 0.0045,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.1045,
                "energy_total_exact_next": 0.104,
            }
        if abs(drive_amp - 2.8) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.9973,
                "abs_energy_total_error_next": 0.0045,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.1095,
                "energy_total_exact_next": 0.108,
            }
        return {
            "fidelity_exact_next": 0.90,
            "abs_energy_total_error_next": 0.4,
            "abs_staggered_error_next": 0.2,
            "abs_doublon_error_next": 0.1,
            "site_occupations_abs_error_max_next": 0.2,
            "energy_total_controller_next": 0.0,
            "energy_total_exact_next": 0.0,
        }

    controller_plain = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_gain_scales=(1.0, 1.2),
            exact_forecast_tracking_horizon_steps=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_plain, "_exact_step_forecast", _fake_exact_step_forecast)
    _theta_plain, _step_plain, _blend_plain, gain_plain, forecast_plain = controller_plain._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=1.0,
    )

    controller_exc = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_gain_scales=(1.0, 1.2),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_excursion_under_weight=200.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_exc, "_exact_step_forecast", _fake_exact_step_forecast)
    theta_exc, step_exc, blend_exc, gain_exc, forecast_exc = controller_exc._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=1.0,
    )

    assert float(gain_plain) == pytest.approx(1.0)
    assert float(forecast_plain["tracking_energy_excursion_under_response_mean"]) == pytest.approx(0.0)
    assert float(forecast_plain["tracking_energy_excursion_under_weight"]) == pytest.approx(0.0)
    assert float(step_exc) == pytest.approx(1.0)
    assert float(blend_exc) == pytest.approx(0.0)
    assert float(gain_exc) == pytest.approx(1.2)
    assert np.asarray(theta_exc, dtype=float) == pytest.approx(
        np.array([0.24, 0.0, 1.2], dtype=float)
    )
    assert float(forecast_exc["tracking_energy_excursion_under_response_mean"]) == pytest.approx(0.0015)
    assert float(forecast_exc["tracking_energy_excursion_under_weight"]) == pytest.approx(200.0)

    controller_under_only = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_gain_scales=(1.0, 1.2, 1.4),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_excursion_under_weight=200.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_under_only, "_exact_step_forecast", _fake_exact_step_forecast)
    _theta_under_only, _step_under_only, _blend_under_only, gain_under_only, _forecast_under_only = controller_under_only._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=1.0,
    )

    assert float(gain_under_only) == pytest.approx(1.4)

    controller_band = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_gain_scales=(1.0, 1.2, 1.4),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_excursion_under_weight=200.0,
            exact_forecast_energy_excursion_over_weight=500.0,
            exact_forecast_energy_excursion_rel_tolerance=0.03,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_band, "_exact_step_forecast", _fake_exact_step_forecast)
    theta_band, step_band, blend_band, gain_band, forecast_band = controller_band._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=1.0,
    )

    assert float(step_band) == pytest.approx(1.0)
    assert float(blend_band) == pytest.approx(0.0)
    assert float(gain_band) == pytest.approx(1.2)
    assert np.asarray(theta_band, dtype=float) == pytest.approx(
        np.array([0.24, 0.0, 1.2], dtype=float)
    )
    assert float(forecast_band["tracking_energy_excursion_over_weight"]) == pytest.approx(500.0)
    assert float(forecast_band["tracking_energy_excursion_rel_tolerance"]) == pytest.approx(0.03)


def test_exact_v1_horizon_prefers_gentler_baseline_step_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_tracking_horizon_steps=3,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=3.0,
        num_times=4,
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

    def _fake_exact_step_forecast(**kwargs):
        drive_amp = float(np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)[-1])
        if abs(drive_amp - 1.0) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.995,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
            }
        if 0.08 <= drive_amp <= 0.35:
            return {
                "fidelity_exact_next": 0.97,
                "abs_energy_total_error_next": 0.02,
                "abs_staggered_error_next": 0.02,
                "abs_doublon_error_next": 0.01,
                "site_occupations_abs_error_max_next": 0.02,
            }
        if drive_amp < 0.08:
            return {
                "fidelity_exact_next": 0.95,
                "abs_energy_total_error_next": 0.03,
                "abs_staggered_error_next": 0.03,
                "abs_doublon_error_next": 0.015,
                "site_occupations_abs_error_max_next": 0.03,
            }
        return {
            "fidelity_exact_next": 0.80,
            "abs_energy_total_error_next": 0.40,
            "abs_staggered_error_next": 0.20,
            "abs_doublon_error_next": 0.10,
            "site_occupations_abs_error_max_next": 0.30,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([0.2, 0.0, 1.0], dtype=float),
        dt=1.0,
        time_stop=1.0,
    )

    assert float(step_scale) == pytest.approx(0.1)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(np.array([0.02, 0.0, 0.1], dtype=float))
    assert int(forecast["tracking_horizon_steps_scored"]) == 3
    assert list(forecast["tracking_horizon_weights_used"]) == pytest.approx([1.0, 1.0, 1.0])


def test_exact_v1_energy_slope_term_prefers_shape_matched_baseline_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    baseline_theta_dot = np.array([0.2, 0.0, 1.0], dtype=float)

    def _fake_exact_step_forecast(**kwargs):
        drive_amp = float(np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)[-1])
        if abs(drive_amp - 1.0) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.100,
                "energy_total_exact_next": 0.105,
            }
        if abs(drive_amp - 2.0) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.997,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.100,
                "energy_total_exact_next": 0.115,
            }
        if abs(drive_amp - 0.2) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.996,
                "abs_energy_total_error_next": 0.006,
                "abs_staggered_error_next": 0.006,
                "abs_doublon_error_next": 0.0025,
                "site_occupations_abs_error_max_next": 0.006,
                "energy_total_controller_next": 0.102,
                "energy_total_exact_next": 0.105,
            }
        if abs(drive_amp - 0.4) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.996,
                "abs_energy_total_error_next": 0.006,
                "abs_staggered_error_next": 0.006,
                "abs_doublon_error_next": 0.0025,
                "site_occupations_abs_error_max_next": 0.006,
                "energy_total_controller_next": 0.112,
                "energy_total_exact_next": 0.115,
            }
        return {
            "fidelity_exact_next": 0.80,
            "abs_energy_total_error_next": 0.40,
            "abs_staggered_error_next": 0.20,
            "abs_doublon_error_next": 0.10,
            "site_occupations_abs_error_max_next": 0.30,
            "energy_total_controller_next": 0.0,
            "energy_total_exact_next": 0.0,
        }

    controller_h2 = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.2, 1.0),
            exact_forecast_tracking_horizon_steps=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_h2, "_exact_step_forecast", _fake_exact_step_forecast)
    _scaled_theta_dot_h2, step_scale_h2, blend_weight_h2, gain_scale_h2, _forecast_h2 = controller_h2._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=1.0,
    )

    controller_shape = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.2, 1.0),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_slope_weight=500.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_shape, "_exact_step_forecast", _fake_exact_step_forecast)
    scaled_theta_dot_shape, step_scale_shape, blend_weight_shape, gain_scale_shape, forecast_shape = controller_shape._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=1.0,
    )

    assert float(step_scale_h2) == pytest.approx(1.0)
    assert float(blend_weight_h2) == pytest.approx(0.0)
    assert float(gain_scale_h2) == pytest.approx(1.0)
    assert float(step_scale_shape) == pytest.approx(0.2)
    assert float(blend_weight_shape) == pytest.approx(0.0)
    assert float(gain_scale_shape) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot_shape, dtype=float) == pytest.approx(
        np.array([0.04, 0.0, 0.2], dtype=float)
    )
    assert float(forecast_shape["tracking_energy_slope_abs_error_mean"]) == pytest.approx(0.0)
    assert float(forecast_shape["tracking_energy_slope_weight"]) == pytest.approx(500.0)


def test_exact_v1_energy_curvature_term_is_active_for_h2_with_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    baseline_theta_dot = np.array([0.2, 0.0, 1.0], dtype=float)

    def _fake_exact_step_forecast(**kwargs):
        drive_amp = float(np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)[-1])
        time_stop = float(kwargs["time_stop"])
        if abs(drive_amp) <= 1.0e-9:
            if abs(time_stop) <= 1.0e-9:
                return {
                    "fidelity_exact_next": 0.999,
                    "abs_energy_total_error_next": 0.01,
                    "abs_staggered_error_next": 0.005,
                    "abs_doublon_error_next": 0.002,
                    "site_occupations_abs_error_max_next": 0.005,
                    "energy_total_controller_next": 0.090,
                    "energy_total_exact_next": 0.100,
                }
            return {
                "fidelity_exact_next": 0.995,
                "abs_energy_total_error_next": 0.015,
                "abs_staggered_error_next": 0.008,
                "abs_doublon_error_next": 0.003,
                "site_occupations_abs_error_max_next": 0.008,
                "energy_total_controller_next": 0.090,
                "energy_total_exact_next": 0.105,
            }
        if abs(drive_amp - 0.2) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.999,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.110,
                "energy_total_exact_next": 0.105,
            }
        if abs(drive_amp - 0.4) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.999,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.120,
                "energy_total_exact_next": 0.115,
            }
        if abs(drive_amp - 1.0) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.999,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.100,
                "energy_total_exact_next": 0.105,
            }
        if abs(drive_amp - 2.0) <= 1.0e-9:
            return {
                "fidelity_exact_next": 0.999,
                "abs_energy_total_error_next": 0.005,
                "abs_staggered_error_next": 0.005,
                "abs_doublon_error_next": 0.002,
                "site_occupations_abs_error_max_next": 0.005,
                "energy_total_controller_next": 0.110,
                "energy_total_exact_next": 0.115,
            }
        return {
            "fidelity_exact_next": 0.80,
            "abs_energy_total_error_next": 0.40,
            "abs_staggered_error_next": 0.20,
            "abs_doublon_error_next": 0.10,
            "site_occupations_abs_error_max_next": 0.30,
            "energy_total_controller_next": 0.0,
            "energy_total_exact_next": 0.0,
        }

    controller_no_curv = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.2, 1.0),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_slope_weight=500.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_no_curv, "_exact_step_forecast", _fake_exact_step_forecast)
    _scaled_theta_dot_no_curv, step_scale_no_curv, blend_weight_no_curv, gain_scale_no_curv, _forecast_no_curv = (
        controller_no_curv._select_exact_v1_baseline_step_scale(
            baseline_theta_dot=baseline_theta_dot,
            dt=1.0,
            time_stop=1.0,
        )
    )

    controller_curv = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.2, 1.0),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_slope_weight=500.0,
            exact_forecast_energy_curvature_weight=200.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=2.0,
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
    monkeypatch.setattr(controller_curv, "_exact_step_forecast", _fake_exact_step_forecast)
    scaled_theta_dot_curv, step_scale_curv, blend_weight_curv, gain_scale_curv, forecast_curv = (
        controller_curv._select_exact_v1_baseline_step_scale(
            baseline_theta_dot=baseline_theta_dot,
            dt=1.0,
            time_stop=1.0,
        )
    )

    assert float(step_scale_no_curv) == pytest.approx(0.2)
    assert float(blend_weight_no_curv) == pytest.approx(0.0)
    assert float(gain_scale_no_curv) == pytest.approx(1.0)
    assert float(step_scale_curv) == pytest.approx(1.0)
    assert float(blend_weight_curv) == pytest.approx(0.0)
    assert float(gain_scale_curv) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot_curv, dtype=float) == pytest.approx(
        np.array([0.2, 0.0, 1.0], dtype=float)
    )
    assert float(forecast_curv["tracking_energy_curvature_abs_error_mean"]) > 0.0
    assert float(forecast_curv["tracking_energy_curvature_weight"]) == pytest.approx(200.0)


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
