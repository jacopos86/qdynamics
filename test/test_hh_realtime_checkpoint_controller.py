from __future__ import annotations

from dataclasses import replace as dataclass_replace
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# This checkout no longer ships the legacy module below; skip instead of
# failing collection so the suite has a clean baseline for current work.
# Quarantined 2026-08-15 on paper-ii-exchange-selector; delete the test or
# restore the module to reactivate.
pytest.importorskip("pipelines.hardcoded.hh_realtime_checkpoint_controller")

import pipelines.hardcoded.hh_realtime_checkpoint_controller as controller_mod
import pipelines.hardcoded.hh_realtime_measurement as measurement_mod
from pipelines.exact_bench.noise_oracle_runtime import OracleConfig
from pipelines.hardcoded.hh_realtime_checkpoint_controller import (
    ControllerDriveConfig,
    MotionSchedulerTelemetry,
    RealtimeCheckpointController,
    _build_candidate_carrier,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON,
    HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING,
    BaselineGeometrySummary,
    CandidateProbeSummary,
    RealtimeCheckpointConfig,
    make_checkpoint_context,
    make_measurement_checkpoint_context,
    physical_trajectory_rows,
    strict_qpu_faithful_decision_contract,
)
from pipelines.hardcoded.hh_realtime_exact_audit import (
    RealtimeExactAuditHelper,
    build_exact_audit_helper_for_controller,
    run_controller_with_exact_audit,
)
from pipelines.hardcoded.hh_realtime_measurement import DerivedGeometryMemo, ExactCheckpointValueCache
from pipelines.hardcoded.hh_realtime_measurement import OracleCheckpointValueCache
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
from pipelines.time_dynamics.legacy.checkpoint_progress import write_json_atomic
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm, hamiltonian_matrix
from test_support.json_parity import assert_json_parity


def _basis(idx: int) -> np.ndarray:
    out = np.zeros(2, dtype=complex)
    out[int(idx)] = 1.0
    return out


def _state_sha256(psi: np.ndarray) -> str:
    arr = np.asarray(psi, dtype=np.complex128).reshape(-1)
    payload = np.ascontiguousarray(
        np.stack([arr.real, arr.imag], axis=1).astype("<f8", copy=False)
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


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


def _drive_cfg(*, drive_t0: float = 0.0, exact_steps_multiplier: int = 4) -> ControllerDriveConfig:
    return ControllerDriveConfig(
        enabled=True,
        n_sites=2,
        ordering="blocked",
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        drive_phi=0.0,
        drive_pattern="staggered",
        drive_custom_weights=None,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        drive_t0=float(drive_t0),
        exact_steps_multiplier=int(exact_steps_multiplier),
    )


def _controller_checkpoint_geometry(
    controller: RealtimeCheckpointController,
    *,
    checkpoint_index: int = 0,
    time_start: float = 0.0,
    time_stop: float | None = 0.1,
) -> tuple[object, ExactCheckpointValueCache, DerivedGeometryMemo, dict[str, object]]:
    psi_current = controller.current_executor.prepare_state(
        controller.current_theta, controller.replay_context.psi_ref
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=int(checkpoint_index),
        time_start=float(time_start),
        time_stop=(None if time_stop is None else float(time_stop)),
        scaffold_labels=controller._current_scaffold_labels(),
        theta=controller.current_theta,
        psi=psi_current,
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family=str(controller.replay_context.family_info.get("resolved", "unknown")),
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
    )
    baseline = controller._baseline_geometry(checkpoint_ctx, cache, geometry_memo)
    return checkpoint_ctx, cache, geometry_memo, baseline


def _baseline_geometry_payload(controller: RealtimeCheckpointController) -> dict[str, object]:
    _checkpoint_ctx, _cache, _geometry_memo, baseline = _controller_checkpoint_geometry(controller)
    return baseline


def _no_harm_controller(**cfg_kwargs: object) -> RealtimeCheckpointController:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    cfg_base = {
        "mode": "exact_v1",
        "miss_threshold": 0.0,
        "gain_ratio_threshold": 1.0e-9,
        "append_margin_abs": 1.0e-12,
    }
    cfg_base.update(cfg_kwargs)
    return RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(**cfg_base),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )


def _no_harm_candidate_record(
    label: str = "candidate_a",
    *,
    gain_ratio: float = 1.0,
    gain_exact: float = 1.0,
    confirm_score: float = 2.0,
    pool_index: int = 0,
) -> dict[str, object]:
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


def _no_harm_forecast(
    *,
    score: float,
    rho_miss: float,
    condition: float,
    step_gain: float,
    displacement: float,
    step_residual: float = 0.1,
    **extra: object,
) -> dict[str, object]:
    return {
        "forecast_mode": "local_projective_v1",
        "local_projective_score_total": float(score),
        "tracking_score_horizon": float(score),
        "rho_miss_next": float(rho_miss),
        "condition_number_next": float(condition),
        "step_gain_ratio_next": float(step_gain),
        "predicted_displacement_next": float(displacement),
        "epsilon_step_ratio_next": float(step_residual),
        "step_residual_ratio_next": float(step_residual),
        "rows": [
            {
                "rho_miss": float(rho_miss),
                "condition_number": float(condition),
                "step_gain_ratio": float(step_gain),
                "predicted_displacement": float(displacement),
                "epsilon_step_ratio": float(step_residual),
            }
        ],
        **extra,
    }


def _calm_motion() -> MotionSchedulerTelemetry:
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


def _kink_motion() -> MotionSchedulerTelemetry:
    return MotionSchedulerTelemetry(
        regime="kink",
        direction_cosine=-1.0,
        rate_change_l2=1.0,
        rate_change_ratio=1.0,
        acceleration_l2=1.0,
        curvature_cosine=-1.0,
        direction_reversal=True,
        curvature_sign_flip=True,
        kink_score=1.0,
    )


def _steady_motion() -> MotionSchedulerTelemetry:
    return MotionSchedulerTelemetry(
        regime="steady",
        direction_cosine=0.75,
        rate_change_l2=0.10,
        rate_change_ratio=0.20,
        acceleration_l2=0.10,
        curvature_cosine=0.50,
        direction_reversal=False,
        curvature_sign_flip=False,
        kink_score=0.20,
    )


def _strict_oracle_config(*, noise_mode: str = "ideal") -> OracleConfig:
    return OracleConfig(
        noise_mode=str(noise_mode),
        shots=32,
        oracle_repeats=1,
        oracle_aggregate="mean",
        allow_aer_fallback=False,
        mitigation={"mode": "none"},
        symmetry_mitigation={"mode": "off"},
    )


def _strict_controller(
    *,
    miss_threshold: float = 2.0,
    cfg_overrides: dict[str, object] | None = None,
    oracle_noise_mode: str = "ideal",
) -> RealtimeCheckpointController:
    replay_context, h_poly, _hmat, psi_initial = _toy_context(theta_x=0.2)
    cfg_kwargs: dict[str, object] = {
        "mode": "oracle_v1",
        "reference_mode": "off",
        "miss_threshold": float(miss_threshold),
        "integrator_policy": "euler",
        "append_no_harm_guard_enabled": False,
    }
    if cfg_overrides:
        cfg_kwargs.update(dict(cfg_overrides))
    return RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(**cfg_kwargs),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(noise_mode=str(oracle_noise_mode)),
        strict_qpu_hh=True,
    )


def _strict_driven_controller(
    *,
    miss_threshold: float = 2.0,
    oracle_noise_mode: str = "ideal",
    integrator_policy: str = "euler",
) -> RealtimeCheckpointController:
    replay_context, h_poly, _hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    drive_cfg = ControllerDriveConfig(
        enabled=True,
        n_sites=1,
        ordering="blocked",
        drive_A=0.55,
        drive_omega=2.0,
        drive_tbar=4.0,
        drive_phi=0.0,
        drive_pattern="staggered",
        drive_custom_weights=None,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        drive_t0=4.0,
        exact_steps_multiplier=1,
    )
    return RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=float(miss_threshold),
            integrator_policy=str(integrator_policy),
            append_no_harm_guard_enabled=False,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        drive_config=drive_cfg,
        oracle_base_config=_strict_oracle_config(noise_mode=str(oracle_noise_mode)),
        strict_qpu_hh=True,
    )


def _strict_measured_baseline(
    controller: RealtimeCheckpointController,
    *,
    rho_miss: float = 0.0,
) -> dict[str, object]:
    theta_dot = np.zeros_like(np.asarray(controller.current_theta, dtype=float))
    summary = BaselineGeometrySummary(
        energy=-1.0,
        variance=1.0,
        epsilon_proj_sq=float(rho_miss),
        epsilon_step_sq=float(rho_miss),
        rho_miss=float(rho_miss),
        rho_real=float(rho_miss),
        rho_num=0.0,
        step_objective_value=0.0,
        step_gain_ratio=0.0,
        theta_dot_l2=0.0,
        matrix_rank=int(theta_dot.size),
        condition_number=1.0,
        regularization_lambda=1.0e-8,
        solve_mode="grouped_raw_measured",
        logical_block_count=int(controller.current_layout.logical_parameter_count),
        runtime_parameter_count=int(controller.current_layout.runtime_parameter_count),
        planning_summary={},
        exact_cache_summary={},
    )
    return {
        "summary": summary,
        "theta_dot_step": theta_dot,
        "backend_info": {"noise_mode": "shots", "strict_test": True},
        "observable_estimates": {"baseline": {"mean": -1.0}},
        "raw_group_pool_summary": {},
        "plan_stats": {},
        "rho_real": float(rho_miss),
        "rho_num": 0.0,
    }


def test_strict_qpu_hh_rejects_dense_hmat_boundary() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)

    with pytest.raises(ValueError, match="strict_qpu_hh forbids dense hmat"):
        RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(
                mode="oracle_v1",
                reference_mode="off",
                integrator_policy="euler",
                append_no_harm_guard_enabled=False,
            ),
            replay_context=replay_context,
            h_poly=h_poly,
            hmat=hmat,
            psi_initial=psi_initial,
            best_theta=[0.2],
            allow_repeats=False,
            t_final=0.1,
            num_times=2,
            oracle_base_config=_strict_oracle_config(),
            strict_qpu_hh=True,
        )


def _non_basis_state_prep_contract(
    *,
    psi_ref: np.ndarray,
    psi_initial: np.ndarray,
    ansatz_source: str = "uncoupled_ground",
    ansatz_kind: str = "reference_state",
    ansatz_source_allowlist: tuple[str, ...] | None = None,
    initial_source: str = "adapt_vqe",
    initial_kind: str = "prepared_state",
    initial_source_allowlist: tuple[str, ...] = ("adapt_vqe", "reconstructed_from_scaffold"),
) -> dict[str, object]:
    ansatz_allowlist = (
        tuple(ansatz_source_allowlist)
        if ansatz_source_allowlist is not None
        else (str(ansatz_source),)
    )
    return {
        "version": "strict_state_prep_v1",
        "role": "prepared_seed_state_only",
        "feeds_controller_decisions": "prepared_ansatz_observables_only",
        "exact_target_or_reference_trajectory": False,
        "ansatz_input_state": {
            "role": "ansatz_input_state",
            "source_location": "payload.ansatz_input_state",
            "source": str(ansatz_source),
            "source_allowlist": [str(item) for item in ansatz_allowlist],
            "handoff_state_kind": str(ansatz_kind),
            "state_sha256": _state_sha256(np.asarray(psi_ref, dtype=complex)),
        },
        "initial_state": {
            "role": "prepared_ansatz_state",
            "source_location": "payload.initial_state",
            "source": str(initial_source),
            "source_allowlist": [str(item) for item in initial_source_allowlist],
            "handoff_state_kind": str(initial_kind),
            "state_sha256": _state_sha256(np.asarray(psi_initial, dtype=complex)),
        },
    }


def _resolved_problem_with_reference(psi_ref: np.ndarray, *, family_key: str = "hh") -> SimpleNamespace:
    psi = np.asarray(psi_ref, dtype=complex).reshape(-1)
    return SimpleNamespace(
        family_key=str(family_key),
        reference_state=SimpleNamespace(build_state=lambda: np.array(psi, copy=True)),
    )


def test_strict_qpu_hh_rejects_non_basis_reference_state_without_seed_contract() -> None:
    replay_context, h_poly, _hmat, psi_initial = _toy_context(theta_x=0.2)
    bad_ref = np.asarray([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    replay_context = dataclass_replace(replay_context, psi_ref=bad_ref)

    with pytest.raises(ValueError, match="non-basis state prep requires seed/artifact"):
        RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(
                mode="oracle_v1",
                reference_mode="off",
                integrator_policy="euler",
                append_no_harm_guard_enabled=False,
            ),
            replay_context=replay_context,
            h_poly=h_poly,
            hmat=None,
            psi_initial=psi_initial,
            best_theta=[0.2],
            allow_repeats=False,
            t_final=0.1,
            num_times=2,
            oracle_base_config=_strict_oracle_config(),
            strict_qpu_hh=True,
        )


def test_strict_qpu_faithful_accepts_non_basis_seed_ansatz_input_state() -> None:
    replay_context, h_poly, _hmat, _psi_initial = _toy_context(theta_x=0.2)
    psi_ref = np.asarray([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    executor = CompiledAnsatzExecutor(
        list(replay_context.replay_terms),
        parameterization_mode="per_pauli_term",
        parameterization_layout=replay_context.base_layout,
    )
    theta = np.asarray([0.2], dtype=float)
    psi_initial = executor.prepare_state(theta, psi_ref)
    replay_context = dataclass_replace(
        replay_context,
        psi_ref=psi_ref,
        pool_meta={
            **dict(replay_context.pool_meta),
            "strict_state_prep_contract": _non_basis_state_prep_contract(
                psi_ref=psi_ref,
                psi_initial=psi_initial,
            ),
        },
    )

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="euler",
            append_no_harm_guard_enabled=False,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=theta,
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(),
        resolved_problem=_resolved_problem_with_reference(psi_ref),
        strict_qpu_faithful=True,
    )

    contract = controller._strict_state_prep_contract
    assert controller.strict_qpu_faithful is True
    assert contract["state_prep_kind"] == "non_basis_seed_ansatz_input"
    assert contract["non_basis_ansatz_input_allowed"] is True
    assert contract["prepared_state_reconstruction_passed"] is True
    assert contract["ansatz_input_state"]["source"] == "uncoupled_ground"


def test_strict_qpu_faithful_rejects_exact_target_state_prep_metadata() -> None:
    replay_context, h_poly, _hmat, _psi_initial = _toy_context(theta_x=0.2)
    psi_ref = np.asarray([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    executor = CompiledAnsatzExecutor(
        list(replay_context.replay_terms),
        parameterization_mode="per_pauli_term",
        parameterization_layout=replay_context.base_layout,
    )
    theta = np.asarray([0.2], dtype=float)
    psi_initial = executor.prepare_state(theta, psi_ref)
    replay_context = dataclass_replace(
        replay_context,
        psi_ref=psi_ref,
        pool_meta={
            **dict(replay_context.pool_meta),
            "strict_state_prep_contract": _non_basis_state_prep_contract(
                psi_ref=psi_ref,
                psi_initial=psi_initial,
                ansatz_source="static_hamiltonian_ed_ground_state",
                ansatz_source_allowlist=("static_hamiltonian_ed_ground_state",),
            ),
        },
    )

    with pytest.raises(ValueError, match="forbids exact-target/reference state prep metadata"):
        RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(
                mode="oracle_v1",
                reference_mode="off",
                miss_threshold=2.0,
                integrator_policy="euler",
                append_no_harm_guard_enabled=False,
                measurement_active_window_size=2,
            ),
            replay_context=replay_context,
            h_poly=h_poly,
            hmat=None,
            psi_initial=psi_initial,
            best_theta=theta,
            allow_repeats=False,
            t_final=0.1,
            num_times=2,
            oracle_base_config=_strict_oracle_config(),
            resolved_problem=_resolved_problem_with_reference(psi_ref),
            strict_qpu_faithful=True,
        )


def test_strict_qpu_faithful_rejects_non_basis_seed_not_matching_trusted_reference() -> None:
    replay_context, h_poly, _hmat, _psi_initial = _toy_context(theta_x=0.2)
    psi_ref = np.asarray([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    executor = CompiledAnsatzExecutor(
        list(replay_context.replay_terms),
        parameterization_mode="per_pauli_term",
        parameterization_layout=replay_context.base_layout,
    )
    theta = np.asarray([0.2], dtype=float)
    psi_initial = executor.prepare_state(theta, psi_ref)
    replay_context = dataclass_replace(
        replay_context,
        psi_ref=psi_ref,
        pool_meta={
            **dict(replay_context.pool_meta),
            "strict_state_prep_contract": _non_basis_state_prep_contract(
                psi_ref=psi_ref,
                psi_initial=psi_initial,
            ),
        },
    )

    with pytest.raises(ValueError, match="must match resolved_problem.reference_state"):
        RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(
                mode="oracle_v1",
                reference_mode="off",
                miss_threshold=2.0,
                integrator_policy="euler",
                append_no_harm_guard_enabled=False,
            ),
            replay_context=replay_context,
            h_poly=h_poly,
            hmat=None,
            psi_initial=psi_initial,
            best_theta=theta,
            allow_repeats=False,
            t_final=0.1,
            num_times=2,
            oracle_base_config=_strict_oracle_config(),
            resolved_problem=_resolved_problem_with_reference(_basis(0)),
            strict_qpu_faithful=True,
        )


def test_strict_qpu_faithful_takes_precedence_over_false_legacy_alias() -> None:
    replay_context, h_poly, _hmat, psi_initial = _toy_context(theta_x=0.2)

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="euler",
            append_no_harm_guard_enabled=False,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(),
        strict_qpu_faithful=True,
        strict_qpu_hh=False,
    )

    assert controller.strict_qpu_faithful is True
    assert controller.strict_qpu_hh is True


def test_strict_qpu_hh_rejects_incompatible_controller_config() -> None:
    with pytest.raises(ValueError, match="strict_qpu_faithful.*mode=observable_v1 or oracle_v1"):
        _strict_controller(cfg_overrides={"mode": "exact_v1"})
    assert _strict_controller(cfg_overrides={"integrator_policy": "rk4"})._integrator_policy() == "rk4"
    assert (
        _strict_controller(
            cfg_overrides={"integrator_policy": "auto_euler_rk4"}
        )._integrator_policy()
        == "auto_euler_rk4"
    )
    with pytest.raises(ValueError, match="append-no-harm"):
        _strict_controller(cfg_overrides={"append_no_harm_guard_enabled": True})
    with pytest.raises(ValueError, match="oracle_selection_policy"):
        _strict_controller(
            cfg_overrides={"oracle_selection_policy": "measured_topk_oracle_energy"}
        )
    shots_controller = _strict_controller(oracle_noise_mode="shots")
    assert shots_controller._oracle_base_config.noise_mode == "shots"
    ideal_controller = _strict_controller(oracle_noise_mode="ideal")
    assert ideal_controller._oracle_base_config.noise_mode == "ideal"


def test_strict_observable_v1_accepts_no_oracle_config() -> None:
    replay_context, h_poly, _hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="observable_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="euler",
            append_no_harm_guard_enabled=True,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=None,
        strict_qpu_faithful=True,
    )

    result = controller.run()

    assert result.summary["mode"] == "observable_v1"
    assert result.summary["strict_qpu_faithful"] is True
    assert result.summary["qpu_faithful_decisions_passed"] is True
    assert result.summary["decision_backend"] == "ideal_observable"
    assert result.summary["ideal_observable_decision_checkpoints"] == 2
    assert result.summary["exact_decision_checkpoints"] == 0
    assert set(result.summary["executed_decision_backends"]) == {"ideal_observable"}
    assert result.summary["controller_exact_input_mode"] == "off"
    assert result.summary["decision_data_flow"] == "ideal_observable_estimator"
    assert result.summary["uses_reference_for_decision"] is False
    assert result.summary["uses_future_exact_forecast_for_decision"] is False
    assert result.summary["uses_statevector_as_ideal_observable_estimator"] is True
    assert result.summary["strict_measurement_oracle_certified"] is True
    assert all(
        row["decision_data_flow"] == "ideal_observable_estimator"
        for row in result.ledger
    )


def test_selected_runtime_observable_plan_bounds_pair_surface() -> None:
    from pipelines.hardcoded.hh_fixed_manifold_observables import (
        build_checkpoint_observable_plan_from_layout,
    )

    replay_context, h_poly, _hmat, _psi_initial = _multi_block_context(block_count=5)
    plan = build_checkpoint_observable_plan_from_layout(
        replay_context.base_layout,
        replay_context.adapt_theta_runtime,
        psi_ref=replay_context.psi_ref,
        h_poly=h_poly,
        selected_runtime_indices=(3, 4),
    )

    assert [int(rot.runtime_index) for rot in plan.runtime_rotations] == [3, 4]
    assert plan.stats["observable_selection_mode"] == "runtime_subset"
    assert int(plan.stats["total_runtime_parameter_count"]) == 5
    assert int(plan.stats["runtime_parameter_count"]) == 2
    assert int(plan.stats["pair_anticommutator_count"]) == 1
    assert int(plan.stats["selected_observable_count_total"]) == 7


def test_strict_ideal_measurement_active_window_expands_baseline_shapes() -> None:
    replay_context, h_poly, _hmat, psi_initial = _multi_block_context(block_count=5)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="euler",
            append_no_harm_guard_enabled=False,
            measurement_active_window_size=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(noise_mode="ideal"),
        strict_qpu_faithful=True,
    )
    checkpoint_ctx = make_measurement_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=controller._current_scaffold_labels(),
        theta=controller.current_theta,
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_multi_block",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )

    assert controller._selected_measurement_runtime_indices(
        layout=controller.current_layout
    ) == (3, 4)
    measured = controller._oracle_measured_baseline_geometry(
        checkpoint_ctx=checkpoint_ctx,
        cache=ExactCheckpointValueCache(
            checkpoint_id=str(checkpoint_ctx.checkpoint_id),
            grouping_mode=str(controller.cfg.grouping_mode),
        ),
        geometry_memo=DerivedGeometryMemo(
            checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        ),
        raw_group_pool=None,
        h_poly_step=h_poly,
        tier_name="confirm",
    )

    assert np.asarray(measured["G"]).shape == (5, 5)
    assert np.asarray(measured["theta_dot_step"]).shape == (5,)
    assert measured["plan_stats"]["observable_selection_mode"] == "runtime_subset"
    assert measured["plan_stats"]["selected_runtime_indices"] == [3, 4]
    assert int(measured["plan_stats"]["selected_pair_anticommutator_count"]) == 1


def test_strict_ideal_measurement_active_window_is_ignored_outside_strict_ideal() -> None:
    replay_context, h_poly, _hmat, psi_initial = _multi_block_context(block_count=5)
    strict_shots = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="euler",
            append_no_harm_guard_enabled=False,
            measurement_active_window_size=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(noise_mode="shots"),
        strict_qpu_faithful=True,
    )
    non_strict_ideal = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="euler",
            append_no_harm_guard_enabled=False,
            measurement_active_window_size=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=np.asarray(hamiltonian_matrix(h_poly), dtype=complex),
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(noise_mode="ideal"),
    )

    assert strict_shots._selected_measurement_runtime_indices(
        layout=strict_shots.current_layout
    ) is None
    assert non_strict_ideal._selected_measurement_runtime_indices(
        layout=non_strict_ideal.current_layout
    ) is None


def test_strict_active_window_append_forced_euler_preserves_outside_tail_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, _hmat, psi_initial = _multi_block_context(block_count=5)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="auto_euler_rk4",
            append_no_harm_guard_enabled=False,
            measurement_active_window_size=1,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(noise_mode="ideal"),
        strict_qpu_faithful=True,
    )
    assert controller._selected_measurement_runtime_indices(
        layout=controller.current_layout
    ) == (4,)

    def _stage_should_not_run(**kwargs):
        raise AssertionError("active-window append commits must not rewindow through RK stages")

    monkeypatch.setattr(
        controller,
        "_strict_qpu_hh_measured_integrator_stage_baseline",
        _stage_should_not_run,
    )
    theta0 = np.asarray(controller.current_theta, dtype=float).copy()
    theta_dot = np.zeros_like(theta0)
    theta_dot[0] = 0.5
    theta_next, theta_dot_out, diagnostics = controller._strict_qpu_hh_integrate_theta_one_step(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        layout=controller.current_layout,
        theta_runtime=theta0,
        baseline={
            "theta_dot_step": np.zeros_like(theta0),
            "summary": SimpleNamespace(condition_number=1.0, rho_miss=0.0),
        },
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
        tier_name="confirm",
        budget_scale=1.0,
        euler_theta_dot=theta_dot,
        forced_policy="euler",
        forced_policy_reason="measurement_active_window_append_euler",
    )

    np.testing.assert_allclose(theta_dot_out, theta_dot)
    np.testing.assert_allclose(theta_next, theta0 + 0.1 * theta_dot)
    assert diagnostics["integrator_policy"] == "euler"
    assert diagnostics["integrator_used"] == "euler"
    assert diagnostics["integrator_forced_policy"] == "euler"
    assert diagnostics["integrator_forced_policy_reason"] == "measurement_active_window_append_euler"


def test_strict_ideal_measurement_active_window_expands_incremental_candidate_shapes() -> None:
    replay_context, h_poly, _hmat, psi_initial = _multi_block_context(block_count=5)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            miss_threshold=0.0,
            integrator_policy="euler",
            append_no_harm_guard_enabled=False,
            measurement_active_window_size=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(noise_mode="ideal"),
        strict_qpu_faithful=True,
    )
    checkpoint_ctx = make_measurement_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=controller._current_scaffold_labels(),
        theta=controller.current_theta,
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_multi_block",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline = controller._oracle_measured_baseline_geometry(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        raw_group_pool=None,
        h_poly_step=h_poly,
        tier_name="confirm",
    )
    candidate_term = tuple(replay_context.family_pool)[-1]
    candidate_data = controller._strict_qpu_hh_candidate_data(
        candidate_term=candidate_term,
        candidate_pool_index=int(len(tuple(replay_context.family_pool)) - 1),
        position_id=3,
    )
    measured_candidate = controller._oracle_measured_candidate_incremental_block(
        checkpoint_ctx=checkpoint_ctx,
        geometry_memo=geometry_memo,
        raw_group_pool=None,
        tier_name="confirm",
        baseline_measured=baseline,
        record={
            "candidate_label": str(candidate_term.label),
            "candidate_identity": f"{candidate_term.label}__pool5",
            "position_id": 3,
            "candidate_data": candidate_data,
        },
        h_poly_step=h_poly,
    )

    assert np.asarray(measured_candidate["B"]).shape == (5, 1)
    assert np.asarray(measured_candidate["theta_dot_aug_existing"]).shape == (5,)
    assert np.asarray(measured_candidate["theta_dot_step"]).shape == (6,)
    assert measured_candidate["measurement_active_baseline_runtime_indices"] == [3, 4]
    assert measured_candidate["measurement_active_augmented_runtime_indices"] == [3, 4, 5]
    assert measured_candidate["plan_stats"]["observable_selection_mode"] == "runtime_subset"


@pytest.mark.parametrize(
    ("family_key", "expected_error"),
    [
        ("ttprime_hubbard", "no drive-term seam"),
        ("spinless_tv", "no drive-term seam"),
        ("bose_hubbard", "no drive-term seam"),
    ],
)
def test_driven_neutral_additional_families_require_resolved_problem_metadata(
    monkeypatch: pytest.MonkeyPatch,
    family_key: str,
    expected_error: str,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    monkeypatch.setattr(
        controller_mod,
        "_controller_family_key",
        lambda *, resolved_problem=None, replay_context=None: family_key,
    )

    with pytest.raises(ValueError, match=expected_error):
        RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(mode="exact_v1", reference_mode="benchmark_exact"),
            replay_context=replay_context,
            h_poly=h_poly,
            hmat=hmat,
            psi_initial=psi_initial,
            best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
            allow_repeats=False,
            t_final=0.1,
            num_times=2,
            drive_config=_drive_cfg(),
        )


def test_strict_qpu_hh_accepts_driven_construction_without_dense_hamiltonian() -> None:
    controller = _strict_driven_controller()

    assert controller.strict_qpu_faithful is True
    assert controller.strict_qpu_hh is True
    assert controller.hmat is None
    assert controller._drive_config is not None
    assert controller._drive_config.drive_A == pytest.approx(0.55)
    assert controller._drive_aligned_density_active is True
    assert controller._drive_aligned_density_label == "drive_aligned_density(pattern=staggered)"
    assert controller.current_layout.runtime_parameter_count > 1
    step = controller._step_hamiltonian_artifacts(0.05)
    assert step.physical_time == pytest.approx(4.05)
    assert step.drive_term_count > 0
    assert step.hmat.shape == (0, 0)
    assert step.oracle_observable is not None


def test_strict_qpu_hh_driven_checkpoint_path_does_not_touch_exact_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _strict_driven_controller(
        miss_threshold=2.0,
        oracle_noise_mode="ideal",
        integrator_policy="auto_euler_rk4",
    )

    def _sentinel(*args, **kwargs):
        raise AssertionError("strict driven route touched exact decision helper")

    monkeypatch.setattr(controller_mod, "make_checkpoint_context", _sentinel)
    monkeypatch.setattr(controller_mod, "hamiltonian_matrix", _sentinel)
    monkeypatch.setattr(controller, "_baseline_geometry", _sentinel)
    monkeypatch.setattr(controller, "_confirm_candidates", _sentinel)
    monkeypatch.setattr(controller, "_select_action_exact_v1", _sentinel)
    monkeypatch.setattr(controller, "_exact_step_forecast", _sentinel)
    monkeypatch.setattr(controller, "_integrate_theta_one_step", _sentinel)
    monkeypatch.setattr(controller, "_integrator_stage_baseline", _sentinel)
    monkeypatch.setattr(
        type(controller.current_executor),
        "prepare_state",
        lambda self, *args, **kwargs: _sentinel(*args, **kwargs),
    )

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["strict_fail_closed"] is False
    assert result.summary["qpu_faithful_decisions_passed"] is True
    assert result.summary["controller_exact_input_mode"] == "off"
    assert result.summary["decision_data_flow"] == "ideal_observable_estimator"
    assert result.summary["uses_reference_for_decision"] is False
    assert result.summary["uses_future_exact_forecast_for_decision"] is False
    assert result.summary["uses_statevector_as_ideal_observable_estimator"] is True
    assert result.summary["strict_measurement_oracle_certified"] is True
    assert result.summary["exact_decision_checkpoints"] == 0
    assert result.ledger
    assert result.ledger[0]["physical_time"] == pytest.approx(4.05)
    assert any(int(row["drive_term_count"]) > 0 for row in result.ledger)
    assert {row["decision_backend"] for row in result.ledger} == {"oracle"}
    assert {row["decision_data_flow"] for row in result.ledger} == {
        "ideal_observable_estimator"
    }
    assert all(row["integrator_used"] in {"euler", "rk4", "none"} for row in result.ledger)
    assert all(row["integrator_policy"] == "auto_euler_rk4" for row in result.ledger)
    assert any(float(row["baseline_geometry"]["theta_dot_l2"]) > 1.0e-6 for row in result.ledger)
    assert all(row["observable_telemetry_supported"] is True for row in result.ledger)
    assert all(row["site_occupations"] for row in result.ledger)
    assert all(row["site_occupations_up"] for row in result.ledger)
    assert all(row["site_occupations_dn"] for row in result.ledger)
    assert all(row["staggered"] is not None for row in result.ledger)
    assert all(row["doublon"] is not None for row in result.ledger)
    assert all(row["primary_density"] is not None for row in result.ledger)


def test_strict_qpu_hh_driven_measured_baseline_matches_exact_local_geometry() -> None:
    controller = _strict_driven_controller(miss_threshold=2.0, oracle_noise_mode="ideal")
    time_start = 0.0
    time_stop = 0.1
    sample_time = controller._projection_sample_time(time_start, time_stop)
    step_hamiltonian = controller._step_hamiltonian_artifacts(sample_time)
    measurement_ctx = make_measurement_checkpoint_context(
        checkpoint_index=0,
        time_start=float(time_start),
        time_stop=float(time_stop),
        scaffold_labels=controller._current_scaffold_labels(),
        theta=controller.current_theta,
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="hh",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    measured = controller._oracle_measured_baseline_geometry(
        checkpoint_ctx=measurement_ctx,
        cache=ExactCheckpointValueCache(
            checkpoint_id=str(measurement_ctx.checkpoint_id),
            grouping_mode=str(controller.cfg.grouping_mode),
        ),
        geometry_memo=DerivedGeometryMemo(checkpoint_id=str(measurement_ctx.checkpoint_id)),
        raw_group_pool=None,
        h_poly_step=step_hamiltonian.h_poly,
        tier_name="confirm",
    )

    # Diagnostic comparison only: exact local state/tangent geometry is allowed in this test
    # to prove the measurement observable path matches the local McLachlan equations.
    psi = controller.current_executor.prepare_state(
        controller.current_theta,
        controller.replay_context.psi_ref,
    )
    exact_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=float(time_start),
        time_stop=float(time_stop),
        scaffold_labels=controller._current_scaffold_labels(),
        theta=controller.current_theta,
        psi=psi,
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="hh",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    exact = controller._compute_baseline_geometry_for_runtime_state(
        checkpoint_ctx=exact_ctx,
        cache=ExactCheckpointValueCache(
            checkpoint_id=str(exact_ctx.checkpoint_id),
            grouping_mode=str(controller.cfg.grouping_mode),
        ),
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        planning_audit=controller._planning_audit,
        step_hamiltonian=step_hamiltonian,
    )

    measured_f = np.asarray(measured["f"], dtype=float).reshape(-1)
    exact_f = np.asarray(exact["f"], dtype=float).reshape(-1)
    measured_theta_dot = np.asarray(measured["theta_dot_step"], dtype=float).reshape(-1)
    exact_theta_dot = np.asarray(exact["theta_dot_step"], dtype=float).reshape(-1)
    assert step_hamiltonian.drive_term_count > 0
    assert np.linalg.norm(measured_f) > 1.0e-6
    assert np.linalg.norm(measured_theta_dot) > 1.0e-6
    np.testing.assert_allclose(measured_f, exact_f, rtol=1.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(measured_theta_dot, exact_theta_dot, rtol=1.0e-5, atol=1.0e-7)
    assert measured["summary"].rho_miss == pytest.approx(exact["summary"].rho_miss, rel=1.0e-5, abs=1.0e-7)


def test_strict_qpu_hh_driven_measured_rk4_matches_exact_local_rk4() -> None:
    strict_controller = _strict_driven_controller(
        miss_threshold=2.0,
        oracle_noise_mode="ideal",
        integrator_policy="rk4",
    )
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    exact_controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="rk4",
            append_no_harm_guard_enabled=False,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        drive_config=strict_controller._drive_config,
    )
    time_start = 0.0
    time_stop = 0.1
    sample_time = strict_controller._projection_sample_time(time_start, time_stop)
    strict_step = strict_controller._step_hamiltonian_artifacts(sample_time)
    measurement_ctx = make_measurement_checkpoint_context(
        checkpoint_index=0,
        time_start=float(time_start),
        time_stop=float(time_stop),
        scaffold_labels=strict_controller._current_scaffold_labels(),
        theta=strict_controller.current_theta,
        logical_count=int(strict_controller.current_layout.logical_parameter_count),
        runtime_count=int(strict_controller.current_layout.runtime_parameter_count),
        resolved_family="hh",
        grouping_mode=str(strict_controller.cfg.grouping_mode),
        structure_locked=False,
    )
    measured_baseline = strict_controller._oracle_measured_baseline_geometry(
        checkpoint_ctx=measurement_ctx,
        cache=ExactCheckpointValueCache(
            checkpoint_id=str(measurement_ctx.checkpoint_id),
            grouping_mode=str(strict_controller.cfg.grouping_mode),
        ),
        geometry_memo=DerivedGeometryMemo(checkpoint_id=str(measurement_ctx.checkpoint_id)),
        raw_group_pool=None,
        h_poly_step=strict_step.h_poly,
        tier_name="confirm",
    )
    measured_next, measured_dot, measured_diag = (
        strict_controller._strict_qpu_hh_integrate_theta_one_step(
            checkpoint_index=0,
            time_start=float(time_start),
            time_stop=float(time_stop),
            layout=strict_controller.current_layout,
            theta_runtime=strict_controller.current_theta,
            baseline=measured_baseline,
            planning_audit=strict_controller._planning_audit,
            scaffold_labels=strict_controller._current_scaffold_labels(),
            tier_name="confirm",
            budget_scale=1.0,
        )
    )

    exact_step = exact_controller._step_hamiltonian_artifacts(sample_time)
    psi = exact_controller.current_executor.prepare_state(
        exact_controller.current_theta,
        exact_controller.replay_context.psi_ref,
    )
    exact_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=float(time_start),
        time_stop=float(time_stop),
        scaffold_labels=exact_controller._current_scaffold_labels(),
        theta=exact_controller.current_theta,
        psi=psi,
        logical_count=int(exact_controller.current_layout.logical_parameter_count),
        runtime_count=int(exact_controller.current_layout.runtime_parameter_count),
        resolved_family="hh",
        grouping_mode=str(exact_controller.cfg.grouping_mode),
        structure_locked=False,
    )
    exact_baseline = exact_controller._compute_baseline_geometry_for_runtime_state(
        checkpoint_ctx=exact_ctx,
        cache=ExactCheckpointValueCache(
            checkpoint_id=str(exact_ctx.checkpoint_id),
            grouping_mode=str(exact_controller.cfg.grouping_mode),
        ),
        executor=exact_controller.current_executor,
        layout=exact_controller.current_layout,
        theta_runtime=exact_controller.current_theta,
        planning_audit=exact_controller._planning_audit,
        step_hamiltonian=exact_step,
    )
    exact_next, exact_dot, exact_diag = exact_controller._integrate_theta_one_step(
        checkpoint_index=0,
        time_start=float(time_start),
        time_stop=float(time_stop),
        executor=exact_controller.current_executor,
        layout=exact_controller.current_layout,
        theta_runtime=exact_controller.current_theta,
        baseline=exact_baseline,
        planning_audit=exact_controller._planning_audit,
        scaffold_labels=exact_controller._current_scaffold_labels(),
    )

    assert measured_diag["integrator_used"] == "rk4"
    assert exact_diag["integrator_used"] == "rk4"
    np.testing.assert_allclose(measured_dot, exact_dot, rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(measured_next, exact_next, rtol=1.0e-5, atol=1.0e-7)


def test_strict_qpu_hh_driven_measured_observable_telemetry_matches_exact_snapshot() -> None:
    controller = _strict_driven_controller(miss_threshold=2.0, oracle_noise_mode="ideal")
    measurement_ctx = make_measurement_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=controller._current_scaffold_labels(),
        theta=controller.current_theta,
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="hh",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    telemetry = controller._strict_qpu_hh_measured_observable_telemetry(
        checkpoint_ctx=measurement_ctx,
        raw_group_pool=None,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        tier_name="confirm",
        budget_scale=1.0,
    )

    # Diagnostic exact-state snapshot is used only in the test assertion; the
    # strict telemetry helper above goes through the oracle observable interface.
    psi = controller.current_executor.prepare_state(
        controller.current_theta,
        controller.replay_context.psi_ref,
    )
    exact_snapshot = controller._observable_snapshot(psi)
    assert telemetry["observable_telemetry_supported"] is True
    np.testing.assert_allclose(
        telemetry["site_occupations"],
        exact_snapshot["site_occupations"],
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        telemetry["site_occupations_up"],
        exact_snapshot["n_up_site"],
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        telemetry["site_occupations_dn"],
        exact_snapshot["n_dn_site"],
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert telemetry["staggered"] == pytest.approx(exact_snapshot["staggered"], abs=1.0e-10)
    assert telemetry["doublon"] == pytest.approx(exact_snapshot["doublon"], abs=1.0e-10)
    assert telemetry["primary_density"] == pytest.approx(
        controller._primary_density_value_from_snapshot(exact_snapshot),
        abs=1.0e-10,
    )
    assert "primary_density" not in telemetry["observable_telemetry_estimates"]
    assert telemetry["observable_telemetry_kind"] == "oracle_measured"


def test_strict_qpu_hh_constructor_does_not_prepare_exact_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, _hmat, psi_initial = _toy_context(theta_x=0.2)

    def _sentinel(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("strict constructor prepared an exact state")

    monkeypatch.setattr(CompiledAnsatzExecutor, "prepare_state", _sentinel)

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            integrator_policy="euler",
            append_no_harm_guard_enabled=False,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=_strict_oracle_config(),
        strict_qpu_hh=True,
    )

    assert controller.strict_qpu_faithful is True
    assert controller.strict_qpu_hh is True
    assert controller.hmat is None


def test_strict_qpu_hh_rejects_checkpoint_observer() -> None:
    controller = _strict_controller()

    with pytest.raises(ValueError, match="checkpoint_observer"):
        controller.run(checkpoint_observer=SimpleNamespace(on_checkpoint=lambda payload: None))


def test_strict_qpu_faithful_rejects_exact_audit_wrapper() -> None:
    controller = _strict_controller()

    with pytest.raises(ValueError, match="strict QPU-faithful controllers"):
        build_exact_audit_helper_for_controller(controller)
    with pytest.raises(ValueError, match="strict QPU-faithful controllers"):
        run_controller_with_exact_audit(controller)


def test_strict_qpu_hh_sentinel_does_not_touch_exact_decision_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _strict_controller(miss_threshold=2.0)
    monkeypatch.setattr(
        controller,
        "_oracle_measured_baseline_geometry",
        lambda **kwargs: _strict_measured_baseline(controller, rho_miss=0.0),
    )

    def _sentinel(*args, **kwargs):
        raise AssertionError("strict route touched exact decision helper")

    monkeypatch.setattr(controller_mod, "make_checkpoint_context", _sentinel)
    monkeypatch.setattr(controller, "_baseline_geometry", _sentinel)
    monkeypatch.setattr(controller, "_confirm_candidates", _sentinel)
    monkeypatch.setattr(controller, "_select_action_exact_v1", _sentinel)
    monkeypatch.setattr(controller, "_exact_step_forecast", _sentinel)
    monkeypatch.setattr(controller, "_integrate_theta_one_step", _sentinel)
    monkeypatch.setattr(controller, "_integrator_stage_baseline", _sentinel)
    monkeypatch.setattr(
        type(controller.current_executor),
        "prepare_state",
        lambda self, *args, **kwargs: _sentinel(*args, **kwargs),
    )

    result = controller.run()

    assert result.summary["decision_path_kind"] == "strict_qpu_faithful_observable_v1"
    assert result.summary["strict_qpu_faithful"] is True
    assert result.summary["strict_qpu_hh"] is True
    assert result.summary["strict_fail_closed"] is False
    assert result.summary["qpu_faithful_decisions_expected"] is True
    assert result.summary["qpu_faithful_decisions_passed"] is True
    assert result.summary["exact_decision_checkpoints"] == 0
    assert result.summary["reference_enabled"] is False
    assert {row["decision_backend"] for row in result.ledger} == {"oracle"}
    assert all(row["integrator_used"] in {"euler", "none"} for row in result.ledger)


def test_strict_qpu_hh_ideal_baseline_treats_tiny_empty_observables_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qiskit.quantum_info import SparsePauliOp

    controller = _strict_controller(miss_threshold=2.0, oracle_noise_mode="ideal")
    original_builder = measurement_mod.build_checkpoint_observable_plan_from_layout

    def _empty_spec(spec):
        sparse = getattr(spec, "sparse_op", None)
        nq = int(getattr(sparse, "num_qubits", controller._num_qubits))
        tiny_op = SparsePauliOp.from_list([("I" * int(nq), 1.25e-12)]).simplify(atol=1.0e-12)
        return dataclass_replace(spec, sparse_op=tiny_op, term_count=1, is_zero=False)

    def _empty_observable_plan(*args, **kwargs):
        plan = original_builder(*args, **kwargs)
        return dataclass_replace(
            plan,
            energy=_empty_spec(plan.energy),
            variance_h2=_empty_spec(plan.variance_h2),
            generator_means=tuple(_empty_spec(spec) for spec in plan.generator_means),
            pair_anticommutators={
                pair: _empty_spec(spec)
                for pair, spec in plan.pair_anticommutators.items()
            },
            force_anticommutators=tuple(
                _empty_spec(spec) for spec in plan.force_anticommutators
            ),
        )

    monkeypatch.setattr(
        measurement_mod,
        "build_checkpoint_observable_plan_from_layout",
        _empty_observable_plan,
    )
    monkeypatch.setattr(controller, "_oracle_commit_payload", lambda **kwargs: ({}, None))

    def _sentinel(*args, **kwargs):
        raise AssertionError("strict route touched exact decision helper")

    monkeypatch.setattr(controller_mod, "make_checkpoint_context", _sentinel)
    monkeypatch.setattr(controller, "_baseline_geometry", _sentinel)
    monkeypatch.setattr(controller, "_confirm_candidates", _sentinel)
    monkeypatch.setattr(controller, "_select_action_exact_v1", _sentinel)
    monkeypatch.setattr(controller, "_exact_step_forecast", _sentinel)
    monkeypatch.setattr(controller, "_integrate_theta_one_step", _sentinel)
    monkeypatch.setattr(controller, "_integrator_stage_baseline", _sentinel)
    monkeypatch.setattr(
        type(controller.current_executor),
        "prepare_state",
        lambda self, *args, **kwargs: _sentinel(*args, **kwargs),
    )

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["strict_fail_closed"] is False
    assert result.summary["decision_noise_mode"] == "ideal"
    assert result.summary["exact_decision_checkpoints"] == 0
    assert result.ledger
    baseline = result.ledger[0]["baseline_geometry"]
    assert baseline["energy"] == pytest.approx(0.0)
    assert baseline["variance"] == pytest.approx(0.0)
    assert result.ledger[0]["baseline_backend_info"]["noise_mode"] == "ideal"


def test_strict_ideal_observable_helper_zeroes_estimator_empty_error() -> None:
    from qiskit.quantum_info import SparsePauliOp

    spec = measurement_mod.ObservableSpec(
        name="AAsym_11_25",
        kind="pair_anticommutator",
        runtime_index=None,
        runtime_pair=(11, 25),
        poly=None,
        sparse_op=SparsePauliOp.from_list(
            [
                ("YIXYXX", 2.767675310124968e-11),
                ("YIYXXX", -3.1884956211298003e-12),
                ("YYXXXX", 3.676648796950067e-11),
                ("YYYYXX", 3.1914016229510865e-10),
            ]
        ),
        term_count=4,
        is_zero=False,
    )

    class _EmptyEstimatorOracle:
        def evaluate(self, circuit, observable):
            del circuit, observable
            raise RuntimeError(
                "Estimator execution failed across known call paths. Details: "
                "ValueError: Empty observable was detected.; ValueError: Empty observable was detected."
            )

    estimate = measurement_mod._observable_spec_mean(
        raw_group_pool=None,
        oracle=_EmptyEstimatorOracle(),
        circuit=SimpleNamespace(),
        spec=spec,
        observable_family="candidate_incremental_block:AAsym_11_25",
        candidate_label="candidate",
        position_id=0,
        state_key="state",
        min_total_shots=1,
        min_samples=1,
    )

    assert estimate["mean"] == pytest.approx(0.0)
    assert estimate["stderr"] == pytest.approx(0.0)
    assert estimate["n_samples"] == 0


def test_strict_qpu_hh_measured_candidate_scoring_supplies_gain_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _strict_controller(
        miss_threshold=0.1,
        cfg_overrides={"gain_ratio_threshold": 1.0, "append_margin_abs": 1.0},
    )
    candidate = _no_harm_candidate_record(
        label="candidate_a",
        gain_ratio=0.0,
        gain_exact=0.0,
        confirm_score=0.0,
    )
    candidate.update(
        {
            "runtime_insert_position": 0,
            "runtime_block_indices": (0,),
            "residual_overlap_l2": 0.0,
            "compile_proxy_total": 1.0,
            "groups_new": 0.0,
            "position_jump_penalty": 0.0,
        }
    )

    monkeypatch.setattr(
        controller,
        "_oracle_measured_baseline_geometry",
        lambda **kwargs: _strict_measured_baseline(controller, rho_miss=1.0),
    )
    monkeypatch.setattr(
        controller,
        "_strict_qpu_hh_shortlist",
        lambda **kwargs: ([dict(candidate)], [dict(candidate)]),
    )
    monkeypatch.setattr(
        controller,
        "_oracle_confirm_limit_with_selection_policy",
        lambda **kwargs: 1,
    )
    monkeypatch.setattr(
        controller,
        "_oracle_commit_payload",
        lambda **kwargs: ({}, None),
    )

    def _measured_candidate(**kwargs):
        theta_dot = np.zeros_like(np.asarray(controller.current_theta, dtype=float))
        return {
            "theta_dot_step": theta_dot,
            "theta_dot_aug_existing": theta_dot,
            "eta_dot": np.zeros(1, dtype=float),
            "gain_exact": 0.25,
            "gain_ratio": 0.25,
            "B": np.zeros((1, 1), dtype=float),
            "C": np.zeros((1, 1), dtype=float),
            "q": np.zeros(1, dtype=float),
            "w": np.zeros(1, dtype=float),
            "backend_info": {"noise_mode": "shots", "strict_test": True},
        }

    monkeypatch.setattr(
        controller,
        "_oracle_measured_candidate_incremental_block",
        _measured_candidate,
    )

    def _sentinel(*args, **kwargs):
        raise AssertionError("strict route touched exact decision helper")

    monkeypatch.setattr(controller_mod, "make_checkpoint_context", _sentinel)
    monkeypatch.setattr(controller, "_baseline_geometry", _sentinel)
    monkeypatch.setattr(controller, "_confirm_candidates", _sentinel)
    monkeypatch.setattr(controller, "_select_action_exact_v1", _sentinel)
    monkeypatch.setattr(controller, "_exact_step_forecast", _sentinel)
    monkeypatch.setattr(controller, "_integrate_theta_one_step", _sentinel)
    monkeypatch.setattr(controller, "_integrator_stage_baseline", _sentinel)
    monkeypatch.setattr(
        type(controller.current_executor),
        "prepare_state",
        lambda self, *args, **kwargs: _sentinel(*args, **kwargs),
    )

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["strict_fail_closed"] is False
    assert result.summary["qpu_faithful_decisions_passed"] is True
    first_confirmed = result.ledger[0]["confirmed"][0]
    assert first_confirmed["gain_exact"] == pytest.approx(0.25)
    assert first_confirmed["gain_ratio"] == pytest.approx(0.25)
    assert first_confirmed["confirm_error"] is None


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda summary, reference, row: summary.update({"exact_decision_checkpoints": 1}),
            "summary.exact_decision_checkpoints=1",
        ),
        (
            lambda summary, reference, row: row.update({"decision_backend": "exact"}),
            "row[0].decision_backend=exact",
        ),
        (
            lambda summary, reference, row: summary.update({"reference_enabled": True}),
            "summary.reference_enabled=true",
        ),
        (
            lambda summary, reference, row: reference.update({"reference_enabled": True}),
            "reference.reference_enabled=true",
        ),
        (
            lambda summary, reference, row: summary.update({"reference_mode": "benchmark_exact"}),
            "summary.reference_mode=benchmark_exact",
        ),
        (
            lambda summary, reference, row: reference.update({"reference_mode": "benchmark_exact"}),
            "reference.reference_mode=benchmark_exact",
        ),
        (
            lambda summary, reference, row: summary.update(
                {"controller_exact_input_mode": "benchmark_exact"}
            ),
            "summary.controller_exact_input_mode=benchmark_exact",
        ),
        (
            lambda summary, reference, row: row.update(
                {"uses_reference_for_decision": True}
            ),
            "row[0].uses_reference_for_decision=true",
        ),
        (
            lambda summary, reference, row: row.update(
                {"uses_future_exact_forecast_for_decision": True}
            ),
            "row[0].uses_future_exact_forecast_for_decision=true",
        ),
        (
            lambda summary, reference, row: row.update(
                {"decision_data_flow": "exact_assisted_controller"}
            ),
            "row[0].decision_data_flow=exact_assisted_controller",
        ),
        (
            lambda summary, reference, row: row.update({"exact_forecast_error": "used"}),
            "row[0].exact_forecast_error=present",
        ),
        (
            lambda summary, reference, row: row.update(
                {"decision_override_reason": "exact_forecast_dual_metric_regression"}
            ),
            "row[0].decision_override_reason=exact_forecast_dual_metric_regression",
        ),
        (
            lambda summary, reference, row: summary.update({"exact_audit_helper_active": True}),
            "summary.exact_audit_helper_active=active",
        ),
        (
            lambda summary, reference, row: row.update(
                {
                    "repair_no_admit_diagnostics": {
                        "forecast_veto_reason": "exact_forecast_nonimproving_tracking_score"
                    }
                }
            ),
            "row[0].repair_no_admit_diagnostics.forecast_veto_reason=exact_forecast_nonimproving_tracking_score",
        ),
    ],
)
def test_strict_qpu_contract_rejects_exact_decision_leaks(mutate, expected: str) -> None:
    summary = {
        "reference_mode": "off",
        "reference_enabled": False,
        "exact_decision_checkpoints": 0,
        "oracle_decision_checkpoints": 1,
    }
    reference = {"reference_mode": "off", "reference_enabled": False, "kind": None}
    row = {
        "decision_backend": "oracle",
        "decision_noise_mode": "ideal",
        "oracle_decision_used": True,
    }

    mutate(summary, reference, row)
    report = strict_qpu_faithful_decision_contract(
        summary=summary,
        reference=reference,
        decision_rows=[row],
    )

    assert report["passed"] is False
    assert expected in report["violations"]


def test_strict_qpu_contract_allows_post_run_exact_observable_diagnostics() -> None:
    report = strict_qpu_faithful_decision_contract(
        summary={
            "reference_mode": "off",
            "reference_enabled": False,
            "exact_decision_checkpoints": 0,
            "oracle_decision_checkpoints": 1,
            # Post-run diagnostics/report overlays may record exact quality fields.
            "final_abs_energy_total_error": 0.01,
            "final_fidelity_exact": 0.99,
        },
        reference={"reference_mode": "off", "reference_enabled": False, "kind": None},
        decision_rows=[
            {
                "decision_backend": "oracle",
                "decision_noise_mode": "ideal",
                "oracle_decision_used": True,
                "energy_total_exact": 0.2,
                "abs_energy_total_error": 0.01,
                "fidelity_exact": 0.99,
                "site_occupations_exact": [1.0, 1.0],
            }
        ],
    )

    assert report["passed"] is True
    assert report["violations"] == []


def test_strict_qpu_contract_allows_inactive_exact_forecast_config_knobs() -> None:
    report = strict_qpu_faithful_decision_contract(
        summary={
            "reference_mode": "off",
            "reference_enabled": False,
            "exact_decision_checkpoints": 0,
            "oracle_decision_checkpoints": 1,
            "uses_future_exact_forecast_for_decision": False,
            "exact_forecast_guardrail_mode": "off",
            "exact_forecast_veto_count": 0,
            "exact_forecast_baseline_proposal_mode": "norm_locked_blend_v1",
            "exact_forecast_baseline_gain_scales": [1.0],
            "exact_forecast_tracking_horizon_steps": 1,
            "exact_forecast_tracking_horizon_weights": [1.0],
            "exact_forecast_primary_density_target_mode": "pair_difference",
            "exact_forecast_tracking_primary_density_error_weight": 1.0,
            "exact_forecast_density_slope_weight": 1.0,
        },
        reference={"reference_mode": "off", "reference_enabled": False, "kind": None},
        decision_rows=[
            {
                "decision_backend": "oracle",
                "decision_noise_mode": "ideal",
                "oracle_decision_used": True,
            }
        ],
    )

    assert report["passed"] is True
    assert report["violations"] == []


def test_strict_qpu_contract_allows_inactive_exact_forecast_veto_counter() -> None:
    report = strict_qpu_faithful_decision_contract(
        summary={
            "reference_mode": "off",
            "reference_enabled": False,
            "exact_decision_checkpoints": 0,
            "oracle_decision_checkpoints": 1,
            "uses_future_exact_forecast_for_decision": False,
            "exact_forecast_guardrail_mode": "off",
            # Legacy summary name: in strict observable routes this counts
            # local forecast vetoes and must not by itself imply exact-assisted
            # decisioning.
            "exact_forecast_veto_count": 7,
            "exact_forecast_baseline_proposal_mode": "norm_locked_blend_v1",
        },
        reference={"reference_mode": "off", "reference_enabled": False, "kind": None},
        decision_rows=[
            {
                "decision_backend": "oracle",
                "decision_noise_mode": "ideal",
                "oracle_decision_used": True,
                "uses_future_exact_forecast_for_decision": False,
            }
        ],
    )

    assert report["passed"] is True
    assert report["violations"] == []


def test_strict_qpu_contract_rejects_active_exact_forecast_summary_data() -> None:
    report = strict_qpu_faithful_decision_contract(
        summary={
            "reference_mode": "off",
            "reference_enabled": False,
            "exact_decision_checkpoints": 0,
            "oracle_decision_checkpoints": 1,
            "uses_future_exact_forecast_for_decision": False,
            "exact_forecast_error": 0.1,
        },
        reference={"reference_mode": "off", "reference_enabled": False, "kind": None},
        decision_rows=[
            {
                "decision_backend": "oracle",
                "decision_noise_mode": "ideal",
                "oracle_decision_used": True,
            }
        ],
    )

    assert report["passed"] is False
    assert "summary.exact_forecast_error=present" in report["violations"]


def test_strict_qpu_hh_summary_contract_violation_fails_closed() -> None:
    controller = _strict_controller()
    controller._ledger = [
        {
            "checkpoint_index": 0,
            "time": 0.0,
            "physical_time": 0.0,
            "decision_backend": "exact",
            "decision_noise_mode": "ideal",
            "action_kind": "stay",
        }
    ]

    summary = controller._strict_qpu_hh_summary(
        strict_fail_closed=False,
        strict_fail_closed_reason=None,
        early_stop_reason=None,
        status="completed",
    )

    assert summary["status"] == "strict_fail_closed"
    assert summary["strict_fail_closed"] is True
    assert summary["qpu_faithful_decisions_passed"] is False
    assert summary["strict_decision_contract_passed"] is False
    assert summary["exact_decision_checkpoints"] == 1
    assert "row[0].decision_backend=exact" in summary["strict_decision_contract_violations"]
    assert "strict_decision_contract_violation" in summary["strict_fail_closed_reason"]


def test_strict_qpu_hh_measured_baseline_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _strict_controller(miss_threshold=2.0)

    def _raise_baseline(**kwargs):
        raise RuntimeError("shot budget unavailable")

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _raise_baseline)

    result = controller.run()

    assert result.summary["status"] == "strict_fail_closed"
    assert result.summary["strict_fail_closed"] is True
    assert "shot budget unavailable" in result.summary["strict_fail_closed_reason"]
    assert result.summary["qpu_faithful_decisions_expected"] is True
    assert result.summary["qpu_faithful_decisions_passed"] is False
    assert result.summary["exact_decision_checkpoints"] == 0
    assert result.ledger == []


def test_realtime_controller_baseline_geometry_persists_real_and_numeric_miss() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
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

    baseline = _baseline_geometry_payload(controller)
    summary = baseline["summary"]
    expected_rho_real = float(summary.epsilon_step_sq / max(summary.variance, 1.0e-14))
    expected_rho_num = float(max(0.0, expected_rho_real - float(summary.rho_miss)))

    assert float(summary.rho_real) == pytest.approx(expected_rho_real)
    assert float(summary.rho_num) == pytest.approx(expected_rho_num)
    assert float(baseline["rho_real"]) == pytest.approx(expected_rho_real)
    assert float(baseline["rho_num"]) == pytest.approx(expected_rho_num)


def test_realtime_controller_high_miss_active_uses_abs_threshold_and_persistence() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.2,
            miss_abs_threshold=0.5,
            miss_persistence_window=2,
            miss_persistence_count=2,
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
    low_abs = {"summary": SimpleNamespace(rho_miss=0.4, epsilon_proj_sq=0.1)}
    high_abs = {"summary": SimpleNamespace(rho_miss=0.4, epsilon_proj_sq=0.8)}

    assert controller._high_miss_active(baseline=low_abs) is False
    assert controller._high_miss_active(baseline=high_abs) is True
    controller._record_high_miss_history(baseline=high_abs)
    assert controller._high_miss_active(baseline=high_abs) is True
    assert controller._high_miss_active(baseline=low_abs) is True
    controller._high_miss_history = [False, False]
    controller._high_miss_relative_history = [True, True]
    assert controller._high_miss_active(baseline=low_abs) is True


def test_realtime_controller_euler_integrator_helper_matches_legacy_update() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="euler",
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
    baseline = _baseline_geometry_payload(controller)
    theta0 = np.asarray(controller.current_theta, dtype=float).copy()

    theta_next, theta_dot, diagnostics = controller._integrate_theta_one_step(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=theta0,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
    )

    np.testing.assert_allclose(
        theta_next,
        theta0 + 0.1 * np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1),
    )
    np.testing.assert_allclose(theta_dot, np.asarray(baseline["theta_dot_step"], dtype=float))
    assert diagnostics["integrator_policy"] == "euler"
    assert diagnostics["integrator_used"] == "euler"
    assert diagnostics["integrator_euler_fs_error"] is None


def test_realtime_controller_zero_dt_integrator_reports_no_advance() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="rk4",
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
    baseline = _baseline_geometry_payload(controller)

    theta_next, _, diagnostics = controller._integrate_theta_one_step(
        checkpoint_index=0,
        time_start=0.1,
        time_stop=None,
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
    )

    np.testing.assert_allclose(theta_next, controller.current_theta)
    assert diagnostics["integrator_policy"] == "rk4"
    assert diagnostics["integrator_used"] == "none"
    assert diagnostics["integrator_euler_fs_error"] is None


def test_realtime_controller_auto_integrator_can_select_rk4_from_condition_gate() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            integrator_condition_max=0.0,
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
    baseline = _baseline_geometry_payload(controller)

    theta_next, theta_dot, diagnostics = controller._integrate_theta_one_step(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
    )

    assert theta_next.shape == controller.current_theta.shape
    assert theta_dot.shape == controller.current_theta.shape
    assert diagnostics["integrator_policy"] == "auto_euler_rk4"
    assert diagnostics["integrator_used"] == "rk4"
    assert diagnostics["integrator_condition_pass"] is False
    assert float(diagnostics["integrator_euler_fs_error"]) >= 0.0


def test_realtime_controller_auto_integrator_blocks_early_euler_by_time_gate() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            miss_threshold=2.0,
            integrator_columnarity_threshold=0.0,
            integrator_curvature_threshold=10.0,
            integrator_euler_fs_error_threshold=10.0,
            integrator_euler_min_time_fraction=0.5,
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
    baseline = _baseline_geometry_payload(controller)

    _, _, diagnostics = controller._integrate_theta_one_step(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
    )

    assert diagnostics["integrator_policy"] == "auto_euler_rk4"
    assert diagnostics["integrator_used"] == "rk4"
    assert diagnostics["integrator_euler_time_gate_pass"] is False
    assert diagnostics["integrator_time_fraction"] == pytest.approx(0.0)
    assert diagnostics["integrator_euler_min_time_fraction"] == pytest.approx(0.5)


def test_realtime_controller_default_euler_observable_gate_is_disabled_without_rows() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
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

    gate = controller._integrator_euler_observable_gate()

    assert gate["integrator_euler_observable_gate_pass"] is True
    assert gate["integrator_euler_site_span"] is None
    assert gate["integrator_euler_primary_density_span"] is None



def test_realtime_controller_euler_observable_gate_fails_closed_without_rows() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            integrator_euler_site_span_max=1.0e-3,
            integrator_euler_primary_density_span_max=1.0e-3,
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

    gate = controller._integrator_euler_observable_gate()

    assert gate["integrator_euler_observable_gate_pass"] is False
    assert gate["integrator_euler_site_span"] is None
    assert gate["integrator_euler_primary_density_span"] is None


def test_realtime_controller_euler_observable_gate_ignores_exact_calm_fields() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            integrator_euler_observable_window=2,
            integrator_euler_site_span_max=1.0e-3,
            integrator_euler_primary_density_span_max=1.0e-3,
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
    controller._trajectory = [
        {
            "trajectory_sample_kind": "state_sample",
            "advances_time": True,
            "site_occupations": [0.50, 0.50],
            "site_occupations_exact": [0.50, 0.50],
            "primary_density": 0.0,
            "primary_density_exact": 0.0,
        },
        {
            "trajectory_sample_kind": "state_sample",
            "advances_time": True,
            "site_occupations": [0.55, 0.45],
            "site_occupations_exact": [0.50, 0.50],
            "primary_density": 0.10,
            "primary_density_exact": 0.0,
        },
    ]

    gate = controller._integrator_euler_observable_gate()

    assert gate["integrator_euler_observable_gate_pass"] is False
    assert gate["integrator_euler_site_span"] == pytest.approx(0.05)
    assert gate["integrator_euler_primary_density_span"] == pytest.approx(0.10)



def test_realtime_controller_auto_integrator_blocks_euler_when_geometry_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            miss_threshold=2.0,
            integrator_columnarity_threshold=0.99,
            integrator_curvature_threshold=0.01,
            integrator_euler_fs_error_threshold=1.0e-9,
            integrator_euler_min_time_fraction=0.0,
            append_no_harm_guard_enabled=False,
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
    baseline = _baseline_geometry_payload(controller)
    theta_dot = np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1)

    def _same_as_euler(**kwargs):
        theta0 = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        dt = float(kwargs["time_stop"]) - float(kwargs["time_start"])
        return theta0 + dt * theta_dot, theta_dot

    monkeypatch.setattr(controller, "_integrator_vector_diagnostics", lambda _theta_dot: (0.0, 10.0))
    monkeypatch.setattr(controller, "_rk4_integrate_theta_one_step", _same_as_euler)

    _, _, diagnostics = controller._integrate_theta_one_step(
        checkpoint_index=1,
        time_start=0.1,
        time_stop=0.2,
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
    )

    assert diagnostics["integrator_geometry_gate_pass"] is False
    assert diagnostics["integrator_euler_error_pass"] is True
    assert diagnostics["integrator_euler_observable_gate_pass"] is True
    assert diagnostics["integrator_auto_admit_euler"] is False
    assert diagnostics["integrator_euler_blockers"] == ["geometry"]
    assert diagnostics["integrator_used"] == "rk4"


def test_realtime_controller_auto_integrator_selects_euler_when_all_gates_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            miss_threshold=2.0,
            integrator_columnarity_threshold=0.99,
            integrator_curvature_threshold=0.01,
            integrator_euler_fs_error_threshold=1.0e-9,
            integrator_euler_min_time_fraction=0.0,
            append_no_harm_guard_enabled=False,
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
    baseline = _baseline_geometry_payload(controller)
    theta_dot = np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1)

    def _same_as_euler(**kwargs):
        theta0 = np.asarray(kwargs["theta_runtime"], dtype=float).reshape(-1)
        dt = float(kwargs["time_stop"]) - float(kwargs["time_start"])
        return theta0 + dt * theta_dot, theta_dot

    monkeypatch.setattr(controller, "_integrator_vector_diagnostics", lambda _theta_dot: (1.0, 0.0))
    monkeypatch.setattr(controller, "_rk4_integrate_theta_one_step", _same_as_euler)

    _, _, diagnostics = controller._integrate_theta_one_step(
        checkpoint_index=1,
        time_start=0.1,
        time_stop=0.2,
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
    )

    assert diagnostics["integrator_geometry_gate_pass"] is True
    assert diagnostics["integrator_euler_error_pass"] is True
    assert diagnostics["integrator_auto_admit_euler"] is True
    assert diagnostics["integrator_euler_blockers"] == []
    assert diagnostics["integrator_used"] == "euler"


def test_strict_qpu_auto_integrator_blocks_euler_when_geometry_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, _hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            miss_threshold=2.0,
            integrator_columnarity_threshold=0.99,
            integrator_curvature_threshold=0.01,
            integrator_euler_fs_error_threshold=1.0e-9,
            integrator_euler_min_time_fraction=0.0,
            append_no_harm_guard_enabled=False,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=_strict_oracle_config(noise_mode="ideal"),
        strict_qpu_faithful=True,
    )
    baseline = {
        "theta_dot_step": np.asarray([0.1], dtype=float),
        "summary": SimpleNamespace(condition_number=1.0, rho_miss=0.0),
        "G": np.eye(1),
    }
    theta_dot = np.asarray(baseline["theta_dot_step"], dtype=float)

    monkeypatch.setattr(controller, "_integrator_vector_diagnostics", lambda _theta_dot: (0.0, 10.0))
    monkeypatch.setattr(
        controller,
        "_strict_qpu_hh_measured_integrator_stage_baseline",
        lambda **_kwargs: {"theta_dot_step": theta_dot},
    )

    _, _, diagnostics = controller._strict_qpu_hh_integrate_theta_one_step(
        checkpoint_index=1,
        time_start=0.1,
        time_stop=0.2,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
        tier_name="confirm",
        budget_scale=1.0,
    )

    assert diagnostics["integrator_geometry_gate_pass"] is False
    assert diagnostics["integrator_euler_error_pass"] is True
    assert diagnostics["integrator_auto_admit_euler"] is False
    assert diagnostics["integrator_euler_blockers"] == ["geometry"]
    assert diagnostics["integrator_used"] == "rk4"


def test_strict_qpu_auto_integrator_selects_euler_when_all_gates_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, _hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            miss_threshold=2.0,
            integrator_columnarity_threshold=0.99,
            integrator_curvature_threshold=0.01,
            integrator_euler_fs_error_threshold=1.0e-9,
            integrator_euler_min_time_fraction=0.0,
            append_no_harm_guard_enabled=False,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=None,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        oracle_base_config=_strict_oracle_config(noise_mode="ideal"),
        strict_qpu_faithful=True,
    )
    baseline = {
        "theta_dot_step": np.asarray([0.1], dtype=float),
        "summary": SimpleNamespace(condition_number=1.0, rho_miss=0.0),
        "G": np.eye(1),
    }
    theta_dot = np.asarray(baseline["theta_dot_step"], dtype=float)

    monkeypatch.setattr(controller, "_integrator_vector_diagnostics", lambda _theta_dot: (1.0, 0.0))
    monkeypatch.setattr(
        controller,
        "_strict_qpu_hh_measured_integrator_stage_baseline",
        lambda **_kwargs: {"theta_dot_step": theta_dot},
    )

    _, _, diagnostics = controller._strict_qpu_hh_integrate_theta_one_step(
        checkpoint_index=1,
        time_start=0.1,
        time_stop=0.2,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
        tier_name="confirm",
        budget_scale=1.0,
    )

    assert diagnostics["integrator_geometry_gate_pass"] is True
    assert diagnostics["integrator_euler_error_pass"] is True
    assert diagnostics["integrator_auto_admit_euler"] is True
    assert diagnostics["integrator_euler_blockers"] == []
    assert diagnostics["integrator_used"] == "euler"


def test_realtime_controller_auto_integrator_blocks_euler_until_observables_are_calm() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
            miss_threshold=2.0,
            integrator_columnarity_threshold=0.0,
            integrator_curvature_threshold=10.0,
            integrator_euler_fs_error_threshold=10.0,
            integrator_euler_min_time_fraction=0.0,
            integrator_euler_observable_window=2,
            integrator_euler_site_span_max=1.0e-3,
            integrator_euler_primary_density_span_max=1.0e-3,
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
    controller._trajectory = [
        {
            "trajectory_sample_kind": "state_sample",
            "advances_time": True,
            "site_occupations": [0.50, 0.50],
            "site_occupations_exact": [0.50, 0.50],
            "primary_density": 0.0,
            "primary_density_exact": 0.0,
            "energy_total": -1.0,
        },
        {
            "trajectory_sample_kind": "state_sample",
            "advances_time": True,
            "site_occupations": [0.55, 0.45],
            "site_occupations_exact": [0.50, 0.50],
            "primary_density": 0.10,
            "primary_density_exact": 0.0,
            "energy_total": -1.0,
        },
    ]
    baseline = _baseline_geometry_payload(controller)

    _, _, diagnostics = controller._integrate_theta_one_step(
        checkpoint_index=1,
        time_start=0.1,
        time_stop=0.2,
        executor=controller.current_executor,
        layout=controller.current_layout,
        theta_runtime=controller.current_theta,
        baseline=baseline,
        planning_audit=controller._planning_audit,
        scaffold_labels=controller._current_scaffold_labels(),
    )

    assert diagnostics["integrator_policy"] == "auto_euler_rk4"
    assert diagnostics["integrator_euler_time_gate_pass"] is True
    assert diagnostics["integrator_euler_observable_gate_pass"] is False
    assert diagnostics["integrator_euler_site_span"] == pytest.approx(0.05)
    assert diagnostics["integrator_euler_primary_density_span"] == pytest.approx(0.10)
    assert diagnostics["integrator_used"] == "rk4"



def test_realtime_controller_auto_integrator_does_not_silently_fallback_to_euler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            integrator_policy="auto_euler_rk4",
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
    baseline = _baseline_geometry_payload(controller)

    def _raise_rk4(**kwargs):
        raise RuntimeError("rk4 unavailable")

    monkeypatch.setattr(controller, "_rk4_integrate_theta_one_step", _raise_rk4)

    with pytest.raises(RuntimeError, match="rk4 unavailable"):
        controller._integrate_theta_one_step(
            checkpoint_index=0,
            time_start=0.0,
            time_stop=0.1,
            executor=controller.current_executor,
            layout=controller.current_layout,
            theta_runtime=controller.current_theta,
            baseline=baseline,
            planning_audit=controller._planning_audit,
            scaffold_labels=controller._current_scaffold_labels(),
        )


def test_realtime_controller_persists_integrator_diagnostics_in_rows() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=2.0,
            integrator_policy="rk4",
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )

    result = controller.run()

    assert result.trajectory[0]["integrator_policy"] == "rk4"
    assert result.trajectory[0]["integrator_used"] == "rk4"
    assert result.trajectory[0]["integrator_condition_pass"] is True
    assert result.trajectory[0]["integrator_euler_fs_error"] is not None
    assert result.ledger[0]["integrator_policy"] == "rk4"
    assert result.ledger[0]["integrator_used"] == "rk4"
    assert result.summary["integrator_policy"] == "rk4"
    assert result.summary["integrator_rk4_count"] >= 1


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


def _multi_block_context(
    *,
    block_count: int = 5,
) -> tuple[ReplayScaffoldContext, np.ndarray, np.ndarray, np.ndarray]:
    paulis = ("x", "y", "z", "x", "y", "z")
    terms = tuple(
        AnsatzTerm(
            label=f"op_{idx}_{paulis[int(idx) % len(paulis)]}",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(1, ps=paulis[int(idx) % len(paulis)], pc=1.0)],
            ),
        )
        for idx in range(int(block_count))
    )
    candidate = AnsatzTerm(
        label="op_candidate",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    hmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    psi_ref = _basis(0)
    base_layout = build_parameter_layout(
        list(terms),
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        list(terms),
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    best_theta = np.linspace(0.02, 0.02 * int(block_count), int(block_count), dtype=float)
    psi_initial = executor.prepare_state(best_theta, psi_ref)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1, L=1, ordering="blocked"),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}},
        family_info={"resolved": "toy_multi_block"},
        family_pool=tuple([*terms, candidate]),
        pool_meta={"candidate_pool_complete": True},
        replay_terms=tuple(terms),
        base_layout=base_layout,
        adapt_theta_runtime=np.asarray(best_theta, dtype=float),
        adapt_theta_logical=np.asarray(best_theta, dtype=float),
        adapt_depth=int(block_count),
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=int(len(terms) + 1),
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


def test_realtime_controller_reuses_driven_exact_reference_cache_for_matching_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    cache: dict[str, object] = {}
    call_counter = {"count": 0}
    original = RealtimeExactAuditHelper._build_exact_reference_artifacts

    def _wrapped(self, metadata):
        call_counter["count"] += 1
        return original(self, metadata)

    monkeypatch.setattr(
        RealtimeExactAuditHelper,
        "_build_exact_reference_artifacts",
        _wrapped,
    )
    common = dict(
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        drive_config=_drive_cfg(),
        exact_reference_cache=cache,
    )
    controller_a = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        **common,
    )
    controller_b = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", miss_threshold=0.5),
        **common,
    )

    helper_a = build_exact_audit_helper_for_controller(
        controller_a,
        exact_reference_cache=cache,
    )
    helper_b = build_exact_audit_helper_for_controller(
        controller_b,
        exact_reference_cache=cache,
    )
    helper_a.ensure_ready()
    helper_b.ensure_ready()

    assert call_counter["count"] == 1
    assert len(cache) == 1
    assert helper_a._exact_reference_cache_key == helper_b._exact_reference_cache_key


def test_realtime_controller_exact_reference_cache_misses_when_time_grid_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    cache: dict[str, object] = {}
    call_counter = {"count": 0}
    original = RealtimeExactAuditHelper._build_exact_reference_artifacts

    def _wrapped(self, metadata):
        call_counter["count"] += 1
        return original(self, metadata)

    monkeypatch.setattr(
        RealtimeExactAuditHelper,
        "_build_exact_reference_artifacts",
        _wrapped,
    )
    common = dict(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.2,
        drive_config=_drive_cfg(),
        exact_reference_cache=cache,
    )
    controller_a = RealtimeCheckpointController(num_times=3, **common)
    controller_b = RealtimeCheckpointController(num_times=4, **common)
    helper_a = build_exact_audit_helper_for_controller(
        controller_a,
        exact_reference_cache=cache,
    )
    helper_b = build_exact_audit_helper_for_controller(
        controller_b,
        exact_reference_cache=cache,
    )
    helper_a.ensure_ready()
    helper_b.ensure_ready()

    assert call_counter["count"] == 2
    assert len(cache) == 2


def test_realtime_controller_reuses_static_exact_reference_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    cache: dict[str, object] = {}
    call_counter = {"count": 0}
    original = RealtimeExactAuditHelper._build_exact_reference_artifacts

    def _wrapped(self, metadata):
        call_counter["count"] += 1
        return original(self, metadata)

    monkeypatch.setattr(
        RealtimeExactAuditHelper,
        "_build_exact_reference_artifacts",
        _wrapped,
    )
    common = dict(
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
        exact_reference_cache=cache,
    )
    helper_a = build_exact_audit_helper_for_controller(
        RealtimeCheckpointController(cfg=RealtimeCheckpointConfig(mode="exact_v1"), **common),
        exact_reference_cache=cache,
    )
    helper_b = build_exact_audit_helper_for_controller(
        RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(mode="exact_v1", gain_ratio_threshold=0.5),
            **common,
        ),
        exact_reference_cache=cache,
    )
    helper_a.ensure_ready()
    helper_b.ensure_ready()

    assert call_counter["count"] == 1
    assert len(cache) == 1


def _exhausted_repeat_label_context() -> tuple[ReplayScaffoldContext, np.ndarray, np.ndarray, np.ndarray]:
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
    base_layout = build_parameter_layout([x_term, dup_y], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    executor = CompiledAnsatzExecutor(
        [x_term, dup_y],
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    best_theta = np.array([0.2, 0.1], dtype=float)
    psi_initial = executor.prepare_state(best_theta, psi_ref)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1, L=1, ordering="blocked"),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}},
        family_info={"resolved": "toy_pool_dup_exhausted"},
        family_pool=(x_term, dup_y, dup_z),
        pool_meta={"candidate_pool_complete": True},
        replay_terms=(x_term, dup_y),
        base_layout=base_layout,
        adapt_theta_runtime=np.array([0.2, 0.1], dtype=float),
        adapt_theta_logical=np.array([0.2, 0.1], dtype=float),
        adapt_depth=2,
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
            append_no_harm_guard_enabled=False,
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


def test_candidate_insert_theta_block_preserves_old_theta_and_state() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
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
    old_theta = np.asarray(controller.current_theta, dtype=float).reshape(-1).copy()
    old_state = controller.current_executor.prepare_state(old_theta, replay_context.psi_ref)
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=old_theta,
        psi=old_state,
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

    candidate_data = controller._candidate_executor_data(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        candidate_term=replay_context.family_pool[1],
        candidate_pool_index=1,
        position_id=1,
    )

    theta_aug = np.asarray(candidate_data["theta_aug"], dtype=float).reshape(-1)
    runtime_pos = int(candidate_data["runtime_insert_position"])
    width = int(len(candidate_data["runtime_block_indices"]))
    assert width > 0
    assert theta_aug[runtime_pos : runtime_pos + width] == pytest.approx(np.zeros(width))
    theta_without_insert = np.concatenate(
        [theta_aug[:runtime_pos], theta_aug[runtime_pos + width :]]
    )
    assert theta_without_insert == pytest.approx(old_theta)

    aug_state = candidate_data["aug_executor"].prepare_state(theta_aug, replay_context.psi_ref)
    assert np.allclose(aug_state, old_state, atol=1.0e-12)
    assert np.allclose(candidate_data["aug_psi"], old_state, atol=1.0e-12)


def test_append_commit_with_damped_candidate_step_obeys_auto_rk4_euler_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            integrator_policy="auto_euler_rk4",
            candidate_step_scales=(0.15,),
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            append_no_harm_guard_enabled=False,
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
    monkeypatch.setattr(controller, "_exact_v1_forecast_override_reason", lambda **kwargs: None)

    result = controller.run()

    assert int(result.summary["append_count"]) >= 1
    assert str(result.trajectory[0]["action_kind"]) == "append_candidate"
    assert result.trajectory[0]["selected_step_scale"] == pytest.approx(0.15)
    assert str(result.trajectory[0]["integrator_policy"]) == "auto_euler_rk4"
    assert str(result.trajectory[0]["integrator_used"]) == "rk4"
    assert result.trajectory[0]["integrator_euler_time_gate_pass"] is False
    assert result.trajectory[0]["integrator_euler_observable_gate_pass"] is True
    assert str(result.ledger[0]["integrator_used"]) == "rk4"


def test_append_commit_with_damped_candidate_step_does_not_override_explicit_rk4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            integrator_policy="rk4",
            candidate_step_scales=(0.15,),
            miss_threshold=0.0,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            append_no_harm_guard_enabled=False,
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
    monkeypatch.setattr(controller, "_exact_v1_forecast_override_reason", lambda **kwargs: None)

    result = controller.run()

    assert int(result.summary["append_count"]) >= 1
    assert str(result.trajectory[0]["action_kind"]) == "append_candidate"
    assert result.trajectory[0]["selected_step_scale"] == pytest.approx(0.15)
    assert str(result.trajectory[0]["integrator_policy"]) == "rk4"
    assert str(result.trajectory[0]["integrator_used"]) == "rk4"
    assert str(result.ledger[0]["integrator_used"]) == "rk4"


def test_realtime_controller_run_persists_postcross_compare_diag(
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
            exact_v1_postcross_compare_diag=True,
            exact_forecast_density_postcross_wrong_sign_weight=1.0,
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

    diag_rows = [
        row["exact_v1_postcross_compare_diag"]
        for row in result.trajectory
        if row.get("exact_v1_postcross_compare_diag") is not None
    ]
    assert diag_rows
    diag = diag_rows[0]
    assert diag is not None
    assert float(diag["weight"]) == pytest.approx(1.0)
    assert "stay" in diag
    assert "selected_pre_override" in diag
    assert "runner_up_compare" in diag
    payloads = [
        branch
        for branch in (diag.get("stay"), diag.get("selected_pre_override"), diag.get("runner_up_compare"))
        if branch is not None
    ]
    assert payloads
    assert any("site_turn" in branch for branch in payloads)


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


def test_exact_v1_select_action_rejects_candidate_below_gain_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
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

    monkeypatch.setattr(
        controller,
        "_select_exact_v1_candidate_step_scale",
        lambda **kwargs: pytest.fail("live gain gate should reject before forecasting"),
    )

    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _record(
                "candidate_a",
                confirm_score=2.0,
                gain_ratio=0.015,
                gain_exact=1.0e-6,
                pool_index=0,
            )
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast={"local_projective_score_total": 1.0},
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "gain_ratio_below_threshold"


def test_exact_v1_componentwise_aspiration_allows_site_win_vs_stay() -> None:
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

    ok, reason = controller._exact_v1_componentwise_aspiration_result(
        stay_forecast={
            "fidelity_exact_next": 0.96,
            "normalized_primary_density_error_next": 0.30,
            "abs_primary_density_error_next": 0.30,
            "primary_density_slope_error_next": 0.20,
            "abs_primary_density_slope_error_next": 0.20,
            "normalized_energy_total_error_next": 0.10,
            "abs_energy_total_error_next": 0.10,
            "site_occupations_abs_error_max_next": 0.10,
        },
        selected_forecast={
            "fidelity_exact_next": 0.96,
            "normalized_primary_density_error_next": 0.30,
            "abs_primary_density_error_next": 0.30,
            "primary_density_slope_error_next": 0.20,
            "abs_primary_density_slope_error_next": 0.20,
            "normalized_energy_total_error_next": 0.12,
            "abs_energy_total_error_next": 0.12,
            "site_occupations_abs_error_max_next": 0.07,
        },
    )

    assert ok is True
    assert reason is None


def test_exact_v1_select_action_records_postcross_compare_diag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            exact_v1_postcross_compare_diag=True,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=1.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=1.0,
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

    def _record(label: str, *, confirm_score: float, pool_index: int) -> dict[str, object]:
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": int(pool_index),
            "position_id": int(pool_index),
            "adjusted_gain": float(confirm_score),
            "confirm_score": float(confirm_score),
            "gain_exact": 1.0,
            "gain_ratio": 1.0,
            "groups_new": 0.0,
            "candidate_summary": CandidateProbeSummary(
                candidate_label=label,
                candidate_pool_index=int(pool_index),
                position_id=int(pool_index),
                runtime_insert_position=0,
                runtime_block_indices=[],
                residual_overlap_l2=0.0,
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
            ),
        }

    stay_forecast = {
        "fidelity_exact_next": 1.0,
        "normalized_primary_density_error_next": 0.50,
        "abs_primary_density_error_next": 0.50,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.0,
        "abs_energy_total_error_next": 0.0,
        "tracking_primary_density_slope_abs_error_mean": 0.45,
        "tracking_primary_density_postcross_wrong_sign_active": 1.0,
        "tracking_primary_density_postcross_wrong_sign_error_mean": 0.30,
        "tracking_primary_density_postcross_wrong_sign_abs_error_mean": 0.30,
        "tracking_d_curvature_abs_error_mean": 0.20,
        "tracking_d_excursion_under_response_mean": 0.30,
        "tracking_d_excursion_over_response_mean": 0.08,
        "tracking_total_occupation_abs_error_next": 0.12,
        "tracking_total_occupation_abs_error_mean": 0.10,
        "tracking_site_slope_abs_error_mean_by_site": [0.40, 0.10],
        "tracking_site_curvature_abs_error_mean_by_site": [0.20, 0.05],
        "tracking_site_excursion_under_response_mean_by_site": [0.30, 0.06],
        "tracking_site_excursion_over_response_mean_by_site": [0.08, 0.01],
    }
    forecast_a = {
        "fidelity_exact_next": 1.0,
        "normalized_primary_density_error_next": 0.10,
        "abs_primary_density_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.0,
        "abs_energy_total_error_next": 0.0,
        "tracking_primary_density_slope_abs_error_mean": 0.25,
        "tracking_primary_density_postcross_wrong_sign_active": 1.0,
        "tracking_primary_density_postcross_wrong_sign_error_mean": 0.20,
        "tracking_primary_density_postcross_wrong_sign_abs_error_mean": 0.20,
        "tracking_d_curvature_abs_error_mean": 0.10,
        "tracking_d_excursion_under_response_mean": 0.18,
        "tracking_d_excursion_over_response_mean": 0.05,
        "tracking_total_occupation_abs_error_next": 0.06,
        "tracking_total_occupation_abs_error_mean": 0.04,
        "tracking_site_slope_abs_error_mean_by_site": [0.25, 0.08],
        "tracking_site_curvature_abs_error_mean_by_site": [0.10, 0.04],
        "tracking_site_excursion_under_response_mean_by_site": [0.18, 0.02],
        "tracking_site_excursion_over_response_mean_by_site": [0.05, 0.02],
    }
    forecast_b = {
        "fidelity_exact_next": 1.0,
        "normalized_primary_density_error_next": 0.31,
        "abs_primary_density_error_next": 0.31,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.0,
        "abs_energy_total_error_next": 0.0,
        "tracking_primary_density_slope_abs_error_mean": 0.50,
        "tracking_primary_density_postcross_wrong_sign_active": 0.0,
        "tracking_primary_density_postcross_wrong_sign_error_mean": 0.0,
        "tracking_primary_density_postcross_wrong_sign_abs_error_mean": 0.0,
        "tracking_d_curvature_abs_error_mean": 0.22,
        "tracking_d_excursion_under_response_mean": 0.35,
        "tracking_d_excursion_over_response_mean": 0.10,
        "tracking_total_occupation_abs_error_next": 0.20,
        "tracking_total_occupation_abs_error_mean": 0.18,
        "tracking_site_slope_abs_error_mean_by_site": [0.50, 0.12],
        "tracking_site_curvature_abs_error_mean_by_site": [0.22, 0.06],
        "tracking_site_excursion_under_response_mean_by_site": [0.35, 0.07],
        "tracking_site_excursion_over_response_mean_by_site": [0.10, 0.03],
    }

    monkeypatch.setattr(controller, "_passes_exact_confirm_thresholds", lambda record: True)

    def _fake_scale(**kwargs):
        selected = dict(kwargs["selected"])
        forecast = forecast_a if str(selected["candidate_label"]) == "candidate_a" else forecast_b
        return selected, dict(forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)

    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _record("candidate_a", confirm_score=2.0, pool_index=0),
            _record("candidate_b", confirm_score=1.0, pool_index=1),
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
    )

    diag = controller._last_exact_v1_postcross_compare_diag
    assert str(action_kind) == "append_candidate"
    assert selected is not None and str(selected["candidate_label"]) == "candidate_a"
    assert diag is not None
    assert float(diag["weight"]) == pytest.approx(1.0)
    assert int(diag["evaluated_count"]) == 2
    assert int(diag["admitted_count"]) == 2
    assert int(diag["postcross_active_evaluated_count"]) == 1
    assert int(diag["postcross_active_admitted_count"]) == 1
    assert int(diag["postcross_active_rejected_count"]) == 0
    assert bool(diag["stay"]["postcross_active"]) is True
    assert str(diag["selected_pre_override"]["candidate_label"]) == "candidate_a"
    assert str(diag["runner_up_compare"]["candidate_label"]) == "candidate_b"
    assert float(diag["selected_pre_override"]["postcross_contribution"]) == pytest.approx(0.20)
    assert float(diag["runner_up_compare"]["postcross_contribution"]) == pytest.approx(0.0)
    assert float(diag["stay"]["d_shape"]["shadow_only_total"]) == pytest.approx(0.58)
    assert float(diag["selected_pre_override"]["d_shape"]["shadow_only_total"]) == pytest.approx(
        0.33
    )
    assert float(
        diag["selected_pre_override"]["d_shape_delta_vs_stay"]["shadow_only_total"]
    ) == pytest.approx(-0.25)
    assert float(diag["selected_pre_override"]["d_shape"]["total_with_slope"]) == pytest.approx(0.58)
    assert float(diag["selected_pre_override"]["total_occupation"]["abs_error_next"]) == pytest.approx(0.06)
    assert float(
        diag["selected_pre_override"]["total_occupation_delta_vs_stay"]["abs_error_next"]
    ) == pytest.approx(-0.06)
    assert diag["stay"]["site_turn"]["slope_abs_error_mean_by_site"] == pytest.approx([0.40, 0.10])
    assert diag["selected_pre_override"]["site_turn"]["slope_abs_error_mean_by_site"] == pytest.approx(
        [0.25, 0.08]
    )
    assert diag["selected_pre_override"]["site_turn_delta_vs_stay"] is not None
    assert diag["selected_pre_override"]["site_turn_delta_vs_stay"][
        "slope_abs_error_mean_by_site"
    ] == pytest.approx([-0.15, -0.02])
    assert diag["runner_up_compare"]["site_turn_delta_vs_stay"] is not None
    assert diag["runner_up_compare"]["site_turn_delta_vs_stay"][
        "excursion_under_response_mean_by_site"
    ] == pytest.approx([0.05, 0.01])


def test_exact_v1_select_action_rejects_missing_confirm_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
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

    def _record(label: str, *, gain_ratio: float, gain_exact: float, pool_index: int) -> dict[str, object]:
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": int(pool_index),
            "position_id": int(pool_index),
            "adjusted_gain": 2.0,
            "confirm_score": None,
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

    monkeypatch.setattr(
        controller,
        "_select_exact_v1_candidate_step_scale",
        lambda **kwargs: pytest.fail("missing confirm score should reject before forecasting"),
    )

    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _record(
                "candidate_a",
                gain_ratio=0.05,
                gain_exact=2.0e-6,
                pool_index=0,
            )
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast={"local_projective_score_total": 1.0},
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "confirm_score_missing"


def test_exact_v1_no_harm_rejects_rho_gain_when_condition_worsens_beyond_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller(
        append_no_harm_condition_ratio_cap=2.0,
        append_no_harm_condition_abs_floor=1.0,
    )
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=0.50,
        condition=10.0,
        step_gain=0.10,
        displacement=0.05,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.05,
        condition=25.0,
        step_gain=0.20,
        displacement=0.05,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record()],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_calm_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_condition_worse"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(2.5)
    assert float(diag["rho_miss_delta_stay_minus_selected"]) == pytest.approx(0.45)


def test_exact_v1_no_harm_rejects_kink_motion_with_weak_nonrho_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller(
        append_no_harm_condition_ratio_cap=2.0,
        append_no_harm_kink_min_step_gain_delta=0.05,
        append_no_harm_kink_max_condition_ratio=2.0,
        append_no_harm_kink_max_displacement_ratio=2.0,
    )
    stay_forecast = _no_harm_forecast(
        score=5.0,
        rho_miss=0.40,
        condition=10.0,
        step_gain=0.20,
        displacement=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.05,
        condition=11.0,
        step_gain=0.20,
        displacement=0.10,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _no_harm_candidate_record(gain_ratio=0.01, gain_exact=1.0, confirm_score=2.0)
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_kink_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_motion_kink"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert bool(diag["motion_bad"]) is True
    assert bool(diag["stability_support"]) is False


def test_exact_v1_no_harm_rejects_kink_curvature_flip_despite_large_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller()
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=1.000,
        condition=100.0,
        step_gain=0.100,
        displacement=1.000,
        step_residual=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.010,
        condition=100.5,
        step_gain=1.100,
        displacement=1.242,
        step_residual=0.10,
    )

    def _fake_scale(**kwargs):
        selected = dict(kwargs["selected"])
        selected["candidate_step_scale"] = 1.0
        return selected, dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record("paop_full:paop_dbl(site=0)", gain_ratio=1.0)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_kink_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_condition_worse"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert str(diag["motion_regime"]) == "kink"
    assert bool(diag["motion_curvature_sign_flip"]) is True
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(1.005)
    assert float(diag["displacement_ratio_selected_vs_stay"]) == pytest.approx(1.242)
    assert bool(diag["stability_support"]) is False


def test_exact_v1_no_harm_rejects_kink_reversal_despite_damped_large_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller()
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=1.0000,
        condition=10000.0,
        step_gain=0.100,
        displacement=1.000,
        step_residual=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.0100,
        condition=10002.0,
        step_gain=1.100,
        displacement=1.101,
        step_residual=0.10,
    )

    def _fake_scale(**kwargs):
        selected = dict(kwargs["selected"])
        selected["candidate_step_scale"] = 0.15
        return selected, dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record("phonon_layer", gain_ratio=1.0)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_kink_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_condition_worse"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert str(diag["motion_regime"]) == "kink"
    assert bool(diag["motion_direction_reversal"]) is True
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(1.0002)
    assert float(diag["displacement_ratio_selected_vs_stay"]) == pytest.approx(1.101)
    assert bool(diag["stability_support"]) is False


def test_exact_v1_no_harm_rejects_rho_miss_only_with_bad_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller(
        append_no_harm_condition_ratio_cap=10.0,
        append_no_harm_rho_only_condition_ratio_cap=1.2,
        append_no_harm_rho_only_min_step_gain_delta=0.02,
    )
    stay_forecast = _no_harm_forecast(
        score=5.0,
        rho_miss=0.50,
        condition=10.0,
        step_gain=0.20,
        displacement=0.10,
        step_residual=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.05,
        condition=13.0,
        step_gain=0.20,
        displacement=0.10,
        step_residual=0.10,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record(gain_ratio=0.05)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_calm_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_rho_miss_only"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert float(diag["rho_miss_delta_stay_minus_selected"]) == pytest.approx(0.45)
    assert float(diag["step_gain_delta_selected_minus_stay"]) == pytest.approx(0.0)
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(1.3)


def test_exact_v1_no_harm_rejects_steady_strong_gain_with_large_condition_and_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller()
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=1.000,
        condition=10.0,
        step_gain=0.100,
        displacement=0.100,
        step_residual=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.124,
        condition=27.17,
        step_gain=0.976,
        displacement=0.343,
        step_residual=0.10,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record(gain_ratio=0.90)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_steady_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_condition_worse"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert str(diag["motion_regime"]) == "steady"
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(2.717)
    assert float(diag["displacement_ratio_selected_vs_stay"]) == pytest.approx(3.43)
    assert float(diag["rho_miss_delta_stay_minus_selected"]) == pytest.approx(0.876)
    assert float(diag["step_gain_delta_selected_minus_stay"]) == pytest.approx(0.876)


def test_exact_v1_no_harm_rejects_steady_strong_gain_with_large_displacement_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller()
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=1.000,
        condition=10.0,
        step_gain=0.100,
        displacement=0.100,
        step_residual=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.124,
        condition=10.0,
        step_gain=0.976,
        displacement=0.343,
        step_residual=0.10,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record(gain_ratio=0.90)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_steady_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_displacement_worse"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(1.0)
    assert float(diag["displacement_ratio_selected_vs_stay"]) == pytest.approx(3.43)


def test_exact_v1_no_harm_rejects_checkpoint40_steady_tiny_condition_and_large_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller()
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=1.000,
        condition=10000.0,
        step_gain=0.100,
        displacement=1.000,
        step_residual=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.010,
        condition=10007.7,
        step_gain=1.100,
        displacement=1.234,
        step_residual=0.10,
    )

    def _fake_scale(**kwargs):
        selected = dict(kwargs["selected"])
        selected["candidate_step_scale"] = 0.15
        return selected, dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record("paop_full:paop_dbl(site=0)", gain_ratio=1.0)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_steady_motion(),
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "no_harm_condition_worse"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert str(diag["motion_regime"]) == "steady"
    assert bool(diag["motion_bad"]) is False
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(1.00077)
    assert float(diag["displacement_ratio_selected_vs_stay"]) == pytest.approx(1.234)
    assert bool(diag["stability_support"]) is False


def test_exact_v1_no_harm_allows_stable_motion_with_meaningful_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller()
    stay_forecast = _no_harm_forecast(
        score=5.0,
        rho_miss=0.30,
        condition=10.0,
        step_gain=0.10,
        displacement=0.10,
        step_residual=0.10,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.10,
        condition=10.0,
        step_gain=0.13,
        displacement=0.10,
        step_residual=0.10,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record(gain_ratio=0.20)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_calm_motion(),
    )

    assert str(action_kind) == "append_candidate"
    assert selected is not None
    assert str(controller._last_exact_v1_selection_reason) == "live_local_gates_passed"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert diag["veto_reason"] is None
    assert float(diag["condition_ratio_selected_vs_stay"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "family_key",
    ["hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard"],
)
def test_exact_v1_no_harm_allows_spinful_lattice_zero_motion_projective_support(
    monkeypatch: pytest.MonkeyPatch,
    family_key: str,
) -> None:
    controller = _no_harm_controller()
    controller._family_key = str(family_key)
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=1.0,
        condition=5.0e7,
        step_gain=0.0,
        displacement=0.0,
        step_residual=1.0,
    )
    selected_forecast = _no_harm_forecast(
        score=-100.0,
        rho_miss=0.006,
        condition=5.0e7 * 1.10,
        step_gain=0.994,
        displacement=0.033,
        step_residual=0.006,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _no_harm_candidate_record(
                "ham_term(zeze)",
                gain_ratio=0.996,
                gain_exact=1.0,
                confirm_score=1.0,
            )
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_calm_motion(),
    )

    assert str(action_kind) == "append_candidate"
    assert selected is not None
    assert str(controller._last_exact_v1_selection_reason) == "live_local_gates_passed"
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert diag["veto_reason"] is None
    assert bool(diag["zero_motion_projective_support"]) is True
    assert float(diag["condition_ratio_selected_vs_stay"]) > 1.0
    assert float(diag["displacement_ratio_selected_vs_stay"]) == math.inf


def test_observable_v1_spin_boson_projective_support_overrides_local_forecast_score_veto() -> None:
    controller = _no_harm_controller(mode="observable_v1")
    controller._family_key = "spin_boson"
    stay_forecast = _no_harm_forecast(
        score=1.6,
        rho_miss=1.0,
        condition=2.8e8,
        step_gain=0.0,
        displacement=0.0,
        step_residual=1.0,
    )
    selected_forecast = _no_harm_forecast(
        score=3.8e6,
        rho_miss=1.0e-10,
        condition=2.8e8,
        step_gain=0.97,
        displacement=5.0e-4,
        step_residual=0.025,
    )

    reason = controller._local_forecast_override_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        selected=_no_harm_candidate_record("full_meta::hop_0_1", gain_ratio=0.79),
        motion=_calm_motion(),
    )

    assert reason is None
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert bool(diag["projective_complete_support"]) is True
    assert bool(diag["residual_collapse_projective_support"]) is True


def test_observable_v1_molecular_projective_support_overrides_local_forecast_score_veto() -> None:
    controller = _no_harm_controller(
        mode="observable_v1",
        gain_ratio_threshold=0.01,
        append_no_harm_kink_min_step_gain_delta=0.0001,
        append_no_harm_rho_only_step_residual_ratio_cap=1.5,
        append_no_harm_rho_only_displacement_ratio_cap=2.0,
    )
    controller._family_key = "molecular_vibronic_h2"
    stay_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.7305,
        condition=6.6e6,
        step_gain=0.2694,
        displacement=1.00e-3,
        step_residual=0.7305,
    )
    selected_forecast = _no_harm_forecast(
        score=1.1,
        rho_miss=0.7174,
        condition=6.6e6,
        step_gain=0.2825,
        displacement=1.024e-3,
        step_residual=0.7174,
    )

    reason = controller._local_forecast_override_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        selected=_no_harm_candidate_record("ham_term(ezeee)", gain_ratio=0.015),
        motion=_calm_motion(),
    )

    assert reason is None
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert bool(diag["steady_projective_support"]) is True
    assert bool(diag["projective_complete_support"]) is True



def test_observable_v1_harmonic_kerr_projective_support_does_not_override_score_veto() -> None:
    controller = _no_harm_controller(mode="observable_v1")
    controller._family_key = "harmonic_kerr_chain"
    stay_forecast = _no_harm_forecast(
        score=1.6,
        rho_miss=1.0,
        condition=2.8e8,
        step_gain=0.0,
        displacement=0.0,
        step_residual=1.0,
    )
    selected_forecast = _no_harm_forecast(
        score=3.8e6,
        rho_miss=1.0e-10,
        condition=2.8e8,
        step_gain=0.97,
        displacement=5.0e-4,
        step_residual=0.025,
    )

    reason = controller._local_forecast_override_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        selected=_no_harm_candidate_record("full_meta::hop_0_1", gain_ratio=0.79),
        motion=_calm_motion(),
    )

    assert reason == "local_forecast_no_advantage"


def test_observable_v1_no_harm_allows_hamiltonian_flow_residual_collapse_support() -> None:
    controller = _no_harm_controller(
        mode="observable_v1",
        append_no_harm_condition_ratio_cap=1.25,
        append_no_harm_displacement_ratio_cap=1.25,
    )
    controller._family_key = "spin_boson"
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=0.9861,
        condition=1.0e6,
        step_gain=0.0138,
        displacement=6.6e-5,
        step_residual=0.9861,
    )
    selected_forecast = _no_harm_forecast(
        score=-100.0,
        rho_miss=0.0003,
        condition=1.0e6,
        step_gain=0.9997,
        displacement=5.7e-4,
        step_residual=0.0003,
    )

    reason, diag = controller._append_no_harm_guard_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        selected=_no_harm_candidate_record("full_meta::emitter_imbalance", gain_ratio=0.888),
        motion=_calm_motion(),
    )

    assert reason is None
    assert bool(diag["residual_collapse_projective_support"]) is True
    assert bool(diag["projective_complete_support"]) is True
    assert float(diag["displacement_ratio_selected_vs_stay"]) > 1.25


def test_observable_v1_no_harm_allows_hamiltonian_flow_zero_motion_projective_support() -> None:
    controller = _no_harm_controller(mode="observable_v1")
    controller._family_key = "spinless_tv"
    stay_forecast = _no_harm_forecast(
        score=10.0,
        rho_miss=1.0,
        condition=5.0e9,
        step_gain=0.0,
        displacement=0.0,
        step_residual=1.0,
    )
    selected_forecast = _no_harm_forecast(
        score=-100.0,
        rho_miss=0.0,
        condition=5.0e9,
        step_gain=1.0,
        displacement=2.0e-4,
        step_residual=1.0e-12,
    )

    reason, diag = controller._append_no_harm_guard_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        selected=_no_harm_candidate_record("ham_quad::nn_density(0,1)", gain_ratio=1.0),
        motion=_calm_motion(),
    )

    assert reason is None
    assert bool(diag["projective_append_support_mode"]) is True
    assert bool(diag["zero_motion_projective_support"]) is True
    assert float(diag["rho_miss_delta_stay_minus_selected"]) == pytest.approx(1.0)
    assert float(diag["step_gain_delta_selected_minus_stay"]) == pytest.approx(1.0)
    assert float(diag["displacement_ratio_selected_vs_stay"]) == math.inf


def test_exact_v1_no_harm_logs_bad_benchmark_exact_fields_without_veto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _no_harm_controller()
    stay_forecast = _no_harm_forecast(
        score=5.0,
        rho_miss=0.30,
        condition=10.0,
        step_gain=0.10,
        displacement=0.10,
        fidelity_exact_next=0.99,
        abs_energy_total_error_next=0.01,
        site_occupations_abs_error_max_next=0.02,
    )
    selected_forecast = _no_harm_forecast(
        score=1.0,
        rho_miss=0.10,
        condition=10.0,
        step_gain=0.13,
        displacement=0.10,
        fidelity_exact_next=0.10,
        abs_energy_total_error_next=99.0,
        site_occupations_abs_error_max_next=9.0,
    )

    def _fake_scale(**kwargs):
        return dict(kwargs["selected"]), dict(selected_forecast)

    monkeypatch.setattr(controller, "_select_exact_v1_candidate_step_scale", _fake_scale)
    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[_no_harm_candidate_record(gain_ratio=0.20)],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
        motion=_calm_motion(),
    )

    assert str(action_kind) == "append_candidate"
    assert selected is not None
    diag = controller._last_append_no_harm_diagnostics
    assert diag is not None
    assert diag["veto_reason"] is None
    exact_log = diag["exact_reference_logging"]
    assert exact_log["logging_only"] is True
    assert exact_log["used_for_veto"] is False
    assert float(exact_log["selected_fidelity_exact_next"]) == pytest.approx(0.10)
    assert float(exact_log["selected_abs_energy_total_error_next"]) == pytest.approx(99.0)


def test_exact_v1_below_floor_energy_safe_window_turn_escape_allows_better_turn_candidate() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_turn_escape=True,
            exact_forecast_density_sign_lag_weight=1.0,
            exact_v1_sign_lag_window_activation=0.10,
            exact_v1_sign_lag_window_target_gain_floor=0.005,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.40, 0.10],
        "tracking_site_curvature_abs_error_mean_by_site": [0.20, 0.05],
        "tracking_site_excursion_under_response_mean_by_site": [0.30, 0.06],
        "tracking_site_excursion_over_response_mean_by_site": [0.08, 0.01],
        "primary_density_sign_lag_next": 0.20,
        "abs_primary_density_sign_lag_next": 0.20,
        "normalized_primary_density_error_next": 0.30,
        "abs_primary_density_error_next": 0.30,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.0,
        "abs_energy_total_error_next": 1.95e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.30, 0.09],
        "tracking_site_curvature_abs_error_mean_by_site": [0.15, 0.04],
        "tracking_site_excursion_under_response_mean_by_site": [0.22, 0.05],
        "tracking_site_excursion_over_response_mean_by_site": [0.05, 0.01],
        "primary_density_sign_lag_next": 0.12,
        "abs_primary_density_sign_lag_next": 0.12,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is True
    assert reason is None


def test_exact_v1_below_floor_energy_safe_window_turn_escape_keeps_raw_energy_guard() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_turn_escape=True,
            exact_forecast_density_sign_lag_weight=1.0,
            exact_v1_sign_lag_window_activation=0.10,
            exact_v1_sign_lag_window_target_gain_floor=0.005,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.40, 0.10],
        "tracking_site_curvature_abs_error_mean_by_site": [0.20, 0.05],
        "tracking_site_excursion_under_response_mean_by_site": [0.30, 0.06],
        "tracking_site_excursion_over_response_mean_by_site": [0.08, 0.01],
        "primary_density_sign_lag_next": 0.20,
        "abs_primary_density_sign_lag_next": 0.20,
        "normalized_primary_density_error_next": 0.30,
        "abs_primary_density_error_next": 0.30,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.0,
        "abs_energy_total_error_next": 2.05e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.30, 0.09],
        "tracking_site_curvature_abs_error_mean_by_site": [0.15, 0.04],
        "tracking_site_excursion_under_response_mean_by_site": [0.22, 0.05],
        "tracking_site_excursion_over_response_mean_by_site": [0.05, 0.01],
        "primary_density_sign_lag_next": 0.12,
        "abs_primary_density_sign_lag_next": 0.12,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is False
    assert str(reason) == "outside_energy_safe_window"


def test_exact_v1_below_floor_energy_safe_window_turn_escape_requires_stay_side_sign_lag_failure() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_turn_escape=True,
            exact_forecast_density_sign_lag_weight=1.0,
            exact_v1_sign_lag_window_activation=0.10,
            exact_v1_sign_lag_window_target_gain_floor=0.005,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.40, 0.10],
        "tracking_site_curvature_abs_error_mean_by_site": [0.20, 0.05],
        "tracking_site_excursion_under_response_mean_by_site": [0.30, 0.06],
        "tracking_site_excursion_over_response_mean_by_site": [0.08, 0.01],
        "primary_density_sign_lag_next": 0.05,
        "abs_primary_density_sign_lag_next": 0.05,
        "normalized_primary_density_error_next": 0.30,
        "abs_primary_density_error_next": 0.30,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.0,
        "abs_energy_total_error_next": 1.95e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.30, 0.09],
        "tracking_site_curvature_abs_error_mean_by_site": [0.15, 0.04],
        "tracking_site_excursion_under_response_mean_by_site": [0.22, 0.05],
        "tracking_site_excursion_over_response_mean_by_site": [0.05, 0.01],
        "primary_density_sign_lag_next": 0.25,
        "abs_primary_density_sign_lag_next": 0.25,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is False
    assert str(reason) == "outside_energy_safe_window"


def test_exact_v1_below_floor_energy_safe_window_turn_escape_requires_density_win() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_turn_escape=True,
            exact_forecast_density_sign_lag_weight=1.0,
            exact_v1_sign_lag_window_activation=0.10,
            exact_v1_sign_lag_window_target_gain_floor=0.005,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.40, 0.10],
        "tracking_site_curvature_abs_error_mean_by_site": [0.20, 0.05],
        "tracking_site_excursion_under_response_mean_by_site": [0.30, 0.06],
        "tracking_site_excursion_over_response_mean_by_site": [0.08, 0.01],
        "primary_density_sign_lag_next": 0.20,
        "abs_primary_density_sign_lag_next": 0.20,
        "normalized_primary_density_error_next": 0.30,
        "abs_primary_density_error_next": 0.30,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.0,
        "abs_energy_total_error_next": 1.95e-2,
        "tracking_site_slope_abs_error_mean_by_site": [0.30, 0.09],
        "tracking_site_curvature_abs_error_mean_by_site": [0.15, 0.04],
        "tracking_site_excursion_under_response_mean_by_site": [0.22, 0.05],
        "tracking_site_excursion_over_response_mean_by_site": [0.05, 0.01],
        "primary_density_sign_lag_next": 0.12,
        "abs_primary_density_sign_lag_next": 0.12,
        "normalized_primary_density_error_next": 0.297,
        "abs_primary_density_error_next": 0.297,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is False
    assert str(reason) == "outside_energy_safe_window"


def test_exact_v1_below_floor_energy_safe_window_d_shape_escape_allows_better_d_shape_candidate() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_d_shape_escape=True,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_d_curvature_abs_error_mean": 0.05,
        "tracking_d_excursion_under_response_mean": 0.04,
        "tracking_d_excursion_over_response_mean": 0.03,
        "tracking_total_occupation_abs_error_next": 2.0e-3,
        "tracking_total_occupation_abs_error_mean": 1.5e-3,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.6,
        "abs_energy_total_error_next": 1.95e-2,
        "tracking_d_curvature_abs_error_mean": 0.04,
        "tracking_d_excursion_under_response_mean": 0.03,
        "tracking_d_excursion_over_response_mean": 0.02,
        "tracking_total_occupation_abs_error_next": 2.0e-3,
        "tracking_total_occupation_abs_error_mean": 1.4e-3,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is True
    assert reason is None


def test_exact_v1_below_floor_energy_safe_window_d_shape_escape_requires_d_shape_win() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_d_shape_escape=True,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_d_curvature_abs_error_mean": 0.05,
        "tracking_d_excursion_under_response_mean": 0.04,
        "tracking_d_excursion_over_response_mean": 0.03,
        "tracking_total_occupation_abs_error_next": 2.0e-3,
        "tracking_total_occupation_abs_error_mean": 1.5e-3,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.6,
        "abs_energy_total_error_next": 1.95e-2,
        "tracking_d_curvature_abs_error_mean": 0.05,
        "tracking_d_excursion_under_response_mean": 0.04,
        "tracking_d_excursion_over_response_mean": 0.03,
        "tracking_total_occupation_abs_error_next": 2.0e-3,
        "tracking_total_occupation_abs_error_mean": 1.4e-3,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is False
    assert str(reason) == "outside_energy_safe_window"


def test_exact_v1_below_floor_energy_safe_window_d_shape_escape_requires_total_occupation_nonworsening() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_d_shape_escape=True,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_d_curvature_abs_error_mean": 0.05,
        "tracking_d_excursion_under_response_mean": 0.04,
        "tracking_d_excursion_over_response_mean": 0.03,
        "tracking_total_occupation_abs_error_next": 2.0e-3,
        "tracking_total_occupation_abs_error_mean": 1.5e-3,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.6,
        "abs_energy_total_error_next": 1.95e-2,
        "tracking_d_curvature_abs_error_mean": 0.04,
        "tracking_d_excursion_under_response_mean": 0.03,
        "tracking_d_excursion_over_response_mean": 0.02,
        "tracking_total_occupation_abs_error_next": 2.1e-3,
        "tracking_total_occupation_abs_error_mean": 1.4e-3,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is False
    assert str(reason) == "outside_energy_safe_window"


def test_exact_v1_below_floor_energy_safe_window_d_shape_escape_still_respects_energy_shape_caps() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_below_floor_energy_safe_d_shape_escape=True,
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
    stay_forecast = {
        "tracking_score_horizon": 5.0,
        "abs_energy_total_error_next": 2.0e-2,
        "tracking_d_curvature_abs_error_mean": 0.05,
        "tracking_d_excursion_under_response_mean": 0.04,
        "tracking_d_excursion_over_response_mean": 0.03,
        "tracking_total_occupation_abs_error_next": 2.0e-3,
        "tracking_total_occupation_abs_error_mean": 1.5e-3,
        "tracking_energy_curvature_abs_error_mean": 0.01,
    }
    selected_forecast = {
        "tracking_score_horizon": 4.6,
        "abs_energy_total_error_next": 1.95e-2,
        "tracking_d_curvature_abs_error_mean": 0.04,
        "tracking_d_excursion_under_response_mean": 0.03,
        "tracking_d_excursion_over_response_mean": 0.02,
        "tracking_total_occupation_abs_error_next": 2.0e-3,
        "tracking_total_occupation_abs_error_mean": 1.4e-3,
        "tracking_energy_curvature_abs_error_mean": 0.0401,
    }

    ok, reason = controller._exact_v1_below_floor_energy_safe_window(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert ok is False
    assert str(reason) == "fails_energy_curvature_window"


def test_exact_v1_select_action_rejects_negative_confirm_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
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

    monkeypatch.setattr(
        controller,
        "_select_exact_v1_candidate_step_scale",
        lambda **kwargs: pytest.fail("negative confirm score should reject before forecasting"),
    )

    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _record(
                "candidate_a",
                confirm_score=-1.0e-3,
                gain_ratio=0.05,
                gain_exact=2.0e-6,
                pool_index=0,
            )
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast={"local_projective_score_total": 1.0},
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "confirm_score_below_threshold"


def test_exact_v1_select_action_keeps_stay_when_append_lacks_local_advantage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
            forecast_accept_margin=0.0,
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

    monkeypatch.setattr(
        controller,
        "_select_exact_v1_candidate_step_scale",
        lambda **kwargs: (
            dict(kwargs["selected"]),
            {"local_projective_score_total": 1.2, "tracking_score_horizon": 1.2},
        ),
    )

    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _record(
                "candidate_a",
                confirm_score=2.0,
                gain_ratio=0.05,
                gain_exact=2.0e-6,
                pool_index=0,
            )
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast={"local_projective_score_total": 0.8, "tracking_score_horizon": 0.8},
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "local_forecast_no_advantage"


def test_primary_density_sign_lag_terms_penalize_delayed_reversal() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", exact_forecast_density_sign_lag_weight=1.0),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    terms = controller._primary_density_sign_lag_terms(
        forecasts=[
            {
                "primary_density_controller_next": 0.8,
                "primary_density_exact_next": -0.2,
            },
            {
                "primary_density_controller_next": 0.6,
                "primary_density_exact_next": -0.4,
            },
        ],
        weights=[2.0, 1.0],
        anchor={
            "primary_density_controller_next": 0.9,
            "primary_density_exact_next": 0.5,
        },
        primary_density_scale=1.0,
    )
    assert terms["abs_primary_density_sign_lag_next"] == pytest.approx(1.0)
    assert terms["primary_density_sign_lag_next"] == pytest.approx(1.0)
    assert terms["primary_density_sign_lag_abs_error_mean"] == pytest.approx(1.0)
    assert terms["primary_density_sign_lag_error_mean"] == pytest.approx(1.0)


def test_exact_v1_componentwise_aspiration_allows_sign_lag_win_when_enabled() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", exact_forecast_density_sign_lag_weight=1.0),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    stay_forecast = {
        "fidelity_exact_next": 0.98,
        "normalized_primary_density_error_next": 0.40,
        "abs_primary_density_error_next": 0.40,
        "primary_density_slope_error_next": 0.40,
        "abs_primary_density_slope_error_next": 0.40,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "site_occupations_abs_error_max_next": 0.10,
        "primary_density_sign_lag_next": 0.20,
        "abs_primary_density_sign_lag_next": 0.20,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.98,
        "normalized_primary_density_error_next": 0.40,
        "abs_primary_density_error_next": 0.40,
        "primary_density_slope_error_next": 0.40,
        "abs_primary_density_slope_error_next": 0.40,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "site_occupations_abs_error_max_next": 0.10,
        "primary_density_sign_lag_next": 0.00,
        "abs_primary_density_sign_lag_next": 0.00,
    }
    allowed, reason = controller._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )
    assert allowed is True
    assert reason is None


def test_exact_v1_select_action_low_gain_records_gain_gate_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
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

    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
        "primary_density_slope_error_next": 0.10,
        "abs_primary_density_slope_error_next": 0.10,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "site_occupations_abs_error_max_next": 0.04,
    }
    selected_forecast = dict(stay_forecast)

    monkeypatch.setattr(
        controller,
        "_select_exact_v1_candidate_step_scale",
        lambda **kwargs: (dict(kwargs["selected"]), dict(selected_forecast)),
    )

    action_kind, selected = controller._select_action_exact_v1(
        baseline={
            "summary": SimpleNamespace(rho_miss=1.0),
            "theta_dot_step": np.asarray([0.0], dtype=float),
        },
        confirmed=[
            _record(
                "candidate_a",
                confirm_score=2.0,
                gain_ratio=0.015,
                gain_exact=1.0e-6,
                pool_index=0,
            )
        ],
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "gain_ratio_below_threshold"


def test_realtime_controller_analytic_noise_zero_std_matches_exact_baseline() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller_plain = RealtimeCheckpointController(
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
    controller_zero = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_std=0.0,
            analytic_noise_seed=17,
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

    baseline_plain = _baseline_geometry_payload(controller_plain)
    baseline_zero = _baseline_geometry_payload(controller_zero)

    assert bool(baseline_plain["analytic_noise_applied"]) is False
    assert bool(baseline_zero["analytic_noise_applied"]) is False
    assert baseline_zero["summary"].rho_miss == pytest.approx(baseline_plain["summary"].rho_miss)
    assert np.asarray(baseline_zero["G"], dtype=float) == pytest.approx(
        np.asarray(baseline_plain["G"], dtype=float)
    )
    assert np.asarray(baseline_zero["f"], dtype=float) == pytest.approx(
        np.asarray(baseline_plain["f"], dtype=float)
    )
    assert np.asarray(baseline_zero["theta_dot_step"], dtype=float) == pytest.approx(
        np.asarray(baseline_plain["theta_dot_step"], dtype=float)
    )


def test_realtime_controller_analytic_noise_seed_is_reproducible_and_symmetric() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller_a = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_std=0.25,
            analytic_noise_seed=7,
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
    controller_b = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_std=0.25,
            analytic_noise_seed=7,
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
    controller_c = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_std=0.25,
            analytic_noise_seed=8,
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

    baseline_a = _baseline_geometry_payload(controller_a)
    baseline_b = _baseline_geometry_payload(controller_b)
    baseline_c = _baseline_geometry_payload(controller_c)

    assert bool(baseline_a["analytic_noise_applied"]) is True
    assert baseline_a["analytic_noise_degraded_reason"] is None
    assert np.asarray(baseline_a["G"], dtype=float) == pytest.approx(
        np.asarray(baseline_b["G"], dtype=float)
    )
    assert np.asarray(baseline_a["f"], dtype=float) == pytest.approx(
        np.asarray(baseline_b["f"], dtype=float)
    )
    assert np.asarray(baseline_a["G"], dtype=float) == pytest.approx(
        np.asarray(baseline_a["G"], dtype=float).T
    )
    assert not np.allclose(
        np.asarray(baseline_a["G"], dtype=float),
        np.asarray(baseline_c["G"], dtype=float),
    )


def test_realtime_controller_hybrid_proxy_shot_scaling_and_psd_guard() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    common_kwargs = dict(
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    controller_low = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_model="hybrid_qpu_proxy_v1",
            analytic_noise_std=0.35,
            analytic_noise_seed=21,
            analytic_noise_nominal_shots=256,
            analytic_noise_two_qubit_depth_scale=0.2,
            analytic_noise_groups_new_scale=0.1,
            analytic_noise_force_psd=True,
        ),
        **common_kwargs,
    )
    controller_high = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_model="hybrid_qpu_proxy_v1",
            analytic_noise_std=0.35,
            analytic_noise_seed=21,
            analytic_noise_nominal_shots=8192,
            analytic_noise_two_qubit_depth_scale=0.2,
            analytic_noise_groups_new_scale=0.1,
            analytic_noise_force_psd=True,
        ),
        **common_kwargs,
    )

    baseline_low = _baseline_geometry_payload(controller_low)
    baseline_high = _baseline_geometry_payload(controller_high)

    assert bool(baseline_low["analytic_noise_applied"]) is True
    assert str(baseline_low["analytic_noise_model"]) == "hybrid_qpu_proxy_v1"
    assert float(baseline_low["analytic_noise_features"]["resolved_scale"]) > float(
        baseline_high["analytic_noise_features"]["resolved_scale"]
    )
    assert float(baseline_low["analytic_noise_features"]["shots_eff"]) < float(
        baseline_high["analytic_noise_features"]["shots_eff"]
    )
    eigvals = np.linalg.eigvalsh(np.asarray(baseline_low["G"], dtype=float))
    assert float(np.min(eigvals)) >= -1.0e-8


def test_realtime_controller_hybrid_proxy_group_burden_increases_noise_scale() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    common_kwargs = dict(
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    controller_light = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_model="hybrid_qpu_proxy_v1",
            analytic_noise_std=0.25,
            analytic_noise_seed=33,
            analytic_noise_groups_new_scale=0.35,
        ),
        **common_kwargs,
    )
    controller_heavy = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_model="hybrid_qpu_proxy_v1",
            analytic_noise_std=0.25,
            analytic_noise_seed=33,
            analytic_noise_groups_new_scale=0.35,
        ),
        **common_kwargs,
    )
    controller_light._planning_group_burden = lambda summary: 1.0
    controller_heavy._planning_group_burden = lambda summary: 16.0

    baseline_light = _baseline_geometry_payload(controller_light)
    baseline_heavy = _baseline_geometry_payload(controller_heavy)

    assert float(baseline_heavy["analytic_noise_features"]["group_burden"]) > float(
        baseline_light["analytic_noise_features"]["group_burden"]
    )
    assert float(baseline_heavy["analytic_noise_features"]["resolved_scale"]) > float(
        baseline_light["analytic_noise_features"]["resolved_scale"]
    )


def test_realtime_controller_hybrid_proxy_time_correlation_reduces_successive_jump() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    common_kwargs = dict(
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    controller_uncorrelated = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_model="hybrid_qpu_proxy_v1",
            analytic_noise_std=0.25,
            analytic_noise_seed=51,
            analytic_noise_time_corr=0.0,
        ),
        **common_kwargs,
    )
    controller_correlated = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            analytic_noise_model="hybrid_qpu_proxy_v1",
            analytic_noise_std=0.25,
            analytic_noise_seed=51,
            analytic_noise_time_corr=0.9,
        ),
        **common_kwargs,
    )

    first_uncorrelated = np.asarray(
        controller_uncorrelated._add_vector_gaussian_noise(np.zeros(6, dtype=float)),
        dtype=float,
    )
    second_uncorrelated = np.asarray(
        controller_uncorrelated._add_vector_gaussian_noise(np.zeros(6, dtype=float)),
        dtype=float,
    )
    first_correlated = np.asarray(
        controller_correlated._add_vector_gaussian_noise(np.zeros(6, dtype=float)),
        dtype=float,
    )
    second_correlated = np.asarray(
        controller_correlated._add_vector_gaussian_noise(np.zeros(6, dtype=float)),
        dtype=float,
    )

    assert first_correlated == pytest.approx(first_uncorrelated)
    assert np.linalg.norm(second_correlated - first_correlated) < np.linalg.norm(
        second_uncorrelated - first_uncorrelated
    )


def test_realtime_controller_analytic_noise_nonfinite_metrics_degrade_to_stay(
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
            analytic_noise_std=0.5,
            analytic_noise_seed=23,
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
        "_add_symmetric_gaussian_noise",
        lambda value: np.full_like(np.asarray(value, dtype=float), np.inf),
    )
    monkeypatch.setattr(controller, "_exact_v1_forecast_override_reason", lambda **kwargs: None)

    result = controller.run()

    assert str(result.trajectory[0]["action_kind"]) == "stay"
    assert str(result.trajectory[0]["controller_lane"]) == "stay"
    assert str(result.trajectory[0]["controller_lane_reason"]) == "analytic_noise_nonfinite_metric"
    assert str(result.trajectory[0]["degraded_reason"]) == "analytic_noise_nonfinite_metric"
    assert float(result.ledger[0]["analytic_noise_std"]) == pytest.approx(0.5)
    assert int(result.ledger[0]["analytic_noise_seed"]) == 23


def test_realtime_controller_prune_lane_can_commit_coordinate_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=1.0e-3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_appended_origin_target_policy="bias_only",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0,
            prune_theta_block_tol=1.0,
            prune_state_jump_l2_tol=1.0,
            prune_safe_miss_increase_tol=1.0,
            prune_no_harm_guard_enabled=False,
            prune_max_candidates=1,
            prune_initial_scaffold_grace_steps=0,
            prune_state_jump_l2_hard_cap=1.0,
            prune_active_block_theta_dot_rel_tol=2.0,
            prune_active_block_theta_dot_abs_hard_tol=1.0,
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
    assert str(result.trajectory[0]["selected_prune_origin_kind"]) == "initial_scaffold"
    assert result.trajectory[0]["selected_prune_appended_origin_bias_applied"] is False
    assert result.trajectory[0]["post_prune_energy_total"] is not None
    assert result.trajectory[0]["post_prune_fidelity_exact"] is not None


def test_realtime_controller_records_compile_audit_prune_boundary_snapshot() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=1.0e-3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 1.0e-3],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )

    reduced_state = controller._build_pruned_runtime_state(logical_index=1)
    controller._record_compile_audit_prune_event(
        checkpoint_index=3,
        time_value=0.15,
        selected_candidate_label=str(reduced_state["removed_label"]),
        removed_label=str(reduced_state["removed_label"]),
        logical_before=int(controller.current_layout.logical_parameter_count),
        runtime_before=int(controller.current_layout.runtime_parameter_count),
        reduced_state=reduced_state,
    )

    assert len(controller._compile_audit_prune_events) == 1
    prune_event = controller._compile_audit_prune_events[0]
    assert int(prune_event["checkpoint_index"]) == 3
    assert int(prune_event["runtime_parameter_count_delta"]) < 0
    assert len(prune_event["before"]["labels"]) == 2
    assert len(prune_event["after"]["labels"]) == 1
    assert str(prune_event["before"]["labels"][1]).startswith("op_y")
    assert str(prune_event["after"]["labels"][0]).startswith("op_x")


def test_realtime_controller_prune_no_harm_rejects_segment_residual_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=1.0e-3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_appended_origin_target_policy="bias_only",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0,
            prune_theta_block_tol=1.0,
            prune_state_jump_l2_tol=1.0,
            prune_safe_miss_increase_tol=1.0,
            prune_max_candidates=1,
            prune_initial_scaffold_grace_steps=0,
            prune_state_jump_l2_hard_cap=1.0,
            prune_active_block_theta_dot_rel_tol=1.0,
            prune_active_block_theta_dot_abs_hard_tol=1.0,
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

    assert int(result.summary["prune_count"]) == 0
    assert str(result.trajectory[0]["controller_lane"]) == "prune"
    assert str(result.trajectory[0]["action_kind"]) == "stay"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "prune_coordinate"
    assert str(result.trajectory[0]["decision_override_reason"]) == "prune_rejected_prune_no_harm_score_increase_above_tol"
    candidate = result.trajectory[0]["prune_candidates"][0]
    assert candidate["prune_accept"] is False
    assert str(candidate["prune_rejection_reason"]) == "prune_no_harm_score_increase_above_tol"
    assert candidate["prune_no_harm_diagnostics"]["prune_no_harm_uses_exact_reference"] is False
    assert float(candidate["prune_no_harm_step_residual_ratio_delta"]) > 0.0


def test_realtime_controller_stays_when_prune_theta_block_is_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=0.3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_appended_origin_target_policy="bias_only",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0,
            prune_theta_block_tol=1.0e-4,
            prune_state_jump_l2_tol=1.0,
            prune_safe_miss_increase_tol=1.0,
            prune_max_candidates=1,
            prune_initial_scaffold_grace_steps=0,
            prune_state_jump_l2_hard_cap=1.0,
            prune_active_block_theta_dot_rel_tol=1.0,
            prune_active_block_theta_dot_abs_hard_tol=1.0,
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
            prune_appended_origin_target_policy="bias_only",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0e-3,
            prune_theta_block_tol=1.0,
            prune_state_jump_l2_tol=1.0,
            prune_safe_miss_increase_tol=1.0,
            prune_max_candidates=2,
            prune_initial_scaffold_grace_steps=0,
            prune_state_jump_l2_hard_cap=1.0,
            prune_active_block_theta_dot_rel_tol=1.0,
            prune_active_block_theta_dot_abs_hard_tol=1.0,
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


def test_realtime_controller_recoverability_prune_emits_schur_projection_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=1.0e-3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="schur_projected_shadow_v1",
            prune_appended_origin_target_policy="bias_only",
            prune_miss_threshold=2.0,
            prune_loss_threshold=1.0,
            prune_theta_block_tol=1.0e-8,
            prune_state_jump_l2_tol=1.0,
            prune_state_jump_l2_hard_cap=1.0,
            prune_differential_miss_tol=1.0,
            prune_no_harm_guard_enabled=False,
            prune_persistence_window=1,
            prune_persistence_required=1,
            prune_max_candidates=1,
            prune_initial_scaffold_grace_steps=0,
            prune_active_block_theta_dot_rel_tol=1.0,
            prune_active_block_theta_dot_abs_hard_tol=1.0,
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
    monkeypatch.setattr(controller, "_motion_telemetry", lambda **kwargs: _calm_motion())

    result = controller.run()

    assert int(result.summary["prune_count"]) == 1
    row = result.trajectory[0]
    assert str(row["action_kind"]) == "prune_coordinate"
    candidate = row["prune_candidates"][0]
    assert str(candidate["cached_prune_loss_semantics"]) == "schur_normalized_v1"
    assert str(candidate["prune_loss_selected_kind"]) == "compat_schur_normalized_v1"
    assert str(candidate["prune_loss_denominator_kind"]) == "max_norm_b_sq_epsilon_compat_v1"
    assert str(candidate["prune_loss_theorem_denominator_kind"]) == "norm_b_sq_plus_epsilon_v1"
    assert str(candidate["prune_loss_matrix_for_selection"]) == "compat_schur_k"
    assert candidate["prune_loss_delta_g_theorem"] is not None
    assert candidate["prune_loss_delta_k_damped"] is not None
    assert candidate["prune_loss_selected"] == pytest.approx(float(candidate["cached_prune_loss"]))
    assert str(candidate["prune_permit_path"]) == "low_miss_standard"
    assert candidate["prune_schur_normalized_loss"] is not None
    assert candidate["prune_projection_objective"] is not None
    assert candidate["prune_projected_state_jump_l2"] is not None
    assert candidate["prune_persistence_passed"] is True
    assert row["selected_prune_loss"] == pytest.approx(float(candidate["cached_prune_loss"]))
    assert str(row["selected_prune_loss_kind"]) == "compat_schur_normalized_v1"
    assert result.ledger[0]["prune_schur_normalized_loss_selected"] is not None
    assert result.ledger[0]["selected_prune_loss"] == pytest.approx(float(candidate["cached_prune_loss"]))
    assert str(result.ledger[0]["selected_prune_loss_kind"]) == "compat_schur_normalized_v1"
    assert result.ledger[0]["prune_projection_objective_selected"] is not None


def test_realtime_controller_recoverability_prune_can_nominate_high_miss_noncalm_block() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=1.0e-3)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            prune_mode="schur_projected_shadow_v1",
            prune_appended_origin_target_policy="bias_only",
            prune_miss_threshold=0.0,
            prune_loss_threshold=1.0,
            prune_protection_steps=0,
            prune_max_candidates=1,
            prune_initial_scaffold_grace_steps=0,
            prune_active_block_theta_dot_rel_tol=1.0,
            prune_active_block_theta_dot_abs_hard_tol=1.0,
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
    baseline = dict(_baseline_geometry_payload(controller))
    baseline["summary"] = dataclass_replace(
        baseline["summary"],
        rho_miss=0.5,
    )
    noncalm = MotionSchedulerTelemetry(
        regime="kink",
        direction_cosine=0.0,
        rate_change_l2=1.0,
        rate_change_ratio=10.0,
        acceleration_l2=1.0,
        curvature_cosine=-1.0,
        direction_reversal=True,
        curvature_sign_flip=True,
        kink_score=1.0,
    )

    rows, reason = controller._prune_candidates(
        checkpoint_index=4,
        baseline=baseline,
        motion=noncalm,
    )

    assert str(reason) == "prune_candidates_available"
    assert rows
    assert str(rows[0]["prune_permit_path"]) == "high_miss_differential"
    assert str(rows[0]["cached_prune_loss_semantics"]) == "schur_normalized_v1"


def test_realtime_controller_prune_active_block_absolute_hard_cap_blocks_low_relative_motion() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.0, theta_y=0.0)

    def _controller(*, abs_hard_tol: float) -> RealtimeCheckpointController:
        controller = RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(
                mode="exact_v1",
                miss_threshold=2.0,
                prune_mode="exact_local_v1",
                prune_miss_threshold=2.0,
                prune_protection_steps=0,
                prune_initial_scaffold_grace_steps=0,
                prune_stagnation_window=1,
                prune_stale_score_threshold=0.0,
                prune_appended_origin_target_policy="bias_only",
                prune_active_block_theta_dot_rel_tol=0.03,
                prune_active_block_theta_dot_abs_hard_tol=float(abs_hard_tol),
            ),
            replay_context=replay_context,
            h_poly=h_poly,
            hmat=hmat,
            psi_initial=psi_initial,
            best_theta=[0.0, 0.0],
            allow_repeats=False,
            t_final=0.2,
            num_times=2,
        )
        for label in controller._current_scaffold_labels():
            controller._block_motion_history[label] = [0.0]
            controller._block_fit_history[label] = [0.0]
        return controller

    permissive = _controller(abs_hard_tol=1.0)
    baseline = dict(_baseline_geometry_payload(permissive))
    baseline["theta_dot_step"] = np.asarray([0.06, 10.0], dtype=float)
    rows, reason = permissive._prune_candidates(
        checkpoint_index=4,
        baseline=baseline,
        motion=_calm_motion(),
    )
    assert str(reason) == "prune_candidates_available"
    assert any(str(row["candidate_label"]).startswith("op_x") for row in rows)

    guarded = _controller(abs_hard_tol=5.0e-2)
    baseline = dict(_baseline_geometry_payload(guarded))
    baseline["theta_dot_step"] = np.asarray([0.06, 10.0], dtype=float)
    rows, reason = guarded._prune_candidates(
        checkpoint_index=4,
        baseline=baseline,
        motion=_calm_motion(),
    )
    assert rows == []
    assert str(reason) == "no_prune_eligible_coordinates"


def test_realtime_controller_default_prune_policy_protects_original_scaffold_until_append() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=0.0)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_miss_threshold=2.0,
            prune_protection_steps=0,
            prune_stagnation_window=1,
            prune_stale_score_threshold=0.0,
            prune_appended_origin_bias_enabled=True,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 0.0],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    for label in controller._current_scaffold_labels():
        controller._block_motion_history[label] = [0.0]
        controller._block_fit_history[label] = [0.0]
    baseline = _baseline_geometry_payload(controller)

    rows, reason = controller._prune_candidates(
        checkpoint_index=4,
        baseline=baseline,
        motion=_calm_motion(),
    )

    assert rows == []
    assert str(reason) == "no_appended_prune_targets"


def test_realtime_controller_prune_bias_does_not_change_append_scout_scoring() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)

    def _controller(*, enabled: bool) -> RealtimeCheckpointController:
        return RealtimeCheckpointController(
            cfg=RealtimeCheckpointConfig(
                mode="exact_v1",
                shortlist_size=4,
                shortlist_fraction=1.0,
                prune_appended_origin_bias_enabled=bool(enabled),
                prune_appended_origin_bias_scale=100.0,
                prune_appended_origin_bias_max_factor=100.0,
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

    unbiased = _controller(enabled=False)
    biased = _controller(enabled=True)
    ctx_u, cache_u, memo_u, baseline_u = _controller_checkpoint_geometry(unbiased)
    ctx_b, cache_b, memo_b, baseline_b = _controller_checkpoint_geometry(biased)
    shortlist_u, records_u = unbiased._scout_candidates_with_records(
        checkpoint_ctx=ctx_u,
        cache=cache_u,
        geometry_memo=memo_u,
        baseline=baseline_u,
        predicted_displacement=unbiased._predicted_displacement(dt=0.1, baseline=baseline_u),
    )
    shortlist_b, records_b = biased._scout_candidates_with_records(
        checkpoint_ctx=ctx_b,
        cache=cache_b,
        geometry_memo=memo_b,
        baseline=baseline_b,
        predicted_displacement=biased._predicted_displacement(dt=0.1, baseline=baseline_b),
    )

    numeric_keys = (
        "residual_overlap_l2",
        "compile_proxy_total",
        "groups_new",
        "position_jump_penalty",
        "temporal_prior_bonus",
        "scout_lower_gain",
        "scout_gain_ratio",
        "scout_score",
        "simple_score",
    )
    identity_keys = ("candidate_label", "candidate_identity", "candidate_pool_index", "position_id")
    assert len(records_b) == len(records_u)
    assert [str(row["candidate_label"]) for row in shortlist_b] == [
        str(row["candidate_label"]) for row in shortlist_u
    ]
    for row_b, row_u in zip(records_b, records_u):
        for key in identity_keys:
            assert row_b[key] == row_u[key]
        for key in numeric_keys:
            assert float(row_b[key]) == pytest.approx(float(row_u[key]))


def test_realtime_controller_prune_bias_grace_protects_just_appended_zero_coordinate() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=0.0)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=2.0,
            prune_mode="exact_local_v1",
            prune_miss_threshold=2.0,
            prune_protection_steps=0,
            prune_stagnation_window=1,
            prune_stale_score_threshold=0.75,
            prune_appended_origin_bias_enabled=True,
            prune_appended_origin_grace_steps=1,
            prune_appended_origin_bias_scale=1.0,
            prune_appended_origin_bias_max_factor=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 0.0],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    x_label, y_label = controller._current_scaffold_labels()
    controller._block_origin[y_label] = "append"
    controller._block_birth_checkpoint[y_label] = 0
    controller._block_motion_history[x_label] = [1.0]
    controller._block_fit_history[x_label] = [1.0]
    controller._block_motion_history[y_label] = [0.0]
    controller._block_fit_history[y_label] = [0.0]
    baseline = _baseline_geometry_payload(controller)

    rows, reason = controller._prune_candidates(
        checkpoint_index=1,
        baseline=baseline,
        motion=_calm_motion(),
    )

    assert rows == []
    assert str(reason) == "no_appended_prune_eligible_coordinates"


def test_realtime_controller_prune_bias_prefers_stale_appended_over_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_block_context(theta_x=0.2, theta_y=0.0)
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
            prune_no_harm_guard_enabled=False,
            prune_protection_steps=0,
            prune_stagnation_window=1,
            prune_stale_score_threshold=0.75,
            prune_max_candidates=1,
            prune_appended_origin_bias_enabled=True,
            prune_appended_origin_grace_steps=1,
            prune_appended_origin_bias_scale=0.50,
            prune_appended_origin_bias_max_factor=1.0,
            prune_initial_scaffold_grace_steps=0,
            prune_state_jump_l2_hard_cap=1.0,
            prune_active_block_theta_dot_rel_tol=2.0,
            prune_active_block_theta_dot_abs_hard_tol=1.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 0.0],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    x_label, y_label = controller._current_scaffold_labels()
    controller._block_origin[y_label] = "append"
    controller._block_birth_checkpoint[y_label] = 0
    for label in (x_label, y_label):
        controller._block_motion_history[label] = [0.0]
        controller._block_fit_history[label] = [0.0]

    def _fake_prune_loss(**kwargs: object) -> float:
        runtime_indices = list(kwargs["runtime_indices"])
        return 0.10 if int(runtime_indices[0]) == 0 else 0.15

    monkeypatch.setattr(controller, "_cached_prune_loss", _fake_prune_loss)
    baseline = _baseline_geometry_payload(controller)
    rows, reason = controller._prune_candidates(
        checkpoint_index=4,
        baseline=baseline,
        motion=_calm_motion(),
    )

    assert str(reason) == "prune_candidates_available"
    assert len(rows) == 1
    assert str(rows[0]["candidate_label"]) == str(y_label)
    assert str(rows[0]["origin_kind"]) == "append"
    assert str(rows[0]["appended_origin_target_policy"]) == "append_only"
    assert bool(rows[0]["appended_origin_target_policy_applied"]) is True
    assert bool(rows[0]["appended_origin_bias_applied"]) is True
    assert float(rows[0]["appended_origin_bias_factor"]) > 0.0
    assert float(rows[0]["prune_selection_score"]) < float(rows[0]["cached_prune_loss"])

    step_hamiltonian = controller._step_hamiltonian_artifacts(0.0)
    action_kind, selected, _proposed, evaluated, error = controller._select_prune_action(
        checkpoint_index=4,
        time_value=0.0,
        time_stop=0.1,
        baseline=baseline,
        step_hamiltonian=step_hamiltonian,
        prune_candidates=rows,
    )

    assert error is None
    assert str(action_kind) == "prune_coordinate"
    assert selected is not None
    assert str(selected["candidate_label"]) == str(y_label)
    assert str(evaluated[0]["origin_kind"]) == "append"


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
    assert progress["summary"] == result.summary
    assert progress["append_count"] == result.summary["append_count"]
    assert progress["trajectory_points"] == len(result.trajectory)
    assert progress["ledger_entries"] == len(result.ledger)
    assert float(progress["latest_fidelity_exact"]) == pytest.approx(float(result.trajectory[-1]["fidelity_exact"]))
    assert float(progress["latest_abs_energy_total_error"]) == pytest.approx(
        float(result.trajectory[-1]["abs_energy_total_error"])
    )
    assert np.isfinite(float(progress["wallclock_elapsed_s"]))
    assert float(progress["wallclock_elapsed_s"]) >= 0.0


def test_realtime_controller_write_progress_respects_throttle_and_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
        progress_path=tmp_path / "controller_progress.json",
        progress_every_s=10.0,
    )
    emitted: list[tuple[str, dict[str, object]]] = []
    perf_counter_vals = iter([1.0, 5.0, 5.5])

    monkeypatch.setattr(controller, "_progress_payload", lambda *, stage, **extra: {"stage": str(stage), **extra})
    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.checkpoint_controller.time.perf_counter",
        lambda: next(perf_counter_vals),
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.checkpoint_controller._realtime_progress.write_json_atomic",
        lambda path, payload: emitted.append((path.name, dict(payload))),
    )

    controller._write_progress(stage="first")
    controller._write_progress(stage="throttled")
    controller._write_progress(stage="forced", force=True)

    assert emitted == [
        ("controller_progress.json", {"stage": "first"}),
        ("controller_progress.json", {"stage": "forced"}),
    ]
    assert controller._last_progress_emit_wallclock == pytest.approx(5.5)


def test_write_json_atomic_replaces_target_via_tmp_file(tmp_path: Path) -> None:
    target_path = tmp_path / "nested" / "progress.json"
    payload = {"stage": "checkpoint_done", "status": "running"}

    write_json_atomic(target_path, payload)

    assert target_path.exists()
    assert not target_path.with_suffix(".json.tmp").exists()
    assert json.loads(target_path.read_text(encoding="utf-8")) == payload


def test_realtime_controller_write_progress_does_not_advance_emit_time_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
        progress_path=tmp_path / "controller_progress.json",
        progress_every_s=0.0,
    )

    monkeypatch.setattr(controller, "_progress_payload", lambda *, stage, **extra: {"stage": str(stage), **extra})
    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.checkpoint_controller.time.perf_counter",
        lambda: 1.0,
    )

    def _raise_write(_path: Path, _payload: dict[str, object]) -> None:
        raise OSError("boom")

    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.checkpoint_controller._realtime_progress.write_json_atomic",
        _raise_write,
    )

    with pytest.raises(OSError, match="boom"):
        controller._write_progress(stage="first_fail", force=True)

    assert controller._last_progress_emit_wallclock is None


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
    expected_partial = {
        "status": "completed",
        "stage": "run_complete",
        "mode": str(controller.cfg.mode),
        "trajectory": [dict(row) for row in result.trajectory],
        "ledger": [dict(row) for row in result.ledger],
        "reference": {
            "controller_state": controller._controller_state_payload(),
        },
        "summary": dict(result.summary),
    }

    assert set(partial["reference"].keys()) == {"controller_state"}
    assert set(partial["reference"]["controller_state"].keys()) == {
        "logical_block_count",
        "runtime_parameter_count",
        "labels",
    }
    assert_json_parity(
        partial,
        expected_partial,
        approx_prefixes=(("trajectory",), ("ledger",)),
    )


def test_realtime_controller_emits_early_stop_progress_and_partial_before_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
        progress_path=tmp_path / "controller_progress.json",
        progress_every_s=0.0,
        partial_payload_path=tmp_path / "controller_partial.json",
    )
    writes: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        controller,
        "_progress_early_stop_reason",
        lambda *, checkpoint_index: "forced_stop" if int(checkpoint_index) >= 0 else None,
    )

    def _capture_write(path: Path, payload: dict[str, object]) -> None:
        writes.append((path.name, json.loads(json.dumps(payload))))

    monkeypatch.setattr(
        "pipelines.time_dynamics.legacy.checkpoint_controller._realtime_progress.write_json_atomic",
        _capture_write,
    )

    result = controller.run()
    progress_payloads = [
        payload
        for name, payload in writes
        if name == "controller_progress.json"
    ]
    partial_payloads = [
        payload
        for name, payload in writes
        if name == "controller_partial.json"
    ]
    progress_stages = [(payload["stage"], payload["status"]) for payload in progress_payloads]
    partial_stages = [(payload["stage"], payload["status"]) for payload in partial_payloads]
    early_stop_partial = next(payload for payload in partial_payloads if payload["stage"] == "early_stop")

    assert result.summary["status"] == "stopped_early"
    assert progress_stages[-2:] == [
        ("early_stop", "stopped_early"),
        ("run_complete", "stopped_early"),
    ]
    assert partial_stages[-2:] == [
        ("early_stop", "stopped_early"),
        ("run_complete", "stopped_early"),
    ]
    assert set(early_stop_partial["reference"].keys()) == {"controller_state"}
    assert set(early_stop_partial["reference"]["controller_state"].keys()) == {
        "logical_block_count",
        "runtime_parameter_count",
        "labels",
    }
    expected_summary_keys = {
        "append_count",
        "prune_count",
        "repair_count",
        "repair_retry_attempt_count",
        "repair_retry_exhausted_count",
        "stay_count",
        "high_miss_no_admit_soft_fallback_count",
        "high_miss_no_admit_soft_fallback_warning_count",
        "ordinary_stay_count",
        "high_miss_no_admit_soft_fallback_reason_counts",
        "executed_decision_backends",
        "final_logical_block_count",
        "final_runtime_parameter_count",
    }
    assert expected_summary_keys <= set(early_stop_partial["summary"].keys())
    assert "high_miss_no_admit_count" in early_stop_partial["summary"]
    assert "first_bad_high_miss_no_admit_checkpoint_diagnostic" in early_stop_partial["summary"]
    assert early_stop_partial["summary"]["append_count"] == result.summary["append_count"]
    assert early_stop_partial["summary"]["prune_count"] == result.summary["prune_count"]
    assert early_stop_partial["summary"]["repair_count"] == result.summary["repair_count"]
    assert early_stop_partial["summary"]["stay_count"] == result.summary["stay_count"]
    assert early_stop_partial["summary"]["high_miss_no_admit_soft_fallback_count"] == 0
    assert early_stop_partial["summary"]["ordinary_stay_count"] == result.summary["stay_count"]


def test_realtime_controller_default_high_miss_no_admit_soft_fallback_advances_physical_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            miss_abs_threshold=0.0,
            miss_persistence_window=1,
            miss_persistence_count=1,
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
    original_baseline_geometry = controller._baseline_geometry

    def _baseline_with_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_with_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", lambda **kwargs: [])

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["high_miss_no_admit_policy"] == "bounded_stay_advance"
    assert result.summary["repair_count"] == 0
    assert result.summary["repair_event_row_count"] == 0
    assert result.summary["full_horizon_gate_passed"] is True
    assert result.summary["full_horizon_gate_reason"] == "passed"
    assert result.summary["high_miss_count"] >= 1
    assert result.summary["high_miss_no_admit_count"] >= 1
    assert result.summary["first_bad_high_miss_no_admit_checkpoint_diagnostic"] is not None
    assert result.summary["high_miss_no_admit_soft_fallback_count"] >= 1
    assert result.summary["high_miss_no_admit_soft_fallback_warning_count"] == result.summary[
        "high_miss_no_admit_soft_fallback_count"
    ]
    row = next(row for row in result.trajectory if row["high_miss_no_admit_soft_fallback"])
    assert row["action_kind"] == "stay"
    assert row["trajectory_sample_kind"] == "state_sample"
    assert row["advances_time"] is True
    assert row["repair_retry_next"] is False
    assert row["repair_terminal"] is False
    assert row["repair_failure_reason"] is None
    assert row["high_miss_no_admit_soft_fallback_policy"] == "bounded_stay_advance"
    assert row["high_miss_no_admit_soft_fallback_reason"] == HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON
    assert row["high_miss_no_admit_soft_fallback_warning"] == HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING
    diag = row["repair_no_admit_diagnostics"]
    assert diag["strict_no_admit_reason"] == "no_confirmed_candidates"
    assert diag["no_admit_resolution"] == "bounded_stay_advance"
    assert diag["no_admit_resolution_advances_time"] is True
    assert diag["high_miss_no_admit_soft_fallback"] is True
    assert diag["high_miss_no_admit_soft_fallback_reason"] == HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON
    ledger_row = next(row for row in result.ledger if row["high_miss_no_admit_soft_fallback"])
    assert ledger_row["trajectory_sample_kind"] == "state_sample"
    assert ledger_row["advances_time"] is True


def test_realtime_controller_repair_stop_on_high_miss_no_admit_does_not_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_stop",
            miss_abs_threshold=0.0,
            miss_persistence_window=1,
            miss_persistence_count=1,
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
    initial_theta = np.asarray(controller.current_theta, dtype=float).copy()
    original_baseline_geometry = controller._baseline_geometry

    def _baseline_with_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_with_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", lambda **kwargs: [])

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_required_high_miss_no_admit"
    assert result.summary["full_horizon_gate_passed"] is False
    assert result.summary["full_horizon_gate_reason"] == "early_stop:repair_required_high_miss_no_admit"
    assert result.summary["high_miss_count"] == 1
    assert result.summary["high_miss_no_admit_count"] == 1
    assert result.summary["high_miss_no_admit_reason_counts"] == {"no_confirmed_candidates": 1}
    assert result.summary["first_bad_high_miss_no_admit_checkpoint_diagnostic"]["checkpoint_index"] == 0
    assert result.summary["repair_count"] == 1
    assert result.summary["append_count"] == 0
    assert result.summary["stay_count"] == 0
    assert result.summary["high_miss_no_admit_policy"] == "repair_stop"
    assert len(result.trajectory) == 1
    assert str(result.trajectory[0]["action_kind"]) == "repair_miss"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "stay"
    assert str(result.trajectory[0]["controller_lane"]) == "append"
    assert str(result.trajectory[0]["decision_override_reason"]) == "repair_required_high_miss_no_admit"
    diag = result.trajectory[0]["repair_no_admit_diagnostics"]
    assert diag["controller_lane"] == "append"
    assert diag["scout_candidate_count"] == 0
    assert diag["confirmed_candidate_count"] == 0
    assert diag["admissible_candidate_count"] == 0
    assert diag["strict_no_admit_reason"] == "no_confirmed_candidates"
    assert diag["high_miss_no_admit_policy"] == "repair_stop"
    assert diag["no_admit_resolution"] == "repair_stop_terminal"
    assert diag["no_admit_resolution_advances_time"] is False
    assert diag["high_miss_no_admit_soft_fallback"] is False
    assert diag["thresholds"]["miss_threshold"] == pytest.approx(0.05)
    assert diag["thresholds"]["miss_abs_threshold"] == pytest.approx(0.0)
    assert result.ledger[0]["repair_no_admit_diagnostics"] == diag
    assert result.trajectory[0]["repair_rescue_admitted"] is False
    assert float(result.trajectory[0]["rho_real"]) == pytest.approx(0.6)
    assert float(result.trajectory[0]["rho_num"]) == pytest.approx(0.1)
    assert str(result.ledger[0]["action_kind"]) == "repair_miss"
    assert str(result.ledger[0]["tier_reached"]) == "repair"
    assert result.ledger[0]["high_miss_no_admit_soft_fallback"] is False
    assert result.summary["high_miss_no_admit_soft_fallback_count"] == 0
    assert result.summary["ordinary_stay_count"] == 0
    assert float(result.ledger[0]["rho_real"]) == pytest.approx(0.6)
    assert float(result.ledger[0]["rho_num"]) == pytest.approx(0.1)
    np.testing.assert_allclose(controller.current_theta, initial_theta)


def test_realtime_controller_repair_stop_on_append_like_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_stop",
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
    original_baseline_geometry = controller._baseline_geometry

    def _baseline_with_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        return baseline

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_with_high_miss)
    monkeypatch.setattr(
        controller,
        "_select_action_exact_v1",
        lambda **kwargs: ("append_candidate", None),
    )

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_required_high_miss_no_admit"
    assert str(result.trajectory[0]["action_kind"]) == "repair_miss"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "append_candidate"
    assert result.trajectory[0]["proposed_candidate_label"] is None
    assert str(result.ledger[0]["action_kind"]) == "repair_miss"


def test_progress_metrics_ignore_repair_only_rows_without_raw_fallback() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            progress_observable_window=7,
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
    repair_row = {
        "trajectory_sample_kind": "repair_event",
        "advances_time": False,
        "fidelity_exact": 0.123,
        "abs_energy_total_error": 4.56,
        "site_occupations_abs_error_max": 7.89,
        "abs_primary_density_error": 1.23,
    }
    controller._trajectory = [dict(repair_row)]

    assert physical_trajectory_rows([repair_row]) == [repair_row]
    assert physical_trajectory_rows([repair_row], fallback_to_raw=False) == []
    metrics = controller._progress_observable_metrics()

    assert metrics["latest_fidelity_exact"] is None
    assert metrics["latest_abs_energy_total_error"] is None
    assert metrics["latest_site_occupations_abs_error_max"] is None
    assert metrics["latest_abs_primary_density_error"] is None
    assert metrics["progress_observable_window"] == 7
    assert metrics["rolling_fidelity_exact_mean"] is None
    assert metrics["rolling_abs_energy_total_error_mean"] is None
    assert metrics["rolling_site_occupations_abs_error_max_mean"] is None
    assert metrics["rolling_abs_primary_density_error_mean"] is None



def test_realtime_controller_repair_retry_success_retries_same_checkpoint_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=2,
            miss_persistence_window=4,
            miss_persistence_count=2,
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
        num_times=2,
    )
    original_baseline_geometry = controller._baseline_geometry
    geometry_memos: list[object] = []

    def _baseline_first_attempt_high_miss(*args, **kwargs):
        checkpoint_ctx = args[0] if args else kwargs["checkpoint_ctx"]
        geometry_memo = args[2] if len(args) >= 3 else kwargs["geometry_memo"]
        geometry_memos.append(geometry_memo)
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        if (
            int(getattr(checkpoint_ctx, "checkpoint_index")) == 0
            and int(controller._repair_attempt_state.attempt_index) == 0
        ):
            baseline["summary"] = dataclass_replace(
                baseline["summary"],
                epsilon_proj_sq=1.0,
                rho_miss=0.5,
                rho_real=0.6,
                rho_num=0.1,
            )
            baseline["rho_miss"] = 0.5
            baseline["rho_real"] = 0.6
            baseline["rho_num"] = 0.1
        else:
            baseline["summary"] = dataclass_replace(
                baseline["summary"],
                epsilon_proj_sq=0.0,
                rho_miss=0.0,
                rho_real=0.0,
                rho_num=0.0,
            )
            baseline["rho_miss"] = 0.0
            baseline["rho_real"] = 0.0
            baseline["rho_num"] = 0.0
        return baseline

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_first_attempt_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", lambda **kwargs: [])

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["high_miss_no_admit_policy"] == "repair_retry"
    assert result.summary["repair_retry_attempt_count"] == 1
    assert result.summary["repair_retry_exhausted_count"] == 0
    assert result.summary["repair_event_row_count"] == 1
    assert result.summary["trajectory_state_sample_count"] == 2
    assert str(result.trajectory[0]["action_kind"]) == "repair_miss"
    assert result.trajectory[0]["trajectory_sample_kind"] == "repair_event"
    assert result.trajectory[0]["advances_time"] is False
    assert result.trajectory[0]["repair_retry_next"] is True
    assert result.trajectory[0]["repair_attempt_index"] == 0
    assert result.trajectory[1]["checkpoint_index"] == 0
    assert result.trajectory[1]["trajectory_sample_kind"] == "state_sample"
    assert result.trajectory[1]["advances_time"] is True
    assert result.trajectory[1]["accepted_after_repair"] is True
    assert result.trajectory[1]["repair_attempt_index"] == 1
    assert geometry_memos[0] is not geometry_memos[1]
    assert controller._high_miss_history
    assert all(bool(item) is False for item in controller._high_miss_history)
    assert all(bool(item) is False for item in controller._high_miss_relative_history)


def test_realtime_controller_repair_retry_benchmark_exact_observer_keeps_physical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="benchmark_exact",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=1,
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
        num_times=2,
    )
    original_baseline_geometry = controller._baseline_geometry

    def _baseline_first_attempt_high_miss(*args, **kwargs):
        checkpoint_ctx = args[0] if args else kwargs["checkpoint_ctx"]
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        if (
            int(getattr(checkpoint_ctx, "checkpoint_index")) == 0
            and int(controller._repair_attempt_state.attempt_index) == 0
        ):
            baseline["summary"] = dataclass_replace(
                baseline["summary"],
                epsilon_proj_sq=1.0,
                rho_miss=0.5,
                rho_real=0.6,
                rho_num=0.1,
            )
            baseline["rho_miss"] = 0.5
            baseline["rho_real"] = 0.6
            baseline["rho_num"] = 0.1
        else:
            baseline["summary"] = dataclass_replace(
                baseline["summary"],
                epsilon_proj_sq=0.0,
                rho_miss=0.0,
                rho_real=0.0,
                rho_num=0.0,
            )
            baseline["rho_miss"] = 0.0
            baseline["rho_real"] = 0.0
            baseline["rho_num"] = 0.0
        return baseline

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_first_attempt_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", lambda **kwargs: [])

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["reference_mode"] == "benchmark_exact"
    assert result.summary["reference_enabled"] is True
    assert result.summary["repair_event_row_count"] == 1
    assert result.summary["trajectory_state_sample_count"] == 2
    assert str(result.trajectory[0]["trajectory_sample_kind"]) == "repair_event"
    assert result.trajectory[0]["advances_time"] is False
    assert str(result.trajectory[1]["trajectory_sample_kind"]) == "state_sample"
    assert result.trajectory[1]["advances_time"] is True
    assert result.trajectory[1]["accepted_after_repair"] is True
    assert result.trajectory[1]["fidelity_exact"] is not None
    physical_rows = [
        row
        for row in result.trajectory
        if str(row.get("trajectory_sample_kind", "state_sample")) != "repair_event"
        and row.get("advances_time", True) is not False
    ]
    assert [int(row["checkpoint_index"]) for row in physical_rows] == [0, 1]



def test_realtime_controller_repair_retry_success_can_append_once_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.0,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=2,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            append_no_harm_guard_enabled=False,
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
    initial_logical_count = int(controller.current_layout.logical_parameter_count)
    initial_theta = np.asarray(controller.current_theta, dtype=float).copy()
    original_scout_candidates = controller._scout_candidates
    geometry_memos: list[object] = []

    def _scout_empty_then_real(**kwargs):
        checkpoint_ctx = kwargs["checkpoint_ctx"]
        geometry_memos.append(kwargs["geometry_memo"])
        if (
            int(getattr(checkpoint_ctx, "checkpoint_index")) == 0
            and int(controller._repair_attempt_state.attempt_index) == 0
        ):
            return []
        return original_scout_candidates(**kwargs)

    monkeypatch.setattr(controller, "_scout_candidates", _scout_empty_then_real)
    monkeypatch.setattr(controller, "_exact_v1_forecast_override_reason", lambda **kwargs: None)

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["repair_retry_attempt_count"] == 1
    assert result.summary["append_count"] >= 1
    assert str(result.trajectory[0]["action_kind"]) == "repair_miss"
    assert result.trajectory[0]["checkpoint_index"] == 0
    assert result.trajectory[0]["advances_time"] is False
    assert result.trajectory[0]["logical_block_count"] == initial_logical_count
    assert result.ledger[0]["logical_block_count_before"] == initial_logical_count
    assert result.ledger[0]["logical_block_count_after"] == initial_logical_count
    assert str(result.trajectory[1]["action_kind"]) == "append_candidate"
    assert result.trajectory[1]["checkpoint_index"] == 0
    assert result.trajectory[1]["advances_time"] is True
    assert result.trajectory[1]["accepted_after_repair"] is True
    assert result.trajectory[1]["repair_attempt_index"] == 1
    assert result.trajectory[1]["logical_block_count"] == initial_logical_count
    assert result.ledger[1]["logical_block_count_before"] == initial_logical_count
    assert result.ledger[1]["logical_block_count_after"] == initial_logical_count + 1
    assert int(controller.current_layout.logical_parameter_count) >= initial_logical_count + 1
    assert np.asarray(controller.current_theta, dtype=float).size > int(initial_theta.size)
    assert geometry_memos[0] is not geometry_memos[1]



def test_realtime_controller_repair_retry_exhaustion_no_hidden_stay_and_escalates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=2,
            gain_ratio_threshold=1e-9,
            append_margin_abs=1e-12,
            shortlist_size=4,
            shortlist_fraction=0.15,
            max_probe_positions=4,
            regularization_lambda=1e-8,
            candidate_regularization_lambda=1e-8,
            pinv_rcond=1e-10,
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
    initial_theta = np.asarray(controller.current_theta, dtype=float).copy()
    original_baseline_geometry = controller._baseline_geometry
    seen_effective: list[tuple[int, int, float, int, float, float, float]] = []
    geometry_memos: list[object] = []

    def _baseline_always_high_miss(*args, **kwargs):
        geometry_memo = args[2] if len(args) >= 3 else kwargs["geometry_memo"]
        geometry_memos.append(geometry_memo)
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    def _no_candidates_with_effective_cfg(**kwargs):
        del kwargs
        cfg = controller._active_cfg()
        seen_effective.append(
            (
                int(controller._repair_attempt_state.attempt_index),
                int(cfg.shortlist_size),
                float(cfg.shortlist_fraction),
                int(cfg.max_probe_positions),
                float(cfg.regularization_lambda),
                float(cfg.candidate_regularization_lambda),
                float(cfg.pinv_rcond),
            )
        )
        return []

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_always_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", _no_candidates_with_effective_cfg)

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert result.summary["repair_count"] == 3
    assert result.summary["repair_retry_attempt_count"] == 3
    assert result.summary["repair_retry_exhausted_count"] == 1
    assert result.summary["repair_retry_admission_policy"] == "strict"
    assert result.summary["repair_rescue_admitted_count"] == 0
    assert result.summary["append_count"] == 0
    assert result.summary["stay_count"] == 0
    assert [row["repair_attempt_index"] for row in result.trajectory] == [0, 1, 2]
    assert [row["advances_time"] for row in result.trajectory] == [False, False, False]
    assert [row["repair_retry_next"] for row in result.trajectory] == [True, True, False]
    assert result.trajectory[-1]["repair_terminal"] is True
    assert result.trajectory[-1]["repair_failure_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert str(result.trajectory[-1]["action_kind"]) == "repair_miss"
    assert all(row["repair_no_admit_diagnostics"] is not None for row in result.trajectory)
    assert all(str(row["action_kind"]) != "stay" for row in result.trajectory)
    assert seen_effective[0] == (0, 4, pytest.approx(0.15), 4, pytest.approx(1e-8), pytest.approx(1e-8), pytest.approx(1e-10))
    assert seen_effective[1] == (1, 8, pytest.approx(0.30), 2, pytest.approx(1e-8), pytest.approx(1e-8), pytest.approx(1e-10))
    assert seen_effective[2] == (2, 16, pytest.approx(0.50), 2, pytest.approx(1e-7), pytest.approx(1e-7), pytest.approx(1e-9))
    assert geometry_memos[0] is not geometry_memos[1]
    assert geometry_memos[1] is not geometry_memos[2]
    assert geometry_memos[0] is not geometry_memos[2]
    np.testing.assert_allclose(controller.current_theta, initial_theta)


def test_realtime_controller_repair_retry_rescue_terminal_confirmed_candidate_appends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=1,
            repair_retry_admission_policy="rescue_best_confirmed_append_v1",
            repair_retry_rescue_min_gain_ratio=0.0,
            gain_ratio_threshold=10.0,
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
    initial_logical_count = int(controller.current_layout.logical_parameter_count)
    original_baseline_geometry = controller._baseline_geometry
    original_scout_candidates = controller._scout_candidates

    def _baseline_always_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    def _scout_empty_then_real(**kwargs):
        if int(controller._repair_attempt_state.attempt_index) == 0:
            return []
        return original_scout_candidates(**kwargs)

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_always_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", _scout_empty_then_real)
    monkeypatch.setattr(controller, "_local_forecast_override_reason", lambda **kwargs: None)

    result = controller.run()

    assert result.summary["status"] == "completed"
    assert result.summary["append_count"] == 1
    assert result.summary["repair_retry_attempt_count"] == 1
    assert result.summary["repair_rescue_admitted_count"] == 1
    assert result.summary["repair_retry_admission_policy"] == "rescue_best_confirmed_append_v1"
    assert result.trajectory[0]["action_kind"] == "repair_miss"
    assert result.trajectory[0]["repair_retry_next"] is True
    rescue_row = result.trajectory[1]
    assert rescue_row["checkpoint_index"] == 0
    assert rescue_row["action_kind"] == "append_candidate"
    assert rescue_row["proposed_action_kind"] == "stay"
    assert rescue_row["accepted_after_repair"] is True
    assert rescue_row["repair_rescue_admitted"] is True
    assert rescue_row["repair_rescue_candidate_label"] == rescue_row["candidate_label"]
    assert rescue_row["repair_rescue_reason"] == "repair_retry_rescue_best_confirmed_append_v1"
    assert rescue_row["repair_no_admit_diagnostics"]["confirmed_candidate_count"] >= 1
    assert rescue_row["repair_no_admit_diagnostics"]["admissible_candidate_count"] == 0
    assert result.ledger[1]["repair_rescue_admitted"] is True
    assert int(controller.current_layout.logical_parameter_count) == initial_logical_count + 1
    assert all(
        str(row["action_kind"]) != "stay"
        for row in result.trajectory
        if row.get("repair_no_admit_diagnostics") is not None
    )


def test_realtime_controller_repair_retry_forecast_vetoed_strict_append_repairs_not_stays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=0,
            repair_retry_admission_policy="rescue_best_confirmed_append_v1",
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
    original_baseline_geometry = controller._baseline_geometry

    def _baseline_with_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    def _select_first_confirmed(**kwargs):
        confirmed = list(kwargs["confirmed"])
        assert confirmed
        return "append_candidate", confirmed[0]

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_with_high_miss)
    monkeypatch.setattr(controller, "_select_action_exact_v1", _select_first_confirmed)
    monkeypatch.setattr(
        controller,
        "_local_forecast_override_reason",
        lambda **kwargs: "local_forecast_no_advantage",
    )

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert result.summary["append_count"] == 0
    assert result.summary["stay_count"] == 0
    assert result.summary["repair_rescue_admitted_count"] == 0
    assert len(result.trajectory) == 1
    row = result.trajectory[0]
    assert row["action_kind"] == "repair_miss"
    assert row["proposed_action_kind"] == "append_candidate"
    assert row["proposed_candidate_label"] is not None
    assert row["repair_rescue_admitted"] is False
    assert row["repair_rescue_reason"] == "strict_selection_admitted"
    assert row["repair_no_admit_diagnostics"]["forecast_veto_reason"] == "local_forecast_no_advantage"
    assert row["repair_no_admit_diagnostics"]["strict_no_admit_reason"] == "local_forecast_no_advantage"


def test_realtime_controller_repair_retry_rescue_no_candidate_terminal_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=0,
            repair_retry_admission_policy="rescue_best_confirmed_append_v1",
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
        num_times=2,
    )
    original_baseline_geometry = controller._baseline_geometry

    def _baseline_with_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_with_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", lambda **kwargs: [])

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert result.summary["append_count"] == 0
    assert result.summary["repair_rescue_admitted_count"] == 0
    assert len(result.trajectory) == 1
    row = result.trajectory[0]
    assert row["action_kind"] == "repair_miss"
    assert row["repair_rescue_admitted"] is False
    assert row["repair_rescue_reason"] == "no_confirmed_candidates"
    assert row["repair_no_admit_diagnostics"]["confirmed_candidate_count"] == 0
    assert row["repair_no_admit_diagnostics"]["scout_candidate_count"] == 0
    assert all(str(item["action_kind"]) != "stay" for item in result.trajectory)


def test_realtime_controller_repair_retry_rescue_forecast_veto_remains_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=1,
            repair_retry_admission_policy="rescue_best_confirmed_append_v1",
            repair_retry_rescue_min_gain_ratio=0.0,
            gain_ratio_threshold=10.0,
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
    original_baseline_geometry = controller._baseline_geometry
    original_scout_candidates = controller._scout_candidates

    def _baseline_always_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    def _scout_empty_then_real(**kwargs):
        if int(controller._repair_attempt_state.attempt_index) == 0:
            return []
        return original_scout_candidates(**kwargs)

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_always_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", _scout_empty_then_real)
    monkeypatch.setattr(
        controller,
        "_local_forecast_override_reason",
        lambda **kwargs: "local_forecast_no_advantage",
    )

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert result.summary["append_count"] == 0
    assert result.summary["repair_rescue_admitted_count"] == 0
    assert [row["action_kind"] for row in result.trajectory] == ["repair_miss", "repair_miss"]
    terminal_row = result.trajectory[-1]
    assert terminal_row["repair_rescue_admitted"] is False
    assert terminal_row["repair_rescue_reason"] == "local_forecast_no_advantage"
    assert terminal_row["repair_no_admit_diagnostics"]["forecast_veto_reason"] == "local_forecast_no_advantage"
    assert terminal_row["repair_no_admit_diagnostics"]["confirmed_candidate_count"] >= 1
    assert all(str(row["action_kind"]) != "stay" for row in result.trajectory)


def test_realtime_controller_repair_retry_rescue_exact_v1_forecast_veto_remains_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=1,
            repair_retry_admission_policy="rescue_best_confirmed_append_v1",
            repair_retry_rescue_min_gain_ratio=0.0,
            gain_ratio_threshold=10.0,
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
    original_baseline_geometry = controller._baseline_geometry
    original_scout_candidates = controller._scout_candidates

    def _baseline_always_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    def _scout_empty_then_real(**kwargs):
        if int(controller._repair_attempt_state.attempt_index) == 0:
            return []
        return original_scout_candidates(**kwargs)

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_always_high_miss)
    monkeypatch.setattr(controller, "_scout_candidates", _scout_empty_then_real)
    monkeypatch.setattr(controller, "_local_forecast_override_reason", lambda **kwargs: None)
    monkeypatch.setattr(
        controller,
        "_exact_v1_forecast_override_reason",
        lambda **kwargs: "exact_forecast_nonimproving_tracking_score",
    )

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert result.summary["append_count"] == 0
    assert result.summary["stay_count"] == 0
    assert result.summary["repair_rescue_admitted_count"] == 0
    assert [row["action_kind"] for row in result.trajectory] == ["repair_miss", "repair_miss"]
    terminal_row = result.trajectory[-1]
    assert terminal_row["action_kind"] == "repair_miss"
    assert terminal_row["proposed_action_kind"] == "stay"
    assert terminal_row["repair_terminal"] is True
    assert terminal_row["repair_rescue_admitted"] is False
    assert terminal_row["repair_rescue_reason"] == "exact_forecast_nonimproving_tracking_score"
    assert (
        terminal_row["repair_no_admit_diagnostics"]["forecast_veto_reason"]
        == "exact_forecast_nonimproving_tracking_score"
    )
    assert terminal_row["repair_no_admit_diagnostics"]["confirmed_candidate_count"] >= 1
    assert all(str(row["action_kind"]) != "stay" for row in result.trajectory)


def test_realtime_controller_repair_retry_max_zero_terminal_no_hidden_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=0.05,
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=0,
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
    original_baseline_geometry = controller._baseline_geometry

    def _baseline_with_high_miss(*args, **kwargs):
        baseline = dict(original_baseline_geometry(*args, **kwargs))
        baseline["summary"] = dataclass_replace(
            baseline["summary"],
            epsilon_proj_sq=1.0,
            rho_miss=0.5,
            rho_real=0.6,
            rho_num=0.1,
        )
        baseline["rho_miss"] = 0.5
        baseline["rho_real"] = 0.6
        baseline["rho_num"] = 0.1
        return baseline

    monkeypatch.setattr(controller, "_baseline_geometry", _baseline_with_high_miss)
    monkeypatch.setattr(
        controller,
        "_select_action_exact_v1",
        lambda **kwargs: ("append_candidate", None),
    )

    result = controller.run()

    assert result.summary["status"] == "stopped_early"
    assert result.summary["early_stop_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert len(result.trajectory) == 1
    assert result.trajectory[0]["action_kind"] == "repair_miss"
    assert result.trajectory[0]["proposed_action_kind"] == "append_candidate"
    assert result.trajectory[0]["repair_retry_next"] is False
    assert result.trajectory[0]["repair_terminal"] is True
    assert result.trajectory[0]["advances_time"] is False


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


def test_exact_forecast_override_reason_rejects_d_shape_barrier_energy_regression() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
            "tracking_total_occupation_abs_error_next": 0.01,
            "tracking_total_occupation_abs_error_mean": 0.02,
        },
        selected_forecast={
            "fidelity_exact_next": 0.795,
            "abs_energy_total_error_next": 0.14,
            "tracking_total_occupation_abs_error_next": 0.01,
            "tracking_total_occupation_abs_error_mean": 0.02,
        },
    )

    assert reason == "exact_forecast_d_shape_energy_regression"


def test_exact_forecast_override_reason_rejects_d_shape_barrier_total_occupation_regression() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
            "tracking_total_occupation_abs_error_next": 0.01,
            "tracking_total_occupation_abs_error_mean": 0.02,
        },
        selected_forecast={
            "fidelity_exact_next": 0.795,
            "abs_energy_total_error_next": 0.11,
            "tracking_total_occupation_abs_error_next": 0.03,
            "tracking_total_occupation_abs_error_mean": 0.025,
        },
    )

    assert reason == "exact_forecast_d_shape_total_occupation_regression"


def test_exact_forecast_override_reason_allows_d_shape_barrier_candidate_within_guardrails() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="oracle_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
            "tracking_total_occupation_abs_error_next": 0.01,
            "tracking_total_occupation_abs_error_mean": 0.02,
        },
        selected_forecast={
            "fidelity_exact_next": 0.795,
            "abs_energy_total_error_next": 0.11,
            "tracking_total_occupation_abs_error_next": 0.015,
            "tracking_total_occupation_abs_error_mean": 0.021,
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


def test_candidate_pool_terms_reopens_repeats_only_when_unique_pool_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _exhausted_repeat_label_context()
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_repeat_reopen_mode="sign_reversal_window",
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2, 0.1],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )
    assert controller._candidate_pool_terms() == []
    monkeypatch.setattr(
        controller,
        "_exact_v1_sign_reversal_repeat_reopen_active",
        lambda **kwargs: True,
    )
    reopened = controller._candidate_pool_terms(
        baseline={"psi": psi_initial},
        time_start=0.0,
        time_stop=0.1,
    )
    assert [int(pool_index) for pool_index, _ in reopened] == [0, 1, 2]


def test_candidate_pool_terms_does_not_reopen_repeats_when_unique_terms_still_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _duplicate_label_context()
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_repeat_reopen_mode="sign_reversal_window",
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )
    monkeypatch.setattr(
        controller,
        "_exact_v1_sign_reversal_repeat_reopen_active",
        lambda **kwargs: True,
    )
    reopened = controller._candidate_pool_terms(
        baseline={"psi": psi_initial},
        time_start=0.0,
        time_stop=0.1,
    )
    assert [int(pool_index) for pool_index, _ in reopened] == [1, 2]


def test_candidate_pool_terms_uses_append_family_and_suppresses_current_source_label() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    x_term, y_term = replay_context.family_pool
    z_term = AnsatzTerm(
        label="op_z_extra",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
    )
    replay_context = dataclass_replace(
        replay_context,
        append_family_info={
            "requested": "full_meta",
            "resolved": "full_meta",
            "resolution_source": "cli.append_pool_family",
            "fallback_used": False,
            "uses_replay_pool": False,
        },
        append_family_pool=(x_term, y_term, z_term),
        append_pool_meta={
            "family": "full_meta",
            "candidate_pool_complete": True,
            "append_pool_source": "append_pool_family",
        },
        append_family_terms_count=3,
    )
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

    available = controller._candidate_pool_terms()
    assert [term.label for _, term in available] == ["op_y", "op_z_extra"]
    diag = controller._last_candidate_pool_diagnostics
    assert diag["append_family_resolved"] == "full_meta"
    assert diag["family_pool_sizes"]["append_family_pool_count"] == 3
    assert diag["family_pool_sizes"]["repeated_suppressed_count"] == 1

    y_carrier = _build_candidate_carrier(
        y_term,
        logical_index=1,
        unique_label="op_y__pool1__append0_p1",
        template_layout=controller.current_layout,
        candidate_pool_index=1,
    )
    controller.current_terms = [*controller.current_terms, y_carrier]
    available_after_append = controller._candidate_pool_terms()
    assert [term.label for _, term in available_after_append] == ["op_z_extra"]
    assert controller._last_candidate_pool_diagnostics["family_pool_sizes"]["repeated_suppressed_count"] == 2


def test_explicit_empty_complete_append_pool_does_not_fallback_to_replay_pool() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    replay_context = dataclass_replace(
        replay_context,
        append_family_info={
            "requested": "empty_debug_family",
            "resolved": "empty_debug_family",
            "resolution_source": "cli.append_pool_family",
            "fallback_used": False,
            "uses_replay_pool": False,
        },
        append_family_pool=(),
        append_pool_meta={
            "family": "empty_debug_family",
            "candidate_pool_complete": True,
            "append_pool_source": "append_pool_family",
        },
        append_family_terms_count=0,
    )
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

    assert controller._candidate_pool_terms() == []
    diag = controller._last_candidate_pool_diagnostics
    assert diag["append_family_resolved"] == "empty_debug_family"
    assert diag["family_pool_sizes"]["append_family_pool_count"] == 0
    assert diag["family_pool_sizes"]["available_candidate_count"] == 0


def test_controller_rejects_incomplete_append_pool() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    replay_context = dataclass_replace(
        replay_context,
        append_family_info={"requested": "full_meta", "resolved": "full_meta"},
        append_family_pool=(),
        append_pool_meta={
            "family": "full_meta",
            "candidate_pool_complete": False,
            "append_pool_source": "explicit_family_incomplete",
            "incomplete_reason": "full_meta_append_pool_incomplete_for_n_ph_max_ge_2",
        },
        append_family_terms_count=0,
    )
    with pytest.raises(ValueError, match="complete append candidate family pool"):
        RealtimeCheckpointController(
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


def test_confirm_score_payload_raw_mode_exposes_penalty_breakdown() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            confirm_score_mode="exact_gain_ratio",
            gain_ratio_threshold=0.01,
            append_margin_abs=0.01,
            directional_penalty_weight=3.0,
            measurement_penalty_weight=2.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )

    payload = controller._confirm_score_payload(
        baseline={"norm_b_sq": 1.0, "K": np.eye(1), "f": np.zeros(1)},
        B=np.zeros((1, 1)),
        C=np.eye(1),
        q=np.zeros(1),
        w=np.zeros(1),
        gain_ratio=0.5,
        gain_exact=0.4,
        groups_new=0.1,
        directional_change_l2=0.05,
    )

    assert payload["confirm_score_kind"] == "geometry_gain_ratio_minus_penalties"
    assert payload["confirm_gain_ratio_raw"] == pytest.approx(0.5)
    assert payload["confirm_gain_exact_raw"] == pytest.approx(0.4)
    assert payload["confirm_directional_penalty_value"] == pytest.approx(0.15)
    assert payload["confirm_measurement_penalty_value"] == pytest.approx(0.2)
    assert payload["confirm_score"] == pytest.approx(0.15)
    assert payload["confirm_gate_passed"] is True
    assert payload["confirm_gate_reason"] is None


def test_confirm_score_payload_penalty_dominance_fails_score_gate() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            confirm_score_mode="exact_gain_ratio",
            gain_ratio_threshold=0.01,
            append_margin_abs=0.01,
            measurement_penalty_weight=2.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )

    payload = controller._confirm_score_payload(
        baseline={"norm_b_sq": 1.0, "K": np.eye(1), "f": np.zeros(1)},
        B=np.zeros((1, 1)),
        C=np.eye(1),
        q=np.zeros(1),
        w=np.zeros(1),
        gain_ratio=0.5,
        gain_exact=0.4,
        groups_new=1.0,
        directional_change_l2=0.0,
    )

    assert payload["confirm_score"] == pytest.approx(-1.5)
    assert payload["confirm_gain_ratio_gate"] is True
    assert payload["confirm_gain_exact_gate"] is True
    assert payload["confirm_score_gate"] is False
    assert payload["confirm_gate_reason"] == "confirm_score_below_threshold"


def test_confirm_score_payload_compressed_mode_exposes_lower_gain_breakdown() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            confirm_score_mode="compressed_whitened_v1",
            gain_ratio_threshold=0.0,
            append_margin_abs=0.0,
            directional_penalty_weight=0.2,
            measurement_penalty_weight=0.1,
            candidate_regularization_lambda=0.0,
            pinv_rcond=1.0e-12,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )

    payload = controller._confirm_score_payload(
        baseline={"norm_b_sq": 1.0, "K": np.eye(1), "f": np.zeros(1)},
        B=np.zeros((1, 1)),
        C=np.eye(1),
        q=np.array([0.5]),
        w=np.array([0.5]),
        gain_ratio=0.8,
        gain_exact=0.8,
        groups_new=1.0,
        directional_change_l2=0.5,
    )

    assert payload["confirm_score_kind"] == "compressed_whitened_lower_gain_ratio_minus_penalties"
    assert payload["confirm_compressed_gain_exact"] == pytest.approx(0.25)
    assert payload["confirm_compressed_gain_ratio"] == pytest.approx(0.25)
    assert payload["confirm_directional_penalty_value"] == pytest.approx(0.1)
    assert payload["confirm_measurement_penalty_value"] == pytest.approx(0.1)
    assert payload["confirm_score"] == pytest.approx(0.05)


def test_local_forecast_scores_are_lower_is_better() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", forecast_accept_margin=0.0),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )

    stay = {"local_projective_score_total": 1.0}
    assert controller._local_forecast_override_reason(
        stay_forecast=stay,
        selected_forecast={"local_projective_score_total": 0.9},
    ) is None
    assert controller._local_forecast_override_reason(
        stay_forecast=stay,
        selected_forecast={"local_projective_score_total": 1.0},
    ) == "local_forecast_no_advantage"


def test_controller_serializes_candidate_forecast_and_prune_diagnostics() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            reference_mode="off",
            miss_threshold=999.0,
            prune_mode="off",
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.05,
        num_times=2,
    )

    result = controller.run()
    row = result.trajectory[0]
    assert row["forecast_score_interpretation"] == "lower_is_better"
    assert "forecast_selected_lower_than_stay" in row
    assert row["candidate_pool_diagnostics"]["append_family_resolved"] == "toy_pool"
    assert row["candidate_pool_diagnostics"]["family_pool_sizes"]["current_source_label_count"] == 1
    assert "raw_scout_record_count" in row
    assert result.summary["forecast_score_interpretation"] == "lower_is_better"
    assert result.summary["prune_blocker_reason_counts"]["prune_disabled"] >= 1
    assert result.summary["prune_blocker_category_counts"]["disabled"] >= 1


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


def test_oracle_confirm_limit_with_selection_policy_preserves_current_topk_bridge_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    motion = MotionSchedulerTelemetry(
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

    controller_topk = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="oracle_v1", oracle_selection_policy="measured_topk_oracle_energy"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(noise_mode="ideal", oracle_aggregate="mean"),
    )
    monkeypatch.setattr(controller_topk, "_oracle_confirm_limit_for_motion", lambda **kwargs: 1)
    controller_topk._oracle_base_config = SimpleNamespace(noise_mode="shots")
    assert controller_topk._oracle_confirm_limit_with_selection_policy(
        confirmed_count=5,
        refresh_pressure="low",
        motion=motion,
    ) == 3
    controller_topk._oracle_base_config = SimpleNamespace(noise_mode="backend_scheduled")
    assert controller_topk._oracle_confirm_limit_with_selection_policy(
        confirmed_count=5,
        refresh_pressure="low",
        motion=motion,
    ) == 1

    controller_other_policy = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="oracle_v1", oracle_selection_policy="exact_gain_ratio"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
        oracle_base_config=OracleConfig(noise_mode="ideal", oracle_aggregate="mean"),
    )
    monkeypatch.setattr(controller_other_policy, "_oracle_confirm_limit_for_motion", lambda **kwargs: 1)
    controller_other_policy._oracle_base_config = SimpleNamespace(noise_mode="shots")
    assert controller_other_policy._oracle_confirm_limit_with_selection_policy(
        confirmed_count=5,
        refresh_pressure="low",
        motion=motion,
    ) == 1


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


def test_realtime_controller_summary_surfaces_oracle_backend_provenance(
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
            backend_name="FakeMarrakesh",
            use_fake_backend=True,
            shots=32,
            oracle_repeats=1,
        ),
    )

    def _fake_measured_baseline(**kwargs):
        baseline = controller._baseline_geometry(
            kwargs["checkpoint_ctx"],
            kwargs["cache"],
            kwargs["geometry_memo"],
            step_hamiltonian=controller._step_hamiltonian_artifacts(
                float(kwargs["checkpoint_ctx"].time_start)
            ),
        )
        return {
            **baseline,
            "summary": dataclass_replace(
                baseline["summary"],
                energy=float(baseline["summary"].energy) + 0.25,
                solve_mode="grouped_raw_measured",
            ),
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "backend_name": "FakeMarrakesh",
                "estimator_kind": "fake_backend.run(counts)",
                "using_fake_backend": True,
                "details": {
                    "backend_snapshot": {"backend_name": "FakeMarrakesh"},
                    "runtime_profile": {"name": "legacy_runtime_v0"},
                    "runtime_raw_profile": {"name": "legacy_runtime_v0"},
                    "runtime_session_policy": {"mode": "prefer_session"},
                    "raw_transport": "auto",
                    "runtime_job_ids": ["job-1"],
                    "transpile_optimization_level": 2,
                    "transpile_seed": 0,
                    "compiled_depth": 12,
                    "compiled_size": 20,
                    "compiled_count_2q": 5,
                    "compiled_cx_count": 5,
                    "compiled_ecr_count": 0,
                    "compiled_num_qubits": 2,
                    "layout_physical_qubits": [0, 1],
                },
            },
            "observable_estimates": {
                "baseline": {"mean": float(baseline["summary"].energy) + 0.25}
            },
            "raw_group_pool_summary": {"calls": 1},
        }

    monkeypatch.setattr(controller, "_oracle_measured_baseline_geometry", _fake_measured_baseline)

    result = controller.run()

    assert result.summary["oracle_backend_snapshot"] == {"backend_name": "FakeMarrakesh"}
    assert result.summary["oracle_runtime_job_ids"] == ["job-1"]
    assert result.summary["oracle_compile_request"]["transpile_optimization_level"] == 2
    assert result.summary["oracle_compile_observation"]["compiled_count_2q"] == 5


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
            "confirm_score": 1.0,
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
            append_no_harm_guard_enabled=False,
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
            "confirm_score": float(adjusted_gain),
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


def test_realtime_controller_oracle_v1_ignores_exact_benchmark_guardrail_for_live_append(
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

    assert forecast_calls == []
    assert int(result.summary["append_count"]) == 1
    assert int(result.summary["decision_override_count"]) == 0
    assert int(result.summary["exact_forecast_veto_count"]) == 0
    assert str(result.summary["exact_forecast_guardrail_mode"]) == "dual_metric_v1"
    assert str(result.summary["decision_forecast_mode"]) == "local_projective_v1"
    assert str(result.trajectory[0]["action_kind"]) == "append_candidate"
    assert str(result.trajectory[0]["controller_lane"]) == "append"
    assert str(result.trajectory[0]["controller_lane_reason"]) == "exact_rho_miss_above_threshold"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "append_candidate"
    assert str(result.trajectory[0]["proposed_candidate_label"]) == "candidate_a"
    assert result.trajectory[0]["decision_override_reason"] is None
    assert result.trajectory[0]["forecast_stay_fidelity_exact_next"] is None
    assert result.trajectory[0]["forecast_selected_abs_energy_total_error_next"] is None
    assert result.ledger[0]["decision_override_reason"] is None


def test_realtime_controller_oracle_v1_ignores_exact_benchmark_forecast_failures(
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
    assert str(result.summary["decision_forecast_mode"]) == "local_projective_v1"
    assert str(result.trajectory[0]["action_kind"]) == "append_candidate"
    assert result.trajectory[0]["decision_override_reason"] is None
    assert result.trajectory[0]["exact_forecast_error"] is None
    assert result.trajectory[0]["degraded_reason"] is None


def test_realtime_controller_exact_v1_ignores_benchmark_bounded_defect_veto_in_live_mode(
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
            "confirm_score": float(adjusted_gain),
        }

    forecast_calls: list[str] = []

    def _fake_exact_step_forecast(**kwargs):
        forecast_calls.append("call")
        if len(forecast_calls) == 1:
            return {
                "fidelity_exact_next": 0.9995,
                "abs_energy_total_error_next": 1.0e-4,
                "abs_primary_density_error_next": 5.0e-3,
                "abs_primary_density_slope_error_next": 5.0e-3,
                "abs_staggered_error_next": 5.0e-3,
                "abs_doublon_error_next": 5.0e-4,
                "site_occupations_abs_error_max_next": 5.0e-3,
            }
        return {
            "fidelity_exact_next": 0.9997,
            "abs_energy_total_error_next": 1.2e-4,
            "abs_primary_density_error_next": 4.0e-3,
            "abs_primary_density_slope_error_next": 4.0e-3,
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
    monkeypatch.setattr(
        controller,
        "_select_action_exact_v1",
        lambda **kwargs: ("append_candidate", _candidate_record("candidate_a", adjusted_gain=3.0)),
    )
    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    result = controller.run()

    assert forecast_calls == []
    assert int(result.summary["append_count"]) == 1
    assert int(result.summary["decision_override_count"]) == 0
    assert int(result.summary["exact_forecast_veto_count"]) == 0
    assert str(result.summary["decision_forecast_mode"]) == "local_projective_v1"
    assert str(result.trajectory[0]["action_kind"]) == "append_candidate"
    assert str(result.trajectory[0]["controller_lane"]) == "append"
    assert str(result.trajectory[0]["proposed_action_kind"]) == "append_candidate"
    assert result.trajectory[0]["decision_override_reason"] is None
    assert result.trajectory[0]["forecast_stay_site_occupations_abs_error_max_next"] is None
    assert result.ledger[0]["decision_override_reason"] is None


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


def test_exact_v1_forecast_override_reason_bypasses_legacy_stay_bounded_defect_under_d_shape_barrier() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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

    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast={
            "fidelity_exact_next": 0.999,
            "abs_energy_total_error_next": 1.0e-3,
            "abs_primary_density_error_next": 1.0e-3,
            "abs_primary_density_slope_error_next": 1.0e-3,
            "abs_doublon_error_next": 1.0e-4,
            "site_occupations_abs_error_max_next": 1.0e-3,
            "tracking_total_occupation_abs_error_next": 1.0e-3,
            "tracking_total_occupation_abs_error_mean": 1.0e-3,
            "tracking_score_horizon": 0.02,
        },
        selected_forecast={
            "fidelity_exact_next": 0.999,
            "abs_energy_total_error_next": 1.5e-3,
            "abs_primary_density_error_next": 1.0e-3,
            "abs_primary_density_slope_error_next": 1.0e-3,
            "abs_doublon_error_next": 1.0e-4,
            "site_occupations_abs_error_max_next": 1.0e-3,
            "tracking_total_occupation_abs_error_next": 1.0e-3,
            "tracking_total_occupation_abs_error_mean": 1.0e-3,
            "tracking_score_horizon": 0.01,
        },
        action_kind="append_candidate",
        selected={"candidate_label": "candidate_a"},
    )

    assert reason is None


def test_exact_v1_forecast_override_reason_allows_protected_horizon_append_despite_nonimproving_tracking_score() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
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

    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast={
            "fidelity_exact_next": 0.78,
            "abs_energy_total_error_next": 0.18,
            "abs_primary_density_error_next": 0.0114,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0057,
            "tracking_score_horizon": 0.18,
        },
        selected_forecast={
            "fidelity_exact_next": 0.778,
            "abs_energy_total_error_next": 0.18,
            "abs_primary_density_error_next": 0.0113,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0056,
            "tracking_score_horizon": 0.19,
        },
        action_kind="append_candidate",
        selected={
            "candidate_label": "candidate_a",
            "exact_v1_admission_reason": "d_shape_barrier_protected_horizon",
        },
    )

    assert reason is None


def test_exact_v1_forecast_override_reason_allows_fidelity_first_turn_local_target_win_append_despite_nonimproving_tracking_score() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
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

    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast={
            "fidelity_exact_next": 0.78,
            "abs_energy_total_error_next": 0.18,
            "abs_primary_density_error_next": 0.0114,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0057,
            "tracking_score_horizon": 0.18,
        },
        selected_forecast={
            "fidelity_exact_next": 0.779,
            "abs_energy_total_error_next": 0.18,
            "abs_primary_density_error_next": 0.0113,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0056,
            "tracking_score_horizon": 0.19,
        },
        action_kind="append_candidate",
        selected={
            "candidate_label": "candidate_a",
            "exact_v1_admission_reason": "fidelity_first_turn_local_target_win",
        },
    )

    assert reason is None


def test_exact_v1_forecast_override_reason_allows_density_first_append_despite_nonimproving_tracking_score() -> None:
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
    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
        "tracking_primary_density_slope_error_mean": 0.20,
        "primary_density_slope_error_next": 0.20,
        "abs_primary_density_slope_error_next": 0.20,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.19,
        "abs_primary_density_error_next": 0.19,
        "tracking_primary_density_slope_error_mean": 0.10,
        "primary_density_slope_error_next": 0.10,
        "abs_primary_density_slope_error_next": 0.10,
        "normalized_energy_total_error_next": 0.31,
        "abs_energy_total_error_next": 0.31,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
    }

    assert controller._forecast_tracking_score(forecast=selected_forecast) > controller._forecast_tracking_score(
        forecast=stay_forecast
    )
    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        action_kind="append_candidate",
        selected={"candidate_label": "candidate_a"},
    )

    assert reason is None


def test_exact_v1_forecast_override_reason_allows_sign_lag_site_win_when_enabled() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_density_sign_lag_weight=1.0,
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
    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
        "tracking_primary_density_slope_error_mean": 0.20,
        "primary_density_slope_error_next": 0.20,
        "abs_primary_density_slope_error_next": 0.20,
        "primary_density_sign_lag_next": 0.40,
        "abs_primary_density_sign_lag_next": 0.40,
        "normalized_site_occupations_abs_error_max_next": 0.40,
        "site_occupations_abs_error_max_next": 0.40,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.94,
        "normalized_primary_density_error_next": 0.22,
        "abs_primary_density_error_next": 0.22,
        "tracking_primary_density_slope_error_mean": 0.22,
        "primary_density_slope_error_next": 0.22,
        "abs_primary_density_slope_error_next": 0.22,
        "primary_density_sign_lag_next": 0.28,
        "abs_primary_density_sign_lag_next": 0.28,
        "normalized_site_occupations_abs_error_max_next": 0.28,
        "site_occupations_abs_error_max_next": 0.28,
        "normalized_energy_total_error_next": 0.18,
        "abs_energy_total_error_next": 0.18,
        "abs_doublon_error_next": 0.0,
    }

    assert controller._forecast_tracking_score(forecast=selected_forecast) > controller._forecast_tracking_score(
        forecast=stay_forecast
    )
    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        action_kind="append_candidate",
        selected={"candidate_label": "candidate_a"},
    )

    assert reason is None


def test_exact_v1_componentwise_aspiration_uses_reduced_floor_in_sign_lag_window() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_density_sign_lag_weight=1.0,
            exact_v1_sign_lag_window_activation=0.30,
            exact_v1_sign_lag_window_target_gain_floor=5.0e-3,
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
    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
        "tracking_primary_density_slope_error_mean": 0.20,
        "primary_density_slope_error_next": 0.20,
        "abs_primary_density_slope_error_next": 0.20,
        "primary_density_sign_lag_next": 0.40,
        "abs_primary_density_sign_lag_next": 0.40,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.40,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.194,
        "abs_primary_density_error_next": 0.194,
        "tracking_primary_density_slope_error_mean": 0.194,
        "primary_density_slope_error_next": 0.194,
        "abs_primary_density_slope_error_next": 0.194,
        "primary_density_sign_lag_next": 0.34,
        "abs_primary_density_sign_lag_next": 0.34,
        "normalized_energy_total_error_next": 0.11,
        "abs_energy_total_error_next": 0.11,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.394,
    }

    allowed, reason = controller._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )

    assert allowed is True
    assert reason is None


def test_exact_v1_nonimproving_score_escape_uses_reduced_floor_in_sign_lag_window() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_density_sign_lag_weight=1.0,
            exact_v1_sign_lag_window_activation=0.30,
            exact_v1_sign_lag_window_target_gain_floor=5.0e-3,
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
    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
        "tracking_primary_density_slope_error_mean": 0.20,
        "primary_density_slope_error_next": 0.20,
        "abs_primary_density_slope_error_next": 0.20,
        "primary_density_sign_lag_next": 0.40,
        "abs_primary_density_sign_lag_next": 0.40,
        "normalized_site_occupations_abs_error_max_next": 0.40,
        "site_occupations_abs_error_max_next": 0.40,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.945,
        "normalized_primary_density_error_next": 0.196,
        "abs_primary_density_error_next": 0.196,
        "tracking_primary_density_slope_error_mean": 0.196,
        "primary_density_slope_error_next": 0.196,
        "abs_primary_density_slope_error_next": 0.196,
        "primary_density_sign_lag_next": 0.34,
        "abs_primary_density_sign_lag_next": 0.34,
        "normalized_site_occupations_abs_error_max_next": 0.394,
        "site_occupations_abs_error_max_next": 0.394,
        "normalized_energy_total_error_next": 0.12,
        "abs_energy_total_error_next": 0.12,
        "abs_doublon_error_next": 0.0,
    }

    assert (
        controller._exact_v1_nonimproving_score_allows_density_first_append(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
        )
        is True
    )


def test_primary_density_postcross_wrong_sign_terms_penalize_wrong_side_after_crossing() -> None:
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
    terms = controller._primary_density_postcross_wrong_sign_terms(
        forecasts=[
            {
                "primary_density_controller_next": 0.4,
                "primary_density_exact_next": -0.5,
            },
            {
                "primary_density_controller_next": 0.2,
                "primary_density_exact_next": -0.6,
            },
        ],
        weights=(2.0, 1.0),
        anchor={
            "primary_density_controller_next": 0.6,
            "primary_density_exact_next": 0.7,
        },
        primary_density_scale=1.0,
    )

    assert terms["primary_density_postcross_wrong_sign_active"] == pytest.approx(1.0)
    assert terms["primary_density_postcross_wrong_sign_abs_error_mean"] == pytest.approx((2.0 * 0.4 + 1.0 * 0.2) / 3.0)
    assert terms["primary_density_postcross_wrong_sign_error_mean"] == pytest.approx((2.0 * 0.4 + 1.0 * 0.2) / 3.0)


def test_exact_v1_componentwise_aspiration_uses_reduced_floor_in_postcross_wrong_sign_window() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_postcross_wrong_sign_activation=0.30,
            exact_v1_postcross_wrong_sign_target_gain_floor=5.0e-3,
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
    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
        "tracking_primary_density_slope_error_mean": 0.20,
        "primary_density_slope_error_next": 0.20,
        "abs_primary_density_slope_error_next": 0.20,
        "tracking_primary_density_postcross_wrong_sign_active": 1.0,
        "tracking_primary_density_postcross_wrong_sign_error_mean": 0.40,
        "tracking_primary_density_postcross_wrong_sign_abs_error_mean": 0.40,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.40,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.194,
        "abs_primary_density_error_next": 0.194,
        "tracking_primary_density_slope_error_mean": 0.194,
        "primary_density_slope_error_next": 0.194,
        "abs_primary_density_slope_error_next": 0.194,
        "tracking_primary_density_postcross_wrong_sign_active": 1.0,
        "tracking_primary_density_postcross_wrong_sign_error_mean": 0.20,
        "tracking_primary_density_postcross_wrong_sign_abs_error_mean": 0.20,
        "normalized_energy_total_error_next": 0.11,
        "abs_energy_total_error_next": 0.11,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.394,
    }

    allowed, reason = controller._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )

    assert allowed is True
    assert reason is None


def test_exact_v1_forecast_override_reason_keeps_veto_for_sign_lag_site_win_when_disabled() -> None:
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
    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.20,
        "abs_primary_density_error_next": 0.20,
        "tracking_primary_density_slope_error_mean": 0.20,
        "primary_density_slope_error_next": 0.20,
        "abs_primary_density_slope_error_next": 0.20,
        "abs_primary_density_sign_lag_next": 0.40,
        "site_occupations_abs_error_max_next": 0.40,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.94,
        "normalized_primary_density_error_next": 0.22,
        "abs_primary_density_error_next": 0.22,
        "tracking_primary_density_slope_error_mean": 0.22,
        "primary_density_slope_error_next": 0.22,
        "abs_primary_density_slope_error_next": 0.22,
        "abs_primary_density_sign_lag_next": 0.28,
        "site_occupations_abs_error_max_next": 0.28,
        "normalized_energy_total_error_next": 0.18,
        "abs_energy_total_error_next": 0.18,
        "abs_doublon_error_next": 0.0,
    }

    assert controller._forecast_tracking_score(forecast=selected_forecast) > controller._forecast_tracking_score(
        forecast=stay_forecast
    )
    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        action_kind="append_candidate",
        selected={"candidate_label": "candidate_a"},
    )

    assert reason == "exact_forecast_nonimproving_tracking_score"


def test_exact_v1_forecast_override_reason_keeps_stay_for_severe_energy_fidelity_regression_without_density_slope_gain() -> None:
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
    stay_forecast = {
        "fidelity_exact_next": 0.95,
        "normalized_primary_density_error_next": 0.40,
        "abs_primary_density_error_next": 0.40,
        "tracking_primary_density_slope_error_mean": 0.20,
        "primary_density_slope_error_next": 0.20,
        "abs_primary_density_slope_error_next": 0.20,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.92,
        "normalized_primary_density_error_next": 0.10,
        "abs_primary_density_error_next": 0.10,
        "tracking_primary_density_slope_error_mean": 0.19,
        "primary_density_slope_error_next": 0.19,
        "abs_primary_density_slope_error_next": 0.19,
        "normalized_energy_total_error_next": 0.60,
        "abs_energy_total_error_next": 0.60,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
    }

    assert controller._forecast_tracking_score(forecast=selected_forecast) > controller._forecast_tracking_score(
        forecast=stay_forecast
    )
    reason = controller._exact_v1_forecast_override_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
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


@pytest.mark.parametrize(
    "guardrail_mode",
    ("d_shape_barrier_v1", "fidelity_first_barrier_v1"),
)
def test_candidate_step_scales_for_selected_extend_preferred_site_turn_family(
    guardrail_mode: str,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode=guardrail_mode,
            candidate_step_scales=(0.25, 1.0),
            exact_v1_d_shape_turn_window_abs_activation=0.04,
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
    scales = controller._candidate_step_scales_for_selected(
        selected={"candidate_label": "paop_full:paop_disp(site=1)"},
        time_stop=0.2,
    )

    assert scales == pytest.approx((0.25, 1.0))


def test_exact_v1_selects_damped_candidate_step_scale_from_local_forecast(
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

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        append_amp = float(theta_runtime[-1])
        if append_amp <= 0.3:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.1,
                "rho_miss_next": 0.1,
                "step_gain_ratio_next": 1.0,
                "condition_number_next": 1.0,
            }
            return dict(forecast), [], 0.1
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": 1.0,
            "rho_miss_next": 0.8,
            "step_gain_ratio_next": 0.2,
            "condition_number_next": 5.0,
        }
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

    scaled_selected, scaled_forecast = controller._select_exact_v1_candidate_step_scale(
        baseline_theta_dot=np.array([0.2], dtype=float),
        selected=selected,
        dt=1.0,
        time_stop=0.2,
    )

    assert float(scaled_selected["candidate_step_scale"]) == pytest.approx(0.25)
    assert np.asarray(scaled_selected["theta_dot_aug"], dtype=float) == pytest.approx(np.array([0.2, 0.25]))
    assert float(scaled_selected["candidate_summary"].selected_step_scale) == pytest.approx(0.25)
    assert str(scaled_forecast["forecast_mode"]) == "local_projective_v1"
    assert float(scaled_forecast["local_projective_score_total"]) == pytest.approx(0.1)


def test_exact_v1_baseline_step_scale_rollout_uses_local_gain_not_anchor_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_baseline_gain_scales=(0.5, 1.0, 2.0),
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
    baseline_theta_dot = np.zeros(int(controller.current_theta.size), dtype=float)
    baseline_theta_dot[0] = 0.2
    baseline_theta_dot[-1] = 0.4
    seen_immediate_gain_ratio: list[object] = []

    def _fake_local_projective_forecast_rollout(**kwargs):
        seen_immediate_gain_ratio.append(kwargs.get("immediate_gain_ratio"))
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        amp = float(theta_runtime[-1])
        score = abs(amp - 0.04)
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": float(score),
            "baseline_proposal_kind": "norm_locked_blend_v1",
        }
        return dict(forecast), [], float(score)

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=0.2,
    )

    assert seen_immediate_gain_ratio
    assert all(value is None for value in seen_immediate_gain_ratio)
    assert float(step_scale) == pytest.approx(0.1)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(np.array([0.02, 0.0, 0.04]))
    assert str(forecast["forecast_mode"]) == "local_projective_v1"


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


def test_exact_v1_horizon_prefers_gentler_candidate_step_scale_from_local_rollout(
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

    def _fake_local_projective_forecast_rollout(**kwargs):
        append_amp = float(np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)[-1])
        if append_amp < 0.9:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.4,
                "tracking_horizon_steps_scored": 3,
                "tracking_horizon_weights_used": [1.0, 1.0, 1.0],
            }
            return dict(forecast), [], 0.4
        if append_amp < 1.1:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 1.2,
                "tracking_horizon_steps_scored": 3,
                "tracking_horizon_weights_used": [1.0, 1.0, 1.0],
            }
            return dict(forecast), [], 1.2
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": 2.0,
            "tracking_horizon_steps_scored": 3,
            "tracking_horizon_weights_used": [1.0, 1.0, 1.0],
        }
        return dict(forecast), [], 2.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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


def test_exact_v1_selects_damped_drive_aligned_baseline_step_scale_from_local_rollout(
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

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        amp = float(theta_runtime[-1])
        if 0.035 <= amp <= 0.045:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.1,
            }
            return dict(forecast), [], 0.1
        if abs(amp) <= 1.0e-12:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.3,
            }
            return dict(forecast), [], 0.3
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": 1.0,
        }
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=baseline_theta_dot,
        dt=1.0,
        time_stop=0.2,
    )

    assert float(step_scale) == pytest.approx(0.1)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(np.array([0.02, 0.0, 0.04]))
    assert str(forecast["forecast_mode"]) == "local_projective_v1"
    assert float(forecast["local_projective_score_total"]) == pytest.approx(0.1)


def test_drive_aligned_baseline_step_scales_for_time_extend_in_turn_window() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            candidate_step_scales=(0.25, 1.0),
            exact_v1_d_shape_turn_window_abs_activation=0.04,
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
    scales = controller._drive_aligned_baseline_step_scales_for_time(time_stop=0.2)

    assert scales == pytest.approx((0.0, 0.05, 0.1, 0.25, 1.0))


def test_exact_v1_baseline_step_scale_excludes_zero_when_rho_miss_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", miss_threshold=0.05),
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
    baseline = _baseline_geometry_payload(controller)
    baseline["summary"] = dataclass_replace(baseline["summary"], rho_miss=0.2)
    seen_drive_amp: list[float] = []

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        drive_amp = float(theta_runtime[-1])
        seen_drive_amp.append(drive_amp)
        forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": abs(drive_amp - 0.1)}
        return dict(forecast), [], float(forecast["local_projective_score_total"])

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

    scaled_theta_dot, step_scale, _blend_weight, _gain_scale, _forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([0.2, 0.0, 1.0], dtype=float),
        baseline=baseline,
        dt=1.0,
        time_stop=0.2,
    )

    assert seen_drive_amp
    assert all(abs(value) > 1.0e-12 for value in seen_drive_amp)
    assert float(step_scale) > 0.0
    assert np.asarray(scaled_theta_dot, dtype=float)[-1] > 0.0



def test_exact_v1_selects_damped_baseline_step_scale_from_local_forecast(
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

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        drive_amp = float(theta_runtime[-1])
        if abs(drive_amp - 0.1) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.1}
            return dict(forecast), [], 0.1
        if drive_amp < 0.1:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.3}
            return dict(forecast), [], 0.3
        forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 1.0}
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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
    assert str(forecast["forecast_mode"]) == "local_projective_v1"
    assert float(forecast["local_projective_score_total"]) == pytest.approx(0.1)


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

    def _fake_local_projective_forecast_rollout(**kwargs):
        drive_amp = float(np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)[-1])
        if abs(drive_amp - 0.15) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.05}
            return dict(forecast), [], 0.05
        if abs(drive_amp - 0.1) <= 1.0e-9 or abs(drive_amp - 0.2) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.1}
            return dict(forecast), [], 0.1
        if drive_amp < 0.25:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.3}
            return dict(forecast), [], 0.3
        forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 1.0}
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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
    assert str(forecast["forecast_mode"]) == "local_projective_v1"
    assert float(forecast["local_projective_score_total"]) == pytest.approx(0.05)


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

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        drive_amp = float(theta_runtime[-1])
        lead_amp = float(theta_runtime[0])
        if abs(drive_amp - (1.0 / np.sqrt(2.0))) <= 1.0e-9 and abs(lead_amp - (0.2 + (1.0 / np.sqrt(2.0)))) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.1}
            return dict(forecast), [], 0.1
        if abs(lead_amp - 1.2) <= 1.0e-9 and abs(drive_amp) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.3}
            return dict(forecast), [], 0.3
        forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 1.0}
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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
    assert str(forecast["forecast_mode"]) == "local_projective_v1"
    assert float(forecast["local_projective_score_total"]) == pytest.approx(0.1)


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

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        lead_amp = float(theta_runtime[0])
        drive_amp = float(theta_runtime[-1])
        if abs(lead_amp - (0.2 + 1.0 / np.sqrt(1.25))) <= 1.0e-9 and abs(drive_amp + (0.5 / np.sqrt(1.25))) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.05}
            return dict(forecast), [], 0.05
        if abs(lead_amp - 1.2) <= 1.0e-9 and abs(drive_amp) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.1}
            return dict(forecast), [], 0.1
        if abs(lead_amp - (0.2 + 1.0 / np.sqrt(2.0))) <= 1.0e-9 and abs(drive_amp - (1.0 / np.sqrt(2.0))) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.2}
            return dict(forecast), [], 0.2
        forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 1.0}
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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
    assert str(forecast["forecast_mode"]) == "local_projective_v1"
    assert float(forecast["local_projective_score_total"]) == pytest.approx(0.05)


def test_exact_tangent_secant_proposal_disabled_in_exact_free_controller() -> None:
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
    proposal = controller._exact_tangent_secant_proposal(
        baseline=baseline,
        dt=0.5,
        time_stop=0.5,
    )

    assert proposal is None


def test_exact_tangent_secant_proposal_remains_disabled_with_signed_energy_lead_limit() -> None:
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

    proposal = controller._exact_tangent_secant_proposal(
        baseline=baseline,
        dt=0.5,
        time_stop=0.5,
    )

    assert proposal is None


def test_exact_tangent_secant_proposal_remains_disabled_in_fidelity_first_mode() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
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

    proposal = controller._exact_tangent_secant_proposal(
        baseline=baseline,
        dt=0.5,
        time_stop=0.5,
    )

    assert proposal is None


def test_exact_v1_selection_can_pick_tangent_secant_proposal_from_local_rollout(
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
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_dot_step = np.asarray(kwargs["theta_dot_step"], dtype=float).reshape(-1)
        if abs(float(theta_dot_step[2])) > abs(float(theta_dot_step[0])):
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.01,
            }
            return dict(forecast), [dict(forecast)], 0.01
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": 1.0,
        }
        return dict(forecast), [dict(forecast)], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

    scaled_theta_dot, step_scale, blend_weight, gain_scale, forecast = controller._select_exact_v1_baseline_step_scale(
        baseline_theta_dot=np.array([1.0, 0.0, 0.0], dtype=float),
        baseline={"G": np.eye(3, dtype=float)},
        dt=0.1,
        time_stop=0.1,
    )

    assert np.asarray(scaled_theta_dot, dtype=float) == pytest.approx(
        np.array([0.05, 0.0, 0.0], dtype=float)
    )
    assert float(step_scale) == pytest.approx(0.05)
    assert float(blend_weight) == pytest.approx(0.0)
    assert float(gain_scale) == pytest.approx(1.0)
    assert str(forecast["baseline_proposal_kind"]) == "norm_locked_blend_v1"
    assert bool(forecast["baseline_include_tangent_secant_proposal"]) is False
    assert forecast["baseline_tangent_secant_projection_quality"] is None
    assert forecast["baseline_tangent_secant_displacement_norm"] is None


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

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_dot_step = np.asarray(kwargs["theta_dot_step"], dtype=float).reshape(-1)
        if abs(float(theta_dot_step[2])) > abs(float(theta_dot_step[0])):
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.01,
            }
            return dict(forecast), [dict(forecast)], 0.01
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": 1.0,
        }
        return dict(forecast), [dict(forecast)], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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


def test_exact_v1_selects_baseline_gain_scale_above_one_when_local_forecast_prefers_stronger_same_shape(
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

    def _fake_local_projective_forecast_rollout(**kwargs):
        theta_runtime = np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)
        lead_amp = float(theta_runtime[0])
        drive_amp = float(theta_runtime[-1])
        if abs(lead_amp - (0.2 + 1.2 / np.sqrt(2.0))) <= 1.0e-9 and abs(drive_amp - (1.2 / np.sqrt(2.0))) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.01}
            return dict(forecast), [], 0.01
        if abs(lead_amp - (0.2 + 1.0 / np.sqrt(2.0))) <= 1.0e-9 and abs(drive_amp - (1.0 / np.sqrt(2.0))) <= 1.0e-9:
            forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 0.1}
            return dict(forecast), [], 0.1
        forecast = {"forecast_mode": "local_projective_v1", "local_projective_score_total": 1.0}
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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
    assert str(forecast["forecast_mode"]) == "local_projective_v1"
    assert float(forecast["local_projective_score_total"]) == pytest.approx(0.01)


def test_exact_v1_energy_excursion_under_term_prefers_higher_post_step_gain_in_local_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    baseline_theta_dot = np.array([0.2, 0.0, 1.0], dtype=float)

    def _local_rollout_factory(
        *,
        score_map: dict[float, float],
        under_mean_map: dict[float, float] | None = None,
        under_weight: float = 0.0,
        over_weight: float = 0.0,
        rel_tol: float = 0.0,
    ):
        local_under_mean_map = {} if under_mean_map is None else dict(under_mean_map)

        def _fake_local_projective_forecast_rollout(**kwargs):
            drive_amp = round(
                float(np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)[-1]),
                1,
            )
            score = float(score_map.get(float(drive_amp), 1.0))
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": float(score),
                "tracking_energy_excursion_under_response_mean": float(
                    local_under_mean_map.get(float(drive_amp), 0.0)
                ),
                "tracking_energy_excursion_under_weight": float(under_weight),
                "tracking_energy_excursion_over_weight": float(over_weight),
                "tracking_energy_excursion_rel_tolerance": float(rel_tol),
            }
            return dict(forecast), [dict(forecast)], float(score)

        return _fake_local_projective_forecast_rollout

    controller_plain = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(1.0,),
            exact_forecast_baseline_gain_scales=(1.0, 1.2),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_plain,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(score_map={1.0: 0.1, 1.2: 0.2}),
    )
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
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_exc,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(
            score_map={1.0: 0.2, 1.2: 0.1},
            under_mean_map={1.2: 0.0015},
            under_weight=200.0,
        ),
    )
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
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_under_only,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(
            score_map={1.0: 0.3, 1.2: 0.2, 1.4: 0.1},
            under_mean_map={1.2: 0.0015, 1.4: 0.0020},
            under_weight=200.0,
        ),
    )
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
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_band,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(
            score_map={1.0: 0.3, 1.2: 0.1, 1.4: 0.2},
            under_mean_map={1.2: 0.0015, 1.4: 0.0020},
            under_weight=200.0,
            over_weight=500.0,
            rel_tol=0.03,
        ),
    )
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


def test_exact_v1_horizon_prefers_gentler_baseline_step_scale_in_local_rollout(
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

    def _fake_local_projective_forecast_rollout(**kwargs):
        drive_amp = float(np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)[-1])
        if abs(drive_amp - 1.0) <= 1.0e-9:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.3,
                "tracking_horizon_steps_scored": 3,
                "tracking_horizon_weights_used": [1.0, 1.0, 1.0],
            }
            return dict(forecast), [], 0.3
        if 0.08 <= drive_amp <= 0.35:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.1,
                "tracking_horizon_steps_scored": 3,
                "tracking_horizon_weights_used": [1.0, 1.0, 1.0],
            }
            return dict(forecast), [], 0.1
        if drive_amp < 0.08:
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": 0.2,
                "tracking_horizon_steps_scored": 3,
                "tracking_horizon_weights_used": [1.0, 1.0, 1.0],
            }
            return dict(forecast), [], 0.2
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": 1.0,
            "tracking_horizon_steps_scored": 3,
            "tracking_horizon_weights_used": [1.0, 1.0, 1.0],
        }
        return dict(forecast), [], 1.0

    monkeypatch.setattr(controller, "_local_projective_forecast_rollout", _fake_local_projective_forecast_rollout)

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


def test_exact_v1_energy_slope_term_prefers_shape_matched_baseline_step_in_local_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    baseline_theta_dot = np.array([0.2, 0.0, 1.0], dtype=float)

    def _local_rollout_factory(*, score_map: dict[float, float], slope_mean_map: dict[float, float]):
        def _fake_local_projective_forecast_rollout(**kwargs):
            drive_amp = round(
                float(np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)[-1]),
                1,
            )
            score = float(score_map.get(float(drive_amp), 1.0))
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": float(score),
                "tracking_energy_slope_abs_error_mean": float(
                    slope_mean_map.get(float(drive_amp), 0.01)
                ),
                "tracking_energy_slope_weight": 500.0,
            }
            return dict(forecast), [dict(forecast)], float(score)

        return _fake_local_projective_forecast_rollout

    controller_h2 = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.2, 1.0),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_h2,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(score_map={1.0: 0.1, 0.2: 0.2}, slope_mean_map={1.0: 0.01, 0.2: 0.0}),
    )
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
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_shape,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(score_map={1.0: 0.2, 0.2: 0.1}, slope_mean_map={1.0: 0.01, 0.2: 0.0}),
    )
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


def test_exact_v1_energy_curvature_term_is_active_for_h2_with_anchor_in_local_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    baseline_theta_dot = np.array([0.2, 0.0, 1.0], dtype=float)

    def _local_rollout_factory(*, score_map: dict[float, float], curvature_mean_map: dict[float, float]):
        def _fake_local_projective_forecast_rollout(**kwargs):
            drive_amp = round(
                float(np.asarray(kwargs["theta_runtime_start"], dtype=float).reshape(-1)[-1]),
                1,
            )
            score = float(score_map.get(float(drive_amp), 1.0))
            forecast = {
                "forecast_mode": "local_projective_v1",
                "local_projective_score_total": float(score),
                "tracking_energy_curvature_abs_error_mean": float(
                    curvature_mean_map.get(float(drive_amp), 0.0)
                ),
                "tracking_energy_curvature_weight": 200.0,
            }
            return dict(forecast), [dict(forecast)], float(score)

        return _fake_local_projective_forecast_rollout

    controller_no_curv = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            candidate_step_scales=(0.2, 1.0),
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_energy_slope_weight=500.0,
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_no_curv,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(score_map={0.2: 0.1, 1.0: 0.2}, curvature_mean_map={0.2: 0.0, 1.0: 0.05}),
    )
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
            exact_forecast_density_slope_weight=0.0,
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
    monkeypatch.setattr(
        controller_curv,
        "_local_projective_forecast_rollout",
        _local_rollout_factory(score_map={0.2: 0.2, 1.0: 0.1}, curvature_mean_map={0.2: 0.0, 1.0: 0.05}),
    )
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
            append_no_harm_guard_enabled=False,
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


def test_exact_forecast_tracking_score_defaults_match_unweighted_sum() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
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
    forecasts = [
        {
            "fidelity_exact_next": 0.97,
            "abs_staggered_error_next": 0.02,
            "abs_doublon_error_next": 0.03,
            "site_occupations_abs_error_max_next": 0.04,
            "abs_energy_total_error_next": 0.05,
        },
        {
            "fidelity_exact_next": 0.96,
            "abs_staggered_error_next": 0.01,
            "abs_doublon_error_next": 0.02,
            "site_occupations_abs_error_max_next": 0.03,
            "abs_energy_total_error_next": 0.04,
        },
    ]
    expected = (
        2.0 * ((1.0 - 0.97) + 0.02 + 0.03 + 0.04 + 0.05)
        + 1.0 * ((1.0 - 0.96) + 0.01 + 0.02 + 0.03 + 0.04)
    ) / 3.0

    actual = controller._forecast_tracking_score(forecast=forecasts)

    assert float(actual) == pytest.approx(float(expected))


def test_exact_forecast_tracking_score_respects_explicit_doublon_weight() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    baseline = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=3,
    )
    doublon_heavy = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_doublon_error_weight=10.0,
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
    forecast_a = {
        "fidelity_exact_next": 0.99,
        "abs_staggered_error_next": 0.01,
        "abs_doublon_error_next": 0.10,
        "site_occupations_abs_error_max_next": 0.01,
        "abs_energy_total_error_next": 0.01,
    }
    forecast_b = {
        "fidelity_exact_next": 0.99,
        "abs_staggered_error_next": 0.01,
        "abs_doublon_error_next": 0.01,
        "site_occupations_abs_error_max_next": 0.01,
        "abs_energy_total_error_next": 0.14,
    }

    baseline_a = baseline._forecast_tracking_score(forecast=forecast_a)
    baseline_b = baseline._forecast_tracking_score(forecast=forecast_b)
    weighted_a = doublon_heavy._forecast_tracking_score(forecast=forecast_a)
    weighted_b = doublon_heavy._forecast_tracking_score(forecast=forecast_b)

    assert float(baseline_a) < float(baseline_b)
    assert float(weighted_b) < float(weighted_a)


def test_exact_forecast_tracking_score_adds_density_slope_term() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=7.0,
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
    forecasts = [
        {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": 0.0,
            "primary_density_exact_next": 0.0,
            "site_occupations_exact_next": [0.0, 0.0],
            "doublon_exact_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.0,
        },
        {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": 0.0,
            "primary_density_exact_next": 1.0,
            "site_occupations_exact_next": [1.0, 0.0],
            "doublon_exact_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.0,
        },
    ]

    score = controller._forecast_tracking_score(forecast=forecasts)

    assert float(score) == pytest.approx(7.0)


def test_exact_forecast_tracking_score_adds_postcross_wrong_sign_term_from_stored_metric() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=4.0,
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
    forecast = {
        "fidelity_exact_next": 1.0,
        "abs_primary_density_error_next": 1.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.0,
        "tracking_primary_density_postcross_wrong_sign_error_mean": 0.25,
    }

    score = controller._forecast_tracking_score(forecast=forecast)

    assert float(score) == pytest.approx(1.0)


def test_exact_forecast_rollout_builds_postcross_anchor_when_only_postcross_weight_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=6.0,
            exact_forecast_drive_harmonic_weight=0.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )

    def _fake_exact_step_forecast(**kwargs):
        time_stop = float(kwargs["time_stop"])
        if time_stop < 0.2:
            exact_value = 0.5
            ctrl_value = 0.6
        elif time_stop < 0.3:
            exact_value = -0.5
            ctrl_value = 0.4
        else:
            exact_value = -0.5
            ctrl_value = 0.2
        return {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": ctrl_value,
            "primary_density_exact_next": exact_value,
            "site_occupations_exact_next": [0.0, 0.0],
            "doublon_exact_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": abs(ctrl_value - exact_value),
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.0,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    first, _forecasts, score = controller._exact_forecast_rollout(
        time_stop=0.2,
        dt=0.1,
        executor=controller.current_executor,
        theta_runtime_start=np.asarray(controller.current_theta, dtype=float),
        theta_dot_step=np.asarray([0.0], dtype=float),
    )

    expected_error = (2.0 * 0.4 + 1.0 * 0.2) / 3.0
    assert float(first["tracking_primary_density_postcross_wrong_sign_active"]) == pytest.approx(1.0)
    assert float(first["tracking_primary_density_postcross_wrong_sign_error_mean"]) == pytest.approx(
        expected_error
    )
    assert float(first["tracking_primary_density_postcross_wrong_sign_weight"]) == pytest.approx(6.0)
    assert float(score) == pytest.approx(6.0 * expected_error)


def test_exact_forecast_rollout_stores_shadow_d_shape_and_total_occupation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_postcross_compare_diag=True,
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=0.0,
            exact_forecast_drive_harmonic_weight=0.0,
            exact_forecast_energy_slope_weight=0.0,
            exact_forecast_energy_curvature_weight=0.0,
            exact_forecast_energy_excursion_under_weight=0.0,
            exact_forecast_energy_excursion_over_weight=0.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )

    def _fake_exact_step_forecast(**kwargs):
        time_stop = float(kwargs["time_stop"])
        if time_stop < 0.2:
            exact_site = [0.5, 0.5]
            ctrl_site = [0.5, 0.5]
        elif time_stop < 0.3:
            exact_site = [1.0, 0.0]
            ctrl_site = [0.9, 0.2]
        else:
            exact_site = [0.5, 0.5]
            ctrl_site = [0.7, 0.3]
        exact_d = float(exact_site[0] - exact_site[1])
        ctrl_d = float(ctrl_site[0] - ctrl_site[1])
        return {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": ctrl_d,
            "primary_density_exact_next": exact_d,
            "site_occupations_controller_next": [float(x) for x in ctrl_site],
            "site_occupations_exact_next": [float(x) for x in exact_site],
            "doublon_controller_next": 0.0,
            "doublon_exact_next": 0.0,
            "energy_total_controller_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": abs(ctrl_d - exact_d),
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": float(
                max(abs(ctrl_site[0] - exact_site[0]), abs(ctrl_site[1] - exact_site[1]))
            ),
            "abs_energy_total_error_next": 0.0,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    first, _forecasts, score = controller._exact_forecast_rollout(
        time_stop=0.2,
        dt=0.1,
        executor=controller.current_executor,
        theta_runtime_start=np.asarray(controller.current_theta, dtype=float),
        theta_dot_step=np.asarray([0.0], dtype=float),
    )

    assert float(first["tracking_d_curvature_abs_error_mean"]) == pytest.approx(1.0)
    assert float(first["tracking_d_excursion_under_response_mean"]) == pytest.approx(0.2)
    assert float(first["tracking_d_excursion_over_response_mean"]) == pytest.approx(0.0)
    assert float(first["tracking_total_occupation_abs_error_next"]) == pytest.approx(0.1)
    assert float(first["tracking_total_occupation_abs_error_mean"]) == pytest.approx(2.0 / 30.0)
    assert float(score) == pytest.approx(0.0)


def test_exact_forecast_tracking_score_suppresses_energy_and_fidelity_terms_under_d_shape_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    base_cfg = RealtimeCheckpointConfig(
        mode="exact_v1",
        exact_forecast_guardrail_mode="off",
        exact_forecast_tracking_horizon_steps=1,
        exact_forecast_tracking_fidelity_defect_weight=3.0,
        exact_forecast_tracking_primary_density_error_weight=0.0,
        exact_forecast_tracking_doublon_error_weight=0.0,
        exact_forecast_tracking_site_occupations_error_weight=0.0,
        exact_forecast_tracking_energy_total_error_weight=7.0,
        exact_forecast_density_slope_weight=0.0,
        exact_forecast_density_curvature_weight=0.0,
        exact_forecast_density_excursion_under_weight=0.0,
        exact_forecast_density_excursion_over_weight=0.0,
        exact_forecast_density_sign_lag_weight=0.0,
        exact_forecast_density_postcross_wrong_sign_weight=0.0,
        exact_forecast_drive_harmonic_weight=0.0,
        exact_forecast_energy_slope_weight=11.0,
        exact_forecast_energy_curvature_weight=13.0,
        exact_forecast_energy_excursion_under_weight=17.0,
        exact_forecast_energy_excursion_over_weight=19.0,
    )
    off_controller = RealtimeCheckpointController(
        cfg=base_cfg,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    barrier_controller = RealtimeCheckpointController(
        cfg=dataclass_replace(
            base_cfg,
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
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
    monkeypatch.setattr(
        off_controller,
        "_energy_shape_tracking_terms",
        lambda **kwargs: {
            "energy_slope_abs_error_mean": 0.2,
            "energy_curvature_abs_error_mean": 0.3,
        },
    )
    monkeypatch.setattr(
        barrier_controller,
        "_energy_shape_tracking_terms",
        lambda **kwargs: {
            "energy_slope_abs_error_mean": 0.2,
            "energy_curvature_abs_error_mean": 0.3,
        },
    )
    monkeypatch.setattr(
        off_controller,
        "_energy_excursion_tracking_terms",
        lambda **kwargs: {
            "energy_excursion_under_response_mean": 0.4,
            "energy_excursion_over_response_mean": 0.5,
        },
    )
    monkeypatch.setattr(
        barrier_controller,
        "_energy_excursion_tracking_terms",
        lambda **kwargs: {
            "energy_excursion_under_response_mean": 0.4,
            "energy_excursion_over_response_mean": 0.5,
        },
    )
    forecast = {
        "fidelity_exact_next": 1.0,
        "normalized_primary_density_error_next": 0.25,
        "abs_primary_density_error_next": 0.25,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.01,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    outside_turn_forecast = {
        "fidelity_exact_next": 1.0,
        "normalized_primary_density_error_next": 0.40,
        "abs_primary_density_error_next": 0.40,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.01,
        "primary_density_exact_next": 0.05,
        "tracking_primary_density_exact_abs_min_horizon": 0.05,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    assert float(off_controller._forecast_tracking_score(forecast=forecast)) == pytest.approx(22.47)
    assert float(barrier_controller._forecast_tracking_score(forecast=forecast)) == pytest.approx(0.25)
    assert float(barrier_controller._forecast_tracking_score(forecast=outside_turn_forecast)) == pytest.approx(
        0.40
    )


def test_exact_forecast_tracking_score_adds_soft_barrier_penalty_under_d_shape_barrier() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
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
    safe_forecast = {
        "fidelity_exact_next": 0.995,
        "normalized_primary_density_error_next": 0.10,
        "abs_primary_density_error_next": 0.10,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.01,
        "abs_energy_total_error_next": 0.05,
        "tracking_energy_slope_abs_error_mean": 0.02,
        "tracking_energy_curvature_abs_error_mean": 0.01,
        "tracking_energy_excursion_under_response_mean": 0.01,
        "tracking_energy_excursion_over_response_mean": 0.0,
    }
    unsafe_forecast = {
        "fidelity_exact_next": 0.995,
        "normalized_primary_density_error_next": 0.10,
        "abs_primary_density_error_next": 0.10,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.01,
        "abs_energy_total_error_next": 0.30,
        "tracking_energy_slope_abs_error_mean": 0.20,
        "tracking_energy_curvature_abs_error_mean": 0.15,
        "tracking_energy_excursion_under_response_mean": 0.12,
        "tracking_energy_excursion_over_response_mean": 0.12,
    }

    safe_score = float(controller._forecast_tracking_score(forecast=safe_forecast))
    unsafe_score = float(controller._forecast_tracking_score(forecast=unsafe_forecast))

    assert safe_score == pytest.approx(0.10)
    assert unsafe_score > safe_score + 5.0


def test_fidelity_first_barrier_score_prefers_better_fidelity_when_energy_is_only_a_guardrail() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    fidelity_first = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=20.0,
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
    legacy = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="off",
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=20.0,
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
    better_fidelity = {
        "fidelity_exact_next": 0.92,
        "normalized_primary_density_error_next": 0.0,
        "normalized_doublon_error_next": 0.0,
        "normalized_site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.15,
        "abs_energy_total_error_next": 0.15,
        "tracking_total_occupation_abs_error_next": 0.0,
        "tracking_total_occupation_abs_error_mean": 0.0,
    }
    better_energy = {
        "fidelity_exact_next": 0.80,
        "normalized_primary_density_error_next": 0.0,
        "normalized_doublon_error_next": 0.0,
        "normalized_site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.01,
        "abs_energy_total_error_next": 0.01,
        "tracking_total_occupation_abs_error_next": 0.0,
        "tracking_total_occupation_abs_error_mean": 0.0,
    }

    assert float(fidelity_first._forecast_tracking_score(forecast=better_fidelity)) < float(
        fidelity_first._forecast_tracking_score(forecast=better_energy)
    )
    assert float(legacy._forecast_tracking_score(forecast=better_fidelity)) > float(
        legacy._forecast_tracking_score(forecast=better_energy)
    )


def test_exact_v1_d_shape_barrier_protected_horizon_requires_core_and_shadow_d_shape_win_before_soft_barrier() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_energy_slope_abs_error_mean": 0.01,
        "tracking_energy_curvature_abs_error_mean": 0.0,
        "tracking_energy_excursion_under_response_mean": 0.0,
        "tracking_energy_excursion_over_response_mean": 0.0,
        "tracking_d_curvature_abs_error_mean": 0.4,
        "tracking_d_excursion_under_response_mean": 0.3,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.799,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.11,
        "tracking_energy_slope_abs_error_mean": 0.18,
        "tracking_energy_curvature_abs_error_mean": 0.0,
        "tracking_energy_excursion_under_response_mean": 0.0,
        "tracking_energy_excursion_over_response_mean": 0.0,
        "tracking_d_curvature_abs_error_mean": 0.1,
        "tracking_d_excursion_under_response_mean": 0.0,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
    }

    assert float(controller._exact_v1_live_d_shape_core_score(forecast=selected_forecast)) < float(
        controller._exact_v1_live_d_shape_core_score(forecast=stay_forecast)
    )
    assert float(controller._exact_v1_live_d_score(forecast=selected_forecast)) > float(
        controller._exact_v1_live_d_score(forecast=stay_forecast)
    )

    allowed, reason = controller._exact_v1_d_shape_barrier_protected_horizon_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert bool(allowed) is True
    assert reason is None


def test_exact_v1_d_shape_barrier_protected_horizon_blocks_core_only_win_without_shadow_turn_improvement() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=2.0,
            exact_forecast_density_curvature_weight=50.0,
            exact_forecast_density_excursion_under_weight=225.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "abs_primary_density_error_next": 1.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_energy_slope_abs_error_mean": 0.0,
        "tracking_energy_curvature_abs_error_mean": 0.0,
        "tracking_energy_excursion_under_response_mean": 0.0,
        "tracking_energy_excursion_over_response_mean": 0.0,
        "abs_primary_density_slope_error_next": 1.0,
        "tracking_d_curvature_abs_error_mean": 0.002532196574743605,
        "tracking_d_excursion_under_response_mean": 0.005910965620336395,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.799,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.100001,
        "tracking_energy_slope_abs_error_mean": 0.0,
        "tracking_energy_curvature_abs_error_mean": 0.0,
        "tracking_energy_excursion_under_response_mean": 0.0,
        "tracking_energy_excursion_over_response_mean": 0.0,
        "abs_primary_density_slope_error_next": 0.0,
        "tracking_d_curvature_abs_error_mean": 0.0035,
        "tracking_d_excursion_under_response_mean": 0.0070,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
    }

    assert float(controller._exact_v1_live_d_shape_core_score(forecast=selected_forecast)) < float(
        controller._exact_v1_live_d_shape_core_score(forecast=stay_forecast)
    )
    assert controller._exact_v1_d_shape_shadow_only_total(selected_forecast) > controller._exact_v1_d_shape_shadow_only_total(stay_forecast)

    allowed, reason = controller._exact_v1_d_shape_barrier_protected_horizon_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert bool(allowed) is False
    assert str(reason) == "no_shadow_turn_win_vs_stay"


def test_exact_v1_fidelity_first_protected_horizon_uses_fidelity_core_with_shadow_guard() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=12.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=0.0,
            exact_forecast_density_excursion_under_weight=0.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.08,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.60,
        "abs_primary_density_error_next": 0.10,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.30,
        "tracking_d_excursion_under_response_mean": 0.30,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.90,
        "abs_primary_density_error_next": 0.70,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.17,
        "abs_energy_total_error_next": 0.17,
        "tracking_d_curvature_abs_error_mean": 0.05,
        "tracking_d_excursion_under_response_mean": 0.05,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
    }

    assert float(controller._exact_v1_live_d_shape_core_score(forecast=selected_forecast)) > float(
        controller._exact_v1_live_d_shape_core_score(forecast=stay_forecast)
    )
    assert float(controller._exact_v1_guarded_turn_window_core_score(forecast=selected_forecast)) < float(
        controller._exact_v1_guarded_turn_window_core_score(forecast=stay_forecast)
    )

    allowed, reason = controller._exact_v1_d_shape_barrier_protected_horizon_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert bool(allowed) is True
    assert reason is None


def test_exact_v1_fidelity_first_protected_horizon_allows_material_turn_local_win_despite_nonimproving_core() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=0.0,
            exact_forecast_density_excursion_under_weight=0.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.02,
            exact_forecast_abs_energy_error_increase_tol=0.08,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.90,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.08,
        "tracking_d_excursion_under_response_mean": 0.06,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.03, 0.03],
        "tracking_site_curvature_abs_error_mean_by_site": [0.01, 0.01],
        "tracking_site_excursion_under_response_mean_by_site": [0.02, 0.02],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.89,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.03,
        "tracking_d_excursion_under_response_mean": 0.01,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.02, 0.02],
        "tracking_site_curvature_abs_error_mean_by_site": [0.005, 0.005],
        "tracking_site_excursion_under_response_mean_by_site": [0.005, 0.005],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    assert float(controller._exact_v1_guarded_turn_window_core_score(forecast=selected_forecast)) > float(
        controller._exact_v1_guarded_turn_window_core_score(forecast=stay_forecast)
    )
    assert controller._exact_v1_d_shape_shadow_only_total(selected_forecast) < controller._exact_v1_d_shape_shadow_only_total(stay_forecast)

    allowed, reason = controller._exact_v1_d_shape_barrier_protected_horizon_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert bool(allowed) is True
    assert reason is None


def test_exact_v1_fidelity_first_protected_horizon_allows_combined_onset_win_before_signlag_window() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=0.0,
            exact_forecast_density_excursion_under_weight=0.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.02,
            exact_forecast_abs_energy_error_increase_tol=0.08,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.90,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.08,
        "tracking_d_excursion_under_response_mean": 0.06,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.03, 0.03],
        "tracking_site_curvature_abs_error_mean_by_site": [0.01, 0.01],
        "tracking_site_excursion_under_response_mean_by_site": [0.02, 0.02],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.89,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.075,
        "tracking_d_excursion_under_response_mean": 0.054,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.027, 0.027],
        "tracking_site_curvature_abs_error_mean_by_site": [0.009, 0.009],
        "tracking_site_excursion_under_response_mean_by_site": [0.019, 0.019],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    assert float(controller._exact_v1_guarded_turn_window_core_score(forecast=selected_forecast)) > float(
        controller._exact_v1_guarded_turn_window_core_score(forecast=stay_forecast)
    )

    allowed, reason = controller._exact_v1_d_shape_barrier_protected_horizon_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert bool(allowed) is True
    assert reason is None


def test_exact_v1_fidelity_first_turn_local_target_win_rejects_tiny_combined_onset_gains() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=0.0,
            exact_forecast_density_excursion_under_weight=0.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.02,
            exact_forecast_abs_energy_error_increase_tol=0.08,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.90,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.08,
        "tracking_d_excursion_under_response_mean": 0.06,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.03, 0.03],
        "tracking_site_curvature_abs_error_mean_by_site": [0.01, 0.01],
        "tracking_site_excursion_under_response_mean_by_site": [0.02, 0.02],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.89,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "normalized_energy_total_error_next": 0.10,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.0785,
        "tracking_d_excursion_under_response_mean": 0.059,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.0297, 0.0297],
        "tracking_site_curvature_abs_error_mean_by_site": [0.0099, 0.0099],
        "tracking_site_excursion_under_response_mean_by_site": [0.0198, 0.0198],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    allowed, reason = controller._exact_v1_fidelity_first_turn_local_target_win_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )

    assert bool(allowed) is False
    assert str(reason) == "no_turn_local_target_win_vs_stay"


@pytest.mark.parametrize(
    "guardrail_mode",
    ("d_shape_barrier_v1", "fidelity_first_barrier_v1"),
)
def test_exact_v1_preferred_site_index_at_time_uses_site_occupations_snapshot(
    guardrail_mode: str,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode=guardrail_mode,
            exact_v1_d_shape_turn_window_abs_activation=0.04,
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
    controller._observable_snapshot = lambda psi: {
        "site_occupations": np.asarray([0.2, 0.8], dtype=float),
        "staggered": -0.6,
    }

    preferred_site = controller._exact_v1_preferred_site_index_at_time(
        baseline={"theta_dot_step": np.asarray([1.0], dtype=float)},
        time_start=0.0,
        time_stop=0.1,
    )

    assert preferred_site == 1


@pytest.mark.parametrize(
    "guardrail_mode",
    ("d_shape_barrier_v1", "fidelity_first_barrier_v1"),
)
def test_inject_preferred_site_shortlist_record_replaces_worst_with_preferred_site_candidate(
    guardrail_mode: str,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", exact_forecast_guardrail_mode=guardrail_mode),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    records = [
        {
            "candidate_label": "paop_full:paop_disp(site=0)",
            "candidate_identity": "site0_a",
            "position_id": 0,
            "scout_score": -0.35,
            "simple_score": -0.35,
        },
        {
            "candidate_label": "paop_full:paop_cloud_p(site=0->phonon=1)",
            "candidate_identity": "site0_b",
            "position_id": 1,
            "scout_score": -0.36,
            "simple_score": -0.36,
        },
        {
            "candidate_label": "paop_lf_full:paop_dbl_p(site=1->phonon=1)",
            "candidate_identity": "site1_best",
            "position_id": 2,
            "scout_score": -0.70,
            "simple_score": -0.70,
        },
    ]
    shortlist = records[:2]

    injected = controller._inject_preferred_site_shortlist_record(
        records=records,
        shortlist=shortlist,
        preferred_site_index=1,
    )

    assert len(injected) == 2
    assert any(str(item["candidate_identity"]) == "site1_best" for item in injected)
    assert any(str(item["candidate_identity"]) == "site0_a" for item in injected)


@pytest.mark.parametrize(
    "guardrail_mode",
    ("d_shape_barrier_v1", "fidelity_first_barrier_v1"),
)
def test_inject_preferred_site_shortlist_record_keeps_shortlist_when_preferred_site_already_present(
    guardrail_mode: str,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", exact_forecast_guardrail_mode=guardrail_mode),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    shortlist = [
        {
            "candidate_label": "paop_lf_full:paop_dbl_p(site=1->phonon=1)",
            "candidate_identity": "site1_best",
            "position_id": 2,
            "scout_score": -0.70,
            "simple_score": -0.70,
        },
        {
            "candidate_label": "paop_full:paop_disp(site=0)",
            "candidate_identity": "site0_a",
            "position_id": 0,
            "scout_score": -0.35,
            "simple_score": -0.35,
        },
    ]

    injected = controller._inject_preferred_site_shortlist_record(
        records=shortlist,
        shortlist=shortlist,
        preferred_site_index=1,
    )

    assert injected == shortlist


@pytest.mark.parametrize(
    "guardrail_mode",
    ("d_shape_barrier_v1", "fidelity_first_barrier_v1"),
)
def test_inject_preferred_site_shortlist_record_upgrades_generic_preferred_site_to_turn_family(
    guardrail_mode: str,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1", exact_forecast_guardrail_mode=guardrail_mode),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.2,
        num_times=2,
    )
    records = [
        {
            "candidate_label": "paop_lf_full:paop_dbl_p(site=1->phonon=1)",
            "candidate_identity": "site1_generic",
            "position_id": 2,
            "scout_score": -0.70,
            "simple_score": -0.70,
        },
        {
            "candidate_label": "paop_full:paop_disp(site=1)",
            "candidate_identity": "site1_turn",
            "position_id": 3,
            "scout_score": -0.55,
            "simple_score": -0.55,
        },
        {
            "candidate_label": "paop_full:paop_disp(site=0)",
            "candidate_identity": "site0_a",
            "position_id": 0,
            "scout_score": -0.35,
            "simple_score": -0.35,
        },
    ]
    shortlist = [records[0], records[2]]

    injected = controller._inject_preferred_site_shortlist_record(
        records=records,
        shortlist=shortlist,
        preferred_site_index=1,
    )

    assert len(injected) == 2
    assert any(str(item["candidate_identity"]) == "site1_turn" for item in injected)
    assert not any(str(item["candidate_identity"]) == "site1_generic" for item in injected)


@pytest.mark.parametrize(
    "guardrail_mode",
    ("d_shape_barrier_v1", "fidelity_first_barrier_v1"),
)
def test_candidate_pool_terms_reopens_preferred_site_turn_family_when_only_generic_nonrepeats_remain(
    monkeypatch: pytest.MonkeyPatch,
    guardrail_mode: str,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode=guardrail_mode,
            exact_v1_d_shape_turn_window_abs_activation=0.04,
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
    controller.current_terms = [
        SimpleNamespace(
            label="paop_full:paop_disp(site=1)__r0",
            source_label="paop_full:paop_disp(site=1)",
        ),
        SimpleNamespace(
            label="paop_full:paop_cloud_p(site=1->phonon=0)__r0",
            source_label="paop_full:paop_cloud_p(site=1->phonon=0)",
        ),
    ]
    controller.replay_context = dataclass_replace(
        controller.replay_context,
        family_pool=[
            SimpleNamespace(label="paop_full:paop_disp(site=1)"),
            SimpleNamespace(label="paop_full:paop_cloud_p(site=1->phonon=0)"),
            SimpleNamespace(label="paop_lf_full:paop_dbl_p(site=1->phonon=1)"),
        ],
    )
    monkeypatch.setattr(
        controller,
        "_observable_snapshot",
        lambda psi: {
            "n_site": np.asarray([0.1, 0.9], dtype=float),
            "staggered": -0.8,
        },
    )

    available = controller._candidate_pool_terms(
        baseline={"theta_dot_step": np.asarray([1.0], dtype=float)},
        time_start=0.0,
        time_stop=0.1,
    )
    labels = [str(term.label) for _, term in available]

    assert "paop_lf_full:paop_dbl_p(site=1->phonon=1)" in labels
    assert "paop_full:paop_disp(site=1)" in labels
    assert "paop_full:paop_cloud_p(site=1->phonon=0)" in labels


def test_exact_v1_d_shape_barrier_protected_horizon_respects_barrier_guardrail() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "abs_primary_density_error_next": 0.0,
        "abs_primary_density_slope_error_next": 0.0,
        "abs_primary_density_sign_lag_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.4,
        "tracking_d_excursion_under_response_mean": 0.3,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.795,
        "abs_primary_density_error_next": 0.0,
        "abs_primary_density_slope_error_next": 0.0,
        "abs_primary_density_sign_lag_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.14,
        "tracking_d_curvature_abs_error_mean": 0.1,
        "tracking_d_excursion_under_response_mean": 0.0,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
    }

    allowed, reason = controller._exact_v1_d_shape_barrier_protected_horizon_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert bool(allowed) is False
    assert str(reason) == "exact_forecast_d_shape_energy_regression"


def test_exact_v1_live_d_breakdowns_match_scores() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
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
    forecast = {
        "normalized_primary_density_error_next": 0.1,
        "tracking_primary_density_slope_error_mean": 0.2,
        "tracking_d_curvature_abs_error_mean": 0.4,
        "tracking_d_excursion_under_response_mean": 0.3,
        "tracking_d_excursion_over_response_mean": 0.05,
        "abs_energy_total_error_next": 0.22,
        "tracking_energy_slope_abs_error_mean": 0.08,
        "tracking_energy_curvature_abs_error_mean": 0.1,
        "tracking_energy_excursion_under_response_mean": 0.12,
        "tracking_energy_excursion_over_response_mean": 0.11,
        "tracking_total_occupation_abs_error_next": 0.05,
        "tracking_total_occupation_abs_error_mean": 0.06,
    }

    core = controller._exact_v1_live_d_shape_core_breakdown(forecast=forecast)
    barrier = controller._exact_v1_live_d_barrier_breakdown(forecast=forecast)
    total = controller._exact_v1_live_d_score_breakdown(forecast=forecast)

    assert core["total"] == pytest.approx(
        controller._exact_v1_live_d_shape_core_score(forecast=forecast)
    )
    assert barrier["total"] == pytest.approx(
        controller._exact_v1_live_d_barrier_penalty(forecast=forecast)
    )
    assert total["total"] == pytest.approx(
        controller._exact_v1_live_d_score(forecast=forecast)
    )


def test_debug_probe_exact_v1_returns_checkpoint_payload() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.0,
            append_margin_abs=1.0e-6,
            exact_v1_postcross_compare_diag=True,
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.0,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_horizon_weights=(1.0,),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=1.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=1.0,
            candidate_step_scales=(0.2,),
            exact_forecast_baseline_blend_weights=(0.0,),
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

    payload = controller.debug_probe_exact_v1(
        probe_checkpoints=[0],
        candidate_rank_limit=2,
        baseline_variant_limit=3,
    )

    assert payload["mode"] == "exact_v1_debug_probe"
    assert payload["probe_checkpoints"] == [0]
    assert len(payload["checkpoints"]) == 1
    row = payload["checkpoints"][0]
    assert row["checkpoint_index"] == 0
    assert "baseline_variants" in row
    assert "stay" in row
    assert "candidates" in row
    assert "candidate_stage_of_death" in row
    assert isinstance(row["candidate_stage_of_death"], list)
    assert row["guarded_commit_surface_mode"] == "guarded_turn_window_core"


def test_debug_probe_exact_v1_reports_single_surface_commit_mode_when_enabled() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.0,
            gain_ratio_threshold=0.0,
            append_margin_abs=1.0e-6,
            exact_v1_postcross_compare_diag=True,
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.0,
            exact_v1_single_surface_commit_law=True,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_horizon_weights=(1.0,),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=1.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=1.0,
            candidate_step_scales=(0.2,),
            exact_forecast_baseline_blend_weights=(0.0,),
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

    payload = controller.debug_probe_exact_v1(
        probe_checkpoints=[0],
        candidate_rank_limit=2,
        baseline_variant_limit=3,
    )

    assert payload["checkpoints"][0]["guarded_commit_surface_mode"] == "forecast_tracking_total"


def test_exact_v1_d_shape_turn_window_active_requires_exact_turn_signal() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
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

    inactive = {
        "primary_density_exact_next": 0.05,
        "tracking_primary_density_exact_abs_min_horizon": 0.05,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    active_by_min_abs = {
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.03,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    active_by_cross = {
        "primary_density_exact_next": 0.20,
        "tracking_primary_density_exact_abs_min_horizon": 0.20,
        "tracking_primary_density_exact_zero_crossed_horizon": 1.0,
    }

    assert bool(controller._exact_v1_d_shape_turn_window_active(stay_forecast=inactive)) is False
    assert bool(controller._exact_v1_d_shape_turn_window_active(stay_forecast=active_by_min_abs)) is True
    assert bool(controller._exact_v1_d_shape_turn_window_active(stay_forecast=active_by_cross)) is True


def test_select_action_exact_v1_appends_when_live_gates_and_forecast_pass() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
            forecast_accept_margin=0.0,
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
    stay_forecast = {"local_projective_score_total": 0.8, "tracking_score_horizon": 0.8}
    selected_forecast = {"local_projective_score_total": 0.4, "tracking_score_horizon": 0.4}

    def _fake_select_exact_v1_candidate_step_scale(**kwargs):
        selected = dict(kwargs["selected"])
        return selected, dict(selected_forecast)

    def _unexpected_scale(**kwargs):
        pytest.fail("live path should reject below-threshold gain before forecast probe logic")

    controller._select_exact_v1_candidate_step_scale = _fake_select_exact_v1_candidate_step_scale  # type: ignore[method-assign]
    baseline = {"summary": SimpleNamespace(rho_miss=0.2), "theta_dot_step": np.asarray([0.0], dtype=float)}
    confirmed = [
        {
            "candidate_label": "candidate_a",
            "candidate_identity": "candidate_a",
            "candidate_pool_index": 0,
            "position_id": 0,
            "gain_exact": 2.0e-6,
            "gain_ratio": 0.05,
            "adjusted_gain": 2.0,
            "confirm_score": 2.0,
            "candidate_summary": SimpleNamespace(
                position_jump_penalty=0.0,
                compile_proxy_total=0.0,
                groups_new=0.0,
                candidate_pool_index=0,
                position_id=0,
            ),
        }
    ]

    action_kind, selected = controller._select_action_exact_v1(
        baseline=baseline,
        confirmed=confirmed,
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
    )

    assert str(action_kind) == "append_candidate"
    assert selected is not None
    assert str(selected["candidate_label"]) == "candidate_a"
    assert bool(selected["exact_confirm_passed"]) is True
    assert bool(selected["exact_confirm_near_miss_admitted"]) is False
    assert bool(selected["exact_confirm_below_floor_probed"]) is False
    assert str(selected["exact_v1_admission_reason"]) == "live_local_gates_passed"
    assert str(controller._last_exact_v1_selection_reason) == "live_local_gates_passed"


def test_exact_v1_componentwise_aspiration_allows_d_shape_turn_window_target_win() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "normalized_primary_density_error_next": 0.05,
        "primary_density_slope_error_next": 0.02,
        "abs_primary_density_sign_lag_next": 0.01,
        "site_occupations_abs_error_max_next": 0.01,
        "normalized_energy_total_error_next": 0.10,
        "abs_primary_density_error_next": 0.05,
        "abs_primary_density_slope_error_next": 0.02,
        "abs_doublon_error_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.4,
        "tracking_d_excursion_under_response_mean": 0.3,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "tracking_score_horizon": 7.0,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.799,
        "normalized_primary_density_error_next": 0.05,
        "primary_density_slope_error_next": 0.02,
        "abs_primary_density_sign_lag_next": 0.01,
        "site_occupations_abs_error_max_next": 0.01,
        "normalized_energy_total_error_next": 0.11,
        "abs_primary_density_error_next": 0.05,
        "abs_primary_density_slope_error_next": 0.02,
        "abs_doublon_error_next": 0.0,
        "abs_energy_total_error_next": 0.11,
        "tracking_d_curvature_abs_error_mean": 0.1,
        "tracking_d_excursion_under_response_mean": 0.0,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "tracking_score_horizon": 3.0,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    allowed, reason = controller._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )

    assert bool(allowed) is True
    assert reason is None


def test_exact_v1_componentwise_aspiration_allows_fidelity_first_turn_local_target_win_despite_nonimproving_full_score() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
            exact_v1_single_surface_commit_law=True,
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=0.0,
            exact_forecast_density_excursion_under_weight=0.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_score_horizon": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.08,
        "tracking_d_excursion_under_response_mean": 0.06,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.03, 0.03],
        "tracking_site_curvature_abs_error_mean_by_site": [0.01, 0.01],
        "tracking_site_excursion_under_response_mean_by_site": [0.02, 0.02],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.801,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_score_horizon": 0.19,
        "tracking_d_curvature_abs_error_mean": 0.03,
        "tracking_d_excursion_under_response_mean": 0.01,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.02, 0.02],
        "tracking_site_curvature_abs_error_mean_by_site": [0.005, 0.005],
        "tracking_site_excursion_under_response_mean_by_site": [0.005, 0.005],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    allowed, reason = controller._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )
    admission_reason = controller._exact_v1_fidelity_first_turn_local_target_admission_reason(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )

    assert bool(allowed) is True
    assert reason is None
    assert str(admission_reason) == "fidelity_first_turn_local_target_win"


def test_exact_v1_componentwise_aspiration_keeps_fidelity_first_turn_local_target_win_material() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="fidelity_first_barrier_v1",
            exact_v1_single_surface_commit_law=True,
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=1.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=0.0,
            exact_forecast_density_excursion_under_weight=0.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_score_horizon": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.08,
        "tracking_d_excursion_under_response_mean": 0.06,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.03, 0.03],
        "tracking_site_curvature_abs_error_mean_by_site": [0.01, 0.01],
        "tracking_site_excursion_under_response_mean_by_site": [0.02, 0.02],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.801,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_score_horizon": 0.19,
        "tracking_d_curvature_abs_error_mean": 0.079,
        "tracking_d_excursion_under_response_mean": 0.059,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_site_slope_abs_error_mean_by_site": [0.0298, 0.0298],
        "tracking_site_curvature_abs_error_mean_by_site": [0.0098, 0.0098],
        "tracking_site_excursion_under_response_mean_by_site": [0.0198, 0.0198],
        "tracking_site_excursion_over_response_mean_by_site": [0.0, 0.0],
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    allowed, reason = controller._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )

    assert bool(allowed) is False
    assert str(reason) == "no_target_win_vs_stay"


def test_forecast_postcross_compare_summary_exposes_truth_metrics() -> None:
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
    forecast = {
        "fidelity_exact_next": 0.875,
        "abs_energy_total_error_next": 0.012,
        "site_occupations_abs_error_max_next": 0.034,
        "abs_primary_density_error_next": 0.056,
        "tracking_primary_density_postcross_wrong_sign_error_mean": 0.0,
        "tracking_primary_density_postcross_wrong_sign_abs_error_mean": 0.0,
        "tracking_primary_density_postcross_wrong_sign_active": 0.0,
        "tracking_d_curvature_abs_error_mean": 0.01,
        "tracking_d_excursion_under_response_mean": 0.02,
        "tracking_d_excursion_over_response_mean": 0.03,
        "tracking_site_slope_abs_error_mean_by_site": [0.04, 0.05],
        "tracking_site_curvature_abs_error_mean_by_site": [0.06, 0.07],
        "tracking_site_excursion_under_response_mean_by_site": [0.08, 0.09],
        "tracking_site_excursion_over_response_mean_by_site": [0.01, 0.02],
        "tracking_total_occupation_abs_error_next": 0.003,
        "tracking_total_occupation_abs_error_mean": 0.004,
    }

    summary = controller._forecast_postcross_compare_summary(forecast=forecast, score_total=1.23)

    assert summary["fidelity_exact_next"] == pytest.approx(0.875)
    assert summary["abs_energy_total_error_next"] == pytest.approx(0.012)
    assert summary["site_occupations_abs_error_max_next"] == pytest.approx(0.034)
    assert summary["abs_primary_density_error_next"] == pytest.approx(0.056)



def test_exact_v1_componentwise_aspiration_uses_full_commit_surface_when_flag_enabled() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    legacy = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    single_surface = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_v1_single_surface_commit_law=True,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_energy_slope_abs_error_mean": 0.01,
        "tracking_energy_curvature_abs_error_mean": 0.0,
        "tracking_energy_excursion_under_response_mean": 0.0,
        "tracking_energy_excursion_over_response_mean": 0.0,
        "tracking_d_curvature_abs_error_mean": 0.4,
        "tracking_d_excursion_under_response_mean": 0.3,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.799,
        "abs_primary_density_error_next": 0.0,
        "abs_doublon_error_next": 0.0,
        "site_occupations_abs_error_max_next": 0.0,
        "abs_energy_total_error_next": 0.11,
        "tracking_energy_slope_abs_error_mean": 0.18,
        "tracking_energy_curvature_abs_error_mean": 0.0,
        "tracking_energy_excursion_under_response_mean": 0.0,
        "tracking_energy_excursion_over_response_mean": 0.0,
        "tracking_d_curvature_abs_error_mean": 0.1,
        "tracking_d_excursion_under_response_mean": 0.0,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    assert float(legacy._exact_v1_guarded_turn_window_core_score(forecast=selected_forecast)) < float(
        legacy._exact_v1_guarded_turn_window_core_score(forecast=stay_forecast)
    )
    assert float(legacy._forecast_tracking_score(forecast=selected_forecast)) > float(
        legacy._forecast_tracking_score(forecast=stay_forecast)
    )

    legacy_allowed, legacy_reason = legacy._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )
    single_allowed, single_reason = single_surface._exact_v1_componentwise_aspiration_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
        below_floor_probe=False,
    )

    assert bool(legacy_allowed) is True
    assert legacy_reason is None
    assert bool(single_allowed) is False
    assert str(single_reason) == "no_target_win_vs_stay"


def test_exact_v1_below_floor_probe_limit_uses_d_shape_outside_turn_override() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold=7,
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
    outside_turn_forecast = {
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.08,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    inside_turn_forecast = {
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.02,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    controller._exact_v1_append_lane_stall_streak = 5
    assert controller._exact_v1_below_floor_probe_limit(stay_forecast=outside_turn_forecast) == 0

    controller._exact_v1_append_lane_stall_streak = 7
    assert controller._exact_v1_below_floor_probe_limit(stay_forecast=outside_turn_forecast) == 1

    controller._exact_v1_append_lane_stall_streak = 5
    assert controller._exact_v1_below_floor_probe_limit(stay_forecast=inside_turn_forecast) == 1


def test_select_action_exact_v1_ignores_below_floor_probe_delay_in_live_path() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold=7,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=1.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.90,
        "normalized_primary_density_error_next": 0.06,
        "primary_density_slope_error_next": 0.03,
        "abs_primary_density_sign_lag_next": 0.03,
        "site_occupations_abs_error_max_next": 0.02,
        "normalized_energy_total_error_next": 0.05,
        "abs_primary_density_error_next": 0.06,
        "abs_primary_density_slope_error_next": 0.03,
        "abs_doublon_error_next": 0.0,
        "abs_energy_total_error_next": 0.05,
        "primary_density_exact_next": 0.08,
        "tracking_primary_density_exact_abs_min_horizon": 0.08,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.90,
        "normalized_primary_density_error_next": 0.01,
        "primary_density_slope_error_next": 0.03,
        "abs_primary_density_sign_lag_next": 0.03,
        "site_occupations_abs_error_max_next": 0.02,
        "normalized_energy_total_error_next": 0.05,
        "abs_primary_density_error_next": 0.01,
        "abs_primary_density_slope_error_next": 0.03,
        "abs_doublon_error_next": 0.0,
        "abs_energy_total_error_next": 0.05,
    }

    def _fake_select_exact_v1_candidate_step_scale(**kwargs):
        selected = dict(kwargs["selected"])
        return selected, dict(selected_forecast)

    controller._select_exact_v1_candidate_step_scale = _fake_select_exact_v1_candidate_step_scale  # type: ignore[method-assign]
    baseline = {"summary": SimpleNamespace(rho_miss=0.2), "theta_dot_step": np.asarray([0.0], dtype=float)}
    confirmed = [
        {
            "candidate_label": "candidate_a",
            "candidate_identity": "candidate_a",
            "candidate_pool_index": 0,
            "position_id": 0,
            "gain_exact": 1.0e-12,
            "gain_ratio": 1.0e-12,
            "adjusted_gain": -1.0e-3,
            "confirm_score": -1.0e-3,
            "candidate_summary": SimpleNamespace(
                position_jump_penalty=0.0,
                compile_proxy_total=0.0,
                groups_new=0.0,
                candidate_pool_index=0,
                position_id=0,
            ),
        }
    ]

    controller._exact_v1_append_lane_stall_streak = 5
    action_kind, selected = controller._select_action_exact_v1(
        baseline=baseline,
        confirmed=confirmed,
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
    )
    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "gain_ratio_below_threshold"

    controller._exact_v1_append_lane_stall_streak = 7
    action_kind, selected = controller._select_action_exact_v1(
        baseline=baseline,
        confirmed=confirmed,
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
    )
    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "gain_ratio_below_threshold"


def test_exact_v1_d_shape_turn_window_target_win_allows_pre_turn_shadow_bridge_when_enabled() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_v1_d_shape_pre_turn_shadow_bridge=True,
            exact_forecast_tracking_horizon_steps=1,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=10.0,
            exact_forecast_density_excursion_under_weight=10.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_fidelity_loss_tol=0.01,
            exact_forecast_abs_energy_error_increase_tol=0.02,
            exact_forecast_total_occupation_error_increase_tol=0.01,
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
    stay_forecast = {
        "fidelity_exact_next": 0.80,
        "normalized_primary_density_error_next": 0.05,
        "primary_density_slope_error_next": 0.02,
        "abs_primary_density_sign_lag_next": 0.01,
        "site_occupations_abs_error_max_next": 0.01,
        "normalized_energy_total_error_next": 0.10,
        "abs_primary_density_error_next": 0.05,
        "abs_primary_density_slope_error_next": 0.02,
        "abs_doublon_error_next": 0.0,
        "abs_energy_total_error_next": 0.10,
        "tracking_d_curvature_abs_error_mean": 0.4,
        "tracking_d_excursion_under_response_mean": 0.3,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "tracking_score_horizon": 7.0,
        "primary_density_exact_next": 0.20,
        "tracking_primary_density_exact_abs_min_horizon": 0.08,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }
    selected_forecast = {
        "fidelity_exact_next": 0.799,
        "normalized_primary_density_error_next": 0.05,
        "primary_density_slope_error_next": 0.02,
        "abs_primary_density_sign_lag_next": 0.01,
        "site_occupations_abs_error_max_next": 0.01,
        "normalized_energy_total_error_next": 0.11,
        "abs_primary_density_error_next": 0.05,
        "abs_primary_density_slope_error_next": 0.02,
        "abs_doublon_error_next": 0.0,
        "abs_energy_total_error_next": 0.11,
        "tracking_d_curvature_abs_error_mean": 0.1,
        "tracking_d_excursion_under_response_mean": 0.0,
        "tracking_d_excursion_over_response_mean": 0.0,
        "tracking_total_occupation_abs_error_next": 0.01,
        "tracking_total_occupation_abs_error_mean": 0.02,
        "tracking_score_horizon": 3.0,
        "primary_density_exact_next": 0.20,
        "tracking_primary_density_exact_abs_min_horizon": 0.08,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    allowed, reason = controller._exact_v1_d_shape_turn_window_target_win_result(
        stay_forecast=stay_forecast,
        selected_forecast=selected_forecast,
    )

    assert bool(allowed) is True
    assert str(reason) == "d_shape_barrier_pre_turn_shadow_bridge"


def test_select_action_exact_v1_ignores_exact_overlay_rescue_when_live_gain_gate_fails() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
            exact_v1_d_shape_pre_turn_shadow_bridge=True,
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
    stay_forecast = {
        "local_projective_score_total": 0.8,
        "tracking_score_horizon": 0.8,
        "primary_density_exact_next": 0.20,
        "tracking_primary_density_exact_abs_min_horizon": 0.08,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    def _unexpected_scale(**kwargs):
        pytest.fail("exact overlay rescue should not bypass live gain gate")

    controller._select_exact_v1_candidate_step_scale = _unexpected_scale  # type: ignore[method-assign]
    baseline = {"summary": SimpleNamespace(rho_miss=0.2), "theta_dot_step": np.asarray([0.0], dtype=float)}
    confirmed = [
        {
            "candidate_label": "candidate_a",
            "candidate_identity": "candidate_a",
            "candidate_pool_index": 0,
            "position_id": 0,
            "gain_exact": 1.0e-12,
            "gain_ratio": 1.0e-12,
            "adjusted_gain": -1.0e-3,
            "confirm_score": -1.0e-3,
            "candidate_summary": SimpleNamespace(
                position_jump_penalty=0.0,
                compile_proxy_total=0.0,
                groups_new=0.0,
                candidate_pool_index=0,
                position_id=0,
            ),
        }
    ]

    action_kind, selected = controller._select_action_exact_v1(
        baseline=baseline,
        confirmed=confirmed,
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "gain_ratio_below_threshold"


def test_select_action_exact_v1_ignores_exact_turn_window_guard_in_live_path() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
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
    stay_forecast = {
        "local_projective_score_total": 0.8,
        "tracking_score_horizon": 0.8,
        "primary_density_exact_next": 0.05,
        "tracking_primary_density_exact_abs_min_horizon": 0.05,
        "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
    }

    def _unexpected_scale(**kwargs):
        pytest.fail("exact turn-window guard should not control live append admission")

    controller._select_exact_v1_candidate_step_scale = _unexpected_scale  # type: ignore[method-assign]
    baseline = {"summary": SimpleNamespace(rho_miss=0.2), "theta_dot_step": np.asarray([0.0], dtype=float)}
    confirmed = [
        {
            "candidate_label": "candidate_a",
            "candidate_identity": "candidate_a",
            "candidate_pool_index": 0,
            "position_id": 0,
            "gain_exact": 2.0e-6,
            "gain_ratio": 0.05,
            "adjusted_gain": -1.0e-3,
            "confirm_score": -1.0e-3,
            "candidate_summary": SimpleNamespace(
                position_jump_penalty=0.0,
                compile_proxy_total=0.0,
                groups_new=0.0,
                candidate_pool_index=0,
                position_id=0,
            ),
        }
    ]

    action_kind, selected = controller._select_action_exact_v1(
        baseline=baseline,
        confirmed=confirmed,
        dt=0.1,
        time_stop=0.1,
        stay_forecast=stay_forecast,
    )

    assert str(action_kind) == "stay"
    assert selected is None
    assert str(controller._last_exact_v1_selection_reason) == "confirm_score_below_threshold"


def test_select_action_exact_v1_prefers_d_shape_candidate_under_d_shape_barrier() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    base_cfg = RealtimeCheckpointConfig(
        mode="exact_v1",
        miss_threshold=0.05,
        gain_ratio_threshold=0.02,
        append_margin_abs=1.0e-6,
        exact_forecast_guardrail_mode="off",
        exact_forecast_tracking_horizon_steps=1,
        exact_forecast_tracking_fidelity_defect_weight=0.0,
        exact_forecast_tracking_primary_density_error_weight=0.0,
        exact_forecast_tracking_doublon_error_weight=0.0,
        exact_forecast_tracking_site_occupations_error_weight=0.0,
        exact_forecast_tracking_energy_total_error_weight=100.0,
        exact_forecast_density_slope_weight=0.0,
        exact_forecast_density_curvature_weight=10.0,
        exact_forecast_density_excursion_under_weight=10.0,
        exact_forecast_density_excursion_over_weight=0.0,
        exact_forecast_density_sign_lag_weight=0.0,
        exact_forecast_density_postcross_wrong_sign_weight=0.0,
        exact_forecast_drive_harmonic_weight=0.0,
        exact_forecast_energy_slope_weight=100.0,
        exact_forecast_energy_curvature_weight=0.0,
        exact_forecast_energy_excursion_under_weight=0.0,
        exact_forecast_energy_excursion_over_weight=0.0,
    )

    def _make_controller(cfg: RealtimeCheckpointConfig) -> RealtimeCheckpointController:
        return RealtimeCheckpointController(
            cfg=cfg,
            replay_context=replay_context,
            h_poly=h_poly,
            hmat=hmat,
            psi_initial=psi_initial,
            best_theta=[0.2],
            allow_repeats=False,
            t_final=0.2,
            num_times=2,
        )

    off_controller = _make_controller(base_cfg)
    barrier_controller = _make_controller(
        dataclass_replace(
            base_cfg,
            exact_forecast_guardrail_mode="d_shape_barrier_v1",
            exact_v1_d_shape_turn_window_abs_activation=0.04,
        )
    )

    def _record(label: str, idx: int) -> dict[str, object]:
        return {
            "candidate_label": label,
            "candidate_identity": label,
            "candidate_pool_index": idx,
            "position_id": idx,
            "gain_exact": 0.1,
            "gain_ratio": 0.1,
            "adjusted_gain": 0.1,
            "confirm_score": 0.1,
            "candidate_summary": SimpleNamespace(
                position_jump_penalty=0.0,
                compile_proxy_total=0.0,
                groups_new=0.0,
                candidate_pool_index=idx,
                position_id=idx,
            ),
        }

    forecasts = {
        "energy_first": {
            "fidelity_exact_next": 1.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.01,
            "primary_density_exact_next": 0.08,
            "tracking_primary_density_exact_abs_min_horizon": 0.02,
            "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
            "tracking_energy_slope_abs_error_mean": 0.01,
            "tracking_energy_curvature_abs_error_mean": 0.0,
            "tracking_energy_excursion_under_response_mean": 0.0,
            "tracking_energy_excursion_over_response_mean": 0.0,
            "tracking_d_curvature_abs_error_mean": 0.4,
            "tracking_d_excursion_under_response_mean": 0.3,
            "tracking_d_excursion_over_response_mean": 0.0,
        },
        "d_first": {
            "fidelity_exact_next": 1.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.12,
            "primary_density_exact_next": 0.08,
            "tracking_primary_density_exact_abs_min_horizon": 0.02,
            "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
            "tracking_energy_slope_abs_error_mean": 0.03,
            "tracking_energy_curvature_abs_error_mean": 0.02,
            "tracking_energy_excursion_under_response_mean": 0.02,
            "tracking_energy_excursion_over_response_mean": 0.0,
            "tracking_d_curvature_abs_error_mean": 0.1,
            "tracking_d_excursion_under_response_mean": 0.0,
            "tracking_d_excursion_over_response_mean": 0.0,
        },
    }

    def _patch_selector(controller: RealtimeCheckpointController) -> None:
        def _fake_select_exact_v1_candidate_step_scale(**kwargs):
            selected = dict(kwargs["selected"])
            return selected, dict(forecasts[str(selected["candidate_label"])])

        controller._select_exact_v1_candidate_step_scale = _fake_select_exact_v1_candidate_step_scale  # type: ignore[method-assign]

    _patch_selector(off_controller)
    _patch_selector(barrier_controller)

    baseline = {"summary": SimpleNamespace(rho_miss=0.2), "theta_dot_step": np.asarray([0.0], dtype=float)}
    confirmed = [_record("energy_first", 0), _record("d_first", 1)]

    action_off, selected_off = off_controller._select_action_exact_v1(
        baseline=baseline,
        confirmed=confirmed,
        dt=0.1,
        time_stop=0.1,
        stay_forecast=None,
    )
    action_barrier, selected_barrier = barrier_controller._select_action_exact_v1(
        baseline=baseline,
        confirmed=confirmed,
        dt=0.1,
        time_stop=0.1,
        stay_forecast=None,
    )

    assert str(action_off) == "append_candidate"
    assert selected_off is not None
    assert str(selected_off["candidate_label"]) == "energy_first"
    assert str(action_barrier) == "append_candidate"
    assert selected_barrier is not None
    assert str(selected_barrier["candidate_label"]) == "d_first"


def test_progress_payload_includes_latest_and_rolling_exact_deltas() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            progress_observable_window=2,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )
    controller._append_counter = 2
    controller._ledger = [{"action_kind": "prune_coordinate"}]
    controller._trajectory = [
        {
            "fidelity_exact": 0.90,
            "abs_energy_total_error": 0.01,
            "site_occupations_abs_error_max": 0.20,
            "abs_primary_density_error": 0.30,
            "energy_total": 1.00,
            "primary_density": 0.10,
            "site_occupations": [0.40, 0.60],
        },
        {
            "fidelity_exact": 0.80,
            "abs_energy_total_error": 0.03,
            "site_occupations_abs_error_max": 0.40,
            "abs_primary_density_error": 0.50,
            "energy_total": 1.02,
            "primary_density": 0.12,
            "site_occupations": [0.42, 0.58],
        },
    ]

    payload = controller._progress_payload(stage="checkpoint_done", status="checkpoint_complete")

    assert payload["stage"] == "checkpoint_done"
    assert payload["status"] == "checkpoint_complete"
    assert int(payload["append_count"]) == 2
    assert int(payload["prune_count"]) == 1
    assert float(payload["latest_fidelity_exact"]) == pytest.approx(0.80)
    assert float(payload["latest_abs_energy_total_error"]) == pytest.approx(0.03)
    assert float(payload["latest_site_occupations_abs_error_max"]) == pytest.approx(0.40)
    assert float(payload["latest_abs_primary_density_error"]) == pytest.approx(0.50)
    assert int(payload["progress_observable_window"]) == 2
    assert float(payload["rolling_fidelity_exact_mean"]) == pytest.approx(0.85)
    assert float(payload["rolling_abs_energy_total_error_mean"]) == pytest.approx(0.02)
    assert float(payload["rolling_site_occupations_abs_error_max_mean"]) == pytest.approx(0.30)
    assert float(payload["rolling_abs_primary_density_error_mean"]) == pytest.approx(0.40)
    assert float(payload["rolling_energy_total_span"]) == pytest.approx(0.02)
    assert float(payload["rolling_site_occupations_span_max"]) == pytest.approx(0.02)
    assert float(payload["rolling_primary_density_span"]) == pytest.approx(0.02)


def test_progress_early_stop_reason_uses_rolling_exact_delta_thresholds() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            progress_observable_window=2,
            progress_early_stop_min_checkpoint=1,
            progress_early_stop_site_error_mean_max=0.25,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )
    controller._trajectory = [
        {
            "fidelity_exact": 0.90,
            "abs_energy_total_error": 0.01,
            "site_occupations_abs_error_max": 0.20,
            "abs_primary_density_error": 0.30,
        },
        {
            "fidelity_exact": 0.80,
            "abs_energy_total_error": 0.03,
            "site_occupations_abs_error_max": 0.40,
            "abs_primary_density_error": 0.50,
        },
    ]

    reason = controller._progress_early_stop_reason(checkpoint_index=1)

    assert str(reason).startswith("progress_site_error_mean_exceeds_threshold:")


def test_progress_early_stop_reason_can_stop_on_observable_stability() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            progress_observable_window=3,
            progress_early_stop_min_checkpoint=2,
            progress_early_stop_site_span_max=0.01,
            progress_early_stop_primary_density_span_max=0.01,
            progress_early_stop_energy_span_max=0.01,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )
    controller._trajectory = [
        {"energy_total": 1.000, "primary_density": 0.100, "site_occupations": [0.40, 0.60]},
        {"energy_total": 1.002, "primary_density": 0.101, "site_occupations": [0.401, 0.599]},
        {"energy_total": 1.003, "primary_density": 0.102, "site_occupations": [0.402, 0.598]},
    ]

    reason = controller._progress_early_stop_reason(checkpoint_index=2)

    assert str(reason).startswith("progress_observables_stable:")


def test_exact_forecast_tracking_score_adds_live_d_shape_terms_from_sequence() -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=2.0,
            exact_forecast_density_excursion_under_weight=3.0,
            exact_forecast_density_excursion_over_weight=5.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=0.0,
            exact_forecast_drive_harmonic_weight=0.0,
            exact_forecast_energy_slope_weight=0.0,
            exact_forecast_energy_curvature_weight=0.0,
            exact_forecast_energy_excursion_under_weight=0.0,
            exact_forecast_energy_excursion_over_weight=0.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )
    forecasts = [
        {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": 0.0,
            "primary_density_exact_next": 1.0,
            "site_occupations_exact_next": [1.0],
            "doublon_exact_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.0,
        },
        {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": 0.0,
            "primary_density_exact_next": 0.0,
            "site_occupations_exact_next": [0.0],
            "doublon_exact_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.0,
        },
    ]
    anchor = {
        "primary_density_controller_next": 0.0,
        "primary_density_exact_next": 0.0,
        "site_occupations_exact_next": [0.0],
        "doublon_exact_next": 0.0,
        "energy_total_exact_next": 0.0,
    }

    score = controller._forecast_tracking_score(
        forecast=forecasts,
        curvature_anchor=anchor,
    )

    assert float(score) == pytest.approx(6.0)


def test_exact_forecast_rollout_stores_shadow_d_shape_metrics_when_d_shape_escape_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_postcross_compare_diag=False,
            exact_v1_below_floor_energy_safe_d_shape_escape=True,
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=0.0,
            exact_forecast_drive_harmonic_weight=0.0,
            exact_forecast_energy_slope_weight=0.0,
            exact_forecast_energy_curvature_weight=0.0,
            exact_forecast_energy_excursion_under_weight=0.0,
            exact_forecast_energy_excursion_over_weight=0.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )

    def _fake_exact_step_forecast(**kwargs):
        time_stop = float(kwargs["time_stop"])
        if time_stop < 0.2:
            exact_site = [0.5, 0.5]
            ctrl_site = [0.5, 0.5]
        elif time_stop < 0.3:
            exact_site = [1.0, 0.0]
            ctrl_site = [0.9, 0.2]
        else:
            exact_site = [0.5, 0.5]
            ctrl_site = [0.7, 0.3]
        exact_d = float(exact_site[0] - exact_site[1])
        ctrl_d = float(ctrl_site[0] - ctrl_site[1])
        return {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": ctrl_d,
            "primary_density_exact_next": exact_d,
            "site_occupations_controller_next": [float(x) for x in ctrl_site],
            "site_occupations_exact_next": [float(x) for x in exact_site],
            "doublon_controller_next": 0.0,
            "doublon_exact_next": 0.0,
            "energy_total_controller_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": abs(ctrl_d - exact_d),
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": float(
                max(abs(ctrl_site[0] - exact_site[0]), abs(ctrl_site[1] - exact_site[1]))
            ),
            "abs_energy_total_error_next": 0.0,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    first, _forecasts, _score = controller._exact_forecast_rollout(
        time_stop=0.2,
        dt=0.1,
        executor=controller.current_executor,
        theta_runtime_start=np.asarray(controller.current_theta, dtype=float),
        theta_dot_step=np.asarray([0.0], dtype=float),
    )

    assert float(first["tracking_d_curvature_abs_error_mean"]) == pytest.approx(1.0)
    assert float(first["tracking_d_excursion_under_response_mean"]) == pytest.approx(0.2)
    assert float(first["tracking_d_excursion_over_response_mean"]) == pytest.approx(0.0)
    assert float(first["tracking_total_occupation_abs_error_next"]) == pytest.approx(0.1)
    assert float(first["tracking_total_occupation_abs_error_mean"]) == pytest.approx(2.0 / 30.0)


def test_exact_forecast_rollout_stores_shadow_d_shape_metrics_when_live_d_shape_weights_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_postcross_compare_diag=False,
            exact_v1_below_floor_energy_safe_d_shape_escape=False,
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_curvature_weight=1.0,
            exact_forecast_density_excursion_under_weight=0.0,
            exact_forecast_density_excursion_over_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=0.0,
            exact_forecast_drive_harmonic_weight=0.0,
            exact_forecast_energy_slope_weight=0.0,
            exact_forecast_energy_curvature_weight=0.0,
            exact_forecast_energy_excursion_under_weight=0.0,
            exact_forecast_energy_excursion_over_weight=0.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )

    def _fake_exact_step_forecast(**kwargs):
        time_stop = float(kwargs["time_stop"])
        if time_stop < 0.2:
            exact_site = [0.5, 0.5]
            ctrl_site = [0.5, 0.5]
        elif time_stop < 0.3:
            exact_site = [1.0, 0.0]
            ctrl_site = [0.9, 0.2]
        else:
            exact_site = [0.5, 0.5]
            ctrl_site = [0.7, 0.3]
        exact_d = float(exact_site[0] - exact_site[1])
        ctrl_d = float(ctrl_site[0] - ctrl_site[1])
        return {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": ctrl_d,
            "primary_density_exact_next": exact_d,
            "site_occupations_controller_next": [float(x) for x in ctrl_site],
            "site_occupations_exact_next": [float(x) for x in exact_site],
            "doublon_controller_next": 0.0,
            "doublon_exact_next": 0.0,
            "energy_total_controller_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": abs(ctrl_d - exact_d),
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": float(
                max(abs(ctrl_site[0] - exact_site[0]), abs(ctrl_site[1] - exact_site[1]))
            ),
            "abs_energy_total_error_next": 0.0,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    first, _forecasts, score = controller._exact_forecast_rollout(
        time_stop=0.2,
        dt=0.1,
        executor=controller.current_executor,
        theta_runtime_start=np.asarray(controller.current_theta, dtype=float),
        theta_dot_step=np.asarray([0.0], dtype=float),
    )

    assert float(first["tracking_d_curvature_abs_error_mean"]) == pytest.approx(1.0)
    assert float(first["tracking_d_curvature_weight"]) == pytest.approx(1.0)
    assert float(first["tracking_d_excursion_under_weight"]) == pytest.approx(0.0)
    assert float(first["tracking_d_excursion_over_weight"]) == pytest.approx(0.0)
    assert float(score) == pytest.approx(1.0)


def test_exact_forecast_rollout_omits_shadow_d_shape_metrics_when_compare_diag_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _toy_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_v1_postcross_compare_diag=False,
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_density_postcross_wrong_sign_weight=0.0,
            exact_forecast_drive_harmonic_weight=0.0,
            exact_forecast_energy_slope_weight=0.0,
            exact_forecast_energy_curvature_weight=0.0,
            exact_forecast_energy_excursion_under_weight=0.0,
            exact_forecast_energy_excursion_over_weight=0.0,
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.3,
        num_times=4,
    )

    def _fake_exact_step_forecast(**kwargs):
        time_stop = float(kwargs["time_stop"])
        exact_site = [1.0, 0.0] if time_stop >= 0.2 else [0.5, 0.5]
        ctrl_site = [0.9, 0.2] if time_stop >= 0.2 else [0.5, 0.5]
        exact_d = float(exact_site[0] - exact_site[1])
        ctrl_d = float(ctrl_site[0] - ctrl_site[1])
        return {
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": ctrl_d,
            "primary_density_exact_next": exact_d,
            "site_occupations_controller_next": [float(x) for x in ctrl_site],
            "site_occupations_exact_next": [float(x) for x in exact_site],
            "doublon_controller_next": 0.0,
            "doublon_exact_next": 0.0,
            "energy_total_controller_next": 0.0,
            "energy_total_exact_next": 0.0,
            "abs_primary_density_error_next": abs(ctrl_d - exact_d),
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": float(
                max(abs(ctrl_site[0] - exact_site[0]), abs(ctrl_site[1] - exact_site[1]))
            ),
            "abs_energy_total_error_next": 0.0,
        }

    monkeypatch.setattr(controller, "_exact_step_forecast", _fake_exact_step_forecast)

    first, _forecasts, _score = controller._exact_forecast_rollout(
        time_stop=0.2,
        dt=0.1,
        executor=controller.current_executor,
        theta_runtime_start=np.asarray(controller.current_theta, dtype=float),
        theta_dot_step=np.asarray([0.0], dtype=float),
    )

    assert "tracking_d_curvature_abs_error_mean" not in first
    assert "tracking_d_excursion_under_response_mean" not in first
    assert "tracking_d_excursion_over_response_mean" not in first
    assert "tracking_total_occupation_abs_error_next" not in first
    assert "tracking_total_occupation_abs_error_mean" not in first


def test_exact_forecast_tracking_score_adds_drive_harmonic_term() -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_fidelity_defect_weight=0.0,
            exact_forecast_tracking_primary_density_error_weight=0.0,
            exact_forecast_tracking_doublon_error_weight=0.0,
            exact_forecast_tracking_site_occupations_error_weight=0.0,
            exact_forecast_tracking_energy_total_error_weight=0.0,
            exact_forecast_density_slope_weight=0.0,
            exact_forecast_density_sign_lag_weight=0.0,
            exact_forecast_drive_harmonic_weight=2.0,
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
    forecasts = [
        {
            "time_stop_next": 0.0,
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": 1.0,
            "primary_density_exact_next": 1.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.0,
        },
        {
            "time_stop_next": float(np.pi / 2.0),
            "fidelity_exact_next": 1.0,
            "primary_density_controller_next": 0.0,
            "primary_density_exact_next": 1.0,
            "abs_primary_density_error_next": 1.0,
            "abs_doublon_error_next": 0.0,
            "site_occupations_abs_error_max_next": 0.0,
            "abs_energy_total_error_next": 0.0,
        },
    ]

    z_ctrl = 0.5 * 1.0 * np.exp(-1j * 0.0) + 0.5 * 0.0 * np.exp(-1j * (np.pi / 2.0))
    z_exact = 0.5 * 1.0 * np.exp(-1j * 0.0) + 0.5 * 1.0 * np.exp(-1j * (np.pi / 2.0))
    mismatch = float((abs(z_ctrl - z_exact) ** 2) / (1.0e-8 + abs(z_exact) ** 2))

    score = controller._forecast_tracking_score(forecast=forecasts)

    assert float(score) == pytest.approx(2.0 * mismatch)


def test_stay_forecast_within_exact_v1_bounded_defect_requires_primary_density_slope() -> None:
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
    )
    forecast = {
        "fidelity_exact_next": 0.9995,
        "abs_primary_density_error_next": 5.0e-3,
        "abs_primary_density_slope_error_next": 3.0e-2,
        "abs_staggered_error_next": 5.0e-3,
        "abs_doublon_error_next": 5.0e-4,
        "site_occupations_abs_error_max_next": 5.0e-3,
        "abs_energy_total_error_next": 1.0e-4,
    }

    assert controller._stay_forecast_within_exact_v1_bounded_defect(forecast=forecast) is False

    forecast["abs_primary_density_slope_error_next"] = 5.0e-3

    assert controller._stay_forecast_within_exact_v1_bounded_defect(forecast=forecast) is True
