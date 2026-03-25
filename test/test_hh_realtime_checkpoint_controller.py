from __future__ import annotations

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
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    RealtimeCheckpointConfig,
    make_checkpoint_context,
)
from pipelines.hardcoded.hh_realtime_measurement import ExactCheckpointValueCache
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
        cfg=SimpleNamespace(reps=1),
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
    assert int(result.ledger[0]["exact_cache_hits"]) >= 1
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
    baseline = controller._baseline_geometry(checkpoint_ctx, cache)
    shortlist = controller._scout_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        baseline=baseline,
    )
    confirmed = controller._confirm_candidates(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
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
