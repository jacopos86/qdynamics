from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_replay import (
    ReplayControllerConfig,
    build_replay_plan,
    run_phase1_replay,
    run_phase2_replay,
    run_phase3_replay,
)


class _DummyAnsatz:
    def __init__(self, npar: int) -> None:
        self.num_parameters = int(npar)

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        # Deterministic, normalized return for tests.
        out = np.array(psi_ref, copy=True)
        out[0] = complex(1.0 + 0.0 * float(np.sum(theta)))
        return out / np.linalg.norm(out)


@dataclass
class _DummyRes:
    x: np.ndarray
    energy: float
    nfev: int
    nit: int
    success: bool
    message: str
    best_restart: int = 0
    restart_summaries: list[dict[str, float]] | None = None
    optimizer_memory: dict[str, object] | None = None


def _fake_vqe_minimize(_h, ansatz, _psi_ref, **kwargs):
    x0 = np.asarray(kwargs.get("initial_point", np.zeros(int(ansatz.num_parameters))), dtype=float)
    x = np.array(x0, copy=True)
    e = float(np.sum(x**2))
    return _DummyRes(
        x=x,
        energy=e,
        nfev=5,
        nit=3,
        success=True,
        message="ok",
        best_restart=0,
        restart_summaries=[{"best_energy": e + 0.1}],
        optimizer_memory={
            "version": "phase2_spsa_diag_memory_v1",
            "optimizer": "SPSA",
            "parameter_count": int(x.size),
            "available": True,
            "source": "dummy",
            "reused": True,
            "preconditioner_diag": [1.0] * int(x.size),
            "grad_sq_ema": [0.1] * int(x.size),
            "history_tail": [],
            "refresh_points": [2] if int(kwargs.get("spsa_refresh_every", 0)) > 0 else [],
            "remap_events": [],
        },
    )


def test_build_replay_plan_splits_steps() -> None:
    cfg = ReplayControllerConfig(freeze_fraction=0.2, unfreeze_fraction=0.3, full_fraction=0.5)
    plan = build_replay_plan(
        continuation_mode="phase1_v1",
        seed_policy_resolved="residual_only",
        handoff_state_kind="prepared_state",
        scaffold_block_indices=[0, 1],
        residual_block_indices=[2, 3, 4],
        maxiter=100,
        cfg=cfg,
    )
    assert plan.freeze_scaffold_steps > 0
    assert plan.unfreeze_steps > 0
    assert plan.full_replay_steps > 0
    assert len(plan.trust_radius_schedule) == 3
    assert plan.handoff_state_kind == "prepared_state"
    assert plan.scaffold_block_indices == [0, 1]
    assert plan.residual_block_indices == [2, 3, 4]


def test_run_phase1_replay_emits_phase_history() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase1_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="residual_only",
        handoff_state_kind="prepared_state",
        cfg=ReplayControllerConfig(),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
    )
    assert len(theta) == 6
    assert len(hist) == 3
    assert hist[0]["phase_name"] == "seed_burn_in"
    assert hist[1]["phase_name"] == "constrained_unfreeze"
    assert hist[2]["phase_name"] == "full_replay"
    assert "replay_phase_config" in meta
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["handoff_state_kind"] == "prepared_state"
    assert cfg_out["scaffold_block_indices"] == [0, 1]
    assert cfg_out["residual_block_indices"] == [2, 3, 4, 5]
    assert "qn_spsa_refresh_every" in cfg_out


def test_run_phase2_replay_reuses_memory_and_logs_refresh() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase2_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="scaffold_plus_zero",
        handoff_state_kind="reference_state",
        cfg=ReplayControllerConfig(qn_spsa_refresh_every=2),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
        incoming_optimizer_memory={
            "version": "phase2_optimizer_memory_v1",
            "optimizer": "SPSA",
            "parameter_count": 2,
            "available": True,
            "source": "handoff",
            "reused": True,
            "preconditioner_diag": [1.0, 1.0],
            "grad_sq_ema": [0.1, 0.1],
            "history_tail": [],
            "refresh_points": [],
            "remap_events": [],
        },
    )
    assert len(theta) == 6
    assert len(hist) == 3
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["continuation_mode"] == "phase2_v1"
    assert cfg_out["optimizer_memory_reused"] is True
    assert cfg_out["optimizer_memory_source"] in {"handoff_scaffold_expand", "handoff_full", "handoff_resized"}
    assert cfg_out["qn_spsa_refresh"]["enabled"] is True
    assert cfg_out["qn_spsa_refresh"]["refresh_points"] == [2]
    assert cfg_out["optimizer_memory"]["parameter_count"] == 6


def test_run_phase2_replay_missing_optimizer_memory_degrades_gracefully() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase2_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="scaffold_plus_zero",
        handoff_state_kind="reference_state",
        cfg=ReplayControllerConfig(),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
        incoming_optimizer_memory=None,
    )
    assert len(theta) == 6
    assert len(hist) == 3
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["continuation_mode"] == "phase2_v1"
    assert cfg_out["optimizer_memory_source"] == "missing_handoff_optimizer_memory"
    assert cfg_out["optimizer_memory_reused"] is False
    assert cfg_out["optimizer_memory"]["parameter_count"] == 6
    assert cfg_out["optimizer_memory"]["available"] is True


def test_run_phase3_replay_emits_generator_motif_and_symmetry_fields() -> None:
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    theta, hist, meta = run_phase3_replay(
        vqe_minimize_fn=_fake_vqe_minimize,
        h_poly=None,
        ansatz=_DummyAnsatz(6),
        psi_ref=psi_ref,
        seed_theta=np.zeros(6, dtype=float),
        scaffold_block_size=2,
        seed_policy_resolved="scaffold_plus_zero",
        handoff_state_kind="reference_state",
        cfg=ReplayControllerConfig(
            qn_spsa_refresh_every=2,
            qn_spsa_refresh_mode="diag_rms_grad",
            symmetry_mitigation_mode="verify_only",
        ),
        restarts=2,
        seed=7,
        maxiter=60,
        method="SPSA",
        progress_every_s=60.0,
        exact_energy=None,
        kwargs={},
        incoming_optimizer_memory={
            "version": "phase2_optimizer_memory_v1",
            "optimizer": "SPSA",
            "parameter_count": 2,
            "available": True,
            "source": "handoff",
            "reused": True,
            "preconditioner_diag": [1.0, 1.0],
            "grad_sq_ema": [0.1, 0.1],
            "history_tail": [],
            "refresh_points": [],
            "remap_events": [],
        },
        generator_ids=["gen:a", "gen:b"],
        motif_reference_ids=["motif:1"],
    )
    assert len(theta) == 6
    assert len(hist) == 3
    cfg_out = meta["replay_phase_config"]
    assert cfg_out["continuation_mode"] == "phase3_v1"
    assert cfg_out["symmetry_mitigation_mode"] == "verify_only"
    assert cfg_out["generator_ids"] == ["gen:a", "gen:b"]
    assert cfg_out["motif_reference_ids"] == ["motif:1"]
    assert cfg_out["qn_spsa_refresh"]["enabled"] is True
