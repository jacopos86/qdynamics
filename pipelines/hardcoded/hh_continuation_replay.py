#!/usr/bin/env python3
"""Replay controllers for HH ADAPT-family continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_types import (
    Phase2OptimizerMemoryAdapter,
    QNSPSARefreshPlan,
    ReplayPhaseTelemetry,
    ReplayPlan,
)


@dataclass(frozen=True)
class ReplayControllerConfig:
    freeze_fraction: float = 0.2
    unfreeze_fraction: float = 0.3
    full_fraction: float = 0.5
    trust_radius_initial: float = 0.1
    trust_radius_growth: float = 2.0
    trust_radius_max: float = 0.4
    qn_spsa_refresh_every: int = 0
    qn_spsa_refresh_mode: str = "diag_rms_grad"
    symmetry_mitigation_mode: str = "off"


class RestrictedAnsatzView:
    """Ansatz view that masks parameters outside an active set."""

    def __init__(
        self,
        *,
        base_ansatz: Any,
        base_point: np.ndarray,
        active_indices: Sequence[int],
    ) -> None:
        self._base_ansatz = base_ansatz
        self._base_point = np.asarray(base_point, dtype=float).reshape(-1)
        self._active = [int(i) for i in active_indices]
        self.num_parameters = int(len(self._active))

    def _merge(self, x: np.ndarray) -> np.ndarray:
        merged = np.array(self._base_point, copy=True)
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        for k, idx in enumerate(self._active):
            merged[idx] = float(x_arr[k])
        return merged

    def prepare_state(self, x: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        return self._base_ansatz.prepare_state(self._merge(x), psi_ref)

    def parameter_labels(self) -> list[str]:
        if hasattr(self._base_ansatz, "parameter_labels"):
            labels = list(getattr(self._base_ansatz, "parameter_labels")())
        else:
            labels = [f"theta_{i}" for i in range(int(self._base_point.size))]
        return [str(labels[i]) for i in self._active]


def build_replay_plan(
    *,
    continuation_mode: str,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    scaffold_block_indices: Sequence[int],
    residual_block_indices: Sequence[int],
    maxiter: int,
    cfg: ReplayControllerConfig,
    symmetry_mitigation_mode: str = "off",
    generator_ids: Sequence[str] | None = None,
    motif_reference_ids: Sequence[str] | None = None,
) -> ReplayPlan:
    total = max(3, int(maxiter))
    freeze_steps = max(1, int(round(float(cfg.freeze_fraction) * total)))
    unfreeze_steps = max(1, int(round(float(cfg.unfreeze_fraction) * total)))
    full_steps = max(1, total - freeze_steps - unfreeze_steps)
    trust_initial = float(cfg.trust_radius_initial)
    trust_growth = float(max(cfg.trust_radius_growth, 1.0))
    trust_max = float(max(cfg.trust_radius_max, trust_initial))
    trust = [
        trust_initial,
        min(trust_max, trust_initial * trust_growth),
        trust_max,
    ]
    return ReplayPlan(
        continuation_mode=str(continuation_mode),
        seed_policy_resolved=str(seed_policy_resolved),
        handoff_state_kind=str(handoff_state_kind),
        freeze_scaffold_steps=int(freeze_steps),
        unfreeze_steps=int(unfreeze_steps),
        full_replay_steps=int(full_steps),
        trust_radius_initial=float(trust_initial),
        trust_radius_growth=float(trust_growth),
        trust_radius_max=float(trust_max),
        scaffold_block_indices=[int(i) for i in scaffold_block_indices],
        residual_block_indices=[int(i) for i in residual_block_indices],
        qn_spsa_refresh_every=int(max(0, cfg.qn_spsa_refresh_every)),
        trust_radius_schedule=trust,
        optimizer_memory_source="unavailable",
        optimizer_memory_reused=False,
        refresh_mode=(
            str(cfg.qn_spsa_refresh_mode)
            if int(max(0, cfg.qn_spsa_refresh_every)) > 0
            else "disabled"
        ),
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        generator_ids=[str(x) for x in list(generator_ids or [])],
        motif_reference_ids=[str(x) for x in list(motif_reference_ids or [])],
    )


def _run_phase(
    *,
    phase_name: str,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    active_indices: list[int],
    full_theta_seed: np.ndarray,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None,
    kwargs: dict[str, Any],
    optimizer_memory: Mapping[str, Any] | None,
    refresh_plan: QNSPSARefreshPlan,
) -> tuple[np.ndarray, ReplayPhaseTelemetry, Any]:
    view = RestrictedAnsatzView(
        base_ansatz=ansatz,
        base_point=np.asarray(full_theta_seed, dtype=float),
        active_indices=active_indices,
    )
    x0 = np.asarray(full_theta_seed, dtype=float)[active_indices] if active_indices else np.zeros(0, dtype=float)
    result = vqe_minimize_fn(
        h_poly,
        view,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        initial_point=x0,
        use_initial_point_first_restart=True,
        method=str(method),
        maxiter=int(maxiter),
        progress_every_s=float(progress_every_s),
        track_history=True,
        optimizer_memory=(dict(optimizer_memory) if isinstance(optimizer_memory, Mapping) else None),
        spsa_refresh_every=(int(refresh_plan.refresh_every) if bool(refresh_plan.enabled) else 0),
        spsa_precondition_mode=(str(refresh_plan.mode) if bool(refresh_plan.enabled) else "none"),
        **kwargs,
    )
    x_opt = np.asarray(result.x, dtype=float).reshape(-1)
    full = np.asarray(full_theta_seed, dtype=float).copy()
    for k, idx in enumerate(active_indices):
        full[idx] = float(x_opt[k])

    e_before = float(result.restart_summaries[0]["best_energy"]) if getattr(result, "restart_summaries", None) else float(result.energy)
    e_after = float(result.energy)
    d_before = float(abs(e_before - exact_energy)) if exact_energy is not None else None
    d_after = float(abs(e_after - exact_energy)) if exact_energy is not None else None
    tel = ReplayPhaseTelemetry(
        phase_name=str(phase_name),
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        success=bool(getattr(result, "success", False)),
        energy_before=float(e_before),
        energy_after=float(e_after),
        delta_abs_before=d_before,
        delta_abs_after=d_after,
        active_count=int(len(active_indices)),
        frozen_count=int(len(full) - len(active_indices)),
        optimizer_memory_reused=bool(
            isinstance(optimizer_memory, Mapping) and bool(optimizer_memory.get("reused", False))
        ),
        optimizer_memory_source=str(
            optimizer_memory.get("source", "unavailable")
            if isinstance(optimizer_memory, Mapping)
            else "unavailable"
        ),
        qn_spsa_refresh_points=[
            int(x) for x in getattr(result, "optimizer_memory", {}) .get("refresh_points", [])
        ] if hasattr(result, "optimizer_memory") and isinstance(getattr(result, "optimizer_memory"), Mapping) else [],
    )
    return full, tel, result


def _refresh_plan_for_phase(
    *,
    phase_name: str,
    method: str,
    cfg: ReplayControllerConfig,
) -> QNSPSARefreshPlan:
    if str(method).strip().lower() != "spsa":
        return QNSPSARefreshPlan(enabled=False, refresh_every=0, mode="disabled", skip_reason="method_not_spsa")
    if str(phase_name) != "constrained_unfreeze":
        return QNSPSARefreshPlan(enabled=False, refresh_every=0, mode="disabled", skip_reason="phase_not_refreshed")
    cadence = int(max(0, cfg.qn_spsa_refresh_every))
    if cadence <= 0:
        return QNSPSARefreshPlan(enabled=False, refresh_every=0, mode="disabled", skip_reason="refresh_disabled")
    return QNSPSARefreshPlan(
        enabled=True,
        refresh_every=int(cadence),
        mode=str(cfg.qn_spsa_refresh_mode),
    )


def _seed_full_optimizer_memory(
    *,
    adapter: Phase2OptimizerMemoryAdapter,
    incoming_memory: Mapping[str, Any] | None,
    total_parameters: int,
    scaffold_block_size: int,
) -> tuple[dict[str, Any], str, bool]:
    total_n = int(max(0, total_parameters))
    scaffold_n = int(max(0, min(scaffold_block_size, total_n)))
    incoming_n = int(incoming_memory.get("parameter_count", 0)) if isinstance(incoming_memory, Mapping) else 0
    if not isinstance(incoming_memory, Mapping):
        return (
            adapter.unavailable(method="SPSA", parameter_count=total_n, reason="missing_handoff_optimizer_memory"),
            "missing_handoff_optimizer_memory",
            False,
        )
    if incoming_n == total_n:
        state = adapter.from_result(
            type("MemoryCarrier", (), {"optimizer_memory": dict(incoming_memory)})(),
            method=str(incoming_memory.get("optimizer", "SPSA")),
            parameter_count=int(total_n),
            source="handoff_full",
        )
        state["reused"] = bool(state.get("available", False))
        return state, "handoff_full", bool(state.get("reused", False))
    if incoming_n == scaffold_n:
        base = adapter.unavailable(method="SPSA", parameter_count=total_n, reason="replay_block_expansion")
        scaffold_state = adapter.from_result(
            type("MemoryCarrier", (), {"optimizer_memory": dict(incoming_memory)})(),
            method=str(incoming_memory.get("optimizer", "SPSA")),
            parameter_count=int(scaffold_n),
            source="handoff_scaffold",
        )
        merged = adapter.merge_active(
            base,
            active_indices=list(range(scaffold_n)),
            active_state=scaffold_state,
            source="handoff_scaffold_expand",
        )
        merged["reused"] = bool(scaffold_state.get("available", False))
        return merged, "handoff_scaffold_expand", bool(merged.get("reused", False))
    normalized = adapter.from_result(
        type("MemoryCarrier", (), {"optimizer_memory": dict(incoming_memory)})(),
        method=str(incoming_memory.get("optimizer", "SPSA")),
        parameter_count=int(total_n),
        source="handoff_resized",
    )
    normalized["reused"] = bool(normalized.get("available", False))
    return normalized, "handoff_resized", bool(normalized.get("reused", False))


def _run_phase_replay_controller(
    *,
    continuation_mode: str,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
    incoming_optimizer_memory: Mapping[str, Any] | None = None,
    symmetry_mitigation_mode: str = "off",
    generator_ids: Sequence[str] | None = None,
    motif_reference_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    extra = dict(kwargs or {})
    theta_cur = np.asarray(seed_theta, dtype=float).reshape(-1).copy()
    npar = int(theta_cur.size)
    scaffold_n = max(0, min(int(scaffold_block_size), npar))
    residual_indices = [i for i in range(scaffold_n, npar)]
    scaffold_indices = [i for i in range(scaffold_n)]
    plan = build_replay_plan(
        continuation_mode=str(continuation_mode),
        seed_policy_resolved=str(seed_policy_resolved),
        handoff_state_kind=str(handoff_state_kind),
        scaffold_block_indices=scaffold_indices,
        residual_block_indices=residual_indices,
        maxiter=int(maxiter),
        cfg=cfg,
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        generator_ids=list(generator_ids or []),
        motif_reference_ids=list(motif_reference_ids or []),
    )
    history: list[dict[str, Any]] = []
    adapter = Phase2OptimizerMemoryAdapter()
    full_memory, memory_source, memory_reused = _seed_full_optimizer_memory(
        adapter=adapter,
        incoming_memory=incoming_optimizer_memory,
        total_parameters=int(npar),
        scaffold_block_size=int(scaffold_n),
    )
    plan = ReplayPlan(
        **{
            **plan.__dict__,
            "optimizer_memory_source": str(memory_source),
            "optimizer_memory_reused": bool(memory_reused),
        }
    )
    refresh_points_total: list[int] = []
    refresh_plans: list[QNSPSARefreshPlan] = []

    phase_specs = [
        ("seed_burn_in", residual_indices, int(plan.freeze_scaffold_steps), 1, int(seed)),
        (
            "constrained_unfreeze",
            scaffold_indices[-max(0, min(len(scaffold_indices), max(1, len(scaffold_indices) // 3))):] + residual_indices,
            int(plan.unfreeze_steps),
            1,
            int(seed) + 1,
        ),
        ("full_replay", list(range(npar)), int(plan.full_replay_steps), int(restarts), int(seed) + 2),
    ]

    last_result: Any = None
    for phase_name, active_indices, phase_steps, phase_restarts, phase_seed in phase_specs:
        active_memory = adapter.select_active(
            full_memory,
            active_indices=list(active_indices),
            source=f"{continuation_mode}.{phase_name}.active_subset",
        )
        refresh_plan = _refresh_plan_for_phase(
            phase_name=str(phase_name),
            method=str(method),
            cfg=cfg,
        )
        refresh_plans.append(refresh_plan)
        theta_cur, tel, result = _run_phase(
            phase_name=str(phase_name),
            vqe_minimize_fn=vqe_minimize_fn,
            h_poly=h_poly,
            ansatz=ansatz,
            psi_ref=psi_ref,
            active_indices=list(active_indices),
            full_theta_seed=theta_cur,
            restarts=int(phase_restarts),
            seed=int(phase_seed),
            maxiter=int(phase_steps),
            method=str(method),
            progress_every_s=float(progress_every_s),
            exact_energy=exact_energy,
            kwargs=extra,
            optimizer_memory=active_memory,
            refresh_plan=refresh_plan,
        )
        last_result = result
        merged_memory = adapter.from_result(
            result,
            method=str(method),
            parameter_count=int(len(active_indices)),
            source=f"{continuation_mode}.{phase_name}.result",
        )
        full_memory = adapter.merge_active(
            full_memory,
            active_indices=list(active_indices),
            active_state=merged_memory,
            source=f"{continuation_mode}.{phase_name}.merge",
        )
        refresh_points = [
            int(x)
            for x in (
                getattr(result, "optimizer_memory", {}).get("refresh_points", [])
                if hasattr(result, "optimizer_memory") and isinstance(getattr(result, "optimizer_memory"), Mapping)
                else []
            )
        ]
        refresh_points_total.extend(x for x in refresh_points if x not in refresh_points_total)
        history.append(tel.__dict__)

    replay_meta = {
        "replay_phase_config": {
            "continuation_mode": str(plan.continuation_mode),
            "seed_policy_resolved": str(seed_policy_resolved),
            "handoff_state_kind": str(handoff_state_kind),
            "freeze_scaffold_steps": int(plan.freeze_scaffold_steps),
            "unfreeze_steps": int(plan.unfreeze_steps),
            "full_replay_steps": int(plan.full_replay_steps),
            "trust_radius_initial": float(plan.trust_radius_initial),
            "trust_radius_growth": float(plan.trust_radius_growth),
            "trust_radius_max": float(plan.trust_radius_max),
            "scaffold_block_indices": [int(i) for i in plan.scaffold_block_indices],
            "residual_block_indices": [int(i) for i in plan.residual_block_indices],
            "qn_spsa_refresh_every": int(plan.qn_spsa_refresh_every),
            "trust_radius_schedule": [float(x) for x in plan.trust_radius_schedule],
            "optimizer_memory_source": str(plan.optimizer_memory_source),
            "optimizer_memory_reused": bool(plan.optimizer_memory_reused),
            "symmetry_mitigation_mode": str(plan.symmetry_mitigation_mode),
            "generator_ids": [str(x) for x in plan.generator_ids],
            "motif_reference_ids": [str(x) for x in plan.motif_reference_ids],
            "optimizer_memory": dict(full_memory),
            "residual_zero_initialized": True,
            "qn_spsa_refresh": {
                "enabled": any(bool(rp.enabled) for rp in refresh_plans),
                "refresh_every": int(plan.qn_spsa_refresh_every),
                "mode": str(plan.refresh_mode),
                "refresh_points": [int(x) for x in refresh_points_total],
                "phase_plans": [dict(rp.__dict__) for rp in refresh_plans],
            },
        },
        "result": {
            "energy": float(getattr(last_result, "energy", float("nan"))),
            "nfev": int(getattr(last_result, "nfev", 0)),
            "nit": int(getattr(last_result, "nit", 0)),
            "success": bool(getattr(last_result, "success", False)),
            "message": str(getattr(last_result, "message", "")),
        },
    }
    return theta_cur, history, replay_meta


def run_phase1_replay(
    *,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Built-in math expression:
    replay = burn_in(residual) -> constrained_unfreeze -> full
    """
    phase1_cfg = ReplayControllerConfig(
        freeze_fraction=float(cfg.freeze_fraction),
        unfreeze_fraction=float(cfg.unfreeze_fraction),
        full_fraction=float(cfg.full_fraction),
        trust_radius_initial=float(cfg.trust_radius_initial),
        trust_radius_growth=float(cfg.trust_radius_growth),
        trust_radius_max=float(cfg.trust_radius_max),
        qn_spsa_refresh_every=0,
        qn_spsa_refresh_mode="disabled",
    )
    return _run_phase_replay_controller(
        continuation_mode="phase1_v1",
        vqe_minimize_fn=vqe_minimize_fn,
        h_poly=h_poly,
        ansatz=ansatz,
        psi_ref=psi_ref,
        seed_theta=seed_theta,
        scaffold_block_size=scaffold_block_size,
        seed_policy_resolved=seed_policy_resolved,
        handoff_state_kind=handoff_state_kind,
        cfg=phase1_cfg,
        restarts=restarts,
        seed=seed,
        maxiter=maxiter,
        method=method,
        progress_every_s=progress_every_s,
        exact_energy=exact_energy,
        kwargs=kwargs,
        incoming_optimizer_memory=None,
        symmetry_mitigation_mode="off",
        generator_ids=None,
        motif_reference_ids=None,
    )


def run_phase2_replay(
    *,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
    incoming_optimizer_memory: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    return _run_phase_replay_controller(
        continuation_mode="phase2_v1",
        vqe_minimize_fn=vqe_minimize_fn,
        h_poly=h_poly,
        ansatz=ansatz,
        psi_ref=psi_ref,
        seed_theta=seed_theta,
        scaffold_block_size=scaffold_block_size,
        seed_policy_resolved=seed_policy_resolved,
        handoff_state_kind=handoff_state_kind,
        cfg=cfg,
        restarts=restarts,
        seed=seed,
        maxiter=maxiter,
        method=method,
        progress_every_s=progress_every_s,
        exact_energy=exact_energy,
        kwargs=kwargs,
        incoming_optimizer_memory=incoming_optimizer_memory,
        symmetry_mitigation_mode=str(cfg.symmetry_mitigation_mode),
        generator_ids=None,
        motif_reference_ids=None,
    )


def run_phase3_replay(
    *,
    vqe_minimize_fn: Callable[..., Any],
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    seed_theta: np.ndarray,
    scaffold_block_size: int,
    seed_policy_resolved: str,
    handoff_state_kind: str,
    cfg: ReplayControllerConfig,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    progress_every_s: float,
    exact_energy: float | None = None,
    kwargs: dict[str, Any] | None = None,
    incoming_optimizer_memory: Mapping[str, Any] | None = None,
    generator_ids: Sequence[str] | None = None,
    motif_reference_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    return _run_phase_replay_controller(
        continuation_mode="phase3_v1",
        vqe_minimize_fn=vqe_minimize_fn,
        h_poly=h_poly,
        ansatz=ansatz,
        psi_ref=psi_ref,
        seed_theta=seed_theta,
        scaffold_block_size=scaffold_block_size,
        seed_policy_resolved=seed_policy_resolved,
        handoff_state_kind=handoff_state_kind,
        cfg=cfg,
        restarts=restarts,
        seed=seed,
        maxiter=maxiter,
        method=method,
        progress_every_s=progress_every_s,
        exact_energy=exact_energy,
        kwargs=kwargs,
        incoming_optimizer_memory=incoming_optimizer_memory,
        symmetry_mitigation_mode=str(cfg.symmetry_mitigation_mode),
        generator_ids=list(generator_ids or []),
        motif_reference_ids=list(motif_reference_ids or []),
    )
