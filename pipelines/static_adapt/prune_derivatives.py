"""Derivative propagation helpers for static-ADAPT prune diagnostics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import _rotation_triplet
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor

__all__ = ["_propagate_runtime_prune_derivatives"]


def _propagate_runtime_prune_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref_state: np.ndarray,
    active_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    expected_runtime_count = int(
        getattr(executor, "runtime_parameter_count", theta_vec.size)
    )
    if int(theta_vec.size) != expected_runtime_count:
        raise ValueError(
            "runtime theta length mismatch for prune Schur derivative propagation: "
            f"got {int(theta_vec.size)}, expected {expected_runtime_count}."
        )
    active = [int(i) for i in active_indices]
    active_map = {
        int(global_idx): int(local_idx)
        for local_idx, global_idx in enumerate(active)
    }
    runtime_steps: list[tuple[int, Any]] = []
    for plan in getattr(executor, "_plans", []):
        for local_idx, step in enumerate(getattr(plan, "steps", ())):
            runtime_steps.append((int(plan.runtime_start + local_idx), step))
    runtime_steps.sort(key=lambda item: int(item[0]))
    psi = np.asarray(psi_ref_state, dtype=complex).reshape(-1).copy()
    n_active = int(len(active))
    dpsi = [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
    d2psi = [
        [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
        for __ in range(n_active)
    ]
    if not runtime_steps:
        return np.asarray(psi, dtype=complex), dpsi, d2psi
    for runtime_idx, step in runtime_steps:
        theta_k = float(theta_vec[int(runtime_idx)])
        local = active_map.get(int(runtime_idx), None)
        old_psi = psi
        old_dpsi = dpsi
        old_d2psi = d2psi

        psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, theta_k)
        psi = psi_u

        next_dpsi: list[np.ndarray] = []
        d_old: list[np.ndarray] = []
        for idx in range(n_active):
            vec_u, vec_d, _vec_s = _rotation_triplet(old_dpsi[idx], step, theta_k)
            next_dpsi.append(vec_u)
            d_old.append(vec_d)
        if local is not None:
            next_dpsi[int(local)] = np.asarray(
                next_dpsi[int(local)] + psi_d,
                dtype=complex,
            )

        next_d2psi: list[list[np.ndarray]] = [
            [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
            for __ in range(n_active)
        ]
        for row in range(n_active):
            for col in range(n_active):
                vec_u, _vec_d, _vec_s = _rotation_triplet(
                    old_d2psi[row][col],
                    step,
                    theta_k,
                )
                updated = vec_u
                if local is not None:
                    if row == int(local):
                        updated = np.asarray(updated + d_old[col], dtype=complex)
                    if col == int(local):
                        updated = np.asarray(updated + d_old[row], dtype=complex)
                    if row == int(local) and col == int(local):
                        updated = np.asarray(updated + psi_s, dtype=complex)
                next_d2psi[row][col] = np.asarray(updated, dtype=complex)
        dpsi = next_dpsi
        d2psi = next_d2psi
    return np.asarray(psi, dtype=complex), dpsi, d2psi
