"""Generate post-run reference energy caches for AP-McLachlan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.observables import (
    build_site_doublon_observable_plan,
    observable_row_fields,
)
from pipelines.time_dynamics.ap_mclachlan.reference_diagnostics import (
    REFERENCE_TIME_MATCH_EXACT_V1,
    ReferenceEnergyPoint,
    ReferenceEnergyTrajectory,
)
from pipelines.time_dynamics.ap_mclachlan.state import APMcLachlanState
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


REFERENCE_KIND_EXACT_INITIAL_STATE_V1 = "exact_initial_state_v1"
REFERENCE_KIND_SEED_PREPARED_STATE_V1 = "seed_prepared_state_v1"
REFERENCE_METHOD_AUTO = "auto"
REFERENCE_METHOD_STATIC_SPECTRAL = "static_spectral"
REFERENCE_METHOD_SOLVE_IVP_DOP853 = "solve_ivp_dop853"


@dataclass(frozen=True)
class ReferenceEnergyGenerationConfig:
    """Settings for dense reference-energy generation used in reporting."""

    reference_kind: str = REFERENCE_KIND_EXACT_INITIAL_STATE_V1
    method: str = REFERENCE_METHOD_AUTO
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_internal_step: float | None = None
    norm_drift_tolerance: float = 1.0e-8
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.reference_kind) not in {
            REFERENCE_KIND_EXACT_INITIAL_STATE_V1,
            REFERENCE_KIND_SEED_PREPARED_STATE_V1,
        }:
            raise ValueError(f"Unsupported reference_kind: {self.reference_kind!r}")
        if str(self.method) not in {
            REFERENCE_METHOD_AUTO,
            REFERENCE_METHOD_STATIC_SPECTRAL,
            REFERENCE_METHOD_SOLVE_IVP_DOP853,
        }:
            raise ValueError(f"Unsupported reference generation method: {self.method!r}")
        for name, value in (("rtol", self.rtol), ("atol", self.atol), ("norm_drift_tolerance", self.norm_drift_tolerance)):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.max_internal_step is not None and (
            not np.isfinite(float(self.max_internal_step)) or float(self.max_internal_step) <= 0.0
        ):
            raise ValueError("max_internal_step must be positive when provided.")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "reference_kind": str(self.reference_kind),
            "method": str(self.method),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "max_internal_step": (
                None if self.max_internal_step is None else float(self.max_internal_step)
            ),
            "norm_drift_tolerance": float(self.norm_drift_tolerance),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class _DenseHamiltonianProvider:
    static_matrix: np.ndarray
    drive_matrix: np.ndarray | None
    hamiltonian: TimeDependentHamiltonian

    def matrix_at(self, time_value: float) -> np.ndarray:
        if self.drive_matrix is None:
            return self.static_matrix
        coeff = float(self.hamiltonian.drive_coefficient_at(float(time_value)))
        return self.static_matrix + coeff * self.drive_matrix


def generate_reference_energy_trajectory(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    times: Sequence[float],
    config: ReferenceEnergyGenerationConfig = ReferenceEnergyGenerationConfig(),
    metadata: Mapping[str, Any] | None = None,
) -> ReferenceEnergyTrajectory:
    """Generate an exact-reference energy cache for post-run reporting."""

    time_grid = _time_grid(times)
    dense_hamiltonian = _dense_hamiltonian_provider(hamiltonian)
    psi0, initial_meta = _initial_reference_state(
        state=state,
        hmat=np.asarray(dense_hamiltonian.static_matrix, dtype=complex),
        config=config,
    )
    method = _resolve_method(hamiltonian=hamiltonian, config=config)
    if method == REFERENCE_METHOD_STATIC_SPECTRAL:
        states = _static_reference_states(
            psi0=psi0,
            hmat=np.asarray(dense_hamiltonian.matrix_at(float(time_grid[0])), dtype=complex),
            times=time_grid,
        )
    else:
        states = _driven_reference_states(
            psi0=psi0,
            hamiltonian=dense_hamiltonian,
            times=time_grid,
            config=config,
        )
    norm_drift_max = float(
        max(abs(float(np.linalg.norm(psi)) - 1.0) for psi in states)
    )
    if norm_drift_max > float(config.norm_drift_tolerance):
        raise ValueError(
            "Reference propagation norm drift exceeded tolerance: "
            f"{norm_drift_max:.3e} > {float(config.norm_drift_tolerance):.3e}."
        )
    observable_plan = build_site_doublon_observable_plan(
        state.resolved_problem,
        dimension=int(np.asarray(psi0, dtype=complex).reshape(-1).size),
    )
    points = [
        ReferenceEnergyPoint(
            time=float(time_value),
            energy=_energy_expectation(
                psi=np.asarray(psi, dtype=complex),
                hmat=np.asarray(dense_hamiltonian.matrix_at(float(time_value)), dtype=complex),
            ),
            source=str(initial_meta["point_source"]),
            observables=observable_row_fields(
                np.asarray(psi, dtype=complex),
                plan=observable_plan,
            ),
            metadata={"point_index": int(index)},
        )
        for index, (time_value, psi) in enumerate(zip(time_grid, states))
    ]
    return ReferenceEnergyTrajectory(
        points=tuple(points),
        source=str(initial_meta["trajectory_source"]),
        match_kind=REFERENCE_TIME_MATCH_EXACT_V1,
        metadata={
            "reference_scope": "post_run_reporting",
            "reference_kind": str(config.reference_kind),
            "propagator_method": str(method),
            "time_initial": float(time_grid[0]),
            "time_final": float(time_grid[-1]),
            "point_count": int(len(time_grid)),
            "state_parameterization_mode": str(state.parameterization_mode),
            "state_parameterization_label": str(state.parameterization_label),
            "state_active_parameter_count": int(state.active_parameter_count),
            "state_runtime_parameter_count": int(state.runtime_parameter_count),
            "state_runtime_pauli_parameter_count": int(state.runtime_pauli_parameter_count),
            "state_logical_parameter_count": int(state.logical_parameter_count),
            "drive_enabled": bool(hamiltonian.drive_enabled),
            "norm_drift_max": float(norm_drift_max),
            "reference_observable_support": (
                "hh_site_doublon"
                if observable_plan is not None
                else "not_available_for_problem_family"
            ),
            "reference_observable_evaluation_policy": (
                "same_exact_statevector_single_pass"
                if observable_plan is not None
                else None
            ),
            "generation_config": config.to_json_dict(),
            **dict(initial_meta["metadata"]),
            **dict(metadata or {}),
        },
    )


def generate_seed_reference_energy_trajectory(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    times: Sequence[float],
    config: ReferenceEnergyGenerationConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReferenceEnergyTrajectory:
    """Backward-compatible wrapper for the explicit seed-prepared reference."""

    resolved_config = (
        ReferenceEnergyGenerationConfig(reference_kind=REFERENCE_KIND_SEED_PREPARED_STATE_V1)
        if config is None
        else config
    )
    return generate_reference_energy_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=times,
        config=resolved_config,
        metadata=metadata,
    )


def _dense_hamiltonian_provider(
    hamiltonian: TimeDependentHamiltonian,
) -> _DenseHamiltonianProvider:
    static_matrix = np.asarray(hamiltonian_matrix(hamiltonian.static_poly), dtype=complex)
    drive_matrix = None
    if hamiltonian.drive_poly is not None:
        drive_matrix = np.asarray(hamiltonian_matrix(hamiltonian.drive_poly), dtype=complex)
        if drive_matrix.shape != static_matrix.shape:
            raise ValueError(
                "Drive Hamiltonian matrix shape does not match static Hamiltonian: "
                f"{drive_matrix.shape} vs {static_matrix.shape}."
            )
    return _DenseHamiltonianProvider(
        static_matrix=static_matrix,
        drive_matrix=drive_matrix,
        hamiltonian=hamiltonian,
    )


def _initial_reference_state(
    *,
    state: APMcLachlanState,
    hmat: np.ndarray,
    config: ReferenceEnergyGenerationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    kind = str(config.reference_kind)
    if kind == REFERENCE_KIND_SEED_PREPARED_STATE_V1:
        psi0 = state.prepare_state(state.theta_runtime)
        energy = _energy_expectation(psi=np.asarray(psi0, dtype=complex), hmat=hmat)
        return (
            psi0,
            {
                "point_source": "seed_prepared_reference",
                "trajectory_source": "seed_prepared_reference_energy",
                "metadata": {
                    "initial_reference_state_source": "imported_seed_prepared_state",
                    "initial_reference_energy_static": float(energy),
                    "exact_energy_target": None if state.exact_energy is None else float(state.exact_energy),
                    "exact_energy_target_delta": (
                        None
                        if state.exact_energy is None
                        else float(abs(float(energy) - float(state.exact_energy)))
                    ),
                },
            },
        )
    if kind != REFERENCE_KIND_EXACT_INITIAL_STATE_V1:
        raise ValueError(f"Unsupported reference_kind: {kind!r}")
    evals, evecs = np.linalg.eigh(np.asarray(hmat, dtype=complex))
    target = state.exact_energy
    if target is None:
        index = int(np.argmin(evals))
        source = "dense_static_ground_state_lowest_eigenvalue"
    else:
        deltas = np.abs(np.asarray(evals, dtype=float) - float(target))
        index = int(np.argmin(deltas))
        source = "dense_static_eigenstate_nearest_imported_exact_energy"
    psi0 = np.asarray(evecs[:, index], dtype=complex).reshape(-1)
    energy = float(np.real(evals[index]))
    delta = None if target is None else float(abs(energy - float(target)))
    return (
        psi0,
        {
            "point_source": "exact_initial_state_reference",
            "trajectory_source": "exact_initial_state_reference_energy",
            "metadata": {
                "initial_reference_state_source": source,
                "initial_reference_energy_static": float(energy),
                "exact_energy_target": None if target is None else float(target),
                "exact_energy_target_delta": delta,
                "exact_energy_target_eigen_index": int(index),
            },
        },
    )


def _time_grid(times: Sequence[float]) -> np.ndarray:
    grid = np.asarray(times, dtype=float).reshape(-1)
    if int(grid.size) == 0:
        raise ValueError("Reference generation requires at least one time.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("Reference generation times must be finite.")
    if len(set(float(x) for x in grid)) != int(grid.size):
        raise ValueError("Reference generation times must not contain duplicates.")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("Reference generation times must be monotonically increasing.")
    return grid


def _resolve_method(
    *,
    hamiltonian: TimeDependentHamiltonian,
    config: ReferenceEnergyGenerationConfig,
) -> str:
    if str(config.method) != REFERENCE_METHOD_AUTO:
        return str(config.method)
    if not bool(hamiltonian.drive_enabled):
        return REFERENCE_METHOD_STATIC_SPECTRAL
    return REFERENCE_METHOD_SOLVE_IVP_DOP853


def _static_reference_states(
    *,
    psi0: np.ndarray,
    hmat: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if int(times.size) == 1:
        return (np.asarray(psi0, dtype=complex),)
    t0 = float(times[0])
    evals, evecs = np.linalg.eigh(hmat)
    coeffs = evecs.conj().T @ np.asarray(psi0, dtype=complex).reshape(-1)
    states: list[np.ndarray] = []
    for time_value in times:
        phase = np.exp(-1.0j * evals * (float(time_value) - t0))
        states.append(np.asarray(evecs @ (phase * coeffs), dtype=complex).reshape(-1))
    return tuple(states)


def _driven_reference_states(
    *,
    psi0: np.ndarray,
    hamiltonian: _DenseHamiltonianProvider,
    times: np.ndarray,
    config: ReferenceEnergyGenerationConfig,
) -> tuple[np.ndarray, ...]:
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Driven reference generation requires scipy.") from exc

    if int(times.size) == 1:
        return (np.asarray(psi0, dtype=complex),)
    t0 = float(times[0])
    tf = float(times[-1])
    output_dt = float(np.min(np.diff(times)))
    max_step = (
        min(output_dt / 5.0, 0.005)
        if config.max_internal_step is None
        else float(config.max_internal_step)
    )

    def rhs(time_value: float, psi_value: np.ndarray) -> np.ndarray:
        hmat = np.asarray(hamiltonian.matrix_at(float(time_value)), dtype=complex)
        return np.asarray(-1.0j * (hmat @ psi_value), dtype=complex)

    result = solve_ivp(
        rhs,
        (t0, tf),
        np.asarray(psi0, dtype=complex).reshape(-1),
        method="DOP853",
        t_eval=np.asarray(times, dtype=float),
        rtol=float(config.rtol),
        atol=float(config.atol),
        max_step=float(max_step),
    )
    if not bool(result.success):
        raise RuntimeError(f"Driven reference generation failed: {result.message}")
    return tuple(np.asarray(result.y[:, idx], dtype=complex).reshape(-1) for idx in range(result.y.shape[1]))


def _energy_expectation(*, psi: np.ndarray, hmat: np.ndarray) -> float:
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    return float(np.real(np.vdot(psi_vec, hmat @ psi_vec)))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "REFERENCE_KIND_EXACT_INITIAL_STATE_V1",
    "REFERENCE_KIND_SEED_PREPARED_STATE_V1",
    "REFERENCE_METHOD_AUTO",
    "REFERENCE_METHOD_SOLVE_IVP_DOP853",
    "REFERENCE_METHOD_STATIC_SPECTRAL",
    "ReferenceEnergyGenerationConfig",
    "generate_reference_energy_trajectory",
    "generate_seed_reference_energy_trajectory",
]
