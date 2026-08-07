"""Coordinate-descent optimizers for variational quantum refits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CoordinateDescentResult:
    """Small scipy-like result container used by ADAPT refit code."""

    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str
    accepted_steps: int = 0


def _wrap_periodic_delta(delta: float, period: float) -> float:
    half = 0.5 * float(period)
    return float(((float(delta) + half) % float(period)) - half)


def _finite_or_inf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return out if math.isfinite(out) else float("inf")


def _coordinate_values(value: Any, *, size: int, field: str, nonzero: bool) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if int(arr.size) == 1:
        arr = np.full(int(size), float(arr[0]), dtype=float)
    elif int(arr.size) != int(size):
        raise ValueError(f"{field} must be scalar or length {int(size)}; got length {int(arr.size)}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} entries must be finite.")
    if nonzero:
        if np.any(arr == 0.0):
            raise ValueError(f"{field} entries must be nonzero.")
    elif np.any(arr <= 0.0):
        raise ValueError(f"{field} entries must be positive.")
    return np.asarray(arr, dtype=float)


def rotosolve_stencil_from_parameterization_layout(
    layout: Any,
    *,
    parameterization_mode: str,
    num_parameters: int | None = None,
    active_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return coefficient-aware ROTOSOLVE periods/shifts for ansatz coordinates.

    The repo ansatz applies each runtime Pauli term as ``exp(-i theta c P)``.
    A single Pauli coordinate therefore has period ``pi / |c|`` and the
    canonical three-point ROTOSOLVE stencil shift is one quarter period.  Macro
    generators are only supported when represented as ``per_pauli_term``
    coordinates; a logical shared macro coordinate is not a single-frequency
    sinusoid and returns ``None``.
    """

    blocks = tuple(getattr(layout, "blocks", ()) or ())
    mode = str(parameterization_mode or "logical_shared").strip().lower()
    coeffs: list[float] = []
    if mode.startswith("per_pauli"):
        for block in blocks:
            for spec in tuple(getattr(block, "terms", ()) or ()):  # runtime order
                coeffs.append(abs(float(getattr(spec, "coeff_real"))))
    else:
        for block in blocks:
            terms = tuple(getattr(block, "terms", ()) or ())
            if int(len(terms)) != 1:
                return None
            coeffs.append(abs(float(getattr(terms[0], "coeff_real"))))

    if num_parameters is not None and int(len(coeffs)) != int(num_parameters):
        return None
    coeff_arr = np.asarray(coeffs, dtype=float).reshape(-1)
    if active_indices is not None:
        active = [int(i) for i in active_indices]
        if any(i < 0 or i >= int(coeff_arr.size) for i in active):
            return None
        coeff_arr = coeff_arr[np.asarray(active, dtype=int)]
    if coeff_arr.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty
    if (not np.all(np.isfinite(coeff_arr))) or np.any(coeff_arr <= 0.0):
        return None
    periods = np.asarray(math.pi / coeff_arr, dtype=float)
    shifts = 0.25 * periods
    return periods, shifts


def rotosolve_stencil_from_executor(
    executor: Any,
    *,
    active_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return coefficient-aware ROTOSOLVE periods/shifts for executor coordinates."""

    return rotosolve_stencil_from_parameterization_layout(
        getattr(executor, "layout", None),
        parameterization_mode=str(getattr(executor, "parameterization_mode", "logical_shared")),
        num_parameters=int(getattr(executor, "num_parameters")),
        active_indices=active_indices,
    )


def rotosolve_coordinate_descent(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int,
    tol: float = 1e-10,
    period: Any = 2.0 * math.pi,
    shift: Any = 0.5 * math.pi,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> CoordinateDescentResult:
    """Run a fixed-stencil rotosolve-style coordinate descent.

    Each coordinate is fit with the single-frequency stencil
    ``f(theta), f(theta + pi/2), f(theta - pi/2)`` and then evaluated at the
    analytic minimum of that sinusoid. For non-single-frequency coordinates
    this remains a deterministic coordinate-descent step by accepting the best
    of the current, stencil, and analytic-minimum points.
    """

    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    max_sweeps = max(0, int(maxiter))
    tol_val = max(0.0, float(tol))
    period_values = _coordinate_values(period, size=int(x.size), field="period", nonzero=False)
    shift_values = _coordinate_values(shift, size=int(x.size), field="shift", nonzero=True)

    current_fun = _finite_or_inf(objective(x.copy()))
    nfev = 1
    accepted_steps = 0
    if int(x.size) == 0:
        return CoordinateDescentResult(
            x=x,
            fun=float(current_fun),
            nfev=int(nfev),
            nit=0,
            success=math.isfinite(current_fun),
            message="empty_ansatz",
            accepted_steps=0,
        )
    if max_sweeps == 0:
        return CoordinateDescentResult(
            x=x,
            fun=float(current_fun),
            nfev=int(nfev),
            nit=0,
            success=math.isfinite(current_fun),
            message="maxiter_zero",
            accepted_steps=0,
        )

    sweeps_completed = 0
    for sweep in range(max_sweeps):
        sweep_start_fun = float(current_fun)
        for coord in range(int(x.size)):
            theta0 = float(x[coord])
            period_val = float(period_values[int(coord)])
            shift_val = float(shift_values[int(coord)])

            x_plus = x.copy()
            x_plus[coord] = theta0 + shift_val
            f_plus = _finite_or_inf(objective(x_plus))
            nfev += 1

            x_minus = x.copy()
            x_minus[coord] = theta0 - shift_val
            f_minus = _finite_or_inf(objective(x_minus))
            nfev += 1

            phase_shift = 2.0 * math.pi * float(shift_val) / float(period_val)
            sin_shift = math.sin(phase_shift)
            one_minus_cos_shift = 1.0 - math.cos(phase_shift)
            if abs(sin_shift) <= 1.0e-15 or abs(one_minus_cos_shift) <= 1.0e-15:
                delta_star = 0.0
            else:
                cosine_coeff = (float(current_fun) - 0.5 * (float(f_plus) + float(f_minus))) / one_minus_cos_shift
                sine_coeff = 0.5 * (float(f_plus) - float(f_minus)) / sin_shift
                phase_star = math.atan2(sine_coeff, cosine_coeff) + math.pi
                delta_star = _wrap_periodic_delta(
                    phase_star * float(period_val) / (2.0 * math.pi),
                    period_val,
                )
            x_star = x.copy()
            x_star[coord] = theta0 + float(delta_star)
            f_star = _finite_or_inf(objective(x_star))
            nfev += 1

            best_x = x
            best_fun = float(current_fun)
            for candidate_x, candidate_fun in (
                (x_plus, f_plus),
                (x_minus, f_minus),
                (x_star, f_star),
            ):
                if float(candidate_fun) < float(best_fun) - tol_val:
                    best_x = candidate_x
                    best_fun = float(candidate_fun)

            if best_x is not x:
                x = np.asarray(best_x, dtype=float).reshape(-1).copy()
                current_fun = float(best_fun)
                accepted_steps += 1

        sweeps_completed = int(sweep + 1)
        sweep_improvement = float(sweep_start_fun) - float(current_fun)
        if callback is not None:
            callback(
                {
                    "iter": int(sweeps_completed),
                    "nfev_so_far": int(nfev),
                    "best_fun": float(current_fun),
                    "sweep_improvement": float(sweep_improvement),
                    "accepted_steps": int(accepted_steps),
                }
            )
        if float(sweep_improvement) <= tol_val:
            return CoordinateDescentResult(
                x=x,
                fun=float(current_fun),
                nfev=int(nfev),
                nit=int(sweeps_completed),
                success=math.isfinite(current_fun),
                message="converged",
                accepted_steps=int(accepted_steps),
            )

    return CoordinateDescentResult(
        x=x,
        fun=float(current_fun),
        nfev=int(nfev),
        nit=int(sweeps_completed),
        success=False,
        message="maxiter_reached",
        accepted_steps=int(accepted_steps),
    )
