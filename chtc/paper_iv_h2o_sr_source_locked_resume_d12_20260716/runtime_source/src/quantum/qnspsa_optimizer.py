"""Pure NumPy QN-SPSA minimizer utility.

QN-SPSA augments the ordinary two-point SPSA gradient estimate with a
stochastic estimate of the parameter-space quantum metric built from an ansatz
fidelity callable.  This implementation is intentionally repo-native and does
not import Qiskit; callers provide ``fidelity(x, y) -> float`` for the same
parameter surface as ``fun``.

Schedules match :mod:`src.quantum.spsa_optimizer`:
    c_k = c / (k + 1)^gamma
    a_k = a / (A + k + 1)^alpha

Per resampling, QN-SPSA uses two objective evaluations and four fidelity
evaluations.  The public result shape mirrors ``SPSAResult`` so static ADAPT can
use the optimizer as an explicit opt-in without changing default SPSA behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from src.quantum.spsa_optimizer import _aggregate_eval, _clip_if_needed, _validate_inputs

__all__ = ["QNSPSAResult", "qnspsa_minimize"]


@dataclass
class QNSPSAResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str
    history: list[dict[str, Any]]
    optimizer_memory: dict[str, Any] | None = None
    objective_nfev: int = 0
    fidelity_nfev: int = 0


def _rademacher(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0], dtype=float), size=int(size))


def _eval_fidelity(
    fidelity: Callable[[np.ndarray, np.ndarray], float],
    x_left: np.ndarray,
    x_right: np.ndarray,
    *,
    nfev_counter: list[int],
) -> float:
    value = float(fidelity(np.asarray(x_left, dtype=float), np.asarray(x_right, dtype=float)))
    nfev_counter[0] += 1
    if not np.isfinite(value):
        raise FloatingPointError("fidelity returned a non-finite value")
    return value


def _spd_from_metric_sample(
    hessian: np.ndarray,
    *,
    regularization: float,
    psd_floor: float,
) -> tuple[np.ndarray, float]:
    sym = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    n = int(sym.shape[0])
    sym = sym + float(regularization) * np.eye(n, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(sym)
    floor = float(psd_floor)
    clipped = np.maximum(np.asarray(eigvals, dtype=float), floor)
    if np.any(~np.isfinite(clipped)):
        raise np.linalg.LinAlgError("non-finite metric eigenvalues")
    max_eval = float(np.max(clipped)) if clipped.size else 0.0
    min_eval = float(np.min(clipped)) if clipped.size else 0.0
    cond = float("inf") if min_eval <= 0.0 else float(max_eval / min_eval)
    spd = (eigvecs * clipped) @ eigvecs.T
    spd = 0.5 * (spd + spd.T)
    return spd, cond


def _memory_payload(
    *,
    npar: int,
    hessian_smooth: np.ndarray,
    history: list[dict[str, Any]],
    objective_nfev: int,
    fidelity_nfev: int,
    source: str,
    reason: str,
    remap_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    diag = np.diag(np.asarray(hessian_smooth, dtype=float)) if int(npar) > 0 else np.zeros(0)
    safe_diag = np.where(np.isfinite(diag), np.maximum(np.abs(diag), 1e-12), 1.0)
    preconditioner_diag = 1.0 / np.sqrt(safe_diag)
    return {
        "version": "qnspsa_optimizer_memory_v1",
        "optimizer": "QNSPSA",
        "parameter_count": int(npar),
        "available": False,
        "source": str(source),
        "reused": False,
        "reason": str(reason),
        "preconditioner_diag": [float(v) for v in np.asarray(preconditioner_diag, dtype=float).tolist()],
        "grad_sq_ema": [0.0] * int(npar),
        "history_tail": [dict(x) for x in history[-32:]],
        "refresh_points": [],
        "remap_events": [dict(x) for x in (remap_events or []) if isinstance(x, dict)],
        "hessian_diag": [float(v) for v in np.asarray(diag, dtype=float).tolist()],
        "nfev_objective": int(objective_nfev),
        "nfev_fidelity": int(fidelity_nfev),
    }


def qnspsa_minimize(
    fun: Callable[[np.ndarray], float],
    fidelity: Callable[[np.ndarray, np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int,
    seed: int,
    a: float = 0.2,
    c: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
    A: float = 10.0,
    bounds: Optional[Sequence[tuple[float, float]]] = None,
    project: str = "clip",
    eval_repeats: int = 1,
    eval_agg: str = "mean",
    avg_last: int = 0,
    callback: Optional[Callable[[dict[str, Any]], None]] = None,
    callback_every: int = 1,
    memory: dict[str, Any] | None = None,
    refresh_every: int = 0,
    precondition_mode: str = "qnspsa",
    resamplings: int = 1,
    regularization: float = 1e-3,
    psd_floor: float = 1e-8,
    hessian_delay: int = 0,
) -> QNSPSAResult:
    """Minimize ``fun`` with a pure NumPy QN-SPSA update.

    ``nfev`` is the total scalar work count: objective evaluations plus
    fidelity evaluations.  ``optimizer_memory`` is intentionally marked
    unavailable for reuse in this first slice, while preserving the vector-key
    shape expected by the Phase2 optimizer-memory adapter.
    """
    x, lo, hi = _validate_inputs(
        x0=np.asarray(x0, dtype=float),
        maxiter=int(maxiter),
        eval_repeats=int(eval_repeats),
        eval_agg=str(eval_agg),
        project=str(project),
        callback_every=int(callback_every),
        bounds=bounds,
    )
    if int(resamplings) < 1:
        raise ValueError("resamplings must be >= 1")
    if float(regularization) < 0.0:
        raise ValueError("regularization must be >= 0")
    if float(psd_floor) < 0.0:
        raise ValueError("psd_floor must be >= 0")
    if int(hessian_delay) < 0:
        raise ValueError("hessian_delay must be >= 0")
    precondition_key = str(precondition_mode).strip().lower()
    if precondition_key not in {"qnspsa", "none"}:
        raise ValueError("precondition_mode must be 'qnspsa' or 'none'")
    # Accepted for API compatibility with SPSA; memory reuse is deliberately disabled.
    _ = memory, refresh_every

    rng = np.random.default_rng(int(seed))
    objective_counter = [0]
    fidelity_counter = [0]
    history: list[dict[str, Any]] = []
    iterates: list[np.ndarray] = []
    npar = int(x.size)
    x_current = _clip_if_needed(x, lo, hi, project)
    hessian_smooth = np.eye(npar, dtype=float)
    best_x_observed = np.array(x_current, copy=True)
    best_y_observed = float("inf")
    nit = 0

    try:
        # Seed best-tracking with f(x0), matching native SPSA's no-worse-than-x0 policy.
        best_y_observed = _aggregate_eval(
            fun,
            x_current,
            eval_repeats=int(eval_repeats),
            eval_agg=str(eval_agg),
            nfev_counter=objective_counter,
        )
        if not np.isfinite(best_y_observed):
            raise FloatingPointError("objective returned a non-finite value at x0")

        for k in range(int(maxiter)):
            ck = float(c) / ((k + 1.0) ** float(gamma))
            ak = float(a) / ((float(A) + k + 1.0) ** float(alpha))
            grad_accum = np.zeros(npar, dtype=float)
            hess_accum = np.zeros((npar, npar), dtype=float)
            y_plus_last = float("nan")
            y_minus_last = float("nan")

            for _sample in range(int(resamplings)):
                delta1 = _rademacher(rng, npar)
                delta2 = _rademacher(rng, npar)

                x_plus = _clip_if_needed(x_current + ck * delta1, lo, hi, project)
                x_minus = _clip_if_needed(x_current - ck * delta1, lo, hi, project)
                x_f_p = _clip_if_needed(x_current + ck * delta1, lo, hi, project)
                x_f_m = _clip_if_needed(x_current - ck * delta1, lo, hi, project)
                x_f_pp = _clip_if_needed(x_current + ck * (delta1 + delta2), lo, hi, project)
                x_f_mp = _clip_if_needed(x_current + ck * (-delta1 + delta2), lo, hi, project)

                y_plus = _aggregate_eval(
                    fun,
                    x_plus,
                    eval_repeats=int(eval_repeats),
                    eval_agg=str(eval_agg),
                    nfev_counter=objective_counter,
                )
                y_minus = _aggregate_eval(
                    fun,
                    x_minus,
                    eval_repeats=int(eval_repeats),
                    eval_agg=str(eval_agg),
                    nfev_counter=objective_counter,
                )
                if not np.isfinite(y_plus) or not np.isfinite(y_minus):
                    raise FloatingPointError("objective returned a non-finite value")
                if y_plus < best_y_observed:
                    best_y_observed = float(y_plus)
                    best_x_observed = np.array(x_plus, copy=True)
                if y_minus < best_y_observed:
                    best_y_observed = float(y_minus)
                    best_x_observed = np.array(x_minus, copy=True)

                f_p = _eval_fidelity(fidelity, x_current, x_f_p, nfev_counter=fidelity_counter)
                f_m = _eval_fidelity(fidelity, x_current, x_f_m, nfev_counter=fidelity_counter)
                f_pp = _eval_fidelity(fidelity, x_current, x_f_pp, nfev_counter=fidelity_counter)
                f_mp = _eval_fidelity(fidelity, x_current, x_f_mp, nfev_counter=fidelity_counter)

                grad_accum += ((float(y_plus) - float(y_minus)) / (2.0 * ck)) * delta1
                diff = ((float(f_pp) - float(f_p)) - (float(f_mp) - float(f_m))) / (2.0 * ck * ck)
                rank_one = np.outer(delta1, delta2)
                hess_accum += -0.5 * float(diff) * (rank_one + rank_one.T) / 2.0
                y_plus_last = float(y_plus)
                y_minus_last = float(y_minus)

            ghat = grad_accum / float(resamplings)
            hessian_est = hess_accum / float(resamplings)
            hessian_smooth = (float(k) / float(k + 1)) * hessian_smooth + (1.0 / float(k + 1)) * hessian_est

            update_direction = np.asarray(ghat, dtype=float)
            preconditioned = False
            hessian_condition_estimate: float | None = None
            hessian_skip_reason: str | None = None
            if precondition_key == "none":
                hessian_skip_reason = "preconditioner_disabled"
            elif int(k + 1) <= int(hessian_delay):
                hessian_skip_reason = "hessian_delay"
            else:
                try:
                    spd_hessian, cond_est = _spd_from_metric_sample(
                        hessian_smooth,
                        regularization=float(regularization),
                        psd_floor=float(psd_floor),
                    )
                    hessian_condition_estimate = float(cond_est)
                    candidate_direction = np.linalg.solve(spd_hessian, ghat)
                    grad_norm_for_guard = float(max(1.0, np.linalg.norm(ghat)))
                    candidate_norm = float(np.linalg.norm(candidate_direction))
                    if (
                        (not np.all(np.isfinite(candidate_direction)))
                        or (not np.isfinite(candidate_norm))
                        or (not np.isfinite(cond_est))
                        or float(cond_est) > 1e12
                        or candidate_norm > 1e6 * grad_norm_for_guard
                    ):
                        hessian_skip_reason = "metric_solve_bad"
                    else:
                        update_direction = np.asarray(candidate_direction, dtype=float)
                        preconditioned = True
                except Exception:
                    hessian_skip_reason = "metric_solve_failed"

            grad_norm = float(np.linalg.norm(ghat))
            x_current = _clip_if_needed(x_current - ak * update_direction, lo, hi, project)
            iterates.append(np.array(x_current, copy=True))
            nit = k + 1

            item = {
                "iter": int(k + 1),
                "ak": float(ak),
                "ck": float(ck),
                "y_plus": float(y_plus_last),
                "y_minus": float(y_minus_last),
                "grad_norm": float(grad_norm),
                "best_fun": float(best_y_observed),
                "nfev_so_far": int(objective_counter[0] + fidelity_counter[0]),
                "objective_nfev_so_far": int(objective_counter[0]),
                "fidelity_nfev_so_far": int(fidelity_counter[0]),
                "preconditioned": bool(preconditioned),
                "hessian_condition_estimate": (
                    None if hessian_condition_estimate is None else float(hessian_condition_estimate)
                ),
                "hessian_skip_reason": hessian_skip_reason,
            }
            history.append(item)

            if callback is not None and ((k + 1) % int(callback_every) == 0):
                payload = dict(item)
                payload["x_current"] = np.array(x_current, copy=True)
                callback(payload)

        if int(avg_last) > 0:
            tail = int(min(int(avg_last), len(iterates)))
            x_out = np.mean(np.asarray(iterates[-tail:], dtype=float), axis=0)
            fun_out = _aggregate_eval(
                fun,
                x_out,
                eval_repeats=int(eval_repeats),
                eval_agg=str(eval_agg),
                nfev_counter=objective_counter,
            )
        else:
            x_out = np.array(best_x_observed, copy=True)
            fun_out = float(best_y_observed)

        return QNSPSAResult(
            x=np.asarray(x_out, dtype=float),
            fun=float(fun_out),
            nfev=int(objective_counter[0] + fidelity_counter[0]),
            nit=int(nit),
            success=True,
            message=(
                "qnspsa_completed(maxiter="
                f"{int(maxiter)},a={float(a)},c={float(c)},alpha={float(alpha)},"
                f"gamma={float(gamma)},A={float(A)},eval_repeats={int(eval_repeats)},"
                f"eval_agg={str(eval_agg)},avg_last={int(avg_last)},resamplings={int(resamplings)})"
            ),
            history=history,
            optimizer_memory=_memory_payload(
                npar=int(npar),
                hessian_smooth=hessian_smooth,
                history=history,
                objective_nfev=int(objective_counter[0]),
                fidelity_nfev=int(fidelity_counter[0]),
                source="fresh",
                reason="qnspsa_memory_reuse_disabled_in_slice",
            ),
            objective_nfev=int(objective_counter[0]),
            fidelity_nfev=int(fidelity_counter[0]),
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return QNSPSAResult(
            x=np.asarray(best_x_observed, dtype=float),
            fun=float(best_y_observed),
            nfev=int(objective_counter[0] + fidelity_counter[0]),
            nit=int(nit),
            success=False,
            message=f"qnspsa_failed({type(exc).__name__}: {exc})",
            history=history,
            optimizer_memory=_memory_payload(
                npar=int(npar),
                hessian_smooth=hessian_smooth,
                history=history,
                objective_nfev=int(objective_counter[0]),
                fidelity_nfev=int(fidelity_counter[0]),
                source="fresh",
                reason=f"qnspsa_failed({type(exc).__name__})",
            ),
            objective_nfev=int(objective_counter[0]),
            fidelity_nfev=int(fidelity_counter[0]),
        )
