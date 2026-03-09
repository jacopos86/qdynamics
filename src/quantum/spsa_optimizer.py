"""Pure NumPy SPSA minimizer utility.

Schedules:
    c_k = c / (k + 1)^gamma
    a_k = a / (A + k + 1)^alpha

Gradient estimate:
    y_plus  = f(x + c_k * Delta_k)
    y_minus = f(x - c_k * Delta_k)
    g_hat   = ((y_plus - y_minus) / (2 * c_k)) * Delta_k

Update:
    x <- x - a_k * g_hat

Projection behavior:
    If bounds are provided and project == "clip", clipping is applied:
    1) to x_plus and x_minus before objective evaluation,
    2) to x after each parameter update.

Noise handling:
    Each function evaluation can be repeated eval_repeats times and aggregated
    using mean or median.

Return policy:
    - avg_last > 0: return Polyak-style average over the last avg_last iterates
      and evaluate the averaged point once more via the same aggregate evaluator.
    - avg_last == 0: return the best observed point among evaluated x_plus/x_minus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = ["SPSAResult", "spsa_minimize"]


@dataclass
class SPSAResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str
    history: list[dict[str, Any]]
    optimizer_memory: dict[str, Any] | None = None


def _validate_inputs(
    x0: np.ndarray,
    maxiter: int,
    eval_repeats: int,
    eval_agg: str,
    project: str,
    callback_every: int,
    bounds: Optional[Sequence[tuple[float, float]]],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1:
        raise ValueError("x0 must be a 1-D array")
    if x.size <= 0:
        raise ValueError("x0 must contain at least one parameter")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be >= 1")
    if int(eval_repeats) < 1:
        raise ValueError("eval_repeats must be >= 1")
    if str(eval_agg) not in {"mean", "median"}:
        raise ValueError("eval_agg must be 'mean' or 'median'")
    if str(project) not in {"clip", "none"}:
        raise ValueError("project must be 'clip' or 'none'")
    if int(callback_every) < 1:
        raise ValueError("callback_every must be >= 1")

    if bounds is None:
        return x, None, None

    if len(bounds) != int(x.size):
        raise ValueError("bounds length must equal x0 size")

    lo = np.empty(x.size, dtype=float)
    hi = np.empty(x.size, dtype=float)
    for i, bnd in enumerate(bounds):
        if not isinstance(bnd, tuple) or len(bnd) != 2:
            raise ValueError("each bounds entry must be a tuple(lo, hi)")
        lo_i = float(bnd[0])
        hi_i = float(bnd[1])
        if not np.isfinite(lo_i) or not np.isfinite(hi_i):
            raise ValueError("bounds must be finite")
        if lo_i > hi_i:
            raise ValueError("each bounds entry must satisfy lo <= hi")
        lo[i] = lo_i
        hi[i] = hi_i
    return x, lo, hi


def _clip_if_needed(
    x: np.ndarray,
    lo: Optional[np.ndarray],
    hi: Optional[np.ndarray],
    project: str,
) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if lo is None or hi is None or str(project) != "clip":
        return np.array(arr, copy=True)
    return np.clip(arr, lo, hi)


def _aggregate_eval(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    eval_repeats: int,
    eval_agg: str,
    nfev_counter: list[int],
) -> float:
    vals = np.empty(int(eval_repeats), dtype=float)
    x_eval = np.asarray(x, dtype=float)
    for rep in range(int(eval_repeats)):
        vals[rep] = float(fun(x_eval))
        nfev_counter[0] += 1
    if str(eval_agg) == "mean":
        return float(np.mean(vals))
    return float(np.median(vals))


_SPSA_CK_SYMBOL = "c_k = c / (k + 1)^gamma"
_SPSA_AK_SYMBOL = "a_k = a / (A + k + 1)^alpha"
_SPSA_GRAD_SYMBOL = "g_hat = ((y_plus - y_minus) / (2 * c_k)) * Delta_k"
_SPSA_UPDATE_SYMBOL = "x <- x - a_k * g_hat"


def spsa_minimize(
    fun: Callable[[np.ndarray], float],
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
    precondition_mode: str = "none",
) -> SPSAResult:
    x, lo, hi = _validate_inputs(
        x0=np.asarray(x0, dtype=float),
        maxiter=int(maxiter),
        eval_repeats=int(eval_repeats),
        eval_agg=str(eval_agg),
        project=str(project),
        callback_every=int(callback_every),
        bounds=bounds,
    )

    rng = np.random.default_rng(int(seed))
    nfev_counter = [0]
    history: list[dict[str, Any]] = []
    iterates: list[np.ndarray] = []
    npar = int(x.size)

    precondition_mode_key = str(precondition_mode).strip().lower()
    if precondition_mode_key not in {"none", "diag_rms_grad"}:
        raise ValueError("precondition_mode must be 'none' or 'diag_rms_grad'")
    refresh_every_val = int(max(0, refresh_every))
    if isinstance(memory, dict):
        raw_precond = list(memory.get("preconditioner_diag", []))
        raw_grad_sq = list(memory.get("grad_sq_ema", []))
        preconditioner_diag = np.ones(npar, dtype=float)
        grad_sq_ema = np.zeros(npar, dtype=float)
        for i in range(min(npar, len(raw_precond))):
            preconditioner_diag[i] = float(raw_precond[i])
        for i in range(min(npar, len(raw_grad_sq))):
            grad_sq_ema[i] = float(raw_grad_sq[i])
        memory_available = bool(memory.get("available", False))
        memory_source = str(memory.get("source", "input_memory"))
        refresh_points = [int(v) for v in memory.get("refresh_points", [])]
        remap_events = [dict(x) for x in memory.get("remap_events", []) if isinstance(x, dict)]
    else:
        preconditioner_diag = np.ones(npar, dtype=float)
        grad_sq_ema = np.zeros(npar, dtype=float)
        memory_available = False
        memory_source = "fresh"
        refresh_points = []
        remap_events = []
    grad_ema_beta = 0.9
    preconditioner_eps = 1e-8

    x_current = _clip_if_needed(x, lo, hi, project)

    # Seed best-tracking with f(x0) so we never return worse than entry.
    best_x_observed = np.array(x_current, copy=True)
    best_y_observed = _aggregate_eval(
        fun,
        x_current,
        eval_repeats=int(eval_repeats),
        eval_agg=str(eval_agg),
        nfev_counter=nfev_counter,
    )
    nit = 0

    try:
        for k in range(int(maxiter)):
            ck = float(c) / ((k + 1.0) ** float(gamma))
            ak = float(a) / ((float(A) + k + 1.0) ** float(alpha))

            delta = rng.choice(np.array([-1.0, 1.0], dtype=float), size=x_current.size)

            x_plus = _clip_if_needed(x_current + ck * delta, lo, hi, project)
            x_minus = _clip_if_needed(x_current - ck * delta, lo, hi, project)

            y_plus = _aggregate_eval(
                fun,
                x_plus,
                eval_repeats=int(eval_repeats),
                eval_agg=str(eval_agg),
                nfev_counter=nfev_counter,
            )
            y_minus = _aggregate_eval(
                fun,
                x_minus,
                eval_repeats=int(eval_repeats),
                eval_agg=str(eval_agg),
                nfev_counter=nfev_counter,
            )

            if y_plus < best_y_observed:
                best_y_observed = float(y_plus)
                best_x_observed = np.array(x_plus, copy=True)
            if y_minus < best_y_observed:
                best_y_observed = float(y_minus)
                best_x_observed = np.array(x_minus, copy=True)

            ghat = ((float(y_plus) - float(y_minus)) / (2.0 * ck)) * delta
            grad_norm = float(np.linalg.norm(ghat))
            grad_sq_ema = float(grad_ema_beta) * grad_sq_ema + (1.0 - float(grad_ema_beta)) * (ghat ** 2)
            refresh_triggered = False
            if (
                precondition_mode_key == "diag_rms_grad"
                and refresh_every_val > 0
                and ((k + 1) % refresh_every_val == 0)
            ):
                preconditioner_diag = 1.0 / np.sqrt(np.maximum(grad_sq_ema, float(preconditioner_eps)))
                preconditioner_diag = np.clip(preconditioner_diag, 0.1, 10.0)
                refresh_points.append(int(k + 1))
                refresh_triggered = True
            update_direction = np.asarray(ghat, dtype=float)
            if precondition_mode_key == "diag_rms_grad":
                update_direction = np.asarray(preconditioner_diag, dtype=float) * update_direction
            x_current = _clip_if_needed(x_current - ak * update_direction, lo, hi, project)
            iterates.append(np.array(x_current, copy=True))
            nit = k + 1

            item = {
                "iter": int(k + 1),
                "ak": float(ak),
                "ck": float(ck),
                "y_plus": float(y_plus),
                "y_minus": float(y_minus),
                "grad_norm": float(grad_norm),
                "best_fun": float(best_y_observed),
                "nfev_so_far": int(nfev_counter[0]),
                "preconditioner_mean": float(np.mean(preconditioner_diag)),
                "refresh_triggered": bool(refresh_triggered),
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
                nfev_counter=nfev_counter,
            )
        else:
            # best_x_observed is always seeded with f(x0), so never None here.
            x_out = np.array(best_x_observed, copy=True)
            fun_out = float(best_y_observed)

        return SPSAResult(
            x=np.asarray(x_out, dtype=float),
            fun=float(fun_out),
            nfev=int(nfev_counter[0]),
            nit=int(nit),
            success=True,
            message=(
                "spsa_completed(maxiter="
                f"{int(maxiter)},a={float(a)},c={float(c)},alpha={float(alpha)},"
                f"gamma={float(gamma)},A={float(A)},eval_repeats={int(eval_repeats)},"
                f"eval_agg={str(eval_agg)},avg_last={int(avg_last)})"
            ),
            history=history,
            optimizer_memory={
                "version": "phase2_spsa_diag_memory_v1",
                "optimizer": "SPSA",
                "parameter_count": int(npar),
                "available": bool(precondition_mode_key == "diag_rms_grad"),
                "source": str(memory_source),
                "reused": bool(memory_available),
                "reason": ("" if precondition_mode_key == "diag_rms_grad" else "preconditioner_disabled"),
                "preconditioner_diag": [float(v) for v in np.asarray(preconditioner_diag, dtype=float).tolist()],
                "grad_sq_ema": [float(v) for v in np.asarray(grad_sq_ema, dtype=float).tolist()],
                "history_tail": [dict(x) for x in history[-32:]],
                "refresh_points": [int(v) for v in refresh_points],
                "remap_events": [dict(x) for x in remap_events],
            },
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if best_x_observed is None:
            x_fail = np.array(x_current, copy=True)
            fun_fail = float("inf")
        else:
            x_fail = np.array(best_x_observed, copy=True)
            fun_fail = float(best_y_observed)
        return SPSAResult(
            x=np.asarray(x_fail, dtype=float),
            fun=float(fun_fail),
            nfev=int(nfev_counter[0]),
            nit=int(nit),
            success=False,
            message=f"spsa_failed({type(exc).__name__}: {exc})",
            history=history,
            optimizer_memory={
                "version": "phase2_spsa_diag_memory_v1",
                "optimizer": "SPSA",
                "parameter_count": int(npar),
                "available": bool(precondition_mode_key == "diag_rms_grad"),
                "source": str(memory_source),
                "reused": bool(memory_available),
                "reason": f"spsa_failed({type(exc).__name__})",
                "preconditioner_diag": [float(v) for v in np.asarray(preconditioner_diag, dtype=float).tolist()],
                "grad_sq_ema": [float(v) for v in np.asarray(grad_sq_ema, dtype=float).tolist()],
                "history_tail": [dict(x) for x in history[-32:]],
                "refresh_points": [int(v) for v in refresh_points],
                "remap_events": [dict(x) for x in remap_events],
            },
        )
