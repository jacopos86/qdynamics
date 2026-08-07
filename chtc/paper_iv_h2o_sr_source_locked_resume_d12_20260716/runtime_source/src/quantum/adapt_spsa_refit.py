"""Shared ADAPT SPSA refit engine selection and legacy descent helper."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.quantum.spsa_optimizer import SPSAResult, _clip_if_needed, _validate_inputs

NATIVE_SPSA_OPTIMIZER_LABEL = "src.quantum.spsa_optimizer:spsa_minimize"
LEGACY_SPSA_POLISH_OPTIMIZER_LABEL = "exact_bench_spsa:energy_only_descent"
ADAPT_SPSA_REFIT_ENGINE_ENV = "ADAPT_SPSA_REFIT_ENGINE"
GENERIC_STATIC_ADAPT_SPSA_REFIT_ENGINE_ENV = "GENERIC_STATIC_TABLE_ADAPT_SPSA_REFIT_ENGINE"


@dataclass(frozen=True)
class SPSAEnergyDescentSchedule:
    a: float = 0.05
    c: float = 0.05
    alpha: float = 0.602
    gamma: float = 0.101
    big_a: float = 5.0


def default_spsa_energy_descent_schedule() -> SPSAEnergyDescentSchedule:
    return SPSAEnergyDescentSchedule()


def resolve_adapt_spsa_refit_engine_label(
    *,
    env: Mapping[str, str] | None = None,
    env_names: Sequence[str] = (
        ADAPT_SPSA_REFIT_ENGINE_ENV,
        GENERIC_STATIC_ADAPT_SPSA_REFIT_ENGINE_ENV,
    ),
) -> str:
    environ = os.environ if env is None else env
    raw = ""
    used_name = ""
    for name in env_names:
        value = environ.get(str(name), "")
        if value is not None and str(value).strip():
            raw = str(value).strip().lower()
            used_name = str(name)
            break
    if raw in {"", "native", "native_spsa", "spsa_minimize", NATIVE_SPSA_OPTIMIZER_LABEL.lower()}:
        return NATIVE_SPSA_OPTIMIZER_LABEL
    if raw in {
        "legacy",
        "legacy_cap",
        "legacy_energy_descent",
        "energy_only_descent",
        "spsa_polish",
        LEGACY_SPSA_POLISH_OPTIMIZER_LABEL.lower(),
    }:
        return LEGACY_SPSA_POLISH_OPTIMIZER_LABEL
    env_hint = used_name or "/".join(str(name) for name in env_names)
    raise ValueError(f"{env_hint} must be native or legacy_energy_descent; got {raw!r}.")


def spsa_energy_descent_minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int,
    seed: int,
    initial_fun: float | None = None,
    schedule: SPSAEnergyDescentSchedule | None = None,
    max_abs_step: float = 0.25,
    accept_tol: float = 1e-12,
    bounds: Sequence[tuple[float, float]] | None = None,
    project: str = "none",
    decision_value: Callable[[float, str, int], float] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    callback_every: int = 1,
) -> SPSAResult:
    """Legacy ADAPT SPSA: accept energy-decreasing steps and stop after stationarity.

    ``maxiter`` is a cap. The optimizer stops after the first rejected candidate
    following at least one accepted step. This is intentionally separate from
    :func:`src.quantum.spsa_optimizer.spsa_minimize`, whose native policy runs
    all iterations and returns the best observed probe.
    """

    schedule_val = schedule or default_spsa_energy_descent_schedule()
    x0_arr = np.asarray(x0, dtype=float).reshape(-1)
    if int(maxiter) < 1 or int(x0_arr.size) <= 0:
        current = float(initial_fun) if initial_fun is not None else float(fun(x0_arr))
        nfev_initial = 0 if initial_fun is not None else 1
        return SPSAResult(
            x=np.asarray(x0_arr, dtype=float),
            fun=float(current),
            nfev=int(nfev_initial),
            nit=0,
            success=False,
            message="spsa_polish_no_parameters_or_budget",
            history=[],
            optimizer_memory={
                "version": "adapt_spsa_refit_engine_v1",
                "optimizer": "SPSA",
                "spsa_refit_engine": LEGACY_SPSA_POLISH_OPTIMIZER_LABEL,
                "available": False,
                "reason": "legacy_energy_descent_no_parameters_or_budget",
                "accepted_step_count": 0,
                "energy_before": float(current),
                "energy_after": float(current),
                "return_policy": "legacy_energy_descent_stop_after_stationary",
            },
        )
    x, lo, hi = _validate_inputs(
        x0=x0_arr,
        maxiter=int(maxiter),
        eval_repeats=1,
        eval_agg="mean",
        project=str(project),
        callback_every=int(callback_every),
        bounds=bounds,
    )
    x = _clip_if_needed(x, lo, hi, str(project))
    nfev = 0

    current_energy = float(initial_fun) if initial_fun is not None else float(fun(x))
    if initial_fun is None:
        nfev += 1
    current_decision = (
        float(decision_value(float(current_energy), "current", 0))
        if decision_value is not None
        else float(current_energy)
    )

    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    accepted = 0
    decrease_total = 0.0
    message = "spsa_polish_no_accepted_step"
    start_energy = float(current_energy)

    for k in range(int(maxiter)):
        ck = float(schedule_val.c) / ((k + 1.0) ** float(schedule_val.gamma))
        ak = float(schedule_val.a) / ((float(schedule_val.big_a) + k + 1.0) ** float(schedule_val.alpha))
        delta = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=int(x.size))

        x_plus = _clip_if_needed(x + ck * delta, lo, hi, str(project))
        x_minus = _clip_if_needed(x - ck * delta, lo, hi, str(project))
        y_plus = float(fun(x_plus))
        y_minus = float(fun(x_minus))
        nfev += 2

        if not (math.isfinite(y_plus) and math.isfinite(y_minus)):
            message = "spsa_polish_nonfinite_probe"
            continue

        y_plus_decision = (
            float(decision_value(float(y_plus), "probe_plus", int(k)))
            if decision_value is not None
            else float(y_plus)
        )
        y_minus_decision = (
            float(decision_value(float(y_minus), "probe_minus", int(k)))
            if decision_value is not None
            else float(y_minus)
        )
        grad = ((y_plus_decision - y_minus_decision) / (2.0 * ck)) * delta
        step = -ak * np.asarray(grad, dtype=float).reshape(-1)
        max_step = float(np.max(np.abs(step))) if step.size else 0.0
        if max_step > float(max_abs_step) > 0.0:
            step = step * float(max_abs_step / max_step)

        x_candidate = _clip_if_needed(x + step, lo, hi, str(project))
        candidate_energy = float(fun(x_candidate))
        nfev += 1
        if not math.isfinite(candidate_energy):
            message = "spsa_polish_nonfinite_candidate"
            continue

        candidate_decision = (
            float(decision_value(float(candidate_energy), "candidate", int(k)))
            if decision_value is not None
            else float(candidate_energy)
        )
        accepted_step = bool(candidate_decision <= current_decision - float(accept_tol))
        if accepted_step:
            decrease_total += max(0.0, float(current_energy) - float(candidate_energy))
            x = np.asarray(x_candidate, dtype=float).reshape(-1)
            current_energy = float(candidate_energy)
            current_decision = float(candidate_decision)
            accepted += 1
            message = "spsa_polish_descent"
        elif accepted > 0:
            message = "spsa_polish_stationary_after_descent"

        item = {
            "iter": int(k + 1),
            "ak": float(ak),
            "ck": float(ck),
            "y_plus": float(y_plus),
            "y_minus": float(y_minus),
            "candidate_fun": float(candidate_energy),
            "current_fun": float(current_energy),
            "accepted": bool(accepted_step),
            "accepted_step_count": int(accepted),
            "nfev_so_far": int(nfev),
        }
        history.append(item)
        if callback is not None and ((k + 1) % int(callback_every) == 0):
            payload = dict(item)
            payload["x_current"] = np.array(x, copy=True)
            callback(payload)
        if (not accepted_step) and accepted > 0:
            break

    return SPSAResult(
        x=np.asarray(x, dtype=float),
        fun=float(current_decision),
        nfev=int(nfev),
        nit=int(accepted),
        success=bool(accepted > 0),
        message=str(message),
        history=history,
        optimizer_memory={
            "version": "adapt_spsa_refit_engine_v1",
            "optimizer": "SPSA",
            "spsa_refit_engine": LEGACY_SPSA_POLISH_OPTIMIZER_LABEL,
            "available": False,
            "reason": "legacy_energy_descent_no_preconditioner_memory",
            "return_policy": "legacy_energy_descent_stop_after_stationary",
            "accepted_step_count": int(accepted),
            "energy_decrease_total": float(decrease_total),
            "energy_before": float(start_energy),
            "energy_after": float(current_energy),
            "spsa_a": float(schedule_val.a),
            "spsa_c": float(schedule_val.c),
            "spsa_alpha": float(schedule_val.alpha),
            "spsa_gamma": float(schedule_val.gamma),
            "spsa_A": float(schedule_val.big_a),
            "spsa_big_a": float(schedule_val.big_a),
            "history_tail": [dict(item) for item in history[-32:]],
        },
    )


__all__ = [
    "ADAPT_SPSA_REFIT_ENGINE_ENV",
    "GENERIC_STATIC_ADAPT_SPSA_REFIT_ENGINE_ENV",
    "LEGACY_SPSA_POLISH_OPTIMIZER_LABEL",
    "NATIVE_SPSA_OPTIMIZER_LABEL",
    "SPSAEnergyDescentSchedule",
    "default_spsa_energy_descent_schedule",
    "resolve_adapt_spsa_refit_engine_label",
    "spsa_energy_descent_minimize",
]
