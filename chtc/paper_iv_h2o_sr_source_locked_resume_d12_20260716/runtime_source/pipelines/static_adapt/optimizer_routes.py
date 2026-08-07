"""Inner-optimizer config and dispatch helpers for static ADAPT.

This module owns optimizer selection and concrete SPSA/QNSPSA/SciPy/ROTOSOLVE
dispatch only. State-dependent fidelity construction, refit windows, optimizer
memory routing, and ADAPT result accounting stay with the pipeline caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from pipelines.static_adapt.engine_support import (
    _VALID_ADAPT_INNER_OPTIMIZERS,
    _run_rotosolve_adapt_optimizer,
    _run_scipy_adapt_optimizer,
)
from src.quantum.adapt_spsa_refit import (
    ADAPT_SPSA_REFIT_ENGINE_ENV,
    LEGACY_SPSA_POLISH_OPTIMIZER_LABEL,
    NATIVE_SPSA_OPTIMIZER_LABEL,
    SPSAEnergyDescentSchedule,
    resolve_adapt_spsa_refit_engine_label,
    spsa_energy_descent_minimize,
)
from src.quantum.qnspsa_optimizer import qnspsa_minimize
from src.quantum.spsa_optimizer import spsa_minimize

STOCHASTIC_ADAPT_INNER_OPTIMIZERS = frozenset({"SPSA", "QNSPSA"})

__all__ = [
    "AdaptInnerOptimizerDispatch",
    "AdaptInnerOptimizerConfig",
    "AdaptSPSAConfig",
    "STOCHASTIC_ADAPT_INNER_OPTIMIZERS",
    "adapt_optimizer_progress_interval_s",
    "adapt_spsa_params_payload",
    "resolve_adapt_inner_optimizer_config",
    "resolve_effective_adapt_optimizer_key",
    "run_deterministic_inner_optimizer",
    "run_stochastic_inner_optimizer",
    "stochastic_heartbeat_event",
    "stochastic_heartbeat_event_from_config",
]


@dataclass(frozen=True)
class AdaptSPSAConfig:
    a: float
    c: float
    alpha: float
    gamma: float
    A: float
    avg_last: int
    eval_repeats: int
    eval_agg: str
    callback_every: int
    progress_every_s: float
    parallel_evaluations: int


@dataclass(frozen=True)
class AdaptInnerOptimizerConfig:
    inner_optimizer_key: str
    spsa_refit_engine_label: str
    spsa_refit_engine_env: str
    spsa: AdaptSPSAConfig
    scipy_maxfev: int | None = None


@dataclass(frozen=True)
class AdaptInnerOptimizerDispatch:
    """Bind one resolved optimizer configuration to all ADAPT refit calls."""

    config: AdaptInnerOptimizerConfig

    def run_stochastic(
        self,
        *,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        fidelity: Callable[[np.ndarray, np.ndarray], float] | None,
        maxiter_value: int,
        seed_value: int,
        callback: Callable[[Mapping[str, Any]], None] | None = None,
        memory: Mapping[str, Any] | None = None,
        precondition_mode: str = "none",
        optimizer_key: str | None = None,
    ) -> Any:
        return run_stochastic_inner_optimizer(
            config=self.config,
            fun=fun,
            x0=np.asarray(x0, dtype=float),
            fidelity=fidelity,
            maxiter_value=int(maxiter_value),
            seed_value=int(seed_value),
            callback=callback,
            memory=memory,
            precondition_mode=str(precondition_mode),
            optimizer_key=optimizer_key,
        )

    def run_deterministic(
        self,
        *,
        method_key: str,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        maxiter_value: int,
        context_label: str,
        callback: Callable[[Mapping[str, Any]], None] | None = None,
        rotosolve_period: Any | None = None,
        rotosolve_shift: Any | None = None,
    ) -> Any:
        return run_deterministic_inner_optimizer(
            config=self.config,
            method_key=str(method_key),
            objective=objective,
            x0=np.asarray(x0, dtype=float),
            maxiter_value=int(maxiter_value),
            context_label=str(context_label),
            callback=callback,
            rotosolve_period=rotosolve_period,
            rotosolve_shift=rotosolve_shift,
        )


def resolve_adapt_inner_optimizer_config(
    *,
    adapt_inner_optimizer: str,
    adapt_scipy_maxfev: int | None = None,
    adapt_spsa_a: float,
    adapt_spsa_c: float,
    adapt_spsa_alpha: float,
    adapt_spsa_gamma: float,
    adapt_spsa_A: float,
    adapt_spsa_avg_last: int,
    adapt_spsa_eval_repeats: int,
    adapt_spsa_eval_agg: str,
    adapt_spsa_callback_every: int,
    adapt_spsa_progress_every_s: float,
    adapt_spsa_parallel_evaluations: int,
) -> AdaptInnerOptimizerConfig:
    optimizer_key = str(adapt_inner_optimizer).strip().upper()
    if optimizer_key not in _VALID_ADAPT_INNER_OPTIMIZERS:
        raise ValueError(
            "adapt_inner_optimizer must be one of {'BFGS','COBYLA','POWELL','ROTOSOLVE','SPSA','QNSPSA'}."
        )
    eval_agg_key = str(adapt_spsa_eval_agg).strip().lower()
    if eval_agg_key not in {"mean", "median"}:
        raise ValueError("adapt_spsa_eval_agg must be one of {'mean','median'}.")
    if int(adapt_spsa_callback_every) < 1:
        raise ValueError("adapt_spsa_callback_every must be >= 1.")
    if float(adapt_spsa_progress_every_s) < 0.0:
        raise ValueError("adapt_spsa_progress_every_s must be >= 0.")
    if int(adapt_spsa_parallel_evaluations) < 1:
        raise ValueError("adapt_spsa_parallel_evaluations must be >= 1.")
    scipy_maxfev = (
        None
        if adapt_scipy_maxfev is None or int(adapt_scipy_maxfev) <= 0
        else int(adapt_scipy_maxfev)
    )
    return AdaptInnerOptimizerConfig(
        inner_optimizer_key=str(optimizer_key),
        spsa_refit_engine_label=str(resolve_adapt_spsa_refit_engine_label()),
        spsa_refit_engine_env=ADAPT_SPSA_REFIT_ENGINE_ENV,
        spsa=AdaptSPSAConfig(
            a=float(adapt_spsa_a),
            c=float(adapt_spsa_c),
            alpha=float(adapt_spsa_alpha),
            gamma=float(adapt_spsa_gamma),
            A=float(adapt_spsa_A),
            avg_last=int(adapt_spsa_avg_last),
            eval_repeats=int(adapt_spsa_eval_repeats),
            eval_agg=str(eval_agg_key),
            callback_every=int(adapt_spsa_callback_every),
            progress_every_s=float(adapt_spsa_progress_every_s),
            parallel_evaluations=int(adapt_spsa_parallel_evaluations),
        ),
        scipy_maxfev=scipy_maxfev,
    )


def adapt_spsa_params_payload(config: AdaptInnerOptimizerConfig) -> dict[str, Any]:
    spsa = config.spsa
    return {
        "refit_engine": str(config.spsa_refit_engine_label),
        "refit_engine_env": str(config.spsa_refit_engine_env),
        "a": float(spsa.a),
        "c": float(spsa.c),
        "alpha": float(spsa.alpha),
        "gamma": float(spsa.gamma),
        "A": float(spsa.A),
        "avg_last": int(spsa.avg_last),
        "eval_repeats": int(spsa.eval_repeats),
        "eval_agg": str(spsa.eval_agg),
        "callback_every": int(spsa.callback_every),
        "progress_every_s": float(spsa.progress_every_s),
        "scipy_maxfev": (
            None if config.scipy_maxfev is None else int(config.scipy_maxfev)
        ),
    }


def resolve_effective_adapt_optimizer_key(
    config: AdaptInnerOptimizerConfig,
    optimizer_key: str | None = None,
) -> str:
    key = None if optimizer_key is None else str(optimizer_key).strip()
    if key is None or key == "":
        key = str(config.inner_optimizer_key)
    return str(key).strip().upper()


def adapt_optimizer_progress_interval_s(config: AdaptInnerOptimizerConfig) -> float:
    return float(config.spsa.progress_every_s)


def stochastic_heartbeat_event(optimizer_key: str) -> str:
    return (
        "hardcoded_adapt_qnspsa_heartbeat"
        if str(optimizer_key).strip().upper() == "QNSPSA"
        else "hardcoded_adapt_spsa_heartbeat"
    )


def stochastic_heartbeat_event_from_config(
    config: AdaptInnerOptimizerConfig,
    optimizer_key: str | None = None,
) -> str:
    return stochastic_heartbeat_event(
        resolve_effective_adapt_optimizer_key(config, optimizer_key=optimizer_key)
    )


def run_stochastic_inner_optimizer(
    *,
    config: AdaptInnerOptimizerConfig,
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    fidelity: Callable[[np.ndarray, np.ndarray], float] | None,
    maxiter_value: int,
    seed_value: int,
    callback: Callable[[Mapping[str, Any]], None] | None = None,
    memory: Mapping[str, Any] | None = None,
    precondition_mode: str = "none",
    optimizer_key: str | None = None,
) -> Any:
    local_optimizer_key = resolve_effective_adapt_optimizer_key(
        config,
        optimizer_key=optimizer_key,
    )
    spsa = config.spsa
    if local_optimizer_key == "SPSA":
        if str(config.spsa_refit_engine_label) == LEGACY_SPSA_POLISH_OPTIMIZER_LABEL:
            return spsa_energy_descent_minimize(
                fun=fun,
                x0=np.asarray(x0, dtype=float),
                maxiter=int(maxiter_value),
                seed=int(seed_value),
                schedule=SPSAEnergyDescentSchedule(
                    a=float(spsa.a),
                    c=float(spsa.c),
                    alpha=float(spsa.alpha),
                    gamma=float(spsa.gamma),
                    big_a=float(spsa.A),
                ),
                max_abs_step=0.25,
                accept_tol=1e-12,
                bounds=None,
                project="none",
                callback=callback,
                callback_every=int(spsa.callback_every),
            )
        if str(config.spsa_refit_engine_label) != NATIVE_SPSA_OPTIMIZER_LABEL:
            raise ValueError(
                f"Unsupported ADAPT SPSA refit engine: {config.spsa_refit_engine_label!r}"
            )
        return spsa_minimize(
            fun=fun,
            x0=np.asarray(x0, dtype=float),
            maxiter=int(maxiter_value),
            seed=int(seed_value),
            a=float(spsa.a),
            c=float(spsa.c),
            alpha=float(spsa.alpha),
            gamma=float(spsa.gamma),
            A=float(spsa.A),
            bounds=None,
            project="none",
            eval_repeats=int(spsa.eval_repeats),
            eval_agg=str(spsa.eval_agg),
            avg_last=int(spsa.avg_last),
            callback=callback,
            callback_every=int(spsa.callback_every),
            memory=(dict(memory) if isinstance(memory, Mapping) else None),
            refresh_every=0,
            precondition_mode=str(precondition_mode),
            parallel_evaluations=int(spsa.parallel_evaluations),
        )
    if local_optimizer_key == "QNSPSA":
        if fidelity is None:
            raise ValueError("QNSPSA requires a repo-native state fidelity callable.")
        return qnspsa_minimize(
            fun=fun,
            fidelity=fidelity,
            x0=np.asarray(x0, dtype=float),
            maxiter=int(maxiter_value),
            seed=int(seed_value),
            a=float(spsa.a),
            c=float(spsa.c),
            alpha=float(spsa.alpha),
            gamma=float(spsa.gamma),
            A=float(spsa.A),
            bounds=None,
            project="none",
            eval_repeats=int(spsa.eval_repeats),
            eval_agg=str(spsa.eval_agg),
            avg_last=int(spsa.avg_last),
            callback=callback,
            callback_every=int(spsa.callback_every),
            memory=None,
            refresh_every=0,
            precondition_mode="qnspsa",
        )
    raise ValueError(f"Unsupported stochastic ADAPT inner optimizer: {local_optimizer_key}")


def run_deterministic_inner_optimizer(
    *,
    config: AdaptInnerOptimizerConfig,
    method_key: str,
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    maxiter_value: int,
    context_label: str,
    callback: Callable[[Mapping[str, Any]], None] | None = None,
    rotosolve_period: Any | None = None,
    rotosolve_shift: Any | None = None,
) -> Any:
    method = str(method_key).strip().upper()
    if method == "ROTOSOLVE":
        return _run_rotosolve_adapt_optimizer(
            objective=objective,
            x0=np.asarray(x0, dtype=float),
            maxiter=int(maxiter_value),
            context_label=str(context_label),
            callback=callback,
            period=rotosolve_period,
            shift=rotosolve_shift,
        )
    from scipy.optimize import minimize as scipy_minimize

    return _run_scipy_adapt_optimizer(
        method_key=method,
        objective=objective,
        x0=np.asarray(x0, dtype=float),
        maxiter=int(maxiter_value),
        maxfev=config.scipy_maxfev,
        context_label=str(context_label),
        scipy_minimize_fn=scipy_minimize,
    )
