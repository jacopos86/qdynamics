from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.static_adapt import optimizer_routes
from src.quantum.adapt_spsa_refit import LEGACY_SPSA_POLISH_OPTIMIZER_LABEL, NATIVE_SPSA_OPTIMIZER_LABEL


def _config(
    key: str = "SPSA",
    *,
    refit_engine_label: str = NATIVE_SPSA_OPTIMIZER_LABEL,
) -> optimizer_routes.AdaptInnerOptimizerConfig:
    return optimizer_routes.AdaptInnerOptimizerConfig(
        inner_optimizer_key=key,
        spsa_refit_engine_label=refit_engine_label,
        spsa_refit_engine_env="ADAPT_SPSA_REFIT_ENGINE",
        spsa=optimizer_routes.AdaptSPSAConfig(
            a=0.1,
            c=0.2,
            alpha=0.602,
            gamma=0.101,
            A=5.0,
            avg_last=3,
            eval_repeats=2,
            eval_agg="median",
            callback_every=4,
            progress_every_s=0.5,
            parallel_evaluations=2,
        ),
    )


def test_resolve_adapt_inner_optimizer_config_and_payload() -> None:
    cfg = optimizer_routes.resolve_adapt_inner_optimizer_config(
        adapt_inner_optimizer="qnspsa",
        adapt_spsa_a=0.1,
        adapt_spsa_c=0.2,
        adapt_spsa_alpha=0.6,
        adapt_spsa_gamma=0.1,
        adapt_spsa_A=5.0,
        adapt_spsa_avg_last=3,
        adapt_spsa_eval_repeats=2,
        adapt_spsa_eval_agg="median",
        adapt_spsa_callback_every=4,
        adapt_spsa_progress_every_s=0.5,
        adapt_spsa_parallel_evaluations=2,
    )

    assert cfg.inner_optimizer_key == "QNSPSA"
    assert cfg.spsa.eval_agg == "median"
    assert cfg.spsa.parallel_evaluations == 2
    assert optimizer_routes.stochastic_heartbeat_event("QNSPSA") == (
        "hardcoded_adapt_qnspsa_heartbeat"
    )
    assert optimizer_routes.stochastic_heartbeat_event("SPSA") == (
        "hardcoded_adapt_spsa_heartbeat"
    )

    payload = optimizer_routes.adapt_spsa_params_payload(cfg)
    assert payload == {
        "refit_engine": cfg.spsa_refit_engine_label,
        "refit_engine_env": cfg.spsa_refit_engine_env,
        "a": 0.1,
        "c": 0.2,
        "alpha": 0.6,
        "gamma": 0.1,
        "A": 5.0,
        "avg_last": 3,
        "eval_repeats": 2,
        "eval_agg": "median",
        "callback_every": 4,
        "progress_every_s": 0.5,
        "scipy_maxfev": None,
    }


def test_optimizer_config_helpers_resolve_keys_progress_and_events() -> None:
    cfg = _config("SPSA")

    assert optimizer_routes.resolve_effective_adapt_optimizer_key(cfg) == "SPSA"
    assert optimizer_routes.resolve_effective_adapt_optimizer_key(cfg, optimizer_key=None) == "SPSA"
    assert optimizer_routes.resolve_effective_adapt_optimizer_key(cfg, optimizer_key="") == "SPSA"
    assert optimizer_routes.resolve_effective_adapt_optimizer_key(cfg, optimizer_key="  ") == "SPSA"
    assert optimizer_routes.resolve_effective_adapt_optimizer_key(cfg, optimizer_key="qnspsa") == "QNSPSA"
    assert optimizer_routes.adapt_optimizer_progress_interval_s(cfg) == pytest.approx(0.5)
    assert optimizer_routes.stochastic_heartbeat_event_from_config(cfg) == (
        "hardcoded_adapt_spsa_heartbeat"
    )
    assert optimizer_routes.stochastic_heartbeat_event_from_config(
        cfg,
        optimizer_key="qnspsa",
    ) == "hardcoded_adapt_qnspsa_heartbeat"
    assert optimizer_routes.stochastic_heartbeat_event_from_config(_config("QNSPSA")) == (
        "hardcoded_adapt_qnspsa_heartbeat"
    )


def test_optimizer_dispatch_binds_one_resolved_config(monkeypatch) -> None:
    cfg = _config("SPSA")
    dispatch = optimizer_routes.AdaptInnerOptimizerDispatch(cfg)
    seen: list[tuple[str, object]] = []

    def fake_stochastic(**kwargs):
        seen.append(("stochastic", kwargs["config"]))
        return SimpleNamespace(fun=1.0)

    def fake_deterministic(**kwargs):
        seen.append(("deterministic", kwargs["config"]))
        return SimpleNamespace(fun=2.0)

    monkeypatch.setattr(
        optimizer_routes,
        "run_stochastic_inner_optimizer",
        fake_stochastic,
    )
    monkeypatch.setattr(
        optimizer_routes,
        "run_deterministic_inner_optimizer",
        fake_deterministic,
    )

    stochastic = dispatch.run_stochastic(
        fun=lambda x: float(np.sum(x)),
        x0=np.asarray([0.1]),
        fidelity=None,
        maxiter_value=2,
        seed_value=3,
    )
    deterministic = dispatch.run_deterministic(
        method_key="POWELL",
        objective=lambda x: float(np.sum(x)),
        x0=np.asarray([0.2]),
        maxiter_value=4,
        context_label="unit",
    )

    assert stochastic.fun == 1.0
    assert deterministic.fun == 2.0
    assert seen == [("stochastic", cfg), ("deterministic", cfg)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"adapt_inner_optimizer": "bad"}, "adapt_inner_optimizer"),
        ({"adapt_spsa_eval_agg": "mode"}, "adapt_spsa_eval_agg"),
        ({"adapt_spsa_callback_every": 0}, "adapt_spsa_callback_every"),
        ({"adapt_spsa_progress_every_s": -1.0}, "adapt_spsa_progress_every_s"),
        ({"adapt_spsa_parallel_evaluations": 0}, "adapt_spsa_parallel_evaluations"),
    ],
)
def test_resolve_adapt_inner_optimizer_config_validates_inputs(kwargs, message) -> None:
    base = {
        "adapt_inner_optimizer": "SPSA",
        "adapt_spsa_a": 0.1,
        "adapt_spsa_c": 0.2,
        "adapt_spsa_alpha": 0.6,
        "adapt_spsa_gamma": 0.1,
        "adapt_spsa_A": 5.0,
        "adapt_spsa_avg_last": 3,
        "adapt_spsa_eval_repeats": 2,
        "adapt_spsa_eval_agg": "mean",
        "adapt_spsa_callback_every": 1,
        "adapt_spsa_progress_every_s": 0.0,
        "adapt_spsa_parallel_evaluations": 1,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        optimizer_routes.resolve_adapt_inner_optimizer_config(**base)


def test_run_stochastic_inner_optimizer_forwards_native_spsa(monkeypatch) -> None:
    seen = {}

    def fake_spsa_minimize(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(x=np.asarray(kwargs["x0"]), fun=1.0)

    monkeypatch.setattr(optimizer_routes, "spsa_minimize", fake_spsa_minimize)

    result = optimizer_routes.run_stochastic_inner_optimizer(
        config=_config("SPSA"),
        fun=lambda x: float(np.sum(x)),
        x0=np.asarray([1.0, 2.0]),
        fidelity=None,
        maxiter_value=5,
        seed_value=11,
        memory={"inverse_hessian": [[1.0]]},
        precondition_mode="phase2",
    )

    assert result.fun == 1.0
    assert seen["maxiter"] == 5
    assert seen["seed"] == 11
    assert seen["eval_agg"] == "median"
    assert seen["avg_last"] == 3
    assert seen["callback_every"] == 4
    assert seen["parallel_evaluations"] == 2
    assert seen["precondition_mode"] == "phase2"
    assert seen["memory"] == {"inverse_hessian": [[1.0]]}


def test_run_stochastic_inner_optimizer_forwards_legacy_spsa(monkeypatch) -> None:
    seen = {}
    callback = lambda payload: None

    def fake_legacy_spsa(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(x=np.asarray(kwargs["x0"]), fun=2.0)

    monkeypatch.setattr(optimizer_routes, "spsa_energy_descent_minimize", fake_legacy_spsa)

    result = optimizer_routes.run_stochastic_inner_optimizer(
        config=_config("SPSA", refit_engine_label=LEGACY_SPSA_POLISH_OPTIMIZER_LABEL),
        fun=lambda x: float(np.sum(x)),
        x0=np.asarray([1.0, 2.0]),
        fidelity=None,
        maxiter_value=5,
        seed_value=11,
        callback=callback,
    )

    assert result.fun == 2.0
    assert seen["maxiter"] == 5
    assert seen["seed"] == 11
    assert seen["max_abs_step"] == 0.25
    assert seen["accept_tol"] == 1e-12
    assert seen["callback"] is callback
    assert seen["callback_every"] == 4
    assert seen["schedule"].a == 0.1
    assert seen["schedule"].c == 0.2
    assert seen["schedule"].alpha == 0.602
    assert seen["schedule"].gamma == 0.101
    assert seen["schedule"].big_a == 5.0


def test_run_stochastic_inner_optimizer_qnspsa_requires_and_forwards_fidelity(monkeypatch) -> None:
    cfg = _config("QNSPSA")
    with pytest.raises(ValueError, match="QNSPSA requires"):
        optimizer_routes.run_stochastic_inner_optimizer(
            config=cfg,
            fun=lambda x: float(np.sum(x)),
            x0=np.asarray([1.0]),
            fidelity=None,
            maxiter_value=1,
            seed_value=1,
        )

    seen = {}

    def fake_qnspsa_minimize(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(x=np.asarray(kwargs["x0"]), fun=0.0)

    monkeypatch.setattr(optimizer_routes, "qnspsa_minimize", fake_qnspsa_minimize)
    fidelity = lambda x, y: 1.0

    result = optimizer_routes.run_stochastic_inner_optimizer(
        config=cfg,
        fun=lambda x: float(np.sum(x)),
        x0=np.asarray([1.0]),
        fidelity=fidelity,
        maxiter_value=2,
        seed_value=3,
        memory={"ignored": True},
    )

    assert result.fun == 0.0
    assert seen["fidelity"] is fidelity
    assert seen["memory"] is None
    assert seen["precondition_mode"] == "qnspsa"


def test_run_deterministic_inner_optimizer_forwards_rotosolve(monkeypatch) -> None:
    seen = {}

    def fake_rotosolve(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(x=np.asarray(kwargs["x0"]), fun=-1.0)

    monkeypatch.setattr(
        optimizer_routes,
        "_run_rotosolve_adapt_optimizer",
        fake_rotosolve,
    )

    result = optimizer_routes.run_deterministic_inner_optimizer(
        config=_config("ROTOSOLVE"),
        method_key="ROTOSOLVE",
        objective=lambda x: float(np.sum(x)),
        x0=np.asarray([0.1]),
        maxiter_value=7,
        context_label="unit",
        rotosolve_period=[1.0],
        rotosolve_shift=[0.5],
    )

    assert result.fun == -1.0
    assert seen["maxiter"] == 7
    assert seen["context_label"] == "unit"
    assert seen["period"] == [1.0]
    assert seen["shift"] == [0.5]


def test_run_deterministic_inner_optimizer_forwards_scipy(monkeypatch) -> None:
    seen = {}
    callback = lambda payload: None

    def fake_scipy_dispatch(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(x=np.asarray(kwargs["x0"]), fun=-2.0)

    monkeypatch.setattr(
        optimizer_routes,
        "_run_scipy_adapt_optimizer",
        fake_scipy_dispatch,
    )

    result = optimizer_routes.run_deterministic_inner_optimizer(
        config=_config("POWELL"),
        method_key="POWELL",
        objective=lambda x: float(np.sum(x)),
        x0=np.asarray([0.1, 0.2]),
        maxiter_value=8,
        context_label="unit-scipy",
        callback=callback,
    )

    assert result.fun == -2.0
    assert seen["method_key"] == "POWELL"
    assert seen["maxiter"] == 8
    assert seen["context_label"] == "unit-scipy"
    assert "callback" not in seen
    assert callable(seen["scipy_minimize_fn"])
