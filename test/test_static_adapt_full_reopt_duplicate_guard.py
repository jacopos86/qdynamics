from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.exact_bench.static_benchmark_runtime import (
    StaticScaffoldPolicy,
)
from pipelines.static_adapt.extensions import (
    BATCH_RUNTIME_KEYS,
    BEAM_RUNTIME_KEYS,
    PRUNING_RUNTIME_KEYS,
)
from src.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian


_REMOVED_SYMBOLS = (
    "_zero_gain_duplicate_identity",
    "_filter_zero_gain_duplicate_records",
    "_prefilter_zero_gain_duplicate_records",
    "_replace_cooled_zero_gain_fallback_feature",
    "_zero_gain_duplicate_policy_enabled",
    "_zero_gain_duplicate_cooldown_record_keys",
    "_build_zero_gain_duplicate_guard_payload",
)

_REMOVED_TELEMETRY_FIELDS = {
    "depth_rollback",
    "structural_rollback",
    "rollback_mode",
    "rollback_tolerance",
    "zero_gain_duplicate_filter",
    "zero_gain_duplicate_guard",
}


def _hubbard_problem():
    return build_hubbard_hamiltonian(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.0,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )


def _run_flat_two_rounds() -> dict[str, object]:
    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(
        h_poly=_hubbard_problem(),
        num_sites=2,
        ordering="blocked",
        problem="hubbard",
        adapt_pool="uccsd",
        t=1.0,
        u=4.0,
        dv=0.0,
        boundary="periodic",
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        max_depth=2,
        eps_grad=0.0,
        eps_energy=0.0,
        maxiter=1,
        seed=11,
        allow_repeats=True,
        finite_angle_fallback=False,
        finite_angle=0.1,
        finite_angle_min_improvement=1e-12,
        adapt_reopt_policy="full",
        adapt_final_full_refit=False,
    )
    return payload


@pytest.fixture
def flat_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    original_resolve_pool_plan = adapt_pipeline.resolve_pool_plan

    def _flat_run(_dispatch, *, fun, x0, **_kwargs):
        x = np.asarray(x0, dtype=float).copy()
        return SimpleNamespace(
            x=x,
            fun=float(fun(x)),
            nfev=1,
            nit=0,
            success=True,
            message="flat_test_optimizer",
        )

    monkeypatch.setattr(
        adapt_pipeline.AdaptInnerOptimizerDispatch,
        "run_stochastic",
        _flat_run,
    )

    def _single_generator_pool(*args, **kwargs):
        plan = original_resolve_pool_plan(*args, **kwargs)
        return replace(
            plan,
            pool=plan.pool[:1],
            pool_stage_family=plan.pool_stage_family[:1],
            pool_family_ids=plan.pool_family_ids[:1],
            phase1_core_limit=1,
            phase1_residual_indices=set(),
        )

    monkeypatch.setattr(
        adapt_pipeline,
        "resolve_pool_plan",
        _single_generator_pool,
    )


def test_removed_duplicate_guard_symbols_are_absent() -> None:
    assert all(not hasattr(adapt_pipeline, name) for name in _REMOVED_SYMBOLS)


def test_retained_static_policy_has_no_admission_rollback_controls() -> None:
    field_names = {field.name for field in fields(StaticScaffoldPolicy)}
    assert "adapt_rollback_mode" not in field_names
    assert "adapt_rollback_tolerance" not in field_names
    assert (
        BATCH_RUNTIME_KEYS | BEAM_RUNTIME_KEYS | PRUNING_RUNTIME_KEYS
    ).isdisjoint(field_names)


def test_flat_consecutive_repeat_is_committed_without_rollback(
    flat_optimizer: None,
) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        payload = _run_flat_two_rounds()

    history = list(payload.get("history", []))
    assert len(history) == 2
    assert payload.get("ansatz_depth") == 2
    assert history[0]["selected_op"] == history[1]["selected_op"]
    assert history[0]["delta_energy"] == pytest.approx(0.0)
    assert history[1]["delta_energy"] == pytest.approx(0.0)
    assert all(row.get("selected_logical_size") == 1 for row in history)
    assert all(
        _REMOVED_TELEMETRY_FIELDS.isdisjoint(row)
        for row in history
    )


def test_ranked_admission_has_no_absolute_score_or_repeat_threshold() -> None:
    records = [
        {"candidate_label": "same", "selector_score": -0.1},
        {"candidate_label": "same", "selector_score": -2.0},
    ]
    assert adapt_pipeline._ranked_admission_records(records) == records


def test_final_refit_metadata_has_no_rollback_field() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(
            h_poly=_hubbard_problem(),
            num_sites=2,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=1.0,
            u=4.0,
            dv=0.0,
            boundary="periodic",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-2,
            eps_energy=1e-6,
            maxiter=2,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="full",
            adapt_final_full_refit=True,
        )
    assert "rollback" not in payload.get("final_full_refit", {})


def test_final_refit_regression_is_retained_without_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def _controlled_run(_dispatch, *, fun, x0, **_kwargs):
        nonlocal call_count
        call_count += 1
        x = np.asarray(x0, dtype=float).copy()
        energy = float(fun(x))
        if call_count == 3:
            energy += 1.0
        return SimpleNamespace(
            x=x,
            fun=energy,
            nfev=1,
            nit=0,
            success=True,
            message="controlled_final_refit_regression",
        )

    monkeypatch.setattr(
        adapt_pipeline.AdaptInnerOptimizerDispatch,
        "run_stochastic",
        _controlled_run,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(
            h_poly=_hubbard_problem(),
            num_sites=2,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=1.0,
            u=4.0,
            dv=0.0,
            boundary="periodic",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=0.0,
            eps_energy=0.0,
            maxiter=1,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_full_refit_every=0,
            adapt_final_full_refit=True,
        )

    final_refit = payload["final_full_refit"]
    assert call_count == 3
    assert final_refit["executed"] is True
    assert final_refit["energy_after"] > final_refit["energy_before"]
    assert payload["energy"] == pytest.approx(final_refit["energy_after"])
    assert "rollback" not in final_refit


def test_append_only_final_refit_request_semantics_unchanged() -> None:
    assert adapt_pipeline._final_full_refit_policy_supported("append_only") is False
    assert adapt_pipeline._final_full_refit_policy_supported("windowed") is True
    assert adapt_pipeline._final_full_refit_policy_supported("full") is True
