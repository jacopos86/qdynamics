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
    assert not any("maturity" in field_name for field_name in field_names)
    assert (
        BATCH_RUNTIME_KEYS | BEAM_RUNTIME_KEYS | PRUNING_RUNTIME_KEYS
    ).isdisjoint(field_names)




def test_ranked_admission_has_no_absolute_score_or_repeat_threshold() -> None:
    records = [
        {"candidate_label": "same", "selector_score": -0.1},
        {"candidate_label": "same", "selector_score": -2.0},
    ]
    assert adapt_pipeline._ranked_admission_records(records) == records






def test_append_only_final_refit_request_semantics_unchanged() -> None:
    assert adapt_pipeline._final_full_refit_policy_supported("append_only") is False
    assert adapt_pipeline._final_full_refit_policy_supported("windowed") is True
    assert adapt_pipeline._final_full_refit_policy_supported("full") is True
