"""Unit tests for the matched-accuracy campaign's pure cell logic and
reference store (no heavy compute)."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.exact_bench.paper_iii_matched_accuracy_campaign import (
    ERROR_TARGET_LADDER,
    RESIDUAL_RUNG_LADDER,
    STATUS_NOT_IN_POOL,
    STATUS_REACHED,
    STATUS_UNATTAINABLE,
    _reference_key,
    resolve_cell,
)


def test_resolve_cell_selects_minimum_cost_reaching_rung() -> None:
    rungs = [
        {"total_2q": 500.0, "max_root_abs_error": 5.0e-7},
        {"total_2q": 200.0, "max_root_abs_error": 9.0e-5},
        {"total_2q": 900.0, "max_root_abs_error": 1.0e-8},
    ]
    cell = resolve_cell(rungs, 1.0e-4, extendable=True)
    assert cell["status"] == STATUS_REACHED
    assert cell["cost_at_target"] == 200.0
    tighter = resolve_cell(rungs, 1.0e-6, extendable=True)
    assert tighter["cost_at_target"] == 500.0


def test_resolve_cell_failure_states_distinguish_class_from_pool() -> None:
    rungs = [{"total_2q": 100.0, "max_root_abs_error": 3.0e-1}]
    fixed = resolve_cell(rungs, 1.0e-4, extendable=False)
    adaptive = resolve_cell(rungs, 1.0e-4, extendable=True)
    assert fixed["status"] == STATUS_UNATTAINABLE
    assert adaptive["status"] == STATUS_NOT_IN_POOL
    assert fixed["cost_at_target"] is None
    assert fixed["terminal"]["max_root_abs_error"] == 3.0e-1


def test_resolve_cell_ignores_unresolved_windows() -> None:
    rungs = [
        {"total_2q": 50.0, "max_root_abs_error": None},
        {"total_2q": 80.0, "max_root_abs_error": 2.0e-5},
    ]
    cell = resolve_cell(rungs, 1.0e-4, extendable=True)
    assert cell["status"] == STATUS_REACHED
    assert cell["cost_at_target"] == 80.0


def test_ladders_are_declared_and_ordered() -> None:
    assert list(ERROR_TARGET_LADDER) == sorted(ERROR_TARGET_LADDER, reverse=True)
    assert list(RESIDUAL_RUNG_LADDER) == sorted(RESIDUAL_RUNG_LADDER, reverse=True)


def test_reference_key_is_identity_sensitive() -> None:
    base = {"regime": "weak_weak", "u": 0.25, "g_ep": 0.35, "n_ph_max": 3, "count": 7}
    assert _reference_key(base) == _reference_key(dict(base))
    changed = dict(base)
    changed["n_ph_max"] = 7
    assert _reference_key(base) != _reference_key(changed)
