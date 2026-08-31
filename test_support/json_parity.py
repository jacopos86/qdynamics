from __future__ import annotations

import math
from typing import Any, Iterable

import pytest


"Built Math: actual ≈ expected on approved float paths; otherwise actual = expected exactly."
def assert_json_parity(
    actual: Any,
    expected: Any,
    *,
    ignore_paths: Iterable[tuple[str | int, ...]] = (),
    approx_prefixes: Iterable[tuple[str | int, ...]] = (),
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> None:
    ignore = {tuple(path) for path in ignore_paths}
    approx = tuple(tuple(path) for path in approx_prefixes)

    def _approx_path(path: tuple[str | int, ...]) -> bool:
        return any(path[: len(prefix)] == prefix for prefix in approx)

    def _assert(lhs: Any, rhs: Any, path: tuple[str | int, ...]) -> None:
        if path in ignore:
            return
        if isinstance(rhs, dict):
            assert isinstance(lhs, dict), f"Expected dict at {path}, got {type(lhs)!r}"
            assert set(lhs.keys()) == set(rhs.keys()), f"Key mismatch at {path}: {set(lhs.keys())} != {set(rhs.keys())}"
            for key in rhs:
                _assert(lhs[key], rhs[key], path + (key,))
            return
        if isinstance(rhs, list):
            assert isinstance(lhs, list), f"Expected list at {path}, got {type(lhs)!r}"
            assert len(lhs) == len(rhs), f"Length mismatch at {path}: {len(lhs)} != {len(rhs)}"
            for idx, (lhs_item, rhs_item) in enumerate(zip(lhs, rhs)):
                _assert(lhs_item, rhs_item, path + (idx,))
            return
        if isinstance(rhs, float):
            assert isinstance(lhs, (int, float)), f"Expected numeric at {path}, got {type(lhs)!r}"
            if math.isnan(rhs):
                assert math.isnan(float(lhs)), f"Expected NaN at {path}, got {lhs!r}"
                return
            if _approx_path(path):
                assert float(lhs) == pytest.approx(float(rhs), abs=atol, rel=rtol), f"Float mismatch at {path}"
                return
        assert lhs == rhs, f"Mismatch at {path}: {lhs!r} != {rhs!r}"

    _assert(actual, expected, ())
