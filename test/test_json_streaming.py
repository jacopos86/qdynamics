"""Byte-identity tests for the chunked JSON encoder used by checkpoint writers."""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from pipelines.static_adapt import json_streaming
from pipelines.static_adapt.json_streaming import (
    iter_canonical_json,
    iter_pretty_json,
)


def _reference_compact(value, *, ensure_ascii=True, allow_nan=False) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )


def _reference_pretty(value, *, ensure_ascii=True, allow_nan=False) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )


def _random_scalar(rng: random.Random):
    kind = rng.randrange(8)
    if kind == 0:
        return None
    if kind == 1:
        return rng.random() < 0.5
    if kind == 2:
        return rng.randrange(-10**12, 10**12)
    if kind == 3:
        return rng.choice([0.0, -0.0, 1e300, -1e-300, 0.1, 1.5, 2.0 ** 60, 3.141592653589793])
    if kind == 4:
        return rng.uniform(-1e6, 1e6)
    if kind == 5:
        return "".join(rng.choice("abc xyz\"\\/\n\t") for _ in range(rng.randrange(0, 12)))
    if kind == 6:
        return "".join(rng.choice("ünï  €𝄞 ") for _ in range(rng.randrange(0, 6)))
    return ""


def _random_value(rng: random.Random, depth: int, *, str_keys_only: bool = True):
    if depth <= 0 or rng.random() < 0.35:
        return _random_scalar(rng)
    if rng.random() < 0.5:
        keys = [f"k{rng.randrange(40)}" for _ in range(rng.randrange(0, 7))]
        if not str_keys_only and rng.random() < 0.2:
            keys.append(rng.randrange(5))  # type: ignore[arg-type]
        return {key: _random_value(rng, depth - 1, str_keys_only=str_keys_only) for key in keys}
    length = rng.choice([0, 1, 2, 5, 17, 300, 700, 2100, 4200])
    if rng.random() < 0.5:
        # Long list of small leaves (the occurrence-row shape).
        return [
            {"sequence": index, "primitive_id": f"p{index % 97}", "charged": index % 3 == 0, "branch_id": None}
            for index in range(length)
        ]
    return [_random_value(rng, depth - 1, str_keys_only=str_keys_only) for _ in range(min(length, 9))]


@pytest.mark.parametrize("seed", list(range(40)))
def test_compact_and_pretty_match_json_module(seed: int) -> None:
    rng = random.Random(seed)
    value = _random_value(rng, depth=5)
    assert "".join(iter_canonical_json(value)) == _reference_compact(value)
    assert "".join(iter_pretty_json(value)) == _reference_pretty(value)


@pytest.mark.parametrize("seed", list(range(10)))
def test_ensure_ascii_false_matches(seed: int) -> None:
    rng = random.Random(1000 + seed)
    value = _random_value(rng, depth=4)
    assert "".join(iter_canonical_json(value, ensure_ascii=False)) == _reference_compact(
        value, ensure_ascii=False
    )
    assert "".join(iter_pretty_json(value, ensure_ascii=False)) == _reference_pretty(
        value, ensure_ascii=False
    )


def test_edge_containers() -> None:
    cases = [
        {},
        [],
        [[]],
        [{}],
        {"a": {}},
        {"a": []},
        {"a": [[], {}, [[]], [{}]]},
        [[[[1]]]],
        {"z": 1, "a": [1, 2, {"q": [None, True, False, 1.0]}]},
        [1] * 5000,
        [{"x": i} for i in range(4097)],
        {"big": [{"x": i, "y": [i, i + 1]} for i in range(3000)], "small": 1},
        # A short list whose elements are large containers.
        [{"rows": [{"i": i} for i in range(600)]} for _ in range(3)],
        # Mixed: leaves interleaved with large containers in one list.
        [1, {"rows": list(range(1000))}, 2, 3, {"rows": list(range(1000))}, 4],
    ]
    for value in cases:
        assert "".join(iter_canonical_json(value)) == _reference_compact(value)
        assert "".join(iter_pretty_json(value)) == _reference_pretty(value)


def test_non_string_keys_fall_back_identically() -> None:
    value = {"outer": {1: "a", 2: "b"}, "list": [{3: [1, 2]}]}
    assert "".join(iter_canonical_json(value)) == _reference_compact(value)
    assert "".join(iter_pretty_json(value)) == _reference_pretty(value)
    mixed = {"outer": {1: "a", "b": 2}}
    with pytest.raises(TypeError):
        _reference_compact(mixed)
    with pytest.raises(TypeError):
        "".join(iter_canonical_json(mixed))


def test_nan_rejected_like_json_module() -> None:
    value = {"rows": [float("nan")] * 3000}
    with pytest.raises(ValueError):
        _reference_compact(value)
    with pytest.raises(ValueError):
        "".join(iter_canonical_json(value))
    assert "".join(iter_canonical_json(value, allow_nan=True)) == _reference_compact(
        value, allow_nan=True
    )


def test_chunks_are_bounded_for_long_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(json_streaming, "TARGET_SLICE_BYTES", 4096)
    monkeypatch.setattr(json_streaming, "RECURSE_CHILD_BYTES", 8 * 4096)
    monkeypatch.setattr(json_streaming, "MAX_SLICE_ITEMS", 64)
    rows = [{"sequence": i, "primitive_id": "p", "charged": False} for i in range(10_000)]
    chunks = list(iter_canonical_json({"occurrences": rows}))
    assert max(len(chunk) for chunk in chunks) < 4 * 4096
    assert len(chunks) > 100
    assert "".join(chunks) == _reference_compact({"occurrences": rows})
    pretty_chunks = list(iter_pretty_json({"occurrences": rows}))
    assert max(len(chunk) for chunk in pretty_chunks) < 6 * 4096
    assert "".join(pretty_chunks) == _reference_pretty({"occurrences": rows})


def test_large_lone_child_is_recursed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(json_streaming, "TARGET_SLICE_BYTES", 2048)
    monkeypatch.setattr(json_streaming, "RECURSE_CHILD_BYTES", 8 * 2048)
    big_row = {"population": [{"i": i, "g": float(i)} for i in range(5000)]}
    value = {"history": [big_row, big_row, {"tiny": 1}], "after": [1, 2, 3]}
    chunks = list(iter_canonical_json(value))
    assert max(len(chunk) for chunk in chunks) < 16 * 2048 + 64
    assert "".join(chunks) == _reference_compact(value)
    pretty_chunks = list(iter_pretty_json(value))
    assert "".join(pretty_chunks) == _reference_pretty(value)


def test_digest_equivalence() -> None:
    rng = random.Random(7)
    value = {"history": [_random_value(rng, depth=4) for _ in range(30)]}
    digest = hashlib.sha256()
    for chunk in iter_canonical_json(value):
        digest.update(chunk.encode("utf-8"))
    expected = hashlib.sha256(_reference_compact(value).encode("utf-8")).hexdigest()
    assert digest.hexdigest() == expected
