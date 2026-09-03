"""Checkpoint JSON writer: byte identity of the default layout and digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.static_adapt import current_checkpoint as cc


def _payload() -> dict:
    return {
        "schema": "test_v1",
        "generated": "2026-09-03T00:00:00+00:00",
        "ledger": {
            "entries": [
                {"primitive_id": f"p{index:04d}", "occurrence_count": index % 3 + 1}
                for index in range(1500)
            ],
            "occurrences": [
                {
                    "sequence": index,
                    "primitive_id": f"p{index % 97:04d}",
                    "component": "N_grad",
                    "consumer_scope": "phase0",
                    "branch_id": None,
                    "charged": index % 5 == 0,
                }
                for index in range(1, 4001)
            ],
        },
        "history": [{"depth": d, "energy": -1.0 / (d + 1), "unicode": "ü€𝄞"} for d in range(7)],
        "empty": {},
        "empty_list": [],
    }


def _reference_compact(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _reference_pretty(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def test_stable_json_digest_matches_reference_encoding() -> None:
    payload = _payload()
    expected = hashlib.sha256(_reference_compact(payload).encode("utf-8")).hexdigest()
    assert cc._stable_json_digest(payload) == expected
    assert cc._stable_json_digest([]) == hashlib.sha256(b"[]").hexdigest()
    with pytest.raises(ValueError):
        cc._stable_json_digest({"a": float("nan")})


def test_default_pretty_layout_is_byte_identical_and_self_digesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, raising=False)
    payload = _payload()
    current = tmp_path / "current.json"
    destination, sha256 = cc._atomic_write_content_addressed_json(
        current, filename_role="estimator_call_ledger_checkpoint", payload=payload
    )
    expected_bytes = (_reference_pretty(payload) + "\n").encode("utf-8")
    assert destination.read_bytes() == expected_bytes
    assert sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert sha256 == cc._sha256_file(destination)
    assert destination.name == f"current.estimator_call_ledger_checkpoint.{sha256[:16]}.json"
    assert not list(tmp_path.glob(".current.json.*.tmp"))

    cc._atomic_write_json(current, payload)
    assert current.read_bytes() == expected_bytes


def test_compact_layout_writes_canonical_bytes_with_matching_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, "compact")
    payload = _payload()
    current = tmp_path / "current.json"
    destination, sha256 = cc._atomic_write_content_addressed_json(
        current, filename_role="estimator_call_ledger_checkpoint", payload=payload
    )
    expected_bytes = (_reference_compact(payload) + "\n").encode("utf-8")
    assert destination.read_bytes() == expected_bytes
    assert sha256 == cc._sha256_file(destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert len(expected_bytes) < len((_reference_pretty(payload) + "\n").encode("utf-8"))
    # The projection digest does not depend on the on-disk layout.
    monkeypatch.delenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, raising=False)
    assert cc._stable_json_digest(payload) == hashlib.sha256(
        _reference_compact(payload).encode("utf-8")
    ).hexdigest()


def test_invalid_layout_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, "yaml")
    with pytest.raises(ValueError):
        cc._atomic_write_json(tmp_path / "current.json", {"a": 1})
    monkeypatch.setenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, " Pretty ")
    assert cc._checkpoint_json_layout() == "pretty"


def test_temporary_file_is_removed_when_encoding_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, raising=False)
    with pytest.raises(ValueError):
        cc._atomic_write_json(tmp_path / "current.json", {"a": float("nan")})
    assert not list(tmp_path.iterdir())
