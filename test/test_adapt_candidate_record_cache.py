from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.static_adapt import adapt_candidate_record_cache as cache


def _clear_cache() -> None:
    with cache._CANDIDATE_RECORD_CACHE_LOCK:
        cache._CANDIDATE_RECORD_MEMORY_CACHE.clear()


def test_candidate_record_cache_mode_and_dir_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(cache._CANDIDATE_RECORD_CACHE_ENV, raising=False)
    monkeypatch.delenv(cache._CANDIDATE_RECORD_CACHE_DIR_ENV, raising=False)
    assert cache._candidate_record_cache_mode() == "disk"
    assert cache._candidate_record_cache_dir() == (
        cache.REPO_ROOT / "raw_outputs" / "cache" / "static_adapt_candidate_records_v1"
    )

    monkeypatch.setenv(cache._CANDIDATE_RECORD_CACHE_ENV, "mem")
    assert cache._candidate_record_cache_mode() == "memory"
    monkeypatch.setenv(cache._CANDIDATE_RECORD_CACHE_ENV, "off")
    assert cache._candidate_record_cache_mode() == "off"
    monkeypatch.setenv(cache._CANDIDATE_RECORD_CACHE_ENV, "yes")
    assert cache._candidate_record_cache_mode() == "disk"

    monkeypatch.setenv(cache._CANDIDATE_RECORD_CACHE_DIR_ENV, str(tmp_path / "cache-dir"))
    assert cache._candidate_record_cache_dir() == tmp_path / "cache-dir"

    monkeypatch.setenv(cache._CANDIDATE_RECORD_CACHE_ENV, "bad-mode")
    with pytest.raises(ValueError, match="STATIC_ADAPT_CANDIDATE_RECORD_CACHE must be one of disk, memory, or off"):
        cache._candidate_record_cache_mode()


def test_candidate_record_memory_cache_limit_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        cache._CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV,
        raising=False,
    )
    assert cache._candidate_record_memory_cache_max_entries() == 512

    monkeypatch.setenv(
        cache._CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV,
        "3",
    )
    assert cache._candidate_record_memory_cache_max_entries() == 3

    monkeypatch.setenv(
        cache._CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV,
        "0",
    )
    with pytest.raises(ValueError, match="must be >= 1"):
        cache._candidate_record_memory_cache_max_entries()


def test_candidate_record_array_fingerprint_and_payload_digest_are_deterministic() -> None:
    arr = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    same = np.asfortranarray(arr)
    different_dtype = arr.astype(np.float32)

    fp = cache._candidate_record_array_fingerprint(arr)
    assert fp == cache._candidate_record_array_fingerprint(same)
    assert fp["schema"] == "ndarray_sha256_v1"
    assert fp["shape"] == [2, 2]
    assert fp["dtype"] == "float64"
    assert fp != cache._candidate_record_array_fingerprint(different_dtype)

    digest_a = cache._candidate_record_payload_digest({"b": [2, 1], "a": arr})
    digest_b = cache._candidate_record_payload_digest({"a": same, "b": [2, 1]})
    digest_c = cache._candidate_record_payload_digest({"a": different_dtype, "b": [2, 1]})
    assert digest_a == digest_b
    assert digest_a != digest_c


def test_outer_curvature_prior_changes_candidate_cache_identity() -> None:
    def _prior(fingerprint: str) -> SimpleNamespace:
        return SimpleNamespace(
            prior_fingerprint=fingerprint,
            source_prior_id="prior-1",
            source_state_id="state-1",
            source_frame_id="frame-1",
            source_support_id="support-1",
            source_geometry_status="predicted_unvalidated",
            source_provenance_ids=("transport-1",),
        )

    first = cache._outer_curvature_prior_cache_identity(
        enabled=True,
        prior=_prior("hessian-a"),
    )
    second = cache._outer_curvature_prior_cache_identity(
        enabled=True,
        prior=_prior("hessian-b"),
    )

    assert first is not None and second is not None
    assert first["status"] == "predicted_unvalidated"
    assert first["prior_fingerprint"] == "hessian-a"
    assert second["prior_fingerprint"] == "hessian-b"
    assert first != second
    assert first["source_frame_id"] == "frame-1"
    assert cache._candidate_record_payload_digest(
        {"state": "same", "outer_curvature_prior": first}
    ) != cache._candidate_record_payload_digest(
        {"state": "same", "outer_curvature_prior": second}
    )
    assert cache._outer_curvature_prior_cache_identity(
        enabled=False,
        prior=_prior("ignored"),
    ) is None


def test_candidate_record_cache_jsonable_handles_paths_sets_and_objects(tmp_path: Path) -> None:
    class TinyObject:
        def __init__(self) -> None:
            self.path = tmp_path / "x"
            self.values = {3, 1, 2}

    converted = cache._candidate_record_cache_jsonable(
        {"path": tmp_path / "file", "set": {"b", "a"}, "object": TinyObject()}
    )

    assert converted["path"] == str(tmp_path / "file")
    assert converted["set"] == ["a", "b"]
    assert converted["object"]["schema"] == "object_fields_v1"
    assert converted["object"]["class"] == "TinyObject"
    assert converted["object"]["fields"]["path"] == str(tmp_path / "x")
    assert converted["object"]["fields"]["values"] == [1, 2, 3]


def test_candidate_record_memory_cache_get_put_deepcopies(tmp_path: Path) -> None:
    _clear_cache()
    record = {"value": [1, {"x": 2}]}

    assert cache._candidate_record_cache_put(
        cache_key="abc123",
        cache_dir=tmp_path,
        mode="memory",
        record=record,
    ) == "memory"
    record["value"][1]["x"] = 99

    cached, source = cache._candidate_record_cache_get(cache_key="abc123", cache_dir=tmp_path, mode="memory")
    assert source == "memory"
    assert cached == {"value": [1, {"x": 2}]}
    assert cached is not None
    cached["value"][1]["x"] = 100

    cached_again, source_again = cache._candidate_record_cache_get(cache_key="abc123", cache_dir=tmp_path, mode="memory")
    assert source_again == "memory"
    assert cached_again == {"value": [1, {"x": 2}]}


def test_candidate_record_memory_cache_is_bounded_lru(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cache()
    monkeypatch.setenv(
        cache._CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV,
        "2",
    )
    for key in ("a", "b"):
        assert cache._candidate_record_cache_put(
            cache_key=key,
            cache_dir=tmp_path,
            mode="memory",
            record={"key": key},
        ) == "memory"

    cached_a, source_a = cache._candidate_record_cache_get(
        cache_key="a",
        cache_dir=tmp_path,
        mode="memory",
    )
    assert (cached_a, source_a) == ({"key": "a"}, "memory")
    cache._candidate_record_cache_put(
        cache_key="c",
        cache_dir=tmp_path,
        mode="memory",
        record={"key": "c"},
    )

    assert list(cache._CANDIDATE_RECORD_MEMORY_CACHE) == ["a", "c"]
    assert cache._candidate_record_cache_get(
        cache_key="b",
        cache_dir=tmp_path,
        mode="memory",
    ) == (None, "miss")


def test_candidate_record_disk_cache_get_put_and_path(tmp_path: Path) -> None:
    _clear_cache()
    cache_key = "abcdef012345"
    record = {"score": 1.25, "nested": {"ok": True}}

    assert cache._candidate_record_cache_path(tmp_path, cache_key) == tmp_path / "ab" / f"{cache_key}.pkl"
    assert cache._candidate_record_cache_get(cache_key=cache_key, cache_dir=tmp_path, mode="disk") == (None, "miss")
    assert cache._candidate_record_cache_put(
        cache_key=cache_key,
        cache_dir=tmp_path,
        mode="disk",
        record=record,
    ) == "disk"

    _clear_cache()
    cached, source = cache._candidate_record_cache_get(cache_key=cache_key, cache_dir=tmp_path, mode="disk")
    assert source == "disk"
    assert cached == record

    # Disk reads populate the memory cache and still return deep copies.
    cached["nested"]["ok"] = False
    cached_again, source_again = cache._candidate_record_cache_get(cache_key=cache_key, cache_dir=tmp_path, mode="disk")
    assert source_again == "memory"
    assert cached_again == record


def test_candidate_record_cache_put_copy_failure_is_nonfatal(tmp_path: Path) -> None:
    _clear_cache()

    class Uncopyable:
        def __deepcopy__(self, memo):  # type: ignore[no-untyped-def]
            raise TypeError("not copyable")

    status = cache._candidate_record_cache_put(
        cache_key="uncopyable",
        cache_dir=tmp_path,
        mode="disk",
        record={"live_object": Uncopyable()},
    )

    assert status == "copy_error:TypeError"
    cached, source = cache._candidate_record_cache_get(
        cache_key="uncopyable",
        cache_dir=tmp_path,
        mode="disk",
    )
    assert cached is None
    assert source == "miss"


def test_candidate_record_cache_get_rejects_invalid_payloads(tmp_path: Path) -> None:
    _clear_cache()
    cache_key = "badbad"
    path = cache._candidate_record_cache_path(tmp_path, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as handle:
        pickle.dump({"schema": "wrong", "code_version": cache._CANDIDATE_RECORD_CACHE_CODE_VERSION, "cache_key": cache_key, "record": {}}, handle)
    assert cache._candidate_record_cache_get(cache_key=cache_key, cache_dir=tmp_path, mode="disk") == (None, "schema_mismatch")

    with path.open("wb") as handle:
        pickle.dump({"schema": cache._CANDIDATE_RECORD_CACHE_SCHEMA, "code_version": "old", "cache_key": cache_key, "record": {}}, handle)
    assert cache._candidate_record_cache_get(cache_key=cache_key, cache_dir=tmp_path, mode="disk") == (None, "code_version_mismatch")

    with path.open("wb") as handle:
        pickle.dump({"schema": cache._CANDIDATE_RECORD_CACHE_SCHEMA, "code_version": cache._CANDIDATE_RECORD_CACHE_CODE_VERSION, "cache_key": "other", "record": {}}, handle)
    assert cache._candidate_record_cache_get(cache_key=cache_key, cache_dir=tmp_path, mode="disk") == (None, "key_mismatch")

    with path.open("wb") as handle:
        pickle.dump({"schema": cache._CANDIDATE_RECORD_CACHE_SCHEMA, "code_version": cache._CANDIDATE_RECORD_CACHE_CODE_VERSION, "cache_key": cache_key, "record": []}, handle)
    assert cache._candidate_record_cache_get(cache_key=cache_key, cache_dir=tmp_path, mode="disk") == (None, "record_missing")


def test_adapt_pipeline_preserves_candidate_record_cache_import_compatibility() -> None:
    from pipelines.static_adapt import adapt_pipeline

    for name in (
        "_CANDIDATE_RECORD_CACHE_SCHEMA",
        "_CANDIDATE_RECORD_CACHE_CODE_VERSION",
        "_CANDIDATE_RECORD_CACHE_ENV",
        "_CANDIDATE_RECORD_CACHE_DIR_ENV",
        "_CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV",
        "_CANDIDATE_RECORD_MEMORY_CACHE",
        "_CANDIDATE_RECORD_CACHE_LOCK",
        "_candidate_record_cache_mode",
        "_candidate_record_cache_dir",
        "_candidate_record_cache_path",
        "_candidate_record_array_fingerprint",
        "_candidate_record_cache_jsonable",
        "_candidate_record_memory_cache_max_entries",
        "_candidate_record_payload_digest",
        "_candidate_record_cache_get",
        "_candidate_record_cache_put",
    ):
        assert getattr(adapt_pipeline, name) is getattr(cache, name)
