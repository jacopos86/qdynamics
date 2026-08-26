"""Candidate-record cache helpers for static ADAPT execution."""

from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import numpy as np

from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.builders.primitive_pools import _polynomial_signature_digest
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


REPO_ROOT = Path(__file__).resolve().parents[2]

_CANDIDATE_RECORD_CACHE_SCHEMA = "static_adapt_candidate_record_cache_v1"
_CANDIDATE_RECORD_CACHE_CODE_VERSION = (
    "20260731_phase3_candidate_gain_policy_feature_stamp_v3"
)
_CANDIDATE_RECORD_CACHE_ENV = "STATIC_ADAPT_CANDIDATE_RECORD_CACHE"
_CANDIDATE_RECORD_CACHE_DIR_ENV = "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR"
_CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV = (
    "STATIC_ADAPT_CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES"
)
_CANDIDATE_RECORD_MEMORY_CACHE_DEFAULT_MAX_ENTRIES = 512
_CANDIDATE_RECORD_MEMORY_CACHE: OrderedDict[str, Any] = OrderedDict()
_CANDIDATE_RECORD_CACHE_LOCK = Lock()


def _outer_curvature_prior_cache_identity(
    *,
    enabled: bool,
    prior: Any | None,
) -> dict[str, Any] | None:
    """Return the typed identity that makes Phase-III cache reuse safe.

    Candidate records at the same endpoint can differ when their old--old
    Hessian block came from a different transported outer prior.  The cache
    key must therefore include the prior fingerprint and its frame/support
    provenance rather than keying only on the state and parameter vector.
    """

    if not bool(enabled):
        return None
    if prior is None:
        return {
            "schema": "sr_outer_phase3_curvature_cache_identity_v1",
            "status": "exact_fallback_or_initial_anchor",
            "prior_fingerprint": None,
        }
    required = (
        "prior_fingerprint",
        "source_prior_id",
        "source_state_id",
        "source_frame_id",
        "source_support_id",
        "source_geometry_status",
    )
    values = {name: str(getattr(prior, name, "")) for name in required}
    if any(not value for value in values.values()):
        raise ValueError(
            "Outer curvature prior cache identity is missing typed provenance."
        )
    return {
        "schema": "sr_outer_phase3_curvature_cache_identity_v1",
        "status": "predicted_unvalidated",
        **values,
        "source_provenance_ids": [
            str(value)
            for value in getattr(prior, "source_provenance_ids", ())
        ],
    }


def _candidate_record_cache_mode() -> str:
    raw = str(os.environ.get(_CANDIDATE_RECORD_CACHE_ENV, "disk") or "disk").strip().lower()
    if raw in {"", "1", "true", "yes", "on", "always", "disk"}:
        return "disk"
    if raw in {"memory", "mem"}:
        return "memory"
    if raw in {"0", "false", "no", "off", "none", "disabled"}:
        return "off"
    raise ValueError(
        f"{_CANDIDATE_RECORD_CACHE_ENV} must be one of disk, memory, or off; got {raw!r}."
    )


def _candidate_record_cache_dir() -> Path:
    raw = str(os.environ.get(_CANDIDATE_RECORD_CACHE_DIR_ENV, "") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return REPO_ROOT / "raw_outputs" / "cache" / "static_adapt_candidate_records_v1"


def _candidate_record_memory_cache_max_entries() -> int:
    raw = str(
        os.environ.get(
            _CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV,
            _CANDIDATE_RECORD_MEMORY_CACHE_DEFAULT_MAX_ENTRIES,
        )
    ).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV} must be a positive integer."
        ) from exc
    if value < 1:
        raise ValueError(
            f"{_CANDIDATE_RECORD_MEMORY_CACHE_MAX_ENTRIES_ENV} must be >= 1."
        )
    return int(value)


def _candidate_record_memory_cache_store(cache_key: str, record: Any) -> None:
    max_entries = _candidate_record_memory_cache_max_entries()
    with _CANDIDATE_RECORD_CACHE_LOCK:
        key = str(cache_key)
        _CANDIDATE_RECORD_MEMORY_CACHE[key] = record
        _CANDIDATE_RECORD_MEMORY_CACHE.move_to_end(key)
        while len(_CANDIDATE_RECORD_MEMORY_CACHE) > int(max_entries):
            _CANDIDATE_RECORD_MEMORY_CACHE.popitem(last=False)


def _candidate_record_cache_path(cache_dir: Path, cache_key: str) -> Path:
    digest = str(cache_key)
    return Path(cache_dir) / digest[:2] / f"{digest}.pkl"


def _candidate_record_array_fingerprint(value: Any) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(tuple(int(x) for x in arr.shape)).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return {
        "schema": "ndarray_sha256_v1",
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "sha256": h.hexdigest(),
    }


def _candidate_record_cache_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _candidate_record_array_fingerprint(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, AnsatzTerm):
        return {
            "schema": "ansatz_term_digest_v1",
            "label": str(value.label),
            "polynomial_sha1": _polynomial_signature_digest(value.polynomial),
        }
    if isinstance(value, PauliPolynomial):
        return {
            "schema": "pauli_polynomial_digest_v1",
            "polynomial_sha1": _polynomial_signature_digest(value),
        }
    if isinstance(value, Mapping):
        return {
            str(k): _candidate_record_cache_jsonable(value[k])
            for k in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_candidate_record_cache_jsonable(x) for x in value]
    if isinstance(value, set):
        return sorted(_candidate_record_cache_jsonable(x) for x in value)
    if isinstance(value, CandidateFeatures):
        return {
            "schema": "candidate_features_fields_v1",
            "fields": _candidate_record_cache_jsonable(dict(value.__dict__)),
        }
    if hasattr(value, "__dict__"):
        return {
            "schema": "object_fields_v1",
            "class": value.__class__.__name__,
            "fields": _candidate_record_cache_jsonable(dict(value.__dict__)),
        }
    return str(value)


def _candidate_record_payload_digest(payload: Mapping[str, Any]) -> str:
    body = json.dumps(
        _candidate_record_cache_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _candidate_record_cache_get(
    *,
    cache_key: str,
    cache_dir: Path,
    mode: str,
) -> tuple[dict[str, Any] | None, str]:
    mode_key = str(mode).strip().lower()
    if mode_key == "off":
        return None, "off"
    with _CANDIDATE_RECORD_CACHE_LOCK:
        cached = _CANDIDATE_RECORD_MEMORY_CACHE.get(str(cache_key))
        if cached is not None:
            _CANDIDATE_RECORD_MEMORY_CACHE.move_to_end(str(cache_key))
    if cached is not None:
        try:
            return copy.deepcopy(cached), "memory"
        except Exception as exc:
            return None, f"memory_copy_error:{exc.__class__.__name__}"
    if mode_key != "disk":
        return None, "miss"
    path = _candidate_record_cache_path(Path(cache_dir), str(cache_key))
    if not path.is_file():
        return None, "miss"
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        return None, f"read_error:{exc.__class__.__name__}"
    if not isinstance(payload, Mapping):
        return None, "invalid_payload"
    if payload.get("schema") != _CANDIDATE_RECORD_CACHE_SCHEMA:
        return None, "schema_mismatch"
    if payload.get("code_version") != _CANDIDATE_RECORD_CACHE_CODE_VERSION:
        return None, "code_version_mismatch"
    if str(payload.get("cache_key", "")) != str(cache_key):
        return None, "key_mismatch"
    record = payload.get("record")
    if not isinstance(record, Mapping):
        return None, "record_missing"
    # ``payload`` was just unpickled and is reachable from nowhere else, so the
    # record handed to the caller is already private to them; only the
    # long-lived memory-cache entry needs a copy of its own.  This is one deep
    # copy per disk hit instead of two, with the same independence guarantee.
    record_copy = dict(record)
    try:
        record_copy_for_memory = copy.deepcopy(record_copy)
    except Exception as exc:
        return record_copy, f"disk_memory_copy_error:{exc.__class__.__name__}"
    _candidate_record_memory_cache_store(str(cache_key), record_copy_for_memory)
    return record_copy, "disk"


def _candidate_record_cache_put(
    *,
    cache_key: str,
    cache_dir: Path,
    mode: str,
    record: Mapping[str, Any],
) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key == "off":
        return "off"
    try:
        record_copy = copy.deepcopy(dict(record))
    except Exception as exc:
        return f"copy_error:{exc.__class__.__name__}"
    # ``record_copy`` is already independent of the caller's ``record``, and the
    # only other use below is ``pickle.dump``, which does not mutate what it
    # serializes.  One deep copy therefore serves both the memory entry and the
    # on-disk payload.
    _candidate_record_memory_cache_store(str(cache_key), record_copy)
    if mode_key != "disk":
        return "memory"
    path = _candidate_record_cache_path(Path(cache_dir), str(cache_key))
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = {
        "schema": _CANDIDATE_RECORD_CACHE_SCHEMA,
        "code_version": _CANDIDATE_RECORD_CACHE_CODE_VERSION,
        "cache_key": str(cache_key),
        "stored_utc": datetime.now(timezone.utc).isoformat(),
        "record": record_copy,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return f"write_error:{exc.__class__.__name__}"
    return "disk"
