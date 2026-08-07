#!/usr/bin/env python3
"""Runtime contract shared by one-row local threshold-sweep packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent


class PackageContractError(RuntimeError):
    """Fail-closed package or execution-contract violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    observed = canonical_sha256(unsigned)
    if value.get("sha256") != observed:
        raise PackageContractError(f"{label} self-digest drifted.")
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PackageContractError(f"{label} must be a nonempty path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PackageContractError(f"{label} is unsafe: {value!r}.")
    return Path(*pure.parts)


CONTROL_PATH = PACKAGE_DIR / "package_control.json"
CONTROL = load_json(CONTROL_PATH, label="package control")
verify_self_digest(CONTROL, label="package control")

PACKAGE_ID = str(CONTROL["package_id"])
PACKAGE_STATUS = str(CONTROL["package_status"])
CAMPAIGN_ID = str(CONTROL["campaign_id"])
ALGORITHM_ID = str(CONTROL["algorithm_id"])
ROUTE_CONTRACT_SHA256 = str(CONTROL["route_contract_sha256"])
PARENT_ROUTE_CONTRACT_SHA256 = str(CONTROL["parent_route_contract_sha256"])
ROUTE_PROFILE = str(CONTROL["route_profile"])
PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD = float(CONTROL["threshold"])
PLATEAU_COMPARISON = str(CONTROL["plateau_comparison"])
PLATEAU_TRIGGER = str(CONTROL["plateau_trigger"])
PLATEAU_CALIBRATION = str(CONTROL["plateau_calibration"])
ACTIVE_GRADIENT_POLICY = str(CONTROL["active_gradient_policy"])
RESOURCE_WEIGHTING_SCOPE = str(CONTROL["resource_weighting_scope"])
CANDIDATE_REPRESENTATION = str(CONTROL["candidate_representation"])
TARGET_HORIZON = int(CONTROL["target_horizon"])
EXECUTION_TARGET = str(CONTROL["execution_target"])
PACKAGE_MANIFEST_SCHEMA = str(CONTROL["package_manifest_schema"])
JOB_SCHEMA = str(CONTROL["job_schema"])
AUTHORIZATION_SCHEMA = str(CONTROL["authorization_schema"])


def expected_execution_ids() -> tuple[str, ...]:
    values = CONTROL.get("execution_ids")
    if not isinstance(values, list) or len(values) != 1:
        raise PackageContractError("Package control must declare one execution id.")
    return tuple(str(value) for value in values)


__all__ = [name for name in globals() if name.isupper()] + [
    "PackageContractError",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "expected_execution_ids",
    "load_json",
    "safe_relative_path",
    "sha256_file",
    "verify_self_digest",
]
