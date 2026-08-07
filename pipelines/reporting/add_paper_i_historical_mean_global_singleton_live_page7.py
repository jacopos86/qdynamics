#!/usr/bin/env python3
"""Replace page 7 with guarded live nph=7 RA checkpoint projections.

This companion deliberately leaves the completed-result page-7 generator
unchanged.  It copies the three completed nph=3 cells and all six authenticated
Append-ADAPT overlays byte-for-byte from that generator's current adapter, then
adds three nonterminal nph=7 trajectories from compact, self-digested live
projections.  Live rows expose only the closed prefix ``S_alg``; Qiskit fields
remain explicitly pending until a completed attempt can be validated.

The live page retains the completed generator's page identity so its strict
three-to-six completion update can replace this diagnostic later.  PDF assembly
always copies pages 1--6 and verifies their content hashes before replacement.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    add_paper_i_historical_mean_global_singleton_full6_page as completed,
)


LIVE_PROJECTION_SCHEMA = (
    "paper_i_historical_mean_global_singleton_page7_live_projection_v1"
)
LIVE_ADAPTER_SCHEMA = (
    "paper_i_historical_mean_global_singleton_vs_append_live_full6_adapter_v1"
)
LIVE_PROJECTION_STATUS = "passed_authenticated_live_partial"
LIVE_ADAPTER_STATUS = "passed_live_partial"
LIVE_CLASSIFICATION = "supplemental_live_diagnostic_not_adopted_evidence"
LIVE_QISKIT_STATUS = "pending_live_prefix_not_compiled"
LIVE_MARKER_POLICY = "terminal_observed_live_prefix"
LIVE_LIMITATION = (
    "Page 7 is a supplemental live diagnostic. The nph=7 RA curves stop at "
    "their authenticated nonterminal checkpoint prefixes; only closed "
    "occurrence-based S_alg is shown for those prefixes, and all Qiskit costs "
    "remain explicitly pending. No live cell is adopted Paper-I evidence."
)
PLOT_FLOOR = 1.0e-16
QISKIT_PENDING = {field: None for field in completed.QISKIT_FIELDS}
SNAPSHOT_VALIDATION = {
    "archive_file_sha256_verified": True,
    "archive_size_verified": True,
    "validation_receipt_archive_binding_verified": True,
    "archive_member_set_exact": True,
    "checkpoint_file_sha256_verified": True,
    "estimator_ledger_checkpoint_file_sha256_verified": True,
    "verified_resume_sidecar_file_sha256_verified": True,
    "checkpoint_triplet_pointer_closed": True,
    "accepted_trajectory_projected_from_checkpoint": True,
    "algorithmic_work_closed_against_ledger_checkpoint": True,
}
SNAPSHOT_RECEIPT_SCHEMA = "paper_i_live_checkpoint_snapshot_validation_v1"
SNAPSHOT_ARCHIVE_TIMESTAMP_RE = re.compile(r"__(\d{8}T\d{6}Z)(?:\.tar\.gz)?$")
STREAM_COMPONENTS = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
SCALAR_EVENTS = frozenset(("string", "number", "boolean", "null"))
STREAM_CHUNK_SIZE = 8 * 1024 * 1024
ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1"
)
SNAPSHOT_REGIME_TOKENS = {
    "weak_strong": "weak_strong",
    "intermediate_strong": "intermediate_strong",
    "strong_strong_u8": "strong_strong_u8",
}
IMPLEMENTATION_REPAIR = {
    "repair_id": "accepted_round_current_checkpoint_receipt_loader_fix_v2",
    "path": "pipelines/static_adapt/sr_snake/_resume.py",
    "before_sha256": (
        "6d3753f22071cae21eb5eb006e634655be0fb4a9ec60054d61dfef2a3625e37f"
    ),
    "after_sha256": (
        "173fcbc219453b4a90d604afdfe117718a34318bc621a11ab178a63304e72032"
    ),
    "scientific_protocol_changed": False,
    "scientific_settings_changed": [],
}

CAMPAIGN_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "20260802_v3_resume128gb_loaderfix_v2_chtc"
)
ACTIVATION_DIR = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "20260802_v3_resume128gb_loaderfix_v2_chtc_activation_ordinary_v1"
)
SUBMISSION_RECEIPT = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "20260802_v3_resume128gb_loaderfix_v2_chtc_runtime/submission_receipt.json"
)
PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "20260802_v3_resume128gb_loaderfix_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "resume128gb_loaderfix_v2"
)
PACKAGE_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "package_manifest_v2"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "activation_manifest_v2"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_job_v2"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "execution_authorization_v2"
)
SUBMISSION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "submission_receipt_v2"
)
PACKAGE_CANONICAL_SHA256 = (
    "84d8f7bdcc79e986c8bbd22af8f3c1c5ed2d5c1b95aeb1e84affb5c3ae87e1a1"
)
PACKAGE_FILE_SHA256 = (
    "0b81923cbc691fb18ca58bb78da73d6ce9ba501e6717ee315b2a2fb8744b293d"
)
ACTIVATION_CANONICAL_SHA256 = (
    "36bd7278293f4a32f010e1a6b733a35159ac8785a420113a3980276d4ee935c5"
)
ACTIVATION_FILE_SHA256 = (
    "9d4a32527557cf6abe328113aa5ce0e608cfa3e5a6344bc56f17ee0aef720edc"
)
SUBMISSION_CANONICAL_SHA256 = (
    "0506a91bcd187253c6e96f31481caf78a927e38322f66a5a1552f44751b0b2e1"
)
SUBMISSION_FILE_SHA256 = (
    "59a36a9ae29fcac9d71ad72b3aab6563836ab850dc89404bdd900f6a51b159eb"
)
CLUSTER_ID = 9_401_106
RESUME_ROUNDS = {
    "weak_strong": 35,
    "intermediate_strong": 31,
    "strong_strong_u8": 17,
}
PROC_IDS = {
    "weak_strong": 0,
    "intermediate_strong": 1,
    "strong_strong_u8": 2,
}
HEX64 = re.compile(r"[0-9a-f]{64}")


class LivePage7InputError(ValueError):
    """Raised when a live projection or replacement is not monotone and closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "sha256" in result:
        raise LivePage7InputError("self-digest input already contains sha256")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LivePage7InputError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise LivePage7InputError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LivePage7InputError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise LivePage7InputError(f"{label} must be an array")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LivePage7InputError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LivePage7InputError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise LivePage7InputError(f"{label} is outside its finite range")
    return result


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = value.get("sha256")
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise LivePage7InputError(f"{label} self-digest drifted")
    return str(observed)


def _bound_json(
    path: Path,
    *,
    label: str,
    canonical_sha256: str | None = None,
    file_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise LivePage7InputError(f"{label} is unavailable or unsafe")
    value = _load_json(path, label=label)
    canonical = _verify_self_digest(value, label=label)
    binding = file_binding(path)
    if canonical_sha256 is not None and canonical != canonical_sha256:
        raise LivePage7InputError(f"{label} canonical binding drifted")
    if file_sha256 is not None and binding["sha256"] != file_sha256:
        raise LivePage7InputError(f"{label} file binding drifted")
    binding["canonical_sha256"] = canonical
    return value, binding


def _expected_execution_id(regime: str) -> str:
    if regime not in completed.NPH7_REGIMES:
        raise LivePage7InputError(f"{regime}: live page supports nph=7 only")
    return (
        "historical_mean_global_singleton_v2_nph7_r50__"
        f"{regime}__nph7__ra_global_singleton_plateau__resume_from_d"
        f"{RESUME_ROUNDS[regime]}_to_r50_loaderfix_v2"
    )


def _snapshot_binding(value: Any, *, label: str) -> dict[str, Any]:
    row = dict(_mapping(value, label=label))
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise LivePage7InputError(f"{label} binding fields drifted")
    path_text = row.get("path")
    if not isinstance(path_text, str) or not path_text or "\x00" in path_text:
        raise LivePage7InputError(f"{label} path is invalid")
    pure = PurePosixPath(path_text)
    if "." in pure.parts or ".." in pure.parts or pure.as_posix() != path_text:
        raise LivePage7InputError(f"{label} path is not normalized")
    digest = row.get("sha256")
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise LivePage7InputError(f"{label} SHA-256 is invalid")
    size = _integer(row.get("size_bytes"), label=f"{label} size", minimum=1)
    return {"path": path_text, "sha256": digest, "size_bytes": size}


def _validate_observed_utc(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LivePage7InputError("observed_utc must be UTC text ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise LivePage7InputError("observed_utc is malformed") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise LivePage7InputError("observed_utc is not UTC")
    return value


def _validate_work(value: Any, *, label: str) -> dict[str, Any]:
    work = _mapping(value, label=label)
    components = _mapping(work.get("components"), label=f"{label} components")
    expected_fields = {"N_H_outer", "N_H_refit", "N_grad", "N_metric"}
    if set(components) != expected_fields:
        raise LivePage7InputError(f"{label} component fields drifted")
    normalized = {
        field: _integer(components[field], label=f"{label} {field}")
        for field in sorted(expected_fields)
    }
    s_alg = _integer(work.get("S_alg"), label=f"{label} S_alg")
    if sum(normalized.values()) != s_alg:
        raise LivePage7InputError(f"{label} S_alg does not close")
    return {"components": normalized, "S_alg": s_alg}


class _DigestingReader:
    """Hash and count a tar member while a streaming JSON parser consumes it."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        payload = self._stream.read(size)
        if payload:
            self._digest.update(payload)
            self.size_bytes += len(payload)
        return payload

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _verified_snapshot_member(
    value: Any, *, name: str, label: str
) -> dict[str, Any]:
    binding = dict(_mapping(value, label=label))
    if set(binding) != {"sha256", "size_bytes"}:
        raise LivePage7InputError(f"{label} binding fields drifted")
    digest = binding.get("sha256")
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise LivePage7InputError(f"{label} SHA-256 is invalid")
    size = _integer(binding.get("size_bytes"), label=f"{label} size", minimum=1)
    return {"path": name, "sha256": digest, "size_bytes": size}


def _snapshot_observed_utc(archive_path: Path) -> str:
    name = archive_path.name
    if name.endswith(".tar.gz"):
        name = name[: -len(".tar.gz")]
    match = SNAPSHOT_ARCHIVE_TIMESTAMP_RE.search(name)
    if match is None:
        raise LivePage7InputError(
            "snapshot archive name must end in __YYYYMMDDTHHMMSSZ.tar.gz"
        )
    parsed = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_snapshot_receipt(
    *, archive_path: Path, validation_path: Path, regime: str
) -> dict[str, Any]:
    if regime not in completed.NPH7_REGIMES:
        raise LivePage7InputError("snapshot receipt regime is unsupported")
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or not validation_path.is_file()
        or validation_path.is_symlink()
    ):
        raise LivePage7InputError("snapshot archive or validation receipt is unsafe")
    archive_stem = archive_path.name
    if not archive_stem.endswith(".tar.gz"):
        raise LivePage7InputError("snapshot archive suffix drifted")
    archive_stem = archive_stem[: -len(".tar.gz")]
    expected_prefix = (
        f"{CLUSTER_ID}.{PROC_IDS[regime]}__{SNAPSHOT_REGIME_TOKENS[regime]}__"
    )
    if (
        not archive_stem.startswith(expected_prefix)
        or validation_path.name != f"{archive_stem}.validation.json"
        or validation_path.parent.resolve() != archive_path.parent.resolve()
    ):
        raise LivePage7InputError(
            f"{regime}: snapshot cluster/proc/regime filename binding drifted"
        )
    receipt = _load_json(validation_path, label="snapshot validation receipt")
    archive_text = receipt.get("archive")
    if not isinstance(archive_text, str) or not archive_text:
        raise LivePage7InputError("snapshot validation archive path is invalid")
    receipt_archive = Path(archive_text).expanduser()
    if not receipt_archive.is_absolute():
        receipt_archive = REPO_ROOT / receipt_archive
    archive_binding = file_binding(archive_path)
    if (
        receipt.get("schema") != SNAPSHOT_RECEIPT_SCHEMA
        or receipt.get("validation") != "passed"
        or receipt_archive.resolve() != archive_path.resolve()
        or receipt.get("archive_sha256") != archive_binding["sha256"]
        or receipt.get("archive_size_bytes") != archive_binding["size_bytes"]
    ):
        raise LivePage7InputError("snapshot validation/archive binding drifted")
    checkpoint_depth = _integer(
        receipt.get("checkpoint_depth"), label="snapshot checkpoint depth", minimum=1
    )
    raw_pointers = _mapping(receipt.get("pointers"), label="snapshot pointers")
    if set(raw_pointers) != {"ledger", "resume"}:
        raise LivePage7InputError("snapshot pointer roles drifted")
    pointers: dict[str, dict[str, str]] = {}
    for role in ("ledger", "resume"):
        row = dict(_mapping(raw_pointers[role], label=f"snapshot {role} pointer"))
        if set(row) != {"path", "sha256"}:
            raise LivePage7InputError(f"snapshot {role} pointer fields drifted")
        name = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or "." in PurePosixPath(name).parts
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name
            or not isinstance(digest, str)
            or not HEX64.fullmatch(digest)
        ):
            raise LivePage7InputError(f"snapshot {role} pointer is invalid")
        pointers[role] = {"path": name, "sha256": digest}
    raw_members = _mapping(receipt.get("members"), label="snapshot members")
    expected_names = {
        "checkpoint.json",
        pointers["ledger"]["path"],
        pointers["resume"]["path"],
    }
    if set(raw_members) != expected_names:
        raise LivePage7InputError("snapshot member set does not close to pointers")
    members = {
        name: _verified_snapshot_member(
            raw_members[name], name=name, label=f"snapshot member {name}"
        )
        for name in expected_names
    }
    for role in ("ledger", "resume"):
        name = pointers[role]["path"]
        if (
            members[name]["sha256"] != pointers[role]["sha256"]
            or not members[name]["sha256"].startswith(name.rsplit(".", 2)[-2])
        ):
            raise LivePage7InputError(f"snapshot {role} pointer hash drifted")
    return {
        "receipt": receipt,
        "receipt_binding": file_binding(validation_path),
        "archive_binding": archive_binding,
        "checkpoint_depth": checkpoint_depth,
        "pointers": pointers,
        "members": members,
        "observed_utc": _snapshot_observed_utc(archive_path),
    }


def _finish_member_binding(
    reader: _DigestingReader, *, expected: Mapping[str, Any], label: str
) -> None:
    while reader.read(STREAM_CHUNK_SIZE):
        pass
    if (
        reader.sha256 != expected.get("sha256")
        or reader.size_bytes != expected.get("size_bytes")
    ):
        raise LivePage7InputError(f"{label} member byte binding drifted")


def _stream_checkpoint_projection(
    stream: Any, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import ijson
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise LivePage7InputError("live projection streaming requires ijson") from exc

    reader = _DigestingReader(stream)
    rows: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    scalars: dict[str, Any] = {}
    scalar_prefixes = {
        "adapt_vqe.history_count",
        "adapt_vqe.active_ansatz_depth",
        "adapt_vqe.ansatz_depth",
        "adapt_vqe.terminal_active_prefix_checkpoint.active_ansatz_depth",
        "adapt_vqe.estimator_call_accounting.complete",
        "adapt_vqe.estimator_call_accounting.S_alg",
        "adapt_vqe.S_alg",
        "adapt_vqe.route_profile",
        "adapt_vqe.sr_route_profile_contract_sha256",
        "adapt_vqe.accepted_state_resume.schema",
        "adapt_vqe.accepted_state_resume.source_sha256",
        "adapt_vqe.accepted_state_resume.source_controller_round",
        "adapt_vqe.accepted_state_resume.strict_numerical_replay_passed",
        "adapt_vqe.accepted_state_resume.route_and_problem_binding_passed",
        "adapt_vqe.estimator_call_ledger_checkpoint.path",
        "adapt_vqe.estimator_call_ledger_checkpoint.sha256",
        "adapt_vqe.estimator_call_ledger_checkpoint.S_alg",
        "adapt_vqe.estimator_call_ledger_checkpoint.raw_occurrence_count",
        "adapt_vqe.estimator_call_ledger_checkpoint.checkpoint_depth",
        "adapt_vqe.estimator_call_ledger_checkpoint.status",
        "adapt_vqe.estimator_call_ledger_checkpoint.current_round_finalized",
        "adapt_vqe.verified_singleton_resume_sidecar.path",
        "adapt_vqe.verified_singleton_resume_sidecar.sha256",
        "adapt_vqe.verified_singleton_resume_sidecar.status",
        "adapt_vqe.verified_singleton_resume_sidecar.source_projection_sha256",
    }
    for field in STREAM_COMPONENTS:
        scalar_prefixes.add(f"adapt_vqe.estimator_call_accounting.components.{field}")
        scalar_prefixes.add(f"adapt_vqe.S_alg_components.{field}")
    try:
        for prefix, event, value in ijson.parse(reader, use_float=True):
            if prefix == "adapt_vqe.history.item" and event == "start_map":
                if current is not None:
                    raise LivePage7InputError("nested checkpoint history row")
                current = {}
            elif (
                current is not None
                and event in SCALAR_EVENTS
                and prefix
                in {
                    "adapt_vqe.history.item.energy_before_opt",
                    "adapt_vqe.history.item.energy_after_opt",
                }
            ):
                current[prefix.rsplit(".", 1)[-1]] = _finite(
                    value, label=prefix
                )
            elif prefix == "adapt_vqe.history.item" and event == "end_map":
                if current is None or set(current) != {
                    "energy_before_opt",
                    "energy_after_opt",
                }:
                    raise LivePage7InputError(
                        "checkpoint history row lacks accepted energies"
                    )
                rows.append(current)
                current = None
            elif event in SCALAR_EVENTS and prefix in scalar_prefixes:
                if prefix in scalars and scalars[prefix] != value:
                    raise LivePage7InputError(f"duplicate checkpoint scalar {prefix}")
                scalars[prefix] = value
    except (ValueError, TypeError) as exc:
        if isinstance(exc, LivePage7InputError):
            raise
        raise LivePage7InputError("checkpoint streaming JSON parse failed") from exc
    _finish_member_binding(reader, expected=expected, label="checkpoint")

    depth = _integer(
        scalars.get("adapt_vqe.history_count"), label="checkpoint history count"
    )
    if depth != len(rows):
        raise LivePage7InputError("checkpoint history count/rows drifted")
    depth_values = [
        scalars[prefix]
        for prefix in (
            "adapt_vqe.active_ansatz_depth",
            "adapt_vqe.ansatz_depth",
            "adapt_vqe.terminal_active_prefix_checkpoint.active_ansatz_depth",
        )
        if prefix in scalars
    ]
    if not depth_values:
        raise LivePage7InputError("checkpoint active ansatz depth is missing")
    active_depths = {
        _integer(value, label="checkpoint active ansatz depth")
        for value in depth_values
    }
    if len(active_depths) != 1:
        raise LivePage7InputError("checkpoint active ansatz depth fields disagree")
    energies: list[float] = []
    previous: float | None = None
    for index, row in enumerate(rows, start=1):
        before = row["energy_before_opt"]
        after = row["energy_after_opt"]
        if (
            (previous is not None and not math.isclose(
                before, previous, rel_tol=0.0, abs_tol=1.0e-9
            ))
            or after > before + 1.0e-9
        ):
            raise LivePage7InputError(
                f"checkpoint accepted-energy chain drifted at round {index}"
            )
        if previous is None:
            energies.append(before)
        energies.append(after)
        previous = after

    component_sets: list[dict[str, int]] = []
    for stem in (
        "adapt_vqe.estimator_call_accounting.components",
        "adapt_vqe.S_alg_components",
    ):
        values = {
            field: _integer(
                scalars.get(f"{stem}.{field}"), label=f"checkpoint {stem} {field}"
            )
            for field in STREAM_COMPONENTS
            if f"{stem}.{field}" in scalars
        }
        if values:
            if set(values) != set(STREAM_COMPONENTS):
                raise LivePage7InputError("checkpoint component set is partial")
            component_sets.append(values)
    if not component_sets or any(row != component_sets[0] for row in component_sets[1:]):
        raise LivePage7InputError("checkpoint accounting components disagree")
    s_alg_values = {
        _integer(scalars[prefix], label=f"checkpoint {prefix}")
        for prefix in (
            "adapt_vqe.estimator_call_accounting.S_alg",
            "adapt_vqe.S_alg",
        )
        if prefix in scalars
    }
    if (
        scalars.get("adapt_vqe.estimator_call_accounting.complete") is not True
        or len(s_alg_values) != 1
        or sum(component_sets[0].values()) != next(iter(s_alg_values))
    ):
        raise LivePage7InputError("checkpoint accounting does not close")
    return {
        "history_count": depth,
        "active_ansatz_depth": next(iter(active_depths)),
        "energies": energies,
        "components": component_sets[0],
        "S_alg": next(iter(s_alg_values)),
        "execution_identity": {
            "route_profile": scalars.get("adapt_vqe.route_profile"),
            "route_contract_sha256": scalars.get(
                "adapt_vqe.sr_route_profile_contract_sha256"
            ),
            "accepted_state_resume_schema": scalars.get(
                "adapt_vqe.accepted_state_resume.schema"
            ),
            "source_checkpoint_sha256": scalars.get(
                "adapt_vqe.accepted_state_resume.source_sha256"
            ),
            "source_controller_round": scalars.get(
                "adapt_vqe.accepted_state_resume.source_controller_round"
            ),
            "strict_numerical_replay_passed": scalars.get(
                "adapt_vqe.accepted_state_resume.strict_numerical_replay_passed"
            ),
            "route_and_problem_binding_passed": scalars.get(
                "adapt_vqe.accepted_state_resume.route_and_problem_binding_passed"
            ),
        },
        "ledger_pointer": {
            "path": scalars.get(
                "adapt_vqe.estimator_call_ledger_checkpoint.path"
            ),
            "sha256": scalars.get(
                "adapt_vqe.estimator_call_ledger_checkpoint.sha256"
            ),
            "S_alg": scalars.get(
                "adapt_vqe.estimator_call_ledger_checkpoint.S_alg"
            ),
            "raw_occurrence_count": scalars.get(
                "adapt_vqe.estimator_call_ledger_checkpoint.raw_occurrence_count"
            ),
            "checkpoint_depth": scalars.get(
                "adapt_vqe.estimator_call_ledger_checkpoint.checkpoint_depth"
            ),
            "status": scalars.get(
                "adapt_vqe.estimator_call_ledger_checkpoint.status"
            ),
            "current_round_finalized": scalars.get(
                "adapt_vqe.estimator_call_ledger_checkpoint.current_round_finalized"
            ),
        },
        "resume_pointer": {
            "path": scalars.get("adapt_vqe.verified_singleton_resume_sidecar.path"),
            "sha256": scalars.get(
                "adapt_vqe.verified_singleton_resume_sidecar.sha256"
            ),
            "status": scalars.get(
                "adapt_vqe.verified_singleton_resume_sidecar.status"
            ),
            "source_projection_sha256": scalars.get(
                "adapt_vqe.verified_singleton_resume_sidecar.source_projection_sha256"
            ),
        },
    }


def _stream_ledger_projection(
    stream: Any, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import ijson
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise LivePage7InputError("live projection streaming requires ijson") from exc

    reader = _DigestingReader(stream)
    scalars: dict[str, Any] = {}
    current_occurrence: dict[str, Any] | None = None
    counted = {field: 0 for field in STREAM_COMPONENTS}
    expected_sequence = 1
    scalar_prefixes = {
        "schema",
        "checkpoint.depth",
        "checkpoint.current_round_finalized",
        "S_alg",
        "raw_occurrence_count",
        "ledger.schema",
        "ledger.occurrence_summary.total_call_occurrences",
    }
    for field in STREAM_COMPONENTS:
        scalar_prefixes.add(f"ledger.occurrence_summary.components.{field}")
        scalar_prefixes.add(
            f"ledger.occurrence_summary.component_occurrence_counts.{field}"
        )
        scalar_prefixes.add(f"ledger.occurrence_summary.{field}")
    try:
        for prefix, event, value in ijson.parse(reader, use_float=True):
            if prefix == "ledger.occurrences.item" and event == "start_map":
                if current_occurrence is not None:
                    raise LivePage7InputError("nested ledger occurrence")
                current_occurrence = {}
            elif (
                current_occurrence is not None
                and event in SCALAR_EVENTS
                and prefix
                in {
                    "ledger.occurrences.item.sequence",
                    "ledger.occurrences.item.component",
                }
            ):
                current_occurrence[prefix.rsplit(".", 1)[-1]] = value
            elif prefix == "ledger.occurrences.item" and event == "end_map":
                if current_occurrence is None:
                    raise LivePage7InputError("ledger occurrence state drifted")
                sequence = _integer(
                    current_occurrence.get("sequence"),
                    label="ledger occurrence sequence",
                    minimum=1,
                )
                component = current_occurrence.get("component")
                if sequence != expected_sequence or component not in counted:
                    raise LivePage7InputError("ledger occurrence stream is not closed")
                counted[str(component)] += 1
                expected_sequence += 1
                current_occurrence = None
            elif event in SCALAR_EVENTS and prefix in scalar_prefixes:
                if prefix in scalars and scalars[prefix] != value:
                    raise LivePage7InputError(f"duplicate ledger scalar {prefix}")
                scalars[prefix] = value
    except (ValueError, TypeError) as exc:
        if isinstance(exc, LivePage7InputError):
            raise
        raise LivePage7InputError("ledger streaming JSON parse failed") from exc
    _finish_member_binding(reader, expected=expected, label="ledger checkpoint")

    total = expected_sequence - 1
    serialized_components: list[dict[str, int]] = []
    for stem in (
        "ledger.occurrence_summary.components",
        "ledger.occurrence_summary.component_occurrence_counts",
        "ledger.occurrence_summary",
    ):
        values = {
            field: _integer(
                scalars.get(f"{stem}.{field}"), label=f"ledger {stem} {field}"
            )
            for field in STREAM_COMPONENTS
            if f"{stem}.{field}" in scalars
        }
        if values:
            if set(values) != set(STREAM_COMPONENTS):
                raise LivePage7InputError("ledger component summary is partial")
            serialized_components.append(values)
    if not serialized_components or any(
        row != counted for row in serialized_components
    ):
        raise LivePage7InputError("ledger component summaries do not close")
    total_values = {
        _integer(scalars[prefix], label=f"ledger {prefix}")
        for prefix in (
            "S_alg",
            "raw_occurrence_count",
            "ledger.occurrence_summary.total_call_occurrences",
        )
        if prefix in scalars
    }
    if (
        scalars.get("schema")
        != "paper_i_estimator_call_ledger_checkpoint_sidecar_v2"
        or scalars.get("checkpoint.current_round_finalized") is not True
        or len(total_values) != 1
        or next(iter(total_values)) != total
        or sum(counted.values()) != total
    ):
        raise LivePage7InputError("ledger checkpoint S_alg does not close")
    return {
        "checkpoint_depth": _integer(
            scalars.get("checkpoint.depth"), label="ledger checkpoint depth"
        ),
        "components": counted,
        "S_alg": total,
    }


def _read_resume_projection(
    stream: Any, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    reader = _DigestingReader(stream)
    chunks: list[bytes] = []
    while payload := reader.read(STREAM_CHUNK_SIZE):
        chunks.append(payload)
        if reader.size_bytes > 4 * 1024 * 1024:
            raise LivePage7InputError("verified resume sidecar is unexpectedly large")
    _finish_member_binding(reader, expected=expected, label="verified resume")
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LivePage7InputError("verified resume sidecar is invalid JSON") from exc
    resume = _mapping(value, label="verified resume sidecar")
    controller = _mapping(
        resume.get("controller_state"), label="verified resume controller state"
    )
    selection = _mapping(
        resume.get("selection_state"), label="verified resume selection state"
    )
    source_sha256 = resume.get("source_result_sha256")
    if (
        resume.get("schema") != "static_adapt_signed_active_prefix_resume_sidecar_v2"
        or not isinstance(source_sha256, str)
        or not HEX64.fullmatch(source_sha256)
    ):
        raise LivePage7InputError("verified resume identity drifted")
    controller_round = _integer(
        controller.get("controller_round"), label="resume controller round"
    )
    if _integer(
        selection.get("controller_round"), label="resume selection round"
    ) != controller_round:
        raise LivePage7InputError("verified resume controller rounds disagree")
    return {
        "controller_round": controller_round,
        "source_projection_sha256": source_sha256,
    }


def _stream_snapshot_archive(
    *, archive_path: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    members = _mapping(receipt.get("members"), label="validated snapshot members")
    pointers = _mapping(receipt.get("pointers"), label="validated snapshot pointers")
    ledger_name = str(_mapping(pointers["ledger"], label="ledger pointer")["path"])
    resume_name = str(_mapping(pointers["resume"], label="resume pointer")["path"])
    expected_names = {"checkpoint.json", ledger_name, resume_name}
    seen: set[str] = set()
    checkpoint: dict[str, Any] | None = None
    ledger: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                if (
                    not member.isfile()
                    or member.name not in expected_names
                    or member.name in seen
                    or member.name.startswith("/")
                    or ".." in PurePosixPath(member.name).parts
                ):
                    raise LivePage7InputError(
                        f"snapshot archive member is unexpected: {member.name}"
                    )
                expected = _mapping(
                    members[member.name], label=f"receipt member {member.name}"
                )
                if member.size != expected.get("size_bytes"):
                    raise LivePage7InputError(
                        f"snapshot member size header drifted: {member.name}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise LivePage7InputError(
                        f"snapshot member cannot be streamed: {member.name}"
                    )
                if member.name == "checkpoint.json":
                    checkpoint = _stream_checkpoint_projection(
                        stream, expected=expected
                    )
                elif member.name == ledger_name:
                    ledger = _stream_ledger_projection(stream, expected=expected)
                else:
                    resume = _read_resume_projection(stream, expected=expected)
                seen.add(member.name)
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise LivePage7InputError("snapshot archive streaming failed") from exc
    if seen != expected_names or checkpoint is None or ledger is None or resume is None:
        raise LivePage7InputError("snapshot archive member closure failed")

    ledger_pointer = _mapping(
        checkpoint.get("ledger_pointer"), label="checkpoint ledger pointer"
    )
    resume_pointer = _mapping(
        checkpoint.get("resume_pointer"), label="checkpoint resume pointer"
    )
    receipt_ledger = _mapping(pointers["ledger"], label="receipt ledger pointer")
    receipt_resume = _mapping(pointers["resume"], label="receipt resume pointer")
    depth = int(checkpoint["history_count"])
    if (
        ledger_pointer.get("path") != ledger_name
        or ledger_pointer.get("sha256") != receipt_ledger.get("sha256")
        or ledger_pointer.get("status") != "complete"
        or ledger_pointer.get("current_round_finalized") is not True
        or ledger_pointer.get("checkpoint_depth") != depth
        or ledger_pointer.get("S_alg") != ledger["S_alg"]
        or ledger_pointer.get("raw_occurrence_count") != ledger["S_alg"]
        or ledger["checkpoint_depth"] != depth
        or ledger["components"] != checkpoint["components"]
        or ledger["S_alg"] != checkpoint["S_alg"]
        or resume_pointer.get("path") != resume_name
        or resume_pointer.get("sha256") != receipt_resume.get("sha256")
        or resume_pointer.get("status") != "complete"
        or resume_pointer.get("source_projection_sha256")
        != resume["source_projection_sha256"]
        or resume["controller_round"] != depth
    ):
        raise LivePage7InputError("checkpoint triplet pointer/accounting closure failed")
    return {
        "history_count": depth,
        "active_ansatz_depth": checkpoint["active_ansatz_depth"],
        "energies": checkpoint["energies"],
        "execution_identity": copy.deepcopy(checkpoint["execution_identity"]),
        "algorithmic_work": {
            "components": copy.deepcopy(ledger["components"]),
            "S_alg": ledger["S_alg"],
        },
    }


def _validate_loaderfix_authority(
    regime: str, *, execution_id: str, protocol_sha256: str
) -> dict[str, Any]:
    package, package_binding = _bound_json(
        PACKAGE_DIR / "package_manifest.json",
        label="loaderfix package manifest",
        canonical_sha256=PACKAGE_CANONICAL_SHA256,
        file_sha256=PACKAGE_FILE_SHA256,
    )
    activation, activation_binding = _bound_json(
        ACTIVATION_DIR / "activation_manifest.json",
        label="loaderfix activation manifest",
        canonical_sha256=ACTIVATION_CANONICAL_SHA256,
        file_sha256=ACTIVATION_FILE_SHA256,
    )
    submission, submission_binding = _bound_json(
        SUBMISSION_RECEIPT,
        label="loaderfix submission receipt",
        canonical_sha256=SUBMISSION_CANONICAL_SHA256,
        file_sha256=SUBMISSION_FILE_SHA256,
    )
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("status") != "passed_inert_three_authenticated_resumes"
        or package.get("package_id") != PACKAGE_ID
        or package.get("campaign_id") != CAMPAIGN_ID
        or package.get("row_count") != 3
        or package.get("implementation_repair") != IMPLEMENTATION_REPAIR
        or package.get("scientific_protocol_changed") is not False
        or package.get("scientific_settings_changed") != []
        or package.get("source_held_jobs_preserved") is not True
        or set(package.get("execution_ids", ()))
        != {_expected_execution_id(item) for item in completed.NPH7_REGIMES}
    ):
        raise LivePage7InputError("loaderfix package authority drifted")
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("package_id") != PACKAGE_ID
        or activation.get("campaign_id") != CAMPAIGN_ID
        or activation.get("implementation_repair") != IMPLEMENTATION_REPAIR
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
        or activation.get("paper_evidence_adopted") is not False
        or activation.get("source_held_jobs_preserved") is not True
        or activation.get("scientific_protocol_changed") is not False
        or activation.get("scientific_settings_changed") != []
    ):
        raise LivePage7InputError("loaderfix activation authority drifted")
    if (
        submission.get("schema") != SUBMISSION_SCHEMA
        or submission.get("status")
        != "passed_submitted_three_authenticated_resumes"
        or submission.get("cluster_id") != CLUSTER_ID
        or submission.get("direct_job_count") != 3
        or submission.get("implementation_repair") != IMPLEMENTATION_REPAIR
        or submission.get("scientific_protocol_changed") is not False
        or submission.get("scientific_settings_changed") != []
    ):
        raise LivePage7InputError("loaderfix submission authority drifted")

    job_path = PACKAGE_DIR / "jobs" / f"{execution_id}.json"
    authorization_path = ACTIVATION_DIR / "authorizations" / f"{execution_id}.json"
    job, job_binding = _bound_json(job_path, label=f"{regime} loaderfix job")
    authorization, authorization_binding = _bound_json(
        authorization_path, label=f"{regime} loaderfix authorization"
    )
    source_job_binding = _mapping(
        job.get("source_job"), label=f"{regime} source job binding"
    )
    source_job_path = REPO_ROOT / str(source_job_binding.get("path", ""))
    source_job, normalized_source_job_binding = _bound_json(
        source_job_path, label=f"{regime} source job"
    )
    if any(
        source_job_binding.get(key) != normalized_source_job_binding.get(key)
        for key in ("sha256", "size_bytes", "canonical_sha256")
    ):
        raise LivePage7InputError(f"{regime}: source job byte binding drifted")

    resume = _mapping(job.get("resume_input"), label=f"{regime} resume input")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("execution_id") != execution_id
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("regime_id") != regime
        or job.get("nph") != 7
        or job.get("target_horizon") != 50
        or job.get("execution_mode")
        != "authenticated_accepted_state_resume_to_50"
        or job.get("route_contract_sha256") != completed.ROUTE_CONTRACT_SHA256
        or job.get("route_profile") != ROUTE_PROFILE
        or job.get("scientific_protocol_sha256") != protocol_sha256
        or job.get("implementation_repair") != IMPLEMENTATION_REPAIR
        or job.get("scientific_protocol_changed") is not False
        or job.get("scientific_settings_changed") != []
        or job.get("source_job_preserved_held") is not True
        or resume.get("resume_controller_round") != RESUME_ROUNDS[regime]
        or resume.get("validation_status") != "passed"
        or resume.get("pointer_closed") is not True
    ):
        raise LivePage7InputError(f"{regime}: loaderfix job authority drifted")
    exact = _finite(
        source_job.get("exact_same_cutoff_energy"), label=f"{regime} exact energy"
    )
    if (
        source_job.get("route_contract_sha256") != completed.ROUTE_CONTRACT_SHA256
        or source_job.get("protocol_sha256") != protocol_sha256
        or source_job.get("candidate_representation") != "single_pauli_word_v1"
        or source_job.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or source_job.get("resource_weighting_scope")
        != "all_phase_resource_weighting_v1"
        or source_job.get("phase_i_candidate_supply")
        != "global_guarded_singleton_pool_v1"
        or source_job.get("phase_i_shortlist_size") != 24
        or source_job.get("phase_ii_shortlist_size") != 12
        or source_job.get("phase_iii_admission_cardinality") != 1
        or source_job.get("insertion_policy") != "plateau_commutation"
    ):
        raise LivePage7InputError(f"{regime}: source scientific route drifted")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("execution_id") != execution_id
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("job_sha256") != job.get("sha256")
        or authorization.get("scientific_protocol_sha256") != protocol_sha256
        or authorization.get("implementation_repair") != IMPLEMENTATION_REPAIR
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise LivePage7InputError(f"{regime}: authorization authority drifted")

    activation_rows = {
        str(_mapping(row, label="activation row").get("execution_id")): _mapping(
            row, label="activation row"
        )
        for row in _sequence(activation.get("executions"), label="activation rows")
    }
    activation_authorizations = {
        str(_mapping(row, label="activation authorization").get("execution_id")):
        _mapping(row, label="activation authorization")
        for row in _sequence(
            activation.get("execution_authorizations"),
            label="activation authorizations",
        )
    }
    activation_row = _mapping(
        activation_rows.get(execution_id), label=f"{regime} activation row"
    )
    activation_job = _mapping(
        activation_row.get("job"), label=f"{regime} activation job"
    )
    activation_auth = _mapping(
        activation_authorizations.get(execution_id),
        label=f"{regime} activation authorization",
    )
    if (
        activation_row.get("queue_index") != PROC_IDS[regime]
        or activation_job.get("canonical_sha256") != job.get("sha256")
        or activation_job.get("sha256") != job_binding["sha256"]
        or activation_auth.get("canonical_sha256") != authorization.get("sha256")
        or activation_auth.get("sha256") != authorization_binding["sha256"]
    ):
        raise LivePage7InputError(f"{regime}: activation row binding drifted")
    submission_rows = {
        str(_mapping(row, label="submission row").get("regime_id")): _mapping(
            row, label="submission row"
        )
        for row in _sequence(submission.get("rows"), label="submission rows")
    }
    submission_row = _mapping(
        submission_rows.get(regime), label=f"{regime} submission row"
    )
    if (
        submission_row.get("proc_id") != PROC_IDS[regime]
        or submission_row.get("resume_controller_round") != RESUME_ROUNDS[regime]
    ):
        raise LivePage7InputError(f"{regime}: submission row drifted")
    return {
        "exact_same_cutoff_energy": exact,
        "resume_controller_round": RESUME_ROUNDS[regime],
        "execution_id": execution_id,
        "regime_id": regime,
        "cluster_id": CLUSTER_ID,
        "proc_id": PROC_IDS[regime],
        "scientific_protocol_sha256": protocol_sha256,
        "route_contract_sha256": completed.ROUTE_CONTRACT_SHA256,
        "route_profile": ROUTE_PROFILE,
        "package_manifest": package_binding,
        "activation_manifest": activation_binding,
        "submission_receipt": submission_binding,
        "job": job_binding,
        "authorization": authorization_binding,
        "source_job": normalized_source_job_binding,
        "source_checkpoint_sha256": resume.get("checkpoint_sha256"),
    }


def _loaderfix_protocol_sha256(regime: str) -> str:
    execution_id = _expected_execution_id(regime)
    job, _ = _bound_json(
        PACKAGE_DIR / "jobs" / f"{execution_id}.json",
        label=f"{regime} loaderfix job",
    )
    protocol = job.get("scientific_protocol_sha256")
    if not isinstance(protocol, str) or not HEX64.fullmatch(protocol):
        raise LivePage7InputError(f"{regime}: loaderfix protocol SHA-256 is invalid")
    return protocol


def _expected_snapshot_execution_binding(
    *, regime: str, authority: Mapping[str, Any], protocol_sha256: str
) -> dict[str, Any]:
    expected = {
        "execution_id": _expected_execution_id(regime),
        "regime_id": regime,
        "cluster_id": CLUSTER_ID,
        "proc_id": PROC_IDS[regime],
        "scientific_protocol_sha256": protocol_sha256,
        "route_contract_sha256": completed.ROUTE_CONTRACT_SHA256,
        "route_profile": ROUTE_PROFILE,
        "accepted_state_resume_schema": "paper_i_canonical_accepted_state_resume_v1",
        "source_checkpoint_sha256": authority.get("source_checkpoint_sha256"),
        "source_controller_round": authority.get("resume_controller_round"),
        "strict_numerical_replay_passed": True,
        "route_and_problem_binding_passed": True,
    }
    for field in (
        "execution_id",
        "regime_id",
        "cluster_id",
        "proc_id",
        "scientific_protocol_sha256",
        "route_contract_sha256",
        "route_profile",
    ):
        if authority.get(field) != expected[field]:
            raise LivePage7InputError(f"{regime}: authority {field} drifted")
    source_sha = expected["source_checkpoint_sha256"]
    if not isinstance(source_sha, str) or not HEX64.fullmatch(source_sha):
        raise LivePage7InputError(f"{regime}: authority source checkpoint is invalid")
    return expected


def _validate_streamed_execution_identity(
    *, regime: str, streamed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    observed = _mapping(
        streamed.get("execution_identity"),
        label=f"{regime} streamed checkpoint execution identity",
    )
    checkpoint_fields = {
        "route_profile",
        "route_contract_sha256",
        "accepted_state_resume_schema",
        "source_checkpoint_sha256",
        "source_controller_round",
        "strict_numerical_replay_passed",
        "route_and_problem_binding_passed",
    }
    if set(observed) != checkpoint_fields or any(
        observed.get(field) != expected.get(field) for field in checkpoint_fields
    ):
        raise LivePage7InputError(
            f"{regime}: checkpoint execution/scientific identity drifted"
        )


def _snapshot_projection_binding(
    archive_path: Path, member: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "path": f"{archive_path.resolve().as_posix()}#{member['path']}",
        "sha256": str(member["sha256"]),
        "size_bytes": int(member["size_bytes"]),
    }


def build_live_projection_from_snapshot(
    *,
    base_adapter_path: Path,
    regime: str,
    archive_path: Path,
    validation_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Stream one validated checkpoint triplet into a compact live projection."""

    if regime not in completed.NPH7_REGIMES:
        raise LivePage7InputError("live snapshot projection supports nph=7 only")
    base = completed.validate_adapter(base_adapter_path)
    base_cells = {
        str(cell["regime_id"]): _mapping(cell, label="base adapter cell")
        for cell in base["cells"]
    }
    append = _mapping(base_cells[regime].get("append"), label=f"{regime} Append")
    protocol_sha256 = _loaderfix_protocol_sha256(regime)
    authority = _validate_loaderfix_authority(
        regime,
        execution_id=_expected_execution_id(regime),
        protocol_sha256=protocol_sha256,
    )
    receipt = _validate_snapshot_receipt(
        archive_path=archive_path,
        validation_path=validation_path,
        regime=regime,
    )
    streamed = _stream_snapshot_archive(archive_path=archive_path, receipt=receipt)
    execution_binding = _expected_snapshot_execution_binding(
        regime=regime, authority=authority, protocol_sha256=protocol_sha256
    )
    _validate_streamed_execution_identity(
        regime=regime, streamed=streamed, expected=execution_binding
    )
    live_round = int(streamed["history_count"])
    if (
        live_round != receipt["checkpoint_depth"]
        or not authority["resume_controller_round"] <= live_round < 50
    ):
        raise LivePage7InputError(f"{regime}: snapshot live horizon is invalid")
    exact = float(authority["exact_same_cutoff_energy"])
    append_exact = _finite(
        append.get("exact_same_cutoff_energy"), label=f"{regime} Append exact"
    )
    if not math.isclose(exact, append_exact, rel_tol=0.0, abs_tol=1.0e-12):
        raise LivePage7InputError(f"{regime}: snapshot exact reference drifted")
    energies = [
        _finite(value, label=f"{regime} streamed energy")
        for value in streamed["energies"]
    ]
    if len(energies) != live_round + 1:
        raise LivePage7InputError(f"{regime}: streamed trajectory length drifted")
    append_points = _sequence(append.get("points"), label=f"{regime} Append points")
    append_initial = _finite(
        _mapping(append_points[0], label=f"{regime} Append initial point").get(
            "energy"
        ),
        label=f"{regime} Append initial energy",
    )
    if not math.isclose(energies[0], append_initial, rel_tol=0.0, abs_tol=1.0e-12):
        raise LivePage7InputError(f"{regime}: snapshot initial energy drifted")
    points = [
        {"round": index, "energy": energy, "delta_e": abs(energy - exact)}
        for index, energy in enumerate(energies)
    ]
    members = receipt["members"]
    archive_binding = copy.deepcopy(receipt["archive_binding"])
    validation_binding = copy.deepcopy(receipt["receipt_binding"])
    projection = digested(
        {
            "schema": LIVE_PROJECTION_SCHEMA,
            "status": LIVE_PROJECTION_STATUS,
            "classification": LIVE_CLASSIFICATION,
            "paper_evidence_adopted": False,
            "regime_id": regime,
            "execution_id": _expected_execution_id(regime),
            "cluster_id": CLUSTER_ID,
            "proc_id": PROC_IDS[regime],
            "observed_utc": receipt["observed_utc"],
            "scheduler_state": "running",
            "target_horizon": 50,
            "terminal": False,
            "route_contract_sha256": completed.ROUTE_CONTRACT_SHA256,
            "scientific_protocol_sha256": protocol_sha256,
            "implementation_repair": copy.deepcopy(IMPLEMENTATION_REPAIR),
            "scientific_protocol_changed": False,
            "scientific_settings_changed": [],
            "exact_same_cutoff_energy": exact,
            "live_controller_round": live_round,
            "active_ansatz_depth": int(streamed["active_ansatz_depth"]),
            "points": points,
            "algorithmic_work": copy.deepcopy(streamed["algorithmic_work"]),
            "qiskit_status": LIVE_QISKIT_STATUS,
            "qiskit_costs": copy.deepcopy(QISKIT_PENDING),
            "snapshot_archive": archive_binding,
            "snapshot_validation_receipt": validation_binding,
            "snapshot_execution_binding": execution_binding,
            "snapshot": {
                "checkpoint": _snapshot_projection_binding(
                    archive_path, members["checkpoint.json"]
                ),
                "estimator_ledger_checkpoint": _snapshot_projection_binding(
                    archive_path, members[receipt["pointers"]["ledger"]["path"]]
                ),
                "verified_resume_sidecar": _snapshot_projection_binding(
                    archive_path, members[receipt["pointers"]["resume"]["path"]]
                ),
            },
            "snapshot_validation": copy.deepcopy(SNAPSHOT_VALIDATION),
        }
    )
    if output.exists():
        existing = _load_json(output, label="existing live projection")
        if canonical_json_bytes(existing) != canonical_json_bytes(projection):
            raise LivePage7InputError("live projection output already differs")
    else:
        completed.legacy_page._atomic_write_json(output, projection)
    return copy.deepcopy(projection)


def _reauthenticate_projection_snapshot(
    *,
    projection: Mapping[str, Any],
    regime: str,
    authority: Mapping[str, Any],
    protocol_sha256: str,
    exact: float,
    observed_utc: str,
    live_round: int,
    active_depth: int,
    points: Sequence[Mapping[str, Any]],
    work: Mapping[str, Any],
) -> dict[str, Any]:
    archive_binding = _snapshot_binding(
        projection.get("snapshot_archive"), label=f"{regime} snapshot archive"
    )
    validation_binding = _snapshot_binding(
        projection.get("snapshot_validation_receipt"),
        label=f"{regime} snapshot validation receipt",
    )
    archive_path = Path(archive_binding["path"])
    validation_path = Path(validation_binding["path"])
    if not archive_path.is_absolute() or not validation_path.is_absolute():
        raise LivePage7InputError(f"{regime}: snapshot source paths are not absolute")
    receipt = _validate_snapshot_receipt(
        archive_path=archive_path,
        validation_path=validation_path,
        regime=regime,
    )
    if (
        receipt["archive_binding"] != archive_binding
        or receipt["receipt_binding"] != validation_binding
    ):
        raise LivePage7InputError(f"{regime}: snapshot source byte binding drifted")
    streamed = _stream_snapshot_archive(archive_path=archive_path, receipt=receipt)
    execution_binding = _expected_snapshot_execution_binding(
        regime=regime, authority=authority, protocol_sha256=protocol_sha256
    )
    _validate_streamed_execution_identity(
        regime=regime, streamed=streamed, expected=execution_binding
    )
    declared_execution = _mapping(
        projection.get("snapshot_execution_binding"),
        label=f"{regime} snapshot execution binding",
    )
    if dict(declared_execution) != execution_binding:
        raise LivePage7InputError(f"{regime}: declared snapshot execution drifted")
    snapshot = _mapping(projection.get("snapshot"), label=f"{regime} snapshot")
    if set(snapshot) != {
        "checkpoint",
        "estimator_ledger_checkpoint",
        "verified_resume_sidecar",
    }:
        raise LivePage7InputError(f"{regime}: checkpoint triplet fields drifted")
    snapshot_bindings = {
        key: _snapshot_binding(snapshot[key], label=f"{regime} {key}")
        for key in snapshot
    }
    expected_snapshot = {
        "checkpoint": _snapshot_projection_binding(
            archive_path, receipt["members"]["checkpoint.json"]
        ),
        "estimator_ledger_checkpoint": _snapshot_projection_binding(
            archive_path,
            receipt["members"][receipt["pointers"]["ledger"]["path"]],
        ),
        "verified_resume_sidecar": _snapshot_projection_binding(
            archive_path,
            receipt["members"][receipt["pointers"]["resume"]["path"]],
        ),
    }
    streamed_points = [
        {
            "round": index,
            "energy": float(energy),
            "delta_e": abs(float(energy) - exact),
        }
        for index, energy in enumerate(streamed["energies"])
    ]
    if (
        snapshot_bindings != expected_snapshot
        or receipt["observed_utc"] != observed_utc
        or receipt["checkpoint_depth"] != live_round
        or streamed["history_count"] != live_round
        or streamed["active_ansatz_depth"] != active_depth
        or canonical_json_bytes(streamed_points) != canonical_json_bytes(points)
        or canonical_json_bytes(streamed["algorithmic_work"])
        != canonical_json_bytes(work)
    ):
        raise LivePage7InputError(
            f"{regime}: compact projection does not equal authenticated snapshot"
        )
    return {
        "archive_binding": archive_binding,
        "validation_binding": validation_binding,
        "snapshot_bindings": snapshot_bindings,
        "execution_binding": execution_binding,
    }


def validate_live_projection(
    path: Path, *, regime: str, append_cell: Mapping[str, Any]
) -> dict[str, Any]:
    if regime not in completed.NPH7_REGIMES or not path.is_file() or path.is_symlink():
        raise LivePage7InputError(f"{regime}: live projection is unavailable")
    projection = _load_json(path, label=f"{regime} live projection")
    canonical = _verify_self_digest(projection, label=f"{regime} live projection")
    execution_id = _expected_execution_id(regime)
    protocol_sha256 = projection.get("scientific_protocol_sha256")
    if not isinstance(protocol_sha256, str) or not HEX64.fullmatch(protocol_sha256):
        raise LivePage7InputError(f"{regime}: scientific protocol SHA-256 is invalid")
    authority = _validate_loaderfix_authority(
        regime, execution_id=execution_id, protocol_sha256=protocol_sha256
    )
    if (
        projection.get("schema") != LIVE_PROJECTION_SCHEMA
        or projection.get("status") != LIVE_PROJECTION_STATUS
        or projection.get("classification") != LIVE_CLASSIFICATION
        or projection.get("paper_evidence_adopted") is not False
        or projection.get("regime_id") != regime
        or projection.get("execution_id") != execution_id
        or projection.get("cluster_id") != CLUSTER_ID
        or projection.get("proc_id") != PROC_IDS[regime]
        or projection.get("scheduler_state") != "running"
        or projection.get("target_horizon") != 50
        or projection.get("terminal") is not False
        or projection.get("route_contract_sha256")
        != completed.ROUTE_CONTRACT_SHA256
        or projection.get("implementation_repair") != IMPLEMENTATION_REPAIR
        or projection.get("scientific_protocol_changed") is not False
        or projection.get("scientific_settings_changed") != []
        or projection.get("qiskit_status") != LIVE_QISKIT_STATUS
        or projection.get("qiskit_costs") != QISKIT_PENDING
        or projection.get("snapshot_validation") != SNAPSHOT_VALIDATION
    ):
        raise LivePage7InputError(f"{regime}: live projection identity drifted")
    observed_utc = _validate_observed_utc(projection.get("observed_utc"))
    exact = _finite(
        projection.get("exact_same_cutoff_energy"),
        label=f"{regime} exact energy",
    )
    append_exact = _finite(
        append_cell.get("exact_same_cutoff_energy"),
        label=f"{regime} Append exact energy",
    )
    if not (
        math.isclose(exact, append_exact, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(
            exact,
            float(authority["exact_same_cutoff_energy"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise LivePage7InputError(f"{regime}: same-cutoff reference drifted")
    live_round = _integer(
        projection.get("live_controller_round"), label=f"{regime} live round"
    )
    if not authority["resume_controller_round"] <= live_round < 50:
        raise LivePage7InputError(f"{regime}: live round is outside resume prefix")
    depth = _integer(
        projection.get("active_ansatz_depth"), label=f"{regime} active depth"
    )
    raw_points = _sequence(projection.get("points"), label=f"{regime} points")
    if len(raw_points) != live_round + 1:
        raise LivePage7InputError(f"{regime}: trajectory length does not match live round")
    points: list[dict[str, Any]] = []
    previous_energy: float | None = None
    for expected_round, raw in enumerate(raw_points):
        point = _mapping(raw, label=f"{regime} point {expected_round}")
        energy = _finite(point.get("energy"), label=f"{regime} energy")
        delta_e = _finite(
            point.get("delta_e"), label=f"{regime} delta E", minimum=0.0
        )
        if (
            point.get("round") != expected_round
            or not math.isclose(
                delta_e,
                abs(energy - exact),
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            )
            or (previous_energy is not None and energy > previous_energy + 1.0e-10)
        ):
            raise LivePage7InputError(f"{regime}: trajectory closure drifted")
        previous_energy = energy
        points.append(
            {"round": expected_round, "energy": energy, "delta_e": delta_e}
        )
    append_points = _sequence(
        append_cell.get("points"), label=f"{regime} Append points"
    )
    if not math.isclose(
        points[0]["energy"],
        _finite(
            _mapping(append_points[0], label="Append initial point").get("energy"),
            label="Append initial energy",
        ),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise LivePage7InputError(f"{regime}: initial-state energy drifted")
    work = _validate_work(
        projection.get("algorithmic_work"), label=f"{regime} algorithmic work"
    )
    authenticated = _reauthenticate_projection_snapshot(
        projection=projection,
        regime=regime,
        authority=authority,
        protocol_sha256=protocol_sha256,
        exact=exact,
        observed_utc=observed_utc,
        live_round=live_round,
        active_depth=depth,
        points=points,
        work=work,
    )
    binding = file_binding(path)
    return {
        "schema": LIVE_PROJECTION_SCHEMA,
        "status": "live_partial",
        "regime_id": regime,
        "display_name": completed.REGIME_LABELS[regime],
        "nph": 7,
        "execution_id": execution_id,
        "cluster_id": CLUSTER_ID,
        "proc_id": PROC_IDS[regime],
        "observed_utc": observed_utc,
        "scheduler_state": "running",
        "target_horizon": 50,
        "terminal": False,
        "exact_same_cutoff_energy": exact,
        "live_controller_round": live_round,
        "active_ansatz_depth": depth,
        "points": points,
        "available_prefix_effective_plateau": completed._effective_plateau(
            points, label=f"{regime} live prefix"
        ),
        "algorithmic_work": work,
        "qiskit_status": LIVE_QISKIT_STATUS,
        "qiskit_costs": copy.deepcopy(QISKIT_PENDING),
        "source": {
            "projection": {
                **binding,
                "canonical_sha256": canonical,
            },
            "snapshot_archive": authenticated["archive_binding"],
            "snapshot_validation_receipt": authenticated["validation_binding"],
            "snapshot_execution_binding": authenticated["execution_binding"],
            "snapshot": authenticated["snapshot_bindings"],
            "snapshot_validation": copy.deepcopy(SNAPSHOT_VALIDATION),
            "scientific_protocol_sha256": protocol_sha256,
            "route_contract_sha256": completed.ROUTE_CONTRACT_SHA256,
            "implementation_repair": copy.deepcopy(IMPLEMENTATION_REPAIR),
            **authority,
        },
    }


def _cell_digest(cell: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(cell)).hexdigest()


def _append_digest(cell: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(cell.get("append"))).hexdigest()


def _write_monotone_live_adapter(
    path: Path, adapter: Mapping[str, Any]
) -> dict[str, Any]:
    if not path.exists():
        completed.legacy_page._atomic_write_json(path, adapter)
        return copy.deepcopy(dict(adapter))
    existing_raw = _load_json(path, label="existing live page-7 adapter")
    existing_validated = validate_live_adapter(path)
    if canonical_json_bytes(existing_raw) == canonical_json_bytes(adapter):
        return copy.deepcopy(dict(existing_raw))
    existing = {
        key: value
        for key, value in existing_validated.items()
        if key not in {"file_binding"}
    }
    if existing.get("base_adapter") != adapter.get("base_adapter"):
        raise LivePage7InputError("live adapter base binding drifted")
    if existing.get("completed_cell_sha256") != adapter.get("completed_cell_sha256"):
        raise LivePage7InputError("completed nph=3 cells drifted")
    if existing.get("append_cell_sha256") != adapter.get("append_cell_sha256"):
        raise LivePage7InputError("Append overlays drifted")
    old_cells = {
        str(cell["regime_id"]): _mapping(cell, label="old live cell")
        for cell in existing["cells"]
    }
    new_cells = {
        str(cell["regime_id"]): _mapping(cell, label="new live cell")
        for cell in adapter["cells"]
    }
    advanced = False
    for regime in completed.NPH7_REGIMES:
        old_ra = _mapping(old_cells[regime].get("ra"), label=f"old {regime} RA")
        new_ra = _mapping(new_cells[regime].get("ra"), label=f"new {regime} RA")
        old_round = int(old_ra["live_controller_round"])
        new_round = int(new_ra["live_controller_round"])
        if new_round < old_round:
            raise LivePage7InputError(f"{regime}: live horizon regressed")
        old_points = old_ra["points"]
        if canonical_json_bytes(new_ra["points"][: len(old_points)]) != canonical_json_bytes(
            old_points
        ):
            raise LivePage7InputError(f"{regime}: accepted trajectory prefix drifted")
        if new_ra["algorithmic_work"]["S_alg"] < old_ra["algorithmic_work"]["S_alg"]:
            raise LivePage7InputError(f"{regime}: S_alg regressed")
        if new_round > old_round:
            advanced = True
        elif canonical_json_bytes(old_cells[regime]) != canonical_json_bytes(
            new_cells[regime]
        ):
            raise LivePage7InputError(
                f"{regime}: same-horizon live cell changed"
            )
    if not advanced:
        raise LivePage7InputError("live adapter replacement has no horizon advance")
    completed.legacy_page._atomic_write_json(path, adapter)
    return copy.deepcopy(dict(adapter))


def build_live_adapter(
    *,
    base_adapter_path: Path,
    live_projections: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    if set(live_projections) != set(completed.NPH7_REGIMES):
        raise LivePage7InputError("exactly three nph=7 live projections are required")
    base = completed.validate_adapter(base_adapter_path)
    if (
        tuple(base.get("completed_regimes", ())) != tuple(completed.REGIME_ORDER[:3])
        or tuple(base.get("pending_regimes", ())) != tuple(completed.REGIME_ORDER[3:])
    ):
        raise LivePage7InputError("base adapter is not the expected 3+3 state")
    base_cells = {
        str(cell["regime_id"]): _mapping(cell, label="base cell")
        for cell in base["cells"]
    }
    validated = {
        regime: validate_live_projection(
            Path(live_projections[regime]),
            regime=regime,
            append_cell=_mapping(
                base_cells[regime].get("append"), label=f"{regime} Append"
            ),
        )
        for regime in completed.REGIME_ORDER[3:]
    }
    cells: list[dict[str, Any]] = []
    for regime in completed.REGIME_ORDER:
        base_cell = base_cells[regime]
        if regime in completed.NPH3_REGIMES:
            cells.append(copy.deepcopy(dict(base_cell)))
            continue
        live = validated[regime]
        final_point = _mapping(live["points"][-1], label=f"{regime} live terminal")
        cells.append(
            {
                "regime_id": regime,
                "display_name": completed.REGIME_LABELS[regime],
                "nph": 7,
                "status": "live_partial",
                "append": copy.deepcopy(base_cell["append"]),
                "ra": {
                    **copy.deepcopy(live),
                    "terminal": {
                        "round": live["live_controller_round"],
                        "delta_e": float(final_point["delta_e"]),
                        "energy": float(final_point["energy"]),
                        "costs": {
                            **copy.deepcopy(QISKIT_PENDING),
                            "S_alg": live["algorithmic_work"]["S_alg"],
                        },
                        "qiskit_status": LIVE_QISKIT_STATUS,
                    },
                },
                "common_accuracy": None,
            }
        )
    completed_cell_sha256 = {
        regime: _cell_digest(base_cells[regime])
        for regime in completed.REGIME_ORDER[:3]
    }
    append_cell_sha256 = {
        regime: _append_digest(base_cells[regime])
        for regime in completed.REGIME_ORDER
    }
    adapter = digested(
        {
            "schema": LIVE_ADAPTER_SCHEMA,
            "status": LIVE_ADAPTER_STATUS,
            "classification": LIVE_CLASSIFICATION,
            "paper_evidence_adopted": False,
            "regime_order": list(completed.REGIME_ORDER),
            "completed_regimes": list(completed.REGIME_ORDER[:3]),
            "live_partial_regimes": list(completed.REGIME_ORDER[3:]),
            "pending_regimes": [],
            "base_adapter": {
                **base["file_binding"],
                "canonical_sha256": base["sha256"],
            },
            "append_adapter": copy.deepcopy(base["append_adapter"]),
            "completed_cell_sha256": completed_cell_sha256,
            "append_cell_sha256": append_cell_sha256,
            "same_cutoff_reference": copy.deepcopy(base["same_cutoff_reference"]),
            "route_description": base["route_description"],
            "layout": {"panel_count": 6, "grid": "2x3", "page_count": 1},
            "marker_policy": {
                "complete": base["marker_policy"],
                "live_partial": LIVE_MARKER_POLICY,
            },
            "cost_policy": {
                **copy.deepcopy(base["cost_policy"]),
                "live_partial": {
                    "S_alg": "closed_occurrence_prefix",
                    "qiskit_fields": LIVE_QISKIT_STATUS,
                    "matched_costs": "pending_until_completed_round_50",
                },
            },
            "limitations": [LIVE_LIMITATION],
            "cells": cells,
        }
    )
    return _write_monotone_live_adapter(output, adapter)


def validate_live_adapter(path: Path) -> dict[str, Any]:
    adapter = _load_json(path, label="live page-7 adapter")
    canonical = _verify_self_digest(adapter, label="live page-7 adapter")
    if (
        adapter.get("schema") != LIVE_ADAPTER_SCHEMA
        or adapter.get("status") != LIVE_ADAPTER_STATUS
        or adapter.get("classification") != LIVE_CLASSIFICATION
        or adapter.get("paper_evidence_adopted") is not False
        or tuple(adapter.get("regime_order", ())) != completed.REGIME_ORDER
        or tuple(adapter.get("completed_regimes", ()))
        != tuple(completed.REGIME_ORDER[:3])
        or tuple(adapter.get("live_partial_regimes", ()))
        != tuple(completed.REGIME_ORDER[3:])
        or adapter.get("pending_regimes") != []
        or adapter.get("layout")
        != {"panel_count": 6, "grid": "2x3", "page_count": 1}
        or adapter.get("limitations") != [LIVE_LIMITATION]
    ):
        raise LivePage7InputError("live adapter identity drifted")
    base_binding = _mapping(adapter.get("base_adapter"), label="base adapter binding")
    base_path = Path(str(base_binding.get("path", ""))).expanduser().resolve()
    base = completed.validate_adapter(base_path)
    if (
        base_binding.get("canonical_sha256") != base.get("sha256")
        or base_binding.get("sha256") != base["file_binding"]["sha256"]
        or base_binding.get("size_bytes") != base["file_binding"]["size_bytes"]
        or adapter.get("append_adapter") != base.get("append_adapter")
        or adapter.get("same_cutoff_reference") != base.get("same_cutoff_reference")
    ):
        raise LivePage7InputError("live adapter base binding drifted")
    base_cells = {
        str(cell["regime_id"]): _mapping(cell, label="base cell")
        for cell in base["cells"]
    }
    cells = {
        str(cell["regime_id"]): _mapping(cell, label="live adapter cell")
        for cell in _sequence(adapter.get("cells"), label="live adapter cells")
    }
    if len(cells) != 6 or set(cells) != set(completed.REGIME_ORDER):
        raise LivePage7InputError("live adapter cell closure drifted")
    expected_completed = {
        regime: _cell_digest(base_cells[regime])
        for regime in completed.REGIME_ORDER[:3]
    }
    expected_append = {
        regime: _append_digest(base_cells[regime])
        for regime in completed.REGIME_ORDER
    }
    if (
        adapter.get("completed_cell_sha256") != expected_completed
        or adapter.get("append_cell_sha256") != expected_append
    ):
        raise LivePage7InputError("preserved cell digests drifted")
    for regime in completed.REGIME_ORDER:
        cell = cells[regime]
        if canonical_json_bytes(cell.get("append")) != canonical_json_bytes(
            base_cells[regime].get("append")
        ):
            raise LivePage7InputError(f"{regime}: Append overlay drifted")
        if regime in completed.NPH3_REGIMES:
            if canonical_json_bytes(cell) != canonical_json_bytes(base_cells[regime]):
                raise LivePage7InputError(f"{regime}: completed cell drifted")
            continue
        if cell.get("status") != "live_partial" or cell.get("common_accuracy") is not None:
            raise LivePage7InputError(f"{regime}: live cell status drifted")
        ra = _mapping(cell.get("ra"), label=f"{regime} live RA")
        source = _mapping(ra.get("source"), label=f"{regime} live source")
        projection_binding = _mapping(
            source.get("projection"), label=f"{regime} projection binding"
        )
        projection_path = Path(str(projection_binding.get("path", ""))).resolve()
        expected_ra = validate_live_projection(
            projection_path,
            regime=regime,
            append_cell=_mapping(cell.get("append"), label=f"{regime} Append"),
        )
        terminal_point = expected_ra["points"][-1]
        expected_ra["terminal"] = {
            "round": expected_ra["live_controller_round"],
            "delta_e": float(terminal_point["delta_e"]),
            "energy": float(terminal_point["energy"]),
            "costs": {
                **copy.deepcopy(QISKIT_PENDING),
                "S_alg": expected_ra["algorithmic_work"]["S_alg"],
            },
            "qiskit_status": LIVE_QISKIT_STATUS,
        }
        if canonical_json_bytes(ra) != canonical_json_bytes(expected_ra):
            raise LivePage7InputError(f"{regime}: live RA projection drifted")
    result = copy.deepcopy(adapter)
    result["sha256"] = canonical
    result["file_binding"] = file_binding(path)
    return result


def _format_delta_e(value: float) -> str:
    return completed.format_delta_e(value)


def _format_live_cost(ra: Mapping[str, Any]) -> str:
    return (
        r"$\mathrm{Qiskit\ pending};\ "
        rf"S_{{\rm alg}}={int(ra['algorithmic_work']['S_alg']):,}$"
    )


def render_plot(
    adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.1,
            "axes.labelsize": 8.3,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 4.05), constrained_layout=True)
    for index, regime in enumerate(completed.REGIME_ORDER):
        ax = axes.flat[index]
        cell = cells[regime]
        append = cell["append"]
        append_points = append["points"]
        ax.plot(
            [point["round"] for point in append_points],
            [max(float(point["delta_e"]), PLOT_FLOOR) for point in append_points],
            color="#4C78A8",
            linewidth=1.55,
        )
        append_marker = append["effective_plateau"]
        ax.scatter(
            [append_marker["round"]],
            [max(float(append_marker["delta_e"]), PLOT_FLOOR)],
            marker="o",
            color="#4C78A8",
            s=27,
            zorder=5,
        )
        ra = _mapping(cell.get("ra"), label=f"{regime} RA")
        ra_points = ra["points"]
        ax.plot(
            [point["round"] for point in ra_points],
            [max(float(point["delta_e"]), PLOT_FLOOR) for point in ra_points],
            color="#E45756",
            linewidth=1.75,
        )
        if cell["status"] == "complete":
            marker = ra["effective_plateau"]
            marker_round = int(marker["round"])
            marker_error = float(marker["delta_e"])
        else:
            marker_round = int(ra["live_controller_round"])
            marker_error = float(ra_points[-1]["delta_e"])
            ax.text(
                0.98,
                0.96,
                (
                    f"live k={marker_round}; DE={marker_error:.2e}\n"
                    f"S_alg={int(ra['algorithmic_work']['S_alg']):,}; "
                    "Qiskit pending"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.0,
                color="#9C2F2F",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#CCCCCC",
                    "alpha": 0.86,
                },
            )
        ax.scatter(
            [marker_round],
            [max(marker_error, PLOT_FLOOR)],
            marker="D",
            color="#E45756",
            s=28,
            zorder=5,
        )
        ax.set_title(str(cell["display_name"]))
        maximum = int(append["display_terminal_round"])
        ax.set_xlim(0, maximum)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", linewidth=0.4, alpha=0.28)
        ax.grid(True, which="minor", linewidth=0.22, alpha=0.12)
        if index // 3 == 1:
            ax.set_xlabel("ADAPT iteration")
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $\Delta E$")
    fig.suptitle(
        "Global-singleton RA plateau vs fresh Append-ADAPT singleton - live prefixes",
        fontsize=11.2,
        fontweight="bold",
    )
    fig.legend(
        handles=(
            Line2D(
                [0],
                [0],
                color="#4C78A8",
                marker="o",
                label="Fresh Append-ADAPT singleton",
            ),
            Line2D(
                [0],
                [0],
                color="#E45756",
                marker="D",
                label=(
                    "Historical-mean global-singleton RA "
                    "(complete plateau / live terminal marker)"
                ),
            ),
        ),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=7.7,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_page_tex(
    adapter: Mapping[str, Any], *, plot_pdf: Path, tex_path: Path
) -> None:
    endpoint_rows: list[str] = []
    matched_rows: list[str] = []
    for cell in adapter["cells"]:
        regime = str(cell["regime_id"])
        ra = cell["ra"]
        append_terminal = cell["append"]["terminal"]
        append_round = int(cell["append"]["display_terminal_round"])
        if cell["status"] == "complete":
            ra_terminal = ra["terminal"]
            ra_state = f"complete k={int(ra_terminal['round'])}"
            ra_cost = completed.format_costs(ra_terminal["costs"])
            common = cell["common_accuracy"]
            matched_rows.append(
                " & ".join(
                    (
                        completed.REGIME_ABBREVIATIONS[regime],
                        _format_delta_e(float(common["target_delta_e"])),
                        str(common["ra"]["round"]),
                        completed.format_costs(common["ra"]["costs"]),
                        str(common["append"]["round"]),
                        completed.format_costs(common["append"]["costs"]),
                    )
                )
                + r" \\"
            )
        else:
            ra_terminal = ra["terminal"]
            ra_state = f"live k={int(ra_terminal['round'])}"
            ra_cost = _format_live_cost(ra)
            matched_rows.append(
                " & ".join(
                    (
                        completed.REGIME_ABBREVIATIONS[regime],
                        r"$\text{nonterminal}$",
                        str(ra_terminal["round"]),
                        r"$\text{Qiskit pending}$",
                        "--",
                        r"$\text{pending to k=50}$",
                    )
                )
                + r" \\"
            )
        endpoint_rows.append(
            " & ".join(
                (
                    str(cell["display_name"]),
                    ra_state,
                    _format_delta_e(float(ra_terminal["delta_e"])),
                    ra_cost,
                    str(append_round),
                    _format_delta_e(float(append_terminal["delta_e"])),
                    completed.format_costs(append_terminal["costs"]),
                )
            )
            + r" \\"
        )
    plot_argument = completed.latex_escape(plot_pdf.resolve().as_posix())
    route = completed.latex_escape(str(adapter["route_description"]))
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.14in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.96\textwidth,height=3.30in,keepaspectratio]{{{plot_argument}}}
\vspace{{0.16em}}

\fontsize{{6.4}}{{6.8}}\selectfont
\setlength{{\tabcolsep}}{{2.1pt}}
\renewcommand{{\arraystretch}}{{0.82}}
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
Regime & RA state & $\Delta E^{{\rm RA}}$ & RA prefix cost & $k_A$ &
$\Delta E^{{\rm Append}}$ & $C^{{\rm Append}}$ \\
\midrule
{chr(10).join(endpoint_rows)}
\bottomrule
\end{{tabular}}
\vspace{{0.10em}}

{{\scriptsize\bfseries Equal-attainable-error costs for complete trajectories only}}
\vspace{{-0.12em}}

\fontsize{{6.1}}{{6.5}}\selectfont
\setlength{{\tabcolsep}}{{1.8pt}}
\begin{{tabular}}{{@{{}}ccrrrr@{{}}}}
\toprule
Reg. & $\Delta E_\cap$ & $k_\cap^{{\rm RA}}$ & $C_\cap^{{\rm RA}}$ &
$k_\cap^{{\rm Append}}$ & $C_\cap^{{\rm Append}}$ \\
\midrule
{chr(10).join(matched_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
\vspace{{-0.42em}}
\tiny
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$. For live nph=7
prefixes, $S_{{\rm alg}}$ is closed from the authenticated estimator-ledger
checkpoint; $N_{{2q}},D_{{2q}},D_c,W_{{1q}}$ are explicitly pending and are
not inferred. Complete rows retain the source-locked Table-I compiler
(optimization level 0, seed 7, reference state included). {route}
\end{{document}}
"""
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")


def build_assets(
    adapter: Mapping[str, Any], *, asset_dir: Path, asset_stem: str
) -> dict[str, Path]:
    if not completed.ASSET_STEM_RE.fullmatch(asset_stem) or asset_stem in {".", ".."}:
        raise LivePage7InputError("asset_stem must be a safe filename component")
    assets = {
        "plot_png": asset_dir / f"{asset_stem}_plot.png",
        "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
        "page_tex": asset_dir / f"{asset_stem}.tex",
        "page_pdf": asset_dir / f"{asset_stem}.pdf",
    }
    render_plot(adapter, png_path=assets["plot_png"], pdf_path=assets["plot_pdf"])
    write_page_tex(adapter, plot_pdf=assets["plot_pdf"], tex_path=assets["page_tex"])
    completed.legacy_page._compile_page(assets["page_tex"], assets["page_pdf"])
    return assets


def _report_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "regime_id": cell["regime_id"],
        "status": cell["status"],
        "append_terminal": copy.deepcopy(cell["append"]["terminal"]),
        "append_terminal_round": cell["append"]["display_terminal_round"],
    }
    if cell["status"] == "complete":
        result.update(
            {
                "ra_round_50": copy.deepcopy(cell["ra"]["terminal"]),
                "common_accuracy": copy.deepcopy(cell["common_accuracy"]),
                "ra_source": copy.deepcopy(cell["ra"]["source"]),
            }
        )
    else:
        ra = cell["ra"]
        result.update(
            {
                "live_controller_round": ra["live_controller_round"],
                "active_ansatz_depth": ra["active_ansatz_depth"],
                "live_delta_e": ra["terminal"]["delta_e"],
                "S_alg": ra["algorithmic_work"]["S_alg"],
                "qiskit_status": ra["qiskit_status"],
                "ra_source": copy.deepcopy(ra["source"]),
            }
        )
    return result


def _publication_replace(source: Path, target: Path) -> None:
    """One injectable atomic replacement boundary for transaction tests."""

    os.replace(source, target)


def _same_file_binding(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    actual = file_binding(path)
    return (
        actual["sha256"] == expected.get("sha256")
        and actual["size_bytes"] == expected.get("size_bytes")
    )


def update_page7(
    *,
    target_pdf: Path,
    target_provenance: Path,
    adapter_path: Path,
    asset_dir: Path,
    asset_stem: str,
) -> dict[str, Any]:
    adapter = validate_live_adapter(adapter_path)
    provenance = _load_json(target_provenance, label="target provenance")
    outputs = _mapping(provenance.get("outputs"), label="target outputs")
    pdf_binding = _mapping(
        outputs.get("partial_progress_pdf"), label="target PDF binding"
    )
    if (
        not target_pdf.is_file()
        or target_pdf.is_symlink()
        or target_provenance.is_symlink()
        or pdf_binding.get("sha256") != sha256_file(target_pdf)
        or pdf_binding.get("size_bytes") != target_pdf.stat().st_size
    ):
        raise LivePage7InputError("target PDF/provenance byte binding drifted")
    layout = _mapping(provenance.get("layout"), label="target layout")
    before_hashes = completed.legacy_page._page_content_hashes(target_pdf)
    if (
        len(before_hashes) != 7
        or layout.get("page_count") != 7
        or layout.get("page_6") != completed.EXPECTED_BASE_PAGE_6
        or layout.get("page_7") != completed.PAGE_ID
    ):
        raise LivePage7InputError("target is not the supported seven-page report")
    report = _mapping(
        provenance.get(completed.REPORT_KEY), label="existing page-7 report"
    )
    old_adapter_binding = _mapping(
        report.get("adapter"), label="existing page-7 adapter binding"
    )
    if old_adapter_binding.get("canonical_sha256") == adapter["sha256"]:
        return {
            "status": "already_current",
            "output_pdf": str(target_pdf),
            "output_provenance": str(target_provenance),
            "sha256": sha256_file(target_pdf),
            "pages": 7,
            "preserved_pages_1_6": True,
        }
    if (
        report.get("completed_cell_sha256") != adapter["completed_cell_sha256"]
        or report.get("append_cell_sha256") != adapter["append_cell_sha256"]
    ):
        raise LivePage7InputError("existing completed/Append page-7 cells drifted")
    old_horizons = report.get("live_horizons")
    if old_horizons is not None:
        old_horizons = _mapping(old_horizons, label="existing live horizons")
        new_horizons = {
            str(cell["regime_id"]): int(cell["ra"]["live_controller_round"])
            for cell in adapter["cells"]
            if cell["status"] == "live_partial"
        }
        if any(
            new_horizons.get(regime, -1) < int(old_horizons.get(regime, -1))
            for regime in completed.NPH7_REGIMES
        ) or not any(
            new_horizons[regime] > int(old_horizons[regime])
            for regime in completed.NPH7_REGIMES
        ):
            raise LivePage7InputError("page-7 live horizons did not advance monotonically")

    assets = build_assets(adapter, asset_dir=asset_dir, asset_stem=asset_stem)
    from pypdf import PdfReader, PdfWriter

    page_reader = PdfReader(str(assets["page_pdf"]), strict=False)
    if len(page_reader.pages) != 1:
        raise LivePage7InputError("live page asset is not exactly one page")
    writer = PdfWriter()
    existing_pages = PdfReader(str(target_pdf), strict=False).pages
    for page in existing_pages[:6]:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])
    temporary_pdf = target_pdf.with_name(f".{target_pdf.name}.live-page7.stage")
    temporary_provenance = target_provenance.with_name(
        f".{target_provenance.name}.live-page7.stage"
    )
    pdf_backup = target_pdf.with_name(f".{target_pdf.name}.live-page7.backup")
    provenance_backup = target_provenance.with_name(
        f".{target_provenance.name}.live-page7.backup"
    )
    transaction_paths = (
        temporary_pdf,
        temporary_provenance,
        pdf_backup,
        provenance_backup,
    )
    if any(path.exists() or path.is_symlink() for path in transaction_paths):
        raise LivePage7InputError("stale live-page7 transaction files exist")
    original_pdf_binding = file_binding(target_pdf)
    original_provenance_binding = file_binding(target_provenance)
    publication_started = False
    publication_complete = False
    rollback_complete = False
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
        after_hashes = completed.legacy_page._page_content_hashes(temporary_pdf)
        if len(after_hashes) != 7 or after_hashes[:6] != before_hashes[:6]:
            raise LivePage7InputError("live page update altered pages 1--6")
        new_pdf_binding = file_binding(temporary_pdf)
        new_pdf_binding["path"] = str(target_pdf.resolve())
        updated = copy.deepcopy(provenance)
        updated["outputs"]["partial_progress_pdf"] = new_pdf_binding
        output_keys: list[str] = []
        for output_key, asset_key in (
            ("historical_mean_global_singleton_full6_plot_png", "plot_png"),
            ("historical_mean_global_singleton_full6_plot_pdf", "plot_pdf"),
            ("historical_mean_global_singleton_full6_page_tex", "page_tex"),
            ("historical_mean_global_singleton_full6_page_pdf", "page_pdf"),
        ):
            updated["outputs"][output_key] = file_binding(assets[asset_key])
            output_keys.append(output_key)
        live_horizons = {
            str(cell["regime_id"]): int(cell["ra"]["live_controller_round"])
            for cell in adapter["cells"]
            if cell["status"] == "live_partial"
        }
        updated[completed.REPORT_KEY] = {
            "schema": completed.PAGE_ID,
            "classification": LIVE_CLASSIFICATION,
            "paper_evidence_adopted": False,
            "page_id": completed.PAGE_ID,
            "adapter": {
                **adapter["file_binding"],
                "canonical_sha256": adapter["sha256"],
            },
            "base_adapter": copy.deepcopy(adapter["base_adapter"]),
            "prior_adapter": copy.deepcopy(report.get("adapter")),
            "completed_regimes": copy.deepcopy(adapter["completed_regimes"]),
            "live_partial_regimes": copy.deepcopy(adapter["live_partial_regimes"]),
            "pending_regimes": [],
            "completed_cell_sha256": copy.deepcopy(
                adapter["completed_cell_sha256"]
            ),
            "append_cell_sha256": copy.deepcopy(adapter["append_cell_sha256"]),
            "live_horizons": live_horizons,
            "route_description": adapter["route_description"],
            "marker_policy": copy.deepcopy(adapter["marker_policy"]),
            "cost_policy": copy.deepcopy(adapter["cost_policy"]),
            "limitations": copy.deepcopy(adapter["limitations"]),
            "cells": [_report_cell(cell) for cell in adapter["cells"]],
            "structural_validation": {
                "pages": 7,
                "preserved_pages_1_6_content_sha256": before_hashes[:6],
                "prior_page_7_content_sha256": before_hashes[6],
                "new_page_7_content_sha256": after_hashes[6],
            },
            "outputs": {
                key: copy.deepcopy(updated["outputs"][key]) for key in output_keys
            },
        }
        limitations = [
            item
            for item in updated.get("limitations", ())
            if item not in {completed.LIMITATION, LIVE_LIMITATION}
        ]
        limitations.append(LIVE_LIMITATION)
        updated["limitations"] = limitations
        completed.legacy_page._atomic_write_json(temporary_provenance, updated)
        staged_provenance = _load_json(
            temporary_provenance, label="staged target provenance"
        )
        staged_pdf_binding = _mapping(
            _mapping(staged_provenance.get("outputs"), label="staged outputs").get(
                "partial_progress_pdf"
            ),
            label="staged target PDF binding",
        )
        if (
            canonical_json_bytes(staged_provenance) != canonical_json_bytes(updated)
            or staged_pdf_binding.get("sha256") != sha256_file(temporary_pdf)
            or staged_pdf_binding.get("size_bytes") != temporary_pdf.stat().st_size
            or completed.legacy_page._page_content_hashes(temporary_pdf)
            != after_hashes
        ):
            raise LivePage7InputError("staged PDF/provenance pair failed validation")
        if (
            not _same_file_binding(target_pdf, original_pdf_binding)
            or not _same_file_binding(
                target_provenance, original_provenance_binding
            )
        ):
            raise LivePage7InputError(
                "target PDF/provenance changed while page assets were built"
            )
        staged_provenance_binding = file_binding(temporary_provenance)
        shutil.copy2(target_pdf, pdf_backup)
        shutil.copy2(target_provenance, provenance_backup)
        if (
            not _same_file_binding(pdf_backup, original_pdf_binding)
            or not _same_file_binding(
                provenance_backup, original_provenance_binding
            )
        ):
            raise LivePage7InputError("transaction backups failed byte validation")
        if (
            not _same_file_binding(target_pdf, original_pdf_binding)
            or not _same_file_binding(
                target_provenance, original_provenance_binding
            )
        ):
            raise LivePage7InputError(
                "target PDF/provenance changed immediately before publication"
            )
        publication_started = True
        try:
            _publication_replace(temporary_pdf, target_pdf)
            _publication_replace(temporary_provenance, target_provenance)
            final_provenance = _load_json(
                target_provenance, label="published target provenance"
            )
            if (
                canonical_json_bytes(final_provenance)
                != canonical_json_bytes(updated)
                or not _same_file_binding(target_pdf, new_pdf_binding)
                or not _same_file_binding(
                    target_provenance, staged_provenance_binding
                )
                or completed.legacy_page._page_content_hashes(target_pdf)
                != after_hashes
            ):
                raise LivePage7InputError(
                    "published PDF/provenance pair failed validation"
                )
            publication_complete = True
        except BaseException as publication_error:
            rollback_errors: list[str] = []
            for backup, target in (
                (pdf_backup, target_pdf),
                (provenance_backup, target_provenance),
            ):
                try:
                    os.replace(backup, target)
                except OSError as exc:
                    rollback_errors.append(f"{target}: {exc}")
            rollback_complete = (
                not rollback_errors
                and _same_file_binding(target_pdf, original_pdf_binding)
                and _same_file_binding(
                    target_provenance, original_provenance_binding
                )
            )
            if not rollback_complete:
                detail = "; ".join(rollback_errors) or "restored bytes drifted"
                raise LivePage7InputError(
                    f"live-page7 publication and rollback failed: {detail}"
                ) from publication_error
            raise LivePage7InputError(
                "live-page7 publication failed and was rolled back"
            ) from publication_error
    finally:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        if publication_complete or rollback_complete or not publication_started:
            pdf_backup.unlink(missing_ok=True)
            provenance_backup.unlink(missing_ok=True)
    return {
        "status": "replaced_page_7_with_live_prefixes",
        "output_pdf": str(target_pdf),
        "output_provenance": str(target_provenance),
        "sha256": sha256_file(target_pdf),
        "pages": 7,
        "live_horizons": live_horizons,
        "preserved_pages_1_6": True,
    }


def _projection_args(values: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for raw in values:
        regime, separator, path_text = raw.partition("=")
        if (
            not separator
            or regime not in completed.NPH7_REGIMES
            or not path_text
            or regime in paths
        ):
            raise LivePage7InputError(
                "--live-projection must be one unique nph7 REGIME=/path.json"
            )
        paths[regime] = Path(path_text).expanduser().resolve()
    if set(paths) != set(completed.NPH7_REGIMES):
        raise LivePage7InputError("all three nph=7 live projections are required")
    return paths


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    projection = commands.add_parser(
        "build-projection",
        help="Stream one validated live checkpoint tar into a compact projection.",
    )
    projection.add_argument("--base-adapter", type=Path, required=True)
    projection.add_argument(
        "--regime", choices=tuple(sorted(completed.NPH7_REGIMES)), required=True
    )
    projection.add_argument("--snapshot-tar", type=Path, required=True)
    projection.add_argument("--snapshot-validation", type=Path, required=True)
    projection.add_argument("--output", type=Path, required=True)

    update = commands.add_parser(
        "update-page",
        help="Build the guarded six-cell adapter and replace only PDF page 7.",
    )
    update.add_argument("--base-adapter", type=Path, required=True)
    update.add_argument(
        "--live-projection",
        action="append",
        default=[],
        metavar="REGIME=PROJECTION.json",
        help="Repeat exactly once for weak_strong, intermediate_strong, and strong_strong_u8.",
    )
    update.add_argument("--adapter", type=Path, required=True)
    update.add_argument("--target-pdf", type=Path, required=True)
    update.add_argument("--target-provenance", type=Path, required=True)
    update.add_argument("--asset-dir", type=Path, required=True)
    update.add_argument("--asset-stem", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build-projection":
            projection = build_live_projection_from_snapshot(
                base_adapter_path=args.base_adapter.resolve(),
                regime=args.regime,
                archive_path=args.snapshot_tar.resolve(),
                validation_path=args.snapshot_validation.resolve(),
                output=args.output.resolve(),
            )
            result = {
                "status": "built_live_projection",
                "output": file_binding(args.output.resolve()),
                "canonical_sha256": projection["sha256"],
                "regime_id": projection["regime_id"],
                "live_controller_round": projection["live_controller_round"],
                "delta_e": projection["points"][-1]["delta_e"],
                "S_alg": projection["algorithmic_work"]["S_alg"],
                "qiskit_status": projection["qiskit_status"],
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        projections = _projection_args(args.live_projection)
        build_live_adapter(
            base_adapter_path=args.base_adapter.resolve(),
            live_projections=projections,
            output=args.adapter.resolve(),
        )
        result = update_page7(
            target_pdf=args.target_pdf.resolve(),
            target_provenance=args.target_provenance.resolve(),
            adapter_path=args.adapter.resolve(),
            asset_dir=args.asset_dir.resolve(),
            asset_stem=args.asset_stem,
        )
    except (OSError, RuntimeError, ValueError, LivePage7InputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
