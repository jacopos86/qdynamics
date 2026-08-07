#!/usr/bin/env python3
"""Fail-closed contract for the selected stationary Paper-I core package.

This module is data and validation only.  It never authorizes execution,
launches a scientific calculation, or talks to HTCondor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True


PACKAGE_ID = "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v1_chtc"
PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v1_chtc"
)
RUNTIME_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v1_chtc_runtime"
)
CAMPAIGN_ID = "paper_i_ra_adapt_stationary_late_core_v1"
RUN_CLASS = "paper_facing"
EXECUTION_TARGET = "chtc"
BATCH_NAME = "paper-i-ra-adapt-stationary-core-full48-r50-20260728-v1"

CORE_MATERIALIZATION_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v9"
)
CORE_BUNDLE_ID = "ra_repair_stationary_late_core_v1"
CORE_FINAL_RECEIPT_NAME = "final_publication_receipt.json"
CORE_FINAL_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_stationary_late_core_materialization_receipt_v1"
)
CORE_BUNDLE_MANIFEST_SCHEMA = "ra_adapt_run_bundle_v1"
CORE_SOURCE_LOCKS_SCHEMA = "ra_adapt_source_locks_v1"
CORE_EXPECTED_ARTIFACTS_SCHEMA = "ra_adapt_expected_artifacts_v1"
CORE_VALIDATION_SCHEMA = "ra_adapt_bundle_validation_report_v1"
RA_PROTOCOL_SCHEMA = "paper_i_ra_adapt_resolved_protocol_v1"
APPEND_PROTOCOL_SCHEMA = "paper_i_append_adapt_resolved_protocol_v1"
EXECUTION_TEMPLATE_SCHEMA = "ra_adapt_execution_manifest_template_v1"

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_package_manifest_v1"
)
EXECUTION_PLAN_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_execution_plan_v1"
)
JOB_SPEC_SCHEMA = "paper_i_ra_adapt_stationary_core_job_spec_v1"
P2_RECEIPT_SCHEMA = "paper_i_ra_adapt_stationary_core_p2_receipt_v1"
P3_RECEIPT_SCHEMA = "paper_i_ra_adapt_stationary_core_p3_receipt_v1"
P4_RECEIPT_SCHEMA = "paper_i_ra_adapt_stationary_core_p4_receipt_v1"
P4_SMOKE_SPEC_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_p4_smoke_spec_v1"
)
P4_SMOKE_RESULT_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_p4_smoke_result_v1"
)
PACKAGE_PREAUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_package_preauthorization_v1"
)
CONTROL_PLANE_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_control_plane_v1"
)
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_source_archive_manifest_v1"
)
SUBMISSION_AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_submission_authorization_v1"
)
WORKER_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_worker_receipt_v1"
)
FULL_RUN_SCIENTIFIC_CLOSURE_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_full_run_scientific_closure_v1"
)
G11_BOUNDED_REPLAY_DIAGNOSTIC_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_g11_bounded_replay_diagnostic_v1"
)
FETCH_VALIDATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_fetched_validation_v1"
)

P2_RECEIPT_RELATIVE = "authority/p2_source_validation_receipt.json"
P3_RECEIPT_RELATIVE = "authority/p3_semantic_preflight_receipt.json"
P4_RECEIPT_RELATIVE = "authority/p4_packaged_dispatch_receipt.json"
PACKAGE_PREAUTHORIZATION_RELATIVE = (
    "authority/package_preauthorization_receipt.json"
)
SUBMISSION_AUTHORIZATION_RELATIVE = (
    "authority/submission_authorization_receipt.json"
)
CORE_FINAL_COPY_RELATIVE = "authority/core_final_publication_receipt.json"
USER_SELECTION_AUTHORITY_RELATIVE = (
    "agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/"
    "user-review-stationary-core-20260728.json"
)
USER_SELECTION_AUTHORITY_FILE_SHA256 = (
    "1b9e35d956ab7c93a1c02f0c4dd086906e8c7619cb182c064f259173b0fafad2"
)
USER_SELECTION_COPY_RELATIVE = (
    "authority/user_review_stationary_core_20260728.json"
)

REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
MAX_RUNTIME_SECONDS = 72 * 60 * 60

REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ED_REGIME_NAME_BY_ID = {
    "weak_weak": "weak-weak",
    "intermediate_weak": "intermediate-weak",
    "strong_weak_u8": "strong-weak",
    "weak_strong": "weak-strong",
    "intermediate_strong": "intermediate-strong",
    "strong_strong_u8": "strong-strong",
}
MACRO_ROUTES = (
    "append_macro",
    "ra_macro_append_only",
    "ra_macro_plateau",
    "ra_macro_always",
)
SINGLETON_ROUTES = (
    "append_singleton",
    "ra_singleton_append_only",
    "ra_singleton_plateau",
    "ra_singleton_always",
)
ROUTE_IDS = (*MACRO_ROUTES, *SINGLETON_ROUTES)
APPEND_ROUTES = frozenset({"append_macro", "append_singleton"})
RA_ROUTES = frozenset(set(ROUTE_IDS) - APPEND_ROUTES)
INSERTION_CAPABLE_ROUTES = frozenset(
    {
        "ra_macro_plateau",
        "ra_macro_always",
        "ra_singleton_plateau",
        "ra_singleton_always",
    }
)
G11_DIAGNOSTIC_ROUTES = frozenset(
    {"append_macro", "ra_macro_append_only"}
)
POOL_AUTHORITY_BY_NPH = {
    3: {
        "parent_count": 123,
        "parent_ordered_labels_sha256": (
            "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
        ),
        "macro_count": 102,
        "macro_ordered_labels_sha256": (
            "a8831528590e870a09ce08492b6f61da4a4d377e63fa8983b30ca9698af5d3d9"
        ),
        "guarded_singleton_count": 948,
        "guarded_singleton_ordered_labels_sha256": (
            "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
        ),
    },
    7: {
        "parent_count": 171,
        "parent_ordered_labels_sha256": (
            "389ce1382b57b916e15e170c641f3884ed1ce33e9913d6eb709f24490739e93f"
        ),
        "macro_count": 148,
        "macro_ordered_labels_sha256": (
            "e6de937476653868f7d3974ad67c467c2f2e2496770e256671b2e807a5b5b03a"
        ),
        "guarded_singleton_count": 6508,
        "guarded_singleton_ordered_labels_sha256": (
            "079478057eea213139dc2f3c7486097496454421a44677c290b5dc55860accb7"
        ),
    },
}

# P3 uses the exact nph=3 strong_weak_u8 bundle authority.  Ordinary facade
# and replay smokes remain two rounds.  G5 is a separate fresh witness with a
# characterized 13-round plateau cap; always-insertion uses a three-round cap.
P3_REGIME_ID = "strong_weak_u8"
P3_NPH = 3
P3_SHORT_ROUNDS = 2
P3_PLATEAU_G5_ROUNDS = 13
P3_ALWAYS_G5_ROUNDS = 3
P3_FIXTURE_ID = "final_stationary_core_strong_weak_u8_nph3_authority_v1"

EXPECTED_ARTIFACT_ROLES = (
    "execution_manifest",
    "checkpoint",
    "estimator_ledger",
    "result",
    "summary",
)
EXPECTED_ARTIFACT_SUFFIXES = {
    "execution_manifest": "execution_manifest.json",
    "checkpoint": "checkpoints/current.json",
    "estimator_ledger": "result/estimator_ledger.json",
    "result": "result/result.json",
    "summary": "summary/summary.json",
}

# These are conservative scheduler requests based on prior full-50 request
# tiers, not claims about observed peak usage.  The repaired RA rows retain
# explicit headroom because full-position scoring can enlarge transient
# candidate populations.  The nph=7 singleton tier exactly retains the
# 90,112 MiB / 98,304 MiB envelope that completed the validated full-50
# strong_strong_u8 singleton run at CHTC cluster 8890777 proc 5.  These
# envelopes do not alter the scientific late-resource-weighting policy.
RESOURCE_ENVELOPES = {
    ("ra", "macro_generator_v1", 3): (4, 49_152, 61_440),
    ("ra", "single_pauli_word_v1", 3): (4, 57_344, 61_440),
    ("ra", "macro_generator_v1", 7): (4, 65_536, 81_920),
    ("ra", "single_pauli_word_v1", 7): (4, 90_112, 98_304),
    ("append", "macro_generator_v1", 3): (1, 32_768, 20_480),
    ("append", "single_pauli_word_v1", 3): (1, 32_768, 20_480),
    ("append", "macro_generator_v1", 7): (1, 65_536, 40_960),
    ("append", "single_pauli_word_v1", 7): (1, 65_536, 40_960),
}

CONTROL_PLANE_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_semantic_preflight.py",
    "validate_package.py",
    "run_cell.py",
    "execute_source_locked_job.sh",
    "build_attempt_selection.py",
    "validate_fetched.py",
    "submit.sub",
)
MUTABLE_RUNTIME_DIRECTORIES: tuple[str, ...] = ()
DECLARED_OVERLAY_FILES = (
    P4_RECEIPT_RELATIVE,
    PACKAGE_PREAUTHORIZATION_RELATIVE,
    SUBMISSION_AUTHORIZATION_RELATIVE,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MACOS_UF_COMPRESSED = 0x00000020
MACOS_SF_DATALESS = 0x40000000


class PackageContractError(ValueError):
    """Raised when immutable authority or package bytes drift."""


def repo_root_from_script(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parents[3]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PackageContractError(f"{label} is not a lowercase SHA-256.")
    return value


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"{label} must be a nonempty path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise PackageContractError(f"{label} is unsafe: {value!r}.")
    return path


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise PackageContractError(f"{label} is unavailable or unsafe: {path}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return payload


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(payload)
    digest = unsigned.pop("sha256", None)
    require_sha256(digest, label=f"{label} self digest")
    actual = canonical_sha256(unsigned)
    if digest != actual:
        raise PackageContractError(
            f"{label} self digest drifted: {digest} != {actual}."
        )
    return str(digest)


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def atomic_publish_noreplace(temporary: Path, destination: Path) -> None:
    """Atomically publish same-filesystem bytes without replacing a race."""

    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise PackageContractError(
            f"Refusing to overwrite raced destination: {destination}"
        ) from exc
    temporary.unlink()
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise PackageContractError(
            f"Refusing to overwrite stale temporary: {temporary}"
        )
    data = canonical_json_bytes(payload) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_publish_noreplace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def core_cell_id(regime_id: str, nph: int, route_id: str) -> str:
    return f"core__{regime_id}__nph{int(nph)}__{route_id}"


def source_lock_id(regime_id: str, nph: int, route_id: str) -> str:
    return f"{regime_id}__nph{int(nph)}__{route_id}"


def representation_for_route(route_id: str) -> str:
    if route_id in MACRO_ROUTES:
        return "macro_generator_v1"
    if route_id in SINGLETON_ROUTES:
        return "single_pauli_word_v1"
    raise PackageContractError(f"Unknown core route: {route_id!r}.")


def direct_execution_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for regime_id, nph in REGIME_CUTOFF_PAIRS:
        for route_id in ROUTE_IDS:
            representation = representation_for_route(route_id)
            method = "append" if route_id in APPEND_ROUTES else "ra"
            cpus, memory_mb, disk_mb = RESOURCE_ENVELOPES[
                (method, representation, nph)
            ]
            cell_id = core_cell_id(regime_id, nph, route_id)
            rows.append(
                {
                    "execution_id": cell_id,
                    "cell_id": cell_id,
                    "source_lock_id": source_lock_id(
                        regime_id, nph, route_id
                    ),
                    "regime_id": regime_id,
                    "nph": nph,
                    "route_id": route_id,
                    "candidate_representation": representation,
                    "execution_entrypoint": (
                        "run_append_adapt"
                        if route_id in APPEND_ROUTES
                        else "run_ra_adapt"
                    ),
                    "resources": {
                        "request_cpus": cpus,
                        "request_memory_mb": memory_mb,
                        "request_disk_mb": disk_mb,
                        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                        "basis": (
                            "prior_full50_requested_tier_with_repaired_"
                            "full_position_headroom_v1"
                        ),
                    },
                    "g11_bounded_replay_diagnostic": {
                        "selected": route_id in G11_DIAGNOSTIC_ROUTES,
                        "run_class": (
                            "bounded_nonpaper_diagnostic_v1"
                            if route_id in G11_DIAGNOSTIC_ROUTES
                            else "not_selected_v1"
                        ),
                        "independent_replay_rounds": (
                            2 if route_id in G11_DIAGNOSTIC_ROUTES else 0
                        ),
                        "ra_resume_prefix_rounds": (
                            1
                            if route_id == "ra_macro_append_only"
                            else 0
                        ),
                        "ra_resumed_rounds": (
                            2
                            if route_id == "ra_macro_append_only"
                            else 0
                        ),
                        "append_resume_boundary": (
                            "authenticated_reconstruction_only_v1"
                            if route_id == "append_macro"
                            else "not_applicable"
                        ),
                        "paper_facing_result_allowed": False,
                    },
                }
            )
    if len(rows) != 48 or len({row["execution_id"] for row in rows}) != 48:
        raise AssertionError("The selected Paper-I core must be exactly 48 rows.")
    return tuple(rows)


def direct_execution_ids() -> tuple[str, ...]:
    return tuple(str(row["execution_id"]) for row in direct_execution_rows())


def expected_artifact_path(cell_id: str, role: str) -> str:
    try:
        suffix = EXPECTED_ARTIFACT_SUFFIXES[role]
    except KeyError as exc:
        raise PackageContractError(f"Unknown artifact role: {role}") from exc
    return f"runs/{cell_id}/{suffix}"


def _regular_file_binding(
    path: Path,
    *,
    repo_root: Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Required regular file is unavailable: {path}")
    flags = int(getattr(path.stat(), "st_flags", 0))
    if flags & (MACOS_UF_COMPRESSED | MACOS_SF_DATALESS):
        raise PackageContractError(f"Source file is compressed/dataless: {path}")
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise PackageContractError(
            f"Authority file escapes the active repository: {path}"
        ) from exc
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise PackageContractError(
            f"Authority file drifted: {relative}: {digest} != {expected_sha256}"
        )
    return {
        "path": relative,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _binding_with_canonical(
    path: Path,
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    row = _regular_file_binding(path, repo_root=repo_root)
    return {
        **row,
        "canonical_sha256": verify_self_digest(
            payload, label=row["path"]
        ),
    }


def _manifest_rows_by_id(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("cells")
    if not isinstance(raw, list):
        raise PackageContractError("Core bundle manifest has no cell list.")
    rows: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise PackageContractError(
                f"Core bundle cell row {index} is not an object."
            )
        cell_id = str(row.get("cell_id", ""))
        if cell_id in rows:
            raise PackageContractError(f"Duplicate core cell: {cell_id}")
        rows[cell_id] = row
    return rows


def _validate_protocol(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    verify_self_digest(payload, label=f"protocol {expected['cell_id']}")
    route_id = str(expected["route_id"])
    expected_schema = (
        APPEND_PROTOCOL_SCHEMA if route_id in APPEND_ROUTES else RA_PROTOCOL_SCHEMA
    )
    problem = payload.get("problem")
    seeds = payload.get("seeds")
    stopping = payload.get("stopping_rule")
    request = payload.get("request")
    request_execution = (
        request.get("execution") if isinstance(request, Mapping) else None
    )
    request_stop = (
        request_execution.get("stop")
        if isinstance(request_execution, Mapping)
        else None
    )
    if (
        payload.get("schema") != expected_schema
        or int(payload.get("horizon", -1)) != 50
        or payload.get("candidate_representation")
        != expected["candidate_representation"]
        or payload.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or payload.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
        or str(payload.get("optimizer", "")).lower() != "powell"
        or int(payload.get("optimizer_maxiter", -1)) != 200
        or not isinstance(problem, Mapping)
        or int(problem.get("n_ph_max", -1)) != int(expected["nph"])
        or str(problem.get("family_key", "")).lower() != "hh"
        or int(problem.get("num_sites", -1)) != 2
        or not isinstance(seeds, Mapping)
        or int(seeds.get("adapt", -1)) != 7
        or int(seeds.get("transpiler", -1)) != 7
        or not isinstance(stopping, Mapping)
        or dict(stopping) != {"maximum_controller_rounds": 50}
        or not isinstance(request_stop, Mapping)
        or dict(request_stop) != {"maximum_controller_rounds": 50}
    ):
        raise PackageContractError(
            f"Core protocol semantic binding drifted: {expected['cell_id']}"
        )
    route = payload.get("route_contract")
    if not isinstance(route, Mapping):
        raise PackageContractError(
            f"Core protocol lacks route contract: {expected['cell_id']}"
        )
    invariants = route.get("semantic_invariants")
    if not isinstance(invariants, Mapping):
        raise PackageContractError(
            f"Core protocol lacks route invariants: {expected['cell_id']}"
        )
    if (
        invariants.get("candidate_representation")
        != expected["candidate_representation"]
        or invariants.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or invariants.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
    ):
        raise PackageContractError(
            f"Core route invariants drifted: {expected['cell_id']}"
        )
    if route_id in APPEND_ROUTES:
        if (
            route.get("route_family") != "append_adapt"
            or invariants.get("accepted_refit_coordinate_chart") != "native_v1"
            or invariants.get("selector_scope")
            != "conventional_append_no_phase3_no_trust_v1"
        ):
            raise PackageContractError(
                f"Conventional unwhitened baseline drifted: {expected['cell_id']}"
            )
    else:
        request = payload.get("request")
        method = request.get("method") if isinstance(request, Mapping) else None
        insertion = method.get("insertion") if isinstance(method, Mapping) else None
        expected_kind = (
            "append_only"
            if route_id.endswith("append_only")
            else (
                "plateau_commutation"
                if route_id.endswith("plateau")
                else "full_commutation"
            )
        )
        if (
            route.get("route_family") != "ra_adapt"
            or not isinstance(insertion, Mapping)
            or insertion.get("kind") != expected_kind
        ):
            raise PackageContractError(
                f"Typed RA insertion policy drifted: {expected['cell_id']}"
            )


def validate_core_authority(
    repo_root: str | Path,
    *,
    materialization_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the exact published stationary core and every consumed byte."""

    root = Path(repo_root).resolve()
    core_root = (
        root / CORE_MATERIALIZATION_RELATIVE_ROOT
        if materialization_root is None
        else Path(materialization_root).resolve()
    )
    expected_core_root = root / CORE_MATERIALIZATION_RELATIVE_ROOT
    if core_root != expected_core_root.resolve():
        raise PackageContractError(
            "Core materialization must use its fixed non-colliding authority path."
        )
    final_path = core_root / CORE_FINAL_RECEIPT_NAME
    final = load_json_object(final_path, label="core final publication receipt")
    final_binding = _binding_with_canonical(
        final_path, final, repo_root=root
    )
    authorization = final.get("authorization")
    matrix = final.get("matrix")
    stationarity = final.get("stationarity_selection")
    if (
        final.get("schema") != CORE_FINAL_RECEIPT_SCHEMA
        or final.get("status") != "passed"
        or final.get("campaign_id") != CAMPAIGN_ID
        or final.get("bundle_id") != CORE_BUNDLE_ID
        or final.get("run_class") != RUN_CLASS
        or final.get("publication_status") != "passed"
        or matrix
        != {
            "cell_count": 48,
            "regime_cutoff_pairs": [
                {"regime_id": regime_id, "nph": nph}
                for regime_id, nph in REGIME_CUTOFF_PAIRS
            ],
            "semantic_route_ids": list(ROUTE_IDS),
            "horizon": 50,
            "direct_execution_cell_count": 48,
        }
        or stationarity
        != {
            "winner_selected": True,
            "active_gradient_policy": "stationary_source_response_v1",
            "authority": {
                "path": USER_SELECTION_AUTHORITY_RELATIVE,
                "sha256": USER_SELECTION_AUTHORITY_FILE_SHA256,
            },
        }
        or authorization
        != {
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
            "explicit_future_user_authorization_required": True,
        }
        or final.get("execution_authorized") is not False
        or final.get("submission_authorized") is not False
        or final.get("submission_state") != "not_submitted"
        or final.get("submitted") is not False
    ):
        raise PackageContractError(
            "Core final receipt is not a passed, unexecuted publication."
        )

    bundle_root = core_root / CORE_BUNDLE_ID
    fixed_files = {
        "bundle_manifest.json": CORE_BUNDLE_MANIFEST_SCHEMA,
        "source_locks.json": CORE_SOURCE_LOCKS_SCHEMA,
        "expected_artifacts.json": CORE_EXPECTED_ARTIFACTS_SCHEMA,
        "validation_report.json": CORE_VALIDATION_SCHEMA,
    }
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for name, schema in fixed_files.items():
        path = bundle_root / name
        payload = load_json_object(path, label=f"core {name}")
        if payload.get("schema") != schema:
            raise PackageContractError(f"Core {name} schema drifted.")
        documents[name] = payload
        bindings[name] = _binding_with_canonical(
            path, payload, repo_root=root
        )

    manifest = documents["bundle_manifest.json"]
    source_locks = documents["source_locks.json"]
    expected_artifacts = documents["expected_artifacts.json"]
    validation = documents["validation_report.json"]
    manifest_rows = _manifest_rows_by_id(manifest)
    expected_rows = {
        str(row["cell_id"]): row for row in direct_execution_rows()
    }
    if (
        manifest.get("bundle_id") != CORE_BUNDLE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or manifest.get("run_class") != RUN_CLASS
        or manifest.get("stationarity_winner_selected") is not True
        or int(manifest.get("cell_count", -1)) != 48
        or set(manifest_rows) != set(expected_rows)
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_state") != "not_submitted"
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Core bundle manifest matrix/state drifted.")

    source_cells = source_locks.get("cell_locks")
    expected_cells = expected_artifacts.get("cells")
    if (
        not isinstance(source_cells, Mapping)
        or set(source_cells)
        != {str(row["source_lock_id"]) for row in expected_rows.values()}
        or int(source_locks.get("required_cell_lock_count", -1)) != 48
        or source_locks.get("all_required_files_verified") is not True
        or not isinstance(expected_cells, Mapping)
        or set(expected_cells) != set(expected_rows)
        or int(expected_artifacts.get("cell_count", -1)) != 48
        or validation.get("materialization_status") != "passed"
        or validation.get("execution_authorized") is not False
        or validation.get("submitted") is not False
    ):
        raise PackageContractError(
            "Core source-lock/expected/validation surface drifted."
        )
    raw_checks = validation.get("checks")
    if not isinstance(raw_checks, list) or any(
        not isinstance(row, Mapping) or row.get("status") != "passed"
        for row in raw_checks
    ):
        raise PackageContractError("Core materialization has a failed check.")
    core_validation_binding = validation.get("core_validation_binding")
    if (
        not isinstance(core_validation_binding, Mapping)
        or int(
            core_validation_binding.get("direct_execution_cell_count", -1)
        )
        != 48
        or core_validation_binding.get("semantic_route_ids")
        != list(ROUTE_IDS)
    ):
        raise PackageContractError(
            "Core validation binding matrix drifted."
        )
    core_validation_binding_sha256 = verify_self_digest(
        core_validation_binding, label="core validation binding"
    )
    campaign_authorities = source_locks.get("campaign_authorities")
    stationarity_authority = (
        campaign_authorities.get("stationarity_selection")
        if isinstance(campaign_authorities, Mapping)
        else None
    )
    if (
        not isinstance(stationarity_authority, Mapping)
        or stationarity_authority.get("path")
        != USER_SELECTION_AUTHORITY_RELATIVE
        or stationarity_authority.get("sha256")
        != USER_SELECTION_AUTHORITY_FILE_SHA256
        or stationarity_authority.get("verified") is not True
    ):
        raise PackageContractError(
            "Core source locks do not bind the selected-policy authority."
        )
    raw_global_sources = source_locks.get("global_sources")
    if not isinstance(raw_global_sources, Mapping) or not raw_global_sources:
        raise PackageContractError(
            "Core source locks have no global-source byte authority."
        )
    global_source_files: dict[str, dict[str, Any]] = {}
    for source_id, raw_source in sorted(raw_global_sources.items()):
        if (
            not isinstance(raw_source, Mapping)
            or raw_source.get("verified") is not True
        ):
            raise PackageContractError(
                f"Core global source is not verified: {source_id}."
            )
        relative = safe_relative_path(
            raw_source.get("path"),
            label=f"global source {source_id}",
        ).as_posix()
        global_source_files[str(source_id)] = _regular_file_binding(
            root / relative,
            repo_root=root,
            expected_sha256=require_sha256(
                raw_source.get("sha256"),
                label=f"global source {source_id}",
            ),
        )

    protocol_dir = bundle_root / "protocols"
    template_dir = bundle_root / "execution_templates"
    protocol_paths = sorted(protocol_dir.glob("*.json"))
    template_paths = sorted(template_dir.glob("*.json"))
    if (
        {path.stem for path in protocol_paths} != set(expected_rows)
        or {path.stem for path in template_paths} != set(expected_rows)
        or len(protocol_paths) != 48
        or len(template_paths) != 48
    ):
        raise PackageContractError(
            "Core protocol/template file set is not exactly 48 + 48."
        )

    protocol_bindings: dict[str, dict[str, Any]] = {}
    template_bindings: dict[str, dict[str, Any]] = {}
    for cell_id, row in expected_rows.items():
        manifest_row = manifest_rows[cell_id]
        if (
            manifest_row.get("stage") != "core"
            or manifest_row.get("regime_id") != row["regime_id"]
            or int(manifest_row.get("nph", -1)) != row["nph"]
            or manifest_row.get("route_id") != row["route_id"]
            or manifest_row.get("candidate_representation")
            != row["candidate_representation"]
            or int(manifest_row.get("horizon", -1)) != 50
        ):
            raise PackageContractError(f"Core cell row drifted: {cell_id}")
        protocol_path = protocol_dir / f"{cell_id}.json"
        protocol = load_json_object(
            protocol_path, label=f"core protocol {cell_id}"
        )
        _validate_protocol(protocol, expected=row)
        protocol_bindings[cell_id] = _binding_with_canonical(
            protocol_path, protocol, repo_root=root
        )
        template_path = template_dir / f"{cell_id}.json"
        template = load_json_object(
            template_path, label=f"core template {cell_id}"
        )
        protocol_pointer = template.get("protocol")
        fulfillment = template.get("execution_fulfillment")
        expected_entrypoint = (
            "pipelines.static_adapt.ra_adapt.run_append_adapt"
            if row["route_id"] in APPEND_ROUTES
            else "pipelines.static_adapt.ra_adapt.run_ra_adapt"
        )
        if (
            template.get("schema") != EXECUTION_TEMPLATE_SCHEMA
            or template.get("cell_id") != cell_id
            or template.get("study_id") != CAMPAIGN_ID
            or template.get("campaign_id") != CAMPAIGN_ID
            or template.get("run_class") != RUN_CLASS
            or template.get("execution_entrypoint") != expected_entrypoint
            or template.get("execution_authorized") is not False
            or template.get("submission_state") != "not_submitted"
            or template.get("submitted") is not False
            or template.get("seeds") != {"adapt": 7, "transpiler": 7}
            or not isinstance(protocol_pointer, Mapping)
            or protocol_pointer.get("path") != f"protocols/{cell_id}.json"
            or protocol_pointer.get("sha256") != protocol["sha256"]
            or fulfillment
            != {
                "fulfillment_kind": "direct_execution_v1",
                "canonical_execution": {
                    "bundle_id": CORE_BUNDLE_ID,
                    "cell_id": cell_id,
                },
            }
        ):
            raise PackageContractError(
                f"Core execution template drifted: {cell_id}"
            )
        template_bindings[cell_id] = _binding_with_canonical(
            template_path, template, repo_root=root
        )

    exact_bundle_files = {
        *(bundle_root / name for name in fixed_files),
        *protocol_paths,
        *template_paths,
    }
    observed_bundle_files: set[Path] = set()
    observed_bundle_directories: set[Path] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise PackageContractError(
                f"Core bundle contains a forbidden symlink: {path}"
            )
        if path.is_file():
            observed_bundle_files.add(path)
        elif path.is_dir():
            observed_bundle_directories.add(path)
        else:
            raise PackageContractError(
                f"Core bundle contains a non-regular entry: {path}"
            )
    if observed_bundle_files != exact_bundle_files:
        extras = sorted(str(path) for path in observed_bundle_files-exact_bundle_files)
        missing = sorted(str(path) for path in exact_bundle_files-observed_bundle_files)
        raise PackageContractError(
            f"Core bundle recursive allowlist drifted: extras={extras}, "
            f"missing={missing}."
        )
    if observed_bundle_directories != {protocol_dir, template_dir}:
        raise PackageContractError(
            "Core bundle directory allowlist drifted."
        )

    implementation = source_locks.get("implementation_sources")
    final_implementation = final.get("implementation_source_inventory")
    if (
        not isinstance(implementation, Mapping)
        or implementation != final_implementation
    ):
        raise PackageContractError(
            "Core implementation inventory/final receipt binding drifted."
        )
    implementation_sha256 = verify_self_digest(
        implementation, label="core implementation inventory"
    )
    expected_final_bundle_receipt = {
        "bundle_id": CORE_BUNDLE_ID,
        "status": "passed",
        "cell_count": 48,
        "direct_execution_cell_count": 48,
        "semantic_route_ids": list(ROUTE_IDS),
        "manifest": {
            "path": f"{CORE_BUNDLE_ID}/bundle_manifest.json",
            "canonical_sha256": bindings["bundle_manifest.json"][
                "canonical_sha256"
            ],
            "file_sha256": bindings["bundle_manifest.json"]["sha256"],
        },
        "source_locks": {
            "path": f"{CORE_BUNDLE_ID}/source_locks.json",
            "canonical_sha256": bindings["source_locks.json"][
                "canonical_sha256"
            ],
            "file_sha256": bindings["source_locks.json"]["sha256"],
        },
        "expected_artifacts": {
            "path": f"{CORE_BUNDLE_ID}/expected_artifacts.json",
            "canonical_sha256": bindings["expected_artifacts.json"][
                "canonical_sha256"
            ],
            "file_sha256": bindings["expected_artifacts.json"]["sha256"],
        },
        "validation": {
            "path": f"{CORE_BUNDLE_ID}/validation_report.json",
            "canonical_sha256": bindings["validation_report.json"][
                "canonical_sha256"
            ],
            "file_sha256": bindings["validation_report.json"]["sha256"],
        },
        "core_validation_binding_sha256": (
            core_validation_binding_sha256
        ),
        "implementation_source_inventory_sha256": implementation_sha256,
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_state": "not_submitted",
        "submitted": False,
    }
    final_validation = final.get("validation")
    if (
        final.get("bundle_receipt") != expected_final_bundle_receipt
        or not isinstance(final_validation, Mapping)
        or final_validation.get("core_validation_binding_sha256")
        != core_validation_binding_sha256
        or final.get("implementation_source_inventory_binding")
        != {
            "source_pointer": (
                f"{CORE_BUNDLE_ID}/source_locks.json"
                "#/implementation_sources"
            ),
            "stable": True,
        }
    ):
        raise PackageContractError(
            "Core final publication receipt does not exactly bind the "
            "validated bundle and implementation inventory."
        )
    raw_files = implementation.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise PackageContractError("Core implementation inventory is empty.")
    source_files: list[dict[str, Any]] = []
    observed_source_paths: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"Implementation inventory row {index} is invalid."
            )
        relative = safe_relative_path(
            raw.get("path"), label=f"implementation row {index}"
        ).as_posix()
        if relative in observed_source_paths:
            raise PackageContractError(
                f"Duplicate implementation source: {relative}"
            )
        observed_source_paths.add(relative)
        source_files.append(
            _regular_file_binding(
                root / relative,
                repo_root=root,
                expected_sha256=require_sha256(
                    raw.get("sha256"),
                    label=f"implementation source {relative}",
                ),
            )
        )
    if int(implementation.get("file_count", -1)) != len(source_files):
        raise PackageContractError("Implementation inventory count drifted.")

    bundle_files = [
        _regular_file_binding(path, repo_root=root)
        for path in sorted(exact_bundle_files)
    ]
    return {
        "core_root": core_root.as_posix(),
        "bundle_root": bundle_root.as_posix(),
        "final_receipt": final,
        "final_receipt_binding": final_binding,
        "document_bindings": bindings,
        "protocol_bindings": protocol_bindings,
        "template_bindings": template_bindings,
        "implementation_inventory": dict(implementation),
        "implementation_inventory_sha256": implementation_sha256,
        "source_files": source_files,
        "global_source_files": global_source_files,
        "bundle_files": bundle_files,
        "source_lock_cells": {
            str(key): dict(value)
            for key, value in source_cells.items()
            if isinstance(value, Mapping)
        },
        "global_source_locks": {
            str(key): dict(value)
            for key, value in (
                source_locks.get("global_sources", {})
                if isinstance(
                    source_locks.get("global_sources"), Mapping
                )
                else {}
            ).items()
            if isinstance(value, Mapping)
        },
        "cell_rows": list(direct_execution_rows()),
    }


def validate_user_selection_authority(
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = root / USER_SELECTION_AUTHORITY_RELATIVE
    payload = load_json_object(path, label="stationary-core user decision")
    binding = _regular_file_binding(
        path,
        repo_root=root,
        expected_sha256=USER_SELECTION_AUTHORITY_FILE_SHA256,
    )
    decision = payload.get("decision")
    if (
        payload.get("schema") != "ra_adapt_icm_stage_receipt_v1"
        or payload.get("stage") != "user-review"
        or payload.get("state") != "complete"
        or not isinstance(decision, Mapping)
        or decision.get("core_campaign_id") != CAMPAIGN_ID
        or decision.get("core_bundle_id") != CORE_BUNDLE_ID
        or int(decision.get("core_direct_cell_count", -1)) != 48
        or int(decision.get("core_horizon", -1)) != 50
        or decision.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or decision.get("phase_i_resource_weighting_selection")
        != "late_resource_weighting_v1"
        or decision.get("execution_authorized") is not False
        or decision.get("submission_authorized") is not False
        or decision.get("study1_disposition")
        != "canceled_unsubmitted_superseded_by_explicit_user_selection"
    ):
        raise PackageContractError(
            "Stationary-core user-selection authority drifted."
        )
    return {"payload": payload, "binding": binding}


def control_plane_receipt(package_dir: str | Path) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    rows = []
    for relative in CONTROL_PLANE_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Control-plane member is unavailable: {path}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & 0o111),
            }
        )
    return digested(
        {
            "schema": CONTROL_PLANE_SCHEMA,
            "package_id": PACKAGE_ID,
            "files": rows,
            "file_count": len(rows),
            "all_files_verified": True,
        }
    )


def validate_p3_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
    authority: Mapping[str, Any],
    control_plane: Mapping[str, Any],
) -> dict[str, Any]:
    digest = verify_self_digest(receipt, label="P3 semantic preflight receipt")
    final_binding = authority["final_receipt_binding"]
    required = {
        "schema": P3_RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "core_final_receipt_canonical_sha256": final_binding[
            "canonical_sha256"
        ],
        "core_final_receipt_file_sha256": final_binding["sha256"],
        "implementation_source_inventory_sha256": authority[
            "implementation_inventory_sha256"
        ],
        "active_gradient_policy": "stationary_source_response_v1",
        "resource_weighting_scope": "late_resource_weighting_v1",
        "status": "passed",
        "p3_passed": True,
        "execution_mode": "bounded_non_paper_semantic_preflight_v1",
        "full_horizon_executed": False,
        "paper_facing_result_allowed": False,
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_state": "not_submitted",
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise PackageContractError(f"P3 receipt drifted at {field}.")
    if receipt.get("governing_plan_p3_alignment") != {
        "regime_id": P3_REGIME_ID,
        "nph": P3_NPH,
        "ordinary_smoke_controller_rounds": P3_SHORT_ROUNDS,
        "route_coverage": "all_eight_selected_routes_v1",
        "ra_protocol_authority": (
            "exact_final_stationary_core_protocol_v1"
        ),
        "append_protocol_authority": (
            "exact_final_problem_and_source_authority_bounded_v1"
        ),
        "g5_execution_boundary": (
            "separate_independent_fresh_witness_v1"
        ),
        "plateau_g5_round_cap": P3_PLATEAU_G5_ROUNDS,
        "always_g5_round_cap": P3_ALWAYS_G5_ROUNDS,
    }:
        raise PackageContractError(
            "P3 governing-plan alignment receipt drifted."
        )
    coverage = receipt.get("semantic_coverage")
    if not isinstance(coverage, Mapping):
        raise PackageContractError("P3 receipt has no semantic coverage.")
    if (
        set(coverage.get("route_families", ())) != set(ROUTE_IDS)
        or set(coverage.get("candidate_representations", ()))
        != {"macro_generator_v1", "single_pauli_word_v1"}
        or int(coverage.get("pool_construction_regime_count", -1))
        != 6
        or set(coverage.get("cutoff_pool_coverage", ())) != {3, 7}
        or set(coverage.get("ra_fresh_resume_replay_routes", ())) != RA_ROUTES
        or set(coverage.get("append_fresh_reconstruction_routes", ()))
        != APPEND_ROUTES
        or set(coverage.get("nonvacuous_g5_routes", ()))
        != INSERTION_CAPABLE_ROUTES
    ):
        raise PackageContractError(
            "P3 receipt does not cover all selected semantic branches."
        )
    control_rows = control_plane.get("files")
    if not isinstance(control_rows, list):
        raise PackageContractError("Control-plane receipt has no files.")
    generator_rows = [
        row
        for row in control_rows
        if isinstance(row, Mapping)
        and row.get("path") == "run_semantic_preflight.py"
    ]
    generator = receipt.get("generator")
    if (
        len(generator_rows) != 1
        or not isinstance(generator, Mapping)
        or generator
        != {
            "path": "run_semantic_preflight.py",
            "sha256": generator_rows[0]["sha256"],
            "size_bytes": generator_rows[0]["size_bytes"],
        }
    ):
        raise PackageContractError(
            "P3 receipt does not bind its packaged generator source."
        )
    pool_proof = receipt.get("p2_pool_construction_proof")
    if not isinstance(pool_proof, Mapping):
        raise PackageContractError(
            "P3 has no six-regime pool/construction proof."
        )
    pool_proof_sha = verify_self_digest(
        pool_proof, label="P3 six-regime pool/construction proof"
    )
    proof_rows = pool_proof.get("rows")
    if (
        receipt.get("p2_pool_construction_proof_sha256")
        != pool_proof_sha
        or pool_proof.get("schema")
        != (
            "paper_i_stationary_core_six_regime_"
            "pool_construction_proof_v1"
        )
        or int(pool_proof.get("regime_count", -1)) != 6
        or pool_proof.get("regime_cutoff_pairs")
        != [
            [regime_id, int(nph)]
            for regime_id, nph in REGIME_CUTOFF_PAIRS
        ]
        or pool_proof.get("macro_ra_append_equality_all_regimes")
        is not True
        or pool_proof.get(
            "singleton_construction_equivalence_all_regimes"
        )
        is not True
        or pool_proof.get("status") != "passed"
        or not isinstance(proof_rows, list)
        or len(proof_rows) != 6
    ):
        raise PackageContractError(
            "P3 six-regime pool/construction proof drifted."
        )
    for index, ((regime_id, nph), raw) in enumerate(
        zip(REGIME_CUTOFF_PAIRS, proof_rows, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"P3 pool/construction row {index} is invalid."
            )
        verify_self_digest(
            raw, label=f"P3 {regime_id} pool/construction row"
        )
        expected_pool = POOL_AUTHORITY_BY_NPH[int(nph)]
        parent = raw.get("parent_inventory")
        macro = raw.get("macro_coefficient_pool")
        singleton_parent = raw.get("singleton_parent_inventory")
        global_children = raw.get("singleton_append_global_pool")
        construction = raw.get(
            "singleton_construction_equivalence"
        )
        if not all(
            isinstance(value, Mapping)
            for value in (
                parent,
                macro,
                singleton_parent,
                global_children,
                construction,
            )
        ):
            raise PackageContractError(
                f"P3 {regime_id} pool projections are invalid."
            )
        verify_self_digest(
            construction,
            label=f"P3 {regime_id} singleton construction",
        )
        expected_protocols = {
            "ra_macro": authority["protocol_bindings"][
                core_cell_id(
                    regime_id, nph, "ra_macro_append_only"
                )
            ]["canonical_sha256"],
            "append_macro": authority["protocol_bindings"][
                core_cell_id(regime_id, nph, "append_macro")
            ]["canonical_sha256"],
            "ra_singleton": authority["protocol_bindings"][
                core_cell_id(
                    regime_id, nph, "ra_singleton_append_only"
                )
            ]["canonical_sha256"],
            "append_singleton": authority["protocol_bindings"][
                core_cell_id(
                    regime_id, nph, "append_singleton"
                )
            ]["canonical_sha256"],
        }
        protocol_payloads = {
            name: load_json_object(
                Path(str(authority["bundle_root"]))
                / "protocols"
                / f"{cell_id}.json",
                label=f"P3 {regime_id} {name} protocol",
            )
            for name, cell_id in {
                "ra_macro": core_cell_id(
                    regime_id, nph, "ra_macro_append_only"
                ),
                "append_macro": core_cell_id(
                    regime_id, nph, "append_macro"
                ),
                "ra_singleton": core_cell_id(
                    regime_id, nph, "ra_singleton_append_only"
                ),
                "append_singleton": core_cell_id(
                    regime_id, nph, "append_singleton"
                ),
            }.items()
        }
        macro_parent_protocol = protocol_payloads["ra_macro"].get(
            "parent_inventory"
        )
        macro_pool_protocol = protocol_payloads["ra_macro"].get(
            "executable_pool"
        )
        singleton_parent_protocol = protocol_payloads[
            "ra_singleton"
        ].get("parent_inventory")
        singleton_append_pool_protocol = protocol_payloads[
            "append_singleton"
        ].get("executable_pool")
        protocol_problem = protocol_payloads["ra_macro"].get(
            "problem"
        )
        if (
            raw.get("schema")
            != (
                "paper_i_stationary_core_regime_"
                "pool_construction_proof_v1"
            )
            or raw.get("regime_id") != regime_id
            or int(raw.get("nph", -1)) != int(nph)
            or raw.get("protocol_sha256s") != expected_protocols
            or raw.get("ra_append_macro_pool_equal") is not True
            or raw.get("ra_append_singleton_parent_equal") is not True
            or raw.get("status") != "passed"
            or not isinstance(protocol_problem, Mapping)
            or any(
                payload.get("problem") != protocol_problem
                for payload in protocol_payloads.values()
            )
            or raw.get("problem_receipt_sha256")
            != canonical_sha256(protocol_problem)
            or not isinstance(macro_parent_protocol, Mapping)
            or not isinstance(macro_pool_protocol, Mapping)
            or not isinstance(singleton_parent_protocol, Mapping)
            or not isinstance(singleton_append_pool_protocol, Mapping)
            or protocol_payloads["append_macro"].get(
                "parent_inventory"
            )
            != macro_parent_protocol
            or protocol_payloads["append_macro"].get(
                "executable_pool"
            )
            != macro_pool_protocol
            or protocol_payloads["append_singleton"].get(
                "parent_inventory"
            )
            != singleton_parent_protocol
            or protocol_payloads["ra_singleton"].get(
                "executable_pool"
            )
            != singleton_parent_protocol
            or int(parent.get("count", -1))
            != expected_pool["parent_count"]
            or parent.get("ordered_labels_sha256")
            != expected_pool["parent_ordered_labels_sha256"]
            or parent.get("ordered_pool_sha256")
            != macro_parent_protocol.get("ordered_pool_sha256")
            or int(macro.get("count", -1))
            != expected_pool["macro_count"]
            or macro.get("ordered_labels_sha256")
            != expected_pool["macro_ordered_labels_sha256"]
            or macro.get("ordered_pool_sha256")
            != macro_pool_protocol.get("ordered_pool_sha256")
            or int(singleton_parent.get("count", -1))
            != expected_pool["parent_count"]
            or singleton_parent.get("ordered_labels_sha256")
            != expected_pool["parent_ordered_labels_sha256"]
            or singleton_parent.get("ordered_pool_sha256")
            != singleton_parent_protocol.get("ordered_pool_sha256")
            or singleton_parent.get("ordered_pool_sha256")
            != parent.get("ordered_pool_sha256")
            or int(global_children.get("count", -1))
            != expected_pool["guarded_singleton_count"]
            or global_children.get("ordered_labels_sha256")
            != expected_pool[
                "guarded_singleton_ordered_labels_sha256"
            ]
            or global_children.get("ordered_pool_sha256")
            != singleton_append_pool_protocol.get(
                "ordered_pool_sha256"
            )
            or construction.get("regime_id") != regime_id
            or int(construction.get("nph", -1)) != int(nph)
            or construction.get(
                "construction_equivalent_for_identical_parent_supply"
            )
            is not True
            or construction.get("canonical_unit_pauli_representatives")
            is not True
            or construction.get("hard_guarded") is not True
            or construction.get("status") != "passed"
            or construction.get("append_global_child_pool")
            != global_children
            or construction.get("ra_staged_child_pool", {}).get(
                "count"
            )
            != expected_pool["guarded_singleton_count"]
            or construction.get("ra_staged_child_pool", {}).get(
                "ordered_labels_sha256"
            )
            != expected_pool[
                "guarded_singleton_ordered_labels_sha256"
            ]
            or construction.get("ra_staged_child_pool", {}).get(
                "ordered_pool_sha256"
            )
            != global_children.get("ordered_pool_sha256")
        ):
            raise PackageContractError(
                f"P3 {regime_id} pool/construction proof failed."
            )
        require_sha256(
            raw.get("problem_receipt_sha256"),
            label=f"P3 {regime_id} problem receipt",
        )
        require_sha256(
            construction.get("ordered_child_manifest_sha256"),
            label=f"P3 {regime_id} child manifest",
        )

    raw_routes = receipt.get("route_observations")
    if not isinstance(raw_routes, list) or len(raw_routes) != len(ROUTE_IDS):
        raise PackageContractError("P3 route-observation count drifted.")
    route_rows: dict[str, Mapping[str, Any]] = {}
    for raw in raw_routes:
        if not isinstance(raw, Mapping):
            raise PackageContractError("P3 route observation is invalid.")
        route_id = str(raw.get("route_id", ""))
        if route_id in route_rows or route_id not in ROUTE_IDS:
            raise PackageContractError(
                f"P3 route observation is duplicated/unknown: {route_id}"
            )
        final_cell_id = core_cell_id(P3_REGIME_ID, P3_NPH, route_id)
        final_protocol = load_json_object(
            Path(str(authority["bundle_root"]))
            / "protocols"
            / f"{final_cell_id}.json",
            label=f"P3 {route_id} final protocol authority",
        )
        expected_protocol_mode = (
            "final_bundle_problem_and_source_authority_bounded_protocol_v1"
            if route_id in APPEND_ROUTES
            else "exact_final_bundle_protocol_with_operational_round_cap_v1"
        )
        expected_maximum_rounds = (
            P3_PLATEAU_G5_ROUNDS
            if route_id.endswith("plateau")
            else P3_ALWAYS_G5_ROUNDS
            if route_id.endswith("always")
            else P3_SHORT_ROUNDS
        )
        if (
            raw.get("status") != "passed"
            or raw.get("candidate_representation")
            != representation_for_route(route_id)
            or int(raw.get("fixture_nph", -1)) != P3_NPH
            or raw.get("fixture_identity") != P3_FIXTURE_ID
            or raw.get("fixture_regime_id") != P3_REGIME_ID
            or raw.get("fixture_problem_receipt")
            != final_protocol.get("problem")
            or raw.get("bounded_protocol_mode")
            != expected_protocol_mode
            or int(raw.get("ordinary_smoke_controller_rounds", -1))
            != P3_SHORT_ROUNDS
            or int(raw.get("final_protocol_nph", -1)) != P3_NPH
            or raw.get("final_protocol_cell_id") != final_cell_id
            or raw.get("protocol_sha256")
            != authority["protocol_bindings"][final_cell_id][
                "canonical_sha256"
            ]
            or raw.get("run_class") != "smoke"
            or raw.get("paper_facing_result_allowed") is not False
            or int(raw.get("maximum_controller_rounds_executed", -1))
            != expected_maximum_rounds
            or int(raw.get("maximum_controller_rounds_executed", 50)) >= 50
        ):
            raise PackageContractError(
                f"P3 bounded route observation drifted: {route_id}"
            )
        require_sha256(
            raw.get("protocol_sha256"),
            label=f"P3 {route_id} final protocol",
        )
        require_sha256(
            raw.get("fixture_protocol_sha256"),
            label=f"P3 {route_id} fixture protocol",
        )
        require_sha256(
            raw.get("fixture_construction_sha256"),
            label=f"P3 {route_id} fixture construction",
        )
        invocations = raw.get("facade_invocations")
        fresh = raw.get("fresh_execution")
        replay = raw.get("independent_replay")
        expected_entrypoint = (
            "run_append_adapt"
            if route_id in APPEND_ROUTES
            else "run_ra_adapt"
        )
        if (
            not isinstance(invocations, list)
            or not invocations
            or any(
                not isinstance(call, Mapping)
                or call.get("entrypoint") != expected_entrypoint
                or not isinstance(call.get("purpose"), str)
                or not call["purpose"]
                or int(call.get("maximum_controller_rounds", 50)) >= 50
                for call in invocations
            )
            or not isinstance(fresh, Mapping)
            or fresh.get("status") != "passed"
            or not isinstance(replay, Mapping)
            or replay.get("status") != "passed"
            or replay.get("matched") is not True
        ):
            raise PackageContractError(
                f"P3 facade evidence is incomplete: {route_id}"
            )
        purposes = {
            str(call["purpose"])
            for call in invocations
            if isinstance(call, Mapping)
        }
        required_purposes = (
            {"fresh_execution", "independent_reconstruction"}
            if route_id in APPEND_ROUTES
            else {
                "independent_primary",
                "fresh_resume_prefix",
                "authenticated_resume",
            }
        )
        if not required_purposes.issubset(purposes):
            raise PackageContractError(
                f"P3 route invocations are incomplete: {route_id}"
            )
        invocation_by_purpose = {
            str(call["purpose"]): call
            for call in invocations
            if isinstance(call, Mapping)
        }
        expected_rounds_by_purpose = (
            {
                "fresh_execution": P3_SHORT_ROUNDS,
                "independent_reconstruction": P3_SHORT_ROUNDS,
            }
            if route_id in APPEND_ROUTES
            else {
                "independent_primary": P3_SHORT_ROUNDS,
                "fresh_resume_prefix": 1,
                "authenticated_resume": P3_SHORT_ROUNDS,
                **(
                    {
                        "g5_scored_position_witness": (
                            P3_PLATEAU_G5_ROUNDS
                            if route_id.endswith("plateau")
                            else P3_ALWAYS_G5_ROUNDS
                        )
                    }
                    if route_id in INSERTION_CAPABLE_ROUTES
                    else {}
                ),
            }
        )
        if (
            set(invocation_by_purpose) != set(expected_rounds_by_purpose)
            or any(
                int(
                    invocation_by_purpose[purpose].get(
                        "maximum_controller_rounds", -1
                    )
                )
                != rounds
                for purpose, rounds in expected_rounds_by_purpose.items()
            )
        ):
            raise PackageContractError(
                f"P3 route round caps drifted: {route_id}"
            )
        for label, row in (("fresh", fresh), ("replay", replay)):
            for field in ("result_sha256", "trajectory_sha256"):
                require_sha256(
                    row.get(field),
                    label=f"P3 {route_id} {label} {field}",
                )
        if route_id in RA_ROUTES:
            resume = raw.get("authenticated_resume")
            if (
                not isinstance(resume, Mapping)
                or resume.get("status") != "passed"
                or resume.get("authenticated") is not True
                or resume.get("trajectory_prefix_matched") is not True
            ):
                raise PackageContractError(
                    f"P3 authenticated resume is incomplete: {route_id}"
                )
            for field in (
                "checkpoint_file_sha256",
                "resumed_result_sha256",
                "comparison_sha256",
            ):
                require_sha256(
                    resume.get(field),
                    label=f"P3 {route_id} resume {field}",
                )
        else:
            boundary = raw.get("reconstruction_boundary")
            if (
                not isinstance(boundary, Mapping)
                or boundary.get("status")
                != "authenticated_reconstruction_only_verified"
                or boundary.get("public_resume_execution_supported") is not False
                or boundary.get("reconstruction_fields_complete") is not True
            ):
                raise PackageContractError(
                    f"P3 Append reconstruction boundary drifted: {route_id}"
                )
        if route_id in INSERTION_CAPABLE_ROUTES:
            if "g5_scored_position_witness" not in purposes:
                raise PackageContractError(
                    f"P3 G5 invocation is absent: {route_id}"
                )
            witness = raw.get("g5_scored_position_witness")
            witness_cap = (
                P3_PLATEAU_G5_ROUNDS
                if route_id.endswith("plateau")
                else P3_ALWAYS_G5_ROUNDS
            )
            first_interior = (
                int(witness.get("first_interior_controller_round", -1))
                if isinstance(witness, Mapping)
                and witness.get("first_interior_controller_round")
                is not None
                else -1
            )
            shared_witness_invalid = (
                not isinstance(witness, Mapping)
                or witness.get("status") != "passed"
                or witness.get("aggregate_g5_passed") is not True
                or witness.get("execution_mode")
                != "independent_fresh_exact_final_nph3_protocol_v1"
                or witness.get("trajectory_prefix_matched") is not True
                or int(
                    witness.get(
                        "authenticated_prefix_controller_rounds", -1
                    )
                )
                != P3_SHORT_ROUNDS
                or int(witness.get("witness_controller_rounds", -1))
                != witness_cap
                or int(witness.get("scored_position_count", 0)) <= 0
            )
            interior_count = (
                int(witness.get("interior_scored_count", -1))
                if isinstance(witness, Mapping)
                else -1
            )
            if route_id.endswith("plateau"):
                route_witness_invalid = (
                    first_interior < 2
                    or first_interior > witness_cap
                    or interior_count <= 0
                    or witness.get("interior_witness_status")
                    != "observed"
                )
            else:
                route_witness_invalid = (
                    witness.get("full_insertion_policy_verified")
                    is not True
                    or (
                        interior_count > 0
                        and (
                            first_interior < 2
                            or first_interior > witness_cap
                            or witness.get("interior_witness_status")
                            != "observed"
                        )
                    )
                    or (
                        interior_count == 0
                        and (
                            first_interior != -1
                            or witness.get("interior_witness_status")
                            != (
                                "not_serialized_by_v9_selected_phase_"
                                "population_projection"
                            )
                            or witness.get("limitation")
                            != (
                                "immutable_v9_full_insertion_population_"
                                "receipt_retains_selected_phase_records_"
                                "but_not_the_exhaustive_domain_population"
                            )
                        )
                    )
                    or interior_count < 0
                )
            if shared_witness_invalid or route_witness_invalid:
                raise PackageContractError(
                    f"P3 nonvacuous G5 witness drifted: {route_id}"
                )
            require_sha256(
                witness.get("population_receipt_sha256"),
                label=f"P3 {route_id} G5 population receipt",
            )
        route_rows[route_id] = raw
    if set(route_rows) != set(ROUTE_IDS):
        raise PackageContractError("P3 does not observe all eight routes.")
    return {
        "path": P3_RECEIPT_RELATIVE,
        "canonical_sha256": digest,
        "file_sha256": require_sha256(
            receipt_file_sha256, label="P3 receipt file SHA-256"
        ),
    }


def validate_submission_authorization(
    receipt: Mapping[str, Any],
    *,
    package_manifest: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    p4_receipt: Mapping[str, Any],
) -> str:
    digest = verify_self_digest(
        receipt, label="submission authorization receipt"
    )
    required = {
        "schema": SUBMISSION_AUTHORIZATION_SCHEMA,
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "execution_target": EXECUTION_TARGET,
        "package_manifest_sha256": package_manifest["sha256"],
        "execution_plan_sha256": execution_plan["sha256"],
        "source_archive_sha256": execution_plan["source_archive"]["sha256"],
        "p4_receipt_sha256": p4_receipt["sha256"],
        "authorized_execution_ids": list(direct_execution_ids()),
        "direct_execution_count": 48,
        "batch_name": BATCH_NAME,
        "remote_image_path": REMOTE_IMAGE_PATH,
        "remote_image_sha256": REMOTE_IMAGE_SHA256,
        "remote_image_byte_verification_passed": True,
        "remote_image_verification_scope": (
            "authorized_remote_pre_submit_exact_bytes_v1"
        ),
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise PackageContractError(
                f"Submission authorization drifted at {field}."
            )
    for field in ("authorization_id", "authorized_utc"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise PackageContractError(
                f"Submission authorization has no {field}."
            )
    return digest


def verify_exact_set(
    observed: Iterable[str],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    observed_set = set(observed)
    expected_set = set(expected)
    if observed_set != expected_set:
        raise PackageContractError(
            f"{label} drifted: missing={sorted(expected_set-observed_set)}, "
            f"extra={sorted(observed_set-expected_set)}."
        )


__all__ = [
    "APPEND_ROUTES",
    "BATCH_NAME",
    "CAMPAIGN_ID",
    "CONTROL_PLANE_FILES",
    "CONTROL_PLANE_SCHEMA",
    "CORE_BUNDLE_ID",
    "CORE_FINAL_COPY_RELATIVE",
    "CORE_FINAL_RECEIPT_NAME",
    "CORE_MATERIALIZATION_RELATIVE_ROOT",
    "DECLARED_OVERLAY_FILES",
    "ED_REGIME_NAME_BY_ID",
    "EXECUTION_PLAN_SCHEMA",
    "EXECUTION_TARGET",
    "EXPECTED_ARTIFACT_ROLES",
    "FETCH_VALIDATION_SCHEMA",
    "FULL_RUN_SCIENTIFIC_CLOSURE_SCHEMA",
    "G11_BOUNDED_REPLAY_DIAGNOSTIC_SCHEMA",
    "G11_DIAGNOSTIC_ROUTES",
    "INSERTION_CAPABLE_ROUTES",
    "JOB_SPEC_SCHEMA",
    "MAX_RUNTIME_SECONDS",
    "MUTABLE_RUNTIME_DIRECTORIES",
    "P2_RECEIPT_RELATIVE",
    "P2_RECEIPT_SCHEMA",
    "P3_RECEIPT_RELATIVE",
    "P3_RECEIPT_SCHEMA",
    "P3_ALWAYS_G5_ROUNDS",
    "P3_FIXTURE_ID",
    "P3_NPH",
    "P3_PLATEAU_G5_ROUNDS",
    "P3_REGIME_ID",
    "P3_SHORT_ROUNDS",
    "P4_RECEIPT_RELATIVE",
    "P4_RECEIPT_SCHEMA",
    "P4_SMOKE_RESULT_SCHEMA",
    "P4_SMOKE_SPEC_SCHEMA",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_SCHEMA",
    "PACKAGE_PREAUTHORIZATION_RELATIVE",
    "PACKAGE_PREAUTHORIZATION_SCHEMA",
    "PACKAGE_RELATIVE_ROOT",
    "PackageContractError",
    "POOL_AUTHORITY_BY_NPH",
    "RA_ROUTES",
    "REGIME_CUTOFF_PAIRS",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "ROUTE_IDS",
    "RUN_CLASS",
    "RUNTIME_RELATIVE_ROOT",
    "SOURCE_ARCHIVE_MANIFEST_SCHEMA",
    "SUBMISSION_AUTHORIZATION_RELATIVE",
    "SUBMISSION_AUTHORIZATION_SCHEMA",
    "USER_SELECTION_AUTHORITY_FILE_SHA256",
    "USER_SELECTION_AUTHORITY_RELATIVE",
    "USER_SELECTION_COPY_RELATIVE",
    "WORKER_RECEIPT_SCHEMA",
    "atomic_write_json",
    "atomic_publish_noreplace",
    "canonical_json_bytes",
    "canonical_sha256",
    "control_plane_receipt",
    "core_cell_id",
    "digested",
    "direct_execution_ids",
    "direct_execution_rows",
    "expected_artifact_path",
    "load_json_object",
    "repo_root_from_script",
    "representation_for_route",
    "safe_relative_path",
    "sha256_file",
    "validate_core_authority",
    "validate_p3_receipt",
    "validate_submission_authorization",
    "validate_user_selection_authority",
    "verify_exact_set",
    "verify_self_digest",
]
