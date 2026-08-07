"""Static ADAPT structural-resume helpers.

This module intentionally keeps resume loading, validation, digesting, and
secret-value guards out of ``adapt_pipeline.py``.  It is the first narrow slice
of a static ADAPT continuation route; it does not build IBM account
orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.scaffold.imported_artifact_resolution import ImportedArtifactResolution
from pipelines.scaffold.runtime_contract import ScaffoldRuntimeInput
from pipelines.scaffold.runtime_loader import (
    load_scaffold_runtime_input_from_payload,
)
from pipelines.static_adapt.formal_manifold_route_profile import (
    FORMAL_MANIFOLD_ROUTE_PROFILE_OFF,
    resolve_formal_manifold_route_profile,
)
from pipelines.static_adapt.formal_manifold_warm_start import (
    FORMAL_MANIFOLD_ROUTE,
    FORMAL_MANIFOLD_ROUTE_FAMILY,
    FORMAL_MANIFOLD_SR_SELECTOR_FAMILY,
    FormalManifoldRouteComposition,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_POWELL_COORDINATE_CHART_POLICY_CHOICES,
    SR_ROUTE_FAMILY,
    SR_ROUTE_PROFILE_CONFORMANCE_CHOICES,
    SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
    SR_ROUTE_PROFILE_DISABLED,
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
    SR_ROUTE_PROFILE_REDUCED_POWELL,
    SR_ROUTE_PROFILE_SADDLE_ONLY,
    SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_CANONICAL_V1,
    SR_ROUTE_PROFILE_REQUEST_OFF,
    normalize_sr_route_profile_request,
    validate_sr_route_profile_contract,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    expand_legacy_logical_theta,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


_SECRET_MARKER_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:qiskit[_-]?ibm[_-]?)?"
    r"(?:token|api[_-]?key|apikey|secret|password|credential)"
    r"(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

_STATIC_RESUME_ALLOWED_PROBLEMS = frozenset(
    {
        "hh",
        "molecular_vibronic_h2o_linear_fd",
    }
)

_OBSOLETE_ADMISSION_ROLLBACK_FIELDS = frozenset(
    {
        "adapt_rollback_mode",
        "adapt_rollback_tolerance",
        "structural_rollback",
        "depth_rollback",
        "zero_gain_duplicate_filter",
        "zero_gain_duplicate_guard",
        "duplicate_cooldown_policy",
        "pre_child_phase1_filter",
        "cooldown_excluded_record_count",
        "skipped_structural_rollback_rows",
    }
)

_MODELED_MINIMUM_EXECUTION_CHECKPOINT_FIELD = (
    "modeled_minimum_execution_checkpoint"
)


def _assert_no_unsupported_modeled_minimum_execution_checkpoint(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Fail closed before a legacy resume path can discard Stage-B state.

    Modeled-minimum execution checkpoints carry separate incumbent and working
    states plus scheduler service state.  The legacy scaffold and preserved
    best-frontier loaders cannot round-trip that contract, so accepting one
    would silently resume only the incumbent/static-ADAPT portion.
    """

    field = _MODELED_MINIMUM_EXECUTION_CHECKPOINT_FIELD
    present_paths: list[str] = []
    if field in payload:
        present_paths.append(field)
    adapt = payload.get("adapt_vqe", None)
    if isinstance(adapt, Mapping) and field in adapt:
        present_paths.append(f"adapt_vqe.{field}")
    checkpoint = payload.get("checkpoint", None)
    if isinstance(checkpoint, Mapping) and field in checkpoint:
        present_paths.append(f"checkpoint.{field}")
    if present_paths:
        raise ValueError(
            f"{context} cannot consume a modeled-minimum execution checkpoint; "
            "the legacy/scaffold resume contract cannot preserve separate "
            "incumbent/working state and scheduler service state. Rejected "
            "field path(s): "
            + ", ".join(present_paths)
        )


def _drop_obsolete_admission_rollback_state(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    removed_counts: dict[str, int] | None = None,
) -> tuple[Any, dict[str, int]]:
    counts = {} if removed_counts is None else removed_counts
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            remove = key in _OBSOLETE_ADMISSION_ROLLBACK_FIELDS
            remove = remove or (
                key == "suppressed_reason" and str(child) == "structural_rollback"
            )
            remove = remove or (
                key == "rollback"
                and path
                and path[-1] in {"final_full_refit", "resume_boundary_refit"}
            )
            if remove:
                counts[key] = int(counts.get(key, 0)) + 1
                continue
            cleaned_child, counts = _drop_obsolete_admission_rollback_state(
                child,
                path=(*path, key),
                removed_counts=counts,
            )
            cleaned[key] = cleaned_child
        return cleaned, counts
    if isinstance(value, list):
        cleaned_list: list[Any] = []
        for index, child in enumerate(value):
            cleaned_child, counts = _drop_obsolete_admission_rollback_state(
                child,
                path=(*path, str(index)),
                removed_counts=counts,
            )
            cleaned_list.append(cleaned_child)
        return cleaned_list, counts
    if isinstance(value, tuple):
        cleaned_tuple, counts = _drop_obsolete_admission_rollback_state(
            list(value),
            path=path,
            removed_counts=counts,
        )
        return tuple(cleaned_tuple), counts
    return value, counts


@dataclass(frozen=True)
class ResumeScaffoldSource:
    artifact_json: Path
    artifact_sha256: str
    payload: Mapping[str, Any]
    runtime_input: ScaffoldRuntimeInput
    import_summary: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeMatchedScaffold:
    selected_ops: tuple[AnsatzTerm, ...]
    selected_layout: AnsatzParameterLayout
    theta_runtime: np.ndarray
    theta_logical: np.ndarray | None
    selected_pool_indices: tuple[int, ...]
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeBestFrontierCheckpoint:
    """Strict, source-complete state for one preserved best beam branch."""

    history: tuple[Mapping[str, Any], ...]
    controller_round: int
    ansatz_depth: int
    branch_id: int
    parent_branch_id: int | None
    operator_labels: tuple[str, ...]
    theta_runtime: tuple[float, ...]
    theta_logical: tuple[float, ...]
    route_a_trust_region_state: Mapping[str, Any]
    beam_checkpoint_branch: Mapping[str, Any]
    frontier_prune_key: Mapping[str, Any]
    source_energy: float
    initial_state_digest: str
    ansatz_input_state_digest: str
    powell_coordinate_chart_policy: str | None
    route_profile_conformance: str | None
    sr_route_profile_request: str | None
    sr_route_profile_contract: Mapping[str, Any] | None
    sr_route_profile_contract_sha256: str | None
    formal_manifold_runtime_checkpoint: Mapping[str, Any] | None
    formal_manifold_route_composition: Mapping[str, Any] | None
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class ResumeCompileSmokeResult:
    required: bool
    executed: bool
    success: bool
    backend_name: str | None
    compiled_depth: int | None
    compiled_size: int | None
    compiled_count_2q: int | None
    output_json: str | None
    error: str | None

    def to_payload(self) -> dict[str, Any]:
        return dict(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"re": float(np.real(value)), "im": float(np.imag(value))}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def digest_jsonable(value: Any) -> str:
    """Return a stable SHA256 digest for a JSON-like value."""

    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_secret_marker(value: Any) -> bool:
    if value is None:
        return False
    text = str(value)
    if text == "":
        return False
    return bool(_SECRET_MARKER_RE.search(text))


def _iter_secret_value_hits(value: Any, *, path: str) -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            hits.extend(_iter_secret_value_hits(child, path=f"{path}.{key}"))
        return hits
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            hits.extend(_iter_secret_value_hits(child, path=f"{path}[{idx}]"))
        return hits
    if isinstance(value, (str, Path)) and contains_secret_marker(value):
        hits.append(path)
    return hits


def assert_no_secret_material(value: Any, *, context: str = "resume_scaffold") -> None:
    """Reject secret-like string values.

    The scan is intentionally value-only so audit keys like
    ``credential_audit`` can exist while raw token/API-key-like material cannot.
    """

    hits = _iter_secret_value_hits(value, path=str(context))
    if hits:
        preview = ", ".join(hits[:6])
        raise ValueError(
            "Secret-like token/API-key/credential value is not allowed in "
            f"{context}. Offending value path(s): {preview}"
        )


def assert_no_secret_cli_values(args_or_mapping: Any) -> None:
    mapping = vars(args_or_mapping) if not isinstance(args_or_mapping, Mapping) else args_or_mapping
    assert_no_secret_material(mapping, context="CLI arguments")


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Resume scaffold artifact must be a JSON object: {path}")
    return dict(payload)


def _adapt_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    block = payload.get("adapt_vqe", {})
    if not isinstance(block, Mapping):
        raise ValueError("Resume scaffold artifact is missing adapt_vqe object.")
    return dict(block)


def _settings_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    block = payload.get("settings", {})
    if not isinstance(block, Mapping):
        raise ValueError("Resume scaffold artifact is missing settings object.")
    return dict(block)


def _state_digest(payload: Mapping[str, Any], key: str) -> str | None:
    block = payload.get(key, None)
    if not isinstance(block, Mapping):
        return None
    return digest_jsonable(block)


def _source_depth(payload: Mapping[str, Any], runtime_input: ScaffoldRuntimeInput) -> int:
    adapt = _adapt_block(payload)
    raw = adapt.get("ansatz_depth", None)
    try:
        depth = int(raw)
        if depth >= 0:
            return depth
    except Exception:
        pass
    return int(len(runtime_input.selected_terms))


def _source_controller_round(payload: Mapping[str, Any]) -> int:
    adapt = _adapt_block(payload)
    for raw in (
        adapt.get("history_count"),
        payload.get("checkpoint", {}).get("depth")
        if isinstance(payload.get("checkpoint"), Mapping)
        else None,
    ):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    history = adapt.get("history", ())
    if isinstance(history, Sequence) and not isinstance(
        history,
        (str, bytes, bytearray),
    ):
        return int(len(history))
    return 0


def _source_continuation_mode(payload: Mapping[str, Any]) -> str | None:
    settings = _settings_block(payload)
    adapt = _adapt_block(payload)
    continuation = adapt.get("continuation", {}) if isinstance(adapt.get("continuation", None), Mapping) else {}
    candidates: list[tuple[str, str]] = []
    for label, raw in (
        ("settings.adapt_continuation_mode", settings.get("adapt_continuation_mode")),
        ("settings.continuation_mode", settings.get("continuation_mode")),
        ("adapt_vqe.continuation_mode", adapt.get("continuation_mode")),
        ("adapt_vqe.continuation.mode", continuation.get("mode")),
        ("adapt_vqe.continuation.continuation_mode", continuation.get("continuation_mode")),
    ):
        if raw not in {None, ""}:
            candidates.append((label, str(raw).strip()))
    if not candidates:
        return None
    normalized = {mode.lower() for _label, mode in candidates}
    if len(normalized) > 1:
        raise ValueError(
            "Resume artifact has conflicting continuation mode fields: "
            + json.dumps({label: mode for label, mode in candidates}, sort_keys=True)
        )
    return candidates[0][1]


_SR_POWELL_CHART_POLICY_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_powell_coordinate_chart_policy"),
    (
        "settings",
        "formal_manifold_route_composition",
        "singleton_response_selector",
        "sr_powell_coordinate_chart_policy",
    ),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "powell_coordinate_chart_policy",
    ),
    ("settings", "static_route_identity", "powell_coordinate_chart_policy"),
    ("adapt_vqe", "static_route_identity", "powell_coordinate_chart_policy"),
    (
        "adapt_vqe",
        "formal_manifold_route_composition",
        "singleton_response_selector",
        "sr_powell_coordinate_chart_policy",
    ),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "powell_coordinate_chart_policy",
    ),
    ("adapt_vqe", "optimizer_coordinate_chart", "powell_coordinate_chart_policy"),
    (
        "adapt_vqe",
        "terminal_active_prefix_checkpoint",
        "optimizer_coordinate_chart",
        "powell_coordinate_chart_policy",
    ),
    (
        "adapt_vqe",
        "continuation",
        "terminal_active_prefix_checkpoint",
        "optimizer_coordinate_chart",
        "powell_coordinate_chart_policy",
    ),
    ("checkpoint", "powell_coordinate_chart_policy"),
    ("checkpoint", "static_route_identity", "powell_coordinate_chart_policy"),
    ("checkpoint", "optimizer_coordinate_chart", "powell_coordinate_chart_policy"),
    (
        "checkpoint",
        "historical_singleton_coordinate_trust_overlay",
        "powell_coordinate_chart_policy",
    ),
    (
        "terminal_active_prefix_checkpoint",
        "optimizer_coordinate_chart",
        "powell_coordinate_chart_policy",
    ),
    ("static_route_identity", "powell_coordinate_chart_policy"),
    (
        "formal_manifold_route_composition",
        "singleton_response_selector",
        "sr_powell_coordinate_chart_policy",
    ),
    ("optimizer_coordinate_chart", "powell_coordinate_chart_policy"),
)

_SR_ROUTE_PROFILE_CONFORMANCE_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "route_profile_conformance"),
    ("settings", "sr_powell_route_instance", "route_profile_conformance"),
    ("settings", "static_route_identity", "route_profile_conformance"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "route_profile_conformance",
    ),
    ("adapt_vqe", "static_route_identity", "route_profile_conformance"),
    ("adapt_vqe", "sr_powell_route_instance", "route_profile_conformance"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "route_profile_conformance",
    ),
    ("checkpoint", "route_profile_conformance"),
    ("checkpoint", "static_route_identity", "route_profile_conformance"),
    ("checkpoint", "sr_powell_route_instance", "route_profile_conformance"),
    ("static_route_identity", "route_profile_conformance"),
    ("sr_powell_route_instance", "route_profile_conformance"),
)

_SR_COORDINATE_SOLVE_SCOPE_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "historical_singleton_coordinate_solve_scope"),
    ("settings", "sr_powell_route_instance", "coordinate_solve_scope"),
    ("settings", "static_route_identity", "coordinate_solve_scope"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "coordinate_solve_scope",
    ),
    ("adapt_vqe", "static_route_identity", "coordinate_solve_scope"),
    ("adapt_vqe", "sr_powell_route_instance", "coordinate_solve_scope"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "coordinate_solve_scope",
    ),
    ("checkpoint", "static_route_identity", "coordinate_solve_scope"),
    ("checkpoint", "sr_powell_route_instance", "coordinate_solve_scope"),
    ("static_route_identity", "coordinate_solve_scope"),
    ("sr_powell_route_instance", "coordinate_solve_scope"),
)

_SR_ROUTE_FAMILY_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "route_family"),
    ("settings", "formal_manifold_route_composition", "route_family"),
    ("settings", "static_route_identity", "route_family"),
    ("adapt_vqe", "formal_manifold_route_composition", "route_family"),
    ("adapt_vqe", "static_route_identity", "route_family"),
    ("checkpoint", "static_route_identity", "route_family"),
    ("static_route_identity", "route_family"),
    ("formal_manifold_route_composition", "route_family"),
)

_SR_ROUTE_PROFILE_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "route_profile"),
    ("settings", "formal_manifold_route_composition", "route_profile"),
    ("settings", "static_route_identity", "route_profile"),
    ("adapt_vqe", "static_route_identity", "route_profile"),
    ("adapt_vqe", "formal_manifold_route_composition", "route_profile"),
    ("checkpoint", "static_route_identity", "route_profile"),
    ("static_route_identity", "route_profile"),
    ("formal_manifold_route_composition", "route_profile"),
)

_SR_ROUTE_PROFILES = frozenset(
    {
        SR_ROUTE_PROFILE_DISABLED,
        SR_ROUTE_PROFILE_REDUCED_POWELL,
        SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
        SR_ROUTE_PROFILE_SADDLE_ONLY,
        SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM,
    }
)

_SR_ROUTE_PROFILE_TO_POWELL_CHART = {
    SR_ROUTE_PROFILE_DISABLED: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_REDUCED_POWELL: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_SADDLE_ONLY: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
}

_SOURCE_LOCK_INDICATOR_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "phase3_source_lock_preferred_sequence"),
    ("adapt_vqe", "phase3_source_lock_preferred_sequence"),
    ("phase3_source_lock_preferred_sequence",),
    ("phase3_source_lock",),
)

_SR_ROUTE_PROFILE_REQUEST_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_route_profile_request"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_request",
    ),
    ("settings", "static_route_identity", "sr_route_profile_request"),
    ("adapt_vqe", "sr_route_profile_request"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_request",
    ),
    ("adapt_vqe", "static_route_identity", "sr_route_profile_request"),
    ("checkpoint", "sr_route_profile_request"),
    ("checkpoint", "settings", "sr_route_profile_request"),
    ("checkpoint", "static_route_identity", "sr_route_profile_request"),
    ("sr_route_profile_request",),
    ("static_route_identity", "sr_route_profile_request"),
)

_SR_ROUTE_PROFILE_CONTRACT_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_route_profile_contract"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract",
    ),
    ("settings", "static_route_identity", "sr_route_profile_contract"),
    ("adapt_vqe", "sr_route_profile_contract"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract",
    ),
    ("adapt_vqe", "static_route_identity", "sr_route_profile_contract"),
    ("checkpoint", "settings", "sr_route_profile_contract"),
    ("checkpoint", "static_route_identity", "sr_route_profile_contract"),
    ("sr_route_profile_contract",),
    ("static_route_identity", "sr_route_profile_contract"),
)

_SR_ROUTE_PROFILE_CONTRACT_SHA256_PATHS: tuple[tuple[str, ...], ...] = (
    ("settings", "sr_route_profile_contract_sha256"),
    (
        "settings",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract_sha256",
    ),
    (
        "settings",
        "static_route_identity",
        "sr_route_profile_contract_sha256",
    ),
    ("adapt_vqe", "sr_route_profile_contract_sha256"),
    (
        "adapt_vqe",
        "historical_singleton_coordinate_trust_overlay",
        "sr_route_profile_contract_sha256",
    ),
    (
        "adapt_vqe",
        "static_route_identity",
        "sr_route_profile_contract_sha256",
    ),
    ("checkpoint", "sr_route_profile_contract_sha256"),
    ("checkpoint", "settings", "sr_route_profile_contract_sha256"),
    (
        "checkpoint",
        "static_route_identity",
        "sr_route_profile_contract_sha256",
    ),
    ("sr_route_profile_contract_sha256",),
    ("static_route_identity", "sr_route_profile_contract_sha256"),
)


def _nested_payload_value(
    payload: Mapping[str, Any], path: tuple[str, ...]
) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _path_label(path: tuple[str, ...]) -> str:
    return ".".join(path)


def validate_resume_sr_route_profile_contract(
    payload: Mapping[str, Any],
    *,
    expected_profile_request: str | None = None,
    expected_contract: Mapping[str, Any] | None = None,
    expected_contract_sha256: str | None = None,
    context: str = "Static scaffold resume",
) -> dict[str, Any]:
    """Validate the complete canonical SR-SNAKE v1 replay identity.

    The Powell-chart validator protects one important optimizer setting.  The
    canonical route selector additionally locks the complete historical
    execution contract.  Every serialized alias must agree, and an invocation
    that explicitly requests SR-SNAKE v1 may not consume a legacy artifact
    that lacks this contract.
    """

    request_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_REQUEST_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        try:
            value = normalize_sr_route_profile_request(raw)
        except ValueError as exc:
            raise ValueError(
                f"{context} has an unknown SR route-profile request at "
                f"{_path_label(path)}: {raw!r}."
            ) from exc
        request_fields[_path_label(path)] = value
    distinct_requests = sorted(set(request_fields.values()))
    if len(distinct_requests) > 1:
        raise ValueError(
            f"{context} has conflicting SR route-profile requests: "
            + json.dumps(request_fields, sort_keys=True)
        )
    artifact_request = distinct_requests[0] if distinct_requests else None

    contract_fields: dict[str, Mapping[str, Any]] = {}
    for path in _SR_ROUTE_PROFILE_CONTRACT_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{context} has a non-object SR route-profile contract at "
                f"{_path_label(path)}."
            )
        contract_fields[_path_label(path)] = dict(raw)
    contract_digests = {
        digest_jsonable(dict(contract)) for contract in contract_fields.values()
    }
    if len(contract_digests) > 1:
        raise ValueError(
            f"{context} has conflicting serialized SR route-profile contracts "
            f"at {sorted(contract_fields)}."
        )
    artifact_contract = (
        dict(next(iter(contract_fields.values()))) if contract_fields else None
    )

    sha_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_CONTRACT_SHA256_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        value = str(raw).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                f"{context} has an invalid SR route-profile contract SHA-256 "
                f"at {_path_label(path)}: {raw!r}."
            )
        sha_fields[_path_label(path)] = value
    distinct_sha = sorted(set(sha_fields.values()))
    if len(distinct_sha) > 1:
        raise ValueError(
            f"{context} has conflicting SR route-profile contract SHA-256 "
            "fields: "
            + json.dumps(sha_fields, sort_keys=True)
        )
    artifact_sha256 = distinct_sha[0] if distinct_sha else None

    expected_request = (
        None
        if expected_profile_request in {None, ""}
        else normalize_sr_route_profile_request(expected_profile_request)
    )
    canonical_present = bool(
        artifact_request == SR_ROUTE_PROFILE_CANONICAL_V1
        or artifact_contract is not None
        or artifact_sha256 is not None
    )
    canonical_expected = expected_request == SR_ROUTE_PROFILE_CANONICAL_V1

    if canonical_present:
        if artifact_request is None:
            raise ValueError(
                f"{context} has an SR route-profile contract but no serialized "
                "route-profile request."
            )
        if artifact_request != SR_ROUTE_PROFILE_CANONICAL_V1:
            raise ValueError(
                f"{context} associates the canonical SR contract with "
                f"profile request {artifact_request!r}."
            )
        validate_sr_route_profile_contract(
            profile_request=artifact_request,
            contract=artifact_contract,
            contract_sha256=artifact_sha256,
        )

    if canonical_expected:
        if not canonical_present:
            raise ValueError(
                f"{context} was invoked as SR-SNAKE v1, but the resume "
                "artifact lacks its complete route-profile contract."
            )
        expected_payload = validate_sr_route_profile_contract(
            profile_request=expected_request,
            contract=expected_contract,
            contract_sha256=expected_contract_sha256,
        )
        if artifact_contract != expected_payload:
            raise ValueError(
                f"{context} SR-SNAKE v1 route-profile contract does not match "
                "the current invocation."
            )
        if artifact_sha256 != str(expected_contract_sha256 or "").lower():
            raise ValueError(
                f"{context} SR-SNAKE v1 route-profile contract SHA-256 does "
                "not match the current invocation."
            )
    elif expected_request == SR_ROUTE_PROFILE_REQUEST_OFF and canonical_present:
        raise ValueError(
            f"{context} artifact is canonical SR-SNAKE v1, but the current "
            "invocation did not explicitly request that route profile."
        )

    return {
        "schema_version": "static_adapt_resume_sr_route_profile_contract_v1",
        "status": "pass" if canonical_present else "not_applicable",
        "artifact_profile_request": artifact_request,
        "expected_profile_request": expected_request,
        "contract_sha256": artifact_sha256,
        "expected_contract_sha256": (
            None
            if expected_contract_sha256 in {None, ""}
            else str(expected_contract_sha256).strip().lower()
        ),
        "request_fields": dict(sorted(request_fields.items())),
        "contract_source_fields": sorted(contract_fields),
        "contract_sha256_fields": dict(sorted(sha_fields.items())),
        "inferred": False,
    }


def _resume_powell_chart_policy_required(payload: Mapping[str, Any]) -> bool:
    route_families = {
        str(value).strip().lower()
        for path in _SR_ROUTE_FAMILY_PATHS
        if (value := _nested_payload_value(payload, path)) is not None
        and value != ""
    }
    if str(SR_ROUTE_FAMILY).strip().lower() in route_families:
        return True
    if str(FORMAL_MANIFOLD_ROUTE_FAMILY).strip().lower() in route_families:
        composition = extract_formal_manifold_route_composition(payload)
        if composition is not None and str(
            composition.get("candidate_selector_family") or ""
        ).strip().lower() == str(
            FORMAL_MANIFOLD_SR_SELECTOR_FAMILY
        ).strip().lower():
            return True
    route_profiles = {
        str(value).strip().lower()
        for path in _SR_ROUTE_PROFILE_PATHS
        if (value := _nested_payload_value(payload, path)) is not None
        and value != ""
    }
    if route_profiles & {
        str(value).strip().lower() for value in _SR_ROUTE_PROFILES
    }:
        return True
    for path in _SOURCE_LOCK_INDICATOR_PATHS:
        value = _nested_payload_value(payload, path)
        if isinstance(value, str):
            if value.strip():
                return True
        elif value:
            return True
    return False


def validate_resume_powell_coordinate_chart_policy(
    payload: Mapping[str, Any],
    *,
    expected_policy: str | None = None,
    expected_route_profile_conformance: str | None = None,
    required: bool | None = None,
    context: str = "Static scaffold resume",
) -> dict[str, Any]:
    """Extract and strictly validate serialized SR-SNAKE Powell-chart identity.

    A historical or source-locked SR replay must never infer this execution
    policy from the live optimizer.  Every serialized alias is therefore
    collected and required to agree.  Callers may additionally supply the
    already-resolved current route policy to detect resume drift.
    """

    required_value = (
        _resume_powell_chart_policy_required(payload)
        if required is None
        else bool(required)
    )
    source_fields: dict[str, str] = {}
    for path in _SR_POWELL_CHART_POLICY_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw is None or raw == "":
            continue
        value = str(raw).strip().lower()
        if value not in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
            raise ValueError(
                f"{context} has unknown Powell coordinate-chart policy at "
                f"{_path_label(path)}: {raw!r}. Expected one of "
                f"{list(SR_POWELL_COORDINATE_CHART_POLICY_CHOICES)}."
            )
        source_fields[_path_label(path)] = value

    distinct = sorted(set(source_fields.values()))
    if len(distinct) > 1:
        raise ValueError(
            f"{context} has conflicting Powell coordinate-chart policies: "
            + json.dumps(source_fields, sort_keys=True)
        )
    resolved = distinct[0] if distinct else None
    if required_value and resolved is None:
        raise ValueError(
            f"{context} is SR/source-locked but is missing the explicit "
            "Powell coordinate-chart policy; replay fails closed."
        )

    conformance_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_CONFORMANCE_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        value = str(raw).strip().lower()
        if value not in SR_ROUTE_PROFILE_CONFORMANCE_CHOICES:
            raise ValueError(
                f"{context} has unknown route-profile conformance marker at "
                f"{_path_label(path)}: {raw!r}. Expected one of "
                f"{list(SR_ROUTE_PROFILE_CONFORMANCE_CHOICES)}."
            )
        conformance_fields[_path_label(path)] = value
    distinct_conformance = sorted(set(conformance_fields.values()))
    if len(distinct_conformance) > 1:
        raise ValueError(
            f"{context} has conflicting route-profile conformance markers: "
            + json.dumps(conformance_fields, sort_keys=True)
        )
    route_profile_conformance = (
        distinct_conformance[0] if distinct_conformance else None
    )

    coordinate_scope_fields: dict[str, str] = {}
    for path in _SR_COORDINATE_SOLVE_SCOPE_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        coordinate_scope_fields[_path_label(path)] = str(raw).strip().lower()
    distinct_coordinate_scopes = sorted(set(coordinate_scope_fields.values()))
    if len(distinct_coordinate_scopes) > 1:
        raise ValueError(
            f"{context} has conflicting SR coordinate-solve scopes: "
            + json.dumps(coordinate_scope_fields, sort_keys=True)
        )
    coordinate_solve_scope = (
        distinct_coordinate_scopes[0] if distinct_coordinate_scopes else None
    )

    route_profile_fields: dict[str, str] = {}
    for path in _SR_ROUTE_PROFILE_PATHS:
        raw = _nested_payload_value(payload, path)
        if raw in {None, ""}:
            continue
        profile = str(raw).strip().lower()
        if profile in _SR_ROUTE_PROFILE_TO_POWELL_CHART:
            route_profile_fields[_path_label(path)] = profile
    profile_expected_charts = {
        _SR_ROUTE_PROFILE_TO_POWELL_CHART[profile]
        for profile in route_profile_fields.values()
    }
    if len(profile_expected_charts) > 1:
        raise ValueError(
            f"{context} has SR route profiles with incompatible Powell "
            "coordinate-chart policies: "
            + json.dumps(route_profile_fields, sort_keys=True)
        )
    route_profile_expected = (
        next(iter(profile_expected_charts)) if profile_expected_charts else None
    )
    phase2_phase3_expanded_pair = bool(
        resolved
        == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        and route_profile_fields
        and set(route_profile_fields.values())
        == {SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED}
    )
    explicit_unpromoted_ablation = bool(
        phase2_phase3_expanded_pair
        and route_profile_conformance
        == SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
        and coordinate_solve_scope
        == SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
    )
    if phase2_phase3_expanded_pair and not explicit_unpromoted_ablation:
        raise ValueError(
            f"{context} explicit Phase-II+III expanded-chart resume requires "
            "the serialized unpromoted route-profile conformance marker and "
            f"coordinate_solve_scope={SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1!r}; "
            "replay refuses to infer either field."
        )
    if (
        route_profile_conformance
        == SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
        and not explicit_unpromoted_ablation
    ):
        raise ValueError(
            f"{context} serialized the unpromoted explicit-ablation marker "
            "without the exact Phase-II+III expanded-chart route instance."
        )
    if (
        resolved is not None
        and route_profile_expected is not None
        and resolved != route_profile_expected
        and not explicit_unpromoted_ablation
    ):
        raise ValueError(
            f"{context} route-profile/Powell-chart mismatch: "
            f"profiles={json.dumps(route_profile_fields, sort_keys=True)}, "
            f"chart={resolved!r}, expected={route_profile_expected!r}."
        )

    expected: str | None = None
    if expected_policy not in {None, ""}:
        expected = str(expected_policy).strip().lower()
        if expected not in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
            raise ValueError(
                f"{context} expected Powell coordinate-chart policy is unknown: "
                f"{expected_policy!r}."
            )
        if resolved is None:
            raise ValueError(
                f"{context} is missing the Powell coordinate-chart policy "
                f"required by the current route ({expected})."
            )
        if resolved != expected:
            raise ValueError(
                f"{context} Powell coordinate-chart policy mismatch: "
                f"artifact={resolved!r}, current={expected!r}."
            )

    expected_conformance: str | None = None
    if expected_route_profile_conformance not in {None, ""}:
        expected_conformance = str(
            expected_route_profile_conformance
        ).strip().lower()
        if expected_conformance not in SR_ROUTE_PROFILE_CONFORMANCE_CHOICES:
            raise ValueError(
                f"{context} expected route-profile conformance marker is "
                f"unknown: {expected_route_profile_conformance!r}."
            )
        if route_profile_conformance is None:
            raise ValueError(
                f"{context} is missing the route-profile conformance marker "
                f"required by the current route ({expected_conformance})."
            )
        if route_profile_conformance != expected_conformance:
            raise ValueError(
                f"{context} route-profile conformance mismatch: "
                f"artifact={route_profile_conformance!r}, "
                f"current={expected_conformance!r}."
            )

    return {
        "schema_version": "static_adapt_resume_powell_chart_policy_v1",
        "status": "pass" if resolved is not None else "not_applicable",
        "required": bool(required_value or expected is not None),
        "resolved_policy": resolved,
        "expected_policy": expected,
        "source_fields": dict(sorted(source_fields.items())),
        "source_field_count": int(len(source_fields)),
        "route_profile_fields": dict(sorted(route_profile_fields.items())),
        "route_profile_expected_policy": route_profile_expected,
        "route_profile_conformance": route_profile_conformance,
        "expected_route_profile_conformance": expected_conformance,
        "route_profile_conformance_fields": dict(
            sorted(conformance_fields.items())
        ),
        "coordinate_solve_scope": coordinate_solve_scope,
        "coordinate_solve_scope_fields": dict(
            sorted(coordinate_scope_fields.items())
        ),
        "explicit_unpromoted_ablation": bool(explicit_unpromoted_ablation),
        "inferred": False,
    }


def build_resume_import_summary(
    source: ResumeScaffoldSource | None = None,
    *,
    artifact_json: str | Path | None = None,
    artifact_sha256: str | None = None,
    payload: Mapping[str, Any] | None = None,
    runtime_input: ScaffoldRuntimeInput | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if source is not None:
        artifact_json = source.artifact_json
        artifact_sha256 = source.artifact_sha256
        payload = source.payload
        runtime_input = source.runtime_input
    if artifact_json is None or artifact_sha256 is None or payload is None or runtime_input is None:
        raise ValueError("build_resume_import_summary requires source or explicit artifact/payload/runtime_input.")
    adapt = _adapt_block(payload)
    settings = _settings_block(payload)
    operator_labels = [str(x) for x in adapt.get("operators", [])]
    theta_runtime = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)
    theta_logical = (
        None
        if runtime_input.theta_logical is None
        else np.asarray(runtime_input.theta_logical, dtype=float).reshape(-1)
    )
    parameterization = adapt.get("parameterization", None)
    powell_chart_validation = validate_resume_powell_coordinate_chart_policy(
        payload,
        context="Resume import summary",
    )
    sr_route_profile_validation = validate_resume_sr_route_profile_contract(
        payload,
        context="Resume import summary",
    )
    _cleaned_payload, removed_obsolete_counts = (
        _drop_obsolete_admission_rollback_state(payload)
    )
    summary = {
        "schema_version": "static_hh_adapt_resume_import_v1",
        "path": str(Path(artifact_json)),
        "artifact_sha256": str(artifact_sha256),
        "source_ansatz_depth": int(_source_depth(payload, runtime_input)),
        "source_controller_round": int(_source_controller_round(payload)),
        "source_num_parameters": int(theta_runtime.size),
        "source_logical_num_parameters": int(runtime_input.base_layout.logical_parameter_count),
        "source_pool_type": (
            str(adapt.get("pool_type"))
            if adapt.get("pool_type") not in {None, ""}
            else (
                str(settings.get("adapt_pool"))
                if settings.get("adapt_pool") not in {None, ""}
                else None
            )
        ),
        "source_continuation_mode": _source_continuation_mode(payload),
        "powell_coordinate_chart_policy": powell_chart_validation[
            "resolved_policy"
        ],
        "route_profile_conformance": powell_chart_validation[
            "route_profile_conformance"
        ],
        "powell_coordinate_chart_policy_validation": powell_chart_validation,
        "sr_route_profile_request": sr_route_profile_validation[
            "artifact_profile_request"
        ],
        "sr_route_profile_contract_sha256": sr_route_profile_validation[
            "contract_sha256"
        ],
        "sr_route_profile_contract_validation": sr_route_profile_validation,
        "operator_count": int(len(operator_labels)),
        "operator_labels_digest": digest_jsonable(operator_labels),
        "parameterization_digest": digest_jsonable(parameterization),
        "theta_runtime_digest": digest_jsonable([float(x) for x in theta_runtime.tolist()]),
        "theta_logical_digest": (
            None if theta_logical is None else digest_jsonable([float(x) for x in theta_logical.tolist()])
        ),
        "initial_state_digest": _state_digest(payload, "initial_state"),
        "ansatz_input_state_digest": _state_digest(payload, "ansatz_input_state"),
        "runtime_loader_provenance": dict(getattr(runtime_input, "provenance", {}) or {}),
        "obsolete_admission_rollback_state_migration": {
            "schema": "obsolete_admission_rollback_state_drop_v1",
            "applied": bool(removed_obsolete_counts),
            "behavior": "ignored_and_dropped_before_resume",
            "removed_field_counts": dict(sorted(removed_obsolete_counts.items())),
        },
        "validation": dict(validation or {}),
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(summary, context="resume import summary")
    return summary


def load_static_resume_source(
    artifact_json: str | Path,
    *,
    loader_mode: str = "replay_family",
    settings_overrides: Mapping[str, Any] | None = None,
) -> ResumeScaffoldSource:
    path = Path(artifact_json)
    raw_payload = _read_json_object(path)
    payload_cleaned, removed_obsolete_counts = (
        _drop_obsolete_admission_rollback_state(raw_payload)
    )
    payload = dict(payload_cleaned)
    runtime_payload: Mapping[str, Any]
    if settings_overrides:
        payload_copy = dict(payload)
        settings_copy = dict(_settings_block(payload))
        for key, value in settings_overrides.items():
            if value is None or value == "":
                continue
            settings_copy[str(key)] = value
        payload_copy["settings"] = settings_copy
        runtime_payload = payload_copy
    else:
        runtime_payload = payload
    assert_no_secret_material(raw_payload, context=f"resume artifact {path}")
    assert_no_secret_material(runtime_payload, context=f"resume runtime payload {path}")
    runtime_input = load_scaffold_runtime_input_from_payload(
        runtime_payload,
        artifact_json=path,
        loader_mode=str(loader_mode),
        generator_family="match_adapt",
        fallback_family="full_meta",
    )
    sha = file_sha256(path)
    summary = build_resume_import_summary(
        artifact_json=path,
        artifact_sha256=sha,
        payload=runtime_payload,
        runtime_input=runtime_input,
    )
    summary["obsolete_admission_rollback_state_migration"] = {
        "schema": "obsolete_admission_rollback_state_drop_v1",
        "applied": bool(removed_obsolete_counts),
        "behavior": "ignored_and_dropped_before_resume",
        "removed_field_counts": dict(sorted(removed_obsolete_counts.items())),
    }
    if settings_overrides:
        summary["loader_settings_overrides"] = {
            str(key): str(value)
            for key, value in settings_overrides.items()
            if value is not None and value != ""
        }
        summary["artifact_payload_runtime_patched"] = True
        assert_no_secret_material(summary, context="resume import summary with overrides")
    return ResumeScaffoldSource(
        artifact_json=path,
        artifact_sha256=sha,
        payload=runtime_payload,
        runtime_input=runtime_input,
        import_summary=summary,
    )


def _arg_value(args: Any, name: str, default: Any = None) -> Any:
    if args is None:
        return default
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def extract_formal_manifold_route_composition(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract one internally consistent normalized FM composition identity."""

    if not isinstance(payload, Mapping):
        raise TypeError("resume payload must be a mapping.")
    adapt = _adapt_block(payload)
    settings = _settings_block(payload)
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for owner_name, owner in (
        ("payload", payload),
        ("adapt_vqe", adapt),
        ("settings", settings),
    ):
        direct = owner.get("formal_manifold_route_composition")
        if isinstance(direct, Mapping):
            candidates.append(
                (f"{owner_name}.formal_manifold_route_composition", direct)
            )
        identity = owner.get("static_route_identity")
        if isinstance(identity, Mapping) and (
            str(identity.get("route_family", ""))
            == FORMAL_MANIFOLD_ROUTE_FAMILY
            or str(identity.get("adapt_reoptimization_route", ""))
            == FORMAL_MANIFOLD_ROUTE
        ):
            candidates.append((f"{owner_name}.static_route_identity", identity))
    if not candidates:
        for owner_name, owner in (("adapt_vqe", adapt), ("settings", settings)):
            if (
                str(owner.get("route_family", ""))
                == FORMAL_MANIFOLD_ROUTE_FAMILY
                or str(owner.get("adapt_reoptimization_route", ""))
                == FORMAL_MANIFOLD_ROUTE
            ):
                candidates.append((owner_name, owner))
    if not candidates:
        return None
    normalized: list[tuple[str, dict[str, Any]]] = []
    for field_path, candidate in candidates:
        try:
            resolved = FormalManifoldRouteComposition.from_mapping(
                candidate
            ).as_dict()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Formal-Manifold route composition at {field_path}: {exc}"
            ) from exc
        normalized.append((field_path, resolved))
    reference_path, reference = normalized[0]
    disagreements = [
        field_path
        for field_path, candidate in normalized[1:]
        if candidate != reference
    ]
    if disagreements:
        raise ValueError(
            "Formal-Manifold route composition fields disagree: "
            + ", ".join([reference_path, *disagreements])
        )
    return dict(reference)


def _expected_formal_manifold_route_composition(
    *,
    args: Any | None,
    explicit: Mapping[str, Any] | FormalManifoldRouteComposition | None,
) -> dict[str, Any] | None:
    if explicit is not None:
        return FormalManifoldRouteComposition.from_mapping(explicit).as_dict()
    from_args = _arg_value(args, "formal_manifold_route_composition", None)
    if from_args is not None:
        return FormalManifoldRouteComposition.from_mapping(from_args).as_dict()
    requested_profile = str(
        _arg_value(args, "formal_manifold_route_profile", "") or ""
    ).strip().lower()
    if requested_profile in {"", FORMAL_MANIFOLD_ROUTE_PROFILE_OFF}:
        return None
    resolved = resolve_formal_manifold_route_profile(requested_profile)
    if resolved is None:
        return None
    return FormalManifoldRouteComposition.from_mapping(
        resolved.as_dict()
    ).as_dict()


def _compare_setting(
    mismatches: list[dict[str, Any]],
    *,
    name: str,
    artifact_value: Any,
    current_value: Any,
    cast: Callable[[Any], Any] = str,
    abs_tol: float | None = None,
) -> None:
    try:
        artifact_cast = cast(artifact_value)
        current_cast = cast(current_value)
    except Exception:
        artifact_cast = artifact_value
        current_cast = current_value
    if abs_tol is not None:
        try:
            if abs(float(artifact_cast) - float(current_cast)) <= float(abs_tol):
                return
        except Exception:
            pass
    if artifact_cast != current_cast:
        mismatches.append(
            {
                "field": str(name),
                "artifact": artifact_value,
                "current": current_value,
            }
        )


def validate_static_hh_resume_source(
    source: ResumeScaffoldSource,
    *,
    args: Any | None = None,
    continuation_mode: str | None = None,
    expected_powell_coordinate_chart_policy: str | None = None,
    expected_route_profile_conformance: str | None = None,
    formal_manifold_route_composition: (
        Mapping[str, Any] | FormalManifoldRouteComposition | None
    ) = None,
) -> dict[str, Any]:
    payload = source.payload
    _assert_no_unsupported_modeled_minimum_execution_checkpoint(
        payload,
        context="Static scaffold resume",
    )
    settings = _settings_block(payload)
    adapt = _adapt_block(payload)
    artifact_fm_composition = extract_formal_manifold_route_composition(payload)
    expected_fm_composition = _expected_formal_manifold_route_composition(
        args=args,
        explicit=formal_manifold_route_composition,
    )
    if expected_fm_composition is not None:
        if artifact_fm_composition is None:
            raise ValueError(
                "Static scaffold resume artifact lacks the required "
                "Formal-Manifold route composition."
            )
        if artifact_fm_composition != expected_fm_composition:
            raise ValueError(
                "Static scaffold resume Formal-Manifold route composition drifted."
            )
    elif artifact_fm_composition is not None and args is not None:
        requested_profile = str(
            _arg_value(args, "formal_manifold_route_profile", "") or ""
        ).strip().lower()
        if requested_profile in {"", FORMAL_MANIFOLD_ROUTE_PROFILE_OFF}:
            raise ValueError(
                "Static scaffold resume artifact is Formal-Manifold but the "
                "current route profile is off or unspecified."
            )
    effective_expected_powell_policy = expected_powell_coordinate_chart_policy
    if effective_expected_powell_policy in {None, ""} and args is not None:
        requested_powell_policy = _arg_value(
            args,
            "sr_powell_coordinate_chart_policy",
            None,
        )
        requested_powell_policy_key = str(
            requested_powell_policy or ""
        ).strip().lower()
        if requested_powell_policy_key in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
            effective_expected_powell_policy = requested_powell_policy_key
        elif requested_powell_policy_key not in {
            "",
            SR_POWELL_COORDINATE_CHART_AUTO,
        }:
            raise ValueError(
                "Static scaffold resume current Powell coordinate-chart request "
                f"is unknown: {requested_powell_policy!r}."
            )
    powell_chart_validation = validate_resume_powell_coordinate_chart_policy(
        payload,
        expected_policy=effective_expected_powell_policy,
        expected_route_profile_conformance=(
            expected_route_profile_conformance
        ),
        context="Static scaffold resume",
    )
    expected_sr_profile_request = None
    expected_sr_contract = None
    expected_sr_contract_sha256 = None
    if args is not None:
        expected_sr_profile_request = _arg_value(
            args,
            "sr_route_profile_request",
            SR_ROUTE_PROFILE_REQUEST_OFF,
        )
        expected_sr_contract = _arg_value(
            args,
            "sr_route_profile_contract",
            None,
        )
        expected_sr_contract_sha256 = _arg_value(
            args,
            "sr_route_profile_contract_sha256",
            None,
        )
    sr_route_profile_validation = validate_resume_sr_route_profile_contract(
        payload,
        expected_profile_request=expected_sr_profile_request,
        expected_contract=expected_sr_contract,
        expected_contract_sha256=expected_sr_contract_sha256,
        context="Static scaffold resume",
    )
    problem = str(settings.get("problem", "")).strip().lower()
    if problem not in _STATIC_RESUME_ALLOWED_PROBLEMS:
        raise ValueError(
            "Static scaffold resume only supports "
            f"{sorted(_STATIC_RESUME_ALLOWED_PROBLEMS)} for this slice; artifact problem={problem!r}."
        )
    current_problem = _arg_value(args, "problem", problem)
    if current_problem is not None and str(current_problem).strip().lower() != problem:
        raise ValueError(
            "Resume artifact problem does not match current static request: "
            f"{problem!r} != {str(current_problem).strip().lower()!r}."
        )
    if str(_arg_value(args, "adapt_resume_mode", "scaffold_v1")) != "scaffold_v1":
        raise ValueError("Only --adapt-resume-mode scaffold_v1 is supported.")
    if adapt.get("logical_parameterization") not in {None, "", "single_term"}:
        raise ValueError("seq2p/logical-product resume artifacts are out of scope for scaffold_v1.")
    if not isinstance(adapt.get("parameterization", None), Mapping):
        raise ValueError("Structural resume requires adapt_vqe.parameterization.")
    theta_runtime = np.asarray(source.runtime_input.theta_runtime, dtype=float).reshape(-1)
    if int(theta_runtime.size) != int(source.runtime_input.base_layout.runtime_parameter_count):
        raise ValueError("Resume theta length does not match runtime layout parameter count.")
    if not isinstance(payload.get("ansatz_input_state", None), Mapping):
        raise ValueError("Structural resume requires ansatz_input_state in the source artifact.")
    if not isinstance(payload.get("initial_state", None), Mapping):
        raise ValueError("Structural resume requires initial_state in the source artifact.")

    mismatches: list[dict[str, Any]] = []
    if args is not None:
        generic_comparisons: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
            ("L", "L", int),
            ("n_ph_max", "n_ph_max", int),
            ("boson_encoding", "boson_encoding", str),
            ("ordering", "ordering", str),
            ("include_zero_point", "include_zero_point", bool),
        )
        hh_comparisons: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
            ("t", "t", float),
            ("u", "u", float),
            ("dv", "dv", float),
            ("omega0", "omega0", float),
            ("g_ep", "g_ep", float),
            ("boundary", "boundary", str),
        )
        h2o_comparisons: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
            (
                "molecular_vibronic_h2o_linear_fd_fixture_json",
                "molecular_vibronic_h2o_linear_fd_fixture_json",
                str,
            ),
        )
        comparisons = list(generic_comparisons)
        if problem == "hh":
            comparisons.extend(hh_comparisons)
        elif problem == "molecular_vibronic_h2o_linear_fd":
            comparisons.extend(h2o_comparisons)
        for setting_name, arg_name, caster in comparisons:
            if setting_name not in settings:
                # Older ADAPT artifacts did not always serialize defaults such
                # as include_zero_point.  Missing legacy defaults are recorded
                # in validation metadata instead of causing a hard failure;
                # fields that are present must still match exactly.
                continue
            _compare_setting(
                mismatches,
                name=setting_name,
                artifact_value=settings.get(setting_name),
                current_value=_arg_value(args, arg_name),
                cast=caster,
                abs_tol=(1e-10 if caster is float else None),
            )
        artifact_coordinate_scope = settings.get(
            "historical_singleton_coordinate_solve_scope"
        )
        if artifact_coordinate_scope in {None, ""}:
            artifact_overlay = settings.get(
                "historical_singleton_coordinate_trust_overlay"
            )
            if isinstance(artifact_overlay, Mapping):
                artifact_coordinate_scope = artifact_overlay.get(
                    "coordinate_solve_scope"
                )
        if artifact_coordinate_scope in {None, ""}:
            artifact_coordinate_scope = "phase3_only_v1"
        _compare_setting(
            mismatches,
            name="historical_singleton_coordinate_solve_scope",
            artifact_value=artifact_coordinate_scope,
            current_value=_arg_value(
                args,
                "historical_singleton_coordinate_solve_scope",
                "phase3_only_v1",
            ),
            cast=lambda value: str(value).strip().lower(),
        )
        artifact_pool = settings.get("adapt_pool", adapt.get("pool_type", None))
        if str(artifact_pool).strip().lower() in {"phase3_v1", "phase2_v1", "legacy_v0"}:
            # Some current-json snapshots written before structural resume
            # separated pool and continuation metadata stored the continuation
            # mode in the pool field.  Do not reject an otherwise replayable
            # scaffold on that legacy serialization error.
            artifact_pool = None
        current_pool = _arg_value(args, "adapt_pool", None)
        if current_pool not in {None, ""} and artifact_pool not in {None, ""}:
            _compare_setting(
                mismatches,
                name="adapt_pool",
                artifact_value=artifact_pool,
                current_value=current_pool,
                cast=lambda x: str(x).strip().lower(),
            )
    if mismatches:
        raise ValueError(
            "Resume artifact settings do not match current static HH request: "
            + json.dumps(mismatches[:8], sort_keys=True)
        )
    source_mode = _source_continuation_mode(payload)
    if continuation_mode not in {None, ""} and source_mode not in {None, ""}:
        if str(source_mode).strip().lower() != str(continuation_mode).strip().lower():
            raise ValueError(
                "Resume artifact continuation mode does not match current request: "
                f"{source_mode!r} != {continuation_mode!r}."
            )

    validation = {
        "schema_version": "static_hh_adapt_resume_validation_v1",
        "problem": str(problem),
        "settings_match": True,
        "continuation_mode": source_mode,
        "current_continuation_mode": continuation_mode,
        "powell_coordinate_chart_policy": powell_chart_validation[
            "resolved_policy"
        ],
        "route_profile_conformance": powell_chart_validation[
            "route_profile_conformance"
        ],
        "powell_coordinate_chart_policy_validation": powell_chart_validation,
        "sr_route_profile_request": sr_route_profile_validation[
            "artifact_profile_request"
        ],
        "sr_route_profile_contract_sha256": sr_route_profile_validation[
            "contract_sha256"
        ],
        "sr_route_profile_contract_validation": sr_route_profile_validation,
        "formal_manifold_route_composition": artifact_fm_composition,
        "formal_manifold_route_composition_sha256": (
            None
            if artifact_fm_composition is None
            else str(artifact_fm_composition["sha256"])
        ),
        "formal_manifold_full_profile_match": bool(
            expected_fm_composition is None
            or artifact_fm_composition == expected_fm_composition
        ),
        "runtime_parameter_count": int(theta_runtime.size),
        "logical_parameter_count": int(source.runtime_input.base_layout.logical_parameter_count),
        "selected_term_count": int(len(source.runtime_input.selected_terms)),
        "candidate_pool_complete": bool(source.runtime_input.candidate_pool_source.candidate_pool_complete),
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(validation, context="resume validation")
    return validation


def _normalize_resume_parameterization_mode(value: Any, *, field: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "logical_shared": "logical_shared",
        "logical_shared_v1": "logical_shared",
        "per_pauli_term": "per_pauli_term",
        "per_pauli_term_v1": "per_pauli_term",
    }
    resolved = aliases.get(raw)
    if resolved is None:
        raise ValueError(
            f"Unsupported resume parameterization mode in {field}: {value!r}. "
            "Expected logical_shared or per_pauli_term."
        )
    return str(resolved)


def _explicit_resume_parameterization_modes(
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve every serialized execution-mode alias without guessing.

    Older checkpoints used one of three locations.  Multiple locations are
    allowed only when they resolve to the same execution contract.
    """

    adapt = payload.get("adapt_vqe", {})
    checkpoint = payload.get("checkpoint", {})
    raw_fields: tuple[tuple[str, Any], ...] = (
        (
            "adapt_vqe.parameterization_execution_mode",
            adapt.get("parameterization_execution_mode")
            if isinstance(adapt, Mapping)
            else None,
        ),
        (
            "adapt_vqe.parameterization_mode",
            adapt.get("parameterization_mode")
            if isinstance(adapt, Mapping)
            else None,
        ),
        (
            "checkpoint.parameterization_execution_mode",
            checkpoint.get("parameterization_execution_mode")
            if isinstance(checkpoint, Mapping)
            else None,
        ),
        (
            "checkpoint.parameterization_mode",
            checkpoint.get("parameterization_mode")
            if isinstance(checkpoint, Mapping)
            else None,
        ),
    )
    resolved: dict[str, str] = {}
    for field, value in raw_fields:
        if value is None or value == "":
            continue
        resolved[str(field)] = _normalize_resume_parameterization_mode(
            value,
            field=str(field),
        )
    return resolved


def _replay_resume_in_expected_parameterization(
    *,
    selected_ops: Sequence[AnsatzTerm],
    selected_layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray,
    theta_logical: np.ndarray | None,
    psi_ref: np.ndarray,
    psi_initial: np.ndarray,
    expected_parameterization_mode: str,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    mode = _normalize_resume_parameterization_mode(
        expected_parameterization_mode,
        field="expected_parameterization_mode",
    )
    theta_exec = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if mode == "logical_shared":
        if theta_logical is None:
            raise ValueError(
                "logical_shared resume cannot replay without logical theta."
            )
        theta_exec = np.asarray(theta_logical, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_exec)):
        raise ValueError("Resume theta contains a non-finite value.")

    try:
        executor = CompiledAnsatzExecutor(
            list(selected_ops),
            coefficient_tolerance=float(selected_layout.coefficient_tolerance),
            ignore_identity=bool(selected_layout.ignore_identity),
            sort_terms=(str(selected_layout.term_order).strip().lower() == "sorted"),
            parameterization_mode=mode,
            parameterization_layout=selected_layout,
        )
        replayed = np.asarray(
            executor.prepare_state(
                theta_exec,
                np.asarray(psi_ref, dtype=complex).reshape(-1),
            ),
            dtype=complex,
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Resume scaffold cannot execute under the expected parameterization "
            f"mode {mode!r}: {exc}"
        ) from exc

    expected = np.asarray(psi_initial, dtype=complex).reshape(-1)
    if int(replayed.size) != int(expected.size):
        raise ValueError(
            "Resume expected-mode replay state dimension mismatch: "
            f"replayed={replayed.size}, checkpoint={expected.size}."
        )
    replay_norm = float(np.linalg.norm(replayed))
    checkpoint_norm = float(np.linalg.norm(expected))
    if not (math.isfinite(replay_norm) and math.isfinite(checkpoint_norm)):
        raise ValueError("Resume expected-mode replay produced a non-finite norm.")
    if replay_norm <= float(atol) or checkpoint_norm <= float(atol):
        raise ValueError("Resume expected-mode replay cannot compare a zero-norm state.")

    overlap = complex(np.vdot(expected, replayed))
    alignment = (
        complex(np.exp(-1.0j * np.angle(overlap)))
        if abs(overlap) > float(atol)
        else 1.0 + 0.0j
    )
    replay_error = float(np.linalg.norm(expected - alignment * replayed))
    if not math.isfinite(replay_error) or replay_error > float(atol):
        raise ValueError(
            "Resume scaffold strict expected-mode replay failed up to global "
            f"phase: mode={mode!r}, l2_error={replay_error:.3e}, "
            f"tolerance={float(atol):.3e}."
        )
    return {
        "schema_version": "static_adapt_resume_expected_mode_replay_v1",
        "passed": True,
        "parameterization_mode": str(mode),
        "global_phase_invariant": True,
        "l2_error_up_to_global_phase": float(replay_error),
        "tolerance": float(atol),
        "checkpoint_state_norm": float(checkpoint_norm),
        "replayed_state_norm": float(replay_norm),
        "overlap_abs": float(abs(overlap)),
    }


def match_resume_scaffold_to_pool(
    source: ResumeScaffoldSource,
    *,
    pool: Sequence[AnsatzTerm],
    build_selected_layout: Callable[[list[AnsatzTerm]], AnsatzParameterLayout],
    expected_parameterization_mode: str,
) -> ResumeMatchedScaffold:
    selected_terms = tuple(source.runtime_input.selected_terms)
    if len(selected_terms) == 0:
        raise ValueError("Structural resume requires at least one selected scaffold generator.")
    by_label: dict[str, list[int]] = {}
    for idx, term in enumerate(pool):
        by_label.setdefault(str(term.label), []).append(int(idx))
    selected_ops: list[AnsatzTerm] = []
    selected_pool_indices: list[int] = []
    missing: list[str] = []
    selected_outside_pool_labels: list[str] = []
    for term in selected_terms:
        label = str(term.label)
        matches = by_label.get(label, [])
        if not matches:
            if "::child_set[" in label:
                selected_ops.append(term)
                selected_outside_pool_labels.append(label)
            else:
                missing.append(label)
            continue
        idx = int(matches[0])
        selected_pool_indices.append(idx)
        selected_ops.append(pool[idx])
    if missing:
        raise ValueError(
            "Resume scaffold selected generator(s) are absent from the current pool: "
            + ", ".join(missing[:8])
        )
    selected_layout = build_selected_layout(list(selected_ops))
    artifact_layout_digest = digest_jsonable(serialize_layout(source.runtime_input.base_layout))
    current_layout_digest = digest_jsonable(serialize_layout(selected_layout))
    if artifact_layout_digest != current_layout_digest:
        raise ValueError(
            "Resume scaffold layout does not match current pool reconstruction "
            f"(artifact={artifact_layout_digest}, current={current_layout_digest})."
        )
    theta_runtime = np.asarray(source.runtime_input.theta_runtime, dtype=float).reshape(-1)
    if int(theta_runtime.size) != int(selected_layout.runtime_parameter_count):
        raise ValueError(
            "Resume runtime theta length does not match reconstructed selected layout."
        )
    theta_logical = (
        None
        if source.runtime_input.theta_logical is None
        else np.asarray(source.runtime_input.theta_logical, dtype=float).reshape(-1)
    )
    expected_mode = _normalize_resume_parameterization_mode(
        expected_parameterization_mode,
        field="expected_parameterization_mode",
    )
    source_mode_fields = _explicit_resume_parameterization_modes(source.payload)
    source_modes = set(source_mode_fields.values())
    if len(source_modes) > 1:
        raise ValueError(
            "Resume scaffold has conflicting explicit parameterization modes: "
            + json.dumps(source_mode_fields, sort_keys=True)
        )
    source_mode = next(iter(source_modes), None)
    if source_mode is not None and source_mode != expected_mode:
        raise ValueError(
            "Resume scaffold parameterization mode does not match the current "
            f"route: source={source_mode!r}, expected={expected_mode!r}."
        )

    logical_alias_max_abs_error: float | None = None
    if expected_mode == "logical_shared":
        if theta_logical is None:
            raise ValueError(
                "logical_shared resume requires an explicit logical_optimal_point "
                "vector; projecting an independent runtime vector is not allowed."
            )
        if int(theta_logical.size) != int(selected_layout.logical_parameter_count):
            raise ValueError(
                "logical_shared resume logical theta length does not match the "
                "reconstructed selected layout."
            )
        runtime_alias = np.asarray(
            expand_legacy_logical_theta(theta_logical, selected_layout),
            dtype=float,
        ).reshape(-1)
        logical_alias_max_abs_error = float(
            np.max(np.abs(theta_runtime - runtime_alias))
            if int(theta_runtime.size)
            else 0.0
        )
        if not np.allclose(
            theta_runtime,
            runtime_alias,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError(
                "logical_shared resume requires runtime theta to be a blockwise "
                "alias of logical theta; found max_abs_error="
                f"{logical_alias_max_abs_error:.3e}."
            )

    replay = _replay_resume_in_expected_parameterization(
        selected_ops=selected_ops,
        selected_layout=selected_layout,
        theta_runtime=theta_runtime,
        theta_logical=theta_logical,
        psi_ref=np.asarray(source.runtime_input.psi_ref, dtype=complex).reshape(-1),
        psi_initial=np.asarray(
            source.runtime_input.psi_initial,
            dtype=complex,
        ).reshape(-1),
        expected_parameterization_mode=expected_mode,
    )
    source_mode_inferred = source_mode is None
    validation = {
        "schema_version": "static_hh_adapt_resume_pool_match_v2",
        "selected_pool_indices": [int(x) for x in selected_pool_indices],
        "operator_labels_digest": digest_jsonable([str(op.label) for op in selected_ops]),
        "parameterization_digest": current_layout_digest,
        "theta_runtime_digest": digest_jsonable([float(x) for x in theta_runtime.tolist()]),
        "selected_term_count": int(len(selected_ops)),
        "selected_terms_outside_pool_count": int(len(selected_outside_pool_labels)),
        "selected_terms_outside_pool_reason": (
            None
            if not selected_outside_pool_labels
            else "runtime_split_child_set_terms_are_terminal_scaffold_terms"
        ),
        "selected_terms_outside_pool_labels": [str(x) for x in selected_outside_pool_labels],
        "runtime_parameter_count": int(theta_runtime.size),
        "expected_parameterization_mode": str(expected_mode),
        "source_parameterization_mode": str(source_mode or expected_mode),
        "source_parameterization_mode_fields": dict(source_mode_fields),
        "source_parameterization_mode_inferred": bool(source_mode_inferred),
        "source_parameterization_mode_resolution": (
            "expected_mode_strict_replay_inference"
            if source_mode_inferred
            else "explicit_source_metadata"
        ),
        "logical_runtime_block_alias_checked": bool(
            expected_mode == "logical_shared"
        ),
        "logical_runtime_block_alias_max_abs_error": logical_alias_max_abs_error,
        "strict_expected_mode_replay": replay,
        "no_credentials_serialized": True,
    }
    assert_no_secret_material(validation, context="resume pool match")
    return ResumeMatchedScaffold(
        selected_ops=tuple(selected_ops),
        selected_layout=selected_layout,
        theta_runtime=theta_runtime,
        theta_logical=theta_logical,
        selected_pool_indices=tuple(int(x) for x in selected_pool_indices),
        validation=validation,
    )


def _validate_best_frontier_formal_manifold_runtime(
    *,
    payload: Mapping[str, Any],
    adapt: Mapping[str, Any],
    beam_branch: Mapping[str, Any],
    branch_id: int,
    expected_route_composition: (
        Mapping[str, Any] | FormalManifoldRouteComposition | None
    ),
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    artifact_composition = extract_formal_manifold_route_composition(payload)
    expected_composition = _expected_formal_manifold_route_composition(
        args=None,
        explicit=expected_route_composition,
    )
    if expected_composition is not None:
        if artifact_composition is None:
            raise ValueError(
                "Best-frontier resume artifact lacks the required "
                "Formal-Manifold route composition."
            )
        if artifact_composition != expected_composition:
            raise ValueError(
                "Best-frontier resume Formal-Manifold route composition drifted."
            )
    runtime_raw = beam_branch.get("formal_manifold_runtime_checkpoint")
    if runtime_raw is None:
        if artifact_composition is not None:
            raise ValueError(
                "Best-frontier Formal-Manifold resume lacks branch-local runtime state."
            )
        return None, None, {
            "formal_manifold_active": False,
            "formal_manifold_runtime_checkpoint_digest": None,
            "formal_manifold_route_composition_sha256": None,
        }
    if artifact_composition is None:
        raise ValueError(
            "Best-frontier checkpoint contains Formal-Manifold runtime state "
            "without a route composition."
        )
    if not isinstance(runtime_raw, Mapping):
        raise ValueError(
            "Best-frontier formal_manifold_runtime_checkpoint must be a mapping."
        )
    runtime = dict(runtime_raw)
    if runtime.get("schema") != (
        "formal_manifold_beam_branch_runtime_checkpoint_v1"
    ):
        raise ValueError("Best-frontier Formal-Manifold runtime schema is unsupported.")
    canonical_branch_id = f"beam_branch:{int(branch_id)}"
    if str(runtime.get("branch_id", "")) != canonical_branch_id:
        raise ValueError(
            "Best-frontier Formal-Manifold runtime branch identity disagrees."
        )
    if bool(runtime.get("structural_rollback_supported", False)):
        raise ValueError(
            "Best-frontier Formal-Manifold checkpoint enables structural rollback."
        )
    if str(runtime.get("rollback_scope", "")) != "pending_proposal_only":
        raise ValueError(
            "Best-frontier Formal-Manifold checkpoint rollback scope drifted."
        )
    runtime_composition = FormalManifoldRouteComposition.from_mapping(
        runtime.get("route_composition")
    ).as_dict()
    if runtime_composition != artifact_composition:
        raise ValueError(
            "Best-frontier Formal-Manifold runtime composition disagrees with artifact."
        )
    transaction_raw = runtime.get("transaction_state")
    if not isinstance(transaction_raw, Mapping):
        raise ValueError(
            "Best-frontier Formal-Manifold runtime lacks transaction state."
        )
    transaction = dict(transaction_raw)
    if transaction.get("schema") != "formal_manifold_transaction_state_v1":
        raise ValueError(
            "Best-frontier Formal-Manifold transaction schema is unsupported."
        )
    if str(transaction.get("branch_id", "")) != canonical_branch_id:
        raise ValueError(
            "Best-frontier Formal-Manifold transaction branch identity disagrees."
        )
    if bool(transaction.get("pending", False)):
        raise ValueError(
            "Best-frontier Formal-Manifold checkpoint contains a pending proposal."
        )
    if bool(transaction.get("structural_rollback_supported", False)):
        raise ValueError(
            "Best-frontier Formal-Manifold transaction enables structural rollback."
        )
    if str(transaction.get("rollback_scope", "")) != "pending_proposal_only":
        raise ValueError(
            "Best-frontier Formal-Manifold transaction rollback scope drifted."
        )
    transaction_composition = FormalManifoldRouteComposition.from_mapping(
        transaction.get("route_composition")
    ).as_dict()
    if transaction_composition != artifact_composition:
        raise ValueError(
            "Best-frontier Formal-Manifold transaction composition disagrees."
        )
    config = transaction.get("formal_manifold_config")
    if not isinstance(config, Mapping):
        raise ValueError(
            "Best-frontier Formal-Manifold transaction lacks its full config."
        )
    config_digest = digest_jsonable(config)
    if str(transaction.get("formal_manifold_config_sha256", "")) != config_digest:
        raise ValueError(
            "Best-frontier Formal-Manifold transaction config fingerprint disagrees."
        )
    if str(runtime.get("formal_manifold_config_sha256", "")) != config_digest:
        raise ValueError(
            "Best-frontier Formal-Manifold runtime config fingerprint disagrees."
        )
    warm_raw = runtime.get("warm_state")
    if warm_raw is not None:
        if not isinstance(warm_raw, Mapping):
            raise ValueError(
                "Best-frontier Formal-Manifold warm state must be a mapping."
            )
        warm = dict(warm_raw)
        if warm.get("schema") != "formal_manifold_warm_state_checkpoint_v1":
            raise ValueError(
                "Best-frontier Formal-Manifold warm-state schema is unsupported."
            )
        if str(warm.get("branch_id", "")) != canonical_branch_id:
            raise ValueError(
                "Best-frontier Formal-Manifold warm-state branch identity disagrees."
            )
        warm_composition = FormalManifoldRouteComposition.from_mapping(
            warm.get("route_composition")
        ).as_dict()
        if warm_composition != artifact_composition:
            raise ValueError(
                "Best-frontier Formal-Manifold warm-state composition disagrees."
            )
        if str(warm.get("formal_manifold_config_sha256", "")) != config_digest:
            raise ValueError(
                "Best-frontier Formal-Manifold warm-state config fingerprint disagrees."
            )
        warm_transaction = warm.get("transaction_state")
        if not isinstance(warm_transaction, Mapping):
            raise ValueError(
                "Best-frontier Formal-Manifold warm state lacks transaction counters."
            )
        for field_name in (
            "last_reset_reason",
            "reset_count",
            "commit_count",
            "rollback_count",
            "rollback_scope",
        ):
            if warm_transaction.get(field_name) != transaction.get(field_name):
                raise ValueError(
                    "Best-frontier Formal-Manifold warm/transaction state "
                    f"disagrees for {field_name}."
                )
        top_level_warm = adapt.get("formal_manifold_warm_state_checkpoint")
        if isinstance(top_level_warm, Mapping) and digest_jsonable(
            top_level_warm
        ) != digest_jsonable(warm):
            raise ValueError(
                "Best-frontier Formal-Manifold branch warm state disagrees "
                "with adapt_vqe."
            )
    query_ledger = runtime.get("query_ledger")
    if not isinstance(query_ledger, Mapping) or query_ledger.get("schema") != (
        "formal_manifold_query_primitive_ledger_checkpoint_v1"
    ):
        raise ValueError(
            "Best-frontier Formal-Manifold runtime lacks a valid query ledger."
        )
    validation = {
        "formal_manifold_active": True,
        "formal_manifold_runtime_checkpoint_digest": digest_jsonable(runtime),
        "formal_manifold_route_composition_sha256": str(
            artifact_composition["sha256"]
        ),
        "formal_manifold_config_sha256": config_digest,
        "formal_manifold_branch_id": canonical_branch_id,
        "formal_manifold_pending_proposal": False,
        "formal_manifold_structural_rollback_supported": False,
        "formal_manifold_rollback_scope": "pending_proposal_only",
    }
    return runtime, artifact_composition, validation


def extract_best_frontier_resume_checkpoint(
    source: ResumeScaffoldSource,
    *,
    expected_powell_coordinate_chart_policy: str | None = None,
    expected_route_profile_conformance: str | None = None,
    expected_sr_route_profile_request: str | None = None,
    expected_sr_route_profile_contract: Mapping[str, Any] | None = None,
    expected_sr_route_profile_contract_sha256: str | None = None,
    formal_manifold_route_composition: (
        Mapping[str, Any] | FormalManifoldRouteComposition | None
    ) = None,
) -> ResumeBestFrontierCheckpoint:
    """Validate and extract one complete ``beam_round_done`` winning lineage.

    This seam is intentionally stricter than the general scaffold loader.  It
    accepts only a complete singleton, append-only winning lineage with no
    accepted prune deletion.  Those restrictions let a caller resume the
    preserved best branch without pretending that the discarded beam frontier
    or an unrecorded structural transition can be reconstructed.
    """

    payload = source.payload
    _assert_no_unsupported_modeled_minimum_execution_checkpoint(
        payload,
        context="Best-frontier resume",
    )
    powell_chart_validation = validate_resume_powell_coordinate_chart_policy(
        payload,
        expected_policy=expected_powell_coordinate_chart_policy,
        expected_route_profile_conformance=(
            expected_route_profile_conformance
        ),
        context="Best-frontier resume",
    )
    sr_route_profile_validation = validate_resume_sr_route_profile_contract(
        payload,
        expected_profile_request=expected_sr_route_profile_request,
        expected_contract=expected_sr_route_profile_contract,
        expected_contract_sha256=expected_sr_route_profile_contract_sha256,
        context="Best-frontier resume",
    )
    adapt = _adapt_block(payload)
    checkpoint_raw = payload.get("checkpoint", None)
    if not isinstance(checkpoint_raw, Mapping):
        raise ValueError("Best-frontier resume requires a checkpoint object.")
    checkpoint = dict(checkpoint_raw)

    if str(checkpoint.get("reason", "")) != "beam_round_done":
        raise ValueError(
            "Best-frontier resume requires checkpoint.reason='beam_round_done'."
        )
    if str(checkpoint.get("checkpoint_branch_policy", "")) != "best_frontier_branch":
        raise ValueError(
            "Best-frontier resume requires checkpoint_branch_policy="
            "'best_frontier_branch'."
        )
    if not bool(checkpoint.get("beam_enabled", False)):
        raise ValueError("Best-frontier resume requires a beam-enabled checkpoint.")
    if not bool(adapt.get("partial_checkpoint", False)):
        raise ValueError("Best-frontier resume requires adapt_vqe.partial_checkpoint=true.")
    if not bool(adapt.get("adapt_beam_enabled", False)):
        raise ValueError("Best-frontier resume requires adapt_vqe.adapt_beam_enabled=true.")
    if adapt.get("history_checkpoint_complete", None) is not True:
        raise ValueError(
            "Best-frontier resume requires history_checkpoint_complete=true."
        )

    history_raw = adapt.get("history", None)
    if not isinstance(history_raw, Sequence) or isinstance(
        history_raw, (str, bytes, bytearray)
    ):
        raise ValueError("Best-frontier resume requires adapt_vqe.history array.")
    if any(not isinstance(row, Mapping) for row in history_raw):
        raise ValueError("Best-frontier resume history rows must be JSON objects.")
    history_cleaned_raw, _removed = _drop_obsolete_admission_rollback_state(
        [dict(row) for row in history_raw]
    )
    history = [dict(row) for row in history_cleaned_raw]
    if not history:
        raise ValueError("Best-frontier resume history must be non-empty.")

    def _int_field(
        block: Mapping[str, Any],
        field: str,
        *,
        context: str,
        minimum: int = 0,
    ) -> int:
        try:
            value = int(block.get(field))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume requires integer {context}.{field}."
            ) from None
        if value < int(minimum):
            raise ValueError(
                f"Best-frontier resume requires {context}.{field} >= {minimum}."
            )
        return value

    history_count = _int_field(
        adapt,
        "history_count",
        context="adapt_vqe",
        minimum=1,
    )
    if history_count != len(history):
        raise ValueError(
            "Best-frontier resume history_count does not match complete history length."
        )
    controller_round = _int_field(
        checkpoint,
        "depth",
        context="checkpoint",
        minimum=1,
    )
    if controller_round != history_count:
        raise ValueError(
            "Best-frontier resume checkpoint depth does not match history_count."
        )

    branch_id = _int_field(
        checkpoint,
        "branch_id",
        context="checkpoint",
        minimum=0,
    )
    parent_raw = checkpoint.get("parent_branch_id", None)
    parent_branch_id = None if parent_raw is None else int(parent_raw)
    if _int_field(adapt, "branch_id", context="adapt_vqe") != branch_id:
        raise ValueError("Best-frontier resume branch_id fields disagree.")
    if adapt.get("parent_branch_id", None) != parent_raw:
        raise ValueError("Best-frontier resume parent_branch_id fields disagree.")

    operator_labels_raw = adapt.get("operators", None)
    if not isinstance(operator_labels_raw, Sequence) or isinstance(
        operator_labels_raw, (str, bytes, bytearray)
    ):
        raise ValueError("Best-frontier resume requires adapt_vqe.operators array.")
    operator_labels = tuple(str(label) for label in operator_labels_raw)
    if not operator_labels or any(label == "" for label in operator_labels):
        raise ValueError("Best-frontier resume operator labels must be non-empty.")
    ansatz_depth = _int_field(
        adapt,
        "ansatz_depth",
        context="adapt_vqe",
        minimum=1,
    )
    if ansatz_depth != len(operator_labels):
        raise ValueError(
            "Best-frontier resume ansatz_depth does not match operator count."
        )
    if _int_field(
        checkpoint,
        "ansatz_depth",
        context="checkpoint",
        minimum=1,
    ) != ansatz_depth:
        raise ValueError("Best-frontier resume checkpoint ansatz depth disagrees.")
    if ansatz_depth != history_count:
        raise ValueError(
            "Best-frontier singleton resume requires ansatz depth to equal history count."
        )

    selected_labels: list[str] = []
    previous_branch_id: int | None = None
    for index, row in enumerate(history):
        expected_round = int(index + 1)
        if _int_field(row, "depth", context=f"history[{index}]", minimum=1) != expected_round:
            raise ValueError(
                "Best-frontier resume history depths must be contiguous from one."
            )
        row_branch_id = _int_field(
            row,
            "branch_id",
            context=f"history[{index}]",
            minimum=0,
        )
        row_parent_id = _int_field(
            row,
            "parent_branch_id",
            context=f"history[{index}]",
            minimum=0,
        )
        if previous_branch_id is not None and row_parent_id != previous_branch_id:
            raise ValueError(
                "Best-frontier resume history is not one continuous winning branch."
            )
        previous_branch_id = row_branch_id
        if _int_field(row, "batch_size", context=f"history[{index}]", minimum=1) != 1:
            raise ValueError(
                "Best-frontier resume only supports preserved singleton admissions."
            )
        label = str(row.get("selected_op", ""))
        if label == "":
            raise ValueError(
                f"Best-frontier resume history[{index}].selected_op is missing."
            )
        selected_labels.append(label)
        try:
            selected_position = int(row.get("selected_position"))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume history[{index}].selected_position is missing."
            ) from None
        if selected_position != index:
            raise ValueError(
                "Best-frontier resume requires append-only ordered insertion positions."
            )
        prune = row.get("post_admission_prune", {})
        if not isinstance(prune, Mapping):
            raise ValueError(
                f"Best-frontier resume history[{index}].post_admission_prune is invalid."
            )
        try:
            accepted_prune_count = int(prune.get("accepted_count", 0) or 0)
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume history[{index}] prune count is invalid."
            ) from None
        if accepted_prune_count != 0:
            raise ValueError(
                "Best-frontier resume cannot reconstruct a lineage with accepted prune deletion."
            )
    if previous_branch_id != branch_id:
        raise ValueError(
            "Best-frontier resume last history branch does not match checkpoint branch."
        )
    if tuple(selected_labels) != operator_labels:
        raise ValueError(
            "Best-frontier resume ordered history operators do not match active operators."
        )

    def _finite_vector(value: Any, *, field: str) -> np.ndarray:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError(f"Best-frontier resume requires {field} array.")
        try:
            vector = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(f"Best-frontier resume {field} is not numeric.") from None
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"Best-frontier resume {field} must be finite and non-empty.")
        return vector

    theta_runtime = _finite_vector(
        adapt.get("optimal_point", None),
        field="adapt_vqe.optimal_point",
    )
    theta_logical = _finite_vector(
        adapt.get("logical_optimal_point", None),
        field="adapt_vqe.logical_optimal_point",
    )
    if _int_field(adapt, "num_parameters", context="adapt_vqe", minimum=1) != int(
        theta_runtime.size
    ):
        raise ValueError("Best-frontier resume runtime theta count disagrees.")
    if _int_field(
        adapt,
        "logical_num_parameters",
        context="adapt_vqe",
        minimum=1,
    ) != int(theta_logical.size):
        raise ValueError("Best-frontier resume logical theta count disagrees.")
    if int(theta_logical.size) != ansatz_depth:
        raise ValueError(
            "Best-frontier singleton resume requires one logical theta per operator."
        )
    parameterization = adapt.get("parameterization", None)
    if not isinstance(parameterization, Mapping):
        raise ValueError("Best-frontier resume requires adapt_vqe.parameterization.")
    if _int_field(
        parameterization,
        "runtime_parameter_count",
        context="adapt_vqe.parameterization",
        minimum=1,
    ) != int(theta_runtime.size):
        raise ValueError(
            "Best-frontier resume parameterization runtime count disagrees."
        )
    if _int_field(
        parameterization,
        "logical_operator_count",
        context="adapt_vqe.parameterization",
        minimum=1,
    ) != ansatz_depth:
        raise ValueError(
            "Best-frontier resume parameterization logical count disagrees."
        )

    runtime_labels = tuple(str(term.label) for term in source.runtime_input.selected_terms)
    if runtime_labels != operator_labels:
        raise ValueError(
            "Best-frontier resume runtime-loader operators disagree with the checkpoint."
        )
    runtime_theta = np.asarray(source.runtime_input.theta_runtime, dtype=float).reshape(-1)
    if not np.array_equal(runtime_theta, theta_runtime):
        raise ValueError(
            "Best-frontier resume runtime-loader theta disagrees with the checkpoint."
        )
    runtime_theta_logical = source.runtime_input.theta_logical
    if runtime_theta_logical is None or not np.array_equal(
        np.asarray(runtime_theta_logical, dtype=float).reshape(-1),
        theta_logical,
    ):
        raise ValueError(
            "Best-frontier resume runtime-loader logical theta disagrees with the checkpoint."
        )

    state_digests: dict[str, str] = {}
    state_nq: dict[str, int] = {}
    for state_name, runtime_state in (
        ("initial_state", source.runtime_input.psi_initial),
        ("ansatz_input_state", source.runtime_input.psi_ref),
    ):
        state_manifest = payload.get(state_name, None)
        if not isinstance(state_manifest, Mapping):
            raise ValueError(f"Best-frontier resume requires {state_name} manifest.")
        nq_total = _int_field(
            state_manifest,
            "nq_total",
            context=state_name,
            minimum=1,
        )
        state_nq[state_name] = nq_total
        try:
            norm = float(state_manifest.get("norm"))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume {state_name}.norm is invalid."
            ) from None
        if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-8:
            raise ValueError(
                f"Best-frontier resume {state_name} must be normalized."
            )
        runtime_size = int(np.asarray(runtime_state, dtype=complex).reshape(-1).size)
        if runtime_size != (1 << nq_total):
            raise ValueError(
                f"Best-frontier resume {state_name} dimension disagrees with nq_total."
            )
        state_digests[state_name] = digest_jsonable(state_manifest)
    if state_nq["initial_state"] != state_nq["ansatz_input_state"]:
        raise ValueError("Best-frontier resume state manifests use different qubit counts.")

    trust_raw = adapt.get("route_a_trust_region_state", None)
    if not isinstance(trust_raw, Mapping):
        raise ValueError(
            "Best-frontier resume requires route_a_trust_region_state."
        )
    trust = dict(trust_raw)
    if str(trust.get("schema", "")) != "route_a_trust_region_state_v1":
        raise ValueError("Best-frontier resume trust-state schema is unsupported.")
    trust_update_count = _int_field(
        trust,
        "update_count",
        context="route_a_trust_region_state",
        minimum=1,
    )
    if trust_update_count != controller_round:
        raise ValueError(
            "Best-frontier resume trust update_count does not match controller round."
        )
    for field, allow_zero in (("radius", True), ("reference_radius", False)):
        try:
            value = float(trust.get(field))
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume trust-state {field} is invalid."
            ) from None
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            raise ValueError(
                f"Best-frontier resume trust-state {field} is invalid."
            )
    last_update = trust.get("last_update", None)
    if not isinstance(last_update, Mapping):
        raise ValueError("Best-frontier resume trust state lacks last_update.")
    history_last_update = history[-1].get("route_a_trust_region_update", None)
    if not isinstance(history_last_update, Mapping) or digest_jsonable(
        history_last_update
    ) != digest_jsonable(last_update):
        raise ValueError(
            "Best-frontier resume last history trust update disagrees with trust state."
        )

    beam_replay = adapt.get("beam_replay_telemetry", None)
    if not isinstance(beam_replay, Mapping):
        raise ValueError("Best-frontier resume lacks beam replay telemetry.")
    beam_branch_raw = beam_replay.get("checkpoint_branch", None)
    if not isinstance(beam_branch_raw, Mapping):
        raise ValueError("Best-frontier resume lacks checkpoint branch telemetry.")
    beam_branch = dict(beam_branch_raw)
    if str(beam_branch.get("status", "")) != "frontier" or bool(
        beam_branch.get("terminated", False)
    ):
        raise ValueError("Best-frontier resume checkpoint branch is not live frontier state.")
    for field, expected in (
        ("branch_id", branch_id),
        ("depth_local", controller_round),
        ("history_count", history_count),
        ("ansatz_depth", ansatz_depth),
    ):
        if _int_field(
            beam_branch,
            field,
            context="beam_replay_telemetry.checkpoint_branch",
            minimum=0,
        ) != expected:
            raise ValueError(
                f"Best-frontier resume beam checkpoint {field} disagrees."
            )
    if beam_branch.get("parent_branch_id", None) != parent_raw:
        raise ValueError("Best-frontier resume beam parent branch disagrees.")
    beam_labels = beam_branch.get("operator_labels", None)
    if not isinstance(beam_labels, Sequence) or isinstance(
        beam_labels, (str, bytes, bytearray)
    ) or tuple(str(label) for label in beam_labels) != operator_labels:
        raise ValueError("Best-frontier resume beam operator labels disagree.")
    beam_trust = beam_branch.get("route_a_trust_region_state", None)
    if not isinstance(beam_trust, Mapping) or digest_jsonable(beam_trust) != digest_jsonable(
        trust
    ):
        raise ValueError("Best-frontier resume beam trust state disagrees.")
    (
        formal_manifold_runtime_checkpoint,
        formal_manifold_route_composition_resolved,
        formal_manifold_validation,
    ) = _validate_best_frontier_formal_manifold_runtime(
        payload=payload,
        adapt=adapt,
        beam_branch=beam_branch,
        branch_id=branch_id,
        expected_route_composition=formal_manifold_route_composition,
    )

    for context, tail_raw, tail_count_raw in (
        (
            "adapt_vqe",
            adapt.get("history_tail", None),
            adapt.get("history_tail_count", None),
        ),
        (
            "beam checkpoint",
            beam_branch.get("history_tail", None),
            beam_branch.get("history_tail_count", None),
        ),
    ):
        if not isinstance(tail_raw, Sequence) or isinstance(
            tail_raw, (str, bytes, bytearray)
        ) or any(not isinstance(row, Mapping) for row in tail_raw):
            raise ValueError(f"Best-frontier resume {context} history tail is invalid.")
        tail_cleaned, _tail_removed = _drop_obsolete_admission_rollback_state(
            [dict(row) for row in tail_raw]
        )
        try:
            tail_count = int(tail_count_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Best-frontier resume {context} history tail count is invalid."
            ) from None
        if tail_count != len(tail_cleaned) or tail_count > history_count:
            raise ValueError(
                f"Best-frontier resume {context} history tail count disagrees."
            )
        if digest_jsonable(tail_cleaned) != digest_jsonable(history[-tail_count:]):
            raise ValueError(
                f"Best-frontier resume {context} history tail is not the winning-lineage suffix."
            )

    frontier_prune_key_raw = beam_branch.get("frontier_prune_key", None)
    if not isinstance(frontier_prune_key_raw, Mapping):
        raise ValueError("Best-frontier resume lacks frontier_prune_key.")
    frontier_prune_key = dict(frontier_prune_key_raw)
    prune_labels = frontier_prune_key.get("labels", None)
    if not isinstance(prune_labels, Sequence) or isinstance(
        prune_labels, (str, bytes, bytearray)
    ) or tuple(str(label) for label in prune_labels) != operator_labels:
        raise ValueError("Best-frontier resume frontier-prune operators disagree.")
    theta_round10 = _finite_vector(
        frontier_prune_key.get("theta_round10", None),
        field="frontier_prune_key.theta_round10",
    )
    try:
        theta_round_digits = int(frontier_prune_key.get("theta_round10_digits"))
    except (TypeError, ValueError):
        raise ValueError(
            "Best-frontier resume frontier-prune theta precision is invalid."
        ) from None
    if theta_round10.size != theta_runtime.size or theta_round_digits < 0:
        raise ValueError("Best-frontier resume frontier-prune theta count disagrees.")
    round_tolerance = max(
        np.finfo(float).eps,
        0.51 * (10.0 ** (-theta_round_digits)),
    )
    if not np.allclose(
        theta_round10,
        theta_runtime,
        rtol=0.0,
        atol=round_tolerance,
    ):
        raise ValueError("Best-frontier resume frontier-prune theta disagrees.")

    try:
        source_energy = float(adapt.get("energy"))
        beam_energy = float(beam_branch.get("energy"))
    except (TypeError, ValueError):
        raise ValueError("Best-frontier resume source energy is invalid.") from None
    if not math.isfinite(source_energy) or not math.isfinite(beam_energy):
        raise ValueError("Best-frontier resume source energy must be finite.")
    if abs(source_energy - beam_energy) > 1.0e-12:
        raise ValueError("Best-frontier resume beam energy disagrees with source energy.")

    validation = {
        "schema_version": "static_adapt_best_frontier_resume_checkpoint_v1",
        "checkpoint_reason": "beam_round_done",
        "checkpoint_branch_policy": "best_frontier_branch",
        "history_checkpoint_complete": True,
        "history_count": int(history_count),
        "history_digest": digest_jsonable(history),
        "controller_round": int(controller_round),
        "ansatz_depth": int(ansatz_depth),
        "branch_id": int(branch_id),
        "parent_branch_id": parent_branch_id,
        "operator_labels_digest": digest_jsonable(operator_labels),
        "theta_runtime_digest": digest_jsonable(theta_runtime.tolist()),
        "theta_logical_digest": digest_jsonable(theta_logical.tolist()),
        "initial_state_digest": state_digests["initial_state"],
        "ansatz_input_state_digest": state_digests["ansatz_input_state"],
        "route_a_trust_region_state_digest": digest_jsonable(trust),
        "frontier_prune_key_digest": digest_jsonable(frontier_prune_key),
        "powell_coordinate_chart_policy": powell_chart_validation[
            "resolved_policy"
        ],
        "route_profile_conformance": powell_chart_validation[
            "route_profile_conformance"
        ],
        "powell_coordinate_chart_policy_validation": powell_chart_validation,
        "sr_route_profile_request": sr_route_profile_validation[
            "artifact_profile_request"
        ],
        "sr_route_profile_contract_sha256": sr_route_profile_validation[
            "contract_sha256"
        ],
        "sr_route_profile_contract_validation": sr_route_profile_validation,
        "discarded_frontier_reconstructed": False,
        "lineage_scope": "preserved_best_frontier_branch_only",
        "no_credentials_serialized": True,
    }
    validation.update(formal_manifold_validation)
    assert_no_secret_material(validation, context="best-frontier resume validation")
    return ResumeBestFrontierCheckpoint(
        history=tuple(dict(row) for row in history),
        controller_round=int(controller_round),
        ansatz_depth=int(ansatz_depth),
        branch_id=int(branch_id),
        parent_branch_id=parent_branch_id,
        operator_labels=operator_labels,
        theta_runtime=tuple(float(value) for value in theta_runtime.tolist()),
        theta_logical=tuple(float(value) for value in theta_logical.tolist()),
        route_a_trust_region_state=trust,
        beam_checkpoint_branch=beam_branch,
        frontier_prune_key=frontier_prune_key,
        source_energy=float(source_energy),
        initial_state_digest=state_digests["initial_state"],
        ansatz_input_state_digest=state_digests["ansatz_input_state"],
        powell_coordinate_chart_policy=powell_chart_validation[
            "resolved_policy"
        ],
        route_profile_conformance=powell_chart_validation[
            "route_profile_conformance"
        ],
        sr_route_profile_request=sr_route_profile_validation[
            "artifact_profile_request"
        ],
        sr_route_profile_contract=(
            None
            if sr_route_profile_validation["contract_sha256"] is None
            else dict(
                next(
                    iter(
                        {
                            _path_label(path): _nested_payload_value(payload, path)
                            for path in _SR_ROUTE_PROFILE_CONTRACT_PATHS
                            if isinstance(_nested_payload_value(payload, path), Mapping)
                        }.values()
                    )
                )
            )
        ),
        sr_route_profile_contract_sha256=sr_route_profile_validation[
            "contract_sha256"
        ],
        formal_manifold_runtime_checkpoint=(
            None
            if formal_manifold_runtime_checkpoint is None
            else dict(formal_manifold_runtime_checkpoint)
        ),
        formal_manifold_route_composition=(
            None
            if formal_manifold_route_composition_resolved is None
            else dict(formal_manifold_route_composition_resolved)
        ),
        validation=validation,
    )


def _extract_continuation_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    top = payload.get("continuation", None)
    if isinstance(top, Mapping):
        return dict(top)
    adapt = payload.get("adapt_vqe", None)
    if isinstance(adapt, Mapping) and isinstance(adapt.get("continuation", None), Mapping):
        return dict(adapt.get("continuation", {}))
    return {}


def extract_resume_history(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    continuation = _extract_continuation_block(payload)
    rows = continuation.get("selected_scaffold_history", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    cleaned_rows, _removed = _drop_obsolete_admission_rollback_state(
        [dict(row) for row in rows if isinstance(row, Mapping)]
    )
    return [dict(row) for row in cleaned_rows if isinstance(row, Mapping)]


def extract_resume_optimizer_memory(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    continuation = _extract_continuation_block(payload)
    memory = continuation.get("optimizer_memory", None)
    return dict(memory) if isinstance(memory, Mapping) else None


def run_resume_compile_smoke(
    source: ResumeScaffoldSource,
    *,
    mode: str,
    backend_name: str = "FakeMarrakesh",
    seed_transpiler: int = 7,
    optimization_level: int = 1,
) -> ResumeCompileSmokeResult:
    smoke_mode = str(mode).strip().lower()
    required = bool(smoke_mode == "required")
    if smoke_mode == "off":
        return ResumeCompileSmokeResult(
            required=False,
            executed=False,
            success=False,
            backend_name=str(backend_name),
            compiled_depth=None,
            compiled_size=None,
            compiled_count_2q=None,
            output_json=None,
            error=None,
        )
    try:
        from pipelines.scaffold.adapt_circuit_cost import (
            CompileScoutConfig,
            run_compile_scout,
        )

        cfg = CompileScoutConfig(
            source=ImportedArtifactResolution(
                mode="imported_artifact",
                requested_json=Path(source.artifact_json),
                resolved_json=Path(source.artifact_json),
                source_kind="direct_payload",
                default_subject=False,
            ),
            requested_backend_name=str(backend_name),
            candidate_backends=(str(backend_name),),
            sweep_backends=False,
            seed_transpiler=int(seed_transpiler),
            optimization_level=int(optimization_level),
            output_json=Path(source.artifact_json).with_name(
                f"{Path(source.artifact_json).stem}_resume_compile_smoke.json"
            ),
        )
        payload = run_compile_scout(cfg)
        selected = payload.get("selected_backend", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(selected, Mapping):
            selected = {}
        result = ResumeCompileSmokeResult(
            required=required,
            executed=True,
            success=bool(payload.get("success", False)) if isinstance(payload, Mapping) else False,
            backend_name=(
                str(selected.get("transpile_backend"))
                if selected.get("transpile_backend") not in {None, ""}
                else str(backend_name)
            ),
            compiled_depth=(
                None if selected.get("compiled_depth") is None else int(selected.get("compiled_depth"))
            ),
            compiled_size=(
                None if selected.get("compiled_size") is None else int(selected.get("compiled_size"))
            ),
            compiled_count_2q=(
                None if selected.get("compiled_count_2q") is None else int(selected.get("compiled_count_2q"))
            ),
            output_json=(
                str(payload.get("artifacts", {}).get("output_json"))
                if isinstance(payload.get("artifacts", None), Mapping)
                and payload.get("artifacts", {}).get("output_json") is not None
                else None
            ),
            error=None,
        )
        assert_no_secret_material(result.to_payload(), context="resume compile smoke result")
        return result
    except Exception as exc:
        return ResumeCompileSmokeResult(
            required=required,
            executed=False,
            success=False,
            backend_name=str(backend_name),
            compiled_depth=None,
            compiled_size=None,
            compiled_count_2q=None,
            output_json=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def build_credential_audit() -> dict[str, Any]:
    return {
        "schema_version": "static_hh_adapt_runtime_audit_v1",
        "cli_accepts_credentials": False,
        "environment_serialized": False,
        "runtime_credentials_serialized": False,
        "no_credentials_serialized": True,
    }


__all__ = [
    "ResumeBestFrontierCheckpoint",
    "ResumeCompileSmokeResult",
    "ResumeMatchedScaffold",
    "ResumeScaffoldSource",
    "assert_no_secret_cli_values",
    "assert_no_secret_material",
    "build_credential_audit",
    "build_resume_import_summary",
    "contains_secret_marker",
    "digest_jsonable",
    "extract_best_frontier_resume_checkpoint",
    "extract_formal_manifold_route_composition",
    "extract_resume_history",
    "extract_resume_optimizer_memory",
    "file_sha256",
    "load_static_resume_source",
    "match_resume_scaffold_to_pool",
    "run_resume_compile_smoke",
    "validate_resume_powell_coordinate_chart_policy",
    "validate_resume_sr_route_profile_contract",
    "validate_static_hh_resume_source",
]
