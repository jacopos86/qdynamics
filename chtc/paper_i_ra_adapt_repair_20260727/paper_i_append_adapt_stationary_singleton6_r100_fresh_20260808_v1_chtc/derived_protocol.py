"""Derive one fresh round-100 Append protocol inside the locked round-50 tree."""

from __future__ import annotations

import copy
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from package_contract import (
    ALLOWED_PROTOCOL_DELTA_PATHS,
    SOURCE_BUNDLE_RELATIVE_ROOT,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_sha256,
    load_json_object,
    sha256_file,
)


def _module_is_under(module: ModuleType, root: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if origin is not None:
        try:
            Path(origin).resolve().relative_to(root)
            return True
        except ValueError:
            return False
    locations = getattr(module, "__path__", None)
    if locations is None:
        return True
    for location in locations:
        try:
            Path(location).resolve().relative_to(root)
        except ValueError:
            return False
    return True


def activate_source_root(source_root: Path) -> None:
    """Make the extracted source the sole source of project imports."""

    root = source_root.resolve()
    if not root.is_dir():
        raise PackageContractError(
            f"Extracted source root is unavailable: {root}"
        )
    drifted: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            continue
        if not _module_is_under(module, root):
            drifted.append(name)
    if drifted:
        raise PackageContractError(
            "Project modules were imported before source-lock activation: "
            + ", ".join(sorted(drifted))
        )
    source_text = str(root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        ResolvedProblemReceipt,
    )

    receipt = protocol.problem
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )
    if (
        ResolvedProblemReceipt.from_problem(problem).to_dict()
        != receipt.to_dict()
    ):
        raise PackageContractError(
            "Reconstructed problem drifted from the source protocol."
        )
    return problem


def _scalar_differences(
    before: Any,
    after: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            return [(path, before, after)]
        result: list[tuple[tuple[str | int, ...], Any, Any]] = []
        for key in sorted(before):
            result.extend(
                _scalar_differences(
                    before[key], after[key], path=(*path, str(key))
                )
            )
        return result
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [(path, before, after)]
        result = []
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(
                _scalar_differences(
                    left, right, path=(*path, index)
                )
            )
        return result
    return [] if before == after else [(path, before, after)]


def _path_text(path: tuple[str | int, ...]) -> str:
    return ".".join(str(component) for component in path)


def _replace_path(
    payload: dict[str, Any],
    path: str,
    value: Any,
) -> None:
    components = path.split(".")
    cursor: Any = payload
    for component in components[:-1]:
        if not isinstance(cursor, dict) or component not in cursor:
            raise PackageContractError(
                f"Cannot normalize missing protocol path {path!r}."
            )
        cursor = cursor[component]
    if not isinstance(cursor, dict) or components[-1] not in cursor:
        raise PackageContractError(
            f"Cannot normalize missing protocol path {path!r}."
        )
    cursor[components[-1]] = value


def normalized_protocol_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    for path in ALLOWED_PROTOCOL_DELTA_PATHS:
        _replace_path(normalized, path, "<authorized-horizon-or-output-delta>")
    return normalized


def _source_cell(job: Mapping[str, Any], source_protocol: Any) -> Any:
    from pipelines.static_adapt.ra_adapt.bundles import BundleCellSpec

    return BundleCellSpec(
        cell_id=str(job["source_execution_id"]),
        stage="core",
        regime_id=str(job["regime_id"]),
        nph=int(job["nph"]),
        route_id=str(job["route_id"]),
        algorithm_id=str(source_protocol.algorithm_id),
        selector_family="append_adapt",
        candidate_representation=str(job["candidate_representation"]),
        horizon=SOURCE_HORIZON,
        source_lock_id=str(job["source_lock_id"]),
    )


def _derived_cell(job: Mapping[str, Any], source_protocol: Any) -> Any:
    from pipelines.static_adapt.ra_adapt.bundles import BundleCellSpec

    return BundleCellSpec(
        cell_id=str(job["execution_id"]),
        stage="core",
        regime_id=str(job["regime_id"]),
        nph=int(job["nph"]),
        route_id=str(job["route_id"]),
        algorithm_id=str(source_protocol.algorithm_id),
        selector_family="append_adapt",
        candidate_representation=str(job["candidate_representation"]),
        horizon=TARGET_HORIZON,
        source_lock_id=str(job["source_lock_id"]),
    )


def build_derived_protocol(
    *,
    job: Mapping[str, Any],
    source_root: Path,
    validate_entire_bundle: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(derived_protocol, problem, horizon_delta_row)``."""

    from pipelines.static_adapt.ra_adapt.append import (
        build_resolved_append_protocol,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        _as_protocol_payload,
        _build_request,
        _bundle_protocol_materialization_authority,
        _decorate_protocol_payload,
        _source_lock_refs,
        _validate_protocol_payload,
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )
    from pipelines.static_adapt.sr_snake.contracts import FreshStart

    bundle_root = source_root / SOURCE_BUNDLE_RELATIVE_ROOT
    source_protocol_path = source_root / str(job["source_protocol"]["path"])
    if source_protocol_path.parent.parent != bundle_root:
        raise PackageContractError(
            "Source protocol escaped the locked stationary-core bundle."
        )
    if (
        sha256_file(source_protocol_path)
        != job["source_protocol"]["sha256"]
        or source_protocol_path.stat().st_size
        != int(job["source_protocol"]["size_bytes"])
    ):
        raise PackageContractError("Source protocol byte binding drifted.")
    if validate_entire_bundle:
        source_protocol = load_validated_bundle_protocol(
            source_protocol_path
        )
    else:
        source_payload = load_json_object(
            source_protocol_path, label="source protocol"
        )
        source_protocol = resolved_ra_adapt_protocol_from_mapping(
            source_payload
        )
    if (
        source_protocol.sha256
        != job["source_protocol"]["canonical_sha256"]
        or int(source_protocol.horizon) != SOURCE_HORIZON
        or source_protocol.stopping_rule
        != {"maximum_controller_rounds": SOURCE_HORIZON}
        or not isinstance(source_protocol.request.execution.resume, FreshStart)
    ):
        raise PackageContractError(
            "Source protocol is not the bound fresh round-50 protocol."
        )
    problem = _problem_from_protocol(source_protocol)
    manifest = load_json_object(
        bundle_root / "bundle_manifest.json", label="bundle manifest"
    )
    source_locks = load_json_object(
        bundle_root / "source_locks.json", label="source locks"
    )
    source_cell = _source_cell(job, source_protocol)
    derived_cell = _derived_cell(job, source_protocol)
    source_lock_refs = _source_lock_refs(
        source_locks, cell=source_cell
    )
    source_lock = _mapping(
        source_locks.get("cell_locks"), label="cell source-lock map"
    ).get(str(job["source_lock_id"]))
    source_lock = _mapping(source_lock, label="cell source lock")
    request = _build_request(derived_cell, bundle_dir=bundle_root)
    if not isinstance(request.execution.resume, FreshStart):
        raise PackageContractError(
            "Derived Append protocol attempted to consume a resume input."
        )
    authority_kwargs = {
        "cell": derived_cell,
        "bundle_id": str(source_protocol.bundle_id),
        "bundle_manifest_sha256": str(manifest["sha256"]),
        "source_locks_sha256": str(source_locks["sha256"]),
        "source_lock_refs": source_lock_refs,
        "active_gradient_policy": str(
            source_protocol.active_gradient_policy
        ),
        "resource_weighting_scope": str(
            source_protocol.resource_weighting_scope
        ),
    }
    initial_authority = _bundle_protocol_materialization_authority(
        **authority_kwargs
    )
    resolved = build_resolved_append_protocol(
        problem,
        request,
        materialization_authority=initial_authority,
    )
    payload = _as_protocol_payload(resolved, cell=derived_cell)
    payload = _decorate_protocol_payload(
        payload,
        cell=derived_cell,
        request=request,
        cell_source_lock=source_lock,
        materialization_authority=initial_authority,
    )
    validation_kwargs = {
        "cell": derived_cell,
        "bundle_id": str(source_protocol.bundle_id),
        "bundle_manifest_sha256": str(manifest["sha256"]),
        "active_gradient_policy": str(
            source_protocol.active_gradient_policy
        ),
        "resource_weighting_scope": str(
            source_protocol.resource_weighting_scope
        ),
        "source_lock_refs": source_lock_refs,
        "cell_source_lock": source_lock,
        "source_locks_sha256": str(source_locks["sha256"]),
    }
    accepted_validation_parameters = inspect.signature(
        _validate_protocol_payload
    ).parameters
    _validate_protocol_payload(
        payload,
        **{
            key: value
            for key, value in validation_kwargs.items()
            if key in accepted_validation_parameters
        },
    )
    derived_protocol = resolved_ra_adapt_protocol_from_mapping(payload)
    final_authority = _bundle_protocol_materialization_authority(
        **authority_kwargs,
        protocol_sha256=derived_protocol.sha256,
    )
    derived_protocol = _attach_validated_bundle_protocol_authority(
        derived_protocol, final_authority
    )

    source_payload = source_protocol.to_dict()
    derived_payload = derived_protocol.to_dict()
    differences = _scalar_differences(source_payload, derived_payload)
    changed_paths = sorted(_path_text(path) for path, _, _ in differences)
    if changed_paths != list(ALLOWED_PROTOCOL_DELTA_PATHS):
        raise PackageContractError(
            f"{job['execution_id']} changed forbidden protocol fields: "
            f"{changed_paths}"
        )
    normalized_source = normalized_protocol_payload(source_payload)
    normalized_derived = normalized_protocol_payload(derived_payload)
    if normalized_source != normalized_derived:
        raise PackageContractError(
            f"{job['execution_id']} has a non-horizon settings delta."
        )
    baseline = _mapping(
        derived_payload.get("baseline_consumption"),
        label="derived baseline consumption",
    )
    if (
        baseline.get("status") != "passed"
        or baseline.get("unconsumed_declared_field_paths") != []
        or baseline.get("unapproved_change_ids") != []
    ):
        raise PackageContractError(
            f"{job['execution_id']} has incomplete source consumption."
        )
    changed_values = [
        {
            "path": _path_text(path),
            "source": before,
            "target": after,
        }
        for path, before, after in differences
    ]
    audit_row = {
        "execution_id": str(job["execution_id"]),
        "source_execution_id": str(job["source_execution_id"]),
        "regime_id": str(job["regime_id"]),
        "nph": int(job["nph"]),
        "route_id": str(job["route_id"]),
        "candidate_representation": str(
            job["candidate_representation"]
        ),
        "source_protocol_path": str(job["source_protocol"]["path"]),
        "source_protocol_sha256": str(source_protocol.sha256),
        "derived_protocol_sha256": str(derived_protocol.sha256),
        "source_horizon": SOURCE_HORIZON,
        "target_horizon": TARGET_HORIZON,
        "changed_scalar_paths": changed_paths,
        "changed_values": changed_values,
        "normalized_non_horizon_settings_match": True,
        "normalized_non_horizon_settings_sha256": canonical_sha256(
            normalized_source
        ),
        "fresh_start": True,
        "source_checkpoint_consumed": False,
        "source_result_consumed": False,
        "resume_claimed": False,
        "source_baseline_consumption_status": "passed",
        "unresolved_source_fields": [],
        "fields_added_by_current_defaults": [],
    }
    return derived_protocol, problem, audit_row


__all__ = [
    "activate_source_root",
    "build_derived_protocol",
    "normalized_protocol_payload",
]
