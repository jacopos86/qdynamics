#!/usr/bin/env python3
"""Execute one source-locked core cell or the bounded local P4 dispatch."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tarfile
import tempfile
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    ED_REGIME_NAME_BY_ID,
    EXPECTED_ARTIFACT_ROLES,
    FULL_RUN_SCIENTIFIC_CLOSURE_SCHEMA,
    G11_BOUNDED_REPLAY_DIAGNOSTIC_SCHEMA,
    G11_DIAGNOSTIC_ROUTES,
    JOB_SPEC_SCHEMA,
    P2_RECEIPT_RELATIVE,
    P3_RECEIPT_RELATIVE,
    P4_EXECUTION_ID,
    P4_SMOKE_ROUNDS,
    P4_SMOKE_RESULT_SCHEMA,
    PACKAGE_ID,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    SUBMISSION_AUTHORIZATION_RELATIVE,
    WORKER_RECEIPT_SCHEMA,
    PackageContractError,
    atomic_publish_noreplace,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    load_json_object,
    safe_relative_path,
    sha256_file,
    validate_core_authority,
    validate_submission_authorization,
    validate_user_selection_authority,
    verify_self_digest,
)

GEOMETRY_EXPANSION_TRUST_SOLVE_LIMITATION = {
    "policy": "source_metric_inverse_sqrt_no_overlap_v1",
    "update_reason": (
        "geometry_expansion_no_coordinate_prediction_no_overlap_hold"
    ),
    "endpoint_overlap_query_charge": 0,
    "transaction_failure": (
        "not_applicable_geometry_expansion_without_coordinate_prediction"
    ),
}


def _atomic_write_json_value(path: Path, value: Any) -> None:
    """Atomically stream canonical JSON without materializing encoded bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {path}")
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(raw_temporary)
    encoder = json.JSONEncoder(
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for chunk in encoder.iterencode(value):
                stream.write(chunk)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_publish_noreplace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _safe_extract_source_archive(destination: Path) -> None:
    manifest = load_json_object(
        PACKAGE_DIR / "source_archive_manifest.json",
        label="source archive manifest",
    )
    verify_self_digest(manifest, label="source archive manifest")
    rows = manifest.get("members")
    if not isinstance(rows, list):
        raise PackageContractError("Source manifest has no members.")
    declared = {
        safe_relative_path(row["path"], label="source member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(declared) != len(rows):
        raise PackageContractError("Source manifest duplicates a member.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(PACKAGE_DIR / "source_locked.tar.gz", "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(
                member.name, label="tar member"
            ).as_posix()
            if (
                relative not in declared
                or relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise PackageContractError(
                    f"Unsafe/undeclared source member: {relative}"
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(
                    f"Archive member has no bytes: {relative}"
                )
            with target.open("xb") as stream:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            row = declared[relative]
            if (
                sha256_file(target) != row.get("sha256")
                or target.stat().st_size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"Extracted source member drifted: {relative}"
                )
            observed.add(relative)
    if observed != set(declared):
        raise PackageContractError("Extracted source member set drifted.")


def _module_is_source_locked(
    *, name: str, module: Any, source_root: Path
) -> bool:
    """Return whether a regular or namespace package resolves only in-root."""

    root = source_root.resolve()
    origin = getattr(module, "__file__", None)
    if origin is not None:
        try:
            Path(origin).resolve().relative_to(root)
        except ValueError:
            return False
        return True

    # ``pipelines`` intentionally has no ``__init__.py``.  Treat its
    # namespace search locations as provenance, and reject a namespace merged
    # with any ambient checkout.
    locations = getattr(module, "__path__", None)
    if locations is None:
        return False
    resolved_locations = tuple(Path(item).resolve() for item in locations)
    if not resolved_locations:
        return False
    for location in resolved_locations:
        try:
            location.relative_to(root)
        except ValueError:
            return False
    return True


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            del sys.modules[name]
    retained_path: list[str] = []
    for item in sys.path:
        candidate = Path(item or ".").resolve()
        if candidate == root:
            continue
        # ``pipelines`` is a namespace package.  Any other search root that
        # exposes ``pipelines`` (or the sibling ``src`` tree) would merge the
        # extracted source lock with the ambient checkout.
        if (candidate / "pipelines").exists() or (
            candidate / "src"
        ).exists():
            continue
        retained_path.append(item)
    sys.path[:] = retained_path
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()
    namespace = importlib.import_module("pipelines")
    concrete = importlib.import_module("pipelines.static_adapt.ra_adapt")
    if not _module_is_source_locked(
        name="pipelines", module=namespace, source_root=root
    ) or not _module_is_source_locked(
        name="pipelines.static_adapt.ra_adapt",
        module=concrete,
        source_root=root,
    ):
        raise PackageContractError(
            "Ambient pipelines package masked or merged with the "
            "source-locked checkout."
        )


def _assert_source_locked_imports(source_root: Path) -> None:
    drifted: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            continue
        if not _module_is_source_locked(
            name=name, module=module, source_root=source_root
        ):
            origin = getattr(module, "__file__", None)
            locations = getattr(module, "__path__", None)
            drifted.append(
                f"{name}=origin:{origin!r},namespace:{list(locations or ())!r}"
            )
    if drifted:
        raise PackageContractError(
            "Runtime imported non-source-locked modules: "
            + ", ".join(sorted(drifted))
        )


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
    if ResolvedProblemReceipt.from_problem(problem).to_dict() != receipt.to_dict():
        raise PackageContractError(
            "Reconstructed worker problem drifted from protocol."
        )
    return problem


def _typed_summary(result: Any, *, entrypoint: str) -> Mapping[str, Any]:
    if entrypoint == "run_append_adapt":
        summary = getattr(result, "paper_i_summary", None)
    else:
        run = getattr(result, "run", None)
        summary = getattr(run, "paper_i_summary", None)
    if (
        summary is None
        or not is_dataclass(summary)
        or not callable(getattr(summary, "to_dict", None))
    ):
        raise PackageContractError("Facade result lacks a typed summary.")
    payload = summary.to_dict()
    if not isinstance(payload, Mapping):
        raise PackageContractError("Typed summary did not serialize.")
    return payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def _v9_embedded_summary_matches_typed_summary(
    embedded: Any,
    typed: Mapping[str, Any],
) -> bool:
    """Compare the two summary projections immutable v9 actually emits."""

    if not isinstance(embedded, Mapping):
        return False

    def omit_none(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): omit_none(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, (list, tuple)):
            return [omit_none(item) for item in value]
        return value

    return embedded == omit_none(
        {
            key: value
            for key, value in typed.items()
            if key != "schema"
        }
    )


def _sequence(value: Any, *, label: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise PackageContractError(f"{label} must be a JSON array.")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise PackageContractError(f"{label} must be an integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(f"{label} must be an integer.") from exc
    if resolved != value or resolved < minimum:
        raise PackageContractError(
            f"{label} must be an integer >= {minimum}."
        )
    return resolved


def _finite(value: Any, *, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(f"{label} must be finite.") from exc
    if not math.isfinite(resolved):
        raise PackageContractError(f"{label} must be finite.")
    return resolved


def _artifact_map(
    artifacts: list[dict[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    mapped = {
        str(row.get("role", "")): row
        for row in artifacts
        if isinstance(row, Mapping)
    }
    if (
        len(mapped) != len(EXPECTED_ARTIFACT_ROLES)
        or set(mapped) != set(EXPECTED_ARTIFACT_ROLES)
    ):
        raise PackageContractError(
            "Scientific closure lacks the exact five artifact bindings."
        )
    return mapped


def _validate_stationary_active_gradient_payload(
    payload: Mapping[str, Any],
    *,
    append: bool = False,
) -> dict[str, Any]:
    """Validate the stationary policy evidence serialized by immutable v9.

    Traversal is canonical: object keys are visited in sorted order and array
    positions in index order.  The compact ordered digest therefore binds both
    the location and canonical content of any exact accounting receipt that is
    present without claiming per-round coverage that the v9 result contract
    does not serialize.
    """

    from pipelines.static_adapt.ra_adapt.contracts import PolicyEchoReceipt

    stationary_policy = "stationary_source_response_v1"
    late_weighting = "late_resource_weighting_v1"
    accounting_schema = "phase3_active_gradient_query_accounting_v1"
    ledger_schema = "estimator_call_ledger_v1"
    policy = _mapping(
        payload.get("policy"),
        label="stationary active-gradient top policy",
    )
    policy_indices = _sequence(
        policy.get("active_gradient_indices_acquired"),
        label="stationary active-gradient policy indices",
    )
    policy_charge = _integer(
        policy.get("active_gradient_charge"),
        label="stationary active-gradient policy charge",
    )
    try:
        typed_policy = PolicyEchoReceipt(
            active_gradient_policy=str(
                policy.get("active_gradient_policy", "")
            ),
            resource_weighting_scope=str(
                policy.get("resource_weighting_scope", "")
            ),
            active_gradient_indices_acquired=tuple(
                _integer(
                    value,
                    label="stationary active-gradient policy index",
                )
                for value in policy_indices
            ),
            active_gradient_charge=policy_charge,
        )
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"Stationary active-gradient policy echo is invalid: {exc}"
        ) from exc
    if (
        policy.get("active_gradient_policy") != stationary_policy
        or policy.get("resource_weighting_scope") != late_weighting
        or list(policy_indices) != []
        or policy_charge != 0
        or typed_policy.to_dict() != dict(policy)
    ):
        raise PackageContractError(
            "Stationary active-gradient top policy is not a zero-acquisition "
            "policy."
        )
    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="stationary active-gradient scientific receipts",
    )
    if scientific.get("policy") != policy:
        raise PackageContractError(
            "Scientific policy echo drifted from the typed result policy."
        )
    if append and (
        scientific.get("phase3_solver_invoked") is not False
        or scientific.get("trust_transaction_invoked") is not False
    ):
        raise PackageContractError(
            "Conventional Append did not retain its inert Phase-III policy "
            "boundary."
        )

    def _pointer_child(path: str, token: str | int) -> str:
        encoded = str(token).replace("~", "~0").replace("/", "~1")
        return f"{path}/{encoded}"

    def _is_active_gradient_scope(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        normalized = "".join(
            character.lower() if character.isalnum() else "_"
            for character in value
        )
        return "active_gradient" in normalized

    def _ledger_has_active_gradient_charge(
        value: Mapping[str, Any],
        *,
        path: str,
    ) -> None:
        stack: list[tuple[str, Any]] = [(path, value)]
        while stack:
            current_path, current = stack.pop()
            if isinstance(current, Mapping):
                keys = list(current)
                if any(not isinstance(key, str) for key in keys):
                    raise PackageContractError(
                        "Estimator ledger contains a non-text JSON key."
                    )
                for key in keys:
                    field = str(key)
                    field_normalized = "".join(
                        character.lower()
                        if character.isalnum()
                        else "_"
                        for character in field
                    )
                    item = current[key]
                    if (
                        field in {"consumer_scope", "scope"}
                        and _is_active_gradient_scope(item)
                    ):
                        raise PackageContractError(
                            "Stationary active-gradient estimator-ledger "
                            f"consumer occurrence found at "
                            f"{_pointer_child(current_path, field)}."
                        )
                    if (
                        "active_gradient" in field_normalized
                        and "charge" in field_normalized
                        and _integer(
                            item,
                            label=(
                                "stationary active-gradient estimator-ledger "
                                f"charge at "
                                f"{_pointer_child(current_path, field)}"
                            ),
                        )
                        != 0
                    ):
                        raise PackageContractError(
                            "Stationary active-gradient estimator-ledger "
                            f"charge occurrence found at "
                            f"{_pointer_child(current_path, field)}."
                        )
                for key in reversed(sorted(keys)):
                    stack.append(
                        (
                            _pointer_child(current_path, key),
                            current[key],
                        )
                    )
            elif isinstance(current, (list, tuple)):
                for index in reversed(range(len(current))):
                    stack.append(
                        (
                            _pointer_child(current_path, index),
                            current[index],
                        )
                    )

    policy_projection = {
        "active_gradient_policy": stationary_policy,
        "active_gradient_indices_acquired": [],
        "active_gradient_charge": 0,
    }
    receipt_bindings: list[dict[str, str]] = []
    estimator_ledger_count = 0
    stack: list[tuple[str, Any]] = [("$", payload)]
    while stack:
        path, current = stack.pop()
        if isinstance(current, Mapping):
            keys = list(current)
            if any(not isinstance(key, str) for key in keys):
                raise PackageContractError(
                    "Result payload contains a non-text JSON key."
                )
            if current.get("schema") == accounting_schema:
                indices = _sequence(
                    current.get("active_gradient_indices_acquired"),
                    label=f"stationary active-gradient indices at {path}",
                )
                primitive_ids = _sequence(
                    current.get("primitive_ids"),
                    label=f"stationary active-gradient primitive IDs at {path}",
                )
                newly_charged = _integer(
                    current.get("new_unique_gradients_charged"),
                    label=f"stationary active-gradient unique charge at {path}",
                )
                local_charge = _integer(
                    current.get("active_gradient_charge", 0),
                    label=f"stationary active-gradient charge at {path}",
                )
                if (
                    current.get("active_gradient_policy")
                    != policy_projection["active_gradient_policy"]
                    or list(indices)
                    != policy_projection[
                        "active_gradient_indices_acquired"
                    ]
                    or list(primitive_ids) != []
                    or newly_charged != policy_charge
                    or local_charge != policy_charge
                    or current.get("status")
                    != "not_acquired_stationary_source_protocol"
                    or current.get("component") != "N_grad"
                    or not _is_active_gradient_scope(
                        current.get("consumer_scope")
                    )
                    or _integer(
                        current.get(
                            "deduplicated_or_ledger_disabled_count"
                        ),
                        label=(
                            "stationary active-gradient deduplicated count "
                            f"at {path}"
                        ),
                    )
                    != 0
                    or _integer(
                        current.get("active_coordinate_count"),
                        label=(
                            "stationary active-gradient coordinate count "
                            f"at {path}"
                        ),
                    )
                    < 0
                ):
                    raise PackageContractError(
                        "Nested Phase-III active-gradient accounting drifted "
                        f"from the stationary top policy at {path}."
                    )
                receipt_bindings.append(
                    {
                        "path": path,
                        "canonical_sha256": canonical_sha256(current),
                    }
                )
            if current.get("schema") == ledger_schema:
                estimator_ledger_count += 1
                _ledger_has_active_gradient_charge(current, path=path)
            for key in reversed(sorted(keys)):
                stack.append((_pointer_child(path, key), current[key]))
        elif isinstance(current, (list, tuple)):
            for index in reversed(range(len(current))):
                stack.append(
                    (_pointer_child(path, index), current[index])
                )

    if append and receipt_bindings:
        raise PackageContractError(
            "Conventional Append serialized a Phase-III active-gradient "
            "accounting receipt."
        )

    ordered_digest = canonical_sha256(
        {
            "schema": (
                "paper_i_stationary_active_gradient_accounting_order_v1"
            ),
            "policy": policy_projection,
            "occurrences": receipt_bindings,
        }
    )
    return {
        "schema": "paper_i_stationary_active_gradient_payload_closure_v1",
        **policy_projection,
        "resource_weighting_scope": late_weighting,
        "serialized_policy_echo_sha256": canonical_sha256(policy),
        "serialized_policy_echo_validation": (
            "typed_policy_echo_receipt_exact_v1"
        ),
        "per_round_phase3_accounting_coverage": (
            "not_serialized_by_v9_result_contract"
        ),
        "limitation": (
            "per_round_phase3_active_gradient_accounting_is_not_"
            "serialized_by_immutable_v9"
        ),
        "phase3_active_gradient_accounting_occurrence_count": len(
            receipt_bindings
        ),
        "phase3_active_gradient_accounting_ordered_sha256": ordered_digest,
        "estimator_ledger_count_checked": estimator_ledger_count,
        "active_gradient_estimator_ledger_occurrence_count": 0,
        "status": "passed",
    }


def _accepted_round_count(
    *,
    append: bool,
    payload: Mapping[str, Any],
) -> int:
    if append:
        body = _mapping(
            payload.get("result_payload"), label="Append result payload"
        )
        history = _sequence(
            body.get("history"), label="Append accepted history"
        )
        completed = _integer(
            body.get("controller_rounds_completed"),
            label="Append completed controller rounds",
        )
        if completed != len(history):
            raise PackageContractError(
                "Append history/completed-round count drifted."
            )
        return completed
    run = _mapping(payload.get("run"), label="RA run")
    trajectory = _sequence(
        run.get("accepted_trajectory"), label="RA accepted trajectory"
    )
    transitions = _sequence(
        run.get("accepted_transitions"), label="RA accepted transitions"
    )
    replay = _sequence(
        run.get("scientific_replay"), label="RA scientific replay"
    )
    if len(trajectory) != len(transitions) or len(trajectory) != len(replay):
        raise PackageContractError(
            "RA trajectory/transition/replay counts drifted."
        )
    return len(trajectory)


def _verified_same_cutoff_ed_reference(
    *,
    job: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    source_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash and parse the source-locked ED row used only for reporting."""

    resolver = _mapping(
        source_lock.get("resolver_trace"),
        label="same-cutoff ED resolver trace",
    )
    declared = _mapping(
        resolver.get("same_cutoff_ed_reference"),
        label="same-cutoff ED declaration",
    )
    global_metadata = _mapping(
        authority["global_source_locks"].get("ed_cutoff_reference"),
        label="global ED source lock",
    )
    global_binding = _mapping(
        authority["global_source_files"].get("ed_cutoff_reference"),
        label="global ED source bytes",
    )
    relative = safe_relative_path(
        declared.get("path"), label="same-cutoff ED reference path"
    )
    target = source_root / relative
    if (
        declared.get("path") != global_metadata.get("path")
        or declared.get("path") != global_binding.get("path")
        or declared.get("sha256") != global_metadata.get("sha256")
        or declared.get("sha256") != global_binding.get("sha256")
        or global_metadata.get("verified") is not True
        or not target.is_file()
        or target.is_symlink()
        or sha256_file(target) != declared.get("sha256")
        or target.stat().st_size
        != _integer(
            global_binding.get("size_bytes"),
            label="same-cutoff ED reference size",
        )
    ):
        raise PackageContractError(
            "Same-cutoff ED source bytes drifted from global authority."
        )
    payload = load_json_object(
        target, label="same-cutoff ED reference JSON"
    )
    regimes = _sequence(
        payload.get("regimes"), label="same-cutoff ED regimes"
    )
    regime_name = str(declared.get("regime_name", ""))
    matching_regimes = [
        _mapping(row, label="same-cutoff ED regime row")
        for row in regimes
        if isinstance(row, Mapping)
        and row.get("name") == regime_name
    ]
    if len(matching_regimes) != 1:
        raise PackageContractError(
            "Same-cutoff ED JSON lacks one exact regime row."
        )
    regime = matching_regimes[0]
    cells = _sequence(
        regime.get("cells"), label="same-cutoff ED cells"
    )
    matching_cells = [
        _mapping(row, label="same-cutoff ED cell")
        for row in cells
        if isinstance(row, Mapping)
        and int(row.get("M", -1)) == int(job["nph"])
    ]
    if len(matching_cells) != 1:
        raise PackageContractError(
            "Same-cutoff ED JSON lacks one exact cutoff cell."
        )
    cell = matching_cells[0]
    exact_energy = _finite(
        cell.get("E_ED"), label="same-cutoff ED JSON energy"
    )
    if (
        payload.get("schema")
        != "paper_i_hh_ed_cutoff_reference_six_regime_v1"
        or payload.get("validation", {}).get("status") != "pass"
        or regime_name
        != ED_REGIME_NAME_BY_ID.get(str(job["regime_id"]))
        or int(declared.get("nph", -1)) != int(job["nph"])
        or int(regime.get("working_cutoff", -1)) != int(job["nph"])
        or _finite(
            declared.get("E_ED"),
            label="same-cutoff ED declared energy",
        )
        != exact_energy
        or cell.get("basis_dimension_matches") is not True
    ):
        raise PackageContractError(
            "Same-cutoff ED regime/cutoff/value projection drifted."
        )
    return digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_"
                "verified_same_cutoff_ed_reference_v1"
            ),
            "path": relative.as_posix(),
            "file_sha256": declared["sha256"],
            "file_size_bytes": target.stat().st_size,
            "source_payload_schema": payload["schema"],
            "regime_id": job["regime_id"],
            "regime_name": regime_name,
            "n_ph_work": int(job["nph"]),
            "n_ph_reference": int(cell["M"]),
            "E_ED": exact_energy,
            "cell_projection_sha256": canonical_sha256(cell),
            "reference_role": "same_cutoff_reporting_reference",
            "controller_decision_influence": False,
            "status": "passed",
        }
    )


def _validate_g8_ra_reporting_projection(
    *,
    summary: Mapping[str, Any],
    source_locked_exact_energy: float,
) -> dict[str, Any]:
    """Bind the typed RA summary to the full-precision locked ED reference."""

    provenance = _mapping(
        summary.get("provenance"), label="G8 RA summary provenance"
    )
    typed_summary_exact_energy = _finite(
        provenance.get("exact_same_cutoff_energy"),
        label="G8 RA summary exact energy",
    )
    source_exact_energy = _finite(
        source_locked_exact_energy,
        label="G8 source-locked exact energy",
    )
    absolute_delta = abs(
        typed_summary_exact_energy - source_exact_energy
    )
    relative_tolerance = 0.0
    absolute_tolerance = 1.0e-12
    if not math.isclose(
        typed_summary_exact_energy,
        source_exact_energy,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise PackageContractError(
            "G8 RA reporting projection drifted from the source lock."
        )
    return {
        "source": "paper_i_run_summary_v1.provenance",
        "controller_decision_influence": False,
        "reporting_only": True,
        "typed_summary_exact_same_cutoff_energy": (
            typed_summary_exact_energy
        ),
        "source_locked_exact_same_cutoff_energy": source_exact_energy,
        "absolute_delta": absolute_delta,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "matched_within_tolerance": True,
        "serialized_parameter_limitation": (
            "typed_summary_ed_uses_serialized_protocol_g_ep_while_"
            "locked_reference_ed_uses_full_precision_regime_g_ep_v1"
        ),
    }


def _build_core_g8_attestation(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    source_root: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    controller_replay: Mapping[str, Any],
    finalized_rounds: int,
    package_state: Mapping[str, Any],
    verified_ed_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a core-specific G8 successor after controller finalization."""

    from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
        build_study1_trusted_execution_receipt,
        validate_study1_trusted_execution_receipt,
    )

    trusted = build_study1_trusted_execution_receipt(
        source_root=source_root
    ).to_dict()
    validated = validate_study1_trusted_execution_receipt(
        trusted,
        source_root=source_root,
        reverify_source=True,
    )
    if validated != trusted:
        raise PackageContractError(
            "G8 trusted source/dataflow receipt changed on reverification."
        )
    package_manifest = load_json_object(
        PACKAGE_DIR / "package_manifest.json",
        label="G8 package manifest",
    )
    execution_plan = load_json_object(
        PACKAGE_DIR / "execution_plan.json",
        label="G8 execution plan",
    )
    source_manifest = load_json_object(
        PACKAGE_DIR / "source_archive_manifest.json",
        label="G8 source archive manifest",
    )
    control_plane = load_json_object(
        PACKAGE_DIR / "control_plane_receipt.json",
        label="G8 control-plane receipt",
    )
    for label, receipt in (
        ("G8 package manifest", package_manifest),
        ("G8 execution plan", execution_plan),
        ("G8 source archive manifest", source_manifest),
        ("G8 control-plane receipt", control_plane),
    ):
        verify_self_digest(receipt, label=label)
    source_archive = _mapping(
        source_manifest.get("archive"),
        label="G8 source archive binding",
    )
    if (
        package_manifest.get("sha256")
        != package_state["package_manifest_sha256"]
        or execution_plan.get("sha256")
        != package_state["execution_plan_sha256"]
        or source_archive.get("sha256")
        != package_state["source_archive_sha256"]
        or package_manifest.get("source_archive_manifest_sha256")
        != source_manifest["sha256"]
        or execution_plan.get("source_archive_manifest_sha256")
        != source_manifest["sha256"]
        or package_manifest.get("package_control_plane_sha256")
        != control_plane["sha256"]
        or execution_plan.get("package_control_plane", {}).get(
            "sha256"
        )
        != control_plane["sha256"]
        or job.get("package_control_plane_sha256")
        != control_plane["sha256"]
    ):
        raise PackageContractError(
            "G8 package/source/control-plane authority drifted."
        )
    protocol_path = source_root / safe_relative_path(
        job["protocol"]["path"], label="G8 job protocol"
    )
    loaded_protocol = load_json_object(
        protocol_path, label="G8 source-locked protocol"
    )
    if (
        verify_self_digest(
            loaded_protocol, label="G8 source-locked protocol"
        )
        != job["protocol"]["canonical_sha256"]
        or sha256_file(protocol_path) != job["protocol"]["sha256"]
        or protocol_path.stat().st_size
        != int(job["protocol"]["size_bytes"])
        or loaded_protocol != protocol
        or _mapping(
            payload.get("protocol"), label="G8 result protocol"
        )
        != loaded_protocol
    ):
        raise PackageContractError(
            "G8 result/job/source-locked protocol equality failed."
        )
    request = _mapping(protocol.get("request"), label="G8 protocol request")
    execution = _mapping(
        request.get("execution"), label="G8 request execution"
    )
    request_stop = _mapping(
        execution.get("stop"), label="G8 request stop"
    )
    serialized_stop = _mapping(
        protocol.get("stopping_rule"), label="G8 stopping rule"
    )
    if (
        request_stop.get("exact_ed_target") is not None
        or serialized_stop.get("exact_ed_target") is not None
        or set(request_stop) != {"maximum_controller_rounds"}
        or set(serialized_stop) != {"maximum_controller_rounds"}
    ):
        raise PackageContractError(
            "G8 core protocol permits an online exact-reference target."
        )
    resolver = _mapping(
        source_lock.get("resolver_trace"), label="G8 resolver trace"
    )
    exact = _mapping(
        resolver.get("same_cutoff_ed_reference"),
        label="G8 same-cutoff reference",
    )
    exact_value = _finite(exact.get("E_ED"), label="G8 exact energy")
    exact_value_sha256 = canonical_sha256(
        {
            "role": "same_cutoff_exact_energy_reporting_only",
            "value": exact_value,
        }
    )
    if (
        exact.get("required") is not True
        or exact.get("reference_role")
        != "same_cutoff_reporting_reference"
        or int(exact.get("nph", -1)) != int(job["nph"])
        or exact.get("sha256")
        != protocol["source_locks"]["ed_cutoff_reference_sha256"]
        or verified_ed_reference.get("file_sha256")
        != exact["sha256"]
        or int(
            verified_ed_reference.get("n_ph_reference", -1)
        )
        != int(job["nph"])
        or _finite(
            verified_ed_reference.get("E_ED"),
            label="G8 verified ED energy",
        )
        != exact_value
        or verified_ed_reference.get("status") != "passed"
    ):
        raise PackageContractError(
            "G8 exact-reference source binding drifted."
        )
    append = job["execution_entrypoint"] == "run_append_adapt"
    if not append:
        typed_reporting_projection = (
            _validate_g8_ra_reporting_projection(
                summary=summary,
                source_locked_exact_energy=exact_value,
            )
        )
    else:
        typed_reporting_projection = {
            "source": "core_g8_post_controller_receipt_v1",
            "controller_decision_influence": False,
            "reporting_only": True,
        }
    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="G8 scientific receipts",
    )
    if (
        controller_replay.get("sha256")
        != scientific.get("controller_replay_evidence_sha256")
        or len(
            _sequence(
                controller_replay.get("signed_controller_round_prefixes"),
                label="G8 finalized signed prefixes",
            )
        )
        != finalized_rounds
    ):
        raise PackageContractError(
            "G8 reporting event preceded replay finalization."
        )
    return digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_"
                "exact_reference_isolation_v1"
            ),
            "package_id": PACKAGE_ID,
            "execution_id": job["execution_id"],
            "protocol_sha256": protocol["sha256"],
            "result_protocol_sha256": _mapping(
                payload.get("protocol"), label="G8 result protocol"
            )["sha256"],
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": package_manifest["sha256"],
            "execution_plan_sha256": execution_plan["sha256"],
            "source_archive_sha256": source_archive["sha256"],
            "source_archive_manifest_sha256": source_manifest[
                "sha256"
            ],
            "package_control_plane_sha256": control_plane["sha256"],
            "core_final_receipt_canonical_sha256": job[
                "core_final_receipt_canonical_sha256"
            ],
            "method": "append_adapt" if append else "ra_adapt",
            "controller_consumed_exact_reference": False,
            "reference_usage": (
                "reporting_only_after_controller_finalization_v1"
            ),
            "protocol_exact_target_absent": True,
            "trusted_execution_receipt_sha256": trusted["sha256"],
            "trusted_execution_receipt": trusted,
            "source_dataflow_regression_receipt_sha256": trusted[
                "source_dataflow_regression_receipt_sha256"
            ],
            "controller_instrumentation_sha256": trusted[
                "controller_instrumentation_sha256"
            ],
            "reporting_boundary_sha256": trusted[
                "reporting_boundary_sha256"
            ],
            "ed_reference": {
                "path": exact["path"],
                "file_sha256": exact["sha256"],
                "n_ph_work": int(job["nph"]),
                "n_ph_reference": int(exact["nph"]),
                "exact_reference_value_sha256": exact_value_sha256,
                "verified_source_receipt": dict(
                    verified_ed_reference
                ),
                "verified_source_receipt_sha256": (
                    verified_ed_reference["sha256"]
                ),
            },
            "controller_replay_evidence_sha256": controller_replay[
                "sha256"
            ],
            "exact_reference_events": [
                {
                    "phase": "reporting_after_controller_finalization",
                    "event_id": (
                        "same_cutoff_exact_energy_reporting_projection_v1"
                    ),
                    "method": "append_adapt" if append else "ra_adapt",
                    "finalized_controller_rounds": finalized_rounds,
                    "exact_reference_value_sha256": exact_value_sha256,
                }
            ],
            "typed_reporting_projection": typed_reporting_projection,
            "artifact_bindings": {
                role: {
                    "sha256": artifacts[role]["sha256"],
                    "size_bytes": artifacts[role]["size_bytes"],
                }
                for role in EXPECTED_ARTIFACT_ROLES
            },
            "status": "passed",
        }
    )


def _accounting_components(
    value: Mapping[str, Any],
) -> dict[str, int]:
    source = value.get("components", value)
    raw = _mapping(source, label="estimator accounting components")
    aliases = {
        "N_H_outer": "n_h_outer",
        "N_H_refit": "n_h_refit",
        "N_grad": "n_grad",
        "N_metric": "n_metric",
    }
    return {
        name: _integer(
            raw.get(name, raw.get(alias)),
            label=f"estimator accounting {name}",
        )
        for name, alias in aliases.items()
    }


def _compiled_resource_projection(
    value: Any, *, label: str
) -> dict[str, int]:
    """Project only the named, method-specific Table-I resource mapping."""

    resources = _mapping(value, label=label)
    if resources.get("compile_convention") != (
        "table_i_basis_gate_transpile_v1"
    ):
        raise PackageContractError(
            f"{label} uses a foreign compile convention."
        )
    return {
        "N2q": _integer(
            resources.get(
                "compiled_count_2q_total",
                resources.get("compiled_two_qubit_count"),
            ),
            label=f"{label} compiled two-qubit count",
        ),
        "D2q": _integer(
            resources.get(
                "compiled_depth_2q_total",
                resources.get("compiled_two_qubit_depth"),
            ),
            label=f"{label} compiled two-qubit depth",
        ),
        "Dc": _integer(
            resources.get(
                "compiled_depth_total",
                resources.get("compiled_total_depth"),
            ),
            label=f"{label} compiled total depth",
        ),
    }


def _append_ledger_occurrence_closure(
    *,
    body: Mapping[str, Any],
    accounting: Mapping[str, Any],
    components: Mapping[str, int],
    s_alg: int,
) -> dict[str, Any]:
    """Validate the full in-memory Append occurrence stream for G10."""

    ledger = _mapping(
        body.get("estimator_call_ledger"),
        label="Append estimator-call ledger",
    )
    occurrences = _sequence(
        ledger.get("occurrences"),
        label="Append estimator-call occurrences",
    )
    occurrence_summary = _mapping(
        ledger.get("occurrence_summary"),
        label="Append ledger occurrence summary",
    )
    accounting_occurrence_summary = _mapping(
        accounting.get("occurrence_summary"),
        label="Append accounting occurrence summary",
    )
    if (
        ledger.get("schema") != "estimator_call_ledger_v1"
        or _integer(
            occurrence_summary.get("S_alg"),
            label="Append ledger occurrence S_alg",
        )
        != s_alg
        or _integer(
            accounting_occurrence_summary.get("S_alg"),
            label="Append accounting occurrence S_alg",
        )
        != s_alg
        or _accounting_components(occurrence_summary)
        != dict(components)
        or _accounting_components(accounting_occurrence_summary)
        != dict(components)
        or len(occurrences) != s_alg
    ):
        raise PackageContractError(
            "G10 Append ledger occurrence summary does not close."
        )
    component_counts = {name: 0 for name in components}
    forbidden_paths: list[str] = []
    for index, raw in enumerate(occurrences, start=1):
        row = _mapping(
            raw, label=f"Append estimator occurrence {index}"
        )
        component = str(row.get("component", ""))
        if (
            _integer(
                row.get("sequence"),
                label=f"Append occurrence sequence {index}",
            )
            != index
            or component not in component_counts
        ):
            raise PackageContractError(
                "G10 Append occurrence ordering/component drifted."
            )
        component_counts[component] += 1
        # Append is the conventional unwhitened comparator.  Its occurrence
        # stream may not smuggle a whitening/metric-inverse consumer label or
        # branch into otherwise closed component totals.
        for field in ("consumer_scope", "branch_id", "primitive_id"):
            value = row.get(field)
            if value is None:
                continue
            text = str(value).lower()
            if (
                "whiten" in text
                or "metric_inverse_sqrt" in text
                or "supported_metric_inverse" in text
            ):
                forbidden_paths.append(f"{index}.{field}")
    if component_counts != dict(components) or forbidden_paths:
        raise PackageContractError(
            "G10 Append occurrence stream contains forbidden whitening "
            "or component-count drift."
        )
    return {
        "schema": ledger["schema"],
        "occurrence_count": len(occurrences),
        "occurrence_summary_sha256": canonical_sha256(
            occurrence_summary
        ),
        "forbidden_whitening_occurrence_paths": [],
    }


def _validate_g11_bounded_diagnostic(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    primary_replay: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Revalidate the selected bounded executions and their exact sidecars."""

    from pipelines.static_adapt.estimator_call_ledger import (
        EstimatorCallLedger,
    )
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        bounded_prefix_replay_identity,
        compare_bounded_controller_replays,
        validate_controller_replay_evidence,
    )

    contract = _mapping(
        job.get("g11_bounded_replay_diagnostic"),
        label="G11 job diagnostic contract",
    )
    selected = job["route_id"] in G11_DIAGNOSTIC_ROUTES
    expected_contract = {
        "selected": selected,
        "run_class": (
            "bounded_nonpaper_diagnostic_v1"
            if selected
            else "not_selected_v1"
        ),
        "independent_replay_rounds": 2 if selected else 0,
        "ra_resume_prefix_rounds": (
            1 if job["route_id"] == "ra_macro_append_only" else 0
        ),
        "ra_resumed_rounds": (
            2 if job["route_id"] == "ra_macro_append_only" else 0
        ),
        "append_resume_boundary": (
            "authenticated_reconstruction_only_v1"
            if job["route_id"] == "append_macro"
            else "not_applicable"
        ),
        "paper_facing_result_allowed": False,
    }
    verify_self_digest(diagnostic, label="G11 in-job diagnostic")
    if (
        dict(contract) != expected_contract
        or diagnostic.get("schema")
        != G11_BOUNDED_REPLAY_DIAGNOSTIC_SCHEMA
        or diagnostic.get("package_id") != PACKAGE_ID
        or diagnostic.get("execution_id") != job["execution_id"]
        or diagnostic.get("regime_id") != job["regime_id"]
        or diagnostic.get("selected") is not selected
        or diagnostic.get("paper_facing_result_allowed") is not False
        or diagnostic.get("status")
        != ("passed" if selected else "not_selected")
    ):
        raise PackageContractError(
            f"G11 bounded diagnostic contract drifted: "
            f"{job['execution_id']}."
        )
    bindings = _sequence(
        diagnostic.get("artifact_bindings"),
        label="G11 diagnostic artifact bindings",
    )
    if not selected:
        if bindings or "evidence" in diagnostic:
            raise PackageContractError(
                "Unselected G11 diagnostic carries execution evidence."
            )
        return {
            "selected": False,
            "status": "not_selected",
            "artifact_count": 0,
        }

    append = job["execution_entrypoint"] == "run_append_adapt"
    expected_names = (
        {
            "g11_diagnostic/independent_primary.checkpoint.json",
            "g11_diagnostic/independent_primary.ledger.json",
            "g11_diagnostic/independent_replay.checkpoint.json",
            "g11_diagnostic/independent_replay.ledger.json",
        }
        if append
        else {
            "g11_diagnostic/independent.checkpoint.json",
            "g11_diagnostic/independent.ledger.json",
            "g11_diagnostic/resume_prefix.checkpoint.json",
            "g11_diagnostic/resume_prefix.ledger.json",
            "g11_diagnostic/resumed.checkpoint.json",
            "g11_diagnostic/resumed.ledger.json",
        }
    )
    by_path: dict[str, Mapping[str, Any]] = {}
    for raw in bindings:
        row = _mapping(raw, label="G11 diagnostic artifact binding")
        relative = safe_relative_path(
            row.get("path"), label="G11 diagnostic artifact"
        ).as_posix()
        if relative in by_path or set(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise PackageContractError(
                "G11 diagnostic artifact binding is ambiguous."
            )
        target = output_root / relative
        if (
            not target.is_file()
            or target.is_symlink()
            or sha256_file(target) != row.get("sha256")
            or target.stat().st_size
            != _integer(
                row.get("size_bytes"),
                label="G11 diagnostic artifact size",
            )
        ):
            raise PackageContractError(
                f"G11 diagnostic artifact bytes drifted: {relative}."
            )
        by_path[relative] = row
    if set(by_path) != expected_names:
        raise PackageContractError(
            "G11 diagnostic artifact set is not method-exact."
        )

    evidence = _mapping(
        diagnostic.get("evidence"), label="G11 diagnostic evidence"
    )
    if (
        diagnostic.get("run_class")
        != "bounded_nonpaper_diagnostic_v1"
        or evidence.get("final_protocol_sha256")
        != protocol["sha256"]
        or evidence.get("method_family")
        != ("append_adapt" if append else "ra_adapt")
        or evidence.get("primary_full_replay_evidence_sha256")
        != primary_replay["sha256"]
    ):
        raise PackageContractError(
            "G11 diagnostic primary/full-protocol binding drifted."
        )

    def _validated_replay(name: str) -> dict[str, Any]:
        try:
            return validate_controller_replay_evidence(
                _mapping(evidence.get(name), label=f"G11 {name}")
            )
        except (TypeError, ValueError) as exc:
            raise PackageContractError(
                f"G11 embedded replay {name} failed: {exc}"
            ) from exc

    def _prefix_count(value: Mapping[str, Any]) -> int:
        return len(
            _sequence(
                value.get("signed_controller_round_prefixes"),
                label="G11 replay signed prefixes",
            )
        )

    if append:
        first = _validated_replay("first_replay_evidence")
        second = _validated_replay("second_replay_evidence")
        comparison = _mapping(
            evidence.get("bounded_replay_comparison"),
            label="G11 Append replay comparison",
        )
        verify_self_digest(
            comparison, label="G11 Append replay comparison"
        )
        rebuilt = compare_bounded_controller_replays(
            first, second, controller_round=2
        )
        primary_prefix = bounded_prefix_replay_identity(
            primary_replay, controller_round=2
        )
        bounded_prefixes = {
            bounded_prefix_replay_identity(
                replay, controller_round=2
            )
            for replay in (first, second)
        }
        if (
            rebuilt != comparison
            or _prefix_count(first) != 2
            or _prefix_count(second) != 2
            or len(bounded_prefixes | {primary_prefix}) != 1
            or evidence.get("primary_prefix_replay_identity_sha256")
            != primary_prefix
            or evidence.get("bounded_prefix_replay_identity_sha256")
            != primary_prefix
            or _integer(
                evidence.get("independent_controller_rounds"),
                label="G11 Append bounded rounds",
            )
            != 2
            or evidence.get("public_resume_execution_supported")
            is not False
            or evidence.get("reconstruction_fields_complete")
            is not True
        ):
            raise PackageContractError(
                "G11 Append bounded replay linkage drifted."
            )
        for stem, replay in (
            ("independent_primary", first),
            ("independent_replay", second),
        ):
            boundary = _mapping(
                replay.get("resume_sidecar_closure"),
                label=f"G11 Append {stem} reconstruction boundary",
            )
            checkpoint = load_json_object(
                output_root
                / f"g11_diagnostic/{stem}.checkpoint.json",
                label=f"G11 Append {stem} checkpoint",
            )
            verify_self_digest(
                checkpoint, label=f"G11 Append {stem} checkpoint"
            )
            ledger = load_json_object(
                output_root / f"g11_diagnostic/{stem}.ledger.json",
                label=f"G11 Append {stem} ledger",
            )
            try:
                restored = EstimatorCallLedger.from_payload(
                    ledger
                ).to_payload()
            except (TypeError, ValueError) as exc:
                raise PackageContractError(
                    f"G11 Append {stem} ledger failed: {exc}"
                ) from exc
            if (
                boundary.get("resume_mode")
                != "authenticated_reconstruction_only_v1"
                or boundary.get(
                    "public_resume_execution_supported"
                )
                is not False
                or boundary.get("reconstruction_fields_complete")
                is not True
                or checkpoint.get("controller_rounds_completed") != 2
                or checkpoint.get(
                    "controller_replay_evidence_sha256"
                )
                != replay["sha256"]
                or checkpoint.get(
                    "public_resume_execution_supported"
                )
                is not False
                or checkpoint.get("reconstruction_fields_complete")
                is not True
                or restored != ledger
            ):
                raise PackageContractError(
                    f"G11 Append {stem} reconstruction sidecars "
                    "drifted."
                )
    else:
        independent = _validated_replay(
            "independent_replay_evidence"
        )
        prefix = _validated_replay("prefix_replay_evidence")
        resumed = _validated_replay("resumed_replay_evidence")
        primary_comparison = _mapping(
            evidence.get("primary_bounded_comparison"),
            label="G11 RA primary comparison",
        )
        resume_comparison = _mapping(
            evidence.get("resume_prefix_comparison"),
            label="G11 RA resume-prefix comparison",
        )
        resumed_comparison = _mapping(
            evidence.get("resumed_bounded_comparison"),
            label="G11 RA resumed comparison",
        )
        for label, comparison in (
            ("primary", primary_comparison),
            ("resume prefix", resume_comparison),
            ("resumed", resumed_comparison),
        ):
            verify_self_digest(
                comparison, label=f"G11 RA {label} comparison"
            )
        if (
            compare_bounded_controller_replays(
                primary_replay,
                independent,
                controller_round=2,
            )
            != primary_comparison
            or compare_bounded_controller_replays(
                prefix, resumed, controller_round=1
            )
            != resume_comparison
            or compare_bounded_controller_replays(
                independent, resumed, controller_round=2
            )
            != resumed_comparison
            or _prefix_count(independent) != 2
            or _prefix_count(prefix) != 1
            or _prefix_count(resumed) != 2
            or any(
                replay.get("protocol_sha256") != protocol["sha256"]
                for replay in (independent, prefix, resumed)
            )
            or evidence.get("authenticated_resume_performed")
            is not True
        ):
            raise PackageContractError(
                "G11 RA bounded replay/resume linkage drifted."
            )
        resume_input = _mapping(
            evidence.get("authenticated_resume_input"),
            label="G11 RA authenticated resume input",
        )
        resume_input_binding = by_path[
            "g11_diagnostic/resume_prefix.checkpoint.json"
        ]
        resumed_boundary = _mapping(
            resumed.get("resume_sidecar_closure"),
            label="G11 RA resumed sidecar closure",
        )
        if (
            resume_input
            != {
                "path": resume_input_binding["path"],
                "sha256": resume_input_binding["sha256"],
                "size_bytes": resume_input_binding["size_bytes"],
            }
            or resumed_boundary.get("resume_mode")
            != "canonical_accepted_state_resume_v1"
            or resumed_boundary.get(
                "public_resume_execution_supported"
            )
            is not True
            or resumed_boundary.get(
                "authentication_binding_complete"
            )
            is not True
        ):
            raise PackageContractError(
                "G11 RA authenticated resume binding drifted."
            )
        for stem in ("independent", "resume_prefix", "resumed"):
            ledger = load_json_object(
                output_root / f"g11_diagnostic/{stem}.ledger.json",
                label=f"G11 RA {stem} ledger sidecar",
            )
            nested = _mapping(
                ledger.get("ledger"),
                label=f"G11 RA {stem} nested ledger",
            )
            try:
                restored = EstimatorCallLedger.from_payload(
                    nested
                ).to_payload()
            except (TypeError, ValueError) as exc:
                raise PackageContractError(
                    f"G11 RA {stem} ledger failed: {exc}"
                ) from exc
            accounting = _mapping(
                ledger.get("accounting"),
                label=f"G11 RA {stem} sidecar accounting",
            )
            if (
                ledger.get("schema")
                != "paper_i_estimator_call_ledger_sidecar_v2"
                or ledger.get("adapt_success") is not True
                or ledger.get("adapt_error") not in (None, "")
                or accounting.get("complete") is not True
                or accounting.get("exact_blockers") != []
                or restored != nested
            ):
                raise PackageContractError(
                    f"G11 RA {stem} estimator sidecar drifted."
                )
    return {
        "selected": True,
        "status": "passed",
        "artifact_count": len(bindings),
        "artifact_bindings_sha256": canonical_sha256(bindings),
        "diagnostic_sha256": diagnostic["sha256"],
    }


def _validate_g3_pool_construction_gate(
    *,
    job: Mapping[str, Any],
    result_parent: Mapping[str, Any],
    result_pool: Mapping[str, Any],
    append: bool,
) -> dict[str, Mapping[str, Any]]:
    """Bind one RA/Append result to the exact P2 pool construction proof."""

    p2 = load_json_object(
        PACKAGE_DIR / P2_RECEIPT_RELATIVE, label="G1-G3 P2 receipt"
    )
    verify_self_digest(p2, label="G1-G3 P2 receipt")
    pool_proof = _mapping(
        p2.get("six_regime_pool_construction_proof"),
        label="G3 six-regime pool/construction proof",
    )
    if (
        verify_self_digest(
            pool_proof,
            label="G3 six-regime pool/construction proof",
        )
        != p2.get("six_regime_pool_construction_proof_sha256")
    ):
        raise PackageContractError("G3 P2 proof binding drifted.")
    proof_rows = _sequence(
        pool_proof.get("rows"), label="G3 proof rows"
    )
    matching_proof_rows = [
        _mapping(row, label="G3 regime proof row")
        for row in proof_rows
        if isinstance(row, Mapping)
        and row.get("regime_id") == job["regime_id"]
        and int(row.get("nph", -1)) == int(job["nph"])
    ]
    if len(matching_proof_rows) != 1:
        raise PackageContractError(
            f"G3 lacks one exact regime proof: {job['execution_id']}."
        )
    pool_proof_row = matching_proof_rows[0]
    verify_self_digest(
        pool_proof_row,
        label=f"G3 {job['regime_id']} proof row",
    )
    expected_parent_projection = _mapping(
        (
            pool_proof_row["parent_inventory"]
            if job["candidate_representation"]
            == "macro_generator_v1"
            else pool_proof_row["singleton_parent_inventory"]
        ),
        label="G3 expected parent projection",
    )
    expected_executable_projection = _mapping(
        (
            pool_proof_row["macro_coefficient_pool"]
            if job["candidate_representation"]
            == "macro_generator_v1"
            else pool_proof_row["singleton_append_global_pool"]
            if append
            else pool_proof_row["singleton_parent_inventory"]
        ),
        label="G3 expected executable projection",
    )

    def _pool_matches(
        observed: Mapping[str, Any],
        expected_projection: Mapping[str, Any],
    ) -> bool:
        return (
            int(observed.get("count", -1))
            == int(expected_projection.get("count", -2))
            and observed.get("ordered_labels_sha256")
            == expected_projection.get("ordered_labels_sha256")
            and observed.get("ordered_pool_sha256")
            == expected_projection.get("ordered_pool_sha256")
        )

    if (
        not _pool_matches(result_parent, expected_parent_projection)
        or not _pool_matches(result_pool, expected_executable_projection)
        or pool_proof_row.get("ra_append_macro_pool_equal")
        is not True
        or pool_proof_row.get("ra_append_singleton_parent_equal")
        is not True
        or (
            job["candidate_representation"]
            == "single_pauli_word_v1"
            and _mapping(
                pool_proof_row.get(
                    "singleton_construction_equivalence"
                ),
                label="G3 singleton construction equivalence",
            ).get("status")
            != "passed"
        )
    ):
        raise PackageContractError(
            f"G3 pool/construction proof failed: "
            f"{job['execution_id']}."
        )
    return {
        "p2": p2,
        "pool_proof": pool_proof,
        "pool_proof_row": pool_proof_row,
    }


def _validate_g6_ra_round(
    *,
    job_id: str,
    index: int,
    raw: Mapping[str, Any],
    raw_replay: Mapping[str, Any],
) -> None:
    """Validate one existing G6 RA receipt without inventing missing proof."""

    row = _mapping(raw, label=f"RA G6 round {index}")
    replay = _mapping(raw_replay, label=f"RA G6 replay {index}")
    support = _mapping(
        row.get("retained_support"), label="RA G6 support"
    )
    stabilization = _mapping(
        row.get("phase3_stabilization"),
        label="RA G6 stabilization",
    )
    kappa = _finite(
        stabilization.get("kappa_stabilization_shift"),
        label="RA G6 kappa",
    )
    boundary = _finite(
        stabilization.get("trust_boundary_multiplier_lambda"),
        label="RA G6 lambda",
    )
    total = _finite(
        stabilization.get("total_metric_multiplier_mu"),
        label="RA G6 mu",
    )

    raw_trust = row.get("source_gram_no_overlap_trust")
    trust: Mapping[str, Any] | None
    if raw_trust is None:
        replay_trust = replay.get("trust_solve")
        if (
            "source_gram_no_overlap_trust" not in row
            or not isinstance(replay_trust, Mapping)
            or canonical_json_bytes(dict(replay_trust))
            != canonical_json_bytes(
                GEOMETRY_EXPANSION_TRUST_SOLVE_LIMITATION
            )
        ):
            raise PackageContractError(
                f"G6 source-Gram trust limitation failed: "
                f"{job_id} round {index}."
            )
        trust = None
    else:
        trust = _mapping(raw_trust, label="RA G6 trust")

    if (
        _finite(
            support.get("rank_relative_tolerance"),
            label="RA G6 support tolerance",
        )
        != 1.0e-6
        or _finite(
            support.get("metric_regularization"),
            label="RA G6 Phase-III support ridge",
        )
        != 0.0
        or stabilization.get("metric_whitening_active") is not False
        or stabilization.get("metric_inverse_sqrt_constructed") is not False
        or not math.isclose(
            total,
            kappa + boundary,
            rel_tol=0.0,
            abs_tol=128.0 * math.ulp(max(1.0, abs(total))),
        )
        or bool(stabilization.get("trust_boundary_active"))
        != bool(boundary > 0.0)
        or (
            trust is not None
            and (
                trust.get("supported_metric_whitening_active") is not False
                or trust.get(
                    "supported_metric_inverse_sqrt_constructed"
                )
                is not False
                or _integer(
                    trust.get("endpoint_overlap_query_charge"),
                    label="RA G6 endpoint charge",
                )
                != 0
            )
        )
    ):
        raise PackageContractError(
            f"G6 Phase-III integrity failed: {job_id} round {index}."
        )


def _validate_core_gates(
    *,
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    source_root: Path,
    authority: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
    sidecars: Mapping[str, Mapping[str, Any]],
    g11_diagnostic: Mapping[str, Any],
    output_root: Path,
    package_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate G1-G13 for one exact 50-round full-core result."""

    append = job["execution_entrypoint"] == "run_append_adapt"
    protocol = _mapping(payload.get("protocol"), label="result protocol")
    verify_self_digest(protocol, label="result protocol")
    artifact_by_role = _artifact_map(artifacts)
    source_lock = _mapping(
        authority["source_lock_cells"].get(job["source_lock_id"]),
        label="core source-lock cell",
    )
    verify_self_digest(source_lock, label="core source-lock cell")
    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="result scientific receipts",
    )
    result_parent = _mapping(
        payload.get("parent_inventory"), label="result parent inventory"
    )
    result_pool = _mapping(
        payload.get("executable_pool"), label="result executable pool"
    )
    if (
        protocol.get("sha256")
        != job["protocol"]["canonical_sha256"]
        or protocol.get("source_locks", {}).get(
            "cell_source_lock_sha256"
        )
        != source_lock["sha256"]
        or source_lock.get("regime_id") != job["regime_id"]
        or int(source_lock.get("nph", -1)) != int(job["nph"])
        or source_lock.get("route_id") != job["route_id"]
        or int(protocol["problem"].get("n_ph_max", -1))
        != int(job["nph"])
        or result_parent != protocol.get("parent_inventory")
        or result_pool != protocol.get("executable_pool")
    ):
        raise PackageContractError(
            f"G1/G3 source/protocol/pool equality failed: "
            f"{job['execution_id']}."
        )
    g3_authority = _validate_g3_pool_construction_gate(
        job=job,
        result_parent=result_parent,
        result_pool=result_pool,
        append=append,
    )
    p2 = g3_authority["p2"]
    pool_proof = g3_authority["pool_proof"]
    pool_proof_row = g3_authority["pool_proof_row"]
    verified_ed_reference = _verified_same_cutoff_ed_reference(
        job=job,
        source_lock=source_lock,
        source_root=source_root,
        authority=authority,
    )
    completed = _accepted_round_count(append=append, payload=payload)
    if (
        completed != 50
        or int(protocol.get("horizon", -1)) != 50
        or protocol.get("stopping_rule")
        != {"maximum_controller_rounds": 50}
        or str(protocol.get("optimizer", "")).lower() != "powell"
        or int(protocol.get("optimizer_maxiter", -1)) != 200
        or protocol.get("seeds") != {"adapt": 7, "transpiler": 7}
    ):
        raise PackageContractError(
            f"Full-horizon protocol/result closure failed: "
            f"{job['execution_id']}."
        )
    if append:
        body = _mapping(
            payload.get("result_payload"), label="Append result payload"
        )
        if (
            body.get("stop_reason") != "maximum_controller_rounds"
            or summary.get("schema")
            != "paper_i_append_run_summary_v1"
            or int(summary.get("controller_rounds_completed", -1)) != 50
            or int(summary.get("protocol_horizon", -1)) != 50
            or summary.get("stop_reason")
            != "maximum_controller_rounds"
            or summary.get("protocol_sha256") != protocol["sha256"]
            or scientific.get("paper_i_append_run_summary") != summary
        ):
            raise PackageContractError(
                f"Append summary/stop closure failed: "
                f"{job['execution_id']}."
            )
    else:
        run = _mapping(payload.get("run"), label="RA run")
        stop = _mapping(run.get("stop"), label="RA stop")
        if (
            stop.get("primary_reason") != "maximum_controller_rounds"
            or int(stop.get("completed_controller_rounds", -1)) != 50
            or int(stop.get("accepted_operator_count", -1)) != 50
            or summary.get("schema") != "paper_i_run_summary_v1"
            or int(summary.get("available_controller_rounds", -1)) != 50
            or not _v9_embedded_summary_matches_typed_summary(
                run.get("paper_i_summary"),
                summary,
            )
        ):
            raise PackageContractError(
                f"RA summary/stop closure failed: "
                f"{job['execution_id']}."
            )

    resolver = _mapping(
        source_lock.get("resolver_trace"), label="source-lock resolver trace"
    )
    exact = _mapping(
        resolver.get("same_cutoff_ed_reference"),
        label="same-cutoff ED reference",
    )
    if (
        exact.get("required") is not True
        or int(exact.get("nph", -1)) != int(job["nph"])
        or exact.get("reference_role")
        != "same_cutoff_reporting_reference"
        or exact.get("sha256")
        != protocol["source_locks"]["ed_cutoff_reference_sha256"]
        or verified_ed_reference.get("file_sha256")
        != exact.get("sha256")
        or int(
            verified_ed_reference.get("n_ph_reference", -1)
        )
        != int(job["nph"])
        or _finite(
            verified_ed_reference.get("E_ED"),
            label="G2 verified same-cutoff energy",
        )
        != _finite(exact.get("E_ED"), label="G2 declared energy")
        or verified_ed_reference.get("status") != "passed"
        or protocol["problem"].get("exact_target_label")
        != "exact_ground_energy_sector_hh"
    ):
        raise PackageContractError(
            f"G2 same-cutoff identity failed: {job['execution_id']}."
        )

    expected_refit = (
        "native_v1" if append else "supported_fs_whitened_fixed_v1"
    )
    if (
        protocol.get("derivative_chart_id")
        != "exact_ordered_insertion_zero_angle_v1"
        or protocol.get("accepted_refit_coordinate_chart")
        != expected_refit
        or scientific.get("accepted_refit_coordinate_chart")
        != expected_refit
    ):
        raise PackageContractError(
            f"G4 refit chart failed: {job['execution_id']}."
        )

    if append:
        history = _sequence(
            body.get("history"), label="Append G5 history"
        )
        for index, raw in enumerate(history):
            row = _mapping(raw, label=f"Append G5 history[{index}]")
            if _integer(
                row.get("insertion_position"),
                label=f"Append G5 insertion position {index}",
            ) != index:
                raise PackageContractError(
                    f"G5 Append endpoint drifted: {job['execution_id']}."
                )
        g5 = {
            "domain": "endpoint_only",
            "accepted_round_population_count": 50,
            "interior_scored_count": 0,
            "append_position_only": body.get("append_position_only") is True,
        }
        if g5["append_position_only"] is not True:
            raise PackageContractError("G5 Append selector is not endpoint-only.")
    else:
        accepted_receipts = _sequence(
            scientific.get("accepted_round_receipts"),
            label="RA accepted-round scientific receipts",
        )
        if len(accepted_receipts) != 50:
            raise PackageContractError(
                f"G5 accepted receipt count drifted: {job['execution_id']}."
            )
        interior = 0
        appended = 0
        population_hashes: list[str] = []
        for index, raw in enumerate(accepted_receipts, start=1):
            accepted = _mapping(
                raw, label=f"RA accepted receipt {index}"
            )
            if (
                _integer(
                    accepted.get("accepted_round_ordinal"),
                    label=f"RA G5 accepted-round ordinal {index}",
                )
                != index
            ):
                raise PackageContractError(
                    f"G5 accepted-round order drifted: "
                    f"{job['execution_id']} round {index}."
                )
            population = _mapping(
                accepted.get("scored_insertion_position_population"),
                label=f"RA G5 population {index}",
            )
            verify_self_digest(
                population, label=f"RA G5 population {index}"
            )
            phases = _sequence(
                population.get("phases"),
                label=f"RA G5 phases {index}",
            )
            append_position = _integer(
                population.get("append_position"),
                label=f"RA G5 append position {index}",
            )
            if (
                population.get("schema")
                != "paper_i_scored_insertion_position_population_v1"
                or population.get("coordinate_chart")
                != "exact_ordered_insertion_zero_angle_v1"
                or population.get("phase_order")
                != ["phase_i", "phase_ii", "phase_iii"]
                or len(phases) != 3
            ):
                raise PackageContractError(
                    f"G5 scored-position receipt drifted: "
                    f"{job['execution_id']} round {index}."
                )
            observed_records = 0
            round_interior = 0
            round_append = 0
            phase_iii_identities: set[tuple[str, int]] = set()
            for phase_index, raw_phase in enumerate(phases):
                phase = _mapping(
                    raw_phase,
                    label=f"RA G5 phase {index}.{phase_index + 1}",
                )
                expected_phase = (
                    "phase_i",
                    "phase_ii",
                    "phase_iii",
                )[phase_index]
                records = _sequence(
                    phase.get("records"),
                    label=f"RA G5 {expected_phase} records {index}",
                )
                if (
                    phase.get("phase") != expected_phase
                    or not records
                    or _integer(
                        phase.get("population_count"),
                        label=(
                            f"RA G5 {expected_phase} population "
                            f"count {index}"
                        ),
                    )
                    != len(records)
                    or phase.get("ordered_population_sha256")
                    != canonical_sha256(records)
                ):
                    raise PackageContractError(
                        f"G5 ordered {expected_phase} population "
                        f"drifted: {job['execution_id']} round {index}."
                    )
                identities: set[tuple[str, str]] = set()
                phase_i_positions_by_generator: dict[
                    tuple[int, str], set[int]
                ] = {}
                for raw_record in records:
                    record = _mapping(
                        raw_record,
                        label=(
                            f"RA G5 {expected_phase} record {index}"
                        ),
                    )
                    domain_id = str(
                        record.get("domain_record_id", "")
                    ).strip()
                    generator_id = str(
                        record.get("generator_id", "")
                    ).strip()
                    pool_label = str(
                        record.get("pool_label", "")
                    ).strip()
                    pool_index = _integer(
                        record.get("pool_index"),
                        label="RA G5 pool index",
                    )
                    position = _integer(
                        record.get("insertion_position"),
                        label="RA G5 insertion position",
                    )
                    position_class = (
                        "interior"
                        if position < append_position
                        else "append"
                    )
                    if (
                        not domain_id
                        or not generator_id
                        or not pool_label
                        or position > append_position
                        or record.get("position_class")
                        != position_class
                        or (domain_id, generator_id) in identities
                    ):
                        raise PackageContractError(
                            f"G5 scored record identity drifted: "
                            f"{job['execution_id']} round {index}."
                        )
                    identities.add((domain_id, generator_id))
                    if expected_phase == "phase_i":
                        phase_i_positions_by_generator.setdefault(
                            (pool_index, generator_id),
                            set(),
                        ).add(position)
                    if expected_phase == "phase_iii":
                        phase_iii_identities.add(
                            (generator_id, position)
                        )
                    observed_records += 1
                    round_interior += int(
                        position_class == "interior"
                    )
                    round_append += int(
                        position_class == "append"
                    )
                if (
                    expected_phase == "phase_i"
                    and job["route_id"].endswith("always")
                    and (
                        not phase_i_positions_by_generator
                        or any(
                            positions
                            != set(range(append_position + 1))
                            for positions in (
                                phase_i_positions_by_generator.values()
                            )
                        )
                    )
                ):
                    raise PackageContractError(
                        "G5 full insertion did not score every Phase-I "
                        f"generator at every logical position: "
                        f"{job['execution_id']} round {index}."
                    )
            if (
                observed_records
                != _integer(
                    population.get("scored_record_count"),
                    label=f"RA G5 scored count {index}",
                )
                or round_interior
                != _integer(
                    population.get("interior_scored_count"),
                    label=f"RA G5 interior count {index}",
                )
                or round_append
                != _integer(
                    population.get("append_scored_count"),
                    label=f"RA G5 append count {index}",
                )
            ):
                raise PackageContractError(
                    f"G5 scored population totals drifted: "
                    f"{job['execution_id']} round {index}."
                )
            lineage = _sequence(
                accepted.get("accepted_candidate_lineage"),
                label=f"RA G5 accepted lineage {index}",
            )
            if not lineage:
                raise PackageContractError(
                    f"G5 accepted lineage is empty: "
                    f"{job['execution_id']} round {index}."
                )
            for raw_lineage in lineage:
                admitted = _mapping(
                    raw_lineage,
                    label=f"RA G5 admitted lineage {index}",
                )
                verify_self_digest(
                    admitted,
                    label=f"RA G5 admitted lineage {index}",
                )
                identity = (
                    str(
                        admitted.get("generator_identity", "")
                    ).strip(),
                    _integer(
                        admitted.get("insertion_position"),
                        label="RA G5 accepted insertion position",
                    ),
                )
                if (
                    not identity[0]
                    or identity not in phase_iii_identities
                ):
                    raise PackageContractError(
                        "G5 accepted admission was not in the scored "
                        f"Phase-III population: {job['execution_id']} "
                        f"round {index}."
                    )
            interior += round_interior
            appended += round_append
            population_hashes.append(population["sha256"])
        plateau_route = job["route_id"].endswith("plateau")
        always_route = job["route_id"].endswith("always")
        full_insertion_policy_verified = (
            _mapping(
                _mapping(
                    _mapping(
                        protocol.get("request"),
                        label="G5 protocol request",
                    ).get("method"),
                    label="G5 protocol method",
                ).get("insertion"),
                label="G5 protocol insertion policy",
            ).get("kind")
            == "full_commutation"
            if always_route
            else False
        )
        if (
            plateau_route and interior < 1
        ) or (
            always_route
            and (
                not full_insertion_policy_verified
                or interior < 1
            )
        ) or (
            job["route_id"].endswith("append_only") and interior != 0
        ):
            raise PackageContractError(
                f"G5 route domain failed: {job['execution_id']}."
            )
        g5 = {
            "domain": (
                "endpoint_only"
                if job["route_id"].endswith("append_only")
                else "full_commutation_or_plateau"
            ),
            "accepted_round_population_count": 50,
            "interior_scored_count": interior,
            "append_scored_count": appended,
            "population_receipt_sha256s": population_hashes,
            "full_insertion_policy_verified": (
                full_insertion_policy_verified
                if always_route
                else None
            ),
            "phase_i_full_logical_positions_verified": (
                True if always_route else None
            ),
            "interior_witness_status": (
                "observed"
                if interior > 0
                else "not_applicable"
            ),
        }

    if append:
        if (
            scientific.get("phase3_solver_invoked") is not False
            or scientific.get("trust_transaction_invoked") is not False
        ):
            raise PackageContractError(
                f"G6 Append crossed the RA boundary: {job['execution_id']}."
            )
        g6 = {
            "applicability": "not_applicable_to_append",
            "phase3_solver_invoked": False,
            "trust_transaction_invoked": False,
        }
    else:
        accepted_receipts = _sequence(
            scientific.get("accepted_round_receipts"),
            label="RA G6 accepted receipts",
        )
        replay_rows = _sequence(
            _mapping(payload.get("run"), label="RA G6 run").get(
                "scientific_replay"
            ),
            label="RA G6 scientific replay",
        )
        if len(accepted_receipts) != 50 or len(replay_rows) != 50:
            raise PackageContractError("G6 requires 50 RA round receipts.")
        for index, (raw, raw_replay) in enumerate(
            zip(accepted_receipts, replay_rows, strict=True), start=1
        ):
            _validate_g6_ra_round(
                job_id=str(job["execution_id"]),
                index=index,
                raw=_mapping(raw, label=f"RA G6 round {index}"),
                raw_replay=_mapping(
                    raw_replay, label=f"RA G6 replay {index}"
                ),
            )
        g6 = {
            "applicability": "ra_phase_iii",
            "accepted_round_receipt_count": 50,
            "metric_ridge": 0.0,
            "endpoint_overlap_query_charge": 0,
        }

    policy = _mapping(payload.get("policy"), label="policy echo")
    if (
        protocol.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or protocol.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
        or policy.get("active_gradient_policy")
        != protocol["active_gradient_policy"]
        or policy.get("resource_weighting_scope")
        != protocol["resource_weighting_scope"]
        or policy.get("active_gradient_indices_acquired") != []
        or _integer(
            policy.get("active_gradient_charge"),
            label="active-gradient charge",
        )
        != 0
        or scientific.get("policy") != policy
    ):
        raise PackageContractError(
            f"G7 policy echo failed: {job['execution_id']}."
        )
    active_gradient_closure = _validate_stationary_active_gradient_payload(
        payload,
        append=append,
    )

    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        validate_controller_replay_evidence,
    )
    from pipelines.static_adapt.numerical_physical_integrity import (
        numerical_physical_integrity_from_mapping,
    )

    try:
        controller_replay = validate_controller_replay_evidence(
            _mapping(
                scientific.get("controller_replay_evidence"),
                label="controller replay evidence",
            )
        )
        integrity = _mapping(
            scientific.get("numerical_physical_integrity"),
            label="numerical/physical integrity",
        )
        numerical_physical_integrity_from_mapping(integrity)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"Typed G8/G9 validation failed: {job['execution_id']}: {exc}"
        ) from exc
    g8 = _build_core_g8_attestation(
        job=job,
        protocol=protocol,
        payload=payload,
        summary=summary,
        source_lock=source_lock,
        source_root=source_root,
        artifacts=artifact_by_role,
        controller_replay=controller_replay,
        finalized_rounds=completed,
        package_state=package_state,
        verified_ed_reference=verified_ed_reference,
    )
    top_integrity = _mapping(
        payload.get("numerical_physical_integrity"),
        label="top numerical/physical integrity",
    )
    expected_integrity_method = (
        "append_adapt" if append else "ra_adapt"
    )
    integrity_transitions = _sequence(
        integrity.get("accepted_energy_transitions"),
        label="G9 accepted transition receipts",
    )
    completed_transitions = (
        _sequence(body.get("history"), label="G9 Append history")
        if append
        else _sequence(
            _mapping(payload.get("run"), label="G9 RA run").get(
                "accepted_transitions"
            ),
            label="G9 RA accepted transitions",
        )
    )
    if (
        top_integrity != integrity
        or scientific.get("numerical_physical_integrity_sha256")
        != canonical_sha256(integrity)
        or integrity.get("method") != expected_integrity_method
        or integrity.get("reporting_only") is not True
        or integrity.get("controller_decision_influence") is not False
        or integrity.get("finite_values_passed") is not True
        or integrity.get("nonfinite_value_paths") != []
        or integrity.get("sector_leak_flag") is not False
        or integrity.get("boson_truncation_leak_flag") is not False
        or integrity.get("accepted_energy_integrity_passed") is not True
        or integrity.get("integrity_passed") is not True
        or len(integrity_transitions) != 50
        or len(completed_transitions) != 50
    ):
        raise PackageContractError(
            f"G9 numerical/physical integrity failed: "
            f"{job['execution_id']}."
        )
    for index, (raw_integrity, raw_completed) in enumerate(
        zip(
            integrity_transitions,
            completed_transitions,
            strict=True,
        ),
        start=1,
    ):
        transition = _mapping(
            raw_integrity, label=f"G9 integrity transition {index}"
        )
        completed_transition = _mapping(
            raw_completed, label=f"G9 completed transition {index}"
        )
        before = _finite(
            transition.get("energy_before"),
            label=f"G9 energy before {index}",
        )
        after = _finite(
            transition.get("energy_after"),
            label=f"G9 energy after {index}",
        )
        tolerance = _finite(
            transition.get("absolute_tolerance"),
            label=f"G9 tolerance {index}",
        )
        nonincrease = after <= before + tolerance
        rollback = transition.get("typed_rollback_receipt")
        if (
            transition.get("schema")
            != "paper_i_accepted_energy_transition_integrity_v1"
            or _integer(
                transition.get("controller_round"),
                label=f"G9 transition ordinal {index}",
            )
            != index
            or _integer(
                completed_transition.get("controller_round"),
                label=f"G9 completed ordinal {index}",
            )
            != index
            or _finite(
                completed_transition.get("energy_before"),
                label=f"G9 completed energy before {index}",
            )
            != before
            or _finite(
                completed_transition.get("energy_after"),
                label=f"G9 completed energy after {index}",
            )
            != after
            or transition.get("nonincrease_passed") is not nonincrease
            or transition.get("gate_passed")
            is not bool(nonincrease or isinstance(rollback, Mapping))
            or (not nonincrease and not isinstance(rollback, Mapping))
            or (
                isinstance(rollback, Mapping)
                and not rollback
            )
        ):
            raise PackageContractError(
                f"G9 transition projection drifted: "
                f"{job['execution_id']} round {index}."
            )

    if append:
        accounting = _mapping(
            body.get("estimator_accounting"),
            label="Append estimator accounting",
        )
        work = accounting
    else:
        accounting = _mapping(
            _mapping(payload.get("run"), label="RA G10 run").get(
                "estimator_accounting"
            ),
            label="RA estimator accounting",
        )
        work = _mapping(
            accounting.get("all_work"), label="RA all-work accounting"
        )
    components = _accounting_components(work)
    s_alg = _integer(
        work.get("S_alg", work.get("s_alg")),
        label="G10 S_alg",
    )
    if s_alg != sum(components.values()):
        raise PackageContractError(
            f"G10 S_alg does not close: {job['execution_id']}."
        )
    if append:
        instrumentation = _mapping(
            accounting.get("executed_occurrence_instrumentation"),
            label="Append executed instrumentation",
        )
        if (
            accounting.get("closed_occurrence_reconciliation") is not True
            or components["N_metric"] != 0
            or instrumentation.get(
                "closed_against_estimator_ledger"
            )
            is not True
        ):
            raise PackageContractError(
                f"G10 Append accounting failed: {job['execution_id']}."
            )
        ledger_closure = _append_ledger_occurrence_closure(
            body=body,
            accounting=accounting,
            components=components,
            s_alg=s_alg,
        )
    elif (
        accounting.get("complete") is not True
        or accounting.get("status")
        != "resolved_from_live_state_keyed_instrumentation"
        or accounting.get("exact_blockers") != []
        or accounting.get("prefix_closure_passed") is not True
        or accounting.get("prefix_closure_status") != "complete"
        or _integer(
            accounting.get("raw_occurrence_total"),
            label="RA raw occurrence total",
        )
        != s_alg
        or _accounting_components(
            _mapping(
                accounting.get("raw_occurrences"),
                label="RA raw occurrences",
            )
        )
        != components
    ):
        raise PackageContractError(
            f"G10 RA accounting failed: {job['execution_id']}."
        )
    else:
        run_observation = _mapping(
            _mapping(payload.get("run"), label="RA G10 run").get(
                "observation"
            ),
            label="RA G10 observation",
        )
        observation_artifacts = _sequence(
            run_observation.get("artifacts"),
            label="RA G10 observation artifacts",
        )
        observed_by_kind = {
            str(_mapping(row, label="RA G10 observation artifact").get(
                "kind", ""
            )): _mapping(row, label="RA G10 observation artifact")
            for row in observation_artifacts
        }
        if (
            len(observed_by_kind) != 2
            or set(observed_by_kind)
            != {"accepted_state_checkpoint", "estimator_ledger"}
            or observed_by_kind["accepted_state_checkpoint"].get(
                "sha256"
            )
            != sidecars["checkpoint"]["sha256"]
            or int(
                observed_by_kind["accepted_state_checkpoint"].get(
                    "size_bytes", -1
                )
            )
            != int(sidecars["checkpoint"]["size_bytes"])
            or observed_by_kind["estimator_ledger"].get("sha256")
            != sidecars["estimator_ledger"]["sha256"]
            or int(
                observed_by_kind["estimator_ledger"].get(
                    "size_bytes", -1
                )
            )
            != int(sidecars["estimator_ledger"]["size_bytes"])
        ):
            raise PackageContractError(
                f"G10 RA authenticated observation-sidecar binding "
                f"failed: {job['execution_id']}."
            )
        ledger_closure = {
            "schema": "paper_i_estimator_call_ledger_sidecar_v2",
            "authentication": (
                "typed_observation_and_controller_replay_artifact_"
                "binding_v1"
            ),
            "compact_accounting_sha256": canonical_sha256(accounting),
            "observation_artifact_count": 2,
        }

    prefixes = _sequence(
        controller_replay.get("signed_controller_round_prefixes"),
        label="G11 signed controller prefixes",
    )
    resume = _mapping(
        controller_replay.get("resume_sidecar_closure"),
        label="G11 resume sidecar closure",
    )
    if (
        controller_replay.get("protocol_sha256") != protocol["sha256"]
        or controller_replay.get("problem_request_sha256")
        != protocol["problem"]["problem_request_sha256"]
        or len(prefixes) != 50
        or controller_replay.get("sha256")
        != scientific.get("controller_replay_evidence_sha256")
    ):
        raise PackageContractError(
            f"G11 signed-prefix closure failed: {job['execution_id']}."
        )
    if append:
        if (
            resume.get("resume_mode")
            != "authenticated_reconstruction_only_v1"
            or resume.get("public_resume_execution_supported") is not False
            or resume.get("reconstruction_fields_complete") is not True
        ):
            raise PackageContractError(
                f"G11 Append reconstruction boundary failed: "
                f"{job['execution_id']}."
            )
    else:
        checkpoint_artifact = _mapping(
            resume.get("checkpoint_artifact"),
            label="G11 RA checkpoint artifact",
        )
        ledger_artifact = _mapping(
            resume.get("estimator_ledger_artifact"),
            label="G11 RA estimator-ledger artifact",
        )
        if (
            resume.get("resume_mode")
            != "canonical_accepted_state_resume_v1"
            or resume.get("public_resume_execution_supported") is not True
            or resume.get("authentication_binding_complete") is not True
            or checkpoint_artifact.get("sha256")
            != sidecars["checkpoint"]["sha256"]
            or int(checkpoint_artifact.get("size_bytes", -1))
            != int(sidecars["checkpoint"]["size_bytes"])
            or ledger_artifact.get("sha256")
            != sidecars["estimator_ledger"]["sha256"]
            or int(ledger_artifact.get("size_bytes", -1))
            != int(sidecars["estimator_ledger"]["size_bytes"])
            or _mapping(
                resume.get("estimator_prefix_closure"),
                label="G11 estimator prefix closure",
            ).get("passed")
            is not True
        ):
            raise PackageContractError(
                f"G11 RA resume closure failed: {job['execution_id']}."
            )
    g11_bounded_validation = _validate_g11_bounded_diagnostic(
        job=job,
        protocol=protocol,
        primary_replay=controller_replay,
        diagnostic=g11_diagnostic,
        output_root=output_root,
    )

    compile_identity = _mapping(
        protocol.get("compile_identity"), label="compile identity"
    )
    if append:
        summary_compile_identity = _mapping(
            summary.get("compile_identity"),
            label="Append summary compile identity",
        )
        resource_summary = _mapping(
            summary.get("resources"),
            label="Append summary resources",
        )
        terminal_resources = _mapping(
            resource_summary.get("terminal_compiled_resources"),
            label="Append terminal compiled resources",
        )
        if (
            summary_compile_identity != compile_identity
            or resource_summary.get("schema")
            != "paper_i_append_resource_summary_v1"
            or resource_summary.get("terminal_observation_status")
            != "ok"
            or terminal_resources.get(
                "compiled_circuit_stats_status"
            )
            != "ok"
        ):
            raise PackageContractError(
                f"G12 Append reporting-resource path failed: "
                f"{job['execution_id']}."
            )
        resources = _compiled_resource_projection(
            terminal_resources,
            label="Append terminal compiled resources",
        )
        resource_path = (
            "paper_i_append_run_summary_v1.resources."
            "terminal_compiled_resources"
        )
    else:
        plateau = _mapping(
            summary.get("effective_plateau"),
            label="RA effective-plateau summary",
        )
        provenance = _mapping(
            summary.get("provenance"),
            label="RA G12 summary provenance",
        )
        plateau_resources = _mapping(
            plateau.get("resources"),
            label="RA effective-plateau resources",
        )
        if (
            plateau.get("status") != "available"
            or provenance.get("qiskit_compile_convention")
            != "table_i_basis_gate_transpile_v1"
        ):
            raise PackageContractError(
                f"G12 RA reporting-resource path failed: "
                f"{job['execution_id']}."
            )
        resources = _compiled_resource_projection(
            plateau_resources,
            label="RA effective-plateau resources",
        )
        resource_path = (
            "paper_i_run_summary_v1.effective_plateau.resources"
        )
    if (
        compile_identity.get("policy")
        != "table_i_basis_gate_transpile_v1"
        or int(compile_identity.get("optimization_level", -1)) != 0
        or int(compile_identity.get("transpiler_seed", -1)) != 7
        or compile_identity.get("coupling_map") is not None
        or compile_identity.get("reference_preparation_included") is not True
    ):
        raise PackageContractError(
            f"G12 compile identity/resources failed: "
            f"{job['execution_id']}."
        )

    stationarity = _mapping(
        authority["final_receipt"].get("stationarity_selection"),
        label="G13 stationarity selection",
    )
    user_selection = validate_user_selection_authority(source_root)
    user_decision = _mapping(
        user_selection["payload"].get("decision"),
        label="G13 explicit user decision",
    )
    user_binding = _mapping(
        user_selection.get("binding"),
        label="G13 explicit user-decision binding",
    )
    if (
        stationarity.get("winner_selected") is not True
        or stationarity.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or stationarity.get("authority", {}).get("sha256")
        != user_binding["sha256"]
        or stationarity.get("authority", {}).get("path")
        != user_binding["path"]
        or user_decision.get("core_campaign_id") != CAMPAIGN_ID
        or int(user_decision.get("core_direct_cell_count", -1)) != 48
        or int(user_decision.get("core_horizon", -1)) != 50
        or user_decision.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or user_decision.get("phase_i_resource_weighting_selection")
        != "late_resource_weighting_v1"
        or user_decision.get("study1_disposition")
        != (
            "canceled_unsubmitted_superseded_by_explicit_user_selection"
        )
        or user_decision.get("execution_authorized") is not False
        or user_decision.get("submission_authorized") is not False
        or policy.get("active_gradient_indices_acquired") != []
        or policy.get("active_gradient_charge") != 0
    ):
        raise PackageContractError(
            f"G13 stationary preservation failed: "
            f"{job['execution_id']}."
        )

    p3 = load_json_object(
        PACKAGE_DIR / P3_RECEIPT_RELATIVE, label="G5-G11 P3 receipt"
    )
    verify_self_digest(p2, label="G1-G3 P2 receipt")
    verify_self_digest(p3, label="G5-G11 P3 receipt")
    gate_rows = {
        "G1": {
            "cell_source_lock_sha256": source_lock["sha256"],
            "source_archive_sha256": source_lock["archive"]["sha256"],
            "source_member_sha256": source_lock["member"]["sha256"],
            "protocol_sha256": protocol["sha256"],
            "problem_reconstruction": (
                "typed_problem_receipt_reconstructed_before_facade_v1"
            ),
        },
        "G2": {
            "n_ph_work": int(job["nph"]),
            "n_ph_reference": int(exact["nph"]),
            "same_cutoff_reference": True,
            "exact_target_label": protocol["problem"][
                "exact_target_label"
            ],
            "ed_reference_file_sha256": exact["sha256"],
            "verified_ed_reference": verified_ed_reference,
            "verified_ed_reference_sha256": (
                verified_ed_reference["sha256"]
            ),
        },
        "G3": {
            "parent_inventory_sha256": result_parent["sha256"],
            "executable_pool_sha256": result_pool["sha256"],
            "candidate_representation": job[
                "candidate_representation"
            ],
            "p2_receipt_sha256": p2["sha256"],
            "six_regime_pool_construction_proof_sha256": (
                pool_proof["sha256"]
            ),
            "regime_pool_construction_proof_sha256": (
                pool_proof_row["sha256"]
            ),
            "parent_projection": {
                "count": int(result_parent["count"]),
                "ordered_labels_sha256": result_parent[
                    "ordered_labels_sha256"
                ],
                "ordered_pool_sha256": result_parent[
                    "ordered_pool_sha256"
                ],
            },
            "executable_projection": {
                "count": int(result_pool["count"]),
                "ordered_labels_sha256": result_pool[
                    "ordered_labels_sha256"
                ],
                "ordered_pool_sha256": result_pool[
                    "ordered_pool_sha256"
                ],
            },
            "singleton_construction_equivalence_sha256": (
                None
                if job["candidate_representation"]
                != "single_pauli_word_v1"
                else pool_proof_row[
                    "singleton_construction_equivalence"
                ]["sha256"]
            ),
        },
        "G4": {"accepted_refit_coordinate_chart": expected_refit},
        "G5": g5,
        "G6": g6,
        "G7": {
            "active_gradient_policy": policy[
                "active_gradient_policy"
            ],
            "resource_weighting_scope": policy[
                "resource_weighting_scope"
            ],
            "active_gradient_indices_acquired": [],
            "active_gradient_charge": 0,
            "phase3_active_gradient_accounting_occurrence_count": (
                active_gradient_closure[
                    "phase3_active_gradient_accounting_occurrence_count"
                ]
            ),
            "phase3_active_gradient_accounting_ordered_sha256": (
                active_gradient_closure[
                    "phase3_active_gradient_accounting_ordered_sha256"
                ]
            ),
            "serialized_policy_echo_sha256": active_gradient_closure[
                "serialized_policy_echo_sha256"
            ],
            "per_round_phase3_accounting_coverage": (
                active_gradient_closure[
                    "per_round_phase3_accounting_coverage"
                ]
            ),
            "limitation": active_gradient_closure["limitation"],
        },
        "G8": {
            "core_exact_reference_isolation": g8,
            "core_exact_reference_isolation_sha256": g8["sha256"],
        },
        "G9": {
            "evidence_sha256": canonical_sha256(integrity),
            "accepted_transition_check_count": 50,
            "sector_leak_flag": False,
            "boson_truncation_leak_flag": False,
        },
        "G10": {
            "components": components,
            "S_alg": s_alg,
            "ledger_closure": ledger_closure,
            "ledger_file_sha256": sidecars[
                "estimator_ledger"
            ]["sha256"],
            "ledger_size_bytes": sidecars[
                "estimator_ledger"
            ]["size_bytes"],
        },
        "G11": {
            "controller_replay_evidence_sha256": controller_replay[
                "sha256"
            ],
            "signed_prefix_count": 50,
            "resume_mode": resume["resume_mode"],
            "checkpoint_file_sha256": sidecars[
                "checkpoint"
            ]["sha256"],
            "checkpoint_size_bytes": sidecars[
                "checkpoint"
            ]["size_bytes"],
            "in_job_bounded_replay": dict(g11_diagnostic),
            "in_job_bounded_replay_validation": (
                g11_bounded_validation
            ),
            "p3_receipt_sha256": p3["sha256"],
        },
        "G12": {
            "compile_identity": "table_i_basis_gate_transpile_v1",
            "resource_path": resource_path,
            "resources": resources,
        },
        "G13": {
            "stationary_policy_selected": True,
            "stationarity_authority_sha256": stationarity[
                "authority"
            ]["sha256"],
            "explicit_user_selection": {
                "path": user_binding["path"],
                "sha256": user_binding["sha256"],
                "size_bytes": user_binding["size_bytes"],
                "study1_disposition": user_decision[
                    "study1_disposition"
                ],
                "execution_authorized": False,
                "submission_authorized": False,
            },
            "zero_active_gradient_acquisition": True,
            "phase3_active_gradient_accounting_occurrence_count": (
                active_gradient_closure[
                    "phase3_active_gradient_accounting_occurrence_count"
                ]
            ),
            "phase3_active_gradient_accounting_ordered_sha256": (
                active_gradient_closure[
                    "phase3_active_gradient_accounting_ordered_sha256"
                ]
            ),
            "serialized_policy_echo_sha256": active_gradient_closure[
                "serialized_policy_echo_sha256"
            ],
            "per_round_phase3_accounting_coverage": (
                active_gradient_closure[
                    "per_round_phase3_accounting_coverage"
                ]
            ),
            "limitation": active_gradient_closure["limitation"],
            "trajectory_outcome_selection_influence": False,
        },
    }
    return {
        gate_id: {
            "gate_id": gate_id,
            "status": "passed",
            "evidence": evidence,
        }
        for gate_id, evidence in gate_rows.items()
    }


def _full_run_scientific_closure(
    *,
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    source_root: Path,
    authority: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
    sidecars: Mapping[str, Mapping[str, Any]],
    g11_diagnostic: Mapping[str, Any],
    package_state: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    gates = _validate_core_gates(
        job=job,
        payload=payload,
        summary=summary,
        source_root=source_root,
        authority=authority,
        artifacts=artifacts,
        sidecars=sidecars,
        g11_diagnostic=g11_diagnostic,
        output_root=output_root,
        package_state=package_state,
    )
    expected = [f"G{index}" for index in range(1, 14)]
    if list(gates) != expected:
        raise PackageContractError(
            "Full-run gate map is not exactly ordered G1-G13."
        )
    manifest = load_json_object(
        PACKAGE_DIR / "package_manifest.json",
        label="scientific-closure package manifest",
    )
    plan = load_json_object(
        PACKAGE_DIR / "execution_plan.json",
        label="scientific-closure execution plan",
    )
    source_manifest = load_json_object(
        PACKAGE_DIR / "source_archive_manifest.json",
        label="scientific-closure source archive manifest",
    )
    control = load_json_object(
        PACKAGE_DIR / "control_plane_receipt.json",
        label="scientific-closure control plane",
    )
    for label, receipt in (
        ("scientific-closure package manifest", manifest),
        ("scientific-closure execution plan", plan),
        ("scientific-closure source archive manifest", source_manifest),
        ("scientific-closure control plane", control),
    ):
        verify_self_digest(receipt, label=label)
    archive = _mapping(
        source_manifest.get("archive"),
        label="scientific-closure source archive binding",
    )
    if (
        manifest["sha256"] != package_state["package_manifest_sha256"]
        or plan["sha256"] != package_state["execution_plan_sha256"]
        or archive["sha256"] != package_state["source_archive_sha256"]
        or manifest.get("execution_plan_sha256") != plan["sha256"]
        or manifest.get("source_archive_manifest_sha256")
        != source_manifest["sha256"]
        or plan.get("source_archive_manifest_sha256")
        != source_manifest["sha256"]
        or manifest.get("package_control_plane_sha256")
        != control["sha256"]
        or plan.get("package_control_plane", {}).get("sha256")
        != control["sha256"]
        or job.get("package_control_plane_sha256") != control["sha256"]
        or job.get("execution_plan_sha256") != plan["sha256"]
    ):
        raise PackageContractError(
            "Full-run package/source/control authority drifted."
        )
    artifact_by_role = _artifact_map(artifacts)
    return digested(
        {
            "schema": FULL_RUN_SCIENTIFIC_CLOSURE_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "cell_id": job["cell_id"],
            "protocol_sha256": job["protocol"][
                "canonical_sha256"
            ],
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "source_archive_sha256": archive["sha256"],
            "source_archive_manifest_sha256": source_manifest["sha256"],
            "package_control_plane_sha256": control["sha256"],
            "core_final_receipt_canonical_sha256": job[
                "core_final_receipt_canonical_sha256"
            ],
            "full_controller_rounds": 50,
            "artifact_bindings": {
                role: {
                    "sha256": artifact_by_role[role]["sha256"],
                    "size_bytes": artifact_by_role[role]["size_bytes"],
                }
                for role in EXPECTED_ARTIFACT_ROLES
            },
            "gate_ids": expected,
            "gates": gates,
            "g14_claimed": False,
            "paper_evidence_adopted": False,
            "status": "passed",
        }
    )


def _diagnostic_observation(root: Path, *, stem: str) -> Any:
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        SRObservationPolicy,
    )

    return SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=root / f"{stem}.checkpoint.json",
            every_controller_rounds=1,
            keep_history_tail=100,
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=root / f"{stem}.ledger.json"
        ),
    )


def _diagnostic_trajectory(
    payload: Mapping[str, Any], *, append: bool
) -> list[Mapping[str, Any]]:
    if append:
        rows = _sequence(
            _mapping(
                payload.get("result_payload"),
                label="diagnostic Append result",
            ).get("history"),
            label="diagnostic Append history",
        )
    else:
        rows = _sequence(
            _mapping(
                payload.get("run"), label="diagnostic RA run"
            ).get("accepted_trajectory"),
            label="diagnostic RA trajectory",
        )
    return [
        dict(_mapping(row, label="diagnostic trajectory row"))
        for row in rows
    ]


def _diagnostic_replay(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        validate_controller_replay_evidence,
    )

    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="diagnostic scientific receipts",
    )
    try:
        replay = validate_controller_replay_evidence(
            _mapping(
                scientific.get("controller_replay_evidence"),
                label="diagnostic replay evidence",
            )
        )
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"Diagnostic replay evidence failed validation: {exc}"
        ) from exc
    if (
        replay.get("sha256")
        != scientific.get("controller_replay_evidence_sha256")
    ):
        raise PackageContractError(
            "Diagnostic replay evidence digest binding drifted."
        )
    return replay


def _bounded_append_protocol(
    *,
    job: Mapping[str, Any],
    final_protocol: Any,
    problem: Any,
    authority: Mapping[str, Any],
    rounds: int,
) -> Any:
    from pipelines.static_adapt.ra_adapt import (
        AppendAdaptRequest,
        MacroCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt import bundles as bundle_module
    from pipelines.static_adapt.ra_adapt.append import (
        build_resolved_append_protocol,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_LATE,
        _attach_validated_bundle_protocol_authority,
    )
    from pipelines.static_adapt.sr_snake import (
        SRExecutionPolicy,
        SRStopPolicy,
    )

    lock = _mapping(
        authority["source_lock_cells"].get(job["source_lock_id"]),
        label="G11 Append source lock",
    )
    resolver = _mapping(
        lock.get("resolver_trace"), label="G11 Append resolver"
    )
    reference = _mapping(
        resolver.get("same_cutoff_ed_reference"),
        label="G11 Append same-cutoff reference",
    )
    resolver_source = _mapping(
        authority["global_source_locks"].get(
            "visible_settings_resolver"
        ),
        label="G11 Append resolver source",
    )
    source_refs = {
        "source_locks_manifest_sha256": authority["document_bindings"][
            "source_locks.json"
        ]["canonical_sha256"],
        "implementation_source_inventory_sha256": authority[
            "implementation_inventory_sha256"
        ],
        "cell_source_lock_id": job["source_lock_id"],
        "cell_source_lock_sha256": lock["sha256"],
        "visible_provenance_sha256": lock["member"]["sha256"],
        "provenance_tracker_sha256": lock["archive"]["sha256"],
        "ed_cutoff_reference_sha256": reference["sha256"],
        "resolver_script_sha256": resolver_source["sha256"],
    }
    cell = bundle_module.BundleCellSpec(
        cell_id=f"g11_diagnostic__{job['cell_id']}",
        stage="validation",
        regime_id=job["regime_id"],
        nph=int(job["nph"]),
        route_id=job["route_id"],
        algorithm_id=final_protocol.algorithm_id,
        selector_family="append_adapt",
        candidate_representation=job["candidate_representation"],
        horizon=rounds,
        source_lock_id=job["source_lock_id"],
    )
    kwargs = {
        "cell": cell,
        "bundle_id": bundle_module.CORE_BUNDLE_ID,
        "bundle_manifest_sha256": authority["document_bindings"][
            "bundle_manifest.json"
        ]["canonical_sha256"],
        "source_locks_sha256": authority["document_bindings"][
            "source_locks.json"
        ]["canonical_sha256"],
        "source_lock_refs": source_refs,
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_LATE,
    }
    first_authority = (
        bundle_module._bundle_protocol_materialization_authority(**kwargs)
    )
    protocol = build_resolved_append_protocol(
        problem,
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=rounds)
            ),
        ),
        materialization_authority=first_authority,
    )
    attached = _attach_validated_bundle_protocol_authority(
        protocol,
        bundle_module._bundle_protocol_materialization_authority(
            **kwargs,
            protocol_sha256=protocol.sha256,
        ),
    )
    if (
        attached.problem != final_protocol.problem
        or attached.parent_inventory != final_protocol.parent_inventory
        or attached.executable_pool != final_protocol.executable_pool
        or attached.active_gradient_policy
        != final_protocol.active_gradient_policy
        or attached.resource_weighting_scope
        != final_protocol.resource_weighting_scope
        or attached.optimizer != final_protocol.optimizer
        or attached.optimizer_maxiter != final_protocol.optimizer_maxiter
        or attached.seeds != final_protocol.seeds
    ):
        raise PackageContractError(
            "G11 bounded Append protocol drifted from final-cell authority."
        )
    return attached


def _write_append_diagnostic_sidecars(
    *,
    root: Path,
    stem: str,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[Path, Path]:
    body = _mapping(
        payload.get("result_payload"),
        label="G11 bounded Append result payload",
    )
    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="G11 bounded Append scientific receipts",
    )
    replay = _mapping(
        scientific.get("controller_replay_evidence"),
        label="G11 bounded Append replay",
    )
    checkpoint = digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_"
                "bounded_append_reconstruction_checkpoint_v1"
            ),
            "protocol_sha256": _mapping(
                payload.get("protocol"), label="G11 bounded protocol"
            )["sha256"],
            "controller_rounds_completed": body[
                "controller_rounds_completed"
            ],
            "accepted_operator_labels": body[
                "accepted_operator_labels"
            ],
            "accepted_generator_identities": body[
                "accepted_generator_identities"
            ],
            "logical_theta": body["logical_theta"],
            "controller_replay_evidence_sha256": replay["sha256"],
            "source_result_payload_sha256": summary[
                "source_result_payload_sha256"
            ],
            "public_resume_execution_supported": False,
            "reconstruction_fields_complete": True,
        }
    )
    ledger = _mapping(
        body.get("estimator_call_ledger"),
        label="G11 bounded Append ledger",
    )
    checkpoint_path = root / f"{stem}.checkpoint.json"
    ledger_path = root / f"{stem}.ledger.json"
    _atomic_write_json_value(checkpoint_path, checkpoint)
    _atomic_write_json_value(ledger_path, ledger)
    return checkpoint_path, ledger_path


def _diagnostic_artifact_bindings(
    root: Path, paths: list[Path]
) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root.parent).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]


def _g11_ra_diagnostic_artifact_paths(root: Path) -> list[Path]:
    return [
        root / "independent.checkpoint.json",
        root / "independent.ledger.json",
        root / "resume_prefix.checkpoint.json",
        root / "resume_prefix.ledger.json",
        root / "resumed.checkpoint.json",
        root / "resumed.ledger.json",
    ]


def _run_g11_bounded_diagnostic(
    *,
    job: Mapping[str, Any],
    primary_payload: Mapping[str, Any],
    protocol: Any,
    problem: Any,
    authority: Mapping[str, Any] | None,
    output_root: Path,
) -> dict[str, Any]:
    contract = _mapping(
        job.get("g11_bounded_replay_diagnostic"),
        label="G11 bounded-replay contract",
    )
    selected = contract.get("selected") is True
    expected_selected = job["route_id"] in G11_DIAGNOSTIC_ROUTES
    if selected is not expected_selected:
        raise PackageContractError(
            "G11 bounded-replay designation drifted."
        )
    if not selected:
        return digested(
            {
                "schema": G11_BOUNDED_REPLAY_DIAGNOSTIC_SCHEMA,
                "package_id": PACKAGE_ID,
                "execution_id": job["execution_id"],
                "regime_id": job["regime_id"],
                "selected": False,
                "paper_facing_result_allowed": False,
                "artifact_bindings": [],
                "status": "not_selected",
            }
        )
    if authority is None:
        raise PackageContractError(
            "Selected G11 diagnostic lacks core authority."
        )
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_append_adapt,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        bounded_prefix_replay_identity,
        compare_bounded_controller_replays,
    )
    from pipelines.static_adapt.sr_snake import AcceptedStateResume

    root = output_root / "g11_diagnostic"
    root.mkdir(parents=False, exist_ok=False)
    paths: list[Path] = []
    original = Path.cwd()
    try:
        os.chdir(root)
        if job["execution_entrypoint"] == "run_ra_adapt":
            independent = run_ra_adapt(
                problem,
                protocol,
                operational_controls=RAAdaptOperationalControls(
                    maximum_controller_rounds=2,
                    observation=_diagnostic_observation(
                        root, stem="independent"
                    ),
                ),
            )
            prefix = run_ra_adapt(
                problem,
                protocol,
                operational_controls=RAAdaptOperationalControls(
                    maximum_controller_rounds=1,
                    observation=_diagnostic_observation(
                        root, stem="resume_prefix"
                    ),
                ),
            )
            prefix_checkpoint = root / "resume_prefix.checkpoint.json"
            resumed = run_ra_adapt(
                problem,
                protocol,
                operational_controls=RAAdaptOperationalControls(
                    maximum_controller_rounds=2,
                    resume=AcceptedStateResume(
                        checkpoint_path=prefix_checkpoint,
                        checkpoint_sha256=sha256_file(
                            prefix_checkpoint
                        ),
                    ),
                    observation=_diagnostic_observation(
                        root, stem="resumed"
                    ),
                ),
            )
            independent_payload = independent.to_dict()
            prefix_payload = prefix.to_dict()
            resumed_payload = resumed.to_dict()
            primary_trajectory = _diagnostic_trajectory(
                primary_payload, append=False
            )
            independent_trajectory = _diagnostic_trajectory(
                independent_payload, append=False
            )
            prefix_trajectory = _diagnostic_trajectory(
                prefix_payload, append=False
            )
            resumed_trajectory = _diagnostic_trajectory(
                resumed_payload, append=False
            )
            primary_replay = _diagnostic_replay(primary_payload)
            independent_replay = _diagnostic_replay(
                independent_payload
            )
            prefix_replay = _diagnostic_replay(prefix_payload)
            resumed_replay = _diagnostic_replay(resumed_payload)
            primary_comparison = compare_bounded_controller_replays(
                primary_replay,
                independent_replay,
                controller_round=2,
            )
            resume_comparison = compare_bounded_controller_replays(
                prefix_replay,
                resumed_replay,
                controller_round=1,
            )
            resumed_comparison = compare_bounded_controller_replays(
                independent_replay,
                resumed_replay,
                controller_round=2,
            )
            if (
                len(primary_trajectory) != 50
                or primary_trajectory[:2] != independent_trajectory
                or prefix_trajectory != independent_trajectory[:1]
                or resumed_trajectory != independent_trajectory
                or primary_comparison.get("matched") is not True
                or resume_comparison.get("matched") is not True
                or resumed_comparison.get("matched") is not True
            ):
                raise PackageContractError(
                    "G11 RA bounded replay/resume identity failed."
                )
            paths.extend(_g11_ra_diagnostic_artifact_paths(root))
            evidence = {
                "method_family": "ra_adapt",
                "final_protocol_sha256": protocol.sha256,
                "primary_full_replay_evidence_sha256": primary_replay[
                    "sha256"
                ],
                "independent_replay_evidence": dict(
                    independent_replay
                ),
                "prefix_replay_evidence": dict(prefix_replay),
                "resumed_replay_evidence": dict(resumed_replay),
                "primary_bounded_comparison": dict(
                    primary_comparison
                ),
                "resume_prefix_comparison": dict(resume_comparison),
                "resumed_bounded_comparison": dict(
                    resumed_comparison
                ),
                "independent_controller_rounds": 2,
                "resume_prefix_controller_rounds": 1,
                "resumed_controller_rounds": 2,
                "primary_prefix_trajectory_sha256": canonical_sha256(
                    primary_trajectory[:2]
                ),
                "independent_trajectory_sha256": canonical_sha256(
                    independent_trajectory
                ),
                "resumed_trajectory_sha256": canonical_sha256(
                    resumed_trajectory
                ),
                "authenticated_resume_input": {
                    "path": (
                        "g11_diagnostic/"
                        "resume_prefix.checkpoint.json"
                    ),
                    "sha256": sha256_file(prefix_checkpoint),
                    "size_bytes": prefix_checkpoint.stat().st_size,
                },
                "authenticated_resume_performed": True,
            }
        else:
            bounded = _bounded_append_protocol(
                job=job,
                final_protocol=protocol,
                problem=problem,
                authority=authority,
                rounds=2,
            )
            first = run_append_adapt(problem, bounded)
            second = run_append_adapt(problem, bounded)
            first_payload = first.to_dict()
            second_payload = second.to_dict()
            first_summary = _typed_summary(
                first, entrypoint="run_append_adapt"
            )
            second_summary = _typed_summary(
                second, entrypoint="run_append_adapt"
            )
            paths.extend(
                _write_append_diagnostic_sidecars(
                    root=root,
                    stem="independent_primary",
                    payload=first_payload,
                    summary=first_summary,
                )
            )
            paths.extend(
                _write_append_diagnostic_sidecars(
                    root=root,
                    stem="independent_replay",
                    payload=second_payload,
                    summary=second_summary,
                )
            )
            first_trajectory = _diagnostic_trajectory(
                first_payload, append=True
            )
            second_trajectory = _diagnostic_trajectory(
                second_payload, append=True
            )
            primary_trajectory = _diagnostic_trajectory(
                primary_payload, append=True
            )
            primary_replay = _diagnostic_replay(primary_payload)
            first_replay = _diagnostic_replay(first_payload)
            second_replay = _diagnostic_replay(second_payload)
            comparison = compare_bounded_controller_replays(
                first_replay,
                second_replay,
                controller_round=2,
            )
            first_resume_boundary = _mapping(
                first_replay.get("resume_sidecar_closure"),
                label="G11 first Append resume boundary",
            )
            second_resume_boundary = _mapping(
                second_replay.get("resume_sidecar_closure"),
                label="G11 second Append resume boundary",
            )
            primary_prefix_identity = bounded_prefix_replay_identity(
                primary_replay, controller_round=2
            )
            first_prefix_identity = bounded_prefix_replay_identity(
                first_replay, controller_round=2
            )
            second_prefix_identity = bounded_prefix_replay_identity(
                second_replay, controller_round=2
            )
            if (
                len(primary_trajectory) != 50
                or first_trajectory != second_trajectory
                or len(first_trajectory) != 2
                or comparison.get("matched") is not True
                or len(
                    {
                        primary_prefix_identity,
                        first_prefix_identity,
                        second_prefix_identity,
                    }
                )
                != 1
                or any(
                    boundary.get("resume_mode")
                    != "authenticated_reconstruction_only_v1"
                    or boundary.get(
                        "public_resume_execution_supported"
                    )
                    is not False
                    or boundary.get(
                        "reconstruction_fields_complete"
                    )
                    is not True
                    for boundary in (
                        first_resume_boundary,
                        second_resume_boundary,
                    )
                )
            ):
                raise PackageContractError(
                    "G11 Append bounded replay/reconstruction failed."
                )
            evidence = {
                "method_family": "append_adapt",
                "final_protocol_sha256": protocol.sha256,
                "primary_full_replay_evidence_sha256": primary_replay[
                    "sha256"
                ],
                "bounded_protocol_sha256": bounded.sha256,
                "bounded_protocol_authority": (
                    "exact_final_problem_pool_source_lock_"
                    "derived_nonpaper_protocol_v1"
                ),
                "first_replay_evidence": dict(first_replay),
                "second_replay_evidence": dict(second_replay),
                "bounded_replay_comparison": dict(comparison),
                "independent_controller_rounds": 2,
                "primary_prefix_trajectory_sha256": canonical_sha256(
                    primary_trajectory[:2]
                ),
                "trajectory_sha256": canonical_sha256(
                    first_trajectory
                ),
                "primary_prefix_replay_identity_sha256": (
                    primary_prefix_identity
                ),
                "bounded_prefix_replay_identity_sha256": (
                    first_prefix_identity
                ),
                "public_resume_execution_supported": False,
                "reconstruction_fields_complete": True,
            }
    finally:
        os.chdir(original)
    return digested(
        {
            "schema": G11_BOUNDED_REPLAY_DIAGNOSTIC_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_id": job["execution_id"],
            "regime_id": job["regime_id"],
            "selected": True,
            "run_class": "bounded_nonpaper_diagnostic_v1",
            "paper_facing_result_allowed": False,
            "evidence": evidence,
            "artifact_bindings": _diagnostic_artifact_bindings(
                root, paths
            ),
            "status": "passed",
        }
    )


def _invoke(
    *,
    job: Mapping[str, Any],
    source_root: Path,
    output_root: Path,
    maximum_controller_rounds: int | None,
    authority: Mapping[str, Any] | None,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[dict[str, Any]],
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Any] | None,
]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_append_adapt,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        SRObservationPolicy,
    )

    protocol_path = source_root / safe_relative_path(
        job["protocol"]["path"], label="job protocol"
    )
    if (
        sha256_file(protocol_path) != job["protocol"]["sha256"]
        or protocol_path.stat().st_size
        != int(job["protocol"]["size_bytes"])
    ):
        raise PackageContractError("Worker protocol binding drifted.")
    protocol = load_validated_bundle_protocol(protocol_path)
    problem = _problem_from_protocol(protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    original = Path.cwd()
    os.chdir(protocol_path.parent.parent)
    try:
        if job["execution_entrypoint"] == "run_append_adapt":
            if maximum_controller_rounds is not None:
                raise PackageContractError(
                    "P4 selected an RA cell; Append cannot be shortened here."
                )
            result = run_append_adapt(problem, protocol)
        elif job["execution_entrypoint"] == "run_ra_adapt":
            controls = RAAdaptOperationalControls(
                maximum_controller_rounds=(
                    int(protocol.horizon)
                    if maximum_controller_rounds is None
                    else maximum_controller_rounds
                ),
                observation=SRObservationPolicy(
                    checkpoint=CheckpointObservation(
                        path=output_root / "checkpoint.json",
                        every_controller_rounds=1,
                        keep_history_tail=100,
                    ),
                    estimator_ledger=EstimatorLedgerObservation(
                        path=output_root / "estimator_ledger.json"
                    ),
                ),
            )
            result = run_ra_adapt(
                problem, protocol, operational_controls=controls
            )
        else:
            raise PackageContractError("Unknown job facade entrypoint.")
    finally:
        os.chdir(original)
    payload = result.to_dict()
    if not isinstance(payload, Mapping):
        raise PackageContractError("Facade result did not serialize.")
    summary = _typed_summary(
        result, entrypoint=str(job["execution_entrypoint"])
    )
    if job["execution_entrypoint"] == "run_append_adapt":
        result_body = payload.get("result_payload")
        scientific = payload.get("scientific_receipts")
        if not isinstance(result_body, Mapping) or not isinstance(
            scientific, Mapping
        ):
            raise PackageContractError(
                "Append result lacks typed sidecar source payloads."
            )
        ledger = result_body.get("estimator_call_ledger")
        replay = result_body.get("controller_replay_evidence")
        if not isinstance(ledger, Mapping) or not isinstance(replay, Mapping):
            raise PackageContractError(
                "Append result lacks authentic ledger/replay payloads."
            )
        checkpoint = digested(
            {
                "schema": (
                    "paper_i_append_adapt_reconstruction_checkpoint_v1"
                ),
                "continuation_boundary": (
                    "authenticated_reconstruction_only_v1"
                ),
                "public_resume_execution_supported": False,
                "reconstruction_fields_complete": True,
                "execution_id": job["execution_id"],
                "protocol_sha256": protocol.sha256,
                "controller_rounds_completed": result_body[
                    "controller_rounds_completed"
                ],
                "accepted_operator_labels": result_body[
                    "accepted_operator_labels"
                ],
                "accepted_generator_identities": result_body[
                    "accepted_generator_identities"
                ],
                "logical_theta": result_body["logical_theta"],
                "controller_replay_evidence": replay,
                "controller_replay_evidence_sha256": scientific[
                    "controller_replay_evidence_sha256"
                ],
                "result_payload_sha256": summary[
                    "source_result_payload_sha256"
                ],
            }
        )
        _atomic_write_json_value(output_root / "checkpoint.json", checkpoint)
        _atomic_write_json_value(
            output_root / "estimator_ledger.json", ledger
        )
    artifact_paths: dict[str, Path] = {}
    for role, path in (
        ("result", output_root / "result.json"),
        ("summary", output_root / "summary.json"),
    ):
        _atomic_write_json_value(
            path, payload if role == "result" else summary
        )
        artifact_paths[role] = path
    authentic_sidecars: dict[str, Mapping[str, Any]] = {}
    for role, name in (
        ("checkpoint", "checkpoint.json"),
        ("estimator_ledger", "estimator_ledger.json"),
    ):
        destination = output_root / name
        if not destination.is_file() or destination.is_symlink():
            raise PackageContractError(
                f"Facade did not produce the required {role} sidecar."
            )
        authentic_sidecars[role] = {
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
        artifact_paths[role] = destination
    g11_diagnostic = (
        None
        if maximum_controller_rounds is not None
        else _run_g11_bounded_diagnostic(
            job=job,
            primary_payload=payload,
            protocol=protocol,
            problem=problem,
            authority=authority,
            output_root=output_root,
        )
    )
    manifest_path = output_root / "execution_manifest.json"
    execution_manifest = digested(
        {
            "schema": "paper_i_ra_adapt_stationary_core_execution_manifest_v1",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "cell_id": job["cell_id"],
            "execution_entrypoint": job["execution_entrypoint"],
            "protocol_sha256": protocol.sha256,
            "job_spec_sha256": job["sha256"],
            "maximum_controller_rounds_override": (
                maximum_controller_rounds
            ),
            "run_class": (
                "smoke"
                if maximum_controller_rounds is not None
                else job["run_class"]
            ),
            "paper_facing_result_allowed": (
                maximum_controller_rounds is None
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "output_payloads": {
                role: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for role, path in sorted(artifact_paths.items())
            },
            "g11_bounded_replay_diagnostic": (
                None
                if g11_diagnostic is None
                else {
                    "selected": g11_diagnostic["selected"],
                    "sha256": g11_diagnostic["sha256"],
                    "artifact_bindings": g11_diagnostic[
                        "artifact_bindings"
                    ],
                }
            ),
            "status": "passed",
        }
    )
    atomic_write_json(manifest_path, execution_manifest)
    artifact_paths["execution_manifest"] = manifest_path
    if set(artifact_paths) != set(EXPECTED_ARTIFACT_ROLES):
        raise PackageContractError("Worker did not close all five artifact roles.")
    artifacts = [
        {
            "role": role,
            "path": artifact_paths[role].name,
            "declared_canonical_path": job["artifact_paths"][role],
            "mapping_kind": (
                "bounded_smoke_shadow_not_fulfillment_v1"
                if maximum_controller_rounds is not None
                else "worker_archive_copy_of_declared_output_v1"
            ),
            "sha256": sha256_file(artifact_paths[role]),
            "size_bytes": artifact_paths[role].stat().st_size,
        }
        for role in EXPECTED_ARTIFACT_ROLES
    ]
    return (
        payload,
        summary,
        artifacts,
        authentic_sidecars,
        g11_diagnostic,
    )


def _p4_full_insertion_witness(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _mapping(
        payload.get("protocol"),
        label="P4 full-insertion protocol",
    )
    request = _mapping(
        protocol.get("request"),
        label="P4 full-insertion request",
    )
    method = _mapping(
        request.get("method"),
        label="P4 full-insertion method",
    )
    insertion = _mapping(
        method.get("insertion"),
        label="P4 full-insertion policy",
    )
    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="P4 scientific receipts",
    )
    accepted_rounds = _sequence(
        scientific.get("accepted_round_receipts"),
        label="P4 accepted-round receipts",
    )
    if (
        insertion.get("kind") != "full_commutation"
        or len(accepted_rounds) != P4_SMOKE_ROUNDS
    ):
        raise PackageContractError(
            "P4 did not execute the typed two-round full-insertion route."
        )
    interior_scored_count = 0
    rows: list[dict[str, Any]] = []
    for round_index, raw_round in enumerate(accepted_rounds, start=1):
        accepted = _mapping(
            raw_round,
            label=f"P4 accepted-round receipt {round_index}",
        )
        population = _mapping(
            accepted.get("scored_insertion_position_population"),
            label=f"P4 scored population {round_index}",
        )
        population_sha256 = verify_self_digest(
            population,
            label=f"P4 scored population {round_index}",
        )
        append_position = _integer(
            population.get("append_position"),
            label=f"P4 append position {round_index}",
        )
        phases = _sequence(
            population.get("phases"),
            label=f"P4 scored phases {round_index}",
        )
        if len(phases) != 3:
            raise PackageContractError(
                "P4 scored-position receipt does not retain three phases."
            )
        phase_i = _mapping(
            phases[0],
            label=f"P4 Phase-I population {round_index}",
        )
        records = _sequence(
            phase_i.get("records"),
            label=f"P4 Phase-I records {round_index}",
        )
        positions_by_generator: dict[tuple[int, str], set[int]] = {}
        for raw_record in records:
            record = _mapping(
                raw_record,
                label=f"P4 Phase-I record {round_index}",
            )
            key = (
                _integer(
                    record.get("pool_index"),
                    label="P4 Phase-I pool index",
                ),
                str(record.get("generator_id", "")).strip(),
            )
            if not key[1]:
                raise PackageContractError(
                    "P4 Phase-I record has no generator identity."
                )
            positions_by_generator.setdefault(key, set()).add(
                _integer(
                    record.get("insertion_position"),
                    label="P4 Phase-I insertion position",
                )
            )
        expected_positions = set(range(append_position + 1))
        if (
            append_position != round_index - 1
            or phase_i.get("phase") != "phase_i"
            or not positions_by_generator
            or any(
                positions != expected_positions
                for positions in positions_by_generator.values()
            )
        ):
            raise PackageContractError(
                "P4 full insertion did not score every Phase-I generator "
                f"at every logical position in round {round_index}."
            )
        round_interior = _integer(
            population.get("interior_scored_count"),
            label=f"P4 interior scored count {round_index}",
        )
        interior_scored_count += round_interior
        rows.append(
            {
                "controller_round": round_index,
                "append_position": append_position,
                "phase_i_generator_count": len(
                    positions_by_generator
                ),
                "phase_i_logical_positions": sorted(expected_positions),
                "interior_scored_count": round_interior,
                "population_receipt_sha256": population_sha256,
            }
        )
    if interior_scored_count <= 0:
        raise PackageContractError(
            "P4 RA-always smoke produced no interior scored position."
        )
    return digested(
        {
            "schema": (
                "paper_i_ra_adapt_always_p4_full_insertion_witness_v1"
            ),
            "typed_full_insertion_verified": True,
            "phase_i_full_logical_positions_verified": True,
            "controller_round_count": P4_SMOKE_ROUNDS,
            "interior_scored_count": interior_scored_count,
            "rounds": rows,
            "status": "passed",
        }
    )


def run_p4(*, output: Path) -> dict[str, Any]:
    from validate_package import validate_package

    spec = load_json_object(
        PACKAGE_DIR / "p4_smoke_spec.json", label="P4 smoke spec"
    )
    job = load_json_object(
        PACKAGE_DIR / str(spec["source_job_spec_path"]),
        label="P4 source job",
    )
    if (
        spec.get("source_execution_id") != P4_EXECUTION_ID
        or int(spec.get("maximum_controller_rounds", -1))
        != P4_SMOKE_ROUNDS
        or job.get("execution_id") != P4_EXECUTION_ID
        or job.get("route_id") != "ra_macro_always"
    ):
        raise PackageContractError(
            "P4 spec did not select the two-round macro RA-always cell."
        )
    with tempfile.TemporaryDirectory(
        prefix="paper_i_stationary_core_p4_"
    ) as raw:
        root = Path(raw)
        source_root = root / "source"
        artifacts_root = root / "artifacts"
        _safe_extract_source_archive(source_root)
        base = validate_package(
            require_p4=False, require_authorization=False
        )
        authority = validate_core_authority(source_root)
        p4_source_lock = _mapping(
            authority["source_lock_cells"].get(job["source_lock_id"]),
            label="P4 source-lock cell",
        )
        verified_ed_reference = _verified_same_cutoff_ed_reference(
            job=job,
            source_lock=p4_source_lock,
            source_root=source_root,
            authority=authority,
        )
        _activate_source_root(source_root)
        from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
            build_study1_trusted_execution_receipt,
            validate_study1_trusted_execution_receipt,
        )

        trusted_execution = build_study1_trusted_execution_receipt(
            source_root=source_root
        ).to_dict()
        if (
            validate_study1_trusted_execution_receipt(
                trusted_execution,
                source_root=source_root,
                reverify_source=True,
            )
            != trusted_execution
        ):
            raise PackageContractError(
                "P4 trusted execution/source-dataflow receipt drifted."
            )
        (
            payload,
            _summary,
            artifacts,
            _sidecars,
            _diagnostic,
        ) = _invoke(
            job=job,
            source_root=source_root,
            output_root=artifacts_root,
            maximum_controller_rounds=P4_SMOKE_ROUNDS,
            authority=authority,
        )
        full_insertion_witness = _p4_full_insertion_witness(payload)
        _assert_source_locked_imports(source_root)
        retained_artifacts = [
            {
                **row,
                "retention": "embedded_exact_utf8_json_bytes_v1",
                "exact_file_text": (
                    artifacts_root / str(row["path"])
                ).read_text(encoding="utf-8"),
                "canonical_payload": load_json_object(
                    artifacts_root / str(row["path"]),
                    label=f"P4 {row['role']} artifact",
                ),
            }
            for row in artifacts
        ]
    result = digested(
        {
            "schema": P4_SMOKE_RESULT_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_execution_id": spec["source_execution_id"],
            "p4_smoke_spec_sha256": spec["sha256"],
            "package_manifest_sha256": base["package_manifest_sha256"],
            "execution_plan_sha256": base["execution_plan_sha256"],
            "source_archive_sha256": base["source_archive_sha256"],
            "maximum_controller_rounds": P4_SMOKE_ROUNDS,
            "run_class": "smoke",
            "artifact_bindings": retained_artifacts,
            "trusted_execution_source_dataflow_receipt": (
                trusted_execution
            ),
            "trusted_execution_source_dataflow_receipt_sha256": (
                trusted_execution["sha256"]
            ),
            "verified_same_cutoff_ed_reference": verified_ed_reference,
            "verified_same_cutoff_ed_reference_sha256": (
                verified_ed_reference["sha256"]
            ),
            "bounded_dispatch_passed": True,
            "source_locked_archive_validated": True,
            "full_insertion_witness": full_insertion_witness,
            "full_insertion_witness_sha256": (
                full_insertion_witness["sha256"]
            ),
            "paper_facing_result_allowed": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "status": "passed",
        }
    )
    atomic_write_json(output, result)
    return result


def run_authorized_job(
    *,
    source_root: Path,
    job_path: Path,
    expected_job_sha256: str,
    output_root: Path,
    scheduler_attempt_ordinal: int,
    scheduler_cluster_id: int,
    scheduler_proc_id: int,
    verified_image_path: str,
    verified_image_sha256: str,
) -> dict[str, Any]:
    from validate_package import validate_package

    if not source_root.exists():
        _safe_extract_source_archive(source_root)
    elif not source_root.is_dir() or source_root.is_symlink():
        raise PackageContractError("Worker source root is unsafe.")
    authority = validate_core_authority(source_root)
    _activate_source_root(source_root)
    package_state = validate_package(
        require_p4=True, require_authorization=True
    )
    job = load_json_object(job_path, label="worker job spec")
    verify_self_digest(job, label="worker job spec")
    if (
        sha256_file(job_path) != expected_job_sha256
        or job.get("schema") != JOB_SPEC_SCHEMA
        or job.get("package_id") != PACKAGE_ID
    ):
        raise PackageContractError("Worker job binding drifted.")
    manifest = load_json_object(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    plan = load_json_object(
        PACKAGE_DIR / "execution_plan.json", label="execution plan"
    )
    p4 = load_json_object(
        PACKAGE_DIR / "authority/p4_packaged_dispatch_receipt.json",
        label="P4 receipt",
    )
    authorization = load_json_object(
        PACKAGE_DIR / SUBMISSION_AUTHORIZATION_RELATIVE,
        label="submission authorization",
    )
    authorization_sha = validate_submission_authorization(
        authorization,
        package_manifest=manifest,
        execution_plan=plan,
        p4_receipt=p4,
    )
    if (
        scheduler_attempt_ordinal < 1
        or scheduler_cluster_id < 0
        or scheduler_proc_id < 0
        or verified_image_path != REMOTE_IMAGE_PATH
        or verified_image_sha256 != REMOTE_IMAGE_SHA256
    ):
        raise PackageContractError(
            "Scheduler identity or execution-image binding drifted."
        )
    (
        payload,
        summary,
        artifacts,
        sidecars,
        g11_diagnostic,
    ) = _invoke(
        job=job,
        source_root=source_root,
        output_root=output_root,
        maximum_controller_rounds=None,
        authority=authority,
    )
    if g11_diagnostic is None:
        raise PackageContractError(
            "Full worker omitted the G11 diagnostic disposition."
        )
    roles = {row["role"] for row in artifacts}
    if (
        len(artifacts) != len(EXPECTED_ARTIFACT_ROLES)
        or roles != set(EXPECTED_ARTIFACT_ROLES)
    ):
        raise PackageContractError("Worker output closure is incomplete.")
    scientific_closure = _full_run_scientific_closure(
        job=job,
        payload=payload,
        summary=summary,
        source_root=source_root,
        authority=authority,
        artifacts=artifacts,
        sidecars=sidecars,
        g11_diagnostic=g11_diagnostic,
        package_state=package_state,
        output_root=output_root,
    )
    _assert_source_locked_imports(source_root)
    receipt = digested(
        {
            "schema": WORKER_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "scheduler_attempt_ordinal": scheduler_attempt_ordinal,
            "scheduler_cluster_id": scheduler_cluster_id,
            "scheduler_proc_id": scheduler_proc_id,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "package_manifest_sha256": package_state[
                "package_manifest_sha256"
            ],
            "package_manifest_file_sha256": sha256_file(
                PACKAGE_DIR / "package_manifest.json"
            ),
            "execution_plan_sha256": package_state[
                "execution_plan_sha256"
            ],
            "execution_plan_file_sha256": sha256_file(
                PACKAGE_DIR / "execution_plan.json"
            ),
            "job_spec_sha256": job["sha256"],
            "job_spec_file_sha256": sha256_file(job_path),
            "job_spec_path": f"jobs/{job['execution_id']}.json",
            "source_archive_sha256": package_state[
                "source_archive_sha256"
            ],
            "submission_authorization_sha256": authorization_sha,
            "submission_authorization_file_sha256": sha256_file(
                PACKAGE_DIR / SUBMISSION_AUTHORIZATION_RELATIVE
            ),
            "remote_image_path": verified_image_path,
            "remote_image_sha256": verified_image_sha256,
            "artifact_bindings": artifacts,
            "scientific_closure": scientific_closure,
            "scientific_closure_sha256": scientific_closure[
                "sha256"
            ],
            "status": "passed",
        }
    )
    atomic_write_json(output_root / "worker_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("p4-smoke", "execute"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--job-spec", type=Path)
    parser.add_argument("--job-spec-sha256")
    parser.add_argument("--scheduler-attempt-ordinal", type=int)
    parser.add_argument("--scheduler-cluster-id", type=int)
    parser.add_argument("--scheduler-proc-id", type=int)
    parser.add_argument("--verified-image-path")
    parser.add_argument("--verified-image-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "p4-smoke":
            result = run_p4(output=args.output.resolve())
        else:
            if (
                args.source_root is None
                or args.job_spec is None
                or not args.job_spec_sha256
                or args.scheduler_attempt_ordinal is None
                or args.scheduler_attempt_ordinal < 1
                or args.scheduler_cluster_id is None
                or args.scheduler_cluster_id < 0
                or args.scheduler_proc_id is None
                or args.scheduler_proc_id < 0
                or not args.verified_image_path
                or not args.verified_image_sha256
            ):
                raise PackageContractError(
                    "execute requires source root, job spec, and job hash."
                )
            result = run_authorized_job(
                source_root=args.source_root.resolve(),
                job_path=args.job_spec.resolve(),
                expected_job_sha256=args.job_spec_sha256,
                output_root=args.output.resolve(),
                scheduler_attempt_ordinal=args.scheduler_attempt_ordinal,
                scheduler_cluster_id=args.scheduler_cluster_id,
                scheduler_proc_id=args.scheduler_proc_id,
                verified_image_path=args.verified_image_path,
                verified_image_sha256=args.verified_image_sha256,
            )
        print(canonical_json_bytes(result).decode("utf-8"))
        return 0
    except (OSError, PackageContractError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
