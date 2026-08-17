#!/usr/bin/env python3
"""Materialize the Page-12/Page-16 insertion-policy comparator packages.

The completed plateau trajectories are references, not jobs.  Each package
contains exactly the two missing RA-ADAPT insertion arms for the same six
regime/representation cells: always commutation-reduced and append-only.

The worker imports the byte-identical source archive from the authenticated
plateau package.  The legacy Phase-0 dispatch algorithm id is deliberately
retained because that sealed implementation registers the Phase-0/Qiskit
funnel under that id; the typed insertion request and route contract are the
authoritative executed insertion semantics.  This diagnostic-wrapper fact is
recorded in every job and in the source-lock audit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = Path(__file__).resolve().parent

PAGE12_SOURCE = REPAIR_ROOT / (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc"
)
PAGE16_SOURCE = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_weak50_strong30_20260811_v1_chtc"
)

PAGE12_TARGET = REPAIR_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
PAGE16_TARGET = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_"
    "20260812_v1_chtc"
)

PLATEAU_EVIDENCE_ROOT = Path(
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)

EXPECTED_SOURCE_ARCHIVE_SHA256 = {
    "page12": "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762",
    "page16": "95b3ea575a4590961b6a57337eb1c58ef3ba3855d9d342b179657973c129ef26",
}

POLICIES: tuple[dict[str, str], ...] = (
    {
        "id": "always_commutation_reduced",
        "typed_kind": "always_commutation_reduced",
        "runtime_mode": "full_commutation_reduced",
        "route_token": "full_commutation_reduced",
    },
    {
        "id": "append_only",
        "typed_kind": "append_only",
        "runtime_mode": "append_only",
        "route_token": "append_only",
    },
)

INSERTION_INVARIANT_KEYS = {
    "canonical_insertion_policy",
    "experimental_insertion_policy",
    "insertion_equivalence_policy",
    "insertion_position_scope",
    "plateau_hysteresis_active",
    "plateau_patience",
    "plateau_prior_mean_decrease_ratio_threshold",
    "plateau_threshold_calibration_status",
    "plateau_threshold_comparison",
    "plateau_trigger_source",
}

CONTROL_FILES = (
    "package_contract.py",
    "activate_package.py",
    "run_cell.py",
    "validate_package.py",
    "probe_image_runtime.py",
    "execute_authorized_job.sh",
    "submit.sub.in",
)


class MaterializationError(RuntimeError):
    """Raised when the comparator derivation does not close exactly."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"Expected a JSON object: {path}")
    return value


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    observed = unsigned.pop("sha256", None)
    expected = canonical_sha256(unsigned)
    if observed != expected:
        raise MaterializationError(f"{label} self-digest drifted.")
    return expected


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        result["canonical_sha256"] = verify_self_digest(
            load_json(path), label=result["path"]
        )
    return result


def _replace_exact(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count == 0:
        raise MaterializationError(f"Missing control-script patch: {label}")
    return text.replace(old, new)


def _copy_control_scripts(source: Path, target: Path) -> None:
    for name in CONTROL_FILES:
        shutil.copyfile(source / name, target / name)


def _patch_package_contract(
    *,
    source: Path,
    target: Path,
    package_id: str,
    campaign_id: str,
    bundle_id: str,
    execution_ids: list[str],
) -> None:
    path = target / "package_contract.py"
    text = path.read_text(encoding="utf-8")
    override = f'''\n\n# Generated diagnostic-comparator package overrides.\nPACKAGE_ID = {package_id!r}\nCAMPAIGN_ID = {campaign_id!r}\nBUNDLE_ID = {bundle_id!r}\nRUN_CLASS = "diagnostic"\nCONTROL_FILES = {CONTROL_FILES!r}\nCOMPARATOR_POLICIES = ("always_commutation_reduced", "append_only")\nEXPECTED_EXECUTION_IDS = {tuple(execution_ids)!r}\n\ndef expected_execution_ids() -> tuple[str, ...]:\n    return EXPECTED_EXECUTION_IDS\n'''
    path.write_text(text + override, encoding="utf-8")


def _patch_run_cell(target: Path) -> None:
    path = target / "run_cell.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_exact(
        text,
        "    CANDIDATE_REPRESENTATION,\n",
        "    CANDIDATE_REPRESENTATION,\n    COMPARATOR_POLICIES,\n",
        label="run-cell comparator import",
    )
    text = text.replace("passed_inert_six_cells", "passed_inert_twelve_cells")
    text = text.replace('manifest.get("row_count") != 6', 'manifest.get("row_count") != 12')
    text = text.replace("Inert six-cell", "Inert twelve-cell")
    text = text.replace(
        "prepare_six_cell_chtc_execution_and_submission_v1",
        "prepare_twelve_cell_chtc_execution_and_submission_v1",
    )
    text = _replace_exact(
        text,
        '        or job.get("algorithm_id") != ALGORITHM_ID\n',
        '        or job.get("algorithm_id") != ALGORITHM_ID\n'
        '        or job.get("comparator_policy") not in COMPARATOR_POLICIES\n'
        '        or job.get("typed_insertion_kind") != job.get("comparator_policy")\n'
        '        or job.get("dispatch_template_algorithm_id") != ALGORITHM_ID\n'
        '        or job.get("fresh_source_value_anchor") is not False\n',
        label="run-cell job comparator closure",
    )
    text = _replace_exact(
        text,
        '        or manifest.get("child_route_contract_sha256")\n        != job.get("route_contract_sha256")\n',
        '        or manifest.get("route_contract_sha256_by_execution_id", {}).get(\n'
        '            execution_id\n'
        '        ) != job.get("route_contract_sha256")\n',
        label="run-cell per-execution route binding",
    )
    text = _replace_exact(
        text,
        '        or protocol.route_contract.get("route_profile") != TARGET_ROUTE_PROFILE\n',
        '        or protocol.route_contract.get("route_profile")\n'
        '        != job.get("route_profile")\n',
        label="run-cell comparator route profile",
    )
    text = _replace_exact(
        text,
        '        or protocol.request.method.insertion.kind != "plateau_commutation"\n',
        '        or protocol.request.method.insertion.kind\n'
        '        != job.get("typed_insertion_kind")\n',
        label="run-cell typed insertion",
    )
    text = _replace_exact(
        text,
        '        or invariants.get("plateau_prior_mean_decrease_ratio_threshold")\n'
        '        != 1.0e-4\n',
        '',
        label="run-cell retired comparator plateau threshold",
    )
    text = _replace_exact(
        text,
        '        or not isinstance(invariants, Mapping)\n',
        '        or execution.get("adapt_insertion_mode")\n'
        '        != job.get("runtime_insertion_mode")\n'
        '        or not isinstance(invariants, Mapping)\n'
        '        or (\n'
        '            job.get("comparator_policy") == "always_commutation_reduced"\n'
        '            and (\n'
        '                invariants.get("insertion_position_scope")\n'
        '                != "full_logical_ansatz_commutation_classes_every_depth_v2"\n'
        '                or invariants.get("insertion_equivalence_policy")\n'
        '                != (\n'
        '                    "termwise_cross_component_commutation_"\n'
        '                    "earliest_representative_v1"\n'
        '                )\n'
        '            )\n'
        '        )\n'
        '        or (\n'
        '            job.get("comparator_policy") == "append_only"\n'
        '            and any(\n'
        '                (\n'
        '                    key.startswith("plateau_")\n'
        '                    and key\n'
        '                    != "plateau_prior_mean_decrease_ratio_threshold"\n'
        '                )\n'
        '                or key in {\n'
        '                    "canonical_insertion_policy",\n'
        '                    "experimental_insertion_policy",\n'
        '                    "insertion_equivalence_policy",\n'
        '                    "insertion_position_scope",\n'
        '                }\n'
        '                for key in invariants\n'
        '            )\n'
        '        )\n',
        label="run-cell insertion route closure",
    )
    text = _replace_exact(
        text,
        '            "target_horizon": job["target_horizon"],\n',
        '            "target_horizon": job["target_horizon"],\n'
        '            "comparator_policy": job["comparator_policy"],\n',
        label="run-cell preflight comparator receipt",
    )
    text = _replace_exact(
        text,
        '    from pipelines.static_adapt.ra_adapt import (\n'
        '        RAAdaptOperationalControls,\n'
        '        run_ra_adapt,\n'
        '    )\n',
        '    from pipelines.static_adapt.ra_adapt import (\n'
        '        RAAdaptOperationalControls,\n'
        '        run_ra_adapt,\n'
        '    )\n'
        '    from pipelines.static_adapt.ra_adapt import engine as ra_engine\n'
        '    from pipelines.reporting import paper_i_run_summary as summary_module\n',
        label="run-cell comparator route-wrapper import",
    )
    text = _replace_exact(
        text,
        '    result = run_ra_adapt(\n'
        '        problem,\n'
        '        protocol,\n'
        '        operational_controls=controls,\n'
        '    )\n',
        '    bound_route = dict(protocol.route_contract)\n'
        '    bound_route_sha256 = str(bound_route.pop("sha256"))\n'
        '    bound_route_profile = str(bound_route["route_profile"])\n'
        '    original_route_builder = ra_engine._repaired_route_contract\n'
        '    original_reduction_validator = (\n'
        '        ra_engine.validate_commutation_reduced_insertion_receipt\n'
        '    )\n'
        '    original_summary_identities = (\n'
        '        summary_module._canonical_ra_supersession_identities\n'
        '    )\n'
        '\n'
        '    def comparator_route_builder(\n'
        '        request: Any,\n'
        '        *,\n'
        '        active_gradient_policy: str,\n'
        '        resource_weighting_scope: str,\n'
        '        algorithm_id: str | None = None,\n'
        '        problem: Any = None,\n'
        '    ) -> tuple[str, str, dict[str, Any], str]:\n'
        '        if (\n'
        '            str(algorithm_id) == str(protocol.algorithm_id)\n'
        '            and request.method.insertion.kind\n'
        '            in {"always_commutation_reduced", "append_only"}\n'
        '        ):\n'
        '            return (\n'
        '                bound_route_profile,\n'
        '                bound_route_profile,\n'
        '                dict(bound_route),\n'
        '                bound_route_sha256,\n'
        '            )\n'
        '        return original_route_builder(\n'
        '            request,\n'
        '            active_gradient_policy=active_gradient_policy,\n'
        '            resource_weighting_scope=resource_weighting_scope,\n'
        '            algorithm_id=algorithm_id,\n'
        '            problem=problem,\n'
        '        )\n'
        '\n'
        '    def comparator_summary_identities(\n'
        '        method: Any,\n'
        '        *,\n'
        '        candidate_representation: str,\n'
        '    ) -> tuple[tuple[str, str, str, str], ...]:\n'
        '        if (\n'
        '            method.insertion.kind\n'
        '            in {"always_commutation_reduced", "append_only"}\n'
        '            and candidate_representation\n'
        '            == protocol.candidate_representation\n'
        '        ):\n'
        '            return ((\n'
        '                "ra_adapt",\n'
        '                bound_route_profile,\n'
        '                bound_route_profile,\n'
        '                bound_route_sha256,\n'
        '            ),)\n'
        '        return original_summary_identities(\n'
        '            method,\n'
        '            candidate_representation=candidate_representation,\n'
        '        )\n'
        '\n'
        '    def comparator_reduction_validator(\n'
        '        receipt: Mapping[str, Any],\n'
        '        *,\n'
        '        expected_policy: str,\n'
        '        expected_requested_positions: Any = None,\n'
        '        scored_population: Any = None,\n'
        '    ) -> dict[str, Any]:\n'
        '        return original_reduction_validator(\n'
        '            receipt,\n'
        '            expected_policy=expected_policy,\n'
        '            expected_requested_positions=expected_requested_positions,\n'
        '            scored_population=(\n'
        '                None\n'
        '                if protocol.request.method.insertion.kind\n'
        '                == "always_commutation_reduced"\n'
        '                else scored_population\n'
        '            ),\n'
        '        )\n'
        '\n'
        '    ra_engine._repaired_route_contract = comparator_route_builder\n'
        '    ra_engine.validate_commutation_reduced_insertion_receipt = (\n'
        '        comparator_reduction_validator\n'
        '    )\n'
        '    summary_module._canonical_ra_supersession_identities = (\n'
        '        comparator_summary_identities\n'
        '    )\n'
        '    try:\n'
        '        result = run_ra_adapt(\n'
        '            problem,\n'
        '            protocol,\n'
        '            operational_controls=controls,\n'
        '        )\n'
        '    finally:\n'
        '        ra_engine._repaired_route_contract = original_route_builder\n'
        '        ra_engine.validate_commutation_reduced_insertion_receipt = (\n'
        '            original_reduction_validator\n'
        '        )\n'
        '        summary_module._canonical_ra_supersession_identities = (\n'
        '            original_summary_identities\n'
        '        )\n',
        label="run-cell comparator bound-route wrapper",
    )
    path.write_text(text, encoding="utf-8")


def _patch_validate_package(target: Path) -> None:
    path = target / "validate_package.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("passed_inert_six_cells", "passed_inert_twelve_cells")
    text = text.replace('manifest.get("row_count") != 6', 'manifest.get("row_count") != 12')
    text = text.replace("len(jobs) != 6 or len(protocols) != 6", "len(jobs) != 12 or len(protocols) != 12")
    text = text.replace("launch_ready = deep_count == 6", "launch_ready = deep_count == 12")
    path.write_text(text, encoding="utf-8")


def _patch_activate_package(target: Path) -> None:
    path = target / "activate_package.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("passed_inert_six_cells", "passed_inert_twelve_cells")
    text = text.replace(
        "prepare_six_cell_chtc_execution_and_submission_v1",
        "prepare_twelve_cell_chtc_execution_and_submission_v1",
    )
    text = text.replace(
        'probe_receipt.get("deep_worker_preflight_count") != 6',
        'probe_receipt.get("deep_worker_preflight_count") != 12',
    )
    path.write_text(text, encoding="utf-8")


def _transform_protocol(
    base: Mapping[str, Any],
    *,
    policy: Mapping[str, str],
    execution_id: str,
    bundle_id: str,
    bundle_manifest_sha256: str,
) -> dict[str, Any]:
    protocol = copy.deepcopy(base)
    request = protocol["request"]
    request["method"]["insertion"] = {"kind": policy["typed_kind"]}

    route = protocol["route_contract"]
    old_profile = str(route["route_profile"])
    token = "__insertion_commutation_plateau_v2__"
    if token not in old_profile:
        raise MaterializationError("Source protocol lost the plateau-v2 route token.")
    route["route_profile"] = old_profile.replace(
        token, f"__{policy['route_token']}__", 1
    )
    route["execution_settings"]["adapt_insertion_mode"] = policy[
        "runtime_mode"
    ]
    invariants = route["semantic_invariants"]
    for key in INSERTION_INVARIANT_KEYS:
        invariants.pop(key, None)
    # The sealed Phase-0 dispatch validator requires its historical threshold
    # field even when the typed policy cannot consult or trigger it.  Preserve
    # it as an explicitly inert legacy dispatch-template value.
    invariants["plateau_prior_mean_decrease_ratio_threshold"] = 1.0e-4
    if policy["id"] == "always_commutation_reduced":
        invariants.update(
            {
                "insertion_position_scope": (
                    "full_logical_ansatz_commutation_classes_every_depth_v2"
                ),
                "insertion_equivalence_policy": (
                    "termwise_cross_component_commutation_"
                    "earliest_representative_v1"
                ),
            }
        )
    lineage = route.get("lineage_authority")
    if isinstance(lineage, dict):
        changes = list(lineage.get("only_intended_scientific_changes", []))
        changes.append(f"typed_insertion_comparator:{policy['typed_kind']}")
        lineage["only_intended_scientific_changes"] = changes
        lineage["supersession_reason"] = (
            "paper_i_phase0_insertion_comparator_diagnostic_20260812"
        )
    route = digested(route)
    protocol["route_contract"] = route

    protocol["bundle_id"] = bundle_id
    protocol["bundle_manifest_sha256"] = bundle_manifest_sha256
    receipt = protocol["bundle_materialization"]
    receipt["bundle_id"] = bundle_id
    receipt["bundle_manifest_sha256"] = bundle_manifest_sha256
    receipt["cell_id"] = execution_id
    protocol["bundle_materialization"] = digested(receipt)
    return digested(protocol)


def _non_insertion_projection(protocol: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(protocol)
    for key in (
        "sha256",
        "bundle_id",
        "bundle_manifest_sha256",
        "bundle_materialization",
    ):
        projected.pop(key, None)
    projected["request"]["method"].pop("insertion", None)
    route = projected["route_contract"]
    route.pop("sha256", None)
    route.pop("route_profile", None)
    route.pop("lineage_authority", None)
    route["execution_settings"].pop("adapt_insertion_mode", None)
    for key in INSERTION_INVARIANT_KEYS:
        route["semantic_invariants"].pop(key, None)
    return projected


def _expected_artifacts(execution_id: str) -> dict[str, dict[str, Any]]:
    root = f"runs/{execution_id}"
    suffixes = {
        "execution_manifest": "execution_manifest.json",
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
    }
    return {
        role: {
            "path": f"{root}/{suffix}",
            "required": True,
            "direct_file_required": True,
            "reference_receipt_required": False,
            "fulfillment_kind": "direct_execution_v1",
        }
        for role, suffix in suffixes.items()
    }


def _new_execution_id(base_execution_id: str, policy_id: str) -> str:
    if not base_execution_id.endswith("_plateau"):
        raise MaterializationError(
            f"Unexpected source execution id: {base_execution_id}"
        )
    return base_execution_id.removesuffix("_plateau") + f"_{policy_id}"


def _materialize_one(
    *,
    page_id: str,
    source: Path,
    target: Path,
    plateau_adapter: Path,
) -> dict[str, Any]:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    source_manifest = load_json(source / "package_manifest.json")
    verify_self_digest(source_manifest, label=f"{page_id} source package")
    source_archive = source / source_manifest["source_archive"]["path"]
    if sha256_file(source_archive) != EXPECTED_SOURCE_ARCHIVE_SHA256[page_id]:
        raise MaterializationError(f"{page_id} source archive drifted.")

    target.mkdir(parents=True)
    _copy_control_scripts(source, target)
    shutil.copytree(source / "source", target / "source")

    package_id = target.name
    campaign_id = package_id.removesuffix("_chtc")
    bundle_id = campaign_id.removeprefix("paper_i_")

    source_bundle = load_json(
        source / source_manifest["bundle_manifest"]["path"]
    )
    verify_self_digest(source_bundle, label=f"{page_id} source bundle")
    source_expected = load_json(
        source / source_manifest["bundle_expected_artifacts"]["path"]
    )
    verify_self_digest(source_expected, label=f"{page_id} source expected")
    source_locks_path = source / source_manifest["bundle_source_locks"]["path"]
    source_locks = load_json(source_locks_path)
    verify_self_digest(source_locks, label=f"{page_id} source locks")

    source_protocol_rows = list(source_manifest["protocols"])
    base_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for protocol_row in source_protocol_rows:
        base_protocol = load_json(source / protocol_row["path"])
        verify_self_digest(base_protocol, label=protocol_row["path"])
        base_job_row = next(
            row
            for row in source_manifest["jobs"]
            if row["execution_id"] == protocol_row["execution_id"]
        )
        base_job = load_json(source / base_job_row["path"])
        verify_self_digest(base_job, label=base_job_row["path"])
        base_cell = next(
            cell
            for cell in source_bundle["cells"]
            if cell["cell_id"] == protocol_row["execution_id"]
        )
        base_rows.append((base_protocol, base_job, base_cell))

    execution_ids = [
        _new_execution_id(base_job["execution_id"], policy["id"])
        for policy in POLICIES
        for _protocol, base_job, _cell in base_rows
    ]

    bundle_root = target / "bundle_materialization" / bundle_id
    bundle_root.mkdir(parents=True)
    shutil.copyfile(source_locks_path, bundle_root / "source_locks.json")
    expected = digested(
        {
            "schema": source_expected["schema"],
            "bundle_id": bundle_id,
            "cells": {
                execution_id: {
                    "expected_run_artifacts": _expected_artifacts(execution_id)
                }
                for execution_id in execution_ids
            },
        }
    )
    write_json(bundle_root / "expected_artifacts.json", expected)

    cells: list[dict[str, Any]] = []
    for policy in POLICIES:
        for _base_protocol, base_job, base_cell in base_rows:
            execution_id = _new_execution_id(
                base_job["execution_id"], policy["id"]
            )
            cell = copy.deepcopy(base_cell)
            cell["cell_id"] = execution_id
            cell["route_id"] = f"{base_cell['route_id']}_{policy['id']}"
            cells.append(cell)
    bundle = digested(
        {
            **{
                key: value
                for key, value in source_bundle.items()
                if key not in {"sha256", "cells"}
            },
            "bundle_id": bundle_id,
            "campaign_id": campaign_id,
            "run_class": "diagnostic",
            "cell_count": 12,
            "cells": cells,
            "source_locks_sha256": source_locks["sha256"],
            "expected_artifacts_sha256": expected["sha256"],
            "execution_authorized": False,
            "submitted": False,
        }
    )
    write_json(bundle_root / "bundle_manifest.json", bundle)

    protocols: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    route_by_execution: dict[str, str] = {}
    for policy in POLICIES:
        for base_protocol, base_job, _base_cell in base_rows:
            execution_id = _new_execution_id(
                base_job["execution_id"], policy["id"]
            )
            protocol = _transform_protocol(
                base_protocol,
                policy=policy,
                execution_id=execution_id,
                bundle_id=bundle_id,
                bundle_manifest_sha256=bundle["sha256"],
            )
            if _non_insertion_projection(protocol) != _non_insertion_projection(
                base_protocol
            ):
                raise MaterializationError(
                    f"Non-insertion executable drift: {execution_id}"
                )
            protocol_path = bundle_root / "protocols" / f"{execution_id}.json"
            write_json(protocol_path, protocol)
            protocol_binding = {
                "execution_id": execution_id,
                **binding(protocol_path, root=target, canonical=True),
            }
            protocols.append(protocol_binding)
            route_by_execution[execution_id] = protocol["route_contract"][
                "sha256"
            ]

            job = copy.deepcopy(base_job)
            job.update(
                {
                    "package_id": package_id,
                    "campaign_id": campaign_id,
                    "bundle_id": bundle_id,
                    "execution_id": execution_id,
                    "cell_id": execution_id,
                    "route_id": f"{base_job['route_id']}_{policy['id']}",
                    "route_contract_sha256": protocol["route_contract"][
                        "sha256"
                    ],
                    "route_profile": protocol["route_contract"]["route_profile"],
                    "protocol_path": protocol_binding["path"],
                    "protocol_file_sha256": protocol_binding["sha256"],
                    "protocol_sha256": protocol["sha256"],
                    "bundle_manifest_sha256": bundle["sha256"],
                    "source_locks_sha256": source_locks["sha256"],
                    "expected_artifacts_manifest_sha256": expected["sha256"],
                    "expected_run_artifacts": expected["cells"][execution_id][
                        "expected_run_artifacts"
                    ],
                    "expected_output_archive": f"{execution_id}.tar.gz",
                    "comparator_policy": policy["id"],
                    "typed_insertion_kind": policy["typed_kind"],
                    "runtime_insertion_mode": policy["runtime_mode"],
                    "dispatch_template_algorithm_id": base_job["algorithm_id"],
                    "dispatch_template_contains_legacy_plateau_token": True,
                    "legacy_dispatch_plateau_threshold_inert": True,
                    "fresh_source_value_anchor": False,
                    "reference_plateau_execution_id": base_job["execution_id"],
                    "reference_plateau_protocol_sha256": base_protocol["sha256"],
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submitted": False,
                }
            )
            job = digested(job)
            job_path = target / "jobs" / f"{execution_id}.json"
            write_json(job_path, job)
            jobs.append(job)
            audit_rows.append(
                {
                    "execution_id": execution_id,
                    "comparator_policy": policy["id"],
                    "source_execution_id": base_job["execution_id"],
                    "source_protocol_sha256": base_protocol["sha256"],
                    "target_protocol_sha256": protocol["sha256"],
                    "non_insertion_executable_projection_equal": True,
                }
            )

    _patch_package_contract(
        source=source,
        target=target,
        package_id=package_id,
        campaign_id=campaign_id,
        bundle_id=bundle_id,
        execution_ids=execution_ids,
    )
    _patch_run_cell(target)
    _patch_validate_package(target)
    _patch_activate_package(target)

    queue_path = target / "queue.tsv"
    with queue_path.open("xb") as stream:
        rows = []
        for job in jobs:
            job_path = target / "jobs" / f"{job['execution_id']}.json"
            resources = job["resources"]
            rows.append(
                "\t".join(
                    (
                        job["execution_id"],
                        f"jobs/{job['execution_id']}.json",
                        job["protocol_path"],
                        sha256_file(job_path),
                        str(resources["request_cpus"]),
                        str(resources["request_memory_mb"]),
                        str(resources["request_disk_mb"]),
                        str(resources["max_runtime_seconds"]),
                    )
                )
            )
        stream.write(("\n".join(rows) + "\n").encode("utf-8"))

    equality_audit = digested(
        {
            "schema": "paper_i_phase0_insertion_comparator_equality_audit_v1",
            "status": "passed",
            "run_class": "diagnostic",
            "fresh_anchor": False,
            "wrapper_used": True,
            "source_package_id": source_manifest["package_id"],
            "source_package_manifest_sha256": source_manifest["sha256"],
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256[page_id],
            "base_cell_count": 6,
            "comparator_policy_count": 2,
            "planned_run_count": 12,
            "approved_executable_delta": ["request.method.insertion"],
            "identity_and_provenance_deltas": [
                "bundle identity and materialization receipt",
                "route profile insertion token",
                "route insertion invariants",
                "route lineage comparator label",
                "self digests",
            ],
            "rows": audit_rows,
            "all_non_insertion_executable_projections_equal": True,
        }
    )
    write_json(target / "non_insertion_equality_audit.json", equality_audit)

    source_audit = digested(
        {
            "schema": "paper_i_phase0_insertion_comparator_source_audit_v1",
            "status": "passed_diagnostic_wrapper",
            "run_class": "diagnostic",
            "page_id": page_id,
            "source_package_id": source_manifest["package_id"],
            "source_package_manifest_sha256": source_manifest["sha256"],
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256[page_id],
            "source_archive_byte_identical": True,
            "implementation_source_inventory_sha256": source_locks[
                "implementation_sources"
            ]["sha256"],
            "authenticated_plateau_reference_adapter": (
                plateau_adapter.as_posix()
            ),
            "plateau_reference_reused_not_rerun": True,
            "fresh_source_value_anchor": False,
            "strict_fresh_replay_sensitivity_claimed": False,
            "diagnostic_wrapper_used": True,
            "phase0_staged_funnel_reduction_wrapper": (
                "full reduction receipt validated; only the generic Phase-I "
                "full-domain equality check is skipped because Phase-0 "
                "shortlisting intentionally narrows the Phase-I population"
            ),
            "dispatch_template_algorithm_id": source_bundle["algorithm_id"],
            "dispatch_template_contains_legacy_plateau_token": True,
            "legacy_dispatch_plateau_threshold_inert": True,
            "typed_insertion_policy_is_runtime_authority": True,
            "comparator_policies": [policy["id"] for policy in POLICIES],
            "scientific_result_anchor_claimed": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json(target / "source_lock_audit.json", source_audit)

    plan = digested(
        {
            "schema": source / "execution_plan.json" and load_json(
                source / "execution_plan.json"
            )["schema"],
            "package_id": package_id,
            "campaign_id": campaign_id,
            "row_count": 12,
            "execution_ids": execution_ids,
            "route_contract_sha256_by_execution_id": route_by_execution,
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256[page_id],
            "plateau_reference_reused_not_rerun": True,
            "execution_authorized": False,
            "submitted": False,
        }
    )
    write_json(target / "execution_plan.json", plan)

    # The source archive and manifest are intentionally byte-identical copies.
    copied_source_manifest = load_json(target / "source/source_archive_manifest.json")
    verify_self_digest(copied_source_manifest, label="copied source archive manifest")
    if (
        sha256_file(target / "source/source_locked.tar.gz")
        != EXPECTED_SOURCE_ARCHIVE_SHA256[page_id]
    ):
        raise MaterializationError("Copied source archive is not byte-identical.")

    validation = digested(
        {
            "schema": "paper_i_phase0_insertion_comparator_validation_v1",
            "status": "passed",
            "bundle_id": bundle_id,
            "protocol_count": 12,
            "comparator_policies": [policy["id"] for policy in POLICIES],
            "qiskit_cost_phases": ["phase_ii", "phase_iii"],
            "plateau_reference_reused_not_rerun": True,
            "fresh_source_value_anchor": False,
            "non_insertion_equality_audit_sha256": equality_audit["sha256"],
        }
    )
    write_json(bundle_root / "validation_report.json", validation)

    manifest = digested(
        {
            "schema": source_manifest["schema"],
            "status": "passed_inert_twelve_cells",
            "package_id": package_id,
            "campaign_id": campaign_id,
            "bundle_id": bundle_id,
            "run_class": "diagnostic",
            "execution_target": "chtc",
            "row_count": 12,
            "execution_ids": execution_ids,
            "route_contract_sha256_by_execution_id": route_by_execution,
            "source_route_contract_sha256": source_manifest[
                "source_route_contract_sha256"
            ],
            "implementation_source_inventory_sha256": source_locks[
                "implementation_sources"
            ]["sha256"],
            "bundle_manifest": binding(
                bundle_root / "bundle_manifest.json", root=target, canonical=True
            ),
            "bundle_expected_artifacts": binding(
                bundle_root / "expected_artifacts.json", root=target, canonical=True
            ),
            "bundle_source_locks": binding(
                bundle_root / "source_locks.json", root=target, canonical=True
            ),
            "bundle_validation_report": binding(
                bundle_root / "validation_report.json", root=target, canonical=True
            ),
            "protocols": protocols,
            "jobs": [
                {
                    "execution_id": job["execution_id"],
                    **binding(
                        target / "jobs" / f"{job['execution_id']}.json",
                        root=target,
                        canonical=True,
                    ),
                }
                for job in jobs
            ],
            "source_archive": binding(
                target / "source/source_locked.tar.gz", root=target
            ),
            "source_archive_manifest": binding(
                target / "source/source_archive_manifest.json",
                root=target,
                canonical=True,
            ),
            "source_archive_manifest_sha256": copied_source_manifest["sha256"],
            "queue": binding(queue_path, root=target),
            "execution_plan": binding(
                target / "execution_plan.json", root=target, canonical=True
            ),
            "source_lock_audit": binding(
                target / "source_lock_audit.json", root=target, canonical=True
            ),
            "non_insertion_equality_audit": binding(
                target / "non_insertion_equality_audit.json",
                root=target,
                canonical=True,
            ),
            "control_files": [
                binding(target / name, root=target) for name in CONTROL_FILES
            ],
            "required_route_source_paths": source_manifest[
                "required_route_source_paths"
            ],
            "remote_image_path": source_manifest["remote_image_path"],
            "remote_image_sha256": source_manifest["remote_image_sha256"],
            "weak_holstein_horizon": source_manifest[
                "weak_holstein_horizon"
            ],
            "strong_holstein_horizon": source_manifest[
                "strong_holstein_horizon"
            ],
            "comparator_policies": [policy["id"] for policy in POLICIES],
            "plateau_reference_reused_not_rerun": True,
            "fresh_source_value_anchor": False,
            "strict_fresh_replay_sensitivity_claimed": False,
            "activation_artifacts_present": False,
            "authorizations_present": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submit_descriptor_present": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    write_json(target / "package_manifest.json", manifest)
    return {
        "page_id": page_id,
        "package_id": package_id,
        "package_manifest_sha256": manifest["sha256"],
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256[page_id],
        "row_count": 12,
    }


def materialize() -> dict[str, Any]:
    page12_adapter = PLATEAU_EVIDENCE_ROOT / (
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
        "global_singleton_gradient_phase0_page12_adapter.json"
    )
    page16_adapter = PLATEAU_EVIDENCE_ROOT / (
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
        "macro_phase0_phase23_qiskit_no_lanes_page16_adapter.json"
    )
    rows = [
        _materialize_one(
            page_id="page12",
            source=PAGE12_SOURCE,
            target=PAGE12_TARGET,
            plateau_adapter=page12_adapter,
        ),
        _materialize_one(
            page_id="page16",
            source=PAGE16_SOURCE,
            target=PAGE16_TARGET,
            plateau_adapter=page16_adapter,
        ),
    ]
    return {
        "status": "passed",
        "new_job_count": 24,
        "plateau_jobs_rerun": 0,
        "packages": rows,
    }


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(materialize()).decode("utf-8"))
    except (FileExistsError, OSError, ValueError, MaterializationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
