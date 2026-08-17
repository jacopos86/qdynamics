#!/usr/bin/env python3
"""Build the inert, one-row-blocked Page-9 strong-sector continuation package."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import (  # noqa: E402
    ALGORITHM_ID,
    BASE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    BASE_PACKAGE_MANIFEST_FILE_SHA256,
    BASE_PACKAGE_RELATIVE,
    BASE_RUNNER_SHA256,
    BASE_SOURCE_ARCHIVE_SHA256,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CONTROLLER_AFTER_SHA256,
    CONTROLLER_BEFORE_SHA256,
    CONTROLLER_REGRESSION,
    CONTROLLER_RELATIVE_PATH,
    CONTROLLER_REPAIR_ID,
    CONTROL_FILES,
    GENERATED_PATHS,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_SCHEMA,
    REGIMES,
    RESOURCE_ENVELOPE,
    ROUTE_CONTRACT_SHA256,
    ROUTE_ID,
    ROUTE_PROFILE,
    RUN_CLASS,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    VISIBLE_ADAPTER_CANONICAL_SHA256,
    VISIBLE_ADAPTER_FILE_SHA256,
    VISIBLE_ADAPTER_RELATIVE,
    VISIBLE_ADAPTER_SCHEMA,
    VISIBLE_PAGE_ID,
    PackageContractError,
    canonical_json_bytes,
    continuation_execution_id,
    digested,
    expected_execution_ids,
    file_binding,
    json_binding,
    load_json,
    prefix_projection,
    repo_root_from_script,
    sha256_file,
    source_execution_id,
    verify_self_digest,
)


REPO_ROOT = repo_root_from_script(__file__)
BASE_PACKAGE = REPO_ROOT / BASE_PACKAGE_RELATIVE
VISIBLE_ADAPTER = REPO_ROOT / VISIBLE_ADAPTER_RELATIVE
RESOLVER = (
    REPO_ROOT
    / "agent_guidance/skills/shared/scripts/resolve_visible_settings.py"
)


def _write_bytes(path: Path, data: bytes, *, created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        created.append(path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    created: list[Path],
) -> None:
    _write_bytes(path, canonical_json_bytes(value) + b"\n", created=created)


def _verified_base_manifest() -> dict[str, Any]:
    path = BASE_PACKAGE / "package_manifest.json"
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != BASE_PACKAGE_MANIFEST_FILE_SHA256
    ):
        raise PackageContractError("Page-9 v3 package manifest bytes drifted.")
    payload = load_json(path, label="Page-9 v3 package manifest")
    if (
        verify_self_digest(payload, label="Page-9 v3 package manifest")
        != BASE_PACKAGE_MANIFEST_CANONICAL_SHA256
        or payload.get("child_route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
        or payload.get("source_archive", {}).get("sha256")
        != BASE_SOURCE_ARCHIVE_SHA256
        or payload.get("row_count") != 6
    ):
        raise PackageContractError("Page-9 v3 package identity drifted.")
    runner = BASE_PACKAGE / "run_cell.py"
    if sha256_file(runner) != BASE_RUNNER_SHA256:
        raise PackageContractError("Page-9 v3 source runner bytes drifted.")
    return payload


def _verified_adapter() -> dict[str, Any]:
    if (
        not VISIBLE_ADAPTER.is_file()
        or VISIBLE_ADAPTER.is_symlink()
        or sha256_file(VISIBLE_ADAPTER) != VISIBLE_ADAPTER_FILE_SHA256
    ):
        raise PackageContractError("Visible Page-9 adapter bytes drifted.")
    payload = load_json(VISIBLE_ADAPTER, label="visible Page-9 adapter")
    if (
        verify_self_digest(payload, label="visible Page-9 adapter")
        != VISIBLE_ADAPTER_CANONICAL_SHA256
        or payload.get("schema") != VISIBLE_ADAPTER_SCHEMA
        or payload.get("page_id") != VISIBLE_PAGE_ID
        or payload.get("route_profile") != ROUTE_PROFILE
        or payload.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError("Visible Page-9 adapter identity drifted.")
    return payload


def _base_runner() -> ModuleType:
    local_contract = sys.modules.get("package_contract")
    original_path = list(sys.path)
    sys.modules.pop("package_contract", None)
    sys.path.insert(0, BASE_PACKAGE.as_posix())
    try:
        spec = importlib.util.spec_from_file_location(
            "page9_v3_source_runner", BASE_PACKAGE / "run_cell.py"
        )
        if spec is None or spec.loader is None:
            raise PackageContractError("Cannot load the Page-9 v3 source runner.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        sys.modules.pop("package_contract", None)
        if local_contract is not None:
            sys.modules["package_contract"] = local_contract


def _bound_base_job(
    manifest: Mapping[str, Any], regime: str
) -> tuple[Path, dict[str, Any], Mapping[str, Any]]:
    identifier = source_execution_id(regime)
    rows = manifest.get("jobs")
    if not isinstance(rows, list):
        raise PackageContractError("Page-9 v3 job inventory is absent.")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("execution_id") == identifier
    ]
    if len(matches) != 1:
        raise PackageContractError(f"Page-9 source job is not unique: {regime}")
    row = matches[0]
    path = BASE_PACKAGE / str(row["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise PackageContractError(f"Page-9 source job bytes drifted: {regime}")
    job = load_json(path, label=f"Page-9 source job {regime}")
    if (
        verify_self_digest(job, label=f"Page-9 source job {regime}")
        != row.get("canonical_sha256")
        or job.get("execution_id") != identifier
        or job.get("target_horizon") != SOURCE_HORIZON
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("algorithm_id") != ALGORITHM_ID
        or job.get("candidate_adapter_id") != CANDIDATE_ADAPTER_ID
    ):
        raise PackageContractError(f"Page-9 source job identity drifted: {regime}")
    return path, job, row


def _adapter_cell(adapter: Mapping[str, Any], regime: str) -> Mapping[str, Any]:
    cells = adapter.get("cells")
    if not isinstance(cells, list):
        raise PackageContractError("Visible adapter cells are absent.")
    matches = [
        cell
        for cell in cells
        if isinstance(cell, Mapping) and cell.get("regime_id") == regime
    ]
    if len(matches) != 1:
        raise PackageContractError(f"Visible Page-9 cell is not unique: {regime}")
    return matches[0]


def _verify_external_binding(raw: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise PackageContractError(f"{label} binding is absent.")
    path = Path(str(raw.get("path", "")))
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or sha256_file(path) != raw.get("sha256")
    ):
        raise PackageContractError(f"{label} bytes drifted.")
    return path, dict(raw)


def _triplet_from_receipt(
    receipt: Mapping[str, Any], *, execution_id: str
) -> list[dict[str, Any]]:
    rows = receipt.get("artifacts")
    if not isinstance(rows, list):
        raise PackageContractError("Source worker receipt lacks artifacts.")
    selected = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("path", "")).startswith(f"runs/{execution_id}/")
        and "/checkpoints/current" in str(row.get("path", ""))
    ]
    roles: dict[str, dict[str, Any]] = {}
    for row in selected:
        path = str(row.get("path", ""))
        name = Path(path).name
        if name == "current.json":
            role = "checkpoint"
        elif ".estimator_call_ledger_checkpoint." in name:
            role = "estimator_ledger_checkpoint"
        elif ".verified_singleton_resume." in name:
            role = "verified_resume_sidecar"
        else:
            continue
        if role in roles:
            raise PackageContractError(f"Duplicate resume role: {role}")
        sha = str(row.get("sha256", ""))
        if role != "checkpoint" and f".{sha[:16]}.json" not in name:
            raise PackageContractError(f"Content-addressed sidecar name drifted: {name}")
        roles[role] = {
            "role": role,
            "source_path": path,
            "archive_member": f"./{path}",
            "materialized_path": name,
            "sha256": sha,
            "size_bytes": int(row.get("size_bytes", -1)),
        }
    expected = {
        "checkpoint",
        "estimator_ledger_checkpoint",
        "verified_resume_sidecar",
    }
    if set(roles) != expected or any(row["size_bytes"] <= 0 for row in roles.values()):
        raise PackageContractError("Source checkpoint triplet is not pointer-closed.")
    return [roles[role] for role in sorted(roles)]


def _source_resume_contract(
    *,
    adapter: Mapping[str, Any],
    regime: str,
    source_job: Mapping[str, Any],
    created: list[Path],
) -> dict[str, Any]:
    cell = _adapter_cell(adapter, regime)
    route = cell.get("phase3_qiskit_no_lanes")
    if regime == "strong_strong_u8":
        if route is not None or cell.get("current_status") != "pending_on_chtc":
            raise PackageContractError("Strong--strong is no longer the expected blocked row.")
        return {
            "state": "blocked_predecessor_terminal_missing",
            "source_execution_id": source_job["execution_id"],
            "required_completion_evidence": [
                "updated_visible_page9_adapter_with_complete_cell",
                "terminal_k50_worker_receipt",
                "terminal_k50_full_archive",
                "terminal_k50_summary",
            ],
            "remote_full_archive": None,
            "source_worker_receipt": None,
            "resume_triplet": None,
            "prefix_anchor": None,
        }
    if not isinstance(route, Mapping) or route.get("status") != "complete":
        raise PackageContractError(f"Page-9 predecessor is not complete: {regime}")
    bindings = route.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise PackageContractError(f"Page-9 source bindings are absent: {regime}")
    remote = bindings.get("remote_full_archive")
    if (
        not isinstance(remote, Mapping)
        or not str(remote.get("preserved_location", "")).startswith(
            "/staging/jsstrobel/"
        )
        or int(remote.get("size_bytes", -1)) <= 0
    ):
        raise PackageContractError(f"Remote archive binding drifted: {regime}")
    receipt_path, receipt_binding = _verify_external_binding(
        bindings.get("worker_receipt"), label=f"{regime} worker receipt"
    )
    receipt = load_json(receipt_path, label=f"{regime} worker receipt")
    verify_self_digest(receipt, label=f"{regime} worker receipt")
    if (
        receipt.get("status") != "passed"
        or receipt.get("execution_id") != source_job["execution_id"]
        or receipt.get("job_spec_sha256") != source_job["sha256"]
        or receipt.get("controller_rounds_completed") != SOURCE_HORIZON
    ):
        raise PackageContractError(f"Page-9 worker receipt drifted: {regime}")
    triplet = _triplet_from_receipt(
        receipt, execution_id=str(source_job["execution_id"])
    )
    summary_path, summary_binding = _verify_external_binding(
        bindings.get("summary"), label=f"{regime} summary"
    )
    summary = load_json(summary_path, label=f"{regime} summary")
    projection = prefix_projection(summary)
    anchor_path = PACKAGE_DIR / "prefix_anchors" / f"{regime}.json"
    _write_json(anchor_path, projection, created=created)
    return {
        "state": "remote_archive_preserved_materialization_pending",
        "source_execution_id": source_job["execution_id"],
        "remote_full_archive": {
            "path": remote["preserved_location"],
            "sha256": remote["sha256"],
            "size_bytes": int(remote["size_bytes"]),
        },
        "source_worker_receipt": receipt_binding,
        "source_summary": summary_binding,
        "resume_triplet": triplet,
        "triplet_pointer_closure": (
            "receipt_exact_members_plus_runtime_AcceptedStateResume_v1"
        ),
        "prefix_anchor": json_binding(anchor_path, relative_to=PACKAGE_DIR),
    }


def _request_without_horizon(request: Any) -> dict[str, Any]:
    payload = request.to_dict()
    payload["execution"]["stop"].pop("maximum_controller_rounds", None)
    return payload


def _derive_protocol(
    *,
    base: ModuleType,
    source_job_path: Path,
    continuation_id: str,
    regime: str,
    bundle_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _job, _manifest, source, problem, temporary = base._prepare(source_job_path)
    try:
        from pipelines.static_adapt.ra_adapt.bundles import (
            BundleCellSpec,
            _bundle_protocol_materialization_authority,
        )
        from pipelines.static_adapt.ra_adapt.engine import build_resolved_ra_protocol

        source_authority = source.bundle_materialization
        if source_authority is None:
            raise PackageContractError("Source protocol lost bundle authority.")
        request = replace(
            source.request,
            execution=replace(
                source.request.execution,
                stop=replace(
                    source.request.execution.stop,
                    maximum_controller_rounds=TARGET_HORIZON,
                ),
            ),
        )
        cell = BundleCellSpec(
            cell_id=continuation_id,
            stage="page9_strong_sector_r50_to_r70_continuation",
            regime_id=regime,
            nph=7,
            route_id=ROUTE_ID,
            algorithm_id=ALGORITHM_ID,
            selector_family="ra_adapt",
            candidate_representation="single_pauli_word_v1",
            horizon=TARGET_HORIZON,
            source_lock_id=source_authority.source_lock_id,
        )
        authority = _bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id=BUNDLE_ID,
            bundle_manifest_sha256=bundle_manifest_sha256,
            source_locks_sha256=source_authority.source_locks_sha256,
            source_lock_refs=source.source_locks,
            active_gradient_policy=source.active_gradient_policy,
            resource_weighting_scope=source.resource_weighting_scope,
        )
        derived = build_resolved_ra_protocol(
            problem,
            request,
            materialization_authority=authority,
        )
        if (
            int(source.horizon) != SOURCE_HORIZON
            or int(derived.horizon) != TARGET_HORIZON
            or source.route_contract != derived.route_contract
            or derived.route_contract.get("sha256") != ROUTE_CONTRACT_SHA256
            or derived.route_contract.get("route_profile") != ROUTE_PROFILE
            or _request_without_horizon(source.request)
            != _request_without_horizon(derived.request)
            or source.source_locks != derived.source_locks
            or derived.algorithm_id != ALGORITHM_ID
        ):
            raise PackageContractError(f"Derived protocol drifted: {regime}")
        return source.to_dict(), derived.to_dict()
    finally:
        temporary.cleanup()


def build() -> dict[str, Any]:
    cache_members = [
        path
        for path in PACKAGE_DIR.rglob("*")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
    ]
    if cache_members:
        raise PackageContractError(
            "Bytecode/cache members are forbidden: "
            + ", ".join(path.relative_to(PACKAGE_DIR).as_posix() for path in cache_members)
        )
    for relative in GENERATED_PATHS:
        path = PACKAGE_DIR / relative
        if path.exists() or path.is_symlink():
            raise PackageContractError(f"Refusing an in-place rebuild: {path}")
    for name in CONTROL_FILES:
        if not (PACKAGE_DIR / name).is_file():
            raise PackageContractError(f"Missing control file: {name}")
    base_manifest = _verified_base_manifest()
    adapter = _verified_adapter()
    base = _base_runner()
    created: list[Path] = []
    try:
        source_jobs: dict[str, tuple[Path, dict[str, Any], Mapping[str, Any]]] = {
            regime: _bound_base_job(base_manifest, regime) for regime in REGIMES
        }
        controller_source = REPO_ROOT / CONTROLLER_RELATIVE_PATH
        if sha256_file(controller_source) != CONTROLLER_AFTER_SHA256:
            raise PackageContractError("Accepted-energy-only overlay bytes drifted.")
        source_manifest = load_json(
            BASE_PACKAGE / "source/source_archive_manifest.json",
            label="Page-9 source archive manifest",
        )
        controller_rows = [
            row
            for row in source_manifest.get("members", [])
            if isinstance(row, Mapping)
            and row.get("path") == CONTROLLER_RELATIVE_PATH
        ]
        if (
            len(controller_rows) != 1
            or controller_rows[0].get("sha256") != CONTROLLER_BEFORE_SHA256
        ):
            raise PackageContractError("Page-9 controller-before binding drifted.")
        overlay_path = PACKAGE_DIR / "source_overlay" / CONTROLLER_RELATIVE_PATH
        _write_bytes(overlay_path, controller_source.read_bytes(), created=created)
        regression_path = REPO_ROOT / "test/test_static_adapt_sr_snake_controller.py"
        source_composition = digested(
            {
                "schema": "paper_i_page9_r70_runtime_source_composition_v2",
                "status": "passed",
                "base_package_manifest_sha256": base_manifest["sha256"],
                "base_source_archive_sha256": BASE_SOURCE_ARCHIVE_SHA256,
                "operational_overlay": {
                    "repair_id": CONTROLLER_REPAIR_ID,
                    "path": CONTROLLER_RELATIVE_PATH,
                    "before_sha256": CONTROLLER_BEFORE_SHA256,
                    "after": file_binding(overlay_path, relative_to=PACKAGE_DIR),
                    "semantic_scope": "accepted_energy_roundoff_only",
                    "absolute_tolerance": "128*ulp(max(1,abs(E1),abs(E2)))",
                    "all_non_energy_fields_exact": True,
                    "scientific_protocol_changed": False,
                    "scientific_settings_changed": [],
                    "regression": {
                        "nodeid": CONTROLLER_REGRESSION,
                        **file_binding(regression_path, relative_to=REPO_ROOT),
                    },
                },
                "no_ambient_repo_imports": True,
                "streaming_json_runtime": {
                    "distribution": "ijson",
                    "upstream_version": "3.5.1",
                    "backend": "python",
                    "vendored_surface": "synchronous_parse_only",
                    "ambient_install_required": False,
                    "source": file_binding(
                        PACKAGE_DIR / "vendored_ijson_python.py",
                        relative_to=PACKAGE_DIR,
                    ),
                    "license": file_binding(
                        PACKAGE_DIR / "IJSON_LICENSE.txt",
                        relative_to=PACKAGE_DIR,
                    ),
                },
            }
        )
        source_composition_path = PACKAGE_DIR / "source_composition.json"
        _write_json(source_composition_path, source_composition, created=created)
        source_map = digested(
            {
                "schema": "paper_i_page9_strong3_visible_source_map_v2",
                "figure_label": VISIBLE_PAGE_ID,
                "visible_page9_adapter": {
                    **file_binding(VISIBLE_ADAPTER, relative_to=REPO_ROOT),
                    "canonical_sha256": VISIBLE_ADAPTER_CANONICAL_SHA256,
                },
                "regimes": {
                    regime: {
                        "nph": 7,
                        "methods": {
                            "page9_ra": {
                                "visible_value": _adapter_cell(adapter, regime).get(
                                    "current_status"
                                ),
                                "source_json": source_jobs[regime][0]
                                .relative_to(REPO_ROOT)
                                .as_posix(),
                                "source_sha256": source_jobs[regime][2]["sha256"],
                                "algorithm_id": ALGORITHM_ID,
                                "route_contract": {
                                    "route_profile": ROUTE_PROFILE,
                                    "sha256": ROUTE_CONTRACT_SHA256,
                                },
                                "settings": {
                                    "source_horizon": SOURCE_HORIZON,
                                    "target_horizon": TARGET_HORIZON,
                                    "only_scientific_change": (
                                        "maximum_controller_rounds_50_to_70"
                                    ),
                                },
                            }
                        },
                    }
                    for regime in REGIMES
                },
            }
        )
        source_map_path = PACKAGE_DIR / "visible_source_map.json"
        _write_json(source_map_path, source_map, created=created)
        resolver_bindings: list[dict[str, Any]] = []
        for regime in REGIMES:
            trace_path = PACKAGE_DIR / "resolver_traces" / f"{regime}.json"
            command = [
                sys.executable,
                RESOLVER.relative_to(REPO_ROOT).as_posix(),
                "--source-map",
                source_map_path.relative_to(REPO_ROOT).as_posix(),
                "--target-axis",
                "regimes",
                "--regime",
                regime,
                "--method",
                "page9_ra",
                "--output-json",
                trace_path.relative_to(REPO_ROOT).as_posix(),
            ]
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0 or not trace_path.is_file():
                raise PackageContractError(
                    f"Visible-source resolver failed for {regime}: {result.stderr}"
                )
            created.append(trace_path)
            trace = load_json(trace_path, label=f"{regime} resolver trace")
            if (
                trace.get("status") != "ok"
                or trace.get("source_sha256_match") is not True
                or trace.get("regime_or_case") != regime
                or trace.get("method") != "page9_ra"
                or trace.get("settings_changed") != []
            ):
                raise PackageContractError(f"Resolver trace drifted: {regime}")
            resolver_bindings.append(
                {"regime_id": regime, **file_binding(trace_path, relative_to=PACKAGE_DIR)}
            )

        bundle_manifest = digested(
            {
                "schema": "paper_i_page9_strong3_r70_bundle_manifest_v2",
                "status": "inert_one_predecessor_blocked",
                "bundle_id": BUNDLE_ID,
                "campaign_id": CAMPAIGN_ID,
                "source_package_manifest_sha256": base_manifest["sha256"],
                "source_route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "only_scientific_change": (
                    "request.execution.stop.maximum_controller_rounds_50_to_70"
                ),
                "regimes": list(REGIMES),
                "row_count": 3,
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        bundle_path = PACKAGE_DIR / "bundle_manifest.json"
        _write_json(bundle_path, bundle_manifest, created=created)

        jobs: list[dict[str, Any]] = []
        protocol_bindings: list[dict[str, Any]] = []
        source_audit_rows: list[dict[str, Any]] = []
        queue_lines: list[str] = []
        for regime in REGIMES:
            source_job_path, source_job, source_job_manifest_row = source_jobs[regime]
            identifier = continuation_execution_id(regime)
            source_protocol, derived_protocol = _derive_protocol(
                base=base,
                source_job_path=source_job_path,
                continuation_id=identifier,
                regime=regime,
                bundle_manifest_sha256=bundle_manifest["sha256"],
            )
            protocol_path = PACKAGE_DIR / "protocols" / f"{identifier}.json"
            _write_json(protocol_path, derived_protocol, created=created)
            protocol_binding = json_binding(protocol_path, relative_to=PACKAGE_DIR)
            protocol_bindings.append(
                {"execution_id": identifier, **protocol_binding}
            )
            resume_contract = _source_resume_contract(
                adapter=adapter,
                regime=regime,
                source_job=source_job,
                created=created,
            )
            job = digested(
                {
                    "schema": JOB_SCHEMA,
                    "package_id": PACKAGE_ID,
                    "campaign_id": CAMPAIGN_ID,
                    "bundle_id": BUNDLE_ID,
                    "execution_id": identifier,
                    "source_execution_id": source_job["execution_id"],
                    "regime_id": regime,
                    "nph": 7,
                    "run_class": RUN_CLASS,
                    "execution_target": "chtc",
                    "execution_mode": "authenticated_accepted_state_resume",
                    "source_horizon": SOURCE_HORIZON,
                    "target_horizon": TARGET_HORIZON,
                    "only_scientific_change": (
                        "maximum_controller_rounds_50_to_70"
                    ),
                    "route_id": ROUTE_ID,
                    "route_profile": ROUTE_PROFILE,
                    "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                    "algorithm_id": ALGORITHM_ID,
                    "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
                    "source_package_manifest_sha256": base_manifest["sha256"],
                    "source_job": {
                        "path": source_job_path.relative_to(REPO_ROOT).as_posix(),
                        "sha256": source_job_manifest_row["sha256"],
                        "canonical_sha256": source_job_manifest_row[
                            "canonical_sha256"
                        ],
                        "size_bytes": source_job_manifest_row["size_bytes"],
                    },
                    "source_protocol_sha256": source_protocol["sha256"],
                    "derived_protocol": protocol_binding,
                    "derived_protocol_sha256": derived_protocol["sha256"],
                    "resume_source": resume_contract,
                    "resources": dict(RESOURCE_ENVELOPE),
                    "expected_output_archive": f"{identifier}.tar.gz",
                    "accepted_state_resume_required": True,
                    "triplet_pointer_closure_required": True,
                    "prefix_equality_required": True,
                    "failure_safe_attempt_capture_required": True,
                    "accepted_energy_roundoff_overlay": {
                        "repair_id": CONTROLLER_REPAIR_ID,
                        "before_sha256": CONTROLLER_BEFORE_SHA256,
                        "after_sha256": CONTROLLER_AFTER_SHA256,
                        "all_non_energy_fields_exact": True,
                    },
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submitted": False,
                }
            )
            job_path = PACKAGE_DIR / "jobs" / f"{identifier}.json"
            _write_json(job_path, job, created=created)
            jobs.append(job)
            queue_lines.append(
                "\t".join(
                    (
                        identifier,
                        f"jobs/{identifier}.json",
                        protocol_binding["path"],
                        sha256_file(job_path),
                        str(resume_contract["state"]),
                        str(RESOURCE_ENVELOPE["request_cpus"]),
                        str(RESOURCE_ENVELOPE["request_memory_mb"]),
                        str(RESOURCE_ENVELOPE["request_disk_mb"]),
                        str(RESOURCE_ENVELOPE["max_runtime_seconds"]),
                    )
                )
            )
            source_audit_rows.append(
                {
                    "execution_id": identifier,
                    "regime_id": regime,
                    "source_job_sha256": source_job["sha256"],
                    "source_protocol_sha256": source_protocol["sha256"],
                    "derived_protocol_sha256": derived_protocol["sha256"],
                    "common_route_contract_sha256": ROUTE_CONTRACT_SHA256,
                    "resume_source_state": resume_contract["state"],
                    "non_horizon_route_diff": [],
                }
            )
        queue_path = PACKAGE_DIR / "queue.tsv"
        _write_bytes(
            queue_path,
            ("\n".join(queue_lines) + "\n").encode("utf-8"),
            created=created,
        )
        blocked = [
            job["execution_id"]
            for job in jobs
            if job["resume_source"]["state"]
            == "blocked_predecessor_terminal_missing"
        ]
        if blocked != [continuation_execution_id("strong_strong_u8")]:
            raise PackageContractError("Expected exactly the strong--strong blocker.")
        plan = digested(
            {
                "schema": "paper_i_page9_strong3_r70_execution_plan_v2",
                "status": "blocked_1_of_3_resume_inputs",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "row_count": 3,
                "execution_ids": list(expected_execution_ids()),
                "blocked_execution_ids": blocked,
                "materializable_execution_ids": [
                    continuation_execution_id("weak_strong"),
                    continuation_execution_id("intermediate_strong"),
                ],
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "resources": dict(RESOURCE_ENVELOPE),
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submitted": False,
            }
        )
        plan_path = PACKAGE_DIR / "execution_plan.json"
        _write_json(plan_path, plan, created=created)
        audit = digested(
            {
                "schema": "paper_i_page9_strong3_r70_source_lock_audit_v2",
                "status": "passed_with_strong_strong_resume_blocked",
                "package_id": PACKAGE_ID,
                "visible_page9_adapter": {
                    **file_binding(VISIBLE_ADAPTER, relative_to=REPO_ROOT),
                    "canonical_sha256": VISIBLE_ADAPTER_CANONICAL_SHA256,
                },
                "visible_source_map": json_binding(
                    source_map_path, relative_to=PACKAGE_DIR
                ),
                "resolver_script": file_binding(RESOLVER, relative_to=REPO_ROOT),
                "resolver_traces": resolver_bindings,
                "source_package_manifest_sha256": base_manifest["sha256"],
                "source_archive_sha256": base_manifest["source_archive"]["sha256"],
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "only_scientific_change": "maximum_controller_rounds_50_to_70",
                "common_route_profile": ROUTE_PROFILE,
                "common_route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "rows": source_audit_rows,
                "paper_evidence_adopted": False,
            }
        )
        audit_path = PACKAGE_DIR / "source_lock_audit.json"
        _write_json(audit_path, audit, created=created)
        controls = [
            file_binding(PACKAGE_DIR / name, relative_to=PACKAGE_DIR)
            for name in CONTROL_FILES
        ]
        manifest = digested(
            {
                "schema": PACKAGE_SCHEMA,
                "status": "passed_inert_blocked_1_of_3_resume_inputs",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                "run_class": RUN_CLASS,
                "row_count": 3,
                "execution_ids": list(expected_execution_ids()),
                "blocked_execution_ids": blocked,
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "route_profile": ROUTE_PROFILE,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "base_package": {
                    "path": BASE_PACKAGE_RELATIVE.as_posix(),
                    "manifest_file_sha256": BASE_PACKAGE_MANIFEST_FILE_SHA256,
                    "manifest_canonical_sha256": base_manifest["sha256"],
                    "runner_sha256": BASE_RUNNER_SHA256,
                    "source_archive_sha256": BASE_SOURCE_ARCHIVE_SHA256,
                },
                "source_composition": json_binding(
                    source_composition_path, relative_to=PACKAGE_DIR
                ),
                "bundle_manifest": json_binding(
                    bundle_path, relative_to=PACKAGE_DIR
                ),
                "protocols": protocol_bindings,
                "jobs": [
                    {
                        "execution_id": job["execution_id"],
                        **json_binding(
                            PACKAGE_DIR / "jobs" / f"{job['execution_id']}.json",
                            relative_to=PACKAGE_DIR,
                        ),
                    }
                    for job in jobs
                ],
                "queue": file_binding(queue_path, relative_to=PACKAGE_DIR),
                "execution_plan": json_binding(plan_path, relative_to=PACKAGE_DIR),
                "source_lock_audit": json_binding(
                    audit_path, relative_to=PACKAGE_DIR
                ),
                "visible_source_map": json_binding(
                    source_map_path, relative_to=PACKAGE_DIR
                ),
                "resolver_traces": resolver_bindings,
                "control_files": controls,
                "accepted_state_resume_required": True,
                "triplet_pointer_closure_required": True,
                "prefix_equality_required": True,
                "failure_safe_attempt_capture_required": True,
                "explicit_transfer_output_files_required": True,
                "posix_staging_output_remaps_required": True,
                "activation_artifacts_present": False,
                "submit_descriptor_present": False,
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submitted": False,
                "remote_stage": False,
                "condor_submit": False,
            }
        )
        manifest_path = PACKAGE_DIR / "package_manifest.json"
        _write_json(manifest_path, manifest, created=created)
        return {
            "status": manifest["status"],
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "row_count": 3,
            "blocked_execution_ids": blocked,
            "submission_ready": False,
        }
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for relative in (
            "protocols",
            "jobs",
            "prefix_anchors",
            "resolver_traces",
            "source_overlay/pipelines/static_adapt/sr_snake",
            "source_overlay/pipelines/static_adapt",
            "source_overlay/pipelines",
            "source_overlay",
        ):
            path = PACKAGE_DIR / relative
            try:
                path.rmdir()
            except OSError:
                pass
        raise


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(build()).decode("utf-8"))
    except (OSError, ValueError, PackageContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
