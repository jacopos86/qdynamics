#!/usr/bin/env python3
"""Seal the inert matched six-cell L=3 weak-Holstein package."""
from __future__ import annotations

import gzip
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import *  # noqa: E402,F403

REPO_ROOT = repo_root_from_script(__file__)
sys.path.insert(0, str(REPO_ROOT))
V3_PACKAGE_DIR = REPO_ROOT / V3_PACKAGE_RELATIVE_ROOT

from pipelines.static_adapt.ra_adapt.append import (  # noqa: E402
    APPEND_ADAPT_ALGORITHM_ID,
    build_resolved_append_protocol,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    BundleCellSpec,
    _bundle_protocol_materialization_authority,
    _implementation_source_inventory,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    AppendAdaptRequest,
)
from pipelines.static_adapt.ra_adapt.engine import build_resolved_ra_protocol  # noqa: E402
from pipelines.static_adapt.ra_adapt.l3_page12 import (  # noqa: E402
    PAPER_I_L3_PAGE12_ADAPTER_ID,
    PAPER_I_L3_PAGE12_ALGORITHM_ID,
    PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256,
    PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256,
    PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
    build_paper_i_l3_page12_problem,
    build_paper_i_l3_page12_request,
    paper_i_l3_page12_application_source_contract,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    SRExecutionPolicy,
    SRObservationPolicy,
    SRStopPolicy,
)

BUNDLE_ROOT = PACKAGE_DIR / "bundle_materialization" / BUNDLE_ID


def _verified_binding(
    root: Path,
    raw: Mapping[str, Any],
    *,
    label: str,
    canonical: bool,
) -> tuple[Path, dict[str, Any] | None]:
    path = root / safe_relative_path(raw.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or sha256_file(path) != raw.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != raw.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _validated_v3_anchor() -> dict[str, Any]:
    manifest_path = V3_PACKAGE_DIR / "package_manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_size != V3_PACKAGE_MANIFEST_SIZE_BYTES
        or sha256_file(manifest_path) != V3_PACKAGE_MANIFEST_FILE_SHA256
    ):
        raise PackageContractError("The sealed v3 package-manifest bytes drifted.")
    manifest = load_json(manifest_path, label="sealed v3 package manifest")
    if (
        verify_self_digest(manifest, label="sealed v3 package manifest")
        != V3_PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("package_id") != V3_PACKAGE_ID
        or manifest.get("campaign_id") != V3_CAMPAIGN_ID
        or manifest.get("bundle_id") != V3_BUNDLE_ID
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids") != list(expected_execution_ids())
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("The sealed v3 package identity drifted.")

    canonical_keys = (
        "bundle_manifest",
        "bundle_expected_artifacts",
        "bundle_source_locks",
        "bundle_validation_report",
        "source_archive_manifest",
        "execution_plan",
        "source_lock_audit",
    )
    byte_keys = ("source_archive", "queue")
    for key in canonical_keys:
        raw = manifest.get(key)
        if not isinstance(raw, Mapping):
            raise PackageContractError(f"The sealed v3 {key} binding is absent.")
        _verified_binding(
            V3_PACKAGE_DIR,
            raw,
            label=f"sealed v3 {key}",
            canonical=True,
        )
    for key in byte_keys:
        raw = manifest.get(key)
        if not isinstance(raw, Mapping):
            raise PackageContractError(f"The sealed v3 {key} binding is absent.")
        _verified_binding(
            V3_PACKAGE_DIR,
            raw,
            label=f"sealed v3 {key}",
            canonical=False,
        )
    for key in (
        "control_files",
        "application_source_contracts",
        "jobs",
        "protocols",
    ):
        rows = manifest.get(key)
        if not isinstance(rows, list) or not rows:
            raise PackageContractError(f"The sealed v3 {key} closure is absent.")
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                raise PackageContractError(f"The sealed v3 {key} row is invalid.")
            _verified_binding(
                V3_PACKAGE_DIR,
                raw,
                label=f"sealed v3 {key} row {index}",
                canonical="canonical_sha256" in raw,
            )

    source_archive = manifest.get("source_archive")
    assert isinstance(source_archive, Mapping)
    if (
        source_archive.get("sha256") != V3_SOURCE_ARCHIVE_SHA256
        or int(source_archive.get("size_bytes", -1))
        != V3_SOURCE_ARCHIVE_SIZE_BYTES
    ):
        raise PackageContractError("The sealed v3 source-archive anchor drifted.")
    source_locks_binding = manifest.get("bundle_source_locks")
    assert isinstance(source_locks_binding, Mapping)
    _path, source_locks = _verified_binding(
        V3_PACKAGE_DIR,
        source_locks_binding,
        label="sealed v3 source locks",
        canonical=True,
    )
    assert source_locks is not None
    if (
        source_locks.get("implementation_sources", {}).get("sha256")
        != V3_IMPLEMENTATION_SOURCE_INVENTORY_SHA256
    ):
        raise PackageContractError(
            "The sealed v3 implementation-source inventory drifted."
        )
    return manifest


def _normalized_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("sha256", None)
    normalized["bundle_id"] = "<package-revision-bundle-id>"
    normalized["bundle_manifest_sha256"] = "<package-revision-binding>"
    materialization = dict(normalized["bundle_materialization"])
    materialization.pop("sha256", None)
    materialization["bundle_id"] = "<package-revision-bundle-id>"
    materialization["bundle_manifest_sha256"] = "<package-revision-binding>"
    materialization["source_locks_sha256"] = "<package-revision-binding>"
    materialization["source_lock_refs_sha256"] = "<package-revision-binding>"
    normalized["bundle_materialization"] = materialization
    source_locks = dict(normalized["source_locks"])
    source_locks["source_locks_manifest_sha256"] = "<package-revision-binding>"
    normalized["source_locks"] = source_locks
    return normalized


def _normalized_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("sha256", None)
    normalized["package_id"] = "<package-revision-id>"
    normalized["campaign_id"] = "<package-revision-campaign-id>"
    normalized["bundle_id"] = "<package-revision-bundle-id>"
    normalized["protocol_path"] = Path(str(normalized["protocol_path"])).name
    for key in (
        "protocol_file_sha256",
        "protocol_sha256",
        "bundle_manifest_sha256",
        "source_locks_sha256",
        "expected_artifacts_manifest_sha256",
    ):
        normalized[key] = "<package-revision-binding>"
    resources = dict(normalized["resources"])
    normalized["resources"] = {
        "request_cpus": resources["request_cpus"],
        "max_runtime_seconds": resources["max_runtime_seconds"],
    }
    return normalized


def _normalized_document(
    payload: Mapping[str, Any],
    *,
    identity_keys: tuple[str, ...],
    binding_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("sha256", None)
    for key in identity_keys:
        normalized[key] = f"<package-revision-{key}>"
    for key in binding_keys:
        normalized[key] = "<package-revision-binding>"
    return normalized


def _bound_payload_by_execution(
    root: Path,
    rows: Any,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise PackageContractError(f"{label} closure is absent.")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise PackageContractError(f"{label} binding {index} is invalid.")
        _path, payload = _verified_binding(
            root,
            raw,
            label=f"{label} binding {index}",
            canonical=True,
        )
        assert payload is not None
        execution = str(raw.get("execution_id", ""))
        if execution in result or execution not in expected_execution_ids():
            raise PackageContractError(f"{label} execution closure drifted.")
        result[execution] = payload
    if set(result) != set(expected_execution_ids()):
        raise PackageContractError(f"{label} execution set drifted.")
    return result

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush(); os.fsync(stream.fileno())

def _cells() -> tuple[BundleCellSpec, ...]:
    rows = []
    for regime in REGIMES:
        for method in METHODS:
            rows.append(BundleCellSpec(
                cell_id=execution_id(regime, method),
                stage=f"l3_{regime}_{method}_candidate",
                regime_id=regime,
                nph=3,
                route_id=RA_ROUTE_ID if method == "ra_page12" else APPEND_ROUTE_ID,
                algorithm_id=RA_ALGORITHM_ID if method == "ra_page12" else APPEND_ALGORITHM_ID,
                selector_family="ra_adapt" if method == "ra_page12" else "append_adapt",
                candidate_representation=CANDIDATE_REPRESENTATION,
                horizon=TARGET_HORIZON,
                source_lock_id=source_lock_id(regime),
            ))
    return tuple(rows)

def _expected_artifacts(execution: str) -> dict[str, Any]:
    root = f"runs/{execution}"
    suffixes = {
        "execution_manifest": "execution_manifest.json",
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
    }
    return {role: {"path": f"{root}/{suffix}", "required": True,
                   "direct_file_required": True, "reference_receipt_required": False,
                   "fulfillment_kind": "direct_execution_v1"}
            for role, suffix in suffixes.items()}

def _source_locks(implementation: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cell_locks = {}
    for regime, source in sources.items():
        lock_id = source_lock_id(regime)
        cell_locks[lock_id] = digested({
            "schema": "paper_i_l3_weak_holstein_cell_source_lock_v1",
            "source_lock_id": lock_id,
            "regime_id": regime,
            "application_source_contract_sha256": source["sha256"],
            "problem_request_sha256": source["problem_request_sha256"],
            "hamiltonian_terms_sha256": source["hamiltonian_terms_sha256"],
            "same_cutoff_exact_reference_receipt_sha256": source["same_cutoff_exact_reference"]["receipt_sha256"],
            "ra_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
        })
    return digested({
        "schema": "paper_i_l3_weak_holstein_source_locks_v1",
        "package_id": PACKAGE_ID,
        "implementation_sources": dict(implementation),
        "application_source_contracts": {
            regime: {"path": f"source_authority/{regime}_application_source_contract.json",
                     "canonical_sha256": source["sha256"]}
            for regime, source in sources.items()
        },
        "cell_locks": cell_locks,
        "append_runtime_hash_dependencies": list(APPEND_RUNTIME_SOURCE_DEPENDENCIES),
        "l2_source_protocol_count": 0,
        "l2_cell_source_lock_count": 0,
    })

def _refs(locks: Mapping[str, Any], source: Mapping[str, Any], regime: str) -> dict[str, str]:
    lock_id = source_lock_id(regime); lock = locks["cell_locks"][lock_id]
    return {
        "source_locks_manifest_sha256": str(locks["sha256"]),
        "implementation_source_inventory_sha256": str(locks["implementation_sources"]["sha256"]),
        "cell_source_lock_id": lock_id,
        "cell_source_lock_sha256": str(lock["sha256"]),
        "ed_cutoff_reference_sha256": str(source["same_cutoff_exact_reference"]["receipt_sha256"]),
        APPLICATION_SOURCE_LOCK_KEY: str(source["sha256"]),
        "l3_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
    }

def _source_archive(locks: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for raw in locks["implementation_sources"]["files"]:
        source = REPO_ROOT / str(raw["path"])
        rows.append({"path": str(raw["path"]), "sha256": str(raw["sha256"]),
                     "size_bytes": source.stat().st_size,
                     "source_kind": "verified_current_implementation_inventory"})
    by_path = {row["path"]: row for row in rows}
    for raw in APPEND_RUNTIME_SOURCE_DEPENDENCIES:
        relative = str(raw["path"]); source = REPO_ROOT / relative
        candidate = {"path": relative, "sha256": str(raw["sha256"]),
                     "size_bytes": int(raw["size_bytes"]),
                     "source_kind": "append_runtime_hash_dependency"}
        if relative in by_path and by_path[relative]["sha256"] != candidate["sha256"]:
            raise PackageContractError("Append runtime dependency conflicts with implementation inventory.")
        by_path[relative] = candidate
    rows = sorted(by_path.values(), key=lambda row: row["path"])
    missing = sorted(set(REQUIRED_ROUTE_SOURCE_PATHS) - set(by_path))
    if missing: raise PackageContractError("Source archive omitted required route source: " + ", ".join(missing))
    for row in rows:
        source = REPO_ROOT / row["path"]
        if (not source.is_file() or source.is_symlink() or sha256_file(source) != row["sha256"]
                or source.stat().st_size != row["size_bytes"]):
            raise PackageContractError(f"Source member drifted: {source}")
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    source = REPO_ROOT / row["path"]
                    info = tarfile.TarInfo(row["path"]); info.size = source.stat().st_size
                    info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                    info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
                    with source.open("rb") as stream: archive.addfile(info, stream)
    if (
        archive_path.stat().st_size != V3_SOURCE_ARCHIVE_SIZE_BYTES
        or sha256_file(archive_path) != V3_SOURCE_ARCHIVE_SHA256
    ):
        raise PackageContractError(
            "The v4 source archive is not byte-identical to sealed v3."
        )
    manifest = digested({
        "schema": SOURCE_ARCHIVE_MANIFEST_SCHEMA, "status": "passed",
        "package_id": PACKAGE_ID,
        "implementation_source_inventory_sha256": locks["implementation_sources"]["sha256"],
        "member_count": len(rows), "members": rows,
        "archive": binding(archive_path, root=PACKAGE_DIR), "no_ambient_repo_imports": True,
    })
    _write_json(PACKAGE_DIR / "source/source_archive_manifest.json", manifest)
    return manifest

def _append_request(_execution: str) -> AppendAdaptRequest:
    return AppendAdaptRequest(
        adapter=PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter(),
        execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=50), resume=FreshStart()),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(path=Path("checkpoints/current.json"),
                                             every_controller_rounds=1, keep_history_tail=100),
            estimator_ledger=EstimatorLedgerObservation(path=Path("result/estimator_ledger.json")),
            resource_rounds=(50,),
        ),
    )


def _v3_scientific_equivalence(
    *,
    v3_manifest: Mapping[str, Any],
    bundle_manifest: Mapping[str, Any],
    expected_artifacts: Mapping[str, Any],
    source_locks: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_lock_audit: Mapping[str, Any],
    protocol_bindings: list[dict[str, Any]],
    job_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    def v3_document(key: str) -> dict[str, Any]:
        raw = v3_manifest.get(key)
        if not isinstance(raw, Mapping):
            raise PackageContractError(f"The sealed v3 {key} binding is absent.")
        _path, payload = _verified_binding(
            V3_PACKAGE_DIR,
            raw,
            label=f"sealed v3 {key}",
            canonical=True,
        )
        assert payload is not None
        return payload

    v3_documents = {
        "bundle_manifest": v3_document("bundle_manifest"),
        "expected_artifacts": v3_document("bundle_expected_artifacts"),
        "source_locks": v3_document("bundle_source_locks"),
        "validation_report": v3_document("bundle_validation_report"),
        "source_manifest": v3_document("source_archive_manifest"),
        "source_lock_audit": v3_document("source_lock_audit"),
    }
    v4_documents = {
        "bundle_manifest": dict(bundle_manifest),
        "expected_artifacts": dict(expected_artifacts),
        "source_locks": dict(source_locks),
        "validation_report": dict(validation_report),
        "source_manifest": dict(source_manifest),
        "source_lock_audit": dict(source_lock_audit),
    }
    document_projections = {
        "bundle_manifest": (
            ("bundle_id", "campaign_id"),
            ("source_locks_sha256", "expected_artifacts_sha256"),
        ),
        "expected_artifacts": (("bundle_id",), ()),
        "source_locks": (("package_id",), ()),
        "validation_report": (("bundle_id",), ()),
        "source_manifest": (("package_id",), ()),
        "source_lock_audit": ((), ()),
    }
    materialization_projection_sha256: dict[str, str] = {}
    for key, (identity_keys, binding_keys) in document_projections.items():
        v3_projection = _normalized_document(
            v3_documents[key],
            identity_keys=identity_keys,
            binding_keys=binding_keys,
        )
        v4_projection = _normalized_document(
            v4_documents[key],
            identity_keys=identity_keys,
            binding_keys=binding_keys,
        )
        if v3_projection != v4_projection:
            raise PackageContractError(
                f"The v4 {key} scientific materialization drifted from v3."
            )
        materialization_projection_sha256[key] = canonical_sha256(v4_projection)

    v3_application_rows = v3_manifest.get("application_source_contracts")
    if not isinstance(v3_application_rows, list) or len(v3_application_rows) != 3:
        raise PackageContractError("The sealed v3 application-source closure drifted.")
    v3_application_by_path = {
        str(row.get("path")): row
        for row in v3_application_rows
        if isinstance(row, Mapping)
    }
    application_source_bindings: dict[str, dict[str, Any]] = {}
    for regime in REGIMES:
        relative = f"source_authority/{regime}_application_source_contract.json"
        raw = v3_application_by_path.get(relative)
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"The sealed v3 {regime} application source is absent."
            )
        v3_path, v3_payload = _verified_binding(
            V3_PACKAGE_DIR,
            raw,
            label=f"sealed v3 {regime} application source",
            canonical=True,
        )
        v4_path = PACKAGE_DIR / relative
        if (
            not v4_path.is_file()
            or v4_path.is_symlink()
            or v4_path.read_bytes() != v3_path.read_bytes()
        ):
            raise PackageContractError(
                f"The v4 {regime} application-source bytes drifted from v3."
            )
        assert v3_payload is not None
        application_source_bindings[regime] = {
            "sha256": sha256_file(v4_path),
            "canonical_sha256": v3_payload["sha256"],
            "size_bytes": v4_path.stat().st_size,
            "exact_v3_byte_identity": True,
        }

    v3_protocols = _bound_payload_by_execution(
        V3_PACKAGE_DIR,
        v3_manifest.get("protocols"),
        label="sealed v3 protocols",
    )
    v4_protocols = _bound_payload_by_execution(
        PACKAGE_DIR,
        protocol_bindings,
        label="v4 protocols",
    )
    protocol_projection_sha256: dict[str, str] = {}
    for execution in expected_execution_ids():
        v3_projection = _normalized_protocol(v3_protocols[execution])
        v4_projection = _normalized_protocol(v4_protocols[execution])
        if v3_projection != v4_projection:
            raise PackageContractError(
                f"The v4 {execution} protocol semantics drifted from v3."
            )
        protocol_projection_sha256[execution] = canonical_sha256(v4_projection)

    v3_jobs = _bound_payload_by_execution(
        V3_PACKAGE_DIR,
        v3_manifest.get("jobs"),
        label="sealed v3 jobs",
    )
    v4_jobs = _bound_payload_by_execution(
        PACKAGE_DIR,
        job_bindings,
        label="v4 jobs",
    )
    expected_v3_resources = {
        "ra_page12": {
            "request_cpus": 4,
            "request_memory_mb": 90_112,
            "request_disk_mb": 98_304,
            "max_runtime_seconds": 259_200,
            "basis": "conservative_l3_nph3_page12_candidate_envelope_v1",
        },
        "append_adapt": {
            "request_cpus": 1,
            "request_memory_mb": 65_536,
            "request_disk_mb": 81_920,
            "max_runtime_seconds": 259_200,
            "basis": "conservative_l3_nph3_conventional_append_envelope_v1",
        },
    }
    job_projection_sha256: dict[str, str] = {}
    for execution in expected_execution_ids():
        v3_job = v3_jobs[execution]
        v4_job = v4_jobs[execution]
        method = str(v4_job.get("method", ""))
        if (
            method not in RESOURCE_ENVELOPES
            or v3_job.get("resources") != expected_v3_resources[method]
            or v4_job.get("resources") != RESOURCE_ENVELOPES[method]
        ):
            raise PackageContractError(
                f"The {execution} resource-only revision boundary drifted."
            )
        v3_projection = _normalized_job(v3_job)
        v4_projection = _normalized_job(v4_job)
        if v3_projection != v4_projection:
            raise PackageContractError(
                f"The v4 {execution} job semantics drifted from v3."
            )
        job_projection_sha256[execution] = canonical_sha256(v4_projection)

    source_archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    v3_archive_binding = v3_manifest.get("source_archive")
    assert isinstance(v3_archive_binding, Mapping)
    v3_archive_path, _payload = _verified_binding(
        V3_PACKAGE_DIR,
        v3_archive_binding,
        label="sealed v3 source archive",
        canonical=False,
    )
    if source_archive_path.read_bytes() != v3_archive_path.read_bytes():
        raise PackageContractError("The v4 source-archive bytes drifted from v3.")

    return digested(
        {
            "schema": "paper_i_l3_weak_holstein_v3_v4_equivalence_v1",
            "status": "passed_exact_source_and_scientific_semantics",
            "v3_package_id": V3_PACKAGE_ID,
            "v3_package_manifest_canonical_sha256": (
                V3_PACKAGE_MANIFEST_CANONICAL_SHA256
            ),
            "v3_package_manifest_file_sha256": (
                V3_PACKAGE_MANIFEST_FILE_SHA256
            ),
            "v4_package_id": PACKAGE_ID,
            "allowed_changes": [
                "package_campaign_bundle_batch_and_staging_revision_identity",
                "ra_request_memory_mb_90112_to_65536",
                "ra_request_disk_mb_98304_to_81920",
                "append_request_memory_mb_65536_to_49152",
                "append_request_disk_mb_81920_to_61440",
                "resource_basis_labels",
                "identity_derived_bindings_and_digests",
            ],
            "source_archive": {
                "sha256": V3_SOURCE_ARCHIVE_SHA256,
                "size_bytes": V3_SOURCE_ARCHIVE_SIZE_BYTES,
                "exact_v3_byte_identity": True,
            },
            "implementation_source_inventory_sha256": (
                V3_IMPLEMENTATION_SOURCE_INVENTORY_SHA256
            ),
            "application_source_contracts": application_source_bindings,
            "materialization_projection_sha256": (
                materialization_projection_sha256
            ),
            "protocol_projection_sha256": protocol_projection_sha256,
            "job_projection_sha256": job_projection_sha256,
            "resource_envelopes_v3": expected_v3_resources,
            "resource_envelopes_v4": RESOURCE_ENVELOPES,
            "scientific_semantic_drift_detected": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )

def build() -> dict[str, Any]:
    v3_manifest = _validated_v3_anchor()
    if any((PACKAGE_DIR / name).exists() for name in GENERATED_PATHS):
        raise FileExistsError("Refusing to overwrite an existing package seal.")
    for name in CONTROL_FILES:
        if not (PACKAGE_DIR / name).is_file(): raise PackageContractError(f"Missing control file: {name}")
    if (RA_ALGORITHM_ID != PAPER_I_L3_PAGE12_ALGORITHM_ID or APPEND_ALGORITHM_ID != APPEND_ADAPT_ALGORITHM_ID
            or CANDIDATE_ADAPTER_ID != PAPER_I_L3_PAGE12_ADAPTER_ID
            or APPLICATION_SOURCE_LOCK_KEY != PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY
            or TARGET_ROUTE_CONTRACT_SHA256 != PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256):
        raise PackageContractError("Package constants drifted from named L3 seams.")
    sources = {}; problems = {}
    for regime in REGIMES:
        problem = build_paper_i_l3_page12_problem(regime, nph=3)
        source = paper_i_l3_page12_application_source_contract(problem)
        if source["sha256"] != PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256[regime]:
            raise PackageContractError("Application source lock drifted.")
        problems[regime] = problem; sources[regime] = source
        _write_json(PACKAGE_DIR / f"source_authority/{regime}_application_source_contract.json", source)
    implementation = _implementation_source_inventory(REPO_ROOT)
    if implementation.get("sha256") != V3_IMPLEMENTATION_SOURCE_INVENTORY_SHA256:
        raise PackageContractError(
            "Current implementation sources drifted from the sealed v3 inventory."
        )
    locks = _source_locks(implementation, sources)
    cells = _cells(); BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(BUNDLE_ROOT / "source_locks.json", locks)
    expected = digested({"schema": "paper_i_l3_weak_holstein_expected_artifacts_v1",
                         "bundle_id": BUNDLE_ID,
                         "cells": {cell.cell_id: {"expected_run_artifacts": _expected_artifacts(cell.cell_id)} for cell in cells}})
    _write_json(BUNDLE_ROOT / "expected_artifacts.json", expected)
    bundle_manifest = digested({
        "schema": "paper_i_l3_weak_holstein_bundle_manifest_v1", "bundle_id": BUNDLE_ID,
        "campaign_id": CAMPAIGN_ID, "run_class": RUN_CLASS, "cell_count": len(cells),
        "cells": [cell.to_dict() for cell in cells], "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
        "target_horizon": TARGET_HORIZON, "source_locks_sha256": locks["sha256"],
        "expected_artifacts_sha256": expected["sha256"], "execution_authorized": False, "submitted": False,
    })
    _write_json(BUNDLE_ROOT / "bundle_manifest.json", bundle_manifest)
    protocols = []; jobs = []; queue_rows = []
    for cell in cells:
        method = "ra_page12" if cell.selector_family == "ra_adapt" else "append_adapt"
        source = sources[cell.regime_id]; refs = _refs(locks, source, cell.regime_id)
        policy = ACTIVE_GRADIENT_RA if method == "ra_page12" else ACTIVE_GRADIENT_APPEND
        authority = _bundle_protocol_materialization_authority(
            cell=cell, bundle_id=BUNDLE_ID, bundle_manifest_sha256=bundle_manifest["sha256"],
            source_locks_sha256=locks["sha256"], source_lock_refs=refs,
            active_gradient_policy=policy, resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
        )
        protocol = (build_resolved_ra_protocol(problems[cell.regime_id], build_paper_i_l3_page12_request(),
                                               materialization_authority=authority)
                    if method == "ra_page12" else
                    build_resolved_append_protocol(problems[cell.regime_id], _append_request(cell.cell_id),
                                                   materialization_authority=authority))
        if method == "ra_page12":
            if (protocol.route_contract["sha256"] != TARGET_ROUTE_CONTRACT_SHA256
                    or protocol.request.method.pruning.kind != "off" or protocol.request.method.beam.kind != "off"):
                raise PackageContractError("L3 Page12 RA route drifted.")
            route_sha = TARGET_ROUTE_CONTRACT_SHA256; entrypoint = "run_ra_adapt"
        else:
            if (protocol.selector_identity != APPEND_CONVENTIONAL_SELECTOR_ID
                    or protocol.lineage_authority.get("ra_staged_funnel_invoked") is not False
                    or protocol.algorithm_id != APPEND_ALGORITHM_ID):
                raise PackageContractError("Conventional Append facade drifted.")
            route_sha = digested({"schema": "paper_i_l3_conventional_append_route_v1",
                                  "selector_identity": protocol.selector_identity,
                                  "selector_scope": protocol.selector_scope,
                                  "ra_staged_funnel_invoked": False})["sha256"]
            entrypoint = "run_append_adapt"
        if (protocol.problem.num_sites != 3 or protocol.problem.n_ph_max != 3 or protocol.horizon != 50
                or protocol.optimizer != "powell" or protocol.optimizer_maxiter != 200 or protocol.seeds["adapt"] != 7):
            raise PackageContractError("Matched scientific settings drifted.")
        protocol_path = BUNDLE_ROOT / "protocols" / f"{cell.cell_id}.json"; _write_json(protocol_path, protocol.to_dict())
        protocol_binding = {"execution_id": cell.cell_id, **binding(protocol_path, root=PACKAGE_DIR, canonical=True)}
        resources = RESOURCE_ENVELOPES[method]
        job = digested({
            "schema": JOB_SCHEMA, "package_id": PACKAGE_ID, "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID, "execution_id": cell.cell_id, "cell_id": cell.cell_id,
            "regime_id": cell.regime_id, "method": method, "execution_entrypoint": entrypoint,
            "num_sites": 3, "nph": 3, "target_horizon": 50, "algorithm_id": cell.algorithm_id,
            "route_id": cell.route_id, "route_contract_sha256": route_sha,
            "application_source_contract_sha256": source["sha256"],
            "active_gradient_policy": policy, "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "candidate_representation": CANDIDATE_REPRESENTATION, "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
            "protocol_path": protocol_binding["path"], "protocol_file_sha256": protocol_binding["sha256"],
            "protocol_sha256": protocol.sha256, "bundle_manifest_sha256": bundle_manifest["sha256"],
            "source_locks_sha256": locks["sha256"], "implementation_source_inventory_sha256": implementation["sha256"],
            "expected_artifacts_manifest_sha256": expected["sha256"],
            "expected_run_artifacts": _expected_artifacts(cell.cell_id), "resources": dict(resources),
            "expected_output_archive": f"{cell.cell_id}.tar.gz",
            "fresh_start_contract": {"kind": "fresh_start", "source_checkpoint": None, "resume_archive": None},
            "execution_authorized": False, "submission_authorized": False, "submitted": False,
        })
        job_path = PACKAGE_DIR / "jobs" / f"{cell.cell_id}.json"; _write_json(job_path, job)
        protocols.append(protocol_binding); jobs.append({"execution_id": cell.cell_id, **binding(job_path, root=PACKAGE_DIR, canonical=True)})
        queue_rows.append("\t".join((cell.cell_id, f"jobs/{cell.cell_id}.json", protocol_binding["path"],
                                     sha256_file(job_path), str(resources["request_cpus"]),
                                     str(resources["request_memory_mb"]), str(resources["request_disk_mb"]),
                                     str(resources["max_runtime_seconds"]))))
    validation = digested({
        "schema": "paper_i_l3_weak_holstein_matched_validation_v1", "status": "passed",
        "bundle_id": BUNDLE_ID, "protocol_count": 6, "regimes": list(REGIMES),
        "method_counts": {"ra_page12": 3, "append_adapt": 3}, "distinct_execution_facades": True,
        "ra_pruning": "off", "ra_beam": "off", "target_horizon": 50,
    }); _write_json(BUNDLE_ROOT / "validation_report.json", validation)
    source_manifest = _source_archive(locks)
    queue_path = PACKAGE_DIR / "queue.tsv"
    with queue_path.open("xb") as stream: stream.write(("\n".join(queue_rows) + "\n").encode())
    plan = digested({"schema": EXECUTION_PLAN_SCHEMA, "package_id": PACKAGE_ID, "campaign_id": CAMPAIGN_ID,
                     "row_count": 6, "execution_ids": list(expected_execution_ids()),
                     "execution_entrypoint_counts": {"run_ra_adapt": 3, "run_append_adapt": 3},
                     "source_archive_sha256": source_manifest["archive"]["sha256"],
                     "execution_authorized": False, "submitted": False})
    _write_json(PACKAGE_DIR / "execution_plan.json", plan)
    audit = digested({"schema": "paper_i_l3_weak_holstein_source_lock_audit_v1", "status": "passed",
                      "implementation_source_inventory_sha256": implementation["sha256"],
                      "explicit_append_runtime_dependency_count": len(APPEND_RUNTIME_SOURCE_DEPENDENCIES),
                      "l2_source_protocol_count": 0, "scientific_result_anchor_claimed": False})
    _write_json(PACKAGE_DIR / "source_lock_audit.json", audit)
    equivalence = _v3_scientific_equivalence(
        v3_manifest=v3_manifest,
        bundle_manifest=bundle_manifest,
        expected_artifacts=expected,
        source_locks=locks,
        validation_report=validation,
        source_manifest=source_manifest,
        source_lock_audit=audit,
        protocol_bindings=protocols,
        job_bindings=jobs,
    )
    _write_json(PACKAGE_DIR / SCIENTIFIC_EQUIVALENCE_RELATIVE, equivalence)
    manifest = digested({
        "schema": PACKAGE_MANIFEST_SCHEMA, "status": "passed_inert_matched_six_cell",
        "package_id": PACKAGE_ID, "campaign_id": CAMPAIGN_ID, "bundle_id": BUNDLE_ID,
        "run_class": RUN_CLASS, "execution_target": EXECUTION_TARGET, "row_count": 6,
        "execution_ids": list(expected_execution_ids()), "execution_entrypoint_counts": {"run_ra_adapt": 3, "run_append_adapt": 3},
        "bundle_manifest": binding(BUNDLE_ROOT / "bundle_manifest.json", root=PACKAGE_DIR, canonical=True),
        "bundle_expected_artifacts": binding(BUNDLE_ROOT / "expected_artifacts.json", root=PACKAGE_DIR, canonical=True),
        "bundle_source_locks": binding(BUNDLE_ROOT / "source_locks.json", root=PACKAGE_DIR, canonical=True),
        "bundle_validation_report": binding(BUNDLE_ROOT / "validation_report.json", root=PACKAGE_DIR, canonical=True),
        "application_source_contracts": [binding(PACKAGE_DIR / f"source_authority/{r}_application_source_contract.json", root=PACKAGE_DIR, canonical=True) for r in REGIMES],
        "protocols": protocols, "jobs": jobs, "source_archive": source_manifest["archive"],
        "source_archive_manifest": binding(PACKAGE_DIR / "source/source_archive_manifest.json", root=PACKAGE_DIR, canonical=True),
        "source_archive_manifest_sha256": source_manifest["sha256"], "queue": binding(queue_path, root=PACKAGE_DIR),
        "execution_plan": binding(PACKAGE_DIR / "execution_plan.json", root=PACKAGE_DIR, canonical=True),
        "source_lock_audit": binding(PACKAGE_DIR / "source_lock_audit.json", root=PACKAGE_DIR, canonical=True),
        "scientific_equivalence": binding(
            PACKAGE_DIR / SCIENTIFIC_EQUIVALENCE_RELATIVE,
            root=PACKAGE_DIR,
            canonical=True,
        ),
        "control_files": [binding(PACKAGE_DIR / name, root=PACKAGE_DIR) for name in CONTROL_FILES],
        "required_route_source_paths": list(REQUIRED_ROUTE_SOURCE_PATHS),
        "remote_image_path": REMOTE_IMAGE_PATH, "remote_image_sha256": REMOTE_IMAGE_SHA256,
        "target_horizon": 50, "activation_artifacts_present": False, "authorizations_present": False,
        "execution_authorized": False, "submission_authorized": False, "submission_ready": False,
        "submit_descriptor_present": False, "submitted": False, "remote_stage": False, "condor_submit": False,
    }); _write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    return {"status": manifest["status"], "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"], "source_archive_sha256": source_manifest["archive"]["sha256"],
            "v3_scientific_equivalence_sha256": equivalence["sha256"],
            "row_count": 6}

if __name__ == "__main__":
    try: print(canonical_json_bytes(build()).decode())
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
