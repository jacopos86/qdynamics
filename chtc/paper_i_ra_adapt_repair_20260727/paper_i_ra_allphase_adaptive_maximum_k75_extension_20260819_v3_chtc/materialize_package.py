#!/usr/bin/env python3
"""Materialize the k75 recovery package (v2): 6 live resumes + 3 completions."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]

from package_contract import (
    ARMS,
    AUTH_SCHEMA,
    CAMPAIGN_ID,
    CRASH_TARBALLS,
    IMAGE_PATH,
    IMAGE_SHA256,
    JOB_SCHEMA,
    MANIFEST_SCHEMA,
    PACKAGE_ID,
    PLAN_SCHEMA,
    PULLED_CHECKPOINTS,
    REGIMES,
    RESOURCE_ENVELOPES,
    TARGET_HORIZON,
    canonical_sha256,
    digested,
    execution_id,
    expected_cells,
    file_sha256,
)

from pipelines.static_adapt.ra_adapt import (
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
    semantic_closure_source_implementation_inventory,
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite generated artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite generated artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _source_paths() -> list[Path]:
    rows: list[Path] = []
    for root_name in ("docs", "pipelines", "src"):
        root = REPO_ROOT / root_name
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            ):
                rows.append(path)
    return rows


def _build_source_archive(path: Path) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite source archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    members: list[dict[str, Any]] = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for source in _source_paths():
            relative = source.relative_to(REPO_ROOT).as_posix()
            data = source.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o755 if data.startswith(b"#!") else 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(data))
            members.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6) as zipped:
            zipped.write(buffer.getvalue())
    return {
        "member_count": len(members),
        "members_sha256": canonical_sha256({"members": members}),
        "archive_sha256": file_sha256(path),
        "archive_size_bytes": path.stat().st_size,
    }


def _bundle_from_members(eid: str, members_data: list[tuple[str, bytes]], origin: dict[str, Any]) -> dict[str, Any]:
    """Deterministically pack (canonical_name, bytes) members as resume_inputs/<eid>.tar.gz."""
    bundle_path = PACKAGE_DIR / f"resume_inputs/{eid}.tar.gz"
    if bundle_path.exists() or bundle_path.is_symlink():
        raise RuntimeError(f"refusing to overwrite resume bundle: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_sha: str | None = None
    members: list[dict[str, Any]] = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as bundle:
        for name, data in sorted(members_data):
            digest = hashlib.sha256(data).hexdigest()
            if name == "current.json":
                checkpoint_sha = digest
            info = tarfile.TarInfo(f"resume/{name}")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            bundle.addfile(info, io.BytesIO(data))
            members.append(
                {"path": f"resume/{name}", "sha256": digest, "size_bytes": len(data)}
            )
    if checkpoint_sha is None or len(members) < 3:
        raise RuntimeError(f"incomplete checkpoint trio for {eid}")
    with bundle_path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6) as zipped:
            zipped.write(buffer.getvalue())
    return {
        "bundle_path": f"resume_inputs/{eid}.tar.gz",
        "bundle_sha256": file_sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "origin": origin,
        "checkpoint_sha256": checkpoint_sha,
        "members": sorted(members, key=lambda row: row["path"]),
    }


def _sidecar_canonical_name(kind: str, data: bytes) -> str:
    return f"current.{kind}.{hashlib.sha256(data).hexdigest()[:16]}.json"


def _live_bundle(eid: str, cell: str) -> dict[str, Any]:
    """Rebuild the canonical trio from the pulled .gz files for one cell."""
    root = Path(PULLED_CHECKPOINTS)
    trio: list[tuple[str, bytes]] = []
    origin_files = {}
    mapping = (
        ("current.json", "current.json.gz", None),
        ("estimator_call_ledger_checkpoint", "current.estimator_call_ledger_checkpoint.gz", "sidecar"),
        ("verified_singleton_resume", "current.verified_singleton_resume.gz", "sidecar"),
    )
    for kind, suffix, role in mapping:
        source = root / f"{cell}.{suffix}"
        if not source.is_file():
            raise RuntimeError(f"pulled member missing for {cell}: {source.name}")
        data = gzip.decompress(source.read_bytes())
        name = kind if role is None else _sidecar_canonical_name(kind, data)
        trio.append((name, data))
    # Pair-coherence gate (HANDOFF_REFACTOR_cost_path item 6): the state
    # file must reference each sidecar's content hash, or the bundle is a
    # cross-epoch pair that would fail resume on the execute node.
    state_bytes = trio[0][1]
    for name, data in trio[1:]:
        digest = hashlib.sha256(data).hexdigest().encode("ascii")
        if digest not in state_bytes:
            raise RuntimeError(
                f"cross-epoch checkpoint pair for {cell}: current.json does "
                f"not reference sidecar {name}"
            )
        origin_files[kind] = {
            "pulled_file": source.name,
            "pulled_at": (root / f"{cell}.pulled_at.txt").read_text().strip()
            if (root / f"{cell}.pulled_at.txt").is_file()
            else None,
        }
    return _bundle_from_members(
        eid, trio, {"mode": "resume_live", "pulled": origin_files}
    )


def _completion_bundle(eid: str, cell: str) -> dict[str, Any]:
    """Extract the round-75 trio from the v1 crash transfer tarball."""
    root = Path(CRASH_TARBALLS)
    matches = sorted(root.glob(f"allphase_k75__{cell}__nph7__*.tar.gz"))
    if not matches:
        raise RuntimeError(f"crash tarball missing for {cell} under {root}")
    archive_path = matches[-1]
    wanted = (
        "current.json",
        "current.estimator_call_ledger_checkpoint.",
        "current.verified_singleton_resume.",
    )
    trio: list[tuple[str, bytes]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if "run/checkpoints/" not in member.name.replace("./", ""):
                continue
            if not any(name.startswith(prefix) for prefix in wanted):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"unreadable member {member.name}")
            trio.append((name, stream.read()))
    return _bundle_from_members(
        eid,
        trio,
        {
            "mode": "completion",
            "crash_tarball": archive_path.name,
            "crash_tarball_sha256": file_sha256(archive_path),
        },
    )


def _assert_protocol(protocol: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    native = protocol["route_contract"]["native_semantic_contract"]
    phase0 = native["phase0_policy"]
    expected_population = (
        "same_ordered_append_endpoint_generator_population_v1"
        if arm["phase0_population"] == "append_endpoint_generators"
        else "current_commutation_reduced_candidate_position_records_v1"
    )
    if (
        native["horizon"] != TARGET_HORIZON
        or native["phase0_estimator_components"] != ["N_grad"]
        or phase0["population"] != expected_population
        or native["optimizer"] != "powell"
        or native["optimizer_maxiter"] != 200
        or native["seeds"] != {"adapt": 7, "transpiler": 7}
    ):
        raise RuntimeError(f"protocol contract drifted for {arm['arm_id']}")


def main() -> int:
    import pipelines.static_adapt.ra_adapt as public_ra

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_inventory = semantic_closure_source_implementation_inventory()
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive = _build_source_archive(archive_path)
    source_manifest = digested(
        {
            "schema": "paper_i_ra_allphase_adaptive_maximum_k75_source_archive_v3",
            "source_inventory_sha256": source_inventory["sha256"],
            **archive,
        }
    )
    write_json(PACKAGE_DIR / "source/source_archive_manifest.json", source_manifest)

    resume_rows: list[str] = []
    completion_rows: list[str] = []
    fresh_rows: list[str] = []
    jobs: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    for arm, regime_id, nph, mode, _src in expected_cells():
        builder = getattr(public_ra, str(arm["builder"]), None)
        route_variant = getattr(public_ra, str(arm["route_constant"]), None)
        if not callable(builder) or not isinstance(route_variant, str):
            raise RuntimeError(f"required frozen route is unavailable: {arm['arm_id']}")
        eid = execution_id(str(arm["arm_id"]), regime_id, nph)
        cell = f"{arm['arm_id']}__{regime_id}"
        problem = build_paper_i_ra_hh_regime_problem(regime_id)
        request = builder(
            insertion_policy=str(arm["insertion_policy"]),
            maximum_controller_rounds=TARGET_HORIZON,
        )
        protocol_obj = materialize_paper_i_ra_semantic_protocol(problem, request)
        protocol = protocol_obj.to_dict()
        if protocol["route_contract"]["native_semantic_contract"]["route_variant"] != route_variant:
            raise RuntimeError(f"route identity drifted: {eid}")
        _assert_protocol(protocol, arm)
        protocol_rel = f"protocols/{eid}.json"
        protocol_path = PACKAGE_DIR / protocol_rel
        write_json(protocol_path, protocol)
        bundle = None
        if mode == "resume_live":
            bundle = _live_bundle(eid, cell)
        elif mode == "completion":
            bundle = _completion_bundle(eid, cell)
        envelope = RESOURCE_ENVELOPES[nph]
        payload = {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": eid,
                "arm_id": arm["arm_id"],
                "regime_id": regime_id,
                "nph": nph,
                "mode": mode,
                "builder": arm["builder"],
                "route_constant": arm["route_constant"],
                "route_variant": route_variant,
                "insertion_policy": arm["insertion_policy"],
                "phase0_population": arm["phase0_population"],
                "phase123_population": arm["phase123_population"],
                "maximum_controller_rounds": TARGET_HORIZON,
                "protocol_path": protocol_rel,
                "protocol_sha256": protocol_obj.sha256,
                "protocol_file_sha256": file_sha256(protocol_path),
                "route_contract_sha256": protocol["route_contract"]["sha256"],
                "source_inventory_sha256": source_inventory["sha256"],
                "resource_request": envelope,
                "resume_input": bundle,
                "execution_authorized": True,
                "submission_authorized": True,
                "paper_adoption_authorized": False,
        }
        if bundle is not None:
            payload["resume_input"] = bundle
        job = digested(payload)
        job_rel = f"jobs/{eid}.json"
        job_path = PACKAGE_DIR / job_rel
        write_json(job_path, job)
        jobs.append(
            {
                "execution_id": eid,
                "mode": mode,
                "path": job_rel,
                "sha256": job["sha256"],
                "file_sha256": file_sha256(job_path),
            }
        )
        protocols.append(
            {
                "execution_id": eid,
                "path": protocol_rel,
                "sha256": protocol_obj.sha256,
                "file_sha256": file_sha256(protocol_path),
            }
        )
        columns = (
            eid,
            job_rel,
            protocol_rel,
            str(envelope["cpus"]),
            str(envelope["memory_mb"]),
            str(envelope["disk_mb"]),
            str(envelope["runtime_seconds"]),
        )
        target = {
            "completion": completion_rows,
            "resume_live": resume_rows,
            "fresh": fresh_rows,
        }[mode]
        target.append("\t".join(columns))
    if len(jobs) != 6 or len(resume_rows) != 3 or completion_rows or len(fresh_rows) != 3:
        raise RuntimeError(
            "recovery matrix drifted from 3 resume + 3 fresh (completion deferred)"
        )
    write_text(PACKAGE_DIR / "queue_resume.tsv", "\n".join(resume_rows) + "\n")
    write_text(PACKAGE_DIR / "queue_fresh.tsv", "\n".join(fresh_rows) + "\n")

    plan = digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "matrix_shape": {"arms": 3, "regimes": 3, "active_cells": 6, "deferred_completion_cells": 3},
            "recovery_of": "paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v1 (cluster 9663770)",
            "defect": (
                "sealed v1 source: pipelines/reporting/paper_i_run_summary.py "
                "enumerated canonical RA identities for horizons 1..50 only; "
                "every v1 cell crashes in summarize_paper_i_run after round 75"
            ),
            "regimes": [{"regime_id": row[0], "nph": row[1]} for row in REGIMES],
            "maximum_controller_rounds": TARGET_HORIZON,
            "allowed_completion_kinds": [
                "reached_maximum_controller_rounds_v1",
                "authenticated_phase3_no_positive_natural_terminal_v1",
            ],
            "source_inventory_sha256": source_inventory["sha256"],
            "source_archive_sha256": archive["archive_sha256"],
            "jobs": jobs,
            "protocols": protocols,
            "canonical_execution_order": [row["execution_id"] for row in jobs],
            "resource_envelopes": RESOURCE_ENVELOPES,
            "execution_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
        }
    )
    write_json(PACKAGE_DIR / "execution_plan.json", plan)
    authorization = digested(
        {
            "schema": AUTH_SCHEMA,
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "plan_sha256": plan["sha256"],
            "source_inventory_sha256": source_inventory["sha256"],
            "source_archive_sha256": archive["archive_sha256"],
            "execution_ids": [row["execution_id"] for row in jobs],
            "execution_authorized": True,
            "submission_authorized": True,
            "paper_adoption_authorized": False,
            "authority_basis": (
                "explicit_user_authorization_in_chat_20260819_proceed_with_fix_"
                "so_75_results_are_not_wasted"
            ),
        }
    )
    write_json(PACKAGE_DIR / "execution_authorization.json", authorization)

    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "authorized_six_cells_3resume_3fresh_completion_deferred",
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": 9,
            "execution_ids": [row["execution_id"] for row in jobs],
            "source_inventory_sha256": source_inventory["sha256"],
            "source_archive": source_manifest,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "worker_file_sha256": file_sha256(PACKAGE_DIR / "worker.py"),
            "execute_file_sha256": file_sha256(PACKAGE_DIR / "execute_job.sh"),
            "queue_resume_sha256": file_sha256(PACKAGE_DIR / "queue_resume.tsv"),
            "queue_fresh_sha256": file_sha256(PACKAGE_DIR / "queue_fresh.tsv"),
            "image_path": IMAGE_PATH,
            "image_sha256": IMAGE_SHA256,
            "execution_authorized": True,
            "submission_authorized": True,
            "submitted": False,
            "paper_adoption_authorized": False,
        }
    )
    write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    print(json.dumps({"package": PACKAGE_ID, "cells": len(jobs), "sha256": manifest["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
