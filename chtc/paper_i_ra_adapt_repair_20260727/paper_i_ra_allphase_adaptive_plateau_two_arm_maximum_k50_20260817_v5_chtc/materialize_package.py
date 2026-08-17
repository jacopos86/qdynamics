#!/usr/bin/env python3
"""Materialize the authorized source-locked eighteen-cell CHTC package."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
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
    IMAGE_PATH,
    IMAGE_SHA256,
    JOB_SCHEMA,
    MANIFEST_SCHEMA,
    PACKAGE_ID,
    PLAN_SCHEMA,
    REGIMES,
    RESOURCE_ENVELOPES,
    TARGET_HORIZON,
    canonical_sha256,
    digested,
    execution_id,
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
                    "sha256": __import__("hashlib").sha256(data).hexdigest(),
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
        or phase0["fubini_study_metric"] != "off"
        or phase0["qiskit_compile"] != "off"
        or phase0["graph_proxy_cost"] != "off"
        or native["qiskit_active_phases"] != ["phase_i", "phase_ii", "phase_iii"]
        or native["phase_shortlist_maxima"]
        != {"phase_i": 24, "phase_ii": 12, "phase_iii": 12}
        or native["phase_frontier_ratios"]
        != {"phase_i": 0.9, "phase_ii": 0.9, "phase_iii": 0.9}
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
            "schema": "paper_i_ra_allphase_adaptive_plateau_two_arm_source_archive_v1",
            "source_inventory_sha256": source_inventory["sha256"],
            **archive,
        }
    )
    write_json(PACKAGE_DIR / "source/source_archive_manifest.json", source_manifest)

    queue_rows: list[str] = []
    jobs: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    for arm in ARMS:
        builder = getattr(public_ra, str(arm["builder"]), None)
        route_variant = getattr(public_ra, str(arm["route_constant"]), None)
        if not callable(builder) or not isinstance(route_variant, str):
            raise RuntimeError(f"required frozen route is unavailable: {arm['arm_id']}")
        for regime_id, nph in REGIMES:
            eid = execution_id(str(arm["arm_id"]), regime_id, nph)
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
            envelope = RESOURCE_ENVELOPES[nph]
            job = digested(
                {
                    "schema": JOB_SCHEMA,
                    "package_id": PACKAGE_ID,
                    "campaign_id": CAMPAIGN_ID,
                    "execution_id": eid,
                    "arm_id": arm["arm_id"],
                    "regime_id": regime_id,
                    "nph": nph,
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
                    "fresh_start": True,
                    "execution_authorized": True,
                    "submission_authorized": True,
                    "paper_adoption_authorized": False,
                }
            )
            job_rel = f"jobs/{eid}.json"
            job_path = PACKAGE_DIR / job_rel
            write_json(job_path, job)
            jobs.append(
                {
                    "execution_id": eid,
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
            queue_rows.append(
                "\t".join(
                    (
                        eid,
                        job_rel,
                        protocol_rel,
                        str(envelope["cpus"]),
                        str(envelope["memory_mb"]),
                        str(envelope["disk_mb"]),
                        str(envelope["runtime_seconds"]),
                    )
                )
            )
    if len(jobs) != 12 or len({row["execution_id"] for row in jobs}) != 12:
        raise RuntimeError("matrix is not exactly twelve unique cells")
    write_text(PACKAGE_DIR / "queue.tsv", "\n".join(queue_rows) + "\n")

    plan = digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "matrix_shape": {"arms": 2, "regimes": 6, "cells": 12},
            "arms": list(ARMS),
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
            "authority_basis": "explicit_user_authorization_in_active_side_conversation_20260817",
        }
    )
    write_json(PACKAGE_DIR / "execution_authorization.json", authorization)

    package_rel = PACKAGE_DIR.relative_to(REPO_ROOT).as_posix()
    template = (PACKAGE_DIR / "submit.sub.in").read_text(encoding="utf-8")
    submit = template.replace("__PACKAGE_REL__", package_rel).replace(
        "__SOURCE_SHA256__", archive["archive_sha256"]
    )
    write_text(PACKAGE_DIR / "submit.sub", submit)

    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "authorized_twelve_cells",
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": 12,
            "execution_ids": [row["execution_id"] for row in jobs],
            "source_inventory_sha256": source_inventory["sha256"],
            "source_archive": source_manifest,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "worker_file_sha256": file_sha256(PACKAGE_DIR / "worker.py"),
            "execute_file_sha256": file_sha256(PACKAGE_DIR / "execute_job.sh"),
            "queue_file_sha256": file_sha256(PACKAGE_DIR / "queue.tsv"),
            "submit_file_sha256": file_sha256(PACKAGE_DIR / "submit.sub"),
            "image_path": IMAGE_PATH,
            "image_sha256": IMAGE_SHA256,
            "execution_authorized": True,
            "submission_authorized": True,
            "submitted": False,
            "paper_adoption_authorized": False,
        }
    )
    write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
