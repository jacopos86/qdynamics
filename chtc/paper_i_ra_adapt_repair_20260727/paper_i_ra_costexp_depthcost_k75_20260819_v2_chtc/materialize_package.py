#!/usr/bin/env python3
"""Materialize the costexp + explicit-depth k75 package: 15 fresh cells."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ARMS,
    AUTH_SCHEMA,
    CAMPAIGN_ID,
    COSTEXP_EQUIVALENCE,
    EXPECTED_COSTEXP_BASELINE_PENALTY_WEIGHTS,
    EXPECTED_DEPTH_PENALTY_WEIGHTS,
    EXPECTED_DEPTH_WEIGHTS,
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
    expected_cells,
    file_sha256,
)

from pipelines.static_adapt.ra_adapt import (  # noqa: E402
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
    semantic_closure_source_implementation_inventory,
)
from pipelines.static_adapt.ra_adapt.costexp_realization import (  # noqa: E402
    costexp_penalty_weights_from_route_contract,
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite generated artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    with tarfile.open(
        fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT
    ) as archive:
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
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6
        ) as zipped:
            zipped.write(buffer.getvalue())
    return {
        "member_count": len(members),
        "members_sha256": canonical_sha256({"members": members}),
        "archive_sha256": file_sha256(path),
        "archive_size_bytes": path.stat().st_size,
    }


def _assert_protocol(protocol: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    contract = protocol["route_contract"]
    native = contract["native_semantic_contract"]
    invariants = contract["semantic_invariants"]
    execution = contract["execution_settings"]
    profile = str(contract["route_profile"])
    if (
        native["horizon"] != TARGET_HORIZON
        or native["optimizer"] != "powell"
        or native["optimizer_maxiter"] != 200
        or native["seeds"] != {"adapt": 7, "transpiler": 7}
        or native["insertion_policy"] != arm["insertion_policy"]
    ):
        raise RuntimeError(f"protocol contract drifted for {arm['arm_id']}")
    if arm["family"] == "costexp":
        if (
            invariants.get("insertion_equivalence_policy")
            != COSTEXP_EQUIVALENCE
            or f"__insertion-{arm['insertion_policy']}" not in profile
            or "compile_cost_policy" in invariants
            or costexp_penalty_weights_from_route_contract(contract)
            != EXPECTED_COSTEXP_BASELINE_PENALTY_WEIGHTS
        ):
            raise RuntimeError(
                f"costexp identity drifted for {arm['arm_id']}"
            )
    else:
        if (
            invariants.get("compile_cost_policy")
            != arm["compile_cost_policy"]
            or invariants.get("compile_cost_weights")
            != EXPECTED_DEPTH_WEIGHTS
            or native.get("compile_cost_policy") != arm["compile_cost_policy"]
            or native.get("compile_cost_weights") != EXPECTED_DEPTH_WEIGHTS
            or execution.get("phase3_backend_w_2q") != 0.3
            or execution.get("phase3_backend_w_depth") != 0.25
            or execution.get("phase3_backend_w_size") != 0.0
            or execution.get("phase3_backend_w_depth_2q") != 0.3
            or execution.get("phase3_backend_w_1q") != 0.0
            or f"__cost-{arm['compile_cost_policy']}" not in profile
        ):
            raise RuntimeError(
                f"explicit-depth identity drifted for {arm['arm_id']}"
            )


def main() -> int:
    import pipelines.static_adapt.ra_adapt as public_ra

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_inventory = semantic_closure_source_implementation_inventory()
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive = _build_source_archive(archive_path)
    source_manifest = digested(
        {
            "schema": "ra_adapt_costexp_depthcost_k75_source_archive_v2",
            "source_inventory_sha256": source_inventory["sha256"],
            **archive,
        }
    )
    write_json(
        PACKAGE_DIR / "source/source_archive_manifest.json", source_manifest
    )

    fresh_rows: list[str] = []
    jobs: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    route_shas: set[str] = set()
    for arm, regime_id, nph in expected_cells():
        builder = getattr(public_ra, str(arm["builder"]), None)
        route_variant = getattr(public_ra, str(arm["route_constant"]), None)
        if not callable(builder) or not isinstance(route_variant, str):
            raise RuntimeError(
                f"required frozen route is unavailable: {arm['arm_id']}"
            )
        eid = execution_id(str(arm["arm_id"]), regime_id, nph)
        problem = build_paper_i_ra_hh_regime_problem(regime_id)
        builder_kwargs: dict[str, Any] = {
            "insertion_policy": str(arm["insertion_policy"]),
            "maximum_controller_rounds": TARGET_HORIZON,
        }
        if arm["compile_cost_policy"] is not None:
            builder_kwargs["compile_cost_policy"] = str(
                arm["compile_cost_policy"]
            )
        request = builder(**builder_kwargs)
        protocol_obj = materialize_paper_i_ra_semantic_protocol(
            problem, request
        )
        protocol = protocol_obj.to_dict()
        if (
            protocol["route_contract"]["native_semantic_contract"][
                "route_variant"
            ]
            != route_variant
        ):
            raise RuntimeError(f"route identity drifted: {eid}")
        _assert_protocol(protocol, arm)
        route_shas.add(str(protocol["route_contract"]["sha256"]))
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
                "family": arm["family"],
                "regime_id": regime_id,
                "nph": nph,
                "mode": "fresh",
                "builder": arm["builder"],
                "route_constant": arm["route_constant"],
                "route_variant": route_variant,
                "insertion_policy": arm["insertion_policy"],
                "compile_cost_policy": arm["compile_cost_policy"],
                "expected_signed_penalty_weights": (
                    EXPECTED_COSTEXP_BASELINE_PENALTY_WEIGHTS
                    if arm["family"] == "costexp"
                    else EXPECTED_DEPTH_PENALTY_WEIGHTS
                ),
                "maximum_controller_rounds": TARGET_HORIZON,
                "protocol_path": protocol_rel,
                "protocol_sha256": protocol_obj.sha256,
                "protocol_file_sha256": file_sha256(protocol_path),
                "route_contract_sha256": protocol["route_contract"]["sha256"],
                "source_inventory_sha256": source_inventory["sha256"],
                "resource_request": envelope,
                "execution_authorized": True,
                "submission_authorized": True,
                "manuscript_adoption_authorized": False,
            }
        )
        job_rel = f"jobs/{eid}.json"
        job_path = PACKAGE_DIR / job_rel
        write_json(job_path, job)
        jobs.append(
            {
                "execution_id": eid,
                "mode": "fresh",
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
        fresh_rows.append(
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
    if len(jobs) != 15 or len(route_shas) != 5:
        raise RuntimeError(
            "matrix drifted from 5 arms x 3 regimes with 5 distinct "
            f"route identities (jobs={len(jobs)}, routes={len(route_shas)})"
        )
    write_text(PACKAGE_DIR / "queue_fresh.tsv", "\n".join(fresh_rows) + "\n")

    plan = digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "matrix_shape": {
                "arms": 5,
                "regimes": 3,
                "active_cells": 15,
                "families": {"costexp": 6, "explicit_depth": 9},
            },
            "supersedes": (
                "paper_i_ra_costexp_depthcost_k75_20260819_v1 "
                "(cluster 9664290: costexp cells crashed at the first "
                "off-representative realization under the v1 sealed "
                "source; depth cells reminted with weight_1q=0)"
            ),
            "regimes": [
                {"regime_id": row[0], "nph": row[1]} for row in REGIMES
            ],
            "maximum_controller_rounds": TARGET_HORIZON,
            "allowed_completion_kinds": [
                "reached_maximum_controller_rounds_v1",
                "authenticated_phase3_no_positive_natural_terminal_v1",
            ],
            "source_inventory_sha256": source_inventory["sha256"],
            "source_archive_sha256": archive["archive_sha256"],
            "jobs": jobs,
            "protocols": protocols,
            "canonical_execution_order": [
                row["execution_id"] for row in jobs
            ],
            "resource_envelopes": RESOURCE_ENVELOPES,
            "execution_authorized": False,
            "submission_authorized": False,
            "manuscript_adoption_authorized": False,
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
            "manuscript_adoption_authorized": False,
            "authority_basis": (
                "explicit_user_directive_20260819_costexp_position_arms_plus_"
                "explicit_depth_cost_matrix_all_three_ra_variants"
            ),
        }
    )
    write_json(PACKAGE_DIR / "execution_authorization.json", authorization)

    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "authorized_fifteen_fresh_cells",
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": 15,
            "execution_ids": [row["execution_id"] for row in jobs],
            "source_inventory_sha256": source_inventory["sha256"],
            "source_archive": source_manifest,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "worker_file_sha256": file_sha256(PACKAGE_DIR / "worker.py"),
            "execute_file_sha256": file_sha256(
                PACKAGE_DIR / "execute_job.sh"
            ),
            "queue_fresh_sha256": file_sha256(
                PACKAGE_DIR / "queue_fresh.tsv"
            ),
            "image_path": IMAGE_PATH,
            "image_sha256": IMAGE_SHA256,
            "execution_authorized": True,
            "submission_authorized": True,
            "submitted": False,
            "manuscript_adoption_authorized": False,
        }
    )
    write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    print(
        json.dumps(
            {
                "package": PACKAGE_ID,
                "cells": len(jobs),
                "sha256": manifest["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
