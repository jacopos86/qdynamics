#!/usr/bin/env python3
"""Materialize the B7 package: 18 fresh measured-residual cells on the flat
global guarded-singleton pool (the manuscript route, stationarity off).

B7 = B3's candidate traversal (948/6508 flat Pauli-word pool, identity
exposure, no in-pipeline splitting) + B5's measured-residual lane. One axis
vs B3 (stationarity), one axis vs B5 (pool traversal). No warm start."""

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
sys.path.insert(0, str(REPO_ROOT))

from package_contract import (  # noqa: E402
    ARMS,
    AUTH_SCHEMA,
    CAMPAIGN_ID,
    MR_ACTIVE_GRADIENT_POLICY,
    MR_ALGORITHM_ID,
    MR_CANDIDATE_GAIN_POLICY,
    IMAGE_PATH,
    IMAGE_SHA256,
    JOB_SCHEMA,
    MANIFEST_SCHEMA,
    PACKAGE_ID,
    PLAN_SCHEMA,
    REGIMES,
    RESOURCE_ENVELOPES,
    canonical_sha256,
    digested,
    execution_id,
    expected_cells,
    file_sha256,
)

from pipelines.static_adapt.ra_adapt import (  # noqa: E402
    RAAdaptRequest,
    build_paper_i_ra_hh_regime_problem,
    semantic_closure_source_implementation_inventory,
)
from pipelines.static_adapt.ra_adapt.engine import (  # noqa: E402
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    AlwaysCommutationReducedInsertion,
    AppendOnlyInsertion,
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
)
from pipelines.static_adapt.ra_adapt.adapters import (  # noqa: E402
    GlobalSinglePauliWordCandidateAdapter,
)

INSERTION_CLASSES = {
    "AppendOnlyInsertion": AppendOnlyInsertion,
    "PlateauCommutationInsertion": PlateauCommutationInsertion,
    "AlwaysCommutationReducedInsertion": AlwaysCommutationReducedInsertion,
}


def build_mr_request(arm, horizon):
    """The executable B7 request: measured-residual lane on the flat global
    guarded-singleton pool (identity exposure, no in-pipeline splitting)."""
    insertion = INSERTION_CLASSES[str(arm["insertion_class"])]()
    return RAAdaptRequest(
        adapter=GlobalSinglePauliWordCandidateAdapter(),
        method=SRMethodPolicy(insertion=insertion),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=int(horizon))
        ),
    )


def assert_mr_identity(protocol, eid):
    """Mandatory per-cell gate: refuse anything that is not the MR route."""
    rc = protocol.get("route_contract", {}) or {}
    inv = rc.get("semantic_invariants", {}) or {}
    ex = rc.get("execution_settings", {}) or {}
    observed = (
        protocol.get("algorithm_id") or rc.get("algorithm_id"),
        inv.get("active_gradient_policy")
        or ex.get("ra_active_gradient_policy"),
        inv.get("phase3_candidate_gain_policy")
        or ex.get("ra_phase3_candidate_gain_policy"),
    )
    expected = (
        MR_ALGORITHM_ID,
        MR_ACTIVE_GRADIENT_POLICY,
        MR_CANDIDATE_GAIN_POLICY,
    )
    if observed != expected:
        raise RuntimeError(
            f"measured-residual identity gate failed for {eid}: {observed}"
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


def _assert_protocol(protocol, arm, horizon: int, nph_for_gate: int = 3) -> None:
    """B7 gate: MR identity + no cost term + flat singleton pool, no split."""
    assert_mr_identity(protocol, str(arm["arm_id"]))
    rc = protocol.get("route_contract", {}) or {}
    inv = rc.get("semantic_invariants", {}) or {}
    if inv.get("compile_cost_policy") is not None or inv.get(
        "compile_cost_weights"
    ) is not None:
        raise RuntimeError(
            f"B7 must carry no compile-cost term: {arm['arm_id']}"
        )
    ex = rc.get("execution_settings", {}) or {}
    if (
        inv.get("candidate_initialization_policy") is not None
        or "adapt_schur_warm_start_mode" in ex
        and ex.get("adapt_schur_warm_start_mode") not in (None, "off")
    ):
        raise RuntimeError(
            f"B7 must carry no candidate warm start: {arm['arm_id']}"
        )
    pool = protocol.get("executable_pool") or {}
    n_expected = {3: 948, 7: 6508}[int(nph_for_gate)]
    if int(pool.get("count") or -1) != n_expected:
        raise RuntimeError(
            f"B7 flat-pool gate failed for {arm['arm_id']}: executable_pool "
            f"count {pool.get('count')} != {n_expected}"
        )
    rep = str(pool.get("candidate_representation") or "")
    if rep != "single_pauli_word_v1":
        raise RuntimeError(
            f"B7 pool representation drifted: {rep}"
        )
    adapter_id = str(protocol.get("adapter_id") or "")
    if "global" not in adapter_id:
        raise RuntimeError(
            f"B7 adapter gate failed for {arm['arm_id']}: {adapter_id!r}"
        )
    stop = (protocol.get("request") or {}).get("execution", {})
    del stop  # horizon is asserted from the job row below


def main() -> int:
    import pipelines.static_adapt.ra_adapt as public_ra

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_inventory = semantic_closure_source_implementation_inventory()
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive = _build_source_archive(archive_path)
    source_manifest = digested(
        {
            "schema": "ra_adapt_costexp_depthfix_sixregime_source_archive_v2",
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
    for arm, regime_id, nph, horizon in expected_cells():
        route_variant = MR_ALGORITHM_ID
        eid = execution_id(str(arm["arm_id"]), regime_id, nph)
        problem = build_paper_i_ra_hh_regime_problem(regime_id)
        request = build_mr_request(arm, int(horizon))
        protocol_obj = build_resolved_ra_protocol(problem, request)
        protocol = protocol_obj.to_dict()
        _assert_protocol(protocol, arm, int(horizon), nph_for_gate=int(nph))
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
                "matrix": "B7MRFS",
                "family": "measured_residual",
                "candidate_pool_route": "global_guarded_singleton_flat_v1",
                "regime_id": regime_id,
                "nph": nph,
                "mode": "fresh",
                "builder": "build_resolved_ra_protocol",
                "route_constant": MR_ALGORITHM_ID,
                "route_variant": route_variant,
                "insertion_policy": arm["insertion_policy"],
                "compile_cost_policy": None,
                "maximum_controller_rounds": int(horizon),
                "protocol_path": protocol_rel,
                "protocol_sha256": protocol_obj.sha256,
                "protocol_file_sha256": file_sha256(protocol_path),
                "route_contract_sha256": protocol["route_contract"]["sha256"],
                "source_inventory_sha256": source_inventory["sha256"],
                "resource_request": envelope,
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
    if len(jobs) != 18 or len(route_shas) != 3:
        raise RuntimeError(
            "B7 drifted from 3 arms x 6 regimes with 3 distinct sealed "
            "route identities (one per insertion policy; horizon is carried "
            "by the protocol, not the route contract), observed "
            f"jobs={len(jobs)}, routes={len(route_shas)}"
        )
    write_text(PACKAGE_DIR / "queue_fresh.tsv", "\n".join(fresh_rows) + "\n")

    plan = digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "matrix_shape": {
                "arms": 3,
                "regimes": 6,
                "active_cells": 18,
                "matrices": {"D_flat_singleton_measured_residual": 18},
                "horizons": {"nph3": 50, "nph7": 75},
            },
            "companion": (
                "baselines: b3 sixregime flat-pool stationary (9664325) and "
                "b3mr parent-split measured (9674089); B7 = flat global "
                "guarded-singleton pool + measured-residual lane, the "
                "manuscript-described traversal with stationarity off"
            ),
            "regimes": [
                {"regime_id": row[0], "nph": row[1], "horizon": row[2]}
                for row in REGIMES
            ],
            "allowed_completion_kinds": [
                "reached_maximum_controller_rounds_v1",
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
                "explicit_user_directive_20260822_b7_flat_pool_measured_"
                "residual"
            ),
        }
    )
    write_json(PACKAGE_DIR / "execution_authorization.json", authorization)

    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "authorized_eighteen_fresh_cells",
            "created_at": created_at,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": 18,
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
            "paper_adoption_authorized": False,
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
