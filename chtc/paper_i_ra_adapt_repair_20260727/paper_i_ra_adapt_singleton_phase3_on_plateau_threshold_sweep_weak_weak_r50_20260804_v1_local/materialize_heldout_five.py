#!/usr/bin/env python3
"""Expand the winning weak--weak threshold to the five held-out regimes.

This command is intentionally unusable until a CHTC source-value anchor and
both CHTC threshold variants have completed.  It selects the smaller terminal
R50 same-cutoff error, requires a strict win over Append, and reuses that
winner's sealed source archive without another source edit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize_threshold_sweep import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ALGORITHM_ID,
    APPEND_TARGET_ABSOLUTE_ENERGY_ERROR,
    CANDIDATE_REPRESENTATION,
    EXPECTED_DERIVATIVE_HASHES,
    PACKAGE_SCHEMA_PREFIX,
    PLATEAU_CALIBRATION,
    PLATEAU_COMPARISON,
    PLATEAU_TRIGGER,
    REMOTE_IMAGE,
    REPO_ROOT,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_PROFILE,
    SOURCE_PACKAGE,
    SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    SOURCE_PACKAGE_RELATIVE,
    SOURCE_PROTOCOL_BUNDLE_CANONICAL_SHA256,
    SOURCE_THRESHOLD,
    TARGET_HORIZON,
    SweepError,
    _activate_source_root,
    _exact_energy,
    _extract_source,
    _load_bound_source,
    _problem_from_receipt,
    _result_projection,
    _scientific_audit,
    binding,
    canonical_json_bytes,
    digested,
    load_json,
    package_dir,
    sha256_file,
    verify_self_digest,
    write_json,
)


HELDOUT_ROWS: tuple[tuple[str, int], ...] = (
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
RESOURCE_ENVELOPES = {
    3: {
        "request_cpus": 4,
        "request_memory_mb": 24_576,
        "request_disk_mb": 40_960,
        "max_runtime_seconds": 259_200,
    },
    7: {
        "request_cpus": 4,
        "request_memory_mb": 32_768,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
    },
}


def _threshold_token(value: float) -> str:
    if value == 1.0e-5:
        return "1em5"
    if value == 1.0e-6:
        return "1em6"
    raise SweepError("Held-out expansion accepts only a calibrated variant.")


def _result_receipt(path: Path, *, threshold: float) -> dict[str, Any]:
    payload = load_json(path, label=f"tau={threshold:g} result")
    projection = _result_projection(payload)
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise SweepError("Threshold result lost its protocol receipt.")
    route = protocol.get("route_contract")
    if (
        not isinstance(route, Mapping)
        or route.get("sha256")
        != EXPECTED_DERIVATIVE_HASHES[threshold]["route_contract_sha256"]
        or projection["controller_rounds"] != TARGET_HORIZON
    ):
        raise SweepError(f"Tau={threshold:g} result identity drifted.")
    return {
        "threshold": threshold,
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "file_sha256": sha256_file(path),
        "route_contract_sha256": route["sha256"],
        **projection,
    }


def _select_winner(result_1em5: Path, result_1em6: Path) -> dict[str, Any]:
    rows = [
        _result_receipt(result_1em5, threshold=1.0e-5),
        _result_receipt(result_1em6, threshold=1.0e-6),
    ]
    rows.sort(key=lambda row: (row["terminal_absolute_energy_error"], row["threshold"]))
    if (
        rows[0]["terminal_absolute_energy_error"]
        >= APPEND_TARGET_ABSOLUTE_ENERGY_ERROR
    ):
        raise SweepError(
            "Neither predeclared threshold strictly beats the weak--weak Append target."
        )
    return {
        "selection_rule": (
            "minimum_terminal_r50_same_cutoff_absolute_energy_error_"
            "requiring_strictly_below_append"
        ),
        "append_target_absolute_energy_error": APPEND_TARGET_ABSOLUTE_ENERGY_ERROR,
        "rows": rows,
        "winning_threshold": rows[0]["threshold"],
        "winning_result_file_sha256": rows[0]["file_sha256"],
    }


def _validate_anchor(path: Path) -> dict[str, Any]:
    payload = load_json(path, label="CHTC anchor comparison")
    verify_self_digest(payload, label="CHTC anchor comparison")
    if (
        payload.get("status") != "passed"
        or payload.get("value") != SOURCE_THRESHOLD
        or payload.get("anchor_reproduces_source") is not True
        or payload.get("operator_sequence_match") is not True
        or payload.get("insertion_position_sequence_match") is not True
        or float(payload.get("metric_abs_diff", float("inf"))) > 1.0e-12
    ):
        raise SweepError("Held-out expansion requires a passing CHTC anchor.")
    return payload


def _winner_package(threshold: float) -> dict[str, Any]:
    root = package_dir(threshold)
    manifest = load_json(root / "package_manifest.json", label="winner manifest")
    verify_self_digest(manifest, label="winner manifest")
    audit = load_json(root / "source_lock_audit.json", label="winner source audit")
    verify_self_digest(audit, label="winner source audit")
    expected = EXPECTED_DERIVATIVE_HASHES[threshold]
    if (
        manifest.get("status") != "passed_inert_one_row"
        or manifest.get("threshold") != threshold
        or manifest.get("execution_target") != "chtc"
        or audit.get("status") != "passed"
        or audit.get("non_swept_settings_diff") != []
        or audit.get("target_route_contract_sha256")
        != expected["route_contract_sha256"]
        or audit.get("target_implementation_inventory_sha256")
        != expected["implementation_inventory_sha256"]
    ):
        raise SweepError("Winning sealed package authority drifted.")
    archive = root / str(manifest["source_archive"]["path"])
    locks = root / str(manifest["source_locks_snapshot"]["path"])
    archive_manifest = root / str(manifest["source_archive_manifest"]["path"])
    if (
        sha256_file(archive) != manifest["source_archive"]["sha256"]
        or sha256_file(locks) != manifest["source_locks_snapshot"]["sha256"]
        or sha256_file(archive_manifest)
        != manifest["source_archive_manifest"]["sha256"]
    ):
        raise SweepError("Winning source bytes drifted.")
    return {
        "root": root,
        "manifest": manifest,
        "audit": audit,
        "archive": archive,
        "locks": locks,
        "archive_manifest": archive_manifest,
    }


def _queue_line(job: Mapping[str, Any]) -> str:
    resources = job["resources"]
    return "\t".join(
        (
            str(job["execution_id"]),
            str(job["job_path"]),
            str(job["protocol_path"]),
            str(job["sha256"]),
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        )
    )


def materialize(
    *,
    anchor_comparison_path: Path,
    result_1em5_path: Path,
    result_1em6_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite held-out package: {output_dir}")
    try:
        output_dir.resolve().relative_to(
            (REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727").resolve()
        )
    except ValueError as exc:
        raise SweepError("Held-out output must remain in the Paper-I CHTC workspace.") from exc
    anchor = _validate_anchor(anchor_comparison_path)
    selection = _select_winner(result_1em5_path, result_1em6_path)
    threshold = float(selection["winning_threshold"])
    token = _threshold_token(threshold)
    winner = _winner_package(threshold)
    source = _load_bound_source()
    temporary = tempfile.TemporaryDirectory(
        prefix=f"paper-i-threshold-{token}-heldout5-"
    )
    try:
        source_root = (Path(temporary.name) / "source_locked_checkout").resolve()
        winner_archive_manifest = load_json(
            winner["archive_manifest"], label="winner archive manifest"
        )
        verify_self_digest(winner_archive_manifest, label="winner archive manifest")
        _extract_source(
            {
                "archive": winner["archive"],
                "source archive manifest": winner_archive_manifest,
            },
            source_root,
        )
        _activate_source_root(source_root)

        from pipelines.static_adapt.ra_adapt.bundles import (
            _build_request,
            _bundle_protocol_materialization_authority,
            _cell_from_manifest_row,
            _decorate_protocol_payload,
            _implementation_source_inventory,
            _source_lock_refs,
            _validate_protocol_payload,
        )
        from pipelines.static_adapt.ra_adapt.contracts import (
            resolved_ra_adapt_protocol_from_mapping,
        )
        from pipelines.static_adapt.ra_adapt.engine import (
            build_resolved_ra_protocol,
        )

        implementation_inventory = _implementation_source_inventory(source_root)
        expected = EXPECTED_DERIVATIVE_HASHES[threshold]
        source_locks = load_json(winner["locks"], label="winner source locks")
        verify_self_digest(source_locks, label="winner source locks")
        if (
            implementation_inventory.get("sha256")
            != expected["implementation_inventory_sha256"]
            or source_locks.get("sha256")
            != expected["source_locks_canonical_sha256"]
            or sha256_file(winner["locks"])
            != expected["source_locks_file_sha256"]
        ):
            raise SweepError("Winning extracted source-lock closure drifted.")

        package_id = output_dir.name
        campaign_id = (
            "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_"
            f"{token}_heldout5_r50_v1"
        )
        source_rows = {
            str(row["cell_id"]): row
            for row in source["protocol bundle"]["cells"]
        }
        cells = []
        source_cells = {}
        for regime_id, nph in HELDOUT_ROWS:
            execution_id = (
                f"phase3_on_plateau_r50__{regime_id}__nph{nph}__"
                "ra_singleton_plateau"
            )
            if execution_id not in source_rows:
                raise SweepError(f"Held-out source cell is absent: {execution_id}")
            cell = _cell_from_manifest_row(source_rows[execution_id])
            cells.append(cell)
            source_cells[execution_id] = cell

        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "source").mkdir(exist_ok=False)
        for source_name, target_name in (
            (ROOT / "package_contract_template.py", output_dir / "package_contract.py"),
            (ROOT / "run_cell_template.py", output_dir / "run_cell.py"),
            (SOURCE_PACKAGE / "execute_authorized_job.sh", output_dir / "execute_authorized_job.sh"),
            (SOURCE_PACKAGE / "submit.sub.in", output_dir / "submit.sub.in"),
        ):
            shutil.copyfile(source_name, target_name)
        os.chmod(output_dir / "execute_authorized_job.sh", 0o755)
        shutil.copyfile(winner["archive"], output_dir / "source/source_locked.tar.gz")
        shutil.copyfile(
            winner["archive_manifest"],
            output_dir / "source/source_archive_manifest.json",
        )
        shutil.copyfile(winner["locks"], output_dir / "source_locks_snapshot.json")

        bundle_manifest = digested(
            {
                "schema": f"{PACKAGE_SCHEMA_PREFIX}_heldout5_protocol_bundle_v1",
                "package_id": package_id,
                "campaign_id": campaign_id,
                "source_winner": {
                    "package_id": winner["manifest"]["package_id"],
                    "package_manifest_sha256": winner["manifest"]["sha256"],
                    "source_archive_sha256": sha256_file(winner["archive"]),
                    "source_locks_sha256": source_locks["sha256"],
                    "winning_threshold": threshold,
                    "winning_result_file_sha256": selection[
                        "winning_result_file_sha256"
                    ],
                },
                "anchor_comparison_sha256": anchor["sha256"],
                "cells": [cell.to_dict() for cell in cells],
                "execution_authorized": False,
                "submission_authorized": False,
            }
        )
        write_json(output_dir / "protocol_bundle_manifest.json", bundle_manifest)

        protocols = []
        jobs = []
        audit_rows = []
        route_sha = None
        parent_route_sha = None
        source_protocol_bindings = {
            str(row["execution_id"]): row
            for row in source["manifest"]["protocols"]
        }
        for cell in cells:
            execution_id = cell.cell_id
            source_protocol_path = SOURCE_PACKAGE / "protocols" / f"{execution_id}.json"
            source_payload = load_json(
                source_protocol_path, label=f"source protocol {execution_id}"
            )
            source_protocol = resolved_ra_adapt_protocol_from_mapping(source_payload)
            source_binding = source_protocol_bindings.get(execution_id)
            if (
                not isinstance(source_binding, Mapping)
                or sha256_file(source_protocol_path) != source_binding.get("sha256")
                or source_protocol.sha256
                != source_binding.get("canonical_sha256")
            ):
                raise SweepError(f"Source protocol binding drifted: {execution_id}")
            problem = _problem_from_receipt(source_protocol.problem)
            refs = _source_lock_refs(source_locks, cell=cell)
            authority = _bundle_protocol_materialization_authority(
                cell=cell,
                bundle_id=package_id,
                bundle_manifest_sha256=bundle_manifest["sha256"],
                source_locks_sha256=source_locks["sha256"],
                source_lock_refs=refs,
                active_gradient_policy=ACTIVE_GRADIENT_POLICY,
                resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
            )
            request = _build_request(cell, bundle_dir=output_dir)
            resolved = build_resolved_ra_protocol(
                problem, request, materialization_authority=authority
            )
            target_payload = _decorate_protocol_payload(
                resolved.to_dict(),
                cell=cell,
                request=request,
                cell_source_lock=source_locks["cell_locks"][cell.source_lock_id],
                materialization_authority=authority,
            )
            _validate_protocol_payload(
                target_payload,
                cell=cell,
                bundle_id=package_id,
                bundle_manifest_sha256=bundle_manifest["sha256"],
                active_gradient_policy=ACTIVE_GRADIENT_POLICY,
                resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
                source_lock_refs=refs,
                cell_source_lock=source_locks["cell_locks"][cell.source_lock_id],
                source_locks_sha256=source_locks["sha256"],
            )
            science_audit = _scientific_audit(
                source_payload, target_payload, threshold=threshold
            )
            route = target_payload["route_contract"]
            this_parent = route["lineage_authority"]["parent_contract_sha256"]
            if (
                route["sha256"] != expected["route_contract_sha256"]
                or this_parent != expected["parent_route_contract_sha256"]
            ):
                raise SweepError("Held-out route hash differs from weak--weak winner.")
            route_sha = route["sha256"]
            parent_route_sha = this_parent
            protocol_path = output_dir / "protocols" / f"{execution_id}.json"
            write_json(protocol_path, target_payload)
            protocol_binding = binding(
                protocol_path, root=output_dir, canonical=True
            )
            protocols.append(
                {
                    "execution_id": execution_id,
                    "source_cell_id": execution_id,
                    **protocol_binding,
                }
            )
            target_protocol = resolved_ra_adapt_protocol_from_mapping(target_payload)
            cell_lock = source_locks["cell_locks"][cell.source_lock_id]
            job = digested(
                {
                    "schema": f"{PACKAGE_SCHEMA_PREFIX}_heldout5_job_v1",
                    "package_id": package_id,
                    "campaign_id": campaign_id,
                    "execution_id": execution_id,
                    "source_cell_id": execution_id,
                    "source_lock_id": cell.source_lock_id,
                    "source_lock_sha256": cell_lock["sha256"],
                    "regime_id": cell.regime_id,
                    "nph": int(cell.nph),
                    "run_class": "candidate",
                    "execution_target": "chtc",
                    "execution_mode": "fresh_0_to_50",
                    "source_horizon": TARGET_HORIZON,
                    "target_horizon": TARGET_HORIZON,
                    "protocol_path": protocol_binding["path"],
                    "protocol_sha256": target_protocol.sha256,
                    "protocol_file_sha256": protocol_binding["sha256"],
                    "protocol_bundle_manifest_sha256": bundle_manifest["sha256"],
                    "source_locks_snapshot_sha256": source_locks["sha256"],
                    "implementation_source_inventory_sha256": implementation_inventory[
                        "sha256"
                    ],
                    "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                    "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                    "candidate_representation": CANDIDATE_REPRESENTATION,
                    "insertion_policy": "plateau_commutation",
                    "plateau_prior_mean_decrease_ratio_threshold": threshold,
                    "plateau_threshold_comparison": PLATEAU_COMPARISON,
                    "plateau_trigger_source": PLATEAU_TRIGGER,
                    "route_contract_sha256": route_sha,
                    "exact_same_cutoff_energy": _exact_energy(
                        source_locks, cell.source_lock_id
                    ),
                    "resources": copy.deepcopy(RESOURCE_ENVELOPES[int(cell.nph)]),
                    "fresh_start_contract": {
                        "kind": "fresh_start",
                        "source_checkpoint": None,
                        "resume_archive": None,
                    },
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submitted": False,
                }
            )
            job["job_path"] = f"jobs/{execution_id}.json"
            job = digested(job)
            write_json(output_dir / job["job_path"], job)
            jobs.append(job)
            audit_rows.append(
                {
                    "execution_id": execution_id,
                    "source_protocol": {
                        "path": source_protocol_path.relative_to(REPO_ROOT).as_posix(),
                        "file_sha256": sha256_file(source_protocol_path),
                        "canonical_sha256": source_protocol.sha256,
                    },
                    "target_protocol": protocol_binding,
                    **science_audit,
                    "status": "passed",
                }
            )

        assert route_sha is not None and parent_route_sha is not None
        control = digested(
            {
                "schema": f"{PACKAGE_SCHEMA_PREFIX}_heldout5_package_control_v1",
                "package_id": package_id,
                "package_status": "passed_inert_five_rows",
                "campaign_id": campaign_id,
                "algorithm_id": ALGORITHM_ID,
                "route_contract_sha256": route_sha,
                "parent_route_contract_sha256": parent_route_sha,
                "route_profile": ROUTE_PROFILE,
                "threshold": threshold,
                "plateau_comparison": PLATEAU_COMPARISON,
                "plateau_trigger": PLATEAU_TRIGGER,
                "plateau_calibration": PLATEAU_CALIBRATION,
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "target_horizon": TARGET_HORIZON,
                "execution_target": "chtc",
                "execution_ids": [cell.cell_id for cell in cells],
                "package_manifest_schema": f"{PACKAGE_SCHEMA_PREFIX}_heldout5_package_manifest_v1",
                "job_schema": f"{PACKAGE_SCHEMA_PREFIX}_heldout5_job_v1",
                "authorization_schema": f"{PACKAGE_SCHEMA_PREFIX}_heldout5_execution_authorization_v1",
            }
        )
        write_json(output_dir / "package_control.json", control)

        audit = digested(
            {
                "schema": "source_locked_sensitivity_heldout5_audit_v1",
                "status": "passed",
                "anchor_comparison_sha256": anchor["sha256"],
                "selection": selection,
                "winning_package_manifest_sha256": winner["manifest"]["sha256"],
                "source_archive_sha256": sha256_file(winner["archive"]),
                "source_locks_sha256": source_locks["sha256"],
                "implementation_source_inventory_sha256": implementation_inventory[
                    "sha256"
                ],
                "threshold": threshold,
                "non_threshold_settings_diff": [],
                "rows": audit_rows,
                "heldout_regime_count": len(audit_rows),
                "execution_authorized": False,
                "submission_authorized": False,
            }
        )
        write_json(output_dir / "source_lock_audit.json", audit)
        queue_path = output_dir / "queue.tsv"
        with queue_path.open("xb") as stream:
            stream.write(
                ("\n".join(_queue_line(job) for job in jobs) + "\n").encode(
                    "utf-8"
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        plan = digested(
            {
                "schema": f"{PACKAGE_SCHEMA_PREFIX}_heldout5_execution_plan_v1",
                "package_id": package_id,
                "campaign_id": campaign_id,
                "run_class": "candidate",
                "execution_target": "chtc",
                "execution_ids": [job["execution_id"] for job in jobs],
                "row_count": 5,
                "threshold": threshold,
                "anchor_comparison_sha256": anchor["sha256"],
                "selection": selection,
                "remote_image": dict(REMOTE_IMAGE),
                "queue_sha256": sha256_file(queue_path),
                "source_lock_audit_sha256": audit["sha256"],
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        write_json(output_dir / "execution_plan.json", plan)
        source_archive = output_dir / "source/source_locked.tar.gz"
        source_archive_manifest = output_dir / "source/source_archive_manifest.json"
        manifest = digested(
            {
                "schema": control["package_manifest_schema"],
                "status": control["package_status"],
                "package_id": package_id,
                "campaign_id": campaign_id,
                "row_count": 5,
                "execution_ids": [job["execution_id"] for job in jobs],
                "source_cell_ids": [job["source_cell_id"] for job in jobs],
                "source_horizon": TARGET_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "threshold": threshold,
                "execution_target": "chtc",
                "remote_image": dict(REMOTE_IMAGE),
                "control_files": [
                    binding(output_dir / name, root=output_dir)
                    for name in (
                        "package_control.json",
                        "package_contract.py",
                        "run_cell.py",
                        "execute_authorized_job.sh",
                        "submit.sub.in",
                    )
                ],
                "protocol_bundle_manifest": binding(
                    output_dir / "protocol_bundle_manifest.json",
                    root=output_dir,
                    canonical=True,
                ),
                "source_locks_snapshot": binding(
                    output_dir / "source_locks_snapshot.json",
                    root=output_dir,
                    canonical=True,
                ),
                "source_archive": binding(source_archive, root=output_dir),
                "source_archive_manifest": binding(
                    source_archive_manifest, root=output_dir, canonical=True
                ),
                "source_lock_audit": binding(
                    output_dir / "source_lock_audit.json",
                    root=output_dir,
                    canonical=True,
                ),
                "execution_plan": binding(
                    output_dir / "execution_plan.json",
                    root=output_dir,
                    canonical=True,
                ),
                "queue": binding(queue_path, root=output_dir),
                "protocols": protocols,
                "jobs": [
                    {
                        "execution_id": job["execution_id"],
                        **binding(
                            output_dir / job["job_path"],
                            root=output_dir,
                            canonical=True,
                        ),
                    }
                    for job in jobs
                ],
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submit_descriptor_present": False,
                "submitted": False,
            }
        )
        write_json(output_dir / "package_manifest.json", manifest)
        return {
            "status": "passed_inert_five_rows",
            "package_dir": output_dir.relative_to(REPO_ROOT).as_posix(),
            "package_id": package_id,
            "package_manifest_sha256": manifest["sha256"],
            "source_archive_sha256": sha256_file(source_archive),
            "route_contract_sha256": route_sha,
            "winning_threshold": threshold,
            "row_count": 5,
        }
    except BaseException:
        if output_dir.exists() and not (output_dir / "package_manifest.json").exists():
            shutil.rmtree(output_dir)
        raise
    finally:
        temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-comparison", type=Path, required=True)
    parser.add_argument("--tau-1em5-result", type=Path, required=True)
    parser.add_argument("--tau-1em6-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = materialize(
            anchor_comparison_path=args.anchor_comparison.resolve(),
            result_1em5_path=args.tau_1em5_result.resolve(),
            result_1em6_path=args.tau_1em6_result.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    except (FileExistsError, OSError, SweepError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
