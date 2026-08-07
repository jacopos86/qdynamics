#!/usr/bin/env python3
"""Validate one CHTC source-value anchor before DAG fan-out."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_result(transfer_tar: Path) -> tuple[dict[str, Any], str]:
    with tarfile.open(transfer_tar, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (
                member.name.endswith("/json/result.json")
                or member.name.endswith("/json/current.json")
            )
        ]
        result_members = [
            member for member in members if member.name.endswith("/json/result.json")
        ]
        selected = result_members[0] if result_members else members[0]
        handle = archive.extractfile(selected)
        if handle is None:
            raise ValueError(f"could not read {selected.name}")
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("anchor result payload is not an object")
    return payload, selected.name


def _adapt(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = payload.get("adapt_vqe")
    if isinstance(direct, Mapping):
        return direct
    wrapped = payload.get("result")
    if isinstance(wrapped, Mapping) and isinstance(wrapped.get("adapt_vqe"), Mapping):
        return wrapped["adapt_vqe"]
    raise ValueError("anchor result lacks adapt_vqe")


def _selected_label(row: Mapping[str, Any]) -> str:
    selected = row.get("selected_op")
    if isinstance(selected, str):
        return selected
    if isinstance(selected, Mapping):
        return str(selected.get("label") or selected.get("candidate_label") or "")
    return ""


def _patch_child_bundle(child_bundle: Path, *, passed: bool) -> None:
    status = "pass" if passed else "diagnostic_invalid"
    for directory in (child_bundle / "jobs", child_bundle / "normalized_manifests"):
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            sensitivity = payload.get("sensitivity_study")
            if isinstance(sensitivity, dict):
                sensitivity["anchor_reproduces_source"] = passed
                sensitivity["status"] = status
            _dump(path, payload)
    manifest_path = child_bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = (
        "anchor_passed_ready_for_fanout" if passed else "anchor_failed_fanout_blocked"
    )
    _dump(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-tar", type=Path, required=True)
    parser.add_argument("--expectation", type=Path, required=True)
    parser.add_argument("--anchor-audit", type=Path, required=True)
    parser.add_argument("--child-audit", type=Path, required=True)
    parser.add_argument("--child-bundle", type=Path, required=True)
    args = parser.parse_args()

    expectation = json.loads(args.expectation.read_text(encoding="utf-8"))
    anchor_audit = json.loads(args.anchor_audit.read_text(encoding="utf-8"))
    child_audit = json.loads(args.child_audit.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    passed = False
    error: str | None = None
    member_name: str | None = None
    try:
        payload, member_name = _load_result(args.transfer_tar)
        adapt = _adapt(payload)
        history = adapt.get("history")
        if not isinstance(history, list):
            raise ValueError("anchor result history is absent")
        expected_depth = int(expectation["depth"])
        observed = history[:expected_depth]
        observed_labels = [_selected_label(row) for row in observed]
        observed_energies = [float(row["energy_after_opt"]) for row in observed]
        expected_labels = [str(value) for value in expectation["selected_labels"]]
        expected_energies = [float(value) for value in expectation["energies"]]
        energy_diffs = [
            abs(actual - expected)
            for actual, expected in zip(observed_energies, expected_energies, strict=True)
        ]
        checks = {
            "depth_match": len(observed) == expected_depth,
            "operator_sequence_match": observed_labels == expected_labels,
            "energy_trajectory_max_abs_diff": max(energy_diffs, default=0.0),
            "energy_trajectory_tolerance": float(
                expectation["energy_abs_tolerance"]
            ),
            "final_error_abs_diff": abs(
                float(observed[-1]["delta_abs_current"])
                - float(expectation["final_error"])
            ),
            "final_error_tolerance": float(
                expectation["final_error_abs_tolerance"]
            ),
            "route_profile_match": (
                str(adapt.get("sr_route_profile") or "")
                == str(expectation["profile_resolved"])
            ),
            "route_contract_sha256_match": (
                str(adapt.get("sr_route_profile_contract_sha256") or "")
                == str(expectation["profile_contract_sha256"])
            ),
            "stop_depth_match": int(adapt.get("ansatz_depth") or 0)
            == expected_depth,
        }
        passed = bool(
            checks["depth_match"]
            and checks["operator_sequence_match"]
            and checks["energy_trajectory_max_abs_diff"]
            <= checks["energy_trajectory_tolerance"]
            and checks["final_error_abs_diff"] <= checks["final_error_tolerance"]
            and checks["route_profile_match"]
            and checks["route_contract_sha256_match"]
            and checks["stop_depth_match"]
        )
        if not passed:
            error = "source-value anchor did not reproduce the visible source prefix"
    except Exception as exc:  # fail closed and persist the reason
        error = f"{type(exc).__name__}: {exc}"

    anchor_record = {
        "value": "append_only",
        "anchor_result_transfer_tar": str(args.transfer_tar),
        "anchor_result_transfer_tar_sha256": (
            _sha256(args.transfer_tar) if args.transfer_tar.is_file() else None
        ),
        "anchor_result_member": member_name,
        "anchor_reproduces_source": passed,
        "checks": checks,
        "error": error,
        "non_swept_settings_diff": [],
    }
    for payload in (anchor_audit, child_audit):
        payload["anchor"] = anchor_record
        payload["status"] = "pass" if passed else "diagnostic_invalid"
    _dump(args.anchor_audit, anchor_audit)
    _dump(args.child_audit, child_audit)
    _patch_child_bundle(args.child_bundle, passed=passed)
    print(json.dumps(anchor_record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
