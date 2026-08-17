#!/usr/bin/env python3
"""Derive one horizon-70 protocol inside the exact sealed Page-10 source."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--base-job", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--regime-id", required=True)
    parser.add_argument("--target-horizon", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_package = args.base_package.resolve()
    sys.path.insert(0, base_package.as_posix())
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
    os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
    base = importlib.import_module("run_cell")
    job, _manifest, source_protocol, problem, temporary = base._prepare(
        args.base_job.resolve()
    )
    try:
        from pipelines.static_adapt.ra_adapt.bundles import (
            BundleCellSpec,
            _bundle_protocol_materialization_authority,
        )
        from pipelines.static_adapt.ra_adapt.contracts import (
            _attach_validated_bundle_protocol_authority,
        )
        from pipelines.static_adapt.ra_adapt.engine import (
            build_resolved_ra_protocol,
        )

        bundle_manifest = json.loads(
            args.bundle_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(bundle_manifest, dict):
            raise RuntimeError("Bundle manifest must be a JSON object.")
        unsigned = dict(bundle_manifest)
        expected_manifest_sha = unsigned.pop("sha256", None)
        if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != expected_manifest_sha:
            raise RuntimeError("Bundle manifest self-digest drifted.")
        source_receipt = source_protocol.bundle_materialization
        if source_receipt is None:
            raise RuntimeError("Source protocol lacks bundle authority.")
        request = replace(
            source_protocol.request,
            execution=replace(
                source_protocol.request.execution,
                stop=replace(
                    source_protocol.request.execution.stop,
                    maximum_controller_rounds=args.target_horizon,
                ),
            ),
        )
        cell = BundleCellSpec(
            cell_id=args.execution_id,
            stage="page10_strong_holstein_accepted_state_continuation",
            regime_id=args.regime_id,
            nph=7,
            route_id=str(job["route_id"]),
            algorithm_id=str(job["algorithm_id"]),
            selector_family="ra_adapt",
            candidate_representation=str(job["candidate_representation"]),
            horizon=args.target_horizon,
            source_lock_id=source_receipt.source_lock_id,
        )
        authority = _bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id=str(bundle_manifest["bundle_id"]),
            bundle_manifest_sha256=str(bundle_manifest["sha256"]),
            source_locks_sha256=source_receipt.source_locks_sha256,
            source_lock_refs=source_protocol.source_locks,
            active_gradient_policy=source_protocol.active_gradient_policy,
            resource_weighting_scope=source_protocol.resource_weighting_scope,
        )
        protocol = build_resolved_ra_protocol(
            problem,
            request,
            materialization_authority=authority,
        )
        bound_authority = _bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id=str(bundle_manifest["bundle_id"]),
            bundle_manifest_sha256=str(bundle_manifest["sha256"]),
            source_locks_sha256=source_receipt.source_locks_sha256,
            source_lock_refs=source_protocol.source_locks,
            active_gradient_policy=source_protocol.active_gradient_policy,
            resource_weighting_scope=source_protocol.resource_weighting_scope,
            protocol_sha256=protocol.sha256,
        )
        protocol = _attach_validated_bundle_protocol_authority(
            protocol,
            bound_authority,
        )
        source_request = source_protocol.request.to_dict()
        target_request = protocol.request.to_dict()
        source_request["execution"]["stop"].pop(
            "maximum_controller_rounds", None
        )
        target_request["execution"]["stop"].pop(
            "maximum_controller_rounds", None
        )
        if (
            int(source_protocol.horizon) != 50
            or int(protocol.horizon) != args.target_horizon
            or source_request != target_request
            or protocol.route_contract.get("sha256")
            != source_protocol.route_contract.get("sha256")
            or protocol.algorithm_id != source_protocol.algorithm_id
        ):
            raise RuntimeError(
                "Derived protocol changed more than the target horizon."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(canonical_bytes(protocol.to_dict()) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                {
                    "status": "passed",
                    "execution_id": args.execution_id,
                    "source_protocol_sha256": source_protocol.sha256,
                    "target_protocol_sha256": protocol.sha256,
                    "route_contract_sha256": protocol.route_contract["sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
