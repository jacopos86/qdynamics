#!/usr/bin/env python3
"""Resume the two user-authorized weak-Holstein macro append cells to k=50.

The earlier conditional campaign correctly closed these cells at k=30 under
its plateau gate.  This adapter preserves that historical receipt and records
the later, explicit Paper-I decision to expose both weak-Holstein append-only
comparators through the same k=50 plotting horizon.  No scientific setting
other than the already source-authorized controller horizon is changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONTINUATION = HERE / "continue_local_page16_insertion_comparators_k30_to_k50_20260813.py"
ACTIVATION_DIR = HERE / (
    "paper_i_ra_adapt_page16_insertion_comparators_k30_to_k50_"
    "20260813_v2_local_activation"
)
RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_weak_append_user_authorized_k30_to_k50_20260813_v3"
)
TARGETS = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_append_only",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_weak__nph3__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_append_only",
)
AUTHORIZATION_SCHEMA = "paper_i_page16_user_authorized_weak_append_k50_resume_v1"
RUNTIME_SCHEMA = "paper_i_page16_user_authorized_weak_append_k50_runtime_v1"
EFFECTIVE_GATE_SCHEMA = "paper_i_page16_user_authorized_effective_resume_gate_v1"


class OverrideError(RuntimeError):
    """The source checkpoint or explicit-override contract did not close."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    return {
        **unsigned,
        "sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location("page16_k50_adapter", CONTINUATION)
    if spec is None or spec.loader is None:
        raise OverrideError("Cannot load the pinned Page-16 continuation adapter.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value) + b"\n")


def _prepare_override(adapter: Any, execution_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    worker = adapter.k30._load_worker()
    jobs = adapter._job_by_id(worker)
    job = jobs[execution_id]
    decision = adapter._closed_k30_decision(
        worker,
        execution_id=execution_id,
        job=job,
    )
    if decision is None or decision.get("extension_decision") != "stop_at_k30":
        raise OverrideError(f"Expected an authenticated stop-at-k30 source: {execution_id}")
    if int(job["target_horizon"]) != adapter.TARGET_HORIZON:
        raise OverrideError(f"Weak source protocol was not already authorized to k=50: {execution_id}")

    activation, bundle = adapter._validate_activation(worker, ACTIVATION_DIR)
    conditional = adapter._conditional_authorization(
        ACTIVATION_DIR,
        activation,
        execution_id,
    )
    if conditional.get("target_protocol", {}).get("kind") != "source_protocol_reused_exactly":
        raise OverrideError(f"Target protocol is not an exact source reuse: {execution_id}")

    source_gate = adapter.k30._load_digested(
        worker,
        adapter.K30_RUNTIME_DIR / "plateau_gates" / f"{execution_id}.json",
        label=f"source plateau gate {execution_id}",
    )
    adapter._validate_resume_gate_files(
        worker,
        job=job,
        run_root=Path(str(decision["run_root"])),
        gate=source_gate,
    )
    effective_gate = _digest(
        {
            **{key: value for key, value in source_gate.items() if key != "sha256"},
            "schema": EFFECTIVE_GATE_SCHEMA,
            "status": "passed_user_authorized_resume_to_k50",
            "extension_decision": "eligible_for_authenticated_resume_to_k50",
            "historical_plateau_gate_sha256": source_gate["sha256"],
            "decision_override_scope": "display_horizon_only_weak_holstein_macro_append_only",
            "scientific_setting_changes": [],
            "user_authorized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    authority = _digest(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_authenticated_resume_to_k50",
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "source_protocol_sha256": job["protocol_sha256"],
            "target_protocol": conditional["target_protocol"],
            "route_contract_sha256": job["route_contract_sha256"],
            "comparator_policy": job["comparator_policy"],
            "regime_id": job["regime_id"],
            "nph": int(job["nph"]),
            "source_authorized_horizon": int(job["target_horizon"]),
            "resume_round": adapter.SOURCE_HORIZON,
            "target_horizon": adapter.TARGET_HORIZON,
            "continuation_materialization_requirement": decision[
                "continuation_materialization_requirement"
            ],
            "k30_execution_manifest_sha256": decision["k30_execution_manifest_sha256"],
            "k30_worker_receipt_sha256": decision["k30_worker_receipt_sha256"],
            "k30_plateau_gate_sha256": effective_gate["sha256"],
            "historical_k30_plateau_gate_sha256": source_gate["sha256"],
            "resume_checkpoint": decision["resume_checkpoint"],
            "resume_checkpoint_siblings": decision["resume_checkpoint_siblings"],
            "source_run_root": decision["run_root"],
            "conditional_authorization_sha256": conditional["sha256"],
            "activation_manifest_sha256": activation["sha256"],
            "continuation_bundle_manifest_sha256": bundle["sha256"],
            "accepted_state_resume_required": True,
            "fresh_start_authorized": False,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": True,
            "authorization_basis": "direct_user_instruction_2026-08-13",
            "scientific_setting_changes": [],
        }
    )
    return authority, effective_gate, bundle


def run(execution_id: str) -> dict[str, Any]:
    adapter = _load_adapter()
    runtime_dir = RUNTIME_DIR / execution_id
    if runtime_dir.exists() or runtime_dir.is_symlink():
        raise OverrideError(f"Refusing to overwrite an existing override runtime: {runtime_dir}")
    authority, effective_gate, bundle = _prepare_override(adapter, execution_id)
    for name in ("runs", "worker_receipts", "quarantine", "provenance"):
        (runtime_dir / name).mkdir(parents=True, exist_ok=True)
    _write_json(runtime_dir / "provenance/resume_authorization.json", authority)
    _write_json(runtime_dir / "provenance/effective_resume_gate.json", effective_gate)
    runtime = _digest(
        {
            "schema": RUNTIME_SCHEMA,
            "status": "authorized_single_cell_resume",
            "execution_id": execution_id,
            "source_horizon": adapter.SOURCE_HORIZON,
            "target_horizon": adapter.TARGET_HORIZON,
            "resume_authorization_sha256": authority["sha256"],
            "effective_resume_gate_sha256": effective_gate["sha256"],
            "continuation_bundle_manifest_sha256": bundle["sha256"],
            "adapter_sha256": _sha256_file(CONTINUATION),
            "activation_manifest_sha256": authority["activation_manifest_sha256"],
            "maximum_concurrency": 1,
            "execution_authorized": True,
            "paper_evidence_adoption_authorized": True,
        }
    )
    _write_json(runtime_dir / "runtime_manifest.json", runtime)

    original_authorization = adapter._resume_authorization
    original_gate_loader = adapter.k30._load_digested
    original_gate_validator = adapter._validate_resume_gate_files
    source_gate_path = adapter.K30_RUNTIME_DIR / "plateau_gates" / f"{execution_id}.json"

    def override_authorization(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return authority, runtime, bundle

    def override_gate_loader(worker: Any, path: Path, *, label: str) -> dict[str, Any]:
        if Path(path) == source_gate_path:
            return dict(effective_gate)
        return original_gate_loader(worker, path, label=label)

    def override_gate_validator(
        worker: Any,
        *,
        job: Mapping[str, Any],
        run_root: Path,
        gate: Mapping[str, Any],
    ) -> dict[str, Any]:
        if gate.get("sha256") != effective_gate["sha256"]:
            raise OverrideError("Effective resume gate identity drifted.")
        historical = original_gate_loader(
            worker,
            source_gate_path,
            label=f"historical plateau gate {execution_id}",
        )
        original_gate_validator(worker, job=job, run_root=run_root, gate=historical)
        return dict(gate)

    adapter._resume_authorization = override_authorization
    adapter.k30._load_digested = override_gate_loader
    adapter._validate_resume_gate_files = override_gate_validator
    token_name = adapter.LOCAL_CHILD_TOKEN_ENV
    previous_token = os.environ.get(token_name)
    os.environ[token_name] = f"{runtime['sha256']}:{execution_id}"
    try:
        receipt = adapter.run_cell(
            execution_id=execution_id,
            activation_dir=ACTIVATION_DIR,
            runtime_dir=runtime_dir,
        )
    finally:
        adapter._resume_authorization = original_authorization
        adapter.k30._load_digested = original_gate_loader
        adapter._validate_resume_gate_files = original_gate_validator
        if previous_token is None:
            os.environ.pop(token_name, None)
        else:
            os.environ[token_name] = previous_token
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-id", choices=TARGETS, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.execution_id), sort_keys=True))
        return 0
    except (OverrideError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
