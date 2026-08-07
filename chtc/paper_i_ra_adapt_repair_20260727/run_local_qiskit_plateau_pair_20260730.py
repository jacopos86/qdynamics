#!/usr/bin/env python3
"""Run one authorized cell from the local Qiskit-cost plateau diagnostic.

The runner is intentionally protocol-driven.  It does not reconstruct route
settings, import a package-local worker module, or mutate a materialized
bundle.  A caller selects exactly one validated protocol and a fresh output
root; invoking the runner requires the explicit ``--execution-authorized``
switch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FULL_HORIZON = 50
RUNNER_SCHEMA = "paper_i_ra_adapt_local_qiskit_plateau_runner_v1"
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_local_execution_authorization_v1"
)
MANIFEST_SCHEMA = "paper_i_ra_adapt_local_run_manifest_v1"
TERMINAL_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_local_terminal_execution_receipt_v1"
)
FAILURE_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_local_failed_execution_receipt_v1"
)
QISKIT_PLATEAU_ALGORITHM_IDS = frozenset(
    {
        (
            "paper_i_ra_adapt_global_singleton_plateau_commutation_"
            "qiskit_transpile_cost_v1"
        ),
        (
            "paper_i_ra_adapt_macro_plateau_insertion_"
            "qiskit_transpile_cost_v1"
        ),
    }
)


class LocalRunContractError(RuntimeError):
    """Fail-closed local-run contract violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "sha256" in value:
        raise LocalRunContractError(
            "A digested payload cannot supply its own SHA-256."
        )
    value["sha256"] = _canonical_sha256(value)
    return value


def _load_self_digested_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LocalRunContractError(f"{label} is unavailable or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalRunContractError(f"Could not load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise LocalRunContractError(f"{label} must be a JSON object.")
    supplied = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise LocalRunContractError(f"{label} self-digest drifted.")
    return value


def _file_binding(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LocalRunContractError(
            f"Artifact is unavailable or unsafe: {path}"
        )
    resolved = path.resolve()
    rendered = resolved.as_posix()
    if root is not None:
        try:
            rendered = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise LocalRunContractError(
                f"Artifact escaped its output root: {resolved}"
            ) from exc
    return {
        "path": rendered,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _optional_file_binding(
    path: Path,
    *,
    root: Path,
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _file_binding(path, root=root)


def _atomic_write_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(
            f"Refusing stale temporary artifact: {temporary}"
        )
    payload = _canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_repo_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise LocalRunContractError(
            f"{label} must live in the active checkout: {REPO_ROOT}"
        ) from exc
    return resolved


def _problem_from_protocol(protocol: Any) -> Any:
    """Rebuild the physical problem through the public problem registry."""

    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        ResolvedProblemReceipt,
    )

    receipt = protocol.problem
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )
    if ResolvedProblemReceipt.from_problem(problem) != receipt:
        raise LocalRunContractError(
            "The reconstructed problem drifted from the validated protocol."
        )
    return problem


def _validate_protocol_and_inventory(
    protocol_path: Path,
) -> tuple[Any, dict[str, Any]]:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _implementation_source_inventory,
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        RESOURCE_WEIGHTING_ALL_PHASE,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        FreshStart,
        PlateauCommutationInsertion,
    )

    protocol = load_validated_bundle_protocol(protocol_path)
    if protocol.algorithm_id not in QISKIT_PLATEAU_ALGORITHM_IDS:
        raise LocalRunContractError(
            "The selected protocol is not an authorized Qiskit-cost plateau "
            "diagnostic cell."
        )
    if protocol.candidate_representation not in {
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    }:
        raise LocalRunContractError(
            "The selected protocol has an unsupported representation."
        )
    if not isinstance(
        protocol.request.method.insertion,
        PlateauCommutationInsertion,
    ):
        raise LocalRunContractError(
            "The selected protocol does not use plateau insertion."
        )
    if protocol.resource_weighting_scope != RESOURCE_WEIGHTING_ALL_PHASE:
        raise LocalRunContractError(
            "The Qiskit-cost diagnostic requires all-phase resource "
            "weighting."
        )
    if int(protocol.horizon) != FULL_HORIZON:
        raise LocalRunContractError(
            f"The diagnostic protocol horizon must be {FULL_HORIZON}."
        )
    if (
        int(
            protocol.request.execution.stop.maximum_controller_rounds
        )
        != FULL_HORIZON
        or not isinstance(protocol.request.execution.resume, FreshStart)
    ):
        raise LocalRunContractError(
            "The validated protocol lost its full-horizon fresh-start "
            "execution contract."
        )

    source_locks_path = protocol_path.parent.parent / "source_locks.json"
    source_locks = _load_self_digested_json(
        source_locks_path,
        label="bundle source-lock authority",
    )
    expected_inventory = source_locks.get("implementation_sources")
    if not isinstance(expected_inventory, Mapping):
        raise LocalRunContractError(
            "The bundle has no implementation-source authority."
        )
    current_inventory = _implementation_source_inventory(REPO_ROOT)
    if dict(expected_inventory) != current_inventory:
        raise LocalRunContractError(
            "The active implementation inventory drifted from the bundle "
            "authority: expected "
            f"{expected_inventory.get('sha256')}, observed "
            f"{current_inventory.get('sha256')}."
        )
    return protocol, current_inventory


def _round_count(value: int | None, *, protocol_horizon: int) -> int:
    if value is None:
        return int(protocol_horizon)
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise LocalRunContractError("--rounds must be a positive integer.")
    rounds = int(value)
    if rounds > int(protocol_horizon):
        raise LocalRunContractError(
            "--rounds may shorten but cannot extend the protocol horizon."
        )
    return rounds


def _prepare_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"Refusing existing output root: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(exist_ok=False)
    return resolved


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "execution_authorization": (
            output_root / "execution_authorization.json"
        ),
        "run_manifest": output_root / "run_manifest.json",
        "checkpoint": output_root / "checkpoint.json",
        "estimator_ledger": output_root / "estimator_ledger.json",
        "result": output_root / "result.json",
        "summary": output_root / "summary.json",
        "scientific_receipts": output_root / "scientific_receipts.json",
        "terminal_receipt": output_root / "terminal_receipt.json",
        "failure_receipt": output_root / "failure_receipt.json",
    }


def _authorization_receipt(
    *,
    protocol_path: Path,
    protocol: Any,
    rounds: int,
    output_root: Path,
    resume_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _digested(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "run_class": "diagnostic",
            "execution_target": "local",
            "authorization_basis": (
                "explicit_user_local_qiskit_plateau_resume_authorization_"
                "20260731"
                if resume_binding is not None
                else "explicit_user_local_qiskit_plateau_pair_authorization_"
                "20260730"
            ),
            "authorization_scope": "one_validated_protocol_cell_v1",
            "protocol_path": protocol_path.relative_to(REPO_ROOT).as_posix(),
            "protocol_sha256": protocol.sha256,
            "bundle_id": protocol.bundle_id,
            "bundle_manifest_sha256": protocol.bundle_manifest_sha256,
            "maximum_controller_rounds": rounds,
            "execution_mode": (
                "authenticated_accepted_state_resume_v1"
                if resume_binding is not None
                else "fresh_start_v1"
            ),
            "resume_input": (
                None if resume_binding is None else dict(resume_binding)
            ),
            "output_root": output_root.as_posix(),
            "execution_authorized": True,
            "submission_authorized": False,
            "authorized_at_utc": _utc_now(),
        }
    )


def _run_manifest(
    *,
    protocol_path: Path,
    protocol: Any,
    inventory: Mapping[str, Any],
    rounds: int,
    output_root: Path,
    paths: Mapping[str, Path],
    authorization_binding: Mapping[str, Any],
    resume_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    compact_resume_evidence = resume_binding is not None
    return _digested(
        {
            "schema": MANIFEST_SCHEMA,
            "runner_schema": RUNNER_SCHEMA,
            "run_class": "diagnostic",
            "execution_target": "local",
            "execution_id": protocol_path.stem,
            "algorithm_id": protocol.algorithm_id,
            "candidate_representation": protocol.candidate_representation,
            "active_gradient_policy": protocol.active_gradient_policy,
            "resource_weighting_scope": protocol.resource_weighting_scope,
            "insertion_policy": protocol.request.method.insertion.kind,
            "selector_cost_policy": (
                "qiskit_full_trial_ansatz_delta_all_phases_v1"
            ),
            "selector_cost_phase_reuse": (
                "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
            ),
            "protocol": {
                **_file_binding(protocol_path),
                "canonical_sha256": protocol.sha256,
                "bundle_id": protocol.bundle_id,
                "bundle_manifest_sha256": (
                    protocol.bundle_manifest_sha256
                ),
            },
            "implementation_inventory_sha256": inventory["sha256"],
            "runner": _file_binding(Path(__file__).resolve()),
            "execution_authorization": dict(authorization_binding),
            "protocol_horizon": int(protocol.horizon),
            "maximum_controller_rounds": rounds,
            "execution_mode": (
                "authenticated_accepted_state_resume_v1"
                if resume_binding is not None
                else "fresh_start_v1"
            ),
            "evidence_materialization": (
                "checkpoint_ledger_summary_receipts_v1"
                if compact_resume_evidence
                else "full_terminal_artifacts_v1"
            ),
            "resume_input": (
                None if resume_binding is None else dict(resume_binding)
            ),
            "optimizer": protocol.optimizer,
            "optimizer_maxiter": int(protocol.optimizer_maxiter),
            "seeds": dict(protocol.seeds),
            "output_root": output_root.as_posix(),
            "declared_artifacts": {
                role: path.relative_to(output_root).as_posix()
                for role, path in paths.items()
                if not compact_resume_evidence
                or role not in {"estimator_ledger", "result"}
            },
            "execution_authorized": True,
            "submission_authorized": False,
            "created_at_utc": _utc_now(),
        }
    )


def _write_failure_receipt(
    *,
    output_root: Path,
    paths: Mapping[str, Path],
    protocol: Any,
    rounds: int,
    error: BaseException,
    resume_binding: Mapping[str, Any] | None,
) -> None:
    checkpoint = _optional_file_binding(
        paths["checkpoint"],
        root=output_root,
    )
    ledger = _optional_file_binding(
        paths["estimator_ledger"],
        root=output_root,
    )
    receipt = _digested(
        {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "status": (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            ),
            "algorithm_id": protocol.algorithm_id,
            "protocol_sha256": protocol.sha256,
            "maximum_controller_rounds": rounds,
            "resume_input": (
                None if resume_binding is None else dict(resume_binding)
            ),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "checkpoint": checkpoint,
            "estimator_ledger": ledger,
            "failed_at_utc": _utc_now(),
            "execution_authorized": True,
            "submission_authorized": False,
        }
    )
    _atomic_write_json_noreplace(paths["failure_receipt"], receipt)


def run_one(
    *,
    protocol_path: Path,
    output_root: Path,
    rounds_override: int | None,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    protocol_path = _require_repo_path(
        protocol_path,
        label="protocol",
    )
    if protocol_path.is_symlink():
        raise LocalRunContractError("The protocol path cannot be a symlink.")
    protocol, inventory = _validate_protocol_and_inventory(protocol_path)
    rounds = _round_count(
        rounds_override,
        protocol_horizon=int(protocol.horizon),
    )
    resume_binding: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resolved_resume = resume_checkpoint.expanduser().resolve()
        if not resolved_resume.is_file() or resolved_resume.is_symlink():
            raise LocalRunContractError(
                "The resume checkpoint is unavailable or unsafe."
            )
        resume_binding = _file_binding(resolved_resume)
    output_root = _prepare_output_root(output_root)
    paths = _artifact_paths(output_root)

    authorization = _authorization_receipt(
        protocol_path=protocol_path,
        protocol=protocol,
        rounds=rounds,
        output_root=output_root,
        resume_binding=resume_binding,
    )
    _atomic_write_json_noreplace(
        paths["execution_authorization"],
        authorization,
    )
    authorization_binding = _file_binding(
        paths["execution_authorization"],
        root=output_root,
    )
    manifest = _run_manifest(
        protocol_path=protocol_path,
        protocol=protocol,
        inventory=inventory,
        rounds=rounds,
        output_root=output_root,
        paths=paths,
        authorization_binding=authorization_binding,
        resume_binding=resume_binding,
    )
    _atomic_write_json_noreplace(paths["run_manifest"], manifest)

    try:
        from pipelines.static_adapt.ra_adapt import run_ra_adapt
        from pipelines.static_adapt.ra_adapt.contracts import (
            RAAdaptOperationalControls,
        )
        from pipelines.static_adapt.sr_snake.contracts import (
            AcceptedStateResume,
            CheckpointObservation,
            EstimatorLedgerObservation,
            FreshStart,
            SRObservationPolicy,
        )

        resume_policy = (
            FreshStart()
            if resume_binding is None
            else AcceptedStateResume(
                checkpoint_path=Path(str(resume_binding["path"])),
                checkpoint_sha256=str(resume_binding["sha256"]),
            )
        )
        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=rounds,
            resume=resume_policy,
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=paths["checkpoint"],
                    every_controller_rounds=(
                        rounds if resume_binding is not None else 1
                    ),
                    keep_history_tail=100,
                ),
                estimator_ledger=(
                    None
                    if resume_binding is not None
                    else EstimatorLedgerObservation(
                        path=paths["estimator_ledger"],
                    )
                ),
            ),
        )
        problem = _problem_from_protocol(protocol)
        original_cwd = Path.cwd()
        os.chdir(protocol_path.parent.parent)
        try:
            result = run_ra_adapt(
                problem,
                protocol,
                operational_controls=controls,
            )
        finally:
            os.chdir(original_cwd)

        if result.run.paper_i_summary is None:
            raise LocalRunContractError(
                "The completed plateau run has no Paper-I summary."
            )
        summary_payload = result.run.paper_i_summary.to_dict()
        receipts_payload = dict(result.scientific_receipts)
        if resume_binding is None:
            _atomic_write_json_noreplace(
                paths["result"], result.to_dict()
            )
        _atomic_write_json_noreplace(paths["summary"], summary_payload)
        _atomic_write_json_noreplace(
            paths["scientific_receipts"],
            receipts_payload,
        )

        artifact_roles = [
            "execution_authorization",
            "run_manifest",
            "checkpoint",
            "summary",
            "scientific_receipts",
        ]
        if resume_binding is None:
            artifact_roles.extend(("estimator_ledger", "result"))
        artifact_bindings = {
            role: _file_binding(paths[role], root=output_root)
            for role in artifact_roles
        }
        terminal = _digested(
            {
                "schema": TERMINAL_RECEIPT_SCHEMA,
                "status": "passed",
                "execution_id": protocol_path.stem,
                "algorithm_id": protocol.algorithm_id,
                "protocol_sha256": protocol.sha256,
                "bundle_id": protocol.bundle_id,
                "implementation_inventory_sha256": inventory["sha256"],
                "maximum_controller_rounds": rounds,
                "accepted_controller_rounds": len(
                    result.accepted_trajectory
                ),
                "execution_mode": (
                    "authenticated_accepted_state_resume_v1"
                    if resume_binding is not None
                    else "fresh_start_v1"
                ),
                "resume_input": (
                    None
                    if resume_binding is None
                    else dict(resume_binding)
                ),
                "final_energy": float(result.final_state.energy),
                "stop": result.run.stop.to_dict(),
                "artifacts": artifact_bindings,
                "completed_at_utc": _utc_now(),
                "execution_authorized": True,
                "submission_authorized": False,
            }
        )
        _atomic_write_json_noreplace(paths["terminal_receipt"], terminal)
        return terminal
    except BaseException as exc:
        try:
            _write_failure_receipt(
                output_root=output_root,
                paths=paths,
                protocol=protocol,
                rounds=rounds,
                error=exc,
                resume_binding=resume_binding,
            )
        except Exception:
            traceback.print_exc()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one validated 50-round Qiskit-cost RA-ADAPT plateau "
            "diagnostic locally."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Validated bundle protocol JSON for one diagnostic cell.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Fresh directory in which all run artifacts will be written.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help=(
            "Optional shortened controller horizon; defaults to the "
            "protocol's required horizon of 50."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Optional authenticated accepted-state checkpoint. The protocol "
            "remains source-locked fresh-start authority; continuation is "
            "applied only through operational controls."
        ),
    )
    parser.add_argument(
        "--execution-authorized",
        action="store_true",
        help="Required explicit acknowledgement of local run authorization.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.execution_authorized:
        print(
            "ERROR: --execution-authorized is required; no run was started.",
            file=sys.stderr,
        )
        return 2
    try:
        terminal = run_one(
            protocol_path=args.protocol,
            output_root=args.output_root,
            rounds_override=args.rounds,
            resume_checkpoint=args.resume_checkpoint,
        )
    except KeyboardInterrupt:
        print("INTERRUPTED: checkpoint and failure receipt preserved.", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    print(_canonical_json_bytes(terminal).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
