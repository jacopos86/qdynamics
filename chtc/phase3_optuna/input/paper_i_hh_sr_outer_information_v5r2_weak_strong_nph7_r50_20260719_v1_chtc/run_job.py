#!/usr/bin/env python3
"""Validate and execute one node of the weak-strong control/reuse pair."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidence_validation import checkpoint_sha256, validate_parent_evidence
from pair_contract import (
    BUNDLE_ID,
    CONTROL_MODE,
    EXPECTED_EXACT_ENERGY,
    EXPECTED_TARGET_ROUND,
    EXACT_ENERGY_TOLERANCE,
    IMAGE_SHA256,
    KEY_SOURCE_SHA256,
    OUTER_PROFILE,
    PAIR_ID,
    REUSE_MODE,
    RUNTIME_ROOT,
    SOURCE_ARCHIVE_SHA256,
    SR_CONTRACT_SHA256,
    SR_PROFILE_REQUEST,
    SR_PROFILE_RESOLVED,
    bundle_dir,
    checkpoint_resume_dir,
    dump_json,
    load_json,
    sha256,
    validate_job,
    validate_resume_current,
    validate_source_lock,
)


SIDECAR_SCHEMA = "paper_i_hh_recovery_prune_aware_qiskit_sidecar_v1"
CHECKPOINT_REPAIR_SCHEMA = "paper_i_checkpoint_execution_order_repair_v1"
RESUME_SCAFFOLD_ENV = "BUNDLE_RESUME_SCAFFOLD_JSON"


def work_root() -> Path:
    explicit = os.environ.get("BUNDLE_WORK_ROOT")
    return Path(explicit).resolve() if explicit else Path(__file__).resolve().parents[4]


def runtime_root() -> Path:
    explicit = os.environ.get("BUNDLE_RUNTIME_ROOT")
    if explicit:
        return Path(explicit).resolve()
    candidate = work_root() / "runtime_source" / RUNTIME_ROOT
    return candidate if candidate.is_dir() else Path.cwd().resolve()


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (work_root() if root is None else root) / path


def build_effective_command(
    manifest: Mapping[str, Any],
    resume_scaffold_json: str | Path | None = None,
    *,
    work_root_override: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Add only the operational structural-resume flags to the locked command."""

    command = [str(token) for token in manifest["command"]["argv"]]
    forbidden = {
        "--adapt-ref-json",
        "--adapt-resume-scaffold-json",
        "--adapt-resume-mode",
        "--adapt-resume-boundary-refit-policy",
        "--adapt-resume-compile-smoke",
    }
    if any(token in forbidden for token in command):
        raise ValueError("locked cold-start command unexpectedly contains resume flags")
    if resume_scaffold_json in {None, ""}:
        return command, {
            "schema": "paper_i_sr_outer_information_operational_resume_v1",
            "active": False,
            "source_controller_round": 0,
            "target_controller_round": EXPECTED_TARGET_ROUND,
        }
    root = work_root() if work_root_override is None else work_root_override.resolve()
    current_path = resolve_path(Path(str(resume_scaffold_json)), root=root).resolve()
    mode = str(manifest.get("pair_contract", {}).get("mode", ""))
    expected_current = (checkpoint_resume_dir(root, mode) / "current.json").resolve()
    if current_path != expected_current:
        raise ValueError("resume current is not the validated mode-private restore")
    validation = validate_resume_current(manifest, current_path)
    source_round = int(validation["source_controller_round"])
    if source_round <= 0:
        raise ValueError("operational continuation would restart at round zero")
    command.extend(
        [
            "--adapt-resume-scaffold-json",
            str(current_path),
            "--adapt-resume-mode",
            "scaffold_v1",
            "--adapt-resume-boundary-refit-policy",
            "verified_checkpoint_no_refit_v1",
            "--adapt-resume-compile-smoke",
            "off",
        ]
    )
    return command, {
        **validation,
        "schema": "paper_i_sr_outer_information_operational_resume_v1",
        "active": True,
        "resume_scaffold_json": str(current_path),
        "resume_boundary_refit_policy": "verified_checkpoint_no_refit_v1",
        "resume_compile_smoke": "off",
        "scientific_command_unchanged": True,
    }


def manifest_paths(
    manifest: Mapping[str, Any],
    *,
    output_root_override: Path | None = None,
) -> dict[str, Path]:
    if output_root_override is None:
        return {
            key: resolve_path(value)
            for key, value in manifest["paths"].items()
            if key != "output_root"
        }
    relative = {
        "current_json": "json/current.json",
        "ledger_json": "json/estimator_call_ledger.json",
        "result_json": "json/result.json",
        "execution_json": "execution.json",
        "normalized_runtime_manifest_json": "normalized_run_manifest.json",
        "validation_json": "validation.json",
        "qiskit_cost_sidecar_json": "qiskit_cost_sidecar.json",
        "repaired_terminal_checkpoint_json": (
            "terminal_checkpoint.execution_order_repaired.json"
        ),
        "anchor_gate_json": "anchor_gate.json",
        "wrapper_exit_receipt_json": "wrapper_exit_receipt.json",
    }
    return {key: output_root_override / value for key, value in relative.items()}


def validate_submission_gate() -> None:
    gate_path = bundle_dir() / "submission_gate.json"
    gate = load_json(gate_path)
    if gate.get("submission_enabled") is not True or gate.get("status") != "pass":
        raise RuntimeError(
            "bundle submission is blocked by submission_gate.json: "
            f"{gate.get('reason')}"
        )


def validate_runtime_source(
    manifest: Mapping[str, Any],
    mode: str,
    *,
    effective_command: Sequence[str] | None = None,
) -> None:
    validate_source_lock(bundle_dir())
    root = runtime_root()
    for relative, expected in KEY_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"runtime key source hash drift: {relative}")
    sys.path.insert(0, str(root))
    from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
    from pipelines.static_adapt.sr_snake_route_profile import (
        canonical_sr_snake_contract,
        canonical_sr_snake_contract_sha256,
        normalize_sr_route_profile_request,
    )

    if normalize_sr_route_profile_request(SR_PROFILE_REQUEST) != SR_PROFILE_RESOLVED:
        raise ValueError("runtime source resolves the wrong SR profile")
    if canonical_sr_snake_contract_sha256(SR_PROFILE_REQUEST) != SR_CONTRACT_SHA256:
        raise ValueError("runtime source resolves the wrong SR contract digest")
    if canonical_sr_snake_contract(SR_PROFILE_REQUEST) != manifest["route_identity"][
        "profile_contract"
    ]:
        raise ValueError("runtime SR contract differs from the authoritative job lock")
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)
    parsed = parser.parse_args(
        list(
            manifest["command"]["argv"]
            if effective_command is None
            else effective_command
        )[3:]
    )
    if parsed.sr_route_profile_contract_sha256 != SR_CONTRACT_SHA256:
        raise ValueError("exact argv resolved the wrong SR contract digest")
    outer = str(parsed.adapt_formal_manifold_route_profile)
    expected = "off" if mode == CONTROL_MODE else OUTER_PROFILE
    if outer != expected:
        raise ValueError("exact argv resolved the wrong outer-information profile")
    if effective_command is not None and RESUME_SCAFFOLD_ENV in os.environ:
        if (
            parsed.adapt_resume_scaffold_json is None
            or parsed.adapt_resume_mode != "scaffold_v1"
            or parsed.adapt_resume_boundary_refit_policy
            != "verified_checkpoint_no_refit_v1"
            or parsed.adapt_resume_compile_smoke != "off"
        ):
            raise ValueError("effective command does not preserve structural continuation")
        from pipelines.static_adapt.resume_scaffold import (
            load_checkpoint_estimator_call_ledger,
            load_static_resume_source,
            validate_static_hh_resume_source,
        )

        resume_source = load_static_resume_source(parsed.adapt_resume_scaffold_json)
        validate_static_hh_resume_source(
            resume_source,
            args=parsed,
            continuation_mode="phase3_v1",
        )
        if load_checkpoint_estimator_call_ledger(resume_source) is None:
            raise ValueError("frozen runtime found no authenticated prefix ledger")
        if mode == REUSE_MODE:
            from pipelines.static_adapt.formal_manifold_outer_information import (
                OuterInformationSession,
            )

            checkpoint = resume_source.payload.get("adapt_vqe", {}).get(
                "sr_outer_information_checkpoint"
            )
            if not isinstance(checkpoint, Mapping):
                raise ValueError("reuse resume lacks its outer-information checkpoint")
            OuterInformationSession.from_checkpoint_payload(checkpoint)


def nested_true(payload: Any, key: str) -> bool:
    if isinstance(payload, Mapping):
        if payload.get(key) is True:
            return True
        return any(nested_true(value, key) for value in payload.values())
    if isinstance(payload, list):
        return any(nested_true(value, key) for value in payload)
    return False


def terminal_checkpoint(result: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in result.get("adapt_vqe", {}).get("active_prefix_checkpoints", [])
        if isinstance(row, Mapping)
        and int(row.get("outer_iteration", -1)) == EXPECTED_TARGET_ROUND
        and row.get("checkpoint_kind") == "post_admission_prune"
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one round-50 post-admission/prune checkpoint; "
            f"got {len(matches)}"
        )
    digest = str(matches[0].get("checkpoint_sha256") or "")
    if len(digest) != 64:
        raise ValueError("terminal checkpoint has no SHA-256")
    return matches[0]


def compile_qiskit_sidecar(
    paths: Mapping[str, Path],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = terminal_checkpoint(result)
    result_digest = sha256(paths["result_json"])
    script = runtime_root() / (
        "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py"
    )
    command = [
        sys.executable,
        str(script),
        "--result-json",
        str(paths["result_json"]),
        "--outer-iteration",
        str(EXPECTED_TARGET_ROUND),
        "--checkpoint-kind",
        "post_admission_prune",
        "--expected-result-sha256",
        result_digest,
        "--expected-checkpoint-sha256",
        str(checkpoint["checkpoint_sha256"]),
        "--repair-permutation-only-execution-order",
        "--repaired-checkpoint-json",
        str(paths["repaired_terminal_checkpoint_json"]),
        "--require-fixed-prefix-replay",
        "--output-json",
        str(paths["qiskit_cost_sidecar_json"]),
    ]
    completed = subprocess.run(command, cwd=work_root(), check=False)
    if completed.returncode != 0:
        raise ValueError(f"terminal Qiskit sidecar failed with exit {completed.returncode}")
    sidecar = load_json(paths["qiskit_cost_sidecar_json"])
    repaired = load_json(paths["repaired_terminal_checkpoint_json"])
    if sidecar.get("schema") != SIDECAR_SCHEMA or sidecar.get("status") != "ok":
        raise ValueError("Qiskit sidecar schema/status drift")
    replay = sidecar.get("fixed_prefix_replay", {})
    if replay.get("status") != "pass" or replay.get("prefix_reconstructed") is not True:
        raise ValueError("Qiskit fixed-prefix replay failed")
    if float(replay.get("energy_abs_discrepancy", float("inf"))) > 1.0e-12:
        raise ValueError("Qiskit fixed-prefix replay energy mismatch")
    source = sidecar.get("source", {})
    if (
        source.get("result_sha256") != result_digest
        or source.get("source_checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or int(source.get("outer_iteration", -1)) != EXPECTED_TARGET_ROUND
        or source.get("checkpoint_kind") != "post_admission_prune"
    ):
        raise ValueError("Qiskit sidecar source provenance drift")
    if repaired.get("schema") != CHECKPOINT_REPAIR_SCHEMA:
        raise ValueError("checkpoint repair schema drift")
    repaired_checkpoint = repaired.get("repaired_checkpoint", {})
    repaired_digest = checkpoint_sha256(repaired_checkpoint)
    if (
        repaired_checkpoint.get("checkpoint_sha256") != repaired_digest
        or repaired.get("repair", {}).get("repaired_checkpoint_sha256")
        != repaired_digest
        or repaired.get("repair", {}).get("substantive_term_changes") is not False
    ):
        raise ValueError("checkpoint repair was not a hash-valid permutation repair")
    historical = sidecar.get("historical_displayed_convention", {})
    current = sidecar.get("current_jr_fake_marrakesh_convention", {})
    if (
        historical.get("status") != "ok"
        or historical.get("identity") != "table_i_basis_gate_transpile_v1"
        or historical.get("backend") is not None
        or int(historical.get("optimization_level", -1)) != 0
        or int(historical.get("seed_transpiler", -1)) != 7
    ):
        raise ValueError("historical Paper-I Qiskit convention drift")
    if (
        current.get("status") != "ok"
        or current.get("identity")
        != "jr_signed_runtime_fake_marrakesh_transpile_v1"
        or current.get("requested_backend") != "FakeMarrakesh"
        or int(current.get("optimization_level", -1)) != 1
        or int(current.get("seed_transpiler", -1)) != 7
    ):
        raise ValueError("current FakeMarrakesh Qiskit convention drift")
    for convention in (historical, current):
        for key in ("N2q", "D2q", "Dc"):
            if int(convention.get("metrics", {}).get(key, -1)) < 0:
                raise ValueError(f"Qiskit metric is missing: {key}")
    return sidecar


def validate_completed_outputs(
    manifest: Mapping[str, Any],
    *,
    output_root_override: Path | None = None,
    compile_qiskit: bool,
) -> dict[str, Any]:
    mode = str(manifest["pair_contract"]["mode"])
    paths = manifest_paths(manifest, output_root_override=output_root_override)
    result = load_json(paths["result_json"])
    current = load_json(paths["current_json"])
    ledger = load_json(paths["ledger_json"])
    exact = float(result["ground_state"]["exact_energy"])
    if abs(exact - EXPECTED_EXACT_ENERGY) > EXACT_ENERGY_TOLERANCE:
        raise ValueError(f"runtime exact-energy drift: {exact}")
    adapt = result.get("adapt_vqe", {})
    settings = result.get("settings", {})
    for payload in (adapt, settings):
        if payload.get("sr_route_profile_resolved") != SR_PROFILE_RESOLVED:
            raise ValueError("result SR profile drift")
        if payload.get("sr_route_profile_contract_sha256") != SR_CONTRACT_SHA256:
            raise ValueError("result SR contract digest drift")
    if adapt.get("route_family") != "singleton_response_snake":
        raise ValueError("result route family drift")
    if adapt.get("candidate_selector_family") != "singleton_response_snake":
        raise ValueError("result selector family drift")
    if adapt.get("adapt_beam_enabled") is not False:
        raise ValueError("beam unexpectedly active")
    sr_contract = adapt.get("sr_route_profile_contract", {})
    sr_execution = sr_contract.get("execution_settings", {})
    sr_semantics = sr_contract.get("semantic_invariants", {})
    if (
        sr_execution.get("phase2_enable_batching") is not False
        or sr_execution.get("phase3_enable_batching") is not False
        or sr_execution.get("phase2_gram_novelty_policy") != "fallback_only_v1"
        or sr_execution.get("phase3_gram_novelty_policy") != "fallback_only_v1"
        or sr_semantics.get("pruning_active") is not False
        or sr_semantics.get("ordinary_phase2_novelty_multiplier_active") is not False
        or sr_semantics.get("ordinary_phase3_novelty_multiplier_active") is not False
        or sr_semantics.get("all_energy_models_infeasible_novelty_fallback_active")
        is not True
    ):
        raise ValueError("result no-beam/no-batch/no-prune/fallback-only contract drift")
    if adapt.get("formal_manifold_query_accounting_complete") is not True:
        raise ValueError("formal query accounting is not closed")
    if adapt.get("nfev_reconciled") is not True:
        raise ValueError("optimizer/guard nfev are not reconciled")
    if nested_true(adapt, "joint_response_selector_invoked"):
        raise ValueError("route accidentally invoked JR selection")
    if nested_true(adapt, "formal_combinatorial_selector_invoked"):
        raise ValueError("route accidentally invoked formal combinatorial selection")
    if nested_true(adapt, "structural_rollback_performed"):
        raise ValueError("route performed forbidden structural rollback")
    composition = adapt.get("formal_manifold_route_composition")
    closure = adapt.get("formal_manifold_query_closure", {})
    outer_stats = closure.get("outer_information_active_reuse", {})
    if mode == CONTROL_MODE:
        if isinstance(composition, Mapping) and composition.get("outer_information_mode") == "active_reuse_v1":
            raise ValueError("control unexpectedly enabled outer reuse")
        if outer_stats:
            raise ValueError("control unexpectedly emitted active outer-reuse statistics")
    else:
        if not isinstance(composition, Mapping):
            raise ValueError("reuse result has no route-composition receipt")
        if (
            composition.get("route_profile") != OUTER_PROFILE
            or composition.get("outer_information_mode") != "active_reuse_v1"
            or composition.get("candidate_selector_family")
            != "singleton_response_snake"
            or composition.get("structural_rollback_enabled") is not False
        ):
            raise ValueError("reuse route-composition receipt drift")
        if not isinstance(outer_stats, Mapping) or int(
            outer_stats.get("active_phase3_contexts", 0)
        ) <= 0:
            raise ValueError("reuse result has no active Phase-III outer contexts")
    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=SR_PROFILE_RESOLVED,
        digest=SR_CONTRACT_SHA256,
        target_round=EXPECTED_TARGET_ROUND,
        target_new_admissions=EXPECTED_TARGET_ROUND,
        require_supported_rank=True,
    )
    checkpoint = terminal_checkpoint(result)
    sidecar = (
        compile_qiskit_sidecar(paths, result)
        if compile_qiskit
        else load_json(paths["qiskit_cost_sidecar_json"])
    )
    estimator = adapt.get("estimator_call_accounting", {})
    winning = estimator.get("winning_branch", {})
    if int(winning.get("S_alg", -1)) < 0:
        raise ValueError("winning-branch S_alg is missing")
    return {
        "schema": "paper_i_sr_outer_information_matched_node_validation_v1",
        "status": "pass",
        "pair_id": PAIR_ID,
        "mode": mode,
        "route_family": "singleton_response_snake",
        "sr_profile": SR_PROFILE_RESOLVED,
        "sr_contract_sha256": SR_CONTRACT_SHA256,
        "outer_information_profile": "off" if mode == CONTROL_MODE else OUTER_PROFILE,
        "exact_energy": exact,
        "variational_energy": float(adapt["energy"]),
        "abs_delta_e": abs(float(adapt["energy"]) - exact),
        "S_alg": int(winning["S_alg"]),
        "nfev_total": int(adapt["nfev_total"]),
        "result_sha256": sha256(paths["result_json"]),
        "current_sha256": sha256(paths["current_json"]),
        "ledger_sha256": sha256(paths["ledger_json"]),
        "terminal_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "qiskit_sidecar_sha256": sha256(paths["qiskit_cost_sidecar_json"]),
        "repaired_terminal_checkpoint_sha256": sha256(
            paths["repaired_terminal_checkpoint_json"]
        ),
        "qiskit": {
            "historical_displayed_convention": sidecar[
                "historical_displayed_convention"
            ]["metrics"],
            "current_fake_marrakesh_convention": sidecar[
                "current_jr_fake_marrakesh_convention"
            ]["metrics"],
        },
        "scientific_evidence_validation": evidence,
        "outer_information_active_reuse": outer_stats if mode == REUSE_MODE else None,
    }


def write_control_gate(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    paths = manifest_paths(manifest)
    dump_json(
        paths["anchor_gate_json"],
        {
            "schema": "paper_i_sr_outer_information_control_gate_v1",
            "status": "pass",
            "pair_id": PAIR_ID,
            "current_runtime_control_validated": True,
            "historical_anchor_reproduction_status": "not_claimed",
            "control_job_manifest_sha256": sha256(manifest_path),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "image_sha256": IMAGE_SHA256,
            "result_sha256": validation["result_sha256"],
            "validation_sha256": None,
            "exact_energy": validation["exact_energy"],
            "abs_delta_e": validation["abs_delta_e"],
            "S_alg": validation["S_alg"],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    validate_only = bool(args and args[0] == "--validate-only")
    if validate_only:
        args = args[1:]
    if len(args) != 1:
        raise SystemExit("usage: run_job.py [--validate-only] JOB.json")
    manifest_path = resolve_path(args[0])
    manifest = load_json(manifest_path)
    mode = str(manifest.get("pair_contract", {}).get("mode", ""))
    validate_job(
        manifest,
        expected_mode=mode,
        require_anchor=not validate_only,
        work_root=work_root(),
    )
    effective_command, resume_provenance = build_effective_command(
        manifest,
        os.environ.get(RESUME_SCAFFOLD_ENV),
    )
    validate_runtime_source(
        manifest,
        mode,
        effective_command=effective_command,
    )
    if validate_only:
        print(json.dumps({"status": "pass", "mode": mode}, sort_keys=True))
        return 0
    validate_submission_gate()
    paths = manifest_paths(manifest)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    environment = {str(key): str(value) for key, value in manifest["environment"].items()}
    environment["PYTHONPATH"] = str(runtime_root())
    for key, value in environment.items():
        if key.endswith("_CACHE_DIR"):
            cache = resolve_path(value)
            if cache.exists():
                raise ValueError(f"job-local cache must start absent: {cache}")
            cache.mkdir(parents=True)
    control_gate_provenance = None
    if mode == REUSE_MODE:
        control_gate_path = resolve_path(
            manifest["pair_contract"]["control_gate_path"]
        )
        control_gate_provenance = {
            "path": str(control_gate_path),
            "sha256": sha256(control_gate_path),
            "payload": load_json(control_gate_path),
        }
    normalized = {
        "schema": "paper_i_sr_outer_information_runtime_manifest_v1",
        "pair_id": PAIR_ID,
        "mode": mode,
        "job_manifest": str(manifest_path),
        "job_manifest_sha256": sha256(manifest_path),
        "command_argv": manifest["command"]["argv"],
        "effective_command_argv": effective_command,
        "operational_resume": resume_provenance,
        "environment": environment,
        "physics": manifest["physics"],
        "route_identity": manifest["route_identity"],
        "segment": manifest["segment"],
        "source_lock": manifest["source_lock"],
        "image_sha256": IMAGE_SHA256,
        "control_gate_provenance": control_gate_provenance,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(paths["normalized_runtime_manifest_json"], normalized)
    execution = {
        **normalized,
        "schema": "paper_i_sr_outer_information_execution_v1",
        "status": "running",
        "exit_code": None,
    }
    dump_json(paths["execution_json"], execution)
    env = os.environ.copy()
    env.update(environment)
    returncode = 70
    try:
        completed = subprocess.run(
            effective_command,
            cwd=work_root(),
            env=env,
            check=False,
        )
        returncode = int(completed.returncode)
        if returncode == 0:
            validation = validate_completed_outputs(
                manifest,
                compile_qiskit=True,
            )
            dump_json(paths["validation_json"], validation)
            if mode == CONTROL_MODE:
                write_control_gate(manifest_path, manifest, validation)
        execution["status"] = "completed" if returncode == 0 else "failed"
        execution["exit_code"] = returncode
        return returncode
    except Exception as exc:
        if returncode == 0:
            returncode = 70
        execution["status"] = "failed"
        execution["exit_code"] = returncode
        execution["validation_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        execution["finished_utc"] = datetime.now(timezone.utc).isoformat()
        execution["artifacts"] = {
            key: {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
            for key, path in paths.items()
            if key not in {"execution_json", "wrapper_exit_receipt_json"}
        }
        dump_json(paths["execution_json"], execution)


if __name__ == "__main__":
    raise SystemExit(main())
