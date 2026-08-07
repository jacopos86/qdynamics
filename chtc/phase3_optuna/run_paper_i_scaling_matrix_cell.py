#!/usr/bin/env python3
"""Run one parent-only Paper-I scaling-matrix cell."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCALING_PROFILE = "paper_i_scaling_matrix_20260710_v1"
SECTOR_CONTRACT_SENTINEL_BATCH_ID = "paper_i_snake_sector_contract_sentinels_20260713_v1"
METHOD_ALGORITHMS = {
    "snake": "static_family_native_adapt_phase3",
    "geo": "static_geo_adapt_vqe",
    "append": "static_full_meta_append_adapt_vqe",
}
POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING = (
    "accept_finite_nonincreasing_v1"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_record(record_id: str, records_path: Path) -> dict[str, str]:
    with records_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if str(row.get("record_id") or "") == str(record_id):
                return {str(key): "" if value is None else str(value) for key, value in row.items()}
    raise KeyError(f"record_id={record_id!r} not found in {records_path}")


def _positive_int(row: Mapping[str, str], field: str) -> int:
    raw = str(row.get(field) or "").strip()
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"{field} must be an integer; got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{field} must be >= 1; got {value}")
    return value


def _is_sector_contract_sentinel(row: Mapping[str, str]) -> bool:
    return (
        str(row.get("batch_id") or "").strip() == SECTOR_CONTRACT_SENTINEL_BATCH_ID
        and str(row.get("sentinel_contract_id") or "").strip()
        == SECTOR_CONTRACT_SENTINEL_BATCH_ID
    )


def _validate_sector_contract_sentinel(row: Mapping[str, str]) -> None:
    problems: list[str] = []
    expected = {
        "suite_profile": SCALING_PROFILE,
        "run_class": "diagnostic",
        "runnable": "true",
        "method_key": "snake",
        "algorithm_id": METHOD_ALGORITHMS["snake"],
        "adapt_optimizer_kind": "powell",
        "optimizer": "POWELL",
        "budget": "200",
        "phase3_adapt_maxiter": "200",
        "phase3_refit_maxiter": "200",
        "phase3_final_maxiter": "200",
        "pool_contract": "full_meta_execution_sector_filtered",
        "child_policy": "macro_only",
        "generic_adapt_runtime_split_mode": "off",
        "snake_phase3_runtime_split_mode": "off",
        "shared_pauli_pool_mode": "off",
        "phase2_batching": "off",
        "phase3_batching": "off",
        "one_accepted_parent_per_outer_iteration": "true",
        "adapt_allow_repeats": "true",
        "phase1_pruning": "off",
        "structural_rollback": "off",
        "cost_steering": "off",
        "state_sector_audit_required": "true",
        "generator_execution_sector_audit_required": "true",
        "strict_replay_required": "true",
        "optimizer_coordinate_contract": "logical_shared",
        "adapt_parallel_gradient_workers": "1",
        "adapt_beam_parent_workers": "1",
        "phase3_adapt_parallel_gradient_workers": "1",
        "phase3_adapt_beam_parent_workers": "1",
        "exact_fidelity_max_qubits": "10",
        "resource_qubit_cap": "16",
        "resource_pool_term_cap": "1024",
    }
    for field, value in expected.items():
        if str(row.get(field) or "").strip() != value:
            problems.append(f"{field}={row.get(field)!r}, expected {value!r}")
    horizon = _positive_int(row, "expected_horizon")
    if horizon not in {8, 10}:
        problems.append(f"expected_horizon={horizon!r}, expected one of 8 or 10")
    for field in ("max_depth", "phase3_adapt_max_depth"):
        if str(row.get(field) or "").strip() != str(horizon):
            problems.append(f"{field}={row.get(field)!r}, expected {horizon!r}")
    for field in (
        "request_cpus",
        "request_memory_mb",
        "request_disk_mb",
        "adapt_parallel_gradient_workers",
        "adapt_beam_parent_workers",
        "phase3_adapt_parallel_gradient_workers",
        "phase3_adapt_beam_parent_workers",
    ):
        _positive_int(row, field)
    if str(row.get("request_cpus") or "").strip() != "4":
        problems.append("sector sentinels require four requested CPUs")
    policy = str(row.get("phase3_policy_json") or "").strip()
    policy_path = Path(policy) if Path(policy).is_absolute() else ROOT / policy
    if not policy or not policy_path.is_file():
        problems.append(f"phase3_policy_json missing: {policy_path}")
    if problems:
        raise ValueError("Sector-contract sentinel record failed: " + "; ".join(problems))


def validate_record(row: Mapping[str, str]) -> None:
    if _is_sector_contract_sentinel(row):
        _validate_sector_contract_sentinel(row)
        return
    method = str(row.get("method_key") or "").strip()
    expected_algorithm = METHOD_ALGORITHMS.get(method)
    problems: list[str] = []
    if expected_algorithm is None:
        problems.append(f"unknown method_key={method!r}")
    elif str(row.get("algorithm_id") or "").strip() != expected_algorithm:
        problems.append(f"algorithm_id={row.get('algorithm_id')!r}, expected {expected_algorithm!r}")
    expected = {
        "suite_profile": SCALING_PROFILE,
        "run_class": "candidate",
        "runnable": "true",
        "adapt_optimizer_kind": "powell",
        "optimizer": "POWELL",
        "budget": "200",
        "phase3_adapt_maxiter": "200",
        "phase3_refit_maxiter": "200",
        "phase3_final_maxiter": "200",
        "pool_contract": "full_meta_unfiltered",
        "child_policy": "macro_only",
        "generic_adapt_runtime_split_mode": "off",
        "snake_phase3_runtime_split_mode": "off",
        "shared_pauli_pool_mode": "off",
        "phase2_batching": "off",
        "phase3_batching": "off",
        "one_accepted_parent_per_outer_iteration": "true",
        "exact_fidelity_max_qubits": "10",
        "resource_qubit_cap": "16",
        "resource_pool_term_cap": "1024",
    }
    for field, value in expected.items():
        if str(row.get(field) or "").strip() != value:
            problems.append(f"{field}={row.get(field)!r}, expected {value!r}")
    horizon = _positive_int(row, "expected_horizon")
    for field in ("max_depth", "phase3_adapt_max_depth"):
        if str(row.get(field) or "").strip() != str(horizon):
            problems.append(f"{field}={row.get(field)!r}, expected {horizon!r}")
    if method == "snake":
        if str(row.get("adapt_allow_repeats") or "").strip() != "true":
            problems.append("SNAKE scaling rows require operator reuse")
        policy = str(row.get("phase3_policy_json") or "").strip()
        policy_path = Path(policy) if Path(policy).is_absolute() else ROOT / policy
        if not policy or not policy_path.is_file():
            problems.append(f"phase3_policy_json missing: {policy_path}")
        if str(row.get("generic_adapt_stop_policy") or "").strip():
            problems.append("SNAKE must not carry the generic comparator stop-policy field")
        if str(row.get("request_cpus") or "").strip() != "4":
            problems.append("SNAKE scaling rows require four requested CPUs")
        if str(row.get("adapt_parallel_gradient_workers") or "").strip() != "2":
            problems.append("SNAKE scaling rows require two parallel gradient workers")
        if str(row.get("adapt_beam_parent_workers") or "").strip() != "2":
            problems.append("SNAKE scaling rows require two beam-parent workers")
    else:
        if str(row.get("adapt_allow_repeats") or "").strip() != "true":
            problems.append("Geo/Append scaling rows require full-pool selection with replacement")
        if str(row.get("generic_adapt_stop_policy") or "").strip() != "fixed_horizon_no_target_v1":
            problems.append("Geo/Append require fixed_horizon_no_target_v1")
        if str(row.get("request_cpus") or "").strip() != "1":
            problems.append("Geo/Append scaling rows require one requested CPU")
    powell_cap_policy = str(row.get("powell_maxiter_cap_policy") or "").strip()
    if powell_cap_policy:
        if powell_cap_policy != POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING:
            problems.append(
                "unsupported powell_maxiter_cap_policy="
                f"{powell_cap_policy!r}"
            )
        if method != "append":
            problems.append(
                "Powell maxiter cap acceptance is restricted to append-only repair rows"
            )
        if str(row.get("generic_adapt_stop_policy") or "").strip() != "fixed_horizon_no_target_v1":
            problems.append(
                "Powell maxiter cap acceptance requires fixed_horizon_no_target_v1"
            )
    if str(row.get("family") or "").strip() == "hh":
        cache = str(row.get("hh_pool_cache_dir") or "").strip()
        cache_path = Path(cache) if Path(cache).is_absolute() else ROOT / cache
        if not cache or not cache_path.is_dir():
            problems.append(f"HH pool cache directory missing: {cache_path}")
        if str(row.get("hh_pool_cache_scope") or "").strip() != "exact":
            problems.append("HH scaling rows require exact cache scope")
        registry_cache = str(row.get("hh_generator_registry_cache_dir") or "").strip()
        registry_cache_path = (
            Path(registry_cache) if Path(registry_cache).is_absolute() else ROOT / registry_cache
        )
        if not registry_cache or not registry_cache_path.is_dir():
            problems.append(f"HH generator-registry cache directory missing: {registry_cache_path}")
        if str(row.get("hh_generator_registry_cache_mode") or "").strip() != "disk":
            problems.append("HH scaling rows require disk generator-registry cache mode")
    if problems:
        raise ValueError("Scaling record contract failed: " + "; ".join(problems))


def build_environment(row: Mapping[str, str], output_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return the subprocess environment and its explicit scaling overlay."""

    method = str(row["method_key"])
    overlay = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "HOLSTEIN_SKIP_MATPLOTLIB_IMPORT": "1",
        "TABLE_I_STATIC_SUITE_PROFILE": str(row["suite_profile"]),
        "GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH": str(row["expected_horizon"]),
        "GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAXITER": str(row["phase3_adapt_maxiter"]),
        "GENERIC_STATIC_TABLE_PHASE3_REFIT_MAXITER": str(row["phase3_refit_maxiter"]),
        "GENERIC_STATIC_TABLE_PHASE3_FINAL_MAXITER": str(row["phase3_final_maxiter"]),
        "GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET": "",
        "GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS": "0.0002,0.001,0.01",
        "GENERIC_STATIC_TABLE_PRIMARY_ENERGY_METRIC": "same_cutoff_abs_delta_e",
        "GENERIC_STATIC_TABLE_SAME_CUTOFF_ERROR_ROLE": "primary",
        "GENERIC_STATIC_TABLE_SAME_CUTOFF_EXACT_GS_ENERGY": str(row["same_cutoff_exact_gs_energy"]),
        "GENERIC_STATIC_TABLE_EXACT_REFERENCE_ENERGY": str(row["exact_reference_energy"]),
        "GENERIC_STATIC_TABLE_PROGRESS_JSONL_PATH": str(output_root / "adapt_iteration_progress.jsonl"),
        "GENERIC_STATIC_TABLE_PROGRESS_STDOUT": "1",
        "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE": "off",
        "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY": "off",
        "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE": "1",
        "GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS": str(row["exact_fidelity_max_qubits"]),
        "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP": str(row["resource_qubit_cap"]),
        "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP": str(row["resource_pool_term_cap"]),
        "STATIC_ADAPT_SUPPRESS_DROP_PLATEAU_TERMINAL_STOP": "1",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": str(
            ROOT / "tmp" / "paper_i_scaling_matrix" / str(row["record_id"]) / "adapt_candidate_record_cache_v1"
        ),
    }
    exact_reference_n_ph_max = str(row.get("exact_reference_n_ph_max") or "").strip()
    if exact_reference_n_ph_max:
        overlay["GENERIC_STATIC_TABLE_EXACT_REFERENCE_N_PH_MAX"] = exact_reference_n_ph_max
    if method == "snake":
        policy = Path(str(row["phase3_policy_json"]))
        if not policy.is_absolute():
            policy = ROOT / policy
        overlay.update(
            {
                "PHASE3_POLICY_INNER_OPTIMIZER": "POWELL",
                "GENERIC_STATIC_TABLE_PHASE3_POLICY_JSON": str(policy),
                "PHASE3_POLICY_PHASE2_NOVELTY_MODE": "collective_span_v1",
                "GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS": "true",
                "GENERIC_STATIC_TABLE_PHASE3_ADAPT_PARALLEL_GRADIENT_WORKERS": str(
                    row["phase3_adapt_parallel_gradient_workers"]
                ),
                "GENERIC_STATIC_TABLE_PHASE3_ADAPT_BEAM_PARENT_WORKERS": str(
                    row["phase3_adapt_beam_parent_workers"]
                ),
            }
        )
    else:
        overlay.update(
            {
                "GENERIC_STATIC_TABLE_ADAPT_OPTIMIZER_KIND": "powell",
                "GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY": "fixed_horizon_no_target_v1",
                "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE": "off",
                "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY": "off",
                "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE": "1",
                "GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS": "true",
            }
        )
        powell_cap_policy = str(row.get("powell_maxiter_cap_policy") or "").strip()
        if powell_cap_policy:
            overlay["GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY"] = powell_cap_policy
    if str(row.get("family") or "") == "hh":
        cache = Path(str(row["hh_pool_cache_dir"]))
        if not cache.is_absolute():
            cache = ROOT / cache
        registry_cache = Path(str(row["hh_generator_registry_cache_dir"]))
        if not registry_cache.is_absolute():
            registry_cache = ROOT / registry_cache
        overlay.update(
            {
                "STATIC_ADAPT_HH_POOL_CACHE": "disk",
                "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
                "STATIC_ADAPT_HH_POOL_CACHE_DIR": str(cache),
                "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE": "disk",
                "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": str(registry_cache),
                "GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE": "full_meta_unfiltered",
                "GENERIC_STATIC_TABLE_HH_FULL_META_CLASS_FILTER_JSON": "off",
            }
        )
    env = dict(os.environ)
    env.update(overlay)
    return env, overlay


def build_command(row: Mapping[str, str], output_root: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "pipelines.exact_bench.generic_static_benchmark",
        "--run-single",
        "--family",
        str(row["family"]),
        "--case-id",
        str(row["case_id"]),
        "--algorithm-id",
        str(row["algorithm_id"]),
        "--output-dir",
        str(output_root / "result"),
    ]


def _pump(stream: Any, target: Any, mirror: Any) -> None:
    try:
        for line in iter(stream.readline, ""):
            target.write(line)
            target.flush()
            mirror.write(line)
            mirror.flush()
    finally:
        stream.close()


def run_subprocess(command: Sequence[str], env: Mapping[str, str], output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "stdout.log").open("w", encoding="utf-8") as stdout_file, (
        output_root / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        threads = (
            threading.Thread(target=_pump, args=(proc.stdout, stdout_file, sys.stdout), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, stderr_file, sys.stderr), daemon=True),
        )
        for thread in threads:
            thread.start()
        returncode = int(proc.wait())
        for thread in threads:
            thread.join()
    return returncode


def _first_value(payload: Any, keys: Sequence[str]) -> Any:
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload and payload[key] is not None and payload[key] != "":
                return payload[key]
        for value in payload.values():
            found = _first_value(value, keys)
            if found is not None and found != "":
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_value(value, keys)
            if found is not None and found != "":
                return found
    return None


def result_summary(result_path: Path) -> dict[str, Any]:
    if not result_path.is_file():
        return {"status": "missing", "result_json": str(result_path)}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "status": payload.get("status"),
        "result_json": str(result_path),
        "abs_delta_e_same_cutoff": _first_value(
            payload,
            ("abs_delta_e_same_cutoff", "same_cutoff_abs_delta_e", "abs_delta_e"),
        ),
        "adapt_iteration": _first_value(
            payload,
            ("adapt_num_iterations", "adapt_depth_reached", "ansatz_depth", "depth"),
        ),
        "optimizer": _first_value(payload, ("adapt_inner_optimizer", "adapt_optimizer_kind", "optimizer")),
        "pool": _first_value(payload, ("base_pool_name", "adapt_pool", "resolved_pool_key")),
    }


def run_cell(record_id: str, records_path: Path, output_root: Path) -> int:
    row = load_record(record_id, records_path)
    validate_record(row)
    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    command = build_command(row, output_root)
    env, overlay = build_environment(row, output_root)
    _write_json(output_root / "effective_command.json", {"command": command})
    _write_json(output_root / "effective_env_overlay.json", overlay)
    try:
        returncode = run_subprocess(command, env, output_root)
        result_path = output_root / "result" / "generic_static_single.json"
        summary = result_summary(result_path)
        if returncode == 0 and summary.get("status") not in {"completed", "ok"}:
            returncode = 3
        manifest = {
            "schema": "paper_i_scaling_matrix_cell_manifest_v1",
            "record_id": record_id,
            "status": "ok" if returncode == 0 else "failed",
            "returncode": int(returncode),
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "env_overlay": overlay,
            "result_summary": summary,
            "row": dict(row),
        }
        _write_json(output_root / "cell_manifest.json", manifest)
        return int(returncode)
    except Exception as exc:
        _write_json(
            output_root / "cell_manifest.json",
            {
                "schema": "paper_i_scaling_matrix_cell_manifest_v1",
                "record_id": record_id,
                "status": "runner_exception",
                "exception": repr(exc),
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "row": dict(row),
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_id")
    parser.add_argument("records_path", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args(argv)
    return run_cell(str(args.record_id), Path(args.records_path), Path(args.output_root))


if __name__ == "__main__":
    raise SystemExit(main())
