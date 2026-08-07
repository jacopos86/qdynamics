#!/usr/bin/env python3
"""Generate the user-locked Paper-I finite-size scaling CHTC matrix.

The matrix is deliberately expressed as ordered ``(L, n_ph_max)`` pairs.  It
contains exactly three parent-generator methods for every physics case:
SNAKE, Geo-ADAPT, and append-only ADAPT.  This generator prepares records and
a Condor submit descriptor only; it never submits or mutates a live batch.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.exact_bench.table_i_canonical_cases import (  # noqa: E402
    TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
    table_i_executable_specs,
)


DEFAULT_BATCH_ID = "paper_i_scaling_matrix_parent_powell200_20260710_v1"
DEFAULT_OUTPUT_DIR = ROOT / "chtc" / "phase3_optuna" / "input" / DEFAULT_BATCH_ID
DEFAULT_SUBMIT_PATH = ROOT / "chtc" / "phase3_optuna" / f"submit_{DEFAULT_BATCH_ID}.sub"
DEFAULT_SPIN_BOSON_HORIZON = 30
DEFAULT_BOSE_HUBBARD_HORIZON = 30
POWELL_MAXITER = 200
EXACT_FIDELITY_MAX_QUBITS = 10
RESOURCE_QUBIT_CAP = 16
RESOURCE_POOL_TERM_CAP = 1024

METHODS = (
    ("snake", "SNAKE", "static_family_native_adapt_phase3"),
    ("geo", "Geo-ADAPT", "static_geo_adapt_vqe"),
    ("append", "Append-ADAPT", "static_full_meta_append_adapt_vqe"),
)

IMPLEMENTATION_FILES = (
    "chtc/phase3_optuna/generate_paper_i_scaling_matrix_records.py",
    "chtc/phase3_optuna/prepare_paper_i_snake_sector_contract_sentinels.py",
    "chtc/phase3_optuna/preflight_paper_i_snake_sector_contract_sentinels.py",
    "chtc/phase3_optuna/prepare_paper_i_scaling_snake_overlay_repair.py",
    "pipelines/exact_bench/table_i_canonical_cases.py",
    "pipelines/exact_bench/static_reference_metrics.py",
    "pipelines/exact_bench/generic_static_benchmark.py",
    "pipelines/exact_bench/generic_static_adapt_variants.py",
    "pipelines/static_adapt/optimization/phase3_policy_optuna.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/cli_config.py",
    "pipelines/static_adapt/engine_support.py",
    "pipelines/static_adapt/joint_step_warm_start.py",
    "pipelines/static_adapt/paper_i_runner.py",
    "pipelines/static_adapt/resume_scaffold.py",
    "pipelines/static_adapt/sector_invariants.py",
    "src/quantum/compiled_ansatz.py",
    "chtc/phase3_optuna/run_paper_i_scaling_matrix_cell.py",
    "chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh",
    "chtc/phase3_optuna/prewarm_paper_i_scaling_hh_pool_cache.py",
    "chtc/phase3_optuna/preflight_submit.py",
)

CRITICAL_BUNDLE_MEMBERS = frozenset(
    {
        "pipelines/exact_bench/table_i_canonical_cases.py",
        "pipelines/exact_bench/static_reference_metrics.py",
        "pipelines/exact_bench/generic_static_benchmark.py",
        "pipelines/exact_bench/generic_static_adapt_variants.py",
        "pipelines/static_adapt/optimization/phase3_policy_optuna.py",
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/cli_config.py",
        "pipelines/static_adapt/engine_support.py",
        "pipelines/static_adapt/joint_step_warm_start.py",
        "pipelines/static_adapt/paper_i_runner.py",
        "pipelines/static_adapt/resume_scaffold.py",
        "pipelines/static_adapt/sector_invariants.py",
        "src/quantum/compiled_ansatz.py",
        "chtc/phase3_optuna/run_paper_i_scaling_matrix_cell.py",
    }
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _bundle_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
        return None
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_code_bundle(output_dir: Path) -> dict[str, Any]:
    """Package the exact corrected source without touching shared remote trees."""

    bundle_path = output_dir / "paper_i_scaling_matrix_code.tar.gz"
    sources = (
        ROOT / "pipelines",
        ROOT / "src",
        ROOT / "docs",
        ROOT / "chtc" / "phase3_optuna" / "run_paper_i_scaling_matrix_cell.py",
    )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Scaling code-bundle source is missing: {missing}")
    with bundle_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for init_name in ("chtc/__init__.py", "chtc/phase3_optuna/__init__.py"):
                    data = b'"""Batch-local CHTC package."""\n'
                    info = tarfile.TarInfo(init_name)
                    info.size = len(data)
                    info.mode = 0o644
                    info = _bundle_filter(info)
                    assert info is not None
                    archive.addfile(info, io.BytesIO(data))
                for source in sources:
                    archive.add(
                        source,
                        arcname=str(source.relative_to(ROOT)),
                        recursive=True,
                        filter=_bundle_filter,
                    )
    return {
        "path": _repo_path(bundle_path),
        "sha256": _sha256(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "members": [
            "chtc/__init__.py",
            "chtc/phase3_optuna/__init__.py",
            *[str(path.relative_to(ROOT)) for path in sources],
        ],
        "excludes": ["__pycache__", "*.pyc", "*.pyo"],
    }


def _write_implementation_lock(output_dir: Path, code_bundle: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    """Bind critical local implementation hashes to the exact code tar members."""

    bundle_path = ROOT / str(code_bundle["path"])
    bundle_member_hashes: dict[str, str] = {}
    with tarfile.open(bundle_path, "r:gz") as archive:
        names = set(archive.getnames())
        missing = sorted(CRITICAL_BUNDLE_MEMBERS - names)
        if missing:
            raise FileNotFoundError(f"Critical implementation files are missing from the code bundle: {missing}")
        for rel in sorted(CRITICAL_BUNDLE_MEMBERS):
            extracted = archive.extractfile(rel)
            if extracted is None:
                raise FileNotFoundError(f"Could not read critical code-bundle member: {rel}")
            bundle_member_hashes[rel] = hashlib.sha256(extracted.read()).hexdigest()

    entries: list[dict[str, Any]] = []
    for rel in IMPLEMENTATION_FILES:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Implementation-lock file is missing: {path}")
        local_sha = _sha256(path)
        bundled_sha = bundle_member_hashes.get(rel)
        if rel in CRITICAL_BUNDLE_MEMBERS and bundled_sha != local_sha:
            raise RuntimeError(
                f"Critical code-bundle member differs from the local implementation: {rel} "
                f"bundle={bundled_sha} local={local_sha}"
            )
        entries.append(
            {
                "path": rel,
                "sha256": local_sha,
                "size_bytes": path.stat().st_size,
                "critical_bundle_member": rel in CRITICAL_BUNDLE_MEMBERS,
                "bundle_member_sha256": bundled_sha,
            }
        )
    payload = {
        "schema": "paper_i_scaling_matrix_implementation_lock_v1",
        "code_bundle": {
            "path": str(code_bundle["path"]),
            "sha256": str(code_bundle["sha256"]),
        },
        "critical_bundle_member_count": len(CRITICAL_BUNDLE_MEMBERS),
        "entries": entries,
        "status": "pass",
    }
    path = output_dir / "implementation_lock.json"
    _write_json(path, payload)
    return payload, path


def _write_exact_energy_manifest(
    specs: Sequence[Any],
    *,
    output_dir: Path,
    resolver: Any | None = None,
) -> tuple[dict[str, Any], Path]:
    if resolver is None:
        from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec

        resolver = exact_energy_for_spec
    records: dict[str, dict[str, Any]] = {}
    for spec in specs:
        is_bosonic = bool(getattr(getattr(spec, "features", None), "bosonic", False))
        n_ph_raw = _arg_value(spec, "--n-ph-max") if is_bosonic else None
        resolver_n_ph_max = 1 if n_ph_raw in {None, ""} else int(str(n_ph_raw))
        energy, key_hash, key = resolver(spec, n_ph_max=resolver_n_ph_max)
        records[str(spec.benchmark_id)] = {
            "case_id": str(spec.benchmark_id),
            "family": str(spec.family),
            "n_ph_work": None if n_ph_raw in {None, ""} else int(str(n_ph_raw)),
            "n_ph_applicability": "physical_cutoff" if is_bosonic else "not_applicable_nonbosonic",
            "resolver_n_ph_max": int(resolver_n_ph_max) if is_bosonic else None,
            "compatibility_call_n_ph_max": None if is_bosonic else int(resolver_n_ph_max),
            "exact_energy": float(energy),
            "key_hash": str(key_hash),
            "key": dict(key),
            "method": "same_cutoff_exact_diagonalization",
            "source": "pipelines.exact_bench.static_reference_metrics.exact_energy_for_spec",
            "status": "ok",
        }
    payload = {
        "schema": "paper_i_scaling_matrix_exact_energy_manifest_v1",
        "suite_profile": TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
        "record_count": len(records),
        "reference_policy": "same_working_cutoff",
        "records": records,
        "status": "pass",
    }
    path = output_dir / "exact_energy_manifest.json"
    _write_json(path, payload)
    return payload, path


def _arg_value(spec: Any, flag: str) -> str | None:
    args = tuple(str(value) for value in spec.base_pipeline_args)
    try:
        index = args.index(flag)
    except ValueError:
        return None
    return args[index + 1] if index + 1 < len(args) else None


def _positive_horizon(value: int, *, name: str) -> int:
    horizon = int(value)
    if horizon < 1:
        raise ValueError(f"{name} must be >= 1; got {horizon}")
    return horizon


def _horizon_for_spec(
    spec: Any,
    *,
    spin_boson_horizon: int,
    bose_hubbard_horizon: int,
) -> tuple[int, str]:
    family = str(spec.family)
    L = int(str(_arg_value(spec, "--L")))
    if family == "hh":
        return 50, "locked:hh_outer_iteration_50"
    if family == "hubbard":
        return (20, "locked:hubbard_L2_outer_iteration_20") if L == 2 else (
            30,
            "locked:hubbard_L3_L4_outer_iteration_30",
        )
    if family == "spin_boson":
        return _positive_horizon(spin_boson_horizon, name="spin_boson_horizon"), (
            "cli:spin_boson_horizon"
        )
    if family == "bose_hubbard":
        return _positive_horizon(bose_hubbard_horizon, name="bose_hubbard_horizon"), (
            "cli:bose_hubbard_horizon"
        )
    raise ValueError(f"Unexpected scaling-matrix family: {family!r}")


def _display_regime(case_id: str, family: str) -> str:
    suffix = str(case_id).rsplit("_", 1)[-1]
    if family == "hh":
        marker = "_scaling_"
        if marker not in case_id:
            raise ValueError(f"HH scaling case lacks {marker!r}: {case_id}")
        suffix = case_id.split(marker, 1)[1]
    return suffix.replace("_", "-")


def _resource_request(family: str) -> tuple[int, int, str]:
    if family == "hh":
        return 32768, 61440, "hh_12q_high_memory"
    return 16384, 32768, "appendix_standard"


def _snake_policy_payload() -> dict[str, Any]:
    """Return the shared parent-only, no-batching SNAKE policy."""

    return {
        "schema": "paper_i_scaling_matrix_snake_policy_v1",
        "pool": {
            "pool_key": "full_meta",
            "family_repeat_penalty": 0.0,
            "novelty_bonus": 0.0,
        },
        "static": {
            "static_meta_feature_profile": "paper_i_production_v1",
            "static_route_id": "route_a",
            "static_lane_route": "physical_operator_type",
            "physical_lane_shortlist_aggressiveness": 3,
            "adapt_max_depth": 50,
            "adapt_maxiter": POWELL_MAXITER,
            "adapt_drop_floor": -1.0,
            "adapt_drop_patience": 0,
            "adapt_drop_min_depth": 0,
            "adapt_eps_grad": 0.0,
            "adapt_eps_energy": 0.0,
            "adapt_reopt_policy": "full",
            "adapt_window_size": 99,
            "adapt_window_topk": 0,
            "adapt_full_refit_every": 1,
            "adapt_final_full_refit": True,
            "adapt_final_refit_maxiter": POWELL_MAXITER,
            "adapt_insertion_mode": "full_commutation_reduced",
            "adapt_allow_repeats": True,
            "adapt_parallel_gradient_workers": 2,
            "adapt_beam_parent_workers": 2,
            "adapt_beam_live_branches": 3,
            "adapt_beam_children_per_parent": 2,
            "adapt_beam_terminated_keep": 3,
            "adapt_beam_lambda": 0.005,
            "phase1_probe_max_positions": 999999,
            "phase2_enable_batching": False,
            "phase3_enable_batching": False,
            "phase3_runtime_split_mode": "off",
            "allow_archival_phase3_runtime_split": False,
            "shared_pauli_pool_mode": "off",
            "shared_pauli_pool_symmetry_policy": "off",
            "shared_pauli_pool_max_subset_size": 1,
            "phase_live_hysteresis_enabled": False,
            "phase1_prune_amplitude_witness_required": False,
            "compile_position_shift_weight": 0.0,
        },
        "inner_optimizer": {
            "inner_optimizer": "POWELL",
            "final_optimizer_type": "POWELL",
            "refit_maxiter": POWELL_MAXITER,
            "final_maxiter": POWELL_MAXITER,
            "grad_tol": 1.0e-9,
            "energy_tol": 1.0e-13,
        },
    }


def _ensure_fresh_targets(output_dir: Path, submit_path: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty; choose a new batch id or pass --force: {output_dir}")
    if submit_path.exists() and not force:
        raise FileExistsError(f"Submit descriptor already exists; choose a new path or pass --force: {submit_path}")
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    if force and submit_path.exists():
        submit_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    submit_path.parent.mkdir(parents=True, exist_ok=True)


def _write_submit(
    *,
    submit_path: Path,
    batch_id: str,
    records_path: Path,
    queue_path: Path,
    output_dir: Path,
    job_batch_name: str = "paper-i-scaling-parent-powell200",
    stream_output: bool = True,
    stream_error: bool = True,
) -> None:
    records_rel = _repo_path(records_path)
    queue_rel = _repo_path(queue_path)
    input_rel = _repo_path(output_dir)
    text = f"""universe = vanilla
executable = chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh
arguments = $(record_id) {records_rel} raw_outputs/{batch_id}/$(record_id)

should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = chtc/phase3_optuna/image.sif, chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh, {input_rel}
transfer_output_files = raw_outputs/{batch_id}/$(record_id)

stream_output = {str(bool(stream_output))}
stream_error = {str(bool(stream_error))}
log = logs/{batch_id}.$(Cluster).$(Process).log
output = logs/{batch_id}.$(Cluster).$(Process).out
error = logs/{batch_id}.$(Cluster).$(Process).err

requirements = TARGET.HasSIF
request_cpus = $(cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = \"{job_batch_name}\"
notification = Never

queue record_id, cpus, memory_mb, disk_mb from {queue_rel}
"""
    submit_path.write_text(text, encoding="utf-8")


def generate(
    *,
    batch_id: str = DEFAULT_BATCH_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    submit_path: Path = DEFAULT_SUBMIT_PATH,
    spin_boson_horizon: int = DEFAULT_SPIN_BOSON_HORIZON,
    bose_hubbard_horizon: int = DEFAULT_BOSE_HUBBARD_HORIZON,
    force: bool = False,
    exact_energy_resolver: Any | None = None,
) -> dict[str, Any]:
    batch_id = str(batch_id).strip()
    if not batch_id.startswith("paper_i_scaling_matrix_"):
        raise ValueError("Scaling batches must use the fail-closed prefix 'paper_i_scaling_matrix_'.")
    output_dir = Path(output_dir).expanduser().resolve()
    submit_path = Path(submit_path).expanduser().resolve()
    _ensure_fresh_targets(output_dir, submit_path, force=bool(force))

    spin_horizon = _positive_horizon(spin_boson_horizon, name="spin_boson_horizon")
    bose_horizon = _positive_horizon(bose_hubbard_horizon, name="bose_hubbard_horizon")
    specs = table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
    if len(specs) != 34:
        raise ValueError(f"Expected 34 scaling specs, got {len(specs)}")
    exact_energy_payload, exact_energy_manifest = _write_exact_energy_manifest(
        specs,
        output_dir=output_dir,
        resolver=exact_energy_resolver,
    )
    exact_energy_manifest_sha256 = _sha256(exact_energy_manifest)

    policy_path = output_dir / "paper_i_scaling_matrix_snake_policy.json"
    _write_json(policy_path, _snake_policy_payload())
    policy_sha256 = _sha256(policy_path)
    code_bundle = _write_code_bundle(output_dir)
    implementation_lock, implementation_lock_path = _write_implementation_lock(output_dir, code_bundle)
    implementation_lock_sha256 = _sha256(implementation_lock_path)
    hh_pool_cache_dir = output_dir / "hh_pool_cache_v1"
    hh_pool_cache_dir.mkdir(parents=True, exist_ok=True)
    hh_generator_registry_cache_dir = output_dir / "hh_generator_registry_cache_v1"
    hh_generator_registry_cache_dir.mkdir(parents=True, exist_ok=True)
    hh_pool_cache_manifest = output_dir / "hh_cache_prewarm_manifest.json"
    _write_json(
        hh_pool_cache_manifest,
        {
            "schema": "paper_i_scaling_matrix_hh_dual_cache_prewarm_v1",
            "status": "pending",
            "pool_cache": {
                "mode": "disk",
                "scope": "exact",
                "cache_dir": _repo_path(hh_pool_cache_dir),
            },
            "generator_registry_cache": {
                "mode": "disk",
                "cache_dir": _repo_path(hh_generator_registry_cache_dir),
            },
            "required_case_count": 12,
            "prewarm_command": (
                "python3 chtc/phase3_optuna/prewarm_paper_i_scaling_hh_pool_cache.py "
                f"--cache-dir {_repo_path(hh_pool_cache_dir)} "
                f"--generator-registry-cache-dir {_repo_path(hh_generator_registry_cache_dir)} "
                f"--manifest {_repo_path(hh_pool_cache_manifest)}"
            ),
        },
    )

    rows: list[dict[str, str]] = []
    for spec in specs:
        family = str(spec.family)
        case_id = str(spec.benchmark_id)
        L = int(str(_arg_value(spec, "--L")))
        n_ph_raw = (
            _arg_value(spec, "--n-ph-max")
            if bool(getattr(getattr(spec, "features", None), "bosonic", False))
            else None
        )
        n_ph_work = None if n_ph_raw in {None, ""} else int(str(n_ph_raw))
        horizon, horizon_source = _horizon_for_spec(
            spec,
            spin_boson_horizon=spin_horizon,
            bose_hubbard_horizon=bose_horizon,
        )
        memory_mb, disk_mb, resource_tier = _resource_request(family)
        regime = _display_regime(case_id, family)
        exact_record = dict(exact_energy_payload["records"][case_id])
        for method_key, method_label, algorithm_id in METHODS:
            record_id = (
                f"{batch_id}__{family}__{case_id}__{method_key}__"
                f"parent_powell200__iter{horizon}"
            )
            record_output_dir = f"raw_outputs/{batch_id}/{record_id}"
            generic_stop = "" if method_key == "snake" else "fixed_horizon_no_target_v1"
            request_cpus = 4 if method_key == "snake" else 1
            snake_gradient_workers = "2" if method_key == "snake" else "not_applicable"
            snake_beam_parent_workers = "2" if method_key == "snake" else "not_applicable"
            row = {
                "record_id": record_id,
                "batch_id": batch_id,
                "run_class": "candidate",
                "runnable": "true",
                "blocker": "",
                "family": family,
                "case_id": case_id,
                "algorithm_id": algorithm_id,
                "method_key": method_key,
                "method_label": method_label,
                "display_regime": regime,
                "suite_profile": TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
                "L": str(L),
                "n_ph_work": "" if n_ph_work is None else str(n_ph_work),
                "n_ph_ref": "" if n_ph_work is None else str(n_ph_work),
                "exact_reference_n_ph_max": "" if n_ph_work is None else str(n_ph_work),
                "same_cutoff_exact_gs_energy": repr(float(exact_record["exact_energy"])),
                "exact_reference_energy": repr(float(exact_record["exact_energy"])),
                "exact_energy_key": str(exact_record["key_hash"]),
                "exact_energy_manifest": _repo_path(exact_energy_manifest),
                "exact_energy_manifest_sha256": exact_energy_manifest_sha256,
                "exact_energy_method": str(exact_record["method"]),
                "exact_energy_source": str(exact_record["source"]),
                "primary_energy_metric": "same_cutoff_abs_delta_e",
                "same_cutoff_error_role": "primary",
                "optimizer": "POWELL",
                "adapt_optimizer_kind": "powell",
                "budget": str(POWELL_MAXITER),
                "phase3_adapt_maxiter": str(POWELL_MAXITER),
                "phase3_refit_maxiter": str(POWELL_MAXITER),
                "phase3_final_maxiter": str(POWELL_MAXITER),
                "max_depth": str(horizon),
                "phase3_adapt_max_depth": str(horizon),
                "expected_horizon": str(horizon),
                "horizon_source": horizon_source,
                "selector_horizon_semantics": "outer_selector_iterations_not_guaranteed_ansatz_depth",
                "snake_progress_reference_transport": (
                    "adapt_pipeline_internal_exact_gs_same_numeric"
                    if method_key == "snake"
                    else "generic_static_exact_energy_env"
                ),
                "generic_adapt_stop_policy": generic_stop,
                "snake_fixed_horizon_no_target": "true" if method_key == "snake" else "not_applicable",
                "pool_contract": "full_meta_unfiltered",
                "hh_adaptive_pool_profile": "full_meta_unfiltered" if family == "hh" else "not_applicable",
                "adapt_pool_class_filter_json": "off",
                "hh_pool_cache_mode": "disk" if family == "hh" else "not_applicable",
                "hh_pool_cache_scope": "exact" if family == "hh" else "not_applicable",
                "hh_pool_cache_dir": _repo_path(hh_pool_cache_dir) if family == "hh" else "",
                "hh_pool_cache_manifest": _repo_path(hh_pool_cache_manifest) if family == "hh" else "",
                "hh_pool_cache_required": "true" if family == "hh" else "false",
                "hh_generator_registry_cache_mode": "disk" if family == "hh" else "not_applicable",
                "hh_generator_registry_cache_dir": (
                    _repo_path(hh_generator_registry_cache_dir) if family == "hh" else ""
                ),
                "hh_generator_registry_cache_required": "true" if family == "hh" else "false",
                "matrix_label": "paper_i_scaling_matrix_parent_only",
                "matrix_role": "finite_size_cutoff_scaling_parent_generator_comparison",
                "child_policy": "macro_only",
                "parent_generator_policy": "full_meta_parent_macro_generators_only_all_methods",
                "generic_adapt_runtime_split_mode": "off",
                "generic_adapt_runtime_split_symmetry_policy": "off",
                "generic_adapt_runtime_split_max_subset_size": "1",
                "snake_phase3_runtime_split_mode": "off",
                "snake_phase3_runtime_split_selection_mode": "off",
                "snake_phase3_runtime_split_child_set_symmetry_policy": "off",
                "snake_phase3_runtime_split_max_subset_size": "1",
                "shared_pauli_pool_mode": "off",
                "shared_pauli_pool_symmetry_policy": "off",
                "shared_pauli_pool_max_subset_size": "1",
                "phase2_batching": "off",
                "phase3_batching": "off",
                "one_accepted_parent_per_outer_iteration": "true",
                "adapt_allow_repeats": "true",
                "adapt_parallel_gradient_workers": snake_gradient_workers,
                "adapt_beam_parent_workers": snake_beam_parent_workers,
                "phase3_adapt_parallel_gradient_workers": "2" if method_key == "snake" else "",
                "phase3_adapt_beam_parent_workers": "2" if method_key == "snake" else "",
                "geo_immediate_repeat_policy": "block_only_adjacent_repeat_after_full_pool_selection",
                "append_selection_policy": "append_only_with_replacement",
                "exact_fidelity_max_qubits": str(EXACT_FIDELITY_MAX_QUBITS),
                "resource_qubit_cap": str(RESOURCE_QUBIT_CAP),
                "resource_pool_term_cap": str(RESOURCE_POOL_TERM_CAP),
                "request_memory_mb": str(memory_mb),
                "request_disk_mb": str(disk_mb),
                "request_cpus": str(request_cpus),
                "resource_tier": resource_tier,
                "phase3_policy_json": _repo_path(policy_path) if method_key == "snake" else "",
                "phase3_policy_json_sha256": policy_sha256 if method_key == "snake" else "",
                "source_settings_status": "user_locked_scaling_matrix_profile_no_visible_higher_size_anchor",
                "settings_reused": "physics_from_table_i_scaling_profile;Powell200;full_meta;parent_only",
                "settings_changed": "L;n_ph_work;outer_iteration_horizon;new_output_identity",
                "implementation_contract_id": "corrected_geo_append_estimator_accounting_20260710_v1",
                "code_bundle": str(code_bundle["path"]),
                "code_bundle_sha256": str(code_bundle["sha256"]),
                "implementation_lock": _repo_path(implementation_lock_path),
                "implementation_lock_sha256": implementation_lock_sha256,
                "record_output_dir": record_output_dir,
                "result_json_rel": f"{record_output_dir}/result/generic_static_single.json",
                "current_json_rel": (
                    f"{record_output_dir}/result/{case_id}/json/current.json"
                    if method_key == "snake"
                    else f"{record_output_dir}/adapt_iteration_progress.jsonl"
                ),
                "stdout_rel": f"{record_output_dir}/stdout.log",
                "stderr_rel": f"{record_output_dir}/stderr.log",
                "cell_manifest_rel": f"{record_output_dir}/cell_manifest.json",
            }
            rows.append(row)

    if len(rows) != 102:
        raise ValueError(f"Expected 102 scaling records, generated {len(rows)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Scaling record ids are not unique")

    records_path = output_dir / "paper_i_scaling_matrix_records.tsv"
    fieldnames = sorted({key for row in rows for key in row})
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    ids_path = output_dir / "paper_i_scaling_matrix_record_ids.txt"
    ids_path.write_text("\n".join(row["record_id"] for row in rows) + "\n", encoding="utf-8")
    queue_path = output_dir / "paper_i_scaling_matrix_record_queue.tsv"
    queue_path.write_text(
        "".join(
            f"{row['record_id']}\t{row['request_cpus']}\t{row['request_memory_mb']}\t{row['request_disk_mb']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    _write_submit(
        submit_path=submit_path,
        batch_id=batch_id,
        records_path=records_path,
        queue_path=queue_path,
        output_dir=output_dir,
    )

    generated_utc = datetime.now(timezone.utc).isoformat()
    audit_path = output_dir / "submission_audit.json"
    audit = {
        "schema": "paper_i_scaling_matrix_submission_audit_v1",
        "generated_utc": generated_utc,
        "batch_id": batch_id,
        "run_class": "candidate",
        "record_count": len(rows),
        "physical_case_count": len(specs),
        "method_count": len(METHODS),
        "suite_profile": TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
        "ordered_pair_policy": "explicit_pairs_not_cartesian_product",
        "horizons": {
            "hh": 50,
            "hubbard_L2": 20,
            "hubbard_L3_L4": 30,
            "spin_boson": spin_horizon,
            "bose_hubbard": bose_horizon,
        },
        "method_contract": {
            "methods": [algorithm_id for _key, _label, algorithm_id in METHODS],
            "optimizer": "POWELL",
            "optimizer_maxiter": POWELL_MAXITER,
            "final_refit_maxiter": POWELL_MAXITER,
            "pool": "full_meta_unfiltered",
            "parent_generators_only": True,
            "runtime_pauli_child_split": "off_all_methods",
            "shared_pauli_pool": "off_all_methods",
            "snake_phase2_phase3_batching": "off",
            "snake_repeat_policy": "operator_reuse_enabled",
            "snake_request_cpus": 4,
            "snake_adapt_parallel_gradient_workers": 2,
            "snake_adapt_beam_parent_workers": 2,
            "comparator_request_cpus": 1,
            "geo_immediate_repeat": "disabled_only_for_adjacent_repeat",
            "append_only": True,
        },
        "snake_policy": {"path": _repo_path(policy_path), "sha256": policy_sha256},
        "guard_caps": {
            "resource_qubit_cap": RESOURCE_QUBIT_CAP,
            "resource_pool_term_cap": RESOURCE_POOL_TERM_CAP,
            "exact_fidelity_max_qubits": EXACT_FIDELITY_MAX_QUBITS,
        },
        "hh_pool_cache": {
            "required": True,
            "mode": "disk",
            "scope": "exact",
            "cache_dir": _repo_path(hh_pool_cache_dir),
            "prewarm_manifest": _repo_path(hh_pool_cache_manifest),
            "status": "pending",
        },
        "hh_generator_registry_cache": {
            "required": True,
            "mode": "disk",
            "cache_dir": _repo_path(hh_generator_registry_cache_dir),
            "prewarm_manifest": _repo_path(hh_pool_cache_manifest),
            "status": "pending",
        },
        "reference_contract": "same_working_cutoff_exact_reference_for_all_bosonic_rows",
        "exact_energy_manifest": {
            "path": _repo_path(exact_energy_manifest),
            "sha256": exact_energy_manifest_sha256,
            "record_count": len(exact_energy_payload["records"]),
        },
        "source_lock_status": "not_applicable_user_requested_new_size_cutoff_matrix",
        "implementation_lock": {
            "path": _repo_path(implementation_lock_path),
            "sha256": implementation_lock_sha256,
            "entry_count": len(implementation_lock["entries"]),
        },
        "code_bundle": code_bundle,
        "git_head": _git_head(),
        "git_worktree_note": "Exact file hashes identify the prepared worktree, including uncommitted corrections.",
        "durability": {
            "condor_transfer_policy": "ON_EXIT_OR_EVICT",
            "per_record_output_transfer": True,
            "restartable_from_iteration_checkpoint": False,
        },
        "status": "awaiting_hh_dual_cache_prewarm",
    }
    _write_json(audit_path, audit)

    manifest = {
        "schema": "paper_i_scaling_matrix_manifest_v1",
        "generated_utc": generated_utc,
        "batch_id": batch_id,
        "run_class": "candidate",
        "suite_profile": TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
        "record_count": len(rows),
        "physical_case_count": len(specs),
        "method_count": len(METHODS),
        "record_ids": [row["record_id"] for row in rows],
        "records_path": _repo_path(records_path),
        "record_ids_path": _repo_path(ids_path),
        "record_queue_path": _repo_path(queue_path),
        "snake_policy_path": _repo_path(policy_path),
        "snake_policy_sha256": policy_sha256,
        "exact_energy_manifest": _repo_path(exact_energy_manifest),
        "exact_energy_manifest_sha256": exact_energy_manifest_sha256,
        "code_bundle": code_bundle,
        "hh_pool_cache_dir": _repo_path(hh_pool_cache_dir),
        "hh_pool_cache_manifest": _repo_path(hh_pool_cache_manifest),
        "hh_generator_registry_cache_dir": _repo_path(hh_generator_registry_cache_dir),
        "implementation_lock": _repo_path(implementation_lock_path),
        "implementation_lock_sha256": implementation_lock_sha256,
        "submit_path": _repo_path(submit_path),
        "submission_audit": _repo_path(audit_path),
        "spin_boson_horizon": spin_horizon,
        "bose_hubbard_horizon": bose_horizon,
        "status": "awaiting_hh_dual_cache_prewarm",
    }
    manifest_path = output_dir / "paper_i_scaling_matrix_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--submit-path", type=Path, default=None)
    parser.add_argument("--spin-boson-horizon", type=int, default=DEFAULT_SPIN_BOSON_HORIZON)
    parser.add_argument("--bose-hubbard-horizon", type=int, default=DEFAULT_BOSE_HUBBARD_HORIZON)
    parser.add_argument("--force", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_id = str(args.batch_id)
    output_dir = args.output_dir or ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    submit_path = args.submit_path or ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    manifest = generate(
        batch_id=batch_id,
        output_dir=Path(output_dir),
        submit_path=Path(submit_path),
        spin_boson_horizon=int(args.spin_boson_horizon),
        bose_hubbard_horizon=int(args.bose_hubbard_horizon),
        force=bool(args.force),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
