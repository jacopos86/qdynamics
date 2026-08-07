#!/usr/bin/env python3
"""Build the five-job higher-L Append projected-singleton diagnostic bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_ID = (
    "paper_i_hh_l3nph3_hubbard_l6_append_projected_singleton_"
    "r50_20260719_v1_chtc"
)
SCHEMA = "paper_i_higher_l_append_projected_singleton_bundle_v1"
SUITE_PROFILE = "paper_i_higher_l_discriminator_20260719_v1"
BASE_BUNDLE_ID = (
    "paper_i_hh_append_projected_singleton_all_six_"
    "r50_20260719_v4_chtc"
)
BASE_ARCHIVE_SHA256 = "8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd"
EXPECTED_IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
REMOTE_EXECUTION_GATE_SCHEMA = "paper_i_hh_sr_symcost_noprune_remote_execution_gate_v1"
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_QISKIT_VERSION = "2.3.1"
REMOTE_FAKE_BACKEND_RESOLVED = "fake_marrakesh"
REMOTE_FAKE_BACKEND_QUBITS = 156
# Deliberate source-visible switch. A valid remote gate is necessary but cannot
# enable submission by itself.
SUBMISSION_ENABLED = True
GENERIC_SOURCE_SHA256 = "33f0e8ffba1d532e86037077e99d8578423fc7a52842e2479a40afea1588ed3d"

REGIMES = (
    {
        "slug": "hh_L3_nph3_weak_strong",
        "label": "HH L=3 nph=3 weak-strong",
        "family": "hh",
        "case_id": "hh_L3_nph3_higher_l_weak_strong",
        "L": 3,
        "num_qubits": 12,
        "u": 0.25,
        "g_ep": 0.7905694150420949,
        "n_ph": 3,
        "n_ph_ref": 3,
        "same_cutoff_reference": True,
        "exact_energy": -1.4689835578168593,
        "reference_hash": "23b65b343cc68dcf27671356",
        "sector_particles": [2, 1],
        "sector_label": "half_filled_fermion_sector",
        "parent_pool": "full_meta_unfiltered",
        "hva_included": True,
        "parent_count": 251,
        "raw_term_count": 5554,
        "child_count": 725,
        "null_count": 13,
        "label_hash": "ab695760046ea49d2a4d185a97cd4016fcc7c614fdf64d572b1f4739b1c68288",
        "pool_hash": "636b834c5c0ef389966f9d9aaa23e1567af489409ebc088ed8bf52cbc11ee259",
        "contract_source_slug": "weak_strong",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "hh_L3_nph3_intermediate_strong",
        "label": "HH L=3 nph=3 intermediate-strong",
        "family": "hh",
        "case_id": "hh_L3_nph3_higher_l_intermediate_strong",
        "L": 3,
        "num_qubits": 12,
        "u": 1.25,
        "g_ep": 0.7905694150420949,
        "n_ph": 3,
        "n_ph_ref": 3,
        "same_cutoff_reference": True,
        "exact_energy": -0.8367272230014656,
        "reference_hash": "3f96027e3d36ac18dabb937c",
        "sector_particles": [2, 1],
        "sector_label": "half_filled_fermion_sector",
        "parent_pool": "full_meta_unfiltered",
        "hva_included": True,
        "parent_count": 251,
        "raw_term_count": 5554,
        "child_count": 725,
        "null_count": 13,
        "label_hash": "ab695760046ea49d2a4d185a97cd4016fcc7c614fdf64d572b1f4739b1c68288",
        "pool_hash": "09fa1ea018c355bad4763052077584fc607d0ca2003a9d5a63ac7a16e9ea4e5b",
        "contract_source_slug": "intermediate_strong",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "hh_L3_nph3_strong_strong",
        "label": "HH L=3 nph=3 strong-strong",
        "family": "hh",
        "case_id": "hh_L3_nph3_higher_l_strong_strong",
        "L": 3,
        "num_qubits": 12,
        "u": 8.0,
        "g_ep": 0.7905694150420949,
        "n_ph": 3,
        "n_ph_ref": 3,
        "same_cutoff_reference": True,
        "exact_energy": 0.7813398458338501,
        "reference_hash": "e157d07b035c6ed09f426577",
        "sector_particles": [2, 1],
        "sector_label": "half_filled_fermion_sector",
        "parent_pool": "full_meta_unfiltered",
        "hva_included": True,
        "parent_count": 251,
        "raw_term_count": 5554,
        "child_count": 725,
        "null_count": 13,
        "label_hash": "ab695760046ea49d2a4d185a97cd4016fcc7c614fdf64d572b1f4739b1c68288",
        "pool_hash": "9dc6df2cd84f9ad3f403a670fed497c99d201e11882fe7a6df47c18e8a3441e2",
        "contract_source_slug": "strong_strong",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "hubbard_L6_weak",
        "label": "Hubbard L=6 weak",
        "family": "hubbard",
        "case_id": "hubbard_L6_higher_l_weak",
        "L": 6,
        "num_qubits": 12,
        "u": 0.25,
        "g_ep": 0.0,
        "n_ph": 1,
        "n_ph_ref": None,
        "same_cutoff_reference": False,
        "exact_energy": -6.621747713931651,
        "reference_hash": "62c1e7829d994a2cabeb48b7",
        "sector_particles": [3, 3],
        "sector_label": "half_filled_spin_sector",
        "parent_pool": "full_meta_family_native",
        "hva_included": False,
        "parent_count": 172,
        "raw_term_count": 948,
        "child_count": 54,
        "null_count": 6,
        "label_hash": "51806ecb80b6f04163ca4496d6b39962366c558eb8beed2377a8ffbee6a93da6",
        "pool_hash": "fd367d46c8974aa2ed8f36be74ea6210772a13deba91adc17378b1681197eb97",
        "contract_source_slug": "weak_weak",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    {
        "slug": "hubbard_L6_strong",
        "label": "Hubbard L=6 strong",
        "family": "hubbard",
        "case_id": "hubbard_L6_higher_l_strong",
        "L": 6,
        "num_qubits": 12,
        "u": 8.0,
        "g_ep": 0.0,
        "n_ph": 1,
        "n_ph_ref": None,
        "same_cutoff_reference": False,
        "exact_energy": -1.7680987552612777,
        "reference_hash": "bfba503ceb147c8a94634adb",
        "sector_particles": [3, 3],
        "sector_label": "half_filled_spin_sector",
        "parent_pool": "full_meta_family_native",
        "hva_included": False,
        "parent_count": 172,
        "raw_term_count": 948,
        "child_count": 54,
        "null_count": 6,
        "label_hash": "51806ecb80b6f04163ca4496d6b39962366c558eb8beed2377a8ffbee6a93da6",
        "pool_hash": "c058de476971503a5264b462c7d57e4647a87fd5696af1e804ad7b804ef4eb3d",
        "contract_source_slug": "strong_weak",
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
)

VARIANTS = (
    {
        "slug": "projected_singleton",
        "display": "Append-ADAPT projected singleton",
        "shared_pauli_pool_mode": "projected_singleton_children_only_v1",
        "shared_pauli_pool_symmetry_policy": "hard_guard",
        "shared_pauli_pool_max_subset_size": 1,
        "representation": "symmetry_and_padding_valid_projected_singleton_child",
    },
)

SOURCE_OVERLAYS = (
    "pipelines/exact_bench/table_i_canonical_cases.py",
    "test/test_paper_i_higher_l_discriminator_cases.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remote_execution_gate_status(bundle: Path) -> dict[str, Any]:
    gate_path = bundle / "remote_execution_gate.json"
    base = {
        "gate_path": gate_path.name,
        "gate_sha256": None,
        "schema_expected": REMOTE_EXECUTION_GATE_SCHEMA,
        "passed": False,
    }
    if not gate_path.is_file():
        return {**base, "reason": "missing"}
    base["gate_sha256"] = _sha256(gate_path)
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {**base, "reason": f"unreadable:{type(exc).__name__}"}
    remote = gate.get("remote_execution_preflight", {})
    checks = {
        "schema": gate.get("schema") == REMOTE_EXECUTION_GATE_SCHEMA,
        "status": gate.get("status") == "pass",
        "image_path": remote.get("image_path") == REMOTE_IMAGE_PATH,
        "image_sha256": remote.get("image_sha256") == EXPECTED_IMAGE_SHA256,
        "qiskit_import": remote.get("qiskit_import_passed") is True,
        "qiskit_version": remote.get("qiskit_version") == REMOTE_QISKIT_VERSION,
        "fake_backend_instantiation": (
            remote.get("fake_backend_instantiation_passed") is True
        ),
        "fake_backend_identity": (
            remote.get("fake_backend_resolved") == REMOTE_FAKE_BACKEND_RESOLVED
        ),
        "fake_backend_qubits": (
            remote.get("fake_backend_qubits") == REMOTE_FAKE_BACKEND_QUBITS
        ),
    }
    passed = all(checks.values())
    return {
        **base,
        "checks": checks,
        "passed": passed,
        "reason": "pass" if passed else "failed_checks",
    }


def _submission_requirements(*, submission_enabled: bool, remote_gate_passed: bool) -> str:
    if submission_enabled and not remote_gate_passed:
        raise RuntimeError(
            "submission cannot be enabled before remote_execution_gate.json passes"
        )
    return "TARGET.HasSIF" if submission_enabled else "False"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _deterministic_archive(source_root: Path, output: Path) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(source_root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(source_root).as_posix()
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def _freeze_source(repo: Path, bundle: Path) -> dict[str, Any]:
    """Overlay only the new case registry and its tests on immutable Append-v4."""

    base_bundle = repo / "chtc" / "phase3_optuna" / "input" / BASE_BUNDLE_ID
    base_archive = base_bundle / "source_locked.tar.gz"
    if _sha256(base_archive) != BASE_ARCHIVE_SHA256:
        raise RuntimeError("immutable parent source archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="append-comparator-source-") as tmp_name:
        root = Path(tmp_name) / "source"
        root.mkdir(parents=True)
        with tarfile.open(base_archive, "r:gz") as tar:
            tar.extractall(root, filter="data")
        overlay_records: dict[str, dict[str, Any]] = {}
        for relative in SOURCE_OVERLAYS:
            source = repo / relative
            if not source.is_file():
                raise RuntimeError(f"required source overlay is absent: {relative}")
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            overlay_records[relative] = {
                "sha256": _sha256(source),
                "size_bytes": source.stat().st_size,
            }
        archive_path = bundle / "source_locked.tar.gz"
        _deterministic_archive(root, archive_path)
        files = {
            path.relative_to(root).as_posix(): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    if files["pipelines/exact_bench/generic_static_adapt_variants.py"]["sha256"] != GENERIC_SOURCE_SHA256:
        raise RuntimeError("generic comparator source drifted from immutable Append-v4")
    payload = {
        "schema": "paper_i_higher_l_append_source_archive_v1",
        "generated_utc": _utc_now(),
        "archive": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "file_count": len(files),
        "files": files,
        "immutable_parent_archive": {
            "bundle_id": BASE_BUNDLE_ID,
            "sha256": BASE_ARCHIVE_SHA256,
        },
        "overlays": overlay_records,
        "executable_source_authority": "complete archive inventory and hashes",
        "git_role": "ancestry metadata only; dirty live worktree is not executable authority",
    }
    _write_json(bundle / "source_archive_manifest.json", payload)
    return payload


def _copy_append_v4_contract_locks(repo: Path, bundle: Path) -> dict[str, Any]:
    source_dir = (
        repo
        / "chtc"
        / "phase3_optuna"
        / "input"
        / BASE_BUNDLE_ID
        / "jobs"
    )
    target_dir = bundle / "source_contract_locks"
    target_dir.mkdir(parents=True, exist_ok=True)
    locks: dict[str, Any] = {}
    for regime in REGIMES:
        source = source_dir / (
            f"append_projected_singleton__{regime['contract_source_slug']}__r50.json"
        )
        if not source.is_file():
            raise RuntimeError(f"Append-v4 source contract lock missing: {source}")
        target = target_dir / f"{regime['slug']}__append_v4_job.json"
        shutil.copy2(source, target)
        locks[regime["slug"]] = {
            "path": str(target.relative_to(repo)),
            "sha256": _sha256(target),
        }
    return locks


def _job_manifest(
    *,
    variant: dict[str, Any],
    regime: dict[str, Any],
    source_archive_sha256: str,
) -> dict[str, Any]:
    job_id = f"append_{variant['slug']}__{regime['slug']}__r50"
    return {
        "schema": "paper_i_higher_l_append_projected_singleton_job_v1",
        "bundle_id": BUNDLE_ID,
        "job_id": job_id,
        "family": regime["family"],
        "algorithm_id": "static_full_meta_append_adapt_vqe",
        "method_id": "static_full_meta_append_adapt_vqe",
        "variant": {
            "slug": variant["slug"],
            "display": variant["display"],
            "candidate_representation": variant["representation"],
        },
        "regime": {
            "slug": regime["slug"],
            "label": regime["label"],
            "case_id": regime["case_id"],
        },
        "physics": {
            "L": regime["L"],
            "t": 1.0,
            "u": regime["u"],
            "omega0": 1.0,
            "g_ep": regime["g_ep"],
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "n_ph_work": regime["n_ph"],
            "n_ph_ref": regime["n_ph_ref"],
            "same_cutoff_reference": regime["same_cutoff_reference"],
            "suite_profile": SUITE_PROFILE,
        },
        "exact_reference": {
            "energy": regime["exact_energy"],
            "n_ph_max": regime["n_ph_ref"],
            "reference_hash": regime["reference_hash"],
            "usage": "reporting_only_after_optimization",
            "primary_metric": "same_cutoff_abs_delta_e",
        },
        "controller": {
            "fresh_round_zero": True,
            "max_adapt_iterations": 50,
            "stop_policy": "fixed_horizon_no_target_v1",
            "energy_stop_target": None,
            "gradient_threshold": 0.0,
            "allow_repeats": True,
            "selection_with_replacement": True,
            "hh_preseed": "off",
            "selected_logical_route": "standard",
            "initial_selected_operator_count": 0,
            "initial_theta_count": 0,
        },
        "runtime_limits": {
            "resource_qubit_cap": regime["num_qubits"],
            "exact_fidelity_max_qubits": regime["num_qubits"],
        },
        "optimizer": {
            "kind": "powell",
            "maxiter": 200,
            "powell_maxiter_cap_policy": "accept_finite_nonincreasing_v1",
            "overlay_source": "paper_i_completion_tracker_20260717",
        },
        "candidate_pool": {
            "parent_pool": regime["parent_pool"],
            "hva_included": regime["hva_included"],
            "generic_runtime_split_mode": "off",
            "shared_pauli_pool_mode": variant["shared_pauli_pool_mode"],
            "shared_pauli_pool_symmetry_policy": variant["shared_pauli_pool_symmetry_policy"],
            "shared_pauli_pool_max_subset_size": variant["shared_pauli_pool_max_subset_size"],
            "expanded_pool_term_cap": 9000,
            "macro_and_projected_singleton_mixing": False,
            "expected_parent_pool_count": regime["parent_count"],
            "expected_raw_pauli_term_count": regime["raw_term_count"],
            "expected_projected_singleton_child_count": regime["child_count"],
            "expected_null_child_count": regime["null_count"],
            "expected_ordered_label_hash": regime["label_hash"],
            "expected_ordered_pool_hash": regime["pool_hash"],
        },
        "sector_lock": {
            "label": regime["sector_label"],
            "num_particles": regime["sector_particles"],
            "state_norm_required": 1.0,
            "sector_leak_flag_required": False,
            "boson_truncation_leak_flag_required": False,
        },
        "accounting": {
            "S_alg_definition": "N_H_outer+N_H_refit+N_grad+N_metric",
            "state_keyed_estimator_call_ledger_required": True,
            "round_receipts_required": True,
            "cache_policy": "cache_off_but_measurement_occurrences_preserved",
            "exact_reference_and_fidelity_charged": False,
        },
        "fidelity": {
            "required": True,
            "same_cutoff_ground_space_convention_required": True,
            "reporting_only": True,
        },
        "qiskit_cost": {
            "required": True,
            "qiskit_version": "2.3.1",
            "transpile_seed": 7,
            "optimization_level": 0,
            "metrics": ["compiled_count_2q_total", "compiled_depth_2q_total", "compiled_depth_total"],
        },
        "seed": 7,
        "source_lock": {
            "archive": "source_locked.tar.gz",
            "archive_sha256": source_archive_sha256,
            "generic_static_adapt_variants_sha256": GENERIC_SOURCE_SHA256,
        },
        "output_contract": {
            "result_json": "generic_static_single.json",
            "runtime_seed_json": "runtime_seed.json",
            "progress_jsonl": "adapt_iteration_progress.jsonl",
            "validation_receipt": "validation_receipt.json",
            "transfer_archive": f"{job_id}_transfer.tar.gz",
        },
    }


def _write_submit(
    bundle: Path,
    source_sha: str,
    *,
    submission_enabled: bool,
    remote_gate_passed: bool,
) -> None:
    relative = f"chtc/phase3_optuna/input/{BUNDLE_ID}"
    requirements = _submission_requirements(
        submission_enabled=submission_enabled,
        remote_gate_passed=remote_gate_passed,
    )
    lines = [
        "universe = vanilla",
        "batch_name = paper-i-higher-l-append-projected-singleton-r50-20260719-v1",
        f"executable = {relative}/execute_source_locked_job.sh",
        (
            f"arguments = $(job_manifest) {relative}/source_locked.tar.gz {source_sha} "
            f"chtc/phase3_optuna/image.sif {EXPECTED_IMAGE_SHA256} $(job_id)"
        ),
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "preserve_relative_paths = True",
        (
            f"transfer_input_files = {relative}/run_job.py, $(job_manifest), "
            f"$(normalized_manifest), {relative}/source_archive_manifest.json, "
            f"{relative}/bundle_manifest.json, {relative}/source_locked.tar.gz, "
            "chtc/phase3_optuna/image.sif"
        ),
        f"transfer_output_files = raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz",
        (
            f'transfer_output_remaps = "raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz = '
            '$(job_id)_transfer.tar.gz"'
        ),
        "request_cpus = $(request_cpus)",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        "+WantFlocking = true",
        f"log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log",
        f"output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out",
        f"error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err",
        f"requirements = {requirements}",
        "# Both a valid remote_execution_gate.json and explicit SUBMISSION_ENABLED=True are required.",
        (
            "queue job_id, job_manifest, normalized_manifest, memory_mb, disk_mb, request_cpus "
            f"from {relative}/queue.tsv"
        ),
        "",
    ]
    (bundle / "submit.sub").write_text("\n".join(lines), encoding="utf-8")


def build() -> dict[str, Any]:
    repo = _repo_root()
    bundle = Path(__file__).resolve().parent
    remote_gate = _remote_execution_gate_status(bundle)
    _submission_requirements(
        submission_enabled=SUBMISSION_ENABLED,
        remote_gate_passed=bool(remote_gate["passed"]),
    )
    source = _freeze_source(repo, bundle)
    source_contract_locks = _copy_append_v4_contract_locks(repo, bundle)
    for path in (bundle / "jobs", bundle / "normalized_manifests"):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    queue_rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for regime in REGIMES:
            job = _job_manifest(
                variant=variant,
                regime=regime,
                source_archive_sha256=source["archive_sha256"],
            )
            job_path = bundle / "jobs" / f"{job['job_id']}.json"
            normalized_path = bundle / "normalized_manifests" / f"{job['job_id']}.json"
            _write_json(job_path, job)
            _write_json(normalized_path, job)
            queue_rows.append(
                {
                    "job_id": job["job_id"],
                    "job_manifest": str(job_path.relative_to(repo)),
                    "normalized_manifest": str(normalized_path.relative_to(repo)),
                    "memory_mb": regime["memory_mb"],
                    "disk_mb": regime["disk_mb"],
                    "request_cpus": 1,
                }
            )
            manifest_records.append(
                {
                    "job_id": job["job_id"],
                    "job_manifest_sha256": _sha256(job_path),
                    "normalized_manifest_sha256": _sha256(normalized_path),
                    "variant": variant["slug"],
                    "regime": regime["label"],
                }
            )
    with (bundle / "queue.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(queue_rows[0]), delimiter="\t")
        # Condor's `queue <vars> from <file>` treats a header as a data row.
        # Field names already live in submit.sub, so the execution queue is
        # intentionally headerless and maps to exactly five procs.
        writer.writerows(queue_rows)
    settings_audit = {
        "schema": "paper_i_higher_l_append_settings_difference_audit_v1",
        "generated_utc": _utc_now(),
        "status": "pass",
        "append_v4_source_contract_locks": source_contract_locks,
        "source_contract_bundle_id": BASE_BUNDLE_ID,
        "preserved_method_contract": {
            "algorithm_id": "static_full_meta_append_adapt_vqe",
            "selection": "full projected-singleton child candidate pool with replacement",
            "insertion": "append",
            "shared_pauli_pool_mode": "projected_singleton_children_only_v1",
            "shared_pauli_pool_symmetry_policy": "hard_guard",
            "shared_pauli_pool_max_subset_size": 1,
            "macro_and_projected_singleton_mixing": False,
            "optimizer": "Powell",
            "optimizer_maxiter": 200,
            "powell_maxiter_cap_policy": "accept_finite_nonincreasing_v1",
            "parent_pool": "family-native full_meta; HH unfiltered with HVA included",
        },
        "authorized_exploratory_changes": [
            "fresh output identity",
            "HH L=3,n_ph=3 strong-Holstein weak/intermediate/strong U rows",
            "Hubbard L=6 weak/strong U rows",
            "new diagnostic suite-profile case identifiers and exact-reference locks",
            "resource requests of 32 GB RAM and 60 GB disk",
        ],
        "unapproved_drift": [],
        "operational_repairs": [
            "overlay only the method-agnostic canonical case registry and its exact-reference/sector regression test",
            "generalize the byte-derived worker validation from HH-only to HH/Hubbard manifests",
            "raise and lock the generic resource/fidelity qubit caps to the declared 12-qubit cases",
            "accept path-safe ASCII uppercase letters in canonical higher-L job identifiers",
            "retain the Append-v4 generic algorithm source hash unchanged",
        ],
    }
    _write_json(bundle / "settings_difference_audit.json", settings_audit)
    bundle_manifest = {
        "schema": SCHEMA,
        "bundle_id": BUNDLE_ID,
        "generated_utc": _utc_now(),
        "status": "prepared_not_submitted",
        "submission_enabled": SUBMISSION_ENABLED,
        "job_count": len(manifest_records),
        "variant_count": len(VARIANTS),
        "regime_count_per_variant": len(REGIMES),
        "records": manifest_records,
        "source_archive": {
            "path": str((bundle / "source_locked.tar.gz").relative_to(repo)),
            "sha256": source["archive_sha256"],
        },
        "expected_image_sha256": EXPECTED_IMAGE_SHA256,
        "remote_image_gate": remote_gate,
        "submission_requirements": (
            "TARGET.HasSIF" if SUBMISSION_ENABLED else "False"
        ),
        "scientific_blockers": [],
        "operational_blockers": (
            ([] if remote_gate["passed"] else ["remote image/Qiskit gate not passed"])
            + ([] if SUBMISSION_ENABLED else ["submission deliberately disabled"])
        ),
    }
    _write_json(bundle / "bundle_manifest.json", bundle_manifest)
    _write_submit(
        bundle,
        source["archive_sha256"],
        submission_enabled=SUBMISSION_ENABLED,
        remote_gate_passed=bool(remote_gate["passed"]),
    )
    hashes: dict[str, str] = {}
    for path in sorted(bundle.rglob("*")):
        if (
            path.is_file()
            and path.name != "submission_artifact_hashes.json"
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        ):
            hashes[path.relative_to(bundle).as_posix()] = _sha256(path)
    _write_json(
        bundle / "submission_artifact_hashes.json",
        {"schema": "paper_i_higher_l_append_artifact_hashes_v1", "files": hashes},
    )
    return bundle_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("pass --build")
    payload = build()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
