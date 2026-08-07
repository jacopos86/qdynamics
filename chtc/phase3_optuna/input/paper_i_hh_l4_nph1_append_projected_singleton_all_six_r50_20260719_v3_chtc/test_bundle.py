from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import tarfile
from pathlib import Path

import pytest


BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
_SPEC = importlib.util.spec_from_file_location("append_l4_nph1_bundle_builder", BUNDLE / "build_bundle.py")
assert _SPEC is not None and _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)
EXPECTED_SOURCE = "33f0e8ffba1d532e86037077e99d8578423fc7a52842e2479a40afea1588ed3d"
EXPECTED_REGIMES = {
    "weak_weak": (1, 0.25, 0.3535533905932738, -2.3048285705420595, "6afec225c633ee4f6eda41ae"),
    "intermediate_weak": (1, 1.25, 0.3535533905932738, -1.4337263224699544, "51427f68812dba5c5799afa5"),
    "strong_weak": (1, 8.0, 0.3535533905932738, 0.8794946903883553, "0a5ae9b243ee5e620ed3ac56"),
    "weak_strong": (1, 0.25, 0.7905694150420949, -2.634636761963219, "72ca58b50fdd3dc00d562246"),
    "intermediate_strong": (1, 1.25, 0.7905694150420949, -1.655459805225902, "2b05b5daedbe05ede60a12bc"),
    "strong_strong": (1, 8.0, 0.7905694150420949, 0.8657815466399882, "9a3fa7ec3fd5f7213c30b83b"),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_is_prepared_and_submission_is_enabled() -> None:
    manifest = _load(BUNDLE / "bundle_manifest.json")
    assert manifest["status"] == "prepared_not_submitted"
    assert manifest["submission_enabled"] is True
    assert manifest["job_count"] == 6
    assert manifest["scientific_blockers"] == []
    assert manifest["operational_blockers"] == []
    assert manifest["remote_image_gate"]["passed"] is True
    assert manifest["remote_image_gate"]["reason"] == "pass"
    submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
    assert "requirements = TARGET.HasSIF" in submit
    assert "queue job_id, job_manifest, normalized_manifest" in submit


def _write_remote_gate(path: Path, *, image_sha256: str, status: str = "pass") -> None:
    path.write_text(
        json.dumps(
            {
                "schema": builder.REMOTE_EXECUTION_GATE_SCHEMA,
                "status": status,
                "remote_execution_preflight": {
                    "image_path": builder.REMOTE_IMAGE_PATH,
                    "image_sha256": image_sha256,
                    "qiskit_import_passed": True,
                    "qiskit_version": "2.3.1",
                    "fake_backend_instantiation_passed": True,
                    "fake_backend_resolved": "fake_marrakesh",
                    "fake_backend_qubits": 156,
                },
            }
        ),
        encoding="utf-8",
    )


def test_remote_execution_gate_and_switch_are_both_required(tmp_path: Path) -> None:
    assert builder.SUBMISSION_ENABLED is True
    absent = builder._remote_execution_gate_status(tmp_path)
    assert absent["passed"] is False
    assert absent["reason"] == "missing"
    assert builder._submission_requirements(
        submission_enabled=False, remote_gate_passed=False
    ) == "False"
    with pytest.raises(RuntimeError, match="remote_execution_gate.json passes"):
        builder._submission_requirements(
            submission_enabled=True, remote_gate_passed=False
        )

    _write_remote_gate(
        tmp_path / "remote_execution_gate.json",
        image_sha256="0" * 64,
    )
    failed = builder._remote_execution_gate_status(tmp_path)
    assert failed["passed"] is False
    assert failed["checks"]["image_sha256"] is False

    _write_remote_gate(
        tmp_path / "remote_execution_gate.json",
        image_sha256=builder.EXPECTED_IMAGE_SHA256,
    )
    valid = builder._remote_execution_gate_status(tmp_path)
    assert valid["passed"] is True
    assert builder._submission_requirements(
        submission_enabled=False, remote_gate_passed=True
    ) == "False"
    assert builder._submission_requirements(
        submission_enabled=True, remote_gate_passed=True
    ) == "TARGET.HasSIF"
    builder._write_submit(
        tmp_path,
        "a" * 64,
        submission_enabled=True,
        remote_gate_passed=True,
    )
    assert "requirements = TARGET.HasSIF" in (
        tmp_path / "submit.sub"
    ).read_text(encoding="utf-8")
    builder._write_submit(
        tmp_path,
        "a" * 64,
        submission_enabled=False,
        remote_gate_passed=True,
    )
    assert "requirements = False" in (tmp_path / "submit.sub").read_text(
        encoding="utf-8"
    )


def test_queue_is_exact_projected_singleton_six_regime_matrix() -> None:
    fields = (
        "job_id",
        "job_manifest",
        "normalized_manifest",
        "memory_mb",
        "disk_mb",
        "request_cpus",
    )
    with (BUNDLE / "queue.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, fieldnames=fields, delimiter="\t"))
    assert len(rows) == 6
    job_ids = [row["job_id"] for row in rows]
    assert job_ids[0] == "append_projected_singleton__hh_L4_nph1__weak_weak__r50"
    assert "job_id" not in job_ids
    first_job = REPO / rows[0]["job_manifest"]
    first_normalized = REPO / rows[0]["normalized_manifest"]
    assert first_job.is_file()
    assert first_normalized.is_file()
    assert _load(first_job)["job_id"] == job_ids[0]
    assert _load(first_normalized)["job_id"] == job_ids[0]
    submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
    assert "queue job_id, job_manifest, normalized_manifest, memory_mb, disk_mb, request_cpus from" in submit
    assert len(set(job_ids)) == 6
    assert {job.split("__")[2] for job in job_ids} == set(EXPECTED_REGIMES)
    assert {int(row["memory_mb"]) for row in rows} == {32768}
    assert {int(row["disk_mb"]) for row in rows} == {61440}
    assert {int(row["request_cpus"]) for row in rows} == {1}


@pytest.mark.parametrize("regime", sorted(EXPECTED_REGIMES))
def test_normalized_manifest_contract(regime: str) -> None:
    variant = "projected_singleton"
    job_id = f"append_{variant}__hh_L4_nph1__{regime}__r50"
    job = _load(BUNDLE / "jobs" / f"{job_id}.json")
    normalized = _load(BUNDLE / "normalized_manifests" / f"{job_id}.json")
    assert job == normalized
    n_ph, u, g_ep, exact, reference_hash = EXPECTED_REGIMES[regime]
    assert job["algorithm_id"] == "static_full_meta_append_adapt_vqe"
    assert job["physics"]["L"] == 4
    assert job["physics"]["suite_profile"] == "paper_i_scaling_matrix_20260710_v1"
    assert job["physics"]["n_ph_work"] == n_ph
    assert job["physics"]["n_ph_ref"] == n_ph
    assert job["physics"]["same_cutoff_reference"] is True
    assert job["physics"]["u"] == u
    assert job["physics"]["g_ep"] == g_ep
    assert job["exact_reference"]["energy"] == exact
    assert job["exact_reference"]["reference_hash"] == reference_hash
    assert job["exact_reference"]["usage"] == "reporting_only_after_optimization"
    assert job["controller"] == {
        "allow_repeats": True,
        "energy_stop_target": None,
        "fresh_round_zero": True,
        "gradient_threshold": 0.0,
        "hh_preseed": "off",
        "initial_selected_operator_count": 0,
        "initial_theta_count": 0,
        "max_adapt_iterations": 50,
        "selected_logical_route": "standard",
        "selection_with_replacement": True,
        "stop_policy": "fixed_horizon_no_target_v1",
    }
    assert job["optimizer"]["kind"] == "powell"
    assert job["optimizer"]["maxiter"] == 200
    assert job["optimizer"]["powell_maxiter_cap_policy"] == "accept_finite_nonincreasing_v1"
    assert job["seed"] == 7
    pool = job["candidate_pool"]
    assert pool["parent_pool"] == "full_meta_unfiltered"
    assert pool["hva_included"] is True
    assert pool["generic_runtime_split_mode"] == "off"
    assert pool["macro_and_projected_singleton_mixing"] is False
    assert pool["shared_pauli_pool_mode"] == "projected_singleton_children_only_v1"
    assert pool["shared_pauli_pool_symmetry_policy"] == "hard_guard"
    assert pool["shared_pauli_pool_max_subset_size"] == 1
    assert pool["resource_qubit_cap"] == 12
    assert pool["expanded_pool_term_cap"] == 9000


def test_source_archive_and_inventory_are_immutable_and_closed() -> None:
    manifest = _load(BUNDLE / "source_archive_manifest.json")
    archive = BUNDLE / "source_locked.tar.gz"
    assert manifest["archive_sha256"] == _sha256(archive)
    assert manifest["file_count"] == len(manifest["files"])
    assert (
        manifest["files"]["pipelines/exact_bench/generic_static_adapt_variants.py"]["sha256"]
        == EXPECTED_SOURCE
    )
    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        assert set(members) == set(manifest["files"])
        for relative, record in manifest["files"].items():
            stream = tar.extractfile(members[relative])
            assert stream is not None
            assert hashlib.sha256(stream.read()).hexdigest() == record["sha256"]


def test_append_v4_source_contracts_are_vendored_by_hash() -> None:
    audit = _load(BUNDLE / "settings_difference_audit.json")
    assert audit["status"] == "pass"
    assert audit["unapproved_drift"] == []
    assert audit["source_contract_bundle_id"] == builder.BASE_BUNDLE_ID
    assert set(audit["append_v4_source_contract_locks"]) == set(EXPECTED_REGIMES)
    for record in audit["append_v4_source_contract_locks"].values():
        path = REPO / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_worker_requires_receipts_fidelity_and_qiskit_costs() -> None:
    worker = (BUNDLE / "run_job.py").read_text(encoding="utf-8")
    for token in (
        "expected 50 estimator round receipts",
        "exact_state_fidelity_reference_convention",
        "exact_state_fidelity_s_alg_charged",
        "compiled_count_2q_total",
        "compiled_depth_2q_total",
        "compiled_depth_total",
        "sector_leak_flag",
        "boson_truncation_leak_flag",
        "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP",
        "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP",
        "projected-singleton pool was not fully expanded",
    ):
        assert token in worker


def test_worker_uses_manifest_cap_policy_and_is_true_fresh_round_zero() -> None:
    worker = (BUNDLE / "run_job.py").read_text(encoding="utf-8")
    assert 'powell_maxiter_cap_policy=optimizer["powell_maxiter_cap_policy"]' in worker
    for forbidden in (
        "initial_selected_operator_labels=",
        "initial_selected_operator_batches=",
        "initial_theta=",
        "initial_adapt_history=",
        'powell_maxiter_cap_policy="strict_failure_v1"',
    ):
        assert forbidden not in worker
    for required in (
        "finite_nonincreasing_powell_maxiter_accepted",
        "Append capped parameters",
        "Append capped objective",
        "Append capped energy",
        "expected 50 completed Append rounds",
    ):
        assert required in worker


def test_wrapper_uses_immutable_successor_path() -> None:
    wrapper = (BUNDLE / "execute_source_locked_job.sh").read_text(encoding="utf-8")
    assert f'BUNDLE_ID="{builder.BUNDLE_ID}"' in wrapper
    assert '"chtc/phase3_optuna/input/${BUNDLE_ID}/run_job.py"' in wrapper


def test_wrapper_accepts_canonical_ids_but_remains_path_safe() -> None:
    wrapper = (BUNDLE / "execute_source_locked_job.sh").read_text(encoding="utf-8")
    assert '[[ "$JOB_ID" =~ ^[A-Za-z0-9_]+$ ]]' in wrapper
    fields = (
        "job_id",
        "job_manifest",
        "normalized_manifest",
        "memory_mb",
        "disk_mb",
        "request_cpus",
    )
    with (BUNDLE / "queue.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, fieldnames=fields, delimiter="\t"))
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", row["job_id"]) for row in rows)
    assert any("L4" in row["job_id"] for row in rows)
