from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc"
    / "paper_i_ra_adapt_repair_20260727"
    / "stationary_core_ra36_r70_continuation_20260731_v1_chtc"
)
CHECKPOINT_MEMBER = "pipelines/static_adapt/current_checkpoint.py"
OLD_CHECKPOINT_SHA256 = (
    "16ffddfdbf20674c50af7b797131efa40478c5281d16f4f034d7db49b8249cb8"
)
REPAIRED_CHECKPOINT_SHA256 = (
    "87e032010e009261de415101b717ff38fdb3d9b894b18d1939e6b219d94219f3"
)
MINIMUM_FREE_BYTES = 10 * 1024**3
PROTOCOL_DERIVATION_TIMEOUT_SECONDS = 600
ISOLATED_PROTOCOL_DERIVATION = r"""
import json
from pathlib import Path
import sys
import tempfile

package_dir = Path(sys.argv[1]).resolve()
job_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(package_dir))

import run_cell_v2 as runtime

job = runtime._load_v2_job(job_path)
with tempfile.TemporaryDirectory(
    prefix=f".{job['execution_id']}.protocol-validation."
) as raw:
    source_root = Path(raw) / "source"
    runtime.base_runtime._extract_source(job, source_root)
    runtime.base_runtime._activate_source_root(source_root)
    protocol, _problem, _delta = runtime.base_runtime._derived_protocol(
        job=job,
        source_root=source_root,
    )
    effective = runtime.build_effective_execution_contract(
        job=job,
        derived_protocol_payload=protocol.to_dict(),
    )
    if effective != job["effective_execution_contract"]:
        raise RuntimeError(
            "rederived effective execution contract drifted"
        )
    contract_sha256 = runtime.effective_contract_sha256(effective)
    if contract_sha256 != job["effective_execution_contract_sha256"]:
        raise RuntimeError(
            "rederived effective execution contract digest drifted"
        )
    print(
        json.dumps(
            {
                "execution_id": job["execution_id"],
                "protocol_sha256": protocol.sha256,
                "contract_sha256": contract_sha256,
            },
            sort_keys=True,
        )
    )
"""


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE_DIR / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(item, key) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _member_hash(path: Path, member_name: str) -> str:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(member_name)
        stream = archive.extractfile(member)
        assert stream is not None
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_overlay_metadata_and_cross_document_contracts_pass() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(
                PACKAGE_DIR
                / "validate_operational_overlay_v2.py"
            ),
            "--metadata-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed_inert_collision_blocked"
    assert result["cell_count"] == 36
    assert result["authenticated_resume_count"] == 27
    assert result["fresh_count"] == 9
    assert result["effective_source_family_count"] == 2
    assert result["changed_source_members"] == [CHECKPOINT_MEMBER]
    assert result["immutable_compact_resume_inputs"] is True
    assert result["settings_hash_removed"] is True
    assert result["operator_sequence_match_claim_count"] == 0
    assert result["submission_ready"] is False
    assert result["submitted"] is False


def test_effective_sources_are_explicit_one_member_operational_deltas() -> None:
    sources = _json(PACKAGE_DIR / "effective_source_archives_v2.json")
    base_sources = _json(PACKAGE_DIR / "source_archives.json")
    families = sources["families"]

    assert set(families) == {
        "stationary_core_v11_retention_v2",
        "always_factorial_retention_v2",
    }
    assert sources["changed_source_members"] == [CHECKPOINT_MEMBER]
    assert sources["protocol_members_byte_identical"] is True
    assert sources["scientific_settings_changed"] == []
    assert sources["controller_semantics_changed"] is False
    for row in families.values():
        delta = _json(PACKAGE_DIR / row["delta_receipt"]["path"])
        assert delta["changed_member_count"] == 1
        assert delta["changed_members"][0]["path"] == CHECKPOINT_MEMBER
        assert (
            delta["changed_members"][0]["parent_sha256"]
            == OLD_CHECKPOINT_SHA256
        )
        assert (
            delta["changed_members"][0]["repaired_sha256"]
            == REPAIRED_CHECKPOINT_SHA256
        )
        assert (
            delta["changed_members"][0]["scientific_protocol_change"]
            is False
        )
        assert (
            delta["changed_members"][0]["controller_semantics_change"]
            is False
        )
        assert delta["protocol_members_byte_identical"] is True
        assert delta["scientific_settings_changed"] == []
        effective_archive = (
            PACKAGE_DIR / row["effective_archive"]["path"]
        )
        assert (
            _member_hash(effective_archive, CHECKPOINT_MEMBER)
            == REPAIRED_CHECKPOINT_SHA256
        )

    core_parent = (
        PACKAGE_DIR
        / base_sources["families"]["stationary_core_v11"][
            "packaged_archive"
        ]["path"]
    )
    always_parent = (
        PACKAGE_DIR
        / base_sources["families"]["always_factorial_v1"][
            "packaged_archive"
        ]["path"]
    )
    assert (
        _member_hash(core_parent, CHECKPOINT_MEMBER)
        == OLD_CHECKPOINT_SHA256
    )
    assert (
        _member_hash(always_parent, CHECKPOINT_MEMBER)
        == OLD_CHECKPOINT_SHA256
    )


def test_all_36_jobs_reconstruct_truthful_split_settings_contracts() -> None:
    contract_module = _module(
        "stationary_core_r70_overlay_contract_test",
        "operational_overlay_v2_contract.py",
    )
    manifest = _json(
        PACKAGE_DIR / "operational_overlay_v2_manifest.json"
    )
    plan = _json(PACKAGE_DIR / "execution_plan_v2.json")
    audit = _json(PACKAGE_DIR / "source_lock_audit_v2.json")
    audit_by_id = {
        row["execution_id"]: row for row in audit["planned_rows"]
    }

    assert len(manifest["jobs"]) == 36
    assert len(plan["effective_execution_contracts"]) == 36
    assert audit["anchor"]["operator_sequence_match_claim_count"] == 0
    for binding in manifest["jobs"]:
        job = _json(PACKAGE_DIR / binding["path"])
        execution_id = job["execution_id"]
        effective = job["effective_execution_contract"]
        derived = effective["scientific_settings"][
            "derived_protocol_payload"
        ]
        rebuilt = contract_module.build_effective_execution_contract(
            job=job, derived_protocol_payload=derived
        )

        assert rebuilt == effective
        assert (
            contract_module.effective_contract_sha256(effective)
            == job["effective_execution_contract_sha256"]
        )
        assert (
            effective["scientific_settings_sha256"]
            == job["scientific_settings_sha256"]
            == plan["effective_execution_contracts"][execution_id][
                "scientific_settings_sha256"
            ]
        )
        assert (
            effective["operational_settings_sha256"]
            == job["operational_settings_sha256"]
            == plan["effective_execution_contracts"][execution_id][
                "operational_settings_sha256"
            ]
        )
        assert not _contains_key(job, "settings_hash")
        resume_policy = effective["operational_settings"][
            "resume_policy"
        ]
        if job["execution_mode"] == "authenticated_resume_50_to_70":
            assert resume_policy["kind"] == "accepted_state_resume"
            assert set(resume_policy["members_by_role"]) == {
                "checkpoint",
                "estimator_ledger_checkpoint",
                "verified_resume_sidecar",
            }
            assert resume_policy["pointer_closed"] is True
        else:
            assert resume_policy["kind"] == "fresh_start"
            assert resume_policy["collision_clearance_required"] is True
        assert (
            audit_by_id[execution_id]["anchor"][
                "operator_sequence_digest_available"
            ]
            is False
        )
        assert (
            audit_by_id[execution_id]["anchor"][
                "operator_sequence_match_claimed"
            ]
            is False
        )
        assert (
            "operator_sequence_match"
            not in audit_by_id[execution_id]["anchor"]
        )


def test_queue_binds_all_resource_and_runtime_columns() -> None:
    plan = _json(PACKAGE_DIR / "execution_plan_v2.json")
    lines = (
        PACKAGE_DIR / "queue_v2.tsv"
    ).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 36
    for execution_id, line in zip(plan["execution_ids"], lines):
        job = _json(
            PACKAGE_DIR / "jobs_v2" / f"{execution_id}.json"
        )
        assert line.split("\t") == [
            execution_id,
            job["execution_mode"],
            job["collision_status"],
            job["effective_source_family"],
            str(job["resources"]["request_cpus"]),
            str(job["resources"]["request_memory_mb"]),
            str(job["resources"]["request_disk_mb"]),
            str(job["resources"]["max_runtime_seconds"]),
        ]


def test_fresh_worker_requires_external_exact_proc_clearance(
    tmp_path: Path,
) -> None:
    fresh = next(
        path
        for path in sorted((PACKAGE_DIR / "jobs_v2").glob("*.json"))
        if _json(path)["execution_mode"] == "fresh_0_to_70"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "run_cell_v2.py"),
            "--job",
            str(fresh),
            "--execution-authorization",
            str(tmp_path / "absent-authorization.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 2
    assert "remains blocked by predecessor 9397758." in completed.stderr
    assert "sealed external collision clearance is required" in (
        completed.stderr
    )
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_all_36_protocols_rederive_from_effective_source_trees() -> None:
    plan = _json(PACKAGE_DIR / "execution_plan_v2.json")
    expected_ids = plan["execution_ids"]
    assert len(expected_ids) == 36
    observed_ids: list[str] = []

    for execution_id in expected_ids:
        job_path = (
            PACKAGE_DIR / "jobs_v2" / f"{execution_id}.json"
        )
        free_before = shutil.disk_usage(REPO_ROOT).free
        assert free_before >= MINIMUM_FREE_BYTES, (
            f"refusing to derive {execution_id}: only "
            f"{free_before / 1024**3:.2f} GiB free"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                ISOLATED_PROTOCOL_DERIVATION,
                str(PACKAGE_DIR),
                str(job_path),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        started = time.monotonic()
        while process.poll() is None:
            free_now = shutil.disk_usage(REPO_ROOT).free
            elapsed = time.monotonic() - started
            if (
                free_now < MINIMUM_FREE_BYTES
                or elapsed > PROTOCOL_DERIVATION_TIMEOUT_SECONDS
            ):
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=10)
                assert False, (
                    f"stopped isolated derivation for {execution_id}: "
                    f"{free_now / 1024**3:.2f} GiB free after "
                    f"{elapsed:.1f}s\nstdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )
            time.sleep(0.5)
        stdout, stderr = process.communicate(timeout=10)
        free_after = shutil.disk_usage(REPO_ROOT).free
        assert free_after >= MINIMUM_FREE_BYTES, (
            f"{execution_id} left only "
            f"{free_after / 1024**3:.2f} GiB free"
        )
        assert process.returncode == 0, (
            f"{execution_id} failed in its isolated process\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
        result = json.loads(stdout.splitlines()[-1])
        assert result["execution_id"] == execution_id
        assert result["protocol_sha256"]
        assert result["contract_sha256"]
        observed_ids.append(result["execution_id"])

    assert observed_ids == expected_ids
