from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v3_chtc"
)
V4_PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4_chtc"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_all_python_control_files_compile_without_bytecode_side_effects() -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    python_controls = [
        PACKAGE_DIR / str(row["path"])
        for row in manifest["control_files"]
        if str(row["path"]).endswith(".py")
    ]
    assert python_controls
    for path in python_controls:
        compile(path.read_text(encoding="utf-8"), path.as_posix(), "exec")


def test_l3_matched_package_has_exact_six_rows_and_separate_facades() -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    source = _load(PACKAGE_DIR / "source/source_archive_manifest.json")
    jobs = [_load(PACKAGE_DIR / str(row["path"])) for row in manifest["jobs"]]
    assert manifest["status"] == "passed_inert_matched_six_cell"
    assert manifest["row_count"] == len(jobs) == 6
    assert manifest["execution_entrypoint_counts"] == {
        "run_ra_adapt": 3,
        "run_append_adapt": 3,
    }
    assert {(row["regime_id"], row["method"]) for row in jobs} == {
        (regime, method)
        for regime in ("weak_weak", "intermediate_weak", "strong_weak_u8")
        for method in ("ra_page12", "append_adapt")
    }
    for row in jobs:
        assert row["nph"] == 3
        assert row["target_horizon"] == 50
        assert row["execution_entrypoint"] == (
            "run_ra_adapt"
            if row["method"] == "ra_page12"
            else "run_append_adapt"
        )
        protocol = _load(PACKAGE_DIR / str(row["protocol_path"]))
        if row["method"] == "ra_page12":
            assert protocol["request"]["method"]["pruning"]["kind"] == "off"
            assert protocol["request"]["method"]["beam"]["kind"] == "off"
            assert protocol["request"]["method"]["insertion"]["kind"] == (
                "plateau_commutation"
            )
        else:
            assert protocol["schema"] == "paper_i_append_adapt_resolved_protocol_v1"
            assert protocol["selector_scope"] == (
                "conventional_append_no_phase3_no_trust_v1"
            )
            assert protocol["lineage_authority"]["ra_staged_funnel_invoked"] is False
    members = {str(row["path"]): row for row in source["members"]}
    assert "pipelines/static_adapt/ra_adapt/l3_page12.py" in members
    selector = members["pipelines/exact_bench/generic_static_adapt_variants.py"]
    assert selector["source_kind"] == "append_runtime_hash_dependency"
    assert selector["sha256"] == (
        "1a82945bfcc8e4273c09e2c4f24fb7c1f85df71bb1b952163afe8f349d4262e1"
    )


def test_extracted_archive_satisfies_real_append_path_hash_access() -> None:
    execution_id = (
        "l3_weak_holstein__weak_weak__nph3__append_conventional_unwhitened"
    )
    script = r"""
from pathlib import Path
import sys
package = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(package))
import run_cell
job = package / "jobs" / (sys.argv[2] + ".json")
closed, manifest, protocol, problem, temporary = run_cell._prepare(job)
try:
    from pipelines.static_adapt.ra_adapt.append import _source_lock_receipts
    receipts = _source_lock_receipts(problem)
    assert receipts["selector_module_sha256"] == protocol.source_locks["selector_module_sha256"]
    assert receipts["selector_module_sha256"] == "1a82945bfcc8e4273c09e2c4f24fb7c1f85df71bb1b952163afe8f349d4262e1"
    assert closed["execution_entrypoint"] == "run_append_adapt"
finally:
    temporary.cleanup()
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, str(PACKAGE_DIR), execution_id],
        cwd=PACKAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _normalized_protocol_revision(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized.pop("sha256", None)
    normalized["bundle_id"] = "<revision>"
    normalized["bundle_manifest_sha256"] = "<revision>"
    materialization = dict(normalized["bundle_materialization"])
    materialization.pop("sha256", None)
    for key in (
        "bundle_id",
        "bundle_manifest_sha256",
        "source_locks_sha256",
        "source_lock_refs_sha256",
    ):
        materialization[key] = "<revision>"
    normalized["bundle_materialization"] = materialization
    source_locks = dict(normalized["source_locks"])
    source_locks["source_locks_manifest_sha256"] = "<revision>"
    normalized["source_locks"] = source_locks
    return normalized


def _normalized_job_revision(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized.pop("sha256", None)
    for key in ("package_id", "campaign_id", "bundle_id"):
        normalized[key] = "<revision>"
    normalized["protocol_path"] = Path(str(normalized["protocol_path"])).name
    for key in (
        "protocol_file_sha256",
        "protocol_sha256",
        "bundle_manifest_sha256",
        "source_locks_sha256",
        "expected_artifacts_manifest_sha256",
    ):
        normalized[key] = "<revision>"
    resources = dict(normalized["resources"])
    normalized["resources"] = {
        "request_cpus": resources["request_cpus"],
        "max_runtime_seconds": resources["max_runtime_seconds"],
    }
    return normalized


def test_v4_is_an_exact_source_locked_resource_only_revision_of_v3() -> None:
    v3_manifest_path = PACKAGE_DIR / "package_manifest.json"
    assert hashlib.sha256(v3_manifest_path.read_bytes()).hexdigest() == (
        "bf7416fc0601a812ffc58a6345cf5eaa895f95d8d7ef3837f462d842860ee320"
    )
    v3_manifest = _load(v3_manifest_path)
    v4_manifest = _load(V4_PACKAGE_DIR / "package_manifest.json")
    equivalence = _load(V4_PACKAGE_DIR / "v3_scientific_equivalence.json")
    assert equivalence["status"] == (
        "passed_exact_source_and_scientific_semantics"
    )
    assert equivalence["v3_package_manifest_canonical_sha256"] == (
        v3_manifest["sha256"]
    )
    assert equivalence["scientific_semantic_drift_detected"] is False
    assert v4_manifest["scientific_equivalence"]["canonical_sha256"] == (
        equivalence["sha256"]
    )

    v3_archive = PACKAGE_DIR / str(v3_manifest["source_archive"]["path"])
    v4_archive = V4_PACKAGE_DIR / str(v4_manifest["source_archive"]["path"])
    assert v3_archive.read_bytes() == v4_archive.read_bytes()
    assert hashlib.sha256(v4_archive.read_bytes()).hexdigest() == (
        "2aa61620dee19e9dcadb9e90a1008969e8c1ce752f1ad7ee9ccfdc94c7973400"
    )

    v3_jobs = {
        str(row["execution_id"]): _load(PACKAGE_DIR / str(row["path"]))
        for row in v3_manifest["jobs"]
    }
    v4_jobs = {
        str(row["execution_id"]): _load(V4_PACKAGE_DIR / str(row["path"]))
        for row in v4_manifest["jobs"]
    }
    expected_resources = {
        "ra_page12": {
            "request_cpus": 4,
            "request_memory_mb": 65_536,
            "request_disk_mb": 81_920,
            "max_runtime_seconds": 259_200,
            "basis": "v3_resource_only_rightsizing_ra_page12_v1",
        },
        "append_adapt": {
            "request_cpus": 1,
            "request_memory_mb": 49_152,
            "request_disk_mb": 61_440,
            "max_runtime_seconds": 259_200,
            "basis": "v3_resource_only_rightsizing_conventional_append_v1",
        },
    }
    assert set(v3_jobs) == set(v4_jobs)
    for execution_id, v4_job in v4_jobs.items():
        assert v4_job["resources"] == expected_resources[v4_job["method"]]
        assert _normalized_job_revision(v3_jobs[execution_id]) == (
            _normalized_job_revision(v4_job)
        )

    v3_protocols = {
        str(row["execution_id"]): _load(PACKAGE_DIR / str(row["path"]))
        for row in v3_manifest["protocols"]
    }
    v4_protocols = {
        str(row["execution_id"]): _load(V4_PACKAGE_DIR / str(row["path"]))
        for row in v4_manifest["protocols"]
    }
    assert set(v3_protocols) == set(v4_protocols)
    for execution_id, v4_protocol in v4_protocols.items():
        assert _normalized_protocol_revision(v3_protocols[execution_id]) == (
            _normalized_protocol_revision(v4_protocol)
        )


def test_v4_queue_and_submit_template_use_only_the_new_revision() -> None:
    queue_rows = [
        line.split("\t")
        for line in (V4_PACKAGE_DIR / "queue.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(queue_rows) == 6
    for execution_id, _job, _protocol, _job_sha, cpus, memory, disk, runtime in (
        queue_rows
    ):
        is_ra = execution_id.endswith(
            "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau"
        )
        assert (cpus, memory, disk, runtime) == (
            ("4", "65536", "81920", "259200")
            if is_ra
            else ("1", "49152", "61440", "259200")
        )
    submit_template = (V4_PACKAGE_DIR / "submit.sub.in").read_text(
        encoding="utf-8"
    )
    assert "paper-i-l3-weak-holstein-page12-append6-r50-20260812-v4" in (
        submit_template
    )
    assert "/paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4/" in (
        submit_template
    )
    assert "20260812_v3" not in submit_template


def test_v4_activation_request_is_self_digested_and_package_exact() -> None:
    request_path = (
        V4_PACKAGE_DIR.parent
        / "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4_"
        "activation_request.json"
    )
    request = _load(request_path)
    manifest = _load(V4_PACKAGE_DIR / "package_manifest.json")
    unsigned = dict(request)
    observed = str(unsigned.pop("sha256"))
    assert observed == hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert request["package_id"] == manifest["package_id"]
    assert request["campaign_id"] == manifest["campaign_id"]
    assert request["bundle_id"] == manifest["bundle_id"]
    assert request["package_manifest_sha256"] == manifest["sha256"]
    assert request["requested_execution_ids"] == manifest["execution_ids"]
    assert request["execution_authorized"] is True
    assert request["submission_authorized"] is True
    assert request["submitted"] is False


@pytest.mark.parametrize(
    ("manifest_key", "target_relative"),
    (
        ("control_files", "run_cell.py"),
        (
            "application_source_contracts",
            "source_authority/weak_weak_application_source_contract.json",
        ),
    ),
)
def test_v4_validator_rejects_control_or_application_source_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_key: str,
    target_relative: str,
) -> None:
    copied_package = tmp_path / "package"
    shutil.copytree(V4_PACKAGE_DIR, copied_package)
    target = copied_package / target_relative
    target.write_bytes(target.read_bytes() + b"\n")

    module_path = V4_PACKAGE_DIR / "validate_package.py"
    specification = importlib.util.spec_from_file_location(
        f"l3_v4_validate_{manifest_key}", module_path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(module, "PACKAGE_DIR", copied_package)
    with pytest.raises(module.PackageContractError, match="byte binding drifted"):
        module.validate_package()
