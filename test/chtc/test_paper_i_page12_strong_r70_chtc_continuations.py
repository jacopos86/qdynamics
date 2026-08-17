from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260811_v1_chtc"
)
LOCAL_SOURCE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260810_v2_local"
)
ROUTE_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)
SOURCE_ARCHIVE_SHA256 = (
    "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762"
)
CONTROLLER_SHA256 = (
    "e25c0281373b828f75200410aa0e5364eaebe5a78f517421bc8c7bdc73c20327"
)
RESUME_SHA256 = (
    "00a06606cf69dce5ee749172839b2115a5b5bb7dce72b170fc58d792ee1d79a6"
)
EXPECTED_RESUMES = {
    "weak_strong": (
        "f0bac1a44be1394625f67568e46035cdf66556469c8ce2c8b2c1aef804d30ae8",
        "f803ea2b1d744cec09be9fea0333dec876dc568ef2c8f248d38435ecf4aa83c9",
    ),
    "intermediate_strong": (
        "c3c86df45c7547c5cd6dd19439aac36ce6354ef999f3c33e6c8a911b6caa7bcc",
        "313107ad03f2fd4e3d6bfd8dba20140845d2c961e747660edf6315b841544a29",
    ),
    "strong_strong_u8": (
        "3d4ceec0c0383537442cae6bd8d6c9b1d9d79339f90a7431ae282dfd45716710",
        "77d4e109956c56e869c0a20453f815a1934f3a2a33542267a3ae7dbd44d015c1",
    ),
}


def _json(root: Path, relative: str) -> dict:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _without_horizon(protocol: dict) -> dict:
    request = copy.deepcopy(protocol["request"])
    request["execution"]["stop"].pop("maximum_controller_rounds")
    return request


def test_page12_chtc_fork_preserves_exact_science_and_k50_resumes() -> None:
    manifest = _json(PACKAGE_DIR, "package_manifest.json")
    plan = _json(PACKAGE_DIR, "execution_plan.json")
    composition = _json(
        PACKAGE_DIR, manifest["runtime_source_composition"]["path"]
    )
    assert manifest["execution_target"] == "chtc"
    assert manifest["row_count"] == 3
    assert manifest["route_contract_sha256"] == ROUTE_SHA256
    assert plan["source_horizon"] == 50
    assert plan["target_horizon"] == 70
    assert plan["resume_rounds"] == {
        "weak_strong": 50,
        "intermediate_strong": 50,
        "strong_strong_u8": 50,
    }
    assert composition["base_archive"]["sha256"] == SOURCE_ARCHIVE_SHA256
    assert [row["after"]["sha256"] for row in composition["operational_overlays"]] == [
        CONTROLLER_SHA256,
        RESUME_SHA256,
    ]

    source_manifest = _json(LOCAL_SOURCE_DIR, "package_manifest.json")
    source_resumes = {
        row["regime_id"]: row for row in source_manifest["resume_inputs"]
    }
    for row in manifest["resume_inputs"]:
        archive_sha256, checkpoint_sha256 = EXPECTED_RESUMES[row["regime_id"]]
        assert row["archive"]["sha256"] == archive_sha256
        assert row["archive"] == source_resumes[row["regime_id"]]["archive"]
        resume = _json(PACKAGE_DIR, row["manifest"]["path"])
        receipt = _json(PACKAGE_DIR, row["checkpoint_validation"]["path"])
        assert resume["resume_round"] == 50
        assert resume["target_round"] == 70
        assert resume["checkpoint_sha256"] == checkpoint_sha256
        assert resume["pointer_closed"] is True
        assert receipt["status"] == "passed"
        assert receipt["metadata"]["strict_replay_passed"] is True
        assert receipt["metadata"]["route_contract_sha256"] == ROUTE_SHA256


def test_page12_chtc_fork_changes_protocol_only_at_horizon() -> None:
    manifest = _json(PACKAGE_DIR, "package_manifest.json")
    jobs = {
        row["execution_id"]: _json(PACKAGE_DIR, row["path"])
        for row in manifest["jobs"]
    }
    for row in manifest["protocols"]:
        target = _json(PACKAGE_DIR, row["path"])
        job = jobs[row["execution_id"]]
        source = _json(
            REPO_ROOT,
            (
                "chtc/paper_i_ra_adapt_repair_20260727/"
                "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_"
                "qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc/"
                "bundle_materialization/"
                "ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
                "phase23_no_lanes_cap24_tau1em4_r50_v1/protocols/"
                f"{job['source_execution_id']}.json"
            ),
        )
        assert source["horizon"] == 50
        assert target["horizon"] == 70
        assert source["route_contract"]["sha256"] == ROUTE_SHA256
        assert target["route_contract"]["sha256"] == ROUTE_SHA256
        assert _without_horizon(source) == _without_horizon(target)


def test_page12_chtc_queue_and_submit_are_safe_and_explicit() -> None:
    rows = [
        line.split("\t")
        for line in (PACKAGE_DIR / "queue.tsv").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(len(row) == 12 for row in rows)
    assert all(row[8:] == ["4", "90112", "102400", "259200"] for row in rows)

    submit = (PACKAGE_DIR / "submit.sub").read_text(encoding="utf-8")
    output = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
    assert f"transfer_output_files = {output}" in submit
    assert (
        f'transfer_output_remaps = "{output}=/staging/jsstrobel/'
        "paper_i_page12_strong_r70_continuations_20260811_v1/outputs/"
        '$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"'
        in submit
    )
    assert "leave_in_queue = False" in submit
    assert "periodic_release = False" in submit
    assert "max_materialize" not in submit
    assert "paper-i-page12-strong-r70-cont-v1" in submit
    assert "20260810_v2_local" not in submit
    assert "page10" not in submit.lower()


def test_page12_chtc_validator_and_worker_preflight_pass() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "validate_package.py"),
            "--worker-preflight",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed"
    assert receipt["row_count"] == 3
    assert receipt["worker_preflight_count"] == 3
    assert receipt["submitted"] is False

