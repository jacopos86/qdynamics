from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260810_v2_local"
)
V1_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260810_v1_local"
)
ROUTE_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)
CONTROLLER_SHA256 = (
    "e25c0281373b828f75200410aa0e5364eaebe5a78f517421bc8c7bdc73c20327"
)
RESUME_SHA256 = (
    "00a06606cf69dce5ee749172839b2115a5b5bb7dce72b170fc58d792ee1d79a6"
)


def _json(root: Path, relative: str) -> dict:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_horizon(protocol: dict) -> dict:
    request = copy.deepcopy(protocol["request"])
    request["execution"]["stop"].pop("maximum_controller_rounds")
    return request


def test_v2_is_local_only_and_changes_only_horizon() -> None:
    manifest = _json(PACKAGE_DIR, "package_manifest.json")
    plan = _json(PACKAGE_DIR, "execution_plan.json")
    bundle = _json(PACKAGE_DIR, manifest["bundle_manifest"]["path"])
    assert manifest["package_id"].endswith("20260810_v2_local")
    assert manifest["execution_target"] == "local_mac_serial"
    assert manifest["row_count"] == 3
    assert manifest["max_concurrency"] == 1
    assert manifest["route_contract_sha256"] == ROUTE_SHA256
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert manifest["remote_stage"] is False
    assert manifest["condor_submit"] is False
    assert plan["source_horizon"] == 50
    assert plan["target_horizon"] == 70
    assert plan["max_concurrency"] == 1
    assert bundle["only_scientific_change"] == {
        "path": "request.execution.stop.maximum_controller_rounds",
        "before": 50,
        "after": 70,
    }
    assert not (PACKAGE_DIR / "submit.sub").exists()
    assert not (PACKAGE_DIR / "queue.tsv").exists()


def test_v2_binds_exact_two_operational_source_overlays() -> None:
    manifest = _json(PACKAGE_DIR, "package_manifest.json")
    composition = _json(
        PACKAGE_DIR, manifest["runtime_source_composition"]["path"]
    )
    overlays = composition["operational_overlays"]
    assert [row["repair_id"] for row in overlays] == [
        "accepted_energy_roundoff_only_128ulp_v1",
        "phase0_gradient_screen_resume_closure_v1",
    ]
    assert [row["after"]["sha256"] for row in overlays] == [
        CONTROLLER_SHA256,
        RESUME_SHA256,
    ]
    for row in overlays:
        assert row["scientific_protocol_changed"] is False
        assert row["scientific_settings_changed"] == []
        path = PACKAGE_DIR / row["after"]["path"]
        assert not path.is_symlink()
        assert _sha256(path) == row["after"]["sha256"]
    assert overlays[1]["actual_page12_weak_snapshot_hydration_passed"] is True
    assert overlays[1]["actual_snapshot_controller_round"] == 50
    assert overlays[1]["actual_snapshot_route_contract_sha256"] == ROUTE_SHA256


def test_v2_reuses_exact_v1_compact_resume_archives() -> None:
    manifest = _json(PACKAGE_DIR, "package_manifest.json")
    v1_manifest = _json(V1_PACKAGE_DIR, "package_manifest.json")
    v1_by_regime = {
        row["regime_id"]: row for row in v1_manifest["resume_inputs"]
    }
    for row in manifest["resume_inputs"]:
        regime = row["regime_id"]
        v2_resume = _json(PACKAGE_DIR, row["manifest"]["path"])
        v2_receipt = _json(
            PACKAGE_DIR, row["checkpoint_validation"]["path"]
        )
        v1_row = v1_by_regime[regime]
        assert row["archive"]["sha256"] == v1_row["archive"]["sha256"]
        assert row["archive"]["size_bytes"] == v1_row["archive"]["size_bytes"]
        assert _sha256(PACKAGE_DIR / row["archive"]["path"]) == _sha256(
            V1_PACKAGE_DIR / v1_row["archive"]["path"]
        )
        inherited = v2_resume["inherited_v1_authority"]
        assert inherited == v2_receipt["inherited_v1_authority"]
        assert inherited["resume_manifest"] == v1_row["manifest"]
        assert inherited["checkpoint_validation"] == v1_row[
            "checkpoint_validation"
        ]
        assert inherited["archive_byte_identity_preserved"] is True
        assert inherited["member_validation_inherited"] is True
        assert inherited["checkpoint_stream_validation_inherited"] is True
        assert v2_receipt["metadata"]["history_count"] == 50
        assert v2_receipt["metadata"]["strict_replay_passed"] is True
        assert v2_receipt["metadata"]["route_contract_sha256"] == ROUTE_SHA256


def test_v2_protocols_preserve_page12_route_except_round_70_horizon() -> None:
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


def test_v2_validator_worker_preflight_and_local_serial_gate_pass() -> None:
    validation = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "validate_package.py"),
            "--worker-preflight",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert validation.returncode == 0, validation.stderr
    receipt = json.loads(validation.stdout)
    assert receipt["status"] == "passed"
    assert receipt["worker_preflight_count"] == 3
    assert receipt["scientific_execution_performed"] is False
    source = (PACKAGE_DIR / "local_serial.py").read_text(encoding="utf-8")
    assert "condor_submit" not in source
    assert "subprocess.Popen" not in source
    assert "MAX_CONCURRENCY = 1" in source
