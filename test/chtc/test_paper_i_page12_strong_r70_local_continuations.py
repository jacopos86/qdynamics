from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import ijson
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260810_v1_local"
)
ROUTE_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)


def _json(relative: str) -> dict:
    value = json.loads((PACKAGE_DIR / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _local_serial_module():
    spec = importlib.util.spec_from_file_location(
        "paper_i_page12_local_serial_test",
        PACKAGE_DIR / "local_serial.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package_contract_module():
    spec = importlib.util.spec_from_file_location(
        "paper_i_page12_package_contract_test",
        PACKAGE_DIR / "package_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(PACKAGE_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _validator_module():
    spec = importlib.util.spec_from_file_location(
        "paper_i_page12_validate_package_test",
        PACKAGE_DIR / "validate_package.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(PACKAGE_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_page12_validator_authenticates_controller_overlay_bytes(
    tmp_path: Path,
) -> None:
    validator = _validator_module()
    validator.PACKAGE_DIR = tmp_path
    relative = Path("source_overlay") / validator.CONTROLLER_RELATIVE_PATH
    overlay = tmp_path / relative
    overlay.parent.mkdir(parents=True)
    locked = b"source-locked-controller\n"
    overlay.write_bytes(locked)
    binding = {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(locked).hexdigest(),
        "size_bytes": len(locked),
    }
    validator._validate_controller_overlay_binding({"after": binding})
    overlay.write_bytes(b"drifted-controller\n")
    with pytest.raises(
        validator.PackageContractError,
        match="controller overlay bytes drifted",
    ):
        validator._validate_controller_overlay_binding({"after": binding})


def test_page12_preserved_weak_checkpoint_schema_projects_terminal_prefix() -> None:
    """Exercise the exact immutable Page-12 checkpoint that caught the drift."""

    package_contract = _package_contract_module()
    # The C backend keeps this exact 3.4-GB checkpoint regression bounded;
    # production uses the vendored parser with identical ijson event semantics.
    package_contract.streaming_json.parse = ijson.parse
    archive = PACKAGE_DIR / "resume_inputs/weak_strong.tar.gz"
    with tarfile.open(archive, "r:gz") as opened:
        checkpoint = opened.extractfile("resume/current.json")
        assert checkpoint is not None
        metadata = package_contract._checkpoint_metadata(checkpoint)
    assert metadata["active_prefix_checkpoint_count"] == 50
    assert metadata["accepted_prefix_terminal_energy"] == -1.1386283278209075
    assert metadata["accepted_prefix_terminal_state_fingerprint"] == (
        "projective_state_v1:"
        "695c75cb3812d6e718a5c3f3452ef1d2c5db744a3edeac5e6474a54901c9d6d4"
    )


def test_page12_continuation_is_local_only_three_row_horizon_change() -> None:
    manifest = _json("package_manifest.json")
    plan = _json("execution_plan.json")
    bundle = _json(manifest["bundle_manifest"]["path"])
    activation = _json("activation/activation_manifest.json")
    assert manifest["row_count"] == 3
    assert manifest["route_contract_sha256"] == ROUTE_SHA256
    assert manifest["execution_target"] == "local_mac_serial"
    assert manifest["max_concurrency"] == 1
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert plan["source_horizon"] == 50
    assert plan["target_horizon"] == 70
    assert plan["max_concurrency"] == 1
    assert bundle["only_scientific_change"] == {
        "path": "request.execution.stop.maximum_controller_rounds",
        "before": 50,
        "after": 70,
    }
    assert activation["execution_target"] == "local_mac_serial"
    assert activation["execution_authorized"] is True
    assert activation["submission_authorized"] is False
    assert activation["launch_ready"] is False
    assert not (PACKAGE_DIR / "submit.sub").exists()
    assert not (PACKAGE_DIR / "queue.tsv").exists()


def test_page12_continuation_has_three_pointer_closed_k50_inputs() -> None:
    manifest = _json("package_manifest.json")
    expected_checkpoints = {
        "weak_strong": (
            "f803ea2b1d744cec09be9fea0333dec876dc568ef2c8f248d38435ecf4aa83c9"
        ),
        "intermediate_strong": (
            "313107ad03f2fd4e3d6bfd8dba20140845d2c961e747660edf6315b841544a29"
        ),
        "strong_strong_u8": (
            "77d4e109956c56e869c0a20453f815a1934f3a2a33542267a3ae7dbd44d015c1"
        ),
    }
    for row in manifest["resume_inputs"]:
        regime = row["regime_id"]
        resume = _json(row["manifest"]["path"])
        receipt = _json(row["checkpoint_validation"]["path"])
        assert resume["resume_round"] == 50
        assert resume["target_round"] == 70
        assert resume["member_count"] == 3
        assert resume["pointer_closed"] is True
        assert resume["checkpoint_sha256"] == expected_checkpoints[regime]
        assert {member["role"] for member in resume["members"]} == {
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        }
        metadata = receipt["metadata"]
        assert metadata["checkpoint_depth"] == 50
        assert metadata["history_count"] == 50
        assert metadata["active_prefix_checkpoint_count"] == 50
        assert metadata["history_checkpoint_complete"] is True
        assert metadata["strict_replay_passed"] is True
        assert metadata["route_contract_sha256"] == ROUTE_SHA256
        assert receipt["accepted_state_resume_semantic_replay_required"] is True


def test_page12_local_launcher_gate_is_closed_and_never_submits() -> None:
    local_serial = _local_serial_module()
    gate = local_serial.validate_page13_completion_gate()
    assert gate["status"] == "passed_all_six_authenticated_and_refreshed"
    assert gate["round50_closure_count"] == 6
    assert local_serial.MAX_CONCURRENCY == 1
    source = (PACKAGE_DIR / "local_serial.py").read_text(encoding="utf-8")
    assert "condor_submit" not in source
    assert "subprocess.Popen" not in source
    assert 'subprocess.run(' in source


def test_page12_local_package_validator_and_worker_preflight_pass() -> None:
    completed = subprocess.run(
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
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed"
    assert receipt["row_count"] == 3
    assert receipt["worker_preflight_count"] == 3
    assert receipt["submission_authorized"] is False
    assert receipt["scientific_execution_performed"] is False
