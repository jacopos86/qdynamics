from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_average_singleton_plateau6_"
    "r70_fresh_20260801_v3_chtc"
)
EXPECTED_IDS = (
    "historical_average_v3_r70_fresh__weak_weak__nph3__ra_singleton_plateau",
    "historical_average_v3_r70_fresh__intermediate_weak__nph3__ra_singleton_plateau",
    "historical_average_v3_r70_fresh__strong_weak_u8__nph3__ra_singleton_plateau",
    "historical_average_v3_r70_fresh__weak_strong__nph7__ra_singleton_plateau",
    "historical_average_v3_r70_fresh__intermediate_strong__nph7__ra_singleton_plateau",
    "historical_average_v3_r70_fresh__strong_strong_u8__nph7__ra_singleton_plateau",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", *args],
        cwd=PACKAGE_DIR,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
        },
        text=True,
        capture_output=True,
        check=False,
    )


def test_exact_six_fresh_historical_average_singleton_cells_validate() -> None:
    completed = _run("validate_package.py", "--deep")
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed"
    assert receipt["row_count"] == 6
    assert tuple(receipt["execution_ids"]) == EXPECTED_IDS
    assert receipt["deep_preflight_count"] == 6
    assert receipt["global_source_count"] == 4
    assert receipt["execution_authorized"] is False
    assert receipt["submission_authorized"] is False

    manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
    assert tuple(manifest["execution_ids"]) == EXPECTED_IDS
    for execution_id in EXPECTED_IDS:
        job = json.loads(
            (PACKAGE_DIR / "jobs" / f"{execution_id}.json").read_text()
        )
        protocol = json.loads((PACKAGE_DIR / job["protocol_path"]).read_text())
        method = protocol["request"]["method"]
        invariants = protocol["route_contract"]["semantic_invariants"]
        assert protocol["horizon"] == 70
        assert protocol["request"]["execution"]["stop"][
            "maximum_controller_rounds"
        ] == 70
        assert protocol["request"]["execution"]["resume"] == {
            "kind": "fresh_start"
        }
        assert method["admission"]["kind"] == "singleton"
        assert method["insertion"]["kind"] == "plateau_commutation"
        assert method["pruning"]["kind"] == "off"
        assert method["beam"]["kind"] == "off"
        assert invariants[
            "plateau_prior_mean_decrease_ratio_threshold"
        ] == 1.0e-4
        assert invariants["experimental_insertion_policy"] == (
            "insertion_commutation_plateau_v2"
        )
        assert invariants["plateau_threshold_comparison"] == (
            "marginal_to_prior_mean_strictly_below_v2"
        )
        assert invariants["plateau_trigger_source"] == (
            "immediately_preceding_marginal_over_prior_mean_"
            "accepted_post_full_refit_energy_decrease_v2"
        )
        assert invariants["plateau_threshold_calibration_status"] == (
            "source_locked_counterfactual_trigger_replay_v2"
        )
        assert invariants["online_exact_reference_used"] is False
        assert job["fresh_start_contract"]["source_checkpoint"] is None


def test_clean_archive_enters_real_run_ra_adapt_for_one_round() -> None:
    completed = _run("validate_package.py", "--smoke-one-round")
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    smoke = receipt["real_one_round_smoke"]
    assert smoke["status"] == "passed_real_run_ra_adapt_one_round"
    assert smoke["controller_rounds_completed"] == 1
    assert smoke["fresh_start"] is True
    assert smoke["source_archive_import_isolated"] is True


def test_source_archive_is_exact_predecessor_plus_declared_v2_patch() -> None:
    spec = importlib.util.spec_from_file_location(
        "historical_average_package_contract",
        PACKAGE_DIR / "package_contract.py",
    )
    assert spec is not None and spec.loader is not None
    contract = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(contract)
    finally:
        sys.dont_write_bytecode = previous
    old_manifest = json.loads(
        (
            REPO_ROOT
            / contract.SOURCE_PACKAGE_RELATIVE
            / "source/source_archive_manifest.json"
        ).read_text()
    )
    new_manifest = json.loads(
        (PACKAGE_DIR / "source/source_archive_manifest.json").read_text()
    )
    old_rows = {row["path"]: row for row in old_manifest["members"]}
    new_rows = {row["path"]: row for row in new_manifest["members"]}
    bindings = {
        path: (before, after)
        for path, before, after in contract.SOURCE_PATCH_BINDINGS
    }
    assert set(new_rows) == set(old_rows)
    assert {
        path for path in new_rows if new_rows[path] != old_rows[path]
    } == set(bindings)
    for path, (before, after) in bindings.items():
        assert old_rows[path]["sha256"] == before
        assert new_rows[path]["sha256"] == after

    patch_headers = {
        line[6:]
        for line in (PACKAGE_DIR / "source_patch.diff")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("+++ b/")
    }
    assert patch_headers == set(bindings)


def test_worker_fails_closed_without_external_authorization(tmp_path: Path) -> None:
    execution_id = EXPECTED_IDS[0]
    output = tmp_path / "artifacts"
    receipt = tmp_path / "receipt.json"
    completed = _run(
        "run_cell.py",
        "--run",
        "--job",
        str(PACKAGE_DIR / "jobs" / f"{execution_id}.json"),
        "--execution-authorization",
        str(tmp_path / "absent_authorization.json"),
        "--output-dir",
        str(output),
        "--receipt",
        str(receipt),
    )
    assert completed.returncode == 2
    assert not output.exists()
    assert not receipt.exists()
