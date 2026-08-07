from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_"
    "r70_fresh_20260731_v1_chtc"
)
EXPECTED_IDS = (
    "cumulative_r70_fresh__weak_weak__nph3__ra_singleton_plateau",
    "cumulative_r70_fresh__intermediate_weak__nph3__ra_singleton_plateau",
    "cumulative_r70_fresh__strong_weak_u8__nph3__ra_singleton_plateau",
    "cumulative_r70_fresh__weak_strong__nph7__ra_singleton_plateau",
    "cumulative_r70_fresh__intermediate_strong__nph7__ra_singleton_plateau",
    "cumulative_r70_fresh__strong_strong_u8__nph7__ra_singleton_plateau",
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


def test_exact_six_fresh_cumulative_relative_singleton_cells_validate() -> None:
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
            "plateau_cumulative_decrease_ratio_threshold"
        ] == 1.0e-4
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
