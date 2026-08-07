from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc"
    / "paper_i_ra_adapt_repair_20260727"
    / "stationary_core_ra36_r70_continuation_20260731_v1_chtc"
)


def _contract():
    path = PACKAGE_DIR / "package_contract.py"
    spec = importlib.util.spec_from_file_location(
        "stationary_core_ra36_r70_contract_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(name: str) -> dict:
    value = json.loads(
        (PACKAGE_DIR / name).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_ra36_plan_is_27_authenticated_resumes_plus_9_blocked_fresh() -> None:
    contract = _contract()
    provenance = contract.load_json(
        REPO_ROOT / contract.SOURCE_REPORT_RELATIVE,
        label="stationary-core provenance",
    )
    rows = contract.planned_rows(
        repo_root=REPO_ROOT, provenance=provenance
    )
    resumes = [
        row
        for row in rows
        if row["execution_mode"]
        == "authenticated_resume_50_to_70"
    ]
    fresh = [
        row
        for row in rows
        if row["execution_mode"] == "fresh_0_to_70"
    ]

    assert len(rows) == 36
    assert len(resumes) == 27
    assert len(fresh) == 9
    assert len({row["execution_id"] for row in rows}) == 36
    assert {
        (row["regime_id"], row["nph"], row["route_id"])
        for row in rows
    } == {
        (regime, nph, route)
        for regime, nph in contract.REGIME_CUTOFF_PAIRS
        for route in contract.ROUTE_IDS
    }
    assert all(
        row["source_horizon"] == 50
        and row["target_horizon"] == 70
        and row["active_gradient_policy"]
        == "stationary_source_response_v1"
        and row["resource_weighting_scope"]
        == "late_resource_weighting_v1"
        for row in rows
    )
    assert all(
        row["collision_status"]
        == "blocked_live_r50_predecessor"
        and row["collision"]["cluster_id"] == 9397758
        and row["collision"]["proc_id"] == index
        and row["route_id"].endswith("_always")
        for index, row in enumerate(fresh)
    )
    assert not (PACKAGE_DIR / "submit.sub").exists()
    assert not (PACKAGE_DIR / "authority").exists()


def test_materialized_package_passes_metadata_validation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_DIR / "validate_package.py"),
            "--metadata-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed_inert_collision_blocked"
    assert result["cell_count"] == 36
    assert result["authenticated_resume_count"] == 27
    assert result["fresh_count"] == 9
    assert result["collision_cluster_id"] == 9397758
    assert result["collision_proc_ids"] == list(range(9))
    assert result["submission_ready"] is False
    assert result["submitted"] is False


def test_resume_archives_are_terminal_pointer_closed_triplets() -> None:
    manifest = _json("resume_inputs_manifest.json")
    cells = manifest["cells"]

    assert manifest["resume_cell_count"] == 27
    assert len(cells) == 27
    for execution_id, row in cells.items():
        assert execution_id.endswith("__r70")
        assert row["member_count"] == 3
        assert row["pointer_closed"] is True
        assert row["superseded_sidecars_retained"] is False
        assert {member["role"] for member in row["members"]} == {
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        }
        assert all(
            "superseded" not in member["path"]
            and "superseded" not in member["source_member"]
            for member in row["members"]
        )
        assert row["authentication"]["checkpoint_depth"] == 50
        assert row["authentication"]["history_count"] == 50
        assert (
            row["authentication"][
                "active_prefix_checkpoint_count"
            ]
            == 50
        )
        assert (
            row["authentication"]["history_checkpoint_complete"]
            is True
        )
        assert row["authentication"]["strict_replay_passed"] is True


def test_each_exact_source_family_derives_only_the_r70_horizon() -> None:
    jobs = [
        next(
            path
            for path in sorted((PACKAGE_DIR / "jobs").glob("*.json"))
            if json.loads(path.read_text(encoding="utf-8"))[
                "source_family"
            ]
            == family
        )
        for family in (
            "stationary_core_v11",
            "always_factorial_v1",
            "always_factorial_v2",
        )
    ]
    script = r"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

package = Path(sys.argv[1]).resolve()
job_path = Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location(
    "stationary_core_r70_runner_preflight",
    package / "run_cell.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
job = module._load_runtime_job(job_path)
with tempfile.TemporaryDirectory(prefix="r70-source-preflight.") as raw:
    source = Path(raw) / "source"
    module._extract_source(job, source)
    module._activate_source_root(source)
    protocol, _problem, delta = module._derived_protocol(
        job=job, source_root=source
    )
    print(json.dumps({
        "family": job["source_family"],
        "horizon": protocol.horizon,
        "gradient": protocol.active_gradient_policy,
        "weighting": protocol.resource_weighting_scope,
        "changed_paths": delta["changed_paths"],
        "non_swept_settings_diff": delta[
            "non_swept_settings_diff"
        ],
    }, sort_keys=True))
"""
    observed = []
    for job in jobs:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(PACKAGE_DIR),
                str(job),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr
        observed.append(json.loads(completed.stdout))

    assert {row["family"] for row in observed} == {
        "stationary_core_v11",
        "always_factorial_v1",
        "always_factorial_v2",
    }
    assert all(
        row["horizon"] == 70
        and row["gradient"] == "stationary_source_response_v1"
        and row["weighting"] == "late_resource_weighting_v1"
        and row["changed_paths"]
        == [
            "horizon",
            "request.execution.stop.maximum_controller_rounds",
            "sha256",
            "stopping_rule.maximum_controller_rounds",
        ]
        and row["non_swept_settings_diff"] == []
        for row in observed
    )


def test_fresh_rows_fail_before_authorization_or_execution(
    tmp_path: Path,
) -> None:
    fresh_job = next(
        path
        for path in sorted((PACKAGE_DIR / "jobs").glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8"))[
            "execution_mode"
        ]
        == "fresh_0_to_70"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_DIR / "run_cell.py"),
            "--job",
            str(fresh_job),
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
        timeout=180,
    )
    assert completed.returncode == 2
    assert "blocked by its live exact r50 predecessor 9397758." in (
        completed.stderr
    )
    assert "no execution or supersession is permitted" in (
        completed.stderr
    )
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "receipt.json").exists()
