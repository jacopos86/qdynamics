from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page10_strong_holstein_r70_accepted_continuations_"
    "20260809_v2_chtc"
)
EXPECTED_IDS = [
    (
        "page10_r70_resume__weak_strong__nph7__"
        "ra_macro_then_singleton_phase123_qiskit_phase23_plateau"
    ),
    (
        "page10_r70_resume__intermediate_strong__nph7__"
        "ra_macro_then_singleton_phase123_qiskit_phase23_plateau"
    ),
    (
        "page10_r70_resume__strong_strong_u8__nph7__"
        "ra_macro_then_singleton_phase123_qiskit_phase23_plateau"
    ),
]
ROUTE_SHA256 = (
    "83b5e5cb17bdfbfc8e8efb22a586d952b3343f430de15ffb58550082d17e3cf0"
)


def _json(relative: str) -> dict:
    payload = json.loads((PACKAGE_DIR / relative).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_page10_continuation_is_exactly_three_authenticated_prefixes() -> None:
    manifest = _json("package_manifest.json")
    assert manifest["row_count"] == 3
    assert manifest["execution_ids"] == EXPECTED_IDS
    assert manifest["route_contract_sha256"] == ROUTE_SHA256
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert manifest["remote_stage"] is False
    assert manifest["condor_submit"] is False

    expected_rounds = [56, 51, 50]
    for row, expected_round, execution in zip(
        manifest["resume_inputs"], expected_rounds, EXPECTED_IDS, strict=True
    ):
        resume = _json(row["manifest"]["path"])
        checkpoint_validation = _json(row["checkpoint_validation"]["path"])
        assert resume["resume_round"] == expected_round
        assert resume["target_round"] == 70
        assert resume["member_count"] == 3
        assert resume["pointer_closed"] is True
        roles = {member["role"] for member in resume["members"]}
        assert roles == {
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        }
        assert (
            checkpoint_validation["validation_authority"]
            == "inherited_v1_full_stream_validation_exact_bytes_v1"
        )
        assert checkpoint_validation["archive"] == row["archive"]
        assert checkpoint_validation["members"] == resume["members"]
        assert (
            checkpoint_validation["metadata"]["checkpoint_depth"]
            == expected_round
        )
        assert (
            checkpoint_validation["metadata"]["history_count"]
            == expected_round
        )
        assert (
            checkpoint_validation["metadata"]["route_contract_sha256"]
            == ROUTE_SHA256
        )
        assert (
            checkpoint_validation["worker_validation_scope"]
            == "stream_authenticate_all_three_members_then_strict_resume_replay_v1"
        )
        assert (
            checkpoint_validation[
                "accepted_state_resume_semantic_replay_required"
            ]
            is True
        )
        assert checkpoint_validation["ambient_ijson_required"] is False
        job = _json(f"jobs/{execution}.json")
        assert job["resume_round"] == expected_round
        assert job["checkpoint_sha256"] == resume["checkpoint_sha256"]
        assert job["checkpoint_validation"] == row["checkpoint_validation"]


def test_page10_continuation_preserves_route_and_changes_only_horizon() -> None:
    manifest = _json("package_manifest.json")
    bundle = _json(manifest["bundle_manifest"]["path"])
    composition = _json(manifest["runtime_source_composition"]["path"])
    assert bundle["only_scientific_change"] == {
        "path": "request.execution.stop.maximum_controller_rounds",
        "before": 50,
        "after": 70,
    }
    assert bundle["route_contract_sha256"] == ROUTE_SHA256
    overlay = composition["operational_overlay"]
    assert overlay["semantic_scope"] == "accepted_energy_roundoff_only"
    assert overlay["all_non_energy_fields_exact"] is True
    assert overlay["scientific_protocol_changed"] is False
    assert overlay["scientific_settings_changed"] == []
    assert overlay["absolute_tolerance"] == (
        "128*ulp(max(1,abs(E1),abs(E2)))"
    )
    for execution in EXPECTED_IDS:
        job = _json(f"jobs/{execution}.json")
        protocol = _json(job["protocol"]["path"])
        assert job["resources"] == {
            "request_cpus": 4,
            "request_memory_mb": 32768,
            "request_disk_mb": 61440,
            "max_runtime_seconds": 259200,
            "basis": (
                "page10_observed_peak_memory_4883_7325_12208_mib_"
                "plus_checkpoint_hydration_headroom_v1"
            ),
        }
        assert protocol["horizon"] == 70
        assert protocol["route_contract"]["sha256"] == ROUTE_SHA256
        assert protocol["request"]["execution"]["stop"] == {
            "maximum_controller_rounds": 70
        }


def test_page10_continuation_submit_has_three_unique_posix_outputs() -> None:
    rows = [
        line.split("\t")
        for line in (PACKAGE_DIR / "queue.tsv").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(len(row) == 12 for row in rows)
    assert [row[0] for row in rows] == EXPECTED_IDS
    assert all(row[8:] == ["4", "32768", "61440", "259200"] for row in rows)
    submit = (PACKAGE_DIR / "submit.sub").read_text(encoding="utf-8")
    output = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
    assert f"transfer_output_files = {output}" in submit
    assert (
        f'transfer_output_remaps = "{output}=/staging/jsstrobel/'
        "paper_i_page10_strong_r70_continuations_20260809_v2/outputs/"
        '$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"'
        in submit
    )
    assert "periodic_release = False" in submit
    assert "paper-i-page10-strong-r70-cont-v2" in submit
    assert (
        "paper_i_page10_strong_holstein_r70_accepted_continuations_"
        "20260809_v1_chtc"
        not in submit
    )
    assert "paper-i-page10-strong-r70-cont-v1" not in submit
    assert (
        "paper_i_page10_strong_r70_continuations_20260809_v1/outputs"
        not in submit
    )


def test_page10_continuation_parser_is_vendored_and_ambient_free() -> None:
    script = r"""
from io import BytesIO
from pathlib import Path
import sys

package = Path(sys.argv[1]).resolve()
sys.path.insert(0, package.as_posix())
import package_contract

assert package_contract.streaming_json.backend == "python"
events = list(
    package_contract.streaming_json.parse(
        BytesIO(b'{"checkpoint":{"depth":50}}')
    )
)
assert ("checkpoint.depth", "number", 50) in events
assert "ijson" not in sys.modules
print("passed")
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-B", "-c", script, str(PACKAGE_DIR)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "passed"


def test_page10_continuation_validator_and_worker_preflight_pass() -> None:
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
