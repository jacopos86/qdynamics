from __future__ import annotations

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

from pair_contract import (
    AUTHORITATIVE_JOB_SHA256,
    BUNDLE_ID,
    CHECKPOINT_EXIT_CODE,
    CONTROL_MODE,
    HISTORICAL_REPAIRED_FAILURE,
    OUTER_PROFILE,
    PAIR_ID,
    REUSE_MODE,
    RUNTIME_ROOT,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_AUDIT_SHA256,
    bundle_dir,
    checkpoint_archive_path,
    checkpoint_resume_dir,
    dump_json,
    inspect_source_archive,
    load_json,
    pack_resume_checkpoint,
    restore_resume_checkpoint,
    safe_extract_source,
    sha256,
    validate_job,
    validate_pair_diff,
    validate_resume_current,
    validate_source_lock,
)
from run_job import build_effective_command
from evidence_validation import validate_resumed_segment_horizon


BUNDLE = bundle_dir()


@pytest.fixture(scope="module")
def jobs() -> tuple[dict, dict]:
    return load_json(BUNDLE / "jobs/control.json"), load_json(BUNDLE / "jobs/reuse.json")


def write_partial_checkpoint(
    root: Path,
    job: dict,
    *,
    source_round: int = 7,
) -> tuple[Path, Path, Path]:
    mode = str(job["pair_contract"]["mode"])
    manifest_path = root / "jobs" / f"{mode}.json"
    dump_json(manifest_path, job)
    current_path = root / job["paths"]["current_json"]
    ledger_name = "current.estimator_call_ledger_checkpoint.unit.json"
    ledger_path = current_path.with_name(ledger_name)
    immutable_ids = ["primitive-unit"] if mode == REUSE_MODE else []
    ledger = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v1",
        "checkpoint": {
            "depth": source_round,
            "reason": "iteration_done",
            "current_round_finalized": True,
        },
        "ledger": {
            "schema": "estimator_call_ledger_v1",
            "entries": [
                {"primitive_id": value} for value in immutable_ids
            ],
        },
        "ledger_fingerprint": "ledger-unit",
        "unique_primitive_count": len(immutable_ids),
        "raw_occurrence_count": source_round,
        "S_alg": source_round,
        "no_credentials_serialized": True,
    }
    dump_json(ledger_path, ledger)
    pointer = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v1",
        "enabled": True,
        "status": "complete",
        "current_round_finalized": True,
        "checkpoint_depth": source_round,
        "checkpoint_reason": "iteration_done",
        "path": ledger_name,
        "sha256": sha256(ledger_path),
        "ledger_schema": "estimator_call_ledger_v1",
        "ledger_fingerprint": "ledger-unit",
        "unique_primitive_count": len(immutable_ids),
        "raw_occurrence_count": source_round,
        "S_alg": source_round,
    }
    composition = {
        "route_family": "singleton_response_snake",
        "phase1_prune_enabled": False,
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "structural_rollback_enabled": False,
        "sr_controller_contract_sha256": job["route_identity"][
            "profile_contract_sha256"
        ],
    }
    adapt = {
        "checkpoint_reason": "iteration_done",
        "partial_checkpoint": True,
        "adapt_beam_enabled": False,
        "branch_id": None,
        "parent_branch_id": None,
        "stop_reason": None,
        "history_checkpoint_complete": True,
        "history_count": source_round,
        "history_tail_count": source_round,
        "history": [{"depth": value} for value in range(1, source_round + 1)],
        "history_tail": [{"depth": value} for value in range(1, source_round + 1)],
        "ansatz_depth": source_round,
        "estimator_call_ledger_checkpoint": pointer,
    }
    if mode == REUSE_MODE:
        outer = {
            "schema": "formal_manifold_outer_information_checkpoint_v2",
            "config": {},
            "stage": "idle",
            "committed": None,
            "pending": None,
            "pending_source": None,
            "failed_closure": None,
            "immutable_primitive_ids": immutable_ids,
            "immutable_observation_ids": [],
            "invalidations": [],
        }
        from pair_contract import digest_jsonable

        outer["checkpoint_sha256"] = digest_jsonable(outer)
        adapt.update(
            {
                "sr_outer_information_checkpoint": outer,
                "formal_manifold_outer_information_checkpoint": copy.deepcopy(outer),
                "sr_outer_information_resume_policy": (
                    "validated_outer_geometry_transport_restore_v1"
                ),
                "formal_manifold_outer_information_resume_policy": (
                    "validated_outer_geometry_transport_restore_v1"
                ),
            }
        )
    current = {
        "schema_version": "static_adapt_current_checkpoint_v1",
        "no_credentials_serialized": True,
        "settings": {
            "problem": "hh",
            "L": 2,
            "u": 0.25,
            "g_ep": 0.790569415042,
            "t": 1.0,
            "omega0": 1.0,
            "n_ph_max": 7,
            "route_family": "singleton_response_snake",
            "sr_route_profile_request": job["route_identity"]["profile_request"],
            "sr_route_profile_resolved": job["route_identity"]["profile_resolved"],
            "sr_route_profile_contract_sha256": job["route_identity"][
                "profile_contract_sha256"
            ],
            "formal_manifold_route_profile": (
                "off" if mode == CONTROL_MODE else OUTER_PROFILE
            ),
            "formal_manifold_route_composition": composition,
        },
        "adapt_vqe": adapt,
        "checkpoint": {
            "complete": False,
            "depth": source_round,
            "ansatz_depth": source_round,
            "reason": "iteration_done",
            "branch_id": None,
            "parent_branch_id": None,
            "stop_reason": None,
            "estimator_call_ledger_checkpoint": pointer,
        },
    }
    dump_json(current_path, current)
    return manifest_path, current_path, ledger_path


def test_authoritative_source_hashes_and_fixed_runtime_root() -> None:
    assert sha256(BUNDLE / "authoritative_weak_strong_job_lock.json") == AUTHORITATIVE_JOB_SHA256
    assert sha256(BUNDLE / "source_locked_v5r2.tar.gz") == SOURCE_ARCHIVE_SHA256
    assert sha256(BUNDLE / "source_lock_audit_v5r2.json") == SOURCE_AUDIT_SHA256
    inventory = validate_source_lock(BUNDLE)
    assert inventory["runtime_root"] == RUNTIME_ROOT
    assert inventory["file_count"] > 100
    assert inventory["appledouble_ignored_count"] >= 0


def test_exact_control_to_reuse_scientific_diff(jobs: tuple[dict, dict]) -> None:
    control, reuse = jobs
    validate_job(control, expected_mode=CONTROL_MODE)
    validate_job(reuse, expected_mode=REUSE_MODE)
    assert validate_pair_diff(control, reuse) == [
        "adapt_formal_manifold_route_profile"
    ]
    assert OUTER_PROFILE in reuse["command"]["argv"]
    assert OUTER_PROFILE not in control["command"]["argv"]


def test_modes_have_disjoint_outputs_and_cold_caches(jobs: tuple[dict, dict]) -> None:
    control, reuse = jobs
    assert control["paths"]["output_root"] != reuse["paths"]["output_root"]
    control_caches = {
        value
        for key, value in control["environment"].items()
        if key.endswith("_CACHE_DIR")
    }
    reuse_caches = {
        value
        for key, value in reuse["environment"].items()
        if key.endswith("_CACHE_DIR")
    }
    assert control_caches.isdisjoint(reuse_caches)


@pytest.mark.parametrize("mode", [CONTROL_MODE, REUSE_MODE])
def test_evicted_partial_restores_beyond_completed_round(
    jobs: tuple[dict, dict],
    tmp_path: Path,
    mode: str,
) -> None:
    job = copy.deepcopy(jobs[0] if mode == CONTROL_MODE else jobs[1])
    manifest_path, current_path, ledger_path = write_partial_checkpoint(
        tmp_path,
        job,
        source_round=7,
    )
    validation = validate_resume_current(job, current_path, ledger_path=ledger_path)
    assert validation["source_controller_round"] == 7
    cache_file = tmp_path / job["paths"]["output_root"] / "cache" / "stale.bin"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(b"stale")
    packed = pack_resume_checkpoint(mode, manifest_path, work_root=tmp_path)
    assert packed["resume_available"] is True
    assert packed["source_controller_round"] == 7
    restored = restore_resume_checkpoint(mode, manifest_path, work_root=tmp_path)
    assert restored["resume_available"] is True
    assert restored["source_controller_round"] == 7
    assert not cache_file.exists()
    restored_current = Path(restored["resume_current_json"])
    assert restored_current == checkpoint_resume_dir(tmp_path, mode) / "current.json"
    effective, provenance = build_effective_command(
        job,
        restored_current,
        work_root_override=tmp_path,
    )
    assert effective[: len(job["command"]["argv"])] == job["command"]["argv"]
    assert provenance["active"] is True
    assert provenance["source_controller_round"] == 7
    assert provenance["target_controller_round"] == 50
    assert effective[-8:] == [
        "--adapt-resume-scaffold-json",
        str(restored_current),
        "--adapt-resume-mode",
        "scaffold_v1",
        "--adapt-resume-boundary-refit-policy",
        "verified_checkpoint_no_refit_v1",
        "--adapt-resume-compile-smoke",
        "off",
    ]


def test_repeated_eviction_preserves_last_completed_round(
    jobs: tuple[dict, dict], tmp_path: Path
) -> None:
    reuse = copy.deepcopy(jobs[1])
    manifest_path, _, _ = write_partial_checkpoint(
        tmp_path,
        reuse,
        source_round=7,
    )
    pack_resume_checkpoint(REUSE_MODE, manifest_path, work_root=tmp_path)
    first = restore_resume_checkpoint(REUSE_MODE, manifest_path, work_root=tmp_path)
    assert first["source_controller_round"] == 7
    # No resumed round has completed.  A second eviction must not replace r7
    # with a round-zero sentinel merely because the fresh output root is empty.
    second_pack = pack_resume_checkpoint(REUSE_MODE, manifest_path, work_root=tmp_path)
    assert second_pack["resume_available"] is True
    assert second_pack["source_controller_round"] == 7
    second = restore_resume_checkpoint(REUSE_MODE, manifest_path, work_root=tmp_path)
    assert second["source_controller_round"] == 7


def test_reuse_resume_accepts_prediction_stage_and_rejects_stage_drift(
    jobs: tuple[dict, dict], tmp_path: Path
) -> None:
    reuse = copy.deepcopy(jobs[1])
    _, current_path, ledger_path = write_partial_checkpoint(
        tmp_path,
        reuse,
        source_round=7,
    )
    current = load_json(current_path)
    outer = current["adapt_vqe"]["sr_outer_information_checkpoint"]
    outer.pop("checkpoint_sha256")
    outer["stage"] = "refit_prediction"
    outer["pending"] = {"geometry_status": "predicted"}
    outer["pending_source"] = {"geometry_status": "acquired"}
    from pair_contract import digest_jsonable

    outer["checkpoint_sha256"] = digest_jsonable(outer)
    current["adapt_vqe"]["formal_manifold_outer_information_checkpoint"] = (
        copy.deepcopy(outer)
    )
    dump_json(current_path, current)
    assert validate_resume_current(
        reuse,
        current_path,
        ledger_path=ledger_path,
    )["source_controller_round"] == 7

    current = load_json(current_path)
    outer = current["adapt_vqe"]["sr_outer_information_checkpoint"]
    outer.pop("checkpoint_sha256")
    outer["stage"] = "unknown_stage"
    outer["checkpoint_sha256"] = digest_jsonable(outer)
    current["adapt_vqe"]["formal_manifold_outer_information_checkpoint"] = (
        copy.deepcopy(outer)
    )
    dump_json(current_path, current)
    with pytest.raises(ValueError, match="stage-consistent"):
        validate_resume_current(reuse, current_path, ledger_path=ledger_path)


def test_checkpoint_cold_sentinel_and_cross_mode_fail_closed(
    jobs: tuple[dict, dict], tmp_path: Path
) -> None:
    control, reuse = (copy.deepcopy(jobs[0]), copy.deepcopy(jobs[1]))
    control_manifest = tmp_path / "jobs/control.json"
    reuse_manifest = tmp_path / "jobs/reuse.json"
    dump_json(control_manifest, control)
    dump_json(reuse_manifest, reuse)
    cold = pack_resume_checkpoint(CONTROL_MODE, control_manifest, work_root=tmp_path)
    assert cold["resume_available"] is False
    restored_cold = restore_resume_checkpoint(
        CONTROL_MODE,
        control_manifest,
        work_root=tmp_path,
    )
    assert restored_cold["resume_available"] is False
    assert restored_cold["source_controller_round"] == 0
    assert restored_cold["resume_current_json"] is None

    # A control checkpoint copied into the reuse filename cannot cross the
    # cold mode boundary because its envelope and job hash remain control-bound.
    shutil.copyfile(
        checkpoint_archive_path(tmp_path, CONTROL_MODE),
        checkpoint_archive_path(tmp_path, REUSE_MODE),
    )
    with pytest.raises(ValueError, match="checkpoint envelope drift"):
        restore_resume_checkpoint(REUSE_MODE, reuse_manifest, work_root=tmp_path)


def test_malformed_partial_and_arbitrary_resume_path_fail_closed(
    jobs: tuple[dict, dict], tmp_path: Path
) -> None:
    control = copy.deepcopy(jobs[0])
    manifest_path, current_path, _ = write_partial_checkpoint(
        tmp_path,
        control,
        source_round=7,
    )
    current = load_json(current_path)
    current["settings"]["u"] = 8.0
    dump_json(current_path, current)
    with pytest.raises(ValueError, match="resume physics drift"):
        pack_resume_checkpoint(CONTROL_MODE, manifest_path, work_root=tmp_path)
    with pytest.raises(ValueError, match="mode-private"):
        build_effective_command(
            control,
            current_path,
            work_root_override=tmp_path,
        )


def test_resumed_segment_horizon_closes_cumulatively() -> None:
    closure = validate_resumed_segment_horizon(
        {
            "source_controller_round": 17,
            "final_controller_round": 50,
            "new_admission_records": 33,
            "max_new_admissions": 50,
        },
        target_round=50,
        target_new_admissions=50,
    )
    assert closure == {
        "source_controller_round": 17,
        "segment_new_admissions": 33,
    }
    with pytest.raises(ValueError, match="does not close"):
        validate_resumed_segment_horizon(
            {
                "source_controller_round": 17,
                "final_controller_round": 50,
                "new_admission_records": 32,
                "max_new_admissions": 50,
            },
            target_round=50,
            target_new_admissions=50,
        )


def test_mutated_cutoff_profile_and_seed_fail_closed(jobs: tuple[dict, dict]) -> None:
    control, _ = jobs
    for mutate in ("cutoff", "profile", "seed"):
        bad = copy.deepcopy(control)
        if mutate == "cutoff":
            index = bad["command"]["argv"].index("--n-ph-max")
            bad["command"]["argv"][index + 1] = "3"
        elif mutate == "profile":
            bad["route_identity"]["profile_resolved"] = "drifted"
        else:
            bad["route_identity"]["profile_contract"]["execution_settings"][
                "adapt_seed"
            ] = 8
        with pytest.raises(ValueError):
            validate_job(bad, expected_mode=CONTROL_MODE)


def test_reuse_requires_current_control_gate(jobs: tuple[dict, dict], tmp_path: Path) -> None:
    _, reuse = jobs
    with pytest.raises(ValueError, match="control gate"):
        validate_job(
            reuse,
            expected_mode=REUSE_MODE,
            require_anchor=True,
            work_root=tmp_path,
        )
    gate_path = tmp_path / reuse["pair_contract"]["control_gate_path"]
    dump_json(
        gate_path,
        {
            "schema": "paper_i_sr_outer_information_control_gate_v1",
            "status": "pass",
            "pair_id": PAIR_ID,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "control_job_manifest_sha256": reuse["pair_contract"][
                "expected_control_job_manifest_sha256"
            ],
        },
    )
    validate_job(
        reuse,
        expected_mode=REUSE_MODE,
        require_anchor=True,
        work_root=tmp_path,
    )
    gate = load_json(gate_path)
    gate["source_archive_sha256"] = "0" * 64
    dump_json(gate_path, gate)
    with pytest.raises(ValueError, match="stale or incompatible"):
        validate_job(
            reuse,
            expected_mode=REUSE_MODE,
            require_anchor=True,
            work_root=tmp_path,
        )


def test_unsafe_source_archive_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../escape.py")
        payload = b"pass\n"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inspect_source_archive(archive)


def test_archive_only_profile_import_from_nested_root(tmp_path: Path) -> None:
    runtime = safe_extract_source(BUNDLE / "source_locked_v5r2.tar.gz", tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(runtime),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    command = [
        sys.executable,
        "-c",
        (
            "from pipelines.static_adapt.formal_manifold_route_profile import "
            "FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES; "
            f"assert {OUTER_PROFILE!r} in FORMAL_MANIFOLD_ROUTE_PROFILE_CHOICES"
        ),
    ]
    completed = subprocess.run(command, cwd=runtime, env=env, check=False)
    assert completed.returncode == 0


def test_dag_is_exactly_control_post_gate_then_reuse() -> None:
    text = (BUNDLE / "pair.dag").read_text(encoding="utf-8")
    job_lines = [line for line in text.splitlines() if line.startswith("JOB ")]
    assert len(job_lines) == 2
    assert "JOB CONTROL" in job_lines[0]
    assert "JOB REUSE" in job_lines[1]
    assert "SCRIPT POST CONTROL" in text
    assert "PARENT CONTROL CHILD REUSE" in text
    assert "RETRY" not in text


@pytest.mark.parametrize("name", ["submit_control.sub", "submit_reuse.sub"])
def test_submit_contract(name: str) -> None:
    text = (BUNDLE / name).read_text(encoding="utf-8")
    for required in (
        "universe = vanilla",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        f"checkpoint_exit_code = {CHECKPOINT_EXIT_CODE}",
        "transfer_checkpoint_files = raw_outputs/",
        "job_max_vacate_time = 600",
        "erase_output_and_error_on_restart = False",
        "stream_output = False",
        "stream_error = False",
        "requirements = TARGET.HasSIF",
        "request_cpus = 4",
        "request_memory = 49152MB",
        "request_disk = 81920MB",
        "+MaxRuntime = 259200",
        "queue 1",
    ):
        assert required in text
    mode = CONTROL_MODE if "control" in name else REUSE_MODE
    assert f"/{mode}_checkpoint.tar.gz" in text
    other = REUSE_MODE if mode == CONTROL_MODE else CONTROL_MODE
    assert f"/{other}_checkpoint.tar.gz" not in text
    assert text.count("queue 1") == 1


def test_reuse_submit_consumes_dynamic_gate() -> None:
    text = (BUNDLE / "submit_reuse.sub").read_text(encoding="utf-8")
    assert "anchor_gate.control.json" in text
    assert not (BUNDLE / "anchor_gate.control.json").exists()


def test_wrapper_has_signal_forwarding_atomic_whitelist_packaging() -> None:
    text = (BUNDLE / "execute_source_locked_job.sh").read_text(encoding="utf-8")
    assert "trap finalize_outputs EXIT" in text
    assert "forward_signal 143" in text
    assert "forward_signal 130" in text
    assert 'kill -TERM -- "-$CHILD_PID"' in text
    assert "checkpoint-restore" in text
    assert "checkpoint-pack" in text
    assert "CHECKPOINT_EXIT_CODE=85" in text
    assert "setsid" in text
    assert 'mv "${TRANSFER_ARCHIVE}.tmp" "$TRANSFER_ARCHIVE"' in text
    assert "json/current.json" in text
    assert "json/estimator_call_ledger.json" in text
    assert "runtime_source" not in text.split("for relative in", 1)[1].split("; do", 1)[0]
    assert "cache" not in text.split("for relative in", 1)[1].split("; do", 1)[0]


def test_post_gate_refuses_missing_or_partial_control(tmp_path: Path) -> None:
    output = tmp_path / "anchor.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(BUNDLE / "post_control_gate.py"),
            "--archive",
            str(tmp_path / "missing.tar.gz"),
            "--output",
            str(output),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode != 0
    assert not output.exists()


def test_submission_gate_records_verified_v5r2_repair() -> None:
    gate = load_json(BUNDLE / "submission_gate.json")
    assert gate["submission_enabled"] is True
    assert gate["status"] == "pass"
    assert gate["source_runtime_revision"] == "v5r2"
    assert (
        gate["historical_repaired_failure"]["id"]
        == HISTORICAL_REPAIRED_FAILURE["id"]
    )
    audit = load_json(BUNDLE / "source_locked_sensitivity_audit.json")
    assert audit["status"] == "pass"
    assert audit["required_suppressed_eps_grad_fallback_support"] is True
    assert audit["changed_fields_control_to_reuse"] == [
        "adapt_formal_manifold_route_profile"
    ]


def test_shell_and_python_syntax() -> None:
    shell = subprocess.run(
        ["bash", "-n", str(BUNDLE / "execute_source_locked_job.sh")],
        check=False,
    )
    assert shell.returncode == 0
    python_files = [
        BUNDLE / name
        for name in (
            "build_bundle.py",
            "pair_contract.py",
            "post_control_gate.py",
            "run_job.py",
            "validate_fetched.py",
        )
    ]
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, python_files)],
        check=False,
    )
    assert compiled.returncode == 0
