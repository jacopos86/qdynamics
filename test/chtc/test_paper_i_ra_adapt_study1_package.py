from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "study1_minimal_20260728_v1_chtc"
)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_DIR))

from link_shared_append import build_shared_append_receipts  # noqa: E402
from build_attempt_selection import (  # noqa: E402
    attempt_archive_name,
    build_attempt_selection,
)
from objective_gates import _validate_g10, _validate_g9  # noqa: E402
from package_contract import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    EXPECTED_ARTIFACT_ROLES,
    MEASURED_BUNDLE_ID,
    PACKAGE_ID,
    REMOTE_IMAGE_SHA256,
    STATIONARY_BUNDLE_ID,
    PackageContractError,
    atomic_write_json,
    digested,
    direct_execution_ids,
    direct_execution_rows,
    expected_artifact_path,
    logical_cell_keys,
    objective_gate_diagnostic_contract,
    package_control_plane_receipt,
    canonical_sha256,
    sha256_file,
    shared_append_rows,
    validate_authorization_receipt,
    validate_v7_authority,
)
from stage_transferred_executable import (  # noqa: E402
    ExecutableStagingError,
    stage_transferred_executable,
)
import run_scientific_preflight_smokes as preflight_smokes  # noqa: E402
from run_scientific_preflight_smokes import (  # noqa: E402
    G5_FIRST_INTERIOR_ROUND_BY_REPRESENTATION,
    G5_MODE_SHORT_PREFIX,
    PreflightError,
    _assert_checkpoint_signatures,
    _first_interior_witness_receipt,
    _immutable_tree_snapshot,
    _validate_g5_for_preflight,
)


def test_exact_study1_matrix_is_20_logical_and_18_direct() -> None:
    logical = logical_cell_keys()
    direct = direct_execution_rows()
    assert len(logical) == len(set(logical)) == 20
    assert len(direct) == len({row["execution_id"] for row in direct}) == 18
    measured_append = {
        f"{MEASURED_BUNDLE_ID}__{row['cell_id']}"
        for row in shared_append_rows()
    }
    assert measured_append.isdisjoint(direct_execution_ids())
    assert {
        row["canonical_execution_id"] for row in shared_append_rows()
    }.issubset(direct_execution_ids())
    assert sum(
        row["bundle_id"] == STATIONARY_BUNDLE_ID for row in direct
    ) == 10
    assert sum(row["bundle_id"] == MEASURED_BUNDLE_ID for row in direct) == 8


def test_g11_diagnostic_selection_is_fixed_six_job_method_regime_cover() -> None:
    selected = []
    for row in direct_execution_rows():
        contract = objective_gate_diagnostic_contract(
            bundle_id=row["bundle_id"],
            regime_id=row["regime_id"],
            route_id=row["route_id"],
        )
        if contract["selected"]:
            selected.append((row, contract))
    assert len(selected) == 6
    assert sum(
        row["route_id"] == "singleton_plateau"
        for row, _contract in selected
    ) == 4
    assert sum(
        row["route_id"] == "append_macro"
        for row, _contract in selected
    ) == 2
    assert {
        (contract["method_family"], row["regime_id"])
        for row, contract in selected
    } == {
        ("ra_adapt", "strong_weak_u8"),
        ("ra_adapt", "strong_strong_u8"),
        ("append_adapt", "strong_weak_u8"),
        ("append_adapt", "strong_strong_u8"),
    }
    for _row, contract in selected:
        if contract["method_family"] == "ra_adapt":
            assert contract[
                "ra_fresh_leg_maximum_controller_rounds"
            ] == 2
            assert contract[
                "ra_resumed_maximum_controller_rounds"
            ] == 3
        else:
            assert contract["append_resume_boundary"] == (
                "authenticated_reconstruction_only_v1"
            )


def test_expected_artifact_contract_is_exactly_five_roles() -> None:
    cell_id = "validation__strong_weak_u8__nph3__append_macro"
    assert set(EXPECTED_ARTIFACT_ROLES) == {
        "execution_manifest",
        "checkpoint",
        "estimator_ledger",
        "result",
        "summary",
    }
    assert expected_artifact_path(cell_id, "execution_manifest") == (
        f"runs/{cell_id}/execution_manifest.json"
    )
    assert expected_artifact_path(cell_id, "checkpoint") == (
        f"runs/{cell_id}/checkpoints/current.json"
    )


def test_g9_accepts_typed_receipt_without_nonexistent_protocol_field() -> None:
    integrity = {
        "schema": "paper_i_numerical_physical_integrity_v1",
        "method": "ra_adapt",
        "derivation_policy": (
            "post_controller_typed_result_and_signed_terminal_checkpoint_v1"
        ),
        "reporting_only": True,
        "controller_decision_influence": False,
        "finite_values_passed": True,
        "checked_energy_value_count": 1,
        "checked_parameter_value_count": 0,
        "nonfinite_value_paths": [],
        "sector_diagnostic_policy": (
            "reporting_only_post_controller_sector_probability_v1"
        ),
        "state_fingerprint": "projective:test",
        "sector_leak_threshold": 1.0e-8,
        "fixed_count_sector_probability": 1.0,
        "fixed_count_sector_leak_probability": 0.0,
        "sector_leak_flag": False,
        "boson_legal_probability_min": 1.0,
        "boson_illegal_probability_max": 0.0,
        "boson_truncation_leak_flag": False,
        "accepted_energy_transitions": [],
        "accepted_energy_integrity_passed": True,
        "integrity_passed": True,
    }
    result = {
        "numerical_physical_integrity": integrity,
        "scientific_receipts": {
            "numerical_physical_integrity": integrity,
            "numerical_physical_integrity_sha256": canonical_sha256(
                integrity
            ),
        },
        "run": {"accepted_transitions": []},
    }
    evidence = _validate_g9(
        job={
            "execution_id": "g9-regression",
            "execution_entrypoint": "run_ra_adapt",
        },
        protocol={"sha256": "f" * 64},
        result=result,
    )
    assert evidence["accepted_transition_check_count"] == 0
    assert "protocol_sha256" not in integrity


def test_g10_reads_typed_ra_all_work_accounting() -> None:
    components = {
        "n_h_outer": 3,
        "n_h_refit": 4,
        "n_grad": 5,
        "n_metric": 6,
    }
    result = {
        "run": {
            "estimator_accounting": {
                "complete": True,
                "status": (
                    "resolved_from_live_state_keyed_instrumentation"
                ),
                "exact_blockers": [],
                "prefix_closure_passed": True,
                "prefix_closure_status": "complete",
                "all_work": {
                    "components": components,
                    "s_alg": 18,
                },
                "winning_lineage": {
                    "components": components,
                    "s_alg": 18,
                },
                "raw_occurrences": components,
                "raw_occurrence_total": 18,
            }
        }
    }
    ledger_payload = {
        "schema": "estimator_call_ledger_v1",
        "occurrence_summary": {
            "N_H_outer": 3,
            "N_H_refit": 4,
            "N_grad": 5,
            "N_metric": 6,
            "S_alg": 18,
        },
        "occurrences": [],
    }
    ledger = {
        "schema": "paper_i_estimator_call_ledger_sidecar_v2",
        "accounting": {
            "schema": "paper_i_current_s_alg_accounting_v2",
            "enabled": True,
            "complete": True,
            "status": "resolved_from_live_state_keyed_instrumentation",
            "exact_blockers": [],
            "components": components,
            "S_alg": 18,
        },
        "ledger": ledger_payload,
        "adapt_success": True,
        "adapt_error": None,
    }
    evidence = _validate_g10(
        job={
            "execution_id": "typed-ra-accounting",
            "execution_entrypoint": "run_ra_adapt",
        },
        result=result,
        ledger=ledger,
    )
    assert evidence["components"] == {
        "N_H_outer": 3,
        "N_H_refit": 4,
        "N_grad": 5,
        "N_metric": 6,
    }
    assert evidence["S_alg"] == 18

    with pytest.raises(PackageContractError, match="sidecar schema"):
        _validate_g10(
            job={
                "execution_id": "typed-ra-accounting",
                "execution_entrypoint": "run_ra_adapt",
            },
            result=result,
            ledger=ledger_payload,
        )

    ledger["ledger"]["schema"] = "estimator_call_ledger_v0"
    with pytest.raises(PackageContractError, match="nested estimator ledger"):
        _validate_g10(
            job={
                "execution_id": "typed-ra-accounting",
                "execution_entrypoint": "run_ra_adapt",
            },
            result=result,
            ledger=ledger,
        )
    ledger["ledger"]["schema"] = "estimator_call_ledger_v1"

    result["run"]["estimator_accounting"]["all_work"]["s_alg"] = 19
    with pytest.raises(PackageContractError, match="component closure"):
        _validate_g10(
            job={
                "execution_id": "typed-ra-accounting",
                "execution_entrypoint": "run_ra_adapt",
            },
            result=result,
            ledger=ledger,
        )


def test_g10_requires_append_raw_ledger_shape() -> None:
    components = {
        "N_H_outer": 1,
        "N_H_refit": 2,
        "N_grad": 3,
        "N_metric": 0,
    }
    result = {
        "result_payload": {
            "estimator_accounting": {
                "components": components,
                "S_alg": 6,
                "closed_occurrence_reconciliation": True,
            }
        }
    }
    ledger = {
        "schema": "estimator_call_ledger_v1",
        "occurrence_summary": {
            **components,
            "S_alg": 6,
        },
        "occurrences": [],
    }
    evidence = _validate_g10(
        job={
            "execution_id": "typed-append-accounting",
            "execution_entrypoint": "run_append_adapt",
        },
        result=result,
        ledger=ledger,
    )
    assert evidence["components"] == components
    assert evidence["S_alg"] == 6

    with pytest.raises(PackageContractError, match="Append estimator ledger"):
        _validate_g10(
            job={
                "execution_id": "typed-append-accounting",
                "execution_entrypoint": "run_append_adapt",
            },
            result=result,
            ledger={
                "schema": "paper_i_estimator_call_ledger_sidecar_v2",
                "ledger": ledger,
            },
        )


def test_builder_waits_for_immutable_v7_final_receipt(
    tmp_path: Path,
) -> None:
    with pytest.raises(PackageContractError, match="wait"):
        validate_v7_authority(tmp_path)


def test_authorization_must_postdate_and_exactly_bind_v7() -> None:
    authority = {
        "final_receipt": {
            "finalized_utc": "2026-07-28T12:00:00Z",
        },
        "final_receipt_binding": {
            "canonical_sha256": "a" * 64,
            "file_sha256": "b" * 64,
        },
        "objective_gate_authority": {"sha256": "d" * 64},
        "dedupe_sha256": "c" * 64,
    }
    base = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_id": "study1-test-auth",
        "authorized_utc": "2026-07-28T12:00:01Z",
        "package_id": PACKAGE_ID,
        "campaign_id": "paper_i_ra_adapt_stationarity_comparison_v1",
        "run_class": "candidate",
        "execution_target": "chtc",
        "execution_authorized": True,
        "submission_authorized": True,
        "v7_final_receipt_file_sha256": "b" * 64,
        "v7_final_receipt_canonical_sha256": "a" * 64,
        "study1_objective_gate_authority_sha256": "d" * 64,
        "study1_dedupe_sha256": "c" * 64,
        "package_control_plane_sha256": "e" * 64,
        "logical_cell_count": 20,
        "direct_execution_count": 18,
        "authorized_logical_cell_keys": list(logical_cell_keys()),
        "authorized_direct_execution_ids": list(direct_execution_ids()),
        "remote_image_sha256": REMOTE_IMAGE_SHA256,
    }
    receipt = digested(base)
    assert (
        validate_authorization_receipt(
            receipt,
            v7_authority=authority,
            package_control_plane_sha256="e" * 64,
        )
        == receipt["sha256"]
    )
    stale = dict(base)
    stale["authorized_utc"] = "2026-07-28T11:59:59Z"
    with pytest.raises(PackageContractError):
        validate_authorization_receipt(
            digested(stale),
            v7_authority=authority,
            package_control_plane_sha256="e" * 64,
        )


def test_package_control_plane_digest_covers_every_static_member(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "package"
    copied.mkdir()
    baseline = package_control_plane_receipt(PACKAGE_DIR)
    for row in baseline["files"]:
        shutil.copy2(PACKAGE_DIR / row["path"], copied / row["path"])
    observed = package_control_plane_receipt(copied)
    assert observed == baseline
    changed = copied / "run_cell.py"
    changed.write_bytes(changed.read_bytes() + b"\n# drift probe\n")
    assert package_control_plane_receipt(copied)["sha256"] != baseline["sha256"]


def test_attempt_selector_binds_cluster_proc_and_ignores_older_attempts(
    tmp_path: Path,
) -> None:
    plan = digested(
        {
            "schema": "paper_i_ra_adapt_study1_execution_plan_v2",
            "package_id": PACKAGE_ID,
            "direct_executions": [
                {"execution_id": execution_id}
                for execution_id in direct_execution_ids()
            ],
        }
    )
    plan_path = tmp_path / "execution_plan.json"
    atomic_write_json(plan_path, plan)
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    cluster_id = 12345
    for proc_id, execution_id in enumerate(direct_execution_ids()):
        name = attempt_archive_name(
            execution_id,
            cluster_id=cluster_id,
            proc_id=proc_id,
        )
        (fetched / name).write_bytes(
            f"{execution_id}:{cluster_id}:{proc_id}".encode()
        )
    first_execution = direct_execution_ids()[0]
    older = attempt_archive_name(
        first_execution,
        cluster_id=999,
        proc_id=0,
    )
    (fetched / older).write_bytes(b"preserved older attempt")

    selection = build_attempt_selection(
        plan_path=plan_path,
        fetched_dir=fetched,
        cluster_id=cluster_id,
    )
    assert selection["status"] == "ready"
    assert len(selection["selections"]) == 18
    assert selection["selections"][0]["archive_name"] != older
    assert (fetched / older).is_file()
    assert all(
        row["archive_name"]
        == attempt_archive_name(
            row["execution_id"],
            cluster_id=cluster_id,
            proc_id=index,
        )
        for index, row in enumerate(selection["selections"])
    )


def _protocol_payload(
    *,
    bundle_id: str,
    problem_token: str,
) -> dict[str, object]:
    return digested(
        {
            "schema": "paper_i_append_adapt_resolved_protocol_v1",
            "bundle_id": bundle_id,
            "algorithm_id": "paper_i_append_adapt_v1",
            "candidate_representation": "macro_generator_v1",
            "problem": {"token": problem_token},
        }
    )


def test_shared_append_linker_writes_receipts_not_measured_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    fetched_root = tmp_path / "fetched"
    output_root = tmp_path / "validated"
    source_root.mkdir()
    fetched_root.mkdir()
    required_equal_fields = [
        "algorithm_id",
        "candidate_representation",
        "problem",
    ]
    dedupe = digested(
        {
            "schema": "paper_i_ra_adapt_study1_execution_dedupe_v1",
            "scientific_equivalence_projection": {
                "required_equal_fields": required_equal_fields,
            },
        }
    )
    logical_rows = []
    direct_rows = []
    measured_paths = []
    for shared in shared_append_rows():
        cell_id = shared["cell_id"]
        stationary_protocol_path = (
            f"bundles/{STATIONARY_BUNDLE_ID}/protocols/{cell_id}.json"
        )
        measured_protocol_path = (
            f"bundles/{MEASURED_BUNDLE_ID}/protocols/{cell_id}.json"
        )
        stationary_protocol = _protocol_payload(
            bundle_id=STATIONARY_BUNDLE_ID,
            problem_token=shared["regime_id"],
        )
        measured_protocol = _protocol_payload(
            bundle_id=MEASURED_BUNDLE_ID,
            problem_token=shared["regime_id"],
        )
        atomic_write_json(
            source_root / stationary_protocol_path, stationary_protocol
        )
        atomic_write_json(
            source_root / measured_protocol_path, measured_protocol
        )
        stationary_binding = {
            "path": stationary_protocol_path,
            "canonical_sha256": stationary_protocol["sha256"],
            "file_sha256": sha256_file(
                source_root / stationary_protocol_path
            ),
        }
        measured_binding = {
            "path": measured_protocol_path,
            "canonical_sha256": measured_protocol["sha256"],
            "file_sha256": sha256_file(source_root / measured_protocol_path),
        }
        canonical_artifacts = {}
        reference_artifacts = {}
        for role in EXPECTED_ARTIFACT_ROLES:
            canonical_relative = (
                f"bundles/{STATIONARY_BUNDLE_ID}/"
                f"{expected_artifact_path(cell_id, role)}"
            )
            path = fetched_root / canonical_relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"role": role, "regime_id": shared["regime_id"]}
                ),
                encoding="utf-8",
            )
            canonical_artifacts[role] = canonical_relative
            measured_relative = (
                f"bundles/{MEASURED_BUNDLE_ID}/"
                f"{expected_artifact_path(cell_id, role)}"
            )
            measured_paths.append(fetched_root / measured_relative)
            reference_artifacts[role] = {
                "path": expected_artifact_path(cell_id, role),
                "required": True,
                "fulfillment_kind": "shared_result_reference_v1",
                "direct_file_required": False,
                "reference_receipt_required": True,
            }
        logical_rows.extend(
            [
                {
                    "logical_key": (
                        f"{STATIONARY_BUNDLE_ID}::{cell_id}"
                    ),
                    "direct_execution_required": True,
                    "protocol": stationary_binding,
                    "execution_fulfillment": {
                        "fulfillment_kind": (
                            "canonical_shared_execution_v1"
                        )
                    },
                    "expected_run_artifacts": {},
                },
                {
                    "logical_key": (
                        f"{MEASURED_BUNDLE_ID}::{cell_id}"
                    ),
                    "direct_execution_required": False,
                    "canonical_execution_id": shared[
                        "canonical_execution_id"
                    ],
                    "protocol": measured_binding,
                    "execution_fulfillment": {
                        "fulfillment_kind": "shared_result_reference_v1",
                        "group_id": (
                            f"study1_append_shared__"
                            f"{shared['regime_id']}__nph3"
                        ),
                    },
                    "expected_run_artifacts": reference_artifacts,
                },
            ]
        )
        direct_rows.append(
            {
                "execution_id": shared["canonical_execution_id"],
                "artifact_paths": canonical_artifacts,
            }
        )
    plan = digested(
        {
            "package_id": PACKAGE_ID,
            "study1_dedupe_sha256": dedupe["sha256"],
            "logical_cells": logical_rows,
            "direct_executions": direct_rows,
        }
    )
    output_root.mkdir()
    receipts = build_shared_append_receipts(
        source_root=source_root,
        fetched_root=fetched_root,
        output_dir=output_root,
        plan=plan,
        dedupe=dedupe,
    )
    assert len(receipts) == 2
    assert all(not path.exists() for path in measured_paths)
    for row in receipts:
        payload = json.loads(
            (output_root / row["path"]).read_text(encoding="utf-8")
        )
        assert payload["all_required_fields_equal"] is True
        assert payload["all_output_hashes_equal"] is True
        assert (
            payload["reference_fulfillment"][
                "physical_files_materialized"
            ]
            is False
        )


def test_submit_and_wrapper_lock_image_resources_and_cleanenv() -> None:
    submit = (PACKAGE_DIR / "submit.sub").read_text(encoding="utf-8")
    wrapper = (PACKAGE_DIR / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    )
    assert "request_cpus = 4" in submit
    assert "initialdir =" not in submit
    assert (
        "study1_minimal_20260728_v1_chtc/$(job_spec)"
        in submit
    )
    assert "when_to_transfer_output = ON_EXIT" in submit
    assert "ON_EXIT_OR_EVICT" not in submit
    assert "stream_output = False" in submit
    assert "stream_error = False" in submit
    assert "+MaxRuntime = 259200" in submit
    assert "requirements = TARGET.HasSIF" in submit
    assert "$(memory_mb)MB" in submit
    assert "$(disk_mb)MB" in submit
    assert REMOTE_IMAGE_SHA256 in submit
    assert (
        "$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz"
        in submit
    )
    assert "--cleanenv" in wrapper
    assert "actual_source_sha256" in wrapper
    assert "actual_image_sha256" in wrapper
    assert "authorization-bound package control plane drifted" in wrapper
    assert '--wrapper-source "$0"' in wrapper
    assert (
        "stage_transferred_executable.py"
        in submit
    )
    assert (
        "execute_source_locked_job.sh"
        not in submit.split("transfer_input_files =", maxsplit=1)[1].split(
            "\n", maxsplit=1
        )[0]
    )
    assert "worker_status == 0 and len(members) != 6" in wrapper


def test_p4_renamed_transfer_executable_stages_without_collision(
    tmp_path: Path,
) -> None:
    scratch_package = (
        tmp_path
        / "chtc/paper_i_ra_adapt_repair_20260727/"
        "study1_minimal_20260728_v1_chtc"
    )
    scratch_package.mkdir(parents=True)
    wrapper = PACKAGE_DIR / "execute_source_locked_job.sh"
    renamed = tmp_path / "condor_exec.exe"
    shutil.copy2(wrapper, renamed)

    first = stage_transferred_executable(
        wrapper_source=renamed,
        package_dir=scratch_package,
    )
    staged = scratch_package / "execute_source_locked_job.sh"
    assert first["action"] == "staged_renamed_transfer_executable"
    assert staged.read_bytes() == wrapper.read_bytes()
    assert sha256_file(staged) == sha256_file(renamed)

    second = stage_transferred_executable(
        wrapper_source=staged,
        package_dir=scratch_package,
    )
    assert second["action"] == "already_at_authenticated_path"

    with pytest.raises(
        ExecutableStagingError,
        match="duplicate transferred-executable collision",
    ):
        stage_transferred_executable(
            wrapper_source=renamed,
            package_dir=scratch_package,
        )


def test_p4_tampered_staging_helper_fails_before_helper_execution(
    tmp_path: Path,
) -> None:
    wrapper_text = (
        PACKAGE_DIR / "execute_source_locked_job.sh"
    ).read_text(encoding="utf-8")
    marker = (
        'python3 - "$package_dir" "$authorization_receipt" "$1" '
        "<<'PY'\n"
    )
    verifier = wrapper_text.split(marker, maxsplit=1)[1].split(
        "\nPY\n}", maxsplit=1
    )[0]
    scratch_package = tmp_path / "package"
    scratch_package.mkdir()
    control = package_control_plane_receipt(PACKAGE_DIR)
    renamed_wrapper = tmp_path / "condor_exec.exe"
    shutil.copy2(
        PACKAGE_DIR / "execute_source_locked_job.sh",
        renamed_wrapper,
    )
    for row in control["files"]:
        if row["path"] == "execute_source_locked_job.sh":
            continue
        shutil.copy2(
            PACKAGE_DIR / row["path"],
            scratch_package / row["path"],
        )
    authorization = digested(
        {"package_control_plane_sha256": control["sha256"]}
    )
    authorization_path = tmp_path / "authorization.json"
    atomic_write_json(authorization_path, authorization)
    subprocess.run(
        [
            sys.executable,
            "-",
            str(scratch_package),
            str(authorization_path),
            str(renamed_wrapper),
        ],
        input=verifier,
        text=True,
        check=True,
    )

    helper = scratch_package / "stage_transferred_executable.py"
    helper.write_bytes(helper.read_bytes() + b"\n# tampered before auth\n")
    failed = subprocess.run(
        [
            sys.executable,
            "-",
            str(scratch_package),
            str(authorization_path),
            str(renamed_wrapper),
        ],
        input=verifier,
        text=True,
        capture_output=True,
    )
    assert failed.returncode != 0
    assert "authorization-bound package control plane drifted" in (
        failed.stderr
    )
    assert not (
        scratch_package / "execute_source_locked_job.sh"
    ).exists()


def test_preflight_checkpoint_signatures_are_method_specific() -> None:
    append_checkpoint = {
        "schema": "paper_i_append_adapt_checkpoint_v1",
        "controller_rounds_completed": 1,
    }
    append_checkpoint["sha256"] = canonical_sha256(append_checkpoint)
    assert (
        _assert_checkpoint_signatures(
            append_checkpoint,
            label="append fixture",
            append=True,
            expected_rounds=1,
        )
        == append_checkpoint["sha256"]
    )

    ra_prefix = {
        "schema": "paper_i_signed_active_prefix_checkpoint_v1",
        "outer_iteration": 1,
        "checkpoint_kind": "post_admission_prune",
        "scientific_payload": "fixture",
    }
    ra_prefix["checkpoint_sha256"] = canonical_sha256(ra_prefix)
    ra_terminal = {
        **{
            key: value
            for key, value in ra_prefix.items()
            if key != "checkpoint_sha256"
        },
        "checkpoint_kind": "terminal_post_final_refit_and_prune",
    }
    ra_terminal["checkpoint_sha256"] = canonical_sha256(ra_terminal)
    ra_checkpoint = {
        "schema_version": "static_adapt_current_checkpoint_v1",
        "checkpoint": {"depth": 1},
        "adapt_vqe": {
            "history_count": 1,
            "active_prefix_checkpoints": [ra_prefix],
            "terminal_active_prefix_checkpoint": ra_terminal,
            "continuation": {
                "terminal_active_prefix_checkpoint": ra_terminal,
            },
        },
    }
    assert (
        _assert_checkpoint_signatures(
            ra_checkpoint,
            label="RA fixture",
            append=False,
            expected_rounds=1,
        )
        == ra_terminal["checkpoint_sha256"]
    )
    wrong_field = dict(ra_prefix)
    wrong_field["sha256"] = wrong_field.pop("checkpoint_sha256")
    with pytest.raises(PreflightError):
        _assert_checkpoint_signatures(
            {
                **ra_checkpoint,
                "adapt_vqe": {
                    **ra_checkpoint["adapt_vqe"],
                    "active_prefix_checkpoints": [wrong_field],
                    "terminal_active_prefix_checkpoint": wrong_field,
                },
            },
            label="wrong RA fixture",
            append=False,
            expected_rounds=1,
        )


def test_preflight_short_ra_prefix_defers_only_missing_interior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = {
        "execution_id": "short_ra",
        "execution_entrypoint": "run_ra_adapt",
        "route_id": "ra_macro_plateau",
    }

    def missing_interior(**_kwargs: object) -> dict[str, object]:
        raise PackageContractError(
            "G5 requires an interior scored receipt for short_ra."
        )

    monkeypatch.setattr(
        preflight_smokes,
        "_validate_g5",
        missing_interior,
    )
    receipt = _validate_g5_for_preflight(
        job=job,
        protocol={},
        result={},
        mode=G5_MODE_SHORT_PREFIX,
    )
    assert receipt["aggregate_g5_passed"] is False
    assert receipt["strict_validator_outcome"] == (
        "expected_missing_interior_only"
    )
    assert receipt["deferred_witness_role"] == (
        "separate_g5_plateau_witness"
    )

    def wrong_failure(**_kwargs: object) -> dict[str, object]:
        raise PackageContractError("G5 structural drift.")

    monkeypatch.setattr(
        preflight_smokes,
        "_validate_g5",
        wrong_failure,
    )
    with pytest.raises(
        PreflightError,
        match="failed G5 before the intended",
    ):
        _validate_g5_for_preflight(
            job=job,
            protocol={},
            result={},
            mode=G5_MODE_SHORT_PREFIX,
        )

    monkeypatch.setattr(
        preflight_smokes,
        "_validate_g5",
        lambda **_kwargs: {"interior_scored_count": 1},
    )
    with pytest.raises(PreflightError, match="unexpectedly satisfied"):
        _validate_g5_for_preflight(
            job=job,
            protocol={},
            result={},
            mode=G5_MODE_SHORT_PREFIX,
        )


def test_preflight_first_interior_bounds_are_exact_and_fail_closed() -> None:
    assert G5_FIRST_INTERIOR_ROUND_BY_REPRESENTATION == {
        "macro_generator_v1": 13,
        "single_pauli_word_v1": 13,
    }
    payload = {
        "scientific_receipts": {
            "accepted_round_receipts": [
                {
                    "accepted_round_ordinal": ordinal,
                    "scored_insertion_position_population": {
                        "interior_scored_count": int(ordinal == 13),
                    },
                }
                for ordinal in range(1, 14)
            ]
        }
    }
    receipt = _first_interior_witness_receipt(
        case_id="witness",
        payload=payload,
        expected_round=13,
    )
    assert receipt["observed_first_interior_round"] == 13
    assert receipt["rounds_before_witness_have_zero_interior"] is True
    assert receipt["witness_round_interior_scored_count"] == 1

    payload["scientific_receipts"]["accepted_round_receipts"][11][
        "scored_insertion_position_population"
    ]["interior_scored_count"] = 1
    with pytest.raises(PreflightError, match="first interior round is 12"):
        _first_interior_witness_receipt(
            case_id="early_witness",
            payload=payload,
            expected_round=13,
        )


def test_preflight_runtime_copy_preserves_authority_content_and_stat(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    nested = authority / "nested"
    nested.mkdir()
    source = nested / "protocol.json"
    source.write_text('{"authority":"immutable"}\n', encoding="utf-8")
    before = _immutable_tree_snapshot(authority)

    runtime_copy = tmp_path / "runtime-copy"
    shutil.copytree(authority, runtime_copy, copy_function=shutil.copy2)
    copied = runtime_copy / "nested" / "protocol.json"
    copied.write_text('{"runtime":"mutable"}\n', encoding="utf-8")
    (runtime_copy / "raw_outputs").mkdir()
    (runtime_copy / "raw_outputs" / "cache.bin").write_bytes(b"cache")

    after = _immutable_tree_snapshot(authority)
    assert after == before
    assert source.read_text(encoding="utf-8") == (
        '{"authority":"immutable"}\n'
    )


def test_wrapper_output_packager_writes_exact_six_member_archive(
    tmp_path: Path,
) -> None:
    wrapper = (PACKAGE_DIR / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    )
    marker = "\"$worker_status\" <<'PY'\n"
    embedded = wrapper.split(marker, maxsplit=1)[1].split(
        "\nPY\n}", maxsplit=1
    )[0]
    root = tmp_path / "scratch"
    root.mkdir()
    cell_id = "validation__strong_weak_u8__nph3__append_macro"
    artifacts = {
        role: (
            f"bundles/stationary/runs/{cell_id}/"
            f"{role}.json"
        )
        for role in EXPECTED_ARTIFACT_ROLES
    }
    worker_receipt = f"worker_receipts/{cell_id}.json"
    job = {
        "artifact_paths": artifacts,
        "worker_receipt_path": worker_receipt,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    for relative in (*artifacts.values(), worker_receipt):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"probe"}\n', encoding="utf-8")
    destination = tmp_path / "transfer" / "probe.tar.gz"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(job_path),
            str(root),
            str(destination),
            "0",
        ],
        input=embedded,
        text=True,
        check=True,
    )
    with tarfile.open(destination, "r:gz") as archive:
        assert set(archive.getnames()) == {
            *artifacts.values(),
            worker_receipt,
        }


def test_worker_consumes_typed_summary_without_recomputation() -> None:
    worker = (PACKAGE_DIR / "run_cell.py").read_text(encoding="utf-8")
    assert "summarize_paper_i" not in worker
    assert 'getattr(result, "paper_i_summary", None)' in worker
    assert 'getattr(run, "paper_i_summary", None)' in worker
    assert 'expected_schema = "paper_i_append_run_summary_v1"' in worker
    assert 'expected_schema = "paper_i_run_summary_v1"' in worker
