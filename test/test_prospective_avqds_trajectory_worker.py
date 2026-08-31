"""Focused contracts for the scientific prospective AVQDS executor."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import pipelines.time_dynamics.prospective_avqds_trajectory_worker as worker
from pipelines.time_dynamics.prospective_avqds_trajectory_chtc_package import (
    AVQDSTrajectoryCHTCResources,
    _dependency_files,
    _descriptor,
    _worker as package_worker,
)
from chtc.validate_condor_submit_lifecycle import validate_submit_lifecycle
from pipelines.time_dynamics.prospective_avqds_trajectory_worker import (
    AVQDSTrajectoryExecutorOverlayIdentity,
    COMPLETED_UNSCORED,
    EXECUTION_CONTRACT_SCHEMA_V1,
    EXECUTOR_OVERLAY_PATHS,
    build_work_order,
    load_work_order,
    materialization_request,
    materialize_avqds_trajectory,
    validate_canonical_avqds_request,
    write_work_order,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ProspectiveExecutionError,
    VerifiedFile,
    resolve_scientific_action,
    write_run_lifecycle_receipt,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_RUN,
    ProspectiveNextAction,
    plan_prospective_cell,
    prospective_run_action_id,
)
from test_prospective_execution_bindings import (
    _action_registry_binding,
    _run_state_evidence,
    _scientific_context,
    _sha,
    _write,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _overlay(root: Path, context: dict) -> AVQDSTrajectoryExecutorOverlayIdentity:
    rows = {}
    for index, relative in enumerate(EXECUTOR_OVERLAY_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"# scientific avqds test overlay {index}\n".encode()
        path.write_bytes(content)
        rows[relative] = {
            "base_present": False,
            "base_sha256": None,
            "overlay_sha256": hashlib.sha256(content).hexdigest(),
        }
    provisional = AVQDSTrajectoryExecutorOverlayIdentity(
        base_source_epoch_identity_sha256=context["source"].identity_sha256,
        base_source_revision=str(context["source"].source_revision),
        overlay_revision="b" * 40,
        files=rows,
        files_sha256=_canonical_sha(rows),
    )
    return AVQDSTrajectoryExecutorOverlayIdentity.from_mapping(
        provisional.to_json_dict()
    )


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker,
        "_runner_argv",
        lambda *args, **kwargs: ("--exact-blind-test-runner",),
    )

    def contract(**kwargs):
        request = kwargs["request"]
        core = {
            "schema": EXECUTION_CONTRACT_SCHEMA_V1,
            "execution_request_sha256": request.digest,
            "bindings_sha256": request.inputs["bindings_sha256"],
            "exact_reference_data_access": False,
            "physical_hilbert_dimension": 2,
        }
        return {**core, "execution_contract_sha256": _canonical_sha(core)}

    monkeypatch.setattr(worker, "_build_execution_contract", contract)


def _prefix_context(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    context = _scientific_context(root)
    plan = plan_prospective_cell(context["bindings"], context["policy"])
    action = plan.next_actions[0]
    assert action.arm == ARM_AVQDS and action.horizon == HORIZON_PREFIX
    request, action_registry = _action_registry_binding(root, context, action)
    _patch_runtime(monkeypatch)
    context.update(
        {
            "action": action,
            "request": request,
            "action_registry": action_registry,
            "overlay": _overlay(root, context),
        }
    )
    return context


def _prefix_order(root: Path, monkeypatch: pytest.MonkeyPatch):
    context = _prefix_context(root, monkeypatch)
    order = build_work_order(
        context["request"],
        action_registry=context["action_registry"],
        source_epoch_receipt=context["source_receipt"],
        seed_receipt=context["seed"],
        repository_root=root,
        executor_overlay=context["overlay"],
    )
    return context, order


def _completed_prefix(root: Path, context: dict) -> tuple[VerifiedFile, VerifiedFile]:
    run_path = root / "raw/prefix.json"
    _write(run_path, {"schema": "exact_blind_prefix_test"})
    run = VerifiedFile(run_path.relative_to(root).as_posix(), _sha(run_path))
    submitted = write_run_lifecycle_receipt(
        root / "state/prefix_submitted.json",
        request=context["request"],
        state="submitted",
        action_registry=context["action_registry"],
        state_evidence=_run_state_evidence(
            root, context["action"], "submitted", token="avqds_prefix_submitted"
        ),
        repository_root=root,
    )
    completed = write_run_lifecycle_receipt(
        root / "state/prefix_completed.json",
        request=context["request"],
        state="completed",
        action_registry=context["action_registry"],
        state_evidence=_run_state_evidence(
            root, context["action"], "completed", token="avqds_prefix_completed"
        ),
        repository_root=root,
        previous_receipt=VerifiedFile(submitted.receipt_path, submitted.receipt_sha256),
        run=run,
    )
    return run, VerifiedFile(completed.receipt_path, completed.receipt_sha256)


def _full_request(root: Path, context: dict, run: VerifiedFile, receipt: VerifiedFile):
    bindings = context["bindings"]
    cut = context["action"].activation_cut
    action_id = prospective_run_action_id(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_FULL,
        activation_cut=cut,
    )
    action = ProspectiveNextAction(
        kind=KIND_RUN,
        action_id=action_id,
        execution_id=f"{action_id}::attempt0::mem8g",
        arm=ARM_AVQDS,
        horizon=HORIZON_FULL,
        activation_cut=cut,
        ladder_index=context["action"].ladder_index,
        attempt=0,
        planner_sequence=2,
        requested_memory_gib=8,
        reason="continue selected exact-scored AVQDS prefix",
        inputs={
            **bindings.to_json_dict(),
            "arm": ARM_AVQDS,
            "horizon": HORIZON_FULL,
            "activation_cut": cut,
            "ladder_index": context["action"].ladder_index,
            "record_statevector": True,
            "exact_reference_blind": True,
            "resource_only_retry": False,
            "scientific_action_id": action_id,
            "seeded_from_scout_action_id": None,
            "resume_from_action_id": context["action"].action_id,
            "resume_run_receipt_sha256": receipt.sha256,
            "resume_run_sha256": run.sha256,
        },
        exact_reference_blind=True,
    )
    request, registry = _action_registry_binding(
        root, context, action, token="full_avqds"
    )
    return action, request, registry


def test_prefix_accepts_every_frozen_rung_and_never_requires_exact_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _prefix_context(tmp_path, monkeypatch)
    cases = []
    for index, cut in enumerate(worker.HH_HUBBARD_ACTIVATION_CUT_LADDER):
        bindings = context["bindings"]
        action_id = prospective_run_action_id(
            bindings,
            arm=ARM_AVQDS,
            horizon=HORIZON_PREFIX,
            activation_cut=cut,
        )
        action = replace(
            context["action"],
            action_id=action_id,
            execution_id=f"{action_id}::attempt0::mem8g",
            activation_cut=cut,
            ladder_index=index,
            inputs={
                **context["action"].inputs,
                "activation_cut": cut,
                "ladder_index": index,
                "scientific_action_id": action_id,
            },
        )
        # Reuse is intentionally impossible across action IDs; write a fresh
        # registry envelope, which still binds only the exact receipt metadata.
        request, registry = _action_registry_binding(
            tmp_path, context, action, token=f"rung_{index}"
        )
        cases.append((request, registry))
    # Registry construction is the campaign-boundary exact join.  AVQDS
    # preparation and execution retain only its opaque identity after that
    # boundary, so none of the exact receipt/manifest/data files may be needed.
    for binding in (
        context["reference"].data,
        context["reference"].manifest,
    ):
        (tmp_path / binding.path).unlink()
    (tmp_path / context["reference"].receipt_path).unlink()
    for request, registry in cases:
        validated = validate_canonical_avqds_request(
            request,
            action_registry=registry,
            source_epoch_receipt=context["source_receipt"],
            seed_receipt=context["seed"],
            repository_root=tmp_path,
        )
        assert validated[0].digest == bindings.digest


def test_full_continuation_requires_same_cut_completed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _prefix_context(tmp_path, monkeypatch)
    run, receipt = _completed_prefix(tmp_path, context)
    action, request, registry = _full_request(tmp_path, context, run, receipt)
    order = build_work_order(
        request,
        action_registry=registry,
        source_epoch_receipt=context["source_receipt"],
        seed_receipt=context["seed"],
        repository_root=tmp_path,
        executor_overlay=context["overlay"],
        resume_run=run,
        resume_receipt=receipt,
    )
    assert order.request.inputs["horizon"] == HORIZON_FULL
    assert order.resume_run == run
    assert order.resume_receipt == receipt
    assert order.request.inputs["resume_from_action_id"] == context["action"].action_id
    assert materialization_request(request)["required_receipts"][-2:] == [
        "prefix_run",
        "prefix_run_lifecycle_receipt",
    ]

    wrong = VerifiedFile(receipt.path, "0" * 64)
    with pytest.raises(ProspectiveExecutionError, match="artifact digest mismatch"):
        build_work_order(
            request,
            action_registry=registry,
            source_epoch_receipt=context["source_receipt"],
            seed_receipt=context["seed"],
            repository_root=tmp_path,
            executor_overlay=context["overlay"],
            resume_run=run,
            resume_receipt=wrong,
        )

    missing_request = replace(
        request,
        inputs={**request.inputs, "resume_run_sha256": None},
    )
    with pytest.raises(ProspectiveExecutionError, match="registry names another request"):
        build_work_order(
            missing_request,
            action_registry=registry,
            source_epoch_receipt=context["source_receipt"],
            seed_receipt=context["seed"],
            repository_root=tmp_path,
            executor_overlay=context["overlay"],
            resume_run=run,
            resume_receipt=receipt,
        )


def test_persisted_runtime_and_package_dependencies_are_exact_artifact_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, order = _prefix_order(tmp_path, monkeypatch)
    work_order_path = tmp_path / "work/order.json"
    write_work_order(work_order_path, order)
    payload = json.loads(work_order_path.read_text())
    serialized = json.dumps(payload, sort_keys=True)
    forbidden = (
        "reference_receipt_sha256",
        "reference_manifest_sha256",
        "reference_data_sha256",
        "response_scales",
        '"action_registry"',
    )
    assert all(token not in serialized for token in forbidden)
    assert payload["runtime_bindings"]["reference_identity_sha256"] == (
        context["bindings"].reference_identity_sha256
    )
    assert payload["runtime_request"]["execution_request_sha256"] == (
        context["request"].digest
    )

    exact_paths = {
        context["reference"].data.path,
        context["reference"].manifest.path,
        context["reference"].receipt_path,
        context["action_registry"].path,
    }
    for path in exact_paths:
        (tmp_path / path).unlink()
    loaded, _source, _seed = load_work_order(
        work_order_path, repository_root=tmp_path
    )
    dependencies = _dependency_files(loaded, root=tmp_path)
    assert all(
        name.removeprefix("repository/") not in exact_paths
        for name in dependencies
    )
    assert loaded.bindings.reference_identity_sha256 == (
        context["bindings"].reference_identity_sha256
    )


def test_runtime_work_order_rejects_reintroduced_response_scales(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _context, order = _prefix_order(tmp_path, monkeypatch)
    path = tmp_path / "work/order.json"
    write_work_order(path, order)
    payload = json.loads(path.read_text())
    payload["runtime_request"]["inputs"]["response_scales"] = {"t5": {}}
    payload.pop("work_order_sha256")
    payload["work_order_sha256"] = _canonical_sha(payload)
    _write(path, payload)
    with pytest.raises(ProspectiveExecutionError, match="forbidden exact inputs"):
        load_work_order(path, repository_root=tmp_path)


def test_runtime_work_order_replay_does_not_reopen_campaign_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _context, order = _prefix_order(tmp_path, monkeypatch)
    path = tmp_path / "orders/matrix_blind_scientific.json"
    write_work_order(path, order)

    def forbidden_registry(*_args, **_kwargs):
        raise AssertionError("remote work-order replay reopened campaign matrix")

    monkeypatch.setattr(worker, "build_identity_registry", forbidden_registry)
    loaded, _source, _seed = load_work_order(path, repository_root=tmp_path)
    assert loaded.to_json_dict() == order.to_json_dict()


@pytest.mark.parametrize(
    "mutation",
    ("source", "seed", "reference", "arm", "cut", "controller", "cell"),
)
def test_rejects_cross_identity_and_controller_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    context = _prefix_context(tmp_path, monkeypatch)
    request = context["request"]
    inputs = dict(request.inputs)
    arm = request.arm
    if mutation == "source":
        inputs["source_epoch_identity_sha256"] = "0" * 64
    elif mutation == "seed":
        inputs["seed_artifact_sha256"] = "0" * 64
    elif mutation == "reference":
        inputs["reference_receipt_sha256"] = "0" * 64
    elif mutation == "arm":
        arm = ARM_APM
        inputs["arm"] = ARM_APM
    elif mutation == "cut":
        inputs["activation_cut"] = 2.0e-3
    elif mutation == "controller":
        inputs["controller"] = "state_motion_1e-2"
    else:
        inputs["cell_id"] = "another-cell"
    changed = replace(request, inputs=inputs, arm=arm)
    with pytest.raises((ProspectiveExecutionError, ValueError)):
        validate_canonical_avqds_request(
            changed,
            action_registry=context["action_registry"],
            source_epoch_receipt=context["source_receipt"],
            seed_receipt=context["seed"],
            repository_root=tmp_path,
        )


def test_materialization_finalizes_interrupted_part_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, order = _prefix_order(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "validate_raw_run", lambda *args, **kwargs: None)
    part = tmp_path / order.run_part_path
    part.parent.mkdir(parents=True, exist_ok=True)
    _write(part, {"schema": "already_attested_interrupted_part"})
    called = False

    def forbidden_execute(*args, **kwargs):
        nonlocal called
        called = True

    first = materialize_avqds_trajectory(
        order, repository_root=tmp_path, execute=forbidden_execute
    )
    second = materialize_avqds_trajectory(
        order, repository_root=tmp_path, execute=forbidden_execute
    )
    assert not called
    assert first == second
    assert first.status == COMPLETED_UNSCORED
    assert first.run.path == order.run_path
    assert not part.exists()
    receipt = json.loads((tmp_path / first.receipt_path).read_text())
    assert receipt["schema"] == "paper_ii_avqds_trajectory_completion_receipt_v1"
    assert receipt["method"] == ARM_AVQDS
    assert receipt["bindings_sha256"] == context["bindings"].digest
    assert receipt["reference_identity_sha256"] == (
        context["bindings"].reference_identity_sha256
    )
    assert "reference_receipt_sha256" not in receipt
    assert "action_registry" not in receipt
    assert receipt["exact_reference_runtime_artifacts_present"] is False
    assert receipt["exact_reference_binding_mode"] == "opaque_identity_only"
    assert receipt["remote_cleanup_authorized"] is False
    assert receipt["qiskit_compilation_used"] is False


def test_descriptor_is_inert_queue_one_with_scientific_identity_classads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, order = _prefix_order(tmp_path, monkeypatch)
    text = _descriptor(
        output_relative="chtc/scientific_avqds_test",
        order=order,
        source_epoch_id=context["source"].source_epoch_id,
        source_archive_relative="chtc/runtime_source.tar.gz",
        image_relative="containers/runtime.sif",
        resources=AVQDSTrajectoryCHTCResources(),
    ).decode()
    assert text.endswith("queue 1\n")
    assert f'+PaperIIActionId = "{order.request.action_id}"' in text
    assert f'+PaperIIExecutionRequestSha256 = "{order.request.digest}"' in text
    assert f'+PaperIIBindingsSha256 = "{order.bindings.digest}"' in text
    assert "+PaperIIExactReferenceBlind = True" in text
    assert "+PaperIIProductionSubmissionEnabled = False" in text
    assert "condor_submit" not in text
    validate_submit_lifecycle(text)

    script = package_worker(
        order=order,
        source_archive_name="runtime_source.tar.gz",
        source_archive_sha256="1" * 64,
        image_name="runtime.sif",
        image_sha256="2" * 64,
        rows={
            "input/work_order.json": {
                "path": "input/work_order.json",
                "sha256": "3" * 64,
                "bytes": 1,
                "mode": 0o644,
            }
        },
    )
    script_path = tmp_path / "worker.sh"
    script_path.write_bytes(script)
    subprocess.run(("bash", "-n", script_path), check=True)
    assert "PAPER_II_VERIFIED_PACKAGED_SOURCE_SHA256" in script.decode()


def test_pure_hubbard_gate_fails_closed_on_missing_declaration() -> None:
    class Reference:
        model = "hubbard"

    with pytest.raises(ProspectiveExecutionError, match="declared complete filtered pool"):
        worker._pure_hubbard_gate(
            {"candidate_pool_source": {"candidate_pool_complete": True}},
            Reference(),
        )
