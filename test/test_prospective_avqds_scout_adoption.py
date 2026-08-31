"""Bootstrap-scout adoption into the typed scientific lifecycle."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.prospective_avqds_scout_adoption import (
    SCOUT_ADOPTION_STATUS,
    ScoutAdoptionReceipt,
    adopt_completed_scout,
)
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    build_work_order,
    materialize_scout,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ProspectiveExecutionError,
    build_scientific_bindings,
)
from pipelines.time_dynamics.runners.adopt_prospective_avqds_scout import main
from test_prospective_avqds_scout_worker import (
    _fake_raw,
    _git,
    _receipts_and_request,
    _write,
)
from test_prospective_execution_bindings import _reference_receipt


def _context(root: Path):
    source, seed, bootstrap_request = _receipts_and_request(root)
    order = build_work_order(
        bootstrap_request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=root,
    )
    calls = 0

    def execute(actual_order, repository_root: Path) -> None:
        nonlocal calls
        calls += 1
        _write(
            repository_root / actual_order.run_part_path,
            _fake_raw(actual_order, seed, source.source_epoch),
        )

    completion = materialize_scout(
        order, repository_root=root, execute=execute
    )
    registry = build_identity_registry()
    reference_identity = next(
        row
        for row in registry.references
        if row.cell_id == order.reference.cell_id
    )
    seed_identity = next(
        row for row in registry.seeds if row.seed_id == reference_identity.seed_id
    )
    reference = _reference_receipt(
        root,
        seed,
        source_epoch=source.source_epoch,
        reference_identity=reference_identity,
        seed_identity=seed_identity,
        token="adoption",
    )
    bindings = build_scientific_bindings(
        reference=reference_identity,
        seed=seed_identity,
        seed_receipt=seed,
        reference_receipt=reference,
        source_epoch=source.source_epoch,
        source_epoch_receipt=source,
    )
    return source, seed, reference, bindings, order, completion, lambda: calls


def test_adoption_reuses_one_verified_scout_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source, seed, reference, bindings, order, completion, calls = _context(
        tmp_path
    )
    first = adopt_completed_scout(
        bootstrap_order=order,
        source_receipt=source,
        seed_receipt=seed,
        reference_receipt=reference,
        bindings=bindings,
        repository_root=tmp_path,
    )
    second = adopt_completed_scout(
        bootstrap_order=order,
        source_receipt=source,
        seed_receipt=seed,
        reference_receipt=reference,
        bindings=bindings,
        repository_root=tmp_path,
    )
    assert calls() == 1
    assert first == second
    assert first.bootstrap_action_id == order.request.action_id
    assert first.run == completion.run

    completed = json.loads(
        (tmp_path / first.completed_lifecycle.path).read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (tmp_path / completed["state_evidence"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    receipt = json.loads(
        (tmp_path / first.receipt_path).read_text(encoding="utf-8")
    )
    assert receipt["status"] == SCOUT_ADOPTION_STATUS
    assert receipt["scheduler_submission_performed"] is False
    assert receipt["remote_raw_evidence_policy"] == "preserve"
    assert receipt["scientific_qualification"] is None
    assert evidence["scheduler_submission_performed"] is False
    assert evidence["bootstrap_action_id"] == order.request.action_id
    assert evidence["reused_run"] == completion.run.to_json_dict()


def test_adoption_rejects_bootstrap_scientific_cut_drift(tmp_path: Path) -> None:
    source, seed, reference, bindings, order, _completion, _calls = _context(
        tmp_path
    )
    inputs = dict(order.request.inputs)
    inputs["activation_cut"] = float(inputs["activation_cut"]) / 10.0
    drifted_request = replace(order.request, inputs=inputs)
    drifted_order = replace(order, request=drifted_request)
    with pytest.raises(ProspectiveExecutionError, match="physically identical"):
        adopt_completed_scout(
            bootstrap_order=drifted_order,
            source_receipt=source,
            seed_receipt=seed,
            reference_receipt=reference,
            bindings=bindings,
            repository_root=tmp_path,
        )


def test_adoption_receipt_rejects_tampered_reuse_disposition(
    tmp_path: Path,
) -> None:
    source, seed, reference, bindings, order, _completion, _calls = _context(
        tmp_path
    )
    adopted = adopt_completed_scout(
        bootstrap_order=order,
        source_receipt=source,
        seed_receipt=seed,
        reference_receipt=reference,
        bindings=bindings,
        repository_root=tmp_path,
    )
    path = tmp_path / adopted.receipt_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scheduler_submission_performed"] = True
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ProspectiveExecutionError, match="digest mismatch"):
        ScoutAdoptionReceipt.load(
            path,
            repository_root=tmp_path,
            bootstrap_order=order,
            source_receipt=source,
            seed_receipt=seed,
            reference_receipt=reference,
            bindings=bindings,
        )


def test_adoption_cli_materializes_then_validates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, seed, reference, _bindings, order, _completion, _calls = _context(
        tmp_path
    )
    order_path = tmp_path / "orders/scout.json"
    from pipelines.time_dynamics.prospective_avqds_scout_worker import (
        write_work_order,
    )

    write_work_order(order_path, order)
    arguments = [
        "--repository-root",
        str(tmp_path),
        "--work-order",
        str(order_path),
        "--reference-receipt",
        str(tmp_path / reference.receipt_path),
    ]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "validated"
    assert main(
        [
            *arguments,
            "--validate-receipt",
            str(tmp_path / first["scout_adoption_receipt"]),
        ]
    ) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first


def test_adoption_cli_replays_submitted_scout_after_overlay_evolves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source, _seed, reference, _bindings, order, _completion, _calls = _context(
        tmp_path
    )
    order_path = tmp_path / "orders/scout.json"
    from pipelines.time_dynamics.prospective_avqds_scout_worker import (
        write_work_order,
    )

    write_work_order(order_path, order)
    evolved = tmp_path / "pipelines/scaffold/runtime_loader.py"
    evolved.write_text(
        evolved.read_text(encoding="utf-8")
        + "\n# post-submission committed overlay evolution\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", "pipelines/scaffold/runtime_loader.py")
    _git(tmp_path, "commit", "-qm", "evolve overlay after scout submission")

    assert main(
        [
            "--repository-root",
            str(tmp_path),
            "--work-order",
            str(order_path),
            "--reference-receipt",
            str(tmp_path / reference.receipt_path),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "validated"
    assert result["completed_lifecycle"]["path"].endswith(
        "completed_lifecycle.json"
    )
