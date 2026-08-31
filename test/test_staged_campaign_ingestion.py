"""Paper-II hook bindings for the ingestion transaction engine.

Guard behavior is tested per hook; the heavy interior of ``ledger_append``
(achieved-configuration validation and exact evaluation inside
``record_prepared_action_outputs``) already has its own suite in
``test_staged_campaign_artifacts.py`` and is not re-tested through the engine.
The end-to-end test binds the real reconcile/storage/transfer/extract hooks on
a synthetic delta archive and stubs only the ledger and report phases.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from pipelines.artifacts.ingestion import (
    TransactionSpec,
    newly_completed,
    run_ingestion_transaction,
)
from pipelines.time_dynamics.staged_campaign_ingestion import (
    archive_verification_hook,
    build_paper_ii_hooks,
    extraction_hook,
    ledger_append_hook,
    reconcile_hook,
    report_refresh_hook,
)
from pipelines.artifacts.ingestion import PhaseFailed


def _spec(tmp_path: Path, processes=(0, 1)) -> TransactionSpec:
    return TransactionSpec(
        campaign_id="unit",
        cluster_id=42,
        process_ids=tuple(processes),
        increment_root=tmp_path / "attempts/cluster_42/increments/delta",
        owner="test",
    )


def _make_archive(path: Path, members: dict[str, bytes]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, blob in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- reconcile: the in-lock race check -------------------------------------


def test_reconcile_refuses_a_delta_covered_by_a_racing_receipt(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    winner = (
        tmp_path / "attempts/cluster_42/increments/other/evidence/increment_receipt.json"
    )
    winner.parent.mkdir(parents=True)
    winner.write_text(json.dumps({"cluster_id": 42, "process_ids": [0]}))
    with pytest.raises(PhaseFailed, match="delta_already_covered"):
        reconcile_hook(spec)({})


def test_reconcile_passes_an_uncovered_delta(tmp_path: Path) -> None:
    payload = reconcile_hook(_spec(tmp_path))({})
    assert payload["newly_completed"] == [0, 1]


# --- transfer: integrity ---------------------------------------------------


def test_archive_verification_records_hashes(tmp_path: Path) -> None:
    archive = tmp_path / "delta.tar.gz"
    sha = _make_archive(archive, {"raw_outputs/a/run.json": b"{}"})
    payload = archive_verification_hook(archive, expected_sha256=sha)({})
    assert payload["archive_sha256"] == sha
    assert payload["archive_member_count"] == 1


def test_archive_verification_fails_on_sha_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "delta.tar.gz"
    _make_archive(archive, {"x": b"1"})
    with pytest.raises(PhaseFailed, match="sha256_mismatch"):
        archive_verification_hook(archive, expected_sha256="0" * 64)({})


def test_archive_verification_fails_on_corrupt_gzip(tmp_path: Path) -> None:
    archive = tmp_path / "delta.tar.gz"
    _make_archive(archive, {"x": b"1"})
    blob = bytearray(archive.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    archive.write_bytes(bytes(blob))
    with pytest.raises(PhaseFailed, match="gzip_integrity|tar_integrity|sha256"):
        archive_verification_hook(archive)({})


# --- extract: path confinement --------------------------------------------


def test_extraction_confines_members_to_the_root(tmp_path: Path) -> None:
    archive = tmp_path / "delta.tar.gz"
    _make_archive(archive, {"../escape.txt": b"nope"})
    with pytest.raises(PhaseFailed, match="path_escape"):
        extraction_hook(archive, tmp_path / "out")({})


def test_extraction_extracts_and_counts(tmp_path: Path) -> None:
    archive = tmp_path / "delta.tar.gz"
    _make_archive(archive, {"a/run.json": b"{}", "a/log.txt": b"ok"})
    payload = extraction_hook(archive, tmp_path / "out")({})
    assert payload["extracted_members"] == 2
    assert (tmp_path / "out/a/run.json").exists()


def test_extraction_rejects_symbolic_and_hard_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "links.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        link = tarfile.TarInfo("a/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../escape"
        archive.addfile(link)
    with pytest.raises(PhaseFailed, match="unsafe_tar_member"):
        extraction_hook(archive_path, tmp_path / "out")({})


# --- ledger_append: contract with the extract phase ------------------------


def test_ledger_append_requires_an_extraction_root(tmp_path: Path) -> None:
    hook = ledger_append_hook(
        campaign=None, ledger_path=tmp_path / "ledger.json", action_rows=(),
        seed_path=tmp_path / "seed.json", numerics="n", gate="g",
        search_profile="p",
    )
    with pytest.raises(PhaseFailed, match="no_extraction_root"):
        hook({})


def test_ledger_append_exposes_process_coverage_and_scientific_state_binding(
    tmp_path: Path, monkeypatch
) -> None:
    extraction_root = tmp_path / "verified_extract"
    action_rows = (
        {"action_id": "action-a"},
        {"action_id": "action-b"},
    )
    run_paths = tuple(
        extraction_root / f"raw_outputs/campaign/{row['action_id']}/run.json"
        for row in action_rows
    )
    for index, run_path in enumerate(run_paths):
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps({"index": index}), encoding="utf-8")

    class FakeLedger:
        records = ()

        def write(self, path: Path) -> None:
            path.write_text('{"records": []}\n', encoding="utf-8")

    def fake_record_outputs(**_kwargs):
        return FakeLedger(), tuple(
            {
                "action_id": row["action_id"],
                "run_path": str(run_path),
                "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                "status": "recorded",
                "ledger_status": "requires_threshold_extension",
            }
            for row, run_path in zip(action_rows, run_paths, strict=True)
        )

    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign.CampaignLedger", FakeLedger
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign_artifacts.record_prepared_action_outputs",
        fake_record_outputs,
    )
    ledger_path = tmp_path / "ledger.json"
    payload = ledger_append_hook(
        campaign=None,
        ledger_path=ledger_path,
        action_rows=action_rows,
        process_ids=(3, 7),
        seed_path=tmp_path / "seed.json",
        numerics="n",
        gate="g",
        search_profile="p",
        record_terminal_logical_resources=False,
    )({"extract": {"extraction_root": str(extraction_root)}})

    assert payload["action_dispositions"] == [
        {
            "process_id": process_id,
            "action_id": row["action_id"],
            "run_path": str(run_path),
            "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
            "status": "recorded",
            "ledger_status": "requires_threshold_extension",
        }
        for process_id, row, run_path in zip(
            (3, 7), action_rows, run_paths, strict=True
        )
    ]
    assert payload["scientific_state_after"] == {
        "path": str(ledger_path),
        "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
    }


def test_ledger_append_records_logical_resources_before_committing(
    tmp_path: Path, monkeypatch
) -> None:
    extraction_root = tmp_path / "verified_extract"
    run_path = extraction_root / "raw_outputs/campaign/action-a/run.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text('{"valid": true}\n', encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"

    class FakeLedger:
        records = ()

        def write(self, path: Path) -> None:
            assert run_path.with_name(
                "terminal_logical_graph_resources.json"
            ).is_file()
            path.write_text('{"records": []}\n', encoding="utf-8")

    def fake_record_outputs(**_kwargs):
        return FakeLedger(), (
            {
                "action_id": "action-a",
                "run_path": str(run_path),
                "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                "status": "recorded",
            },
        )

    def fake_resource_record(path: Path):
        assert path == run_path
        return {"resource_sha256": "a" * 64, "terminal_resource_tuple": {}}

    def fake_resource_write(path: Path, payload):
        assert payload["resource_sha256"] == "a" * 64
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign.CampaignLedger", FakeLedger
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign_artifacts.record_prepared_action_outputs",
        fake_record_outputs,
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources.build_terminal_logical_graph_resource_record",
        fake_resource_record,
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources.write_terminal_logical_graph_resource_record",
        fake_resource_write,
    )

    payload = ledger_append_hook(
        campaign=None,
        ledger_path=ledger_path,
        action_rows=({"action_id": "action-a"},),
        process_ids=(3,),
        seed_path=tmp_path / "seed.json",
        numerics="n",
        gate="g",
        search_profile="p",
    )({"extract": {"extraction_root": str(extraction_root)}})

    resource = payload["action_dispositions"][0][
        "terminal_logical_graph_resources"
    ]
    assert resource["path"] == str(
        run_path.with_name("terminal_logical_graph_resources.json")
    )
    assert resource["resource_sha256"] == "a" * 64
    assert resource["sha256"] == hashlib.sha256(
        run_path.with_name("terminal_logical_graph_resources.json").read_bytes()
    ).hexdigest()


def test_ledger_append_does_not_commit_before_process_action_validation(
    tmp_path: Path, monkeypatch
) -> None:
    extraction_root = tmp_path / "verified_extract"
    run_path = extraction_root / "raw_outputs/campaign/action-a/run.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text('{"valid": true}\n', encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"
    original = b'{"records": ["prior"]}\n'
    ledger_path.write_bytes(original)

    class FakeLedger:
        records = ()

        @classmethod
        def read(cls, _path: Path):
            return cls()

        def write(self, path: Path) -> None:
            path.write_text('{"records": ["mutated"]}\n', encoding="utf-8")

    def fake_record_outputs(**_kwargs):
        return FakeLedger(), (
            {
                "action_id": "wrong-action",
                "run_path": str(run_path),
                "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                "status": "recorded",
            },
        )

    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign.CampaignLedger", FakeLedger
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign_artifacts.record_prepared_action_outputs",
        fake_record_outputs,
    )
    hook = ledger_append_hook(
        campaign=None,
        ledger_path=ledger_path,
        action_rows=({"action_id": "action-a"},),
        process_ids=(3,),
        seed_path=tmp_path / "seed.json",
        numerics="n",
        gate="g",
        search_profile="p",
    )
    with pytest.raises(PhaseFailed, match="action_coverage_mismatch"):
        hook({"extract": {"extraction_root": str(extraction_root)}})
    assert ledger_path.read_bytes() == original


# --- report_refresh: crash becomes a named guard ---------------------------


def test_report_crash_is_a_named_guard_not_a_traceback() -> None:
    def broken():
        raise RuntimeError("latex exploded")

    with pytest.raises(PhaseFailed, match="report_builder_error"):
        report_refresh_hook(broken)({})


# --- end to end on a synthetic delta ---------------------------------------


def test_end_to_end_transaction_with_real_transport_hooks(tmp_path: Path) -> None:
    """Real reconcile/storage/transfer/extract; stubbed ledger and report."""

    spec = _spec(tmp_path)
    archive = tmp_path / "delta.tar.gz"
    sha = _make_archive(
        archive, {"raw_outputs/act_a/run.json": b"{}", "logs/act_a.log": b"ok"}
    )
    hooks = build_paper_ii_hooks(
        spec,
        campaign=None,
        ledger_path=tmp_path / "ledger.json",
        archive_path=archive,
        action_rows=(),
        seed_path=tmp_path / "seed.json",
        numerics="n", gate="g", search_profile="p",
        expected_archive_sha256=sha,
        report_refresh=lambda: {"report_manifest_sha256": "stub"},
    )
    hooks["ledger_append"] = lambda context: {
        "ledger_record_count_after": 1,
        "extraction_root_seen": context["extract"]["extraction_root"],
    }
    result = run_ingestion_transaction(spec, hooks)
    assert result.status == "ingested"
    receipt = result.receipt
    assert receipt["transfer"]["archive_sha256"] == sha
    assert receipt["extract"]["extracted_members"] == 2
    assert receipt["ledger_append"]["extraction_root_seen"].endswith("evidence/extracted")
    assert receipt["report_refresh"]["report_manifest_sha256"] == "stub"
    # the receipt now covers the delta for every future reconciliation
    assert newly_completed(42, [0, 1], [receipt]) == ()
    # and a second run is a no-op
    assert run_ingestion_transaction(spec, hooks).status == "ingested"
