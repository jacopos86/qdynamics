"""Bootstrap source/package provenance before prospective lane activation."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from pipelines.time_dynamics.prospective_bootstrap_provenance import (
    BOOTSTRAP_INTENT_FIRST_REQUIREMENTS,
    BootstrapPackageManifest,
    BootstrapPreSubmitAudit,
    BootstrapProvenanceError,
    main,
    write_bootstrap_package_manifest,
    write_bootstrap_pre_submit_audit,
    write_source_epoch_contracts,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _committed_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test Author")
    (repo / "worker.py").write_text("print('frozen')\n", encoding="utf-8")
    _git(repo, "add", "worker.py")
    _git(repo, "commit", "-qm", "frozen source")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_source_epoch_builder_is_deterministic_and_loader_compatible(
    tmp_path: Path,
) -> None:
    repo, revision = _committed_repository(tmp_path)
    output = repo / "output/paper_ii_hh_product_v1/source_epochs/frozen"

    first = write_source_epoch_contracts(
        repository_root=repo,
        lane_id="paper_ii_hh_product_v1",
        source_revision=revision,
        output_dir=output,
    )
    (repo / "worker.py").write_text("print('dirty')\n", encoding="utf-8")
    second = write_source_epoch_contracts(
        repository_root=repo,
        lane_id="paper_ii_hh_product_v1",
        source_revision=revision,
        output_dir=output,
    )

    assert first == second
    assert first.source_epoch.source_revision == revision
    assert first.source_epoch.source_archive_sha256 is not None
    assert first.source_tree_sha256
    assert SourceEpochReceipt.load(
        repo / first.receipt_path,
        repository_root=repo,
        expected_epoch=first.source_epoch,
    ) == first


def test_source_epoch_builder_rejects_abbreviated_revision(tmp_path: Path) -> None:
    repo, revision = _committed_repository(tmp_path)

    with pytest.raises(
        BootstrapProvenanceError, match="full 40- or 64-character"
    ):
        write_source_epoch_contracts(
            repository_root=repo,
            lane_id=LANE_HH,
            source_revision=revision[:12],
            output_dir=repo / "output/source",
        )


def _bootstrap_request(source: SourceEpochReceipt):
    plan = plan_bootstrap_actions(
        build_identity_registry(),
        evidence=BootstrapEvidence(source_epochs={LANE_HH: source.source_epoch}),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(row for row in plan.ready_actions if row.lane_id == LANE_HH)
    return resolve_bootstrap_action(action)


def _write_package(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, content in (
            ("submit.sub", b"queue 1\n"),
            ("input/request.json", b"{}\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _write_disabled_registry(path: Path, lane_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "time_dynamics_acquisition_registry_v1",
                "registry_id": "bootstrap-local-only",
                "supervisor_state_root": "output/supervisor",
                "production_operations": None,
                "lanes": [
                    {
                        "lane_id": lane_id,
                        "kind": "prospective_bootstrap",
                        "monitor_enabled": False,
                        "submission_enabled": False,
                        "attempts_root": f"output/{lane_id}/attempts",
                        "state_root": f"output/{lane_id}/state",
                        "scientific_state": f"output/{lane_id}/scientific.json",
                        "report_contract": f"output/{lane_id}/report.json",
                        "planner_authority": (
                            "pipelines.time_dynamics.prospective_scientific_planner:"
                            "plan_prospective_cell"
                        ),
                        "blocked_by": ["bootstrap_not_activated"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _prepared_bootstrap(tmp_path: Path):
    repo, revision = _committed_repository(tmp_path)
    source = write_source_epoch_contracts(
        repository_root=repo,
        lane_id=LANE_HH,
        source_revision=revision,
        output_dir=repo / "output/source",
    )
    request = _bootstrap_request(source)
    package = repo / "packages/seed-bootstrap.tar.gz"
    _write_package(package)
    package_manifest = write_bootstrap_package_manifest(
        repo / "packages/seed-bootstrap.manifest.json",
        request=request,
        source_epoch_receipt=source,
        package_path=package,
        submit_descriptor="submit.sub",
        job_batch_name="paper_ii_hh_seed_bootstrap_v1",
        repository_root=repo,
    )
    registry = repo / "registry.json"
    _write_disabled_registry(registry, request.lane_id)
    return repo, source, request, package, package_manifest, registry


def test_bootstrap_audit_binds_exact_request_package_and_disabled_registry(
    tmp_path: Path,
) -> None:
    repo, source, request, _, package_manifest, registry = _prepared_bootstrap(
        tmp_path
    )

    audit = write_bootstrap_pre_submit_audit(
        repo / "packages/seed-bootstrap.audit.json",
        request=request,
        source_epoch_receipt=source,
        package_manifest=package_manifest,
        registry_path=registry,
        repository_root=repo,
    )
    reloaded = BootstrapPreSubmitAudit.load(
        repo / audit.receipt_path, repository_root=repo
    )

    assert reloaded == audit
    assert audit.status == "passed_local_only"
    assert audit.execution_request_sha256 == request.digest
    assert audit.production_submission_enabled is False
    assert audit.scheduler_submission_authorized is False
    assert audit.intent_first == BOOTSTRAP_INTENT_FIRST_REQUIREMENTS


def test_bootstrap_package_manifest_rejects_changed_archive_bytes(
    tmp_path: Path,
) -> None:
    repo, _, _, package, manifest, _ = _prepared_bootstrap(tmp_path)
    package.write_bytes(package.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        BootstrapPackageManifest.load(
            repo / manifest.manifest_path, repository_root=repo
        )


def test_bootstrap_package_manifest_rejects_qiskit_flag(tmp_path: Path) -> None:
    repo, _, _, _, manifest, _ = _prepared_bootstrap(tmp_path)
    path = repo / manifest.manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["qiskit_compilation"] = True
    payload.pop("bootstrap_package_manifest_sha256")
    payload["bootstrap_package_manifest_sha256"] = _canonical_sha(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(BootstrapProvenanceError, match="not local-only"):
        BootstrapPackageManifest.load(path, repository_root=repo)


def test_bootstrap_audit_rejects_submission_enabled_registry(
    tmp_path: Path,
) -> None:
    repo, source, request, _, package_manifest, registry = _prepared_bootstrap(
        tmp_path
    )
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["lanes"][0]["monitor_enabled"] = True
    payload["lanes"][0]["submission_enabled"] = True
    payload["lanes"][0]["blocked_by"] = []
    registry.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        BootstrapProvenanceError, match="production to remain disabled"
    ):
        write_bootstrap_pre_submit_audit(
            repo / "packages/seed-bootstrap.audit.json",
            request=request,
            source_epoch_receipt=source,
            package_manifest=package_manifest,
            registry_path=registry,
            repository_root=repo,
        )


def test_bootstrap_audit_rejects_changed_intent_first_contract(
    tmp_path: Path,
) -> None:
    repo, source, request, _, package_manifest, registry = _prepared_bootstrap(
        tmp_path
    )
    audit = write_bootstrap_pre_submit_audit(
        repo / "packages/seed-bootstrap.audit.json",
        request=request,
        source_epoch_receipt=source,
        package_manifest=package_manifest,
        registry_path=registry,
        repository_root=repo,
    )
    path = repo / audit.receipt_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["intent_first"]["first_snapshot_outcome"] = "verified_absent"
    payload.pop("bootstrap_pre_submit_audit_sha256")
    payload["bootstrap_pre_submit_audit_sha256"] = _canonical_sha(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        BootstrapProvenanceError, match="passed local-only contract"
    ):
        BootstrapPreSubmitAudit.load(path, repository_root=repo)


def test_package_and_audit_cli_round_trip(tmp_path: Path, capsys) -> None:
    repo, source, request, package, _, registry = _prepared_bootstrap(tmp_path)
    request_path = repo / "packages/request.json"
    request_path.write_text(
        json.dumps(request.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = repo / "packages/from-cli.manifest.json"

    assert (
        main(
            [
                "package-manifest",
                "--repo-root",
                str(repo),
                "--request",
                str(request_path),
                "--source-epoch-receipt",
                str(repo / source.receipt_path),
                "--package",
                str(package),
                "--submit-descriptor",
                "submit.sub",
                "--job-batch-name",
                "paper_ii_hh_seed_bootstrap_v1",
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )
    package_summary = json.loads(capsys.readouterr().out)
    assert package_summary["production_submission_enabled"] is False

    audit_path = repo / "packages/from-cli.audit.json"
    assert (
        main(
            [
                "audit",
                "--repo-root",
                str(repo),
                "--request",
                str(request_path),
                "--source-epoch-receipt",
                str(repo / source.receipt_path),
                "--package-manifest",
                str(manifest_path),
                "--registry",
                str(registry),
                "--output",
                str(audit_path),
            ]
        )
        == 0
    )
    audit_summary = json.loads(capsys.readouterr().out)
    assert audit_summary["production_submission_enabled"] is False
    assert audit_summary["scheduler_submission_authorized"] is False
