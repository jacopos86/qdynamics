from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from pipelines.reporting import (
    watch_paper_i_page12_insertion_comparator_snapshot as watcher,
)
from pipelines.reporting import (
    append_paper_i_insertion_comparator_snapshot_pages as macro_snapshot,
)

_FINALIZER_FIXTURE_PATH = Path(__file__).parent / (
    "chtc/test_finalize_page12_insertion_comparator_closure.py"
)
_FINALIZER_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "paper_i_page12_finalizer_test_fixture", _FINALIZER_FIXTURE_PATH
)
assert _FINALIZER_FIXTURE_SPEC is not None and _FINALIZER_FIXTURE_SPEC.loader is not None
finalizer_fixture = importlib.util.module_from_spec(_FINALIZER_FIXTURE_SPEC)
sys.modules[_FINALIZER_FIXTURE_SPEC.name] = finalizer_fixture
_FINALIZER_FIXTURE_SPEC.loader.exec_module(finalizer_fixture)


def _write_digested(path: Path, unsigned: dict[str, object]) -> dict[str, object]:
    value = {**unsigned, "sha256": watcher._canonical_sha256(unsigned)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return value


def _write_pdf(path: Path, payloads: list[bytes]) -> None:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = pypdf.PdfWriter()
    for index, payload in enumerate(payloads, 1):
        page = writer.add_blank_page(width=600 + index, height=800)
        stream = DecodedStreamObject()
        stream.set_data(payload)
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def _content_hashes(path: Path) -> list[str]:
    pypdf = pytest.importorskip("pypdf")
    result: list[str] = []
    for page in pypdf.PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        payload = b"" if contents is None else contents.get_data()
        result.append(hashlib.sha256(payload).hexdigest())
    return result


def _summary(*, exact: float = -1.0) -> dict[str, object]:
    return {
        "schema": "paper_i_run_summary_v1",
        "provenance": {"exact_same_cutoff_energy": exact},
        "accepted_error_trace": [
            {
                "controller_round": k,
                "accepted_energy": exact + 1.0 / (k + 1),
                "absolute_energy_error": 1.0 / (k + 1),
                "active_ansatz_depth": k,
            }
            for k in range(1, 51)
        ],
    }


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_once_publishes_pending_snapshot_before_a_receipt_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    written: list[dict[str, object]] = []
    monkeypatch.setattr(
        watcher,
        "_write_status",
        lambda value: written.append(copy.deepcopy(dict(value))) or dict(value),
    )
    empty_revision = watcher._canonical_sha256({})
    reported_revisions = iter([(set(), None), (set(), empty_revision)])
    monkeypatch.setattr(watcher, "_reported_revision", lambda: next(reported_revisions))
    refreshed: list[list[dict[str, object]]] = []

    def refresh(results: list[dict[str, object]]) -> dict[str, object]:
        refreshed.append(copy.deepcopy(results))
        return {"status": "updated_existing_report_in_place", "page_count": 18}

    monkeypatch.setattr(watcher, "refresh_report", refresh)

    assert watcher.watch(receipt_dir=receipt_dir, poll_seconds=30.0, once=True) == 0
    assert refreshed == [[]]
    assert written[-1]["status"] == "waiting_for_first_authenticated_receipt"
    assert written[-1]["authenticated_receipt_count"] == 0
    assert written[-1]["receipt_evidence_revision"] == empty_revision
    assert written[-1]["reported_evidence_revision"] == empty_revision


def test_zero_receipt_adapter_keeps_both_baselines_and_all_cells_pending() -> None:
    adapter = watcher.build_adapter([])

    assert adapter["status"] == "provisional_page12_0_of_12_authenticated"
    assert adapter["campaign_counts"] == {
        "planned_comparator_cells": 12,
        "authenticated_comparator_cells": 0,
        "always_insertion_authenticated": 0,
        "append_always_authenticated": 0,
        "pending_comparator_cells": 12,
    }
    assert len(adapter["reference_cells"]) == 6
    assert all(
        len(row["points"]) == 50
        and len(row["current_adapt"]["points"]) == 51
        and row["plotted_horizon"] == 50
        and row["current_adapt"]["plotted_horizon"] == 50
        for row in adapter["reference_cells"]
    )
    assert all(
        row["always_insertion"] == "pending / awaiting authenticated receipt"
        and row["append_always"] == "pending / awaiting authenticated receipt"
        for row in adapter["matrix"]
    )


def test_receipt_loader_rejects_path_only_claim_without_finalizer_contract(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "fake.json"
    _write_digested(
        receipt,
        {
            "schema": watcher.RECEIPT_SCHEMA,
            "status": watcher.RECEIPT_STATUS,
            "cluster_id": watcher.CLUSTER_ID,
            "proc_id": 0,
            "run_id": "not-an-authorized-run",
            "regime_id": "weak_weak",
            "comparator_policy": watcher.EXPECTED_POLICIES[0],
            "archive": {"path": "somewhere/result.tar.gz"},
        },
    )

    with pytest.raises(watcher.WatchError, match="finalizer authentication"):
        watcher.authenticate_receipt(receipt)


def test_watcher_pins_the_exact_finalizer_contract_source() -> None:
    assert watcher._sha256_file(watcher.FINALIZER_PATH) == (
        watcher.EXPECTED_FINALIZER_SHA256
    )


def test_direct_script_entrypoint_can_resolve_lazy_reporting_imports() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(watcher.__file__).resolve()),
            "--help",
        ],
        cwd=Path(watcher.__file__).resolve().parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert str(watcher.REPO_ROOT) in proc.stdout


def test_authenticated_receipt_streams_bound_summary_from_full_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = "fixture-run"
    run_root = f"runs/{run_id}"
    summary_bytes = json.dumps(_summary(), separators=(",", ":")).encode() + b"\n"
    role_paths = {
        "checkpoint": f"{run_root}/checkpoints/current.json",
        "estimator_ledger": f"{run_root}/result/estimator_ledger.json",
        "result": f"{run_root}/result/result.json",
        "summary": f"{run_root}/summary/summary.json",
    }
    scientific_members = {
        role_paths["checkpoint"]: b'{"checkpoint":true}\n',
        role_paths["estimator_ledger"]: b'{"ledger":true}\n',
        role_paths["result"]: b'{"result":true}\n',
        role_paths["summary"]: summary_bytes,
    }
    manifest_unsigned: dict[str, object] = {
        "schema": (
            "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_execution_manifest_v1"
        ),
        "status": "passed",
        "execution_id": run_id,
        "job_spec_sha256": "1" * 64,
        "authorization_sha256": "2" * 64,
        "protocol_sha256": "4" * 64,
        "route_contract_sha256": "5" * 64,
        "target_horizon": 50,
        "comparator_policy": watcher.EXPECTED_POLICIES[0],
        "controller_rounds_completed": 50,
        "fresh_start": True,
        "source_checkpoint_consumed": False,
        "output_payloads": {
            role: {
                "path": role_paths[role],
                "sha256": hashlib.sha256(
                    scientific_members[role_paths[role]]
                ).hexdigest(),
                "size_bytes": len(scientific_members[role_paths[role]]),
            }
            for role in role_paths
        },
    }
    manifest = {
        **manifest_unsigned,
        "sha256": watcher._canonical_sha256(manifest_unsigned),
    }
    preliminary_members = {
        **scientific_members,
        f"{run_root}/execution_manifest.json": (
            json.dumps(manifest, separators=(",", ":")).encode() + b"\n"
        ),
    }
    worker_unsigned: dict[str, object] = {
        "schema": (
            "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_worker_receipt_v1"
        ),
        "status": "passed",
        "execution_id": run_id,
        "job_spec_sha256": "1" * 64,
        "authorization_sha256": "2" * 64,
        "execution_manifest_sha256": manifest["sha256"],
        "controller_rounds_completed": 50,
        "fresh_start": True,
        "artifacts": [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in preliminary_members.items()
        ],
    }
    worker = {
        **worker_unsigned,
        "sha256": watcher._canonical_sha256(worker_unsigned),
    }
    members = {
        "worker_exit_status.txt": b"0\n",
        **preliminary_members,
        "worker_receipt.json": (
            json.dumps(worker, separators=(",", ":")).encode() + b"\n"
        ),
    }
    archive = tmp_path / f"{run_id}__9647385__0.tar.gz"
    archive.write_bytes(_tar_bytes(members))
    inventory = [
        {
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in members.items()
    ]
    job = {
        "execution_id": run_id,
        "regime_id": "weak_weak",
        "comparator_policy": watcher.EXPECTED_POLICIES[0],
        "target_horizon": 50,
        "sha256": "1" * 64,
        "protocol_sha256": "4" * 64,
        "route_contract_sha256": "5" * 64,
        "protocol_path": "protocol.json",
        "typed_insertion_kind": watcher.EXPECTED_POLICIES[0],
        "runtime_insertion_mode": "full_commutation_reduced",
        "expected_run_artifacts": {
            **{
                role: {"path": path}
                for role, path in role_paths.items()
            },
            "execution_manifest": {
                "path": f"{run_root}/execution_manifest.json"
            },
        },
    }
    package = {"sha256": "6" * 64}
    monkeypatch.setattr(
        watcher,
        "_authorized_job",
        lambda _run_id: (tmp_path / "job.json", job, package),
    )
    monkeypatch.setattr(watcher, "RETRIEVED_DIR", tmp_path)
    monkeypatch.setattr(watcher, "IDENTITY_DIR", tmp_path)

    remote_path = f"osdf:///outputs/transfer/{archive.name}"
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    def local_binding(
        value: object,
        *,
        label: str,
        expected_path: Path | None = None,
        canonical: bool = True,
    ) -> tuple[Path, dict[str, object]]:
        assert isinstance(value, dict)
        path = expected_path or tmp_path / "bound.json"
        if label == "receipt activation":
            payload: dict[str, object] = {
                "package_manifest_sha256": package["sha256"],
                "sha256": watcher.EXPECTED_ACTIVATION_SHA256,
                "schema": (
                    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                    "activation_manifest_v1"
                ),
                "status": "passed_activation_prepared_no_submission",
                "authorization_count": 12,
                "execution_authorized": True,
                "submission_authorized": True,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        elif label == "receipt authorization":
            payload = {
                "schema": (
                    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                    "execution_authorization_v1"
                ),
                "authorization_kind": (
                    "explicit_user_execution_and_submission_authority"
                ),
                "scope": "single_cell_chtc_execution_only",
                "execution_id": run_id,
                "job_spec_sha256": job["sha256"],
                "protocol_sha256": job["protocol_sha256"],
                "package_manifest_sha256": package["sha256"],
                "execution_authorized": True,
                "submission_authorized": True,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
                "sha256": value["canonical_sha256"],
            }
        elif label == "receipt remote/local identity":
            payload = {
                "schema": (
                    "paper_i_ra_adapt_page12_insertion_comparator_"
                    "remote_archive_identity_v1"
                ),
                "status": (
                    "passed_remote_local_size_sha256_match_after_atomic_rename"
                ),
                "cluster_id": watcher.CLUSTER_ID,
                "proc_id": 0,
                "execution_id": run_id,
                "remote_path": remote_path,
                "local_path": str(archive),
                "remote_size_bytes": archive.stat().st_size,
                "local_size_bytes": archive.stat().st_size,
                "remote_sha256": archive_sha,
                "local_sha256": archive_sha,
                "gzip_integrity_passed": True,
                "tar_readability_passed": True,
                "atomic_local_rename_completed": True,
                "sha256": value["canonical_sha256"],
            }
        else:
            payload = {"sha256": value["canonical_sha256"]}
        return path, payload

    monkeypatch.setattr(
        watcher,
        "_verify_local_binding",
        local_binding,
    )
    receipt_unsigned: dict[str, object] = {
        "schema": watcher.RECEIPT_SCHEMA,
        "status": watcher.RECEIPT_STATUS,
        "cluster_id": watcher.CLUSTER_ID,
        "proc_id": 0,
        "run_id": run_id,
        "regime_id": "weak_weak",
        "comparator_policy": watcher.EXPECTED_POLICIES[0],
        "controller_rounds_completed": 50,
        "package_manifest": {
            "canonical_sha256": package["sha256"],
            "sha256": "7" * 64,
            "size_bytes": 1,
        },
        "job": {
            "canonical_sha256": job["sha256"],
            "sha256": "8" * 64,
            "size_bytes": 1,
        },
        "protocol": {
            "canonical_sha256": job["protocol_sha256"],
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "route_contract_canonical_sha256": job["route_contract_sha256"],
        "typed_insertion_kind": job["typed_insertion_kind"],
        "runtime_insertion_mode": job["runtime_insertion_mode"],
        "activation_manifest": {"canonical_sha256": "9" * 64},
        "authorization": {"canonical_sha256": "2" * 64},
        "remote_local_identity_evidence": {
            "canonical_sha256": "b" * 64,
            "sha256": "c" * 64,
            "size_bytes": 1,
        },
        "archive": {
            "path": str(archive),
            "remote_path": remote_path,
            "sha256": archive_sha,
            "size_bytes": archive.stat().st_size,
            "inventory": inventory,
        },
        "worker_receipt": {
            "path_inside_archive": "worker_receipt.json",
            "schema": worker["schema"],
            "status": "passed",
            "canonical_sha256": worker["sha256"],
        },
        "execution_manifest": {
            "path_inside_archive": f"{run_root}/execution_manifest.json",
            "schema": manifest["schema"],
            "status": "passed",
            "canonical_sha256": manifest["sha256"],
        },
        "summary_json": {
            "path_inside_archive": role_paths["summary"],
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "size_bytes": len(summary_bytes),
        },
        "authentication_checks": {
            "full_regular_member_inventory_closed": True,
            "all_member_hashes_and_sizes_verified": True,
            "package_job_protocol_route_insertion_identity_closed": True,
            "activation_authorization_identity_closed": True,
        },
    }
    receipt = tmp_path / watcher.receipt_filename(0, run_id)
    _write_digested(receipt, receipt_unsigned)

    result = watcher.authenticate_receipt(receipt)

    assert result["run_id"] == run_id
    assert result["terminal"] == {"k": 50, "error": pytest.approx(1 / 51)}
    assert len(result["points"]) == 50
    assert result["receipt_sha256"] == watcher.load(receipt)["sha256"]


def test_authenticates_receipt_minted_by_the_real_finalizer_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, _worker, _manifest, files = finalizer_fixture._sealed_archive(
        workspace_root
    )
    summary_name = next(
        name for name in files if name.endswith("/summary/summary.json")
    )
    summary = _summary()
    summary_bytes = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    # Rebuild the real finalizer fixture with a reportable Paper-I trajectory,
    # preserving the exact fixed job/activation/receipt producer contract.
    with tarfile.open(archive_path, "r:gz") as archive:
        raw_members = {
            member.name.lstrip("./"): archive.extractfile(member).read()
            for member in archive
            if member.isfile() and archive.extractfile(member) is not None
        }
    raw_members[summary_name] = summary_bytes
    job = finalizer_fixture._job(0)
    authority = finalizer_fixture._authorization(job["execution_id"])
    manifest_name = job["expected_run_artifacts"]["execution_manifest"]["path"]
    worker_name = "worker_receipt.json"
    manifest = json.loads(raw_members[manifest_name])
    manifest["output_payloads"]["summary"] = {
        "path": summary_name,
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "size_bytes": len(summary_bytes),
    }
    manifest = finalizer_fixture._digested(manifest)
    raw_members[manifest_name] = finalizer_fixture._json_bytes(manifest)
    worker = json.loads(raw_members[worker_name])
    worker["execution_manifest_sha256"] = manifest["sha256"]
    worker["authorization_sha256"] = authority["sha256"]
    worker["artifacts"] = [
        {
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in sorted(raw_members.items())
        if name not in {"worker_exit_status.txt", worker_name}
    ]
    worker = finalizer_fixture._digested(worker)
    raw_members[worker_name] = finalizer_fixture._json_bytes(worker)
    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, payload in raw_members.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    finalizer_fixture._remote_identity(workspace_root, archive_path)
    completed = finalizer_fixture._run_helper(workspace_root, "--finalize")
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    receipt_path = workspace_root / result["receipt_path"]

    monkeypatch.setattr(watcher, "RETRIEVED_DIR", archive_path.parent)
    fixture_identity_dir = workspace_root / finalizer_fixture.EVIDENCE_DIR_RELATIVE
    monkeypatch.setattr(watcher, "IDENTITY_DIR", fixture_identity_dir)
    original_resolver = watcher._resolve_repo_or_absolute

    def resolve_fixture_archive(raw: object, *, label: str) -> Path:
        candidate = Path(str(raw))
        if not candidate.is_absolute() and (
            candidate.as_posix().startswith(
                finalizer_fixture.ARCHIVE_DIR_RELATIVE.as_posix()
            )
            or candidate.as_posix().startswith(
                finalizer_fixture.EVIDENCE_DIR_RELATIVE.as_posix()
            )
        ):
            return workspace_root / candidate
        return original_resolver(raw, label=label)

    monkeypatch.setattr(watcher, "_resolve_repo_or_absolute", resolve_fixture_archive)

    authenticated = watcher.authenticate_receipt(receipt_path)

    assert authenticated["run_id"] == job["execution_id"]
    assert authenticated["terminal"] == {"k": 50, "error": pytest.approx(1 / 51)}
    assert len(authenticated["points"]) == 50
    assert authenticated["exact_same_cutoff_energy"] == -1.0
    assert authenticated["full_source_horizon"] == 50
    assert authenticated["plotted_horizon"] == 50
    assert authenticated["full_source_point_count"] == 50
    assert authenticated["plotted_point_count"] == 50
    assert authenticated["display_crop"] == "common_comparator_horizon_k_le_50"


def test_adapter_keeps_all_six_regimes_and_pending_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = {
        "schema": "paper_i_phase0_route_progress_adapter_v1",
        "status": "completed_six_regime_evidence_ready",
        "cells": [
            {
                "regime_id": regime,
                "regime_label": watcher.REGIME_LABELS[regime],
                "nph": 3 if regime in watcher.REGIME_ORDER[:3] else 7,
                "exact_same_cutoff_energy": -1.0,
                "phase0_route": {
                    "status": "completed_authenticated_remote_summary",
                    "points": [
                        {"k": k, "error": 1.0 / (k + 1), "energy": -1 + 1 / (k + 1)}
                        for k in range(1, 51)
                    ],
                    "latest": {"k": 50, "error": 1 / 51, "energy": -1 + 1 / 51},
                    "source": {"history": f"plateau-{regime}.json"},
                },
                "append_adapt": {
                    "execution_id": f"current-adapt-{regime}",
                    "exact_same_cutoff_energy": -1.0,
                    "points": [
                        {"k": k, "error": 2.0 / (k + 1)}
                        for k in range(0, 71)
                    ],
                    "marker": {
                        "k": 50,
                        "error": 2.0 / 51,
                        "policy": "terminal_common_horizon",
                    },
                    "source": {
                        "history": f"append-{regime}.json",
                        "limitations": (
                            ["no independent remote/local retrieval receipt"]
                            if regime == "weak_weak"
                            else []
                        ),
                    },
                },
            }
            for regime in watcher.REGIME_ORDER
        ],
        "sha256": "a" * 64,
    }
    monkeypatch.setattr(watcher, "_load_reference_adapter", lambda: reference)
    complete = {
        "run_id": "fixture-run",
        "regime_id": "weak_weak",
        "comparator_policy": watcher.EXPECTED_POLICIES[0],
        "proc_id": 0,
        "controller_rounds_completed": 50,
        "exact_same_cutoff_energy": -1.0,
        "points": [{"k": k, "error": 0.5 / (k + 1)} for k in range(1, 51)],
        "terminal": {"k": 50, "error": 0.5 / 51},
        "receipt_sha256": "b" * 64,
        "source": {},
    }

    adapter = watcher.build_adapter([complete])

    assert [row["regime_id"] for row in adapter["reference_cells"]] == list(
        watcher.REGIME_ORDER
    )
    assert all(
        row["current_adapt"]["execution_id"] == f"current-adapt-{row['regime_id']}"
        and [point["k"] for point in row["current_adapt"]["points"]]
        == list(range(0, 51))
        and row["current_adapt"]["marker"]["k"] == 50
        and row["current_adapt"]["full_source_horizon"] == 70
        and row["current_adapt"]["plotted_horizon"] == 50
        and len(row["current_adapt"]["points"]) == 51
        and row["current_adapt"]["source"]["history"]
        == f"append-{row['regime_id']}.json"
        and row["source"]["history"] == f"plateau-{row['regime_id']}.json"
        for row in adapter["reference_cells"]
    )
    assert adapter["campaign_counts"] == {
        "planned_comparator_cells": 12,
        "authenticated_comparator_cells": 1,
        "always_insertion_authenticated": 1,
        "append_always_authenticated": 0,
        "pending_comparator_cells": 11,
    }
    matrix = {row["regime_id"]: row for row in adapter["matrix"]}
    assert matrix["weak_weak"]["always_insertion"].startswith("complete / authenticated")
    assert matrix["weak_weak"]["append_always"] == "pending / awaiting authenticated receipt"
    assert all(
        matrix[regime]["always_insertion"] == "pending / awaiting authenticated receipt"
        and matrix[regime]["append_always"] == "pending / awaiting authenticated receipt"
        for regime in watcher.REGIME_ORDER[1:]
    )
    assert adapter["parameter_manifest"]["reference_curves"] == [
        "current Append-ADAPT baseline",
        "current plateau-insertion RA-ADAPT",
    ]
    assert adapter["reference_cells"][0]["current_adapt"]["source_limitations"] == [
        "no independent remote/local retrieval receipt"
    ]
    assert any(
        limitation.startswith("Weak--weak current Append-ADAPT source:")
        for limitation in adapter["limitations"]
    )


def test_page18_renders_current_adapt_and_plateau_as_distinct_reference_curves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference = {
        "schema": "paper_i_phase0_route_progress_adapter_v1",
        "status": "completed_six_regime_evidence_ready",
        "cells": [
            {
                "regime_id": regime,
                "regime_label": watcher.REGIME_LABELS[regime],
                "nph": 3 if regime in watcher.REGIME_ORDER[:3] else 7,
                "exact_same_cutoff_energy": -1.0,
                "phase0_route": {
                    "status": "completed_authenticated_remote_summary",
                    "points": [
                        {
                            "k": k,
                            "energy": -1.0 + 1.0 / (k + 1),
                            "error": 1.0 / (k + 1),
                        }
                        for k in range(1, 51)
                    ],
                    "source": {"history": f"plateau-{regime}.json"},
                },
                "append_adapt": {
                    "execution_id": f"current-adapt-{regime}",
                    "exact_same_cutoff_energy": -1.0,
                    "points": [
                        {"k": k, "error": 2.0 / (k + 1)}
                        for k in range(0, 71)
                    ],
                    "marker": {
                        "k": 50,
                        "error": 2.0 / 51,
                        "policy": "terminal_common_horizon",
                    },
                    "source": {"history": f"append-{regime}.json"},
                },
            }
            for regime in watcher.REGIME_ORDER
        ],
        "sha256": "a" * 64,
    }
    monkeypatch.setattr(watcher, "_load_reference_adapter", lambda: reference)
    monkeypatch.setattr(watcher, "ADAPTER_PATH", tmp_path / "adapter.json")
    monkeypatch.setattr(watcher, "PAGE18_PNG", tmp_path / "page18.png")
    monkeypatch.setattr(watcher, "PAGE18_PDF", tmp_path / "page18.pdf")
    complete = {
        "run_id": "fixture-run",
        "regime_id": "weak_weak",
        "comparator_policy": watcher.EXPECTED_POLICIES[0],
        "proc_id": 0,
        "controller_rounds_completed": 50,
        "exact_same_cutoff_energy": -1.0,
        "points": [{"k": k, "error": 0.5 / (k + 1)} for k in range(1, 51)],
        "marker": {"k": 30, "error": 0.5 / 31},
        "terminal": {"k": 50, "error": 0.5 / 51},
        "receipt_sha256": "b" * 64,
        "source": {},
    }
    captured: dict[str, object] = {}
    from pipelines.reporting import append_paper_i_completed_beam_noise_pages

    monkeypatch.setattr(
        append_paper_i_completed_beam_noise_pages,
        "_save_page",
        lambda figure, **_kwargs: captured.setdefault("figure", figure),
    )

    watcher.render_page(watcher.build_adapter([complete]))

    figure = captured["figure"]
    plot_axes = figure.axes[:6]
    assert len(plot_axes[0].lines) == 3
    assert all(len(axis.lines) == 2 for axis in plot_axes[1:])
    assert [point.get_xdata().tolist() for point in plot_axes[0].lines[:2]] == [
        list(range(0, 51)),
        list(range(1, 51)),
    ]
    assert all(tuple(axis.get_xlim()) == pytest.approx((0.0, 50.0)) for axis in plot_axes)
    assert all(
        all(float(tick).is_integer() for tick in axis.get_xticks())
        for axis in plot_axes
    )
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "current ADAPT (Append-ADAPT baseline)",
        "current plateau-insertion RA-ADAPT",
        "RA-ADAPT insertion always",
        "RA-ADAPT append-only insertion (append always)",
    ]


def test_adapter_rejects_cross_curve_same_cutoff_reference_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = {
        "schema": "paper_i_phase0_route_progress_adapter_v1",
        "status": "completed_six_regime_evidence_ready",
        "cells": [
            {
                "regime_id": regime,
                "regime_label": watcher.REGIME_LABELS[regime],
                "nph": 3 if regime in watcher.REGIME_ORDER[:3] else 7,
                "exact_same_cutoff_energy": -1.0,
                "phase0_route": {
                    "status": "completed_authenticated_remote_summary",
                    "points": [
                        {"k": k, "energy": -1 + 1 / (k + 1), "error": 1 / (k + 1)}
                        for k in range(1, 51)
                    ],
                    "source": {},
                },
                "append_adapt": {
                    "execution_id": f"current-adapt-{regime}",
                    "exact_same_cutoff_energy": (-0.9 if regime == "weak_weak" else -1.0),
                    "points": [{"k": k, "error": 2 / (k + 1)} for k in range(71)],
                    "marker": {
                        "k": 50,
                        "error": 2 / 51,
                        "policy": "terminal_common_horizon",
                    },
                    "source": {},
                },
            }
            for regime in watcher.REGIME_ORDER
        ],
        "sha256": "a" * 64,
    }
    monkeypatch.setattr(watcher, "_load_reference_adapter", lambda: reference)
    complete = {
        "run_id": "fixture-run",
        "regime_id": "weak_weak",
        "comparator_policy": watcher.EXPECTED_POLICIES[0],
        "proc_id": 0,
        "controller_rounds_completed": 50,
        "exact_same_cutoff_energy": -1.0,
        "points": [{"k": k, "error": 0.5 / (k + 1)} for k in range(1, 51)],
        "terminal": {"k": 50, "error": 0.5 / 51},
        "receipt_sha256": "b" * 64,
        "source": {},
    }

    with pytest.raises(watcher.WatchError, match="same-cutoff reference drifted"):
        watcher.build_adapter([complete])


def test_page18_append_and_replacement_preserve_first_seventeen_content_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "report.pdf"
    provenance_path = tmp_path / "provenance.json"
    page18 = tmp_path / "page18.pdf"
    adapter_path = tmp_path / "adapter.json"
    page18_png = tmp_path / "page18.png"
    page18_png.write_bytes(b"png")
    adapter_path.write_text("{}\n", encoding="utf-8")
    original_payloads = [f"page-{index}".encode() for index in range(1, 18)]
    _write_pdf(target, original_payloads)
    _write_pdf(page18, [b"page18-v1"])
    original_hashes = _content_hashes(target)
    provenance = {
        "layout": {
            **{f"page_{index}": f"page-{index}" for index in range(1, 17)},
            "page_17": watcher.PAGE17_ID,
            "page_count": 17,
        },
        "outputs": {"partial_progress_pdf": watcher.binding(target)},
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    for name, value in (
        ("TARGET_PDF", target),
        ("TARGET_PROVENANCE", provenance_path),
        ("PAGE18_PDF", page18),
        ("PAGE18_PNG", page18_png),
        ("ADAPTER_PATH", adapter_path),
    ):
        monkeypatch.setattr(watcher, name, value)
    adapter = {
        "schema": watcher.ADAPTER_SCHEMA,
        "status": "provisional_page12_1_of_12_authenticated",
        "sha256": "c" * 64,
        "campaign_counts": {},
        "matrix": [],
        "completed_comparators": {},
        "sources": {},
        "limitations": [],
    }

    first = watcher.append_or_replace_page(adapter)
    assert first["page_count"] == 18
    assert _content_hashes(target)[:17] == original_hashes
    first_page18_hash = _content_hashes(target)[17]

    _write_pdf(page18, [b"page18-v2"])
    second = watcher.append_or_replace_page(adapter)
    assert second["page_count"] == 18
    assert _content_hashes(target)[:17] == original_hashes
    assert _content_hashes(target)[17] != first_page18_hash
    updated = json.loads(provenance_path.read_text())
    assert updated["layout"]["page_17"] == watcher.PAGE17_ID
    assert updated["layout"]["page_18"] == watcher.PAGE18_ID


def test_page18_replacement_preserves_existing_page19(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pipelines.reporting import append_paper_i_l3_weak_holstein_page19

    target = tmp_path / "report.pdf"
    provenance_path = tmp_path / "provenance.json"
    page18 = tmp_path / "page18.pdf"
    adapter_path = tmp_path / "adapter.json"
    page18_png = tmp_path / "page18.png"
    page18_png.write_bytes(b"png")
    adapter_path.write_text("{}\n", encoding="utf-8")
    _write_pdf(target, [f"page-{index}".encode() for index in range(1, 20)])
    _write_pdf(page18, [b"page18-updated"])
    before = _content_hashes(target)
    page19_report = {"preserve": True}
    provenance = {
        "layout": {
            "page_count": 19,
            "page_17": watcher.PAGE17_ID,
            "page_18": watcher.PAGE18_ID,
            "page_19": append_paper_i_l3_weak_holstein_page19.PAGE_ID,
        },
        "outputs": {"partial_progress_pdf": watcher.binding(target)},
        "l3_weak_holstein_append_page19": page19_report,
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    for name, value in (
        ("TARGET_PDF", target),
        ("TARGET_PROVENANCE", provenance_path),
        ("PAGE18_PDF", page18),
        ("PAGE18_PNG", page18_png),
        ("ADAPTER_PATH", adapter_path),
    ):
        monkeypatch.setattr(watcher, name, value)
    adapter = {
        "schema": watcher.ADAPTER_SCHEMA,
        "status": "provisional_page12_7_of_12_authenticated",
        "sha256": "c" * 64,
        "campaign_counts": {},
        "matrix": [],
        "completed_comparators": {},
        "sources": {},
        "limitations": [],
    }

    result = watcher.append_or_replace_page(adapter)

    after = _content_hashes(target)
    updated = json.loads(provenance_path.read_text())
    assert result["page_count"] == 19
    assert after[:17] == before[:17]
    assert after[17] != before[17]
    assert after[18] == before[18]
    assert updated["layout"]["page_19"] == append_paper_i_l3_weak_holstein_page19.PAGE_ID
    assert updated["l3_weak_holstein_append_page19"] == page19_report


def test_macro_page17_refresh_preserves_current_singleton_page18(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "report.pdf"
    provenance_path = tmp_path / "provenance.json"
    page17 = tmp_path / "page17.pdf"
    page17_png = tmp_path / "page17.png"
    adapter_path = tmp_path / "macro-adapter.json"
    page17_png.write_bytes(b"page17 png")
    adapter_path.write_text("{}\n", encoding="utf-8")
    _write_pdf(target, [f"page-{index}".encode() for index in range(1, 19)])
    _write_pdf(page17, [b"new-page17"])
    before = _content_hashes(target)
    page12_report = {"receipt_evidence_revision": "a" * 64}
    provenance = {
        "layout": {
            "page_count": 18,
            "page_16": macro_snapshot.PAGE16_ID,
            "page_17": macro_snapshot.PAGE17_ID,
            "page_18": watcher.PAGE18_ID,
        },
        "outputs": {
            "partial_progress_pdf": macro_snapshot.binding(target),
            "page12_insertion_comparator_snapshot_page18_pdf": {"preserve": True},
        },
        "phase0_page12_insertion_comparator_snapshot": page12_report,
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    for name, value in (
        ("TARGET_PDF", target),
        ("TARGET_PROVENANCE", provenance_path),
        ("PAGE17_PDF", page17),
        ("PAGE17_PNG", page17_png),
        ("ADAPTER_PATH", adapter_path),
    ):
        monkeypatch.setattr(macro_snapshot, name, value)
    adapter = {
        "status": "fixture",
        "sha256": "b" * 64,
        "campaign_counts": {
            "authenticated_curves_plotted": 1,
            "local_cells_closed_authenticated": 0,
            "local_cells_completed_at_k30": 0,
            "local_cells_right_censored_at_k30": 0,
        },
        "campaign_execution_state": {},
        "matrix": [],
        "completed_comparators": {},
        "sources": {},
        "limitations": [],
    }

    result = macro_snapshot.append_or_replace_pages(adapter, provenance)

    after = _content_hashes(target)
    updated = json.loads(provenance_path.read_text())
    assert result["page_count"] == 18
    assert result["preserved_page_count"] == 17
    assert after[:16] == before[:16]
    assert after[16] != before[16]
    assert after[17] == before[17]
    assert updated["layout"]["page_18"] == watcher.PAGE18_ID
    assert updated["phase0_page12_insertion_comparator_snapshot"] == page12_report
    assert updated["outputs"][
        "page12_insertion_comparator_snapshot_page18_pdf"
    ] == {"preserve": True}


def test_macro_watcher_accepts_page17_refresh_with_or_without_page18() -> None:
    from pipelines.reporting import (
        watch_paper_i_page16_insertion_comparator_snapshot as macro_watcher,
    )

    assert macro_watcher._valid_update_page_shape(17, 16) is True
    assert macro_watcher._valid_update_page_shape(18, 17) is True
    assert macro_watcher._valid_update_page_shape(18, 16) is False
    assert macro_watcher._valid_update_page_shape(19, 18) is False
