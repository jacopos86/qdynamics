from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from pipelines.reporting import (
    append_paper_i_macro_phase23_qiskit_beam_metric_page18 as page18,
)


def _digested(value: dict[str, object]) -> dict[str, object]:
    return {**value, "sha256": page18._canonical_sha256(value)}


def _write_json(path: Path, value: dict[str, object]) -> bytes:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    return raw


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
    result = []
    for page in pypdf.PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        raw = b"" if contents is None else contents.get_data()
        result.append(hashlib.sha256(raw).hexdigest())
    return result


def _archive_fixture(
    path: Path,
    *,
    job: dict[str, object],
    trace_rounds: int = 20,
) -> dict[str, object]:
    execution_id = str(job["execution_id"])
    expected = job["expected_run_artifacts"]
    assert isinstance(expected, dict)
    exact = -1.234
    trace = [
        {
            "controller_round": k,
            "accepted_energy": exact + 1.0 / (k + 1),
            "absolute_energy_error": 1.0 / (k + 1),
            "active_ansatz_depth": k,
            "exact_same_cutoff_energy": exact,
        }
        for k in range(1, trace_rounds + 1)
    ]
    summary = {
        "schema": "paper_i_run_summary_v1",
        "accepted_error_trace": trace,
        "provenance": {"exact_same_cutoff_energy": exact},
        "requested_rounds": [],
    }
    payloads: dict[str, bytes] = {}
    for role in ("checkpoint", "estimator_ledger", "result"):
        relative = str(expected[role]["path"])
        payloads[relative] = f"{role}-fixture".encode()
    summary_name = str(expected["summary"]["path"])
    payloads[summary_name] = json.dumps(summary, sort_keys=True).encode()
    output_payloads = {
        role: {
            "path": str(expected[role]["path"]),
            "sha256": hashlib.sha256(payloads[str(expected[role]["path"])]).hexdigest(),
            "size_bytes": len(payloads[str(expected[role]["path"])]),
        }
        for role in ("checkpoint", "estimator_ledger", "result", "summary")
    }
    execution_manifest = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_page16_macro_phase23_qiskit_beam_metric_"
                "execution_manifest_v1"
            ),
            "status": "passed",
            "package_id": page18.PACKAGE_ID,
            "campaign_id": page18.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": "a" * 64,
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": page18.ROUTE_CONTRACT_SHA256,
            "target_horizon": 20,
            "controller_rounds_completed": 20,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "output_payloads": output_payloads,
        }
    )
    manifest_name = str(expected["execution_manifest"]["path"])
    payloads[manifest_name] = json.dumps(
        execution_manifest, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    artifacts = [
        {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for relative, raw in sorted(payloads.items())
    ]
    worker = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_page16_macro_phase23_qiskit_beam_metric_"
                "worker_receipt_v1"
            ),
            "status": "passed",
            "package_id": page18.PACKAGE_ID,
            "campaign_id": page18.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": "a" * 64,
            "execution_manifest_sha256": execution_manifest["sha256"],
            "controller_rounds_completed": 20,
            "fresh_start": True,
            "artifacts": artifacts,
        }
    )
    roots = {
        "worker_exit_status.txt": b"0\n",
        "worker_receipt.json": json.dumps(
            worker, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n",
    }
    with tarfile.open(path, "w:gz") as archive:
        for relative, raw in {**roots, **payloads}.items():
            info = tarfile.TarInfo(f"./{relative}")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return {
        "execution_id": execution_id,
        "proc_id": 0,
        "filename": path.name,
        "remote_path": f"osdf:///example/transfer/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "fetch_verification": "remote_and_local_size_sha256_match_v1",
    }


def test_real_package_closes_and_absent_manifest_stays_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(page18, "RETRIEVED_DIR", tmp_path / "retrieved")
    monkeypatch.setattr(
        page18, "RETRIEVAL_MANIFEST", tmp_path / "retrieved/retrieval_manifest.json"
    )

    adapter = page18.build_adapter()

    assert adapter["status"] == "partial_0_of_6_exact_k20"
    assert adapter["completed_regime_count"] == 0
    assert adapter["pending_regime_count"] == 6
    assert [row["regime_id"] for row in adapter["cells"]] == list(
        page18.REGIME_ORDER
    )
    assert all(
        row["page18_qiskit_beam_metric"] is None
        and row["status"] == "pending_no_schema_locked_retrieval"
        for row in adapter["cells"]
    )
    assert all(
        [point["k"] for point in row["page14_proxy_beam"]["points"]]
        == list(range(1, 21))
        and [point["k"] for point in row["page16_unpruned_qiskit"]["points"]]
        == list(range(1, 21))
        for row in adapter["cells"]
    )


def test_schema_locked_retrieval_manifest_accepts_only_bound_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = page18._validate_package_authority()
    jobs = page18._package_jobs(manifest)
    execution_id, (_job_path, _job) = next(iter(jobs.items()))
    retrieved = tmp_path / "retrieved"
    retrieved.mkdir()
    filename = f"{execution_id}__{page18.CLUSTER_ID}__0.tar.gz"
    archive = retrieved / filename
    archive.write_bytes(b"archive fixture")
    row = {
        "execution_id": execution_id,
        "proc_id": 0,
        "filename": filename,
        "remote_path": f"osdf:///example/transfer/{filename}",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "size_bytes": archive.stat().st_size,
        "fetch_verification": "remote_and_local_size_sha256_match_v1",
    }
    unsigned = {
        "schema": page18.RETRIEVAL_SCHEMA,
        "status": "partial_verified_fetches",
        "cluster_id": page18.CLUSTER_ID,
        "package_id": page18.PACKAGE_ID,
        "package_manifest_canonical_sha256": (
            page18.PACKAGE_MANIFEST_CANONICAL_SHA256
        ),
        "source_archive_sha256": page18.SOURCE_ARCHIVE_SHA256,
        "archive_count": 1,
        "archives": [row],
    }
    receipt = _digested(unsigned)
    receipt_path = retrieved / "retrieval_manifest.json"
    _write_json(receipt_path, receipt)
    monkeypatch.setattr(page18, "RETRIEVED_DIR", retrieved)
    monkeypatch.setattr(page18, "RETRIEVAL_MANIFEST", receipt_path)

    assert page18._validate_retrieval_manifest(jobs) == {execution_id: row}

    archive.write_bytes(b"tampered")
    with pytest.raises(page18.UpdateError, match="retrieved archive bytes drifted"):
        page18._validate_retrieval_manifest(jobs)


def test_archive_closure_requires_exact_k20_and_common_cost_tuple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = page18._validate_package_authority()
    jobs = page18._package_jobs(manifest)
    job_path, job = next(iter(jobs.values()))
    filename = f"{job['execution_id']}__{page18.CLUSTER_ID}__0.tar.gz"
    archive = tmp_path / filename
    retrieval = _archive_fixture(archive, job=job)
    monkeypatch.setattr(
        page18.completed_pages,
        "_compile_cost_tuple",
        lambda summary, round_index: (
            {"N2q": 1, "D2q": 2, "Dc": 3, "W1q": 4, "S_alg": 5000},
            {
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "qiskit_basis_work_status": "ok",
                "round": round_index,
            },
        ),
    )

    result = page18._close_archive(
        path=archive,
        retrieval=retrieval,
        job_path=job_path,
        job=job,
    )

    assert result["terminal"]["k"] == 20
    assert result["costs"] == {
        "N2q": 1,
        "D2q": 2,
        "Dc": 3,
        "W1q": 4,
        "S_alg": 5000,
    }
    assert result["sources"]["closure"] == {
        "worker_exit_status": 0,
        "declared_artifact_count": 5,
        "all_declared_artifact_hashes_verified": True,
        "unbound_file_count": 0,
        "exact_controller_rounds": 20,
        "route_contract_sha256": page18.ROUTE_CONTRACT_SHA256,
    }

    short = tmp_path / f"short__{page18.CLUSTER_ID}__0.tar.gz"
    short_retrieval = _archive_fixture(short, job=job, trace_rounds=19)
    with pytest.raises(page18.UpdateError, match="not exactly k=20"):
        page18._close_archive(
            path=short,
            retrieval=short_retrieval,
            job_path=job_path,
            job=job,
        )


def test_watcher_guard_refuses_even_when_lock_is_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = tmp_path / "watch.lock"
    lock.write_text("", encoding="utf-8")
    monkeypatch.setattr(page18, "WATCH_LOCK", lock)
    monkeypatch.setattr(page18, "_active_watcher_pids", lambda: [12345])

    with pytest.raises(page18.UpdateError, match="watcher process is active"):
        with page18._exclusive_watcher_absence():
            pytest.fail("active watcher must prevent canonical access")


def test_canonical_append_preserves_all_seventeen_pages_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_pdf = tmp_path / "master.pdf"
    target_provenance = tmp_path / "master-provenance.json"
    page_pdf = tmp_path / "page18.pdf"
    page_png = tmp_path / "page18.png"
    adapter_path = tmp_path / "page18-adapter.json"
    standalone_path = tmp_path / "page18-provenance.json"
    lock = tmp_path / "watch.lock"
    _write_pdf(
        target_pdf,
        [f"q {index} 0 1 1 re f Q\n".encode() for index in range(1, 18)],
    )
    _write_pdf(page_pdf, [b"q 18 0 1 1 re f Q\n"])
    page_png.write_bytes(b"page18")
    adapter_path.write_text("{}\n", encoding="utf-8")
    lock.write_text("", encoding="utf-8")
    provenance = {
        "schema": "fixture",
        "sentinel": {"preserve": [1, 2, 3]},
        "layout": {
            "page_count": 17,
            "page_16": page18.PAGE16_ID,
            "page_17": page18.PAGE17_ID,
        },
        "outputs": {"partial_progress_pdf": page18.binding(target_pdf)},
    }
    _write_json(target_provenance, provenance)
    adapter = {
        "status": "completed_6_of_6_exact_k20",
        "sha256": "a" * 64,
        "cells": [],
        "limitations": [],
    }
    standalone_unsigned = {
        "schema": page18.STANDALONE_SCHEMA,
        "page_id": page18.PAGE_ID,
        "adapter": {
            **page18.binding(adapter_path),
            "canonical_sha256": adapter["sha256"],
        },
        "outputs": {"page_pdf": page18.binding(page_pdf)},
    }
    standalone = _digested(standalone_unsigned)
    _write_json(standalone_path, standalone)
    for name, value in (
        ("TARGET_PDF", target_pdf),
        ("TARGET_PROVENANCE", target_provenance),
        ("PAGE_PDF", page_pdf),
        ("PAGE_PNG", page_png),
        ("ADAPTER_PATH", adapter_path),
        ("STANDALONE_PROVENANCE", standalone_path),
        ("WATCH_LOCK", lock),
    ):
        monkeypatch.setattr(page18, name, value)
    monkeypatch.setattr(page18, "_active_watcher_pids", lambda: [])
    before_hashes = _content_hashes(target_pdf)

    result = page18.append_to_canonical(adapter, standalone)

    updated = json.loads(target_provenance.read_text(encoding="utf-8"))
    assert result["page_count"] == 18
    assert _content_hashes(target_pdf)[:17] == before_hashes
    assert updated["sentinel"] == provenance["sentinel"]
    assert updated["layout"]["page_18"] == page18.PAGE_ID
    assert updated["layout"]["page_count"] == 18

