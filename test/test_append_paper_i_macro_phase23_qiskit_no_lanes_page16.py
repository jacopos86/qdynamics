from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.reporting import (
    append_paper_i_macro_phase23_qiskit_no_lanes_page16 as page16,
)


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
        payload = b"" if contents is None else contents.get_data()
        result.append(hashlib.sha256(payload).hexdigest())
    return result


def _adapter(path: Path, *, revision: str) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema": "paper_i_macro_phase0_phase23_qiskit_no_lanes_page16_adapter_v1",
        "page_id": page16.PAGE_ID,
        "status": "partial_1_of_6_completed",
        "cells": [{"page16_qiskit_route": {"revision": revision}}],
        "limitations": ["fixture"],
    }
    value = {**unsigned, "sha256": page16._canonical_sha256(unsigned)}
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path, Path]:
    target_pdf = tmp_path / "report.pdf"
    target_provenance = tmp_path / "report-provenance.json"
    page_pdf = tmp_path / "page16.pdf"
    page_png = tmp_path / "page16.png"
    adapter = tmp_path / "page16-adapter.json"
    for name, value in (
        ("TARGET_PDF", target_pdf),
        ("TARGET_PROVENANCE", target_provenance),
        ("PAGE_PDF", page_pdf),
        ("PAGE_PNG", page_png),
        ("ADAPTER_PATH", adapter),
    ):
        monkeypatch.setattr(page16, name, value)
    return target_pdf, target_provenance, page_pdf, page_png, adapter


def _provenance(target_pdf: Path) -> dict[str, object]:
    return {
        "schema": "fixture",
        "layout": {
            "page_count": 15,
            "page_13": "macro_gradient_phase0_macro_phase123_proxy_no_lanes_partial_v1",
            "page_14": page16.completed_pages.PAGE14_ID,
            "page_15": page16.completed_pages.PAGE15_ID,
        },
        "outputs": {"partial_progress_pdf": page16.binding(target_pdf)},
    }


def test_completed_page16_archive_closes_and_compiles_shared_cost_tuple() -> None:
    adapter = page16.build_adapter()

    assert adapter["status"] == "completed_6_of_6_mixed_horizon"
    completed = [
        cell["page16_qiskit_route"]
        for cell in adapter["cells"]
        if cell["page16_qiskit_route"] is not None
    ]
    assert len(completed) == 6
    assert [row["proc_id"] for row in completed] == [0, 1, 2, 3, 4, 5]
    assert [row["terminal"]["k"] for row in completed] == [
        50,
        50,
        50,
        30,
        30,
        30,
    ]
    assert all(
        set(row["costs"]) == {"N2q", "D2q", "Dc", "W1q", "S_alg"}
        and row["compile"]["qiskit_basis_work_status"] == "ok"
        and row["sources"]["archive_closure"][
            "all_declared_payload_hashes_verified"
        ]
        is True
        for row in completed
    )
    result = completed[0]
    assert result["terminal"]["k"] == 50
    assert result["terminal"]["error"] == pytest.approx(1.985722419926006e-4)
    assert result["costs"] == {
        "N2q": 2098,
        "D2q": 1878,
        "Dc": 9176,
        "W1q": 3974,
        "S_alg": 200621,
    }
    assert result["sources"]["archive_closure"] == {
        "worker_exit_status": 0,
        "declared_payload_count": 7,
        "all_declared_payload_hashes_verified": True,
        "unbound_file_count": 0,
        "worker_receipt_canonical_sha256": (
            "ec0514d1caa5ebb92f5269686fbe606d5e2126d3cdbccff546a43a39ef6aa64c"
        ),
        "execution_manifest_canonical_sha256": (
            "384fb110f258fbb44f203297a21a668d676d304b9e50144b45a5df018054f7c7"
        ),
        "authorization_sha256_bound_by_worker": (
            "d0dc43cf7291f8832b8833240cf4bddec309951b0167f1bc06c43d70476bc591"
        ),
    }


def test_append_and_replace_page16_preserve_first_fifteen_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pypdf = pytest.importorskip("pypdf")
    target_pdf, target_provenance, page_pdf, page_png, adapter_path = _patch_paths(
        monkeypatch, tmp_path
    )
    _write_pdf(
        target_pdf,
        [f"q {index} 0 1 1 re f Q\n".encode() for index in range(1, 16)],
    )
    _write_pdf(page_pdf, [b"q 16 0 1 1 re f Q\n"])
    page_png.write_bytes(b"page16 fixture")
    adapter = _adapter(adapter_path, revision="a")
    provenance = _provenance(target_pdf)
    target_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    preserved = _content_hashes(target_pdf)

    result = page16.append_or_replace_page(adapter, provenance)

    updated = json.loads(target_provenance.read_text())
    assert result["page_count"] == 16
    assert len(pypdf.PdfReader(str(target_pdf), strict=False).pages) == 16
    assert _content_hashes(target_pdf)[:15] == preserved
    assert updated["layout"]["page_16"] == page16.PAGE_ID
    assert updated["layout"]["page_count"] == 16
    first_page16_hash = _content_hashes(target_pdf)[15]

    _write_pdf(page_pdf, [b"q 116 0 2 2 re f Q\n"])
    replacement = _adapter(adapter_path, revision="b")
    current = json.loads(target_provenance.read_text())
    result = page16.append_or_replace_page(replacement, current)

    assert result["page_count"] == 16
    assert len(pypdf.PdfReader(str(target_pdf), strict=False).pages) == 16
    hashes = _content_hashes(target_pdf)
    assert hashes[:15] == preserved
    assert hashes[15] != first_page16_hash
