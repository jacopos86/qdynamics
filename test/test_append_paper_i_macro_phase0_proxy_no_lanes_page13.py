from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.reporting import (
    append_paper_i_macro_phase0_proxy_no_lanes_page13 as page13,
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
    result: list[str] = []
    for page in pypdf.PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        payload = b"" if contents is None else contents.get_data()
        result.append(hashlib.sha256(payload).hexdigest())
    return result


def _fixture_adapter(path: Path, *, digest_character: str) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema": "paper_i_macro_phase0_proxy_no_lanes_page13_adapter_v1",
        "status": "partial_live_local_run",
        "cells": [],
        "limitations": ["fixture"],
        "fixture_revision": digest_character,
    }
    adapter = {
        **unsigned,
        "sha256": hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    path.write_text(
        json.dumps(adapter, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return adapter


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    target_pdf = tmp_path / "partial-progress.pdf"
    target_provenance = tmp_path / "partial-progress-provenance.json"
    page_pdf = tmp_path / "page-13.pdf"
    page_png = tmp_path / "page-13.png"
    adapter_path = tmp_path / "page-13-adapter.json"
    monkeypatch.setattr(page13, "TARGET_PDF", target_pdf)
    monkeypatch.setattr(page13, "TARGET_PROVENANCE", target_provenance)
    monkeypatch.setattr(page13, "PAGE_PDF", page_pdf)
    monkeypatch.setattr(page13, "PAGE_PNG", page_png)
    monkeypatch.setattr(page13, "ADAPTER_PATH", adapter_path)
    return target_pdf, target_provenance, page_pdf, page_png, adapter_path


def _base_provenance(target_pdf: Path) -> dict[str, object]:
    layout: dict[str, object] = {"page_count": 12}
    layout.update(
        {
            f"page_{index}": f"fixture-page-{index}"
            for index in range(1, 13)
        }
    )
    layout["page_11"] = "macro_gradient_phase0_then_singleton_partial_v1"
    layout["page_12"] = "global_singleton_gradient_phase0_partial_v1"
    return {
        "schema": "synthetic_existing_report_v1",
        "layout": layout,
        "outputs": {"partial_progress_pdf": page13.binding(target_pdf)},
    }


def test_parse_ai_log_points_filters_events_and_respects_maximum_k() -> None:
    text = "\n".join(
        (
            "ordinary output",
            "AI_LOG not-json",
            'AI_LOG {"event":"other","depth":1}',
            'AI_LOG {"event":"hardcoded_adapt_iter","depth":1,'
            '"energy":-0.9,"selected_position":0,"ts_utc":"t1"}',
            'AI_LOG {"event":"hardcoded_adapt_iter","depth":2,'
            '"energy":-0.95,"selected_position":1,"ts_utc":"t2"}',
            'AI_LOG {"event":"hardcoded_adapt_iter","depth":3,'
            '"energy":-0.99,"selected_position":2,"ts_utc":"t3"}',
        )
    )

    points = page13.parse_ai_log_points(text, exact=-1.0, maximum_k=2)

    assert [point["k"] for point in points] == [1, 2]
    assert [point["error"] for point in points] == pytest.approx([0.1, 0.05])
    assert [point["selected_position"] for point in points] == [0, 1]


def test_parse_ai_log_points_rejects_noncontiguous_accepted_rounds() -> None:
    text = (
        'AI_LOG {"event":"hardcoded_adapt_iter","depth":2,'
        '"energy":-0.9,"selected_position":0,"ts_utc":"t2"}'
    )

    with pytest.raises(page13.UpdateError, match="not contiguous"):
        page13.parse_ai_log_points(text, exact=-1.0)


def test_append_page_13_preserves_first_twelve_pages_and_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    target_pdf, target_provenance, page_pdf, page_png, adapter_path = (
        _patch_paths(monkeypatch, tmp_path)
    )
    _write_pdf(
        target_pdf,
        [f"q {index} 0 1 1 re f Q\n".encode() for index in range(1, 13)],
    )
    _write_pdf(page_pdf, [b"q 13 0 1 1 re f Q\n"])
    page_png.write_bytes(b"page-13 fixture png")
    adapter = _fixture_adapter(adapter_path, digest_character="a")
    provenance = _base_provenance(target_pdf)
    target_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_layout = dict(provenance["layout"])
    original_hashes = _content_hashes(target_pdf)

    result = page13.append_or_replace_page(adapter, provenance)

    updated = json.loads(target_provenance.read_text(encoding="utf-8"))
    reader = pypdf.PdfReader(str(target_pdf), strict=False)
    assert result["page_count"] == 13
    assert len(reader.pages) == 13
    assert _content_hashes(target_pdf)[:12] == original_hashes
    for index in range(1, 13):
        assert updated["layout"][f"page_{index}"] == original_layout[f"page_{index}"]
    assert updated["layout"]["page_13"] == page13.PAGE_ID
    assert updated["layout"]["page_count"] == 13
    assert updated["outputs"]["partial_progress_pdf"] == page13.binding(target_pdf)


def test_replace_page_13_preserves_first_twelve_pages_without_growing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    target_pdf, target_provenance, page_pdf, page_png, adapter_path = (
        _patch_paths(monkeypatch, tmp_path)
    )
    _write_pdf(
        target_pdf,
        [f"q {index} 0 1 1 re f Q\n".encode() for index in range(1, 13)],
    )
    _write_pdf(page_pdf, [b"q 13 0 1 1 re f Q\n"])
    page_png.write_bytes(b"page-13 fixture png")
    adapter = _fixture_adapter(adapter_path, digest_character="a")
    provenance = _base_provenance(target_pdf)
    target_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_layout = dict(provenance["layout"])
    original_hashes = _content_hashes(target_pdf)
    page13.append_or_replace_page(adapter, provenance)
    first_page_13_hash = _content_hashes(target_pdf)[12]

    _write_pdf(page_pdf, [b"q 113 0 2 2 re f Q\n"])
    replacement_adapter = _fixture_adapter(adapter_path, digest_character="b")
    current_provenance = json.loads(
        target_provenance.read_text(encoding="utf-8")
    )
    result = page13.append_or_replace_page(
        replacement_adapter,
        current_provenance,
    )

    updated = json.loads(target_provenance.read_text(encoding="utf-8"))
    hashes = _content_hashes(target_pdf)
    assert result["page_count"] == 13
    assert len(pypdf.PdfReader(str(target_pdf), strict=False).pages) == 13
    assert hashes[:12] == original_hashes
    assert hashes[12] != first_page_13_hash
    for index in range(1, 13):
        assert updated["layout"][f"page_{index}"] == original_layout[f"page_{index}"]
    assert updated["layout"]["page_13"] == page13.PAGE_ID
    assert updated["layout"]["page_count"] == 13
    assert updated["outputs"]["partial_progress_pdf"] == page13.binding(target_pdf)
