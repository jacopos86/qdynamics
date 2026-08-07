from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from pypdf import PdfReader, PdfWriter

from pipelines.reporting import (
    add_paper_i_append_r70_singleton_progress_page as page_builder,
)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["sha256"] = _canonical_digest(result)
    return result


def _adapter() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell_index, regime_id in enumerate(page_builder.COMPLETED_REGIMES):
        points = [
            {
                "round": round_index,
                "energy": -1.0 - cell_index * 0.1 - round_index * 1e-5,
                "delta_e": 10.0 ** (-1.0 - round_index / 14.0 - cell_index / 4.0),
            }
            for round_index in range(71)
        ]
        endpoints = {}
        for round_index in (50, 70):
            point = points[round_index]
            endpoints[f"round_{round_index}"] = {
                "round": round_index,
                "energy": point["energy"],
                "delta_e": point["delta_e"],
                "checkpoint_sha256": hashlib.sha256(
                    f"{regime_id}:{round_index}".encode()
                ).hexdigest(),
                "costs": {
                    "N2q": 100 + round_index + cell_index,
                    "D2q": 80 + round_index + cell_index,
                    "Dc": 300 + round_index + cell_index,
                    "W1q": 200 + round_index + cell_index,
                    "S_alg": 100_000 + round_index + cell_index,
                },
                "compile": {"schema": "test_compile_v1", "status": "passed"},
            }
        cells.append(
            {
                "regime_id": regime_id,
                "display_name": page_builder.REGIME_LABELS[regime_id],
                "nph": 3 if cell_index < 3 else 7,
                "execution_id": f"test__{regime_id}__append_singleton",
                "source": {"archive_sha256": "a" * 64},
                "points": points,
                "endpoints": endpoints,
            }
        )
    return _digested(
        {
            "schema": page_builder.ADAPTER_SCHEMA,
            "status": "passed",
            "classification": "diagnostic",
            "package_id": "test_append_r70_package",
            "cluster_id": 12345,
            "regime_order": list(page_builder.REGIME_ORDER),
            "completed_regimes": list(page_builder.COMPLETED_REGIMES),
            "pending_regimes": list(page_builder.PENDING_REGIMES),
            "source_authentication_summary": {
                "all_retained_members_bound_to_embedded_worker_inventories": True,
                "all_full_archives_remote_local_identity_authenticated": False,
                "paper_evidence_adopted": False,
            },
            "limitations": [
                "Intermediate--weak giant-member transport was not locally "
                "reauthenticated."
            ],
            "same_cutoff_reference": {
                "path": "test/reference.json",
                "sha256": "b" * 64,
            },
            "cost_policy": {
                "round_50": {
                    "classification": "canonical_paper_comparable",
                    "compiler": "common_qiskit",
                },
                "round_70": {
                    "classification": "diagnostic_extension",
                    "compiler": "common_qiskit",
                },
            },
            "cells": cells,
        }
    )


def _write_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for index in range(pages):
        writer.add_blank_page(width=500 + index, height=700 + index)
    with path.open("wb") as stream:
        writer.write(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_report(tmp_path: Path) -> tuple[Path, Path]:
    pdf = tmp_path / "base.pdf"
    provenance = tmp_path / "base_provenance.json"
    _write_pdf(pdf, 5)
    provenance.write_text(
        json.dumps(
            {
                "schema": "test_evolving_report_v1",
                "layout": {
                    "page_count": 5,
                    "page_1": "one",
                    "page_2": "two",
                    "page_3": "three",
                    "page_4": "four",
                    "page_5": "five",
                },
                "limitations": ["pre-existing limitation"],
                "outputs": {
                    "partial_progress_pdf": {
                        "path": str(pdf),
                        "sha256": _sha256(pdf),
                    },
                    "pre_existing_asset": {
                        "path": "untouched",
                        "sha256": "c" * 64,
                    },
                },
                "unrelated": {"must": ["remain", "unchanged"]},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return pdf, provenance


def _fake_assets(
    _adapter: Mapping[str, Any], *, asset_dir: Path, asset_stem: str
) -> dict[str, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plot_png": asset_dir / f"{asset_stem}_plot.png",
        "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
        "page_tex": asset_dir / f"{asset_stem}.tex",
        "page_pdf": asset_dir / f"{asset_stem}.pdf",
    }
    paths["plot_png"].write_bytes(b"synthetic png")
    _write_pdf(paths["plot_pdf"], 1)
    paths["page_tex"].write_text("synthetic tex\n", encoding="utf-8")
    _write_pdf(paths["page_pdf"], 1)
    return paths


def test_append_is_additive_preserves_five_pages_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_pdf, base_provenance = _base_report(tmp_path)
    adapter_path = tmp_path / "adapter.json"
    adapter = _adapter()
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    output_pdf = tmp_path / "combined.pdf"
    output_provenance = tmp_path / "combined_provenance.json"
    asset_dir = tmp_path / "assets"
    before_hashes = page_builder._page_content_hashes(base_pdf)
    base_payload = json.loads(base_provenance.read_text(encoding="utf-8"))
    monkeypatch.setattr(page_builder, "_build_page_assets", _fake_assets)

    result = page_builder.append_r70_singleton_progress_page(
        base_pdf=base_pdf,
        base_provenance=base_provenance,
        output_pdf=output_pdf,
        output_provenance=output_provenance,
        adapter_path=adapter_path,
        asset_dir=asset_dir,
        asset_stem="r70_page6",
    )

    assert result["status"] == "appended"
    assert len(PdfReader(str(output_pdf)).pages) == 6
    assert page_builder._page_content_hashes(output_pdf)[:5] == before_hashes
    updated = json.loads(output_provenance.read_text(encoding="utf-8"))
    assert updated["layout"] == {
        **base_payload["layout"],
        "page_count": 6,
        "page_6": page_builder.PAGE_ID,
    }
    assert updated["unrelated"] == base_payload["unrelated"]
    assert (
        updated["outputs"]["pre_existing_asset"]
        == base_payload["outputs"]["pre_existing_asset"]
    )
    progress = updated["append_singleton_r70_progress"]
    assert progress["adapter"]["canonical_sha256"] == adapter["sha256"]
    assert progress["completed_regimes"] == list(page_builder.COMPLETED_REGIMES)
    assert progress["pending_regimes"] == list(page_builder.PENDING_REGIMES)
    assert progress["source_authentication_summary"] == (
        adapter["source_authentication_summary"]
    )
    assert progress["limitations"] == adapter["limitations"]
    assert progress["structural_validation"][
        "preserved_page_content_sha256"
    ] == before_hashes
    assert updated["limitations"] == [
        "pre-existing limitation",
        page_builder.PAGE_LIMITATION,
    ]

    pdf_bytes = output_pdf.read_bytes()
    provenance_bytes = output_provenance.read_bytes()
    second = page_builder.append_r70_singleton_progress_page(
        base_pdf=base_pdf,
        base_provenance=base_provenance,
        output_pdf=output_pdf,
        output_provenance=output_provenance,
        adapter_path=adapter_path,
        asset_dir=asset_dir,
        asset_stem="r70_page6",
    )
    assert second["status"] == "already_present"
    assert output_pdf.read_bytes() == pdf_bytes
    assert output_provenance.read_bytes() == provenance_bytes


def test_existing_page_for_different_adapter_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_pdf, base_provenance = _base_report(tmp_path)
    first_adapter = _adapter()
    first_path = tmp_path / "adapter_a.json"
    first_path.write_text(json.dumps(first_adapter), encoding="utf-8")
    output_pdf = tmp_path / "combined.pdf"
    output_provenance = tmp_path / "combined_provenance.json"
    monkeypatch.setattr(page_builder, "_build_page_assets", _fake_assets)
    page_builder.append_r70_singleton_progress_page(
        base_pdf=base_pdf,
        base_provenance=base_provenance,
        output_pdf=output_pdf,
        output_provenance=output_provenance,
        adapter_path=first_path,
        asset_dir=tmp_path / "assets",
        asset_stem="r70_page6",
    )

    changed = copy.deepcopy(first_adapter)
    changed.pop("sha256")
    changed["cluster_id"] = 99999
    changed = _digested(changed)
    changed_path = tmp_path / "adapter_b.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(page_builder.R70PageError, match="different R70 adapter"):
        page_builder.append_r70_singleton_progress_page(
            base_pdf=base_pdf,
            base_provenance=base_provenance,
            output_pdf=output_pdf,
            output_provenance=output_provenance,
            adapter_path=changed_path,
            asset_dir=tmp_path / "assets",
            asset_stem="r70_page6",
        )


def test_idempotency_rejects_structural_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_pdf, base_provenance = _base_report(tmp_path)
    adapter = _adapter()
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    output_pdf = tmp_path / "combined.pdf"
    output_provenance = tmp_path / "combined_provenance.json"
    monkeypatch.setattr(page_builder, "_build_page_assets", _fake_assets)
    page_builder.append_r70_singleton_progress_page(
        base_pdf=base_pdf,
        base_provenance=base_provenance,
        output_pdf=output_pdf,
        output_provenance=output_provenance,
        adapter_path=adapter_path,
        asset_dir=tmp_path / "assets",
        asset_stem="r70_page6",
    )
    provenance = json.loads(output_provenance.read_text(encoding="utf-8"))
    provenance["append_singleton_r70_progress"]["structural_validation"][
        "new_page_content_sha256"
    ] = "0" * 64
    output_provenance.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(page_builder.R70PageError, match="structural validation"):
        page_builder.append_r70_singleton_progress_page(
            base_pdf=base_pdf,
            base_provenance=base_provenance,
            output_pdf=output_pdf,
            output_provenance=output_provenance,
            adapter_path=adapter_path,
            asset_dir=tmp_path / "assets",
            asset_stem="r70_page6",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.__setitem__("sha256", "0" * 64), "self-digest"),
        (
            lambda payload: payload["regime_order"].reverse(),
            "regime_order drifted",
        ),
        (
            lambda payload: payload["cost_policy"]["round_70"].__setitem__(
                "classification", "canonical_paper_comparable"
            ),
            "round-70 cost classification drifted",
        ),
    ),
)
def test_adapter_validation_fails_closed(
    mutation: Any, message: str
) -> None:
    adapter = _adapter()
    mutation(adapter)
    if message != "self-digest":
        adapter.pop("sha256")
        adapter = _digested(adapter)
    with pytest.raises(page_builder.R70PageError, match=message):
        page_builder._validate_adapter(adapter)


def test_plot_and_tex_show_completed_pending_and_endpoint_semantics(
    tmp_path: Path,
) -> None:
    adapter = page_builder._validate_adapter(_adapter())
    plot_png = tmp_path / "plot.png"
    plot_pdf = tmp_path / "plot.pdf"
    tex_path = tmp_path / "page.tex"
    page_builder._render_plot(adapter, png_path=plot_png, pdf_path=plot_pdf)
    page_builder._write_page_tex(
        adapter, plot_pdf=plot_pdf, tex_path=tex_path
    )

    assert plot_png.stat().st_size > 10_000
    assert plot_pdf.stat().st_size > 1_000
    tex = tex_path.read_text(encoding="utf-8")
    assert "Fresh Append-ADAPT singleton extension to 70 rounds" not in tex
    assert "canonical" in tex
    assert "diagnostic" in tex
    assert tex.count("pending") == 2
    for label in page_builder.REGIME_LABELS.values():
        assert label in tex
