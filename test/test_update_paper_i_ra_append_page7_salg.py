from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipelines.reporting import update_paper_i_ra_append_page7_salg as page7


def _r70_adapter(path: Path) -> dict[str, object]:
    cells = []
    for index, regime in enumerate(page7.REGIME_ORDER):
        exact = -1.0 - index
        points = [
            {
                "round": round_index,
                "energy": exact + 1.0 / (round_index + 2),
                "delta_e": 1.0 / (round_index + 2),
            }
            for round_index in range(71)
        ]
        cells.append(
            {
                "regime_id": regime,
                "execution_id": f"append-{regime}",
                "exact_same_cutoff_energy": exact,
                "points": points,
                "endpoints": {
                    "round_50": {
                        **copy.deepcopy(points[50]),
                        "costs": {
                            "N2q": 100 + index,
                            "D2q": 200 + index,
                            "Dc": 300 + index,
                            "W1q": 400 + index,
                            "S_alg": 500_000 + index,
                        },
                        "compile": {"compile_convention": "fixture"},
                    },
                    "round_70": {
                        **copy.deepcopy(points[70]),
                        "costs": {
                            "N2q": 150 + index,
                            "D2q": 250 + index,
                            "Dc": 350 + index,
                            "W1q": 450 + index,
                            "S_alg": 700_000 + index,
                        },
                    },
                },
                "source": {"archive": {"path": f"{regime}.tar.gz"}},
            }
        )
    value: dict[str, object] = {
        "schema": page7.APPEND_R70_SCHEMA,
        "status": "passed",
        "package_id": page7.APPEND_R70_PACKAGE_ID,
        "regime_order": list(page7.REGIME_ORDER),
        "completed_regimes": list(page7.REGIME_ORDER),
        "pending_regimes": [],
        "cells": cells,
    }
    value["sha256"] = page7.canonical_digest(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_merge_append_r70_keeps_round_50_cost_anchor(tmp_path: Path) -> None:
    path = tmp_path / "append-r70.json"
    r70 = _r70_adapter(path)
    base = {
        "cells": [
            {
                "regime_id": regime,
                "append": {
                    "execution_id": f"append-{regime}",
                    "exact_same_cutoff_energy": -1.0 - index,
                    "points": [],
                    "terminal": {},
                    "source": {},
                },
            }
            for index, regime in enumerate(page7.REGIME_ORDER)
        ]
    }

    merged = page7.merge_append_r70(
        base, r70, append_r70_path=path
    )

    for cell in merged["cells"]:
        append = cell["append"]
        assert [point["round"] for point in append["points"]] == list(
            range(71)
        )
        assert append["terminal"]["round"] == 50
        assert append["trajectory_terminal"]["round"] == 70
    assert merged["paper_facing_cost_round"] == 50
    assert merged["append_trajectory_round"] == 70


def test_curve_accepts_diagnostic_tail_after_fixed_s_alg_anchor() -> None:
    method = {
        "points": [
            {"round": k, "delta_e": 1.0 / (k + 2)} for k in range(71)
        ]
    }
    curve = page7.curve_from_points(
        method,
        [10 * k for k in range(1, 71)],
        expected_terminal_s_alg=500,
        expected_s_alg_round=50,
        label="fixture Append",
    )
    assert curve[50]["S_alg"] == 500
    assert curve[-1]["round"] == 70
    assert curve[-1]["S_alg"] == 700

    with pytest.raises(page7.PageUpdateError, match="round-50 endpoint"):
        page7.curve_from_points(
            method,
            [10 * k for k in range(1, 71)],
            expected_terminal_s_alg=501,
            expected_s_alg_round=50,
            label="fixture Append",
        )


def _write_distinct_pdf(path: Path, page_count: int) -> None:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = pypdf.PdfWriter()
    for index in range(page_count):
        page = writer.add_blank_page(width=792, height=612)
        contents = DecodedStreamObject()
        contents.set_data(
            f"BT /F1 10 Tf 10 10 Td (fixture page {index + 1}) Tj ET".encode()
        )
        page[NameObject("/Contents")] = writer._add_object(contents)
    with path.open("wb") as stream:
        writer.write(stream)


def _replacement_curves() -> dict[str, object]:
    cells = []
    for regime in page7.REGIME_ORDER:
        append_points = [
            {"round": k, "S_alg": 10 * k, "delta_e": 1.0 / (k + 2)}
            for k in range(71)
        ]
        ra_points = [
            {"round": k, "S_alg": 20 * k, "delta_e": 1.0 / (k + 2)}
            for k in range(51)
        ]
        cells.append(
            {
                "regime_id": regime,
                "append": {
                    "points": append_points,
                    "marker": copy.deepcopy(append_points[50]),
                },
                "ra": {
                    "points": ra_points,
                    "marker": copy.deepcopy(ra_points[-1]),
                    "marker_policy": "last_authenticated_plotted_prefix",
                    "unplotted_stdout_tail": None,
                },
            }
        )
    return {"cells": cells}


def test_replace_page_preserves_all_other_pages_in_twelve_page_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pypdf = pytest.importorskip("pypdf")
    master = tmp_path / "report.pdf"
    replacement = tmp_path / "page7.pdf"
    provenance_path = tmp_path / "report_provenance.json"
    curve_json = tmp_path / "curves.json"
    page_png = tmp_path / "page7.png"
    _write_distinct_pdf(master, 12)
    _write_distinct_pdf(replacement, 1)
    curve_json.write_text("{}", encoding="utf-8")
    page_png.write_bytes(b"fixture png")
    master_binding = page7.binding(master)
    provenance = {
        "layout": {
            "page_count": 12,
            **{f"page_{index}": f"fixture_{index}" for index in range(1, 13)},
        },
        "outputs": {"partial_progress_pdf": master_binding},
        "limitations": [],
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    before_reader = pypdf.PdfReader(str(master), strict=False)
    before_hashes = [page7.page_content_hash(page) for page in before_reader.pages]
    monkeypatch.setattr(page7, "MASTER_PDF", master)
    monkeypatch.setattr(page7, "MASTER_PROVENANCE", provenance_path)
    monkeypatch.setattr(page7, "PAGE_PDF", replacement)
    monkeypatch.setattr(page7, "PAGE_PNG", page_png)
    monkeypatch.setattr(page7, "CURVE_JSON", curve_json)

    result = page7.replace_page(_replacement_curves())

    updated_reader = pypdf.PdfReader(str(master), strict=False)
    after_hashes = [page7.page_content_hash(page) for page in updated_reader.pages]
    assert result["page_count"] == 12
    assert len(after_hashes) == 12
    assert after_hashes[:6] == before_hashes[:6]
    assert after_hashes[7:] == before_hashes[7:]
    updated = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert updated["layout"]["page_count"] == 12
    assert updated["layout"]["page_12"] == "fixture_12"
    structural = updated["page_7_deltae_vs_salg"]["structural_validation"]
    assert structural["page_count"] == 12
    assert structural["preserved_page_content_sha256"] == (
        before_hashes[:6] + before_hashes[7:]
    )
    assert updated["outputs"]["partial_progress_pdf"]["sha256"] == (
        hashlib.sha256(master.read_bytes()).hexdigest()
    )
