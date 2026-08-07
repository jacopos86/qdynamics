import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject

from pipelines.reporting.add_paper_i_hh_singleton_own_plateau_page import (
    GEO_METHOD,
    MACRO_METHODS,
    METHODS,
    REGIMES,
    _latex_box_validation,
    _replace_promotion_state_with_evidence_status,
    _route_contract_parameter_row,
    _scope_geo_adapt_page_roles,
    promote_canonical_pdf,
)


def _write_blank_pdf(
    path: Path,
    pages: int,
    *,
    width: float = 612,
) -> None:
    writer = PdfWriter()
    for index in range(pages):
        page = writer.add_blank_page(width=width, height=792)
        contents = DecodedStreamObject()
        contents.set_data(
            (
                f"q\n% {path.name} page {index + 1}\n"
                f"1 0 0 1 {index} 0 cm\nQ\n"
            ).encode()
        )
        page.replace_contents(contents)
    with path.open("wb") as handle:
        writer.write(handle)


def _content_digest(page: object) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def test_promote_canonical_pdf_preserves_insertion_tail(tmp_path: Path) -> None:
    review_pdf = tmp_path / "review6.pdf"
    preserved_review_pdf = tmp_path / "review9.pdf"
    preserved_terminal_page_pdf = tmp_path / "page10.pdf"
    parameter_manifest_pdf = tmp_path / "parameter-manifest.pdf"
    final_pdf = tmp_path / "final.pdf"
    _write_blank_pdf(review_pdf, 6)
    _write_blank_pdf(preserved_review_pdf, 9)
    _write_blank_pdf(preserved_terminal_page_pdf, 1)
    _write_blank_pdf(parameter_manifest_pdf, 2)

    page_count = promote_canonical_pdf(
        review_pdf=review_pdf,
        preserved_review_pdf=preserved_review_pdf,
        preserved_terminal_page_pdf=preserved_terminal_page_pdf,
        parameter_manifest_pdf=parameter_manifest_pdf,
        final_pdf=final_pdf,
    )

    assert page_count == 12
    final_pages = PdfReader(str(final_pdf)).pages
    assert len(final_pages) == 12
    expected_pages = [
        *PdfReader(str(review_pdf)).pages,
        *PdfReader(str(preserved_review_pdf)).pages[6:9],
        PdfReader(str(preserved_terminal_page_pdf)).pages[0],
        *PdfReader(str(parameter_manifest_pdf)).pages,
    ]
    assert [_content_digest(page) for page in final_pages] == [
        _content_digest(page) for page in expected_pages
    ]


def test_canonical_pdf_fails_closed_on_nonletter_source(
    tmp_path: Path,
) -> None:
    review_pdf = tmp_path / "review6.pdf"
    preserved_review_pdf = tmp_path / "review9.pdf"
    preserved_terminal_page_pdf = tmp_path / "page10.pdf"
    parameter_manifest_pdf = tmp_path / "parameter-manifest.pdf"
    final_pdf = tmp_path / "final.pdf"
    _write_blank_pdf(review_pdf, 6)
    _write_blank_pdf(preserved_review_pdf, 9)
    _write_blank_pdf(preserved_terminal_page_pdf, 1, width=333)
    _write_blank_pdf(parameter_manifest_pdf, 2)

    with pytest.raises(RuntimeError, match="not letter size"):
        promote_canonical_pdf(
            review_pdf=review_pdf,
            preserved_review_pdf=preserved_review_pdf,
            preserved_terminal_page_pdf=preserved_terminal_page_pdf,
            parameter_manifest_pdf=parameter_manifest_pdf,
            final_pdf=final_pdf,
        )

    assert not final_pdf.exists()


def test_canonical_pdf_fails_closed_without_preserved_insertion_tail(
    tmp_path: Path,
) -> None:
    review_pdf = tmp_path / "review6.pdf"
    preserved_review_pdf = tmp_path / "missing-review9.pdf"
    preserved_terminal_page_pdf = tmp_path / "page10.pdf"
    parameter_manifest_pdf = tmp_path / "parameter-manifest.pdf"
    final_pdf = tmp_path / "final.pdf"
    _write_blank_pdf(review_pdf, 6)
    _write_blank_pdf(preserved_terminal_page_pdf, 1)
    _write_blank_pdf(parameter_manifest_pdf, 2)

    with pytest.raises(FileNotFoundError, match="nine-page"):
        promote_canonical_pdf(
            review_pdf=review_pdf,
            preserved_review_pdf=preserved_review_pdf,
            preserved_terminal_page_pdf=preserved_terminal_page_pdf,
            parameter_manifest_pdf=parameter_manifest_pdf,
            final_pdf=final_pdf,
        )

    assert not final_pdf.exists()


def test_canonical_pdf_fails_closed_without_preserved_page10(
    tmp_path: Path,
) -> None:
    review_pdf = tmp_path / "review6.pdf"
    preserved_review_pdf = tmp_path / "review9.pdf"
    preserved_terminal_page_pdf = tmp_path / "missing-page10.pdf"
    parameter_manifest_pdf = tmp_path / "parameter-manifest.pdf"
    final_pdf = tmp_path / "final.pdf"
    _write_blank_pdf(review_pdf, 6)
    _write_blank_pdf(preserved_review_pdf, 9)
    _write_blank_pdf(parameter_manifest_pdf, 2)

    with pytest.raises(FileNotFoundError, match="singleton insertion"):
        promote_canonical_pdf(
            review_pdf=review_pdf,
            preserved_review_pdf=preserved_review_pdf,
            preserved_terminal_page_pdf=preserved_terminal_page_pdf,
            parameter_manifest_pdf=parameter_manifest_pdf,
            final_pdf=final_pdf,
        )

    assert not final_pdf.exists()


def test_provenance_uses_objective_evidence_status_only() -> None:
    payload = {
        "manuscript_promotion": {
            "projected_numeric_claims_reportable": True,
        }
    }

    _replace_promotion_state_with_evidence_status(payload)

    assert "manuscript_promotion" not in payload
    assert payload["evidence_status"][
        "final_parameter_manifest_appended"
    ] is True
    assert payload["evidence_status"][
        "retained_diagnostic_pages_manifested"
    ] is True
    assert "reportable" not in json.dumps(payload)


def test_geo_page_roles_replace_ambiguous_top_level_exclusion() -> None:
    payload = {"geo_adapt_excluded": True}
    manifest = [
        {
            "method": "geo",
            "regime": regime,
        }
        for regime, *_rest in REGIMES
    ]

    _scope_geo_adapt_page_roles(
        payload,
        parameter_manifest=manifest,
    )

    assert "geo_adapt_excluded" not in payload
    assert payload["method_page_roles"]["geo_adapt"] == {
        "artifact_included": True,
        "route_id": GEO_METHOD["route_id"],
        "historical_macro_comparison_page_1": "included",
        "clean_common_accuracy_pages": "excluded",
        "final_parameter_manifest": (
            "included_as_six_historical_noncanonical_rows"
        ),
        "accounting_status": "historical_noncanonical",
    }


def test_latex_box_validation_reads_logs_instead_of_asserting_zero(
    tmp_path: Path,
) -> None:
    tex_path = tmp_path / "report.tex"
    tex_path.write_text("", encoding="utf-8")
    log_path = tex_path.with_suffix(".log")
    log_path.write_text("Output written on report.pdf.\n", encoding="utf-8")

    validation = _latex_box_validation(tex_path)

    assert validation["overfull_or_underfull_boxes"] == 0
    assert validation["logs"][0]["sha256"] == hashlib.sha256(
        log_path.read_bytes()
    ).hexdigest()
    log_path.write_text(
        "Overfull \\hbox (1.0pt too wide)\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="overfull or underfull"):
        _latex_box_validation(tex_path)


def test_final_manifest_method_labels_are_unambiguous() -> None:
    labels = [
        str(method["label"])
        for method in (*MACRO_METHODS, GEO_METHOD, *METHODS)
    ]

    assert len(labels) == len(set(labels))
    assert "SNAKE singleton" in labels


def test_retained_result_route_contract_projects_manifest_row() -> None:
    route_profile = "singleton_insertion_profile_v1"
    execution = {
        "adapt_inner_optimizer": "POWELL",
        "adapt_maxiter": 200,
        "adapt_seed": 7,
        "adapt_pool": "full_meta",
        "adapt_insertion_mode": "insertion_commutation_plateau_v1",
        "phase3_runtime_split_mode": "shortlist_pauli_children_v1",
        "phase3_runtime_split_max_subset_size": 1,
        "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "phase3_runtime_split_child_padding_policy": (
            "exact_projected_grouped_v1"
        ),
    }
    result = {
        "settings": {
            "L": 2,
            "t": 1.0,
            "u": 0.25,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.353553390593,
            "n_ph_max": 3,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "sr_route_profile_resolved": route_profile,
            "sr_route_profile_contract": {
                "route_profile": route_profile,
                "execution_settings": execution,
                "semantic_invariants": {
                    "full_meta_hva_policy": "included_no_filters_v1",
                },
            },
        },
        "adapt_vqe": {
            "exact_gs_energy": -0.918380919994822,
            "benchmark_stop_reference_energy": -0.918380919994822,
        },
    }

    row = _route_contract_parameter_row(
        regime="weak_weak",
        method_key="singleton_plateau_insertion",
        method_label="Singleton plateau-insertion SNAKE",
        route_id=route_profile,
        result_payload=result,
        source_receipt={
            "path": "locked/current.json",
            "sha256": "a" * 64,
        },
        compile_policy={
            "identity": "table_i_basis_gate_transpile_v1",
            "optimization_level": 0,
            "seed_transpiler": 7,
            "reference_state_included": True,
        },
    )
    assert row["physics"]["drive_enabled"] is False

    assert row["candidate"]["representation"] == "projected_singleton"
    assert row["candidate"]["sector_policy"] == "hard_guard"
    assert row["candidate"]["padding_policy"] == (
        "exact_projected_grouped_v1"
    )
    assert row["physics"]["same_cutoff_reference"] is True
    assert row["source"]["manifest_member"] == (
        "settings.sr_route_profile_contract"
    )
