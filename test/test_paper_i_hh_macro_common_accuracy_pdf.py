from __future__ import annotations

import io
import json
import tarfile

import pytest
from pypdf import PdfReader, PdfWriter

from pipelines.reporting import (
    build_paper_i_hh_macro_common_accuracy_pdf as report,
)


def _compile_policy() -> dict:
    return {
        "identity": "table_i_basis_gate_transpile_v1",
        "optimization_level": 0,
        "seed_transpiler": 7,
        "reference_state_included": True,
    }


def _append_manifest() -> dict:
    return {
        "schema": "paper_i_hh_append_completion_job_v1",
        "family": "hh",
        "seed": 7,
        "physics": {
            "L": 2,
            "t": 1.0,
            "u": 0.25,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.353553390593,
            "n_ph_work": 3,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "same_cutoff_reference": True,
        },
        "optimizer": {"kind": "powell", "maxiter": 200},
        "candidate_pool": {"parent_pool": "full_meta_unfiltered"},
        "variant": {
            "candidate_representation": "unsplit_full_meta_macro_parent"
        },
        "exact_reference": {
            "energy": -0.918380919994822,
            "usage": "reporting_only_after_optimization",
        },
    }


def _snake_manifest() -> dict:
    return {
        "schema": "paper_i_hh_sr_symcost_noprune_runtime_manifest_v1",
        "physics": {
            "problem": "hh",
            "L": 2,
            "t": 1.0,
            "u_over_t": 0.25,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.353553390593,
            "n_ph_work": 3,
            "same_cutoff_reference": True,
            "expected_exact_energy": -0.918380919994822,
        },
        "route_identity": {
            "profile_resolved": "sr_snake_macro_only_physical_lanes_v1",
            "profile_contract": {
                "execution_settings": {
                    "adapt_inner_optimizer": "POWELL",
                    "adapt_maxiter": 200,
                    "adapt_seed": 7,
                    "adapt_pool": "full_meta",
                    "adapt_insertion_mode": "append_only",
                    "phase3_runtime_split_mode": "off",
                    "phase3_runtime_split_max_subset_size": 1,
                    "phase3_runtime_split_child_set_symmetry_policy": (
                        "hard_guard"
                    ),
                    "phase3_runtime_split_child_padding_policy": (
                        "unchecked_diagnostic_v1"
                    ),
                },
                "semantic_invariants": {
                    "full_meta_hva_policy": "included_no_filters_v1",
                }
            },
        },
    }


def _geo_manifest() -> dict:
    payload = _append_manifest()
    payload.update(
        {
            "schema": "paper_i_hh_geo_completion_job_v1",
            "algorithm_id": "geo_adapt",
            "method_id": "geo_macro",
            "optimizer": {
                "kind": "powell",
                "maxiter": 200,
                "position_policy": "append",
            },
        }
    )
    return payload


def _source_receipt() -> dict:
    return {
        "path": "locked/source.tar.gz",
        "sha256": "a" * 64,
        "manifest_member": "locked/normalized_manifest.json",
        "manifest_member_sha256": "b" * 64,
    }


def test_parameter_manifest_projects_both_locked_manifest_schemas() -> None:
    snake = report._parameter_manifest_row(
        regime="weak_weak",
        method=report.METHODS[0],
        normalized_manifest=_snake_manifest(),
        source_receipt=_source_receipt(),
        compile_policy=_compile_policy(),
    )
    append = report._parameter_manifest_row(
        regime="weak_weak",
        method=report.METHODS[1],
        normalized_manifest=_append_manifest(),
        source_receipt=_source_receipt(),
        compile_policy=_compile_policy(),
    )
    geo = report._parameter_manifest_row(
        regime="weak_weak",
        method={
            "key": "geo",
            "label": "Geo-ADAPT",
            "route_id": "geo_adapt_macro_nph3_7",
        },
        normalized_manifest=_geo_manifest(),
        source_receipt=_source_receipt(),
        compile_policy=_compile_policy(),
    )

    assert snake["physics"]["u"] == 0.25
    assert snake["optimizer"] == {
        "kind": "POWELL",
        "maxiter": 200,
        "seed": 7,
    }
    assert append["candidate"]["identity"] == (
        "unsplit_full_meta_macro_parent"
    )
    assert snake["candidate"]["identity"] == (
        "sr_snake_macro_only_physical_lanes_v1"
    )
    assert snake["candidate"]["sector_policy"] == (
        "not_applicable_intact_macro"
    )
    assert snake["candidate"]["padding_policy"] == (
        "not_applicable_intact_macro"
    )
    assert snake["physics"]["exact_gs_energy"] == (
        -0.918380919994822
    )
    assert snake["physics"]["drive_enabled"] is False
    assert append["physics"]["drive_enabled"] is False
    assert geo["physics"]["drive_enabled"] is False
    assert append["compile"]["identity"] == (
        "table_i_basis_gate_transpile_v1"
    )
    assert geo["candidate"]["insertion"] == "append"
    assert geo["accounting"] == {
        "schema": "paper_i_historical_geo_accounting_v1",
        "contract": "closed_cumulative_unique_estimator_prefix",
        "canonical_s_alg": False,
    }


def test_parameter_manifest_rejects_drive_enabled_static_source() -> None:
    manifest = _append_manifest()
    manifest["physics"]["drive_enabled"] = True

    with pytest.raises(ValueError, match="static no-drive"):
        report._parameter_manifest_row(
            regime="weak_weak",
            method=report.METHODS[1],
            normalized_manifest=manifest,
            source_receipt=_source_receipt(),
            compile_policy=_compile_policy(),
        )


def test_normalized_manifest_reader_streams_only_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    archive_path = tmp_path / "source.tar.gz"
    member_name = "locked/run/normalized_run_manifest.json"
    raw = json.dumps(_snake_manifest(), sort_keys=True).encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
        result = b"should-not-be-read"
        result_info = tarfile.TarInfo("locked/run/json/result.json")
        result_info.size = len(result)
        archive.addfile(result_info, io.BytesIO(result))
    monkeypatch.setattr(report, "_source_path", lambda _source: archive_path)
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)

    manifest, receipt = report._read_normalized_source_manifest(
        {"member": "locked/run/json/result.json"}
    )

    assert manifest == _snake_manifest()
    assert receipt["manifest_member"] == member_name
    assert receipt["path"] == "source.tar.gz"


def test_write_tex_appends_manifest_and_status_has_no_promotion_claim(
    tmp_path,
) -> None:
    rows = []
    plateau_rows = []
    manifest = []
    for regime, _title, _abbreviation, n_ph in report.REGIMES:
        for method in report.METHODS:
            rows.append(
                {
                    "regime": regime,
                    "method": method["key"],
                    "method_label": method["label"],
                    "common_window_end": 2,
                    "common_error": 0.1,
                    "k_cross": 2,
                    "crossing_error": 0.1,
                    "N2q": 4,
                    "D2q": 3,
                    "Dc": 8,
                    "S_alg": 20,
                }
            )
            plateau_rows.append(
                {
                    "regime": regime,
                    "method": method["key"],
                    "method_label": method["label"],
                    "k_pl": 2,
                    "error": 0.1,
                    "N2q": 4,
                    "D2q": 3,
                    "Dc": 8,
                    "S_alg": 20,
                }
            )
            manifest.append(
                {
                    "regime": regime,
                    "method": method["key"],
                    "method_label": method["label"],
                    "route_id": method["route_id"],
                    "physics": {
                        "family": "hh",
                        "L": 2,
                        "t": 1.0,
                        "u": 0.25,
                        "dv": 0.0,
                        "omega0": 1.0,
                        "g_ep": 0.353553390593,
                        "n_ph_max": n_ph,
                        "exact_gs_energy": -0.918380919994822,
                        "boson_encoding": "binary",
                        "ordering": "blocked",
                            "boundary": "open",
                            "drive_enabled": False,
                            "same_cutoff_reference": True,
                        "exact_reference_usage": "reporting_only",
                    },
                    "optimizer": {
                        "kind": "POWELL",
                        "maxiter": 200,
                        "seed": 7,
                    },
                    "candidate": {
                        "identity": "intact_macro",
                        "pool": "full_meta",
                        "insertion": "append_only",
                        "representation": "intact_macro",
                        "hva_policy": "included_no_filters_v1",
                        "sector_policy": "hard_guard",
                        "padding_policy": "unchecked_diagnostic_v1",
                    },
                    "compile": _compile_policy(),
                    "accounting": {
                        "schema": "paper_i_clean_algorithm_s_alg_v3",
                        "contract": (
                            "required_executed_logical_scalar_estimator_"
                            "invocations_v1"
                        ),
                        "canonical_s_alg": True,
                    },
                    "source": _source_receipt(),
                }
            )
    tex_path = tmp_path / "report.tex"
    report.write_tex(
        rows=rows,
        plateau_rows=plateau_rows,
        parameter_manifest=manifest,
        plateau_plot=tmp_path / "plateau.png",
        crossing_plot=tmp_path / "crossing.png",
        tex_path=tex_path,
    )

    tex = tex_path.read_text(encoding="utf-8")
    assert tex.index(r"\section*{Parameter manifest}") > tex.index(
        "shared pre-plateau accuracy"
    )
    assert "Normalized contract SHA-256" in tex
    assert r"\texttt{paper\_\allowbreak{}i" in tex
    assert r"Same-cutoff reference:" in tex
    assert r"\texttt{binary}" in tex
    assert r"\texttt{blocked}" in tex
    assert r"\texttt{open}" in tex
    assert r"Drive enabled:" in tex
    assert r"\texttt{false}" in tex
    manifest_tex_path = tmp_path / "parameter-manifest.tex"
    report.write_parameter_manifest_tex(
        parameter_manifest=manifest,
        tex_path=manifest_tex_path,
        retained_page_manifest=[
            {
                "pages": "7--10",
                "label": "Retained insertion diagnostics",
                "route_ids": [
                    "sr_macro_commutation_reduced_insertion_nph3_7",
                    "singleton_insertion_profile_v1",
                ],
                "provenance": {
                    "path": "locked/insertion-provenance.json",
                    "sha256": "c" * 64,
                },
            }
        ],
    )
    standalone = manifest_tex_path.read_text(encoding="utf-8")
    assert r"\section*{Parameter manifest}" in standalone
    assert r"\includegraphics" not in standalone
    assert "Retained insertion diagnostics" in standalone
    assert r"sr\_\allowbreak{}macro\_\allowbreak{}commutation" in standalone
    status = report._evidence_status()
    assert "promotion" not in status
    assert all("reportable" not in key for key in status)


def test_base_builder_has_distinct_output_identity() -> None:
    assert report.BASE_REPORT_STEM != report.STEM
    assert report.BASE_REPORT_STEM.endswith("_macro_base_with_manifest")


def test_reader_pages_compatibility_copy_contains_exactly_two_pages(
    tmp_path,
) -> None:
    source = tmp_path / "source.pdf"
    target = tmp_path / "reader-pages.pdf"
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=72, height=72)
    with source.open("wb") as handle:
        writer.write(handle)

    report._write_reader_pages_copy(
        source_pdf=source,
        target_pdf=target,
    )

    assert len(PdfReader(str(target)).pages) == 2
