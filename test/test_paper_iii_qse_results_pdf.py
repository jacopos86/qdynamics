"""Paper-III QSE results PDF tests.

PYTEST_DONT_REWRITE

This module carries large inline JSON/LaTeX fixtures. On some local agent
machines pytest assertion rewriting can dominate collection time; keep this file
on plain assertion mode so reporting smoke checks remain fast.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.source_maps import (  # noqa: E402
    PaperIIIQSESourceSpec,
    build_paper_iii_qse_source_map,
    sha256_file,
    validate_paper_iii_qse_source_map,
    write_paper_iii_qse_source_map,
)
from pipelines.qse_spectra.table_aggregate import (  # noqa: E402
    QSETableAggregateConfig,
    build_qse_table_aggregate,
)
from pipelines.reporting import build_paper_iii_qse_results_pdf as builder  # noqa: E402
from pipelines.reporting.build_paper_iii_qse_results_pdf import (  # noqa: E402
    COMPILE_AUTO,
    COMPILE_REQUIRE,
    COMPILE_SKIP,
    DEFAULT_AUDIT_JSON,
    DEFAULT_COMMAND_LOG_MD,
    DEFAULT_PDF,
    DEFAULT_REPORT_MANIFEST_JSON,
    DEFAULT_RUN_MANIFEST_JSON,
    DEFAULT_TEX,
    DEFAULT_WORK_DIR,
    REPORT_AUDIT_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    PaperIIIQSEResultsReportConfig,
    PaperIIIQSEResultsReportError,
    build_paper_iii_qse_results_report,
    default_output_paths,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _report_input(
    *,
    case_id: str,
    generated_utc: str,
    run_class: str = "candidate",
    approval_status: str = "user_review_required",
) -> dict[str, Any]:
    return {
        "schema_version": "paper_iii_qse_report_input_v1",
        "pipeline": "paper_iii_qse_results_report_input",
        "method_id": "qse_selection::geometry_selected",
        "run_class": run_class,
        "compatibility_tier": "n_ph_2_compatibility",
        "regime": case_id,
        "case_id": case_id,
        "visible_target": "tab:qse_static_claims",
        "approval_status": approval_status,
        "generated_utc": generated_utc,
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "exact_reference_role": "diagnostic_reporting_only",
        },
    }


def _qse_candidate_manifest() -> dict[str, Any]:
    boundary = {
        "feeds_controller_decisions": False,
        "uses_exact_reference_for_decision": False,
        "uses_future_exact_forecast_for_decision": False,
        "exact_reference_role": "diagnostic_reporting_only",
    }
    return {
        "schema_version": "qse_spectra_v1",
        "pipeline": "qse_spectra",
        "generated_utc": "2026-06-08T00:00:00Z",
        "run_tag": "hh_strong_weak_nph2_current_green_candidate",
        "settings": {"regime_label": "hh_strong_weak", "n_ph_max": 2},
        "input": {
            "hamiltonian": {"source_schema": "terms", "term_count_input": 4, "term_count_output": 3, "n_ph_max": 2},
            "operator_basis": {"source_schema": "artifact_basis_source:full_meta_filtered", "basis_size": 2},
        },
        "operator_basis": [
            {"basis_index": 0, "name": "e", "kind": "pauli_string", "pauli_exyz": "e"},
            {"basis_index": 1, "name": "x", "kind": "pauli_string", "pauli_exyz": "x"},
        ],
        "transition_observables": [
            {"name": "density", "transition_strengths": [0.0, 1.0]},
            {"name": "hh_J[positive_chain]", "transition_strengths": [0.0, 0.4]},
            {"name": "hh_K[positive_chain]", "transition_strengths": [0.1, 0.0]},
        ],
        "diagnostics": {
            "num_qubits": 8,
            "basis_size": 2,
            "retained_rank": 2,
            "discarded_rank": 0,
            "overlap_condition_estimate": 4.0,
        },
        "eigenvalues": [
            {"state_index": 0, "energy": -1.0, "generalized_residual_norm": 1.0e-10},
            {"state_index": 1, "energy": 0.2, "generalized_residual_norm": 2.0e-9},
        ],
        "static_record_selection": {
            "selection_config": {"mode": "geometry_selected", "geometry_target_roots": 6},
            "controller_boundary": {"controller_usable": False, "feeds_controller_decisions": False},
        },
        "paper_iii_contract": {
            "schema_version": "paper_iii_qse_production_contract_v1",
            "run_class": "candidate",
            "visible_target": "tab:qse_static_claims",
            "compatibility_tier": "n_ph_2_compatibility",
            "approval_status": "user_review_required",
            "controller_boundary": boundary,
            "hh_full_meta_provenance": {
                "layout": {"problem_key": "hh", "num_sites": 2, "n_ph_max": 2, "boson_encoding": "binary"}
            },
        },
        "qse_response_functions_v1": {
            "schema_version": "qse_response_functions_v1",
            "policy": "diagnostic_only_neutral_response_postprocessing",
            "response_kind": "neutral",
            "controller_boundary": {"feeds_controller_decisions": False, "controller_usable": False},
            "observables": [{"label": "density"}],
            "channels": [
                {
                    "A_label": "density",
                    "B_label": "density",
                    "channel_kind": "nn",
                    "sum_rule_deficits": {"status": "evaluated", "m0": {"deficit_abs": 0.0}},
                }
            ],
        },
        "qse_conductivity_response_v1": {
            "schema_version": "qse_conductivity_response_v1",
            "policy": "diagnostic_only_current_response_postprocessing",
            "response_kind": "conductivity_current",
            "controller_boundary": {"feeds_controller_decisions": False, "controller_usable": False},
            "frequency_grid": {"num_points": 3, "values": [0.0, 1.0, 2.0]},
            "contact_policy": {"name": "contact_record_only_no_drude_delta"},
            "peierls_policy": {"name": "standard_hh_1d_charge_peierls"},
            "regular_conductivity_policy": {"name": "positive_frequency_paramagnetic_sjj_over_omega"},
            "observables": [{"label": "hh_J[positive_chain]"}, {"label": "hh_K[positive_chain]"}],
            "channels": [
                {
                    "current_label": "hh_J[positive_chain]",
                    "contact_label": "hh_K[positive_chain]",
                    "channel_kind": "longitudinal_charge",
                    "current_source": {"status": "evaluated", "zero_current_source": False},
                    "contact_term": {"status": "evaluated"},
                    "drude_weight": {"status": "not_evaluated"},
                }
            ],
        },
        "qse_green_function_v1": {
            "schema_version": "qse_green_function_v1",
            "policy": "diagnostic_only_single_particle_green_function_postprocessing",
            "response_kind": "single_particle_retarded_green_function_diagonal",
            "controller_boundary": {"feeds_controller_decisions": False, "controller_usable": False},
            "frequency_grid": {"num_points": 3, "values": [0.0, 1.0, 2.0]},
            "mode_domain": {"fermion_mode_count": 4},
            "summary": {"mode_count": 2, "sector_count": 4, "solved_sector_count": 3, "zero_source_sector_count": 1},
            "modes": [
                {
                    "label": "up0",
                    "mode_index": 0,
                    "diagonal_sum_rule_diagnostics": {
                        "status": "evaluated",
                        "source_norm_canonical_deficit_abs": 0.0,
                    },
                },
                {
                    "label": "dn0",
                    "mode_index": 1,
                    "diagonal_sum_rule_diagnostics": {
                        "status": "evaluated",
                        "source_norm_canonical_deficit_abs": 0.0,
                    },
                },
            ],
        },
        "paper_iii_production_gate": {
            "schema_version": "paper_iii_qse_production_gate_v1",
            "ok": True,
            "production_ready": True,
            "exact_reference_boundary_status": {"status": "pass", "violation_count": 0},
        },
    }


def _source_map_json(tmp_path: Path, records: list[tuple[str, dict[str, Any]]]) -> Path:
    specs = []
    for stem, payload in records:
        path = _write_json(tmp_path / f"{stem}.json", payload)
        specs.append(PaperIIIQSESourceSpec(role="report_input", path=path))
    source_map = build_paper_iii_qse_source_map(specs, source_root=tmp_path, map_id="tmp_paper_iii_qse_results_map")
    output = tmp_path / "source_map.json"
    write_paper_iii_qse_source_map(source_map, output)
    return output


def _config(tmp_path: Path, source_map_json: Path, *, compile_mode: str = COMPILE_SKIP) -> PaperIIIQSEResultsReportConfig:
    work_dir = tmp_path / "paper_iii_qse_results"
    return PaperIIIQSEResultsReportConfig(
        source_map_jsons=(source_map_json,),
        base_dir=tmp_path,
        output_tex=work_dir / "paper_iii_qse_results.tex",
        output_pdf=tmp_path / "pdf" / "paper_iii_qse_results.pdf",
        report_manifest_json=work_dir / "paper_iii_qse_results.manifest.json",
        audit_json=work_dir / "paper_iii_qse_results.audit.json",
        run_manifest_json=work_dir / "run_manifest.json",
        command_log_md=work_dir / "command_log.md",
        compile_mode=compile_mode,
    )


def _machine_block(tex: str, name: str) -> dict[str, Any]:
    match = re.search(
        rf"^% BEGIN_MACHINE_READABLE_{re.escape(name)}\n(?P<body>.*?)^% END_MACHINE_READABLE_{re.escape(name)}",
        tex,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    body_lines = []
    for line in match.group("body").splitlines():
        body_lines.append(line[2:] if line.startswith("% ") else line[1:] if line.startswith("%") else line)
    return json.loads("\n".join(body_lines))


def test_stable_default_output_paths_are_singular() -> None:
    paths = default_output_paths()
    assert paths["work_dir"] == DEFAULT_WORK_DIR == Path("output/paper_iii_qse_results")
    assert paths["tex"] == DEFAULT_TEX == Path("output/paper_iii_qse_results/paper_iii_qse_results.tex")
    assert paths["pdf"] == DEFAULT_PDF == Path("output/pdf/paper_iii_qse_results.pdf")
    assert paths["report_manifest_json"] == DEFAULT_REPORT_MANIFEST_JSON
    assert paths["audit_json"] == DEFAULT_AUDIT_JSON
    assert paths["run_manifest_json"] == DEFAULT_RUN_MANIFEST_JSON
    assert paths["command_log_md"] == DEFAULT_COMMAND_LOG_MD


def test_build_report_writes_tex_sidecars_and_orders_newest_first(tmp_path: Path) -> None:
    source_map_json = _source_map_json(
        tmp_path,
        [
            ("old_case", _report_input(case_id="old_case", generated_utc="2026-06-07T00:00:00Z")),
            (
                "paper_facing_mid_case",
                _report_input(
                    case_id="paper_facing_mid_case",
                    generated_utc="2026-06-07T00:30:00Z",
                    run_class="paper_facing",
                    approval_status="paper_facing_approved",
                ),
            ),
            ("new_case", _report_input(case_id="new_case", generated_utc="2026-06-07T01:00:00Z")),
        ],
    )
    config = _config(tmp_path, source_map_json)

    manifest = build_paper_iii_qse_results_report(
        config,
        command=("python", "-m", "pipelines.reporting.build_paper_iii_qse_results_pdf", "--source-map-json", str(source_map_json)),
    )

    assert manifest["schema_version"] == REPORT_SCHEMA_VERSION
    assert manifest["compile"]["status"] == "skipped_disabled"
    assert config.output_tex.exists()
    assert config.report_manifest_json.exists()
    assert config.audit_json.exists()
    assert config.run_manifest_json.exists()
    assert config.command_log_md.exists()

    tex = config.output_tex.read_text(encoding="utf-8")
    assert tex.index(r"new\_case") < tex.index(r"paper\_facing\_mid\_case") < tex.index(r"old\_case")
    assert r"Run class: candidate; approval: user\_review\_required" in tex
    assert r"Run class: paper\_facing; approval: paper\_facing\_approved" in tex
    assert "candidate and user-review-required records are not promoted" in tex

    audit = json.loads(config.audit_json.read_text(encoding="utf-8"))
    assert audit["schema_version"] == REPORT_AUDIT_SCHEMA_VERSION
    assert audit["ok"] is True
    assert audit["source_map_references"][0]["source_map_sha256"] == sha256_file(source_map_json)

    run_manifest = json.loads(config.run_manifest_json.read_text(encoding="utf-8"))
    assert run_manifest["schema_version"] == "agent_run_manifest_v1"
    assert run_manifest["no_run_or_chtc_work"] is True
    assert run_manifest["no_manuscript_edit"] is True


def test_tex_machine_readable_comments_carry_required_provenance(tmp_path: Path) -> None:
    source_map_json = _source_map_json(
        tmp_path,
        [("case_a", _report_input(case_id="case_a", generated_utc="2026-06-07T00:00:00Z"))],
    )
    config = _config(tmp_path, source_map_json)
    build_paper_iii_qse_results_report(config, command="paper-iii-qse-report-test-command")

    tex = config.output_tex.read_text(encoding="utf-8")
    report_block = _machine_block(tex, "PAPER_III_QSE_RESULTS_REPORT")
    assert report_block["schema_version"] == REPORT_SCHEMA_VERSION
    assert report_block["command_line"] == "paper-iii-qse-report-test-command"
    assert report_block["source_maps"][0]["map_id"] == "tmp_paper_iii_qse_results_map"
    assert report_block["source_maps"][0]["source_map_sha256"] == sha256_file(source_map_json)

    source = report_block["sources"][0]
    assert source["source_id"].startswith("report_input:")
    assert source["source_path"] == "case_a.json"
    assert source["source_sha256"] == sha256_file(tmp_path / "case_a.json")
    assert source["schema_version"] == "paper_iii_qse_report_input_v1"
    assert source["run_class"] == "candidate"
    assert source["approval_status"] == "user_review_required"
    assert source["compatibility_tier"] == "n_ph_2_compatibility"

    section_block = _machine_block(tex, "PAPER_III_QSE_RESULT_SECTION")
    assert section_block["source"]["source_path"] == "case_a.json"
    assert section_block["source_map"]["sha256"] == sha256_file(source_map_json)


def test_tmp_path_hh_strong_weak_current_green_workflow_reaches_no_compile_report(tmp_path: Path) -> None:
    qse_manifest = _write_json(tmp_path / "qse_candidate.json", _qse_candidate_manifest())
    aggregate_payload = build_qse_table_aggregate(
        QSETableAggregateConfig(qse_manifest_paths=(qse_manifest,), output_json=tmp_path / "aggregate.json")
    )
    aggregate_json = _write_json(tmp_path / "aggregate.json", aggregate_payload)

    assert aggregate_payload["summary"]["rows_with_response_payload"] == 1
    assert aggregate_payload["summary"]["rows_with_conductivity_payload"] == 1
    assert aggregate_payload["summary"]["rows_with_green_function_payload"] == 1

    source_map = build_paper_iii_qse_source_map(
        [
            PaperIIIQSESourceSpec(role="qse_manifest", path=qse_manifest),
            PaperIIIQSESourceSpec(role="aggregate", path=aggregate_json),
        ],
        source_root=tmp_path,
        map_id="tmp_hh_strong_weak_nph2_current_green",
    )
    audit = validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)
    assert audit["ok"] is True
    source_map_json = tmp_path / "source_map_current_green.json"
    write_paper_iii_qse_source_map(source_map, source_map_json)

    config = _config(tmp_path, source_map_json, compile_mode=COMPILE_SKIP)
    manifest = build_paper_iii_qse_results_report(config, command="tmp-current-green-workflow")

    assert manifest["compile"]["status"] == "skipped_disabled"
    assert manifest["summary"]["source_role_counts"] == {"aggregate": 1, "qse_manifest": 1}
    tex = config.output_tex.read_text(encoding="utf-8")
    assert "current/conductivity channels" in tex
    assert "Green-function modes" in tex
    assert "rows with conductivity/current" in tex
    assert "rows with Green functions" in tex


def test_manifest_page_contains_provenance_context(tmp_path: Path) -> None:
    source_map_json = _source_map_json(
        tmp_path,
        [("case_manifest", _report_input(case_id="case_manifest", generated_utc="2026-06-07T00:00:00Z"))],
    )
    config = _config(tmp_path, source_map_json)
    build_paper_iii_qse_results_report(config, command="manifest-page-test")

    tex = config.output_tex.read_text(encoding="utf-8")
    assert "PARAMETER / PROVENANCE MANIFEST" in tex
    assert "Source-map count" in tex
    assert "Approval policy" in tex
    assert "Exact/reference data policy" in tex
    assert "Newest-first sort" in tex
    assert "This results PDF is separate from the Paper III manuscript" in tex
    assert "\\clearpage" in tex


def test_auto_compile_skips_gracefully_when_tex_is_unavailable(tmp_path: Path, monkeypatch: Any) -> None:
    source_map_json = _source_map_json(
        tmp_path,
        [("case_auto", _report_input(case_id="case_auto", generated_utc="2026-06-07T00:00:00Z"))],
    )
    config = _config(tmp_path, source_map_json, compile_mode=COMPILE_AUTO)
    monkeypatch.setattr(builder.shutil, "which", lambda _name: None)

    manifest = build_paper_iii_qse_results_report(config, command="auto-compile-test")

    assert manifest["compile"]["status"] == "skipped_tex_unavailable"
    assert manifest["compile"]["engine"] is None
    assert config.output_tex.exists()
    assert not config.output_pdf.exists()
    command_log = config.command_log_md.read_text(encoding="utf-8")
    assert "skipped_tex_unavailable" in command_log


def test_require_compile_fails_when_tex_is_unavailable(tmp_path: Path, monkeypatch: Any) -> None:
    source_map_json = _source_map_json(
        tmp_path,
        [("case_require", _report_input(case_id="case_require", generated_utc="2026-06-07T00:00:00Z"))],
    )
    config = _config(tmp_path, source_map_json, compile_mode=COMPILE_REQUIRE)
    monkeypatch.setattr(builder.shutil, "which", lambda _name: None)

    with pytest.raises(PaperIIIQSEResultsReportError, match="No LaTeX engine available"):
        build_paper_iii_qse_results_report(config, command="require-compile-test")


def test_builder_fails_closed_on_stale_source_map_hash(tmp_path: Path) -> None:
    source_map_json = _source_map_json(
        tmp_path,
        [("case_stale", _report_input(case_id="case_stale", generated_utc="2026-06-07T00:00:00Z"))],
    )
    source = tmp_path / "case_stale.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["case_id"] = "mutated_after_source_map"
    _write_json(source, payload)

    with pytest.raises(Exception, match="source_sha256_mismatch"):
        build_paper_iii_qse_results_report(_config(tmp_path, source_map_json), command="stale-source-test")


def test_main_records_actual_cli_arguments(tmp_path: Path, monkeypatch: Any) -> None:
    source_map_json = _source_map_json(
        tmp_path,
        [("case_cli", _report_input(case_id="case_cli", generated_utc="2026-06-07T00:00:00Z"))],
    )
    config = _config(tmp_path, source_map_json, compile_mode=COMPILE_SKIP)
    argv = [
        "build_paper_iii_qse_results_pdf.py",
        "--source-map-json",
        str(source_map_json),
        "--base-dir",
        str(tmp_path),
        "--output-tex",
        str(config.output_tex),
        "--output-pdf",
        str(config.output_pdf),
        "--report-manifest-json",
        str(config.report_manifest_json),
        "--audit-json",
        str(config.audit_json),
        "--run-manifest-json",
        str(config.run_manifest_json),
        "--command-log-md",
        str(config.command_log_md),
        "--no-compile",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert builder.main(None) == 0

    report_manifest = json.loads(config.report_manifest_json.read_text(encoding="utf-8"))
    assert "--source-map-json" in report_manifest["command_line"]
    assert str(source_map_json) in report_manifest["command_line"]
    tex = config.output_tex.read_text(encoding="utf-8")
    report_block = _machine_block(tex, "PAPER_III_QSE_RESULTS_REPORT")
    assert "--no-compile" in report_block["command_line"]
