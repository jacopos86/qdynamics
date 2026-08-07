"""Paper-I fixed-accuracy PDF table tests.

PYTEST_DONT_REWRITE

This module carries large inline JSON/LaTeX fixtures. On some local agent
machines pytest assertion rewriting can dominate collection time; keep this file
on plain assertion mode so reporting smoke checks remain fast.
"""

from pathlib import Path

from pipelines.reporting.build_paper_i_fixed_accuracy_table_pdf import (
    TableCell,
    build_audit_manifest,
    cell_from_row,
    format_delta,
    load_rows_with_diagnostics,
    normalize_family,
    render_diagnostics,
    render_tex,
)


def test_zero_values_render_as_zero_not_missing():
    row = {
        "abs_delta_e": 0.0,
        "threshold_status": "ok_native_first_hit",
        "cost_included": True,
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        "count_2q": 0,
        "depth_2q": 0,
        "circuit_depth": 0,
        "S_alg": 0,
    }
    assert format_delta(row, 2e-4) == "0"
    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "0", "0", "0", "0")


def test_not_reached_rows_render_valid_s_alg_work():
    row = {
        "abs_delta_e": 0.042,
        "threshold_status": "not_reached",
        "cost_included": False,
        "count_2q": 0,
        "depth_2q": 0,
        "circuit_depth": 0,
        "S_alg": 123,
    }
    assert cell_from_row(row, 2e-4) == TableCell("4.18e-02", "--", "0", "0", "0", "123")


def test_raw_shot_values_do_not_render_as_s_alg():
    row = {
        "abs_delta_e": 1e-5,
        "threshold_status": "ok_native_first_hit",
        "cost_included": True,
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        "count_2q": 12,
        "depth_2q": 7,
        "circuit_depth": 21,
        "shot_cost_proxy": 999,
        "measurement_shots_proxy": 888,
        "shots_total": 777,
    }
    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "12", "7", "21", "--")


def test_terminal_upper_bound_s_alg_proxy_does_not_render_as_s_alg():
    row = {
        "abs_delta_e": 1e-5,
        "threshold_status": "ok_native_first_hit",
        "cost_included": True,
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        "count_2q": 12,
        "depth_2q": 7,
        "circuit_depth": 21,
        "terminal_S_alg_upper_bound": 999,
    }
    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "12", "7", "21", "--")


def test_legacy_s_norm_values_do_not_render_as_s_alg():
    row = {
        "abs_delta_e": 1e-5,
        "threshold_status": "ok_native_first_hit",
        "cost_included": True,
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        "count_2q": 12,
        "depth_2q": 7,
        "circuit_depth": 21,
        "S_norm": 31,
    }
    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "12", "7", "21", "--")


def test_running_current_best_reached_needs_complete_compiled_triplet():
    row = {
        "abs_delta_e": 1e-8,
        "threshold_status": "running_current_best_reached",
        "cost_included": False,
        "count_2q": 64,
        "circuit_depth": 237,
        "S_alg": 97,
    }
    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "--", "--", "--", "--")


def test_valid_qiskit_compiled_first_hit_cost_displays():
    row = {
        "abs_delta_e": 1e-5,
        "threshold_status": "ok_native_first_hit",
        "cost_included": True,
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        "count_2q": 12,
        "depth_2q": 7,
        "circuit_depth": 21,
        "S_alg": 31,
    }

    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "12", "7", "21", "31")


def test_enrichment_row_updates_hydrate_current_best_resources(tmp_path: Path):
    record_id = "static_table__hubbard__hubbard_L2_clean_weak__static_family_informed_vqe"
    enrichment = tmp_path / "enrichment" / record_id / "result"
    enrichment.mkdir(parents=True)
    (enrichment / "generic_static_metric_enrichment.json").write_text(
        """
        {
          "row_updates": {
            "compiled_count_2q_total": 56,
            "compiled_depth_2q_total": 52,
            "compiled_depth_total": 280,
            "S_alg": 147,
            "infidelity_same": 0.000123
          }
        }
        """,
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        f"""
        {{
          "enrichment_root":"{tmp_path / 'enrichment'}",
          "row_results": [
            {{"record_id":"{record_id}","family":"hubbard","case_id":"hubbard_L2_clean_weak","method":"family-informed VQE","threshold":0.0002,"abs_delta_e":0.0,"threshold_status":"ok_native_first_hit","cost_included":false}}
          ]
        }}
        """,
        encoding="utf-8",
    )

    rows, _diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    assert rows[("hubbard", "family-informed VQE", "weak")] == TableCell("0", "1.23e-04", "56", "52", "280", "147")


def test_payload_compile_scout_hydrates_missing_two_qubit_depth(tmp_path: Path):
    result_dir = tmp_path / "result"
    compile_dir = result_dir / "h2/json"
    compile_dir.mkdir(parents=True)
    payload = result_dir / "generic_static_single.json"
    payload.write_text(
        """
        {
          "result": {
            "family":"molecular_vibronic_h2",
            "case_id":"molecular_vibronic_h2_L2_nph1_clean_weak",
            "abs_delta_e":0.00003,
            "S_alg":152,
            "compiled_count_2q_total":70,
            "compiled_depth_total":274,
            "compile_json":"h2/json/compile_scout_fake_marrakesh.json"
          }
        }
        """,
        encoding="utf-8",
    )
    (compile_dir / "compile_scout_fake_marrakesh.json").write_text(
        """
        {"selected_backend":{"compiled_count_2q":70,"compiled_depth_2q":66,"compiled_depth":274}}
        """,
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        f"""
        {{
          "row_results": [
            {{"family":"molecular_vibronic_h2","case_id":"molecular_vibronic_h2_L2_nph1_clean_weak","method":"SNAKE","threshold":0.0002,"abs_delta_e":0.00003,"threshold_status":"terminal_upper_bound_missing_native_first_hit","cost_included":false,"payload_path":"{payload}"}}
          ]
        }}
        """,
        encoding="utf-8",
    )

    rows, _diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    assert rows[("molecular_vibronic_h2", "SNAKE", "weak")] == TableCell("0", "--", "70", "66", "274", "152")


def test_expected_audit_preserves_valid_resource_display_decision(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
        {
          "row_results": [
            {"family":"hubbard","case_id":"hubbard_L2_clean_weak","method":"Pos-Geo-ADAPT","threshold":0.0002,"abs_delta_e":0.00001,"threshold_status":"ok_native_first_hit","cost_included":true,"cost_source":"algorithmic_measurement_work","source":"native_adaptive_iteration","resource_display_allowed":true,"compiled_resource_validation_status":"ok","first_hit_cost_source_kind":"qiskit_compiled_first_hit_ansatz_circuit","count_2q":12,"depth_2q":7,"circuit_depth":21,"S_alg":31}
          ]
        }
        """,
        encoding="utf-8",
    )

    _rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    hubbard = by_key[("hubbard", "Pos-Geo-ADAPT", "weak")]

    assert hubbard["display_n_2q"] == "12"
    assert hubbard["resource_display_allowed"] is True
    assert hubbard["resource_display_reason"] == "qiskit_compiled_current_best_or_first_hit_cost"
    assert audit["cost_semantics"]["numeric_resource_semantics_pass"] is True


def test_missing_two_qubit_depth_suppresses_qiskit_resource_costs():
    row = {
        "abs_delta_e": 1e-5,
        "threshold_status": "ok_native_first_hit",
        "cost_included": True,
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        "count_2q": 12,
        "circuit_depth": 21,
        "S_alg": 31,
    }

    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "--", "--", "--", "--")


def test_invalid_two_qubit_depth_ordering_suppresses_costs():
    row = {
        "abs_delta_e": 1e-5,
        "threshold_status": "ok_native_first_hit",
        "cost_included": True,
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        "count_2q": 12,
        "depth_2q": 22,
        "circuit_depth": 21,
        "S_alg": 31,
    }

    assert cell_from_row(row, 2e-4) == TableCell("0", "--", "--", "--", "--", "--")


def test_delta_e_fallback_does_not_double_subtract_ambiguous_delta_e():
    assert format_delta({"delta_e": 1e-3}, 2e-4) == "--"
    assert format_delta({"delta_e_excess": 1e-3}, 2e-4) == "1.00e-03"


def test_family_fallback_prefers_longest_specific_alias():
    assert normalize_family({"case_id": "ionic_hubbard_L2_clean_weak"}) == "ionic_hubbard"
    assert normalize_family({"case_id": "extended_hubbard_L2_clean_weak"}) == "extended_hubbard"
    assert normalize_family({"case_id": "hh_L2_nph2_clean_weak"}) == "hh"
    assert normalize_family({"family": "harmonic_kerr"}) == "harmonic_kerr_chain"


def test_render_tex_includes_cutoff_strength_column():
    tex = render_tex(
        source_pdf=None,
        summary_json=None,
        threshold=2e-4,
        rows={},
        n_ph_algorithm="2",
        n_ph_ed="4",
    )

    assert "cutoff/strength" in tex
    assert "S_{\\rm alg}" in tex
    assert "S_{\\rm norm}" not in tex
    assert r"\multicolumn{15}" in tex
    assert r"Hubbard & --; $2/8$ & HEA VQE" in tex
    assert r"Bose-Hubbard & $(1,4),(1,4);\,2/6$ & HEA VQE" in tex
    assert r"molecular-vibronic H$_2$ & \(\begin{array}{@{}c@{}}(1,4),(3,6);\\0.25/1.0\end{array}\) & HEA VQE" in tex


def test_display_overlay_overrides_newer_diagnostic_snake_row(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
        {
          "row_results": [
            {"family":"hh","case_id":"hh_L2_nph2_clean_weak","method":"SNAKE","threshold":0.0002,"abs_delta_e":0.000235,"threshold_status":"terminal_upper_bound_missing_native_first_hit","cost_included":false,"resource_display_allowed":false,"count_2q":686,"depth_2q":615,"circuit_depth":2186,"S_alg":6117}
          ]
        }
        """,
        encoding="utf-8",
    )
    overlay = tmp_path / "approved_overlay.json"
    overlay.write_text(
        """
        {
          "schema":"paper_i_display_cell_overlay_v1",
          "threshold":0.0002,
          "source_label":"approved_pdf_text_recovery_test",
          "row_results": [
            {"family":"hh","case_id":"hh_L2_nph1_clean_weak","method":"SNAKE","regime":"weak","display_delta_e":"0","display_infidelity":"3.27e-05","display_n_2q":"8","display_d_2q":"4","display_d_circ":"16","S_alg":1768}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4, display_overlay_json=overlay)

    assert rows[("hh", "SNAKE", "weak")] == TableCell("0", "3.27e-05", "8", "4", "16", "1768")
    assert diagnostics.display_overlay_metadata["accepted_display_row_count"] == 1
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    hh_snake = by_key[("hh", "SNAKE", "weak")]
    assert hh_snake["display_overlay_role"] == "approved_table_cell"
    assert hh_snake["display_n_2q"] == "8"
    assert hh_snake["source"] == "approved_pdf_recovered_cell_overlay"


def test_display_overlay_keeps_qiskit_triplet_without_legacy_s_norm_as_s_alg(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text("""{"row_results": []}""", encoding="utf-8")
    overlay = tmp_path / "approved_overlay.json"
    overlay.write_text(
        """
        {
          "schema":"paper_i_display_cell_overlay_v1",
          "threshold":0.0002,
          "source_label":"approved_pdf_text_recovery_test",
          "row_results": [
            {"family":"hh","case_id":"hh_L2_nph1_clean_weak","method":"SNAKE","regime":"weak","display_delta_e":"0","display_infidelity":"3.27e-05","display_n_2q":"8","display_d_2q":"4","display_d_circ":"16","display_s_norm":"1768","S_norm":1768}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4, display_overlay_json=overlay)

    assert rows[("hh", "SNAKE", "weak")] == TableCell("0", "3.27e-05", "8", "4", "16", "--")
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    hh_snake = by_key[("hh", "SNAKE", "weak")]
    assert hh_snake["display_overlay_role"] == "approved_table_cell"
    assert hh_snake["resource_display_allowed"] is True
    assert hh_snake["resource_display_reason"] == "approved_pdf_recovered_qiskit_cost_s_alg_missing"
    assert hh_snake["display_n_2q"] == "8"
    assert hh_snake["display_s_alg"] == "--"
    assert audit["cost_semantics"]["forbidden_numeric_resource_source_count"] == 0
    assert audit["cost_semantics"]["numeric_resource_semantics_pass"] is True


def test_display_overlay_text_placeholders_do_not_count_as_numeric_resources(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text("""{"row_results": []}""", encoding="utf-8")
    overlay = tmp_path / "approved_overlay.json"
    overlay.write_text(
        """
        {
          "schema":"paper_i_display_cell_overlay_v1",
          "threshold":0.0002,
          "source_label":"approved_pdf_text_recovery_test",
          "row_results": [
            {"family":"harmonic_kerr_chain","case_id":"harmonic_kerr_chain_L2_nph4_clean_weak","method":"Pos-Geo-ADAPT","regime":"weak","display_delta_e":"n/a","display_infidelity":"n/a","display_n_2q":"n/a","display_d_2q":"n/a","display_d_circ":"n/a","display_s_alg":"--"}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4, display_overlay_json=overlay)

    assert rows[("harmonic_kerr_chain", "Pos-Geo-ADAPT", "weak")] == TableCell(
        "n/a", "n/a", "n/a", "n/a", "n/a", "--"
    )
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    row = by_key[("harmonic_kerr_chain", "Pos-Geo-ADAPT", "weak")]
    assert row["resource_numeric_displayed"] is False
    assert row["resource_display_allowed"] is False
    assert audit["cost_semantics"]["forbidden_numeric_resource_source_count"] == 0
    assert audit["cost_semantics"]["numeric_resource_semantics_pass"] is True


def test_status_only_display_overlay_does_not_hide_current_best_resources(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
        {
          "row_results": [
            {"family":"hh","case_id":"hh_L2_nph2_clean_strong","method":"SNAKE","threshold":0.0002,"abs_delta_e":0.0002217,"threshold_status":"not_reached","cost_included":false,"resource_display_allowed":true,"compiled_resource_validation_status":"ok_current_best_display","method_cost_semantics":"current_best_diagnostic_not_first_hit","count_2q":243,"depth_2q":200,"circuit_depth":649,"S_alg":2250}
          ]
        }
        """,
        encoding="utf-8",
    )
    overlay = tmp_path / "approved_overlay.json"
    overlay.write_text(
        """
        {
          "schema":"paper_i_display_cell_overlay_v1",
          "threshold":0.0002,
          "source_label":"approved_pdf_text_recovery_test",
          "row_results": [
            {"family":"hh","case_id":"hh_L2_nph2_clean_strong","method":"SNAKE","regime":"strong","display_delta_e":"NR","display_infidelity":"n/a","display_n_2q":"NR","display_d_2q":"NR","display_d_circ":"NR","display_s_alg":"NR"}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4, display_overlay_json=overlay)

    assert rows[("hh", "SNAKE", "strong")] == TableCell("2.17e-05", "--", "243", "200", "649", "2250")
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    hh_snake = by_key[("hh", "SNAKE", "strong")]
    assert hh_snake["source"] is None
    assert hh_snake["display_overlay_role"] is None
    assert hh_snake["resource_display_allowed"] is True


def test_load_rows_reports_skips_and_duplicate_keys(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
        {
          "row_results": [
            {"family":"spinless_tv","case_id":"spinless_tv_L2_clean_weak","method":"HEA VQE","threshold":0.0002,"abs_delta_e":0.0,"threshold_status":"ok_terminal_only_method","cost_included":true,"resource_display_allowed":true,"compiled_resource_validation_status":"ok","first_hit_cost_source_kind":"qiskit_compiled_terminal_only_fixed_ansatz","count_2q":2,"depth_2q":2,"circuit_depth":6},
            {"family":"spinless_tv","case_id":"spinless_tv_L2_clean_weak","method":"HEA VQE","threshold":0.0002,"abs_delta_e":0.0,"threshold_status":"ok_terminal_only_method","cost_included":true,"resource_display_allowed":true,"compiled_resource_validation_status":"ok","first_hit_cost_source_kind":"qiskit_compiled_terminal_only_fixed_ansatz","count_2q":4,"depth_2q":3,"circuit_depth":7},
            {"family":"spinless_tv","case_id":"spinless_tv_L2_clean_weak","threshold":0.0002,"abs_delta_e":0.0}
          ]
        }
        """
    )
    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    assert rows[("spinless_tv", "HEA VQE", "weak")].n_2q == "4"
    assert diagnostics.duplicate_keys == ["spinless_tv|HEA VQE|weak"]
    assert len(diagnostics.skipped_rows) == 1


def test_load_rows_audits_threshold_mismatches_and_harmonic_kerr_posgeo(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
        {
          "target_profile":"paper_i_phys_v1",
          "thresholds":[0.0002],
          "row_results": [
            {"family":"harmonic_kerr_chain","case_id":"harmonic_kerr_chain_L2_clean_weak","method":"Pos-Geo-ADAPT","threshold":0.000001,"abs_delta_e":0.0,"threshold_status":"ok_native_first_hit","cost_included":true,"resource_display_allowed":true,"compiled_resource_validation_status":"ok","first_hit_cost_source_kind":"qiskit_compiled_first_hit_ansatz_circuit","count_2q":2,"depth_2q":2,"circuit_depth":6},
            {"family":"harmonic_kerr_chain","case_id":"harmonic_kerr_chain_L2_clean_strong","method":"Pos-Geo-ADAPT","threshold":0.0002,"abs_delta_e":0.0,"threshold_status":"ok_native_first_hit","cost_included":true,"resource_display_allowed":true,"compiled_resource_validation_status":"ok","first_hit_cost_source_kind":"qiskit_compiled_first_hit_ansatz_circuit","count_2q":4,"depth_2q":3,"circuit_depth":7}
          ]
        }
        """
    )
    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    assert ("harmonic_kerr_chain", "Pos-Geo-ADAPT", "weak") not in rows
    assert diagnostics.threshold_mismatch_rows == ["row[0]: threshold=1e-06"]
    finding = diagnostics.special_findings["harmonic_kerr_weak_pos_geo"]
    assert finding["status"] == "missing"
    assert finding["valid_clean_ok_native_first_hit"] is False
    assert finding["near_miss_count"] == 1
    assert "Harmonic/Kerr weak Pos-Geo" in render_diagnostics(diagnostics)
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    assert audit["schema"] == "paper_i_fixed_accuracy_table_audit_v1"
    assert audit["special_findings"]["harmonic_kerr_weak_pos_geo"]["near_miss_count"] == 1


def test_duplicate_resolution_prefers_clean_valid_hit_over_later_not_reached(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        """
        {
          "row_results": [
            {"family":"hubbard","case_id":"hubbard_L2_clean_weak","method":"Pos-Geo-ADAPT","threshold":0.0002,"abs_delta_e":0.0,"threshold_status":"ok_native_first_hit","cost_included":true,"resource_display_allowed":true,"compiled_resource_validation_status":"ok","first_hit_cost_source_kind":"qiskit_compiled_first_hit_ansatz_circuit","count_2q":12,"depth_2q":8,"circuit_depth":20},
            {"family":"hubbard","case_id":"hubbard_L2_clean_weak","method":"Pos-Geo-ADAPT","threshold":0.0002,"abs_delta_e":0.1,"threshold_status":"not_reached","cost_included":false,"count_2q":999}
          ]
        }
        """
    )
    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    assert rows[("hubbard", "Pos-Geo-ADAPT", "weak")].n_2q == "12"
    assert diagnostics.duplicate_resolutions[0]["chosen"] == "existing"


def test_audit_manifest_carries_item6_handoff_fields_for_phonon_and_snake_rows(tmp_path: Path):
    payload_path = tmp_path / "generic_static_single.json"
    payload_path.write_text('{"status":"completed"}\n', encoding="utf-8")
    snake_overlay = tmp_path / "live_snake_best_summary.json"
    snake_overlay.write_text('{"schema":"paper_i_live_snake_overlay_v1"}\n', encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        f"""
        {{
          "target_profile":"paper_i_phys_v1",
          "thresholds":[0.0002],
          "live_snake_overlay":{{"source_path":"{snake_overlay}"}},
          "row_results": [
            {{"family":"bose_hubbard","case_id":"bose_hubbard_L2_nph2_clean_weak","method":"Pos-Geo-ADAPT","threshold":0.0002,"abs_delta_e":0.01,"threshold_status":"not_reached","cost_included":false,"payload_path":"{payload_path}"}},
            {{"family":"bose_hubbard","case_id":"bose_hubbard_L2_nph2_clean_weak","method":"SNAKE","threshold":0.0002,"threshold_status":"running_no_completed_trial","complete_trial_count":0,"running_trial_count":1,"trial_count":1,"source_condor_job":"6635208.0","source":"live_condor_optuna_sqlite_current_best_v1"}}
          ]
        }}
        """,
        encoding="utf-8",
    )

    _rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}

    posgeo = by_key[("bose_hubbard", "Pos-Geo-ADAPT", "weak")]
    assert posgeo["case_id"] == "bose_hubbard_L2_nph2_clean_weak"
    assert posgeo["n_ph_work"] == 2
    assert posgeo["n_ph_ref"] == 4
    assert posgeo["paper_i_ladder_stage"] == "locked_known_cutoff"
    assert posgeo["source_payload_path"] == str(payload_path)
    assert posgeo["source_payload_missing_reason"] is None
    assert posgeo["eligible_for_escalation"] is False
    assert posgeo["next_stage"] is None
    assert posgeo["next_stage_case_id"] is None
    assert posgeo["eligibility_reason"] == "completed_not_reached_locked_cutoff_repair_or_settings_rerun"

    snake = by_key[("bose_hubbard", "SNAKE", "weak")]
    assert snake["cost_included"] is False
    assert snake["source_payload_path"] == str(snake_overlay)
    assert snake["source_payload_path_kind"] == "live_snake_overlay_summary"
    assert snake["snake_current_state"] == "running_no_completed_trial"
    assert snake["snake_running_state"] == "running_no_completed_trial"
    assert snake["snake_terminal_state"] == "not_terminal_running"
    assert snake["snake_not_reached_state"] == "no_completed_trial"
    assert snake["snake_complete_trial_count"] == 0
    assert snake["snake_running_trial_count"] == 1
    assert snake["snake_trial_count"] == 1
    assert snake["eligible_for_escalation"] is False
    assert snake["eligibility_reason"] == "running_not_terminal"

    finding = audit["special_findings"]["snake_audited_first_crossing_costs"]
    assert finding["audited_first_crossing_compiled_cost_summary"] == "0/20"
    assert finding["sidecar_status_counts"] == {"absent": 20}


def test_audit_ladder_parser_preserves_nondefault_phonon_stages(tmp_path: Path):
    payload_path = tmp_path / "generic_static_single.json"
    payload_path.write_text('{"status":"completed"}\n', encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        f"""
        {{
          "row_results": [
            {{"family":"bose_hubbard","case_id":"bose_hubbard_L2_nph3_clean_weak","method":"Pos-Geo-ADAPT","threshold":0.0002,"abs_delta_e":0.01,"threshold_status":"not_reached","cost_included":false,"payload_path":"{payload_path}"}},
            {{"family":"bose_hubbard","case_id":"bose_hubbard_L2_nph4_ref5_clean_strong","method":"Pos-Geo-ADAPT","threshold":0.0002,"abs_delta_e":0.02,"threshold_status":"not_reached","cost_included":false,"payload_path":"{payload_path}"}}
          ]
        }}
        """,
        encoding="utf-8",
    )

    _rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}

    nph3 = by_key[("bose_hubbard", "Pos-Geo-ADAPT", "weak")]
    assert nph3["n_ph_work"] == 3
    assert nph3["n_ph_ref"] == 4
    assert nph3["paper_i_ladder_stage"] == "locked_known_cutoff"
    assert nph3["eligible_for_escalation"] is False
    assert nph3["eligibility_reason"] == "completed_not_reached_locked_cutoff_repair_or_settings_rerun"

    nph4 = by_key[("bose_hubbard", "Pos-Geo-ADAPT", "strong")]
    assert nph4["n_ph_work"] == 4
    assert nph4["n_ph_ref"] == 5
    assert nph4["paper_i_ladder_stage"] == "locked_known_cutoff"
    assert nph4["eligible_for_escalation"] is False


def test_molecular_vibronic_clean_cells_are_real_actionable_missing(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"row_results": []}\n', encoding="utf-8")

    _rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    molecular = [
        item
        for item in audit["expected_cell_audits"]
        if item["expected_key"][0] == "molecular_vibronic_h2"
    ]

    assert len(molecular) == 14
    assert {item["status"] for item in molecular} == {"missing"}
    assert {item["case_id"] for item in molecular} == {
        "molecular_vibronic_h2_L2_nph1_clean_weak",
        "molecular_vibronic_h2_L2_nph1_clean_strong",
    }
    assert {item["n_ph_work"] for item in molecular} == {1}
    assert {item["n_ph_ref"] for item in molecular} == {4}
    assert {item["eligible_for_display"] for item in molecular} == {False}
    assert {item["eligible_for_rerun"] for item in molecular} == {True}
    assert audit["diagnostics"]["deferred_placeholder_count"] == 0
    assert audit["diagnostics"]["missing_expected_count"] == len(audit["expected_cell_audits"])


def test_supplemental_remote_outputs_fill_missing_completed_comparator_cells(tmp_path: Path):
    record_id = "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph2_clean_weak__static_pos_geo_adapt_vqe"
    result_dir = tmp_path / "remote_raw_outputs" / "paper_i_clean_ladder_nph2_ref4_v1" / record_id / "result"
    result_dir.mkdir(parents=True)
    (result_dir / "generic_static_single.json").write_text(
        """
        {
          "result": {
            "family":"harmonic_kerr_chain",
            "case_id":"harmonic_kerr_chain_L2_nph2_clean_weak",
            "algorithm_id":"static_pos_geo_adapt_vqe",
            "method_id":"static_pos_geo_adapt_vqe",
            "abs_delta_e":0.00001,
            "S_alg":123,
            "compiled_count_2q_total":10,
            "compiled_depth_2q_total":10,
            "compiled_depth_total":20,
            "count_2q":10,
            "circuit_depth":20
          }
        }
        """,
        encoding="utf-8",
    )
    summary_dir = tmp_path / "fixed_accuracy_summary"
    summary_dir.mkdir()
    summary = summary_dir / "summary.json"
    summary.write_text('{"row_results": []}\n', encoding="utf-8")

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    cell = rows[("harmonic_kerr_chain", "Pos-Geo-ADAPT", "weak")]
    assert cell == TableCell("0", "--", "10", "10", "20", "123")

    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    hk = by_key[("harmonic_kerr_chain", "Pos-Geo-ADAPT", "weak")]
    assert hk["status"] == "terminal_upper_bound"
    assert hk["threshold_status"] == "terminal_upper_bound_missing_native_first_hit"
    assert hk["source_payload_path"].endswith("generic_static_single.json")
    assert audit["special_findings"]["harmonic_kerr_weak_pos_geo"]["status"] == "terminal_upper_bound"



def test_live_snake_overlay_first_crossing_compiled_ties_keep_costs_blank(tmp_path: Path):
    snake_overlay = tmp_path / "live_snake_best_summary.json"
    snake_overlay.write_text(
        """
        [
          {
            "record_id":"live_snake_current_best__bose_hubbard_L2_nph2_clean_weak",
            "benchmark_id":"bose_hubbard_L2_nph2_clean_weak",
            "case_id":"bose_hubbard_L2_nph2_clean_weak",
            "S_alg":97,
            "objective_score_components":{
              "compiled_two_qubit_count_tie":64,
              "compiled_depth_tie":237,
              "paper_i_first_crossing":{
                "reached":true,
                "status":"reached",
                "benchmark_id":"bose_hubbard_L2_nph2_clean_weak",
                "history_position_tau":5,
                "primary_error_at_crossing":0.00001
              }
            }
          }
        ]
        """,
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        f"""
        {{
          "target_profile":"paper_i_phys_v1",
          "thresholds":[0.0002],
          "live_snake_overlay":{{"source_path":"{snake_overlay}"}},
          "row_results": [
            {{"family":"bose_hubbard","case_id":"bose_hubbard_L2_nph2_clean_weak","benchmark_id":"bose_hubbard_L2_nph2_clean_weak","method":"SNAKE","threshold":0.0002,"abs_delta_e":0.00001,"threshold_status":"running_current_best_reached","complete_trial_count":1,"running_trial_count":1,"trial_count":2,"source":"live_condor_optuna_sqlite_current_best_v1"}}
          ]
        }}
        """,
        encoding="utf-8",
    )

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    cell = rows[("bose_hubbard", "SNAKE", "weak")]
    assert cell == TableCell("0", "--", "--", "--", "--", "--")

    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    snake = by_key[("bose_hubbard", "SNAKE", "weak")]
    assert snake["audited_first_crossing_compiled_cost_sidecar_present"] is False
    assert snake["audited_first_crossing_compiled_cost_sidecar_status"] == "missing"
    assert snake["cost_included"] is False
    assert snake["cell_state"] == "running-current-best-hit-missing-first-hit-cost"
    assert snake["resource_display_reason"] == "cost_included_false"
    assert snake["source_payload_path"] == str(snake_overlay)

    finding = audit["special_findings"]["snake_audited_first_crossing_costs"]
    assert finding["audited_first_crossing_compiled_cost_summary"] == "0/20"
    assert finding["sidecar_status_counts"] == {"absent": 19, "missing": 1}
    assert audit["cost_semantics"]["forbidden_numeric_resource_source_count"] == 0


def test_supplemental_remote_outputs_do_not_promote_snake_terminal_compiled_totals(tmp_path: Path):
    record_id = "static_table__bose_hubbard__bose_hubbard_L2_nph2_clean_weak__static_family_native_adapt_phase3"
    result_dir = tmp_path / "remote_raw_outputs" / "paper_i_clean_ladder_nph2_ref4_v1" / record_id / "result"
    result_dir.mkdir(parents=True)
    (result_dir / "generic_static_single.json").write_text(
        """
        {
          "algorithm_id":"static_family_native_adapt_phase3",
          "case_id":"bose_hubbard_L2_nph2_clean_weak",
          "family":"bose_hubbard",
          "status":"completed",
          "result": {
            "benchmark_id":"bose_hubbard_L2_nph2_clean_weak",
            "family":"bose_hubbard",
            "abs_delta_e":0.00000019,
            "S_alg":593,
            "compiled_count_2q_total":621,
            "compiled_depth_total":2213,
            "count_2q":621,
            "circuit_depth":2213,
            "paper_i_first_crossing":{
              "reached":true,
              "status":"reached",
              "benchmark_id":"bose_hubbard_L2_nph2_clean_weak",
              "history_position_tau":3,
              "primary_error_at_crossing":0.00000019,
              "tau_phys":0.002,
              "tau_tight":0.001
            }
          }
        }
        """,
        encoding="utf-8",
    )
    summary_dir = tmp_path / "fixed_accuracy_summary"
    summary_dir.mkdir()
    summary = summary_dir / "summary.json"
    summary.write_text(
        """
        {
          "row_results": [
            {"family":"bose_hubbard","case_id":"bose_hubbard_L2_nph2_clean_weak","method":"SNAKE","threshold":0.0002,"threshold_status":"running_no_completed_trial","complete_trial_count":0,"running_trial_count":1,"trial_count":1,"source":"live_condor_optuna_sqlite_current_best_v1"}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    cell = rows[("bose_hubbard", "SNAKE", "weak")]
    assert cell == TableCell("running", "--", "--", "--", "--", "--")

    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    snake = by_key[("bose_hubbard", "SNAKE", "weak")]
    assert snake["threshold_status"] == "running_no_completed_trial"
    assert snake["audited_first_crossing_compiled_cost_sidecar_present"] is False
    assert snake["source_payload_path"] is None
    assert snake["status"] == "running"
    assert snake["cell_state"] == "running-no-completed-trial"
    assert audit["cost_semantics"]["synthetic_snake_sidecar_count"] == 0


def test_supplemental_remote_snake_not_reached_overrides_running_placeholder_status(tmp_path: Path):
    record_id = "static_table__hh__hh_L2_nph2_clean_strong__static_family_native_adapt_phase3"
    result_dir = tmp_path / "remote_raw_outputs" / "paper_i_clean_ladder_nph2_ref4_v1" / record_id / "result"
    result_dir.mkdir(parents=True)
    (result_dir / "generic_static_single.json").write_text(
        """
        {
          "algorithm_id":"static_family_native_adapt_phase3",
          "case_id":"hh_L2_nph2_clean_strong",
          "family":"hh",
          "status":"completed",
          "result": {
            "benchmark_id":"hh_L2_nph2_clean_strong",
            "family":"hh",
            "abs_delta_e":0.0002217347,
            "S_alg":2250,
            "compiled_count_2q_total":243,
            "compiled_depth_2q_total":200,
            "compiled_depth_total":649,
            "paper_i_first_crossing":{
              "reached":true,
              "status":"reached",
              "benchmark_id":"hh_L2_nph2_clean_strong",
              "history_position_tau":5,
              "primary_error_at_crossing":0.001489,
              "tau_phys":0.002,
              "tau_tight":0.001
            }
          }
        }
        """,
        encoding="utf-8",
    )
    summary_dir = tmp_path / "fixed_accuracy_summary"
    summary_dir.mkdir()
    summary = summary_dir / "summary.json"
    summary.write_text(
        """
        {
          "row_results": [
            {"family":"hh","case_id":"hh_L2_nph2_clean_strong","method":"SNAKE","threshold":0.0002,"threshold_status":"running_no_completed_trial","complete_trial_count":0,"running_trial_count":1,"trial_count":1,"source":"live_condor_optuna_sqlite_current_best_v1"}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows, diagnostics = load_rows_with_diagnostics(summary, 2e-4)
    cell = rows[("hh", "SNAKE", "strong")]
    assert cell == TableCell("2.17e-05", "--", "243", "200", "649", "2250")

    audit = build_audit_manifest(source_pdf=None, summary_json=summary, threshold=2e-4, diagnostics=diagnostics)
    by_key = {tuple(item["expected_key"]): item for item in audit["expected_cell_audits"]}
    snake = by_key[("hh", "SNAKE", "strong")]
    assert snake["threshold_status"] == "not_reached"
    assert snake["audited_first_crossing_compiled_cost_sidecar_present"] is False
    assert snake["cell_state"] == "not-reached-with-final-ansatz-cost"
    assert snake["resource_display_allowed"] is True
    assert snake["cost_included"] is False
    assert audit["cost_semantics"]["synthetic_snake_sidecar_count"] == 0
