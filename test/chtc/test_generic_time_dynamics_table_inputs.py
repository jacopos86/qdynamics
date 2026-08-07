from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DYNAMICS_COARSE_TUNING_CLASSES
from pipelines.time_dynamics.tables.table_lock_contract import validate_class_settings_lock_manifest
from chtc.time_dynamics_optuna.build_paper_ii_all_algorithm_class_calibration_inputs import (
    OUTPUT_STEM as PAPER_II_CLASS_CALIBRATION_OUTPUT_STEM,
    build_inputs as build_class_calibration_inputs,
    validate_inputs_manifest as validate_class_calibration_inputs_manifest,
)
from chtc.generic_time_dynamics_table.build_paper_ii_snake_recovery_manifest import (
    EXPECTED_FAMILIES as PAPER_II_RECOVERY_EXPECTED_FAMILIES,
    EXPECTED_SNAKE_CASE_COUNT,
    PARITY_CORRECTNESS_MATRIX as PAPER_II_PARITY_CORRECTNESS_MATRIX,
    RECOVERY_MANIFEST as PAPER_II_SNAKE_RECOVERY_MANIFEST,
    SCHEMA as PAPER_II_SNAKE_RECOVERY_SCHEMA,
    RecoveryManifestValidationError,
    build_recovery_manifest,
    validate_recovery_manifest,
)
from chtc.generic_time_dynamics_table.build_paper_ii_hh_seed_track_inputs import (
    CONTROLLER_IDS as PAPER_II_HH_CONTROLLER_IDS,
    FULL_IDS as PAPER_II_HH_FULL_IDS,
    RECORDS_TSV as PAPER_II_HH_RECORDS_TSV,
    SEED_LEDGER as PAPER_II_HH_SEED_LEDGER,
    SMOKE_IDS as PAPER_II_HH_SMOKE_IDS,
    VISIBLE_IDS as PAPER_II_HH_VISIBLE_IDS,
    build_inputs as build_hh_seed_track_inputs,
)
from chtc.generic_time_dynamics_table.paper_ii_seed_track_common import (
    DRIVES as PAPER_II_HH_DRIVES,
    FULL_BENCHMARK_ALGORITHMS as PAPER_II_HH_FULL_BENCHMARK_ALGORITHMS,
    HH_REGIMES as PAPER_II_HH_REGIMES,
    SEED_TRACKS_BY_ID as PAPER_II_HH_SEED_TRACKS_BY_ID,
    SEED_TRACK_SPECS as PAPER_II_HH_SEED_TRACK_SPECS,
    SOURCE_REGISTRY_SCHEMA as PAPER_II_HH_SOURCE_REGISTRY_SCHEMA,
    VISIBLE_ALGORITHMS as PAPER_II_HH_VISIBLE_ALGORITHMS,
    SeedTrackValidationError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "chtc" / "generic_time_dynamics_table" / "input"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def _hh_fixture_state(nq: int = 1) -> dict[str, object]:
    return {
        "amplitude_cutoff": 0.0,
        "amplitudes_qn_to_q0": {"0" * int(nq): {"re": 1.0, "im": 0.0}},
        "handoff_state_kind": "prepared_state",
        "norm": 1.0,
        "nq_total": int(nq),
        "source": "unit_fixture",
    }


def _hh_fixture_seed_payload(regime, track_spec) -> dict[str, object]:
    return {
        "family": "hh",
        "case_id": regime.static_case_id,
        "algorithm_id": track_spec.required_static_algorithm_id,
        "settings": {
            "problem": "hh",
            "L": 2,
            "t": 1.0,
            "u": regime.u_over_t,
            "omega0": 1.0,
            "g_ep": math.sqrt(regime.lambda_ep / 2.0),
            "n_ph_max": regime.n_ph_work,
            "boson_encoding": "binary",
            "boundary": "open",
            "ordering": "blocked",
            "n_fermions": 2,
        },
        "adapt_vqe": {
            "algorithm_id": track_spec.required_static_algorithm_id,
            "ansatz_depth": 1,
            "operators": ["fixture_operator"],
            "optimal_point": [0.125],
            "num_parameters": 1,
            "abs_delta_e": 1.0e-4,
            "pool_type": "full_meta",
        },
        "initial_state": _hh_fixture_state(),
        "ansatz_input_state": _hh_fixture_state(),
        "ground_state": {"exact_energy": -1.0, "exact_energy_source": "unit_fixture"},
    }


def _write_hh_seed_registry_fixture(tmp_path: Path, *, mutate=None) -> Path:
    sources: list[dict[str, str]] = []
    source_root = tmp_path / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    for regime in PAPER_II_HH_REGIMES:
        for track_spec in PAPER_II_HH_SEED_TRACK_SPECS:
            payload = _hh_fixture_seed_payload(regime, track_spec)
            rel_source = Path("sources") / f"{regime.regime_id}_{track_spec.seed_track}.json"
            entry = {
                "hh_regime_id": regime.regime_id,
                "hh_static_case_id": regime.static_case_id,
                "seed_track": track_spec.seed_track,
                "static_algorithm_id": track_spec.required_static_algorithm_id,
                "source_artifact_json": str(rel_source),
                "source_record_id": f"unit_{regime.regime_id}_{track_spec.seed_track}",
            }
            if mutate is not None:
                mutated = mutate(regime, track_spec, payload, entry)
                if mutated is not None:
                    payload, entry = mutated
                    rel_source = Path(str(entry["source_artifact_json"]))
            source_path = tmp_path / rel_source
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            sources.append(entry)
    registry = tmp_path / "paper_ii_hh_static_seed_sources_v1.json"
    registry.write_text(
        json.dumps({"schema": PAPER_II_HH_SOURCE_REGISTRY_SCHEMA, "sources": sources}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry


def test_paper_ii_hh_seed_track_contract_is_strict_revised_hh_surface() -> None:
    assert [track.seed_track for track in PAPER_II_HH_SEED_TRACK_SPECS] == ["geo", "snake", "append"]
    assert "posgeo" not in PAPER_II_HH_SEED_TRACKS_BY_ID
    assert [regime.regime_id for regime in PAPER_II_HH_REGIMES] == [
        "weak_weak",
        "strong_weak",
        "weak_strong",
        "strong_strong",
    ]
    assert [(regime.n_ph_work, regime.n_ph_ref) for regime in PAPER_II_HH_REGIMES] == [
        (2, 5),
        (2, 5),
        (4, 7),
        (4, 7),
    ]
    assert PAPER_II_HH_DRIVES == (("A0p2", 0.2), ("A0p6", 0.6))
    assert PAPER_II_HH_VISIBLE_ALGORITHMS == (
        "dyn_controller_full",
        "dyn_fixed_mclachlan",
        "dyn_adaptive_pvqd",
        "dyn_avqds_t",
    )


def test_paper_ii_hh_seed_track_builder_emits_24_cases_and_track_separated_records(tmp_path: Path) -> None:
    registry = _write_hh_seed_registry_fixture(tmp_path)

    manifest = build_hh_seed_track_inputs(root=tmp_path, source_registry_path=registry, require_runtime_dry_load=False)

    assert manifest["schema"] == "paper_ii_hh_seed_track_case_manifest_v1"
    assert manifest["case_count"] == 24
    assert manifest["seed_source_count"] == 12
    assert manifest["seed_tracks"] == ["geo", "snake", "append"]
    assert manifest["hh_regimes"] == ["weak_weak", "strong_weak", "weak_strong", "strong_strong"]
    assert all("posgeo" not in case["case_id"] for case in manifest["cases"])
    assert len({case["case_id"] for case in manifest["cases"]}) == 24

    for case in manifest["cases"]:
        seed_lock = case["metadata"]["seed_lock"]
        assert case["family"] == "hh"
        assert case["table_class"] == "hubbard_holstein"
        assert case["tuning_class"] == "hybrid"
        assert case["loader_mode"] == "replay_family"
        assert case["t_final"] == 8.0
        assert case["num_times"] == 321
        assert seed_lock["seed_track"] in {"geo", "snake", "append"}
        assert seed_lock["static_algorithm_id"] == PAPER_II_HH_SEED_TRACKS_BY_ID[seed_lock["seed_track"]].required_static_algorithm_id
        assert seed_lock["hh_regime_id"] in {regime.regime_id for regime in PAPER_II_HH_REGIMES}
        assert seed_lock["same_seed_comparator_group_id"].endswith("_t8_dt321_same_seed_v1")
        assert seed_lock["runtime_loadability_status"] == "payload_fields_present_dry_load_not_run"
        assert seed_lock["latest_phase3_source_artifact_missing_locally"] is False

    rows = _rows(tmp_path / PAPER_II_HH_RECORDS_TSV)
    visible_ids = [line.strip() for line in (tmp_path / PAPER_II_HH_VISIBLE_IDS).read_text(encoding="utf-8").splitlines() if line.strip()]
    full_ids = [line.strip() for line in (tmp_path / PAPER_II_HH_FULL_IDS).read_text(encoding="utf-8").splitlines() if line.strip()]
    controller_ids = [line.strip() for line in (tmp_path / PAPER_II_HH_CONTROLLER_IDS).read_text(encoding="utf-8").splitlines() if line.strip()]
    smoke_ids = [line.strip() for line in (tmp_path / PAPER_II_HH_SMOKE_IDS).read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 24 * (len(PAPER_II_HH_FULL_BENCHMARK_ALGORITHMS) + 1)
    assert len(full_ids) == len(rows)
    assert len(visible_ids) == 24 * len(PAPER_II_HH_VISIBLE_ALGORITHMS)
    assert len(controller_ids) == 24
    assert len(smoke_ids) == 4
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert {row["seed_track"] for row in rows} == {"geo", "snake", "append"}
    assert {row["visible_table_method"] for row in rows} == {"0", "1"}
    assert all(row["case_manifest"] == "chtc/generic_time_dynamics_table/input/paper_ii_hh_seed_tracks_cases_v1.json" for row in rows)

    ledger = json.loads((tmp_path / PAPER_II_HH_SEED_LEDGER).read_text(encoding="utf-8"))
    assert ledger["schema"] == "paper_ii_hh_seed_track_ledger_v1"
    assert ledger["seed_source_count"] == 12
    assert {entry["seed_track"] for entry in ledger["sources"]} == {"geo", "snake", "append"}


def test_paper_ii_hh_seed_track_builder_can_emit_filtered_weak_surfaces(tmp_path: Path) -> None:
    registry = _write_hh_seed_registry_fixture(tmp_path)

    manifest = build_hh_seed_track_inputs(
        root=tmp_path,
        source_registry_path=registry,
        regime_ids=["weak_weak", "strong_weak"],
        require_runtime_dry_load=False,
    )

    assert manifest["case_count"] == 12
    assert manifest["seed_source_count"] == 6
    assert manifest["hh_regimes"] == ["weak_weak", "strong_weak"]
    assert {case["metadata"]["seed_lock"]["hh_regime_id"] for case in manifest["cases"]} == {
        "weak_weak",
        "strong_weak",
    }

    rows = _rows(tmp_path / PAPER_II_HH_RECORDS_TSV)
    visible_ids = [
        line.strip()
        for line in (tmp_path / PAPER_II_HH_VISIBLE_IDS).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    full_ids = [
        line.strip()
        for line in (tmp_path / PAPER_II_HH_FULL_IDS).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 12 * (len(PAPER_II_HH_FULL_BENCHMARK_ALGORITHMS) + 1)
    assert len(visible_ids) == 12 * len(PAPER_II_HH_VISIBLE_ALGORITHMS)
    assert len(full_ids) == len(rows)


def test_paper_ii_hh_seed_track_builder_rejects_posgeo_for_geo_track(tmp_path: Path) -> None:
    def mutate(regime, track_spec, payload, entry):
        if regime.regime_id == "weak_weak" and track_spec.seed_track == "geo":
            entry = dict(entry)
            entry["source_artifact_json"] = "sources/posgeo_bad_geo_seed.json"
            return payload, entry
        return None

    registry = _write_hh_seed_registry_fixture(tmp_path, mutate=mutate)

    with pytest.raises(SeedTrackValidationError, match="forbidden token"):
        build_hh_seed_track_inputs(
            root=tmp_path,
            source_registry_path=registry,
            write_submit_files=False,
            require_runtime_dry_load=False,
        )


def test_paper_ii_hh_seed_track_builder_rejects_registry_identity_override(tmp_path: Path) -> None:
    def mutate(regime, track_spec, payload, entry):
        if regime.regime_id == "weak_weak" and track_spec.seed_track == "geo":
            payload = dict(payload)
            payload.pop("algorithm_id", None)
            payload["adapt_vqe"] = dict(payload["adapt_vqe"])
            payload["adapt_vqe"].pop("algorithm_id", None)
            # Registry still declares the right ID; validation must inspect the payload.
            return payload, entry
        return None

    registry = _write_hh_seed_registry_fixture(tmp_path, mutate=mutate)

    with pytest.raises(SeedTrackValidationError, match="payload algorithm mismatch"):
        build_hh_seed_track_inputs(
            root=tmp_path,
            source_registry_path=registry,
            write_submit_files=False,
            require_runtime_dry_load=False,
        )


def test_paper_ii_hh_seed_track_builder_rejects_wrapper_only_hh_seed(tmp_path: Path) -> None:
    def mutate(regime, track_spec, payload, entry):
        if regime.regime_id == "weak_weak" and track_spec.seed_track == "append":
            payload = dict(payload)
            payload.pop("ground_state", None)
            return payload, entry
        return None

    registry = _write_hh_seed_registry_fixture(tmp_path, mutate=mutate)

    with pytest.raises(SeedTrackValidationError, match="ground_state"):
        build_hh_seed_track_inputs(
            root=tmp_path,
            source_registry_path=registry,
            write_submit_files=False,
            require_runtime_dry_load=False,
        )


def test_paper_ii_class_tuned_case_manifest_has_exact_three_classes_and_20_cases() -> None:
    payload = json.loads((INPUT / "paper_ii_class_tuned_cases_v1.json").read_text(encoding="utf-8"))
    cases = payload["cases"]

    assert payload["schema"] == "paper_ii_class_tuned_case_manifest_v1"
    assert payload["canonical_controller_policy_classes"] == list(DYNAMICS_COARSE_TUNING_CLASSES)
    assert len(cases) == 20
    assert sorted({case["tuning_class"] for case in cases}) == sorted(DYNAMICS_COARSE_TUNING_CLASSES)
    assert any(case["family"] == "harmonic_kerr_chain" and case["tuning_class"] == "bosonic" for case in cases)
    assert any(case["family"] == "molecular_vibronic_h2" and case["tuning_class"] == "hybrid" for case in cases)
    assert all(case["metadata"]["static_scaffold_scope"] == "benchmark_point" for case in cases)
    assert all(case["metadata"]["controller_settings_scope"] == "coarse_hamiltonian_class" for case in cases)


def test_paper_ii_class_tuned_smoke_records_cover_all_classes_and_locked_controller() -> None:
    rows = _rows(INPUT / "paper_ii_class_tuned_smoke_records.tsv")

    assert sorted({row["tuning_class"] for row in rows}) == sorted(DYNAMICS_COARSE_TUNING_CLASSES)
    assert {row["kind"] for row in rows} == {"benchmark", "ablation"}
    assert any(row["algorithm_id"] == "dyn_controller_ablation_matrix" for row in rows)
    assert all(row["case_manifest"] == "chtc/generic_time_dynamics_table/input/paper_ii_class_tuned_cases_v1.json" for row in rows)


def test_smoke_class_settings_lock_is_exact_three_strict_controller_classes() -> None:
    payload = json.loads((INPUT / "class_settings" / "paper_ii_class_settings_smoke_lock_v1.json").read_text(encoding="utf-8"))

    validation = validate_class_settings_lock_manifest(payload, require_exact_controller_classes=True)

    assert validation["canonical_controller_classes"] == sorted(DYNAMICS_COARSE_TUNING_CLASSES)
    assert payload["manifest_role"] == "local_smoke_only_not_paper_evidence"


def test_submit_files_require_locked_class_settings_and_apptainer_passes_env() -> None:
    smoke_submit = (REPO_ROOT / "chtc" / "generic_time_dynamics_table" / "submit_paper_ii_class_tuned_smoke.sub").read_text(encoding="utf-8")
    full_submit = (REPO_ROOT / "chtc" / "generic_time_dynamics_table" / "submit_paper_ii_class_tuned_full.sub").read_text(encoding="utf-8")
    wrapper = (REPO_ROOT / "chtc" / "generic_time_dynamics_table" / "run_task_apptainer.sh").read_text(encoding="utf-8")

    for text in (smoke_submit, full_submit):
        assert "GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1" in text
        assert "GENERIC_TD_CLASS_SETTINGS_MANIFEST=" in text
    assert "paper_ii_class_settings_smoke_lock_v1.json" in smoke_submit
    assert "paper_ii_class_settings_lock_v1.json" in full_submit
    assert "--cleanenv" in wrapper
    assert "GENERIC_TD_CLASS_SETTINGS_MANIFEST" in wrapper
    assert "GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS" in wrapper



def test_paper_ii_snake_recovery_manifest_selects_only_v2_snake_cases() -> None:
    payload = json.loads((REPO_ROOT / PAPER_II_SNAKE_RECOVERY_MANIFEST).read_text(encoding="utf-8"))
    source = json.loads((INPUT / "paper_ii_seed_tracks_cases_v2.json").read_text(encoding="utf-8"))
    source_snake = {
        case["case_id"]: case
        for case in source["cases"]
        if case["metadata"]["seed_lock"]["seed_track"] == "snake"
    }

    validation = validate_recovery_manifest(payload)
    selected = payload["selected_cases"]
    excluded = payload["excluded_cases"]

    assert payload["schema"] == PAPER_II_SNAKE_RECOVERY_SCHEMA
    assert validation["status"] == "passed"
    assert payload["manifest_role"] == "diagnostic_freeze_and_snake_recovery_contract_only_not_paper_evidence"
    assert payload["selected_case_count"] == EXPECTED_SNAKE_CASE_COUNT == 20
    assert len(selected) == 20
    assert len(excluded) == 18
    assert {case["seed_track"] for case in selected} == {"snake"}
    assert all("posgeo" not in case["case_id"].lower() for case in selected)
    assert {case["family"] for case in selected} == set(PAPER_II_RECOVERY_EXPECTED_FAMILIES)
    assert all(case["seed_track"] == "posgeo" for case in excluded)

    for family in PAPER_II_RECOVERY_EXPECTED_FAMILIES:
        assert sorted(case["drive_metadata"]["A"] for case in selected if case["family"] == family) == [0.2, 0.6]

    for case in selected:
        source_case = source_snake[case["case_id"]]
        source_seed_lock = source_case["metadata"]["seed_lock"]
        assert case["seed_artifact_sha256"] == source_seed_lock["seed_artifact_sha256"]
        assert case["same_seed_comparator_group_id"] == source_seed_lock["same_seed_comparator_group_id"]
        assert case["artifact_json"] == source_case["artifact_json"]
        assert case["t_final"] == source_case["t_final"]
        assert case["num_times"] == source_case["num_times"]
        assert case["diagnostic_exact_reference_policy"] == "benchmark_exact_reporting_only"
        assert case["observable_policy"]["qpu_faithful_controller_data_contract"] == (
            "measurement_compatible_prepared_state_observables_only"
        )
        assert case["compile_target"]["class_settings_manifest"] == (
            "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"
        )

    assert payload["source_hashes"] == {
        "source_cases_manifest_sha256": hashlib.sha256((INPUT / "paper_ii_seed_tracks_cases_v2.json").read_bytes()).hexdigest(),
        "source_seed_ledger_sha256": hashlib.sha256((INPUT / "paper_ii_seed_tracks_seed_ledger_v2.json").read_bytes()).hexdigest(),
    }


def test_paper_ii_snake_recovery_manifest_freezes_current_table_i_as_diagnostic_only() -> None:
    payload = json.loads((REPO_ROOT / PAPER_II_SNAKE_RECOVERY_MANIFEST).read_text(encoding="utf-8"))

    assert payload["selection_policy"]["seed_regeneration_policy"] == (
        "forbidden_preserve_v2_seed_artifact_hashes_exactly"
    )
    assert payload["selection_policy"]["seed_substitution_policy"] == (
        "forbidden_no_staged_fallback_pending_or_recovery_seed_sources"
    )
    assert payload["validation_contract"]["future_evidence_root_policy"] == (
        "create_new_root_never_overwrite_diagnostic_baseline_sources"
    )
    parity = payload["parity_correctness_requirements"]
    assert parity["schema"] == "paper_ii_table_i_parity_correctness_matrix_v1"
    assert parity["status"] == "specified_work_items_4_6_sidecar_contracts_implemented_no_recovery_rerun_evidence_yet"
    assert parity["matrix"] == [dict(item) for item in PAPER_II_PARITY_CORRECTNESS_MATRIX]
    assert parity["missing_failed_or_not_applicable_required_check_blocks_table_i_use"] is True
    matrix_by_algorithm = {}
    for item in parity["matrix"]:
        matrix_by_algorithm.setdefault(item["algorithm_id"], set()).add(item["sidecar_name"])
    assert matrix_by_algorithm["dyn_fixed_mclachlan"] == {"qiskit_parity.json", "mclachlan_correctness.json"}
    assert matrix_by_algorithm["dyn_avqds"] == {"avqds_correctness.json"}
    assert matrix_by_algorithm["dyn_avqds_t"] == {"avqds_t_correctness.json"}
    assert payload["missing_evidence_fields"][0]["status"] == (
        "requirements_specified_sidecar_contracts_implemented_no_recovery_rerun_evidence_yet"
    )
    assert payload["missing_evidence_fields"][0]["blocks_paper_facing_use"] is True
    calibration = payload["class_calibration_requirements"]
    assert calibration["status"] == "candidate_contract_implemented_not_promoted_no_recovery_rerun_evidence_yet"
    assert calibration["candidate_only_not_promoted"] is True
    assert len(calibration["required_algorithm_settings"]) == 8
    assert payload["class_settings_sources"]["all_algorithm_candidate_lock"].endswith(
        "paper_ii_all_algorithm_class_settings_candidate_lock_v1.json"
    )
    assert payload["class_settings_sources"]["all_algorithm_candidate_lock_status"] == (
        "candidate_not_promoted_not_paper_evidence"
    )
    assert payload["calibration_scoring_selection_rules"]["promotion_policy"].startswith("explicit_user_approval")
    gate_by_id = {gate["gate_id"]: gate for gate in payload["recovery_validation_gates"]}
    assert gate_by_id["class_level_settings_for_checkpoint_fixed_and_all_comparators"]["severity"] == "ERROR"
    assert gate_by_id["missing_epsilon_spec_shots_or_fidelity_marked_missing_evidence"]["severity"] == "WARN"
    family_gate = payload["family_repair_rerun_eligibility_gate"]
    assert family_gate["status"] == "blocked_until_recovery_parity_calibration_gates_pass_and_user_approves"
    assert family_gate["family_repair_can_resume"] is False
    assert family_gate["rerun_eligibility"] == "blocked"
    assert family_gate["prior_family_bugfix_plan_role"] == "later_stage_dependency_not_immediate_next_step"
    assert family_gate["repair_order"] == [
        "observable_and_primary_density_semantics",
        "drive_and_operator_semantics",
        "frozen_no_dynamics_telemetry",
        "frozen_no_dynamics_tranche",
        "wrong_observable_tranche",
        "hh_damping_undershoot_tranche",
    ]
    assert family_gate["tranche_review_policy"]["user_visual_approval_required_before_any_rerun_escalation"] is True
    assert family_gate["rerun_preservation_policy"]["rerun_checkpoint_fixed_and_every_comparator_for_affected_same_seed_group"] is True
    closeout_gate = payload["final_aggregation_promotion_closeout_gate"]
    assert closeout_gate["status"].startswith("blocked_pending_user_approved_repaired_reruns")
    assert closeout_gate["final_aggregation_allowed"] is False
    assert closeout_gate["table_promotion_allowed"] is False
    assert "missing_evidence_reporting" in closeout_gate["regression_coverage_required"]
    assert "manuscript_tex_edits" in closeout_gate["forbidden_without_separate_explicit_workflow"]
    missing_by_field = {item["field"]: item for item in payload["missing_evidence_fields"]}
    for field in (
        "family_repair_tranche_pdfs_user_approved",
        "post_fetch_audit_no_blocking_errors",
        "final_aggregation_user_approval",
        "table_promotion_user_approval",
    ):
        assert missing_by_field[field]["blocks_paper_facing_use"] is True
    assert all("diagnostic" in source["status"] for source in payload["diagnostic_baseline_sources"])
    assert all(
        source["mutation_policy"] in {"do_not_edit_in_recovery_manifest_wave", "do_not_overwrite_or_reaggregate"}
        for source in payload["diagnostic_baseline_sources"]
    )
    assert {algorithm["algorithm_id"] for algorithm in payload["algorithms"]} == {
        "dyn_controller_full",
        "dyn_fixed_mclachlan",
        "dyn_product_formula_envelope",
        "dyn_qdrift",
        "dyn_fixed_pvqd",
        "dyn_adaptive_pvqd",
        "dyn_avqds",
        "dyn_avqds_t",
    }


def test_paper_ii_snake_recovery_builder_matches_checked_in_manifest() -> None:
    checked_in = json.loads((REPO_ROOT / PAPER_II_SNAKE_RECOVERY_MANIFEST).read_text(encoding="utf-8"))

    assert build_recovery_manifest() == checked_in


def test_paper_ii_all_algorithm_candidate_lock_is_complete_and_not_promoted() -> None:
    payload = json.loads(
        (INPUT / "class_settings" / "paper_ii_all_algorithm_class_settings_candidate_lock_v1.json").read_text(
            encoding="utf-8"
        )
    )

    validation = validate_class_settings_lock_manifest(payload, require_all_table_i_algorithm_classes=True)

    assert payload["lock_status"] == "candidate_not_promoted"
    assert payload["candidate_only_not_promoted"] is True
    assert validation["required_algorithm_class_entry_count"] == 24
    assert validation["candidate_only_entry_count"] == 24
    assert {entry["settings_kind"] for entry in payload["settings"]} == {"controller", "mclachlan", "comparator"}
    assert all(entry["class_tuned_result_locked"] is False for entry in payload["settings"])
    assert all(entry["promotion_status"] == "candidate_not_promoted_user_approval_required" for entry in payload["settings"])


def test_paper_ii_all_algorithm_class_calibration_smoke_submit_surface_is_diagnostic_only() -> None:
    optuna_input = REPO_ROOT / "chtc" / "time_dynamics_optuna" / "input"
    submit = (REPO_ROOT / "chtc" / "time_dynamics_optuna" / "submit_paper_ii_all_algorithm_class_calibration_v1_smoke.sub").read_text(
        encoding="utf-8"
    )
    upload = (REPO_ROOT / "chtc" / "time_dynamics_optuna" / "upload_submit_chtc.sh").read_text(encoding="utf-8")
    smoke_ids = [
        line.strip()
        for line in (optuna_input / "paper_ii_all_algorithm_class_calibration_v1_smoke_record_ids.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    full_ids = [
        line.strip()
        for line in (optuna_input / "paper_ii_all_algorithm_class_calibration_v1_record_ids.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(smoke_ids) == 24
    assert len(full_ids) == 160
    assert "candidate/diagnostic only" in submit
    assert "Not promoted, not paper-facing" in submit
    assert "paper_ii_all_algorithm_class_calibration_v1_smoke_records.tsv" in submit
    assert "paper_ii_all_algorithm_class_calibration_v1_smoke_record_ids.txt" in submit
    assert "paper_ii_all_algorithm_class_calibration_v1_records.tsv" not in submit
    assert "chtc/generic_time_dynamics_table" in submit
    assert "GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1" not in submit
    assert "+JobBatchName = \"holstein-paper-ii-all-algo-class-cal-v1-smoke-diagnostic\"" in submit
    assert "paper-ii-all-algorithm-class-calibration-v1-smoke)" in upload
    assert "paper-ii-all-algorithm-class-calibration-v1-full)" in upload
    assert "--no-stage" in upload
    assert "submit_paper_ii_all_algorithm_class_calibration_v1_smoke.sub" in upload
    assert "submit_paper_ii_all_algorithm_class_calibration_v1.sub" in upload


def test_paper_ii_all_algorithm_class_calibration_inputs_are_snake_only_candidate_contract() -> None:
    manifest_path = INPUT.parents[1] / "time_dynamics_optuna" / "input" / f"{PAPER_II_CLASS_CALIBRATION_OUTPUT_STEM}_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    validation = validate_class_calibration_inputs_manifest(payload)

    assert validation["status"] == "passed"
    assert payload["schema"] == "paper_ii_all_algorithm_class_calibration_inputs_v1"
    assert payload["candidate_only_not_promoted"] is True
    assert payload["no_chtc_submission_performed"] is True
    assert payload["seed_track_filter"] == ["snake"]
    assert payload["case_count"] == 20
    assert payload["algorithm_count"] == 8
    assert payload["candidate_record_count"] == 160
    assert payload["smoke_record_count"] == 24
    assert build_class_calibration_inputs() == payload

    case_manifest = json.loads((REPO_ROOT / payload["case_manifest"]).read_text(encoding="utf-8"))
    assert case_manifest["schema"] == "paper_ii_all_algorithm_class_calibration_case_manifest_v1"
    assert case_manifest["case_count"] == 20
    assert all(
        case["metadata"]["controller_settings_scope"] == "coarse_hamiltonian_class"
        for case in case_manifest["cases"]
    )
    assert all(case["metadata"]["static_scaffold_scope"] == "benchmark_point" for case in case_manifest["cases"])
    assert all(case["metadata"]["paper_ii_table_lock"] is True for case in case_manifest["cases"])
    assert all(
        case["metadata"]["canonical_case_manifest_id"] == "paper_ii_seed_tracks_cases_v2"
        for case in case_manifest["cases"]
    )

    rows = _rows(REPO_ROOT / payload["candidate_records"])
    assert len(rows) == 160
    assert {row["seed_track"] for row in rows} == {"snake"}
    assert {row["candidate_only_not_promoted"] for row in rows} == {"1"}
    assert {row["require_algorithm_class_settings"] for row in rows} == {"1"}
    assert {row["require_parity_correctness_sidecars"] for row in rows} == {"1"}
    assert payload["candidate_records"] != "chtc/time_dynamics_optuna/input/records.tsv"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"manifest_role": "paper_facing"}), "diagnostic-only"),
        (
            lambda payload: payload["algorithms"][0].update({"parity_correctness_status": "passed"}),
            "algorithms must match the Work Items 4-6 parity/correctness contract",
        ),
        (
            lambda payload: payload["parity_correctness_requirements"].update({"matrix": []}),
            "parity/correctness matrix does not match",
        ),
        (
            lambda payload: payload["parity_correctness_requirements"]["matrix"][1].update(
                {"sidecar_name": "qiskit_parity.json_only"}
            ),
            "parity/correctness matrix does not match",
        ),
        (
            lambda payload: payload["parity_correctness_requirements"].update(
                {"missing_failed_or_not_applicable_required_check_blocks_table_i_use": False}
            ),
            "missing/failed/not-applicable correctness checks must block",
        ),
        (
            lambda payload: payload["class_settings_sources"].update(
                {"all_algorithm_candidate_lock": "chtc/generic_time_dynamics_table/input/class_settings/future.json"}
            ),
            "all_algorithm_candidate_lock path must match",
        ),
        (
            lambda payload: payload["class_calibration_requirements"].update(
                {"candidate_only_not_promoted": False}
            ),
            "candidate-only/not-promoted",
        ),
        (
            lambda payload: payload["family_repair_rerun_eligibility_gate"].update(
                {"family_repair_can_resume": True}
            ),
            "family repair/rerun eligibility gate changed",
        ),
        (
            lambda payload: payload["final_aggregation_promotion_closeout_gate"].update(
                {"final_aggregation_allowed": True}
            ),
            "final aggregation/promotion closeout gate changed",
        ),
        (
            lambda payload: payload["missing_evidence_fields"][2].update({"blocks_paper_facing_use": False}),
            "must block paper-facing use",
        ),
        (
            lambda payload: payload["source_hashes"].update({"source_cases_manifest_sha256": "bad"}),
            "source_hashes do not match",
        ),
        (
            lambda payload: payload["selected_cases"][0].update({"t_final": 9.0}),
            "selected_cases do not exactly match",
        ),
    ],
)
def test_paper_ii_snake_recovery_validation_rejects_scope_or_source_drift(mutation, message: str) -> None:
    payload = build_recovery_manifest()
    mutated = copy.deepcopy(payload)
    mutation(mutated)

    with pytest.raises(RecoveryManifestValidationError, match=message):
        validate_recovery_manifest(mutated)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda case: case.update({"seed_track": "posgeo"}), "not SNAKE"),
        (lambda case: case.update({"seed_artifact_sha256": ""}), "lacks seed_artifact_sha256"),
        (lambda case: case.update({"same_seed_comparator_group_id": ""}), "lacks same_seed_comparator_group_id"),
        (
            lambda case: case.update(
                {"source_artifact_json": "chtc/time_dynamics_optuna/input/seed_artifacts/staged_pending_recovery_seed.json"}
            ),
            "forbidden seed source token",
        ),
        (
            lambda case: case.update({"latest_phase3_source_artifact_missing_locally": True}),
            "latest_phase3_source_artifact_missing_locally=true",
        ),
    ],
)
def test_paper_ii_snake_recovery_validation_fails_closed(mutation, message: str) -> None:
    payload = build_recovery_manifest()
    mutated = copy.deepcopy(payload)
    mutation(mutated["selected_cases"][0])

    with pytest.raises(RecoveryManifestValidationError, match=message):
        validate_recovery_manifest(mutated)
