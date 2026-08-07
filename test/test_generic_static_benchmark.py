#!/usr/bin/env python3
"""Tests for generic benchmark manifest generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pipelines.exact_bench.generic_static_benchmark import (
    _apply_benchmark_value_noise_to_row,
    _apply_primary_reference_metrics_to_row,
    _phase3_static_algorithmic_work_fields_from_result,
    _phase3_static_table_contract_fields_from_result,
    build_static_jobs,
    run_single,
)
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_CLEAN_NPH1_REF4_PROFILE,
    TABLE_I_CLEAN_NPH2_REF3_PROFILE,
    TABLE_I_CLEAN_NPH2_REF4_PROFILE,
    TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE,
    TABLE_I_CLEAN_NPH3_REF4_PROFILE,
    TABLE_I_DEFERRED_CASE_IDS_BY_FAMILY,
    TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE,
    TABLE_I_STANDARD_PROFILE,
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
    table_i_canonical_spec_by_case_id,
    table_i_declared_case_ids,
    table_i_deferred_case_reason,
    table_i_executable_case_ids,
    table_i_executable_case_ids_by_family,
    table_i_executable_specs,
    table_i_suite_profile,
)
from pipelines.exact_bench.table_i_static_benchmark import (
    TABLE_I_STATIC_ALGORITHM_IDS,
    build_table_i_static_jobs,
    summarize_table_i_jobs,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS,
    PAPER_I_MAIN_TABLES_SPSA_CASE_IDS,
    PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY,
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
    PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS,
)
from pipelines.reporting.benchmark_manifest import write_manifest_bundle
from pipelines.static_adapt.hardware_resolution_profiles import (
    HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA,
    HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA,
    HARDWARE_RESOLUTION_PROFILE_UNITS,
)
from pipelines.static_adapt.resume_scaffold import digest_jsonable, file_sha256
from pipelines.time_dynamics.tables.generic_dynamics_benchmark import build_dynamics_jobs


_NON_HH_STATIC_ED_REFERENCE_FAMILIES = tuple(
    family for family in TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY if family != "hh"
)
_GENERIC_STATIC_FIRST_SLICE_FAMILIES = tuple(TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY)

_STATIC_HEA_QISKIT_TRAIN_CASE_COUNTS = {
    family: len(case_ids)
    for family, case_ids in TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY.items()
    if family != "hh"
}


def _table_i_executable_case_count(profile: str | None = None) -> int:
    return sum(len(case_ids) for case_ids in table_i_executable_case_ids_by_family(profile).values())


def _write_local_fixture_hardware_resolution_profile(
    tmp_path: Path,
    *,
    name: str = "unit1g_synthetic_profile",
    hw_floor: float = 0.2,
    drift_floor: float = 0.05,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    selected_profile: dict[str, object] = {
        "schema": HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA,
        "name": name,
        "gradient_hw_floor": float(hw_floor),
        "gradient_drift_floor": float(drift_floor),
        "units": HARDWARE_RESOLUTION_PROFILE_UNITS,
        "provenance": {
            "source": "unit1g-local-synthetic-dry-run",
            "generated_utc": "2026-05-16T00:00:00Z",
            "calibration_semantic": "synthetic_nonphysical_test_fixture",
        },
    }
    manifest: dict[str, object] = {
        "schema": HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA,
        "profiles": {name: selected_profile},
    }
    path = tmp_path / "unit1g_synthetic_hardware_resolution_profiles.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path, manifest, selected_profile


def test_primary_reference_metric_rewrites_terminal_and_first_hits() -> None:
    row = {
        "energy": -1.9985,
        "exact_energy": -2.0,
        "delta_E_abs": 0.0015,
        "abs_delta_e": 0.0015,
        "benchmark_first_hits": {
            "1e-03": {"threshold_abs_delta_e": 1e-3, "energy": -1.9992, "delta_E_abs": 8e-4, "abs_delta_e": 8e-4},
            "1e-06": {"threshold_abs_delta_e": 1e-6, "energy": -1.9999995, "delta_E_abs": 5e-7, "abs_delta_e": 5e-7},
        },
    }
    status = _apply_primary_reference_metrics_to_row(
        row,
        {
            "same_cutoff_exact_gs_energy": "-2.0",
            "exact_reference_energy": "-1.999",
            "exact_reference_n_ph_max": "4",
            "primary_energy_metric": "higher_cutoff_reference_abs_delta_e",
            "same_cutoff_error_role": "diagnostic_only",
        },
    )

    assert status == "ok"
    assert row["abs_delta_e_same_cutoff"] == pytest.approx(1.5e-3)
    assert row["abs_delta_e_reference"] == pytest.approx(5e-4)
    assert row["abs_delta_e"] == pytest.approx(5e-4)
    assert row["delta_E_abs"] == pytest.approx(5e-4)
    assert row["primary_reference_source"] == "exact_reference_energy"
    assert set(row["benchmark_first_hits"]) == {"1e-03"}
    hit = row["benchmark_first_hits"]["1e-03"]
    assert hit["abs_delta_e_same_cutoff"] == pytest.approx(8e-4)
    assert hit["abs_delta_e_reference"] == pytest.approx(2e-4)
    assert hit["abs_delta_e"] == pytest.approx(2e-4)
    assert hit["exact_reference_n_ph_max"] == 4


def test_table_i_molecular_vibronic_h2_is_declared_but_deferred_at_head() -> None:
    assert table_i_declared_case_ids("molecular_vibronic_h2") == ("molecular_vibronic_h2_L2",)
    assert table_i_executable_case_ids("molecular_vibronic_h2") == ()
    assert TABLE_I_DEFERRED_CASE_IDS_BY_FAMILY["molecular_vibronic_h2"] == ("molecular_vibronic_h2_L2",)
    reason = table_i_deferred_case_reason("molecular_vibronic_h2", "molecular_vibronic_h2_L2")
    assert reason is not None
    assert "deferred" in reason
    assert "not executable at HEAD" in reason
    assert "molecular_restricted_closed_shell_L2" in reason
    with pytest.raises(ValueError, match="deferred.*not executable at HEAD|not executable at HEAD.*deferred"):
        table_i_canonical_spec_by_case_id("molecular_vibronic_h2", "molecular_vibronic_h2_L2")
    clean_ids = (
        "molecular_vibronic_h2_L2_nph1_clean_weak",
        "molecular_vibronic_h2_L2_nph1_clean_strong",
    )
    assert table_i_declared_case_ids("molecular_vibronic_h2", TABLE_I_CLEAN_NPH2_REF4_PROFILE) == clean_ids
    assert table_i_executable_case_ids("molecular_vibronic_h2", TABLE_I_CLEAN_NPH2_REF4_PROFILE) == clean_ids
    assert table_i_executable_case_ids("molecular_vibronic_h2", TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE) == (
        "molecular_vibronic_h2_L2_nph3_clean_strong",
    )


def _arg_value(args: tuple[str, ...] | list[str], flag: str) -> str | None:
    values = tuple(str(x) for x in args)
    try:
        idx = values.index(str(flag))
    except ValueError:
        return None
    return values[idx + 1] if idx + 1 < len(values) else None


def test_paper_i_main_tables_spsa_profile_resolves_exact_visible_case_set() -> None:
    assert TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert table_i_suite_profile(PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID) == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert table_i_suite_profile("paper_i_main_tables_spsa") == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert table_i_suite_profile("three_model_main") == TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE
    assert table_i_executable_case_ids_by_family(PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID) == (
        PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY
    )
    assert sum(len(case_ids) for case_ids in PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY.values()) == 8
    assert PAPER_I_MAIN_TABLES_SPSA_CASE_IDS == (
        "hubbard_L2_three_model_weak",
        "hubbard_L2_three_model_strong",
        "spin_boson_L2_nph1_three_model_weak",
        "spin_boson_L2_nph2_three_model_strong",
        "hh_L2_nph2_three_model_sym_weak_weak",
        "hh_L2_nph2_three_model_sym_strong_weak",
        "hh_L2_nph4_three_model_sym_weak_strong",
        "hh_L2_nph4_three_model_sym_strong_strong",
    )
    specs = table_i_executable_specs(PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    assert tuple(str(spec.benchmark_id) for spec in specs) == PAPER_I_MAIN_TABLES_SPSA_CASE_IDS
    assert tuple(str(spec.family) for spec in specs) == (
        "hubbard",
        "hubbard",
        "spin_boson",
        "spin_boson",
        "hh",
        "hh",
        "hh",
        "hh",
    )
    assert all(
        f"optimizer_profile:{PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID}" in tuple(spec.tags)
        for spec in specs
    )
    assert table_i_executable_case_ids("hubbard", TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE) == ()
    assert table_i_executable_case_ids("spin_boson", TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE) == ()
    assert table_i_executable_case_ids("hh", TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE) == (
        "hh_L2_nph2_three_model_sym_weak_weak",
        "hh_L2_nph2_three_model_sym_strong_weak",
        "hh_L2_nph4_three_model_sym_weak_strong",
        "hh_L2_nph4_three_model_sym_strong_strong",
    )


def test_clean_nph1_ref4_profile_resolves_hh_weak_strong() -> None:
    assert table_i_suite_profile("clean_nph1_ref4") == TABLE_I_CLEAN_NPH1_REF4_PROFILE
    assert table_i_executable_case_ids("hh", TABLE_I_CLEAN_NPH1_REF4_PROFILE) == (
        "hh_L2_nph1_clean_weak",
        "hh_L2_nph1_clean_strong",
    )
    specs = {
        str(spec.benchmark_id): spec
        for spec in table_i_executable_specs(TABLE_I_CLEAN_NPH1_REF4_PROFILE)
        if str(spec.family) == "hh"
    }
    assert set(specs) == {"hh_L2_nph1_clean_weak", "hh_L2_nph1_clean_strong"}
    assert all(_arg_value(spec.base_pipeline_args, "--n-ph-max") == "1" for spec in specs.values())
    assert all(spec.exact_reference_n_ph_max == 4 for spec in specs.values())


def test_paper_i_main_tables_spsa_profile_contract_constants_are_centralized() -> None:
    assert PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS == TABLE_I_STATIC_ALGORITHM_IDS[:-1]
    assert PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["hea"] == {
        "optimizer": "spsa",
        "spsa_maxiter": 800,
        "spsa_seed": 42,
    }
    assert PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["family_informed"]["optimizer"] == "spsa"
    assert PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["adapt"]["optimizer_kind"] == "spsa"
    assert PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"] == (
        "GENERIC_STATIC_TABLE_OPTIMIZER_PROFILE"
    )
    assert PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_optimizer_kind"] == (
        "GENERIC_STATIC_TABLE_ADAPT_OPTIMIZER_KIND"
    )
    assert PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS == (
        "hea_spsa_learning_rate",
        "hea_spsa_perturbation",
    )
    assert "family_informed_spsa_avg_last" in PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS
    assert PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS == (
        "adapt_spsa_a",
        "adapt_spsa_c",
        "adapt_spsa_alpha",
        "adapt_spsa_gamma",
        "adapt_spsa_big_a",
    )
    assert set(PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS).issubset(
        set(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES)
    )
    assert PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"] == (
        "GENERIC_STATIC_TABLE_HEA_SPSA_LEARNING_RATE"
    )


def test_exact_bench_h2_fixture_env_threads_to_comparator_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants
    import pipelines.exact_bench.generic_static_benchmark as static_benchmark
    import pipelines.exact_bench.generic_static_family_informed_vqe as family_vqe
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea
    import pipelines.exact_bench.generic_static_qiskit_adapt_vqe as qadapt

    fixture = tmp_path / "h2_runtime_fixture.json"
    fixture.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", TABLE_I_CLEAN_NPH2_REF4_PROFILE)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_MOLECULAR_VIBRONIC_H2_FIXTURE_JSON", str(fixture))

    weak_case = "molecular_vibronic_h2_L2_nph1_clean_weak"
    specs = (
        (hea, hea._spec_by_case_id("molecular_vibronic_h2", weak_case)),
        (family_vqe, family_vqe._spec_by_case_id("molecular_vibronic_h2", weak_case)),
        (qadapt, qadapt._spec_by_case_id("molecular_vibronic_h2", weak_case)),
        (
            variants,
            variants._spec_by_case_id(
                "molecular_vibronic_h2",
                weak_case,
                "static_full_meta_append_adapt_vqe",
            ),
        ),
    )
    for module, spec in specs:
        assert _arg_value(spec.base_pipeline_args, "--molecular-vibronic-h2-fixture-json") == str(fixture)
        namespace = module._namespace_from_base_args(spec.base_pipeline_args)
        assert namespace.molecular_vibronic_h2_fixture_json == str(fixture)

    static_benchmark._phase3_specs_for_family_profile.cache_clear()
    phase3_spec = static_benchmark._phase3_static_spec_for_case(
        family="molecular_vibronic_h2",
        case_id=weak_case,
        algorithm_id="static_family_native_adapt_phase3",
    )
    assert phase3_spec is not None
    assert _arg_value(phase3_spec.base_pipeline_args, "--molecular-vibronic-h2-fixture-json") == str(fixture)


def test_table_i_clean_profiles_encode_weak_strong_points_and_deferred_rows() -> None:
    nph3 = table_i_executable_case_ids_by_family(TABLE_I_CLEAN_NPH3_REF4_PROFILE)
    nph2 = table_i_executable_case_ids_by_family(TABLE_I_CLEAN_NPH2_REF3_PROFILE)

    assert sum(len(case_ids) for case_ids in nph3.values()) == 20
    assert sum(len(case_ids) for case_ids in nph2.values()) == 20
    assert set(nph3) == {
        "hubbard",
        "ionic_hubbard",
        "extended_hubbard",
        "ttprime_hubbard",
        "spinless_tv",
        "bose_hubbard",
        "harmonic_kerr_chain",
        "spin_boson",
        "hh",
        "molecular_vibronic_h2",
    }
    assert nph3["hubbard"] == ("hubbard_L2_clean_weak", "hubbard_L2_clean_strong")
    assert nph3["bose_hubbard"] == ("bose_hubbard_L2_nph3_clean_weak", "bose_hubbard_L2_nph3_clean_strong")
    assert nph2["hh"] == ("hh_L2_nph2_clean_weak", "hh_L2_nph2_clean_strong")
    assert nph2["molecular_vibronic_h2"] == (
        "molecular_vibronic_h2_L2_nph1_clean_weak",
        "molecular_vibronic_h2_L2_nph1_clean_strong",
    )

    weak_hubbard = table_i_canonical_spec_by_case_id("hubbard", "hubbard_L2_clean_weak", TABLE_I_CLEAN_NPH3_REF4_PROFILE)
    strong_hubbard = table_i_canonical_spec_by_case_id("hubbard", "hubbard_L2_clean_strong", TABLE_I_CLEAN_NPH3_REF4_PROFILE)
    assert weak_hubbard.base_pipeline_args[weak_hubbard.base_pipeline_args.index("--u") + 1] == "2.0"
    assert strong_hubbard.base_pipeline_args[strong_hubbard.base_pipeline_args.index("--u") + 1] == "8.0"

    weak_extended = table_i_canonical_spec_by_case_id(
        "extended_hubbard", "extended_hubbard_L2_clean_weak", TABLE_I_CLEAN_NPH3_REF4_PROFILE
    )
    strong_extended = table_i_canonical_spec_by_case_id(
        "extended_hubbard", "extended_hubbard_L2_clean_strong", TABLE_I_CLEAN_NPH3_REF4_PROFILE
    )
    assert weak_extended.base_pipeline_args[weak_extended.base_pipeline_args.index("--v-nn") + 1] == "0.5"
    assert strong_extended.base_pipeline_args[strong_extended.base_pipeline_args.index("--v-nn") + 1] == "1.5"

    weak_hh = table_i_canonical_spec_by_case_id("hh", "hh_L2_nph3_clean_weak", TABLE_I_CLEAN_NPH3_REF4_PROFILE)
    strong_hh = table_i_canonical_spec_by_case_id("hh", "hh_L2_nph3_clean_strong", TABLE_I_CLEAN_NPH3_REF4_PROFILE)
    assert weak_hh.base_pipeline_args[weak_hh.base_pipeline_args.index("--n-ph-max") + 1] == "3"
    assert weak_hh.exact_reference_n_ph_max == 4
    assert weak_hh.base_pipeline_args[weak_hh.base_pipeline_args.index("--u") + 1] == "2.0"
    assert weak_hh.base_pipeline_args[weak_hh.base_pipeline_args.index("--g-ep") + 1] == "0.25"
    assert strong_hh.base_pipeline_args[strong_hh.base_pipeline_args.index("--u") + 1] == "8.0"
    assert strong_hh.base_pipeline_args[strong_hh.base_pipeline_args.index("--g-ep") + 1] == "1.0"

    assert table_i_executable_case_ids("spin_boson", TABLE_I_CLEAN_NPH3_REF4_PROFILE) == (
        "spin_boson_L2_nph3_clean_weak",
        "spin_boson_L2_nph3_clean_strong",
    )
    weak_spin_boson = table_i_canonical_spec_by_case_id(
        "spin_boson", "spin_boson_L2_nph3_clean_weak", TABLE_I_CLEAN_NPH3_REF4_PROFILE
    )
    strong_spin_boson = table_i_canonical_spec_by_case_id(
        "spin_boson", "spin_boson_L2_nph3_clean_strong", TABLE_I_CLEAN_NPH3_REF4_PROFILE
    )
    assert weak_spin_boson.features.L == 2
    assert weak_spin_boson.base_pipeline_args[weak_spin_boson.base_pipeline_args.index("--g-ep") + 1] == "0.25"
    assert strong_spin_boson.base_pipeline_args[strong_spin_boson.base_pipeline_args.index("--g-ep") + 1] == "1.0"

    weak_h2 = table_i_canonical_spec_by_case_id(
        "molecular_vibronic_h2",
        "molecular_vibronic_h2_L2_nph1_clean_weak",
        TABLE_I_CLEAN_NPH2_REF4_PROFILE,
    )
    strong_h2 = table_i_canonical_spec_by_case_id(
        "molecular_vibronic_h2",
        "molecular_vibronic_h2_L2_nph1_clean_strong",
        TABLE_I_CLEAN_NPH2_REF4_PROFILE,
    )
    assert weak_h2.features.L == 2
    assert weak_h2.features.n_qubits == 5
    assert weak_h2.exact_reference_n_ph_max == 4
    assert weak_h2.base_pipeline_args[weak_h2.base_pipeline_args.index("--problem") + 1] == "molecular_vibronic_h2"
    assert weak_h2.base_pipeline_args[weak_h2.base_pipeline_args.index("--n-ph-max") + 1] == "1"
    assert weak_h2.base_pipeline_args[weak_h2.base_pipeline_args.index("--g-ep") + 1] == "0.25"
    assert strong_h2.base_pipeline_args[strong_h2.base_pipeline_args.index("--g-ep") + 1] == "1.0"

    escalated_h2 = table_i_canonical_spec_by_case_id(
        "molecular_vibronic_h2",
        "molecular_vibronic_h2_L2_nph3_clean_strong",
        TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE,
    )
    assert escalated_h2.features.L == 2
    assert escalated_h2.features.n_qubits == 6
    assert escalated_h2.exact_reference_n_ph_max == 6
    assert escalated_h2.base_pipeline_args[escalated_h2.base_pipeline_args.index("--n-ph-max") + 1] == "3"
    assert escalated_h2.base_pipeline_args[escalated_h2.base_pipeline_args.index("--g-ep") + 1] == "1.0"


def test_static_manifest_emits_runnable_hh_and_skipped_non_hh(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=("hh", "hubbard"),
        algorithm_ids=("static_hea_qiskit_vqe", "static_lang_firsov_vqe"),
        include_skipped=True,
    )

    # HEA now uses the Paper-I Table-I canonical case contract for HH and
    # Hubbard; Lang-Firsov remains an HH-specific legacy row plus a Hubbard skip.
    assert len(jobs) == 1 + 4 + 2 + 1
    hh_hea = [job for job in jobs if job.family == "hh" and job.algorithm_id == "static_hea_qiskit_vqe"]
    assert len(hh_hea) == 1
    assert {job.case_id for job in hh_hea} == {"hh_L2"}
    assert all(job.status == "runnable" for job in hh_hea)
    assert all(job.command for job in hh_hea)
    assert all(job.metadata.get("dispatch") == "generic_static_hea_qiskit_vqe" for job in hh_hea)

    hubbard_hea = [
        job for job in jobs if job.family == "hubbard" and job.algorithm_id == "static_hea_qiskit_vqe"
    ]
    assert len(hubbard_hea) == 2
    assert {job.case_id for job in hubbard_hea} == {"hubbard_L2", "hubbard_L2_u6"}
    assert all(job.status == "runnable" for job in hubbard_hea)
    assert all(job.metadata.get("dispatch") == "generic_static_hea_qiskit_vqe" for job in hubbard_hea)
    assert all(job.metadata.get("external_algorithm") is True for job in hubbard_hea)
    assert all(job.metadata.get("phase3_controller_called") is False for job in hubbard_hea)

    hubbard_lang = [
        job for job in jobs if job.family == "hubbard" and job.algorithm_id == "static_lang_firsov_vqe"
    ]
    assert len(hubbard_lang) == 1
    assert hubbard_lang[0].status == "skipped_unsupported"
    assert not hubbard_lang[0].command


def test_static_hea_qiskit_manifest_promotes_all_feasible_train_cases(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=_NON_HH_STATIC_ED_REFERENCE_FAMILIES,
        algorithm_ids=("static_hea_qiskit_vqe",),
        include_skipped=True,
    )

    expected_count = sum(_STATIC_HEA_QISKIT_TRAIN_CASE_COUNTS.values())
    assert len(jobs) == expected_count
    promoted = [job for job in jobs if job.family in _STATIC_HEA_QISKIT_TRAIN_CASE_COUNTS]
    assert len(promoted) == expected_count
    assert {job.family for job in jobs} == set(_NON_HH_STATIC_ED_REFERENCE_FAMILIES)
    counts = {family: 0 for family in _NON_HH_STATIC_ED_REFERENCE_FAMILIES}

    for job in promoted:
        counts[job.family] += 1
        assert job.status == "runnable", job.family
        assert job.command
        assert job.metadata.get("dispatch") == "generic_static_hea_qiskit_vqe"
        assert job.metadata.get("external_algorithm") is True
        assert job.metadata.get("optional_dependencies") == ["qiskit"]
        assert job.metadata.get("phase3_controller_called") is False
        assert job.metadata["comparator_source"]["execution_surface_role"] == "primary_execution_surface"
        assert job.metadata["comparator_source"]["external_reference_status"] == "primary_execution_surface"
        assert job.resources == {"request_cpus": 1, "request_memory": "8GB", "request_disk": "8GB"}
    assert counts == _STATIC_HEA_QISKIT_TRAIN_CASE_COUNTS


def test_static_qiskit_adapt_vqe_manifest_emits_canonical_table_i_cases(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=_GENERIC_STATIC_FIRST_SLICE_FAMILIES,
        algorithm_ids=("static_qiskit_adapt_vqe",),
        include_skipped=True,
    )

    expected_count = sum(len(case_ids) for case_ids in TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY.values())
    assert len(jobs) == expected_count
    assert {job.status for job in jobs} == {"runnable"}
    assert {job.family for job in jobs} == set(_GENERIC_STATIC_FIRST_SLICE_FAMILIES)
    expected_cases = {family: set(case_ids) for family, case_ids in TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY.items()}
    seen_cases = {family: set() for family in expected_cases}
    for job in jobs:
        assert job.case_id in expected_cases[job.family]
        seen_cases[job.family].add(job.case_id)
        assert job.command
        assert job.metadata.get("dispatch") == "generic_static_qiskit_adapt_vqe"
        assert job.metadata.get("external_algorithm") is True
        assert job.metadata.get("optional_dependencies") == ["qiskit", "qiskit_algorithms"]
        assert job.metadata.get("phase3_controller_called") is False
        assert job.metadata.get("resource_guarded_execution") is True
        assert job.metadata["comparator_source"]["execution_surface"] == "qiskit_algorithms_adaptvqe_full_meta_sparse_pauli_ops"
        assert job.metadata["comparator_source"]["parity_reference_algorithm_id"] == "static_full_meta_append_adapt_vqe"
        assert job.resources == {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
    assert seen_cases == expected_cases


def test_generic_static_adapt_variant_manifest_emits_canonical_table_i_cases(tmp_path: Path) -> None:
    expected_cases = {family: set(case_ids) for family, case_ids in TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY.items()}
    expected_count = sum(len(case_ids) for case_ids in TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY.values())
    for algorithm_id in (
        "static_full_meta_append_adapt_vqe",
        "static_qubit_qeb_adapt_vqe",
        "static_geo_qubit_adapt_vqe",
        "static_geo_qeb_adapt_vqe",
        "static_geo_adapt_vqe",
    ):
        jobs = build_static_jobs(
            output_root=tmp_path,
            families=_GENERIC_STATIC_FIRST_SLICE_FAMILIES,
            algorithm_ids=(algorithm_id,),
            include_skipped=True,
        )

        assert len(jobs) == expected_count
        assert {job.status for job in jobs} == {"runnable"}
        assert {job.family for job in jobs} == set(_GENERIC_STATIC_FIRST_SLICE_FAMILIES)
        seen_cases = {family: set() for family in expected_cases}
        for job in jobs:
            assert job.case_id in expected_cases[job.family]
            seen_cases[job.family].add(job.case_id)
            assert job.command
            assert job.metadata.get("dispatch") == "generic_static_adapt_variants"
            assert job.metadata.get("external_algorithm") is False
            assert job.metadata.get("benchmark_local_competitor") is True
            expected_optional_dependencies = (
                [] if algorithm_id in {"static_geo_qeb_adapt_vqe", "static_geo_adapt_vqe"} else ["scipy"]
            )
            assert job.metadata.get("optional_dependencies") == expected_optional_dependencies
            assert job.metadata.get("phase3_controller_called") is False
            assert job.metadata.get("resource_guarded_execution") is True
            assert job.metadata["comparator_source"]["execution_surface_role"] == "primary_execution_surface"
            assert job.resources == {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
        assert seen_cases == expected_cases


def test_static_ed_reference_manifest_emits_all_non_hh_train_cases(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=_NON_HH_STATIC_ED_REFERENCE_FAMILIES,
        algorithm_ids=("static_ed_reference",),
        include_skipped=False,
    )

    assert len(jobs) == sum(len(TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY[family]) for family in _NON_HH_STATIC_ED_REFERENCE_FAMILIES)
    assert {job.status for job in jobs} == {"runnable"}
    assert {job.family for job in jobs} == set(_NON_HH_STATIC_ED_REFERENCE_FAMILIES)
    counts = {family: 0 for family in _NON_HH_STATIC_ED_REFERENCE_FAMILIES}
    for job in jobs:
        counts[job.family] += 1
        assert job.command
        assert job.metadata.get("dispatch") == "generic_static_ed_reference"
        assert job.metadata.get("external_algorithm") is False
        assert job.metadata.get("optional_dependencies") == []
        assert job.metadata.get("phase3_controller_called") is False
        assert job.metadata.get("uses_existing_exact_target") is True
        assert job.metadata.get("resource_guarded_execution") is True
        assert job.resources == {"request_cpus": 1, "request_memory": "16GB", "request_disk": "20GB"}
    assert counts == {
        family: len(TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY[family])
        for family in _NON_HH_STATIC_ED_REFERENCE_FAMILIES
    }


def test_static_manifest_emits_project_controller_append_only_and_external_adapt_jobs(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=("hubbard",),
        algorithm_ids=(
            "static_family_native_adapt_phase3",
            "static_append_only_adapt_phase3",
            "static_ceo_adapt_phase3",
            "static_tetris_adapt_phase3",
        ),
        include_skipped=True,
    )

    runnable = [job for job in jobs if job.status == "runnable"]
    skipped = [job for job in jobs if job.status != "runnable"]
    assert runnable
    assert {job.algorithm_id for job in runnable} == {
        "static_family_native_adapt_phase3",
        "static_append_only_adapt_phase3",
        "static_ceo_adapt_phase3",
        "static_tetris_adapt_phase3",
    }
    phase3_jobs = [
        job
        for job in runnable
        if job.algorithm_id not in {"static_ceo_adapt_phase3", "static_tetris_adapt_phase3"}
    ]
    assert all(job.metadata.get("dispatch") == "phase3_static_adapt" for job in phase3_jobs)
    ceo_jobs = [job for job in runnable if job.algorithm_id == "static_ceo_adapt_phase3"]
    tetris_jobs = [job for job in runnable if job.algorithm_id == "static_tetris_adapt_phase3"]
    assert len(ceo_jobs) == 1
    assert len(tetris_jobs) == 3
    assert ceo_jobs[0].case_id == "hubbard_L2"
    assert tuple(job.case_id for job in tetris_jobs) == (
        "hubbard_L2",
        "hubbard_L2_three_model_weak",
        "hubbard_L2_three_model_strong",
    )
    assert ceo_jobs[0].metadata.get("dispatch") == "external_static_adapt_ceo_public_code"
    assert all(job.metadata.get("dispatch") == "external_static_adapt_tetris_public_code" for job in tetris_jobs)
    assert ceo_jobs[0].metadata.get("external_algorithm") is True
    assert all(job.metadata.get("external_algorithm") is True for job in tetris_jobs)
    assert ceo_jobs[0].metadata.get("phase3_controller_called") is False
    assert all(job.metadata.get("phase3_controller_called") is False for job in tetris_jobs)
    assert ceo_jobs[0].metadata.get("external_adapt_pinned_commits", {}).get("ceo_adapt_vqe")
    assert all(job.metadata.get("external_adapt_pinned_commits", {}).get("ceo_adapt_vqe") for job in tetris_jobs)
    assert skipped == []


def test_run_single_non_hh_project_controller_uses_static_adapt_runner(monkeypatch, tmp_path: Path) -> None:
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    captured = {}

    def fake_run_static_benchmark(spec, policy, *, output_dir, **kwargs):
        captured["spec"] = spec
        captured["policy"] = policy
        return p3opt.BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=True,
            abs_delta_e=1e-6,
        )

    monkeypatch.setattr(p3opt, "run_static_benchmark", fake_run_static_benchmark)

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_family_native_adapt_phase3",
        output_dir=tmp_path / "controller",
    )

    assert payload["schema"] == "generic_static_benchmark_phase3_single_v1"
    assert payload["status"] == "completed"
    assert captured["spec"].family == "hubbard"
    assert captured["policy"].pool.pool_key == "full_meta"


def test_run_single_non_hh_append_only_uses_append_only_policy(monkeypatch, tmp_path: Path) -> None:
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    captured = {}

    def fake_run_static_benchmark(spec, policy, *, output_dir, **kwargs):
        captured["policy"] = policy
        return p3opt.BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=True,
            abs_delta_e=1e-6,
        )

    monkeypatch.setattr(p3opt, "run_static_benchmark", fake_run_static_benchmark)

    run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_append_only_adapt_phase3",
        output_dir=tmp_path,
    )

    policy = captured["policy"]
    assert policy.pool.pool_key == "full_meta"
    assert policy.static.static_route_id == "unspecified"
    assert policy.static.adapt_reopt_policy == "append_only"
    assert policy.static.adapt_insertion_mode == "append_only"
    assert policy.static.phase1_prune_enabled is False


@pytest.mark.parametrize(
    "algorithm_id",
    (
        "static_family_native_adapt_phase3",
        "static_append_only_adapt_phase3",
    ),
)
def test_run_single_policy_json_historical_route_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    algorithm_id: str,
) -> None:
    policy_json = tmp_path / "policy.json"
    policy_json.write_text(
        json.dumps({"policy": {"pool": {}, "static": {"static_route_id": "route_a"}, "inner_optimizer": {}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_POLICY_JSON", str(policy_json))

    with pytest.raises(
        ValueError,
        match=(
            rf"{algorithm_id} cannot execute historical "
            r"static_route_id=.*must be 'unspecified'"
        ),
    ):
        run_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id=algorithm_id,
            output_dir=tmp_path / "append_policy_json",
        )


def test_invalid_phase3_emulated_competitor_ids_fail_closed(tmp_path: Path) -> None:
    for algorithm_id in (
        "static_qubit_qeb_adapt_phase3",
        "static_tetris_ceo_style_adapt_phase3",
    ):
        with pytest.raises(ValueError, match="Unknown benchmark algorithm"):
            run_single(
                family="hubbard",
                case_id="hubbard_L2",
                algorithm_id=algorithm_id,
                output_dir=tmp_path / algorithm_id,
            )
        with pytest.raises(ValueError, match="Unknown static benchmark algorithm"):
            build_static_jobs(
                output_root=tmp_path,
                families=("hubbard",),
                algorithm_ids=(algorithm_id,),
                include_skipped=True,
            )


def test_table_i_static_manifest_promotes_all_current_manuscript_methods(tmp_path: Path) -> None:
    jobs = build_table_i_static_jobs(output_root=tmp_path, include_skipped=True)

    assert jobs
    assert {job.algorithm_id for job in jobs} == set(TABLE_I_STATIC_ALGORITHM_IDS)
    assert "static_full_meta_append_adapt_vqe" in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_qiskit_adapt_vqe" not in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_qubit_qeb_adapt_vqe" in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_tetris_qubit_adapt_vqe" not in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_geo_qubit_adapt_vqe" not in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_geo_adapt_vqe" in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_family_informed_vqe" in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_qubit_qeb_adapt_phase3" not in TABLE_I_STATIC_ALGORITHM_IDS
    assert "static_tetris_ceo_style_adapt_phase3" not in TABLE_I_STATIC_ALGORITHM_IDS
    assert {job.status for job in jobs} == {"runnable"}
    assert all(job.command for job in jobs if job.status == "runnable")

    summary = summarize_table_i_jobs(jobs)
    expected_case_count = _table_i_executable_case_count()
    assert summary["table_label"] == "main_condensed Table I / tab:static_claims"
    assert summary["status_by_method"]["SNAKE"] == {"runnable": expected_case_count}
    assert summary["status_by_method"]["append-only ADAPT"] == {"runnable": expected_case_count}
    assert summary["status_by_method"]["HEA VQE"] == {"runnable": expected_case_count}
    assert summary["status_by_method"]["family-informed VQE"] == {"runnable": expected_case_count}
    assert summary["status_by_method"]["Qubit/QEB-ADAPT-VQE"] == {"runnable": expected_case_count}
    assert summary["status_by_method"]["TETRIS-ADAPT-VQE"] == {"runnable": expected_case_count}
    assert "full-meta metric ADAPT (Geo-style)" not in summary["status_by_method"]
    assert summary["status_by_method"]["Geo-ADAPT-VQE"] == {"runnable": expected_case_count}
    assert "TETRIS/CEO-style ADAPT" not in summary["status_by_method"]


def test_hh_tetris_manifest_is_explicit_skip_not_phase3_emulation(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=("hh",),
        algorithm_ids=("static_tetris_adapt_phase3",),
        include_skipped=True,
    )

    assert jobs
    assert {job.status for job in jobs} == {"skipped_not_implemented"}
    assert all(not job.command for job in jobs)

def test_static_manifest_can_filter_to_runnable_only(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=("hh", "hubbard"),
        algorithm_ids=("static_hea_qiskit_vqe", "static_lang_firsov_vqe"),
        include_skipped=False,
    )

    assert jobs
    assert {job.status for job in jobs} == {"runnable"}
    assert {job.family for job in jobs} == {"hh", "hubbard"}
    assert any(job.family == "hubbard" and job.case_id == "hubbard_L2" for job in jobs)


def test_run_single_qiskit_adapt_vqe_uses_generic_qiskit_runner_not_phase3(
    monkeypatch, tmp_path: Path
) -> None:
    import pipelines.exact_bench.generic_static_qiskit_adapt_vqe as qadapt
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    def _forbidden_phase3(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Qiskit AdaptVQE benchmark must not call Phase3 static ADAPT")

    captured = {}

    def _fake_runner(*, family: str, case_id: str, output_dir: Path, **kwargs):  # noqa: ANN003
        captured["family"] = family
        captured["case_id"] = case_id
        captured["output_dir"] = output_dir
        return {
            "schema": "generic_static_qiskit_adapt_vqe_v2",
            "family": family,
            "case_id": case_id,
            "algorithm_id": "static_qiskit_adapt_vqe",
            "status": "completed",
            "guardrails": {"phase3_controller_called": False},
        }

    monkeypatch.setattr(p3opt, "run_static_benchmark", _forbidden_phase3)
    monkeypatch.setattr(qadapt, "run_static_qiskit_adapt_vqe_single", _fake_runner)

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_qiskit_adapt_vqe",
        output_dir=tmp_path / "qadapt",
    )

    assert payload["schema"] == "generic_static_qiskit_adapt_vqe_v2"
    assert payload["status"] == "completed"
    assert captured == {
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "output_dir": tmp_path / "qadapt",
    }


def test_run_single_family_informed_vqe_uses_benchmark_local_runner_not_phase3(
    monkeypatch, tmp_path: Path
) -> None:
    import pipelines.exact_bench.generic_static_family_informed_vqe as family_vqe
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    def _forbidden_phase3(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("family-informed VQE benchmark must not call Phase3 static ADAPT")

    captured = {}

    def _fake_runner(*, family: str, case_id: str, output_dir: Path, **kwargs):  # noqa: ANN003
        captured["family"] = family
        captured["case_id"] = case_id
        captured["output_dir"] = output_dir
        return {
            "schema": "generic_static_family_informed_vqe_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": "static_family_informed_vqe",
            "status": "completed",
            "guardrails": {"phase3_controller_called": False, "phase3_emulation": False},
        }

    monkeypatch.setattr(p3opt, "run_static_benchmark", _forbidden_phase3)
    monkeypatch.setattr(family_vqe, "run_static_family_informed_vqe_single", _fake_runner)

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_family_informed_vqe",
        output_dir=tmp_path / "family_vqe",
    )

    assert payload["schema"] == "generic_static_family_informed_vqe_v1"
    assert payload["status"] == "completed"
    assert captured == {
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "output_dir": tmp_path / "family_vqe",
    }


def test_run_single_generic_static_adapt_variants_use_benchmark_local_runner_not_phase3(
    monkeypatch, tmp_path: Path
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    def _forbidden_phase3(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("generic static ADAPT competitors must not call Phase3 static ADAPT")

    captured = {}

    def _fake_runner(*, family: str, case_id: str, algorithm_id: str, output_dir: Path, **kwargs):  # noqa: ANN003
        captured[algorithm_id] = {
            "family": family,
            "case_id": case_id,
            "output_dir": output_dir,
        }
        return {
            "schema": "generic_static_adapt_variants_v3",
            "family": family,
            "case_id": case_id,
            "algorithm_id": algorithm_id,
            "status": "completed",
            "guardrails": {"phase3_controller_called": False, "phase3_emulation": False},
        }

    monkeypatch.setattr(p3opt, "run_static_benchmark", _forbidden_phase3)
    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)

    for algorithm_id in (
        "static_full_meta_append_adapt_vqe",
        "static_qubit_qeb_adapt_vqe",
        "static_geo_qubit_adapt_vqe",
        "static_geo_qeb_adapt_vqe",
        "static_geo_adapt_vqe",
    ):
        output_dir = tmp_path / algorithm_id
        payload = run_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id=algorithm_id,
            output_dir=output_dir,
        )
        assert payload["schema"] == "generic_static_adapt_variants_v3"
        assert payload["status"] == "completed"
        assert captured[algorithm_id] == {
            "family": "hubbard",
            "case_id": "hubbard_L2",
            "output_dir": output_dir,
        }


def test_run_single_selected_logical_env_threads_to_posgeo_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SELECTED_LOGICAL_ROUTE", "historical_selected")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SELECTED_LOGICAL_SOURCE_JSON", "chtc/phase3_optuna/input/selected/hk.json")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SELECTED_LOGICAL_TRANSFER_MODE", "boundary_v1")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "posgeo",
    )

    assert payload["status"] == "completed"
    assert captured["selected_logical_route"] == "historical_selected"
    assert captured["selected_logical_source_json"] == "chtc/phase3_optuna/input/selected/hk.json"
    assert captured["selected_logical_transfer_mode"] == "boundary_v1"


def test_run_single_generic_static_adapt_threads_progress_jsonl_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    progress_path = tmp_path / "progress" / "generic_static_adapt_progress.jsonl"
    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PROGRESS_JSONL_PATH", str(progress_path))

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "geo",
    )

    assert payload["status"] == "completed"
    assert captured["progress_jsonl_path"] == str(progress_path)


def test_run_single_generic_static_adapt_threads_runtime_split_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    import pipelines.exact_bench.generic_static_benchmark as benchmark
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setattr(
        benchmark,
        "evaluate_algorithm_for_family",
        lambda algorithm_id, family: SimpleNamespace(status="runnable", resolved_pool_key="full_meta"),
    )
    monkeypatch.setattr(benchmark, "_dispatch_kind", lambda **kwargs: "generic_static_adapt_variants")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE", "shortlist_pauli_children_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY", "hard_guard")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE", "4")

    payload = run_single(
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "geo_runtime_split",
    )

    assert payload["status"] == "completed"
    assert captured["generic_adapt_runtime_split_mode"] == "shortlist_pauli_children_v1"
    assert captured["generic_adapt_runtime_split_symmetry_policy"] == "hard_guard"
    assert captured["generic_adapt_runtime_split_max_subset_size"] == 4


def test_run_single_generic_static_adapt_threads_shared_pauli_pool_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    import pipelines.exact_bench.generic_static_benchmark as benchmark
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setattr(
        benchmark,
        "evaluate_algorithm_for_family",
        lambda algorithm_id, family: SimpleNamespace(status="runnable", resolved_pool_key="full_meta"),
    )
    monkeypatch.setattr(benchmark, "_dispatch_kind", lambda **kwargs: "generic_static_adapt_variants")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE", "shared_pauli_child_sets_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY", "hard_guard")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE", "4")

    payload = run_single(
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "geo_shared_pool",
    )

    assert payload["status"] == "completed"
    assert captured["shared_pauli_pool_mode"] == "shared_pauli_child_sets_v1"
    assert captured["shared_pauli_pool_symmetry_policy"] == "hard_guard"
    assert captured["shared_pauli_pool_max_subset_size"] == 4


def test_run_single_generic_static_adapt_threads_shared_pauli_pool_no_guard_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    import pipelines.exact_bench.generic_static_benchmark as benchmark
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setattr(
        benchmark,
        "evaluate_algorithm_for_family",
        lambda algorithm_id, family: SimpleNamespace(status="runnable", resolved_pool_key="full_meta"),
    )
    monkeypatch.setattr(benchmark, "_dispatch_kind", lambda **kwargs: "generic_static_adapt_variants")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE", "shared_pauli_child_sets_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY", "off")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE", "1")

    payload = run_single(
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "geo_shared_pool_no_guard",
    )

    assert payload["status"] == "completed"
    assert captured["shared_pauli_pool_mode"] == "shared_pauli_child_sets_v1"
    assert captured["shared_pauli_pool_symmetry_policy"] == "off"
    assert captured["shared_pauli_pool_max_subset_size"] == 1
    assert payload["shared_pauli_pool_env_overlay"] == {
        "shared_pauli_pool_mode": "shared_pauli_child_sets_v1",
        "shared_pauli_pool_symmetry_policy": "off",
        "shared_pauli_pool_max_subset_size": 1,
    }


def test_run_single_rejects_shared_pauli_pool_with_runtime_split_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    import pipelines.exact_bench.generic_static_benchmark as benchmark

    monkeypatch.setattr(
        benchmark,
        "evaluate_algorithm_for_family",
        lambda algorithm_id, family: SimpleNamespace(status="runnable", resolved_pool_key="full_meta"),
    )
    monkeypatch.setattr(benchmark, "_dispatch_kind", lambda **kwargs: "generic_static_adapt_variants")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE", "shared_pauli_child_sets_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY", "hard_guard")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE", "shortlist_pauli_children_v1")

    with pytest.raises(ValueError, match="cannot be combined"):
        run_single(
            family="hh",
            case_id="hh_L2_nph2_three_model_sym_weak_weak",
            algorithm_id="static_geo_adapt_vqe",
            output_dir=tmp_path / "blocked_shared_plus_runtime_split",
        )


def test_run_single_phase3_budget_env_threads_to_generic_adapt_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH", "500")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAXITER", "5000")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS", "true")
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", "paper_i_three_model_main_20260525_v1")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2_three_model_weak",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "append",
    )

    assert payload["status"] == "completed"
    assert captured["max_adapt_iterations"] == 500
    assert captured["optimizer_maxiter"] == 5000
    assert captured["allow_repeats"] is True
    assert "adapt_optimizer_kind" not in captured
    assert "optimizer_profile" not in captured


def test_run_single_fixed_horizon_stop_policy_disables_generic_adapt_target_and_gradient_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "metadata": {},
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY", "fixed_horizon_no_target_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH", "20")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS", "false")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS", "2e-4")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "geo_fixed_horizon",
    )

    assert payload["status"] == "completed"
    assert payload["generic_adapt_stop_policy"] == "fixed_horizon_no_target_v1"
    assert payload["metadata"]["generic_adapt_stop_policy"] == "fixed_horizon_no_target_v1"
    assert captured["max_adapt_iterations"] == 20
    assert captured["allow_repeats"] is False
    assert captured["energy_stop_target"] is None
    assert captured["first_hit_thresholds"] == (2e-4,)
    assert captured["gradient_threshold"] == 0.0
    assert captured["generic_adapt_stop_policy"] == "fixed_horizon_no_target_v1"


def test_run_single_fixed_horizon_stop_policy_rejects_target_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    def _unexpected_runner(**kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("fixed-horizon target-stop validation must happen before dispatch")

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _unexpected_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY", "fixed_horizon_no_target_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH", "20")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET", "2e-4")

    with pytest.raises(ValueError, match="requires energy_stop_target to be absent"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id="static_geo_adapt_vqe",
            output_dir=tmp_path / "bad_fixed_horizon",
        )


def test_run_single_fixed_horizon_stop_policy_rejects_wrong_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY", "fixed_horizon_no_target_v1")

    with pytest.raises(ValueError, match="generic_adapt_stop_policy env overlay is only valid"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_strong",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "bad_dispatch",
        )


def test_run_single_optimizer_profile_env_threads_to_hea_dispatch_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_hea_qiskit_vqe_v1",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", _fake_runner)
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_maxiter"], "7")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"], "0.04")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_perturbation"], "0.01")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2_three_model_weak",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "hea_profile",
    )

    assert payload["status"] == "completed"
    assert captured["optimizer_profile"] == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert captured["optimizer_profile_source"] == "env"
    assert captured["hea_optimizer"] == "spsa"
    assert captured["hea_spsa_maxiter"] == 7
    assert captured["hea_spsa_seed"] == PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["hea"]["spsa_seed"]
    assert captured["hea_spsa_learning_rate"] == pytest.approx(0.04)
    assert captured["hea_spsa_perturbation"] == pytest.approx(0.01)
    assert "optimizer" not in captured


def test_run_single_optimizer_unprefixed_env_is_ignored_for_legacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_hea_qiskit_vqe_v1",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", _fake_runner)
    monkeypatch.setenv("HEA_OPTIMIZER", "SPSA")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_optimizer"], "")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "hea_legacy_defaults",
    )

    assert payload["status"] == "completed"
    assert captured == {
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "output_dir": tmp_path / "hea_legacy_defaults",
    }


def test_run_single_optimizer_profile_env_threads_to_family_dispatch_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_family_informed_vqe as family_vqe

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_family_informed_vqe_v1",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(family_vqe, "run_static_family_informed_vqe_single", _fake_runner)
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_seed"], "123")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_a"], "0.05")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_c"], "0.02")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_eval_repeats"], "2")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_avg_last"], "3")

    payload = run_single(
        family="spin_boson",
        case_id="spin_boson_L2_nph1_three_model_weak",
        algorithm_id="static_family_informed_vqe",
        output_dir=tmp_path / "family_profile",
    )

    assert payload["status"] == "completed"
    assert captured["optimizer_profile"] == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert captured["family_informed_optimizer"] == "spsa"
    assert captured["family_informed_spsa_maxiter"] == (
        PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["family_informed"]["spsa_maxiter"]
    )
    assert captured["family_informed_spsa_seed"] == 123
    assert captured["family_informed_spsa_a"] == pytest.approx(0.05)
    assert captured["family_informed_spsa_c"] == pytest.approx(0.02)
    assert captured["family_informed_spsa_eval_repeats"] == 2
    assert captured["family_informed_spsa_avg_last"] == 3


def test_run_single_optimizer_profile_env_threads_to_generic_adapt_dispatch_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_maxiter"], "9")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_seed"], "321")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_a"], "0.07")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_c"], "0.03")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_alpha"], "0.602")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_gamma"], "0.101")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_big_a"], "50.0")

    payload = run_single(
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "append_profile",
    )

    assert payload["status"] == "completed"
    assert captured["optimizer_profile"] == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert captured["adapt_optimizer_kind"] == "spsa"
    assert captured["adapt_spsa_maxiter"] == 9
    assert captured["adapt_spsa_seed"] == 321
    assert captured["adapt_spsa_a"] == pytest.approx(0.07)
    assert captured["adapt_spsa_c"] == pytest.approx(0.03)
    assert captured["adapt_spsa_alpha"] == pytest.approx(0.602)
    assert captured["adapt_spsa_gamma"] == pytest.approx(0.101)
    assert captured["adapt_spsa_big_a"] == pytest.approx(50.0)
    assert captured["energy_stop_target"] is None


def test_run_single_optimizer_env_rejects_wrong_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_optimizer"], "SPSA")

    with pytest.raises(ValueError, match="HEA optimizer env overlay is only valid"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_family_informed_vqe",
            output_dir=tmp_path / "wrong_dispatch",
        )


def test_run_single_optimizer_schedule_env_rejects_wrong_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_a"], "0.05")

    with pytest.raises(ValueError, match="generic-ADAPT optimizer env overlay is only valid"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "wrong_schedule_dispatch",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("hea_spsa_learning_rate", "0", "positive finite float"),
        ("adapt_spsa_c", "nan", "positive finite float"),
        ("family_informed_spsa_eval_repeats", "0", "positive integer"),
    ],
)
def test_run_single_optimizer_schedule_env_validates_numeric_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES[field], value)

    algorithm_id = "static_hea_qiskit_vqe"
    if field.startswith("adapt_"):
        algorithm_id = "static_full_meta_append_adapt_vqe"
    elif field.startswith("family_informed_"):
        algorithm_id = "static_family_informed_vqe"

    with pytest.raises(ValueError, match=match):
        run_single(
            family="hubbard" if algorithm_id != "static_full_meta_append_adapt_vqe" else "hh",
            case_id=(
                "hubbard_L2_three_model_weak"
                if algorithm_id != "static_full_meta_append_adapt_vqe"
                else "hh_L2_nph2_three_model_sym_weak_weak"
            ),
            algorithm_id=algorithm_id,
            output_dir=tmp_path / "bad_schedule_value",
        )


def test_run_single_hea_schedule_requires_learning_rate_perturbation_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"], "0.04")

    with pytest.raises(ValueError, match="provided together"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "hea_one_sided_schedule",
        )


def test_run_single_schedule_fields_require_spsa_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_optimizer"], "bfgs")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_a"], "0.05")

    with pytest.raises(ValueError, match="require family_informed_optimizer=spsa"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_family_informed_vqe",
            output_dir=tmp_path / "schedule_requires_spsa",
        )


def test_run_single_optimizer_profile_requires_spsa_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_optimizer"], "COBYLA")

    with pytest.raises(ValueError, match="requires hea_optimizer=spsa"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "bad_profile_value",
        )


def test_run_single_optimizer_profile_rejects_non_visible_cases_and_posgeo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    with pytest.raises(ValueError, match="exact visible Paper-I main-table case set"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "legacy_case",
        )

    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    with pytest.raises(ValueError, match="displayed Paper-I methods"):
        run_single(
            family="hh",
            case_id="hh_L2_nph2_three_model_sym_weak_weak",
            algorithm_id="static_pos_geo_adapt_vqe",
            output_dir=tmp_path / "posgeo_profile",
        )


def test_run_single_optimizer_profile_reaches_hea_runner_after_internals_accept_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    monkeypatch.setattr(hea, "has_qiskit_hea_support", lambda: False)
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2_three_model_weak",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "hea_actual_runner_profile",
    )

    assert payload["status"] == "skipped_optional_dependency"
    assert payload["rows"][0]["status"] == "skipped_optional_dependency"


def test_run_single_selected_logical_env_rejects_non_generic_static_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SELECTED_LOGICAL_ROUTE", "historical_selected")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SELECTED_LOGICAL_SOURCE_JSON", "selected.json")

    with pytest.raises(ValueError, match="selected-logical CHTC env overlay"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "hea",
        )


def test_run_single_static_ed_reference_uses_generic_exact_target_runner_not_phase3(
    monkeypatch, tmp_path: Path
) -> None:
    import pipelines.exact_bench.generic_static_ed_reference as edref
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    def _forbidden_phase3(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("static ED reference benchmark must not call Phase3 static ADAPT")

    captured = {}

    def _fake_runner(*, family: str, case_id: str, output_dir: Path):
        captured["family"] = family
        captured["case_id"] = case_id
        captured["output_dir"] = output_dir
        return {
            "schema": "generic_static_ed_reference_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": "static_ed_reference",
            "status": "completed",
            "guardrails": {"phase3_controller_called": False},
        }

    monkeypatch.setattr(p3opt, "run_static_benchmark", _forbidden_phase3)
    monkeypatch.setattr(edref, "run_static_ed_reference_single", _fake_runner)

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_ed_reference",
        output_dir=tmp_path / "edref",
    )

    assert payload["schema"] == "generic_static_ed_reference_v1"
    assert payload["status"] == "completed"
    assert captured == {
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "output_dir": tmp_path / "edref",
    }


@pytest.mark.parametrize(
    ("family", "case_id"),
    (
        ("hubbard", "hubbard_L2"),
        ("ionic_hubbard", "ionic_hubbard_L2"),
        ("spinless_tv", "spinless_tv_L2"),
        ("spin_boson", "spin_boson_L1"),
        ("bose_hubbard", "bose_hubbard_L2"),
        ("harmonic_kerr_chain", "harmonic_kerr_chain_L2"),
    ),
)
def test_run_single_fixed_count_hea_uses_generic_qiskit_runner(
    family: str,
    case_id: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    def _forbidden_phase3(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("HEA benchmark must not call Phase3 static ADAPT")

    captured = {}

    def _fake_runner(*, family: str, case_id: str, output_dir: Path, **kwargs):  # noqa: ANN003
        captured["family"] = family
        captured["case_id"] = case_id
        captured["output_dir"] = output_dir
        return {
            "schema": "generic_static_hea_qiskit_vqe_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": "static_hea_qiskit_vqe",
            "status": "completed",
        }

    monkeypatch.setattr(p3opt, "run_static_benchmark", _forbidden_phase3)
    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", _fake_runner)

    output_dir = tmp_path / "hea" / family
    payload = run_single(
        family=family,
        case_id=case_id,
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=output_dir,
    )

    assert payload["schema"] == "generic_static_hea_qiskit_vqe_v1"
    assert payload["status"] == "completed"
    assert captured == {
        "family": family,
        "case_id": case_id,
        "output_dir": output_dir,
    }


def test_run_single_molecular_vibronic_h2_is_deferred_not_dispatched(monkeypatch, tmp_path: Path) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    def _forbidden_runner(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("deferred molecular_vibronic_h2_L2 must not dispatch to HEA")

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", _forbidden_runner)

    payload = run_single(
        family="molecular_vibronic_h2",
        case_id="molecular_vibronic_h2_L2",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "deferred_molecular",
    )

    assert payload["schema"] == "generic_static_benchmark_single_v1"
    assert payload["status"] == "skipped_not_implemented"
    assert "deferred" in payload["reason"]
    assert "not executable at HEAD" in payload["reason"]
    assert payload["metadata"]["table_i_deferred_case"] is True
    assert "molecular_restricted_closed_shell_L2" in payload["metadata"]["table_i_deferred_reason"]


def test_dynamics_manifest_promotes_generic_comparator_slice_but_keeps_other_non_hh_rows_skipped(tmp_path: Path) -> None:
    jobs = build_dynamics_jobs(
        output_root=tmp_path,
        families=("hh", "hubbard"),
        algorithm_ids=(
            "dyn_exact_reference",
            "dyn_fixed_mclachlan",
            "dyn_product_formula_envelope",
            "dyn_qdrift",
            "dyn_fixed_pvqd",
            "dyn_avqds",
            "dyn_adaptive_pvqd",
            "dyn_avqds_t",
            "dyn_vff_like",
        ),
        include_skipped=True,
    )

    hh_jobs = [job for job in jobs if job.family == "hh"]
    hubbard_jobs = {job.algorithm_id: job for job in jobs if job.family == "hubbard"}
    assert {job.status for job in hh_jobs} == {"runnable"}
    assert all(job.command for job in hh_jobs)
    assert hubbard_jobs["dyn_exact_reference"].status == "runnable"
    assert hubbard_jobs["dyn_fixed_mclachlan"].status == "runnable"
    assert hubbard_jobs["dyn_product_formula_envelope"].status == "runnable"
    assert hubbard_jobs["dyn_qdrift"].status == "runnable"
    assert hubbard_jobs["dyn_fixed_pvqd"].status == "runnable"
    assert hubbard_jobs["dyn_adaptive_pvqd"].status == "runnable"
    assert hubbard_jobs["dyn_avqds"].status == "runnable"
    assert hubbard_jobs["dyn_avqds_t"].status == "runnable"
    assert hubbard_jobs["dyn_exact_reference"].command
    assert hubbard_jobs["dyn_fixed_mclachlan"].command
    assert hubbard_jobs["dyn_product_formula_envelope"].command
    assert hubbard_jobs["dyn_qdrift"].command
    assert hubbard_jobs["dyn_fixed_pvqd"].command
    assert hubbard_jobs["dyn_adaptive_pvqd"].command
    assert hubbard_jobs["dyn_avqds"].command
    assert hubbard_jobs["dyn_avqds_t"].command
    assert hubbard_jobs["dyn_vff_like"].status == "skipped_unsupported"
    assert not hubbard_jobs["dyn_vff_like"].command


def test_generic_static_table_record_generator_includes_new_benchmark_rows_without_snake(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(output_dir=tmp_path / "input", queue_output_root=tmp_path / "queue")

    assert summary["include_snake"] is False
    assert "static_family_native_adapt_phase3" not in summary["algorithm_ids"]
    assert "static_family_native_adapt_phase3" not in summary["table_i"]["algorithm_ids"]
    assert "static_family_native_adapt_phase3" in summary["table_i"]["catalog_algorithm_ids"]
    assert "static_family_informed_vqe" in summary["algorithm_ids"]
    assert "static_full_meta_append_adapt_vqe" in summary["algorithm_ids"]
    assert "static_qiskit_adapt_vqe" not in summary["algorithm_ids"]
    assert "static_qubit_qeb_adapt_vqe" in summary["algorithm_ids"]
    assert "static_tetris_qubit_adapt_vqe" not in summary["algorithm_ids"]
    assert "static_geo_qubit_adapt_vqe" not in summary["algorithm_ids"]
    assert "static_geo_adapt_vqe" in summary["algorithm_ids"]
    expected_case_count = _table_i_executable_case_count()
    assert summary["status_by_algorithm"]["static_full_meta_append_adapt_vqe"] == {"runnable": expected_case_count}
    assert summary["status_by_algorithm"]["static_qubit_qeb_adapt_vqe"] == {"runnable": expected_case_count}
    assert "static_geo_qubit_adapt_vqe" not in summary["status_by_algorithm"]
    assert summary["status_by_algorithm"]["static_geo_adapt_vqe"] == {"runnable": expected_case_count}
    assert summary["status_by_algorithm"]["static_family_informed_vqe"] == {"runnable": expected_case_count}
    assert summary["runnable_record_count"] == expected_case_count * len(summary["algorithm_ids"])
    assert summary["smoke_record_count"] == 14

    records_path = Path(summary["paths"]["records_tsv"])
    records_text = records_path.read_text(encoding="utf-8")
    smoke_text = Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8")
    rows = list(csv.DictReader(records_text.splitlines(), delimiter="\t"))
    assert rows
    assert all(row["phase3_oracle_gradient_mode"] == "" for row in rows)
    assert all(row["phase3_oracle_value_noise_model"] == "" for row in rows)
    assert all(row["benchmark_value_noise_model"] == "" for row in rows)
    assert all(row["phase3_adapt_max_depth"] == "" for row in rows)
    assert all(row["phase3_adapt_maxiter"] == "" for row in rows)
    assert all(row["hardware_resolution_mode"] == "" for row in rows)
    assert all(row["hardware_resolution_profile_json"] == "" for row in rows)
    assert all(row["hardware_resolution_profile_name"] == "" for row in rows)
    assert all(row["static_route_id"] == "" for row in rows)
    assert summary["static_route_overlay"] == {
        "applied_record_count": 0,
        "route_a_record_count": 0,
        "unspecified_record_count": 0,
        "hardware_profile_rows_marked_diagnostic": False,
    }
    assert summary["hardware_resolution_profile_overlay"] == {
        "requested": False,
        "applied": False,
        "applied_record_count": 0,
        "fields": {
            "hardware_resolution_mode": "",
            "hardware_resolution_profile_json": "",
            "hardware_resolution_profile_name": "",
        },
    }
    assert summary["phase3_budget_overlay"] == {
        "requested": False,
        "applied": False,
        "applied_record_count": 0,
        "profile": "off",
        "fields": {
            "phase3_adapt_max_depth": "",
            "phase3_adapt_maxiter": "",
            "phase3_refit_maxiter": "",
            "phase3_final_maxiter": "",
            "phase3_adapt_spsa_a": "",
            "phase3_adapt_spsa_c": "",
            "phase3_adapt_spsa_big_a": "",
            "phase3_adapt_spsa_alpha": "",
            "phase3_adapt_spsa_gamma": "",
            "phase3_adapt_spsa_eval_repeats": "",
            "phase3_adapt_spsa_avg_last": "",
            "phase3_adapt_allow_repeats": "",
        },
    }
    assert "static_table__hubbard__hubbard_L2__static_family_informed_vqe" in smoke_text
    assert "static_table__bose_hubbard__bose_hubbard_L2__static_family_informed_vqe" in smoke_text
    assert "static_table__hh__hh_L2__static_full_meta_append_adapt_vqe" in smoke_text
    assert "static_qiskit_adapt_vqe" not in records_text
    assert "static_table__hh__hh_L2__static_qubit_qeb_adapt_vqe" in records_text
    assert "static_table__spinless_tv__spinless_tv_L2_v1p5__static_qubit_qeb_adapt_vqe" in records_text
    assert "static_tetris_qubit_adapt_vqe" not in records_text
    assert "static_table__molecular_vibronic_h2__molecular_vibronic_h2_L2" not in records_text
    assert "static_table__molecular_vibronic_h2__molecular_vibronic_h2_L2" not in smoke_text
    assert table_i_deferred_case_reason("molecular_vibronic_h2", "molecular_vibronic_h2_L2") is not None
    assert "static_geo_qubit_adapt_vqe" not in records_text
    assert "static_geo_qubit_adapt_vqe" not in smoke_text
    assert "static_table__hubbard__hubbard_L2__static_geo_adapt_vqe" in records_text
    assert "static_table__bose_hubbard__bose_hubbard_L2__static_geo_adapt_vqe" in smoke_text
    assert "static_family_native_adapt_phase3" not in records_text


def test_generic_static_table_generic_adapt_budget_profile_depth500_applies_to_comparators(
    tmp_path: Path,
) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    output_dir = tmp_path / "input"
    summary = generate_records(
        output_dir=output_dir,
        queue_output_root=tmp_path / "queue",
        suite_profile="paper_i_three_model_main_20260525_v1",
        family_filter=("hubbard",),
        algorithm_filter=(
            "static_full_meta_append_adapt_vqe",
            "static_geo_adapt_vqe",
            "static_qubit_qeb_adapt_vqe",
        ),
        energy_stop_target=0.0002,
        first_hit_thresholds=(0.0002,),
        disable_resource_guards=True,
        generic_adapt_budget_profile="paper_i_first_hit_depth500_v1",
        selected_logical_route="standard",
    )

    rows = list(
        csv.DictReader(
            (output_dir / "generic_static_table_records.tsv").read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
    )
    assert summary["runnable_record_count"] == 6
    assert summary["generic_adapt_budget_overlay"]["applied_record_count"] == 6
    assert {row["family"] for row in rows} == {"hubbard"}
    assert {row["algorithm_id"] for row in rows} == {
        "static_full_meta_append_adapt_vqe",
        "static_geo_adapt_vqe",
        "static_qubit_qeb_adapt_vqe",
    }
    assert all(row["selected_logical_route"] == "" for row in rows)
    assert all(row["phase3_adapt_max_depth"] == "500" for row in rows)
    assert all(row["phase3_adapt_maxiter"] == "5000" for row in rows)
    assert all(row["phase3_adapt_allow_repeats"] == "true" for row in rows)
    assert all(row["resource_qubit_cap"] == "0" and row["resource_pool_term_cap"] == "0" for row in rows)


def test_generic_static_table_hh_symmetric_full_meta_smokes_each_comparator_with_tiny_budget(
    tmp_path: Path,
) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    output_dir = tmp_path / "input"
    summary = generate_records(
        output_dir=output_dir,
        queue_output_root=tmp_path / "queue",
        suite_profile="paper_i_three_model_hh_symmetric_20260527_v1",
        family_filter=("hh",),
        algorithm_filter=(
            "static_full_meta_append_adapt_vqe",
            "static_geo_adapt_vqe",
        ),
        energy_stop_target=0.0002,
        first_hit_thresholds=(0.0002,),
        disable_resource_guards=True,
        generic_adapt_budget_profile="paper_i_first_hit_depth500_v1",
        generic_adapt_smoke_budget_profile="weak_local_v1",
        selected_logical_route="standard",
        smoke_case_ids=("hh_L2_nph2_three_model_sym_weak_weak",),
    )

    full_rows = list(
        csv.DictReader(
            (output_dir / "generic_static_table_records.tsv").read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
    )
    smoke_rows = list(
        csv.DictReader(
            (output_dir / "generic_static_table_smoke_records.tsv").read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
    )
    assert summary["runnable_record_count"] == 8
    assert summary["smoke_record_count"] == 2
    assert summary["generic_adapt_budget_overlay"]["applied_record_count"] == 8
    assert summary["generic_adapt_smoke_budget_overlay"]["applied_record_count"] == 2
    assert {row["algorithm_id"] for row in smoke_rows} == {
        "static_full_meta_append_adapt_vqe",
        "static_geo_adapt_vqe",
    }
    assert {row["case_id"] for row in smoke_rows} == {"hh_L2_nph2_three_model_sym_weak_weak"}
    assert all(row["phase3_adapt_max_depth"] == "500" for row in full_rows)
    assert all(row["phase3_adapt_maxiter"] == "5000" for row in full_rows)
    assert all(row["phase3_adapt_allow_repeats"] == "true" for row in full_rows)
    assert all(row["phase3_adapt_max_depth"] == "1" for row in smoke_rows)
    assert all(row["phase3_adapt_maxiter"] == "1" for row in smoke_rows)
    assert all(row["phase3_refit_maxiter"] == "1" for row in smoke_rows)
    assert all(row["phase3_final_maxiter"] == "1" for row in smoke_rows)


def test_generic_static_table_runtime_split_records_only_target_hh_append_and_geo(
    tmp_path: Path,
) -> None:
    from chtc.phase3_optuna import generate_generic_static_table_records as records_mod
    from pipelines.reporting.benchmark_manifest import BenchmarkJob

    jobs = (
        BenchmarkJob(
            job_id="append",
            domain="static",
            family="hh",
            case_id="hh_case",
            algorithm_id="static_full_meta_append_adapt_vqe",
            status="runnable",
            reason="",
            metadata={"dispatch": "generic_static_adapt_variants"},
        ),
        BenchmarkJob(
            job_id="geo",
            domain="static",
            family="hh",
            case_id="hh_case",
            algorithm_id="static_geo_adapt_vqe",
            status="runnable",
            reason="",
            metadata={"dispatch": "generic_static_adapt_variants"},
        ),
        BenchmarkJob(
            job_id="qeb",
            domain="static",
            family="hh",
            case_id="hh_case",
            algorithm_id="static_qubit_qeb_adapt_vqe",
            status="runnable",
            reason="",
            metadata={"dispatch": "generic_static_adapt_variants"},
        ),
        BenchmarkJob(
            job_id="hubbard_append",
            domain="static",
            family="hubbard",
            case_id="hubbard_case",
            algorithm_id="static_full_meta_append_adapt_vqe",
            status="runnable",
            reason="",
            metadata={"dispatch": "generic_static_adapt_variants"},
        ),
    )
    overlay = records_mod._normalize_generic_adapt_runtime_split_overlay(
        generic_adapt_runtime_split_mode="shortlist_pauli_children_v1",
        generic_adapt_runtime_split_symmetry_policy="hard_guard",
        generic_adapt_runtime_split_max_subset_size=4,
    )
    rows = records_mod._records_from_jobs(
        jobs,
        suite_profile=TABLE_I_STANDARD_PROFILE,
        energy_stop_target=0.0002,
        first_hit_thresholds=(0.0002,),
        generic_adapt_runtime_split_overlay=overlay,
    )
    records_mod._write_records(tmp_path / "records.tsv", rows)

    by_id = {row["record_id"]: row for row in rows}
    for record_id in ("static_table__hh__hh_case__static_full_meta_append_adapt_vqe", "static_table__hh__hh_case__static_geo_adapt_vqe"):
        assert by_id[record_id]["generic_adapt_runtime_split_mode"] == "shortlist_pauli_children_v1"
        assert by_id[record_id]["generic_adapt_runtime_split_symmetry_policy"] == "hard_guard"
        assert by_id[record_id]["generic_adapt_runtime_split_max_subset_size"] == "4"
    assert by_id["static_table__hh__hh_case__static_qubit_qeb_adapt_vqe"]["generic_adapt_runtime_split_mode"] == ""
    assert by_id["static_table__hubbard__hubbard_case__static_full_meta_append_adapt_vqe"]["generic_adapt_runtime_split_mode"] == "shortlist_pauli_children_v1"
    header = (tmp_path / "records.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert "generic_adapt_runtime_split_mode" in header
    assert "generic_adapt_runtime_split_symmetry_policy" in header
    assert "generic_adapt_runtime_split_max_subset_size" in header


def test_generic_static_table_shared_pauli_pool_no_guard_targets_snake_append_and_geo(
    tmp_path: Path,
) -> None:
    from chtc.phase3_optuna import generate_generic_static_table_records as records_mod
    from pipelines.reporting.benchmark_manifest import BenchmarkJob

    jobs = (
        BenchmarkJob(
            job_id="snake",
            domain="static",
            family="hh",
            case_id="hh_case",
            algorithm_id="static_family_native_adapt_phase3",
            status="runnable",
            reason="",
            metadata={"dispatch": "phase3_static_adapt"},
        ),
        BenchmarkJob(
            job_id="append",
            domain="static",
            family="hh",
            case_id="hh_case",
            algorithm_id="static_full_meta_append_adapt_vqe",
            status="runnable",
            reason="",
            metadata={"dispatch": "generic_static_adapt_variants"},
        ),
        BenchmarkJob(
            job_id="geo",
            domain="static",
            family="hh",
            case_id="hh_case",
            algorithm_id="static_geo_adapt_vqe",
            status="runnable",
            reason="",
            metadata={"dispatch": "generic_static_adapt_variants"},
        ),
        BenchmarkJob(
            job_id="qeb",
            domain="static",
            family="hh",
            case_id="hh_case",
            algorithm_id="static_qubit_qeb_adapt_vqe",
            status="runnable",
            reason="",
            metadata={"dispatch": "generic_static_adapt_variants"},
        ),
    )
    overlay = records_mod._normalize_shared_pauli_pool_overlay(
        shared_pauli_pool_mode="shared_pauli_child_sets_v1",
        shared_pauli_pool_symmetry_policy="off",
        shared_pauli_pool_max_subset_size=1,
    )
    rows = records_mod._records_from_jobs(
        jobs,
        suite_profile=TABLE_I_STANDARD_PROFILE,
        energy_stop_target=0.0002,
        first_hit_thresholds=(0.0002,),
        shared_pauli_pool_overlay=overlay,
    )
    records_mod._write_records(tmp_path / "records.tsv", rows)

    by_id = {row["record_id"]: row for row in rows}
    for record_id in (
        "static_table__hh__hh_case__static_family_native_adapt_phase3",
        "static_table__hh__hh_case__static_full_meta_append_adapt_vqe",
        "static_table__hh__hh_case__static_geo_adapt_vqe",
    ):
        assert by_id[record_id]["shared_pauli_pool_mode"] == "shared_pauli_child_sets_v1"
        assert by_id[record_id]["shared_pauli_pool_symmetry_policy"] == "off"
        assert by_id[record_id]["shared_pauli_pool_max_subset_size"] == "1"
    assert by_id["static_table__hh__hh_case__static_qubit_qeb_adapt_vqe"]["shared_pauli_pool_mode"] == ""
    header = (tmp_path / "records.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert "shared_pauli_pool_mode" in header
    assert "shared_pauli_pool_symmetry_policy" in header
    assert "shared_pauli_pool_max_subset_size" in header


def test_generic_static_table_hh_symmetric_selected_overlay_only_supported_rows(
    tmp_path: Path,
) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    output_dir = tmp_path / "input"
    source = "chtc/phase3_optuna/input/paper_i_three_model_reduced_pool_selected_logical_20260525_v1/hh_L2_from_result.selected_logical.json"
    summary = generate_records(
        output_dir=output_dir,
        queue_output_root=tmp_path / "queue",
        suite_profile="paper_i_three_model_hh_symmetric_20260527_v1",
        family_filter=("hh",),
        energy_stop_target=0.0002,
        first_hit_thresholds=(0.0002,),
        disable_resource_guards=True,
        generic_adapt_budget_profile="paper_i_first_hit_depth256_v1",
        selected_logical_route="historical_selected",
        selected_logical_source_json=source,
        selected_logical_transfer_mode="boundary_v1",
        selected_logical_supported_algorithms_only=True,
    )

    rows = list(
        csv.DictReader(
            (output_dir / "generic_static_table_records.tsv").read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
    )
    selected = [row for row in rows if row["selected_logical_route"] == "historical_selected"]
    assert summary["runnable_record_count"] == 20
    assert summary["generic_adapt_budget_overlay"]["applied_record_count"] == 12
    assert summary["selected_logical_overlay"]["applied_record_count"] == 8
    assert {row["case_id"] for row in rows} == {
        "hh_L2_nph2_three_model_sym_weak_weak",
        "hh_L2_nph2_three_model_sym_strong_weak",
        "hh_L2_nph4_three_model_sym_weak_strong",
        "hh_L2_nph4_three_model_sym_strong_strong",
    }
    assert {row["algorithm_id"] for row in selected} == {
        "static_full_meta_append_adapt_vqe",
        "static_geo_adapt_vqe",
    }
    assert all(row["selected_logical_transfer_mode"] == "boundary_v1" for row in selected)
    assert all(row["phase3_adapt_max_depth"] == "256" for row in rows if row["algorithm_id"].endswith("_adapt_vqe"))
    assert all(row["n_ph_ref"] in {"5", "7"} for row in rows)


def test_generic_static_table_geo_comparator_is_append_only_without_posgeo_override(
    tmp_path: Path,
) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    output_dir = tmp_path / "input"
    generate_records(
        output_dir=output_dir,
        queue_output_root=tmp_path / "queue",
        suite_profile="paper_i_three_model_main_20260525_v1",
        family_filter=("hh",),
        algorithm_filter=("static_geo_adapt_vqe",),
        energy_stop_target=0.0002,
        first_hit_thresholds=(0.0002,),
        disable_resource_guards=True,
        generic_adapt_budget_profile="paper_i_first_hit_depth500_v1",
        selected_logical_route="standard",
    )

    rows = list(
        csv.DictReader(
            (output_dir / "generic_static_table_records.tsv").read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
    )
    assert rows
    assert {row["algorithm_id"] for row in rows} == {"static_geo_adapt_vqe"}
    assert {row["family"] for row in rows} == {"hh"}
    assert all(row["phase3_pos_geo_position_policy"] == "" for row in rows)



def test_generic_static_table_nph2_ref3_record_generator_uses_cutoff2_cases(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
    )

    assert summary["suite_profile"] == "nph2_ref3_v1"
    assert summary["energy_stop_target"] == 1e-8
    assert summary["first_hit_thresholds"] == [1e-6, 1e-8]
    expected_case_count = _table_i_executable_case_count("nph2_ref3_v1")
    assert summary["runnable_record_count"] == expected_case_count * len(summary["algorithm_ids"])
    assert summary["smoke_record_count"] == 12
    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    assert "static_table__bose_hubbard__bose_hubbard_L2_nph2__static_geo_adapt_vqe" in records_text
    assert "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph2_w0p75__static_geo_adapt_vqe" in records_text
    assert "static_tetris_qubit_adapt_vqe" not in records_text
    assert "static_table__hh__hh_L2_nph2__static_full_meta_append_adapt_vqe" in records_text
    assert "static_table__molecular_vibronic_h2__molecular_vibronic_h2_L2" not in records_text
    assert table_i_deferred_case_reason("molecular_vibronic_h2", "molecular_vibronic_h2_L2", "nph2_ref3_v1") is not None
    assert "static_qiskit_adapt_vqe" not in records_text
    assert "\tnph2_ref3_v1\t1e-08\t1e-06,1e-08" in records_text


def test_generic_static_table_clean_record_generator_is_benchmark_only_with_explicit_cutoffs(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        suite_profile=TABLE_I_CLEAN_NPH3_REF4_PROFILE,
    )

    assert summary["suite_profile"] == TABLE_I_CLEAN_NPH3_REF4_PROFILE
    assert summary["include_snake"] is False
    assert summary["snake_only"] is False
    assert "static_family_native_adapt_phase3" not in summary["algorithm_ids"]
    expected_case_count = _table_i_executable_case_count(TABLE_I_CLEAN_NPH3_REF4_PROFILE)
    assert expected_case_count == 20
    assert summary["runnable_record_count"] == expected_case_count * len(summary["algorithm_ids"])
    assert summary["smoke_record_count"] == 3
    assert summary["phonon_cutoff_fields"]["explicit_record_fields"] is True

    rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert rows
    phonon_rows = [row for row in rows if row["family"] in {"bose_hubbard", "harmonic_kerr_chain", "spin_boson", "hh"}]
    h2_rows = [row for row in rows if row["family"] == "molecular_vibronic_h2"]
    fermionic_rows = [row for row in rows if row["family"] in {"hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard", "spinless_tv"}]
    assert phonon_rows
    assert h2_rows
    assert fermionic_rows
    assert {row["n_ph_work"] for row in phonon_rows} == {"3"}
    assert {row["n_ph_ref"] for row in phonon_rows} == {"4"}
    assert {row["primary_energy_metric"] for row in phonon_rows} == {"higher_cutoff_reference_abs_delta_e"}
    assert {row["case_id"] for row in h2_rows} == {
        "molecular_vibronic_h2_L2_nph1_clean_weak",
        "molecular_vibronic_h2_L2_nph1_clean_strong",
    }
    assert {row["n_ph_work"] for row in h2_rows} == {"1"}
    assert {row["n_ph_ref"] for row in h2_rows} == {"4"}
    assert {row["primary_energy_metric"] for row in h2_rows} == {"higher_cutoff_reference_abs_delta_e"}
    assert all(row["n_ph_work"] == "" and row["n_ph_ref"] == "" and row["primary_energy_metric"] == "" for row in fermionic_rows)

    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    smoke_text = Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8")
    assert "static_table__bose_hubbard__bose_hubbard_L2_nph3_clean_weak__static_hea_qiskit_vqe" in smoke_text
    assert "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph3_clean_weak__static_family_informed_vqe" in smoke_text
    assert "static_table__hh__hh_L2_nph3_clean_weak__static_full_meta_append_adapt_vqe" in smoke_text
    assert "static_family_native_adapt_phase3" not in records_text
    assert "spin_boson_L2_nph3_clean_weak" in records_text
    assert "molecular_vibronic_h2_L2_nph1_clean_weak" in records_text
    assert "molecular_restricted_closed_shell" not in records_text


def test_generic_static_table_clean_generator_rejects_snake_rows(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="benchmark-only"):
        generate_records(
            output_dir=tmp_path / "input",
            queue_output_root=tmp_path / "queue",
            suite_profile=TABLE_I_CLEAN_NPH3_REF4_PROFILE,
            include_snake=True,
        )

    with pytest.raises(ValueError, match="benchmark-only"):
        generate_records(
            output_dir=tmp_path / "input2",
            queue_output_root=tmp_path / "queue2",
            suite_profile=TABLE_I_CLEAN_NPH2_REF3_PROFILE,
            snake_only=True,
        )


def test_generic_static_table_h2_clean_comparator_records_are_fixture_backed(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        suite_profile=TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        family_filter=("molecular_vibronic_h2",),
        energy_stop_target=2e-4,
        first_hit_thresholds=(2e-4,),
    )

    assert summary["suite_profile"] == TABLE_I_CLEAN_NPH2_REF4_PROFILE
    assert summary["family_filter"] == ["molecular_vibronic_h2"]
    assert summary["runnable_record_count"] == 10
    assert summary["smoke_record_count"] == 1
    assert "static_family_native_adapt_phase3" not in summary["algorithm_ids"]

    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    smoke_text = Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8")
    assert "molecular_restricted_closed_shell" not in records_text
    assert "molecular_vibronic_h2_L2_nph1_clean_weak" in records_text
    assert "molecular_vibronic_h2_L2_nph1_clean_strong" in records_text
    assert "molecular_vibronic_h2" in smoke_text

    rows = list(csv.DictReader(records_text.splitlines(), delimiter="\t"))
    assert {row["family"] for row in rows} == {"molecular_vibronic_h2"}
    assert {row["case_id"] for row in rows} == {
        "molecular_vibronic_h2_L2_nph1_clean_weak",
        "molecular_vibronic_h2_L2_nph1_clean_strong",
    }
    assert {row["algorithm_id"] for row in rows} == {
        "static_hea_qiskit_vqe",
        "static_family_informed_vqe",
        "static_full_meta_append_adapt_vqe",
        "static_qubit_qeb_adapt_vqe",
        "static_geo_adapt_vqe",
    }
    assert {row["n_ph_work"] for row in rows} == {"1"}
    assert {row["n_ph_ref"] for row in rows} == {"4"}
    assert {row["reference_energy_status"] for row in rows} == {"ok"}
    assert all(row["same_cutoff_exact_gs_energy"] for row in rows)
    assert all(row["exact_reference_energy"] for row in rows)
    assert all(row["exact_reference_n_ph_max"] == "4" for row in rows)


def test_generic_static_table_h2_clean_snake_records_are_separate_and_seeded(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        suite_profile=TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        family_filter=("molecular_vibronic_h2",),
        snake_only=True,
        energy_stop_target=2e-4,
        first_hit_thresholds=(2e-4,),
        phase3_policy_profile="spsa_prior_best_v1",
        phase3_oracle_seed=7,
        phase3_adapt_parallel_gradient_workers=10,
        phase3_adapt_beam_parent_workers=4,
    )

    assert summary["snake_only"] is True
    assert summary["include_snake"] is False
    assert summary["algorithm_ids"] == ["static_family_native_adapt_phase3"]
    assert summary["runnable_record_count"] == 2
    assert summary["smoke_record_count"] == 1
    assert summary["phase3_policy_overlay"]["applied_record_count"] == 2
    assert summary["phase3_runtime_overlay"]["applied_record_count"] == 2
    assert summary["phase3_oracle_overlay"]["applied_record_count"] == 2
    assert summary["static_route_overlay"]["route_a_record_count"] == 2

    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    smoke_text = Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8")
    assert "molecular_restricted_closed_shell" not in records_text
    assert "molecular_vibronic_h2" in smoke_text
    rows = list(csv.DictReader(records_text.splitlines(), delimiter="\t"))
    assert len(rows) == 2
    assert {row["case_id"] for row in rows} == {
        "molecular_vibronic_h2_L2_nph1_clean_weak",
        "molecular_vibronic_h2_L2_nph1_clean_strong",
    }
    assert all(row["algorithm_id"] == "static_family_native_adapt_phase3" for row in rows)
    assert all(row["n_ph_work"] == "1" and row["n_ph_ref"] == "4" for row in rows)
    assert all(row["phase3_adapt_allow_repeats"] == "true" for row in rows)
    assert all(row["phase3_oracle_seed"] == "7" for row in rows)
    assert all(row["phase3_adapt_parallel_gradient_workers"] == "10" for row in rows)
    assert all(row["phase3_adapt_beam_parent_workers"] == "4" for row in rows)
    assert all(row["static_route_id"] == "route_a" for row in rows)
    assert all(row["reference_energy_status"] == "ok" for row in rows)


def test_generic_static_table_h2_strong_nph3_ref6_full_suite_records(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        suite_profile=TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE,
        family_filter=("molecular_vibronic_h2",),
        include_snake=True,
        energy_stop_target=2e-4,
        first_hit_thresholds=(2e-4,),
        phase3_policy_profile="spsa_prior_best_v1",
        phase3_oracle_seed=7,
        phase3_adapt_parallel_gradient_workers=10,
        phase3_adapt_beam_parent_workers=4,
    )

    assert summary["suite_profile"] == TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE
    assert summary["include_snake"] is True
    assert summary["runnable_record_count"] == len(TABLE_I_STATIC_ALGORITHM_IDS)
    assert summary["smoke_record_count"] == 1
    assert summary["phase3_policy_overlay"]["applied_record_count"] == 1
    assert summary["phase3_runtime_overlay"]["applied_record_count"] == 1
    assert summary["phase3_oracle_overlay"]["applied_record_count"] == 1
    assert summary["static_route_overlay"]["route_a_record_count"] == 1

    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    assert "molecular_restricted_closed_shell" not in records_text
    rows = list(csv.DictReader(records_text.splitlines(), delimiter="\t"))
    assert {row["family"] for row in rows} == {"molecular_vibronic_h2"}
    assert {row["case_id"] for row in rows} == {"molecular_vibronic_h2_L2_nph3_clean_strong"}
    assert {row["algorithm_id"] for row in rows} == set(TABLE_I_STATIC_ALGORITHM_IDS)
    assert {row["n_ph_work"] for row in rows} == {"3"}
    assert {row["n_ph_ref"] for row in rows} == {"6"}
    assert {row["exact_reference_n_ph_max"] for row in rows} == {"6"}
    assert {row["reference_energy_status"] for row in rows} == {"ok"}


def test_generic_static_table_clean_ladder_generates_explicit_ref4_and_escalation_rows(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "ladder_nph2_ref4",
        queue_output_root=tmp_path / "queue",
        paper_i_cutoff_ladder_stage="nph2_ref4_screen",
    )
    assert summary["suite_profile"] == TABLE_I_CLEAN_NPH2_REF4_PROFILE
    assert summary["paper_i_cutoff_ladder"]["stage"] == "nph2_ref4_screen"
    assert summary["paper_i_cutoff_ladder"]["acceptance_threshold"] == pytest.approx(2e-4)
    assert summary["runnable_record_count"] == 56
    assert summary["paper_i_cutoff_ladder"]["snake_policy"] == "included"

    rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert {row["family"] for row in rows} == {"bose_hubbard", "harmonic_kerr_chain", "spin_boson", "hh"}
    assert {row["n_ph_work"] for row in rows} == {"2"}
    assert {row["n_ph_ref"] for row in rows} == {"4"}
    assert {row["paper_i_cutoff_ladder_stage"] for row in rows} == {"nph2_ref4_screen"}
    assert {row["primary_energy_metric"] for row in rows} == {"higher_cutoff_reference_abs_delta_e"}
    assert {row["same_cutoff_error_role"] for row in rows} == {"diagnostic_only"}
    assert {row["tau_phys"] for row in rows} == {"0.0002"}
    assert {row["tau_tight"] for row in rows} == {"0.0002"}
    assert "static_family_native_adapt_phase3" in {row["algorithm_id"] for row in rows}
    assert {row["paper_i_ladder_snake_policy"] for row in rows} == {"included"}
    assert {row["reference_energy_status"] for row in rows} == {"ok"}
    assert all(row["exact_reference_energy"] for row in rows)
    assert all(row["same_cutoff_exact_gs_energy"] for row in rows)

    with pytest.raises(ValueError, match="requires explicit"):
        generate_records(
            output_dir=tmp_path / "bad_nph3",
            queue_output_root=tmp_path / "queue3",
            paper_i_cutoff_ladder_stage="nph3_ref4_escalation",
        )
    with pytest.raises(ValueError, match="missing requested case_id"):
        generate_records(
            output_dir=tmp_path / "bad_nph3_case",
            queue_output_root=tmp_path / "queue3_case",
            paper_i_cutoff_ladder_stage="nph3_ref4_escalation",
            paper_i_ladder_case_ids=("bose_hubbard_L2_nph3_clean_weak", "not_a_case"),
            paper_i_ladder_escalation_reason="prior_stage_failed",
        )
    with pytest.raises(ValueError, match="allow-ref5"):
        generate_records(
            output_dir=tmp_path / "bad_nph4",
            queue_output_root=tmp_path / "queue4",
            paper_i_cutoff_ladder_stage="nph4_ref5_optional",
            paper_i_ladder_case_ids=("bose_hubbard_L2_nph4_clean_weak",),
            paper_i_ladder_escalation_reason="prior_stage_failed",
        )


def test_generic_static_table_ladder_candidate_manifest_filters_comparator_records(tmp_path: Path) -> None:
    from chtc.phase3_optuna import paper_i_table_i_audit_escalation as audit_escalation
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    audit_path = tmp_path / "audit.json"
    audit_path.write_text('{"schema":"paper_i_fixed_accuracy_table_audit_v1"}\n', encoding="utf-8")
    candidate_manifest = tmp_path / "comparator_candidates.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "schema": audit_escalation.CANDIDATE_SCHEMA,
                "threshold": 2e-4,
                "target_profile": "paper_i_phys_v1",
                "source_stage": "nph2_ref4_screen",
                "next_stage": "nph3_ref4_escalation",
                "candidates": [
                    {
                        "candidate_key": (
                            "comparator|harmonic_kerr_chain|HEA VQE|weak|"
                            "harmonic_kerr_chain_L2_nph3_clean_weak|static_hea_qiskit_vqe"
                        ),
                        "lane": "comparator",
                        "family": "harmonic_kerr_chain",
                        "method": "HEA VQE",
                        "regime": "weak",
                        "algorithm_id": "static_hea_qiskit_vqe",
                        "source_case_id": "harmonic_kerr_chain_L2_nph2_clean_weak",
                        "source_stage": "nph2_ref4_screen",
                        "source_n_ph_work": 2,
                        "source_n_ph_ref": 4,
                        "source_status": "not_reached",
                        "source_threshold_status": "not_reached",
                        "source_record_id": (
                            "static_table__harmonic_kerr_chain__"
                            "harmonic_kerr_chain_L2_nph2_clean_weak__static_hea_qiskit_vqe"
                        ),
                        "source_payload_path": "raw_outputs/example/generic_static_single.json",
                        "source_payload_sha256": "abc123",
                        "source_audit_json": str(audit_path),
                        "source_audit_sha256": "def456",
                        "source_target_profile": "paper_i_phys_v1",
                        "source_threshold": 2e-4,
                        "next_stage": "nph3_ref4_escalation",
                        "next_stage_case_id": "harmonic_kerr_chain_L2_nph3_clean_weak",
                        "target_n_ph_work": 3,
                        "target_n_ph_ref": 4,
                        "target_suite_profile": "paper_i_clean_nph3_ref4_v1",
                        "escalation_reason": "completed_not_reached_phonon_ladder_row",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = generate_records(
        output_dir=tmp_path / "candidate_filtered",
        queue_output_root=tmp_path / "queue_candidate",
        paper_i_cutoff_ladder_stage="nph3_ref4_escalation",
        paper_i_ladder_benchmarks_only=True,
        paper_i_ladder_candidate_manifest=candidate_manifest,
    )

    assert summary["runnable_record_count"] == 1
    assert summary["paper_i_cutoff_ladder"]["candidate_manifest_filter"]["candidate_count"] == 1
    rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert len(rows) == 1
    row = rows[0]
    assert row["record_id"] == (
        "static_table__harmonic_kerr_chain__harmonic_kerr_chain_L2_nph3_clean_weak__"
        "static_hea_qiskit_vqe"
    )
    assert row["n_ph_work"] == "3"
    assert row["n_ph_ref"] == "4"
    assert row["paper_i_ladder_candidate_manifest_json"] == str(candidate_manifest)
    assert row["paper_i_ladder_source_case_id"] == "harmonic_kerr_chain_L2_nph2_clean_weak"
    assert row["paper_i_ladder_source_n_ph_work"] == "2"
    assert row["paper_i_ladder_source_n_ph_ref"] == "4"
    assert row["paper_i_ladder_escalation_reason"] == "completed_not_reached_phonon_ladder_row"


def test_generic_static_table_snake_only_generator_uses_matched_profile(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        snake_only=True,
    )

    assert summary["snake_only"] is True
    assert summary["include_snake"] is False
    assert summary["algorithm_ids"] == ["static_family_native_adapt_phase3"]
    assert summary["suite_profile"] == "nph2_ref3_v1"
    assert summary["energy_stop_target"] == 1e-8
    assert summary["first_hit_thresholds"] == [1e-6, 1e-8]
    assert summary["runnable_record_count"] == _table_i_executable_case_count("nph2_ref3_v1")
    assert summary["smoke_record_count"] == 1
    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    smoke_text = Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8")
    rows = list(csv.DictReader(records_text.splitlines(), delimiter="\t"))
    assert rows
    assert all(row["static_route_id"] == "route_a" for row in rows)
    assert summary["static_route_overlay"]["route_a_record_count"] == len(rows)
    assert summary["static_route_overlay"]["unspecified_record_count"] == 0
    assert "static_table__hh__hh_L2_nph2__static_family_native_adapt_phase3" in records_text
    assert "static_table__hh__hh_L2_nph2__static_family_native_adapt_phase3" in smoke_text
    assert "static_geo_adapt_vqe" not in records_text
    assert "\tnph2_ref3_v1\t1e-08\t1e-06,1e-08" in records_text


def test_generic_static_table_calibration_profile_hardware_resolution_is_diagnostic(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        calibration_profile="nph2_route_a_hk_hh_v1",
        hardware_resolution_mode="profile",
        hardware_resolution_profile_json="calibrations/small_noise.json",
        hardware_resolution_profile_name="small_noise_v1",
    )

    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    rows = list(csv.DictReader(records_text.splitlines(), delimiter="\t"))
    assert rows
    assert all(row["static_route_id"] == "unspecified" for row in rows)
    assert summary["calibration"]["expected_route_identity"] is None
    assert summary["calibration"]["canonical_route_a_expected"] is False
    assert summary["calibration"]["declared_static_route_ids"] == ["unspecified"]
    assert summary["calibration"]["diagnostic_hardware_profile_route"] is True


def test_generic_static_table_hardware_resolution_profile_generator_applies_only_to_phase3_static_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chtc.phase3_optuna.generate_generic_static_table_records as records_mod
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records
    from pipelines.reporting.benchmark_manifest import BenchmarkJob

    def fake_table_i_jobs(*, output_root, algorithm_ids, include_skipped):
        return (
            BenchmarkJob(
                job_id="phase3",
                domain="static",
                family="hh",
                case_id="hh_L2_nph2",
                algorithm_id="static_family_native_adapt_phase3",
                status="runnable",
                reason="",
                command=("python", "phase3"),
                output_dir=str(Path(output_root) / "phase3"),
                metadata={"dispatch": "phase3_static_adapt"},
            ),
            BenchmarkJob(
                job_id="hea",
                domain="static",
                family="hubbard",
                case_id="hubbard_L2",
                algorithm_id="static_hea_qiskit_vqe",
                status="runnable",
                reason="",
                command=("python", "hea"),
                output_dir=str(Path(output_root) / "hea"),
                metadata={"dispatch": "generic_static_hea_qiskit_vqe"},
            ),
        )

    monkeypatch.setattr(records_mod, "build_table_i_static_jobs", fake_table_i_jobs)
    monkeypatch.setattr(records_mod, "summarize_table_i_jobs", lambda jobs, **kwargs: {"job_count": len(tuple(jobs))})
    monkeypatch.setattr(records_mod, "_select_smoke_records", lambda records, **kwargs: (list(records), 0))

    summary = generate_records(
        output_dir=tmp_path / "input",
        include_snake=True,
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        hardware_resolution_mode="profile",
        hardware_resolution_profile_json="calibrations/small_noise.json",
        hardware_resolution_profile_name="small_noise_v1",
    )

    rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    phase3_rows = [row for row in rows if row["algorithm_id"] == "static_family_native_adapt_phase3"]
    non_phase3_rows = [row for row in rows if row["algorithm_id"] != "static_family_native_adapt_phase3"]

    assert phase3_rows
    assert summary["hardware_resolution_profile_overlay"] == {
        "requested": True,
        "applied": True,
        "applied_record_count": len(phase3_rows),
        "fields": {
            "hardware_resolution_mode": "profile",
            "hardware_resolution_profile_json": "calibrations/small_noise.json",
            "hardware_resolution_profile_name": "small_noise_v1",
        },
    }
    assert summary["static_route_overlay"] == {
        "applied_record_count": len(phase3_rows),
        "route_a_record_count": 0,
        "unspecified_record_count": len(phase3_rows),
        "hardware_profile_rows_marked_diagnostic": True,
    }
    assert all(row["hardware_resolution_mode"] == "profile" for row in phase3_rows)
    assert all(row["hardware_resolution_profile_json"] == "calibrations/small_noise.json" for row in phase3_rows)
    assert all(row["hardware_resolution_profile_name"] == "small_noise_v1" for row in phase3_rows)
    assert all(row["static_route_id"] == "unspecified" for row in phase3_rows)
    assert all(row["hardware_resolution_mode"] == "" for row in non_phase3_rows)
    assert all(row["hardware_resolution_profile_json"] == "" for row in non_phase3_rows)
    assert all(row["hardware_resolution_profile_name"] == "" for row in non_phase3_rows)
    assert all(row["static_route_id"] == "" for row in non_phase3_rows)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"hardware_resolution_profile_json": "calibrations/small_noise.json"},
        {"hardware_resolution_profile_name": "small_noise_v1"},
        {"hardware_resolution_mode": "profile"},
        {
            "hardware_resolution_mode": "ideal",
            "hardware_resolution_profile_json": "calibrations/small_noise.json",
            "hardware_resolution_profile_name": "small_noise_v1",
        },
    ),
)
def test_generic_static_table_hardware_resolution_profile_generator_partial_overlays_fail_closed(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="hardware_resolution"):
        generate_records(
            output_dir=tmp_path / "input",
            include_snake=True,
            queue_output_root=tmp_path / "queue",
            suite_profile="nph2_ref3_v1",
            **kwargs,
        )


def test_generate_records_hardware_resolution_profile_requires_phase3_static_rows(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="Hardware-resolution profile overlay requested.*include-snake"):
        generate_records(
            output_dir=tmp_path / "input",
            queue_output_root=tmp_path / "queue",
            suite_profile="nph2_ref3_v1",
            hardware_resolution_mode="profile",
            hardware_resolution_profile_json="calibrations/small_noise.json",
            hardware_resolution_profile_name="small_noise_v1",
        )


def test_generic_static_table_phase3_oracle_value_noise_generator_applies_only_to_phase3_static_records(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        include_snake=True,
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        phase3_oracle_gradient_mode="aer_density_matrix",
        phase3_oracle_backend_name="FakeGuadalupeV2",
        phase3_oracle_use_fake_backend=True,
        phase3_oracle_shots=64,
        phase3_oracle_repeats=1,
        phase3_oracle_aggregate="mean",
        phase3_oracle_seed=7,
        phase3_oracle_execution_surface="expectation_v1",
        phase3_oracle_inner_objective_mode="noisy_v1",
        phase3_oracle_value_noise_model="gaussian_iid_v1",
        phase3_oracle_value_noise_std=1e-6,
        phase3_oracle_value_noise_seed=20260514,
    )

    assert summary["phase3_oracle_overlay"]["requested"] is True
    assert summary["phase3_oracle_overlay"]["applied_record_count"] > 0
    assert summary["smoke_record_count"] == 15
    records_text = Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8")
    smoke_text = Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8")
    rows = list(csv.DictReader(records_text.splitlines(), delimiter="\t"))
    phase3_rows = [row for row in rows if row["algorithm_id"] == "static_family_native_adapt_phase3"]
    non_phase3_rows = [row for row in rows if row["algorithm_id"] != "static_family_native_adapt_phase3"]

    assert phase3_rows
    assert all(row["phase3_oracle_value_noise_model"] == "gaussian_iid_v1" for row in phase3_rows)
    assert all(row["phase3_oracle_value_noise_std"] == "1e-06" for row in phase3_rows)
    assert all(row["phase3_oracle_value_noise_seed"] == "20260514" for row in phase3_rows)
    assert all(row["phase3_oracle_value_noise_model"] == "" for row in non_phase3_rows)
    assert "static_table__hh__hh_L2_nph2__static_family_native_adapt_phase3" in smoke_text


def test_generic_static_table_benchmark_value_noise_generator_populates_all_runnable_rows(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        benchmark_value_noise_model="gaussian_iid_v1",
        benchmark_value_noise_std=2e-6,
        benchmark_value_noise_seed=20260514,
    )

    assert summary["benchmark_value_noise_overlay"] == {
        "requested": True,
        "applied": True,
        "applied_record_count": summary["runnable_record_count"],
        "semantic": "post_static_result_value_noise_not_physical_shots",
        "fields": {
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "2e-06",
            "benchmark_value_noise_seed": "20260514",
        },
    }
    rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    smoke_rows = list(csv.DictReader(Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert rows
    assert all(row["benchmark_value_noise_model"] == "gaussian_iid_v1" for row in rows)
    assert all(row["benchmark_value_noise_std"] == "2e-06" for row in rows)
    assert all(row["benchmark_value_noise_seed"] == "20260514" for row in rows)
    assert all(row["benchmark_decision_noise_model"] == "" for row in rows)
    assert all(row["phase3_oracle_value_noise_model"] == "" for row in rows)
    assert smoke_rows
    assert all(row["benchmark_value_noise_model"] == "gaussian_iid_v1" for row in smoke_rows)
    assert any(row["algorithm_id"] == "static_hea_qiskit_vqe" for row in rows)
    assert any(row["algorithm_id"] == "static_geo_adapt_vqe" for row in rows)


def test_generic_static_table_benchmark_decision_noise_generator_populates_only_non_phase3_rows(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        include_snake=True,
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        benchmark_decision_noise_model="gaussian_iid_v1",
        benchmark_decision_noise_std=3e-6,
        benchmark_decision_noise_seed=20260515,
    )

    rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    phase3_rows = [row for row in rows if row["algorithm_id"] == "static_family_native_adapt_phase3"]
    benchmark_rows = [row for row in rows if row["algorithm_id"] != "static_family_native_adapt_phase3"]
    assert phase3_rows
    assert benchmark_rows
    assert summary["benchmark_decision_noise_overlay"] == {
        "requested": True,
        "applied": True,
        "applied_record_count": len(benchmark_rows),
        "semantic": "benchmark_decision_value_noise_not_physical_shots_v1",
        "fields": {
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "3e-06",
            "benchmark_decision_noise_seed": "20260515",
        },
    }
    assert all(row["benchmark_decision_noise_model"] == "gaussian_iid_v1" for row in benchmark_rows)
    assert all(row["benchmark_decision_noise_std"] == "3e-06" for row in benchmark_rows)
    assert all(row["benchmark_decision_noise_seed"] == "20260515" for row in benchmark_rows)
    assert all(row["benchmark_decision_noise_model"] == "" for row in phase3_rows)
    assert all(row["phase3_oracle_value_noise_model"] == "" for row in rows)
    assert all(row["benchmark_value_noise_model"] == "" for row in rows)


def test_generate_records_benchmark_decision_noise_requires_non_phase3_rows(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="use phase3_oracle_value_noise"):
        generate_records(
            output_dir=tmp_path / "input",
            snake_only=True,
            queue_output_root=tmp_path / "queue",
            suite_profile="nph2_ref3_v1",
            benchmark_decision_noise_model="gaussian_iid_v1",
            benchmark_decision_noise_std=3e-6,
        )


def test_generic_static_table_phase3_budget_generator_weak_local_profile_is_smoke_scoped(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        include_snake=True,
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        phase3_smoke_budget_profile="weak_local_v1",
    )

    expected_budget = {
        "phase3_adapt_max_depth": "1",
        "phase3_adapt_maxiter": "1",
        "phase3_refit_maxiter": "1",
        "phase3_final_maxiter": "1",
        "phase3_adapt_spsa_a": "",
        "phase3_adapt_spsa_c": "",
        "phase3_adapt_spsa_big_a": "",
        "phase3_adapt_spsa_alpha": "",
        "phase3_adapt_spsa_gamma": "",
        "phase3_adapt_spsa_eval_repeats": "1",
        "phase3_adapt_spsa_avg_last": "0",
        "phase3_adapt_allow_repeats": "",
    }
    assert summary["phase3_budget_overlay"]["requested"] is True
    assert summary["phase3_budget_overlay"]["applied"] is True
    assert summary["phase3_budget_overlay"]["applied_record_count"] == 1
    assert summary["phase3_budget_overlay"]["profile"] == "weak_local_v1"
    assert summary["phase3_budget_overlay"]["fields"] == expected_budget
    assert summary["smoke_record_count"] == 15

    records_rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    smoke_rows = list(csv.DictReader(Path(summary["paths"]["smoke_records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    target_id = "static_table__hh__hh_L2_nph2__static_family_native_adapt_phase3"
    full_target = [row for row in records_rows if row["record_id"] == target_id]
    smoke_target = [row for row in smoke_rows if row["record_id"] == target_id]

    assert full_target
    assert smoke_target
    assert all(row[field] == "" for row in full_target for field in expected_budget)
    assert {field: smoke_target[0][field] for field in expected_budget} == expected_budget
    assert all(
        row[field] == ""
        for row in smoke_rows
        if row["record_id"] != target_id
        for field in expected_budget
    )


def test_generic_static_table_phase3_policy_profile_applies_prior_best_spsa_to_snake_rows(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    summary = generate_records(
        output_dir=tmp_path / "input",
        snake_only=True,
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        phase3_policy_profile="spsa_prior_best_v1",
        phase3_adapt_parallel_gradient_workers=2,
        phase3_adapt_beam_parent_workers=2,
    )

    expected_policy = {
        "phase3_adapt_max_depth": "",
        "phase3_adapt_maxiter": "8000",
        "phase3_refit_maxiter": "8000",
        "phase3_final_maxiter": "8000",
        "phase3_adapt_spsa_a": "0.05",
        "phase3_adapt_spsa_c": "0.02",
        "phase3_adapt_spsa_big_a": "50.0",
        "phase3_adapt_spsa_alpha": "0.602",
        "phase3_adapt_spsa_gamma": "0.101",
        "phase3_adapt_spsa_eval_repeats": "1",
        "phase3_adapt_spsa_avg_last": "0",
        "phase3_adapt_allow_repeats": "true",
    }
    assert summary["snake_only"] is True
    assert summary["phase3_policy_overlay"] == {
        "requested": True,
        "applied": True,
        "applied_record_count": summary["runnable_record_count"],
        "profile": "spsa_prior_best_v1",
        "fields": expected_policy,
    }
    assert summary["phase3_runtime_overlay"]["applied_record_count"] == summary["runnable_record_count"]
    assert summary["static_route_overlay"]["route_a_record_count"] == summary["runnable_record_count"]

    records_rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert records_rows
    assert all(row["algorithm_id"] == "static_family_native_adapt_phase3" for row in records_rows)
    assert all({field: row[field] for field in expected_policy} == expected_policy for row in records_rows)
    assert all(row["phase3_adapt_parallel_gradient_workers"] == "2" for row in records_rows)
    assert all(row["phase3_adapt_beam_parent_workers"] == "2" for row in records_rows)
    assert all(row["static_route_id"] == "route_a" for row in records_rows)
    assert all(row["phase2_novelty_mode"] == "collective_span_v1" for row in records_rows)


def test_generic_static_table_phase3_depth12_policy_json_and_collective_route_a_are_forwarded(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    policy_json = tmp_path / "input" / "policies" / "snake_route_a_depth12.json"
    policy_json.parent.mkdir(parents=True, exist_ok=True)
    policy_json.write_text(json.dumps({"policy": {"pool": {}, "static": {}, "inner_optimizer": {}}}), encoding="utf-8")

    summary = generate_records(
        output_dir=tmp_path / "input",
        snake_only=True,
        queue_output_root=tmp_path / "queue",
        suite_profile="nph2_ref3_v1",
        phase3_policy_profile="spsa_prior_depth12_v1",
        phase3_policy_json=str(policy_json),
        phase2_novelty_mode="collective_span_v1",
    )

    records_rows = list(csv.DictReader(Path(summary["paths"]["records_tsv"]).read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert records_rows
    assert summary["phase3_policy_overlay"]["fields"]["phase3_adapt_max_depth"] == "12"
    assert summary["phase3_policy_json_overlay"] == {
        "requested": True,
        "applied": True,
        "applied_record_count": summary["runnable_record_count"],
        "fields": {"phase3_policy_json": str(policy_json)},
    }
    assert summary["phase2_novelty_mode"] == "collective_span_v1"
    assert summary["static_route_overlay"]["route_a_record_count"] == summary["runnable_record_count"]
    assert summary["static_route_overlay"]["applied_record_count"] == summary["runnable_record_count"]
    assert all(row["phase3_adapt_max_depth"] == "12" for row in records_rows)
    assert all(row["phase3_policy_json"] == str(policy_json) for row in records_rows)
    assert all(row["phase2_novelty_mode"] == "collective_span_v1" for row in records_rows)
    assert all(row["static_route_id"] == "route_a" for row in records_rows)


def test_generate_records_phase3_budget_requires_phase3_static_records(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="Phase3 smoke budget overlay requested.*use --include-snake"):
        generate_records(
            output_dir=tmp_path / "input",
            include_snake=False,
            queue_output_root=tmp_path / "queue",
            suite_profile="nph2_ref3_v1",
            phase3_smoke_budget_profile="weak_local_v1",
        )


def test_generate_records_phase3_oracle_value_noise_seed_requires_enabled_model(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="value_noise_seed requires"):
        generate_records(
            output_dir=tmp_path / "input",
            include_snake=True,
            queue_output_root=tmp_path / "queue",
            suite_profile="nph2_ref3_v1",
            phase3_oracle_value_noise_seed=20260514,
        )


def test_generate_records_benchmark_value_noise_seed_requires_enabled_model(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="benchmark_value_noise_seed requires"):
        generate_records(
            output_dir=tmp_path / "input",
            queue_output_root=tmp_path / "queue",
            suite_profile="nph2_ref3_v1",
            benchmark_value_noise_seed=20260514,
        )


def test_generate_records_benchmark_decision_noise_seed_requires_enabled_model(tmp_path: Path) -> None:
    from chtc.phase3_optuna.generate_generic_static_table_records import generate_records

    with pytest.raises(ValueError, match="benchmark_decision_noise_seed requires"):
        generate_records(
            output_dir=tmp_path / "input",
            queue_output_root=tmp_path / "queue",
            suite_profile="nph2_ref3_v1",
            benchmark_decision_noise_seed=20260515,
        )


def test_run_single_benchmark_decision_noise_env_writes_unsupported_for_hidden_qiskit_adapt_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_qiskit_adapt_vqe as qadapt

    def forbidden_runner(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("decision-noise support must fail closed before Qiskit AdaptVQE runner")

    monkeypatch.setattr(qadapt, "run_static_qiskit_adapt_vqe_single", forbidden_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_STD", "1e-3")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_SEED", "20260515")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_qiskit_adapt_vqe",
        output_dir=tmp_path / "decision_noise",
    )

    row = payload["result"]
    assert payload["status"] == "skipped_unsupported_decision_noise"
    assert payload["benchmark_decision_noise_status"] == "unsupported"
    assert payload["benchmark_decision_noise"]["semantic"] == "benchmark_decision_value_noise_not_physical_shots_v1"
    assert payload["benchmark_decision_noise"]["model"] == "gaussian_iid_v1"
    assert payload["benchmark_decision_noise"]["std"] == pytest.approx(1e-3)
    assert payload["benchmark_decision_noise"]["seed"] == 20260515
    assert payload["benchmark_decision_noise"]["supported"] is False
    assert payload["benchmark_decision_noise"]["applied"] is False
    assert row["status"] == "skipped_unsupported_decision_noise"
    assert row["quality_gate_reason"] == "benchmark_decision_noise_unsupported"
    assert row["phase3_controller_called"] is False
    assert row["shots_total"] == 0
    assert row["compiled_depth_total"] == 0
    assert row["compiled_count_2q_total"] == 0
    assert row["benchmark_decision_noise_status"] == "unsupported"
    assert row["benchmark_decision_noise"]["reason"]
    for name in ("generic_static_single.json", "result.json", "manifest.json", "rows.json"):
        assert (tmp_path / "decision_noise" / name).exists()


def test_run_single_benchmark_decision_noise_env_threads_to_supported_hea_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    captured: dict[str, object] = {}

    def fake_runner(*, family: str, case_id: str, output_dir: Path, benchmark_decision_noise_config=None, **kwargs):  # noqa: ANN003
        captured["family"] = family
        captured["case_id"] = case_id
        captured["output_dir"] = output_dir
        captured["config"] = benchmark_decision_noise_config
        return {"schema": "fake_hea", "status": "completed", "rows": [{"status": "ok"}]}

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_STD", "1e-3")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_SEED", "20260515")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "hea_decision_noise",
    )

    assert payload["status"] == "completed"
    cfg = captured["config"]
    assert cfg.enabled is True
    assert cfg.seed == 20260515


def test_run_single_benchmark_decision_noise_env_threads_to_supported_family_vqe_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_family_informed_vqe as family_vqe

    captured: dict[str, object] = {}

    def fake_runner(*, family: str, case_id: str, output_dir: Path, benchmark_decision_noise_config=None, **kwargs):  # noqa: ANN003
        captured["family"] = family
        captured["case_id"] = case_id
        captured["output_dir"] = output_dir
        captured["config"] = benchmark_decision_noise_config
        return {"schema": "fake_family_vqe", "status": "completed", "rows": [{"status": "ok"}]}

    monkeypatch.setattr(family_vqe, "run_static_family_informed_vqe_single", fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_STD", "1e-3")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_SEED", "20260515")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_family_informed_vqe",
        output_dir=tmp_path / "family_decision_noise",
    )

    assert payload["status"] == "completed"
    cfg = captured["config"]
    assert cfg.enabled is True
    assert cfg.seed == 20260515


def test_run_single_benchmark_decision_noise_env_threads_to_supported_adapt_variant_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def fake_runner(
        *,
        family: str,
        case_id: str,
        algorithm_id: str,
        output_dir: Path,
        benchmark_decision_noise_config=None,
        **kwargs,
    ):  # noqa: ANN003, ANN201
        captured[algorithm_id] = {
            "family": family,
            "case_id": case_id,
            "output_dir": output_dir,
            "config": benchmark_decision_noise_config,
        }
        return {"schema": "fake_adapt_variant", "status": "completed", "rows": [{"status": "ok"}]}

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_STD", "1e-3")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_SEED", "20260515")

    for algorithm_id in variants.GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS:
        output_dir = tmp_path / algorithm_id
        payload = run_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id=algorithm_id,
            output_dir=output_dir,
        )

        assert payload["status"] == "completed"
        cfg = captured[algorithm_id]["config"]
        assert cfg.enabled is True
        assert cfg.seed == 20260515
        assert captured[algorithm_id]["output_dir"] == output_dir


def test_run_single_benchmark_decision_noise_env_threads_to_supported_hh_vqe_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipelines.exact_bench import hh_static_ground_state_benchmark as hhbench

    captured: dict[str, object] = {}

    def fake_hh_runner(*, output_dir, cases, algorithms, benchmark_decision_noise_config=None, **kwargs):  # noqa: ANN003
        captured["output_dir"] = output_dir
        captured["case_ids"] = [case.case_id for case in cases]
        captured["algorithm_ids"] = [algorithm.algorithm_id for algorithm in algorithms]
        captured["config"] = benchmark_decision_noise_config
        return {"schema": "fake_hh", "status": "completed", "rows": [{"status": "ok"}]}

    monkeypatch.setattr(hhbench, "run_hh_static_ground_state_benchmark", fake_hh_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_STD", "1e-3")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_SEED", "20260515")

    payload = run_single(
        family="hh",
        case_id="hh_L2_strong_canonical",
        algorithm_id="static_uccsd_vqe",
        output_dir=tmp_path / "hh_decision_noise",
    )

    assert payload["status"] == "completed"
    assert captured["case_ids"] == ["hh_L2_strong_canonical"]
    assert captured["algorithm_ids"] == ["hh_uccsd_lifted_vqe"]
    cfg = captured["config"]
    assert cfg.enabled is True
    assert cfg.seed == 20260515


def test_run_single_benchmark_decision_noise_env_rejects_phase3_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_STD", "1e-3")

    with pytest.raises(ValueError, match="use phase3_oracle_value_noise"):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id="static_family_native_adapt_phase3",
            output_dir=tmp_path / "controller",
        )


def test_phase3_static_table_contract_fields_use_phase3_compile_and_measurement_proxies() -> None:
    fields = _phase3_static_table_contract_fields_from_result(
        {
            "circuit_depth": 166,
            "count_2q": 65,
            "measurement_shots_proxy": 107.0,
            "shot_cost_proxy": 999.0,
        }
    )

    assert fields["phase3_controller_called"] is True
    assert fields["compiled_depth_total"] == 166
    assert fields["compiled_count_2q_total"] == 65
    assert fields["compiled_circuit_stats_status"] == "phase3_compile_json_metrics_v1"
    assert "shots_total" not in fields
    assert "static_shot_estimate_status" not in fields
    assert fields["legacy_work_proxy_status"] == "controller_proxy_diagnostic_only"
    assert fields["legacy_work_proxies"]["measurement_shots_proxy"] == 107.0
    assert fields["legacy_work_proxies"]["shot_cost_proxy"] == 999.0


def test_phase3_static_algorithmic_work_fields_use_native_controller_records(tmp_path: Path) -> None:
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "hamiltonian_pauli_term_count": 7,
                "adapt_vqe": {
                    "controller_measurement_work_summary": {
                        "schema": "controller_measurement_work_proxy_v1",
                        "source": "native_controller_live_decision_work_v1",
                        "source_kind": "native_controller_work",
                        "legacy_fallback_used": False,
                        "candidate_work_ledger_schema": "controller_candidate_work_ledger_v1",
                        "candidate_work_ledger_status": "explicit_candidate_work_ledger_v1",
                        "candidate_work_event_count": 3,
                        "candidate_work_missing_event_count": 0,
                        "candidate_count_total": 105,
                        "evaluated_count_total": 105,
                        "pre_shortlist_count_total": 105,
                        "shortlist_size_total": 105,
                        "retained_count_total": 105,
                        "rejected_count_total": 0,
                        "candidate_work_ledger_scope": "event_records_measured_v1",
                        "candidate_work_ledger_scopes": {"event_records_measured_v1": 3},
                        "by_phase": {
                            "phase1": {
                                "records_with_group_keys": 79,
                                "groups_total": 79,
                                "shots_total": 10_000,
                                "actual_operator_probe_count": 79,
                                "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
                            },
                            "phase2": {
                                "records_with_group_keys": 13,
                                "groups_total": 13,
                                "actual_operator_probe_count": 13,
                                "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
                            },
                            "phase3": {
                                "records_with_group_keys": 13,
                                "groups_total": 13,
                                "actual_operator_probe_count": 13,
                                "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
                            },
                        },
                    },
                    "history": [{"nfev_opt": 7}],
                    "resume_boundary_refit": {"executed": False},
                    "final_full_refit": {"executed": False},
                    "nfev_total": 7,
                    "continuation": {
                        "oracle_gradient_config": {
                            "value_noise": {
                                "enabled": True,
                                "model": "gaussian_iid_v1",
                                "std": 1.0,
                                "seed": 123,
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    fields = _phase3_static_algorithmic_work_fields_from_result(
        {"result_json": str(result_json), "measurement_shots_proxy": 999_999.0, "shots_total": 888_888},
        shots_per_pauli_term_proxy=10,
    )

    assert fields["algorithmic_measurement_work_source"] == "snake_canonical_runtime_reconstruction_v1"
    assert fields["S_alg_status"] == "ok"
    assert fields["S_alg"] == 112.0
    assert fields["S_alg_N_grad_probe"] == 79.0
    assert fields["S_alg_N_metric_probe"] == 26.0
    assert fields["S_alg_N_H_outer_eval"] == 0.0
    assert fields["S_alg_N_H_refit_eval"] == 7.0
    assert fields["S_alg_N_other_quantum"] == 0.0
    assert fields["snake_deterministic_shot_proxy"]["status"] == "ok"
    assert fields["snake_deterministic_shot_proxy"]["shots_per_pauli_term_proxy_source"] == (
        "argument.shots_per_pauli_term_proxy"
    )
    assert fields["shots_total"] == 10 * 7 * (7 + 79 + 26)
    assert fields["static_shot_estimate_status"] == "deterministic_proxy_not_physical_shots"
    assert fields["shots_total"] != 999_999
    assert fields["shots_total"] != 888_888
    ledger = fields["table_i_measurement_event_ledger"]
    assert ledger["schema"] == "table_i_measurement_event_ledger_v1"
    assert ledger["component_totals"]["N_grad_probe"] == 79.0
    assert ledger["component_totals"]["N_metric_probe"] == 26.0
    assert ledger["component_totals"]["N_H_refit_eval"] == 7.0


def test_phase3_static_algorithmic_work_fields_block_deterministic_shots_without_explicit_shots_per_term(
    tmp_path: Path,
) -> None:
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "hamiltonian_pauli_term_count": 7,
                "adapt_vqe": {
                    "controller_measurement_work_summary": {
                        "schema": "controller_measurement_work_proxy_v1",
                        "source": "native_controller_live_decision_work_v1",
                        "source_kind": "native_controller_work",
                        "legacy_fallback_used": False,
                        "candidate_work_ledger_schema": "controller_candidate_work_ledger_v1",
                        "candidate_work_ledger_status": "explicit_candidate_work_ledger_v1",
                        "candidate_work_event_count": 3,
                        "candidate_work_missing_event_count": 0,
                        "candidate_count_total": 105,
                        "evaluated_count_total": 105,
                        "pre_shortlist_count_total": 105,
                        "shortlist_size_total": 105,
                        "retained_count_total": 105,
                        "rejected_count_total": 0,
                        "candidate_work_ledger_scope": "event_records_measured_v1",
                        "candidate_work_ledger_scopes": {"event_records_measured_v1": 3},
                        "by_phase": {
                            "phase1": {
                                "records_with_group_keys": 79,
                                "groups_total": 79,
                                "shots_total": 10_000,
                                "actual_operator_probe_count": 79,
                                "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
                            },
                            "phase2": {
                                "records_with_group_keys": 13,
                                "groups_total": 13,
                                "actual_operator_probe_count": 13,
                                "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
                            },
                            "phase3": {
                                "records_with_group_keys": 13,
                                "groups_total": 13,
                                "actual_operator_probe_count": 13,
                                "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
                            },
                        },
                    },
                    "history": [{"nfev_opt": 7}],
                    "resume_boundary_refit": {"executed": False},
                    "final_full_refit": {"executed": False},
                    "nfev_total": 7,
                },
            }
        ),
        encoding="utf-8",
    )

    fields = _phase3_static_algorithmic_work_fields_from_result({"result_json": str(result_json)})

    assert fields["S_alg_status"] == "ok"
    assert fields["S_alg"] == 112.0
    assert fields["snake_deterministic_shot_proxy"]["status"] == "missing_shots_per_pauli_term_proxy"
    assert "shots_total" not in fields
    assert "static_shot_estimate_status" not in fields


def test_run_single_benchmark_value_noise_env_applies_to_non_phase3_result_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    def fake_run_static_hea_qiskit_vqe_single(*, family, case_id, output_dir):
        row = {
            "schema": "fake_static_hea_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": "static_hea_qiskit_vqe",
            "method_id": "static_hea_qiskit_vqe",
            "status": "ok",
            "energy": -1.0,
            "energy_ideal": -999.0,
            "exact_energy": -1.25,
            "exact_gs_energy": -1.25,
            "delta_E_abs": 0.25,
            "abs_delta_e": 0.25,
            "shots_total": 123,
            "phase3_controller_called": False,
            "compiled_depth_total": 2,
            "compiled_count_2q_total": 1,
        }
        payload = {
            "schema": "fake_static_hea_payload_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": "static_hea_qiskit_vqe",
            "status": "completed",
            "result": row,
            "rows": [dict(row)],
        }
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, content in {
            "generic_static_single.json": payload,
            "result.json": payload,
            "manifest.json": {"schema": "fake_manifest_v1", **payload},
            "rows.json": {"schema": "fake_rows_v1", "rows": [dict(row)]},
        }.items():
            (output / name).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", fake_run_static_hea_qiskit_vqe_single)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_VALUE_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_VALUE_NOISE_STD", "1e-3")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_VALUE_NOISE_SEED", "20260514")

    payload_a = run_single(
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "a",
    )
    payload_b = run_single(
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "b",
    )

    row_a = payload_a["result"]
    row_b = payload_b["result"]
    assert row_a["benchmark_value_noise_status"] == "ok"
    assert row_a["energy_ideal"] == -999.0
    assert row_a["energy_pre_benchmark_value_noise"] == -1.0
    assert row_a["benchmark_value_noise_energy_ideal"] == -1.0
    assert row_a["benchmark_value_noise"]["energy_pre_benchmark_value_noise"] == -1.0
    assert row_a["delta_E_abs_ideal"] == 0.25
    assert row_a["abs_delta_e_ideal"] == 0.25
    assert row_a["benchmark_value_noise"]["semantic"] == "post_static_result_value_noise_not_physical_shots"
    assert row_a["benchmark_value_noise"]["physical_shots_unchanged"] is True
    assert row_a["benchmark_value_noise"]["scope"]["algorithm_id"] == "static_hea_qiskit_vqe"
    assert row_a["benchmark_value_noise"]["noise_draw"] == pytest.approx(row_b["benchmark_value_noise"]["noise_draw"])
    assert row_a["energy"] == pytest.approx(row_b["energy"])
    assert row_a["energy"] == pytest.approx(-1.0 + row_a["benchmark_value_noise"]["noise_draw"])
    assert row_a["abs_delta_e"] == pytest.approx(abs(row_a["energy"] - (-1.25)))
    assert row_a["delta_E_abs"] == pytest.approx(row_a["abs_delta_e"])
    assert row_a["shots_total"] == 123
    assert payload_a["benchmark_value_noise"]["status"] == "ok"

    generic_payload = json.loads((tmp_path / "a" / "generic_static_single.json").read_text(encoding="utf-8"))
    result_payload = json.loads((tmp_path / "a" / "result.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    rows_payload = json.loads((tmp_path / "a" / "rows.json").read_text(encoding="utf-8"))
    assert generic_payload["result"]["benchmark_value_noise_status"] == "ok"
    assert result_payload["result"]["energy"] == pytest.approx(row_a["energy"])
    assert manifest_payload["result"]["energy_ideal"] == -999.0
    assert manifest_payload["result"]["benchmark_value_noise_energy_ideal"] == -1.0
    assert rows_payload["rows"][0]["benchmark_value_noise"]["noise_draw"] == pytest.approx(
        row_a["benchmark_value_noise"]["noise_draw"]
    )


def test_benchmark_value_noise_reapplication_with_changed_config_rebases_to_pre_noise_energy() -> None:
    row = {
        "family": "hh",
        "case_id": "hh_L2",
        "algorithm_id": "static_hea_qiskit_vqe",
        "status": "ok",
        "energy": -1.0,
        "exact_energy": -1.25,
        "delta_E_abs": 0.25,
        "abs_delta_e": 0.25,
    }
    config_a = {
        "enabled": True,
        "model": "gaussian_iid_v1",
        "std": 1e-3,
        "seed": 111,
        "seed_source": "test",
    }
    config_b = {
        "enabled": True,
        "model": "gaussian_iid_v1",
        "std": 1e-3,
        "seed": 222,
        "seed_source": "test",
    }

    assert _apply_benchmark_value_noise_to_row(
        row,
        config_a,
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
    ) == "ok"
    first_energy = row["energy"]
    first_draw = row["benchmark_value_noise"]["noise_draw"]
    assert first_energy == pytest.approx(-1.0 + first_draw)
    assert row["energy_pre_benchmark_value_noise"] == -1.0

    assert _apply_benchmark_value_noise_to_row(
        row,
        config_b,
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
    ) == "ok"
    second_draw = row["benchmark_value_noise"]["noise_draw"]
    assert row["benchmark_value_noise"]["seed"] == 222
    assert row["energy_pre_benchmark_value_noise"] == -1.0
    assert row["energy"] == pytest.approx(-1.0 + second_draw)
    assert row["energy"] != pytest.approx(first_energy + second_draw)



def test_run_single_benchmark_value_noise_env_requires_finite_energy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    def fake_run_static_hea_qiskit_vqe_single(*, family, case_id, output_dir):
        row = {
            "family": family,
            "case_id": case_id,
            "algorithm_id": "static_hea_qiskit_vqe",
            "status": "failed",
            "energy": None,
            "exact_energy": -1.25,
            "phase3_controller_called": False,
        }
        payload = {"schema": "fake_static_hea_payload_v1", "status": "failed", "result": row, "rows": [dict(row)]}
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "generic_static_single.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", fake_run_static_hea_qiskit_vqe_single)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_VALUE_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_BENCHMARK_VALUE_NOISE_STD", "1e-3")

    payload = run_single(
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "missing_energy",
    )

    assert payload["result"]["benchmark_value_noise_status"] == "missing_energy"
    assert payload["result"]["benchmark_value_noise"]["applied"] is False
    assert payload["benchmark_value_noise"]["status"] == "missing_energy"


def test_run_single_phase3_oracle_value_noise_env_applies_to_phase3_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    captured = {}

    def fake_run_static_benchmark(spec, policy, *, output_dir, **kwargs):
        captured["spec"] = spec
        captured["policy"] = policy
        return p3opt.BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=True,
            abs_delta_e=1e-6,
        )

    monkeypatch.setattr(p3opt, "run_static_benchmark", fake_run_static_benchmark)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_GRADIENT_MODE", "aer_density_matrix")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_BACKEND_NAME", "FakeGuadalupeV2")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_USE_FAKE_BACKEND", "true")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_SHOTS", "64")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_REPEATS", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_AGGREGATE", "mean")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_SEED", "7")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_EXECUTION_SURFACE", "expectation_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_INNER_OBJECTIVE_MODE", "noisy_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_VALUE_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_VALUE_NOISE_STD", "1e-6")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_VALUE_NOISE_SEED", "20260514")

    payload = run_single(
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_family_native_adapt_phase3",
        output_dir=tmp_path / "controller",
    )

    static = captured["policy"].static
    assert payload["phase3_oracle_env_overlay"]["phase3_oracle_value_noise_model"] == "gaussian_iid_v1"
    assert static.phase3_oracle_gradient_mode == "aer_density_matrix"
    assert static.phase3_oracle_backend_name == "FakeGuadalupeV2"
    assert static.phase3_oracle_use_fake_backend is True
    assert static.phase3_oracle_shots == 64
    assert static.phase3_oracle_inner_objective_mode == "noisy_v1"
    assert static.phase3_oracle_value_noise_std == pytest.approx(1e-6)
    assert static.phase3_oracle_value_noise_seed == 20260514


def test_run_single_hardware_resolution_profile_env_applies_to_phase3_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_benchmark as gsb
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    spec = p3opt.HamiltonianBenchmarkSpec(
        benchmark_id="hh_L2",
        family="hh",
        features=p3opt.ProblemFeatureVector(
            problem="hh",
            size_label="L2",
            L=2,
            n_qubits=4,
            pool_size_hint=100,
            spinful=True,
        ),
        base_pipeline_args=("--problem", "hh", "--L", "2", "--adapt-max-depth", "12"),
        baseline_abs_delta_e=1e-3,
        baseline_count_2q=100,
        baseline_depth_2q=300,
        baseline_parameter_count=20,
    )
    monkeypatch.setattr(gsb, "_phase3_static_spec_for_case", lambda **kwargs: spec)

    captured = {}

    def fake_run_static_benchmark(spec, policy, *, output_dir, **kwargs):
        captured["policy"] = policy
        return p3opt.BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=True,
            abs_delta_e=1e-6,
        )

    monkeypatch.setattr(p3opt, "run_static_benchmark", fake_run_static_benchmark)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_MODE", "profile")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_JSON", "calibrations/small_noise.json")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_NAME", "small_noise_v1")
    payload = run_single(
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_family_native_adapt_phase3",
        output_dir=tmp_path / "controller",
    )

    static = captured["policy"].static
    assert payload["hardware_resolution_profile_env_overlay"] == {
        "hardware_resolution_mode": "profile",
        "hardware_resolution_profile_json": "calibrations/small_noise.json",
        "hardware_resolution_profile_name": "small_noise_v1",
    }
    assert payload["static_route_env_overlay"] == {}
    assert static.static_route_id == "unspecified"
    assert static.hardware_resolution_mode == "profile"
    assert static.hardware_resolution_profile_json == "calibrations/small_noise.json"
    assert static.hardware_resolution_profile_name == "small_noise_v1"


def test_run_single_phase3_local_fixture_profile_dry_run_emits_hardware_resolution_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_name = "unit1g_synthetic_profile"
    profile_path, profile_manifest, selected_profile = _write_local_fixture_hardware_resolution_profile(
        tmp_path,
        name=profile_name,
        hw_floor=0.2,
        drift_floor=0.05,
    )
    env_fields = (
        "hardware_resolution_mode",
        "hardware_resolution_profile_json",
        "hardware_resolution_profile_name",
        "phase3_adapt_max_depth",
        "phase3_adapt_maxiter",
        "phase3_refit_maxiter",
        "phase3_final_maxiter",
        "phase3_adapt_spsa_a",
        "phase3_adapt_spsa_c",
        "phase3_adapt_spsa_big_a",
        "phase3_adapt_spsa_alpha",
        "phase3_adapt_spsa_gamma",
        "phase3_adapt_spsa_eval_repeats",
        "phase3_adapt_spsa_avg_last",
        "phase3_adapt_allow_repeats",
        "phase3_oracle_gradient_mode",
        "phase3_oracle_backend_name",
        "phase3_oracle_use_fake_backend",
        "phase3_oracle_shots",
        "phase3_oracle_repeats",
        "phase3_oracle_aggregate",
        "phase3_oracle_seed",
        "phase3_oracle_execution_surface",
        "phase3_oracle_inner_objective_mode",
        "phase3_oracle_value_noise_model",
        "phase3_oracle_value_noise_std",
        "phase3_oracle_value_noise_seed",
        "benchmark_value_noise_model",
        "benchmark_value_noise_std",
        "benchmark_value_noise_seed",
        "benchmark_decision_noise_model",
        "benchmark_decision_noise_std",
        "benchmark_decision_noise_seed",
    )
    for field in env_fields:
        monkeypatch.delenv(f"GENERIC_STATIC_TABLE_{field.upper()}", raising=False)
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.delenv("GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET", raising=False)
    monkeypatch.delenv("GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS", raising=False)

    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_MODE", "profile")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_JSON", str(profile_path))
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_NAME", profile_name)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAXITER", "20")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_REFIT_MAXITER", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_FINAL_MAXITER", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_A", "0.05")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_C", "0.02")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_BIG_A", "50.0")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_ALPHA", "0.602")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_GAMMA", "0.101")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_EVAL_REPEATS", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_AVG_LAST", "0")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS", "true")

    output_dir = tmp_path / "unit1g_local_profile_dry_run"
    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_family_native_adapt_phase3",
        output_dir=output_dir,
    )

    assert payload["schema"] == "generic_static_benchmark_phase3_single_v1"
    assert payload["hardware_resolution_profile_env_overlay"] == {
        "hardware_resolution_mode": "profile",
        "hardware_resolution_profile_json": str(profile_path),
        "hardware_resolution_profile_name": profile_name,
    }
    assert payload["static_route_env_overlay"] == {}
    assert payload["phase3_budget_env_overlay"] == {
        "adapt_max_depth": 1,
        "adapt_maxiter": 20,
        "refit_maxiter": 1,
        "final_maxiter": 1,
        "spsa_a": 0.05,
        "spsa_c": 0.02,
        "spsa_A": 50.0,
        "spsa_alpha": 0.602,
        "spsa_gamma": 0.101,
        "adapt_spsa_eval_repeats": 1,
        "adapt_spsa_avg_last": 0,
        "adapt_allow_repeats": True,
    }
    assert json.loads((output_dir / "generic_static_single.json").read_text(encoding="utf-8")) == payload

    result_json_raw = payload["result"].get("result_json")
    policy_json_raw = payload["result"].get("policy_json")
    assert result_json_raw
    assert policy_json_raw
    result_json = Path(str(result_json_raw))
    policy_json = Path(str(policy_json_raw))
    assert result_json.exists()
    assert policy_json.exists()
    policy_payload = json.loads(policy_json.read_text(encoding="utf-8"))
    static_policy = policy_payload["policy"]["static"]
    assert static_policy["static_route_id"] == "unspecified"
    assert static_policy["hardware_resolution_mode"] == "profile"
    assert static_policy["hardware_resolution_profile_json"] == str(profile_path)
    assert static_policy["hardware_resolution_profile_name"] == profile_name

    command_text = (result_json.parent.parent / "logs" / "command.sh").read_text(encoding="utf-8")
    assert "--static-route-id" not in command_text
    assert "--static-meta-feature-profile" not in command_text
    assert "--hardware-resolution-mode profile" in command_text
    assert "--hardware-resolution-profile-json" in command_text
    assert str(profile_path) in command_text
    assert f"--hardware-resolution-profile-name {profile_name}" in command_text
    assert "--adapt-max-depth 1" in command_text
    assert "--adapt-maxiter 20" in command_text
    assert "--adapt-spsa-eval-repeats 1" in command_text
    assert "--adapt-spsa-avg-last 0" in command_text

    result_payload = json.loads(result_json.read_text(encoding="utf-8"))
    adapt_payload = result_payload["adapt_vqe"]
    continuation = adapt_payload["continuation"]
    assert continuation["static_route_identity"]["route_id"] == "unspecified"
    assert continuation["static_route_identity"]["canonical_snake_eligible"] is False
    hardware = continuation["hardware_resolution"]
    assert hardware["schema"] == "gradient_resolution_v1"
    assert hardware["mode_requested"] == "profile"
    assert hardware["mode_effective"] == "manual"
    assert hardware["mode"] == "manual"
    assert hardware["floor_source"] == "profile_manifest"
    assert hardware["gradient_hw_floor"] == pytest.approx(0.2)
    assert hardware["gradient_drift_floor"] == pytest.approx(0.05)
    assert hardware["profile_name"] == profile_name
    assert hardware["profile_json"] == str(profile_path)
    assert hardware["profile_json_sha256"] == file_sha256(profile_path)
    assert hardware["profile_manifest_digest"] == digest_jsonable(profile_manifest)
    assert hardware["profile_digest"] == digest_jsonable({"name": profile_name, "profile": selected_profile})
    assert hardware["profile_manifest_schema"] == HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA
    assert hardware["profile_schema"] == HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA
    assert hardware["profile_units"] == HARDWARE_RESOLUTION_PROFILE_UNITS
    assert hardware["profile_provenance"]["source"] == "unit1g-local-synthetic-dry-run"

    assert continuation["phase0_pilot"]["satisfies_strict_route_a"] is False
    assert continuation["phase1"]["hardware_resolution_mode"] == "manual"
    assert continuation["phase1"]["gradient_hw_floor"] == pytest.approx(0.2)
    assert continuation["phase1"]["gradient_drift_floor"] == pytest.approx(0.05)
    assert continuation["phase2"]["hardware_resolution_mode"] == "manual"
    assert continuation["phase2"]["gradient_hw_floor"] == pytest.approx(0.2)
    assert continuation["phase2"]["gradient_drift_floor"] == pytest.approx(0.05)

    pilot_rows = continuation["phase0_last_pilot_rows"]
    assert pilot_rows
    pilot_row = pilot_rows[0]
    assert pilot_row["phase0_hardware_resolution_mode"] == "manual"
    assert pilot_row["phase0_hardware_resolution_source"] == "manual_scalar_floors"
    assert pilot_row["phase0_b_g_hw"] == pytest.approx(0.2)
    assert pilot_row["phase0_b_g_drift"] == pytest.approx(0.05)
    assert pilot_row["phase0_epsilon_g_res"] == pytest.approx(0.25)
    assert pilot_row["phase0_g_upper_hw"] == pytest.approx(
        float(pilot_row["phase0_raw_gradient_abs"]) + 0.25
    )

    scored_rows = continuation["phase2_scored_rows"]
    assert scored_rows
    scored_row = scored_rows[0]
    assert scored_row["hardware_resolution_mode"] == "manual"
    assert scored_row["hardware_resolution_source"] == "manual_scalar_floors"
    assert scored_row["b_g_hw"] == pytest.approx(0.2)
    assert scored_row["b_g_drift"] == pytest.approx(0.05)
    assert scored_row["epsilon_g_res"] == pytest.approx(0.25)
    assert scored_row["g_hw_lcb"] == pytest.approx(
        max(float(scored_row["g_abs"]) - float(scored_row["epsilon_g_res"]), 0.0)
    )


def test_run_single_hardware_resolution_profile_env_rejects_non_phase3_static_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_MODE", "profile")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_JSON", "calibrations/small_noise.json")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_NAME", "small_noise_v1")

    with pytest.raises(ValueError, match="hardware-resolution profile CHTC env overlay is only valid"):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "hea",
        )


@pytest.mark.parametrize(
    ("algorithm_id", "route_id"),
    (
        ("static_hea_qiskit_vqe", "route_a"),
        ("static_family_native_adapt_phase3", "route_a"),
        ("static_family_native_adapt_phase3", "unspecified"),
    ),
)
def test_run_single_static_route_env_is_retired_for_every_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    algorithm_id: str,
    route_id: str,
) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_STATIC_ROUTE_ID", route_id)

    with pytest.raises(
        ValueError,
        match="Historical static-route execution controls are retired",
    ):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id=algorithm_id,
            output_dir=tmp_path / "hea",
        )


@pytest.mark.parametrize(
    "env_updates",
    (
        {"GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_MODE": "profile"},
        {
            "GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_MODE": "profile",
            "GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_JSON": "calibrations/small_noise.json",
        },
        {
            "GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_JSON": "calibrations/small_noise.json",
            "GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_PROFILE_NAME": "small_noise_v1",
        },
    ),
)
def test_run_single_hardware_resolution_profile_env_partial_overlays_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_updates: dict[str, str],
) -> None:
    for name, value in env_updates.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="hardware_resolution|hardware-resolution"):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id="static_family_native_adapt_phase3",
            output_dir=tmp_path / "controller",
        )


def test_run_single_phase3_budget_env_applies_to_phase3_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    captured = {}

    def fake_run_static_benchmark(spec, policy, *, output_dir, **kwargs):
        captured["spec"] = spec
        captured["policy"] = policy
        return p3opt.BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=True,
            abs_delta_e=1e-6,
        )

    monkeypatch.setattr(p3opt, "run_static_benchmark", fake_run_static_benchmark)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAXITER", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_REFIT_MAXITER", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_FINAL_MAXITER", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_EVAL_REPEATS", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_AVG_LAST", "0")

    payload = run_single(
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_family_native_adapt_phase3",
        output_dir=tmp_path / "controller",
    )

    static = captured["policy"].static
    inner = captured["policy"].inner_optimizer
    args = tuple(captured["spec"].base_pipeline_args)
    assert payload["phase3_budget_env_overlay"] == {
        "adapt_max_depth": 1,
        "adapt_maxiter": 1,
        "refit_maxiter": 1,
        "final_maxiter": 1,
        "adapt_spsa_eval_repeats": 1,
        "adapt_spsa_avg_last": 0,
    }
    assert static.adapt_max_depth == 1
    assert static.adapt_maxiter == 1
    assert inner.refit_maxiter == 1
    assert inner.final_maxiter == 1
    assert args[args.index("--adapt-spsa-eval-repeats") + 1] == "1"
    assert args[args.index("--adapt-spsa-avg-last") + 1] == "0"


def test_run_single_phase3_runtime_env_applies_parallel_gradient_worker_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    captured = {}

    def fake_run_static_benchmark(spec, policy, *, output_dir, **kwargs):
        captured["spec"] = spec
        captured["policy"] = policy
        return p3opt.BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            family=spec.family,
            success=True,
            abs_delta_e=1e-6,
        )

    monkeypatch.setattr(p3opt, "run_static_benchmark", fake_run_static_benchmark)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_PARALLEL_GRADIENT_WORKERS", "2")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_BEAM_PARENT_WORKERS", "3")

    payload = run_single(
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_family_native_adapt_phase3",
        output_dir=tmp_path / "controller",
    )

    args = tuple(captured["spec"].base_pipeline_args)
    assert payload["phase3_budget_env_overlay"] == {}
    assert payload["phase3_runtime_env_overlay"] == {
        "adapt_parallel_gradient_workers": 2,
        "adapt_beam_parent_workers": 3,
    }
    assert "--adapt-parallel-gradient-workers" in args
    assert args[args.index("--adapt-parallel-gradient-workers") + 1] == "2"
    assert "--adapt-beam-parent-workers" in args
    assert args[args.index("--adapt-beam-parent-workers") + 1] == "3"
    assert captured["policy"].static.adapt_parallel_gradient_workers == 2
    assert captured["policy"].static.adapt_beam_parent_workers == 3


def test_run_single_phase3_runtime_env_reaches_nested_adapt_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    captured: dict[str, tuple[str, ...]] = {}

    def _option_value(command: tuple[str, ...], flag: str) -> str:
        assert flag in command
        return command[command.index(flag) + 1]

    def fake_run_subprocess_logged(
        command,
        *,
        stdout_path: Path,
        stderr_path: Path,
        subprocess_label: str = "subprocess",
        **kwargs,
    ):
        command_tuple = tuple(str(x) for x in command)
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stdout_path).write_text("", encoding="utf-8")
        Path(stderr_path).write_text("", encoding="utf-8")

        if subprocess_label == "adapt":
            workers = int(_option_value(command_tuple, "--adapt-parallel-gradient-workers"))
            captured["adapt_command"] = command_tuple
            output_json = Path(_option_value(command_tuple, "--output-json"))
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(
                    {
                        "adapt_vqe": {
                            "energy": -1.0,
                            "exact_gs_energy": -1.0,
                            "abs_delta_e": 0.0,
                            "ansatz_depth": 1,
                            "stop_reason": "max_depth",
                            "adapt_parallel_gradient_workers": workers,
                            "history": [
                                {
                                    "depth": 1,
                                    "energy_before_opt": -0.9,
                                    "max_grad": 0.1,
                                    "gradient_parallel_requested_workers": workers,
                                    "gradient_parallel_effective_workers": workers,
                                    "gradient_parallel_enabled": workers > 1,
                                    "gradient_parallel_mode": (
                                        "exact_noiseless_parallel" if workers > 1 else "serial"
                                    ),
                                }
                            ],
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return 0, 0.01

        if subprocess_label == "compile":
            output_json = Path(_option_value(command_tuple, "--output-json"))
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(
                    {
                        "selected_backend": {
                            "compiled_count_2q": 0,
                            "compiled_depth": 0,
                        },
                        "logical_circuit": {
                            "logical_parameter_count": 1,
                            "runtime_parameter_count": 1,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return 0, 0.01

        raise AssertionError(f"unexpected subprocess label {subprocess_label!r}")

    monkeypatch.setattr(p3opt, "_run_subprocess_logged", fake_run_subprocess_logged)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_PARALLEL_GRADIENT_WORKERS", "2")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_BEAM_PARENT_WORKERS", "3")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_family_native_adapt_phase3",
        output_dir=tmp_path / "controller",
    )

    command = captured["adapt_command"]
    assert command[command.index("--adapt-parallel-gradient-workers") + 1] == "2"
    assert command[command.index("--adapt-beam-parent-workers") + 1] == "3"
    assert payload["phase3_runtime_env_overlay"] == {
        "adapt_parallel_gradient_workers": 2,
        "adapt_beam_parent_workers": 3,
    }
    assert payload["status"] == "completed"

    nested_result = json.loads(Path(payload["result"]["result_json"]).read_text(encoding="utf-8"))
    adapt_payload = nested_result["adapt_vqe"]
    history_row = adapt_payload["history"][0]
    assert adapt_payload["adapt_parallel_gradient_workers"] == 2
    assert history_row["gradient_parallel_requested_workers"] == 2
    assert history_row["gradient_parallel_effective_workers"] == 2
    assert history_row["gradient_parallel_enabled"] is True


def test_run_single_phase3_budget_env_rejects_non_phase3_static_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH", "1")

    with pytest.raises(ValueError, match="budget.*only valid for phase3_static_adapt or generic_static_adapt_variants"):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "hea",
        )


@pytest.mark.parametrize(
    ("env_name", "value", "match"),
    (
        ("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAXITER", "0", ">= 1"),
        ("GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_AVG_LAST", "-1", ">= 0"),
    ),
)
def test_run_single_phase3_budget_env_validates_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_name: str,
    value: str,
    match: str,
) -> None:
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ValueError, match=match):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id="static_family_native_adapt_phase3",
            output_dir=tmp_path / "controller",
        )


def test_run_single_value_noise_env_rejects_non_phase3_static_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_GRADIENT_MODE", "aer_density_matrix")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_VALUE_NOISE_MODEL", "gaussian_iid_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_VALUE_NOISE_STD", "1e-6")

    with pytest.raises(ValueError, match="only valid for phase3_static_adapt"):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "hea",
        )


def test_run_single_phase3_oracle_value_noise_seed_env_requires_enabled_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ORACLE_VALUE_NOISE_SEED", "20260514")

    with pytest.raises(ValueError, match="value_noise_seed requires"):
        run_single(
            family="hh",
            case_id="hh_L2",
            algorithm_id="static_family_native_adapt_phase3",
            output_dir=tmp_path / "controller",
        )


def test_manifest_bundle_writes_jsonl_csv_and_summary(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path / "jobs-root",
        families=("hh",),
        algorithm_ids=("static_hea_qiskit_vqe",),
        include_skipped=False,
    )
    summary = write_manifest_bundle(output_dir=tmp_path / "manifest", jobs=jobs, label="test")

    assert summary["runnable_count"] == 1
    assert Path(summary["paths"]["jobs_jsonl"]).exists()
    assert Path(summary["paths"]["jobs_csv"]).exists()
    loaded = [json.loads(line) for line in Path(summary["paths"]["jobs_jsonl"]).read_text().splitlines()]
    assert len(loaded) == 1
    assert all(row["command_shell"] for row in loaded)


def test_external_adapt_manifest_metadata_is_provenance_tracked(tmp_path: Path) -> None:
    jobs = build_static_jobs(
        output_root=tmp_path,
        families=("hh", "hubbard"),
        algorithm_ids=("static_ceo_adapt_phase3", "static_tetris_adapt_phase3", "static_overlap_adapt_phase3"),
        include_skipped=True,
    )

    assert jobs
    ceo_hubbard = [job for job in jobs if job.family == "hubbard" and job.algorithm_id == "static_ceo_adapt_phase3"]
    tetris_hubbard = [
        job for job in jobs if job.family == "hubbard" and job.algorithm_id == "static_tetris_adapt_phase3"
    ]
    assert len(ceo_hubbard) == 1
    assert len(tetris_hubbard) == 3
    assert ceo_hubbard[0].status == "runnable"
    assert all(job.status == "runnable" for job in tetris_hubbard)
    assert ceo_hubbard[0].command
    assert all(job.command for job in tetris_hubbard)
    assert ceo_hubbard[0].metadata.get("external_adapt_dispatch") == "external_static_adapt_ceo_public_code"
    assert all(
        job.metadata.get("external_adapt_dispatch") == "external_static_adapt_tetris_public_code"
        for job in tetris_hubbard
    )

    runnable_external_hubbard_ids = {ceo_hubbard[0].job_id, *(job.job_id for job in tetris_hubbard)}
    for job in jobs:
        assert job.metadata.get("external_algorithm") is True
        assert job.metadata.get("phase3_controller_called") is False
        assert job.metadata.get("external_adapt_policy") == "do_not_emulate_through_phase3_controller"
        assert job.metadata.get("external_adapt_reference_ids")
        if job.job_id not in runnable_external_hubbard_ids:
            assert job.status == "skipped_not_implemented"
            assert not job.command


def test_run_single_hubbard_ceo_uses_external_adapter_not_phase3(monkeypatch, tmp_path: Path) -> None:
    import pipelines.exact_bench.external_adapt.external_static_adapt_benchmark as external
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    def _forbidden_phase3(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("external CEO benchmark must not call Phase3 static ADAPT")

    captured = {}

    def _fake_runner(*, family: str, case_id: str, algorithm_id: str, output_dir: Path):
        captured["family"] = family
        captured["case_id"] = case_id
        captured["algorithm_id"] = algorithm_id
        captured["output_dir"] = output_dir
        return {
            "schema": "external_static_adapt_benchmark_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": algorithm_id,
            "status": "completed",
            "guardrails": {"phase3_controller_called": False},
        }

    monkeypatch.setattr(p3opt, "run_static_benchmark", _forbidden_phase3)
    monkeypatch.setattr(external, "run_external_static_adapt_single", _fake_runner)

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_ceo_adapt_phase3",
        output_dir=tmp_path / "ceo",
    )

    assert payload["schema"] == "external_static_adapt_benchmark_v1"
    assert payload["status"] == "completed"
    assert captured == {
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "algorithm_id": "static_ceo_adapt_phase3",
        "output_dir": tmp_path / "ceo",
    }


def test_run_single_hubbard_tetris_uses_external_adapter_not_phase3(monkeypatch, tmp_path: Path) -> None:
    import pipelines.exact_bench.external_adapt.external_static_adapt_benchmark as external
    import pipelines.exact_bench.static_benchmark_runtime as p3opt

    def _forbidden_phase3(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("external TETRIS benchmark must not call Phase3 static ADAPT")

    captured = {}

    def _fake_runner(*, family: str, case_id: str, algorithm_id: str, output_dir: Path):
        captured["family"] = family
        captured["case_id"] = case_id
        captured["algorithm_id"] = algorithm_id
        captured["output_dir"] = output_dir
        return {
            "schema": "external_static_adapt_benchmark_v1",
            "family": family,
            "case_id": case_id,
            "algorithm_id": algorithm_id,
            "status": "completed",
            "dispatch": "external_static_adapt_tetris_public_code",
            "guardrails": {"phase3_controller_called": False, "tetris_row_promoted": True},
        }

    monkeypatch.setattr(p3opt, "run_static_benchmark", _forbidden_phase3)
    monkeypatch.setattr(external, "run_external_static_adapt_single", _fake_runner)

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_tetris_adapt_phase3",
        output_dir=tmp_path / "tetris",
    )

    assert payload["schema"] == "external_static_adapt_benchmark_v1"
    assert payload["status"] == "completed"
    assert payload["dispatch"] == "external_static_adapt_tetris_public_code"
    assert captured == {
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "algorithm_id": "static_tetris_adapt_phase3",
        "output_dir": tmp_path / "tetris",
    }


def test_run_single_external_adapt_skip_payload_has_metadata(tmp_path: Path) -> None:
    payload = run_single(
        family="hh",
        case_id="hh_L2_half_filling",
        algorithm_id="static_tetris_adapt_phase3",
        output_dir=tmp_path / "tetris",
    )

    assert payload["status"] == "skipped_not_implemented"
    assert payload["metadata"]["external_algorithm"] is True
    assert payload["metadata"]["phase3_controller_called"] is False
    assert (tmp_path / "tetris" / "skip.json").exists()
