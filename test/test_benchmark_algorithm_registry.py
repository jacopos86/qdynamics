#!/usr/bin/env python3
"""Tests for benchmark family/algorithm applicability decisions."""

from __future__ import annotations

from pipelines.exact_bench.benchmark_algorithm_registry import (
    compatibility_matrix,
    default_benchmark_algorithms,
    evaluate_algorithm_for_family,
    get_benchmark_algorithm,
)


_NON_HH_STATIC_ED_REFERENCE_FAMILIES = (
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)

_STATIC_HEA_QISKIT_RUNNABLE_FAMILIES = (
    "hh",
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)

_STATIC_QISKIT_ADAPTVQE_RUNNABLE_FAMILIES = ("hh",) + _NON_HH_STATIC_ED_REFERENCE_FAMILIES
_STATIC_ADAPT_VARIANT_RUNNABLE_FAMILIES = ("hh",) + _NON_HH_STATIC_ED_REFERENCE_FAMILIES


def test_registry_contains_static_and_dynamics_algorithms() -> None:
    static_ids = {alg.algorithm_id for alg in default_benchmark_algorithms(domain="static")}
    dynamics_ids = {alg.algorithm_id for alg in default_benchmark_algorithms(domain="dynamics")}

    assert "static_hea_qiskit_vqe" in static_ids
    assert "static_lang_firsov_vqe" in static_ids
    assert "static_family_informed_vqe" in static_ids
    assert "static_family_native_adapt_phase3" in static_ids
    assert "static_append_only_adapt_phase3" in static_ids
    assert "static_qiskit_adapt_vqe" in static_ids
    assert "static_full_meta_append_adapt_vqe" in static_ids
    assert "static_qubit_qeb_adapt_vqe" in static_ids
    assert "static_tetris_qubit_adapt_vqe" not in static_ids
    assert "static_geo_qubit_adapt_vqe" in static_ids
    assert "static_geo_qeb_adapt_vqe" in static_ids
    assert "static_geo_adapt_vqe" in static_ids
    assert "static_qubit_qeb_adapt_phase3" not in static_ids
    assert "static_tetris_ceo_style_adapt_phase3" not in static_ids
    assert "dyn_product_formula_envelope" in dynamics_ids
    assert "dyn_qiskit_trotter_qrte" in dynamics_ids
    assert "dyn_qiskit_pvqd" in dynamics_ids
    assert "dyn_qiskit_varqrte" in dynamics_ids
    assert "dyn_vff_like" in dynamics_ids
    assert "dyn_controller_full" in dynamics_ids
    assert "dyn_controller_ablation_matrix" in dynamics_ids


def test_hh_specific_algorithms_do_not_apply_to_pure_hubbard() -> None:
    lang = evaluate_algorithm_for_family("static_lang_firsov_vqe", "hubbard")
    qeb = evaluate_algorithm_for_family("static_qeb_sq_lf_adapt", "hubbard")

    assert lang.status == "skipped_unsupported"
    assert "not meaningful" in lang.reason
    assert qeb.status == "skipped_unsupported"


def test_static_hea_qiskit_vqe_is_runnable_for_first_slices() -> None:
    for family in _STATIC_HEA_QISKIT_RUNNABLE_FAMILIES:
        app = evaluate_algorithm_for_family("static_hea_qiskit_vqe", family)
        assert app.status == "runnable", family
        assert app.runner_module == "pipelines.exact_bench.generic_static_benchmark"
        assert app.hamiltonian_generic is True

    vff = evaluate_algorithm_for_family("dyn_vff_like", "hh")
    assert vff.status == "runnable"
    assert vff.diagnostic is True


def test_static_ed_reference_is_runnable_for_all_non_hh_generic_families() -> None:
    for family in _NON_HH_STATIC_ED_REFERENCE_FAMILIES:
        app = evaluate_algorithm_for_family("static_ed_reference", family)
        assert app.status == "runnable", family
        assert app.runner_module == "pipelines.exact_bench.generic_static_benchmark"
        assert app.qpu_faithful is False
        assert app.exact_assisted is True
        assert app.diagnostic is True
        assert app.hamiltonian_generic is True

    hh_app = evaluate_algorithm_for_family("static_ed_reference", "hh")
    assert hh_app.status == "skipped_not_implemented"
    assert hh_app.runner_module == "pipelines.exact_bench.generic_static_benchmark"


def test_project_controller_and_append_only_rows_are_runnable_for_non_hh() -> None:
    native = evaluate_algorithm_for_family("static_family_native_adapt_phase3", "hubbard")
    append_only = evaluate_algorithm_for_family("static_append_only_adapt_phase3", "hubbard")
    ceo = evaluate_algorithm_for_family("static_ceo_adapt_phase3", "hubbard")

    assert native.status == "runnable"
    assert native.resolved_pool_key == "full_meta"
    assert append_only.status == "runnable"
    assert append_only.resolved_pool_key == "full_meta"
    assert append_only.hamiltonian_generic is True
    assert ceo.status == "runnable"
    assert ceo.resolved_pool_key == "uccsd"


def test_static_qiskit_adapt_vqe_is_runnable_for_non_hh_first_slices() -> None:
    for family in _STATIC_QISKIT_ADAPTVQE_RUNNABLE_FAMILIES:
        app = evaluate_algorithm_for_family("static_qiskit_adapt_vqe", family)
        assert app.status == "runnable", family
        assert app.runner_module == "pipelines.exact_bench.generic_static_benchmark"
        assert app.required_pool_key == "full_meta"
        assert app.resolved_pool_key == "full_meta"
        assert app.exact_assisted is False
        assert app.diagnostic is False
        assert app.hamiltonian_generic is True


def test_generic_static_adapt_variants_are_runnable_for_non_hh_first_slices() -> None:
    expected_pool = {
        "static_full_meta_append_adapt_vqe": "full_meta",
        "static_qubit_qeb_adapt_vqe": None,
        "static_geo_qubit_adapt_vqe": "full_meta",
        "static_geo_qeb_adapt_vqe": None,
        "static_geo_adapt_vqe": "full_meta",
    }
    for algorithm_id, pool_key in expected_pool.items():
        for family in _STATIC_ADAPT_VARIANT_RUNNABLE_FAMILIES:
            app = evaluate_algorithm_for_family(algorithm_id, family)
            assert app.status == "runnable", (algorithm_id, family)
            assert app.runner_module == "pipelines.exact_bench.generic_static_benchmark"
            assert app.required_pool_key == pool_key
            assert app.resolved_pool_key == pool_key
            assert app.exact_assisted is False
            assert app.diagnostic is False
            assert app.hamiltonian_generic is True
            if algorithm_id == "static_full_meta_append_adapt_vqe":
                assert app.required_pool_key == app.resolved_pool_key == "full_meta"

    old_geo = next(alg for alg in default_benchmark_algorithms(domain="static") if alg.algorithm_id == "static_geo_qubit_adapt_vqe")
    new_geo = next(alg for alg in default_benchmark_algorithms(domain="static") if alg.algorithm_id == "static_geo_qeb_adapt_vqe")
    same_pool_geo = next(alg for alg in default_benchmark_algorithms(domain="static") if alg.algorithm_id == "static_geo_adapt_vqe")
    assert old_geo.display_name == "legacy geometry diagnostic (removed from Table I)"
    assert old_geo.required_pool_key == "full_meta"
    assert new_geo.display_name == "Geo-ADAPT-VQE (QEB reference)"
    assert new_geo.required_pool_key is None
    assert same_pool_geo.display_name == "Geo-ADAPT-VQE"
    assert same_pool_geo.required_pool_key == "full_meta"


def test_family_informed_fixed_vqe_row_is_runnable_with_full_meta_pool() -> None:
    for family in _STATIC_QISKIT_ADAPTVQE_RUNNABLE_FAMILIES:
        app = evaluate_algorithm_for_family("static_family_informed_vqe", family)
        assert app.status == "runnable", family
        assert app.runner_module == "pipelines.exact_bench.generic_static_benchmark"
        assert app.hamiltonian_generic is True
        assert app.required_pool_key == "full_meta"
        assert app.resolved_pool_key == "full_meta"


def test_static_comparator_registry_declares_source_roles() -> None:
    hea = get_benchmark_algorithm("static_hea_qiskit_vqe")
    family = get_benchmark_algorithm("static_family_informed_vqe")
    append = get_benchmark_algorithm("static_full_meta_append_adapt_vqe")
    snake = get_benchmark_algorithm("static_family_native_adapt_phase3")

    assert hea.execution_surface_role == "primary_execution_surface"
    assert hea.external_reference_status == "primary_execution_surface"
    assert family.algorithm_origin == "benchmark_local_fixed_ansatz_statevector_vqe"
    assert family.external_reference_status == "parity_surface"
    assert append.parity_reference_algorithm_id == "static_qiskit_adapt_vqe"
    assert snake.external_reference_status is None
    assert snake.algorithm_origin == "repo_native_phase3_static_adapt_snake"


def test_generic_dynamics_rows_are_runnable_for_supported_non_hh_families() -> None:
    exact_hubbard = evaluate_algorithm_for_family("dyn_exact_reference", "hubbard")
    fixed_spin_boson = evaluate_algorithm_for_family("dyn_fixed_mclachlan", "spin_boson")
    pf_hubbard = evaluate_algorithm_for_family("dyn_product_formula_envelope", "hubbard")
    qdrift_spin_boson = evaluate_algorithm_for_family("dyn_qdrift", "spin_boson")
    fixed_pvqd_hubbard = evaluate_algorithm_for_family("dyn_fixed_pvqd", "hubbard")
    adaptive_pvqd_hubbard = evaluate_algorithm_for_family("dyn_adaptive_pvqd", "hubbard")
    avqds_spin_boson = evaluate_algorithm_for_family("dyn_avqds", "spin_boson")
    avqds_t_hubbard = evaluate_algorithm_for_family("dyn_avqds_t", "hubbard")
    avqds_t_extended = evaluate_algorithm_for_family("dyn_avqds_t", "extended_hubbard")
    controller_full_hubbard = evaluate_algorithm_for_family("dyn_controller_full", "hubbard")
    controller_full_hh = evaluate_algorithm_for_family("dyn_controller_full", "hh")
    controller_ablation_hubbard = evaluate_algorithm_for_family(
        "dyn_controller_ablation_matrix", "hubbard"
    )
    controller_ablation_hh = evaluate_algorithm_for_family("dyn_controller_ablation_matrix", "hh")
    qiskit_trotter_hubbard = evaluate_algorithm_for_family("dyn_qiskit_trotter_qrte", "hubbard")
    qiskit_pvqd_hubbard = evaluate_algorithm_for_family("dyn_qiskit_pvqd", "hubbard")
    qiskit_varqrte_hubbard = evaluate_algorithm_for_family("dyn_qiskit_varqrte", "hubbard")

    assert exact_hubbard.status == "runnable"
    assert exact_hubbard.exact_assisted is True
    assert exact_hubbard.diagnostic is True
    assert exact_hubbard.qpu_faithful is False
    assert fixed_spin_boson.status == "runnable"
    assert fixed_spin_boson.exact_assisted is False
    assert fixed_spin_boson.diagnostic is True
    assert fixed_spin_boson.qpu_faithful is True
    assert pf_hubbard.status == "runnable"
    assert pf_hubbard.exact_assisted is False
    assert pf_hubbard.diagnostic is True
    assert pf_hubbard.qpu_faithful is True
    assert qdrift_spin_boson.status == "runnable"
    assert qdrift_spin_boson.exact_assisted is False
    assert qdrift_spin_boson.diagnostic is True
    assert qdrift_spin_boson.qpu_faithful is True
    assert fixed_pvqd_hubbard.status == "runnable"
    assert fixed_pvqd_hubbard.exact_assisted is False
    assert fixed_pvqd_hubbard.diagnostic is True
    assert fixed_pvqd_hubbard.qpu_faithful is True
    assert adaptive_pvqd_hubbard.status == "runnable"
    assert adaptive_pvqd_hubbard.exact_assisted is False
    assert adaptive_pvqd_hubbard.diagnostic is True
    assert adaptive_pvqd_hubbard.qpu_faithful is True
    assert avqds_spin_boson.status == "runnable"
    assert avqds_spin_boson.exact_assisted is False
    assert avqds_spin_boson.diagnostic is True
    assert avqds_spin_boson.qpu_faithful is True
    assert avqds_t_hubbard.status == "runnable"
    assert avqds_t_hubbard.exact_assisted is False
    assert avqds_t_hubbard.diagnostic is True
    assert avqds_t_hubbard.qpu_faithful is True
    assert avqds_t_extended.status == "runnable"
    for qiskit_row in (qiskit_trotter_hubbard, qiskit_pvqd_hubbard, qiskit_varqrte_hubbard):
        assert qiskit_row.status == "runnable"
        assert qiskit_row.exact_assisted is False
        assert qiskit_row.diagnostic is True
        assert qiskit_row.qpu_faithful is True
        assert qiskit_row.hamiltonian_generic is True
    assert controller_full_hubbard.status == "runnable"
    assert controller_full_hubbard.exact_assisted is False
    assert controller_full_hubbard.diagnostic is True
    assert controller_full_hubbard.qpu_faithful is True
    assert controller_full_hh.status == "runnable"
    assert controller_full_hh.exact_assisted is False
    assert controller_full_hh.diagnostic is True
    assert controller_full_hh.qpu_faithful is True
    assert controller_ablation_hubbard.status == "runnable"
    assert controller_ablation_hubbard.exact_assisted is False
    assert controller_ablation_hubbard.diagnostic is True
    assert controller_ablation_hubbard.qpu_faithful is True
    assert controller_ablation_hh.status == "runnable"
    assert controller_ablation_hh.exact_assisted is False
    assert controller_ablation_hh.diagnostic is True
    assert controller_ablation_hh.qpu_faithful is True


def test_static_hea_qiskit_vqe_promotes_truncated_boson_first_slices() -> None:
    for family in ("spin_boson", "bose_hubbard", "harmonic_kerr_chain"):
        app = evaluate_algorithm_for_family("static_hea_qiskit_vqe", family)
        assert app.status == "runnable", family
        assert app.runner_module == "pipelines.exact_bench.generic_static_benchmark"
        assert app.hamiltonian_generic is True

    vff = evaluate_algorithm_for_family("dyn_vff_like", "hubbard")
    assert vff.status == "skipped_unsupported"


def test_external_ceo_and_tetris_promote_only_hubbard_public_code_slice() -> None:
    ceo_hubbard = evaluate_algorithm_for_family("static_ceo_adapt_phase3", "hubbard")
    ceo_hh = evaluate_algorithm_for_family("static_ceo_adapt_phase3", "hh")
    tetris_hubbard = evaluate_algorithm_for_family("static_tetris_adapt_phase3", "hubbard")
    tetris_hh = evaluate_algorithm_for_family("static_tetris_adapt_phase3", "hh")

    assert ceo_hubbard.status == "runnable"
    assert ceo_hubbard.runner_module == "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark"
    assert ceo_hubbard.resolved_pool_key == "uccsd"
    assert ceo_hh.status == "skipped_not_implemented"
    assert tetris_hubbard.status == "runnable"
    assert tetris_hubbard.runner_module == "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark"
    assert tetris_hubbard.resolved_pool_key == "uccsd"
    assert tetris_hh.status == "skipped_not_implemented"

    for family in ("hh", "hubbard"):
        app = evaluate_algorithm_for_family("static_overlap_adapt_phase3", family)
        assert app.status == "skipped_not_implemented"


def test_incompatible_operator_family_is_rejected() -> None:
    uccsd_spin_boson = evaluate_algorithm_for_family("static_uccsd_vqe", "spin_boson")

    assert uccsd_spin_boson.status == "skipped_unsupported"
    assert uccsd_spin_boson.runner_module is not None


def test_compatibility_matrix_covers_requested_families() -> None:
    matrix = compatibility_matrix(
        families=("hh", "hubbard"),
        domain="static",
    )
    assert len(matrix) == 2 * len(default_benchmark_algorithms(domain="static"))
    assert {row.family for row in matrix} == {"hh", "hubbard"}


def test_external_adapt_rows_point_to_benchmark_local_runner() -> None:
    for algorithm_id in (
        "static_ceo_adapt_phase3",
        "static_tetris_adapt_phase3",
        "static_overlap_adapt_phase3",
    ):
        app = evaluate_algorithm_for_family(algorithm_id, "hh")
        assert app.status == "skipped_not_implemented"
        assert app.runner_module == "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark"
        assert app.resolved_pool_key is not None

    hubbard_ceo = evaluate_algorithm_for_family("static_ceo_adapt_phase3", "hubbard")
    hubbard_tetris = evaluate_algorithm_for_family("static_tetris_adapt_phase3", "hubbard")
    assert hubbard_ceo.status == "runnable"
    assert hubbard_ceo.runner_module == "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark"
    assert hubbard_tetris.status == "runnable"
    assert hubbard_tetris.runner_module == "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark"
