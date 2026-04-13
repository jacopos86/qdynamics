from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.adapt_pipeline as hc_adapt
import pipelines.static_adapt.adapt_pipeline_legacy_20260322 as legacy_adapt
from pipelines.static_adapt.builders import hh_pool_presets, primitive_pools, problem_setup
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
)


def _assert_coeff_maps_match(lhs: dict[str, complex], rhs: dict[str, complex], tol: float = 1e-12) -> None:
    assert set(lhs) == set(rhs)
    for label in lhs:
        assert abs(complex(lhs[label]) - complex(rhs[label])) <= float(tol), label


def _assert_ansatz_terms_match(lhs_terms: list, rhs_terms: list, tol: float = 1e-12) -> None:
    assert len(lhs_terms) == len(rhs_terms)
    for lhs_term, rhs_term in zip(lhs_terms, rhs_terms):
        assert str(lhs_term.label) == str(rhs_term.label)
        _, lhs_map = problem_setup._collect_hardcoded_terms_exyz(lhs_term.polynomial)
        _, rhs_map = problem_setup._collect_hardcoded_terms_exyz(rhs_term.polynomial)
        _assert_coeff_maps_match(lhs_map, rhs_map, tol=tol)


def test_build_problem_hamiltonian_matches_hubbard_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="periodic",
    )
    direct = build_hubbard_hamiltonian(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.25,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_build_problem_hamiltonian_matches_hh_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.1,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    direct = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=0.1,
        v0=None,
        t_eval=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_exact_gs_energy_for_problem_matches_built_hh_hamiltonian_with_static_dv() -> None:
    h_poly = problem_setup.build_problem_hamiltonian(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=0.0,
        dv=1.0,
        omega0=0.5,
        g_ep=0.3535534,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    num_particles = half_filled_num_particles(2)
    e_dispatch = problem_setup._exact_gs_energy_for_problem(
        h_poly,
        problem="hh",
        num_sites=2,
        num_particles=num_particles,
        indexing="blocked",
        n_ph_max=1,
        boson_encoding="binary",
        t=1.0,
        u=0.0,
        dv=1.0,
        omega0=0.5,
        g_ep=0.3535534,
        boundary="open",
    )
    e_direct = exact_ground_energy_sector_hh(
        h_poly,
        num_sites=2,
        num_particles=num_particles,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )
    assert abs(e_dispatch - e_direct) < 1e-10


def test_build_hva_pool_matches_layerwise_hh_drive_convention() -> None:
    dv = 0.1
    pool = primitive_pools._build_hva_pool(
        num_sites=2,
        t=1.0,
        u=4.0,
        omega0=1.0,
        g_ep=0.5,
        dv=dv,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    direct_layerwise = HubbardHolsteinLayerwiseAnsatz(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        v=None,
        v_t=dv,
        v0=None,
        t_eval=None,
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )
    _assert_ansatz_terms_match(pool[: len(direct_layerwise.base_terms)], direct_layerwise.base_terms)


def test_legacy_wrapper_patches_static_dv_convention_in_loaded_source() -> None:
    source = legacy_adapt._load_legacy_source()
    assert "v_t=float(dv)," in source
    assert "v_t=float(args.dv)," in source
    assert "v_t=None,\n        v0=float(dv)," not in source
    assert "v_t=None,\n            v0=float(args.dv)," not in source


def test_adapt_pipeline_reexports_extracted_helpers() -> None:
    assert hc_adapt._collect_hardcoded_terms_exyz is problem_setup._collect_hardcoded_terms_exyz
    assert hc_adapt._build_uccsd_pool is primitive_pools._build_uccsd_pool
    assert hc_adapt._build_paop_pool is primitive_pools._build_paop_pool
    assert hc_adapt._build_hh_full_meta_pool is hh_pool_presets._build_hh_full_meta_pool
    assert hc_adapt._exact_gs_energy_for_problem is problem_setup._exact_gs_energy_for_problem
    assert hc_adapt._exact_reference_state_for_hh is problem_setup._exact_reference_state_for_hh
    assert hc_adapt.build_hh_pool_by_key is hh_pool_presets.build_hh_pool_by_key


def test_build_hh_pool_by_key_matches_direct_full_meta_builder() -> None:
    num_particles = half_filled_num_particles(2)
    h_poly = problem_setup.build_problem_hamiltonian(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    direct_pool, direct_meta = hh_pool_presets._build_hh_full_meta_pool(
        h_poly=h_poly,
        num_sites=2,
        t=1.0,
        u=4.0,
        omega0=1.0,
        g_ep=0.5,
        dv=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=2,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        num_particles=num_particles,
    )
    dispatch_pool, method_name, class_meta, label_meta = hh_pool_presets.build_hh_pool_by_key(
        pool_key_hh="full_meta",
        h_poly=h_poly,
        num_sites=2,
        t=1.0,
        u=4.0,
        omega0=1.0,
        g_ep=0.5,
        dv=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=2,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        num_particles=num_particles,
    )
    assert method_name == "hardcoded_adapt_vqe_full_meta"
    assert class_meta is None
    assert label_meta is None
    assert direct_meta["raw_total"] >= len(direct_pool)
    assert len(dispatch_pool) == len(direct_pool)
    assert [term.label for term in dispatch_pool] == [term.label for term in direct_pool]
