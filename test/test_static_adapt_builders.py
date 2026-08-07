from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.static_adapt.adapt_pipeline as hc_adapt
from pipelines.contracts.static_provenance import summarize_static_physical_operator_pool_labels
from pipelines.static_adapt.builders import hh_pool_presets, primitive_pools, problem_setup
from pipelines.static_adapt.builders.legal_subspace_filter import LEGAL_SUBSPACE_FILTER_SCHEMA
from pipelines.static_adapt.builders.lattice_hamiltonians import (
    build_extended_hubbard_blocks,
    build_extended_hubbard_hva_terms,
    build_extended_hubbard_hamiltonian,
    build_extended_hubbard_quadratures,
    build_ionic_hubbard_blocks,
    build_ionic_hubbard_hva_terms,
    build_ionic_hubbard_hamiltonian,
    build_ionic_hubbard_quadratures,
    build_spinless_tv_blocks,
    build_spinless_tv_hva_terms,
    build_spinless_tv_hamiltonian,
    build_spinless_tv_quadratures,
    build_ttprime_hubbard_blocks,
    build_ttprime_hubbard_hva_terms,
    build_ttprime_hubbard_hamiltonian,
    build_ttprime_hubbard_quadratures,
)
from src.quantum.operator_pools.boson_chains import (
    build_bose_hubbard_blocks,
    build_bose_hubbard_hamiltonian,
    build_bose_hubbard_hva_terms,
    build_bose_hubbard_quadratures,
    build_harmonic_kerr_chain_blocks,
    build_harmonic_kerr_chain_hamiltonian,
    build_harmonic_kerr_chain_hva_terms,
    build_harmonic_kerr_chain_quadratures,
)
from src.quantum.operator_pools.spin_boson import (
    build_spin_boson_blocks,
    build_spin_boson_hamiltonian,
    build_spin_boson_hva_terms,
    build_spin_boson_quadratures,
)
from src.quantum.chemistry.molecular_hamiltonian import (
    build_one_body_jw_polynomial,
    build_two_body_jw_polynomial,
)
from src.quantum.chemistry.psi4_adapter import (
    load_restricted_closed_shell_problem_from_json,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    HubbardTermwiseAnsatz,
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


def _write_molecular_problem_json(tmp_path: Path) -> Path:
    payload: dict[str, Any] = {
        "geometry_spec": "H 0.0 0.0 0.0\nH 0.0 0.0 0.7414",
        "basis": "sto-3g",
        "charge": 0,
        "multiplicity": 1,
        "reference": "rhf",
        "n_spatial_orbitals": 2,
        "n_spin_orbitals": 4,
        "n_alpha": 1,
        "n_beta": 1,
        "hf_energy": -1.1166843871,
        "nuclear_repulsion_energy": 0.7151043391,
        "one_body_integrals_mo": [
            [-1.252477303982, 0.0],
            [0.0, -0.475934275355],
        ],
        "two_body_integrals_mo": [
            [
                [
                    [0.674493166181, 0.0],
                    [0.0, 0.181287518779],
                ],
                [
                    [0.0, 0.181287518779],
                    [0.181287518779, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.181287518779],
                    [0.181287518779, 0.0],
                ],
                [
                    [0.6634721010, 0.0],
                    [0.0, 0.6973980100],
                ],
            ],
        ],
    }
    path = tmp_path / "molecular_problem.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def test_build_problem_hamiltonian_rejects_unknown_problem_family() -> None:
    try:
        problem_setup.build_problem_hamiltonian(
            problem_key="not_a_real_problem",
            num_sites=2,
            t=0.0,
            u=0.0,
            dv=0.0,
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    except ValueError as exc:
        assert "Unsupported problem family" in str(exc)
    else:
        raise AssertionError("Expected unsupported family to raise.")


def test_build_problem_hamiltonian_matches_ionic_hubbard_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="ionic_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    direct = build_ionic_hubbard_hamiltonian(
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        ordering="blocked",
        boundary="open",
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_build_problem_hamiltonian_matches_extended_hubbard_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="extended_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        v_nn=1.5,
    )
    direct = build_extended_hubbard_hamiltonian(
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_build_problem_hamiltonian_matches_ttprime_hubbard_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="ttprime_hubbard",
        num_sites=4,
        t=1.0,
        u=4.0,
        dv=0.25,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        t_prime=0.4,
    )
    direct = build_ttprime_hubbard_hamiltonian(
        num_sites=4,
        t=1.0,
        u=4.0,
        dv=0.25,
        t_prime=0.4,
        ordering="blocked",
        boundary="open",
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_build_problem_hamiltonian_matches_spin_boson_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="spin_boson",
        num_sites=1,
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    direct = build_spin_boson_hamiltonian(
        num_sites=1,
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        include_zero_point=True,
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_build_problem_hamiltonian_matches_bose_hubbard_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="bose_hubbard",
        num_sites=2,
        t=0.7,
        u=0.4,
        dv=0.2,
        omega0=1.0,
        g_ep=0.0,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    direct = build_bose_hubbard_hamiltonian(
        num_sites=2,
        t=0.7,
        u=0.4,
        dv=0.2,
        omega0=1.0,
        n_ph_max=2,
        boson_encoding="binary",
        boundary="open",
        include_zero_point=True,
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_build_problem_hamiltonian_matches_harmonic_kerr_chain_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="harmonic_kerr_chain",
        num_sites=2,
        t=0.5,
        u=0.3,
        dv=0.1,
        omega0=1.0,
        g_ep=0.0,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    direct = build_harmonic_kerr_chain_hamiltonian(
        num_sites=2,
        t=0.5,
        u=0.3,
        dv=0.1,
        omega0=1.0,
        n_ph_max=2,
        boson_encoding="binary",
        boundary="open",
        include_zero_point=True,
    )
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _, direct_map = problem_setup._collect_hardcoded_terms_exyz(direct)
    _assert_coeff_maps_match(wrapped_map, direct_map)


def test_build_problem_hamiltonian_matches_spinless_tv_builder() -> None:
    wrapped = problem_setup.build_problem_hamiltonian(
        problem_key="spinless_tv",
        num_sites=4,
        t=1.0,
        u=0.0,
        dv=0.1,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        v_nn=1.5,
    )
    direct = build_spinless_tv_hamiltonian(
        num_sites=4,
        t=1.0,
        v_nn=1.5,
        dv=0.1,
        boundary="open",
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


def test_exact_gs_energy_for_problem_matches_spinless_sector_helper() -> None:
    h_poly = problem_setup.build_problem_hamiltonian(
        problem_key="spinless_tv",
        num_sites=4,
        t=1.0,
        u=0.0,
        dv=0.1,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        v_nn=1.5,
    )
    e_dispatch = problem_setup._exact_gs_energy_for_problem(
        h_poly,
        problem="spinless_tv",
        num_sites=4,
        num_particles=(2, 0),
        indexing="blocked",
        n_ph_max=0,
        boson_encoding="binary",
        t=1.0,
        u=0.0,
        dv=0.1,
        v_nn=1.5,
        omega0=0.0,
        g_ep=0.0,
        boundary="open",
    )
    assert np.isfinite(float(e_dispatch))


def test_build_hamiltonian_blocks_pool_matches_hubbard_termwise_generators() -> None:
    pool = primitive_pools._build_hamiltonian_blocks_pool(
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
    direct = HubbardTermwiseAnsatz(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.25,
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
        include_potential_terms=True,
    )
    assert [str(term.label) for term in pool] == [f"ham_block::{term.label}" for term in direct.base_terms]
    _assert_ansatz_terms_match(pool, [
        type(term)(label=f"ham_block::{term.label}", polynomial=term.polynomial)
        for term in direct.base_terms
    ])
    assert any(len(term.polynomial.return_polynomial()) > 1 for term in pool)


def test_build_hamiltonian_blocks_pool_matches_hh_physical_termwise_generators() -> None:
    pool = primitive_pools._build_hamiltonian_blocks_pool(
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
    direct = HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        v=None,
        v_t=0.1,
        v0=None,
        t_eval=None,
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )
    assert [str(term.label) for term in pool] == [f"ham_block::{term.label}" for term in direct.base_terms]
    _assert_ansatz_terms_match(pool, [
        type(term)(label=f"ham_block::{term.label}", polynomial=term.polynomial)
        for term in direct.base_terms
    ])
    assert any(len(term.polynomial.return_polynomial()) > 1 for term in pool)


def test_build_hamiltonian_blocks_pool_matches_molecular_grouped_builders(
    tmp_path: Path,
) -> None:
    molecular_problem = load_restricted_closed_shell_problem_from_json(
        _write_molecular_problem_json(tmp_path)
    )
    pool = primitive_pools._build_hamiltonian_blocks_pool(
        problem_key="molecular_restricted_closed_shell",
        num_sites=2,
        t=0.0,
        u=0.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        molecular_problem=molecular_problem,
    )
    direct = [
        type(pool[0])(
            label="ham_block::molecular_one_body",
            polynomial=build_one_body_jw_polynomial(molecular_problem, ordering="blocked"),
        ),
        type(pool[0])(
            label="ham_block::molecular_two_body",
            polynomial=build_two_body_jw_polynomial(molecular_problem, ordering="blocked"),
        ),
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any(len(term.polynomial.return_polynomial()) > 1 for term in pool)


def test_build_hamiltonian_blocks_pool_matches_ionic_grouped_builders() -> None:
    pool = primitive_pools._build_hamiltonian_blocks_pool(
        problem_key="ionic_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_block::{label}", polynomial=poly)
        for label, poly in build_ionic_hubbard_blocks(
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.25,
            ordering="blocked",
            boundary="open",
        )
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


def test_build_hamiltonian_blocks_pool_matches_extended_grouped_builders() -> None:
    pool = primitive_pools._build_hamiltonian_blocks_pool(
        problem_key="extended_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_block::{label}", polynomial=poly)
        for label, poly in build_extended_hubbard_blocks(
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.25,
            v_nn=1.5,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


def test_build_hamiltonian_blocks_pool_matches_ttprime_grouped_builders() -> None:
    pool = primitive_pools._build_hamiltonian_blocks_pool(
        problem_key="ttprime_hubbard",
        num_sites=4,
        t=1.0,
        u=4.0,
        dv=0.25,
        t_prime=0.4,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_block::{label}", polynomial=poly)
        for label, poly in build_ttprime_hubbard_blocks(
            num_sites=4,
            t=1.0,
            u=4.0,
            dv=0.25,
            t_prime=0.4,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


def test_build_hamiltonian_blocks_pool_matches_spin_boson_grouped_builders() -> None:
    pool = primitive_pools._build_hamiltonian_blocks_pool(
        problem_key="spin_boson",
        num_sites=1,
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    direct = [
        type(pool[0])(label=f"ham_block::{label}", polynomial=poly)
        for label, poly in build_spin_boson_blocks(
            num_sites=1,
            t=0.7,
            u=0.4,
            dv=0.3,
            omega0=1.0,
            g_ep=0.6,
            n_ph_max=2,
            boson_encoding="binary",
            include_zero_point=True,
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


def test_build_hamiltonian_blocks_pool_matches_spinless_grouped_builders() -> None:
    pool = primitive_pools._build_hamiltonian_blocks_pool(
        problem_key="spinless_tv",
        num_sites=4,
        t=1.0,
        u=0.0,
        dv=0.1,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_block::{label}", polynomial=poly)
        for label, poly in build_spinless_tv_blocks(
            num_sites=4,
            t=1.0,
            v_nn=1.5,
            dv=0.1,
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


def test_build_family_hva_pool_matches_ionic_physical_primitives() -> None:
    pool = primitive_pools._build_family_hva_pool(
        problem_key="ionic_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"hva_term::{label}", polynomial=poly)
        for label, poly in build_ionic_hubbard_hva_terms(
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.25,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


def test_build_family_hva_pool_matches_extended_physical_primitives() -> None:
    pool = primitive_pools._build_family_hva_pool(
        problem_key="extended_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"hva_term::{label}", polynomial=poly)
        for label, poly in build_extended_hubbard_hva_terms(
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.25,
            v_nn=1.5,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any("pair_hop" in str(term.label) for term in pool)
    assert any("exchange_" in str(term.label) for term in pool)


def test_build_family_hva_pool_matches_ttprime_physical_primitives() -> None:
    pool = primitive_pools._build_family_hva_pool(
        problem_key="ttprime_hubbard",
        num_sites=4,
        t=1.0,
        u=4.0,
        dv=0.25,
        t_prime=0.4,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"hva_term::{label}", polynomial=poly)
        for label, poly in build_ttprime_hubbard_hva_terms(
            num_sites=4,
            t=1.0,
            u=4.0,
            dv=0.25,
            t_prime=0.4,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any("opp_spin_assist" in str(term.label) for term in pool)
    assert any("three_site_bridge" in str(term.label) for term in pool)


def test_build_family_hva_pool_matches_spinless_physical_primitives() -> None:
    pool = primitive_pools._build_family_hva_pool(
        problem_key="spinless_tv",
        num_sites=4,
        t=1.0,
        u=0.0,
        dv=0.1,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"hva_term::{label}", polynomial=poly)
        for label, poly in build_spinless_tv_hva_terms(
            num_sites=4,
            t=1.0,
            v_nn=1.5,
            dv=0.1,
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ] 
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any("three_site_bridge" in str(term.label) for term in pool)


def test_build_family_max_pool_unions_spinful_lattice_generators() -> None:
    pool = primitive_pools._build_family_max_pool(
        problem_key="extended_hubbard",
        num_sites=2,
        num_particles=tuple(half_filled_num_particles(2)),
        t=1.0,
        u=4.0,
        dv=0.25,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    labels = [str(term.label) for term in pool]
    assert any(label.startswith("uccsd_") for label in labels)
    assert any(label.startswith("ham_quad::") for label in labels)
    assert any(label.startswith("ham_block::") for label in labels)
    assert any("pair_hop" in label for label in labels)
    assert any("exchange_" in label for label in labels)
    sigs = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
    assert len(sigs) == len(set(sigs))


def test_build_family_max_pool_unions_spin_boson_generators() -> None:
    pool = primitive_pools._build_family_max_pool(
        problem_key="spin_boson",
        num_sites=1,
        num_particles=(0, 0),
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    labels = [str(term.label) for term in pool]
    assert not any(label.startswith("uccsd_") for label in labels)
    assert any(label.startswith("ham_quad::") for label in labels)
    assert any(label.startswith("ham_block::") for label in labels)
    assert sum(label.startswith(("ham_quad::", "ham_block::", "hva_term::")) for label in labels) >= 2
    sigs = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
    assert len(sigs) == len(set(sigs))


def test_build_full_meta_pool_prepends_ham_full_for_static_spinful_lattice_families() -> None:
    cases = [
        (
            "ionic_hubbard",
            2,
            build_ionic_hubbard_hamiltonian(
                num_sites=2,
                t=1.0,
                u=4.0,
                dv=0.25,
                ordering="blocked",
                boundary="open",
            ),
            {"v_nn": 0.0, "t_prime": 0.0},
        ),
        (
            "extended_hubbard",
            2,
            build_extended_hubbard_hamiltonian(
                num_sites=2,
                t=1.0,
                u=4.0,
                dv=0.25,
                v_nn=1.5,
                ordering="blocked",
                boundary="open",
            ),
            {"v_nn": 1.5, "t_prime": 0.0},
        ),
        (
            "ttprime_hubbard",
            3,
            build_ttprime_hubbard_hamiltonian(
                num_sites=3,
                t=1.0,
                u=4.0,
                dv=0.25,
                t_prime=0.2,
                ordering="blocked",
                boundary="open",
            ),
            {"v_nn": 0.0, "t_prime": 0.2},
        ),
    ]

    for problem_key, num_sites, h_poly, extra in cases:
        pool = primitive_pools._build_full_meta_pool(
            problem_key=problem_key,
            h_poly=h_poly,
            num_sites=int(num_sites),
            num_particles=half_filled_num_particles(int(num_sites)),
            t=1.0,
            u=4.0,
            dv=0.25,
            v_nn=float(extra["v_nn"]),
            t_prime=float(extra["t_prime"]),
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
        )
        labels = [str(term.label) for term in pool]
        assert labels[0] == "ham_full"
        assert labels.count("ham_full") == 1
        assert any(label.startswith("ham_term(") for label in labels)
        sigs = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
        assert len(sigs) == len(set(sigs))


def test_build_full_meta_pool_prepends_ham_full_for_spinless_tv() -> None:
    h_poly = build_spinless_tv_hamiltonian(
        num_sites=4,
        t=1.0,
        v_nn=1.5,
        dv=0.1,
        boundary="open",
    )
    pool = primitive_pools._build_full_meta_pool(
        problem_key="spinless_tv",
        h_poly=h_poly,
        num_sites=4,
        num_particles=(2, 0),
        t=1.0,
        u=0.0,
        dv=0.1,
        v_nn=1.5,
        t_prime=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    labels = [str(term.label) for term in pool]
    assert labels[0] == "ham_full"
    assert labels.count("ham_full") == 1
    assert any(label.startswith("ham_term(") for label in labels)


def test_build_full_meta_pool_filters_spin_boson_generators_and_full_hamiltonian() -> None:
    h_poly = build_spin_boson_hamiltonian(
        num_sites=1,
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        include_zero_point=True,
    )
    pool, filter_meta = primitive_pools._build_full_meta_pool(
        problem_key="spin_boson",
        h_poly=h_poly,
        num_sites=1,
        num_particles=(0, 0),
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        return_legal_subspace_filter_meta=True,
    )
    labels = [str(term.label) for term in pool]
    assert labels[0] == "ham_full"
    assert labels.count("ham_full") == 1
    assert filter_meta["schema"] == LEGAL_SUBSPACE_FILTER_SCHEMA
    assert filter_meta["active"] is True
    assert filter_meta["original_pool_size"] == 59
    assert filter_meta["filtered_pool_size"] == len(pool) == 43
    assert filter_meta["pre_dedup_filtered_pool_size"] == 43
    assert filter_meta["grouped_legal_count"] == 43
    assert filter_meta["execution_legal_count"] == 43
    assert filter_meta["termwise_component_risk_generator_count"] == 41
    assert filter_meta["kept_with_component_risk_count"] == 25
    assert filter_meta["sanitized_generator_count"] == 25
    assert filter_meta["grouped_exact_execution_generator_count"] == 25
    assert filter_meta["dropped_generator_count"] == 16
    assert filter_meta["post_filter_duplicate_generator_count"] == 0
    assert filter_meta["termwise_component_risk_count"] == 160
    assert filter_meta["termwise_component_filtered_count"] == 0
    assert "full_meta::boson_number" in labels
    assert "full_meta::boson_x" not in labels
    assert "full_meta::boson_displacement" in labels
    assert any(label.startswith("ham_term(") for label in labels)
    sigs = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
    assert len(sigs) == len(set(sigs))


def test_build_full_meta_pool_filters_bose_hubbard_generators_and_full_hamiltonian() -> None:
    h_poly = build_bose_hubbard_hamiltonian(
        num_sites=2,
        t=0.7,
        u=0.4,
        dv=0.2,
        omega0=1.0,
        n_ph_max=2,
        boson_encoding="binary",
        boundary="open",
        include_zero_point=True,
    )
    pool, filter_meta = primitive_pools._build_full_meta_pool(
        problem_key="bose_hubbard",
        h_poly=h_poly,
        num_sites=2,
        num_particles=(1, 1),
        t=0.7,
        u=0.4,
        dv=0.2,
        omega0=1.0,
        g_ep=0.0,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        return_legal_subspace_filter_meta=True,
    )
    labels = [str(term.label) for term in pool]
    assert filter_meta["schema"] == LEGAL_SUBSPACE_FILTER_SCHEMA
    assert filter_meta["active"] is True
    assert filter_meta["original_pool_size"] == 78
    assert filter_meta["filtered_pool_size"] == len(pool) == 46
    assert filter_meta["grouped_legal_count"] == 46
    assert filter_meta["execution_legal_count"] == 46
    assert filter_meta["termwise_component_risk_generator_count"] == 63
    assert filter_meta["kept_with_component_risk_count"] == 31
    assert filter_meta["sanitized_generator_count"] == 31
    assert filter_meta["grouped_exact_execution_generator_count"] == 31
    assert filter_meta["termwise_product_execution_generator_count"] == 15
    assert filter_meta["dropped_generator_count"] == 32
    assert filter_meta["post_filter_duplicate_generator_count"] == 0
    assert filter_meta["termwise_component_risk_count"] == 360
    assert filter_meta["termwise_component_filtered_count"] == 0
    assert "full_meta::n_0" in labels
    assert "full_meta::nn_0_1" in labels
    assert "full_meta::n_x_0" in labels
    assert "full_meta::density_hop_0_1_left" in labels
    assert "full_meta::pair_hop_0_1" in labels
    assert any(label.startswith("ham_term(") for label in labels)


def test_build_full_meta_pool_filters_harmonic_kerr_chain_generators_and_full_hamiltonian() -> None:
    h_poly = build_harmonic_kerr_chain_hamiltonian(
        num_sites=2,
        t=0.5,
        u=0.3,
        dv=0.1,
        omega0=1.0,
        n_ph_max=2,
        boson_encoding="binary",
        boundary="open",
        include_zero_point=True,
    )
    pool, filter_meta = primitive_pools._build_full_meta_pool(
        problem_key="harmonic_kerr_chain",
        h_poly=h_poly,
        num_sites=2,
        num_particles=(1, 1),
        t=0.5,
        u=0.3,
        dv=0.1,
        omega0=1.0,
        g_ep=0.0,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        return_legal_subspace_filter_meta=True,
    )
    labels = [str(term.label) for term in pool]
    assert filter_meta["schema"] == LEGAL_SUBSPACE_FILTER_SCHEMA
    assert filter_meta["active"] is True
    assert filter_meta["original_pool_size"] == 61
    assert filter_meta["filtered_pool_size"] == len(pool) == 37
    assert filter_meta["grouped_legal_count"] == 37
    assert filter_meta["execution_legal_count"] == 37
    assert filter_meta["termwise_component_risk_generator_count"] == 49
    assert filter_meta["kept_with_component_risk_count"] == 25
    assert filter_meta["sanitized_generator_count"] == 25
    assert filter_meta["grouped_exact_execution_generator_count"] == 25
    assert filter_meta["termwise_product_execution_generator_count"] == 12
    assert filter_meta["dropped_generator_count"] == 24
    assert filter_meta["post_filter_duplicate_generator_count"] == 0
    assert filter_meta["termwise_component_risk_count"] == 196
    assert filter_meta["termwise_component_filtered_count"] == 0
    assert "full_meta::n_0" in labels
    assert "full_meta::kerr_1" in labels
    assert "full_meta::n_x_0" in labels
    assert "full_meta::x_p_sym_1" in labels
    assert any(label.startswith("ham_term(") for label in labels)


def test_build_family_max_pool_unions_spinless_lattice_generators() -> None:
    pool = primitive_pools._build_family_max_pool(
        problem_key="spinless_tv",
        num_sites=4,
        num_particles=(2, 0),
        t=1.0,
        u=0.0,
        dv=0.1,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    labels = [str(term.label) for term in pool]
    assert not any(label.startswith("uccsd_") for label in labels)
    assert any(label.startswith("ham_quad::") for label in labels)
    assert any(label.startswith("ham_block::") for label in labels)
    assert any("three_site_bridge" in label for label in labels)
    sigs = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
    assert len(sigs) == len(set(sigs))


def test_build_hamiltonian_quadratures_pool_matches_ionic_quadratures() -> None:
    pool = primitive_pools._build_hamiltonian_quadratures_pool(
        problem_key="ionic_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_quad::{label}", polynomial=poly)
        for label, poly in build_ionic_hubbard_quadratures(
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.25,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any("quadrature" in str(term.label) for term in pool)


def test_build_hamiltonian_quadratures_pool_matches_extended_quadratures() -> None:
    pool = primitive_pools._build_hamiltonian_quadratures_pool(
        problem_key="extended_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_quad::{label}", polynomial=poly)
        for label, poly in build_extended_hubbard_quadratures(
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.25,
            v_nn=1.5,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any("pair_hop" in str(term.label) for term in pool)
    assert any("exchange_" in str(term.label) for term in pool)


def test_build_hamiltonian_quadratures_pool_matches_ttprime_quadratures() -> None:
    pool = primitive_pools._build_hamiltonian_quadratures_pool(
        problem_key="ttprime_hubbard",
        num_sites=4,
        t=1.0,
        u=4.0,
        dv=0.25,
        t_prime=0.4,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_quad::{label}", polynomial=poly)
        for label, poly in build_ttprime_hubbard_quadratures(
            num_sites=4,
            t=1.0,
            u=4.0,
            dv=0.25,
            t_prime=0.4,
            ordering="blocked",
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any("opp_spin_assist" in str(term.label) for term in pool)
    assert any("three_site_bridge" in str(term.label) for term in pool)


def test_build_family_hva_pool_matches_spin_boson_physical_primitives() -> None:
    pool = primitive_pools._build_family_hva_pool(
        problem_key="spin_boson",
        num_sites=1,
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    direct = [
        type(pool[0])(label=f"hva_term::{label}", polynomial=poly)
        for label, poly in build_spin_boson_hva_terms(
            num_sites=1,
            t=0.7,
            u=0.4,
            dv=0.3,
            omega0=1.0,
            g_ep=0.6,
            n_ph_max=2,
            boson_encoding="binary",
            include_zero_point=True,
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


def test_build_hamiltonian_quadratures_pool_matches_spinless_quadratures() -> None:
    pool = primitive_pools._build_hamiltonian_quadratures_pool(
        problem_key="spinless_tv",
        num_sites=4,
        t=1.0,
        u=0.0,
        dv=0.1,
        v_nn=1.5,
        ordering="blocked",
        boundary="open",
    )
    direct = [
        type(pool[0])(label=f"ham_quad::{label}", polynomial=poly)
        for label, poly in build_spinless_tv_quadratures(
            num_sites=4,
            t=1.0,
            v_nn=1.5,
            dv=0.1,
            boundary="open",
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)
    assert any("three_site_bridge" in str(term.label) for term in pool)


def test_build_hamiltonian_quadratures_pool_matches_spin_boson_quadratures() -> None:
    pool = primitive_pools._build_hamiltonian_quadratures_pool(
        problem_key="spin_boson",
        num_sites=1,
        t=0.7,
        u=0.4,
        dv=0.3,
        omega0=1.0,
        g_ep=0.6,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    direct = [
        type(pool[0])(label=f"ham_quad::{label}", polynomial=poly)
        for label, poly in build_spin_boson_quadratures(
            num_sites=1,
            t=0.7,
            u=0.4,
            dv=0.3,
            omega0=1.0,
            g_ep=0.6,
            n_ph_max=2,
            boson_encoding="binary",
            include_zero_point=True,
        )
        if len(poly.return_polynomial()) > 0
    ]
    assert [str(term.label) for term in pool] == [str(term.label) for term in direct]
    _assert_ansatz_terms_match(pool, direct)


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


def test_adapt_pipeline_reexports_extracted_helpers() -> None:
    assert hc_adapt._collect_hardcoded_terms_exyz is problem_setup._collect_hardcoded_terms_exyz
    assert hc_adapt._build_uccsd_pool is primitive_pools._build_uccsd_pool
    assert hc_adapt._build_paop_pool is primitive_pools._build_paop_pool
    assert hc_adapt._build_hh_full_meta_pool is hh_pool_presets._build_hh_full_meta_pool
    assert hc_adapt._exact_gs_energy_for_problem is problem_setup._exact_gs_energy_for_problem
    assert hc_adapt._exact_reference_state_for_hh is problem_setup._exact_reference_state_for_hh
    assert hc_adapt.build_hh_pool_by_key is hh_pool_presets.build_hh_pool_by_key


def test_build_hh_pool_by_key_accepts_hamiltonian_blocks() -> None:
    num_particles = half_filled_num_particles(2)
    h_poly = problem_setup.build_problem_hamiltonian(
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
    dispatch_pool, method_name, class_meta, label_meta = hh_pool_presets.build_hh_pool_by_key(
        pool_key_hh="hamiltonian_blocks",
        h_poly=h_poly,
        num_sites=2,
        t=1.0,
        u=4.0,
        omega0=1.0,
        g_ep=0.5,
        dv=0.1,
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
    assert method_name == "hardcoded_adapt_vqe_hamiltonian_blocks_hh"
    assert class_meta is None
    assert label_meta is None
    assert len(dispatch_pool) > 0
    assert all(str(term.label).startswith("ham_block::") for term in dispatch_pool)


def test_build_hh_fermionic_reusable_pool_lifts_problem_local_spinful_generators() -> None:
    pool = primitive_pools._build_hh_fermionic_reusable_pool(
        num_sites=3,
        t=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        prune_eps=0.0,
    )
    labels = [str(term.label) for term in pool]
    assert labels
    assert any(label.startswith("hh_fermionic_reusable::bond_charge_hop_nn_up") for label in labels)
    assert any("exchange_nn" in label for label in labels)
    assert any("three_site_bridge" in label for label in labels)
    sigs = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
    assert len(sigs) == len(set(sigs))


def test_build_hh_full_meta_pool_includes_problem_local_blocks_and_reusable_generators() -> None:
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
    pool, meta = hh_pool_presets._build_hh_full_meta_pool(
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
    labels = [str(term.label) for term in pool]
    assert meta["pool_surface_key"] == hh_pool_presets.HH_MATH_MD_FULL_META_POOL_KEY
    assert meta["pool_display_name"] == hh_pool_presets.HH_MATH_MD_FULL_META_DISPLAY_NAME
    assert {"hamiltonian_blocks", "hh_fermionic_reusable", "hh_pure_phonon"}.issubset(
        set(meta["built_component_keys"])
    )
    assert int(meta["raw_hamiltonian_blocks"]) > 0
    assert int(meta["raw_hh_fermionic_reusable"]) > 0
    assert int(meta["raw_hh_pure_phonon"]) > 0
    assert any(label.startswith("ham_block::") for label in labels)
    assert any(label.startswith("hh_fermionic_reusable::") for label in labels)
    assert any(label.startswith("hh_phonon::") for label in labels)
    built = set(meta["built_component_keys"])
    assert set(meta["skipped_component_keys"]) == set()
    assert "paop_lf3_std" in built
    assert "paop_lf4_std" in built
    assert "paop_bond_disp_std" in built
    assert "vlf_only" in built
    assert "uccsd_otimes_paop_lf_std" in built


def test_hh_optional_pool_keys_emit_their_named_operator_families() -> None:
    num_particles = half_filled_num_particles(2)
    h_poly = problem_setup.build_problem_hamiltonian(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=0.75,
        dv=0.0,
        omega0=1.0,
        g_ep=0.75,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )

    expected_prefixes = {
        "paop_lf3_std": "paop_lf3_std:paop_lf3(",
        "paop_lf4_std": "paop_lf4_std:paop_lf4(",
        "paop_bond_disp_std": "paop_bond_disp_std:paop_bond_disp(",
        "paop_hop_sq_std": "paop_hop_sq_std:paop_hop_sq(",
        "paop_pair_sq_std": "paop_pair_sq_std:paop_pair_sq(",
        "vlf_only": "vlf_only:lf_disp(",
        "sq_only": "sq_only:sq(",
        "sq_dens_only": "sq_dens_only:dens_sq(",
        "uccsd_otimes_paop_lf_std": "uccsd_otimes_paop::",
        "uccsd_otimes_paop_bond_disp_std": "uccsd_otimes_paop::",
    }
    for pool_key, expected_prefix in expected_prefixes.items():
        dispatch_pool, method_name, class_meta, label_meta = hh_pool_presets.build_hh_pool_by_key(
            pool_key_hh=pool_key,
            h_poly=h_poly,
            num_sites=2,
            t=1.0,
            u=0.75,
            omega0=1.0,
            g_ep=0.75,
            dv=0.0,
            n_ph_max=3,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            paop_r=2,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        assert method_name == f"hardcoded_adapt_vqe_{pool_key}"
        assert class_meta is None
        assert label_meta is None
        assert any(str(term.label).startswith(expected_prefix) for term in dispatch_pool), pool_key


def test_build_hh_full_meta_pool_keeps_termwise_component_when_g_ep_zero() -> None:
    num_particles = half_filled_num_particles(2)
    h_poly = problem_setup.build_problem_hamiltonian(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    _pool, meta = hh_pool_presets._build_hh_full_meta_pool(
        h_poly=h_poly,
        num_sites=2,
        t=1.0,
        u=4.0,
        omega0=1.0,
        g_ep=0.0,
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
    assert int(meta["raw_hh_termwise_augmented"]) > 0


def test_build_hh_full_meta_paop_prune_knob_does_not_change_reusable_fermionic_component() -> None:
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
    pool_ref, meta_ref = hh_pool_presets._build_hh_full_meta_pool(
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
    pool_pruned, meta_pruned = hh_pool_presets._build_hh_full_meta_pool(
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
        paop_prune_eps=1e6,
        paop_normalization="none",
        num_particles=num_particles,
    )
    assert int(meta_ref["raw_hh_fermionic_reusable"]) == int(meta_pruned["raw_hh_fermionic_reusable"])
    reusable_ref = {
        str(term.label)
        for term in pool_ref
        if str(term.label).startswith("hh_fermionic_reusable::")
    }
    reusable_pruned = {
        str(term.label)
        for term in pool_pruned
        if str(term.label).startswith("hh_fermionic_reusable::")
    }
    assert reusable_ref == reusable_pruned


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


def test_build_hh_pool_by_key_uses_disk_cache_for_full_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE_DIR", str(tmp_path / "hh_pool_cache"))
    monkeypatch.delenv("STATIC_ADAPT_HH_POOL_CACHE", raising=False)
    hh_pool_presets.clear_hh_pool_cache_memory()
    num_particles = half_filled_num_particles(2)
    h_poly = problem_setup.build_problem_hamiltonian(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=0.5,
        dv=0.0,
        omega0=1.0,
        g_ep=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    common_kwargs = dict(
        pool_key_hh="full_meta",
        h_poly=h_poly,
        num_sites=2,
        t=1.0,
        u=0.5,
        omega0=1.0,
        g_ep=0.2,
        dv=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        num_particles=num_particles,
        include_legal_subspace_filter_meta=True,
    )
    first_events: list[tuple[str, dict[str, Any]]] = []
    first_pool, first_method, first_class_meta, first_label_meta, first_legal_meta = hh_pool_presets.build_hh_pool_by_key(
        **common_kwargs,
        ai_log=lambda event, **fields: first_events.append((event, fields)),
    )
    cache_files = list((tmp_path / "hh_pool_cache").glob("*.pickle"))
    assert len(cache_files) == 1
    assert any(event == "hardcoded_adapt_pool_cache_stored" for event, _fields in first_events)

    hh_pool_presets.clear_hh_pool_cache_memory()
    second_events: list[tuple[str, dict[str, Any]]] = []
    second_pool, second_method, second_class_meta, second_label_meta, second_legal_meta = hh_pool_presets.build_hh_pool_by_key(
        **common_kwargs,
        ai_log=lambda event, **fields: second_events.append((event, fields)),
    )

    assert second_pool is not first_pool
    assert second_method == first_method
    assert second_class_meta == first_class_meta
    assert second_label_meta == first_label_meta
    assert second_legal_meta == first_legal_meta
    assert [str(term.label) for term in second_pool] == [str(term.label) for term in first_pool]
    assert [primitive_pools._polynomial_signature(term.polynomial) for term in second_pool] == [
        primitive_pools._polynomial_signature(term.polynomial) for term in first_pool
    ]
    hit_events = [fields for event, fields in second_events if event == "hardcoded_adapt_pool_cache_hit"]
    assert hit_events and hit_events[0]["cache_level"] == "disk"
    assert not any(event == "hardcoded_adapt_full_meta_pool_built" for event, _fields in second_events)


def test_build_hh_math_md_full_meta_alias_matches_legacy_full_meta_surface() -> None:
    num_particles = half_filled_num_particles(2)
    h_poly = problem_setup.build_problem_hamiltonian(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=0.75,
        dv=0.0,
        omega0=1.0,
        g_ep=0.75,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    common_kwargs = dict(
        h_poly=h_poly,
        num_sites=2,
        t=1.0,
        u=0.75,
        omega0=1.0,
        g_ep=0.75,
        dv=0.0,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=2,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        num_particles=num_particles,
        include_legal_subspace_filter_meta=True,
    )
    legacy_pool, legacy_method, legacy_class_meta, legacy_label_meta, legacy_legal_meta = hh_pool_presets.build_hh_pool_by_key(
        pool_key_hh="full_meta",
        **common_kwargs,
    )
    math_pool, math_method, math_class_meta, math_label_meta, math_legal_meta = hh_pool_presets.build_hh_pool_by_key(
        pool_key_hh=hh_pool_presets.HH_MATH_MD_FULL_META_POOL_KEY,
        **common_kwargs,
    )

    assert legacy_method == "hardcoded_adapt_vqe_full_meta"
    assert math_method == "hardcoded_adapt_vqe_math_md_full_meta_v1"
    assert legacy_class_meta is None
    assert legacy_label_meta is None
    assert math_class_meta is None
    assert math_label_meta is None
    assert [primitive_pools._polynomial_signature(term.polynomial) for term in math_pool] == [
        primitive_pools._polynomial_signature(term.polynomial) for term in legacy_pool
    ]
    assert [str(term.label) for term in math_pool] == [str(term.label) for term in legacy_pool]
    assert legacy_legal_meta["pool_surface_key"] == hh_pool_presets.HH_MATH_MD_FULL_META_POOL_KEY
    assert legacy_legal_meta["adapt_pool_requested"] == "full_meta"
    assert math_legal_meta["pool_surface_key"] == hh_pool_presets.HH_MATH_MD_FULL_META_POOL_KEY
    assert math_legal_meta["pool_display_name"] == hh_pool_presets.HH_MATH_MD_FULL_META_DISPLAY_NAME
    assert math_legal_meta["adapt_pool_requested"] == hh_pool_presets.HH_MATH_MD_FULL_META_POOL_KEY
    labels = [str(term.label) for term in math_pool]
    assert any(label.startswith("paop_lf3_std:paop_lf3(") for label in labels)
    assert any(label.startswith("paop_lf4_std:paop_lf4(") for label in labels)
    assert any(label.startswith("paop_bond_disp_std:paop_bond_disp(") for label in labels)
    assert any(label.startswith("hh_phonon::") for label in labels)
    assert any(label.startswith("uccsd_otimes_paop::") for label in labels)


def test_build_hh_pool_by_key_dispatches_uccsd_paop_product_family(monkeypatch) -> None:
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
    fake_pool = primitive_pools._build_full_hamiltonian_pool(h_poly, normalize_coeff=True)[:1]

    def _fake_product_pool(*_args, **_kwargs):
        return list(fake_pool), {"family": "uccsd_otimes_paop_lf_std"}

    monkeypatch.setattr(
        hh_pool_presets,
        "_build_hh_uccsd_paop_product_pool",
        _fake_product_pool,
    )

    dispatch_pool, method_name, class_meta, label_meta = hh_pool_presets.build_hh_pool_by_key(
        pool_key_hh="uccsd_otimes_paop_lf_std",
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
        num_particles=half_filled_num_particles(2),
    )
    assert method_name == "hardcoded_adapt_vqe_uccsd_otimes_paop_lf_std"
    assert class_meta is None
    assert label_meta is None
    assert [term.label for term in dispatch_pool] == [term.label for term in fake_pool]


def test_hubbard_uccsd_qeb_pool_is_deduplicated_and_keeps_qeb_terms() -> None:
    pool = primitive_pools._build_hubbard_uccsd_qeb_pool(
        num_sites=2,
        num_particles=half_filled_num_particles(2),
        ordering="blocked",
    )
    labels = [str(term.label) for term in pool]
    signatures = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
    assert len(signatures) == len(set(signatures))
    assert any(label.startswith("uccsd_sing(") for label in labels)
    assert any(label.startswith("uccsd_dbl(") for label in labels)
    assert any(label.startswith(("qeb_pair(", "qeb_double(")) for label in labels)


def test_hubbard_uccsd_qeb_hva_blocks_pool_is_grouped_deduped_and_audited() -> None:
    hva_pool = primitive_pools._build_hubbard_hva_blocks_pool(
        num_sites=2,
        t=1.0,
        u=0.25,
        dv=0.0,
        ordering="blocked",
        boundary="open",
    )
    hva_labels = [str(term.label) for term in hva_pool]
    assert "hva_block::hop_layer" in hva_labels
    assert "hva_block::onsite_layer" in hva_labels
    assert "hva_block::potential_layer" not in hva_labels
    assert all(getattr(term, "execution_mode", None) == "grouped_exact" for term in hva_pool)

    pool = primitive_pools._build_hubbard_uccsd_qeb_hva_blocks_pool(
        num_sites=2,
        num_particles=half_filled_num_particles(2),
        ordering="blocked",
        t=1.0,
        u=0.25,
        dv=0.0,
        boundary="open",
    )
    labels = [str(term.label) for term in pool]
    signatures = [primitive_pools._polynomial_signature(term.polynomial) for term in pool]
    assert len(signatures) == len(set(signatures))
    assert any(label.startswith("uccsd_sing(") for label in labels)
    assert any(label.startswith("uccsd_dbl(") for label in labels)
    assert any(label.startswith(("qeb_pair(", "qeb_double(")) for label in labels)
    assert "hva_block::hop_layer" in labels
    assert "hva_block::onsite_layer" in labels
    assert "hva_block::potential_layer" not in labels

    audit = summarize_static_physical_operator_pool_labels(labels, problem="hubbard")
    assert audit["other_count"] == 0
    assert audit["exact_other_labels"] == []
    assert audit["lane_counts"]["qeb_excitation"] > 0
    assert audit["lane_counts"]["hva_hamiltonian_blocks"] == 2
