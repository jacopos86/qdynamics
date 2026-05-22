from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.builders import problem_registry, problem_setup
from pipelines.static_adapt.builders.pool_resolution import resolve_pool_plan
from src.quantum.compiled_polynomial import (
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.chemistry.molecular_hamiltonian import (
    build_restricted_closed_shell_molecular_hamiltonian,
)
from src.quantum.chemistry.psi4_adapter import (
    load_restricted_closed_shell_problem_from_json,
)


def _assert_coeff_maps_match(lhs: dict[str, complex], rhs: dict[str, complex], tol: float = 1e-12) -> None:
    assert set(lhs) == set(rhs)
    for label in lhs:
        assert abs(complex(lhs[label]) - complex(rhs[label])) <= float(tol), label


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


def test_available_problem_keys_match_current_surface() -> None:
    assert problem_registry.available_problem_keys() == (
        "hubbard",
        "hh",
        "molecular_restricted_closed_shell",
        "molecular_vibronic_h2",
        "ionic_hubbard",
        "extended_hubbard",
        "ttprime_hubbard",
        "spinless_tv",
        "spin_boson",
        "bose_hubbard",
        "harmonic_kerr_chain",
    )


def test_available_adapt_pool_keys_include_current_hubbard_and_hh_pools() -> None:
    assert problem_registry.available_adapt_pool_keys() == (
        "uccsd",
        "cse",
        "full_hamiltonian",
        "hamiltonian_blocks",
        "full_meta",
        "hva",
        "pareto_lean",
        "pareto_lean_l3",
        "pareto_lean_l2",
        "pareto_lean_gate_pruned",
        "uccsd_paop_lf_full",
        "uccsd_otimes_paop_lf_std",
        "uccsd_otimes_paop_lf2_std",
        "uccsd_otimes_paop_bond_disp_std",
        "uccsd_otimes_paop_lf_std_seq2p",
        "uccsd_otimes_paop_lf2_std_seq2p",
        "uccsd_otimes_paop_bond_disp_std_seq2p",
        "sq_lf_std",
        "paop",
        "paop_min",
        "paop_std",
        "paop_full",
        "paop_lf",
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
        "vlf_only",
        "sq_only",
        "vlf_sq",
        "sq_dens_only",
        "vlf_sq_dens",
        "family_max",
        "hamiltonian_quadratures",
    )


def test_resolve_hubbard_problem_context_matches_existing_builder_and_layout() -> None:
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
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
        include_zero_point=True,
    )
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.total_qubits == 4
    assert resolved.layout.fermion_qubits == 4
    assert resolved.layout.boson_qubits == 0
    assert resolved.layout.block("fermion") is not None
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key == "uccsd"
    assert resolved.default_pool_resolution_scope == "family_default"
    assert resolved.default_sector_label == "half_filled_spin_sector"
    assert tuple(resolved.default_num_particles) == (1, 1)
    assert resolved.sector.label == "half_filled_spin_sector"
    assert resolved.sector.comparison_space_label == "full_register"
    assert [c.kind for c in resolved.sector.constraints] == ["fixed_count", "fixed_count"]
    assert resolved.reference_state.kind == "hartree_fock"
    assert resolved.reference_state.source_label == "hf"
    assert resolved.reference_state.state_kind == "reference_state"
    assert resolved.exact_target.kind == "exact_ground_energy_sector"
    assert resolved.exact_target.comparison_space_label == "full_register"
    assert resolved.exact_target.fallback_policy == "reference_state_anchor_when_exact_state_unavailable"
    ref_state = resolved.reference_state.build_state()
    anchor_state = resolved.exact_target.build_fallback_anchor_state()
    assert ref_state.shape == anchor_state.shape
    assert abs(float((ref_state.conj() @ ref_state).real) - 1.0) < 1e-10
    assert abs(float((anchor_state.conj() @ anchor_state).real) - 1.0) < 1e-10


def test_resolve_hh_problem_context_matches_existing_builder_and_layout() -> None:
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
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
        include_zero_point=True,
    )
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.total_qubits == 6
    assert resolved.layout.fermion_qubits == 4
    assert resolved.layout.boson_qubits == 2
    assert resolved.layout.block("fermion") is not None
    assert resolved.layout.block("boson") is not None
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key is None
    assert resolved.default_pool_resolution_scope == "controller_resolved"
    assert resolved.default_sector_label == "half_filled_fermion_sector"
    assert tuple(resolved.default_num_particles) == (1, 1)
    assert resolved.sector.label == "half_filled_fermion_sector"
    assert resolved.sector.comparison_space_label == "fermion_register_with_truncated_bosons"
    assert [c.kind for c in resolved.sector.constraints] == [
        "fixed_count",
        "fixed_count",
        "truncation",
    ]
    assert resolved.reference_state.kind == "hubbard_holstein_reference_state"
    assert resolved.reference_state.source_label == "hf"
    assert resolved.reference_state.state_kind == "reference_state"
    assert resolved.exact_target.kind == "exact_ground_energy_sector_hh"
    assert (
        resolved.exact_target.comparison_space_label
        == "fermion_register_with_truncated_bosons"
    )
    assert resolved.exact_target.fallback_policy == "reference_state_anchor_when_exact_state_unavailable"
    ref_state = resolved.reference_state.build_state()
    anchor_state = resolved.exact_target.build_fallback_anchor_state()
    assert ref_state.shape == anchor_state.shape
    assert abs(float((ref_state.conj() @ ref_state).real) - 1.0) < 1e-10
    assert abs(float((anchor_state.conj() @ anchor_state).real) - 1.0) < 1e-10


def test_resolve_ionic_hubbard_problem_context_matches_builder_and_defaults() -> None:
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
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
        include_zero_point=True,
    )
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.total_qubits == 4
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key == "hamiltonian_quadratures"
    assert resolved.sector.label == "half_filled_spin_sector"
    assert resolved.reference_state.kind == "hartree_fock"
    assert resolved.exact_target.kind == "exact_ground_energy_sector"


def test_resolve_spin_boson_problem_context_matches_builder_and_defaults() -> None:
    request = problem_registry.ProblemRequest(
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
    resolved = problem_registry.resolve_problem_context(request)
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
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.total_qubits == 4
    assert resolved.layout.fermion_qubits == 2
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key == "full_meta"
    assert resolved.sector.label == "single_emitter_truncated_boson_sector"
    assert resolved.sector.comparison_space_label == "one_emitter_truncated_boson_register"
    assert [c.kind for c in resolved.sector.constraints] == ["weighted_charge", "truncation"]
    assert resolved.reference_state.kind == "spin_boson_uncoupled_ground"
    assert resolved.exact_target.kind == "exact_ground_energy_spin_boson"
    ref_state = resolved.reference_state.build_state()
    anchor_state = resolved.exact_target.build_fallback_anchor_state()
    assert ref_state.shape == anchor_state.shape
    assert abs(float((ref_state.conj() @ ref_state).real) - 1.0) < 1e-10
    assert abs(float((anchor_state.conj() @ anchor_state).real) - 1.0) < 1e-10


def test_resolve_spinless_problem_context_matches_builder_and_defaults() -> None:
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
        v_nn=1.5,
        n_fermions=2,
    )
    resolved = problem_registry.resolve_problem_context(request)
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
        include_zero_point=True,
        v_nn=1.5,
    )
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.total_qubits == 4
    assert resolved.layout.fermion_qubits == 4
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key == "hamiltonian_quadratures"
    assert resolved.sector.label == "fixed_spinless_sector"
    assert resolved.sector.comparison_space_label == "spinless_fermion_register"
    assert [c.kind for c in resolved.sector.constraints] == ["fixed_count"]
    assert resolved.reference_state.kind == "spinless_fermion_filling"
    assert resolved.exact_target.kind == "exact_ground_energy_spinless_fixed_count"
    ref_state = resolved.reference_state.build_state()
    anchor_state = resolved.exact_target.build_fallback_anchor_state()
    assert ref_state.shape == anchor_state.shape
    assert abs(float((ref_state.conj() @ ref_state).real) - 1.0) < 1e-10
    assert abs(float((anchor_state.conj() @ anchor_state).real) - 1.0) < 1e-10


def test_resolve_bose_hubbard_problem_context_matches_builder_and_defaults() -> None:
    request = problem_registry.ProblemRequest(
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
    resolved = problem_registry.resolve_problem_context(request)
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
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.fermion_qubits == 0
    assert resolved.layout.boson_qubits > 0
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key == "full_meta"
    assert resolved.reference_state.kind == "bose_hubbard_one_boson_fock"
    assert resolved.exact_target.kind == "exact_ground_energy_boson_only"


def test_bose_hubbard_reference_has_nonzero_phase3_full_meta_gradient() -> None:
    request = problem_registry.ProblemRequest(
        problem_key="bose_hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.25,
        omega0=1.0,
        g_ep=0.0,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
    pool_resolution = resolve_pool_plan(
        resolved_problem=resolved,
        continuation_mode="phase3_v1",
        adapt_pool="full_meta",
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=1e-12,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="off",
    )
    psi = resolved.reference_state.build_state()
    h_compiled = compile_polynomial_action(resolved.hamiltonian)
    hpsi = apply_compiled_polynomial(psi, h_compiled)
    max_grad = 0.0
    for term in pool_resolution.pool:
        apsi = apply_compiled_polynomial(psi, compile_polynomial_action(term.polynomial))
        max_grad = max(max_grad, abs(float(adapt_commutator_grad_from_hpsi(hpsi, apsi))))
    assert max_grad > 1e-8


def test_resolve_harmonic_kerr_chain_problem_context_matches_builder_and_defaults() -> None:
    request = problem_registry.ProblemRequest(
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
    resolved = problem_registry.resolve_problem_context(request)
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
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.fermion_qubits == 0
    assert resolved.layout.boson_qubits > 0
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key == "full_meta"
    assert resolved.reference_state.kind == "boson_vacuum"
    assert resolved.exact_target.kind == "exact_ground_energy_boson_only"


def test_resolve_problem_context_from_namespace_uses_existing_cli_fields() -> None:
    args = SimpleNamespace(
        problem="hh",
        L=3,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context_from_namespace(args)
    assert resolved.family_key == "hh"
    assert resolved.request.num_sites == 3
    assert resolved.layout.total_qubits == 9
    assert resolved.default_continuation_mode == "phase3_v1"


def test_resolve_molecular_problem_context_matches_loader_and_layout(tmp_path: Path) -> None:
    json_path = _write_molecular_problem_json(tmp_path)
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
        molecular_problem_json=str(json_path),
    )
    resolved = problem_registry.resolve_problem_context(request)
    loaded_problem = load_restricted_closed_shell_problem_from_json(json_path)
    wrapped = build_restricted_closed_shell_molecular_hamiltonian(
        loaded_problem,
        ordering="blocked",
    )
    _, resolved_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    _, wrapped_map = problem_setup._collect_hardcoded_terms_exyz(wrapped)
    _assert_coeff_maps_match(resolved_map, wrapped_map)
    assert resolved.layout.total_qubits == 4
    assert resolved.layout.fermion_qubits == 4
    assert resolved.layout.boson_qubits == 0
    assert resolved.default_controller_profile == "phase3_v1"
    assert resolved.default_continuation_mode == "phase3_v1"
    assert resolved.default_pool_key == "uccsd"
    assert resolved.default_pool_resolution_scope == "family_default"
    assert resolved.default_sector_label == "closed_shell_fixed_number_sector"
    assert resolved.reference_state.kind == "restricted_hartree_fock"
    assert resolved.exact_target.kind == "exact_ground_energy_sector_molecular"
    assert resolved.sector.comparison_space_label == "spin_orbital_register"
    assert tuple(resolved.default_num_particles) == (1, 1)
    assert resolved.runtime_data is not None
    assert str(resolved.runtime_data.get("molecular_problem_json")) == str(json_path)


def test_resolve_molecular_problem_context_from_namespace_derives_num_sites_from_json(
    tmp_path: Path,
) -> None:
    json_path = _write_molecular_problem_json(tmp_path)
    args = SimpleNamespace(
        problem="molecular_restricted_closed_shell",
        molecular_problem_json=json_path,
        L=99,
        t=0.0,
        u=0.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context_from_namespace(args)
    assert resolved.family_key == "molecular_restricted_closed_shell"
    assert resolved.request.num_sites == 2
    assert resolved.layout.total_qubits == 4


def test_lih_molecular_problem_context_from_namespace_derives_l6_from_fixture() -> None:
    json_path = REPO_ROOT / "test_support" / "molecular_problem_lih_sto3g.json"
    args = SimpleNamespace(
        problem="molecular_restricted_closed_shell",
        molecular_problem_json=json_path,
        L=99,
        t=0.0,
        u=0.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    request = problem_registry.ProblemRequest.from_namespace(args)
    assert request.num_sites == 6

    resolved = problem_registry.resolve_problem_context(
        request,
        hamiltonian=PauliPolynomial("JW"),
    )
    assert resolved.family_key == "molecular_restricted_closed_shell"
    assert resolved.request.num_sites == 6
    assert resolved.layout.total_qubits == 12
    assert resolved.layout.fermion_qubits == 12
    assert resolved.layout.boson_qubits == 0
    assert resolved.reference_state.kind == "restricted_hartree_fock"
    assert resolved.exact_target.kind == "exact_ground_energy_sector_molecular"
    assert tuple(resolved.default_num_particles) == (2, 2)
    assert resolved.runtime_data is not None
    assert str(resolved.runtime_data.get("molecular_problem_json")) == str(json_path)


def test_resolve_molecular_problem_context_requires_problem_json() -> None:
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
        molecular_problem_json=None,
    )
    try:
        problem_registry.resolve_problem_context(request)
    except ValueError as exc:
        assert "molecular-problem-json" in str(exc)
    else:
        raise AssertionError("Expected a ValueError when the molecular JSON path is missing.")


def test_resolve_molecular_problem_context_rejects_interleaved_ordering(tmp_path: Path) -> None:
    request = problem_registry.ProblemRequest(
        problem_key="molecular_restricted_closed_shell",
        num_sites=2,
        t=0.0,
        u=0.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="interleaved",
        boundary="open",
        include_zero_point=True,
        molecular_problem_json=str(_write_molecular_problem_json(tmp_path)),
    )
    try:
        problem_registry.resolve_problem_context(request)
    except ValueError as exc:
        assert "ordering='blocked'" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for interleaved molecular ordering.")


def _molecular_vibronic_h2_request(*, g_ep: float, n_ph_max: int) -> problem_registry.ProblemRequest:
    return problem_registry.ProblemRequest(
        problem_key="molecular_vibronic_h2",
        num_sites=2,
        t=1.0,
        u=0.0,
        dv=0.0,
        omega0=0.022328470326434775,
        g_ep=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )


def test_resolve_molecular_vibronic_h2_context_uses_fixture_backed_cutoffs() -> None:
    weak = problem_registry.resolve_problem_context(_molecular_vibronic_h2_request(g_ep=0.25, n_ph_max=1))
    strong = problem_registry.resolve_problem_context(_molecular_vibronic_h2_request(g_ep=1.0, n_ph_max=1))

    for resolved, coupling in ((weak, 0.25), (strong, 1.0)):
        assert resolved.family_key == "molecular_vibronic_h2"
        assert resolved.layout.total_qubits == 5
        assert resolved.layout.fermion_qubits == 4
        assert resolved.layout.boson_qubits == 1
        assert resolved.default_pool_key == "full_meta"
        assert resolved.default_sector_label == "closed_shell_fermions_with_truncated_vibration"
        assert tuple(resolved.default_num_particles) == (1, 1)
        assert resolved.sector.constraints[-1].max_local_occupancy == 1
        assert resolved.reference_state.kind == "restricted_hf_times_boson_vacuum"
        assert resolved.exact_target.kind == "exact_ground_energy_molecular_vibronic_h2_physical_sector"
        assert resolved.runtime_data["vibronic_h2_coupling_scale"] == coupling
        assert "molecular_vibronic_h2_sto3g_fd001.json" in str(resolved.runtime_data["vibronic_h2_fixture_path"])
        psi_ref = resolved.reference_state.build_state()
        assert psi_ref.shape == (1 << 5,)
        assert abs(float(np.linalg.norm(psi_ref)) - 1.0) < 1e-12
        assert np.isfinite(float(resolved.exact_target.resolve_energy()))

    assert not np.isclose(
        float(weak.exact_target.resolve_energy()),
        float(strong.exact_target.resolve_energy()),
    )


def test_resolve_molecular_vibronic_h2_exact_reference_supports_nph4_fixture_rebuild() -> None:
    resolved = problem_registry.resolve_problem_context(_molecular_vibronic_h2_request(g_ep=0.25, n_ph_max=4))

    assert resolved.layout.total_qubits == 7
    assert resolved.layout.fermion_qubits == 4
    assert resolved.layout.boson_qubits == 3
    assert resolved.sector.constraints[-1].max_local_occupancy == 4
    psi_ref = resolved.reference_state.build_state()
    assert psi_ref.shape == (1 << 7,)
    assert abs(float(np.linalg.norm(psi_ref)) - 1.0) < 1e-12
    assert np.isfinite(float(resolved.exact_target.resolve_energy()))


def test_load_molecular_problem_json_rejects_open_shell_payload(tmp_path: Path) -> None:
    json_path = _write_molecular_problem_json(tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["n_beta"] = 0
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_restricted_closed_shell_problem_from_json(json_path)
    except ValueError as exc:
        assert "n_alpha == n_beta" in str(exc)
    else:
        raise AssertionError("Expected open-shell molecular JSON to be rejected.")


def test_problem_registry_continuation_defaults_match_current_runtime_behavior() -> None:
    assert problem_registry.default_continuation_mode_for_problem("hubbard") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("hh") == "phase3_v1"
    assert (
        problem_registry.default_continuation_mode_for_problem("molecular_restricted_closed_shell")
        == "phase3_v1"
    )
    assert problem_registry.default_continuation_mode_for_problem("molecular_vibronic_h2") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("ionic_hubbard") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("extended_hubbard") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("ttprime_hubbard") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("spinless_tv") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("spin_boson") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("bose_hubbard") == "phase3_v1"
    assert problem_registry.default_continuation_mode_for_problem("harmonic_kerr_chain") == "phase3_v1"
    assert problem_registry.supported_continuation_modes_for_problem(
        "molecular_restricted_closed_shell"
    ) == ("legacy", "phase1_v1", "phase2_v1", "phase3_v1")
    assert problem_registry.supported_continuation_modes_for_problem("molecular_vibronic_h2") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )
    assert problem_registry.supported_continuation_modes_for_problem("ionic_hubbard") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )
    assert problem_registry.supported_continuation_modes_for_problem("extended_hubbard") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )
    assert problem_registry.supported_continuation_modes_for_problem("ttprime_hubbard") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )
    assert problem_registry.supported_continuation_modes_for_problem("spinless_tv") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )
    assert problem_registry.supported_continuation_modes_for_problem("spin_boson") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )
    assert problem_registry.supported_continuation_modes_for_problem("bose_hubbard") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )
    assert problem_registry.supported_continuation_modes_for_problem("harmonic_kerr_chain") == (
        "legacy",
        "phase1_v1",
        "phase2_v1",
        "phase3_v1",
    )


def test_problem_registry_runtime_default_pool_resolution_matches_current_behavior(
    tmp_path: Path,
) -> None:
    resolved_hubbard = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
            problem_key="hubbard",
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.0,
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
        )
    )
    resolved_hh = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
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
            include_zero_point=True,
        )
    )
    resolved_molecular = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
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
            include_zero_point=True,
            molecular_problem_json=str(_write_molecular_problem_json(tmp_path)),
        )
    )
    resolved_molecular_vibronic = problem_registry.resolve_problem_context(
        _molecular_vibronic_h2_request(g_ep=0.25, n_ph_max=1)
    )
    resolved_ionic = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
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
            include_zero_point=True,
        )
    )
    resolved_spinless = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
            problem_key="spinless_tv",
            num_sites=4,
            t=1.0,
            u=0.0,
            dv=0.0,
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
            v_nn=1.5,
            n_fermions=2,
        )
    )
    resolved_spin_boson = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
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
    )
    resolved_bose_hubbard = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
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
    )
    resolved_harmonic_kerr = problem_registry.resolve_problem_context(
        problem_registry.ProblemRequest(
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
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_hubbard,
            continuation_mode="legacy",
        )
        == "uccsd"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_hh,
            continuation_mode="legacy",
        )
        == "full_meta"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_hh,
            continuation_mode="phase3_v1",
        )
        == "paop_lf_std"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_molecular,
            continuation_mode="legacy",
        )
        == "uccsd"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_molecular_vibronic,
            continuation_mode="legacy",
        )
        == "full_meta"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_ionic,
            continuation_mode="legacy",
        )
        == "hamiltonian_quadratures"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_spinless,
            continuation_mode="legacy",
        )
        == "hamiltonian_quadratures"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_spin_boson,
            continuation_mode="legacy",
        )
        == "full_meta"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_bose_hubbard,
            continuation_mode="legacy",
        )
        == "full_meta"
    )
    assert (
        problem_registry.resolve_runtime_default_pool_key(
            resolved_harmonic_kerr,
            continuation_mode="legacy",
        )
        == "full_meta"
    )


def test_problem_registry_exact_target_can_use_injected_exact_energy_impl() -> None:
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
    )
    sentinel = 1.2345
    captured: dict[str, object] = {}

    def _fake_exact_energy(*args, **kwargs):
        captured["problem"] = kwargs.get("problem")
        captured["num_sites"] = kwargs.get("num_sites")
        return float(sentinel)

    resolved = problem_registry.resolve_problem_context(
        request,
        exact_energy_impl=_fake_exact_energy,
    )
    assert resolved.exact_target.resolve_energy() == sentinel
    assert captured["problem"] == "hh"
    assert captured["num_sites"] == 2


def test_problem_registry_exact_target_energy_matches_existing_dispatch_for_hh() -> None:
    request = problem_registry.ProblemRequest(
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
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
    direct = problem_setup._exact_gs_energy_for_problem(
        resolved.hamiltonian,
        problem="hh",
        num_sites=2,
        num_particles=(1, 1),
        indexing="blocked",
        n_ph_max=1,
        boson_encoding="binary",
        t=1.0,
        u=0.0,
        dv=1.0,
        omega0=0.5,
        g_ep=0.3535534,
        boundary="open",
        include_zero_point=True,
    )
    assert abs(float(resolved.exact_target.resolve_energy()) - float(direct)) < 1e-10


def test_resolve_exact_reference_state_for_problem_hubbard_returns_full_register_sector_state() -> None:
    request = problem_registry.ProblemRequest(
        problem_key="hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="periodic",
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
    resolution = problem_setup.resolve_exact_reference_state_for_problem(
        resolved.hamiltonian,
        resolved_problem=resolved,
    )

    assert resolution.available is True
    assert resolution.source == "dense_spin_sector"
    assert resolution.comparison_space_label == "full_register"
    assert resolution.state is not None
    psi = np.asarray(resolution.state, dtype=complex).reshape(-1)
    assert psi.shape == (1 << int(resolved.layout.total_qubits),)
    assert abs(float(np.linalg.norm(psi)) - 1.0) < 1e-10

    _, coeff_map = problem_setup._collect_hardcoded_terms_exyz(resolved.hamiltonian)
    hmat = problem_setup._build_hamiltonian_matrix(coeff_map)
    energy = float(np.real(np.vdot(psi, hmat @ psi)))
    assert abs(energy - float(resolved.exact_target.resolve_energy())) < 1e-10



def test_resolve_exact_reference_state_for_problem_hh_reuses_sparse_sector_path() -> None:
    request = problem_registry.ProblemRequest(
        problem_key="hh",
        num_sites=2,
        t=1.0,
        u=2.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
    resolution = problem_setup.resolve_exact_reference_state_for_problem(
        resolved.hamiltonian,
        resolved_problem=resolved,
    )

    assert resolution.available is True
    assert resolution.source == "hh_sector_sparse"
    assert resolution.comparison_space_label == "fermion_register_with_truncated_bosons"
    assert resolution.state is not None
    assert np.asarray(resolution.state, dtype=complex).reshape(-1).shape == (
        1 << int(resolved.layout.total_qubits),
    )



def test_resolve_exact_reference_state_for_problem_respects_dense_dimension_cap() -> None:
    request = problem_registry.ProblemRequest(
        problem_key="hubbard",
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="periodic",
        include_zero_point=True,
    )
    resolved = problem_registry.resolve_problem_context(request)
    resolution = problem_setup.resolve_exact_reference_state_for_problem(
        resolved.hamiltonian,
        resolved_problem=resolved,
        max_dense_dim=8,
    )

    assert resolution.available is False
    assert resolution.source == "dense_solver_guard"
    assert resolution.skip_reason == "dense_dimension_cap_exceeded"
    assert resolution.state is None
    assert resolution.state_dimension == (1 << int(resolved.layout.total_qubits))


def test_lih_exact_reference_state_respects_dense_dimension_cap() -> None:
    json_path = REPO_ROOT / "test_support" / "molecular_problem_lih_sto3g.json"
    request = problem_registry.ProblemRequest(
        problem_key="molecular_restricted_closed_shell",
        num_sites=6,
        t=0.0,
        u=0.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        molecular_problem_json=str(json_path),
    )
    resolved = problem_registry.resolve_problem_context(
        request,
        hamiltonian=PauliPolynomial("JW"),
    )
    resolution = problem_setup.resolve_exact_reference_state_for_problem(
        resolved.hamiltonian,
        resolved_problem=resolved,
        max_dense_dim=1024,
    )

    assert resolution.available is False
    assert resolution.source == "dense_solver_guard"
    assert resolution.skip_reason == "dense_dimension_cap_exceeded"
    assert resolution.state is None
    assert resolution.state_dimension == 4096
