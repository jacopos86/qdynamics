from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    _freeze_core_active_tensors,
    _mass_weighted_vibrational_modes_from_hessian,
    build_fixed_particle_sector_sparse_matrix,
)
from src.quantum.chemistry.molecular_vibronic import (
    AMU_TO_ELECTRON_MASS,
    HARTREE_TO_CM_INV,
    ActiveSpace,
    ActiveTensors,
    ElectronicSnapshot,
    LinearCoupling,
    MolecularGeometry,
    NormalMode,
    _check_degenerate_orbital_boundaries,
    align_to_center,
    assemble_vibronic_model,
    build_register_layout,
    closed_shell_determinant_energy,
    exact_sector_energies,
    frozen_core_contract,
    frozen_core_contract_one_body,
    hartree_fock_state,
    isotope_masses_me,
    load_vibronic_hamiltonian,
    parse_args,
    parse_geometry,
    physical_sector_basis,
    reference_force,
    sector_sparse_matrix,
    select_active_space,
    subtract_reference_force,
    translation_rotation_basis,
    vibrational_modes_from_hessian,
    vibronic_model_to_jsonable,
    write_vibronic_model,
)
from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem
from src.quantum.chemistry.vibronic_h2 import pauli_polynomial_from_jsonable
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

H2O_FIXTURE = Path(
    os.environ.get(
        "PAPER_IV_H2O_LINEAR_FD_FIXTURE",
        "/Users/lordskyler/chtc_rescue/home/chtc/paper_iv_h2o_reencode_20260826/"
        "h2o_linear_fd_nph3_reference_nph4_fixture_20260731.json",
    )
)


def _random_chemist_tensors(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    g = rng.normal(size=(n, n, n, n))
    # Real chemists' ERI symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq).
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    return h, 0.1 * g


def _synthetic_model(*, n_orbitals=2, n_occ=1, cutoffs=(2, 3), encoding="binary", two_body_coupling=True,
                     coupling_scale=0.05, seed=7):
    rng = np.random.default_rng(seed)
    h, g = _random_chemist_tensors(rng, n_orbitals)
    tensors = ActiveTensors(-1.0, h, g)
    space = ActiveSpace((), tuple(range(n_orbitals)), (), n_occ, n_occ, "full")
    modes = (NormalMode(0, 0.02, np.zeros((2, 3)), "mode_0"), NormalMode(1, 0.03, np.zeros((2, 3)), "mode_1"))
    couplings = []
    for k, mode in enumerate(modes):
        dh, dg = _random_chemist_tensors(rng, n_orbitals)
        couplings.append(LinearCoupling(k, "full_linear", 0.01 * (k + 1) * (coupling_scale / 0.05), coupling_scale * dh,
                                        coupling_scale * dg if two_body_coupling else None, 0.1))
    geometry = MolecularGeometry(("H", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    return assemble_vibronic_model(geometry=geometry, basis="synthetic", active_space=space, active_tensors=tensors,
                                   modes=modes, couplings=couplings, cutoffs=cutoffs, encoding=encoding)


def _synthetic_two_orbital_model(**kwargs):
    return _synthetic_model(n_orbitals=2, n_occ=1, **kwargs)


def test_frozen_core_contract_matches_h2o_generator_formula():
    rng = np.random.default_rng(3)
    n = 5
    h, g = _random_chemist_tensors(rng, n)
    problem = RestrictedClosedShellMolecularProblem(
        geometry_spec="x", basis="x", charge=0, multiplicity=1, reference="rhf", n_spatial_orbitals=n,
        n_alpha=3, n_beta=3, hf_energy=0.0, nuclear_repulsion_energy=0.7, one_body_integrals_mo=h, two_body_integrals_mo=g)
    scalar_ref, h_ref, g_ref = _freeze_core_active_tensors(problem, active_indices=(1, 2, 3, 4), frozen_core_indices=(0,))
    space = select_active_space(n, 3, 3, policy="frozen_core:1")
    got = frozen_core_contract(h, g, 0.7, space)
    assert got.scalar == pytest.approx(scalar_ref, abs=1e-12)
    np.testing.assert_allclose(got.one_body, h_ref, atol=1e-12)
    np.testing.assert_allclose(got.two_body, g_ref, atol=1e-12)
    # The HF determinant energy of the contracted problem equals that of the full problem.
    full = ActiveTensors(0.7, h, g)
    assert closed_shell_determinant_energy(got, 2) == pytest.approx(closed_shell_determinant_energy(full, 3), abs=1e-12)
    scalar_one, active_one = frozen_core_contract_one_body(h, space)
    assert scalar_one == pytest.approx(2.0 * h[0, 0])
    np.testing.assert_allclose(active_one, h[1:, 1:])


def test_select_active_space_policies():
    assert select_active_space(7, 5, 5, policy="full").active == tuple(range(7))
    fc = select_active_space(7, 5, 5, policy="frozen_core", symbols=("O", "H", "H"))
    assert fc.frozen_core == (0,) and fc.active == tuple(range(1, 7)) and (fc.n_alpha, fc.n_beta) == (4, 4)
    cas = select_active_space(12, 8, 8, policy="cas(6,6)")
    assert cas.frozen_core == tuple(range(5)) and cas.active == tuple(range(5, 11)) and cas.external == (11,)
    with pytest.raises(ValueError):
        select_active_space(7, 5, 5, policy={"frozen_core": [0], "active": [1, 2, 3]})  # discards occupied orbitals
    with pytest.raises(ValueError):
        select_active_space(7, 5, 5, policy="cas(20,2)")


def test_vibrational_modes_project_translations_and_rotations():
    rng = np.random.default_rng(11)
    coords = np.array([[0.0, 0.0, -0.06], [0.0, 1.43, 1.14], [0.0, -1.43, 1.14]])
    masses = isotope_masses_me(("O", "H", "H"))
    basis = translation_rotation_basis(coords, masses)
    assert basis.shape == (9, 6)
    np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=1e-12)
    # A Hessian with exact vibrations plus rigid-body contamination: the projected modes must ignore it.
    vib = rng.normal(size=(9, 3))
    vib -= basis @ (basis.T @ vib)
    vib, _ = np.linalg.qr(vib)
    omegas = np.array([0.01, 0.02, 0.03])
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    weighted = vib @ np.diag(omegas**2) @ vib.T + 1e-3 * basis @ basis.T
    hessian = weighted * np.outer(sqrt_m, sqrt_m)
    modes = vibrational_modes_from_hessian(hessian, coordinates_bohr=coords, masses_me=masses)
    assert [m.frequency_hartree for m in modes] == pytest.approx(list(omegas), abs=1e-12)
    for mode, column in zip(modes, vib.T):
        assert abs(float(mode.mass_weighted_vector.reshape(-1) @ column)) == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(ValueError):
        vibrational_modes_from_hessian(-hessian, coordinates_bohr=coords, masses_me=masses)


def test_sector_matrix_matches_dense_and_generator_builder():
    model = _synthetic_two_orbital_model(cutoffs=(2, 3))
    basis = physical_sector_basis(model.layout, n_spatial_orbitals=2, num_particles=(1, 1))
    assert len(basis) == 4 * 3 * 4
    dense = np.asarray(hamiltonian_matrix(model.h_vibronic))
    sparse = sector_sparse_matrix(model.h_vibronic, basis).toarray()
    np.testing.assert_allclose(sparse, dense[np.ix_(basis, basis)], atol=1e-12)
    reference, ref_basis = build_fixed_particle_sector_sparse_matrix(
        model.h_vibronic, layout=model.layout, n_spatial_orbitals=2, num_particles=(1, 1))
    order = np.argsort(np.asarray(ref_basis))
    np.testing.assert_allclose(reference.toarray()[np.ix_(order, order)], sparse, atol=1e-12)
    energies, state = exact_sector_energies(model, n_low=3, return_state=True)
    sector_eigs = np.linalg.eigvalsh(sparse)
    assert list(energies) == pytest.approx(list(sector_eigs[:3]), abs=1e-9)
    assert float(np.real(np.vdot(state, dense @ state))) == pytest.approx(energies[0], abs=1e-8)


def test_unary_encoding_matches_binary_spectrum():
    binary = _synthetic_two_orbital_model(cutoffs=(2, 2), encoding="binary")
    unary = _synthetic_two_orbital_model(cutoffs=(2, 2), encoding="unary")
    assert binary.n_qubits == 4 + 2 + 2 and unary.n_qubits == 4 + 3 + 3
    e_b, _ = exact_sector_energies(binary, n_low=4)
    e_u, _ = exact_sector_energies(unary, n_low=4)
    assert list(e_b) == pytest.approx(list(e_u), abs=1e-9)


def test_hartree_fock_expectation_and_reference_force():
    model = _synthetic_two_orbital_model(cutoffs=(1, 1))
    dense = np.asarray(hamiltonian_matrix(model.h_vibronic))
    hf = hartree_fock_state(model)
    expected = closed_shell_determinant_energy(model.active_tensors, 1) + 0.5 * sum(m.frequency_hartree for m in model.modes)
    assert float(np.real(np.vdot(hf, dense @ hf))) == pytest.approx(expected, abs=1e-12)
    # <HF| C_nu |HF> equals the determinant energy of the derivative tensors, and the counter-term zeroes it.
    for coupling in model.couplings:
        assert reference_force(coupling, 1) == pytest.approx(
            closed_shell_determinant_energy(ActiveTensors(coupling.scalar, coupling.one_body, coupling.two_body), 1))
    shifted = subtract_reference_force(model.couplings, 1)
    assert all(abs(reference_force(c, 1)) < 1e-12 for c in shifted)


def test_one_body_couplings_and_json_round_trip(tmp_path):
    model = _synthetic_two_orbital_model(cutoffs=(1, 2), two_body_coupling=False)
    payload = vibronic_model_to_jsonable(model)
    assert payload["schema"] == "molecular_vibronic_qubit_hamiltonian_v1"
    assert payload["couplings"][0]["two_body"] is None
    path = tmp_path / "model.json"
    write_vibronic_model(path, model)
    poly, loaded = load_vibronic_hamiltonian(path)
    original = {t.pw2strng(): complex(t.p_coeff) for t in model.h_vibronic.return_polynomial()}
    restored = {t.pw2strng(): complex(t.p_coeff) for t in poly.return_polynomial()}
    assert original.keys() == restored.keys()
    assert all(abs(original[k] - restored[k]) < 1e-12 for k in original)
    assert loaded["register_layout"]["n_total_qubits"] == model.n_qubits


def test_parse_geometry_units_and_xyz_header():
    bohr = parse_geometry("H 0 0 0; H 0 0 1.4", units="bohr")
    angstrom = parse_geometry("H 0 0 0\nH 0 0 0.740848", units="angstrom")
    assert bohr.coordinates_bohr[1, 2] == pytest.approx(angstrom.coordinates_bohr[1, 2], abs=1e-6)
    xyz = parse_geometry("2\nhydrogen molecule, any comment\nH 0 0 0\nH 0 0 0.740848")
    np.testing.assert_allclose(xyz.coordinates_bohr, angstrom.coordinates_bohr)
    with pytest.raises(ValueError):
        parse_geometry("3\ncomment\nH 0 0 0\nH 0 0 0.74")
    assert isotope_masses_me(("O",), {"o": 18.0})[0] == pytest.approx(18.0 * AMU_TO_ELECTRON_MASS)
    assert isotope_masses_me(("O",))[0] == pytest.approx(15.99491461957 * AMU_TO_ELECTRON_MASS)
    geometry = parse_geometry("H 0 0 0; H 0 0 1.4", units="bohr")
    with pytest.raises(ValueError):
        geometry.coordinates_bohr[0, 0] = 1.0   # read-only: callers cannot alias the stored array
    long_inline = "; ".join(["H 0 0 %d" % k for k in range(80)])
    assert len(long_inline) > 255
    args = parse_args(["--geometry", long_inline, "--output", "x.json"])
    assert args.geometry == long_inline


def test_hartree_fock_state_lies_in_the_physical_sector_for_both_encodings():
    for encoding in ("binary", "unary"):
        model = _synthetic_model(n_orbitals=3, n_occ=2, cutoffs=(2, 2), encoding=encoding)
        basis = physical_sector_basis(model.layout, n_spatial_orbitals=3, num_particles=(2, 2))
        state = hartree_fock_state(model)
        index = int(np.flatnonzero(state)[0])
        assert index in set(int(v) for v in basis)
        dense = np.asarray(hamiltonian_matrix(model.h_vibronic))
        image = dense @ state
        outside = np.ones(len(state), dtype=bool)
        outside[basis] = False
        assert np.linalg.norm(image[outside]) < 1e-12
        assert np.linalg.norm(image[basis]) > 1e-3
        # Q_nu |HF> is nonzero: the vacuum couples to the one-quantum state.
        q0 = np.asarray(hamiltonian_matrix(model.q_operators[0]))
        assert np.linalg.norm(q0 @ state) > 1e-3


def test_reference_force_matches_dense_expectation_with_two_occupied_orbitals():
    for two_body in (True, False):
        model = _synthetic_model(n_orbitals=3, n_occ=2, cutoffs=(1, 1), two_body_coupling=two_body)
        hf = hartree_fock_state(model)
        for coupling, operator in zip(model.couplings, model.coupling_operators):
            lifted = np.asarray(hamiltonian_matrix(operator))
            # The coupling operator lives on the fermion register; embed |HF> there.
            fermion_index = int(np.flatnonzero(hf)[0]) & ((1 << model.layout.n_fermion_qubits) - 1)
            state = np.zeros(1 << model.layout.n_fermion_qubits, dtype=complex)
            state[fermion_index] = 1.0
            expectation = float(np.real(np.vdot(state, lifted @ state)))
            assert reference_force(coupling, 2) == pytest.approx(expectation, abs=1e-12)
        shifted = subtract_reference_force(model.couplings, 2)
        assert all(abs(reference_force(c, 2)) < 1e-12 for c in shifted)


def test_zero_coupling_model_is_the_analytic_oscillator_ladder():
    rng = np.random.default_rng(5)
    h, g = _random_chemist_tensors(rng, 2)
    tensors = ActiveTensors(-1.0, h, g)
    space = ActiveSpace((), (0, 1), (), 1, 1, "full")
    modes = (NormalMode(0, 0.02, np.zeros((2, 3)), "mode_0"), NormalMode(1, 0.031, np.zeros((2, 3)), "mode_1"))
    couplings = tuple(LinearCoupling(k, "full_linear", 0.0, np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 0.1) for k in range(2))
    geometry = MolecularGeometry(("H", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    model = assemble_vibronic_model(geometry=geometry, basis="synthetic", active_space=space, active_tensors=tensors,
                                    modes=modes, couplings=couplings, cutoffs=(5, 2))
    electronic_basis = physical_sector_basis(build_register_layout(2, (), (), "binary"), n_spatial_orbitals=2, num_particles=(1, 1))
    electronic = np.linalg.eigvalsh(np.asarray(hamiltonian_matrix(model.h_electronic))[np.ix_(electronic_basis, electronic_basis)])
    ladder = sorted(e + 0.02 * (n0 + 0.5) + 0.031 * (n1 + 0.5) for e in electronic for n0 in range(6) for n1 in range(3))
    energies, _ = exact_sector_energies(model, n_low=6)
    assert list(energies) == pytest.approx(ladder[:6], abs=1e-9)


def test_alignment_recovers_a_known_orbital_rotation():
    rng = np.random.default_rng(2)
    n_ao, n_occ = 4, 2
    h, g = _random_chemist_tensors(rng, n_ao)
    fock = 0.5 * (rng.normal(size=(n_ao, n_ao)) + rng.normal(size=(n_ao, n_ao)).T)
    overlap = np.eye(n_ao)
    coeff = np.linalg.qr(rng.normal(size=(n_ao, n_ao)))[0]
    geometry = MolecularGeometry(("H", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))

    def snapshot(c, h_mo, g_mo, f_mo):
        return ElectronicSnapshot(geometry, "x", 0.0, 0.0, c, np.zeros(n_ao), h_mo, g_mo, f_mo, n_occ, n_occ, "test", "0")

    center = snapshot(coeff, h, g, fock)
    rotation = np.linalg.qr(rng.normal(size=(n_ao, n_ao)))[0]
    rotation[:, 0] *= np.sign(np.linalg.det(rotation))
    # The displaced orbitals are the center orbitals rotated; its tensors transform accordingly.
    rotated = snapshot(coeff @ rotation, rotation.T @ h @ rotation,
                       np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation, rotation, rotation, rotation, g),
                       rotation.T @ fock @ rotation)
    h_back, g_back, f_back, diag = align_to_center(rotated, center, cross_overlap_ao=overlap)
    np.testing.assert_allclose(h_back, h, atol=1e-10)
    np.testing.assert_allclose(g_back, g, atol=1e-10)
    np.testing.assert_allclose(f_back, fock, atol=1e-10)
    assert diag["alignment_residual_fro"] < 1e-10 and diag["min_singular_value"] == pytest.approx(1.0)
    assert diag["occupied_block_min_singular_value"] <= 1.0 + 1e-12


def test_explicit_active_space_is_sorted_and_validated():
    space = select_active_space(7, 5, 5, policy={"frozen_core": [0], "active": [1, 2, 3, 5, 4, 6]})
    assert space.active == (1, 2, 3, 4, 5, 6)
    with pytest.raises(ValueError):
        select_active_space(7, 5, 5, policy={"frozen_core": [0], "active": [1, 1, 2, 3, 4, 5, 6]})
    with pytest.raises(ValueError):
        select_active_space(7, 5, 5, policy="frozen_core", symbols=("Xe", "H"))
    mo_energy = np.array([-20.0, -1.3, -0.7, -0.5, -0.5 + 1e-9, 0.4, 0.6])
    _check_degenerate_orbital_boundaries(mo_energy, select_active_space(7, 5, 5, policy="frozen_core:1"), tolerance=1e-6)
    # cas(2,3) freezes orbitals 0-3 and keeps 4-6: the frozen/active boundary splits the degenerate pair 3, 4.
    with pytest.raises(ValueError):
        _check_degenerate_orbital_boundaries(mo_energy, select_active_space(7, 5, 5, policy="cas(2,3)"), tolerance=1e-6)


def test_assemble_rejects_misordered_couplings_and_unsorted_sector_basis():
    model = _synthetic_two_orbital_model(cutoffs=(1, 1))
    with pytest.raises(ValueError):
        assemble_vibronic_model(geometry=model.geometry, basis="x", active_space=model.active_space,
                                active_tensors=model.active_tensors, modes=model.modes,
                                couplings=model.couplings[::-1], cutoffs=(1, 1))
    basis = physical_sector_basis(model.layout, n_spatial_orbitals=2, num_particles=(1, 1))
    with pytest.raises(ValueError):
        sector_sparse_matrix(model.h_vibronic, basis[::-1])


@pytest.mark.skipif(not H2O_FIXTURE.is_file(), reason="Paper IV H2O linear-FD fixture not present")
def test_assembly_reproduces_paper_iv_h2o_fixture_operators():
    fixture = json.loads(H2O_FIXTURE.read_text())
    active = fixture["active_space"]
    tensors = ActiveTensors(
        float(active["scalar_energy_hartree"]),
        np.array(active["one_body_integrals"]["data"]).reshape(6, 6),
        np.array(active["two_body_integrals"]["data"]).reshape(6, 6, 6, 6),
    )
    space = ActiveSpace((0,), (1, 2, 3, 4, 5, 6), (), 4, 4, "cas_8e_6o_valence")
    modes = tuple(NormalMode(m["mode_index"], m["frequency_hartree"], np.zeros((3, 3)), m["label"]) for m in fixture["normal_modes"])
    couplings = tuple(
        LinearCoupling(r["mode_index"], "full_linear", r["scalar_derivative_hartree_per_q"],
                       np.array(r["one_body_derivative"]["data"]).reshape(6, 6),
                       np.array(r["two_body_derivative"]["data"]).reshape(6, 6, 6, 6), r["q_step_au"])
        for r in fixture["first_derivative_records"]
    )
    geometry = MolecularGeometry(tuple(fixture["geometry"]["symbols"]),
                                 np.array(fixture["geometry"]["coordinates_bohr"]["data"]).reshape(3, 3))
    model = assemble_vibronic_model(geometry=geometry, basis="sto-3g", active_space=space, active_tensors=tensors,
                                    modes=modes, couplings=couplings, cutoffs=3)
    assert model.n_qubits == 18
    for ours, theirs in [(model.h_electronic, fixture["encoded_operators"]["h_electronic"]),
                         (model.h_vibronic, fixture["encoded_operators"]["h_vibronic"]),
                         *zip(model.coupling_operators, fixture["encoded_operators"]["dH_dQ_by_mode"]),
                         *zip(model.q_operators, fixture["encoded_operators"]["q_by_mode"])]:
        reference = {t.pw2strng(): complex(t.p_coeff) for t in pauli_polynomial_from_jsonable(theirs).return_polynomial()}
        got = {t.pw2strng(): complex(t.p_coeff) for t in ours.return_polynomial()}
        assert got.keys() == reference.keys()
        assert max(abs(got[k] - reference[k]) for k in got) < 1e-12
    energies, _ = exact_sector_energies(model, n_low=2)
    assert energies[0] == pytest.approx(fixture["exact_reference"]["ground_energy_hartree"], abs=1e-8)


def test_pyscf_h2_end_to_end():
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.molecular_vibronic import build_molecular_vibronic_model

    geometry = parse_geometry("H 0 0 0; H 0 0 0.7414")
    model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="full", cutoffs=3)
    assert model.n_qubits == 6 and len(model.modes) == 1
    assert model.modes[0].frequency_cm1 == pytest.approx(5019.1, abs=1.0)
    dense = np.asarray(hamiltonian_matrix(model.h_vibronic))
    hf = hartree_fock_state(model)
    assert float(np.real(np.vdot(hf, dense @ hf))) == pytest.approx(
        computation.center.hf_energy + 0.5 * model.modes[0].frequency_hartree, abs=1e-10)
    drift = computation.couplings["full_linear"][0].diagnostics["finite_difference_drift"]
    assert all(v["rel"] < 1e-3 for v in drift.values())
    # The residual HF force equals the analytic RHF gradient projected on the mode, dE/dQ = sum g . e / sqrt(M).
    mf = computation.center.backend_handle["mf"]
    gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
    masses = computation.masses_me
    projected = float(np.sum(gradient * model.modes[0].mass_weighted_vector / np.sqrt(masses)[:, None]))
    assert computation.hf_force_by_mode["full_linear"][0] == pytest.approx(projected, abs=1e-6)
    # The frequency equals PySCF's own harmonic analysis with the same masses.
    from pyscf.hessian import thermo

    hessian = mf.Hessian().kernel()
    analysis = thermo.harmonic_analysis(mf.mol, hessian, exclude_trans=True, exclude_rot=True, mass=masses / AMU_TO_ELECTRON_MASS)
    assert model.modes[0].frequency_cm1 == pytest.approx(float(np.real(analysis["freq_wavenumber"][0])), abs=1e-3)
    # The bare frozen-reference curvature exceeds the relaxed RHF curvature (orbital relaxation lowers the energy).
    diag = computation.couplings["full_linear"][0].diagnostics
    assert diag["frozen_reference_curvature_hartree_per_q2"] > diag["rhf_hessian_curvature_hartree_per_q2"]
    coupled, _ = exact_sector_energies(model, n_low=1)
    electronic_basis = physical_sector_basis(build_register_layout(2, (), (), "binary"), n_spatial_orbitals=2, num_particles=(1, 1))
    electronic = np.linalg.eigvalsh(np.asarray(hamiltonian_matrix(model.h_electronic))[np.ix_(electronic_basis, electronic_basis)])[0]
    # Linear coupling lowers the ground state below the decoupled FCI plus zero-point energy.
    assert coupled[0] < electronic + 0.5 * model.modes[0].frequency_hartree


def test_pyscf_nearly_linear_molecule_keeps_both_bends():
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.molecular_vibronic import pyscf_hessian, pyscf_rhf_snapshot

    for offset in (0.0, 1e-9):
        geometry = MolecularGeometry(("H", "Be", "H"), np.array([[0.0, 0.0, -1.30], [offset, 0.0, 0.0], [0.0, 0.0, 1.30]]) / 0.529177210903)
        center = pyscf_rhf_snapshot(geometry, basis="sto-3g")
        masses = isotope_masses_me(geometry.symbols)
        modes = vibrational_modes_from_hessian(pyscf_hessian(center), coordinates_bohr=geometry.coordinates_bohr, masses_me=masses)
        assert len(modes) == 4
        assert modes[0].frequency_cm1 == pytest.approx(modes[1].frequency_cm1, abs=1e-2)


@pytest.mark.skipif(not H2O_FIXTURE.is_file(), reason="Paper IV H2O linear-FD fixture not present")
def test_pyscf_h2o_reproduces_paper_iv_model():
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.molecular_vibronic import build_molecular_vibronic_model

    fixture = json.loads(H2O_FIXTURE.read_text())
    symbols = tuple(fixture["geometry"]["symbols"])
    masses_amu = {s: m / AMU_TO_ELECTRON_MASS for s, m in zip(symbols, fixture["geometry"]["masses_me"]["data"])}
    geometry = MolecularGeometry(symbols, np.array(fixture["geometry"]["coordinates_bohr"]["data"]).reshape(3, 3))
    model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="frozen_core", cutoffs=3,
                                                        masses_amu=masses_amu)
    assert model.n_qubits == 18 and model.active_space.active == (1, 2, 3, 4, 5, 6)
    for ours, theirs in zip(model.modes, fixture["normal_modes"]):
        assert ours.frequency_cm1 == pytest.approx(theirs["frequency_cm1"], abs=2e-3)
    assert computation.active_tensors.scalar == pytest.approx(fixture["active_space"]["scalar_energy_hartree"], abs=1e-9)
    energies, _ = exact_sector_energies(model, n_low=1)
    assert energies[0] == pytest.approx(fixture["exact_reference"]["ground_energy_hartree"], abs=2e-7)
