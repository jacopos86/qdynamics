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
    ALIGNMENTS,
    AMU_TO_ELECTRON_MASS,
    HARMONIC_CURVATURES,
    HARTREE_TO_CM_INV,
    MODE_BASES,
    ActiveSpace,
    ActiveTensors,
    ElectronicSnapshot,
    LinearCoupling,
    MolecularGeometry,
    NormalMode,
    _check_degenerate_orbital_boundaries,
    _oscillator_frequency,
    active_response_curvature,
    active_response_matrix,
    align_to_center,
    assemble_vibronic_model,
    build_register_layout,
    closed_shell_determinant_energy,
    compute_linear_couplings,
    displace_along_mode,
    exact_sector_energies,
    fermion_polynomial_from_tensors,
    frozen_core_contract,
    frozen_core_contract_one_body,
    hartree_fock_state,
    isotope_masses_me,
    load_vibronic_hamiltonian,
    model_consistent_modes,
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
    # The decided defaults (2026-11-02): K_PES + chi oscillators, canonical frozen core.
    assert args.harmonic_curvature == "pes_consistent" == HARMONIC_CURVATURES[0]
    assert args.alignment == "block_diagonal" == ALIGNMENTS[0]
    assert args.mode_basis is None and MODE_BASES[0] == "model_consistent"   # resolved by the builder per curvature


def test_oscillator_frequency_refuses_a_non_positive_curvature():
    mode = NormalMode(0, 0.01, np.zeros((2, 3)), "mode_0")
    assert _oscillator_frequency(4.0e-4, mode=mode, source="test") == pytest.approx(0.02)
    for bad in (0.0, -1.0e-5, float("nan")):
        with pytest.raises(ValueError, match="not a minimum"):
            _oscillator_frequency(bad, mode=mode, source="test")


@pytest.mark.parametrize("two_body_coupling", [True, False])
def test_active_response_curvature_matches_second_order_perturbation_theory(two_body_coupling):
    model = _synthetic_two_orbital_model(two_body_coupling=two_body_coupling)
    t0, coupling = model.active_tensors, model.couplings[0]
    assert (coupling.two_body is None) is (not two_body_coupling)
    particles = (1, 1)
    basis = physical_sector_basis(build_register_layout(2, (), (), "binary"), n_spatial_orbitals=2, num_particles=particles)

    def energy(t, num_particles):
        dense = np.asarray(hamiltonian_matrix(fermion_polynomial_from_tensors(t.scalar, t.one_body, t.two_body, num_particles)))
        return float(np.linalg.eigvalsh(dense[np.ix_(basis, basis)])[0])

    chi = active_response_curvature(t0, coupling, particles, q_step_au=1.0e-3, energy=energy)
    h_el = np.asarray(hamiltonian_matrix(model.h_electronic))[np.ix_(basis, basis)]
    c = np.asarray(hamiltonian_matrix(model.coupling_operators[0]))[np.ix_(basis, basis)]
    evals, evecs = np.linalg.eigh(h_el)
    amplitudes = evecs[:, 1:].conj().T @ c @ evecs[:, 0]
    # chi = -d2E/dQ2 of E[H_el + Q C] is the second-order lowering 2 sum_k |<k|C|0>|^2 / (E_k - E_0).
    expected = float(2.0 * np.sum(np.abs(amplitudes) ** 2 / (evals[1:] - evals[0])))
    assert expected > 0.0 and chi == pytest.approx(expected, rel=1e-5)
    # A scalar in the coupling (the reference-force counter-term) shifts E linearly and drops out
    # (to round-off: the shift enters the energies at +/- 0.3 q and is differenced away over q^2).
    shifted = LinearCoupling(coupling.mode_index, coupling.model, coupling.scalar + 0.3, coupling.one_body,
                             coupling.two_body, coupling.q_step_au)
    assert active_response_curvature(t0, shifted, particles, q_step_au=1.0e-3, energy=energy) == pytest.approx(chi, rel=1e-6)
    with pytest.raises(ValueError):
        active_response_curvature(t0, coupling, particles, q_step_au=0.0, energy=energy)


def _dense_sector_energy_factory(n_orbitals, particles):
    basis = physical_sector_basis(build_register_layout(n_orbitals, (), (), "binary"), n_spatial_orbitals=n_orbitals, num_particles=particles)

    def energy(t, num_particles):
        dense = np.asarray(hamiltonian_matrix(fermion_polynomial_from_tensors(t.scalar, t.one_body, t.two_body, num_particles)))
        return float(np.linalg.eigvalsh(dense[np.ix_(basis, basis)])[0])

    return basis, energy


def test_active_response_matrix_matches_second_order_perturbation_theory():
    model = _synthetic_two_orbital_model()
    particles = (1, 1)
    basis, energy = _dense_sector_energy_factory(2, particles)
    chi = active_response_matrix(model.active_tensors, model.couplings, particles, q_step_au=1.0e-3, energy=energy)
    h_el = np.asarray(hamiltonian_matrix(model.h_electronic))[np.ix_(basis, basis)]
    evals, evecs = np.linalg.eigh(h_el)
    amplitudes = [evecs[:, 1:].conj().T @ np.asarray(hamiltonian_matrix(c))[np.ix_(basis, basis)] @ evecs[:, 0]
                  for c in model.coupling_operators]
    # chi_{mu nu} = 2 sum_k <0|C_mu|k><k|C_nu|0> / (E_k - E_0): symmetric, positive on the diagonal.
    expected = np.array([[2.0 * float(np.real(np.sum(np.conj(a) * b / (evals[1:] - evals[0])))) for b in amplitudes] for a in amplitudes])
    np.testing.assert_allclose(chi, expected, rtol=1e-5, atol=1e-12)
    assert chi[0, 1] == chi[1, 0] and abs(chi[0, 1]) > 1e-4
    diagonal = [active_response_curvature(model.active_tensors, c, particles, q_step_au=1.0e-3, energy=energy) for c in model.couplings]
    np.testing.assert_allclose(np.diag(chi), diagonal, rtol=1e-12)
    given = active_response_matrix(model.active_tensors, model.couplings, particles, q_step_au=1.0e-3, energy=energy, diagonal=(1.0, 2.0))
    assert given[0, 0] == 1.0 and given[1, 1] == 2.0 and given[0, 1] == chi[0, 1]
    with pytest.raises(ValueError):
        active_response_matrix(model.active_tensors, model.couplings, particles, q_step_au=1.0e-3, energy=energy, diagonal=(1.0,))


def test_model_consistent_modes_rotate_modes_and_couplings_together():
    model = _synthetic_two_orbital_model()
    vectors = np.zeros((2, 2, 3))
    vectors[0, 0, 0] = 1.0
    vectors[1, 1, 2] = 1.0
    modes = tuple(NormalMode(k, 0.02 * (k + 1), vectors[k], f"mode_{k}") for k in range(2))
    w = np.array([[4.0e-4, 1.0e-4], [1.0e-4, 9.0e-4]])
    new_modes, new_couplings, u = model_consistent_modes(modes, model.couplings, w)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-12)
    evals = np.linalg.eigvalsh(w)
    assert [m.frequency_hartree for m in new_modes] == pytest.approx(np.sqrt(evals).tolist(), rel=1e-12)
    assert [m.mode_index for m in new_modes] == [0, 1] == [c.mode_index for c in new_couplings]
    for a, (mode, coupling) in enumerate(zip(new_modes, new_couplings)):
        np.testing.assert_allclose(mode.mass_weighted_vector, sum(u[nu, a] * vectors[nu] for nu in range(2)), atol=1e-12)
        assert float(np.linalg.norm(mode.mass_weighted_vector)) == pytest.approx(1.0)
        assert u[np.argmax(np.abs(u[:, a])), a] > 0.0
        assert coupling.diagnostics["mode_basis"] == "model_consistent" and coupling.diagnostics["input_mode_weights"] == u[:, a].tolist()
    # The couplings rotate with the modes: C_nu = sum_a U_{nu a} C'_a reconstructs the inputs exactly.
    for nu, c in enumerate(model.couplings):
        assert sum(u[nu, a] * new_couplings[a].scalar for a in range(2)) == pytest.approx(c.scalar, abs=1e-14)
        np.testing.assert_allclose(sum(u[nu, a] * new_couplings[a].one_body for a in range(2)), c.one_body, atol=1e-14)
        np.testing.assert_allclose(sum(u[nu, a] * new_couplings[a].two_body for a in range(2)), c.two_body, atol=1e-14)
    # A diagonal W leaves everything in place (identity rotation, positive orientation).
    same_modes, same_couplings, identity = model_consistent_modes(modes, model.couplings, np.diag([4.0e-4, 9.0e-4]))
    np.testing.assert_allclose(identity, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(same_couplings[1].one_body, model.couplings[1].one_body, atol=1e-14)
    assert same_modes[1].frequency_hartree == pytest.approx(0.03)
    with pytest.raises(ValueError, match="symmetric"):
        model_consistent_modes(modes, model.couplings, np.array([[4.0e-4, 1.0e-4], [2.0e-4, 9.0e-4]]))
    with pytest.raises(ValueError, match="not a minimum"):
        model_consistent_modes(modes, model.couplings, np.diag([4.0e-4, -1.0e-5]))
    one_body_model = _synthetic_two_orbital_model(two_body_coupling=False)
    with pytest.raises(ValueError, match="two-body"):
        model_consistent_modes(modes, (model.couplings[0], one_body_model.couplings[1]), w)
    rotated_one_body = model_consistent_modes(modes, one_body_model.couplings, w)[1]
    assert all(c.two_body is None for c in rotated_one_body)


def test_block_overlap_guard_detects_an_orbital_crossing_a_block_boundary():
    rng = np.random.default_rng(11)
    n_ao, n_occ = 4, 2
    h, g = _random_chemist_tensors(rng, n_ao)
    fock = np.diag([-20.0, -1.0, 0.5, 1.0])
    geometry = MolecularGeometry(("H", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))

    def snapshot(rotation):
        return ElectronicSnapshot(geometry, "x", 0.0, 0.0, np.eye(n_ao) @ rotation, np.diag(rotation.T @ fock @ rotation),
                                  rotation.T @ h @ rotation,
                                  np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation, rotation, rotation, rotation, g),
                                  rotation.T @ fock @ rotation, n_occ, n_occ, "test", "0")

    center = snapshot(np.eye(n_ao))
    space = ActiveSpace((0,), (1, 2, 3), (), 1, 1, "frozen_core:1")
    partition = (space.frozen_core, space.active, space.external)
    # The "displaced" SCF returns the same orbitals with the core and the first valence orbital
    # swapped in energy order: by canonical index the wrong orbital is frozen.
    swapped = snapshot(np.eye(n_ao)[[1, 0, 2, 3]])
    _, _, _, diag = align_to_center(swapped, center, cross_overlap_ao=np.eye(n_ao), blocks=partition)
    assert diag["block_min_singular_values"][0] == pytest.approx(0.0, abs=1e-12)
    assert diag["block_min_singular_values"][2] == 1.0   # empty external block
    # The swap stays inside the occupied set and inside the full orbital set, so the older
    # diagnostics are blind to it.
    assert diag["min_singular_value"] == pytest.approx(1.0) and diag["occupied_block_min_singular_value"] == pytest.approx(1.0)
    _, _, _, diag_full = align_to_center(swapped, center, cross_overlap_ao=np.eye(n_ao), partition=partition)
    assert diag_full["block_diagonal"] is False and diag_full["block_min_singular_values"][0] == pytest.approx(0.0, abs=1e-12)
    mode = NormalMode(0, 0.01, np.zeros((2, 3)), "mode_0")
    for alignment in ALIGNMENTS:
        with pytest.raises(ValueError, match="crossed"):
            compute_linear_couplings(center, (mode,), masses_me=np.ones(2), active_space=space, q_step_alt_au=None,
                                     alignment=alignment, snapshot_factory=lambda geometry: swapped,
                                     cross_overlap_factory=lambda c, s: np.eye(n_ao))
    # Without a crossing the same path runs and records the per-block overlaps.
    computation = compute_linear_couplings(center, (mode,), masses_me=np.ones(2), active_space=space, q_step_alt_au=None,
                                           snapshot_factory=lambda geometry: center, cross_overlap_factory=lambda c, s: np.eye(n_ao))
    assert computation.couplings["full_linear"][0].diagnostics["alignment_block_min_singular_values"] == [1.0, 1.0, 1.0]
    assert computation.fci_surface_curvature_by_mode == () and computation.center_fci_energy is None


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
    assert diag["block_diagonal"] is False


def test_block_diagonal_alignment_recovers_a_block_rotation_and_validates_blocks():
    rng = np.random.default_rng(5)
    n_ao, n_occ = 4, 2
    h, g = _random_chemist_tensors(rng, n_ao)
    fock = 0.5 * (rng.normal(size=(n_ao, n_ao)) + rng.normal(size=(n_ao, n_ao)).T)
    coeff = np.linalg.qr(rng.normal(size=(n_ao, n_ao)))[0]
    geometry = MolecularGeometry(("H", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))

    def snapshot(c, h_mo, g_mo, f_mo):
        return ElectronicSnapshot(geometry, "x", 0.0, 0.0, c, np.zeros(n_ao), h_mo, g_mo, f_mo, n_occ, n_occ, "test", "0")

    def rotated_by(rotation):
        return snapshot(coeff @ rotation, rotation.T @ h @ rotation,
                        np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation, rotation, rotation, rotation, g), rotation.T @ fock @ rotation)

    center = snapshot(coeff, h, g, fock)
    blocks = ((0, 1), (2, 3))
    block_rotation = np.zeros((n_ao, n_ao))
    for block in blocks:
        q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        block_rotation[np.ix_(block, block)] = q
    h_back, g_back, f_back, diag = align_to_center(rotated_by(block_rotation), center, cross_overlap_ao=np.eye(n_ao), blocks=blocks)
    np.testing.assert_allclose(h_back, h, atol=1e-10)
    np.testing.assert_allclose(g_back, g, atol=1e-10)
    np.testing.assert_allclose(f_back, fock, atol=1e-10)
    assert diag["alignment_residual_fro"] < 1e-10 and diag["block_diagonal"] is True
    # A rotation that mixes the blocks is undone by the full-set alignment but, by design,
    # not by the block-diagonal one: the inter-block part stays in the residual.
    mixing = np.linalg.qr(rng.normal(size=(n_ao, n_ao)))[0]
    mixed = rotated_by(mixing)
    h_full, _, _, diag_full = align_to_center(mixed, center, cross_overlap_ao=np.eye(n_ao))
    np.testing.assert_allclose(h_full, h, atol=1e-10)
    _, _, _, diag_block = align_to_center(mixed, center, cross_overlap_ao=np.eye(n_ao), blocks=blocks)
    assert diag_full["alignment_residual_fro"] < 1e-10 < diag_block["alignment_residual_fro"]
    assert diag_block["min_singular_value"] == pytest.approx(diag_full["min_singular_value"])
    for bad in (((0, 1), (2,)), ((0, 1), (1, 2, 3)), ((0, 1, 2, 3, 4),)):
        with pytest.raises(ValueError, match="partition"):
            align_to_center(mixed, center, cross_overlap_ao=np.eye(n_ao), blocks=bad)
    # Empty blocks (no frozen core, no external orbitals) are allowed.
    h_one, _, _, _ = align_to_center(mixed, center, cross_overlap_ao=np.eye(n_ao), blocks=((), (0, 1, 2, 3), ()))
    np.testing.assert_allclose(h_one, h, atol=1e-10)


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
    from src.quantum.chemistry.molecular_vibronic import build_molecular_vibronic_model, pyscf_rhf_snapshot

    geometry = parse_geometry("H 0 0 0; H 0 0 0.7414")
    model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="full", cutoffs=3)
    assert model.n_qubits == 6 and len(model.modes) == 1
    rhf_mode = computation.modes[0]   # the RHF normal mode; the model's oscillator is K_PES + chi
    assert rhf_mode.frequency_cm1 == pytest.approx(5019.1, abs=1.0)
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
    projected = float(np.sum(gradient * rhf_mode.mass_weighted_vector / np.sqrt(masses)[:, None]))
    assert computation.hf_force_by_mode["full_linear"][0] == pytest.approx(projected, abs=1e-6)
    # The frequency equals PySCF's own harmonic analysis with the same masses.
    from pyscf import fci
    from pyscf.hessian import thermo

    hessian = mf.Hessian().kernel()
    analysis = thermo.harmonic_analysis(mf.mol, hessian, exclude_trans=True, exclude_rot=True, mass=masses / AMU_TO_ELECTRON_MASS)
    assert rhf_mode.frequency_cm1 == pytest.approx(float(np.real(analysis["freq_wavenumber"][0])), abs=1e-3)
    # The bare frozen-reference curvature exceeds the relaxed RHF curvature (orbital relaxation lowers the energy).
    diag = computation.couplings["full_linear"][0].diagnostics
    assert diag["frozen_reference_curvature_hartree_per_q2"] > diag["rhf_hessian_curvature_hartree_per_q2"]
    # Default oscillator: omega^2 = K_PES + chi, with K_PES the curvature of PySCF's own FCI energy
    # along the mode (recomputed here independently of the builder) and chi > 0 the model's response.
    prov = model.provenance
    assert prov["harmonic_curvature"] == "pes_consistent" and prov["alignment"] == "block_diagonal"
    k_pes, chi = prov["fci_surface_curvature_hartree_per_q2"][0], prov["active_response_hartree_per_q2"][0]
    assert model.modes[0].frequency_hartree ** 2 == pytest.approx(k_pes + chi, rel=1e-12)
    assert model.modes[0].frequency_cm1 == pytest.approx(prov["pes_consistent_frequencies_cm1"][0])
    assert chi > 0.0 and k_pes < rhf_mode.frequency_hartree ** 2   # the FCI stretch is softer than the RHF one

    def fci_energy_at(q):
        snap = pyscf_rhf_snapshot(displace_along_mode(geometry, rhf_mode, masses, q), basis="sto-3g")
        return float(fci.FCI(snap.backend_handle["mf"]).kernel()[0])

    assert k_pes == pytest.approx((fci_energy_at(0.1) - 2.0 * fci_energy_at(0.0) + fci_energy_at(-0.1)) / 0.01, rel=1e-6)
    assert computation.center_fci_energy == pytest.approx(fci_energy_at(0.0), abs=1e-9)
    assert diag["fci_surface_curvature_hartree_per_q2"] == k_pes and prov["active_response_drift"][0]["rel"] < 1e-3
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
    # The Paper IV H2O model as run: RHF-Hessian oscillators and the full-set orbital alignment,
    # both legacy choices now (the defaults are pes_consistent and block_diagonal).
    model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="frozen_core", cutoffs=3,
                                                        masses_amu=masses_amu, harmonic_curvature="rhf_hessian", alignment="full")
    assert model.n_qubits == 18 and model.active_space.active == (1, 2, 3, 4, 5, 6)
    assert model.provenance["alignment"] == "full" and computation.fci_surface_curvature_by_mode == ()
    assert model.provenance["mode_basis"] == "rhf_normal_modes"   # resolved: no response matrix under rhf_hessian
    for ours, theirs in zip(model.modes, fixture["normal_modes"]):
        assert ours.frequency_cm1 == pytest.approx(theirs["frequency_cm1"], abs=2e-3)
    assert computation.active_tensors.scalar == pytest.approx(fixture["active_space"]["scalar_energy_hartree"], abs=1e-9)
    energies, _ = exact_sector_energies(model, n_low=1)
    assert energies[0] == pytest.approx(fixture["exact_reference"]["ground_energy_hartree"], abs=2e-7)


def test_block_diagonal_alignment_keeps_the_canonical_frozen_core():
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.molecular_vibronic import (
        active_space_fci_energy, pyscf_ao_cross_overlap, pyscf_hessian, pyscf_rhf_snapshot)

    geometry = parse_geometry("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692")
    center = pyscf_rhf_snapshot(geometry, basis="sto-3g")
    space = select_active_space(center.n_spatial_orbitals, center.n_alpha, center.n_beta, policy="frozen_core", symbols=geometry.symbols)
    assert space.frozen_core == (0,)
    masses = isotope_masses_me(geometry.symbols)
    modes = vibrational_modes_from_hessian(pyscf_hessian(center), coordinates_bohr=geometry.coordinates_bohr, masses_me=masses)
    particles = (space.n_alpha, space.n_beta)
    blocks = (space.frozen_core, space.active, space.external)
    center_fci = active_space_fci_energy(frozen_core_contract(center.h_mo, center.eri_mo, center.nuclear_repulsion, space), particles)
    for mode in modes:
        energies = {"canonical": [], "block_diagonal": [], "full": []}
        for q in (0.1, -0.1):
            snap = pyscf_rhf_snapshot(displace_along_mode(geometry, mode, masses, q), basis="sto-3g")
            overlap = pyscf_ao_cross_overlap(center, snap)
            energies["canonical"].append(active_space_fci_energy(
                frozen_core_contract(snap.h_mo, snap.eri_mo, snap.nuclear_repulsion, space), particles))
            for name, kwargs in (("block_diagonal", {"blocks": blocks}), ("full", {})):
                h, g, _, diag = align_to_center(snap, center, cross_overlap_ao=overlap, **kwargs)
                assert diag["block_diagonal"] is (name == "block_diagonal")
                energies[name].append(active_space_fci_energy(frozen_core_contract(h, g, snap.nuclear_repulsion, space), particles))
        # The frozen-core FCI energy is invariant to rotations inside the blocks, so the block-diagonal
        # alignment reproduces the displaced SCF's own canonical-core energy at every point.
        np.testing.assert_allclose(energies["block_diagonal"], energies["canonical"], rtol=0.0, atol=1e-10)
        # The full-set rotation carries the R0 core along and freezes it. That core differs from the
        # displaced canonical core by a part odd in the displacement, which cancels in the second
        # difference, and an even part that raises the energy at second order, which the second
        # difference turns into spurious curvature of the size of the force constants (~1e-4 Ha/Q^2).
        curvature = {k: (v[0] - 2.0 * center_fci + v[1]) / 0.01 for k, v in energies.items()}
        assert curvature["full"] - curvature["canonical"] > 1.0e-5
        assert abs(curvature["block_diagonal"] - curvature["canonical"]) < 1.0e-8
        # K_PES must not depend on the alignment: the coupling computation evaluates the FCI surface
        # from each displaced SCF's own canonical tensors under either alignment.
        for name in ALIGNMENTS:
            computation = compute_linear_couplings(center, (mode,), masses_me=masses, active_space=space,
                                                   q_step_alt_au=None, alignment=name, fci_surface=True)
            assert computation.alignment == name
            assert computation.fci_surface_curvature_by_mode[0] == pytest.approx(curvature["canonical"], abs=1e-9)
            assert computation.center_fci_energy == pytest.approx(center_fci, abs=1e-10)


def test_pes_consistent_model_carries_the_fci_surface_curvature():
    pytest.importorskip("pyscf")
    from pyscf import fci

    from src.quantum.chemistry.molecular_vibronic import build_molecular_vibronic_model, pyscf_ao_cross_overlap, pyscf_rhf_snapshot

    # The Paper IV H2O fixture geometry (Bohr); its masses are the module's default isotope masses.
    geometry = MolecularGeometry(("O", "H", "H"), np.array([[0.0, 7.452305433104561e-16, -0.06340060329635555],
                                                            [0.0, 1.432278617817422, 1.1388573462159217],
                                                            [0.0, -1.4322786178174232, 1.1388573462159226]]))
    # The diagonal construction in the RHF normal-mode basis (the model-consistent default is tested below).
    model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="frozen_core", cutoffs=3,
                                                        mode_basis="rhf_normal_modes")
    prov = model.provenance
    assert prov["harmonic_curvature"] == "pes_consistent" and prov["alignment"] == "block_diagonal"
    assert prov["mode_basis"] == "rhf_normal_modes" and "model_consistent_rotation" not in prov
    k_pes = prov["fci_surface_curvature_hartree_per_q2"]
    chi = prov["active_response_hartree_per_q2"]
    for mode, k, c in zip(model.modes, k_pes, chi):
        assert mode.frequency_hartree ** 2 == pytest.approx(k + c, rel=1e-12)
        assert c > 0.0 and k > 0.0
    # Measured 2026-11-02 for the Paper IV geometry and masses (the hand scripts, whose couplings
    # carried the full-set alignment, gave 2138 / 5302 / 5061); the RHF Hessian gives 2171 / 4139 / 4389.
    assert [m.frequency_cm1 for m in model.modes] == pytest.approx([2142.2, 5309.8, 5061.8], abs=0.5)
    assert computation.center_fci_energy == pytest.approx(-75.0203648544, abs=1e-8)
    assert all(v > 0.99 for c in model.couplings for v in c.diagnostics["alignment_block_min_singular_values"])
    # Second-order identity, exact for the analytic derivatives: the curvature of the frozen-core FCI
    # surface equals the curvature of <Psi0|H(Q)|Psi0> (the R0 FCI state carried rigidly along the
    # mode with the block-diagonal alignment) minus the model's response chi. With the full-set
    # alignment the residual is ~1e-4 Ha/Q^2 (the frozen rotated core); here it is finite-difference noise.
    t0 = computation.active_tensors
    n = t0.one_body.shape[0]
    particles = (computation.active_space.n_alpha, computation.active_space.n_beta)
    blocks = (computation.active_space.frozen_core, computation.active_space.active, computation.active_space.external)
    e0, c0 = fci.direct_spin1.kernel(t0.one_body, t0.two_body, n, particles, ecore=t0.scalar, tol=1e-12, max_cycle=300)
    for nu, mode in enumerate(computation.modes):
        rigid = []
        for q in (0.1, -0.1):
            snap = pyscf_rhf_snapshot(displace_along_mode(geometry, mode, computation.masses_me, q), basis="sto-3g")
            h, g, _, _ = align_to_center(snap, computation.center, cross_overlap_ao=pyscf_ao_cross_overlap(computation.center, snap), blocks=blocks)
            t = frozen_core_contract(h, g, snap.nuclear_repulsion, computation.active_space)
            rigid.append(float(fci.direct_spin1.energy(t.one_body, t.two_body, c0, n, particles)) + t.scalar)
        k_rigid = (rigid[0] - 2.0 * e0 + rigid[1]) / 0.01
        assert abs(k_rigid - chi[nu] - k_pes[nu]) < 1e-8
    # The exactly solved model sits near HF (x) vacuum: no mode holds more than a fraction of a quantum
    # (at these cutoffs the symmetric stretch holds 0.21 quanta here against 0.64 for the RHF-Hessian
    # model, and 0.27 against 2.3 at cutoffs (4, 8, 4)).
    basis = physical_sector_basis(model.layout, n_spatial_orbitals=n, num_particles=particles)
    energies, state = exact_sector_energies(model, n_low=1, return_state=True)
    vector = state[basis]
    occupations = [float(np.real(np.vdot(vector, sector_sparse_matrix(op, basis) @ vector))) for op in model.n_operators]
    assert max(occupations) < 0.4
    assert energies[0] < computation.center_fci_energy + 0.5 * sum(m.frequency_hartree for m in model.modes)
    assert energies[0] == pytest.approx(-75.0002755275, abs=1e-6)   # the decided model's exact reference at cutoffs 3


def _adiabatic_hessian(model, particles, step=0.1):
    """Mixed central second differences of E_FCI[H_el + sum_a Q_a C_a] + 1/2 sum_a omega_a^2 Q_a^2 at Q = 0."""
    from src.quantum.chemistry.molecular_vibronic import active_space_fci_energy

    t0, modes, couplings = model.active_tensors, model.modes, model.couplings
    m = len(modes)

    def energy(q):
        scalar, one, two = t0.scalar, t0.one_body, t0.two_body
        for qa, c in zip(q, couplings):
            scalar, one = scalar + qa * c.scalar, one + qa * np.asarray(c.one_body)
            two = two if c.two_body is None else two + qa * np.asarray(c.two_body)
        return active_space_fci_energy(ActiveTensors(scalar, one, two), particles) + 0.5 * sum(
            float(md.frequency_hartree) ** 2 * qa ** 2 for md, qa in zip(modes, q))

    e0 = energy(np.zeros(m))
    hess = np.zeros((m, m))
    for a in range(m):
        q = np.zeros(m); q[a] = step
        hess[a, a] = (energy(q) - 2.0 * e0 + energy(-q)) / step ** 2
        for b in range(a + 1, m):
            vals = {}
            for sa in (1, -1):
                for sb in (1, -1):
                    q = np.zeros(m); q[a], q[b] = sa * step, sb * step
                    vals[sa, sb] = energy(q)
            hess[a, b] = hess[b, a] = (vals[1, 1] - vals[1, -1] - vals[-1, 1] + vals[-1, -1]) / (4.0 * step ** 2)
    return hess


def test_model_consistent_modes_carry_the_full_fci_hessian_for_h2o():
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.molecular_vibronic import build_molecular_vibronic_model

    geometry = MolecularGeometry(("O", "H", "H"), np.array([[0.0, 7.452305433104561e-16, -0.06340060329635555],
                                                            [0.0, 1.432278617817422, 1.1388573462159217],
                                                            [0.0, -1.4322786178174232, 1.1388573462159226]]))
    with pytest.raises(ValueError, match="model_consistent"):
        build_molecular_vibronic_model(geometry, basis="sto-3g", cutoffs=1, harmonic_curvature="rhf_hessian", mode_basis="model_consistent")
    # The default since the 2026-11-02 decision: model-consistent modes.
    model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="frozen_core", cutoffs=3)
    prov = model.provenance
    particles = (computation.active_space.n_alpha, computation.active_space.n_beta)
    k_pes = np.array(prov["fci_surface_hessian_hartree_per_q2"])
    chi = np.array(prov["active_response_matrix_hartree_per_q2"])
    u = np.array(prov["model_consistent_rotation"])
    assert prov["mode_basis"] == "model_consistent" == MODE_BASES[0]
    np.testing.assert_allclose(k_pes, k_pes.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(k_pes), prov["fci_surface_curvature_hartree_per_q2"], rtol=0, atol=1e-15)
    # Bend and symmetric stretch (A1) couple through the electrons, chi_01 = 2.3e-5 Ha/Q^2, while the FCI
    # surface itself is nearly separable there (K_01 = -6.4e-7); the asymmetric stretch (B2) couples to neither.
    assert chi[0, 1] == pytest.approx(2.29e-5, abs=1e-6) and abs(k_pes[0, 1]) < 2e-6
    assert max(abs(k_pes[0, 2]), abs(k_pes[1, 2]), abs(chi[0, 2]), abs(chi[1, 2])) < 1e-9
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=1e-12)
    assert abs(abs(u[2, 1]) - 1.0) < 1e-6   # the B2 mode stays pure and lands second in ascending frequency
    assert abs(np.degrees(np.arctan2(u[1, 0], u[0, 0]))) == pytest.approx(2.6, abs=0.1)   # A1 mixing
    assert [m.frequency_hartree ** 2 for m in model.modes] == pytest.approx(np.linalg.eigvalsh(k_pes + chi).tolist(), rel=1e-10)
    assert [m.frequency_cm1 for m in model.modes] == pytest.approx([2130.8, 5061.8, 5314.4], abs=0.5)
    # The model's adiabatic Hessian (independent mixed differences of its own FCI surface) is the full
    # FCI Hessian rotated into the model modes, so its eigenfrequencies are those of the FCI surface.
    hessian = _adiabatic_hessian(model, particles)
    np.testing.assert_allclose(hessian, u.T @ k_pes @ u, atol=1e-9)
    assert (np.sqrt(np.linalg.eigvalsh(hessian)) * HARTREE_TO_CM_INV).tolist() == pytest.approx([1974.9, 3997.7, 4281.8], abs=0.2)
    # Exact reference of this model at cutoffs 3 (18 qubits), and its ground state near HF (x) vacuum.
    basis = physical_sector_basis(model.layout, n_spatial_orbitals=6, num_particles=particles)
    energies, state = exact_sector_energies(model, n_low=1, return_state=True)
    vector = state[basis]
    occupations = [float(np.real(np.vdot(vector, sector_sparse_matrix(op, basis) @ vector))) for op in model.n_operators]
    assert energies[0] == pytest.approx(-74.9999117706, abs=1e-6) and max(occupations) < 0.3


def test_rhf_normal_mode_model_records_the_uncompensated_cross_response():
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.molecular_vibronic import build_molecular_vibronic_model

    geometry = parse_geometry("H 0 0 0; H 0 0 0.7414")
    for mode_basis in MODE_BASES:
        model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="full", cutoffs=1, mode_basis=mode_basis)
        # One mode: the model-consistent rotation is the identity and nothing changes.
        assert model.modes[0].frequency_hartree ** 2 == pytest.approx(
            computation.fci_surface_curvature_by_mode[0] + model.provenance["active_response_hartree_per_q2"][0], rel=1e-12)
        assert np.array(model.provenance["active_response_matrix_hartree_per_q2"]).shape == (1, 1)
        if mode_basis == "model_consistent":
            assert model.provenance["model_consistent_rotation"] == [[1.0]]
            assert model.provenance["model_consistent_frequencies_cm1"] == [model.modes[0].frequency_cm1]
        else:
            assert "model_consistent_rotation" not in model.provenance


def test_pes_consistent_response_follows_the_assembled_coupling_model():
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.molecular_vibronic import build_molecular_vibronic_model

    geometry = parse_geometry("H 0 0 0; H 0 0 0.7414")
    chi = {}
    for coupling_model in ("full_linear", "mean_field_fd"):
        model, computation = build_molecular_vibronic_model(geometry, basis="sto-3g", active_space="full", cutoffs=1,
                                                            coupling_model=coupling_model)
        coupling = model.couplings[0]
        assert coupling.model == coupling_model and (coupling.two_body is None) is (coupling_model == "mean_field_fd")
        chi[coupling_model] = active_response_curvature(model.active_tensors, coupling, (1, 1), q_step_au=0.1)
        # omega^2 - chi[C] = K_PES for the coupling that went into the model, counter-term included.
        assert model.modes[0].frequency_hartree ** 2 - chi[coupling_model] == pytest.approx(
            computation.fci_surface_curvature_by_mode[0], rel=1e-10)
        assert model.provenance["active_response_hartree_per_q2"][0] == pytest.approx(chi[coupling_model], rel=1e-12)
    assert chi["mean_field_fd"] != pytest.approx(chi["full_linear"], rel=1e-3)
