from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Sequence

import numpy as np

from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action, energy_via_one_apply
from src.quantum.hartree_fock_reference_state import hartree_fock_bitstring
from src.quantum.hubbard_latex_python_pairs import (
    boson_operator,
    boson_qubits_per_site,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial, fermion_minus_operator, fermion_plus_operator
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

from src.quantum.chemistry.molecular_hamiltonian import build_restricted_closed_shell_molecular_hamiltonian
from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
from src.quantum.chemistry.psi4_adapter import (
    RestrictedClosedShellMolecularProblem,
    RestrictedClosedShellPsi4Snapshot,
    build_h2_snapshot_from_psi4,
)


ANGSTROM_TO_BOHR = 1.8897259886
PROTON_MASS_ELECTRON = 1836.15267343


@dataclass(frozen=True)
class VibronicH2Model:
    bond_length_angstrom: float
    bond_step_angstrom: float
    basis: str
    n_ph_max: int
    boson_encoding: str
    n_fermion_qubits: int
    n_boson_qubits: int
    n_total_qubits: int
    omega_au: float
    reduced_mass_au: float
    x_zpf_bohr: float
    curvature_au_per_bohr2: float
    electronic_exact_energy_minus: float
    electronic_exact_energy_center: float
    electronic_exact_energy_plus: float
    h_electronic: PauliPolynomial
    dH_dR: PauliPolynomial
    h_vibronic: PauliPolynomial
    pool: tuple[AnsatzTerm, ...]
    psi_ref: np.ndarray

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "bond_length_angstrom": float(self.bond_length_angstrom),
            "bond_step_angstrom": float(self.bond_step_angstrom),
            "basis": str(self.basis),
            "n_ph_max": int(self.n_ph_max),
            "boson_encoding": str(self.boson_encoding),
            "n_fermion_qubits": int(self.n_fermion_qubits),
            "n_boson_qubits": int(self.n_boson_qubits),
            "n_total_qubits": int(self.n_total_qubits),
            "omega_au": float(self.omega_au),
            "reduced_mass_au": float(self.reduced_mass_au),
            "x_zpf_bohr": float(self.x_zpf_bohr),
            "curvature_au_per_bohr2": float(self.curvature_au_per_bohr2),
            "electronic_exact_energy_minus": float(self.electronic_exact_energy_minus),
            "electronic_exact_energy_center": float(self.electronic_exact_energy_center),
            "electronic_exact_energy_plus": float(self.electronic_exact_energy_plus),
            "pool_size": int(len(self.pool)),
            "model_status": "center_mo_overlap_aligned_prototype",
        }


_MATH_VIBRONIC_H2 = (
    r"H_{\\mathrm{vib}} = H_{\\mathrm{el}}(R_0) + \\omega (b^\\dagger b + 1/2) + x_{\\mathrm{zpf}} \\frac{dH}{dR}\\big|_{R_0}(b+b^\\dagger)"
)


def _clean_real_polynomial(poly: PauliPolynomial, *, tol: float = 1e-12) -> PauliPolynomial:
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    nq = int(terms[0].nqubit())
    cleaned = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > float(tol):
            raise ValueError(f"Non-negligible imaginary coefficient in polynomial cleanup: {coeff}")
        cleaned.add_term(PauliTerm(nq, ps=str(term.pw2strng()), pc=float(coeff.real)))
    cleaned._reduce()
    return cleaned


def _lift_fermion_polynomial(poly: PauliPolynomial, *, boson_qubits: int, tol: float = 1e-12) -> PauliPolynomial:
    terms = list(poly.return_polynomial())
    if not terms:
        return PauliPolynomial("JW")
    ferm_nq = int(terms[0].nqubit())
    total_nq = int(ferm_nq) + int(boson_qubits)
    lifted = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        label = str(term.pw2strng())
        if len(label) != ferm_nq:
            raise ValueError(f"Unexpected fermionic Pauli label length: {label}")
        lifted.add_term(PauliTerm(total_nq, ps=("e" * int(boson_qubits)) + label, pc=coeff))
    lifted._reduce()
    return lifted


def _boson_vacuum_bitstring(*, qpb: int, boson_encoding: str) -> str:
    encoding = str(boson_encoding).strip().lower()
    if encoding == "binary":
        return "0" * int(qpb)
    if encoding == "unary":
        return ("0" * (int(qpb) - 1)) + "1"
    raise ValueError(f"Unknown boson encoding '{boson_encoding}'")


def build_vibronic_reference_state(
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    n_ph_max: int,
    boson_encoding: str,
    ordering: str = "blocked",
) -> np.ndarray:
    if str(ordering).strip().lower() != "blocked":
        raise ValueError("Vibronic H2 prototype currently supports ordering='blocked' only.")
    n_ferm = 2 * int(n_spatial_orbitals)
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    total_nq = n_ferm + qpb
    fermion_bs = hartree_fock_bitstring(
        int(n_spatial_orbitals),
        tuple(int(x) for x in num_particles),
        indexing="blocked",
    )
    boson_bs = _boson_vacuum_bitstring(qpb=qpb, boson_encoding=str(boson_encoding))
    full_bitstring = boson_bs + fermion_bs
    dim = 1 << int(total_nq)
    psi = np.zeros(dim, dtype=complex)
    psi[int(full_bitstring, 2)] = 1.0 + 0.0j
    return psi


def _dense_matrix_from_polynomial(poly: PauliPolynomial) -> np.ndarray:
    compiled = compile_polynomial_action(poly)
    dim = 1 << int(compiled.nq)
    dense = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[col] = 1.0 + 0.0j
        dense[:, col] = apply_compiled_polynomial(basis, compiled)
    return dense


def exact_ground_energy_dense(poly: PauliPolynomial) -> float:
    dense = _dense_matrix_from_polynomial(poly)
    evals = np.linalg.eigvalsh(dense)
    return float(np.min(np.real(evals)))


def _fermion_sector_bits(*, n_spatial_orbitals: int, num_particles: tuple[int, int]) -> list[int]:
    n_alpha, n_beta = (int(num_particles[0]), int(num_particles[1]))
    n_spatial = int(n_spatial_orbitals)
    out: list[int] = []
    for occ_alpha in itertools.combinations(range(n_spatial), n_alpha):
        for occ_beta in itertools.combinations(range(n_spatial), n_beta):
            bits = 0
            for p in occ_alpha:
                bits |= (1 << int(p))
            for p in occ_beta:
                bits |= (1 << int(n_spatial + p))
            out.append(int(bits))
    return out


def _boson_code_bits(*, n_ph_max: int, boson_encoding: str) -> list[int]:
    d = int(n_ph_max) + 1
    encoding = str(boson_encoding).strip().lower()
    if encoding == "binary":
        return [int(level) for level in range(d)]
    if encoding == "unary":
        return [int(1 << level) for level in range(d)]
    raise ValueError(f"Unknown boson encoding '{boson_encoding}'")


def exact_ground_energy_physical_sector(
    poly: PauliPolynomial,
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    n_ph_max: int,
    boson_encoding: str,
) -> float:
    dense = _dense_matrix_from_polynomial(poly)
    n_fermion_qubits = 2 * int(n_spatial_orbitals)
    fermion_bits = _fermion_sector_bits(
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(x) for x in num_particles),
    )
    boson_bits = _boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    basis_indices = [
        int(f_bits + (b_bits << n_fermion_qubits))
        for b_bits in boson_bits
        for f_bits in fermion_bits
    ]
    sub = dense[np.ix_(basis_indices, basis_indices)]
    evals = np.linalg.eigvalsh(sub)
    return float(np.min(np.real(evals)))


def hf_reference_energy(poly: PauliPolynomial, psi_ref: np.ndarray) -> float:
    energy, _ = energy_via_one_apply(np.asarray(psi_ref, dtype=complex), compile_polynomial_action(poly))
    return float(energy)


def _matrix_to_real_ndarray(obj: Any) -> np.ndarray:
    if hasattr(obj, "np"):
        return np.asarray(obj.np, dtype=float)
    if hasattr(obj, "to_array"):
        return np.asarray(obj.to_array(), dtype=float)
    return np.asarray(obj, dtype=float)


def _align_snapshot_to_center_mo(
    snapshot: RestrictedClosedShellPsi4Snapshot,
    *,
    center_snapshot: RestrictedClosedShellPsi4Snapshot,
) -> RestrictedClosedShellMolecularProblem:
    if int(snapshot.problem.n_spatial_orbitals) != int(center_snapshot.problem.n_spatial_orbitals):
        raise ValueError("Snapshot orbital counts do not match center geometry.")
    try:
        import psi4
    except Exception as exc:  # pragma: no cover
        raise ImportError("Psi4 is required for center-MO overlap alignment.") from exc

    mints = psi4.core.MintsHelper(center_snapshot.basis_set)
    s_cross = _matrix_to_real_ndarray(mints.ao_overlap(center_snapshot.basis_set, snapshot.basis_set))
    overlap_mo = np.asarray(center_snapshot.coeff_alpha_mo, dtype=float).T @ s_cross @ np.asarray(snapshot.coeff_alpha_mo, dtype=float)
    u, _singular_values, vh = np.linalg.svd(overlap_mo, full_matrices=False)
    rotation = np.asarray(vh.T @ u.T, dtype=float)

    h_old = np.asarray(snapshot.problem.one_body_integrals_mo, dtype=float)
    eri_old = np.asarray(snapshot.problem.two_body_integrals_mo, dtype=float)
    h_aligned = rotation.T @ h_old @ rotation
    eri_aligned = np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation, rotation, rotation, rotation, eri_old, optimize=True)

    return RestrictedClosedShellMolecularProblem(
        geometry_spec=str(snapshot.problem.geometry_spec),
        basis=str(snapshot.problem.basis),
        charge=int(snapshot.problem.charge),
        multiplicity=int(snapshot.problem.multiplicity),
        reference=str(snapshot.problem.reference),
        n_spatial_orbitals=int(snapshot.problem.n_spatial_orbitals),
        n_alpha=int(snapshot.problem.n_alpha),
        n_beta=int(snapshot.problem.n_beta),
        hf_energy=float(snapshot.problem.hf_energy),
        nuclear_repulsion_energy=float(snapshot.problem.nuclear_repulsion_energy),
        one_body_integrals_mo=np.asarray(h_aligned, dtype=float),
        two_body_integrals_mo=np.asarray(eri_aligned, dtype=float),
    )


def _boson_momentum_operator(*, nq_total: int, boson_qubits: Sequence[int], n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    b_op = boson_operator(
        "JW",
        int(nq_total),
        boson_qubits,
        which="b",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )
    bdag_op = boson_operator(
        "JW",
        int(nq_total),
        boson_qubits,
        which="bdag",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )
    return _clean_real_polynomial((1j) * (bdag_op - b_op))


def _fermion_number_operator(*, nq_fermion: int, orbital: int) -> PauliPolynomial:
    create = fermion_plus_operator("JW", int(nq_fermion), int(orbital))
    destroy = fermion_minus_operator("JW", int(nq_fermion), int(orbital))
    return _clean_real_polynomial(create * destroy)


_MATH_FERMION_MIXING = r"K = \sum_{\sigma\in\{\alpha,\beta\}} (a^\dagger_{b\sigma} a_{a\sigma} + a^\dagger_{a\sigma} a_{b\sigma})"


def _fermion_orbital_mixing_operator(*, nq_fermion: int, left_orbital: int, right_orbital: int) -> PauliPolynomial:
    create_left = fermion_plus_operator("JW", int(nq_fermion), int(left_orbital))
    destroy_left = fermion_minus_operator("JW", int(nq_fermion), int(left_orbital))
    create_right = fermion_plus_operator("JW", int(nq_fermion), int(right_orbital))
    destroy_right = fermion_minus_operator("JW", int(nq_fermion), int(right_orbital))
    return _clean_real_polynomial((create_left * destroy_right) + (create_right * destroy_left))


_MATH_FERMION_PAIR = r"P = a^\dagger_{a\alpha} a^\dagger_{a\beta} a_{b\beta} a_{b\alpha} + \mathrm{h.c.}"


def _fermion_pair_exchange_operator(*, nq_fermion: int, bond_alpha: int, anti_alpha: int, bond_beta: int, anti_beta: int) -> PauliPolynomial:
    forward = (
        fermion_plus_operator("JW", int(nq_fermion), int(anti_alpha))
        * fermion_plus_operator("JW", int(nq_fermion), int(anti_beta))
        * fermion_minus_operator("JW", int(nq_fermion), int(bond_beta))
        * fermion_minus_operator("JW", int(nq_fermion), int(bond_alpha))
    )
    backward = (
        fermion_plus_operator("JW", int(nq_fermion), int(bond_alpha))
        * fermion_plus_operator("JW", int(nq_fermion), int(bond_beta))
        * fermion_minus_operator("JW", int(nq_fermion), int(anti_beta))
        * fermion_minus_operator("JW", int(nq_fermion), int(anti_alpha))
    )
    return _clean_real_polynomial(forward + backward)


_MATH_FERMION_DENSITY = r"N_S = \prod_{p\in S} n_p"


def _fermion_density_product_operator(*, nq_fermion: int, orbitals: Sequence[int]) -> PauliPolynomial:
    selected = tuple(int(x) for x in orbitals)
    if not selected:
        raise ValueError("orbitals must be non-empty for density-product operator.")
    product = _fermion_number_operator(nq_fermion=int(nq_fermion), orbital=selected[0])
    for orbital in selected[1:]:
        product = _clean_real_polynomial(product * _fermion_number_operator(nq_fermion=int(nq_fermion), orbital=int(orbital)))
    return _clean_real_polynomial(product)


def build_vibronic_h2_model(
    *,
    bond_length_angstrom: float = 0.7414,
    bond_step_angstrom: float = 0.01,
    basis: str = "sto-3g",
    n_ph_max: int = 3,
    boson_encoding: str = "binary",
    coupling_scale: float = 1.0,
    ordering: str = "blocked",
) -> VibronicH2Model:
    if str(ordering).strip().lower() != "blocked":
        raise ValueError("Vibronic H2 prototype currently supports ordering='blocked' only.")
    if float(bond_length_angstrom) <= 0.0:
        raise ValueError("bond_length_angstrom must be > 0.")
    if float(bond_step_angstrom) <= 0.0:
        raise ValueError("bond_step_angstrom must be > 0.")
    if int(n_ph_max) < 1:
        raise ValueError("n_ph_max must be >= 1.")

    r0 = float(bond_length_angstrom)
    dr_ang = float(bond_step_angstrom)
    dr_bohr = float(dr_ang) * ANGSTROM_TO_BOHR

    snapshot_minus = build_h2_snapshot_from_psi4(bond_length_angstrom=r0 - dr_ang, basis=str(basis))
    snapshot_center = build_h2_snapshot_from_psi4(bond_length_angstrom=r0, basis=str(basis))
    snapshot_plus = build_h2_snapshot_from_psi4(bond_length_angstrom=r0 + dr_ang, basis=str(basis))

    problem_center = snapshot_center.problem
    problem_minus = _align_snapshot_to_center_mo(snapshot_minus, center_snapshot=snapshot_center)
    problem_plus = _align_snapshot_to_center_mo(snapshot_plus, center_snapshot=snapshot_center)

    h_minus = build_restricted_closed_shell_molecular_hamiltonian(problem_minus, ordering="blocked")
    h_center = build_restricted_closed_shell_molecular_hamiltonian(problem_center, ordering="blocked")
    h_plus = build_restricted_closed_shell_molecular_hamiltonian(problem_plus, ordering="blocked")

    e_minus = exact_ground_energy_physical_sector(
        h_minus,
        n_spatial_orbitals=int(problem_minus.n_spatial_orbitals),
        num_particles=tuple(problem_minus.num_particles),
        n_ph_max=0,
        boson_encoding="binary",
    )
    e_center = exact_ground_energy_physical_sector(
        h_center,
        n_spatial_orbitals=int(problem_center.n_spatial_orbitals),
        num_particles=tuple(problem_center.num_particles),
        n_ph_max=0,
        boson_encoding="binary",
    )
    e_plus = exact_ground_energy_physical_sector(
        h_plus,
        n_spatial_orbitals=int(problem_plus.n_spatial_orbitals),
        num_particles=tuple(problem_plus.num_particles),
        n_ph_max=0,
        boson_encoding="binary",
    )
    curvature = float((e_plus - 2.0 * e_center + e_minus) / (dr_bohr * dr_bohr))
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError(f"Non-positive local curvature for H2 stretch mode: {curvature}")

    reduced_mass = 0.5 * PROTON_MASS_ELECTRON
    omega_au = float(math.sqrt(curvature / reduced_mass))
    if omega_au <= 0.0 or not np.isfinite(omega_au):
        raise ValueError(f"Invalid stretch frequency: {omega_au}")
    x_zpf_bohr = float(math.sqrt(1.0 / (2.0 * reduced_mass * omega_au)))

    dH_dR = _clean_real_polynomial((1.0 / (2.0 * dr_bohr)) * (h_plus - h_minus))

    n_fermion_qubits = int(problem_center.n_spin_orbitals)
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    n_total_qubits = n_fermion_qubits + qpb
    boson_qubit_block = phonon_qubit_indices_for_site(
        0,
        n_sites=1,
        qpb=qpb,
        fermion_qubits=n_fermion_qubits,
    )

    h_center_lifted = _lift_fermion_polynomial(h_center, boson_qubits=qpb)
    dH_dR_lifted = _lift_fermion_polynomial(dH_dR, boson_qubits=qpb)
    n_b = _clean_real_polynomial(
        boson_operator(
            "JW",
            int(n_total_qubits),
            boson_qubit_block,
            which="n",
            n_ph_max=int(n_ph_max),
            encoding=str(boson_encoding),
        )
    )
    x_b = _clean_real_polynomial(
        boson_operator(
            "JW",
            int(n_total_qubits),
            boson_qubit_block,
            which="x",
            n_ph_max=int(n_ph_max),
            encoding=str(boson_encoding),
        )
    )
    p_b = _boson_momentum_operator(
        nq_total=int(n_total_qubits),
        boson_qubits=boson_qubit_block,
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )

    h_vibronic = _clean_real_polynomial(
        h_center_lifted
        + (float(omega_au) * n_b)
        + float(0.5 * omega_au)
        + (float(coupling_scale) * float(x_zpf_bohr)) * (dH_dR_lifted * x_b)
    )

    psi_ref = build_vibronic_reference_state(
        n_spatial_orbitals=int(problem_center.n_spatial_orbitals),
        num_particles=tuple(problem_center.num_particles),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering="blocked",
    )

    electronic_pool = build_molecular_uccsd_pool(
        n_spatial_orbitals=int(problem_center.n_spatial_orbitals),
        num_particles=tuple(problem_center.num_particles),
        ordering="blocked",
    )
    lifted_electronic_pool: list[AnsatzTerm] = []
    for term in electronic_pool:
        lifted_poly = _clean_real_polynomial(_lift_fermion_polynomial(term.polynomial, boson_qubits=qpb))
        if lifted_poly.count_number_terms() == 0:
            continue
        lifted_electronic_pool.append(AnsatzTerm(label=f"el::{term.label}", polynomial=lifted_poly))

    pool: list[AnsatzTerm] = list(lifted_electronic_pool)
    pool.append(AnsatzTerm(label="boson::p", polynomial=p_b))
    coupled_dhdr_p = _clean_real_polynomial(dH_dR_lifted * p_b)
    if coupled_dhdr_p.count_number_terms() > 0:
        pool.append(AnsatzTerm(label="coupled::dH_dR_times_p", polynomial=coupled_dhdr_p))

    fermion_number_ops_lifted = [
        _clean_real_polynomial(
            _lift_fermion_polynomial(
                _fermion_number_operator(nq_fermion=int(n_fermion_qubits), orbital=int(j)),
                boson_qubits=qpb,
            )
        )
        for j in range(int(n_fermion_qubits))
    ]
    bond_occ = _clean_real_polynomial(fermion_number_ops_lifted[0] + fermion_number_ops_lifted[2])
    anti_occ = _clean_real_polynomial(fermion_number_ops_lifted[1] + fermion_number_ops_lifted[3])
    orbital_imbalance = _clean_real_polynomial(anti_occ - bond_occ)
    for label, el_channel in (
        ("bond_occ", bond_occ),
        ("anti_occ", anti_occ),
        ("orbital_imbalance", orbital_imbalance),
    ):
        coupled_occ = _clean_real_polynomial(el_channel * p_b)
        if coupled_occ.count_number_terms() == 0:
            continue
        pool.append(AnsatzTerm(label=f"coupled::occ::{label}::p", polynomial=coupled_occ))

    factored_channels = (
        (
            "mix_x",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _clean_real_polynomial(
                        _fermion_orbital_mixing_operator(nq_fermion=int(n_fermion_qubits), left_orbital=0, right_orbital=1)
                        + _fermion_orbital_mixing_operator(nq_fermion=int(n_fermion_qubits), left_orbital=2, right_orbital=3)
                    ),
                    boson_qubits=qpb,
                )
            ),
        ),
        (
            "pair_x",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_pair_exchange_operator(
                        nq_fermion=int(n_fermion_qubits),
                        bond_alpha=0,
                        anti_alpha=1,
                        bond_beta=2,
                        anti_beta=3,
                    ),
                    boson_qubits=qpb,
                )
            ),
        ),
        (
            "n_bond_pair",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_density_product_operator(nq_fermion=int(n_fermion_qubits), orbitals=(0, 2)),
                    boson_qubits=qpb,
                )
            ),
        ),
        (
            "n_anti_pair",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_density_product_operator(nq_fermion=int(n_fermion_qubits), orbitals=(1, 3)),
                    boson_qubits=qpb,
                )
            ),
        ),
        (
            "cross_pair",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _clean_real_polynomial(
                        _fermion_density_product_operator(nq_fermion=int(n_fermion_qubits), orbitals=(0, 3))
                        + _fermion_density_product_operator(nq_fermion=int(n_fermion_qubits), orbitals=(1, 2))
                    ),
                    boson_qubits=qpb,
                )
            ),
        ),
    )
    for label, el_channel in factored_channels:
        coupled_factored = _clean_real_polynomial(el_channel * p_b)
        if coupled_factored.count_number_terms() == 0:
            continue
        pool.append(AnsatzTerm(label=f"coupled::factored::{label}::p", polynomial=coupled_factored))

    return VibronicH2Model(
        bond_length_angstrom=float(r0),
        bond_step_angstrom=float(dr_ang),
        basis=str(basis),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        n_fermion_qubits=int(n_fermion_qubits),
        n_boson_qubits=int(qpb),
        n_total_qubits=int(n_total_qubits),
        omega_au=float(omega_au),
        reduced_mass_au=float(reduced_mass),
        x_zpf_bohr=float(x_zpf_bohr),
        curvature_au_per_bohr2=float(curvature),
        electronic_exact_energy_minus=float(e_minus),
        electronic_exact_energy_center=float(e_center),
        electronic_exact_energy_plus=float(e_plus),
        h_electronic=h_center,
        dH_dR=dH_dR,
        h_vibronic=h_vibronic,
        pool=tuple(pool),
        psi_ref=np.asarray(psi_ref, dtype=complex),
    )
