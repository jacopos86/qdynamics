from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
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
    coupling_scale: float = 1.0

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
            "coupling_scale": float(self.coupling_scale),
            "curvature_au_per_bohr2": float(self.curvature_au_per_bohr2),
            "electronic_exact_energy_minus": float(self.electronic_exact_energy_minus),
            "electronic_exact_energy_center": float(self.electronic_exact_energy_center),
            "electronic_exact_energy_plus": float(self.electronic_exact_energy_plus),
            "pool_size": int(len(self.pool)),
            "model_status": "center_mo_overlap_aligned_prototype",
        }


@dataclass(frozen=True)
class CachedVibronicH2Fixture:
    model: VibronicH2Model
    exact_ground_energy: float | None
    fixture_path: Path
    metadata: dict[str, Any]


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
    compiled = compile_polynomial_action(poly)
    full_dim = 1 << int(compiled.nq)
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
    if not basis_indices:
        raise ValueError("Physical-sector basis is empty.")
    if max(basis_indices) >= int(full_dim):
        raise ValueError(
            f"Physical-sector basis index exceeds Hamiltonian dimension: max={max(basis_indices)}, dim={full_dim}."
        )
    sub = np.zeros((len(basis_indices), len(basis_indices)), dtype=complex)
    row_indices = np.asarray(basis_indices, dtype=int)
    for col_pos, basis_index in enumerate(basis_indices):
        basis = np.zeros(full_dim, dtype=complex)
        basis[int(basis_index)] = 1.0 + 0.0j
        sub[:, col_pos] = apply_compiled_polynomial(basis, compiled)[row_indices]
    evals = np.linalg.eigvalsh(sub)
    return float(np.min(np.real(evals)))


def hf_reference_energy(poly: PauliPolynomial, psi_ref: np.ndarray) -> float:
    energy, _ = energy_via_one_apply(np.asarray(psi_ref, dtype=complex), compile_polynomial_action(poly))
    return float(energy)


def _polynomial_to_jsonable(poly: PauliPolynomial) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    nq: int | None = None
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if nq is None:
            nq = int(term.nqubit())
        terms.append(
            {
                "pauli": label,
                "coeff": {"re": float(coeff.real), "im": float(coeff.imag)},
            }
        )
    return {"repr": "JW", "n_qubits": int(nq or 0), "terms": terms}


def _polynomial_from_jsonable(payload: Any, *, expected_nq: int | None = None) -> PauliPolynomial:
    if not isinstance(payload, dict):
        raise ValueError("Serialized PauliPolynomial payload must be an object.")
    raw_terms = payload.get("terms")
    if not isinstance(raw_terms, list):
        raise ValueError("Serialized PauliPolynomial payload missing list key 'terms'.")
    nq = int(payload.get("n_qubits", expected_nq or 0))
    if expected_nq is not None and int(nq) != int(expected_nq):
        raise ValueError(f"Serialized polynomial n_qubits={nq} does not match expected {expected_nq}.")
    poly = PauliPolynomial(str(payload.get("repr", "JW")))
    for row in raw_terms:
        if not isinstance(row, dict):
            raise ValueError("Serialized Pauli term must be an object.")
        label = str(row.get("pauli", ""))
        if not label:
            raise ValueError("Serialized Pauli term missing label.")
        if int(nq) == 0:
            nq = len(label)
        if len(label) != int(nq):
            raise ValueError(f"Pauli label length {len(label)} does not match n_qubits={nq}: {label!r}")
        coeff_payload = row.get("coeff", {})
        if isinstance(coeff_payload, dict):
            coeff = complex(float(coeff_payload.get("re", 0.0)), float(coeff_payload.get("im", 0.0)))
        else:
            coeff = complex(coeff_payload)
        if abs(coeff) <= 0.0:
            continue
        poly.add_term(PauliTerm(int(nq), ps=label, pc=coeff))
    poly._reduce()
    return poly


def _state_to_sparse_jsonable(psi: np.ndarray, *, tol: float = 1e-14) -> dict[str, Any]:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    if arr.size <= 0 or arr.size & (arr.size - 1):
        raise ValueError("State vector length must be a positive power of two.")
    nq = int(round(math.log2(arr.size)))
    amps: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(arr):
        val = complex(amp)
        if abs(val) <= float(tol):
            continue
        amps[format(int(idx), f"0{nq}b")] = {"re": float(val.real), "im": float(val.imag)}
    return {"nq_total": int(nq), "amplitudes_qn_to_q0": amps}


def _state_from_sparse_jsonable(payload: Any, *, expected_nq: int | None = None) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError("Serialized state payload must be an object.")
    nq = int(payload.get("nq_total", expected_nq or 0))
    if expected_nq is not None and int(nq) != int(expected_nq):
        raise ValueError(f"Serialized state nq_total={nq} does not match expected {expected_nq}.")
    amps = payload.get("amplitudes_qn_to_q0")
    if not isinstance(amps, dict) or not amps:
        raise ValueError("Serialized state missing amplitudes_qn_to_q0.")
    psi = np.zeros(1 << int(nq), dtype=complex)
    for bitstr, coeff_payload in amps.items():
        if not isinstance(bitstr, str) or len(bitstr) != int(nq) or any(ch not in "01" for ch in bitstr):
            raise ValueError(f"Invalid serialized state bitstring: {bitstr!r}")
        if not isinstance(coeff_payload, dict):
            raise ValueError(f"Amplitude payload for {bitstr!r} must be an object.")
        psi[int(bitstr, 2)] = complex(float(coeff_payload.get("re", 0.0)), float(coeff_payload.get("im", 0.0)))
    return np.asarray(psi, dtype=complex).reshape(-1)


def vibronic_h2_fixture_to_jsonable(
    model: VibronicH2Model,
    *,
    exact_ground_energy: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a fixed H2 vibronic model for no-Psi4 static ADAPT tests/runs."""
    exact_payload = None if exact_ground_energy is None else float(exact_ground_energy)
    return {
        "schema": "molecular_vibronic_h2_fixture_v1",
        "family_key": "molecular_vibronic_h2",
        "model": model.to_jsonable(),
        "polynomials": {
            "h_electronic": _polynomial_to_jsonable(model.h_electronic),
            "dH_dR": _polynomial_to_jsonable(model.dH_dR),
            "h_vibronic": _polynomial_to_jsonable(model.h_vibronic),
        },
        "pool": [
            {
                "label": str(term.label),
                "polynomial": _polynomial_to_jsonable(term.polynomial),
            }
            for term in model.pool
        ],
        "reference_state": {
            "kind": "restricted_hf_times_boson_vacuum",
            **_state_to_sparse_jsonable(model.psi_ref),
        },
        "exact": {
            "ground_energy_physical_sector": exact_payload,
            "method": "dense_physical_sector",
        },
        "provenance": dict(provenance or {}),
    }


def default_vibronic_h2_fixture_path() -> Path:
    return Path(__file__).resolve().parents[3] / "test_support" / "molecular_vibronic_h2_sto3g_fd001.json"


def _legacy_vibronic_h2_fixture_path() -> Path:
    return Path(__file__).resolve().parents[3] / "test_support" / "molecular_vibronic_h2_sto3g_nph1_binary.json"


def load_cached_vibronic_h2_fixture(path: str | Path | None = None) -> CachedVibronicH2Fixture:
    fixture_path = default_vibronic_h2_fixture_path() if path in {None, ""} else Path(path)
    if path in {None, ""} and not Path(fixture_path).exists():
        fixture_path = _legacy_vibronic_h2_fixture_path()
    raw = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Vibronic H2 fixture must be a JSON object.")
    if raw.get("schema") != "molecular_vibronic_h2_fixture_v1":
        raise ValueError(f"Unsupported vibronic H2 fixture schema: {raw.get('schema')!r}")
    if raw.get("family_key") != "molecular_vibronic_h2":
        raise ValueError(f"Unexpected vibronic H2 fixture family_key: {raw.get('family_key')!r}")
    model_payload = raw.get("model")
    polynomials = raw.get("polynomials")
    if not isinstance(model_payload, dict) or not isinstance(polynomials, dict):
        raise ValueError("Vibronic H2 fixture missing model/polynomials payloads.")
    n_total = int(model_payload["n_total_qubits"])
    n_fermion = int(model_payload["n_fermion_qubits"])
    n_boson = int(model_payload["n_boson_qubits"])
    if int(n_total) != 5 or int(n_fermion) != 4 or int(n_boson) != 1:
        raise ValueError("Vibronic H2 canonical fixture must have 4 fermion qubits, 1 boson qubit, 5 total qubits.")
    if n_total != n_fermion + n_boson:
        raise ValueError("Vibronic H2 fixture qubit counts are inconsistent.")
    if int(model_payload["n_ph_max"]) != 1 or str(model_payload["boson_encoding"]) != "binary":
        raise ValueError("Vibronic H2 canonical fixture must be n_ph_max=1, binary encoding.")
    for scalar_key in (
        "bond_length_angstrom",
        "bond_step_angstrom",
        "omega_au",
        "reduced_mass_au",
        "x_zpf_bohr",
        "curvature_au_per_bohr2",
        "electronic_exact_energy_minus",
        "electronic_exact_energy_center",
        "electronic_exact_energy_plus",
    ):
        if not np.isfinite(float(model_payload[scalar_key])):
            raise ValueError(f"Vibronic H2 fixture scalar is non-finite: {scalar_key}")
    h_electronic = _polynomial_from_jsonable(polynomials.get("h_electronic"), expected_nq=int(n_fermion))
    dH_dR = _polynomial_from_jsonable(polynomials.get("dH_dR"), expected_nq=int(n_fermion))
    h_vibronic = _polynomial_from_jsonable(polynomials.get("h_vibronic"), expected_nq=int(n_total))
    pool_payload = raw.get("pool")
    if not isinstance(pool_payload, list) or not pool_payload:
        raise ValueError("Vibronic H2 fixture must contain a non-empty pool.")
    pool: list[AnsatzTerm] = []
    for row in pool_payload:
        if not isinstance(row, dict):
            raise ValueError("Vibronic H2 pool row must be an object.")
        label = str(row.get("label", ""))
        if not label:
            raise ValueError("Vibronic H2 fixture contains an empty pool label.")
        pool.append(
            AnsatzTerm(
                label=label,
                polynomial=_polynomial_from_jsonable(row.get("polynomial"), expected_nq=int(n_total)),
            )
        )
    psi_ref = _state_from_sparse_jsonable(raw.get("reference_state"), expected_nq=int(n_total))
    norm = float(np.linalg.norm(psi_ref))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Vibronic H2 fixture reference state has invalid norm.")
    psi_ref = psi_ref / norm
    exact_payload = raw.get("exact", {})
    exact_energy = None
    if isinstance(exact_payload, dict) and exact_payload.get("ground_energy_physical_sector") is not None:
        exact_energy = float(exact_payload["ground_energy_physical_sector"])
        if not np.isfinite(float(exact_energy)):
            raise ValueError("Vibronic H2 fixture exact energy is non-finite.")
    if exact_energy is None:
        exact_energy = exact_ground_energy_physical_sector(
            h_vibronic,
            n_spatial_orbitals=2,
            num_particles=(1, 1),
            n_ph_max=1,
            boson_encoding="binary",
        )
    model = VibronicH2Model(
        bond_length_angstrom=float(model_payload["bond_length_angstrom"]),
        bond_step_angstrom=float(model_payload["bond_step_angstrom"]),
        basis=str(model_payload["basis"]),
        n_ph_max=int(model_payload["n_ph_max"]),
        boson_encoding=str(model_payload["boson_encoding"]),
        n_fermion_qubits=int(n_fermion),
        n_boson_qubits=int(n_boson),
        n_total_qubits=int(n_total),
        omega_au=float(model_payload["omega_au"]),
        reduced_mass_au=float(model_payload["reduced_mass_au"]),
        x_zpf_bohr=float(model_payload["x_zpf_bohr"]),
        curvature_au_per_bohr2=float(model_payload["curvature_au_per_bohr2"]),
        electronic_exact_energy_minus=float(model_payload["electronic_exact_energy_minus"]),
        electronic_exact_energy_center=float(model_payload["electronic_exact_energy_center"]),
        electronic_exact_energy_plus=float(model_payload["electronic_exact_energy_plus"]),
        h_electronic=h_electronic,
        dH_dR=dH_dR,
        h_vibronic=h_vibronic,
        pool=tuple(pool),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        coupling_scale=float(model_payload.get("coupling_scale", 1.0)),
    )
    return CachedVibronicH2Fixture(
        model=model,
        exact_ground_energy=float(exact_energy),
        fixture_path=Path(fixture_path),
        metadata={
            "schema": str(raw.get("schema")),
            "family_key": str(raw.get("family_key")),
            "model": dict(model_payload),
            "provenance": dict(raw.get("provenance", {})) if isinstance(raw.get("provenance", {}), dict) else {},
        },
    )


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


def _build_vibronic_h2_model_from_cached_components(
    *,
    h_electronic: PauliPolynomial,
    dH_dR: PauliPolynomial,
    bond_length_angstrom: float,
    bond_step_angstrom: float,
    basis: str,
    omega_au: float,
    reduced_mass_au: float,
    x_zpf_bohr: float,
    curvature_au_per_bohr2: float,
    electronic_exact_energy_minus: float,
    electronic_exact_energy_center: float,
    electronic_exact_energy_plus: float,
    n_ph_max: int,
    boson_encoding: str,
    coupling_scale: float,
    ordering: str = "blocked",
) -> VibronicH2Model:
    """Rebuild a vibronic H2 model from cached electronic-surface data.

    The checked-in fixture stores the center-geometry electronic Hamiltonian and
    finite-difference derivative.  Runtime reconstruction only changes the
    truncated oscillator register and coupling multiplier; it never calls Psi4.
    """

    if str(ordering).strip().lower() != "blocked":
        raise ValueError("Vibronic H2 prototype currently supports ordering='blocked' only.")
    if int(n_ph_max) < 1:
        raise ValueError("n_ph_max must be >= 1.")
    if str(boson_encoding).strip().lower() != "binary":
        raise ValueError("molecular_vibronic_h2 currently supports boson_encoding='binary' only.")
    if not np.isfinite(float(coupling_scale)):
        raise ValueError(f"Invalid molecular_vibronic_h2 coupling_scale: {coupling_scale!r}")

    n_fermion_qubits = 4
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    n_total_qubits = int(n_fermion_qubits) + int(qpb)
    boson_qubit_block = phonon_qubit_indices_for_site(
        0,
        n_sites=1,
        qpb=int(qpb),
        fermion_qubits=int(n_fermion_qubits),
    )

    h_center_lifted = _lift_fermion_polynomial(h_electronic, boson_qubits=int(qpb))
    dH_dR_lifted = _lift_fermion_polynomial(dH_dR, boson_qubits=int(qpb))
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
        n_spatial_orbitals=2,
        num_particles=(1, 1),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering="blocked",
    )

    electronic_pool = build_molecular_uccsd_pool(
        n_spatial_orbitals=2,
        num_particles=(1, 1),
        ordering="blocked",
    )
    lifted_electronic_pool: list[AnsatzTerm] = []
    for term in electronic_pool:
        lifted_poly = _clean_real_polynomial(_lift_fermion_polynomial(term.polynomial, boson_qubits=int(qpb)))
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
                boson_qubits=int(qpb),
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
                    boson_qubits=int(qpb),
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
                    boson_qubits=int(qpb),
                )
            ),
        ),
        (
            "n_bond_pair",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_density_product_operator(nq_fermion=int(n_fermion_qubits), orbitals=(0, 2)),
                    boson_qubits=int(qpb),
                )
            ),
        ),
        (
            "n_anti_pair",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_density_product_operator(nq_fermion=int(n_fermion_qubits), orbitals=(1, 3)),
                    boson_qubits=int(qpb),
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
                    boson_qubits=int(qpb),
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
        bond_length_angstrom=float(bond_length_angstrom),
        bond_step_angstrom=float(bond_step_angstrom),
        basis=str(basis),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        n_fermion_qubits=int(n_fermion_qubits),
        n_boson_qubits=int(qpb),
        n_total_qubits=int(n_total_qubits),
        omega_au=float(omega_au),
        reduced_mass_au=float(reduced_mass_au),
        x_zpf_bohr=float(x_zpf_bohr),
        curvature_au_per_bohr2=float(curvature_au_per_bohr2),
        electronic_exact_energy_minus=float(electronic_exact_energy_minus),
        electronic_exact_energy_center=float(electronic_exact_energy_center),
        electronic_exact_energy_plus=float(electronic_exact_energy_plus),
        h_electronic=h_electronic,
        dH_dR=dH_dR,
        h_vibronic=h_vibronic,
        pool=tuple(pool),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        coupling_scale=float(coupling_scale),
    )


def build_cached_vibronic_h2_model(
    *,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    coupling_scale: float = 1.0,
    ordering: str = "blocked",
    fixture_path: str | Path | None = None,
) -> VibronicH2Model:
    fixture = load_cached_vibronic_h2_fixture(fixture_path)
    base = fixture.model
    if (
        int(n_ph_max) == int(base.n_ph_max)
        and str(boson_encoding) == str(base.boson_encoding)
        and math.isclose(float(coupling_scale), float(base.coupling_scale), rel_tol=0.0, abs_tol=1e-15)
        and str(ordering).strip().lower() == "blocked"
    ):
        return base
    return _build_vibronic_h2_model_from_cached_components(
        h_electronic=base.h_electronic,
        dH_dR=base.dH_dR,
        bond_length_angstrom=float(base.bond_length_angstrom),
        bond_step_angstrom=float(base.bond_step_angstrom),
        basis=str(base.basis),
        omega_au=float(base.omega_au),
        reduced_mass_au=float(base.reduced_mass_au),
        x_zpf_bohr=float(base.x_zpf_bohr),
        curvature_au_per_bohr2=float(base.curvature_au_per_bohr2),
        electronic_exact_energy_minus=float(base.electronic_exact_energy_minus),
        electronic_exact_energy_center=float(base.electronic_exact_energy_center),
        electronic_exact_energy_plus=float(base.electronic_exact_energy_plus),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        coupling_scale=float(coupling_scale),
        ordering=str(ordering),
    )


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
        coupling_scale=float(coupling_scale),
    )
