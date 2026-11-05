"""Interoperability with Doron Wu's qubitized molecular Hamiltonians (qdynamics, branch doron-chemistry).

His ``src/chemistry_qubitization`` package (github.com/jacopos86/qdynamics at 38f40805,
2026-10-27) maps PySCF or Psi4 matrix elements to a Pauli Hamiltonian and writes
``qdynamics_qubit_hamiltonian_v1`` JSON: Hartree units, blocked spin order (all alpha
orbitals, then all beta), Pauli words over ``e/x/y/z`` in ``q_(n-1)...q_0`` display order,
binary or unary bosons with cutoff N keeping levels 0..N on ceil(log2(N+1)) (binary) or N+1
(unary) qubits above the fermion register, mode 0 first, and the model

    H = E_nuc + sum_pq h_pq a+_p a_q + 1/2 sum (pq|rs) a+ a+ a a
        + sum_I omega_I (n_I + 1/2) + sum_I g_I,pq a+_p a_q (b_I + b_I+),

with g_I = <p| dV_SCF/dQ_I |q> / sqrt(2 omega_I) from the PySCF ``eph`` module: the
mean-field vertex in the full MO space of a closed-shell RHF, no frozen core, no
reference-force counter-term. Those conventions coincide with this repository's
(``molecular_vibronic.py``, ``vibronic_h2.py``): same spin order, same word order, same
boson registers, Q_I = (b + b+)/sqrt(2 omega_I), so his g_I (b + b+) is our C_I Q_I with
C_I = sqrt(2 omega_I) g_I, the coupling the builder calls ``mean_field_pyscf_eph``
assembled without the counter-term.

This module loads his files into our ``PauliPolynomial`` and register layout (so the exact
sector solver, RA-ADAPT, AP-McLachlan and QSE can consume them), writes our models in his
schema, evaluates the Hartree-Fock (x) vacuum energy and the exact physical-sector
spectrum of either, and compares two Hamiltonians term by term. Nothing here imports or
runs his code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.molecular_vibronic import (
    MolecularGeometry,
    NormalMode,
    VibronicQubitModel,
    build_molecular_vibronic_model,
    build_register_layout,
    physical_sector_basis,
    sector_sparse_matrix,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import RegisterLayout
from src.quantum.hartree_fock_reference_state import hartree_fock_occupied_qubits
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

QDYNAMICS_SCHEMA = "qdynamics_qubit_hamiltonian_v1"
QDYNAMICS_CONVENTIONS = {"energy_units": "hartree", "spin_ordering": "blocked", "pauli_ordering": "q_(n-1)...q_0"}
# His name for the one coupling model his assembler implements; ours for the same physics.
QDYNAMICS_COUPLING_MODEL = "mean_field_potential_eph"
OUR_EQUIVALENT_COUPLING_MODEL = "mean_field_pyscf_eph"
QDYNAMICS_SOURCE = "github.com/jacopos86/qdynamics branch doron-chemistry, src/chemistry_qubitization at 38f40805 (2026-10-27)"


@dataclass(frozen=True)
class QdynamicsQubitHamiltonian:
    """A ``qdynamics_qubit_hamiltonian_v1`` file in this repository's objects."""

    operator: PauliPolynomial
    n_qubits: int
    n_fermion_qubits: int
    num_particles: tuple[int, int]
    mode_cutoffs: tuple[int, ...]
    encoding: str | None
    frequencies_hartree: tuple[float, ...]
    coupling_model: str | None
    source_files: tuple[str, ...]
    metadata: Mapping[str, Any]
    layout: RegisterLayout
    path: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_spatial_orbitals(self) -> int:
        return int(self.n_fermion_qubits) // 2


def _layout_for(n_fermion_qubits: int, cutoffs: Sequence[int], frequencies: Sequence[float], encoding: str | None) -> RegisterLayout:
    if int(n_fermion_qubits) <= 0 or int(n_fermion_qubits) % 2:
        raise ValueError(f"n_fermion_qubits must be a positive even number (blocked spin order); got {n_fermion_qubits}.")
    n_spatial = int(n_fermion_qubits) // 2
    cutoffs = tuple(int(v) for v in cutoffs)
    if cutoffs and encoding not in ("binary", "unary"):
        raise ValueError(f"boson_encoding must be 'binary' or 'unary' when modes are present; got {encoding!r}.")
    if len(frequencies) != len(cutoffs):
        raise ValueError(f"{len(frequencies)} frequencies for {len(cutoffs)} mode cutoffs.")
    modes = tuple(NormalMode(i, float(w), np.zeros((1, 3)), f"mode_{i}") for i, w in enumerate(frequencies))
    return build_register_layout(n_spatial, modes, cutoffs, encoding or "binary")


def load_qdynamics_qubit_hamiltonian(path: str | Path) -> QdynamicsQubitHamiltonian:
    """Read his JSON with the checks his own loader makes, into our operator and layout."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return qdynamics_qubit_hamiltonian_from_payload(payload, path=str(path))


def qdynamics_qubit_hamiltonian_from_payload(payload: Mapping[str, Any], *, path: str | None = None) -> QdynamicsQubitHamiltonian:
    if payload.get("schema") != QDYNAMICS_SCHEMA:
        raise ValueError(f"unsupported schema {payload.get('schema')!r}; expected {QDYNAMICS_SCHEMA!r}.")
    for key, expected in QDYNAMICS_CONVENTIONS.items():
        if payload.get(key) != expected:
            raise ValueError(f"unsupported convention {key}={payload.get(key)!r}; this reader takes {expected!r} only.")
    nq = int(payload["n_qubits"])
    n_fermion = int(payload["n_fermion_qubits"])
    particles = tuple(int(v) for v in payload["num_particles"])
    if len(particles) != 2 or any(v < 0 or v > n_fermion // 2 for v in particles):
        raise ValueError(f"num_particles {particles} does not fit {n_fermion // 2} spatial orbitals.")
    cutoffs = tuple(int(v) for v in payload.get("mode_cutoffs") or ())
    frequencies = tuple(float(v) for v in payload.get("omega_hartree") or ())
    encoding = payload.get("boson_encoding")
    layout = _layout_for(n_fermion, cutoffs, frequencies, encoding)
    if int(layout.n_total_qubits) != nq:
        raise ValueError(f"n_qubits {nq} disagrees with {n_fermion} fermion qubits plus boson registers for cutoffs {cutoffs} ({encoding}): {layout.n_total_qubits}.")
    terms = []
    for row in payload["terms"]:
        word = str(row["word"])
        coefficient = complex(*[float(v) for v in row["coefficient"]])
        if len(word) != nq or set(word) - set("exyz") or not np.isfinite(coefficient):
            raise ValueError(f"invalid Pauli term {row!r}.")
        if coefficient != 0:
            terms.append(PauliTerm(nq, ps=word, pc=coefficient))
    operator = PauliPolynomial("JW", terms)
    known = {"schema", "energy_units", "spin_ordering", "pauli_ordering", "coupling_model", "n_qubits", "n_fermion_qubits",
             "num_particles", "mode_cutoffs", "boson_encoding", "omega_hartree", "source_files", "metadata", "terms"}
    return QdynamicsQubitHamiltonian(
        operator=operator, n_qubits=nq, n_fermion_qubits=n_fermion, num_particles=(particles[0], particles[1]),
        mode_cutoffs=cutoffs, encoding=encoding if cutoffs else None, frequencies_hartree=frequencies,
        coupling_model=payload.get("coupling_model"), source_files=tuple(str(v) for v in payload.get("source_files") or ()),
        metadata=dict(payload.get("metadata") or {}), layout=layout, path=path,
        extra={k: v for k, v in payload.items() if k not in known})


def qdynamics_payload_from_model(
    model: VibronicQubitModel,
    *,
    source_files: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Our vibronic qubit model in his schema (his loader reads it back unchanged)."""
    coupling = QDYNAMICS_COUPLING_MODEL if str(model.coupling_model) == OUR_EQUIVALENT_COUPLING_MODEL else str(model.coupling_model)
    terms = []
    for term in model.h_vibronic.return_polynomial():
        coefficient = complex(term.p_coeff)
        if coefficient == 0:
            continue
        terms.append({"word": str(term.pw2strng()), "coefficient": [float(coefficient.real), float(coefficient.imag)]})
    terms.sort(key=lambda t: t["word"])
    meta = {
        "producer": "Holstein_test src.quantum.chemistry.molecular_vibronic", "builder_model_schema": model.provenance.get("schema"),
        "basis": str(model.basis), "active_space": model.active_space.to_jsonable(),
        "harmonic_curvature": model.provenance.get("harmonic_curvature"), "alignment": model.provenance.get("alignment"),
        "mode_basis": model.provenance.get("mode_basis"),
        "reference_force_counter_term": model.provenance.get("reference_force_counter_term"),
        "coupling_model_ours": str(model.coupling_model),
        "note": ("Q_I = (b + b+)/sqrt(2 omega_I); our coupling C_I per unit Q equals sqrt(2 omega_I) times the "
                 "qdynamics g_I, which already carries the zero-point factor."),
    }
    if metadata:
        meta.update(dict(metadata))
    return {
        "schema": QDYNAMICS_SCHEMA, **QDYNAMICS_CONVENTIONS,
        "coupling_model": coupling if model.modes else None,
        "n_qubits": int(model.n_qubits), "n_fermion_qubits": int(model.layout.n_fermion_qubits),
        "num_particles": [int(model.active_space.n_alpha), int(model.active_space.n_beta)],
        "mode_cutoffs": [int(v) for v in model.cutoffs], "boson_encoding": str(model.encoding) if model.modes else None,
        "omega_hartree": [float(m.frequency_hartree) for m in model.modes],
        "source_files": [str(v) for v in source_files], "metadata": meta, "terms": terms,
    }


def save_qdynamics_qubit_hamiltonian(model: VibronicQubitModel, path: str | Path, *, overwrite: bool = False, **kwargs: Any) -> Path:
    """Write our model in his schema; like his writer it refuses to overwrite unless told to."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(qdynamics_payload_from_model(model, **kwargs), indent=2, allow_nan=False) + "\n"
    with target.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(text)
    return target


def _as_operator_and_layout(h: QdynamicsQubitHamiltonian | VibronicQubitModel) -> tuple[PauliPolynomial, RegisterLayout, int, tuple[int, int]]:
    if isinstance(h, QdynamicsQubitHamiltonian):
        return h.operator, h.layout, h.n_spatial_orbitals, h.num_particles
    return (h.h_vibronic, h.layout, int(h.active_tensors.one_body.shape[0]),
            (int(h.active_space.n_alpha), int(h.active_space.n_beta)))


def hartree_fock_vacuum_index(layout: RegisterLayout, *, n_spatial_orbitals: int, num_particles: tuple[int, int]) -> int:
    """Computational-basis index of |HF> (x) |vacuum>: blocked occupied orbitals, boson level 0."""
    occupied = hartree_fock_occupied_qubits(int(n_spatial_orbitals), (int(num_particles[0]), int(num_particles[1])), indexing="blocked")
    index = sum(1 << int(q) for q in occupied)
    for block in layout.boson_modes:
        if str(block.encoding) == "unary":
            index |= 1 << int(block.qubit_start)
    return index


def diagonal_expectation(operator: PauliPolynomial, index: int, n_qubits: int) -> float:
    """<index| H |index> for a computational-basis state: only words without x or y contribute."""
    total = 0.0
    for term in operator.return_polynomial():
        word = str(term.pw2strng())
        if "x" in word or "y" in word:
            continue
        sign = 1.0
        for position, letter in enumerate(word):
            if letter == "z" and (index >> (n_qubits - 1 - position)) & 1:
                sign = -sign
        total += sign * float(complex(term.p_coeff).real)
    return total


def hartree_fock_vacuum_energy(h: QdynamicsQubitHamiltonian | VibronicQubitModel) -> float:
    """E_RHF + sum_I omega_I / 2 for a model whose coupling has no reference-state expectation value.

    This is the number his 2026-11-02 benchmark table reports as the 'direct classical ground
    state energy' (H2O: -74.94152753613099 Ha = E_RHF + ZPE at the RHF/STO-3G geometry to 4e-9);
    it is the energy of the reference state, not a ground state, and it does not see the
    electron-phonon coupling at all because <vacuum| b + b+ |vacuum> = 0.
    """
    operator, layout, n_spatial, particles = _as_operator_and_layout(h)
    index = hartree_fock_vacuum_index(layout, n_spatial_orbitals=n_spatial, num_particles=particles)
    return diagonal_expectation(operator, index, int(layout.n_total_qubits))


def physical_sector_energies(h: QdynamicsQubitHamiltonian | VibronicQubitModel, *, n_low: int = 1, tol: float = 1.0e-10) -> tuple[float, ...]:
    """Lowest eigenvalues in the physical sector (N_alpha, N_beta electrons, n_I <= cutoff_I)."""
    from scipy.sparse.linalg import eigsh  # type: ignore

    operator, layout, n_spatial, particles = _as_operator_and_layout(h)
    basis = physical_sector_basis(layout, n_spatial_orbitals=n_spatial, num_particles=particles)
    matrix = sector_sparse_matrix(operator, basis)
    dim = int(matrix.shape[0])
    k = max(1, int(n_low))
    if dim <= k + 1:
        values = np.linalg.eigvalsh(matrix.toarray())
    else:
        values = eigsh(matrix, k=min(k, dim - 1), which="SA", tol=float(tol), return_eigenvectors=False)
    return tuple(float(v) for v in np.sort(np.real(values))[:k])


def compare_pauli_polynomials(a: PauliPolynomial, b: PauliPolynomial, *, atol: float = 1.0e-10) -> dict[str, Any]:
    """Term-by-term comparison on the Pauli words; coefficients below ``atol`` count as absent."""
    def table(poly: PauliPolynomial) -> dict[str, complex]:
        out: dict[str, complex] = {}
        for term in poly.return_polynomial():
            c = complex(term.p_coeff)
            if abs(c) > atol:
                out[str(term.pw2strng())] = out.get(str(term.pw2strng()), 0j) + c
        return out

    ta, tb = table(a), table(b)
    shared = sorted(set(ta) & set(tb))
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    max_diff = max((abs(ta[w] - tb[w]) for w in shared), default=0.0)
    worst = max(shared, key=lambda w: abs(ta[w] - tb[w])) if shared else None
    return {
        "n_terms_a": len(ta), "n_terms_b": len(tb), "n_shared": len(shared),
        "max_abs_difference_shared": float(max_diff), "worst_shared_word": worst,
        "n_only_in_a": len(only_a), "n_only_in_b": len(only_b),
        "max_abs_only_in_a": float(max((abs(ta[w]) for w in only_a), default=0.0)),
        "max_abs_only_in_b": float(max((abs(tb[w]) for w in only_b), default=0.0)),
        "examples_only_in_a": only_a[:8], "examples_only_in_b": only_b[:8],
        "equal_within_atol": bool(not only_a and not only_b and max_diff <= atol),
    }


def builder_model_in_qdynamics_conventions(
    geometry: MolecularGeometry,
    *,
    basis: str = "sto-3g",
    cutoffs: Sequence[int] | int = 1,
    encoding: str = "binary",
    **kwargs: Any,
) -> tuple[VibronicQubitModel, Any]:
    """Our builder set to his model: full MO space, RHF-Hessian frequencies and modes, the PySCF
    mean-field vertex, no counter-term. Everything else the builder offers (frozen core, the
    full linear coupling, K_PES + chi oscillators, model-consistent modes) is what his model lacks."""
    return build_molecular_vibronic_model(
        geometry, basis=basis, active_space="full", cutoffs=cutoffs, encoding=encoding,
        coupling_model=OUR_EQUIVALENT_COUPLING_MODEL, reference_force_counter_term=False,
        harmonic_curvature="rhf_hessian", mode_basis="rhf_normal_modes", **kwargs)
