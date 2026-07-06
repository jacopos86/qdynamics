import itertools
import math
from functools import lru_cache
from typing import Dict, Sequence
import numpy as np
from src.qubitization.pauli_polynomial_class import PauliPolynomial
from src.qubitization.qubitization_module import PauliTerm
from src.utilities.log import log
from src.hamiltonians.containers import ProblemHamiltonianData
from src.hamiltonians.hubbard import (
    SPIN_DN,
    SPIN_UP,
    build_hubbard_hamiltonian,
    jw_number_operator,
    mode_index,
    n_sites_from_dims,
)



def boson_qubits_per_site(n_ph_max, boson_encoding="binary"):
    """Return qubits needed for one truncated local phonon mode."""
    n_ph_i = int(n_ph_max)
    if n_ph_i < 0:
        log.error("n_ph_max must be >= 0")
    enc = str(boson_encoding).strip().lower()
    if enc == "binary":
        return max(1, int(math.ceil(math.log2(n_ph_i + 1))))
    if enc == "unary":
        return n_ph_i + 1
    log.error(f"Unsupported boson_encoding: {boson_encoding}")


def phonon_qubits_for_site(site, *, n_sites, qpb, fermion_qubits):
    """Global qubit indices for a site's phonon register."""
    site_i = int(site)
    if site_i < 0 or site_i >= int(n_sites):
        log.error("site index out of bounds")
    base = int(fermion_qubits) + site_i * int(qpb)
    return list(range(base, base + int(qpb)))


def _identity_pauli(repr_mode, nq):
    return PauliPolynomial(repr_mode, [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


def _pauli_monomial(repr_mode, nq, ops: Dict[int, str]):
    chars = ["e"] * int(nq)
    for qubit, symbol in ops.items():
        q = int(qubit)
        if q < 0 or q >= int(nq):
            log.error("qubit index out of range")
        s = str(symbol).lower().replace("i", "e")
        if s not in ("e", "x", "y", "z"):
            log.error(f"Invalid Pauli symbol: {symbol}")
        chars[int(nq) - 1 - q] = s
    return PauliPolynomial(repr_mode, [PauliTerm(int(nq), ps="".join(chars), pc=1.0)])


@lru_cache(maxsize=None)
def _pauli_basis_mats(n_qubits):
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    oneq = {"I": I, "X": X, "Y": Y, "Z": Z}
    labels = []
    mats = []
    for label_tuple in itertools.product("IXYZ", repeat=int(n_qubits)):
        label = "".join(label_tuple)
        mat = oneq[label[0]]
        for symbol in label[1:]:
            mat = np.kron(mat, oneq[symbol])
        labels.append(label)
        mats.append(mat)
    return labels, mats


def _binary_displacement_pauli(repr_mode, nq, qubits, *, n_ph_max, tol=1e-12):
    """Pauli decomposition of x = b + bdag for binary phonon encoding."""
    qpb = len(qubits)
    d_ph = int(n_ph_max) + 1
    dim = 1 << qpb
    x_op = np.zeros((dim, dim), dtype=complex)
    for n in range(d_ph - 1):
        amp = math.sqrt(float(n + 1))
        x_op[n + 1, n] = amp
        x_op[n, n + 1] = amp

    labels, mats = _pauli_basis_mats(qpb)
    x_pauli = PauliPolynomial(repr_mode)
    for label, mat in zip(labels, mats):
        coeff = np.trace(mat.conj().T @ x_op) / float(dim)
        if abs(coeff) <= tol:
            continue
        ops = {}
        for local_pos, symbol in enumerate(label):
            if symbol == "I":
                continue
            global_q = qubits[qpb - 1 - local_pos]
            ops[global_q] = symbol
        x_pauli += complex(coeff) * _pauli_monomial(repr_mode, nq, ops)
    return x_pauli


def _unary_displacement_pauli(repr_mode, nq, qubits, *, n_ph_max):
    """Pauli form of x = b + bdag for one-hot phonon encoding."""
    x_pauli = PauliPolynomial(repr_mode)
    for n in range(int(n_ph_max)):
        qn = int(qubits[n])
        qp = int(qubits[n + 1])
        coeff = 0.5 * math.sqrt(float(n + 1))
        x_pauli += coeff * _pauli_monomial(repr_mode, nq, {qn: "X", qp: "X"})
        x_pauli += coeff * _pauli_monomial(repr_mode, nq, {qn: "Y", qp: "Y"})
    return x_pauli


def phonon_displacement_operator(
    repr_mode,
    nq,
    qubits: Sequence[int],
    *,
    n_ph_max,
    boson_encoding="binary",
):
    """Build the local phonon displacement operator x = b + bdag."""
    enc = str(boson_encoding).strip().lower()
    if enc == "binary":
        return _binary_displacement_pauli(
            repr_mode, int(nq), list(qubits), n_ph_max=int(n_ph_max)
        )
    if enc == "unary":
        return _unary_displacement_pauli(
            repr_mode, int(nq), list(qubits), n_ph_max=int(n_ph_max)
        )
    log.error(f"Unsupported boson_encoding: {boson_encoding}")

#
#  build electron-boson term
#

def build_electron_phonon_coupling_term(
    *,
    L,
    g_ep,
    n_ph_max,
    boson_encoding="binary",
    ordering="blocked",
    repr_mode="JW",
    density_shift=0.0,
):
    """Build the electron-phonon coupling term H_eph = g(n↑ + n↓)(b + b†)"""
    n_sites = n_sites_from_dims(L)
    fermion_qubits = 2 * n_sites
    qpb = boson_qubits_per_site(n_ph_max, boson_encoding)
    nq_total = fermion_qubits + n_sites * qpb
    identity = _identity_pauli(repr_mode, nq_total)
    H_eph = PauliPolynomial(repr_mode)

    for site in range(n_sites):
        p_up = mode_index(site, SPIN_UP, indexing=ordering, n_sites=n_sites)
        p_dn = mode_index(site, SPIN_DN, indexing=ordering, n_sites=n_sites)
        n_el = (
            jw_number_operator(repr_mode, nq_total, p_up)
            + jw_number_operator(repr_mode, nq_total, p_dn)
        )
        if abs(float(density_shift)) > 0.0:
            n_el += (-float(density_shift)) * identity

        ph_qubits = phonon_qubits_for_site(
            site,
            n_sites=n_sites,
            qpb=qpb,
            fermion_qubits=fermion_qubits,
        )
        x_ph = phonon_displacement_operator(
            repr_mode,
            nq_total,
            ph_qubits,
            n_ph_max=n_ph_max,
            boson_encoding=boson_encoding,
        )
        H_eph += float(g_ep) * (n_el * x_ph)
    return H_eph

#
#   Hubbard-Holstein model
#

def build_hubbard_holstein_hamiltonian(
    L,
    t,
    u,
    dv,
    omega0,
    g_ep,
    n_ph_max,
    boundary="open",
    ordering="blocked",
    boson_encoding="binary",
    repr_mode="JW",
    **kwargs,
):
    """
    Holstein-Hubbard builder.
    
    Repo architecture requires:
    build_problem_hamiltonian() -> build_hubbard_holstein_hamiltonian()
    
    Current status:
    1. Hubbard electronic Pauli terms are built.
    2. Electron-phonon coupling H_eph = g(n↑ + n↓)(b + b†) is built.
    3. Phonon energy H_ph = ω₀(b†b + 1/2) is still pending.
    """
    log.warning("build_hubbard_holstein_hamiltonian() is NOT YET FULLY IMPLEMENTED")
    log.warning("Needs: phonon energy term")
    n_sites = n_sites_from_dims(L)
    qpb = boson_qubits_per_site(n_ph_max, boson_encoding)
    total_qubits = 2 * n_sites + n_sites * qpb
    H_pauli = build_hubbard_hamiltonian(
        dims=L,
        t=t,
        U=u,
        v=dv,
        repr_mode=repr_mode,
        indexing=ordering,
        pbc=(boundary == "periodic"),
        nq_override=total_qubits,
    )
    # get electron - boson term here
    H_pauli += build_electron_phonon_coupling_term(
        L=L,
        g_ep=g_ep,
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        ordering=ordering,
        repr_mode=repr_mode,
    )
    # At this stage, return None to indicate incomplete implementation
    return ProblemHamiltonianData(
        problem_type="hh",
        H=H_pauli,
        basis=None,
        subspaces={
            "electronic": None,
            "phonon": None,
        },
        metadata={
            "L": L,
            "t": t,
            "u": u,
            "dv": dv,
            "omega0": omega0,
            "g_ep": g_ep,
            "n_ph_max": n_ph_max,
            "boundary": boundary,
            "ordering": ordering,
            "boson_encoding": boson_encoding,
            "qpb": qpb,
            "total_qubits": total_qubits,
            "status": "INCOMPLETE - missing phonon energy term",
        },
    )
