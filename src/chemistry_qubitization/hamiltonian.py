"""Compose electronic and optional linear mean-field electron-phonon terms."""
import math
from .data import QubitHamiltonian
from .fermion import electronic, one_body
from .boson import oscillator, register_size
from .pauli import PauliPolynomial, PauliTerm


def build_hamiltonian(integrals, *, cutoffs=None, encoding="binary", tolerance=1e-12):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    n_fermion = 2 * len(integrals.h1)
    n_modes = 0 if integrals.frequencies is None else len(integrals.frequencies)
    values = () if cutoffs is None else (cutoffs,) if isinstance(cutoffs, int) else tuple(cutoffs)
    if n_modes:
        if len(values) == 1:
            values *= n_modes
        if len(values) != n_modes:
            raise ValueError("Provide one cutoff or one cutoff per phonon mode")
    elif values:
        raise ValueError("Boson cutoffs supplied without e-ph input")
    widths = tuple(register_size(value, encoding) for value in values)
    nq = n_fermion + sum(widths)
    result = electronic(integrals, nq)
    offset = n_fermion
    for mode, (cutoff, width) in enumerate(zip(values, widths)):
        number, displacement = oscillator(cutoff, encoding, offset, nq)
        result += float(integrals.frequencies[mode]) * number
        result += PauliPolynomial("JW", [PauliTerm(nq, ps="e"*nq, pc=0.5*float(integrals.frequencies[mode]))])
        result += one_body(integrals.coupling[mode], nq) * displacement
        offset += width
    result._reduce()
    terms = result.return_polynomial()
    if any(abs(complex(term.p_coeff).imag) > max(tolerance, 1e-10) for term in terms):
        raise ValueError("Mapped Hamiltonian is not Hermitian")
    result = PauliPolynomial("JW", [PauliTerm(nq, ps=term.pw2strng(), pc=float(complex(term.p_coeff).real))
                                   for term in terms if abs(term.p_coeff) > tolerance])
    return QubitHamiltonian(result, nq, n_fermion, integrals.particles, values,
                            encoding if n_modes else None,
                            tuple(map(float, integrals.frequencies)) if n_modes else (),
                            integrals.source_files, {**integrals.metadata, "coefficient_tolerance": tolerance})
