"""Bridge-local generator pools for the molecular restricted closed-shell family.

Hermitian, number-conserving excitation blocks for real-time dynamics::

    G_single(p -> a; sigma) = a†_a a_p + a†_p a_a                 (2 Pauli words)
    G_double(i j -> a b)    = a†_a a†_b a_j a_i + a†_i a†_j a_b a_a  (8 Pauli words)

Each executed unitary is ``exp(-i theta G)`` (the repository ``AnsatzTerm``
convention).  Every word of such a block has an even number of ``Y`` factors
with a real coefficient, so the block is a real symmetric matrix and its
exponential is genuinely complex; the words of one block mutually commute, so
the repository's termwise-product execution of a single shared angle is the
exact exponential (the audit records ``all_nonzero_components_mutually_commute``
per generator; the product-vs-``expm`` equality itself was measured at 1e-15 on
the witnesses and on H2O, not checked at run time) and preserves
``(n_up, n_dn)`` (the fixed-count preservation IS verified at run time by the
executed-unitary sector audit, never assumed).

Contrast with the repository UCCSD pool ``i (T - T†)``: those words carry an
odd number of ``Y`` factors, ``exp(-i theta G)`` is then real orthogonal, a
real reference state stays real, and the McLachlan velocity from the real
Hartree--Fock reference vanishes identically (the "frozen" trajectories recorded
on 2026-10-19/20).  Energy-minimising ADAPT wants the anti-Hermitian family;
real-time propagation from a real reference needs this Hermitian family.

The pool is bridge-local on purpose: ``resolve_pool_plan`` and the molecular
UCCSD builder are rated CRITICAL by the repository impact graph (49 and 38
upstream dependents in the Paper-I lanes), so the molecular dynamics pool is
resolved inside ``pipelines.chemistry_bridge`` instead of adding a key there.
"""

from __future__ import annotations

from typing import Sequence

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    AnsatzTerm,
    fermion_minus_operator,
    fermion_plus_operator,
    mode_index,
)

HERMITIAN_EXCITATIONS_POOL_KEY = "hermitian_excitations"
HERMITIAN_EXCITATIONS_METHOD_NAME = "bridge_hermitian_excitations_molecular"
BRIDGE_LOCAL_POOL_KEYS: frozenset[str] = frozenset({HERMITIAN_EXCITATIONS_POOL_KEY})

__all__ = [
    "BRIDGE_LOCAL_POOL_KEYS",
    "HERMITIAN_EXCITATIONS_METHOD_NAME",
    "HERMITIAN_EXCITATIONS_POOL_KEY",
    "build_hermitian_excitation_pool",
    "hermitian_block_polynomial",
    "is_bridge_local_pool_key",
]


def is_bridge_local_pool_key(pool_key: str) -> bool:
    return str(pool_key).strip().lower() in BRIDGE_LOCAL_POOL_KEYS


def _real_clean(poly: PauliPolynomial, *, nq: int, tol: float = 1.0e-12) -> PauliPolynomial:
    """Rebuild ``poly`` with real coefficients, dropping numerically zero words.

    Raises if any surviving coefficient has an imaginary part above ``tol``:
    a Hermitian block ``T + T†`` must expand with real coefficients, and a
    non-real residue would mean the ladder-operator algebra was misapplied.
    """

    cleaned = PauliPolynomial("JW")
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(f"Hermitian excitation block has a non-real coefficient {coeff} on {term.pw2strng()}")
        cleaned.add_term(PauliTerm(int(nq), ps=str(term.pw2strng()), pc=float(coeff.real)))
    return cleaned


def hermitian_block_polynomial(nq: int, creators: Sequence[int], annihilators: Sequence[int], *, repr_mode: str = "JW") -> PauliPolynomial:
    """``prod a†_c (c in creators) * prod a_d (d in annihilators)`` plus its Hermitian conjugate."""

    excite = None
    for c in creators:
        op = fermion_plus_operator(repr_mode, int(nq), int(c))
        excite = op if excite is None else excite * op
    for d in annihilators:
        op = fermion_minus_operator(repr_mode, int(nq), int(d))
        excite = op if excite is None else excite * op
    deexcite = None
    for d in reversed(list(annihilators)):
        op = fermion_plus_operator(repr_mode, int(nq), int(d))
        deexcite = op if deexcite is None else deexcite * op
    for c in reversed(list(creators)):
        op = fermion_minus_operator(repr_mode, int(nq), int(c))
        deexcite = op if deexcite is None else deexcite * op
    return _real_clean(excite + deexcite, nq=int(nq))


def build_hermitian_excitation_pool(
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    ordering: str = "blocked",
    include_singles: bool = True,
    include_doubles: bool = True,
) -> list[AnsatzTerm]:
    """Hermitian singles and doubles relative to the occupied-prefix reference.

    Enumeration mirrors the repository UCCSD pool (same occupied/virtual split
    per spin, same spin classes ``alpha``/``beta`` for singles and ``aa``/``bb``/``ab``
    for doubles) so that every ``herm_*`` label pairs with a ``uccsd_*`` label.
    """

    if str(ordering).strip().lower() != "blocked":
        raise ValueError("hermitian_excitations pool supports ordering='blocked' only")
    n_sites = int(n_spatial_orbitals)
    n_alpha, n_beta = (int(num_particles[0]), int(num_particles[1]))
    if not (0 <= n_alpha <= n_sites and 0 <= n_beta <= n_sites):
        raise ValueError("num_particles must lie within the spatial orbital count per spin")
    nq = 2 * n_sites
    alpha_all = [mode_index(i, SPIN_UP, indexing="blocked", n_sites=n_sites) for i in range(n_sites)]
    beta_all = [mode_index(i, SPIN_DN, indexing="blocked", n_sites=n_sites) for i in range(n_sites)]
    alpha_occ, alpha_virt = alpha_all[:n_alpha], alpha_all[n_alpha:]
    beta_occ, beta_virt = beta_all[:n_beta], beta_all[n_beta:]
    pool: list[AnsatzTerm] = []
    if include_singles:
        for spin, occ, virt in (("alpha", alpha_occ, alpha_virt), ("beta", beta_occ, beta_virt)):
            for i in occ:
                for a in virt:
                    pool.append(AnsatzTerm(label=f"herm_sing({spin}:{i}->{a})", polynomial=hermitian_block_polynomial(nq, (a,), (i,))))
    if include_doubles:
        for spin, occ, virt in (("aa", alpha_occ, alpha_virt), ("bb", beta_occ, beta_virt)):
            for ip in range(len(occ)):
                for jp in range(ip + 1, len(occ)):
                    for ap in range(len(virt)):
                        for bp in range(ap + 1, len(virt)):
                            i, j, a, b = occ[ip], occ[jp], virt[ap], virt[bp]
                            pool.append(AnsatzTerm(label=f"herm_dbl({spin}:{i},{j}->{a},{b})", polynomial=hermitian_block_polynomial(nq, (a, b), (j, i))))
        for i in alpha_occ:
            for j in beta_occ:
                for a in alpha_virt:
                    for b in beta_virt:
                        pool.append(AnsatzTerm(label=f"herm_dbl(ab:{i},{j}->{a},{b})", polynomial=hermitian_block_polynomial(nq, (a, b), (j, i))))
    return pool


# ---------------------------------------------------------------------------
# Excitation-level QSE bases built from products of commuting UCCSD generators
# ---------------------------------------------------------------------------

import re as _re

_EXCITATION_LABEL = _re.compile(r"^(uccsd_sing|uccsd_dbl|herm_sing|herm_dbl)\((?:alpha|beta|aa|bb|ab):([0-9,]+)->([0-9,]+)\)$")


def excitation_signature(label: str) -> tuple[frozenset[int], frozenset[int]]:
    """``(annihilated occupied spin-orbitals, created virtual spin-orbitals)`` of a pool label."""

    match = _EXCITATION_LABEL.match(str(label))
    if match is None:
        raise ValueError(f"not an excitation label: {label!r}")
    occ = frozenset(int(x) for x in match.group(2).split(",") if x)
    virt = frozenset(int(x) for x in match.group(3).split(",") if x)
    return occ, virt


def build_product_excitation_basis(pool: Sequence[AnsatzTerm], *, max_level: int) -> list[tuple[str, PauliPolynomial, int]]:
    """Singles and doubles of ``pool`` plus products of commuting generators up to ``max_level``.

    Two pool generators with disjoint occupied sets and disjoint virtual sets
    act on different spin-orbitals, commute, and their product applied to the
    reference determinant is (up to a phase) the single determinant excited by
    both: a double times a single is a triple, a double times a double is a
    quadruple.  One representative is kept per excited determinant, so the
    returned elements are linearly independent on the reference.  Levels 5 and 6
    are not generated (products of three generators are out of scope); whether
    the result spans the fixed-count sector is measured by the caller through
    the pencil rank, never assumed.  Returns ``(label, polynomial, level)``.
    """

    level_cap = int(max_level)
    if level_cap < 2 or level_cap > 4:
        raise ValueError("max_level must be 2, 3 or 4")
    base: list[tuple[str, PauliPolynomial, int, frozenset[int], frozenset[int]]] = []
    for term in pool:
        occ, virt = excitation_signature(str(term.label))
        base.append((str(term.label), term.polynomial, int(len(occ)), occ, virt))
    seen: set[tuple[frozenset[int], frozenset[int]]] = {(o, v) for _, _, _, o, v in base}
    out: list[tuple[str, PauliPolynomial, int]] = [(label, poly, level) for label, poly, level, _, _ in base]
    doubles = [row for row in base if row[2] == 2]
    partners = [row for row in base if row[2] in (1, 2)]
    for d_label, d_poly, d_level, d_occ, d_virt in doubles:
        for p_label, p_poly, p_level, p_occ, p_virt in partners:
            level = d_level + p_level
            if level > level_cap:
                continue
            if p_level == 2 and p_label <= d_label:
                continue  # unordered pair of doubles, once
            if (d_occ & p_occ) or (d_virt & p_virt):
                continue  # overlapping spin-orbitals: not a commuting, non-vanishing product
            key = (d_occ | p_occ, d_virt | p_virt)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"prod[{d_label}*{p_label}]", d_poly * p_poly, level))
    return out


__all__ += ["build_product_excitation_basis", "excitation_signature"]
