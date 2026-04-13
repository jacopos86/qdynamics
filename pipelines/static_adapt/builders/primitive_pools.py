"""Primitive ADAPT pool builders extracted from the static ADAPT pipeline."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from src.quantum.hubbard_latex_python_pairs import (
    bravais_nearest_neighbor_edges,
    boson_qubits_per_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinLayerwiseAnsatz,
    HubbardTermwiseAnsatz,
    half_filled_num_particles,
)

try:
    from src.quantum.operator_pools import make_pool as make_paop_pool
except Exception as exc:  # pragma: no cover - defensive fallback
    make_paop_pool = None
    _PAOP_IMPORT_ERROR = str(exc)
else:
    _PAOP_IMPORT_ERROR = ""

try:
    from src.quantum.operator_pools.polaron_paop import make_phonon_motifs
except Exception:  # pragma: no cover - optional legacy product-family seam
    make_phonon_motifs = None

try:
    from src.quantum.operator_pools.vlf_sq import build_vlf_sq_pool as build_vlf_sq_family
except Exception:  # pragma: no cover - optional VLF/SQ family seam
    build_vlf_sq_family = None

_HH_UCCSD_PAOP_PRODUCT_SPECS: dict[str, dict[str, Any]] = {
    "uccsd_otimes_paop_lf_std": {
        "motif_family": "paop_lf_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf2_std": {
        "motif_family": "paop_lf2_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_bond_disp_std": {
        "motif_family": "paop_bond_disp_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf_std_seq2p": {
        "motif_family": "paop_lf_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf2_std_seq2p": {
        "motif_family": "paop_lf2_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_bond_disp_std_seq2p": {
        "motif_family": "paop_bond_disp_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
}

_UCCSD_SINGLE_LABEL_RE = re.compile(r"^uccsd_sing\((alpha|beta):(\d+)->(\d+)\)$")
_UCCSD_DOUBLE_LABEL_RE = re.compile(r"^uccsd_dbl\((aa|bb|ab):(\d+),(\d+)->(\d+),(\d+)\)$")


def _build_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    """Build the UCCSD operator pool using HardcodedUCCSDAnsatz.base_terms."""
    dummy_ansatz = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=num_particles,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_cse_pool(
    num_sites: int,
    ordering: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
) -> list[AnsatzTerm]:
    """Build a CSE-style pool from the term-wise Hubbard ansatz base terms."""
    dummy_ansatz = HubbardTermwiseAnsatz(
        dims=int(num_sites),
        t=float(t),
        U=float(u),
        v=float(dv),
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_potential_terms=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_full_hamiltonian_pool(
    h_poly: Any,
    tol: float = 1e-12,
    normalize_coeff: bool = False,
) -> list[AnsatzTerm]:
    """Build a pool with one generator per non-identity Hamiltonian Pauli term."""
    pool: list[AnsatzTerm] = []
    terms = h_poly.return_polynomial()
    if not terms:
        return pool
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label:
            continue
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(
                f"Non-negligible imaginary Hamiltonian coefficient for term {label}: {coeff}"
            )
        generator = PauliPolynomial("JW")
        term_coeff = 1.0 if bool(normalize_coeff) else float(coeff.real)
        label_prefix = "ham_unit_term" if bool(normalize_coeff) else "ham_term"
        generator.add_term(PauliTerm(nq, ps=label, pc=float(term_coeff)))
        pool.append(AnsatzTerm(label=f"{label_prefix}({label})", polynomial=generator))
    return pool


def _polynomial_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    """Canonical real-valued signature for deduplicating PauliPolynomial generators."""
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(f"Non-negligible imaginary coefficient in pool polynomial: {coeff} ({label})")
        items.append((label, round(float(coeff.real), 12)))
    items.sort()
    return tuple(items)


def _build_hh_termwise_augmented_pool(h_poly: Any, tol: float = 1e-12) -> list[AnsatzTerm]:
    """HH-only termwise pool: unit-normalized Hamiltonian terms + x->y quadrature partners."""
    base_pool = _build_full_hamiltonian_pool(h_poly, tol=tol, normalize_coeff=True)
    if not base_pool:
        return []

    terms = h_poly.return_polynomial()
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    seen_labels: set[str] = set()
    for op in base_pool:
        op_terms = op.polynomial.return_polynomial()
        if not op_terms:
            continue
        seen_labels.add(str(op_terms[0].pw2strng()))

    aug_pool = list(base_pool)
    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label or abs(coeff) <= tol:
            continue
        if "x" not in label:
            continue
        y_label = label.replace("x", "y")
        if y_label in seen_labels:
            continue
        gen = PauliPolynomial("JW")
        y_coeff = abs(float(coeff.real))
        if y_coeff <= tol:
            y_coeff = 1.0
        gen.add_term(PauliTerm(nq, ps=y_label, pc=y_coeff))
        aug_pool.append(AnsatzTerm(label=f"ham_quadrature_term({y_label})", polynomial=gen))
        seen_labels.add(y_label)
    return aug_pool


def _build_hva_pool(
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
) -> list[AnsatzTerm]:
    layerwise = HubbardHolsteinLayerwiseAnsatz(
        dims=int(num_sites),
        J=float(t),
        U=float(u),
        omega0=float(omega0),
        g=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        v=None,
        v_t=float(dv),
        v0=None,
        t_eval=None,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )
    pool: list[AnsatzTerm] = list(layerwise.base_terms)
    n_sites = int(num_sites)
    pool.extend(
        _build_hh_uccsd_fermion_lifted_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            num_particles=tuple(half_filled_num_particles(n_sites)),
        )
    )
    return pool


def _build_hh_uccsd_fermion_lifted_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    num_particles: tuple[int, int] | None = None,
) -> list[AnsatzTerm]:
    """HH-only UCCSD pool lifted into full HH register with boson identity prefix."""
    n_sites = int(num_sites)
    num_particles_eff = tuple(num_particles) if num_particles is not None else tuple(half_filled_num_particles(n_sites))
    ferm_nq = 2 * n_sites
    boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = ferm_nq + boson_bits

    uccsd_kwargs = {
        "dims": n_sites,
        "num_particles": num_particles_eff,
        "include_singles": True,
        "include_doubles": True,
        "repr_mode": "JW",
        "indexing": str(ordering),
    }
    if str(boundary).strip().lower() == "periodic":
        try:
            uccsd_kwargs["pbc"] = True
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
        except TypeError as exc:
            if "pbc" not in str(exc):
                raise
            uccsd_kwargs.pop("pbc", None)
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    else:
        uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)

    lifted_pool: list[AnsatzTerm] = []
    for op in uccsd.base_terms:
        lifted = PauliPolynomial("JW")
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-12:
                raise ValueError(f"Non-negligible imaginary UCCSD coefficient in {op.label}: {coeff}")
            ferm_ps = str(term.pw2strng())
            if len(ferm_ps) != ferm_nq:
                raise ValueError(
                    f"Unexpected fermion Pauli length {len(ferm_ps)} != {ferm_nq} for UCCSD operator {op.label}"
                )
            full_ps = ("e" * boson_bits) + ferm_ps
            lifted.add_term(PauliTerm(nq_total, ps=full_ps, pc=float(coeff.real)))
        if len(lifted.return_polynomial()) == 0:
            continue
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))
    return lifted_pool


def _build_paop_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
    if make_paop_pool is None:
        raise RuntimeError(f"PAOP pool requested but operator_pools module unavailable: {_PAOP_IMPORT_ERROR}")

    pool_specs = make_paop_pool(
        pool_key,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=tuple(num_particles),
    )
    return [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs]


def _build_vlf_sq_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    if build_vlf_sq_family is None:
        raise RuntimeError(f"VLF/SQ pool requested but operator_pools module unavailable: {_PAOP_IMPORT_ERROR}")
    if bool(paop_split_paulis):
        raise ValueError("VLF/SQ macro families do not support --paop-split-paulis; keep grouped macro generators intact.")
    pool_specs, meta = build_vlf_sq_family(
        pool_key,
        num_sites=int(num_sites),
        num_particles=tuple(num_particles),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        shell_radius=None,
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
    )
    return [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs], dict(meta)


def _clean_real_pool_polynomial(poly: Any, prune_eps: float = 0.0) -> PauliPolynomial:
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    nq = int(terms[0].nqubit())
    cleaned = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(prune_eps):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in product-family pool term: {coeff}")
        cleaned.add_term(PauliTerm(nq, ps=str(term.pw2strng()), pc=float(coeff.real)))
    cleaned._reduce()
    return cleaned


def _fermion_mode_to_site(mode: int, *, num_sites: int, ordering: str) -> int:
    mode_i = int(mode)
    n_sites = int(num_sites)
    ordering_key = str(ordering).strip().lower()
    if mode_i < 0 or mode_i >= 2 * n_sites:
        raise ValueError(f"Fermion mode {mode_i} out of range for num_sites={n_sites}")
    if ordering_key == "interleaved":
        return mode_i // 2
    if ordering_key == "blocked":
        if mode_i < n_sites:
            return mode_i
        return mode_i - n_sites
    raise ValueError(f"Unsupported fermion ordering '{ordering}'.")


def _parse_lifted_uccsd_support(
    label: str,
    *,
    num_sites: int,
    ordering: str,
) -> tuple[str, tuple[int, ...]]:
    raw = str(label).strip()
    prefix = "uccsd_ferm_lifted::"
    if not raw.startswith(prefix):
        raise ValueError(f"Unsupported lifted UCCSD label '{raw}'.")
    body = raw[len(prefix):]

    m_single = _UCCSD_SINGLE_LABEL_RE.match(body)
    if m_single is not None:
        modes = [int(m_single.group(2)), int(m_single.group(3))]
        kind = "single"
    else:
        m_double = _UCCSD_DOUBLE_LABEL_RE.match(body)
        if m_double is None:
            raise ValueError(f"Could not parse lifted UCCSD label '{raw}'.")
        modes = [
            int(m_double.group(2)),
            int(m_double.group(3)),
            int(m_double.group(4)),
            int(m_double.group(5)),
        ]
        kind = "double"

    sites = tuple(
        sorted(
            {
                _fermion_mode_to_site(mode, num_sites=int(num_sites), ordering=str(ordering))
                for mode in modes
            }
        )
    )
    return kind, sites


def _motif_matches_excitation_support(
    *,
    motif: Any,
    motif_family: str,
    support_sites: tuple[int, ...],
    nearest_neighbor_bonds: set[tuple[int, int]],
) -> bool:
    support_set = {int(site) for site in support_sites}
    motif_sites = {int(site) for site in getattr(motif, "sites", ())}
    motif_bonds = tuple(tuple(sorted((int(i), int(j)))) for i, j in getattr(motif, "bonds", ()))
    if not motif_sites:
        return False
    if not motif_bonds:
        return bool(motif_sites & support_set)

    if str(motif_family).strip().lower() == "paop_bond_disp_std":
        for bond in motif_bonds:
            if set(bond).issubset(support_set):
                return True
            if bond in nearest_neighbor_bonds and bond[0] in support_set and bond[1] in support_set:
                return True
        return False

    return bool(motif_sites & support_set)


def _build_hh_uccsd_paop_product_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    family_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    del paop_r
    if make_phonon_motifs is None:
        raise RuntimeError(f"PAOP product pool requested but operator_pools module unavailable: {_PAOP_IMPORT_ERROR}")

    family_key_norm = str(family_key).strip().lower()
    spec = _HH_UCCSD_PAOP_PRODUCT_SPECS.get(family_key_norm)
    if spec is None:
        raise ValueError(f"Unsupported HH UCCSD⊗PAOP product family '{family_key}'.")
    if bool(paop_split_paulis):
        raise ValueError("UCCSD⊗PAOP product families do not support --paop-split-paulis; keep grouped logical generators intact.")

    motif_family = str(spec["motif_family"])
    parameterization = str(spec["parameterization"])
    seq2p = parameterization == "double_sequential"
    family_label_prefix = "uccsd_otimes_paop_seq2p" if seq2p else "uccsd_otimes_paop"

    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=tuple(num_particles),
    )
    motifs = make_phonon_motifs(
        motif_family,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        boundary=str(boundary),
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
    )
    nearest_neighbor_bonds = {
        tuple(sorted((int(i), int(j))))
        for i, j in bravais_nearest_neighbor_edges(
            int(num_sites),
            pbc=(str(boundary).strip().lower() == "periodic"),
        )
    }

    sorted_uccsd = sorted(
        list(uccsd_lifted_pool),
        key=lambda op: (
            0 if _parse_lifted_uccsd_support(str(op.label), num_sites=int(num_sites), ordering=str(ordering))[0] == "single" else 1,
            str(op.label),
        ),
    )
    ordered_motifs = sorted(list(motifs), key=lambda motif: (str(motif.family), str(motif.label)))

    raw_pool: list[AnsatzTerm] = []
    raw_pair_count = 0
    for op in sorted_uccsd:
        _kind, support_sites = _parse_lifted_uccsd_support(
            str(op.label),
            num_sites=int(num_sites),
            ordering=str(ordering),
        )
        for motif in ordered_motifs:
            if not _motif_matches_excitation_support(
                motif=motif,
                motif_family=motif_family,
                support_sites=support_sites,
                nearest_neighbor_bonds=nearest_neighbor_bonds,
            ):
                continue
            raw_pair_count += 1
            base_label = f"{family_label_prefix}::{op.label}::{motif.family}::{motif.label}"
            if seq2p:
                raw_pool.append(AnsatzTerm(label=f"{base_label}::step=ferm", polynomial=op.polynomial))
                raw_pool.append(AnsatzTerm(label=f"{base_label}::step=motif", polynomial=motif.poly))
                continue
            product_poly = _clean_real_pool_polynomial(op.polynomial * motif.poly, float(paop_prune_eps))
            if not product_poly.return_polynomial():
                continue
            raw_pool.append(AnsatzTerm(label=base_label, polynomial=product_poly))

    if seq2p:
        pool = list(raw_pool)
        dedup_strategy = "disabled_pair_label_preserving"
    elif int(n_ph_max) >= 2:
        pool = _deduplicate_pool_terms_lightweight(raw_pool)
        dedup_strategy = "signature_digest"
    else:
        pool = _deduplicate_pool_terms(raw_pool)
        dedup_strategy = "signature"

    return list(pool), {
        "family": family_key_norm,
        "family_kind": "uccsd_paop_product",
        "parameterization": parameterization,
        "motif_family": motif_family,
        "locality_rule": (
            "lf_overlap"
            if motif_family in {"paop_lf_std", "paop_lf2_std"}
            else "bond_disp_local_compatible"
        ),
        "raw_sizes": {
            "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
            "raw_phonon_motifs": int(len(motifs)),
            "raw_logical_pairs": int(raw_pair_count),
            "raw_emitted_terms": int(len(raw_pool)),
        },
        "logical_element_count": int(raw_pair_count),
        "expanded_term_count": int(len(pool)),
        "dedup_strategy": dedup_strategy,
        "dedup_total": int(len(pool)),
    }


def _deduplicate_pool_terms(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def _polynomial_signature_digest(poly: Any, tol: float = 1e-12) -> str:
    h = hashlib.sha1()
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in pool term: {coeff}")
        label = str(term.pw2strng())
        coeff_real = round(float(coeff.real), 12)
        h.update(label.encode("ascii", errors="ignore"))
        h.update(b":")
        h.update(f"{coeff_real:+.12e}".encode("ascii"))
        h.update(b";")
    return h.hexdigest()


def _deduplicate_pool_terms_lightweight(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    seen: set[str] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature_digest(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


__all__ = [
    "_HH_UCCSD_PAOP_PRODUCT_SPECS",
    "_build_cse_pool",
    "_build_full_hamiltonian_pool",
    "_build_hh_termwise_augmented_pool",
    "_build_hh_uccsd_fermion_lifted_pool",
    "_build_hh_uccsd_paop_product_pool",
    "_build_hva_pool",
    "_build_paop_pool",
    "_build_uccsd_pool",
    "_build_vlf_sq_pool",
    "_clean_real_pool_polynomial",
    "_deduplicate_pool_terms",
    "_deduplicate_pool_terms_lightweight",
    "_fermion_mode_to_site",
    "_motif_matches_excitation_support",
    "_parse_lifted_uccsd_support",
    "_polynomial_signature",
    "_polynomial_signature_digest",
]
