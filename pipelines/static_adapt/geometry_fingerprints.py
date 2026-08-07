"""Stable fingerprints shared by exact static-ADAPT geometry modules."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from pipelines.scaffold.hh_continuation_scoring import (
    _candidate_coordinate_fingerprint,
    _compiled_polynomial_fingerprint,
    _ordered_scaffold_fingerprint,
)


def fingerprint_jsonable(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_generator_fingerprint(candidate_term: Any) -> str:
    polynomial = getattr(candidate_term, "polynomial", None)
    term_provider = getattr(polynomial, "return_polynomial", None)
    polynomial_terms = (
        tuple(term_provider())
        if callable(term_provider)
        else tuple(getattr(polynomial, "terms", ()))
    )
    terms = []
    for term in polynomial_terms:
        coefficient = complex(
            getattr(term, "p_coeff", getattr(term, "coeff", 0.0))
        )
        word_builder = getattr(term, "pw2strng", None)
        if callable(word_builder):
            pauli_word = str(word_builder())
        else:
            pauli_word = str(
                getattr(term, "pauli", getattr(term, "pauli_exyz", ""))
            )
        nq_builder = getattr(term, "nqubit", None)
        nq = (
            int(nq_builder())
            if callable(nq_builder)
            else int(getattr(term, "nq", len(pauli_word)))
        )
        terms.append(
            {
                "coeff_real": float(coefficient.real),
                "coeff_imag": float(coefficient.imag),
                "nq": nq,
                "pauli": pauli_word,
            }
        )
    return fingerprint_jsonable(
        {
            "label": str(getattr(candidate_term, "label", "")),
            "execution_mode": str(
                getattr(candidate_term, "execution_mode", "termwise_product")
                or "termwise_product"
            )
            .strip()
            .lower(),
            "terms": terms,
        }
    )


def candidate_coordinate_fingerprint(
    candidate_term: Any, *, insertion_position: int
) -> str:
    return str(
        _candidate_coordinate_fingerprint(
            candidate_term,
            position_id=int(insertion_position),
        )
    )


def compiled_hamiltonian_fingerprint(h_compiled: Any) -> str:
    return str(_compiled_polynomial_fingerprint(h_compiled))


def ordered_scaffold_fingerprint(selected_ops: Sequence[Any]) -> str:
    return str(_ordered_scaffold_fingerprint(selected_ops))


__all__ = [
    "candidate_coordinate_fingerprint",
    "candidate_generator_fingerprint",
    "compiled_hamiltonian_fingerprint",
    "fingerprint_jsonable",
    "ordered_scaffold_fingerprint",
]
