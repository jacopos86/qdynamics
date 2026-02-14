#!/usr/bin/env python3
"""Standalone integration smoke test for core pydephasing primitives."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np


def _add_repo_paths() -> Path:
    """Add repo root and available pydephasing-containing roots to sys.path."""
    repo_root = Path(__file__).resolve().parent
    candidate_roots = [
        repo_root,
        repo_root / "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE",
    ]

    for candidate in candidate_roots:
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    return repo_root


def _run_import_check(
    label: str,
    importer: Callable[[], Dict[str, object]],
) -> Tuple[bool, Dict[str, object]]:
    try:
        symbols = importer()
        print(f"PASS: {label}")
        return True, symbols
    except Exception as exc:  # pragma: no cover - integration failure path
        print(f"FAIL: {label} ({exc})")
        return False, {}


def _import_pauli_letters() -> Dict[str, object]:
    from pydephasing.quantum.pauli_letters_module import PauliLetter, symbol_product_map

    return {
        "PauliLetter": PauliLetter,
        "symbol_product_map": symbol_product_map,
    }


def _import_pauli_term() -> Dict[str, object]:
    from pydephasing.quantum.pauli_words import PauliTerm

    return {"PauliTerm": PauliTerm}


def _import_pauli_polynomial() -> Dict[str, object]:
    from pydephasing.quantum.pauli_polynomial_class import (
        PauliPolynomial,
        fermion_minus_operator,
        fermion_plus_operator,
    )

    return {
        "PauliPolynomial": PauliPolynomial,
        "fermion_plus_operator": fermion_plus_operator,
        "fermion_minus_operator": fermion_minus_operator,
    }


def _import_hubbard_helpers() -> Dict[str, object]:
    from pydephasing.quantum.hubbard_latex_python_pairs import (
        build_hubbard_hamiltonian,
        jw_number_operator,
        mode_index,
    )

    return {
        "build_hubbard_hamiltonian": build_hubbard_hamiltonian,
        "jw_number_operator": jw_number_operator,
        "mode_index": mode_index,
    }


def _import_hf_helpers() -> Dict[str, object]:
    from pydephasing.quantum.hartree_fock_reference_state import (
        hartree_fock_bitstring,
        hartree_fock_statevector,
    )

    return {
        "hartree_fock_bitstring": hartree_fock_bitstring,
        "hartree_fock_statevector": hartree_fock_statevector,
    }


def main() -> int:
    _add_repo_paths()

    import_results: Dict[str, object] = {}
    all_ok = True

    checks = [
        (
            "Import PauliLetter, symbol_product_map",
            _import_pauli_letters,
        ),
        (
            "Import PauliTerm",
            _import_pauli_term,
        ),
        (
            "Import PauliPolynomial, fermion_plus_operator, fermion_minus_operator",
            _import_pauli_polynomial,
        ),
        (
            "Import build_hubbard_hamiltonian, jw_number_operator, mode_index",
            _import_hubbard_helpers,
        ),
        (
            "Import hartree_fock_bitstring, hartree_fock_statevector",
            _import_hf_helpers,
        ),
    ]

    for label, importer in checks:
        ok, imported = _run_import_check(label, importer)
        all_ok = all_ok and ok
        import_results.update(imported)

    smoke_ok = False
    if all_ok:
        try:
            PauliPolynomial = import_results["PauliPolynomial"]
            build_hubbard_hamiltonian = import_results["build_hubbard_hamiltonian"]

            hamiltonian = build_hubbard_hamiltonian(
                dims=2,
                t=1.0,
                U=4.0,
                pbc=True,  # periodic
                indexing="blocked",
            )

            is_poly = isinstance(hamiltonian, PauliPolynomial)
            term_count = int(hamiltonian.count_number_terms()) if is_poly else 0
            smoke_ok = bool(is_poly and term_count > 0)

            if smoke_ok:
                print(f"PASS: L=2 Hubbard smoke test (terms={term_count})")
            else:
                print(
                    "FAIL: L=2 Hubbard smoke test "
                    f"(is_pauli_polynomial={is_poly}, terms={term_count})"
                )
        except Exception as exc:  # pragma: no cover - integration failure path
            print(f"FAIL: L=2 Hubbard smoke test ({exc})")
            smoke_ok = False
    else:
        print("FAIL: L=2 Hubbard smoke test (skipped because imports failed)")

    # Numpy dependency sanity check to keep this script dependency-minimal and explicit.
    _ = np.__version__

    return 0 if (all_ok and smoke_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
