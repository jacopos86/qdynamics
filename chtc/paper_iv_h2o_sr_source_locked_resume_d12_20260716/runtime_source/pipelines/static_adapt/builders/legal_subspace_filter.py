"""Legal-subspace filters for truncated binary-boson ADAPT pools.

For binary encodings with unused local codewords (for example ``n_ph_max=2``
embedded in two qubits), many useful encoded boson generators only preserve the
physical subspace through cancellations across Pauli components.  The static
ADAPT gradient path evaluates the grouped infinitesimal generator ``G|psi>``
exactly.  For finite angles, component-risk grouped generators must therefore
be executed as the exact grouped unitary ``exp(-i theta G)`` rather than as a
termwise Pauli-product approximation.  These helpers keep whole grouped/legal
generators and tag component-risk survivors for grouped-exact execution; they
drop generators whose grouped action leaks.  Individual Pauli component leakage
is reported as risk telemetry, not used by itself as a deletion criterion.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.pauli_actions import apply_exp_term, compile_pauli_action_exyz
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

LEGAL_SUBSPACE_FILTER_SCHEMA = "binary_boson_legal_subspace_pool_filter_v1"
LEGAL_SUBSPACE_FILTER_METHOD = "grouped_exact_execution_encoded_basis_action"
DEFAULT_LEGAL_SUBSPACE_FILTER_TOLERANCE = 1e-10
DEFAULT_LEGAL_SUBSPACE_EXECUTION_TEST_ANGLE = 0.1
LEGAL_SUBSPACE_FILTER_SUPPORTED_FAMILIES = frozenset(
    {"hh", "spin_boson", "bose_hubbard", "harmonic_kerr_chain"}
)


def _boson_code_bits(*, n_ph_max: int, boson_encoding: str) -> tuple[int, ...]:
    d = int(n_ph_max) + 1
    encoding_key = str(boson_encoding).strip().lower()
    if encoding_key == "binary":
        return tuple(int(level) for level in range(d))
    if encoding_key == "unary":
        return tuple(int(1 << level) for level in range(d))
    raise ValueError(f"Unsupported boson encoding for legal-subspace filter: {boson_encoding!r}")


def has_unused_binary_boson_codewords(*, n_ph_max: int, boson_encoding: str) -> bool:
    """Return whether the local boson register has unused binary codewords."""

    if str(boson_encoding).strip().lower() != "binary":
        return False
    d = int(n_ph_max) + 1
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    return bool(d < (1 << int(qpb)))


def boson_legal_register_indices(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
) -> tuple[int, ...]:
    """Return legal basis indices for a compact boson-only register."""

    n_sites = int(num_sites)
    if n_sites < 1:
        raise ValueError("num_sites must be positive for legal-subspace filter")
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    code_bits = _boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    indices: list[int] = []
    for levels in np.ndindex(*([len(code_bits)] * n_sites)):
        basis_index = 0
        for site, level in enumerate(levels):
            basis_index |= int(code_bits[int(level)]) << int(site * qpb)
        indices.append(int(basis_index))
    return tuple(sorted(indices))


def legal_subspace_basis_for_problem(
    *,
    problem_key: str,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    total_register_width: int,
) -> dict[str, Any]:
    """Return full-register legal indices and public layout metadata."""

    problem = str(problem_key).strip().lower()
    n_sites = int(num_sites)
    if problem in {"bose_hubbard", "harmonic_kerr_chain"}:
        boson_site_count = n_sites
        non_boson_register_width = 0
        legal_subspace_scope = "boson_codewords_only"
    elif problem == "hh":
        boson_site_count = n_sites
        non_boson_register_width = 2 * n_sites
        legal_subspace_scope = "boson_codewords_with_full_fermion_register"
    elif problem == "spin_boson":
        boson_site_count = n_sites
        non_boson_register_width = 2
        legal_subspace_scope = "boson_codewords_with_full_emitter_register"
    else:
        raise ValueError(f"Unsupported problem for legal-subspace filter: {problem!r}")

    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    boson_register_width = int(boson_site_count) * int(qpb)
    expected_width = int(non_boson_register_width) + int(boson_register_width)
    if int(total_register_width) != expected_width:
        raise ValueError(
            "register_width_incompatible_with_legal_layout:"
            f"{int(total_register_width)}!={expected_width}"
        )

    legal_boson_indices = boson_legal_register_indices(
        num_sites=int(boson_site_count),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )
    non_boson_dim = 1 << int(non_boson_register_width)
    legal_indices: list[int] = []
    for boson_index in legal_boson_indices:
        shifted_boson = int(boson_index) << int(non_boson_register_width)
        for non_boson_index in range(non_boson_dim):
            legal_indices.append(int(shifted_boson | int(non_boson_index)))
    legal_indices_tuple = tuple(sorted(legal_indices))
    full_dim = 1 << int(total_register_width)
    return {
        "legal_subspace_scope": legal_subspace_scope,
        "total_register_width": int(total_register_width),
        "full_hilbert_dim": int(full_dim),
        "non_boson_register_width": int(non_boson_register_width),
        "non_boson_register_dim": int(non_boson_dim),
        "boson_site_count": int(boson_site_count),
        "bits_per_boson_site": int(qpb),
        "boson_register_width": int(boson_register_width),
        "boson_legal_codeword_count": int(len(legal_boson_indices)),
        "legal_indices": legal_indices_tuple,
        "legal_state_count": int(len(legal_indices_tuple)),
        "legal_subspace_dim": int(len(legal_indices_tuple)),
        "illegal_state_count": int(full_dim - len(legal_indices_tuple)),
    }


def pauli_action_on_basis_index(label: str, basis_index: int) -> tuple[int, complex]:
    """Apply an exyz Pauli word to a computational basis index."""

    nq = len(str(label))
    out_index = int(basis_index)
    phase = 1.0 + 0.0j
    for q in range(nq):
        op = str(label)[nq - 1 - q]
        bit = (int(basis_index) >> int(q)) & 1
        sign = 1 if bit == 0 else -1
        if op == "e":
            continue
        if op == "x":
            out_index ^= 1 << int(q)
            continue
        if op == "y":
            out_index ^= 1 << int(q)
            phase *= 1j * sign
            continue
        if op == "z":
            phase *= sign
            continue
        raise ValueError(f"Unsupported Pauli symbol {op!r} in {label!r}")
    return int(out_index), complex(phase)


def pauli_word_illegal_hit_count(
    label: str,
    *,
    legal_indices: Sequence[int],
    legal_set: set[int],
) -> int:
    """Count legal basis states sent outside the legal set by one Pauli word."""

    hits = 0
    for basis_index in legal_indices:
        out_index, _phase = pauli_action_on_basis_index(str(label), int(basis_index))
        if int(out_index) not in legal_set:
            hits += 1
    return int(hits)


def _infer_total_register_width(pool: Sequence[AnsatzTerm]) -> int | None:
    for ansatz_term in pool:
        poly = getattr(ansatz_term, "polynomial", None)
        if poly is None or not hasattr(poly, "return_polynomial"):
            continue
        for term in poly.return_polynomial():
            return int(term.nqubit())
    return None


def _pool_label_class(problem: str, label: str) -> str:
    problem_key = str(problem).strip().lower()
    label_text = str(label)
    base = label_text
    for prefix in ("full_meta::", "ham_quad::", "ham_block::", "hva_term::"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if problem_key == "hh" and ":" in base:
        base = base.split(":")[-1]
    if "(" in base:
        return base.split("(", 1)[0] or "unknown"
    parts = [part for part in base.split("_") if part != ""]
    while parts and (parts[-1].isdigit() or parts[-1] in {"left", "right"}):
        parts.pop()
    return "_".join(parts) if parts else (base or "unknown")


def _top_class_counts(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for detail in details:
        cls = str(detail.get("class") or "unknown")
        counts[cls] += 1
    return [
        {"class": cls, "count": int(count)}
        for cls, count in counts.most_common(10)
    ]


def _inactive_meta(
    *,
    reason: str,
    problem_key: str,
    n_ph_max: int,
    boson_encoding: str,
    original_pool_size: int,
    total_register_width: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": LEGAL_SUBSPACE_FILTER_SCHEMA,
        "active": False,
        "reason": str(reason),
        "problem": str(problem_key),
        "n_ph_max": int(n_ph_max),
        "boson_encoding": str(boson_encoding),
        "method": LEGAL_SUBSPACE_FILTER_METHOD,
        "filter_mode": "grouped_execution_generator_filter",
        "projection_mode": "not_applied",
        "projection_feasible_with_existing_utilities": False,
        "projection_tradeoff": (
            "component projection is not applied; component-risk grouped generators are "
            "retained only when their grouped action is legal and are tagged for grouped_exact execution"
        ),
        "total_register_width": None if total_register_width is None else int(total_register_width),
        "original_pool_size": int(original_pool_size),
        "filtered_pool_size": int(original_pool_size),
        "pre_dedup_filtered_pool_size": int(original_pool_size),
        "kept_generator_count": int(original_pool_size),
        "grouped_legal_count": int(original_pool_size),
        "grouped_legal_generator_count": int(original_pool_size),
        "grouped_leaking_generator_count": 0,
        "termwise_component_risk_count": 0,
        "termwise_component_risk_generator_count": 0,
        "execution_legal_count": int(original_pool_size),
        "execution_legal_generator_count": int(original_pool_size),
        "execution_leaking_generator_count": 0,
        "kept_with_component_risk_count": 0,
        "grouped_exact_execution_generator_count": 0,
        "termwise_product_execution_generator_count": int(original_pool_size),
        "sanitization_mode": "none",
        "legal_preserving_generator_count": int(original_pool_size),
        "legal_leaking_generator_count": 0,
        "unknown_generator_count": 0,
        "sanitized_generator_count": 0,
        "dropped_generator_count": 0,
        "filtered_generator_count": 0,
        "post_filter_duplicate_generator_count": 0,
        "post_filter_duplicate_labels": [],
        "projected_generator_count": 0,
        "termwise_component_original_count": None,
        "termwise_component_legal_preserving_count": None,
        "termwise_component_filtered_count": 0,
        "offender_labels": [],
        "offender_classes": [],
    }


def _coefficients_by_label(
    poly: Any,
    *,
    total_register_width: int,
    tolerance: float,
) -> tuple[list[str], dict[str, complex]]:
    if poly is None or not hasattr(poly, "return_polynomial"):
        raise ValueError("generator_missing_pauli_polynomial")
    order: list[str] = []
    coeffs: dict[str, complex] = {}
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        if len(label) != int(total_register_width):
            raise ValueError(f"generator_register_width:{len(label)}!={int(total_register_width)}")
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tolerance):
            continue
        if label not in coeffs:
            order.append(label)
            coeffs[label] = 0.0 + 0.0j
        coeffs[label] += coeff
    coeffs = {
        label: coeff
        for label, coeff in coeffs.items()
        if abs(complex(coeff)) > float(tolerance)
    }
    order = [label for label in order if label in coeffs]
    return order, coeffs


def _generator_grouped_action_stats(
    labels: Sequence[str],
    coeffs: Mapping[str, complex],
    *,
    legal_indices: Sequence[int],
    legal_set: set[int],
    tolerance: float,
) -> dict[str, Any]:
    """Return leakage stats for the grouped infinitesimal action G|b>."""

    illegal_basis_hit_count = 0
    max_illegal_action_norm = 0.0
    for basis_index in legal_indices:
        amplitudes: dict[int, complex] = {}
        for pauli_label in labels:
            coeff = complex(coeffs[str(pauli_label)])
            if abs(coeff) <= float(tolerance):
                continue
            out_index, phase = pauli_action_on_basis_index(str(pauli_label), int(basis_index))
            amplitudes[int(out_index)] = amplitudes.get(int(out_index), 0.0 + 0.0j) + coeff * phase
        illegal_norm_sq = float(
            sum(
                abs(complex(amp)) ** 2
                for out_index, amp in amplitudes.items()
                if int(out_index) not in legal_set
            )
        )
        illegal_norm = float(illegal_norm_sq ** 0.5)
        if illegal_norm > float(tolerance):
            illegal_basis_hit_count += 1
            max_illegal_action_norm = max(max_illegal_action_norm, illegal_norm)
    return {
        "legal_preserving": bool(int(illegal_basis_hit_count) == 0),
        "illegal_basis_hit_count": int(illegal_basis_hit_count),
        "max_illegal_action_norm": float(max_illegal_action_norm),
    }


def _termwise_component_risk_stats(
    labels: Sequence[str],
    *,
    id_label: str,
    legal_indices: Sequence[int],
    legal_set: set[int],
) -> dict[str, Any]:
    """Return component-level legal leakage risk without filtering components."""

    leaking_labels: list[str] = []
    leaking_component_count = 0
    termwise_illegal_basis_hit_count = 0
    runtime_component_count = 0
    for pauli_label in labels:
        if str(pauli_label) == str(id_label):
            continue
        runtime_component_count += 1
        leaked_for_label = pauli_word_illegal_hit_count(
            str(pauli_label),
            legal_indices=legal_indices,
            legal_set=legal_set,
        )
        if int(leaked_for_label) > 0:
            leaking_component_count += 1
            termwise_illegal_basis_hit_count += int(leaked_for_label)
            leaking_labels.append(str(pauli_label))
    return {
        "runtime_component_count": int(runtime_component_count),
        "termwise_component_leaking_term_count": int(leaking_component_count),
        "termwise_component_illegal_basis_hit_count": int(termwise_illegal_basis_hit_count),
        "termwise_component_leaking_labels_sample": leaking_labels[:5],
        "has_component_risk": bool(leaking_component_count > 0),
    }


def _execution_product_action_stats(
    labels: Sequence[str],
    coeffs: Mapping[str, complex],
    *,
    id_label: str,
    legal_indices: Sequence[int],
    illegal_indices: Sequence[int],
    total_register_width: int,
    tolerance: float,
    execution_test_angles: Sequence[float],
) -> dict[str, Any]:
    """Probe the implemented logical-shared product action on legal basis states.

    The native executor sorts active Pauli terms and applies
    ``prod_j exp(-i theta c_j P_j)`` for a shared logical theta.  This probe is a
    local execution certification, not an all-angle proof; the grouped
    infinitesimal action above remains the exact ADAPT-gradient legality test.
    """

    active_labels = [
        str(label)
        for label in sorted(str(label) for label in labels)
        if str(label) != str(id_label) and abs(complex(coeffs[str(label)])) > float(tolerance)
    ]
    if not active_labels:
        return {
            "execution_legal": True,
            "execution_illegal_basis_hit_count": 0,
            "max_execution_illegal_probability": 0.0,
            "execution_test_angles": [float(x) for x in execution_test_angles],
        }

    action_cache = {
        label: compile_pauli_action_exyz(label, int(total_register_width))
        for label in active_labels
    }
    illegal_index_array = np.asarray([int(idx) for idx in illegal_indices], dtype=np.int64)
    max_illegal_probability = 0.0
    illegal_basis_hit_count = 0
    for theta in execution_test_angles:
        theta_f = float(theta)
        for basis_index in legal_indices:
            psi = np.zeros(1 << int(total_register_width), dtype=complex)
            psi[int(basis_index)] = 1.0 + 0.0j
            for label in active_labels:
                coeff = complex(coeffs[str(label)])
                if abs(coeff.imag) > float(tolerance):
                    raise ValueError(f"generator_imaginary_execution_coefficient:{label}:{coeff}")
                psi = apply_exp_term(
                    psi,
                    action_cache[str(label)],
                    coeff=coeff,
                    dt=theta_f,
                    tol=float(tolerance),
                )
            illegal_probability = (
                0.0
                if illegal_index_array.size == 0
                else float(np.sum(np.abs(psi[illegal_index_array]) ** 2).real)
            )
            if illegal_probability > float(tolerance):
                illegal_basis_hit_count += 1
                max_illegal_probability = max(max_illegal_probability, illegal_probability)
    return {
        "execution_legal": bool(int(illegal_basis_hit_count) == 0),
        "execution_illegal_basis_hit_count": int(illegal_basis_hit_count),
        "max_execution_illegal_probability": float(max_illegal_probability),
        "execution_test_angles": [float(x) for x in execution_test_angles],
    }


def _polynomial_signature(poly: Any, *, tolerance: float) -> tuple[tuple[str, float, float], ...]:
    items: list[tuple[str, float, float]] = []
    if poly is None or not hasattr(poly, "return_polynomial"):
        return tuple()
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tolerance):
            continue
        items.append(
            (
                label,
                round(float(coeff.real), 12),
                round(float(coeff.imag), 12),
            )
        )
    items.sort()
    return tuple(items)


def _polynomial_from_coefficients(
    labels: Sequence[str],
    coeffs: Mapping[str, complex],
    *,
    total_register_width: int,
    tolerance: float,
) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for label in labels:
        coeff = complex(coeffs[str(label)])
        if abs(coeff) <= float(tolerance):
            continue
        if abs(coeff.imag) <= float(tolerance):
            coeff_out: float | complex = float(coeff.real)
        else:
            coeff_out = coeff
        poly.add_term(PauliTerm(int(total_register_width), ps=str(label), pc=coeff_out))
    poly._reduce()
    return poly


def sanitize_pool_for_binary_boson_legal_subspace(
    pool: Sequence[AnsatzTerm],
    *,
    problem_key: str,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    total_register_width: int | None = None,
    tolerance: float = DEFAULT_LEGAL_SUBSPACE_FILTER_TOLERANCE,
    execution_test_angle: float = DEFAULT_LEGAL_SUBSPACE_EXECUTION_TEST_ANGLE,
    label_classifier: Callable[[str], str | None] | None = None,
    fail_on_unknown: bool = True,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    """Filter whole generators that leave the legal binary-boson subspace.

    The filter is active only for supported binary-boson families with unused
    computational codewords.  Component-level leakage is retained as telemetry;
    the deletion criterion is grouped/execution leakage of the implemented
    logical generator.
    """

    problem = str(problem_key).strip().lower()
    pool_list = list(pool)
    original_pool_size = int(len(pool_list))
    width = total_register_width if total_register_width is not None else _infer_total_register_width(pool_list)
    if problem not in LEGAL_SUBSPACE_FILTER_SUPPORTED_FAMILIES:
        return pool_list, _inactive_meta(
            reason=f"unsupported_problem:{problem}",
            problem_key=problem,
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            original_pool_size=original_pool_size,
            total_register_width=width,
        )
    if not has_unused_binary_boson_codewords(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)):
        return pool_list, _inactive_meta(
            reason="no_unused_binary_boson_codewords",
            problem_key=problem,
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            original_pool_size=original_pool_size,
            total_register_width=width,
        )
    if width is None:
        return pool_list, _inactive_meta(
            reason="missing_total_register_width",
            problem_key=problem,
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            original_pool_size=original_pool_size,
            total_register_width=None,
        )

    layout = legal_subspace_basis_for_problem(
        problem_key=problem,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        total_register_width=int(width),
    )
    legal_indices = tuple(int(idx) for idx in layout["legal_indices"])
    legal_set = set(legal_indices)
    full_dim = 1 << int(width)
    illegal_indices = tuple(idx for idx in range(full_dim) if int(idx) not in legal_set)
    id_label = "e" * int(width)
    execution_test_angles = (
        float(abs(float(execution_test_angle))),
        -float(abs(float(execution_test_angle))),
    )

    filtered_pool: list[AnsatzTerm] = []
    offender_details: list[dict[str, Any]] = []
    component_risk_details: list[dict[str, Any]] = []
    grouped_legal_generator_count = 0
    grouped_leaking_generator_count = 0
    execution_legal_generator_count = 0
    execution_leaking_generator_count = 0
    unknown_generator_count = 0
    dropped_generator_count = 0
    kept_with_component_risk_count = 0
    grouped_exact_execution_generator_count = 0
    termwise_product_execution_generator_count = 0
    termwise_component_original_count = 0
    termwise_component_legal_preserving_count = 0
    termwise_component_risk_count = 0

    for ansatz_term in pool_list:
        label = str(getattr(ansatz_term, "label", ""))
        cls = None if label_classifier is None else label_classifier(label)
        class_name = str(cls or _pool_label_class(problem, label))
        try:
            order, coeffs = _coefficients_by_label(
                getattr(ansatz_term, "polynomial", None),
                total_register_width=int(width),
                tolerance=float(tolerance),
            )
        except Exception as exc:
            if bool(fail_on_unknown):
                raise ValueError(
                    "legal_subspace_filter_unknown_generator:"
                    f"label={label!r}:class={class_name!r}:"
                    f"{type(exc).__name__}:{exc}"
                ) from exc
            unknown_generator_count += 1
            dropped_generator_count += 1
            execution_leaking_generator_count += 1
            offender_details.append(
                {
                    "label": label,
                    "class": class_name,
                    "action": "dropped_unknown",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "original_runtime_term_count": None,
                    "kept_runtime_term_count": 0,
                    "filtered_runtime_term_count": None,
                    "leaking_pauli_labels_sample": [],
                }
            )
            continue

        grouped_stats = _generator_grouped_action_stats(
            order,
            coeffs,
            legal_indices=legal_indices,
            legal_set=legal_set,
            tolerance=float(tolerance),
        )
        component_stats = _termwise_component_risk_stats(
            order,
            id_label=id_label,
            legal_indices=legal_indices,
            legal_set=legal_set,
        )
        product_execution_stats = _execution_product_action_stats(
            order,
            coeffs,
            id_label=id_label,
            legal_indices=legal_indices,
            illegal_indices=illegal_indices,
            total_register_width=int(width),
            tolerance=float(tolerance),
            execution_test_angles=execution_test_angles,
        )
        original_runtime_count = int(component_stats["runtime_component_count"])
        component_risk_runtime_count = int(
            component_stats["termwise_component_leaking_term_count"]
        )
        kept_runtime_count = int(original_runtime_count)
        termwise_component_original_count += int(original_runtime_count)
        termwise_component_legal_preserving_count += int(
            original_runtime_count - component_risk_runtime_count
        )
        termwise_component_risk_count += int(component_risk_runtime_count)
        has_component_risk = bool(component_stats["has_component_risk"])

        if bool(grouped_stats["legal_preserving"]):
            grouped_legal_generator_count += 1
        else:
            grouped_leaking_generator_count += 1

        execution_mode = "grouped_exact" if (has_component_risk and bool(grouped_stats["legal_preserving"])) else "termwise_product"
        if execution_mode == "grouped_exact":
            execution_stats = {
                "execution_legal": True,
                "execution_illegal_basis_hit_count": 0,
                "max_execution_illegal_probability": 0.0,
                "execution_test_angles": [float(x) for x in execution_test_angles],
            }
        else:
            execution_stats = product_execution_stats
        execution_legal = bool(grouped_stats["legal_preserving"]) and bool(
            execution_stats["execution_legal"]
        )
        detail = {
            "label": label,
            "class": class_name,
            "original_runtime_term_count": int(original_runtime_count),
            "kept_runtime_term_count": int(kept_runtime_count if execution_legal else 0),
            "filtered_runtime_term_count": 0,
            "termwise_component_leaking_term_count": int(component_risk_runtime_count),
            "termwise_component_illegal_basis_hit_count": int(
                component_stats["termwise_component_illegal_basis_hit_count"]
            ),
            "leaking_pauli_labels_sample": list(
                component_stats["termwise_component_leaking_labels_sample"]
            ),
            "grouped_illegal_basis_hit_count": int(
                grouped_stats["illegal_basis_hit_count"]
            ),
            "max_grouped_illegal_action_norm": float(
                grouped_stats["max_illegal_action_norm"]
            ),
            "execution_mode": execution_mode,
            "execution_illegal_basis_hit_count": int(
                execution_stats["execution_illegal_basis_hit_count"]
            ),
            "max_execution_illegal_probability": float(
                execution_stats["max_execution_illegal_probability"]
            ),
            "termwise_product_probe_illegal_basis_hit_count": int(
                product_execution_stats["execution_illegal_basis_hit_count"]
            ),
            "termwise_product_probe_max_illegal_probability": float(
                product_execution_stats["max_execution_illegal_probability"]
            ),
        }
        if has_component_risk:
            component_risk_details.append({**detail, "action": "kept_with_component_risk" if execution_legal else "dropped"})

        if execution_legal:
            execution_legal_generator_count += 1
            if execution_mode == "grouped_exact":
                grouped_exact_execution_generator_count += 1
            else:
                termwise_product_execution_generator_count += 1
            if has_component_risk:
                kept_with_component_risk_count += 1
            if execution_mode == "grouped_exact":
                filtered_pool.append(
                    AnsatzTerm(
                        label=str(getattr(ansatz_term, "label", label)),
                        polynomial=getattr(ansatz_term, "polynomial"),
                        execution_mode="grouped_exact",
                    )
                )
            else:
                filtered_pool.append(ansatz_term)
            continue

        execution_leaking_generator_count += 1
        dropped_generator_count += 1
        offender_details.append(
            {
                **detail,
                "action": "dropped",
            }
        )

    pre_dedup_filtered_pool_size = int(len(filtered_pool))
    deduped_filtered_pool: list[AnsatzTerm] = []
    seen_signatures: set[tuple[tuple[str, float, float], ...]] = set()
    post_filter_duplicate_labels: list[str] = []
    for ansatz_term in filtered_pool:
        sig = _polynomial_signature(ansatz_term.polynomial, tolerance=float(tolerance))
        if sig in seen_signatures:
            post_filter_duplicate_labels.append(str(getattr(ansatz_term, "label", "")))
            continue
        seen_signatures.add(sig)
        deduped_filtered_pool.append(ansatz_term)
    filtered_pool = deduped_filtered_pool
    post_filter_duplicate_generator_count = int(
        pre_dedup_filtered_pool_size - len(deduped_filtered_pool)
    )

    public_layout = {key: value for key, value in layout.items() if key != "legal_indices"}
    filtered_generator_count = int(dropped_generator_count)
    meta = {
        "schema": LEGAL_SUBSPACE_FILTER_SCHEMA,
        "active": True,
        "reason": "unused_binary_boson_codewords",
        "problem": problem,
        "n_ph_max": int(n_ph_max),
        "boson_encoding": str(boson_encoding),
        "method": LEGAL_SUBSPACE_FILTER_METHOD,
        "filter_mode": "grouped_execution_generator_filter",
        "projection_mode": "not_applied",
        "projection_feasible_with_existing_utilities": False,
        "projection_tradeoff": (
            "component projection is not applied; component-risk grouped generators are "
            "kept as whole generators and tagged for grouped_exact execution so finite angles "
            "use exp(-i theta G) instead of a leaking termwise product"
        ),
        "tolerance": float(tolerance),
        "execution_test_angle": float(abs(float(execution_test_angle))),
        "execution_test_angles": [float(x) for x in execution_test_angles],
        **public_layout,
        "original_pool_size": int(original_pool_size),
        "filtered_pool_size": int(len(filtered_pool)),
        "pre_dedup_filtered_pool_size": int(pre_dedup_filtered_pool_size),
        "kept_generator_count": int(len(filtered_pool)),
        "grouped_legal_count": int(grouped_legal_generator_count),
        "grouped_legal_generator_count": int(grouped_legal_generator_count),
        "grouped_leaking_generator_count": int(grouped_leaking_generator_count),
        "termwise_component_risk_count": int(termwise_component_risk_count),
        "termwise_component_risk_generator_count": int(len(component_risk_details)),
        "execution_legal_count": int(execution_legal_generator_count),
        "execution_legal_generator_count": int(execution_legal_generator_count),
        "execution_leaking_generator_count": int(execution_leaking_generator_count),
        "kept_with_component_risk_count": int(kept_with_component_risk_count),
        "grouped_exact_execution_generator_count": int(grouped_exact_execution_generator_count),
        "termwise_product_execution_generator_count": int(termwise_product_execution_generator_count),
        "legal_preserving_generator_count": int(execution_legal_generator_count),
        "legal_leaking_generator_count": int(execution_leaking_generator_count),
        "unknown_generator_count": int(unknown_generator_count),
        "sanitized_generator_count": int(grouped_exact_execution_generator_count),
        "sanitization_mode": "grouped_exact_execution_for_component_risk",
        "dropped_generator_count": int(dropped_generator_count),
        "filtered_generator_count": int(filtered_generator_count),
        "post_filter_duplicate_generator_count": int(post_filter_duplicate_generator_count),
        "post_filter_duplicate_labels": post_filter_duplicate_labels[:20],
        "projected_generator_count": 0,
        "termwise_component_original_count": int(termwise_component_original_count),
        "termwise_component_legal_preserving_count": int(termwise_component_legal_preserving_count),
        "termwise_component_filtered_count": 0,
        "offender_labels": offender_details[:20],
        "offender_classes": _top_class_counts(offender_details),
        "component_risk_labels": component_risk_details[:20],
        "component_risk_classes": _top_class_counts(component_risk_details),
    }
    return filtered_pool, meta


__all__ = [
    "DEFAULT_LEGAL_SUBSPACE_FILTER_TOLERANCE",
    "DEFAULT_LEGAL_SUBSPACE_EXECUTION_TEST_ANGLE",
    "LEGAL_SUBSPACE_FILTER_METHOD",
    "LEGAL_SUBSPACE_FILTER_SCHEMA",
    "LEGAL_SUBSPACE_FILTER_SUPPORTED_FAMILIES",
    "boson_legal_register_indices",
    "has_unused_binary_boson_codewords",
    "legal_subspace_basis_for_problem",
    "pauli_action_on_basis_index",
    "pauli_word_illegal_hit_count",
    "sanitize_pool_for_binary_boson_legal_subspace",
]
