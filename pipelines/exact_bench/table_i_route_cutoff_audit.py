#!/usr/bin/env python3
"""Dry-run route-identity and cutoff/profile audit for Table-I static cases.

This module is diagnostic support only.  It does not run ADAPT/VQE
optimization; it materializes the Table-I benchmark specs, the static Phase-3
policy metadata that would be emitted for a row, and cheap physical-problem
accounting such as Hamiltonian register width, optional pool size, and optional
same/reference-cutoff exact energies.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.generic_static_benchmark import _phase3_policy_for_algorithm
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_NPH2_REF3_PROFILE,
    table_i_canonical_specs,
    table_i_suite_profile,
)
from pipelines.static_adapt.builders.problem_setup import (
    _exact_gs_energy_for_problem,
    build_problem_hamiltonian,
)
from pipelines.static_adapt.historical_route_identity import (
    ROUTE_ID_A,
    ROUTE_ID_UNSPECIFIED,
    read_historical_route_identity,
)
from pipelines.exact_bench.static_benchmark_runtime import (
    HamiltonianBenchmarkSpec,
    policy_to_cli_args,
)
from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site

AUDIT_SCHEMA = "table_i_route_cutoff_audit_v1"
LEGAL_SUBSPACE_AUDIT_SCHEMA = "nph_legal_subspace_pool_audit_v1"
DEFAULT_ALGORITHM_ID = "static_family_native_adapt_phase3"
DEFAULT_POOL_KEY = "full_meta"
DEFAULT_LEGAL_SUBSPACE_TOLERANCE = 1e-10

_NPH2_REF3_WORKING_FAMILIES = frozenset(
    {"hh", "spin_boson", "bose_hubbard", "harmonic_kerr_chain"}
)
_NPH2_REF3_NPH1_CONTRACT_FAMILIES = frozenset({"molecular_vibronic_h2"})
_LEGAL_SUBSPACE_SUPPORTED_FAMILIES = frozenset(
    {"hh", "spin_boson", "bose_hubbard", "harmonic_kerr_chain"}
)
_DEFAULT_RESULT_ROOTS = {
    "standard": Path("raw_outputs/generic_static_table"),
    TABLE_I_NPH2_REF3_PROFILE: Path("raw_outputs/generic_static_table_nph2_ref3_v1"),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _parse_cli_option_map(args: Sequence[Any]) -> dict[str, Any]:
    tokens = [str(token) for token in args]
    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        if "=" in token:
            key, value = token[2:].split("=", 1)
            out[key.replace("-", "_")] = value
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = tokens[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _option_text(options: Mapping[str, Any], key: str, default: str | None = None) -> str | None:
    value = options.get(key, default)
    if value is None or value == "":
        return default
    return str(value)


def _option_int(options: Mapping[str, Any], key: str, default: int | None = None) -> int | None:
    value = options.get(key, default)
    if value is None or value == "":
        return default
    return int(float(str(value)))


def _option_float(options: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    value = options.get(key, default)
    if value is None or value == "":
        return default
    return float(str(value))


def _option_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _spec_options(spec: HamiltonianBenchmarkSpec) -> dict[str, Any]:
    return _parse_cli_option_map(spec.base_pipeline_args)


def _policy_options_for_spec(
    *,
    spec: HamiltonianBenchmarkSpec,
    algorithm_id: str,
    pool_key: str,
) -> tuple[dict[str, Any], Any]:
    policy = _phase3_policy_for_algorithm(
        algorithm_id=str(algorithm_id),
        pool_key=str(pool_key),
    )
    return _parse_cli_option_map(policy_to_cli_args(policy, spec)), policy


def _fallback_if_none(value: Any, default: Any) -> Any:
    return default if value is None else value


def _problem_params_from_spec(spec: HamiltonianBenchmarkSpec) -> dict[str, Any]:
    options = _spec_options(spec)
    L = _fallback_if_none(_option_int(options, "L", int(spec.features.L)), int(spec.features.L))
    return {
        "problem_key": _fallback_if_none(_option_text(options, "problem", str(spec.family)), str(spec.family)),
        "num_sites": int(L),
        "t": _fallback_if_none(_option_float(options, "t", 1.0), 1.0),
        "u": _fallback_if_none(_option_float(options, "u", 4.0), 4.0),
        "dv": _fallback_if_none(_option_float(options, "dv", 0.0), 0.0),
        "omega0": _fallback_if_none(_option_float(options, "omega0", 1.0), 1.0),
        "g_ep": _fallback_if_none(_option_float(options, "g_ep", 0.5), 0.5),
        "n_ph_max": _fallback_if_none(_option_int(options, "n_ph_max", 1), 1),
        "boson_encoding": _fallback_if_none(_option_text(options, "boson_encoding", "binary"), "binary"),
        "ordering": _fallback_if_none(_option_text(options, "ordering", "blocked"), "blocked"),
        "boundary": _fallback_if_none(_option_text(options, "boundary", "open"), "open"),
        "v_nn": _fallback_if_none(_option_float(options, "v_nn", 0.0), 0.0),
        "t_prime": _fallback_if_none(_option_float(options, "t_prime", 0.0), 0.0),
        "include_zero_point": True,
    }


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    half = max(1, int(num_sites) // 2)
    return (half, half)


def _pauli_widths(poly: Any) -> tuple[int, ...]:
    widths: set[int] = set()
    if poly is None or not hasattr(poly, "return_polynomial"):
        return ()
    for term in poly.return_polynomial():
        try:
            widths.add(len(str(term.pw2strng())))
        except Exception:
            continue
    return tuple(sorted(widths))


def _hamiltonian_metadata(params: Mapping[str, Any]) -> tuple[Any | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        h_poly = build_problem_hamiltonian(**dict(params))
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return None, {
            "hamiltonian_term_count": None,
            "hamiltonian_register_width": None,
            "hamiltonian_register_widths": [],
            "full_hilbert_dim": None,
        }, [f"hamiltonian_build_failed:{type(exc).__name__}:{exc}"]
    terms = h_poly.return_polynomial() if hasattr(h_poly, "return_polynomial") else []
    widths = _pauli_widths(h_poly)
    width = widths[0] if len(widths) == 1 else None
    if len(widths) > 1:
        warnings.append(f"hamiltonian_mixed_register_widths:{list(widths)}")
    return h_poly, {
        "hamiltonian_term_count": int(len(terms)),
        "hamiltonian_register_width": width,
        "hamiltonian_register_widths": list(widths),
        "full_hilbert_dim": None if width is None else int(1 << int(width)),
    }, warnings


def _exact_energy_for_params(params: Mapping[str, Any], *, h_poly: Any | None = None, n_ph_max: int | None = None) -> float:
    effective = dict(params)
    if n_ph_max is not None:
        effective["n_ph_max"] = int(n_ph_max)
        h_poly = None
    if h_poly is None:
        h_poly = build_problem_hamiltonian(**effective)
    num_sites = int(effective["num_sites"])
    return float(
        _exact_gs_energy_for_problem(
            h_poly,
            problem=str(effective["problem_key"]),
            num_sites=num_sites,
            num_particles=_half_filled_particles(num_sites),
            indexing=str(effective["ordering"]),
            n_ph_max=int(effective["n_ph_max"]),
            boson_encoding=str(effective["boson_encoding"]),
            t=float(effective["t"]),
            u=float(effective["u"]),
            dv=float(effective["dv"]),
            v_nn=float(effective.get("v_nn", 0.0)),
            t_prime=float(effective.get("t_prime", 0.0)),
            omega0=float(effective["omega0"]),
            g_ep=float(effective["g_ep"]),
            boundary=str(effective["boundary"]),
            include_zero_point=bool(effective.get("include_zero_point", True)),
        )
    )


def _materialize_pool_terms(
    *,
    h_poly: Any,
    params: Mapping[str, Any],
    pool_key: str,
) -> tuple[list[Any] | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    problem = str(params["problem_key"]).strip().lower()
    num_sites = int(params["num_sites"])
    try:
        if problem == "hh":
            from pipelines.static_adapt.builders.hh_pool_presets import build_hh_pool_by_key

            pool, resolved_pool_builder, class_filter_meta, label_filter_meta, legal_filter_meta = build_hh_pool_by_key(
                pool_key_hh=str(pool_key),
                h_poly=h_poly,
                num_sites=num_sites,
                t=float(params["t"]),
                u=float(params["u"]),
                omega0=float(params["omega0"]),
                g_ep=float(params["g_ep"]),
                dv=float(params["dv"]),
                n_ph_max=int(params["n_ph_max"]),
                boson_encoding=str(params["boson_encoding"]),
                ordering=str(params["ordering"]),
                boundary=str(params["boundary"]),
                paop_r=1,
                paop_split_paulis=False,
                paop_prune_eps=0.0,
                paop_normalization="none",
                num_particles=_half_filled_particles(num_sites),
                include_legal_subspace_filter_meta=True,
            )
            pool_meta: Mapping[str, Any] = {
                "resolved_pool_builder": resolved_pool_builder,
                "full_meta_class_filter_meta": class_filter_meta,
                "full_meta_label_filter_meta": label_filter_meta,
                "pool_legal_subspace_filter": legal_filter_meta,
            }
        elif problem == "molecular_vibronic_h2":
            return None, {
                "pool_size": None,
                "pool_register_width": None,
                "pool_register_widths": [],
                "pool_metadata": {},
            }, ["pool_size_not_materialized:molecular_vibronic_h2_fixture_pool"]
        else:
            from pipelines.static_adapt.builders.primitive_pools import build_runtime_pool_terms

            pool, pool_meta = build_runtime_pool_terms(
                pool_key=str(pool_key),
                problem_key=problem,
                h_poly=h_poly,
                num_sites=num_sites,
                num_particles=_half_filled_particles(num_sites),
                t=float(params["t"]),
                u=float(params["u"]),
                dv=float(params["dv"]),
                v_nn=float(params.get("v_nn", 0.0)),
                t_prime=float(params.get("t_prime", 0.0)),
                omega0=float(params["omega0"]),
                g_ep=float(params["g_ep"]),
                n_ph_max=int(params["n_ph_max"]),
                boson_encoding=str(params["boson_encoding"]),
                ordering=str(params["ordering"]),
                boundary=str(params["boundary"]),
                include_zero_point=bool(params.get("include_zero_point", True)),
            )
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return None, {
            "pool_size": None,
            "pool_register_width": None,
            "pool_register_widths": [],
            "pool_metadata": {},
        }, [f"pool_size_unavailable:{type(exc).__name__}:{exc}"]

    widths: set[int] = set()
    for ansatz_term in pool:
        poly = getattr(ansatz_term, "polynomial", None)
        widths.update(_pauli_widths(poly))
    width_tuple = tuple(sorted(widths))
    width = width_tuple[0] if len(width_tuple) == 1 else None
    if len(width_tuple) > 1:
        warnings.append(f"pool_mixed_register_widths:{list(width_tuple)}")
    pool_metadata = dict(pool_meta or {})
    return list(pool), {
        "pool_size": int(len(pool)),
        "pool_register_width": width,
        "pool_register_widths": list(width_tuple),
        "pool_metadata": pool_metadata,
        "pool_legal_subspace_filter": pool_metadata.get("pool_legal_subspace_filter"),
    }, warnings


def _pool_metadata(
    *,
    h_poly: Any,
    params: Mapping[str, Any],
    pool_key: str,
) -> tuple[dict[str, Any], list[str]]:
    _pool, meta, warnings = _materialize_pool_terms(
        h_poly=h_poly,
        params=params,
        pool_key=str(pool_key),
    )
    return meta, warnings


def _boson_code_bits(*, n_ph_max: int, boson_encoding: str) -> tuple[int, ...]:
    d = int(n_ph_max) + 1
    encoding_key = str(boson_encoding).strip().lower()
    if encoding_key == "binary":
        return tuple(int(level) for level in range(d))
    if encoding_key == "unary":
        return tuple(int(1 << level) for level in range(d))
    raise ValueError(f"Unsupported boson encoding for legal-subspace audit: {boson_encoding!r}")


def _boson_legal_register_indices(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
) -> tuple[int, ...]:
    n_sites = int(num_sites)
    if n_sites < 1:
        raise ValueError("num_sites must be positive for legal-subspace audit")
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    code_bits = _boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    indices: list[int] = []
    for levels in np.ndindex(*([len(code_bits)] * n_sites)):
        basis_index = 0
        for site, level in enumerate(levels):
            basis_index |= int(code_bits[int(level)]) << int(site * qpb)
        indices.append(int(basis_index))
    return tuple(sorted(indices))


def _legal_subspace_basis_for_params(
    params: Mapping[str, Any],
    *,
    total_register_width: int,
) -> dict[str, Any]:
    problem = str(params["problem_key"]).strip().lower()
    n_sites = int(params["num_sites"])
    n_ph_max = int(params["n_ph_max"])
    boson_encoding = str(params["boson_encoding"])
    if problem in {"bose_hubbard", "harmonic_kerr_chain"}:
        boson_site_count = n_sites
        non_boson_register_width = 0
        legal_subspace_scope = "boson_codewords_only"
    elif problem == "hh":
        boson_site_count = n_sites
        non_boson_register_width = 2 * n_sites
        legal_subspace_scope = "boson_codewords_with_full_fermion_register"
    elif problem == "spin_boson":
        boson_site_count = 1
        non_boson_register_width = 2
        legal_subspace_scope = "boson_codewords_with_full_emitter_register"
    else:
        raise ValueError(f"Unsupported problem for nph legal-subspace audit: {problem!r}")

    qpb = int(boson_qubits_per_site(n_ph_max, boson_encoding))
    boson_register_width = int(boson_site_count) * int(qpb)
    expected_width = int(non_boson_register_width) + int(boson_register_width)
    if int(total_register_width) != expected_width:
        raise ValueError(
            "register_width_incompatible_with_legal_layout:"
            f"{int(total_register_width)}!={expected_width}"
        )

    legal_boson_indices = _boson_legal_register_indices(
        num_sites=int(boson_site_count),
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
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


def _pauli_action_on_basis_index(label: str, basis_index: int) -> tuple[int, complex]:
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


def _polynomial_coefficients_by_label(
    poly: Any,
    *,
    total_register_width: int,
    tolerance: float,
) -> dict[str, complex]:
    if poly is None or not hasattr(poly, "return_polynomial"):
        raise ValueError("generator_missing_pauli_polynomial")
    coeffs: dict[str, complex] = {}
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        if len(label) != int(total_register_width):
            raise ValueError(
                f"generator_register_width:{len(label)}!={int(total_register_width)}"
            )
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tolerance):
            continue
        coeffs[label] = coeffs.get(label, 0.0 + 0.0j) + coeff
    return {
        label: coeff
        for label, coeff in coeffs.items()
        if abs(complex(coeff)) > float(tolerance)
    }


def _pool_label_class(problem: str, label: str) -> str:
    problem_key = str(problem).strip().lower()
    label_text = str(label)
    if problem_key == "hh":
        try:
            from pipelines.static_adapt.builders.hh_pool_presets import _classify_hh_full_meta_label

            classified = _classify_hh_full_meta_label(label_text)
            if classified:
                return str(classified)
        except Exception:
            pass
    base = label_text
    for prefix in ("full_meta::", "ham_quad::", "ham_block::", "hva_term::"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if "(" in base:
        return base.split("(", 1)[0] or "unknown"
    parts = [part for part in base.split("_") if part != ""]
    while parts and (parts[-1].isdigit() or parts[-1] in {"left", "right"}):
        parts.pop()
    return "_".join(parts) if parts else (base or "unknown")


def _generator_legal_action_stats(
    ansatz_term: Any,
    *,
    legal_indices: Sequence[int],
    legal_set: set[int],
    total_register_width: int,
    tolerance: float,
) -> dict[str, Any]:
    label = str(getattr(ansatz_term, "label", ""))
    try:
        coeffs = _polynomial_coefficients_by_label(
            getattr(ansatz_term, "polynomial", None),
            total_register_width=int(total_register_width),
            tolerance=float(tolerance),
        )
        active_items = tuple(sorted(coeffs.items()))
        illegal_basis_hit_count = 0
        max_illegal_action_norm = 0.0
        for basis_index in legal_indices:
            amplitudes: dict[int, complex] = {}
            for pauli_label, coeff in active_items:
                out_index, phase = _pauli_action_on_basis_index(pauli_label, int(basis_index))
                amplitudes[out_index] = amplitudes.get(out_index, 0.0 + 0.0j) + complex(coeff) * phase
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

        termwise_leaking_labels: list[str] = []
        termwise_illegal_basis_hit_count = 0
        id_label = "e" * int(total_register_width)
        for pauli_label, coeff in active_items:
            if pauli_label == id_label or abs(complex(coeff)) <= float(tolerance):
                continue
            leaked_for_label = 0
            for basis_index in legal_indices:
                out_index, _phase = _pauli_action_on_basis_index(pauli_label, int(basis_index))
                if int(out_index) not in legal_set:
                    leaked_for_label += 1
            if leaked_for_label > 0:
                termwise_leaking_labels.append(str(pauli_label))
                termwise_illegal_basis_hit_count += int(leaked_for_label)

        status = (
            "legal_leaking"
            if int(illegal_basis_hit_count) > 0 and max_illegal_action_norm > float(tolerance)
            else "legal_preserving"
        )
        return {
            "label": label,
            "status": status,
            "active_pauli_term_count": int(len(active_items)),
            "max_illegal_action_norm": float(max_illegal_action_norm),
            "illegal_basis_hit_count": int(illegal_basis_hit_count),
            "termwise_component_leaking_term_count": int(len(termwise_leaking_labels)),
            "termwise_component_illegal_basis_hit_count": int(termwise_illegal_basis_hit_count),
            "termwise_component_leaking_labels_sample": termwise_leaking_labels[:5],
        }
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return {
            "label": label,
            "status": "unknown",
            "reason": f"{type(exc).__name__}:{exc}",
            "active_pauli_term_count": None,
            "max_illegal_action_norm": None,
            "illegal_basis_hit_count": None,
            "termwise_component_leaking_term_count": None,
            "termwise_component_illegal_basis_hit_count": None,
            "termwise_component_leaking_labels_sample": [],
        }


def _top_class_counts(details: Sequence[Mapping[str, Any]], *, count_key: str | None = None) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for detail in details:
        cls = str(detail.get("class") or "unknown")
        increment = 1 if count_key is None else int(detail.get(count_key) or 0)
        if increment <= 0:
            continue
        counts[cls] += increment
    return [
        {"class": cls, "count": int(count)}
        for cls, count in counts.most_common(10)
    ]


def build_pool_legal_subspace_audit(
    *,
    pool_terms: Sequence[Any] | None,
    params: Mapping[str, Any],
    total_register_width: int | None,
    pool_size: int | None,
    tolerance: float = DEFAULT_LEGAL_SUBSPACE_TOLERANCE,
) -> dict[str, Any]:
    """Audit whether grouped generators preserve the encoded legal boson subspace.

    The main test is exact algebraic infinitesimal action: a generator is
    legal-preserving when ``G |b>`` has no support on illegal computational
    basis states for every legal computational basis state ``|b>``.  A separate
    termwise-component risk field is reported because current compiled runtime
    paths apply products of individual Pauli rotations; it is not counted as a
    grouped-generator leak, but it is runtime-relevant for termwise execution.
    """

    problem = str(params.get("problem_key", "")).strip().lower()
    base: dict[str, Any] = {
        "schema": LEGAL_SUBSPACE_AUDIT_SCHEMA,
        "status": "unavailable",
        "problem": problem,
        "n_ph_max": int(params.get("n_ph_max", 0)),
        "boson_encoding": str(params.get("boson_encoding", "binary")),
        "pool_size": None if pool_size is None else int(pool_size),
        "method": "grouped_infinitesimal_exact_encoded_basis_action",
        "method_strength": "exact_algebraic",
        "tolerance": float(tolerance),
        "weaker_test": False,
        "generators_tested": 0,
        "number_generators_tested": 0,
        "legal_preserving_count": 0,
        "legal_leaking_count": 0,
        "unknown_count": 0,
        "top_offending_labels": [],
        "top_offending_classes": [],
        "termwise_component_leak_risk_count": 0,
        "termwise_component_leak_risk_top_labels": [],
        "termwise_component_leak_risk_top_classes": [],
        "termwise_component_risk_interpretation": (
            "component-level risk if Pauli terms were independently admitted; grouped infinitesimal "
            "action is audited here and local logical-shared product execution is certified by "
            "pool_legal_subspace_filter when present"
        ),
    }
    if problem == "molecular_vibronic_h2":
        base.update(
            {
                "status": "not_applicable",
                "reason": "molecular_vibronic_h2_nph1_only_contract",
                "method_strength": "not_applicable",
            }
        )
        return base
    if problem not in _LEGAL_SUBSPACE_SUPPORTED_FAMILIES:
        base.update(
            {
                "status": "not_applicable",
                "reason": f"unsupported_family:{problem}",
                "method_strength": "not_applicable",
            }
        )
        return base
    if total_register_width is None:
        base["reason"] = "missing_total_register_width"
        return base
    if pool_terms is None:
        base["reason"] = "pool_terms_unavailable"
        return base

    try:
        layout = _legal_subspace_basis_for_params(
            params,
            total_register_width=int(total_register_width),
        )
    except Exception as exc:
        base["reason"] = f"legal_subspace_layout_unavailable:{type(exc).__name__}:{exc}"
        return base

    legal_indices = tuple(int(idx) for idx in layout["legal_indices"])
    legal_set = set(legal_indices)
    details: list[dict[str, Any]] = []
    leaking_details: list[dict[str, Any]] = []
    termwise_risk_details: list[dict[str, Any]] = []
    legal_preserving_count = 0
    legal_leaking_count = 0
    unknown_count = 0
    for ansatz_term in pool_terms:
        detail = _generator_legal_action_stats(
            ansatz_term,
            legal_indices=legal_indices,
            legal_set=legal_set,
            total_register_width=int(total_register_width),
            tolerance=float(tolerance),
        )
        detail["class"] = _pool_label_class(problem, str(detail.get("label", "")))
        details.append(detail)
        status = str(detail.get("status"))
        if status == "legal_preserving":
            legal_preserving_count += 1
        elif status == "legal_leaking":
            legal_leaking_count += 1
            leaking_details.append(detail)
        else:
            unknown_count += 1
        if int(detail.get("termwise_component_leaking_term_count") or 0) > 0:
            termwise_risk_details.append(detail)

    leaking_details_sorted = sorted(
        leaking_details,
        key=lambda item: float(item.get("max_illegal_action_norm") or 0.0),
        reverse=True,
    )
    termwise_risk_sorted = sorted(
        termwise_risk_details,
        key=lambda item: int(item.get("termwise_component_leaking_term_count") or 0),
        reverse=True,
    )
    if legal_leaking_count > 0:
        status = "legal_leaking_generators_found"
        conclusion = "grouped_generators_leak_legal_subspace"
    elif unknown_count > 0:
        status = "incomplete_unknown_generators"
        conclusion = "no_grouped_leaks_observed_but_unknown_generators_remain"
    else:
        status = "all_generators_legal_preserving"
        conclusion = "grouped_generators_certified_legal_preserving"

    public_layout = {key: value for key, value in layout.items() if key != "legal_indices"}
    base.update(
        {
            **public_layout,
            "status": status,
            "conclusion": conclusion,
            "generators_tested": int(len(details)),
            "number_generators_tested": int(len(details)),
            "legal_preserving_count": int(legal_preserving_count),
            "legal_leaking_count": int(legal_leaking_count),
            "unknown_count": int(unknown_count),
            "top_offending_labels": [
                {
                    "label": str(item.get("label", "")),
                    "class": str(item.get("class", "unknown")),
                    "max_illegal_action_norm": item.get("max_illegal_action_norm"),
                    "illegal_basis_hit_count": item.get("illegal_basis_hit_count"),
                }
                for item in leaking_details_sorted[:10]
            ],
            "top_offending_classes": _top_class_counts(leaking_details_sorted),
            "termwise_component_leak_risk_count": int(len(termwise_risk_details)),
            "termwise_component_leak_risk_top_labels": [
                {
                    "label": str(item.get("label", "")),
                    "class": str(item.get("class", "unknown")),
                    "termwise_component_leaking_term_count": int(
                        item.get("termwise_component_leaking_term_count") or 0
                    ),
                    "termwise_component_illegal_basis_hit_count": int(
                        item.get("termwise_component_illegal_basis_hit_count") or 0
                    ),
                    "termwise_component_leaking_labels_sample": list(
                        item.get("termwise_component_leaking_labels_sample") or []
                    ),
                }
                for item in termwise_risk_sorted[:10]
            ],
            "termwise_component_leak_risk_top_classes": _top_class_counts(
                termwise_risk_sorted,
                count_key="termwise_component_leaking_term_count",
            ),
        }
    )
    return base


def _route_identity_class(
    *,
    route_payload: Mapping[str, Any],
    selected_logical_route: str,
    selected_logical_source: str | None,
    working_n_ph_max: int | None,
    algorithm_id: str,
) -> str:
    selected_route = str(selected_logical_route or "standard").strip().lower().replace("-", "_")
    source_text = str(selected_logical_source or "").strip().lower()
    if selected_route == "historical_selected":
        if working_n_ph_max is not None and int(working_n_ph_max) > 1 and "nph1" in source_text:
            return "invalid_mixed_identity"
        return "historical_selected_replay"
    if bool(route_payload.get("canonical_snake_eligible", False)) and str(route_payload.get("route_id")) == ROUTE_ID_A:
        return "canonical_route_a_matched"
    if str(algorithm_id) == DEFAULT_ALGORITHM_ID:
        return "generic_phase3_native"
    return "generic_phase3_native"


def _nph2_profile_contract(
    *,
    profile: str,
    family: str,
) -> dict[str, Any]:
    if table_i_suite_profile(profile) != TABLE_I_NPH2_REF3_PROFILE:
        return {
            "expected_working_n_ph_max": None,
            "expected_reference_n_ph_ref": None,
            "n_ph_contract": "profile_not_nph2_ref3",
        }
    family_key = str(family).strip().lower()
    if family_key in _NPH2_REF3_WORKING_FAMILIES:
        return {
            "expected_working_n_ph_max": 2,
            "expected_reference_n_ph_ref": 3,
            "n_ph_contract": "nph2_work_ref3",
        }
    if family_key in _NPH2_REF3_NPH1_CONTRACT_FAMILIES:
        return {
            "expected_working_n_ph_max": 1,
            "expected_reference_n_ph_ref": None,
            "n_ph_contract": "nph1_by_family_contract",
        }
    return {
        "expected_working_n_ph_max": None,
        "expected_reference_n_ph_ref": None,
        "n_ph_contract": "non_bosonic_profile_member",
    }


def _default_result_path(*, profile: str, family: str, case_id: str, algorithm_id: str) -> Path:
    root = _DEFAULT_RESULT_ROOTS.get(table_i_suite_profile(profile), Path("raw_outputs/generic_static_table"))
    return root / f"static_table__{family}__{case_id}__{algorithm_id}" / "result" / "generic_static_single.json"


def _existing_result_summary(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {"found": False, "path": str(path)}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return {"found": False, "path": str(path), "error": f"{type(exc).__name__}:{exc}"}
    result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
    audit = result.get("policy_roundtrip_audit", {}) if isinstance(result, Mapping) else {}
    runtime_identity = audit.get("runtime_static_route_identity", {}) if isinstance(audit, Mapping) else {}
    emitted = audit.get("emitted_options", {}) if isinstance(audit, Mapping) else {}
    return {
        "found": True,
        "path": str(path),
        "status": payload.get("status") if isinstance(payload, Mapping) else None,
        "abs_delta_e": result.get("abs_delta_e"),
        "abs_delta_e_same_cutoff": result.get("abs_delta_e_same_cutoff"),
        "abs_delta_e_reference": result.get("abs_delta_e_reference"),
        "cutoff_abs_delta_e": result.get("cutoff_abs_delta_e"),
        "exact_reference_n_ph_max": result.get("exact_reference_n_ph_max"),
        "stop_reason": result.get("stop_reason"),
        "failure_reason": result.get("failure_reason"),
        "ansatz_depth": result.get("ansatz_depth"),
        "policy_static_route_id": emitted.get("static_route_id") if isinstance(emitted, Mapping) else None,
        "policy_selected_logical_mode": emitted.get("adapt_selected_logical_mode") if isinstance(emitted, Mapping) else None,
        "runtime_route_id": runtime_identity.get("route_id") if isinstance(runtime_identity, Mapping) else None,
        "runtime_route_valid": runtime_identity.get("valid") if isinstance(runtime_identity, Mapping) else None,
        "runtime_canonical_snake_eligible": (
            runtime_identity.get("canonical_snake_eligible") if isinstance(runtime_identity, Mapping) else None
        ),
    }


def build_route_cutoff_audit_row(
    spec: HamiltonianBenchmarkSpec,
    *,
    profile: str,
    algorithm_id: str = DEFAULT_ALGORITHM_ID,
    pool_key: str = DEFAULT_POOL_KEY,
    include_exact_energies: bool = False,
    include_pool_size: bool = False,
    include_legal_subspace_audit: bool = False,
    legal_subspace_tolerance: float = DEFAULT_LEGAL_SUBSPACE_TOLERANCE,
    attach_existing_result: bool = False,
) -> dict[str, Any]:
    """Build one diagnostic audit row without running an optimizer."""

    profile_key = table_i_suite_profile(profile)
    spec_options = _spec_options(spec)
    policy_options, policy = _policy_options_for_spec(
        spec=spec,
        algorithm_id=str(algorithm_id),
        pool_key=str(pool_key),
    )
    static = policy.static
    pool = policy.pool
    inner = policy.inner_optimizer
    working_n_ph = _option_int(spec_options, "n_ph_max")
    reference_n_ph = spec.exact_reference_n_ph_max
    resolved_pool_key = _option_text(policy_options, "adapt_pool", str(pool.pool_key)) or str(pool.pool_key)
    static_route_id = _option_text(policy_options, "static_route_id", ROUTE_ID_UNSPECIFIED) or ROUTE_ID_UNSPECIFIED
    selected_source = getattr(spec, "selected_logical_source_json", None)
    selected_route = str(getattr(spec, "selected_logical_route", "standard") or "standard")
    selected_mode = _option_text(policy_options, "adapt_selected_logical_mode", "off") or "off"
    optimizer = _option_text(policy_options, "adapt_inner_optimizer", str(inner.inner_optimizer)) or str(inner.inner_optimizer)

    route_observed = {
        "base_pool_key": str(resolved_pool_key),
        "continuation_mode": _option_text(policy_options, "adapt_continuation_mode", "phase3_v1"),
        "phase2_novelty_mode": str(static.phase2_novelty_mode),
        "phase3_selector_policy": str(static.phase3_selector_policy),
        "phase3_selector_geometry_mode": str(static.phase3_selector_geometry_mode),
        "algebraic_shortlisting_enabled": bool(str(static.phase3_selector_policy) == "algebraic_nested_v1"),
        "hardware_resolution_schema": "gradient_resolution_v1",
        "hardware_resolution_mode": _option_text(policy_options, "hardware_resolution_mode", "ideal"),
        "phase2_raw_score_formula": "DeltaE_TR_raw * N2 / (1 + K2)",
        "canonical_score_formula": "DeltaE_TR * N3 / (1 + K3)",
        "primary_selector_score_key": "full_v2_score",
        "auxiliary_terms_primary_mode": "tie_break_only",
        "phase3_novelty_ablation_mode": str(static.phase3_novelty_ablation_mode),
        "phase3_window_relaxation_mode": str(static.phase3_window_relaxation_mode),
    }
    route_payload = read_historical_route_identity(
        route_observed,
        declared_route_id=static_route_id,
        optimizer_lane=optimizer,
    )

    params = _problem_params_from_spec(spec)
    h_poly, ham_meta, build_warnings = _hamiltonian_metadata(params)
    warnings = list(build_warnings)
    mismatches: list[str] = []
    incomplete_validation_fields: list[str] = []
    if h_poly is None:
        incomplete_validation_fields.append("hamiltonian")
        mismatches.append("hamiltonian_build_failed")
    if not bool(route_payload.get("valid", False)):
        mismatches.append(
            "static_route_identity_invalid:"
            + ";".join(str(reason) for reason in route_payload.get("noncanonical_reasons", ()))
        )
    contract = _nph2_profile_contract(profile=profile_key, family=spec.family)
    expected_work = contract["expected_working_n_ph_max"]
    expected_ref = contract["expected_reference_n_ph_ref"]
    if expected_work is not None and working_n_ph != int(expected_work):
        mismatches.append(f"working_n_ph_max:{working_n_ph}!={expected_work}")
    if expected_ref is not None and reference_n_ph != int(expected_ref):
        mismatches.append(f"reference_n_ph_ref:{reference_n_ph}!={expected_ref}")
    if profile_key == TABLE_I_NPH2_REF3_PROFILE and expected_work == 2 and "_nph2" not in str(spec.benchmark_id):
        mismatches.append("nph2_profile_case_id_missing_nph2_suffix")

    feature_nq = int(spec.features.n_qubits)
    h_width = ham_meta.get("hamiltonian_register_width")
    if h_width is not None and int(feature_nq) != int(h_width):
        warnings.append(f"feature_n_qubits_hint_differs_from_hamiltonian_register_width:{feature_nq}!={h_width}")

    pool_meta = {
        "pool_size": None,
        "pool_register_width": None,
        "pool_register_widths": [],
        "pool_metadata": {},
    }
    pool_terms: list[Any] | None = None
    problem_key = str(params["problem_key"]).strip().lower()
    should_materialize_pool = bool(include_pool_size) or (
        bool(include_legal_subspace_audit) and problem_key in _LEGAL_SUBSPACE_SUPPORTED_FAMILIES
    )
    if should_materialize_pool:
        if h_poly is None:
            incomplete_validation_fields.append("pool_size")
        else:
            pool_terms, pool_meta, pool_warnings = _materialize_pool_terms(
                h_poly=h_poly,
                params=params,
                pool_key=str(resolved_pool_key),
            )
            warnings.extend(pool_warnings)
            if pool_meta.get("pool_size") is None:
                incomplete_validation_fields.append("pool_size")
            pool_width = pool_meta.get("pool_register_width")
            if pool_width is not None and h_width is not None and int(pool_width) != int(h_width):
                mismatches.append(f"pool_register_width:{pool_width}!={h_width}")

    legal_subspace_audit = None
    if include_legal_subspace_audit:
        total_width_for_legal = h_width if h_width is not None else pool_meta.get("pool_register_width")
        legal_subspace_audit = build_pool_legal_subspace_audit(
            pool_terms=pool_terms,
            params=params,
            total_register_width=None if total_width_for_legal is None else int(total_width_for_legal),
            pool_size=pool_meta.get("pool_size"),
            tolerance=float(legal_subspace_tolerance),
        )
        if (
            problem_key in _LEGAL_SUBSPACE_SUPPORTED_FAMILIES
            and isinstance(legal_subspace_audit, Mapping)
            and str(legal_subspace_audit.get("status")) == "unavailable"
        ):
            reason = str(legal_subspace_audit.get("reason", "unknown"))
            incomplete_validation_fields.append(f"legal_subspace_audit:{reason}")

    exact_same = None
    exact_ref = None
    cutoff_gap = None
    if include_exact_energies:
        if h_poly is None:
            incomplete_validation_fields.append("exact_energy")
        else:
            try:
                exact_same = _exact_energy_for_params(params, h_poly=h_poly)
            except Exception as exc:  # pragma: no cover - defensive diagnostic path
                incomplete_validation_fields.append("exact_same_cutoff_energy")
                warnings.append(f"same_cutoff_exact_energy_unavailable:{type(exc).__name__}:{exc}")
            if reference_n_ph is not None and working_n_ph is not None and int(reference_n_ph) > int(working_n_ph):
                try:
                    exact_ref = _exact_energy_for_params(params, n_ph_max=int(reference_n_ph))
                except Exception as exc:  # pragma: no cover - defensive diagnostic path
                    incomplete_validation_fields.append("exact_reference_cutoff_energy")
                    warnings.append(f"reference_cutoff_exact_energy_unavailable:{type(exc).__name__}:{exc}")
            if exact_same is not None and exact_ref is not None:
                cutoff_gap = abs(float(exact_same) - float(exact_ref))

    row = {
        "schema": AUDIT_SCHEMA,
        "profile": profile_key,
        "case_id": str(spec.benchmark_id),
        "family": str(spec.family),
        "problem": str(params["problem_key"]),
        "algorithm_id": str(algorithm_id),
        "route_identity_class": _route_identity_class(
            route_payload=route_payload,
            selected_logical_route=selected_route,
            selected_logical_source=selected_source,
            working_n_ph_max=working_n_ph,
            algorithm_id=str(algorithm_id),
        ),
        "static_route_id": str(static_route_id),
        "static_route_identity": route_payload,
        "selected_logical_route": selected_route,
        "selected_logical_mode": selected_mode,
        "selected_logical_source_json": selected_source,
        "selected_logical_source_kind": getattr(spec, "selected_logical_source_kind", None),
        "selected_logical_source_record_count": int(getattr(spec, "selected_logical_source_record_count", 0) or 0),
        "selected_logical_transfer_mode": _option_text(
            policy_options,
            "adapt_selected_logical_transfer_mode",
            str(getattr(spec, "selected_logical_transfer_mode", "exact_match_v1") or "exact_match_v1"),
        ),
        "pool_key": str(resolved_pool_key),
        "pool_key_requested": str(pool.pool_key),
        "optimizer": str(optimizer),
        "novelty_mode": str(static.phase2_novelty_mode),
        "adapt_max_depth": int(static.adapt_max_depth),
        "adapt_reopt_policy": str(static.adapt_reopt_policy),
        "adapt_window_size": int(static.adapt_window_size),
        "adapt_window_topk": int(static.adapt_window_topk),
        "adapt_full_refit_every": int(static.adapt_full_refit_every),
        "adapt_final_full_refit": bool(static.adapt_final_full_refit),
        "working_n_ph_max": working_n_ph,
        "reference_n_ph_ref": reference_n_ph,
        **contract,
        "L": int(params["num_sites"]),
        "boson_encoding": str(params["boson_encoding"]),
        "ordering": str(params["ordering"]),
        "boundary": str(params["boundary"]),
        "feature_n_qubits_hint": feature_nq,
        "feature_pool_size_hint": int(spec.features.pool_size_hint),
        **ham_meta,
        **pool_meta,
        "pool_legal_subspace_filter": pool_meta.get("pool_legal_subspace_filter"),
        "pool_legal_subspace_audit": legal_subspace_audit,
        "pool_legal_subspace_status": (
            legal_subspace_audit.get("status") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "legal_subspace_dim": (
            legal_subspace_audit.get("legal_subspace_dim") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "legal_state_count": (
            legal_subspace_audit.get("legal_state_count") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "illegal_state_count": (
            legal_subspace_audit.get("illegal_state_count") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "pool_generators_tested": (
            legal_subspace_audit.get("generators_tested") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "legal_preserving_count": (
            legal_subspace_audit.get("legal_preserving_count") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "legal_leaking_count": (
            legal_subspace_audit.get("legal_leaking_count") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "unknown_count": (
            legal_subspace_audit.get("unknown_count") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "top_offending_labels": (
            legal_subspace_audit.get("top_offending_labels") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "top_offending_classes": (
            legal_subspace_audit.get("top_offending_classes") if isinstance(legal_subspace_audit, Mapping) else None
        ),
        "exact_same_cutoff_energy": exact_same,
        "exact_reference_cutoff_energy": exact_ref,
        "exact_cutoff_gap": cutoff_gap,
        "validation_complete": len(incomplete_validation_fields) == 0,
        "incomplete_validation_fields": incomplete_validation_fields,
        "mismatches": mismatches,
        "warnings": warnings,
    }
    if attach_existing_result:
        row["existing_result"] = _existing_result_summary(
            _default_result_path(
                profile=profile_key,
                family=str(spec.family),
                case_id=str(spec.benchmark_id),
                algorithm_id=str(algorithm_id),
            )
        )
    return row


def build_route_cutoff_audit_rows(
    *,
    profile: str = TABLE_I_NPH2_REF3_PROFILE,
    families: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    algorithm_id: str = DEFAULT_ALGORITHM_ID,
    pool_key: str = DEFAULT_POOL_KEY,
    include_exact_energies: bool = False,
    include_pool_size: bool = False,
    include_legal_subspace_audit: bool = False,
    legal_subspace_tolerance: float = DEFAULT_LEGAL_SUBSPACE_TOLERANCE,
    attach_existing_results: bool = False,
) -> list[dict[str, Any]]:
    profile_key = table_i_suite_profile(profile)
    family_filter = None if families is None else {str(family) for family in families}
    case_filter = None if case_ids is None else {str(case_id) for case_id in case_ids}
    rows: list[dict[str, Any]] = []
    for spec in table_i_canonical_specs(profile_key):
        if family_filter is not None and str(spec.family) not in family_filter:
            continue
        if case_filter is not None and str(spec.benchmark_id) not in case_filter:
            continue
        rows.append(
            build_route_cutoff_audit_row(
                spec,
                profile=profile_key,
                algorithm_id=str(algorithm_id),
                pool_key=str(pool_key),
                include_exact_energies=bool(include_exact_energies),
                include_pool_size=bool(include_pool_size),
                include_legal_subspace_audit=bool(include_legal_subspace_audit),
                legal_subspace_tolerance=float(legal_subspace_tolerance),
                attach_existing_result=bool(attach_existing_results),
            )
        )
    return rows


def summarize_route_cutoff_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    route_counts = Counter(str(row.get("route_identity_class")) for row in rows)
    profile_counts = Counter(str(row.get("profile")) for row in rows)
    mismatch_rows = [row for row in rows if row.get("mismatches")]
    warning_rows = [row for row in rows if row.get("warnings")]
    existing_result_rows = [row for row in rows if isinstance(row.get("existing_result"), Mapping)]
    existing_result_found = [row for row in existing_result_rows if bool(row["existing_result"].get("found"))]
    incomplete_rows = [row for row in rows if not bool(row.get("validation_complete", True))]
    nph_contract_counts = Counter(str(row.get("n_ph_contract")) for row in rows)
    legal_audit_rows = [
        row for row in rows if isinstance(row.get("pool_legal_subspace_audit"), Mapping)
    ]
    legal_status_counts = Counter(
        str(row["pool_legal_subspace_audit"].get("status")) for row in legal_audit_rows
    )
    legal_leaking_rows = [
        row for row in legal_audit_rows
        if int(row["pool_legal_subspace_audit"].get("legal_leaking_count") or 0) > 0
    ]
    legal_unknown_rows = [
        row for row in legal_audit_rows
        if int(row["pool_legal_subspace_audit"].get("unknown_count") or 0) > 0
    ]
    termwise_risk_rows = [
        row for row in legal_audit_rows
        if int(row["pool_legal_subspace_audit"].get("termwise_component_leak_risk_count") or 0) > 0
    ]
    def _execution_certified(row: Mapping[str, Any]) -> bool:
        filter_meta = row.get("pool_legal_subspace_filter")
        if not isinstance(filter_meta, Mapping):
            return False
        if not bool(filter_meta.get("active", False)):
            return False
        pool_size = row.get("pool_size")
        execution_legal = filter_meta.get("execution_legal_generator_count")
        if pool_size is None or execution_legal is None:
            return False
        return int(execution_legal) >= int(pool_size)

    execution_certified_rows = [
        row for row in legal_audit_rows if _execution_certified(row)
    ]
    uncertified_termwise_risk_rows = [
        row for row in termwise_risk_rows if not _execution_certified(row)
    ]
    return {
        "schema": "table_i_route_cutoff_audit_summary_v1",
        "row_count": int(len(rows)),
        "profile_counts": dict(sorted(profile_counts.items())),
        "route_identity_class_counts": dict(sorted(route_counts.items())),
        "n_ph_contract_counts": dict(sorted(nph_contract_counts.items())),
        "mismatch_row_count": int(len(mismatch_rows)),
        "warning_row_count": int(len(warning_rows)),
        "existing_result_found_count": int(len(existing_result_found)),
        "incomplete_row_count": int(len(incomplete_rows)),
        "likely_route_or_cutoff_mismatch": bool(len(mismatch_rows) > 0),
        "legal_subspace_audit_status_counts": dict(sorted(legal_status_counts.items())),
        "legal_subspace_leaking_row_count": int(len(legal_leaking_rows)),
        "legal_subspace_unknown_row_count": int(len(legal_unknown_rows)),
        "termwise_component_leak_risk_row_count": int(len(termwise_risk_rows)),
        "execution_legal_certified_row_count": int(len(execution_certified_rows)),
        "termwise_component_risk_but_execution_certified_row_count": int(
            len(termwise_risk_rows) - len(uncertified_termwise_risk_rows)
        ),
        "likely_pool_legal_subspace_leak": bool(len(legal_leaking_rows) > 0),
        "likely_termwise_execution_legal_subspace_leak": bool(
            len(uncertified_termwise_risk_rows) > 0
        ),
        "termwise_component_leak_risk_cases": [
            {
                "profile": row.get("profile"),
                "family": row.get("family"),
                "case_id": row.get("case_id"),
                "termwise_component_leak_risk_count": row["pool_legal_subspace_audit"].get(
                    "termwise_component_leak_risk_count"
                ),
                "termwise_component_leak_risk_top_labels": row["pool_legal_subspace_audit"].get(
                    "termwise_component_leak_risk_top_labels"
                ),
            }
            for row in termwise_risk_rows
        ],
        "mismatch_cases": [
            {
                "profile": row.get("profile"),
                "family": row.get("family"),
                "case_id": row.get("case_id"),
                "mismatches": list(row.get("mismatches") or []),
            }
            for row in mismatch_rows
        ],
    }


def build_route_cutoff_audit_payload(
    *,
    profiles: Sequence[str] = (TABLE_I_NPH2_REF3_PROFILE,),
    families: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    algorithm_id: str = DEFAULT_ALGORITHM_ID,
    pool_key: str = DEFAULT_POOL_KEY,
    include_exact_energies: bool = False,
    include_pool_size: bool = False,
    include_legal_subspace_audit: bool = False,
    legal_subspace_tolerance: float = DEFAULT_LEGAL_SUBSPACE_TOLERANCE,
    attach_existing_results: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    profile_keys = tuple(table_i_suite_profile(profile) for profile in profiles)
    for profile_key in profile_keys:
        rows.extend(
            build_route_cutoff_audit_rows(
                profile=profile_key,
                families=families,
                case_ids=case_ids,
                algorithm_id=str(algorithm_id),
                pool_key=str(pool_key),
                include_exact_energies=bool(include_exact_energies),
                include_pool_size=bool(include_pool_size),
                include_legal_subspace_audit=bool(include_legal_subspace_audit),
                legal_subspace_tolerance=float(legal_subspace_tolerance),
                attach_existing_results=bool(attach_existing_results),
            )
        )
    return {
        "schema": AUDIT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": list(profile_keys),
        "families": None if families is None else [str(family) for family in families],
        "case_ids": None if case_ids is None else [str(case_id) for case_id in case_ids],
        "algorithm_id": str(algorithm_id),
        "pool_key": str(pool_key),
        "include_exact_energies": bool(include_exact_energies),
        "include_pool_size": bool(include_pool_size),
        "include_legal_subspace_audit": bool(include_legal_subspace_audit),
        "legal_subspace_tolerance": float(legal_subspace_tolerance),
        "attach_existing_results": bool(attach_existing_results),
        "summary": summarize_route_cutoff_audit(rows),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=[TABLE_I_NPH2_REF3_PROFILE])
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--algorithm-id", default=DEFAULT_ALGORITHM_ID)
    parser.add_argument("--pool-key", default=DEFAULT_POOL_KEY)
    parser.add_argument("--include-exact-energies", action="store_true")
    parser.add_argument("--include-pool-size", action="store_true")
    parser.add_argument("--include-legal-subspace-audit", action="store_true")
    parser.add_argument("--legal-subspace-tolerance", type=float, default=DEFAULT_LEGAL_SUBSPACE_TOLERANCE)
    parser.add_argument("--attach-existing-results", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_route_cutoff_audit_payload(
        profiles=tuple(args.profiles),
        families=None if args.families is None else tuple(args.families),
        case_ids=None if args.case_ids is None else tuple(args.case_ids),
        algorithm_id=str(args.algorithm_id),
        pool_key=str(args.pool_key),
        include_exact_energies=bool(args.include_exact_energies),
        include_pool_size=bool(args.include_pool_size),
        include_legal_subspace_audit=bool(args.include_legal_subspace_audit),
        legal_subspace_tolerance=float(args.legal_subspace_tolerance),
        attach_existing_results=bool(args.attach_existing_results),
    )
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


__all__ = [
    "AUDIT_SCHEMA",
    "LEGAL_SUBSPACE_AUDIT_SCHEMA",
    "DEFAULT_ALGORITHM_ID",
    "DEFAULT_LEGAL_SUBSPACE_TOLERANCE",
    "DEFAULT_POOL_KEY",
    "build_pool_legal_subspace_audit",
    "build_route_cutoff_audit_payload",
    "build_route_cutoff_audit_row",
    "build_route_cutoff_audit_rows",
    "summarize_route_cutoff_audit",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
