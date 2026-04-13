#!/usr/bin/env python3
"""Observable builders for fixed-manifold measured McLachlan."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.noise_oracle_runtime import (
    build_runtime_layout_circuit,
    pauli_poly_to_sparse_pauli_op,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
from src.quantum.ansatz_parameterization import AnsatzParameterLayout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


@dataclass(frozen=True)
class RuntimeRotationSpec:
    runtime_index: int
    logical_index: int
    candidate_label: str
    pauli_exyz: str
    coeff_real: float
    nq: int


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    kind: str
    runtime_index: int | None
    runtime_pair: tuple[int, int] | None
    poly: PauliPolynomial | None
    sparse_op: Any | None
    term_count: int
    is_zero: bool


@dataclass(frozen=True)
class CheckpointObservablePlan:
    circuit: Any
    runtime_rotations: tuple[RuntimeRotationSpec, ...]
    energy: ObservableSpec
    variance_h2: ObservableSpec
    generator_means: tuple[ObservableSpec, ...]
    pair_anticommutators: dict[tuple[int, int], ObservableSpec]
    force_anticommutators: tuple[ObservableSpec, ...]
    stats: dict[str, Any]


PolyMap = dict[str, complex]


"""
runtime_rotations(layout) = ordered runtime Pauli rotations from serialized layout order
"""
def flatten_runtime_rotations(layout: AnsatzParameterLayout) -> tuple[RuntimeRotationSpec, ...]:
    out: list[RuntimeRotationSpec] = []
    runtime_expected = 0
    for block in layout.blocks:
        if int(block.runtime_start) != int(runtime_expected):
            raise ValueError(
                f"Runtime layout discontinuity: block {block.candidate_label!r} starts at "
                f"{block.runtime_start}, expected {runtime_expected}."
            )
        for local_idx, term in enumerate(block.terms):
            out.append(
                RuntimeRotationSpec(
                    runtime_index=int(runtime_expected + local_idx),
                    logical_index=int(block.logical_index),
                    candidate_label=str(block.candidate_label),
                    pauli_exyz=str(term.pauli_exyz),
                    coeff_real=float(term.coeff_real),
                    nq=int(term.nq),
                )
            )
        runtime_expected = int(block.runtime_stop)
    if int(runtime_expected) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"Runtime layout count mismatch: flattened {runtime_expected} vs layout {layout.runtime_parameter_count}."
        )
    return tuple(out)


@lru_cache(maxsize=None)
def _unit_pauli_term(label: str) -> PauliTerm:
    return PauliTerm(len(label), ps=str(label), pc=1.0)


@lru_cache(maxsize=None)
def _label_product(lhs: str, rhs: str) -> tuple[str, complex]:
    prod = _unit_pauli_term(str(lhs)) * _unit_pauli_term(str(rhs))
    return str(prod.pw2strng()), complex(prod.p_coeff)


@lru_cache(maxsize=None)
def _labels_commute(lhs: str, rhs: str) -> bool:
    if len(lhs) != len(rhs):
        raise ValueError(f"Pauli label length mismatch: {lhs!r} vs {rhs!r}.")
    anti_count = 0
    for lch, rch in zip(str(lhs), str(rhs)):
        if lch == "e" or rch == "e" or lch == rch:
            continue
        anti_count += 1
    return bool((anti_count % 2) == 0)


def _aggregate_map(dst: PolyMap, label: str, coeff: complex) -> None:
    dst[str(label)] = complex(dst.get(str(label), 0.0 + 0.0j) + complex(coeff))


def _drop_small_terms(poly_map: PolyMap, *, drop_abs_tol: float) -> PolyMap:
    out: PolyMap = {}
    for label, coeff in poly_map.items():
        if abs(complex(coeff)) > float(drop_abs_tol):
            out[str(label)] = complex(coeff)
    return out


def _validate_real_coefficients(
    poly_map: PolyMap,
    *,
    hermiticity_tol: float,
    context: str,
) -> None:
    for label, coeff in poly_map.items():
        if abs(complex(coeff).imag) > float(hermiticity_tol):
            raise ValueError(
                f"{context} produced non-Hermitian coefficient for {label}: {coeff}."
            )


def _poly_map_to_pauli_polynomial(
    poly_map: PolyMap,
    *,
    nq: int,
    drop_abs_tol: float,
    hermiticity_tol: float,
    context: str,
) -> PauliPolynomial | None:
    cleaned = _drop_small_terms(poly_map, drop_abs_tol=float(drop_abs_tol))
    if not cleaned:
        return None
    _validate_real_coefficients(cleaned, hermiticity_tol=float(hermiticity_tol), context=context)
    poly = PauliPolynomial("JW")
    for label in sorted(cleaned.keys()):
        coeff = complex(cleaned[label])
        poly.add_term(PauliTerm(int(nq), ps=str(label), pc=float(np.real(coeff))))
    return poly


def _poly_map_from_polynomial(poly: Any, *, drop_abs_tol: float) -> tuple[int, PolyMap]:
    terms = list(poly.return_polynomial())
    if not terms:
        raise ValueError("Polynomial is empty; cannot infer qubit count.")
    nq = int(terms[0].nqubit())
    out: PolyMap = {}
    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        _aggregate_map(out, label, coeff)
    return int(nq), _drop_small_terms(out, drop_abs_tol=float(drop_abs_tol))


def _runtime_term_map(rotation: RuntimeRotationSpec) -> PolyMap:
    return {str(rotation.pauli_exyz): 1.0 + 0.0j}


"""
U_j Q U_j^† with U_j = exp(-i θ_j c_j P_j)
"""
def _conjugate_poly_map_by_runtime_rotation(
    poly_map: PolyMap,
    rotation: RuntimeRotationSpec,
    theta_runtime_value: float,
    *,
    drop_abs_tol: float,
    hermiticity_tol: float,
) -> PolyMap:
    theta_eff = float(theta_runtime_value) * float(rotation.coeff_real)
    if abs(theta_eff) <= float(drop_abs_tol):
        return dict(poly_map)
    rot_label = str(rotation.pauli_exyz)
    c = float(math.cos(2.0 * float(theta_eff)))
    s = float(math.sin(2.0 * float(theta_eff)))
    out: PolyMap = {}
    for label, coeff in poly_map.items():
        coeff_c = complex(coeff)
        if _labels_commute(rot_label, str(label)):
            _aggregate_map(out, str(label), coeff_c)
            continue
        if abs(c) > float(drop_abs_tol):
            _aggregate_map(out, str(label), coeff_c * float(c))
        prod_label, prod_phase = _label_product(rot_label, str(label))
        rotated_coeff = coeff_c * (-1.0j) * float(s) * complex(prod_phase)
        if abs(rotated_coeff) > float(drop_abs_tol):
            _aggregate_map(out, prod_label, rotated_coeff)
    cleaned = _drop_small_terms(out, drop_abs_tol=float(drop_abs_tol))
    _validate_real_coefficients(
        cleaned,
        hermiticity_tol=float(hermiticity_tol),
        context=f"conjugation of {rot_label}",
    )
    return cleaned


"""
A_k(θ) = U_suffix(θ) P_k U_suffix(θ)^†
"""
def build_heisenberg_generator_map(
    rotations: Sequence[RuntimeRotationSpec],
    theta_runtime: Sequence[float],
    target_runtime_index: int,
    *,
    drop_abs_tol: float,
    hermiticity_tol: float,
) -> tuple[int, PolyMap]:
    rots = tuple(rotations)
    idx = int(target_runtime_index)
    if idx < 0 or idx >= len(rots):
        raise ValueError(f"target_runtime_index {idx} out of range for {len(rots)} rotations.")
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(theta_arr.size) != int(len(rots)):
        raise ValueError(
            f"theta_runtime length mismatch: got {theta_arr.size}, expected {len(rots)}."
        )
    target = rots[idx]
    out = _runtime_term_map(target)
    for suffix_idx in range(int(idx) + 1, int(len(rots))):
        out = _conjugate_poly_map_by_runtime_rotation(
            out,
            rots[int(suffix_idx)],
            float(theta_arr[int(suffix_idx)]),
            drop_abs_tol=float(drop_abs_tol),
            hermiticity_tol=float(hermiticity_tol),
        )
    return int(target.nq), dict(out)


"""
A_k(θ) = U_suffix(θ) P_k U_suffix(θ)^†
"""
def build_heisenberg_generator(
    rotations: Sequence[RuntimeRotationSpec],
    theta_runtime: Sequence[float],
    target_runtime_index: int,
    *,
    drop_abs_tol: float,
    hermiticity_tol: float,
) -> PauliPolynomial:
    nq, poly_map = build_heisenberg_generator_map(
        rotations,
        theta_runtime,
        target_runtime_index,
        drop_abs_tol=float(drop_abs_tol),
        hermiticity_tol=float(hermiticity_tol),
    )
    poly = _poly_map_to_pauli_polynomial(
        poly_map,
        nq=int(nq),
        drop_abs_tol=float(drop_abs_tol),
        hermiticity_tol=float(hermiticity_tol),
        context=f"Heisenberg generator {target_runtime_index}",
    )
    if poly is None:
        raise ValueError(f"Heisenberg generator {target_runtime_index} collapsed to zero.")
    return poly


"""
0.5 {L, R} = 0.5 (L R + R L)
"""
def build_symmetrized_product_map(
    lhs_map: PolyMap,
    rhs_map: PolyMap,
    *,
    drop_abs_tol: float,
    hermiticity_tol: float,
) -> PolyMap:
    out: PolyMap = {}
    for lhs_label, lhs_coeff in lhs_map.items():
        for rhs_label, rhs_coeff in rhs_map.items():
            if not _labels_commute(str(lhs_label), str(rhs_label)):
                continue
            prod_label, prod_phase = _label_product(str(lhs_label), str(rhs_label))
            _aggregate_map(
                out,
                prod_label,
                complex(lhs_coeff) * complex(rhs_coeff) * complex(prod_phase),
            )
    cleaned = _drop_small_terms(out, drop_abs_tol=float(drop_abs_tol))
    _validate_real_coefficients(
        cleaned,
        hermiticity_tol=float(hermiticity_tol),
        context="symmetrized product",
    )
    return cleaned


def build_full_product_map(
    lhs_map: PolyMap,
    rhs_map: PolyMap,
    *,
    drop_abs_tol: float,
    hermiticity_tol: float,
) -> PolyMap:
    out: PolyMap = {}
    for lhs_label, lhs_coeff in lhs_map.items():
        for rhs_label, rhs_coeff in rhs_map.items():
            prod_label, prod_phase = _label_product(str(lhs_label), str(rhs_label))
            _aggregate_map(
                out,
                prod_label,
                complex(lhs_coeff) * complex(rhs_coeff) * complex(prod_phase),
            )
    cleaned = _drop_small_terms(out, drop_abs_tol=float(drop_abs_tol))
    _validate_real_coefficients(
        cleaned,
        hermiticity_tol=float(hermiticity_tol),
        context="full product",
    )
    return cleaned


def _make_observable_spec(
    *,
    name: str,
    kind: str,
    nq: int,
    poly_map: PolyMap,
    runtime_index: int | None,
    runtime_pair: tuple[int, int] | None,
    drop_abs_tol: float,
    hermiticity_tol: float,
) -> ObservableSpec:
    poly = _poly_map_to_pauli_polynomial(
        poly_map,
        nq=int(nq),
        drop_abs_tol=float(drop_abs_tol),
        hermiticity_tol=float(hermiticity_tol),
        context=str(name),
    )
    if poly is None:
        return ObservableSpec(
            name=str(name),
            kind=str(kind),
            runtime_index=(None if runtime_index is None else int(runtime_index)),
            runtime_pair=(None if runtime_pair is None else tuple(int(x) for x in runtime_pair)),
            poly=None,
            sparse_op=None,
            term_count=0,
            is_zero=True,
        )
    sparse = pauli_poly_to_sparse_pauli_op(poly, tol=float(drop_abs_tol))
    return ObservableSpec(
        name=str(name),
        kind=str(kind),
        runtime_index=(None if runtime_index is None else int(runtime_index)),
        runtime_pair=(None if runtime_pair is None else tuple(int(x) for x in runtime_pair)),
        poly=poly,
        sparse_op=sparse,
        term_count=int(poly.count_number_terms()),
        is_zero=False,
    )


def build_checkpoint_observable_plan_from_layout(
    layout: AnsatzParameterLayout,
    theta_runtime: Sequence[float],
    *,
    psi_ref: np.ndarray | Sequence[complex],
    h_poly: Any,
    drop_abs_tol: float = 1.0e-12,
    hermiticity_tol: float = 1.0e-10,
    max_observable_terms: int = 512,
) -> CheckpointObservablePlan:
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(theta_arr.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"theta_runtime length mismatch: got {theta_arr.size}, expected {layout.runtime_parameter_count}."
        )
    rotations = flatten_runtime_rotations(layout)
    if not rotations:
        raise ValueError("Measured fixed-manifold runner requires at least one runtime rotation.")
    nq = int(rotations[0].nq)
    h_nq, h_map = _poly_map_from_polynomial(h_poly, drop_abs_tol=float(drop_abs_tol))
    if int(h_nq) != int(nq):
        raise ValueError(f"Hamiltonian nq mismatch: {h_nq} vs {nq}.")

    circuit = build_runtime_layout_circuit(
        layout,
        theta_arr,
        int(nq),
        reference_state=np.asarray(psi_ref, dtype=complex).reshape(-1),
    )
    energy = _make_observable_spec(
        name="energy",
        kind="energy",
        nq=int(nq),
        poly_map=dict(h_map),
        runtime_index=None,
        runtime_pair=None,
        drop_abs_tol=float(drop_abs_tol),
        hermiticity_tol=float(hermiticity_tol),
    )
    h2_map = build_full_product_map(
        h_map,
        h_map,
        drop_abs_tol=float(drop_abs_tol),
        hermiticity_tol=float(hermiticity_tol),
    )
    variance_h2 = _make_observable_spec(
        name="h2",
        kind="variance_h2",
        nq=int(nq),
        poly_map=h2_map,
        runtime_index=None,
        runtime_pair=None,
        drop_abs_tol=float(drop_abs_tol),
        hermiticity_tol=float(hermiticity_tol),
    )

    generator_maps: list[PolyMap] = []
    generator_means: list[ObservableSpec] = []
    force_anticommutators: list[ObservableSpec] = []
    pair_anticommutators: dict[tuple[int, int], ObservableSpec] = {}
    max_generator_terms = 0
    max_force_terms = 0
    max_pair_terms = 0
    zero_count = 0
    max_terms_any = 0

    for idx in range(len(rotations)):
        _, a_map = build_heisenberg_generator_map(
            rotations,
            theta_arr,
            idx,
            drop_abs_tol=float(drop_abs_tol),
            hermiticity_tol=float(hermiticity_tol),
        )
        generator_maps.append(dict(a_map))
        gen_spec = _make_observable_spec(
            name=f"A_{idx}",
            kind="generator_mean",
            nq=int(nq),
            poly_map=a_map,
            runtime_index=int(idx),
            runtime_pair=None,
            drop_abs_tol=float(drop_abs_tol),
            hermiticity_tol=float(hermiticity_tol),
        )
        generator_means.append(gen_spec)
        max_generator_terms = max(int(max_generator_terms), int(gen_spec.term_count))
        zero_count += int(gen_spec.is_zero)
        max_terms_any = max(int(max_terms_any), int(gen_spec.term_count))

        force_map = build_symmetrized_product_map(
            a_map,
            h_map,
            drop_abs_tol=float(drop_abs_tol),
            hermiticity_tol=float(hermiticity_tol),
        )
        force_spec = _make_observable_spec(
            name=f"AHsym_{idx}",
            kind="force_anticommutator",
            nq=int(nq),
            poly_map=force_map,
            runtime_index=int(idx),
            runtime_pair=None,
            drop_abs_tol=float(drop_abs_tol),
            hermiticity_tol=float(hermiticity_tol),
        )
        force_anticommutators.append(force_spec)
        max_force_terms = max(int(max_force_terms), int(force_spec.term_count))
        zero_count += int(force_spec.is_zero)
        max_terms_any = max(int(max_terms_any), int(force_spec.term_count))

    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            pair_map = build_symmetrized_product_map(
                generator_maps[i],
                generator_maps[j],
                drop_abs_tol=float(drop_abs_tol),
                hermiticity_tol=float(hermiticity_tol),
            )
            pair_spec = _make_observable_spec(
                name=f"AAsym_{i}_{j}",
                kind="pair_anticommutator",
                nq=int(nq),
                poly_map=pair_map,
                runtime_index=None,
                runtime_pair=(int(i), int(j)),
                drop_abs_tol=float(drop_abs_tol),
                hermiticity_tol=float(hermiticity_tol),
            )
            pair_anticommutators[(int(i), int(j))] = pair_spec
            max_pair_terms = max(int(max_pair_terms), int(pair_spec.term_count))
            zero_count += int(pair_spec.is_zero)
            max_terms_any = max(int(max_terms_any), int(pair_spec.term_count))

    zero_count += int(energy.is_zero) + int(variance_h2.is_zero)
    max_terms_any = max(int(max_terms_any), int(energy.term_count), int(variance_h2.term_count))
    nonzero_specs = [energy, variance_h2, *generator_means, *force_anticommutators, *pair_anticommutators.values()]
    for spec in nonzero_specs:
        if (not spec.is_zero) and int(spec.term_count) > int(max_observable_terms):
            raise ValueError(
                f"Observable {spec.name} exceeds term cap: {spec.term_count} > {max_observable_terms}."
            )

    return CheckpointObservablePlan(
        circuit=circuit,
        runtime_rotations=tuple(rotations),
        energy=energy,
        variance_h2=variance_h2,
        generator_means=tuple(generator_means),
        pair_anticommutators=dict(pair_anticommutators),
        force_anticommutators=tuple(force_anticommutators),
        stats={
            "runtime_parameter_count": int(len(rotations)),
            "generator_mean_count": int(len(generator_means)),
            "pair_anticommutator_count": int(len(pair_anticommutators)),
            "force_anticommutator_count": int(len(force_anticommutators)),
            "zero_observable_count": int(zero_count),
            "max_generator_terms": int(max_generator_terms),
            "max_pair_terms": int(max_pair_terms),
            "max_force_terms": int(max_force_terms),
            "max_observable_terms_any": int(max_terms_any),
            "observable_max_terms_cap": int(max_observable_terms),
        },
    )


def build_checkpoint_observable_plan(
    replay_context: ReplayScaffoldContext,
    theta_runtime: Sequence[float],
    *,
    h_poly: Any | None = None,
    drop_abs_tol: float = 1.0e-12,
    hermiticity_tol: float = 1.0e-10,
    max_observable_terms: int = 512,
) -> CheckpointObservablePlan:
    return build_checkpoint_observable_plan_from_layout(
        replay_context.base_layout,
        theta_runtime,
        psi_ref=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
        h_poly=(replay_context.h_poly if h_poly is None else h_poly),
        drop_abs_tol=float(drop_abs_tol),
        hermiticity_tol=float(hermiticity_tol),
        max_observable_terms=int(max_observable_terms),
    )


__all__ = [
    "CheckpointObservablePlan",
    "ObservableSpec",
    "RuntimeRotationSpec",
    "build_checkpoint_observable_plan",
    "build_checkpoint_observable_plan_from_layout",
    "build_full_product_map",
    "build_heisenberg_generator",
    "build_heisenberg_generator_map",
    "build_symmetrized_product_map",
    "flatten_runtime_rotations",
]
