from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RotationTermSpec:
    pauli_exyz: str
    coeff_real: float
    nq: int


@dataclass(frozen=True)
class GeneratorParameterBlock:
    candidate_label: str
    logical_index: int
    runtime_start: int
    terms: tuple[RotationTermSpec, ...]

    @property
    def runtime_count(self) -> int:
        return int(len(self.terms))

    @property
    def runtime_stop(self) -> int:
        return int(self.runtime_start + len(self.terms))


@dataclass(frozen=True)
class AnsatzParameterLayout:
    mode: str
    term_order: str
    ignore_identity: bool
    coefficient_tolerance: float
    blocks: tuple[GeneratorParameterBlock, ...]

    @property
    def logical_parameter_count(self) -> int:
        return int(len(self.blocks))

    @property
    def runtime_parameter_count(self) -> int:
        if not self.blocks:
            return 0
        return int(self.blocks[-1].runtime_stop)

    def runtime_slice_for_logical_index(self, logical_index: int) -> slice:
        block = self.blocks[int(logical_index)]
        return slice(int(block.runtime_start), int(block.runtime_stop))


"""
H = Σ_j c_j P_j
active_terms(H) = ordered, filtered {(P_j, Re[c_j])}
"""
def iter_runtime_rotation_terms(
    poly: Any,
    *,
    ignore_identity: bool = True,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> tuple[RotationTermSpec, ...]:
    poly_terms = list(poly.return_polynomial())
    if not poly_terms:
        return tuple()

    nq = int(poly_terms[0].nqubit())
    id_label = "e" * nq
    ordered = list(poly_terms)
    if sort_terms:
        ordered.sort(key=lambda t: t.pw2strng())

    out: list[RotationTermSpec] = []
    tol = float(coefficient_tolerance)
    for term in ordered:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if int(term.nqubit()) != nq:
            raise ValueError(
                f"Inconsistent polynomial qubit count: expected {nq}, got {term.nqubit()}."
            )
        if len(label) != nq:
            raise ValueError(f"Invalid Pauli label length for '{label}': expected {nq}.")
        if abs(coeff) < tol:
            continue
        if ignore_identity and label == id_label:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(f"Non-negligible imaginary coefficient in term {label}: {coeff}.")
        out.append(RotationTermSpec(pauli_exyz=label, coeff_real=float(coeff.real), nq=nq))
    return tuple(out)


"""
layout = ⨆_k block_k, block_k = ordered active Pauli terms for logical generator k
"""
def build_parameter_layout(
    terms: Sequence[Any],
    *,
    ignore_identity: bool = True,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> AnsatzParameterLayout:
    blocks: list[GeneratorParameterBlock] = []
    runtime_start = 0
    for logical_index, term in enumerate(terms):
        specs = iter_runtime_rotation_terms(
            getattr(term, "polynomial"),
            ignore_identity=ignore_identity,
            coefficient_tolerance=coefficient_tolerance,
            sort_terms=sort_terms,
        )
        block = GeneratorParameterBlock(
            candidate_label=str(getattr(term, "label", f"term_{logical_index}")),
            logical_index=int(logical_index),
            runtime_start=int(runtime_start),
            terms=tuple(specs),
        )
        blocks.append(block)
        runtime_start += int(block.runtime_count)
    return AnsatzParameterLayout(
        mode="per_pauli_term_v1",
        term_order=("sorted" if sort_terms else "native"),
        ignore_identity=bool(ignore_identity),
        coefficient_tolerance=float(coefficient_tolerance),
        blocks=tuple(blocks),
    )


def runtime_insert_position(layout: AnsatzParameterLayout, logical_position: int) -> int:
    pos = max(0, min(int(logical_position), int(layout.logical_parameter_count)))
    if pos >= int(layout.logical_parameter_count):
        return int(layout.runtime_parameter_count)
    return int(layout.blocks[pos].runtime_start)


def runtime_indices_for_logical_indices(
    layout: AnsatzParameterLayout,
    logical_indices: Sequence[int],
) -> list[int]:
    out: list[int] = []
    for logical_index in logical_indices:
        idx = int(logical_index)
        if idx < 0 or idx >= int(layout.logical_parameter_count):
            continue
        block = layout.blocks[idx]
        out.extend(range(int(block.runtime_start), int(block.runtime_stop)))
    return [int(x) for x in out]


def expand_legacy_logical_theta(
    theta_logical: np.ndarray | Sequence[float],
    layout: AnsatzParameterLayout,
) -> np.ndarray:
    base = np.asarray(theta_logical, dtype=float).reshape(-1)
    if int(base.size) != int(layout.logical_parameter_count):
        raise ValueError(
            f"logical theta length mismatch: got {base.size}, expected {layout.logical_parameter_count}."
        )
    out = np.zeros(int(layout.runtime_parameter_count), dtype=float)
    for block, theta_val in zip(layout.blocks, base):
        if int(block.runtime_count) <= 0:
            continue
        out[block.runtime_start:block.runtime_stop] = float(theta_val)
    return out


def project_runtime_theta_block_mean(
    theta_runtime: np.ndarray | Sequence[float],
    layout: AnsatzParameterLayout,
) -> np.ndarray:
    arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(arr.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"runtime theta length mismatch: got {arr.size}, expected {layout.runtime_parameter_count}."
        )
    out = np.zeros(int(layout.logical_parameter_count), dtype=float)
    for block in layout.blocks:
        if int(block.runtime_count) <= 0:
            out[block.logical_index] = 0.0
            continue
        vals = arr[block.runtime_start:block.runtime_stop]
        out[block.logical_index] = float(np.mean(vals))
    return out


def serialize_layout(layout: AnsatzParameterLayout) -> dict[str, Any]:
    return {
        "mode": str(layout.mode),
        "term_order": str(layout.term_order),
        "ignore_identity": bool(layout.ignore_identity),
        "coefficient_tolerance": float(layout.coefficient_tolerance),
        "logical_operator_count": int(layout.logical_parameter_count),
        "runtime_parameter_count": int(layout.runtime_parameter_count),
        "blocks": [
            {
                "candidate_label": str(block.candidate_label),
                "logical_index": int(block.logical_index),
                "runtime_start": int(block.runtime_start),
                "runtime_count": int(block.runtime_count),
                "runtime_terms_exyz": [
                    {
                        "pauli_exyz": str(spec.pauli_exyz),
                        "coeff_re": float(spec.coeff_real),
                        "coeff_im": 0.0,
                        "nq": int(spec.nq),
                    }
                    for spec in block.terms
                ],
            }
            for block in layout.blocks
        ],
    }


def deserialize_layout(payload: Mapping[str, Any]) -> AnsatzParameterLayout:
    blocks_raw = payload.get("blocks", [])
    if not isinstance(blocks_raw, Sequence):
        raise ValueError("parameterization.blocks must be a sequence.")
    blocks: list[GeneratorParameterBlock] = []
    runtime_start_expected = 0
    for logical_index, raw_block in enumerate(blocks_raw):
        if not isinstance(raw_block, Mapping):
            raise ValueError("Each parameterization block must be a mapping.")
        terms_raw = raw_block.get("runtime_terms_exyz", [])
        if not isinstance(terms_raw, Sequence):
            raise ValueError("parameterization block runtime_terms_exyz must be a sequence.")
        terms: list[RotationTermSpec] = []
        for raw_term in terms_raw:
            if not isinstance(raw_term, Mapping):
                raise ValueError("parameterization runtime term must be a mapping.")
            label = str(raw_term.get("pauli_exyz", "")).strip()
            if label == "":
                raise ValueError("parameterization runtime term missing pauli_exyz.")
            coeff_re = float(raw_term.get("coeff_re", 0.0))
            coeff_im = float(raw_term.get("coeff_im", 0.0))
            if abs(coeff_im) > float(payload.get("coefficient_tolerance", 1e-12)):
                raise ValueError(f"parameterization runtime term {label} has non-zero coeff_im={coeff_im}.")
            nq = int(raw_term.get("nq", len(label)))
            if len(label) != nq:
                raise ValueError(f"parameterization runtime term {label} length mismatch vs nq={nq}.")
            terms.append(RotationTermSpec(pauli_exyz=label, coeff_real=coeff_re, nq=nq))
        runtime_start = int(raw_block.get("runtime_start", runtime_start_expected))
        if runtime_start != runtime_start_expected:
            raise ValueError(
                f"parameterization block runtime_start mismatch: got {runtime_start}, expected {runtime_start_expected}."
            )
        block = GeneratorParameterBlock(
            candidate_label=str(raw_block.get("candidate_label", f"term_{logical_index}")),
            logical_index=int(raw_block.get("logical_index", logical_index)),
            runtime_start=int(runtime_start),
            terms=tuple(terms),
        )
        if int(block.logical_index) != int(logical_index):
            raise ValueError(
                f"parameterization block logical_index mismatch: got {block.logical_index}, expected {logical_index}."
            )
        runtime_count = int(raw_block.get("runtime_count", block.runtime_count))
        if runtime_count != int(block.runtime_count):
            raise ValueError(
                f"parameterization block runtime_count mismatch for {block.candidate_label}: got {runtime_count}, expected {block.runtime_count}."
            )
        blocks.append(block)
        runtime_start_expected = int(block.runtime_stop)

    layout = AnsatzParameterLayout(
        mode=str(payload.get("mode", "per_pauli_term_v1")),
        term_order=str(payload.get("term_order", "sorted")),
        ignore_identity=bool(payload.get("ignore_identity", True)),
        coefficient_tolerance=float(payload.get("coefficient_tolerance", 1e-12)),
        blocks=tuple(blocks),
    )
    runtime_parameter_count = int(payload.get("runtime_parameter_count", layout.runtime_parameter_count))
    if runtime_parameter_count != int(layout.runtime_parameter_count):
        raise ValueError(
            f"parameterization runtime_parameter_count mismatch: got {runtime_parameter_count}, expected {layout.runtime_parameter_count}."
        )
    logical_operator_count = int(payload.get("logical_operator_count", layout.logical_parameter_count))
    if logical_operator_count != int(layout.logical_parameter_count):
        raise ValueError(
            f"parameterization logical_operator_count mismatch: got {logical_operator_count}, expected {layout.logical_parameter_count}."
        )
    return layout


__all__ = [
    "AnsatzParameterLayout",
    "GeneratorParameterBlock",
    "RotationTermSpec",
    "build_parameter_layout",
    "deserialize_layout",
    "expand_legacy_logical_theta",
    "iter_runtime_rotation_terms",
    "project_runtime_theta_block_mean",
    "runtime_indices_for_logical_indices",
    "runtime_insert_position",
    "serialize_layout",
]
