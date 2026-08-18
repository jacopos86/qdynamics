#!/usr/bin/env python3
"""Single-particle retarded Green functions from source-specific QSE sectors.

RECONSTRUCTION (2026-08-18): the original module was never committed and was
lost when snapshot commit 6442fbb5 captured its importers without it. This
implementation is reconstructed against the committed behavioral spec in
``test/test_qse_green_functions.py`` and the CLI wiring in ``__main__.py``.

For each fermionic mode ``p`` the diagonal retarded Green function is

    G^R_pp(omega) = sum_n |<n^{+}|c_p^dag|psi>|^2 / (omega - (E_n^{+} - E_ref) + i eta)
                  + sum_m |<m^{-}|c_p|psi>|^2 / (omega + (E_m^{-} - E_ref) + i eta)

with ``E_ref = <psi|H|psi>``. Jordan--Wigner ladder sources use the repo
convention (qubit 0 is the least-significant computational index; the JW
string collects parity from modes below ``p``). Each non-zero source gets its
own QSE solve: the parent operator basis is re-applied to the normalized
source state with identity sector projection and no reference projection
(``source_specific_qse_solves``); zero sources are reported explicitly and
skipped. The diagonal spectral function is ``A(omega) = -Im G^R / pi``, and
canonical anticommutator sum rules are recorded as diagnostics. Only the
Lorentzian kernel corresponds to the ``i eta`` retarded prescription; other
kernels are rejected. All payloads are post-run diagnostics and never feed
controller decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEBasisElement,
    QSEBasisVectorPolicy,
    QSEPruningConfig,
    QSEResult,
    _apply_basis_element,
    _apply_polynomial_operator,
    _config as _core_config,
    normalize_statevector,
)
from pipelines.qse_spectra.spectral_functions import BroadeningKernelConfig, SpectralGrid

GREEN_FUNCTION_SCHEMA_VERSION = "qse_green_function_v1"

_ZERO_SOURCE_NORM_TOLERANCE = 1.0e-12
_SECTOR_OVERLAP_RELATIVE_CUTOFF = 1.0e-12


@dataclass(frozen=True)
class GreenFunctionMode:
    """One fermionic mode request: display label and JW mode index."""

    label: str
    mode_index: int

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("GreenFunctionMode label must be non-empty.")
        if int(self.mode_index) < 0:
            raise ValueError("GreenFunctionMode mode_index must be >= 0.")


@dataclass(frozen=True)
class GreenFunctionSourceState:
    """A JW ladder source ``c_p|psi>`` or ``c_p^dag|psi>`` with its norm."""

    source_state: np.ndarray
    operation: str
    mode_label: str
    mode_index: int
    source_norm: float
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_green_function_mode_spec(spec: str) -> GreenFunctionMode:
    """Parse ``label=mode_index`` CLI syntax into a :class:`GreenFunctionMode`."""

    text = str(spec).strip()
    if "=" not in text:
        raise ValueError(f"Invalid green function mode spec {text!r}; expected label=mode_index.")
    label, _, raw_index = text.partition("=")
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid green function mode spec {text!r}: label must be non-empty.")
    try:
        index = int(raw_index.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid green function mode spec {text!r}: mode_index must be an integer."
        ) from exc
    return GreenFunctionMode(label=label, mode_index=index)


def parse_green_function_mode_specs(specs: Sequence[str]) -> tuple[GreenFunctionMode, ...]:
    return tuple(parse_green_function_mode_spec(spec) for spec in specs)


def jw_ladder_source_state(
    state: np.ndarray,
    *,
    mode: GreenFunctionMode,
    operation: str,
    expected_nq: int | None = None,
) -> GreenFunctionSourceState:
    """Apply the JW ladder operator ``c_p`` (removal) or ``c_p^dag`` (addition)."""

    op = str(operation)
    if op not in {"addition", "removal"}:
        raise ValueError(f"JW ladder operation must be 'addition' or 'removal'; got {op!r}.")
    psi = np.asarray(state, dtype=complex).reshape(-1)
    dim = int(psi.size)
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("JW ladder source state dimension must be a power of two.")
    nq = dim.bit_length() - 1
    if expected_nq is not None and int(expected_nq) != nq:
        raise ValueError(f"JW ladder state has nq={nq}; expected {int(expected_nq)}.")
    p = int(mode.mode_index)
    if p < 0 or p >= nq:
        raise ValueError(f"JW ladder mode index {p} out of range for nq={nq}.")

    out = np.zeros_like(psi)
    bit = 1 << p
    lower_mask = bit - 1
    for index in range(dim):
        amplitude = psi[index]
        if amplitude == 0.0:
            continue
        occupied = bool(index & bit)
        if op == "addition" and occupied:
            continue
        if op == "removal" and not occupied:
            continue
        parity = -1.0 if (bin(index & lower_mask).count("1") % 2) else 1.0
        target = index | bit if op == "addition" else index & ~bit
        out[target] += parity * amplitude
    return GreenFunctionSourceState(
        source_state=out,
        operation=op,
        mode_label=str(mode.label),
        mode_index=p,
        source_norm=float(np.linalg.norm(out)),
    )


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _solve_source_sector(
    *,
    hamiltonian: Any,
    source: GreenFunctionSourceState,
    basis_elements: Sequence[QSEBasisElement],
    config: QSEPruningConfig,
) -> list[dict[str, float | complex]]:
    """Solve the source-specific QSE sector; return (energy, residue-amplitude) roots."""

    raw = source.source_state
    norm = float(source.source_norm)
    normalized = raw / norm
    nq = int(raw.size.bit_length() - 1)
    cache: dict[str, Any] = {}
    vectors = [
        np.asarray(
            _apply_basis_element(element, normalized, nq=nq, config=config, pauli_action_cache=cache),
            dtype=complex,
        ).reshape(-1)
        for element in basis_elements
    ]
    if not vectors:
        raise ValueError("Green-function sector solve requires at least one basis element.")
    count = len(vectors)
    overlap = np.empty((count, count), dtype=complex)
    ham = np.empty((count, count), dtype=complex)
    h_vectors = [
        np.asarray(
            _apply_polynomial_operator(
                hamiltonian, vector, nq=nq, name="hamiltonian", config=config, pauli_action_cache=cache
            ),
            dtype=complex,
        ).reshape(-1)
        for vector in vectors
    ]
    for i in range(count):
        for j in range(count):
            overlap[i, j] = complex(np.vdot(vectors[i], vectors[j]))
            ham[i, j] = complex(np.vdot(vectors[i], h_vectors[j]))
    overlap = 0.5 * (overlap + overlap.conj().T)
    ham = 0.5 * (ham + ham.conj().T)

    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(overlap)
    cutoff = float(_SECTOR_OVERLAP_RELATIVE_CUTOFF) * float(max(overlap_eigenvalues.max(), 0.0))
    retained = overlap_eigenvalues > cutoff
    if not bool(retained.any()):
        return []
    transform = overlap_eigenvectors[:, retained] / np.sqrt(overlap_eigenvalues[retained])
    reduced = transform.conj().T @ ham @ transform
    energies, reduced_vectors = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
    coefficients = transform @ reduced_vectors

    roots: list[dict[str, float | complex]] = []
    for root_index in range(int(energies.size)):
        state = np.zeros_like(raw)
        for basis_index in range(count):
            state += complex(coefficients[basis_index, root_index]) * vectors[basis_index]
        amplitude = complex(np.vdot(state, raw))
        roots.append({"energy": float(energies[root_index]), "amplitude": amplitude})
    return roots


def build_green_function_payload(
    result: QSEResult,
    *,
    hamiltonian: Any,
    prepared_state: np.ndarray,
    modes: Sequence[GreenFunctionMode],
    grid: SpectralGrid,
    kernel_config: BroadeningKernelConfig,
    fermion_mode_count: int,
    basis_elements: Sequence[QSEBasisElement] | None = None,
    config: Any = None,
) -> dict[str, Any]:
    """Build the additive ``qse_green_function_v1`` manifest payload."""

    if str(kernel_config.kernel) != "lorentzian":
        raise ValueError(
            "green-function retarded broadening requires the lorentzian kernel "
            f"(the i*eta prescription); got {kernel_config.kernel!r}."
        )
    mode_count_limit = int(fermion_mode_count)
    if mode_count_limit <= 0:
        raise ValueError("fermion_mode_count must be positive.")
    labels = [str(mode.label) for mode in modes]
    if len(labels) != len(set(labels)):
        raise ValueError("green-function mode labels must be unique.")
    for mode in modes:
        if int(mode.mode_index) < 0 or int(mode.mode_index) >= mode_count_limit:
            raise ValueError(
                f"green-function mode {mode.label!r} index {int(mode.mode_index)} is outside the "
                f"valid range [0, {mode_count_limit})."
            )

    cfg = _core_config(config)
    reused_parent_basis = basis_elements is None
    sector_basis: tuple[QSEBasisElement, ...] = (
        tuple(result.matrices.basis_elements) if reused_parent_basis else tuple(basis_elements)
    )
    psi, _, nq = normalize_statevector(np.asarray(prepared_state, dtype=complex).reshape(-1))
    cache: dict[str, Any] = {}
    h_psi = np.asarray(
        _apply_polynomial_operator(
            hamiltonian, psi, nq=int(nq), name="hamiltonian", config=cfg, pauli_action_cache=cache
        ),
        dtype=complex,
    ).reshape(-1)
    reference_energy = float(complex(np.vdot(psi, h_psi)).real)
    eta = float(kernel_config.eta)
    grid_values = grid.values()

    sector_policy_payload = {
        "sector_projection": "identity",
        "reference_projection": "none",
    }

    mode_payloads: list[dict[str, Any]] = []
    solved_sector_count = 0
    zero_source_sector_count = 0
    sector_count = 0
    for mode in modes:
        sector_records: dict[str, dict[str, Any]] = {}
        mode_green = np.zeros(grid_values.size, dtype=complex)
        residue_sum = 0.0 + 0.0j
        source_norms_squared: dict[str, float] = {}
        zero_flags: dict[str, bool] = {}
        for operation in ("addition", "removal"):
            sector_count += 1
            source = jw_ladder_source_state(
                psi, mode=mode, operation=operation, expected_nq=int(nq)
            )
            norm = float(source.source_norm)
            source_norms_squared[operation] = norm * norm
            zero_source = norm <= _ZERO_SOURCE_NORM_TOLERANCE
            zero_flags[operation] = bool(zero_source)
            sector_green = np.zeros(grid_values.size, dtype=complex)
            roots_payload: list[dict[str, Any]] = []
            if zero_source:
                zero_source_sector_count += 1
                status = "skipped_zero_source"
            else:
                solved_sector_count += 1
                status = "solved_source_specific_qse_sector"
                for root in _solve_source_sector(
                    hamiltonian=hamiltonian,
                    source=source,
                    basis_elements=sector_basis,
                    config=cfg,
                ):
                    energy = float(root["energy"])
                    amplitude = complex(root["amplitude"])
                    residue = float(abs(amplitude) ** 2)
                    offset = energy - reference_energy
                    pole = offset if operation == "addition" else -offset
                    residue_sum += residue
                    sector_green += residue / (grid_values - pole + 1j * eta)
                    roots_payload.append(
                        {
                            "sector_energy": energy,
                            "energy_offset_from_reference": offset,
                            "retarded_pole_omega": pole,
                            "residue": _complex_pair(complex(residue)),
                        }
                    )
            mode_green += sector_green
            sector_records[operation] = {
                "operation": operation,
                "source_norm": norm,
                "zero_source_sector": bool(zero_source),
                "qse_solve_status": status,
                "sector_policy": dict(sector_policy_payload),
                "roots": roots_payload,
                "retarded_green_function": {
                    "values": [_complex_pair(complex(value)) for value in sector_green]
                },
            }

        spectral_function = -np.imag(mode_green) / math.pi
        norm_total = source_norms_squared["addition"] + source_norms_squared["removal"]
        mode_payloads.append(
            {
                "label": str(mode.label),
                "mode_index": int(mode.mode_index),
                "addition": sector_records["addition"],
                "removal": sector_records["removal"],
                "retarded_green_function": {
                    "values": [_complex_pair(complex(value)) for value in mode_green]
                },
                "diagonal_spectral_function": {
                    "values": [float(value) for value in spectral_function]
                },
                "diagonal_sum_rule_diagnostics": {
                    "status": "evaluated",
                    "addition_source_norm_squared": source_norms_squared["addition"],
                    "removal_source_norm_squared": source_norms_squared["removal"],
                    "total_residue_sum": _complex_pair(residue_sum),
                    "source_norm_canonical_deficit_abs": abs(1.0 - norm_total),
                    "residue_canonical_deficit_abs": abs(1.0 - complex(residue_sum)),
                    "zero_source_sectors": dict(zero_flags),
                },
            }
        )

    return {
        "schema_version": GREEN_FUNCTION_SCHEMA_VERSION,
        "policy": "diagnostic_only_single_particle_green_function_postprocessing",
        "response_kind": "single_particle_retarded_green_function_diagonal",
        "complex_scalar_encoding": "array_real_imag",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "sector_policy": {
            "source_specific_qse_solves": True,
            "neutral_qse_matrices_reused_for_green_sectors": False,
            "explicit_particle_number_projection": False,
            "operator_basis_reused_from_parent_qse": bool(reused_parent_basis),
        },
        "mode_domain": {
            "fermion_mode_count": mode_count_limit,
            "fermion_mode_count_source": "caller_supplied",
        },
        "frequency_convention": {
            "reference_energy": reference_energy,
            "omega_zero": "reference_energy",
            "retarded_prescription": "omega_plus_i_eta",
        },
        "frequency_grid": grid.to_manifest(),
        "kernel": {"name": str(kernel_config.kernel), "eta": eta},
        "summary": {
            "mode_count": len(mode_payloads),
            "sector_count": sector_count,
            "solved_sector_count": solved_sector_count,
            "zero_source_sector_count": zero_source_sector_count,
        },
        "modes": mode_payloads,
    }
