"""Exact diagnostic spectral references for Paper-III QSE runs.

This module builds same-cutoff exact spectra from a serialized Hamiltonian and
prepared state, then emits the lightweight reference format consumed by
``pipelines.qse_spectra`` spectral-window postprocessing.  The output is
diagnostic/reporting-only and must not feed controller decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import QSEObservable, QSEPruningConfig, _apply_observable
from pipelines.qse_spectra.io import (
    load_polynomial_json,
    load_state_json,
    load_transition_observables_json,
    transition_observables_from_labels,
)
from pipelines.qse_spectra.spectral_functions import (
    BroadeningKernelConfig,
    SpectralGrid,
    evaluate_broadening_kernel,
)
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.pauli_actions import CompiledPauliAction
from src.quantum.pauli_polynomial_class import PauliPolynomial


EXACT_REFERENCE_SCHEMA_VERSION = "qse_exact_spectral_reference_v1"
EXACT_REFERENCE_PIPELINE = "qse_exact_spectral_reference"
_CONTROLLER_BOUNDARY = {
    "feeds_controller_decisions": False,
    "controller_usable": False,
    "post_run_diagnostic_only": True,
}


@dataclass(frozen=True)
class ExactSpectralReferenceConfig:
    hamiltonian_json: Path
    state_json: Path
    state_json_key: str = "auto"
    transition_observable_labels: tuple[str, ...] = ()
    transition_observable_jsons: tuple[Path, ...] = ()
    grid: SpectralGrid = SpectralGrid(omega_min=0.0, omega_max=4.0, num_points=201)
    kernel: BroadeningKernelConfig = BroadeningKernelConfig(kernel="lorentzian", eta=0.05)
    output_json: Path = Path("qse_exact_spectral_reference.json")
    polynomial_drop_abs_tol: float = 1.0e-15
    hamiltonian_coeff_imag_absolute_tolerance: float = 1.0e-12


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _polynomial_nq(poly: PauliPolynomial) -> int:
    terms = list(poly.return_polynomial())
    if not terms:
        raise ValueError("Hamiltonian polynomial is empty.")
    nq = int(terms[0].nqubit())
    for term in terms:
        if int(term.nqubit()) != int(nq):
            raise ValueError("Hamiltonian polynomial has inconsistent qubit counts.")
    return int(nq)


def _dense_hamiltonian(poly: PauliPolynomial, *, nq: int, drop_abs_tol: float) -> np.ndarray:
    dim = 1 << int(nq)
    compiled = compile_polynomial_action(poly, tol=float(drop_abs_tol), pauli_action_cache={})
    out = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[col] = 1.0
        out[:, col] = apply_compiled_polynomial(basis, compiled)
    return out


def _observable_action(
    observable: QSEObservable,
    state: np.ndarray,
    *,
    nq: int,
    config: QSEPruningConfig,
) -> np.ndarray:
    cache: dict[str, CompiledPauliAction] = {}
    return _apply_observable(
        observable,
        state,
        nq=int(nq),
        config=config,
        pauli_action_cache=cache,
    )


def _json_safe_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; got {out!r}.")
    return out


def _load_observables(config: ExactSpectralReferenceConfig, *, nq: int) -> tuple[QSEObservable, ...]:
    observables: list[QSEObservable] = []
    if config.transition_observable_labels:
        observables.extend(
            transition_observables_from_labels(config.transition_observable_labels, nq=int(nq))
        )
    for path in config.transition_observable_jsons:
        loaded, _ = load_transition_observables_json(path, nq=int(nq))
        observables.extend(loaded)
    if not observables:
        raise ValueError("At least one transition observable is required.")
    return tuple(observables)


def build_exact_spectral_reference(config: ExactSpectralReferenceConfig) -> dict[str, Any]:
    hamiltonian, h_provenance = load_polynomial_json(
        config.hamiltonian_json,
        drop_abs_tol=float(config.polynomial_drop_abs_tol),
        require_real_coefficients=True,
        coeff_imag_abs_tol=float(config.hamiltonian_coeff_imag_absolute_tolerance),
    )
    nq = _polynomial_nq(hamiltonian)
    state, state_provenance = load_state_json(
        config.state_json,
        expected_nq=int(nq),
        state_key=str(config.state_json_key),
    )
    observables = _load_observables(config, nq=int(nq))
    qse_config = QSEPruningConfig(
        polynomial_drop_abs_tol=float(config.polynomial_drop_abs_tol),
        hamiltonian_coeff_imag_absolute_tolerance=float(config.hamiltonian_coeff_imag_absolute_tolerance),
    )
    hmat = _dense_hamiltonian(
        hamiltonian,
        nq=int(nq),
        drop_abs_tol=float(config.polynomial_drop_abs_tol),
    )
    hermitian_residual = float(np.max(np.abs(hmat - hmat.conj().T))) if hmat.size else 0.0
    if hermitian_residual > 1.0e-8:
        raise ValueError(f"Exact-reference Hamiltonian is not Hermitian; residual={hermitian_residual}.")
    evals, evecs = np.linalg.eigh(hmat)
    reference_energy = complex(np.vdot(state, hmat @ state))
    if abs(reference_energy.imag) > 1.0e-8:
        raise ValueError(f"Prepared-state reference energy has non-negligible imaginary part {reference_energy.imag}.")

    grid_values = np.asarray(config.grid.values(), dtype=float)
    references: list[dict[str, Any]] = []
    roots_by_observable: list[dict[str, Any]] = []
    for observable in observables:
        o_state = _observable_action(observable, state, nq=int(nq), config=qse_config)
        amplitudes = evecs.conj().T @ o_state
        strengths = np.abs(amplitudes) ** 2
        omegas = np.asarray(evals - float(reference_energy.real), dtype=float)
        values = np.zeros_like(grid_values, dtype=float)
        roots: list[dict[str, float | int]] = []
        for idx, (energy, omega, strength) in enumerate(zip(evals, omegas, strengths, strict=True)):
            values += float(strength) * evaluate_broadening_kernel(
                grid_values - float(omega),
                config.kernel,
            )
            roots.append(
                {
                    "state_index": int(idx),
                    "energy": _json_safe_float(energy, name="energy"),
                    "omega": _json_safe_float(omega, name="omega"),
                    "transition_strength": _json_safe_float(strength, name="transition_strength"),
                }
            )
        references.append(
            {
                "observable_name": str(observable.name),
                "label": "same_cutoff_exact",
                "grid": [float(x) for x in grid_values],
                "values": [float(x) for x in values],
                "metadata": {
                    "reference_kind": "same_cutoff_exact_diagonalization",
                    "num_qubits": int(nq),
                    "hilbert_dim": int(1 << int(nq)),
                    "reference_energy": float(reference_energy.real),
                    "kernel": config.kernel.to_manifest(),
                },
            }
        )
        roots_by_observable.append(
            {
                "observable_name": str(observable.name),
                "roots": roots,
            }
        )

    return {
        "schema_version": EXACT_REFERENCE_SCHEMA_VERSION,
        "pipeline": EXACT_REFERENCE_PIPELINE,
        "generated_utc": _utc_now(),
        "policy": "diagnostic_only_same_cutoff_exact_spectral_reference",
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
        "input": {
            "hamiltonian": h_provenance,
            "state": state_provenance,
        },
        "settings": {
            "grid": config.grid.to_manifest(include_values=False),
            "kernel": config.kernel.to_manifest(),
            "polynomial_drop_abs_tol": float(config.polynomial_drop_abs_tol),
            "hamiltonian_coeff_imag_absolute_tolerance": float(
                config.hamiltonian_coeff_imag_absolute_tolerance
            ),
        },
        "diagnostics": {
            "num_qubits": int(nq),
            "hilbert_dim": int(1 << int(nq)),
            "eigenvalue_count": int(len(evals)),
            "reference_energy": float(reference_energy.real),
            "hamiltonian_hermitian_residual_max_abs": float(hermitian_residual),
        },
        "references": references,
        "exact_roots": roots_by_observable,
    }


def write_exact_spectral_reference(config: ExactSpectralReferenceConfig) -> dict[str, Any]:
    payload = build_exact_spectral_reference(config)
    Path(config.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(config.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build diagnostic exact spectral references for QSE.")
    parser.add_argument("--hamiltonian-json", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument(
        "--state-json-key",
        choices=["auto", "initial_state", "ansatz_input_state"],
        default="auto",
    )
    parser.add_argument("--transition-observable-label", action="append", default=None)
    parser.add_argument("--transition-observable-json", action="append", type=Path, default=None)
    parser.add_argument("--spectral-grid-min", type=float, required=True)
    parser.add_argument("--spectral-grid-max", type=float, required=True)
    parser.add_argument("--spectral-grid-num", type=int, required=True)
    parser.add_argument("--spectral-eta", type=float, required=True)
    parser.add_argument("--spectral-kernel", choices=["lorentzian", "gaussian"], default="lorentzian")
    parser.add_argument("--polynomial-drop-abs-tol", type=float, default=1.0e-15)
    parser.add_argument("--hamiltonian-coeff-imag-absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExactSpectralReferenceConfig(
        hamiltonian_json=args.hamiltonian_json,
        state_json=args.state_json,
        state_json_key=str(args.state_json_key),
        transition_observable_labels=tuple(args.transition_observable_label or ()),
        transition_observable_jsons=tuple(args.transition_observable_json or ()),
        grid=SpectralGrid(
            omega_min=float(args.spectral_grid_min),
            omega_max=float(args.spectral_grid_max),
            num_points=int(args.spectral_grid_num),
        ),
        kernel=BroadeningKernelConfig(
            kernel=str(args.spectral_kernel),
            eta=float(args.spectral_eta),
        ),
        output_json=args.output_json,
        polynomial_drop_abs_tol=float(args.polynomial_drop_abs_tol),
        hamiltonian_coeff_imag_absolute_tolerance=float(args.hamiltonian_coeff_imag_absolute_tolerance),
    )
    payload = write_exact_spectral_reference(config)
    print(f"output_json: {config.output_json}")
    print(f"num_qubits: {payload['diagnostics']['num_qubits']}")
    print(f"eigenvalue_count: {payload['diagnostics']['eigenvalue_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
