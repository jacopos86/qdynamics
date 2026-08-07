"""Neutral response-function postprocessing for QSE spectra manifests.

This module consumes already-computed QSE Ritz roots and transition amplitudes
and emits an additive, versioned response payload.  It does not change the QSE
solve path and must not be used as a controller decision input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEObservable,
    QSEPruningConfig,
    QSEResult,
    QSETransitionObservableResult,
    _apply_observable,
    _clean_polynomial_terms,
    normalize_statevector,
)
from pipelines.qse_spectra.spectral_functions import (
    BroadeningKernelConfig,
    SpectralGrid,
    evaluate_broadening_kernel,
)
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.pauli_actions import CompiledPauliAction
from src.quantum.pauli_polynomial_class import PauliPolynomial


RESPONSE_FUNCTIONS_SCHEMA_VERSION = "qse_response_functions_v1"
_COMPLEX_SCALAR_CONVENTION = "[real, imag]"
_COMPLEX_SCALAR_ENCODING = "array_real_imag"
_CONTROLLER_BOUNDARY = {
    "feeds_controller_decisions": False,
    "controller_usable": False,
    "post_run_diagnostic_only": True,
}


@dataclass(frozen=True)
class ResponseTimeGrid:
    """Uniform time grid for QSE correlation reconstruction.

    Units are inverse Hamiltonian-energy units: if QSE energies are reported in
    the Hamiltonian's energy unit, then time values are in the reciprocal unit.
    """

    t_min: float
    t_max: float
    num_points: int

    def __post_init__(self) -> None:
        t_min = _finite_float(self.t_min, name="t_min")
        t_max = _finite_float(self.t_max, name="t_max")
        num_points = _strict_int(self.num_points, name="num_points", min_value=1)
        if t_min > t_max:
            raise ValueError("ResponseTimeGrid requires t_min <= t_max.")
        if num_points > 1 and not t_min < t_max:
            raise ValueError("ResponseTimeGrid requires t_min < t_max when num_points > 1.")
        object.__setattr__(self, "t_min", t_min)
        object.__setattr__(self, "t_max", t_max)
        object.__setattr__(self, "num_points", num_points)

    def values(self) -> np.ndarray:
        if int(self.num_points) == 1:
            return np.asarray([float(self.t_min)], dtype=float)
        return np.linspace(float(self.t_min), float(self.t_max), int(self.num_points), dtype=float)

    def to_manifest(self, *, include_values: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "t_min": float(self.t_min),
            "t_max": float(self.t_max),
            "num_points": int(self.num_points),
            "units": "inverse_hamiltonian_energy",
        }
        if include_values:
            out["values"] = [float(x) for x in self.values()]
        return out


@dataclass(frozen=True)
class ResponseChannel:
    """One neutral response channel pair ``(A, B)``.

    ``A_label`` and ``B_label`` refer to names of QSE transition observables
    already attached to a ``QSEResult``.  ``channel_kind`` is a code-facing label
    such as ``nn``, ``XX``, ``nX``, ``PP``, or ``custom``.
    """

    A_label: str
    B_label: str
    channel_kind: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        a_label = str(self.A_label).strip()
        b_label = str(self.B_label).strip()
        kind = str(self.channel_kind).strip() or "custom"
        if a_label == "" or b_label == "":
            raise ValueError("ResponseChannel labels must be non-empty.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("ResponseChannel.metadata must be a mapping.")
        object.__setattr__(self, "A_label", a_label)
        object.__setattr__(self, "B_label", b_label)
        object.__setattr__(self, "channel_kind", kind)
        object.__setattr__(self, "metadata", dict(self.metadata))


def parse_response_channel_spec(spec: str) -> ResponseChannel:
    """Parse CLI channel syntax ``A:B`` or ``A:B:channel_kind``."""

    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) == 2:
        a_label, b_label = parts
        channel_kind = "custom"
    elif len(parts) == 3:
        a_label, b_label, channel_kind = parts
    else:
        raise ValueError(f"Invalid response channel {spec!r}; expected A:B or A:B:channel_kind.")
    return ResponseChannel(A_label=a_label, B_label=b_label, channel_kind=channel_kind)


def parse_response_channel_specs(specs: Sequence[str] | None) -> tuple[ResponseChannel, ...]:
    """Parse zero or more CLI response channel specifications."""

    return tuple(parse_response_channel_spec(spec) for spec in tuple(specs or ()))


def build_response_functions_payload(
    result: QSEResult,
    *,
    grid: SpectralGrid,
    kernel_config: BroadeningKernelConfig,
    time_grid: ResponseTimeGrid,
    channels: Sequence[ResponseChannel] | None = None,
    max_moment_order: int = 1,
    hamiltonian: PauliPolynomial | None = None,
    prepared_state: np.ndarray | None = None,
    config: QSEPruningConfig | None = None,
    evaluate_sum_rules: bool = True,
) -> dict[str, Any]:
    """Build the additive ``qse_response_functions_v1`` payload.

    Complex scalar values in this payload use the documented ``[real, imag]``
    JSON convention.  Existing ``qse_spectra_v1`` complex dictionaries are left
    untouched for backward compatibility.
    """

    if not result.transition_observables:
        raise ValueError("Response functions require at least one QSE transition observable.")
    max_order = _strict_int(max_moment_order, name="max_moment_order", min_value=0)
    transition_by_label = _transition_results_by_label(result.transition_observables)
    channel_tuple = tuple(channels or _all_ordered_channels(tuple(transition_by_label.keys())))
    if not channel_tuple:
        raise ValueError("At least one response channel is required.")
    for channel in channel_tuple:
        if channel.A_label not in transition_by_label:
            raise ValueError(f"Response channel A_label {channel.A_label!r} has no matching transition observable.")
        if channel.B_label not in transition_by_label:
            raise ValueError(f"Response channel B_label {channel.B_label!r} has no matching transition observable.")

    grid_values = grid.values()
    time_values = time_grid.values()
    _finite_real_array(grid_values, name="response frequency grid")
    _finite_real_array(time_values, name="response time grid")
    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    _finite_real_array(energies, name="QSE eigenvalues")
    reference_energy = _finite_float(result.matrices.reference_energy, name="reference_energy")
    omegas = np.asarray(energies - reference_energy, dtype=float)
    _finite_real_array(omegas, name="QSE excitation frequencies")

    direct_context = None
    if bool(evaluate_sum_rules) and hamiltonian is not None and prepared_state is not None:
        direct_context = _build_direct_sum_rule_context(
            result,
            hamiltonian=hamiltonian,
            prepared_state=prepared_state,
            observables=[item.observable for item in transition_by_label.values()],
            config=config,
        )

    channel_payloads: list[dict[str, Any]] = []
    for channel in channel_tuple:
        transition_a = transition_by_label[channel.A_label]
        transition_b = transition_by_label[channel.B_label]
        amplitudes_a = _transition_amplitudes(transition_a, expected_shape=energies.shape)
        amplitudes_b = _transition_amplitudes(transition_b, expected_shape=energies.shape)
        residues = np.conjugate(amplitudes_a) * amplitudes_b
        _finite_complex_array(residues, name=f"response residues {channel.A_label},{channel.B_label}")

        frequency_values = _frequency_response_values(
            residues,
            omegas,
            grid_values,
            kernel_config=kernel_config,
        )
        correlation_values = _time_correlation_values(residues, omegas, time_values)
        moments = _moments(residues, omegas, max_order=max_order)
        sum_rule_moments = moments if max_order >= 1 else _moments(residues, omegas, max_order=1)
        sum_rule_payload = _sum_rule_payload(
            channel,
            moments=sum_rule_moments,
            direct_context=direct_context,
        )

        roots = []
        for state_index, (energy, omega, amp_a, amp_b, residue) in enumerate(
            zip(energies, omegas, amplitudes_a, amplitudes_b, residues, strict=True)
        ):
            roots.append(
                {
                    "state_index": int(state_index),
                    "energy": float(energy),
                    "omega": float(omega),
                    "A_amplitude": _complex_pair(amp_a),
                    "B_amplitude": _complex_pair(amp_b),
                    "residue": _complex_pair(residue),
                }
            )

        channel_payloads.append(
            {
                "A_label": str(channel.A_label),
                "B_label": str(channel.B_label),
                "A_operator_source": _observable_source(transition_a.observable),
                "B_operator_source": _observable_source(transition_b.observable),
                "A_operator_kind": str(transition_a.observable.kind),
                "B_operator_kind": str(transition_b.observable.kind),
                "channel_kind": str(channel.channel_kind),
                "metadata": _json_safe_mapping(channel.metadata),
                "roots": roots,
                "frequency_response": {
                    "quantity": "S_AB(omega)",
                    "grid_ref": "frequency_grid.values",
                    "values": [_complex_pair(value) for value in frequency_values],
                },
                "time_correlation": {
                    "quantity": "C_AB(t)",
                    "grid_ref": "time_grid.values",
                    "values": [_complex_pair(value) for value in correlation_values],
                },
                "moments": [
                    {
                        "quantity": "m_k^{AB}",
                        "order": int(order),
                        "value": _complex_pair(value),
                    }
                    for order, value in enumerate(moments)
                ],
                "sum_rule_deficits": sum_rule_payload,
            }
        )

    return {
        "schema_version": RESPONSE_FUNCTIONS_SCHEMA_VERSION,
        "policy": "diagnostic_only_neutral_response_postprocessing",
        "response_kind": "neutral",
        "complex_scalar_convention": _COMPLEX_SCALAR_CONVENTION,
        "complex_scalar_encoding": _COMPLEX_SCALAR_ENCODING,
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
        "frequency_convention": {
            "omega": "qse_energy_minus_reference_energy",
            "reference_energy": float(reference_energy),
            "matches_payload": "qse_spectral_functions_v1",
        },
        "time_convention": {
            "units": "inverse_hamiltonian_energy",
            "phase": "exp(-i * omega * t)",
        },
        "frequency_grid": grid.to_manifest(include_values=True),
        "kernel": kernel_config.to_manifest(),
        "time_grid": time_grid.to_manifest(include_values=True),
        "moment_orders": [int(order) for order in range(max_order + 1)],
        "observables": [
            {
                "label": str(label),
                "operator_source": _observable_source(transition.observable),
                "operator_kind": str(transition.observable.kind),
            }
            for label, transition in transition_by_label.items()
        ],
        "channels": channel_payloads,
    }


def _all_ordered_channels(labels: Sequence[str]) -> tuple[ResponseChannel, ...]:
    return tuple(
        ResponseChannel(A_label=a_label, B_label=b_label, channel_kind="custom")
        for a_label in labels
        for b_label in labels
    )


def _transition_results_by_label(
    transitions: Sequence[QSETransitionObservableResult],
) -> dict[str, QSETransitionObservableResult]:
    out: dict[str, QSETransitionObservableResult] = {}
    duplicates: set[str] = set()
    for transition in transitions:
        label = str(transition.observable.name)
        if label in out:
            duplicates.add(label)
        out[label] = transition
    if duplicates:
        raise ValueError(f"Response functions require unique transition observable names; duplicates: {sorted(duplicates)!r}.")
    return out


def _transition_amplitudes(transition: QSETransitionObservableResult, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    amplitudes = np.asarray(transition.transition_amplitudes, dtype=complex).reshape(-1)
    if amplitudes.shape != expected_shape:
        raise ValueError(
            f"Transition amplitudes for {transition.observable.name!r} have shape {amplitudes.shape}; "
            f"expected {expected_shape}."
        )
    _finite_complex_array(amplitudes, name=f"transition amplitudes {transition.observable.name}")
    return amplitudes


def _frequency_response_values(
    residues: np.ndarray,
    omegas: np.ndarray,
    grid_values: np.ndarray,
    *,
    kernel_config: BroadeningKernelConfig,
) -> np.ndarray:
    values = np.zeros_like(grid_values, dtype=complex)
    for omega, residue in zip(omegas, residues, strict=True):
        values += complex(residue) * evaluate_broadening_kernel(grid_values - float(omega), kernel_config)
    _finite_complex_array(values, name="frequency response values")
    return values


def _time_correlation_values(residues: np.ndarray, omegas: np.ndarray, time_values: np.ndarray) -> np.ndarray:
    values = np.zeros_like(time_values, dtype=complex)
    for omega, residue in zip(omegas, residues, strict=True):
        values += complex(residue) * np.exp(-1.0j * float(omega) * time_values)
    _finite_complex_array(values, name="time correlation values")
    return values


def _moments(residues: np.ndarray, omegas: np.ndarray, *, max_order: int) -> list[complex]:
    return [complex(np.sum((omegas ** int(order)) * residues)) for order in range(int(max_order) + 1)]


@dataclass(frozen=True)
class _DirectSumRuleContext:
    observable_vectors: Mapping[str, np.ndarray]
    hamiltonian_action_vectors: Mapping[str, np.ndarray]
    direct_state_energy: float
    direct_state_energy_imag_abs: float
    reference_energy: float


def _build_direct_sum_rule_context(
    result: QSEResult,
    *,
    hamiltonian: PauliPolynomial,
    prepared_state: np.ndarray,
    observables: Sequence[QSEObservable],
    config: QSEPruningConfig | None,
) -> _DirectSumRuleContext:
    cfg = config if config is not None else QSEPruningConfig()
    psi, _, nq = normalize_statevector(prepared_state)
    if int(nq) != int(result.matrices.nq):
        raise ValueError(f"Direct sum-rule state has nq={nq}; QSE result has nq={result.matrices.nq}.")

    clean_h = _clean_polynomial_terms(
        hamiltonian,
        drop_abs_tol=float(cfg.polynomial_drop_abs_tol),
        require_real_coefficients=True,
        coeff_imag_abs_tol=float(cfg.hamiltonian_coeff_imag_absolute_tolerance),
    )
    if int(clean_h.nq) != int(nq):
        raise ValueError(f"Direct sum-rule Hamiltonian has nq={clean_h.nq}; state has nq={nq}.")
    pauli_action_cache: dict[str, CompiledPauliAction] = {}
    compiled_h = compile_polynomial_action(
        clean_h.polynomial,
        tol=float(cfg.polynomial_drop_abs_tol),
        pauli_action_cache=pauli_action_cache,
    )
    hpsi = apply_compiled_polynomial(psi, compiled_h)
    direct_energy = complex(np.vdot(psi, hpsi))
    reference_energy = _finite_float(result.matrices.reference_energy, name="reference_energy")

    observable_vectors: dict[str, np.ndarray] = {}
    hamiltonian_action_vectors: dict[str, np.ndarray] = {}
    for observable in observables:
        label = str(observable.name)
        if label in observable_vectors:
            continue
        opsi = _apply_observable(
            observable,
            psi,
            nq=int(nq),
            config=cfg,
            pauli_action_cache=pauli_action_cache,
        )
        _finite_complex_array(opsi, name=f"direct observable vector {label}")
        hopsi = apply_compiled_polynomial(np.asarray(opsi, dtype=complex).reshape(-1), compiled_h)
        centered = np.asarray(hopsi, dtype=complex).reshape(-1) - float(reference_energy) * np.asarray(opsi, dtype=complex).reshape(-1)
        _finite_complex_array(centered, name=f"direct centered Hamiltonian observable vector {label}")
        observable_vectors[label] = np.asarray(opsi, dtype=complex).reshape(-1)
        hamiltonian_action_vectors[label] = np.asarray(centered, dtype=complex).reshape(-1)

    return _DirectSumRuleContext(
        observable_vectors=observable_vectors,
        hamiltonian_action_vectors=hamiltonian_action_vectors,
        direct_state_energy=float(direct_energy.real),
        direct_state_energy_imag_abs=abs(float(direct_energy.imag)),
        reference_energy=float(reference_energy),
    )


def _sum_rule_payload(
    channel: ResponseChannel,
    *,
    moments: Sequence[complex],
    direct_context: _DirectSumRuleContext | None,
) -> dict[str, Any]:
    if direct_context is None:
        return {
            "status": "not_evaluated",
            "reason": "direct_state_hamiltonian_or_evaluation_not_supplied",
        }
    a_vec = direct_context.observable_vectors.get(channel.A_label)
    b_vec = direct_context.observable_vectors.get(channel.B_label)
    hb_vec = direct_context.hamiltonian_action_vectors.get(channel.B_label)
    if a_vec is None or b_vec is None or hb_vec is None:
        return {
            "status": "not_evaluated",
            "reason": "channel_observable_vector_missing",
        }
    m0_target = complex(np.vdot(a_vec, b_vec))
    m1_target = complex(np.vdot(a_vec, hb_vec))
    m0_qse = complex(moments[0]) if len(moments) > 0 else 0.0 + 0.0j
    m1_qse = _moment_order_one(moments)
    return {
        "status": "evaluated",
        "target_method": "direct_state_expectations_same_hamiltonian_convention",
        "reference_energy": float(direct_context.reference_energy),
        "direct_state_energy": float(direct_context.direct_state_energy),
        "direct_state_energy_imag_abs": float(direct_context.direct_state_energy_imag_abs),
        "m0": _sum_rule_record(target=m0_target, qse=m0_qse),
        "m1": _sum_rule_record(target=m1_target, qse=m1_qse),
    }


def _moment_order_one(moments: Sequence[complex]) -> complex:
    if len(moments) > 1:
        return complex(moments[1])
    return 0.0 + 0.0j


def _sum_rule_record(*, target: complex, qse: complex) -> dict[str, Any]:
    deficit = complex(target) - complex(qse)
    return {
        "target": _complex_pair(target),
        "qse": _complex_pair(qse),
        "deficit": _complex_pair(deficit),
        "deficit_abs": float(abs(deficit)),
    }


def _observable_source(observable: QSEObservable) -> str:
    metadata = observable.metadata if isinstance(observable.metadata, Mapping) else {}
    for key in ("operator_source", "source", "source_schema"):
        value = metadata.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return str(observable.kind)


def _complex_pair(value: complex | float | int | np.generic) -> list[float]:
    value_c = complex(value)
    re = float(value_c.real)
    im = float(value_c.imag)
    if not math.isfinite(re) or not math.isfinite(im):
        raise ValueError(f"Cannot serialize non-finite complex value {value_c!r}.")
    return [re, im]


def _finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(f"{name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; got {out!r}.")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    out = int(value)
    if min_value is not None and out < int(min_value):
        raise ValueError(f"{name} must be >= {int(min_value)}; got {out}.")
    return out


def _finite_real_array(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")


def _finite_complex_array(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(np.real(values))) or not np.all(np.isfinite(np.imag(values))):
        raise ValueError(f"{name} contains non-finite values.")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    def convert(item: Any) -> Any:
        if isinstance(item, complex):
            return _complex_pair(item)
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"Cannot serialize non-finite float {item!r}.")
            return float(item)
        if isinstance(item, Mapping):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [convert(child) for child in item]
        return str(item)

    return {str(key): convert(item) for key, item in value.items()}


__all__ = [
    "RESPONSE_FUNCTIONS_SCHEMA_VERSION",
    "ResponseChannel",
    "ResponseTimeGrid",
    "build_response_functions_payload",
    "parse_response_channel_spec",
    "parse_response_channel_specs",
]
