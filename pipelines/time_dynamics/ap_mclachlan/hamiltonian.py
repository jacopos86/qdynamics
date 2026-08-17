"""Neutral time-dependent Hamiltonian provider for AP-McLachlan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from pipelines.time_dynamics.adapters.drive_terms import (
    count_nonidentity_pauli_terms,
    resolve_realtime_drive_model,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


TIME_DEPENDENT_HAMILTONIAN_SCHEMA_V1 = "ap_mclachlan_time_dependent_hamiltonian_v1"
STATIC_HAMILTONIAN_PARITY_V1 = "static_hamiltonian_pauli_coeff_parity_v1"


@dataclass(frozen=True)
class TimeDependentHamiltonian:
    """Provider for ``H(t) = H_static + c(t) D``.

    The provider is intentionally family-neutral.  Hubbard-Holstein, Hubbard,
    spin-boson, or molecular details must enter through the shared resolved
    problem and drive adapter, not through AP-specific code paths.
    """

    static_poly: Any
    drive_model: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def drive_enabled(self) -> bool:
        return self.drive_model is not None

    # -- dense operator cache -------------------------------------------
    #
    # ``H(t) = H_static + c(t) D`` has two time-independent operators and one
    # scalar envelope, so the dense matrices can be built once per provider and
    # recombined per checkpoint.  Rebuilding them from Pauli terms on every call
    # dominated ``geometry.hamiltonian_action`` in the AP-McLachlan profile.
    #
    # This is sound only because the provider is frozen and its polynomials are
    # owned for its lifetime.  Mutating ``static_poly`` or the drive model's
    # ``drive_poly`` after construction would invalidate the cache; nothing in
    # the AP route does so.

    def _dense_static(self) -> np.ndarray:
        cached = self.__dict__.get("_dense_static_cache")
        if cached is None:
            cached = np.asarray(hamiltonian_matrix(self.static_poly), dtype=complex)
            cached.setflags(write=False)
            object.__setattr__(self, "_dense_static_cache", cached)
        return cached

    def _dense_drive(self) -> np.ndarray | None:
        if self.drive_model is None:
            return None
        cached = self.__dict__.get("_dense_drive_cache")
        if cached is None:
            cached = np.asarray(
                hamiltonian_matrix(getattr(self.drive_model, "drive_poly")),
                dtype=complex,
            )
            cached.setflags(write=False)
            object.__setattr__(self, "_dense_drive_cache", cached)
        return cached

    @property
    def drive_poly(self) -> Any | None:
        if self.drive_model is None:
            return None
        return getattr(self.drive_model, "drive_poly")

    def drive_coefficient_at(self, time: float) -> float:
        if self.drive_model is None:
            return 0.0
        return float(self.drive_model.coefficient_at(float(time)))

    def polynomial_at(self, time: float) -> Any:
        static = _clone_polynomial(self.static_poly)
        if self.drive_model is None:
            return static
        coeff = self.drive_coefficient_at(float(time))
        return static + float(coeff) * getattr(self.drive_model, "drive_poly")

    def matrix_at(self, time: float) -> np.ndarray:
        r"""Return the dense ``H(t) = H_static + c(t) D``.

        Callers receive a fresh writable array, as they did when this rebuilt
        the matrix from Pauli terms on every call.

        The cached recombination is not bitwise identical to rebuilding the
        combined polynomial: it sums two dense matrices instead of summing
        per-Pauli coefficients before the dense mapping, so results differ at
        floating-point rounding, and a drive term whose *scaled* coefficient
        straddles the ``1e-12`` term tolerance may be kept here where the
        rebuild dropped it. Both effects are bounded by ~1e-12 absolute. The
        ``c(t) == 0`` branch is exactly equal to the static matrix, which keeps
        the ``A=0`` drive parity requirement intact.
        """

        static = self._dense_static()
        drive = self._dense_drive()
        if drive is None:
            return np.array(static, dtype=complex)
        coeff = self.drive_coefficient_at(float(time))
        if coeff == 0.0:
            return np.array(static, dtype=complex)
        if static.shape != drive.shape:
            # Degenerate polynomials (e.g. an empty static operator) fall back
            # to the uncached construction rather than guessing a broadcast.
            return np.asarray(
                hamiltonian_matrix(self.polynomial_at(float(time))), dtype=complex
            )
        return np.asarray(static + float(coeff) * drive, dtype=complex)

    def to_json_dict(self) -> dict[str, Any]:
        drive_model = self.drive_model
        return {
            "schema": TIME_DEPENDENT_HAMILTONIAN_SCHEMA_V1,
            "drive_enabled": bool(self.drive_enabled),
            "static_nonidentity_term_count": int(count_nonidentity_pauli_terms(self.static_poly)),
            "drive_nonidentity_term_count": (
                0 if drive_model is None else int(getattr(drive_model, "drive_term_count", 0))
            ),
            "drive_family_key": None if drive_model is None else str(getattr(drive_model, "family_key", "")),
            "drive_operator_label": None if drive_model is None else str(getattr(drive_model, "operator_label", "")),
            "drive_profile": (
                {}
                if drive_model is None
                else _json_safe(dict(getattr(drive_model, "profile_payload", {}) or {}))
            ),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


def _clone_polynomial(poly: Any) -> Any:
    return 1.0 * poly


def _poly_coeff_map(poly: Any, *, tol: float = 1.0e-12) -> dict[str, complex]:
    coeff_map: dict[str, complex] = {}
    for term in tuple(poly.return_polynomial()):
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        coeff_map[label] = coeff_map.get(label, 0.0 + 0.0j) + coeff
    return {
        str(label): complex(coeff)
        for label, coeff in coeff_map.items()
        if abs(complex(coeff)) > float(tol)
    }


def _hamiltonian_parity_payload(
    *,
    left: Any,
    right: Any,
    left_label: str,
    right_label: str,
    atol: float,
) -> dict[str, Any]:
    left_map = _poly_coeff_map(left)
    right_map = _poly_coeff_map(right)
    labels = sorted(set(left_map) | set(right_map))
    deltas = {
        label: abs(left_map.get(label, 0.0 + 0.0j) - right_map.get(label, 0.0 + 0.0j))
        for label in labels
    }
    max_label = None if not deltas else max(deltas, key=lambda label: float(deltas[label]))
    max_abs_delta = 0.0 if max_label is None else float(deltas[max_label])
    mismatch_labels = [
        label
        for label in labels
        if float(deltas[label]) > float(atol)
    ]
    return {
        "schema": STATIC_HAMILTONIAN_PARITY_V1,
        "left_label": str(left_label),
        "right_label": str(right_label),
        "atol": float(atol),
        "left_term_count": int(len(left_map)),
        "right_term_count": int(len(right_map)),
        "max_abs_delta": float(max_abs_delta),
        "max_delta_label": max_label,
        "mismatch_count": int(len(mismatch_labels)),
        "mismatch_preview": [str(label) for label in mismatch_labels[:8]],
        "passed": bool(len(mismatch_labels) == 0),
    }


def assert_static_hamiltonian_parity(
    runtime_input: Any,
    *,
    static_poly: Any | None = None,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    """Fail closed if the seed/static Hamiltonian and resolved problem disagree."""

    seed_static = getattr(runtime_input, "h_poly", None) if static_poly is None else static_poly
    resolved_problem = getattr(runtime_input, "resolved_problem", None)
    resolved_static = getattr(resolved_problem, "hamiltonian", None)
    if seed_static is None:
        raise ValueError("Static Hamiltonian parity gate requires runtime_input.h_poly.")
    if resolved_static is None:
        raise ValueError(
            "Static Hamiltonian parity gate requires runtime_input.resolved_problem.hamiltonian."
        )
    payload = _hamiltonian_parity_payload(
        left=seed_static,
        right=resolved_static,
        left_label="runtime_input.h_poly",
        right_label="runtime_input.resolved_problem.hamiltonian",
        atol=float(atol),
    )
    if not bool(payload["passed"]):
        raise ValueError(
            "Static Hamiltonian parity check failed: runtime seed/static Hamiltonian "
            "does not match resolved dynamics Hamiltonian "
            f"(max_abs_delta={payload['max_abs_delta']:.3e}, "
            f"mismatch_count={payload['mismatch_count']}, "
            f"preview={payload['mismatch_preview']})."
        )
    return payload


def assert_zero_drive_static_parity(
    provider: TimeDependentHamiltonian,
    *,
    atol: float = 1.0e-10,
) -> dict[str, Any] | None:
    """When an enabled drive has A=0, require H(t) to equal the static Hamiltonian."""

    drive_model = provider.drive_model
    if drive_model is None:
        return None
    drive_a = getattr(drive_model, "drive_A", None)
    if drive_a is None or abs(float(drive_a)) > float(atol):
        return None
    payload = _hamiltonian_parity_payload(
        left=provider.static_poly,
        right=provider.polynomial_at(0.0),
        left_label="static_poly",
        right_label="polynomial_at_t0_with_zero_drive",
        atol=float(atol),
    )
    if not bool(payload["passed"]):
        raise ValueError(
            "A=0 drive parity check failed: driven Hamiltonian does not reduce "
            "to the static Hamiltonian "
            f"(max_abs_delta={payload['max_abs_delta']:.3e}, "
            f"mismatch_count={payload['mismatch_count']}, "
            f"preview={payload['mismatch_preview']})."
        )
    return payload


def time_dependent_hamiltonian_from_runtime_input(
    runtime_input: Any,
    *,
    drive_config: Any | None = None,
    drive_model: Any | None = None,
    parity_atol: float = 1.0e-10,
) -> TimeDependentHamiltonian:
    if drive_config is not None and drive_model is not None:
        raise ValueError("Provide either drive_config or drive_model, not both.")
    static_poly = getattr(runtime_input, "h_poly")
    static_parity = assert_static_hamiltonian_parity(
        runtime_input,
        static_poly=static_poly,
        atol=float(parity_atol),
    )
    resolved_drive_model = drive_model
    if drive_config is not None:
        resolved_drive_model = resolve_realtime_drive_model(
            resolved_problem=getattr(runtime_input, "resolved_problem"),
            drive_config=drive_config,
        )
    provider = TimeDependentHamiltonian(
        static_poly=static_poly,
        drive_model=resolved_drive_model,
        metadata={
            "runtime_provenance": dict(getattr(runtime_input, "provenance", {}) or {}),
            "static_hamiltonian_parity": static_parity,
            "drive_source": (
                "none"
                if resolved_drive_model is None
                else ("provided_drive_model" if drive_model is not None else "resolve_realtime_drive_model")
            ),
        },
    )
    zero_drive_parity = assert_zero_drive_static_parity(
        provider,
        atol=float(parity_atol),
    )
    if zero_drive_parity is None:
        return provider
    return TimeDependentHamiltonian(
        static_poly=provider.static_poly,
        drive_model=provider.drive_model,
        metadata={
            **dict(provider.metadata or {}),
            "zero_drive_static_parity": zero_drive_parity,
        },
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "STATIC_HAMILTONIAN_PARITY_V1",
    "TIME_DEPENDENT_HAMILTONIAN_SCHEMA_V1",
    "TimeDependentHamiltonian",
    "assert_static_hamiltonian_parity",
    "assert_zero_drive_static_parity",
    "time_dependent_hamiltonian_from_runtime_input",
]
