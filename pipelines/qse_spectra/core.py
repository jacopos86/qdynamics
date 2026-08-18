"""Ideal/statevector Quantum Subspace Expansion spectra helpers.

This package is intentionally isolated from the repo's existing ADAPT and
realtime routes.  It reuses only the shared PauliPolynomial and compiled Pauli
statevector primitives, and keeps all labels internally in the repo's exyz
convention with words ordered q_(n-1) ... q_0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli,
    compile_pauli_action_exyz,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


_VALID_INTERNAL_PAULIS = set("exyz")
_INPUT_PAULI_TRANSLATION = {
    "e": "e",
    "E": "e",
    "i": "e",
    "I": "e",
    "x": "x",
    "X": "x",
    "y": "y",
    "Y": "y",
    "z": "z",
    "Z": "z",
}


@dataclass(frozen=True)
class QSEBasisElement:
    """One operator used to generate a QSE vector ``B_i |psi>``."""

    name: str
    kind: str
    pauli_label_exyz: str | None = None
    polynomial: PauliPolynomial | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in {"pauli_string", "pauli_polynomial"}:
            raise ValueError("QSEBasisElement.kind must be 'pauli_string' or 'pauli_polynomial'.")
        if str(self.name).strip() == "":
            raise ValueError("QSEBasisElement.name must be non-empty.")
        if kind == "pauli_string" and self.pauli_label_exyz is None:
            raise ValueError("pauli_string basis elements require pauli_label_exyz.")
        if kind == "pauli_polynomial" and self.polynomial is None:
            raise ValueError("pauli_polynomial basis elements require polynomial.")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("QSEBasisElement.metadata must be a mapping when supplied.")


@dataclass(frozen=True)
class QSEBasisVectorPolicy:
    """Policy for constructing QSE basis vectors from ``B_i |psi>`` records."""

    reference_projection: str = "none"
    basis_vector_normalization: str = "normalized"
    sector_projection: str = "identity"
    sector_label: str | None = None

    def __post_init__(self) -> None:
        if str(self.reference_projection) not in {"none", "q0"}:
            raise ValueError("reference_projection must be 'none' or 'q0'.")
        if str(self.basis_vector_normalization) not in {"normalized", "raw_projected"}:
            raise ValueError("basis_vector_normalization must be 'normalized' or 'raw_projected'.")
        if str(self.sector_projection) != "identity":
            raise ValueError("Only identity sector_projection is supported in this sidecar slice.")
        if self.sector_label is not None and str(self.sector_label).strip() == "":
            raise ValueError("sector_label must be non-empty when supplied.")


@dataclass(frozen=True)
class QSEBasisVectorDiagnostics:
    """Per-record diagnostics for source, projection, normalization, and Q0 removal."""

    basis_index: int
    name: str
    kind: str
    reference_projection: str
    basis_vector_normalization: str
    sector_projection: str
    sector_label: str | None
    raw_action_norm: float
    projected_norm: float
    matrix_vector_norm: float
    reference_overlap_before_projection: complex
    reference_overlap_after_projection: complex
    reference_overlap_before_projection_abs: float
    reference_overlap_after_projection_abs: float
    normalized_for_matrices: bool
    zero_vector: bool
    projected_out_by_q0: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QSEObservable:
    """Observable used for QSE transition matrix/vector calculations."""

    name: str
    kind: str
    pauli_label_exyz: str | None = None
    polynomial: PauliPolynomial | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in {"pauli_string", "pauli_polynomial"}:
            raise ValueError("QSEObservable.kind must be 'pauli_string' or 'pauli_polynomial'.")
        if str(self.name).strip() == "":
            raise ValueError("QSEObservable.name must be non-empty.")
        if kind == "pauli_string" and self.pauli_label_exyz is None:
            raise ValueError("pauli_string observables require pauli_label_exyz.")
        if kind == "pauli_polynomial" and self.polynomial is None:
            raise ValueError("pauli_polynomial observables require polynomial.")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("QSEObservable.metadata must be a mapping when supplied.")


@dataclass(frozen=True)
class QSETransitionObservableResult:
    """Transition-observable data for one QSE observable."""

    observable: QSEObservable
    observable_matrix: np.ndarray
    transition_vector: np.ndarray
    transition_amplitudes: np.ndarray
    transition_strengths: np.ndarray
    observable_matrix_hermitian_residual_max_abs: float | None


@dataclass(frozen=True)
class QSEPruningConfig:
    """Numerical tolerances for overlap pruning and safety checks."""

    overlap_relative_cutoff: float = 1.0e-10
    overlap_absolute_cutoff: float = 1.0e-12
    overlap_negative_absolute_tolerance: float = 1.0e-12
    overlap_negative_relative_tolerance: float = 1.0e-9
    hermitian_absolute_tolerance: float = 1.0e-10
    hermitian_relative_tolerance: float = 1.0e-8
    hamiltonian_coeff_imag_absolute_tolerance: float = 1.0e-12
    polynomial_drop_abs_tol: float = 1.0e-15

    def __post_init__(self) -> None:
        values = {
            "overlap_relative_cutoff": self.overlap_relative_cutoff,
            "overlap_absolute_cutoff": self.overlap_absolute_cutoff,
            "overlap_negative_absolute_tolerance": self.overlap_negative_absolute_tolerance,
            "overlap_negative_relative_tolerance": self.overlap_negative_relative_tolerance,
            "hermitian_absolute_tolerance": self.hermitian_absolute_tolerance,
            "hermitian_relative_tolerance": self.hermitian_relative_tolerance,
            "hamiltonian_coeff_imag_absolute_tolerance": self.hamiltonian_coeff_imag_absolute_tolerance,
            "polynomial_drop_abs_tol": self.polynomial_drop_abs_tol,
        }
        for key, raw in values.items():
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")
            if value < 0.0:
                raise ValueError(f"{key} must be non-negative.")
        if float(self.overlap_relative_cutoff) == 0.0 and float(self.overlap_absolute_cutoff) == 0.0:
            raise ValueError("At least one overlap cutoff must be positive.")


@dataclass(frozen=True)
class QSEMatrices:
    nq: int
    hilbert_dim: int
    basis_elements: tuple[QSEBasisElement, ...]
    reference_energy: float
    reference_energy_imag_abs: float
    basis_vector_norms: tuple[float, ...]
    overlap: np.ndarray
    hamiltonian: np.ndarray
    overlap_hermitian_residual_max_abs_raw: float
    hamiltonian_hermitian_residual_max_abs_raw: float
    hamiltonian_coeff_imag_max_abs: float
    basis_vector_policy: QSEBasisVectorPolicy = field(default_factory=QSEBasisVectorPolicy)
    basis_action_norms: tuple[float, ...] = ()
    basis_projected_norms: tuple[float, ...] = ()
    basis_matrix_vector_norms: tuple[float, ...] = ()
    basis_vector_diagnostics: tuple[QSEBasisVectorDiagnostics, ...] = ()
    basis_matrix_vectors: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class QSEResult:
    matrices: QSEMatrices
    eigenvalues: np.ndarray
    eigenvectors_basis: np.ndarray
    overlap_eigenvalues_raw: np.ndarray
    overlap_eigenvalues_clamped: np.ndarray
    retained_overlap_indices: tuple[int, ...]
    overlap_pruning_threshold: float
    retained_rank: int
    discarded_rank: int
    overlap_condition_estimate: float | None
    overlap_min_eigenvalue_raw: float
    overlap_max_eigenvalue_raw: float
    generalized_residual_norms: tuple[float, ...]
    solver_status: str
    transition_observables: tuple[QSETransitionObservableResult, ...] = ()


@dataclass(frozen=True)
class _CleanPolynomial:
    polynomial: PauliPolynomial
    nq: int
    coeff_imag_max_abs: float
    retained_term_count: int


@dataclass(frozen=True)
class _PreparedBasisVectors:
    matrix_vectors: tuple[np.ndarray, ...]
    diagnostics: tuple[QSEBasisVectorDiagnostics, ...]
    raw_action_norms: tuple[float, ...]
    projected_norms: tuple[float, ...]
    matrix_vector_norms: tuple[float, ...]


def _config(config: QSEPruningConfig | None) -> QSEPruningConfig:
    return config if config is not None else QSEPruningConfig()


def _is_power_of_two(value: int) -> bool:
    value_i = int(value)
    return value_i > 0 and (value_i & (value_i - 1)) == 0


def _infer_nq_from_dim(dim: int) -> int:
    dim_i = int(dim)
    if not _is_power_of_two(dim_i):
        raise ValueError(f"Statevector length must be a power of two; got {dim_i}.")
    return int(math.log2(dim_i))


def _normalize_pauli_label(label: str, *, nq: int | None = None) -> str:
    raw = str(label)
    try:
        normalized = "".join(_INPUT_PAULI_TRANSLATION[ch] for ch in raw)
    except KeyError as exc:
        raise ValueError(f"Unsupported Pauli symbol {exc.args[0]!r} in label {raw!r}.") from exc
    if nq is not None and len(normalized) != int(nq):
        raise ValueError(f"Pauli label {raw!r} has length {len(normalized)}; expected {int(nq)}.")
    return normalized


def _validate_internal_label(label: str, *, nq: int) -> str:
    label_s = str(label)
    if len(label_s) != int(nq):
        raise ValueError(f"Pauli label {label_s!r} has length {len(label_s)}; expected {int(nq)}.")
    bad = sorted(set(label_s) - _VALID_INTERNAL_PAULIS)
    if bad:
        raise ValueError(f"Pauli label {label_s!r} contains unsupported internal symbols {bad!r}.")
    return label_s


def _finite_complex_array(vec: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(np.real(vec))) or not np.all(np.isfinite(np.imag(vec))):
        raise ValueError(f"{name} contains non-finite values.")


def normalize_statevector(state: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Return ``(normalized_state, original_norm, nq)`` for a dense statevector."""

    psi = np.asarray(state, dtype=complex).reshape(-1)
    if psi.size == 0:
        raise ValueError("statevector must be non-empty.")
    _finite_complex_array(psi, name="statevector")
    nq = _infer_nq_from_dim(int(psi.size))
    norm = float(np.linalg.norm(psi))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("statevector norm must be positive and finite.")
    return np.asarray(psi / norm, dtype=complex), norm, int(nq)


def computational_basis_state(nq: int, bitstring: str) -> np.ndarray:
    """Return ``|bitstring>`` using q_(n-1)...q_0 bitstring ordering."""

    nq_i = int(nq)
    if nq_i <= 0:
        raise ValueError("nq must be positive.")
    bits = str(bitstring).strip()
    if len(bits) != nq_i:
        raise ValueError(f"bitstring length {len(bits)} does not match nq={nq_i}.")
    if set(bits) - {"0", "1"}:
        raise ValueError(f"bitstring must contain only 0/1 symbols; got {bits!r}.")
    psi = np.zeros(1 << nq_i, dtype=complex)
    psi[int(bits, 2)] = 1.0 + 0.0j
    return psi


def pauli_string_basis_element(
    label: str,
    *,
    nq: int,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QSEBasisElement:
    """Create a Pauli-string QSE basis element from an input-boundary label."""

    label_exyz = _normalize_pauli_label(label, nq=int(nq))
    return QSEBasisElement(
        name=str(name) if name is not None else label_exyz,
        kind="pauli_string",
        pauli_label_exyz=label_exyz,
        metadata=metadata,
    )


def _polynomial_nq(poly: PauliPolynomial) -> int:
    terms = list(poly.return_polynomial())
    if not terms:
        raise ValueError("PauliPolynomial must contain at least one term.")
    nq = int(terms[0].nqubit())
    for term in terms:
        term_nq = int(term.nqubit())
        if term_nq != nq:
            raise ValueError(f"Inconsistent PauliPolynomial qubit count: expected {nq}, got {term_nq}.")
        _validate_internal_label(str(term.pw2strng()), nq=nq)
    return nq


def polynomial_basis_element(
    poly: PauliPolynomial,
    *,
    name: str,
    metadata: Mapping[str, Any] | None = None,
) -> QSEBasisElement:
    """Create a polynomial QSE basis element without mutating ``poly``."""

    _polynomial_nq(poly)
    return QSEBasisElement(name=str(name), kind="pauli_polynomial", polynomial=poly, metadata=metadata)


def pauli_string_observable(
    label: str,
    *,
    nq: int,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QSEObservable:
    """Create a Pauli-string transition observable from an input-boundary label."""

    label_exyz = _normalize_pauli_label(label, nq=int(nq))
    return QSEObservable(
        name=str(name) if name is not None else label_exyz,
        kind="pauli_string",
        pauli_label_exyz=label_exyz,
        metadata=metadata,
    )


def polynomial_observable(
    poly: PauliPolynomial,
    *,
    name: str,
    metadata: Mapping[str, Any] | None = None,
) -> QSEObservable:
    """Create a polynomial transition observable without mutating ``poly``.

    An empty polynomial (all terms dropped, e.g. an explicit zero current
    source) is accepted; its application yields the zero vector.
    """

    if list(poly.return_polynomial()):
        _polynomial_nq(poly)
    return QSEObservable(name=str(name), kind="pauli_polynomial", polynomial=poly, metadata=metadata)


def _clean_polynomial_terms(
    poly: PauliPolynomial,
    *,
    drop_abs_tol: float,
    require_real_coefficients: bool,
    coeff_imag_abs_tol: float,
    allow_empty_after_pruning: bool = False,
) -> _CleanPolynomial:
    terms = list(poly.return_polynomial())
    if not terms:
        raise ValueError("PauliPolynomial must contain at least one term.")
    nq = int(terms[0].nqubit())
    coeff_by_label: dict[str, complex] = {}
    order: list[str] = []
    for term in terms:
        if int(term.nqubit()) != nq:
            raise ValueError(f"Inconsistent PauliPolynomial qubit count: expected {nq}, got {term.nqubit()}.")
        label = _validate_internal_label(str(term.pw2strng()), nq=nq)
        coeff = complex(term.p_coeff)
        if label not in coeff_by_label:
            order.append(label)
            coeff_by_label[label] = 0.0 + 0.0j
        coeff_by_label[label] += coeff

    out = PauliPolynomial("JW")
    coeff_imag_max_abs = 0.0
    retained = 0
    for label in order:
        coeff = complex(coeff_by_label[label])
        if abs(coeff) <= float(drop_abs_tol):
            continue
        coeff_imag_max_abs = max(coeff_imag_max_abs, abs(float(coeff.imag)))
        if require_real_coefficients:
            if abs(float(coeff.imag)) > float(coeff_imag_abs_tol):
                raise ValueError(
                    f"Hamiltonian coefficient for {label!r} has imaginary part {coeff.imag}, "
                    f"exceeding tolerance {coeff_imag_abs_tol}."
                )
            coeff = float(coeff.real) + 0.0j
        out.add_term(PauliTerm(nq, ps=label, pc=coeff))
        retained += 1

    if retained == 0 and not bool(allow_empty_after_pruning):
        raise ValueError("PauliPolynomial has no retained terms after coefficient pruning.")
    return _CleanPolynomial(
        polynomial=out,
        nq=int(nq),
        coeff_imag_max_abs=float(coeff_imag_max_abs),
        retained_term_count=int(retained),
    )


def _apply_pauli_label(
    label_exyz: str,
    psi: np.ndarray,
    *,
    nq: int,
    pauli_action_cache: dict[str, CompiledPauliAction],
) -> np.ndarray:
    label = _validate_internal_label(str(label_exyz), nq=int(nq))
    action = pauli_action_cache.get(label)
    if action is None:
        action = compile_pauli_action_exyz(label, int(nq))
        pauli_action_cache[label] = action
    return apply_compiled_pauli(psi, action)


def _apply_polynomial_operator(
    poly: PauliPolynomial,
    psi: np.ndarray,
    *,
    nq: int,
    name: str,
    config: QSEPruningConfig,
    pauli_action_cache: dict[str, CompiledPauliAction],
) -> np.ndarray:
    if not list(poly.return_polynomial()):
        return np.zeros_like(psi, dtype=complex)
    clean = _clean_polynomial_terms(
        poly,
        drop_abs_tol=float(config.polynomial_drop_abs_tol),
        require_real_coefficients=False,
        coeff_imag_abs_tol=float(config.hamiltonian_coeff_imag_absolute_tolerance),
        allow_empty_after_pruning=True,
    )
    if int(clean.retained_term_count) == 0:
        return np.zeros_like(psi, dtype=complex)
    if int(clean.nq) != int(nq):
        raise ValueError(f"Operator {name!r} has nq={clean.nq}; expected {nq}.")
    compiled = compile_polynomial_action(
        clean.polynomial,
        tol=float(config.polynomial_drop_abs_tol),
        pauli_action_cache=pauli_action_cache,
    )
    return apply_compiled_polynomial(psi, compiled)


def _apply_basis_element(
    basis: QSEBasisElement,
    psi: np.ndarray,
    *,
    nq: int,
    config: QSEPruningConfig,
    pauli_action_cache: dict[str, CompiledPauliAction],
) -> np.ndarray:
    if basis.kind == "pauli_string":
        return _apply_pauli_label(
            str(basis.pauli_label_exyz),
            psi,
            nq=int(nq),
            pauli_action_cache=pauli_action_cache,
        )

    if basis.kind == "pauli_polynomial":
        if basis.polynomial is None:
            raise ValueError(f"Basis element {basis.name!r} is missing its polynomial.")
        return _apply_polynomial_operator(
            basis.polynomial,
            psi,
            nq=int(nq),
            name=str(basis.name),
            config=config,
            pauli_action_cache=pauli_action_cache,
        )

    raise ValueError(f"Unsupported QSE basis kind {basis.kind!r}.")


def _apply_observable(
    observable: QSEObservable,
    psi: np.ndarray,
    *,
    nq: int,
    config: QSEPruningConfig,
    pauli_action_cache: dict[str, CompiledPauliAction],
) -> np.ndarray:
    if observable.kind == "pauli_string":
        return _apply_pauli_label(
            str(observable.pauli_label_exyz),
            psi,
            nq=int(nq),
            pauli_action_cache=pauli_action_cache,
        )

    if observable.kind == "pauli_polynomial":
        if observable.polynomial is None:
            raise ValueError(f"Observable {observable.name!r} is missing its polynomial.")
        return _apply_polynomial_operator(
            observable.polynomial,
            psi,
            nq=int(nq),
            name=str(observable.name),
            config=config,
            pauli_action_cache=pauli_action_cache,
        )

    raise ValueError(f"Unsupported QSE observable kind {observable.kind!r}.")


def _public_statevector(
    state: np.ndarray,
    *,
    normalize_state: bool,
    expected_nq: int | None,
) -> tuple[np.ndarray, int]:
    if bool(normalize_state):
        psi, _, nq = normalize_statevector(state)
    else:
        psi = np.asarray(state, dtype=complex).reshape(-1)
        if psi.size == 0:
            raise ValueError("statevector must be non-empty.")
        _finite_complex_array(psi, name="statevector")
        nq = _infer_nq_from_dim(int(psi.size))
    if expected_nq is not None and int(nq) != int(expected_nq):
        raise ValueError(f"statevector has nq={nq}; expected {int(expected_nq)}.")
    return np.asarray(psi, dtype=complex).reshape(-1), int(nq)


def apply_qse_observable(
    observable: QSEObservable,
    state: np.ndarray,
    *,
    config: QSEPruningConfig | None = None,
    normalize_state: bool = True,
    expected_nq: int | None = None,
) -> np.ndarray:
    """Apply a QSE observable to a dense statevector using repo Pauli conventions.

    The helper is a public, minimal wrapper around the same compiled-Pauli path
    used by QSE transition observables.  By default the input state is normalized
    before application, matching the transition/response postprocessing paths.
    """

    cfg = _config(config)
    psi, nq = _public_statevector(state, normalize_state=bool(normalize_state), expected_nq=expected_nq)
    pauli_action_cache: dict[str, CompiledPauliAction] = {}
    out = _apply_observable(
        observable,
        psi,
        nq=int(nq),
        config=cfg,
        pauli_action_cache=pauli_action_cache,
    )
    out_vec = np.asarray(out, dtype=complex).reshape(-1)
    _finite_complex_array(out_vec, name=f"observable {observable.name} applied state")
    return out_vec


def expect_qse_observable(
    observable: QSEObservable,
    state: np.ndarray,
    *,
    config: QSEPruningConfig | None = None,
    normalize_state: bool = True,
    expected_nq: int | None = None,
) -> complex:
    """Return ``<state|O|state>`` for a QSE observable as a complex scalar."""

    cfg = _config(config)
    psi, nq = _public_statevector(state, normalize_state=bool(normalize_state), expected_nq=expected_nq)
    pauli_action_cache: dict[str, CompiledPauliAction] = {}
    opsi = _apply_observable(
        observable,
        psi,
        nq=int(nq),
        config=cfg,
        pauli_action_cache=pauli_action_cache,
    )
    out = complex(np.vdot(psi, np.asarray(opsi, dtype=complex).reshape(-1)))
    if not math.isfinite(float(out.real)) or not math.isfinite(float(out.imag)):
        raise ValueError(f"Observable expectation for {observable.name!r} is non-finite: {out!r}.")
    return out


def _basis_vector_policy(policy: QSEBasisVectorPolicy | None) -> QSEBasisVectorPolicy:
    return policy if policy is not None else QSEBasisVectorPolicy()


def _basis_metadata_with_policy(basis: QSEBasisElement, policy: QSEBasisVectorPolicy) -> dict[str, Any]:
    metadata = dict(basis.metadata) if basis.metadata is not None else {}
    if policy.sector_label is not None and "sector_label" not in metadata:
        metadata["sector_label"] = str(policy.sector_label)
    return metadata


def _prepare_basis_vectors(
    basis_elements: Sequence[QSEBasisElement],
    psi: np.ndarray,
    *,
    nq: int,
    config: QSEPruningConfig,
    policy: QSEBasisVectorPolicy,
    pauli_action_cache: dict[str, CompiledPauliAction],
) -> _PreparedBasisVectors:
    matrix_vectors: list[np.ndarray] = []
    diagnostics: list[QSEBasisVectorDiagnostics] = []
    raw_action_norms: list[float] = []
    projected_norms: list[float] = []
    matrix_vector_norms: list[float] = []

    for idx, basis in enumerate(basis_elements):
        raw = _apply_basis_element(
            basis,
            psi,
            nq=int(nq),
            config=config,
            pauli_action_cache=pauli_action_cache,
        )
        _finite_complex_array(raw, name=f"basis vector {basis.name}")
        raw_vec = np.asarray(raw, dtype=complex).reshape(-1)
        raw_norm = float(np.linalg.norm(raw_vec))
        reference_overlap_before = complex(np.vdot(psi, raw_vec))

        if policy.reference_projection == "q0":
            projected_vec = raw_vec - np.asarray(psi, dtype=complex).reshape(-1) * reference_overlap_before
        else:
            projected_vec = raw_vec.copy()
        reference_overlap_after = complex(np.vdot(psi, projected_vec))
        projected_norm = float(np.linalg.norm(projected_vec))

        normalized_for_matrices = False
        if policy.basis_vector_normalization == "normalized":
            if projected_norm > 0.0:
                matrix_vec = projected_vec / projected_norm
                normalized_for_matrices = True
            else:
                matrix_vec = projected_vec.copy()
        elif policy.basis_vector_normalization == "raw_projected":
            matrix_vec = projected_vec.copy()
        else:
            raise ValueError(f"Unsupported basis_vector_normalization {policy.basis_vector_normalization!r}.")

        matrix_norm = float(np.linalg.norm(matrix_vec))
        zero_vector = bool(matrix_norm == 0.0)
        projected_out_tolerance = 10.0 * np.finfo(float).eps * max(1.0, raw_norm)
        projected_out_by_q0 = bool(
            policy.reference_projection == "q0"
            and raw_norm > 0.0
            and projected_norm <= projected_out_tolerance
        )
        metadata = _basis_metadata_with_policy(basis, policy)
        sector_label = metadata.get("sector_label", policy.sector_label)
        if sector_label is not None:
            sector_label = str(sector_label)

        matrix_vectors.append(np.asarray(matrix_vec, dtype=complex))
        raw_action_norms.append(float(raw_norm))
        projected_norms.append(float(projected_norm))
        matrix_vector_norms.append(float(matrix_norm))
        diagnostics.append(
            QSEBasisVectorDiagnostics(
                basis_index=int(idx),
                name=str(basis.name),
                kind=str(basis.kind),
                reference_projection=str(policy.reference_projection),
                basis_vector_normalization=str(policy.basis_vector_normalization),
                sector_projection=str(policy.sector_projection),
                sector_label=sector_label,
                raw_action_norm=float(raw_norm),
                projected_norm=float(projected_norm),
                matrix_vector_norm=float(matrix_norm),
                reference_overlap_before_projection=reference_overlap_before,
                reference_overlap_after_projection=reference_overlap_after,
                reference_overlap_before_projection_abs=float(abs(reference_overlap_before)),
                reference_overlap_after_projection_abs=float(abs(reference_overlap_after)),
                normalized_for_matrices=bool(normalized_for_matrices),
                zero_vector=zero_vector,
                projected_out_by_q0=projected_out_by_q0,
                metadata=metadata,
            )
        )

    return _PreparedBasisVectors(
        matrix_vectors=tuple(matrix_vectors),
        diagnostics=tuple(diagnostics),
        raw_action_norms=tuple(float(x) for x in raw_action_norms),
        projected_norms=tuple(float(x) for x in projected_norms),
        matrix_vector_norms=tuple(float(x) for x in matrix_vector_norms),
    )


def _matrix_residual_max_abs(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix - matrix.conj().T)))


def _max_abs_entry(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix)))


def _hermitian_allowed(matrix: np.ndarray, config: QSEPruningConfig) -> float:
    return max(
        float(config.hermitian_absolute_tolerance),
        float(config.hermitian_relative_tolerance) * max(1.0, _max_abs_entry(matrix)),
    )


def build_qse_matrices(
    hamiltonian: PauliPolynomial,
    prepared_state: np.ndarray,
    basis_elements: Sequence[QSEBasisElement],
    *,
    config: QSEPruningConfig | None = None,
    basis_vector_policy: QSEBasisVectorPolicy | None = None,
) -> QSEMatrices:
    """Build projected QSE overlap and Hamiltonian matrices."""

    cfg = _config(config)
    policy = _basis_vector_policy(basis_vector_policy)
    psi, _, nq = normalize_statevector(prepared_state)
    clean_h = _clean_polynomial_terms(
        hamiltonian,
        drop_abs_tol=float(cfg.polynomial_drop_abs_tol),
        require_real_coefficients=True,
        coeff_imag_abs_tol=float(cfg.hamiltonian_coeff_imag_absolute_tolerance),
    )
    if int(clean_h.nq) != int(nq):
        raise ValueError(f"Hamiltonian has nq={clean_h.nq}; state has nq={nq}.")

    basis_tuple = tuple(basis_elements)
    if len(basis_tuple) == 0:
        raise ValueError("At least one QSE basis element is required.")

    pauli_action_cache: dict[str, CompiledPauliAction] = {}
    compiled_h = compile_polynomial_action(
        clean_h.polynomial,
        tol=float(cfg.polynomial_drop_abs_tol),
        pauli_action_cache=pauli_action_cache,
    )

    hpsi = apply_compiled_polynomial(psi, compiled_h)
    reference_energy_complex = complex(np.vdot(psi, hpsi))
    reference_energy = float(reference_energy_complex.real)
    reference_energy_imag_abs = abs(float(reference_energy_complex.imag))

    prepared = _prepare_basis_vectors(
        basis_tuple,
        psi,
        nq=int(nq),
        config=cfg,
        policy=policy,
        pauli_action_cache=pauli_action_cache,
    )
    phi_matrix = np.column_stack(prepared.matrix_vectors)
    overlap_raw = phi_matrix.conj().T @ phi_matrix
    hamiltonian_raw = np.zeros((len(basis_tuple), len(basis_tuple)), dtype=complex)
    for col in range(len(basis_tuple)):
        hphi = apply_compiled_polynomial(phi_matrix[:, col], compiled_h)
        hamiltonian_raw[:, col] = phi_matrix.conj().T @ hphi

    _finite_complex_array(overlap_raw.reshape(-1), name="QSE overlap matrix")
    _finite_complex_array(hamiltonian_raw.reshape(-1), name="QSE Hamiltonian matrix")

    overlap_residual = _matrix_residual_max_abs(overlap_raw)
    hamiltonian_residual = _matrix_residual_max_abs(hamiltonian_raw)
    overlap_allowed = _hermitian_allowed(overlap_raw, cfg)
    hamiltonian_allowed = _hermitian_allowed(hamiltonian_raw, cfg)
    if overlap_residual > overlap_allowed:
        raise ValueError(
            f"Projected overlap matrix is non-Hermitian: residual {overlap_residual} > {overlap_allowed}."
        )
    if hamiltonian_residual > hamiltonian_allowed:
        raise ValueError(
            f"Projected Hamiltonian matrix is non-Hermitian: residual {hamiltonian_residual} > {hamiltonian_allowed}."
        )

    overlap = 0.5 * (overlap_raw + overlap_raw.conj().T)
    hamiltonian_qse = 0.5 * (hamiltonian_raw + hamiltonian_raw.conj().T)

    return QSEMatrices(
        nq=int(nq),
        hilbert_dim=int(1 << int(nq)),
        basis_elements=basis_tuple,
        reference_energy=float(reference_energy),
        reference_energy_imag_abs=float(reference_energy_imag_abs),
        basis_vector_norms=tuple(float(x) for x in prepared.projected_norms),
        overlap=overlap,
        hamiltonian=hamiltonian_qse,
        overlap_hermitian_residual_max_abs_raw=float(overlap_residual),
        hamiltonian_hermitian_residual_max_abs_raw=float(hamiltonian_residual),
        hamiltonian_coeff_imag_max_abs=float(clean_h.coeff_imag_max_abs),
        basis_vector_policy=policy,
        basis_action_norms=tuple(float(x) for x in prepared.raw_action_norms),
        basis_projected_norms=tuple(float(x) for x in prepared.projected_norms),
        basis_matrix_vector_norms=tuple(float(x) for x in prepared.matrix_vector_norms),
        basis_vector_diagnostics=prepared.diagnostics,
        basis_matrix_vectors=prepared.matrix_vectors,
    )


def solve_qse_generalized_eigenproblem(
    matrices: QSEMatrices,
    *,
    config: QSEPruningConfig | None = None,
) -> QSEResult:
    """Solve ``H c = E S c`` by Löwdin overlap pruning."""

    cfg = _config(config)
    overlap = np.asarray(matrices.overlap, dtype=complex)
    hamiltonian = np.asarray(matrices.hamiltonian, dtype=complex)
    _finite_complex_array(overlap.reshape(-1), name="QSE overlap matrix")
    _finite_complex_array(hamiltonian.reshape(-1), name="QSE Hamiltonian matrix")
    if overlap.ndim != 2 or hamiltonian.ndim != 2:
        raise ValueError("Hamiltonian and overlap matrices must be 2D arrays.")
    if overlap.shape[0] != overlap.shape[1]:
        raise ValueError("Overlap matrix must be square.")
    if hamiltonian.shape != overlap.shape:
        raise ValueError("Hamiltonian and overlap matrix shapes must match.")
    if overlap.shape[0] == 0:
        raise ValueError("Cannot solve an empty QSE generalized eigenproblem.")
    overlap_residual = _matrix_residual_max_abs(overlap)
    hamiltonian_residual = _matrix_residual_max_abs(hamiltonian)
    if overlap_residual > _hermitian_allowed(overlap, cfg):
        raise ValueError("Overlap matrix is non-Hermitian beyond configured tolerance.")
    if hamiltonian_residual > _hermitian_allowed(hamiltonian, cfg):
        raise ValueError("Hamiltonian matrix is non-Hermitian beyond configured tolerance.")
    overlap = 0.5 * (overlap + overlap.conj().T)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)

    s_raw, u = np.linalg.eigh(overlap)
    s_raw = np.asarray(s_raw, dtype=float)
    max_abs_s = float(np.max(np.abs(s_raw))) if s_raw.size else 0.0
    negative_tol = max(
        float(cfg.overlap_negative_absolute_tolerance),
        float(cfg.overlap_negative_relative_tolerance) * max_abs_s,
    )
    min_raw = float(np.min(s_raw))
    max_raw = float(np.max(s_raw))
    if min_raw < -negative_tol:
        raise ValueError(f"Overlap matrix has negative eigenvalue {min_raw} below tolerance {-negative_tol}.")

    s_clamped = np.where(s_raw < 0.0, 0.0, s_raw)
    max_clamped = float(np.max(s_clamped)) if s_clamped.size else 0.0
    threshold = max(
        float(cfg.overlap_absolute_cutoff),
        float(cfg.overlap_relative_cutoff) * max_clamped,
    )
    if threshold <= 0.0:
        raise ValueError("Overlap pruning threshold must be positive.")

    retained_mask = s_clamped >= threshold
    retained_indices = tuple(int(i) for i in np.nonzero(retained_mask)[0])
    if not retained_indices:
        raise ValueError(
            f"QSE overlap retained rank is zero; max overlap eigenvalue is {max_clamped}, threshold is {threshold}."
        )

    s_retained = s_clamped[list(retained_indices)]
    u_retained = u[:, list(retained_indices)]
    x_map = u_retained @ np.diag(1.0 / np.sqrt(s_retained))
    h_orth = x_map.conj().T @ hamiltonian @ x_map
    h_orth = 0.5 * (h_orth + h_orth.conj().T)

    evals, evecs_orth = np.linalg.eigh(h_orth)
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=float)
    evecs_orth = evecs_orth[:, order]
    coeffs = x_map @ evecs_orth

    residuals: list[float] = []
    for idx, energy in enumerate(evals):
        c = coeffs[:, idx]
        residual = hamiltonian @ c - float(energy) * (overlap @ c)
        residuals.append(float(np.linalg.norm(residual)))

    condition = None
    if s_retained.size > 0:
        condition = float(np.max(s_retained) / np.min(s_retained))

    return QSEResult(
        matrices=matrices,
        eigenvalues=evals,
        eigenvectors_basis=np.asarray(coeffs, dtype=complex),
        overlap_eigenvalues_raw=s_raw,
        overlap_eigenvalues_clamped=np.asarray(s_clamped, dtype=float),
        retained_overlap_indices=retained_indices,
        overlap_pruning_threshold=float(threshold),
        retained_rank=int(len(retained_indices)),
        discarded_rank=int(overlap.shape[0] - len(retained_indices)),
        overlap_condition_estimate=condition,
        overlap_min_eigenvalue_raw=float(min_raw),
        overlap_max_eigenvalue_raw=float(max_raw),
        generalized_residual_norms=tuple(float(x) for x in residuals),
        solver_status="lowdin_overlap_pruned_eigh",
    )


def compute_transition_observables(
    result: QSEResult,
    transition_source_state: np.ndarray,
    observables: Sequence[QSEObservable],
    *,
    config: QSEPruningConfig | None = None,
) -> tuple[QSETransitionObservableResult, ...]:
    """Compute QSE transition vectors, amplitudes, and strengths.

    The QSE basis vectors and Ritz eigenvectors are read from ``result``.  The
    source state is supplied explicitly so response, conductivity, and Green-
    function layers can evaluate transitions from a state that is distinct from
    the state used to build the QSE matrices.
    """

    if not observables:
        return ()
    cfg = _config(config)
    matrices = result.matrices
    if len(matrices.basis_matrix_vectors) != len(matrices.basis_elements):
        raise ValueError("Transition observables require QSE basis matrix vectors from build_qse_matrices().")

    source_state, _, nq = normalize_statevector(transition_source_state)
    if int(nq) != int(matrices.nq):
        raise ValueError(f"Transition state has nq={nq}; matrices have nq={matrices.nq}.")
    phi_matrix = np.column_stack(tuple(np.asarray(v, dtype=complex).reshape(-1) for v in matrices.basis_matrix_vectors))
    if phi_matrix.shape[0] != int(matrices.hilbert_dim):
        raise ValueError("QSE basis matrix vector dimension does not match matrices.hilbert_dim.")

    out: list[QSETransitionObservableResult] = []
    pauli_action_cache: dict[str, CompiledPauliAction] = {}
    coeffs = np.asarray(result.eigenvectors_basis, dtype=complex)
    for observable in tuple(observables):
        opsi = _apply_observable(
            observable,
            source_state,
            nq=int(nq),
            config=cfg,
            pauli_action_cache=pauli_action_cache,
        )
        _finite_complex_array(opsi, name=f"transition observable {observable.name} applied to source")
        transition_vector = phi_matrix.conj().T @ np.asarray(opsi, dtype=complex).reshape(-1)

        observable_matrix = np.zeros((len(matrices.basis_elements), len(matrices.basis_elements)), dtype=complex)
        for col in range(len(matrices.basis_elements)):
            ophi = _apply_observable(
                observable,
                phi_matrix[:, col],
                nq=int(nq),
                config=cfg,
                pauli_action_cache=pauli_action_cache,
            )
            _finite_complex_array(ophi, name=f"transition observable {observable.name} matrix column {col}")
            observable_matrix[:, col] = phi_matrix.conj().T @ np.asarray(ophi, dtype=complex).reshape(-1)

        _finite_complex_array(observable_matrix.reshape(-1), name=f"transition observable {observable.name} matrix")
        _finite_complex_array(transition_vector.reshape(-1), name=f"transition observable {observable.name} vector")
        amplitudes = coeffs.conj().T @ transition_vector
        strengths = np.abs(amplitudes) ** 2
        out.append(
            QSETransitionObservableResult(
                observable=observable,
                observable_matrix=np.asarray(observable_matrix, dtype=complex),
                transition_vector=np.asarray(transition_vector, dtype=complex),
                transition_amplitudes=np.asarray(amplitudes, dtype=complex),
                transition_strengths=np.asarray(strengths, dtype=float),
                observable_matrix_hermitian_residual_max_abs=float(_matrix_residual_max_abs(observable_matrix)),
            )
        )
    return tuple(out)


def compute_qse_spectra(
    hamiltonian: PauliPolynomial,
    prepared_state: np.ndarray,
    basis_elements: Sequence[QSEBasisElement],
    *,
    config: QSEPruningConfig | None = None,
    basis_vector_policy: QSEBasisVectorPolicy | None = None,
    transition_observables: Sequence[QSEObservable] | None = None,
) -> QSEResult:
    """Build QSE matrices and return the overlap-pruned Ritz spectrum."""

    cfg = _config(config)
    matrices = build_qse_matrices(
        hamiltonian,
        prepared_state,
        basis_elements,
        config=cfg,
        basis_vector_policy=basis_vector_policy,
    )
    result = solve_qse_generalized_eigenproblem(matrices, config=cfg)
    transition_results = compute_transition_observables(
        result,
        prepared_state,
        tuple(transition_observables or ()),
        config=cfg,
    )
    if transition_results:
        result = replace(result, transition_observables=transition_results)
    return result


__all__ = [
    "QSEBasisElement",
    "QSEBasisVectorDiagnostics",
    "QSEBasisVectorPolicy",
    "QSEObservable",
    "QSETransitionObservableResult",
    "QSEPruningConfig",
    "QSEMatrices",
    "QSEResult",
    "normalize_statevector",
    "computational_basis_state",
    "pauli_string_basis_element",
    "polynomial_basis_element",
    "pauli_string_observable",
    "polynomial_observable",
    "apply_qse_observable",
    "expect_qse_observable",
    "build_qse_matrices",
    "solve_qse_generalized_eigenproblem",
    "compute_transition_observables",
    "compute_qse_spectra",
]
