"""Static QSE record selection sidecar helpers.

This module is intentionally pure and pre-QSE for selection: candidates are
scored only from operator structure, metadata, and input order.  Post-QSE guard
values are appended after the solve strictly as diagnostics and never feed back
into selection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEBasisElement,
    QSEResult,
    _apply_basis_element,
    _apply_observable,
    _apply_polynomial_operator,
    _config as _core_config,
    normalize_statevector,
)


STATIC_RECORD_SELECTION_SCHEMA_VERSION = "qse_static_record_selection_v1"
STATIC_RECORD_SELECTION_POLICY = "deterministic_operator_structure_only_pre_qse_screen"
_STATIC_RECORD_SELECTION_MODES = {
    "input_order",
    "cost_proxy",
    "geometry_selected",
    "compiled_cost",
}
_GEOMETRY_WEIGHT_FIELDS = (
    "geometry_metric_novelty_weight",
    "geometry_residual_weight",
    "geometry_cost_weight",
)
_VALID_INTERNAL_PAULIS = set("exyz")

GEOMETRY_SCORE_NORMALIZED_ANGULAR = "normalized_angular"
GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN = "raw_overlap_2x2"
GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN = "residualized_2x2"
GEOMETRY_SCORE_FULL_PENCIL_KY_FAN = "full_pencil_ky_fan"
GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST = (
    "residualized_2x2_then_ky_fan"
)
GEOMETRY_SCORE_MODES = frozenset(
    {
        GEOMETRY_SCORE_NORMALIZED_ANGULAR,
        GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN,
        GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN,
        GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
        GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST,
    }
)


@dataclass(frozen=True)
class StaticRecordSelectionConfig:
    """Validated static QSE record-selection policy."""

    mode: str
    max_records: int
    max_term_count: int | None = None
    max_pauli_weight: int | None = None
    min_retained_rank: int | None = None
    max_overlap_condition: float | None = None
    geometry_target_roots: int = 6
    geometry_score_mode: str = GEOMETRY_SCORE_NORMALIZED_ANGULAR
    geometry_additive_novelty_reward: bool = True
    geometry_require_supported_image: bool = False
    geometry_ky_fan_shortlist_size: int | None = None
    geometry_metric_novelty_weight: float = 0.25
    geometry_residual_weight: float = 1.0
    geometry_cost_weight: float = 1.0
    geometry_min_metric_novelty: float = 1.0e-12
    geometry_cost_discount_alpha: float | None = None
    geometry_cost_discount_floor: float = 0.05
    geometry_residual_stop: float | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode)
        if mode not in _STATIC_RECORD_SELECTION_MODES:
            raise ValueError(
                "Static record selection mode must be one of "
                f"{sorted(_STATIC_RECORD_SELECTION_MODES)!r}; got {mode!r}."
            )
        _validate_int(self.max_records, name="max_records", min_value=1)
        _validate_optional_int(self.max_term_count, name="max_term_count", min_value=1)
        _validate_optional_int(self.max_pauli_weight, name="max_pauli_weight", min_value=0)
        _validate_optional_int(self.min_retained_rank, name="min_retained_rank", min_value=0)
        if self.max_overlap_condition is not None:
            value = float(self.max_overlap_condition)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("max_overlap_condition must be finite and > 0 when supplied.")
        _validate_int(self.geometry_target_roots, name="geometry_target_roots", min_value=1)
        if str(self.geometry_score_mode) not in GEOMETRY_SCORE_MODES:
            raise ValueError(
                "geometry_score_mode must be one of "
                f"{sorted(GEOMETRY_SCORE_MODES)!r}; got {self.geometry_score_mode!r}."
            )
        if not isinstance(self.geometry_additive_novelty_reward, bool):
            raise ValueError("geometry_additive_novelty_reward must be boolean.")
        if not isinstance(self.geometry_require_supported_image, bool):
            raise ValueError("geometry_require_supported_image must be boolean.")
        if (
            str(self.geometry_score_mode)
            == GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST
        ):
            _validate_int(
                self.geometry_ky_fan_shortlist_size,
                name="geometry_ky_fan_shortlist_size",
                min_value=1,
            )
        elif self.geometry_ky_fan_shortlist_size is not None:
            raise ValueError(
                "geometry_ky_fan_shortlist_size is only valid for the "
                "residualized_2x2_then_ky_fan score mode."
            )
        for field_name in _GEOMETRY_WEIGHT_FIELDS:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and >= 0.")
        floor = float(self.geometry_min_metric_novelty)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("geometry_min_metric_novelty must be finite and >= 0.")
        if self.geometry_cost_discount_alpha is not None:
            alpha = float(self.geometry_cost_discount_alpha)
            if not math.isfinite(alpha) or alpha <= 0.0:
                raise ValueError("geometry_cost_discount_alpha must be finite and > 0 when supplied.")
        floor_value = float(self.geometry_cost_discount_floor)
        if not math.isfinite(floor_value) or floor_value <= 0.0 or floor_value > 1.0:
            raise ValueError("geometry_cost_discount_floor must be in (0, 1].")
        if self.geometry_residual_stop is not None:
            stop = float(self.geometry_residual_stop)
            if not math.isfinite(stop) or stop <= 0.0:
                raise ValueError("geometry_residual_stop must be finite and > 0 when supplied.")


@dataclass(frozen=True)
class StaticRecordCandidate:
    """Operator-structure features for one input QSE basis candidate."""

    original_basis_index: int
    name: str
    kind: str
    metadata: Mapping[str, Any]
    nq: int
    pauli_label_exyz: str | None
    term_count: int
    max_pauli_weight: int
    mean_pauli_weight: float
    support_qubit_count: int
    coefficient_l1: float
    cost_proxy: float


@dataclass(frozen=True)
class StaticRecordSelectionResult:
    """Selected basis plus full pre-QSE audit data."""

    config: StaticRecordSelectionConfig
    candidates: tuple[StaticRecordCandidate, ...]
    selected_basis_elements: tuple[QSEBasisElement, ...]
    selected_original_indices: tuple[int, ...]
    candidate_decisions: tuple[Mapping[str, Any], ...]
    compiled_costs: tuple[float, ...] | None = None
    geometry_cost_source: str | None = None
    geometry_stop: Mapping[str, Any] | None = None


def _validate_int(value: Any, *, name: str, min_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= {int(min_value)}.")
    out = int(value)
    if out < int(min_value):
        raise ValueError(f"{name} must be >= {int(min_value)}; got {out}.")
    return out


def _validate_optional_int(value: Any, *, name: str, min_value: int) -> int | None:
    if value is None:
        return None
    return _validate_int(value, name=name, min_value=int(min_value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        re = float(value.real)
        im = float(value.imag)
        if not math.isfinite(re) or not math.isfinite(im):
            raise ValueError(f"Cannot serialize non-finite complex value {value!r}.")
        return {"re": re, "im": im}
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (int, float, str, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Cannot serialize non-finite float {value!r}.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _validate_internal_label(label: str, *, nq: int | None = None) -> str:
    label_s = str(label)
    bad = sorted(set(label_s) - _VALID_INTERNAL_PAULIS)
    if bad:
        raise ValueError(f"Pauli label {label_s!r} contains unsupported internal symbols {bad!r}.")
    if nq is not None and len(label_s) != int(nq):
        raise ValueError(f"Pauli label {label_s!r} has length {len(label_s)}; expected {int(nq)}.")
    return label_s


def _pauli_weight(label: str) -> int:
    return sum(1 for symbol in str(label) if symbol != "e")


def _cost_proxy(
    *,
    nq: int,
    term_count: int,
    max_pauli_weight: int,
    mean_pauli_weight: float,
    support_qubit_count: int,
    coefficient_l1: float,
) -> float:
    denom = max(1, int(nq))
    value = (
        float(term_count)
        + 0.25 * (float(max_pauli_weight) / float(denom))
        + 0.05 * (float(mean_pauli_weight) / float(denom))
        + 0.01 * (float(support_qubit_count) / float(denom))
        + 1.0e-9 * math.log1p(float(coefficient_l1))
    )
    if not math.isfinite(value):
        raise ValueError("Static record cost proxy must be finite.")
    return float(value)


def extract_static_record_candidate(
    element: QSEBasisElement,
    *,
    original_basis_index: int,
) -> StaticRecordCandidate:
    """Extract pre-QSE operator-structure features for one candidate."""

    index = _validate_int(original_basis_index, name="original_basis_index", min_value=0)
    metadata = _json_safe(dict(element.metadata) if element.metadata is not None else {})

    if element.kind == "pauli_string":
        if element.pauli_label_exyz is None:
            raise ValueError(f"Basis element {element.name!r} is missing pauli_label_exyz.")
        label = _validate_internal_label(str(element.pauli_label_exyz))
        nq = len(label)
        weight = _pauli_weight(label)
        coefficient_l1 = 1.0
        cost = _cost_proxy(
            nq=int(nq),
            term_count=1,
            max_pauli_weight=int(weight),
            mean_pauli_weight=float(weight),
            support_qubit_count=int(weight),
            coefficient_l1=coefficient_l1,
        )
        return StaticRecordCandidate(
            original_basis_index=index,
            name=str(element.name),
            kind=str(element.kind),
            metadata=metadata,
            nq=int(nq),
            pauli_label_exyz=label,
            term_count=1,
            max_pauli_weight=int(weight),
            mean_pauli_weight=float(weight),
            support_qubit_count=int(weight),
            coefficient_l1=coefficient_l1,
            cost_proxy=cost,
        )

    if element.kind != "pauli_polynomial" or element.polynomial is None:
        raise ValueError(f"Unsupported or incomplete QSE basis element {element.name!r}.")

    terms = list(element.polynomial.return_polynomial())
    if not terms:
        raise ValueError(f"Polynomial basis element {element.name!r} contains no Pauli terms.")

    nq = int(terms[0].nqubit())
    weights: list[int] = []
    support_positions: set[int] = set()
    coefficient_l1 = 0.0
    for term_index, term in enumerate(terms):
        term_nq = int(term.nqubit())
        if term_nq != int(nq):
            raise ValueError(
                f"Polynomial basis element {element.name!r} has inconsistent nq at term {term_index}: "
                f"expected {nq}, got {term_nq}."
            )
        label = _validate_internal_label(str(term.pw2strng()), nq=int(nq))
        weight = _pauli_weight(label)
        weights.append(int(weight))
        support_positions.update(pos for pos, symbol in enumerate(label) if symbol != "e")
        coeff_abs = float(abs(complex(term.p_coeff)))
        if not math.isfinite(coeff_abs):
            raise ValueError(
                f"Polynomial basis element {element.name!r} has non-finite coefficient magnitude."
            )
        coefficient_l1 += coeff_abs

    term_count = len(terms)
    max_weight = max(weights)
    mean_weight = float(sum(weights) / float(term_count))
    support_count = int(len(support_positions))
    if not math.isfinite(coefficient_l1):
        raise ValueError(f"Polynomial basis element {element.name!r} has non-finite coefficient L1 norm.")
    cost = _cost_proxy(
        nq=int(nq),
        term_count=int(term_count),
        max_pauli_weight=int(max_weight),
        mean_pauli_weight=float(mean_weight),
        support_qubit_count=int(support_count),
        coefficient_l1=float(coefficient_l1),
    )
    return StaticRecordCandidate(
        original_basis_index=index,
        name=str(element.name),
        kind=str(element.kind),
        metadata=metadata,
        nq=int(nq),
        pauli_label_exyz=None,
        term_count=int(term_count),
        max_pauli_weight=int(max_weight),
        mean_pauli_weight=float(mean_weight),
        support_qubit_count=int(support_count),
        coefficient_l1=float(coefficient_l1),
        cost_proxy=float(cost),
    )


def _hard_rejection_reasons(
    candidate: StaticRecordCandidate,
    config: StaticRecordSelectionConfig,
) -> list[str]:
    reasons: list[str] = []
    if config.max_term_count is not None and candidate.term_count > int(config.max_term_count):
        reasons.append("max_term_count")
    if config.max_pauli_weight is not None and candidate.max_pauli_weight > int(config.max_pauli_weight):
        reasons.append("max_pauli_weight")
    return reasons


def _rank_key(candidate: StaticRecordCandidate, config: StaticRecordSelectionConfig) -> tuple[float, int] | tuple[int]:
    if config.mode == "input_order":
        return (int(candidate.original_basis_index),)
    if config.mode == "cost_proxy":
        return (float(candidate.cost_proxy), int(candidate.original_basis_index))
    raise ValueError(f"Unsupported static record selection mode {config.mode!r}.")


def _two_dimensional_ritz_gain(
    *,
    reference_state: np.ndarray,
    h_reference_state: np.ndarray,
    reference_energy: float,
    candidate_state: np.ndarray,
    h_candidate_state: np.ndarray,
    relative_cutoff: float,
    absolute_cutoff: float,
) -> float:
    """Return the overlap-aware lowering from one supported 2x2 pencil."""

    overlap = np.array(
        [
            [complex(np.vdot(reference_state, reference_state)), complex(np.vdot(reference_state, candidate_state))],
            [complex(np.vdot(candidate_state, reference_state)), complex(np.vdot(candidate_state, candidate_state))],
        ],
        dtype=complex,
    )
    projected_h = np.array(
        [
            [complex(np.vdot(reference_state, h_reference_state)), complex(np.vdot(reference_state, h_candidate_state))],
            [complex(np.vdot(candidate_state, h_reference_state)), complex(np.vdot(candidate_state, h_candidate_state))],
        ],
        dtype=complex,
    )
    overlap = 0.5 * (overlap + overlap.conj().T)
    projected_h = 0.5 * (projected_h + projected_h.conj().T)
    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    largest = float(max(float(overlap_values[-1]), 0.0))
    cutoff = max(float(absolute_cutoff), float(relative_cutoff) * largest)
    keep = overlap_values > cutoff
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    whitening = overlap_vectors[:, keep] / np.sqrt(overlap_values[keep])
    reduced = whitening.conj().T @ projected_h @ whitening
    eigenvalues = np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T))
    return max(0.0, float(reference_energy) - float(eigenvalues[0]))


@dataclass(frozen=True)
class _GeometryRoundState:
    residual_hats: tuple[np.ndarray, ...]
    max_target_residual_norm: float | None
    frame_size: int
    window_top: float | None
    target_states: tuple[np.ndarray, ...]
    target_h_states: tuple[np.ndarray, ...]
    target_energies: tuple[float, ...]
    retained_pencil: np.ndarray


def _geometry_select(
    *,
    basis_tuple: tuple[QSEBasisElement, ...],
    eligible_candidates: tuple["StaticRecordCandidate", ...],
    config: StaticRecordSelectionConfig,
    hamiltonian: Any,
    prepared_state: Any,
    qse_config: Any,
    basis_vector_policy: Any,
    transition_observables: Sequence[Any],
    compiled_costs: tuple[float, ...] | None = None,
) -> tuple[
    list["StaticRecordCandidate"],
    set[int],
    set[int],
    dict[int, dict[str, Any]],
    dict[int, float],
    dict[str, Any],
]:
    """Greedy geometry-scored selection over pre-QSE candidate images.

    For each candidate image ``w_i``, two-pass Gram--Schmidt against the
    numerically retained frame produces its novel direction ``u_i``. The
    novelty fraction is always an admissibility floor and may additionally
    enter the utility as a reward. ``geometry_score_mode`` supplies the
    spectral coordinate: normalized residual angle, a raw-candidate 2x2 Ritz
    gain, a residualized 2x2 Ritz gain, the full proposed-pencil Ky Fan gain,
    or residualized 2x2 screening followed by Ky Fan reranking of a declared
    shortlist. Energy-valued gains are normalized by one common round-local
    scale only when combining them with the dimensionless novelty reward.

    With a compiled-cost discount the selection rule is ``utility / K^alpha``;
    otherwise the configured cost coordinate is subtracted from utility.

    Candidates whose novelty falls below geometry_min_metric_novelty are
    rejected with reason "metric_novelty_floor"; novelty is non-increasing
    as the span grows, so the rejection is permanent.

    Multi-root targeting (geometry_target_roots = R): before the first
    acceptance the residual is the ground-reference residual (H - <H>)|psi>.
    Once directions are accepted, the small Ritz pencil over the accepted
    orthonormal units is solved each round and the residuals of the lowest
    min(R, K) Ritz states drive residual capture, so acquisition targets the
    whole low excitation window rather than the first root alone.
    """

    if hamiltonian is None or prepared_state is None or basis_vector_policy is None:
        raise ValueError(
            "geometry_selected static record selection requires hamiltonian, "
            "prepared_state, and basis_vector_policy context."
        )
    cfg = _core_config(qse_config)
    psi, _, nq = normalize_statevector(np.asarray(prepared_state, dtype=complex).reshape(-1))
    cache: dict[str, Any] = {}

    def _project_reference(vec: np.ndarray) -> np.ndarray:
        out = np.asarray(vec, dtype=complex).reshape(-1)
        if str(basis_vector_policy.reference_projection) == "q0":
            out = out - complex(np.vdot(psi, out)) * psi
        return out

    images: dict[int, np.ndarray] = {}
    for candidate in eligible_candidates:
        element = basis_tuple[int(candidate.original_basis_index)]
        raw = _apply_basis_element(element, psi, nq=int(nq), config=cfg, pauli_action_cache=cache)
        images[int(candidate.original_basis_index)] = _project_reference(raw)

    h_psi = np.asarray(
        _apply_polynomial_operator(
            hamiltonian, psi, nq=int(nq), name="hamiltonian", config=cfg, pauli_action_cache=cache
        ),
        dtype=complex,
    ).reshape(-1)
    e0 = float(complex(np.vdot(psi, h_psi)).real)
    residual = h_psi - complex(np.vdot(psi, h_psi)) * psi
    residual_norm = float(np.linalg.norm(residual))
    residual_hat = residual / residual_norm if residual_norm > 0.0 else None

    transition_hats: list[np.ndarray] = []
    for observable in tuple(transition_observables or ()):
        image = _project_reference(
            _apply_observable(observable, psi, nq=int(nq), config=cfg, pauli_action_cache=cache)
        )
        norm = float(np.linalg.norm(image))
        if norm > 0.0:
            transition_hats.append(image / norm)

    def _cost_value(candidate: "StaticRecordCandidate") -> float:
        if compiled_costs is not None:
            return float(compiled_costs[int(candidate.original_basis_index)])
        return float(candidate.cost_proxy)

    max_cost = max((_cost_value(c) for c in eligible_candidates), default=1.0)
    if max_cost <= 0.0:
        max_cost = 1.0

    accepted_units: list[np.ndarray] = []
    accepted_h_units: list[np.ndarray] = []
    accepted_images: list[np.ndarray] = []
    retention_cutoff = float(getattr(cfg, "overlap_relative_cutoff", 1.0e-10))
    selected: list[StaticRecordCandidate] = []

    def _rebuild_retained_frame() -> None:
        # Frame = numerically RETAINED principal directions of the selected
        # images under the same relative overlap cutoff the QSE solver uses.
        # Measuring novelty against the exact span would count directions
        # that are technically present but carried at overlap weights the
        # pencil's stabilization will discard — exactly the failure mode
        # where a support "covers" a root direction yet the solve loses it.
        accepted_units.clear()
        accepted_h_units.clear()
        if not accepted_images:
            return
        count = len(accepted_images)
        gram = np.empty((count, count), dtype=complex)
        for row in range(count):
            for col in range(count):
                gram[row, col] = complex(np.vdot(accepted_images[row], accepted_images[col]))
        gram = 0.5 * (gram + gram.conj().T)
        eigvals, eigvecs = np.linalg.eigh(gram)
        cutoff = retention_cutoff * float(max(eigvals.max(), 0.0))
        for position in range(count):
            if eigvals[position] <= cutoff:
                continue
            coefficients = eigvecs[:, position] / math.sqrt(float(eigvals[position]))
            unit_vec = np.zeros_like(accepted_images[0])
            for k in range(count):
                unit_vec = unit_vec + complex(coefficients[k]) * accepted_images[k]
            h_unit = np.asarray(
                _apply_polynomial_operator(
                    hamiltonian,
                    unit_vec,
                    nq=int(nq),
                    name="hamiltonian",
                    config=cfg,
                    pauli_action_cache=cache,
                ),
                dtype=complex,
            ).reshape(-1)
            accepted_units.append(unit_vec)
            accepted_h_units.append(h_unit)
    floor_rejected: set[int] = set()
    supported_image_rejected: set[int] = set()
    geometry_rows: dict[int, dict[str, Any]] = {}
    scores: dict[int, float] = {}
    pool: dict[int, StaticRecordCandidate] = {
        int(c.original_basis_index): c for c in eligible_candidates
    }

    def _round_state() -> _GeometryRoundState:
        """Return residual and Ritz data from one shared retained frame.

        The max residual norm over the lowest min(R, frame) Ritz roots is the
        natural convergence measure: for Hermitian pencils each Ritz value's
        error is bounded by its residual norm, so it plays the role ADAPT's
        gradient norm plays for ground states.
        """

        if not accepted_units:
            return _GeometryRoundState(
                residual_hats=(() if residual_hat is None else (residual_hat,)),
                max_target_residual_norm=None,
                frame_size=0,
                window_top=None,
                target_states=(psi,),
                target_h_states=(h_psi,),
                target_energies=(float(e0),),
                retained_pencil=np.empty((0, 0), dtype=complex),
            )
        count = len(accepted_units)
        pencil = np.empty((count, count), dtype=complex)
        for row in range(count):
            for col in range(count):
                pencil[row, col] = complex(np.vdot(accepted_units[row], accepted_h_units[col]))
        pencil = 0.5 * (pencil + pencil.conj().T)
        thetas, ritz_vectors = np.linalg.eigh(pencil)
        hats: list[np.ndarray] = []
        states: list[np.ndarray] = []
        h_states: list[np.ndarray] = []
        energies: list[float] = []
        max_norm = 0.0
        for root in range(min(int(config.geometry_target_roots), count)):
            coefficients = ritz_vectors[:, root]
            state = sum(complex(coefficients[k]) * accepted_units[k] for k in range(count))
            h_state = sum(complex(coefficients[k]) * accepted_h_units[k] for k in range(count))
            ritz_residual = h_state - float(thetas[root]) * state
            ritz_residual = ritz_residual - complex(np.vdot(psi, ritz_residual)) * psi
            norm = float(np.linalg.norm(ritz_residual))
            max_norm = max(max_norm, norm)
            states.append(np.asarray(state, dtype=complex))
            h_states.append(np.asarray(h_state, dtype=complex))
            energies.append(float(thetas[root]))
            if norm > 1.0e-12:
                hats.append(ritz_residual / norm)
        if not hats and residual_hat is not None:
            hats = [residual_hat]
        window_top = float(thetas[min(int(config.geometry_target_roots), count) - 1])
        return _GeometryRoundState(
            residual_hats=tuple(hats),
            max_target_residual_norm=float(max_norm),
            frame_size=int(count),
            window_top=float(window_top),
            target_states=tuple(states),
            target_h_states=tuple(h_states),
            target_energies=tuple(energies),
            retained_pencil=np.asarray(pencil, dtype=complex),
        )

    stop_reason = "budget_reached"
    last_max_residual: float | None = None
    pending_convergence = False
    round_trace: list[dict[str, Any]] = []
    while pool and len(selected) < int(config.max_records):
        spectral_state = _round_state()
        round_residuals = spectral_state.residual_hats
        last_max_residual = spectral_state.max_target_residual_norm
        frame_size = spectral_state.frame_size
        window_top = spectral_state.window_top
        pending_convergence = (
            config.geometry_residual_stop is not None
            and last_max_residual is not None
            and frame_size >= int(config.geometry_target_roots)
            and last_max_residual < float(config.geometry_residual_stop)
        )
        round_scores: dict[int, float] = {}
        round_units: dict[int, np.ndarray] = {}
        round_h_units: dict[int, np.ndarray] = {}
        round_thetas: dict[int, float] = {}
        round_features: dict[int, dict[str, float]] = {}
        round_spectral_raw: dict[int, float] = {}
        round_score_diagnostics: dict[int, dict[str, Any]] = {}
        for index, candidate in list(pool.items()):
            w = images[index]
            w_norm_sq = float(np.vdot(w, w).real)
            supported_image_admissible = bool(
                w_norm_sq > float(cfg.overlap_absolute_cutoff)
            )
            if w_norm_sq > 0.0 and (
                supported_image_admissible or not config.geometry_require_supported_image
            ):
                projected = w.copy()
                # Two projection passes (reorthogonalized Gram-Schmidt):
                # a single pass lets numerical noise from near-dependent
                # accepted units masquerade as zero novelty and permanently
                # floor operators that still carry rank.
                for _pass in range(2):
                    for unit in accepted_units:
                        projected = projected - complex(np.vdot(unit, projected)) * unit
                p_norm_sq = float(np.vdot(projected, projected).real)
                novelty = max(0.0, p_norm_sq / w_norm_sq)
            else:
                projected = w
                p_norm_sq = 0.0
                novelty = 0.0

            residual_capture = 0.0
            if p_norm_sq > 0.0:
                unit_vec = projected / math.sqrt(p_norm_sq)
                for hat in round_residuals:
                    residual_capture = max(
                        residual_capture, float(abs(complex(np.vdot(unit_vec, hat))))
                    )
                h_unit = np.asarray(
                    _apply_polynomial_operator(
                        hamiltonian,
                        unit_vec,
                        nq=int(nq),
                        name="hamiltonian",
                        config=cfg,
                        pauli_action_cache=cache,
                    ),
                    dtype=complex,
                ).reshape(-1)
                # h11 is the candidate's one-direction Rayleigh value; the
                # stopping rule's window-closure test consumes it.
                h11 = float(complex(np.vdot(unit_vec, h_unit)).real)
                round_thetas[index] = float(h11)
                round_units[index] = unit_vec
                round_h_units[index] = h_unit

            cost_norm = _cost_value(candidate) / max_cost
            round_features[index] = {
                "supported_image_norm_sq": float(w_norm_sq),
                "supported_image_admissible": bool(supported_image_admissible),
                "novelty": float(novelty),
                "residual_capture": float(residual_capture),
                "cost_norm": float(cost_norm),
            }
            score_mode = str(config.geometry_score_mode)
            if score_mode == GEOMETRY_SCORE_NORMALIZED_ANGULAR:
                spectral_raw = float(residual_capture)
            elif score_mode in {
                GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN,
                GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN,
                GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST,
            }:
                spectral_raw = 0.0
                if p_norm_sq > 0.0:
                    score_candidate = (
                        w
                        if score_mode == GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN
                        else round_units[index]
                    )
                    h_score_candidate = np.asarray(
                        _apply_polynomial_operator(
                            hamiltonian,
                            score_candidate,
                            nq=int(nq),
                            name="hamiltonian",
                            config=cfg,
                            pauli_action_cache=cache,
                        ),
                        dtype=complex,
                    ).reshape(-1)
                    spectral_raw = sum(
                        _two_dimensional_ritz_gain(
                            reference_state=state,
                            h_reference_state=h_state,
                            reference_energy=energy,
                            candidate_state=score_candidate,
                            h_candidate_state=h_score_candidate,
                            relative_cutoff=float(cfg.overlap_relative_cutoff),
                            absolute_cutoff=float(cfg.overlap_absolute_cutoff),
                        )
                        for state, h_state, energy in zip(
                            spectral_state.target_states,
                            spectral_state.target_h_states,
                            spectral_state.target_energies,
                        )
                    )
            elif score_mode == GEOMETRY_SCORE_FULL_PENCIL_KY_FAN:
                # The full proposed-pencil gain needs the common rank-filling
                # ceiling, which is known only after every admissible
                # candidate's h11 has been evaluated.
                spectral_raw = 0.0
            else:
                raise RuntimeError(
                    f"Geometry score mode {score_mode!r} is not implemented by this selector."
                )
            round_spectral_raw[index] = float(spectral_raw)
            if config.geometry_require_supported_image and not supported_image_admissible:
                supported_image_rejected.add(index)
                del pool[index]
            elif novelty < float(config.geometry_min_metric_novelty):
                floor_rejected.add(index)
                del pool[index]

        score_mode = str(config.geometry_score_mode)

        def _score_from_raw(
            index: int,
            spectral_raw: float,
            spectral_scale: float,
        ) -> tuple[float, float, float]:
            features = round_features[index]
            if score_mode == GEOMETRY_SCORE_NORMALIZED_ANGULAR:
                spectral_normalized = float(spectral_raw)
            else:
                spectral_normalized = (
                    float(spectral_raw) / float(spectral_scale)
                    if spectral_scale > 0.0
                    else 0.0
                )
            novelty_reward = (
                float(config.geometry_metric_novelty_weight)
                * float(features["novelty"])
                if config.geometry_additive_novelty_reward
                else 0.0
            )
            utility = (
                novelty_reward
                + float(config.geometry_residual_weight) * spectral_normalized
            )
            cost_norm = float(features["cost_norm"])
            if config.geometry_cost_discount_alpha is not None:
                alpha = float(config.geometry_cost_discount_alpha)
                clamped = max(float(config.geometry_cost_discount_floor), cost_norm)
                score = utility / (clamped**alpha)
            else:
                score = utility - float(config.geometry_cost_weight) * cost_norm
            return float(spectral_normalized), float(novelty_reward), float(score)

        surrogate_scale = max(
            (float(round_spectral_raw[index]) for index in pool),
            default=0.0,
        )
        shortlist_indices: tuple[int, ...] = ()
        full_pencil_indices: tuple[int, ...] = ()
        if score_mode == GEOMETRY_SCORE_FULL_PENCIL_KY_FAN:
            full_pencil_indices = tuple(pool)
        elif score_mode == GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST:
            surrogate_scores = {
                index: _score_from_raw(
                    index,
                    float(round_spectral_raw[index]),
                    float(surrogate_scale),
                )[2]
                for index in pool
            }
            shortlist_cap = min(
                int(config.geometry_ky_fan_shortlist_size or 0), len(pool)
            )
            shortlist_indices = tuple(
                sorted(
                    pool,
                    key=lambda index: (-float(surrogate_scores[index]), int(index)),
                )[:shortlist_cap]
            )
            full_pencil_indices = shortlist_indices
            shortlist_set = set(shortlist_indices)
            for index in round_features:
                round_score_diagnostics.setdefault(index, {}).update(
                    {
                        "ky_fan_shortlisted": bool(index in shortlist_set),
                        "shortlist_surrogate_score_raw": float(
                            round_spectral_raw[index]
                        ),
                        "shortlist_surrogate_selection_score": (
                            float(surrogate_scores[index])
                            if index in surrogate_scores
                            else None
                        ),
                    }
                )

        if full_pencil_indices:
            target_count = int(config.geometry_target_roots)
            current_count = len(accepted_units)
            current_eigenvalues = (
                np.linalg.eigvalsh(spectral_state.retained_pencil)
                if current_count
                else np.asarray([], dtype=float)
            )
            rank_filling_ceiling = max(float(round_thetas[index]) for index in pool)
            for index in full_pencil_indices:
                proposed = np.empty((current_count + 1, current_count + 1), dtype=complex)
                if current_count:
                    proposed[:current_count, :current_count] = spectral_state.retained_pencil
                for row in range(current_count):
                    proposed[row, current_count] = complex(
                        np.vdot(accepted_units[row], round_h_units[index])
                    )
                    proposed[current_count, row] = complex(
                        np.vdot(round_units[index], accepted_h_units[row])
                    )
                proposed[current_count, current_count] = complex(round_thetas[index])
                proposed = 0.5 * (proposed + proposed.conj().T)
                proposed_eigenvalues = np.linalg.eigvalsh(proposed)
                if current_count < target_count:
                    current_objective = float(np.sum(current_eigenvalues)) + (
                        target_count - current_count
                    ) * rank_filling_ceiling
                    proposed_objective = float(np.sum(proposed_eigenvalues)) + (
                        target_count - current_count - 1
                    ) * rank_filling_ceiling
                    bootstrap = True
                else:
                    current_objective = float(np.sum(current_eigenvalues[:target_count]))
                    proposed_objective = float(np.sum(proposed_eigenvalues[:target_count]))
                    bootstrap = False
                gain = max(0.0, current_objective - proposed_objective)
                round_spectral_raw[index] = float(gain)
                round_score_diagnostics.setdefault(index, {}).update(
                    {
                        "ky_fan_bootstrap": bool(bootstrap),
                        "rank_filling_ceiling": (
                            float(rank_filling_ceiling) if bootstrap else None
                        ),
                        "current_ky_fan_objective": float(current_objective),
                        "proposed_ky_fan_objective": float(proposed_objective),
                        "proposed_retained_rank": int(current_count + 1),
                    }
                )

        selection_indices = (
            set(shortlist_indices)
            if score_mode == GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST
            else set(pool)
        )
        positive_scale = max(
            (float(round_spectral_raw[index]) for index in selection_indices),
            default=0.0,
        )
        for index, features in round_features.items():
            novelty = float(features["novelty"])
            residual_capture = float(features["residual_capture"])
            cost_norm = float(features["cost_norm"])
            spectral_raw = float(round_spectral_raw[index])
            scale = (
                surrogate_scale
                if score_mode
                == GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST
                and index not in selection_indices
                else positive_scale
            )
            spectral_normalized, novelty_reward, score = _score_from_raw(
                index,
                spectral_raw,
                float(scale),
            )
            geometry_rows[index] = {
                "supported_image_norm_sq": float(features["supported_image_norm_sq"]),
                "supported_image_admissible": bool(features["supported_image_admissible"]),
                "metric_novelty_fraction": float(novelty),
                "additive_novelty_reward": float(novelty_reward),
                "residual_capture": float(residual_capture),
                "score_mode": str(config.geometry_score_mode),
                "spectral_score_raw": float(spectral_raw),
                "spectral_score_normalized": float(spectral_normalized),
                "cost_norm": float(cost_norm),
                "cost_source": "compiled" if compiled_costs is not None else "cost_proxy",
                **round_score_diagnostics.get(index, {}),
            }
            scores[index] = float(score)
            if index in selection_indices:
                round_scores[index] = float(score)

        window_margin = (
            float(config.geometry_residual_stop)
            if config.geometry_residual_stop is not None
            else 0.0
        )
        window_pressure = bool(
            frame_size >= int(config.geometry_target_roots)
            and window_top is not None
            and any(
                round_thetas.get(idx) is not None
                and float(round_thetas[idx]) < float(window_top) + window_margin
                for idx in pool
            )
        )
        two_dimensional_modes = {
            GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN,
            GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN,
            GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST,
        }
        round_row: dict[str, Any] = {
            "round": len(round_trace) + 1,
            "selected_count_before": len(selected),
            "retained_frame_size": int(frame_size),
            "max_target_residual_norm": last_max_residual,
            "window_top": window_top,
            "window_margin": float(window_margin),
            "window_pressure": bool(window_pressure),
            "window_closed": bool(
                frame_size >= int(config.geometry_target_roots)
                and not window_pressure
            ),
            "novel_candidate_count": len(pool),
            "selection_candidate_count": len(round_scores),
            "screening": {
                "admissible_candidate_count": len(pool),
                "admissible_candidate_original_indices": [
                    int(index) for index in pool
                ],
                "angular_residual_overlap_evaluations": (
                    len(pool) * len(round_residuals)
                    if score_mode == GEOMETRY_SCORE_NORMALIZED_ANGULAR
                    else 0
                ),
                "two_dimensional_pencil_evaluations": (
                    len(pool) * len(spectral_state.target_states)
                    if score_mode in two_dimensional_modes
                    else 0
                ),
                "two_dimensional_candidate_original_indices": (
                    [int(index) for index in pool]
                    if score_mode in two_dimensional_modes
                    else []
                ),
                "full_proposed_pencil_evaluations": len(full_pencil_indices),
                "full_proposed_pencil_candidate_original_indices": [
                    int(index) for index in full_pencil_indices
                ],
                "full_proposed_hamiltonian_row_elements": (
                    len(full_pencil_indices) * (len(accepted_units) + 1)
                ),
                "ky_fan_shortlist_size_requested": (
                    int(config.geometry_ky_fan_shortlist_size)
                    if config.geometry_ky_fan_shortlist_size is not None
                    else None
                ),
                "ky_fan_shortlist_size_realized": len(shortlist_indices),
            },
            "max_candidate_metric_novelty": (
                max(
                    float(geometry_rows[idx]["metric_novelty_fraction"])
                    for idx in pool
                )
                if pool
                else None
            ),
            "max_candidate_residual_capture": (
                max(
                    float(geometry_rows[idx]["residual_capture"])
                    for idx in pool
                )
                if pool
                else None
            ),
            "selected_original_index": None,
            "selected_score": None,
            "selected_metric_novelty": None,
            "selected_residual_capture": None,
        }
        if not round_scores:
            # An exhausted pool with pending residual convergence satisfies
            # both conditions: no remaining pressure, residuals below eps.
            round_trace.append(round_row)
            stop_reason = "residual_converged" if pending_convergence else "pool_exhausted"
            break
        if pending_convergence:
            # Residual convergence alone is Ritz-blind: a frame's lowest-R
            # Ritz states can all be true eigenstates (vanishing residuals)
            # while lower sector states the frame does not overlap remain
            # invisible. The completeness condition is spectral window
            # pressure: a remaining candidate threatens the window only if
            # its (novel-component) Rayleigh quotient lies at or below the
            # current R-th Ritz value, i.e. it could still introduce a new
            # state inside the window. Pure novelty above the window is not
            # pressure.
            if not window_pressure:
                round_trace.append(round_row)
                stop_reason = "residual_converged"
                break
        best_index = min(round_scores, key=lambda idx: (-round_scores[idx], idx))
        round_row.update(
            {
                "selected_original_index": int(best_index),
                "selected_score": float(round_scores[best_index]),
                "selected_metric_novelty": float(
                    geometry_rows[best_index]["metric_novelty_fraction"]
                ),
                "selected_residual_capture": float(
                    geometry_rows[best_index]["residual_capture"]
                ),
            }
        )
        round_trace.append(round_row)
        selected.append(pool.pop(best_index))
        accepted_images.append(images[best_index])
        _rebuild_retained_frame()

    if pool and len(selected) >= int(config.max_records) and stop_reason == "budget_reached":
        pass  # explicit: cap hit with candidates remaining
    elif not pool and stop_reason == "budget_reached":
        stop_reason = "pool_exhausted"
    stop_info = {
        "stop_reason": stop_reason,
        "final_max_target_residual_norm": last_max_residual,
        "residual_converged_pending": bool(pending_convergence),
        "residual_stop_threshold": (
            float(config.geometry_residual_stop)
            if config.geometry_residual_stop is not None
            else None
        ),
        "round_trace": tuple(round_trace),
    }
    return (
        selected,
        floor_rejected,
        supported_image_rejected,
        geometry_rows,
        scores,
        stop_info,
    )


def select_static_qse_records(
    basis_elements: Sequence[QSEBasisElement],
    *,
    config: StaticRecordSelectionConfig,
    hamiltonian: Any = None,
    prepared_state: Any = None,
    qse_config: Any = None,
    basis_vector_policy: Any = None,
    transition_observables: Sequence[Any] = (),
    compiled_costs: Sequence[float] | None = None,
) -> StaticRecordSelectionResult:
    """Select a deterministic static QSE basis subset from pre-QSE candidate records.

    The geometry_selected mode consumes the keyword context (hamiltonian,
    prepared_state, qse_config, basis_vector_policy, transition_observables);
    the input_order and cost_proxy modes accept and ignore it so all callers
    can pass a uniform signature. ``compiled_costs`` supplies per-candidate
    Paper I compiled hardware costs (scalarized, input order): the
    compiled_cost mode requires it, and geometry_selected consumes it as its
    cost coordinate in place of the structural proxy when present.
    """

    basis_tuple = tuple(basis_elements)
    resolved_compiled_costs: tuple[float, ...] | None = None
    if compiled_costs is not None:
        resolved_compiled_costs = tuple(float(value) for value in compiled_costs)
        if len(resolved_compiled_costs) != len(basis_tuple):
            raise ValueError(
                f"compiled_costs length {len(resolved_compiled_costs)} does not match "
                f"candidate basis size {len(basis_tuple)}."
            )
        for value in resolved_compiled_costs:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("compiled_costs entries must be finite and >= 0.")
    if str(config.mode) == "compiled_cost" and resolved_compiled_costs is None:
        raise ValueError(
            "compiled_cost static record selection requires compiled_costs for every candidate."
        )
    candidates = tuple(
        extract_static_record_candidate(element, original_basis_index=idx)
        for idx, element in enumerate(basis_tuple)
    )
    if not candidates:
        raise ValueError("Static record selection requires at least one candidate basis element.")

    hard_reasons_by_index = {
        candidate.original_basis_index: _hard_rejection_reasons(candidate, config)
        for candidate in candidates
    }
    eligible_candidates = tuple(
        candidate
        for candidate in candidates
        if not hard_reasons_by_index[int(candidate.original_basis_index)]
    )
    geometry_floor_rejected: set[int] = set()
    geometry_supported_image_rejected: set[int] = set()
    geometry_rows: dict[int, dict[str, Any]] = {}
    geometry_scores: dict[int, float] = {}
    if str(config.mode) == "geometry_selected":
        (
            selected_list,
            geometry_floor_rejected,
            geometry_supported_image_rejected,
            geometry_rows,
            geometry_scores,
            geometry_stop_info,
        ) = _geometry_select(
            basis_tuple=basis_tuple,
            eligible_candidates=eligible_candidates,
            config=config,
            hamiltonian=hamiltonian,
            prepared_state=prepared_state,
            qse_config=qse_config,
            basis_vector_policy=basis_vector_policy,
            transition_observables=transition_observables,
            compiled_costs=resolved_compiled_costs,
        )
        selected_candidates = tuple(selected_list)
    elif str(config.mode) == "compiled_cost":
        ranked = tuple(
            sorted(
                eligible_candidates,
                key=lambda candidate: (
                    float(resolved_compiled_costs[int(candidate.original_basis_index)]),
                    int(candidate.original_basis_index),
                ),
            )
        )
        selected_candidates = ranked[: int(config.max_records)]
    else:
        ranked = tuple(sorted(eligible_candidates, key=lambda candidate: _rank_key(candidate, config)))
        selected_candidates = ranked[: int(config.max_records)]
    if not selected_candidates:
        raise ValueError("Static record selection retained zero candidates.")

    selected_position_by_original = {
        int(candidate.original_basis_index): int(selected_basis_index)
        for selected_basis_index, candidate in enumerate(selected_candidates)
    }
    selected_indices = tuple(int(candidate.original_basis_index) for candidate in selected_candidates)
    selected_basis = tuple(basis_tuple[index] for index in selected_indices)

    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        original_index = int(candidate.original_basis_index)
        hard_reasons = list(hard_reasons_by_index[original_index])
        selected_basis_index = selected_position_by_original.get(original_index)
        eligible = not hard_reasons
        selected = selected_basis_index is not None
        reasons = list(hard_reasons)
        if eligible and not selected:
            if original_index in geometry_supported_image_rejected:
                reasons.append("supported_image_floor")
            elif original_index in geometry_floor_rejected:
                reasons.append("metric_novelty_floor")
            else:
                reasons.append("rank_limit")
        decision: dict[str, Any] = {
            "original_basis_index": original_index,
            "eligible": bool(eligible),
            "selected": bool(selected),
            "selected_basis_index": selected_basis_index,
            "rejection_reasons": reasons,
        }
        if original_index in geometry_rows:
            decision["geometry"] = dict(geometry_rows[original_index])
        if original_index in geometry_scores:
            decision["selection_score"] = float(geometry_scores[original_index])
        if resolved_compiled_costs is not None:
            decision["compiled_cost"] = float(resolved_compiled_costs[original_index])
        decisions.append(decision)

    geometry_cost_source: str | None = None
    if str(config.mode) == "geometry_selected":
        geometry_cost_source = "compiled" if resolved_compiled_costs is not None else "cost_proxy"
    else:
        geometry_stop_info = None

    return StaticRecordSelectionResult(
        config=config,
        candidates=candidates,
        selected_basis_elements=selected_basis,
        selected_original_indices=selected_indices,
        candidate_decisions=tuple(decisions),
        compiled_costs=resolved_compiled_costs,
        geometry_cost_source=geometry_cost_source,
        geometry_stop=geometry_stop_info,
    )


def _config_to_payload(config: StaticRecordSelectionConfig) -> dict[str, Any]:
    return _json_safe(asdict(config))


def _candidate_to_payload(
    candidate: StaticRecordCandidate,
    *,
    decision: Mapping[str, Any],
    config: StaticRecordSelectionConfig,
) -> dict[str, Any]:
    payload = {
        "original_basis_index": int(candidate.original_basis_index),
        "name": str(candidate.name),
        "kind": str(candidate.kind),
        "metadata": _json_safe(candidate.metadata),
        "pauli_label_exyz": candidate.pauli_label_exyz,
        "features": {
            "nq": int(candidate.nq),
            "term_count": int(candidate.term_count),
            "max_pauli_weight": int(candidate.max_pauli_weight),
            "mean_pauli_weight": float(candidate.mean_pauli_weight),
            "support_qubit_count": int(candidate.support_qubit_count),
            "coefficient_l1": float(candidate.coefficient_l1),
            "cost_proxy": float(candidate.cost_proxy),
        },
        "rank_key": {
            "mode": str(config.mode),
            "score_direction": "lower_is_better",
            "cost_proxy": float(candidate.cost_proxy),
            "input_index": int(candidate.original_basis_index),
        },
        "eligible": bool(decision["eligible"]),
        "selected": bool(decision["selected"]),
        "selected_basis_index": decision["selected_basis_index"],
        "rejection_reasons": list(decision["rejection_reasons"]),
    }
    if "geometry" in decision:
        payload["geometry"] = _json_safe(decision["geometry"])
    if "selection_score" in decision:
        payload["selection_score"] = float(decision["selection_score"])
    if "compiled_cost" in decision:
        payload["compiled_cost"] = float(decision["compiled_cost"])
    return payload


def static_record_selection_payload(result: StaticRecordSelectionResult) -> dict[str, Any]:
    """Return a manifest-ready pre-QSE static-selection audit payload."""

    decisions_by_index = {
        int(decision["original_basis_index"]): decision for decision in result.candidate_decisions
    }
    candidates_payload = [
        _candidate_to_payload(
            candidate,
            decision=decisions_by_index[int(candidate.original_basis_index)],
            config=result.config,
        )
        for candidate in result.candidates
    ]
    selected_records = [
        {
            "original_basis_index": int(candidate.original_basis_index),
            "selected_basis_index": int(decisions_by_index[int(candidate.original_basis_index)]["selected_basis_index"]),
            "name": str(candidate.name),
            "kind": str(candidate.kind),
            "cost_proxy": float(candidate.cost_proxy),
            **(
                {"selection_score": float(decisions_by_index[int(candidate.original_basis_index)]["selection_score"])}
                if "selection_score" in decisions_by_index[int(candidate.original_basis_index)]
                else {}
            ),
        }
        for candidate in result.candidates
        if bool(decisions_by_index[int(candidate.original_basis_index)]["selected"])
    ]
    selected_records.sort(key=lambda row: int(row["selected_basis_index"]))
    rejected_records = [
        {
            "original_basis_index": int(decision["original_basis_index"]),
            "rejection_reasons": list(decision["rejection_reasons"]),
            "eligible": bool(decision["eligible"]),
        }
        for decision in result.candidate_decisions
        if not bool(decision["selected"])
    ]
    hard_rejected_count = sum(1 for decision in result.candidate_decisions if not bool(decision["eligible"]))
    rank_limited_rejected_count = sum(
        1 for decision in result.candidate_decisions if "rank_limit" in decision["rejection_reasons"]
    )
    return {
        "schema_version": STATIC_RECORD_SELECTION_SCHEMA_VERSION,
        "policy": STATIC_RECORD_SELECTION_POLICY,
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "post_run_diagnostic_only": True,
        },
        "selection_config": _config_to_payload(result.config),
        "summary": {
            "input_basis_size": int(len(result.candidates)),
            "eligible_candidate_count": int(sum(1 for decision in result.candidate_decisions if decision["eligible"])),
            "selected_basis_size": int(len(result.selected_basis_elements)),
            "hard_rejected_count": int(hard_rejected_count),
            "rank_limited_rejected_count": int(rank_limited_rejected_count),
        },
        "compiled_costs_present": result.compiled_costs is not None,
        "geometry_cost_source": result.geometry_cost_source,
        "geometry_stop": _json_safe(dict(result.geometry_stop)) if result.geometry_stop else None,
        "candidates": candidates_payload,
        "selected_records": selected_records,
        "selected_mapping": [
            {
                "original_basis_index": int(row["original_basis_index"]),
                "selected_basis_index": int(row["selected_basis_index"]),
            }
            for row in selected_records
        ],
        "selected_original_basis_indices": [int(index) for index in result.selected_original_indices],
        "rejected_records": rejected_records,
        "post_qse_diagnostics": None,
    }


def _guard_payload(*, enabled: bool, configured: Any, actual: Any, passed: bool | None) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "configured": configured,
        "actual": actual,
        "passed": passed,
    }


def finalize_static_record_selection_payload(
    selection_result: StaticRecordSelectionResult,
    qse_result: QSEResult,
) -> dict[str, Any]:
    """Append post-QSE guard diagnostics to a static-selection payload.

    The returned payload is a new mapping; selection indices and the selected
    basis are not modified by guard outcomes.
    """

    payload = static_record_selection_payload(selection_result)
    diagnostics = tuple(qse_result.matrices.basis_vector_diagnostics)
    diagnostics_available = bool(diagnostics)
    zero_count = None
    projected_out_count = None
    if diagnostics_available:
        zero_count = int(sum(1 for row in diagnostics if bool(row.zero_vector)))
        projected_out_count = int(sum(1 for row in diagnostics if bool(row.projected_out_by_q0)))

    min_rank_enabled = selection_result.config.min_retained_rank is not None
    min_rank_configured = selection_result.config.min_retained_rank
    min_rank_passed = None
    if min_rank_enabled:
        min_rank_passed = bool(int(qse_result.retained_rank) >= int(min_rank_configured))

    condition_enabled = selection_result.config.max_overlap_condition is not None
    condition_configured = selection_result.config.max_overlap_condition
    condition_actual = qse_result.overlap_condition_estimate
    condition_passed = None
    if condition_enabled:
        condition_passed = bool(
            condition_actual is not None
            and math.isfinite(float(condition_actual))
            and float(condition_actual) <= float(condition_configured)
        )

    guard_values = []
    if min_rank_enabled:
        guard_values.append(bool(min_rank_passed))
    if condition_enabled:
        guard_values.append(bool(condition_passed))
    all_passed = None if not guard_values else bool(all(guard_values))

    payload["post_qse_diagnostics"] = {
        "retained_rank": int(qse_result.retained_rank),
        "discarded_rank": int(qse_result.discarded_rank),
        "overlap_condition_estimate": condition_actual,
        "overlap_min_eigenvalue_raw": float(qse_result.overlap_min_eigenvalue_raw),
        "overlap_max_eigenvalue_raw": float(qse_result.overlap_max_eigenvalue_raw),
        "overlap_pruning_threshold": float(qse_result.overlap_pruning_threshold),
        "basis_vector_zero_count": zero_count,
        "basis_vector_projected_out_by_q0_count": projected_out_count,
        "basis_vector_diagnostics_available": bool(diagnostics_available),
        "guards": {
            "min_retained_rank": _guard_payload(
                enabled=min_rank_enabled,
                configured=min_rank_configured,
                actual=int(qse_result.retained_rank),
                passed=min_rank_passed,
            ),
            "max_overlap_condition": _guard_payload(
                enabled=condition_enabled,
                configured=condition_configured,
                actual=condition_actual,
                passed=condition_passed,
            ),
            "all_configured_guards_passed": all_passed,
        },
    }
    return payload


__all__ = [
    "STATIC_RECORD_SELECTION_SCHEMA_VERSION",
    "STATIC_RECORD_SELECTION_POLICY",
    "GEOMETRY_SCORE_NORMALIZED_ANGULAR",
    "GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN",
    "GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN",
    "GEOMETRY_SCORE_FULL_PENCIL_KY_FAN",
    "GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST",
    "GEOMETRY_SCORE_MODES",
    "StaticRecordSelectionConfig",
    "StaticRecordCandidate",
    "StaticRecordSelectionResult",
    "extract_static_record_candidate",
    "select_static_qse_records",
    "static_record_selection_payload",
    "finalize_static_record_selection_payload",
]
