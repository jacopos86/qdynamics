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
    "geometry_ritz_weight",
    "geometry_transition_weight",
    "geometry_cost_weight",
    "geometry_condition_penalty_weight",
)
_VALID_INTERNAL_PAULIS = set("exyz")


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
    geometry_metric_novelty_weight: float = 0.25
    geometry_residual_weight: float = 1.0
    # Ablation (2026-08-19): the Ritz window gain and the explicit conditioning
    # penalty do not improve selection at the production stop -- the penalty is
    # bit-identical to the anchor and the Ritz term costs 38% more compiled 2Q
    # for the same accuracy. Both default off; metric novelty already supplies
    # the conditioning control.
    geometry_ritz_weight: float = 0.0
    # Ablation (2026-08-19): probe-transition visibility is inert at the
    # production stop -- disabling it reproduces the anchor bit-for-bit in
    # every regime tested. The minimal score is metric novelty + residual
    # capture, cost-discounted.
    geometry_transition_weight: float = 0.0
    geometry_cost_weight: float = 1.0
    geometry_condition_penalty_weight: float = 0.0
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
) -> tuple[list["StaticRecordCandidate"], set[int], dict[int, dict[str, Any]], dict[int, float]]:
    """Greedy geometry-scored selection over pre-QSE candidate images.

    RECONSTRUCTION NOTE (2026-08-17): the original geometry_selected
    implementation was lost in snapshot commit 6442fbb5 (module never
    committed).  This reconstruction is specified by
    test/test_qse_record_selection.py and the CLI surface in __main__.py.
    Score for candidate i against the accepted span S (all vectors are the
    policy-projected images O_i|psi>):

        novelty_i   = ||P_S w_i||^2 / ||w_i||^2          (0 for zero images)
        residual_i  = max_j |<u_i, r_j>| / ||r_j||       (multi-root residuals)
        ritz_i      = g_i / (1 + g_i), g_i = max(0, E0 - lambda_min) of the
                      2x2 Rayleigh pencil in span{psi, u_i}
        trans_i     = max_k |<u_i, t_k>| over projected transition images
        cost_i      = cost_proxy_i / max cost_proxy over eligible candidates
        score_i     = w_nov*novelty_i + w_res*residual_i + w_ritz*ritz_i
                      + w_trans*trans_i - w_cost*cost_i
                      - w_cond*(1 - novelty_i)

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
    geometry_rows: dict[int, dict[str, Any]] = {}
    scores: dict[int, float] = {}
    pool: dict[int, StaticRecordCandidate] = {
        int(c.original_basis_index): c for c in eligible_candidates
    }

    def _round_residual_hats() -> tuple[list[np.ndarray], float | None, int]:
        """Return (residual hats, max target-root residual norm, frame size).

        The max residual norm over the lowest min(R, frame) Ritz roots is the
        natural convergence measure: for Hermitian pencils each Ritz value's
        error is bounded by its residual norm, so it plays the role ADAPT's
        gradient norm plays for ground states.
        """

        if not accepted_units:
            return ([residual_hat] if residual_hat is not None else [], None, 0, None)
        count = len(accepted_units)
        pencil = np.empty((count, count), dtype=complex)
        for row in range(count):
            for col in range(count):
                pencil[row, col] = complex(np.vdot(accepted_units[row], accepted_h_units[col]))
        pencil = 0.5 * (pencil + pencil.conj().T)
        thetas, ritz_vectors = np.linalg.eigh(pencil)
        hats: list[np.ndarray] = []
        max_norm = 0.0
        for root in range(min(int(config.geometry_target_roots), count)):
            coefficients = ritz_vectors[:, root]
            state = sum(complex(coefficients[k]) * accepted_units[k] for k in range(count))
            h_state = sum(complex(coefficients[k]) * accepted_h_units[k] for k in range(count))
            ritz_residual = h_state - float(thetas[root]) * state
            ritz_residual = ritz_residual - complex(np.vdot(psi, ritz_residual)) * psi
            norm = float(np.linalg.norm(ritz_residual))
            max_norm = max(max_norm, norm)
            if norm > 1.0e-12:
                hats.append(ritz_residual / norm)
        if not hats and residual_hat is not None:
            hats = [residual_hat]
        window_top = float(thetas[min(int(config.geometry_target_roots), count) - 1])
        return hats, max_norm, count, window_top

    stop_reason = "budget_reached"
    last_max_residual: float | None = None
    pending_convergence = False
    while pool and len(selected) < int(config.max_records):
        round_residuals, last_max_residual, frame_size, window_top = _round_residual_hats()
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
        for index, candidate in list(pool.items()):
            w = images[index]
            w_norm_sq = float(np.vdot(w, w).real)
            if w_norm_sq > 0.0:
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
            ritz_term = 0.0
            transition_capture = 0.0
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
                h01 = complex(np.vdot(psi, h_unit))
                h11 = float(complex(np.vdot(unit_vec, h_unit)).real)
                pencil = np.array([[e0, h01], [np.conj(h01), h11]], dtype=complex)
                lam_min = float(np.min(np.linalg.eigvalsh(pencil)))
                gain = max(0.0, e0 - lam_min)
                ritz_term = gain / (1.0 + gain)
                round_thetas[index] = float(h11)
                for t_hat in transition_hats:
                    transition_capture = max(
                        transition_capture, float(abs(complex(np.vdot(unit_vec, t_hat))))
                    )
                round_units[index] = unit_vec
                round_h_units[index] = h_unit

            cost_norm = _cost_value(candidate) / max_cost
            utility = (
                float(config.geometry_metric_novelty_weight) * novelty
                + float(config.geometry_residual_weight) * residual_capture
                + float(config.geometry_ritz_weight) * ritz_term
                + float(config.geometry_transition_weight) * transition_capture
            )
            condition_penalty = float(config.geometry_condition_penalty_weight) * (1.0 - novelty)
            if config.geometry_cost_discount_alpha is not None:
                alpha = float(config.geometry_cost_discount_alpha)
                # Clamp the discount denominator: an unbounded 1/cost blows up
                # for zero-compiled-cost candidates and lets cheap junk crowd
                # out the excitation window at larger pools.
                clamped = max(float(config.geometry_cost_discount_floor), cost_norm)
                score = utility / (clamped ** alpha) - condition_penalty
            else:
                score = utility - float(config.geometry_cost_weight) * cost_norm - condition_penalty
            geometry_rows[index] = {
                "metric_novelty_fraction": float(novelty),
                "residual_capture": float(residual_capture),
                "ritz_gain_term": float(ritz_term),
                "transition_capture": float(transition_capture),
                "cost_norm": float(cost_norm),
                "cost_source": "compiled" if compiled_costs is not None else "cost_proxy",
            }
            scores[index] = float(score)
            if novelty < float(config.geometry_min_metric_novelty):
                floor_rejected.add(index)
                del pool[index]
                continue
            round_scores[index] = float(score)

        if not round_scores:
            # An exhausted pool with pending residual convergence satisfies
            # both conditions: no remaining pressure, residuals below eps.
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
            margin = float(config.geometry_residual_stop)
            window_pressure = window_top is not None and any(
                round_thetas.get(idx) is not None
                and float(round_thetas[idx]) < float(window_top) + margin
                for idx in round_scores
            )
            if not window_pressure:
                stop_reason = "residual_converged"
                break
        best_index = min(round_scores, key=lambda idx: (-round_scores[idx], idx))
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
    }
    return selected, floor_rejected, geometry_rows, scores, stop_info


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
    geometry_rows: dict[int, dict[str, Any]] = {}
    geometry_scores: dict[int, float] = {}
    if str(config.mode) == "geometry_selected":
        (
            selected_list,
            geometry_floor_rejected,
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
            if original_index in geometry_floor_rejected:
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
    "StaticRecordSelectionConfig",
    "StaticRecordCandidate",
    "StaticRecordSelectionResult",
    "extract_static_record_candidate",
    "select_static_qse_records",
    "static_record_selection_payload",
    "finalize_static_record_selection_payload",
]
