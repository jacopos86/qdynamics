"""Pure Phase-III material-coupling window selection.

The geometry window selected here is independent of any optimizer/refit window.
It retains active coordinates whose candidate Gram overlap or Hessian coupling
is material, then expands that union until the omitted normalized coupling tails
satisfy explicit closure tolerances.  No estimator work is performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Sequence


PHASE3_MATERIAL_WINDOW_POLICY_VERSION = "phase3_material_window_policy_v1"
PHASE3_MATERIAL_WINDOW_RECEIPT_VERSION = "phase3_material_window_receipt_v1"


class Phase3MaterialWindowError(ValueError):
    """Raised when material-window inputs violate structural invariants."""


@dataclass(frozen=True)
class Phase3MaterialWindowPolicy:
    """Numerical policy for candidate-coupling-driven geometry selection."""

    policy_version: str = PHASE3_MATERIAL_WINDOW_POLICY_VERSION
    # Frozen from the six-regime no-overlap source replay.  This is the least
    # aggressive tier that preserved all 506 feasibility labels and all 300
    # within-round orders before the independent W-by-O closure gate.
    gram_entry_threshold: float = 4.0e-3
    hessian_entry_threshold: float = 2.0e-22
    gram_omitted_l2_tolerance: float = 1.0
    hessian_omitted_l2_tolerance: float = 1.0
    gram_cross_block_tolerance: float = 1.0e-1
    hessian_cross_block_tolerance: float = 1.0e-1
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if not str(self.policy_version):
            raise Phase3MaterialWindowError("policy_version must be nonempty.")
        for label, value in (
            ("gram_entry_threshold", self.gram_entry_threshold),
            ("hessian_entry_threshold", self.hessian_entry_threshold),
            ("gram_omitted_l2_tolerance", self.gram_omitted_l2_tolerance),
            ("hessian_omitted_l2_tolerance", self.hessian_omitted_l2_tolerance),
            ("gram_cross_block_tolerance", self.gram_cross_block_tolerance),
            (
                "hessian_cross_block_tolerance",
                self.hessian_cross_block_tolerance,
            ),
            ("epsilon", self.epsilon),
        ):
            if not math.isfinite(float(value)):
                raise Phase3MaterialWindowError(f"{label} must be finite.")
        if float(self.gram_entry_threshold) < 0.0:
            raise Phase3MaterialWindowError("gram_entry_threshold must be nonnegative.")
        if float(self.hessian_entry_threshold) < 0.0:
            raise Phase3MaterialWindowError("hessian_entry_threshold must be nonnegative.")
        if not 0.0 <= float(self.gram_omitted_l2_tolerance) <= 1.0:
            raise Phase3MaterialWindowError(
                "gram_omitted_l2_tolerance must lie in [0, 1]."
            )
        if not 0.0 <= float(self.hessian_omitted_l2_tolerance) <= 1.0:
            raise Phase3MaterialWindowError(
                "hessian_omitted_l2_tolerance must lie in [0, 1]."
            )
        if float(self.gram_cross_block_tolerance) < 0.0:
            raise Phase3MaterialWindowError(
                "gram_cross_block_tolerance must be nonnegative."
            )
        if float(self.hessian_cross_block_tolerance) < 0.0:
            raise Phase3MaterialWindowError(
                "hessian_cross_block_tolerance must be nonnegative."
            )
        if float(self.epsilon) <= 0.0:
            raise Phase3MaterialWindowError("epsilon must be strictly positive.")


DEFAULT_PHASE3_MATERIAL_WINDOW_POLICY = Phase3MaterialWindowPolicy()


@dataclass(frozen=True)
class Phase3MaterialWindowReceipt:
    """Immutable selection, closure, and support-rank receipt.

    Masks are aligned with ``active_indices``.  The two initial masks expose the
    independent Gram and Hessian decisions.  ``final_retained_mask`` is their
    deterministic union after any closure-driven expansion.
    """

    receipt_version: str
    policy: Phase3MaterialWindowPolicy
    active_indices: tuple[int, ...]
    prior_active_nullity: int | None
    prior_joint_nullity: int | None
    gram_normalized_scores: tuple[float | None, ...]
    hessian_normalized_scores: tuple[float | None, ...]
    initial_gram_mask: tuple[bool, ...]
    initial_hessian_mask: tuple[bool, ...]
    initial_union_mask: tuple[bool, ...]
    final_retained_mask: tuple[bool, ...]
    closure_added_indices: tuple[int, ...]
    retained_indices: tuple[int, ...]
    omitted_indices: tuple[int, ...]
    initial_gram_omitted_l2_ratio: float | None
    initial_hessian_omitted_l2_ratio: float | None
    final_gram_omitted_l2_ratio: float | None
    final_hessian_omitted_l2_ratio: float | None
    gram_entry_threshold: float
    hessian_entry_threshold: float
    gram_omitted_l2_tolerance: float
    hessian_omitted_l2_tolerance: float
    inputs_finite: bool
    closure_satisfied: bool
    closure_reason: str
    measured_active_supported_rank: int | None = None
    measured_joint_supported_rank: int | None = None
    measured_active_nullity: int | None = None
    measured_joint_nullity: int | None = None
    measured_rank_gain: int | None = None
    support_nullity_drift: bool | None = None
    requires_full_geometry_refresh: bool = False
    refresh_reasons: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe deterministic representation."""

        return _json_safe(asdict(self))

    def finalize_with_support_ranks(
        self,
        *,
        active_supported_rank: int,
        joint_supported_rank: int,
        additional_refresh_reasons: Sequence[str] = (),
    ) -> "Phase3MaterialWindowReceipt":
        """Attach measured ranks and decide whether full geometry is required."""

        return finalize_phase3_material_window_receipt(
            self,
            active_supported_rank=active_supported_rank,
            joint_supported_rank=joint_supported_rank,
            additional_refresh_reasons=additional_refresh_reasons,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _with_digest(receipt: Phase3MaterialWindowReceipt) -> Phase3MaterialWindowReceipt:
    unsigned = replace(receipt, receipt_sha256="")
    payload = unsigned.to_dict()
    payload.pop("receipt_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return replace(receipt, receipt_sha256=hashlib.sha256(encoded).hexdigest())


def _strict_indices(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise Phase3MaterialWindowError("active_indices must be a sequence of integers.")
    indices = tuple(int(value) for value in values)
    if len(set(indices)) != len(indices):
        raise Phase3MaterialWindowError("active_indices must not contain duplicates.")
    if any(index < 0 for index in indices):
        raise Phase3MaterialWindowError("active_indices must be nonnegative.")
    return indices


def _float_vector(
    values: Sequence[float],
    *,
    expected_length: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise Phase3MaterialWindowError(f"{label} must be a numeric sequence.")
    result = tuple(float(value) for value in values)
    if len(result) != int(expected_length):
        raise Phase3MaterialWindowError(
            f"{label} length {len(result)} does not match active coordinate count "
            f"{expected_length}."
        )
    return result


def _optional_nullity(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise Phase3MaterialWindowError(f"{label} must be nonnegative when supplied.")
    return result


def _tail_ratio(
    scores: Sequence[float | None],
    retained_mask: Sequence[bool],
    *,
    epsilon: float,
) -> float | None:
    if any(score is None for score in scores):
        return None
    values = tuple(float(score) for score in scores if score is not None)
    full_norm = math.sqrt(sum(value * value for value in values))
    omitted_norm = math.sqrt(
        sum(
            float(score) * float(score)
            for score, retained in zip(scores, retained_mask, strict=True)
            if score is not None and not bool(retained)
        )
    )
    if full_norm <= float(epsilon):
        return 0.0
    return float(omitted_norm / full_norm)


def _at_or_below(value: float | None, tolerance: float) -> bool:
    return value is not None and float(value) <= float(tolerance)


def build_phase3_material_window(
    *,
    active_indices: Sequence[int],
    gram_diagonal: Sequence[float],
    candidate_gram_cross: Sequence[float],
    candidate_gram_self: float,
    candidate_hessian_cross: Sequence[float],
    candidate_hessian_self: float,
    policy: Phase3MaterialWindowPolicy = DEFAULT_PHASE3_MATERIAL_WINDOW_POLICY,
    prior_active_nullity: int | None = None,
    prior_joint_nullity: int | None = None,
) -> Phase3MaterialWindowReceipt:
    """Select a deterministic candidate-coupled Phase-III geometry window."""

    if not isinstance(policy, Phase3MaterialWindowPolicy):
        raise Phase3MaterialWindowError("policy must be a Phase3MaterialWindowPolicy.")
    indices = _strict_indices(active_indices)
    count = len(indices)
    gram_diag = _float_vector(
        gram_diagonal,
        expected_length=count,
        label="gram_diagonal",
    )
    gram_cross = _float_vector(
        candidate_gram_cross,
        expected_length=count,
        label="candidate_gram_cross",
    )
    hessian_cross = _float_vector(
        candidate_hessian_cross,
        expected_length=count,
        label="candidate_hessian_cross",
    )
    gram_self = float(candidate_gram_self)
    hessian_self = float(candidate_hessian_self)
    prior_active = _optional_nullity(
        prior_active_nullity,
        label="prior_active_nullity",
    )
    prior_joint = _optional_nullity(
        prior_joint_nullity,
        label="prior_joint_nullity",
    )

    finite_flags = tuple(
        math.isfinite(value)
        for value in (*gram_diag, *gram_cross, gram_self, *hessian_cross, hessian_self)
    )
    inputs_finite = all(finite_flags)
    epsilon = float(policy.epsilon)

    gram_scores: list[float | None] = []
    for diagonal, cross in zip(gram_diag, gram_cross, strict=True):
        if not all(math.isfinite(value) for value in (diagonal, cross, gram_self)):
            gram_scores.append(None)
            continue
        denominator = math.sqrt(max(float(diagonal), epsilon) * max(gram_self, epsilon))
        score = abs(float(cross)) / max(denominator, epsilon)
        gram_scores.append(float(score) if math.isfinite(score) else None)

    if all(math.isfinite(value) for value in (*hessian_cross, hessian_self)):
        hessian_scale = max(
            math.sqrt(sum(float(value) * float(value) for value in hessian_cross)),
            abs(hessian_self),
            epsilon,
        )
        hessian_scores: list[float | None] = [
            float(abs(float(value)) / hessian_scale) for value in hessian_cross
        ]
    else:
        hessian_scores = [
            None if not math.isfinite(value) or not math.isfinite(hessian_self) else 0.0
            for value in hessian_cross
        ]

    initial_gram_mask = tuple(
        score is not None and float(score) >= float(policy.gram_entry_threshold)
        for score in gram_scores
    )
    initial_hessian_mask = tuple(
        score is not None and float(score) >= float(policy.hessian_entry_threshold)
        for score in hessian_scores
    )
    initial_union_mask = tuple(
        bool(gram_keep or hessian_keep)
        for gram_keep, hessian_keep in zip(
            initial_gram_mask,
            initial_hessian_mask,
            strict=True,
        )
    )
    final_mask = list(initial_union_mask)
    initial_gram_tail = _tail_ratio(gram_scores, final_mask, epsilon=epsilon)
    initial_hessian_tail = _tail_ratio(hessian_scores, final_mask, epsilon=epsilon)
    closure_added: list[int] = []

    if inputs_finite:
        omitted_positions = [position for position, keep in enumerate(final_mask) if not keep]
        omitted_positions.sort(
            key=lambda position: (
                -max(
                    float(gram_scores[position] or 0.0),
                    float(hessian_scores[position] or 0.0),
                ),
                int(indices[position]),
                int(position),
            )
        )
        for position in omitted_positions:
            gram_tail = _tail_ratio(gram_scores, final_mask, epsilon=epsilon)
            hessian_tail = _tail_ratio(hessian_scores, final_mask, epsilon=epsilon)
            if _at_or_below(gram_tail, policy.gram_omitted_l2_tolerance) and _at_or_below(
                hessian_tail,
                policy.hessian_omitted_l2_tolerance,
            ):
                break
            final_mask[position] = True
            closure_added.append(int(indices[position]))

    final_gram_tail = _tail_ratio(gram_scores, final_mask, epsilon=epsilon)
    final_hessian_tail = _tail_ratio(hessian_scores, final_mask, epsilon=epsilon)
    closure_satisfied = bool(
        inputs_finite
        and _at_or_below(final_gram_tail, policy.gram_omitted_l2_tolerance)
        and _at_or_below(final_hessian_tail, policy.hessian_omitted_l2_tolerance)
    )
    if count == 0 and inputs_finite:
        closure_reason = "candidate_only"
    elif not inputs_finite:
        closure_reason = "nonfinite_input"
    elif closure_satisfied and closure_added:
        closure_reason = "satisfied_after_greedy_expansion"
    elif closure_satisfied:
        closure_reason = "satisfied_by_threshold_union"
    else:
        closure_reason = "omitted_tail_closure_failed"

    final_mask_tuple = tuple(bool(value) for value in final_mask)
    retained_indices = tuple(
        int(index)
        for index, retained in zip(indices, final_mask_tuple, strict=True)
        if retained
    )
    omitted_indices = tuple(
        int(index)
        for index, retained in zip(indices, final_mask_tuple, strict=True)
        if not retained
    )
    receipt = Phase3MaterialWindowReceipt(
        receipt_version=PHASE3_MATERIAL_WINDOW_RECEIPT_VERSION,
        policy=policy,
        active_indices=indices,
        prior_active_nullity=prior_active,
        prior_joint_nullity=prior_joint,
        gram_normalized_scores=tuple(gram_scores),
        hessian_normalized_scores=tuple(hessian_scores),
        initial_gram_mask=initial_gram_mask,
        initial_hessian_mask=initial_hessian_mask,
        initial_union_mask=initial_union_mask,
        final_retained_mask=final_mask_tuple,
        closure_added_indices=tuple(closure_added),
        retained_indices=retained_indices,
        omitted_indices=omitted_indices,
        initial_gram_omitted_l2_ratio=initial_gram_tail,
        initial_hessian_omitted_l2_ratio=initial_hessian_tail,
        final_gram_omitted_l2_ratio=final_gram_tail,
        final_hessian_omitted_l2_ratio=final_hessian_tail,
        gram_entry_threshold=float(policy.gram_entry_threshold),
        hessian_entry_threshold=float(policy.hessian_entry_threshold),
        gram_omitted_l2_tolerance=float(policy.gram_omitted_l2_tolerance),
        hessian_omitted_l2_tolerance=float(policy.hessian_omitted_l2_tolerance),
        inputs_finite=bool(inputs_finite),
        closure_satisfied=closure_satisfied,
        closure_reason=closure_reason,
    )
    return _with_digest(receipt)


def finalize_phase3_material_window_receipt(
    receipt: Phase3MaterialWindowReceipt,
    *,
    active_supported_rank: int,
    joint_supported_rank: int,
    additional_refresh_reasons: Sequence[str] = (),
) -> Phase3MaterialWindowReceipt:
    """Finalize a window receipt using measured active and joint support ranks.

    Ordinary candidate admission grows the joint dimension by one, so a rank
    gain of zero or one is valid.  Drift is detected in support *nullity*, not
    raw rank, which prevents dimensional growth alone from triggering refresh.
    """

    if not isinstance(receipt, Phase3MaterialWindowReceipt):
        raise Phase3MaterialWindowError("receipt must be a Phase3MaterialWindowReceipt.")
    external_reasons = tuple(str(value).strip() for value in additional_refresh_reasons)
    if any(not value for value in external_reasons):
        raise Phase3MaterialWindowError(
            "additional_refresh_reasons must contain only nonempty strings."
        )
    active_rank = int(active_supported_rank)
    joint_rank = int(joint_supported_rank)
    # The supplied ranks belong to the retained Phase-III workspace, not to the
    # unmeasured full active block.  Using the full active count here would turn
    # every omitted coordinate into a fictitious null direction and would force
    # the very full-geometry refresh this selector is meant to avoid.
    active_dimension = len(receipt.retained_indices)
    joint_dimension = active_dimension + 1
    reasons: list[str] = []
    if not receipt.inputs_finite:
        reasons.append("nonfinite_input")
    if not receipt.closure_satisfied:
        reasons.append("closure_failed")

    active_rank_valid = 0 <= active_rank <= active_dimension
    joint_rank_valid = 0 <= joint_rank <= joint_dimension
    if not active_rank_valid:
        reasons.append("invalid_active_supported_rank")
    if not joint_rank_valid:
        reasons.append("invalid_joint_supported_rank")

    active_nullity = active_dimension - active_rank if active_rank_valid else None
    joint_nullity = joint_dimension - joint_rank if joint_rank_valid else None
    rank_gain = joint_rank - active_rank if active_rank_valid and joint_rank_valid else None
    if rank_gain is not None and rank_gain not in (0, 1):
        reasons.append("invalid_rank_gain")

    drift = False
    if (
        active_nullity is not None
        and receipt.prior_active_nullity is not None
        and active_nullity != receipt.prior_active_nullity
    ):
        drift = True
        reasons.append("active_support_nullity_drift")
    if (
        joint_nullity is not None
        and receipt.prior_joint_nullity is not None
        and joint_nullity != receipt.prior_joint_nullity
    ):
        drift = True
        reasons.append("joint_support_nullity_drift")
    for reason in external_reasons:
        if reason not in reasons:
            reasons.append(reason)

    finalized = replace(
        receipt,
        measured_active_supported_rank=active_rank,
        measured_joint_supported_rank=joint_rank,
        measured_active_nullity=active_nullity,
        measured_joint_nullity=joint_nullity,
        measured_rank_gain=rank_gain,
        support_nullity_drift=bool(drift),
        requires_full_geometry_refresh=bool(reasons),
        refresh_reasons=tuple(reasons),
        receipt_sha256="",
    )
    return _with_digest(finalized)
