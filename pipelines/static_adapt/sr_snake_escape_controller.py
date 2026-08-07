"""Typed precedence seam for SR-SNAKE saddle and local-minimum escape.

This module deliberately owns no circuit, optimizer, or pipeline state.  It
accepts certificates produced by those layers and makes only the discrete
selection decision described by the SR-SNAKE mathematical controller:

1. a resolved ordinary gain always keeps the literal ordinary route;
2. resolved active nonstationarity requests a guarded no-admission refit;
3. a certified stationary saddle may receive singleton credit only for a
   positive *marginal* trust gain with resolved quotient participation;
4. modeled-local-minimum exploration is exposed as eligibility only after a
   complete PSD/redundancy audit.

Constructing a ``SaddleCertificate`` or ``PsdCertificate`` asserts that the
upstream numerical-validity conjunction has already passed.  Resolved
nonstationarity is represented separately by ``NonstationaryCertificate``;
failed or unresolved stationarity, support, inertia, KKT, mapping, or
uncertainty checks must instead be represented by ``UnresolvedCertificate``.
Stage-B state transitions are not implemented here and therefore fail closed
at an eligibility decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import TypeAlias


SR_ESCAPE_DISABLED = "disabled"
SR_ESCAPE_SADDLE_ONLY = "saddle_only"
SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM = "saddle_plus_modeled_minimum"
SR_ESCAPE_MODE_CHOICES = (
    SR_ESCAPE_DISABLED,
    SR_ESCAPE_SADDLE_ONLY,
    SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM,
)

SR_ROUTE_FAMILY = "singleton_response_snake"
SR_ROUTE_PROFILE_DISABLED = "supported_whitened_adaptive_trust_v1"
SR_ROUTE_PROFILE_REDUCED_POWELL = (
    "supported_whitened_adaptive_trust_reduced_powell_v2"
)
SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED = (
    "supported_phase2_phase3_whitened_adaptive_trust_v2"
)
SR_ROUTE_PROFILE_SADDLE_ONLY = (
    "supported_whitened_adaptive_trust_saddle_escape_v2"
)
SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM = (
    "supported_whitened_adaptive_trust_saddle_modeled_minimum_escape_v2"
)
SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1 = "phase3_only_v1"
SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1 = "phase2_and_phase3_v1"
SR_COORDINATE_SOLVE_SCOPE_CHOICES = (
    SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
)
SR_POWELL_COORDINATE_CHART_AUTO = "auto"
SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1 = (
    "expanded_runtime_projected_logical_v1"
)
SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1 = (
    "logical_shared_reduced_v1"
)
SR_POWELL_COORDINATE_CHART_POLICY_CHOICES = (
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
)
SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES = (
    SR_POWELL_COORDINATE_CHART_AUTO,
    *SR_POWELL_COORDINATE_CHART_POLICY_CHOICES,
)
SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED = "registered_profile"
SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION = (
    "unpromoted_explicit_ablation"
)
SR_ROUTE_PROFILE_CONFORMANCE_CHOICES = (
    SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED,
    SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
)
SR_CONTROLLER_ABLATION_CONTRACT_OFF = "off"
SR_CONTROLLER_ABLATION_CONTRACT_NOVELTY_PRUNE_CONTROLS_V1 = (
    "novelty_prune_controls_v1"
)
SR_CONTROLLER_ABLATION_CONTRACT_NOVELTY_PRUNE_BEAM_CONTROLS_V1 = (
    "novelty_prune_beam_controls_v1"
)
SR_CONTROLLER_ABLATION_CONTRACT_CHOICES = (
    SR_CONTROLLER_ABLATION_CONTRACT_OFF,
    SR_CONTROLLER_ABLATION_CONTRACT_NOVELTY_PRUNE_CONTROLS_V1,
    SR_CONTROLLER_ABLATION_CONTRACT_NOVELTY_PRUNE_BEAM_CONTROLS_V1,
)
SR_POWELL_ROUTE_INSTANCE_RESOLUTION_SCHEMA = (
    "sr_powell_route_instance_resolution_v1"
)


class SREscapeMode(str, Enum):
    """Enabled SR-SNAKE escape controller depth."""

    DISABLED = SR_ESCAPE_DISABLED
    SADDLE_ONLY = SR_ESCAPE_SADDLE_ONLY
    SADDLE_PLUS_MODELED_MINIMUM = SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM


class SRControllerDecisionKind(str, Enum):
    """Mutually exclusive result of the precedence controller."""

    ORDINARY = "ordinary"
    ACTIVE_STATIONARITY_CORRECTION = "active_stationarity_correction"
    SADDLE_SINGLETON = "saddle_singleton"
    ACTIVE_ONLY_CORRECTION = "active_only_correction"
    MODELED_MINIMUM_ELIGIBLE = "modeled_minimum_eligible"
    ESCAPE_DISABLED = "escape_disabled"
    UNRESOLVED = "unresolved"
    NO_ACTION = "no_action"


def sr_route_profile(
    mode: SREscapeMode | str,
    *,
    coordinate_solve_scope: str = SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    powell_coordinate_chart_policy: str = SR_POWELL_COORDINATE_CHART_AUTO,
) -> str:
    """Return the stable SR-SNAKE profile id for an escape mode."""

    resolved_mode = SREscapeMode(mode)
    resolved_scope = str(coordinate_solve_scope).strip().lower()
    if resolved_scope not in SR_COORDINATE_SOLVE_SCOPE_CHOICES:
        raise ValueError(
            "coordinate_solve_scope must be one of "
            f"{list(SR_COORDINATE_SOLVE_SCOPE_CHOICES)}."
        )
    resolved_powell_chart = resolve_sr_powell_coordinate_chart_policy(
        resolved_mode,
        coordinate_solve_scope=resolved_scope,
        requested_policy=powell_coordinate_chart_policy,
    )
    if resolved_scope == SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1:
        if resolved_mode is not SREscapeMode.DISABLED:
            raise ValueError(
                "Phase-II SR whitening is currently defined only for the "
                "non-escape SR profile."
            )
        return SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED
    if resolved_mode is SREscapeMode.DISABLED:
        if (
            resolved_powell_chart
            == SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ):
            return SR_ROUTE_PROFILE_REDUCED_POWELL
        return SR_ROUTE_PROFILE_DISABLED
    if resolved_mode is SREscapeMode.SADDLE_ONLY:
        return SR_ROUTE_PROFILE_SADDLE_ONLY
    return SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM


def sr_route_profile_conformance(
    mode: SREscapeMode | str,
    *,
    coordinate_solve_scope: str = SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    powell_coordinate_chart_policy: str = SR_POWELL_COORDINATE_CHART_AUTO,
) -> str:
    """Classify one resolved SR route instance against its registered profile.

    The Phase-II+III profile remains registered with the reduced-logical Powell
    chart.  An explicit expanded-chart request is allowed for an unpromoted
    ablation, but the returned marker prevents that route instance from
    masquerading as the registered v2 profile.  ``auto`` never selects this
    ablation.
    """

    resolved_mode = SREscapeMode(mode)
    resolved_scope = str(coordinate_solve_scope).strip().lower()
    requested = str(powell_coordinate_chart_policy).strip().lower()
    resolved_policy = resolve_sr_powell_coordinate_chart_policy(
        resolved_mode,
        coordinate_solve_scope=resolved_scope,
        requested_policy=requested,
    )
    if (
        resolved_mode is SREscapeMode.DISABLED
        and resolved_scope == SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        and requested
        == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        and resolved_policy
        == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ):
        return SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
    return SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED


def resolve_sr_powell_route_instance(
    mode: SREscapeMode | str,
    *,
    coordinate_solve_scope: str = SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    requested_policy: str = SR_POWELL_COORDINATE_CHART_AUTO,
) -> dict[str, object]:
    """Return a serialization-safe Powell route-instance resolution record."""

    resolved_mode = SREscapeMode(mode)
    resolved_scope = str(coordinate_solve_scope).strip().lower()
    requested = str(requested_policy).strip().lower()
    resolved_policy = resolve_sr_powell_coordinate_chart_policy(
        resolved_mode,
        coordinate_solve_scope=resolved_scope,
        requested_policy=requested,
    )
    profile = sr_route_profile(
        resolved_mode,
        coordinate_solve_scope=resolved_scope,
        powell_coordinate_chart_policy=requested,
    )
    conformance = sr_route_profile_conformance(
        resolved_mode,
        coordinate_solve_scope=resolved_scope,
        powell_coordinate_chart_policy=requested,
    )
    return {
        "schema": SR_POWELL_ROUTE_INSTANCE_RESOLUTION_SCHEMA,
        "route_family": SR_ROUTE_FAMILY,
        "route_profile": str(profile),
        "route_profile_conformance": str(conformance),
        "escape_mode": str(resolved_mode.value),
        "coordinate_solve_scope": str(resolved_scope),
        "powell_coordinate_chart_policy_requested": str(requested),
        "powell_coordinate_chart_policy": str(resolved_policy),
        "request_was_auto": bool(requested == SR_POWELL_COORDINATE_CHART_AUTO),
        "inferred_unpromoted_ablation": False,
    }


def resolve_sr_powell_coordinate_chart_policy(
    mode: SREscapeMode | str,
    *,
    coordinate_solve_scope: str = SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    requested_policy: str = SR_POWELL_COORDINATE_CHART_AUTO,
) -> str:
    """Resolve the explicit Powell chart for one SR-SNAKE profile.

    The high-accuracy historical v1 profile is the only SR profile whose
    automatic chart is the expanded runtime chart.  Phase-II-whitened and
    escape profiles retain the newer reduced-logical automatic behavior.  An
    explicit expanded-chart request is additionally allowed for the
    escape-disabled Phase-II+III shape, but its route-instance conformance is
    marked as an unpromoted explicit ablation by
    :func:`resolve_sr_powell_route_instance`.
    """

    resolved_mode = SREscapeMode(mode)
    resolved_scope = str(coordinate_solve_scope).strip().lower()
    if resolved_scope not in SR_COORDINATE_SOLVE_SCOPE_CHOICES:
        raise ValueError(
            "coordinate_solve_scope must be one of "
            f"{list(SR_COORDINATE_SOLVE_SCOPE_CHOICES)}."
        )
    requested = str(requested_policy).strip().lower()
    if requested not in SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES:
        raise ValueError(
            "powell_coordinate_chart_policy must be one of "
            f"{list(SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES)}."
        )
    canonical_v1_shape = bool(
        resolved_mode is SREscapeMode.DISABLED
        and resolved_scope == SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1
    )
    if requested == SR_POWELL_COORDINATE_CHART_AUTO:
        return (
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
            if canonical_v1_shape
            else SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        )
    explicit_phase2_phase3_expanded_ablation = bool(
        requested
        == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        and resolved_mode is SREscapeMode.DISABLED
        and resolved_scope == SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
    )
    if (
        requested
        == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        and not canonical_v1_shape
        and not explicit_phase2_phase3_expanded_ablation
    ):
        raise ValueError(
            "expanded_runtime_projected_logical_v1 is reserved for canonical "
            "SR-SNAKE v1 or the explicit, escape-disabled Phase-II+III "
            "unpromoted ablation."
        )
    return requested


def sr_escape_record_id(
    candidate_label: str,
    candidate_pool_index: int,
    position_id: int,
) -> str:
    """Return the stable finite-population identity for one SR record."""

    label = str(candidate_label).strip()
    if not label:
        raise ValueError("candidate_label must be nonempty.")
    return (
        f"{label}::pool={int(candidate_pool_index)}"
        f"::position={int(position_id)}"
    )


def _validate_record_id(record_id: str) -> None:
    if not str(record_id).strip():
        raise ValueError("record_id must be nonempty.")


def reachable_population_digest(record_ids: tuple[str, ...]) -> str:
    """Return the ordered finite-population digest used by Stage-B tokens."""

    normalized = tuple(str(record_id) for record_id in record_ids)
    for record_id in normalized:
        _validate_record_id(record_id)
    return hashlib.sha256(
        json.dumps(
            list(normalized),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


@dataclass(frozen=True)
class OrdinaryCertificate:
    """Resolved positive-gain result selected by the ordinary SR funnel."""

    record_id: str
    gain_lower_bound: float

    def __post_init__(self) -> None:
        _validate_record_id(self.record_id)
        if _finite("gain_lower_bound", self.gain_lower_bound) <= 0.0:
            raise ValueError("gain_lower_bound must be resolved positive.")


@dataclass(frozen=True)
class SaddleCertificate:
    """Numerically valid, resolution-stationary, negative-curvature record.

    ``full_trust_gain_lower_bound`` and ``active_trust_gain_upper_bound`` are
    bounds from the same enlarged support and transported trust metric.  Their
    difference is the singleton's conservative marginal credit.  The Phase-III
    novelty statistic is carried only for diagnostics; it never multiplies the
    final saddle acquisition.
    """

    record_id: str
    stationarity_margin: float
    minimum_eigenvalue_upper_bound: float
    full_trust_gain_lower_bound: float
    active_trust_gain_lower_bound: float
    active_trust_gain_upper_bound: float
    quotient_participation_lower_bound: float
    phase3_cost: float
    novelty_statistic: float | None = None

    def __post_init__(self) -> None:
        _validate_record_id(self.record_id)
        if _finite("stationarity_margin", self.stationarity_margin) > 0.0:
            raise ValueError("stationarity_margin must certify stationarity.")
        if (
            _finite(
                "minimum_eigenvalue_upper_bound",
                self.minimum_eigenvalue_upper_bound,
            )
            >= 0.0
        ):
            raise ValueError(
                "minimum_eigenvalue_upper_bound must certify negative curvature."
            )
        _nonnegative(
            "full_trust_gain_lower_bound", self.full_trust_gain_lower_bound
        )
        active_lower = _nonnegative(
            "active_trust_gain_lower_bound", self.active_trust_gain_lower_bound
        )
        active_upper = _nonnegative(
            "active_trust_gain_upper_bound", self.active_trust_gain_upper_bound
        )
        if active_lower > active_upper:
            raise ValueError(
                "active_trust_gain_lower_bound cannot exceed its upper bound."
            )
        participation = _nonnegative(
            "quotient_participation_lower_bound",
            self.quotient_participation_lower_bound,
        )
        if participation > 1.0:
            raise ValueError(
                "quotient_participation_lower_bound cannot exceed one."
            )
        _nonnegative("phase3_cost", self.phase3_cost)
        if self.novelty_statistic is not None:
            novelty = _nonnegative("novelty_statistic", self.novelty_statistic)
            if novelty > 1.0:
                raise ValueError("novelty_statistic cannot exceed one.")

    @property
    def marginal_gain_lower_bound(self) -> float:
        """Conservative full-minus-active singleton credit."""

        return float(
            self.full_trust_gain_lower_bound
            - self.active_trust_gain_upper_bound
        )

    @property
    def has_singleton_credit(self) -> bool:
        return bool(
            self.marginal_gain_lower_bound > 0.0
            and self.quotient_participation_lower_bound > 0.0
        )

    @property
    def has_actionable_active_restriction(self) -> bool:
        return bool(self.active_trust_gain_lower_bound > 0.0)


@dataclass(frozen=True)
class NonstationaryCertificate:
    """Valid supported model requiring an active-only stationarity repair.

    This is an ordinary no-admission refit certificate, not a saddle
    certificate.  The active restriction must come from the same supported
    model as the record and must have a resolved positive gain lower bound.
    Its exact mapped seed remains subject to the pipeline's atomic downhill
    guard before any branch state can change.
    """

    record_id: str
    stationarity_margin: float
    active_trust_gain_lower_bound: float
    active_trust_gain_upper_bound: float

    def __post_init__(self) -> None:
        _validate_record_id(self.record_id)
        if _finite("stationarity_margin", self.stationarity_margin) <= 0.0:
            raise ValueError(
                "stationarity_margin must certify resolved nonstationarity."
            )
        active_lower = _nonnegative(
            "active_trust_gain_lower_bound",
            self.active_trust_gain_lower_bound,
        )
        active_upper = _nonnegative(
            "active_trust_gain_upper_bound",
            self.active_trust_gain_upper_bound,
        )
        if active_lower <= 0.0:
            raise ValueError(
                "active_trust_gain_lower_bound must be resolved positive."
            )
        if active_lower > active_upper:
            raise ValueError(
                "active_trust_gain_lower_bound cannot exceed its upper bound."
            )


@dataclass(frozen=True)
class PsdCertificate:
    """Numerically valid, resolution-stationary, certified-PSD record."""

    record_id: str
    stationarity_margin: float
    minimum_eigenvalue_lower_bound: float

    def __post_init__(self) -> None:
        _validate_record_id(self.record_id)
        if _finite("stationarity_margin", self.stationarity_margin) > 0.0:
            raise ValueError("stationarity_margin must certify stationarity.")
        if (
            _finite(
                "minimum_eigenvalue_lower_bound",
                self.minimum_eigenvalue_lower_bound,
            )
            < 0.0
        ):
            raise ValueError(
                "minimum_eigenvalue_lower_bound must certify PSD inertia."
            )


@dataclass(frozen=True)
class QuotientRedundantCertificate:
    """Record certified to add no resolved first-order quotient direction."""

    record_id: str
    quotient_norm_upper_bound: float
    support_resolution_floor: float

    def __post_init__(self) -> None:
        _validate_record_id(self.record_id)
        quotient_upper = _nonnegative(
            "quotient_norm_upper_bound", self.quotient_norm_upper_bound
        )
        support_floor = _nonnegative(
            "support_resolution_floor", self.support_resolution_floor
        )
        if quotient_upper > support_floor:
            raise ValueError(
                "quotient norm must be at or below the support-resolution floor."
            )


@dataclass(frozen=True)
class UnresolvedCertificate:
    """Fail-closed outcome for invalid or unresolved numerical evidence."""

    record_id: str
    reason: str

    def __post_init__(self) -> None:
        _validate_record_id(self.record_id)
        if not str(self.reason).strip():
            raise ValueError("reason must be nonempty.")


@dataclass(frozen=True)
class StateStationarityCertificate:
    """State-bound supported-stationarity token for Stage-B eligibility.

    Per-record quotient redundancy is not a stationarity statement about the
    working physical state.  This independent token therefore binds the
    stationarity comparison to that state, the exact ordered reachable
    population, the live trust radius, the comparison epoch, and the support
    and transported-trust provenance used by the audit.
    """

    state_fingerprint: str
    reachable_population_digest: str
    comparison_epoch: str
    support_provenance_digest: str
    trust_provenance_digest: str
    trust_radius: float
    stationarity_margin: float

    def __post_init__(self) -> None:
        for name, value in (
            ("state_fingerprint", self.state_fingerprint),
            ("reachable_population_digest", self.reachable_population_digest),
            ("comparison_epoch", self.comparison_epoch),
            ("support_provenance_digest", self.support_provenance_digest),
            ("trust_provenance_digest", self.trust_provenance_digest),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be nonempty.")
        if _finite("trust_radius", self.trust_radius) <= 0.0:
            raise ValueError("trust_radius must be strictly positive.")
        if _finite("stationarity_margin", self.stationarity_margin) > 0.0:
            raise ValueError("stationarity_margin must certify stationarity.")

    def as_dict(self) -> dict[str, str | float]:
        return {
            "state_fingerprint": str(self.state_fingerprint),
            "reachable_population_digest": str(
                self.reachable_population_digest
            ),
            "comparison_epoch": str(self.comparison_epoch),
            "support_provenance_digest": str(
                self.support_provenance_digest
            ),
            "trust_provenance_digest": str(self.trust_provenance_digest),
            "trust_radius": float(self.trust_radius),
            "stationarity_margin": float(self.stationarity_margin),
        }

    @property
    def token_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()


ReachableCertificate: TypeAlias = (
    NonstationaryCertificate
    | SaddleCertificate
    | PsdCertificate
    | QuotientRedundantCertificate
    | UnresolvedCertificate
)


@dataclass(frozen=True)
class ReachablePopulationAudit:
    """Finite reachable population and its per-record certificates.

    ``reachable_record_ids`` is also the fixed deterministic order used for
    exact acquisition ties.  A missing certificate makes the audit incomplete;
    an extra or duplicate certificate is rejected as a population mismatch.
    """

    reachable_record_ids: tuple[str, ...]
    certificates: tuple[ReachableCertificate, ...]
    state_stationarity: StateStationarityCertificate | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reachable_record_ids",
            tuple(str(record_id) for record_id in self.reachable_record_ids),
        )
        object.__setattr__(self, "certificates", tuple(self.certificates))
        for record_id in self.reachable_record_ids:
            _validate_record_id(record_id)
        if len(set(self.reachable_record_ids)) != len(self.reachable_record_ids):
            raise ValueError("reachable_record_ids must be unique.")
        certificate_ids = tuple(
            certificate.record_id for certificate in self.certificates
        )
        if len(set(certificate_ids)) != len(certificate_ids):
            raise ValueError("certificates must contain at most one record each.")
        unexpected = set(certificate_ids) - set(self.reachable_record_ids)
        if unexpected:
            raise ValueError(
                "certificates contain records outside the reachable population: "
                f"{sorted(unexpected)}"
            )

    @property
    def complete(self) -> bool:
        return bool(
            set(certificate.record_id for certificate in self.certificates)
            == set(self.reachable_record_ids)
        )

    @property
    def has_unresolved_record(self) -> bool:
        return any(
            isinstance(certificate, UnresolvedCertificate)
            for certificate in self.certificates
        )

    @property
    def records_all_psd_or_redundant(self) -> bool:
        return bool(
            self.complete
            and self.certificates
            and all(
                isinstance(
                    certificate,
                    (PsdCertificate, QuotientRedundantCertificate),
                )
                for certificate in self.certificates
            )
        )

    @property
    def expected_population_digest(self) -> str:
        return reachable_population_digest(self.reachable_record_ids)

    @property
    def state_stationarity_certified(self) -> bool:
        certificate = self.state_stationarity
        return bool(
            certificate is not None
            and certificate.reachable_population_digest
            == self.expected_population_digest
        )

    @property
    def all_psd_or_redundant(self) -> bool:
        """Whether the full represented family and working state are certified."""

        return bool(
            self.records_all_psd_or_redundant
            and self.state_stationarity_certified
        )

    def order_index(self, record_id: str) -> int:
        return self.reachable_record_ids.index(record_id)


@dataclass(frozen=True)
class SRControllerDecision:
    """Pure result of one precedence evaluation."""

    kind: SRControllerDecisionKind
    reason: str
    record_id: str | None = None
    certificate_record_id: str | None = None
    consumes_singleton: bool = False
    actionable: bool = False
    stage_b_eligible: bool = False
    acquisition: float = 0.0
    marginal_gain_lower_bound: float = 0.0


def saddle_acquisition(certificate: SaddleCertificate) -> float:
    """Return ``[Delta q lower]_+ / (1 + K3)``.

    ``certificate.novelty_statistic`` is intentionally absent from the
    expression.  Quotient participation is a strict eligibility gate, not an
    additional score multiplier.
    """

    return float(
        max(certificate.marginal_gain_lower_bound, 0.0)
        / (1.0 + certificate.phase3_cost)
    )


def select_sr_escape_path(
    *,
    mode: SREscapeMode | str,
    ordinary: OrdinaryCertificate | None,
    audit: ReachablePopulationAudit | None,
) -> SRControllerDecision:
    """Apply ordinary -> active-stationarity -> saddle -> modeled-minimum precedence.

    The finite audit must be complete before this exact-selection seam chooses
    an actionable saddle or exposes Stage-B eligibility.  An unresolved
    bystander cannot invalidate an independently certified saddle/active-only
    action, but it does block PSD/no-saddle exhaustion and therefore Stage-B
    eligibility.  This prevents a partial population from manufacturing either
    a global acquisition maximum or a PSD exposed-family label.
    """

    resolved_mode = SREscapeMode(mode)

    if ordinary is not None:
        return SRControllerDecision(
            kind=SRControllerDecisionKind.ORDINARY,
            reason="resolved_positive_ordinary_gain",
            record_id=ordinary.record_id,
            certificate_record_id=ordinary.record_id,
            consumes_singleton=True,
            actionable=True,
        )

    if resolved_mode is SREscapeMode.DISABLED:
        return SRControllerDecision(
            kind=SRControllerDecisionKind.ESCAPE_DISABLED,
            reason="ordinary_unusable_and_escape_disabled",
        )

    if audit is None:
        return SRControllerDecision(
            kind=SRControllerDecisionKind.UNRESOLVED,
            reason="reachable_population_audit_missing",
        )
    if not audit.complete:
        return SRControllerDecision(
            kind=SRControllerDecisionKind.UNRESOLVED,
            reason="reachable_population_audit_incomplete",
        )
    nonstationary_certificates = tuple(
        certificate
        for certificate in audit.certificates
        if isinstance(certificate, NonstationaryCertificate)
    )
    if nonstationary_certificates:
        selected_nonstationary = min(
            nonstationary_certificates,
            key=lambda certificate: (
                -certificate.active_trust_gain_lower_bound,
                audit.order_index(certificate.record_id),
            ),
        )
        return SRControllerDecision(
            kind=SRControllerDecisionKind.ACTIVE_STATIONARITY_CORRECTION,
            reason=(
                "active_nonstationarity_requires_no_admission_refit_before_"
                "saddle_or_psd_classification"
            ),
            certificate_record_id=selected_nonstationary.record_id,
            consumes_singleton=False,
            actionable=True,
        )
    saddle_certificates = tuple(
        certificate
        for certificate in audit.certificates
        if isinstance(certificate, SaddleCertificate)
    )
    credited_saddles = tuple(
        certificate
        for certificate in saddle_certificates
        if certificate.has_singleton_credit
    )
    if credited_saddles:
        selected = min(
            credited_saddles,
            key=lambda certificate: (
                -saddle_acquisition(certificate),
                audit.order_index(certificate.record_id),
            ),
        )
        return SRControllerDecision(
            kind=SRControllerDecisionKind.SADDLE_SINGLETON,
            reason="certified_positive_marginal_saddle_gain",
            record_id=selected.record_id,
            certificate_record_id=selected.record_id,
            consumes_singleton=True,
            actionable=True,
            acquisition=saddle_acquisition(selected),
            marginal_gain_lower_bound=selected.marginal_gain_lower_bound,
        )

    active_only = tuple(
        certificate
        for certificate in saddle_certificates
        if certificate.has_actionable_active_restriction
    )
    if active_only:
        selected_active = min(
            active_only,
            key=lambda certificate: (
                -certificate.active_trust_gain_lower_bound,
                audit.order_index(certificate.record_id),
            ),
        )
        return SRControllerDecision(
            kind=SRControllerDecisionKind.ACTIVE_ONLY_CORRECTION,
            reason="active_instability_has_no_certified_singleton_marginal_credit",
            certificate_record_id=selected_active.record_id,
            consumes_singleton=False,
            actionable=True,
        )

    if audit.has_unresolved_record:
        return SRControllerDecision(
            kind=SRControllerDecisionKind.UNRESOLVED,
            reason="reachable_population_contains_unresolved_certificate",
        )

    if saddle_certificates:
        return SRControllerDecision(
            kind=SRControllerDecisionKind.UNRESOLVED,
            reason="certified_saddle_has_no_actionable_marginal_or_active_step",
        )

    if audit.records_all_psd_or_redundant:
        if resolved_mode is SREscapeMode.SADDLE_PLUS_MODELED_MINIMUM:
            if not audit.state_stationarity_certified:
                return SRControllerDecision(
                    kind=SRControllerDecisionKind.UNRESOLVED,
                    reason=(
                        "state_stationarity_certificate_missing_or_"
                        "population_stale"
                    ),
                )
            return SRControllerDecision(
                kind=SRControllerDecisionKind.MODELED_MINIMUM_ELIGIBLE,
                reason=(
                    "complete_reachable_population_is_psd_or_redundant_"
                    "and_state_stationary"
                ),
                stage_b_eligible=True,
                # Stage B is deliberately not executed by this seam.
                actionable=False,
            )
        return SRControllerDecision(
            kind=SRControllerDecisionKind.NO_ACTION,
            reason="modeled_minimum_escape_not_enabled",
        )

    return SRControllerDecision(
        kind=SRControllerDecisionKind.UNRESOLVED,
        reason="reachable_population_does_not_support_an_escape_classification",
    )


__all__ = [
    "NonstationaryCertificate",
    "OrdinaryCertificate",
    "PsdCertificate",
    "QuotientRedundantCertificate",
    "ReachablePopulationAudit",
    "SRControllerDecision",
    "SRControllerDecisionKind",
    "SREscapeMode",
    "StateStationarityCertificate",
    "SR_ESCAPE_DISABLED",
    "SR_ESCAPE_MODE_CHOICES",
    "SR_ESCAPE_SADDLE_ONLY",
    "SR_ESCAPE_SADDLE_PLUS_MODELED_MINIMUM",
    "SR_ROUTE_FAMILY",
    "SR_ROUTE_PROFILE_DISABLED",
    "SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED",
    "SR_ROUTE_PROFILE_REDUCED_POWELL",
    "SR_ROUTE_PROFILE_SADDLE_ONLY",
    "SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM",
    "SR_POWELL_COORDINATE_CHART_AUTO",
    "SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1",
    "SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1",
    "SR_POWELL_COORDINATE_CHART_POLICY_CHOICES",
    "SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES",
    "SR_POWELL_ROUTE_INSTANCE_RESOLUTION_SCHEMA",
    "SR_ROUTE_PROFILE_CONFORMANCE_CHOICES",
    "SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED",
    "SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION",
    "SaddleCertificate",
    "UnresolvedCertificate",
    "saddle_acquisition",
    "select_sr_escape_path",
    "reachable_population_digest",
    "resolve_sr_powell_coordinate_chart_policy",
    "resolve_sr_powell_route_instance",
    "sr_route_profile",
    "sr_route_profile_conformance",
]
