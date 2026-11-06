"""Pure comparator-first orchestration for the SIAM ``n_bath=4`` pilot.

The planner in this module has no executor side effects.  It accepts the
current immutable action frontier, validates every completed action against a
positive raw worker receipt (and, for a scored continuation, the final stitch
proof), and returns the exact actions that a later CHTC executor may prepare.

Threshold selection is deliberately mechanical: only the next entry in
``SIAM_NB4_ACTIVATION_CUT_LADDER`` can follow an attempted screen.  Measured
accuracy decides whether another rung is needed, but it is never used to skip
rungs or infer monotonicity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
    EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    EXCHANGE_ROUTE_BUNDLES,
)
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    SIAM_NB4_PRIMITIVE_NAMES,
    SIAM_ROUTE_HISTORICAL_UNRECORDED,
    SIAM_ROUTE_NOT_APPLICABLE,
    _verified_score,
    classify_siam_matched_pair,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_STITCH_COMPLETION_V1,
    validate_siam_nb4_stitch_completion_receipt,
)
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_SEGMENT,
    SIAM_NB4_SCREEN_SEGMENT,
)
from pipelines.time_dynamics.siam_activation_ladder import (
    SIAM_NB4_ACTIVATION_CUT_LADDER,
    ladder_contract_json,
)
from pipelines.time_dynamics.siam_chtc_package import (
    validate_siam_nb4_completion_receipt,
)


SIAM_NB4_LADDER_PLAN_V1 = "siam_nb4_comparator_first_ladder_plan_v1"

ARM_AVQDS = "avqds"
ARM_FULL_APM = "full_apm"
ARMS = (ARM_AVQDS, ARM_FULL_APM)

STATE_SUBMITTED = "submitted"
STATE_IN_FLIGHT = "in_flight"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
ACTION_STATES = frozenset(
    {STATE_SUBMITTED, STATE_IN_FLIGHT, STATE_COMPLETED, STATE_FAILED}
)

KIND_RUN = "run"
KIND_RECORD_SAFEGUARD = "record_safeguard"

SEGMENT_SCREEN = SIAM_NB4_SCREEN_SEGMENT.segment_id
SEGMENT_CONTINUATION = SIAM_NB4_CONTINUATION_SEGMENT.segment_id
SEGMENTS = (SEGMENT_SCREEN, SEGMENT_CONTINUATION)

_RECEIPT_ARM = {ARM_AVQDS: "avqds", ARM_FULL_APM: "exchange"}
_SCORE_ARM = _RECEIPT_ARM
_ARM_ORDER = {ARM_AVQDS: 0, ARM_FULL_APM: 1}

# One ladder belongs to exactly one exchange route.  ``historical`` is the
# pre-route lineage and keeps every action identity byte-identical: its token is
# ``None``, so no token is ever inserted into a historical action ID.  A
# registered route bundle gets a short token, which makes a compiled-Q ladder a
# genuinely new ladder -- new action IDs, opening at r00 in the AVQDS lane --
# rather than a continuation of the historical one.
ROUTE_HISTORICAL = "historical"
ROUTE_Q_COMPILED = EXCHANGE_ROUTE_BUNDLE_Q_COMPILED
ROUTE_TOKENS: Mapping[str, str | None] = {
    ROUTE_HISTORICAL: None,
    ROUTE_Q_COMPILED: "qc",
}

# The historical lineage accepts either a score that predates the route guards
# or one that explicitly declares the registered historical bundle.
_HISTORICAL_BUNDLE_ID = EXCHANGE_ROUTE_BUNDLES[
    EXCHANGE_ROUTE_BUNDLE_HISTORICAL
].bundle_id
# A registered route other than the historical one accepts exactly its own
# bundle id, so the table stays derived from ``ROUTE_TOKENS`` and cannot drift
# out of sync with it.
_ACCEPTED_EXCHANGE_SCORE_ROUTES: Mapping[str, frozenset[str]] = {
    route: (
        frozenset({SIAM_ROUTE_HISTORICAL_UNRECORDED, _HISTORICAL_BUNDLE_ID})
        if route == ROUTE_HISTORICAL
        else frozenset({route})
    )
    for route in ROUTE_TOKENS
}
# The comparator arm is on no exchange route at all.  A pre-route AVQDS score
# simply never recorded the field.
_ACCEPTED_AVQDS_SCORE_ROUTES = frozenset(
    {SIAM_ROUTE_NOT_APPLICABLE, SIAM_ROUTE_HISTORICAL_UNRECORDED}
)


class SIAMNb4LadderError(ValueError):
    """The supplied action history cannot drive the canonical SIAM ladder."""


def _registered_cut(value: float) -> tuple[float, int]:
    cut = float(value)
    if not math.isfinite(cut) or cut <= 0.0:
        raise SIAMNb4LadderError(
            "SIAM Nb4 ladder activation cut must be finite and positive."
        )
    for index, rung in enumerate(SIAM_NB4_ACTIVATION_CUT_LADDER):
        if cut == rung:
            return rung, index
    raise SIAMNb4LadderError(
        "SIAM Nb4 ladder action uses a nonregistered activation cut: "
        f"{cut!r}."
    )


def _canonical_route(value: str) -> str:
    route = str(value).strip()
    if route not in ROUTE_TOKENS:
        raise SIAMNb4LadderError(
            f"SIAM Nb4 ladder route must be one of {sorted(ROUTE_TOKENS)!r}; "
            f"got {value!r}."
        )
    return route


def _canonical_arm(value: str) -> str:
    arm = str(value).strip().lower().replace("-", "_")
    aliases = {
        "apm": ARM_FULL_APM,
        "exchange": ARM_FULL_APM,
        "full_apm": ARM_FULL_APM,
        "avqds": ARM_AVQDS,
    }
    try:
        return aliases[arm]
    except KeyError as exc:
        raise SIAMNb4LadderError(
            f"SIAM Nb4 ladder arm must be one of {ARMS!r}; got {value!r}."
        ) from exc


def _canonical_segment(value: str) -> str:
    segment = str(value).strip().lower()
    aliases = {
        "screen": SEGMENT_SCREEN,
        SEGMENT_SCREEN: SEGMENT_SCREEN,
        "continuation": SEGMENT_CONTINUATION,
        "continue": SEGMENT_CONTINUATION,
        SEGMENT_CONTINUATION: SEGMENT_CONTINUATION,
    }
    try:
        return aliases[segment]
    except KeyError as exc:
        raise SIAMNb4LadderError(
            f"SIAM Nb4 ladder segment must be one of {SEGMENTS!r}; "
            f"got {value!r}."
        ) from exc


def _cut_token(cut: float) -> str:
    mantissa, exponent = f"{float(cut):.0e}".split("e")
    sign = "m" if exponent.startswith("-") else "p"
    digits = exponent.lstrip("+-").zfill(2)
    return f"{mantissa}e{sign}{digits}"


def siam_nb4_action_id(
    *,
    arm: str,
    segment: str,
    activation_cut: float,
    route_id: str = ROUTE_HISTORICAL,
) -> str:
    """Return the unique scientific action identity for one exact rung.

    The route token is inserted only when the route declares one.  The
    historical route declares ``None``, so every historical action ID is
    byte-identical to the pre-route identity.
    """

    canonical_arm = _canonical_arm(arm)
    canonical_segment = _canonical_segment(segment)
    cut, index = _registered_cut(activation_cut)
    segment_token = "screen" if canonical_segment == SEGMENT_SCREEN else "continue"
    route_token = ROUTE_TOKENS[_canonical_route(route_id)]
    prefix = "siam-nb4-" if route_token is None else f"siam-nb4-{route_token}-"
    return (
        f"{prefix}{canonical_arm}-{segment_token}-"
        f"r{index:02d}-cut{_cut_token(cut)}"
    )


def _legacy_r00_action_id(*, arm: str, segment: str) -> str:
    """Return the immutable pre-migration identity for cut ``1e-3``."""

    canonical_arm = _canonical_arm(arm)
    canonical_segment = _canonical_segment(segment)
    segment_token = "screen" if canonical_segment == SEGMENT_SCREEN else "continue"
    return f"siam-nb4-{canonical_arm}-{segment_token}-r00-cut1em03"


def _is_legacy_r00_identity(
    action_id: str, *, arm: str, segment: str, activation_cut: float
) -> bool:
    return (
        float(activation_cut) == 1.0e-3
        and str(action_id) == _legacy_r00_action_id(arm=arm, segment=segment)
    )


@dataclass(frozen=True)
class SIAMNb4ActionRecord:
    """Current state and immutable evidence for one previously emitted action."""

    action_id: str
    arm: str
    segment: str
    activation_cut: float
    state: str
    route_id: str = ROUTE_HISTORICAL
    resume_from_action_id: str | None = None
    completion_receipt: Mapping[str, Any] | None = None
    score: Mapping[str, Any] | None = None
    score_completion_proof: Mapping[str, Any] | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        arm = _canonical_arm(self.arm)
        segment = _canonical_segment(self.segment)
        route_id = _canonical_route(self.route_id)
        cut, _index = _registered_cut(self.activation_cut)
        state = str(self.state).strip().lower()
        if state not in ACTION_STATES:
            raise SIAMNb4LadderError(
                f"SIAM Nb4 action state must be one of {sorted(ACTION_STATES)}."
            )
        expected_id = siam_nb4_action_id(
            arm=arm, segment=segment, activation_cut=cut, route_id=route_id
        )
        legacy_identity = route_id == ROUTE_HISTORICAL and _is_legacy_r00_identity(
            str(self.action_id),
            arm=arm,
            segment=segment,
            activation_cut=cut,
        )
        if str(self.action_id) != expected_id and not legacy_identity:
            raise SIAMNb4LadderError(
                "SIAM Nb4 action ID does not match its arm/segment/cut: "
                f"expected {expected_id!r} (or the immutable legacy r00 alias), "
                f"got {self.action_id!r}."
            )
        if segment == SEGMENT_SCREEN and self.resume_from_action_id is not None:
            raise SIAMNb4LadderError("A SIAM screen action cannot resume another action.")
        if segment == SEGMENT_CONTINUATION:
            expected_screen = siam_nb4_action_id(
                arm=arm,
                segment=SEGMENT_SCREEN,
                activation_cut=cut,
                route_id=route_id,
            )
            accepted_screens = {expected_screen}
            if route_id == ROUTE_HISTORICAL and cut == 1.0e-3:
                accepted_screens.add(
                    _legacy_r00_action_id(arm=arm, segment=SEGMENT_SCREEN)
                )
            if self.resume_from_action_id not in accepted_screens:
                raise SIAMNb4LadderError(
                    "A SIAM continuation must name its same-cut screen action."
                )
        if state == STATE_COMPLETED and self.completion_receipt is None:
            raise SIAMNb4LadderError(
                "A completed SIAM action requires its positive raw receipt."
            )
        if state in {STATE_SUBMITTED, STATE_IN_FLIGHT} and any(
            value is not None
            for value in (
                self.completion_receipt,
                self.score,
                self.score_completion_proof,
            )
        ):
            raise SIAMNb4LadderError(
                "A submitted/in-flight SIAM action cannot carry completed evidence."
            )
        if state == STATE_FAILED:
            if not str(self.failure_reason).strip():
                raise SIAMNb4LadderError(
                    "A failed SIAM action requires an explicit failure reason."
                )
            if self.score is not None or self.score_completion_proof is not None:
                raise SIAMNb4LadderError(
                    "A failed SIAM action cannot carry a promotable score."
                )
        if self.score is None and self.score_completion_proof is not None:
            raise SIAMNb4LadderError(
                "A SIAM score-completion proof cannot exist without a score."
            )
        if (
            self.score is not None
            and segment == SEGMENT_CONTINUATION
            and self.score_completion_proof is None
        ):
            raise SIAMNb4LadderError(
                "A scored SIAM continuation requires the final stitch proof."
            )

        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "segment", segment)
        object.__setattr__(self, "route_id", route_id)
        object.__setattr__(self, "activation_cut", cut)
        object.__setattr__(self, "state", state)
        for field_name in (
            "completion_receipt",
            "score",
            "score_completion_proof",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, dict(value))

    @property
    def ladder_index(self) -> int:
        return _registered_cut(self.activation_cut)[1]

    @property
    def scientific_identity(self) -> tuple[str, str, float]:
        return (self.arm, self.segment, self.activation_cut)

    @property
    def uses_legacy_r00_identity(self) -> bool:
        if self.route_id != ROUTE_HISTORICAL:
            return False
        return _is_legacy_r00_identity(
            self.action_id,
            arm=self.arm,
            segment=self.segment,
            activation_cut=self.activation_cut,
        )


@dataclass(frozen=True)
class SIAMNb4SafeguardRecord:
    """Durable terminal marker for one arm that exhausted the fixed ladder."""

    action_id: str
    arm: str
    activation_cut: float
    reason: str

    def __post_init__(self) -> None:
        arm = _canonical_arm(self.arm)
        cut, index = _registered_cut(self.activation_cut)
        if index != len(SIAM_NB4_ACTIVATION_CUT_LADDER) - 1:
            raise SIAMNb4LadderError(
                "A SIAM ladder-floor safeguard must name the registered floor cut."
            )
        expected_id = _safeguard_action_id(arm)
        if str(self.action_id) != expected_id:
            raise SIAMNb4LadderError(
                "SIAM ladder-floor safeguard ID does not match its arm."
            )
        if not str(self.reason).strip():
            raise SIAMNb4LadderError(
                "A SIAM ladder-floor safeguard requires an explicit reason."
            )
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "activation_cut", cut)


@dataclass(frozen=True)
class SIAMNb4NextAction:
    """One exact action that a later executor may prepare or record."""

    kind: str
    action_id: str
    arm: str
    segment: str
    activation_cut: float
    ladder_index: int
    reason: str
    route_id: str = ROUTE_HISTORICAL
    resume_from_action_id: str | None = None
    exact_reference_blind: bool = True
    qiskit_compilation_requested: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "action_id": self.action_id,
            "arm": self.arm,
            "route_id": self.route_id,
            "segment": self.segment,
            "activation_cut": self.activation_cut,
            "ladder_index": self.ladder_index,
            "reason": self.reason,
            "resume_from_action_id": self.resume_from_action_id,
            "exact_reference_blind": self.exact_reference_blind,
            "qiskit_compilation_requested": self.qiskit_compilation_requested,
        }


@dataclass(frozen=True)
class SIAMNb4Disposition:
    """Machine-readable explanation of one planner decision."""

    subject_id: str
    disposition: str
    reason: str
    arm: str | None = None
    segment: str | None = None
    activation_cut: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "disposition": self.disposition,
            "reason": self.reason,
            "arm": self.arm,
            "segment": self.segment,
            "activation_cut": self.activation_cut,
        }


@dataclass(frozen=True)
class SIAMNb4LadderPlan:
    """Pure planner output; no action in this document has been submitted."""

    next_actions: tuple[SIAMNb4NextAction, ...]
    dispositions: tuple[SIAMNb4Disposition, ...]
    candidate_action_ids: tuple[str, ...]
    selected_full_horizon_action_ids: Mapping[str, str] = field(default_factory=dict)
    recorded_safeguard_ids: tuple[str, ...] = ()
    cost_comparison_admissible: bool = False
    complete: bool = False
    blocked: bool = False
    phase: str = "avqds_screening"
    route_id: str = ROUTE_HISTORICAL

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": SIAM_NB4_LADDER_PLAN_V1,
            "route_id": self.route_id,
            "activation_cut_ladder_contract": ladder_contract_json(),
            "next_actions": [action.to_json_dict() for action in self.next_actions],
            "dispositions": [row.to_json_dict() for row in self.dispositions],
            "candidate_action_ids": list(self.candidate_action_ids),
            "selected_full_horizon_action_ids": dict(
                self.selected_full_horizon_action_ids
            ),
            "recorded_safeguard_ids": list(self.recorded_safeguard_ids),
            "cost_comparison_admissible": self.cost_comparison_admissible,
            "complete": self.complete,
            "blocked": self.blocked,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class _VerifiedRecord:
    record: SIAMNb4ActionRecord
    receipt: Mapping[str, Any] | None
    score: Mapping[str, Any] | None
    screen_broad_safe: bool | None
    screen_individual_target: bool | None


@dataclass(frozen=True)
class _LaneState:
    arm: str
    latest_screen: _VerifiedRecord | None
    current_full: _VerifiedRecord | None
    waiting_or_failed: bool


class _DispositionBook:
    def __init__(self) -> None:
        self._rows: dict[str, SIAMNb4Disposition] = {}

    def set(
        self,
        subject_id: str,
        disposition: str,
        reason: str,
        *,
        arm: str | None = None,
        segment: str | None = None,
        activation_cut: float | None = None,
    ) -> None:
        self._rows[subject_id] = SIAMNb4Disposition(
            subject_id=subject_id,
            disposition=disposition,
            reason=reason,
            arm=arm,
            segment=segment,
            activation_cut=activation_cut,
        )

    def values(self) -> tuple[SIAMNb4Disposition, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))


def _score_guard_state(score: Mapping[str, Any]) -> tuple[bool, bool, tuple[str, ...]]:
    thresholds = score.get("thresholds")
    summary = score.get("summary")
    primitive = score.get("primitive_response_metrics")
    diagnostics = score.get("diagnostics")
    if not all(
        isinstance(value, Mapping)
        for value in (thresholds, summary, primitive, diagnostics)
    ):
        raise SIAMNb4LadderError("SIAM ladder score lacks canonical gate fields.")
    try:
        response_values = [
            float(metric["response_normalized_mean_absolute_error"])
            for metric in primitive.values()
        ]
        individual_ceiling = float(thresholds["individual_response_target"])
        broad_ceiling = float(thresholds["broad_response_guard"])
        fidelity = float(summary["mean_infidelity"])
        fidelity_ceiling = float(thresholds["mean_infidelity_ceiling"])
        energy = diagnostics["energy"]
        energy_guard = float(energy["g_E"])
        energy_ceiling = float(thresholds["relative_energy_drift_ceiling"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SIAMNb4LadderError(
            "SIAM ladder score has malformed gate values."
        ) from exc
    values = (*response_values, fidelity, energy_guard)
    if not response_values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise SIAMNb4LadderError("SIAM ladder score gate values must be finite.")
    max_response = max(response_values)
    failures: list[str] = []
    if max_response > broad_ceiling:
        failures.append("broad_response_guard")
    if fidelity > fidelity_ceiling:
        failures.append("fidelity_ceiling")
    if energy_guard > energy_ceiling:
        failures.append("energy_guard")
    broad_safe = not failures
    individual = bool(
        max_response <= individual_ceiling
        and fidelity <= fidelity_ceiling
        and energy_guard <= energy_ceiling
    )
    declared = {
        "individual_target_pass": individual,
        "broad_response_guard_pass": broad_safe,
        "fidelity_pass": fidelity <= fidelity_ceiling,
        "energy_conservation_pass": energy_guard <= energy_ceiling,
    }
    for name, expected in declared.items():
        if summary.get(name) is not expected:
            raise SIAMNb4LadderError(
                f"SIAM ladder score has inconsistent summary.{name}."
            )
    return broad_safe, individual, tuple(failures)


def _verified_completed_record(record: SIAMNb4ActionRecord) -> _VerifiedRecord:
    if record.state != STATE_COMPLETED:
        return _VerifiedRecord(record, None, None, None, None)
    try:
        receipt = validate_siam_nb4_completion_receipt(
            record.completion_receipt or {},
            expected_arm=_RECEIPT_ARM[record.arm],
            expected_segment=record.segment,
        )
    except (ValueError, SystemExit) as exc:
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} has an invalid raw receipt: {exc}"
        ) from exc
    if receipt.get("action_id") != record.action_id:
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} raw-receipt identity mismatch."
        )
    if float(receipt.get("activation_cut", math.nan)) != record.activation_cut:
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} raw-receipt cut mismatch."
        )
    if record.score is None:
        return _VerifiedRecord(record, receipt, None, None, None)
    try:
        verified_score = _verified_score(
            record.score,
            required_primitive_names=SIAM_NB4_PRIMITIVE_NAMES,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} has an invalid score: {exc}"
        ) from exc
    # ``_verified_score`` intentionally returns its digest-stripped core.  The
    # official pair classifier re-verifies complete score documents, so retain
    # the original immutable digest after validating the core once here.
    score = {
        **verified_score,
        "score_sha256": str((record.score or {}).get("score_sha256") or ""),
    }
    if score.get("method", {}).get("structural_policy") != _SCORE_ARM[record.arm]:
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} score arm mismatch."
        )
    observed_route = str(
        score.get("method", {}).get(
            "exchange_route_bundle", SIAM_ROUTE_HISTORICAL_UNRECORDED
        )
    )
    accepted_routes = (
        _ACCEPTED_AVQDS_SCORE_ROUTES
        if record.arm == ARM_AVQDS
        else _ACCEPTED_EXCHANGE_SCORE_ROUTES[record.route_id]
    )
    if observed_route not in accepted_routes:
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} score route {observed_route!r} "
            f"is not on the {record.route_id!r} ladder route."
        )
    if score.get("method", {}).get("code_revision") != receipt.get(
        "source_revision"
    ):
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} score revision mismatch."
        )
    if score.get("source", {}).get("seed_sha256") != receipt.get("seed_sha256"):
        raise SIAMNb4LadderError(
            f"Completed action {record.action_id} score seed mismatch."
        )

    point_count = score.get("grid", {}).get("point_count")
    if record.segment == SEGMENT_SCREEN:
        if point_count != len(SIAM_NB4_SCREEN_SEGMENT.times):
            raise SIAMNb4LadderError("A SIAM screen score must cover 101 points.")
        if score.get("source", {}).get("run_sha256") != receipt.get(
            "trajectory_sha256"
        ):
            raise SIAMNb4LadderError(
                f"Completed action {record.action_id} score trajectory mismatch."
            )
    else:
        if point_count != 501:
            raise SIAMNb4LadderError(
                "A SIAM continuation score must describe the stitched 501-point horizon."
            )
        try:
            proof = validate_siam_nb4_stitch_completion_receipt(
                record.score_completion_proof or {},
                expected_trajectory_sha256=score.get("source", {}).get("run_sha256"),
                expected_seed_sha256=receipt.get("seed_sha256"),
                expected_arm=_RECEIPT_ARM[record.arm],
            )
        except (ValueError, SystemExit) as exc:
            raise SIAMNb4LadderError(
                f"Completed action {record.action_id} has an invalid stitch proof: {exc}"
            ) from exc
        if proof.get("schema") != SIAM_NB4_STITCH_COMPLETION_V1:
            raise SIAMNb4LadderError("SIAM continuation stitch-proof schema mismatch.")
        proof_sources = proof.get("source_segments")
        if not isinstance(proof_sources, list):
            raise SIAMNb4LadderError("SIAM continuation stitch proof lacks sources.")
        embedded = {
            str(source.get("segment")): source
            for source in proof_sources
            if isinstance(source, Mapping)
        }
        if set(embedded) != {SEGMENT_SCREEN, SEGMENT_CONTINUATION}:
            raise SIAMNb4LadderError(
                "SIAM continuation stitch proof must bind one screen and one continuation."
            )
        screen_receipt = embedded[SEGMENT_SCREEN].get("completion_receipt")
        continuation_receipt = embedded[SEGMENT_CONTINUATION].get(
            "completion_receipt"
        )
        if not isinstance(screen_receipt, Mapping) or not isinstance(
            continuation_receipt, Mapping
        ):
            raise SIAMNb4LadderError(
                "SIAM continuation stitch proof lacks embedded raw receipts."
            )
        if screen_receipt.get("action_id") != record.resume_from_action_id:
            raise SIAMNb4LadderError(
                "SIAM continuation stitch proof names the wrong screen action."
            )
        if continuation_receipt.get("action_id") != record.action_id:
            raise SIAMNb4LadderError(
                "SIAM continuation stitch proof names the wrong continuation action."
            )
        if dict(continuation_receipt) != dict(record.completion_receipt or {}):
            raise SIAMNb4LadderError(
                "SIAM continuation raw receipt differs from the stitch proof."
            )
        if any(
            source.get("source_revision")
            != score.get("method", {}).get("code_revision")
            for source in proof_sources
        ):
            raise SIAMNb4LadderError(
                "SIAM continuation score and stitch source revisions differ."
            )

    broad_safe, individual, _failures = _score_guard_state(score)
    return _VerifiedRecord(record, receipt, score, broad_safe, individual)


def _sort_key(record: SIAMNb4ActionRecord) -> tuple[int, int, int, str]:
    return (
        _ARM_ORDER[record.arm],
        0 if record.segment == SEGMENT_SCREEN else 1,
        record.ladder_index,
        record.action_id,
    )


def _pair_mismatch_justifies_tightening(
    *,
    arm: str,
    source_full: _VerifiedRecord,
    verified: Sequence[_VerifiedRecord],
) -> bool:
    """Return whether an earlier proof-valid unmatched pair selected ``arm``.

    The source continuation must be the prior rung for the arm being tightened.
    The independently tuned counterpart may legitimately be at any registered
    rung: for example, full APM can need several tighter screens before its T=5
    result selects a still-coarse AVQDS arm for the next transition.
    """

    if source_full.score is None:
        return False
    counterparts = sorted(
        (
            item
            for item in verified
            if item.record.arm != arm
            and item.record.segment == SEGMENT_CONTINUATION
            and item.record.state == STATE_COMPLETED
            and item.score is not None
        ),
        key=lambda item: item.record.ladder_index,
        reverse=True,
    )
    for counterpart in counterparts:
        assert counterpart.score is not None
        try:
            pair = classify_siam_matched_pair(source_full.score, counterpart.score)
        except (KeyError, TypeError, ValueError):
            continue
        matched = pair.get("matched_accuracy")
        if not isinstance(matched, Mapping) or matched.get(
            "cost_comparison_admissible"
        ) is True:
            continue
        scores = {
            source_full.record.arm: source_full.score,
            counterpart.record.arm: counterpart.score,
        }
        if set(scores) != set(ARMS):
            continue
        offender, _reason = _offending_arm(
            scores[ARM_AVQDS], scores[ARM_FULL_APM]
        )
        if offender == arm:
            return True
    return False


def _validate_history(
    records: Sequence[SIAMNb4ActionRecord | SIAMNb4SafeguardRecord],
    *,
    route_id: str = ROUTE_HISTORICAL,
) -> tuple[tuple[_VerifiedRecord, ...], tuple[SIAMNb4SafeguardRecord, ...]]:
    route_id = _canonical_route(route_id)
    action_ids: set[str] = set()
    identities: set[tuple[str, str, float]] = set()
    action_records: list[SIAMNb4ActionRecord] = []
    safeguards: list[SIAMNb4SafeguardRecord] = []
    for record in records:
        if isinstance(record, SIAMNb4SafeguardRecord):
            if record.action_id in action_ids:
                raise SIAMNb4LadderError(
                    f"SIAM Nb4 ladder history repeats action ID {record.action_id!r}."
                )
            action_ids.add(record.action_id)
            safeguards.append(record)
            continue
        if not isinstance(record, SIAMNb4ActionRecord):
            raise SIAMNb4LadderError(
                "SIAM Nb4 ladder history must contain action or safeguard records."
            )
        if record.action_id in action_ids:
            raise SIAMNb4LadderError(
                f"SIAM Nb4 ladder history repeats action ID {record.action_id!r}."
            )
        if record.scientific_identity in identities:
            raise SIAMNb4LadderError(
                "SIAM Nb4 ladder history repeats a scientific action identity."
            )
        if record.route_id != route_id:
            raise SIAMNb4LadderError(
                f"SIAM Nb4 action {record.action_id!r} belongs to route "
                f"{record.route_id!r}, not the planned route {route_id!r}."
            )
        action_ids.add(record.action_id)
        identities.add(record.scientific_identity)
        action_records.append(record)

    by_arm: dict[str, list[SIAMNb4ActionRecord]] = {arm: [] for arm in ARMS}
    for record in action_records:
        if record.segment == SEGMENT_SCREEN:
            by_arm[record.arm].append(record)
    for arm, screens in by_arm.items():
        ordered = sorted(screens, key=lambda record: record.ladder_index)
        first_index = ordered[0].ladder_index if ordered else 0
        allowed_start_index = 0
        if ordered and arm == ARM_FULL_APM:
            # Full APM starts at the proof-valid AVQDS scout cut.  Once that
            # offset is admitted, subsequent screens must be contiguous from
            # it rather than retroactively requiring skipped coarser rungs.
            allowed_start_index = first_index
        seen: set[int] = set()
        for record in ordered:
            index = record.ladder_index
            missing = set(range(allowed_start_index, index)).difference(seen)
            if not missing:
                seen.add(index)
                continue
            # The immutable legacy r00 alias may remain as canonical-index-two
            # evidence while the canonical coarser ladder is backfilled.  It
            # does not shift the AVQDS start index or cover an omitted rung.
            if record.uses_legacy_r00_identity and index == 2:
                seen.add(index)
                continue
            raise SIAMNb4LadderError(
                f"SIAM Nb4 {arm} screen history skips a registered rung."
            )

    verified = tuple(
        _verified_completed_record(record)
        for record in sorted(action_records, key=_sort_key)
    )
    index = {
        item.record.scientific_identity: item
        for item in verified
    }
    for item in verified:
        record = item.record
        if record.segment != SEGMENT_CONTINUATION:
            continue
        screen = index.get((record.arm, SEGMENT_SCREEN, record.activation_cut))
        if (
            screen is None
            or screen.record.state != STATE_COMPLETED
            or screen.score is None
            or screen.screen_broad_safe is not True
        ):
            raise SIAMNb4LadderError(
                "A SIAM continuation requires a completed, scored, broad-safe "
                "same-cut screen."
            )
        if item.score is not None:
            proof_sources = (item.record.score_completion_proof or {}).get(
                "source_segments"
            )
            if not isinstance(proof_sources, list):
                raise SIAMNb4LadderError(
                    "A scored SIAM continuation lacks stitch-proof sources."
                )
            embedded_screen = next(
                (
                    source.get("completion_receipt")
                    for source in proof_sources
                    if isinstance(source, Mapping)
                    and source.get("segment") == SEGMENT_SCREEN
                ),
                None,
            )
            if not isinstance(embedded_screen, Mapping) or dict(
                embedded_screen
            ) != dict(screen.record.completion_receipt or {}):
                raise SIAMNb4LadderError(
                    "SIAM continuation stitch proof does not bind the recorded "
                    "same-cut screen receipt."
                )

    completed_receipts = [
        item.receipt for item in verified if item.receipt is not None
    ]
    if completed_receipts:
        revisions = {str(receipt["source_revision"]) for receipt in completed_receipts}
        seeds = {str(receipt["seed_sha256"]) for receipt in completed_receipts}
        if len(revisions) != 1 or len(seeds) != 1:
            raise SIAMNb4LadderError(
                "SIAM Nb4 ladder history mixes source revisions or seed digests."
            )

    for arm in ARMS:
        screens = sorted(
            (
                item
                for item in verified
                if item.record.arm == arm
                and item.record.segment == SEGMENT_SCREEN
            ),
            key=lambda item: item.record.ladder_index,
        )
        for current in screens[1:]:
            previous = index.get(
                (
                    arm,
                    SEGMENT_SCREEN,
                    SIAM_NB4_ACTIVATION_CUT_LADDER[
                        current.record.ladder_index - 1
                    ],
                )
            )
            if previous is None and current.record.uses_legacy_r00_identity:
                # Explicit compatibility gap.  The planner will expose the
                # missing coarser AVQDS rungs before using this record as the
                # canonical scout, while retaining the original evidence.
                continue
            if (
                previous is None
                or previous.record.state != STATE_COMPLETED
                or previous.score is None
            ):
                raise SIAMNb4LadderError(
                    f"SIAM Nb4 {arm} rung advance lacks a completed scored predecessor."
                )
            if previous.screen_broad_safe is False:
                continue
            previous_full = index.get(
                (
                    arm,
                    SEGMENT_CONTINUATION,
                    previous.record.activation_cut,
                )
            )
            if previous_full is None or not _pair_mismatch_justifies_tightening(
                arm=arm,
                source_full=previous_full,
                verified=verified,
            ):
                raise SIAMNb4LadderError(
                    f"SIAM Nb4 {arm} tighter rung lacks an unmatched-pair decision."
                )

    apm_records = [item for item in verified if item.record.arm == ARM_FULL_APM]
    if apm_records and not any(
        item.record.arm == ARM_AVQDS
        and item.record.segment == SEGMENT_SCREEN
        and item.record.state == STATE_COMPLETED
        and item.screen_broad_safe is True
        for item in verified
    ):
        raise SIAMNb4LadderError(
            "Full APM cannot open before a valid broad-safe AVQDS screen."
        )

    for safeguard in safeguards:
        floor_screen = index.get(
            (safeguard.arm, SEGMENT_SCREEN, safeguard.activation_cut)
        )
        if (
            floor_screen is None
            or floor_screen.record.state != STATE_COMPLETED
            or floor_screen.score is None
        ):
            raise SIAMNb4LadderError(
                "A SIAM ladder-floor safeguard lacks its completed scored floor screen."
            )
        if floor_screen.screen_broad_safe is False:
            continue
        floor_full = index.get(
            (safeguard.arm, SEGMENT_CONTINUATION, safeguard.activation_cut)
        )
        if floor_full is None or not _pair_mismatch_justifies_tightening(
            arm=safeguard.arm,
            source_full=floor_full,
            verified=verified,
        ):
            raise SIAMNb4LadderError(
                "A SIAM ladder-floor safeguard lacks a floor miss or unmatched-pair "
                "decision."
            )
    return verified, tuple(sorted(safeguards, key=lambda item: item.action_id))


def _next_action(
    *,
    arm: str,
    segment: str,
    cut: float,
    reason: str,
    route_id: str = ROUTE_HISTORICAL,
    resume_from_action_id: str | None = None,
) -> SIAMNb4NextAction:
    registered_cut, index = _registered_cut(cut)
    return SIAMNb4NextAction(
        kind=KIND_RUN,
        action_id=siam_nb4_action_id(
            arm=arm,
            segment=segment,
            activation_cut=registered_cut,
            route_id=route_id,
        ),
        arm=arm,
        segment=segment,
        activation_cut=registered_cut,
        ladder_index=index,
        reason=reason,
        route_id=_canonical_route(route_id),
        resume_from_action_id=resume_from_action_id,
    )


def _safeguard_action(
    *, arm: str, cut: float, reason: str, route_id: str = ROUTE_HISTORICAL
) -> SIAMNb4NextAction:
    registered_cut, index = _registered_cut(cut)
    return SIAMNb4NextAction(
        kind=KIND_RECORD_SAFEGUARD,
        action_id=_safeguard_action_id(arm),
        arm=arm,
        segment=SEGMENT_SCREEN,
        activation_cut=registered_cut,
        ladder_index=index,
        reason=reason,
        route_id=_canonical_route(route_id),
    )


def _safeguard_action_id(arm: str) -> str:
    return f"siam-nb4-{_canonical_arm(arm)}-ladder-floor-safeguard"


def _record_lookup(
    verified: Sequence[_VerifiedRecord],
) -> dict[tuple[str, str, float], _VerifiedRecord]:
    return {item.record.scientific_identity: item for item in verified}


def _emit_if_uncovered(
    action: SIAMNb4NextAction,
    *,
    lookup: Mapping[tuple[str, str, float], _VerifiedRecord],
    actions: list[SIAMNb4NextAction],
    dispositions: _DispositionBook,
) -> bool:
    """Emit once, or explain why an existing action blocks the duplicate."""

    if action.kind == KIND_RECORD_SAFEGUARD:
        actions.append(action)
        dispositions.set(
            action.action_id,
            "safeguard",
            action.reason,
            arm=action.arm,
            segment=action.segment,
            activation_cut=action.activation_cut,
        )
        return True
    existing = lookup.get((action.arm, action.segment, action.activation_cut))
    if existing is None:
        actions.append(action)
        dispositions.set(
            action.action_id,
            "scheduled",
            action.reason,
            arm=action.arm,
            segment=action.segment,
            activation_cut=action.activation_cut,
        )
        return True
    record = existing.record
    if record.state in {STATE_SUBMITTED, STATE_IN_FLIGHT}:
        dispositions.set(
            record.action_id,
            "duplicate_excluded_awaiting_completion",
            "The exact action is already submitted or in flight.",
            arm=record.arm,
            segment=record.segment,
            activation_cut=record.activation_cut,
        )
    elif record.state == STATE_FAILED:
        dispositions.set(
            record.action_id,
            "failed_action_blocks_progress",
            "The exact failed action cannot be duplicated automatically; repair or "
            "explicitly transition that action before replaying the planner.",
            arm=record.arm,
            segment=record.segment,
            activation_cut=record.activation_cut,
        )
    else:
        dispositions.set(
            record.action_id,
            "duplicate_excluded_completed",
            "The exact action is already completed and receipt-covered.",
            arm=record.arm,
            segment=record.segment,
            activation_cut=record.activation_cut,
        )
    return False


def _screen_failure_names(score: Mapping[str, Any]) -> tuple[str, ...]:
    return _score_guard_state(score)[2]


def _lane_state(
    arm: str,
    *,
    verified: Sequence[_VerifiedRecord],
    lookup: Mapping[tuple[str, str, float], _VerifiedRecord],
    actions: list[SIAMNb4NextAction],
    dispositions: _DispositionBook,
    initial_index: int = 0,
    route_id: str = ROUTE_HISTORICAL,
) -> _LaneState:
    if initial_index < 0 or initial_index >= len(SIAM_NB4_ACTIVATION_CUT_LADDER):
        raise SIAMNb4LadderError("SIAM lane initial rung is outside the ladder.")
    screens = {
        item.record.ladder_index: item
        for item in verified
        if item.record.arm == arm and item.record.segment == SEGMENT_SCREEN
    }
    legacy_r00_present = any(
        item.record.uses_legacy_r00_identity for item in screens.values()
    )
    index = initial_index
    while index < len(SIAM_NB4_ACTIVATION_CUT_LADDER):
        cut = SIAM_NB4_ACTIVATION_CUT_LADDER[index]
        current = screens.get(index)
        if current is None:
            reason = (
                "Run the comparator-first AVQDS screen at the canonical coarsest "
                "rung."
                if arm == ARM_AVQDS and index == 0
                else (
                    "Backfill this coarser canonical AVQDS rung before relying on "
                    "the immutable legacy r00/cut1e-3 result."
                    if arm == ARM_AVQDS and index < 2 and legacy_r00_present
                    else (
                        "Open full APM at the AVQDS scout cut; the structural arm "
                        "then owns its ladder."
                        if arm == ARM_FULL_APM and index == initial_index
                        else (
                            f"Tighten {arm} by one exact rung after its measured "
                            "screen miss; no threshold monotonicity is inferred."
                        )
                    )
                )
            )
            _emit_if_uncovered(
                _next_action(
                    arm=arm,
                    segment=SEGMENT_SCREEN,
                    cut=cut,
                    reason=reason,
                    route_id=route_id,
                ),
                lookup=lookup,
                actions=actions,
                dispositions=dispositions,
            )
            return _LaneState(arm, None, None, False)

        record = current.record
        if record.state in {STATE_SUBMITTED, STATE_IN_FLIGHT}:
            dispositions.set(
                record.action_id,
                "awaiting_completion",
                "The current exact screen rung is already submitted or in flight.",
                arm=arm,
                segment=record.segment,
                activation_cut=cut,
            )
            return _LaneState(arm, current, None, True)
        if record.state == STATE_FAILED:
            dispositions.set(
                record.action_id,
                "failed_action_blocks_progress",
                f"Screen failed: {record.failure_reason}",
                arm=arm,
                segment=record.segment,
                activation_cut=cut,
            )
            return _LaneState(arm, current, None, True)
        if current.score is None:
            dispositions.set(
                record.action_id,
                "awaiting_offline_score",
                "The raw screen completed, but exact-blind execution must be "
                "followed by offline exact scoring before the ladder advances.",
                arm=arm,
                segment=record.segment,
                activation_cut=cut,
            )
            return _LaneState(arm, current, None, True)

        if current.screen_broad_safe is not True:
            failed = ", ".join(_screen_failure_names(current.score))
            dispositions.set(
                record.action_id,
                "screen_requires_tighter_rung",
                f"The scored screen failed {failed}; preserve its numbers and "
                "advance exactly one registered rung.",
                arm=arm,
                segment=record.segment,
                activation_cut=cut,
            )
            index += 1
            if index >= len(SIAM_NB4_ACTIVATION_CUT_LADDER):
                _emit_if_uncovered(
                    _safeguard_action(
                        arm=arm,
                        cut=cut,
                        route_id=route_id,
                        reason=(
                            f"{arm} remains outside the broad/fidelity/energy "
                            "screen guards at the predeclared floor; retain all "
                            "numerical candidates and stop this lane."
                        ),
                    ),
                    lookup=lookup,
                    actions=actions,
                    dispositions=dispositions,
                )
                return _LaneState(arm, current, None, False)
            continue

        dispositions.set(
            record.action_id,
            "valid_screen",
            (
                "The screen passes the individual target and all safety guards."
                if current.screen_individual_target
                else "The screen passes the broad response, fidelity, and energy guards."
            ),
            arm=arm,
            segment=record.segment,
            activation_cut=cut,
        )
        continuation = lookup.get((arm, SEGMENT_CONTINUATION, cut))
        if continuation is None:
            _emit_if_uncovered(
                _next_action(
                    arm=arm,
                    segment=SEGMENT_CONTINUATION,
                    cut=cut,
                    reason=(
                        f"Resume the valid {arm} screen at the same method-owned "
                        "cut through T=5."
                    ),
                    route_id=route_id,
                    resume_from_action_id=record.action_id,
                ),
                lookup=lookup,
                actions=actions,
                dispositions=dispositions,
            )
            return _LaneState(arm, current, None, False)
        continuation_record = continuation.record
        if continuation_record.state in {STATE_SUBMITTED, STATE_IN_FLIGHT}:
            dispositions.set(
                continuation_record.action_id,
                "awaiting_completion",
                "The same-cut T=5 continuation is already submitted or in flight.",
                arm=arm,
                segment=SEGMENT_CONTINUATION,
                activation_cut=cut,
            )
            return _LaneState(arm, current, None, True)
        if continuation_record.state == STATE_FAILED:
            dispositions.set(
                continuation_record.action_id,
                "failed_action_blocks_progress",
                f"Continuation failed: {continuation_record.failure_reason}",
                arm=arm,
                segment=SEGMENT_CONTINUATION,
                activation_cut=cut,
            )
            return _LaneState(arm, current, None, True)
        if continuation.score is None:
            dispositions.set(
                continuation_record.action_id,
                "awaiting_offline_score",
                "The continuation completed, but its stitched T=5 trajectory still "
                "requires proof-valid offline scoring.",
                arm=arm,
                segment=SEGMENT_CONTINUATION,
                activation_cut=cut,
            )
            return _LaneState(arm, current, None, True)
        dispositions.set(
            continuation_record.action_id,
            "full_horizon_candidate",
            "The proof-valid numerical T=5 result is retained for pair matching.",
            arm=arm,
            segment=SEGMENT_CONTINUATION,
            activation_cut=cut,
        )

        next_index = index + 1
        next_screen = screens.get(next_index)
        if next_screen is not None and _pair_mismatch_justifies_tightening(
            arm=arm,
            source_full=continuation,
            verified=verified,
        ):
            index = next_index
            continue
        return _LaneState(arm, current, continuation, False)

    raise SIAMNb4LadderError("SIAM lane traversal escaped the registered ladder.")


def _next_tighter_after_pair(
    lane: _LaneState,
    *,
    lookup: Mapping[tuple[str, str, float], _VerifiedRecord],
    actions: list[SIAMNb4NextAction],
    dispositions: _DispositionBook,
    reason: str,
    route_id: str = ROUTE_HISTORICAL,
) -> None:
    if lane.latest_screen is None:
        raise SIAMNb4LadderError("Pair escalation lost its selected screen rung.")
    current = lane.latest_screen.record
    next_index = current.ladder_index + 1
    if next_index >= len(SIAM_NB4_ACTIVATION_CUT_LADDER):
        _emit_if_uncovered(
            _safeguard_action(
                arm=lane.arm,
                cut=current.activation_cut,
                route_id=route_id,
                reason=(
                    f"{reason} The offending arm is already at the predeclared "
                    "activation-cut floor; retain both numerical candidates and "
                    "classify the cost comparison as accuracy-confounded."
                ),
            ),
            lookup=lookup,
            actions=actions,
            dispositions=dispositions,
        )
        return
    _emit_if_uncovered(
        _next_action(
            arm=lane.arm,
            segment=SEGMENT_SCREEN,
            cut=SIAM_NB4_ACTIVATION_CUT_LADDER[next_index],
            route_id=route_id,
            reason=(
                f"{reason} Advance only {lane.arm} by the next exact screen rung; "
                "the other arm's numerical candidate is preserved."
            ),
        ),
        lookup=lookup,
        actions=actions,
        dispositions=dispositions,
    )


def _offending_arm(
    avqds_score: Mapping[str, Any], apm_score: Mapping[str, Any]
) -> tuple[str, str]:
    failures = {
        ARM_AVQDS: _screen_failure_names(avqds_score),
        ARM_FULL_APM: _screen_failure_names(apm_score),
    }
    failing = [arm for arm in ARMS if failures[arm]]
    if len(failing) == 1:
        arm = failing[0]
        return arm, (
            f"The T=5 {arm} result fails " + ", ".join(failures[arm]) + "."
        )
    burdens = {
        ARM_AVQDS: float(avqds_score["summary"]["aggregate_accuracy_burden"]),
        ARM_FULL_APM: float(apm_score["summary"]["aggregate_accuracy_burden"]),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in burdens.values()):
        raise SIAMNb4LadderError("SIAM pair burdens must be finite and nonnegative.")
    if burdens[ARM_AVQDS] > burdens[ARM_FULL_APM]:
        arm = ARM_AVQDS
    else:
        # The structural arm is the deterministic exact-tie choice.  This is
        # an ordering rule only, never a claim about threshold response.
        arm = ARM_FULL_APM
    suffix = (
        "Both arms fail at least one safety guard; select the higher aggregate "
        "burden (full APM on an exact tie)."
        if len(failing) == 2
        else "Both arms are safety-safe but accuracy matching fails; select the "
        "higher aggregate burden (full APM on an exact tie)."
    )
    return arm, suffix


def plan_siam_nb4_ladder(
    records: Sequence[SIAMNb4ActionRecord | SIAMNb4SafeguardRecord],
    *,
    route_id: str = ROUTE_HISTORICAL,
) -> SIAMNb4LadderPlan:
    """Return the duplicate-free exact action frontier for the SIAM pilot.

    The caller owns persistence and execution.  Replaying this function with
    unchanged records returns the same plan; it never submits, writes, scores,
    or opens a trajectory.

    ``route_id`` names the exchange route this ladder belongs to.  A ladder on a
    registered route bundle is a *new* ladder: it shares no action identity with
    the historical one and, with no records, opens at r00 in the AVQDS lane
    exactly like a fresh historical ladder.
    """

    route_id = _canonical_route(route_id)
    verified, recorded_safeguards = _validate_history(
        tuple(records), route_id=route_id
    )
    lookup = _record_lookup(verified)
    dispositions = _DispositionBook()
    actions: list[SIAMNb4NextAction] = []

    for item in verified:
        record = item.record
        if record.uses_legacy_r00_identity:
            dispositions.set(
                record.action_id,
                "legacy_r00_mapped_to_canonical_index_2",
                "Immutable r00/cut1e-3 evidence is retained under its exact action "
                "ID and source epoch, but occupies canonical ladder index 2.  It "
                "does not cover the coarser canonical rungs.",
                arm=record.arm,
                segment=record.segment,
                activation_cut=record.activation_cut,
            )
        if record.state in {STATE_SUBMITTED, STATE_IN_FLIGHT}:
            dispositions.set(
                record.action_id,
                "duplicate_excluded_awaiting_completion",
                "The exact action is already submitted or in flight.",
                arm=record.arm,
                segment=record.segment,
                activation_cut=record.activation_cut,
            )
        elif record.state == STATE_FAILED:
            dispositions.set(
                record.action_id,
                "failed_action_blocks_progress",
                f"Action failed: {record.failure_reason}",
                arm=record.arm,
                segment=record.segment,
                activation_cut=record.activation_cut,
            )
        elif item.score is None:
            dispositions.set(
                record.action_id,
                "awaiting_offline_score",
                "Positive raw completion is recorded; exact scoring is still pending.",
                arm=record.arm,
                segment=record.segment,
                activation_cut=record.activation_cut,
            )

    avqds = _lane_state(
        ARM_AVQDS,
        verified=verified,
        lookup=lookup,
        actions=actions,
        dispositions=dispositions,
        route_id=route_id,
    )
    comparator_ready = bool(
        avqds.latest_screen is not None
        and avqds.latest_screen.record.state == STATE_COMPLETED
        and avqds.latest_screen.score is not None
        and avqds.latest_screen.screen_broad_safe is True
    )
    apm: _LaneState | None = None
    if comparator_ready:
        apm = _lane_state(
            ARM_FULL_APM,
            verified=verified,
            lookup=lookup,
            actions=actions,
            dispositions=dispositions,
            initial_index=avqds.latest_screen.record.ladder_index,
            route_id=route_id,
        )
    else:
        dispositions.set(
            "full_apm_lane",
            "comparator_first_blocked",
            "Full APM remains closed until AVQDS has one proof-valid broad-safe screen.",
            arm=ARM_FULL_APM,
        )

    candidate_ids = tuple(
        item.record.action_id
        for item in verified
        if item.record.segment == SEGMENT_CONTINUATION and item.score is not None
    )
    selected: dict[str, str] = {}
    admissible = False
    complete = False
    if avqds.current_full is not None and apm is not None and apm.current_full is not None:
        assert avqds.current_full.score is not None
        assert apm.current_full.score is not None
        selected = {
            ARM_AVQDS: avqds.current_full.record.action_id,
            ARM_FULL_APM: apm.current_full.record.action_id,
        }
        try:
            pair = classify_siam_matched_pair(
                avqds.current_full.score, apm.current_full.score
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SIAMNb4LadderError(
                f"The selected SIAM T=5 scores cannot form a matched pair: {exc}"
            ) from exc
        matched = pair.get("matched_accuracy")
        if not isinstance(matched, Mapping):
            raise SIAMNb4LadderError("SIAM pair classifier returned no matched state.")
        admissible = matched.get("cost_comparison_admissible") is True
        if admissible:
            complete = True
            actions.clear()
            dispositions.set(
                "siam_nb4_pair",
                "matched_pair_complete",
                "Both proof-valid T=5 candidates satisfy the canonical matched-pair "
                "predicate; the comparator ladder stops.",
            )
        else:
            # A pair decision owns the frontier.  Do not accidentally retain an
            # unrelated speculative action assembled earlier in this replay.
            actions.clear()
            offender, reason = _offending_arm(
                avqds.current_full.score, apm.current_full.score
            )
            dispositions.set(
                "siam_nb4_pair",
                "accuracy_confounded",
                reason + " Both numerical candidates remain visible.",
                arm=offender,
            )
            lane = avqds if offender == ARM_AVQDS else apm
            assert lane is not None
            _next_tighter_after_pair(
                lane,
                lookup=lookup,
                actions=actions,
                dispositions=dispositions,
                reason=reason,
                route_id=route_id,
            )

    emitted_safeguards = [
        action for action in actions if action.kind == KIND_RECORD_SAFEGUARD
    ]
    if emitted_safeguards:
        # A registered floor is terminal for this two-arm cost claim.  Do not
        # continue speculative work in the other lane after the comparison has
        # become irreparably accuracy-confounded.
        actions[:] = emitted_safeguards
    if recorded_safeguards:
        if complete:
            raise SIAMNb4LadderError(
                "SIAM history contains a matched pair after a recorded terminal "
                "ladder-floor safeguard."
            )
        actions.clear()
        for safeguard in recorded_safeguards:
            dispositions.set(
                safeguard.action_id,
                "recorded_safeguard",
                safeguard.reason,
                arm=safeguard.arm,
                segment=SEGMENT_SCREEN,
                activation_cut=safeguard.activation_cut,
            )

    # Stable ordering lets an executor compare a replay byte-for-byte.
    actions.sort(
        key=lambda action: (
            0 if action.kind == KIND_RUN else 1,
            _ARM_ORDER[action.arm],
            0 if action.segment == SEGMENT_SCREEN else 1,
            action.ladder_index,
            action.action_id,
        )
    )
    blocked = bool(
        any(
            row.disposition
            in {"failed_action_blocks_progress", "safeguard", "recorded_safeguard"}
            for row in dispositions.values()
        )
        and not complete
        and not any(action.kind == KIND_RUN for action in actions)
    )
    if complete:
        phase = "complete"
    elif recorded_safeguards or any(
        action.kind == KIND_RECORD_SAFEGUARD for action in actions
    ):
        phase = "safeguard"
    elif blocked:
        phase = "blocked"
    elif comparator_ready and any(
        item.record.segment == SEGMENT_CONTINUATION for item in verified
    ):
        phase = "full_horizon"
    elif comparator_ready:
        phase = "two_arm_screening"
    else:
        phase = "avqds_screening"

    return SIAMNb4LadderPlan(
        next_actions=tuple(actions),
        dispositions=dispositions.values(),
        candidate_action_ids=candidate_ids,
        selected_full_horizon_action_ids=selected,
        recorded_safeguard_ids=tuple(
            safeguard.action_id for safeguard in recorded_safeguards
        ),
        cost_comparison_admissible=admissible,
        complete=complete,
        blocked=blocked,
        phase=phase,
        route_id=route_id,
    )


__all__ = [
    "ACTION_STATES",
    "ARM_AVQDS",
    "ARM_FULL_APM",
    "KIND_RECORD_SAFEGUARD",
    "KIND_RUN",
    "SEGMENT_CONTINUATION",
    "SEGMENT_SCREEN",
    "ROUTE_HISTORICAL",
    "ROUTE_Q_COMPILED",
    "ROUTE_TOKENS",
    "SIAM_NB4_LADDER_PLAN_V1",
    "SIAMNb4ActionRecord",
    "SIAMNb4Disposition",
    "SIAMNb4LadderError",
    "SIAMNb4LadderPlan",
    "SIAMNb4NextAction",
    "SIAMNb4SafeguardRecord",
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_IN_FLIGHT",
    "STATE_SUBMITTED",
    "plan_siam_nb4_ladder",
    "siam_nb4_action_id",
]
