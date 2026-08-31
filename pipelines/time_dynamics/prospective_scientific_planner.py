"""Pure post-bootstrap planner for prospective HH and Hubbard dynamics cells.

The bootstrap planner owns seed and exact-reference construction.  This module
starts only after both artifacts have positive, digest-bound receipts.  It
then owns the within-cell scientific state machine:

* AVQDS scouts the registered prefix ladder until the broad guard is reached;
* full APM starts at that exact cut and calibrates independently;
* the first admissible APM result is frozen as the accuracy anchor;
* AVQDS moves by measured adjacent rungs until aggregate and primitive burdens
  match the anchor;
* the matched prefix pair is continued and rescored at the full horizon; and
* a full-horizon miss reopens only the calibration that the measured result
  implicates.

The planner is deliberately side-effect free.  It does not read the workspace,
submit jobs, score trajectories, or write safeguards.  Its only interface is
``plan_prospective_cell``: immutable verified evidence in, deterministic action
contracts out.  Every emitted action repeats the complete scientific binding,
so an executor cannot silently substitute a cell, seed, reference, source
epoch, policy, or Qiskit compilation step.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CellKey,
    METHOD_APM,
    METHOD_AVQDS,
    POLICY_RESPONSE_NORMALIZED_MATCHED,
)


PROSPECTIVE_SCIENTIFIC_PLAN_V1 = "paper_ii_prospective_scientific_plan_v1"
PROSPECTIVE_SCIENTIFIC_BINDINGS_V1 = (
    "paper_ii_prospective_scientific_bindings_v1"
)

LANE_HH = "paper_ii_hh_product_v1"
LANE_HUBBARD = "paper_ii_hubbard_l4_scale_v1"
OWNED_LANES = frozenset({LANE_HH, LANE_HUBBARD})

ARM_AVQDS = METHOD_AVQDS
ARM_APM = METHOD_APM
ARMS = (ARM_AVQDS, ARM_APM)

HORIZON_PREFIX = "t5"
HORIZON_FULL = "t20"
HORIZONS = (HORIZON_PREFIX, HORIZON_FULL)

STATE_SUBMITTED = "submitted"
STATE_IN_FLIGHT = "in_flight"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
RUN_STATES = frozenset(
    {STATE_SUBMITTED, STATE_IN_FLIGHT, STATE_COMPLETED, STATE_FAILED}
)

SCORE_NOT_STARTED = "not_started"
SCORE_SUBMITTED = "submitted"
SCORE_IN_FLIGHT = "in_flight"
SCORE_COMPLETED = "completed"
SCORE_FAILED = "failed"
SCORE_STATES = frozenset(
    {
        SCORE_NOT_STARTED,
        SCORE_SUBMITTED,
        SCORE_IN_FLIGHT,
        SCORE_COMPLETED,
        SCORE_FAILED,
    }
)

FAILURE_MEMORY = "verified_memory_exhaustion"
FAILURE_OTHER = "other"

KIND_RUN = "run"
KIND_SCORE = "score"
KIND_RECORD_SAFEGUARD = "record_safeguard"

SAFEGUARD_LADDER_UNRESOLVED = "ladder_unresolved"
SAFEGUARD_OPERATIONALLY_INFEASIBLE = "operationally_infeasible"

INITIAL_MEMORY_GIB = 8
MEMORY_TIER_GIB = 4
MAX_MEMORY_GIB = 32


class ProspectivePlannerError(ValueError):
    """The supplied prospective evidence violates the frozen experiment."""


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _deep_frozen_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_frozen_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_frozen_json(item) for item in value)
    return value


def _deep_plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_plain_json(item) for item in value]
    return value


def _require_sha256(value: str, *, label: str) -> str:
    token = str(value).strip().lower()
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise ProspectivePlannerError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _require_nonempty(value: str, *, label: str) -> str:
    token = str(value).strip()
    if not token:
        raise ProspectivePlannerError(f"{label} must be non-empty")
    return token


def policy_sha256(policy: AccuracyPolicy) -> str:
    """Return the canonical identity of the complete accuracy policy."""

    if policy.mode != POLICY_RESPONSE_NORMALIZED_MATCHED:
        raise ProspectivePlannerError(
            "prospective HH/Hubbard planning requires the response-normalized "
            "matched-pair policy"
        )
    return _canonical_sha(policy.to_json_dict())


def _activation_ladder() -> tuple[float, ...]:
    values: list[float] = [1.0e-2]
    for exponent in range(-3, -13, -1):
        values.extend((3.0 * 10.0**exponent, 1.0 * 10.0**exponent))
    return tuple(float(f"{value:.12g}") for value in values)


HH_HUBBARD_ACTIVATION_CUT_LADDER = _activation_ladder()


def _registered_cut(value: float) -> tuple[float, int]:
    cut = float(value)
    if not math.isfinite(cut) or cut <= 0.0:
        raise ProspectivePlannerError("activation cut must be finite and positive")
    for index, rung in enumerate(HH_HUBBARD_ACTIVATION_CUT_LADDER):
        if math.isclose(cut, rung, rel_tol=5.0e-12, abs_tol=0.0):
            return rung, index
    raise ProspectivePlannerError(
        f"activation cut {cut!r} is not on the frozen 1/3-decade ladder"
    )


def _cut_token(value: float) -> str:
    mantissa, exponent = f"{float(value):.0e}".split("e")
    sign = "m" if exponent.startswith("-") else "p"
    return f"{mantissa}e{sign}{exponent.lstrip('+-').zfill(2)}"


@dataclass(frozen=True)
class ProspectiveScientificBindings:
    """Immutable positive bindings shared by every action in one cell."""

    lane_id: str
    cell_id: str
    policy_regime: str
    drive: str
    controller: str
    cell_identity_sha256: str
    seed_id: str
    seed_identity_sha256: str
    seed_artifact_sha256: str
    seed_receipt_sha256: str
    reference_id: str
    reference_identity_sha256: str
    reference_manifest_sha256: str
    reference_data_sha256: str
    reference_receipt_sha256: str
    source_epoch_identity_sha256: str
    source_revision: str
    source_tree_sha256: str
    matrix_id: str
    matrix_raw_sha256: str
    matrix_payload_sha256: str
    matrix_expanded_semantic_sha256: str
    matrix_accuracy_policy_id: str
    matrix_accuracy_policy_sha256: str
    accuracy_policy_mode: str
    accuracy_policy: Mapping[str, Any]
    accuracy_policy_sha256: str
    qiskit_compilation: bool = False

    def __post_init__(self) -> None:
        if self.lane_id not in OWNED_LANES:
            raise ProspectivePlannerError(
                "the prospective scientific planner owns only HH and Hubbard L4"
            )
        for name in (
            "cell_id",
            "policy_regime",
            "drive",
            "controller",
            "seed_id",
            "reference_id",
            "source_revision",
            "matrix_id",
            "matrix_accuracy_policy_id",
            "accuracy_policy_mode",
        ):
            object.__setattr__(
                self, name, _require_nonempty(getattr(self, name), label=name)
            )
        for name in (
            "cell_identity_sha256",
            "seed_identity_sha256",
            "seed_artifact_sha256",
            "seed_receipt_sha256",
            "reference_manifest_sha256",
            "reference_data_sha256",
            "reference_receipt_sha256",
            "reference_identity_sha256",
            "source_epoch_identity_sha256",
            "source_tree_sha256",
            "matrix_raw_sha256",
            "matrix_payload_sha256",
            "matrix_expanded_semantic_sha256",
            "matrix_accuracy_policy_sha256",
            "accuracy_policy_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), label=name)
            )
        if self.controller != "state_motion_5e-5":
            raise ProspectivePlannerError(
                "prospective HH/Hubbard cells require state_motion_5e-5"
            )
        if bool(self.qiskit_compilation):
            raise ProspectivePlannerError(
                "Qiskit compilation is excluded from the prospective campaign"
            )
        if self.cell_identity_sha256 != self.reference_identity_sha256:
            raise ProspectivePlannerError(
                "cell identity must be the canonical exact-reference identity"
            )
        if self.matrix_accuracy_policy_id != "response_normalized_matched_pair_v2":
            raise ProspectivePlannerError(
                "prospective bindings require the frozen matrix policy identifier"
            )
        if not isinstance(self.accuracy_policy, Mapping):
            raise ProspectivePlannerError(
                "prospective bindings require the complete accuracy policy"
            )
        policy_payload = _deep_plain_json(self.accuracy_policy)
        try:
            parsed_policy = AccuracyPolicy.from_json_dict(policy_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProspectivePlannerError(
                "prospective bindings contain an invalid accuracy policy"
            ) from exc
        if parsed_policy.mode != POLICY_RESPONSE_NORMALIZED_MATCHED:
            raise ProspectivePlannerError(
                "prospective bindings require response-normalized matching"
            )
        if self.accuracy_policy_mode != parsed_policy.mode:
            raise ProspectivePlannerError(
                "accuracy policy mode diverges from the complete policy"
            )
        if _canonical_sha(policy_payload) != self.accuracy_policy_sha256:
            raise ProspectivePlannerError(
                "complete accuracy policy digest mismatch"
            )
        object.__setattr__(
            self, "accuracy_policy", _deep_frozen_json(policy_payload)
        )

    @property
    def digest(self) -> str:
        return _canonical_sha(self.to_json_dict(include_digest=False))

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROSPECTIVE_SCIENTIFIC_BINDINGS_V1,
            "lane_id": self.lane_id,
            "cell_id": self.cell_id,
            "policy_regime": self.policy_regime,
            "drive": self.drive,
            "controller": self.controller,
            "cell_identity_sha256": self.cell_identity_sha256,
            "seed_id": self.seed_id,
            "seed_identity_sha256": self.seed_identity_sha256,
            "seed_artifact_sha256": self.seed_artifact_sha256,
            "seed_receipt_sha256": self.seed_receipt_sha256,
            "reference_id": self.reference_id,
            "reference_identity_sha256": self.reference_identity_sha256,
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "reference_data_sha256": self.reference_data_sha256,
            "reference_receipt_sha256": self.reference_receipt_sha256,
            "source_epoch_identity_sha256": self.source_epoch_identity_sha256,
            "source_revision": self.source_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "matrix_id": self.matrix_id,
            "matrix_raw_sha256": self.matrix_raw_sha256,
            "matrix_payload_sha256": self.matrix_payload_sha256,
            "matrix_expanded_semantic_sha256": (
                self.matrix_expanded_semantic_sha256
            ),
            "matrix_accuracy_policy_id": self.matrix_accuracy_policy_id,
            "matrix_accuracy_policy_sha256": (
                self.matrix_accuracy_policy_sha256
            ),
            "accuracy_policy_mode": self.accuracy_policy_mode,
            "accuracy_policy": _deep_plain_json(self.accuracy_policy),
            "accuracy_policy_sha256": self.accuracy_policy_sha256,
            "qiskit_compilation": False,
        }
        if include_digest:
            payload["bindings_sha256"] = self.digest
        return payload

    def validate_against_matrix(self, matrix: Any | None = None) -> None:
        """Authenticate this binding against the frozen matrix and identities."""

        from pipelines.time_dynamics.parallel_benchmark_matrix import (
            ParallelBenchmarkMatrix,
        )
        from pipelines.time_dynamics.prospective_campaign_bootstrap import (
            build_identity_registry,
        )

        campaign = matrix or ParallelBenchmarkMatrix.read()
        try:
            campaign.validate_frozen()
        except (KeyError, TypeError, ValueError) as exc:
            raise ProspectivePlannerError(
                "prospective bindings name an invalid frozen matrix"
            ) from exc
        expected_matrix = (
            campaign.matrix_id,
            campaign.raw_sha256,
            campaign.payload_sha256,
            campaign.expanded_semantic_sha256,
            str(campaign.payload["accuracy_policy"]["policy_id"]),
            campaign.accuracy_policy_spec_sha256,
        )
        observed_matrix = (
            self.matrix_id,
            self.matrix_raw_sha256,
            self.matrix_payload_sha256,
            self.matrix_expanded_semantic_sha256,
            self.matrix_accuracy_policy_id,
            self.matrix_accuracy_policy_sha256,
        )
        if observed_matrix != expected_matrix:
            raise ProspectivePlannerError(
                "prospective bindings diverge from the frozen matrix digests"
            )

        cells = tuple(cell for cell in campaign.cells if cell.cell_id == self.cell_id)
        if (
            len(cells) != 1
            or cells[0].lane_id != self.lane_id
            or cells[0].drive != self.drive
            or cells[0].controller != self.controller
        ):
            raise ProspectivePlannerError(
                "prospective bindings do not name one canonical matrix cell"
            )
        identities = build_identity_registry(campaign)
        references = tuple(
            row for row in identities.references if row.cell_id == self.cell_id
        )
        if (
            len(references) != 1
            or self.reference_id != references[0].reference_id
            or self.reference_identity_sha256 != references[0].identity_sha256
            or self.cell_identity_sha256 != references[0].identity_sha256
        ):
            raise ProspectivePlannerError(
                "prospective bindings diverge from the canonical matrix reference"
            )
        seeds = tuple(
            row for row in identities.seeds if row.seed_id == references[0].seed_id
        )
        if (
            len(seeds) != 1
            or self.seed_id != seeds[0].seed_id
            or self.seed_identity_sha256 != seeds[0].identity_sha256
        ):
            raise ProspectivePlannerError(
                "prospective bindings diverge from the canonical matrix seed"
            )

        try:
            parsed_policy = AccuracyPolicy.from_json_dict(
                _deep_plain_json(self.accuracy_policy)
            )
            canonical_policy = campaign.canonical_accuracy_policy(
                self.lane_id,
                parsed_policy.response_scales,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProspectivePlannerError(
                "prospective bindings contain no matrix-admissible policy"
            ) from exc
        if canonical_policy.to_json_dict() != _deep_plain_json(
            self.accuracy_policy
        ):
            raise ProspectivePlannerError(
                "prospective bindings weaken or alter the frozen matrix policy"
            )

    @classmethod
    def from_json_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ProspectiveScientificBindings":
        """Reconstruct and authenticate a persisted binding contract."""

        if payload.get("schema") != PROSPECTIVE_SCIENTIFIC_BINDINGS_V1:
            raise ProspectivePlannerError(
                "unsupported prospective scientific bindings schema"
            )
        accuracy_policy = payload.get("accuracy_policy")
        if not isinstance(accuracy_policy, Mapping):
            raise ProspectivePlannerError(
                "prospective scientific bindings omit the complete accuracy policy"
            )
        instance = cls(
            lane_id=str(payload.get("lane_id") or ""),
            cell_id=str(payload.get("cell_id") or ""),
            policy_regime=str(payload.get("policy_regime") or ""),
            drive=str(payload.get("drive") or ""),
            controller=str(payload.get("controller") or ""),
            cell_identity_sha256=str(
                payload.get("cell_identity_sha256") or ""
            ),
            seed_id=str(payload.get("seed_id") or ""),
            seed_identity_sha256=str(
                payload.get("seed_identity_sha256") or ""
            ),
            seed_artifact_sha256=str(
                payload.get("seed_artifact_sha256") or ""
            ),
            seed_receipt_sha256=str(
                payload.get("seed_receipt_sha256") or ""
            ),
            reference_id=str(payload.get("reference_id") or ""),
            reference_identity_sha256=str(
                payload.get("reference_identity_sha256") or ""
            ),
            reference_manifest_sha256=str(
                payload.get("reference_manifest_sha256") or ""
            ),
            reference_data_sha256=str(
                payload.get("reference_data_sha256") or ""
            ),
            reference_receipt_sha256=str(
                payload.get("reference_receipt_sha256") or ""
            ),
            source_epoch_identity_sha256=str(
                payload.get("source_epoch_identity_sha256") or ""
            ),
            source_revision=str(payload.get("source_revision") or ""),
            source_tree_sha256=str(payload.get("source_tree_sha256") or ""),
            matrix_id=str(payload.get("matrix_id") or ""),
            matrix_raw_sha256=str(payload.get("matrix_raw_sha256") or ""),
            matrix_payload_sha256=str(
                payload.get("matrix_payload_sha256") or ""
            ),
            matrix_expanded_semantic_sha256=str(
                payload.get("matrix_expanded_semantic_sha256") or ""
            ),
            matrix_accuracy_policy_id=str(
                payload.get("matrix_accuracy_policy_id") or ""
            ),
            matrix_accuracy_policy_sha256=str(
                payload.get("matrix_accuracy_policy_sha256") or ""
            ),
            accuracy_policy_mode=str(payload.get("accuracy_policy_mode") or ""),
            accuracy_policy=dict(accuracy_policy),
            accuracy_policy_sha256=str(
                payload.get("accuracy_policy_sha256") or ""
            ),
            qiskit_compilation=bool(payload.get("qiskit_compilation")),
        )
        if payload.get("bindings_sha256") != instance.digest:
            raise ProspectivePlannerError(
                "prospective scientific bindings self digest mismatch"
            )
        instance.validate_against_matrix()
        return instance

    def key(self, arm: str) -> CellKey:
        if arm not in ARMS:
            raise ProspectivePlannerError(f"unsupported arm {arm!r}")
        return CellKey(
            regime=self.policy_regime,
            drive=self.drive,
            method=arm,
            controller=self.controller,
        )


def prospective_run_action_id(
    bindings: ProspectiveScientificBindings,
    *,
    arm: str,
    horizon: str,
    activation_cut: float,
) -> str:
    """Return the immutable scientific identity of one trajectory action."""

    if arm not in ARMS:
        raise ProspectivePlannerError(f"unsupported arm {arm!r}")
    if horizon not in HORIZONS:
        raise ProspectivePlannerError(f"unsupported horizon {horizon!r}")
    cut, index = _registered_cut(activation_cut)
    return (
        f"prospective::{bindings.lane_id}::{bindings.cell_id}::{arm}::"
        f"{horizon}::r{index:02d}::cut{_cut_token(cut)}::"
        f"bind{bindings.digest[:16]}"
    )


def prospective_safeguard_id(
    bindings: ProspectiveScientificBindings,
    *,
    reason: str,
    subject_action_id: str,
) -> str:
    token = _canonical_sha(
        {
            "bindings_sha256": bindings.digest,
            "reason": reason,
            "subject_action_id": subject_action_id,
        }
    )[:16]
    return f"prospective-safeguard::{bindings.cell_id}::{reason}::{token}"


@dataclass(frozen=True)
class ProspectiveRunEvidence:
    """Verified scheduler/run/score evidence for one scientific action attempt."""

    action_id: str
    bindings_sha256: str
    arm: str
    horizon: str
    activation_cut: float
    attempt: int
    planner_sequence: int
    requested_memory_gib: int
    state: str
    resume_from_action_id: str | None = None
    resume_run_receipt_sha256: str | None = None
    resume_run_sha256: str | None = None
    run_receipt_sha256: str | None = None
    run_sha256: str | None = None
    score_state: str = SCORE_NOT_STARTED
    score_receipt_sha256: str | None = None
    score_sha256: str | None = None
    metrics: AccuracyMetrics | None = None
    failure_kind: str | None = None
    failure_receipt_sha256: str | None = None
    failure_reason: str = ""
    observed_peak_gib: float | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ProspectivePlannerError(f"unsupported arm {self.arm!r}")
        if self.horizon not in HORIZONS:
            raise ProspectivePlannerError(f"unsupported horizon {self.horizon!r}")
        cut, _index = _registered_cut(self.activation_cut)
        object.__setattr__(self, "activation_cut", cut)
        object.__setattr__(
            self,
            "bindings_sha256",
            _require_sha256(self.bindings_sha256, label="bindings_sha256"),
        )
        if self.state not in RUN_STATES:
            raise ProspectivePlannerError(f"unsupported run state {self.state!r}")
        if self.score_state not in SCORE_STATES:
            raise ProspectivePlannerError(
                f"unsupported score state {self.score_state!r}"
            )
        if self.attempt not in (0, 1):
            raise ProspectivePlannerError(
                "only the initial execution and one resource retry are authorized"
            )
        if int(self.planner_sequence) < 0:
            raise ProspectivePlannerError("planner_sequence must be nonnegative")
        if (
            int(self.requested_memory_gib) < MEMORY_TIER_GIB
            or int(self.requested_memory_gib) > MAX_MEMORY_GIB
            or int(self.requested_memory_gib) % MEMORY_TIER_GIB
        ):
            raise ProspectivePlannerError(
                "requested memory must be a 4-GiB tier between 4 and 32 GiB"
            )
        if self.state == STATE_COMPLETED:
            if self.run_receipt_sha256 is None or self.run_sha256 is None:
                raise ProspectivePlannerError(
                    "completed run evidence requires verified receipt and run digests"
                )
            _require_sha256(self.run_receipt_sha256, label="run receipt")
            _require_sha256(self.run_sha256, label="run artifact")
            if self.failure_kind is not None or self.failure_receipt_sha256 is not None:
                raise ProspectivePlannerError(
                    "completed run evidence cannot also be failed"
                )
        elif self.state == STATE_FAILED:
            if not str(self.failure_reason).strip():
                raise ProspectivePlannerError("failed evidence requires a reason")
            if self.failure_kind not in (FAILURE_MEMORY, FAILURE_OTHER):
                raise ProspectivePlannerError(
                    "failed evidence requires a recognized failure kind"
                )
            if self.failure_receipt_sha256 is None:
                raise ProspectivePlannerError(
                    "failed evidence requires a verified failure receipt digest"
                )
            _require_sha256(self.failure_receipt_sha256, label="failure receipt")
            if self.failure_kind == FAILURE_MEMORY:
                if self.observed_peak_gib is None or not math.isfinite(
                    float(self.observed_peak_gib)
                ) or float(self.observed_peak_gib) <= 0.0:
                    raise ProspectivePlannerError(
                        "a memory failure requires a positive observed peak"
                    )
            if any(
                value is not None
                for value in (
                    self.run_receipt_sha256,
                    self.run_sha256,
                    self.score_receipt_sha256,
                    self.score_sha256,
                    self.metrics,
                )
            ):
                raise ProspectivePlannerError(
                    "failed evidence cannot carry successful run or score artifacts"
                )
            if self.score_state != SCORE_NOT_STARTED:
                raise ProspectivePlannerError(
                    "failed run evidence cannot carry a score state"
                )
        else:
            if any(
                value is not None
                for value in (
                    self.run_receipt_sha256,
                    self.run_sha256,
                    self.score_receipt_sha256,
                    self.score_sha256,
                    self.metrics,
                    self.failure_kind,
                    self.failure_receipt_sha256,
                )
            ):
                raise ProspectivePlannerError(
                    "queued run evidence cannot carry terminal artifacts"
                )
            if self.score_state != SCORE_NOT_STARTED:
                raise ProspectivePlannerError(
                    "queued run evidence cannot carry a score state"
                )

        if self.horizon == HORIZON_PREFIX:
            if any(
                value is not None
                for value in (
                    self.resume_from_action_id,
                    self.resume_run_receipt_sha256,
                    self.resume_run_sha256,
                )
            ):
                raise ProspectivePlannerError(
                    "a prefix run cannot carry continuation provenance"
                )
        else:
            if any(
                value is None
                for value in (
                    self.resume_from_action_id,
                    self.resume_run_receipt_sha256,
                    self.resume_run_sha256,
                )
            ):
                raise ProspectivePlannerError(
                    "a full-horizon run requires exact prefix receipt/run provenance"
                )
            _require_sha256(
                str(self.resume_run_receipt_sha256), label="resume run receipt"
            )
            _require_sha256(str(self.resume_run_sha256), label="resume run")

        if self.score_state == SCORE_COMPLETED:
            if self.state != STATE_COMPLETED or any(
                value is None
                for value in (
                    self.score_receipt_sha256,
                    self.score_sha256,
                    self.metrics,
                )
            ):
                raise ProspectivePlannerError(
                    "a completed score requires a completed run, score digests, "
                    "and metrics"
                )
            _require_sha256(str(self.score_receipt_sha256), label="score receipt")
            _require_sha256(str(self.score_sha256), label="score artifact")
        elif self.score_state in {SCORE_SUBMITTED, SCORE_IN_FLIGHT}:
            if self.state != STATE_COMPLETED or any(
                value is not None
                for value in (
                    self.score_receipt_sha256,
                    self.score_sha256,
                    self.metrics,
                )
            ):
                raise ProspectivePlannerError(
                    "an active score may bind only a completed raw run"
                )
        elif self.score_state == SCORE_FAILED:
            if self.state != STATE_COMPLETED or not str(self.failure_reason).strip():
                raise ProspectivePlannerError(
                    "a failed score requires a completed raw run and failure reason"
                )
            if any(
                value is not None
                for value in (
                    self.score_receipt_sha256,
                    self.score_sha256,
                    self.metrics,
                )
            ):
                raise ProspectivePlannerError(
                    "a failed score cannot carry promotable score artifacts"
                )
        elif any(
            value is not None
            for value in (
                self.score_receipt_sha256,
                self.score_sha256,
                self.metrics,
            )
        ):
            raise ProspectivePlannerError(
                "score artifacts require score_state='completed'"
            )

    @property
    def ladder_index(self) -> int:
        return _registered_cut(self.activation_cut)[1]

    @property
    def scored(self) -> bool:
        return self.score_state == SCORE_COMPLETED


@dataclass(frozen=True)
class ProspectiveSafeguardEvidence:
    """Positive durable receipt for a planner-emitted terminal safeguard."""

    safeguard_id: str
    bindings_sha256: str
    reason: str
    subject_action_id: str
    receipt_sha256: str
    planner_sequence: int

    def __post_init__(self) -> None:
        if self.reason not in {
            SAFEGUARD_LADDER_UNRESOLVED,
            SAFEGUARD_OPERATIONALLY_INFEASIBLE,
        }:
            raise ProspectivePlannerError("unsupported safeguard reason")
        _require_sha256(self.bindings_sha256, label="safeguard bindings")
        _require_sha256(self.receipt_sha256, label="safeguard receipt")
        _require_nonempty(self.subject_action_id, label="safeguard subject")
        if int(self.planner_sequence) < 0:
            raise ProspectivePlannerError(
                "safeguard planner_sequence must be nonnegative"
            )


@dataclass(frozen=True)
class ProspectiveEvidence:
    """Immutable input history for one physical cell."""

    runs: tuple[ProspectiveRunEvidence, ...] = ()
    safeguards: tuple[ProspectiveSafeguardEvidence, ...] = ()


@dataclass(frozen=True)
class ProspectiveNextAction:
    """One exact action contract; this value performs no side effect."""

    kind: str
    action_id: str
    execution_id: str
    arm: str | None
    horizon: str | None
    activation_cut: float | None
    ladder_index: int | None
    attempt: int
    planner_sequence: int
    requested_memory_gib: int | None
    reason: str
    inputs: Mapping[str, Any]
    exact_reference_blind: bool
    qiskit_compilation_requested: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "action_id": self.action_id,
            "execution_id": self.execution_id,
            "arm": self.arm,
            "horizon": self.horizon,
            "activation_cut": self.activation_cut,
            "ladder_index": self.ladder_index,
            "attempt": self.attempt,
            "planner_sequence": self.planner_sequence,
            "requested_memory_gib": self.requested_memory_gib,
            "reason": self.reason,
            "inputs": dict(self.inputs),
            "exact_reference_blind": self.exact_reference_blind,
            "qiskit_compilation_requested": False,
        }


@dataclass(frozen=True)
class ProspectiveDisposition:
    subject_id: str
    disposition: str
    reason: str

    def to_json_dict(self) -> dict[str, str]:
        return {
            "subject_id": self.subject_id,
            "disposition": self.disposition,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProspectiveScientificPlan:
    """Deterministic current frontier for one prospective physical cell."""

    bindings_sha256: str
    phase: str
    next_actions: tuple[ProspectiveNextAction, ...]
    dispositions: tuple[ProspectiveDisposition, ...]
    selected_prefix_action_ids: Mapping[str, str] = field(default_factory=dict)
    selected_full_action_ids: Mapping[str, str] = field(default_factory=dict)
    complete: bool = False
    unresolved: bool = False
    blocked: bool = False
    cost_comparison_admissible: bool = False

    @property
    def digest(self) -> str:
        return _canonical_sha(self.to_json_dict(include_digest=False))

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROSPECTIVE_SCIENTIFIC_PLAN_V1,
            "bindings_sha256": self.bindings_sha256,
            "phase": self.phase,
            "next_actions": [row.to_json_dict() for row in self.next_actions],
            "dispositions": [row.to_json_dict() for row in self.dispositions],
            "selected_prefix_action_ids": dict(self.selected_prefix_action_ids),
            "selected_full_action_ids": dict(self.selected_full_action_ids),
            "complete": self.complete,
            "unresolved": self.unresolved,
            "blocked": self.blocked,
            "cost_comparison_admissible": self.cost_comparison_admissible,
            "qiskit_compilation_requested": False,
        }
        if include_digest:
            payload["planner_receipt_sha256"] = self.digest
        return payload


class _Planner:
    def __init__(
        self,
        bindings: ProspectiveScientificBindings,
        policy: AccuracyPolicy,
        evidence: ProspectiveEvidence,
        *,
        initial_memory_gib: int,
    ) -> None:
        self.bindings = bindings
        self.policy = policy
        self.evidence = evidence
        bindings.validate_against_matrix()
        if (
            policy_sha256(policy) != bindings.accuracy_policy_sha256
            or policy.to_json_dict()
            != _deep_plain_json(bindings.accuracy_policy)
        ):
            raise ProspectivePlannerError(
                "accuracy policy digest does not match the immutable binding"
            )
        if (
            initial_memory_gib < MEMORY_TIER_GIB
            or initial_memory_gib > MAX_MEMORY_GIB
            or initial_memory_gib % MEMORY_TIER_GIB
        ):
            raise ProspectivePlannerError(
                "initial memory must be a 4-GiB tier between 4 and 32 GiB"
            )
        self.initial_memory_gib = int(initial_memory_gib)
        self.dispositions: list[ProspectiveDisposition] = []
        self._records: dict[str, list[ProspectiveRunEvidence]] = {}
        self._validate_history()
        self.next_sequence = 1 + max(
            [-1]
            + [row.planner_sequence for row in evidence.runs]
            + [row.planner_sequence for row in evidence.safeguards]
        )

    def _validate_history(self) -> None:
        seen_attempts: set[tuple[str, int]] = set()
        prefix_indexes: dict[str, set[int]] = {arm: set() for arm in ARMS}
        for row in self.evidence.runs:
            if row.bindings_sha256 != self.bindings.digest:
                raise ProspectivePlannerError(
                    "run evidence binds a different physical/scientific cell"
                )
            expected_id = prospective_run_action_id(
                self.bindings,
                arm=row.arm,
                horizon=row.horizon,
                activation_cut=row.activation_cut,
            )
            if row.action_id != expected_id:
                raise ProspectivePlannerError(
                    "run action ID does not match its immutable scientific fields"
                )
            key = (row.action_id, row.attempt)
            if key in seen_attempts:
                raise ProspectivePlannerError(
                    "prospective evidence repeats one execution attempt"
                )
            seen_attempts.add(key)
            self._records.setdefault(row.action_id, []).append(row)
            if row.horizon == HORIZON_PREFIX:
                prefix_indexes[row.arm].add(row.ladder_index)
            if row.horizon == HORIZON_FULL:
                expected_resume = prospective_run_action_id(
                    self.bindings,
                    arm=row.arm,
                    horizon=HORIZON_PREFIX,
                    activation_cut=row.activation_cut,
                )
                if row.resume_from_action_id != expected_resume:
                    raise ProspectivePlannerError(
                        "a full-horizon run must resume its same-arm, same-cut prefix"
                    )

        for action_id, attempts in self._records.items():
            attempts.sort(key=lambda row: row.attempt)
            if len(attempts) > 2:
                raise ProspectivePlannerError(
                    "more than one resource-only retry was recorded"
                )
            if len(attempts) == 2:
                first, second = attempts
                if first.attempt != 0 or second.attempt != 1:
                    raise ProspectivePlannerError(
                        "resource attempts must be contiguous 0 then 1"
                    )
                if first.state != STATE_FAILED or first.failure_kind != FAILURE_MEMORY:
                    raise ProspectivePlannerError(
                        "attempt 1 is authorized only after verified memory exhaustion"
                    )
                if first.requested_memory_gib >= MAX_MEMORY_GIB:
                    raise ProspectivePlannerError(
                        "attempt 1 cannot follow a failure already requesting 32 GiB"
                    )
                expected = self._retry_memory(first)
                if second.requested_memory_gib != expected:
                    raise ProspectivePlannerError(
                        "resource retry does not use the authorized next memory tier"
                    )
            terminal_count = sum(
                row.state in {STATE_COMPLETED, STATE_FAILED} for row in attempts
            )
            if terminal_count > 1 and not (
                len(attempts) == 2
                and attempts[0].state == STATE_FAILED
                and attempts[1].state in {STATE_COMPLETED, STATE_FAILED}
            ):
                raise ProspectivePlannerError(
                    f"contradictory terminal evidence for {action_id}"
                )

        for row in self.evidence.runs:
            if row.horizon != HORIZON_FULL:
                continue
            prefix_attempts = self._records.get(str(row.resume_from_action_id), [])
            prefix = next(
                (
                    candidate
                    for candidate in reversed(prefix_attempts)
                    if candidate.state == STATE_COMPLETED
                ),
                None,
            )
            if prefix is None:
                raise ProspectivePlannerError(
                    "full-horizon evidence has no completed verified prefix"
                )
            if (
                row.resume_run_receipt_sha256 != prefix.run_receipt_sha256
                or row.resume_run_sha256 != prefix.run_sha256
            ):
                raise ProspectivePlannerError(
                    "full-horizon evidence does not bind the completed prefix digests"
                )
            if not prefix.scored or prefix.metrics is None:
                raise ProspectivePlannerError(
                    "full-horizon evidence resumed an unscored prefix"
                )
            peer_arm = ARM_APM if row.arm == ARM_AVQDS else ARM_AVQDS
            matched_peer = False
            for attempts in self._records.values():
                peer = next(
                    (
                        candidate
                        for candidate in reversed(attempts)
                        if candidate.arm == peer_arm
                        and candidate.horizon == HORIZON_PREFIX
                        and candidate.scored
                    ),
                    None,
                )
                if peer is None or peer.metrics is None:
                    continue
                left, right = (
                    (prefix, peer) if prefix.arm == ARM_APM else (peer, prefix)
                )
                assert left.metrics is not None and right.metrics is not None
                if self.policy.pair_matches(
                    left.metrics,
                    left_key=self.bindings.key(ARM_APM),
                    right_metrics=right.metrics,
                    right_key=self.bindings.key(ARM_AVQDS),
                    horizon=HORIZON_PREFIX,
                ):
                    matched_peer = True
                    break
            if not matched_peer:
                raise ProspectivePlannerError(
                    "full-horizon evidence has no verified matched prefix pair"
                )

        # Prefix evidence may expand in either direction, but it may never omit
        # an unmeasured rung between two measured/emitted cuts.
        for arm, indexes in prefix_indexes.items():
            if indexes and indexes != set(range(min(indexes), max(indexes) + 1)):
                raise ProspectivePlannerError(
                    f"{arm} prefix evidence skips a registered activation rung"
                )

        for safeguard in self.evidence.safeguards:
            if safeguard.bindings_sha256 != self.bindings.digest:
                raise ProspectivePlannerError(
                    "safeguard evidence binds a different scientific cell"
                )
            if safeguard.subject_action_id not in self._records:
                raise ProspectivePlannerError(
                    "safeguard evidence names no verified action in this cell"
                )
            expected = prospective_safeguard_id(
                self.bindings,
                reason=safeguard.reason,
                subject_action_id=safeguard.subject_action_id,
            )
            if safeguard.safeguard_id != expected:
                raise ProspectivePlannerError(
                    "safeguard ID does not match its immutable subject"
                )

    @staticmethod
    def _retry_memory(record: ProspectiveRunEvidence) -> int:
        assert record.observed_peak_gib is not None
        scaled = 1.5 * float(record.observed_peak_gib)
        tier = int(math.ceil(scaled / MEMORY_TIER_GIB) * MEMORY_TIER_GIB)
        tier = max(tier, int(record.requested_memory_gib) + MEMORY_TIER_GIB)
        return min(tier, MAX_MEMORY_GIB)

    def _binding_inputs(self) -> dict[str, Any]:
        return self.bindings.to_json_dict()

    def _run_action(
        self,
        *,
        arm: str,
        horizon: str,
        cut: float,
        attempt: int,
        memory_gib: int,
        reason: str,
        resume: ProspectiveRunEvidence | None,
        resource_only_retry: bool,
        seeded_from_scout_action_id: str | None = None,
    ) -> ProspectiveNextAction:
        action_id = prospective_run_action_id(
            self.bindings, arm=arm, horizon=horizon, activation_cut=cut
        )
        inputs = {
            **self._binding_inputs(),
            "arm": arm,
            "horizon": horizon,
            "activation_cut": cut,
            "ladder_index": _registered_cut(cut)[1],
            "record_statevector": True,
            "exact_reference_blind": True,
            "resource_only_retry": resource_only_retry,
            "scientific_action_id": action_id,
            "seeded_from_scout_action_id": seeded_from_scout_action_id,
            "resume_from_action_id": None if resume is None else resume.action_id,
            "resume_run_receipt_sha256": (
                None if resume is None else resume.run_receipt_sha256
            ),
            "resume_run_sha256": None if resume is None else resume.run_sha256,
        }
        return ProspectiveNextAction(
            kind=KIND_RUN,
            action_id=action_id,
            execution_id=f"{action_id}::attempt{attempt}::mem{memory_gib}g",
            arm=arm,
            horizon=horizon,
            activation_cut=cut,
            ladder_index=_registered_cut(cut)[1],
            attempt=attempt,
            planner_sequence=self.next_sequence,
            requested_memory_gib=memory_gib,
            reason=reason,
            inputs=inputs,
            exact_reference_blind=True,
        )

    def _score_action(
        self, record: ProspectiveRunEvidence, *, reason: str
    ) -> ProspectiveNextAction:
        assert record.run_receipt_sha256 is not None
        assert record.run_sha256 is not None
        inputs = {
            **self._binding_inputs(),
            "arm": record.arm,
            "scientific_action_id": record.action_id,
            "run_receipt_sha256": record.run_receipt_sha256,
            "run_sha256": record.run_sha256,
            "reference_manifest_sha256": self.bindings.reference_manifest_sha256,
            "reference_data_sha256": self.bindings.reference_data_sha256,
            "accuracy_policy_sha256": self.bindings.accuracy_policy_sha256,
            "reporting_only_exact_evaluation": True,
        }
        action_id = f"score::{record.action_id}"
        return ProspectiveNextAction(
            kind=KIND_SCORE,
            action_id=action_id,
            execution_id=action_id,
            arm=record.arm,
            horizon=record.horizon,
            activation_cut=record.activation_cut,
            ladder_index=record.ladder_index,
            attempt=record.attempt,
            planner_sequence=self.next_sequence,
            requested_memory_gib=None,
            reason=reason,
            inputs=inputs,
            exact_reference_blind=False,
        )

    def _safeguard_action(
        self,
        *,
        reason: str,
        subject_action_id: str,
        explanation: str,
    ) -> ProspectiveNextAction | None:
        safeguard_id = prospective_safeguard_id(
            self.bindings,
            reason=reason,
            subject_action_id=subject_action_id,
        )
        if any(row.safeguard_id == safeguard_id for row in self.evidence.safeguards):
            return None
        return ProspectiveNextAction(
            kind=KIND_RECORD_SAFEGUARD,
            action_id=safeguard_id,
            execution_id=safeguard_id,
            arm=None,
            horizon=None,
            activation_cut=None,
            ladder_index=None,
            attempt=0,
            planner_sequence=self.next_sequence,
            requested_memory_gib=None,
            reason=explanation,
            inputs={
                **self._binding_inputs(),
                "safeguard_reason": reason,
                "subject_action_id": subject_action_id,
                "retain_all_numerical_evidence": True,
            },
            exact_reference_blind=False,
        )

    def _ensure_run(
        self,
        *,
        arm: str,
        horizon: str,
        index: int,
        reason: str,
        resume: ProspectiveRunEvidence | None = None,
        seeded_from_scout_action_id: str | None = None,
    ) -> tuple[ProspectiveRunEvidence | None, tuple[ProspectiveNextAction, ...], bool]:
        cut = HH_HUBBARD_ACTIVATION_CUT_LADDER[index]
        action_id = prospective_run_action_id(
            self.bindings, arm=arm, horizon=horizon, activation_cut=cut
        )
        attempts = self._records.get(action_id, [])
        completed = next(
            (row for row in reversed(attempts) if row.state == STATE_COMPLETED),
            None,
        )
        if completed is not None:
            if completed.score_state == SCORE_COMPLETED:
                return completed, (), False
            if completed.score_state in {SCORE_SUBMITTED, SCORE_IN_FLIGHT}:
                self.dispositions.append(
                    ProspectiveDisposition(
                        action_id,
                        "waiting_for_score",
                        "the exact-reference score is already active",
                    )
                )
                return None, (), False
            if completed.score_state == SCORE_FAILED:
                self.dispositions.append(
                    ProspectiveDisposition(
                        action_id,
                        "blocked",
                        "offline scoring failed and must be repaired without "
                        "changing the run",
                    )
                )
                return None, (), True
            return None, (self._score_action(completed, reason=reason),), False

        active = next(
            (
                row
                for row in reversed(attempts)
                if row.state in {STATE_SUBMITTED, STATE_IN_FLIGHT}
            ),
            None,
        )
        if active is not None:
            self.dispositions.append(
                ProspectiveDisposition(
                    action_id,
                    "in_flight",
                    "the exact scientific action already has an active execution",
                )
            )
            return None, (), False

        failed = next(
            (row for row in reversed(attempts) if row.state == STATE_FAILED), None
        )
        if failed is not None:
            if failed.failure_kind != FAILURE_MEMORY:
                self.dispositions.append(
                    ProspectiveDisposition(
                        action_id,
                        "blocked",
                        "a non-memory execution failure requires plumbing repair",
                    )
                )
                return None, (), True
            if failed.attempt == 0 and failed.requested_memory_gib < MAX_MEMORY_GIB:
                memory = self._retry_memory(failed)
                return (
                    None,
                    (
                        self._run_action(
                            arm=arm,
                            horizon=horizon,
                            cut=cut,
                            attempt=1,
                            memory_gib=memory,
                            reason=(
                                "one authorized resource-only retry after verified "
                                "memory exhaustion"
                            ),
                            resume=resume,
                            resource_only_retry=True,
                            seeded_from_scout_action_id=seeded_from_scout_action_id,
                        ),
                    ),
                    False,
                )
            safeguard = self._safeguard_action(
                reason=SAFEGUARD_OPERATIONALLY_INFEASIBLE,
                subject_action_id=action_id,
                explanation=(
                    "the authorized resource-only retry is exhausted; classify "
                    "this action as operationally infeasible"
                ),
            )
            return None, (() if safeguard is None else (safeguard,)), False

        return (
            None,
            (
                self._run_action(
                    arm=arm,
                    horizon=horizon,
                    cut=cut,
                    attempt=0,
                    memory_gib=self.initial_memory_gib,
                    reason=reason,
                    resume=resume,
                    resource_only_retry=False,
                    seeded_from_scout_action_id=seeded_from_scout_action_id,
                ),
            ),
            False,
        )

    def _broad(self, record: ProspectiveRunEvidence) -> bool:
        assert record.metrics is not None
        return self.policy.comparison_eligible(
            record.metrics,
            key=self.bindings.key(record.arm),
            horizon=record.horizon,
        )

    def _pair(
        self,
        apm: ProspectiveRunEvidence,
        avqds: ProspectiveRunEvidence,
    ) -> bool:
        assert apm.metrics is not None and avqds.metrics is not None
        if apm.horizon != avqds.horizon:
            return False
        return self.policy.pair_matches(
            apm.metrics,
            left_key=self.bindings.key(ARM_APM),
            right_metrics=avqds.metrics,
            right_key=self.bindings.key(ARM_AVQDS),
            horizon=apm.horizon,
        )

    def _scored(self, arm: str, horizon: str) -> dict[int, ProspectiveRunEvidence]:
        rows: dict[int, ProspectiveRunEvidence] = {}
        for attempts in self._records.values():
            for row in reversed(attempts):
                if row.arm == arm and row.horizon == horizon and row.scored:
                    previous = rows.get(row.ladder_index)
                    if previous is None or row.attempt > previous.attempt:
                        rows[row.ladder_index] = row
                    break
        return rows

    def _alignment_direction(
        self,
        avqds: ProspectiveRunEvidence,
        apm: ProspectiveRunEvidence,
    ) -> int:
        assert avqds.metrics is not None and apm.metrics is not None
        if not self._broad(avqds):
            return 1
        avqds_key = self.bindings.key(ARM_AVQDS)
        apm_key = self.bindings.key(ARM_APM)
        horizon = avqds.horizon
        left = self.policy.response_components(
            avqds.metrics, key=avqds_key, horizon=horizon
        )
        right = self.policy.response_components(
            apm.metrics, key=apm_key, horizon=horizon
        )
        avqds_burden = self.policy.accuracy_burden(
            avqds.metrics, key=avqds_key, horizon=horizon
        )
        apm_burden = self.policy.accuracy_burden(
            apm.metrics, key=apm_key, horizon=horizon
        )
        factor = float(self.policy.matched_accuracy_factor)
        floor = max(float(self.policy.response_scale_floor), 1.0e-300)
        avqds_burden = max(avqds_burden, floor)
        apm_burden = max(apm_burden, floor)
        if avqds_burden / apm_burden > factor:
            return 1
        if apm_burden / avqds_burden > factor:
            return -1
        deadband = float(self.policy.matched_component_deadband)
        mismatches: list[tuple[float, int]] = []
        for name in sorted(left):
            scout = max(float(left[name]), deadband)
            anchor = max(float(right[name]), deadband)
            ratio = max(scout / anchor, anchor / scout)
            if ratio > factor:
                mismatches.append((ratio, 1 if scout > anchor else -1))
        if mismatches:
            return max(mismatches, key=lambda row: row[0])[1]
        return 1 if avqds_burden > apm_burden else -1

    def _terminal_plan(
        self,
        *,
        phase: str,
        unresolved: bool,
        subject: str,
        reason: str,
    ) -> ProspectiveScientificPlan:
        self.dispositions.append(ProspectiveDisposition(subject, phase, reason))
        return ProspectiveScientificPlan(
            bindings_sha256=self.bindings.digest,
            phase=phase,
            next_actions=(),
            dispositions=tuple(self.dispositions),
            complete=True,
            unresolved=unresolved,
            blocked=False,
            cost_comparison_admissible=False,
        )

    def _pending_plan(
        self,
        *,
        phase: str,
        actions: Sequence[ProspectiveNextAction],
        blocked: bool = False,
        selected_prefix: Mapping[str, str] | None = None,
        selected_full: Mapping[str, str] | None = None,
    ) -> ProspectiveScientificPlan:
        return ProspectiveScientificPlan(
            bindings_sha256=self.bindings.digest,
            phase=phase,
            next_actions=tuple(sorted(actions, key=lambda row: row.action_id)),
            dispositions=tuple(self.dispositions),
            selected_prefix_action_ids=dict(selected_prefix or {}),
            selected_full_action_ids=dict(selected_full or {}),
            complete=False,
            unresolved=False,
            blocked=blocked,
            cost_comparison_admissible=False,
        )

    def _recorded_cell_safeguard(self) -> ProspectiveSafeguardEvidence | None:
        return next(
            (
                row
                for row in self.evidence.safeguards
                if row.reason
                in {
                    SAFEGUARD_LADDER_UNRESOLVED,
                    SAFEGUARD_OPERATIONALLY_INFEASIBLE,
                }
            ),
            None,
        )

    def plan(self) -> ProspectiveScientificPlan:
        recorded = self._recorded_cell_safeguard()
        if recorded is not None:
            return self._terminal_plan(
                phase=recorded.reason,
                unresolved=True,
                subject=recorded.subject_action_id,
                reason="the durable terminal safeguard is already recorded",
            )

        # 1. AVQDS prefix scout, strictly coarse to tight.
        scout: ProspectiveRunEvidence | None = None
        for index in range(len(HH_HUBBARD_ACTIVATION_CUT_LADDER)):
            record, actions, blocked = self._ensure_run(
                arm=ARM_AVQDS,
                horizon=HORIZON_PREFIX,
                index=index,
                reason="advance the AVQDS prefix scout by one registered rung",
            )
            if actions or blocked or record is None:
                return self._pending_plan(
                    phase="avqds_prefix_scout",
                    actions=actions,
                    blocked=blocked,
                )
            if self._broad(record):
                scout = record
                break

        if scout is None:
            apm_prefix_indexes = {
                row.ladder_index
                for row in self.evidence.runs
                if row.arm == ARM_APM and row.horizon == HORIZON_PREFIX
            }
            floor_index = len(HH_HUBBARD_ACTIVATION_CUT_LADDER) - 1
            if apm_prefix_indexes and apm_prefix_indexes != {floor_index}:
                raise ProspectivePlannerError(
                    "APM evidence preceded AVQDS broad-guard entry/exhaustion"
                )
            # AVQDS exhausted outside the guard.  One APM floor diagnostic is
            # still required before the cell becomes terminally unresolved.
            diagnostic, actions, blocked = self._ensure_run(
                arm=ARM_APM,
                horizon=HORIZON_PREFIX,
                index=floor_index,
                reason=(
                    "tightest-feasible APM diagnostic after AVQDS ladder exhaustion"
                ),
                seeded_from_scout_action_id=prospective_run_action_id(
                    self.bindings,
                    arm=ARM_AVQDS,
                    horizon=HORIZON_PREFIX,
                    activation_cut=HH_HUBBARD_ACTIVATION_CUT_LADDER[floor_index],
                ),
            )
            if actions or blocked or diagnostic is None:
                return self._pending_plan(
                    phase="avqds_exhausted_apm_diagnostic",
                    actions=actions,
                    blocked=blocked,
                )
            subject = diagnostic.action_id
            safeguard = self._safeguard_action(
                reason=SAFEGUARD_LADDER_UNRESOLVED,
                subject_action_id=subject,
                explanation=(
                    "AVQDS exhausted the registered ladder outside the broad guard; "
                    "retain the scored tightest-APM diagnostic without a cost claim"
                ),
            )
            if safeguard is not None:
                return self._pending_plan(
                    phase="record_ladder_unresolved",
                    actions=(safeguard,),
                )
            return self._terminal_plan(
                phase=SAFEGUARD_LADDER_UNRESOLVED,
                unresolved=True,
                subject=subject,
                reason="AVQDS ladder exhausted outside the broad guard",
            )

        apm_prefix_indexes = {
            row.ladder_index
            for row in self.evidence.runs
            if row.arm == ARM_APM and row.horizon == HORIZON_PREFIX
        }
        if apm_prefix_indexes and (
            scout.ladder_index not in apm_prefix_indexes
            or min(apm_prefix_indexes) < scout.ladder_index
        ):
            raise ProspectivePlannerError(
                "APM calibration did not begin at the verified AVQDS scout cut"
            )

        # Full-horizon APM failures reopen independent APM calibration at the
        # next tighter prefix rung.  Pair mismatches do not move the APM anchor.
        apm_full = self._scored(ARM_APM, HORIZON_FULL)
        rejected_apm_indexes = {
            index for index, row in apm_full.items() if not self._broad(row)
        }
        apm_start = max(
            scout.ladder_index,
            (max(rejected_apm_indexes) + 1) if rejected_apm_indexes else 0,
        )
        anchor: ProspectiveRunEvidence | None = None
        for index in range(apm_start, len(HH_HUBBARD_ACTIVATION_CUT_LADDER)):
            record, actions, blocked = self._ensure_run(
                arm=ARM_APM,
                horizon=HORIZON_PREFIX,
                index=index,
                reason=(
                    "seed the first APM cut from the AVQDS scout"
                    if index == scout.ladder_index and not self._scored(
                        ARM_APM, HORIZON_PREFIX
                    )
                    else "advance independent APM calibration by one registered rung"
                ),
                seeded_from_scout_action_id=(
                    scout.action_id if index == scout.ladder_index else None
                ),
            )
            if actions or blocked or record is None:
                return self._pending_plan(
                    phase="apm_prefix_calibration",
                    actions=actions,
                    blocked=blocked,
                    selected_prefix={ARM_AVQDS: scout.action_id},
                )
            if self._broad(record):
                anchor = record
                break

        if anchor is None:
            floor_action = prospective_run_action_id(
                self.bindings,
                arm=ARM_APM,
                horizon=HORIZON_PREFIX,
                activation_cut=HH_HUBBARD_ACTIVATION_CUT_LADDER[-1],
            )
            safeguard = self._safeguard_action(
                reason=SAFEGUARD_LADDER_UNRESOLVED,
                subject_action_id=floor_action,
                explanation=(
                    "APM exhausted the registered prefix ladder outside the broad guard"
                ),
            )
            if safeguard is not None:
                return self._pending_plan(
                    phase="record_ladder_unresolved", actions=(safeguard,)
                )
            return self._terminal_plan(
                phase=SAFEGUARD_LADDER_UNRESOLVED,
                unresolved=True,
                subject=floor_action,
                reason="APM ladder exhausted outside the broad guard",
            )

        # 3. Freeze APM and align AVQDS.  Full-horizon mismatches for this APM
        # anchor exclude only the already-tested AVQDS cuts and choose direction
        # from the measured full-horizon scores.
        avqds_prefix = self._scored(ARM_AVQDS, HORIZON_PREFIX)
        anchor_full = apm_full.get(anchor.ladder_index)
        rejected_avqds: set[int] = set()
        full_avqds = self._scored(ARM_AVQDS, HORIZON_FULL)
        if anchor_full is not None and self._broad(anchor_full):
            for index, row in full_avqds.items():
                if not self._pair(anchor_full, row):
                    rejected_avqds.add(index)

        matching = [
            row
            for index, row in avqds_prefix.items()
            if index not in rejected_avqds and self._pair(anchor, row)
        ]
        aligned: ProspectiveRunEvidence | None = None
        if matching:
            aligned = max(
                matching, key=lambda row: (row.planner_sequence, row.action_id)
            )
        else:
            current = max(
                avqds_prefix.values(),
                key=lambda row: (row.planner_sequence, row.action_id),
            )
            direction_source = full_avqds.get(current.ladder_index)
            direction_anchor = anchor_full
            if (
                current.ladder_index in rejected_avqds
                and direction_source is not None
                and direction_anchor is not None
            ):
                direction = self._alignment_direction(
                    direction_source, direction_anchor
                )
            else:
                direction = self._alignment_direction(current, anchor)
            preferred_direction = direction
            candidate_index = current.ladder_index + direction

            # An already measured adjacent result is consumed rather than rerun.
            # A direction cycle does not imply monotonicity: continue from the
            # nearest unmeasured edge of the contiguous measured interval.
            visited: set[int] = set()
            while candidate_index in avqds_prefix and candidate_index not in visited:
                visited.add(candidate_index)
                candidate = avqds_prefix[candidate_index]
                if candidate_index not in rejected_avqds and self._pair(
                    anchor, candidate
                ):
                    aligned = candidate
                    break
                current = candidate
                direction = self._alignment_direction(candidate, anchor)
                candidate_index = candidate_index + direction
                if (
                    candidate_index in visited
                    or candidate_index < 0
                    or candidate_index >= len(HH_HUBBARD_ACTIVATION_CUT_LADDER)
                ):
                    low = min(avqds_prefix) - 1
                    high = max(avqds_prefix) + 1
                    frontier = (
                        (low, high)
                        if preferred_direction < 0
                        else (high, low)
                    )
                    candidate_index = next(
                        (
                            value
                            for value in frontier
                            if 0
                            <= value
                            < len(HH_HUBBARD_ACTIVATION_CUT_LADDER)
                            and value not in avqds_prefix
                        ),
                        -1,
                    )
                    break

            if aligned is None:
                if candidate_index < 0 or candidate_index >= len(
                    HH_HUBBARD_ACTIVATION_CUT_LADDER
                ):
                    low = min(avqds_prefix) - 1
                    high = max(avqds_prefix) + 1
                    frontier = (
                        (low, high)
                        if preferred_direction < 0
                        else (high, low)
                    )
                    candidate_index = next(
                        (
                            value
                            for value in frontier
                            if 0
                            <= value
                            < len(HH_HUBBARD_ACTIVATION_CUT_LADDER)
                            and value not in avqds_prefix
                        ),
                        -1,
                    )
                if candidate_index < 0 or candidate_index >= len(
                    HH_HUBBARD_ACTIVATION_CUT_LADDER
                ):
                    subject = current.action_id
                    safeguard = self._safeguard_action(
                        reason=SAFEGUARD_LADDER_UNRESOLVED,
                        subject_action_id=subject,
                        explanation=(
                            "no registered adjacent AVQDS rung can match the frozen "
                            "APM anchor inside the broad guard"
                        ),
                    )
                    if safeguard is not None:
                        return self._pending_plan(
                            phase="record_ladder_unresolved", actions=(safeguard,)
                        )
                    return self._terminal_plan(
                        phase=SAFEGUARD_LADDER_UNRESOLVED,
                        unresolved=True,
                        subject=subject,
                        reason="AVQDS/APM alignment exhausted the registered ladder",
                    )
                direction = 1 if candidate_index > current.ladder_index else -1
                record, actions, blocked = self._ensure_run(
                    arm=ARM_AVQDS,
                    horizon=HORIZON_PREFIX,
                    index=candidate_index,
                    reason=(
                        "move AVQDS one adjacent tighter rung against the frozen "
                        "APM anchor"
                        if direction > 0
                        else "move AVQDS one adjacent coarser rung against the "
                        "frozen APM anchor"
                    ),
                )
                if actions or blocked or record is None:
                    return self._pending_plan(
                        phase="avqds_prefix_alignment",
                        actions=actions,
                        blocked=blocked,
                        selected_prefix={ARM_APM: anchor.action_id},
                    )
                if self._pair(anchor, record):
                    aligned = record

        if aligned is None:
            raise ProspectivePlannerError(
                "internal alignment state did not select or emit an AVQDS rung"
            )

        selected_prefix = {
            ARM_APM: anchor.action_id,
            ARM_AVQDS: aligned.action_id,
        }

        # 4. Continue both exact prefix endpoints.  These two actions are
        # independently ready and therefore returned together when absent.
        full_rows: dict[str, ProspectiveRunEvidence] = {}
        actions: list[ProspectiveNextAction] = []
        blocked = False
        for arm, prefix in ((ARM_APM, anchor), (ARM_AVQDS, aligned)):
            row, emitted, arm_blocked = self._ensure_run(
                arm=arm,
                horizon=HORIZON_FULL,
                index=prefix.ladder_index,
                reason="continue the selected matched prefix endpoint to t=20",
                resume=prefix,
            )
            actions.extend(emitted)
            blocked = blocked or arm_blocked
            if row is not None:
                full_rows[arm] = row
        if actions or blocked or len(full_rows) != 2:
            return self._pending_plan(
                phase="full_horizon_continuation_or_score",
                actions=actions,
                blocked=blocked,
                selected_prefix=selected_prefix,
                selected_full={
                    arm: row.action_id for arm, row in full_rows.items()
                },
            )

        selected_full = {arm: row.action_id for arm, row in full_rows.items()}
        if self._pair(full_rows[ARM_APM], full_rows[ARM_AVQDS]):
            self.dispositions.append(
                ProspectiveDisposition(
                    self.bindings.cell_id,
                    "matched_qualified",
                    "both full-horizon arms pass the guards and the aggregate plus "
                    "deadbanded primitive factor-two rule",
                )
            )
            return ProspectiveScientificPlan(
                bindings_sha256=self.bindings.digest,
                phase="complete",
                next_actions=(),
                dispositions=tuple(self.dispositions),
                selected_prefix_action_ids=selected_prefix,
                selected_full_action_ids=selected_full,
                complete=True,
                unresolved=False,
                blocked=False,
                cost_comparison_admissible=True,
            )

        # No action is emitted directly here.  Re-entering this same pure plan
        # immediately observes the measured T=20 miss through rejected_apm or
        # rejected_avqds above and emits the next adjacent prefix calibration.
        # We recurse once to expose that action in the current transaction.
        return self.plan_after_full_miss(
            anchor=anchor,
            aligned=aligned,
            full_apm=full_rows[ARM_APM],
            full_avqds=full_rows[ARM_AVQDS],
            selected_prefix=selected_prefix,
            selected_full=selected_full,
        )

    def plan_after_full_miss(
        self,
        *,
        anchor: ProspectiveRunEvidence,
        aligned: ProspectiveRunEvidence,
        full_apm: ProspectiveRunEvidence,
        full_avqds: ProspectiveRunEvidence,
        selected_prefix: Mapping[str, str],
        selected_full: Mapping[str, str],
    ) -> ProspectiveScientificPlan:
        """Expose the next calibration action implicated by measured T=20 data."""

        if not self._broad(full_apm):
            next_index = anchor.ladder_index + 1
            if next_index >= len(HH_HUBBARD_ACTIVATION_CUT_LADDER):
                subject = full_apm.action_id
                safeguard = self._safeguard_action(
                    reason=SAFEGUARD_LADDER_UNRESOLVED,
                    subject_action_id=subject,
                    explanation="T=20 APM miss exhausted the registered ladder",
                )
                return self._pending_plan(
                    phase="record_ladder_unresolved",
                    actions=(() if safeguard is None else (safeguard,)),
                    selected_prefix=selected_prefix,
                    selected_full=selected_full,
                )
            row, actions, blocked = self._ensure_run(
                arm=ARM_APM,
                horizon=HORIZON_PREFIX,
                index=next_index,
                reason="T=20 APM guard miss reopens independent APM calibration",
                seeded_from_scout_action_id=None,
            )
            if row is not None and not actions:
                # Existing evidence will be consumed on the next pure planning
                # call; avoid inventing a second transition in one result.
                self.dispositions.append(
                    ProspectiveDisposition(
                        row.action_id,
                        "reopen_calibration",
                        "existing adjacent APM prefix evidence must be reconsidered",
                    )
                )
            return self._pending_plan(
                phase="t20_miss_reopens_apm_calibration",
                actions=actions,
                blocked=blocked,
                selected_prefix=selected_prefix,
                selected_full=selected_full,
            )

        direction = self._alignment_direction(full_avqds, full_apm)
        next_index = aligned.ladder_index + direction
        if next_index < 0 or next_index >= len(HH_HUBBARD_ACTIVATION_CUT_LADDER):
            subject = full_avqds.action_id
            safeguard = self._safeguard_action(
                reason=SAFEGUARD_LADDER_UNRESOLVED,
                subject_action_id=subject,
                explanation=(
                    "T=20 AVQDS/APM mismatch has no registered adjacent AVQDS rung"
                ),
            )
            return self._pending_plan(
                phase="record_ladder_unresolved",
                actions=(() if safeguard is None else (safeguard,)),
                selected_prefix=selected_prefix,
                selected_full=selected_full,
            )
        row, actions, blocked = self._ensure_run(
            arm=ARM_AVQDS,
            horizon=HORIZON_PREFIX,
            index=next_index,
            reason=(
                "T=20 mismatch reopens AVQDS calibration one tighter rung"
                if direction > 0
                else "T=20 mismatch reopens AVQDS calibration one coarser rung"
            ),
        )
        if row is not None and not actions:
            self.dispositions.append(
                ProspectiveDisposition(
                    row.action_id,
                    "reopen_calibration",
                    "existing adjacent AVQDS prefix evidence must be reconsidered",
                )
            )
        return self._pending_plan(
            phase="t20_miss_reopens_avqds_alignment",
            actions=actions,
            blocked=blocked,
            selected_prefix=selected_prefix,
            selected_full=selected_full,
        )


def plan_prospective_cell(
    bindings: ProspectiveScientificBindings,
    policy: AccuracyPolicy,
    *,
    evidence: ProspectiveEvidence | None = None,
    initial_memory_gib: int = INITIAL_MEMORY_GIB,
) -> ProspectiveScientificPlan:
    """Return the exact current action frontier for one HH/Hubbard cell."""

    return _Planner(
        bindings,
        policy,
        evidence or ProspectiveEvidence(),
        initial_memory_gib=initial_memory_gib,
    ).plan()


__all__ = [
    "ARM_APM",
    "ARM_AVQDS",
    "FAILURE_MEMORY",
    "FAILURE_OTHER",
    "HH_HUBBARD_ACTIVATION_CUT_LADDER",
    "HORIZON_FULL",
    "HORIZON_PREFIX",
    "KIND_RECORD_SAFEGUARD",
    "KIND_RUN",
    "KIND_SCORE",
    "LANE_HH",
    "LANE_HUBBARD",
    "MAX_MEMORY_GIB",
    "PROSPECTIVE_SCIENTIFIC_BINDINGS_V1",
    "PROSPECTIVE_SCIENTIFIC_PLAN_V1",
    "ProspectiveEvidence",
    "ProspectiveNextAction",
    "ProspectivePlannerError",
    "ProspectiveRunEvidence",
    "ProspectiveSafeguardEvidence",
    "ProspectiveScientificBindings",
    "ProspectiveScientificPlan",
    "SAFEGUARD_LADDER_UNRESOLVED",
    "SAFEGUARD_OPERATIONALLY_INFEASIBLE",
    "SCORE_COMPLETED",
    "SCORE_FAILED",
    "SCORE_IN_FLIGHT",
    "SCORE_NOT_STARTED",
    "SCORE_SUBMITTED",
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_IN_FLIGHT",
    "STATE_SUBMITTED",
    "plan_prospective_cell",
    "policy_sha256",
    "prospective_run_action_id",
    "prospective_safeguard_id",
]
