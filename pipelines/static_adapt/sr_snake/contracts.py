"""Immutable public contracts for the SR-SNAKE deep run seam.

The types in this module expose only intentional scientific choices.  Legacy
route strings, optimizer controls, numerical guards, and compatibility payloads
remain private implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Mapping

from pipelines.contracts.problem import ResolvedProblemContext

CANONICAL_CANDIDATE_REPRESENTATION = (
    "physical_macro_lanes_to_symmetry_hard_guarded_"
    "cardinality_one_pauli_children_v1"
)


def _serialize(value: Any) -> Any:
    if isinstance(value, SerializableContract):
        return value.to_dict()
    if is_dataclass(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
            if getattr(value, field.name) is not None
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _serialize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _serialize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _require_positive_int(value: int, *, name: str) -> int:
    resolved = int(value)
    if isinstance(value, bool) or resolved != value or resolved < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved


def _require_nonnegative_int(value: int, *, name: str) -> int:
    resolved = int(value)
    if isinstance(value, bool) or resolved != value or resolved < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return resolved


def _require_finite(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _require_positive_finite(value: float, *, name: str) -> float:
    resolved = _require_finite(value, name=name)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _require_nonempty(value: str, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must not be empty.")
    return resolved


def _require_sha256(value: str, *, name: str) -> str:
    resolved = str(value).strip().lower()
    if len(resolved) != 64 or any(
        char not in "0123456789abcdef" for char in resolved
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")
    return resolved


class SerializableContract:
    """Deterministic JSON projection shared by request and receipt contracts."""

    kind: ClassVar[str | None] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            field.name: _serialize(getattr(self, field.name))
            for field in fields(self)
            if getattr(self, field.name) is not None
        }
        if self.kind is not None:
            payload = {"kind": self.kind, **payload}
        return payload

    def to_json(self) -> str:
        return _canonical_json(self)


@dataclass(frozen=True)
class SingletonAdmission(SerializableContract):
    kind: ClassVar[str] = "singleton"


@dataclass(frozen=True)
class GreedyBatchAdmission(SerializableContract):
    maximum_size: int = 3
    search_window_size: int | None = None
    kind: ClassVar[str] = "greedy_batch"

    def __post_init__(self) -> None:
        maximum_size = _require_positive_int(
            self.maximum_size,
            name="maximum_size",
        )
        if maximum_size > 5:
            raise ValueError(
                "maximum_size must not exceed the greedy reduced-plane "
                "kernel ceiling of 5."
            )
        object.__setattr__(self, "maximum_size", maximum_size)
        if self.search_window_size is not None:
            object.__setattr__(
                self,
                "search_window_size",
                _require_positive_int(
                    self.search_window_size,
                    name="search_window_size",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the enabled greedy policy, including its full-window choice."""

        return {
            "kind": self.kind,
            "maximum_size": int(self.maximum_size),
            "search_window_size": (
                None
                if self.search_window_size is None
                else int(self.search_window_size)
            ),
        }


@dataclass(frozen=True)
class FullCombinatorialSearchWindow(SerializableContract):
    """Explicitly request the full ranked Phase-III combinatorial population."""

    kind: ClassVar[str] = "full_ranked_phase3_population"


@dataclass(frozen=True)
class CombinatorialBatchAdmission(SerializableContract):
    maximum_size: int = 3
    search_window_size: int | FullCombinatorialSearchWindow | None = None
    kind: ClassVar[str] = "combinatorial_batch"

    def __post_init__(self) -> None:
        maximum_size = _require_positive_int(
            self.maximum_size,
            name="maximum_size",
        )
        if maximum_size > 5:
            raise ValueError(
                "maximum_size must not exceed the combinatorial reduced-plane "
                "kernel ceiling of 5."
            )
        object.__setattr__(self, "maximum_size", maximum_size)
        if self.search_window_size is None:
            object.__setattr__(
                self,
                "search_window_size",
                min(2 * maximum_size, 10),
            )
        elif not isinstance(
            self.search_window_size,
            FullCombinatorialSearchWindow,
        ):
            object.__setattr__(
                self,
                "search_window_size",
                _require_positive_int(
                    self.search_window_size,
                    name="search_window_size",
                ),
            )

    @property
    def resolved_search_window_size(self) -> int | None:
        """Return the numerical prefix width; ``None`` means the full population."""

        if isinstance(
            self.search_window_size,
            FullCombinatorialSearchWindow,
        ):
            return None
        return int(self.search_window_size)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resolved bounded prefix or explicit full-window choice."""

        return {
            "kind": self.kind,
            "maximum_size": int(self.maximum_size),
            "search_window_size": self.resolved_search_window_size,
        }


AdmissionPolicy = (
    SingletonAdmission | GreedyBatchAdmission | CombinatorialBatchAdmission
)


@dataclass(frozen=True)
class PlateauCommutationInsertion(SerializableContract):
    """Canonical plateau-triggered commutation-reduced insertion."""

    kind: ClassVar[str] = "plateau_commutation"


@dataclass(frozen=True)
class AlwaysCommutationReducedInsertion(SerializableContract):
    """Always open the full domain and score certified class representatives."""

    kind: ClassVar[str] = "always_commutation_reduced"


@dataclass(frozen=True)
class AppendCommutationReducedInsertion(SerializableContract):
    """Score only the endpoint after passing it through the exact reducer."""

    kind: ClassVar[str] = "append_commutation_reduced"
    runtime_mode: ClassVar[str] = "append_commutation_reduced"
    position_scope: ClassVar[str] = "append_endpoint_only_every_depth_v1"
    equivalence_policy: ClassVar[str] = (
        "termwise_cross_component_commutation_earliest_representative_v1"
    )
    receipt_key: ClassVar[str] = "insertion_commutation_reduced"


@dataclass(frozen=True)
class AppendOnlyInsertion(SerializableContract):
    """Explicit historical replay/ablation insertion policy."""

    kind: ClassVar[str] = "append_only"


InsertionPolicy = (
    PlateauCommutationInsertion
    | AlwaysCommutationReducedInsertion
    | AppendCommutationReducedInsertion
    | AppendOnlyInsertion
)


@dataclass(frozen=True)
class PruningOff(SerializableContract):
    kind: ClassVar[str] = "off"


@dataclass(frozen=True)
class MetricPruning(SerializableContract):
    """Regularized local metric/response deletion nomination."""

    kind: ClassVar[str] = "metric"


@dataclass(frozen=True)
class TrustRegionPruning(SerializableContract):
    """Full-logical trust-domain deletion nomination."""

    kind: ClassVar[str] = "trust_region"


@dataclass(frozen=True)
class RecoverabilityPruning(SerializableContract):
    """Historical name for the measured trust-region pruning route."""

    kind: ClassVar[str] = "recoverability"


PruningPolicy = (
    PruningOff
    | MetricPruning
    | TrustRegionPruning
    | RecoverabilityPruning
)


@dataclass(frozen=True)
class BeamOff(SerializableContract):
    kind: ClassVar[str] = "off"


@dataclass(frozen=True)
class ForkLocalBeam(SerializableContract):
    live_parent_branches: int = 3
    admission_children_per_parent: int = 2
    maximum_admission_children_per_round: int = 6
    s_alg_weight: float = 0.01
    calibration_status: str = field(
        default="uncalibrated_default",
        init=False,
    )
    kind: ClassVar[str] = "fork_local"

    def __post_init__(self) -> None:
        live = _require_positive_int(
            self.live_parent_branches,
            name="live_parent_branches",
        )
        children = _require_positive_int(
            self.admission_children_per_parent,
            name="admission_children_per_parent",
        )
        maximum = _require_positive_int(
            self.maximum_admission_children_per_round,
            name="maximum_admission_children_per_round",
        )
        if children < 2:
            raise ValueError(
                "fork-local beam requires at least two admission children "
                "per parent."
            )
        if maximum < children:
            raise ValueError(
                "maximum_admission_children_per_round must be at least "
                "admission_children_per_parent."
            )
        object.__setattr__(self, "live_parent_branches", live)
        object.__setattr__(self, "admission_children_per_parent", children)
        object.__setattr__(
            self,
            "maximum_admission_children_per_round",
            maximum,
        )
        object.__setattr__(
            self,
            "s_alg_weight",
            _require_positive_finite(self.s_alg_weight, name="s_alg_weight"),
        )
        if self.calibration_status != "uncalibrated_default":
            raise ValueError(
                "ForkLocalBeam calibration_status is fixed to "
                "'uncalibrated_default' until a calibrated policy is approved."
            )


BeamPolicy = BeamOff | ForkLocalBeam


@dataclass(frozen=True)
class EndpointOverlapDisplacementTrust(SerializableContract):
    """Use exact endpoint Fubini--Study motion for adaptive trust updates.

    The absence of this explicit ablation keeps the canonical query-neutral
    source-Gram trust policy byte-for-byte unchanged.
    """

    kind: ClassVar[str] = "endpoint_overlap_displacement"


@dataclass(frozen=True)
class SRMethodPolicy(SerializableContract):
    admission: AdmissionPolicy = field(default_factory=SingletonAdmission)
    insertion: InsertionPolicy = field(
        default_factory=PlateauCommutationInsertion
    )
    pruning: PruningPolicy = field(default_factory=PruningOff)
    beam: BeamPolicy = field(default_factory=BeamOff)
    trust_update: EndpointOverlapDisplacementTrust | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.admission,
            (
                SingletonAdmission,
                GreedyBatchAdmission,
                CombinatorialBatchAdmission,
            ),
        ):
            raise TypeError("admission must be an SR-SNAKE admission policy.")
        if not isinstance(
            self.insertion,
            (
                PlateauCommutationInsertion,
                AlwaysCommutationReducedInsertion,
                AppendCommutationReducedInsertion,
                AppendOnlyInsertion,
            ),
        ):
            raise TypeError("insertion must be an SR-SNAKE insertion policy.")
        if not isinstance(
            self.pruning,
            (
                PruningOff,
                MetricPruning,
                TrustRegionPruning,
                RecoverabilityPruning,
            ),
        ):
            raise TypeError("pruning must be an SR-SNAKE pruning policy.")
        if isinstance(self.pruning, RecoverabilityPruning) and not (
            isinstance(self.admission, SingletonAdmission)
            and isinstance(self.insertion, AppendOnlyInsertion)
            and isinstance(self.beam, BeamOff)
        ):
            raise ValueError(
                "RecoverabilityPruning is a preserved historical policy and "
                "requires the explicit singleton + append-only + beam-off "
                "compatibility identity."
            )
        if not isinstance(self.beam, (BeamOff, ForkLocalBeam)):
            raise TypeError("beam must be an SR-SNAKE beam policy.")
        if self.trust_update is not None and not isinstance(
            self.trust_update,
            EndpointOverlapDisplacementTrust,
        ):
            raise TypeError(
                "trust_update must be EndpointOverlapDisplacementTrust or None."
            )


@dataclass(frozen=True)
class ResolvedProblemReceipt(SerializableContract):
    family_key: str
    problem_request_sha256: str
    problem_key: str
    num_sites: int
    t: float
    u: float
    dv: float
    v_nn: float
    t_prime: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    include_zero_point: bool
    n_fermions: int | None
    sector_label: str
    comparison_space_label: str
    reference_label: str
    exact_target_label: str
    total_qubits: int

    @classmethod
    def from_problem(
        cls,
        problem: ResolvedProblemContext,
    ) -> "ResolvedProblemReceipt":
        request = problem.request
        request_payload = {
            field.name: _serialize(getattr(request, field.name))
            for field in fields(request)
        }
        request_sha256 = hashlib.sha256(
            _canonical_json(request_payload).encode("utf-8")
        ).hexdigest()
        return cls(
            family_key=str(problem.family_key),
            problem_request_sha256=request_sha256,
            problem_key=str(request.problem_key),
            num_sites=int(request.num_sites),
            t=float(request.t),
            u=float(request.u),
            dv=float(request.dv),
            v_nn=float(request.v_nn),
            t_prime=float(request.t_prime),
            omega0=float(request.omega0),
            g_ep=float(request.g_ep),
            n_ph_max=int(request.n_ph_max),
            boson_encoding=str(request.boson_encoding),
            ordering=str(request.ordering),
            boundary=str(request.boundary),
            include_zero_point=bool(request.include_zero_point),
            n_fermions=(
                None if request.n_fermions is None else int(request.n_fermions)
            ),
            sector_label=str(problem.sector.label),
            comparison_space_label=str(problem.exact_comparison_space_label),
            reference_label=str(problem.default_reference_label),
            exact_target_label=str(problem.exact_target_label),
            total_qubits=int(problem.layout.total_qubits),
        )


@dataclass(frozen=True)
class ExactEDSourceReceipt(SerializableContract):
    source_id: str
    problem_request_sha256: str
    sector_label: str
    comparison_space_label: str
    n_ph_max: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _require_nonempty(self.source_id, name="source_id"),
        )
        object.__setattr__(
            self,
            "problem_request_sha256",
            _require_sha256(
                self.problem_request_sha256,
                name="problem_request_sha256",
            ),
        )
        object.__setattr__(
            self,
            "sector_label",
            _require_nonempty(self.sector_label, name="sector_label"),
        )
        object.__setattr__(
            self,
            "comparison_space_label",
            _require_nonempty(
                self.comparison_space_label,
                name="comparison_space_label",
            ),
        )
        if int(self.n_ph_max) < 0:
            raise ValueError("n_ph_max must be nonnegative.")
        object.__setattr__(self, "n_ph_max", int(self.n_ph_max))

    @classmethod
    def from_problem(
        cls,
        problem: ResolvedProblemContext,
        *,
        source_id: str,
    ) -> "ExactEDSourceReceipt":
        receipt = ResolvedProblemReceipt.from_problem(problem)
        return cls(
            source_id=source_id,
            problem_request_sha256=receipt.problem_request_sha256,
            sector_label=receipt.sector_label,
            comparison_space_label=receipt.comparison_space_label,
            n_ph_max=receipt.n_ph_max,
        )


@dataclass(frozen=True)
class ExactEDStop(SerializableContract):
    energy: float
    absolute_tolerance: float
    source: ExactEDSourceReceipt
    confirmation_controller_rounds: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy",
            _require_finite(self.energy, name="energy"),
        )
        object.__setattr__(
            self,
            "absolute_tolerance",
            _require_positive_finite(
                self.absolute_tolerance,
                name="absolute_tolerance",
            ),
        )
        if not isinstance(self.source, ExactEDSourceReceipt):
            raise TypeError("source must be an ExactEDSourceReceipt.")
        object.__setattr__(
            self,
            "confirmation_controller_rounds",
            _require_nonnegative_int(
                self.confirmation_controller_rounds,
                name="confirmation_controller_rounds",
            ),
        )


@dataclass(frozen=True)
class SRStopPolicy(SerializableContract):
    maximum_controller_rounds: int = 50
    exact_ed_target: ExactEDStop | None = None
    gradient_tolerance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_controller_rounds",
            _require_positive_int(
                self.maximum_controller_rounds,
                name="maximum_controller_rounds",
            ),
        )
        if self.exact_ed_target is not None and not isinstance(
            self.exact_ed_target,
            ExactEDStop,
        ):
            raise TypeError("exact_ed_target must be an ExactEDStop or None.")
        if self.gradient_tolerance is not None:
            object.__setattr__(
                self,
                "gradient_tolerance",
                _require_positive_finite(
                    self.gradient_tolerance,
                    name="gradient_tolerance",
                ),
            )


@dataclass(frozen=True)
class FreshStart(SerializableContract):
    kind: ClassVar[str] = "fresh_start"


@dataclass(frozen=True)
class AcceptedStateResume(SerializableContract):
    checkpoint_path: Path
    checkpoint_sha256: str
    kind: ClassVar[str] = "accepted_state_resume"

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _require_sha256(
                self.checkpoint_sha256,
                name="checkpoint_sha256",
            ),
        )


ResumePolicy = FreshStart | AcceptedStateResume


@dataclass(frozen=True)
class SRExecutionPolicy(SerializableContract):
    stop: SRStopPolicy = field(default_factory=SRStopPolicy)
    resume: ResumePolicy = field(default_factory=FreshStart)

    def __post_init__(self) -> None:
        if not isinstance(self.stop, SRStopPolicy):
            raise TypeError("stop must be an SRStopPolicy.")
        if not isinstance(self.resume, (FreshStart, AcceptedStateResume)):
            raise TypeError("resume must be a typed SR-SNAKE resume policy.")


@dataclass(frozen=True)
class CheckpointObservation(SerializableContract):
    path: Path
    every_controller_rounds: int = 1
    keep_history_tail: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "every_controller_rounds",
            _require_positive_int(
                self.every_controller_rounds,
                name="every_controller_rounds",
            ),
        )
        if int(self.keep_history_tail) < 0:
            raise ValueError("keep_history_tail must be nonnegative.")
        object.__setattr__(self, "keep_history_tail", int(self.keep_history_tail))


@dataclass(frozen=True)
class EstimatorLedgerObservation(SerializableContract):
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class SRObservationPolicy(SerializableContract):
    checkpoint: CheckpointObservation | None = None
    estimator_ledger: EstimatorLedgerObservation | None = None
    resource_rounds: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.checkpoint is not None and not isinstance(
            self.checkpoint,
            CheckpointObservation,
        ):
            raise TypeError("checkpoint must be a CheckpointObservation or None.")
        if self.estimator_ledger is not None and not isinstance(
            self.estimator_ledger,
            EstimatorLedgerObservation,
        ):
            raise TypeError(
                "estimator_ledger must be an EstimatorLedgerObservation or None."
            )
        if self.resource_rounds is not None:
            rounds = tuple(self.resource_rounds)
            if any(
                isinstance(value, bool)
                or int(value) != value
                or int(value) < 1
                for value in rounds
            ):
                raise ValueError(
                    "resource_rounds must contain positive controller-round "
                    "integers."
                )
            normalized = tuple(int(value) for value in rounds)
            if len(set(normalized)) != len(normalized):
                raise ValueError("resource_rounds must not contain duplicates.")
            object.__setattr__(
                self,
                "resource_rounds",
                tuple(sorted(normalized)),
            )
        if (
            self.checkpoint is not None
            and self.estimator_ledger is not None
            and self.checkpoint.path.resolve(strict=False)
            == self.estimator_ledger.path.resolve(strict=False)
        ):
            raise ValueError(
                "Checkpoint and estimator-ledger destinations must differ."
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.checkpoint is not None:
            payload["checkpoint"] = self.checkpoint.to_dict()
        if self.estimator_ledger is not None:
            payload["estimator_ledger"] = self.estimator_ledger.to_dict()
        if self.resource_rounds is not None:
            payload["resource_rounds"] = list(self.resource_rounds)
        return payload


@dataclass(frozen=True)
class SRRunRequest(SerializableContract):
    method: SRMethodPolicy = field(default_factory=SRMethodPolicy)
    execution: SRExecutionPolicy = field(default_factory=SRExecutionPolicy)
    observation: SRObservationPolicy = field(
        default_factory=SRObservationPolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.method, SRMethodPolicy):
            raise TypeError("method must be an SRMethodPolicy.")
        if not isinstance(self.execution, SRExecutionPolicy):
            raise TypeError("execution must be an SRExecutionPolicy.")
        if not isinstance(self.observation, SRObservationPolicy):
            raise TypeError("observation must be an SRObservationPolicy.")


@dataclass(frozen=True)
class ResolvedExecutionReceipt(SerializableContract):
    pool: str
    optimizer: str
    optimizer_maxiter: int
    seed: int
    phase0_enabled: bool
    phase2_batching_enabled: bool
    phase3_batching_enabled: bool
    pruning_enabled: bool
    beam_enabled: bool
    phase_live_hysteresis_enabled: bool
    phase3_response_coordinate_scope: str
    trust_policy: str
    accepted_refit_policy: str
    accepted_refit_scope: str
    accepted_refit_coordinate_chart: str


@dataclass(frozen=True)
class RouteReceipt(SerializableContract):
    family: str
    profile_request: str
    profile: str
    contract_sha256: str
    method: SRMethodPolicy
    admission_policy: str
    insertion_policy: str
    pruning_policy: str
    beam_policy: str
    execution: ResolvedExecutionReceipt


@dataclass(frozen=True)
class AcceptedStateReceipt(SerializableContract):
    controller_round: int
    operators: tuple[str, ...]
    insertion_positions: tuple[int, ...]
    generator_ids: tuple[str, ...]
    logical_parameters: tuple[float, ...]
    runtime_parameters: tuple[float, ...]
    energy: float
    projective_state_fingerprint: str


@dataclass(frozen=True)
class RecoverabilityPruneReceipt(SerializableContract):
    """Portable receipt for one post-refit recoverability-prune stage."""

    status: str
    reason: str
    policy: str
    nomination_policy: str
    source_state_fingerprint: str
    trust_radius_before: float
    trust_radius_after: float
    metric_damping: float
    endpoint_overlap_query_charge: int
    terminal_prune_active: bool
    nomination_index: int | None = None
    nomination_label: str | None = None
    predicted_energy_change: float | None = None
    surrogate_used_for_acceptance: bool | None = None
    trial_executed: bool = False
    trial_branch_id: str | None = None
    trial_classification: str | None = None
    trial_s_alg: int | None = None
    measured_energy_before: float | None = None
    measured_energy_after: float | None = None
    accepted: bool | None = None
    deleted_index: int | None = None
    deleted_label: str | None = None
    final_state_fingerprint: str | None = None
    kind: ClassVar[str] = "recoverability"

    def __post_init__(self) -> None:
        if self.status not in {"not_executed", "accepted", "rejected"}:
            raise ValueError("recoverability receipt status is invalid.")
        if not self.reason.strip():
            raise ValueError("recoverability receipt reason must be non-empty.")
        if self.policy != "recoverability_ladder_v1":
            raise ValueError("recoverability receipt policy is fixed.")
        if self.nomination_policy not in {
            "full_logical_fs_trust_delete_refit_v1",
            "metric_regularized_v1",
        }:
            raise ValueError("recoverability nomination policy is fixed.")
        if self.endpoint_overlap_query_charge != 0:
            raise ValueError("recoverability pruning must not charge overlap.")
        if self.terminal_prune_active:
            raise ValueError("recoverability terminal prune must remain off.")
        if (
            not math.isfinite(self.trust_radius_before)
            or not math.isfinite(self.trust_radius_after)
            or self.trust_radius_before <= 0.0
            or self.trust_radius_after <= 0.0
        ):
            raise ValueError(
                "recoverability trust radii must be finite and positive."
            )
        if self.metric_damping != 0.0:
            raise ValueError("recoverability metric damping must remain zero.")
        if (
            not self.source_state_fingerprint
            or not self.final_state_fingerprint
        ):
            raise ValueError(
                "recoverability source and final fingerprints are required."
            )
        if not self.trial_executed:
            if self.status != "not_executed":
                raise ValueError(
                    "no-trial recoverability receipt must be not_executed."
                )
            trial_only = {
                "nomination_index": self.nomination_index,
                "nomination_label": self.nomination_label,
                "predicted_energy_change": self.predicted_energy_change,
                "surrogate_used_for_acceptance": (
                    self.surrogate_used_for_acceptance
                ),
                "trial_branch_id": self.trial_branch_id,
                "trial_classification": self.trial_classification,
                "trial_s_alg": self.trial_s_alg,
                "measured_energy_before": self.measured_energy_before,
                "measured_energy_after": self.measured_energy_after,
                "accepted": self.accepted,
                "deleted_index": self.deleted_index,
                "deleted_label": self.deleted_label,
            }
            populated = sorted(
                key for key, value in trial_only.items() if value is not None
            )
            if populated:
                raise ValueError(
                    "no-trial recoverability receipt carries trial-only "
                    f"fields: {populated!r}."
                )
            if self.final_state_fingerprint != self.source_state_fingerprint:
                raise ValueError(
                    "no-trial recoverability receipt must preserve the "
                    "source fingerprint."
                )
            if not math.isclose(
                self.trust_radius_after,
                self.trust_radius_before,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    "no-trial recoverability receipt must preserve trust "
                    "radius."
                )
            return

        required = (
            self.nomination_index,
            self.nomination_label,
            self.predicted_energy_change,
            self.trial_branch_id,
            self.trial_classification,
            self.trial_s_alg,
            self.measured_energy_before,
            self.measured_energy_after,
            self.accepted,
        )
        if any(value is None for value in required):
            raise ValueError(
                "measured recoverability trial receipt is incomplete."
            )
        if self.surrogate_used_for_acceptance is not False:
            raise ValueError(
                "surrogate evidence cannot authorize prune acceptance."
            )
        if self.nomination_index is None or self.nomination_index < 0:
            raise ValueError(
                "measured recoverability nominee index must be non-negative."
            )
        if not self.nomination_label or not self.trial_branch_id:
            raise ValueError(
                "measured recoverability nominee and branch identities are "
                "required."
            )
        if self.trial_s_alg is None or self.trial_s_alg < 0:
            raise ValueError(
                "recoverability trial work must be non-negative."
            )
        trial_values = (
            self.predicted_energy_change,
            self.measured_energy_before,
            self.measured_energy_after,
        )
        if any(
            value is None or not math.isfinite(float(value))
            for value in trial_values
        ):
            raise ValueError(
                "recoverability prediction and measured energies must be "
                "finite."
            )
        if self.accepted:
            if (
                self.status != "accepted"
                or self.trial_classification != "committed_prune"
            ):
                raise ValueError(
                    "accepted recoverability trial requires accepted status "
                    "and committed_prune classification."
                )
            if (
                self.deleted_index is None
                or self.deleted_label is None
                or self.deleted_index != self.nomination_index
                or self.deleted_label != self.nomination_label
            ):
                raise ValueError(
                    "accepted recoverability trial requires the nominated "
                    "deletion identity."
                )
            if not math.isclose(
                self.trust_radius_after,
                self.trust_radius_before,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    "accepted recoverability trial must preserve trust radius."
                )
        else:
            if (
                self.status != "rejected"
                or self.trial_classification != "discarded_prune"
            ):
                raise ValueError(
                    "rejected recoverability trial requires rejected status "
                    "and discarded_prune classification."
                )
            if self.deleted_index is not None or self.deleted_label is not None:
                raise ValueError(
                    "rejected recoverability trial cannot carry deletion "
                    "identity."
                )
            if self.final_state_fingerprint != self.source_state_fingerprint:
                raise ValueError(
                    "rejected recoverability trial must preserve the source "
                    "fingerprint."
                )
            if self.trust_radius_after > self.trust_radius_before:
                raise ValueError(
                    "rejected recoverability trial cannot expand trust radius."
                )


@dataclass(frozen=True)
class AcceptedTransitionReceipt(SerializableContract):
    """Portable controller receipt for one committed singleton transition."""

    controller_round: int
    preceding_state_fingerprint: str
    selected_domain_record_id: str
    generator_id: str
    selected_operator: str
    pool_index: int
    insertion_position: int
    initial_logical_value: float
    accepted_state_fingerprint: str
    energy_before: float
    energy_after: float
    refit_policy: str
    refit_scope: str
    refit_supported_rank: int
    trust_policy: str
    non_worsening_absolute_tolerance: float
    estimator_prefix_before: str
    estimator_prefix_after: str
    ledger_closure_sha256: str
    round_s_alg: int
    round_s_unique: int
    cumulative_s_alg: int
    cumulative_s_unique: int
    pruning: RecoverabilityPruneReceipt | None = None


@dataclass(frozen=True)
class GreedyBatchMemberAdmissionReceipt(SerializableContract):
    """One ordered member of an atomic greedy admission."""

    selected_domain_record_id: str
    generator_id: str
    selected_operator: str
    pool_index: int
    original_insertion_position: int
    effective_insertion_position: int
    inserted_logical_index: int
    initial_logical_value: float
    admitted_runtime_count: int
    runtime_insert_position: int
    inserted_runtime_indices: tuple[int, ...]
    source_identity: str
    child_identity: str


@dataclass(frozen=True)
class GreedyBatchProposalReceipt(SerializableContract):
    """Joint Phase-III proposal that authorized one atomic batch."""

    identity: str
    maximum_size: int
    search_window_size: int | None
    selected_cardinality: int
    selected_record_ids: tuple[str, ...]
    score: float
    modeled_energy_decrease: float
    predictive_cost_excess: float
    denominator: float
    geometry_identity: str
    evaluated_subset_count: int
    estimator_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["search_window_size"] = (
            None
            if self.search_window_size is None
            else int(self.search_window_size)
        )
        return payload


@dataclass(frozen=True)
class GreedyBatchTransitionAdmissionReceipt(SerializableContract):
    """Complete logical/runtime remap for one committed greedy batch."""

    composition_identity: str
    selected_cardinality: int
    members: tuple[GreedyBatchMemberAdmissionReceipt, ...]
    logical_parameter_count_before: int
    logical_parameter_count_after: int
    old_to_new_logical_indices: tuple[int, ...]
    inserted_logical_indices: tuple[int, ...]
    runtime_parameter_count_before: int
    runtime_parameter_count_after: int
    old_to_new_runtime_indices: tuple[int, ...]
    inserted_runtime_indices: tuple[int, ...]
    optimizer_memory_identity_before: str
    optimizer_memory_identity_after: str


@dataclass(frozen=True)
class GreedyBatchAcceptedTransitionReceipt(SerializableContract):
    """Portable controller receipt for one atomic greedy batch transition."""

    controller_round: int
    preceding_state_fingerprint: str
    proposal: GreedyBatchProposalReceipt
    admission: GreedyBatchTransitionAdmissionReceipt
    accepted_state: AcceptedStateReceipt
    energy_before: float
    energy_after: float
    refit_policy: str
    refit_scope: str
    refit_chart_dimension: int
    refit_active_logical_indices: tuple[int, ...]
    refit_supported_rank: int
    trust_policy: str
    non_worsening_absolute_tolerance: float
    estimator_prefix_before: str
    estimator_prefix_after: str
    ledger_closure_sha256: str
    round_s_alg: int
    round_s_unique: int
    cumulative_s_alg: int
    cumulative_s_unique: int
    pruning: RecoverabilityPruneReceipt | None = None


@dataclass(frozen=True)
class CombinatorialBatchMemberAdmissionReceipt(SerializableContract):
    """One fixed generator-position member of an atomic subset admission."""

    selected_domain_record_id: str
    generator_id: str
    selected_operator: str
    pool_index: int
    original_insertion_position: int
    effective_insertion_position: int
    inserted_logical_index: int
    initial_logical_value: float
    admitted_runtime_count: int
    runtime_insert_position: int
    inserted_runtime_indices: tuple[int, ...]
    source_identity: str
    child_identity: str


@dataclass(frozen=True)
class CombinatorialBatchProposalReceipt(SerializableContract):
    """Exhaustive-subset proposal authorizing one atomic admission."""

    identity: str
    maximum_size: int
    search_window_size: int | None
    ranked_population_count: int
    ranked_window_count: int
    selected_cardinality: int
    selected_record_ids: tuple[str, ...]
    score: float
    modeled_energy_decrease: float
    predictive_cost_excess: float
    denominator: float
    geometry_identity: str
    evaluated_subset_count: int
    subset_counts_considered: tuple[tuple[int, int], ...]
    subset_counts_evaluated: tuple[tuple[int, int], ...]
    subset_counts_feasible: tuple[tuple[int, int], ...]
    estimator_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["search_window_size"] = (
            None
            if self.search_window_size is None
            else int(self.search_window_size)
        )
        return payload


@dataclass(frozen=True)
class CombinatorialBatchTransitionAdmissionReceipt(SerializableContract):
    """Complete logical/runtime remap for one committed subset."""

    composition_identity: str
    selected_cardinality: int
    members: tuple[CombinatorialBatchMemberAdmissionReceipt, ...]
    logical_parameter_count_before: int
    logical_parameter_count_after: int
    old_to_new_logical_indices: tuple[int, ...]
    inserted_logical_indices: tuple[int, ...]
    runtime_parameter_count_before: int
    runtime_parameter_count_after: int
    old_to_new_runtime_indices: tuple[int, ...]
    inserted_runtime_indices: tuple[int, ...]
    optimizer_memory_identity_before: str
    optimizer_memory_identity_after: str


@dataclass(frozen=True)
class CombinatorialBatchAcceptedTransitionReceipt(SerializableContract):
    """Portable receipt for one atomic combinatorial transition."""

    controller_round: int
    preceding_state_fingerprint: str
    proposal: CombinatorialBatchProposalReceipt
    admission: CombinatorialBatchTransitionAdmissionReceipt
    accepted_state: AcceptedStateReceipt
    energy_before: float
    energy_after: float
    refit_policy: str
    refit_scope: str
    refit_chart_dimension: int
    refit_active_logical_indices: tuple[int, ...]
    refit_supported_rank: int
    trust_policy: str
    non_worsening_absolute_tolerance: float
    estimator_prefix_before: str
    estimator_prefix_after: str
    ledger_closure_sha256: str
    round_s_alg: int
    round_s_unique: int
    cumulative_s_alg: int
    cumulative_s_unique: int
    pruning: RecoverabilityPruneReceipt | None = None


@dataclass(frozen=True)
class PhaseIIIReceipt(SerializableContract):
    coordinate_scope: str
    coordinate_indices: tuple[int, ...]
    pre_support_count: int
    supported_rank: int


@dataclass(frozen=True)
class PhaseReceipt(SerializableContract):
    phase1_energy_model: str
    phase2_curvature_status: str
    phase3: PhaseIIIReceipt


@dataclass(frozen=True)
class TrustSolveReceipt(SerializableContract):
    policy: str
    update_reason: str
    endpoint_overlap_query_charge: int
    transaction_complete: bool | None = None
    transaction_failure: str | None = None
    supported_rank: int | None = None
    supported_metric_whitening_active: bool | None = None
    supported_metric_inverse_sqrt_constructed: bool | None = None
    predicted_source_metric_displacement: float | None = None
    realized_source_metric_displacement: float | None = None


@dataclass(frozen=True)
class SupportedMetricReceipt(SerializableContract):
    policy: str
    rank_relative_tolerance: float
    metric_regularization: float
    energy_regularization: float
    max_fubini_study_step: float
    global_trust_kkt_residual_accuracy: float
    global_trust_metric_distortion_budget: float


@dataclass(frozen=True)
class AcceptedRefitReceipt(SerializableContract):
    policy: str
    scope: str
    coordinate_chart: str
    base_chart_policy: str
    full_ansatz: bool
    supported_rank: int
    final_energy: float
    symmetric_metric_element_occurrences: int
    supported_metric: SupportedMetricReceipt
    initialization_policy: str | None = None
    initialization_status: str | None = None
    initialization_guard_nfev: int | None = None

    def __post_init__(self) -> None:
        initialization_fields = (
            self.initialization_policy,
            self.initialization_status,
            self.initialization_guard_nfev,
        )
        if all(value is None for value in initialization_fields):
            return
        if any(value is None for value in initialization_fields):
            raise ValueError(
                "Accepted-refit initialization evidence must be complete."
            )
        _require_nonempty(
            str(self.initialization_policy),
            name="initialization_policy",
        )
        if self.initialization_status not in {
            "disabled",
            "accepted",
            "rejected",
            "error",
            "unavailable",
        }:
            raise ValueError("Unknown accepted-refit initialization status.")
        guard_nfev = self.initialization_guard_nfev
        if (
            isinstance(guard_nfev, bool)
            or int(guard_nfev) != guard_nfev
            or int(guard_nfev) < 0
        ):
            raise ValueError(
                "Accepted-refit initialization guard count must be a "
                "non-negative integer."
            )


@dataclass(frozen=True)
class RuntimePauliTermReceipt(SerializableContract):
    pauli_exyz: str
    coefficient_real: float
    coefficient_imaginary: float
    qubit_count: int


@dataclass(frozen=True)
class ParameterBlockReceipt(SerializableContract):
    candidate_label: str
    logical_index: int
    runtime_start: int
    runtime_count: int
    execution_mode: str
    runtime_terms: tuple[RuntimePauliTermReceipt, ...]


@dataclass(frozen=True)
class CheckpointReceipt(SerializableContract):
    outer_iteration: int
    active_ansatz_depth: int
    ordered_operator_labels: tuple[str, ...]
    checkpoint_sha256: str
    projective_state_fingerprint: str
    strict_replay_passed: bool
    strict_replay_fidelity: float
    parameterization_mode: str
    parameterization_term_order: str
    parameter_blocks: tuple[ParameterBlockReceipt, ...]
    logical_parameters: tuple[float, ...]
    runtime_parameters: tuple[float, ...]
    route_profile: str
    route_contract_sha256: str
    estimator_ledger_status: str
    estimator_ledger_s_alg: int


@dataclass(frozen=True)
class ScientificReplayReceipt(SerializableContract):
    controller_round: int
    generator_id: str
    selected_operator: str
    selected_position: int
    energy_before_refit: float
    accepted_state: AcceptedStateReceipt
    phase: PhaseReceipt
    trust_solve: TrustSolveReceipt
    accepted_refit: AcceptedRefitReceipt
    checkpoint: CheckpointReceipt


@dataclass(frozen=True)
class GreedyBatchScientificReplayReceipt(SerializableContract):
    """Reader-facing replay receipt without ambiguous singleton aliases."""

    controller_round: int
    proposal: GreedyBatchProposalReceipt
    admission: GreedyBatchTransitionAdmissionReceipt
    energy_before_refit: float
    accepted_state: AcceptedStateReceipt
    phase: PhaseReceipt
    trust_solve: TrustSolveReceipt
    accepted_refit: AcceptedRefitReceipt
    checkpoint: CheckpointReceipt


@dataclass(frozen=True)
class CombinatorialBatchScientificReplayReceipt(SerializableContract):
    """Replay receipt retaining exhaustive proposal semantics."""

    controller_round: int
    proposal: CombinatorialBatchProposalReceipt
    admission: CombinatorialBatchTransitionAdmissionReceipt
    energy_before_refit: float
    accepted_state: AcceptedStateReceipt
    phase: PhaseReceipt
    trust_solve: TrustSolveReceipt
    accepted_refit: AcceptedRefitReceipt
    checkpoint: CheckpointReceipt


@dataclass(frozen=True)
class AuthenticatedResumeTransitionReceipt(SerializableContract):
    """One accepted historical transition attested by a resume checkpoint."""

    controller_round: int
    route_family: str
    selected_operators: tuple[str, ...]
    selected_pool_indices: tuple[int, ...]
    selected_positions: tuple[int, ...]
    accepted_state: AcceptedStateReceipt
    energy_before: float
    energy_after: float
    cumulative_s_alg: int
    source_checkpoint_sha256: str


@dataclass(frozen=True)
class AuthenticatedResumeScientificReplayReceipt(SerializableContract):
    """Reader-facing historical prefix reconstructed from signed replay data."""

    controller_round: int
    selected_operators: tuple[str, ...]
    energy_before_refit: float
    accepted_state: AcceptedStateReceipt
    phase: PhaseReceipt
    trust_solve: TrustSolveReceipt
    accepted_refit: AcceptedRefitReceipt
    checkpoint: CheckpointReceipt
    source_checkpoint_sha256: str


@dataclass(frozen=True)
class EstimatorComponentsReceipt(SerializableContract):
    n_h_outer: int
    n_h_refit: int
    n_grad: int
    n_metric: int


@dataclass(frozen=True)
class EstimatorWorkReceipt(SerializableContract):
    components: EstimatorComponentsReceipt
    s_alg: int


@dataclass(frozen=True)
class EstimatorAccountingReceipt(SerializableContract):
    complete: bool
    status: str
    exact_blockers: tuple[str, ...]
    all_work: EstimatorWorkReceipt
    winning_lineage: EstimatorWorkReceipt
    raw_occurrences: EstimatorComponentsReceipt
    raw_occurrence_total: int
    prefix_closure_passed: bool
    prefix_closure_status: str


@dataclass(frozen=True)
class ReferenceStateReceipt(SerializableContract):
    """Normalized same-register reference state for prefix observations."""

    amplitudes_real: tuple[float, ...]
    amplitudes_imaginary: tuple[float, ...]
    qubit_count: int
    source_label: str
    state_fingerprint: str

    def __post_init__(self) -> None:
        qubit_count = _require_positive_int(
            self.qubit_count,
            name="qubit_count",
        )
        real = tuple(
            _require_finite(value, name="amplitudes_real")
            for value in self.amplitudes_real
        )
        imaginary = tuple(
            _require_finite(value, name="amplitudes_imaginary")
            for value in self.amplitudes_imaginary
        )
        expected_size = 1 << qubit_count
        if len(real) != expected_size or len(imaginary) != expected_size:
            raise ValueError(
                "Reference-state amplitudes must cover the complete qubit "
                "register."
            )
        norm_squared = sum(
            real_value * real_value + imaginary_value * imaginary_value
            for real_value, imaginary_value in zip(
                real,
                imaginary,
                strict=True,
            )
        )
        if not math.isclose(
            norm_squared,
            1.0,
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        ):
            raise ValueError("Reference-state amplitudes must be normalized.")
        object.__setattr__(self, "amplitudes_real", real)
        object.__setattr__(self, "amplitudes_imaginary", imaginary)
        object.__setattr__(self, "qubit_count", qubit_count)
        object.__setattr__(
            self,
            "source_label",
            _require_nonempty(self.source_label, name="source_label"),
        )
        from pipelines.static_adapt.estimator_call_ledger import (
            projective_state_fingerprint,
        )

        expected_fingerprint = projective_state_fingerprint(
            tuple(
                complex(real_value, imaginary_value)
                for real_value, imaginary_value in zip(
                    real,
                    imaginary,
                    strict=True,
                )
            )
        )
        supplied_fingerprint = _require_nonempty(
            self.state_fingerprint,
            name="state_fingerprint",
        )
        if supplied_fingerprint != expected_fingerprint:
            raise ValueError(
                "Reference-state fingerprint does not authenticate its "
                "amplitudes."
            )
        object.__setattr__(
            self,
            "state_fingerprint",
            supplied_fingerprint,
        )


@dataclass(frozen=True)
class CanonicalReportingReceipt(SerializableContract):
    """Run-complete typed inputs for the single Paper-I summary seam."""

    exact_same_cutoff_energy: float
    reference_state: ReferenceStateReceipt
    horizon_scope: str
    candidate_representation: str
    accepted_prefix_work: tuple[EstimatorWorkReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_same_cutoff_energy",
            _require_finite(
                self.exact_same_cutoff_energy,
                name="exact_same_cutoff_energy",
            ),
        )
        if self.horizon_scope not in {
            "natural_terminal",
            "deliberately_stopped_prefix",
        }:
            raise ValueError(
                "horizon_scope must be natural_terminal or "
                "deliberately_stopped_prefix."
            )
        object.__setattr__(
            self,
            "candidate_representation",
            _require_nonempty(
                self.candidate_representation,
                name="candidate_representation",
            ),
        )
        previous = -1
        for index, work in enumerate(self.accepted_prefix_work, start=1):
            components = work.components
            values = (
                int(components.n_h_outer),
                int(components.n_h_refit),
                int(components.n_grad),
                int(components.n_metric),
            )
            if any(value < 0 for value in values):
                raise ValueError(
                    "Accepted-prefix estimator components must be "
                    "nonnegative."
                )
            if int(work.s_alg) != sum(values):
                raise ValueError(
                    "Accepted-prefix S_alg must close to its component sum."
                )
            if int(work.s_alg) < previous:
                raise ValueError(
                    "Accepted-prefix S_alg must be cumulative and monotone."
                )
            previous = int(work.s_alg)


@dataclass(frozen=True)
class StopConditionReceipt(SerializableContract):
    reason: str
    active: bool
    fired: bool


@dataclass(frozen=True)
class StopReceipt(SerializableContract):
    conditions: tuple[StopConditionReceipt, ...]
    completed_controller_rounds: int
    accepted_operator_count: int
    primary_reason: str
    fired_reasons: tuple[str, ...]
    accepted_energy: float
    terminal_controller_outcome: str | None = None
    exact_target_energy: float | None = None
    exact_absolute_tolerance: float | None = None
    exact_observed_absolute_difference: float | None = None
    exact_source: ExactEDSourceReceipt | None = None
    exact_confirmation_controller_rounds: int | None = None
    exact_first_hit_controller_round: int | None = None


@dataclass(frozen=True)
class ObservationArtifactReceipt(SerializableContract):
    kind: str = field()
    path: Path
    sha256: str
    size_bytes: int
    every_controller_rounds: int | None = None


@dataclass(frozen=True)
class ObservationReceipt(SerializableContract):
    artifacts: tuple[ObservationArtifactReceipt, ...] = ()


@dataclass(frozen=True)
class SRRunResult(SerializableContract):
    final_state: AcceptedStateReceipt
    accepted_trajectory: tuple[AcceptedStateReceipt, ...]
    accepted_transitions: tuple[
        AcceptedTransitionReceipt
        | GreedyBatchAcceptedTransitionReceipt
        | CombinatorialBatchAcceptedTransitionReceipt
        | AuthenticatedResumeTransitionReceipt,
        ...,
    ]
    problem: ResolvedProblemReceipt
    route: RouteReceipt
    stop: StopReceipt
    scientific_replay: tuple[
        ScientificReplayReceipt
        | GreedyBatchScientificReplayReceipt
        | CombinatorialBatchScientificReplayReceipt
        | AuthenticatedResumeScientificReplayReceipt,
        ...,
    ]
    estimator_accounting: EstimatorAccountingReceipt
    observation: ObservationReceipt
    canonical_reporting: CanonicalReportingReceipt
    paper_i_summary: Any | None = None

    def __post_init__(self) -> None:
        if not self.accepted_trajectory:
            raise ValueError("SRRunResult requires an accepted trajectory.")
        if self.final_state != self.accepted_trajectory[-1]:
            raise ValueError(
                "SRRunResult final_state must equal the final accepted "
                "trajectory state."
            )
        if len(self.accepted_trajectory) != len(
            self.canonical_reporting.accepted_prefix_work
        ):
            raise ValueError(
                "Canonical accepted-prefix work must align one-to-one with "
                "the accepted trajectory."
            )
        if (
            self.canonical_reporting.accepted_prefix_work[-1].s_alg
            > self.estimator_accounting.all_work.s_alg
        ):
            raise ValueError(
                "Accepted-prefix work cannot exceed canonical all-executed "
                "work."
            )
        if self.paper_i_summary is not None:
            if (
                not is_dataclass(self.paper_i_summary)
                or getattr(self.paper_i_summary, "schema", None)
                != "paper_i_run_summary_v1"
                or int(
                    getattr(
                        self.paper_i_summary,
                        "available_controller_rounds",
                        -1,
                    )
                )
                != len(self.accepted_trajectory)
            ):
                raise ValueError(
                    "paper_i_summary must be the aligned typed canonical "
                    "Paper-I run summary."
                )


__all__ = [
    "AcceptedRefitReceipt",
    "AcceptedStateReceipt",
    "AcceptedStateResume",
    "AcceptedTransitionReceipt",
    "AuthenticatedResumeScientificReplayReceipt",
    "AuthenticatedResumeTransitionReceipt",
    "AppendCommutationReducedInsertion",
    "AppendOnlyInsertion",
    "BeamOff",
    "CheckpointObservation",
    "CheckpointReceipt",
    "CANONICAL_CANDIDATE_REPRESENTATION",
    "CanonicalReportingReceipt",
    "CombinatorialBatchAdmission",
    "CombinatorialBatchAcceptedTransitionReceipt",
    "CombinatorialBatchMemberAdmissionReceipt",
    "CombinatorialBatchProposalReceipt",
    "CombinatorialBatchScientificReplayReceipt",
    "CombinatorialBatchTransitionAdmissionReceipt",
    "EstimatorAccountingReceipt",
    "EstimatorComponentsReceipt",
    "EstimatorLedgerObservation",
    "EstimatorWorkReceipt",
    "EndpointOverlapDisplacementTrust",
    "ExactEDSourceReceipt",
    "ExactEDStop",
    "ForkLocalBeam",
    "FreshStart",
    "AlwaysCommutationReducedInsertion",
    "FullCombinatorialSearchWindow",
    "GreedyBatchAdmission",
    "GreedyBatchAcceptedTransitionReceipt",
    "GreedyBatchMemberAdmissionReceipt",
    "GreedyBatchProposalReceipt",
    "GreedyBatchScientificReplayReceipt",
    "GreedyBatchTransitionAdmissionReceipt",
    "MetricPruning",
    "ObservationArtifactReceipt",
    "ObservationReceipt",
    "ParameterBlockReceipt",
    "PhaseIIIReceipt",
    "PhaseReceipt",
    "PlateauCommutationInsertion",
    "PruningOff",
    "RecoverabilityPruning",
    "RecoverabilityPruneReceipt",
    "ReferenceStateReceipt",
    "ResolvedExecutionReceipt",
    "ResolvedProblemReceipt",
    "RouteReceipt",
    "RuntimePauliTermReceipt",
    "SRExecutionPolicy",
    "SRMethodPolicy",
    "SRObservationPolicy",
    "SRRunRequest",
    "SRRunResult",
    "SRStopPolicy",
    "ScientificReplayReceipt",
    "SingletonAdmission",
    "StopConditionReceipt",
    "StopReceipt",
    "SupportedMetricReceipt",
    "TrustSolveReceipt",
    "TrustRegionPruning",
]
