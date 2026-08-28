#!/usr/bin/env python3
"""Typed Paper III static-QSE campaign: identities, stage graph, planner, ledger.

The campaign is a **declared matrix**, never a discovery pass over filenames.
Every scientific object has an immutable typed identity whose ``action_id`` /
digest is the SHA-256 of canonical JSON and is stable across replay:

- :class:`StaticQSEPhysicsKey` -- model family, case, Hamiltonian parameters,
  register layout, conserved sector, and cutoff (``n_ph_max`` or ``n_bath``).
- :class:`GroundStateArtifactRef` -- a prepared-state artifact with manifest
  and payload hashes; a path alone is not provenance.
- :class:`StaticQSECellKey` -- physics + ground-state artifact + public method
  + target set + dimension cap + pool/spectral/resource policy digests. A
  post-run accuracy target is **not** part of the identity: one growth trace
  is evaluated afterward at every ``eps_E``. Internal phases (record phase,
  Davidson phase) are **not** methods.
- :class:`StaticQSEAction` -- cell + stage + dependency action ids.
- :class:`StaticQSEAttempt` -- one execution's achieved configuration, digests,
  payload references, disposition, and optional qualification.

Ledger semantics: append-only; exact replay is an idempotent no-op; the same
action id with different achieved settings or output digests is a hard
conflict; a completed numerical result that fails a gate stays visible as
``candidate`` and never regresses to ``pending``. One completed action is
independently ingestible -- nothing waits for siblings, clusters, or tuples.

Reporting/diagnostic layer only; nothing here feeds controller decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from pipelines.qse_spectra.switch_policy import (
    CANONICAL_METHOD_POLICY,
    MethodPolicy,
)

STATIC_CAMPAIGN_SCHEMA_VERSION = "paper_iii_static_qse_campaign_v2"
STATIC_LEDGER_SCHEMA_VERSION = "paper_iii_static_qse_ledger_v2"
STATIC_ATTEMPT_SCHEMA_VERSION = "paper_iii_static_qse_attempt_v2"

# --- status vocabulary ------------------------------------------------------

STATUS_QUALIFIED = "qualified"
STATUS_CANDIDATE = "candidate"
STATUS_PENDING = "pending"
STATUS_GATE_PENDING = "gate_pending"
STATUS_NOT_PLANNED = "not_planned"
_QUALIFICATION_STATUSES = (STATUS_QUALIFIED, STATUS_CANDIDATE, STATUS_GATE_PENDING)

# --- public method identities ----------------------------------------------

METHOD_RECORD_FIRST = "record_first_adaptive_qse"
METHOD_DIRECT_DAVIDSON = "direct_davidson_qse"
METHOD_KRYLOV = "real_time_krylov"
METHOD_FIXED_CLASS = "fixed_linear_response_qse"
PUBLIC_METHODS = (
    METHOD_RECORD_FIRST,
    METHOD_DIRECT_DAVIDSON,
    METHOD_KRYLOV,
    METHOD_FIXED_CLASS,
)
ABLATION_METHODS = ("record_only", "early_switch")

# Unambiguous historical shorthands remain ingestion aliases.  The former
# ``record_first_adaptive_qse_v2`` id is deliberately absent: it named a
# selected-QSE-to-Davidson hybrid and must never normalize onto fixed-pool
# selected QSE.
_METHOD_ALIASES = {
    "adaptive_qse": METHOD_DIRECT_DAVIDSON,
    "krylov": METHOD_KRYLOV,
    "fixed_class": METHOD_FIXED_CLASS,
}
_QUARANTINED_METHOD_IDS = {
    "record_first_adaptive_qse_v2": "legacy selected-QSE-to-Davidson hybrid",
}

# Construction policies are scientific method identity, not incidental run
# settings.  A policy change must update this map and therefore produces new
# cell/action digests instead of reinterpreting an append-only ledger entry.
CONSTRUCTION_SELECTED_QSE_FIXED_POOL = "selected_qse_fixed_pool_v1"
CONSTRUCTION_DIRECT_DAVIDSON = (
    "direct_davidson_fixed_lr_plus_r_plus_1_low_diagonal_sector_v1"
)
CONSTRUCTION_REAL_TIME_KRYLOV = "real_time_krylov_first_order_trotter_v1"
CONSTRUCTION_FIXED_LINEAR_RESPONSE = "fixed_linear_response_complete_class_v1"
CONSTRUCTION_RECORD_ONLY_ABLATION = "selected_qse_record_only_ablation_v1"
CONSTRUCTION_SELECTED_QSE_DAVIDSON_HYBRID = (
    "selected_qse_then_davidson_explicit_ablation_v1"
)
DEFAULT_CONSTRUCTION_POLICY_IDS = MappingProxyType({
    METHOD_RECORD_FIRST: CONSTRUCTION_SELECTED_QSE_FIXED_POOL,
    METHOD_DIRECT_DAVIDSON: CONSTRUCTION_DIRECT_DAVIDSON,
    METHOD_KRYLOV: CONSTRUCTION_REAL_TIME_KRYLOV,
    METHOD_FIXED_CLASS: CONSTRUCTION_FIXED_LINEAR_RESPONSE,
    "record_only": CONSTRUCTION_RECORD_ONLY_ABLATION,
    "early_switch": CONSTRUCTION_SELECTED_QSE_DAVIDSON_HYBRID,
})

# --- stage graph ------------------------------------------------------------

STAGE_GROUND_STATE_IMPORT = "ground_state_import"
STAGE_STATIC_QSE_TRACE = "static_qse_trace"
STAGE_TRANSITIONS = "transitions"
STAGE_GREEN_FUNCTION = "green_function"
STAGE_RESOURCE_EVALUATION = "resource_evaluation"
STAGE_SPECTRAL_POSTPROCESSING = "spectral_postprocessing"
STAGE_QUALIFICATION = "qualification"

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    STAGE_GROUND_STATE_IMPORT: (),
    STAGE_STATIC_QSE_TRACE: (STAGE_GROUND_STATE_IMPORT,),
    STAGE_TRANSITIONS: (STAGE_STATIC_QSE_TRACE,),
    STAGE_GREEN_FUNCTION: (STAGE_STATIC_QSE_TRACE,),
    STAGE_RESOURCE_EVALUATION: (STAGE_STATIC_QSE_TRACE,),
    STAGE_SPECTRAL_POSTPROCESSING: (STAGE_TRANSITIONS, STAGE_GREEN_FUNCTION),
    STAGE_QUALIFICATION: (STAGE_SPECTRAL_POSTPROCESSING, STAGE_RESOURCE_EVALUATION),
}
STAGE_ORDER: tuple[str, ...] = (
    STAGE_GROUND_STATE_IMPORT,
    STAGE_STATIC_QSE_TRACE,
    STAGE_TRANSITIONS,
    STAGE_GREEN_FUNCTION,
    STAGE_RESOURCE_EVALUATION,
    STAGE_SPECTRAL_POSTPROCESSING,
    STAGE_QUALIFICATION,
)

DEFAULT_TARGET_ROOTS = tuple(range(1, 7))
DEFAULT_DIMENSION_CAPS = {
    METHOD_RECORD_FIRST: 64,
    METHOD_DIRECT_DAVIDSON: 64,
    METHOD_KRYLOV: 40,
    METHOD_FIXED_CLASS: 0,  # complete fixed class: no growth cap applies
}
DEFAULT_RESOURCE_POLICY_ID = "graph_span_two_qubit_only_v1_plus_pencil_estimator_v1"

HH_REGIMES: tuple[tuple[str, float, float], ...] = (
    ("weak_weak", 0.25, 0.353553390593),
    ("intermediate_weak", 1.25, 0.353553390593),
    ("strong_weak_u8", 8.0, 0.353553390593),
    ("weak_strong", 0.25, 0.790569415042),
    ("intermediate_strong", 1.25, 0.790569415042),
    ("strong_strong_u8", 8.0, 0.790569415042),
)
HH_CUTOFFS = (1, 3)


class StaticCampaignError(ValueError):
    """Raised on identity, planning, or ledger-admission violations."""


# --- canonical hashing ------------------------------------------------------


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_method_id(method_id: str) -> str:
    """Normalize legacy/ingestion aliases onto the stable public method ids."""

    raw = str(method_id).strip()
    if raw in _QUARANTINED_METHOD_IDS:
        raise StaticCampaignError(
            f"method id {raw!r} is quarantined as "
            f"{_QUARANTINED_METHOD_IDS[raw]}; it cannot be ingested as a "
            "canonical fixed-pool selected-QSE cell."
        )
    normalized = _METHOD_ALIASES.get(raw, raw)
    if normalized not in PUBLIC_METHODS and normalized not in ABLATION_METHODS:
        raise StaticCampaignError(
            f"unknown method id {method_id!r}; public methods are "
            f"{list(PUBLIC_METHODS)}, ablations are {list(ABLATION_METHODS)}."
        )
    return normalized


# --- identities -------------------------------------------------------------


@dataclass(frozen=True)
class StaticQSEPhysicsKey:
    """Immutable physics identity of one campaign case."""

    model_family: str
    case_id: str
    parameters: tuple[tuple[str, float], ...]
    register_layout: str
    sector: str
    cutoff_kind: str
    cutoff_value: int

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "static_qse_physics_key_v1",
            "model_family": str(self.model_family),
            "case_id": str(self.case_id),
            "parameters": {name: float(value) for name, value in self.parameters},
            "register_layout": str(self.register_layout),
            "sector": str(self.sector),
            "cutoff_kind": str(self.cutoff_kind),
            "cutoff_value": int(self.cutoff_value),
        }

    def digest(self) -> str:
        return canonical_digest(self.payload())


def hh_physics_key(regime: str, u: float, g_ep: float, n_ph_max: int) -> StaticQSEPhysicsKey:
    return StaticQSEPhysicsKey(
        model_family="hubbard_holstein_l2",
        case_id=str(regime),
        parameters=(("u", float(u)), ("g_ep", float(g_ep))),
        register_layout="jw_hh_l2_half_filled_v1",
        sector="half_filled_11",
        cutoff_kind="n_ph_max",
        cutoff_value=int(n_ph_max),
    )


def siam_physics_key_typed(case_id: str) -> StaticQSEPhysicsKey:
    from pipelines.qse_spectra.siam_static_adapter import siam_physics_key

    key = siam_physics_key(case_id)
    return StaticQSEPhysicsKey(
        model_family=str(key.model_family),
        case_id=str(key.case_id),
        parameters=(
            ("u", float(key.u)),
            ("v", float(key.v)),
            ("hopping", float(key.hopping)),
        ),
        register_layout=str(key.jw_order),
        sector=f"n_up_{key.n_up}_n_down_{key.n_down}",
        cutoff_kind="n_bath",
        cutoff_value=int(key.n_bath),
    )


@dataclass(frozen=True)
class GroundStateArtifactRef:
    """Provenance of one prepared ground-state artifact."""

    provider_id: str
    manifest_path: str
    manifest_sha256: str
    state_key: str
    state_payload_sha256: str
    num_qubits: int
    sector: str
    energy: float | None
    source_revision: str
    achieved_settings_digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "static_qse_ground_state_artifact_v1",
            "provider_id": str(self.provider_id),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": str(self.manifest_sha256),
            "state_key": str(self.state_key),
            "state_payload_sha256": str(self.state_payload_sha256),
            "num_qubits": int(self.num_qubits),
            "sector": str(self.sector),
            "energy": None if self.energy is None else float(self.energy),
            "source_revision": str(self.source_revision),
            "achieved_settings_digest": str(self.achieved_settings_digest),
        }

    @property
    def artifact_id(self) -> str:
        return canonical_digest(self.payload())


@dataclass(frozen=True)
class StaticQSECellKey:
    """One declared campaign cell. No accuracy target, no internal phase."""

    physics: StaticQSEPhysicsKey
    ground_state_artifact_id: str
    method_id: str
    target_roots: tuple[int, ...]
    dimension_cap: int
    pool_policy_digest: str
    spectral_request_digest: str
    resource_policy_id: str
    construction_policy_id: str = field(init=False)
    # The online switch rule is part of the identity: changing when the
    # method escalates to residual synthesis declares new cells rather than
    # reinterpreting admitted evidence under a rule it never ran.
    method_policy: MethodPolicy = CANONICAL_METHOD_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", normalize_method_id(self.method_id))
        expected_construction = DEFAULT_CONSTRUCTION_POLICY_IDS[self.method_id]
        object.__setattr__(self, "construction_policy_id", expected_construction)
        forbidden = {"record_phase", "davidson_phase", "phase"}
        if str(self.method_id).lower() in forbidden:  # pragma: no cover - typed guard
            raise StaticCampaignError("internal phases are not method identities.")
        if self.method_id in PUBLIC_METHODS and not self.method_policy.is_canonical:
            raise StaticCampaignError(
                f"public method {self.method_id!r} must run the canonical "
                f"exhaustion-only switch policy; trigger "
                f"{self.method_policy.switch_trigger!r} is an ablation until the "
                "canonical policy itself is changed."
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "static_qse_cell_key_v2",
            "physics": self.physics.payload(),
            "ground_state_artifact_id": str(self.ground_state_artifact_id),
            "method_id": str(self.method_id),
            "target_roots": [int(r) for r in self.target_roots],
            "dimension_cap": int(self.dimension_cap),
            "pool_policy_digest": str(self.pool_policy_digest),
            "spectral_request_digest": str(self.spectral_request_digest),
            "resource_policy_id": str(self.resource_policy_id),
            "construction_policy_id": str(self.construction_policy_id),
            "method_policy": self.method_policy.payload(),
            "method_policy_digest": self.method_policy.digest(),
        }

    def digest(self) -> str:
        return canonical_digest(self.payload())


@dataclass(frozen=True)
class StaticQSEAction:
    """One schedulable unit: a cell's stage plus its dependency action ids."""

    cell: StaticQSECellKey
    stage: str
    dependency_action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if str(self.stage) not in STAGE_DEPENDENCIES:
            raise StaticCampaignError(
                f"unknown stage {self.stage!r}; stages are {list(STAGE_ORDER)}."
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "static_qse_action_v2",
            "cell": self.cell.payload(),
            "stage": str(self.stage),
            "dependency_action_ids": list(self.dependency_action_ids),
        }

    @property
    def action_id(self) -> str:
        return canonical_digest(self.payload())


@dataclass(frozen=True)
class StaticQSEAttempt:
    """One recorded execution of an action."""

    action_id: str
    achieved_config: tuple[tuple[str, Any], ...]
    source_revision: str
    input_digests: tuple[tuple[str, str], ...]
    output_digests: tuple[tuple[str, str], ...]
    payload_paths: tuple[str, ...]
    disposition: str
    qualification: tuple[tuple[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if str(self.disposition) not in {"completed", "failed_process"}:
            raise StaticCampaignError(
                "attempt disposition must be 'completed' or 'failed_process'; "
                f"got {self.disposition!r}."
            )
        if self.qualification is not None:
            status = dict(self.qualification).get("status")
            if status not in _QUALIFICATION_STATUSES:
                raise StaticCampaignError(
                    f"qualification status must be one of {_QUALIFICATION_STATUSES}; "
                    f"got {status!r}."
                )

    @staticmethod
    def from_mappings(
        *,
        action_id: str,
        achieved_config: Mapping[str, Any],
        source_revision: str,
        input_digests: Mapping[str, str],
        output_digests: Mapping[str, str],
        payload_paths: Sequence[str],
        disposition: str,
        qualification: Mapping[str, Any] | None = None,
    ) -> "StaticQSEAttempt":
        return StaticQSEAttempt(
            action_id=str(action_id),
            achieved_config=tuple(sorted((str(k), v) for k, v in achieved_config.items())),
            source_revision=str(source_revision),
            input_digests=tuple(sorted((str(k), str(v)) for k, v in input_digests.items())),
            output_digests=tuple(sorted((str(k), str(v)) for k, v in output_digests.items())),
            payload_paths=tuple(str(p) for p in payload_paths),
            disposition=str(disposition),
            qualification=(
                None
                if qualification is None
                else tuple(sorted((str(k), v) for k, v in qualification.items()))
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": STATIC_ATTEMPT_SCHEMA_VERSION,
            "action_id": str(self.action_id),
            "achieved_config": {k: v for k, v in self.achieved_config},
            "source_revision": str(self.source_revision),
            "input_digests": {k: v for k, v in self.input_digests},
            "output_digests": {k: v for k, v in self.output_digests},
            "payload_paths": list(self.payload_paths),
            "disposition": str(self.disposition),
            "qualification": (
                None if self.qualification is None else {k: v for k, v in self.qualification}
            ),
        }

    def identity_payload(self) -> dict[str, Any]:
        """Everything that must match on replay (qualification may differ)."""

        payload = self.payload()
        payload.pop("qualification")
        return payload

    @property
    def qualification_status(self) -> str | None:
        if self.qualification is None:
            return None
        return str(dict(self.qualification).get("status"))


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    idempotent: bool
    reason: str
    action_id: str


@dataclass(frozen=True)
class NotPlannedCell:
    """A declared matrix slot that is intentionally not scheduled."""

    physics: StaticQSEPhysicsKey
    method_id: str
    reason: str

    def payload(self) -> dict[str, Any]:
        return {
            "physics": self.physics.payload(),
            "method_id": normalize_method_id(self.method_id),
            "reason": str(self.reason),
        }


# --- ledger -----------------------------------------------------------------


class StaticQSELedger:
    """Append-only attempt ledger with idempotent replay and hard conflicts."""

    def __init__(self, attempts: Iterable[StaticQSEAttempt] = ()) -> None:
        self._attempts: list[StaticQSEAttempt] = []
        for attempt in attempts:
            self.append(attempt)

    def __len__(self) -> int:
        return len(self._attempts)

    @property
    def attempts(self) -> tuple[StaticQSEAttempt, ...]:
        return tuple(self._attempts)

    def append(self, attempt: StaticQSEAttempt) -> AdmissionResult:
        if not isinstance(attempt, StaticQSEAttempt):
            raise StaticCampaignError("ledger accepts StaticQSEAttempt records only.")
        for existing in self._attempts:
            if existing.action_id != attempt.action_id:
                continue
            if existing.payload() == attempt.payload():
                return AdmissionResult(
                    admitted=False,
                    idempotent=True,
                    reason="exact_replay_noop",
                    action_id=attempt.action_id,
                )
            if existing.identity_payload() != attempt.identity_payload():
                raise StaticCampaignError(
                    "conflicting replay for action "
                    f"{attempt.action_id}: achieved configuration or output "
                    "digests differ from the admitted attempt."
                )
            # Same numerical identity, different qualification: qualification
            # is replayable over stored outputs, so re-classification appends.
        self._attempts.append(attempt)
        return AdmissionResult(
            admitted=True,
            idempotent=False,
            reason="admitted",
            action_id=attempt.action_id,
        )

    def latest(self, action_id: str) -> StaticQSEAttempt | None:
        for attempt in reversed(self._attempts):
            if attempt.action_id == str(action_id):
                return attempt
        return None

    def completed(self, action_id: str) -> StaticQSEAttempt | None:
        attempt = self.latest(action_id)
        if attempt is not None and attempt.disposition == "completed":
            return attempt
        return None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATIC_LEDGER_SCHEMA_VERSION,
            "attempt_count": len(self._attempts),
            "attempts": [attempt.payload() for attempt in self._attempts],
        }

    def digest(self) -> str:
        return canonical_digest(self.payload())

    def save(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.payload(), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> "StaticQSELedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != STATIC_LEDGER_SCHEMA_VERSION:
            raise StaticCampaignError(
                f"ledger schema mismatch at {path}: {payload.get('schema_version')!r}"
            )
        ledger = StaticQSELedger()
        for record in payload.get("attempts", []):
            if not isinstance(record, Mapping) or record.get("schema") != STATIC_ATTEMPT_SCHEMA_VERSION:
                raise StaticCampaignError(
                    "ledger contains an attempt with an unexpected schema."
                )
            ledger.append(
                StaticQSEAttempt.from_mappings(
                    action_id=record["action_id"],
                    achieved_config=record.get("achieved_config", {}),
                    source_revision=record.get("source_revision", "unknown"),
                    input_digests=record.get("input_digests", {}),
                    output_digests=record.get("output_digests", {}),
                    payload_paths=record.get("payload_paths", []),
                    disposition=record.get("disposition", "completed"),
                    qualification=record.get("qualification"),
                )
            )
        return ledger


# --- campaign ---------------------------------------------------------------


@dataclass(frozen=True)
class StaticQSECampaign:
    """The declared matrix: planned cells plus intentionally unplanned slots."""

    campaign_id: str
    cells: tuple[StaticQSECellKey, ...]
    not_planned: tuple[NotPlannedCell, ...] = ()

    def __post_init__(self) -> None:
        digests = [cell.digest() for cell in self.cells]
        if len(set(digests)) != len(digests):
            raise StaticCampaignError("declared campaign cells are not unique.")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATIC_CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": str(self.campaign_id),
            "cells": [cell.payload() for cell in self.cells],
            "not_planned": [slot.payload() for slot in self.not_planned],
        }

    def digest(self) -> str:
        return canonical_digest(self.payload())

    def cell_actions(self, cell: StaticQSECellKey) -> dict[str, StaticQSEAction]:
        """The cell's full stage graph, dependency ids resolved in topo order."""

        return build_cell_actions(cell)

    def all_actions(self) -> tuple[StaticQSEAction, ...]:
        """Every declared action, deterministically ordered.

        Order is (cell digest, stage topological index): independent of
        declaration order and of any directory listing.
        """

        ordered_cells = sorted(self.cells, key=lambda cell: cell.digest())
        out: list[StaticQSEAction] = []
        for cell in ordered_cells:
            actions = self.cell_actions(cell)
            out.extend(actions[stage] for stage in STAGE_ORDER)
        return tuple(out)

    def next_actions(self, ledger: StaticQSELedger) -> tuple[StaticQSEAction, ...]:
        """Ready actions: dependencies completed, no completed attempt yet.

        A numerically ``candidate`` dependency is still a completed
        dependency; only structural absence blocks the downstream stage.
        """

        ready: list[StaticQSEAction] = []
        for action in self.all_actions():
            if ledger.completed(action.action_id) is not None:
                continue
            if all(
                ledger.completed(dep) is not None
                for dep in action.dependency_action_ids
            ):
                ready.append(action)
        return tuple(ready)

    def action_status(self, ledger: StaticQSELedger, action: StaticQSEAction) -> str:
        attempt = ledger.completed(action.action_id)
        if attempt is None:
            return STATUS_PENDING
        status = attempt.qualification_status
        if status is None:
            return STATUS_GATE_PENDING
        return status

    def cell_stage_statuses(
        self, ledger: StaticQSELedger, cell: StaticQSECellKey
    ) -> dict[str, str]:
        actions = self.cell_actions(cell)
        return {
            stage: self.action_status(ledger, actions[stage]) for stage in STAGE_ORDER
        }


def build_cell_actions(cell: StaticQSECellKey) -> dict[str, StaticQSEAction]:
    """One cell's full stage graph, dependency ids resolved in topo order."""

    actions: dict[str, StaticQSEAction] = {}
    for stage in STAGE_ORDER:
        deps = tuple(
            actions[dep_stage].action_id for dep_stage in STAGE_DEPENDENCIES[stage]
        )
        actions[stage] = StaticQSEAction(
            cell=cell, stage=stage, dependency_action_ids=deps
        )
    return actions


# --- payload deserializers --------------------------------------------------


def physics_key_from_payload(payload: Mapping[str, Any]) -> StaticQSEPhysicsKey:
    if payload.get("schema") != "static_qse_physics_key_v1":
        raise StaticCampaignError(
            f"unexpected physics-key schema {payload.get('schema')!r}."
        )
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise StaticCampaignError("physics-key payload requires a parameters mapping.")
    return StaticQSEPhysicsKey(
        model_family=str(payload["model_family"]),
        case_id=str(payload["case_id"]),
        parameters=tuple(sorted((str(k), float(v)) for k, v in parameters.items())),
        register_layout=str(payload["register_layout"]),
        sector=str(payload["sector"]),
        cutoff_kind=str(payload["cutoff_kind"]),
        cutoff_value=int(payload["cutoff_value"]),
    )


def cell_key_from_payload(payload: Mapping[str, Any]) -> StaticQSECellKey:
    if payload.get("schema") != "static_qse_cell_key_v2":
        raise StaticCampaignError(
            f"unexpected cell-key schema {payload.get('schema')!r}."
        )
    raw_method_policy = payload.get("method_policy")
    if not isinstance(raw_method_policy, Mapping):
        raise StaticCampaignError("cell-key payload requires method_policy.")
    method_policy = MethodPolicy.from_mapping(raw_method_policy)
    declared_policy_digest = str(payload.get("method_policy_digest") or "")
    if declared_policy_digest != method_policy.digest():
        raise StaticCampaignError(
            "cell-key method_policy_digest does not match method_policy."
        )
    method_id = str(payload["method_id"])
    normalized_method = normalize_method_id(method_id)
    declared_construction = str(payload.get("construction_policy_id") or "")
    expected_construction = DEFAULT_CONSTRUCTION_POLICY_IDS[normalized_method]
    if declared_construction != expected_construction:
        raise StaticCampaignError(
            "cell-key construction_policy_id does not match the declared method."
        )
    return StaticQSECellKey(
        physics=physics_key_from_payload(payload["physics"]),
        ground_state_artifact_id=str(payload["ground_state_artifact_id"]),
        method_id=method_id,
        target_roots=tuple(int(r) for r in payload["target_roots"]),
        dimension_cap=int(payload["dimension_cap"]),
        pool_policy_digest=str(payload["pool_policy_digest"]),
        spectral_request_digest=str(payload["spectral_request_digest"]),
        resource_policy_id=str(payload["resource_policy_id"]),
        method_policy=method_policy,
    )


def action_from_payload(payload: Mapping[str, Any]) -> StaticQSEAction:
    if payload.get("schema") != "static_qse_action_v2":
        raise StaticCampaignError(
            f"unexpected action schema {payload.get('schema')!r}."
        )
    return StaticQSEAction(
        cell=cell_key_from_payload(payload["cell"]),
        stage=str(payload["stage"]),
        dependency_action_ids=tuple(
            str(x) for x in payload.get("dependency_action_ids", [])
        ),
    )


# --- declared matrices ------------------------------------------------------


def declare_hh_matrix(
    *,
    ground_state_artifacts: Mapping[str, GroundStateArtifactRef],
    pool_policy_digests: Mapping[str, str],
    spectral_request_digest: str,
    target_roots: Sequence[int] = DEFAULT_TARGET_ROOTS,
    dimension_caps: Mapping[str, int] | None = None,
    resource_policy_id: str = DEFAULT_RESOURCE_POLICY_ID,
    cutoffs: Sequence[int] = HH_CUTOFFS,
) -> tuple[StaticQSECellKey, ...]:
    """The first complete Hubbard--Holstein matrix.

    ``ground_state_artifacts`` and ``pool_policy_digests`` are keyed by
    ``f"{regime}_nph{n}"``. Every declared key must be present: the campaign
    is a declared matrix, so a missing artifact is a declaration error, not a
    silent hole.
    """

    caps = dict(DEFAULT_DIMENSION_CAPS)
    if dimension_caps:
        caps.update({normalize_method_id(k): int(v) for k, v in dimension_caps.items()})
    cells: list[StaticQSECellKey] = []
    for n_ph_max in cutoffs:
        for regime, u, g_ep in HH_REGIMES:
            slot = f"{regime}_nph{int(n_ph_max)}"
            if slot not in ground_state_artifacts:
                raise StaticCampaignError(f"missing ground-state artifact for {slot!r}.")
            if slot not in pool_policy_digests:
                raise StaticCampaignError(f"missing pool policy digest for {slot!r}.")
            physics = hh_physics_key(regime, u, g_ep, int(n_ph_max))
            for method in PUBLIC_METHODS:
                cells.append(
                    StaticQSECellKey(
                        physics=physics,
                        ground_state_artifact_id=ground_state_artifacts[slot].artifact_id,
                        method_id=method,
                        target_roots=tuple(int(r) for r in target_roots),
                        dimension_cap=int(caps[method]),
                        pool_policy_digest=str(pool_policy_digests[slot]),
                        spectral_request_digest=str(spectral_request_digest),
                        resource_policy_id=str(resource_policy_id),
                    )
                )
    return tuple(cells)


def declare_siam_matrix(
    *,
    ground_state_artifacts: Mapping[str, GroundStateArtifactRef],
    pool_policy_digests: Mapping[str, str],
    spectral_request_digest: str,
    target_roots: Sequence[int] = DEFAULT_TARGET_ROOTS,
    dimension_caps: Mapping[str, int] | None = None,
    resource_policy_id: str = DEFAULT_RESOURCE_POLICY_ID,
    include_krylov: bool = False,
    cases: Sequence[str] = ("siam_nb2_fixture", "siam_nb4_pilot"),
) -> tuple[tuple[StaticQSECellKey, ...], tuple[NotPlannedCell, ...]]:
    """SIAM matrix: record-first vs direct Davidson (Krylov only if faithful).

    The fixed linear-response slot is ``not_planned``: no scientifically
    defined fixed class exists for SIAM, and fabricating one merely to make
    the matrix rectangular is forbidden.
    """

    caps = dict(DEFAULT_DIMENSION_CAPS)
    if dimension_caps:
        caps.update({normalize_method_id(k): int(v) for k, v in dimension_caps.items()})
    methods = [METHOD_RECORD_FIRST, METHOD_DIRECT_DAVIDSON]
    if include_krylov:
        methods.append(METHOD_KRYLOV)
    cells: list[StaticQSECellKey] = []
    not_planned: list[NotPlannedCell] = []
    for case_id in cases:
        physics = siam_physics_key_typed(str(case_id))
        if case_id not in ground_state_artifacts:
            raise StaticCampaignError(f"missing ground-state artifact for {case_id!r}.")
        if case_id not in pool_policy_digests:
            raise StaticCampaignError(f"missing pool policy digest for {case_id!r}.")
        for method in methods:
            cells.append(
                StaticQSECellKey(
                    physics=physics,
                    ground_state_artifact_id=ground_state_artifacts[case_id].artifact_id,
                    method_id=method,
                    target_roots=tuple(int(r) for r in target_roots),
                    dimension_cap=int(caps[method]),
                    pool_policy_digest=str(pool_policy_digests[case_id]),
                    spectral_request_digest=str(spectral_request_digest),
                    resource_policy_id=str(resource_policy_id),
                )
            )
        not_planned.append(
            NotPlannedCell(
                physics=physics,
                method_id=METHOD_FIXED_CLASS,
                reason=(
                    "no scientifically defined fixed linear-response class exists "
                    "for SIAM; slot stays not_planned until one is declared"
                ),
            )
        )
        if not include_krylov:
            not_planned.append(
                NotPlannedCell(
                    physics=physics,
                    method_id=METHOD_KRYLOV,
                    reason=(
                        "real-time Krylov joins only when the generic adapter is "
                        "faithful without SIAM-specific policy"
                    ),
                )
            )
    return tuple(cells), tuple(not_planned)


__all__ = [
    "ABLATION_METHODS",
    "CANONICAL_METHOD_POLICY",
    "CONSTRUCTION_DIRECT_DAVIDSON",
    "CONSTRUCTION_FIXED_LINEAR_RESPONSE",
    "CONSTRUCTION_REAL_TIME_KRYLOV",
    "CONSTRUCTION_RECORD_ONLY_ABLATION",
    "CONSTRUCTION_SELECTED_QSE_DAVIDSON_HYBRID",
    "CONSTRUCTION_SELECTED_QSE_FIXED_POOL",
    "AdmissionResult",
    "DEFAULT_DIMENSION_CAPS",
    "DEFAULT_CONSTRUCTION_POLICY_IDS",
    "DEFAULT_RESOURCE_POLICY_ID",
    "DEFAULT_TARGET_ROOTS",
    "GroundStateArtifactRef",
    "HH_CUTOFFS",
    "HH_REGIMES",
    "METHOD_DIRECT_DAVIDSON",
    "METHOD_FIXED_CLASS",
    "METHOD_KRYLOV",
    "METHOD_RECORD_FIRST",
    "MethodPolicy",
    "NotPlannedCell",
    "PUBLIC_METHODS",
    "STAGE_DEPENDENCIES",
    "STAGE_GREEN_FUNCTION",
    "STAGE_GROUND_STATE_IMPORT",
    "STAGE_ORDER",
    "STAGE_QUALIFICATION",
    "STAGE_RESOURCE_EVALUATION",
    "STAGE_SPECTRAL_POSTPROCESSING",
    "STAGE_STATIC_QSE_TRACE",
    "STAGE_TRANSITIONS",
    "STATIC_ATTEMPT_SCHEMA_VERSION",
    "STATIC_CAMPAIGN_SCHEMA_VERSION",
    "STATIC_LEDGER_SCHEMA_VERSION",
    "STATUS_CANDIDATE",
    "STATUS_GATE_PENDING",
    "STATUS_NOT_PLANNED",
    "STATUS_PENDING",
    "STATUS_QUALIFIED",
    "StaticCampaignError",
    "StaticQSEAction",
    "StaticQSEAttempt",
    "StaticQSECampaign",
    "StaticQSECellKey",
    "StaticQSELedger",
    "StaticQSEPhysicsKey",
    "action_from_payload",
    "build_cell_actions",
    "canonical_digest",
    "canonical_json",
    "cell_key_from_payload",
    "declare_hh_matrix",
    "declare_siam_matrix",
    "hh_physics_key",
    "normalize_method_id",
    "physics_key_from_payload",
    "siam_physics_key_typed",
]
