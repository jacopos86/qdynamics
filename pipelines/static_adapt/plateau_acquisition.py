"""Pure helpers for Route C plateau-acquisition state.

This module is intentionally orchestration-free: it normalizes Route-C plateau
configuration, builds candidate-position duplicate keys, maps dormant logical
indices across insertions, and serializes small state payloads.  The main ADAPT
loop will consume these helpers in a later integration pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

PLATEAU_ACQUISITION_SCHEMA = "route_c_plateau_acquisition_v1"
PLATEAU_ACQUISITION_MODE_OFF = "off"
PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1 = "novelty_cost_v1"
PLATEAU_ACQUISITION_MODE_CHOICES = (
    PLATEAU_ACQUISITION_MODE_OFF,
    PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1,
)

PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1 = "block_exact_position_v1"
PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY = "allow_exact_position_replay"
PLATEAU_DUPLICATE_POLICY_CHOICES = (
    PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
    PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY,
)

PLATEAU_SEED_PROBE_MODE_OFF = "off"
PLATEAU_SEED_PROBE_MODE_DORMANT_NEW_RANDOM_V1 = "dormant_new_random_v1"
PLATEAU_SEED_PROBE_MODE_CHOICES = (
    PLATEAU_SEED_PROBE_MODE_OFF,
    PLATEAU_SEED_PROBE_MODE_DORMANT_NEW_RANDOM_V1,
)

PLATEAU_TRIAL_OPTIMIZER_INHERIT = "inherit"
PLATEAU_TRIAL_OPTIMIZER_SPSA = "spsa"
PLATEAU_TRIAL_OPTIMIZER_SP_QNGD = "sp_qngd"
PLATEAU_TRIAL_OPTIMIZER_CHOICES = (
    PLATEAU_TRIAL_OPTIMIZER_INHERIT,
    PLATEAU_TRIAL_OPTIMIZER_SPSA,
    PLATEAU_TRIAL_OPTIMIZER_SP_QNGD,
)

PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1 = "fractional_residual_v1"
PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1 = "log_volume_v1"
PLATEAU_ACQUISITION_SCORE_CHOICES = (
    PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
    PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
)

PHASE3_PLATEAU_FRACTIONAL_SCORE_FORMULA = "N3_plat / (1 + K3)"
PHASE3_PLATEAU_LOG_VOLUME_SCORE_FORMULA = "log(1 + sigma_perp_lambda / lambda_vol) / (1 + K3)"
PHASE3_PLATEAU_SCORE_FORMULA = PHASE3_PLATEAU_LOG_VOLUME_SCORE_FORMULA


class PlateauAcquisitionError(ValueError):
    """Raised when Route-C plateau helper inputs violate pure invariants."""


@dataclass(frozen=True)
class PlateauAcquisitionConfig:
    """Normalized plateau-acquisition config surface."""

    schema: str = PLATEAU_ACQUISITION_SCHEMA
    mode: str = PLATEAU_ACQUISITION_MODE_OFF
    acquisition_score: str = PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1
    unlock_margin: float = 1e-8
    duplicate_policy: str = PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1
    lambda_vol: float = 1e-8
    sigma_min: float = 0.0
    nu_min: float = 0.0
    volume_min: float = 0.0
    failed_family_patience: int = 0
    trial_optimizer: str = PLATEAU_TRIAL_OPTIMIZER_INHERIT
    trial_qngd_maxiter: int = 64

    @property
    def enabled(self) -> bool:
        return self.mode == PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "mode": str(self.mode),
            "enabled": bool(self.enabled),
            "acquisition_score": str(self.acquisition_score),
            "unlock_margin": float(self.unlock_margin),
            "duplicate_policy": str(self.duplicate_policy),
            "lambda_vol": float(self.lambda_vol),
            "sigma_min": float(self.sigma_min),
            "nu_min": float(self.nu_min),
            "volume_min": float(self.volume_min),
            "failed_family_patience": int(self.failed_family_patience),
            "trial_optimizer": str(self.trial_optimizer),
            "trial_qngd_maxiter": int(self.trial_qngd_maxiter),
            "score_formula": plateau_score_formula(self.acquisition_score),
        }


@dataclass(frozen=True)
class PlateauCandidateKey:
    """Stable duplicate key for a candidate-position record."""

    candidate_identity: str
    position_id: int

    def as_tuple(self) -> tuple[str, int]:
        return (str(self.candidate_identity), int(self.position_id))

    def as_string(self) -> str:
        return f"{self.candidate_identity}@position:{int(self.position_id)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_identity": str(self.candidate_identity),
            "position_id": int(self.position_id),
            "key": self.as_string(),
        }


@dataclass(frozen=True)
class PlateauDormantRecord:
    """A logically admitted zero-amplitude plateau record."""

    candidate_key: PlateauCandidateKey
    logical_index: int
    candidate_label: str | None = None
    generator_id: str | None = None
    position_id: int | None = None
    admission_step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key.as_dict(),
            "logical_index": int(self.logical_index),
            "candidate_label": self.candidate_label,
            "generator_id": self.generator_id,
            "position_id": None if self.position_id is None else int(self.position_id),
            "admission_step": None if self.admission_step is None else int(self.admission_step),
            "payload": _json_safe(self.payload),
        }


@dataclass(frozen=True)
class PlateauAcquisitionState:
    """Route-C plateau acquisition state, independent of optimizer internals."""

    schema: str = PLATEAU_ACQUISITION_SCHEMA
    active_episode: bool = False
    dormant_records: tuple[PlateauDormantRecord, ...] = ()
    acquired_candidate_keys: tuple[PlateauCandidateKey, ...] = ()
    failed_unlock_count: int = 0
    unlock_count: int = 0
    last_event: dict[str, Any] | None = None

    def acquired_key_set(self) -> set[tuple[str, int]]:
        return {key.as_tuple() for key in self.acquired_candidate_keys}

    def dormant_logical_indices(self) -> tuple[int, ...]:
        return tuple(int(record.logical_index) for record in self.dormant_records)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "active_episode": bool(self.active_episode),
            "dormant_records": [record.as_dict() for record in self.dormant_records],
            "dormant_logical_indices": [int(x) for x in self.dormant_logical_indices()],
            "dormant_count": int(len(self.dormant_records)),
            "acquired_candidate_keys": [key.as_dict() for key in self.acquired_candidate_keys],
            "failed_unlock_count": int(self.failed_unlock_count),
            "unlock_count": int(self.unlock_count),
            "last_event": None if self.last_event is None else _json_safe(self.last_event),
        }


def normalize_plateau_acquisition_mode(value: Any, *, default: str = PLATEAU_ACQUISITION_MODE_OFF) -> str:
    key = str(default if value is None or value == "" else value).strip().lower().replace("-", "_")
    aliases = {
        "none": PLATEAU_ACQUISITION_MODE_OFF,
        "false": PLATEAU_ACQUISITION_MODE_OFF,
        "0": PLATEAU_ACQUISITION_MODE_OFF,
        "disabled": PLATEAU_ACQUISITION_MODE_OFF,
        "novelty_cost": PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1,
        "route_c": PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1,
        "plateau": PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1,
        "plateau_v1": PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1,
    }
    key = aliases.get(key, key)
    if key not in PLATEAU_ACQUISITION_MODE_CHOICES:
        raise PlateauAcquisitionError(
            "phase3_plateau_acquisition_mode must be one of "
            f"{PLATEAU_ACQUISITION_MODE_CHOICES}; got {value!r}."
        )
    return str(key)


def normalize_plateau_acquisition_score(
    value: Any,
    *,
    default: str = PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
) -> str:
    key = str(default if value is None or value == "" else value).strip().lower().replace("-", "_")
    aliases = {
        "volume": PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        "log_volume": PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        "logdet": PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        "log_det": PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        "geo": PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        "geo_volume": PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        "novelty": PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
        "novelty_cost": PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
        "fractional": PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
        "fractional_residual": PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
        "n3_plat": PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
    }
    key = aliases.get(key, key)
    if key not in PLATEAU_ACQUISITION_SCORE_CHOICES:
        raise PlateauAcquisitionError(
            "phase3_plateau_acquisition_score must be one of "
            f"{PLATEAU_ACQUISITION_SCORE_CHOICES}; got {value!r}."
        )
    return str(key)


def plateau_score_formula(acquisition_score: Any) -> str:
    score_key = normalize_plateau_acquisition_score(acquisition_score)
    if score_key == PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1:
        return PHASE3_PLATEAU_FRACTIONAL_SCORE_FORMULA
    return PHASE3_PLATEAU_LOG_VOLUME_SCORE_FORMULA


def normalize_plateau_duplicate_policy(
    value: Any,
    *,
    default: str = PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
) -> str:
    key = str(default if value is None or value == "" else value).strip().lower().replace("-", "_")
    aliases = {
        "block": PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
        "block_exact": PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
        "block_exact_position": PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
        "disallow_exact_position": PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
        "off": PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
        "allow": PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY,
        "allow_replay": PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY,
        "allow_exact_position": PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY,
        "diagnostic_replay": PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY,
    }
    key = aliases.get(key, key)
    if key not in PLATEAU_DUPLICATE_POLICY_CHOICES:
        raise PlateauAcquisitionError(
            "phase3_plateau_duplicate_policy must be one of "
            f"{PLATEAU_DUPLICATE_POLICY_CHOICES}; got {value!r}."
        )
    return str(key)


def normalize_plateau_trial_optimizer(
    value: Any,
    *,
    default: str = PLATEAU_TRIAL_OPTIMIZER_INHERIT,
) -> str:
    key = str(default if value is None or value == "" else value).strip().lower().replace("-", "_")
    aliases = {
        "default": PLATEAU_TRIAL_OPTIMIZER_INHERIT,
        "normal": PLATEAU_TRIAL_OPTIMIZER_INHERIT,
        "adapt": PLATEAU_TRIAL_OPTIMIZER_INHERIT,
        "route_a": PLATEAU_TRIAL_OPTIMIZER_INHERIT,
        "spsa_v1": PLATEAU_TRIAL_OPTIMIZER_SPSA,
        "qngd": PLATEAU_TRIAL_OPTIMIZER_SP_QNGD,
        "spqngd": PLATEAU_TRIAL_OPTIMIZER_SP_QNGD,
        "sp_qngd_v1": PLATEAU_TRIAL_OPTIMIZER_SP_QNGD,
        "state_prepared_qngd": PLATEAU_TRIAL_OPTIMIZER_SP_QNGD,
        "state_prepared": PLATEAU_TRIAL_OPTIMIZER_SP_QNGD,
    }
    key = aliases.get(key, key)
    if key not in PLATEAU_TRIAL_OPTIMIZER_CHOICES:
        raise PlateauAcquisitionError(
            "phase3_plateau_trial_optimizer must be one of "
            f"{PLATEAU_TRIAL_OPTIMIZER_CHOICES}; got {value!r}."
        )
    return str(key)


def normalize_plateau_acquisition_config(
    *,
    mode: Any = PLATEAU_ACQUISITION_MODE_OFF,
    acquisition_score: Any = PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
    unlock_margin: Any = 1e-8,
    duplicate_policy: Any = PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
    lambda_vol: Any = 1e-8,
    sigma_min: Any = 0.0,
    nu_min: Any = 0.0,
    volume_min: Any = 0.0,
    failed_family_patience: Any = 0,
    trial_optimizer: Any = PLATEAU_TRIAL_OPTIMIZER_INHERIT,
    trial_qngd_maxiter: Any = 64,
) -> PlateauAcquisitionConfig:
    mode_key = normalize_plateau_acquisition_mode(mode)
    score_key = normalize_plateau_acquisition_score(acquisition_score)
    duplicate_key = normalize_plateau_duplicate_policy(duplicate_policy)
    trial_optimizer_key = normalize_plateau_trial_optimizer(trial_optimizer)
    try:
        unlock_margin_val = float(unlock_margin)
    except (TypeError, ValueError) as exc:
        raise PlateauAcquisitionError("phase3_plateau_unlock_margin must be finite and nonnegative.") from exc
    if (not math.isfinite(unlock_margin_val)) or unlock_margin_val < 0.0:
        raise PlateauAcquisitionError("phase3_plateau_unlock_margin must be finite and nonnegative.")
    try:
        lambda_vol_val = float(lambda_vol)
    except (TypeError, ValueError) as exc:
        raise PlateauAcquisitionError("phase3_plateau_lambda_vol must be finite and positive.") from exc
    if (not math.isfinite(lambda_vol_val)) or lambda_vol_val <= 0.0:
        raise PlateauAcquisitionError("phase3_plateau_lambda_vol must be finite and positive.")
    try:
        sigma_min_val = float(sigma_min)
        nu_min_val = float(nu_min)
        volume_min_val = float(volume_min)
    except (TypeError, ValueError) as exc:
        raise PlateauAcquisitionError(
            "phase3_plateau_sigma_min, phase3_plateau_nu_min, and "
            "phase3_plateau_volume_min must be finite and nonnegative."
        ) from exc
    if (
        (not math.isfinite(sigma_min_val))
        or (not math.isfinite(nu_min_val))
        or (not math.isfinite(volume_min_val))
        or sigma_min_val < 0.0
        or nu_min_val < 0.0
        or volume_min_val < 0.0
    ):
        raise PlateauAcquisitionError(
            "phase3_plateau_sigma_min, phase3_plateau_nu_min, and "
            "phase3_plateau_volume_min must be finite and nonnegative."
        )
    try:
        failed_family_patience_val = int(failed_family_patience)
    except (TypeError, ValueError) as exc:
        raise PlateauAcquisitionError(
            "phase3_plateau_failed_family_patience must be a nonnegative integer."
        ) from exc
    if failed_family_patience_val < 0:
        raise PlateauAcquisitionError(
            "phase3_plateau_failed_family_patience must be a nonnegative integer."
        )
    try:
        trial_qngd_maxiter_val = int(trial_qngd_maxiter)
    except (TypeError, ValueError) as exc:
        raise PlateauAcquisitionError(
            "phase3_plateau_trial_qngd_maxiter must be a nonnegative integer."
        ) from exc
    if trial_qngd_maxiter_val < 0:
        raise PlateauAcquisitionError(
            "phase3_plateau_trial_qngd_maxiter must be a nonnegative integer."
        )
    return PlateauAcquisitionConfig(
        mode=str(mode_key),
        acquisition_score=str(score_key),
        unlock_margin=float(unlock_margin_val),
        duplicate_policy=str(duplicate_key),
        lambda_vol=float(lambda_vol_val),
        sigma_min=float(sigma_min_val),
        nu_min=float(nu_min_val),
        volume_min=float(volume_min_val),
        failed_family_patience=int(failed_family_patience_val),
        trial_optimizer=str(trial_optimizer_key),
        trial_qngd_maxiter=int(trial_qngd_maxiter_val),
    )


def candidate_key_from_parts(*, candidate_identity: Any, position_id: Any) -> PlateauCandidateKey:
    identity = str(candidate_identity if candidate_identity is not None else "").strip()
    if identity == "":
        raise PlateauAcquisitionError("candidate identity is required for a plateau duplicate key.")
    try:
        pos = int(position_id)
    except (TypeError, ValueError) as exc:
        raise PlateauAcquisitionError("position_id must be an integer for a plateau duplicate key.") from exc
    if pos < 0:
        raise PlateauAcquisitionError("position_id must be nonnegative for a plateau duplicate key.")
    return PlateauCandidateKey(candidate_identity=identity, position_id=pos)


def candidate_key_from_record(record: Any) -> PlateauCandidateKey:
    """Extract a duplicate key from a mapping or dataclass-like feature record."""

    if isinstance(record, Mapping):
        feat = record.get("feature")
        generator_id = record.get("generator_id")
        candidate_label = record.get("candidate_label")
        position_id = record.get("position_id")
        if feat is not None:
            generator_id = generator_id if generator_id not in {None, ""} else getattr(feat, "generator_id", None)
            candidate_label = candidate_label if candidate_label not in {None, ""} else getattr(feat, "candidate_label", None)
            position_id = position_id if position_id is not None else getattr(feat, "position_id", None)
    else:
        generator_id = getattr(record, "generator_id", None)
        candidate_label = getattr(record, "candidate_label", None)
        position_id = getattr(record, "position_id", None)
    identity = generator_id if generator_id not in {None, ""} else candidate_label
    return candidate_key_from_parts(candidate_identity=identity, position_id=position_id)


def duplicate_status(
    state: PlateauAcquisitionState,
    candidate_key: PlateauCandidateKey,
    *,
    duplicate_policy: str = PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
) -> dict[str, Any]:
    policy = normalize_plateau_duplicate_policy(duplicate_policy)
    duplicate = candidate_key.as_tuple() in state.acquired_key_set()
    blocked = bool(duplicate and policy == PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1)
    return {
        "schema": PLATEAU_ACQUISITION_SCHEMA,
        "candidate_key": candidate_key.as_dict(),
        "duplicate_policy": str(policy),
        "duplicate": bool(duplicate),
        "blocked": bool(blocked),
        "block_reason": "exact_candidate_position_duplicate" if blocked else None,
    }


def _event_candidate_identity(event: Mapping[str, Any]) -> str | None:
    """Extract the generator/family identity used for plateau failed-unlock backoff."""

    for raw in (
        event.get("candidate_key"),
        event.get("selected_record", {}).get("candidate_key")
        if isinstance(event.get("selected_record"), Mapping)
        else None,
    ):
        if isinstance(raw, Mapping):
            identity = raw.get("candidate_identity")
            if identity not in {None, ""}:
                return str(identity)
    for field in ("generator_id", "candidate_label"):
        value = event.get(field)
        if value not in {None, ""}:
            return str(value)
    selected = event.get("selected_record")
    if isinstance(selected, Mapping):
        for field in ("generator_id", "candidate_label"):
            value = selected.get(field)
            if value not in {None, ""}:
                return str(value)
    return None


def failed_family_backoff_status(
    events: Sequence[Mapping[str, Any]] | None,
    candidate_key: PlateauCandidateKey,
    *,
    patience: int = 0,
) -> dict[str, Any]:
    """Return same-generator failed-unlock backoff state for a plateau candidate.

    Exact-position duplicate blocking prevents re-buying the same record.  This
    helper prevents the next observed failure mode: re-buying the same generator
    identity at many insertion positions after repeated failed unlocks.  Counts
    reset after the most recent successful unlock so the rule is local to the
    current plateau episode.
    """

    patience_val = int(patience)
    identity = str(candidate_key.candidate_identity)
    count = 0
    if patience_val > 0 and events is not None:
        for raw_event in reversed(list(events)):
            if not isinstance(raw_event, Mapping):
                continue
            event_name = str(raw_event.get("event", ""))
            if event_name == "successful_unlock":
                break
            if event_name != "failed_unlock_dormant_admission":
                continue
            if _event_candidate_identity(raw_event) == identity:
                count += 1
    blocked = bool(patience_val > 0 and count >= patience_val)
    return {
        "schema": PLATEAU_ACQUISITION_SCHEMA,
        "candidate_key": candidate_key.as_dict(),
        "candidate_identity": identity,
        "failed_family_patience": int(patience_val),
        "failed_family_count": int(count),
        "blocked": bool(blocked),
        "block_reason": "failed_family_backoff" if blocked else None,
    }


def remap_logical_index_after_insertion(logical_index: int, insertion_position: int) -> int:
    idx = int(logical_index)
    pos = int(insertion_position)
    if idx < 0:
        raise PlateauAcquisitionError("logical_index must be nonnegative.")
    if pos < 0:
        raise PlateauAcquisitionError("insertion_position must be nonnegative.")
    return int(idx if idx < pos else idx + 1)


def remap_logical_indices_after_insertion(indices: Sequence[int], insertion_position: int) -> tuple[int, ...]:
    return tuple(remap_logical_index_after_insertion(int(idx), int(insertion_position)) for idx in indices)


def remap_state_after_insertion(
    state: PlateauAcquisitionState,
    *,
    insertion_position: int,
) -> PlateauAcquisitionState:
    remapped_records = tuple(
        PlateauDormantRecord(
            candidate_key=record.candidate_key,
            logical_index=remap_logical_index_after_insertion(record.logical_index, insertion_position),
            candidate_label=record.candidate_label,
            generator_id=record.generator_id,
            position_id=record.position_id,
            admission_step=record.admission_step,
            payload=dict(record.payload),
        )
        for record in state.dormant_records
    )
    return PlateauAcquisitionState(
        active_episode=bool(state.active_episode),
        dormant_records=remapped_records,
        acquired_candidate_keys=tuple(state.acquired_candidate_keys),
        failed_unlock_count=int(state.failed_unlock_count),
        unlock_count=int(state.unlock_count),
        last_event=state.last_event,
    )


def admit_failed_unlock_dormant(
    state: PlateauAcquisitionState,
    *,
    candidate_key: PlateauCandidateKey,
    insertion_position: int,
    candidate_label: str | None = None,
    generator_id: str | None = None,
    admission_step: int | None = None,
    duplicate_policy: str = PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
    event_payload: Mapping[str, Any] | None = None,
) -> PlateauAcquisitionState:
    """Commit a failed unlock as a zero-dormant logical record.

    Existing dormant logical indices are interpreted in old-pre space and are
    remapped through the insertion.  The newly inserted candidate occupies the
    insertion position in the post-insertion logical scaffold.
    """

    status = duplicate_status(state, candidate_key, duplicate_policy=duplicate_policy)
    if bool(status["blocked"]):
        raise PlateauAcquisitionError("exact candidate-position duplicate is blocked by plateau policy.")
    remapped = remap_state_after_insertion(state, insertion_position=int(insertion_position))
    keys = tuple(remapped.acquired_candidate_keys) + (candidate_key,)
    new_record = PlateauDormantRecord(
        candidate_key=candidate_key,
        logical_index=int(insertion_position),
        candidate_label=candidate_label,
        generator_id=generator_id,
        position_id=int(candidate_key.position_id),
        admission_step=None if admission_step is None else int(admission_step),
        payload=dict(event_payload or {}),
    )
    last_event = {
        "event": "failed_unlock_dormant_admission",
        "candidate_key": candidate_key.as_dict(),
        "logical_index": int(insertion_position),
        "duplicate_status": status,
    }
    if event_payload:
        last_event.update(_json_safe(dict(event_payload)))
    return PlateauAcquisitionState(
        active_episode=True,
        dormant_records=tuple(remapped.dormant_records) + (new_record,),
        acquired_candidate_keys=keys,
        failed_unlock_count=int(remapped.failed_unlock_count) + 1,
        unlock_count=int(remapped.unlock_count),
        last_event=last_event,
    )


def record_successful_unlock(
    state: PlateauAcquisitionState,
    *,
    event_payload: Mapping[str, Any] | None = None,
    remaining_dormant_records: Sequence[PlateauDormantRecord] | None = None,
) -> PlateauAcquisitionState:
    remaining = tuple(remaining_dormant_records or ())
    last_event = {"event": "successful_unlock", "remaining_dormant_count": int(len(remaining))}
    if event_payload:
        last_event.update(_json_safe(dict(event_payload)))
    return PlateauAcquisitionState(
        active_episode=bool(len(remaining) > 0),
        dormant_records=remaining,
        acquired_candidate_keys=tuple(state.acquired_candidate_keys),
        failed_unlock_count=int(state.failed_unlock_count),
        unlock_count=int(state.unlock_count) + 1,
        last_event=last_event,
    )


def plateau_state_from_payload(payload: Mapping[str, Any] | None) -> PlateauAcquisitionState:
    if payload is None:
        return PlateauAcquisitionState()
    if not isinstance(payload, Mapping):
        raise PlateauAcquisitionError("plateau acquisition state payload must be a mapping.")
    dormant_records: list[PlateauDormantRecord] = []
    for raw in payload.get("dormant_records", ()) or ():
        if not isinstance(raw, Mapping):
            raise PlateauAcquisitionError("dormant record payloads must be mappings.")
        key_payload = raw.get("candidate_key", {})
        if not isinstance(key_payload, Mapping):
            raise PlateauAcquisitionError("dormant candidate_key payload must be a mapping.")
        key = candidate_key_from_parts(
            candidate_identity=key_payload.get("candidate_identity"),
            position_id=key_payload.get("position_id"),
        )
        dormant_records.append(
            PlateauDormantRecord(
                candidate_key=key,
                logical_index=int(raw.get("logical_index", -1)),
                candidate_label=None if raw.get("candidate_label") is None else str(raw.get("candidate_label")),
                generator_id=None if raw.get("generator_id") is None else str(raw.get("generator_id")),
                position_id=None if raw.get("position_id") is None else int(raw.get("position_id")),
                admission_step=None if raw.get("admission_step") is None else int(raw.get("admission_step")),
                payload=dict(raw.get("payload", {}) or {}),
            )
        )
    keys: list[PlateauCandidateKey] = []
    for raw_key in payload.get("acquired_candidate_keys", ()) or ():
        if not isinstance(raw_key, Mapping):
            raise PlateauAcquisitionError("candidate key payloads must be mappings.")
        keys.append(
            candidate_key_from_parts(
                candidate_identity=raw_key.get("candidate_identity"),
                position_id=raw_key.get("position_id"),
            )
        )
    if not keys:
        keys = [record.candidate_key for record in dormant_records]
    state = PlateauAcquisitionState(
        active_episode=bool(payload.get("active_episode", bool(dormant_records))),
        dormant_records=tuple(dormant_records),
        acquired_candidate_keys=tuple(keys),
        failed_unlock_count=int(payload.get("failed_unlock_count", 0)),
        unlock_count=int(payload.get("unlock_count", 0)),
        last_event=(None if payload.get("last_event") is None else dict(payload.get("last_event", {}))),
    )
    _validate_state(state)
    return state


def _validate_state(state: PlateauAcquisitionState) -> None:
    acquired = {key.as_tuple() for key in state.acquired_candidate_keys}
    for record in state.dormant_records:
        if int(record.logical_index) < 0:
            raise PlateauAcquisitionError("dormant logical indices must be nonnegative.")
        if record.candidate_key.as_tuple() not in acquired:
            raise PlateauAcquisitionError("dormant records must have acquired candidate keys.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_safe(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "PHASE3_PLATEAU_FRACTIONAL_SCORE_FORMULA",
    "PHASE3_PLATEAU_LOG_VOLUME_SCORE_FORMULA",
    "PHASE3_PLATEAU_SCORE_FORMULA",
    "PLATEAU_ACQUISITION_SCORE_CHOICES",
    "PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1",
    "PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1",
    "PLATEAU_ACQUISITION_MODE_CHOICES",
    "PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1",
    "PLATEAU_ACQUISITION_MODE_OFF",
    "PLATEAU_ACQUISITION_SCHEMA",
    "PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY",
    "PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1",
    "PLATEAU_DUPLICATE_POLICY_CHOICES",
    "PLATEAU_TRIAL_OPTIMIZER_CHOICES",
    "PLATEAU_TRIAL_OPTIMIZER_INHERIT",
    "PLATEAU_TRIAL_OPTIMIZER_SP_QNGD",
    "PLATEAU_TRIAL_OPTIMIZER_SPSA",
    "PlateauAcquisitionConfig",
    "PlateauAcquisitionError",
    "PlateauAcquisitionState",
    "PlateauCandidateKey",
    "PlateauDormantRecord",
    "admit_failed_unlock_dormant",
    "candidate_key_from_parts",
    "candidate_key_from_record",
    "duplicate_status",
    "failed_family_backoff_status",
    "normalize_plateau_acquisition_config",
    "normalize_plateau_acquisition_mode",
    "normalize_plateau_acquisition_score",
    "normalize_plateau_duplicate_policy",
    "normalize_plateau_trial_optimizer",
    "plateau_score_formula",
    "plateau_state_from_payload",
    "record_successful_unlock",
    "remap_logical_index_after_insertion",
    "remap_logical_indices_after_insertion",
    "remap_state_after_insertion",
]
