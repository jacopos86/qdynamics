"""Exact-blind scientific AVQDS executor for prospective Paper-II cells.

The bootstrap scout and the scientific ladder deliberately have different
identities.  This module owns the latter: every request is rejoined to its
immutable scientific bindings and action-registry envelope, while the exact
trajectory named by that registry is never opened.  A successful execution
produces only raw statevector evidence in ``completed_unscored`` state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pipelines.time_dynamics.paper_ii_runs import build_run
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    _contract_json_value,
    _load_runtime_source_epoch_receipt,
    _normalized_hamiltonian_contract,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    ExactReferenceIdentity,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ACTION_REGISTRY_ENTRY_V1,
    EXEC_PROSPECTIVE_AVQDS,
    ExecutionRequest,
    ProspectiveExecutionError,
    RunLifecycleReceipt,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    VerifiedFile,
    executor_bindings,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_AVQDS,
    HH_HUBBARD_ACTIVATION_CUT_LADDER,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_RUN,
    ProspectiveScientificBindings,
    prospective_run_action_id,
)
from pipelines.time_dynamics.pure_hubbard_fixed_vqe import (
    PURE_HUBBARD_GENERATOR_EXECUTION,
    PURE_HUBBARD_GENERATOR_FILTER,
    PURE_HUBBARD_GENERATOR_ORDER,
    PURE_HUBBARD_GENERATOR_POOL,
)
from pipelines.time_dynamics.resume import runtime_input_from_resume_state
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
)


WORK_ORDER_SCHEMA_V1 = "paper_ii_avqds_trajectory_work_order_v1"
RUNTIME_REQUEST_SCHEMA_V1 = "paper_ii_avqds_exact_blind_runtime_request_v1"
RUNTIME_BINDINGS_SCHEMA_V1 = "paper_ii_avqds_exact_blind_runtime_bindings_v1"
COMPLETION_RECEIPT_SCHEMA_V1 = (
    "paper_ii_avqds_trajectory_completion_receipt_v1"
)
EXECUTOR_OVERLAY_SCHEMA_V1 = "paper_ii_avqds_trajectory_executor_overlay_v1"
EXECUTION_CONTRACT_SCHEMA_V1 = "paper_ii_avqds_trajectory_execution_contract_v1"
EXECUTION_ATTESTATION_SCHEMA_V1 = (
    "paper_ii_avqds_trajectory_execution_attestation_v1"
)
COMPLETED_UNSCORED = "completed_unscored"
VERIFIED_PACKAGED_SOURCE_ENV = "PAPER_II_VERIFIED_PACKAGED_SOURCE_SHA256"

AVQDS_NUMERICS = "euler_ridge1e-6"
AVQDS_CONTROLLER = "state_motion_5e-5"
AVQDS_SEARCH_PROFILE = "paper_declared_v1"
AVQDS_POOL_MODE = "replay_family_pool"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUNTIME_INPUT_KEYS = frozenset(
    {
        "bindings_sha256",
        "cell_id",
        "controller",
        "source_epoch_identity_sha256",
        "source_revision",
        "source_tree_sha256",
        "seed_id",
        "seed_identity_sha256",
        "seed_artifact_sha256",
        "seed_receipt_sha256",
        "reference_identity_sha256",
        "arm",
        "horizon",
        "activation_cut",
        "ladder_index",
        "record_statevector",
        "exact_reference_blind",
        "resource_only_retry",
        "scientific_action_id",
        "seeded_from_scout_action_id",
        "resume_from_action_id",
        "resume_run_receipt_sha256",
        "resume_run_sha256",
        "qiskit_compilation",
    }
)
_FORBIDDEN_RUNTIME_EXACT_KEYS = frozenset(
    {
        "action_registry",
        "accuracy_policy",
        "accuracy_policy_sha256",
        "exact_reference_receipt",
        "exact_reference_receipt_sha256",
        "reference_data_sha256",
        "reference_manifest_sha256",
        "reference_receipt_sha256",
        "response_scale",
        "response_scales",
        "response_scales_sha256",
        "source_receipts",
        "artifact_root",
        "external_manifest_path",
        "external_manifest_sha256",
        "external_data_path",
        "external_data_sha256",
        "materialized_key_requires",
    }
)

_WORKER_PATH = "pipelines/time_dynamics/prospective_avqds_trajectory_worker.py"
_RUNNER_PATH = "pipelines/time_dynamics/runners/run_prospective_avqds_trajectory.py"
EXECUTOR_OVERLAY_PATHS = (
    "pipelines/scaffold/runtime_loader.py",
    "pipelines/time_dynamics/paper_ii_runs.py",
    "pipelines/time_dynamics/prospective_execution_bindings.py",
    "pipelines/time_dynamics/prospective_avqds_scout_worker.py",
    "pipelines/time_dynamics/prospective_scientific_planner.py",
    "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
    _WORKER_PATH,
    _RUNNER_PATH,
    # Keep the uncapped pool interface and lazy enumeration dependencies together.
    "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_config.py",
    "pipelines/time_dynamics/ap_mclachlan/support_atoms.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_structural.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_selector.py",
    "pipelines/time_dynamics/ap_mclachlan/deletion_family.py",
    "pipelines/time_dynamics/ap_mclachlan/patch_enumeration.py",
    "pipelines/time_dynamics/ap_mclachlan/insertion_words.py",
    "pipelines/time_dynamics/generalized_exchange.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProspectiveExecutionError(
                f"refusing to overwrite immutable AVQDS evidence: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _verify_self_digest(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    unsigned = dict(payload)
    declared = unsigned.pop(key, None)
    actual = _canonical_sha(unsigned)
    if declared != actual:
        raise ProspectiveExecutionError(f"{label} self digest mismatch")
    return actual


def _git_bytes(root: Path, *arguments: str, allow_missing: bool = False) -> bytes | None:
    completed = subprocess.run(
        ("git", *arguments), cwd=root, check=False, capture_output=True
    )
    if completed.returncode == 0:
        return completed.stdout
    if allow_missing:
        return None
    detail = completed.stderr.decode("utf-8", errors="replace").strip()
    raise ProspectiveExecutionError(
        f"cannot resolve AVQDS executor overlay Git identity: {detail[:500]}"
    )


@dataclass(frozen=True)
class AVQDSTrajectoryExecutorOverlayIdentity:
    """Committed executor replacements composed with a qualified seed epoch."""

    base_source_epoch_identity_sha256: str
    base_source_revision: str
    overlay_revision: str
    files: Mapping[str, Mapping[str, Any]]
    files_sha256: str

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXECUTOR_OVERLAY_SCHEMA_V1,
            "source_composition": (
                "verified_pristine_seed_epoch_plus_committed_executor_overlay_v1"
            ),
            "base_source_epoch_identity_sha256": (
                self.base_source_epoch_identity_sha256
            ),
            "base_source_revision": self.base_source_revision,
            "overlay_revision": self.overlay_revision,
            "files": {key: dict(self.files[key]) for key in sorted(self.files)},
            "files_sha256": self.files_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha(self._identity_payload())

    @property
    def composite_source_identity_sha256(self) -> str:
        return _canonical_sha(
            {
                "base_source_epoch_identity_sha256": (
                    self.base_source_epoch_identity_sha256
                ),
                "executor_overlay_identity_sha256": self.identity_sha256,
            }
        )

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            **self._identity_payload(),
            "composite_source_identity_sha256": (
                self.composite_source_identity_sha256
            ),
        }
        if include_digest:
            payload["executor_overlay_identity_sha256"] = self.identity_sha256
        return payload

    @classmethod
    def from_mapping(cls, value: Any) -> "AVQDSTrajectoryExecutorOverlayIdentity":
        if not isinstance(value, Mapping):
            raise ProspectiveExecutionError("AVQDS executor overlay is missing")
        if value.get("schema") != EXECUTOR_OVERLAY_SCHEMA_V1:
            raise ProspectiveExecutionError("unsupported AVQDS executor overlay")
        files = value.get("files")
        if not isinstance(files, Mapping) or set(files) != set(EXECUTOR_OVERLAY_PATHS):
            raise ProspectiveExecutionError("AVQDS executor overlay file set changed")
        normalized: dict[str, dict[str, Any]] = {}
        for path in EXECUTOR_OVERLAY_PATHS:
            row = files.get(path)
            if not isinstance(row, Mapping):
                raise ProspectiveExecutionError("AVQDS executor overlay row is malformed")
            base_sha = row.get("base_sha256")
            overlay_sha = str(row.get("overlay_sha256") or "")
            if (
                (base_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(base_sha)))
                or not re.fullmatch(r"[0-9a-f]{64}", overlay_sha)
                or row.get("base_present") is not (base_sha is not None)
            ):
                raise ProspectiveExecutionError("AVQDS executor overlay digest is invalid")
            normalized[path] = {
                "base_present": base_sha is not None,
                "base_sha256": base_sha,
                "overlay_sha256": overlay_sha,
            }
        result = cls(
            base_source_epoch_identity_sha256=str(
                value.get("base_source_epoch_identity_sha256") or ""
            ),
            base_source_revision=str(value.get("base_source_revision") or ""),
            overlay_revision=str(value.get("overlay_revision") or ""),
            files=normalized,
            files_sha256=str(value.get("files_sha256") or ""),
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", result.base_source_epoch_identity_sha256)
            or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", result.base_source_revision)
            or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", result.overlay_revision)
            or result.files_sha256 != _canonical_sha(normalized)
            or value.get("source_composition")
            != "verified_pristine_seed_epoch_plus_committed_executor_overlay_v1"
            or value.get("composite_source_identity_sha256")
            != result.composite_source_identity_sha256
            or value.get("executor_overlay_identity_sha256") != result.identity_sha256
        ):
            raise ProspectiveExecutionError("AVQDS executor overlay identity diverged")
        return result

    def verify_files(self, root: Path, *, require_git_revision: bool) -> None:
        for path in EXECUTOR_OVERLAY_PATHS:
            target = root / path
            if not target.is_file() or _file_sha(target) != self.files[path]["overlay_sha256"]:
                raise ProspectiveExecutionError(f"AVQDS executor overlay bytes changed: {path}")
        if not require_git_revision:
            return
        resolved = _git_bytes(root, "rev-parse", "--verify", f"{self.overlay_revision}^{{commit}}")
        assert resolved is not None
        if resolved.decode("ascii").strip() != self.overlay_revision:
            raise ProspectiveExecutionError("AVQDS overlay revision is not exact")
        for path in EXECUTOR_OVERLAY_PATHS:
            overlay_bytes = _git_bytes(root, "show", f"{self.overlay_revision}:{path}")
            assert overlay_bytes is not None
            if hashlib.sha256(overlay_bytes).hexdigest() != self.files[path]["overlay_sha256"]:
                raise ProspectiveExecutionError(
                    f"AVQDS overlay revision/file digest mismatch: {path}"
                )
            base_bytes = _git_bytes(
                root, "show", f"{self.base_source_revision}:{path}", allow_missing=True
            )
            observed_base = None if base_bytes is None else hashlib.sha256(base_bytes).hexdigest()
            if observed_base != self.files[path]["base_sha256"]:
                raise ProspectiveExecutionError(f"AVQDS pristine base digest mismatch: {path}")


def discover_executor_overlay(
    root: Path, *, base_source_epoch: SourceEpochIdentity
) -> AVQDSTrajectoryExecutorOverlayIdentity:
    if base_source_epoch.source_revision is None:
        raise ProspectiveExecutionError("AVQDS base source epoch has no revision")
    revision_raw = _git_bytes(root, "rev-parse", "HEAD")
    assert revision_raw is not None
    revision = revision_raw.decode("ascii").strip()
    files: dict[str, dict[str, Any]] = {}
    for path in EXECUTOR_OVERLAY_PATHS:
        overlay_bytes = _git_bytes(root, "show", f"{revision}:{path}")
        assert overlay_bytes is not None
        current = root / path
        overlay_sha = hashlib.sha256(overlay_bytes).hexdigest()
        if not current.is_file() or _file_sha(current) != overlay_sha:
            raise ProspectiveExecutionError(
                f"AVQDS executor overlay is not committed at {revision}: {path}"
            )
        base_bytes = _git_bytes(
            root, "show", f"{base_source_epoch.source_revision}:{path}", allow_missing=True
        )
        files[path] = {
            "base_present": base_bytes is not None,
            "base_sha256": (
                None if base_bytes is None else hashlib.sha256(base_bytes).hexdigest()
            ),
            "overlay_sha256": overlay_sha,
        }
    result = AVQDSTrajectoryExecutorOverlayIdentity(
        base_source_epoch_identity_sha256=base_source_epoch.identity_sha256,
        base_source_revision=base_source_epoch.source_revision,
        overlay_revision=revision,
        files=files,
        files_sha256=_canonical_sha(files),
    )
    return AVQDSTrajectoryExecutorOverlayIdentity.from_mapping(result.to_json_dict())


def _require_sha256(value: Any, *, label: str) -> str:
    token = str(value or "")
    if not _SHA256.fullmatch(token):
        raise ProspectiveExecutionError(f"{label} must be one SHA-256 digest")
    return token


def _assert_exact_blind_runtime_payload(value: Any, *, path: str = "runtime") -> None:
    """Reject exact evidence and policy scales from an executable payload."""

    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_RUNTIME_EXACT_KEYS.intersection(value)
        if forbidden:
            raise ProspectiveExecutionError(
                f"{path} exposes forbidden exact inputs: {sorted(forbidden)!r}"
            )
        for key, child in value.items():
            _assert_exact_blind_runtime_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_exact_blind_runtime_payload(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class AVQDSRuntimeBindings:
    """Exact-blind projection of the canonical scientific cell bindings."""

    lane_id: str
    cell_id: str
    controller: str
    seed_id: str
    seed_identity_sha256: str
    seed_artifact_sha256: str
    seed_receipt_sha256: str
    source_epoch_identity_sha256: str
    source_revision: str
    source_tree_sha256: str
    reference_identity_sha256: str
    bindings_sha256: str

    @property
    def digest(self) -> str:
        return self.bindings_sha256

    @classmethod
    def from_canonical(
        cls, bindings: ProspectiveScientificBindings
    ) -> "AVQDSRuntimeBindings":
        return cls(
            lane_id=bindings.lane_id,
            cell_id=bindings.cell_id,
            controller=bindings.controller,
            seed_id=bindings.seed_id,
            seed_identity_sha256=bindings.seed_identity_sha256,
            seed_artifact_sha256=bindings.seed_artifact_sha256,
            seed_receipt_sha256=bindings.seed_receipt_sha256,
            source_epoch_identity_sha256=bindings.source_epoch_identity_sha256,
            source_revision=bindings.source_revision,
            source_tree_sha256=bindings.source_tree_sha256,
            reference_identity_sha256=bindings.reference_identity_sha256,
            bindings_sha256=bindings.digest,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AVQDSRuntimeBindings":
        if payload.get("schema") != RUNTIME_BINDINGS_SCHEMA_V1:
            raise ProspectiveExecutionError("unsupported exact-blind runtime bindings")
        result = cls(
            lane_id=str(payload.get("lane_id") or ""),
            cell_id=str(payload.get("cell_id") or ""),
            controller=str(payload.get("controller") or ""),
            seed_id=str(payload.get("seed_id") or ""),
            seed_identity_sha256=_require_sha256(
                payload.get("seed_identity_sha256"), label="runtime seed identity"
            ),
            seed_artifact_sha256=_require_sha256(
                payload.get("seed_artifact_sha256"), label="runtime seed artifact"
            ),
            seed_receipt_sha256=_require_sha256(
                payload.get("seed_receipt_sha256"), label="runtime seed receipt"
            ),
            source_epoch_identity_sha256=_require_sha256(
                payload.get("source_epoch_identity_sha256"),
                label="runtime source epoch identity",
            ),
            source_revision=str(payload.get("source_revision") or ""),
            source_tree_sha256=_require_sha256(
                payload.get("source_tree_sha256"), label="runtime source tree"
            ),
            reference_identity_sha256=_require_sha256(
                payload.get("reference_identity_sha256"),
                label="runtime opaque exact identity",
            ),
            bindings_sha256=_require_sha256(
                payload.get("bindings_sha256"), label="canonical bindings"
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            not self.lane_id
            or not self.cell_id
            or not self.seed_id
            or not self.source_revision
            or self.controller != AVQDS_CONTROLLER
        ):
            raise ProspectiveExecutionError("invalid exact-blind runtime bindings")
        for label, value in (
            ("runtime seed identity", self.seed_identity_sha256),
            ("runtime seed artifact", self.seed_artifact_sha256),
            ("runtime seed receipt", self.seed_receipt_sha256),
            ("runtime source epoch identity", self.source_epoch_identity_sha256),
            ("runtime source tree", self.source_tree_sha256),
            ("runtime opaque exact identity", self.reference_identity_sha256),
            ("canonical bindings", self.bindings_sha256),
        ):
            _require_sha256(value, label=label)

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "schema": RUNTIME_BINDINGS_SCHEMA_V1,
            "lane_id": self.lane_id,
            "cell_id": self.cell_id,
            "controller": self.controller,
            "seed_id": self.seed_id,
            "seed_identity_sha256": self.seed_identity_sha256,
            "seed_artifact_sha256": self.seed_artifact_sha256,
            "seed_receipt_sha256": self.seed_receipt_sha256,
            "source_epoch_identity_sha256": self.source_epoch_identity_sha256,
            "source_revision": self.source_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "reference_identity_sha256": self.reference_identity_sha256,
            "bindings_sha256": self.bindings_sha256,
        }
        _assert_exact_blind_runtime_payload(payload)
        return payload


def _projected_runtime_subjects(
    bindings: AVQDSRuntimeBindings,
    seed_payload: Mapping[str, Any],
    reference_payload: Mapping[str, Any],
) -> tuple[SeedIdentity, ExactReferenceIdentity]:
    """Authenticate locally sealed identities without reopening campaign data."""

    seed = SeedIdentity(
        seed_id=str(seed_payload.get("seed_id") or ""),
        lane_id=str(seed_payload.get("lane_id") or ""),
        model=str(seed_payload.get("model") or ""),
        physical=dict(seed_payload.get("physical") or {}),
        policy=str(seed_payload.get("policy") or ""),
        energy_gate=seed_payload.get("energy_gate"),
        artifact_path=str(seed_payload.get("artifact_path") or ""),
        external_artifact_sha256=seed_payload.get("external_artifact_sha256"),
        initializer=seed_payload.get("initializer"),
        restarts=seed_payload.get("restarts"),
        optimizer_seed=seed_payload.get("optimizer_seed"),
    )
    allowed_reference_keys = {
        "schema",
        "reference_id",
        "lane_id",
        "cell_id",
        "seed_id",
        "model",
        "physical",
        "drive",
        "grid",
        "propagator",
        "observable_schema",
        "identity_sha256",
    }
    if (
        set(reference_payload) != allowed_reference_keys
        or reference_payload.get("schema")
        != "paper_ii_avqds_runtime_reference_projection_v1"
    ):
        raise ProspectiveExecutionError(
            "AVQDS runtime reference projection exposes unauthorized fields"
        )
    reference = ExactReferenceIdentity(
        reference_id=str(reference_payload.get("reference_id") or ""),
        lane_id=str(reference_payload.get("lane_id") or ""),
        cell_id=str(reference_payload.get("cell_id") or ""),
        seed_id=str(reference_payload.get("seed_id") or ""),
        model=str(reference_payload.get("model") or ""),
        physical=dict(reference_payload.get("physical") or {}),
        drive=dict(reference_payload.get("drive") or {}),
        grid=dict(reference_payload.get("grid") or {}),
        propagator=str(reference_payload.get("propagator") or ""),
        observable_schema=str(reference_payload.get("observable_schema") or ""),
        artifact_root="",
    )
    if (
        seed_payload.get("identity_sha256") != seed.identity_sha256
        or reference_payload.get("identity_sha256") != reference.identity_sha256
        or seed.lane_id != bindings.lane_id
        or reference.lane_id != bindings.lane_id
        or seed.seed_id != bindings.seed_id
        or reference.seed_id != seed.seed_id
        or reference.cell_id != bindings.cell_id
        or seed.identity_sha256 != bindings.seed_identity_sha256
        or reference.identity_sha256 != bindings.reference_identity_sha256
    ):
        raise ProspectiveExecutionError(
            "AVQDS runtime identity projection diverged"
        )
    return seed, reference


def _runtime_reference_projection(
    reference: ExactReferenceIdentity,
) -> dict[str, Any]:
    return {
        "schema": "paper_ii_avqds_runtime_reference_projection_v1",
        "reference_id": reference.reference_id,
        "lane_id": reference.lane_id,
        "cell_id": reference.cell_id,
        "seed_id": reference.seed_id,
        "model": reference.model,
        "physical": dict(reference.physical),
        "drive": dict(reference.drive),
        "grid": dict(reference.grid),
        "propagator": reference.propagator,
        "observable_schema": reference.observable_schema,
        "identity_sha256": reference.identity_sha256,
    }


@dataclass(frozen=True)
class AVQDSRuntimeRequest:
    """Executable projection retaining the canonical request only by digest."""

    action_id: str
    lane_id: str
    kind: str
    binding_id: str
    handler: str
    inputs: Mapping[str, Any]
    declared_outputs: tuple[str, ...]
    exact_reference_blind: bool
    execution_id: str
    attempt: int
    planner_sequence: int
    requested_memory_gib: int
    arm: str
    execution_request_sha256: str

    @property
    def digest(self) -> str:
        return self.execution_request_sha256

    @classmethod
    def from_canonical(cls, request: ExecutionRequest) -> "AVQDSRuntimeRequest":
        inputs = {
            key: request.inputs.get(key)
            for key in sorted(_RUNTIME_INPUT_KEYS)
            if key in request.inputs
        }
        return cls(
            action_id=request.action_id,
            lane_id=request.lane_id,
            kind=request.kind,
            binding_id=request.binding_id,
            handler=request.handler,
            inputs=inputs,
            declared_outputs=request.declared_outputs,
            exact_reference_blind=request.exact_reference_blind,
            execution_id=request.execution_id,
            attempt=request.attempt,
            planner_sequence=request.planner_sequence,
            requested_memory_gib=int(request.requested_memory_gib or 0),
            arm=str(request.arm or ""),
            execution_request_sha256=request.digest,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AVQDSRuntimeRequest":
        if payload.get("schema") != RUNTIME_REQUEST_SCHEMA_V1:
            raise ProspectiveExecutionError("unsupported exact-blind runtime request")
        inputs = payload.get("inputs")
        outputs = payload.get("declared_outputs")
        if not isinstance(inputs, Mapping) or not isinstance(outputs, list):
            raise ProspectiveExecutionError("exact-blind runtime request is incomplete")
        if set(inputs).difference(_RUNTIME_INPUT_KEYS):
            raise ProspectiveExecutionError("runtime request contains an unauthorized input")
        result = cls(
            action_id=str(payload.get("action_id") or ""),
            lane_id=str(payload.get("lane_id") or ""),
            kind=str(payload.get("kind") or ""),
            binding_id=str(payload.get("binding_id") or ""),
            handler=str(payload.get("handler") or ""),
            inputs=dict(inputs),
            declared_outputs=tuple(str(value) for value in outputs),
            exact_reference_blind=bool(payload.get("exact_reference_blind")),
            execution_id=str(payload.get("execution_id") or ""),
            attempt=int(payload.get("attempt")),
            planner_sequence=int(payload.get("planner_sequence")),
            requested_memory_gib=int(payload.get("requested_memory_gib")),
            arm=str(payload.get("arm") or ""),
            execution_request_sha256=_require_sha256(
                payload.get("execution_request_sha256"),
                label="canonical execution request",
            ),
        )
        _assert_exact_blind_runtime_payload(result.to_json_dict())
        return result

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "schema": RUNTIME_REQUEST_SCHEMA_V1,
            "action_id": self.action_id,
            "lane_id": self.lane_id,
            "kind": self.kind,
            "binding_id": self.binding_id,
            "handler": self.handler,
            "inputs": dict(self.inputs),
            "declared_outputs": list(self.declared_outputs),
            "exact_reference_blind": self.exact_reference_blind,
            "execution_id": self.execution_id,
            "attempt": self.attempt,
            "planner_sequence": self.planner_sequence,
            "requested_memory_gib": self.requested_memory_gib,
            "arm": self.arm,
            "execution_request_sha256": self.execution_request_sha256,
            "qiskit_compilation_requested": False,
        }
        _assert_exact_blind_runtime_payload(payload)
        return payload


def _registered_subjects(
    bindings: ProspectiveScientificBindings | AVQDSRuntimeBindings,
) -> tuple[SeedIdentity, ExactReferenceIdentity]:
    registry = build_identity_registry()
    references = tuple(
        row
        for row in registry.references
        if row.cell_id == bindings.cell_id
        and row.identity_sha256 == bindings.reference_identity_sha256
        and row.lane_id == bindings.lane_id
    )
    seeds = tuple(
        row
        for row in registry.seeds
        if row.seed_id == bindings.seed_id
        and row.identity_sha256 == bindings.seed_identity_sha256
        and row.lane_id == bindings.lane_id
    )
    if len(seeds) != 1 or len(references) != 1 or references[0].seed_id != seeds[0].seed_id:
        raise ProspectiveExecutionError("AVQDS request is not matrix-registered")
    return seeds[0], references[0]


def _action_registry_authorization(
    file: VerifiedFile, *, root: Path, request: ExecutionRequest
) -> tuple[Mapping[str, Any], Mapping[str, VerifiedFile], str]:
    # The campaign-boundary registry was constructed only after validating all
    # three source receipts.  Preparation revalidates the source and seed
    # receipts needed by the trajectory, but treats the exact receipt solely as
    # an opaque digest and never opens it.
    file.verify(root)
    payload = json.loads((root / file.path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != ACTION_REGISTRY_ENTRY_V1:
        raise ProspectiveExecutionError("unsupported AVQDS action registry")
    _verify_self_digest(payload, "entry_sha256", label="AVQDS action registry")
    request_row = payload.get("execution_request")
    rows = payload.get("source_receipts")
    if not isinstance(request_row, Mapping) or not isinstance(rows, list):
        raise ProspectiveExecutionError("AVQDS action registry is incomplete")
    registered = ExecutionRequest.from_mapping(request_row)
    if registered.to_json_dict() != request.to_json_dict():
        raise ProspectiveExecutionError("AVQDS action registry names another request")
    by_digest: dict[str, VerifiedFile] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProspectiveExecutionError("AVQDS registry receipt binding is malformed")
        binding = VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))
        if binding.sha256 in by_digest:
            raise ProspectiveExecutionError("AVQDS registry repeats a receipt digest")
        by_digest[binding.sha256] = binding
    roles = {
        "source": str(payload.get("source_receipt_sha256") or ""),
        "seed": str(payload.get("seed_receipt_sha256") or ""),
        "exact": str(payload.get("exact_reference_receipt_sha256") or ""),
    }
    if set(roles.values()) != set(by_digest) or len(by_digest) != 3:
        raise ProspectiveExecutionError("AVQDS action registry receipt set changed")
    selected = {role: by_digest[digest] for role, digest in roles.items()}
    selected["source"].verify(root)
    selected["seed"].verify(root)
    return payload, selected, roles["exact"]


def validate_canonical_avqds_request(
    request: ExecutionRequest,
    *,
    action_registry: VerifiedFile,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    repository_root: str | Path,
) -> tuple[
    ProspectiveScientificBindings,
    SeedIdentity,
    ExactReferenceIdentity,
    SourceEpochIdentity,
]:
    """Authenticate one exact-blind scientific prefix or continuation."""

    root = Path(repository_root).resolve()
    try:
        bindings = ProspectiveScientificBindings.from_json_dict(request.inputs)
        cut = float(request.inputs.get("activation_cut"))
        horizon = str(request.inputs.get("horizon") or "")
        rung = int(request.inputs.get("ladder_index"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError("AVQDS request has invalid scientific bindings") from exc
    binding = next(
        (row for row in executor_bindings() if row.binding_id == EXEC_PROSPECTIVE_AVQDS),
        None,
    )
    expected_action = prospective_run_action_id(
        bindings, arm=ARM_AVQDS, horizon=horizon, activation_cut=cut
    )
    if (
        binding is None
        or request.kind != KIND_RUN
        or request.arm != ARM_AVQDS
        or request.binding_id != EXEC_PROSPECTIVE_AVQDS
        or request.handler != binding.handler
        or request.lane_id != bindings.lane_id
        or request.action_id != expected_action
        or request.inputs.get("scientific_action_id") != expected_action
        or request.declared_outputs
        != ("run_json", "trajectory_statevectors", "run_receipt")
        or not request.exact_reference_blind
        or request.inputs.get("exact_reference_blind") is not True
        or request.inputs.get("record_statevector") is not True
        or bindings.controller != AVQDS_CONTROLLER
        or request.inputs.get("controller") != AVQDS_CONTROLLER
        or horizon not in {HORIZON_PREFIX, HORIZON_FULL}
        or rung < 0
        or rung >= len(HH_HUBBARD_ACTIVATION_CUT_LADDER)
        or not math.isclose(
            cut,
            float(HH_HUBBARD_ACTIVATION_CUT_LADDER[rung]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or bool(request.inputs.get("qiskit_compilation"))
    ):
        raise ProspectiveExecutionError("request is not a canonical scientific AVQDS action")
    seed, reference = _registered_subjects(bindings)
    epoch = source_epoch_receipt.source_epoch
    registry_payload, receipts, opaque_exact_receipt_sha256 = (
        _action_registry_authorization(
        action_registry, root=root, request=request
        )
    )
    if set(receipts) != {"source", "seed", "exact"}:
        raise ProspectiveExecutionError("AVQDS action registry receipt set changed")
    if (
        source_epoch_receipt.receipt_sha256 != receipts["source"].sha256
        or seed_receipt.receipt_sha256 != receipts["seed"].sha256
        or opaque_exact_receipt_sha256 != bindings.reference_receipt_sha256
        or registry_payload.get("bindings_sha256") != bindings.digest
        or epoch.identity_sha256 != bindings.source_epoch_identity_sha256
        or epoch.source_revision != bindings.source_revision
        or source_epoch_receipt.source_tree_sha256 != bindings.source_tree_sha256
        or seed_receipt.seed_id != seed.seed_id
        or seed_receipt.seed_identity_sha256 != seed.identity_sha256
        or seed_receipt.artifact.sha256 != bindings.seed_artifact_sha256
        or seed_receipt.receipt_sha256 != bindings.seed_receipt_sha256
        or seed_receipt.source_epoch_identity_sha256 != epoch.identity_sha256
    ):
        raise ProspectiveExecutionError("AVQDS source/seed/reference binding diverged")
    return bindings, seed, reference, epoch


def _expected_times(reference: ExactReferenceIdentity, *, horizon: str) -> tuple[float, ...]:
    grid = reference.grid
    all_times = tuple(
        float(grid["start"]) + index * float(grid["step"])
        for index in range(int(grid["points"]))
    )
    if horizon == HORIZON_PREFIX:
        return all_times[: int(grid["prefix_points"])]
    return all_times[int(grid["prefix_points"]) - 1 :]


def _validate_resume(
    request: ExecutionRequest,
    *,
    bindings: ProspectiveScientificBindings,
    resume_run: VerifiedFile | None,
    resume_receipt: VerifiedFile | None,
    root: Path,
) -> None:
    horizon = str(request.inputs["horizon"])
    if horizon == HORIZON_PREFIX:
        if (
            resume_run is not None
            or resume_receipt is not None
            or request.inputs.get("resume_from_action_id") is not None
            or request.inputs.get("resume_run_sha256") is not None
            or request.inputs.get("resume_run_receipt_sha256") is not None
        ):
            raise ProspectiveExecutionError("AVQDS prefix cannot carry resume provenance")
        return
    if resume_run is None or resume_receipt is None:
        raise ProspectiveExecutionError("AVQDS continuation requires its selected prefix")
    resume_run.verify(root)
    resume_receipt.verify(root)
    lifecycle = RunLifecycleReceipt.load(resume_receipt.path, repository_root=root)
    lifecycle.validate_bindings(root, bindings)
    expected_prefix_action = prospective_run_action_id(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        activation_cut=float(request.inputs["activation_cut"]),
    )
    if (
        resume_run.sha256 != request.inputs.get("resume_run_sha256")
        or resume_receipt.sha256
        != request.inputs.get("resume_run_receipt_sha256")
        or request.inputs.get("resume_from_action_id") != expected_prefix_action
        or lifecycle.receipt_sha256 != resume_receipt.sha256
        or lifecycle.state != "completed"
        or lifecycle.method != ARM_AVQDS
        or lifecycle.horizon != HORIZON_PREFIX
        or lifecycle.scientific_action_id != expected_prefix_action
        or lifecycle.bindings_sha256 != bindings.digest
        or lifecycle.run != resume_run
        or not math.isclose(
            lifecycle.activation_cut,
            float(request.inputs["activation_cut"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ProspectiveExecutionError("AVQDS continuation resume provenance diverged")


def _output_paths(request: ExecutionRequest) -> tuple[str, str, str]:
    directory = (
        Path("output/prospective_scientific")
        / request.lane_id
        / str(request.inputs["cell_id"])
        / "avqds"
        / request.digest[:20]
    )
    final = (directory / "run.json").as_posix()
    return final, f"{final}.part", (directory / "completion_receipt.json").as_posix()


def _runner_argv(
    request: ExecutionRequest,
    *,
    reference: ExactReferenceIdentity,
    seed_receipt: SeedQualificationReceipt,
    overlay: AVQDSTrajectoryExecutorOverlayIdentity,
    output_path: str,
    resume_run: VerifiedFile | None,
    repository_root: Path,
) -> tuple[str, ...]:
    seed_payload = json.loads(
        (repository_root / seed_receipt.artifact.path).read_text(encoding="utf-8")
    )
    adapt = seed_payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise ProspectiveExecutionError("seed artifact has no runtime adapt_vqe block")
    parameterization = str(
        adapt.get(
            "parameterization_execution_mode",
            adapt.get("parameterization_mode", ""),
        )
    ).strip()
    expected_parameterization = (
        "per_pauli_term" if reference.model == "hubbard_holstein" else "logical_shared"
    )
    if parameterization != expected_parameterization:
        raise ProspectiveExecutionError("seed parameterization mode changed")
    horizon = str(request.inputs["horizon"])
    continuation_flags: tuple[str, ...] = ()
    if horizon == HORIZON_FULL:
        if resume_run is None:
            raise ProspectiveExecutionError("full AVQDS work requires its prefix run")
        times = _expected_times(reference, horizon=horizon)
        continuation_flags = (
            "--resume-from-run-json",
            resume_run.path,
            "--t-initial",
            repr(times[0]),
            "--num-times",
            str(len(times)),
        )
    command = build_run(
        arm="avqds",
        drive=str(reference.drive["drive_id"]),
        horizon=horizon,
        output_json=output_path,
        seed_path=seed_receipt.artifact.path,
        numerics=AVQDS_NUMERICS,
        step_control=AVQDS_CONTROLLER,
        search_profile=AVQDS_SEARCH_PROFILE,
        activation_cut=float(request.inputs["activation_cut"]),
        extra_flags=(
            "--record-statevector",
            "--require-complete-candidate-pool",
            "--loader-mode",
            "replay_family",
            "--replay-candidate-pool-mode",
            AVQDS_POOL_MODE,
            "--parameterization-mode",
            parameterization,
            "--code-revision",
            overlay.overlay_revision,
            *continuation_flags,
        ),
    )
    argv = command.argv()
    if any("exact" in token.lower() and "exact-blind" not in token.lower() for token in argv):
        raise ProspectiveExecutionError("AVQDS runner argv contains exact-reference input")
    if any("qiskit" in token.lower() for token in argv):
        raise ProspectiveExecutionError("Qiskit is excluded from scientific AVQDS")
    return argv


def _pure_hubbard_gate(contract: Mapping[str, Any], reference: ExactReferenceIdentity) -> None:
    if reference.model != "hubbard":
        return
    source = contract.get("candidate_pool_source")
    override = (
        source.get("filter_payload", {}).get("explicit_complete_pool_override")
        if isinstance(source, Mapping)
        else None
    )
    if (
        not isinstance(override, Mapping)
        or source.get("pool_key") != PURE_HUBBARD_GENERATOR_POOL
        or source.get("candidate_pool_complete") is not True
        or override.get("declared_generator_filter") != PURE_HUBBARD_GENERATOR_FILTER
        or override.get("declared_generator_pool_order") != PURE_HUBBARD_GENERATOR_ORDER
        or override.get("declared_generator_pool_execution")
        != PURE_HUBBARD_GENERATOR_EXECUTION
        or override.get("prospective_seed_declarations_verified") is not True
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(override.get("declared_generator_pool_sha256") or "")
        )
        or int(override.get("declared_generator_pool_count", 0)) <= 0
    ):
        raise ProspectiveExecutionError(
            "pure-Hubbard AVQDS requires the declared complete filtered pool"
        )


def _build_execution_contract(
    *,
    request: ExecutionRequest,
    reference: ExactReferenceIdentity,
    seed_receipt: SeedQualificationReceipt,
    overlay: AVQDSTrajectoryExecutorOverlayIdentity,
    runner_argv: tuple[str, ...],
    resume_run: VerifiedFile | None,
    repository_root: Path,
) -> dict[str, Any]:
    """Resolve the live variational contract without opening exact data."""

    import pipelines.time_dynamics.runners.ap_append_from_adapt_artifact as runner

    try:
        args = runner._build_parser().parse_args(list(runner_argv))
        runtime_input = runner._load_runtime_input_or_raise(
            repository_root / seed_receipt.artifact.path,
            loader_mode=args.loader_mode,
            tag=args.tag,
            generator_family=str(args.generator_family),
            fallback_family=str(args.fallback_family),
            replay_candidate_pool_mode=args.replay_candidate_pool_mode,
        )
        resumed = resume_run is not None
        if resumed:
            prior = json.loads((repository_root / resume_run.path).read_text(encoding="utf-8"))
            resume_state = prior.get("resume_state")
            if not isinstance(resume_state, Mapping):
                raise ProspectiveExecutionError("AVQDS prefix has no typed resume state")
            runtime_input = runtime_input_from_resume_state(
                resume_state, base_runtime_input=runtime_input
            )
        drive_config = runner._drive_config_from_args(args, runtime_input)
        hamiltonian = runner.time_dependent_hamiltonian_from_runtime_input(
            runtime_input, drive_config=drive_config if bool(args.enable_drive) else None
        )
        state = runner.state_from_scaffold_runtime_input(
            runtime_input, parameterization_mode=str(args.parameterization_mode)
        )
        if resumed:
            augmentation_payload = {"enabled": False, "reason": "resumed_exact_prefix_state"}
        else:
            augmentation = runner.augment_state_with_drive_aligned_generator(
                state,
                hamiltonian=hamiltonian,
                enabled=bool(args.enable_drive) and bool(args.drive_aligned_ansatz),
            )
            state = augmentation.state
            augmentation_payload = augmentation.to_json_dict()
        support = runner._support_patch_config_from_args(args)
        repair = runner._solve_repair_config_from_args(args)
        controller = runner.AppendControllerConfig(
            append_min_time=float(args.append_min_time),
            residual_ratio_threshold=float(args.residual_ratio_threshold),
            min_logical_parameter_count=int(args.min_logical_parameter_count),
            allow_incomplete_candidate_pool=not bool(args.require_complete_candidate_pool),
        )
        pool = runner.build_candidate_pool_contract(state, support_config=support)
        source = getattr(state, "candidate_pool_source", None)
        source_contract = {
            "source_kind": str(getattr(source, "source_kind", "")),
            "pool_key": getattr(source, "pool_key", None),
            "completeness": str(getattr(source, "completeness", "")),
            "candidate_pool_complete": bool(
                getattr(source, "candidate_pool_complete", False)
            ),
            "pool_build_kwargs": _contract_json_value(
                dict(getattr(source, "pool_build_kwargs", {}) or {})
            ),
            "filter_payload": _contract_json_value(
                dict(getattr(source, "filter_payload", {}) or {})
            ),
        }
        times = tuple(float(value) for value in runner._resolve_times(args))
    except ProspectiveExecutionError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "cannot reproduce scientific AVQDS runtime contract"
        ) from exc
    expected_times = _expected_times(reference, horizon=str(request.inputs["horizon"]))
    if times != expected_times:
        raise ProspectiveExecutionError("AVQDS runner grid differs from the frozen cell grid")
    support_payload = support.to_json_dict()
    repair_payload = repair.to_json_dict()
    hamiltonian_payload = _normalized_hamiltonian_contract(
        hamiltonian.to_json_dict(), seed_artifact_path=seed_receipt.artifact.path
    )
    state_payload = state.to_json_dict()
    core = {
        "schema": EXECUTION_CONTRACT_SCHEMA_V1,
        "execution_request_sha256": request.digest,
        "bindings_sha256": request.inputs["bindings_sha256"],
        "runner_argv_sha256": _canonical_sha(list(runner_argv)),
        "source_epoch_identity_sha256": request.inputs["source_epoch_identity_sha256"],
        "reference_identity_sha256": request.inputs["reference_identity_sha256"],
        "exact_reference_data_access": False,
        "exact_reference_runtime_artifacts_present": False,
        "exact_reference_binding_mode": "opaque_identity_only",
        "composite_source_identity_sha256": overlay.composite_source_identity_sha256,
        "horizon": request.inputs["horizon"],
        "activation_cut": float(request.inputs["activation_cut"]),
        "activation_ladder_index": int(request.inputs["ladder_index"]),
        "reporting_grid": {
            "start": times[0],
            "final": times[-1],
            "point_count": len(times),
            "ordered_values_sha256": _canonical_sha(times),
            "continuation": resume_run is not None,
        },
        "integrator": str(args.integrator),
        "inverse_policy": {
            "pinv_rcond": float(args.pinv_rcond),
            "ridge_lambda": float(args.ridge_lambda),
            "solve_damping": float(args.solve_damping),
        },
        "parameterization_mode": str(args.parameterization_mode),
        "record_statevector": bool(args.record_statevector),
        "require_complete_candidate_pool": bool(args.require_complete_candidate_pool),
        "controller_config": controller.to_json_dict(),
        "support_patch_config": support_payload,
        "solve_repair_config": repair_payload,
        "time_step_controller": {
            "parameter_displacement_max": support_payload.get("avqds_delta_theta_max"),
            "state_motion_l2_step_max": repair_payload.get("state_motion_l2_step_max"),
            "composed_parameter_step_max": repair_payload.get("parameter_step_max"),
            "local_subdivision_enabled": bool(repair_payload.get("local_subdivision_enabled")),
            "max_local_subdivisions": int(repair_payload.get("max_local_subdivisions", 0)),
        },
        "fail_on_unsupported_steps": bool(args.fail_on_unsupported_steps),
        "hamiltonian": hamiltonian_payload,
        "drive_profile": hamiltonian_payload.get("drive_profile"),
        "drive_aligned_ansatz": augmentation_payload,
        "candidate_pool_contract": pool,
        "candidate_pool_source": source_contract,
        "initial_state_contract": {
            key: state_payload[key]
            for key in (
                "parameterization_mode",
                "parameterization_label",
                "active_parameter_count",
                "logical_parameter_count",
                "runtime_parameter_count",
                "runtime_pauli_parameter_count",
                "candidate_pool_term_count",
                "candidate_pool_complete",
            )
        },
        "physical_hilbert_dimension": int(len(state.psi_initial)),
        "qiskit_compilation_requested": False,
    }
    _pure_hubbard_gate(core, reference)
    return {**core, "execution_contract_sha256": _canonical_sha(core)}


@dataclass(frozen=True)
class AVQDSTrajectoryWorkOrder:
    request: AVQDSRuntimeRequest
    bindings: AVQDSRuntimeBindings
    seed: SeedIdentity
    reference: ExactReferenceIdentity
    action_authorization_sha256: str
    source_epoch_receipt: VerifiedFile
    seed_receipt: VerifiedFile
    executor_overlay: AVQDSTrajectoryExecutorOverlayIdentity
    resume_run: VerifiedFile | None
    resume_receipt: VerifiedFile | None
    run_path: str
    run_part_path: str
    completion_receipt_path: str
    runner_argv: tuple[str, ...]
    execution_contract: Mapping[str, Any]
    work_order_sha256: str

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": WORK_ORDER_SCHEMA_V1,
            "runtime_request": self.request.to_json_dict(),
            "runtime_bindings": self.bindings.to_json_dict(),
            "seed_identity": self.seed.to_json_dict(),
            "reference_identity": _runtime_reference_projection(self.reference),
            "action_authorization_sha256": self.action_authorization_sha256,
            "source_epoch_receipt": self.source_epoch_receipt.to_json_dict(),
            "seed_qualification_receipt": self.seed_receipt.to_json_dict(),
            "executor_overlay": self.executor_overlay.to_json_dict(),
            "composite_source_identity_sha256": (
                self.executor_overlay.composite_source_identity_sha256
            ),
            "resume_run": None if self.resume_run is None else self.resume_run.to_json_dict(),
            "resume_receipt": (
                None if self.resume_receipt is None else self.resume_receipt.to_json_dict()
            ),
            "runner": {
                "module": "pipelines.time_dynamics.runners.ap_append_from_adapt_artifact",
                "argv": list(self.runner_argv),
                "argv_sha256": _canonical_sha(list(self.runner_argv)),
            },
            "execution_contract": dict(self.execution_contract),
            "outputs": {
                "run": self.run_path,
                "run_part": self.run_part_path,
                "completion_receipt": self.completion_receipt_path,
            },
            "scientific_status_on_success": COMPLETED_UNSCORED,
            "scoring_required_before_qualification": True,
            "remote_raw_evidence_policy": "preserve",
            "qiskit_compilation_requested": False,
            "production_submission_enabled": False,
        }
        if include_digest:
            payload["work_order_sha256"] = self.work_order_sha256
        _assert_exact_blind_runtime_payload(payload)
        return payload


def _validate_runtime_request(
    request: AVQDSRuntimeRequest, bindings: AVQDSRuntimeBindings
) -> None:
    binding = next(
        (row for row in executor_bindings() if row.binding_id == EXEC_PROSPECTIVE_AVQDS),
        None,
    )
    try:
        cut = float(request.inputs.get("activation_cut"))
        rung = int(request.inputs.get("ladder_index"))
        horizon = str(request.inputs.get("horizon") or "")
        expected_action = prospective_run_action_id(
            bindings, arm=ARM_AVQDS, horizon=horizon, activation_cut=cut
        )
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError("invalid exact-blind runtime request") from exc
    if (
        binding is None
        or request.kind != KIND_RUN
        or request.arm != ARM_AVQDS
        or request.binding_id != EXEC_PROSPECTIVE_AVQDS
        or request.handler != binding.handler
        or request.lane_id != bindings.lane_id
        or request.action_id != expected_action
        or request.inputs.get("scientific_action_id") != expected_action
        or request.inputs.get("bindings_sha256") != bindings.digest
        or request.inputs.get("cell_id") != bindings.cell_id
        or request.inputs.get("controller") != bindings.controller
        or request.inputs.get("seed_id") != bindings.seed_id
        or request.inputs.get("seed_identity_sha256") != bindings.seed_identity_sha256
        or request.inputs.get("seed_artifact_sha256") != bindings.seed_artifact_sha256
        or request.inputs.get("seed_receipt_sha256") != bindings.seed_receipt_sha256
        or request.inputs.get("source_epoch_identity_sha256")
        != bindings.source_epoch_identity_sha256
        or request.inputs.get("source_revision") != bindings.source_revision
        or request.inputs.get("source_tree_sha256") != bindings.source_tree_sha256
        or request.inputs.get("reference_identity_sha256")
        != bindings.reference_identity_sha256
        or request.inputs.get("arm") != ARM_AVQDS
        or horizon not in {HORIZON_PREFIX, HORIZON_FULL}
        or rung < 0
        or rung >= len(HH_HUBBARD_ACTIVATION_CUT_LADDER)
        or not math.isclose(
            cut,
            float(HH_HUBBARD_ACTIVATION_CUT_LADDER[rung]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or request.declared_outputs
        != ("run_json", "trajectory_statevectors", "run_receipt")
        or not request.exact_reference_blind
        or request.inputs.get("exact_reference_blind") is not True
        or request.inputs.get("record_statevector") is not True
        or bool(request.inputs.get("qiskit_compilation"))
        or request.attempt not in (0, 1)
        or request.planner_sequence < 0
        or request.requested_memory_gib <= 0
        or request.execution_id
        != (
            f"{request.action_id}::attempt{request.attempt}::"
            f"mem{request.requested_memory_gib}g"
        )
    ):
        raise ProspectiveExecutionError("exact-blind runtime request diverged")
    _require_sha256(request.digest, label="canonical execution request")
    _assert_exact_blind_runtime_payload(request.to_json_dict())


def _validate_runtime_resume(
    request: AVQDSRuntimeRequest,
    *,
    resume_run: VerifiedFile | None,
    resume_receipt: VerifiedFile | None,
    root: Path,
) -> None:
    horizon = str(request.inputs["horizon"])
    if horizon == HORIZON_PREFIX:
        if (
            resume_run is not None
            or resume_receipt is not None
            or request.inputs.get("resume_from_action_id") is not None
            or request.inputs.get("resume_run_sha256") is not None
            or request.inputs.get("resume_run_receipt_sha256") is not None
        ):
            raise ProspectiveExecutionError("AVQDS prefix cannot carry resume provenance")
        return
    if resume_run is None or resume_receipt is None:
        raise ProspectiveExecutionError("AVQDS continuation requires its selected prefix")
    resume_run.verify(root)
    if (
        resume_run.sha256 != request.inputs.get("resume_run_sha256")
        or resume_receipt.sha256
        != request.inputs.get("resume_run_receipt_sha256")
        or request.inputs.get("resume_from_action_id") in (None, "")
    ):
        raise ProspectiveExecutionError("AVQDS continuation resume digest diverged")


def _assemble_runtime_work_order(
    *,
    request: AVQDSRuntimeRequest,
    bindings: AVQDSRuntimeBindings,
    action_authorization_sha256: str,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    root: Path,
    executor_overlay: AVQDSTrajectoryExecutorOverlayIdentity | None,
    resume_run: VerifiedFile | None,
    resume_receipt: VerifiedFile | None,
    prevalidated_subjects: tuple[SeedIdentity, ExactReferenceIdentity] | None = None,
) -> AVQDSTrajectoryWorkOrder:
    bindings.validate()
    _validate_runtime_request(request, bindings)
    seed, reference = (
        _registered_subjects(bindings)
        if prevalidated_subjects is None
        else prevalidated_subjects
    )
    if (
        seed.lane_id != bindings.lane_id
        or reference.lane_id != bindings.lane_id
        or seed.seed_id != bindings.seed_id
        or reference.seed_id != seed.seed_id
        or reference.cell_id != bindings.cell_id
        or seed.identity_sha256 != bindings.seed_identity_sha256
        or reference.identity_sha256 != bindings.reference_identity_sha256
    ):
        raise ProspectiveExecutionError("AVQDS runtime subject binding diverged")
    epoch = source_epoch_receipt.source_epoch
    _require_sha256(action_authorization_sha256, label="action authorization")
    seed_receipt.artifact.verify(root)
    if (
        epoch.identity_sha256 != bindings.source_epoch_identity_sha256
        or epoch.source_revision != bindings.source_revision
        or source_epoch_receipt.source_tree_sha256 != bindings.source_tree_sha256
        or seed_receipt.seed_id != seed.seed_id
        or seed_receipt.seed_identity_sha256 != seed.identity_sha256
        or seed_receipt.artifact.sha256 != bindings.seed_artifact_sha256
        or seed_receipt.receipt_sha256 != bindings.seed_receipt_sha256
        or seed_receipt.source_epoch_identity_sha256 != epoch.identity_sha256
    ):
        raise ProspectiveExecutionError("AVQDS exact-blind runtime source binding diverged")
    _validate_runtime_resume(
        request,
        resume_run=resume_run,
        resume_receipt=resume_receipt,
        root=root,
    )
    overlay = (
        discover_executor_overlay(root, base_source_epoch=epoch)
        if executor_overlay is None
        else AVQDSTrajectoryExecutorOverlayIdentity.from_mapping(
            executor_overlay.to_json_dict()
        )
    )
    if (
        overlay.base_source_epoch_identity_sha256 != epoch.identity_sha256
        or overlay.base_source_revision != epoch.source_revision
    ):
        raise ProspectiveExecutionError("AVQDS overlay crosses the seed source epoch")
    overlay.verify_files(root, require_git_revision=(root / ".git").exists())
    run_path, part_path, receipt_path = _output_paths(request)
    argv = _runner_argv(
        request,
        reference=reference,
        seed_receipt=seed_receipt,
        overlay=overlay,
        output_path=part_path,
        resume_run=resume_run,
        repository_root=root,
    )
    contract = _build_execution_contract(
        request=request,
        reference=reference,
        seed_receipt=seed_receipt,
        overlay=overlay,
        runner_argv=argv,
        resume_run=resume_run,
        repository_root=root,
    )
    provisional = AVQDSTrajectoryWorkOrder(
        request=request,
        bindings=bindings,
        seed=seed,
        reference=reference,
        action_authorization_sha256=action_authorization_sha256,
        source_epoch_receipt=VerifiedFile(
            source_epoch_receipt.receipt_path, source_epoch_receipt.receipt_sha256
        ),
        seed_receipt=VerifiedFile(seed_receipt.receipt_path, seed_receipt.receipt_sha256),
        executor_overlay=overlay,
        resume_run=resume_run,
        resume_receipt=resume_receipt,
        run_path=run_path,
        run_part_path=part_path,
        completion_receipt_path=receipt_path,
        runner_argv=argv,
        execution_contract=contract,
        work_order_sha256="",
    )
    return AVQDSTrajectoryWorkOrder(
        **{
            **provisional.__dict__,
            "work_order_sha256": _canonical_sha(
                provisional.to_json_dict(include_digest=False)
            ),
        }
    )


def build_work_order(
    request: ExecutionRequest,
    *,
    action_registry: VerifiedFile,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    repository_root: str | Path,
    executor_overlay: AVQDSTrajectoryExecutorOverlayIdentity | None = None,
    resume_run: VerifiedFile | None = None,
    resume_receipt: VerifiedFile | None = None,
) -> AVQDSTrajectoryWorkOrder:
    """Authorize canonically, then discard all readable exact inputs."""

    root = Path(repository_root).resolve()
    bindings, seed, reference, _epoch = validate_canonical_avqds_request(
        request,
        action_registry=action_registry,
        source_epoch_receipt=source_epoch_receipt,
        seed_receipt=seed_receipt,
        repository_root=root,
    )
    _validate_resume(
        request,
        bindings=bindings,
        resume_run=resume_run,
        resume_receipt=resume_receipt,
        root=root,
    )
    return _assemble_runtime_work_order(
        request=AVQDSRuntimeRequest.from_canonical(request),
        bindings=AVQDSRuntimeBindings.from_canonical(bindings),
        action_authorization_sha256=action_registry.sha256,
        source_epoch_receipt=source_epoch_receipt,
        seed_receipt=seed_receipt,
        root=root,
        executor_overlay=executor_overlay,
        resume_run=resume_run,
        resume_receipt=resume_receipt,
        prevalidated_subjects=(seed, reference),
    )


def write_work_order(path: str | Path, order: AVQDSTrajectoryWorkOrder) -> Path:
    target = Path(path)
    _atomic_write_immutable(target, _pretty_bytes(order.to_json_dict()))
    return target


def _optional_verified(payload: Mapping[str, Any], name: str) -> VerifiedFile | None:
    row = payload.get(name)
    if row is None:
        return None
    if not isinstance(row, Mapping):
        raise ProspectiveExecutionError(f"AVQDS work order has malformed {name}")
    return VerifiedFile(str(row.get("path") or ""), str(row.get("sha256") or ""))


def load_work_order(
    path: str | Path, *, repository_root: str | Path
) -> tuple[AVQDSTrajectoryWorkOrder, SourceEpochReceipt, SeedQualificationReceipt]:
    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError("AVQDS work order is outside repository") from exc
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("AVQDS work order is unreadable") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != WORK_ORDER_SCHEMA_V1:
        raise ProspectiveExecutionError("unsupported AVQDS work order")
    work_sha = _verify_self_digest(payload, "work_order_sha256", label="AVQDS work order")
    _assert_exact_blind_runtime_payload(payload)
    request_row = payload.get("runtime_request")
    bindings_row = payload.get("runtime_bindings")
    seed_identity_row = payload.get("seed_identity")
    reference_identity_row = payload.get("reference_identity")
    source_row = payload.get("source_epoch_receipt")
    seed_row = payload.get("seed_qualification_receipt")
    overlay_row = payload.get("executor_overlay")
    if not all(
        isinstance(row, Mapping)
        for row in (
            request_row,
            bindings_row,
            seed_identity_row,
            reference_identity_row,
            source_row,
            seed_row,
            overlay_row,
        )
    ):
        raise ProspectiveExecutionError("AVQDS work order is incomplete")
    request = AVQDSRuntimeRequest.from_mapping(request_row)
    bindings = AVQDSRuntimeBindings.from_mapping(bindings_row)
    seed, reference = _projected_runtime_subjects(
        bindings, seed_identity_row, reference_identity_row
    )
    expected_epoch = SourceEpochIdentity(
        lane_id=bindings.lane_id,
        source_epoch_id=str(
            json.loads(
                (root / str(source_row.get("path") or "")).read_text(encoding="utf-8")
            )["source_epoch"]["source_epoch_id"]
        ),
        source_revision=bindings.source_revision,
        source_archive_sha256=json.loads(
            (root / str(source_row.get("path") or "")).read_text(encoding="utf-8")
        )["source_epoch"].get("source_archive_sha256"),
    )
    source_receipt = _load_runtime_source_epoch_receipt(
        str(source_row.get("path") or ""),
        repository_root=root,
        expected_epoch=expected_epoch,
    )
    seed_receipt = SeedQualificationReceipt.load(
        str(seed_row.get("path") or ""),
        repository_root=root,
        expected_seed=seed,
        expected_source_epoch=source_receipt.source_epoch,
    )
    if (
        _file_sha(root / source_receipt.receipt_path) != source_row.get("sha256")
        or _file_sha(root / seed_receipt.receipt_path) != seed_row.get("sha256")
    ):
        raise ProspectiveExecutionError("AVQDS work-order receipt file changed")
    rebuilt = _assemble_runtime_work_order(
        request=request,
        bindings=bindings,
        action_authorization_sha256=_require_sha256(
            payload.get("action_authorization_sha256"),
            label="action authorization",
        ),
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
        root=root,
        executor_overlay=AVQDSTrajectoryExecutorOverlayIdentity.from_mapping(overlay_row),
        resume_run=_optional_verified(payload, "resume_run"),
        resume_receipt=_optional_verified(payload, "resume_receipt"),
        prevalidated_subjects=(seed, reference),
    )
    if rebuilt.work_order_sha256 != work_sha or rebuilt.to_json_dict() != dict(payload):
        raise ProspectiveExecutionError("AVQDS work order canonical form changed")
    return rebuilt, source_receipt, seed_receipt


def _execution_attestation(order: AVQDSTrajectoryWorkOrder) -> dict[str, Any]:
    core = {
        "schema": EXECUTION_ATTESTATION_SCHEMA_V1,
        "execution_request_sha256": order.request.digest,
        "work_order_sha256": order.work_order_sha256,
        "execution_contract_sha256": order.execution_contract[
            "execution_contract_sha256"
        ],
        "runner_argv_sha256": _canonical_sha(list(order.runner_argv)),
        "composite_source_identity_sha256": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "verified_packaged_source_marker": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "exact_reference_blind": True,
        "exact_reference_data_access": False,
        "exact_reference_runtime_artifacts_present": False,
        "exact_reference_binding_mode": "opaque_identity_only",
        "reference_identity_sha256": order.bindings.reference_identity_sha256,
        "qiskit_compilation_used": False,
    }
    return {**core, "execution_attestation_sha256": _canonical_sha(core)}


def annotate_raw_run(
    path: str | Path, *, order: AVQDSTrajectoryWorkOrder
) -> None:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("AVQDS raw run is unreadable") from exc
    if not isinstance(payload, dict):
        raise ProspectiveExecutionError("AVQDS raw run must be an object")
    attestation = _execution_attestation(order)
    existing = payload.get("paper_ii_execution_attestation")
    if existing is not None and existing != attestation:
        raise ProspectiveExecutionError("AVQDS raw attestation changed")
    payload["paper_ii_execution_attestation"] = attestation
    _atomic_write_immutable(target.with_suffix(target.suffix + ".attested"), _pretty_bytes(payload))
    os.replace(target.with_suffix(target.suffix + ".attested"), target)


def validate_raw_run(
    path: str | Path,
    *,
    order: AVQDSTrajectoryWorkOrder,
    seed_receipt: SeedQualificationReceipt,
) -> None:
    """Validate variational raw bytes without reading exact trajectory data."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("AVQDS raw run is unreadable") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != RUNNER_SCHEMA_V1:
        raise ProspectiveExecutionError("AVQDS raw run has an unsupported schema")
    if {
        "scientific_qualification",
        "score_receipt",
        "qualification_status",
        "exact_reference_receipt",
        "exact_reference_data",
    }.intersection(payload):
        raise ProspectiveExecutionError("raw AVQDS output contains exact/scored data")
    if payload.get("paper_ii_execution_attestation") != _execution_attestation(order):
        raise ProspectiveExecutionError("AVQDS raw run lacks its execution attestation")
    contract = order.execution_contract
    if (
        _verify_self_digest(
            contract, "execution_contract_sha256", label="AVQDS execution contract"
        )
        != contract.get("execution_contract_sha256")
        or contract.get("execution_request_sha256") != order.request.digest
        or contract.get("bindings_sha256") != order.bindings.digest
        or contract.get("exact_reference_data_access") is not False
    ):
        raise ProspectiveExecutionError("AVQDS execution contract diverged")
    flow = payload.get("decision_data_flow")
    lock = payload.get("run_lock")
    rows = payload.get("plot_rows")
    trajectory = payload.get("trajectory")
    initial_state = payload.get("initial_state")
    hamiltonian = payload.get("hamiltonian")
    if (
        not isinstance(flow, Mapping)
        or not isinstance(lock, Mapping)
        or not isinstance(rows, list)
        or not isinstance(trajectory, Mapping)
        or not isinstance(initial_state, Mapping)
        or not isinstance(hamiltonian, Mapping)
    ):
        raise ProspectiveExecutionError("AVQDS raw run is incomplete")
    if any(
        flow.get(key) is not False
        for key in (
            "uses_reference_for_decision",
            "uses_exact_reference_for_decision",
            "uses_future_exact_forecast_for_decision",
        )
    ):
        raise ProspectiveExecutionError("AVQDS controller consumed exact-reference data")
    physics = lock.get("physics")
    numerics = lock.get("numerics")
    policy = lock.get("policy")
    shared = lock.get("shared_protocol")
    if not all(isinstance(row, Mapping) for row in (physics, numerics, policy, shared)):
        raise ProspectiveExecutionError("AVQDS run lock is incomplete")
    times = _expected_times(order.reference, horizon=str(order.request.inputs["horizon"]))
    time_grid = physics.get("time_grid")
    pool = shared.get("candidate_pool")
    if (
        physics.get("seed_sha256") != seed_receipt.artifact.sha256
        or physics.get("exact_reference_json") is not None
        or physics.get("exact_reference_sha256") is not None
        or physics.get("drive_profile") != contract.get("drive_profile")
        or lock.get("code_revision") != order.executor_overlay.overlay_revision
        or not isinstance(time_grid, Mapping)
        or int(time_grid.get("point_count", -1)) != len(times)
        or time_grid.get("ordered_values_sha256") != _canonical_sha(times)
        or not math.isclose(float(time_grid.get("t_initial", math.nan)), times[0], rel_tol=0.0, abs_tol=1e-13)
        or not math.isclose(float(time_grid.get("t_final", math.nan)), times[-1], rel_tol=0.0, abs_tol=1e-13)
        or numerics.get("integrator_method") != contract.get("integrator")
        or numerics.get("inverse_policy") != contract.get("inverse_policy")
        or numerics.get("solve_repair") != contract.get("solve_repair_config")
        or policy.get("structural_policy") != "avqds"
        or shared.get("time_step_controller") != contract.get("time_step_controller")
        or not isinstance(pool, Mapping)
        or dict(pool) != contract.get("candidate_pool_contract")
    ):
        raise ProspectiveExecutionError("AVQDS run lock diverges from the work order")
    if _normalized_hamiltonian_contract(
        hamiltonian, seed_artifact_path=seed_receipt.artifact.path
    ) != contract.get("hamiltonian"):
        raise ProspectiveExecutionError("AVQDS Hamiltonian/drive contract diverged")
    expected_source = contract.get("candidate_pool_source")
    expected_state = contract.get("initial_state_contract")
    if (
        not isinstance(expected_state, Mapping)
        or any(initial_state.get(key) != value for key, value in expected_state.items())
        or initial_state.get("candidate_pool_source") != expected_source
        or initial_state.get("candidate_pool_complete") is not True
        or int(initial_state.get("candidate_pool_term_count", -1))
        != int(contract.get("candidate_pool_contract", {}).get("candidate_pool_raw", -2))
    ):
        raise ProspectiveExecutionError("AVQDS complete candidate-pool identity diverged")
    _pure_hubbard_gate(contract, order.reference)
    metadata = trajectory.get("metadata")
    points = trajectory.get("points")
    if (
        trajectory.get("support_patch_config") != contract.get("support_patch_config")
        or trajectory.get("solve_repair_config") != contract.get("solve_repair_config")
        or trajectory.get("controller_config") != contract.get("controller_config")
        or int(trajectory.get("point_count", -1)) != len(times)
        or not isinstance(metadata, Mapping)
        or metadata.get("replay_candidate_pool_mode") != AVQDS_POOL_MODE
        or not isinstance(points, list)
        or len(points) != len(times)
        or contract.get("fail_on_unsupported_steps") is not True
    ):
        raise ProspectiveExecutionError("AVQDS trajectory controller contract diverged")
    for point in points:
        integration = point.get("integration_to_next") if isinstance(point, Mapping) else None
        if not isinstance(point, Mapping) or (
            isinstance(integration, Mapping)
            and "not_cured" in str(integration.get("local_subdivision_reason") or "")
        ):
            raise ProspectiveExecutionError("AVQDS trajectory has an uncured step")
    if len(rows) != len(times):
        raise ProspectiveExecutionError("AVQDS trajectory has the wrong point count")
    dimension = int(contract.get("physical_hilbert_dimension", 0))
    for index, (row, expected_time) in enumerate(zip(rows, times)):
        real = row.get("psi_real") if isinstance(row, Mapping) else None
        imag = row.get("psi_imag") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not math.isclose(float(row.get("time", math.nan)), expected_time, rel_tol=0.0, abs_tol=1e-13)
            or not isinstance(real, list)
            or not isinstance(imag, list)
            or len(real) != dimension
            or len(imag) != dimension
            or not all(math.isfinite(float(value)) for value in (*real, *imag))
            or not math.isclose(
                math.fsum(float(a) ** 2 + float(b) ** 2 for a, b in zip(real, imag)),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-11,
            )
        ):
            raise ProspectiveExecutionError(f"AVQDS statevector row {index} is invalid")


@dataclass(frozen=True)
class AVQDSTrajectoryCompletionReceipt:
    action_id: str
    execution_request_sha256: str
    lane_id: str
    cell_id: str
    horizon: str
    activation_cut: float
    status: str
    run: VerifiedFile
    work_order_sha256: str
    bindings_sha256: str
    seed_identity_sha256: str
    seed_artifact_sha256: str
    seed_receipt_sha256: str
    source_epoch_identity_sha256: str
    reference_identity_sha256: str
    resume_run: VerifiedFile | None
    resume_receipt: VerifiedFile | None
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        expected_order: AVQDSTrajectoryWorkOrder,
        expected_seed_receipt: SeedQualificationReceipt,
    ) -> "AVQDSTrajectoryCompletionReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError("AVQDS receipt is outside repository") from exc
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema") != COMPLETION_RECEIPT_SCHEMA_V1:
            raise ProspectiveExecutionError("unsupported AVQDS trajectory completion receipt")
        _assert_exact_blind_runtime_payload(payload)
        self_sha = _verify_self_digest(
            payload, "completion_receipt_sha256", label="AVQDS trajectory receipt"
        )
        run = _optional_verified(payload, "run")
        if run is None:
            raise ProspectiveExecutionError("AVQDS trajectory receipt has no run")
        result = cls(
            action_id=str(payload.get("action_id") or ""),
            execution_request_sha256=str(payload.get("execution_request_sha256") or ""),
            lane_id=str(payload.get("lane_id") or ""),
            cell_id=str(payload.get("cell_id") or ""),
            horizon=str(payload.get("horizon") or ""),
            activation_cut=float(payload.get("activation_cut")),
            status=str(payload.get("status") or ""),
            run=run,
            work_order_sha256=str(payload.get("work_order_sha256") or ""),
            bindings_sha256=str(payload.get("bindings_sha256") or ""),
            seed_identity_sha256=str(payload.get("seed_identity_sha256") or ""),
            seed_artifact_sha256=str(payload.get("seed_artifact_sha256") or ""),
            seed_receipt_sha256=str(payload.get("seed_receipt_sha256") or ""),
            source_epoch_identity_sha256=str(payload.get("source_epoch_identity_sha256") or ""),
            reference_identity_sha256=str(payload.get("reference_identity_sha256") or ""),
            resume_run=_optional_verified(payload, "resume_run"),
            resume_receipt=_optional_verified(payload, "resume_receipt"),
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        if (
            result.action_id != expected_order.request.action_id
            or result.execution_request_sha256 != expected_order.request.digest
            or result.lane_id != expected_order.request.lane_id
            or result.cell_id != expected_order.bindings.cell_id
            or result.horizon != expected_order.request.inputs["horizon"]
            or not math.isclose(result.activation_cut, float(expected_order.request.inputs["activation_cut"]), rel_tol=0.0, abs_tol=0.0)
            or result.status != COMPLETED_UNSCORED
            or result.run.path != expected_order.run_path
            or result.work_order_sha256 != expected_order.work_order_sha256
            or result.bindings_sha256 != expected_order.bindings.digest
            or result.seed_identity_sha256 != expected_order.seed.identity_sha256
            or result.seed_artifact_sha256 != expected_seed_receipt.artifact.sha256
            or result.seed_receipt_sha256 != expected_seed_receipt.receipt_sha256
            or result.source_epoch_identity_sha256
            != expected_order.bindings.source_epoch_identity_sha256
            or result.reference_identity_sha256
            != expected_order.bindings.reference_identity_sha256
            or result.resume_run != expected_order.resume_run
            or result.resume_receipt != expected_order.resume_receipt
            or payload.get("method") != ARM_AVQDS
            or payload.get("scientific_qualification") is not None
            or payload.get("score_receipt") is not None
            or payload.get("scoring_required_before_qualification") is not True
            or payload.get("remote_raw_evidence_policy") != "preserve"
            or payload.get("remote_cleanup_authorized") is not False
            or payload.get("qiskit_compilation_used") is not False
            or payload.get("exact_reference_runtime_artifacts_present") is not False
            or payload.get("exact_reference_binding_mode") != "opaque_identity_only"
            or payload.get("action_authorization_sha256")
            != expected_order.action_authorization_sha256
        ):
            raise ProspectiveExecutionError("AVQDS trajectory receipt binding diverged")
        result.run.verify(root)
        validate_raw_run(root / result.run.path, order=expected_order, seed_receipt=expected_seed_receipt)
        return result


def _seal_completion(
    order: AVQDSTrajectoryWorkOrder,
    *,
    root: Path,
    seed_receipt: SeedQualificationReceipt,
) -> AVQDSTrajectoryCompletionReceipt:
    run = VerifiedFile(order.run_path, _file_sha(root / order.run_path))
    core = {
        "schema": COMPLETION_RECEIPT_SCHEMA_V1,
        "action_id": order.request.action_id,
        "execution_request_sha256": order.request.digest,
        "lane_id": order.request.lane_id,
        "cell_id": order.bindings.cell_id,
        "method": ARM_AVQDS,
        "horizon": order.request.inputs["horizon"],
        "activation_cut": float(order.request.inputs["activation_cut"]),
        "ladder_index": int(order.request.inputs["ladder_index"]),
        "status": COMPLETED_UNSCORED,
        "run": run.to_json_dict(),
        "work_order_sha256": order.work_order_sha256,
        "bindings_sha256": order.bindings.digest,
        "action_authorization_sha256": order.action_authorization_sha256,
        "seed_identity_sha256": order.seed.identity_sha256,
        "seed_artifact_sha256": seed_receipt.artifact.sha256,
        "seed_receipt_sha256": seed_receipt.receipt_sha256,
        "source_epoch_identity_sha256": order.bindings.source_epoch_identity_sha256,
        "reference_identity_sha256": order.bindings.reference_identity_sha256,
        "exact_reference_runtime_artifacts_present": False,
        "exact_reference_binding_mode": "opaque_identity_only",
        "executor_overlay_identity_sha256": order.executor_overlay.identity_sha256,
        "executor_overlay_revision": order.executor_overlay.overlay_revision,
        "executor_overlay_files_sha256": order.executor_overlay.files_sha256,
        "composite_source_identity_sha256": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "resume_from_action_id": order.request.inputs.get("resume_from_action_id"),
        "resume_run": None if order.resume_run is None else order.resume_run.to_json_dict(),
        "resume_receipt": (
            None if order.resume_receipt is None else order.resume_receipt.to_json_dict()
        ),
        "scientific_qualification": None,
        "score_receipt": None,
        "scoring_required_before_qualification": True,
        "remote_raw_evidence_policy": "preserve",
        "remote_cleanup_authorized": False,
        "qiskit_compilation_used": False,
    }
    payload = {**core, "completion_receipt_sha256": _canonical_sha(core)}
    target = root / order.completion_receipt_path
    _atomic_write_immutable(target, _pretty_bytes(payload))
    return AVQDSTrajectoryCompletionReceipt.load(
        target,
        repository_root=root,
        expected_order=order,
        expected_seed_receipt=seed_receipt,
    )


def _default_execute(order: AVQDSTrajectoryWorkOrder, root: Path) -> None:
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import main

    if os.environ.get(VERIFIED_PACKAGED_SOURCE_ENV) != (
        order.executor_overlay.composite_source_identity_sha256
    ):
        raise ProspectiveExecutionError(
            "direct AVQDS execution requires the verified packaged-source marker"
        )
    prior = Path.cwd()
    try:
        os.chdir(root)
        result = main(list(order.runner_argv))
    finally:
        os.chdir(prior)
    if result not in {None, 0}:
        raise ProspectiveExecutionError(f"AVQDS runner returned status {result}")


def materialization_request(request: ExecutionRequest) -> dict[str, Any]:
    """Resolve a scientific AVQDS action into an inert package recipe."""

    if not isinstance(request, ExecutionRequest):
        raise ProspectiveExecutionError("AVQDS materialization requires a typed request")
    try:
        bindings = ProspectiveScientificBindings.from_json_dict(request.inputs)
        seed, reference = _registered_subjects(bindings)
        horizon = str(request.inputs.get("horizon") or "")
        expected = prospective_run_action_id(
            bindings,
            arm=ARM_AVQDS,
            horizon=horizon,
            activation_cut=float(request.inputs.get("activation_cut")),
        )
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError("invalid AVQDS materialization request") from exc
    if (
        request.kind != KIND_RUN
        or request.arm != ARM_AVQDS
        or request.binding_id != EXEC_PROSPECTIVE_AVQDS
        or request.action_id != expected
        or not request.exact_reference_blind
    ):
        raise ProspectiveExecutionError("request is not a scientific AVQDS run")
    required = [
        "action_registry",
        "source_epoch_receipt",
        "seed_qualification_receipt",
    ]
    if horizon == HORIZON_FULL:
        required.extend(["prefix_run", "prefix_run_lifecycle_receipt"])
    return {
        "schema": "paper_ii_avqds_trajectory_operational_materialization_request_v1",
        "status": "operational_materialization_ready",
        "execution_request": request.to_json_dict(),
        "execution_request_sha256": request.digest,
        "bindings_sha256": bindings.digest,
        "cell_identity_sha256": bindings.cell_identity_sha256,
        "seed_identity_sha256": seed.identity_sha256,
        "reference_identity_sha256": reference.identity_sha256,
        "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
        "work_order_builder": (
            "pipelines.time_dynamics.prospective_avqds_trajectory_worker:build_work_order"
        ),
        "package_builder": (
            "pipelines.time_dynamics.prospective_avqds_trajectory_chtc_package:"
            "materialize_avqds_trajectory_package"
        ),
        "runner_module": (
            "pipelines.time_dynamics.runners.run_prospective_avqds_trajectory"
        ),
        "required_receipts": required,
        "canonical_execution_mode": (
            "digest_bound_exact_blind_scientific_avqds_v1"
        ),
        "scientific_status_on_success": COMPLETED_UNSCORED,
        "scheduler_contacted": False,
        "scheduler_submission_authorized": False,
        "production_submission_enabled": False,
        "qiskit_compilation_requested": False,
    }


def materialize_avqds_trajectory(
    order: AVQDSTrajectoryWorkOrder,
    *,
    repository_root: str | Path,
    execute: Callable[[AVQDSTrajectoryWorkOrder, Path], None] = _default_execute,
) -> AVQDSTrajectoryCompletionReceipt:
    """Execute once, or finish a digest-valid interrupted ``.part`` replay."""

    root = Path(repository_root).resolve(strict=True)
    source_payload = json.loads(
        (root / order.source_epoch_receipt.path).read_text(encoding="utf-8")
    )
    epoch_row = source_payload["source_epoch"]
    expected_epoch = SourceEpochIdentity(
        lane_id=str(epoch_row["lane_id"]),
        source_epoch_id=str(epoch_row["source_epoch_id"]),
        source_revision=epoch_row.get("source_revision"),
        source_archive_sha256=epoch_row.get("source_archive_sha256"),
    )
    source = _load_runtime_source_epoch_receipt(
        order.source_epoch_receipt.path,
        repository_root=root,
        expected_epoch=expected_epoch,
    )
    seed = SeedQualificationReceipt.load(
        order.seed_receipt.path,
        repository_root=root,
        expected_seed=order.seed,
        expected_source_epoch=source.source_epoch,
    )
    current = _assemble_runtime_work_order(
        request=order.request,
        bindings=order.bindings,
        action_authorization_sha256=order.action_authorization_sha256,
        source_epoch_receipt=source,
        seed_receipt=seed,
        root=root,
        executor_overlay=order.executor_overlay,
        resume_run=order.resume_run,
        resume_receipt=order.resume_receipt,
        prevalidated_subjects=(order.seed, order.reference),
    )
    if current != order:
        raise ProspectiveExecutionError("AVQDS work order changed before execution")
    final_path = root / order.run_path
    part_path = root / order.run_part_path
    receipt_path = root / order.completion_receipt_path
    if receipt_path.exists():
        return AVQDSTrajectoryCompletionReceipt.load(
            receipt_path,
            repository_root=root,
            expected_order=order,
            expected_seed_receipt=seed,
        )
    if final_path.exists() and part_path.exists():
        if _file_sha(final_path) != _file_sha(part_path):
            raise ProspectiveExecutionError("divergent final and partial AVQDS runs")
        part_path.unlink()
    if not final_path.exists():
        if not part_path.exists():
            part_path.parent.mkdir(parents=True, exist_ok=True)
            execute(order, root)
            if not part_path.is_file():
                raise ProspectiveExecutionError("AVQDS runner produced no .part artifact")
            annotate_raw_run(part_path, order=order)
        validate_raw_run(part_path, order=order, seed_receipt=seed)
        with part_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(part_path, final_path)
        directory = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    validate_raw_run(final_path, order=order, seed_receipt=seed)
    return _seal_completion(order, root=root, seed_receipt=seed)


__all__ = [
    "AVQDSTrajectoryCompletionReceipt",
    "AVQDSTrajectoryExecutorOverlayIdentity",
    "AVQDSTrajectoryWorkOrder",
    "AVQDSRuntimeBindings",
    "AVQDSRuntimeRequest",
    "COMPLETED_UNSCORED",
    "COMPLETION_RECEIPT_SCHEMA_V1",
    "EXECUTOR_OVERLAY_PATHS",
    "WORK_ORDER_SCHEMA_V1",
    "annotate_raw_run",
    "build_work_order",
    "discover_executor_overlay",
    "load_work_order",
    "materialization_request",
    "materialize_avqds_trajectory",
    "validate_canonical_avqds_request",
    "validate_raw_run",
    "write_work_order",
]
