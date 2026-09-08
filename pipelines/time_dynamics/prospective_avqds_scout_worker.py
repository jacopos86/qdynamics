"""Exact-blind AVQDS-prefix worker for prospective Paper-II bootstrap cells.

This module owns the narrow boundary between a frozen ``screen_avqds`` action
and one raw trajectory.  It does not score or qualify the trajectory and it
does not contact a scheduler.  Success is represented only by a digest-bound
``completed_unscored`` receipt suitable for the later exact-reference join.
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
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    ExactReferenceIdentity,
    LANE_HH,
    LANE_HUBBARD,
    SeedIdentity,
    SourceEpochIdentity,
    VerifiedActionEvidence,
    activation_cut_ladder,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXEC_HH_AVQDS,
    EXEC_HUBBARD_AVQDS,
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    VerifiedFile,
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
)


WORK_ORDER_SCHEMA_V1 = "paper_ii_avqds_prefix_work_order_v1"
COMPLETION_RECEIPT_SCHEMA_V1 = "paper_ii_avqds_prefix_completion_receipt_v1"
EXECUTION_CONTRACT_SCHEMA_V1 = "paper_ii_avqds_execution_contract_v1"
EXECUTION_ATTESTATION_SCHEMA_V1 = "paper_ii_avqds_execution_attestation_v1"
VERIFIED_PACKAGED_SOURCE_ENV = "PAPER_II_VERIFIED_PACKAGED_SOURCE_SHA256"
SCOUT_SEGMENT = "calibration_prefix_t0_t5_126"
SCOUT_CONTROLLER = "state_motion_5e-5"
SCOUT_STATUS = "completed_unscored"
SCOUT_POOL_MODE = "replay_family_pool"
SCOUT_SEARCH_PROFILE = "paper_declared_v1"
SCOUT_NUMERICS = "euler_ridge1e-6"
EXECUTOR_OVERLAY_SCHEMA_V1 = "paper_ii_avqds_executor_overlay_v1"
EXECUTOR_OVERLAY_PATHS = (
    "pipelines/scaffold/runtime_loader.py",
    "pipelines/time_dynamics/prospective_execution_bindings.py",
    "pipelines/time_dynamics/prospective_avqds_scout_worker.py",
    "pipelines/time_dynamics/runners/run_prospective_avqds_scout.py",
    # Keep the uncapped pool interface and lazy enumeration dependencies together.
    "pipelines/time_dynamics/paper_ii_runs.py",
    "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
    "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_config.py",
    "pipelines/time_dynamics/ap_mclachlan/support_atoms.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_structural.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_selector.py",
    "pipelines/time_dynamics/ap_mclachlan/deletion_family.py",
    "pipelines/time_dynamics/ap_mclachlan/patch_enumeration.py",
    "pipelines/time_dynamics/ap_mclachlan/patch_evaluation.py",
    "pipelines/time_dynamics/ap_mclachlan/gain_evaluation.py",
    "pipelines/time_dynamics/ap_mclachlan/structural_cache.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/__init__.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/problem.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/session.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/scheduling.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/exhaustive.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/conditional_greedy.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/block_local_exchange.py",
    "pipelines/time_dynamics/ap_mclachlan/search_methods/runtime.py",
    "pipelines/time_dynamics/ap_mclachlan/insertion_words.py",
    "pipelines/time_dynamics/generalized_exchange.py",
)
_SUPPORTED_BINDINGS = {
    LANE_HH: EXEC_HH_AVQDS,
    LANE_HUBBARD: EXEC_HUBBARD_AVQDS,
}
_SEED_LOADER_MODE_BY_LANE = {
    LANE_HH: "fixed_scaffold",
    LANE_HUBBARD: "replay_family",
}
_SCOUT_INPUT_KEYS = frozenset(
    {
        "accuracy_policy",
        "accuracy_policy_sha256",
        "activation_cut",
        "activation_ladder",
        "activation_ladder_sha256",
        "cell_id",
        "controller",
        "drive",
        "ladder_index",
        "lane_id",
        "matrix_expanded_semantic_sha256",
        "matrix_payload_sha256",
        "matrix_raw_sha256",
        "method",
        "ordered_grid",
        "physical",
        "record_statevector",
        "reference_id",
        "reference_identity_sha256",
        "seed_artifact_sha256",
        "seed_identity_sha256",
        "segment",
        "source_epoch",
        "source_epoch_identity_sha256",
    }
)
_FORBIDDEN_RUNTIME_EXACT_KEYS = frozenset(
    {
        "action_registry",
        "exact_data_schema",
        "exact_reference_json",
        "exact_reference_receipt",
        "exact_reference_receipt_sha256",
        "exact_reference_sha256",
        "external_data_path",
        "external_data_sha256",
        "external_manifest_path",
        "external_manifest_sha256",
        "materialized_reference_key",
        "reference_contract",
        "reference_data_sha256",
        "reference_manifest_sha256",
        "reference_receipt_sha256",
        "response_scale",
        "response_scales",
        "response_scales_sha256",
    }
)
_SCOUT_MATRIX_DIGESTS = {
    "matrix_raw_sha256": (
        "147b26d9c47621c1eeeba27896624025d1621f36ea084d2e0c163bc3f39979f2"
    ),
    "matrix_payload_sha256": (
        "aaab5e8c025870f972d9170acec8820ecd0092a0ca3e9c48ae39401b282aa4ed"
    ),
    "matrix_expanded_semantic_sha256": (
        "173fa1798406f224bacf223dd73073a7a03b00b5f36675bae355c0ccf766fbaf"
    ),
}
_SCOUT_ACCURACY_POLICY_SHA256 = (
    "7065dc2e535864adc1eae43923f688171a2215bd3fbf0515daf1df172f5de8ec"
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


def _assert_exact_blind_scout_payload(
    value: Any, *, path: str = "AVQDS scout runtime"
) -> None:
    """Reject exact evidence recursively from every executable scout payload."""

    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_RUNTIME_EXACT_KEYS.intersection(value)
        if forbidden:
            raise ProspectiveExecutionError(
                f"{path} exposes forbidden exact inputs: {sorted(forbidden)!r}"
            )
        for key, child in value.items():
            _assert_exact_blind_scout_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_exact_blind_scout_payload(child, path=f"{path}[{index}]")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


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
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _verify_self_digest(
    payload: Mapping[str, Any], key: str, *, label: str
) -> str:
    unsigned = dict(payload)
    declared = unsigned.pop(key, None)
    actual = _canonical_sha(unsigned)
    if declared != actual:
        raise ProspectiveExecutionError(f"{label} self digest mismatch")
    return actual


def _load_runtime_source_epoch_receipt(
    path: str | Path,
    *,
    repository_root: str | Path,
    expected_epoch: SourceEpochIdentity,
) -> SourceEpochReceipt:
    """Load the canonical receipt without opening its repository-wide inventory."""

    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "runtime source epoch receipt is outside repository"
        ) from exc
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != "paper_ii_source_epoch_receipt_v1":
        raise ProspectiveExecutionError("unsupported runtime source epoch receipt")
    self_sha = _verify_self_digest(
        payload, "source_epoch_receipt_sha256", label="runtime source epoch receipt"
    )
    epoch_row = payload.get("source_epoch")
    manifest_row = payload.get("source_tree_manifest")
    if not isinstance(epoch_row, Mapping) or not isinstance(manifest_row, Mapping):
        raise ProspectiveExecutionError("runtime source epoch receipt is incomplete")
    epoch = SourceEpochIdentity(
        lane_id=str(epoch_row.get("lane_id") or ""),
        source_epoch_id=str(epoch_row.get("source_epoch_id") or ""),
        source_revision=epoch_row.get("source_revision"),
        source_archive_sha256=epoch_row.get("source_archive_sha256"),
    )
    if (
        epoch != expected_epoch
        or epoch_row.get("source_epoch_identity_sha256") != epoch.identity_sha256
    ):
        raise ProspectiveExecutionError("runtime source epoch identity diverged")
    return SourceEpochReceipt(
        lane_id=epoch.lane_id,
        source_epoch=epoch,
        source_tree_manifest=VerifiedFile(
            str(manifest_row.get("path") or ""),
            str(manifest_row.get("sha256") or ""),
        ),
        source_tree_sha256=str(payload.get("source_tree_sha256") or ""),
        receipt_path=relative,
        receipt_sha256=_file_sha(source),
        receipt_self_sha256=self_sha,
    )


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
class ExecutorOverlayIdentity:
    """Committed executor replacements composed with an older seed epoch."""

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
    def from_mapping(cls, value: Any) -> "ExecutorOverlayIdentity":
        if not isinstance(value, Mapping):
            raise ProspectiveExecutionError("AVQDS executor overlay is missing")
        unsigned = dict(value)
        declared_identity = unsigned.pop("executor_overlay_identity_sha256", None)
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
                "base_present": bool(row["base_present"]),
                "base_sha256": base_sha,
                "overlay_sha256": overlay_sha,
            }
        files_sha = str(value.get("files_sha256") or "")
        result = cls(
            base_source_epoch_identity_sha256=str(
                value.get("base_source_epoch_identity_sha256") or ""
            ),
            base_source_revision=str(value.get("base_source_revision") or ""),
            overlay_revision=str(value.get("overlay_revision") or ""),
            files=normalized,
            files_sha256=files_sha,
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", result.base_source_epoch_identity_sha256)
            or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", result.base_source_revision)
            or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", result.overlay_revision)
            or files_sha != _canonical_sha(normalized)
            or value.get("source_composition")
            != "verified_pristine_seed_epoch_plus_committed_executor_overlay_v1"
            or value.get("composite_source_identity_sha256")
            != result.composite_source_identity_sha256
            or declared_identity != result.identity_sha256
        ):
            raise ProspectiveExecutionError("AVQDS executor overlay identity diverged")
        return result

    def verify_files(self, root: Path, *, require_git_revision: bool) -> None:
        for path in EXECUTOR_OVERLAY_PATHS:
            target = root / path
            if not target.is_file() or _file_sha(target) != self.files[path]["overlay_sha256"]:
                raise ProspectiveExecutionError(
                    f"AVQDS executor overlay bytes changed: {path}"
                )
        if require_git_revision:
            self.verify_committed_files(root)

    def verify_committed_files(self, root: Path) -> None:
        """Authenticate the frozen overlay and pristine base at their commits."""

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
            observed_base = (
                None if base_bytes is None else hashlib.sha256(base_bytes).hexdigest()
            )
            if observed_base != self.files[path]["base_sha256"]:
                raise ProspectiveExecutionError(
                    f"AVQDS pristine base digest mismatch: {path}"
                )


def discover_executor_overlay(
    root: Path, *, base_source_epoch: SourceEpochIdentity
) -> ExecutorOverlayIdentity:
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
            root,
            "show",
            f"{base_source_epoch.source_revision}:{path}",
            allow_missing=True,
        )
        base_sha = None if base_bytes is None else hashlib.sha256(base_bytes).hexdigest()
        files[path] = {
            "base_present": base_sha is not None,
            "base_sha256": base_sha,
            "overlay_sha256": overlay_sha,
        }
    result = ExecutorOverlayIdentity(
        base_source_epoch_identity_sha256=base_source_epoch.identity_sha256,
        base_source_revision=base_source_epoch.source_revision,
        overlay_revision=revision,
        files=files,
        files_sha256=_canonical_sha(files),
    )
    return ExecutorOverlayIdentity.from_mapping(result.to_json_dict())


def _source_epoch_from_request(request: ExecutionRequest) -> SourceEpochIdentity:
    raw = request.inputs.get("source_epoch")
    if not isinstance(raw, Mapping):
        raise ProspectiveExecutionError("AVQDS request has no source epoch")
    epoch = SourceEpochIdentity(
        lane_id=str(raw.get("lane_id") or ""),
        source_epoch_id=str(raw.get("source_epoch_id") or ""),
        source_revision=raw.get("source_revision"),
        source_archive_sha256=raw.get("source_archive_sha256"),
    )
    if (
        raw.get("source_epoch_identity_sha256") != epoch.identity_sha256
        or request.inputs.get("source_epoch_identity_sha256")
        != epoch.identity_sha256
        or epoch.lane_id != request.lane_id
        or epoch.source_revision is None
    ):
        raise ProspectiveExecutionError("AVQDS request source epoch binding diverged")
    return epoch


def _registered_subjects(
    request: ExecutionRequest,
) -> tuple[SeedIdentity, ExactReferenceIdentity]:
    registry = build_identity_registry()
    seeds = tuple(
        row
        for row in registry.seeds
        if row.lane_id == request.lane_id
        and row.identity_sha256 == request.inputs.get("seed_identity_sha256")
    )
    references = tuple(
        row
        for row in registry.references
        if row.lane_id == request.lane_id
        and row.identity_sha256 == request.inputs.get("reference_identity_sha256")
    )
    if len(seeds) != 1 or len(references) != 1:
        raise ProspectiveExecutionError("AVQDS request is not matrix-registered")
    seed, reference = seeds[0], references[0]
    if (
        seed.seed_id != reference.seed_id
        or reference.reference_id != request.inputs.get("reference_id")
        or reference.cell_id != request.inputs.get("cell_id")
        or dict(reference.physical) != dict(request.inputs.get("physical") or {})
        or dict(reference.drive) != dict(request.inputs.get("drive") or {})
        or dict(reference.grid) != dict(request.inputs.get("ordered_grid") or {})
    ):
        raise ProspectiveExecutionError("AVQDS request seed/reference identity diverged")
    return seed, reference


def _projected_subjects(
    request: ExecutionRequest,
    seed_payload: Mapping[str, Any],
    reference_payload: Mapping[str, Any],
) -> tuple[SeedIdentity, ExactReferenceIdentity]:
    """Reload the matrix-authenticated identities without reopening the matrix.

    Preparation performs the planner/matrix join.  Remote execution instead
    authenticates the immutable projections sealed into that work order; this
    keeps the matrix (which also names exact-reference artifacts) out of the
    exact-blind runtime archive.
    """

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
        or seed.lane_id != request.lane_id
        or reference.lane_id != request.lane_id
        or seed.seed_id != reference.seed_id
        or reference.reference_id != request.inputs.get("reference_id")
        or reference.cell_id != request.inputs.get("cell_id")
        or seed.identity_sha256 != request.inputs.get("seed_identity_sha256")
        or reference.identity_sha256
        != request.inputs.get("reference_identity_sha256")
        or dict(reference.physical) != dict(request.inputs.get("physical") or {})
        or dict(reference.drive) != dict(request.inputs.get("drive") or {})
        or dict(reference.grid) != dict(request.inputs.get("ordered_grid") or {})
    ):
        raise ProspectiveExecutionError(
            "AVQDS work-order identity projection diverged"
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


def _validate_scout_runtime_inputs(request: ExecutionRequest) -> None:
    inputs = request.inputs
    _assert_exact_blind_scout_payload(inputs)
    policy = inputs.get("accuracy_policy")
    if (
        set(inputs) != _SCOUT_INPUT_KEYS
        or not isinstance(policy, Mapping)
        or inputs.get("lane_id") != request.lane_id
        or inputs.get("accuracy_policy_sha256")
        != _SCOUT_ACCURACY_POLICY_SHA256
        or _canonical_sha(policy) != _SCOUT_ACCURACY_POLICY_SHA256
        or any(inputs.get(key) != value for key, value in _SCOUT_MATRIX_DIGESTS.items())
    ):
        raise ProspectiveExecutionError(
            "AVQDS scout runtime request contains unauthorized inputs"
        )


def _validate_prevalidated_scout_request(
    request: ExecutionRequest,
    *,
    seed: SeedIdentity,
    reference: ExactReferenceIdentity,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
) -> SourceEpochIdentity:
    """Recheck the sealed runtime request without any exact-derived registry."""

    _validate_scout_runtime_inputs(request)
    epoch = _source_epoch_from_request(request)
    ladder = request.inputs.get("activation_ladder")
    if not isinstance(ladder, Mapping):
        raise ProspectiveExecutionError("AVQDS request has no frozen ladder")
    contract = ladder.get("contract")
    ordered = ladder.get("ordered_values")
    expected_contract = {
        "start": 1.0e-2,
        "ratio": 3.0,
        "floor": 1.0e-12,
        "include_floor": True,
        "monotonicity_assumed": False,
    }
    expected_ladder = {
        "contract": expected_contract,
        "ordered_values": list(activation_cut_ladder(expected_contract)),
    }
    if (
        request.kind != "screen_avqds"
        or request.lane_id not in _SUPPORTED_BINDINGS
        or request.binding_id != _SUPPORTED_BINDINGS.get(request.lane_id)
        or not request.exact_reference_blind
        or request.execution_id != request.action_id
        or request.attempt != 0
        or request.planner_sequence != 0
        or request.requested_memory_gib is not None
        or request.arm is not None
        or request.declared_outputs
        != ("run_json", "trajectory_statevectors", "run_receipt")
        or request.inputs.get("method") != "avqds"
        or request.inputs.get("segment") != SCOUT_SEGMENT
        or request.inputs.get("controller") != SCOUT_CONTROLLER
        or request.inputs.get("record_statevector") is not True
        or int(request.inputs.get("ladder_index", -1)) != 0
        or ladder != expected_ladder
        or request.inputs.get("activation_ladder_sha256")
        != _canonical_sha(expected_ladder)
        or not math.isclose(
            float(request.inputs.get("activation_cut", math.nan)),
            float(expected_ladder["ordered_values"][0]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or source_epoch_receipt.source_epoch != epoch
        or seed_receipt.seed_id != seed.seed_id
        or seed_receipt.seed_identity_sha256 != seed.identity_sha256
        or seed_receipt.source_epoch_identity_sha256 != epoch.identity_sha256
        or seed_receipt.artifact.sha256
        != request.inputs.get("seed_artifact_sha256")
        or reference.seed_id != seed.seed_id
    ):
        raise ProspectiveExecutionError(
            "AVQDS sealed runtime request/receipt binding diverged"
        )
    return epoch


def validate_canonical_scout_request(
    request: ExecutionRequest,
    *,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
) -> tuple[SeedIdentity, ExactReferenceIdentity, SourceEpochIdentity]:
    """Authenticate a prefix action against the frozen bootstrap planner."""

    _validate_scout_runtime_inputs(request)
    if (
        request.kind != "screen_avqds"
        or request.lane_id not in _SUPPORTED_BINDINGS
        or request.binding_id != _SUPPORTED_BINDINGS.get(request.lane_id)
        or not request.exact_reference_blind
        or request.execution_id != request.action_id
        or request.attempt != 0
        or request.planner_sequence != 0
        or request.requested_memory_gib is not None
        or request.arm is not None
        or request.declared_outputs
        != ("run_json", "trajectory_statevectors", "run_receipt")
        or request.inputs.get("method") != "avqds"
        or request.inputs.get("segment") != SCOUT_SEGMENT
        or request.inputs.get("controller") != SCOUT_CONTROLLER
        or request.inputs.get("record_statevector") is not True
    ):
        raise ProspectiveExecutionError("request is not a canonical AVQDS prefix action")
    epoch = _source_epoch_from_request(request)
    seed, reference = _registered_subjects(request)
    registry = build_identity_registry()
    ladder_contract = dict(registry.activation_ladders["hh_and_hubbard_l4"])
    ladder = {
        "contract": ladder_contract,
        "ordered_values": list(activation_cut_ladder(ladder_contract)),
    }
    if (
        int(request.inputs.get("ladder_index", -1)) != 0
        or not math.isclose(
            float(request.inputs.get("activation_cut", math.nan)),
            float(ladder["ordered_values"][0]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or request.inputs.get("activation_ladder") != ladder
        or request.inputs.get("activation_ladder_sha256") != _canonical_sha(ladder)
    ):
        raise ProspectiveExecutionError("AVQDS request changed the frozen ladder")
    if (
        source_epoch_receipt.source_epoch != epoch
        or source_epoch_receipt.lane_id != request.lane_id
        or seed_receipt.seed_id != seed.seed_id
        or seed_receipt.seed_identity_sha256 != seed.identity_sha256
        or seed_receipt.source_epoch_identity_sha256 != epoch.identity_sha256
        or seed_receipt.artifact.sha256
        != request.inputs.get("seed_artifact_sha256")
    ):
        raise ProspectiveExecutionError("AVQDS request receipt binding diverged")

    evidence = BootstrapEvidence(
        seed_artifact_sha256={seed.seed_id: seed_receipt.artifact.sha256},
        source_epochs={request.lane_id: epoch},
    )
    plan = plan_bootstrap_actions(
        registry,
        evidence=evidence,
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    matches = tuple(
        resolve_bootstrap_action(action)
        for action in plan.ready_actions
        if action.action_id == request.action_id
    )
    if len(matches) != 1 or matches[0].to_json_dict() != request.to_json_dict():
        raise ProspectiveExecutionError("AVQDS request diverges from the frozen planner")
    return seed, reference, epoch


def _output_paths(request: ExecutionRequest) -> tuple[str, str, str]:
    lane = request.lane_id.replace("paper_ii_", "")
    cell = str(request.inputs["cell_id"])
    directory = Path("output/prospective_bootstrap") / lane / cell / (
        f"avqds_prefix_{request.digest[:16]}"
    )
    final = (directory / "run.json").as_posix()
    return final, f"{final}.part", (directory / "completion_receipt.json").as_posix()


def _runner_argv(
    request: ExecutionRequest,
    *,
    seed_receipt: SeedQualificationReceipt,
    executor_overlay: ExecutorOverlayIdentity,
    output_path: str,
    repository_root: Path,
) -> tuple[str, ...]:
    try:
        loader_mode = _SEED_LOADER_MODE_BY_LANE[request.lane_id]
    except KeyError as exc:
        raise ProspectiveExecutionError(
            f"unsupported AVQDS seed-loader lane: {request.lane_id!r}"
        ) from exc
    seed_payload = json.loads(
        (repository_root / seed_receipt.artifact.path).read_text(encoding="utf-8")
    )
    adapt = seed_payload.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise ProspectiveExecutionError("seed artifact has no runtime adapt_vqe block")
    parameterization_mode = str(
        adapt.get(
            "parameterization_execution_mode",
            adapt.get("parameterization_mode", ""),
        )
    ).strip()
    expected_mode = "per_pauli_term" if request.lane_id == LANE_HH else "logical_shared"
    if parameterization_mode != expected_mode:
        raise ProspectiveExecutionError("seed parameterization mode changed")
    command = build_run(
        arm="avqds",
        drive=str(request.inputs["drive"]["drive_id"]),
        horizon="t5",
        output_json=output_path,
        seed_path=seed_receipt.artifact.path,
        numerics=SCOUT_NUMERICS,
        step_control=SCOUT_CONTROLLER,
        search_profile=SCOUT_SEARCH_PROFILE,
        activation_cut=float(request.inputs["activation_cut"]),
        extra_flags=(
            "--record-statevector",
            "--require-complete-candidate-pool",
            "--loader-mode",
            loader_mode,
            "--replay-candidate-pool-mode",
            SCOUT_POOL_MODE,
            "--parameterization-mode",
            parameterization_mode,
            "--code-revision",
            executor_overlay.overlay_revision,
        ),
    )
    return command.argv()


def _contract_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _contract_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_contract_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    item = getattr(value, "item", None)
    if callable(item):
        return _contract_json_value(item())
    return str(value)


def _normalized_hamiltonian_contract(
    value: Mapping[str, Any], *, seed_artifact_path: str
) -> dict[str, Any]:
    """Remove repository-location variance while retaining full drive physics."""

    result = _contract_json_value(dict(value))
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        provenance = metadata.get("runtime_provenance")
        if isinstance(provenance, dict) and "artifact_json" in provenance:
            provenance["artifact_json"] = str(seed_artifact_path)
    return result


def _build_execution_contract(
    *,
    request: ExecutionRequest,
    seed_receipt: SeedQualificationReceipt,
    executor_overlay: ExecutorOverlayIdentity,
    runner_argv: tuple[str, ...],
    repository_root: Path,
) -> dict[str, Any]:
    """Resolve the exact live runner contract before a package can exist."""

    import pipelines.time_dynamics.runners.ap_append_from_adapt_artifact as runner

    args = runner._build_parser().parse_args(list(runner_argv))
    artifact_path = repository_root / seed_receipt.artifact.path
    runtime_input = runner._load_runtime_input_or_raise(
        artifact_path,
        loader_mode=args.loader_mode,
        tag=args.tag,
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
        replay_candidate_pool_mode=args.replay_candidate_pool_mode,
    )
    drive_config = runner._drive_config_from_args(args, runtime_input)
    hamiltonian = runner.time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=drive_config if bool(args.enable_drive) else None,
    )
    state = runner.state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=str(args.parameterization_mode),
    )
    augmentation = runner.augment_state_with_drive_aligned_generator(
        state,
        hamiltonian=hamiltonian,
        enabled=bool(args.enable_drive) and bool(args.drive_aligned_ansatz),
    )
    state = augmentation.state
    support = runner._support_patch_config_from_args(args)
    repair = runner._solve_repair_config_from_args(args)
    controller = runner.AppendControllerConfig(
        append_min_time=float(args.append_min_time),
        residual_ratio_threshold=float(args.residual_ratio_threshold),
        min_logical_parameter_count=int(args.min_logical_parameter_count),
        allow_incomplete_candidate_pool=not bool(
            args.require_complete_candidate_pool
        ),
    )
    candidate_pool = runner.build_candidate_pool_contract(
        state,
        support_config=support,
    )
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
    support_payload = support.to_json_dict()
    repair_payload = repair.to_json_dict()
    hamiltonian_payload = _normalized_hamiltonian_contract(
        hamiltonian.to_json_dict(),
        seed_artifact_path=seed_receipt.artifact.path,
    )
    state_payload = state.to_json_dict()
    guards = {
        key: support_payload.get(key)
        for key in (
            "max_joint_patch_evaluations",
            "max_certification_attempts_per_level",
            "max_certification_attempts_per_deletion_branch",
            "max_insertion_batch_size",
            "residual_ratio_threshold",
        )
    }
    time_step_controller = {
        "parameter_displacement_max": support_payload.get(
            "avqds_delta_theta_max"
        ),
        "state_motion_l2_step_max": repair_payload.get(
            "state_motion_l2_step_max"
        ),
        "composed_parameter_step_max": repair_payload.get(
            "parameter_step_max"
        ),
        "local_subdivision_enabled": bool(
            repair_payload.get("local_subdivision_enabled")
        ),
        "max_local_subdivisions": int(
            repair_payload.get("max_local_subdivisions", 0)
        ),
    }
    core = {
        "schema": EXECUTION_CONTRACT_SCHEMA_V1,
        "execution_request_sha256": request.digest,
        "runner_argv_sha256": _canonical_sha(list(runner_argv)),
        "source_epoch_identity_sha256": request.inputs[
            "source_epoch_identity_sha256"
        ],
        "composite_source_identity_sha256": (
            executor_overlay.composite_source_identity_sha256
        ),
        "activation_ladder_sha256": request.inputs[
            "activation_ladder_sha256"
        ],
        "activation_cut": float(request.inputs["activation_cut"]),
        "activation_ladder_index": int(request.inputs["ladder_index"]),
        "reporting_grid": {
            "segment": SCOUT_SEGMENT,
            "start": float(times[0]),
            "final": float(times[-1]),
            "point_count": int(len(times)),
            "ordered_values_sha256": _canonical_sha(list(times)),
            "reference_ordered_values_sha256": request.inputs[
                "ordered_grid"
            ]["ordered_prefix_sha256"],
        },
        "integrator": str(args.integrator),
        "inverse_policy": {
            "pinv_rcond": float(args.pinv_rcond),
            "ridge_lambda": float(args.ridge_lambda),
            "solve_damping": float(args.solve_damping),
        },
        "parameterization_mode": str(args.parameterization_mode),
        "record_statevector": bool(args.record_statevector),
        "require_complete_candidate_pool": bool(
            args.require_complete_candidate_pool
        ),
        "controller_config": controller.to_json_dict(),
        "support_patch_config": support_payload,
        "solve_repair_config": repair_payload,
        "paper_declared_guards": guards,
        "time_step_controller": time_step_controller,
        "fail_on_unsupported_steps": bool(args.fail_on_unsupported_steps),
        "unsupported_step_acceptance": "reject_any_uncured_not_cured_step_v1",
        "hamiltonian": hamiltonian_payload,
        "drive_profile": hamiltonian_payload.get("drive_profile"),
        "drive_aligned_ansatz": augmentation.to_json_dict(),
        "candidate_pool_contract": candidate_pool,
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
    result = dict(core)
    result["execution_contract_sha256"] = _canonical_sha(core)
    return result


@dataclass(frozen=True)
class AVQDSPrefixWorkOrder:
    request: ExecutionRequest
    seed: SeedIdentity
    reference: ExactReferenceIdentity
    source_epoch_receipt: VerifiedFile
    seed_receipt: VerifiedFile
    executor_overlay: ExecutorOverlayIdentity
    run_path: str
    run_part_path: str
    completion_receipt_path: str
    runner_argv: tuple[str, ...]
    execution_contract: Mapping[str, Any]
    work_order_sha256: str

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": WORK_ORDER_SCHEMA_V1,
            "execution_request": self.request.to_json_dict(),
            "seed_identity": self.seed.to_json_dict(),
            "reference_identity": _runtime_reference_projection(self.reference),
            "source_epoch_receipt": self.source_epoch_receipt.to_json_dict(),
            "seed_qualification_receipt": self.seed_receipt.to_json_dict(),
            "executor_overlay": self.executor_overlay.to_json_dict(),
            "composite_source_identity_sha256": (
                self.executor_overlay.composite_source_identity_sha256
            ),
            "runner": {
                "module": (
                    "pipelines.time_dynamics.runners."
                    "ap_append_from_adapt_artifact"
                ),
                "argv": list(self.runner_argv),
                "argv_sha256": _canonical_sha(list(self.runner_argv)),
            },
            "execution_contract": dict(self.execution_contract),
            "outputs": {
                "run": self.run_path,
                "run_part": self.run_part_path,
                "completion_receipt": self.completion_receipt_path,
            },
            "remote_raw_evidence_policy": "preserve",
            "scientific_status_on_success": SCOUT_STATUS,
            "scoring_required_before_qualification": True,
            "qiskit_compilation_requested": False,
            "production_submission_enabled": False,
        }
        if include_digest:
            payload["work_order_sha256"] = self.work_order_sha256
        _assert_exact_blind_scout_payload(payload)
        return payload


def build_work_order(
    request: ExecutionRequest,
    *,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    repository_root: str | Path,
    executor_overlay: ExecutorOverlayIdentity | None = None,
    _prevalidated_subjects: tuple[SeedIdentity, ExactReferenceIdentity] | None = None,
) -> AVQDSPrefixWorkOrder:
    root = Path(repository_root).resolve()
    if _prevalidated_subjects is None:
        seed, reference, epoch = validate_canonical_scout_request(
            request,
            source_epoch_receipt=source_epoch_receipt,
            seed_receipt=seed_receipt,
        )
    else:
        seed, reference = _prevalidated_subjects
        epoch = _validate_prevalidated_scout_request(
            request,
            seed=seed,
            reference=reference,
            source_epoch_receipt=source_epoch_receipt,
            seed_receipt=seed_receipt,
        )
    if _prevalidated_subjects is None:
        source_epoch_receipt.source_tree_manifest.verify(root)
    seed_receipt.artifact.verify(root)
    overlay = (
        discover_executor_overlay(root, base_source_epoch=epoch)
        if executor_overlay is None
        else ExecutorOverlayIdentity.from_mapping(executor_overlay.to_json_dict())
    )
    if (
        overlay.base_source_epoch_identity_sha256 != epoch.identity_sha256
        or overlay.base_source_revision != epoch.source_revision
    ):
        raise ProspectiveExecutionError("AVQDS overlay crosses the scientific seed epoch")
    overlay.verify_files(root, require_git_revision=(root / ".git").exists())
    run_path, part_path, receipt_path = _output_paths(request)
    argv = _runner_argv(
        request,
        seed_receipt=seed_receipt,
        executor_overlay=overlay,
        output_path=part_path,
        repository_root=root,
    )
    execution_contract = _build_execution_contract(
        request=request,
        seed_receipt=seed_receipt,
        executor_overlay=overlay,
        runner_argv=argv,
        repository_root=root,
    )
    source_file = VerifiedFile(
        source_epoch_receipt.receipt_path, source_epoch_receipt.receipt_sha256
    )
    seed_file = VerifiedFile(seed_receipt.receipt_path, seed_receipt.receipt_sha256)
    provisional = AVQDSPrefixWorkOrder(
        request=request,
        seed=seed,
        reference=reference,
        source_epoch_receipt=source_file,
        seed_receipt=seed_file,
        executor_overlay=overlay,
        run_path=run_path,
        run_part_path=part_path,
        completion_receipt_path=receipt_path,
        runner_argv=argv,
        execution_contract=execution_contract,
        work_order_sha256="",
    )
    return AVQDSPrefixWorkOrder(
        **{
            **provisional.__dict__,
            "work_order_sha256": _canonical_sha(
                provisional.to_json_dict(include_digest=False)
            ),
        }
    )


def write_work_order(path: str | Path, order: AVQDSPrefixWorkOrder) -> Path:
    target = Path(path)
    _atomic_write_immutable(target, _pretty_bytes(order.to_json_dict()))
    return target


def _required_runner_value(argv: tuple[str, ...], flag: str) -> str:
    positions = tuple(index for index, value in enumerate(argv) if value == flag)
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ProspectiveExecutionError(
            f"frozen AVQDS runner lacks one {flag} binding"
        )
    return argv[positions[0] + 1]


def _validate_frozen_runner_and_execution_contract(
    *,
    request: ExecutionRequest,
    seed_receipt: SeedQualificationReceipt,
    executor_overlay: ExecutorOverlayIdentity,
    runner_row: Mapping[str, Any],
    outputs_row: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
) -> tuple[tuple[str, ...], str, str, str]:
    """Cross-check sealed execution fields without importing the old runner."""

    if set(runner_row) != {"module", "argv", "argv_sha256"}:
        raise ProspectiveExecutionError("frozen AVQDS runner binding is malformed")
    argv_raw = runner_row.get("argv")
    if (
        runner_row.get("module")
        != "pipelines.time_dynamics.runners.ap_append_from_adapt_artifact"
        or not isinstance(argv_raw, list)
        or not argv_raw
        or any(not isinstance(value, str) or not value for value in argv_raw)
    ):
        raise ProspectiveExecutionError("frozen AVQDS runner binding diverged")
    argv = tuple(argv_raw)
    argv_sha = _canonical_sha(list(argv))
    if runner_row.get("argv_sha256") != argv_sha:
        raise ProspectiveExecutionError("frozen AVQDS runner argv digest mismatch")

    if set(outputs_row) != {"run", "run_part", "completion_receipt"}:
        raise ProspectiveExecutionError("frozen AVQDS output binding is malformed")
    expected_run, expected_part, expected_receipt = _output_paths(request)
    if dict(outputs_row) != {
        "run": expected_run,
        "run_part": expected_part,
        "completion_receipt": expected_receipt,
    }:
        raise ProspectiveExecutionError("frozen AVQDS output binding diverged")

    contract = dict(execution_contract)
    _verify_self_digest(
        contract,
        "execution_contract_sha256",
        label="frozen AVQDS execution contract",
    )
    grid = request.inputs.get("ordered_grid")
    reporting = contract.get("reporting_grid")
    inverse = contract.get("inverse_policy")
    if not all(
        isinstance(value, Mapping) for value in (grid, reporting, inverse)
    ):
        raise ProspectiveExecutionError(
            "frozen AVQDS execution contract is incomplete"
        )
    try:
        semantic_mismatch = (
            contract.get("schema") != EXECUTION_CONTRACT_SCHEMA_V1
            or contract.get("execution_request_sha256") != request.digest
            or contract.get("runner_argv_sha256") != argv_sha
            or contract.get("source_epoch_identity_sha256")
            != request.inputs.get("source_epoch_identity_sha256")
            or contract.get("composite_source_identity_sha256")
            != executor_overlay.composite_source_identity_sha256
            or contract.get("activation_ladder_sha256")
            != request.inputs.get("activation_ladder_sha256")
            or not math.isclose(
                float(contract.get("activation_cut")),
                float(request.inputs.get("activation_cut")),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or int(contract.get("activation_ladder_index"))
            != int(request.inputs.get("ladder_index"))
            or reporting.get("segment") != request.inputs.get("segment")
            or float(reporting.get("start")) != float(grid.get("start"))
            or float(reporting.get("final")) != float(grid.get("prefix_stop"))
            or int(reporting.get("point_count")) != int(grid.get("prefix_points"))
            or reporting.get("ordered_values_sha256")
            != grid.get("ordered_prefix_sha256")
            or reporting.get("reference_ordered_values_sha256")
            != grid.get("ordered_prefix_sha256")
            or contract.get("record_statevector") is not True
            or contract.get("require_complete_candidate_pool") is not True
            or contract.get("fail_on_unsupported_steps") is not True
            or contract.get("unsupported_step_acceptance")
            != "reject_any_uncured_not_cured_step_v1"
            or contract.get("qiskit_compilation_requested") is not False
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "frozen AVQDS execution contract is malformed"
        ) from exc
    if semantic_mismatch:
        raise ProspectiveExecutionError(
            "frozen AVQDS execution contract binding diverged"
        )

    try:
        runner_mismatch = (
            _required_runner_value(argv, "--artifact-json")
            != seed_receipt.artifact.path
            or _required_runner_value(argv, "--output-json") != expected_part
            or _required_runner_value(argv, "--code-revision")
            != executor_overlay.overlay_revision
            or _required_runner_value(argv, "--loader-mode")
            != _SEED_LOADER_MODE_BY_LANE[request.lane_id]
            or _required_runner_value(argv, "--replay-candidate-pool-mode")
            != SCOUT_POOL_MODE
            or _required_runner_value(argv, "--parameterization-mode")
            != contract.get("parameterization_mode")
            or _required_runner_value(argv, "--integrator")
            != contract.get("integrator")
            or float(_required_runner_value(argv, "--pinv-rcond"))
            != float(inverse.get("pinv_rcond"))
            or float(_required_runner_value(argv, "--ridge-lambda"))
            != float(inverse.get("ridge_lambda"))
            or float(_required_runner_value(argv, "--solve-damping"))
            != float(inverse.get("solve_damping"))
            or not math.isclose(
                float(_required_runner_value(argv, "--avqds-l2-cut")),
                float(request.inputs.get("activation_cut")),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(_required_runner_value(argv, "--t-final")),
                float(grid.get("prefix_stop")),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or int(_required_runner_value(argv, "--num-times"))
            != int(grid.get("prefix_points"))
            or any(
                argv.count(flag) != 1
                for flag in (
                    "--enable-drive",
                    "--record-statevector",
                    "--require-complete-candidate-pool",
                    "--fail-on-unsupported-steps",
                )
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "frozen AVQDS runner binding is malformed"
        ) from exc
    if runner_mismatch:
        raise ProspectiveExecutionError("frozen AVQDS runner binding diverged")

    return argv, expected_run, expected_part, expected_receipt


def load_frozen_submitted_work_order(
    path: str | Path, *, repository_root: str | Path
) -> tuple[AVQDSPrefixWorkOrder, SourceEpochReceipt, SeedQualificationReceipt]:
    """Authenticate a sealed submitted work order at its committed revisions.

    Unlike :func:`load_work_order`, this path never regenerates runner arguments
    or the execution contract from the current checkout.  It is intended only
    for downstream validation of an already-submitted, separately digest-bound
    package.
    """

    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "AVQDS prefix work order is outside repository"
        ) from exc
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError(
            "AVQDS prefix work order is unreadable"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != WORK_ORDER_SCHEMA_V1:
        raise ProspectiveExecutionError("unsupported AVQDS prefix work order")
    work_sha = _verify_self_digest(
        payload, "work_order_sha256", label="AVQDS prefix work order"
    )
    _assert_exact_blind_scout_payload(payload)
    request_row = payload.get("execution_request")
    seed_identity_row = payload.get("seed_identity")
    reference_identity_row = payload.get("reference_identity")
    source_row = payload.get("source_epoch_receipt")
    seed_row = payload.get("seed_qualification_receipt")
    overlay_row = payload.get("executor_overlay")
    runner_row = payload.get("runner")
    outputs_row = payload.get("outputs")
    execution_contract = payload.get("execution_contract")
    if not all(
        isinstance(row, Mapping)
        for row in (
            request_row,
            seed_identity_row,
            reference_identity_row,
            source_row,
            seed_row,
            overlay_row,
            runner_row,
            outputs_row,
            execution_contract,
        )
    ):
        raise ProspectiveExecutionError("AVQDS prefix work order is incomplete")

    request = ExecutionRequest.from_mapping(request_row)
    epoch = _source_epoch_from_request(request)
    seed_identity, reference_identity = _projected_subjects(
        request, seed_identity_row, reference_identity_row
    )
    source_file = VerifiedFile(
        str(source_row.get("path") or ""), str(source_row.get("sha256") or "")
    )
    seed_file = VerifiedFile(
        str(seed_row.get("path") or ""), str(seed_row.get("sha256") or "")
    )
    source_receipt = _load_runtime_source_epoch_receipt(
        source_file.path, repository_root=root, expected_epoch=epoch
    )
    seed_receipt = SeedQualificationReceipt.load(
        seed_file.path,
        repository_root=root,
        expected_seed=seed_identity,
        expected_source_epoch=epoch,
    )
    if (
        source_file
        != VerifiedFile(source_receipt.receipt_path, source_receipt.receipt_sha256)
        or seed_file
        != VerifiedFile(seed_receipt.receipt_path, seed_receipt.receipt_sha256)
    ):
        raise ProspectiveExecutionError("AVQDS work-order receipt file changed")
    _validate_prevalidated_scout_request(
        request,
        seed=seed_identity,
        reference=reference_identity,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
    )
    overlay = ExecutorOverlayIdentity.from_mapping(overlay_row)
    if (
        overlay.base_source_epoch_identity_sha256 != epoch.identity_sha256
        or overlay.base_source_revision != epoch.source_revision
    ):
        raise ProspectiveExecutionError("AVQDS overlay crosses the scientific seed epoch")
    overlay.verify_committed_files(root)
    argv, run_path, run_part_path, receipt_path = (
        _validate_frozen_runner_and_execution_contract(
            request=request,
            seed_receipt=seed_receipt,
            executor_overlay=overlay,
            runner_row=runner_row,
            outputs_row=outputs_row,
            execution_contract=execution_contract,
        )
    )
    result = AVQDSPrefixWorkOrder(
        request=request,
        seed=seed_identity,
        reference=reference_identity,
        source_epoch_receipt=source_file,
        seed_receipt=seed_file,
        executor_overlay=overlay,
        run_path=run_path,
        run_part_path=run_part_path,
        completion_receipt_path=receipt_path,
        runner_argv=argv,
        execution_contract=dict(execution_contract),
        work_order_sha256=work_sha,
    )
    if result.to_json_dict() != dict(payload):
        raise ProspectiveExecutionError("AVQDS prefix work order canonical form changed")
    return result, source_receipt, seed_receipt


def load_work_order(
    path: str | Path, *, repository_root: str | Path
) -> tuple[AVQDSPrefixWorkOrder, SourceEpochReceipt, SeedQualificationReceipt]:
    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    payload = json.loads(source.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != WORK_ORDER_SCHEMA_V1:
        raise ProspectiveExecutionError("unsupported AVQDS prefix work order")
    work_sha = _verify_self_digest(
        payload, "work_order_sha256", label="AVQDS prefix work order"
    )
    _assert_exact_blind_scout_payload(payload)
    request_row = payload.get("execution_request")
    seed_identity_row = payload.get("seed_identity")
    reference_identity_row = payload.get("reference_identity")
    source_row = payload.get("source_epoch_receipt")
    seed_row = payload.get("seed_qualification_receipt")
    overlay_row = payload.get("executor_overlay")
    if not all(
        isinstance(row, Mapping)
        for row in (
            request_row,
            seed_identity_row,
            reference_identity_row,
            source_row,
            seed_row,
            overlay_row,
        )
    ):
        raise ProspectiveExecutionError("AVQDS prefix work order is incomplete")
    request = ExecutionRequest.from_mapping(request_row)
    epoch = _source_epoch_from_request(request)
    seed_identity, reference_identity = _projected_subjects(
        request, seed_identity_row, reference_identity_row
    )
    source_receipt = _load_runtime_source_epoch_receipt(
        str(source_row["path"]), repository_root=root, expected_epoch=epoch
    )
    seed_receipt = SeedQualificationReceipt.load(
        str(seed_row["path"]),
        repository_root=root,
        expected_seed=seed_identity,
        expected_source_epoch=epoch,
    )
    overlay = ExecutorOverlayIdentity.from_mapping(overlay_row)
    if (
        _file_sha(root / str(source_row["path"])) != source_row.get("sha256")
        or _file_sha(root / str(seed_row["path"])) != seed_row.get("sha256")
    ):
        raise ProspectiveExecutionError("AVQDS work-order receipt file changed")
    rebuilt = build_work_order(
        request,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
        repository_root=root,
        executor_overlay=overlay,
        _prevalidated_subjects=(seed_identity, reference_identity),
    )
    if rebuilt.work_order_sha256 != work_sha or rebuilt.to_json_dict() != dict(payload):
        raise ProspectiveExecutionError("AVQDS prefix work order canonical form changed")
    return rebuilt, source_receipt, seed_receipt


def _expected_times(reference: ExactReferenceIdentity) -> tuple[float, ...]:
    grid = reference.grid
    return tuple(
        float(grid["start"]) + index * float(grid["step"])
        for index in range(int(grid["prefix_points"]))
    )


def _execution_attestation(order: AVQDSPrefixWorkOrder) -> dict[str, Any]:
    contract_sha = order.execution_contract.get("execution_contract_sha256")
    core = {
        "schema": EXECUTION_ATTESTATION_SCHEMA_V1,
        "execution_request_sha256": order.request.digest,
        "work_order_sha256": order.work_order_sha256,
        "execution_contract_sha256": contract_sha,
        "runner_argv_sha256": _canonical_sha(list(order.runner_argv)),
        "composite_source_identity_sha256": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "verified_packaged_source_marker": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "fail_on_unsupported_steps": bool(
            order.execution_contract.get("fail_on_unsupported_steps")
        ),
        "exact_reference_blind": True,
        "qiskit_compilation_used": False,
    }
    result = dict(core)
    result["execution_attestation_sha256"] = _canonical_sha(core)
    return result


def validate_raw_run(
    path: str | Path,
    *,
    order: AVQDSPrefixWorkOrder,
    seed_receipt: SeedQualificationReceipt,
    source_epoch: SourceEpochIdentity,
) -> None:
    """Verify the exact-blind raw trajectory without scoring it."""

    run_path = Path(path)
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("AVQDS raw run is unreadable") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != RUNNER_SCHEMA_V1:
        raise ProspectiveExecutionError("AVQDS raw run has an unsupported schema")
    contract = order.execution_contract
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema") != EXECUTION_CONTRACT_SCHEMA_V1
        or _verify_self_digest(
            contract,
            "execution_contract_sha256",
            label="AVQDS execution contract",
        )
        != contract.get("execution_contract_sha256")
        or contract.get("execution_request_sha256") != order.request.digest
        or contract.get("source_epoch_identity_sha256")
        != source_epoch.identity_sha256
        or contract.get("composite_source_identity_sha256")
        != order.executor_overlay.composite_source_identity_sha256
    ):
        raise ProspectiveExecutionError("AVQDS execution contract diverged")
    if payload.get("paper_ii_execution_attestation") != _execution_attestation(
        order
    ):
        raise ProspectiveExecutionError(
            "AVQDS raw run lacks its packaged-source execution attestation"
        )
    forbidden = {
        "scientific_qualification",
        "score_receipt",
        "qualification_status",
        "exact_reference_receipt",
    }
    if forbidden.intersection(payload):
        raise ProspectiveExecutionError("raw AVQDS output contains premature scoring")
    flow = payload.get("decision_data_flow")
    lock = payload.get("run_lock")
    rows = payload.get("plot_rows")
    trajectory = payload.get("trajectory")
    initial_state = payload.get("initial_state")
    hamiltonian = payload.get("hamiltonian")
    drive_augmentation = payload.get("drive_aligned_ansatz")
    if (
        not isinstance(flow, Mapping)
        or not isinstance(lock, Mapping)
        or not isinstance(rows, list)
        or not isinstance(trajectory, Mapping)
        or not isinstance(initial_state, Mapping)
        or not isinstance(hamiltonian, Mapping)
        or not isinstance(drive_augmentation, Mapping)
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
    expected_times = _expected_times(order.reference)
    grid_contract = contract.get("reporting_grid")
    if (
        not isinstance(grid_contract, Mapping)
        or grid_contract.get("segment") != SCOUT_SEGMENT
        or int(grid_contract.get("point_count", -1)) != len(expected_times)
        or not math.isclose(
            float(grid_contract.get("start", math.nan)),
            expected_times[0],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(grid_contract.get("final", math.nan)),
            expected_times[-1],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or grid_contract.get("ordered_values_sha256")
        != _canonical_sha(list(expected_times))
        or grid_contract.get("reference_ordered_values_sha256")
        != order.reference.grid.get("ordered_prefix_sha256")
    ):
        raise ProspectiveExecutionError("AVQDS full prefix horizon diverged")
    time_grid = physics.get("time_grid")
    inverse = numerics.get("inverse_policy")
    pool = shared.get("candidate_pool")
    expected_inverse = contract.get("inverse_policy")
    expected_guards = contract.get("paper_declared_guards")
    expected_controller = contract.get("time_step_controller")
    if (
        physics.get("seed_sha256") != seed_receipt.artifact.sha256
        or physics.get("exact_reference_json") is not None
        or physics.get("exact_reference_sha256") is not None
        or lock.get("code_revision") != order.executor_overlay.overlay_revision
        or not isinstance(time_grid, Mapping)
        or int(time_grid.get("point_count", -1)) != len(expected_times)
        or time_grid.get("ordered_values_sha256")
        != order.reference.grid.get("ordered_prefix_sha256")
        or numerics.get("integrator_method") != contract.get("integrator")
        or not isinstance(inverse, Mapping)
        or dict(inverse) != expected_inverse
        or numerics.get("solve_repair") != contract.get("solve_repair_config")
        or policy.get("structural_policy")
        != contract.get("support_patch_config", {}).get("dynamics_policy")
        or policy.get("guards") != expected_guards
        or shared.get("time_step_controller") != expected_controller
        or not isinstance(pool, Mapping)
        or dict(pool) != contract.get("candidate_pool_contract")
    ):
        raise ProspectiveExecutionError("AVQDS run lock diverges from the work order")
    if (
        _normalized_hamiltonian_contract(
            hamiltonian,
            seed_artifact_path=seed_receipt.artifact.path,
        )
        != contract.get("hamiltonian")
        or physics.get("drive_profile") != contract.get("drive_profile")
        or drive_augmentation != contract.get("drive_aligned_ansatz")
    ):
        raise ProspectiveExecutionError("AVQDS full drive contract diverges")
    expected_state = contract.get("initial_state_contract")
    expected_source = contract.get("candidate_pool_source")
    if (
        not isinstance(expected_state, Mapping)
        or any(initial_state.get(key) != value for key, value in expected_state.items())
        or initial_state.get("candidate_pool_source") != expected_source
        or initial_state.get("parameterization_mode")
        != contract.get("parameterization_mode")
        or initial_state.get("candidate_pool_complete") is not True
    ):
        raise ProspectiveExecutionError(
            "AVQDS parameterization or complete candidate-pool identity diverges"
        )
    support_contract = contract.get("support_patch_config")
    repair_contract = contract.get("solve_repair_config")
    metadata = trajectory.get("metadata")
    points = trajectory.get("points")
    if (
        trajectory.get("support_patch_config") != support_contract
        or trajectory.get("solve_repair_config") != repair_contract
        or trajectory.get("controller_config") != contract.get("controller_config")
        or int(trajectory.get("point_count", -1)) != len(expected_times)
        or not isinstance(metadata, Mapping)
        or metadata.get("parameterization_mode")
        != contract.get("parameterization_mode")
        or metadata.get("replay_candidate_pool_mode") != SCOUT_POOL_MODE
        or metadata.get("solve_repair_requested")
        is not bool(repair_contract.get("enabled"))
        or metadata.get("solve_repair_config") != repair_contract
        or metadata.get("drive_aligned_ansatz_requested") is not True
        or not isinstance(points, list)
        or len(points) != len(expected_times)
    ):
        raise ProspectiveExecutionError(
            "AVQDS calibrated state-motion/repair contract diverges"
        )
    if contract.get("fail_on_unsupported_steps") is not True:
        raise ProspectiveExecutionError(
            "AVQDS unsupported-step failure mode is not active"
        )
    for point in points:
        if not isinstance(point, Mapping):
            raise ProspectiveExecutionError("AVQDS trajectory point is malformed")
        integration = point.get("integration_to_next")
        if isinstance(integration, Mapping) and "not_cured" in str(
            integration.get("local_subdivision_reason") or ""
        ):
            raise ProspectiveExecutionError(
                "AVQDS trajectory contains an uncured unsupported step"
            )
    if len(rows) != len(expected_times):
        raise ProspectiveExecutionError("AVQDS prefix has the wrong point count")
    hilbert_dimension = int(contract.get("physical_hilbert_dimension", 0))
    if hilbert_dimension <= 0:
        raise ProspectiveExecutionError("AVQDS physical Hilbert dimension is invalid")
    for index, (row, expected_time) in enumerate(zip(rows, expected_times)):
        if not isinstance(row, Mapping):
            raise ProspectiveExecutionError("AVQDS plot row is malformed")
        real = row.get("psi_real")
        imag = row.get("psi_imag")
        if (
            not math.isclose(float(row.get("time", math.nan)), expected_time, rel_tol=0.0, abs_tol=1.0e-13)
            or not isinstance(real, list)
            or not isinstance(imag, list)
            or len(real) != hilbert_dimension
            or len(imag) != hilbert_dimension
            or not all(math.isfinite(float(value)) for value in (*real, *imag))
            or not math.isclose(
                math.fsum(float(a) ** 2 + float(b) ** 2 for a, b in zip(real, imag)),
                1.0,
                rel_tol=1.0e-9,
                abs_tol=1.0e-11,
            )
        ):
            raise ProspectiveExecutionError(
                f"AVQDS statevector row {index} is invalid"
            )


@dataclass(frozen=True)
class AVQDSPrefixCompletionReceipt:
    action_id: str
    execution_request_sha256: str
    lane_id: str
    cell_id: str
    status: str
    run: VerifiedFile
    work_order_sha256: str
    seed_identity_sha256: str
    seed_artifact_sha256: str
    seed_receipt_sha256: str
    source_epoch_identity_sha256: str
    source_epoch_receipt_sha256: str
    executor_overlay_identity_sha256: str
    executor_overlay_revision: str
    composite_source_identity_sha256: str
    activation_ladder_sha256: str
    activation_cut: float
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    def to_bootstrap_evidence(self) -> VerifiedActionEvidence:
        return VerifiedActionEvidence(
            action_id=self.action_id,
            run_path=self.run.path,
            trajectory_sha256=self.run.sha256,
            completion_receipt_sha256=self.receipt_sha256,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        expected_order: AVQDSPrefixWorkOrder,
        expected_seed_receipt: SeedQualificationReceipt,
        expected_source_epoch: SourceEpochIdentity,
    ) -> "AVQDSPrefixCompletionReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        payload = json.loads(source.resolve(strict=True).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema") != COMPLETION_RECEIPT_SCHEMA_V1:
            raise ProspectiveExecutionError("unsupported AVQDS completion receipt")
        self_sha = _verify_self_digest(
            payload,
            "completion_receipt_sha256",
            label="AVQDS completion receipt",
        )
        run_row = payload.get("run")
        if not isinstance(run_row, Mapping):
            raise ProspectiveExecutionError("AVQDS completion receipt has no run")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError("AVQDS receipt is outside repository") from exc
        result = cls(
            action_id=str(payload.get("action_id") or ""),
            execution_request_sha256=str(payload.get("execution_request_sha256") or ""),
            lane_id=str(payload.get("lane_id") or ""),
            cell_id=str(payload.get("cell_id") or ""),
            status=str(payload.get("status") or ""),
            run=VerifiedFile(str(run_row.get("path") or ""), str(run_row.get("sha256") or "")),
            work_order_sha256=str(payload.get("work_order_sha256") or ""),
            seed_identity_sha256=str(payload.get("seed_identity_sha256") or ""),
            seed_artifact_sha256=str(payload.get("seed_artifact_sha256") or ""),
            seed_receipt_sha256=str(payload.get("seed_receipt_sha256") or ""),
            source_epoch_identity_sha256=str(payload.get("source_epoch_identity_sha256") or ""),
            source_epoch_receipt_sha256=str(payload.get("source_epoch_receipt_sha256") or ""),
            executor_overlay_identity_sha256=str(
                payload.get("executor_overlay_identity_sha256") or ""
            ),
            executor_overlay_revision=str(payload.get("executor_overlay_revision") or ""),
            composite_source_identity_sha256=str(
                payload.get("composite_source_identity_sha256") or ""
            ),
            activation_ladder_sha256=str(payload.get("activation_ladder_sha256") or ""),
            activation_cut=float(payload.get("activation_cut")),
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        if (
            result.action_id != expected_order.request.action_id
            or result.execution_request_sha256 != expected_order.request.digest
            or result.lane_id != expected_order.request.lane_id
            or result.cell_id != expected_order.request.inputs.get("cell_id")
            or result.status != SCOUT_STATUS
            or result.run.path != expected_order.run_path
            or result.work_order_sha256 != expected_order.work_order_sha256
            or result.seed_identity_sha256 != expected_order.seed.identity_sha256
            or result.seed_artifact_sha256 != expected_seed_receipt.artifact.sha256
            or result.seed_receipt_sha256 != expected_seed_receipt.receipt_sha256
            or result.source_epoch_identity_sha256 != expected_source_epoch.identity_sha256
            or result.source_epoch_receipt_sha256
            != expected_order.source_epoch_receipt.sha256
            or result.executor_overlay_identity_sha256
            != expected_order.executor_overlay.identity_sha256
            or result.executor_overlay_revision
            != expected_order.executor_overlay.overlay_revision
            or result.composite_source_identity_sha256
            != expected_order.executor_overlay.composite_source_identity_sha256
            or result.activation_ladder_sha256
            != expected_order.request.inputs.get("activation_ladder_sha256")
            or not math.isclose(
                result.activation_cut,
                float(expected_order.request.inputs["activation_cut"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or payload.get("scientific_qualification") is not None
            or payload.get("score_receipt") is not None
            or payload.get("scoring_required_before_qualification") is not True
            or payload.get("remote_raw_evidence_policy") != "preserve"
            or payload.get("remote_cleanup_authorized") is not False
            or payload.get("qiskit_compilation_used") is not False
        ):
            raise ProspectiveExecutionError("AVQDS completion receipt binding diverged")
        result.run.verify(root)
        validate_raw_run(
            root / result.run.path,
            order=expected_order,
            seed_receipt=expected_seed_receipt,
            source_epoch=expected_source_epoch,
        )
        return result


def _seal_completion(
    order: AVQDSPrefixWorkOrder,
    *,
    root: Path,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
) -> AVQDSPrefixCompletionReceipt:
    run = VerifiedFile(order.run_path, _file_sha(root / order.run_path))
    core = {
        "schema": COMPLETION_RECEIPT_SCHEMA_V1,
        "action_id": order.request.action_id,
        "execution_request_sha256": order.request.digest,
        "lane_id": order.request.lane_id,
        "cell_id": order.request.inputs["cell_id"],
        "status": SCOUT_STATUS,
        "run": run.to_json_dict(),
        "work_order_sha256": order.work_order_sha256,
        "seed_identity_sha256": order.seed.identity_sha256,
        "seed_artifact_sha256": seed_receipt.artifact.sha256,
        "seed_receipt_sha256": seed_receipt.receipt_sha256,
        "source_epoch_identity_sha256": source_epoch_receipt.source_epoch.identity_sha256,
        "source_epoch_receipt_sha256": source_epoch_receipt.receipt_sha256,
        "executor_overlay_identity_sha256": order.executor_overlay.identity_sha256,
        "executor_overlay_revision": order.executor_overlay.overlay_revision,
        "executor_overlay_files_sha256": order.executor_overlay.files_sha256,
        "composite_source_identity_sha256": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "activation_ladder_sha256": order.request.inputs["activation_ladder_sha256"],
        "activation_cut": float(order.request.inputs["activation_cut"]),
        "scientific_qualification": None,
        "score_receipt": None,
        "scoring_required_before_qualification": True,
        "remote_raw_evidence_policy": "preserve",
        "remote_cleanup_authorized": False,
        "qiskit_compilation_used": False,
    }
    receipt_payload = dict(core)
    receipt_payload["completion_receipt_sha256"] = _canonical_sha(core)
    receipt_path = root / order.completion_receipt_path
    _atomic_write_immutable(receipt_path, _pretty_bytes(receipt_payload))
    return AVQDSPrefixCompletionReceipt.load(
        receipt_path,
        repository_root=root,
        expected_order=order,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=source_epoch_receipt.source_epoch,
    )


def _default_execute(order: AVQDSPrefixWorkOrder, root: Path) -> None:
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import main

    packaged_source_marker = os.environ.get(VERIFIED_PACKAGED_SOURCE_ENV, "")
    if (
        packaged_source_marker
        != order.executor_overlay.composite_source_identity_sha256
    ):
        raise ProspectiveExecutionError(
            "direct AVQDS execution is forbidden without the verified "
            "packaged-source identity marker"
        )
    prior = Path.cwd()
    try:
        os.chdir(root)
        result = main(list(order.runner_argv))
    finally:
        os.chdir(prior)
    if result not in {None, 0}:
        raise ProspectiveExecutionError(f"AVQDS runner returned status {result}")
    part_path = root / order.run_part_path
    try:
        payload = json.loads(part_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError(
            "AVQDS runner output is unreadable before attestation"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProspectiveExecutionError(
            "AVQDS runner output is not a JSON object before attestation"
        )
    if "paper_ii_execution_attestation" in payload:
        raise ProspectiveExecutionError(
            "AVQDS runner output unexpectedly predeclared its attestation"
        )
    attested = dict(payload)
    attested["paper_ii_execution_attestation"] = _execution_attestation(order)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{part_path.name}.", suffix=".tmp", dir=part_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_pretty_bytes(attested))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, part_path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def materialization_request(request: ExecutionRequest) -> dict[str, Any]:
    """Resolve one AVQDS action into an inert, operational package recipe."""

    if not isinstance(request, ExecutionRequest):
        raise ProspectiveExecutionError(
            "AVQDS materialization adapter requires a typed execution request"
        )
    if (
        request.kind != "screen_avqds"
        or request.binding_id not in {EXEC_HH_AVQDS, EXEC_HUBBARD_AVQDS}
        or request.lane_id not in _SUPPORTED_BINDINGS
        or _SUPPORTED_BINDINGS[request.lane_id] != request.binding_id
        or not request.exact_reference_blind
    ):
        raise ProspectiveExecutionError(
            "execution request is not an HH/Hubbard AVQDS materialization"
        )
    return {
        "schema": "paper_ii_avqds_operational_materialization_request_v1",
        "status": "operational_materialization_ready",
        "execution_request": request.to_json_dict(),
        "execution_request_sha256": request.digest,
        "work_order_builder": (
            "pipelines.time_dynamics.prospective_avqds_scout_worker:"
            "build_work_order"
        ),
        "package_builder": (
            "pipelines.time_dynamics.prospective_avqds_scout_chtc_package:"
            "materialize_avqds_prefix_package"
        ),
        "runner_module": (
            "pipelines.time_dynamics.runners.run_prospective_avqds_scout"
        ),
        "required_receipts": [
            "source_epoch_receipt",
            "seed_qualification_receipt",
        ],
        "canonical_execution_mode": "verified_packaged_source_only_v1",
        "scheduler_contacted": False,
        "scheduler_submission_authorized": False,
        "production_submission_enabled": False,
        "qiskit_compilation_requested": False,
    }


def materialize_scout(
    order: AVQDSPrefixWorkOrder,
    *,
    repository_root: str | Path,
    execute: Callable[[AVQDSPrefixWorkOrder, Path], None] = _default_execute,
) -> AVQDSPrefixCompletionReceipt:
    """Execute once, or recover an already durable raw run after a crash."""

    root = Path(repository_root).resolve()
    source_receipt = _load_runtime_source_epoch_receipt(
        order.source_epoch_receipt.path,
        repository_root=root,
        expected_epoch=_source_epoch_from_request(order.request),
    )
    seed_receipt = SeedQualificationReceipt.load(
        order.seed_receipt.path,
        repository_root=root,
        expected_seed=order.seed,
        expected_source_epoch=source_receipt.source_epoch,
    )
    rebuilt = build_work_order(
        order.request,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
        repository_root=root,
        executor_overlay=order.executor_overlay,
        _prevalidated_subjects=(order.seed, order.reference),
    )
    if rebuilt.to_json_dict() != order.to_json_dict():
        raise ProspectiveExecutionError("AVQDS work order changed before execution")
    receipt_path = root / order.completion_receipt_path
    final_path = root / order.run_path
    part_path = root / order.run_part_path
    if receipt_path.exists():
        return AVQDSPrefixCompletionReceipt.load(
            receipt_path,
            repository_root=root,
            expected_order=order,
            expected_seed_receipt=seed_receipt,
            expected_source_epoch=source_receipt.source_epoch,
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
            raise ProspectiveExecutionError("AVQDS runner did not produce its .part file")
        validate_raw_run(
            part_path,
            order=order,
            seed_receipt=seed_receipt,
            source_epoch=source_receipt.source_epoch,
        )
        with part_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(part_path, final_path)
        directory = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    validate_raw_run(
        final_path,
        order=order,
        seed_receipt=seed_receipt,
        source_epoch=source_receipt.source_epoch,
    )
    return _seal_completion(
        order,
        root=root,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
    )


__all__ = [
    "AVQDSPrefixCompletionReceipt",
    "AVQDSPrefixWorkOrder",
    "COMPLETION_RECEIPT_SCHEMA_V1",
    "EXECUTION_ATTESTATION_SCHEMA_V1",
    "EXECUTION_CONTRACT_SCHEMA_V1",
    "EXECUTOR_OVERLAY_PATHS",
    "EXECUTOR_OVERLAY_SCHEMA_V1",
    "ExecutorOverlayIdentity",
    "SCOUT_STATUS",
    "VERIFIED_PACKAGED_SOURCE_ENV",
    "WORK_ORDER_SCHEMA_V1",
    "build_work_order",
    "discover_executor_overlay",
    "load_frozen_submitted_work_order",
    "load_work_order",
    "materialization_request",
    "materialize_scout",
    "validate_canonical_scout_request",
    "validate_raw_run",
    "write_work_order",
]
