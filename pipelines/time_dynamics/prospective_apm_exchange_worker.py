"""Exact-blind full-APM/exchange worker for prospective Paper-II cells.

The qualified VQE seed belongs to a frozen scientific source epoch.  New
execution plumbing and the corrected generalized-exchange implementation are
therefore installed as a separately authenticated executor overlay: the base
archive is verified first, every replaced base file is checked against its
expected digest, and only then are the overlay bytes installed.  A run never
relabels or rebuilds the seed epoch.

This module performs no scheduling, scoring, or scientific qualification.
Its only successful terminal state is a typed ``completed_unscored`` raw
trajectory with statevectors and exact-reference-blind provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipelines.time_dynamics.ap_mclachlan.exchange_selector import (
    GENERALIZED_EXCHANGE_SELECTION_POLICY_V2,
)
from pipelines.time_dynamics.ap_mclachlan.drive_aligned import (
    augment_state_with_drive_aligned_generator,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
    build_candidate_pool_contract,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.paper_ii_runs import build_run
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    ExactReferenceIdentity,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXEC_PROSPECTIVE_APM,
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    VerifiedFile,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    HH_HUBBARD_ACTIVATION_CUT_LADDER,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_RUN,
    ProspectiveScientificBindings,
    prospective_run_action_id,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
    _build_parser as _build_apm_parser,
    _drive_config_from_args,
    _load_runtime_input_or_raise,
    _solve_repair_config_from_args,
    _support_patch_config_from_args,
)
from pipelines.time_dynamics.resume import runtime_input_from_resume_state
from pipelines.time_dynamics.run_lock import (
    RUN_LOCK_SCHEMA_V4,
    comparison_fingerprint,
    physics_fingerprint,
)


WORK_ORDER_SCHEMA_V1 = "paper_ii_apm_exchange_work_order_v1"
COMPLETION_RECEIPT_SCHEMA_V1 = (
    "paper_ii_apm_exchange_completion_receipt_v1"
)
EXECUTOR_OVERLAY_SCHEMA_V1 = "paper_ii_apm_exchange_executor_overlay_v1"
SOURCE_COMPOSITION_SCHEMA_V1 = (
    "paper_ii_apm_exchange_source_composition_v1"
)
POLICY_SCHEMA_V1 = "paper_ii_apm_exchange_execution_policy_v1"
COMPLETED_UNSCORED = "completed_unscored"

APM_NUMERICS = "euler_ridge1e-6"
APM_CONTROLLER = "state_motion_5e-5"
APM_SEARCH_PROFILE = "paper_declared_v1"
APM_COST_NORMALIZATION = "family_robust_v1"
APM_COST_NORMALIZATION_FAMILY = (
    "cumulative_admitted_structural_population_v1"
)
APM_REJECTION_CONTINUATION = "continue_original_complete_ranking_v1"

_WORKER_PATH = (
    "pipelines/time_dynamics/prospective_apm_exchange_worker.py"
)
_RUNNER_PATH = (
    "pipelines/time_dynamics/runners/run_prospective_apm_exchange.py"
)
_RUNTIME_LOADER_PATH = "pipelines/scaffold/runtime_loader.py"
_EXCHANGE_FIX_PATHS = (
    "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_selector.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_structural.py",
    "pipelines/time_dynamics/generalized_exchange.py",
)
EXECUTOR_OVERLAY_PATHS = (
    _WORKER_PATH,
    _RUNNER_PATH,
    _RUNTIME_LOADER_PATH,
    *_EXCHANGE_FIX_PATHS,
    # Keep the uncapped pool interface and lazy enumeration dependencies together.
    "pipelines/time_dynamics/paper_ii_runs.py",
    "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
    "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py",
    "pipelines/time_dynamics/ap_mclachlan/exchange_config.py",
    "pipelines/time_dynamics/ap_mclachlan/support_atoms.py",
    "pipelines/time_dynamics/ap_mclachlan/deletion_family.py",
    "pipelines/time_dynamics/ap_mclachlan/patch_enumeration.py",
    "pipelines/time_dynamics/ap_mclachlan/insertion_words.py",
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
                f"refusing to overwrite immutable APM evidence: {path}"
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


def _git_bytes(
    root: Path, *arguments: str, allow_missing: bool = False
) -> bytes | None:
    completed = subprocess.run(
        ("git", *arguments), cwd=root, check=False, capture_output=True
    )
    if completed.returncode == 0:
        return completed.stdout
    if allow_missing:
        return None
    detail = completed.stderr.decode("utf-8", errors="replace").strip()
    raise ProspectiveExecutionError(
        "cannot resolve APM executor overlay Git identity: "
        f"{detail[:500]}"
    )


@dataclass(frozen=True)
class APMExecutorOverlayIdentity:
    """Committed executor bytes composed with one pristine seed epoch."""

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
            "files": {
                path: dict(self.files[path])
                for path in sorted(self.files)
            },
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
    def from_mapping(cls, value: Any) -> "APMExecutorOverlayIdentity":
        if not isinstance(value, Mapping):
            raise ProspectiveExecutionError("APM executor overlay is missing")
        if value.get("schema") != EXECUTOR_OVERLAY_SCHEMA_V1:
            raise ProspectiveExecutionError("unsupported APM executor overlay")
        files = value.get("files")
        if not isinstance(files, Mapping) or set(files) != set(
            EXECUTOR_OVERLAY_PATHS
        ):
            raise ProspectiveExecutionError("APM executor overlay file set changed")
        normalized: dict[str, dict[str, Any]] = {}
        for path in EXECUTOR_OVERLAY_PATHS:
            row = files.get(path)
            if not isinstance(row, Mapping):
                raise ProspectiveExecutionError(
                    "APM executor overlay row is malformed"
                )
            base_sha = row.get("base_sha256")
            overlay_sha = str(row.get("overlay_sha256") or "")
            if (
                (
                    base_sha is not None
                    and not re.fullmatch(r"[0-9a-f]{64}", str(base_sha))
                )
                or not re.fullmatch(r"[0-9a-f]{64}", overlay_sha)
                or row.get("base_present") is not (base_sha is not None)
            ):
                raise ProspectiveExecutionError(
                    "APM executor overlay digest is invalid"
                )
            normalized[path] = {
                "base_present": bool(row["base_present"]),
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
        revision_pattern = r"[0-9a-f]{40}(?:[0-9a-f]{24})?"
        if (
            not re.fullmatch(
                r"[0-9a-f]{64}", result.base_source_epoch_identity_sha256
            )
            or not re.fullmatch(revision_pattern, result.base_source_revision)
            or not re.fullmatch(revision_pattern, result.overlay_revision)
            or result.files_sha256 != _canonical_sha(normalized)
            or value.get("source_composition")
            != "verified_pristine_seed_epoch_plus_committed_executor_overlay_v1"
            or value.get("composite_source_identity_sha256")
            != result.composite_source_identity_sha256
            or value.get("executor_overlay_identity_sha256")
            != result.identity_sha256
        ):
            raise ProspectiveExecutionError(
                "APM executor overlay identity diverged"
            )
        return result

    def verify_files(self, root: Path, *, require_git_revision: bool) -> None:
        for path in EXECUTOR_OVERLAY_PATHS:
            target = root / path
            if (
                not target.is_file()
                or _file_sha(target) != self.files[path]["overlay_sha256"]
            ):
                raise ProspectiveExecutionError(
                    f"APM executor overlay bytes changed: {path}"
                )
        if not require_git_revision:
            return
        resolved = _git_bytes(
            root,
            "rev-parse",
            "--verify",
            f"{self.overlay_revision}^{{commit}}",
        )
        assert resolved is not None
        if resolved.decode("ascii").strip() != self.overlay_revision:
            raise ProspectiveExecutionError("APM overlay revision is not exact")
        for path in EXECUTOR_OVERLAY_PATHS:
            overlay_bytes = _git_bytes(
                root, "show", f"{self.overlay_revision}:{path}"
            )
            assert overlay_bytes is not None
            if (
                hashlib.sha256(overlay_bytes).hexdigest()
                != self.files[path]["overlay_sha256"]
            ):
                raise ProspectiveExecutionError(
                    f"APM overlay revision/file digest mismatch: {path}"
                )
            base_bytes = _git_bytes(
                root,
                "show",
                f"{self.base_source_revision}:{path}",
                allow_missing=True,
            )
            observed_base = (
                None
                if base_bytes is None
                else hashlib.sha256(base_bytes).hexdigest()
            )
            if observed_base != self.files[path]["base_sha256"]:
                raise ProspectiveExecutionError(
                    f"APM pristine base digest mismatch: {path}"
                )


def discover_executor_overlay(
    root: Path, *, base_source_epoch: SourceEpochIdentity
) -> APMExecutorOverlayIdentity:
    """Bind the committed current executor without changing the seed epoch."""

    if base_source_epoch.source_revision is None:
        raise ProspectiveExecutionError("APM base source epoch has no revision")
    revision_raw = _git_bytes(root, "rev-parse", "HEAD")
    assert revision_raw is not None
    revision = revision_raw.decode("ascii").strip()
    files: dict[str, dict[str, Any]] = {}
    for path in EXECUTOR_OVERLAY_PATHS:
        overlay_bytes = _git_bytes(root, "show", f"{revision}:{path}")
        assert overlay_bytes is not None
        overlay_sha = hashlib.sha256(overlay_bytes).hexdigest()
        current = root / path
        if not current.is_file() or _file_sha(current) != overlay_sha:
            raise ProspectiveExecutionError(
                f"APM executor overlay is not committed at {revision}: {path}"
            )
        base_bytes = _git_bytes(
            root,
            "show",
            f"{base_source_epoch.source_revision}:{path}",
            allow_missing=True,
        )
        base_sha = (
            None
            if base_bytes is None
            else hashlib.sha256(base_bytes).hexdigest()
        )
        files[path] = {
            "base_present": base_sha is not None,
            "base_sha256": base_sha,
            "overlay_sha256": overlay_sha,
        }
    result = APMExecutorOverlayIdentity(
        base_source_epoch_identity_sha256=base_source_epoch.identity_sha256,
        base_source_revision=base_source_epoch.source_revision,
        overlay_revision=revision,
        files=files,
        files_sha256=_canonical_sha(files),
    )
    return APMExecutorOverlayIdentity.from_mapping(result.to_json_dict())


def execution_policy() -> dict[str, Any]:
    """Frozen current full-APM route, including the two repaired semantics."""

    payload = {
        "schema": POLICY_SCHEMA_V1,
        "method_slot": ARM_APM,
        "runner_structural_policy": "exchange",
        "selection_policy": GENERALIZED_EXCHANGE_SELECTION_POLICY_V2,
        "numerics": APM_NUMERICS,
        "controller": APM_CONTROLLER,
        "search_profile": APM_SEARCH_PROFILE,
        "cost_normalization": APM_COST_NORMALIZATION,
        "cost_normalization_family": APM_COST_NORMALIZATION_FAMILY,
        "post_certification_rejection": APM_REJECTION_CONTINUATION,
        "complete_ranked_family_preserved_after_rejection": True,
        "statevectors_required": True,
        "exact_reference_blind": True,
        "fs_radius_experiment_status": "superseded_deferred",
        "fs_radius_tuning_used": False,
        "qiskit_compilation": False,
    }
    payload["execution_policy_sha256"] = _canonical_sha(payload)
    return payload


def _registered_subjects(
    bindings: ProspectiveScientificBindings,
) -> tuple[SeedIdentity, ExactReferenceIdentity]:
    registry = build_identity_registry()
    references = tuple(
        row for row in registry.references if row.cell_id == bindings.cell_id
    )
    if len(references) != 1:
        raise ProspectiveExecutionError("APM action names no registered cell")
    reference = references[0]
    seeds = tuple(
        row for row in registry.seeds if row.seed_id == reference.seed_id
    )
    if len(seeds) != 1:
        raise ProspectiveExecutionError("APM action has no registered seed")
    seed = seeds[0]
    if (
        reference.lane_id != bindings.lane_id
        or reference.reference_id != bindings.reference_id
        or reference.identity_sha256 != bindings.reference_identity_sha256
        or seed.seed_id != bindings.seed_id
        or seed.identity_sha256 != bindings.seed_identity_sha256
    ):
        raise ProspectiveExecutionError("APM seed/reference identity diverged")
    return seed, reference


def _validate_apm_request_subjects(
    request: ExecutionRequest,
) -> tuple[
    ProspectiveScientificBindings,
    SeedIdentity,
    ExactReferenceIdentity,
]:
    """Authenticate planner identity before receipt-bound materialization."""

    try:
        bindings = ProspectiveScientificBindings.from_json_dict(request.inputs)
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "APM request has invalid scientific bindings"
        ) from exc
    if (
        request.kind != KIND_RUN
        or request.binding_id != EXEC_PROSPECTIVE_APM
        or request.arm != ARM_APM
        or not request.exact_reference_blind
        or request.declared_outputs
        != ("run_json", "trajectory_statevectors", "run_receipt")
        or request.lane_id != bindings.lane_id
        or request.inputs.get("arm") != ARM_APM
        or request.inputs.get("exact_reference_blind") is not True
        or request.inputs.get("record_statevector") is not True
        or request.inputs.get("scientific_action_id") != request.action_id
        or request.requested_memory_gib not in range(4, 33, 4)
        or request.inputs.get("resource_only_retry")
        is not (request.attempt == 1)
    ):
        raise ProspectiveExecutionError(
            "request is not a canonical exact-blind APM action"
        )
    horizon = str(request.inputs.get("horizon") or "")
    if horizon not in {HORIZON_PREFIX, HORIZON_FULL}:
        raise ProspectiveExecutionError("APM request has an invalid horizon")
    try:
        ladder_index = int(request.inputs.get("ladder_index"))
        cut = float(request.inputs.get("activation_cut"))
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError("APM request has an invalid rung") from exc
    if (
        ladder_index < 0
        or ladder_index >= len(HH_HUBBARD_ACTIVATION_CUT_LADDER)
        or not math.isclose(
            cut,
            HH_HUBBARD_ACTIVATION_CUT_LADDER[ladder_index],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or request.action_id
        != prospective_run_action_id(
            bindings,
            arm=ARM_APM,
            horizon=horizon,
            activation_cut=cut,
        )
    ):
        raise ProspectiveExecutionError("APM request changed the frozen ladder")

    resume = (
        request.inputs.get("resume_from_action_id"),
        request.inputs.get("resume_run_receipt_sha256"),
        request.inputs.get("resume_run_sha256"),
    )
    seeded_from_scout = request.inputs.get("seeded_from_scout_action_id")
    canonical_scout = prospective_run_action_id(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        activation_cut=cut,
    )
    if horizon == HORIZON_PREFIX:
        if any(value is not None for value in resume) or seeded_from_scout not in {
            None,
            canonical_scout,
        }:
            raise ProspectiveExecutionError(
                "APM prefix request has invalid scout/resume provenance"
            )
    else:
        canonical_prefix = prospective_run_action_id(
            bindings,
            arm=ARM_APM,
            horizon=HORIZON_PREFIX,
            activation_cut=cut,
        )
        if (
            resume[0] != canonical_prefix
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(value or ""))
                for value in resume[1:]
            )
            or seeded_from_scout is not None
        ):
            raise ProspectiveExecutionError(
                "APM full-horizon request has invalid continuation provenance"
            )

    seed, reference = _registered_subjects(bindings)
    return bindings, seed, reference


def validate_canonical_apm_request(
    request: ExecutionRequest,
    *,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
) -> tuple[
    ProspectiveScientificBindings,
    SeedIdentity,
    ExactReferenceIdentity,
    SourceEpochIdentity,
]:
    """Authenticate one planner-emitted APM prefix or continuation request."""

    bindings, seed, reference = _validate_apm_request_subjects(request)
    epoch = source_epoch_receipt.source_epoch
    if (
        source_epoch_receipt.lane_id != bindings.lane_id
        or epoch.identity_sha256 != bindings.source_epoch_identity_sha256
        or epoch.source_revision != bindings.source_revision
        or source_epoch_receipt.source_tree_sha256
        != bindings.source_tree_sha256
        or seed_receipt.seed_id != bindings.seed_id
        or seed_receipt.seed_identity_sha256 != bindings.seed_identity_sha256
        or seed_receipt.artifact.sha256 != bindings.seed_artifact_sha256
        or seed_receipt.receipt_sha256 != bindings.seed_receipt_sha256
        or seed_receipt.source_epoch_identity_sha256 != epoch.identity_sha256
    ):
        raise ProspectiveExecutionError(
            "APM source/qualified-seed binding diverged"
        )
    return bindings, seed, reference, epoch


def _output_paths(request: ExecutionRequest) -> tuple[str, str, str]:
    directory = (
        Path("output/prospective_scientific")
        / request.lane_id
        / str(request.inputs["cell_id"])
        / "apm_exchange"
        / request.digest[:20]
    )
    final = (directory / "run.json").as_posix()
    return final, f"{final}.part", (directory / "completion_receipt.json").as_posix()


def _expected_times(
    reference: ExactReferenceIdentity, *, horizon: str
) -> tuple[float, ...]:
    grid = reference.grid
    all_times = tuple(
        float(grid["start"]) + index * float(grid["step"])
        for index in range(int(grid["points"]))
    )
    if horizon == HORIZON_PREFIX:
        return all_times[: int(grid["prefix_points"])]
    return all_times[int(grid["prefix_points"]) - 1 :]


def _runner_argv(
    request: ExecutionRequest,
    *,
    reference: ExactReferenceIdentity,
    seed_receipt: SeedQualificationReceipt,
    overlay: APMExecutorOverlayIdentity,
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
        "per_pauli_term"
        if reference.model == "hubbard_holstein"
        else "logical_shared"
    )
    if parameterization != expected_parameterization:
        raise ProspectiveExecutionError("seed parameterization mode changed")
    horizon = str(request.inputs["horizon"])
    continuation_flags: tuple[str, ...] = ()
    if horizon == HORIZON_FULL:
        if resume_run is None:
            raise ProspectiveExecutionError(
                "full-horizon APM work requires its exact prefix run"
            )
        expected_times = _expected_times(reference, horizon=horizon)
        continuation_flags = (
            "--resume-from-run-json",
            resume_run.path,
            "--t-initial",
            repr(float(expected_times[0])),
            "--num-times",
            str(len(expected_times)),
        )
    command = build_run(
        arm="exchange",
        drive=str(reference.drive["drive_id"]),
        horizon=horizon,
        output_json=output_path,
        seed_path=seed_receipt.artifact.path,
        numerics=APM_NUMERICS,
        step_control=APM_CONTROLLER,
        search_profile=APM_SEARCH_PROFILE,
        activation_cut=float(request.inputs["activation_cut"]),
        extra_flags=(
            "--dynamics-policy",
            "exchange",
            "--append-cost-normalization-mode",
            APM_COST_NORMALIZATION,
            "--record-statevector",
            "--require-complete-candidate-pool",
            "--loader-mode",
            "replay_family",
            "--replay-candidate-pool-mode",
            "replay_family_pool",
            "--parameterization-mode",
            parameterization,
            "--code-revision",
            overlay.overlay_revision,
            *continuation_flags,
        ),
    )
    return command.argv()


@dataclass(frozen=True)
class APMExchangeWorkOrder:
    request: ExecutionRequest
    bindings: ProspectiveScientificBindings
    seed: SeedIdentity
    reference: ExactReferenceIdentity
    source_epoch_receipt: VerifiedFile
    seed_receipt: VerifiedFile
    executor_overlay: APMExecutorOverlayIdentity
    resume_run: VerifiedFile | None
    resume_receipt: VerifiedFile | None
    run_path: str
    run_part_path: str
    completion_receipt_path: str
    runner_argv: tuple[str, ...]
    policy: Mapping[str, Any]
    work_order_sha256: str

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": WORK_ORDER_SCHEMA_V1,
            "execution_request": self.request.to_json_dict(),
            "scientific_bindings": self.bindings.to_json_dict(),
            "seed_identity": self.seed.to_json_dict(),
            "reference_identity": self.reference.to_json_dict(),
            "base_source_epoch_receipt": self.source_epoch_receipt.to_json_dict(),
            "seed_qualification_receipt": self.seed_receipt.to_json_dict(),
            "executor_overlay": self.executor_overlay.to_json_dict(),
            "source_composition": (
                "verified_pristine_seed_epoch_plus_committed_executor_overlay_v1"
            ),
            "composite_source_identity_sha256": (
                self.executor_overlay.composite_source_identity_sha256
            ),
            "resume_run": (
                None if self.resume_run is None else self.resume_run.to_json_dict()
            ),
            "resume_receipt": (
                None
                if self.resume_receipt is None
                else self.resume_receipt.to_json_dict()
            ),
            "execution_policy": dict(self.policy),
            "runner": {
                "module": (
                    "pipelines.time_dynamics.runners."
                    "ap_append_from_adapt_artifact"
                ),
                "argv": list(self.runner_argv),
                "argv_sha256": _canonical_sha(list(self.runner_argv)),
            },
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
        return payload


def build_work_order(
    request: ExecutionRequest,
    *,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    repository_root: str | Path,
    executor_overlay: APMExecutorOverlayIdentity | None = None,
    resume_run: VerifiedFile | None = None,
    resume_receipt: VerifiedFile | None = None,
) -> APMExchangeWorkOrder:
    root = Path(repository_root).resolve()
    bindings, seed, reference, epoch = validate_canonical_apm_request(
        request,
        source_epoch_receipt=source_epoch_receipt,
        seed_receipt=seed_receipt,
    )
    source_epoch_receipt.source_tree_manifest.verify(root)
    seed_receipt.artifact.verify(root)
    overlay = (
        discover_executor_overlay(root, base_source_epoch=epoch)
        if executor_overlay is None
        else APMExecutorOverlayIdentity.from_mapping(
            executor_overlay.to_json_dict()
        )
    )
    if (
        overlay.base_source_epoch_identity_sha256 != epoch.identity_sha256
        or overlay.base_source_revision != epoch.source_revision
    ):
        raise ProspectiveExecutionError("APM overlay crosses the seed source epoch")
    overlay.verify_files(root, require_git_revision=(root / ".git").exists())

    horizon = str(request.inputs["horizon"])
    expected_resume = (
        request.inputs.get("resume_run_sha256"),
        request.inputs.get("resume_run_receipt_sha256"),
    )
    if horizon == HORIZON_PREFIX:
        if resume_run is not None or resume_receipt is not None:
            raise ProspectiveExecutionError("APM prefix work cannot carry resume files")
    else:
        if resume_run is None or resume_receipt is None:
            raise ProspectiveExecutionError(
                "APM continuation requires digest-bound prefix run and receipt"
            )
        resume_run.verify(root)
        resume_receipt.verify(root)
        if (resume_run.sha256, resume_receipt.sha256) != expected_resume:
            raise ProspectiveExecutionError("APM continuation input digests changed")

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
    provisional = APMExchangeWorkOrder(
        request=request,
        bindings=bindings,
        seed=seed,
        reference=reference,
        source_epoch_receipt=VerifiedFile(
            source_epoch_receipt.receipt_path,
            source_epoch_receipt.receipt_sha256,
        ),
        seed_receipt=VerifiedFile(
            seed_receipt.receipt_path, seed_receipt.receipt_sha256
        ),
        executor_overlay=overlay,
        resume_run=resume_run,
        resume_receipt=resume_receipt,
        run_path=run_path,
        run_part_path=part_path,
        completion_receipt_path=receipt_path,
        runner_argv=argv,
        policy=execution_policy(),
        work_order_sha256="",
    )
    return APMExchangeWorkOrder(
        **{
            **provisional.__dict__,
            "work_order_sha256": _canonical_sha(
                provisional.to_json_dict(include_digest=False)
            ),
        }
    )


def write_work_order(path: str | Path, order: APMExchangeWorkOrder) -> Path:
    target = Path(path)
    _atomic_write_immutable(target, _pretty_bytes(order.to_json_dict()))
    return target


def load_work_order(
    path: str | Path, *, repository_root: str | Path
) -> tuple[APMExchangeWorkOrder, SourceEpochReceipt, SeedQualificationReceipt]:
    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError("APM work order is outside repository") from exc
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("APM work order is unreadable") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != WORK_ORDER_SCHEMA_V1:
        raise ProspectiveExecutionError("unsupported APM work order")
    work_sha = _verify_self_digest(payload, "work_order_sha256", label="APM work order")
    request_row = payload.get("execution_request")
    source_row = payload.get("base_source_epoch_receipt")
    seed_row = payload.get("seed_qualification_receipt")
    overlay_row = payload.get("executor_overlay")
    if not all(
        isinstance(row, Mapping)
        for row in (request_row, source_row, seed_row, overlay_row)
    ):
        raise ProspectiveExecutionError("APM work order is incomplete")
    request = ExecutionRequest.from_mapping(request_row)
    bindings = ProspectiveScientificBindings.from_json_dict(request.inputs)
    seed, _reference = _registered_subjects(bindings)
    epoch_identity = str(request.inputs["source_epoch_identity_sha256"])
    source_receipt = SourceEpochReceipt.load(
        str(source_row.get("path") or ""), repository_root=root
    )
    if source_receipt.source_epoch.identity_sha256 != epoch_identity:
        raise ProspectiveExecutionError("APM work-order source epoch changed")
    seed_receipt = SeedQualificationReceipt.load(
        str(seed_row.get("path") or ""),
        repository_root=root,
        expected_seed=seed,
        expected_source_epoch=source_receipt.source_epoch,
    )
    if (
        _file_sha(root / str(source_row.get("path") or ""))
        != source_row.get("sha256")
        or _file_sha(root / str(seed_row.get("path") or ""))
        != seed_row.get("sha256")
    ):
        raise ProspectiveExecutionError("APM work-order receipt file changed")
    overlay = APMExecutorOverlayIdentity.from_mapping(overlay_row)

    def optional_verified(name: str) -> VerifiedFile | None:
        row = payload.get(name)
        if row is None:
            return None
        if not isinstance(row, Mapping):
            raise ProspectiveExecutionError(f"APM work order has malformed {name}")
        return VerifiedFile(
            str(row.get("path") or ""), str(row.get("sha256") or "")
        )

    rebuilt = build_work_order(
        request,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
        repository_root=root,
        executor_overlay=overlay,
        resume_run=optional_verified("resume_run"),
        resume_receipt=optional_verified("resume_receipt"),
    )
    if rebuilt.work_order_sha256 != work_sha or rebuilt.to_json_dict() != dict(payload):
        raise ProspectiveExecutionError("APM work order canonical form changed")
    return rebuilt, source_receipt, seed_receipt


def _source_composition(
    order: APMExchangeWorkOrder, *, source_epoch: SourceEpochIdentity
) -> dict[str, Any]:
    payload = {
        "schema": SOURCE_COMPOSITION_SCHEMA_V1,
        "source_composition": (
            "verified_pristine_seed_epoch_plus_committed_executor_overlay_v1"
        ),
        "base_seed_source_epoch": {
            "source_epoch_identity_sha256": source_epoch.identity_sha256,
            "source_epoch_id": source_epoch.source_epoch_id,
            "source_revision": source_epoch.source_revision,
            "source_archive_sha256": source_epoch.source_archive_sha256,
        },
        "executed_executor_overlay": {
            "overlay_revision": order.executor_overlay.overlay_revision,
            "executor_overlay_identity_sha256": (
                order.executor_overlay.identity_sha256
            ),
            "executor_overlay_files_sha256": order.executor_overlay.files_sha256,
        },
        "composite_source_identity_sha256": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "executor_delivery_contract": (
            "digest_bound_base_plus_executor_overlay_v1"
        ),
        "base_verification_order": "verify_pristine_base_before_overlay_install_v1",
        "overlay_file_set": list(EXECUTOR_OVERLAY_PATHS),
        "package_join_key": order.executor_overlay.composite_source_identity_sha256,
        "seed_epoch_rerun": False,
        "seed_epoch_relabelled": False,
    }
    payload["source_composition_sha256"] = _canonical_sha(payload)
    return payload


def annotate_raw_run(
    path: str | Path,
    *,
    order: APMExchangeWorkOrder,
    source_epoch: SourceEpochIdentity,
) -> None:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("APM raw run is unreadable") from exc
    if not isinstance(payload, dict):
        raise ProspectiveExecutionError("APM raw run must be an object")
    provenance = _source_composition(order, source_epoch=source_epoch)
    existing = payload.get("prospective_source_composition")
    if existing is not None and existing != provenance:
        raise ProspectiveExecutionError("APM raw source provenance changed")
    payload["prospective_source_composition"] = provenance
    target.write_bytes(_pretty_bytes(payload))


def _expected_physical_qubits(reference: ExactReferenceIdentity) -> int:
    sites = int(reference.physical["lattice_sites"])
    if reference.model == "hubbard":
        return 2 * sites
    if reference.model == "hubbard_holstein":
        phonon_levels = int(reference.physical["n_ph_max"]) + 1
        phonon_qubits_per_site = max(1, (phonon_levels - 1).bit_length())
        return 2 * sites + sites * phonon_qubits_per_site
    raise ProspectiveExecutionError("APM reference has an unsupported Hilbert space")


def _expected_runtime_contract(
    order: APMExchangeWorkOrder,
    *,
    seed_receipt: SeedQualificationReceipt,
    repository_root: Path,
) -> dict[str, Any]:
    """Rebuild the fixed pre-propagation contract from authenticated inputs."""

    try:
        args = _build_apm_parser().parse_args(list(order.runner_argv))
        replay_mode = args.replay_candidate_pool_mode
        if str(args.diagnostic_append_pool_mode) == "replay_family_pool":
            if replay_mode not in {
                None,
                "",
                "diagnostic_replay_family_pool",
                "family_pool",
                "replay_family_pool",
            }:
                raise ProspectiveExecutionError(
                    "APM work order has conflicting complete-pool modes"
                )
            replay_mode = "diagnostic_replay_family_pool"
        runtime_input = _load_runtime_input_or_raise(
            repository_root / seed_receipt.artifact.path,
            loader_mode=args.loader_mode,
            tag=args.tag,
            generator_family=str(args.generator_family),
            fallback_family=str(args.fallback_family),
            replay_candidate_pool_mode=replay_mode,
        )
        resumed = order.resume_run is not None
        if resumed:
            prior = json.loads(
                (repository_root / order.resume_run.path).read_text(
                    encoding="utf-8"
                )
            )
            resume_state = prior.get("resume_state")
            if not isinstance(resume_state, Mapping):
                raise ProspectiveExecutionError(
                    "APM continuation input has no typed resume state"
                )
            runtime_input = runtime_input_from_resume_state(
                resume_state, base_runtime_input=runtime_input
            )
        drive_config = _drive_config_from_args(args, runtime_input)
        hamiltonian = time_dependent_hamiltonian_from_runtime_input(
            runtime_input, drive_config=drive_config
        )
        state = state_from_scaffold_runtime_input(
            runtime_input,
            parameterization_mode=str(args.parameterization_mode),
        )
        if not resumed:
            state = augment_state_with_drive_aligned_generator(
                state,
                hamiltonian=hamiltonian,
                enabled=bool(args.enable_drive)
                and bool(args.drive_aligned_ansatz),
            ).state
        support = _support_patch_config_from_args(args)
        solve_repair = _solve_repair_config_from_args(args)
        candidate_pool = build_candidate_pool_contract(
            state, support_config=support
        )
    except ProspectiveExecutionError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "cannot reproduce the APM runtime contract from its seed"
        ) from exc
    return {
        "parameterization_mode": str(args.parameterization_mode),
        "family_key": str(runtime_input.resolved_problem.family_key),
        "n_ph_max": getattr(runtime_input.resolved_problem.request, "n_ph_max", None),
        "drive_profile": dict(hamiltonian.to_json_dict()["drive_profile"]),
        "inverse_policy": {
            "pinv_rcond": float(args.pinv_rcond),
            "ridge_lambda": float(args.ridge_lambda),
            "solve_damping": float(args.solve_damping),
        },
        "solve_repair": solve_repair.to_json_dict(),
        "time_step_controller": {
            "parameter_displacement_max": support.avqds_delta_theta_max,
            "state_motion_l2_step_max": solve_repair.state_motion_l2_step_max,
            "composed_parameter_step_max": solve_repair.parameter_step_max,
            "local_subdivision_enabled": bool(
                solve_repair.local_subdivision_enabled
            ),
            "max_local_subdivisions": int(
                solve_repair.max_local_subdivisions
            ),
        },
        "candidate_pool": candidate_pool,
        "support_patch_config": support.to_json_dict(),
        "guards": {
            key: getattr(support, key, None)
            for key in (
                "max_joint_patch_evaluations",
                "max_certification_attempts_per_level",
                "max_certification_attempts_per_deletion_branch",
                "max_insertion_batch_size",
                "residual_ratio_threshold",
            )
        },
    }


def validate_raw_run(
    path: str | Path,
    *,
    order: APMExchangeWorkOrder,
    seed_receipt: SeedQualificationReceipt,
    source_epoch: SourceEpochIdentity,
    repository_root: str | Path | None = None,
) -> None:
    """Validate raw APM evidence without joining any exact reference."""

    run_path = Path(path)
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("APM raw run is unreadable") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != RUNNER_SCHEMA_V1:
        raise ProspectiveExecutionError("APM raw run has an unsupported schema")
    if {
        "scientific_qualification",
        "score_receipt",
        "qualification_status",
        "exact_reference_receipt",
    }.intersection(payload):
        raise ProspectiveExecutionError("raw APM output contains premature scoring")
    flow = payload.get("decision_data_flow")
    lock = payload.get("run_lock")
    rows = payload.get("plot_rows")
    summary = payload.get("summary")
    provenance = payload.get("prospective_source_composition")
    if not all(isinstance(value, Mapping) for value in (flow, lock, summary, provenance)):
        raise ProspectiveExecutionError("APM raw run is incomplete")
    if not isinstance(rows, list) or any(
        flow.get(key) is not False
        for key in (
            "uses_reference_for_decision",
            "uses_exact_reference_for_decision",
            "uses_future_exact_forecast_for_decision",
        )
    ) or flow.get("uses_statevector_as_ideal_observable_estimator") is not True:
        raise ProspectiveExecutionError("APM controller consumed exact-reference data")
    physics = lock.get("physics")
    numerics = lock.get("numerics")
    policy = lock.get("policy")
    shared = lock.get("shared_protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (physics, numerics, policy, shared)
    ):
        raise ProspectiveExecutionError("APM run lock is incomplete")
    time_grid = physics.get("time_grid")
    inverse = numerics.get("inverse_policy")
    solve_repair = numerics.get("solve_repair")
    pool = shared.get("candidate_pool")
    controller = shared.get("time_step_controller")
    guards = policy.get("guards")
    expected_times = _expected_times(
        order.reference, horizon=str(order.request.inputs["horizon"])
    )
    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else run_path.resolve(strict=True).parents[len(Path(order.run_path).parts) - 1]
    )
    expected_runtime = _expected_runtime_contract(
        order, seed_receipt=seed_receipt, repository_root=root
    )
    expected_parameterization = expected_runtime["parameterization_mode"]
    if (
        lock.get("schema") != RUN_LOCK_SCHEMA_V4
        or physics.get("seed_sha256") != seed_receipt.artifact.sha256
        or physics.get("family_key") != expected_runtime["family_key"]
        or physics.get("n_ph_max") != expected_runtime["n_ph_max"]
        or physics.get("drive_profile") != expected_runtime["drive_profile"]
        or physics.get("exact_reference_json") is not None
        or physics.get("exact_reference_sha256") is not None
        or lock.get("code_revision") != order.executor_overlay.overlay_revision
        or not isinstance(time_grid, Mapping)
        or int(time_grid.get("point_count", -1)) != len(expected_times)
        or time_grid.get("ordered_values_sha256") != _canonical_sha(expected_times)
        or numerics.get("integrator_method") != "euler"
        or not isinstance(inverse, Mapping)
        or dict(inverse) != expected_runtime["inverse_policy"]
        or not isinstance(solve_repair, Mapping)
        or dict(solve_repair) != expected_runtime["solve_repair"]
        or policy.get("structural_policy") != "exchange"
        or not isinstance(guards, Mapping)
        or dict(guards) != expected_runtime["guards"]
        or not isinstance(pool, Mapping)
        or dict(pool) != expected_runtime["candidate_pool"]
        or not isinstance(controller, Mapping)
        or dict(controller) != expected_runtime["time_step_controller"]
        or shared.get("numerics") != numerics
        or lock.get("physics_fingerprint") != physics_fingerprint(physics)
        or shared.get("physics_fingerprint") != physics_fingerprint(physics)
        or lock.get("comparison_fingerprint")
        != comparison_fingerprint(shared)
    ):
        raise ProspectiveExecutionError("APM run lock diverges from the work order")
    support = summary.get("support_patch_config")
    initial_state = payload.get("initial_state")
    final_state = payload.get("final_state")
    if (
        summary.get("structural_selection_policy")
        != GENERALIZED_EXCHANGE_SELECTION_POLICY_V2
        or not isinstance(support, Mapping)
        or support.get("dynamics_policy") != "exchange"
        or support.get("deletion_enabled") is not True
        or support.get("cost_normalization_mode") != APM_COST_NORMALIZATION
        or support.get("controller_profile")
        != "support_patch_exchange_family_v1"
        or dict(support) != expected_runtime["support_patch_config"]
        or summary.get("parameterization_mode") != expected_parameterization
        or summary.get("candidate_pool_complete") is not True
        or int(summary.get("candidate_pool_term_count", -1))
        != int(pool["candidate_pool_deduplicated"])
        or not isinstance(initial_state, Mapping)
        or not isinstance(final_state, Mapping)
        or initial_state.get("parameterization_mode")
        != expected_parameterization
        or final_state.get("parameterization_mode")
        != expected_parameterization
    ):
        raise ProspectiveExecutionError("APM exchange policy evidence diverged")
    expected_provenance = _source_composition(order, source_epoch=source_epoch)
    if provenance != expected_provenance:
        raise ProspectiveExecutionError("APM source-composition provenance diverged")
    if len(rows) != len(expected_times):
        raise ProspectiveExecutionError("APM trajectory has the wrong point count")
    expected_dimension = 1 << _expected_physical_qubits(order.reference)
    for index, (row, expected_time) in enumerate(zip(rows, expected_times)):
        if not isinstance(row, Mapping):
            raise ProspectiveExecutionError("APM statevector row is malformed")
        real = row.get("psi_real")
        imag = row.get("psi_imag")
        if (
            not math.isclose(
                float(row.get("time", math.nan)),
                expected_time,
                rel_tol=0.0,
                abs_tol=1.0e-13,
            )
            or not isinstance(real, list)
            or not isinstance(imag, list)
            or len(real) != expected_dimension
            or len(imag) != expected_dimension
            or not all(math.isfinite(float(value)) for value in (*real, *imag))
            or not math.isclose(
                math.fsum(
                    float(a) ** 2 + float(b) ** 2
                    for a, b in zip(real, imag)
                ),
                1.0,
                rel_tol=1.0e-9,
                abs_tol=1.0e-11,
            )
        ):
            raise ProspectiveExecutionError(
                f"APM statevector row {index} is invalid"
            )


@dataclass(frozen=True)
class APMExchangeCompletionReceipt:
    action_id: str
    execution_request_sha256: str
    lane_id: str
    cell_id: str
    horizon: str
    status: str
    run: VerifiedFile
    work_order_sha256: str
    seed_identity_sha256: str
    seed_artifact_sha256: str
    seed_receipt_sha256: str
    source_epoch_identity_sha256: str
    executor_overlay_identity_sha256: str
    executor_overlay_revision: str
    composite_source_identity_sha256: str
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
        expected_order: APMExchangeWorkOrder,
        expected_seed_receipt: SeedQualificationReceipt,
        expected_source_epoch: SourceEpochIdentity,
    ) -> "APMExchangeCompletionReceipt":
        root = Path(repository_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != COMPLETION_RECEIPT_SCHEMA_V1
        ):
            raise ProspectiveExecutionError("unsupported APM completion receipt")
        self_sha = _verify_self_digest(
            payload, "completion_receipt_sha256", label="APM completion receipt"
        )

        def optional_file(name: str) -> VerifiedFile | None:
            row = payload.get(name)
            if row is None:
                return None
            if not isinstance(row, Mapping):
                raise ProspectiveExecutionError(
                    f"APM completion receipt has malformed {name}"
                )
            return VerifiedFile(
                str(row.get("path") or ""), str(row.get("sha256") or "")
            )

        run_row = payload.get("run")
        if not isinstance(run_row, Mapping):
            raise ProspectiveExecutionError("APM completion receipt has no run")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProspectiveExecutionError(
                "APM completion receipt is outside repository"
            ) from exc
        result = cls(
            action_id=str(payload.get("action_id") or ""),
            execution_request_sha256=str(
                payload.get("execution_request_sha256") or ""
            ),
            lane_id=str(payload.get("lane_id") or ""),
            cell_id=str(payload.get("cell_id") or ""),
            horizon=str(payload.get("horizon") or ""),
            status=str(payload.get("status") or ""),
            run=VerifiedFile(
                str(run_row.get("path") or ""), str(run_row.get("sha256") or "")
            ),
            work_order_sha256=str(payload.get("work_order_sha256") or ""),
            seed_identity_sha256=str(
                payload.get("seed_identity_sha256") or ""
            ),
            seed_artifact_sha256=str(
                payload.get("seed_artifact_sha256") or ""
            ),
            seed_receipt_sha256=str(payload.get("seed_receipt_sha256") or ""),
            source_epoch_identity_sha256=str(
                payload.get("source_epoch_identity_sha256") or ""
            ),
            executor_overlay_identity_sha256=str(
                payload.get("executor_overlay_identity_sha256") or ""
            ),
            executor_overlay_revision=str(
                payload.get("executor_overlay_revision") or ""
            ),
            composite_source_identity_sha256=str(
                payload.get("composite_source_identity_sha256") or ""
            ),
            resume_run=optional_file("resume_run"),
            resume_receipt=optional_file("resume_receipt"),
            receipt_path=relative,
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )
        if (
            result.action_id != expected_order.request.action_id
            or result.execution_request_sha256 != expected_order.request.digest
            or result.lane_id != expected_order.request.lane_id
            or result.cell_id != expected_order.bindings.cell_id
            or result.horizon != expected_order.request.inputs.get("horizon")
            or result.status != COMPLETED_UNSCORED
            or result.run.path != expected_order.run_path
            or result.work_order_sha256 != expected_order.work_order_sha256
            or result.seed_identity_sha256
            != expected_order.seed.identity_sha256
            or result.seed_artifact_sha256
            != expected_seed_receipt.artifact.sha256
            or result.seed_receipt_sha256
            != expected_seed_receipt.receipt_sha256
            or result.source_epoch_identity_sha256
            != expected_source_epoch.identity_sha256
            or result.executor_overlay_identity_sha256
            != expected_order.executor_overlay.identity_sha256
            or result.executor_overlay_revision
            != expected_order.executor_overlay.overlay_revision
            or result.composite_source_identity_sha256
            != expected_order.executor_overlay.composite_source_identity_sha256
            or result.resume_run != expected_order.resume_run
            or result.resume_receipt != expected_order.resume_receipt
            or payload.get("scientific_qualification") is not None
            or payload.get("score_receipt") is not None
            or payload.get("scoring_required_before_qualification") is not True
            or payload.get("remote_raw_evidence_policy") != "preserve"
            or payload.get("remote_cleanup_authorized") is not False
            or payload.get("qiskit_compilation_used") is not False
            or payload.get("fs_radius_tuning_used") is not False
        ):
            raise ProspectiveExecutionError("APM completion receipt binding diverged")
        result.run.verify(root)
        validate_raw_run(
            root / result.run.path,
            order=expected_order,
            seed_receipt=expected_seed_receipt,
            source_epoch=expected_source_epoch,
            repository_root=root,
        )
        return result


def _seal_completion(
    order: APMExchangeWorkOrder,
    *,
    root: Path,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
) -> APMExchangeCompletionReceipt:
    run = VerifiedFile(order.run_path, _file_sha(root / order.run_path))
    core = {
        "schema": COMPLETION_RECEIPT_SCHEMA_V1,
        "action_id": order.request.action_id,
        "execution_request_sha256": order.request.digest,
        "lane_id": order.request.lane_id,
        "cell_id": order.bindings.cell_id,
        "method": ARM_APM,
        "runner_structural_policy": "exchange",
        "horizon": order.request.inputs["horizon"],
        "activation_cut": float(order.request.inputs["activation_cut"]),
        "ladder_index": int(order.request.inputs["ladder_index"]),
        "status": COMPLETED_UNSCORED,
        "run": run.to_json_dict(),
        "work_order_sha256": order.work_order_sha256,
        "bindings_sha256": order.bindings.digest,
        "seed_identity_sha256": order.seed.identity_sha256,
        "seed_artifact_sha256": seed_receipt.artifact.sha256,
        "seed_receipt_sha256": seed_receipt.receipt_sha256,
        "source_epoch_identity_sha256": (
            source_epoch_receipt.source_epoch.identity_sha256
        ),
        "base_source_revision": source_epoch_receipt.source_epoch.source_revision,
        "executor_overlay_identity_sha256": (
            order.executor_overlay.identity_sha256
        ),
        "executor_overlay_revision": order.executor_overlay.overlay_revision,
        "executor_overlay_files_sha256": order.executor_overlay.files_sha256,
        "composite_source_identity_sha256": (
            order.executor_overlay.composite_source_identity_sha256
        ),
        "resume_from_action_id": order.request.inputs.get(
            "resume_from_action_id"
        ),
        "resume_run": (
            None if order.resume_run is None else order.resume_run.to_json_dict()
        ),
        "resume_receipt": (
            None
            if order.resume_receipt is None
            else order.resume_receipt.to_json_dict()
        ),
        "execution_policy": dict(order.policy),
        "scientific_qualification": None,
        "score_receipt": None,
        "scoring_required_before_qualification": True,
        "remote_raw_evidence_policy": "preserve",
        "remote_cleanup_authorized": False,
        "fs_radius_tuning_used": False,
        "qiskit_compilation_used": False,
    }
    payload = dict(core)
    payload["completion_receipt_sha256"] = _canonical_sha(core)
    receipt_path = root / order.completion_receipt_path
    _atomic_write_immutable(receipt_path, _pretty_bytes(payload))
    return APMExchangeCompletionReceipt.load(
        receipt_path,
        repository_root=root,
        expected_order=order,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=source_epoch_receipt.source_epoch,
    )


def _default_execute(order: APMExchangeWorkOrder, root: Path) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pipelines.time_dynamics.runners.ap_append_from_adapt_artifact",
            *order.runner_argv,
        ),
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        raise ProspectiveExecutionError(
            f"APM runner failed with exit status {completed.returncode}"
        )


def materialization_request(request: ExecutionRequest) -> dict[str, Any]:
    """Resolve one APM action into an inert digest-bound package recipe."""

    if not isinstance(request, ExecutionRequest):
        raise ProspectiveExecutionError(
            "APM materialization adapter requires a typed execution request"
        )
    bindings, seed, reference = _validate_apm_request_subjects(request)
    required_receipts = [
        "source_epoch_receipt",
        "seed_qualification_receipt",
    ]
    if request.inputs.get("horizon") == HORIZON_FULL:
        required_receipts.extend(
            ["prefix_run", "prefix_completion_receipt"]
        )
    return {
        "schema": "paper_ii_apm_exchange_operational_materialization_request_v1",
        "status": "operational_materialization_ready",
        "execution_request": request.to_json_dict(),
        "execution_request_sha256": request.digest,
        "cell_identity_sha256": bindings.cell_identity_sha256,
        "seed_identity_sha256": seed.identity_sha256,
        "reference_identity_sha256": reference.identity_sha256,
        "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
        "work_order_builder": (
            "pipelines.time_dynamics.prospective_apm_exchange_worker:"
            "build_work_order"
        ),
        "package_builder": (
            "pipelines.time_dynamics.prospective_apm_exchange_chtc_package:"
            "materialize_apm_exchange_package"
        ),
        "runner_module": (
            "pipelines.time_dynamics.runners.run_prospective_apm_exchange"
        ),
        "required_receipts": required_receipts,
        "canonical_execution_mode": (
            "digest_bound_qualified_seed_base_plus_executor_overlay_v1"
        ),
        "scientific_status_on_success": COMPLETED_UNSCORED,
        "exact_reference_blind": True,
        "fixed_exchange_policy": execution_policy(),
        "scheduler_contacted": False,
        "scheduler_submission_authorized": False,
        "production_submission_enabled": False,
        "qiskit_compilation_requested": False,
    }


def materialize_apm_exchange(
    order: APMExchangeWorkOrder,
    *,
    repository_root: str | Path,
    execute: Callable[[APMExchangeWorkOrder, Path], None] | None = None,
) -> APMExchangeCompletionReceipt:
    """Execute one work order and atomically seal unscored raw evidence."""

    root = Path(repository_root).resolve(strict=True)
    source_receipt = SourceEpochReceipt.load(
        order.source_epoch_receipt.path, repository_root=root
    )
    seed_receipt = SeedQualificationReceipt.load(
        order.seed_receipt.path,
        repository_root=root,
        expected_seed=order.seed,
        expected_source_epoch=source_receipt.source_epoch,
    )
    current = build_work_order(
        order.request,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
        repository_root=root,
        executor_overlay=order.executor_overlay,
        resume_run=order.resume_run,
        resume_receipt=order.resume_receipt,
    )
    if current != order:
        raise ProspectiveExecutionError("APM work order changed before execution")
    final_path = root / order.run_path
    part_path = root / order.run_part_path
    receipt_path = root / order.completion_receipt_path
    if final_path.exists() and receipt_path.exists():
        return APMExchangeCompletionReceipt.load(
            receipt_path,
            repository_root=root,
            expected_order=order,
            expected_seed_receipt=seed_receipt,
            expected_source_epoch=source_receipt.source_epoch,
        )
    if final_path.exists() or receipt_path.exists():
        raise ProspectiveExecutionError("incomplete APM finalization evidence")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    if part_path.exists():
        raise ProspectiveExecutionError(
            "interrupted APM .part requires explicit recovery/quarantine"
        )
    (execute or _default_execute)(order, root)
    if not part_path.is_file():
        raise ProspectiveExecutionError("APM runner produced no .part artifact")
    annotate_raw_run(
        part_path,
        order=order,
        source_epoch=source_receipt.source_epoch,
    )
    validate_raw_run(
        part_path,
        order=order,
        seed_receipt=seed_receipt,
        source_epoch=source_receipt.source_epoch,
        repository_root=root,
    )
    os.replace(part_path, final_path)
    return _seal_completion(
        order,
        root=root,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
    )


__all__ = [
    "APM_COST_NORMALIZATION",
    "APM_REJECTION_CONTINUATION",
    "APMExecutorOverlayIdentity",
    "APMExchangeCompletionReceipt",
    "APMExchangeWorkOrder",
    "COMPLETED_UNSCORED",
    "COMPLETION_RECEIPT_SCHEMA_V1",
    "EXECUTOR_OVERLAY_PATHS",
    "WORK_ORDER_SCHEMA_V1",
    "annotate_raw_run",
    "build_work_order",
    "discover_executor_overlay",
    "execution_policy",
    "load_work_order",
    "materialization_request",
    "materialize_apm_exchange",
    "validate_canonical_apm_request",
    "validate_raw_run",
    "write_work_order",
]
