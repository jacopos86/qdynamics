"""Deterministically materialize the six prospective Paper-II HH seed actions.

This is a local-only activation boundary.  It resolves the frozen bootstrap
planner into per-seed execution requests and VQE work orders, but it never
executes a work order or authorizes scheduler submission.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    ProspectiveIdentityRegistry,
    SeedIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXEC_HH_SEED,
    ExecutionRequest,
    ProspectiveExecutionError,
    SourceEpochReceipt,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    HHSeedWorkOrder,
    architecture_policy,
    build_work_order,
    load_work_order,
    validate_canonical_seed_request,
    write_work_order,
)


HH_SEED_ACTIVATION_MANIFEST_V1 = "paper_ii_hh_seed_activation_manifest_v1"
CANONICAL_SOURCE_REVISION = "267516eab3b35c2cfb495e543246b48196179322"
PREPARED_STATUS = "PREPARED_NOT_EXECUTED"
CANONICAL_HH_SEED_TOKENS = (
    "nph1__crossover",
    "nph1__mott_dominated",
    "nph1__phonon_dominated",
    "nph3__crossover",
    "nph3__mott_dominated",
    "nph3__phonon_dominated",
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProspectiveExecutionError(
                f"refusing to replace immutable HH activation evidence: {path}"
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


def _repository_path(
    root: Path, value: str | Path, *, must_exist: bool, label: str
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(f"{label} is outside the repository") from exc
    return candidate


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectiveExecutionError("HH activation path escapes repository") from exc


def _verified_file(root: Path, path: Path) -> dict[str, str]:
    return {"path": _relative(root, path), "sha256": _file_sha(path)}


def _seed_token(seed: SeedIdentity) -> str:
    prefix = f"{LANE_HH}::seed::"
    if not seed.seed_id.startswith(prefix):
        raise ProspectiveExecutionError("HH seed identity has an invalid prefix")
    token = seed.seed_id[len(prefix) :]
    if token not in CANONICAL_HH_SEED_TOKENS:
        raise ProspectiveExecutionError("HH seed identity is outside the frozen six")
    return token


def _canonical_hh_seeds(
    registry: ProspectiveIdentityRegistry,
) -> tuple[SeedIdentity, ...]:
    seeds = tuple(
        sorted(
            (row for row in registry.seeds if row.lane_id == LANE_HH),
            key=lambda row: _seed_token(row),
        )
    )
    tokens = tuple(_seed_token(seed) for seed in seeds)
    coordinates = {
        (int(seed.physical["n_ph_max"]), str(seed.physical["regime"]))
        for seed in seeds
    }
    expected_coordinates = {
        (cutoff, regime)
        for cutoff in (1, 3)
        for regime in ("crossover", "mott_dominated", "phonon_dominated")
    }
    if (
        tokens != CANONICAL_HH_SEED_TOKENS
        or len({seed.identity_sha256 for seed in seeds}) != 6
        or coordinates != expected_coordinates
        or any(seed.restarts != 4 or seed.optimizer_seed is None for seed in seeds)
    ):
        raise ProspectiveExecutionError("prospective matrix does not expose the frozen six HH seeds")
    return seeds


def _load_source_epoch(
    root: Path, source_epoch_receipt: str | Path
) -> SourceEpochReceipt:
    receipt_path = _repository_path(
        root,
        source_epoch_receipt,
        must_exist=True,
        label="HH source epoch receipt",
    )
    receipt = SourceEpochReceipt.load(receipt_path, repository_root=root)
    epoch = receipt.source_epoch
    if (
        receipt.lane_id != LANE_HH
        or epoch.source_revision != CANONICAL_SOURCE_REVISION
        or epoch.source_archive_sha256 is None
    ):
        raise ProspectiveExecutionError(
            "HH activation requires the source-archive-bound canonical HH seed "
            "source revision"
        )
    return receipt


def _planned_seed_requests(
    registry: ProspectiveIdentityRegistry,
    source: SourceEpochReceipt,
) -> tuple[tuple[SeedIdentity, ExecutionRequest, HHSeedWorkOrder], ...]:
    seeds = _canonical_hh_seeds(registry)
    seed_by_id = {seed.seed_id: seed for seed in seeds}
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(source_epochs={LANE_HH: source.source_epoch}),
        available_executors=(EXEC_HH_SEED,),
    )
    actions = tuple(
        row
        for row in plan.ready_actions
        if row.lane_id == LANE_HH and row.kind == "build_seed"
    )
    waiting_hh = tuple(row for row in plan.waiting_actions if row["lane_id"] == LANE_HH)
    if len(actions) != 6 or waiting_hh:
        raise ProspectiveExecutionError("all six canonical HH seed actions must be ready")
    if {row.subject_id for row in actions} != set(seed_by_id):
        raise ProspectiveExecutionError("HH planner seed subjects diverged from the registry")

    rows: list[tuple[SeedIdentity, ExecutionRequest, HHSeedWorkOrder]] = []
    for action in actions:
        request = resolve_bootstrap_action(action)
        seed = validate_canonical_seed_request(request)
        if seed != seed_by_id[action.subject_id]:
            raise ProspectiveExecutionError("HH planner request resolved another seed identity")
        order = build_work_order(request, source_epoch_receipt=source)
        rows.append((seed, request, order))
    return tuple(sorted(rows, key=lambda row: _seed_token(row[0])))


@dataclass(frozen=True)
class HHSeedActivationManifest:
    manifest_path: str
    manifest_file_sha256: str
    manifest_sha256: str
    source_epoch_receipt: SourceEpochReceipt
    registry_sha256: str
    actions: tuple[Mapping[str, Any], ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "HHSeedActivationManifest":
        root = Path(repository_root).resolve(strict=True)
        manifest_path = _repository_path(
            root, path, must_exist=True, label="HH activation manifest"
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema") != HH_SEED_ACTIVATION_MANIFEST_V1:
            raise ProspectiveExecutionError("unsupported HH seed activation manifest")
        unsigned = dict(payload)
        declared_sha = str(unsigned.pop("manifest_sha256", ""))
        if declared_sha != _canonical_sha(unsigned):
            raise ProspectiveExecutionError("HH seed activation manifest self digest mismatch")
        if (
            payload.get("status") != PREPARED_STATUS
            or payload.get("action_count") != 6
            or payload.get("production_submission_enabled") is not False
            or payload.get("execution_authorized") is not False
            or payload.get("vqe_executed") is not False
            or payload.get("qiskit_compilation_requested") is not False
        ):
            raise ProspectiveExecutionError("HH activation manifest is not local-only")

        source_row = payload.get("source_epoch_receipt")
        if not isinstance(source_row, Mapping):
            raise ProspectiveExecutionError("HH activation manifest has no source receipt")
        source_path = _repository_path(
            root,
            str(source_row.get("path") or ""),
            must_exist=True,
            label="HH source epoch receipt",
        )
        if source_row.get("sha256") != _file_sha(source_path):
            raise ProspectiveExecutionError("HH activation source receipt file digest mismatch")
        source = _load_source_epoch(root, source_path)

        registry = build_identity_registry()
        seeds = _canonical_hh_seeds(registry)
        expected = _planned_seed_requests(registry, source)
        matrix = payload.get("matrix_binding")
        if not isinstance(matrix, Mapping) or matrix != {
            "matrix_id": registry.matrix_id,
            "payload_sha256": registry.matrix_payload_sha256,
            "raw_sha256": registry.matrix_raw_sha256,
            "expanded_semantic_sha256": registry.matrix_expanded_semantic_sha256,
            "registry_sha256": registry.digest,
        }:
            raise ProspectiveExecutionError("HH activation matrix/registry binding diverged")
        if payload.get("source_epoch") != source.source_epoch.to_json_dict():
            raise ProspectiveExecutionError("HH activation source epoch identity diverged")
        if payload.get("source_tree_manifest") != source.source_tree_manifest.to_json_dict():
            raise ProspectiveExecutionError("HH activation source-tree binding diverged")
        if payload.get("source_tree_sha256") != source.source_tree_sha256:
            raise ProspectiveExecutionError("HH activation source-tree digest diverged")
        if payload.get("architecture_policy") != architecture_policy():
            raise ProspectiveExecutionError("HH activation architecture policy diverged")

        actions_raw = payload.get("actions")
        if not isinstance(actions_raw, list) or len(actions_raw) != 6:
            raise ProspectiveExecutionError("HH activation manifest must contain six actions")
        actions = tuple(dict(row) for row in actions_raw if isinstance(row, Mapping))
        if len(actions) != 6:
            raise ProspectiveExecutionError("HH activation action rows are malformed")
        expected_by_seed = {seed.seed_id: (seed, request, order) for seed, request, order in expected}
        if tuple(row.get("seed_token") for row in actions) != CANONICAL_HH_SEED_TOKENS:
            raise ProspectiveExecutionError("HH activation action order or seed set diverged")

        seen_actions: set[str] = set()
        for index, row in enumerate(actions):
            seed_id = str(row.get("seed_id") or "")
            if seed_id not in expected_by_seed:
                raise ProspectiveExecutionError("HH activation names an unknown seed")
            seed, expected_request, expected_order = expected_by_seed[seed_id]
            request_row = row.get("execution_request")
            order_row = row.get("seed_work_order")
            if not isinstance(request_row, Mapping) or not isinstance(order_row, Mapping):
                raise ProspectiveExecutionError("HH activation action lacks request/work order")
            request_path = _repository_path(
                root,
                str(request_row.get("path") or ""),
                must_exist=True,
                label="HH seed execution request",
            )
            order_path = _repository_path(
                root,
                str(order_row.get("path") or ""),
                must_exist=True,
                label="HH seed work order",
            )
            if request_row.get("sha256") != _file_sha(request_path):
                raise ProspectiveExecutionError("HH seed request file digest mismatch")
            if order_row.get("sha256") != _file_sha(order_path):
                raise ProspectiveExecutionError("HH seed work-order file digest mismatch")
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            request = ExecutionRequest.from_mapping(request_payload)
            order, loaded_source = load_work_order(order_path, repository_root=root)
            if (
                request != expected_request
                or order != expected_order
                or order.request != request
                or loaded_source != source
                or row.get("ordinal") != index
                or row.get("action_id") != request.action_id
                or row.get("execution_id") != request.execution_id
                or row.get("seed_token") != _seed_token(seed)
                or row.get("seed_identity_sha256") != seed.identity_sha256
                or row.get("optimizer_seed") != seed.optimizer_seed
                or row.get("artifact_path") != seed.artifact_path
                or request_row.get("execution_request_sha256") != request.digest
                or order_row.get("work_order_sha256") != order.work_order_sha256
            ):
                raise ProspectiveExecutionError("HH activation action binding diverged")
            if request.action_id in seen_actions:
                raise ProspectiveExecutionError("HH activation repeats an action")
            seen_actions.add(request.action_id)

        if {row.seed_id for row in seeds} != set(expected_by_seed):
            raise ProspectiveExecutionError("HH activation omitted a canonical seed")
        return cls(
            manifest_path=_relative(root, manifest_path),
            manifest_file_sha256=_file_sha(manifest_path),
            manifest_sha256=declared_sha,
            source_epoch_receipt=source,
            registry_sha256=registry.digest,
            actions=actions,
        )


def materialize_hh_seed_activation(
    *,
    repository_root: str | Path,
    source_epoch_receipt: str | Path,
    output_dir: str | Path,
) -> HHSeedActivationManifest:
    """Write exactly six immutable request/work-order pairs and one manifest."""

    root = Path(repository_root).resolve(strict=True)
    output = _repository_path(
        root, output_dir, must_exist=False, label="HH activation output directory"
    )
    source = _load_source_epoch(root, source_epoch_receipt)
    registry = build_identity_registry()
    planned = _planned_seed_requests(registry, source)
    policy = architecture_policy()
    actions: list[dict[str, Any]] = []
    for ordinal, (seed, request, order) in enumerate(planned):
        token = _seed_token(seed)
        action_dir = output / token
        request_path = action_dir / "execution_request.json"
        work_order_path = action_dir / "seed_work_order.json"
        _atomic_write_immutable(request_path, _pretty_bytes(request.to_json_dict()))
        write_work_order(work_order_path, order)
        request_file = _verified_file(root, request_path)
        request_file["execution_request_sha256"] = request.digest
        order_file = _verified_file(root, work_order_path)
        order_file["work_order_sha256"] = order.work_order_sha256
        actions.append(
            {
                "ordinal": ordinal,
                "seed_token": token,
                "seed_id": seed.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "physical_coordinate": {
                    "n_ph_max": int(seed.physical["n_ph_max"]),
                    "regime": str(seed.physical["regime"]),
                },
                "optimizer_seed": seed.optimizer_seed,
                "action_id": request.action_id,
                "execution_id": request.execution_id,
                "execution_request": request_file,
                "seed_work_order": order_file,
                "artifact_path": seed.artifact_path,
                "qualification_receipt_path": str(
                    Path(seed.artifact_path).with_name("seed_qualification_receipt.json")
                ),
                "declared_outputs": list(request.declared_outputs),
                "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
                "architecture_policy_sha256": policy["architecture_policy_sha256"],
                "package_inputs": {
                    "execution_request": request_file,
                    "seed_work_order": order_file,
                    "source_epoch_receipt": {
                        "path": source.receipt_path,
                        "sha256": source.receipt_sha256,
                    },
                    "source_tree_manifest": source.source_tree_manifest.to_json_dict(),
                    "source_archive_sha256": source.source_epoch.source_archive_sha256,
                },
                "status": PREPARED_STATUS,
            }
        )

    payload: dict[str, Any] = {
        "schema": HH_SEED_ACTIVATION_MANIFEST_V1,
        "lane_id": LANE_HH,
        "status": PREPARED_STATUS,
        "source_epoch": source.source_epoch.to_json_dict(),
        "source_epoch_receipt": {
            "path": source.receipt_path,
            "sha256": source.receipt_sha256,
        },
        "source_tree_manifest": source.source_tree_manifest.to_json_dict(),
        "source_tree_sha256": source.source_tree_sha256,
        "matrix_binding": {
            "matrix_id": registry.matrix_id,
            "payload_sha256": registry.matrix_payload_sha256,
            "raw_sha256": registry.matrix_raw_sha256,
            "expanded_semantic_sha256": registry.matrix_expanded_semantic_sha256,
            "registry_sha256": registry.digest,
        },
        "architecture_policy": policy,
        "canonical_seed_tokens": list(CANONICAL_HH_SEED_TOKENS),
        "action_count": len(actions),
        "actions": actions,
        "production_submission_enabled": False,
        "execution_authorized": False,
        "vqe_executed": False,
        "qiskit_compilation_requested": False,
        "next_boundary": "per_action_chtc_package_builder",
    }
    payload["manifest_sha256"] = _canonical_sha(payload)
    manifest_path = output / "action_manifest.json"
    _atomic_write_immutable(manifest_path, _pretty_bytes(payload))
    return HHSeedActivationManifest.load(manifest_path, repository_root=root)


__all__ = [
    "CANONICAL_HH_SEED_TOKENS",
    "CANONICAL_SOURCE_REVISION",
    "HH_SEED_ACTIVATION_MANIFEST_V1",
    "HHSeedActivationManifest",
    "PREPARED_STATUS",
    "materialize_hh_seed_activation",
]
