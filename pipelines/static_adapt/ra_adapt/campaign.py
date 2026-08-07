"""Typed one-cell control plane for canonical Paper-I RA-ADAPT runs.

The scientific seam remains :func:`run_ra_adapt`.  This module only binds one
already-selected physical cell and typed request to immutable materialization,
authorization, execution, and terminal receipts.  It deliberately does not
provide a matrix launcher, a settings registry, or compatibility fallbacks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any, Mapping

from pipelines.contracts.problem import ProblemRequest, ResolvedProblemContext
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.adapters import MacroCandidateAdapter
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    CANDIDATE_REPRESENTATION_MACRO,
    RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID,
    RA_ADAPT_PROTOCOL_SCHEMA_V2,
    RA_ADAPT_RESULT_SCHEMA_V2,
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RAAdaptOperationalControls,
    RAAdaptRequest,
    ResolvedRAAdaptProtocol,
    canonical_sha256,
    load_resolved_ra_adapt_protocol,
)
from pipelines.static_adapt.ra_adapt.engine import (
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
    AlwaysCommutationReducedInsertion,
    BeamOff,
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    PruningOff,
    ResolvedProblemReceipt,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
    SingletonAdmission,
)


PAPER_I_CAMPAIGN_PLAN_SCHEMA = "paper_i_ra_adapt_one_cell_campaign_plan_v1"
PAPER_I_LOCAL_AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_one_cell_local_execution_authorization_v1"
)
PAPER_I_RUN_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_one_cell_local_run_manifest_v1"
)
PAPER_I_TERMINAL_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_one_cell_terminal_execution_receipt_v1"
)
PAPER_I_FAILURE_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_one_cell_failed_execution_receipt_v1"
)
PAPER_I_VALIDATION_SCHEMA = "paper_i_ra_adapt_one_cell_validation_v1"
PAPER_I_QISKIT_RETRY_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_round50_qiskit_observation_retry_v1"
)
PAPER_I_QISKIT_RETRY_FAILURE_SCHEMA = (
    "paper_i_ra_adapt_round50_qiskit_observation_retry_failure_v1"
)
PAPER_I_SOURCE_INVENTORY_SCHEMA = (
    "paper_i_ra_adapt_runtime_source_inventory_v1"
)

_ALLOWED_RUN_CLASSES = frozenset(
    {"smoke", "diagnostic", "candidate", "paper_facing"}
)
_EXPECTED_SOURCE_SCHEMA = "paper_i_hh_ed_cutoff_reference_six_regime_v1"
_EXPECTED_SOURCE_CAMPAIGN = (
    "paper_i_hh_ed_cutoff_reference_six_regime_20260727"
)
_ROUND_50 = 50
_EXPECTED_ROUTE_SHA256 = (
    "04f795b0443c7a1ebcb62e9661669a765d5a0006b282f2bee043135bf390cc6b"
)
_QISKIT_COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
_QISKIT_COMPILED_BASIS_GATES = (
    "id",
    "x",
    "sx",
    "rx",
    "ry",
    "rz",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
)
_QISKIT_COMPILED_SCOPE = "ansatz_circuit_including_reference_state"
_QISKIT_COMPILED_SOURCE_KIND = "canonical_paper_i_accepted_prefix"
_QISKIT_ANGLE_CONVENTION = "structural_nonzero_placeholder_angles_v1"
_QISKIT_OBSERVATION_SOURCE_INVENTORY_SCHEMA = (
    "paper_i_round50_qiskit_observation_source_inventory_v1"
)
_QISKIT_OBSERVATION_SOURCE_PATHS = (
    "pipelines/static_adapt/ra_adapt/campaign.py",
    "pipelines/reporting/paper_i_run_summary.py",
    "pipelines/exact_bench/table_i_qiskit_resource_compile.py",
    "pipelines/qiskit_backend_tools.py",
    "pipelines/hardcoded/adapt_circuit_execution.py",
    "src/quantum/ansatz_parameterization.py",
    "src/quantum/pauli_polynomial_class.py",
    "src/quantum/qubitization_module.py",
    "src/quantum/vqe_latex_python_pairs.py",
)


class PaperICampaignContractError(RuntimeError):
    """Fail-closed one-cell campaign contract violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise PaperICampaignContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return normalized


def _require_nonempty(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise PaperICampaignContractError(f"{name} cannot be empty.")
    return normalized


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PaperICampaignContractError(
            f"{label} is unavailable or unsafe: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperICampaignContractError(
            f"Could not load {label}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PaperICampaignContractError(f"{label} must be a JSON object.")
    return payload


def _atomic_write_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(
            f"Refusing stale temporary artifact: {temporary}"
        )
    payload = _canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "sha256" in value:
        raise PaperICampaignContractError(
            "A self-digested payload cannot supply its own SHA-256."
        )
    value["sha256"] = canonical_sha256(value)
    return value


def _load_self_digested(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json_object(path, label=label)
    supplied = _require_sha256(
        str(payload.get("sha256", "")), name=f"{label}.sha256"
    )
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    if supplied != canonical_sha256(unsigned):
        raise PaperICampaignContractError(f"{label} self-digest drifted.")
    return payload


def _file_binding(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PaperICampaignContractError(
            f"Artifact is unavailable or unsafe: {path}"
        )
    resolved = path.resolve()
    rendered = resolved.as_posix()
    if root is not None:
        try:
            rendered = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise PaperICampaignContractError(
                f"Artifact escaped its declared root: {resolved}"
            ) from exc
    return {
        "path": rendered,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _optional_file_binding(
    path: Path, *, root: Path
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _file_binding(path, root=root)


def _checkpoint_sidecar_bindings(
    checkpoint_path: Path, *, output_root: Path
) -> dict[str, dict[str, Any]]:
    checkpoint = _load_json_object(
        checkpoint_path, label="accepted-state checkpoint"
    )
    checkpoint_block = checkpoint.get("checkpoint")
    adapt_block = checkpoint.get("adapt_vqe")
    if not isinstance(checkpoint_block, Mapping) or not isinstance(
        adapt_block, Mapping
    ):
        raise PaperICampaignContractError(
            "Accepted-state checkpoint lacks authenticated sidecar pointers."
        )
    pointers = {
        "estimator_call_ledger_checkpoint": checkpoint_block.get(
            "estimator_call_ledger_checkpoint"
        ),
        "verified_singleton_resume": adapt_block.get(
            "verified_singleton_resume_sidecar"
        ),
    }
    bindings: dict[str, dict[str, Any]] = {}
    for role, pointer in pointers.items():
        if not isinstance(pointer, Mapping):
            raise PaperICampaignContractError(
                f"Accepted-state checkpoint lacks {role} pointer."
            )
        relative = Path(str(pointer.get("path", "")))
        if relative.is_absolute() or not relative.name:
            raise PaperICampaignContractError(
                f"Accepted-state checkpoint has an unsafe {role} path."
            )
        sidecar = (checkpoint_path.parent / relative).resolve()
        try:
            sidecar.relative_to(output_root.resolve())
        except ValueError as exc:
            raise PaperICampaignContractError(
                f"Accepted-state checkpoint {role} escaped the output root."
            ) from exc
        binding = _file_binding(sidecar, root=output_root)
        if binding["sha256"] != pointer.get("sha256"):
            raise PaperICampaignContractError(
                f"Accepted-state checkpoint {role} digest drifted."
            )
        bindings[role] = binding
    return bindings


def _source_inventory(repository_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative_root in ("pipelines", "src"):
        source_root = repository_root / relative_root
        if not source_root.is_dir():
            raise PaperICampaignContractError(
                f"Active source root is missing: {source_root}"
            )
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if not path.is_file() or path.is_symlink():
                raise PaperICampaignContractError(
                    f"Runtime source is unavailable or unsafe: {path}"
                )
            rows.append(
                {
                    "path": path.relative_to(repository_root).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    payload = {
        "schema": PAPER_I_SOURCE_INVENTORY_SCHEMA,
        "source_roots": ["pipelines", "src"],
        "file_count": len(rows),
        "files": rows,
    }
    return _digested(payload)


def _qiskit_observation_source_inventory(
    repository_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in _QISKIT_OBSERVATION_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file() or path.is_symlink():
            raise PaperICampaignContractError(
                f"Qiskit observation source is unavailable or unsafe: {path}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return _digested(
        {
            "schema": _QISKIT_OBSERVATION_SOURCE_INVENTORY_SCHEMA,
            "file_count": len(rows),
            "files": rows,
        }
    )


def _qiskit_observation_environment() -> dict[str, Any]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for package in ("qiskit", "qiskit-terra"):
        try:
            versions[package] = str(metadata.version(package))
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def _resolved_repository_path(
    repository_root: Path, path: str | Path, *, label: str
) -> Path:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else repository_root / raw
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise PaperICampaignContractError(
            f"{label} must live in the active checkout: {repository_root}"
        ) from exc
    return resolved


def _selected_regime(
    source_payload: Mapping[str, Any], regime_name: str
) -> dict[str, Any]:
    normalized = str(regime_name).strip().lower().replace("_", "-")
    rows = [
        dict(row)
        for row in source_payload.get("regimes", ())
        if isinstance(row, Mapping)
        and str(row.get("name", "")).strip().lower().replace("_", "-")
        == normalized
    ]
    if len(rows) != 1:
        raise PaperICampaignContractError(
            f"Physics source must contain exactly one regime {regime_name!r}."
        )
    return rows[0]


def _validate_source_header(source_payload: Mapping[str, Any]) -> None:
    if (
        source_payload.get("schema") != _EXPECTED_SOURCE_SCHEMA
        or source_payload.get("campaign_id") != _EXPECTED_SOURCE_CAMPAIGN
    ):
        raise PaperICampaignContractError(
            "Paper-I physics source identity drifted."
        )
    physics = source_payload.get("physics")
    if not isinstance(physics, Mapping):
        raise PaperICampaignContractError(
            "Paper-I physics source has no typed physics block."
        )
    expected = {
        "L": 2,
        "family": "Hubbard-Holstein",
        "t": 1.0,
        "omega0": 1.0,
        "boson_encoding": "binary",
        "indexing": "blocked",
        "boundary": "open",
        "include_zero_point": True,
        "num_particles": [1, 1],
    }
    for name, value in expected.items():
        if physics.get(name) != value:
            raise PaperICampaignContractError(
                f"Paper-I physics source drifted at physics.{name}."
            )


def _problem_from_regime(
    source_payload: Mapping[str, Any], regime: Mapping[str, Any]
) -> tuple[ResolvedProblemContext, float, float, float]:
    physics = source_payload["physics"]
    working_cutoff = int(regime["working_cutoff"])
    cells = [
        dict(cell)
        for cell in regime.get("cells", ())
        if isinstance(cell, Mapping)
        and int(cell.get("M", -1)) == working_cutoff
    ]
    if len(cells) != 1:
        raise PaperICampaignContractError(
            "Selected regime lacks exactly one same-cutoff ED cell."
        )
    source_energy = float(cells[0]["E_ED"])
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=int(physics["L"]),
            t=float(physics["t"]),
            u=float(regime["U_over_t"]) * float(physics["t"]),
            dv=0.0,
            omega0=float(physics["omega0"]),
            g_ep=float(regime["g_over_t"]) * float(physics["t"]),
            n_ph_max=working_cutoff,
            boson_encoding=str(physics["boson_encoding"]),
            ordering=str(physics["indexing"]),
            boundary=str(physics["boundary"]),
            include_zero_point=bool(physics["include_zero_point"]),
            v_nn=0.0,
            t_prime=0.0,
            n_fermions=None,
        )
    )
    observed_energy = float(problem.exact_target.resolve_energy(ai_log=None))
    delta = abs(observed_energy - source_energy)
    if not math.isfinite(observed_energy) or delta > 1.0e-10:
        raise PaperICampaignContractError(
            "Resolved same-cutoff exact energy drifted from the source lock."
        )
    return problem, source_energy, observed_energy, delta


def _canonical_request(output_root: Path) -> RAAdaptRequest:
    return RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(
            insertion=AlwaysCommutationReducedInsertion(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=_ROUND_50),
        ),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=output_root / "checkpoint.current.json",
                every_controller_rounds=1,
                keep_history_tail=100,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=output_root / "estimator_ledger.json"
            ),
            resource_rounds=(_ROUND_50,),
        ),
    )


def _validate_exact_request(
    request: RAAdaptRequest, *, output_root: Path
) -> None:
    if not isinstance(request, RAAdaptRequest):
        raise TypeError("request must be a typed RAAdaptRequest.")
    expected_request = _canonical_request(output_root)
    if request.to_dict() != expected_request.to_dict():
        raise PaperICampaignContractError(
            "The campaign request drifted from the complete canonical "
            "macro/always/round-50 request."
        )
    if not isinstance(request.adapter, MacroCandidateAdapter):
        raise PaperICampaignContractError(
            "This authorized one-cell campaign requires macro generators."
        )
    method = request.method
    if not (
        isinstance(method.admission, SingletonAdmission)
        and isinstance(
            method.insertion, AlwaysCommutationReducedInsertion
        )
        and isinstance(method.pruning, PruningOff)
        and isinstance(method.beam, BeamOff)
    ):
        raise PaperICampaignContractError(
            "The one-cell candidate must use singleton macro admission, "
            "always commutation-reduced insertion, pruning off, and beam off."
        )
    if (
        int(request.execution.stop.maximum_controller_rounds) != _ROUND_50
        or request.execution.stop.exact_ed_target is not None
        or not isinstance(request.execution.resume, FreshStart)
    ):
        raise PaperICampaignContractError(
            "The one-cell candidate requires a fresh, unconditional "
            "50-controller-round horizon."
        )
    observation = request.observation
    if observation.resource_rounds != (_ROUND_50,):
        raise PaperICampaignContractError(
            "The one-cell candidate must request Qiskit resources at round 50."
        )
    expected_checkpoint = (output_root / "checkpoint.current.json").resolve()
    expected_ledger = (output_root / "estimator_ledger.json").resolve()
    if (
        observation.checkpoint is None
        or observation.checkpoint.path.resolve() != expected_checkpoint
        or observation.checkpoint.every_controller_rounds != 1
        or observation.estimator_ledger is None
        or observation.estimator_ledger.path.resolve() != expected_ledger
    ):
        raise PaperICampaignContractError(
            "Checkpoint and ledger observations must use the bound campaign "
            "paths, with a checkpoint after every accepted round."
        )


def _expected_artifacts() -> dict[str, str]:
    return {
        "campaign_plan": "campaign_plan.json",
        "resolved_protocol": "resolved_protocol.json",
        "runtime_source_inventory": "runtime_source_inventory.json",
        "execution_authorization": "execution_authorization.json",
        "run_manifest": "run_manifest.json",
        "checkpoint": "checkpoint.current.json",
        "estimator_ledger": "estimator_ledger.json",
        "result": "result.json",
        "summary": "summary.json",
        "scientific_receipts": "scientific_receipts.json",
        "validation": "validation.json",
        "terminal_receipt": "terminal_receipt.json",
        "failure_receipt": "failure_receipt.json",
        "resume_attempts_directory": "resume_attempts",
        "qiskit_observation_retries_directory": (
            "qiskit_observation_retries"
        ),
    }


@dataclass(frozen=True)
class PaperICampaignPlan:
    """Self-digested, non-executing plan for exactly one Paper-I cell."""

    schema: str
    campaign_id: str
    run_class: str
    target: str
    paper_lane: str
    method: str
    execution_target: str
    ordered_cases: tuple[str, ...]
    cell: Mapping[str, Any]
    physics_source_lock: Mapping[str, Any]
    runtime_source_inventory: Mapping[str, Any]
    output_root: str
    expected_artifacts: Mapping[str, str]
    execution_authorized: bool
    sha256: str

    def __post_init__(self) -> None:
        if self.schema != PAPER_I_CAMPAIGN_PLAN_SCHEMA:
            raise PaperICampaignContractError("Unknown campaign-plan schema.")
        _require_nonempty(self.campaign_id, name="campaign_id")
        if self.run_class not in _ALLOWED_RUN_CLASSES:
            raise PaperICampaignContractError("Unknown campaign run class.")
        if (
            self.paper_lane != "paper_i_static_adapt"
            or self.method != "ra_adapt"
            or self.execution_target != "local"
        ):
            raise PaperICampaignContractError(
                "Campaign plan is outside the canonical Paper-I local seam."
            )
        if bool(self.execution_authorized):
            raise PaperICampaignContractError(
                "A materialized campaign plan cannot authorize execution."
            )
        cell_id = str(self.cell.get("cell_id", ""))
        if self.ordered_cases != (cell_id,) or not cell_id:
            raise PaperICampaignContractError(
                "A Paper-I campaign plan must bind exactly one ordered cell."
            )
        _require_sha256(self.sha256, name="plan.sha256")
        unsigned = self.to_dict()
        unsigned.pop("sha256")
        if self.sha256 != canonical_sha256(unsigned):
            raise PaperICampaignContractError("Campaign-plan digest drifted.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "run_class": self.run_class,
            "target": self.target,
            "paper_lane": self.paper_lane,
            "method": self.method,
            "execution_target": self.execution_target,
            "ordered_cases": list(self.ordered_cases),
            "cell": dict(self.cell),
            "physics_source_lock": dict(self.physics_source_lock),
            "runtime_source_inventory": dict(
                self.runtime_source_inventory
            ),
            "output_root": self.output_root,
            "expected_artifacts": dict(self.expected_artifacts),
            "execution_authorized": self.execution_authorized,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PaperILocalExecutionAuthorization:
    """Separate user authorization bound to one immutable plan."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = dict(self.payload)
        if value.get("schema") != PAPER_I_LOCAL_AUTHORIZATION_SCHEMA:
            raise PaperICampaignContractError(
                "Unknown local-execution authorization schema."
            )
        if (
            value.get("execution_authorized") is not True
            or value.get("submission_authorized") is not False
            or value.get("execution_target") != "local"
            or value.get("accepted_state_resume_authorized") is not True
            or value.get("qiskit_observation_retry_authorized") is not True
        ):
            raise PaperICampaignContractError(
                "Local authorization has an invalid execution scope."
            )
        supplied = _require_sha256(
            str(value.get("sha256", "")),
            name="authorization.sha256",
        )
        unsigned = dict(value)
        unsigned.pop("sha256", None)
        if supplied != canonical_sha256(unsigned):
            raise PaperICampaignContractError(
                "Local-execution authorization digest drifted."
            )

    @property
    def sha256(self) -> str:
        return str(self.payload["sha256"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _plan_from_mapping(payload: Mapping[str, Any]) -> PaperICampaignPlan:
    value = dict(payload)
    value["ordered_cases"] = tuple(value.get("ordered_cases", ()))
    return PaperICampaignPlan(**value)


def _load_plan(path: Path) -> PaperICampaignPlan:
    return _plan_from_mapping(
        _load_json_object(path, label="Paper-I campaign plan")
    )


def _artifact_paths(plan: PaperICampaignPlan) -> dict[str, Path]:
    root = Path(plan.output_root).expanduser().resolve()
    paths: dict[str, Path] = {}
    for role, relative in plan.expected_artifacts.items():
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PaperICampaignContractError(
                f"Declared artifact {role!r} escaped the output root."
            ) from exc
        paths[str(role)] = candidate
    if dict(plan.expected_artifacts) != _expected_artifacts():
        raise PaperICampaignContractError(
            "Campaign expected-artifact roles drifted."
        )
    return paths


def _protocol_identity(
    protocol: ResolvedRAAdaptProtocol, *, plan: PaperICampaignPlan
) -> None:
    cell = plan.cell
    route = protocol.route_contract
    if not isinstance(route, Mapping):
        raise PaperICampaignContractError(
            "Resolved v2 protocol has no bound route receipt."
        )
    conditions = (
        cell.get("protocol_schema") == protocol.schema,
        cell.get("algorithm_id") == protocol.algorithm_id,
        cell.get("request") == protocol.request.to_dict(),
        cell.get("problem") == protocol.problem.to_dict(),
        cell.get("controller_horizon") == int(protocol.horizon),
        cell.get("resource_rounds") == [50],
        cell.get("optimizer") == protocol.optimizer,
        cell.get("optimizer_maxiter") == int(protocol.optimizer_maxiter),
        cell.get("seeds") == dict(protocol.seeds),
        cell.get("compile_identity") == dict(protocol.compile_identity),
        cell.get("candidate_representation")
        == protocol.candidate_representation,
        cell.get("active_gradient_policy")
        == protocol.active_gradient_policy,
        cell.get("resource_weighting_scope")
        == protocol.resource_weighting_scope,
        protocol.schema == RA_ADAPT_PROTOCOL_SCHEMA_V2,
        protocol.algorithm_id
        == RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID,
        protocol.candidate_representation
        == CANDIDATE_REPRESENTATION_MACRO,
        isinstance(protocol.request.adapter, MacroCandidateAdapter),
        isinstance(
            protocol.request.method.insertion,
            AlwaysCommutationReducedInsertion,
        ),
        protocol.active_gradient_policy == ACTIVE_GRADIENT_MEASURED,
        protocol.resource_weighting_scope == RESOURCE_WEIGHTING_ALL_PHASE,
        int(protocol.horizon) == _ROUND_50,
        protocol.request.observation.resource_rounds == (_ROUND_50,),
        route.get("schema") == RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
        route.get("sha256") == _EXPECTED_ROUTE_SHA256,
        route.get("sha256") == cell.get("route_contract_sha256"),
        protocol.sha256 == cell.get("protocol_sha256"),
        not protocol.source_locks,
    )
    if not all(conditions):
        raise PaperICampaignContractError(
            "Resolved protocol drifted from the authorized ordinary-v2 "
            "macro/always/round-50 identity."
        )


def materialize_paper_i_campaign(
    *,
    repository_root: Path,
    output_root: Path,
    campaign_id: str,
    run_class: str,
    target: str,
    regime_name: str,
    physics_source_path: Path,
    physics_source_sha256: str,
) -> PaperICampaignPlan:
    """Materialize one source-locked, non-executing canonical campaign."""

    repository_root = repository_root.expanduser().resolve()
    if not (repository_root / "AGENTS.md").is_file():
        raise PaperICampaignContractError(
            "repository_root is not the active Holstein checkout."
        )
    output_root = output_root.expanduser().resolve(strict=False)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing existing output root: {output_root}")
    if run_class not in _ALLOWED_RUN_CLASSES:
        raise PaperICampaignContractError("Unknown campaign run class.")
    _require_nonempty(campaign_id, name="campaign_id")
    _require_nonempty(target, name="target")
    source_path = _resolved_repository_path(
        repository_root, physics_source_path, label="physics source"
    )
    expected_source_sha256 = _require_sha256(
        physics_source_sha256, name="physics_source_sha256"
    )
    if _sha256_file(source_path) != expected_source_sha256:
        raise PaperICampaignContractError(
            "Paper-I physics source file hash drifted."
        )
    source_payload = _load_json_object(
        source_path, label="Paper-I six-regime physics source"
    )
    _validate_source_header(source_payload)
    regime = _selected_regime(source_payload, regime_name)
    problem, source_energy, resolved_energy, exact_delta = _problem_from_regime(
        source_payload, regime
    )
    request = _canonical_request(output_root)
    _validate_exact_request(request, output_root=output_root)
    protocol = build_resolved_ra_protocol(problem, request)
    if protocol.source_locks:
        raise PaperICampaignContractError(
            "Ordinary-v2 protocol unexpectedly acquired internal source locks."
        )
    route = protocol.route_contract
    if not isinstance(route, Mapping):
        raise PaperICampaignContractError(
            "Ordinary-v2 protocol has no bound route receipt."
        )
    if route.get("sha256") != _EXPECTED_ROUTE_SHA256:
        raise PaperICampaignContractError(
            "Ordinary-v2 route drifted from the source-controlled canonical "
            "macro/always identity."
        )
    source_inventory = _source_inventory(repository_root)
    artifacts = _expected_artifacts()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    paths = {role: output_root / name for role, name in artifacts.items()}
    _atomic_write_json_noreplace(
        paths["runtime_source_inventory"], source_inventory
    )
    _atomic_write_json_noreplace(
        paths["resolved_protocol"], protocol.to_dict()
    )
    source_relative = source_path.relative_to(repository_root).as_posix()
    cell_id = (
        f"{regime_name.strip().lower().replace('-', '_')}__"
        "macro__always_commutation_reduced__r50"
    )
    cell = {
        "cell_id": cell_id,
        "regime_name": str(regime["name"]),
        "working_cutoff": int(regime["working_cutoff"]),
        "same_cutoff_exact_energy": source_energy,
        "resolved_exact_energy": resolved_energy,
        "resolved_vs_source_exact_energy_abs": exact_delta,
        "selected_regime_sha256": canonical_sha256(regime),
        "problem": ResolvedProblemReceipt.from_problem(problem).to_dict(),
        "request": request.to_dict(),
        "protocol_schema": protocol.schema,
        "algorithm_id": protocol.algorithm_id,
        "protocol_sha256": protocol.sha256,
        "route_contract_schema": route["schema"],
        "route_contract_sha256": route["sha256"],
        "candidate_representation": protocol.candidate_representation,
        "active_gradient_policy": protocol.active_gradient_policy,
        "resource_weighting_scope": protocol.resource_weighting_scope,
        "optimizer": protocol.optimizer,
        "optimizer_maxiter": int(protocol.optimizer_maxiter),
        "seeds": dict(protocol.seeds),
        "controller_horizon": int(protocol.horizon),
        "resource_rounds": [50],
        "compile_identity": dict(protocol.compile_identity),
    }
    source_lock = {
        "path": source_relative,
        "sha256": expected_source_sha256,
        "schema": source_payload["schema"],
        "campaign_id": source_payload["campaign_id"],
        "selected_regime": regime["name"],
        "selected_regime_sha256": canonical_sha256(regime),
    }
    inventory_binding = _file_binding(
        paths["runtime_source_inventory"], root=output_root
    )
    inventory_binding["canonical_sha256"] = source_inventory["sha256"]
    inventory_binding["file_count"] = source_inventory["file_count"]
    protocol_binding = _file_binding(
        paths["resolved_protocol"], root=output_root
    )
    protocol_binding["canonical_sha256"] = protocol.sha256
    cell["protocol_artifact"] = protocol_binding
    unsigned = {
        "schema": PAPER_I_CAMPAIGN_PLAN_SCHEMA,
        "campaign_id": campaign_id,
        "run_class": run_class,
        "target": target,
        "paper_lane": "paper_i_static_adapt",
        "method": "ra_adapt",
        "execution_target": "local",
        "ordered_cases": [cell_id],
        "cell": cell,
        "physics_source_lock": source_lock,
        "runtime_source_inventory": inventory_binding,
        "output_root": output_root.as_posix(),
        "expected_artifacts": artifacts,
        "execution_authorized": False,
    }
    payload = _digested(unsigned)
    plan = _plan_from_mapping(payload)
    _atomic_write_json_noreplace(paths["campaign_plan"], plan.to_dict())
    return plan


def _problem_from_protocol(
    protocol: ResolvedRAAdaptProtocol,
) -> ResolvedProblemContext:
    receipt = protocol.problem
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )
    if ResolvedProblemReceipt.from_problem(problem) != receipt:
        raise PaperICampaignContractError(
            "Reconstructed problem drifted from the resolved protocol."
        )
    return problem


def _preflight(
    plan_path: Path,
) -> tuple[
    PaperICampaignPlan,
    ResolvedRAAdaptProtocol,
    ResolvedProblemContext,
    dict[str, Path],
]:
    plan_path = plan_path.expanduser().resolve()
    plan = _load_plan(plan_path)
    paths = _artifact_paths(plan)
    if plan_path != paths["campaign_plan"]:
        raise PaperICampaignContractError(
            "Campaign plan is not at its bound artifact path."
        )
    # The active checkout is recovered from the source-lock path by walking
    # from this module, then checked against the materialized source binding.
    active_root = Path(__file__).resolve().parents[3]
    if not (active_root / "AGENTS.md").is_file():
        raise PaperICampaignContractError(
            "Could not recover the active Holstein checkout."
        )
    source_path = _resolved_repository_path(
        active_root,
        str(plan.physics_source_lock["path"]),
        label="physics source",
    )
    if _sha256_file(source_path) != plan.physics_source_lock["sha256"]:
        raise PaperICampaignContractError(
            "Physics source drifted after campaign materialization."
        )
    source_payload = _load_json_object(
        source_path, label="Paper-I six-regime physics source"
    )
    _validate_source_header(source_payload)
    regime = _selected_regime(
        source_payload, str(plan.physics_source_lock["selected_regime"])
    )
    if (
        canonical_sha256(regime)
        != plan.physics_source_lock["selected_regime_sha256"]
        or canonical_sha256(regime)
        != plan.cell["selected_regime_sha256"]
    ):
        raise PaperICampaignContractError(
            "Selected-regime source payload drifted."
        )
    inventory_binding = plan.runtime_source_inventory
    inventory_path = paths["runtime_source_inventory"]
    if (
        _sha256_file(inventory_path) != inventory_binding["sha256"]
        or _load_self_digested(
            inventory_path, label="runtime source inventory"
        ).get("sha256")
        != inventory_binding["canonical_sha256"]
    ):
        raise PaperICampaignContractError(
            "Runtime source-inventory artifact drifted."
        )
    current_inventory = _source_inventory(active_root)
    if current_inventory["sha256"] != inventory_binding["canonical_sha256"]:
        raise PaperICampaignContractError(
            "Active runtime sources drifted after materialization."
        )
    protocol_binding = plan.cell["protocol_artifact"]
    protocol_path = paths["resolved_protocol"]
    if _sha256_file(protocol_path) != protocol_binding["sha256"]:
        raise PaperICampaignContractError(
            "Resolved-protocol file binding drifted."
        )
    protocol = load_resolved_ra_adapt_protocol(protocol_path)
    if protocol.sha256 != protocol_binding["canonical_sha256"]:
        raise PaperICampaignContractError(
            "Resolved-protocol canonical digest drifted."
        )
    _protocol_identity(protocol, plan=plan)
    _validate_exact_request(protocol.request, output_root=Path(plan.output_root))
    problem = _problem_from_protocol(protocol)
    _, source_energy, resolved_energy, exact_delta = _problem_from_regime(
        source_payload, regime
    )
    if (
        not math.isclose(
            source_energy,
            float(plan.cell["same_cutoff_exact_energy"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            exact_delta,
            float(plan.cell["resolved_vs_source_exact_energy_abs"]),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        or not math.isclose(
            resolved_energy,
            float(plan.cell["resolved_exact_energy"]),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
    ):
        raise PaperICampaignContractError(
            "Same-cutoff exact-reference binding drifted."
        )
    rebuilt = build_resolved_ra_protocol(problem, protocol.request)
    if rebuilt != protocol:
        raise PaperICampaignContractError(
            "Ordinary-v2 protocol no longer resolves deterministically."
        )
    return plan, protocol, problem, paths


def _qiskit_observation_preflight(
    plan_path: Path,
) -> tuple[
    PaperICampaignPlan,
    ResolvedRAAdaptProtocol,
    dict[str, Path],
]:
    """Authenticate frozen science without requiring current source equality."""

    resolved_plan_path = plan_path.expanduser().resolve()
    plan = _load_plan(resolved_plan_path)
    paths = _artifact_paths(plan)
    if resolved_plan_path != paths["campaign_plan"]:
        raise PaperICampaignContractError(
            "Campaign plan is not at its bound artifact path."
        )
    inventory_binding = plan.runtime_source_inventory
    inventory_path = paths["runtime_source_inventory"]
    if (
        _sha256_file(inventory_path) != inventory_binding["sha256"]
        or _load_self_digested(
            inventory_path, label="frozen runtime source inventory"
        ).get("sha256")
        != inventory_binding["canonical_sha256"]
    ):
        raise PaperICampaignContractError(
            "Frozen runtime source-inventory artifact drifted."
        )
    protocol_binding = _required_mapping(
        plan.cell.get("protocol_artifact"),
        name="campaign protocol artifact",
    )
    protocol_path = paths["resolved_protocol"]
    if _sha256_file(protocol_path) != protocol_binding.get("sha256"):
        raise PaperICampaignContractError(
            "Frozen resolved-protocol file binding drifted."
        )
    protocol = load_resolved_ra_adapt_protocol(protocol_path)
    if protocol.sha256 != protocol_binding.get("canonical_sha256"):
        raise PaperICampaignContractError(
            "Frozen resolved-protocol canonical digest drifted."
        )
    _protocol_identity(protocol, plan=plan)
    return plan, protocol, paths


def authorize_paper_i_campaign(
    plan_path: Path,
    *,
    authorization_basis: str,
    authorized_at_utc: str | None = None,
) -> PaperILocalExecutionAuthorization:
    """Mint a separate local-only authorization for one validated plan."""

    plan, protocol, _problem, paths = _preflight(plan_path)
    basis = _require_nonempty(
        authorization_basis, name="authorization_basis"
    )
    payload = _digested(
        {
            "schema": PAPER_I_LOCAL_AUTHORIZATION_SCHEMA,
            "authorization_id": (
                f"{plan.campaign_id}__local_execution_authorization"
            ),
            "authorization_basis": basis,
            "authorized_at_utc": authorized_at_utc or _utc_now(),
            "run_class": plan.run_class,
            "execution_target": "local",
            "campaign_id": plan.campaign_id,
            "cell_id": plan.cell["cell_id"],
            "plan": {
                **_file_binding(paths["campaign_plan"], root=Path(plan.output_root)),
                "canonical_sha256": plan.sha256,
            },
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": plan.cell["route_contract_sha256"],
            "physics_source_sha256": plan.physics_source_lock["sha256"],
            "runtime_source_inventory_sha256": (
                plan.runtime_source_inventory["canonical_sha256"]
            ),
            "maximum_controller_rounds": _ROUND_50,
            "output_root": plan.output_root,
            "execution_modes": [
                "fresh_start",
                "authenticated_accepted_state_resume",
                "round50_qiskit_observation_retry",
            ],
            "accepted_state_resume_authorized": True,
            "qiskit_observation_retry_authorized": True,
            "execution_authorized": True,
            "submission_authorized": False,
        }
    )
    authorization = PaperILocalExecutionAuthorization(payload)
    _atomic_write_json_noreplace(
        paths["execution_authorization"], authorization.to_dict()
    )
    return authorization


def _load_authorization(path: Path) -> PaperILocalExecutionAuthorization:
    return PaperILocalExecutionAuthorization(
        _load_json_object(path, label="local execution authorization")
    )


def _validate_authorization(
    authorization_path: Path,
    *,
    plan: PaperICampaignPlan,
    protocol: ResolvedRAAdaptProtocol,
    paths: Mapping[str, Path],
) -> PaperILocalExecutionAuthorization:
    authorization_path = authorization_path.expanduser().resolve()
    if authorization_path != paths["execution_authorization"]:
        raise PaperICampaignContractError(
            "Authorization is not at its plan-bound artifact path."
        )
    authorization = _load_authorization(authorization_path)
    value = authorization.payload
    plan_binding = value.get("plan")
    if not isinstance(plan_binding, Mapping):
        raise PaperICampaignContractError(
            "Authorization has no campaign-plan binding."
        )
    expected = {
        "campaign_id": plan.campaign_id,
        "cell_id": plan.cell["cell_id"],
        "run_class": plan.run_class,
        "output_root": plan.output_root,
        "protocol_sha256": protocol.sha256,
        "route_contract_sha256": plan.cell["route_contract_sha256"],
        "physics_source_sha256": plan.physics_source_lock["sha256"],
        "runtime_source_inventory_sha256": (
            plan.runtime_source_inventory["canonical_sha256"]
        ),
        "maximum_controller_rounds": _ROUND_50,
        "execution_modes": [
            "fresh_start",
            "authenticated_accepted_state_resume",
            "round50_qiskit_observation_retry",
        ],
        "accepted_state_resume_authorized": True,
        "qiskit_observation_retry_authorized": True,
    }
    if any(
        value.get(name) != expected_value
        for name, expected_value in expected.items()
    ):
        raise PaperICampaignContractError(
            "Authorization drifted from the immutable campaign plan."
        )
    if (
        plan_binding.get("canonical_sha256") != plan.sha256
        or plan_binding.get("sha256") != _sha256_file(paths["campaign_plan"])
    ):
        raise PaperICampaignContractError(
            "Authorization campaign-plan file binding drifted."
        )
    return authorization


def _git_snapshot(repository_root: Path) -> dict[str, Any]:
    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty_working_tree": (
            True if status.returncode != 0 else bool(status.stdout)
        ),
    }


def _manifest(
    *,
    plan: PaperICampaignPlan,
    protocol: ResolvedRAAdaptProtocol,
    authorization: PaperILocalExecutionAuthorization,
    paths: Mapping[str, Path],
    execution_mode: str,
    attempt_index: int,
    resume_from: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output_root = Path(plan.output_root)
    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    }
    return _digested(
        {
            "schema": PAPER_I_RUN_MANIFEST_SCHEMA,
            "campaign_id": plan.campaign_id,
            "cell_id": plan.cell["cell_id"],
            "run_class": plan.run_class,
            "target": plan.target,
            "paper_lane": plan.paper_lane,
            "method": plan.method,
            "execution_target": "local",
            "execution_mode": execution_mode,
            "attempt_index": int(attempt_index),
            "process_id": os.getpid(),
            "created_at_utc": _utc_now(),
            "plan": {
                **_file_binding(paths["campaign_plan"], root=output_root),
                "canonical_sha256": plan.sha256,
            },
            "authorization": {
                **_file_binding(
                    paths["execution_authorization"], root=output_root
                ),
                "canonical_sha256": authorization.sha256,
            },
            "protocol": {
                **_file_binding(paths["resolved_protocol"], root=output_root),
                "canonical_sha256": protocol.sha256,
                "route_contract_sha256": plan.cell[
                    "route_contract_sha256"
                ],
            },
            "physics_source_lock": dict(plan.physics_source_lock),
            "runtime_source_inventory": dict(
                plan.runtime_source_inventory
            ),
            "effective_settings": {
                "candidate_representation": protocol.candidate_representation,
                "active_gradient_policy": protocol.active_gradient_policy,
                "resource_weighting_scope": protocol.resource_weighting_scope,
                "insertion_policy": protocol.request.method.insertion.kind,
                "admission_policy": protocol.request.method.admission.kind,
                "pruning_policy": protocol.request.method.pruning.kind,
                "beam_policy": protocol.request.method.beam.kind,
                "optimizer": protocol.optimizer,
                "optimizer_maxiter": int(protocol.optimizer_maxiter),
                "seeds": dict(protocol.seeds),
                "maximum_controller_rounds": _ROUND_50,
                "resource_rounds": [50],
                "compile_identity": dict(protocol.compile_identity),
            },
            "repository": {
                "root": Path(__file__).resolve().parents[3].as_posix(),
                **_git_snapshot(Path(__file__).resolve().parents[3]),
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "thread_environment": thread_environment,
            },
            "declared_artifacts": {
                role: path.relative_to(output_root).as_posix()
                for role, path in paths.items()
            },
            "resume_from": (
                None if resume_from is None else dict(resume_from)
            ),
            "execution_authorized": True,
            "submission_authorized": False,
        }
    )


def _execute_scientific(
    problem: ResolvedProblemContext,
    protocol: ResolvedRAAdaptProtocol,
    *,
    resume_checkpoint: Path | None,
) -> Any:
    from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt

    if resume_checkpoint is not None:
        return run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=_ROUND_50,
                resume=AcceptedStateResume(
                    checkpoint_path=resume_checkpoint,
                    checkpoint_sha256=_sha256_file(resume_checkpoint),
                ),
                observation=protocol.request.observation,
            ),
        )
    return run_ra_adapt(problem, protocol)


def _validate_completed_result(
    result: Any,
    *,
    plan: PaperICampaignPlan,
    protocol: ResolvedRAAdaptProtocol,
) -> dict[str, Any]:
    run = result.run
    if result.protocol != protocol:
        raise PaperICampaignContractError(
            "Completed result lost the authorized protocol identity."
        )
    completed_rounds = int(run.stop.completed_controller_rounds)
    if (
        completed_rounds != _ROUND_50
        or len(run.accepted_trajectory) != _ROUND_50
        or int(run.final_state.controller_round) != _ROUND_50
    ):
        raise PaperICampaignContractError(
            "Candidate run did not produce the complete round-50 prefix."
        )
    accounting = run.estimator_accounting
    if not (
        accounting.complete
        and accounting.prefix_closure_passed
        and accounting.all_work.s_alg
        == accounting.raw_occurrence_total
    ):
        raise PaperICampaignContractError(
            "Candidate estimator ledger did not close canonically."
        )
    replay = tuple(run.scientific_replay)
    if len(replay) != _ROUND_50 or any(
        item.accepted_refit.initialization_policy
        != "exact_applied_joint_step_guarded_v1"
        or item.accepted_refit.initialization_status
        not in {"accepted", "rejected"}
        or item.accepted_refit.initialization_guard_nfev != 1
        for item in replay
    ):
        raise PaperICampaignContractError(
            "Round replay lost the guarded full-coordinate warm-start receipt."
        )
    summary = run.paper_i_summary
    if summary is None:
        raise PaperICampaignContractError(
            "Completed candidate has no canonical Paper-I summary."
        )
    if (
        len(summary.accepted_error_trace) != _ROUND_50
        or int(summary.accepted_error_trace[-1].controller_round)
        != _ROUND_50
    ):
        raise PaperICampaignContractError(
            "Paper-I summary lacks the complete round-50 error trace."
        )
    final_trace = summary.accepted_error_trace[-1]
    requested = tuple(
        row
        for row in summary.requested_rounds
        if int(row.controller_round) == _ROUND_50
    )
    if len(requested) != 1:
        raise PaperICampaignContractError(
            "Paper-I summary has no unique round-50 observation."
        )
    round_50 = requested[0]
    prefix = round_50.prefix
    if (
        int(round_50.active_ansatz_depth)
        != int(final_trace.active_ansatz_depth)
        or not math.isclose(
            float(round_50.absolute_energy_error),
            float(final_trace.absolute_energy_error),
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        or int(round_50.algorithmic_work.s_alg)
        != int(accounting.all_work.s_alg)
        or int(prefix.controller_round) != _ROUND_50
        or int(prefix.active_ansatz_depth)
        != int(final_trace.active_ansatz_depth)
        or prefix.checkpoint_sha256 != final_trace.checkpoint_sha256
        or prefix.problem_request_sha256
        != protocol.problem.problem_request_sha256
        or prefix.route_contract_sha256 != _EXPECTED_ROUTE_SHA256
        or run.route.contract_sha256 != _EXPECTED_ROUTE_SHA256
        or int(prefix.algorithmic_work.s_alg)
        != int(accounting.all_work.s_alg)
    ):
        raise PaperICampaignContractError(
            "Round-50 observation drifted from the exact terminal prefix."
        )
    qiskit_status = str(round_50.status)
    qiskit_resources = None
    if qiskit_status == "available":
        resources = round_50.resources
        if resources is None or round_50.failure is not None:
            raise PaperICampaignContractError(
                "Available round-50 Qiskit observation is incomplete."
            )
        metrics = {
            "compiled_two_qubit_count": resources.compiled_two_qubit_count,
            "compiled_two_qubit_depth": resources.compiled_two_qubit_depth,
            "compiled_total_depth": resources.compiled_total_depth,
        }
        if (
            resources.compile_convention != _QISKIT_COMPILE_CONVENTION
            or any(
                isinstance(value, bool)
                or int(value) != value
                or int(value) < 0
                for value in metrics.values()
            )
        ):
            raise PaperICampaignContractError(
                "Round-50 Qiskit resources violate the locked compiler "
                "contract."
            )
        qiskit_resources = {
            "compile_convention": resources.compile_convention,
            **{name: int(value) for name, value in metrics.items()},
        }
    elif qiskit_status == "retryable_tooling_error":
        if (
            round_50.resources is not None
            or round_50.failure is None
            or round_50.failure.retryable is not True
        ):
            raise PaperICampaignContractError(
                "Retryable round-50 Qiskit failure is internally inconsistent."
            )
    else:
        raise PaperICampaignContractError(
            "Round-50 Qiskit observation has an unknown status."
        )
    failure = (
        None
        if round_50.failure is None
        else {
            "exception_type": round_50.failure.exception_type,
            "message": round_50.failure.message,
            "retryable": round_50.failure.retryable,
        }
    )
    payload = {
        "schema": PAPER_I_VALIDATION_SCHEMA,
        "status": (
            "pass"
            if qiskit_status == "available"
            else "retryable_observation_failure"
        ),
        "campaign_id": plan.campaign_id,
        "cell_id": plan.cell["cell_id"],
        "protocol_sha256": protocol.sha256,
        "route_contract_sha256": plan.cell["route_contract_sha256"],
        "accepted_controller_rounds": completed_rounds,
        "active_ansatz_depth": int(final_trace.active_ansatz_depth),
        "final_energy": float(run.final_state.energy),
        "same_cutoff_exact_energy": float(
            summary.provenance.exact_same_cutoff_energy
        ),
        "final_absolute_energy_error": float(
            final_trace.absolute_energy_error
        ),
        "s_alg": int(accounting.all_work.s_alg),
        "ledger_complete": True,
        "prefix_closure_passed": True,
        "warm_start_policy": "exact_applied_joint_step_guarded_v1",
        "warm_start_rounds_verified": len(replay),
        "round_50_qiskit_status": qiskit_status,
        "round_50_qiskit_resources": qiskit_resources,
        "round_50_qiskit_failure": failure,
    }
    return _digested(payload)


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperICampaignContractError(f"{name} must be a JSON object.")
    return value


def _required_sequence(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise PaperICampaignContractError(f"{name} must be a JSON array.")
    return value


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PaperICampaignContractError(
            f"{name} must be an integer no smaller than {minimum}."
        )
    return value


def _strict_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperICampaignContractError(f"{name} must be a nonempty string.")
    return value


def _prefix_from_json_mapping(value: Mapping[str, Any]) -> Any:
    """Rehydrate the exact typed prefix persisted by ``PaperIRunSummary``."""

    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIPrefixCompileInput,
        PaperIPrefixOperator,
        PaperIPrefixPauliTerm,
        PaperIReferenceState,
        PaperIWorkComponents,
    )

    prefix = _required_mapping(value, name="round-50 prefix")
    reference_payload = _required_mapping(
        prefix.get("reference_state"),
        name="round-50 prefix reference_state",
    )
    reference = PaperIReferenceState(
        amplitudes_real=tuple(
            float(item)
            for item in _required_sequence(
                reference_payload.get("amplitudes_real"),
                name="reference amplitudes_real",
            )
        ),
        amplitudes_imaginary=tuple(
            float(item)
            for item in _required_sequence(
                reference_payload.get("amplitudes_imaginary"),
                name="reference amplitudes_imaginary",
            )
        ),
        qubit_count=_strict_int(
            reference_payload.get("qubit_count"),
            name="reference qubit_count",
            minimum=1,
        ),
        source_label=_strict_string(
            reference_payload.get("source_label"),
            name="reference source_label",
        ),
        state_fingerprint=_strict_string(
            reference_payload.get("state_fingerprint"),
            name="reference state_fingerprint",
        ),
    )
    operators: list[PaperIPrefixOperator] = []
    for operator_index, operator_value in enumerate(
        _required_sequence(
            prefix.get("operators"), name="round-50 prefix operators"
        )
    ):
        operator_payload = _required_mapping(
            operator_value,
            name=f"round-50 prefix operator {operator_index}",
        )
        runtime_terms: list[PaperIPrefixPauliTerm] = []
        for term_index, term_value in enumerate(
            _required_sequence(
                operator_payload.get("runtime_terms"),
                name=f"operator {operator_index} runtime_terms",
            )
        ):
            term_payload = _required_mapping(
                term_value,
                name=f"operator {operator_index} term {term_index}",
            )
            runtime_terms.append(
                PaperIPrefixPauliTerm(
                    pauli_exyz=_strict_string(
                        term_payload.get("pauli_exyz"),
                        name=(
                            f"operator {operator_index} term {term_index} "
                            "pauli_exyz"
                        ),
                    ),
                    coefficient_real=float(
                        term_payload.get("coefficient_real")
                    ),
                    coefficient_imaginary=float(
                        term_payload.get("coefficient_imaginary")
                    ),
                    qubit_count=_strict_int(
                        term_payload.get("qubit_count"),
                        name=(
                            f"operator {operator_index} term {term_index} "
                            "qubit_count"
                        ),
                        minimum=1,
                    ),
                )
            )
        operators.append(
            PaperIPrefixOperator(
                candidate_label=_strict_string(
                    operator_payload.get("candidate_label"),
                    name=f"operator {operator_index} candidate_label",
                ),
                logical_index=_strict_int(
                    operator_payload.get("logical_index"),
                    name=f"operator {operator_index} logical_index",
                ),
                runtime_start=_strict_int(
                    operator_payload.get("runtime_start"),
                    name=f"operator {operator_index} runtime_start",
                ),
                runtime_count=_strict_int(
                    operator_payload.get("runtime_count"),
                    name=f"operator {operator_index} runtime_count",
                    minimum=1,
                ),
                execution_mode=_strict_string(
                    operator_payload.get("execution_mode"),
                    name=f"operator {operator_index} execution_mode",
                ),
                runtime_terms=tuple(runtime_terms),
            )
        )
    work_payload = _required_mapping(
        prefix.get("algorithmic_work"),
        name="round-50 prefix algorithmic_work",
    )
    component_payload = _required_mapping(
        work_payload.get("components"),
        name="round-50 prefix work components",
    )
    work = PaperIAlgorithmicWork(
        components=PaperIWorkComponents(
            n_h_outer=_strict_int(
                component_payload.get("n_h_outer"), name="work.n_h_outer"
            ),
            n_h_refit=_strict_int(
                component_payload.get("n_h_refit"), name="work.n_h_refit"
            ),
            n_grad=_strict_int(
                component_payload.get("n_grad"), name="work.n_grad"
            ),
            n_metric=_strict_int(
                component_payload.get("n_metric"), name="work.n_metric"
            ),
        ),
        s_alg=_strict_int(work_payload.get("s_alg"), name="work.s_alg"),
    )
    try:
        return PaperIPrefixCompileInput(
            source_method=_strict_string(
                prefix.get("source_method"), name="prefix source_method"
            ),
            controller_round=_strict_int(
                prefix.get("controller_round"),
                name="prefix controller_round",
                minimum=1,
            ),
            active_ansatz_depth=_strict_int(
                prefix.get("active_ansatz_depth"),
                name="prefix active_ansatz_depth",
                minimum=1,
            ),
            ordered_operator_labels=tuple(
                _strict_string(item, name="prefix operator label")
                for item in _required_sequence(
                    prefix.get("ordered_operator_labels"),
                    name="prefix ordered_operator_labels",
                )
            ),
            operators=tuple(operators),
            logical_parameters=tuple(
                float(item)
                for item in _required_sequence(
                    prefix.get("logical_parameters"),
                    name="prefix logical_parameters",
                )
            ),
            runtime_parameters=tuple(
                float(item)
                for item in _required_sequence(
                    prefix.get("runtime_parameters"),
                    name="prefix runtime_parameters",
                )
            ),
            reference_state=reference,
            checkpoint_sha256=_require_sha256(
                str(prefix.get("checkpoint_sha256", "")),
                name="prefix checkpoint_sha256",
            ),
            projective_state_fingerprint=_strict_string(
                prefix.get("projective_state_fingerprint"),
                name="prefix projective_state_fingerprint",
            ),
            problem_request_sha256=_require_sha256(
                str(prefix.get("problem_request_sha256", "")),
                name="prefix problem_request_sha256",
            ),
            route_profile=_strict_string(
                prefix.get("route_profile"), name="prefix route_profile"
            ),
            route_contract_sha256=_require_sha256(
                str(prefix.get("route_contract_sha256", "")),
                name="prefix route_contract_sha256",
            ),
            algorithmic_work=work,
        )
    except (TypeError, ValueError) as exc:
        raise PaperICampaignContractError(
            "Persisted round-50 prefix failed typed reconstruction."
        ) from exc


def _prefix_from_mapping(value: Mapping[str, Any]) -> Any:
    try:
        prefix = _prefix_from_json_mapping(value)
    except PaperICampaignContractError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PaperICampaignContractError(
            "Persisted round-50 prefix failed typed reconstruction."
        ) from exc
    if _canonical_json_bytes(asdict(prefix)) != _canonical_json_bytes(
        dict(value)
    ):
        raise PaperICampaignContractError(
            "Persisted round-50 prefix did not rehydrate losslessly."
        )
    return prefix


def _compile_qiskit_prefix(prefix: Any) -> dict[str, Any]:
    from pipelines.reporting.paper_i_run_summary import (
        compile_paper_i_prefix_qiskit_payload,
    )

    return compile_paper_i_prefix_qiskit_payload(prefix)


def _qiskit_resources_from_payload(
    payload: Mapping[str, Any], *, prefix: Any
) -> dict[str, Any]:
    compile_payload = _required_mapping(
        payload, name="round-50 Qiskit compile payload"
    )
    count_1q = _strict_int(
        compile_payload.get("compiled_count_1q_total"),
        name="compiled_count_1q_total",
    )
    count_2q = _strict_int(
        compile_payload.get("compiled_count_2q_total"),
        name="compiled_count_2q_total",
    )
    depth_2q = _strict_int(
        compile_payload.get("compiled_depth_2q_total"),
        name="compiled_depth_2q_total",
    )
    depth_total = _strict_int(
        compile_payload.get("compiled_depth_total"),
        name="compiled_depth_total",
    )
    identity_checks = (
        compile_payload.get("compiled_circuit_stats_status") == "ok",
        compile_payload.get("first_hit_cost_source_kind")
        == _QISKIT_COMPILED_SOURCE_KIND,
        compile_payload.get("compiled_resource_source_kind")
        == _QISKIT_COMPILED_SOURCE_KIND,
        compile_payload.get("compiled_resource_qiskit_validated") is True,
        compile_payload.get("qiskit_first_hit_cost_validated") is False,
        tuple(compile_payload.get("compiled_basis_gates", ()))
        == _QISKIT_COMPILED_BASIS_GATES,
        compile_payload.get("compile_convention")
        == _QISKIT_COMPILE_CONVENTION,
        compile_payload.get("qiskit_transpile_optimization_level") == 0,
        compile_payload.get("qiskit_transpile_seed") == 7,
        compile_payload.get("grouped_exact_coefficient_tolerance") == 1.0e-12,
        compile_payload.get("grouped_exact_max_active_qubits") == 5,
        compile_payload.get("angle_convention")
        == _QISKIT_ANGLE_CONVENTION,
        compile_payload.get("compiled_circuit_scope")
        == _QISKIT_COMPILED_SCOPE,
        compile_payload.get("num_qubits")
        == prefix.reference_state.qubit_count,
        compile_payload.get("logical_operator_count")
        == prefix.active_ansatz_depth,
        compile_payload.get("runtime_rotation_count")
        == len(prefix.runtime_parameters),
        depth_total >= depth_2q,
        count_1q >= 0,
        count_2q >= 0,
    )
    if not all(identity_checks):
        raise PaperICampaignContractError(
            "Retried Qiskit payload violates the full locked compiler identity."
        )
    return {
        "compile_convention": _QISKIT_COMPILE_CONVENTION,
        "compiled_two_qubit_count": count_2q,
        "compiled_two_qubit_depth": depth_2q,
        "compiled_total_depth": depth_total,
    }


def _validated_bound_artifact(
    binding: Any,
    *,
    output_root: Path,
    label: str,
) -> dict[str, Any]:
    value = _required_mapping(binding, name=f"{label} binding")
    relative = Path(str(value.get("path", "")))
    if relative.is_absolute() or not relative.name:
        raise PaperICampaignContractError(
            f"{label} binding is not output-root relative."
        )
    candidate = (output_root / relative).resolve()
    try:
        candidate.relative_to(output_root.resolve())
    except ValueError as exc:
        raise PaperICampaignContractError(
            f"{label} binding escaped the campaign output root."
        ) from exc
    observed = _file_binding(candidate, root=output_root)
    if dict(value) != observed:
        raise PaperICampaignContractError(f"{label} artifact binding drifted.")
    return observed


def _qiskit_retry_authority(
    *,
    plan: PaperICampaignPlan,
    authorization: PaperILocalExecutionAuthorization,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    output_root = Path(plan.output_root)
    return {
        "plan": {
            **_file_binding(paths["campaign_plan"], root=output_root),
            "canonical_sha256": plan.sha256,
        },
        "authorization": {
            **_file_binding(
                paths["execution_authorization"], root=output_root
            ),
            "canonical_sha256": authorization.sha256,
        },
        "terminal_receipt": _file_binding(
            paths["terminal_receipt"], root=output_root
        ),
        "validation": _file_binding(paths["validation"], root=output_root),
        "result": _file_binding(paths["result"], root=output_root),
        "summary": _file_binding(paths["summary"], root=output_root),
    }


def _validate_persisted_qiskit_source_inventory(value: Any) -> None:
    inventory = _required_mapping(
        value, name="Qiskit observation source inventory"
    )
    supplied = _require_sha256(
        str(inventory.get("sha256", "")),
        name="Qiskit observation source inventory sha256",
    )
    unsigned = dict(inventory)
    unsigned.pop("sha256", None)
    files = inventory.get("files")
    if (
        inventory.get("schema")
        != _QISKIT_OBSERVATION_SOURCE_INVENTORY_SCHEMA
        or not isinstance(files, list)
        or inventory.get("file_count") != len(files)
        or supplied != canonical_sha256(unsigned)
    ):
        raise PaperICampaignContractError(
            "Qiskit observation source inventory is invalid."
        )


@contextmanager
def _exclusive_qiskit_retry_lock(root: Path) -> Any:
    import fcntl

    if root.is_symlink():
        raise PaperICampaignContractError(
            "Qiskit-observation retry directory cannot be a symlink."
        )
    if root.exists() and not root.is_dir():
        raise PaperICampaignContractError(
            "Qiskit-observation retry path must be a directory."
        )
    root.mkdir(parents=False, exist_ok=True)
    lock_path = root / ".retry.lock"
    if lock_path.is_symlink():
        raise PaperICampaignContractError(
            "Qiskit-observation retry lock cannot be a symlink."
        )
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PaperICampaignContractError(
                "Another Qiskit-observation retry is already active."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _next_qiskit_retry_attempt(
    *,
    root: Path,
    plan: PaperICampaignPlan,
    protocol: ResolvedRAAdaptProtocol,
    authority: Mapping[str, Any],
    prefix_identity: Mapping[str, Any],
    prefix: Any,
) -> tuple[int, Path, Path]:
    if root.is_symlink():
        raise PaperICampaignContractError(
            "Qiskit-observation retry directory cannot be a symlink."
        )
    if root.exists() and not root.is_dir():
        raise PaperICampaignContractError(
            "Qiskit-observation retry path must be a directory."
        )
    outcomes: dict[int, list[tuple[str, Path]]] = {}
    if root.is_dir():
        for path in root.iterdir():
            if path.name == ".retry.lock":
                if not path.is_file() or path.is_symlink():
                    raise PaperICampaignContractError(
                        "Qiskit-observation retry lock is unsafe."
                    )
                continue
            if path.is_symlink() or not path.is_file():
                raise PaperICampaignContractError(
                    "Qiskit-observation retry history contains an unsafe entry."
                )
            outcome = None
            suffix = None
            for candidate_outcome, candidate_suffix in (
                ("result", ".result.json"),
                ("failure", ".failure.json"),
            ):
                if path.name.startswith("attempt_") and path.name.endswith(
                    candidate_suffix
                ):
                    outcome = candidate_outcome
                    suffix = candidate_suffix
                    break
            if outcome is None or suffix is None:
                raise PaperICampaignContractError(
                    "Qiskit-observation retry history has an unknown artifact."
                )
            token = path.name.removeprefix("attempt_").removesuffix(suffix)
            if (
                not token.isdigit()
                or f"{int(token):03d}" != token
                or int(token) < 1
            ):
                raise PaperICampaignContractError(
                    "Qiskit-observation retry attempt name is invalid."
                )
            outcomes.setdefault(int(token), []).append((outcome, path))
    if outcomes and sorted(outcomes) != list(range(1, max(outcomes) + 1)):
        raise PaperICampaignContractError(
            "Qiskit-observation retry history is not contiguous."
        )
    for attempt_index in sorted(outcomes):
        entries = outcomes[attempt_index]
        if len(entries) != 1:
            raise PaperICampaignContractError(
                "Qiskit-observation retry attempt has multiple outcomes."
            )
        outcome, receipt_path = entries[0]
        receipt = _load_self_digested(
            receipt_path,
            label=f"Qiskit-observation retry attempt {attempt_index}",
        )
        expected_schema = (
            PAPER_I_QISKIT_RETRY_RECEIPT_SCHEMA
            if outcome == "result"
            else PAPER_I_QISKIT_RETRY_FAILURE_SCHEMA
        )
        checks = (
            receipt.get("schema") == expected_schema,
            receipt.get("campaign_id") == plan.campaign_id,
            receipt.get("cell_id") == plan.cell["cell_id"],
            receipt.get("protocol_sha256") == protocol.sha256,
            receipt.get("route_contract_sha256")
            == plan.cell["route_contract_sha256"],
            receipt.get("attempt_index") == attempt_index,
            receipt.get("authority") == dict(authority),
            receipt.get("prefix") == dict(prefix_identity),
            receipt.get("observation_only") is True,
            receipt.get("scientific_execution_invoked") is False,
            receipt.get("execution_authorized") is True,
            receipt.get("submission_authorized") is False,
            isinstance(receipt.get("observation_environment"), Mapping),
        )
        if not all(checks):
            raise PaperICampaignContractError(
                "Qiskit-observation retry history drifted from its authority."
            )
        _validate_persisted_qiskit_source_inventory(
            receipt.get("observation_source_inventory")
        )
        if outcome == "result":
            compile_payload = _required_mapping(
                receipt.get("compile_payload"),
                name="prior Qiskit retry compile payload",
            )
            resources = _qiskit_resources_from_payload(
                compile_payload, prefix=prefix
            )
            if (
                receipt.get("status") != "available"
                or receipt.get("resources") != resources
                or "error_type" in receipt
                or "error_message" in receipt
            ):
                raise PaperICampaignContractError(
                    "Successful Qiskit retry history is malformed."
                )
            raise PaperICampaignContractError(
                "A successful round-50 Qiskit retry already exists."
            )
        if (
            receipt.get("status") not in {"failed", "interrupted"}
            or receipt.get("retryable") is not True
            or not isinstance(receipt.get("error_type"), str)
            or not str(receipt.get("error_type", "")).strip()
            or not isinstance(receipt.get("error_message"), str)
            or "compile_payload" in receipt
            or "resources" in receipt
        ):
            raise PaperICampaignContractError(
                "Failed Qiskit retry history is malformed."
            )
    next_index = max(outcomes, default=0) + 1
    result_path = root / f"attempt_{next_index:03d}.result.json"
    failure_path = root / f"attempt_{next_index:03d}.failure.json"
    if any(
        path.exists() or path.is_symlink()
        for path in (result_path, failure_path)
    ):
        raise FileExistsError(
            "Refusing a pre-existing Qiskit-observation retry outcome path."
        )
    return next_index, result_path, failure_path


def _execute_qiskit_observation_retry_attempt(
    *,
    plan: PaperICampaignPlan,
    protocol: ResolvedRAAdaptProtocol,
    paths: Mapping[str, Path],
    authority: Mapping[str, Any],
    prefix_identity: Mapping[str, Any],
    prefix: Any,
) -> dict[str, Any]:
    active_root = Path(__file__).resolve().parents[3]
    retries_root = paths["qiskit_observation_retries_directory"]
    with _exclusive_qiskit_retry_lock(retries_root):
        observation_inventory = _qiskit_observation_source_inventory(
            active_root
        )
        observation_environment = _qiskit_observation_environment()
        attempt_index, result_path, failure_path = (
            _next_qiskit_retry_attempt(
                root=retries_root,
                plan=plan,
                protocol=protocol,
                authority=authority,
                prefix_identity=prefix_identity,
                prefix=prefix,
            )
        )
        try:
            compile_payload = _compile_qiskit_prefix(prefix)
            resources = _qiskit_resources_from_payload(
                compile_payload, prefix=prefix
            )
            receipt = _digested(
                {
                    "schema": PAPER_I_QISKIT_RETRY_RECEIPT_SCHEMA,
                    "status": "available",
                    "campaign_id": plan.campaign_id,
                    "cell_id": plan.cell["cell_id"],
                    "protocol_sha256": protocol.sha256,
                    "route_contract_sha256": plan.cell[
                        "route_contract_sha256"
                    ],
                    "attempt_index": attempt_index,
                    "authority": dict(authority),
                    "prefix": dict(prefix_identity),
                    "observation_source_inventory": observation_inventory,
                    "observation_environment": observation_environment,
                    "compile_payload": dict(compile_payload),
                    "resources": resources,
                    "completed_at_utc": _utc_now(),
                    "observation_only": True,
                    "scientific_execution_invoked": False,
                    "execution_authorized": True,
                    "submission_authorized": False,
                }
            )
            _canonical_json_bytes(receipt)
        except BaseException as exc:
            failure_receipt = _digested(
                {
                    "schema": PAPER_I_QISKIT_RETRY_FAILURE_SCHEMA,
                    "status": (
                        "interrupted"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed"
                    ),
                    "campaign_id": plan.campaign_id,
                    "cell_id": plan.cell["cell_id"],
                    "protocol_sha256": protocol.sha256,
                    "route_contract_sha256": plan.cell[
                        "route_contract_sha256"
                    ],
                    "attempt_index": attempt_index,
                    "authority": dict(authority),
                    "prefix": dict(prefix_identity),
                    "observation_source_inventory": observation_inventory,
                    "observation_environment": observation_environment,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "retryable": True,
                    "failed_at_utc": _utc_now(),
                    "observation_only": True,
                    "scientific_execution_invoked": False,
                    "execution_authorized": True,
                    "submission_authorized": False,
                }
            )
            _atomic_write_json_noreplace(failure_path, failure_receipt)
            raise
        _atomic_write_json_noreplace(result_path, receipt)
        return receipt


def retry_paper_i_campaign_qiskit_observation(
    plan_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    """Retry only the authenticated round-50 Qiskit observation.

    This seam never invokes the RA-ADAPT engine and never rewrites the
    accepted scientific result, summary, validation, or terminal receipt.
    """

    plan, protocol, paths = _qiskit_observation_preflight(plan_path)
    authorization = _validate_authorization(
        authorization_path,
        plan=plan,
        protocol=protocol,
        paths=paths,
    )
    if (
        authorization.payload.get("qiskit_observation_retry_authorized")
        is not True
    ):
        raise PaperICampaignContractError(
            "Authorization does not permit Qiskit-observation retry."
        )
    terminal = _load_self_digested(
        paths["terminal_receipt"], label="terminal execution receipt"
    )
    validation = _load_self_digested(
        paths["validation"], label="campaign validation"
    )
    terminal_checks = (
        terminal.get("schema") == PAPER_I_TERMINAL_RECEIPT_SCHEMA,
        terminal.get("status")
        == "scientific_complete_retryable_observation_failure",
        terminal.get("campaign_id") == plan.campaign_id,
        terminal.get("cell_id") == plan.cell["cell_id"],
        terminal.get("protocol_sha256") == protocol.sha256,
        terminal.get("route_contract_sha256")
        == plan.cell["route_contract_sha256"],
        terminal.get("accepted_controller_rounds") == _ROUND_50,
        terminal.get("round_50_qiskit_status") == "retryable_tooling_error",
        terminal.get("round_50_qiskit_resources") is None,
        terminal.get("execution_authorized") is True,
        terminal.get("submission_authorized") is False,
    )
    validation_checks = (
        validation.get("schema") == PAPER_I_VALIDATION_SCHEMA,
        validation.get("status") == "retryable_observation_failure",
        validation.get("campaign_id") == plan.campaign_id,
        validation.get("cell_id") == plan.cell["cell_id"],
        validation.get("protocol_sha256") == protocol.sha256,
        validation.get("route_contract_sha256")
        == plan.cell["route_contract_sha256"],
        validation.get("accepted_controller_rounds") == _ROUND_50,
        validation.get("ledger_complete") is True,
        validation.get("prefix_closure_passed") is True,
        validation.get("warm_start_policy")
        == "exact_applied_joint_step_guarded_v1",
        validation.get("warm_start_rounds_verified") == _ROUND_50,
        validation.get("round_50_qiskit_status")
        == "retryable_tooling_error",
        validation.get("round_50_qiskit_resources") is None,
    )
    if not all((*terminal_checks, *validation_checks)):
        raise PaperICampaignContractError(
            "The terminal campaign is not eligible for Qiskit-only retry."
        )
    output_root = Path(plan.output_root)
    artifact_bindings = _required_mapping(
        terminal.get("artifacts"), name="terminal artifacts"
    )
    for role in (
        "campaign_plan",
        "resolved_protocol",
        "runtime_source_inventory",
        "execution_authorization",
        "run_manifest",
        "checkpoint",
        "estimator_ledger",
        "result",
        "summary",
        "scientific_receipts",
        "validation",
    ):
        if artifact_bindings.get(role) != _file_binding(
            paths[role], root=output_root
        ):
            raise PaperICampaignContractError(
                f"Terminal {role} artifact binding drifted."
            )
    if "resume_manifest" in artifact_bindings:
        _validated_bound_artifact(
            artifact_bindings["resume_manifest"],
            output_root=output_root,
            label="terminal resume_manifest",
        )
    if terminal.get("checkpoint_sidecars") != _checkpoint_sidecar_bindings(
        paths["checkpoint"], output_root=output_root
    ):
        raise PaperICampaignContractError(
            "Terminal checkpoint-sidecar bindings drifted."
        )
    if (
        terminal.get("active_ansatz_depth")
        != validation.get("active_ansatz_depth")
        or terminal.get("s_alg") != validation.get("s_alg")
        or terminal.get("final_energy") != validation.get("final_energy")
        or terminal.get("final_absolute_energy_error")
        != validation.get("final_absolute_energy_error")
    ):
        raise PaperICampaignContractError(
            "Terminal receipt and validation disagree on the final prefix."
        )
    summary = _load_json_object(paths["summary"], label="Paper-I summary")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or summary.get("available_controller_rounds") != _ROUND_50
    ):
        raise PaperICampaignContractError(
            "Persisted Paper-I summary lacks the complete round-50 horizon."
        )
    result = _load_json_object(paths["result"], label="RA-ADAPT result")
    result_protocol = _required_mapping(
        result.get("protocol"), name="persisted result protocol"
    )
    result_run = _required_mapping(
        result.get("run"), name="persisted RA-ADAPT run"
    )
    result_route = _required_mapping(
        result_run.get("route"), name="persisted result route"
    )
    result_stop = _required_mapping(
        result_run.get("stop"), name="persisted result stop"
    )
    result_final = _required_mapping(
        result_run.get("final_state"), name="persisted result final state"
    )
    result_accounting = _required_mapping(
        result_run.get("estimator_accounting"),
        name="persisted result estimator accounting",
    )
    result_all_work = _required_mapping(
        result_accounting.get("all_work"),
        name="persisted result all work",
    )
    result_trajectory = _required_sequence(
        result_run.get("accepted_trajectory"),
        name="persisted result accepted trajectory",
    )
    embedded_summary = dict(summary)
    embedded_summary.pop("schema", None)
    if (
        result.get("schema") != RA_ADAPT_RESULT_SCHEMA_V2
        or dict(result_protocol) != protocol.to_dict()
        or result_route.get("family") != "ra_adapt"
        or result_route.get("profile")
        != protocol.route_contract["route_profile"]
        or result_route.get("contract_sha256") != _EXPECTED_ROUTE_SHA256
        or result_stop.get("completed_controller_rounds") != _ROUND_50
        or result_final.get("controller_round") != _ROUND_50
        or result_final.get("energy") != validation.get("final_energy")
        or len(result_trajectory) != _ROUND_50
        or result_trajectory[-1] != result_final
        or result_all_work.get("s_alg") != validation.get("s_alg")
        or result_run.get("paper_i_summary") != embedded_summary
    ):
        raise PaperICampaignContractError(
            "Persisted result does not authenticate the standalone summary."
        )
    trace = _required_sequence(
        summary.get("accepted_error_trace"), name="accepted error trace"
    )
    if len(trace) != _ROUND_50:
        raise PaperICampaignContractError(
            "Persisted Paper-I summary lacks all 50 accepted trace rows."
        )
    for expected_round, row_value in enumerate(trace, start=1):
        row = _required_mapping(
            row_value, name=f"accepted error trace row {expected_round}"
        )
        if row.get("controller_round") != expected_round:
            raise PaperICampaignContractError(
                "Persisted accepted error trace is not round-contiguous."
            )
    final_trace = _required_mapping(trace[-1], name="terminal error trace row")
    requested_rows = [
        _required_mapping(row, name="requested-round observation")
        for row in _required_sequence(
            summary.get("requested_rounds"), name="requested rounds"
        )
        if isinstance(row, Mapping) and row.get("controller_round") == _ROUND_50
    ]
    if len(requested_rows) != 1:
        raise PaperICampaignContractError(
            "Persisted summary has no unique round-50 observation."
        )
    requested = requested_rows[0]
    failure = _required_mapping(
        requested.get("failure"), name="round-50 observation failure"
    )
    if (
        requested.get("purpose") != "requested_controller_round"
        or requested.get("status") != "retryable_tooling_error"
        or requested.get("resources") is not None
        or failure.get("retryable") is not True
        or dict(failure) != validation.get("round_50_qiskit_failure")
        or requested.get("active_ansatz_depth")
        != validation.get("active_ansatz_depth")
        or requested.get("absolute_energy_error")
        != validation.get("final_absolute_energy_error")
    ):
        raise PaperICampaignContractError(
            "Persisted round-50 observation is not the validated retryable row."
        )
    prefix_payload = _required_mapping(
        requested.get("prefix"), name="round-50 prefix"
    )
    prefix = _prefix_from_mapping(prefix_payload)
    requested_work = _required_mapping(
        requested.get("algorithmic_work"),
        name="round-50 requested algorithmic_work",
    )
    final_checkpoint = _require_sha256(
        str(final_trace.get("checkpoint_sha256", "")),
        name="terminal trace checkpoint_sha256",
    )
    final_depth = _strict_int(
        final_trace.get("active_ansatz_depth"),
        name="terminal trace active_ansatz_depth",
        minimum=1,
    )
    canonical_all_work = _required_mapping(
        summary.get("canonical_all_work"),
        name="summary canonical_all_work",
    )
    if (
        prefix.controller_round != _ROUND_50
        or prefix.active_ansatz_depth != validation["active_ansatz_depth"]
        or final_depth != prefix.active_ansatz_depth
        or requested.get("active_ansatz_depth") != final_depth
        or prefix.checkpoint_sha256 != final_checkpoint
        or prefix.projective_state_fingerprint
        != final_trace.get("projective_state_fingerprint")
        or prefix.problem_request_sha256
        != protocol.problem.problem_request_sha256
        or prefix.route_contract_sha256 != _EXPECTED_ROUTE_SHA256
        or prefix.algorithmic_work.s_alg != validation["s_alg"]
        or dict(requested_work) != asdict(prefix.algorithmic_work)
        or dict(canonical_all_work) != asdict(prefix.algorithmic_work)
    ):
        raise PaperICampaignContractError(
            "Rehydrated prefix drifted from the authenticated terminal state."
        )
    provenance = _required_mapping(
        summary.get("provenance"), name="Paper-I summary provenance"
    )
    if (
        provenance.get("problem_request_sha256")
        != protocol.problem.problem_request_sha256
        or provenance.get("route_family") != "ra_adapt"
        or provenance.get("route_profile") != prefix.route_profile
        or provenance.get("route_contract_sha256") != _EXPECTED_ROUTE_SHA256
        or provenance.get("candidate_representation")
        != protocol.candidate_representation
        or provenance.get("reference_state_fingerprint")
        != prefix.reference_state.state_fingerprint
        or provenance.get("qiskit_compile_convention")
        != _QISKIT_COMPILE_CONVENTION
    ):
        raise PaperICampaignContractError(
            "Paper-I summary provenance drifted from the locked compile route."
        )
    authority = _qiskit_retry_authority(
        plan=plan, authorization=authorization, paths=paths
    )
    prefix_identity = {
        "source_method": prefix.source_method,
        "controller_round": prefix.controller_round,
        "active_ansatz_depth": prefix.active_ansatz_depth,
        "checkpoint_sha256": prefix.checkpoint_sha256,
        "projective_state_fingerprint": prefix.projective_state_fingerprint,
        "problem_request_sha256": prefix.problem_request_sha256,
        "route_contract_sha256": prefix.route_contract_sha256,
        "s_alg": prefix.algorithmic_work.s_alg,
        "canonical_sha256": canonical_sha256(prefix_payload),
    }
    return _execute_qiskit_observation_retry_attempt(
        plan=plan,
        protocol=protocol,
        paths=paths,
        authority=authority,
        prefix_identity=prefix_identity,
        prefix=prefix,
    )


def _write_failure_receipt(
    *,
    plan: PaperICampaignPlan,
    paths: Mapping[str, Path],
    error: BaseException,
    execution_mode: str,
    attempt_index: int,
    failure_path: Path,
) -> None:
    if failure_path.exists() or failure_path.is_symlink():
        return
    output_root = Path(plan.output_root)
    receipt = _digested(
        {
            "schema": PAPER_I_FAILURE_RECEIPT_SCHEMA,
            "status": (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            ),
            "campaign_id": plan.campaign_id,
            "cell_id": plan.cell["cell_id"],
            "protocol_sha256": plan.cell["protocol_sha256"],
            "execution_mode": execution_mode,
            "attempt_index": int(attempt_index),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "checkpoint": _optional_file_binding(
                paths["checkpoint"], root=output_root
            ),
            "estimator_ledger": _optional_file_binding(
                paths["estimator_ledger"], root=output_root
            ),
            "failed_at_utc": _utc_now(),
            "execution_authorized": True,
            "submission_authorized": False,
        }
    )
    _atomic_write_json_noreplace(failure_path, receipt)


def _validated_resume_attempt(
    *,
    plan: PaperICampaignPlan,
    protocol: ResolvedRAAdaptProtocol,
    authorization: PaperILocalExecutionAuthorization,
    paths: Mapping[str, Path],
) -> tuple[int, Path, Path, Path, dict[str, Any]]:
    attempts_root = paths["resume_attempts_directory"]
    if attempts_root.is_symlink():
        raise PaperICampaignContractError(
            "Resume-attempt directory cannot be a symlink."
        )
    if attempts_root.exists() and not attempts_root.is_dir():
        raise PaperICampaignContractError(
            "Resume-attempt path must be a directory."
        )
    completed_indices: list[int] = []
    if attempts_root.is_dir():
        for manifest_path in attempts_root.glob("attempt_*.manifest.json"):
            token = manifest_path.name.removeprefix("attempt_").removesuffix(
                ".manifest.json"
            )
            if token.isdigit():
                completed_indices.append(int(token))
    if completed_indices:
        if sorted(set(completed_indices)) != list(
            range(1, max(completed_indices) + 1)
        ):
            raise PaperICampaignContractError(
                "Resume-attempt manifests are not a contiguous history."
            )
        prior_index = max(completed_indices)
        prior_manifest_path = (
            attempts_root / f"attempt_{prior_index:03d}.manifest.json"
        )
        prior_failure_path = (
            attempts_root / f"attempt_{prior_index:03d}.failure.json"
        )
        prior_mode = "authenticated_accepted_state_resume"
    else:
        prior_index = 0
        prior_manifest_path = paths["run_manifest"]
        prior_failure_path = paths["failure_receipt"]
        prior_mode = "fresh_start"
    if not prior_failure_path.is_file() or prior_failure_path.is_symlink():
        raise PaperICampaignContractError(
            "The latest execution attempt has no immutable failure receipt."
        )
    prior_manifest = _load_self_digested(
        prior_manifest_path, label="prior execution manifest"
    )
    prior_failure = _load_self_digested(
        prior_failure_path, label="prior execution failure receipt"
    )
    protocol_binding = prior_manifest.get("protocol")
    plan_binding = prior_manifest.get("plan")
    authorization_binding = prior_manifest.get("authorization")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol_binding, plan_binding, authorization_binding)
    ):
        raise PaperICampaignContractError(
            "Prior execution manifest lacks typed authority bindings."
        )
    manifest_checks = (
        prior_manifest.get("schema") == PAPER_I_RUN_MANIFEST_SCHEMA,
        prior_manifest.get("campaign_id") == plan.campaign_id,
        prior_manifest.get("cell_id") == plan.cell["cell_id"],
        prior_manifest.get("execution_mode") == prior_mode,
        prior_manifest.get("attempt_index") == prior_index,
        protocol_binding.get("canonical_sha256") == protocol.sha256,
        protocol_binding.get("route_contract_sha256")
        == plan.cell["route_contract_sha256"],
        plan_binding.get("canonical_sha256") == plan.sha256,
        plan_binding.get("sha256")
        == _sha256_file(paths["campaign_plan"]),
        authorization_binding.get("canonical_sha256")
        == authorization.sha256,
        authorization_binding.get("sha256")
        == _sha256_file(paths["execution_authorization"]),
        prior_manifest.get("execution_authorized") is True,
        prior_manifest.get("submission_authorized") is False,
    )
    failure_checks = (
        prior_failure.get("schema") == PAPER_I_FAILURE_RECEIPT_SCHEMA,
        prior_failure.get("campaign_id") == plan.campaign_id,
        prior_failure.get("cell_id") == plan.cell["cell_id"],
        prior_failure.get("protocol_sha256") == protocol.sha256,
        prior_failure.get("execution_mode") == prior_mode,
        prior_failure.get("attempt_index") == prior_index,
        prior_failure.get("execution_authorized") is True,
        prior_failure.get("submission_authorized") is False,
    )
    if not all((*manifest_checks, *failure_checks)):
        raise PaperICampaignContractError(
            "Prior execution attempt drifted from this campaign authority."
        )
    checkpoint_path = paths["checkpoint"]
    expected_checkpoint = prior_failure.get("checkpoint")
    if (
        not isinstance(expected_checkpoint, Mapping)
        or dict(expected_checkpoint)
        != _file_binding(checkpoint_path, root=Path(plan.output_root))
    ):
        raise PaperICampaignContractError(
            "Current accepted-state checkpoint is not the checkpoint bound "
            "by the latest failure receipt."
        )
    checkpoint_payload = _load_json_object(
        checkpoint_path, label="accepted-state checkpoint"
    )
    adapt_payload = checkpoint_payload.get("adapt_vqe")
    completed_prefix = (
        int(adapt_payload.get("history_count", -1))
        if isinstance(adapt_payload, Mapping)
        else -1
    )
    if completed_prefix < 1 or completed_prefix >= _ROUND_50:
        raise PaperICampaignContractError(
            "Accepted-state resume requires a complete prefix in "
            "controller rounds 1..49."
        )
    sidecars = _checkpoint_sidecar_bindings(
        checkpoint_path, output_root=Path(plan.output_root)
    )
    attempt_index = prior_index + 1
    attempts_root.mkdir(parents=False, exist_ok=True)
    manifest_path = (
        attempts_root / f"attempt_{attempt_index:03d}.manifest.json"
    )
    failure_path = (
        attempts_root / f"attempt_{attempt_index:03d}.failure.json"
    )
    if (
        manifest_path.exists()
        or manifest_path.is_symlink()
        or failure_path.exists()
        or failure_path.is_symlink()
    ):
        raise FileExistsError(
            "Refusing pre-existing resume-attempt receipt path."
        )
    resume_from = {
        "prior_attempt_index": prior_index,
        "completed_controller_rounds": completed_prefix,
        "checkpoint": _file_binding(
            checkpoint_path, root=Path(plan.output_root)
        ),
        "checkpoint_sidecars": sidecars,
        "prior_manifest": _file_binding(
            prior_manifest_path, root=Path(plan.output_root)
        ),
        "prior_failure_receipt": _file_binding(
            prior_failure_path, root=Path(plan.output_root)
        ),
    }
    return (
        attempt_index,
        manifest_path,
        failure_path,
        checkpoint_path,
        resume_from,
    )


def execute_paper_i_campaign(
    plan_path: Path,
    authorization_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute exactly the hash-bound one-cell plan and emit run receipts."""

    plan = _load_plan(plan_path.expanduser().resolve())
    paths = _artifact_paths(plan)
    execution_mode = (
        "authenticated_accepted_state_resume" if resume else "fresh_start"
    )
    attempt_index = 0
    failure_output_path: Path | None = (
        None if resume else paths["failure_receipt"]
    )
    try:
        plan, protocol, problem, paths = _preflight(plan_path)
        authorization = _validate_authorization(
            authorization_path,
            plan=plan,
            protocol=protocol,
            paths=paths,
        )
        if resume and authorization.payload.get(
            "accepted_state_resume_authorized"
        ) is not True:
            raise PaperICampaignContractError(
                "Authorization does not permit accepted-state resume."
            )
        collision_roles = (
            (
                "run_manifest",
                "checkpoint",
                "estimator_ledger",
                "result",
                "summary",
                "scientific_receipts",
                "validation",
                "terminal_receipt",
                "failure_receipt",
                "resume_attempts_directory",
                "qiskit_observation_retries_directory",
            )
            if not resume
            else (
                "estimator_ledger",
                "result",
                "summary",
                "scientific_receipts",
                "validation",
                "terminal_receipt",
                "qiskit_observation_retries_directory",
            )
        )
        for role in collision_roles:
            if paths[role].exists() or paths[role].is_symlink():
                raise FileExistsError(
                    f"Refusing pre-existing execution artifact: {paths[role]}"
                )
        resume_from = None
        resume_checkpoint = None
        manifest_path = paths["run_manifest"]
        if resume:
            (
                attempt_index,
                manifest_path,
                failure_output_path,
                resume_checkpoint,
                resume_from,
            ) = _validated_resume_attempt(
                plan=plan,
                protocol=protocol,
                authorization=authorization,
                paths=paths,
            )
        manifest = _manifest(
            plan=plan,
            protocol=protocol,
            authorization=authorization,
            paths=paths,
            execution_mode=execution_mode,
            attempt_index=attempt_index,
            resume_from=resume_from,
        )
        _atomic_write_json_noreplace(manifest_path, manifest)
        original_cwd = Path.cwd()
        os.chdir(Path(__file__).resolve().parents[3])
        try:
            result = _execute_scientific(
                problem,
                protocol,
                resume_checkpoint=resume_checkpoint,
            )
        finally:
            os.chdir(original_cwd)
        validation = _validate_completed_result(
            result, plan=plan, protocol=protocol
        )
        summary = result.run.paper_i_summary
        _atomic_write_json_noreplace(paths["result"], result.to_dict())
        _atomic_write_json_noreplace(paths["summary"], summary.to_dict())
        _atomic_write_json_noreplace(
            paths["scientific_receipts"],
            dict(result.scientific_receipts),
        )
        _atomic_write_json_noreplace(paths["validation"], validation)
        output_root = Path(plan.output_root)
        artifact_bindings = {
            role: _file_binding(paths[role], root=output_root)
            for role in (
                "campaign_plan",
                "resolved_protocol",
                "runtime_source_inventory",
                "execution_authorization",
                "run_manifest",
                "checkpoint",
                "estimator_ledger",
                "result",
                "summary",
                "scientific_receipts",
                "validation",
            )
        }
        if resume:
            artifact_bindings["resume_manifest"] = _file_binding(
                manifest_path, root=output_root
            )
        checkpoint_sidecars = _checkpoint_sidecar_bindings(
            paths["checkpoint"], output_root=output_root
        )
        terminal_status = (
            "passed"
            if validation["round_50_qiskit_status"] == "available"
            else "scientific_complete_retryable_observation_failure"
        )
        terminal = _digested(
            {
                "schema": PAPER_I_TERMINAL_RECEIPT_SCHEMA,
                "status": terminal_status,
                "campaign_id": plan.campaign_id,
                "cell_id": plan.cell["cell_id"],
                "execution_mode": execution_mode,
                "attempt_index": attempt_index,
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": plan.cell[
                    "route_contract_sha256"
                ],
                "accepted_controller_rounds": _ROUND_50,
                "active_ansatz_depth": validation[
                    "active_ansatz_depth"
                ],
                "final_energy": validation["final_energy"],
                "final_absolute_energy_error": validation[
                    "final_absolute_energy_error"
                ],
                "s_alg": validation["s_alg"],
                "round_50_qiskit_status": validation[
                    "round_50_qiskit_status"
                ],
                "round_50_qiskit_resources": validation[
                    "round_50_qiskit_resources"
                ],
                "artifacts": artifact_bindings,
                "checkpoint_sidecars": checkpoint_sidecars,
                "completed_at_utc": _utc_now(),
                "execution_authorized": True,
                "submission_authorized": False,
            }
        )
        _atomic_write_json_noreplace(paths["terminal_receipt"], terminal)
        return terminal
    except BaseException as exc:
        try:
            if failure_output_path is not None:
                _write_failure_receipt(
                    plan=plan,
                    paths=paths,
                    error=exc,
                    execution_mode=execution_mode,
                    attempt_index=attempt_index,
                    failure_path=failure_output_path,
                )
        except Exception:
            traceback.print_exc()
        raise


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Execute one pre-materialized and separately authorized "
            "Paper-I RA-ADAPT campaign cell."
        )
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("authorization", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume from this plan's authenticated accepted-state checkpoint.",
    )
    mode.add_argument(
        "--retry-qiskit-observation",
        action="store_true",
        help="Retry only the authenticated terminal round-50 Qiskit compile.",
    )
    args = parser.parse_args(argv)
    if args.retry_qiskit_observation:
        receipt = retry_paper_i_campaign_qiskit_observation(
            args.plan, args.authorization
        )
    else:
        receipt = execute_paper_i_campaign(
            args.plan,
            args.authorization,
            resume=args.resume,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "PaperICampaignContractError",
    "PaperICampaignPlan",
    "PaperILocalExecutionAuthorization",
    "authorize_paper_i_campaign",
    "execute_paper_i_campaign",
    "materialize_paper_i_campaign",
    "retry_paper_i_campaign_qiskit_observation",
]
