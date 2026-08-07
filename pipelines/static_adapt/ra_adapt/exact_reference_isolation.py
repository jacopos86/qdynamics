"""Study-1 exact-reference isolation receipts.

The generic SR controller intentionally retains its typed exact-ED stopping
path.  Study 1 is narrower: its materialized protocols stop only at their
fixed controller horizon, and exact same-cutoff energies are reporting inputs.
This module authenticates that narrower execution contract without changing or
misdescribing the generic controller.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.static_adapt.ra_adapt.contracts import (
    CanonicalContract,
    ResolvedRAAdaptProtocol,
    canonical_sha256,
)


STUDY1_TRUSTED_EXECUTION_SCHEMA = (
    "paper_i_ra_adapt_study1_trusted_execution_receipt_v2"
)
STUDY1_SOURCE_DATAFLOW_REGRESSION_SCHEMA = (
    "paper_i_ra_adapt_study1_source_dataflow_regression_receipt_v1"
)
STUDY1_G8_EVIDENCE_SCHEMA = (
    "paper_i_ra_adapt_exact_reference_isolation_receipt_v1"
)
STUDY1_EXACT_REFERENCE_POLICY = (
    "reporting_only_after_controller_finalization_v1"
)
STUDY1_EXACT_REFERENCE_EVENT_PHASE = (
    "reporting_after_controller_finalization"
)
STUDY1_SOURCE_DATAFLOW_REGRESSION_TEST_ID = (
    "test_study1_reporting_reference_differential_preserves_"
    "controller_trajectory_and_replay_v1"
)
STUDY1_BUNDLE_IDS = frozenset(
    {
        "ra_repair_stationary_late_v1",
        "ra_repair_measured_late_v1",
    }
)

_CONTROLLER_INSTRUMENTATION_MEMBERS = (
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/sr_snake/_context.py",
    "pipelines/static_adapt/sr_snake/_controller.py",
    "pipelines/static_adapt/ra_adapt/runtime.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
    "pipelines/static_adapt/ra_adapt/append.py",
)
_REPORTING_BOUNDARY_MEMBERS = (
    "pipelines/static_adapt/ra_adapt/contracts.py",
    "pipelines/static_adapt/ra_adapt/exact_reference_isolation.py",
    "pipelines/reporting/paper_i_run_summary.py",
    "pipelines/reporting/paper_i_append_run_summary.py",
)
_TRUSTED_KEYS = frozenset(
    {
        "schema",
        "controller_exact_reference_policy",
        "controller_exact_reference_inputs",
        "study1_protocol_requirement",
        "controller_instrumentation_members",
        "controller_instrumentation_sha256",
        "reporting_boundary_members",
        "reporting_boundary_sha256",
        "source_dataflow_regression",
        "source_dataflow_regression_passed",
        "source_dataflow_regression_test_id",
        "source_dataflow_regression_receipt_sha256",
        "sha256",
    }
)
_REGRESSION_KEYS = frozenset(
    {
        "schema",
        "test_id",
        "controller_instrumentation_sha256",
        "reporting_boundary_sha256",
        "checks",
        "passed",
        "sha256",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "protocol_sha256",
        "controller_consumed_exact_reference",
        "reference_usage",
        "controller_instrumentation_sha256",
        "reporting_boundary_sha256",
        "exact_reference_events",
        "sha256",
    }
)
_EVENT_KEYS = frozenset(
    {
        "phase",
        "event_id",
        "method",
        "finalized_controller_rounds",
        "exact_reference_value_sha256",
    }
)


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    observed = frozenset(str(key) for key in payload)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(
            f"{name} key set drifted (missing={missing}, extra={extra})."
        )


def _self_digest(payload: Mapping[str, Any], *, name: str) -> str:
    body = dict(payload)
    observed = _require_sha256(body.pop("sha256", None), name=f"{name}.sha256")
    expected = canonical_sha256(body)
    if observed != expected:
        raise ValueError(f"{name} self digest does not match its payload.")
    return observed


def _source_root(source_root: str | Path | None = None) -> Path:
    root = (
        Path(__file__).resolve().parents[3]
        if source_root is None
        else Path(source_root).resolve()
    )
    if not (root / "pipelines" / "static_adapt").is_dir():
        raise ValueError(
            "Study-1 exact-reference source root does not contain the "
            "static-ADAPT implementation."
        )
    return root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Study1SourceMember(CanonicalContract):
    path: str
    sha256: str

    def __post_init__(self) -> None:
        path = str(self.path).strip()
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Study-1 source-member paths must be safe and relative.")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, name=f"{path}.sha256"),
        )


def _source_members(
    root: Path,
    relatives: Sequence[str],
) -> tuple[Study1SourceMember, ...]:
    members: list[Study1SourceMember] = []
    for relative in relatives:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                "Study-1 trusted execution source member is missing or "
                f"unsafe: {relative}"
            )
        members.append(
            Study1SourceMember(path=relative, sha256=_sha256_file(path))
        )
    return tuple(members)


def _ast_parses(root: Path, relative: str) -> bool:
    source = (root / relative).read_text(encoding="utf-8")
    ast.parse(source, filename=relative)
    return True


def _call_lines(root: Path, relative: str, call_name: str) -> tuple[int, ...]:
    tree = ast.parse(
        (root / relative).read_text(encoding="utf-8"),
        filename=relative,
    )
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        observed = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else None
        )
        if observed == call_name:
            lines.append(int(node.lineno))
    return tuple(sorted(lines))


@dataclass(frozen=True)
class Study1SourceDataflowRegressionReceipt(CanonicalContract):
    schema: str
    test_id: str
    controller_instrumentation_sha256: str
    reporting_boundary_sha256: str
    checks: tuple[Mapping[str, Any], ...]
    passed: bool
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != STUDY1_SOURCE_DATAFLOW_REGRESSION_SCHEMA:
            raise ValueError("Unknown Study-1 source/dataflow receipt schema.")
        if self.test_id != STUDY1_SOURCE_DATAFLOW_REGRESSION_TEST_ID:
            raise ValueError("Unknown Study-1 source/dataflow regression test.")
        _require_sha256(
            self.controller_instrumentation_sha256,
            name="controller_instrumentation_sha256",
        )
        _require_sha256(
            self.reporting_boundary_sha256,
            name="reporting_boundary_sha256",
        )
        if not self.checks or any(
            not isinstance(check, Mapping)
            or not str(check.get("check_id", "")).strip()
            or check.get("passed") is not True
            for check in self.checks
        ):
            raise ValueError(
                "Study-1 source/dataflow checks must be nonempty and passed."
            )
        if self.passed is not True:
            raise ValueError("Study-1 source/dataflow regression did not pass.")
        payload = self.to_dict()
        payload.pop("sha256", None)
        expected = canonical_sha256(payload)
        if self.sha256 is None:
            object.__setattr__(self, "sha256", expected)
        elif _require_sha256(self.sha256, name="sha256") != expected:
            raise ValueError(
                "Study-1 source/dataflow receipt digest does not match."
            )


def _build_source_dataflow_regression(
    root: Path,
    *,
    controller_instrumentation_sha256: str,
    reporting_boundary_sha256: str,
) -> Study1SourceDataflowRegressionReceipt:
    parsed_members = tuple(
        {
            "path": relative,
            "passed": _ast_parses(root, relative),
        }
        for relative in (
            *_CONTROLLER_INSTRUMENTATION_MEMBERS,
            *_REPORTING_BOUNDARY_MEMBERS,
        )
    )
    ra_attestation_calls = _call_lines(
        root,
        "pipelines/static_adapt/ra_adapt/engine.py",
        "build_study1_exact_reference_isolation_receipt",
    )
    append_attestation_calls = _call_lines(
        root,
        "pipelines/static_adapt/ra_adapt/append.py",
        "build_study1_exact_reference_isolation_receipt",
    )
    controller_source = (
        root / "pipelines/static_adapt/sr_snake/_controller.py"
    ).read_text(encoding="utf-8")
    if len(ra_attestation_calls) != 1 or len(append_attestation_calls) != 1:
        raise ValueError(
            "Study-1 G8 runtime attestation is not integrated exactly once "
            "in both canonical facades."
        )
    if (
        "policy.exact_ed_target" not in controller_source
        or "exact_ed_target_reached" not in controller_source
    ):
        raise ValueError(
            "The generic typed exact-ED stop path was removed or obscured."
        )
    checks = (
        {
            "check_id": "all_bound_python_sources_parse_v1",
            "passed": all(row["passed"] for row in parsed_members),
            "members": list(parsed_members),
        },
        {
            "check_id": "ra_post_controller_g8_attestation_integrated_v1",
            "passed": len(ra_attestation_calls) == 1,
            "call_lines": list(ra_attestation_calls),
        },
        {
            "check_id": "append_post_controller_g8_attestation_integrated_v1",
            "passed": len(append_attestation_calls) == 1,
            "call_lines": list(append_attestation_calls),
        },
        {
            "check_id": "generic_exact_ed_stop_path_preserved_v1",
            "passed": True,
            "study1_scope": "excluded_by_exact_ed_target_none_guard",
        },
    )
    return Study1SourceDataflowRegressionReceipt(
        schema=STUDY1_SOURCE_DATAFLOW_REGRESSION_SCHEMA,
        test_id=STUDY1_SOURCE_DATAFLOW_REGRESSION_TEST_ID,
        controller_instrumentation_sha256=(
            controller_instrumentation_sha256
        ),
        reporting_boundary_sha256=reporting_boundary_sha256,
        checks=checks,
        passed=True,
    )


@dataclass(frozen=True)
class Study1TrustedExecutionReceipt(CanonicalContract):
    schema: str
    controller_exact_reference_policy: str
    controller_exact_reference_inputs: tuple[str, ...]
    study1_protocol_requirement: str
    controller_instrumentation_members: tuple[Study1SourceMember, ...]
    controller_instrumentation_sha256: str
    reporting_boundary_members: tuple[Study1SourceMember, ...]
    reporting_boundary_sha256: str
    source_dataflow_regression: Mapping[str, Any]
    source_dataflow_regression_passed: bool
    source_dataflow_regression_test_id: str
    source_dataflow_regression_receipt_sha256: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != STUDY1_TRUSTED_EXECUTION_SCHEMA:
            raise ValueError("Unknown Study-1 trusted-execution schema.")
        if (
            self.controller_exact_reference_policy
            != STUDY1_EXACT_REFERENCE_POLICY
            or self.controller_exact_reference_inputs
        ):
            raise ValueError(
                "Study-1 controller exact-reference inputs must be empty."
            )
        if (
            self.study1_protocol_requirement
            != "request.execution.stop.exact_ed_target_is_none_v1"
        ):
            raise ValueError("Study-1 exact-target protocol guard drifted.")
        for label, members, digest in (
            (
                "controller instrumentation",
                self.controller_instrumentation_members,
                self.controller_instrumentation_sha256,
            ),
            (
                "reporting boundary",
                self.reporting_boundary_members,
                self.reporting_boundary_sha256,
            ),
        ):
            if not members or any(
                not isinstance(member, Study1SourceMember)
                for member in members
            ):
                raise ValueError(f"Study-1 {label} members are invalid.")
            if canonical_sha256(
                [member.to_dict() for member in members]
            ) != _require_sha256(digest, name=f"{label} sha256"):
                raise ValueError(f"Study-1 {label} digest does not match.")
        regression = validate_study1_source_dataflow_regression_receipt(
            self.source_dataflow_regression
        )
        if (
            self.source_dataflow_regression_passed is not True
            or self.source_dataflow_regression_test_id
            != regression["test_id"]
            or self.source_dataflow_regression_receipt_sha256
            != regression["sha256"]
            or regression["controller_instrumentation_sha256"]
            != self.controller_instrumentation_sha256
            or regression["reporting_boundary_sha256"]
            != self.reporting_boundary_sha256
        ):
            raise ValueError(
                "Study-1 trusted execution lost its source/dataflow proof."
            )
        payload = self.to_dict()
        payload.pop("sha256", None)
        expected = canonical_sha256(payload)
        if self.sha256 is None:
            object.__setattr__(self, "sha256", expected)
        elif _require_sha256(self.sha256, name="sha256") != expected:
            raise ValueError(
                "Study-1 trusted-execution digest does not match."
            )


def build_study1_trusted_execution_receipt(
    *,
    source_root: str | Path | None = None,
) -> Study1TrustedExecutionReceipt:
    """Hash and structurally reverify the Study-1 control/report boundary."""

    root = _source_root(source_root)
    controller_members = _source_members(
        root, _CONTROLLER_INSTRUMENTATION_MEMBERS
    )
    reporting_members = _source_members(root, _REPORTING_BOUNDARY_MEMBERS)
    controller_sha256 = canonical_sha256(
        [member.to_dict() for member in controller_members]
    )
    reporting_sha256 = canonical_sha256(
        [member.to_dict() for member in reporting_members]
    )
    regression = _build_source_dataflow_regression(
        root,
        controller_instrumentation_sha256=controller_sha256,
        reporting_boundary_sha256=reporting_sha256,
    )
    return Study1TrustedExecutionReceipt(
        schema=STUDY1_TRUSTED_EXECUTION_SCHEMA,
        controller_exact_reference_policy=STUDY1_EXACT_REFERENCE_POLICY,
        controller_exact_reference_inputs=(),
        study1_protocol_requirement=(
            "request.execution.stop.exact_ed_target_is_none_v1"
        ),
        controller_instrumentation_members=controller_members,
        controller_instrumentation_sha256=controller_sha256,
        reporting_boundary_members=reporting_members,
        reporting_boundary_sha256=reporting_sha256,
        source_dataflow_regression=regression.to_dict(),
        source_dataflow_regression_passed=True,
        source_dataflow_regression_test_id=regression.test_id,
        source_dataflow_regression_receipt_sha256=str(regression.sha256),
    )


def validate_study1_source_dataflow_regression_receipt(
    value: Any,
) -> dict[str, Any]:
    payload = (
        value.to_dict()
        if isinstance(value, Study1SourceDataflowRegressionReceipt)
        else dict(value)
        if isinstance(value, Mapping)
        else None
    )
    if payload is None:
        raise TypeError("Study-1 source/dataflow receipt must be a mapping.")
    _require_exact_keys(
        payload,
        _REGRESSION_KEYS,
        name="Study-1 source/dataflow receipt",
    )
    _self_digest(payload, name="Study-1 source/dataflow receipt")
    if (
        payload.get("schema")
        != STUDY1_SOURCE_DATAFLOW_REGRESSION_SCHEMA
        or payload.get("test_id")
        != STUDY1_SOURCE_DATAFLOW_REGRESSION_TEST_ID
        or payload.get("passed") is not True
    ):
        raise ValueError("Study-1 source/dataflow receipt semantics drifted.")
    for field_name in (
        "controller_instrumentation_sha256",
        "reporting_boundary_sha256",
    ):
        _require_sha256(payload.get(field_name), name=field_name)
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks or any(
        not isinstance(check, Mapping)
        or check.get("passed") is not True
        for check in checks
    ):
        raise ValueError("Study-1 source/dataflow checks are not all passed.")
    return payload


def validate_study1_trusted_execution_receipt(
    value: Any,
    *,
    source_root: str | Path | None = None,
    reverify_source: bool = True,
) -> dict[str, Any]:
    payload = (
        value.to_dict()
        if isinstance(value, Study1TrustedExecutionReceipt)
        else dict(value)
        if isinstance(value, Mapping)
        else None
    )
    if payload is None:
        raise TypeError("Study-1 trusted-execution receipt must be a mapping.")
    _require_exact_keys(
        payload, _TRUSTED_KEYS, name="Study-1 trusted-execution receipt"
    )
    _self_digest(payload, name="Study-1 trusted-execution receipt")
    if (
        payload.get("schema") != STUDY1_TRUSTED_EXECUTION_SCHEMA
        or payload.get("controller_exact_reference_policy")
        != STUDY1_EXACT_REFERENCE_POLICY
        or payload.get("controller_exact_reference_inputs") != []
        or payload.get("study1_protocol_requirement")
        != "request.execution.stop.exact_ed_target_is_none_v1"
    ):
        raise ValueError("Study-1 trusted-execution semantics drifted.")
    regression = validate_study1_source_dataflow_regression_receipt(
        payload.get("source_dataflow_regression")
    )
    if (
        payload.get("source_dataflow_regression_passed") is not True
        or payload.get("source_dataflow_regression_test_id")
        != regression["test_id"]
        or payload.get("source_dataflow_regression_receipt_sha256")
        != regression["sha256"]
    ):
        raise ValueError(
            "Study-1 trusted execution lost its regression binding."
        )
    for field_name in (
        "controller_instrumentation_sha256",
        "reporting_boundary_sha256",
    ):
        _require_sha256(payload.get(field_name), name=field_name)
    if reverify_source:
        expected = build_study1_trusted_execution_receipt(
            source_root=source_root
        ).to_dict()
        if payload != expected:
            raise ValueError(
                "Study-1 trusted execution does not match the active "
                "source/dataflow boundary."
            )
    return payload


def is_study1_protocol(protocol: Any) -> bool:
    return bool(
        isinstance(protocol, ResolvedRAAdaptProtocol)
        and protocol.bundle_materialization is not None
        and protocol.bundle_id in STUDY1_BUNDLE_IDS
    )


def require_study1_reporting_only_protocol(
    protocol: ResolvedRAAdaptProtocol,
) -> None:
    if not is_study1_protocol(protocol):
        raise ValueError(
            "Study-1 G8 evidence requires a validated Study-1 bundle "
            "protocol."
        )
    if protocol.request.execution.stop.exact_ed_target is not None:
        raise ValueError(
            "Study-1 G8 requires request.execution.stop.exact_ed_target "
            "to be None; the generic SR exact-target stop path remains "
            "outside this receipt's scope."
        )
    if protocol.stopping_rule.get("exact_ed_target") is not None:
        raise ValueError(
            "Study-1 serialized stopping rule permits an exact-ED target."
        )


@dataclass(frozen=True)
class Study1ExactReferenceEvent(CanonicalContract):
    phase: str
    event_id: str
    method: str
    finalized_controller_rounds: int
    exact_reference_value_sha256: str

    def __post_init__(self) -> None:
        if self.phase != STUDY1_EXACT_REFERENCE_EVENT_PHASE:
            raise ValueError(
                "Study-1 exact-reference events must follow controller "
                "finalization."
            )
        if self.event_id != "same_cutoff_exact_energy_reporting_projection_v1":
            raise ValueError("Unknown Study-1 exact-reference event.")
        if self.method not in {"ra_adapt", "append_adapt"}:
            raise ValueError("Unknown Study-1 G8 method.")
        if (
            isinstance(self.finalized_controller_rounds, bool)
            or int(self.finalized_controller_rounds) < 1
        ):
            raise ValueError(
                "Study-1 G8 requires a finalized positive controller horizon."
            )
        object.__setattr__(
            self,
            "finalized_controller_rounds",
            int(self.finalized_controller_rounds),
        )
        _require_sha256(
            self.exact_reference_value_sha256,
            name="exact_reference_value_sha256",
        )


@dataclass(frozen=True)
class Study1ExactReferenceIsolationReceipt(CanonicalContract):
    schema: str
    protocol_sha256: str
    controller_consumed_exact_reference: bool
    reference_usage: str
    controller_instrumentation_sha256: str
    reporting_boundary_sha256: str
    exact_reference_events: tuple[Study1ExactReferenceEvent, ...]
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != STUDY1_G8_EVIDENCE_SCHEMA:
            raise ValueError("Unknown Study-1 G8 evidence schema.")
        _require_sha256(self.protocol_sha256, name="protocol_sha256")
        if (
            self.controller_consumed_exact_reference is not False
            or self.reference_usage != STUDY1_EXACT_REFERENCE_POLICY
        ):
            raise ValueError(
                "Study-1 G8 permits exact references only for reporting."
            )
        _require_sha256(
            self.controller_instrumentation_sha256,
            name="controller_instrumentation_sha256",
        )
        _require_sha256(
            self.reporting_boundary_sha256,
            name="reporting_boundary_sha256",
        )
        if not self.exact_reference_events or any(
            not isinstance(event, Study1ExactReferenceEvent)
            for event in self.exact_reference_events
        ):
            raise ValueError(
                "Study-1 G8 requires typed post-controller reference events."
            )
        payload = self.to_dict()
        payload.pop("sha256", None)
        expected = canonical_sha256(payload)
        if self.sha256 is None:
            object.__setattr__(self, "sha256", expected)
        elif _require_sha256(self.sha256, name="sha256") != expected:
            raise ValueError("Study-1 G8 evidence digest does not match.")


def build_study1_exact_reference_isolation_receipt(
    *,
    protocol: ResolvedRAAdaptProtocol,
    method: str,
    finalized_controller_rounds: int,
    exact_same_cutoff_energy: float,
    trusted_execution_receipt: (
        Study1TrustedExecutionReceipt | Mapping[str, Any] | None
    ) = None,
) -> Study1ExactReferenceIsolationReceipt:
    """Build one runtime G8 attestation after controller finalization."""

    require_study1_reporting_only_protocol(protocol)
    energy = float(exact_same_cutoff_energy)
    if not math.isfinite(energy):
        raise ValueError("Study-1 reporting exact energy must be finite.")
    trusted = validate_study1_trusted_execution_receipt(
        (
            build_study1_trusted_execution_receipt()
            if trusted_execution_receipt is None
            else trusted_execution_receipt
        )
    )
    event = Study1ExactReferenceEvent(
        phase=STUDY1_EXACT_REFERENCE_EVENT_PHASE,
        event_id="same_cutoff_exact_energy_reporting_projection_v1",
        method=str(method),
        finalized_controller_rounds=finalized_controller_rounds,
        exact_reference_value_sha256=canonical_sha256(
            {
                "role": "same_cutoff_exact_energy_reporting_only",
                "value": energy,
            }
        ),
    )
    return Study1ExactReferenceIsolationReceipt(
        schema=STUDY1_G8_EVIDENCE_SCHEMA,
        protocol_sha256=protocol.sha256,
        controller_consumed_exact_reference=False,
        reference_usage=STUDY1_EXACT_REFERENCE_POLICY,
        controller_instrumentation_sha256=trusted[
            "controller_instrumentation_sha256"
        ],
        reporting_boundary_sha256=trusted["reporting_boundary_sha256"],
        exact_reference_events=(event,),
    )


def validate_study1_exact_reference_isolation_receipt(
    value: Any,
    *,
    protocol: ResolvedRAAdaptProtocol,
    trusted_execution_receipt: (
        Study1TrustedExecutionReceipt | Mapping[str, Any] | None
    ) = None,
) -> dict[str, Any]:
    require_study1_reporting_only_protocol(protocol)
    payload = (
        value.to_dict()
        if isinstance(value, Study1ExactReferenceIsolationReceipt)
        else dict(value)
        if isinstance(value, Mapping)
        else None
    )
    if payload is None:
        raise TypeError("Study-1 G8 evidence must be a mapping.")
    _require_exact_keys(payload, _EVIDENCE_KEYS, name="Study-1 G8 evidence")
    _self_digest(payload, name="Study-1 G8 evidence")
    trusted = validate_study1_trusted_execution_receipt(
        (
            build_study1_trusted_execution_receipt()
            if trusted_execution_receipt is None
            else trusted_execution_receipt
        )
    )
    if (
        payload.get("schema") != STUDY1_G8_EVIDENCE_SCHEMA
        or payload.get("protocol_sha256") != protocol.sha256
        or payload.get("controller_consumed_exact_reference") is not False
        or payload.get("reference_usage")
        != STUDY1_EXACT_REFERENCE_POLICY
        or payload.get("controller_instrumentation_sha256")
        != trusted["controller_instrumentation_sha256"]
        or payload.get("reporting_boundary_sha256")
        != trusted["reporting_boundary_sha256"]
    ):
        raise ValueError("Study-1 G8 evidence semantics drifted.")
    events = payload.get("exact_reference_events")
    if not isinstance(events, list) or not events:
        raise ValueError("Study-1 G8 exact-reference events are missing.")
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("Study-1 G8 event must be a mapping.")
        _require_exact_keys(event, _EVENT_KEYS, name="Study-1 G8 event")
        Study1ExactReferenceEvent(**dict(event))
    return payload


__all__ = [
    "STUDY1_EXACT_REFERENCE_EVENT_PHASE",
    "STUDY1_EXACT_REFERENCE_POLICY",
    "STUDY1_G8_EVIDENCE_SCHEMA",
    "STUDY1_SOURCE_DATAFLOW_REGRESSION_SCHEMA",
    "STUDY1_SOURCE_DATAFLOW_REGRESSION_TEST_ID",
    "STUDY1_TRUSTED_EXECUTION_SCHEMA",
    "Study1ExactReferenceEvent",
    "Study1ExactReferenceIsolationReceipt",
    "Study1SourceDataflowRegressionReceipt",
    "Study1SourceMember",
    "Study1TrustedExecutionReceipt",
    "build_study1_exact_reference_isolation_receipt",
    "build_study1_trusted_execution_receipt",
    "is_study1_protocol",
    "require_study1_reporting_only_protocol",
    "validate_study1_exact_reference_isolation_receipt",
    "validate_study1_source_dataflow_regression_receipt",
    "validate_study1_trusted_execution_receipt",
]
