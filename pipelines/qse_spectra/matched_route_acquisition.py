"""Resumable matched selected-QSE versus direct-Davidson acquisition.

The public campaign profile is deliberately closed: every physics case has
exactly two routes, canonical fixed-pool selected QSE and an independently
initialized direct-Davidson comparator.  A selected-QSE-to-Davidson hybrid is
not representable through this module's interface.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from pipelines.qse_spectra.static_campaign import (
    CONSTRUCTION_DIRECT_DAVIDSON,
    CONSTRUCTION_SELECTED_QSE_FIXED_POOL,
    HH_CUTOFFS,
    HH_REGIMES,
    METHOD_DIRECT_DAVIDSON as _METHOD_DIRECT_DAVIDSON,
    METHOD_RECORD_FIRST,
)


PROFILE_SCHEMA_VERSION = "paper_iii_selected_vs_direct_profile_v1"
METHOD_SELECTED_QSE = METHOD_RECORD_FIRST
METHOD_DIRECT_DAVIDSON = _METHOD_DIRECT_DAVIDSON
LOCKED_METHODS = (METHOD_SELECTED_QSE, METHOD_DIRECT_DAVIDSON)
LOCKED_CONSTRUCTION_POLICIES = {
    METHOD_SELECTED_QSE: CONSTRUCTION_SELECTED_QSE_FIXED_POOL,
    METHOD_DIRECT_DAVIDSON: CONSTRUCTION_DIRECT_DAVIDSON,
}
LOCKED_ERROR_TARGETS = (1.0e-2, 1.0e-4, 1.0e-6)

ROUTE_ARTIFACT_SCHEMA_VERSION = "paper_iii_matched_route_artifact_v1"
ONLINE_RECEIPT_SCHEMA_VERSION = "paper_iii_matched_online_route_v1"
EXACT_AUDIT_SCHEMA_VERSION = "paper_iii_matched_post_run_exact_audit_v1"
MATCHED_EVIDENCE_SCHEMA_VERSION = "paper_iii_selected_vs_direct_evidence_v1"
DECISION_SCHEMA_VERSION = "paper_iii_selected_vs_direct_decision_v1"
STATUS_SCHEMA_VERSION = "paper_iii_selected_vs_direct_status_v1"
SOURCE_LOCK_SCHEMA_VERSION = "paper_iii_selected_vs_direct_source_lock_v1"
CASE_RESULT_SCHEMA_VERSION = "paper_iii_selected_vs_direct_case_result_v1"

SELECTED_STATUS_CERTIFIED = "TARGET_WINDOW_CERTIFIED"
SELECTED_STATUS_POOL_EXHAUSTED = "POOL_EXHAUSTED"
SELECTED_STATUS_WINDOW_NOT_CERTIFIED = "TARGET_WINDOW_NOT_CERTIFIED"
SELECTED_TERMINAL_STATUSES = {
    SELECTED_STATUS_CERTIFIED,
    SELECTED_STATUS_POOL_EXHAUSTED,
    SELECTED_STATUS_WINDOW_NOT_CERTIFIED,
}

DIRECT_STATUS_RESIDUAL_CERTIFIED = "DAVIDSON_RESIDUAL_CERTIFIED"
DIRECT_STATUS_DIMENSION_CAP_REACHED = "DAVIDSON_DIMENSION_CAP_REACHED"
DIRECT_STATUS_NO_NOVEL_CORRECTION = "DAVIDSON_NO_NOVEL_CORRECTION"
DIRECT_TERMINAL_STATUSES = {
    DIRECT_STATUS_RESIDUAL_CERTIFIED,
    DIRECT_STATUS_DIMENSION_CAP_REACHED,
    DIRECT_STATUS_NO_NOVEL_CORRECTION,
}

AUDIT_STATUS_REACHED = "REACHED"
AUDIT_STATUS_NOT_REACHED = "NOT_REACHED_WITHIN_CAP"

RECOMMEND_SELECTED = "SELECTED_QSE"
RECOMMEND_DIRECT = "DIRECT_DAVIDSON"
RECOMMEND_UNRESOLVED = "UNRESOLVED"

REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    "pipelines/exact_bench/paper_iii_growth_trace_campaign.py",
    "pipelines/exact_bench/paper_iii_selected_vs_direct_case.py",
    "pipelines/exact_bench/paper_iii_selected_vs_direct_collect.py",
    "pipelines/qse_spectra/adaptive_qse_benchmark.py",
    "pipelines/qse_spectra/davidson_initialization.py",
    "pipelines/qse_spectra/exchange_maintenance.py",
    "pipelines/qse_spectra/matched_route_acquisition.py",
    "pipelines/qse_spectra/method_completion.py",
    "pipelines/qse_spectra/paper_iii_problem.py",
    "pipelines/qse_spectra/record_selection.py",
)
_ONLINE_FORBIDDEN_KEYS = {
    "max_root_abs_error",
    "per_root_abs_error",
    "reference_excitations",
    "reference_ground_energy",
    "exact_roots",
    "exact_errors",
}


class MatchedRouteAcquisitionError(RuntimeError):
    """Fail-closed acquisition, identity, or artifact error."""


@dataclass(frozen=True)
class RouteExecutionResult:
    """One executor result with online and post-run data kept separate."""

    online_rows: tuple[Mapping[str, Any], ...]
    completion: Mapping[str, Any]
    audit_rows: tuple[Mapping[str, Any], ...]
    reference_excitations: tuple[float, ...]
    pool_digest: str
    elapsed_seconds: float


class MatchedRouteExecutor(Protocol):
    def execute(
        self,
        case: "MatchedHHCase",
        method_id: str,
        profile: "MatchedHHProfile",
    ) -> RouteExecutionResult: ...


@dataclass(frozen=True)
class MatchedCampaignSnapshot:
    execution_status: str
    profile_digest: str
    source_lock_digest: str
    completed_route_count: int
    total_route_count: int
    pending_routes: tuple[str, ...]
    matched_evidence_path: Path | None
    decision_path: Path | None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "execution_status": str(self.execution_status),
            "profile_digest": str(self.profile_digest),
            "source_lock_digest": str(self.source_lock_digest),
            "completed_route_count": int(self.completed_route_count),
            "total_route_count": int(self.total_route_count),
            "pending_routes": list(self.pending_routes),
            "matched_evidence_path": (
                None
                if self.matched_evidence_path is None
                else str(self.matched_evidence_path)
            ),
            "decision_path": (
                None if self.decision_path is None else str(self.decision_path)
            ),
        }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatchedHHCase:
    regime: str
    u: float
    g_ep: float
    n_ph_max: int

    @property
    def case_key(self) -> str:
        return f"{self.regime}_nph{int(self.n_ph_max)}"

    def payload(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key,
            "regime": str(self.regime),
            "u": float(self.u),
            "g_ep": float(self.g_ep),
            "n_ph_max": int(self.n_ph_max),
        }


@dataclass(frozen=True)
class MatchedHHProfile:
    profile_id: str
    cases: tuple[MatchedHHCase, ...]
    methods: tuple[str, ...]
    target_roots: tuple[int, ...]
    dimension_cap: int
    trace_stride: int
    error_targets: tuple[float, ...]
    resource_policy_id: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_id": str(self.profile_id),
            "cases": [case.payload() for case in self.cases],
            "methods": list(self.methods),
            "construction_policy_ids": {
                method: LOCKED_CONSTRUCTION_POLICIES[method]
                for method in self.methods
            },
            "target_roots": [int(root) for root in self.target_roots],
            "dimension_cap": int(self.dimension_cap),
            "trace_stride": int(self.trace_stride),
            "error_targets": [float(target) for target in self.error_targets],
            "resource_policy_id": str(self.resource_policy_id),
            "primary_cost_axis": "n2q",
            "controller_boundary": {
                "exact_audit_is_post_run_only": True,
                "exact_audit_can_rewrite_terminal_status": False,
                "implicit_fallback_enabled": False,
            },
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.payload())

    def digest(self) -> str:
        return _canonical_digest(self.payload())


def locked_hh_profile() -> MatchedHHProfile:
    """Return the immutable 12-case, two-route acquisition declaration."""

    cases = tuple(
        MatchedHHCase(
            regime=str(regime),
            u=float(u),
            g_ep=float(g_ep),
            n_ph_max=int(n_ph_max),
        )
        for n_ph_max in HH_CUTOFFS
        for regime, u, g_ep in HH_REGIMES
    )
    return MatchedHHProfile(
        profile_id="paper_iii_selected_vs_direct_hh_20260828_v1",
        cases=cases,
        methods=LOCKED_METHODS,
        target_roots=tuple(range(1, 7)),
        dimension_cap=64,
        trace_stride=2,
        error_targets=LOCKED_ERROR_TARGETS,
        resource_policy_id=(
            "graph_span_two_qubit_only_v1_plus_pencil_estimator_v1"
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _current_source_lock() -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for relative in _SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise MatchedRouteAcquisitionError(
                f"SOURCE_LOCK_MISMATCH: required source path is missing: {relative}."
            )
        hashes[relative] = _sha256_file(path)
    return {
        "schema_version": SOURCE_LOCK_SCHEMA_VERSION,
        "source_revision": _git_revision(),
        "relevant_path_sha256": hashes,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedRouteAcquisitionError(
            f"unreadable acquisition artifact at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise MatchedRouteAcquisitionError(
            f"acquisition artifact at {path} must contain a JSON object."
        )
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(dict(payload), indent=1, sort_keys=True, allow_nan=False) + "\n",
    )


@contextmanager
def _workspace_lock(output_root: Path):
    lock_path = Path(output_root) / ".acquisition.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MatchedRouteAcquisitionError(
                f"WORKSPACE_BUSY: another acquisition owns {output_root}."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _materialize_profile(output_root: Path, profile: MatchedHHProfile) -> str:
    path = Path(output_root) / "profile.json"
    expected = profile.payload()
    if path.is_file():
        existing = _read_json(path)
        if existing != expected:
            raise MatchedRouteAcquisitionError(
                "PROFILE_DRIFT: output root belongs to a different campaign profile."
            )
    else:
        _atomic_write_json(path, expected)
    return profile.digest()


def _materialize_source_lock(output_root: Path) -> tuple[dict[str, Any], str]:
    path = Path(output_root) / "source_lock.json"
    current = _current_source_lock()
    if path.is_file():
        locked = _read_json(path)
        if locked.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
            raise MatchedRouteAcquisitionError(
                "SOURCE_LOCK_MISMATCH: source-lock schema drift."
            )
        if locked.get("relevant_path_sha256") != current["relevant_path_sha256"]:
            raise MatchedRouteAcquisitionError(
                "SOURCE_LOCK_MISMATCH: relevant source content changed since materialization."
            )
    else:
        locked = current
        _atomic_write_json(path, locked)
    return locked, _canonical_digest(locked)


def _route_key(case: MatchedHHCase, method_id: str) -> str:
    return f"{case.case_key}::{method_id}"


def _route_path(output_root: Path, case: MatchedHHCase, method_id: str) -> Path:
    return Path(output_root) / "routes" / case.case_key / f"{method_id}.json"


def _find_forbidden_online_key(value: Any, *, path: str = "online") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            if name in _ONLINE_FORBIDDEN_KEYS:
                return f"{path}.{name}"
            found = _find_forbidden_online_key(nested, path=f"{path}.{name}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found = _find_forbidden_online_key(nested, path=f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _validate_online_rows(rows: Any, *, method_id: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise MatchedRouteAcquisitionError(
            f"route {method_id} requires at least one online rung."
        )
    restored: list[dict[str, Any]] = []
    seen_k: set[int] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise MatchedRouteAcquisitionError("online rung must be a mapping.")
        row = dict(raw)
        k = int(row.get("k", 0))
        if k <= 0 or k in seen_k:
            raise MatchedRouteAcquisitionError(
                f"online rung dimensions must be positive and unique; got {k}."
            )
        seen_k.add(k)
        for axis in ("n2q", "d2q", "dc"):
            if axis not in row or float(row[axis]) < 0.0:
                raise MatchedRouteAcquisitionError(
                    f"MISSING_COST_AXIS: rung {k} lacks nonnegative {axis}."
                )
        estimator = row.get("estimator")
        if not isinstance(estimator, Mapping) or "queries" not in estimator:
            raise MatchedRouteAcquisitionError(
                f"MISSING_COST_AXIS: rung {k} lacks estimator work."
            )
        if method_id == METHOD_DIRECT_DAVIDSON and "support_indices" in row:
            raise MatchedRouteAcquisitionError(
                "PAIR_IDENTITY_MISMATCH: direct Davidson cannot claim record support."
            )
        restored.append(row)
    return restored


def _build_crossings(
    *,
    online_rows: list[dict[str, Any]],
    audit_rows: tuple[Mapping[str, Any], ...],
    error_targets: tuple[float, ...],
) -> list[dict[str, Any]]:
    online_by_k = {int(row["k"]): row for row in online_rows}
    audited: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in audit_rows:
        if not isinstance(raw, Mapping):
            raise MatchedRouteAcquisitionError("post-run audit rung must be a mapping.")
        row = dict(raw)
        k = int(row.get("k", 0))
        if k in seen or k not in online_by_k:
            raise MatchedRouteAcquisitionError(
                "post-run audit rung does not bind one-to-one to an online rung."
            )
        seen.add(k)
        value = row.get("max_root_abs_error")
        if value is not None:
            value = float(value)
            if value < 0.0:
                raise MatchedRouteAcquisitionError(
                    "post-run exact errors must be nonnegative."
                )
            row["max_root_abs_error"] = value
        audited.append(row)
    if seen != set(online_by_k):
        raise MatchedRouteAcquisitionError(
            "post-run audit must report every sealed online rung."
        )

    crossings: list[dict[str, Any]] = []
    for target in error_targets:
        reaching = [
            row
            for row in audited
            if row.get("max_root_abs_error") is not None
            and float(row["max_root_abs_error"]) <= float(target)
        ]
        if not reaching:
            crossings.append(
                {
                    "error_target": float(target),
                    "status": AUDIT_STATUS_NOT_REACHED,
                    "k": None,
                    "n2q": None,
                    "d2q": None,
                    "dc": None,
                    "estimator_queries": None,
                }
            )
            continue
        best = min(
            reaching,
            key=lambda row: (
                float(online_by_k[int(row["k"])]["n2q"]),
                int(row["k"]),
            ),
        )
        online = online_by_k[int(best["k"])]
        crossings.append(
            {
                "error_target": float(target),
                "status": AUDIT_STATUS_REACHED,
                "k": int(best["k"]),
                "n2q": float(online["n2q"]),
                "d2q": float(online["d2q"]),
                "dc": float(online["dc"]),
                "estimator_queries": int(online["estimator"]["queries"]),
            }
        )
    return crossings


def _recompute_audit_rows(
    online_rows: list[dict[str, Any]],
    reference_excitations: tuple[float, ...],
) -> list[dict[str, Any]]:
    references = tuple(float(value) for value in reference_excitations)
    if not references:
        raise MatchedRouteAcquisitionError(
            "post-run exact audit requires a nonempty reference window."
        )
    recomputed: list[dict[str, Any]] = []
    for row in online_rows:
        roots = row.get("root_energies")
        if not isinstance(roots, list):
            raise MatchedRouteAcquisitionError(
                "sealed online rung lacks Ritz energies for independent audit."
            )
        if len(roots) < len(references):
            per_root = None
            maximum = None
        else:
            per_root = [
                abs(float(roots[index]) - reference)
                for index, reference in enumerate(references)
            ]
            maximum = max(per_root)
        recomputed.append(
            {
                "k": int(row["k"]),
                "max_root_abs_error": maximum,
                "per_root_abs_error": per_root,
            }
        )
    return recomputed


def _audit_rows_match(
    reported: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    recomputed: list[dict[str, Any]],
) -> bool:
    if len(reported) != len(recomputed):
        return False
    for raw, expected in zip(reported, recomputed):
        row = dict(raw)
        if int(row.get("k", 0)) != int(expected["k"]):
            return False
        actual_max = row.get("max_root_abs_error")
        expected_max = expected["max_root_abs_error"]
        if actual_max is None or expected_max is None:
            if actual_max is not None or expected_max is not None:
                return False
        elif not math.isclose(
            float(actual_max), float(expected_max), rel_tol=1.0e-10, abs_tol=1.0e-14
        ):
            return False
        actual_per = row.get("per_root_abs_error")
        expected_per = expected["per_root_abs_error"]
        if actual_per is None or expected_per is None:
            if actual_per is not None or expected_per is not None:
                return False
            continue
        if len(actual_per) != len(expected_per):
            return False
        if any(
            not math.isclose(
                float(actual), float(wanted), rel_tol=1.0e-10, abs_tol=1.0e-14
            )
            for actual, wanted in zip(actual_per, expected_per)
        ):
            return False
    return True


def _build_route_artifact(
    *,
    case: MatchedHHCase,
    method_id: str,
    profile: MatchedHHProfile,
    profile_digest: str,
    source_lock_digest: str,
    result: RouteExecutionResult,
) -> dict[str, Any]:
    online_rows = [dict(row) for row in result.online_rows]
    completion = dict(result.completion)
    online = {
        "schema_version": ONLINE_RECEIPT_SCHEMA_VERSION,
        "profile_digest": profile_digest,
        "case": case.payload(),
        "method_id": str(method_id),
        "construction_policy_id": LOCKED_CONSTRUCTION_POLICIES[method_id],
        "target_roots": [int(root) for root in profile.target_roots],
        "dimension_cap": int(profile.dimension_cap),
        "trace_stride": int(profile.trace_stride),
        "resource_policy_id": str(profile.resource_policy_id),
        "pool_digest": str(result.pool_digest),
        "rows": online_rows,
        "completion": completion,
        "online_only": True,
    }
    forbidden = _find_forbidden_online_key(online)
    if forbidden is not None:
        raise MatchedRouteAcquisitionError(
            f"EXACT_CONTROL_LEAK: forbidden field at {forbidden}."
        )
    restored_rows = _validate_online_rows(online_rows, method_id=method_id)
    online["rows"] = restored_rows
    online_digest = _canonical_digest(online)
    references = tuple(float(value) for value in result.reference_excitations)
    recomputed_audit_rows = _recompute_audit_rows(restored_rows, references)
    if not _audit_rows_match(result.audit_rows, recomputed_audit_rows):
        raise MatchedRouteAcquisitionError(
            "audit rung drift: executor audit does not match sealed Ritz energies."
        )
    crossings = _build_crossings(
        online_rows=restored_rows,
        audit_rows=tuple(recomputed_audit_rows),
        error_targets=profile.error_targets,
    )
    audit = {
        "schema_version": EXACT_AUDIT_SCHEMA_VERSION,
        "online_receipt_digest": online_digest,
        "reference_excitations": list(references),
        "root_matching_policy": "index_matched_ascending",
        "rungs": recomputed_audit_rows,
        "crossings": crossings,
        "feeds_online_decisions": False,
        "can_rewrite_terminal_status": False,
    }
    return {
        "schema_version": ROUTE_ARTIFACT_SCHEMA_VERSION,
        "profile_digest": profile_digest,
        "source_lock_digest": source_lock_digest,
        "case_key": case.case_key,
        "method_id": str(method_id),
        "online_receipt": online,
        "online_receipt_digest": online_digest,
        "post_run_exact_audit": audit,
        "elapsed_seconds": float(result.elapsed_seconds),
    }


def _validate_route_artifact(
    payload: Mapping[str, Any],
    *,
    case: MatchedHHCase,
    method_id: str,
    profile: MatchedHHProfile,
    profile_digest: str,
    source_lock_digest: str,
) -> dict[str, Any]:
    artifact = dict(payload)
    if artifact.get("schema_version") != ROUTE_ARTIFACT_SCHEMA_VERSION:
        raise MatchedRouteAcquisitionError("route artifact schema mismatch.")
    if artifact.get("profile_digest") != profile_digest:
        raise MatchedRouteAcquisitionError("PROFILE_DRIFT: route profile mismatch.")
    if artifact.get("source_lock_digest") != source_lock_digest:
        raise MatchedRouteAcquisitionError(
            "SOURCE_LOCK_MISMATCH: route source-lock mismatch."
        )
    if artifact.get("case_key") != case.case_key or artifact.get("method_id") != method_id:
        raise MatchedRouteAcquisitionError("PAIR_IDENTITY_MISMATCH: route identity drift.")
    online = artifact.get("online_receipt")
    if not isinstance(online, Mapping):
        raise MatchedRouteAcquisitionError("route artifact lacks an online receipt.")
    if _canonical_digest(dict(online)) != artifact.get("online_receipt_digest"):
        raise MatchedRouteAcquisitionError("online receipt digest drift.")
    forbidden = _find_forbidden_online_key(online)
    if forbidden is not None:
        raise MatchedRouteAcquisitionError(
            f"EXACT_CONTROL_LEAK: forbidden field at {forbidden}."
        )
    if online.get("construction_policy_id") != LOCKED_CONSTRUCTION_POLICIES[method_id]:
        raise MatchedRouteAcquisitionError("route construction-policy drift.")
    if online.get("target_roots") != list(profile.target_roots):
        raise MatchedRouteAcquisitionError("PAIR_IDENTITY_MISMATCH: target-root drift.")
    rows = _validate_online_rows(online.get("rows"), method_id=method_id)
    completion = online.get("completion")
    if not isinstance(completion, Mapping) or not bool(completion.get("online_only")):
        raise MatchedRouteAcquisitionError("online completion receipt is missing or unsealed.")
    if completion.get("method_id") != method_id:
        raise MatchedRouteAcquisitionError("online completion method mismatch.")
    status = str(completion.get("terminal_status"))
    if method_id == METHOD_SELECTED_QSE:
        if completion.get("completion_route") != "selected_qse_fixed_pool":
            raise MatchedRouteAcquisitionError("HIDDEN_HYBRID_VIOLATION: wrong selected route.")
        if bool(completion.get("davidson_invoked")):
            raise MatchedRouteAcquisitionError(
                "HIDDEN_HYBRID_VIOLATION: selected-QSE invoked Davidson."
            )
        if status not in SELECTED_TERMINAL_STATUSES:
            raise MatchedRouteAcquisitionError("FIXED_POOL_STATUS_VIOLATION.")
    else:
        if completion.get("completion_route") != "direct_davidson":
            raise MatchedRouteAcquisitionError("direct-Davidson route mismatch.")
        if not bool(completion.get("davidson_invoked")):
            raise MatchedRouteAcquisitionError("direct-Davidson receipt denies execution.")
        if status not in DIRECT_TERMINAL_STATUSES:
            raise MatchedRouteAcquisitionError("unknown direct-Davidson status.")
    audit = artifact.get("post_run_exact_audit")
    if not isinstance(audit, Mapping):
        raise MatchedRouteAcquisitionError("route artifact lacks post-run audit.")
    if audit.get("schema_version") != EXACT_AUDIT_SCHEMA_VERSION:
        raise MatchedRouteAcquisitionError("post-run exact-audit schema mismatch.")
    if audit.get("online_receipt_digest") != artifact.get("online_receipt_digest"):
        raise MatchedRouteAcquisitionError("post-run audit is not bound to online receipt.")
    if audit.get("feeds_online_decisions") is not False:
        raise MatchedRouteAcquisitionError("EXACT_CONTROL_LEAK: audit feeds online decisions.")
    if audit.get("can_rewrite_terminal_status") is not False:
        raise MatchedRouteAcquisitionError("EXACT_CONTROL_LEAK: audit can rewrite status.")
    references = audit.get("reference_excitations")
    if not isinstance(references, list) or len(references) != len(profile.target_roots):
        raise MatchedRouteAcquisitionError(
            "post-run exact audit reference window does not match target roots."
        )
    recomputed_audit_rows = _recompute_audit_rows(
        rows, tuple(float(value) for value in references)
    )
    reported_audit_rows = list(audit.get("rungs") or ())
    if not _audit_rows_match(reported_audit_rows, recomputed_audit_rows):
        raise MatchedRouteAcquisitionError(
            "audit rung drift: stored errors do not match sealed Ritz energies."
        )
    expected_crossings = _build_crossings(
        online_rows=rows,
        audit_rows=tuple(recomputed_audit_rows),
        error_targets=profile.error_targets,
    )
    if audit.get("crossings") != expected_crossings:
        raise MatchedRouteAcquisitionError(
            "post-run crossing drift: stored crossings do not recompute from sealed rungs."
        )
    return artifact


def _route_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    online = artifact["online_receipt"]
    rows = online["rows"]
    terminal = rows[-1]
    return {
        "method_id": str(artifact["method_id"]),
        "construction_policy_id": str(online["construction_policy_id"]),
        "algorithm_status": str(online["completion"]["terminal_status"]),
        "terminal_k": int(terminal["k"]),
        "terminal_n2q": float(terminal["n2q"]),
        "terminal_d2q": float(terminal["d2q"]),
        "terminal_dc": float(terminal["dc"]),
        "terminal_estimator_queries": int(terminal["estimator"]["queries"]),
        "crossings": list(artifact["post_run_exact_audit"]["crossings"]),
        "online_receipt_digest": str(artifact["online_receipt_digest"]),
    }


def _recommendation(
    *, selected: Mapping[str, Any], direct: Mapping[str, Any]
) -> str:
    selected_reached = selected.get("status") == AUDIT_STATUS_REACHED
    direct_reached = direct.get("status") == AUDIT_STATUS_REACHED
    selected_certified = selected.get("algorithm_status") == SELECTED_STATUS_CERTIFIED
    if selected_certified and selected_reached and (
        not direct_reached or float(selected["n2q"]) <= float(direct["n2q"])
    ):
        return RECOMMEND_SELECTED
    if direct_reached:
        return RECOMMEND_DIRECT
    if selected_certified and selected_reached:
        return RECOMMEND_SELECTED
    return RECOMMEND_UNRESOLVED


def _build_matched_evidence(
    *,
    profile: MatchedHHProfile,
    profile_digest: str,
    source_lock_digest: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for case in profile.cases:
        selected_artifact = artifacts[_route_key(case, METHOD_SELECTED_QSE)]
        direct_artifact = artifacts[_route_key(case, METHOD_DIRECT_DAVIDSON)]
        selected_summary = _route_summary(selected_artifact)
        direct_summary = _route_summary(direct_artifact)
        selected_crossings = {
            float(row["error_target"]): row for row in selected_summary["crossings"]
        }
        direct_crossings = {
            float(row["error_target"]): row for row in direct_summary["crossings"]
        }
        comparisons: list[dict[str, Any]] = []
        for target in profile.error_targets:
            selected = {
                **selected_crossings[float(target)],
                "algorithm_status": selected_summary["algorithm_status"],
            }
            direct = {
                **direct_crossings[float(target)],
                "algorithm_status": direct_summary["algorithm_status"],
            }
            recommendation = _recommendation(selected=selected, direct=direct)
            counts[recommendation] = counts.get(recommendation, 0) + 1
            comparison = {
                "error_target": float(target),
                "selected_qse": selected,
                "direct_davidson": direct,
                "recommendation": recommendation,
                "recommendation_is_post_run_only": True,
            }
            comparisons.append(comparison)
            decision_rows.append(
                {
                    "case_key": case.case_key,
                    **comparison,
                }
            )
        rows.append(
            {
                "case": case.payload(),
                "selected_qse": selected_summary,
                "direct_davidson": direct_summary,
                "comparisons": comparisons,
            }
        )
    evidence = {
        "schema_version": MATCHED_EVIDENCE_SCHEMA_VERSION,
        "profile_digest": profile_digest,
        "source_lock_digest": source_lock_digest,
        "cell_count": len(rows),
        "route_count": len(artifacts),
        "methods": list(profile.methods),
        "rows": rows,
        "controller_boundary": {
            "post_run_recommendation_only": True,
            "can_rewrite_algorithm_status": False,
            "hybrid_route_present": False,
        },
    }
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "profile_digest": profile_digest,
        "source_lock_digest": source_lock_digest,
        "route_recommendation_counts": dict(sorted(counts.items())),
        "rows": decision_rows,
        "hybrid_enabled": False,
    }
    return evidence, decision


def _decision_markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Paper-III selected-QSE versus direct-Davidson decision table",
        "",
        "Post-run comparison only. Recommendations never rewrite either route's online status.",
        "",
        "| case | error target | selected status | direct status | recommendation |",
        "|---|---:|---|---|---|",
    ]
    for row in decision.get("rows", []):
        lines.append(
            "| {case} | {target:.0e} | {selected} | {direct} | {recommendation} |".format(
                case=row["case_key"],
                target=float(row["error_target"]),
                selected=row["selected_qse"]["algorithm_status"],
                direct=row["direct_davidson"]["algorithm_status"],
                recommendation=row["recommendation"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _snapshot(
    *,
    output_root: Path,
    profile_digest: str,
    source_lock_digest: str,
    completed: Mapping[str, Mapping[str, Any]],
    total: int,
    complete: bool,
) -> MatchedCampaignSnapshot:
    all_keys = [
        _route_key(case, method)
        for case in locked_hh_profile().cases
        for method in locked_hh_profile().methods
    ]
    pending = tuple(key for key in all_keys if key not in completed)
    return MatchedCampaignSnapshot(
        execution_status="COMPLETE" if complete else "PARTIAL",
        profile_digest=profile_digest,
        source_lock_digest=source_lock_digest,
        completed_route_count=len(completed),
        total_route_count=int(total),
        pending_routes=pending,
        matched_evidence_path=(
            Path(output_root) / "matched_routes.json" if complete else None
        ),
        decision_path=Path(output_root) / "decision.json" if complete else None,
    )


def ensure_selected_vs_direct_hh_campaign(
    output_root: Path,
    *,
    executor: MatchedRouteExecutor | None = None,
) -> MatchedCampaignSnapshot:
    """Ensure the locked 24-route HH comparison exists and is validated.

    Repeating this call is resume: validated route artifacts are reused and
    only missing routes execute.  Structured scientific failure statuses are
    admitted as completed evidence; process or artifact failures fail closed.
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    profile = locked_hh_profile()
    route_executor: MatchedRouteExecutor = (
        LocalScientificRouteExecutor() if executor is None else executor
    )
    with _workspace_lock(root):
        profile_digest = _materialize_profile(root, profile)
        _source_lock, source_lock_digest = _materialize_source_lock(root)
        total = len(profile.cases) * len(profile.methods)
        artifacts: dict[str, dict[str, Any]] = {}

        for case in profile.cases:
            for method_id in profile.methods:
                key = _route_key(case, method_id)
                path = _route_path(root, case, method_id)
                if path.is_file():
                    artifact = _validate_route_artifact(
                        _read_json(path),
                        case=case,
                        method_id=method_id,
                        profile=profile,
                        profile_digest=profile_digest,
                        source_lock_digest=source_lock_digest,
                    )
                else:
                    try:
                        result = route_executor.execute(case, method_id, profile)
                        artifact = _build_route_artifact(
                            case=case,
                            method_id=method_id,
                            profile=profile,
                            profile_digest=profile_digest,
                            source_lock_digest=source_lock_digest,
                            result=result,
                        )
                        artifact = _validate_route_artifact(
                            artifact,
                            case=case,
                            method_id=method_id,
                            profile=profile,
                            profile_digest=profile_digest,
                            source_lock_digest=source_lock_digest,
                        )
                        _atomic_write_json(path, artifact)
                    except Exception as exc:
                        partial = _snapshot(
                            output_root=root,
                            profile_digest=profile_digest,
                            source_lock_digest=source_lock_digest,
                            completed=artifacts,
                            total=total,
                            complete=False,
                        ).payload()
                        partial["attention"] = {
                            "route_key": key,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                        _atomic_write_json(root / "status.json", partial)
                        raise
                artifacts[key] = artifact
                _atomic_write_json(
                    root / "status.json",
                    _snapshot(
                        output_root=root,
                        profile_digest=profile_digest,
                        source_lock_digest=source_lock_digest,
                        completed=artifacts,
                        total=total,
                        complete=False,
                    ).payload(),
                )

        evidence, decision = _build_matched_evidence(
            profile=profile,
            profile_digest=profile_digest,
            source_lock_digest=source_lock_digest,
            artifacts=artifacts,
        )
        _atomic_write_json(root / "matched_routes.json", evidence)
        _atomic_write_json(root / "decision.json", decision)
        _atomic_write_text(root / "decision.md", _decision_markdown(decision))
        snapshot = _snapshot(
            output_root=root,
            profile_digest=profile_digest,
            source_lock_digest=source_lock_digest,
            completed=artifacts,
            total=total,
            complete=True,
        )
        _atomic_write_json(root / "status.json", snapshot.payload())
        return snapshot


def _case_for_key(profile: MatchedHHProfile, case_key: str) -> MatchedHHCase:
    matches = [case for case in profile.cases if case.case_key == str(case_key)]
    if len(matches) != 1:
        raise MatchedRouteAcquisitionError(
            f"CASE_IDENTITY_MISMATCH: unknown locked case {case_key!r}."
        )
    return matches[0]


def _validate_case_result(
    payload: Mapping[str, Any],
    *,
    profile: MatchedHHProfile,
    expected_case: MatchedHHCase | None = None,
) -> tuple[MatchedHHCase, dict[str, dict[str, Any]], dict[str, Any], str]:
    bundle = dict(payload)
    if bundle.get("schema_version") != CASE_RESULT_SCHEMA_VERSION:
        raise MatchedRouteAcquisitionError("case-result schema mismatch.")
    if bundle.get("profile") != profile.payload():
        raise MatchedRouteAcquisitionError("PROFILE_DRIFT: case-result profile mismatch.")
    profile_digest = profile.digest()
    if bundle.get("profile_digest") != profile_digest:
        raise MatchedRouteAcquisitionError("PROFILE_DRIFT: case-result digest mismatch.")
    case_payload = bundle.get("case")
    if not isinstance(case_payload, Mapping):
        raise MatchedRouteAcquisitionError("case result lacks case identity.")
    case = _case_for_key(profile, str(case_payload.get("case_key", "")))
    if dict(case_payload) != case.payload() or (
        expected_case is not None and case != expected_case
    ):
        raise MatchedRouteAcquisitionError("CASE_IDENTITY_MISMATCH: case payload drift.")
    source_lock = bundle.get("source_lock")
    if not isinstance(source_lock, Mapping):
        raise MatchedRouteAcquisitionError("case result lacks a source lock.")
    restored_source_lock = dict(source_lock)
    if restored_source_lock.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
        raise MatchedRouteAcquisitionError(
            "SOURCE_LOCK_MISMATCH: case source-lock schema drift."
        )
    source_lock_digest = _canonical_digest(restored_source_lock)
    if bundle.get("source_lock_digest") != source_lock_digest:
        raise MatchedRouteAcquisitionError("SOURCE_LOCK_MISMATCH: case digest drift.")
    routes = bundle.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != set(profile.methods):
        raise MatchedRouteAcquisitionError(
            "METHOD_SET_VIOLATION: case result must contain both locked routes."
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for method_id in profile.methods:
        raw_artifact = routes.get(method_id)
        if not isinstance(raw_artifact, Mapping):
            raise MatchedRouteAcquisitionError("case result route must be a mapping.")
        artifacts[method_id] = _validate_route_artifact(
            raw_artifact,
            case=case,
            method_id=method_id,
            profile=profile,
            profile_digest=profile_digest,
            source_lock_digest=source_lock_digest,
        )
    return case, artifacts, restored_source_lock, source_lock_digest


def acquire_selected_vs_direct_hh_case(
    case_key: str,
    output_json: Path,
    *,
    executor: MatchedRouteExecutor | None = None,
) -> dict[str, Any]:
    """Acquire both locked routes for one HH case as one resumable result."""

    profile = locked_hh_profile()
    case = _case_for_key(profile, case_key)
    path = Path(output_json)
    if path.is_file():
        existing = _read_json(path)
        _validate_case_result(existing, profile=profile, expected_case=case)
        current = _current_source_lock()
        if existing["source_lock"].get("relevant_path_sha256") != current.get(
            "relevant_path_sha256"
        ):
            raise MatchedRouteAcquisitionError(
                "SOURCE_LOCK_MISMATCH: existing case result uses different source."
            )
        return existing
    source_lock = _current_source_lock()
    source_lock_digest = _canonical_digest(source_lock)
    profile_digest = profile.digest()
    route_executor: MatchedRouteExecutor = (
        LocalScientificRouteExecutor() if executor is None else executor
    )
    routes: dict[str, dict[str, Any]] = {}
    for method_id in profile.methods:
        result = route_executor.execute(case, method_id, profile)
        artifact = _build_route_artifact(
            case=case,
            method_id=method_id,
            profile=profile,
            profile_digest=profile_digest,
            source_lock_digest=source_lock_digest,
            result=result,
        )
        routes[method_id] = _validate_route_artifact(
            artifact,
            case=case,
            method_id=method_id,
            profile=profile,
            profile_digest=profile_digest,
            source_lock_digest=source_lock_digest,
        )
    bundle = {
        "schema_version": CASE_RESULT_SCHEMA_VERSION,
        "profile": profile.payload(),
        "profile_digest": profile_digest,
        "source_lock": source_lock,
        "source_lock_digest": source_lock_digest,
        "case": case.payload(),
        "routes": routes,
    }
    _validate_case_result(bundle, profile=profile, expected_case=case)
    _atomic_write_json(path, bundle)
    return bundle


def collect_selected_vs_direct_hh_cases(
    case_result_paths: Sequence[Path],
    output_root: Path,
) -> MatchedCampaignSnapshot:
    """Validate and collect the complete 12-case distributed campaign."""

    profile = locked_hh_profile()
    expected_keys = {case.case_key for case in profile.cases}
    bundles: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    common_source_lock: dict[str, Any] | None = None
    common_source_lock_digest: str | None = None
    for path in case_result_paths:
        bundle = _read_json(Path(path))
        case, case_artifacts, source_lock, source_lock_digest = _validate_case_result(
            bundle,
            profile=profile,
        )
        if case.case_key in bundles:
            raise MatchedRouteAcquisitionError(
                f"CASE_IDENTITY_MISMATCH: duplicate case {case.case_key}."
            )
        if common_source_lock is None:
            common_source_lock = source_lock
            common_source_lock_digest = source_lock_digest
        elif source_lock != common_source_lock:
            raise MatchedRouteAcquisitionError(
                "SOURCE_LOCK_MISMATCH: case results were built from different sources."
            )
        bundles[case.case_key] = bundle
        for method_id, artifact in case_artifacts.items():
            artifacts[_route_key(case, method_id)] = artifact
    missing = sorted(expected_keys - set(bundles))
    extra = sorted(set(bundles) - expected_keys)
    if missing or extra or len(case_result_paths) != len(profile.cases):
        raise MatchedRouteAcquisitionError(
            "CASE_MATRIX_INCOMPLETE: expected exactly the 12 locked cases; "
            f"missing={missing}, extra={extra}."
        )
    assert common_source_lock is not None
    assert common_source_lock_digest is not None
    current = _current_source_lock()
    if current.get("relevant_path_sha256") != common_source_lock.get(
        "relevant_path_sha256"
    ):
        raise MatchedRouteAcquisitionError(
            "SOURCE_LOCK_MISMATCH: collector source differs from case workers."
        )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with _workspace_lock(root):
        _materialize_profile(root, profile)
        source_path = root / "source_lock.json"
        if source_path.is_file() and _read_json(source_path) != common_source_lock:
            raise MatchedRouteAcquisitionError(
                "SOURCE_LOCK_MISMATCH: collection output already has another lock."
            )
        if not source_path.is_file():
            _atomic_write_json(source_path, common_source_lock)
        for case_key, bundle in bundles.items():
            _atomic_write_json(root / "cases" / f"{case_key}.json", bundle)
        evidence, decision = _build_matched_evidence(
            profile=profile,
            profile_digest=profile.digest(),
            source_lock_digest=common_source_lock_digest,
            artifacts=artifacts,
        )
        _atomic_write_json(root / "matched_routes.json", evidence)
        _atomic_write_json(root / "decision.json", decision)
        _atomic_write_text(root / "decision.md", _decision_markdown(decision))
        snapshot = _snapshot(
            output_root=root,
            profile_digest=profile.digest(),
            source_lock_digest=common_source_lock_digest,
            completed=artifacts,
            total=len(profile.cases) * len(profile.methods),
            complete=True,
        )
        _atomic_write_json(root / "status.json", snapshot.payload())
        return snapshot


class LocalScientificRouteExecutor:
    """Default in-process adapter for the two locked scientific routes."""

    def __init__(self) -> None:
        self._cached_case_key: str | None = None
        self._cached_problem: Any = None

    def _problem(self, case: MatchedHHCase, profile: MatchedHHProfile) -> Any:
        if self._cached_case_key == case.case_key and self._cached_problem is not None:
            return self._cached_problem
        from pipelines.qse_spectra.paper_iii_problem import load_problem

        problem = load_problem(
            regime=case.regime,
            u=case.u,
            g_ep=case.g_ep,
            n_ph_max=case.n_ph_max,
            target_roots=len(profile.target_roots),
        )
        self._cached_case_key = case.case_key
        self._cached_problem = problem
        return problem

    @staticmethod
    def _split_rows(problem: Any, rows: list[dict[str, Any]]) -> tuple[
        tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]
    ]:
        online_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for raw in rows:
            row = dict(raw)
            audit_rows.append(
                {
                    "k": int(row["k"]),
                    "max_root_abs_error": row.pop("max_root_abs_error", None),
                    "per_root_abs_error": row.pop("per_root_abs_error", None),
                }
            )
            if "estimator" not in row:
                indices = row.get("support_indices")
                if not isinstance(indices, list):
                    raise MatchedRouteAcquisitionError(
                        "selected-QSE rung lacks support indices for estimator costing."
                    )
                estimator = dict(problem.estimator_cost(indices))
                estimator["queries"] = int(estimator["qwc_groups"])
                estimator["convention"] = "record_pencil_qwc_basis_cover"
                row["estimator"] = estimator
            online_rows.append(row)
        return tuple(online_rows), tuple(audit_rows)

    def execute(
        self,
        case: MatchedHHCase,
        method_id: str,
        profile: MatchedHHProfile,
    ) -> RouteExecutionResult:
        if method_id not in LOCKED_METHODS:
            raise MatchedRouteAcquisitionError(
                f"METHOD_SET_VIOLATION: unsupported route {method_id!r}."
            )
        from pipelines.exact_bench import paper_iii_growth_trace_campaign as growth

        problem = self._problem(case, profile)
        started = time.monotonic()
        if method_id == METHOD_SELECTED_QSE:
            from pipelines.qse_spectra.record_selection import (
                StaticRecordSelectionConfig,
                select_static_qse_records,
            )

            marker = select_static_qse_records(
                problem.basis,
                config=StaticRecordSelectionConfig(
                    mode="geometry_selected",
                    max_records=len(problem.basis),
                    geometry_target_roots=len(profile.target_roots),
                    geometry_cost_discount_alpha=1.0,
                    geometry_residual_stop=float(growth.RESIDUAL_MARKER),
                ),
                hamiltonian=problem.hamiltonian,
                prepared_state=problem.ground,
                basis_vector_policy=growth._Q0_POLICY,
                compiled_costs=problem.costs,
            )
            order = [int(index) for index in marker.selected_original_indices]
            record_stop = dict(marker.geometry_stop or {"stop_reason": "missing"})
            rows, completion = growth.trace_selected_qse_fixed_pool(
                problem,
                order,
                k_max=profile.dimension_cap,
                stride=profile.trace_stride,
                record_stop=record_stop,
                target_root_count=len(profile.target_roots),
            )
        else:
            rows, completion = growth.trace_direct_davidson_with_completion(
                problem,
                k_max=profile.dimension_cap,
                target_root_count=len(profile.target_roots),
            )
        online_rows, audit_rows = self._split_rows(problem, rows)
        return RouteExecutionResult(
            online_rows=online_rows,
            completion=dict(completion),
            audit_rows=audit_rows,
            reference_excitations=tuple(float(value) for value in problem.references),
            pool_digest=str(problem.pool_digest),
            elapsed_seconds=float(time.monotonic() - started),
        )


__all__ = [
    "AUDIT_STATUS_NOT_REACHED",
    "AUDIT_STATUS_REACHED",
    "CASE_RESULT_SCHEMA_VERSION",
    "DECISION_SCHEMA_VERSION",
    "DIRECT_TERMINAL_STATUSES",
    "DIRECT_STATUS_DIMENSION_CAP_REACHED",
    "DIRECT_STATUS_NO_NOVEL_CORRECTION",
    "DIRECT_STATUS_RESIDUAL_CERTIFIED",
    "EXACT_AUDIT_SCHEMA_VERSION",
    "LOCKED_CONSTRUCTION_POLICIES",
    "LOCKED_ERROR_TARGETS",
    "LOCKED_METHODS",
    "LocalScientificRouteExecutor",
    "MATCHED_EVIDENCE_SCHEMA_VERSION",
    "METHOD_DIRECT_DAVIDSON",
    "METHOD_SELECTED_QSE",
    "MatchedCampaignSnapshot",
    "MatchedHHCase",
    "MatchedHHProfile",
    "MatchedRouteAcquisitionError",
    "MatchedRouteExecutor",
    "ONLINE_RECEIPT_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION",
    "RECOMMEND_DIRECT",
    "RECOMMEND_SELECTED",
    "RECOMMEND_UNRESOLVED",
    "ROUTE_ARTIFACT_SCHEMA_VERSION",
    "RouteExecutionResult",
    "SELECTED_STATUS_CERTIFIED",
    "SELECTED_STATUS_POOL_EXHAUSTED",
    "SELECTED_STATUS_WINDOW_NOT_CERTIFIED",
    "SOURCE_LOCK_SCHEMA_VERSION",
    "STATUS_SCHEMA_VERSION",
    "acquire_selected_vs_direct_hh_case",
    "collect_selected_vs_direct_hh_cases",
    "ensure_selected_vs_direct_hh_campaign",
    "locked_hh_profile",
]
