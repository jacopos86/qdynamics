#!/usr/bin/env python3
"""Diagnostic-only external dynamics parity runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from pipelines.exact_bench.external_dynamics.local_reference_manager import (
    ResolvedExternalDynamicsReference,
    resolve_external_dynamics_reference,
)
from pipelines.exact_bench.external_dynamics.native_diagnostic_inputs import (
    RHSLimitTolerance,
    run_avqds_t_rhs_limit_diagnostic,
)

EXTERNAL_DYNAMICS_PARITY_SCHEMA = "dynamics_external_parity_v1"
EXTERNAL_DYNAMICS_PARITY_SUMMARY_SCHEMA = "dynamics_external_parity_summary_v1"
DIAGNOSTIC_POLICY = "diagnostic_only_not_paper_evidence"
CHECK_IDS: tuple[str, ...] = (
    "adaptive_pvqd_external",
    "avqds_external",
    "avqds_t_rhs_limit",
)


@dataclass(frozen=True)
class DynamicsParityTolerances:
    tangent_l2_tol: float = 5.0e-3
    theta_dot_abs_tol: float = 5.0e-3
    residual_ratio_abs_tol: float = 5.0e-3

    def rhs_limit(self) -> RHSLimitTolerance:
        return RHSLimitTolerance(
            tangent_l2_tol=float(self.tangent_l2_tol),
            theta_dot_abs_tol=float(self.theta_dot_abs_tol),
            residual_ratio_abs_tol=float(self.residual_ratio_abs_tol),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _guardrails() -> dict[str, Any]:
    return {
        "policy": DIAGNOSTIC_POLICY,
        "paper_facing_table_evidence": False,
        "physical_accuracy_claim": False,
        "controller_decisions_modified": False,
        "exact_reference_decision_input": False,
        "normal_dynamics_row_status_modified": False,
    }


def _base_result(
    *,
    check_id: str,
    algorithm_id: str,
    reference_id: str,
    status: str,
    passed: bool | None,
    reason: str,
    reference: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": EXTERNAL_DYNAMICS_PARITY_SCHEMA,
        "created_at": _utc_now(),
        "check_id": str(check_id),
        "algorithm_id": str(algorithm_id),
        "reference_id": str(reference_id),
        "status": str(status),
        "passed": None if passed is None else bool(passed),
        "reason": str(reason),
        "guardrails": _guardrails(),
        "reference": {} if reference is None else dict(reference),
        "details": {} if details is None else dict(details),
    }


def _skip_from_resolution(*, check_id: str, algorithm_id: str, resolved: ResolvedExternalDynamicsReference) -> dict[str, Any]:
    return _base_result(
        check_id=check_id,
        algorithm_id=algorithm_id,
        reference_id=resolved.reference_id,
        status=resolved.status,
        passed=None,
        reason=resolved.reason,
        reference=resolved.to_dict(),
    )


def _worker_module_for_check(check_id: str) -> str:
    if check_id == "adaptive_pvqd_external":
        return "pipelines.exact_bench.external_dynamics.adaptive_pvqd_worker"
    if check_id == "avqds_external":
        return "pipelines.exact_bench.external_dynamics.avqds_worker"
    raise ValueError(f"No external worker for check_id={check_id!r}")


def _reference_id_for_check(check_id: str) -> str:
    if check_id == "adaptive_pvqd_external":
        return "dalin27_adaptive_pvqd"
    if check_id == "avqds_external":
        return "gqce_ames_avqds"
    if check_id == "avqds_t_rhs_limit":
        return "repo_native_avqds_t_rhs_limit"
    raise ValueError(f"Unknown dynamics parity check {check_id!r}")


def _algorithm_id_for_check(check_id: str) -> str:
    if check_id == "adaptive_pvqd_external":
        return "dyn_adaptive_pvqd"
    if check_id == "avqds_external":
        return "dyn_avqds"
    if check_id == "avqds_t_rhs_limit":
        return "dyn_avqds_t"
    raise ValueError(f"Unknown dynamics parity check {check_id!r}")


def _run_external_source_probe(
    *,
    check_id: str,
    output_dir: Path,
    manifest_path: str | Path | None,
    reference_root: str | Path | None,
    allow_unpinned_local_reference: bool,
) -> dict[str, Any]:
    algorithm_id = _algorithm_id_for_check(check_id)
    reference_id = _reference_id_for_check(check_id)
    resolved = resolve_external_dynamics_reference(
        reference_id,
        manifest_path=manifest_path,
        reference_root=reference_root,
        allow_unpinned_local_reference=allow_unpinned_local_reference,
    )
    check_dir = output_dir / check_id
    check_dir.mkdir(parents=True, exist_ok=True)
    write_json(check_dir / "reference_resolution.json", resolved.to_dict())
    if resolved.status != "available" or not resolved.checkout_path:
        result = _skip_from_resolution(check_id=check_id, algorithm_id=algorithm_id, resolved=resolved)
        write_json(check_dir / "result.json", result)
        return result

    worker_output = check_dir / "worker_result.json"
    command = [
        sys.executable,
        "-m",
        _worker_module_for_check(check_id),
        "--checkout",
        str(resolved.checkout_path),
        "--output",
        str(worker_output),
    ]
    write_json(check_dir / "worker_command.json", {"command": command})
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0 or not worker_output.exists():
        result = _base_result(
            check_id=check_id,
            algorithm_id=algorithm_id,
            reference_id=reference_id,
            status="skipped_worker_failed",
            passed=None,
            reason=f"external source probe worker failed with return code {proc.returncode}",
            reference=resolved.to_dict(),
            details={"stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]},
        )
        write_json(check_dir / "result.json", result)
        return result
    try:
        worker_payload = json.loads(worker_output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result = _base_result(
            check_id=check_id,
            algorithm_id=algorithm_id,
            reference_id=reference_id,
            status="skipped_worker_protocol_error",
            passed=None,
            reason=f"worker output was not valid JSON: {exc}",
            reference=resolved.to_dict(),
        )
        write_json(check_dir / "result.json", result)
        return result
    status = str(worker_payload.get("status", "skipped_worker_protocol_error"))
    result = _base_result(
        check_id=check_id,
        algorithm_id=algorithm_id,
        reference_id=reference_id,
        status=status,
        passed=worker_payload.get("passed") if isinstance(worker_payload.get("passed"), bool) else None,
        reason=str(worker_payload.get("reason", "external source probe completed")),
        reference=resolved.to_dict(),
        details={"worker": worker_payload},
    )
    result["provenance_status"] = (
        "verified_pinned_checkout" if resolved.provenance_verified else "unverified_local_reference"
    )
    write_json(check_dir / "result.json", result)
    return result


def _run_avqds_t_rhs_limit(
    *,
    output_dir: Path,
    tolerances: DynamicsParityTolerances,
    dt_scales: Sequence[float],
) -> dict[str, Any]:
    check_id = "avqds_t_rhs_limit"
    details = run_avqds_t_rhs_limit_diagnostic(
        dt_scales=tuple(float(x) for x in dt_scales),
        tolerance=tolerances.rhs_limit(),
    )
    result = _base_result(
        check_id=check_id,
        algorithm_id="dyn_avqds_t",
        reference_id="repo_native_avqds_t_rhs_limit",
        status=str(details.get("status", "failed")),
        passed=details.get("passed") if isinstance(details.get("passed"), bool) else None,
        reason=str(details.get("reason", "AVQDS-T/RHS diagnostic completed")),
        reference={"reference_id": "repo_native_avqds_t_rhs_limit", "policy": DIAGNOSTIC_POLICY},
        details=details,
    )
    check_dir = output_dir / check_id
    write_json(check_dir / "native_diagnostic.json", details)
    write_json(check_dir / "result.json", result)
    return result


def normalize_checks(checks: Iterable[str] | str | None) -> tuple[str, ...]:
    if checks is None:
        return CHECK_IDS
    if isinstance(checks, str):
        raw = [item.strip() for item in checks.split(",")]
    else:
        raw = [str(item).strip() for item in checks]
    if any(item == "all" for item in raw) or not raw:
        return CHECK_IDS
    unknown = [item for item in raw if item not in CHECK_IDS]
    if unknown:
        raise ValueError(f"Unknown external dynamics parity checks: {unknown}; known={CHECK_IDS}")
    return tuple(raw)


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [row for row in results if row.get("passed") is False or str(row.get("status")) == "failed"]
    skipped = [row for row in results if str(row.get("status", "")).startswith("skipped")]
    source_probe = [row for row in results if str(row.get("status")) == "completed_source_probe"]
    completed = [row for row in results if str(row.get("status")) == "completed"]
    if failed:
        status = "failed"
        passed: bool | None = False
    elif len(completed) == len(results):
        status = "completed"
        passed = True
    else:
        status = "completed_with_diagnostic_skips_or_source_probes"
        passed = None
    return {
        "schema": EXTERNAL_DYNAMICS_PARITY_SUMMARY_SCHEMA,
        "created_at": _utc_now(),
        "status": status,
        "passed": passed,
        "policy": DIAGNOSTIC_POLICY,
        "paper_facing_table_evidence": False,
        "result_count": int(len(results)),
        "completed_count": int(len(completed)),
        "source_probe_count": int(len(source_probe)),
        "skipped_count": int(len(skipped)),
        "failed_count": int(len(failed)),
        "checks": [
            {
                "check_id": row.get("check_id"),
                "algorithm_id": row.get("algorithm_id"),
                "reference_id": row.get("reference_id"),
                "status": row.get("status"),
                "passed": row.get("passed"),
                "reason": row.get("reason"),
            }
            for row in results
        ],
    }


def run_dynamics_parity_checks(
    *,
    checks: Iterable[str] | str | None = None,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    reference_root: str | Path | None = None,
    allow_unpinned_local_reference: bool = False,
    dt_scales: Sequence[float] = (0.2, 0.1, 0.05),
    tolerances: DynamicsParityTolerances | None = None,
) -> dict[str, Any]:
    """Run diagnostic-only external dynamics parity checks."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tol = tolerances or DynamicsParityTolerances()
    selected = normalize_checks(checks)
    results: list[dict[str, Any]] = []
    for check_id in selected:
        if check_id == "avqds_t_rhs_limit":
            result = _run_avqds_t_rhs_limit(output_dir=out, tolerances=tol, dt_scales=dt_scales)
        else:
            result = _run_external_source_probe(
                check_id=check_id,
                output_dir=out,
                manifest_path=manifest_path,
                reference_root=reference_root,
                allow_unpinned_local_reference=allow_unpinned_local_reference,
            )
        results.append(result)
    summary = summarize_results(results)
    write_json(out / "external_dynamics_parity.json", {"summary": summary, "results": results})
    write_json(out / "external_dynamics_parity_summary.json", summary)
    return {"summary": summary, "results": results, "output_dir": str(out)}


__all__ = [
    "CHECK_IDS",
    "DIAGNOSTIC_POLICY",
    "EXTERNAL_DYNAMICS_PARITY_SCHEMA",
    "EXTERNAL_DYNAMICS_PARITY_SUMMARY_SCHEMA",
    "DynamicsParityTolerances",
    "normalize_checks",
    "run_dynamics_parity_checks",
    "summarize_results",
]
