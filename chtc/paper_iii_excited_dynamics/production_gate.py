#!/usr/bin/env python3
"""Read-only Paper III compatibility/production gate.

This module consumes compatibility-matrix artifacts that already exist on disk.
It never stages inputs, submits CHTC work, repairs seed artifacts, or writes
report/source-map outputs.  Future QSE production, Optuna production,
source-map, and report builders can call this module to fail closed before they
interpret compatibility evidence as production-ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_iii_excited_dynamics import validate_outputs  # noqa: E402

PRODUCTION_GATE_SCHEMA_VERSION = "paper_iii_qse_production_gate_v1"
DEFAULT_NPH1_DIR = Path("chtc/paper_iii_excited_dynamics/input/compatibility_matrix_nph1")
DEFAULT_NPH2_DIR = Path("chtc/paper_iii_excited_dynamics/input/compatibility_matrix_nph2")
REQUIRED_TARGET_EXCITED_ROOTS = 6
CONSUMER_IDS = (
    "qse_production_mode",
    "optuna_production_mode",
    "source_map_generation",
    "report_building",
)
EXACT_DIAGNOSTIC_ROLE = "report_only_never_controller_decision_input"
STRICT_POLICY_EXPECTED: dict[str, Any] = {
    "controller_exact_input_mode": "off",
    "diagnostic_exact_reference_mode": "benchmark_exact",
    "uses_reference_for_decision": False,
    "uses_future_exact_forecast_for_decision": False,
    "exact_decision_checkpoints": 0,
    "strict_measurement_oracle_certified": True,
    "qpu_faithful_decisions_passed": True,
}
ROW_EXACT_EXPECTED: dict[str, str] = {
    "controller_exact_input_mode": "off",
    "diagnostic_exact_reference_mode": "benchmark_exact",
    "uses_reference_for_decision": "false",
    "uses_future_exact_forecast_for_decision": "false",
    "exact_decision_checkpoints": "0",
    "strict_measurement_oracle_certified": "true",
    "qpu_faithful_decisions_passed": "true",
}


class ProductionGateError(RuntimeError):
    """Raised when a consumer requires production-ready compatibility evidence."""


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def _split_codes(value: Any) -> list[str]:
    return [item.strip() for item in _clean(value).replace(",", ";").split(";") if item.strip()]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return _clean(value).lower()


def _counter_from_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw_count in value.items():
        try:
            out[str(key)] = int(raw_count)
        except Exception:
            continue
    return dict(sorted(out.items()))


def _summarize_validation_report(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "compatibility_matrix_dir",
        "profile",
        "matrix_batch_id",
        "ok",
        "errors",
        "warnings",
        "record_count",
        "comparator_row_count",
        "scope_decision_row_count",
        "blocker_count",
        "family_count",
        "case_count",
        "comparator_count",
        "seed_repair_unique_slot_count",
        "seed_repair_expected_unique_slot_count",
        "no_submit",
    )
    return {key: report.get(key) for key in keys if key in report}


def _load_matrix_payloads(root: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[Mapping[str, Any]], list[str]]:
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    records: list[dict[str, str]] = []
    blockers: list[Mapping[str, Any]] = []
    try:
        raw_manifest = _load_json(root / "manifest.json")
        if isinstance(raw_manifest, dict):
            manifest = raw_manifest
        else:
            errors.append("manifest.json root must be an object")
    except Exception as exc:
        errors.append(f"failed to load manifest.json: {type(exc).__name__}: {exc}")
    try:
        records = _load_tsv(root / "records.tsv")
    except Exception as exc:
        errors.append(f"failed to load records.tsv: {type(exc).__name__}: {exc}")
    try:
        blockers_payload = _load_json(root / "blockers.json")
        raw_blockers = blockers_payload.get("blockers") if isinstance(blockers_payload, Mapping) else None
        if isinstance(raw_blockers, list):
            blockers = [row for row in raw_blockers if isinstance(row, Mapping)]
        else:
            errors.append("blockers.json must contain a blockers list")
    except Exception as exc:
        errors.append(f"failed to load blockers.json: {type(exc).__name__}: {exc}")
    return manifest, records, blockers, errors


def _missing_comparator_blockers(
    records: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_code: Counter[str] = Counter()
    by_comparator: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    for row in blockers:
        code = _clean(row.get("code"))
        if not code.startswith("comparator."):
            continue
        comparator_id = _clean(row.get("comparator_id")) or "unknown"
        by_code[code] += 1
        by_comparator[comparator_id] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "record_id": _clean(row.get("record_id")),
                    "family": _clean(row.get("family")),
                    "paper_i_case_id": _clean(row.get("paper_i_case_id")),
                    "drive_amplitude": _clean(row.get("drive_amplitude")),
                    "comparator_id": comparator_id,
                    "code": code,
                    "detail": _clean(row.get("detail")),
                }
            )

    if not by_code:
        # Fallback for partially materialized fixtures that carry blocker codes
        # only in records.tsv.
        for row in records:
            comparator_id = _clean(row.get("comparator_id")) or "unknown"
            for code in _split_codes(row.get("blocker_codes")):
                if code.startswith("comparator."):
                    by_code[code] += 1
                    by_comparator[comparator_id] += 1
                    if len(examples) < 12:
                        examples.append(
                            {
                                "record_id": _clean(row.get("record_id")),
                                "family": _clean(row.get("family")),
                                "paper_i_case_id": _clean(row.get("paper_i_case_id")),
                                "drive_amplitude": _clean(row.get("drive_amplitude")),
                                "comparator_id": comparator_id,
                                "code": code,
                                "detail": "",
                            }
                        )

    return {
        "status": "pass" if not by_code else "blocked",
        "blocker_count": sum(by_code.values()),
        "codes": sorted(by_code),
        "by_code": dict(sorted(by_code.items())),
        "by_comparator": dict(sorted(by_comparator.items())),
        "examples": examples,
    }


def _target_root_status(
    manifest: Mapping[str, Any],
    comparator_rows: Sequence[Mapping[str, Any]],
    *,
    required_target_excited_roots: int,
) -> dict[str, Any]:
    manifest_value = _int_or_none(manifest.get("target_excited_roots"))
    row_values = sorted({_int_or_none(row.get("target_excited_roots")) for row in comparator_rows})
    row_values = [value for value in row_values if value is not None]
    violations: list[str] = []
    if manifest_value != required_target_excited_roots:
        violations.append(
            f"manifest.target_excited_roots expected {required_target_excited_roots}, got {manifest_value!r}"
        )
    if row_values != [required_target_excited_roots]:
        violations.append(
            f"comparator row target_excited_roots expected only {required_target_excited_roots}, got {row_values}"
        )
    if not comparator_rows:
        violations.append("records.tsv contains no comparator rows")
    return {
        "status": "pass" if not violations else "fail",
        "required": required_target_excited_roots,
        "manifest": manifest_value,
        "comparator_row_values": row_values,
        "violations": violations,
    }


def _exact_boundary_status(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    violations: list[str] = []
    if _clean(manifest.get("exact_ed_qse_diagnostics_role")) != EXACT_DIAGNOSTIC_ROLE:
        violations.append(
            "manifest.exact_ed_qse_diagnostics_role must be "
            f"{EXACT_DIAGNOSTIC_ROLE!r}, got {_clean(manifest.get('exact_ed_qse_diagnostics_role'))!r}"
        )
    strict_policy = manifest.get("strict_policy") if isinstance(manifest.get("strict_policy"), Mapping) else {}
    for key, expected in STRICT_POLICY_EXPECTED.items():
        actual = strict_policy.get(key) if isinstance(strict_policy, Mapping) else None
        if actual != expected:
            violations.append(f"manifest.strict_policy.{key} expected {expected!r}, got {actual!r}")

    exact_modes: Counter[str] = Counter()
    diagnostic_modes: Counter[str] = Counter()
    strict_statuses: Counter[str] = Counter()
    for row in records:
        record_id = _clean(row.get("record_id")) or "<missing-record-id>"
        row_kind = _clean(row.get("row_kind"))
        exact_modes[_clean(row.get("controller_exact_input_mode"))] += 1
        diagnostic_modes[_clean(row.get("diagnostic_exact_reference_mode"))] += 1
        strict_statuses[_clean(row.get("strict_policy_status"))] += 1
        for key, expected in ROW_EXACT_EXPECTED.items():
            actual = _bool_text(row.get(key)) if expected in {"true", "false"} else _clean(row.get(key))
            if actual != expected:
                violations.append(f"{record_id}: {key} expected {expected!r}, got {actual!r}")
        if row_kind == "comparator_row" and _clean(row.get("strict_policy_status")) != "pass":
            violations.append(
                f"{record_id}: comparator row strict_policy_status expected 'pass', "
                f"got {_clean(row.get('strict_policy_status'))!r}"
            )
    return {
        "status": "pass" if not violations else "fail",
        "violation_count": len(violations),
        "violations": violations[:50],
        "controller_exact_input_mode_counts": dict(sorted(exact_modes.items())),
        "diagnostic_exact_reference_mode_counts": dict(sorted(diagnostic_modes.items())),
        "strict_policy_status_counts": dict(sorted(strict_statuses.items())),
        "diagnostic_exact_role": _clean(manifest.get("exact_ed_qse_diagnostics_role")),
    }


def _matrix_status(
    matrix_dir: str | Path,
    *,
    n_ph: int,
    required_target_excited_roots: int,
) -> dict[str, Any]:
    root = Path(matrix_dir)
    status: dict[str, Any] = {
        "n_ph": n_ph,
        "matrix_dir": str(root),
        "exists": root.exists(),
        "read_only": True,
        "mutates_artifacts": False,
        "validation_ok": False,
        "errors": [],
        "warnings": [],
    }
    if not root.exists():
        status["errors"] = [f"n_ph={n_ph} compatibility matrix directory is missing: {root}"]
        status["first_pass_status"] = "missing" if n_ph == 1 else "not_applicable"
        status["production_status"] = "missing" if n_ph >= 2 else "not_applicable"
        status["first_pass_evidence_ready"] = False
        status["production_ready"] = False
        return status

    validation_report = validate_outputs.validate_compatibility_matrix_dir(root)
    status["validation"] = _summarize_validation_report(validation_report)
    status["validation_ok"] = bool(validation_report.get("ok"))
    status["warnings"] = list(validation_report.get("warnings") or [])
    status["errors"] = list(validation_report.get("errors") or [])

    manifest, records, blockers, load_errors = _load_matrix_payloads(root)
    status["errors"].extend(load_errors)
    loaded = not load_errors
    comparator_rows = [row for row in records if _clean(row.get("row_kind")) == "comparator_row"]
    scope_rows = [row for row in records if _clean(row.get("row_kind")) == "scope_decision"]
    observed_n_ph = _int_or_none(manifest.get("n_ph_max"))
    if observed_n_ph != n_ph:
        status["errors"].append(f"manifest.n_ph_max expected {n_ph}, got {observed_n_ph!r}")

    blocker_code_counts = _counter_from_mapping(manifest.get("blocker_code_counts"))
    if not blocker_code_counts:
        blocker_code_counts = dict(sorted(Counter(_clean(row.get("code")) for row in blockers if _clean(row.get("code"))).items()))
    row_status_counts = _counter_from_mapping(manifest.get("status_counts"))
    if not row_status_counts:
        row_status_counts = dict(sorted(Counter(_clean(row.get("expected_status")) for row in records).items()))
    comparator_status_counts = _counter_from_mapping(manifest.get("comparator_status_counts"))
    if not comparator_status_counts:
        comparator_status_counts = dict(sorted(Counter(_clean(row.get("comparator_status")) for row in comparator_rows).items()))

    target_status = _target_root_status(
        manifest,
        comparator_rows,
        required_target_excited_roots=required_target_excited_roots,
    )
    exact_status = _exact_boundary_status(manifest, records)
    missing_comparators = _missing_comparator_blockers(records, blockers)
    total_blockers = sum(blocker_code_counts.values()) if blocker_code_counts else len(blockers)
    blocked_rows = row_status_counts.get("blocked", 0)
    compatibility_ready = loaded and status["validation_ok"] and observed_n_ph == n_ph
    first_pass_evidence_ready = (
        n_ph == 1
        and compatibility_ready
        and target_status["status"] == "pass"
        and exact_status["status"] == "pass"
    )
    production_ready = (
        n_ph >= 2
        and compatibility_ready
        and target_status["status"] == "pass"
        and exact_status["status"] == "pass"
        and total_blockers == 0
        and blocked_rows == 0
        and missing_comparators["blocker_count"] == 0
    )

    if n_ph == 1:
        if not first_pass_evidence_ready:
            first_pass_status = "blocked"
        elif total_blockers:
            first_pass_status = "available_with_blockers"
        else:
            first_pass_status = "ready"
    else:
        first_pass_status = "not_applicable"
    if n_ph >= 2:
        if production_ready:
            production_status = "ready"
        elif not compatibility_ready:
            production_status = "invalid"
        elif target_status["status"] != "pass" or exact_status["status"] != "pass":
            production_status = "blocked"
        elif total_blockers or blocked_rows:
            production_status = "blocked_by_compatibility"
        else:
            production_status = "blocked"
    else:
        production_status = "not_applicable"

    status.update(
        {
            "profile": manifest.get("profile"),
            "matrix_batch_id": manifest.get("matrix_batch_id"),
            "manifest_n_ph_max": observed_n_ph,
            "target_excited_roots": target_status,
            "exact_reference_boundary": exact_status,
            "missing_comparator_blockers": missing_comparators,
            "record_count": len(records),
            "comparator_row_count": len(comparator_rows),
            "scope_decision_row_count": len(scope_rows),
            "blocker_count": total_blockers,
            "blocker_code_counts": blocker_code_counts,
            "row_status_counts": row_status_counts,
            "comparator_status_counts": comparator_status_counts,
            "first_pass_status": first_pass_status,
            "first_pass_evidence_ready": first_pass_evidence_ready,
            "production_status": production_status,
            "production_ready": production_ready,
        }
    )
    return status


def _combined_exact_boundary(tiers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {
        _clean(tier.get("exact_reference_boundary", {}).get("status"))
        for tier in tiers
        if isinstance(tier.get("exact_reference_boundary"), Mapping)
    }
    violations = [
        violation
        for tier in tiers
        if isinstance(tier.get("exact_reference_boundary"), Mapping)
        for violation in tier["exact_reference_boundary"].get("violations", [])
    ]
    if "fail" in statuses:
        status = "fail"
    elif statuses == {"pass"}:
        status = "pass"
    else:
        status = "missing"
    return {
        "status": status,
        "violation_count": len(violations),
        "violations": violations[:50],
    }


def validate_production_gate(
    *,
    nph1_dir: str | Path = DEFAULT_NPH1_DIR,
    nph2_dir: str | Path = DEFAULT_NPH2_DIR,
    required_target_excited_roots: int = REQUIRED_TARGET_EXCITED_ROOTS,
) -> dict[str, Any]:
    """Return a read-only fail-closed production-gate report.

    The report is intentionally JSON-serializable so production CLI modes,
    Optuna, source-map generation, and report builders can store or inspect the
    result without importing dataclasses.  The function reads existing matrix
    artifacts only and never writes to the matrix directories.
    """

    nph1_status = _matrix_status(
        nph1_dir,
        n_ph=1,
        required_target_excited_roots=required_target_excited_roots,
    )
    nph2_status = _matrix_status(
        nph2_dir,
        n_ph=2,
        required_target_excited_roots=required_target_excited_roots,
    )
    production_ready = bool(nph2_status.get("production_ready"))
    first_pass_ready = bool(nph1_status.get("first_pass_evidence_ready"))
    exact_boundary = _combined_exact_boundary([nph1_status, nph2_status])
    consumer_fail_closed = {consumer_id: not production_ready for consumer_id in CONSUMER_IDS}
    errors = [
        f"n_ph=1: {error}" for error in nph1_status.get("errors", [])
    ] + [
        f"n_ph=2: {error}" for error in nph2_status.get("errors", [])
    ]
    if not first_pass_ready:
        errors.append("n_ph=1 first-pass compatibility evidence is not ready")
    if not production_ready:
        errors.append("n_ph=2 production compatibility evidence is not ready")

    return {
        "schema_version": PRODUCTION_GATE_SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "read_only": True,
        "mutates_chtc_inputs": False,
        "mutates_generated_artifacts": False,
        "required_target_excited_roots": required_target_excited_roots,
        "ok": production_ready,
        "first_pass_ready": first_pass_ready,
        "production_ready": production_ready,
        "n_ph1_first_pass_status": nph1_status.get("first_pass_status"),
        "n_ph2_production_readiness": nph2_status.get("production_status"),
        "target_excited_root_count": {
            "required": required_target_excited_roots,
            "n_ph1": nph1_status.get("target_excited_roots"),
            "n_ph2": nph2_status.get("target_excited_roots"),
        },
        "missing_comparator_blockers": nph2_status.get("missing_comparator_blockers", {}),
        "exact_reference_boundary_status": exact_boundary,
        "consumer_fail_closed": consumer_fail_closed,
        "errors": errors,
        "warnings": list(nph1_status.get("warnings", [])) + list(nph2_status.get("warnings", [])),
        "compatibility_tiers": {
            "n_ph_1": nph1_status,
            "n_ph_2": nph2_status,
        },
    }


def require_production_ready(report: Mapping[str, Any], *, consumer_id: str = "qse_production_mode") -> Mapping[str, Any]:
    """Raise ``ProductionGateError`` unless a gate report is production-ready."""

    if report.get("production_ready") is True and report.get("ok") is True:
        return report
    status = _clean(report.get("n_ph2_production_readiness")) or "unknown"
    blockers = report.get("missing_comparator_blockers") if isinstance(report.get("missing_comparator_blockers"), Mapping) else {}
    codes = blockers.get("codes") if isinstance(blockers, Mapping) else None
    code_text = f"; missing comparator blockers={codes}" if codes else ""
    raise ProductionGateError(
        f"{consumer_id} fail-closed: Paper III n_ph=2 production gate status is {status!r}{code_text}"
    )


def validate_and_require_production_ready(
    *,
    nph1_dir: str | Path = DEFAULT_NPH1_DIR,
    nph2_dir: str | Path = DEFAULT_NPH2_DIR,
    required_target_excited_roots: int = REQUIRED_TARGET_EXCITED_ROOTS,
    consumer_id: str = "qse_production_mode",
) -> dict[str, Any]:
    """Validate matrix artifacts and raise if production evidence is incomplete."""

    report = validate_production_gate(
        nph1_dir=nph1_dir,
        nph2_dir=nph2_dir,
        required_target_excited_roots=required_target_excited_roots,
    )
    require_production_ready(report, consumer_id=consumer_id)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Paper III compatibility/production gate.")
    parser.add_argument("--nph1-dir", type=Path, default=DEFAULT_NPH1_DIR)
    parser.add_argument("--nph2-dir", type=Path, default=DEFAULT_NPH2_DIR)
    parser.add_argument("--target-excited-roots", type=int, default=REQUIRED_TARGET_EXCITED_ROOTS)
    parser.add_argument(
        "--require-production-ready",
        action="store_true",
        help="Exit nonzero if the n_ph=2 production gate is blocked.",
    )
    parser.add_argument("--consumer-id", default="qse_production_mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = validate_production_gate(
        nph1_dir=args.nph1_dir,
        nph2_dir=args.nph2_dir,
        required_target_excited_roots=args.target_excited_roots,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_production_ready:
        try:
            require_production_ready(report, consumer_id=args.consumer_id)
        except ProductionGateError:
            return 1
    return 0


__all__ = [
    "CONSUMER_IDS",
    "DEFAULT_NPH1_DIR",
    "DEFAULT_NPH2_DIR",
    "PRODUCTION_GATE_SCHEMA_VERSION",
    "ProductionGateError",
    "REQUIRED_TARGET_EXCITED_ROOTS",
    "require_production_ready",
    "validate_and_require_production_ready",
    "validate_production_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
