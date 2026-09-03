"""Hydrate the final Paper-II selector lock from verified campaign evidence.

This is intentionally a strict one-way evidence adapter.  It never chooses a
cell or threshold: the fifteen selections must already agree between the
human-reviewed lock and the promotion preview.  It verifies every stitched
screen/continuation source and records the evidence needed by the submission
gate.  A report-refresh resolution receipt is optional; without one the
result remains explicitly blocked from submission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.campaigns.paper_ii_final_selector_comparison_v1 import (
    LOCK_JSON,
)
from pipelines.time_dynamics.run_lock import assert_comparable


REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = (
    REPO_ROOT
    / "MATH/paper_facing/paper_II_dynamics/evidence/"
    "paper_ii_final_selector_comparison_v1"
)
DEFAULT_PREVIEW = EVIDENCE_ROOT / "promotion_preview_20260902.json"
DEFAULT_LEDGER = EVIDENCE_ROOT / "ledger_371_rows_20260902.json"
DEFAULT_REPORT = EVIDENCE_ROOT / "paper_ii_staged_report_20260902.json"
DEFAULT_FINAL_RECEIPT = (
    REPO_ROOT
    / "chtc/paper_ii_full_factorial_coverage_wave2_b1354438_20260902/"
    "final_ingestion_receipt_9718010_20260902.json"
)
EXPECTED_PREVIEW_SHA256 = (
    "40eb31245d369ed2fa335860f832d578e69bc9151c73c91b9457321d4b474fdf"
)
EXPECTED_LEDGER_SHA256 = (
    "830370e43b26a818f92d36d571d1aae4cb7a5475b8df61652dba9e86a5b1b9e6"
)
EXPECTED_REPORT_SHA256 = (
    "2006f8834020134ad8b9329fdfa5d6242026c38f8b2976a65f8a7a559ee1f791"
)
EXPECTED_FINAL_RECEIPT_SHA256 = (
    "5cbfa8c965e63bc52eecdd742e5a2cddab7cdd871d1faf358d65e7db3aa8e615"
)
EXPECTED_RESULTS_PDF_SHA256 = (
    "f5e1666a567a7abe3acea4609ce57a3930038f01749cef02ff411dc65151a2e6"
)
EXPECTED_RECORD_COUNT = 371
EXPECTED_PAIR_COUNT = 15


class HydrationError(ValueError):
    """The supplied evidence does not reproduce the locked cell selection."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HydrationError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HydrationError(message)


def _verify(path: Path, expected: str, *, label: str) -> None:
    _require(path.is_file(), f"{label} is absent: {path}")
    observed = _sha256(path)
    _require(observed == str(expected), f"{label} sha256 drift: {observed}")


def _resolve_source(path: str, source_root: Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else source_root / candidate


def _times(payload: Mapping[str, Any]) -> tuple[float, ...]:
    trajectory = payload.get("trajectory")
    if isinstance(trajectory, Mapping):
        rows = trajectory.get("points") or ()
    elif isinstance(trajectory, list):
        rows = trajectory
    else:
        rows = ()
    return tuple(float(row["time"]) for row in rows if isinstance(row, Mapping))


def _run_lock_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    lock = dict(payload.get("run_lock") or {})
    required = ("schema", "code_revision", "physics_fingerprint", "comparison_fingerprint")
    missing = [key for key in required if not lock.get(key)]
    _require(not missing, f"run payload lacks lock fields: {missing}")
    return {key: lock[key] for key in required}


def _source_receipt(
    row: Mapping[str, Any],
    *,
    source_root: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    screen_path = _resolve_source(str(row["screen_run_path"]), source_root)
    continuation_path = _resolve_source(str(row["run_path"]), source_root)
    cost_path = _resolve_source(str(row["cost_path"]), source_root)
    _verify(screen_path, str(row["screen_run_sha256"]), label=f"{label} screen")
    _verify(
        continuation_path,
        str(row["run_sha256"]),
        label=f"{label} continuation",
    )
    _verify(cost_path, str(row["cost_sha256"]), label=f"{label} terminal cost")

    screen = _read_json(screen_path)
    continuation = _read_json(continuation_path)
    screen_lock_full = dict(screen.get("run_lock") or {})
    screen_times = _times(screen)
    continuation_times = _times(continuation)
    _require(screen_times and continuation_times, f"{label} trajectory is empty")
    _require(
        math.isclose(screen_times[0], 0.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(screen_times[-1], 5.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(
            continuation_times[0], 5.0, rel_tol=0.0, abs_tol=1.0e-12
        )
        and math.isclose(
            continuation_times[-1], 20.0, rel_tol=0.0, abs_tol=1.0e-12
        ),
        f"{label} is not a t=0..5 plus t=5..20 trajectory",
    )
    stitched = (*screen_times[:-1], *continuation_times)
    _require(
        len(stitched) == 501 and all(b > a for a, b in zip(stitched, stitched[1:])),
        f"{label} does not stitch to the locked 501-point t=0..20 grid",
    )
    sources = {
        "screen_run_json": {
            "path": str(row["screen_run_path"]),
            "sha256": str(row["screen_run_sha256"]),
        },
        "continuation_run_json": {
            "path": str(row["run_path"]),
            "sha256": str(row["run_sha256"]),
        },
        "terminal_cost_json": {
            "path": str(row["cost_path"]),
            "sha256": str(row["cost_sha256"]),
        },
        "stitched_point_count": 501,
        "stitched_time_span": [0.0, 20.0],
    }
    locks = {
        "screen": _run_lock_receipt(screen),
        "continuation": _run_lock_receipt(continuation),
    }
    excluded = {
        "run_path",
        "run_sha256",
        "screen_run_path",
        "screen_run_sha256",
        "cost_path",
        "cost_sha256",
    }
    metrics = {key: copy.deepcopy(value) for key, value in row.items() if key not in excluded}
    return sources, locks, metrics, screen_lock_full


def hydrate(
    lock_payload: Mapping[str, Any],
    *,
    preview_path: Path,
    ledger_path: Path,
    report_path: Path,
    final_receipt_path: Path,
    source_root: Path,
    resolution_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Return a fully source-bound copy of the fifteen-cell lock."""

    _verify(preview_path, EXPECTED_PREVIEW_SHA256, label="promotion preview")
    _verify(ledger_path, EXPECTED_LEDGER_SHA256, label="campaign ledger")
    _verify(report_path, EXPECTED_REPORT_SHA256, label="campaign report JSON")
    _verify(
        final_receipt_path,
        EXPECTED_FINAL_RECEIPT_SHA256,
        label="final ingestion receipt",
    )
    preview = _read_json(preview_path)
    ledger = _read_json(ledger_path)
    _require(preview.get("schema") == "paper_ii_promotion_preview_v4", "preview schema drift")
    _require(int(preview.get("pair_count", -1)) == EXPECTED_PAIR_COUNT, "preview pair count drift")
    _require(
        str((preview.get("ledger") or {}).get("sha256")) == EXPECTED_LEDGER_SHA256,
        "preview does not bind the preserved ledger",
    )
    records = ledger.get("records")
    _require(isinstance(records, list), "ledger records are absent")
    _require(len(records) == EXPECTED_RECORD_COUNT, "ledger record count drift")

    pairs: dict[tuple[str, str], Mapping[str, Any]] = {}
    for pair in preview.get("pairs") or ():
        key = (str(pair["drive"]), str(pair["controller"]))
        _require(key not in pairs, f"duplicate preview pair: {key}")
        pairs[key] = pair
    _require(len(pairs) == EXPECTED_PAIR_COUNT, "preview pair mapping is incomplete")

    result = copy.deepcopy(dict(lock_payload))
    for cell in result.get("cells") or ():
        cell_id = str(cell["cell_id"])
        key = (str(cell["drive"]), str(cell["controller"]))
        _require(key in pairs, f"{cell_id}: no preview pair")
        pair = pairs[key]
        _require(pair.get("regime") == "hh_snake_nph1", f"{cell_id}: regime drift")
        _require(pair.get("horizon") == "t20", f"{cell_id}: horizon drift")
        apm_row = dict(pair["rows"]["exchange"])
        avqds_row = dict(pair["rows"]["avqds"])
        _require(
            float(apm_row["threshold"]) == float(cell["apm_cut"])
            and float(avqds_row["threshold"]) == float(cell["avqds_cut"]),
            f"{cell_id}: preview thresholds disagree with the lock",
        )
        basis = str(cell["adoption_status"])
        _require(
            apm_row.get("qualification_basis") == basis
            and avqds_row.get("qualification_basis") == basis,
            f"{cell_id}: qualification basis drift",
        )
        _require(
            apm_row.get("status") == "accuracy_qualified"
            and avqds_row.get("status") == "accuracy_qualified",
            f"{cell_id}: selected source is not accuracy-qualified",
        )
        apm_sources, apm_locks, apm_metrics, apm_screen_lock = _source_receipt(
            apm_row, source_root=source_root, label=f"{cell_id} APM"
        )
        avqds_sources, avqds_locks, avqds_metrics, avqds_screen_lock = _source_receipt(
            avqds_row, source_root=source_root, label=f"{cell_id} AVQDS"
        )
        comparison = assert_comparable((apm_screen_lock, avqds_screen_lock))
        _require(
            comparison == str(pair["source_comparison_fingerprint"]),
            f"{cell_id}: source comparison fingerprint drift",
        )
        exact = dict(pair["exact_reference"])
        exact_path = _resolve_source(str(exact["path"]), source_root)
        _verify(exact_path, str(exact["sha256"]), label=f"{cell_id} exact reference")
        cell["tuple_id"] = str(pair["tuple_id"])
        cell["source_artifacts"] = {
            "apm": apm_sources,
            "avqds": avqds_sources,
            "exact_reference": exact,
        }
        cell["run_lock"] = {
            "source_comparison_fingerprint": comparison,
            "apm": apm_locks,
            "avqds": avqds_locks,
        }
        cell["metrics"] = {"apm": apm_metrics, "avqds": avqds_metrics}
        cell["paper_status"] = {
            "apm": "qualified",
            "avqds": "qualified",
            "qualification_basis": basis,
        }

    external = dict(result.get("external_artifacts") or {})
    external.update(
        promotion_preview={
            "path": str(preview_path.relative_to(REPO_ROOT)),
            "sha256": EXPECTED_PREVIEW_SHA256,
            "resolution_status": "RESOLVED",
        },
        ledger={
            "path": str(ledger_path.relative_to(REPO_ROOT)),
            "sha256": EXPECTED_LEDGER_SHA256,
            "record_count": EXPECTED_RECORD_COUNT,
            "resolution_status": "RESOLVED",
        },
        final_ingestion_receipt={
            "path": str(final_receipt_path.relative_to(REPO_ROOT)),
            "sha256": EXPECTED_FINAL_RECEIPT_SHA256,
            "resolution_status": "RESOLVED",
        },
        report_json={
            "path": str(report_path.relative_to(REPO_ROOT)),
            "sha256": EXPECTED_REPORT_SHA256,
            "resolution_status": "RESOLVED",
        },
    )
    results_pdf = dict(external.get("results_pdf") or {})
    _require(
        results_pdf.get("sha256") == EXPECTED_RESULTS_PDF_SHA256,
        "stable results PDF digest drift",
    )
    results_pdf["currency_claim"] = "current_through_coverage_cluster_9718010"
    external["results_pdf"] = results_pdf

    if resolution_receipt_path is None:
        external["report_refresh_resolution"] = {
            "path": None,
            "sha256": None,
            "resolution_status": "PENDING_CALIBRATION",
        }
        active = ["further_campaign_submission", "claiming_the_visible_pdf_is_current"]
    else:
        _require(
            resolution_receipt_path.is_relative_to(REPO_ROOT),
            "resolution receipt must be tracked inside the repository",
        )
        resolution_sha = _sha256(resolution_receipt_path)
        external["report_refresh_resolution"] = {
            "path": str(resolution_receipt_path.relative_to(REPO_ROOT)),
            "sha256": resolution_sha,
            "resolution_status": "RESOLVED",
        }
        active = []
    result["external_artifacts"] = external
    result["blocks"] = {
        "schema": "paper_ii_report_refresh_required_v1",
        "supersedes": "chtc/paper_ii_staged_accuracy_euler_v1_t20_resume_repair_d3301581_20260826/report_refresh_required.json",
        "superseded_sha256": "84c67ea56fd6267b1fb772178f96d0f5025556ad3b64412bb65edc673fc29a11",
        "active": active,
    }
    result["adoption_note"] = (
        "The fifteen cells, cuts, statuses, stitched source trajectories, run-lock "
        "fingerprints, exact references, and metrics are bound to the verified "
        "371-row post-coverage ledger and promotion preview."
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-json", default=str(LOCK_JSON))
    parser.add_argument("--preview-json", default=str(DEFAULT_PREVIEW))
    parser.add_argument("--ledger-json", default=str(DEFAULT_LEDGER))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT))
    parser.add_argument("--final-receipt-json", default=str(DEFAULT_FINAL_RECEIPT))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--resolution-receipt-json")
    parser.add_argument("--output-json", default=str(LOCK_JSON))
    args = parser.parse_args(argv)
    payload = hydrate(
        _read_json(Path(args.lock_json)),
        preview_path=Path(args.preview_json).resolve(),
        ledger_path=Path(args.ledger_json).resolve(),
        report_path=Path(args.report_json).resolve(),
        final_receipt_path=Path(args.final_receipt_json).resolve(),
        source_root=Path(args.source_root).resolve(),
        resolution_receipt_path=(
            None
            if args.resolution_receipt_json is None
            else Path(args.resolution_receipt_json).resolve()
        ),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "cell_count": len(payload["cells"]),
                "active_blocks": payload["blocks"]["active"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
