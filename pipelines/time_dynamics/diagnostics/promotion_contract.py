"""Verify that Paper-II table rows already exist as results-PDF tuples.

The campaign results PDF is the author-review boundary.  Manuscript generation
may consume a t=20 comparison only after the report manifest proves that the
same exact/APM/AVQDS tuple is present in that PDF.  This module owns the digest
and cell-identity checks so report and manuscript writers cannot implement
different notions of "already reviewed".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROMOTION_PREVIEW_SCHEMA = "paper_ii_promotion_preview_v4"
REPORT_SCHEMA = "paper_ii_staged_campaign_report_v1"
EXACT_REFERENCE_SCHEMA = "paper_ii_exact_reference_store_v1"
APM_METHOD = "exchange"
AVQDS_METHOD = "avqds"


class PromotionCoverageError(ValueError):
    """The manuscript candidate is not covered by the current results PDF."""


@dataclass(frozen=True)
class ResultsPdfTuple:
    """One planned t=20 exact/APM/AVQDS comparison cell."""

    tuple_id: str
    regime: str
    drive: str
    controller: str
    horizon: str
    exact_reference: Mapping[str, str]
    methods: Mapping[str, Mapping[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PromotionCoverageError(f"required promotion artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromotionCoverageError(f"invalid promotion JSON: {path}") from exc


def _resolve(path: str | Path, *, workspace_root: Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = workspace_root / target
    return target.resolve()


def _verify_artifact(
    artifact: Mapping[str, Any],
    *,
    workspace_root: Path,
    label: str,
) -> Path:
    try:
        target = _resolve(str(artifact["path"]), workspace_root=workspace_root)
        expected = str(artifact["sha256"])
    except KeyError as exc:
        raise PromotionCoverageError(f"{label} lacks path/SHA-256") from exc
    if not target.is_file() or target.stat().st_size <= 0:
        raise PromotionCoverageError(f"{label} is missing or empty: {target}")
    observed = _sha256(target)
    if observed != expected:
        raise PromotionCoverageError(
            f"{label} digest mismatch: expected {expected}, observed {observed}"
        )
    return target


def tuple_identity_payload(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical scientific identity covered by a preview page."""

    rows = dict(pair.get("rows") or {})
    try:
        apm = dict(rows[APM_METHOD])
        avqds = dict(rows[AVQDS_METHOD])
        exact = dict(pair["exact_reference"])
        return {
            "regime": str(pair["regime"]),
            "drive": str(pair["drive"]),
            "controller": str(pair["controller"]),
            "horizon": str(pair["horizon"]),
            "source_comparison_fingerprint": str(
                pair["source_comparison_fingerprint"]
            ),
            "exact_reference_sha256": str(exact["sha256"]),
            "apm": {
                "run_sha256": str(apm["run_sha256"]),
                "screen_run_sha256": str(apm["screen_run_sha256"]),
                "cost_sha256": str(apm["cost_sha256"]),
            },
            "avqds": {
                "run_sha256": str(avqds["run_sha256"]),
                "screen_run_sha256": str(avqds["screen_run_sha256"]),
                "cost_sha256": str(avqds["cost_sha256"]),
            },
        }
    except KeyError as exc:
        raise PromotionCoverageError(
            "promotion tuple lacks exact, APM, AVQDS, controller, or provenance data"
        ) from exc


def tuple_id(pair: Mapping[str, Any]) -> str:
    payload = tuple_identity_payload(pair)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def planned_tuple_identity_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable identity of a planned controller-specific tuple."""

    try:
        exact = dict(item["exact_reference"])
        return {
            "regime": str(item["regime"]),
            "drive": str(item["drive"]),
            "controller": str(item["controller"]),
            "horizon": str(item["horizon"]),
            "exact_reference_sha256": str(exact["sha256"]),
            "methods": [APM_METHOD, AVQDS_METHOD],
        }
    except KeyError as exc:
        raise PromotionCoverageError(
            "planned tuple lacks regime, drive, controller, horizon, or exact reference"
        ) from exc


def planned_tuple_id(item: Mapping[str, Any]) -> str:
    payload = planned_tuple_identity_payload(item)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_exact_reference(
    artifact: Mapping[str, Any], *, workspace_root: Path
) -> None:
    manifest_path = _verify_artifact(
        artifact, workspace_root=workspace_root, label="exact-reference manifest"
    )
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != EXACT_REFERENCE_SCHEMA:
        raise PromotionCoverageError(
            f"unexpected exact-reference schema in {manifest_path}"
        )
    if not str(manifest.get("seed_sha256") or ""):
        raise PromotionCoverageError("exact reference does not bind a seed digest")
    times = list(manifest.get("times") or ())
    if len(times) != 501 or abs(float(times[0])) > 1.0e-12:
        raise PromotionCoverageError("exact reference is not the 501-point t=0..20 grid")
    if abs(float(times[-1]) - 20.0) > 1.0e-12:
        raise PromotionCoverageError("exact reference does not reach t=20")


def _verify_row(
    row: Mapping[str, Any],
    *,
    method: str,
    workspace_root: Path,
) -> None:
    if row.get("method") != method:
        raise PromotionCoverageError(
            f"tuple method mismatch: expected {method}, observed {row.get('method')}"
        )
    for prefix in ("run", "screen_run", "cost"):
        _verify_artifact(
            {"path": row.get(f"{prefix}_path"), "sha256": row.get(f"{prefix}_sha256")},
            workspace_root=workspace_root,
            label=f"{method} {prefix}",
        )
    if not isinstance(row.get("graph"), list) or len(row["graph"]) != 3:
        raise PromotionCoverageError(f"{method} row lacks the terminal graph tuple")
    for field in (
        "mean_infidelity",
        "mean_abs_energy_error",
        "mean_abs_doublon_error",
        "mean_abs_charge_imbalance_error",
        "time_step_count",
        "n_theta",
    ):
        if row.get(field) is None:
            raise PromotionCoverageError(f"{method} row lacks {field}")


def verified_results_pdf_tuples(
    *,
    preview_receipt_path: Path,
    report_manifest_path: Path,
    workspace_root: Path = Path("."),
) -> tuple[ResultsPdfTuple, ...]:
    """Return manuscript-eligible tuple data after fail-closed coverage checks.

    Eligibility here means "present in the current results PDF," not scientific
    accuracy qualification.  Gate status remains an independent evidence role.
    """

    workspace_root = workspace_root.resolve()
    preview_receipt_path = _resolve(
        preview_receipt_path, workspace_root=workspace_root
    )
    report_manifest_path = _resolve(
        report_manifest_path, workspace_root=workspace_root
    )
    preview = _read_json(preview_receipt_path)
    report = _read_json(report_manifest_path)
    if preview.get("schema") != PROMOTION_PREVIEW_SCHEMA:
        raise PromotionCoverageError("results-PDF preview receipt is not schema v3")
    if report.get("schema") != REPORT_SCHEMA:
        raise PromotionCoverageError("unexpected campaign-report manifest schema")

    preview_link = dict(report.get("promotion_preview") or {})
    linked_preview = _verify_artifact(
        {
            "path": preview_link.get("receipt"),
            "sha256": preview_link.get("receipt_sha256"),
        },
        workspace_root=workspace_root,
        label="report-linked promotion-preview receipt",
    )
    if linked_preview != preview_receipt_path:
        raise PromotionCoverageError("report manifest links a different preview receipt")
    _verify_artifact(
        {"path": report.get("pdf"), "sha256": report.get("pdf_sha256")},
        workspace_root=workspace_root,
        label="campaign results PDF",
    )
    _verify_artifact(
        preview.get("tex_fragment") or {},
        workspace_root=workspace_root,
        label="promotion-preview TeX fragment",
    )

    preview_ledger = dict(preview.get("ledger") or {})
    ledger_path = _verify_artifact(
        preview_ledger, workspace_root=workspace_root, label="preview source ledger"
    )
    report_ledger = _resolve(report.get("source_ledger"), workspace_root=workspace_root)
    if report_ledger != ledger_path:
        raise PromotionCoverageError("report and preview use different ledgers")
    if report.get("source_ledger_sha256") != preview_ledger.get("sha256"):
        raise PromotionCoverageError("report and preview ledger digests differ")
    if int(report.get("ledger_record_count", -1)) != int(
        preview_ledger.get("record_count", -2)
    ):
        raise PromotionCoverageError("report and preview ledger counts differ")

    pairs = list(preview.get("pairs") or ())
    if int(preview.get("pair_count", -1)) != len(pairs):
        raise PromotionCoverageError("preview pair count is inconsistent")
    if int(preview_link.get("pair_count", -1)) != len(pairs):
        raise PromotionCoverageError("report omits one or more preview tuples")

    complete_pair_identities: set[tuple[str, str, str, str]] = set()
    for pair in pairs:
        identity = (
            str(pair.get("regime") or ""),
            str(pair.get("drive") or ""),
            str(pair.get("controller") or ""),
            str(pair.get("horizon") or ""),
        )
        if not all(identity) or identity[3] != "t20":
            raise PromotionCoverageError(
                "every results-PDF tuple must name regime, drive, controller, and t20"
            )
        if identity in complete_pair_identities:
            raise PromotionCoverageError(f"duplicate results-PDF tuple: {identity}")
        complete_pair_identities.add(identity)
        rows = dict(pair.get("rows") or {})
        if set(rows) != {APM_METHOD, AVQDS_METHOD}:
            raise PromotionCoverageError(
                f"results-PDF tuple {identity} is not an exact APM/AVQDS pair"
            )
        _verify_row(rows[APM_METHOD], method=APM_METHOD, workspace_root=workspace_root)
        _verify_row(
            rows[AVQDS_METHOD], method=AVQDS_METHOD, workspace_root=workspace_root
        )
        exact = dict(pair.get("exact_reference") or {})
        _verify_exact_reference(exact, workspace_root=workspace_root)
        _verify_artifact(
            pair.get("figure") or {},
            workspace_root=workspace_root,
            label=f"results-PDF tuple figure {identity}",
        )
        for source in pair.get("figure_sources") or ():
            _verify_artifact(
                source,
                workspace_root=workspace_root,
                label=f"results-PDF tuple source {identity}",
            )
        observed_tuple_id = str(pair.get("tuple_id") or "")
        expected_tuple_id = tuple_id(pair)
        if observed_tuple_id != expected_tuple_id:
            raise PromotionCoverageError(
                f"results-PDF tuple identity digest mismatch for {identity}"
            )
    planned = list(preview.get("planned_tuples") or ())
    if int(preview.get("planned_tuple_count", -1)) != len(planned):
        raise PromotionCoverageError("planned tuple count is inconsistent")
    if int(preview_link.get("planned_tuple_count", -1)) != len(planned):
        raise PromotionCoverageError("report omits one or more planned tuples")

    result: list[ResultsPdfTuple] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in planned:
        identity = (
            str(item.get("regime") or ""),
            str(item.get("drive") or ""),
            str(item.get("controller") or ""),
            str(item.get("horizon") or ""),
        )
        if not all(identity) or identity[3] != "t20":
            raise PromotionCoverageError(
                "every planned tuple must name regime, drive, controller, and t20"
            )
        if identity in seen:
            raise PromotionCoverageError(f"duplicate planned results tuple: {identity}")
        seen.add(identity)
        exact = dict(item.get("exact_reference") or {})
        _verify_exact_reference(exact, workspace_root=workspace_root)
        methods = dict(item.get("methods") or {})
        if set(methods) != {APM_METHOD, AVQDS_METHOD}:
            raise PromotionCoverageError(
                f"planned tuple {identity} lacks APM or AVQDS slot"
            )
        for method in (APM_METHOD, AVQDS_METHOD):
            slot = dict(methods[method])
            paper_status = str(slot.get("paper_status") or "")
            if paper_status not in {
                "qualified",
                "candidate",
                "pending",
                "unresolved",
            }:
                raise PromotionCoverageError(
                    f"planned tuple {identity} has invalid {method} paper status"
                )
            latest_row = slot.get("latest_row")
            if latest_row is not None:
                _verify_row(
                    dict(latest_row), method=method, workspace_root=workspace_root
                )
            if paper_status == "qualified" and (
                latest_row is None
                or dict(latest_row).get("status") != "accuracy_qualified"
            ):
                raise PromotionCoverageError(
                    f"planned tuple {identity} exposes an unqualified {method} value"
                )
            if paper_status == "candidate" and (
                latest_row is None
                or dict(latest_row).get("status")
                not in {
                    "requires_threshold_extension",
                    "requires_threshold_coarsening",
                }
            ):
                raise PromotionCoverageError(
                    f"planned tuple {identity} lacks completed candidate {method} evidence"
                )
            if paper_status == "unresolved" and latest_row is not None and (
                dict(latest_row).get("status")
                not in {
                    "requires_threshold_extension",
                    "requires_threshold_coarsening",
                }
            ):
                raise PromotionCoverageError(
                    f"planned tuple {identity} has invalid unresolved {method} evidence"
                )
            if (
                paper_status == "pending"
                and latest_row is not None
                and dict(latest_row).get("status") != "safeguard_tripped"
            ):
                raise PromotionCoverageError(
                    f"planned tuple {identity} hides completed {method} evidence as pending"
                )
        observed_tuple_id = str(item.get("tuple_id") or "")
        expected_tuple_id = planned_tuple_id(item)
        if observed_tuple_id != expected_tuple_id:
            raise PromotionCoverageError(
                f"planned results tuple identity digest mismatch for {identity}"
            )
        result.append(
            ResultsPdfTuple(
                tuple_id=observed_tuple_id,
                regime=identity[0],
                drive=identity[1],
                controller=identity[2],
                horizon=identity[3],
                exact_reference=exact,
                methods=methods,
            )
        )
    if not complete_pair_identities.issubset(seen):
        raise PromotionCoverageError(
            "a complete preview pair is absent from the planned tuple matrix"
        )
    return tuple(result)


def manuscript_rows_from_results_pdf(
    tuples: Sequence[ResultsPdfTuple],
) -> list[dict[str, Any]]:
    """Project reviewed tuples into the compact manuscript-table row shape."""

    rows: list[dict[str, Any]] = []
    for item in tuples:
        for method, public_method in ((APM_METHOD, "apm"), (AVQDS_METHOD, "avqds")):
            slot = dict(item.methods[method])
            source = dict(slot.get("latest_row") or {})
            rows.append(
                {
                    **source,
                    "regime": item.regime,
                    "drive": item.drive,
                    "controller": item.controller,
                    "horizon": item.horizon,
                    "method": public_method,
                    "results_pdf_tuple_id": item.tuple_id,
                    "exact_reference": dict(item.exact_reference),
                    "paper_status": slot["paper_status"],
                    "pending_reason": slot.get("pending_reason"),
                    "state_reason": slot.get("state_reason"),
                }
            )
    return rows


__all__ = [
    "PROMOTION_PREVIEW_SCHEMA",
    "PromotionCoverageError",
    "ResultsPdfTuple",
    "manuscript_rows_from_results_pdf",
    "planned_tuple_id",
    "planned_tuple_identity_payload",
    "tuple_id",
    "tuple_identity_payload",
    "verified_results_pdf_tuples",
]
