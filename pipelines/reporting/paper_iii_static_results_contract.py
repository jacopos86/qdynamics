#!/usr/bin/env python3
"""Verified Paper III static-QSE results contract.

Projects the immutable attempt ledger onto the full declared campaign matrix:
every planned cell appears exactly once with honest per-stage statuses
(``pending`` / ``gate_pending`` / ``candidate`` / ``qualified``) and every
intentionally unscheduled slot appears as ``not_planned``. A completed action
is visible without waiting for siblings; a completed numerical result that
failed a gate stays visible as ``candidate`` with its payload references
intact; only absent work is ``pending``.

``manuscript_rows_from_static_results_contract`` is the **sole verified
manuscript-row accessor**: manuscript assets consume these typed rows and
never rediscover raw runs. This module does not edit the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pipelines.qse_spectra.static_campaign import (
    STAGE_ORDER,
    STAGE_QUALIFICATION,
    STATUS_CANDIDATE,
    STATUS_GATE_PENDING,
    STATUS_NOT_PLANNED,
    STATUS_PENDING,
    STATUS_QUALIFIED,
    StaticCampaignError,
    StaticQSECampaign,
    StaticQSELedger,
    canonical_digest,
)

STATIC_RESULTS_CONTRACT_SCHEMA_VERSION = "paper_iii_static_qse_results_contract_v1"

_CELL_STATUSES = (
    STATUS_QUALIFIED,
    STATUS_CANDIDATE,
    STATUS_PENDING,
    STATUS_GATE_PENDING,
)
_STAGE_STATUSES = _CELL_STATUSES


@dataclass(frozen=True)
class StaticResultsContract:
    """The projected matrix plus the digests that pin it to its inputs."""

    payload: dict[str, Any]

    @property
    def campaign_digest(self) -> str:
        return str(self.payload["campaign_digest"])

    @property
    def ledger_digest(self) -> str:
        return str(self.payload["ledger_digest"])

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.payload["rows"])

    def digest(self) -> str:
        return canonical_digest(self.payload)


@dataclass(frozen=True)
class VerifiedManuscriptRow:
    """One typed matrix row, obtainable only through the verified accessor."""

    cell_digest: str
    model_family: str
    case_id: str
    cutoff_kind: str
    cutoff_value: int
    method_id: str
    cell_status: str
    stage_statuses: tuple[tuple[str, str], ...]
    failed_gates: tuple[str, ...]
    payload_paths: tuple[str, ...]
    not_planned_reason: str | None


def _cell_rollup(stage_statuses: Mapping[str, str]) -> str:
    qualification = stage_statuses.get(STAGE_QUALIFICATION, STATUS_PENDING)
    if qualification in (STATUS_QUALIFIED, STATUS_CANDIDATE):
        return qualification
    if all(status == STATUS_PENDING for status in stage_statuses.values()):
        return STATUS_PENDING
    return STATUS_GATE_PENDING


def build_static_results_contract(
    campaign: StaticQSECampaign,
    ledger: StaticQSELedger,
) -> StaticResultsContract:
    """Project the ledger onto the complete declared matrix."""

    rows: list[dict[str, Any]] = []
    for cell in sorted(campaign.cells, key=lambda c: c.digest()):
        actions = campaign.cell_actions(cell)
        stage_statuses = campaign.cell_stage_statuses(ledger, cell)
        stages: dict[str, Any] = {}
        failed_gates: list[str] = []
        payload_paths: list[str] = []
        for stage in STAGE_ORDER:
            action = actions[stage]
            attempt = ledger.completed(action.action_id)
            stage_record: dict[str, Any] = {
                "action_id": action.action_id,
                "status": stage_statuses[stage],
            }
            if attempt is not None:
                stage_record["output_digests"] = {
                    k: v for k, v in attempt.output_digests
                }
                stage_record["payload_paths"] = list(attempt.payload_paths)
                stage_record["source_revision"] = attempt.source_revision
                payload_paths.extend(attempt.payload_paths)
                if attempt.qualification is not None:
                    qualification = {k: v for k, v in attempt.qualification}
                    stage_record["qualification"] = qualification
                    gates = qualification.get("failed_gates")
                    if isinstance(gates, (list, tuple)):
                        failed_gates.extend(str(g) for g in gates)
            stages[stage] = stage_record
        rows.append(
            {
                "row_kind": "planned_cell",
                "cell_digest": cell.digest(),
                "cell": cell.payload(),
                "model_family": str(cell.physics.model_family),
                "case_id": str(cell.physics.case_id),
                "cutoff_kind": str(cell.physics.cutoff_kind),
                "cutoff_value": int(cell.physics.cutoff_value),
                "method_id": str(cell.method_id),
                "cell_status": _cell_rollup(stage_statuses),
                "stage_statuses": dict(stage_statuses),
                "stages": stages,
                "failed_gates": sorted(set(failed_gates)),
                "payload_paths": payload_paths,
            }
        )
    for slot in campaign.not_planned:
        rows.append(
            {
                "row_kind": "not_planned_slot",
                "cell_digest": canonical_digest(slot.payload()),
                "model_family": str(slot.physics.model_family),
                "case_id": str(slot.physics.case_id),
                "cutoff_kind": str(slot.physics.cutoff_kind),
                "cutoff_value": int(slot.physics.cutoff_value),
                "method_id": str(slot.method_id),
                "cell_status": STATUS_NOT_PLANNED,
                "not_planned_reason": str(slot.reason),
            }
        )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["cell_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    payload = {
        "schema_version": STATIC_RESULTS_CONTRACT_SCHEMA_VERSION,
        "campaign_id": str(campaign.campaign_id),
        "campaign_digest": campaign.digest(),
        "ledger_digest": ledger.digest(),
        "ledger_attempt_count": len(ledger),
        "planned_cell_count": len(campaign.cells),
        "not_planned_count": len(campaign.not_planned),
        "status_counts": status_counts,
        "rows": rows,
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
    }
    return StaticResultsContract(payload=payload)


def validate_static_results_contract(contract: StaticResultsContract) -> None:
    """Schema and consistency validation; raises on any violation."""

    payload = contract.payload
    if payload.get("schema_version") != STATIC_RESULTS_CONTRACT_SCHEMA_VERSION:
        raise StaticCampaignError(
            f"results-contract schema mismatch: {payload.get('schema_version')!r}."
        )
    for key in ("campaign_digest", "ledger_digest", "rows"):
        if key not in payload:
            raise StaticCampaignError(f"results contract is missing {key!r}.")
    planned = [row for row in payload["rows"] if row.get("row_kind") == "planned_cell"]
    if len(planned) != int(payload.get("planned_cell_count", -1)):
        raise StaticCampaignError("planned cell count does not match its rows.")
    seen: set[str] = set()
    for row in payload["rows"]:
        digest = str(row.get("cell_digest") or "")
        if not digest:
            raise StaticCampaignError("contract row is missing its cell digest.")
        if digest in seen:
            raise StaticCampaignError(f"duplicate contract row for cell {digest}.")
        seen.add(digest)
        status = str(row.get("cell_status"))
        if row.get("row_kind") == "not_planned_slot":
            if status != STATUS_NOT_PLANNED:
                raise StaticCampaignError(
                    f"not_planned slot {digest} carries status {status!r}."
                )
            continue
        if status not in _CELL_STATUSES:
            raise StaticCampaignError(f"unknown cell status {status!r}.")
        stage_statuses = row.get("stage_statuses")
        if not isinstance(stage_statuses, Mapping) or set(stage_statuses) != set(
            STAGE_ORDER
        ):
            raise StaticCampaignError(
                f"cell {digest} does not report every declared stage."
            )
        for stage, stage_status in stage_statuses.items():
            if stage_status not in _STAGE_STATUSES:
                raise StaticCampaignError(
                    f"cell {digest} stage {stage} has unknown status {stage_status!r}."
                )
        # A failed gate never becomes pending: candidate rows must retain
        # their payload references.
        if status == STATUS_CANDIDATE and not row.get("payload_paths"):
            raise StaticCampaignError(
                f"candidate cell {digest} has no payload references."
            )


def manuscript_rows_from_static_results_contract(
    contract: StaticResultsContract,
) -> tuple[VerifiedManuscriptRow, ...]:
    """The sole verified accessor for manuscript-facing matrix rows."""

    validate_static_results_contract(contract)
    rows: list[VerifiedManuscriptRow] = []
    for row in contract.rows:
        rows.append(
            VerifiedManuscriptRow(
                cell_digest=str(row["cell_digest"]),
                model_family=str(row["model_family"]),
                case_id=str(row["case_id"]),
                cutoff_kind=str(row["cutoff_kind"]),
                cutoff_value=int(row["cutoff_value"]),
                method_id=str(row["method_id"]),
                cell_status=str(row["cell_status"]),
                stage_statuses=tuple(
                    sorted((str(k), str(v)) for k, v in (row.get("stage_statuses") or {}).items())
                ),
                failed_gates=tuple(str(g) for g in row.get("failed_gates", ())),
                payload_paths=tuple(str(p) for p in row.get("payload_paths", ())),
                not_planned_reason=(
                    str(row["not_planned_reason"])
                    if row.get("not_planned_reason") is not None
                    else None
                ),
            )
        )
    return tuple(rows)


def write_static_results_contract(contract: StaticResultsContract, path: Any) -> None:
    import json
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract.payload, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_static_results_contract(path: Any) -> StaticResultsContract:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    contract = StaticResultsContract(payload=dict(payload))
    validate_static_results_contract(contract)
    return contract


__all__ = [
    "STATIC_RESULTS_CONTRACT_SCHEMA_VERSION",
    "StaticResultsContract",
    "VerifiedManuscriptRow",
    "build_static_results_contract",
    "load_static_results_contract",
    "manuscript_rows_from_static_results_contract",
    "validate_static_results_contract",
    "write_static_results_contract",
]
