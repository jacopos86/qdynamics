"""Build the canonical combined ``t5`` + ``t20`` Paper-II campaign report.

This is deliberately separate from :mod:`staged_campaign_report`.  That
legacy writer owns a compact, single-horizon qualified-pair report; calling it
on the canonical output directory would erase the campaign overview, planned
slot matrix, pending evidence, and historical screening sections.

The combined writer has one narrow interface.  It consumes three already
materialized inputs -- the immutable ledger, the promotion-preview receipt,
and an explicit scheduler/frontier snapshot -- verifies their digest links,
renders both horizons plus the planned-slot preview, compiles in a private
staging directory, and atomically publishes the report artifacts.  It never
queries Condor, searches for receipts, or chooses a latest run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from pypdf import PdfReader

from pipelines.time_dynamics.diagnostics.promotion_contract import (
    APM_METHOD,
    AVQDS_METHOD,
    PROMOTION_PREVIEW_SCHEMA,
    REPORT_SCHEMA,
    planned_tuple_id,
    tuple_id,
)
from pipelines.time_dynamics.staged_campaign import (
    CampaignLedger,
    METHOD_APM,
    METHOD_AVQDS,
    RungRecord,
)


COMBINED_FRONTIER_SCHEMA = "paper_ii_combined_report_frontier_v1"
COMBINED_REPORT_BUILDER_SCHEMA = "paper_ii_combined_report_builder_v1"
COMBINED_HORIZON = "t5_and_t20"

_METHOD_LABELS = {METHOD_APM: "full APM", METHOD_AVQDS: "AVQDS"}
_PAPER_STATES = {"qualified", "candidate", "pending", "unresolved"}
_FRONTIER_STATES = {
    "idle",
    "running",
    "held",
    "transfer_pending",
    "gate_pending",
    "failed",
    "completed",
}
_CLUSTER_RE = re.compile(r"(?:^|/)cluster_(\d+)(?:[./]|$)")


class CombinedReportContractError(ValueError):
    """An input cannot support a canonical combined report refresh."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CombinedReportContractError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CombinedReportContractError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CombinedReportContractError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve(path: str | Path, *, workspace_root: Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = workspace_root / target
    return target.resolve()


def _relative(path: Path, *, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _verify_artifact(
    artifact: Mapping[str, Any],
    *,
    workspace_root: Path,
    label: str,
) -> Path:
    path_text = str(artifact.get("path") or "")
    expected = str(artifact.get("sha256") or "")
    if not path_text or len(expected) != 64:
        raise CombinedReportContractError(f"{label} lacks path/SHA-256")
    target = _resolve(path_text, workspace_root=workspace_root)
    if not target.is_file() or target.stat().st_size <= 0:
        raise CombinedReportContractError(f"{label} is missing or empty: {target}")
    observed = _sha256(target)
    if observed != expected:
        raise CombinedReportContractError(
            f"{label} digest mismatch: expected {expected}, observed {observed}"
        )
    return target


def _verify_preview(
    *,
    preview: Mapping[str, Any],
    preview_path: Path,
    ledger: CampaignLedger,
    ledger_path: Path,
    workspace_root: Path,
) -> tuple[Path, Counter[str]]:
    if preview.get("schema") != PROMOTION_PREVIEW_SCHEMA:
        raise CombinedReportContractError(
            "promotion preview does not use the canonical schema"
        )
    ledger_link = dict(preview.get("ledger") or {})
    ledger_sha = _sha256(ledger_path)
    if str(ledger_link.get("sha256") or "") != ledger_sha:
        raise CombinedReportContractError(
            "promotion preview is stale against the canonical ledger"
        )
    if int(ledger_link.get("record_count", -1)) != len(ledger.records):
        raise CombinedReportContractError(
            "promotion preview record count differs from the canonical ledger"
        )
    if dict(preview.get("qualification_policy") or {}) != dict(
        ledger.qualification_policy
    ):
        raise CombinedReportContractError(
            "promotion preview qualification policy differs from the ledger"
        )

    fragment = _verify_artifact(
        dict(preview.get("tex_fragment") or {}),
        workspace_root=workspace_root,
        label="promotion-preview TeX fragment",
    )
    pairs = list(preview.get("pairs") or ())
    if int(preview.get("pair_count", -1)) != len(pairs):
        raise CombinedReportContractError("promotion-preview pair count is inconsistent")
    pair_identities: set[tuple[str, str, str, str]] = set()
    for index, pair in enumerate(pairs):
        pair = dict(pair)
        identity = (
            str(pair.get("regime") or ""),
            str(pair.get("drive") or ""),
            str(pair.get("controller") or ""),
            str(pair.get("horizon") or ""),
        )
        if not all(identity) or identity[3] != "t20":
            raise CombinedReportContractError(
                f"promotion-preview pair {index} lacks a complete t20 identity"
            )
        if identity in pair_identities:
            raise CombinedReportContractError(
                f"duplicate promotion-preview pair identity: {identity}"
            )
        pair_identities.add(identity)
        if str(pair.get("tuple_id") or "") != tuple_id(pair):
            raise CombinedReportContractError(
                f"promotion-preview pair {identity} has a stale tuple ID"
            )
        rows = dict(pair.get("rows") or {})
        if set(rows) != {APM_METHOD, AVQDS_METHOD}:
            raise CombinedReportContractError(
                f"promotion-preview pair {identity} lacks APM or AVQDS"
            )
        for method in (APM_METHOD, AVQDS_METHOD):
            row = dict(rows[method])
            if str(row.get("method") or "") != method:
                raise CombinedReportContractError(
                    f"promotion-preview pair {identity} mislabels {method}"
                )
            for prefix in ("run", "screen_run", "cost"):
                _verify_artifact(
                    {
                        "path": row.get(f"{prefix}_path"),
                        "sha256": row.get(f"{prefix}_sha256"),
                    },
                    workspace_root=workspace_root,
                    label=f"promotion-preview pair {identity} {method} {prefix}",
                )
        _verify_artifact(
            dict(pair.get("exact_reference") or {}),
            workspace_root=workspace_root,
            label=f"promotion-preview pair {identity} exact reference",
        )
        figure = _verify_artifact(
            dict(pair.get("figure") or {}),
            workspace_root=workspace_root,
            label=f"promotion-preview figure {index}",
        )
        if figure.parent.name != "plots":
            raise CombinedReportContractError(
                "promotion-preview figures must remain in the canonical plots directory"
            )
        for source_index, artifact in enumerate(pair.get("figure_sources") or ()):
            _verify_artifact(
                dict(artifact),
                workspace_root=workspace_root,
                label=f"promotion-preview pair {identity} source {source_index}",
            )

    tuples = list(preview.get("planned_tuples") or ())
    if int(preview.get("planned_tuple_count", -1)) != len(tuples):
        raise CombinedReportContractError(
            "promotion-preview planned-tuple count is inconsistent"
        )
    slot_counts: Counter[str] = Counter()
    tuple_ids: set[str] = set()
    planned_identities: set[tuple[str, str, str, str]] = set()
    for index, item in enumerate(tuples):
        item = dict(item)
        observed_tuple_id = str(item.get("tuple_id") or "")
        if not observed_tuple_id or observed_tuple_id in tuple_ids:
            raise CombinedReportContractError(
                f"planned tuple {index} lacks a unique tuple ID"
            )
        tuple_ids.add(observed_tuple_id)
        identity = (
            str(item.get("regime") or ""),
            str(item.get("drive") or ""),
            str(item.get("controller") or ""),
            str(item.get("horizon") or ""),
        )
        if not all(identity) or identity[3] != "t20":
            raise CombinedReportContractError(
                f"planned tuple {observed_tuple_id} lacks a complete t20 identity"
            )
        if identity in planned_identities:
            raise CombinedReportContractError(
                f"duplicate planned tuple identity: {identity}"
            )
        planned_identities.add(identity)
        if observed_tuple_id != planned_tuple_id(item):
            raise CombinedReportContractError(
                f"planned tuple {identity} has a stale tuple ID"
            )
        _verify_artifact(
            dict(item.get("exact_reference") or {}),
            workspace_root=workspace_root,
            label=f"planned tuple {identity} exact reference",
        )
        methods = dict(item.get("methods") or {})
        if set(methods) != {APM_METHOD, AVQDS_METHOD}:
            raise CombinedReportContractError(
                f"planned tuple {observed_tuple_id} does not contain exactly APM and AVQDS"
            )
        for method in (APM_METHOD, AVQDS_METHOD):
            state = str(dict(methods[method]).get("paper_status") or "")
            if state not in _PAPER_STATES:
                raise CombinedReportContractError(
                    f"planned tuple {observed_tuple_id} has unsupported state {state!r}"
                )
            slot = dict(methods[method])
            latest = slot.get("latest_row")
            if latest is not None:
                latest = dict(latest)
                if str(latest.get("method") or "") != method:
                    raise CombinedReportContractError(
                        f"planned tuple {observed_tuple_id} mislabels its {method} row"
                    )
                for prefix in ("run", "screen_run", "cost"):
                    _verify_artifact(
                        {
                            "path": latest.get(f"{prefix}_path"),
                            "sha256": latest.get(f"{prefix}_sha256"),
                        },
                        workspace_root=workspace_root,
                        label=f"planned tuple {identity} {method} {prefix}",
                    )
            if state in {"qualified", "candidate"} and latest is None:
                raise CombinedReportContractError(
                    "planned tuple "
                    f"{observed_tuple_id} hides missing {method} evidence as {state}"
                )
            if state == "unresolved" and latest is not None and str(
                latest.get("status") or ""
            ) not in {
                "requires_threshold_extension",
                "requires_threshold_coarsening",
            }:
                raise CombinedReportContractError(
                    "planned tuple "
                    f"{observed_tuple_id} has invalid unresolved {method} evidence"
                )
            if state == "pending" and latest is not None and str(
                latest.get("status") or ""
            ) != "safeguard_tripped":
                raise CombinedReportContractError(
                    "planned tuple "
                    f"{observed_tuple_id} hides completed {method} evidence as pending"
                )
            slot_counts[state] += 1
    if sum(slot_counts.values()) != 2 * len(tuples):
        raise CombinedReportContractError("promotion-preview slot arithmetic failed")
    if not pair_identities.issubset(planned_identities):
        raise CombinedReportContractError(
            "a complete promotion-preview pair is absent from the planned matrix"
        )
    if not preview_path.is_file():
        raise CombinedReportContractError("promotion-preview receipt disappeared")
    return fragment, slot_counts


def _validate_frontier(
    *,
    frontier: Mapping[str, Any],
    frontier_path: Path,
    ledger_path: Path,
    preview_path: Path,
    ledger: CampaignLedger,
    preview: Mapping[str, Any],
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if frontier.get("schema") != COMBINED_FRONTIER_SCHEMA:
        raise CombinedReportContractError(
            "frontier metadata does not use the combined-report schema"
        )
    snapshot_id = str(frontier.get("snapshot_id") or "")
    snapshot_utc = str(frontier.get("snapshot_utc") or "")
    if not snapshot_id or not snapshot_utc:
        raise CombinedReportContractError(
            "frontier metadata requires snapshot_id and snapshot_utc"
        )
    try:
        parsed = datetime.fromisoformat(snapshot_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CombinedReportContractError("snapshot_utc is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise CombinedReportContractError("snapshot_utc must include a timezone")
    if not str(frontier.get("audit_status") or "").strip():
        raise CombinedReportContractError("frontier metadata lacks audit_status")
    if not str(frontier.get("scheduler_summary") or "").strip():
        raise CombinedReportContractError("frontier metadata lacks scheduler_summary")

    ledger_link = dict(frontier.get("ledger") or {})
    preview_link = dict(frontier.get("promotion_preview") or {})
    if str(ledger_link.get("sha256") or "") != _sha256(ledger_path):
        raise CombinedReportContractError("frontier metadata is stale against the ledger")
    if int(ledger_link.get("record_count", -1)) != len(ledger.records):
        raise CombinedReportContractError(
            "frontier metadata ledger record count is inconsistent"
        )
    if str(preview_link.get("sha256") or "") != _sha256(preview_path):
        raise CombinedReportContractError(
            "frontier metadata is stale against the promotion preview"
        )
    if int(preview_link.get("planned_tuple_count", -1)) != int(
        preview.get("planned_tuple_count", -2)
    ):
        raise CombinedReportContractError(
            "frontier metadata planned-tuple count is inconsistent"
        )

    scheduler_rows: list[dict[str, Any]] = []
    job_ids: set[str] = set()
    for index, raw in enumerate(frontier.get("scheduler_rows") or ()):
        row = dict(raw)
        required = (
            "job_id",
            "method",
            "controller",
            "drive",
            "state",
            "progress",
            "available_data",
        )
        missing = [field for field in required if not str(row.get(field) or "").strip()]
        if missing:
            raise CombinedReportContractError(
                f"scheduler row {index} lacks {', '.join(missing)}"
            )
        job_id = str(row["job_id"])
        if job_id in job_ids:
            raise CombinedReportContractError(f"duplicate scheduler job ID: {job_id}")
        job_ids.add(job_id)
        if str(row["state"]) not in _FRONTIER_STATES:
            raise CombinedReportContractError(
                f"scheduler row {job_id} has unsupported state {row['state']!r}"
            )
        threshold = row.get("threshold")
        if threshold is not None and float(threshold) <= 0.0:
            raise CombinedReportContractError(
                f"scheduler row {job_id} has a nonpositive threshold"
            )
        scheduler_rows.append(row)

    supplemental: list[dict[str, Any]] = []
    for index, raw in enumerate(frontier.get("supplemental_sections") or ()):
        item = dict(raw)
        role = str(item.get("role") or "")
        if role not in {
            "locked_t5_screening",
            "completed_delta",
            "gate_pending_diagnostics",
        }:
            raise CombinedReportContractError(
                f"supplemental section {index} has unsupported role {role!r}"
            )
        fragment = _verify_artifact(
            dict(item.get("tex_fragment") or {}),
            workspace_root=workspace_root,
            label=f"supplemental section {index}",
        )
        for figure_index, artifact in enumerate(item.get("figures") or ()):
            _verify_artifact(
                dict(artifact),
                workspace_root=workspace_root,
                label=f"supplemental section {index} figure {figure_index}",
            )
        summary = str(item.get("markdown_summary") or "").strip()
        if not summary:
            raise CombinedReportContractError(
                f"supplemental section {index} lacks markdown_summary"
            )
        supplemental.append({**item, "_fragment_path": fragment})

    if not frontier_path.is_file():
        raise CombinedReportContractError("frontier metadata disappeared")
    return scheduler_rows, supplemental


def _tex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(old, new)
    return text


def _method_label(method: str) -> str:
    return _METHOD_LABELS.get(method, method.replace("_", " "))


def _status_label(status: str) -> str:
    return status.replace("_", " ")


def _evidence_label(record: RungRecord) -> str:
    for candidate in (record.run_path, record.resume_from_run_path):
        if not candidate:
            continue
        match = _CLUSTER_RE.search(str(candidate))
        if match:
            return match.group(1)
    return "prior"


def _number(value: float | None, *, latex: bool) -> str:
    if value is None:
        return "---" if latex else "--"
    if latex:
        return f"${float(value):.3e}$"
    return f"{float(value):.3e}"


def _metric_records(
    ledger: CampaignLedger, *, horizon: str
) -> tuple[RungRecord, ...]:
    return tuple(record for record in ledger.records if record.horizon == horizon)


def _latex_frontier_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return (
            "\\section*{Scheduler frontier and transfer-pending evidence}\n"
            "No unfinished or transfer-pending scheduler rows were declared in "
            "the verified snapshot.\n"
        )
    body: list[str] = []
    for row in rows:
        threshold = row.get("threshold")
        cut = "---" if threshold is None else f"${float(threshold):.3e}$"
        body.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["job_id"]),
                _tex_escape(row["method"]),
                _tex_escape(row["controller"]),
                _tex_escape(row["drive"]),
                cut,
                _tex_escape(row["state"]),
                _tex_escape(row["progress"]),
                _tex_escape(row["available_data"]),
            )
        )
    return rf"""
\section*{{Scheduler frontier and transfer-pending evidence}}
\begin{{center}}
\scriptsize
\resizebox{{0.99\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llllllll@{{}}}}
\toprule
Job & Method & Controller & Drive & $\tau_{{L^2}}$ & State & Progress & Available data \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\footnotesize
Rows in this section are explicit scheduler/frontier metadata. They are not
silently promoted into the immutable ledger and carry no invented accuracy
values.
\normalsize
"""


def _latex_horizon_tables(
    ledger: CampaignLedger, *, horizon: str, chunk_size: int = 46
) -> str:
    records = _metric_records(ledger, horizon=horizon)
    chunks = [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]
    if not chunks:
        chunks = [tuple()]
    pieces: list[str] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        title = (
            f"Current {horizon} frontier --- all {len(records)} verified records"
            if chunk_index == 1
            else f"Current {horizon} frontier --- continued ({chunk_index}/{len(chunks)})"
        )
        body: list[str] = []
        for record in chunk:
            metrics = record.metrics
            body.append(
                "{} & {} & {} & ${:.3e}$ & {} & {} & {} & {} & {} \\\\".format(
                    _tex_escape(_method_label(record.key.method)),
                    _tex_escape(record.key.controller.replace("_", " ")),
                    _tex_escape(record.key.drive),
                    record.threshold,
                    _number(
                        None if metrics is None else metrics.mean_abs_energy_error,
                        latex=True,
                    ),
                    _number(
                        None if metrics is None else metrics.mean_abs_doublon_error,
                        latex=True,
                    ),
                    _number(
                        None
                        if metrics is None
                        else metrics.mean_max_abs_site_occupation_error,
                        latex=True,
                    ),
                    _tex_escape(_evidence_label(record)),
                    _tex_escape(_status_label(record.status)),
                )
            )
        if not body:
            body.append(r"--- & --- & --- & --- & --- & --- & --- & --- & no records \\")
        pieces.append(
            rf"""
\section*{{{_tex_escape(title)}}}
\begin{{center}}
\tiny
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{0.99\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lllrrrrll@{{}}}}
\toprule
Method & Controller & Drive & $\tau_{{L^2}}$ & $\overline{{|\Delta E|}}$ & $\overline{{|\Delta D|}}$ & $\overline{{\max_i|\Delta n_i|}}$ & Evidence & Status \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
"""
        )
        if chunk_index < len(chunks):
            pieces.append("\\clearpage\n")
    counts = Counter(record.status for record in records)
    pieces.append(
        rf"""
\footnotesize
Recorded {horizon} statuses: """
        + "; ".join(
            f"{_tex_escape(_status_label(status))}: {count}"
            for status, count in sorted(counts.items())
        )
        + ". Completed numerical evidence remains visible regardless of its gate state.\n"
        + "\\normalsize\n"
    )
    return "\n".join(pieces)


def _markdown_frontier(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Scheduler frontier and transfer-pending evidence", ""]
    if not rows:
        lines.extend(
            [
                "No unfinished or transfer-pending scheduler rows were declared in the verified snapshot.",
                "",
            ]
        )
        return lines
    lines.extend(
        [
            "| job | method | controller | drive | cut | state | progress | available data |",
            "|---|---|---|---|---:|---|---|---|",
        ]
    )
    for row in rows:
        threshold = row.get("threshold")
        cut = "--" if threshold is None else f"{float(threshold):.3e}"
        lines.append(
            "| {job} | {method} | {controller} | {drive} | {cut} | {state} | {progress} | {data} |".format(
                job=row["job_id"],
                method=row["method"],
                controller=row["controller"],
                drive=row["drive"],
                cut=cut,
                state=row["state"],
                progress=row["progress"],
                data=row["available_data"],
            )
        )
    lines.extend(
        [
            "",
            "These explicit scheduler rows are not ledger records and carry no invented accuracy values.",
            "",
        ]
    )
    return lines


def _markdown_horizon(ledger: CampaignLedger, *, horizon: str) -> list[str]:
    records = _metric_records(ledger, horizon=horizon)
    lines = [
        f"## Current {horizon} frontier — all {len(records)} verified records",
        "",
        "| method | controller | drive | cut | mean |ΔE| | mean |ΔD| | mean maxᵢ|Δnᵢ| | evidence | status |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        metrics = record.metrics
        lines.append(
            "| {method} | {controller} | {drive} | {cut:.3e} | {energy} | {doublon} | {site} | {evidence} | {status} |".format(
                method=_method_label(record.key.method),
                controller=record.key.controller.replace("_", " "),
                drive=record.key.drive,
                cut=record.threshold,
                energy=_number(
                    None if metrics is None else metrics.mean_abs_energy_error,
                    latex=False,
                ),
                doublon=_number(
                    None if metrics is None else metrics.mean_abs_doublon_error,
                    latex=False,
                ),
                site=_number(
                    None
                    if metrics is None
                    else metrics.mean_max_abs_site_occupation_error,
                    latex=False,
                ),
                evidence=_evidence_label(record),
                status=_status_label(record.status),
            )
        )
    if not records:
        lines.append("| -- | -- | -- | -- | -- | -- | -- | -- | no records |")
    lines.extend(
        [
            "",
            "Completed numerical evidence remains visible regardless of its gate state.",
            "",
        ]
    )
    return lines


def _latex_source(
    *,
    ledger: CampaignLedger,
    frontier: Mapping[str, Any],
    scheduler_rows: Sequence[Mapping[str, Any]],
    preview_fragment: Path,
    supplemental: Sequence[Mapping[str, Any]],
    slot_counts: Mapping[str, int],
) -> str:
    clusters = ", ".join(str(value) for value in frontier.get("cluster_ids") or ())
    cluster_text = clusters or "none declared"
    supplemental_source = []
    for item in supplemental:
        fragment = Path(item["_fragment_path"])
        supplemental_source.extend(
            (
                "\\clearpage",
                f"% BEGIN digest-bound supplemental section: {item['role']} {_sha256(fragment)}",
                fragment.read_text(encoding="utf-8").strip(),
                f"% END digest-bound supplemental section: {item['role']}",
            )
        )
    slot_summary = ", ".join(
        f"{state}: {int(slot_counts.get(state, 0))}"
        for state in ("qualified", "candidate", "pending", "unresolved")
        if int(slot_counts.get(state, 0))
    )
    preview_text = preview_fragment.read_text(encoding="utf-8").strip()
    return rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,landscape,margin=0.43in,headsep=5pt,footskip=18pt]{{geometry}}
\usepackage{{booktabs,graphicx,xcolor,fancyhdr,hyperref}}
\definecolor{{paperblue}}{{HTML}}{{29465B}}
\hypersetup{{colorlinks=true,linkcolor=paperblue,urlcolor=paperblue}}
\pagestyle{{fancy}}
\fancyhf{{}}
\lfoot{{\footnotesize Paper II staged accuracy campaign}}
\rfoot{{\footnotesize\thepage}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{3pt}}
\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries Paper II staged campaign and matched-accuracy results}}
\end{{center}}
\vspace{{-0.6em}}
\footnotesize
\textbf{{Campaign snapshot:}} {_tex_escape(frontier['snapshot_utc'])}\quad
\textbf{{Snapshot ID:}} {_tex_escape(frontier['snapshot_id'])}\\
\textbf{{Scheduler:}} {_tex_escape(frontier['scheduler_summary'])}\\
\textbf{{Verified frontier:}} {len(ledger.records)} ledger records across $t=5$ and $t=20$\\
\textbf{{Planned $t=20$ method slots:}} {_tex_escape(slot_summary)}\\
\textbf{{CHTC clusters:}} {_tex_escape(cluster_text)}\\
\textbf{{Audit:}} {_tex_escape(frontier['audit_status'])}
\par Physical errors are post-run diagnostics and are never supplied to either
online controller. Completed numerical data remain visible; only absent
trajectories are pending in the planned-slot matrix.
\normalsize

{_latex_frontier_table(scheduler_rows)}

\clearpage
{_latex_horizon_tables(ledger, horizon='t20')}

\clearpage
{_latex_horizon_tables(ledger, horizon='t5')}

\clearpage
% BEGIN digest-bound promotion preview {_sha256(preview_fragment)}
{preview_text}
% END digest-bound promotion preview

{chr(10).join(supplemental_source)}

\end{{document}}
"""


def _markdown_source(
    *,
    ledger: CampaignLedger,
    frontier: Mapping[str, Any],
    scheduler_rows: Sequence[Mapping[str, Any]],
    preview: Mapping[str, Any],
    supplemental: Sequence[Mapping[str, Any]],
    slot_counts: Mapping[str, int],
) -> str:
    lines = [
        "# Paper II staged campaign and matched-accuracy results",
        "",
        f"- Campaign snapshot: `{frontier['snapshot_utc']}`",
        f"- Snapshot ID: `{frontier['snapshot_id']}`",
        f"- Scheduler: {frontier['scheduler_summary']}",
        f"- Verified frontier: {len(ledger.records)} immutable records across `t5` and `t20`",
        (
            "- Planned `t20` method slots: "
            + ", ".join(
                f"{state}={int(slot_counts.get(state, 0))}"
                for state in ("qualified", "candidate", "pending", "unresolved")
                if int(slot_counts.get(state, 0))
            )
        ),
        f"- Audit: {frontier['audit_status']}",
        "",
        *_markdown_frontier(scheduler_rows),
        *_markdown_horizon(ledger, horizon="t20"),
        *_markdown_horizon(ledger, horizon="t5"),
        "## Planned t20 exact/APM/AVQDS tuple matrix and promotion previews",
        "",
        (
            f"The digest-bound promotion contract contains {preview['planned_tuple_count']} "
            f"planned controller-aware tuples and {preview['pair_count']} complete pair previews."
        ),
        "",
    ]
    for item in supplemental:
        lines.extend(
            [
                f"## {str(item['role']).replace('_', ' ')}",
                "",
                str(item["markdown_summary"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _compile_pdf(tex_path: Path, *, source_date_epoch: int) -> tuple[Path, str]:
    """Compile a staged TeX source without publishing partial output."""

    build_dir = tex_path.parent / "latex_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, list[str], int]] = []
    if shutil.which("latexmk"):
        candidates.append(
            (
                "latexmk",
                [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-outdir={build_dir.name}",
                    tex_path.name,
                ],
                1,
            )
        )
    if shutil.which("pdflatex"):
        candidates.append(
            (
                "pdflatex",
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={build_dir.name}",
                    tex_path.name,
                ],
                2,
            )
        )
    if not candidates:
        raise RuntimeError("no LaTeX engine is available for the combined report")
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = str(int(source_date_epoch))
    environment["FORCE_SOURCE_DATE"] = "1"
    failures: list[str] = []
    for engine, command, passes in candidates:
        completed = None
        for _ in range(passes):
            completed = subprocess.run(
                command,
                cwd=tex_path.parent,
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
                env=environment,
            )
            if completed.returncode != 0:
                break
        assert completed is not None
        built = build_dir / f"{tex_path.stem}.pdf"
        if completed.returncode == 0 and built.is_file() and built.stat().st_size > 0:
            return built, engine
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-35:])
        failures.append(f"{engine}: {tail}")
    raise RuntimeError("combined report LaTeX build failed:\n" + "\n\n".join(failures))


def build_combined_report(
    *,
    ledger_path: Path,
    promotion_preview_path: Path,
    frontier_metadata_path: Path,
    output_dir: Path,
    workspace_root: Path = Path("."),
) -> dict[str, Any]:
    """Verify, stage, compile, and atomically publish the combined report.

    ``frontier_metadata_path`` is the only scheduler input.  Callers must first
    materialize one scoped queue/history snapshot and bind it to the current
    ledger and promotion-preview digests.  The builder performs no remote I/O.
    """

    workspace_root = workspace_root.resolve()
    ledger_path = ledger_path.resolve()
    promotion_preview_path = promotion_preview_path.resolve()
    frontier_metadata_path = frontier_metadata_path.resolve()
    output_dir = output_dir.resolve()

    ledger = CampaignLedger.read(ledger_path)
    preview = _read_object(promotion_preview_path, label="promotion preview")
    frontier = _read_object(frontier_metadata_path, label="frontier metadata")
    preview_fragment, slot_counts = _verify_preview(
        preview=preview,
        preview_path=promotion_preview_path,
        ledger=ledger,
        ledger_path=ledger_path,
        workspace_root=workspace_root,
    )
    scheduler_rows, supplemental = _validate_frontier(
        frontier=frontier,
        frontier_path=frontier_metadata_path,
        ledger_path=ledger_path,
        preview_path=promotion_preview_path,
        ledger=ledger,
        preview=preview,
        workspace_root=workspace_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if list(preview.get("pairs") or ()) and not plots_dir.is_dir():
        raise CombinedReportContractError(
            f"canonical promotion plots directory is missing: {plots_dir}"
        )

    snapshot_time = datetime.fromisoformat(
        str(frontier["snapshot_utc"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    source_epoch = max(0, int(snapshot_time.timestamp()))
    with tempfile.TemporaryDirectory(
        prefix=".paper_ii_combined_report.", dir=output_dir
    ) as temporary:
        stage = Path(temporary)
        if plots_dir.is_dir():
            (stage / "plots").symlink_to(plots_dir, target_is_directory=True)
        markdown = stage / "paper_ii_staged_report.md"
        tex = stage / "paper_ii_staged_report.tex"
        markdown.write_text(
            _markdown_source(
                ledger=ledger,
                frontier=frontier,
                scheduler_rows=scheduler_rows,
                preview=preview,
                supplemental=supplemental,
                slot_counts=slot_counts,
            ),
            encoding="utf-8",
        )
        tex.write_text(
            _latex_source(
                ledger=ledger,
                frontier=frontier,
                scheduler_rows=scheduler_rows,
                preview_fragment=preview_fragment,
                supplemental=supplemental,
                slot_counts=slot_counts,
            ),
            encoding="utf-8",
        )
        staged_pdf, engine = _compile_pdf(tex, source_date_epoch=source_epoch)
        if not staged_pdf.is_file() or staged_pdf.stat().st_size <= 0:
            raise RuntimeError("combined report compiler returned no nonempty PDF")
        page_count = len(PdfReader(staged_pdf).pages)
        if page_count <= 0:
            raise RuntimeError("combined report PDF has no pages")

        pair_count = int(preview.get("pair_count", 0))
        manifest = {
            "schema": REPORT_SCHEMA,
            "builder_schema": COMBINED_REPORT_BUILDER_SCHEMA,
            "generated_at": str(frontier["snapshot_utc"]),
            "horizon": COMBINED_HORIZON,
            "ledger_record_count": len(ledger.records),
            "verified_frontier_row_count": len(ledger.records),
            "source_ledger": _relative(ledger_path, workspace_root=workspace_root),
            "source_ledger_sha256": _sha256(ledger_path),
            "qualification_policy": dict(ledger.qualification_policy),
            "frontier_metadata": {
                "path": _relative(frontier_metadata_path, workspace_root=workspace_root),
                "sha256": _sha256(frontier_metadata_path),
                "snapshot_id": str(frontier["snapshot_id"]),
            },
            "promotion_preview": {
                "receipt": _relative(
                    promotion_preview_path, workspace_root=workspace_root
                ),
                "receipt_sha256": _sha256(promotion_preview_path),
                "planned_tuple_count": int(preview["planned_tuple_count"]),
                "pair_count": pair_count,
                "tex_fragment": _relative(
                    preview_fragment, workspace_root=workspace_root
                ),
                "tex_fragment_sha256": _sha256(preview_fragment),
                "status": str(preview.get("status") or ""),
            },
            "qualified_pair_count": sum(
                all(
                    str(dict(dict(pair.get("rows") or {})[method]).get("status"))
                    == "accuracy_qualified"
                    for method in (METHOD_APM, METHOD_AVQDS)
                )
                for pair in preview.get("pairs") or ()
            ),
            "pairs": list(preview.get("pairs") or ()),
            "t20_accuracy_qualified_method_arm_count": int(
                slot_counts.get("qualified", 0)
            ),
            "t20_candidate_method_arm_count": int(slot_counts.get("candidate", 0)),
            "t20_pending_method_arm_count": int(slot_counts.get("pending", 0)),
            "t20_unresolved_method_arm_count": int(
                slot_counts.get("unresolved", 0)
            ),
            "cluster_ids": [int(value) for value in frontier.get("cluster_ids") or ()],
            "active_clusters": list(frontier.get("active_clusters") or ()),
            "excluded_cluster_ids": [
                int(value) for value in frontier.get("excluded_cluster_ids") or ()
            ],
            "incremental_ingestion_receipts": list(
                frontier.get("incremental_ingestion_receipts") or ()
            ),
            "submission_receipts": list(frontier.get("submission_receipts") or ()),
            "audit_status": str(frontier["audit_status"]),
            "scheduler_summary": str(frontier["scheduler_summary"]),
            "scheduler_row_count": len(scheduler_rows),
            "supplemental_sections": [
                {
                    "role": str(item["role"]),
                    "tex_fragment": {
                        "path": _relative(
                            Path(item["_fragment_path"]), workspace_root=workspace_root
                        ),
                        "sha256": _sha256(Path(item["_fragment_path"])),
                    },
                }
                for item in supplemental
            ],
            "markdown": _relative(
                output_dir / markdown.name, workspace_root=workspace_root
            ),
            "tex": _relative(output_dir / tex.name, workspace_root=workspace_root),
            "pdf": _relative(
                output_dir / "paper_ii_staged_report.pdf",
                workspace_root=workspace_root,
            ),
            "pdf_engine": engine,
            "page_count": page_count,
        }

        published_markdown = output_dir / markdown.name
        published_tex = output_dir / tex.name
        published_pdf = output_dir / "paper_ii_staged_report.pdf"
        for source, target in (
            (markdown, published_markdown),
            (tex, published_tex),
            (staged_pdf, published_pdf),
        ):
            staged_target = target.with_name(target.name + ".part")
            shutil.copy2(source, staged_target)
            staged_target.replace(target)
        manifest.update(
            {
                "markdown_sha256": _sha256(published_markdown),
                "tex_sha256": _sha256(published_tex),
                "pdf_bytes": published_pdf.stat().st_size,
                "pdf_sha256": _sha256(published_pdf),
            }
        )
        manifest_path = output_dir / "paper_ii_staged_report.json"
        manifest_part = manifest_path.with_name(manifest_path.name + ".part")
        manifest_part.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_part.replace(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--promotion-preview", required=True, type=Path)
    parser.add_argument("--frontier-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    manifest = build_combined_report(
        ledger_path=args.ledger,
        promotion_preview_path=args.promotion_preview,
        frontier_metadata_path=args.frontier_metadata,
        output_dir=args.output_dir,
        workspace_root=args.workspace_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMBINED_FRONTIER_SCHEMA",
    "COMBINED_HORIZON",
    "COMBINED_REPORT_BUILDER_SCHEMA",
    "CombinedReportContractError",
    "build_combined_report",
    "main",
]
