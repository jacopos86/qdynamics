"""Build copy-ready Paper-II promotion previews from the live campaign ledger.

A preview is deliberately weaker than manuscript promotion: it requires a
verified, exact-evaluated t=20 APM/AVQDS pair, but it does not require either
arm to have met the response target or pair-matching rule. The generated
physics panels and compact table already use the manuscript contract so that a
later promotion changes evidence status and provenance, not presentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.diagnostics.exact_reference_store import (
    ExactReferenceStore,
)
from pipelines.time_dynamics.diagnostics.provisional_t20_manuscript_assets import (
    CONTROLLER_TEX,
    DRIVE_ORDER,
    DRIVE_TEX,
    _build_figure,
    _full_trajectory,
    _tex_scientific,
)
from pipelines.time_dynamics.diagnostics.promotion_contract import (
    PROMOTION_PREVIEW_SCHEMA,
    planned_tuple_id,
    tuple_id,
)
from pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources import (
    load_terminal_resources,
)
from pipelines.time_dynamics.run_lock import assert_comparable
from pipelines.time_dynamics.staged_campaign import (
    AccuracyPolicy,
    CampaignLedger,
    METHOD_APM,
    METHOD_AVQDS,
    POLICY_MEAN_INFIDELITY_WINDOW,
    POLICY_RESPONSE_NORMALIZED_MATCHED,
    RungRecord,
    STATUS_ACCURACY_QUALIFIED,
    STATUS_REQUIRES_THRESHOLD_COARSENING,
    STATUS_REQUIRES_THRESHOLD_EXTENSION,
    STATUS_SAFEGUARD_TRIPPED,
)


SCHEMA = PROMOTION_PREVIEW_SCHEMA
METHOD_LABELS = {METHOD_APM: "full APM", METHOD_AVQDS: "AVQDS"}


def _pair_matched(record: RungRecord) -> bool:
    return record.reason.startswith("matched_guard:")


def _select_record(existing: RungRecord | None, record: RungRecord) -> RungRecord:
    """Prefer a matched-pair rung, then the first individual target, then latest."""

    if existing is None:
        return record
    if _pair_matched(record):
        return record
    if _pair_matched(existing):
        return existing
    if existing.status == STATUS_ACCURACY_QUALIFIED:
        return existing
    if record.status == STATUS_ACCURACY_QUALIFIED:
        return record
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError(f"required preview artifact is missing or empty: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def promotion_preview_pairs(ledger: CampaignLedger) -> tuple[dict[str, Any], ...]:
    """Return the selected exact-evaluated t20 record for each complete pair."""

    latest: dict[tuple[str, str, str, str], RungRecord] = {}
    for record in ledger.records:
        if (
            record.horizon != "t20"
            or record.key.method not in (METHOD_APM, METHOD_AVQDS)
            or record.metrics is None
            or record.run_path is None
            or record.resume_from_run_path is None
        ):
            continue
        identity = (
            record.key.regime,
            record.key.drive,
            record.key.controller,
            record.key.method,
        )
        latest[identity] = _select_record(latest.get(identity), record)

    grouped: dict[tuple[str, str, str], dict[str, RungRecord]] = {}
    for (regime, drive, controller, method), record in latest.items():
        grouped.setdefault((regime, drive, controller), {})[method] = record

    pairs: list[dict[str, Any]] = []
    for (regime, drive, controller), methods in grouped.items():
        if METHOD_APM not in methods or METHOD_AVQDS not in methods:
            continue
        records = {
            METHOD_APM: methods[METHOD_APM],
            METHOD_AVQDS: methods[METHOD_AVQDS],
        }
        # The screen locks bind the common source pool.  A continuation's live
        # insertion eligibility is support-dependent and therefore legitimately
        # differs after APM and AVQDS have made different structural decisions.
        source_run_locks = tuple(
            dict(_read_json(record.resume_from_run_path).get("run_lock") or {})
            for record in records.values()
        )
        source_comparison_fingerprint = assert_comparable(source_run_locks)
        pairs.append(
            {
                "regime": regime,
                "drive": drive,
                "controller": controller,
                "records": records,
                "source_comparison_fingerprint": source_comparison_fingerprint,
            }
        )
    pairs.sort(
        key=lambda pair: (
            DRIVE_ORDER.get(pair["drive"], 99),
            pair["controller"],
        )
    )
    return tuple(pairs)


def _summed_steps(record: RungRecord) -> int:
    total = 0
    for path in (record.resume_from_run_path, record.run_path):
        summary = dict(_read_json(path).get("summary") or {})
        total += int(summary.get("accepted_internal_substep_count") or 0)
    return total


def _terminal_resources(record: RungRecord) -> dict[str, Any]:
    if record.run_path is None:
        raise ValueError("preview record has no run path")
    resources = load_terminal_resources(record.run_path)
    return {
        "cost_path": str(resources.artifact_path),
        "cost_sha256": resources.artifact_sha256,
        "cost_source_kind": resources.source_kind,
        "n_theta": resources.n_theta,
        "graph": list(resources.graph),
    }


def _ledger_policy(ledger: CampaignLedger) -> AccuracyPolicy:
    return AccuracyPolicy.from_json_dict(ledger.qualification_policy)


def _failed_gates(
    record: RungRecord,
    policy: AccuracyPolicy,
) -> list[tuple[str, float]]:
    if record.metrics is None or record.metrics.mean_infidelity is None:
        raise ValueError("preview record has no exact-reference metrics")
    if policy.mode == POLICY_MEAN_INFIDELITY_WINDOW:
        value = float(record.metrics.mean_infidelity)
        assert policy.infidelity_floor is not None
        assert policy.infidelity_ceiling is not None
        if value < float(policy.infidelity_floor):
            return [(r"$\overline{1-F}$ below matching window", value)]
        if value > float(policy.infidelity_ceiling):
            return [(r"$\overline{1-F}$ above matching window", value)]
        return []
    if policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
        if record.status == STATUS_ACCURACY_QUALIFIED and record.reason:
            return []
        components = policy.response_components(
            record.metrics,
            key=record.key,
            horizon=record.horizon,
        )
        failures = [
            (r"$q_E$", components["mean_abs_energy_error"]),
            (r"$q_D$", components["mean_abs_doublon_error"]),
            (
                r"$q_n$",
                components["mean_max_abs_site_occupation_error"],
            ),
        ]
        result = [
            (label, float(value))
            for label, value in failures
            if value > 1.0
        ]
        if components["mean_infidelity"] > 1.0:
            result.append(
                (r"$\overline{1-F}/I_{\max}$", components["mean_infidelity"])
            )
        return result
    values = (
        (r"$\overline{|\Delta E|}$", record.metrics.mean_abs_energy_error),
        (r"$\overline{|\Delta D|}$", record.metrics.mean_abs_doublon_error),
        (
            r"$\overline{\max_i|\Delta n_i|}$",
            record.metrics.mean_max_abs_site_occupation_error,
        ),
        (r"$\overline{1-F}$", record.metrics.mean_infidelity),
    )
    return [
        (label, float(value))
        for label, value in values
        if value > float(policy.ceiling)
    ]


def _preview_row(record: RungRecord, policy: AccuracyPolicy) -> dict[str, Any]:
    resources = _terminal_resources(record)
    metrics = record.metrics
    assert metrics is not None and metrics.mean_infidelity is not None
    row = {
        "method": record.key.method,
        "threshold": float(record.threshold),
        "status": record.status,
        "run_path": str(record.run_path),
        "run_sha256": _sha256(Path(record.run_path)),
        "screen_run_path": str(record.resume_from_run_path),
        "screen_run_sha256": _sha256(Path(record.resume_from_run_path)),
        "mean_infidelity": float(metrics.mean_infidelity),
        "mean_abs_energy_error": float(metrics.mean_abs_energy_error),
        "mean_abs_doublon_error": float(metrics.mean_abs_doublon_error),
        # The declared two-site benchmark has fixed total particle number, so
        # |delta(n1-n2)| = 2 max_i |delta n_i| at every checkpoint.
        "mean_abs_charge_imbalance_error": 2.0
        * float(metrics.mean_max_abs_site_occupation_error),
        "time_step_count": _summed_steps(record),
        "failed_gates": [
            {"label": label, "value": value}
            for label, value in _failed_gates(record, policy)
        ],
        **resources,
    }
    if policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
        row["response_normalized_errors"] = policy.response_components(
            metrics,
            key=record.key,
            horizon=record.horizon,
        )
        row["accuracy_burden"] = policy.accuracy_burden(
            metrics,
            key=record.key,
            horizon=record.horizon,
        )
        row["qualification_basis"] = (
            "pair_matched_comparison"
            if record.reason.startswith("matched_guard:")
            else (
                "individual_response_target"
                if record.status == STATUS_ACCURACY_QUALIFIED
                else "outside_individual_response_target"
            )
        )
    return row


def _latest_t20_records(
    ledger: CampaignLedger,
) -> dict[tuple[str, str, str, str], RungRecord]:
    latest: dict[tuple[str, str, str, str], RungRecord] = {}
    for record in ledger.records:
        if (
            record.horizon != "t20"
            or record.key.method not in (METHOD_APM, METHOD_AVQDS)
            or record.metrics is None
            or record.run_path is None
            or record.resume_from_run_path is None
        ):
            continue
        identity = (
            record.key.regime,
            record.key.drive,
            record.key.controller,
            record.key.method,
        )
        latest[identity] = _select_record(latest.get(identity), record)
    return latest


def _latest_t20_outcomes(
    ledger: CampaignLedger,
) -> dict[tuple[str, str, str, str], RungRecord]:
    """Return the last ledger outcome, including provenance-only safeguards."""

    latest: dict[tuple[str, str, str, str], RungRecord] = {}
    for record in ledger.records:
        if (
            record.horizon != "t20"
            or record.key.method not in (METHOD_APM, METHOD_AVQDS)
        ):
            continue
        identity = (
            record.key.regime,
            record.key.drive,
            record.key.controller,
            record.key.method,
        )
        latest[identity] = record
    return latest


def _exact_references_by_drive(
    *,
    ledger: CampaignLedger,
    seed_path: Path,
    store: ExactReferenceStore,
    reference_substeps: int,
) -> dict[tuple[str, str], dict[str, str]]:
    references: dict[tuple[str, str], dict[str, str]] = {}
    for record in reversed(ledger.records):
        identity = (record.key.regime, record.key.drive)
        if identity in references:
            continue
        if (
            record.horizon != "t20"
            or record.metrics is None
            or record.run_path is None
            or record.resume_from_run_path is None
        ):
            continue
        observed = _full_trajectory(
            Path(record.resume_from_run_path), Path(record.run_path)
        )
        reference = store.get_or_build(
            seed_path=seed_path,
            run_path=Path(record.run_path),
            times=[float(row["time"]) for row in observed],
            substeps=int(reference_substeps),
        )
        references[identity] = {
            "path": str(reference.manifest_path),
            "sha256": _sha256(reference.manifest_path),
        }
    return references


def _slot_reason(
    record: RungRecord | None,
    outcome: RungRecord | None,
) -> str:
    if outcome is not None and outcome.status == STATUS_SAFEGUARD_TRIPPED:
        return "ladder_or_safeguard_exhausted"
    if record is None:
        return "t20_not_completed"
    if record.status == "requires_threshold_extension":
        return "accuracy_gate_pending_threshold_extension"
    if record.status == "requires_threshold_coarsening":
        return "infidelity_below_window_pending_threshold_coarsening"
    if record.status == "safeguard_tripped":
        return "ladder_or_safeguard_exhausted"
    if record.status == "accuracy_qualified":
        return ""
    return str(record.status)


def planned_tuple_matrix(
    *,
    ledger: CampaignLedger,
    seed_path: Path,
    store: ExactReferenceStore,
    reference_substeps: int,
) -> tuple[dict[str, Any], ...]:
    """Return every planned controller-specific exact/APM/AVQDS tuple."""

    latest = _latest_t20_records(ledger)
    outcomes = _latest_t20_outcomes(ledger)
    policy = _ledger_policy(ledger)
    references = _exact_references_by_drive(
        ledger=ledger,
        seed_path=seed_path,
        store=store,
        reference_substeps=reference_substeps,
    )
    cells = sorted(
        {
            (record.key.regime, record.key.drive, record.key.controller)
            for record in ledger.records
        },
        key=lambda value: (
            value[0],
            DRIVE_ORDER.get(value[1], 99),
            value[2],
        ),
    )
    result: list[dict[str, Any]] = []
    for regime, drive, controller in cells:
        exact = references.get((regime, drive))
        if exact is None:
            raise ValueError(
                f"planned t20 tuple lacks a verified shared exact reference: "
                f"{regime}/{drive}/{controller}"
            )
        methods: dict[str, dict[str, Any]] = {}
        for method in (METHOD_APM, METHOD_AVQDS):
            identity = (regime, drive, controller, method)
            record = latest.get(identity)
            outcome = outcomes.get(identity)
            row = None
            if (
                record is not None
                and record.metrics is not None
                and record.run_path is not None
                and record.resume_from_run_path is not None
            ):
                row = _preview_row(record, policy)
            if record is not None and record.status == STATUS_ACCURACY_QUALIFIED:
                paper_status = "qualified"
            elif outcome is not None and outcome.status == STATUS_SAFEGUARD_TRIPPED:
                paper_status = "unresolved"
            elif (
                record is not None
                and record.status
                in {
                    STATUS_REQUIRES_THRESHOLD_EXTENSION,
                    STATUS_REQUIRES_THRESHOLD_COARSENING,
                }
                and row is not None
            ):
                paper_status = "candidate"
            else:
                paper_status = "pending"
            state_reason = _slot_reason(record, outcome)
            methods[method] = {
                "paper_status": paper_status,
                "state_reason": state_reason,
                # Retained for backward-compatible consumers; only a pending
                # slot should interpret this field as a pending reason.
                "pending_reason": state_reason,
                "latest_evidence_status": (
                    "not_run" if outcome is None else outcome.status
                ),
                "latest_row": row,
            }
        item: dict[str, Any] = {
            "regime": regime,
            "drive": drive,
            "controller": controller,
            "horizon": "t20",
            "exact_reference": exact,
            "methods": methods,
        }
        item["tuple_id"] = planned_tuple_id(item)
        result.append(item)
    return tuple(result)


def _matrix_status(slot: Mapping[str, Any]) -> str:
    if slot["paper_status"] == "qualified":
        return "qualified"
    if slot["paper_status"] == "candidate":
        return (
            "candidate; coarsen"
            if slot.get("latest_evidence_status")
            == STATUS_REQUIRES_THRESHOLD_COARSENING
            else "candidate; tighten"
        )
    if slot["paper_status"] == "unresolved":
        return "unresolved; ladder/resource exhausted"
    status = str(slot.get("latest_evidence_status") or "not_run")
    labels = {
        "requires_threshold_extension": "evaluated; extension",
        "requires_threshold_coarsening": "evaluated; coarsening",
        "safeguard_tripped": "safeguard",
        "not_run": "not completed",
    }
    return labels.get(status, "pending")


def _planned_matrix_method_row(
    item: Mapping[str, Any], method: str, slot: Mapping[str, Any]
) -> str:
    prefix = (
        f"{DRIVE_TEX.get(item['drive'], item['drive'])} & "
        f"{CONTROLLER_TEX.get(item['controller'], item['controller'])} & "
        f"{METHOD_LABELS[method]}"
    )
    if slot["paper_status"] in {"pending", "unresolved"} and slot.get(
        "latest_row"
    ) is None:
        return (
            f"{prefix} & "
            f"\\multicolumn{{7}}{{c}}{{\\textit{{{_matrix_status(slot)}}}}} \\\\"
        )
    row = dict(slot["latest_row"])
    graph = ",".join(str(value) for value in row["graph"])
    return (
        f"{prefix} & ${_tex_scientific(float(row['mean_infidelity']))}$ & "
        f"${_tex_scientific(float(row['mean_abs_energy_error']))}$ & "
        f"${_tex_scientific(float(row['mean_abs_doublon_error']))}$ & "
        f"${_tex_scientific(float(row['mean_abs_charge_imbalance_error']))}$ & "
        f"{int(row['time_step_count']):,} & {int(row['n_theta'])} & "
        f"$({graph})$ \\\\"
    )


def planned_tuple_matrix_tex(items: Sequence[Mapping[str, Any]]) -> str:
    """Render the complete t20 tuple scaffold before candidate preview pages."""

    body: list[str] = []
    for item in items:
        methods = dict(item["methods"])
        for method in (METHOD_APM, METHOD_AVQDS):
            body.append(_planned_matrix_method_row(item, method, methods[method]))
    return rf"""
\section*{{Planned $t=20$ exact/APM/AVQDS tuple matrix}}
Every physical-regime, drive, and time-step-controller cell exists before all
of its arms finish. One verified exact reference is shared across controller
cells for the same physical drive. Every completed arm is shown numerically;
only arms without a valid completed trajectory are pending. Qualification
uses exact-response-normalized observable errors. Each arm targets at most five
percent of the induced RMS response and $\overline{{1-F}}\leq10^{{-4}}$. A
completed pair may instead be admitted for cost comparison when both arms stay
inside the fifteen-percent physical guard and their aggregate accuracy burdens
agree within a factor of two, with every deadbanded primitive $q$ component
also agreeing within that factor.
\vspace{{-0.5em}}
\begin{{center}}
\tiny
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{0.90\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lllrrrrrrr@{{}}}}
\toprule
Drive & controller & method & {{$\overline{{1-F}}$}} & {{$\overline{{|\Delta E|}}$}} & {{$\overline{{|\Delta D|}}$}} & {{$\overline{{|\Delta(\Delta n)|}}$}} & {{$N_{{\rm step}}$}} & {{$N_\theta$}} & {{$\widehat{{\mathbf C}}_G$}} \\
\midrule
{chr(10).join(body)}
\addlinespace[2pt]
\multicolumn{{3}}{{l}}{{insertion McLachlan (IM)}} & \multicolumn{{7}}{{c}}{{\textit{{$t=20$ matrix pending}}}} \\
\bottomrule
\end{{tabular}}}}
\end{{center}}
"""


def _qualification_note(rows: Mapping[str, Mapping[str, Any]]) -> str:
    notes: list[str] = []
    for method in (METHOD_APM, METHOD_AVQDS):
        row = rows[method]
        failed = list(row["failed_gates"])
        if row.get("qualification_basis") == "pair_matched_comparison":
            note = "response-normalized pair-matching rule passes"
        elif row.get("qualification_basis") == "individual_response_target":
            note = "individual response target passes"
        elif not failed:
            note = "all declared accuracy gates pass"
        else:
            rendered = ", ".join(
                f"{item['label']}=${_tex_scientific(float(item['value']))}$"
                for item in failed
            )
            note = f"threshold calibration pending ({rendered})"
        notes.append(f"{METHOD_LABELS[method]}: {note}")
    return "; ".join(notes)


def preview_pair_tex(
    *,
    pair: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    figure_path: Path,
) -> str:
    """Return one copy-ready plot/table section plus explicit gate status."""

    body: list[str] = []
    for method in (METHOD_APM, METHOD_AVQDS):
        row = rows[method]
        graph = ",".join(str(value) for value in row["graph"])
        body.append(
            f"{METHOD_LABELS[method]} & "
            f"${_tex_scientific(float(row['mean_infidelity']))}$ & "
            f"${_tex_scientific(float(row['mean_abs_energy_error']))}$ & "
            f"${_tex_scientific(float(row['mean_abs_doublon_error']))}$ & "
            f"${_tex_scientific(float(row['mean_abs_charge_imbalance_error']))}$ & "
            f"{int(row['time_step_count']):,} & {int(row['n_theta'])} & "
            f"$({graph})$ \\\\"
        )
    body.append(
        r"insertion McLachlan (IM) & \multicolumn{7}{c}{\textit{t=20 row pending}} \\"
    )
    title = (
        f"{DRIVE_TEX.get(pair['drive'], pair['drive'])}: "
        f"{CONTROLLER_TEX.get(pair['controller'], pair['controller'])} controller"
    )
    return rf"""
\section*{{Promotion preview --- {title}}}
\textbf{{Status: verified exact-evaluated comparison evidence.}}
This page is deliberately formatted as a Paper-II result preview. Promotion
requires either the individual response target or the declared matched-pair
comparison rule.
\begin{{center}}
\includegraphics[width=0.94\textwidth,height=0.50\textheight,keepaspectratio]{{{figure_path.as_posix()}}}
\end{{center}}
\vspace{{-0.4em}}
\begin{{center}}
\scriptsize
\resizebox{{0.99\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
Method & {{$\overline{{1-F}}$}} & {{$\overline{{|\Delta E|}}$}} & {{$\overline{{|\Delta D|}}$}} & {{$\overline{{|\Delta(\Delta n)|}}$}} & {{$N_{{\rm step}}$}} & {{$N_\theta$}} & {{$\widehat{{\mathbf C}}_G=(\widehat C_{{2q}},\widehat C_D,\widehat C_{{1q}})$}} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\footnotesize\textbf{{Current qualification gap.}} {_qualification_note(rows)}.
\par Exact: dotted black; full APM: solid dark red; AVQDS: dashed blue. Mean
infidelity and all three mean observable errors cover the stitched 501-point
trajectory on $0\leq t\leq20$. For this fixed-number two-site benchmark,
$\overline{{|\Delta(\Delta n)|}}=2\,\overline{{\max_i|\Delta n_i|}}$.
The exact reference is post-run and shared by both methods.
\normalsize
"""


def write_promotion_preview(
    *,
    ledger: CampaignLedger,
    ledger_path: Path,
    seed_path: Path,
    reference_root: Path,
    output_dir: Path,
    reference_substeps: int = 40,
) -> dict[str, Any]:
    pairs = promotion_preview_pairs(ledger)
    policy = _ledger_policy(ledger)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    store = ExactReferenceStore(reference_root)
    planned_tuples = planned_tuple_matrix(
        ledger=ledger,
        seed_path=seed_path,
        store=store,
        reference_substeps=int(reference_substeps),
    )
    sections: list[str] = []
    pair_receipts: list[dict[str, Any]] = []
    for pair in pairs:
        records = pair["records"]
        rows = {
            method: _preview_row(records[method], policy)
            for method in (METHOD_APM, METHOD_AVQDS)
        }
        slug = f"promotion_preview__{pair['drive']}__{pair['controller']}__t20"
        figure_path = plot_dir / f"{slug}.png"
        figure_sources = _build_figure(
            apm_row=rows[METHOD_APM],
            avqds_row=rows[METHOD_AVQDS],
            seed_path=seed_path,
            reference_store=store,
            reference_substeps=int(reference_substeps),
            output=figure_path,
        )
        exact_reference = dict(figure_sources[-1])
        sections.append(
            preview_pair_tex(
                pair=pair,
                rows=rows,
                figure_path=figure_path.relative_to(output_dir),
            )
        )
        pair_receipt = {
                "regime": pair["regime"],
                "drive": pair["drive"],
                "controller": pair["controller"],
                "horizon": "t20",
                "source_comparison_fingerprint": pair[
                    "source_comparison_fingerprint"
                ],
                "rows": rows,
                "exact_reference": exact_reference,
                "figure": {
                    "path": str(figure_path),
                    "sha256": _sha256(figure_path),
                },
                "figure_sources": figure_sources,
            }
        pair_receipt["tuple_id"] = tuple_id(pair_receipt)
        pair_receipts.append(pair_receipt)
    fragment = output_dir / "promotion_preview_sections.tex"
    fragment.write_text(
        "\n\\clearpage\n".join(
            [planned_tuple_matrix_tex(planned_tuples), *sections]
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_preview_with_response_normalized_pair_states",
        "qualification_policy": policy.to_json_dict(),
        "selection": (
            "first response-qualified or pair-matched rung when present, "
            "otherwise latest "
            "exact-evaluated t20 rung, per method and "
            "regime-drive-time-step-controller cell"
        ),
        "ledger": {
            "path": str(ledger_path),
            "sha256": _sha256(ledger_path),
            "record_count": len(ledger.records),
        },
        "pair_count": len(pairs),
        "pairs": pair_receipts,
        "planned_tuple_count": len(planned_tuples),
        "planned_tuples": list(planned_tuples),
        "tex_fragment": {
            "path": str(fragment),
            "sha256": _sha256(fragment),
        },
    }
    receipt_path = output_dir / "promotion_preview.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-substeps", type=int, default=40)
    args = parser.parse_args(argv)
    receipt = write_promotion_preview(
        ledger=CampaignLedger.read(args.ledger),
        ledger_path=args.ledger,
        seed_path=args.seed,
        reference_root=args.reference_root,
        output_dir=args.output_dir,
        reference_substeps=int(args.reference_substeps),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "main",
    "preview_pair_tex",
    "planned_tuple_matrix",
    "planned_tuple_matrix_tex",
    "promotion_preview_pairs",
    "write_promotion_preview",
]
