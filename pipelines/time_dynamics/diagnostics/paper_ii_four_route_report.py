#!/usr/bin/env python3
"""Render the current-best Paper-II selector report as LaTeX and Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    ROUTE_ORDER,
    SCHEMA as COMPARISON_SCHEMA,
)


REPORT_SCHEMA = "paper_ii_selector_human_report_v2"
DISPLAY_LABELS = {
    "q_compiled": r"$\Gamma(v^\star)$",
    "gamma_compiled": r"$\Gamma(v)$",
    "repaired_proxy": "repaired proxy",
    "old_proxy": "old proxy$^{*}$",
    "avqds": "AVQDS$^{*}$",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _integer(value: Any, *, name: str) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(numeric)


def load_fresh_terminal_cost(
    path: Path,
    *,
    expected_trajectory_sha256: str,
) -> dict[str, Any]:
    """Validate one fresh, common-profile terminal compilation sidecar."""

    payload = _read_json(path)
    if payload.get("schema") != "ap_terminal_qiskit_cost_table_v1":
        raise ValueError(f"unexpected terminal-cost schema: {path}")
    defaults = payload.get("compile_defaults") or {}
    expected_defaults = {
        "backend_name": "FakeMarrakesh",
        "requested_backend_name": "FakeMarrakesh",
        "local_fake_only": True,
        "optimization_level": 1,
        "seed_transpiler": 7,
    }
    for name, expected in expected_defaults.items():
        if defaults.get(name) != expected:
            raise ValueError(f"terminal compile profile mismatch for {name}: {path}")
    result = payload.get("compile_result") or {}
    parity = result.get("terminal_reconstruction_parity") or {}
    if parity.get("passed") is not True:
        raise ValueError(f"terminal reconstruction parity failed: {path}")
    if (
        result.get("requested_backend_name") != "FakeMarrakesh"
        or result.get("backend_name") != "FakeMarrakesh"
        or result.get("local_fake_only") is not True
        or result.get("runtime_lookup_enabled") is not False
        or result.get("optimization_level") != 1
        or result.get("seed_transpiler") != 7
    ):
        raise ValueError(f"terminal compile result changed the locked profile: {path}")
    audits = result.get("resolution_audit") or ()
    if not audits or any(
        audit.get("success") is not True
        or audit.get("resolved_name") != "FakeMarrakesh"
        or audit.get("fallback_reason") is not None
        for audit in audits
    ):
        raise ValueError(f"terminal backend resolution is not exact: {path}")
    rows = payload.get("rows") or ()
    if len(rows) != 1:
        raise ValueError(f"terminal cost table must contain one row: {path}")
    row = rows[0]
    if row.get("trajectory_json_sha256") != expected_trajectory_sha256:
        raise ValueError(f"terminal cost does not bind selected trajectory: {path}")
    costs = {
        "N2q": _integer(result.get("N2q"), name="N2q"),
        "D2q": _integer(result.get("D2q"), name="D2q"),
        "Dc": _integer(result.get("Dc"), name="Dc"),
    }
    return {
        **costs,
        "path": str(path),
        "sha256": _sha256(path),
        "final_support_layout_sha256": str(result["final_support_layout_sha256"]),
        "final_theta_sha256": str(result["final_theta_sha256"]),
    }


def bind_terminal_costs(
    comparison: Mapping[str, Any],
    *,
    terminal_cost_root: Path,
    include_complete_target: bool = False,
) -> list[dict[str, Any]]:
    """Return same-cell route-best comparisons with verified fresh costs."""

    selected: list[dict[str, Any]] = []
    for raw_cell in comparison.get("cells") or ():
        complete_target = bool(
            raw_cell.get("all_routes_complete")
            and all(
                (raw_cell.get("routes") or {}).get(route, {})
                .get("accuracy", {})
                .get("qualifies_individually")
                is True
                for route in ROUTE_ORDER
            )
        )
        if raw_cell.get("reportable_current_best") is not True and not (
            include_complete_target and complete_target
        ):
            continue
        cell = dict(raw_cell)
        costs: dict[str, Any] = {}
        for route in ROUTE_ORDER:
            if route not in cell["routes"]:
                continue
            route_record = cell["routes"][route]
            trajectory_sha256 = route_record["run_sources"][-1]["sha256"]
            cost_path = terminal_cost_root / str(cell["cell_id"]) / f"{route}.json"
            costs[route] = load_fresh_terminal_cost(
                cost_path,
                expected_trajectory_sha256=str(trajectory_sha256),
            )
        cell["terminal_costs"] = costs
        selected.append(cell)
    if not selected:
        raise ValueError("no reportable same-cell route-best comparisons are ready")

    def priority(cell: Mapping[str, Any]) -> tuple[Any, ...]:
        routes = cell.get("routes") or {}
        all_individual = all(
            record.get("accuracy", {}).get("qualifies_individually") is True
            for record in routes.values()
        )
        worst_burden = max(
            float(record["accuracy"]["accuracy_burden"])
            for record in routes.values()
        )
        return (
            -int(cell.get("all_routes_complete") is True),
            -len(routes),
            -int(all_individual),
            -int(cell.get("available_routes_clique") is True),
            worst_burden,
            str(cell["cell_id"]),
        )

    return sorted(selected, key=priority)


def _sci(value: float) -> str:
    mantissa, exponent = f"{float(value):.3e}".split("e")
    return rf"{mantissa}\!\times\!10^{{{int(exponent)}}}"


def _int(value: int) -> str:
    return f"{int(value):,}"


def _tex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _values(cell: Mapping[str, Any], path: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route in ROUTE_ORDER:
        if route not in cell["routes"]:
            continue
        value: Any = cell["routes"][route]
        for name in path:
            value = value[name]
        result[route] = value
    return result


def _cost_values(cell: Mapping[str, Any], name: str) -> dict[str, int]:
    return {
        route: int(cell["terminal_costs"][route][name])
        for route in ROUTE_ORDER
        if route in cell["terminal_costs"]
    }


def _deletion_values(cell: Mapping[str, Any], name: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for route in ROUTE_ORDER:
        if route not in cell["routes"]:
            continue
        outcomes = cell["routes"][route]["deletion_gate_outcomes"]
        value = outcomes.get(name)
        if value is None:
            result[route] = None
        elif outcomes.get("attempt_record_complete") is True:
            result[route] = _int(int(value))
        else:
            result[route] = rf"$\geq${_int(int(value))}"
    return result


def _row(
    label: str,
    values: Mapping[str, Any],
    *,
    formatter,
    lower_is_better: bool = True,
    highlight_best: bool = True,
) -> str:
    numeric = {
        route: float(value)
        for route, value in values.items()
        if value is not None and isinstance(value, (int, float))
    }
    best = (
        min(numeric.values()) if lower_is_better else max(numeric.values())
    ) if numeric else None
    cells = []
    for route in ROUTE_ORDER:
        if route not in values:
            cells.append(r"\emph{pending}")
            continue
        value = values[route]
        if value is None:
            cells.append(r"---")
            continue
        rendered = formatter(value)
        if (
            highlight_best
            and best is not None
            and route in numeric
            and math.isclose(numeric[route], best, rel_tol=1.0e-12, abs_tol=0.0)
        ):
            rendered = rf"\textbf{{{rendered}}}"
        cells.append(rendered)
    return f"{label} & " + " & ".join(cells) + r" \\"


def _status_row(cell: Mapping[str, Any]) -> str:
    cells = []
    for route in ROUTE_ORDER:
        if route not in cell["routes"]:
            cells.append(r"\emph{pending}")
            continue
        accuracy = cell["routes"][route]["accuracy"]
        cells.append(
            "target"
            if accuracy["qualifies_individually"]
            else "guard"
            if accuracy["comparison_eligible"]
            else "outside"
        )
    return "Accuracy gate & " + " & ".join(cells) + r" \\"


def _cell_table(cell: Mapping[str, Any]) -> str:
    rows = [
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrrr@{}}",
        r"\toprule",
        "Metric & " + " & ".join(DISPLAY_LABELS[route] for route in ROUTE_ORDER) + r" \\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\emph{Average exact errors and achieved accuracy}} \\",
        _row(
            "Mean infidelity",
            _values(cell, ("accuracy", "mean_infidelity")),
            formatter=lambda value: f"${_sci(float(value))}$",
        ),
        _row(
            r"Mean $|\Delta E|$",
            _values(cell, ("accuracy", "mean_abs_energy_error")),
            formatter=lambda value: f"${_sci(float(value))}$",
        ),
        _row(
            "Mean doublon error",
            _values(cell, ("accuracy", "mean_abs_doublon_error")),
            formatter=lambda value: f"${_sci(float(value))}$",
        ),
        _row(
            r"Mean $|\Delta n|$",
            _values(cell, ("accuracy", "mean_abs_charge_imbalance_error")),
            formatter=lambda value: f"${_sci(float(value))}$",
        ),
        _row(
            "Accuracy burden",
            _values(cell, ("accuracy", "accuracy_burden")),
            formatter=lambda value: f"{float(value):.3f}",
        ),
        _status_row(cell),
        r"\addlinespace[2pt]",
        r"\multicolumn{6}{@{}l}{\emph{Propagation and logical estimator work}} \\",
        _row(
            "Accepted timesteps",
            _values(cell, ("time_step_count",)),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            r"Terminal $N_\theta$",
            _values(cell, ("n_theta",)),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            "RHS geometries",
            _values(cell, ("estimator_work", "rhs_geometry_evaluation_count")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            r"Logical geometry work $W_{\rm geom}$",
            _values(cell, ("estimator_work", "logical_integration_geometry_work")),
            formatter=lambda value: _int(int(value)),
        ),
        r"\addlinespace[2pt]",
        r"\multicolumn{6}{@{}l}{\emph{Fresh terminal FakeMarrakesh compilation}} \\",
        _row(r"$N_{2q}$", _cost_values(cell, "N2q"), formatter=lambda value: _int(int(value))),
        _row(r"$d_{2q}$", _cost_values(cell, "D2q"), formatter=lambda value: _int(int(value))),
        _row(r"Total depth $D_c$", _cost_values(cell, "Dc"), formatter=lambda value: _int(int(value))),
        r"\addlinespace[2pt]",
        r"\multicolumn{6}{@{}l}{\emph{Accepted support motion and exact recurrence}} \\",
        _row(
            "Coordinates added",
            _values(cell, ("support_motion", "added_coordinate_count")),
            formatter=lambda value: _int(int(value)),
            highlight_best=False,
        ),
        _row(
            "Coordinates deleted",
            _values(cell, ("support_motion", "deleted_coordinate_count")),
            formatter=lambda value: _int(int(value)),
            highlight_best=False,
        ),
        _row(
            r"Deleted then re-added ($\leq2$ edits)",
            _values(cell, ("support_motion", "short_delete_to_readd_count")),
            formatter=lambda value: _int(int(value)),
            highlight_best=False,
        ),
        _row(
            r"Added then re-deleted ($\leq2$ edits)",
            _values(cell, ("support_motion", "short_add_to_redelete_count")),
            formatter=lambda value: _int(int(value)),
            highlight_best=False,
        ),
        _row(
            "Support-changing checkpoints",
            _values(cell, ("support_motion", "support_changing_checkpoint_count")),
            formatter=lambda value: _int(int(value)),
            highlight_best=False,
        ),
        r"\addlinespace[2pt]",
        r"\multicolumn{6}{@{}l}{\emph{Deletion-bearing finalist outcomes}} \\",
        _row(
            "Finalists tested with deletion",
            _deletion_values(cell, "deletion_bearing_finalist_attempt_count"),
            formatter=str,
            highlight_best=False,
        ),
        _row(
            "Failed post-delete gates",
            _deletion_values(cell, "post_delete_gate_failure_count"),
            formatter=str,
            highlight_best=False,
        ),
        _row(
            "Committed deletion patches",
            _deletion_values(cell, "committed_deletion_patch_count"),
            formatter=str,
            highlight_best=False,
        ),
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    return "\n".join(rows)


def render_latex(
    comparison: Mapping[str, Any],
    selected_cells: Sequence[Mapping[str, Any]],
    *,
    markdown_name: str,
    calibration: Mapping[str, Any] | None = None,
) -> str:
    pages = []
    for index, cell in enumerate(selected_cells):
        achieved = cell["achieved_accuracy_floor"]
        achieved_route = str(achieved["route"])
        missing = tuple(cell.get("missing_routes") or ())
        lead = ""
        if index == 0:
            calibration_note = ""
            if calibration is not None:
                calibration_note = (
                    r"{\footnotesize Calibration gate: \texttt{"
                    + _tex_escape(str(calibration.get("status", "UNKNOWN")))
                    + r"}; this is an analysis-only comparison, not evidence adoption.}"
                    + "\n\n"
                )
            lead = rf"""
\begin{{center}}
{{\LARGE\bfseries Paper II current-best selector comparison}}\\[3pt]
{{\small Five side-by-side routes on locked final $t=20$ cells;
$\Gamma(v^\star)$ is realized and $\Gamma(v)$ is ideal-ranked; generated {_tex_escape(str(comparison['generated_utc'])[:10])}}}
\end{{center}}
\hrule
\vspace{{5pt}}

Each page fixes the drive, controller, physics, and final $t=20$ reporting
grid.  It uses the best currently available final trajectory from each route;
the page is retained at the accuracy achieved by its least accurate available
route rather than being discarded by a common qualification gate.  A pending
entry means that route has not finished that cell.  Bold marks the lowest
available numerical value in each error, work, or circuit-cost row.

{{\footnotesize Agent record and exact provenance:
\href{{run:{_tex_escape(markdown_name)}}}{{\texttt{{{_tex_escape(markdown_name)}}}}}.}}
{calibration_note}
\vspace{{4pt}}
"""
        title = (
            f"{_tex_escape(str(cell['drive']))} --- "
            f"{_tex_escape(str(cell['controller']))} --- "
            f"cut ${float(cell['apm_cut']):.3e}$"
        )
        match_status = (
            "passes"
            if cell.get("available_routes_clique") is True
            else "does not pass"
        )
        pending_note = (
            " Pending: "
            + ", ".join(DISPLAY_LABELS[route] for route in missing)
            + "."
            if missing
            else " All five routes are present."
        )
        match_note = (
            r"\textbf{Accuracy basis:} least accurate available route is "
            + DISPLAY_LABELS[achieved_route]
            + f" (burden {float(achieved['accuracy_burden']):.3f}). "
            + f"The strict factor-two match among available routes {match_status}."
            + pending_note
        )
        page = rf"""{lead}
\section*{{{title}}}
{{\small {match_note}}}

{{\small
{_cell_table(cell)}
}}
\vspace{{4pt}}
{{\footnotesize
$W_{{\rm geom}}=\sum_i r_i[p_i(p_i+1)/2+p_i+2]$ is a logical
integration-geometry estimator proxy, not a physical shot count.  It excludes
candidate acquisition and certification work that the current artifacts do
not ledger completely.  Support motion uses exact coordinate multisets; the
two recurrence rows collapse only insertion-instance labels.  Post-delete
failures count recorded deletion-bearing finalists rejected after
pre-certification resolution; the $\geq$ symbol marks a truncated attempt
ledger.  A dash is not applicable (AVQDS is append-only).  $^*$The old proxy
and AVQDS are verified historical stitched baselines.}}
"""
        pages.append(page.strip())
    body = "\n\n\\newpage\n\n".join(pages)
    return rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,landscape,margin=0.52in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{3pt}}
\setlength{{\tabcolsep}}{{4.5pt}}
\renewcommand{{\arraystretch}}{{1.03}}
\begin{{document}}
{body}
\end{{document}}
"""


def render_markdown(
    comparison: Mapping[str, Any],
    selected_cells: Sequence[Mapping[str, Any]],
    *,
    comparison_path: Path,
    tex_path: Path,
    pdf_path: Path,
    calibration_path: Path | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> str:
    if calibration is not None and calibration_path is not None:
        calibration_lines = [
            f"- Calibration gate: `{calibration.get('status')}`; receipt "
            f"`{calibration_path}` (SHA-256 `{_sha256(calibration_path)}`)."
        ]
        not_checked = tuple(
            str(value) for value in calibration.get("not_checked") or ()
        )
        if not_checked:
            calibration_lines.append(
                "- Calibration required checks not run: "
                + ", ".join(f"`{value}`" for value in not_checked)
                + ". This is no-evidence, not a pass."
            )
    else:
        calibration_lines = ["- Calibration gate: no receipt supplied."]
    lines = [
        "# Paper II current-best selector comparison",
        "",
        "This Markdown file is the agent-facing authority for the same-stem human PDF.",
        "",
        "## Status",
        "",
        f"- Schema: `{REPORT_SCHEMA}`.",
        f"- Included cells: {len(selected_cells)} same-drive, same-controller route-best comparisons.",
        f"- Complete five-route cells: {sum(cell.get('all_routes_complete') is True for cell in selected_cells)}.",
        f"- Available-route strict accuracy cliques: {sum(cell.get('available_routes_clique') is True for cell in selected_cells)}.",
        "- Scope: locked final 501-point t=0..20 cells only; no screens, threshold ladders, smoke runs, or precursor rows.",
        "- Route columns: realized `Gamma(v*)` compiled; ideal `Gamma(v)`-ranked compiled; repaired analytical proxy; historical old proxy; historical AVQDS.",
        "- Internal provenance mapping: `q_compiled` is the manuscript's realized `Gamma(v*)` route; `gamma_compiled` is its ideal `Gamma(v)`-ranked route. The legacy keys remain unchanged so existing actions and receipts retain stable identities.",
        "- A cell is shown once either compiled Gamma route is available alongside both proxy routes and AVQDS. Missing compiled routes remain explicitly pending.",
        f"- Fresh terminal compilation: {sum(len(cell.get('terminal_costs') or {}) for cell in selected_cells)} displayed route/cell arms, all validated under local-only `FakeMarrakesh`, optimization level 1, transpiler seed 7, with terminal-reconstruction parity and trajectory-digest binding.",
        "- Accuracy basis: within the locked cell, retain each route's current best final trajectory and set the achieved comparison level by the least accurate available route; do not discard the page merely because a common qualification gate fails.",
        "- Evidence status: analysis only; no manuscript or source-map adoption.",
        *calibration_lines,
        f"- Comparison JSON: `{comparison_path}` (SHA-256 `{_sha256(comparison_path)}`).",
        f"- LaTeX source: `{tex_path}` (SHA-256 `{_sha256(tex_path)}`)" if tex_path.exists() else f"- LaTeX source: `{tex_path}` (not yet written).",
        f"- Human PDF: `{pdf_path}` (SHA-256 `{_sha256(pdf_path)}`)." if pdf_path.exists() else f"- Human PDF: `{pdf_path}` (not yet built).",
        "",
        "## Measurement and support-decision contract",
        "",
        "`logical_integration_geometry_work = sum_i r_i [p_i(p_i+1)/2 + p_i + 2]`, where `r_i` is the active integration/parameter-control RHS evaluation count and `p_i` is the active runtime support size. It is a logical scalar-estimator proxy, not a hardware shot count. Candidate acquisition, checkpoint base geometry, certification, commuting-group structure, and shot allocation remain outside it because current trajectories do not persist complete ledgers for those quantities.",
        "",
        "Support motion is reconstructed from exact support-label multisets at every checkpoint. Recurrence removes only insertion-instance and append-occurrence suffixes. `deleted_then_readded` and `added_then_redeleted` count inverse events occurring within two later support-changing checkpoints.",
        "",
        "Deletion-bearing finalist counts come from the selector's persisted certification-attempt ledger. `post_delete_gate_failures` excludes insertion-resolution and compiled-word failures that occur before materialization. When a checkpoint's attempt list is truncated, the table marks the affected count as a lower bound. AVQDS is append-only, so deletion rows are not applicable.",
        "",
        "## Included exact values",
    ]
    for cell in selected_cells:
        lines.extend([
            "",
            f"### {cell['cell_id']}",
            "",
            (
                "Achieved-accuracy floor: `"
                + str(cell["achieved_accuracy_floor"]["route"])
                + "` at burden `"
                + f"{float(cell['achieved_accuracy_floor']['accuracy_burden']):.6g}`. "
                + "Missing routes: `"
                + (", ".join(cell.get("missing_routes") or ()) or "none")
                + "`."
            ),
            "",
            "| Metric | Gamma(v*) realized | Gamma(v) ideal-ranked | Repaired proxy | Old proxy | AVQDS |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        metrics = [
            ("mean_infidelity", lambda r: r["accuracy"]["mean_infidelity"]),
            ("mean_abs_energy_error", lambda r: r["accuracy"]["mean_abs_energy_error"]),
            ("mean_abs_doublon_error", lambda r: r["accuracy"]["mean_abs_doublon_error"]),
            ("mean_abs_delta_n_error", lambda r: r["accuracy"]["mean_abs_charge_imbalance_error"]),
            ("accuracy_burden", lambda r: r["accuracy"]["accuracy_burden"]),
            ("accepted_timesteps", lambda r: r["time_step_count"]),
            ("terminal_n_theta", lambda r: r["n_theta"]),
            ("rhs_geometry_evaluations", lambda r: r["estimator_work"]["rhs_geometry_evaluation_count"]),
            ("logical_integration_geometry_work", lambda r: r["estimator_work"]["logical_integration_geometry_work"]),
            ("coordinates_added", lambda r: r["support_motion"]["added_coordinate_count"]),
            ("coordinates_deleted", lambda r: r["support_motion"]["deleted_coordinate_count"]),
            ("deleted_then_readded", lambda r: r["support_motion"]["short_delete_to_readd_count"]),
            ("added_then_redeleted", lambda r: r["support_motion"]["short_add_to_redelete_count"]),
            ("support_changing_checkpoints", lambda r: r["support_motion"]["support_changing_checkpoint_count"]),
            ("deletion_bearing_finalists_tested", lambda r: r["deletion_gate_outcomes"]["deletion_bearing_finalist_attempt_count"]),
            ("post_delete_gate_failures", lambda r: r["deletion_gate_outcomes"]["post_delete_gate_failure_count"]),
            ("committed_deletion_patches", lambda r: r["deletion_gate_outcomes"]["committed_deletion_patch_count"]),
        ]
        for name, getter in metrics:
            values = []
            for route in ROUTE_ORDER:
                if route not in cell["routes"]:
                    values.append("pending")
                    continue
                record = cell["routes"][route]
                value = getter(record)
                if value is None:
                    values.append("n/a")
                elif (
                    name
                    in {
                        "deletion_bearing_finalists_tested",
                        "post_delete_gate_failures",
                    }
                    and record["deletion_gate_outcomes"][
                        "attempt_record_complete"
                    ]
                    is not True
                ):
                    values.append(f">={value}")
                else:
                    values.append(value)
            lines.append("| " + name + " | " + " | ".join(str(value) for value in values) + " |")
        for cost_name in ("N2q", "D2q", "Dc"):
            values = [
                cell["terminal_costs"][route][cost_name]
                if route in cell["terminal_costs"]
                else "pending"
                for route in ROUTE_ORDER
            ]
            lines.append("| terminal_" + cost_name + " | " + " | ".join(str(value) for value in values) + " |")

        lines.extend(["", "Provenance:"])
        for route in ROUTE_ORDER:
            if route not in cell["routes"]:
                lines.append(f"- `{route}`: pending for this cell.")
                continue
            record = cell["routes"][route]
            cost = cell["terminal_costs"][route]
            lines.append(
                f"- `{route}` trajectory `{record['run_sources'][-1]['path']}` "
                f"(SHA-256 `{record['run_sources'][-1]['sha256']}`); fresh terminal cost "
                f"`{cost['path']}` (SHA-256 `{cost['sha256']}`)."
            )
            outcomes = record["deletion_gate_outcomes"]
            if outcomes.get("applicable") is True:
                exactness = (
                    "exact"
                    if outcomes.get("attempt_record_complete") is True
                    else "lower bound"
                )
                reasons = outcomes.get("post_delete_failure_reason_counts") or {}
                reason_text = ", ".join(
                    f"{name}={count}" for name, count in reasons.items()
                ) or "none"
                lines.append(
                    f"  Deletion-attempt ledger: {exactness}; post-delete failure "
                    f"reasons: {reason_text}."
                )
    lines.extend([
        "",
        "## Excluded and pending cells",
        "",
    ])
    selected_ids = {str(cell["cell_id"]) for cell in selected_cells}
    for cell in comparison.get("cells") or ():
        if str(cell["cell_id"]) in selected_ids:
            continue
        missing = ", ".join(cell.get("missing_routes") or ()) or "none"
        lines.append(
            f"- `{cell['cell_id']}`: tier `{cell['comparison_tier']}`; missing routes: {missing}."
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--terminal-cost-root", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--pdf-path", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument(
        "--include-complete-target",
        action="store_true",
        help=(
            "Include complete five-route cells where every route passes the "
            "individual target even if a strict pairwise accuracy match fails."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    comparison_path = args.comparison_json.expanduser().resolve()
    comparison = _read_json(comparison_path)
    if comparison.get("schema") != COMPARISON_SCHEMA:
        raise ValueError("unexpected selector-comparison schema")
    terminal_root = args.terminal_cost_root.expanduser().resolve()
    calibration_path = (
        args.calibration_receipt.expanduser().resolve()
        if args.calibration_receipt is not None
        else None
    )
    calibration = _read_json(calibration_path) if calibration_path is not None else None
    if calibration is not None and str(calibration.get("status", "")).upper() == "ERROR":
        raise ValueError("calibration gate reports ERROR; refusing to build report")
    selected = bind_terminal_costs(
        comparison,
        terminal_cost_root=terminal_root,
        include_complete_target=args.include_complete_target,
    )
    tex_path = args.output_tex.expanduser().resolve()
    md_path = args.output_md.expanduser().resolve()
    pdf_path = args.pdf_path.expanduser().resolve()
    tex = render_latex(
        comparison,
        selected,
        markdown_name=md_path.name,
        calibration=calibration,
    )
    _write_text(tex_path, tex)
    markdown = render_markdown(
        comparison,
        selected,
        comparison_path=comparison_path,
        tex_path=tex_path,
        pdf_path=pdf_path,
        calibration_path=calibration_path,
        calibration=calibration,
    )
    _write_text(md_path, markdown)
    print(json.dumps({"selected_cell_count": len(selected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
