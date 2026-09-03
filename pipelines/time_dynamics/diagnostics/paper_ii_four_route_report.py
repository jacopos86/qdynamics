#!/usr/bin/env python3
"""Render the matched Paper-II four-route decision report as LaTeX and Markdown."""

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


REPORT_SCHEMA = "paper_ii_four_route_human_report_v1"
DISPLAY_LABELS = {
    "q_compiled": r"$Q$ compiled",
    "gamma_compiled": r"support $\Gamma$",
    "repaired_proxy": "repaired proxy",
    "old_proxy": "old proxy$^{*}$",
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
    """Return reportable four-route cells with four verified fresh costs."""

    selected: list[dict[str, Any]] = []
    for raw_cell in comparison.get("cells") or ():
        complete_target = bool(
            raw_cell.get("four_way_complete")
            and all(
                (raw_cell.get("routes") or {}).get(route, {})
                .get("accuracy", {})
                .get("qualifies_individually")
                is True
                for route in ROUTE_ORDER
            )
        )
        if raw_cell.get("comparison_tier") != "individual_target" and not (
            include_complete_target and complete_target
        ):
            continue
        cell = dict(raw_cell)
        costs: dict[str, Any] = {}
        for route in ROUTE_ORDER:
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
        raise ValueError(
            "no reportable complete four-way cells are ready for the human report"
        )
    return selected


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
        value: Any = cell["routes"][route]
        for name in path:
            value = value[name]
        result[route] = value
    return result


def _cost_values(cell: Mapping[str, Any], name: str) -> dict[str, int]:
    return {route: int(cell["terminal_costs"][route][name]) for route in ROUTE_ORDER}


def _row(
    label: str,
    values: Mapping[str, Any],
    *,
    formatter,
    lower_is_better: bool = True,
) -> str:
    numeric = {route: float(values[route]) for route in ROUTE_ORDER}
    best = min(numeric.values()) if lower_is_better else max(numeric.values())
    cells = []
    for route in ROUTE_ORDER:
        rendered = formatter(values[route])
        if math.isclose(numeric[route], best, rel_tol=1.0e-12, abs_tol=0.0):
            rendered = rf"\textbf{{{rendered}}}"
        cells.append(rendered)
    return f"{label} & " + " & ".join(cells) + r" \\"


def _status_row(cell: Mapping[str, Any]) -> str:
    cells = []
    for route in ROUTE_ORDER:
        accuracy = cell["routes"][route]["accuracy"]
        cells.append("target" if accuracy["qualifies_individually"] else "guard")
    return "Accuracy gate & " + " & ".join(cells) + r" \\"


def _cell_table(cell: Mapping[str, Any]) -> str:
    rows = [
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrr@{}}",
        r"\toprule",
        "Metric & " + " & ".join(DISPLAY_LABELS[route] for route in ROUTE_ORDER) + r" \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Average exact errors and matched-accuracy status}} \\",
        _row(
            "Infidelity",
            _values(cell, ("accuracy", "mean_infidelity")),
            formatter=lambda value: f"${_sci(float(value))}$",
        ),
        _row(
            r"$|\Delta E|$",
            _values(cell, ("accuracy", "mean_abs_energy_error")),
            formatter=lambda value: f"${_sci(float(value))}$",
        ),
        _row(
            "Doublon error",
            _values(cell, ("accuracy", "mean_abs_doublon_error")),
            formatter=lambda value: f"${_sci(float(value))}$",
        ),
        _row(
            r"$|\Delta n|$",
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
        r"\multicolumn{5}{@{}l}{\emph{Propagation and logical estimator work}} \\",
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
        r"\multicolumn{5}{@{}l}{\emph{Fresh terminal FakeMarrakesh compilation}} \\",
        _row(r"$N_{2q}$", _cost_values(cell, "N2q"), formatter=lambda value: _int(int(value))),
        _row(r"$d_{2q}$", _cost_values(cell, "D2q"), formatter=lambda value: _int(int(value))),
        _row(r"Total depth $D_c$", _cost_values(cell, "Dc"), formatter=lambda value: _int(int(value))),
        r"\addlinespace[2pt]",
        r"\multicolumn{5}{@{}l}{\emph{Ansatz support motion}} \\",
        _row(
            "Coordinates added",
            _values(cell, ("structural_churn", "added_coordinate_count")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            "Coordinates deleted",
            _values(cell, ("structural_churn", "deleted_coordinate_count")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            "Gross churn",
            _values(cell, ("structural_churn", "gross_churn")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            "Excess churn",
            _values(cell, ("structural_churn", "excess_churn")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            r"Delete $\rightarrow$ re-add ($\leq2$ changes)",
            _values(cell, ("structural_churn", "short_delete_to_readd_count")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            r"Add $\rightarrow$ re-delete ($\leq2$ changes)",
            _values(cell, ("structural_churn", "short_add_to_redelete_count")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            "Support-changing checkpoints",
            _values(cell, ("structural_churn", "support_changing_checkpoint_count")),
            formatter=lambda value: _int(int(value)),
        ),
        _row(
            r"Scored but not selected ($\geq$)",
            _values(
                cell,
                ("structural_churn", "persisted_scored_not_selected_lower_bound"),
            ),
            formatter=lambda value: _int(int(value)),
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
        strict_match = cell.get("comparison_tier") == "individual_target"
        failing_pairs = [
            (name, result)
            for name, result in (cell.get("pairwise_accuracy") or {}).items()
            if result.get("passes") is not True
        ]
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
{{\LARGE\bfseries Paper II four-route selector comparison}}\\[3pt]
{{\small Current complete final $t=20$ AP-McLachlan trajectories; generated {_tex_escape(str(comparison['generated_utc'])[:10])}}}
\end{{center}}
\hrule
\vspace{{5pt}}

Only same-drive, same-controller cells for which all four routes meet the
individual response-normalized accuracy target are shown.  A cell is marked
strictly matched only when every pair also passes the declared factor-of-two
burden and component tests.  Bold marks the lowest value in each row; no
scalar winner is imposed across accuracy, terminal circuit burden, estimator
work, and structural stability.

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
        if strict_match:
            match_note = r"\textbf{Accuracy comparison:} strict four-way match."
        else:
            details = "; ".join(
                rf"{_tex_escape(name.replace('__', ' vs. '))}: burden ratio "
                rf"{float(result['burden_ratio']):.2f}"
                for name, result in failing_pairs
            ) or "strict pairwise match not established"
            match_note = (
                r"\textbf{Accuracy comparison:} all four routes pass individually; "
                r"strict four-way match fails (" + details + ")."
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
not ledger completely.  Churn uses exact coordinate multisets; short cycles
collapse only insertion-instance labels.  $^*$The old proxy is the verified
historical stitched baseline and used the legacy unit cost factor.  The
requested ``$S$'' is the score shared by the compiled routes; the distinct
support route is the $\Gamma$ column.}}
"""
        pages.append(page.strip())
    body = "\n\n\\newpage\n\n".join(pages)
    return rf"""\documentclass[9pt]{{article}}
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
\setlength{{\tabcolsep}}{{6pt}}
\renewcommand{{\arraystretch}}{{1.06}}
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
    lines = [
        "# Paper II four-route selector comparison",
        "",
        "This Markdown file is the agent-facing authority for the same-stem human PDF.",
        "",
        "## Status",
        "",
        f"- Schema: `{REPORT_SCHEMA}`.",
        f"- Included cells: {len(selected_cells)} complete, individually qualified four-route cells.",
        f"- Strict four-way matched cells: {sum(cell.get('comparison_tier') == 'individual_target' for cell in selected_cells)}.",
        f"- Complete individually qualified cells shown provisionally: {sum(cell.get('comparison_tier') != 'individual_target' for cell in selected_cells)}.",
        "- Scope: locked final 501-point t=0..20 cells only; no screens, threshold ladders, smoke runs, or precursor rows.",
        "- Route columns: realized-Q compiled; ideal-support Gamma compiled; repaired analytical proxy; historical old proxy.",
        "- `S` is a shared ranking score, not a separately executed route.",
        "- Evidence status: analysis only; no manuscript or source-map adoption.",
        (
            f"- Calibration gate: `{calibration.get('status')}`; receipt "
            f"`{calibration_path}` (SHA-256 `{_sha256(calibration_path)}`)."
            if calibration is not None and calibration_path is not None
            else "- Calibration gate: no receipt supplied."
        ),
        f"- Comparison JSON: `{comparison_path}` (SHA-256 `{_sha256(comparison_path)}`).",
        f"- LaTeX source: `{tex_path}` (SHA-256 `{_sha256(tex_path)}`)" if tex_path.exists() else f"- LaTeX source: `{tex_path}` (not yet written).",
        f"- Human PDF: `{pdf_path}` (SHA-256 `{_sha256(pdf_path)}`)." if pdf_path.exists() else f"- Human PDF: `{pdf_path}` (not yet built).",
        "",
        "## Measurement and churn contract",
        "",
        "`logical_integration_geometry_work = sum_i r_i [p_i(p_i+1)/2 + p_i + 2]`, where `r_i` is the active integration/parameter-control RHS evaluation count and `p_i` is the active runtime support size. It is a logical scalar-estimator proxy, not a hardware shot count. Candidate acquisition, checkpoint base geometry, certification, commuting-group structure, and shot allocation remain outside it because current trajectories do not persist complete ledgers for those quantities.",
        "",
        "Churn is reconstructed from exact support-label multisets at every checkpoint. Recurrence removes only insertion-instance and append-occurrence suffixes. `delete->readd` and `add->redelete` count inverse events occurring within two later support-changing checkpoints. Persisted candidate/scored totals are lower bounds when a checkpoint executed multiple insertion rounds.",
        "",
        "## Included exact values",
    ]
    for cell in selected_cells:
        lines.extend([
            "",
            f"### {cell['cell_id']}",
            "",
            "| Metric | Q compiled | Gamma support | Repaired proxy | Old proxy |",
            "|---|---:|---:|---:|---:|",
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
            ("coordinates_added", lambda r: r["structural_churn"]["added_coordinate_count"]),
            ("coordinates_deleted", lambda r: r["structural_churn"]["deleted_coordinate_count"]),
            ("gross_churn", lambda r: r["structural_churn"]["gross_churn"]),
            ("excess_churn", lambda r: r["structural_churn"]["excess_churn"]),
            ("short_delete_to_readd", lambda r: r["structural_churn"]["short_delete_to_readd_count"]),
            ("short_add_to_redelete", lambda r: r["structural_churn"]["short_add_to_redelete_count"]),
            ("support_changing_checkpoints", lambda r: r["structural_churn"]["support_changing_checkpoint_count"]),
            ("scored_not_selected_lower_bound", lambda r: r["structural_churn"]["persisted_scored_not_selected_lower_bound"]),
        ]
        for name, getter in metrics:
            values = [getter(cell["routes"][route]) for route in ROUTE_ORDER]
            lines.append("| " + name + " | " + " | ".join(str(value) for value in values) + " |")
        for cost_name in ("N2q", "D2q", "Dc"):
            values = [cell["terminal_costs"][route][cost_name] for route in ROUTE_ORDER]
            lines.append("| terminal_" + cost_name + " | " + " | ".join(str(value) for value in values) + " |")

        lines.extend(["", "Provenance:"])
        for route in ROUTE_ORDER:
            record = cell["routes"][route]
            cost = cell["terminal_costs"][route]
            lines.append(
                f"- `{route}` trajectory `{record['run_sources'][-1]['path']}` "
                f"(SHA-256 `{record['run_sources'][-1]['sha256']}`); fresh terminal cost "
                f"`{cost['path']}` (SHA-256 `{cost['sha256']}`)."
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
            "Include complete four-route cells where every route passes the "
            "individual target even if a strict pairwise accuracy match fails."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    comparison_path = args.comparison_json.expanduser().resolve()
    comparison = _read_json(comparison_path)
    if comparison.get("schema") != COMPARISON_SCHEMA:
        raise ValueError("unexpected four-route comparison schema")
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
