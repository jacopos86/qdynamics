#!/usr/bin/env python3
"""Build HH Qiskit/table marker overlays against current Paper-I trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.reporting.build_paper_i_hh_native200_kpl_marker_plots import (
    METHOD_AUDIT_KEY,
    METHOD_LABEL,
    METHOD_ORDER,
    METHOD_STYLE,
    REGIME_DISPLAY,
    REGIME_ORDER,
    _configure_matplotlib,
    _curve_y_at_k,
    _load_markers,
    _load_rows,
    _trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEM = "paper_i_hh_qiskit_overlay_diagnostic_20260621"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt_sci(value: float | str | None) -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return "--"
    return f"{f:.3e}"


def _fmt_diff(value: float | None) -> str:
    if value is None:
        return "--"
    if abs(value) < 5e-16:
        return "0"
    return f"{value:+.3e}"


def _tex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _latex_path(path: Path) -> str:
    return str(path).replace(os.sep, "/")


def _tex_path(path_text: str) -> str:
    return r"\path{" + path_text.replace(os.sep, "/") + r"}"


def _graphics_path_from_tex(tex_path: Path, repo_relative_path: str) -> str:
    figure_path = (REPO_ROOT / repo_relative_path).resolve()
    try:
        figure_path = figure_path.relative_to(tex_path.parent.resolve())
    except ValueError:
        pass
    return _latex_path(Path(figure_path))


def _qiskit_rows(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    payload = _read_json(path)
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    method_map = {"Append": "Append-ADAPT", "Geo": "Geo-ADAPT", "SNAKE": "SNAKE"}
    for row in payload.get("rows", []):
        regime = str(row.get("regime"))
        method = method_map.get(str(row.get("method")), str(row.get("method")))
        out[(regime, method)] = row
    return out


def _current_plot_markers(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    payload = _read_json(path)
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("marker_audit", []):
        out[(str(row.get("regime")), str(row.get("method")))] = row
    return out


def _render_overlay(
    *,
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    plateau_markers: Mapping[tuple[str, str], Mapping[str, Any]],
    qiskit_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    plot_markers: Mapping[tuple[str, str], Mapping[str, Any]],
    figures_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    Figure, FigureCanvasAgg, Line2D = _configure_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []

    for regime in REGIME_ORDER:
        fig = Figure(figsize=(9.6, 5.35), dpi=160)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        y_values: list[float] = []

        for method in METHOD_ORDER:
            row = rows[(regime, method)]
            qrow = qiskit_rows[(regime, method)]
            marker_row = plateau_markers[(regime, METHOD_AUDIT_KEY[method])]
            points = _trajectory(row)
            xs = [x for x, _ in points]
            ys = [max(y, 1e-12) for _, y in points]
            y_values.extend(ys)
            style = METHOD_STYLE[method]
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linewidth=2.35 if method != "SNAKE" else 2.7,
                alpha=0.96,
                label=f"{METHOD_LABEL[method]} Paper-I curve",
            )

            k_pl = int(round(float(marker_row["k_pl"])))
            current_marker = plot_markers.get((regime, method), {})
            plot_y = current_marker.get("plotted_marker_same_cutoff_abs_delta_e")
            if plot_y is None:
                plot_y, _ = _curve_y_at_k(points, k_pl)
            plot_y = float(plot_y)
            qiskit_y = float(qrow["same_cutoff_abs_delta_e_at_k_pl"])
            y_values.extend([plot_y, qiskit_y])

            ax.plot(
                [k_pl],
                [plot_y],
                color=style["color"],
                marker=style["marker"],
                markersize=style["markersize"],
                markeredgecolor="black",
                markeredgewidth=1.2,
                linestyle="None",
                zorder=6,
            )
            ax.plot(
                [k_pl],
                [qiskit_y],
                color=style["color"],
                marker=style["marker"],
                markersize=style["markersize"] * 1.15,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.45,
                linestyle="None",
                zorder=7,
            )

            display_target = qrow.get("display_target", {})
            replayed_qiskit = qrow.get("replayed_qiskit", {})
            audit_rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "k_pl": k_pl,
                    "plot_marker_same_cutoff_abs_delta_e": plot_y,
                    "qiskit_table_same_cutoff_abs_delta_e": qiskit_y,
                    "qiskit_minus_plot_marker_delta_e": qiskit_y - plot_y,
                    "N2q": display_target.get("N2q"),
                    "D2q": display_target.get("D2q"),
                    "Dc": display_target.get("Dc"),
                    "S": display_target.get("S"),
                    "replayed_N2q": replayed_qiskit.get("N2q"),
                    "replayed_D2q": replayed_qiskit.get("D2q"),
                    "replayed_Dc": replayed_qiskit.get("Dc"),
                    "qiskit_costs_match_display": qrow.get("all_display_costs_match"),
                }
            )

        ax.set_yscale("log")
        ax.set_xlim(0, 30.5)
        ax.set_ylim(max(min(y_values or [1e-4]) * 0.55, 1e-12), 2.0)
        ax.set_xlabel("ADAPT selection round", fontsize=15)
        ax.set_ylabel(r"$|\Delta E|$", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="major", alpha=0.28, linewidth=0.65)
        ax.grid(True, which="minor", alpha=0.13, linewidth=0.45, linestyle=":")
        ax.set_title(f"{REGIME_DISPLAY[regime]}: Paper-I curve with Qiskit/table marker overlay", fontsize=16, pad=8)

        handles: list[Any] = []
        labels: list[str] = []
        for method in METHOD_ORDER:
            style = METHOD_STYLE[method]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=2.35,
                    marker=style["marker"],
                    markersize=style["markersize"] * 0.72,
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                )
            )
            labels.append(f"{METHOD_LABEL[method]}: filled=plot, open=Qiskit/table")
        ax.legend(handles=handles, labels=labels, fontsize=9.5, loc="upper right", framealpha=0.92)
        fig.tight_layout()

        out_path = figures_dir / f"{DEFAULT_STEM}_{regime.replace('-', '_')}.png"
        fig.savefig(out_path)
        outputs[regime] = _rel(out_path)

    return outputs, audit_rows


def _rows_for_regime(audit_rows: Sequence[Mapping[str, Any]], regime: str) -> list[Mapping[str, Any]]:
    by_method = {row["method"]: row for row in audit_rows if row["regime"] == regime}
    return [by_method[method] for method in METHOD_ORDER]


def _write_tex(
    *,
    tex_path: Path,
    figures: Mapping[str, str],
    audit_rows: Sequence[Mapping[str, Any]],
    support_json: Path,
    qiskit_json: Path,
    plot_provenance: Path,
    provenance_json: Path,
) -> None:
    lines: list[str] = []
    lines.extend(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.55in]{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{booktabs}",
            r"\usepackage{array}",
            r"\usepackage{hyperref}",
            r"\usepackage{url}",
            r"\usepackage{xcolor}",
            r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
            r"\Urlmuskip=0mu plus 2mu",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{4pt}",
            r"\begin{document}",
            r"\begin{center}",
            r"{\Large Paper-I HH Qiskit/Table Overlay Diagnostic}\\",
            r"\smallskip",
            r"{\small Solid curves and filled markers reproduce the current Paper-I iteration plots. Open markers use the Qiskit/table row value at the same \(k_{\rm pl}\).}",
            r"\end{center}",
            r"\small",
            r"\begin{tabular}{@{}p{0.26\linewidth}p{0.67\linewidth}@{}}",
            r"\toprule",
            r"Support trajectories & " + _tex_path(_rel(support_json)) + r"\\",
            r"Qiskit table replay & " + _tex_path(_rel(qiskit_json)) + r"\\",
            r"Current plot provenance & " + _tex_path(_rel(plot_provenance)) + r"\\",
            r"Report provenance & " + _tex_path(_rel(provenance_json)) + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\normalsize",
        ]
    )
    for regime in REGIME_ORDER:
        lines.extend(
            [
                r"\clearpage",
                rf"\section*{{{_tex_escape(REGIME_DISPLAY[regime])}}}",
                r"\begin{center}",
                r"\includegraphics[width=0.98\linewidth]{" + _graphics_path_from_tex(tex_path, figures[regime]) + r"}",
                r"\end{center}",
                r"\small",
                r"\begin{center}",
                r"\begin{tabular}{@{}lrrrrrrrr@{}}",
                r"\toprule",
                r"Method & \(k_{\rm pl}\) & plot \(|\Delta E|\) & Qiskit/table \(|\Delta E|\) & table-plot & \(N_{2q}\) & \(D_{2q}\) & \(D_c\) & \(S\)\\",
                r"\midrule",
            ]
        )
        for row in _rows_for_regime(audit_rows, regime):
            lines.append(
                " & ".join(
                    [
                        _tex_escape(METHOD_LABEL[str(row["method"])]),
                        str(row["k_pl"]),
                        _fmt_sci(row["plot_marker_same_cutoff_abs_delta_e"]),
                        _fmt_sci(row["qiskit_table_same_cutoff_abs_delta_e"]),
                        _fmt_diff(float(row["qiskit_minus_plot_marker_delta_e"])),
                        str(row.get("N2q")),
                        str(row.get("D2q")),
                        str(row.get("Dc")),
                        str(row.get("S")),
                    ]
                )
                + r"\\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{center}",
                r"\normalsize",
            ]
        )
    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_provenance(
    *,
    output_path: Path,
    figures: Mapping[str, str],
    audit_rows: Sequence[Mapping[str, Any]],
    support_json: Path,
    qiskit_json: Path,
    plot_provenance: Path,
    tex_path: Path,
    pdf_path: Path,
) -> None:
    payload = {
        "schema": "paper_i_hh_qiskit_overlay_diagnostic_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "HH-only Paper-I current iteration curves with Qiskit/table k_pl marker overlay",
        "support_json": _rel(support_json),
        "support_json_sha256": _sha256(support_json),
        "qiskit_table_replay_json": _rel(qiskit_json),
        "qiskit_table_replay_sha256": _sha256(qiskit_json),
        "plot_provenance_json": _rel(plot_provenance),
        "plot_provenance_sha256": _sha256(plot_provenance),
        "tex_path": _rel(tex_path),
        "pdf_path": _rel(pdf_path),
        "pdf_sha256": _sha256(pdf_path) if pdf_path.exists() else None,
        "figures": {
            regime: {
                "path": path,
                "sha256": _sha256(REPO_ROOT / path),
            }
            for regime, path in figures.items()
        },
        "rows": list(audit_rows),
        "row_count": len(audit_rows),
        "max_abs_qiskit_minus_plot_marker_delta_e": max(
            abs(float(row["qiskit_minus_plot_marker_delta_e"])) for row in audit_rows
        )
        if audit_rows
        else None,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compile_tex(tex_path: Path) -> None:
    cwd = tex_path.parent
    if shutil.which("latexmk"):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    elif shutil.which("tectonic"):
        cmd = ["tectonic", "--keep-logs", "--reruns", "2", tex_path.name]
    else:
        raise RuntimeError("neither latexmk nor tectonic is installed")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def build_report(
    *,
    support_json: Path,
    qiskit_json: Path,
    plot_provenance: Path,
    output_dir: Path,
    compile_pdf: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / f"{DEFAULT_STEM}_figures"
    rows = _load_rows(support_json)
    plot_payload = _read_json(plot_provenance)
    plateau_source = _resolve(plot_payload.get("source_plateau_audit", ""))
    plateau_markers = _load_markers(
        plateau_source
    )
    qrows = _qiskit_rows(qiskit_json)
    plot_markers = _current_plot_markers(plot_provenance)
    figures, audit_rows = _render_overlay(
        rows=rows,
        plateau_markers=plateau_markers,
        qiskit_rows=qrows,
        plot_markers=plot_markers,
        figures_dir=figures_dir,
    )
    tex_path = output_dir / f"{DEFAULT_STEM}.tex"
    pdf_path = output_dir / f"{DEFAULT_STEM}.pdf"
    provenance_path = output_dir / f"{DEFAULT_STEM}.provenance.json"
    _write_tex(
        tex_path=tex_path,
        figures=figures,
        audit_rows=audit_rows,
        support_json=support_json,
        qiskit_json=qiskit_json,
        plot_provenance=plot_provenance,
        provenance_json=provenance_path,
    )
    if compile_pdf:
        _compile_tex(tex_path)
    _write_provenance(
        output_path=provenance_path,
        figures=figures,
        audit_rows=audit_rows,
        support_json=support_json,
        qiskit_json=qiskit_json,
        plot_provenance=plot_provenance,
        tex_path=tex_path,
        pdf_path=pdf_path,
    )
    return {
        "tex": _rel(tex_path),
        "pdf": _rel(pdf_path),
        "provenance": _rel(provenance_path),
        "figures": figures,
        "row_count": len(audit_rows),
        "max_abs_qiskit_minus_plot_marker_delta_e": max(
            abs(float(row["qiskit_minus_plot_marker_delta_e"])) for row in audit_rows
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            This is a diagnostic report only. It does not edit Paper I and does
            not launch benchmark runs.
            """
        ),
    )
    parser.add_argument(
        "--support-json",
        type=Path,
        default=Path("MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_manuscript_update_20260619.json"),
    )
    parser.add_argument(
        "--qiskit-json",
        type=Path,
        default=Path("output/pdf/paper_i_hh_native200_qiskit_table_replay_20260621_v1.json"),
    )
    parser.add_argument(
        "--plot-provenance",
        type=Path,
        default=Path(
            "MATH/paper_facing/paper_I_static_scaffold/preview_plots/"
            "paper_i_hh_native200_full30_kpl_marker_plots_20260619.provenance.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args(argv)
    result = build_report(
        support_json=_resolve(args.support_json),
        qiskit_json=_resolve(args.qiskit_json),
        plot_provenance=_resolve(args.plot_provenance),
        output_dir=_resolve(args.output_dir),
        compile_pdf=not args.no_compile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
