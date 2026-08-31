"""Compact Markdown, PNG, and LaTeX report for qualified staged pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from pipelines.time_dynamics.diagnostics.exact_driven_reference import (
    exact_rows_on_times,
)
from pipelines.time_dynamics.diagnostics.physical_errors import (
    combine_resumed_rows,
)
from pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources import (
    TerminalLogicalGraphResources,
    load_terminal_resources,
)
from pipelines.time_dynamics.run_lock import assert_comparable
from pipelines.time_dynamics.staged_campaign import (
    CampaignLedger,
    METHOD_APM,
    METHOD_AVQDS,
    RungRecord,
    STATUS_ACCURACY_QUALIFIED,
)


REPORT_SCHEMA_V1 = "paper_ii_staged_campaign_report_v1"
METHOD_LABELS = {METHOD_APM: "APM", METHOD_AVQDS: "AVQDS"}
METHOD_COLORS = {METHOD_APM: "#277da1", METHOD_AVQDS: "#f8961e"}


def _display_label(value: Any) -> str:
    return str(value).replace("_", " ")


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError(f"required report artifact is missing or empty: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_run_rows(record: RungRecord) -> tuple[Mapping[str, Any], ...]:
    if record.run_path is None:
        raise ValueError("qualified rung has no run path")
    run = _load_json(record.run_path)
    continuation = tuple(run.get("plot_rows") or ())
    if record.horizon != "t20":
        return continuation
    if record.resume_from_run_path is None:
        raise ValueError("qualified t20 rung has no t5 resume source")
    screen = _load_json(record.resume_from_run_path)
    return combine_resumed_rows(
        tuple(screen.get("plot_rows") or ()),
        continuation,
    )


def _summed_work(record: RungRecord) -> dict[str, int]:
    paths = [record.run_path]
    if record.horizon == "t20":
        paths.insert(0, record.resume_from_run_path)
    summaries = [dict(_load_json(path).get("summary") or {}) for path in paths if path]
    sum_fields = (
        "accepted_internal_substep_count",
        "attempted_internal_step_count",
        "rejected_internal_step_count",
        "rhs_evaluation_count",
        "accepted_insert_count",
        "accepted_delete_count",
        "accepted_exchange_count",
    )
    result = {
        name: int(sum(int(summary.get(name) or 0) for summary in summaries))
        for name in sum_fields
    }
    final = summaries[-1]
    result["runtime_parameter_count_final"] = int(
        final.get("runtime_parameter_count_final") or 0
    )
    return result


def _cost_row(record: RungRecord) -> TerminalLogicalGraphResources:
    if record.run_path is None:
        raise ValueError("qualified rung has no run path")
    return load_terminal_resources(record.run_path)


def _qualified_pairs(
    ledger: CampaignLedger,
    *,
    horizon: str,
) -> tuple[dict[str, Any], ...]:
    by_key: dict[tuple[str, str, str], dict[str, RungRecord]] = {}
    for record in ledger.records:
        if record.horizon != horizon or record.status != STATUS_ACCURACY_QUALIFIED:
            continue
        pair_key = (record.key.regime, record.key.controller, record.key.drive)
        by_key.setdefault(pair_key, {})[record.key.method] = record
    pairs: list[dict[str, Any]] = []
    for (regime, controller, drive), methods in sorted(by_key.items()):
        if METHOD_APM not in methods or METHOD_AVQDS not in methods:
            continue
        records = {method: methods[method] for method in (METHOD_APM, METHOD_AVQDS)}
        runs = {
            method: _load_json(records[method].run_path)
            for method in records
        }
        assert_comparable(tuple(dict(run.get("run_lock") or {}) for run in runs.values()))
        pairs.append(
            {
                "regime": regime,
                "controller": controller,
                "drive": drive,
                "horizon": horizon,
                "records": records,
                "runs": runs,
            }
        )
    return tuple(pairs)


def _state(row: Mapping[str, Any]) -> np.ndarray:
    real = np.asarray(row["psi_real"], dtype=float).reshape(-1)
    imag = np.asarray(row["psi_imag"], dtype=float).reshape(-1)
    return real + 1j * imag


def _infidelity_series(
    rows: Sequence[Mapping[str, Any]],
    exact: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    values: list[float] = []
    for observed, reference in zip(rows, exact):
        left = _state(observed)
        right = np.asarray(reference["statevector"], dtype=complex).reshape(-1)
        fidelity = float(
            abs(np.vdot(left, right)) ** 2
            / (float(np.real(np.vdot(left, left))) * float(np.real(np.vdot(right, right))))
        )
        values.append(float(np.clip(1.0 - fidelity, 0.0, 1.0)))
    return np.asarray(values, dtype=float)


def _plot_pair(
    pair: Mapping[str, Any],
    *,
    seed_path: Path,
    output_path: Path,
    reference_substeps: int,
) -> None:
    records = pair["records"]
    rows = {
        method: _combined_run_rows(records[method])
        for method in (METHOD_APM, METHOD_AVQDS)
    }
    times = np.asarray([float(row["time"]) for row in rows[METHOD_APM]], dtype=float)
    for method in rows:
        method_times = np.asarray([float(row["time"]) for row in rows[method]], dtype=float)
        if method_times.shape != times.shape or not np.allclose(
            method_times, times, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("paired report trajectories do not share one time grid")
    reference = exact_rows_on_times(
        seed_path,
        records[METHOD_APM].run_path,
        times=times,
        substeps=int(reference_substeps),
    )

    figure, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), constrained_layout=True)
    panels = (
        ("energy_expectation", "energy", "Energy"),
        ("doublon", "doublon", "Doublon occupation"),
        ("primary_density", "primary_density", "Primary-site density"),
    )
    for axis, (observed_key, exact_key, label) in zip(axes[0], panels):
        axis.plot(
            times,
            [float(row[exact_key]) for row in reference],
            color="#333333",
            linestyle="--",
            linewidth=1.4,
            label="exact",
        )
        for method in (METHOD_APM, METHOD_AVQDS):
            axis.plot(
                times,
                [float(row[observed_key]) for row in rows[method]],
                color=METHOD_COLORS[method],
                linewidth=1.1,
                label=METHOD_LABELS[method],
            )
        axis.set_ylabel(label)

    site_count = len(reference[0]["site_occupations"])
    for site_index, axis in enumerate(axes[1, :2]):
        if site_index >= site_count:
            axis.set_visible(False)
            continue
        axis.plot(
            times,
            [float(row["site_occupations"][site_index]) for row in reference],
            color="#333333",
            linestyle="--",
            linewidth=1.4,
        )
        for method in (METHOD_APM, METHOD_AVQDS):
            axis.plot(
                times,
                [
                    float(row["site_occupations"][site_index])
                    for row in rows[method]
                ],
                color=METHOD_COLORS[method],
                linewidth=1.1,
            )
        axis.set_ylabel(f"Site {site_index} occupation")

    infidelity_axis = axes[1, 2]
    for method in (METHOD_APM, METHOD_AVQDS):
        infidelity_axis.plot(
            times,
            np.maximum(
                _infidelity_series(rows[method], reference),
                np.finfo(float).eps,
            ),
            color=METHOD_COLORS[method],
            linewidth=1.1,
            label=METHOD_LABELS[method],
        )
    infidelity_axis.set_yscale("log")
    infidelity_axis.set_ylabel("Infidelity")
    for axis in axes.flat:
        if axis.get_visible():
            axis.set_xlabel("time")
            axis.grid(True, alpha=0.2)
    axes[0, 0].legend(loc="best", frameon=False, fontsize=8)
    figure.suptitle(
        f"{_display_label(pair['drive'])} - "
        f"{_display_label(pair['controller'])} - {pair['horizon']}",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def _format_metrics_table(records: Mapping[str, RungRecord]) -> list[str]:
    lines = [
        "| method | cut | mean |ΔE| | mean |Δdoublon| | mean maxᵢ|Δnᵢ| | mean infidelity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in (METHOD_AVQDS, METHOD_APM):
        record = records[method]
        metrics = record.metrics
        assert metrics is not None
        lines.append(
            "| {label} | {cut:.3e} | {energy:.3e} | {doublon:.3e} | "
            "{site:.3e} | {infidelity:.3e} |".format(
                label=METHOD_LABELS[method],
                cut=record.threshold,
                energy=metrics.mean_abs_energy_error,
                doublon=metrics.mean_abs_doublon_error,
                site=metrics.mean_max_abs_site_occupation_error,
                infidelity=metrics.mean_infidelity,
            )
        )
    return lines


def _ledger_overview_rows(
    ledger: CampaignLedger,
    *,
    horizon: str,
) -> list[list[str]]:
    rows = [[
        "method",
        "drive",
        "controller",
        "horizon",
        "cut",
        "status",
        "mean |dE|",
        "mean |dNd|",
        "mean max |dni|",
        "mean infidelity",
    ]]
    for record in sorted(
        (row for row in ledger.records if row.horizon == horizon),
        key=lambda row: (
            row.key.drive,
            row.key.controller,
            row.key.method,
            row.threshold,
        ),
    ):
        metrics = record.metrics
        rows.append([
            METHOD_LABELS.get(record.key.method, record.key.method),
            record.key.drive,
            record.key.controller,
            record.horizon,
            f"{record.threshold:.3e}",
            record.status.replace("_", " "),
            "--" if metrics is None else f"{metrics.mean_abs_energy_error:.3e}",
            "--" if metrics is None else f"{metrics.mean_abs_doublon_error:.3e}",
            (
                "--"
                if metrics is None
                else f"{metrics.mean_max_abs_site_occupation_error:.3e}"
            ),
            (
                "--"
                if metrics is None or metrics.mean_infidelity is None
                else f"{metrics.mean_infidelity:.3e}"
            ),
        ])
    return rows


def _format_ledger_overview_markdown(
    ledger: CampaignLedger,
    *,
    horizon: str,
) -> list[str]:
    rows = _ledger_overview_rows(ledger, horizon=horizon)
    lines = [
        "## Campaign ledger overview",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    if len(rows) == 1:
        lines.append("| -- | -- | -- | -- | -- | no recorded rows | -- | -- | -- | -- |")
    lines.append("")
    return lines


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = (
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
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _latex_pair_section(pair: Mapping[str, Any], *, horizon: str) -> str:
    records = pair["records"]
    metric_rows: list[str] = []
    resource_rows: list[str] = []
    for method in (METHOD_APM, METHOD_AVQDS):
        record = records[method]
        metrics = record.metrics
        assert metrics is not None and metrics.mean_infidelity is not None
        metric_rows.append(
            "{} & {:.6e} & {:.6e} & {:.6e} & {:.6e} & {:.6e} \\\\".format(
                METHOD_LABELS[method],
                record.threshold,
                metrics.mean_abs_energy_error,
                metrics.mean_abs_doublon_error,
                metrics.mean_max_abs_site_occupation_error,
                metrics.mean_infidelity,
            )
        )
        work = _summed_work(record)
        cost = _cost_row(record)
        absolute_text = "({:.0f}, {:.0f}, {:.0f}, {:.0f})".format(
            cost.graph[0], cost.graph[1], cost.graph[2], cost.n_theta
        )
        delta_text = (
            "--"
            if cost.delta_graph is None or cost.delta_n_theta is None
            else "({:+.0f}, {:+.0f}, {:+.0f}, {:+.0f})".format(
                cost.delta_graph[0],
                cost.delta_graph[1],
                cost.delta_graph[2],
                cost.delta_n_theta,
            )
        )
        resource_rows.append(
            "{} & {} & {} & {}/{} & {} & {}/{}/{} \\\\".format(
                METHOD_LABELS[method],
                absolute_text,
                delta_text,
                work["accepted_internal_substep_count"],
                work["attempted_internal_step_count"],
                work["rhs_evaluation_count"],
                work["accepted_insert_count"],
                work["accepted_delete_count"],
                work["accepted_exchange_count"],
            )
        )
    slug = f"{pair['drive']}__{pair['controller']}__{horizon}"
    title = _tex_escape(
        f"{_display_label(pair['drive'])} - "
        f"{_display_label(pair['controller'])} - {horizon}"
    )
    return rf"""
\section*{{{title}}}
\begin{{center}}
\includegraphics[width=0.90\textwidth,height=0.45\textheight,keepaspectratio]{{plots/{slug}.png}}
\end{{center}}
\vspace{{-0.5em}}
\textbf{{Physical errors against the exact diagnostic trajectory.}}
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Method & {{$\tau_{{L^2}}$}} & {{$\langle|\Delta E|\rangle$}} & {{$\langle|\Delta n_d|\rangle$}} & {{$\langle\max_i|\Delta n_i|\rangle$}} & {{mean infidelity}} \\
\midrule
{chr(10).join(metric_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
\textbf{{Terminal resources and integration work.}}
\begin{{center}}
\scriptsize
\resizebox{{0.99\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Method & {{$\widehat C_{{2q}},\widehat C_D,\widehat C_{{1q}},N_\theta$}} & {{$\Delta\widehat C_{{2q}},\Delta\widehat C_D,\Delta\widehat C_{{1q}},\Delta N_\theta$}} & steps acc./att. & RHS & ins./del./exch. \\
\midrule
{chr(10).join(resource_rows)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
"""


def _latex_source(
    *,
    ledger: CampaignLedger,
    pairs: Sequence[Mapping[str, Any]],
    horizon: str,
    source_ledger_path: str | Path | None,
    cluster_ids: Sequence[int],
    audit_status: str,
) -> str:
    source = "not provided" if source_ledger_path is None else str(source_ledger_path)
    clusters = "not provided" if not cluster_ids else ", ".join(str(value) for value in cluster_ids)
    sections = "\n\\clearpage\n".join(
        _latex_pair_section(pair, horizon=horizon) for pair in pairs
    )
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
{{\LARGE\bfseries Paper II matched-accuracy dynamics results}}
\end{{center}}
\vspace{{-0.6em}}
\footnotesize
\textbf{{Source ledger:}} \path{{{source}}}\\
\textbf{{CHTC clusters:}} {_tex_escape(clusters)}\quad
\textbf{{Horizon:}} {_tex_escape(horizon)}\quad
\textbf{{Qualified pairs:}} {len(pairs)}\quad
\textbf{{Ledger rows:}} {len(ledger.records)}\quad
\textbf{{Audit:}} {_tex_escape(audit_status)}
\vspace{{0.2em}}
\par Physical errors are post-run diagnostics against an exact reference and were not supplied to either online controller. Costs are reported without an automatic method judgment.
\normalsize
{sections}
\end{{document}}
"""


def _compile_latex(tex_path: Path) -> tuple[Path, str]:
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
    if shutil.which("tectonic"):
        candidates.append(
            (
                "tectonic",
                ["tectonic", "--outdir", build_dir.name, tex_path.name],
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
        raise RuntimeError("no LaTeX engine is available")

    failures: list[str] = []
    for engine, command, passes in candidates:
        completed = None
        for _ in range(passes):
            try:
                completed = subprocess.run(
                    command,
                    cwd=tex_path.parent,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
            except subprocess.TimeoutExpired as exc:
                failures.append(f"{engine}: timed out after {exc.timeout} seconds")
                completed = None
                break
            if completed.returncode != 0:
                break
        if completed is None:
            continue
        built = build_dir / f"{tex_path.stem}.pdf"
        if completed.returncode == 0 and built.is_file() and built.stat().st_size > 0:
            final = tex_path.with_suffix(".pdf")
            shutil.copy2(built, final)
            return final, engine
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-35:])
        failures.append(f"{engine}: {tail}")
    raise RuntimeError("LaTeX report build failed:\n" + "\n\n".join(failures))


def _format_resource_table(records: Mapping[str, RungRecord]) -> list[str]:
    lines = [
        "| method | (Ĉ₂q, ĈD, Ĉ₁q, Nθ) | Δ(Ĉ₂q, ĈD, Ĉ₁q, Nθ) | "
        "steps accepted/attempted | RHS | insert/delete/exchange |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in (METHOD_AVQDS, METHOD_APM):
        record = records[method]
        work = _summed_work(record)
        cost = _cost_row(record)
        absolute = (*cost.graph, cost.n_theta)
        delta_text = (
            "--"
            if cost.delta_graph is None or cost.delta_n_theta is None
            else "({:+.0f}, {:+.0f}, {:+.0f}, {:+.0f})".format(
                *cost.delta_graph, cost.delta_n_theta
            )
        )
        lines.append(
            "| {label} | ({a0:.0f}, {a1:.0f}, {a2:.0f}, {a3:.0f}) | "
            "{delta} | {accepted}/{attempted} | "
            "{rhs} | {insert}/{delete}/{exchange} |".format(
                label=METHOD_LABELS[method],
                a0=absolute[0], a1=absolute[1], a2=absolute[2], a3=absolute[3],
                delta=delta_text,
                accepted=work["accepted_internal_substep_count"],
                attempted=work["attempted_internal_step_count"],
                rhs=work["rhs_evaluation_count"],
                insert=work["accepted_insert_count"],
                delete=work["accepted_delete_count"],
                exchange=work["accepted_exchange_count"],
            )
        )
    return lines


def write_report(
    *,
    ledger: CampaignLedger,
    seed_path: Path,
    output_dir: Path,
    horizon: str = "t20",
    reference_substeps: int = 40,
    source_ledger_path: str | Path | None = None,
    cluster_ids: Sequence[int] = (),
    audit_status: str = "NOT_PROVIDED",
) -> dict[str, Any]:
    pairs = _qualified_pairs(ledger, horizon=horizon)
    if not pairs:
        raise ValueError(
            "results report requires at least one complete accuracy-qualified "
            "APM/AVQDS pair; use the campaign ledger for incomplete status"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "plots"
    lines = [
        "# Paper II staged accuracy-qualified pairs",
        "",
        (
            f"This report contains {len(pairs)} complete APM/AVQDS pairs at "
            f"`{horizon}`. Every displayed arm independently satisfies the "
            "declared componentwise accuracy ceiling. Cost is reported without "
            "an automatic interpretation."
        ),
        "",
        *_format_ledger_overview_markdown(ledger, horizon=horizon),
    ]
    pair_rows: list[dict[str, Any]] = []
    for pair in pairs:
        slug = f"{pair['drive']}__{pair['controller']}__{horizon}"
        image_path = image_dir / f"{slug}.png"
        _plot_pair(
            pair,
            seed_path=seed_path,
            output_path=image_path,
            reference_substeps=int(reference_substeps),
        )
        lines.extend(
            (
                f"## {pair['drive']} · {pair['controller']}",
                "",
                f"![Observable trajectories](plots/{image_path.name})",
                "",
                *_format_metrics_table(pair["records"]),
                "",
                *_format_resource_table(pair["records"]),
                "",
            )
        )
        pair_rows.append(
            {
                "regime": pair["regime"],
                "drive": pair["drive"],
                "controller": pair["controller"],
                "horizon": pair["horizon"],
                "image": str(image_path),
            }
        )
    markdown = output_dir / "paper_ii_staged_report.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    tex = output_dir / "paper_ii_staged_report.tex"
    tex.write_text(
        _latex_source(
            ledger=ledger,
            pairs=pairs,
            horizon=horizon,
            source_ledger_path=source_ledger_path,
            cluster_ids=cluster_ids,
            audit_status=str(audit_status),
        ),
        encoding="utf-8",
    )
    pdf, pdf_engine = _compile_latex(tex)
    if not pdf.is_file() or pdf.stat().st_size <= 0:
        raise RuntimeError("LaTeX compilation returned without a report PDF")
    manifest = {
        "schema": REPORT_SCHEMA_V1,
        "horizon": horizon,
        "qualified_pair_count": len(pairs),
        "ledger_record_count": len(ledger.records),
        "markdown": str(markdown),
        "tex": str(tex),
        "tex_sha256": _sha256(tex),
        "pdf": str(pdf),
        "pdf_engine": pdf_engine,
        "pdf_sha256": _sha256(pdf),
        "source_ledger": (
            None if source_ledger_path is None else str(source_ledger_path)
        ),
        "cluster_ids": [int(value) for value in cluster_ids],
        "audit_status": str(audit_status),
        "pairs": pair_rows,
    }
    (output_dir / "paper_ii_staged_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--seed-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", choices=("t5", "t20"), default="t20")
    parser.add_argument("--reference-substeps", type=int, default=40)
    parser.add_argument("--source-ledger-path", type=Path)
    parser.add_argument("--cluster-id", action="append", type=int, default=[])
    parser.add_argument("--audit-status", default="NOT_PROVIDED")
    args = parser.parse_args(argv)
    manifest = write_report(
        ledger=CampaignLedger.read(args.ledger),
        seed_path=args.seed_json,
        output_dir=args.output_dir,
        horizon=str(args.horizon),
        reference_substeps=int(args.reference_substeps),
        source_ledger_path=(
            args.ledger if args.source_ledger_path is None else args.source_ledger_path
        ),
        cluster_ids=tuple(args.cluster_id),
        audit_status=str(args.audit_status),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA_V1", "main", "write_report"]
