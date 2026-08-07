#!/usr/bin/env python3
"""Fill authenticated RA-singleton prefix costs and reformat report pages 6--7.

The historical-average round-69 diagnostic remains accuracy-only because its
post-run EXDEV failure lost the estimator ledger.  The current global-singleton
comparison is kept on one page: six accuracy panels above their authenticated
cost table.  For the three nonterminal ``nph=7`` rows, the table compiles the
last authenticated checkpoint prefix and keeps later stdout-only energy points
explicitly separate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    add_paper_i_historical_mean_global_singleton_full6_page as completed,
)
from pipelines.reporting import (  # noqa: E402
    add_paper_i_historical_mean_global_singleton_live_page7 as live,
)
from pipelines.reporting import (  # noqa: E402
    build_paper_i_ra_adapt_stationary_core_master_pdf as master,
)


REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
MASTER_PDF = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress.pdf"
)
MASTER_PROVENANCE = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_provenance.json"
)
PAGE6_ADAPTER = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "ra_append_singleton_r70_page6_adapter.json"
)
PAGE6_COSTS = REPORT_DIR / "paper_i_ra_append_singleton_r70_prefix_costs_v1.json"
PAGE6_PLOT = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "ra_append_singleton_r70_page6_plot.pdf"
)
PAGE7_SOURCE_ADAPTER = REPORT_DIR / (
    "historical_mean_global_singleton_live_page7_"
    "k61_49_38_r70_early_stop_stdout_20260807_v2_adapter.json"
)
OUTPUT_STEM = "historical_mean_global_singleton_live_page7_cost_filled_20260807_v3"
OUTPUT_ADAPTER = REPORT_DIR / f"{OUTPUT_STEM}_adapter.json"
OUTPUT_COST_SIDECAR = REPORT_DIR / f"{OUTPUT_STEM}_authenticated_prefix_costs.json"
PAGE6_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_page6_accuracy_only_reformatted"
)
PAGE7_STEM = f"{OUTPUT_STEM}_accuracy_page"

EXPECTED_MASTER_SHA256 = (
    "b48da6ebf40dae8f5bba3f10cc33e198bb014fd6160d7a97ae3eb7392a2fed9a"
)
EXPECTED_PAGE7_ADAPTER_SHA256 = (
    "6e80a18b05a29c449eba49638a557f75966e4b00d391fe594723b090d47cedb0"
)
LIVE_REGIMES = ("weak_strong", "intermediate_strong", "strong_strong_u8")
PLOT_FLOOR = 1.0e-16
COST_FIELDS = ("N2q", "D2q", "Dc", "W1q", "S_alg")


class ReportUpdateError(ValueError):
    """The report cannot be updated without weakening its evidence boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReportUpdateError(f"unsafe or missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportUpdateError(f"JSON input is not an object: {path}")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(live.canonical_json_bytes(unsigned)).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ReportUpdateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportUpdateError(f"{label} is not finite")
    return result


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ReportUpdateError(f"{label} is not an integer")
    result = int(value)
    if result < minimum or result != value:
        raise ReportUpdateError(f"{label} is outside its integer range")
    return result


def _path_from_binding(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise ReportUpdateError(f"{label} binding is unavailable")
    path_text = str(value["path"]).split("#", 1)[0]
    candidate = Path(path_text)
    path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    if not path.is_file() or path.is_symlink():
        raise ReportUpdateError(f"{label} is unavailable or unsafe: {path}")
    return path.resolve()


def _latex_path(path: Path) -> str:
    return completed.latex_escape(path.resolve().as_posix())


def _latex_text(value: Any) -> str:
    return completed.latex_escape(str(value))


def _sci(value: float) -> str:
    if value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.2f}\\mathord{{\\times}}10^{{{exponent}}}$"


def _cost4(costs: Mapping[str, Any]) -> str:
    return "(" + ", ".join(f"{int(costs[field]):,}" for field in COST_FIELDS[:4]) + ")"


def _cost5(costs: Mapping[str, Any]) -> str:
    return "(" + ", ".join(f"{int(costs[field]):,}" for field in COST_FIELDS) + ")"


def _point_at(points: Sequence[Any], round_index: int, *, label: str) -> Mapping[str, Any]:
    matches = [
        point
        for point in points
        if isinstance(point, Mapping) and point.get("round") == round_index
    ]
    if len(matches) != 1:
        raise ReportUpdateError(f"{label} round {round_index} is not unique")
    return matches[0]


def _extract_active_prefix(
    archive_path: Path,
    *,
    checkpoint_member: str,
    controller_round: int,
) -> dict[str, Any]:
    try:
        import ijson
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ReportUpdateError("authenticated-prefix extraction requires ijson") from exc

    selected: dict[str, Any] | None = None
    member_seen = False
    try:
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if member.name != checkpoint_member:
                    continue
                if member_seen or not member.isfile():
                    raise ReportUpdateError("snapshot checkpoint member is duplicated or unsafe")
                member_seen = True
                stream = archive.extractfile(member)
                if stream is None:
                    raise ReportUpdateError("snapshot checkpoint member is unreadable")
                for index, raw in enumerate(
                    ijson.items(
                        stream,
                        "adapt_vqe.active_prefix_checkpoints.item",
                        use_float=True,
                    ),
                    start=1,
                ):
                    if index == controller_round:
                        if not isinstance(raw, dict):
                            raise ReportUpdateError("authenticated prefix is not an object")
                        selected = raw
                        break
                break
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ReportUpdateError("snapshot prefix stream failed") from exc
    if not member_seen or selected is None:
        raise ReportUpdateError(
            f"snapshot lacks authenticated prefix round {controller_round}"
        )
    return selected


def _compile_authenticated_prefix(
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    ra = cell.get("ra")
    if not isinstance(ra, Mapping):
        raise ReportUpdateError("live cell lacks an RA record")
    controller_round = _integer(
        ra.get("algorithmic_work_prefix_round"),
        label="authenticated cost round",
        minimum=1,
    )
    source = ra.get("source")
    if not isinstance(source, Mapping):
        raise ReportUpdateError("live RA source is unavailable")
    archive_binding = source.get("snapshot_archive")
    archive_path = _path_from_binding(archive_binding, label="snapshot archive")
    if (
        archive_path.stat().st_size != archive_binding.get("size_bytes")
        or _sha256(archive_path) != archive_binding.get("sha256")
    ):
        raise ReportUpdateError("snapshot archive byte binding drifted")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("checkpoint"), Mapping):
        raise ReportUpdateError("snapshot checkpoint binding is unavailable")
    checkpoint_text = str(snapshot["checkpoint"].get("path", ""))
    if "#" not in checkpoint_text:
        raise ReportUpdateError("snapshot checkpoint member name is unavailable")
    checkpoint_member = checkpoint_text.split("#", 1)[1]
    selected = _extract_active_prefix(
        archive_path,
        checkpoint_member=checkpoint_member,
        controller_round=controller_round,
    )

    job_path = _path_from_binding(source.get("job"), label="resume job")
    job = _load(job_path)
    protocol_binding = job.get("source_protocol")
    protocol_path = _path_from_binding(protocol_binding, label="source protocol")
    if (
        protocol_path.stat().st_size != protocol_binding.get("size_bytes")
        or _sha256(protocol_path) != protocol_binding.get("sha256")
    ):
        raise ReportUpdateError("source protocol byte binding drifted")

    compact = {"adapt_vqe": {"active_prefix_checkpoints": [{}] * (controller_round - 1) + [selected]}}
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as stream:
        json.dump(compact, stream, sort_keys=True, allow_nan=False)
        compact_path = Path(stream.name)
    try:
        prefix = master._qiskit_plateau_prefix_from_checkpoint(
            compact_path,
            protocol_path=protocol_path,
            controller_round=controller_round,
        )
    finally:
        compact_path.unlink(missing_ok=True)

    point = _point_at(
        ra.get("points", ()),
        controller_round,
        label=f"{cell.get('regime_id')} RA",
    )
    observation = master._fixed_prefix_qiskit_observation(
        prefix,
        error=_finite(point.get("delta_e"), label="authenticated prefix error"),
        compiler=None,
    )
    expected_work = ra.get("algorithmic_work")
    if not isinstance(expected_work, Mapping):
        raise ReportUpdateError("authenticated prefix work is unavailable")
    expected_s_alg = _integer(
        expected_work.get("S_alg"), label="authenticated prefix S_alg"
    )
    if observation.get("S_alg") != expected_s_alg:
        raise ReportUpdateError("compiled prefix S_alg disagrees with the closed ledger")
    costs = {field: _integer(observation[field], label=field) for field in COST_FIELDS}
    return {
        "round": controller_round,
        "energy": _finite(point.get("energy"), label="authenticated prefix energy"),
        "delta_e": _finite(point.get("delta_e"), label="authenticated prefix error"),
        "costs": costs,
        "compile": {
            "compile_convention": observation["compile_convention"],
            "qiskit_version": observation.get("qiskit_version"),
            "qiskit_basis_work_schema": observation.get("qiskit_basis_work_schema"),
            "qiskit_basis_work_status": observation.get("qiskit_basis_work_status"),
            "source": "authenticated_active_prefix_checkpoint_v1",
        },
        "checkpoint_sha256": observation["checkpoint_sha256"],
        "snapshot_archive": _binding(archive_path),
        "checkpoint_member": checkpoint_member,
        "source_protocol": _binding(protocol_path),
    }


def build_cost_filled_adapter(*, write: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load(PAGE7_SOURCE_ADAPTER)
    if (
        source.get("sha256") != EXPECTED_PAGE7_ADAPTER_SHA256
        or _canonical_digest(source) != EXPECTED_PAGE7_ADAPTER_SHA256
        or tuple(source.get("regime_order", ())) != completed.REGIME_ORDER
    ):
        raise ReportUpdateError("source page-7 adapter identity drifted")
    adapter = copy.deepcopy(source)
    adapter.pop("sha256", None)
    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    prefix_rows: list[dict[str, Any]] = []
    for regime in LIVE_REGIMES:
        cell = cells[regime]
        observation = _compile_authenticated_prefix(cell)
        ra = cell["ra"]
        ra["authenticated_cost_prefix"] = copy.deepcopy(observation)
        ra["qiskit_status"] = "available_at_last_authenticated_checkpoint_prefix"
        ra["qiskit_costs"] = {
            field: observation["costs"][field] for field in COST_FIELDS[:4]
        }
        ra["current_stdout_prefix_cost_status"] = (
            "unavailable_no_checkpoint_or_estimator_ledger"
        )
        prefix_rows.append(
            {
                "regime_id": regime,
                "execution_id": ra["execution_id"],
                **copy.deepcopy(observation),
            }
        )
    adapter["schema"] = (
        "paper_i_historical_mean_global_singleton_vs_append_r70_live_full6_"
        "cost_filled_adapter_v2"
    )
    adapter["status"] = "passed_with_authenticated_prefix_cost_completion"
    adapter["source_adapter"] = {
        **_binding(PAGE7_SOURCE_ADAPTER),
        "canonical_sha256": EXPECTED_PAGE7_ADAPTER_SHA256,
    }
    adapter["layout"] = {"page_count": 2, "accuracy_page_panels": 6, "cost_page_rows": 6}
    adapter["cost_policy"]["live_partial"] = {
        "accuracy": "latest safely observed stdout prefix",
        "cost": "last authenticated closed checkpoint prefix",
        "current_stdout_cost": "unavailable without checkpoint and estimator ledger",
    }
    adapter["limitations"] = [
        "The three nph=7 energy curves extend beyond their last authenticated "
        "checkpoint prefixes. Full cost tuples are reported only at k=49, k=45, "
        "and k=31; later stdout-only points are never assigned inferred costs.",
        "No live or stdout-only cell is adopted Paper-I evidence.",
    ]
    result = live.digested(adapter)
    sidecar = live.digested(
        {
            "schema": "paper_i_historical_mean_global_singleton_authenticated_prefix_costs_v1",
            "status": "passed",
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "source_adapter": {
                **_binding(PAGE7_SOURCE_ADAPTER),
                "canonical_sha256": EXPECTED_PAGE7_ADAPTER_SHA256,
            },
            "cells": prefix_rows,
        }
    )
    if write:
        completed.legacy_page._atomic_write_json(OUTPUT_ADAPTER, result)
        completed.legacy_page._atomic_write_json(OUTPUT_COST_SIDECAR, sidecar)
    return result, sidecar


def _render_accuracy_plot(
    adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 4.55))
    fig.subplots_adjust(
        left=0.070,
        right=0.990,
        bottom=0.165,
        top=0.875,
        wspace=0.205,
        hspace=0.335,
    )
    for index, regime in enumerate(completed.REGIME_ORDER):
        ax = axes.flat[index]
        cell = cells[regime]
        append = cell["append"]
        ra = cell["ra"]
        for method, color in ((append, "#4C78A8"), (ra, "#E45756")):
            ax.plot(
                [point["round"] for point in method["points"]],
                [max(float(point["delta_e"]), PLOT_FLOOR) for point in method["points"]],
                color=color,
                linewidth=1.7,
            )
        append_marker = append["effective_plateau"]
        ra_marker = ra["effective_plateau"] if cell["status"] == "complete" else ra["terminal"]
        ax.scatter(
            [append_marker["round"]],
            [max(float(append_marker["delta_e"]), PLOT_FLOOR)],
            marker="o",
            color="#4C78A8",
            s=30,
            zorder=5,
        )
        ax.scatter(
            [ra_marker["round"]],
            [max(float(ra_marker["delta_e"]), PLOT_FLOOR)],
            marker="D",
            color="#E45756",
            s=32,
            zorder=5,
        )
        if cell["status"] != "complete":
            costed = ra["authenticated_cost_prefix"]
            ax.scatter(
                [costed["round"]],
                [max(float(costed["delta_e"]), PLOT_FLOOR)],
                marker="s",
                facecolor="white",
                edgecolor="#7A5195",
                linewidth=1.2,
                s=34,
                zorder=6,
            )
            ax.annotate(
                f"costed k={costed['round']}",
                (costed["round"], max(float(costed["delta_e"]), PLOT_FLOOR)),
                xytext=(-4, 7),
                textcoords="offset points",
                ha="right",
                fontsize=7.0,
                color="#5B2C6F",
            )
        ax.set_title(str(cell["display_name"]))
        ax.set_yscale("log")
        xmax = max(int(append["points"][-1]["round"]), int(ra["points"][-1]["round"]))
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=70))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#d8d8d8", linewidth=0.55)
        ax.grid(which="minor", color="#efefef", linewidth=0.35)
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $|\Delta E|$")
        if index >= 3:
            ax.set_xlabel("Accepted ADAPT round")
    fig.suptitle(
        "Global-singleton RA versus fresh Append-ADAPT: accuracy trajectories",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#4C78A8", marker="o", label="Append-ADAPT"),
            Line2D([0], [0], color="#E45756", marker="D", label="Global-singleton RA"),
            Line2D(
                [0],
                [0],
                color="none",
                marker="s",
                markerfacecolor="white",
                markeredgecolor="#7A5195",
                label="Last authenticated cost prefix",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=8.4,
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _compile_tex(tex_path: Path) -> Path:
    command = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    completed_process = subprocess.run(
        command,
        cwd=tex_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        raise ReportUpdateError(
            f"LaTeX build failed for {tex_path.name}:\n{completed_process.stdout}\n"
            f"{completed_process.stderr}"
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        raise ReportUpdateError(f"LaTeX did not create {pdf_path.name}")
    return pdf_path


def _write_page6_tex() -> Path:
    adapter = _load(PAGE6_ADAPTER)
    costs = _load(PAGE6_COSTS)
    cost_cells = {str(row["regime_id"]): row for row in costs["cells"]}
    rows: list[str] = []
    crossing_rows: list[str] = []
    for cell in adapter["cells"]:
        regime = str(cell["regime_id"])
        label = _latex_text(cell["display_name"])
        ra = cost_cells[regime]["ra_round_69"]
        append = cell["append"]["endpoints"]["round_70"]
        rows.append(
            f"{label} & {_sci(float(ra['delta_e']))} & {_latex_text(_cost4(ra['costs']))} & "
            f"{_sci(float(append['delta_e']))} & {_latex_text(_cost4(append['costs']))} & "
            f"{int(append['costs']['S_alg']):,} \\\\"
        )
        common = cost_cells[regime]["common_accuracy"]
        crossing_rows.append(
            f"{_latex_text(regime.replace('_u8', '').replace('_', '-').upper())} & "
            f"{_sci(float(common['target_delta_e']))} & {int(common['ra']['round'])} & "
            f"{_sci(float(common['ra']['delta_e']))} & {int(common['append']['round'])} & "
            f"{_sci(float(common['append']['delta_e']))} \\\\"
        )
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.28in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx,xcolor}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\large\bfseries Historical-average global-singleton R70 diagnostic}}\\[-0.1ex]
{{\small Accuracy is recovered through RA round 69; the lost RA estimator ledger excludes this page from $S_{{\rm alg}}$ ranking.}}

\includegraphics[width=0.91\textwidth,height=3.05in,keepaspectratio]{{{_latex_path(PAGE6_PLOT)}}}

\fontsize{{7.25}}{{8.0}}\selectfont
\setlength{{\tabcolsep}}{{4.0pt}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Regime & $|\Delta E_{{69}}^{{\rm RA}}|$ & $(N_{{2q}},D_{{2q}},D_c,W_{{1q}})_{{69}}^{{\rm RA}}$ &
$|\Delta E_{{70}}^{{\rm Append}}|$ & $(N_{{2q}},D_{{2q}},D_c,W_{{1q}})_{{70}}^{{\rm Append}}$ & $S_{{\rm alg,70}}^{{\rm Append}}$ \\
\midrule
{os.linesep.join(rows)}
\bottomrule
\end{{tabular}}

\vspace{{0.30em}}
{{\bfseries\fontsize{{7.4}}{{8.1}}\selectfont Common-accuracy crossings before the earlier effective plateau}}

\fontsize{{6.7}}{{7.35}}\selectfont
\setlength{{\tabcolsep}}{{5.5pt}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Regime & $|\Delta E_\cap|$ & $k_\cap^{{\rm RA}}$ & $|\Delta E_{{\rm RA}}|$ & $k_\cap^{{\rm Append}}$ & $|\Delta E_{{\rm Append}}|$ \\
\midrule
{os.linesep.join(crossing_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\vfill
{{\fontsize{{6.45}}{{7.2}}\selectfont\color{{red!55!black}}
\textbf{{Nonrecoverable field.}} Exact RA $S_{{\rm alg}}$ cannot be reconstructed: the post-run EXDEV failure prevented checkpoint and estimator-ledger transfer. No formula estimate is substituted. The four displayed RA Qiskit fields are compiled from the recovered generator sequence and are not used in the accuracy-first/$S_{{\rm alg}}$ route ordering.}}
\end{{document}}
"""
    path = REPORT_DIR / f"{PAGE6_STEM}.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def _accuracy_winner(ra_error: float, append_error: float) -> str:
    if math.isclose(ra_error, append_error, rel_tol=2.0e-4, abs_tol=1.0e-15):
        return "tie"
    return "RA" if ra_error < append_error else "Append"


def _selected_cost_row(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    ra = cell["ra"]
    if cell["status"] == "complete":
        return ra["terminal"]
    return ra["authenticated_cost_prefix"]


def _write_accuracy_tex(adapter: Mapping[str, Any], plot_pdf: Path) -> Path:
    rows: list[str] = []
    for cell in adapter["cells"]:
        ra = cell["ra"]
        append = cell["append"]
        ra_terminal = ra["terminal"]
        append_terminal = append["terminal"]
        ra_costed = _selected_cost_row(cell)
        rows.append(
            f"{_latex_text(cell['display_name'])} & {int(ra_terminal['round'])} & "
            f"{_sci(float(ra_terminal['delta_e']))} & {int(ra_costed['round'])} & "
            f"{_latex_text(_cost5(ra_costed['costs']))} & "
            f"{int(append_terminal['round'])} & {_sci(float(append_terminal['delta_e']))} & "
            f"{_latex_text(_cost5(append_terminal['costs']))} \\\\"
        )
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.30in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx,xcolor}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\large\bfseries Global-singleton RA versus fresh Append-ADAPT: accuracy and authenticated cost}}\\[-0.1ex]
{{\small Read accuracy first, then compare $S_{{\rm alg}}$ inside the full cost tuple.}}

\includegraphics[width=0.94\textwidth,height=3.55in,keepaspectratio]{{{_latex_path(plot_pdf)}}}

\fontsize{{6.65}}{{7.35}}\selectfont
\setlength{{\tabcolsep}}{{3.6pt}}
\begin{{tabular*}}{{0.985\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrrrrrr@{{}}}}
\toprule
Regime & $k_T^{{\rm RA}}$ & $|\Delta E_T^{{\rm RA}}|$ & $k_C^{{\rm RA}}$ & $C_C^{{\rm RA}}$ &
$k_A$ & $|\Delta E_A|$ & $C_A^{{\rm Append}}$ \\
\midrule
{os.linesep.join(rows)}
\bottomrule
\end{{tabular*}}
\end{{center}}

\vspace{{0.35em}}
{{\fontsize{{6.55}}{{7.25}}\selectfont
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$. The other four Qiskit fields are complete but excluded from route ranking. Diamonds set the displayed accuracy endpoint $k_T$; squares set the authenticated cost prefix $k_C$. For WS, IS, and SS, $k_T>k_C$ where applicable, so no later cost is inferred. Append accuracy and cost share the displayed endpoint $k_A$.}}
\end{{document}}
"""
    path = REPORT_DIR / f"{PAGE7_STEM}.tex"
    path.write_text(tex, encoding="utf-8")
    return path
def build_pages(adapter: Mapping[str, Any]) -> dict[str, Path]:
    accuracy_png = REPORT_DIR / f"{PAGE7_STEM}_plot.png"
    accuracy_plot_pdf = REPORT_DIR / f"{PAGE7_STEM}_plot.pdf"
    _render_accuracy_plot(adapter, png_path=accuracy_png, pdf_path=accuracy_plot_pdf)
    page6_pdf = _compile_tex(_write_page6_tex())
    page7_pdf = _compile_tex(_write_accuracy_tex(adapter, accuracy_plot_pdf))
    return {
        "page6_pdf": page6_pdf,
        "accuracy_plot_png": accuracy_png,
        "accuracy_plot_pdf": accuracy_plot_pdf,
        "accuracy_page_pdf": page7_pdf,
    }


def _page_content_hash(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def replace_report(
    *,
    adapter: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    assets: Mapping[str, Path],
) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    if _sha256(MASTER_PDF) != EXPECTED_MASTER_SHA256:
        raise ReportUpdateError("master PDF byte identity drifted")
    reader = PdfReader(str(MASTER_PDF), strict=False)
    if len(reader.pages) != 8:
        raise ReportUpdateError("expected an eight-page source report")
    page6_reader = PdfReader(str(assets["page6_pdf"]), strict=False)
    page7_reader = PdfReader(str(assets["accuracy_page_pdf"]), strict=False)
    if any(len(candidate.pages) != 1 for candidate in (page6_reader, page7_reader)):
        raise ReportUpdateError("each replacement page asset must contain one page")

    before_hashes = [_page_content_hash(page) for page in reader.pages]
    writer = PdfWriter()
    for page in reader.pages[:5]:
        writer.add_page(page)
    writer.add_page(page6_reader.pages[0])
    writer.add_page(page7_reader.pages[0])
    writer.add_page(reader.pages[7])
    if reader.metadata:
        writer.add_metadata(dict(reader.metadata))

    temporary_pdf = MASTER_PDF.with_suffix(".reformat.tmp.pdf")
    temporary_provenance = MASTER_PROVENANCE.with_suffix(".reformat.tmp.json")
    rollback_pdf = MASTER_PDF.with_suffix(".reformat.rollback.pdf")
    for path in (temporary_pdf, temporary_provenance, rollback_pdf):
        path.unlink(missing_ok=True)
    with temporary_pdf.open("xb") as stream:
        writer.write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    updated_reader = PdfReader(str(temporary_pdf), strict=False)
    if len(updated_reader.pages) != 8:
        raise ReportUpdateError("reformatted report does not contain eight pages")
    after_hashes = [_page_content_hash(page) for page in updated_reader.pages]
    if after_hashes[:5] != before_hashes[:5] or after_hashes[7] != before_hashes[7]:
        raise ReportUpdateError("report replacement changed preserved pages")

    provenance = _load(MASTER_PROVENANCE)
    updated = copy.deepcopy(provenance)
    updated["layout"].update(
        {
            "page_6": "historical_average_r70_accuracy_only_reformatted_v2",
            "page_7": "historical_mean_global_singleton_accuracy_and_cost_v6",
            "page_count": 8,
        }
    )
    updated["historical_mean_global_singleton_cost_completion"] = {
        "schema": "paper_i_ra_singleton_cost_completion_report_v1",
        "status": "passed",
        "source_master_pdf": {
            "sha256": EXPECTED_MASTER_SHA256,
            "page_count": 8,
        },
        "adapter": {
            **_binding(OUTPUT_ADAPTER),
            "canonical_sha256": adapter["sha256"],
        },
        "cost_sidecar": {
            **_binding(OUTPUT_COST_SIDECAR),
            "canonical_sha256": sidecar["sha256"],
        },
        "ranking_policy": {
            "primary": "same_cutoff_absolute_energy_error",
            "secondary": "S_alg",
            "ignored_for_ranking": ["N2q", "D2q", "Dc", "W1q"],
        },
        "structural_validation": {
            "pages_before": 8,
            "pages_after": 8,
            "preserved_source_page_content_sha256": before_hashes[:5] + [before_hashes[7]],
            "new_page_content_sha256": after_hashes[5:7],
        },
        "outputs": {name: _binding(path) for name, path in assets.items()},
    }
    updated["limitations"] = [
        limitation
        for limitation in updated.get("limitations", [])
        if not str(limitation).startswith("Page 7 is a supplemental early-stop diagnostic")
    ]
    updated["limitations"].append(
        "Page 7 places the accuracy panels and authenticated cost table together. "
        "Full RA tuples are available at k=49, k=45, and k=31; later nph=7 "
        "stdout points still have no checkpoint-backed cost."
    )
    updated["outputs"]["partial_progress_pdf"] = {
        **_binding(temporary_pdf),
        "path": str(MASTER_PDF.resolve()),
    }
    updated["outputs"]["historical_mean_global_singleton_cost_filled_adapter"] = _binding(OUTPUT_ADAPTER)
    updated["outputs"]["historical_mean_global_singleton_authenticated_prefix_costs"] = _binding(OUTPUT_COST_SIDECAR)

    try:
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(MASTER_PDF, rollback_pdf)
        os.replace(temporary_pdf, MASTER_PDF)
        try:
            os.replace(temporary_provenance, MASTER_PROVENANCE)
        except Exception:
            os.replace(rollback_pdf, MASTER_PDF)
            raise
        rollback_pdf.unlink(missing_ok=True)
    except Exception:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback_pdf.unlink(missing_ok=True)
        raise
    return {
        "status": "reformatted_and_cost_filled",
        "pages": 8,
        "pdf_sha256": _sha256(MASTER_PDF),
        "costs": [
            {
                "regime_id": row["regime_id"],
                "round": row["round"],
                "costs": row["costs"],
            }
            for row in sidecar["cells"]
        ],
    }


def refresh_report_pages(
    *,
    adapter: Mapping[str, Any],
    assets: Mapping[str, Path],
) -> dict[str, Any]:
    """Combine the current split pages without redoing science extraction."""
    from pypdf import PdfReader, PdfWriter

    provenance = _load(MASTER_PROVENANCE)
    expected_pdf = provenance.get("outputs", {}).get("partial_progress_pdf", {})
    if (
        not isinstance(expected_pdf, Mapping)
        or expected_pdf.get("sha256") != _sha256(MASTER_PDF)
    ):
        raise ReportUpdateError("current report is not bound by its provenance")
    if _canonical_digest(adapter) != adapter.get("sha256"):
        raise ReportUpdateError("cost-filled adapter canonical digest drifted")

    reader = PdfReader(str(MASTER_PDF), strict=False)
    if len(reader.pages) != 9:
        raise ReportUpdateError("page combination expects the current nine-page report")
    replacements = [
        PdfReader(str(assets[key]), strict=False)
        for key in ("page6_pdf", "accuracy_page_pdf")
    ]
    if any(len(candidate.pages) != 1 for candidate in replacements):
        raise ReportUpdateError("each refreshed page asset must contain one page")

    before_hashes = [_page_content_hash(page) for page in reader.pages]
    writer = PdfWriter()
    for page in reader.pages[:5]:
        writer.add_page(page)
    for candidate in replacements:
        writer.add_page(candidate.pages[0])
    writer.add_page(reader.pages[8])
    if reader.metadata:
        writer.add_metadata(dict(reader.metadata))

    temporary_pdf = MASTER_PDF.with_suffix(".refresh.tmp.pdf")
    temporary_provenance = MASTER_PROVENANCE.with_suffix(".refresh.tmp.json")
    rollback_pdf = MASTER_PDF.with_suffix(".refresh.rollback.pdf")
    for path in (temporary_pdf, temporary_provenance, rollback_pdf):
        path.unlink(missing_ok=True)
    with temporary_pdf.open("xb") as stream:
        writer.write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    refreshed = PdfReader(str(temporary_pdf), strict=False)
    after_hashes = [_page_content_hash(page) for page in refreshed.pages]
    if len(refreshed.pages) != 8:
        raise ReportUpdateError("combined report does not contain eight pages")
    if after_hashes[:5] != before_hashes[:5] or after_hashes[7] != before_hashes[8]:
        raise ReportUpdateError("page refresh changed a preserved page")

    updated = copy.deepcopy(provenance)
    completion = updated["historical_mean_global_singleton_cost_completion"]
    completion["structural_validation"].update(
        {
            "pages_after": 8,
            "new_page_content_sha256": after_hashes[5:7],
        }
    )
    completion["presentation"] = (
        "Accuracy panels and their authenticated cost table share page 7."
    )
    completion["outputs"] = {
        name: _binding(path) for name, path in assets.items()
    }
    updated["layout"].update(
        {
            "page_7": "historical_mean_global_singleton_accuracy_and_cost_v6",
            "page_8": updated["layout"].get("page_9", updated["layout"].get("page_8")),
            "page_count": 8,
        }
    )
    updated["layout"].pop("page_9", None)
    updated["limitations"] = [
        limitation
        for limitation in updated.get("limitations", [])
        if not str(limitation).startswith("Pages 7-8 separate")
    ]
    updated["limitations"].append(
        "Page 7 places the accuracy panels and authenticated cost table together. "
        "Full RA tuples are available at k=49, k=45, and k=31; later nph=7 "
        "stdout points still have no checkpoint-backed cost."
    )
    updated["outputs"]["partial_progress_pdf"] = {
        **_binding(temporary_pdf),
        "path": str(MASTER_PDF.resolve()),
    }

    try:
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(MASTER_PDF, rollback_pdf)
        os.replace(temporary_pdf, MASTER_PDF)
        try:
            os.replace(temporary_provenance, MASTER_PROVENANCE)
        except Exception:
            os.replace(rollback_pdf, MASTER_PDF)
            raise
        rollback_pdf.unlink(missing_ok=True)
    except Exception:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback_pdf.unlink(missing_ok=True)
        raise
    return {
        "status": "accuracy_and_cost_combined",
        "pages": 8,
        "pdf_sha256": _sha256(MASTER_PDF),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--refresh-pages", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.dry_run and args.refresh_pages:
            raise ReportUpdateError("choose either --dry-run or --refresh-pages")
        if args.refresh_pages:
            adapter = _load(OUTPUT_ADAPTER)
            assets = build_pages(adapter)
            result = refresh_report_pages(adapter=adapter, assets=assets)
        else:
            adapter, sidecar = build_cost_filled_adapter(write=not args.dry_run)
        if args.dry_run:
            result = {
                "status": "validated_without_writes",
                "costs": [
                    {
                        "regime_id": row["regime_id"],
                        "round": row["round"],
                        "costs": row["costs"],
                    }
                    for row in sidecar["cells"]
                ],
            }
        elif not args.refresh_pages:
            assets = build_pages(adapter)
            result = replace_report(adapter=adapter, sidecar=sidecar, assets=assets)
    except (OSError, RuntimeError, ReportUpdateError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
