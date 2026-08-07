#!/usr/bin/env python3
"""Refresh master-report page 7 from bounded r70 early-stop telemetry.

The established page-7 adapter carries three completed nph=3 RA trajectories,
three nph=7 RA prefixes, and fixed fresh-Append overlays.  This updater extends
all three nph=7 curves using accepted-energy events captured from Condor stdout
before a user-requested early stop.  The termination hook timed out before the
large checkpoint archives transferred, so current-prefix Qiskit and S_alg
values remain explicitly pending.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
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
SOURCE_ADAPTER = REPORT_DIR / (
    "historical_mean_global_singleton_live_page7_"
    "k51_45_31_r70_stdout_20260806_v1_adapter.json"
)
PROGRESS_RECEIPT = REPORT_DIR / (
    "historical_mean_global_singleton_r70_early_stop_stdout_"
    "telemetry_20260807.json"
)
ASSET_STEM = (
    "historical_mean_global_singleton_live_page7_"
    "k61_49_38_r70_early_stop_stdout_20260807_v2"
)
OUTPUT_ADAPTER = REPORT_DIR / f"{ASSET_STEM}_adapter.json"
SOURCE_ADAPTER_CANONICAL_SHA256 = (
    "087ebe4339eacad5fa9697db46159eb8562dc3ed63553557dcc16b5c90401376"
)
REPORT_KEY = "historical_mean_global_singleton_vs_append_mixed_horizon"
LIVE_REGIMES = ("weak_strong", "intermediate_strong", "strong_strong_u8")
EXPECTED_SOURCE_ROUNDS = {
    "weak_strong": 51,
    "intermediate_strong": 45,
    "strong_strong_u8": 31,
}
EXPECTED_LIVE_ROUNDS = {
    "weak_strong": 61,
    "intermediate_strong": 49,
    "strong_strong_u8": 38,
}
EXPECTED_SCHEDULER = {
    "weak_strong": {"cluster_id": 9_572_720, "proc_id": 0},
    "intermediate_strong": {"cluster_id": 9_576_843, "proc_id": 0},
    "strong_strong_u8": {"cluster_id": 9_576_843, "proc_id": 1},
}
TARGET_HORIZON = 70
PLOT_FLOOR = 1.0e-16
HEX64 = re.compile(r"[0-9a-f]{64}")
LIMITATION = (
    "Page 7 is a supplemental early-stop diagnostic. The three nph=7 RA "
    "curves extend to safely observed stdout prefixes k=61, k=49, and k=38. "
    "Their checkpoint archives did not transfer before the graceful-vacate "
    "timeout, so current-prefix Qiskit costs and closed S_alg remain pending. "
    "No stdout-only cell is adopted Paper-I evidence."
)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    add_paper_i_historical_mean_global_singleton_full6_page as completed,
)
from pipelines.reporting import (  # noqa: E402
    add_paper_i_historical_mean_global_singleton_live_page7 as live,
)


class UpdateError(ValueError):
    """Raised when the bounded page refresh cannot be proven safe."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise UpdateError(f"unsafe or missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UpdateError(f"JSON input is not an object: {path}")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(live.canonical_json_bytes(unsigned)).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise UpdateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise UpdateError(f"{label} is not finite")
    return result


def _source_adapter() -> dict[str, Any]:
    adapter = _load(SOURCE_ADAPTER)
    if (
        adapter.get("sha256") != SOURCE_ADAPTER_CANONICAL_SHA256
        or _canonical_digest(adapter) != SOURCE_ADAPTER_CANONICAL_SHA256
        or tuple(adapter.get("regime_order", ())) != completed.REGIME_ORDER
        or adapter.get("layout")
        != {"grid": "2x3", "page_count": 1, "panel_count": 6}
    ):
        raise UpdateError("source page-7 adapter identity drifted")
    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    if set(cells) != set(completed.REGIME_ORDER):
        raise UpdateError("source page-7 adapter cell closure drifted")
    for regime, source_round in EXPECTED_SOURCE_ROUNDS.items():
        ra = cells[regime]["ra"]
        if (
            cells[regime].get("status") != "live_partial"
            or ra.get("live_controller_round") != source_round
            or len(ra.get("points", ())) != source_round + 1
            or int(ra["points"][-1]["round"]) != source_round
        ):
            raise UpdateError(f"source {regime} prefix drifted")
    return adapter


def _progress() -> dict[str, Any]:
    progress = _load(PROGRESS_RECEIPT)
    capture = progress.get("capture")
    regimes = progress.get("regimes")
    if (
        progress.get("schema")
        != "paper_i_ra_global_singleton_r70_early_stop_stdout_telemetry_v1"
        or progress.get("status") != "held_after_user_requested_early_stop"
        or progress.get("paper_evidence_adopted") is not False
        or not isinstance(capture, dict)
        or capture.get("method") != "one_shot_condor_tail_before_eviction"
        or capture.get("energy_mapping")
        != "event_depth_d_energy_is_accepted_round_d_minus_1_energy_before_refit_v1"
        or capture.get("checkpoint_archive_recovered") is not False
        or not isinstance(regimes, dict)
        or set(regimes) != set(LIVE_REGIMES)
    ):
        raise UpdateError("early-stop stdout receipt identity drifted")
    for regime in LIVE_REGIMES:
        row = regimes[regime]
        scheduler = row.get("scheduler")
        events = row.get("events")
        source_round = EXPECTED_SOURCE_ROUNDS[regime]
        live_round = EXPECTED_LIVE_ROUNDS[regime]
        expected_scheduler = EXPECTED_SCHEDULER[regime]
        if (
            not isinstance(scheduler, dict)
            or scheduler.get("cluster_id") != expected_scheduler["cluster_id"]
            or scheduler.get("proc_id") != expected_scheduler["proc_id"]
            or scheduler.get("final_job_status") != 5
            or scheduler.get("num_job_starts") != 1
            or scheduler.get("run_bytes_sent_by_job") != 0
            or not str(scheduler.get("hold_reason", "")).startswith(
                "user-requested early stop"
            )
            or row.get("last_safely_observed_round") != live_round
            or not isinstance(events, list)
            or [event.get("depth") for event in events]
            != list(range(source_round + 1, live_round + 2))
        ):
            raise UpdateError(f"{regime}: early-stop scheduler/events drifted")
        for event in events:
            _finite(event.get("energy"), label=f"{regime} stdout energy")
            _finite(event.get("max_grad"), label=f"{regime} stdout max_grad")
            if event.get("stage_name") != "core":
                raise UpdateError(f"{regime}: stdout event stage drifted")
    return progress


def _plot_provenance(adapter: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell in adapter["cells"]:
        ra = cell["ra"]
        append = cell["append"]
        if cell["status"] == "complete":
            ra_marker = ra["effective_plateau"]
            ra_policy = "first_effective_plateau_prefix"
        else:
            ra_marker = ra["terminal"]
            ra_policy = "terminal_observed_live_prefix"
        rows.append(
            {
                "regime_id": cell["regime_id"],
                "append_point_count": len(append["points"]),
                "append_marker": {
                    "round": int(append["effective_plateau"]["round"]),
                    "delta_e": float(append["effective_plateau"]["delta_e"]),
                    "policy": "first_effective_plateau_prefix",
                },
                "ra_point_count": len(ra["points"]),
                "ra_marker": {
                    "round": int(ra_marker["round"]),
                    "delta_e": float(ra_marker["delta_e"]),
                    "policy": ra_policy,
                },
            }
        )
    return {
        "metric": "same_cutoff_abs_delta_e",
        "x_axis": "accepted_adapt_controller_round",
        "layout": "six_panels_2x3_single_page",
        "curve_style": "solid_lines_one_marker_per_curve",
        "panels": rows,
    }


def build_adapter(*, write: bool) -> dict[str, Any]:
    source = _source_adapter()
    progress = _progress()
    adapter = copy.deepcopy(source)
    adapter.pop("sha256", None)
    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    package_root = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
    job_root = package_root / (
        "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
        "r70_20260804_v1_resume256gb_loaderfix_v2_chtc/jobs"
    )
    progress_binding = live.file_binding(PROGRESS_RECEIPT)
    source_binding = {
        **live.file_binding(SOURCE_ADAPTER),
        "canonical_sha256": SOURCE_ADAPTER_CANONICAL_SHA256,
    }
    for regime in LIVE_REGIMES:
        row = progress["regimes"][regime]
        scheduler = row["scheduler"]
        ra = cells[regime]["ra"]
        old_terminal = copy.deepcopy(ra["terminal"])
        events = row["events"]
        source_round = EXPECTED_SOURCE_ROUNDS[regime]
        authenticated_round = int(row["last_authenticated_closed_prefix_round"])
        exact = _finite(ra["exact_same_cutoff_energy"], label=f"{regime} ED")
        if not math.isclose(
            _finite(events[0]["energy"], label=f"{regime} join energy"),
            _finite(old_terminal["energy"], label=f"{regime} source energy"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise UpdateError(f"{regime}: stdout does not join source k{source_round}")
        points = copy.deepcopy(ra["points"])
        prior_energy = _finite(points[-1]["energy"], label=f"{regime} prior energy")
        for event in events[1:]:
            round_index = int(event["depth"]) - 1
            if round_index != int(points[-1]["round"]) + 1:
                raise UpdateError(f"{regime}: accepted-round sequence is not contiguous")
            energy = _finite(event["energy"], label=f"{regime} round {round_index}")
            if energy > prior_energy + 1.0e-10:
                raise UpdateError(f"{regime}: accepted energy increased")
            points.append(
                {"round": round_index, "energy": energy, "delta_e": abs(energy - exact)}
            )
            prior_energy = energy
        live_round = int(points[-1]["round"])
        if live_round != EXPECTED_LIVE_ROUNDS[regime]:
            raise UpdateError(f"{regime}: bounded terminal round drifted")
        ra["schema"] = "paper_i_ra_global_singleton_r70_early_stop_stdout_projection_v1"
        ra["status"] = "early_stopped_stdout_partial"
        ra["execution_id"] = row["execution_id"]
        ra["cluster_id"] = int(scheduler["cluster_id"])
        ra["proc_id"] = int(scheduler["proc_id"])
        ra["observed_utc"] = events[-1]["ts_utc"]
        ra["scheduler_state"] = "held_after_user_requested_early_stop"
        ra["target_horizon"] = TARGET_HORIZON
        ra["live_controller_round"] = live_round
        ra["active_ansatz_depth"] = live_round
        ra["points"] = points
        ra["available_prefix_effective_plateau"] = completed._effective_plateau(
            points, label=f"{regime} r70 early-stop stdout prefix"
        )
        ra["algorithmic_work_prefix_round"] = authenticated_round
        ra["current_algorithmic_work_status"] = "unavailable_checkpoint_archive_not_recovered"
        ra["qiskit_status"] = "pending_stdout_only_prefix_not_compiled"
        ra["qiskit_costs"] = copy.deepcopy(live.QISKIT_PENDING)
        ra["terminal"] = {
            "round": live_round,
            "energy": float(points[-1]["energy"]),
            "delta_e": float(points[-1]["delta_e"]),
            "costs": {**copy.deepcopy(live.QISKIT_PENDING), "S_alg": None},
            "qiskit_status": "pending_stdout_only_prefix_not_compiled",
            "S_alg_status": (
                f"pending_current_prefix; last_authenticated_round_{authenticated_round}"
            ),
        }
        activation = (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
            "r70_20260804_v1_resume256gb_loaderfix_v2_chtc_activation_"
            + (
                "weak_strong_v1/submission_receipt_9572720.json"
                if regime == "weak_strong"
                else "remaining_strong2_20260806_v1/submission_receipt_9576843.json"
            )
        )
        job_name = (
            "historical_mean_global_singleton_v2_nph7_r70__"
            f"{regime}__nph7__ra_global_singleton_plateau__resume_from_d"
            f"{authenticated_round}_to_r70_256gb_loaderfix_v2.json"
        )
        ra["source"]["r70_continuation"] = {
            "progress_receipt": progress_binding,
            "source_adapter": source_binding,
            "submission_receipt": live.file_binding(package_root / activation),
            "job": live.file_binding(job_root / job_name),
            "cluster_id": int(scheduler["cluster_id"]),
            "proc_id": int(scheduler["proc_id"]),
            "accepted_energy_mapping": progress["capture"]["energy_mapping"],
            "live_energy_status": "scheduler_stdout_telemetry_not_checkpoint",
            "last_authenticated_closed_prefix_round": authenticated_round,
            "checkpoint_archive_recovered": False,
        }
    adapter["schema"] = (
        "paper_i_historical_mean_global_singleton_vs_append_r70_live_full6_"
        "adapter_v1"
    )
    adapter["status"] = "passed_mixed_complete_and_early_stop_stdout_partial"
    adapter["source_adapter"] = {
        **live.file_binding(SOURCE_ADAPTER),
        "canonical_sha256": SOURCE_ADAPTER_CANONICAL_SHA256,
    }
    adapter["limitations"] = [LIMITATION]
    adapter["cost_policy"]["live_partial"]["S_alg"] = (
        "authenticated_checkpoint_prefix_only; current stdout prefix pending"
    )
    adapter["cost_policy"]["terminal"]["ra_round"] = (
        "completed=50; live=observed prefix"
    )
    adapter["plot_provenance"] = _plot_provenance(adapter)
    result = live.digested(adapter)
    if OUTPUT_ADAPTER.exists():
        existing = _load(OUTPUT_ADAPTER)
        if live.canonical_json_bytes(existing) != live.canonical_json_bytes(result):
            raise UpdateError("output adapter already exists with different bytes")
    elif write:
        completed.legacy_page._atomic_write_json(OUTPUT_ADAPTER, result)
    return result


def _format_live_annotation(ra: Mapping[str, Any]) -> str:
    round_index = int(ra["live_controller_round"])
    delta_e = float(ra["points"][-1]["delta_e"])
    if str(ra.get("status", "")).endswith("stdout_partial"):
        work_round = int(ra["algorithmic_work_prefix_round"])
        s_alg = int(ra["algorithmic_work"]["S_alg"])
        return (
            f"stopped stdout k={round_index}; DE={delta_e:.2e}\n"
            f"S_alg@k{work_round}={s_alg:,}; current/Qiskit pending"
        )
    return (
        f"authenticated k={round_index}; DE={delta_e:.2e}\n"
        f"S_alg={int(ra['algorithmic_work']['S_alg']):,}; Qiskit pending"
    )


def render_plot(adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.1,
            "axes.labelsize": 8.3,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 4.05), constrained_layout=True)
    for index, regime in enumerate(completed.REGIME_ORDER):
        ax = axes.flat[index]
        cell = cells[regime]
        append = cell["append"]
        ra = cell["ra"]
        ax.plot(
            [point["round"] for point in append["points"]],
            [max(float(point["delta_e"]), PLOT_FLOOR) for point in append["points"]],
            color="#4C78A8",
            linewidth=1.55,
        )
        append_marker = append["effective_plateau"]
        ax.scatter(
            [append_marker["round"]],
            [max(float(append_marker["delta_e"]), PLOT_FLOOR)],
            marker="o",
            color="#4C78A8",
            s=27,
            zorder=5,
        )
        ax.plot(
            [point["round"] for point in ra["points"]],
            [max(float(point["delta_e"]), PLOT_FLOOR) for point in ra["points"]],
            color="#E45756",
            linewidth=1.75,
        )
        if cell["status"] == "complete":
            marker = ra["effective_plateau"]
            marker_round = int(marker["round"])
            marker_error = float(marker["delta_e"])
        else:
            marker_round = int(ra["live_controller_round"])
            marker_error = float(ra["points"][-1]["delta_e"])
            ax.text(
                0.98,
                0.96,
                _format_live_annotation(ra),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.75,
                color="#9C2F2F",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#CCCCCC",
                    "alpha": 0.88,
                },
            )
        ax.scatter(
            [marker_round],
            [max(marker_error, PLOT_FLOOR)],
            marker="D",
            color="#E45756",
            s=28,
            zorder=5,
        )
        ax.set_title(str(cell["display_name"]))
        ax.set_xlim(0, int(append["display_terminal_round"]))
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", linewidth=0.4, alpha=0.28)
        ax.grid(True, which="minor", linewidth=0.22, alpha=0.12)
        if index // 3 == 1:
            ax.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $|\Delta E|$")
    fig.suptitle(
        "Global-singleton RA plateau vs fresh Append-ADAPT singleton - early-stop prefixes",
        fontsize=11.1,
        fontweight="bold",
    )
    fig.legend(
        handles=(
            Line2D(
                [0], [0], color="#4C78A8", marker="o",
                label="Fresh Append-ADAPT singleton (plateau marker)",
            ),
            Line2D(
                [0], [0], color="#E45756", marker="D",
                label="Global-singleton RA (plateau or live-terminal marker)",
            ),
        ),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=7.6,
    )
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _live_cost_tex(ra: Mapping[str, Any]) -> str:
    s_alg = int(ra["algorithmic_work"]["S_alg"])
    if str(ra.get("status", "")).endswith("stdout_partial"):
        work_round = int(ra["algorithmic_work_prefix_round"])
        return (
            rf"$S_{{\rm alg}}@k{work_round}={s_alg:,}$; current/Qiskit pending"
        )
    return rf"$S_{{\rm alg}}={s_alg:,}$; Qiskit pending"


def write_page_tex(adapter: Mapping[str, Any], *, plot_pdf: Path, tex_path: Path) -> None:
    endpoint_rows: list[str] = []
    matched_rows: list[str] = []
    for cell in adapter["cells"]:
        regime = str(cell["regime_id"])
        ra = cell["ra"]
        append_terminal = cell["append"]["terminal"]
        append_round = int(cell["append"]["display_terminal_round"])
        if cell["status"] == "complete":
            ra_terminal = ra["terminal"]
            ra_state = f"complete k={int(ra_terminal['round'])}"
            ra_cost = completed.format_costs(ra_terminal["costs"])
            common = cell["common_accuracy"]
            matched_rows.append(
                " & ".join(
                    (
                        completed.REGIME_ABBREVIATIONS[regime],
                        live._format_delta_e(float(common["target_delta_e"])),
                        str(common["ra"]["round"]),
                        completed.format_costs(common["ra"]["costs"]),
                        str(common["append"]["round"]),
                        completed.format_costs(common["append"]["costs"]),
                    )
                )
                + r" \\"
            )
        else:
            ra_terminal = ra["terminal"]
            state_prefix = (
                "stopped stdout"
                if str(ra.get("status", "")).endswith("stdout_partial")
                else "authenticated"
            )
            ra_state = f"{state_prefix} k={int(ra_terminal['round'])}"
            ra_cost = _live_cost_tex(ra)
            matched_rows.append(
                " & ".join(
                    (
                        completed.REGIME_ABBREVIATIONS[regime],
                        r"$\text{nonterminal}$",
                        str(ra_terminal["round"]),
                        r"$\text{current costs pending}$",
                        "--",
                        rf"$\text{{Append shown to }}k={append_round}$",
                    )
                )
                + r" \\"
            )
        endpoint_rows.append(
            " & ".join(
                (
                    str(cell["display_name"]),
                    ra_state,
                    live._format_delta_e(float(ra_terminal["delta_e"])),
                    ra_cost,
                    str(append_round),
                    live._format_delta_e(float(append_terminal["delta_e"])),
                    completed.format_costs(append_terminal["costs"]),
                )
            )
            + r" \\"
        )
    plot_argument = completed.latex_escape(plot_pdf.resolve().as_posix())
    route = completed.latex_escape(str(adapter["route_description"]))
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.14in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.96\textwidth,height=3.28in,keepaspectratio]{{{plot_argument}}}
\vspace{{0.12em}}

\fontsize{{6.25}}{{6.65}}\selectfont
\setlength{{\tabcolsep}}{{1.9pt}}
\renewcommand{{\arraystretch}}{{0.82}}
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
Regime & RA state & $|\Delta E^{{\rm RA}}|$ & RA prefix cost & $k_A$ &
$|\Delta E^{{\rm Append}}|$ & $C^{{\rm Append}}$ \\
\midrule
{chr(10).join(endpoint_rows)}
\bottomrule
\end{{tabular}}
\vspace{{0.08em}}

{{\scriptsize\bfseries Equal-attainable-error costs for complete trajectories only}}
\vspace{{-0.12em}}

\fontsize{{6.0}}{{6.4}}\selectfont
\setlength{{\tabcolsep}}{{1.7pt}}
\begin{{tabular}}{{@{{}}ccrrrr@{{}}}}
\toprule
Reg. & $|\Delta E_\cap|$ & $k_\cap^{{\rm RA}}$ & $C_\cap^{{\rm RA}}$ &
$k_\cap^{{\rm Append}}$ & $C_\cap^{{\rm Append}}$ \\
\midrule
{chr(10).join(matched_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
\vspace{{-0.40em}}
\tiny
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$. The nph=7 RA rows are
early-stop scheduler stdout trajectories through k=61, k=49, and k=38, joined
to authenticated checkpoint prefixes k=49, k=45, and k=31. The graceful
termination timed out before checkpoint transfer, so current-prefix Qiskit and
$S_{{\rm alg}}$ fields are not inferred. Complete rows retain the source-locked Table-I compiler
(optimization level 0, seed 7, reference state included). {route}
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def build_assets(adapter: Mapping[str, Any]) -> dict[str, Path]:
    assets = {
        "plot_png": REPORT_DIR / f"{ASSET_STEM}_plot.png",
        "plot_pdf": REPORT_DIR / f"{ASSET_STEM}_plot.pdf",
        "page_tex": REPORT_DIR / f"{ASSET_STEM}.tex",
        "page_pdf": REPORT_DIR / f"{ASSET_STEM}.pdf",
    }
    render_plot(adapter, png_path=assets["plot_png"], pdf_path=assets["plot_pdf"])
    write_page_tex(adapter, plot_pdf=assets["plot_pdf"], tex_path=assets["page_tex"])
    completed.legacy_page._compile_page(assets["page_tex"], assets["page_pdf"])
    return assets


def update_master(adapter: Mapping[str, Any], assets: Mapping[str, Path]) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    provenance = _load(MASTER_PROVENANCE)
    outputs = provenance.get("outputs")
    report = provenance.get(REPORT_KEY)
    layout = provenance.get("layout")
    if not isinstance(outputs, dict) or not isinstance(report, dict) or not isinstance(layout, dict):
        raise UpdateError("master provenance structure drifted")
    expected_pdf = outputs.get("partial_progress_pdf")
    if (
        not isinstance(expected_pdf, dict)
        or expected_pdf.get("sha256") != live.sha256_file(MASTER_PDF)
        or expected_pdf.get("size_bytes") != MASTER_PDF.stat().st_size
        or layout.get("page_count") != 8
        or layout.get("regime_count_per_page") != 6
        or report.get("adapter", {}).get("canonical_sha256")
        != SOURCE_ADAPTER_CANONICAL_SHA256
    ):
        raise UpdateError("master PDF/provenance binding drifted")
    before_hashes = completed.legacy_page._page_content_hashes(MASTER_PDF)
    page_reader = PdfReader(str(assets["page_pdf"]), strict=False)
    master_reader = PdfReader(str(MASTER_PDF), strict=False)
    if len(before_hashes) != 8 or len(page_reader.pages) != 1 or len(master_reader.pages) != 8:
        raise UpdateError("master/page asset cardinality drifted")
    writer = PdfWriter()
    for page in master_reader.pages[:6]:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])
    writer.add_page(master_reader.pages[7])
    stage_pdf = MASTER_PDF.with_name(f".{MASTER_PDF.name}.r70-live-stage")
    stage_provenance = MASTER_PROVENANCE.with_name(
        f".{MASTER_PROVENANCE.name}.r70-live-stage"
    )
    backup_pdf = MASTER_PDF.with_name(f".{MASTER_PDF.name}.r70-live-backup")
    backup_provenance = MASTER_PROVENANCE.with_name(
        f".{MASTER_PROVENANCE.name}.r70-live-backup"
    )
    transaction_paths = (stage_pdf, stage_provenance, backup_pdf, backup_provenance)
    if any(path.exists() or path.is_symlink() for path in transaction_paths):
        raise UpdateError("stale r70 live transaction file exists")
    try:
        with stage_pdf.open("xb") as stream:
            writer.write(stream)
        after_hashes = completed.legacy_page._page_content_hashes(stage_pdf)
        if (
            len(after_hashes) != 8
            or after_hashes[:6] != before_hashes[:6]
            or after_hashes[7] != before_hashes[7]
            or after_hashes[6] == before_hashes[6]
        ):
            raise UpdateError("page replacement changed content outside page 7")
        updated = copy.deepcopy(provenance)
        pdf_binding = live.file_binding(stage_pdf)
        pdf_binding["path"] = str(MASTER_PDF.resolve())
        updated["outputs"]["partial_progress_pdf"] = pdf_binding
        output_map = {
            "historical_mean_global_singleton_full6_plot_png": "plot_png",
            "historical_mean_global_singleton_full6_plot_pdf": "plot_pdf",
            "historical_mean_global_singleton_full6_page_tex": "page_tex",
            "historical_mean_global_singleton_full6_page_pdf": "page_pdf",
        }
        for output_key, asset_key in output_map.items():
            updated["outputs"][output_key] = live.file_binding(assets[asset_key])
        cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
        updated_report = copy.deepcopy(report)
        updated_report["schema"] = adapter["schema"]
        updated_report["adapter"] = {
            **live.file_binding(OUTPUT_ADAPTER),
            "canonical_sha256": adapter["sha256"],
        }
        updated_report["prior_adapter"] = copy.deepcopy(report.get("adapter"))
        updated_report["live_horizons"] = {
            regime: int(cells[regime]["ra"]["live_controller_round"])
            for regime in completed.NPH7_REGIMES
        }
        updated_report["cells"] = [live._report_cell(cell) for cell in adapter["cells"]]
        updated_report["marker_policy"] = copy.deepcopy(adapter["marker_policy"])
        updated_report["cost_policy"] = copy.deepcopy(adapter["cost_policy"])
        updated_report["limitations"] = copy.deepcopy(adapter["limitations"])
        updated_report["runtime_progress_source"] = {
            regime: copy.deepcopy(
                cells[regime]["ra"]["source"]["r70_continuation"]
            )
            for regime in LIVE_REGIMES
        }
        updated_report["plot_provenance"] = copy.deepcopy(adapter["plot_provenance"])
        updated_report["structural_validation"] = {
            "pages": 8,
            "preserved_pages_1_6_content_sha256": before_hashes[:6],
            "prior_page_7_content_sha256": before_hashes[6],
            "new_page_7_content_sha256": after_hashes[6],
            "preserved_page_8_content_sha256": before_hashes[7],
        }
        updated_report["outputs"] = {
            key: copy.deepcopy(updated["outputs"][key]) for key in output_map
        }
        updated[REPORT_KEY] = updated_report
        updated["layout"]["page_7"] = (
            "historical_mean_global_singleton_vs_append_mixed_horizon_"
            "six_regime_v4_r70_early_stop_stdout"
        )
        updated["limitations"] = [
            item
            for item in updated.get("limitations", ())
            if not str(item).startswith("Page 7 is a supplemental live diagnostic.")
        ] + [LIMITATION]
        completed.legacy_page._atomic_write_json(stage_provenance, updated)
        staged = _load(stage_provenance)
        if (
            staged["outputs"]["partial_progress_pdf"]["sha256"]
            != live.sha256_file(stage_pdf)
            or staged["outputs"]["partial_progress_pdf"]["size_bytes"]
            != stage_pdf.stat().st_size
        ):
            raise UpdateError("staged master PDF/provenance binding failed")
        shutil.copy2(MASTER_PDF, backup_pdf)
        shutil.copy2(MASTER_PROVENANCE, backup_provenance)
        os.replace(stage_pdf, MASTER_PDF)
        os.replace(stage_provenance, MASTER_PROVENANCE)
        final_hashes = completed.legacy_page._page_content_hashes(MASTER_PDF)
        final_provenance = _load(MASTER_PROVENANCE)
        if (
            final_hashes != after_hashes
            or final_provenance["outputs"]["partial_progress_pdf"]["sha256"]
            != live.sha256_file(MASTER_PDF)
        ):
            raise UpdateError("published master pair failed validation")
    except BaseException:
        if backup_pdf.exists():
            os.replace(backup_pdf, MASTER_PDF)
        if backup_provenance.exists():
            os.replace(backup_provenance, MASTER_PROVENANCE)
        raise
    finally:
        for path in transaction_paths:
            path.unlink(missing_ok=True)
    return {
        "status": "updated",
        "master_pdf": str(MASTER_PDF),
        "master_pdf_sha256": live.sha256_file(MASTER_PDF),
        "master_provenance": str(MASTER_PROVENANCE),
        "adapter": str(OUTPUT_ADAPTER),
        "adapter_canonical_sha256": adapter["sha256"],
        "page_count": 8,
        "updated_page": 7,
        "preserved_pages": [1, 2, 3, 4, 5, 6, 8],
        "live_horizons": {
            regime: int(
                next(
                    cell["ra"]["live_controller_round"]
                    for cell in adapter["cells"]
                    if cell["regime_id"] == regime
                )
            )
            for regime in completed.NPH7_REGIMES
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and construct the bounded adapter in memory only.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    adapter = build_adapter(write=not args.validate_only)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "adapter_canonical_sha256": adapter["sha256"],
                    "live_horizons": {
                        cell["regime_id"]: cell["ra"]["live_controller_round"]
                        for cell in adapter["cells"]
                        if cell["regime_id"] in LIVE_REGIMES
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    assets = build_assets(adapter)
    print(json.dumps(update_master(adapter, assets), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
