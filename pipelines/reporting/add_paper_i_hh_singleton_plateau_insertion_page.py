#!/usr/bin/env python3
"""Append the repaired weak--weak singleton plateau-insertion diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from pypdf import PdfReader, PdfWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_macro_common_accuracy_pdf import (  # noqa: E402
    _compile_comparator_at_k,
    _selection,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (  # noqa: E402
    _read_source_result,
    _snake_prefix,
)
from pipelines.reporting.paper_i_qiskit_cost_tuple import (  # noqa: E402
    PAPER_I_QISKIT_COST_TUPLE_LATEX,
    paper_i_cost_tuple_latex,
)

OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
STEM = "paper_i_hh_macro_common_accuracy_20260723"
FINAL_PDF = OUTPUT_DIR / f"{STEM}.pdf"
BACKUP_PDF = OUTPUT_DIR / f"{STEM}_pre_singleton_plateau_insertion_page10.pdf"
W1Q_BASE_PDF = OUTPUT_DIR / f"{STEM}_pre_w1q_singleton_plateau_insertion_page10.pdf"
PAGE_STEM = f"{STEM}_singleton_plateau_insertion_page10"
PAGE_PNG = OUTPUT_DIR / f"{PAGE_STEM}_plot.png"
PAGE_TEX = OUTPUT_DIR / f"{PAGE_STEM}.tex"
PAGE_PDF = OUTPUT_DIR / f"{PAGE_STEM}.pdf"
PAGE_PROVENANCE = OUTPUT_DIR / f"{PAGE_STEM}_provenance.json"
MAIN_PROVENANCE = OUTPUT_DIR / f"{STEM}_provenance.json"
TRACKER = REPO_ROOT / (
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
)
INSERTION_CURRENT = REPO_ROOT / (
    "raw_outputs/"
    "paper_i_hh_sr_snake_singleton_insertion_commutation_plateau_"
    "weak_weak_r50_20260725_v2_local_position_fix/json/current.json"
)

SNAKE_ROUTE = "no_overlap_trust_projected_phase3_nph3_7"
APPEND_ROUTE = "append_adapt_projected_singleton_nph3_7"
DISPLAY_K = 40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_content_hashes(path: Path) -> list[str]:
    hashes: list[str] = []
    for page in PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        payload = b"" if contents is None else contents.get_data()
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def _trajectory(route: Mapping[str, Any]) -> list[dict[str, float]]:
    points = route["results"]["weak_weak"]["trajectory"]
    return [
        {"k": int(point["round"]), "error": float(point["error"])}
        for point in points
        if int(point["round"]) <= DISPLAY_K
    ]


def _compact_s_alg(value: int) -> str:
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"{mantissa:.2g}\!\times\!10^{{{exponent}}}"


def _load_data() -> dict[str, Any]:
    tracker = json.loads(TRACKER.read_text(encoding="utf-8"))
    routes = {str(route["id"]): route for route in tracker["routes"]}
    main = json.loads(MAIN_PROVENANCE.read_text(encoding="utf-8"))
    insertion = json.loads(INSERTION_CURRENT.read_text(encoding="utf-8"))

    existing_rows = [
        row
        for row in main["singleton_own_plateau_common_accuracy"]["rows"]
        if row["regime"] == "weak_weak"
    ]
    existing = {str(row["method"]): row for row in existing_rows}
    snake_row = existing["snake_singleton"]
    append_page4_row = existing["append_singleton"]
    if (
        int(snake_row["k_cross"]) != 29
        or int(snake_row["S_alg"]) != 40510
        or int(append_page4_row["k_cross"]) != 26
    ):
        raise RuntimeError("page-4 singleton provenance drift")

    snake_qiskit = {
        key: snake_row.get(key)
        for key in ("N2q", "D2q", "Dc", "W1q", "B1q")
    }
    if snake_qiskit["W1q"] is None:
        snake_result = routes[SNAKE_ROUTE]["results"]["weak_weak"]
        snake_source = snake_result["source"]
        snake_payload, _runtime_seed, snake_source_receipt = _read_source_result(
            snake_source,
            need_runtime_seed=False,
        )
        snake_prefix = _snake_prefix(
            snake_payload,
            selection=_selection(
                trajectory=snake_result["trajectory"],
                k=int(snake_row["k_cross"]),
            ),
            source=snake_source_receipt,
            route_id=SNAKE_ROUTE,
            fallback_source_kind=(
                "paper_i_hh_singleton_append_only_page10_reference"
            ),
        )
        snake_qiskit = dict(snake_prefix["qiskit"])

    adapt = insertion["adapt_vqe"]
    history = adapt["history"]
    if len(history) < DISPLAY_K:
        raise RuntimeError("plateau-insertion history does not reach k=40")
    insertion_points = [
        {
            "k": int(point["depth"]),
            "error": float(point["delta_abs_current"]),
        }
        for point in history[:DISPLAY_K]
    ]
    insertion_row = history[DISPLAY_K - 1]
    insertion_error = float(insertion_row["delta_abs_current"])
    expected_error = 2.7783320089014296e-10
    if not math.isclose(insertion_error, expected_error, rel_tol=0.0, abs_tol=1e-18):
        raise RuntimeError("repaired insertion k=40 error drift")

    receipt = insertion_row["active_prefix_checkpoint"][
        "estimator_ledger_receipt"
    ]["cumulative_executed_queries"]
    insertion_s_alg = int(receipt["S_alg"])
    if insertion_s_alg != 122691:
        raise RuntimeError("repaired insertion k=40 S_alg drift")

    append_trajectory = _trajectory(routes[APPEND_ROUTE])
    append_k40 = next(point for point in append_trajectory if point["k"] == DISPLAY_K)
    if not math.isclose(
        append_k40["error"],
        5.107216871635956e-10,
        rel_tol=0.0,
        abs_tol=1e-18,
    ):
        raise RuntimeError("Append-ADAPT k=40 error drift")

    insertion_compile_trajectory = [
        {"round": point["k"], "error": point["error"]}
        for point in insertion_points
    ]
    insertion_prefix = _snake_prefix(
        insertion,
        selection=_selection(
            trajectory=insertion_compile_trajectory,
            k=DISPLAY_K,
        ),
        source={
            "path": str(INSERTION_CURRENT.relative_to(REPO_ROOT)),
            "result_member": None,
        },
        route_id=(
            "sr_singleton_plateau_triggered_commutation_reduced_insertion_nph3_7"
        ),
        fallback_source_kind=(
            "paper_i_hh_singleton_plateau_insertion_k40"
        ),
    )
    insertion_qiskit = insertion_prefix["qiskit"]
    if (
        int(insertion_qiskit["N2q"]) != 180
        or int(insertion_qiskit["D2q"]) != 159
        or int(insertion_qiskit["Dc"]) != 700
    ):
        raise RuntimeError("repaired insertion k=40 Qiskit costs drift")

    append_source = routes[APPEND_ROUTE]["results"]["weak_weak"]["source"]
    append_prefix, _append_source_receipt = _compile_comparator_at_k(
        source=append_source,
        trajectory=routes[APPEND_ROUTE]["results"]["weak_weak"]["trajectory"],
        k=DISPLAY_K,
        representation="projected_singleton",
    )
    append_qiskit = append_prefix["qiskit"]
    if (
        int(append_qiskit["N2q"]) != 204
        or int(append_qiskit["D2q"]) != 167
        or int(append_qiskit["Dc"]) != 860
    ):
        raise RuntimeError("Append-ADAPT k=40 Qiskit costs drift")

    return {
        "exact_energy": float(adapt["exact_gs_energy"]),
        "snake": {
            "trajectory": _trajectory(routes[SNAKE_ROUTE]),
            "reported_k": int(snake_row["k_cross"]),
            "reported_error": float(snake_row["crossing_error"]),
            "costs": {
                "N2q": int(snake_qiskit["N2q"]),
                "D2q": int(snake_qiskit["D2q"]),
                "Dc": int(snake_qiskit["Dc"]),
                "W1q": int(snake_qiskit["W1q"]),
                "B1q": snake_qiskit.get("B1q"),
                "S_alg": int(snake_row["S_alg"]),
            },
        },
        "insertion": {
            "trajectory": insertion_points,
            "reported_k": DISPLAY_K,
            "reported_error": insertion_error,
            "costs": {
                "N2q": int(insertion_qiskit["N2q"]),
                "D2q": int(insertion_qiskit["D2q"]),
                "Dc": int(insertion_qiskit["Dc"]),
                "W1q": int(insertion_qiskit["W1q"]),
                "B1q": insertion_qiskit.get("B1q"),
                "S_alg": insertion_s_alg,
            },
            "receipt_components": dict(receipt["components"]),
        },
        "append": {
            "trajectory": append_trajectory,
            "reported_k": DISPLAY_K,
            "reported_error": float(append_k40["error"]),
            "costs": {
                "N2q": int(append_qiskit["N2q"]),
                "D2q": int(append_qiskit["D2q"]),
                "Dc": int(append_qiskit["Dc"]),
                "W1q": int(append_qiskit["W1q"]),
                "B1q": append_qiskit.get("B1q"),
                "S_alg": 273183,
            },
        },
    }


def _with_initial(points: list[dict[str, float]], initial_error: float) -> tuple[list[int], list[float]]:
    return (
        [0, *[int(point["k"]) for point in points]],
        [initial_error, *[float(point["error"]) for point in points]],
    )


def _write_plot(data: Mapping[str, Any]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )
    fig, ax = plt.subplots(figsize=(7.45, 5.15), constrained_layout=True)
    initial_error = abs(1.25 - float(data["exact_energy"]))
    styles = (
        ("snake", "Append-only SNAKE", "#c44e52", "--", "s"),
        ("insertion", "Plateau-insertion SNAKE", "#8b1a1a", "-", "*"),
        ("append", "Append-ADAPT", "#4c72b0", ":", "o"),
    )
    for key, label, color, linestyle, marker in styles:
        row = data[key]
        x, y = _with_initial(row["trajectory"], initial_error)
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
            zorder=2,
        )
        ax.scatter(
            [row["reported_k"]],
            [row["reported_error"]],
            color=color,
            marker=marker,
            s=70 if marker == "*" else 38,
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_xlim(0, 46)
    ax.set_ylim(1e-11, 4)
    ax.set_xlabel("ADAPT iteration")
    ax.set_ylabel(r"Energy error, $\Delta E$")
    ax.set_title("Weak--weak projected-singleton plateau-insertion diagnostic")
    ax.grid(True, which="major", linewidth=0.55, alpha=0.38)
    ax.grid(True, which="minor", linewidth=0.35, alpha=0.18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower left", framealpha=0.96)

    tuple_lines = []
    for key, _label, color, _linestyle, marker in styles:
        costs = data[key]["costs"]
        tuple_lines.append(
            (
                color,
                marker,
                paper_i_cost_tuple_latex(
                    costs,
                    marker="",
                    format_s_alg=_compact_s_alg,
                ),
            )
        )
    ax.text(
        0.985,
        0.975,
        f"${PAPER_I_QISKIT_COST_TUPLE_LATEX}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.94, "pad": 3.5},
    )
    y = 0.905
    for color, marker, text in tuple_lines:
        ax.scatter(
            [0.62],
            [y],
            transform=ax.transAxes,
            color=color,
            marker=marker,
            s=45 if marker == "*" else 28,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            0.655,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=color,
            fontsize=8.7,
        )
        y -= 0.062
    fig.savefig(PAGE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compile_page(data: Mapping[str, Any]) -> None:
    PAGE_TEX.write_text(
        rf"""
\documentclass[10pt,letterpaper]{{article}}
\usepackage[margin=0.62in]{{geometry}}
\usepackage{{amsmath,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.965\textwidth]{{{PAGE_PNG.as_posix()}}}
\end{{center}}
\vspace{{-0.7em}}
\small
The repaired plateau-triggered insertion route follows append-only SNAKE until
the weak--weak singleton trajectory stalls, then activates the
commutation-reduced position search.  At iteration \(40\), it reaches
\(\Delta E={data['insertion']['reported_error']:.4e}\), compared with
\(\Delta E={data['append']['reported_error']:.4e}\) for Append-ADAPT at the
same iteration.  The append-only SNAKE point retained from page~4 is its
reported \(k=29\) plateau point.  Tuples report
${PAPER_I_QISKIT_COST_TUPLE_LATEX}$; $W_{{1q}}$ is genuine Qiskit-emitted
Pauli-rotation one-qubit work before transpilation.
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(OUTPUT_DIR),
        str(PAGE_TEX),
    ]
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if len(PdfReader(str(PAGE_PDF)).pages) != 1:
        raise RuntimeError("plateau-insertion page PDF is not one page")


def _append_page() -> dict[str, Any]:
    if len(PdfReader(str(FINAL_PDF)).pages) != 9:
        raise RuntimeError("expected the current report to contain nine pages")
    if not W1Q_BASE_PDF.is_file():
        shutil.copy2(FINAL_PDF, W1Q_BASE_PDF)
    if len(PdfReader(str(W1Q_BASE_PDF)).pages) != 9:
        raise RuntimeError("preserved W1q pre-insertion report is not nine pages")

    before_hashes = _page_content_hashes(W1Q_BASE_PDF)
    writer = PdfWriter()
    for page in PdfReader(str(W1Q_BASE_PDF), strict=False).pages:
        writer.add_page(page)
    writer.add_page(PdfReader(str(PAGE_PDF), strict=False).pages[0])
    temporary = FINAL_PDF.with_name(f".{FINAL_PDF.name}.page10.tmp")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(FINAL_PDF)

    reader = PdfReader(str(FINAL_PDF), strict=False)
    if len(reader.pages) != 10:
        raise RuntimeError("final report does not contain ten pages")
    after_hashes = _page_content_hashes(FINAL_PDF)
    if after_hashes[:9] != before_hashes:
        raise RuntimeError("one or more preserved pages changed during append")
    return {
        "pages_before": 9,
        "pages_after": 10,
        "preserved_page_content_hashes": before_hashes,
        "new_page_content_sha256": after_hashes[9],
    }


def _write_provenance(data: Mapping[str, Any], structural: Mapping[str, Any]) -> None:
    payload = {
        "schema": "paper_i_hh_singleton_plateau_insertion_page_v1",
        "classification": "diagnostic",
        "definition": (
            "Weak-weak projected-singleton comparison at k=40 for repaired "
            "plateau-triggered insertion SNAKE and Append-ADAPT, with the "
            "append-only SNAKE plateau point retained from page 4."
        ),
        "data": data,
        "sources": {
            "tracker": {
                "path": str(TRACKER.relative_to(REPO_ROOT)),
                "sha256": _sha256(TRACKER),
            },
            "insertion_current": {
                "path": str(INSERTION_CURRENT.relative_to(REPO_ROOT)),
                "sha256": _sha256(INSERTION_CURRENT),
            },
            "main_provenance": {
                "path": str(MAIN_PROVENANCE.relative_to(REPO_ROOT)),
                "sha256_before_update": _sha256(MAIN_PROVENANCE),
            },
        },
        "generated": {
            "plot_png": {
                "path": str(PAGE_PNG.relative_to(REPO_ROOT)),
                "sha256": _sha256(PAGE_PNG),
            },
            "page_tex": {
                "path": str(PAGE_TEX.relative_to(REPO_ROOT)),
                "sha256": _sha256(PAGE_TEX),
            },
            "page_pdf": {
                "path": str(PAGE_PDF.relative_to(REPO_ROOT)),
                "sha256": _sha256(PAGE_PDF),
            },
            "combined_pdf": {
                "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
                "sha256": _sha256(FINAL_PDF),
            },
        },
        "structural_validation": dict(structural),
    }
    PAGE_PROVENANCE.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    main = json.loads(MAIN_PROVENANCE.read_text(encoding="utf-8"))
    main["singleton_plateau_insertion_diagnostic"] = payload
    main["generated"]["pdf"] = {
        "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
        "sha256": _sha256(FINAL_PDF),
        "pages": 10,
    }
    MAIN_PROVENANCE.write_text(
        json.dumps(main, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    data = _load_data()
    _write_plot(data)
    _compile_page(data)
    structural = _append_page()
    _write_provenance(data, structural)
    print(
        json.dumps(
            {
                "pdf": str(FINAL_PDF),
                "pages": len(PdfReader(str(FINAL_PDF)).pages),
                "sha256": _sha256(FINAL_PDF),
                "page_provenance": str(PAGE_PROVENANCE),
                "preserved_pages_1_9": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
