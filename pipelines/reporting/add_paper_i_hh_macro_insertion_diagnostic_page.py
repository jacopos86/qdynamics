#!/usr/bin/env python3
"""Append the six-regime macro insertion-search diagnostic to the review PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
FINAL_STEM = "paper_i_hh_macro_common_accuracy_20260723"
FINAL_PDF = OUTPUT_DIR / f"{FINAL_STEM}.pdf"
BASE_REVIEW_PDF = OUTPUT_DIR / f"{FINAL_STEM}_review6.pdf"
PAGE_STEM = f"{FINAL_STEM}_macro_commutation_reduced_insertion_page7"
PAGE_PNG = OUTPUT_DIR / f"{PAGE_STEM}_plot.png"
PAGE_TEX = OUTPUT_DIR / f"{PAGE_STEM}.tex"
PAGE_PDF = OUTPUT_DIR / f"{PAGE_STEM}.pdf"
REVIEW_TEX = OUTPUT_DIR / f"{FINAL_STEM}_review7.tex"
REVIEW_PDF = OUTPUT_DIR / f"{FINAL_STEM}_review7.pdf"
PROVENANCE = OUTPUT_DIR / f"{FINAL_STEM}_macro_commutation_reduced_insertion_provenance.json"
MAIN_PROVENANCE = OUTPUT_DIR / f"{FINAL_STEM}_provenance.json"

TRACKER = REPO_ROOT / (
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
)
NON_STRONG_U_CAMPAIGN = REPO_ROOT / (
    "raw_outputs/"
    "paper_i_hh_sr_snake_macro_commutation_reduced_insertion_"
    "non_strong_u_r15_r20_20260724_v1"
)
STRONG_U_CAMPAIGN = REPO_ROOT / (
    "raw_outputs/"
    "paper_i_hh_sr_snake_macro_commutation_reduced_insertion_"
    "strong_u_r15_20260724_v1"
)
HARVESTED_CAMPAIGN = REPO_ROOT / (
    "raw_outputs/"
    "paper_i_hh_sr_snake_macro_commutation_reduced_insertion_"
    "all_six_r50_20260724_v1_chtc"
)
HARVESTED_REGIMES = frozenset(
    {"intermediate_weak", "strong_weak_u8", "strong_strong_u8"}
)
AUDITS = (
    NON_STRONG_U_CAMPAIGN / "source_lock/source_locked_sensitivity_audit.json",
    STRONG_U_CAMPAIGN / "source_lock/source_locked_sensitivity_audit.json",
)
BASELINE_ROUTE = "sr_macro_physical_lanes_nph3_7"
APPEND_ROUTE = "append_adapt_macro_nph3_7"
EXPECTED_PROFILE_SHA256 = (
    "9086af07111a0b233da798ddce9a6082d5627d5c3753fd437610fefb003ddcbb"
)
REGIMES = (
    ("weak_weak", "Weak--weak", 15, NON_STRONG_U_CAMPAIGN),
    ("intermediate_weak", "Intermediate--weak", 15, NON_STRONG_U_CAMPAIGN),
    ("strong_weak_u8", "Strong--weak", 15, STRONG_U_CAMPAIGN),
    ("weak_strong", "Weak--strong", 20, NON_STRONG_U_CAMPAIGN),
    ("intermediate_strong", "Intermediate--strong", 20, NON_STRONG_U_CAMPAIGN),
    ("strong_strong_u8", "Strong--strong", 15, STRONG_U_CAMPAIGN),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_tex(tex_path: Path) -> Path:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(tex_path.parent),
        str(tex_path),
    ]
    for _ in range(2):
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    return pdf


def load_rows(
    *,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tracker = json.loads(TRACKER.read_text(encoding="utf-8"))
    route = next(item for item in tracker["routes"] if item["id"] == BASELINE_ROUTE)
    append_route = next(
        item for item in tracker["routes"] if item["id"] == APPEND_ROUTE
    )
    rows: list[dict[str, Any]] = []
    for regime, title, horizon, campaign in REGIMES:
        if regime in HARVESTED_REGIMES:
            result_path = HARVESTED_CAMPAIGN / regime / "json/current.json"
        else:
            result_path = campaign / regime / "json/result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        adapt = result["adapt_vqe"]
        actual_horizon = len(adapt["history"])
        pointer = result.get("checkpoint", {}).get(
            "estimator_call_ledger_checkpoint"
        )
        if isinstance(pointer, dict) and pointer.get("path"):
            ledger_path = result_path.parent / str(pointer["path"])
        else:
            ledger_path = campaign / regime / "json/estimator_call_ledger.json"
        if allow_partial and (not result_path.is_file() or not ledger_path.is_file()):
            continue
        if int(adapt["ansatz_depth"]) != actual_horizon:
            raise RuntimeError(f"{regime}: history/depth mismatch")
        if regime not in HARVESTED_REGIMES and actual_horizon != horizon:
            raise RuntimeError(f"{regime}: expected depth {horizon}")
        profile_sha = adapt.get(
            "sr_route_profile_contract_sha256",
            result.get("settings", {}).get("sr_route_profile_contract_sha256"),
        )
        execution_settings = (
            result.get("settings", {})
            .get("sr_route_profile_contract", {})
            .get("execution_settings", {})
        )
        insertion_mode = adapt.get(
            "adapt_insertion_mode",
            execution_settings.get("adapt_insertion_mode"),
        )
        if profile_sha != EXPECTED_PROFILE_SHA256:
            raise RuntimeError(f"{regime}: route contract drift")
        if insertion_mode != "full_commutation_reduced":
            raise RuntimeError(f"{regime}: wrong insertion mode")
        if bool(adapt["adapt_beam_enabled"]):
            raise RuntimeError(f"{regime}: beam unexpectedly enabled")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if regime in HARVESTED_REGIMES:
            pointer_hash = str(pointer.get("sha256") or "")
            if pointer_hash != sha256(ledger_path):
                raise RuntimeError(f"{regime}: checkpoint ledger hash drift")
            if (
                pointer.get("status") != "complete"
                or pointer.get("current_round_finalized") is not True
            ):
                raise RuntimeError(f"{regime}: checkpoint ledger is not closed")
        elif (
            ledger.get("adapt_success") is not True
            or ledger.get("adapt_error") is not None
        ):
            raise RuntimeError(f"{regime}: estimator ledger did not close successfully")

        insertion = [
            {
                "iteration": int(point["depth"]),
                "energy": float(point["energy_after_opt"]),
                "delta_E": float(point["delta_abs_current"]),
                "selected_position": int(point["selected_position"]),
                "selected_operator": str(point["selected_op"]),
                "physical_operator_lane": point.get("physical_operator_lane"),
            }
            for point in adapt["history"]
        ]
        baseline = [
            {
                "iteration": int(point["round"]),
                "delta_E": float(point["error"]),
            }
            for point in route["results"][regime]["trajectory"]
        ]
        append_adapt = [
            {
                "iteration": int(point["round"]),
                "delta_E": float(point["error"]),
            }
            for point in append_route["results"][regime]["trajectory"]
        ]
        if (
            len(insertion) != actual_horizon
            or len(baseline) < 50
            or len(append_adapt) < 50
        ):
            raise RuntimeError(f"{regime}: incomplete per-iteration trajectory")
        baseline = baseline[:50]
        append_adapt = append_adapt[:50]
        baseline_terminal = baseline[-1]["delta_E"]
        append_adapt_terminal = append_adapt[-1]["delta_E"]
        insertion_terminal = insertion[-1]["delta_E"]
        rows.append(
            {
                "regime": regime,
                "title": title,
                "horizon": actual_horizon,
                "endpoint_kind": (
                    "retrieved_finalized_checkpoint"
                    if regime in HARVESTED_REGIMES
                    else "completed_local_horizon"
                ),
                "source_horizon": 50,
                "insertion": insertion,
                "append_only": baseline,
                "append_adapt": append_adapt,
                "terminal": {
                    "insertion_delta_E": insertion_terminal,
                    "append_only_delta_E": baseline_terminal,
                    "append_adapt_delta_E": append_adapt_terminal,
                    "insertion_over_append_only": (
                        insertion_terminal / baseline_terminal
                    ),
                    "relative_error_reduction": (
                        1.0 - insertion_terminal / baseline_terminal
                    ),
                },
                "result_json": {
                    "path": str(result_path.relative_to(REPO_ROOT)),
                    "sha256": sha256(result_path),
                },
                "estimator_ledger": {
                    "path": str(ledger_path.relative_to(REPO_ROOT)),
                    "sha256": sha256(ledger_path),
                },
            }
        )
    if not rows:
        raise RuntimeError("no completed insertion trajectories were found")
    return rows, tracker


def style_axes(ax: mpl.axes.Axes, *, horizon: int) -> None:
    ax.set_yscale("log")
    ax.set_xlim(1, horizon)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55)
    ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(axis="both", labelsize=8)


def make_plot(rows: list[dict[str, Any]]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    if len(rows) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.15), dpi=300)
        axes_iter = np.asarray(axes).flat
    elif len(rows) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(7.55, 5.2), dpi=300)
        axes_iter = np.asarray(axes).flat
    elif len(rows) == 6:
        fig, axes = plt.subplots(2, 3, figsize=(7.75, 4.95), dpi=300)
        axes_iter = np.asarray(axes).flat
    else:
        raise RuntimeError(f"unsupported completed-row count: {len(rows)}")
    for ax, row in zip(axes_iter, rows, strict=True):
        horizon = int(row["horizon"])
        style_axes(ax, horizon=50)
        append_x = [point["iteration"] for point in row["append_only"]]
        append_y = [point["delta_E"] for point in row["append_only"]]
        insertion_x = [point["iteration"] for point in row["insertion"]]
        insertion_y = [point["delta_E"] for point in row["insertion"]]
        append_adapt_x = [point["iteration"] for point in row["append_adapt"]]
        append_adapt_y = [point["delta_E"] for point in row["append_adapt"]]
        ax.plot(
            append_x,
            append_y,
            color="#7F7F7F",
            linewidth=1.6,
            linestyle=(0, (4, 2)),
            solid_capstyle="round",
        )
        ax.plot(
            insertion_x,
            insertion_y,
            color="#E45756",
            linewidth=2.15,
            solid_capstyle="round",
        )
        ax.plot(
            append_adapt_x,
            append_adapt_y,
            color="#4C78A8",
            linewidth=1.55,
            linestyle=(0, (1.2, 2.1)),
            solid_capstyle="round",
        )
        ax.scatter(
            [append_x[-1]],
            [append_y[-1]],
            color="#7F7F7F",
            marker="o",
            s=34,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.scatter(
            [insertion_x[-1]],
            [insertion_y[-1]],
            color="#E45756",
            marker="*",
            s=72,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.scatter(
            [append_adapt_x[-1]],
            [append_adapt_y[-1]],
            color="#4C78A8",
            marker="s",
            s=29,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        terminal = row["terminal"]
        ax.text(
            0.98,
            0.96,
            (
                rf"$\Delta E_{{\rm ins}}={terminal['insertion_delta_E']:.2e}$"
                "\n"
                rf"$\Delta E_{{\rm app}}(50)={terminal['append_only_delta_E']:.2e}$"
                "\n"
                rf"$\Delta E_{{\rm ADAPT}}(50)={terminal['append_adapt_delta_E']:.2e}$"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.4 if len(rows) == 6 else 7.2,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.92,
            },
        )
        ax.set_title(row["title"], fontsize=9.2 if len(rows) == 6 else 10, pad=3)
        ax.set_xlabel("ADAPT iteration", fontsize=8.2 if len(rows) == 6 else 9)
        ax.set_ylabel(
            r"Energy error $\Delta E$", fontsize=8.2 if len(rows) == 6 else 9
        )

    handles = [
        Line2D(
            [0],
            [0],
            color="#E45756",
            linewidth=2.15,
            marker="*",
            markersize=8,
            label="SNAKE: commutation-reduced insertion",
        ),
        Line2D(
            [0],
            [0],
            color="#7F7F7F",
            linewidth=1.6,
            linestyle=(0, (4, 2)),
            marker="o",
            markersize=5,
            label="SNAKE: append-only source trajectory",
        ),
        Line2D(
            [0],
            [0],
            color="#4C78A8",
            linewidth=1.55,
            linestyle=(0, (1.2, 2.1)),
            marker="s",
            markersize=4.5,
            label="Append-ADAPT macro",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.13 if len(rows) == 2 else 0.085,
        top=0.82 if len(rows) == 2 else 0.91,
        hspace=0.38 if len(rows) == 6 else 0.34,
        wspace=0.34 if len(rows) == 6 else 0.25,
    )
    fig.savefig(PAGE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_page_tex(rows: list[dict[str, Any]]) -> None:
    scope = (
        "the two completed weak-Holstein regimes"
        if len(rows) == 2
        else (
            "the four non-strong-Hubbard regimes"
            if len(rows) == 4
            else "all six Hubbard--Holstein interaction regimes"
        )
    )
    PAGE_TEX.write_text(
        rf"""
\documentclass[letterpaper]{{article}}
\usepackage[margin=0.35in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries SNAKE macro insertion-position diagnostic}}\par
\vspace{{2pt}}
{{\small Per-iteration same-cutoff energy error for {scope}.
The insertion route scores one representative from each
termwise-commuting position class and otherwise preserves the macro SNAKE
configuration.  Archived append-only trajectories are shown through
iteration 50; insertion trajectories end at the completed local horizon or
the retrieved finalized CHTC checkpoint used for this review.}}\par
\vspace{{5pt}}
\includegraphics[width=7.75in]{{{PAGE_PNG.as_posix()}}}
\vspace{{2pt}}

{{\footnotesize Diagnostic comparison: the insertion runs use the current
working-tree route and the append-only curves use the locked source
trajectories.  No additional append-only replay anchor was run.}}
\end{{center}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def assemble_review() -> None:
    if len(PdfReader(str(BASE_REVIEW_PDF)).pages) != 6:
        raise RuntimeError("preserved review input is not six pages")
    REVIEW_TEX.write_text(
        rf"""
\documentclass[letterpaper]{{article}}
\usepackage{{pdfpages}}
\pagestyle{{empty}}
\begin{{document}}
\includepdf[pages=-,pagecommand={{}}]{{{BASE_REVIEW_PDF.as_posix()}}}
\includepdf[pages=-,pagecommand={{}}]{{{PAGE_PDF.as_posix()}}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    compiled = compile_tex(REVIEW_TEX)
    if compiled != REVIEW_PDF:
        raise RuntimeError("unexpected review output path")
    if len(PdfReader(str(REVIEW_PDF)).pages) != 7:
        raise RuntimeError("assembled review is not seven pages")
    shutil.copy2(REVIEW_PDF, FINAL_PDF)


def write_provenance(rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema": "paper_i_hh_macro_commutation_reduced_insertion_diagnostic_v1",
        "status": (
            "review_diagnostic_partial_weak_pair"
            if len(rows) == 2
            else (
                "review_diagnostic_complete_four_regimes"
                if len(rows) == 4
                else "review_diagnostic_complete_six_regimes"
            )
        ),
        "comparison": {
            "new_route": "commutation-reduced full-position macro SNAKE",
            "baseline_route": BASELINE_ROUTE,
            "append_adapt_route": APPEND_ROUTE,
            "energy_error": "absolute error against the same-cutoff exact ground-state energy",
            "rows": rows,
        },
        "source_lock": {
            "audits": [
                {
                    "path": str(audit.relative_to(REPO_ROOT)),
                    "sha256": sha256(audit),
                }
                for audit in AUDITS
            ],
            "anchor_reproduces_source": None,
            "interpretation": (
                "Diagnostic comparison against locked append-only trajectories; "
                "not a strict one-variable replay because no new append-only anchor "
                "was included in the user-approved insertion campaigns."
            ),
        },
        "inputs": {
            "tracker": {
                "path": str(TRACKER.relative_to(REPO_ROOT)),
                "sha256": sha256(TRACKER),
            },
            "base_review_pdf": {
                "path": str(BASE_REVIEW_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(BASE_REVIEW_PDF),
                "pages": 6,
            },
        },
        "generated": {
            "page_plot": {
                "path": str(PAGE_PNG.relative_to(REPO_ROOT)),
                "sha256": sha256(PAGE_PNG),
            },
            "page_tex": {
                "path": str(PAGE_TEX.relative_to(REPO_ROOT)),
                "sha256": sha256(PAGE_TEX),
            },
            "page_pdf": {
                "path": str(PAGE_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(PAGE_PDF),
            },
            "review_tex": {
                "path": str(REVIEW_TEX.relative_to(REPO_ROOT)),
                "sha256": sha256(REVIEW_TEX),
            },
            "review_pdf": {
                "path": str(REVIEW_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(REVIEW_PDF),
                "pages": 7,
            },
            "canonical_pdf": {
                "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(FINAL_PDF),
                "pages": 7,
            },
        },
        "validation": {
            "page_count": 7,
            "page_size": "letter",
            "existing_pages_preserved_from": str(BASE_REVIEW_PDF.relative_to(REPO_ROOT)),
            "rendered_pages_inspected": [],
        },
    }
    PROVENANCE.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if MAIN_PROVENANCE.is_file():
        main = json.loads(MAIN_PROVENANCE.read_text(encoding="utf-8"))
        main["macro_commutation_reduced_insertion_diagnostic"] = payload["comparison"]
        main["generated"]["pdf"] = payload["generated"]["canonical_pdf"]
        main["validation"]["page_count"] = 7
        main["validation"]["rendered_pages_inspected"] = []
        MAIN_PROVENANCE.write_text(
            json.dumps(main, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="build from the completed weak pair while the strong pair is running",
    )
    args = parser.parse_args()
    rows, _tracker = load_rows(allow_partial=args.allow_partial)
    make_plot(rows)
    write_page_tex(rows)
    compiled_page = compile_tex(PAGE_TEX)
    if compiled_page != PAGE_PDF or len(PdfReader(str(PAGE_PDF)).pages) != 1:
        raise RuntimeError("diagnostic page compilation failed")
    assemble_review()
    write_provenance(rows)
    print(FINAL_PDF)


if __name__ == "__main__":
    main()
