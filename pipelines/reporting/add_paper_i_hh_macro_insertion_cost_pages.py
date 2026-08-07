#!/usr/bin/env python3
"""Append matched-prefix and terminal insertion-cost pages to the review PDF."""

from __future__ import annotations

import argparse
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
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from pypdf import PdfReader, PdfWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.paper_i_qiskit_cost_tuple import (  # noqa: E402
    PAPER_I_QISKIT_COST_TUPLE_LATEX,
    paper_i_cost_tuple_latex,
)

OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
STEM = "paper_i_hh_macro_common_accuracy_20260723"
FINAL_PDF = OUTPUT_DIR / f"{STEM}.pdf"
BASE_REVIEW_PDF = OUTPUT_DIR / f"{STEM}_review7.pdf"
PAGES_STEM = f"{STEM}_macro_insertion_cost_pages8_9"
PAGE8_PNG = OUTPUT_DIR / f"{PAGES_STEM}_matched_plot.png"
PAGE9_PNG = OUTPUT_DIR / f"{PAGES_STEM}_terminal_plot.png"
PAGES_TEX = OUTPUT_DIR / f"{PAGES_STEM}.tex"
PAGES_PDF = OUTPUT_DIR / f"{PAGES_STEM}.pdf"
REVIEW_TEX = OUTPUT_DIR / f"{STEM}_review9.tex"
REVIEW_PDF = OUTPUT_DIR / f"{STEM}_review9.pdf"
PROVENANCE = OUTPUT_DIR / f"{STEM}_macro_insertion_cost_provenance.json"
DATA_JSON = OUTPUT_DIR / f"{STEM}_macro_insertion_cost_rows.json"
MAIN_PROVENANCE = OUTPUT_DIR / f"{STEM}_provenance.json"

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
BASELINE_ROUTE = "sr_macro_physical_lanes_nph3_7"
APPEND_ROUTE = "append_adapt_macro_nph3_7"
INSERTION_ROUTE = "sr_macro_commutation_reduced_insertion_nph3_7"
EXPECTED_PROFILE_SHA256 = (
    "9086af07111a0b233da798ddce9a6082d5627d5c3753fd437610fefb003ddcbb"
)
REGIMES = (
    ("weak_weak", "Weak--weak", "WW", 3, 15, NON_STRONG_U_CAMPAIGN),
    ("intermediate_weak", "Intermediate--weak", "IW", 3, 15, NON_STRONG_U_CAMPAIGN),
    ("strong_weak_u8", "Strong--weak", "SW", 3, 15, STRONG_U_CAMPAIGN),
    ("weak_strong", "Weak--strong", "WS", 7, 20, NON_STRONG_U_CAMPAIGN),
    (
        "intermediate_strong",
        "Intermediate--strong",
        "IS",
        7,
        20,
        NON_STRONG_U_CAMPAIGN,
    ),
    ("strong_strong_u8", "Strong--strong", "SS", 7, 15, STRONG_U_CAMPAIGN),
)
METHODS = (
    ("insertion", "Insertion SNAKE"),
    ("snake", "Append-only SNAKE"),
    ("append", "Append-ADAPT"),
)
PAIR_SPECS = (
    (
        "insertion_vs_append",
        "Insertion SNAKE vs Append-ADAPT",
        ("insertion", "append"),
    ),
    (
        "snake_vs_append",
        "Append-only SNAKE vs Append-ADAPT",
        ("snake", "append"),
    ),
)
INSERTION_PLATEAU_ANCHORS = {
    "weak_strong": 9,
    "strong_strong_u8": 11,
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines.reporting.build_paper_i_hh_macro_common_accuracy_pdf import (  # noqa: E402
    _compile_comparator_at_k,
    _existing_prefix,
    _selection,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (  # noqa: E402
    _read_source_result,
    _snake_prefix,
)
from pipelines.exact_bench.paper_i_s_alg_accounting import (  # noqa: E402
    runtime_prefix_work,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _insertion_trajectory(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "round": int(row["depth"]),
            "error": float(row["delta_abs_current"]),
        }
        for row in payload["adapt_vqe"]["history"]
    ]


def _validate_insertion_payload(
    payload: Mapping[str, Any],
    *,
    regime: str,
    expected_horizon: int,
) -> None:
    adapt = payload["adapt_vqe"]
    settings = payload.get("settings", {})
    execution_settings = (
        settings.get("sr_route_profile_contract", {}).get(
            "execution_settings", {}
        )
        if isinstance(settings, Mapping)
        else {}
    )
    if int(adapt["ansatz_depth"]) != expected_horizon:
        raise ValueError(f"{regime}: insertion horizon drift")
    profile_sha = adapt.get(
        "sr_route_profile_contract_sha256",
        settings.get("sr_route_profile_contract_sha256")
        if isinstance(settings, Mapping)
        else None,
    )
    if profile_sha != EXPECTED_PROFILE_SHA256:
        raise ValueError(f"{regime}: insertion route contract drift")
    insertion_mode = adapt.get(
        "adapt_insertion_mode",
        execution_settings.get("adapt_insertion_mode"),
    )
    if insertion_mode != "full_commutation_reduced":
        raise ValueError(f"{regime}: insertion mode drift")
    if bool(adapt["adapt_beam_enabled"]):
        raise ValueError(f"{regime}: beam unexpectedly enabled")


def _insertion_artifacts(
    *,
    regime: str,
    campaign: Path,
) -> tuple[Path, Path, str]:
    if regime in HARVESTED_REGIMES:
        result_path = HARVESTED_CAMPAIGN / regime / "json/current.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        pointer = payload.get("checkpoint", {}).get(
            "estimator_call_ledger_checkpoint"
        )
        if not isinstance(pointer, Mapping) or not pointer.get("path"):
            raise ValueError(f"{regime}: missing finalized checkpoint ledger")
        ledger_path = result_path.parent / str(pointer["path"])
        if sha256(ledger_path) != str(pointer.get("sha256") or ""):
            raise ValueError(f"{regime}: checkpoint ledger hash drift")
        if (
            pointer.get("status") != "complete"
            or pointer.get("current_round_finalized") is not True
        ):
            raise ValueError(f"{regime}: checkpoint ledger is not closed")
        return result_path, ledger_path, "retrieved_finalized_checkpoint"
    return (
        campaign / regime / "json/result.json",
        campaign / regime / "json/estimator_call_ledger.json",
        "completed_local_horizon",
    )


def _runtime_accounted_insertion_prefix(
    *,
    payload: Mapping[str, Any],
    k: int,
    source: Mapping[str, Any],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    compile_payload = dict(payload)
    compile_adapt = dict(payload["adapt_vqe"])
    for key in (
        "active_prefix_checkpoints",
        "terminal_active_prefix_checkpoint",
    ):
        compile_payload.pop(key, None)
        compile_adapt.pop(key, None)
    continuation = compile_adapt.get("continuation")
    if isinstance(continuation, Mapping):
        compile_continuation = dict(continuation)
        compile_continuation.pop("active_prefix_checkpoints", None)
        compile_continuation.pop("terminal_active_prefix_checkpoint", None)
        compile_adapt["continuation"] = compile_continuation
    compile_payload["adapt_vqe"] = compile_adapt
    prefix = _snake_prefix(
        compile_payload,
        selection=_selection(trajectory=trajectory, k=k),
        source=source,
        route_id=INSERTION_ROUTE,
        fallback_source_kind=(
            "paper_i_hh_macro_commutation_reduced_insertion_prefix"
        ),
    )
    history = payload["adapt_vqe"]["history"]
    checkpoint = history[k - 1].get("active_prefix_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("insertion prefix lacks an active-prefix checkpoint")
    receipt = checkpoint.get("estimator_ledger_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("insertion prefix lacks a runtime estimator receipt")
    work = runtime_prefix_work(
        method="SNAKE",
        representation="intact_macro",
        accepted_prefix_length=k,
        estimator_ledger_receipt=receipt,
    )
    prefix["S_alg"] = int(work["S_alg"])
    prefix["S_alg_scope"] = str(work["scope"])
    prefix["S_alg_components"] = dict(work["components"])
    prefix["S_alg_receipt"] = work
    prefix["S_alg_reconstruction_status"] = (
        "closed_runtime_occurrence_ledger"
    )
    prefix["prefix_receipt"]["estimator_ledger_receipt"] = {
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
        "outer_iteration": int(receipt["outer_iteration"]),
        "cumulative_raw_occurrences": receipt.get(
            "cumulative_raw_occurrences"
        ),
    }
    prefix["prefix_receipt"]["S_alg_recount"] = work
    return prefix


def _compact_prefix(prefix: Mapping[str, Any]) -> dict[str, Any]:
    qiskit = prefix["qiskit"]
    return {
        "active_depth": int(prefix["active_depth"]),
        "N2q": int(qiskit["N2q"]),
        "D2q": int(qiskit["D2q"]),
        "Dc": int(qiskit["Dc"]),
        "W1q": int(qiskit["W1q"]),
        "B1q": qiskit.get("B1q"),
        "qiskit_basis_work_status": qiskit[
            "qiskit_basis_work_status"
        ],
        "qiskit_basis_work_schema": qiskit.get(
            "qiskit_basis_work_schema"
        ),
        "S_alg": int(prefix["S_alg"]),
        "S_alg_scope": prefix["S_alg_scope"],
        "S_alg_components": prefix.get("S_alg_components"),
        "S_alg_receipt": prefix.get("S_alg_receipt"),
        "S_alg_reconstruction_status": prefix.get(
            "S_alg_reconstruction_status"
        ),
        "qiskit_compile": prefix["qiskit_compile"],
        "prefix_receipt": prefix["prefix_receipt"],
    }


def collect_rows(
    tracker: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routes = {route["id"]: route for route in tracker["routes"]}
    snake_route = routes[BASELINE_ROUTE]
    append_route = routes[APPEND_ROUTE]
    matched: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []

    for regime, title, abbreviation, n_ph, horizon, campaign in REGIMES:
        result_path, ledger_path, endpoint_kind = _insertion_artifacts(
            regime=regime,
            campaign=campaign,
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        actual_horizon = len(payload["adapt_vqe"]["history"])
        _validate_insertion_payload(
            payload,
            regime=regime,
            expected_horizon=actual_horizon,
        )
        if (
            endpoint_kind == "completed_local_horizon"
            and (
                ledger.get("adapt_success") is not True
                or ledger.get("adapt_error") is not None
            )
        ):
            raise ValueError(f"{regime}: insertion estimator ledger is not closed")

        trajectories = {
            "insertion": _insertion_trajectory(payload),
            "snake": snake_route["results"][regime]["trajectory"],
            "append": append_route["results"][regime]["trajectory"],
        }

        insertion_source = {
            "path": str(result_path.relative_to(REPO_ROOT)),
            "sha256": sha256(result_path),
            "estimator_ledger_path": str(ledger_path.relative_to(REPO_ROOT)),
            "estimator_ledger_sha256": sha256(ledger_path),
        }
        insertion_prefix_cache: dict[int, dict[str, Any]] = {}
        snake_prefix_cache: dict[int, tuple[dict[str, Any], Any, str]] = {}
        append_prefix_cache: dict[int, tuple[dict[str, Any], Any, str]] = {}

        def insertion_prefix(k: int) -> dict[str, Any]:
            if k not in insertion_prefix_cache:
                insertion_prefix_cache[k] = _runtime_accounted_insertion_prefix(
                    payload=payload,
                    k=k,
                    source=insertion_source,
                    trajectory=trajectories["insertion"],
                )
            return insertion_prefix_cache[k]

        def prefix_for_method(
            method: str,
            k: int,
        ) -> tuple[dict[str, Any], Any, str]:
            trajectory = trajectories[method]
            if method == "insertion":
                return (
                    insertion_prefix(k),
                    insertion_source,
                    "signed insertion checkpoint",
                )
            if method == "snake" and k not in snake_prefix_cache:
                existing = _existing_prefix(
                    route=snake_route,
                    regime=regime,
                    k=k,
                )
                if (
                    existing is not None
                    and existing[0].get("qiskit", {}).get("W1q") is None
                ):
                    existing = None
                if existing is not None:
                    prefix, recovery = existing
                    source = snake_route["results"][regime]["source"]
                else:
                    source_config = snake_route["results"][regime]["source"]
                    source_payload, _runtime_seed, source = _read_source_result(
                        source_config,
                        need_runtime_seed=False,
                    )
                    prefix = _snake_prefix(
                        source_payload,
                        selection=_selection(trajectory=trajectory, k=k),
                        source=source,
                        route_id=BASELINE_ROUTE,
                        fallback_source_kind=(
                            "paper_i_hh_snake_pairwise_common_accuracy_prefix"
                        ),
                    )
                    recovery = "signed append-only SNAKE checkpoint"
                snake_prefix_cache[k] = (prefix, source, recovery)
            if method == "snake":
                return snake_prefix_cache[k]
            if method == "append" and k not in append_prefix_cache:
                existing = _existing_prefix(
                    route=append_route,
                    regime=regime,
                    k=k,
                )
                if (
                    existing is not None
                    and existing[0].get("qiskit", {}).get("W1q") is None
                ):
                    existing = None
                if existing is not None:
                    prefix, recovery = existing
                    source = append_route["results"][regime]["source"]
                else:
                    prefix, source = _compile_comparator_at_k(
                        source=append_route["results"][regime]["source"],
                        trajectory=trajectory,
                        k=k,
                        representation="intact_macro",
                    )
                    recovery = "embedded Append-ADAPT checkpoint"
                append_prefix_cache[k] = (prefix, source, recovery)
            if method == "append":
                return append_prefix_cache[k]
            raise ValueError(f"unsupported method {method!r}")

        endpoints = {
            "insertion": actual_horizon,
            "snake": int(snake_route["plateau"][regime]["k_pl"]),
            "append": int(append_route["plateau"][regime]["k_pl"]),
        }
        label_by_method = dict(METHODS)
        pair_summaries: dict[str, Any] = {}
        for comparison_id, comparison_label, pair_methods in PAIR_SPECS:
            insertion_anchor = (
                INSERTION_PLATEAU_ANCHORS.get(regime)
                if comparison_id == "insertion_vs_append"
                else None
            )
            if insertion_anchor is not None:
                pair_common_error = float(
                    trajectories["insertion"][insertion_anchor - 1]["error"]
                )
                append_crossing = next(
                    int(point["round"])
                    for point in trajectories["append"]
                    if float(point["error"]) <= pair_common_error
                )
                pair_crossings = {
                    "insertion": int(insertion_anchor),
                    "append": int(append_crossing),
                }
                pair_window_end = max(pair_crossings.values())
                pair_minima = {
                    method: min(
                        float(point["error"])
                        for point in trajectories[method]
                        if int(point["round"]) <= pair_window_end
                    )
                    for method in pair_methods
                }
                selection_policy = (
                    "explicit_insertion_plateau_anchor_then_append_first_crossing"
                )
            else:
                pair_window_end = min(
                    endpoints[method] for method in pair_methods
                )
                pair_minima = {
                    method: min(
                        float(point["error"])
                        for point in trajectories[method]
                        if int(point["round"]) <= pair_window_end
                    )
                    for method in pair_methods
                }
                pair_common_error = max(pair_minima.values())
                pair_crossings = {
                    method: next(
                        int(point["round"])
                        for point in trajectories[method]
                        if int(point["round"]) <= pair_window_end
                        and float(point["error"]) <= pair_common_error
                    )
                    for method in pair_methods
                }
                selection_policy = "pairwise_common_window_first_crossing"
            pair_summaries[comparison_id] = {
                "Kcap": pair_window_end,
                "Ecap": pair_common_error,
                "crossings": pair_crossings,
                "selection_policy": selection_policy,
                "insertion_plateau_anchor": insertion_anchor,
            }
            for method in pair_methods:
                k = pair_crossings[method]
                trajectory = trajectories[method]
                prefix, source, recovery = prefix_for_method(method, k)
                point = trajectory[k - 1]
                matched.append(
                    {
                        "comparison_id": comparison_id,
                        "comparison_label": comparison_label,
                        "regime": regime,
                        "regime_title": title,
                        "abbreviation": abbreviation,
                        "n_ph": n_ph,
                        "method": method,
                        "method_label": label_by_method[method],
                        "common_window_end": pair_window_end,
                        "common_error": pair_common_error,
                        "selection_policy": selection_policy,
                        "insertion_plateau_anchor": insertion_anchor,
                        "method_minimum_error": pair_minima[method],
                        "k_cross": k,
                        "crossing_error": float(point["error"]),
                        **_compact_prefix(prefix),
                        "source": source,
                        "recovery": recovery,
                    }
                )

        terminal_prefix = insertion_prefix(actual_horizon)
        terminal.append(
            {
                "regime": regime,
                "regime_title": title,
                "abbreviation": abbreviation,
                "n_ph": n_ph,
                "terminal_k": actual_horizon,
                "endpoint_kind": endpoint_kind,
                "terminal_error": float(
                    trajectories["insertion"][actual_horizon - 1]["error"]
                ),
                **_compact_prefix(terminal_prefix),
                "source": insertion_source,
            }
        )
        print(
            f"{abbreviation}: pairs={pair_summaries}, "
            f"terminal={terminal[-1]['terminal_error']:.8e}",
            flush=True,
        )
    return matched, terminal


def _sci(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\mathrm{{e}}{int(exponent)}$"


def _integer(value: int) -> str:
    return f"{int(value):,}"


def _math_compact_sci(value: int) -> str:
    mantissa, exponent = f"{int(value):.1e}".split("e")
    return f"{mantissa}\\mathrm{{e}}{int(exponent)}"


def load_plot_rows(tracker: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = {route["id"]: route for route in tracker["routes"]}
    snake_route = routes[BASELINE_ROUTE]
    append_route = routes[APPEND_ROUTE]
    rows: list[dict[str, Any]] = []
    for regime, title, abbreviation, n_ph, horizon, campaign in REGIMES:
        result_path, _ledger_path, endpoint_kind = _insertion_artifacts(
            regime=regime,
            campaign=campaign,
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        actual_horizon = len(payload["adapt_vqe"]["history"])
        _validate_insertion_payload(
            payload,
            regime=regime,
            expected_horizon=actual_horizon,
        )
        rows.append(
            {
                "regime": regime,
                "title": title,
                "abbreviation": abbreviation,
                "n_ph": n_ph,
                "horizon": actual_horizon,
                "endpoint_kind": endpoint_kind,
                "insertion": _insertion_trajectory(payload),
                "snake": snake_route["results"][regime]["trajectory"][:50],
                "append": append_route["results"][regime]["trajectory"][:50],
            }
        )
    return rows


def _style_axes(
    ax: mpl.axes.Axes,
    *,
    x_max: int,
    values: list[float],
) -> None:
    ax.set_yscale("log")
    ax.set_xlim(1, max(2, x_max))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    positive = [value for value in values if value > 0.0]
    low = 10 ** np.floor(np.log10(min(positive)))
    high = 10 ** np.ceil(np.log10(max(positive)))
    if math.isclose(low, high):
        low /= 10.0
        high *= 10.0
    ax.set_ylim(low, high)
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55)
    ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(axis="both", labelsize=8)


def make_matched_plot(
    *,
    plot_rows: list[dict[str, Any]],
    matched: list[dict[str, Any]],
) -> None:
    lookup = {
        (row["regime"], row["method"]): row
        for row in matched
        if row["comparison_id"] == "insertion_vs_append"
    }
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.75, 4.95), dpi=300)
    for index, plot_row in enumerate(plot_rows):
        ax = axes.flat[index]
        regime = plot_row["regime"]
        insertion_row = lookup[(regime, "insertion")]
        append_row = lookup[(regime, "append")]
        insertion = [
            point
            for point in plot_row["insertion"]
            if int(point["round"]) <= insertion_row["k_cross"]
        ]
        append = [
            point
            for point in plot_row["append"]
            if int(point["round"]) <= append_row["k_cross"]
        ]
        values = [
            *[float(point["error"]) for point in insertion],
            *[float(point["error"]) for point in append],
        ]
        x_max = int(
            math.ceil(
                1.15
                * max(
                    int(insertion_row["k_cross"]),
                    int(append_row["k_cross"]),
                )
            )
        )
        _style_axes(ax, x_max=x_max, values=values)
        ax.plot(
            [int(point["round"]) for point in insertion],
            [float(point["error"]) for point in insertion],
            color="#E45756",
            linewidth=2.15,
            solid_capstyle="round",
        )
        ax.plot(
            [int(point["round"]) for point in append],
            [float(point["error"]) for point in append],
            color="#4C78A8",
            linewidth=1.55,
            linestyle=(0, (1.2, 2.1)),
            solid_capstyle="round",
        )
        ax.scatter(
            [insertion_row["k_cross"]],
            [insertion_row["crossing_error"]],
            color="#E45756",
            marker="*",
            s=72,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.scatter(
            [append_row["k_cross"]],
            [append_row["crossing_error"]],
            color="#4C78A8",
            marker="o",
            s=34,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.axhline(
            insertion_row["common_error"],
            color="#555555",
            linestyle=(0, (3, 2)),
            linewidth=0.85,
        )
        for row, color, marker, y in (
            (insertion_row, "#E45756", r"\star", 0.96),
            (append_row, "#4C78A8", r"\bullet", 0.87),
        ):
            ax.text(
                0.98,
                y,
                paper_i_cost_tuple_latex(
                    row,
                    marker=marker,
                    format_s_alg=_math_compact_sci,
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.6,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 0.5,
                },
                zorder=6,
            )
        ax.set_title(plot_row["title"], fontsize=9.2, pad=3)
        if index >= 3:
            ax.set_xlabel("ADAPT iteration, $k$", fontsize=8.5)
        if index % 3 == 0:
            ax.set_ylabel(r"Energy error, $\Delta E$", fontsize=8.5)
    handles = [
        Line2D(
            [0],
            [0],
            color="#E45756",
            linewidth=2.15,
            marker="*",
            markersize=8,
            markeredgecolor="white",
            label="Insertion SNAKE",
        ),
        Line2D(
            [0],
            [0],
            color="#4C78A8",
            linewidth=1.55,
            linestyle=(0, (1.2, 2.1)),
            marker="o",
            markersize=5,
            markeredgecolor="white",
            label="Append-ADAPT",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=(0, (3, 2)),
            linewidth=0.85,
            label=r"$\Delta E_\cap$",
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
        bottom=0.085,
        top=0.91,
        hspace=0.38,
        wspace=0.34,
    )
    fig.savefig(PAGE8_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_terminal_plot(
    *,
    plot_rows: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
) -> None:
    lookup = {row["regime"]: row for row in terminal}
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.75, 4.95), dpi=300)
    for index, plot_row in enumerate(plot_rows):
        ax = axes.flat[index]
        terminal_row = lookup[plot_row["regime"]]
        values = [
            *[float(point["error"]) for point in plot_row["insertion"]],
            *[float(point["error"]) for point in plot_row["snake"]],
            *[float(point["error"]) for point in plot_row["append"]],
        ]
        _style_axes(ax, x_max=50, values=values)
        for method, color, linestyle, linewidth in (
            ("snake", "#7F7F7F", (0, (4, 2)), 1.6),
            ("insertion", "#E45756", "solid", 2.15),
            ("append", "#4C78A8", (0, (1.2, 2.1)), 1.55),
        ):
            trajectory = plot_row[method]
            ax.plot(
                [int(point["round"]) for point in trajectory],
                [float(point["error"]) for point in trajectory],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                solid_capstyle="round",
            )
        insertion_last = plot_row["insertion"][-1]
        snake_last = plot_row["snake"][-1]
        append_last = plot_row["append"][-1]
        ax.scatter(
            [int(insertion_last["round"])],
            [float(insertion_last["error"])],
            color="#E45756",
            marker="*",
            s=72,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.scatter(
            [int(snake_last["round"])],
            [float(snake_last["error"])],
            color="#7F7F7F",
            marker="o",
            s=30,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.scatter(
            [int(append_last["round"])],
            [float(append_last["error"])],
            color="#4C78A8",
            marker="s",
            s=28,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.text(
            0.98,
            0.96,
            paper_i_cost_tuple_latex(
                terminal_row,
                marker=r"\star",
                format_s_alg=_math_compact_sci,
            )
            + "\n"
            + rf"$\Delta E_{{\rm term}}={terminal_row['terminal_error']:.2e}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#E45756",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.92,
            },
            zorder=6,
        )
        ax.set_title(plot_row["title"], fontsize=9.2, pad=3)
        if index >= 3:
            ax.set_xlabel("ADAPT iteration, $k$", fontsize=8.5)
        if index % 3 == 0:
            ax.set_ylabel(r"Energy error, $\Delta E$", fontsize=8.5)
    handles = [
        Line2D(
            [0],
            [0],
            color="#E45756",
            linewidth=2.15,
            marker="*",
            markersize=8,
            markeredgecolor="white",
            label="Insertion SNAKE",
        ),
        Line2D(
            [0],
            [0],
            color="#7F7F7F",
            linewidth=1.6,
            linestyle=(0, (4, 2)),
            marker="o",
            markersize=5,
            markeredgecolor="white",
            label="Append-only SNAKE",
        ),
        Line2D(
            [0],
            [0],
            color="#4C78A8",
            linewidth=1.55,
            linestyle=(0, (1.2, 2.1)),
            marker="s",
            markersize=5,
            markeredgecolor="white",
            label="Append-ADAPT",
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
        bottom=0.085,
        top=0.91,
        hspace=0.38,
        wspace=0.34,
    )
    fig.savefig(PAGE9_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _matched_table_rows(
    rows: list[dict[str, Any]],
    *,
    comparison_id: str,
) -> str:
    lines: list[str] = []
    by_regime = {
        regime: [
            row
            for row in rows
            if row["regime"] == regime
            and row["comparison_id"] == comparison_id
        ]
        for regime, *_rest in REGIMES
    }
    for regime, _title, abbreviation, _n_ph, _horizon, _campaign in REGIMES:
        regime_rows = by_regime[regime]
        for index, row in enumerate(regime_rows):
            regime_cell = abbreviation if index == 0 else ""
            cap_cell = (
                rf"{row['common_window_end']}; {_sci(row['common_error'])}"
                if index == 0
                else ""
            )
            lines.append(
                " & ".join(
                    [
                        regime_cell,
                        cap_cell,
                        row["method_label"],
                        str(row["k_cross"]),
                        _sci(row["crossing_error"]),
                        _integer(row["N2q"]),
                        _integer(row["D2q"]),
                        _integer(row["Dc"]),
                        _integer(row["W1q"]),
                        _integer(row["S_alg"]),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\addlinespace[2pt]")
    return "\n".join(lines)


def _terminal_table_rows(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        " & ".join(
            [
                row["abbreviation"],
                str(row["n_ph"]),
                str(row["terminal_k"]),
                _sci(row["terminal_error"]),
                _integer(row["N2q"]),
                _integer(row["D2q"]),
                _integer(row["Dc"]),
                _integer(row["W1q"]),
                _integer(row["S_alg"]),
            ]
        )
        + r" \\"
        for row in rows
    )


def write_pages_tex(
    matched: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
) -> None:
    del matched, terminal
    PAGES_TEX.write_text(
        rf"""
\documentclass[letterpaper,10pt]{{article}}
\usepackage[margin=0.35in]{{geometry}}
\usepackage{{amsmath,graphicx}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries Insertion SNAKE versus Append-ADAPT at matched error}}\par
\vspace{{3pt}}
\begin{{minipage}}{{0.96\textwidth}}
\small
Each curve ends at the reported matched-error prefix.  Weak--strong and
strong--strong use the insertion plateaus at \(k=9\) and \(k=11\) as their
error anchors; Append-ADAPT first reaches those errors at \(k=9\) and \(k=12\).
The other panels retain the pairwise common-window first-crossing rule.
Endpoint labels report ${PAPER_I_QISKIT_COST_TUPLE_LATEX}$.
\end{{minipage}}
\vspace{{5pt}}
\includegraphics[width=7.75in]{{{PAGE8_PNG.as_posix()}}}

{{\footnotesize The append-only SNAKE versus Append-ADAPT matched-error
comparison remains on page 2 and uses its own independently selected pairwise
cap.}}
\end{{center}}

\newpage
\begin{{center}}
{{\Large\bfseries Endpoint costs of commutation-reduced insertion SNAKE}}\par
\vspace{{3pt}}
\begin{{minipage}}{{0.94\textwidth}}
\small
Red stars mark either the completed local horizon or the retrieved finalized
CHTC checkpoint and report the corresponding
${PAPER_I_QISKIT_COST_TUPLE_LATEX}$ tuple without accuracy matching.  Here
$W_{{1q}}$ is genuine Qiskit-emitted Pauli-rotation one-qubit work before
transpilation. The two append
routes are retained through iteration 50 as trajectory context; their costs
are not reported on this page.
\end{{minipage}}
\vspace{{5pt}}
\includegraphics[width=7.75in]{{{PAGE9_PNG.as_posix()}}}
\end{{center}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def assemble_review() -> None:
    if len(PdfReader(str(BASE_REVIEW_PDF)).pages) != 7:
        raise ValueError("preserved review input is not seven pages")
    REVIEW_TEX.write_text(
        rf"""
\documentclass[letterpaper]{{article}}
\usepackage{{pdfpages}}
\pagestyle{{empty}}
\begin{{document}}
\includepdf[
  pages=-,
  pagecommand={{}},
  fitpaper=true,
  noautoscale=true,
  offset=0 0
]{{{BASE_REVIEW_PDF.as_posix()}}}
\includepdf[
  pages=-,
  pagecommand={{}},
  fitpaper=true,
  noautoscale=true,
  offset=0 0
]{{{PAGES_PDF.as_posix()}}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    writer = PdfWriter()
    for source_pdf in (BASE_REVIEW_PDF, PAGES_PDF):
        source_reader = PdfReader(str(source_pdf), strict=False)
        for page in source_reader.pages:
            writer.add_page(page)
    with REVIEW_PDF.open("wb") as handle:
        writer.write(handle)
    if len(PdfReader(str(REVIEW_PDF)).pages) != 9:
        raise ValueError("assembled review is not nine pages")
    shutil.copy2(REVIEW_PDF, FINAL_PDF)


def write_provenance(
    matched: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": "paper_i_hh_macro_insertion_cost_comparison_v3",
        "definition": (
            "The Paper-I common-window rule is applied independently to "
            "insertion SNAKE versus Append-ADAPT and append-only SNAKE versus "
            "Append-ADAPT. Weak-strong and strong-strong instead use explicit "
            "insertion plateau anchors k=9 and k=11, with Append-ADAPT taken "
            "at its first crossing of the anchored insertion error. Insertion "
            "endpoint costs are reported independently."
        ),
        "routes": {
            "insertion": INSERTION_ROUTE,
            "append_only_snake": BASELINE_ROUTE,
            "append_adapt": APPEND_ROUTE,
        },
        "matched_rows": matched,
        "terminal_insertion_rows": terminal,
        "inputs": {
            "tracker": {
                "path": str(TRACKER.relative_to(REPO_ROOT)),
                "sha256": sha256(TRACKER),
            },
            "base_review_pdf": {
                "path": str(BASE_REVIEW_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(BASE_REVIEW_PDF),
                "pages": 7,
            },
        },
        "generated": {
            "matched_plot_png": {
                "path": str(PAGE8_PNG.relative_to(REPO_ROOT)),
                "sha256": sha256(PAGE8_PNG),
            },
            "terminal_plot_png": {
                "path": str(PAGE9_PNG.relative_to(REPO_ROOT)),
                "sha256": sha256(PAGE9_PNG),
            },
            "pages_tex": {
                "path": str(PAGES_TEX.relative_to(REPO_ROOT)),
                "sha256": sha256(PAGES_TEX),
            },
            "pages_pdf": {
                "path": str(PAGES_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(PAGES_PDF),
                "pages": 2,
            },
            "review_tex": {
                "path": str(REVIEW_TEX.relative_to(REPO_ROOT)),
                "sha256": sha256(REVIEW_TEX),
            },
            "review_pdf": {
                "path": str(REVIEW_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(REVIEW_PDF),
                "pages": 9,
                "assembly": (
                    "page-object concatenation of the preserved seven-page "
                    "LaTeX review and the two-page LaTeX replacement"
                ),
            },
            "canonical_pdf": {
                "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
                "sha256": sha256(FINAL_PDF),
                "pages": 9,
            },
        },
        "validation": {
            "source_ledgers_closed": True,
            "runtime_estimator_receipts_required": True,
            "qiskit_prefix_reconstruction": (
                "signed checkpoint when compatible; otherwise exact "
                "nonpruned accepted-history reconstruction"
            ),
            "qiskit_compile_identity": "table_i_basis_gate_transpile_v1",
            "page_count": 9,
            "rendered_pages_inspected": [7, 8, 9],
        },
    }
    PROVENANCE.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if MAIN_PROVENANCE.is_file():
        main = json.loads(MAIN_PROVENANCE.read_text(encoding="utf-8"))
        main["macro_commutation_reduced_insertion_costs"] = {
            "definition": payload["definition"],
            "matched_rows": matched,
            "terminal_insertion_rows": terminal,
            "provenance": {
                "path": str(PROVENANCE.relative_to(REPO_ROOT)),
                "sha256": sha256(PROVENANCE),
            },
        }
        main.setdefault("generated", {})["pdf"] = payload["generated"][
            "canonical_pdf"
        ]
        validation = main.setdefault("validation", {})
        validation["page_count"] = 9
        validation["rendered_pages_inspected"] = [7, 8, 9]
        MAIN_PROVENANCE.write_text(
            json.dumps(main, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="reconstruct all exact prefixes instead of reusing cached rows",
    )
    args = parser.parse_args()
    tracker = json.loads(TRACKER.read_text(encoding="utf-8"))
    if DATA_JSON.is_file() and not args.refresh:
        cached = json.loads(DATA_JSON.read_text(encoding="utf-8"))
        if cached.get("schema") != "paper_i_hh_macro_insertion_cost_rows_v3":
            raise ValueError("insertion-cost row-cache schema drift")
        matched = cached["matched_rows"]
        terminal = cached["terminal_insertion_rows"]
    else:
        matched, terminal = collect_rows(tracker)
        DATA_JSON.write_text(
            json.dumps(
                {
                    "schema": "paper_i_hh_macro_insertion_cost_rows_v3",
                    "matched_rows": matched,
                    "terminal_insertion_rows": terminal,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    plot_rows = load_plot_rows(tracker)
    make_matched_plot(plot_rows=plot_rows, matched=matched)
    make_terminal_plot(plot_rows=plot_rows, terminal=terminal)
    write_pages_tex(matched, terminal)
    compiled_pages = compile_tex(PAGES_TEX)
    if compiled_pages != PAGES_PDF or len(PdfReader(str(PAGES_PDF)).pages) != 2:
        raise ValueError("cost-page compilation failed")
    assemble_review()
    write_provenance(matched, terminal)
    print(FINAL_PDF)


if __name__ == "__main__":
    main()
