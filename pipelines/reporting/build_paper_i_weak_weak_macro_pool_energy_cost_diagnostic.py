#!/usr/bin/env python3
"""Build the Paper-I weak-weak macro-only operator/cost diagnostic PDF.

The input result is intentionally reduced with ``jq`` before Python sees it.  The
preserved result JSON is large because it includes estimator-ledger receipts and
candidate surfaces; loading it wholesale is unnecessary for this report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch


BG = "#f7f6f2"
INK = "#172033"
MUTED = "#5e6878"
GRID = "#d9dee7"
ACCENT = "#d97706"
CLASS_COLORS = {
    "uccsd_sing": "#2563eb",
    "paop_disp": "#0f766e",
    "paop_cloud_p": "#14b8a6",
    "hh_phonon_quadratic": "#9333ea",
    "hh_termwise_quadrature": "#db2777",
}


JQ_REDUCTION = r"""
{
  run: {
    energy: .adapt_vqe.energy,
    exact_energy: .adapt_vqe.exact_gs_energy,
    abs_delta_e: .adapt_vqe.abs_delta_e,
    depth: .adapt_vqe.ansatz_depth,
    pool_size: .adapt_vqe.pool_size,
    route_family: .adapt_vqe.route_family,
    route_profile: .adapt_vqe.route_profile,
    stop_reason: .adapt_vqe.stop_reason,
    success: .adapt_vqe.success,
    accounting: (.adapt_vqe.estimator_call_accounting.winning_lineage |
      {N_H_outer, N_H_refit, N_grad, N_metric, S_alg}),
    settings: {
      L: .settings.L,
      boundary: .settings.boundary,
      t: .settings.t,
      u: .settings.u,
      dv: .settings.dv,
      omega0: .settings.omega0,
      g_ep: .settings.g_ep,
      n_ph_work: .settings.n_ph_max,
      optimizer: .settings.adapt_inner_optimizer,
      seed: .settings.adapt_seed,
      pool: .settings.adapt_pool,
      max_depth: .settings.adapt_max_depth,
      child_pool_expansion: .settings.adapt_child_pool_expansion_mode,
      runtime_split: .settings.phase3_runtime_split_mode,
      shared_pauli_pool: .settings.shared_pauli_pool_mode
    }
  },
  history: [
    .adapt_vqe.history[] as $h |
    {
      round: $h.depth,
      selected_op: $h.selected_op,
      selected_class: $h.physical_operator_hh_full_meta_class,
      selected_lane: $h.physical_operator_lane,
      delta_energy: $h.delta_energy,
      energy_before: $h.energy_before_opt,
      energy_after: $h.energy_after_opt,
      c_hat_2q: $h.compile_cost_proxy.c_hat_2q,
      c_hat_d: $h.compile_cost_proxy.c_hat_d,
      s_alg_cumulative: $h.active_prefix_checkpoint.estimator_ledger_receipt.cumulative_unique_primitives.S_alg,
      s_alg_delta: $h.active_prefix_checkpoint.estimator_ledger_receipt.unique_primitive_delta.S_alg,
      s_alg_components_cumulative: $h.active_prefix_checkpoint.estimator_ledger_receipt.cumulative_unique_primitives.components,
      candidates: [
        $h.scored_surface_records[] |
        {
          label: .candidate_label,
          class: .physical_operator_hh_full_meta_class,
          lane: .physical_operator_lane,
          predicted_drop: (.phase_score_components.DeltaE_TR //
            .phase3_reduced_trust_gain // .phase2_raw_trust_gain //
            .phase1_trust_region_gain),
          c_hat_2q: .c_hat_2q,
          c_hat_d: .c_hat_d,
          selected: (.candidate_label == $h.selected_op)
        }
      ]
    }
  ]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--qiskit-json", type=Path, required=True)
    parser.add_argument("--prefix-qiskit-cost-json", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-member", required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reduce_result(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["jq", "-c", JQ_REDUCTION, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def class_color(name: str) -> str:
    return CLASS_COLORS.get(name, "#64748b")


def class_name(name: str) -> str:
    return {
        "uccsd_sing": "UCCSD single",
        "paop_disp": "PAOP displacement",
        "paop_cloud_p": "PAOP cloud",
        "hh_phonon_quadratic": "phonon squeeze",
        "hh_termwise_quadrature": "HVA quadrature",
    }.get(name, name.replace("_", " "))


def short_operator(label: str) -> str:
    replacements = {
        "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)": "singlet alpha 0->1",
        "uccsd_ferm_lifted::uccsd_sing(beta:2->3)": "singlet beta 2->3",
        "paop_full:paop_disp(site=1)": "displacement site 1",
        "paop_full:paop_cloud_p(site=1->phonon=0)": "cloud 1->phonon 0",
        "hh_phonon::s(site=1)": "squeeze site 1",
        "hh_phonon::s(site=0)": "squeeze site 0",
        "hh_termwise_ham_quadrature_term(eyeeeeze)": "HVA eyeeeeze",
        "hh_termwise_ham_quadrature_term(zyeezeee)": "HVA zyeezeee",
    }
    return replacements.get(label, label)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str, page: int) -> None:
    fig.text(0.055, 0.955, title, fontsize=17, weight="bold", color=INK, va="top")
    fig.text(0.055, 0.923, subtitle, fontsize=9.2, color=MUTED, va="top")
    fig.text(
        0.945,
        0.955,
        f"PAPER I DESIGN STUDY  |  {page}/3",
        fontsize=7.3,
        color=MUTED,
        ha="right",
        va="top",
        weight="bold",
    )


def add_card(
    fig: plt.Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    accent: str = ACCENT,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=fig.transFigure,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=0.7,
        edgecolor="#d7dbe2",
        facecolor="white",
    )
    fig.patches.append(patch)
    fig.text(x + 0.012, y + h - 0.018, label.upper(), fontsize=6.6, color=MUTED, weight="bold")
    fig.text(x + 0.012, y + 0.016, value, fontsize=9.2, color=accent, weight="bold")


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.055, 0.025, text, fontsize=6.7, color=MUTED, va="bottom")


def prepare_data(reduced: dict[str, Any], prefix_qiskit: dict[str, Any]) -> dict[str, Any]:
    history = reduced["history"]
    if len(history) != 50:
        raise ValueError(f"expected 50 accepted macro rounds, found {len(history)}")

    for row in history:
        row["accepted_drop"] = max(0.0, -float(row["delta_energy"]))
        recomputed = float(row["energy_before"]) - float(row["energy_after"])
        if not math.isclose(row["accepted_drop"], recomputed, rel_tol=1e-7, abs_tol=1e-11):
            raise ValueError(f"round {row['round']} energy-drop mismatch")

    accounting = reduced["run"]["accounting"]
    if history[-1]["s_alg_cumulative"] != accounting["S_alg"]:
        raise ValueError("final active-prefix S_alg does not match winning-lineage accounting")
    qiskit_rows = prefix_qiskit.get("rows")
    if not isinstance(qiskit_rows, list) or len(qiskit_rows) != len(history):
        raise ValueError("prefix Qiskit cost curve does not contain the 50 accepted rounds")

    operators: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "drop": 0.0, "drops": [], "first_round": 10**9}
    )
    for row in history:
        op = operators[row["selected_op"]]
        op["class"] = row["selected_class"]
        op["lane"] = row["selected_lane"]
        op["count"] += 1
        op["drop"] += row["accepted_drop"]
        op["drops"].append(row["accepted_drop"])
        op["first_round"] = min(op["first_round"], row["round"])

    ordered_labels = sorted(operators, key=lambda label: operators[label]["drop"], reverse=True)
    total_drop = sum(info["drop"] for info in operators.values())
    operator_rows = []
    for index, label in enumerate(ordered_labels):
        info = operators[label]
        operator_rows.append(
            {
                "id": chr(ord("A") + index),
                "label": label,
                "short_label": short_operator(label),
                "class": info["class"],
                "lane": info["lane"],
                "count": info["count"],
                "drop": info["drop"],
                "median_drop": float(np.median(info["drops"])),
                "share": info["drop"] / total_drop if total_drop else 0.0,
            }
        )

    candidates = []
    for row in history:
        for candidate in row["candidates"]:
            candidate = dict(candidate)
            candidate["round"] = row["round"]
            candidates.append(candidate)

    generator_types: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "drop": 0.0,
            "drops": [],
            "table_i_N2q": 0,
            "table_i_D2q": 0,
            "current_N2q_signed": 0,
            "current_D2q_signed": 0,
        }
    )
    for history_row, cost_row in zip(history, qiskit_rows, strict=True):
        if int(cost_row["round"]) != int(history_row["round"]):
            raise ValueError("prefix Qiskit round order disagrees with accepted history")
        if str(cost_row["selected_op"]) != str(history_row["selected_op"]):
            raise ValueError("prefix Qiskit selected operator disagrees with accepted history")
        if not math.isclose(
            float(cost_row["accepted_drop"]),
            float(history_row["accepted_drop"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("prefix Qiskit accepted drop disagrees with accepted history")
        class_key = str(history_row["selected_class"])
        entry = generator_types[class_key]
        entry["class"] = class_key
        entry["count"] += 1
        entry["drop"] += float(history_row["accepted_drop"])
        entry["drops"].append(float(history_row["accepted_drop"]))
        table_delta = cost_row["marginal_prefix_delta"]["historical_displayed"]
        current_delta = cost_row["marginal_prefix_delta"]["current_jr_fake_marrakesh"]
        entry["table_i_N2q"] += int(table_delta["N2q"])
        entry["table_i_D2q"] += int(table_delta["D2q"])
        entry["current_N2q_signed"] += int(current_delta["N2q"])
        entry["current_D2q_signed"] += int(current_delta["D2q"])

    type_rows = []
    for class_key, entry in generator_types.items():
        if entry["table_i_N2q"] <= 0 or entry["table_i_D2q"] <= 0:
            raise ValueError(f"generator type {class_key} has a nonpositive Table-I denominator")
        type_rows.append(
            {
                "class": class_key,
                "display_name": class_name(class_key),
                "count": entry["count"],
                "drop": entry["drop"],
                "median_drop": float(np.median(entry["drops"])),
                "share": entry["drop"] / total_drop if total_drop else 0.0,
                "table_i_N2q": entry["table_i_N2q"],
                "table_i_D2q": entry["table_i_D2q"],
                "current_N2q_signed": entry["current_N2q_signed"],
                "current_D2q_signed": entry["current_D2q_signed"],
                "drop_per_table_i_N2q": entry["drop"] / entry["table_i_N2q"],
                "drop_per_table_i_D2q": entry["drop"] / entry["table_i_D2q"],
            }
        )
    type_rows.sort(key=lambda row: row["drop"], reverse=True)
    if sum(row["table_i_N2q"] for row in type_rows) != 1198:
        raise ValueError("generator-type Table-I N2q totals do not close to the locked endpoint")
    if sum(row["table_i_D2q"] for row in type_rows) != 1114:
        raise ValueError("generator-type Table-I D2q totals do not close to the locked endpoint")

    return {
        "history": history,
        "operators": operator_rows,
        "generator_types": type_rows,
        "candidates": candidates,
        "total_drop": total_drop,
    }


def page_one(pdf: PdfPages, reduced: dict[str, Any], prepared: dict[str, Any]) -> None:
    run = reduced["run"]
    settings = run["settings"]
    history = prepared["history"]
    operators = prepared["operators"]
    generator_types = prepared["generator_types"]

    fig = plt.figure(figsize=(11, 8.5))
    add_header(
        fig,
        "Weak-weak macro-only operator energy-drop atlas",
        "Actual accepted post-refit drops across the 50-round preserved SR-SNAKE macro prefix.",
        1,
    )
    add_card(fig, 0.055, 0.825, 0.205, 0.067, "Regime", "U/t = 0.25  |  g/w = 0.353553")
    add_card(fig, 0.275, 0.825, 0.205, 0.067, "Hilbert space", "L = 2  |  n_ph(work) = 3")
    add_card(fig, 0.495, 0.825, 0.205, 0.067, "Controller", "macro-only  |  50 rounds")
    add_card(fig, 0.715, 0.825, 0.23, 0.067, "Endpoint", f"|E - E*| = {run['abs_delta_e']:.3e}")

    gs = fig.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.95,
        bottom=0.09,
        top=0.775,
        width_ratios=[4.1, 2.0],
        height_ratios=[2.2, 1.15],
        hspace=0.42,
        wspace=0.28,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    bottom = gs[1, :].subgridspec(1, 2, wspace=0.34)
    ax_definition = fig.add_subplot(bottom[0, 0])
    ax_error = fig.add_subplot(bottom[0, 1])

    labels = [row["label"] for row in operators]
    label_to_index = {label: index for index, label in enumerate(labels)}
    heat = np.full((len(labels), len(history)), np.nan)
    for col, row in enumerate(history):
        heat[label_to_index[row["selected_op"]], col] = label_to_index[row["selected_op"]]
    cmap = ListedColormap([class_color(row["class"]) for row in operators])
    cmap.set_bad("#e9ebef")
    ax_heat.imshow(
        np.ma.masked_invalid(heat),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, len(operators) + 0.5), len(operators)),
    )
    ax_heat.set_title("Accepted operator use by controller round", loc="left", weight="bold")
    ax_heat.set_xlabel("controller round")
    ax_heat.set_yticks(range(len(operators)))
    ax_heat.set_yticklabels([f"{row['id']}  {row['short_label']}" for row in operators], fontsize=7.8)
    tick_rounds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ax_heat.set_xticks([value - 1 for value in tick_rounds])
    ax_heat.set_xticklabels(tick_rounds)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.text(
        0.0,
        -0.16,
        "one colored cell = one admitted occurrence; color identifies operator class only",
        transform=ax_heat.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="top",
    )

    y = np.arange(len(generator_types))
    totals = np.array([row["drop"] for row in generator_types])
    colors = [class_color(row["class"]) for row in generator_types]
    ax_bar.barh(y, totals, color=colors, alpha=0.9)
    ax_bar.set_xscale("log")
    ax_bar.invert_yaxis()
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([row["display_name"] for row in generator_types], fontsize=7.2)
    ax_bar.set_title("Raw drop by generator type", loc="left", weight="bold", fontsize=9.6)
    ax_bar.set_xlabel(r"sum of one-time accepted drops")
    ax_bar.grid(axis="x", which="both", alpha=0.6)
    ax_bar.set_axisbelow(True)
    for index, row in enumerate(generator_types):
        share_percent = 100 * row["share"]
        share_text = f"{share_percent:.2e}%" if share_percent < 0.01 else f"{share_percent:.2f}%"
        ax_bar.text(
            row["drop"] * 1.18,
            index,
            f"{share_text}  ({row['count']}x)",
            va="center",
            fontsize=6.8,
            color=MUTED,
        )
    ax_bar.set_xlim(float(totals.min()) * 0.65, float(totals.max()) * 18)

    rounds = np.array([row["round"] for row in history])
    early_share = sum(row["accepted_drop"] for row in history[:2]) / prepared["total_drop"]
    ax_definition.axis("off")
    ax_definition.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax_definition.transAxes,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.7,
            edgecolor="#d7dbe2",
            facecolor="white",
        )
    )
    ax_definition.text(0.045, 0.88, "WHAT THE BAR COUNTS", fontsize=7.2, color=MUTED, weight="bold")
    ax_definition.text(
        0.045,
        0.68,
        r"admission $k$:  $d_k=E_{before,k}-E_{after,k}$",
        fontsize=9.1,
        color=INK,
        weight="bold",
    )
    ax_definition.text(
        0.045,
        0.50,
        r"generator type $t$:  $R_t=\sum_{k\in t} d_k$",
        fontsize=9.1,
        color=INK,
        weight="bold",
    )
    ax_definition.text(
        0.045,
        0.31,
        "Each admission counts once; repeated types accumulate. This is path-based,\n"
        "not leave-one-out final-ansatz attribution, so early admissions can dominate.",
        fontsize=6.7,
        color=MUTED,
        va="top",
        linespacing=1.25,
    )
    ax_definition.text(
        0.045,
        0.09,
        f"Path effect: admissions 1-2 account for {100 * early_share:.2f}% of total admission-path drop.",
        fontsize=7.4,
        color=ACCENT,
        weight="bold",
    )

    errors = np.abs(np.array([row["energy_after"] for row in history]) - run["exact_energy"])
    ax_error.plot(rounds, errors, color=ACCENT, linewidth=1.6)
    ax_error.scatter(rounds, errors, color=ACCENT, s=15, zorder=2)
    ax_error.set_yscale("log")
    ax_error.set_title("Same-cutoff error trajectory", loc="left", weight="bold")
    ax_error.set_xlabel("controller round")
    ax_error.set_ylabel(r"$|E_k-E^*_{n_{ph}=3}|$")
    ax_error.grid(True, which="both", alpha=0.55)

    add_footer(
        fig,
        "Diagnostic macro-only ablation. Blank heatmap cells mean no occurrence of that label was admitted in that round. "
        "Admission-path drop is path dependent and is not final-prefix causal attribution.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_cost_benefit_summary(
    ax: plt.Axes,
    points: list[dict[str, Any]],
    cost_key: str,
    title: str,
    xlabel: str,
    display_floor: float = 1e-12,
) -> None:
    levels = sorted(
        {
            float(point[cost_key])
            for point in points
            if finite_positive(point.get(cost_key)) and finite_positive(point.get("predicted_drop"))
        }
    )
    for position, level in enumerate(levels):
        values = np.array(
            [
                float(point["predicted_drop"])
                for point in points
                if point.get(cost_key) == level and finite_positive(point.get("predicted_drop"))
            ]
        )
        q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        q10, q25, median, q75, q90 = [max(display_floor, value) for value in (q10, q25, median, q75, q90)]
        maximum = max(display_floor, float(values.max()))
        ax.vlines(position, q10, q90, color="#94a3b8", linewidth=1.1, zorder=1)
        ax.fill_between(
            [position - 0.28, position + 0.28],
            [q25, q25],
            [q75, q75],
            color="#b9ddd8",
            alpha=0.95,
            zorder=2,
        )
        ax.hlines(median, position - 0.28, position + 0.28, color="#0f766e", linewidth=1.7, zorder=3)
        ax.scatter(position, maximum, marker="^", s=24, color="#64748b", edgecolor="white", linewidth=0.35, zorder=3)

        accepted = np.array(
            [
                float(point["predicted_drop"])
                for point in points
                if point.get("selected")
                and point.get(cost_key) == level
                and finite_positive(point.get("predicted_drop"))
            ]
        )
        if len(accepted):
            accepted_q25, accepted_median, accepted_q75 = np.quantile(accepted, [0.25, 0.50, 0.75])
            ax.vlines(
                position,
                max(display_floor, accepted_q25),
                max(display_floor, accepted_q75),
                color=INK,
                linewidth=2.1,
                zorder=4,
            )
            ax.scatter(
                position,
                max(display_floor, accepted_median),
                marker="D",
                s=33,
                color=ACCENT,
                edgecolor=INK,
                linewidth=0.7,
                zorder=5,
            )
            ax.annotate(
                f"n={len(accepted)}",
                (position, max(display_floor, accepted_median)),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=5.8,
                color=INK,
            )

    ax.set_yscale("log")
    ax.set_ylim(display_floor, 2.0)
    ax.set_xlim(-0.65, len(levels) - 0.35)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{level:g}" for level in levels], rotation=45 if len(levels) > 12 else 0)
    ax.tick_params(axis="x", labelsize=6.7)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"candidate $\Delta E_{TR}$")
    ax.grid(True, axis="y", which="both", alpha=0.55)
    ax.set_axisbelow(True)


def page_two(
    pdf: PdfPages,
    reduced: dict[str, Any],
    prepared: dict[str, Any],
    qiskit: dict[str, Any],
) -> None:
    generator_types = prepared["generator_types"]
    fig = plt.figure(figsize=(11, 8.5))
    add_header(
        fig,
        "Raw and cost-normalized energy drop",
        "Bars group accepted occurrences by generator type; the normalized view divides raw path drop by marginal Qiskit N2q.",
        2,
    )
    add_card(fig, 0.055, 0.825, 0.205, 0.067, "Generator types", f"{len(generator_types)} grouped types")
    add_card(fig, 0.275, 0.825, 0.205, 0.067, "Admissions", f"{sum(row['count'] for row in generator_types)} occurrences")
    add_card(fig, 0.495, 0.825, 0.205, 0.067, "Qiskit denominator", "Table-I  |  O0")
    add_card(fig, 0.715, 0.825, 0.23, 0.067, "Endpoint closure", f"N2q = {sum(row['table_i_N2q'] for row in generator_types):,}")

    gs = fig.add_gridspec(
        2,
        2,
        left=0.11,
        right=0.95,
        bottom=0.10,
        top=0.775,
        height_ratios=[1.55, 0.95],
        hspace=0.38,
        wspace=0.34,
    )
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_norm = fig.add_subplot(gs[0, 1])
    ax_cost = fig.add_subplot(gs[1, 0])
    ax_definition = fig.add_subplot(gs[1, 1])

    y = np.arange(len(generator_types))
    colors = [class_color(row["class"]) for row in generator_types]
    raw_values = np.array([row["drop"] for row in generator_types])
    normalized_values = np.array([row["drop_per_table_i_N2q"] for row in generator_types])

    ax_raw.barh(y, raw_values, color=colors, alpha=0.92)
    ax_raw.set_xscale("log")
    ax_raw.invert_yaxis()
    ax_raw.set_yticks(y)
    ax_raw.set_yticklabels([row["display_name"] for row in generator_types], fontsize=7.6)
    ax_raw.set_title(r"Raw $\sum(-\Delta E)$ by generator type", loc="left", weight="bold")
    ax_raw.set_xlabel("cumulative admission-path drop")
    ax_raw.grid(axis="x", which="both", alpha=0.55)
    ax_raw.set_axisbelow(True)
    ax_raw.set_xlim(float(raw_values.min()) * 0.55, float(raw_values.max()) * 12)
    for index, row in enumerate(generator_types):
        share = 100 * row["share"]
        share_text = f"{share:.2e}%" if share < 0.01 else f"{share:.2f}%"
        ax_raw.text(
            row["drop"] * 1.18,
            index,
            f"{row['drop']:.3g}  |  {share_text}, n={row['count']}",
            va="center",
            fontsize=6.7,
            color=MUTED,
        )

    ax_norm.barh(y, normalized_values, color=colors, alpha=0.92)
    ax_norm.set_xscale("log")
    ax_norm.invert_yaxis()
    ax_norm.set_yticks(y)
    ax_norm.set_yticklabels([row["display_name"] for row in generator_types], fontsize=7.6)
    ax_norm.set_title(r"Normalized $\sum(-\Delta E)/\sum\Delta N_{2q}$", loc="left", weight="bold")
    ax_norm.set_xlabel("energy drop per marginal Qiskit two-qubit gate")
    ax_norm.grid(axis="x", which="both", alpha=0.55)
    ax_norm.set_axisbelow(True)
    ax_norm.set_xlim(float(normalized_values.min()) * 0.55, float(normalized_values.max()) * 18)
    for index, row in enumerate(generator_types):
        ax_norm.text(
            row["drop_per_table_i_N2q"] * 1.18,
            index,
            f"{row['drop_per_table_i_N2q']:.3e}  |  N2q={row['table_i_N2q']}",
            va="center",
            fontsize=6.7,
            color=MUTED,
        )

    short_names = [
        {
            "uccsd_sing": "UCCSD",
            "paop_disp": "PAOP disp",
            "paop_cloud_p": "PAOP cloud",
            "hh_phonon_quadratic": "phonon sq",
            "hh_termwise_quadrature": "HVA quad",
        }.get(row["class"], row["display_name"])
        for row in generator_types
    ]
    x = np.arange(len(generator_types))
    width = 0.34
    n2q = np.array([row["table_i_N2q"] for row in generator_types])
    d2q = np.array([row["table_i_D2q"] for row in generator_types])
    ax_cost.bar(x - width / 2, n2q, width, color="#334155", label="sum marginal N2q")
    ax_cost.bar(x + width / 2, d2q, width, color="#94a3b8", label="sum marginal D2q")
    ax_cost.set_yscale("log")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(short_names, rotation=18, ha="right", fontsize=6.8)
    ax_cost.set_ylabel("summed marginal Qiskit cost")
    ax_cost.set_title("Cost assigned to each generator type", loc="left", weight="bold")
    ax_cost.grid(axis="y", which="both", alpha=0.55)
    ax_cost.legend(frameon=False, fontsize=6.7, loc="upper right")
    for xpos, value in zip(x - width / 2, n2q, strict=True):
        ax_cost.text(xpos, value * 1.12, f"{value}", ha="center", va="bottom", fontsize=5.8, color=INK)
    for xpos, value in zip(x + width / 2, d2q, strict=True):
        ax_cost.text(xpos, value * 1.12, f"{value}", ha="center", va="bottom", fontsize=5.8, color=MUTED)

    ax_definition.axis("off")
    ax_definition.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax_definition.transAxes,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.7,
            edgecolor="#d7dbe2",
            facecolor="white",
        )
    )
    ax_definition.text(0.05, 0.86, "NORMALIZATION CONTRACT", fontsize=7.2, color=MUTED, weight="bold")
    ax_definition.text(0.05, 0.67, r"$R_t=\sum_{k\in t}(-\Delta E_k)$", fontsize=10, color=INK, weight="bold")
    ax_definition.text(0.05, 0.51, r"$C_t=\sum_{k\in t}\Delta N_{2q,k}$", fontsize=10, color=INK, weight="bold")
    ax_definition.text(0.05, 0.35, r"$\eta_t=R_t/C_t$", fontsize=10, color=ACCENT, weight="bold")
    ax_definition.text(
        0.05,
        0.18,
        "Each prefix is compiled independently under the locked Table-I Qiskit convention.\n"
        "All N2q increments are nonnegative and close exactly to N2q(k=50)=1,198.\n"
        "Cost normalization does not remove the early-admission advantage in raw drop.",
        fontsize=6.7,
        color=MUTED,
        va="top",
        linespacing=1.28,
    )

    add_footer(
        fig,
        "Raw bars sum one-time admission drops. Normalized bars use the ratio of sums with marginal Table-I Qiskit N2q; "
        "they are cost efficiencies, not final-prefix causal attributions.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def page_three(
    pdf: PdfPages,
    reduced: dict[str, Any],
    prepared: dict[str, Any],
    qiskit: dict[str, Any],
    source: dict[str, Any],
) -> None:
    run = reduced["run"]
    accounting = run["accounting"]
    generator_types = prepared["generator_types"]
    fig = plt.figure(figsize=(11, 8.5))
    add_header(
        fig,
        "Generator-type ledger and provenance",
        "Everything on pages 1-2 is recoverable from the preserved weak-weak macro-only result and sidecar.",
        3,
    )

    ax_table = fig.add_axes([0.055, 0.54, 0.89, 0.34])
    ax_table.axis("off")
    columns = ["generator type", "admissions", "raw sum(-dE)", "path share", "sum dN2q", "sum dD2q", "raw/N2q"]
    rows = [
        [
            row["display_name"],
            str(row["count"]),
            f"{row['drop']:.6g}",
            (
                f"{100 * row['share']:.3e}%"
                if 100 * row["share"] < 0.001
                else f"{100 * row['share']:.3f}%"
            ),
            str(row["table_i_N2q"]),
            str(row["table_i_D2q"]),
            f"{row['drop_per_table_i_N2q']:.3e}",
        ]
        for row in generator_types
    ]
    table = ax_table.table(
        cellText=rows,
        colLabels=columns,
        colLoc="left",
        cellLoc="left",
        colWidths=[0.22, 0.09, 0.13, 0.11, 0.11, 0.11, 0.14],
        loc="upper left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#dfe3e9")
        cell.set_linewidth(0.55)
        if row_index == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white" if row_index % 2 else "#f0f2f5")
            if col_index == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color(class_color(generator_types[row_index - 1]["class"]))

    hist_metrics = qiskit["historical_displayed_convention"]["metrics"]
    curr_metrics = qiskit["current_jr_fake_marrakesh_convention"]["metrics"]
    left = fig.add_axes([0.055, 0.285, 0.275, 0.19])
    mid = fig.add_axes([0.362, 0.285, 0.275, 0.19])
    right = fig.add_axes([0.67, 0.285, 0.275, 0.19])
    for ax in (left, mid, right):
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                linewidth=0.7,
                edgecolor="#d7dbe2",
                facecolor="white",
            )
        )

    left.text(0.05, 0.88, "FINAL PREFIX", fontsize=7, color=MUTED, weight="bold", va="top")
    left.text(0.05, 0.70, f"E_50 = {run['energy']:.12f}", fontsize=9.2, color=INK, weight="bold")
    left.text(0.05, 0.54, f"E*    = {run['exact_energy']:.12f}", fontsize=8.2, color=MUTED)
    left.text(0.05, 0.38, f"|E_50 - E*| = {run['abs_delta_e']:.6e}", fontsize=8.2, color=ACCENT, weight="bold")
    left.text(0.05, 0.18, "same cutoff: n_ph(work) = n_ph(ref) = 3", fontsize=6.9, color=MUTED)

    mid.text(0.05, 0.88, "ALGORITHMIC WORK", fontsize=7, color=MUTED, weight="bold", va="top")
    mid.text(0.05, 0.69, f"S_alg = {accounting['S_alg']:,}", fontsize=10, color=INK, weight="bold")
    mid.text(0.05, 0.50, f"N_H_outer = {accounting['N_H_outer']:,}", fontsize=7.6, color=MUTED)
    mid.text(0.05, 0.38, f"N_H_refit = {accounting['N_H_refit']:,}", fontsize=7.6, color=MUTED)
    mid.text(0.05, 0.26, f"N_grad = {accounting['N_grad']:,}", fontsize=7.6, color=MUTED)
    mid.text(0.05, 0.14, f"N_metric = {accounting['N_metric']:,}", fontsize=7.6, color=MUTED)

    right.text(0.05, 0.88, "QISKIT ENDPOINT (k = 50)", fontsize=7, color=MUTED, weight="bold", va="top")
    right.text(0.05, 0.68, f"Table-I: N2q {hist_metrics['N2q']:,} | D2q {hist_metrics['D2q']:,} | Dc {hist_metrics['Dc']:,}", fontsize=7.8, color=INK, weight="bold")
    right.text(0.05, 0.48, f"Current: N2q {curr_metrics['N2q']:,} | D2q {curr_metrics['D2q']:,} | Dc {curr_metrics['Dc']:,}", fontsize=7.8, color=INK, weight="bold")
    right.text(0.05, 0.26, "Fixed-prefix replay: PASS", fontsize=7.5, color="#0f766e", weight="bold")
    right.text(0.05, 0.14, "All 50 Table-I prefixes compiled and endpoint-locked.", fontsize=6.9, color=MUTED)

    ax_notes = fig.add_axes([0.055, 0.055, 0.89, 0.19])
    ax_notes.axis("off")
    ax_notes.text(0.0, 0.97, "INTERPRETATION", fontsize=7.2, color=MUTED, weight="bold", va="top")
    interpretation = (
        "Accepted drop is energy_before_opt - energy_after_opt for the admitted macro after refit. "
        "Cumulative admission-path drop sums that one-time drop once per admitted occurrence and groups repeated labels; "
        "it is path dependent, favors early admissions, and is not leave-one-out attribution in the final ansatz. "
        "The normalized bar divides each generator-type raw sum by the summed prefix-to-prefix N2q increments "
        "from the locked Table-I Qiskit convention. This is a cost efficiency, not a correction for order bias. This is an explicitly "
        "requested macro-only ablation mockup and does not replace the canonical child-expanded Paper-I route."
    )
    ax_notes.text(0.0, 0.80, wrapped(interpretation, 72), fontsize=6.3, color=INK, va="top", linespacing=1.25)
    ax_notes.text(0.52, 0.97, "SOURCE LOCK", fontsize=7.2, color=MUTED, weight="bold", va="top")

    def abbreviated_hash(value: str | None) -> str:
        if not value:
            return "n/a"
        return f"{value[:12]}...{value[-12:]}"

    source_lines = [
        f"archive        {Path(source['archive_path']).name if source.get('archive_path') else 'not supplied'}",
        f"archive sha256 {abbreviated_hash(source.get('archive_sha256'))}",
        "member         weak_weak/json/result.json",
        f"member sha256  {abbreviated_hash(source['result_sha256'])}",
        f"prefix costs   {Path(source['prefix_qiskit_cost_path']).name}",
        f"cost sha256    {abbreviated_hash(source['prefix_qiskit_cost_sha256'])}",
        f"route          {run['route_profile']}",
        "full paths and hashes: companion provenance JSON",
    ]
    source_text = "\n".join(wrapped(line, 70) for line in source_lines)
    ax_notes.text(0.52, 0.80, source_text, fontsize=5.7, family="DejaVu Sans Mono", color=MUTED, va="top", linespacing=1.19)

    add_footer(fig, "Companion provenance JSON records the input hashes, metric definitions, and endpoint conventions.")
    pdf.savefig(fig)
    plt.close(fig)


def build_provenance(
    args: argparse.Namespace,
    reduced: dict[str, Any],
    prepared: dict[str, Any],
    qiskit: dict[str, Any],
    prefix_qiskit: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    run = reduced["run"]
    return {
        "schema": "paper_i_weak_weak_macro_only_operator_pool_energy_cost_diagnostic_v3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "diagnostic_mockup",
        "paper_promotion_status": "not_manuscript_evidence",
        "scope": {
            "paper": "Paper I",
            "regime": "weak_weak",
            "route": "SR-SNAKE macro-only physical lanes",
            "macro_only": True,
            "accepted_rounds": len(prepared["history"]),
            "same_cutoff_energy_comparison": True,
        },
        "source": {
            **source,
            "manifest_path": str(args.manifest_json),
            "manifest_sha256": sha256_file(args.manifest_json),
            "qiskit_sidecar_path": str(args.qiskit_json),
            "qiskit_sidecar_sha256": sha256_file(args.qiskit_json),
        },
        "definitions": {
            "accepted_energy_drop": "max(0, energy_before_opt - energy_after_opt) for the accepted macro after refit",
            "cumulative_admission_path_drop": "sum accepted_energy_drop once per admitted occurrence, grouped by generator type; repeated admissions accumulate; path dependent and not final-prefix leave-one-out attribution",
            "marginal_table_i_qiskit_N2q": "N2q(k) - N2q(k-1), where every prefix is independently compiled under the locked Table-I Qiskit convention; the preserved HF reference at k=0 has zero two-qubit cost",
            "normalized_drop_per_marginal_N2q": "for generator type t, sum accepted_energy_drop for admissions of t divided by sum marginal_table_i_qiskit_N2q for those same admissions; ratio of sums, not mean of per-round ratios",
            "denominator_choice": "the Table-I convention is used because all 50 N2q prefix increments are nonnegative and close to the locked N2q=1198 endpoint; current independently optimized FakeMarrakesh prefixes contain one negative N2q increment",
            "same_cutoff_error": "abs(E_k - E_exact) with n_ph_work = n_ph_reference = 3",
            "S_alg": "N_H_outer + N_H_refit + N_grad + N_metric after canonical same-state deduplication on the winning lineage",
        },
        "availability": {
            "accepted_drop_by_round": True,
            "S_alg_by_round": True,
            "final_prefix_leave_one_out_operator_attribution": False,
            "qiskit_N2q_by_round": True,
            "qiskit_D2q_by_round": True,
            "qiskit_endpoint_only": False,
        },
        "summary": {
            "final_energy": run["energy"],
            "exact_same_cutoff_energy": run["exact_energy"],
            "final_abs_delta_e": run["abs_delta_e"],
            "winning_lineage_accounting": run["accounting"],
            "qiskit_historical_displayed_convention": qiskit["historical_displayed_convention"]["metrics"],
            "qiskit_current_jr_fake_marrakesh_convention": qiskit["current_jr_fake_marrakesh_convention"]["metrics"],
            "fixed_prefix_replay_status": qiskit["fixed_prefix_replay"]["status"],
            "prefix_cost_curve_schema": prefix_qiskit["schema"],
            "prefix_cost_negative_delta_counts": prefix_qiskit["negative_delta_counts"],
            "prefix_cost_final_lock": prefix_qiskit["final_prefix_lock"],
        },
        "operator_summary": prepared["operators"],
        "generator_type_summary": prepared["generator_types"],
        "output_pdf": str(args.output_pdf),
    }


def main() -> None:
    args = parse_args()
    for path in (args.result_json, args.manifest_json, args.qiskit_json, args.prefix_qiskit_cost_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.source_archive and not args.source_archive.is_file():
        raise FileNotFoundError(args.source_archive)

    configure_matplotlib()
    reduced = reduce_result(args.result_json)
    qiskit = json.loads(args.qiskit_json.read_text())
    prefix_qiskit = json.loads(args.prefix_qiskit_cost_json.read_text())
    if qiskit["fixed_prefix_replay"]["status"] != "pass":
        raise ValueError("fixed-prefix replay sidecar is not passing")
    if prefix_qiskit.get("schema") != "paper_i_weak_weak_macro_prefix_qiskit_cost_curve_v1":
        raise ValueError("unexpected prefix Qiskit cost-curve schema")

    result_sha256 = sha256_file(args.result_json)
    if prefix_qiskit.get("source", {}).get("result_sha256") != result_sha256:
        raise ValueError("prefix Qiskit cost curve is not locked to the selected result JSON")
    if prefix_qiskit.get("source", {}).get("final_sidecar_sha256") != sha256_file(args.qiskit_json):
        raise ValueError("prefix Qiskit cost curve is not locked to the selected final sidecar")
    historical_metrics = qiskit["historical_displayed_convention"]["metrics"]
    if prefix_qiskit.get("final_prefix_lock", {}).get("historical_displayed") != historical_metrics:
        raise ValueError("prefix Qiskit cost endpoint does not match the final Table-I sidecar")
    prepared = prepare_data(reduced, prefix_qiskit)

    source = {
        "archive_path": str(args.source_archive) if args.source_archive else None,
        "archive_sha256": sha256_file(args.source_archive) if args.source_archive else None,
        "result_member": args.source_member,
        "result_extracted_path": str(args.result_json),
        "result_sha256": result_sha256,
        "prefix_qiskit_cost_path": str(args.prefix_qiskit_cost_json),
        "prefix_qiskit_cost_sha256": sha256_file(args.prefix_qiskit_cost_json),
    }

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output_pdf) as pdf:
        page_one(pdf, reduced, prepared)
        page_two(pdf, reduced, prepared, qiskit)
        page_three(pdf, reduced, prepared, qiskit, source)

    provenance = build_provenance(args, reduced, prepared, qiskit, prefix_qiskit, source)
    args.provenance_json.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.output_pdf}")
    print(f"wrote {args.provenance_json}")


if __name__ == "__main__":
    main()
