#!/usr/bin/env python3
"""Compare hardcoded vs Qiskit Hubbard pipelines for L=2,3,4,5.

Outputs:
- per-L metrics JSON
- per-L comparison PDFs
- overall summary JSON (`all_pass`)
- bundled PDF report + requested alias filename
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pprint
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = {
    "ground_state_energy_abs_delta": 1e-8,
    "fidelity_max_abs_delta": 1e-4,
    "energy_trotter_max_abs_delta": 1e-3,
    "n_up_site0_trotter_max_abs_delta": 5e-3,
    "n_dn_site0_trotter_max_abs_delta": 5e-3,
    "doublon_trotter_max_abs_delta": 1e-3,
}

TARGET_METRICS = [
    "fidelity",
    "energy_trotter",
    "n_up_site0_trotter",
    "n_dn_site0_trotter",
    "doublon_trotter",
]


@dataclass
class RunArtifacts:
    L: int
    hardcoded_json: Path
    hardcoded_pdf: Path
    qiskit_json: Path
    qiskit_pdf: Path
    compare_metrics_json: Path
    compare_pdf: Path


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _first_crossing(times: np.ndarray, vals: np.ndarray, thr: float) -> float | None:
    idx = np.where(vals > thr)[0]
    if idx.size == 0:
        return None
    return float(times[int(idx[0])])


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def _fp(x: float) -> str:
    # Round-trip-safe float text; avoids presentation rounding like %.12e.
    return repr(float(x))


def _delta_metric_definition_text() -> str:
    return (
        "ΔF(t)      = |F_hc(t) - F_qk(t)|\n"
        "ΔE_trot(t) = |E_trot_hc(t) - E_trot_qk(t)|\n"
        "Δn_up0(t)  = |n_up0_hc(t) - n_up0_qk(t)|\n"
        "Δn_dn0(t)  = |n_dn0_hc(t) - n_dn0_qk(t)|\n"
        "ΔD(t)      = |D_hc(t) - D_qk(t)|\n"
        "F_pipeline(t) is the pipeline's stored trajectory fidelity value "
        "(as computed internally vs that pipeline's exact evolution)."
    )


def _fmt_obj(obj: Any, *, width: int = 90) -> str:
    """Pretty-format a dict/list so no single line exceeds *width* chars."""
    formatted = pprint.pformat(obj, width=width, compact=True, sort_dicts=True)
    wrapped_lines: list[str] = []
    for line in formatted.splitlines():
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=width, subsequent_indent="  "))
    return "\n".join(wrapped_lines)


def _render_text_page(
    pdf: "PdfPages",
    lines: list[str],
    *,
    fontsize: int = 9,
    line_spacing: float = 0.028,
    max_line_width: int = 115,
) -> None:
    """Render *lines* onto a text-only PDF page with proper wrapping.

    Dict-like objects should already have been formatted via ``_fmt_obj``
    before being placed into *lines*.  This function applies a final
    safety wrap so that no single rendered line exceeds *max_line_width*
    characters, preventing right-edge truncation.
    """
    # Expand any remaining over-long lines.
    expanded: list[str] = []
    for raw in lines:
        if len(raw) <= max_line_width:
            expanded.append(raw)
        else:
            expanded.extend(
                textwrap.wrap(raw, width=max_line_width, subsequent_indent="    ")
            )

    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    x0 = 0.05
    y = 0.95
    for line in expanded:
        ax.text(
            x0,
            y,
            line,
            transform=ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=fontsize,
        )
        y -= line_spacing
        if y < 0.02:
            # Start a new page if we run out of room.
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(11.0, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.95

    pdf.savefig(fig)
    plt.close(fig)


def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _render_command_page(pdf: "PdfPages", command: str) -> None:
    lines = [
        "Executed Command",
        "",
        "Reference: pipelines/PIPELINE_RUN_GUIDE.md",
        "Script: pipelines/compare_hardcoded_vs_qiskit_pipeline.py",
        "",
        command,
    ]
    _render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=110)


def _compare_payloads(hardcoded: dict[str, Any], qiskit: dict[str, Any]) -> dict[str, Any]:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]

    if len(h_rows) != len(q_rows):
        raise ValueError("Trajectory length mismatch between hardcoded and qiskit outputs.")

    t_h = _arr(h_rows, "time")
    t_q = _arr(q_rows, "time")
    if not np.allclose(t_h, t_q, atol=1e-12, rtol=0.0):
        raise ValueError("Time-grid mismatch between hardcoded and qiskit outputs.")

    out: dict[str, Any] = {
        "time_grid": {
            "num_times": int(t_h.size),
            "t0": float(t_h[0]),
            "t_final": float(t_h[-1]),
            "dt": float(t_h[1] - t_h[0]) if t_h.size > 1 else 0.0,
        },
        "trajectory_deltas": {},
    }

    for key in TARGET_METRICS:
        h = _arr(h_rows, key)
        q = _arr(q_rows, key)
        d = np.abs(h - q)
        out["trajectory_deltas"][key] = {
            "max_abs_delta": float(np.max(d)),
            "mean_abs_delta": float(np.mean(d)),
            "final_abs_delta": float(d[-1]),
            "first_time_abs_delta_gt_1e-4": _first_crossing(t_h, d, 1e-4),
            "first_time_abs_delta_gt_1e-3": _first_crossing(t_h, d, 1e-3),
        }

    gs_h = float(hardcoded["ground_state"]["exact_energy"])
    gs_q = float(qiskit["ground_state"]["exact_energy"])
    out["ground_state_energy"] = {
        "hardcoded_exact_energy": gs_h,
        "qiskit_exact_energy": gs_q,
        "abs_delta": float(abs(gs_h - gs_q)),
    }

    checks = {
        "ground_state_energy_abs_delta": out["ground_state_energy"]["abs_delta"] <= THRESHOLDS["ground_state_energy_abs_delta"],
        "fidelity_max_abs_delta": out["trajectory_deltas"]["fidelity"]["max_abs_delta"] <= THRESHOLDS["fidelity_max_abs_delta"],
        "energy_trotter_max_abs_delta": out["trajectory_deltas"]["energy_trotter"]["max_abs_delta"] <= THRESHOLDS["energy_trotter_max_abs_delta"],
        "n_up_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_up_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_up_site0_trotter_max_abs_delta"],
        "n_dn_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_dn_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_dn_site0_trotter_max_abs_delta"],
        "doublon_trotter_max_abs_delta": out["trajectory_deltas"]["doublon_trotter"]["max_abs_delta"] <= THRESHOLDS["doublon_trotter_max_abs_delta"],
    }
    out["acceptance"] = {
        "thresholds": THRESHOLDS,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    return out


# ---------------------------------------------------------------------------
# Info-box helpers
# ---------------------------------------------------------------------------

def _sci(x: float) -> str:
    """Format a float in compact scientific notation (e.g. ``1.23e-06``)."""
    return f"{x:.2e}"


_INFO_BOX_SETTINGS_KEYS = [
    "L", "t", "u", "dv", "boundary", "ordering",
    "initial_state_source", "t_final", "num_times",
    "suzuki_order", "trotter_steps",
]


def _build_info_box_text(
    settings: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    """Return the multi-line text for a compact on-plot info box.

    Three sections (separated by a blank line):
    1. Run settings
    2. Thresholds
    3. Max |Δ| values + overall PASS/FAIL
    """
    # --- settings ---
    parts: list[str] = []
    for k in _INFO_BOX_SETTINGS_KEYS:
        v = settings.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    settings_line = "  ".join(parts)

    # --- thresholds ---
    thr = metrics["acceptance"]["thresholds"]
    thr_lines = [f"  {k}: {_sci(v)}" for k, v in sorted(thr.items())]

    # --- max_abs_delta + pass/fail ---
    td = metrics["trajectory_deltas"]
    delta_lines = []
    for key in sorted(td.keys()):
        delta_lines.append(f"  {key}: {_sci(td[key]['max_abs_delta'])}")
    gs_delta = metrics["ground_state_energy"]["abs_delta"]
    delta_lines.insert(0, f"  gs_energy: {_sci(gs_delta)}")
    verdict = "PASS" if metrics["acceptance"]["pass"] else "FAIL"

    return "\n".join([
        settings_line,
        "",
        "thresholds:",
        *thr_lines,
        "",
        "max |Δ|:",
        *delta_lines,
        f"result: {verdict}",
    ])


_INFO_BBOX = dict(
    boxstyle="round,pad=0.4",
    facecolor="white",
    edgecolor="#888888",
    alpha=0.80,
)


def _add_info_box(
    fig: Any,
    text: str,
    *,
    x: float = 0.01,
    y: float = 0.88,
    fontsize: float = 6.5,
) -> None:
    """Place a semi-transparent info box on *fig* in figure coordinates."""
    fig.text(
        x, y, text,
        transform=fig.transFigure,
        va="top", ha="left",
        fontsize=fontsize,
        family="monospace",
        bbox=_INFO_BBOX,
    )


def _render_info_page(pdf: "PdfPages", info_text: str, title: str = "") -> None:
    """Render the run-settings / metrics info box on its own dedicated page.

    This avoids overlapping plot content that occurs when the info box is
    placed on the same figure as subplots.
    """
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    ax.text(
        0.05, 0.92, info_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        family="monospace",
        bbox=_INFO_BBOX,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _autozoom(ax: Any, *arrays: np.ndarray, pad_frac: float = 0.05) -> None:
    """Set y-limits tightly around the data with *pad_frac* padding."""
    combined = np.concatenate([a for a in arrays if a.size > 0])
    lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
    span = hi - lo
    pad = span * pad_frac if span > 0 else 1e-8
    ax.set_ylim(lo - pad, hi + pad)


def _write_comparison_pdf(
    *,
    pdf_path: Path,
    L: int,
    hardcoded: dict[str, Any],
    qiskit: dict[str, Any],
    metrics: dict[str, Any],
    run_command: str,
) -> None:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]
    times = _arr(h_rows, "time")
    markevery = max(1, times.size // 25)

    def h(key: str) -> np.ndarray:
        return _arr(h_rows, key)

    def q(key: str) -> np.ndarray:
        return _arr(q_rows, key)

    exact_filtered = float(qiskit.get("vqe", {}).get("exact_filtered_energy", hardcoded["ground_state"]["exact_energy"]))
    hc_vqe = hardcoded.get("vqe", {}).get("energy")
    qk_vqe = qiskit.get("vqe", {}).get("energy")
    hc_vqe_val = float(hc_vqe) if hc_vqe is not None else np.nan
    qk_vqe_val = float(qk_vqe) if qk_vqe is not None else np.nan

    with PdfPages(str(pdf_path)) as pdf:
        _render_command_page(pdf, run_command)
        _info_text = _build_info_box_text(hardcoded.get("settings", {}), metrics)

        # --- Dedicated info page (no overlap with plots) ---
        _render_info_page(pdf, _info_text, title=f"L={L} Run Settings & Metrics Summary")

        # --- Page A: Fidelity + Energy (1x2) ---
        figA, (axF, axE) = plt.subplots(1, 2, figsize=(11.0, 8.5), sharex=True)

        axF.plot(times, q("fidelity"), label="Qiskit fidelity", color="#0b3d91", marker="o", markersize=3, markevery=markevery)
        axF.plot(times, h("fidelity"), label="Hardcoded fidelity", color="#e15759", linestyle="--", marker="^", markersize=3, markevery=markevery)
        axF.set_title("Fidelity")
        axF.set_xlabel("Time")
        axF.grid(alpha=0.25)
        axF.legend(fontsize=8)
        _autozoom(axF, h("fidelity"), q("fidelity"))

        axE.plot(times, q("energy_trotter"), label="Qiskit trotter", color="#2ca02c", marker="s", markersize=3, markevery=markevery)
        axE.plot(times, h("energy_trotter"), label="Hardcoded trotter", color="#d62728", linestyle="--", marker="v", markersize=3, markevery=markevery)
        axE.plot(times, q("energy_exact"), label="Qiskit exact", color="#111111", linewidth=1.4)
        axE.plot(times, h("energy_exact"), label="Hardcoded exact", color="#7f7f7f", linestyle=":", linewidth=1.2)
        axE.set_title("Energy")
        axE.set_xlabel("Time")
        axE.grid(alpha=0.25)
        axE.legend(fontsize=8)
        _autozoom(axE, h("energy_trotter"), q("energy_trotter"), h("energy_exact"), q("energy_exact"))

        figA.suptitle(f"Pipeline Comparison L={L}: Hardcoded vs Qiskit (Fidelity & Energy)", fontsize=13)
        figA.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(figA)
        plt.close(figA)

        # --- Page B: n_up + n_dn + Doublon (1x3) ---
        figB, (axUp, axDn, axD) = plt.subplots(1, 3, figsize=(11.0, 8.5), sharex=True)

        axUp.plot(times, q("n_up_site0_trotter"), label="Qiskit trotter", color="#17becf")
        axUp.plot(times, h("n_up_site0_trotter"), label="Hardcoded trotter", color="#0f7f8b", linestyle="--")
        axUp.set_title("Site-0 n_up")
        axUp.set_xlabel("Time")
        axUp.grid(alpha=0.25)
        axUp.legend(fontsize=7)
        _autozoom(axUp, h("n_up_site0_trotter"), q("n_up_site0_trotter"))

        axDn.plot(times, q("n_dn_site0_trotter"), label="Qiskit trotter", color="#9467bd")
        axDn.plot(times, h("n_dn_site0_trotter"), label="Hardcoded trotter", color="#6f4d8f", linestyle="--")
        axDn.set_title("Site-0 n_dn")
        axDn.set_xlabel("Time")
        axDn.grid(alpha=0.25)
        axDn.legend(fontsize=7)
        _autozoom(axDn, h("n_dn_site0_trotter"), q("n_dn_site0_trotter"))

        axD.plot(times, q("doublon_trotter"), label="Qiskit trotter", color="#e377c2")
        axD.plot(times, h("doublon_trotter"), label="Hardcoded trotter", color="#c251a1", linestyle="--")
        axD.set_title("Doublon")
        axD.set_xlabel("Time")
        axD.grid(alpha=0.25)
        axD.legend(fontsize=7)
        _autozoom(axD, h("doublon_trotter"), q("doublon_trotter"))

        figB.suptitle(f"Pipeline Comparison L={L}: Occupations & Doublon (auto-zoomed)", fontsize=13)
        figB.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(figB)
        plt.close(figB)

        figv, axesv = plt.subplots(1, 2, figsize=(11.0, 8.5))
        vx0, vx1 = axesv[0], axesv[1]
        energy_labels = ["Exact (filtered)", "Hardcoded VQE", "Qiskit VQE"]
        energy_vals = [exact_filtered, hc_vqe_val, qk_vqe_val]
        energy_colors = ["#111111", "#2ca02c", "#ff7f0e"]
        vx0.bar(np.arange(3), energy_vals, color=energy_colors, edgecolor="black", linewidth=0.4)
        vx0.set_xticks(np.arange(3))
        vx0.set_xticklabels(energy_labels, rotation=18, ha="right")
        vx0.set_ylabel("Energy")
        vx0.set_title(f"L={L} VQE Energy (Explicit)")
        vx0.grid(axis="y", alpha=0.25)

        err_h = abs(hc_vqe_val - exact_filtered) if np.isfinite(hc_vqe_val) else np.nan
        err_q = abs(qk_vqe_val - exact_filtered) if np.isfinite(qk_vqe_val) else np.nan
        vx1.bar([0, 1], [err_h, err_q], color=["#2ca02c", "#ff7f0e"], edgecolor="black", linewidth=0.4)
        vx1.set_xticks([0, 1])
        vx1.set_xticklabels(["|Hardcoded-Exact|", "|Qiskit-Exact|"], rotation=18, ha="right")
        vx1.set_ylabel("Absolute Error")
        vx1.set_title(f"L={L} VQE Absolute Error")
        vx1.grid(axis="y", alpha=0.25)

        figv.suptitle(
            "When initial_state_source=vqe, Trotter E(t=0) = ⟨ψ_vqe|H|ψ_vqe⟩ = VQE energy.\n"
            "VQE energy ≠ exact ground state energy unless VQE fully converged.",
            fontsize=10,
        )
        figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.91))
        pdf.savefig(figv)
        plt.close(figv)

        fig2, axes2 = plt.subplots(2, 2, figsize=(11.0, 8.5))
        bx00, bx01 = axes2[0, 0], axes2[0, 1]
        bx10, bx11 = axes2[1, 0], axes2[1, 1]

        bx00.plot(times, np.abs(h("fidelity") - q("fidelity")), color="#1f77b4")
        bx00.set_title("|ΔF(t)|")
        bx00.set_xlabel("Time")
        bx00.grid(alpha=0.25)

        bx01.plot(times, np.abs(h("energy_trotter") - q("energy_trotter")), color="#d62728")
        bx01.set_title("|ΔE_trot(t)|")
        bx01.set_xlabel("Time")
        bx01.grid(alpha=0.25)

        bx10.plot(times, np.abs(h("n_up_site0_trotter") - q("n_up_site0_trotter")), label="|Δn_up0|", color="#17becf")
        bx10.plot(times, np.abs(h("n_dn_site0_trotter") - q("n_dn_site0_trotter")), label="|Δn_dn0|", color="#9467bd")
        bx10.set_title("|Δn_up0(t)| and |Δn_dn0(t)|")
        bx10.set_xlabel("Time")
        bx10.grid(alpha=0.25)
        bx10.legend(fontsize=8)

        bx11.plot(times, np.abs(h("doublon_trotter") - q("doublon_trotter")), color="#8c564b")
        bx11.set_title("|ΔD(t)|")
        bx11.set_xlabel("Time")
        bx11.grid(alpha=0.25)

        fig2.suptitle(f"Delta Diagnostics L={L}", fontsize=14)
        fig2.text(
            0.5, 0.93,
            "ΔX(t) = |X_hc(t) − X_qk(t)|, where X_pipeline(t) is that pipeline's stored trajectory value.",
            ha="center", fontsize=8, style="italic",
        )
        fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.91))
        pdf.savefig(fig2)
        plt.close(fig2)

        td = metrics["trajectory_deltas"]
        lines = [
            f"L={L} metrics summary",
            "",
            "Delta metric definitions:",
            *_delta_metric_definition_text().splitlines(),
            "",
            f"ground_state_energy_abs_delta = {_fp(metrics['ground_state_energy']['abs_delta'])}",
            f"fidelity max/mean/final = {_fp(td['fidelity']['max_abs_delta'])} / {_fp(td['fidelity']['mean_abs_delta'])} / {_fp(td['fidelity']['final_abs_delta'])}",
            f"energy_trotter max/mean/final = {_fp(td['energy_trotter']['max_abs_delta'])} / {_fp(td['energy_trotter']['mean_abs_delta'])} / {_fp(td['energy_trotter']['final_abs_delta'])}",
            f"n_up_site0_trotter max/mean/final = {_fp(td['n_up_site0_trotter']['max_abs_delta'])} / {_fp(td['n_up_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_up_site0_trotter']['final_abs_delta'])}",
            f"n_dn_site0_trotter max/mean/final = {_fp(td['n_dn_site0_trotter']['max_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['final_abs_delta'])}",
            f"doublon_trotter max/mean/final = {_fp(td['doublon_trotter']['max_abs_delta'])} / {_fp(td['doublon_trotter']['mean_abs_delta'])} / {_fp(td['doublon_trotter']['final_abs_delta'])}",
            "",
            "checks:",
            *_fmt_obj(metrics["acceptance"]["checks"]).splitlines(),
            "",
            f"PASS = {metrics['acceptance']['pass']}",
        ]
        _render_text_page(pdf, lines)


def _write_bundle_pdf(
    *,
    bundle_path: Path,
    per_l_data: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]],
    overall_summary: dict[str, Any],
    isolation_check: dict[str, Any],
    include_per_l_pages: bool,
    run_command: str,
) -> None:
    lvals = [L for L, _h, _q, _m in per_l_data]
    exact_global = np.array([float(h["ground_state"]["exact_energy"]) for _L, h, _q, _m in per_l_data], dtype=float)
    exact_filtered = np.array(
        [
            float(q.get("vqe", {}).get("exact_filtered_energy", h["ground_state"]["exact_energy"]))
            for _L, h, q, _m in per_l_data
        ],
        dtype=float,
    )
    hc_vqe = np.array([float(h["vqe"].get("energy", np.nan)) if h["vqe"].get("energy") is not None else np.nan for _L, h, _q, _m in per_l_data], dtype=float)
    qk_vqe = np.array([float(q["vqe"].get("energy", np.nan)) if q["vqe"].get("energy") is not None else np.nan for _L, _h, q, _m in per_l_data], dtype=float)
    hc_qpe = np.array(
        [
            float(h.get("qpe", {}).get("energy_estimate", np.nan))
            if h.get("qpe", {}).get("energy_estimate") is not None
            else np.nan
            for _L, h, _q, _m in per_l_data
        ],
        dtype=float,
    )
    qk_qpe = np.array(
        [
            float(q.get("qpe", {}).get("energy_estimate", np.nan))
            if q.get("qpe", {}).get("energy_estimate") is not None
            else np.nan
            for _L, _h, q, _m in per_l_data
        ],
        dtype=float,
    )
    has_qpe_data = bool(np.isfinite(hc_qpe).any() or np.isfinite(qk_qpe).any())

    with PdfPages(str(bundle_path)) as pdf:
        _render_command_page(pdf, run_command)
        lines = [
            "Hardcoded vs Qiskit Pipeline Comparison Summary",
            "",
            f"generated_utc: {overall_summary['generated_utc']}",
            f"all_pass: {overall_summary['all_pass']}",
            f"l_values: {overall_summary['l_values']}",
            "trajectory_comparison_basis: trotter trajectories start from",
            "  each pipeline's selected initial_state_source (default: vqe)",
            "",
            "thresholds:",
            *_fmt_obj(THRESHOLDS).splitlines(),
            "",
            "hardcoded_qiskit_import_isolation:",
            *_fmt_obj(isolation_check).splitlines(),
            "",
            "Delta metric definitions:",
            *_delta_metric_definition_text().splitlines(),
            "",
            "Per-L pass flags:",
        ]
        for row in overall_summary["results"]:
            lines.append(f"L={row['L']} pass={row['pass']} metrics_json={row['metrics_json']}")
        _render_text_page(pdf, lines)

        x = np.arange(len(lvals), dtype=float)

        # VQE-only energy comparison page.
        fig1, ax1 = plt.subplots(figsize=(11.0, 8.5))
        ax1.plot(x, exact_filtered, marker="D", linewidth=2.0, color="#111111", label="Exact (filtered sector)")
        ax1.plot(x, hc_vqe, marker="s", linewidth=1.8, color="#2ca02c", label="Hardcoded VQE")
        ax1.plot(x, qk_vqe, marker="o", linewidth=1.8, color="#ff7f0e", label="Qiskit VQE")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"L={L}" for L in lvals])
        ax1.set_ylabel("Energy")
        ax1.set_title("VQE Energy Comparison")
        ax1.grid(alpha=0.25)
        ax1.legend()
        pdf.savefig(fig1)
        plt.close(fig1)

        # VQE-only error page.
        fig2, axes2 = plt.subplots(1, 2, figsize=(11.0, 8.5))
        ax2 = axes2[0]
        ax3 = axes2[1]
        width = 0.25
        ax2.bar(
            x - 0.5 * width,
            np.abs(hc_vqe - exact_filtered),
            width=width,
            color="#2ca02c",
            label="|Hardcoded VQE - Exact(filtered)|",
        )
        ax2.bar(
            x + 0.5 * width,
            np.abs(qk_vqe - exact_filtered),
            width=width,
            color="#ff7f0e",
            label="|Qiskit VQE - Exact(filtered)|",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L={L}" for L in lvals])
        ax2.set_ylabel("Absolute Error")
        ax2.set_title("VQE Absolute Error (linear)")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(fontsize=8)
        ax3.plot(
            x,
            np.abs(hc_vqe - qk_vqe),
            marker="o",
            linewidth=1.8,
            color="#9467bd",
            label="|Hardcoded VQE - Qiskit VQE|",
        )
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"L={L}" for L in lvals])
        ax3.set_ylabel("Absolute Delta")
        ax3.set_title("VQE Cross-Implementation Delta")
        ax3.grid(alpha=0.25)
        ax3.legend(fontsize=8)
        pdf.savefig(fig2)
        plt.close(fig2)

        if has_qpe_data:
            # QPE-only energy comparison page.
            fig3, axq = plt.subplots(figsize=(11.0, 8.5))
            axq.plot(x, exact_global, marker="X", linewidth=2.0, color="#111111", label="Exact (global)")
            axq.plot(x, hc_qpe, marker="^", linestyle="--", linewidth=1.8, color="#1f77b4", label="Hardcoded QPE")
            axq.plot(x, qk_qpe, marker="P", linestyle=":", linewidth=1.8, color="#0b3d91", label="Qiskit QPE")
            axq.set_xticks(x)
            axq.set_xticklabels([f"L={L}" for L in lvals])
            axq.set_ylabel("Energy")
            axq.set_title("QPE Energy Comparison")
            axq.grid(alpha=0.25)
            axq.legend()
            pdf.savefig(fig3)
            plt.close(fig3)

            # QPE-only error page.
            fig4, axes4 = plt.subplots(1, 2, figsize=(11.0, 8.5))
            ax4 = axes4[0]
            ax5 = axes4[1]
            ax4.bar(
                x - 0.5 * width,
                np.abs(hc_qpe - exact_global),
                width=width,
                color="#1f77b4",
                label="|Hardcoded QPE - Exact(global)|",
            )
            ax4.bar(
                x + 0.5 * width,
                np.abs(qk_qpe - exact_global),
                width=width,
                color="#0b3d91",
                label="|Qiskit QPE - Exact(global)|",
            )
            ax4.set_xticks(x)
            ax4.set_xticklabels([f"L={L}" for L in lvals])
            ax4.set_ylabel("Absolute Error")
            ax4.set_title("QPE Absolute Error (linear)")
            ax4.grid(axis="y", alpha=0.25)
            ax4.legend(fontsize=8)
            ax5.plot(
                x,
                np.abs(hc_qpe - qk_qpe),
                marker="s",
                linewidth=1.8,
                color="#8c564b",
                label="|Hardcoded QPE - Qiskit QPE|",
            )
            ax5.set_xticks(x)
            ax5.set_xticklabels([f"L={L}" for L in lvals])
            ax5.set_ylabel("Absolute Delta")
            ax5.set_title("QPE Cross-Implementation Delta")
            ax5.grid(alpha=0.25)
            ax5.legend(fontsize=8)
            pdf.savefig(fig4)
            plt.close(fig4)
        else:
            fig3 = plt.figure(figsize=(11.0, 8.5))
            axq = fig3.add_subplot(111)
            axq.axis("off")
            axq.text(
                0.02,
                0.98,
                "QPE comparison skipped: no finite QPE energy estimates were found in per-L payloads.",
                va="top",
                ha="left",
                family="monospace",
                fontsize=11,
            )
            pdf.savefig(fig3)
            plt.close(fig3)

        if include_per_l_pages:
            for L, hardcoded, qiskit, metrics in per_l_data:
                _write_comparison_pages_into_pdf(pdf, L, hardcoded, qiskit, metrics)


def _write_comparison_pages_into_pdf(
    pdf: PdfPages,
    L: int,
    hardcoded: dict[str, Any],
    qiskit: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]
    times = _arr(h_rows, "time")
    markevery = max(1, times.size // 25)

    def h(key: str) -> np.ndarray:
        return _arr(h_rows, key)

    def q(key: str) -> np.ndarray:
        return _arr(q_rows, key)

    exact_filtered = float(qiskit.get("vqe", {}).get("exact_filtered_energy", hardcoded["ground_state"]["exact_energy"]))
    hc_vqe = hardcoded.get("vqe", {}).get("energy")
    qk_vqe = qiskit.get("vqe", {}).get("energy")
    hc_vqe_val = float(hc_vqe) if hc_vqe is not None else np.nan
    qk_vqe_val = float(qk_vqe) if qk_vqe is not None else np.nan

    _info_text = _build_info_box_text(hardcoded.get("settings", {}), metrics)

    # --- Dedicated info page (no overlap with plots) ---
    _render_info_page(pdf, _info_text, title=f"Bundle L={L}: Run Settings & Metrics Summary")

    # --- Page A: Fidelity + Energy (1x2) ---
    figA, (axF, axE) = plt.subplots(1, 2, figsize=(11.0, 8.5), sharex=True)

    axF.plot(times, q("fidelity"), label="Qiskit fidelity", color="#0b3d91", marker="o", markersize=3, markevery=markevery)
    axF.plot(times, h("fidelity"), label="Hardcoded fidelity", color="#e15759", linestyle="--", marker="^", markersize=3, markevery=markevery)
    axF.set_title(f"L={L} Fidelity")
    axF.set_xlabel("Time")
    axF.grid(alpha=0.25)
    axF.legend(fontsize=8)
    _autozoom(axF, h("fidelity"), q("fidelity"))

    axE.plot(times, q("energy_trotter"), label="Qiskit trotter", color="#2ca02c")
    axE.plot(times, h("energy_trotter"), label="Hardcoded trotter", color="#d62728", linestyle="--")
    axE.plot(times, q("energy_exact"), label="Qiskit exact", color="#111111", linewidth=1.2)
    axE.plot(times, h("energy_exact"), label="Hardcoded exact", color="#7f7f7f", linestyle=":", linewidth=1.2)
    axE.set_title(f"L={L} Energy")
    axE.set_xlabel("Time")
    axE.grid(alpha=0.25)
    axE.legend(fontsize=8)
    _autozoom(axE, h("energy_trotter"), q("energy_trotter"), h("energy_exact"), q("energy_exact"))

    figA.suptitle(f"Bundle Page: L={L} Fidelity & Energy", fontsize=14)
    figA.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(figA)
    plt.close(figA)

    # --- Page B: n_up + n_dn + Doublon (1x3, auto-zoomed) ---
    figB, (axUp, axDn, axD) = plt.subplots(1, 3, figsize=(11.0, 8.5), sharex=True)

    axUp.plot(times, q("n_up_site0_trotter"), label="Qiskit trotter", color="#17becf")
    axUp.plot(times, h("n_up_site0_trotter"), label="Hardcoded trotter", color="#0f7f8b", linestyle="--")
    axUp.set_title(f"L={L} Site-0 n_up")
    axUp.set_xlabel("Time")
    axUp.grid(alpha=0.25)
    axUp.legend(fontsize=7)
    _autozoom(axUp, h("n_up_site0_trotter"), q("n_up_site0_trotter"))

    axDn.plot(times, q("n_dn_site0_trotter"), label="Qiskit trotter", color="#9467bd")
    axDn.plot(times, h("n_dn_site0_trotter"), label="Hardcoded trotter", color="#6f4d8f", linestyle="--")
    axDn.set_title(f"L={L} Site-0 n_dn")
    axDn.set_xlabel("Time")
    axDn.grid(alpha=0.25)
    axDn.legend(fontsize=7)
    _autozoom(axDn, h("n_dn_site0_trotter"), q("n_dn_site0_trotter"))

    axD.plot(times, q("doublon_trotter"), label="Qiskit trotter", color="#e377c2")
    axD.plot(times, h("doublon_trotter"), label="Hardcoded trotter", color="#c251a1", linestyle="--")
    axD.set_title(f"L={L} Doublon")
    axD.set_xlabel("Time")
    axD.grid(alpha=0.25)
    axD.legend(fontsize=7)
    _autozoom(axD, h("doublon_trotter"), q("doublon_trotter"))

    figB.suptitle(f"Bundle Page: L={L} Occupations & Doublon (auto-zoomed)", fontsize=13)
    figB.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(figB)
    plt.close(figB)

    figv, axesv = plt.subplots(1, 2, figsize=(11.0, 8.5))
    vx0, vx1 = axesv[0], axesv[1]
    energy_labels = ["Exact (filtered)", "Hardcoded VQE", "Qiskit VQE"]
    energy_vals = [exact_filtered, hc_vqe_val, qk_vqe_val]
    energy_colors = ["#111111", "#2ca02c", "#ff7f0e"]
    vx0.bar(np.arange(3), energy_vals, color=energy_colors, edgecolor="black", linewidth=0.4)
    vx0.set_xticks(np.arange(3))
    vx0.set_xticklabels(energy_labels, rotation=18, ha="right")
    vx0.set_ylabel("Energy")
    vx0.set_title(f"L={L} VQE Energy (Explicit)")
    vx0.grid(axis="y", alpha=0.25)

    err_h = abs(hc_vqe_val - exact_filtered) if np.isfinite(hc_vqe_val) else np.nan
    err_q = abs(qk_vqe_val - exact_filtered) if np.isfinite(qk_vqe_val) else np.nan
    vx1.bar([0, 1], [err_h, err_q], color=["#2ca02c", "#ff7f0e"], edgecolor="black", linewidth=0.4)
    vx1.set_xticks([0, 1])
    vx1.set_xticklabels(["|Hardcoded-Exact|", "|Qiskit-Exact|"], rotation=18, ha="right")
    vx1.set_ylabel("Absolute Error")
    vx1.set_title(f"L={L} VQE Absolute Error")
    vx1.grid(axis="y", alpha=0.25)

    figv.suptitle(
        "When initial_state_source=vqe, Trotter E(t=0) = ⟨ψ_vqe|H|ψ_vqe⟩ = VQE energy.\n"
        "VQE energy ≠ exact ground state energy unless VQE fully converged.",
        fontsize=10,
    )
    figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.91))
    pdf.savefig(figv)
    plt.close(figv)

    fig2, axes2 = plt.subplots(2, 2, figsize=(11.0, 8.5))
    bx00, bx01 = axes2[0, 0], axes2[0, 1]
    bx10, bx11 = axes2[1, 0], axes2[1, 1]

    bx00.plot(times, np.abs(h("fidelity") - q("fidelity")), color="#1f77b4")
    bx00.set_title("|ΔF(t)|")
    bx00.set_xlabel("Time")
    bx00.grid(alpha=0.25)

    bx01.plot(times, np.abs(h("energy_trotter") - q("energy_trotter")), color="#d62728")
    bx01.set_title("|ΔE_trot(t)|")
    bx01.set_xlabel("Time")
    bx01.grid(alpha=0.25)

    bx10.plot(times, np.abs(h("n_up_site0_trotter") - q("n_up_site0_trotter")), label="|Δn_up0|", color="#17becf")
    bx10.plot(times, np.abs(h("n_dn_site0_trotter") - q("n_dn_site0_trotter")), label="|Δn_dn0|", color="#9467bd")
    bx10.set_title("|Δn_up0(t)| and |Δn_dn0(t)|")
    bx10.set_xlabel("Time")
    bx10.grid(alpha=0.25)
    bx10.legend(fontsize=8)

    bx11.plot(times, np.abs(h("doublon_trotter") - q("doublon_trotter")), color="#8c564b")
    bx11.set_title("|ΔD(t)|")
    bx11.set_xlabel("Time")
    bx11.grid(alpha=0.25)

    fig2.suptitle(f"Bundle Delta Diagnostics L={L}", fontsize=14)
    fig2.text(
        0.5, 0.93,
        "ΔX(t) = |X_hc(t) − X_qk(t)|, where X_pipeline(t) is that pipeline's stored trajectory value.",
        ha="center", fontsize=8, style="italic",
    )
    fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.91))
    pdf.savefig(fig2)
    plt.close(fig2)

    td = metrics["trajectory_deltas"]
    lines = [
        f"Bundle metrics page L={L}",
        "",
        "Trotterization comparison uses each path's configured initial state.",
        "For VQE-init runs, both exact(t) and trotter(t) start from the VQE ansatz state.",
        "",
        "Delta metric definitions:",
        *_delta_metric_definition_text().splitlines(),
        "",
        f"ground_state_energy_abs_delta = {_fp(metrics['ground_state_energy']['abs_delta'])}",
        f"fidelity max/mean/final = {_fp(td['fidelity']['max_abs_delta'])} / {_fp(td['fidelity']['mean_abs_delta'])} / {_fp(td['fidelity']['final_abs_delta'])}",
        f"energy_trotter max/mean/final = {_fp(td['energy_trotter']['max_abs_delta'])} / {_fp(td['energy_trotter']['mean_abs_delta'])} / {_fp(td['energy_trotter']['final_abs_delta'])}",
        f"n_up_site0_trotter max/mean/final = {_fp(td['n_up_site0_trotter']['max_abs_delta'])} / {_fp(td['n_up_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_up_site0_trotter']['final_abs_delta'])}",
        f"n_dn_site0_trotter max/mean/final = {_fp(td['n_dn_site0_trotter']['max_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['final_abs_delta'])}",
        f"doublon_trotter max/mean/final = {_fp(td['doublon_trotter']['max_abs_delta'])} / {_fp(td['doublon_trotter']['mean_abs_delta'])} / {_fp(td['doublon_trotter']['final_abs_delta'])}",
        "",
        "checks:",
        *_fmt_obj(metrics["acceptance"]["checks"]).splitlines(),
        "",
        f"PASS = {metrics['acceptance']['pass']}",
    ]
    _render_text_page(pdf, lines)


def _check_hardcoded_qiskit_import_isolation(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    qpe_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_qpe_adapter_qiskit":
            qpe_fn = node
            break

    if qpe_fn is None:
        return {
            "pass": False,
            "reason": "_run_qpe_adapter_qiskit function not found",
        }

    start = int(qpe_fn.lineno)
    end = int(getattr(qpe_fn, "end_lineno", qpe_fn.body[-1].lineno if qpe_fn.body else qpe_fn.lineno))

    offending: list[dict[str, Any]] = []
    all_qiskit_imports: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("qiskit"):
                    rec = {"line": int(node.lineno), "module": alias.name}
                    all_qiskit_imports.append(rec)
                    if not (start <= int(node.lineno) <= end):
                        offending.append(rec)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("qiskit"):
                rec = {"line": int(node.lineno), "module": mod}
                all_qiskit_imports.append(rec)
                if not (start <= int(node.lineno) <= end):
                    offending.append(rec)

    return {
        "pass": len(offending) == 0,
        "qpe_adapter_range": {"start_line": start, "end_line": end},
        "qiskit_imports": all_qiskit_imports,
        "offending_imports": offending,
    }


def _run_command(cmd: list[str]) -> tuple[int, str, str]:
    t0 = time.perf_counter()
    _ai_log("compare_subprocess_start", cmd=cmd)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    _ai_log(
        "compare_subprocess_done",
        cmd=cmd,
        returncode=int(proc.returncode),
        elapsed_sec=round(time.perf_counter() - t0, 6),
        stdout_lines=int(len(proc.stdout.splitlines())),
        stderr_lines=int(len(proc.stderr.splitlines())),
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hardcoded and Qiskit Hubbard pipeline outputs.")
    parser.add_argument("--l-values", type=str, default="2,3,4,5")
    parser.add_argument("--run-pipelines", action="store_true", default=True)
    parser.add_argument("--no-run-pipelines", dest="run_pipelines", action="store_false")
    parser.add_argument(
        "--with-per-l-pdfs",
        action="store_true",
        help="Include per-L comparison pages inside the bundle and emit standalone per-L comparison PDFs.",
    )

    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--suzuki-order", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=64)

    parser.add_argument("--hardcoded-vqe-reps", type=int, default=2)
    parser.add_argument("--hardcoded-vqe-restarts", type=int, default=3)
    parser.add_argument("--hardcoded-vqe-seed", type=int, default=7)
    parser.add_argument("--hardcoded-vqe-maxiter", type=int, default=600)

    parser.add_argument("--qiskit-vqe-reps", type=int, default=2)
    parser.add_argument("--qiskit-vqe-restarts", type=int, default=3)
    parser.add_argument("--qiskit-vqe-seed", type=int, default=7)
    parser.add_argument("--qiskit-vqe-maxiter", type=int, default=600)

    parser.add_argument("--qpe-eval-qubits", type=int, default=5)
    parser.add_argument("--qpe-shots", type=int, default=256)
    parser.add_argument("--qpe-seed", type=int, default=11)
    parser.add_argument("--skip-qpe", action="store_true", help="Pass --skip-qpe to both pipeline runners.")

    parser.add_argument("--initial-state-source", choices=["exact", "vqe", "hf"], default="vqe")

    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("compare_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    l_values = [int(x.strip()) for x in str(args.l_values).split(",") if x.strip()]
    _ai_log("compare_l_values", l_values=l_values)

    command_log_path = artifacts_dir / "pipeline_commands_run.txt"
    command_log: list[str] = []

    run_artifacts: list[RunArtifacts] = []
    for L in l_values:
        _ai_log("compare_l_start", L=int(L))
        hardcoded_json = artifacts_dir / f"hardcoded_pipeline_L{L}.json"
        hardcoded_pdf = artifacts_dir / f"hardcoded_pipeline_L{L}.pdf"
        qiskit_json = artifacts_dir / f"qiskit_pipeline_L{L}.json"
        qiskit_pdf = artifacts_dir / f"qiskit_pipeline_L{L}.pdf"
        compare_metrics_json = artifacts_dir / f"hardcoded_vs_qiskit_pipeline_L{L}_metrics.json"
        compare_pdf = artifacts_dir / f"hardcoded_vs_qiskit_pipeline_L{L}_comparison.pdf"

        if args.run_pipelines:
            hc_cmd = [
                sys.executable,
                "pipelines/hardcoded_hubbard_pipeline.py",
                "--L", str(L),
                "--t", str(args.t),
                "--u", str(args.u),
                "--dv", str(args.dv),
                "--boundary", str(args.boundary),
                "--ordering", str(args.ordering),
                "--t-final", str(args.t_final),
                "--num-times", str(args.num_times),
                "--suzuki-order", str(args.suzuki_order),
                "--trotter-steps", str(args.trotter_steps),
                "--term-order", "sorted",
                "--vqe-reps", str(args.hardcoded_vqe_reps),
                "--vqe-restarts", str(args.hardcoded_vqe_restarts),
                "--vqe-seed", str(args.hardcoded_vqe_seed),
                "--vqe-maxiter", str(args.hardcoded_vqe_maxiter),
                "--qpe-eval-qubits", str(args.qpe_eval_qubits),
                "--qpe-shots", str(args.qpe_shots),
                "--qpe-seed", str(args.qpe_seed),
                "--initial-state-source", str(args.initial_state_source),
                "--output-json", str(hardcoded_json),
                "--output-pdf", str(hardcoded_pdf),
                "--skip-pdf",
            ]
            if args.skip_qpe:
                hc_cmd.append("--skip-qpe")
            command_log.append(" ".join(hc_cmd))
            code, out, err = _run_command(hc_cmd)
            if code != 0:
                raise RuntimeError(f"Hardcoded pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
            _ai_log("compare_l_hardcoded_done", L=int(L), json_path=str(hardcoded_json))

            qk_cmd = [
                sys.executable,
                "pipelines/qiskit_hubbard_baseline_pipeline.py",
                "--L", str(L),
                "--t", str(args.t),
                "--u", str(args.u),
                "--dv", str(args.dv),
                "--boundary", str(args.boundary),
                "--ordering", str(args.ordering),
                "--t-final", str(args.t_final),
                "--num-times", str(args.num_times),
                "--suzuki-order", str(args.suzuki_order),
                "--trotter-steps", str(args.trotter_steps),
                "--term-order", "sorted",
                "--vqe-reps", str(args.qiskit_vqe_reps),
                "--vqe-restarts", str(args.qiskit_vqe_restarts),
                "--vqe-seed", str(args.qiskit_vqe_seed),
                "--vqe-maxiter", str(args.qiskit_vqe_maxiter),
                "--qpe-eval-qubits", str(args.qpe_eval_qubits),
                "--qpe-shots", str(args.qpe_shots),
                "--qpe-seed", str(args.qpe_seed),
                "--initial-state-source", str(args.initial_state_source),
                "--output-json", str(qiskit_json),
                "--output-pdf", str(qiskit_pdf),
                "--skip-pdf",
            ]
            if args.skip_qpe:
                qk_cmd.append("--skip-qpe")
            command_log.append(" ".join(qk_cmd))
            code, out, err = _run_command(qk_cmd)
            if code != 0:
                raise RuntimeError(f"Qiskit pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
            _ai_log("compare_l_qiskit_done", L=int(L), json_path=str(qiskit_json))

        run_artifacts.append(
            RunArtifacts(
                L=L,
                hardcoded_json=hardcoded_json,
                hardcoded_pdf=hardcoded_pdf,
                qiskit_json=qiskit_json,
                qiskit_pdf=qiskit_pdf,
                compare_metrics_json=compare_metrics_json,
                compare_pdf=compare_pdf,
            )
        )
        _ai_log("compare_l_collection_done", L=int(L))

    if command_log:
        with command_log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(command_log) + "\n")
        _ai_log("compare_command_log_written", path=str(command_log_path), commands=int(len(command_log)))

    per_l_data: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    results_rows: list[dict[str, Any]] = []

    for row in run_artifacts:
        if not row.hardcoded_json.exists():
            raise FileNotFoundError(f"Missing hardcoded JSON for L={row.L}: {row.hardcoded_json}")
        if not row.qiskit_json.exists():
            raise FileNotFoundError(f"Missing qiskit JSON for L={row.L}: {row.qiskit_json}")

        hardcoded = json.loads(row.hardcoded_json.read_text(encoding="utf-8"))
        qiskit = json.loads(row.qiskit_json.read_text(encoding="utf-8"))
        metrics = _compare_payloads(hardcoded, qiskit)
        _ai_log(
            "compare_l_metrics",
            L=int(row.L),
            passed=bool(metrics["acceptance"]["pass"]),
            gs_abs_delta=float(metrics["ground_state_energy"]["abs_delta"]),
            fidelity_max_abs_delta=float(metrics["trajectory_deltas"]["fidelity"]["max_abs_delta"]),
            energy_trotter_max_abs_delta=float(metrics["trajectory_deltas"]["energy_trotter"]["max_abs_delta"]),
        )

        metrics_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "L": int(row.L),
            "hardcoded_json": str(row.hardcoded_json),
            "qiskit_json": str(row.qiskit_json),
            "metrics": metrics,
        }
        print(f"Delta metric definitions (L={row.L}):")
        print(_delta_metric_definition_text())
        row.compare_metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        if args.with_per_l_pdfs:
            _write_comparison_pdf(
                pdf_path=row.compare_pdf,
                L=row.L,
                hardcoded=hardcoded,
                qiskit=qiskit,
                metrics=metrics,
                run_command=run_command,
            )

        per_l_data.append((row.L, hardcoded, qiskit, metrics))
        results_rows.append(
            {
                "L": int(row.L),
                "pass": bool(metrics["acceptance"]["pass"]),
                "metrics_json": str(row.compare_metrics_json),
                "comparison_pdf": (str(row.compare_pdf) if args.with_per_l_pdfs else None),
                "hardcoded_settings": hardcoded.get("settings", {}),
                "qiskit_settings": qiskit.get("settings", {}),
                "ground_state_energy_abs_delta": float(metrics["ground_state_energy"]["abs_delta"]),
                "trajectory_max_abs_deltas": {
                    key: float(metrics["trajectory_deltas"][key]["max_abs_delta"]) for key in TARGET_METRICS
                },
            }
        )

    isolation_check = _check_hardcoded_qiskit_import_isolation(ROOT / "pipelines" / "hardcoded_hubbard_pipeline.py")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Hardcoded-first vs Qiskit Hubbard pipeline comparison.",
        "l_values": l_values,
        "thresholds": THRESHOLDS,
        "requested_run_settings": {
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "initial_state_source": str(args.initial_state_source),
            "skip_qpe": bool(args.skip_qpe),
        },
        "hardcoded_qiskit_import_isolation": isolation_check,
        "results": results_rows,
        "all_pass": bool(all(r["pass"] for r in results_rows) and isolation_check.get("pass", False)),
    }

    summary_json = artifacts_dir / "hardcoded_vs_qiskit_pipeline_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bundle_pdf = artifacts_dir / "hardcoded_vs_qiskit_all_results_bundle.pdf"
    _write_bundle_pdf(
        bundle_path=bundle_pdf,
        per_l_data=per_l_data,
        overall_summary=summary,
        isolation_check=isolation_check,
        include_per_l_pages=True,
        run_command=run_command,
    )
    _ai_log(
        "compare_main_done",
        summary_json=str(summary_json),
        bundle_pdf=str(bundle_pdf),
        all_pass=bool(summary["all_pass"]),
    )

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote bundle PDF:   {bundle_pdf}")
    if command_log:
        print(f"Wrote command log:  {command_log_path}")
    for row in results_rows:
        print(f"L={row['L']}: pass={row['pass']} metrics={row['metrics_json']} pdf={row['comparison_pdf']}")


if __name__ == "__main__":
    main()
