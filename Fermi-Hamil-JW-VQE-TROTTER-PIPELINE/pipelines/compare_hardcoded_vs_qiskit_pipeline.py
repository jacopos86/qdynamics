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
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.drives_time_potential import evaluate_drive_waveform

THRESHOLDS = {
    "ground_state_energy_abs_delta": 1e-8,
    "fidelity_max_abs_delta": 1e-4,
    "energy_static_trotter_max_abs_delta": 1e-3,
    "energy_total_trotter_max_abs_delta": 1e-3,
    "n_up_site0_trotter_max_abs_delta": 5e-3,
    "n_dn_site0_trotter_max_abs_delta": 5e-3,
    "doublon_trotter_max_abs_delta": 1e-3,
}

TARGET_METRICS = [
    "fidelity",
    "energy_static_trotter",
    "n_up_site0_trotter",
    "n_dn_site0_trotter",
    "doublon_trotter",
]

EXACT_LABEL_QISKIT = "Exact_Qiskit"
EXACT_LABEL_HARDCODE = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"


@dataclass
class RunArtifacts:
    L: int
    hc_json_by_ansatz: dict[str, Path]
    hc_pdf_by_ansatz: dict[str, Path]
    qk_json: Path
    qk_pdf: Path
    metrics_json: Path
    compare_pdf_by_ansatz: dict[str, Path]


def _artifact_tag(args: argparse.Namespace, L: int) -> str:
    """Build a compact, human-readable tag for artifact filenames.

    Format: ``L{L}_{vt|static}_t{t}_U{u}_S{trotter_steps}``

    Examples:
        ``L2_static_t1.0_U4.0_S32``
        ``L3_vt_t1.0_U4.0_S64``
    """
    mode = "vt" if getattr(args, "enable_drive", False) else "static"
    t_val = getattr(args, "t", 1.0)
    u_val = getattr(args, "u", 4.0)
    steps = getattr(args, "trotter_steps", 64)
    return f"L{L}_{mode}_t{t_val}_U{u_val}_S{steps}"


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


def _require_exact_filtered_energy(qiskit_payload: dict[str, Any], *, L: int | None = None) -> float:
    """Return sector-filtered exact energy, refusing any global-energy fallback."""
    vqe_block = qiskit_payload.get("vqe")
    if not isinstance(vqe_block, dict) or vqe_block.get("exact_filtered_energy") is None:
        suffix = f" for L={L}" if L is not None else ""
        raise KeyError(
            f"Missing qiskit.vqe.exact_filtered_energy{suffix}; "
            "refusing sector-to-total fallback in PDF comparisons."
        )
    val = float(vqe_block["exact_filtered_energy"])
    if not np.isfinite(val):
        suffix = f" for L={L}" if L is not None else ""
        raise ValueError(f"Non-finite qiskit.vqe.exact_filtered_energy{suffix}: {val!r}")
    return val


def _vqe_energy_sanity(
    payload: dict[str, Any],
    *,
    method_label: str,
    L: int | None = None,
) -> dict[str, Any]:
    """Check vqe.energy >= exact_filtered_energy - 1e-8 and return structured diagnostics."""
    suffix = f" for L={L}" if L is not None else ""
    vqe_block = payload.get("vqe")
    out: dict[str, Any] = {
        "method_label": str(method_label),
        "energy": None,
        "exact_filtered_energy": None,
        "delta_energy_minus_exact_filtered": None,
        "threshold": 1e-8,
        "passes_lower_bound": False,
        "reason": "",
    }
    if not isinstance(vqe_block, dict):
        out["reason"] = f"missing vqe payload{suffix}"
        return out

    energy_raw = vqe_block.get("energy")
    exact_raw = vqe_block.get("exact_filtered_energy")
    if energy_raw is None:
        out["reason"] = f"missing vqe.energy{suffix}"
        return out
    if exact_raw is None:
        out["reason"] = f"missing vqe.exact_filtered_energy{suffix}"
        return out

    energy = float(energy_raw)
    exact_filtered = float(exact_raw)
    out["energy"] = energy
    out["exact_filtered_energy"] = exact_filtered
    if not np.isfinite(energy) or not np.isfinite(exact_filtered):
        out["reason"] = f"non-finite energy/exact_filtered_energy{suffix}"
        return out

    delta = float(energy - exact_filtered)
    out["delta_energy_minus_exact_filtered"] = delta
    passes = bool(energy >= (exact_filtered - 1e-8))
    out["passes_lower_bound"] = passes
    out["reason"] = "ok" if passes else f"violates energy >= exact_filtered - 1e-8{suffix}"
    return out


def _fp(x: float) -> str:
    # Round-trip-safe float text; avoids presentation rounding like %.12e.
    return repr(float(x))


def _hardcoded_ansatz_label(hardcoded: dict[str, Any]) -> str:
    """Return normalized hardcoded ansatz label for PDF/report headers."""
    vqe_block = hardcoded.get("vqe")
    if isinstance(vqe_block, dict):
        ans = vqe_block.get("ansatz")
        if isinstance(ans, str) and ans.strip():
            return str(ans).strip().lower()
    settings = hardcoded.get("settings")
    if isinstance(settings, dict):
        ans = settings.get("vqe_ansatz")
        if isinstance(ans, str) and ans.strip():
            return str(ans).strip().lower()
    return "unknown"


def _normalize_ansatz_token(raw: Any) -> str:
    """Map arbitrary ansatz-ish labels to a compact token for report text."""
    txt = str(raw).strip().lower()
    if not txt:
        return "unknown"
    if "uccsd" in txt:
        return "uccsd"
    if "hva" in txt:
        return "hva"
    return txt


def _qiskit_ansatz_label(qiskit: dict[str, Any]) -> str:
    """Best-effort ansatz label extraction for qiskit payloads."""
    settings = qiskit.get("settings")
    if isinstance(settings, dict):
        ans = settings.get("vqe_ansatz")
        if isinstance(ans, str) and ans.strip():
            return _normalize_ansatz_token(ans)
    vqe_block = qiskit.get("vqe")
    if isinstance(vqe_block, dict):
        for key in ("ansatz", "ansatz_name", "ansatz_label", "method"):
            val = vqe_block.get(key)
            if isinstance(val, str) and val.strip():
                return _normalize_ansatz_token(val)
    return "unknown"


def _delta_metric_definition_text() -> str:
    return (
        "ΔF(t)      = |F_hc(t) - F_qk(t)|\n"
        "ΔE_trot(t) = |E_trot_hc(t) - E_trot_qk(t)|\n"
        "Δn_up0(t)  = |n_up0_hc(t) - n_up0_qk(t)|\n"
        "Δn_dn0(t)  = |n_dn0_hc(t) - n_dn0_qk(t)|\n"
        "ΔD(t)      = |D_hc(t) - D_qk(t)|\n"
        "F_pipeline(t) is the pipeline's stored trajectory subspace-fidelity "
        "value (as computed internally against the exact ground-manifold projector)."
    )


# ---- Chemical accuracy in natural units --------------------------------
_CHEM_ACCURACY_HARTREE = 1.6e-3  # 1.6 × 10⁻³ Ha


def _chemical_accuracy_lines(t_hop_hartree: float | None) -> list[str]:
    """Return text lines stating chemical accuracy in model natural units.

    The Hubbard model uses the hopping parameter *t* as the energy unit.
    Given the physical value of *t* in Hartree (``t_hop_hartree``), we
    convert:  ε_chem [t] = 1.6×10⁻³ Ha  /  t [Ha].
    """
    if t_hop_hartree is None or t_hop_hartree <= 0.0:
        return [
            "",
            "Chemical accuracy reference:",
            f"  1.6e-3 Hartree  (no --t-hartree supplied; cannot convert to natural units)",
        ]
    eps = _CHEM_ACCURACY_HARTREE / t_hop_hartree
    return [
        "",
        "Chemical accuracy reference:",
        f"  1.6e-3 Hartree = {eps:.6e} t   (using t_hop = {t_hop_hartree:.6e} Ha)",
        f"  Energy errors below {eps:.4e} t are within chemical accuracy.",
    ]


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


def _render_command_page(
    pdf: "PdfPages",
    command: str,
    *,
    hardcoded_ansatz: str | None = None,
) -> None:
    lines = [
        "Executed Command",
        "",
        "Reference: pipelines/PIPELINE_RUN_GUIDE.md",
        "Script: pipelines/compare_hardcoded_vs_qiskit_pipeline.py",
    ]
    if hardcoded_ansatz is not None:
        ans = str(hardcoded_ansatz).strip().lower() or "unknown"
        lines += [
            f"Hardcoded ansatz: {ans}",
            f"Compare mode: Hardcoded({ans}) vs Qiskit baseline",
        ]
    lines += [
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

    # --- total-energy comparison (always present in new JSONs) ---
    for total_key in ("energy_total_exact", "energy_total_trotter"):
        h_total = _arr_optional(h_rows, total_key)
        q_total = _arr_optional(q_rows, total_key)
        if h_total.size > 0 and q_total.size > 0 and h_total.size == q_total.size:
            d_total = np.abs(h_total - q_total)
            out["trajectory_deltas"][total_key] = {
                "max_abs_delta": float(np.max(d_total)),
                "mean_abs_delta": float(np.mean(d_total)),
                "final_abs_delta": float(d_total[-1]),
                "first_time_abs_delta_gt_1e-4": _first_crossing(t_h, d_total, 1e-4),
                "first_time_abs_delta_gt_1e-3": _first_crossing(t_h, d_total, 1e-3),
            }

    gs_h = float(hardcoded["ground_state"]["exact_energy"])
    gs_q = float(qiskit["ground_state"]["exact_energy"])
    out["ground_state_energy"] = {
        "hardcoded_exact_energy": gs_h,
        "qiskit_exact_energy": gs_q,
        "abs_delta": float(abs(gs_h - gs_q)),
    }

    hardcoded_sanity = _vqe_energy_sanity(hardcoded, method_label="hardcoded")
    qiskit_sanity = _vqe_energy_sanity(qiskit, method_label="qiskit")
    out["vqe_sanity"] = {
        "hardcoded": hardcoded_sanity,
        "qiskit": qiskit_sanity,
    }

    checks = {
        "ground_state_energy_abs_delta": out["ground_state_energy"]["abs_delta"] <= THRESHOLDS["ground_state_energy_abs_delta"],
        "fidelity_max_abs_delta": out["trajectory_deltas"]["fidelity"]["max_abs_delta"] <= THRESHOLDS["fidelity_max_abs_delta"],
        "energy_static_trotter_max_abs_delta": out["trajectory_deltas"]["energy_static_trotter"]["max_abs_delta"] <= THRESHOLDS["energy_static_trotter_max_abs_delta"],
        "n_up_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_up_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_up_site0_trotter_max_abs_delta"],
        "n_dn_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_dn_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_dn_site0_trotter_max_abs_delta"],
        "doublon_trotter_max_abs_delta": out["trajectory_deltas"]["doublon_trotter"]["max_abs_delta"] <= THRESHOLDS["doublon_trotter_max_abs_delta"],
        "vqe_energy_lower_bound_hardcoded": bool(hardcoded_sanity["passes_lower_bound"]),
        "vqe_energy_lower_bound_qiskit": bool(qiskit_sanity["passes_lower_bound"]),
    }
    # Add energy_total_trotter pass/fail gate when data is available.
    if "energy_total_trotter" in out["trajectory_deltas"]:
        checks["energy_total_trotter_max_abs_delta"] = (
            out["trajectory_deltas"]["energy_total_trotter"]["max_abs_delta"]
            <= THRESHOLDS["energy_total_trotter_max_abs_delta"]
        )
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
    "vqe_ansatz",
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


def _set_fidelity_ylim(
    ax: Any,
    *arrays: np.ndarray,
    pad_frac: float = 0.05,
    min_span: float = 1e-4,
) -> None:
    """Clamp fidelity axes to a physical upper bound of 1.0."""
    chunks = [np.asarray(a, dtype=float).ravel() for a in arrays if np.asarray(a).size > 0]
    if not chunks:
        ax.set_ylim(0.0, 1.0)
        return

    combined = np.concatenate(chunks)
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        ax.set_ylim(0.0, 1.0)
        return

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = hi - lo
    pad = span * pad_frac if span > 0.0 else 1e-8
    ymin = lo - pad
    ymax = 1.0
    ymin = max(0.0, min(float(ymin), ymax - min_span))
    if ymax - ymin < min_span:
        ymin = max(0.0, ymax - min_span)
    ax.set_ylim(ymin, ymax)


def _write_comparison_pdf(
    *,
    pdf_path: Path,
    L: int,
    hardcoded: dict[str, Any],
    qiskit: dict[str, Any],
    metrics: dict[str, Any],
    run_command: str,
    t_hartree: float | None = None,
) -> None:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]
    times = _arr(h_rows, "time")
    markevery = max(1, times.size // 25)

    def h(key: str) -> np.ndarray:
        return _arr(h_rows, key)

    def q(key: str) -> np.ndarray:
        return _arr(q_rows, key)

    exact_filtered = _require_exact_filtered_energy(qiskit, L=L)
    hc_vqe = hardcoded.get("vqe", {}).get("energy")
    qk_vqe = qiskit.get("vqe", {}).get("energy")
    hc_vqe_val = float(hc_vqe) if hc_vqe is not None else np.nan
    qk_vqe_val = float(qk_vqe) if qk_vqe is not None else np.nan
    hardcoded_ansatz = _hardcoded_ansatz_label(hardcoded)
    qiskit_ansatz = _qiskit_ansatz_label(qiskit)
    verdict = "PASS" if bool(metrics["acceptance"]["pass"]) else "FAIL"
    settings = hardcoded.get("settings", {})
    _ = run_command  # kept for signature compatibility; command text is logged to commands.txt

    with PdfPages(str(pdf_path)) as pdf:
        summary_lines = [
            f"Comparison Report Summary (L={L})",
            "",
            "Ansatz Used:",
            f"  - Hardcoded: {hardcoded_ansatz}",
            f"  - Qiskit baseline: {qiskit_ansatz}",
            "",
            "Run Settings:",
            f"  - t={settings.get('t')}  u={settings.get('u')}  dv={settings.get('dv')}",
            f"  - boundary={settings.get('boundary')}  ordering={settings.get('ordering')}",
            f"  - trotter_steps={settings.get('trotter_steps')}  suzuki_order={settings.get('suzuki_order')}",
            f"  - t_final={settings.get('t_final')}  num_times={settings.get('num_times')}",
            f"  - initial_state_source={settings.get('initial_state_source')}",
            "",
            "Topline:",
            f"  - overall_result: {verdict}",
            f"  - ground_state_energy_abs_delta: {_fp(metrics['ground_state_energy']['abs_delta'])}",
            f"  - max_abs_delta_fidelity: {_fp(metrics['trajectory_deltas']['fidelity']['max_abs_delta'])}",
            "",
            "Reference:",
            "  - Full executed commands are recorded in artifacts/commands.txt",
        ]
        _render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.029, max_line_width=112)

        _info_text = _build_info_box_text(hardcoded.get("settings", {}), metrics)

        # --- Dedicated info page (no overlap with plots) ---
        _render_info_page(
            pdf,
            _info_text,
            title=f"L={L} Run Settings & Metrics Summary (HC ansatz: {hardcoded_ansatz})",
        )

        # --- Page A: Subspace Fidelity + Energy (1x2) ---
        figA, (axF, axE) = plt.subplots(1, 2, figsize=(11.0, 8.5), sharex=True)

        axF.plot(times, q("fidelity"), label="Qiskit subspace fidelity", color="#0b3d91", marker="o", markersize=3, markevery=markevery)
        axF.plot(times, h("fidelity"), label="Hardcoded subspace fidelity", color="#e15759", linestyle="--", marker="^", markersize=3, markevery=markevery)
        axF.set_title("Subspace Fidelity")
        axF.set_xlabel("Time")
        axF.grid(alpha=0.25)
        axF.legend(fontsize=8)
        _set_fidelity_ylim(axF, h("fidelity"), q("fidelity"))

        axE.plot(times, q("energy_static_trotter"), label="Qiskit trotter (static)", color="#2ca02c", marker="s", markersize=3, markevery=markevery)
        axE.plot(times, h("energy_static_trotter"), label="Hardcoded trotter (static)", color="#d62728", linestyle="--", marker="v", markersize=3, markevery=markevery)
        axE.plot(times, q("energy_static_exact"), label=EXACT_LABEL_QISKIT, color="#111111", linewidth=1.4)
        axE.plot(times, h("energy_static_exact"), label=EXACT_LABEL_HARDCODE, color="#7f7f7f", linestyle=":", linewidth=1.2)

        # --- total-energy overlay (when drive active and differs from static) ---
        h_total_trot = _arr_optional(h_rows, "energy_total_trotter")
        q_total_trot = _arr_optional(q_rows, "energy_total_trotter")
        if h_total_trot.size == times.size and q_total_trot.size == times.size:
            h_static_trot = h("energy_static_trotter")
            q_static_trot = q("energy_static_trotter")
            if not (np.allclose(h_total_trot, h_static_trot, atol=1e-14) and
                    np.allclose(q_total_trot, q_static_trot, atol=1e-14)):
                axE.plot(times, q_total_trot, label="Qiskit trotter (total)", color="#17becf",
                         marker="D", markersize=2.5, markevery=markevery, linewidth=1.0, alpha=0.8)
                axE.plot(times, h_total_trot, label="Hardcoded trotter (total)", color="#ff7f0e",
                         linestyle="--", marker="<", markersize=2.5, markevery=markevery, linewidth=1.0, alpha=0.8)

        axE.set_title("Energy")
        axE.set_xlabel("Time")
        axE.grid(alpha=0.25)
        axE.legend(fontsize=7)
        _autozoom(axE, h("energy_static_trotter"), q("energy_static_trotter"), h("energy_static_exact"), q("energy_static_exact"))

        figA.suptitle(
            f"Pipeline Comparison L={L} [HC ansatz={hardcoded_ansatz}]: "
            "Hardcoded vs Qiskit (Subspace Fidelity & Energy)",
            fontsize=13,
        )
        figA.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(figA)
        plt.close(figA)

        # --- Page B: n_up + n_dn + Doublon (1x3) ---
        figB, (axUp, axDn, axD) = plt.subplots(1, 3, figsize=(11.0, 8.5), sharex=True)

        axUp.plot(times, q("n_up_site0_trotter"), label="Qiskit trotter", color="#17becf")
        axUp.plot(times, h("n_up_site0_trotter"), label="Hardcoded trotter", color="#0f7f8b", linestyle="--")
        axUp.plot(times, q("n_up_site0_exact"), label=f"{EXACT_LABEL_QISKIT} n_up0", color="#111111", linewidth=1.2)
        axUp.plot(times, h("n_up_site0_exact"), label=f"{EXACT_LABEL_HARDCODE} n_up0", color="#7f7f7f", linewidth=1.0, linestyle=":")
        axUp.set_title("Site-0 n_up")
        axUp.set_xlabel("Time")
        axUp.grid(alpha=0.25)
        axUp.legend(fontsize=7)
        _autozoom(axUp, h("n_up_site0_trotter"), q("n_up_site0_trotter"), h("n_up_site0_exact"), q("n_up_site0_exact"))

        axDn.plot(times, q("n_dn_site0_trotter"), label="Qiskit trotter", color="#9467bd")
        axDn.plot(times, h("n_dn_site0_trotter"), label="Hardcoded trotter", color="#6f4d8f", linestyle="--")
        axDn.plot(times, q("n_dn_site0_exact"), label=f"{EXACT_LABEL_QISKIT} n_dn0", color="#111111", linewidth=1.2)
        axDn.plot(times, h("n_dn_site0_exact"), label=f"{EXACT_LABEL_HARDCODE} n_dn0", color="#7f7f7f", linewidth=1.0, linestyle=":")
        axDn.set_title("Site-0 n_dn")
        axDn.set_xlabel("Time")
        axDn.grid(alpha=0.25)
        axDn.legend(fontsize=7)
        _autozoom(axDn, h("n_dn_site0_trotter"), q("n_dn_site0_trotter"), h("n_dn_site0_exact"), q("n_dn_site0_exact"))

        axD.plot(times, q("doublon_trotter"), label="Qiskit trotter", color="#e377c2")
        axD.plot(times, h("doublon_trotter"), label="Hardcoded trotter", color="#c251a1", linestyle="--")
        axD.plot(times, q("doublon_exact"), label=f"{EXACT_LABEL_QISKIT} doublon", color="#111111", linewidth=1.2)
        axD.plot(times, h("doublon_exact"), label=f"{EXACT_LABEL_HARDCODE} doublon", color="#7f7f7f", linewidth=1.0, linestyle=":")
        axD.set_title("Doublon")
        axD.set_xlabel("Time")
        axD.grid(alpha=0.25)
        axD.legend(fontsize=7)
        _autozoom(axD, h("doublon_trotter"), q("doublon_trotter"), h("doublon_exact"), q("doublon_exact"))

        figB.suptitle(
            f"Pipeline Comparison L={L} [HC ansatz={hardcoded_ansatz}]: "
            "Occupations & Doublon (auto-zoomed)",
            fontsize=13,
        )
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
        bx00.set_title("|ΔF_sub(t)|")
        bx00.set_xlabel("Time")
        bx00.grid(alpha=0.25)

        bx01.plot(times, np.abs(h("energy_static_trotter") - q("energy_static_trotter")), color="#d62728")
        bx01.set_title("|ΔE_static_trot(t)|")
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

        fig2.suptitle(f"Delta Diagnostics L={L} [HC ansatz={hardcoded_ansatz}]", fontsize=14)
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
            f"hardcoded_ansatz: {hardcoded_ansatz}",
            "",
            "Delta metric definitions:",
            *_delta_metric_definition_text().splitlines(),
            "",
            f"ground_state_energy_abs_delta = {_fp(metrics['ground_state_energy']['abs_delta'])}",
            f"subspace_fidelity max/mean/final = {_fp(td['fidelity']['max_abs_delta'])} / {_fp(td['fidelity']['mean_abs_delta'])} / {_fp(td['fidelity']['final_abs_delta'])}",
            f"energy_static_trotter max/mean/final = {_fp(td['energy_static_trotter']['max_abs_delta'])} / {_fp(td['energy_static_trotter']['mean_abs_delta'])} / {_fp(td['energy_static_trotter']['final_abs_delta'])}",
            f"n_up_site0_trotter max/mean/final = {_fp(td['n_up_site0_trotter']['max_abs_delta'])} / {_fp(td['n_up_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_up_site0_trotter']['final_abs_delta'])}",
            f"n_dn_site0_trotter max/mean/final = {_fp(td['n_dn_site0_trotter']['max_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['final_abs_delta'])}",
            f"doublon_trotter max/mean/final = {_fp(td['doublon_trotter']['max_abs_delta'])} / {_fp(td['doublon_trotter']['mean_abs_delta'])} / {_fp(td['doublon_trotter']['final_abs_delta'])}",
        ]
        # Include total-energy delta when both pipelines provide it.
        if "energy_total_trotter" in td:
            ett = td["energy_total_trotter"]
            lines.append(
                f"energy_total_trotter max/mean/final = {_fp(ett['max_abs_delta'])} / {_fp(ett['mean_abs_delta'])} / {_fp(ett['final_abs_delta'])}"
            )
        lines += [
            "",
            "checks:",
            *_fmt_obj(metrics["acceptance"]["checks"]).splitlines(),
            "",
            f"PASS = {metrics['acceptance']['pass']}",
            *_chemical_accuracy_lines(t_hartree),
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
    t_hartree: float | None = None,
) -> None:
    lvals = [L for L, _h, _q, _m in per_l_data]
    exact_global = np.array([float(h["ground_state"]["exact_energy"]) for _L, h, _q, _m in per_l_data], dtype=float)
    exact_filtered = np.array(
        [_require_exact_filtered_energy(q, L=int(_L)) for _L, _h, q, _m in per_l_data],
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
    hardcoded_ansatzes = overall_summary.get("requested_run_settings", {}).get("hardcoded_vqe_ansatzes", [])
    primary_hardcoded_ansatz = overall_summary.get("requested_run_settings", {}).get("primary_hardcoded_ansatz")
    qiskit_ansatzes = sorted({_qiskit_ansatz_label(q) for _L, _h, q, _m in per_l_data})
    _ = run_command  # kept for signature compatibility; command text is logged to commands.txt

    with PdfPages(str(bundle_path)) as pdf:
        lines = [
            "Hardcoded vs Qiskit Pipeline Comparison Summary",
            "",
            "Run Header:",
            f"  - generated_utc: {overall_summary['generated_utc']}",
            f"  - all_pass: {overall_summary['all_pass']}",
            f"  - l_values: {overall_summary['l_values']}",
            "",
            "Ansatz Used:",
            f"  - hardcoded set: {hardcoded_ansatzes}",
            f"  - hardcoded primary: {primary_hardcoded_ansatz}",
            f"  - qiskit baseline: {qiskit_ansatzes}",
            "",
            "Comparison Basis:",
            "  - trotter trajectories start from each pipeline's selected initial_state_source",
            f"  - exact_trajectory_labels: {EXACT_LABEL_HARDCODE}, {EXACT_LABEL_QISKIT}",
            f"  - exact_trajectory_method: {EXACT_METHOD}",
            "",
            "Thresholds:",
            *_fmt_obj(THRESHOLDS).splitlines(),
            "",
            "Hardcoded/Qiskit Import Isolation:",
            *_fmt_obj(isolation_check).splitlines(),
            "",
            "Delta metric definitions:",
            *_delta_metric_definition_text().splitlines(),
            "",
            "Per-L pages include explicit 'HC ansatz=...' labels in titles.",
            "",
            "Per-L Pass Flags:",
        ]
        for row in overall_summary["results"]:
            pass_by_ansatz = row.get("pass_by_ansatz")
            if isinstance(pass_by_ansatz, dict):
                lines.append(
                    f"L={row['L']} pass={row['pass']} pass_by_ansatz={pass_by_ansatz} "
                    f"metrics_json={row['metrics_json']}"
                )
            else:
                lines.append(f"L={row['L']} pass={row['pass']} metrics_json={row['metrics_json']}")
        lines.extend(_chemical_accuracy_lines(t_hartree))
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
                _write_comparison_pages_into_pdf(pdf, L, hardcoded, qiskit, metrics, t_hartree=t_hartree)


def _write_comparison_pages_into_pdf(
    pdf: PdfPages,
    L: int,
    hardcoded: dict[str, Any],
    qiskit: dict[str, Any],
    metrics: dict[str, Any],
    t_hartree: float | None = None,
) -> None:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]
    times = _arr(h_rows, "time")
    markevery = max(1, times.size // 25)

    def h(key: str) -> np.ndarray:
        return _arr(h_rows, key)

    def q(key: str) -> np.ndarray:
        return _arr(q_rows, key)

    exact_filtered = _require_exact_filtered_energy(qiskit, L=L)
    hc_vqe = hardcoded.get("vqe", {}).get("energy")
    qk_vqe = qiskit.get("vqe", {}).get("energy")
    hc_vqe_val = float(hc_vqe) if hc_vqe is not None else np.nan
    qk_vqe_val = float(qk_vqe) if qk_vqe is not None else np.nan
    hardcoded_ansatz = _hardcoded_ansatz_label(hardcoded)

    _info_text = _build_info_box_text(hardcoded.get("settings", {}), metrics)

    # --- Dedicated info page (no overlap with plots) ---
    _render_info_page(
        pdf,
        _info_text,
        title=f"Bundle L={L}: Run Settings & Metrics Summary (HC ansatz: {hardcoded_ansatz})",
    )

    # --- Page A: Subspace Fidelity + Energy (1x2) ---
    figA, (axF, axE) = plt.subplots(1, 2, figsize=(11.0, 8.5), sharex=True)

    axF.plot(times, q("fidelity"), label="Qiskit subspace fidelity", color="#0b3d91", marker="o", markersize=3, markevery=markevery)
    axF.plot(times, h("fidelity"), label="Hardcoded subspace fidelity", color="#e15759", linestyle="--", marker="^", markersize=3, markevery=markevery)
    axF.set_title(f"L={L} Subspace Fidelity")
    axF.set_xlabel("Time")
    axF.grid(alpha=0.25)
    axF.legend(fontsize=8)
    _set_fidelity_ylim(axF, h("fidelity"), q("fidelity"))

    axE.plot(times, q("energy_static_trotter"), label="Qiskit trotter", color="#2ca02c")
    axE.plot(times, h("energy_static_trotter"), label="Hardcoded trotter", color="#d62728", linestyle="--")
    axE.plot(times, q("energy_static_exact"), label=EXACT_LABEL_QISKIT, color="#111111", linewidth=1.2)
    axE.plot(times, h("energy_static_exact"), label=EXACT_LABEL_HARDCODE, color="#7f7f7f", linestyle=":", linewidth=1.2)
    axE.set_title(f"L={L} Energy")
    axE.set_xlabel("Time")
    axE.grid(alpha=0.25)
    axE.legend(fontsize=8)
    _autozoom(axE, h("energy_static_trotter"), q("energy_static_trotter"), h("energy_static_exact"), q("energy_static_exact"))

    figA.suptitle(
        f"Bundle Page: L={L} [HC ansatz={hardcoded_ansatz}] Subspace Fidelity & Energy",
        fontsize=14,
    )
    figA.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(figA)
    plt.close(figA)

    # --- Page B: n_up + n_dn + Doublon (1x3, auto-zoomed) ---
    figB, (axUp, axDn, axD) = plt.subplots(1, 3, figsize=(11.0, 8.5), sharex=True)

    axUp.plot(times, q("n_up_site0_trotter"), label="Qiskit trotter", color="#17becf")
    axUp.plot(times, h("n_up_site0_trotter"), label="Hardcoded trotter", color="#0f7f8b", linestyle="--")
    axUp.plot(times, q("n_up_site0_exact"), label=f"{EXACT_LABEL_QISKIT} n_up0", color="#111111", linewidth=1.2)
    axUp.plot(times, h("n_up_site0_exact"), label=f"{EXACT_LABEL_HARDCODE} n_up0", color="#7f7f7f", linewidth=1.0, linestyle=":")
    axUp.set_title(f"L={L} Site-0 n_up")
    axUp.set_xlabel("Time")
    axUp.grid(alpha=0.25)
    axUp.legend(fontsize=7)
    _autozoom(axUp, h("n_up_site0_trotter"), q("n_up_site0_trotter"), h("n_up_site0_exact"), q("n_up_site0_exact"))

    axDn.plot(times, q("n_dn_site0_trotter"), label="Qiskit trotter", color="#9467bd")
    axDn.plot(times, h("n_dn_site0_trotter"), label="Hardcoded trotter", color="#6f4d8f", linestyle="--")
    axDn.plot(times, q("n_dn_site0_exact"), label=f"{EXACT_LABEL_QISKIT} n_dn0", color="#111111", linewidth=1.2)
    axDn.plot(times, h("n_dn_site0_exact"), label=f"{EXACT_LABEL_HARDCODE} n_dn0", color="#7f7f7f", linewidth=1.0, linestyle=":")
    axDn.set_title(f"L={L} Site-0 n_dn")
    axDn.set_xlabel("Time")
    axDn.grid(alpha=0.25)
    axDn.legend(fontsize=7)
    _autozoom(axDn, h("n_dn_site0_trotter"), q("n_dn_site0_trotter"), h("n_dn_site0_exact"), q("n_dn_site0_exact"))

    axD.plot(times, q("doublon_trotter"), label="Qiskit trotter", color="#e377c2")
    axD.plot(times, h("doublon_trotter"), label="Hardcoded trotter", color="#c251a1", linestyle="--")
    axD.plot(times, q("doublon_exact"), label=f"{EXACT_LABEL_QISKIT} doublon", color="#111111", linewidth=1.2)
    axD.plot(times, h("doublon_exact"), label=f"{EXACT_LABEL_HARDCODE} doublon", color="#7f7f7f", linewidth=1.0, linestyle=":")
    axD.set_title(f"L={L} Doublon")
    axD.set_xlabel("Time")
    axD.grid(alpha=0.25)
    axD.legend(fontsize=7)
    _autozoom(axD, h("doublon_trotter"), q("doublon_trotter"), h("doublon_exact"), q("doublon_exact"))

    figB.suptitle(
        f"Bundle Page: L={L} [HC ansatz={hardcoded_ansatz}] Occupations & Doublon (auto-zoomed)",
        fontsize=13,
    )
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
    bx00.set_title("|ΔF_sub(t)|")
    bx00.set_xlabel("Time")
    bx00.grid(alpha=0.25)

    bx01.plot(times, np.abs(h("energy_static_trotter") - q("energy_static_trotter")), color="#d62728")
    bx01.set_title("|ΔE_static_trot(t)|")
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

    fig2.suptitle(f"Bundle Delta Diagnostics L={L} [HC ansatz={hardcoded_ansatz}]", fontsize=14)
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
        f"hardcoded_ansatz: {hardcoded_ansatz}",
        "",
        "Trotterization comparison uses each path's configured initial state.",
        f"Trajectory labels: {EXACT_LABEL_HARDCODE} and {EXACT_LABEL_QISKIT}.",
        f"Exact trajectory method: {EXACT_METHOD}.",
        "For VQE-init runs, both exact(t) and trotter(t) start from the VQE ansatz state.",
        "",
        "Delta metric definitions:",
        *_delta_metric_definition_text().splitlines(),
        "",
        f"ground_state_energy_abs_delta = {_fp(metrics['ground_state_energy']['abs_delta'])}",
        f"subspace_fidelity max/mean/final = {_fp(td['fidelity']['max_abs_delta'])} / {_fp(td['fidelity']['mean_abs_delta'])} / {_fp(td['fidelity']['final_abs_delta'])}",
        f"energy_static_trotter max/mean/final = {_fp(td['energy_static_trotter']['max_abs_delta'])} / {_fp(td['energy_static_trotter']['mean_abs_delta'])} / {_fp(td['energy_static_trotter']['final_abs_delta'])}",
        f"n_up_site0_trotter max/mean/final = {_fp(td['n_up_site0_trotter']['max_abs_delta'])} / {_fp(td['n_up_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_up_site0_trotter']['final_abs_delta'])}",
        f"n_dn_site0_trotter max/mean/final = {_fp(td['n_dn_site0_trotter']['max_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['final_abs_delta'])}",
        f"doublon_trotter max/mean/final = {_fp(td['doublon_trotter']['max_abs_delta'])} / {_fp(td['doublon_trotter']['mean_abs_delta'])} / {_fp(td['doublon_trotter']['final_abs_delta'])}",
        "",
        "checks:",
        *_fmt_obj(metrics["acceptance"]["checks"]).splitlines(),
        "",
        f"PASS = {metrics['acceptance']['pass']}",
        *_chemical_accuracy_lines(t_hartree),
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


# ---------------------------------------------------------------------------
# Drive flag passthrough
# ---------------------------------------------------------------------------

# Default values kept in sync with individual pipeline parse_args().
_DRIVE_FLAG_DEFAULTS: dict[str, Any] = {
    "enable_drive": False,
    "drive_A": 1.0,
    "drive_omega": 1.0,
    "drive_tbar": 5.0,
    "drive_phi": 0.0,
    "drive_pattern": "staggered",
    "drive_custom_s": None,
    "drive_include_identity": False,
    "drive_time_sampling": "midpoint",
    "drive_t0": 0.0,
    "exact_steps_multiplier": 1,
}


def _parse_hardcoded_vqe_ansatzes(raw: str) -> list[str]:
    allowed = {"uccsd", "hva"}
    vals = [str(x).strip().lower() for x in str(raw).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("--hardcoded-vqe-ansatzes must contain at least one ansatz name.")
    out: list[str] = []
    for name in vals:
        if name not in allowed:
            raise ValueError(f"Unsupported ansatz '{name}' in --hardcoded-vqe-ansatzes; allowed: {sorted(allowed)}")
        if name not in out:
            out.append(name)
    return out


def _build_drive_args(args: argparse.Namespace) -> list[str]:
    """Return CLI token list to forward drive settings to a sub-pipeline.

    Returns an empty list when ``--enable-drive`` is not set, preserving
    bit-for-bit backward compatibility with drive-free runs.
    """
    if not bool(getattr(args, "enable_drive", False)):
        return []

    tokens: list[str] = ["--enable-drive"]
    tokens += ["--drive-A", str(float(args.drive_A))]
    tokens += ["--drive-omega", str(float(args.drive_omega))]
    tokens += ["--drive-tbar", str(float(args.drive_tbar))]
    tokens += ["--drive-phi", str(float(args.drive_phi))]
    tokens += ["--drive-pattern", str(args.drive_pattern)]
    if args.drive_custom_s is not None:
        tokens += ["--drive-custom-s", str(args.drive_custom_s)]
    if bool(args.drive_include_identity):
        tokens.append("--drive-include-identity")
    tokens += ["--drive-time-sampling", str(args.drive_time_sampling)]
    tokens += ["--drive-t0", str(float(args.drive_t0))]
    tokens += ["--exact-steps-multiplier", str(int(args.exact_steps_multiplier))]
    return tokens


# ---------------------------------------------------------------------------
# Amplitude-comparison helpers
# ---------------------------------------------------------------------------

# Threshold for the safe-test: drive-disabled vs drive-enabled-A0=0 must agree
# to this tolerance (they are mathematically identical waveforms).
_SAFE_TEST_THRESHOLD: float = 1e-10


def _build_drive_args_with_amplitude(args: argparse.Namespace, amplitude: float) -> list[str]:
    """Return CLI tokens that force-enable the drive and override --drive-A.

    All other drive knobs are taken from *args* unchanged.  This is used by
    the amplitude-comparison flow, which always enables the drive regardless
    of whether ``--enable-drive`` was passed on the compare-pipeline command
    line.
    """
    tokens: list[str] = ["--enable-drive"]
    tokens += ["--drive-A", str(float(amplitude))]
    tokens += ["--drive-omega", str(float(args.drive_omega))]
    tokens += ["--drive-tbar", str(float(args.drive_tbar))]
    tokens += ["--drive-phi", str(float(args.drive_phi))]
    tokens += ["--drive-pattern", str(args.drive_pattern)]
    if args.drive_custom_s is not None:
        tokens += ["--drive-custom-s", str(args.drive_custom_s)]
    if bool(args.drive_include_identity):
        tokens.append("--drive-include-identity")
    tokens += ["--drive-time-sampling", str(args.drive_time_sampling)]
    tokens += ["--drive-t0", str(float(args.drive_t0))]
    tokens += ["--exact-steps-multiplier", str(int(args.exact_steps_multiplier))]
    return tokens


def _safe_test_check(
    no_drive_hc: dict[str, Any],
    no_drive_qk: dict[str, Any],
    zero_amp_hc: dict[str, Any],
    zero_amp_qk: dict[str, Any],
) -> dict[str, Any]:
    """Compare drive-disabled trajectories against drive-enabled-A0=0 trajectories.

    A zero-amplitude drive is mathematically a no-op: v(t)=0 for all t.
    Both pipelines must therefore produce numerically identical trajectories.
    If they diverge beyond *_SAFE_TEST_THRESHOLD* the drive implementation
    has a bug.

    Returns a dict::

        {
            "passed": bool,
            "threshold": float,
            "hc": {"max_fidelity_delta": float, "max_energy_delta": float},
            "qk": {"max_fidelity_delta": float, "max_energy_delta": float},
        }
    """
    def _max_abs(rows_a: list[dict], rows_b: list[dict], key: str) -> float:
        a = _arr(rows_a, key)
        b = _arr(rows_b, key)
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return float("nan")
        return float(np.max(np.abs(a - b)))

    nd_hc = no_drive_hc.get("trajectory", [])
    za_hc = zero_amp_hc.get("trajectory", [])
    nd_qk = no_drive_qk.get("trajectory", [])
    za_qk = zero_amp_qk.get("trajectory", [])

    hc_fid = _max_abs(nd_hc, za_hc, "fidelity")
    hc_ene = _max_abs(nd_hc, za_hc, "energy_static_trotter")
    qk_fid = _max_abs(nd_qk, za_qk, "fidelity")
    qk_ene = _max_abs(nd_qk, za_qk, "energy_static_trotter")

    all_vals = [v for v in (hc_fid, hc_ene, qk_fid, qk_ene) if np.isfinite(v)]
    passed = bool(all_vals) and max(all_vals) < _SAFE_TEST_THRESHOLD

    return {
        "passed": passed,
        "threshold": _SAFE_TEST_THRESHOLD,
        "hc": {"max_fidelity_delta": hc_fid, "max_energy_delta": hc_ene},
        "qk": {"max_fidelity_delta": qk_fid, "max_energy_delta": qk_ene},
    }


def _arr_optional(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    out: list[float] = []
    for row in rows:
        if key not in row:
            return np.array([], dtype=float)
        out.append(float(row[key]))
    return np.array(out, dtype=float)


def _matrix_optional(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if not rows or key not in rows[0]:
        return np.array([[]], dtype=float)
    first = rows[0].get(key)
    if not isinstance(first, list):
        return np.array([[]], dtype=float)
    width = len(first)
    if width == 0:
        return np.array([[]], dtype=float)
    mat = np.zeros((len(rows), width), dtype=float)
    for ridx, row in enumerate(rows):
        vals = row.get(key)
        if not isinstance(vals, list) or len(vals) != width:
            return np.array([[]], dtype=float)
        mat[ridx, :] = np.array([float(x) for x in vals], dtype=float)
    return mat


def _observables_series(rows: list[dict[str, Any]], kind: str, impl: str) -> np.ndarray:
    impl_norm = str(impl).strip().lower()
    if kind == "fidelity":
        return _arr_optional(rows, "fidelity")
    if kind == "energy":
        arr = _arr_optional(rows, f"energy_static_{impl_norm}")
        if arr.size > 0:
            return arr
        return _arr_optional(rows, f"energy_{impl_norm}")
    if kind == "doublon":
        arr = _arr_optional(rows, f"doublon_avg_{impl_norm}")
        if arr.size > 0:
            return arr
        return _arr_optional(rows, f"doublon_{impl_norm}")
    if kind == "staggered":
        return _arr_optional(rows, f"staggered_{impl_norm}")
    if kind == "density_site0":
        mat = _matrix_optional(rows, f"n_site_{impl_norm}")
        if mat.size > 0 and mat.shape[1] > 0:
            return mat[:, 0]
        up = _arr_optional(rows, f"n_up_site0_{impl_norm}")
        dn = _arr_optional(rows, f"n_dn_site0_{impl_norm}")
        if up.size > 0 and dn.size == up.size:
            return up + dn
        return up
    raise ValueError(f"Unknown observable kind: {kind!r}")


def _aligned_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.size == 0 or b.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(int(a.size), int(b.size))
    return np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)


def _max_abs_and_rms(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aa, bb = _aligned_pair(a, b)
    if aa.size == 0:
        return float("nan"), float("nan")
    diff = aa - bb
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff * diff)))


def _response_delta(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    x0, x1 = _aligned_pair(a0, a1)
    if x0.size == 0:
        return np.array([], dtype=float)
    return np.asarray(x1 - x0, dtype=float)


def _response_stats(a0: np.ndarray, a1: np.ndarray) -> tuple[float, float]:
    delta = _response_delta(a0, a1)
    if delta.size == 0:
        return float("nan"), float("nan")
    return float(np.max(np.abs(delta))), float(delta[-1])


def _fmt_metric(x: float, *, nan_text: str = "N/A") -> str:
    if not np.isfinite(float(x)):
        return nan_text
    return f"{float(x):.3e}"


def _render_compact_table(
    ax: Any,
    *,
    title: str,
    col_labels: list[str],
    rows: list[list[str]],
    fontsize: int = 7,
) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=6)
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.3)


def _safe_delta_series(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    key: str,
) -> np.ndarray:
    a = _arr_optional(rows_a, key)
    b = _arr_optional(rows_b, key)
    aa, bb = _aligned_pair(a, b)
    if aa.size == 0:
        return np.array([], dtype=float)
    return np.abs(aa - bb)


def _safe_test_plot_gate(
    safe: dict[str, Any],
    *,
    near_factor: float,
    verbose: bool,
) -> tuple[bool, float]:
    vals = [
        float(safe["hc"]["max_fidelity_delta"]),
        float(safe["hc"]["max_energy_delta"]),
        float(safe["qk"]["max_fidelity_delta"]),
        float(safe["qk"]["max_energy_delta"]),
    ]
    finite = [v for v in vals if np.isfinite(v)]
    max_safe = max(finite) if finite else float("nan")
    thr = float(safe["threshold"])
    near = bool(np.isfinite(max_safe) and np.isfinite(thr) and thr > 0.0 and max_safe >= (thr / max(1.0, float(near_factor))))
    show = bool((not bool(safe["passed"])) or near or bool(verbose))
    return show, max_safe


def _drive_config_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "drive_omega": float(args.drive_omega),
        "drive_tbar": float(args.drive_tbar),
        "drive_phi": float(args.drive_phi),
        "drive_t0": float(args.drive_t0),
    }


def _fft_response(signal: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if signal.size < 4 or times.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(int(signal.size), int(times.size))
    y = np.asarray(signal[:n], dtype=float)
    t = np.asarray(times[:n], dtype=float)
    dt = float(t[1] - t[0])
    if not np.isfinite(dt) or dt <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)
    y0 = y - float(np.mean(y))
    window = np.hanning(n)
    spec = np.fft.rfft(y0 * window)
    freq = np.fft.rfftfreq(n, d=dt)
    omega = 2.0 * np.pi * freq
    mag = np.abs(spec)
    return omega, mag


def _run_amplitude_comparison_for_l(
    args: argparse.Namespace,
    L: int,
    A0: float,
    A1: float,
    json_dir: Path,
    hardcoded_vqe_ansatz: str,
) -> dict[str, Any]:
    """Run hardcoded + qiskit pipelines three times for *L*: disabled, A0, A1.

    Returns::

        {
            "disabled": {"hc": payload_dict, "qk": payload_dict},
            "A0":       {"hc": payload_dict, "qk": payload_dict},
            "A1":       {"hc": payload_dict, "qk": payload_dict},
        }

    All intermediate JSONs are written under *json_dir* with an
    ``amp_H_`` / ``amp_Q_`` prefix so they do not overwrite the main
    pipeline outputs.
    """
    tag = _artifact_tag(args, L)

    def _base_cmd(pipeline_file: str, json_out: Path, *, include_hardcoded_ansatz: bool) -> list[str]:
        cmd = [
            sys.executable,
            f"pipelines/{pipeline_file}",
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
            "--fidelity-subspace-energy-tol", str(args.fidelity_subspace_energy_tol),
            "--term-order", "sorted",
            "--vqe-reps", str(args.hardcoded_vqe_reps),
            "--vqe-restarts", str(args.hardcoded_vqe_restarts),
            "--vqe-seed", str(args.hardcoded_vqe_seed),
            "--vqe-maxiter", str(args.hardcoded_vqe_maxiter),
            "--vqe-method", str(args.hardcoded_vqe_method),
            "--initial-state-source", str(args.initial_state_source),
            "--output-json", str(json_out),
            "--output-pdf", str(json_out).replace(".json", ".pdf"),
            "--skip-pdf",
            "--skip-qpe",
        ]
        if include_hardcoded_ansatz:
            cmd += ["--vqe-ansatz", str(hardcoded_vqe_ansatz)]
        return cmd

    def _base_cmd_qk(json_out: Path) -> list[str]:
        cmd = _base_cmd("qiskit_hubbard_baseline_pipeline.py", json_out, include_hardcoded_ansatz=False)
        # Replace hardcoded VQE args with qiskit-specific ones at the right positions
        for flag in ["--vqe-reps", "--vqe-restarts", "--vqe-seed", "--vqe-maxiter"]:
            idx = cmd.index(flag)
            if flag == "--vqe-reps":
                cmd[idx + 1] = str(args.qiskit_vqe_reps)
            elif flag == "--vqe-restarts":
                cmd[idx + 1] = str(args.qiskit_vqe_restarts)
            elif flag == "--vqe-seed":
                cmd[idx + 1] = str(args.qiskit_vqe_seed)
            elif flag == "--vqe-maxiter":
                cmd[idx + 1] = str(args.qiskit_vqe_maxiter)
        method_idx = cmd.index("--vqe-method")
        cmd[method_idx] = "--vqe-optimizer"
        cmd[method_idx + 1] = str(args.qiskit_vqe_optimizer)
        return cmd

    results: dict[str, Any] = {}
    for label, drive_tokens in [
        ("disabled", []),
        ("A0", _build_drive_args_with_amplitude(args, A0)),
        ("A1", _build_drive_args_with_amplitude(args, A1)),
    ]:
        slug = label.replace(".", "p")
        hc_json = json_dir / f"amp_H_{tag}_{slug}.json"
        qk_json = json_dir / f"amp_Q_{tag}_{slug}.json"

        hc_cmd = _base_cmd(
            "hardcoded_hubbard_pipeline.py",
            hc_json,
            include_hardcoded_ansatz=True,
        ) + drive_tokens
        _ai_log("amp_cmp_run_start", L=int(L), label=label, pipeline="hardcoded")
        code, out, err = _run_command(hc_cmd)
        if code != 0:
            raise RuntimeError(
                f"Amplitude-comparison hardcoded pipeline failed L={L} label={label}.\n"
                f"STDOUT:\n{out}\nSTDERR:\n{err}"
            )

        qk_cmd = _base_cmd_qk(qk_json) + drive_tokens
        _ai_log("amp_cmp_run_start", L=int(L), label=label, pipeline="qiskit")
        code, out, err = _run_command(qk_cmd)
        if code != 0:
            raise RuntimeError(
                f"Amplitude-comparison qiskit pipeline failed L={L} label={label}.\n"
                f"STDOUT:\n{out}\nSTDERR:\n{err}"
            )

        hc_payload = json.loads(hc_json.read_text(encoding="utf-8"))
        qk_payload = json.loads(qk_json.read_text(encoding="utf-8"))
        results[label] = {"hc": hc_payload, "qk": qk_payload}
        _ai_log("amp_cmp_run_done", L=int(L), label=label)

    return results


def _write_amplitude_comparison_pdf(
    pdf_path: Path,
    L: int,
    amp_data: dict[str, Any],
    A0: float,
    A1: float,
    args: argparse.Namespace,
    run_command: str,
    t_hartree: float | None = None,
) -> dict[str, Any]:
    """Write the amplitude-comparison PDF for a single *L* value."""
    disabled_hc = amp_data["disabled"]["hc"]
    disabled_qk = amp_data["disabled"]["qk"]
    a0_hc = amp_data["A0"]["hc"]
    a0_qk = amp_data["A0"]["qk"]
    a1_hc = amp_data["A1"]["hc"]
    a1_qk = amp_data["A1"]["qk"]

    safe = _safe_test_check(disabled_hc, disabled_qk, a0_hc, a0_qk)

    def _vqe_energy(payload: dict[str, Any]) -> float:
        val = payload.get("vqe", {}).get("energy")
        if val is None:
            val = payload.get("ground_state", {}).get("vqe_energy")
        return float(val) if val is not None else float("nan")

    hc_vqe_a0 = _vqe_energy(a0_hc)
    qk_vqe_a0 = _vqe_energy(a0_qk)
    hc_vqe_a1 = _vqe_energy(a1_hc)
    qk_vqe_a1 = _vqe_energy(a1_qk)

    exact_filtered_a0 = float("nan")
    try:
        exact_filtered_a0 = _require_exact_filtered_energy(a0_qk, L=L)
    except Exception:
        pass
    exact_filtered_a1 = float("nan")
    try:
        exact_filtered_a1 = _require_exact_filtered_energy(a1_qk, L=L)
    except Exception:
        pass

    delta_vqe_a0 = float(hc_vqe_a0 - qk_vqe_a0) if (np.isfinite(hc_vqe_a0) and np.isfinite(qk_vqe_a0)) else float("nan")
    delta_vqe_a1 = float(hc_vqe_a1 - qk_vqe_a1) if (np.isfinite(hc_vqe_a1) and np.isfinite(qk_vqe_a1)) else float("nan")

    nd_hc_rows = disabled_hc.get("trajectory", [])
    nd_qk_rows = disabled_qk.get("trajectory", [])
    a0_hc_rows = a0_hc.get("trajectory", [])
    a0_qk_rows = a0_qk.get("trajectory", [])
    a1_hc_rows = a1_hc.get("trajectory", [])
    a1_qk_rows = a1_qk.get("trajectory", [])

    times = _arr_optional(a1_hc_rows, "time")
    if times.size == 0:
        times = _arr_optional(a0_hc_rows, "time")
    if times.size == 0:
        times = _arr_optional(a1_qk_rows, "time")
    if times.size == 0:
        times = _arr_optional(a0_qk_rows, "time")
    if times.size == 0:
        times = np.linspace(0.0, float(args.t_final), int(args.num_times), dtype=float)

    drive_cfg = _drive_config_from_args(args)
    drive_a0 = evaluate_drive_waveform(times, drive_cfg, float(A0))
    drive_a1 = evaluate_drive_waveform(times, drive_cfg, float(A1))
    show_staggered = bool(str(args.drive_pattern).strip().lower() == "staggered")

    overlay_observables: list[tuple[str, str]] = [
        ("Subspace Fidelity", "fidelity"),
        ("Energy E0(t)=<H0>", "energy"),
        ("Site-0 Density <n0>", "density_site0"),
        ("Double Occupancy / site", "doublon"),
    ]
    if show_staggered:
        overlay_observables.insert(2, ("Staggered Imbalance O_stag", "staggered"))

    table_a_rows: list[list[str]] = []
    for label, kind in overlay_observables:
        a0_max, a0_rms = _max_abs_and_rms(
            _observables_series(a0_hc_rows, kind, "trotter"),
            _observables_series(a0_qk_rows, kind, "trotter"),
        )
        a1_max, a1_rms = _max_abs_and_rms(
            _observables_series(a1_hc_rows, kind, "trotter"),
            _observables_series(a1_qk_rows, kind, "trotter"),
        )
        table_a_rows.append(
            [
                label,
                _fmt_metric(a0_max),
                _fmt_metric(a0_rms),
                _fmt_metric(a1_max),
                _fmt_metric(a1_rms),
            ]
        )
    table_a_rows.append(
        [
            "Safe-test (no-drive vs A0)",
            (
                f"HC |ΔF_sub|={_fmt_metric(float(safe['hc']['max_fidelity_delta']))}, "
                f"|ΔE|={_fmt_metric(float(safe['hc']['max_energy_delta']))}\n"
                f"QK |ΔF_sub|={_fmt_metric(float(safe['qk']['max_fidelity_delta']))}, "
                f"|ΔE|={_fmt_metric(float(safe['qk']['max_energy_delta']))}"
            ),
            "-",
            "-",
            "-",
        ]
    )

    response_rows_cfg: list[tuple[str, str]] = []
    if show_staggered:
        response_rows_cfg.append(("O_stag", "staggered"))
    response_rows_cfg.extend(
        [
            ("n_site0", "density_site0"),
            ("doublon/site", "doublon"),
            ("Energy E0", "energy"),
        ]
    )

    table_b_rows: list[list[str]] = []
    for label, kind in response_rows_cfg:
        hc_max, hc_final = _response_stats(
            _observables_series(a0_hc_rows, kind, "trotter"),
            _observables_series(a1_hc_rows, kind, "trotter"),
        )
        exact_a0 = _observables_series(a0_hc_rows, kind, "exact")
        exact_a1 = _observables_series(a1_hc_rows, kind, "exact")
        if exact_a0.size == 0 or exact_a1.size == 0:
            exact_a0 = _observables_series(a0_qk_rows, kind, "exact")
            exact_a1 = _observables_series(a1_qk_rows, kind, "exact")
        ex_max, ex_final = _response_stats(exact_a0, exact_a1)
        table_b_rows.append(
            [
                label,
                _fmt_metric(hc_max),
                _fmt_metric(hc_final),
                _fmt_metric(ex_max),
                _fmt_metric(ex_final),
            ]
        )

    show_safe_page, max_safe = _safe_test_plot_gate(
        safe,
        near_factor=float(getattr(args, "safe_test_near_threshold_factor", 100.0)),
        verbose=bool(getattr(args, "report_verbose", False)),
    )

    metrics: dict[str, Any] = {
        "safe_test": safe,
        "delta_vqe_hc_minus_qk_at_A0": delta_vqe_a0,
        "delta_vqe_hc_minus_qk_at_A1": delta_vqe_a1,
        "safe_test_detail_page_rendered": bool(show_safe_page),
        "safe_test_max_abs": float(max_safe) if np.isfinite(max_safe) else float("nan"),
    }
    hardcoded_ansatz = _hardcoded_ansatz_label(a1_hc)
    qiskit_ansatz = _qiskit_ansatz_label(a1_qk)
    _ = run_command  # kept for signature compatibility; command text is logged to commands.txt

    with PdfPages(str(pdf_path)) as pdf:
        # --- Page 1: summary ---
        summary_lines = [
            f"Amplitude Comparison Summary (L={L})",
            "",
            "Ansatz Used:",
            f"  - Hardcoded: {hardcoded_ansatz}",
            f"  - Qiskit baseline: {qiskit_ansatz}",
            "",
            "Amplitude Cases:",
            f"  - A0 (safe-test): {A0}",
            f"  - A1 (active): {A1}",
            "",
            "Topline:",
            f"  - safe_test: {'PASS' if safe['passed'] else 'FAIL'}",
            f"  - max_safe_delta: {_fmt_metric(max_safe)}",
            f"  - delta_vqe_hc_minus_qk_at_A0: {_fmt_metric(delta_vqe_a0)}",
            f"  - delta_vqe_hc_minus_qk_at_A1: {_fmt_metric(delta_vqe_a1)}",
            "",
            "Reference:",
            "  - Full executed commands are recorded in artifacts/commands.txt",
        ]
        _render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.029, max_line_width=112)

        # --- Page 2: settings / knobs ---
        settings_lines = [
            f"Amplitude Comparison Settings  (L={L})",
            "",
            f"  A0 (trivial / safe-test amplitude) : {A0}",
            f"  A1 (active amplitude)               : {A1}",
            "",
            "  Drive knobs forwarded to sub-pipelines:",
            f"    drive_pattern          : {args.drive_pattern}",
            f"    drive_omega            : {args.drive_omega}",
            f"    drive_tbar             : {args.drive_tbar}",
            f"    drive_phi              : {args.drive_phi}",
            f"    drive_include_identity : {bool(args.drive_include_identity)}",
            f"    drive_time_sampling    : {args.drive_time_sampling}",
            f"    drive_t0               : {args.drive_t0}",
            f"    exact_steps_multiplier : {args.exact_steps_multiplier}",
            "",
            "  Common knobs:",
            f"    ordering               : {args.ordering}",
            f"    t_final                : {args.t_final}",
            f"    num_times              : {args.num_times}",
            f"    trotter_steps          : {args.trotter_steps}",
            f"    suzuki_order           : {args.suzuki_order}",
            f"    initial_state_source   : {args.initial_state_source}",
            "",
            "  Safe-test threshold : {:.2e}".format(_SAFE_TEST_THRESHOLD),
            "",
            "  Three runs per pipeline per L:",
            "    1. drive DISABLED   (baseline)",
            f"   2. --drive-A {A0}  (trivial amplitude; safe-test reference)",
            f"   3. --drive-A {A1}  (active amplitude)",
        ]
        _render_text_page(pdf, settings_lines)

        # --- Page 3: scoreboard + physics headline ---
        fig_score = plt.figure(figsize=(11.0, 8.5))
        gs = fig_score.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.0, 1.0])
        ax_a = fig_score.add_subplot(gs[0, 0])
        ax_b = fig_score.add_subplot(gs[0, 1])
        ax_drive = fig_score.add_subplot(gs[1, :])

        _render_compact_table(
            ax_a,
            title="A) Cross-Implementation Agreement (HC vs QK)",
            col_labels=["Observable", "A0 max|Δ|", "A0 RMS", "A1 max|Δ|", "A1 RMS"],
            rows=table_a_rows,
            fontsize=7,
        )
        _render_compact_table(
            ax_b,
            title="B) Physics Response Magnitude (A1 - A0)",
            col_labels=["Observable", "HC max|Δ|", "HC Δ(final)", "Exact max|Δ|", "Exact Δ(final)"],
            rows=table_b_rows,
            fontsize=7,
        )

        ax_drive.plot(times[: drive_a0.size], drive_a0, color="#1f77b4", linewidth=1.4, label=f"f(t), A0={A0}")
        ax_drive.plot(times[: drive_a1.size], drive_a1, color="#d62728", linewidth=1.4, label=f"f(t), A1={A1}")
        ax_drive.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.8)
        ax_drive.grid(alpha=0.25)
        ax_drive.set_xlabel("Time", fontsize=9)
        ax_drive.set_ylabel("f(t)", fontsize=9)
        ax_drive.legend(fontsize=8, loc="best")
        ax_drive.set_title(
            "Drive waveform used in H(t): "
            f"pattern={args.drive_pattern}, omega={args.drive_omega}, tbar={args.drive_tbar}, "
            f"phi={args.drive_phi}, t0={args.drive_t0}, sampling={args.drive_time_sampling}",
            fontsize=9,
        )
        fig_score.suptitle(
            f"Amplitude Scoreboard + Physics Headline  L={L}\n"
            f"Safe-test: {'PASSED' if safe['passed'] else 'FAILED'}   "
            f"(max safe Δ={_fmt_metric(max_safe)})",
            fontsize=11,
        )
        fig_score.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        pdf.savefig(fig_score)
        plt.close(fig_score)

        # --- Conditional safe-test detail page ---
        if show_safe_page:
            fig_st, axes_st = plt.subplots(2, 2, figsize=(11.0, 8.5))
            nd_times = _arr_optional(nd_hc_rows, "time")
            za_times = _arr_optional(a0_hc_rows, "time")
            t_ref = nd_times if nd_times.size > 0 else za_times
            hc_fid_delta = _safe_delta_series(nd_hc_rows, a0_hc_rows, "fidelity")
            hc_ene_delta = _safe_delta_series(nd_hc_rows, a0_hc_rows, "energy_static_trotter")
            qk_fid_delta = _safe_delta_series(nd_qk_rows, a0_qk_rows, "fidelity")
            qk_ene_delta = _safe_delta_series(nd_qk_rows, a0_qk_rows, "energy_static_trotter")

            for ax, delta, title in [
                (axes_st[0, 0], hc_fid_delta, f"HC |ΔSubspace Fidelity| (no-drive vs A0={A0})"),
                (axes_st[0, 1], qk_fid_delta, f"QK |ΔSubspace Fidelity| (no-drive vs A0={A0})"),
                (axes_st[1, 0], hc_ene_delta, f"HC |ΔE_trot| (no-drive vs A0={A0})"),
                (axes_st[1, 1], qk_ene_delta, f"QK |ΔE_trot| (no-drive vs A0={A0})"),
            ]:
                if delta.size > 0 and t_ref.size > 0:
                    ax.semilogy(t_ref[: delta.size], delta + 1e-20, color="#1f77b4")
                    ax.axhline(
                        _SAFE_TEST_THRESHOLD,
                        color="red",
                        linestyle="--",
                        linewidth=0.8,
                        label=f"threshold {_SAFE_TEST_THRESHOLD:.0e}",
                    )
                    ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("Time", fontsize=8)
                ax.grid(alpha=0.25)
            reason = (
                "FAIL / near-threshold / verbose triggered"
                if (not safe["passed"]) or bool(getattr(args, "report_verbose", False))
                else "near-threshold triggered"
            )
            fig_st.suptitle(
                f"Safe-test Detail L={L}: {'PASSED' if safe['passed'] else 'FAILED'}  "
                f"(reason: {reason})\n"
                f"HC max|ΔF_sub|={safe['hc']['max_fidelity_delta']:.2e}  "
                f"HC max|ΔE|={safe['hc']['max_energy_delta']:.2e}  "
                f"QK max|ΔF_sub|={safe['qk']['max_fidelity_delta']:.2e}  "
                f"QK max|ΔE|={safe['qk']['max_energy_delta']:.2e}",
                fontsize=10,
            )
            fig_st.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
            pdf.savefig(fig_st)
            plt.close(fig_st)

        # --- VQE residual table (perceptible deltas) ---
        vqe_rows = [
            [
                f"A0={A0}",
                _fmt_metric(exact_filtered_a0),
                _fmt_metric(hc_vqe_a0),
                _fmt_metric(qk_vqe_a0),
                _fmt_metric(delta_vqe_a0),
                _fmt_metric(hc_vqe_a0 - exact_filtered_a0),
                _fmt_metric(qk_vqe_a0 - exact_filtered_a0),
            ],
            [
                f"A1={A1}",
                _fmt_metric(exact_filtered_a1),
                _fmt_metric(hc_vqe_a1),
                _fmt_metric(qk_vqe_a1),
                _fmt_metric(delta_vqe_a1),
                _fmt_metric(hc_vqe_a1 - exact_filtered_a1),
                _fmt_metric(qk_vqe_a1 - exact_filtered_a1),
            ],
        ]
        vqe_identical_tol = 1e-12
        vqe_identical = bool(
            np.isfinite(hc_vqe_a0)
            and np.isfinite(hc_vqe_a1)
            and np.isfinite(qk_vqe_a0)
            and np.isfinite(qk_vqe_a1)
            and np.isfinite(exact_filtered_a0)
            and np.isfinite(exact_filtered_a1)
            and abs(hc_vqe_a1 - hc_vqe_a0) <= vqe_identical_tol
            and abs(qk_vqe_a1 - qk_vqe_a0) <= vqe_identical_tol
            and abs(exact_filtered_a1 - exact_filtered_a0) <= vqe_identical_tol
        )
        fig_vqe = plt.figure(figsize=(11.0, 8.5))
        ax_vqe = fig_vqe.add_subplot(111)
        _render_compact_table(
            ax_vqe,
            title="VQE Energy Comparison (Residual-focused)",
            col_labels=[
                "Amplitude",
                "Exact(filtered)",
                "HC",
                "QK",
                "HC-QK",
                "HC-Exact",
                "QK-Exact",
            ],
            rows=vqe_rows,
            fontsize=9,
        )
        vqe_note = "A0 and A1 identical within tol." if vqe_identical else "A0 and A1 differ."
        chem_lines = _chemical_accuracy_lines(t_hartree)
        ax_vqe.text(
            0.02,
            0.06,
            " | ".join([vqe_note, *[line.strip() for line in chem_lines if line.strip()]]),
            transform=ax_vqe.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        fig_vqe.suptitle(f"VQE Energy Comparison L={L}", fontsize=11)
        fig_vqe.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        pdf.savefig(fig_vqe)
        plt.close(fig_vqe)

        # --- Combined trajectory overlay (HC + Exact in one page) ---
        fig_ov, axes_ov = plt.subplots(2, 2, figsize=(11.0, 8.5))
        ax_list = axes_ov.ravel()

        def _plot_combo(ax: Any, kind: str, *, title: str, ylabel: str, include_exact: bool) -> None:
            t_a0_h = _arr_optional(a0_hc_rows, "time")
            t_a1_h = _arr_optional(a1_hc_rows, "time")
            t_a0_q = _arr_optional(a0_qk_rows, "time")
            t_a1_q = _arr_optional(a1_qk_rows, "time")
            y_a0_h = _observables_series(a0_hc_rows, kind, "trotter")
            y_a1_h = _observables_series(a1_hc_rows, kind, "trotter")

            has_data = False
            if y_a0_h.size > 0 and t_a0_h.size > 0:
                ax.plot(t_a0_h[: y_a0_h.size], y_a0_h, color="#1f77b4", linestyle="-", linewidth=1.2, label=f"HC A0={A0}")
                has_data = True
            if y_a1_h.size > 0 and t_a1_h.size > 0:
                ax.plot(t_a1_h[: y_a1_h.size], y_a1_h, color="#d62728", linestyle="-", linewidth=1.2, label=f"HC A1={A1}")
                has_data = True

            if include_exact:
                y_a0_ex = _observables_series(a0_hc_rows, kind, "exact")
                y_a1_ex = _observables_series(a1_hc_rows, kind, "exact")
                t_a0_ex = t_a0_h
                t_a1_ex = t_a1_h
                if y_a0_ex.size == 0:
                    y_a0_ex = _observables_series(a0_qk_rows, kind, "exact")
                    t_a0_ex = t_a0_q
                if y_a1_ex.size == 0:
                    y_a1_ex = _observables_series(a1_qk_rows, kind, "exact")
                    t_a1_ex = t_a1_q
                if y_a0_ex.size > 0 and t_a0_ex.size > 0:
                    ax.plot(t_a0_ex[: y_a0_ex.size], y_a0_ex, color="#1f77b4", linestyle="--", linewidth=1.0, label=f"Exact A0={A0}")
                    has_data = True
                if y_a1_ex.size > 0 and t_a1_ex.size > 0:
                    ax.plot(t_a1_ex[: y_a1_ex.size], y_a1_ex, color="#d62728", linestyle="--", linewidth=1.0, label=f"Exact A1={A1}")
                    has_data = True

            if not has_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=0.25)
            if kind == "fidelity":
                _set_fidelity_ylim(ax, y_a0_h, y_a1_h)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6, loc="best")

        _plot_combo(ax_list[0], "fidelity", title="Subspace Fidelity", ylabel="Subspace Fidelity", include_exact=False)
        _plot_combo(ax_list[1], "energy", title="Energy (E0(t)=<H0>)", ylabel="E0(t)", include_exact=True)
        matched_kind = "staggered" if show_staggered else "density_site0"
        matched_title = "Staggered Imbalance O_stag" if show_staggered else "Site-0 Density <n0>"
        matched_ylabel = "O_stag" if show_staggered else "<n0>"
        _plot_combo(ax_list[2], matched_kind, title=matched_title, ylabel=matched_ylabel, include_exact=True)
        _plot_combo(ax_list[3], "doublon", title="Double Occupancy / site", ylabel="doublon/site", include_exact=True)
        fig_ov.suptitle(
            f"Trajectory Overlay L={L}\n"
            "Color = amplitude (blue A0, red A1); style = HC solid, Exact dashed.",
            fontsize=11,
        )
        fig_ov.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        pdf.savefig(fig_ov)
        plt.close(fig_ov)

        # --- Drive-induced response page ---
        fig_resp, axes_resp = plt.subplots(2, 2, figsize=(11.0, 8.5))
        ax_r = axes_resp.ravel()
        t_ref = times
        delta_kind = matched_kind
        d_hc_match = _response_delta(
            _observables_series(a0_hc_rows, delta_kind, "trotter"),
            _observables_series(a1_hc_rows, delta_kind, "trotter"),
        )
        d_hc_dbl = _response_delta(
            _observables_series(a0_hc_rows, "doublon", "trotter"),
            _observables_series(a1_hc_rows, "doublon", "trotter"),
        )
        d_hc_e = _response_delta(
            _observables_series(a0_hc_rows, "energy", "trotter"),
            _observables_series(a1_hc_rows, "energy", "trotter"),
        )
        d_ex_match = _response_delta(
            _observables_series(a0_hc_rows, delta_kind, "exact"),
            _observables_series(a1_hc_rows, delta_kind, "exact"),
        )
        if d_ex_match.size == 0:
            d_ex_match = _response_delta(
                _observables_series(a0_qk_rows, delta_kind, "exact"),
                _observables_series(a1_qk_rows, delta_kind, "exact"),
            )
        d_ex_dbl = _response_delta(
            _observables_series(a0_hc_rows, "doublon", "exact"),
            _observables_series(a1_hc_rows, "doublon", "exact"),
        )
        if d_ex_dbl.size == 0:
            d_ex_dbl = _response_delta(
                _observables_series(a0_qk_rows, "doublon", "exact"),
                _observables_series(a1_qk_rows, "doublon", "exact"),
            )
        d_ex_e = _response_delta(
            _observables_series(a0_hc_rows, "energy", "exact"),
            _observables_series(a1_hc_rows, "energy", "exact"),
        )
        if d_ex_e.size == 0:
            d_ex_e = _response_delta(
                _observables_series(a0_qk_rows, "energy", "exact"),
                _observables_series(a1_qk_rows, "energy", "exact"),
            )

        ax_r[0].plot(t_ref[: drive_a0.size], drive_a0, color="#1f77b4", linewidth=1.2, label=f"A0={A0}")
        ax_r[0].plot(t_ref[: drive_a1.size], drive_a1, color="#d62728", linewidth=1.2, label=f"A1={A1}")
        ax_r[0].axhline(0.0, color="#555555", linewidth=0.8, alpha=0.8)
        ax_r[0].set_title("Drive waveform f(t)", fontsize=9)
        ax_r[0].set_xlabel("Time", fontsize=8)
        ax_r[0].set_ylabel("f(t)", fontsize=8)
        ax_r[0].grid(alpha=0.25)
        ax_r[0].legend(fontsize=7)

        for ax, d_hc, d_ex, title, ylabel in [
            (
                ax_r[1],
                d_hc_match,
                d_ex_match,
                ("ΔO_stag(t)" if show_staggered else "Δn_site0(t)"),
                ("ΔO_stag" if show_staggered else "Δn_site0"),
            ),
            (ax_r[2], d_hc_dbl, d_ex_dbl, "ΔDoubleOcc(t)", "Δdoublon/site"),
            (ax_r[3], d_hc_e, d_ex_e, "ΔEnergy(t)", "ΔE0(t)"),
        ]:
            if d_hc.size > 0:
                ax.plot(t_ref[: d_hc.size], d_hc, color="#d62728", linewidth=1.2, label="HC Δ")
            if d_ex.size > 0:
                ax.plot(t_ref[: d_ex.size], d_ex, color="#111111", linestyle="--", linewidth=1.0, label="Exact Δ")
            if d_hc.size == 0 and d_ex.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc="best")

        fig_resp.suptitle(f"Drive-Induced Response L={L} (A1 - A0)", fontsize=11)
        fig_resp.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        pdf.savefig(fig_resp)
        plt.close(fig_resp)

        # --- Optional site-resolved heatmaps ---
        n_a1 = _matrix_optional(a1_hc_rows, "n_site_trotter")
        n_a0 = _matrix_optional(a0_hc_rows, "n_site_trotter")
        t_heat = _arr_optional(a1_hc_rows, "time")
        if n_a1.size > 0 and n_a1.shape[1] > 0 and t_heat.size > 0:
            n_rows = min(n_a0.shape[0], n_a1.shape[0])
            if n_rows > 0 and n_a0.shape[1] == n_a1.shape[1]:
                delta_mat = n_a1[:n_rows, :] - n_a0[:n_rows, :]
            else:
                delta_mat = np.zeros_like(n_a1)
            fig_hm, axes_hm = plt.subplots(1, 2, figsize=(11.0, 8.5))
            im0 = axes_hm[0].imshow(
                n_a1.T,
                aspect="auto",
                origin="lower",
                extent=[float(t_heat[0]), float(t_heat[min(len(t_heat) - 1, n_a1.shape[0] - 1)]), -0.5, n_a1.shape[1] - 0.5],
            )
            axes_hm[0].set_title(f"<n_i(t)> A1={A1}", fontsize=10)
            axes_hm[0].set_xlabel("Time")
            axes_hm[0].set_ylabel("Site index")
            fig_hm.colorbar(im0, ax=axes_hm[0], shrink=0.8)
            im1 = axes_hm[1].imshow(
                delta_mat.T,
                aspect="auto",
                origin="lower",
                extent=[float(t_heat[0]), float(t_heat[min(len(t_heat) - 1, delta_mat.shape[0] - 1)]), -0.5, delta_mat.shape[1] - 0.5],
            )
            axes_hm[1].set_title("Δn_i(t) = n_i,A1 - n_i,A0", fontsize=10)
            axes_hm[1].set_xlabel("Time")
            axes_hm[1].set_ylabel("Site index")
            fig_hm.colorbar(im1, ax=axes_hm[1], shrink=0.8)
            fig_hm.suptitle(f"Site-Resolved Density Heatmaps L={L}", fontsize=11)
            fig_hm.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            pdf.savefig(fig_hm)
            plt.close(fig_hm)

        # --- Optional response spectrum panel ---
        spec_signal = d_hc_match if d_hc_match.size > 0 else d_hc_dbl
        omega_grid, mag = _fft_response(spec_signal, t_ref)
        if omega_grid.size > 0 and mag.size > 0:
            fig_fft, ax_fft = plt.subplots(1, 1, figsize=(11.0, 8.5))
            ax_fft.plot(omega_grid, mag, color="#1f77b4", linewidth=1.3, label="|FFT(windowed response)|")
            ax_fft.axvline(float(args.drive_omega), color="#d62728", linestyle="--", linewidth=1.0, label=f"drive ω={float(args.drive_omega):.3f}")
            ax_fft.set_xlabel("ω")
            ax_fft.set_ylabel("Magnitude")
            ax_fft.set_title(
                "Response Spectrum "
                + ("(ΔO_stag)" if d_hc_match.size > 0 else "(Δdoublon/site)"),
                fontsize=10,
            )
            ax_fft.grid(alpha=0.25)
            ax_fft.legend(fontsize=8)
            fig_fft.suptitle(f"Drive-Induced Response Spectrum L={L}", fontsize=11)
            fig_fft.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            pdf.savefig(fig_fft)
            plt.close(fig_fft)

    return metrics


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
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        default=False,
        help="Skip all compare-generated PDFs (bundle/per-L/amplitude). JSON artifacts are still written.",
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
    parser.add_argument(
        "--fidelity-subspace-energy-tol",
        type=float,
        default=1e-8,
        help=(
            "Energy tolerance for filtered-sector ground-manifold selection used by "
            "trajectory subspace fidelity (E <= E0 + tol). Forwarded to both pipelines."
        ),
    )

    parser.add_argument("--hardcoded-vqe-reps", type=int, default=2)
    parser.add_argument("--hardcoded-vqe-restarts", type=int, default=3)
    parser.add_argument("--hardcoded-vqe-seed", type=int, default=7)
    parser.add_argument("--hardcoded-vqe-maxiter", type=int, default=600)
    parser.add_argument(
        "--hardcoded-vqe-method",
        type=str,
        default="COBYLA",
        choices=["SLSQP", "COBYLA", "L-BFGS-B", "Powell", "Nelder-Mead"],
    )
    parser.add_argument(
        "--hardcoded-vqe-ansatzes",
        type=str,
        default="uccsd",
        metavar="A0,A1,...",
        help=(
            "Comma-separated hardcoded ansatz list to benchmark against one shared qiskit baseline. "
            "Allowed values: uccsd,hva."
        ),
    )

    parser.add_argument("--qiskit-vqe-reps", type=int, default=2)
    parser.add_argument("--qiskit-vqe-restarts", type=int, default=3)
    parser.add_argument("--qiskit-vqe-seed", type=int, default=7)
    parser.add_argument("--qiskit-vqe-maxiter", type=int, default=600)
    parser.add_argument(
        "--qiskit-vqe-optimizer",
        type=str,
        default="COBYLA",
        choices=["SLSQP", "COBYLA", "L_BFGS_B"],
    )

    parser.add_argument("--qpe-eval-qubits", type=int, default=5)
    parser.add_argument("--qpe-shots", type=int, default=256)
    parser.add_argument("--qpe-seed", type=int, default=11)
    parser.add_argument("--skip-qpe", action="store_true", help="Pass --skip-qpe to both pipeline runners.")

    # ------------------------------------------------------------------
    # Drive passthrough (forwarded verbatim to both sub-pipelines).
    # Defaults mirror pipelines/hardcoded_hubbard_pipeline.py parse_args().
    # When --enable-drive is absent, no drive flags are forwarded and
    # behaviour is bit-for-bit identical to before this feature was added.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--enable-drive",
        action="store_true",
        default=False,
        help="Enable the time-dependent onsite density drive in both pipelines.",
    )
    parser.add_argument("--drive-A", type=float, default=1.0,
                        help="Drive amplitude A (Gaussian-sinusoid waveform).")
    parser.add_argument("--drive-omega", type=float, default=1.0,
                        help="Drive angular frequency ω.")
    parser.add_argument("--drive-tbar", type=float, default=5.0,
                        help="Drive Gaussian half-width t̄.")
    parser.add_argument("--drive-phi", type=float, default=0.0,
                        help="Drive phase offset φ.")
    parser.add_argument(
        "--drive-pattern",
        type=str,
        default="staggered",
        choices=["staggered", "dimer_bias", "custom"],
        help="Spatial weight pattern for the drive.",
    )
    parser.add_argument(
        "--drive-custom-s",
        type=str,
        default=None,
        metavar="JSON_ARRAY",
        help="JSON-encoded list of custom spatial weights, e.g. '[1.0,-0.5]'. "
             "Required when --drive-pattern=custom.",
    )
    parser.add_argument(
        "--drive-include-identity",
        action="store_true",
        default=False,
        help="Include the identity (global-phase) term in the drive Hamiltonian.",
    )
    parser.add_argument(
        "--drive-time-sampling",
        type=str,
        default="midpoint",
        choices=["midpoint", "left", "right"],
        help="Time-sampling rule for drive coefficients within each Trotter slice.",
    )
    parser.add_argument("--drive-t0", type=float, default=0.0,
                        help="Drive start time t₀ (offset added to each Trotter slice time).")
    parser.add_argument(
        "--exact-steps-multiplier",
        type=int,
        default=1,
        help=(
            "Reference-propagator refinement factor forwarded to both pipelines. "
            "When drive is enabled, the reference runs at "
            "N_ref = exact_steps_multiplier * trotter_steps steps. "
            "Default 1 (reference and Trotter use the same step count)."
        ),
    )

    parser.add_argument("--initial-state-source", choices=["exact", "vqe", "hf"], default="vqe")

    # ------------------------------------------------------------------
    # Drive amplitude comparison (additive — does not affect existing runs)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--drive-amplitudes",
        type=str,
        default="0.0,0.2",
        metavar="A0,A1",
        help=(
            "Comma-separated pair of drive amplitudes used by "
            "--with-drive-amplitude-comparison-pdf.  "
            "A0 is the trivial (zero) amplitude for the safe-test; "
            "A1 is the active amplitude.  Default: '0.0,0.2'."
        ),
    )
    parser.add_argument(
        "--with-drive-amplitude-comparison-pdf",
        action="store_true",
        default=False,
        help=(
            "Generate an amplitude-comparison PDF for each L value.  "
            "Runs both pipelines three times per L: drive disabled, "
            "--drive-A A0, and --drive-A A1.  "
            "Includes a safe-test page (no-drive vs A0=0 must be identical) "
            "and a VQE delta page (ΔE = HC_VQE − QK_VQE at each amplitude).  "
            "Requires --enable-drive (or any drive flags) to be meaningful."
        ),
    )
    parser.add_argument(
        "--report-verbose",
        action="store_true",
        default=False,
        help=(
            "Emit verbose report pages. For amplitude PDFs this forces the "
            "full safe-test detail page even when far from threshold."
        ),
    )
    parser.add_argument(
        "--safe-test-near-threshold-factor",
        type=float,
        default=100.0,
        help=(
            "Near-threshold gate for rendering full safe-test plots. "
            "The detail page appears when max_safe_delta >= threshold/factor, "
            "or on failure, or with --report-verbose."
        ),
    )

    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--t-hartree",
        type=float,
        default=None,
        metavar="HARTREE",
        help=(
            "Physical value of the hopping parameter t in Hartree.  "
            "Used to express chemical accuracy (1.6e-3 Ha) in the "
            "model's natural energy units on every PDF text page.  "
            "Example: for cuprate t≈0.3 eV, pass --t-hartree 0.01102."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("compare_main_start", settings=vars(args))
    run_command = _current_command_string()
    # Resolve to absolute so that paths are consistent between the main
    # process (CWD may differ) and subprocesses (cwd=ROOT).
    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_dir = artifacts_dir / "json"
    pdf_dir = artifacts_dir / "pdf"
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    l_values = [int(x.strip()) for x in str(args.l_values).split(",") if x.strip()]
    _ai_log("compare_l_values", l_values=l_values)

    command_log_path = artifacts_dir / "commands.txt"
    command_log: list[str] = []
    hardcoded_ansatzes = _parse_hardcoded_vqe_ansatzes(str(args.hardcoded_vqe_ansatzes))
    primary_hardcoded_ansatz = "uccsd" if "uccsd" in hardcoded_ansatzes else hardcoded_ansatzes[0]
    ordered_hardcoded_ansatzes = [primary_hardcoded_ansatz] + [
        name for name in hardcoded_ansatzes if name != primary_hardcoded_ansatz
    ]
    _ai_log(
        "compare_hardcoded_ansatzes",
        ansatzes=ordered_hardcoded_ansatzes,
        primary=primary_hardcoded_ansatz,
    )

    run_artifacts: list[RunArtifacts] = []
    for L in l_values:
        _ai_log("compare_l_start", L=int(L))
        tag = _artifact_tag(args, L)
        qk_json = json_dir / f"Q_{tag}.json"
        qk_pdf = pdf_dir / f"Q_{tag}.pdf"
        metrics_json = json_dir / f"HvQ_{tag}_metrics.json"
        hc_json_by_ansatz: dict[str, Path] = {}
        hc_pdf_by_ansatz: dict[str, Path] = {}
        compare_pdf_by_ansatz: dict[str, Path] = {}
        for ansatz_name in ordered_hardcoded_ansatzes:
            if ansatz_name == primary_hardcoded_ansatz:
                hc_json = json_dir / f"H_{tag}.json"
                hc_pdf = pdf_dir / f"H_{tag}.pdf"
                compare_pdf = pdf_dir / f"HvQ_{tag}.pdf"
            else:
                hc_json = json_dir / f"H_{ansatz_name}_{tag}.json"
                hc_pdf = pdf_dir / f"H_{ansatz_name}_{tag}.pdf"
                compare_pdf = pdf_dir / f"HvQ_{ansatz_name}_{tag}.pdf"
            hc_json_by_ansatz[ansatz_name] = hc_json
            hc_pdf_by_ansatz[ansatz_name] = hc_pdf
            compare_pdf_by_ansatz[ansatz_name] = compare_pdf

        if args.run_pipelines:
            for ansatz_name in ordered_hardcoded_ansatzes:
                hc_json = hc_json_by_ansatz[ansatz_name]
                hc_pdf = hc_pdf_by_ansatz[ansatz_name]
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
                    "--fidelity-subspace-energy-tol", str(args.fidelity_subspace_energy_tol),
                    "--term-order", "sorted",
                    "--vqe-ansatz", str(ansatz_name),
                    "--vqe-reps", str(args.hardcoded_vqe_reps),
                    "--vqe-restarts", str(args.hardcoded_vqe_restarts),
                    "--vqe-seed", str(args.hardcoded_vqe_seed),
                    "--vqe-maxiter", str(args.hardcoded_vqe_maxiter),
                    "--vqe-method", str(args.hardcoded_vqe_method),
                    "--qpe-eval-qubits", str(args.qpe_eval_qubits),
                    "--qpe-shots", str(args.qpe_shots),
                    "--qpe-seed", str(args.qpe_seed),
                    "--initial-state-source", str(args.initial_state_source),
                    "--output-json", str(hc_json),
                    "--output-pdf", str(hc_pdf),
                    "--skip-pdf",
                ]
                if args.skip_qpe:
                    hc_cmd.append("--skip-qpe")
                hc_cmd.extend(_build_drive_args(args))
                command_log.append(" ".join(hc_cmd))
                code, out, err = _run_command(hc_cmd)
                if code != 0:
                    raise RuntimeError(
                        f"Hardcoded pipeline failed for L={L}, ansatz={ansatz_name}.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                    )
                _ai_log(
                    "compare_l_hardcoded_done",
                    L=int(L),
                    ansatz=str(ansatz_name),
                    json_path=str(hc_json),
                )

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
                "--fidelity-subspace-energy-tol", str(args.fidelity_subspace_energy_tol),
                "--term-order", "sorted",
                "--vqe-reps", str(args.qiskit_vqe_reps),
                "--vqe-restarts", str(args.qiskit_vqe_restarts),
                "--vqe-seed", str(args.qiskit_vqe_seed),
                "--vqe-maxiter", str(args.qiskit_vqe_maxiter),
                "--vqe-optimizer", str(args.qiskit_vqe_optimizer),
                "--qpe-eval-qubits", str(args.qpe_eval_qubits),
                "--qpe-shots", str(args.qpe_shots),
                "--qpe-seed", str(args.qpe_seed),
                "--initial-state-source", str(args.initial_state_source),
                "--output-json", str(qk_json),
                "--output-pdf", str(qk_pdf),
                "--skip-pdf",
            ]
            if args.skip_qpe:
                qk_cmd.append("--skip-qpe")
            qk_cmd.extend(_build_drive_args(args))
            command_log.append(" ".join(qk_cmd))
            code, out, err = _run_command(qk_cmd)
            if code != 0:
                raise RuntimeError(f"Qiskit pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
            _ai_log("compare_l_qiskit_done", L=int(L), json_path=str(qk_json))

        run_artifacts.append(
            RunArtifacts(
                L=L,
                hc_json_by_ansatz=hc_json_by_ansatz,
                hc_pdf_by_ansatz=hc_pdf_by_ansatz,
                qk_json=qk_json,
                qk_pdf=qk_pdf,
                metrics_json=metrics_json,
                compare_pdf_by_ansatz=compare_pdf_by_ansatz,
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
        if not row.qk_json.exists():
            raise FileNotFoundError(f"Missing qiskit JSON for L={row.L}: {row.qk_json}")

        qiskit = json.loads(row.qk_json.read_text(encoding="utf-8"))
        hardcoded_by_ansatz: dict[str, dict[str, Any]] = {}
        metrics_by_ansatz: dict[str, dict[str, Any]] = {}

        for ansatz_name in ordered_hardcoded_ansatzes:
            hc_json = row.hc_json_by_ansatz.get(ansatz_name)
            if hc_json is None or not hc_json.exists():
                raise FileNotFoundError(f"Missing hardcoded JSON for L={row.L}, ansatz={ansatz_name}: {hc_json}")
            hardcoded = json.loads(hc_json.read_text(encoding="utf-8"))
            metrics = _compare_payloads(hardcoded, qiskit)
            hardcoded_by_ansatz[ansatz_name] = hardcoded
            metrics_by_ansatz[ansatz_name] = metrics
            _ai_log(
                "compare_l_metrics",
                L=int(row.L),
                ansatz=str(ansatz_name),
                passed=bool(metrics["acceptance"]["pass"]),
                gs_abs_delta=float(metrics["ground_state_energy"]["abs_delta"]),
                fidelity_max_abs_delta=float(metrics["trajectory_deltas"]["fidelity"]["max_abs_delta"]),
                energy_static_trotter_max_abs_delta=float(metrics["trajectory_deltas"]["energy_static_trotter"]["max_abs_delta"]),
            )

        primary_metrics = metrics_by_ansatz[primary_hardcoded_ansatz]
        primary_hardcoded = hardcoded_by_ansatz[primary_hardcoded_ansatz]
        metrics_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "L": int(row.L),
            "primary_hardcoded_ansatz": str(primary_hardcoded_ansatz),
            "hc_json_by_ansatz": {
                ans: str(path) for ans, path in row.hc_json_by_ansatz.items()
            },
            "qk_json": str(row.qk_json),
            # Backward-compatible single-metrics view (primary ansatz).
            "metrics": primary_metrics,
            "metrics_by_ansatz": metrics_by_ansatz,
        }
        print(f"Delta metric definitions (L={row.L}):")
        print(_delta_metric_definition_text())
        row.metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        if args.with_per_l_pdfs and (not args.skip_pdf):
            for ansatz_name in ordered_hardcoded_ansatzes:
                _write_comparison_pdf(
                    pdf_path=row.compare_pdf_by_ansatz[ansatz_name],
                    L=row.L,
                    hardcoded=hardcoded_by_ansatz[ansatz_name],
                    qiskit=qiskit,
                    metrics=metrics_by_ansatz[ansatz_name],
                    run_command=run_command,
                    t_hartree=args.t_hartree,
                )

        per_l_data.append((row.L, primary_hardcoded, qiskit, primary_metrics))
        pass_by_ansatz = {
            ansatz_name: bool(metrics_by_ansatz[ansatz_name]["acceptance"]["pass"])
            for ansatz_name in ordered_hardcoded_ansatzes
        }
        results_rows.append(
            {
                "L": int(row.L),
                "pass": bool(all(pass_by_ansatz.values())),
                "pass_by_ansatz": pass_by_ansatz,
                "primary_hardcoded_ansatz": str(primary_hardcoded_ansatz),
                "metrics_json": str(row.metrics_json),
                "comparison_pdf_by_ansatz": {
                    ansatz_name: (
                        str(row.compare_pdf_by_ansatz[ansatz_name])
                        if (args.with_per_l_pdfs and (not args.skip_pdf))
                        else None
                    )
                    for ansatz_name in ordered_hardcoded_ansatzes
                },
                "hardcoded_settings_by_ansatz": {
                    ansatz_name: hardcoded_by_ansatz[ansatz_name].get("settings", {})
                    for ansatz_name in ordered_hardcoded_ansatzes
                },
                "qiskit_settings": qiskit.get("settings", {}),
                "ground_state_energy_abs_delta_by_ansatz": {
                    ansatz_name: float(metrics_by_ansatz[ansatz_name]["ground_state_energy"]["abs_delta"])
                    for ansatz_name in ordered_hardcoded_ansatzes
                },
                "trajectory_max_abs_deltas_by_ansatz": {
                    ansatz_name: {
                        key: float(metrics_by_ansatz[ansatz_name]["trajectory_deltas"][key]["max_abs_delta"])
                        for key in TARGET_METRICS
                    }
                    for ansatz_name in ordered_hardcoded_ansatzes
                },
                "vqe_sanity_by_ansatz": {
                    ansatz_name: metrics_by_ansatz[ansatz_name].get("vqe_sanity", {})
                    for ansatz_name in ordered_hardcoded_ansatzes
                },
            }
        )

    isolation_check = _check_hardcoded_qiskit_import_isolation(ROOT / "pipelines" / "hardcoded_hubbard_pipeline.py")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Hardcoded-first (HVA/UCCSD layer-wise) vs Qiskit Hubbard pipeline comparison.",
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
            "hardcoded_vqe_ansatzes": ordered_hardcoded_ansatzes,
            "primary_hardcoded_ansatz": str(primary_hardcoded_ansatz),
            "initial_state_source": str(args.initial_state_source),
            "skip_qpe": bool(args.skip_qpe),
            "skip_pdf": bool(args.skip_pdf),
            **(
                {
                    "drive": {
                        "enabled": True,
                        "A": float(args.drive_A),
                        "omega": float(args.drive_omega),
                        "tbar": float(args.drive_tbar),
                        "phi": float(args.drive_phi),
                        "pattern": str(args.drive_pattern),
                        "custom_s": (str(args.drive_custom_s) if args.drive_custom_s is not None else None),
                        "include_identity": bool(args.drive_include_identity),
                        "time_sampling": str(args.drive_time_sampling),
                        "t0": float(args.drive_t0),
                        "reference_steps_multiplier": int(args.exact_steps_multiplier),
                    }
                }
                if bool(args.enable_drive)
                else {}
            ),
        },
        "hardcoded_qiskit_import_isolation": isolation_check,
        "results": results_rows,
        "all_pass": bool(all(r["pass"] for r in results_rows) and isolation_check.get("pass", False)),
    }

    summary_json_path = json_dir / "HvQ_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bundle_pdf_path: Path | None = None
    if not args.skip_pdf:
        bundle_pdf_path = pdf_dir / "HvQ_bundle.pdf"
        _write_bundle_pdf(
            bundle_path=bundle_pdf_path,
            per_l_data=per_l_data,
            overall_summary=summary,
            isolation_check=isolation_check,
            include_per_l_pages=True,
            run_command=run_command,
            t_hartree=args.t_hartree,
        )
    _ai_log(
        "compare_main_done",
        summary_json=str(summary_json_path),
        bundle_pdf=(str(bundle_pdf_path) if bundle_pdf_path is not None else None),
        all_pass=bool(summary["all_pass"]),
        skip_pdf=bool(args.skip_pdf),
    )

    print(f"Wrote summary JSON: {summary_json_path}")
    if bundle_pdf_path is not None:
        print(f"Wrote bundle PDF:   {bundle_pdf_path}")
    else:
        print("Skipped all compare-generated PDFs (--skip-pdf).")
    if command_log:
        print(f"Wrote command log:  {command_log_path}")
    for row in results_rows:
        print(
            f"L={row['L']}: pass={row['pass']} pass_by_ansatz={row['pass_by_ansatz']} "
            f"metrics={row['metrics_json']} pdfs={row['comparison_pdf_by_ansatz']}"
        )

    # ------------------------------------------------------------------
    # Amplitude-comparison PDFs (additive, opt-in via CLI flag)
    # ------------------------------------------------------------------
    if bool(getattr(args, "with_drive_amplitude_comparison_pdf", False)):
        if bool(args.skip_pdf):
            _ai_log("amp_cmp_skipped_due_skip_pdf", l_values=l_values)
            print("Skipped amplitude-comparison PDFs because --skip-pdf was set.")
            return
        raw_amps = str(args.drive_amplitudes).split(",")
        if len(raw_amps) != 2:
            raise ValueError(
                f"--drive-amplitudes must be a comma-separated pair, got: {args.drive_amplitudes!r}"
            )
        A0 = float(raw_amps[0].strip())
        A1 = float(raw_amps[1].strip())
        _ai_log("amp_cmp_start", A0=A0, A1=A1, l_values=l_values)

        for L in l_values:
            _ai_log("amp_cmp_l_start", L=int(L), A0=A0, A1=A1)
            amp_data = _run_amplitude_comparison_for_l(
                args,
                L,
                A0,
                A1,
                json_dir,
                hardcoded_vqe_ansatz=str(primary_hardcoded_ansatz),
            )
            tag = _artifact_tag(args, L)
            amp_pdf_path = pdf_dir / f"amp_{tag}.pdf"
            amp_metrics = _write_amplitude_comparison_pdf(
                pdf_path=amp_pdf_path,
                L=L,
                amp_data=amp_data,
                A0=A0,
                A1=A1,
                args=args,
                run_command=run_command,
                t_hartree=args.t_hartree,
            )
            amp_metrics_path = json_dir / f"amp_{tag}_metrics.json"
            amp_metrics_path.write_text(
                json.dumps(
                    {
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "L": int(L),
                        "A0": A0,
                        "A1": A1,
                        "safe_test": amp_metrics["safe_test"],
                        "delta_vqe_hc_minus_qk_at_A0": amp_metrics["delta_vqe_hc_minus_qk_at_A0"],
                        "delta_vqe_hc_minus_qk_at_A1": amp_metrics["delta_vqe_hc_minus_qk_at_A1"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _ai_log(
                "amp_cmp_l_done",
                L=int(L),
                safe_test_passed=bool(amp_metrics["safe_test"]["passed"]),
                delta_vqe_A0=amp_metrics["delta_vqe_hc_minus_qk_at_A0"],
                delta_vqe_A1=amp_metrics["delta_vqe_hc_minus_qk_at_A1"],
                pdf=str(amp_pdf_path),
            )
            print(
                f"Amplitude comparison L={L}: "
                f"safe_test={'PASS' if amp_metrics['safe_test']['passed'] else 'FAIL'}  "
                f"ΔE(A0)={amp_metrics['delta_vqe_hc_minus_qk_at_A0']:.4e}  "
                f"ΔE(A1)={amp_metrics['delta_vqe_hc_minus_qk_at_A1']:.4e}  "
                f"pdf={amp_pdf_path}"
            )
        _ai_log("amp_cmp_done", l_values=l_values)


if __name__ == "__main__":
    main()
