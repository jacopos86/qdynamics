#!/usr/bin/env python3
"""Build a combined HVA heavy GS report for layerwise vs termwise runs.

This script reads heavy-run JSON artifacts and creates a multi-page PDF with:
1) run settings and artifact sources,
2) a summary table across requested L values,
3) per-L comparison pages.

If a termwise JSON is missing for an L (for example L=5 by request), termwise
cells are left blank in the summary table and termwise bars are omitted on the
per-L page.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass(frozen=True)
class ResultRow:
    L: int
    exact_filtered: float
    layer_energy: float
    layer_abs_delta: float
    layer_npar: int
    layer_nfev: int
    layer_msg: str
    term_energy: float | None
    term_abs_delta: float | None
    term_npar: int | None
    term_nfev: int | None
    term_msg: str | None
    termwise_missing: bool


def _parse_l_values(value: str) -> list[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("l-values cannot be empty.")
    out = [int(p) for p in parts]
    if any(v <= 0 for v in out):
        raise ValueError("All L values must be positive integers.")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _must_float(x: Any, *, field: str, path: Path) -> float:
    if x is None:
        raise ValueError(f"Missing required field '{field}' in {path}.")
    return float(x)


def _must_int(x: Any, *, field: str, path: Path) -> int:
    if x is None:
        raise ValueError(f"Missing required field '{field}' in {path}.")
    return int(x)


def _extract_layerwise(path: Path) -> dict[str, Any]:
    p = _load_json(path)
    vqe = p.get("vqe", {})
    exact = vqe.get("exact_filtered_energy")
    if exact is None:
        exact = p.get("ground_state", {}).get("exact_energy_filtered")
    if exact is None:
        raise ValueError(f"Layerwise exact filtered energy not found in {path}.")
    energy = _must_float(vqe.get("energy"), field="vqe.energy", path=path)
    npar = _must_int(vqe.get("num_parameters"), field="vqe.num_parameters", path=path)
    nfev = _must_int(vqe.get("nfev"), field="vqe.nfev", path=path)
    msg = str(vqe.get("message", ""))
    return {
        "exact": float(exact),
        "energy": float(energy),
        "abs_delta": abs(float(energy) - float(exact)),
        "npar": int(npar),
        "nfev": int(nfev),
        "msg": msg,
    }


def _extract_termwise(path: Path, *, exact_fallback: float) -> dict[str, Any]:
    p = _load_json(path)
    vqe = p.get("vqe", {})
    energy = _must_float(vqe.get("energy"), field="vqe.energy", path=path)
    exact = vqe.get("exact_filtered_energy")
    if exact is None:
        exact = p.get("ground_state", {}).get("exact_energy_filtered")
    if exact is None:
        exact = exact_fallback
    npar = _must_int(vqe.get("num_parameters"), field="vqe.num_parameters", path=path)
    nfev = _must_int(vqe.get("nfev"), field="vqe.nfev", path=path)
    msg = str(vqe.get("message", ""))
    return {
        "energy": float(energy),
        "abs_delta": abs(float(energy) - float(exact)),
        "npar": int(npar),
        "nfev": int(nfev),
        "msg": msg,
    }


def _build_rows(layer_dir: Path, term_dir: Path, l_values: list[int]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for L in l_values:
        layer_path = layer_dir / f"hardcoded_hva_pipeline_L{L}_heavy.json"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing layerwise JSON for L={L}: {layer_path}")
        layer = _extract_layerwise(layer_path)

        term_path = term_dir / f"hardcoded_hva_termwise_pipeline_L{L}_heavy.json"
        if term_path.exists():
            term = _extract_termwise(term_path, exact_fallback=layer["exact"])
            row = ResultRow(
                L=L,
                exact_filtered=layer["exact"],
                layer_energy=layer["energy"],
                layer_abs_delta=layer["abs_delta"],
                layer_npar=layer["npar"],
                layer_nfev=layer["nfev"],
                layer_msg=layer["msg"],
                term_energy=term["energy"],
                term_abs_delta=term["abs_delta"],
                term_npar=term["npar"],
                term_nfev=term["nfev"],
                term_msg=term["msg"],
                termwise_missing=False,
            )
        else:
            row = ResultRow(
                L=L,
                exact_filtered=layer["exact"],
                layer_energy=layer["energy"],
                layer_abs_delta=layer["abs_delta"],
                layer_npar=layer["npar"],
                layer_nfev=layer["nfev"],
                layer_msg=layer["msg"],
                term_energy=None,
                term_abs_delta=None,
                term_npar=None,
                term_nfev=None,
                term_msg=None,
                termwise_missing=True,
            )
        rows.append(row)
    return rows


def _fmt_energy(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.12f}"


def _fmt_delta(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.3e}"


def _fmt_int(x: int | None) -> str:
    if x is None:
        return ""
    return str(int(x))


def _render_intro_page(pdf: PdfPages, args: argparse.Namespace, rows: list[ResultRow]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    missing = [r.L for r in rows if r.termwise_missing]
    lines = [
        "HVA Heavy GS Report: Layerwise vs Termwise",
        "",
        f"L values: {','.join(str(v) for v in args.l_values)}",
        f"Layerwise dir: {args.layerwise_dir}",
        f"Termwise dir: {args.termwise_dir}",
        f"Output PDF: {args.output_pdf}",
        "",
        "Table policy:",
        "  - Layerwise rows required for all L values.",
        "  - Termwise rows are optional; missing values are left blank.",
    ]
    if missing:
        lines += [
            "",
            f"Missing termwise L values left blank: {','.join(str(v) for v in missing)}",
        ]
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_table_page(pdf: PdfPages, rows: list[ResultRow]) -> None:
    fig = plt.figure(figsize=(14.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    col_labels = [
        "L",
        "E_exact(filtered)",
        "E_layerwise",
        "|Delta_layerwise|",
        "npar_lw",
        "nfev_lw",
        "E_termwise",
        "|Delta_termwise|",
        "npar_tw",
        "nfev_tw",
    ]
    cell_text: list[list[str]] = []
    for r in rows:
        cell_text.append(
            [
                str(r.L),
                _fmt_energy(r.exact_filtered),
                _fmt_energy(r.layer_energy),
                _fmt_delta(r.layer_abs_delta),
                _fmt_int(r.layer_npar),
                _fmt_int(r.layer_nfev),
                _fmt_energy(r.term_energy),
                _fmt_delta(r.term_abs_delta),
                _fmt_int(r.term_npar),
                _fmt_int(r.term_nfev),
            ]
        )
    tab = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.07, 1.95)
    ax.set_title("Summary Table: Heavy HVA Layerwise vs Termwise", fontsize=14, pad=14)
    ax.text(
        0.01,
        0.02,
        "Note: termwise blanks indicate intentionally missing termwise runs (e.g., L=5).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_per_l_pages(pdf: PdfPages, rows: list[ResultRow]) -> None:
    for r in rows:
        fig, (ax_e, ax_d) = plt.subplots(1, 2, figsize=(13.5, 8.5))

        labels = ["Exact", "Layerwise"]
        vals = [r.exact_filtered, r.layer_energy]
        colors = ["#111111", "#2ca02c"]
        if r.term_energy is not None:
            labels.append("Termwise")
            vals.append(r.term_energy)
            colors.append("#1f77b4")
        ax_e.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax_e.set_title(f"L={r.L} Energies")
        ax_e.set_ylabel("Energy")
        ax_e.grid(axis="y", alpha=0.25)

        d_labels = ["|Delta_layerwise|"]
        d_vals = [r.layer_abs_delta]
        d_colors = ["#2ca02c"]
        if r.term_abs_delta is not None:
            d_labels.append("|Delta_termwise|")
            d_vals.append(r.term_abs_delta)
            d_colors.append("#1f77b4")
        ax_d.bar(d_labels, d_vals, color=d_colors, edgecolor="black", linewidth=0.5)
        ax_d.set_yscale("log")
        ax_d.set_title(f"L={r.L} Absolute Error vs Exact (log)")
        ax_d.grid(axis="y", alpha=0.25)

        term_line = (
            f"Termwise: npar={r.term_npar} nfev={r.term_nfev} msg={r.term_msg}"
            if r.term_energy is not None
            else "Termwise: intentionally blank (not run)."
        )
        note = "\n".join(
            [
                f"Layerwise: npar={r.layer_npar} nfev={r.layer_nfev} msg={r.layer_msg}",
                term_line,
            ]
        )
        fig.text(0.02, 0.02, note, ha="left", va="bottom", family="monospace", fontsize=8.5)
        fig.suptitle(f"L={r.L} Heavy HVA Comparison", fontsize=13)
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PDF report for heavy HVA layerwise vs termwise runs."
    )
    parser.add_argument(
        "--l-values",
        type=str,
        default="2,3,4,5",
        help="Comma-separated L values.",
    )
    parser.add_argument(
        "--layerwise-dir",
        type=Path,
        default=Path("artifacts/hva_vqe_heavy"),
        help="Directory containing hardcoded_hva_pipeline_L{L}_heavy.json files.",
    )
    parser.add_argument(
        "--termwise-dir",
        type=Path,
        default=Path("artifacts/hva_vqe_heavy_termwise"),
        help="Directory containing hardcoded_hva_termwise_pipeline_L{L}_heavy.json files.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("artifacts/hva_vqe_heavy_termwise/hva_layerwise_vs_termwise_L2_L3_L4_heavy_report.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()
    args.l_values = _parse_l_values(args.l_values)

    layer_dir = Path(args.layerwise_dir)
    term_dir = Path(args.termwise_dir)
    output_pdf = Path(args.output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(layer_dir=layer_dir, term_dir=term_dir, l_values=args.l_values)

    with PdfPages(str(output_pdf)) as pdf:
        _render_intro_page(pdf, args, rows)
        _render_table_page(pdf, rows)
        _render_per_l_pages(pdf, rows)

    print(f"Wrote PDF: {output_pdf}")


if __name__ == "__main__":
    main()
