#!/usr/bin/env python3
"""Render a physics-facing PDF report from L=3 HH trend-probe JSON.

Target JSON contract:
- test/artifacts/l3_uccsd_paop_hva_trend_full_*.json

Report layout (cross-check inspired):
1) Parameter manifest
2) Scoreboard table (A/B x medium/heavy)
3) Trend decision summary
4) A-vs-B selected-operator trace comparison
5) Energy-trace overlay
6) Gradient-trace overlay
7) Command/provenance page
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)


RUN_ORDER = ["A_medium", "A_heavy", "B_medium", "B_heavy"]
RUN_TITLES = {
    "A_medium": "A (UCCSD+PAOP) medium",
    "A_heavy": "A (UCCSD+PAOP) heavy",
    "B_medium": "B (UCCSD+PAOP+HVA) medium",
    "B_heavy": "B (UCCSD+PAOP+HVA) heavy",
}
DEFAULT_INPUT = REPO_ROOT / "test" / "artifacts" / "l3_uccsd_paop_hva_trend_full_20260302T000521.json"


def _fmt_float(v: Any, fmt: str = ".6e", fallback: str = "N/A") -> str:
    try:
        if v is None:
            return fallback
        return format(float(v), fmt)
    except Exception:
        return fallback


def _fmt_int(v: Any, fallback: str = "N/A") -> str:
    try:
        if v is None:
            return fallback
        return str(int(v))
    except Exception:
        return fallback


def _bool_str(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def _truncate(s: Any, max_len: int = 60) -> str:
    txt = str(s)
    if len(txt) <= int(max_len):
        return txt
    return txt[: int(max_len) - 3] + "..."


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required_top = ["config", "exact", "runs", "trend"]
    missing_top = [k for k in required_top if k not in data]
    if missing_top:
        raise KeyError(f"Missing required top-level keys: {missing_top}")
    runs = data["runs"]
    for rk in RUN_ORDER:
        if rk not in runs:
            raise KeyError(f"Missing required run key: {rk}")
    return data


def _make_scoreboard_rows(payload: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    runs = payload.get("runs", {})
    for key in RUN_ORDER:
        run = runs.get(key, {})
        ok = bool(run.get("ok", False))
        grad_trace = run.get("grad_max_trace") or []
        last_grad = grad_trace[-1] if grad_trace else None
        leak = (run.get("sector_diag") or {}).get("sector_leak_flag", None)

        if ok:
            row = [
                RUN_TITLES[key],
                "True",
                _fmt_float(run.get("E_best"), ".10f"),
                _fmt_float(run.get("delta_E_abs"), ".3e"),
                _fmt_int(run.get("adapt_depth_reached")),
                _truncate(run.get("adapt_stop_reason", "N/A"), 24),
                _fmt_int(run.get("nfev_total")),
                _fmt_float(last_grad, ".3e"),
                _bool_str(leak),
                _fmt_float(run.get("runtime_s"), ".2f"),
            ]
        else:
            err = _truncate(run.get("error", "run failed"), 26)
            row = [
                RUN_TITLES[key],
                "False",
                "ERR",
                "ERR",
                "ERR",
                err,
                "ERR",
                "ERR",
                "ERR",
                _fmt_float(run.get("runtime_s"), ".2f"),
            ]
        rows.append(row)
    return rows


def _trace_compare(heavy_a: list[dict[str, Any]], heavy_b: list[dict[str, Any]]) -> dict[str, Any]:
    n_a = int(len(heavy_a))
    n_b = int(len(heavy_b))
    n_cmp = min(n_a, n_b)
    first_div = None
    for i in range(n_cmp):
        a_lbl = heavy_a[i].get("selected_label")
        b_lbl = heavy_b[i].get("selected_label")
        a_src = heavy_a[i].get("source")
        b_src = heavy_b[i].get("source")
        if (a_lbl != b_lbl) or (a_src != b_src):
            first_div = int(i + 1)
            break

    shared_prefix = n_cmp if first_div is None else int(first_div - 1)
    full_match = bool(first_div is None and n_a == n_b)
    return {
        "len_A": n_a,
        "len_B": n_b,
        "first_divergence_depth": first_div,
        "shared_prefix_len": int(shared_prefix),
        "full_match": bool(full_match),
    }


def _render_scoreboard_page(pdf: Any, payload: dict[str, Any]) -> None:
    plt = get_plt()
    headers = [
        "Run",
        "ok",
        "E_best",
        "|dE|",
        "depth",
        "stop",
        "nfev",
        "last|max g|",
        "sector_leak",
        "runtime(s)",
    ]
    rows = _make_scoreboard_rows(payload)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    render_compact_table(ax, title="L=3 HH Trend Probe Scoreboard", col_labels=headers, rows=rows, fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_trend_page(pdf: Any, payload: dict[str, Any]) -> None:
    trend = payload.get("trend", {})
    a = trend.get("A_uccsd_plus_paop", {})
    b = trend.get("B_uccsd_plus_paop_plus_hva", {})
    lines = [
        "Trend Decision Summary",
        "",
        f"Overall likely_convergent_with_more_budget: {trend.get('likely_convergent_with_more_budget', 'N/A')}",
        "",
        "Arm A (UCCSD+PAOP):",
        f"  assessable:                {a.get('assessable', 'N/A')}",
        f"  likely_convergent:         {a.get('likely_convergent_with_more_budget', 'N/A')}",
        f"  medium_delta_E_abs:        {a.get('medium_delta_E_abs', 'N/A')}",
        f"  heavy_delta_E_abs:         {a.get('heavy_delta_E_abs', 'N/A')}",
        f"  abs_improvement:           {a.get('abs_improvement', 'N/A')}",
        f"  rel_improvement:           {a.get('rel_improvement', 'N/A')}",
        f"  medium_last_grad_max:      {a.get('medium_last_grad_max', 'N/A')}",
        f"  heavy_last_grad_max:       {a.get('heavy_last_grad_max', 'N/A')}",
        f"  grad_down:                 {a.get('grad_down', 'N/A')}",
        f"  material_improvement:      {a.get('material_improvement', 'N/A')}",
        f"  heavy_stop_reason:         {a.get('heavy_stop_reason', 'N/A')}",
        "",
        "Arm B (UCCSD+PAOP+HVA):",
        f"  assessable:                {b.get('assessable', 'N/A')}",
        f"  likely_convergent:         {b.get('likely_convergent_with_more_budget', 'N/A')}",
        f"  medium_delta_E_abs:        {b.get('medium_delta_E_abs', 'N/A')}",
        f"  heavy_delta_E_abs:         {b.get('heavy_delta_E_abs', 'N/A')}",
        f"  abs_improvement:           {b.get('abs_improvement', 'N/A')}",
        f"  rel_improvement:           {b.get('rel_improvement', 'N/A')}",
        f"  medium_last_grad_max:      {b.get('medium_last_grad_max', 'N/A')}",
        f"  heavy_last_grad_max:       {b.get('heavy_last_grad_max', 'N/A')}",
        f"  grad_down:                 {b.get('grad_down', 'N/A')}",
        f"  material_improvement:      {b.get('material_improvement', 'N/A')}",
        f"  heavy_stop_reason:         {b.get('heavy_stop_reason', 'N/A')}",
    ]
    render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=120)


def _render_hva_composition_page(pdf: Any, payload: dict[str, Any]) -> None:
    """Render explicit HVA/pool-composition metadata for arm B."""
    cfg = payload.get("config", {})
    pool_raw = payload.get("pool_components_raw", {})
    pool_b = payload.get("pool_B_uccsd_plus_paop_plus_hva", {})

    hva_raw = pool_raw.get("hva", "N/A")
    paop_raw = pool_raw.get("paop", "N/A")
    uccsd_raw = pool_raw.get("uccsd_ferm_only_lifted", "N/A")

    raw_sizes = pool_b.get("raw_sizes", {})
    dedup_counts = pool_b.get("dedup_source_presence_counts", {})

    lines = [
        "HVA Data (Arm B Composition)",
        "",
        "Note: This trend-probe JSON does not include a separate warm-start optimizer stage.",
        "Here, HVA appears as an explicit source family merged into arm B pool construction.",
        "",
        f"Config seed: {cfg.get('seed', 'N/A')}",
        f"allow_repeats: {cfg.get('allow_repeats', 'N/A')}",
        "",
        "Raw component sizes before dedup:",
        f"  uccsd_ferm_only_lifted: {uccsd_raw}",
        f"  paop:                   {paop_raw}",
        f"  hva:                    {hva_raw}",
        "",
        "Arm B pool accounting (from payload.pool_B_uccsd_plus_paop_plus_hva):",
        f"  raw_sizes: {raw_sizes}",
        f"  dedup_source_presence_counts: {dedup_counts}",
        f"  overlap_count: {pool_b.get('overlap_count', 'N/A')}",
        f"  dedup_total: {pool_b.get('dedup_total', 'N/A')}",
    ]
    render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=120)


def _render_trace_comparison_pages(pdf: Any, payload: dict[str, Any]) -> None:
    plt = get_plt()
    runs = payload.get("runs", {})
    a_run = runs.get("A_heavy", {})
    b_run = runs.get("B_heavy", {})

    a_trace = a_run.get("selected_trace") if bool(a_run.get("ok")) else []
    b_trace = b_run.get("selected_trace") if bool(b_run.get("ok")) else []
    if not isinstance(a_trace, list):
        a_trace = []
    if not isinstance(b_trace, list):
        b_trace = []

    comp = _trace_compare(a_trace, b_trace)
    src_a = Counter(str(x.get("source", "unknown")) for x in a_trace)
    src_b = Counter(str(x.get("source", "unknown")) for x in b_trace)

    summary_lines = [
        "A vs B selected-operator trace comparison (heavy stage)",
        "",
        f"A_heavy ok: {a_run.get('ok', False)}",
        f"B_heavy ok: {b_run.get('ok', False)}",
        f"len(A_heavy trace): {comp['len_A']}",
        f"len(B_heavy trace): {comp['len_B']}",
        f"first divergence depth: {comp['first_divergence_depth']}",
        f"shared prefix length: {comp['shared_prefix_len']}",
        f"full match across compared depth: {comp['full_match']}",
        "",
        "Source counts (A_heavy):",
        f"  {dict(src_a)}",
        "Source counts (B_heavy):",
        f"  {dict(src_b)}",
    ]
    render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.03, max_line_width=120)

    max_depth = max(len(a_trace), len(b_trace))
    if max_depth == 0:
        render_text_page(pdf, ["Trace table unavailable: one or both heavy runs have no selected_trace."])
        return

    rows: list[list[str]] = []
    for d in range(1, max_depth + 1):
        a = a_trace[d - 1] if d - 1 < len(a_trace) else {}
        b = b_trace[d - 1] if d - 1 < len(b_trace) else {}
        a_label = str(a.get("selected_label", ""))
        b_label = str(b.get("selected_label", ""))
        a_source = str(a.get("source", ""))
        b_source = str(b.get("source", ""))
        same = "==" if (a_label == b_label and a_source == b_source) else "!="
        rows.append([
            str(d),
            _truncate(a_source, 14),
            _truncate(a_label, 44),
            _truncate(b_source, 14),
            _truncate(b_label, 44),
            same,
        ])

    headers = ["depth", "A src", "A label", "B src", "B label", "cmp"]
    chunk_size = 18
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        fig, ax = plt.subplots(figsize=(15, 8))
        title = f"Heavy trace depth table (rows {i+1}-{i+len(chunk)})"
        render_compact_table(ax, title=title, col_labels=headers, rows=chunk, fontsize=6)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _render_overlay_page(
    pdf: Any,
    payload: dict[str, Any],
    *,
    key: str,
    ylabel: str,
    title: str,
    log_if_wide: bool = False,
) -> None:
    plt = get_plt()
    runs = payload.get("runs", {})
    color_map = {
        "A_medium": "#1f77b4",
        "A_heavy": "#0b3d91",
        "B_medium": "#ff7f0e",
        "B_heavy": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    plotted = False
    positive_vals: list[float] = []
    for rk in RUN_ORDER:
        run = runs.get(rk, {})
        if not bool(run.get("ok", False)):
            continue
        arr = run.get(key)
        if not isinstance(arr, list) or len(arr) == 0:
            continue
        y = [float(v) for v in arr]
        x = list(range(len(y)))
        ax.plot(x, y, lw=1.6, color=color_map.get(rk, None), label=RUN_TITLES[rk])
        plotted = True
        for v in y:
            if v > 0.0 and math.isfinite(v):
                positive_vals.append(v)

    if not plotted:
        plt.close(fig)
        render_text_page(pdf, [f"No data available for overlay key '{key}'"])
        return

    if log_if_wide and positive_vals:
        vmin = min(positive_vals)
        vmax = max(positive_vals)
        if vmin > 0.0 and vmax / vmin >= 100.0:
            ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("ADAPT step index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_report(payload: dict[str, Any], input_json: Path, output_pdf: Path) -> None:
    require_matplotlib()
    PdfPages = get_PdfPages()
    cfg = payload.get("config", {})
    exact = payload.get("exact", {})

    extra_manifest = {
        "L": cfg.get("L"),
        "sector": cfg.get("sector"),
        "omega0": cfg.get("omega0"),
        "g_ep": cfg.get("g_ep"),
        "n_ph_max": cfg.get("n_ph_max"),
        "boson_encoding": cfg.get("boson_encoding"),
        "ordering": cfg.get("ordering"),
        "boundary": cfg.get("boundary"),
        "paop_key": cfg.get("paop_key"),
        "paop_r": cfg.get("paop_r"),
        "paop_split_paulis": cfg.get("paop_split_paulis"),
        "paop_prune_eps": cfg.get("paop_prune_eps"),
        "paop_normalization": cfg.get("paop_normalization"),
        "medium_depth": cfg.get("medium_depth"),
        "medium_maxiter": cfg.get("medium_maxiter"),
        "heavy_depth": cfg.get("heavy_depth"),
        "heavy_maxiter": cfg.get("heavy_maxiter"),
        "seed": cfg.get("seed"),
        "exact_filtered_energy": exact.get("E_exact_sector"),
    }

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_pdf)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="ADAPT Trend Probe: A=UCCSD+PAOP, B=UCCSD+PAOP+HVA",
            drive_enabled=False,
            t=float(cfg.get("t", 0.0)),
            U=float(cfg.get("U", 0.0)),
            dv=float(cfg.get("dv", 0.0)),
            extra=extra_manifest,
            command=current_command_string(),
        )

        _render_scoreboard_page(pdf, payload)
        _render_trend_page(pdf, payload)
        _render_hva_composition_page(pdf, payload)
        _render_trace_comparison_pages(pdf, payload)
        _render_overlay_page(
            pdf,
            payload,
            key="energy_trace",
            ylabel="Energy",
            title="Energy trace overlay: A/B medium/heavy",
            log_if_wide=False,
        )
        _render_overlay_page(
            pdf,
            payload,
            key="grad_max_trace",
            ylabel="max |gradient|",
            title="Gradient trace overlay: A/B medium/heavy",
            log_if_wide=True,
        )

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        render_command_page(
            pdf,
            current_command_string(),
            script_name="test/render_l3_hh_uccsd_paop_hva_trend_pdf.py",
            extra_header_lines=[
                f"Source JSON: {input_json}",
                f"Generated UTC: {utc_now}",
            ],
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render PDF report for L=3 HH trend-probe JSON.")
    p.add_argument(
        "--input-json",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to trend-probe JSON payload.",
    )
    p.add_argument(
        "--output-pdf",
        type=str,
        default="",
        help="Output PDF path. Default: input-json path with .pdf suffix.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_json = Path(args.input_json).expanduser().resolve()
    output_pdf = (
        Path(args.output_pdf).expanduser().resolve()
        if str(args.output_pdf).strip()
        else input_json.with_suffix(".pdf")
    )

    payload = _load_payload(input_json)
    _render_report(payload, input_json=input_json, output_pdf=output_pdf)
    print(f"WROTE {output_pdf}")


if __name__ == "__main__":
    main()
