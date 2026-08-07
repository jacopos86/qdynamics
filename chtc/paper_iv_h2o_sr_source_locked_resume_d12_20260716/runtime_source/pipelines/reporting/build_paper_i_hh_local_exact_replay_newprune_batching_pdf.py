#!/usr/bin/env python3
"""Build the local Paper-I HH exact-replay/new-prune batching diagnostic PDF."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_child_fairness_pdf import (  # noqa: E402
    _compile_local_snake_terminal_cost,
)


RUN_ID = "paper_i_hh_local_exact_replay_newprune_batching_20260705_v1"
RAW_ROOT = REPO_ROOT / "raw_outputs" / RUN_ID
OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / RUN_ID
STEM = RUN_ID
SUPPORT_JSON = (
    REPO_ROOT
    / "output"
    / "pdf"
    / "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630"
    / "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.json"
)
HH_TABLEIII_SOURCES_JSON = (
    REPO_ROOT
    / "MATH"
    / "paper_facing"
    / "paper_I_static_scaffold"
    / "hh_tableiii_convergence_sources.json"
)
WEAK_WEAK_SNAKE_PROMOTION_JSON = (
    REPO_ROOT
    / "output"
    / "pdf"
    / "paper_i_table_iii_snake_weak_weak_live_prefix_promotion_20260530.json"
)

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _fmt_float(value: float | None, *, sig: int = 3) -> str:
    if value is None:
        return "--"
    if value == 0:
        return "0"
    return f"{value:.{sig}e}"


def _fmt_num(value: float | int | None, *, sig: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.{sig}e}"


def _fmt_int(value: int | None) -> str:
    return "--" if value is None else str(value)


def _tex_escape(value: Any) -> str:
    out = str(value)
    return (
        out.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _history_points(adapt_vqe: Mapping[str, Any]) -> list[tuple[int, float]]:
    history = adapt_vqe.get("history")
    points: list[tuple[int, float]] = []
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for index, item in enumerate(history):
            if not isinstance(item, Mapping):
                continue
            x = _int_or_none(
                item.get("iteration")
                if item.get("iteration") is not None
                else item.get("k")
                if item.get("k") is not None
                else item.get("k_iter")
            )
            if x is None:
                x = index + 1
            y = (
                _float_or_none(item.get("delta_abs_current"))
                or _float_or_none(item.get("abs_delta_e"))
                or _float_or_none(item.get("benchmark_target_abs_delta_e_current"))
            )
            if y is not None and y > 0:
                points.append((x, y))
    final_error = _float_or_none(adapt_vqe.get("abs_delta_e"))
    if final_error is not None and final_error > 0 and points:
        final_x = points[-1][0]
        points[-1] = (final_x, final_error)
    return points


def _result_path(root: Path) -> Path | None:
    for candidate in (root / "json" / "result.json", root / "result.json"):
        if candidate.exists():
            return candidate
    return None


def _max_batch_size(adapt_vqe: Mapping[str, Any]) -> int | None:
    history = adapt_vqe.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return None
    sizes: list[int] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        value = item.get("batch_size")
        if value is None:
            value = item.get("phase3_batch_size")
        parsed = _int_or_none(value)
        if parsed is not None:
            sizes.append(parsed)
    return max(sizes) if sizes else 1


def _cost_int(cost: Mapping[str, Any], *keys: str) -> int | None:
    status = str(cost.get("status") or "").lower()
    source = str(cost.get("compiled_resource_source_kind") or cost.get("source_kind") or "").lower()
    if status not in {"done", "ok"} or "qiskit" not in source:
        return None
    for key in keys:
        parsed = _int_or_none(cost.get(key))
        if parsed is not None:
            return parsed
    return None


@dataclass
class LocalRow:
    row_id: str
    section: str
    regime: str
    label: str
    status: str
    root: str | None = None
    source_json: str | None = None
    k_iter: int | None = None
    d_ans: int | None = None
    abs_delta_e: float | None = None
    fidelity: float | None = None
    n2q: int | None = None
    d2q: int | None = None
    dc: int | None = None
    s_alg: float | None = None
    s_grad: float | None = None
    s_h: float | None = None
    s_metric: float | None = None
    max_batch_size: int | None = None
    stop_reason: str | None = None
    returncode: int | None = None
    note: str = ""
    history_points: list[tuple[int, float]] | None = None


def _load_local_row(
    *,
    row_id: str,
    section: str,
    regime: str,
    label: str,
    root_name: str,
    note: str = "",
) -> LocalRow:
    root = RAW_ROOT / root_name
    result_path = _result_path(root)
    if result_path is None:
        returncode_path = root / "returncode.txt"
        returncode = _int_or_none(returncode_path.read_text().strip()) if returncode_path.exists() else None
        status = "failed" if returncode not in (None, 0) else "pending"
        return LocalRow(
            row_id=row_id,
            section=section,
            regime=regime,
            label=label,
            status=status,
            root=_rel(root),
            returncode=returncode,
            note=note,
        )
    try:
        payload = _read_json(result_path)
        adapt_vqe = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
        if not isinstance(adapt_vqe, Mapping):
            raise ValueError("missing adapt_vqe")
    except Exception as exc:
        return LocalRow(
            row_id=row_id,
            section=section,
            regime=regime,
            label=label,
            status=f"failed:{type(exc).__name__}",
            root=_rel(root),
            source_json=_rel(result_path),
            note=note,
        )
    cost = _compile_local_snake_terminal_cost(root, result_path)
    history = adapt_vqe.get("history")
    k_iter = len(history) if isinstance(history, Sequence) and not isinstance(history, (str, bytes)) else None
    s_refit = _float_or_none(cost.get("S_alg_N_H_refit_eval"))
    s_outer = _float_or_none(cost.get("S_alg_N_H_outer_eval"))
    s_h = None
    if s_refit is not None or s_outer is not None:
        s_h = (s_refit or 0.0) + (s_outer or 0.0)
    return LocalRow(
        row_id=row_id,
        section=section,
        regime=regime,
        label=label,
        status="done",
        root=_rel(root),
        source_json=_rel(result_path),
        k_iter=k_iter,
        d_ans=_int_or_none(adapt_vqe.get("ansatz_depth")),
        abs_delta_e=_float_or_none(adapt_vqe.get("abs_delta_e")),
        fidelity=_float_or_none(adapt_vqe.get("exact_state_fidelity")),
        n2q=_cost_int(cost, "compiled_count_2q_total", "N2q"),
        d2q=_cost_int(cost, "compiled_depth_2q_total", "D2q"),
        dc=_cost_int(cost, "compiled_depth_total", "Dc"),
        s_alg=_float_or_none(cost.get("S_alg")),
        s_grad=_float_or_none(cost.get("S_alg_N_grad_probe")),
        s_h=s_h,
        s_metric=_float_or_none(cost.get("S_alg_N_metric_probe")),
        max_batch_size=_max_batch_size(adapt_vqe),
        stop_reason=str(adapt_vqe.get("stop_reason") or ""),
        note=note,
        history_points=_history_points(adapt_vqe),
    )


def _source_audit_rows() -> list[dict[str, Any]]:
    if not SUPPORT_JSON.exists():
        return []
    payload = _read_json(SUPPORT_JSON)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        match = next(
            (
                row
                for row in rows
                if isinstance(row, Mapping)
                and row.get("role_key") == "snake_native_a1"
                and row.get("regime") == regime
            ),
            None,
        )
        if not isinstance(match, Mapping):
            continue
        source_path = REPO_ROOT / str(match.get("source_json") or "")
        terminal_abs = None
        terminal_depth = None
        terminal_hist = None
        terminal_max_batch = None
        if source_path.exists():
            try:
                source = _read_json(source_path)
                adapt_vqe = source.get("adapt_vqe") if isinstance(source, Mapping) else None
                if isinstance(adapt_vqe, Mapping):
                    terminal_abs = _float_or_none(adapt_vqe.get("abs_delta_e"))
                    terminal_depth = _int_or_none(adapt_vqe.get("ansatz_depth"))
                    history = adapt_vqe.get("history")
                    terminal_hist = (
                        len(history) if isinstance(history, Sequence) and not isinstance(history, (str, bytes)) else None
                    )
                    terminal_max_batch = _max_batch_size(adapt_vqe)
            except Exception:
                pass
        out.append(
            {
                "regime": regime,
                "selected_prefix_k": match.get("selected_prefix_k"),
                "display_abs_delta_e": match.get("abs_delta_e"),
                "terminal_abs_delta_e": terminal_abs,
                "terminal_depth": terminal_depth,
                "terminal_history_len": terminal_hist,
                "s_alg_display": match.get("s_alg"),
                "terminal_max_batch": terminal_max_batch,
                "source_json": match.get("source_json"),
            }
        )
    return out


def _visible_source_comparison_rows(rows: Sequence[LocalRow]) -> list[dict[str, Any]]:
    """Compare the Paper-I visible source row against local source-lock replays."""

    if not HH_TABLEIII_SOURCES_JSON.exists():
        return []
    try:
        source_map = _read_json(HH_TABLEIII_SOURCES_JSON)
    except Exception:
        return []
    regimes = source_map.get("regimes") if isinstance(source_map, Mapping) else None
    weak_weak = regimes.get("weak_weak") if isinstance(regimes, Mapping) else None
    methods = weak_weak.get("methods") if isinstance(weak_weak, Mapping) else None
    snake = methods.get("SNAKE") if isinstance(methods, Mapping) else None
    if not isinstance(snake, Mapping):
        return []

    promotion: Mapping[str, Any] = {}
    if WEAK_WEAK_SNAKE_PROMOTION_JSON.exists():
        try:
            loaded_promotion = _read_json(WEAK_WEAK_SNAKE_PROMOTION_JSON)
            if isinstance(loaded_promotion, Mapping):
                promotion = loaded_promotion
        except Exception:
            promotion = {}

    local_anchor = next(
        (
            row
            for row in rows
            if row.row_id == "weak-anchor"
            and row.section == "anchor"
            and row.regime == "weak-weak"
        ),
        None,
    )
    if local_anchor is None:
        return []

    paper_cost = snake.get("compiled_resource_cells")
    if not isinstance(paper_cost, Mapping):
        paper_cost = {}
    visible_cells = promotion.get("visible_cells")
    if not isinstance(visible_cells, Mapping):
        visible_cells = {}
    if not paper_cost:
        paper_cost = visible_cells
    return [
        {
            "regime": "weak-weak",
            "row": "Paper-I visible source",
            "role": "table source",
            "abs_delta_e": _float_or_none(snake.get("same_cutoff_plateau_abs_delta_e")),
            "n2q": _int_or_none(paper_cost.get("N2q")),
            "d2q": _int_or_none(paper_cost.get("D2q")),
            "dc": _int_or_none(paper_cost.get("Dc") or paper_cost.get("D_c")),
            "note": "Visible Table-III source-map energy with promotion-JSON resource cells.",
        },
        {
            "regime": "weak-weak",
            "row": "Local anchor replay",
            "role": "single_admission/reduced",
            "abs_delta_e": local_anchor.abs_delta_e,
            "n2q": local_anchor.n2q,
            "d2q": local_anchor.d2q,
            "dc": local_anchor.dc,
            "note": (
                "Lower local same-cutoff error than the visible source; "
                "keep both rows because the compiled-cost sidecars come from different provenance surfaces."
            ),
        },
    ]


def _plot_rows(rows: Sequence[LocalRow], regime: str, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    plotted = False
    styles = {
        "anchor": {"linestyle": "--", "marker": "o"},
        "variant": {"linestyle": "-", "marker": "s"},
    }
    for row in rows:
        if row.regime != regime or not row.history_points:
            continue
        style = styles.get(row.section, {"linestyle": ":", "marker": "^"})
        xs = [x for x, _ in row.history_points]
        ys = [y for _, y in row.history_points]
        ax.plot(
            xs,
            ys,
            linewidth=1.2,
            markersize=2.6,
            label=row.label,
            linestyle=style["linestyle"],
            marker=style["marker"],
        )
        plotted = True
    ax.set_title(regime)
    ax.set_xlabel("ADAPT controller iteration k")
    ax.set_ylabel("abs(Delta E)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_xlim(left=1, right=30)
    if plotted:
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "no completed local trajectory", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    path = figures_dir / f"{regime.replace('-', '_')}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _variant_rows() -> list[LocalRow]:
    return [
        _load_local_row(
            row_id="weak-anchor",
            section="anchor",
            regime="weak-weak",
            label="source-lock anchor k=10",
            root_name="weak_weak_anchor_source_lock_kpl10_replay_v1",
            note="Exact replay of displayed Paper-I source prefix; off-by-one support k maps to source history index 10.",
        ),
        _load_local_row(
            row_id="strong-anchor",
            section="anchor",
            regime="strong-strong",
            label="source-lock anchor k=10",
            root_name="strong_strong_anchor_source_lock_kpl10_replay_v1",
            note="Exact replay of displayed Paper-I source prefix; source-lock child-set override used only when the current hard guard hid the preferred label.",
        ),
        _load_local_row(
            row_id="weak-greedy",
            section="variant",
            regime="weak-weak",
            label="metric prune + greedy batch",
            root_name="weak_weak_variant_metric_prune_greedy_batch_v2",
            note="New metric_regularized_v1 prune route with greedy_reduced_plane batching.",
        ),
        _load_local_row(
            row_id="weak-combinatorial",
            section="variant",
            regime="weak-weak",
            label="metric prune + combinatorial batch",
            root_name="weak_weak_variant_metric_prune_combinatorial_batch_v1",
            note="New metric_regularized_v1 prune route with combinatorial_reduced_plane batching.",
        ),
        _load_local_row(
            row_id="weak-combinatorial-maxb1",
            section="variant",
            regime="weak-weak",
            label="metric prune + combinatorial batch maxB=1",
            root_name="weak_weak_variant_metric_prune_combinatorial_batch_maxb1_v1",
            note="Same saved combinatorial metric-prune command as the maxB=5 row, except phase2 batch target/cap forced to 1.",
        ),
        _load_local_row(
            row_id="strong-greedy",
            section="variant",
            regime="strong-strong",
            label="metric prune + greedy batch",
            root_name="strong_strong_variant_metric_prune_greedy_batch_v1",
            note="Completed local strong-strong greedy diagnostic; expensive terminal branch/final refit observed.",
        ),
        _load_local_row(
            row_id="strong-combinatorial",
            section="variant",
            regime="strong-strong",
            label="metric prune + combinatorial batch",
            root_name="strong_strong_variant_metric_prune_combinatorial_batch_v1",
            note="Active/pending until local result.json is written.",
        ),
        _load_local_row(
            row_id="weak-greedy-v1-failed",
            section="provenance",
            regime="weak-weak",
            label="failed greedy v1 route-guard attempt",
            root_name="weak_weak_variant_metric_prune_greedy_batch_v1",
            note="Provenance only: route identity guard failed before the localized rerun.",
        ),
    ]


def _write_sidecars(
    rows: Sequence[LocalRow],
    source_rows: Sequence[Mapping[str, Any]],
    visible_comparison_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    payload = {
        "schema": "paper_i_hh_local_exact_replay_newprune_batching_report_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "run_class": "local diagnostic evidence only",
        "raw_root": _rel(RAW_ROOT),
        "support_json": _rel(SUPPORT_JSON),
        "source_audit_rows": list(source_rows),
        "visible_source_comparison_rows": list(visible_comparison_rows),
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "row_id",
        "section",
        "regime",
        "label",
        "status",
        "k_iter",
        "d_ans",
        "abs_delta_e",
        "fidelity",
        "n2q",
        "d2q",
        "dc",
        "s_alg",
        "s_grad",
        "s_h",
        "s_metric",
        "max_batch_size",
        "stop_reason",
        "returncode",
        "source_json",
        "root",
        "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data.pop("history_points", None)
            writer.writerow({field: data.get(field) for field in fields})
    return json_path, csv_path


def _visible_comparison_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.get("regime")),
                    _tex_escape(row.get("row")),
                    _tex_escape(row.get("role")),
                    _fmt_float(_float_or_none(row.get("abs_delta_e"))),
                    _fmt_int(_int_or_none(row.get("n2q"))),
                    _fmt_int(_int_or_none(row.get("d2q"))),
                    _fmt_int(_int_or_none(row.get("dc"))),
                    _tex_escape(row.get("note") or ""),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def _source_table(source_rows: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for row in source_rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.get("regime")),
                    _tex_escape(row.get("selected_prefix_k")),
                    _fmt_float(_float_or_none(row.get("display_abs_delta_e"))),
                    _fmt_float(_float_or_none(row.get("terminal_abs_delta_e"))),
                    _fmt_int(_int_or_none(row.get("terminal_depth"))),
                    _fmt_int(_int_or_none(row.get("terminal_history_len"))),
                    _fmt_num(_float_or_none(row.get("s_alg_display"))),
                    _fmt_int(_int_or_none(row.get("terminal_max_batch"))),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def _local_table(rows: Sequence[LocalRow], regime: str) -> str:
    lines: list[str] = []
    for row in rows:
        if row.regime != regime or row.section == "provenance":
            continue
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.label),
                    _tex_escape(row.status),
                    _fmt_int(row.k_iter),
                    _fmt_int(row.d_ans),
                    _fmt_float(row.abs_delta_e),
                    _fmt_float(row.fidelity, sig=4),
                    _fmt_int(row.n2q),
                    _fmt_int(row.d2q),
                    _fmt_int(row.dc),
                    _fmt_num(row.s_alg),
                    _fmt_int(row.max_batch_size),
                    _tex_escape(row.stop_reason or ""),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def _provenance_table(rows: Sequence[LocalRow]) -> str:
    lines: list[str] = []
    for row in rows:
        if row.section != "provenance":
            continue
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.regime),
                    _tex_escape(row.label),
                    _tex_escape(row.status),
                    _tex_escape(row.returncode),
                    _tex_escape(row.note),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def _write_tex(
    *,
    rows: Sequence[LocalRow],
    source_rows: Sequence[Mapping[str, Any]],
    visible_comparison_rows: Sequence[Mapping[str, Any]],
    figures: Mapping[str, Path],
    output_dir: Path,
    stem: str,
    json_path: Path,
    csv_path: Path,
) -> Path:
    done = sum(1 for row in rows if row.status == "done")
    pending = sum(1 for row in rows if row.status == "pending")
    failed = sum(1 for row in rows if row.status.startswith("failed"))
    tex_path = output_dir / f"{stem}.tex"
    generated = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.45in,landscape]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{xcolor}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\begin{document}",
        f"% JSON: {_rel(json_path)}",
        f"% CSV: {_rel(csv_path)}",
        r"\section*{Paper-I HH Local Exact-Replay / New-Prune Batching Diagnostic}",
        r"\small",
        r"\begin{tabular}{p{0.18\textwidth}p{0.78\textwidth}}",
        r"\toprule",
        f"Run ID & \\path{{{RUN_ID}}} \\\\",
        r"Run class & local diagnostic evidence only; no manuscript promotion decision is made here \\",
        r"Methods & SNAKE only; POWELL inner optimizer; Hubbard--Holstein $L=2$ weak--weak and strong--strong diagnostics \\",
        r"Source contract & full\_meta/HVA included; Phase-III archival singleton split; hard\_guard source row; depth cap 30; maxiter 200; final refit 200 \\",
        r"New controls tested & metric\_regularized\_v1 prune nomination with greedy/combinatorial reduced-plane batching, target/cap 5 \\",
        r"Source-lock repair & Exact source-prefix replays compare admitted child-set labels; source-lock-only child-set override is recorded when needed \\",
        f"Rows & {done} done; {pending} pending; {failed} failed/provenance \\\\",
        f"Generated UTC & {_tex_escape(generated)} \\\\",
        f"Raw root & \\path{{{_rel(RAW_ROOT)}}} \\\\",
        f"Support source & \\path{{{_rel(SUPPORT_JSON)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vfill",
        r"Trajectory plots use the first recorded post-selection/post-refit ADAPT history point for every local row. "
        r"Cost columns are terminal Qiskit-compiled circuit costs. "
        r"$S_{\rm alg}$ is algorithmic estimator/probe work reconstructed from the SNAKE sidecar; it is not a physical shot count.",
        r"\clearpage",
        r"\section*{Visible Source versus Local Anchor Replay}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrp{0.42\textwidth}}",
        r"\toprule",
        r"Regime & row & role & absDeltaE & $N_{2q}$ & $D_{2q}$ & $D_c$ & note \\",
        r"\midrule",
        _visible_comparison_table(visible_comparison_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\par\vspace{0.5em}",
        r"\footnotesize This table records the weak--weak comparison highlighted in chat: the local source-lock replay is lower-error than the current Paper-I visible source row, while both provenance surfaces are retained explicitly.",
        r"\clearpage",
        r"\section*{Source-Row Audit}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Regime & support $k$ & support absDeltaE & terminal absDeltaE & terminal $d_{\rm ans}$ & hist & $S_{\rm alg}$ & maxB \\",
        r"\midrule",
        _source_table(source_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{0.5em}",
        r"\footnotesize Source rows are the visible POWELL SNAKE A1 rows from the current pool-exposure support JSON. "
        r"The support $k$ values are displayed-prefix rows, while terminal columns are read from the linked source result JSONs.",
    ]
    for regime in ("weak-weak", "strong-strong"):
        lines.extend(
            [
                r"\clearpage",
                f"\\section*{{{_tex_escape(regime)} Local Replays and Variants}}",
                f"\\includegraphics[width=0.62\\textwidth]{{figures/{figures[regime].name}}}",
                r"\par\vspace{0.3em}",
                r"\scriptsize",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{llrrrrrrrrrl}",
                r"\toprule",
                r"Row & status & $k_{\rm iter}$ & $d_{\rm ans}$ & absDeltaE & F & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ & maxB & stop \\",
                r"\midrule",
                _local_table(rows, regime),
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
            ]
        )
    lines.extend(
        [
            r"\clearpage",
            r"\section*{Settings Diffs and Provenance Notes}",
            r"\small",
            r"\begin{itemize}",
            r"\item Baseline source rows used legacy hessian-coupling prune behavior and did not fire batching: maxB=1 in the source audit.",
            r"\item New variants change only the tested prune/batch controls plus local route-label plumbing: metric\_regularized\_v1, greedy/combinatorial reduced-plane batch mode, and batch target/cap 5.",
            r"\item Source-lock anchor rows are sanity checks for the source replay path, not new algorithm variants.",
            r"\item Strong--strong variants completed, but both exposed expensive terminal tails; combinatorial in particular used a 739-parameter final POWELL refit with 20946 objective evaluations.",
            r"\item Failed attempts are retained below as provenance only and should not be read as scientific rows.",
            r"\end{itemize}",
            r"\scriptsize",
            r"\begin{tabular}{llrlp{0.55\textwidth}}",
            r"\toprule",
            r"Regime & row & status & returncode & note \\",
            r"\midrule",
            _provenance_table(rows),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def build_report() -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = _variant_rows()
    source_rows = _source_audit_rows()
    visible_comparison_rows = _visible_source_comparison_rows(rows)
    figures = {regime: _plot_rows(rows, regime, figures_dir) for regime in ("weak-weak", "strong-strong")}
    json_path, csv_path = _write_sidecars(rows, source_rows, visible_comparison_rows, OUTPUT_DIR, STEM)
    tex_path = _write_tex(
        rows=rows,
        source_rows=source_rows,
        visible_comparison_rows=visible_comparison_rows,
        figures=figures,
        output_dir=OUTPUT_DIR,
        stem=STEM,
        json_path=json_path,
        csv_path=csv_path,
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=OUTPUT_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    summary = {
        "pdf": str(OUTPUT_DIR / f"{STEM}.pdf"),
        "tex": str(tex_path),
        "json": str(json_path),
        "csv": str(csv_path),
        "done": str(sum(1 for row in rows if row.status == "done")),
        "pending": str(sum(1 for row in rows if row.status == "pending")),
        "failed": str(sum(1 for row in rows if row.status.startswith("failed"))),
    }
    return summary


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
